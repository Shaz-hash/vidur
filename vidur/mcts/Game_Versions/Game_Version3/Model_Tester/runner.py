from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ....game_types import AdversaryAction
from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.eval_utils import NoopReplayWriter
from ..DNN.export_torchscript_gv2 import export_torchscript_artifacts_gv2
from ..DNN.value_models import AlphaZeroModel
from ..DNN.selfPlay import SelfPlayRunner
from ..logger.evaluation_pipeline_logger import ArenaGameCycleFileLogger, arena_state_snapshot_for_log
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import (
    _attach_native_runtime,
    _build_env_and_simulator,
    _load_weights_into_model,
    _set_global_seeds,
)
from .config import ModelTesterConfig
from .trivial_controller import select_trivial_controller_action


MODEL_VS_MODEL_LABEL = "model_vs_model"
MODEL_VS_TRIVIAL_LABEL = "model_adv_vs_trivial_ctrl"


@dataclass(frozen=True)
class _RunnerBundle:
    pipeline_cfg: Any
    simulator: Any
    env: Any
    explore_cfg: Any
    spec: Any
    model: AlphaZeroModel
    mcts: VidurMCTS
    runner: SelfPlayRunner
    native_runtime: Any | None


def _safe_close_simulator(simulator: Any) -> None:
    for method_name in ("shutdown", "close", "stop"):
        fn = getattr(simulator, method_name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


def _build_bundle(cfg: ModelTesterConfig) -> _RunnerBundle:
    pipeline_cfg = cfg.to_pipeline_cfg()
    pipeline_cfg.validate()

    _set_global_seeds(
        int(pipeline_cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(pipeline_cfg.game_v2.reproducibility.torch_deterministic),
    )

    simulator, env, _, explore_cfg = _build_env_and_simulator(
        pipeline_cfg, use_virtual_env=bool(pipeline_cfg.use_virtual_env)
    )

    spec = make_dnn_spec(cfg=pipeline_cfg.game_v2)
    model = AlphaZeroModel(spec=spec).to(torch.device(pipeline_cfg.model.device))
    _load_weights_into_model(model, Path(cfg.model_checkpoint_path))

    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        log_path=str(cfg.mcts_iter_log_path()),
        tree_log_path=str(cfg.mcts_root_log_path()),
        logger_flush_every=1,
        verbose=False,
    )

    native_runtime = None
    if bool(getattr(explore_cfg, "native_mcts_enabled", False)):
        from .... import mcts_native_gv2 as _mcts_native_gv2

        native_runtime = _mcts_native_gv2.NativeTorchScriptInferRuntimeGV2(
            str(pipeline_cfg.model.device), -50.0, 100.0
        )
        ts = export_torchscript_artifacts_gv2(
            checkpoint_path=Path(cfg.model_checkpoint_path),
            out_dir=Path(cfg.native_torchscript_dir),
            model_version=int(cfg.native_model_version),
            spec=spec,
            device="cpu",
        )
        model_spec = f"{ts.controller_path}||{ts.adversary_path}"
        native_runtime.load_models({int(cfg.native_model_version): str(model_spec)})
        _attach_native_runtime(model, native_runtime, int(cfg.native_model_version))

    runner = SelfPlayRunner(
        env=env,
        mcts=mcts,
        model=model,
        writer=NoopReplayWriter(),
        device_for_features=torch.device(pipeline_cfg.model.device),
        game_v2_cfg=pipeline_cfg.game_v2,
    )

    return _RunnerBundle(
        pipeline_cfg=pipeline_cfg,
        simulator=simulator,
        env=env,
        explore_cfg=explore_cfg,
        spec=spec,
        model=model,
        mcts=mcts,
        runner=runner,
        native_runtime=native_runtime,
    )


def _prepare_base_state_for_game(
    *,
    cfg: ModelTesterConfig,
    bundle: _RunnerBundle,
    game_id: int,
    history_nontrivial_hops: int,
    cycle_file_logger: ArenaGameCycleFileLogger,
) -> Tuple[Any, Any, str, int]:
    runner = bundle.runner
    state = runner.env.initial_state()
    player = str(cfg.start_player)
    depth = int(cfg.start_root_depth)
    next_log_node_id = int(getattr(runner.mcts, "_node_counter", 0))
    last_log_node_id = None

    history_events: list[dict] = []

    def _history_cb(ev: dict) -> None:
        ev2 = dict(ev)
        ev2["turn"] = len(history_events)
        history_events.append(ev2)

    if int(history_nontrivial_hops) > 0:
        state, player, depth, next_log_node_id, last_log_node_id = runner.history.generate_history_root(
            state,
            player,
            depth,
            nontrivial_hops=int(history_nontrivial_hops),
            game_id=int(game_id),
            root_id_for_logs=0,
            # Keep arena-history behavior but avoid root-log schema coupling in tester runs.
            log_history=False,
            seed=int(cfg.history_seed) + int(game_id),
            max_total_steps=int(bundle.pipeline_cfg.history_max_total_steps),
            log_node_id_start=next_log_node_id,
            log_parent_id_start=last_log_node_id,
            step_callback=_history_cb,
        )
        runner.mcts._node_counter = int(next_log_node_id)

    player = runner._align_player_to_valid_actions(state, player)
    if not runner._valid_action_indices_readonly(state, player):
        state = runner.env.initial_state()
        player = runner._align_player_to_valid_actions(state, str(cfg.start_player))
        depth = int(cfg.start_root_depth)

    if history_events:
        for cycle_label in (MODEL_VS_MODEL_LABEL, MODEL_VS_TRIVIAL_LABEL):
            for ev in history_events:
                cycle_file_logger.write_step(
                    game_id=int(game_id),
                    cycle_label=cycle_label,
                    phase=f"history_step:{ev.get('phase', '')}",
                    turn=int(ev.get("turn", 0)),
                    depth=int(ev.get("depth_before", depth)),
                    player_acted=str(ev.get("player_acted", "")),
                    player_to_act_next=str(ev.get("player_to_act_next", "")),
                    action_repr=str(ev.get("action_repr", "")),
                    sim_time_before=float(ev.get("sim_time_before", state.simulator._time)),
                    sim_time_after=float(ev.get("sim_time_after", state.simulator._time)),
                    total_cost=float(ev.get("total_cost", 0.0)),
                    slo_violations=int(ev.get("slo_violations", 0)),
                    total_lateness=float(ev.get("total_lateness", 0.0)),
                )

    base_snapshot = state.simulator.snapshot_state()
    base_stats = state.stats.clone()
    return base_snapshot, base_stats, str(player), int(depth)


def _run_model_vs_trivial_cycle(
    *,
    bundle: _RunnerBundle,
    cfg: ModelTesterConfig,
    base_snapshot: Any,
    base_stats: Any,
    base_player: str,
    base_depth: int,
    game_id: int,
    root_id_base: int,
    model: AlphaZeroModel,
    cycle_label: str,
    cycle_file_logger: ArenaGameCycleFileLogger | None,
) -> dict:
    runner = bundle.runner
    state = runner.env.clone_state_from_snapshot(base_snapshot, base_stats)
    player = str(base_player)
    depth = int(base_depth)

    pending_adv_pre_ctrl_snapshot = None
    pending_adv_pre_ctrl_stats = None

    turns = 0
    cleanup_steps = 0
    adv_moves_total = 0
    adv_request_moves = 0

    deadline_t = float(state.simulator._time) + float(cfg.arena_time_limit_sec)
    end_reason = ""

    while (
        turns < int(cfg.arena_max_total_turns)
        and float(state.simulator._time) < deadline_t
    ):
        player = runner._align_player_to_valid_actions(state, player)

        iters = int(cfg.cycle2_adv_iterations_per_root)
        selection_info: Dict[str, Any]
        search_basis_state = state
        replay_forbidden_stop_ids = set()

        if player == "adversary":
            search_basis_state, replay_forbidden_stop_ids = runner._build_root_decision_state_for_adversary(
                current_state=state,
                pre_controller_snapshot=pending_adv_pre_ctrl_snapshot,
                pre_controller_stats=pending_adv_pre_ctrl_stats,
            )

            action, _, selection_info = runner._pick_mcts_action_for_arena_step(
                state=state,
                player="adversary",
                model=model,
                game_id=int(game_id),
                root_id=int(root_id_base + turns),
                root_depth=int(depth),
                iterations=int(iters),
                feature_version=int(cfg.feature_version),
                cycle_label=str(cycle_label),
                prefer_nonempty_adversary=True,
                search_state=search_basis_state,
                forbidden_stop_ids=replay_forbidden_stop_ids,
            )
            if action is None:
                end_reason = "no_valid_action"
                break
        else:
            action, selection_info = select_trivial_controller_action(
                runner=runner,
                state=state,
                heuristic=str(cfg.trivial_policy.heuristic),
                budget_tokens=int(cfg.trivial_policy.budget_tokens),
                eviction_rule=str(cfg.trivial_policy.eviction_rule),
            )
            if action is None:
                end_reason = "no_valid_action"
                break

        sim_before = float(state.simulator._time)
        player_before = str(player)
        depth_before = int(depth)
        turn_before = int(turns)

        pre_controller_snapshot = None
        pre_controller_stats = None
        if player == "controller":
            pre_controller_snapshot = state.simulator.snapshot_state()
            pre_controller_stats = state.stats.clone()

        generated = len((action.requests or [])) if isinstance(action, AdversaryAction) else 0

        if player == "adversary":
            state = runner.env.apply_adversary_action_only(state, action, inplace=True)
            player = "controller"
            pending_adv_pre_ctrl_snapshot = None
            pending_adv_pre_ctrl_stats = None
            if generated > 0:
                adv_request_moves += 1
                adv_moves_total += 1
        else:
            state = runner.env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"

            src_fn = getattr(runner.env, "_v2_missed_adv_source", None)
            miss_src = int(src_fn(state)) if callable(src_fn) else 0
            if miss_src == 1 and pre_controller_snapshot is not None and pre_controller_stats is not None:
                pending_adv_pre_ctrl_snapshot = pre_controller_snapshot
                pending_adv_pre_ctrl_stats = pre_controller_stats
            else:
                pending_adv_pre_ctrl_snapshot = None
                pending_adv_pre_ctrl_stats = None

        depth += 1
        turns += 1

        viol_step, lateness_step = runner.env.evaluate_objective(state)
        step_cost = float(viol_step) + float(lateness_step)
        state_log = arena_state_snapshot_for_log(runner.env, state)

        if cycle_file_logger is not None:
            cycle_file_logger.write_step(
                game_id=int(game_id),
                cycle_label=str(cycle_label),
                phase="arena_step",
                turn=int(turn_before),
                depth=int(depth_before),
                player_acted=str(player_before),
                player_to_act_next=str(player),
                action_repr=repr(action),
                sim_time_before=float(sim_before),
                sim_time_after=float(state.simulator._time),
                total_cost=step_cost,
                slo_violations=int(viol_step),
                total_lateness=float(lateness_step),
                **state_log,
                selection_mode=str(selection_info.get("selection_mode", "")),
                valid_action_count=int(selection_info.get("valid_action_count", 0)),
                iterations_requested=int(selection_info.get("iterations_requested", 0)),
                iterations_used=int(selection_info.get("iterations_used", 0)),
            )

    if not end_reason and float(state.simulator._time) >= deadline_t:
        end_reason = "time_limit"

    while (
        turns < int(cfg.arena_max_total_turns)
        and cleanup_steps < int(cfg.arena_max_controller_cleanup_steps)
        and float(state.simulator._time) < deadline_t
        and runner._has_prefill_pending(state)
    ):
        sim_before = float(state.simulator._time)
        player_before = str(player)
        depth_before = int(depth)
        turn_before = int(turns)

        if player == "adversary":
            noop = AdversaryAction(requests=[], stop_decode_ids=[])
            state = runner.env.apply_adversary_action_only(state, noop, inplace=True)
            player = "controller"
            pending_adv_pre_ctrl_snapshot = None
            pending_adv_pre_ctrl_stats = None
            depth += 1
            turns += 1

            viol_step, lateness_step = runner.env.evaluate_objective(state)
            step_cost = float(viol_step) + float(lateness_step)
            state_log = arena_state_snapshot_for_log(runner.env, state)

            if cycle_file_logger is not None:
                cycle_file_logger.write_step(
                    game_id=int(game_id),
                    cycle_label=str(cycle_label),
                    phase="cleanup_step",
                    turn=int(turn_before),
                    depth=int(depth_before),
                    player_acted=str(player_before),
                    player_to_act_next=str(player),
                    action_repr=repr(noop),
                    sim_time_before=float(sim_before),
                    sim_time_after=float(state.simulator._time),
                    total_cost=step_cost,
                    slo_violations=int(viol_step),
                    total_lateness=float(lateness_step),
                    **state_log,
                    selection_mode="single_action_shortcut",
                    valid_action_count=1,
                    iterations_requested=0,
                    iterations_used=0,
                )
            continue

        action, selection_info = select_trivial_controller_action(
            runner=runner,
            state=state,
            heuristic=str(cfg.trivial_policy.heuristic),
            budget_tokens=int(cfg.trivial_policy.budget_tokens),
            eviction_rule=str(cfg.trivial_policy.eviction_rule),
        )
        if action is None:
            end_reason = end_reason or "cleanup_no_valid_action"
            break

        pre_controller_snapshot = state.simulator.snapshot_state()
        pre_controller_stats = state.stats.clone()
        state = runner.env.apply_controller_action_only(state, action, inplace=True)
        player = "adversary"

        src_fn = getattr(runner.env, "_v2_missed_adv_source", None)
        miss_src = int(src_fn(state)) if callable(src_fn) else 0
        if miss_src == 1 and pre_controller_snapshot is not None and pre_controller_stats is not None:
            pending_adv_pre_ctrl_snapshot = pre_controller_snapshot
            pending_adv_pre_ctrl_stats = pre_controller_stats
        else:
            pending_adv_pre_ctrl_snapshot = None
            pending_adv_pre_ctrl_stats = None

        depth += 1
        turns += 1
        cleanup_steps += 1

        viol_step, lateness_step = runner.env.evaluate_objective(state)
        step_cost = float(viol_step) + float(lateness_step)
        state_log = arena_state_snapshot_for_log(runner.env, state)
        if cycle_file_logger is not None:
            cycle_file_logger.write_step(
                game_id=int(game_id),
                cycle_label=str(cycle_label),
                phase="cleanup_step",
                turn=int(turn_before),
                depth=int(depth_before),
                player_acted=str(player_before),
                player_to_act_next=str(player),
                action_repr=repr(action),
                sim_time_before=float(sim_before),
                sim_time_after=float(state.simulator._time),
                total_cost=step_cost,
                slo_violations=int(viol_step),
                total_lateness=float(lateness_step),
                **state_log,
                selection_mode=str(selection_info.get("selection_mode", "")),
                valid_action_count=int(selection_info.get("valid_action_count", 0)),
                iterations_requested=int(selection_info.get("iterations_requested", 0)),
                iterations_used=int(selection_info.get("iterations_used", 0)),
            )

    if not end_reason:
        if float(state.simulator._time) >= deadline_t:
            end_reason = "time_limit"
        elif turns >= int(cfg.arena_max_total_turns):
            end_reason = "max_total_turns"
        elif cleanup_steps >= int(cfg.arena_max_controller_cleanup_steps):
            end_reason = "max_cleanup_steps"

    viol, lateness = runner.env.evaluate_objective(state)
    total_cost = float(viol) + float(lateness)

    if runner.mcts._root_logger is not None:
        runner.mcts._root_logger.log_root(
            game_id=int(game_id),
            root_id=int(root_id_base + turns + 1),
            root_depth=int(depth),
            root_node_id=-1,
            root_player="",
            num_simulations=0,
            model_root_value_controller=0.0,
            model_root_prior=[],
            normalized_root_prior=[],
            valid_action_mask=[],
            mcts_root_value_controller=0.0,
            mcts_root_prior=[],
            best_action_index=None,
            best_action_repr="",
            best_action_json="",
            phase="arena_end",
            cycle_label=str(cycle_label),
            sim_time=float(state.simulator._time),
            slo_violations=int(viol),
            total_lateness=float(lateness),
            total_cost=float(total_cost),
        )

    return {
        "total_cost": float(total_cost),
        "slo_violations": int(viol),
        "total_lateness": float(lateness),
        "turns": int(turns),
        "cleanup_steps": int(cleanup_steps),
        "adversary_moves_total": int(adv_moves_total),
        "adversary_request_moves": int(adv_request_moves),
        "end_reason": str(end_reason),
        "sim_time_end": float(state.simulator._time),
        "sim_time_deadline": float(deadline_t),
    }


def _write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "game_id",
        "history_hops",
        "model_checkpoint",
        "cycle1_label",
        "cycle1_slo_violations",
        "cycle1_total_lateness",
        "cycle1_total_cost",
        "cycle1_end_reason",
        "cycle2_label",
        "cycle2_slo_violations",
        "cycle2_total_lateness",
        "cycle2_total_cost",
        "cycle2_end_reason",
        "cost_delta_cycle2_minus_cycle1",
        "better_cycle",
        "cycle1_log_file",
        "cycle2_log_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def run_model_vs_trivial_tester(cfg: ModelTesterConfig) -> Path:
    cfg.validate()
    cfg.output_dir_path().mkdir(parents=True, exist_ok=True)
    cfg.arena_games_dir_path().mkdir(parents=True, exist_ok=True)
    Path(cfg.native_torchscript_dir).mkdir(parents=True, exist_ok=True)

    # Keep tester runs reproducible and avoid header-mismatch errors when logger schemas evolve.
    iter_log = cfg.mcts_iter_log_path()
    root_log = cfg.mcts_root_log_path()
    for p in (
        iter_log,
        root_log,
        iter_log.with_name(f"{iter_log.stem}_native{iter_log.suffix or '.csv'}"),
        root_log.with_name(f"{root_log.stem}_native{root_log.suffix or '.csv'}"),
        root_log.with_name(f"{root_log.stem}_native.config.json"),
    ):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    bundle = _build_bundle(cfg)
    cycle_logger = ArenaGameCycleFileLogger(cfg.arena_games_dir_path())

    history_hops = cfg.sample_history_hops()
    rows: List[Dict[str, Any]] = []

    try:
        for i in range(int(cfg.num_games)):
            game_id = int(cfg.game_id_start) + int(i)
            hops = int(history_hops[i])

            base_snapshot, base_stats, base_player, base_depth = _prepare_base_state_for_game(
                cfg=cfg,
                bundle=bundle,
                game_id=int(game_id),
                history_nontrivial_hops=int(hops),
                cycle_file_logger=cycle_logger,
            )

            cycle1 = bundle.runner._run_arena_cycle(
                base_snapshot=base_snapshot,
                base_stats=base_stats,
                base_player=str(base_player),
                base_depth=int(base_depth),
                game_id=int(game_id),
                root_id_base=0,
                adversary_model=bundle.model,
                controller_model=bundle.model,
                cycle_label=MODEL_VS_MODEL_LABEL,
                adv_iterations_per_root=int(cfg.cycle1_adv_iterations_per_root),
                cont_iterations_per_root=int(cfg.cycle1_cont_iterations_per_root),
                arena_time_limit_sec=float(cfg.arena_time_limit_sec),
                arena_max_controller_cleanup_steps=int(cfg.arena_max_controller_cleanup_steps),
                arena_max_total_turns=int(cfg.arena_max_total_turns),
                feature_version=int(cfg.feature_version),
                cycle_file_logger=cycle_logger,
            )
            cycle_logger.write_cycle_end(
                game_id=int(game_id),
                cycle_label=MODEL_VS_MODEL_LABEL,
                total_cost=float(cycle1["total_cost"]),
                slo_violations=int(cycle1["slo_violations"]),
                total_lateness=float(cycle1["total_lateness"]),
                end_reason=str(cycle1.get("end_reason", "")),
            )

            cycle2 = _run_model_vs_trivial_cycle(
                bundle=bundle,
                cfg=cfg,
                base_snapshot=base_snapshot,
                base_stats=base_stats,
                base_player=str(base_player),
                base_depth=int(base_depth),
                game_id=int(game_id),
                root_id_base=1_000_000,
                model=bundle.model,
                cycle_label=MODEL_VS_TRIVIAL_LABEL,
                cycle_file_logger=cycle_logger,
            )
            cycle_logger.write_cycle_end(
                game_id=int(game_id),
                cycle_label=MODEL_VS_TRIVIAL_LABEL,
                total_cost=float(cycle2["total_cost"]),
                slo_violations=int(cycle2["slo_violations"]),
                total_lateness=float(cycle2["total_lateness"]),
                end_reason=str(cycle2.get("end_reason", "")),
            )

            delta = float(cycle2["total_cost"]) - float(cycle1["total_cost"])
            if delta < -1e-9:
                better_cycle = MODEL_VS_TRIVIAL_LABEL
            elif delta > 1e-9:
                better_cycle = MODEL_VS_MODEL_LABEL
            else:
                better_cycle = "tie"

            rows.append(
                {
                    "game_id": int(game_id),
                    "history_hops": int(hops),
                    "model_checkpoint": str(cfg.model_checkpoint_path),
                    "cycle1_label": MODEL_VS_MODEL_LABEL,
                    "cycle1_slo_violations": int(cycle1["slo_violations"]),
                    "cycle1_total_lateness": float(cycle1["total_lateness"]),
                    "cycle1_total_cost": float(cycle1["total_cost"]),
                    "cycle1_end_reason": str(cycle1.get("end_reason", "")),
                    "cycle2_label": MODEL_VS_TRIVIAL_LABEL,
                    "cycle2_slo_violations": int(cycle2["slo_violations"]),
                    "cycle2_total_lateness": float(cycle2["total_lateness"]),
                    "cycle2_total_cost": float(cycle2["total_cost"]),
                    "cycle2_end_reason": str(cycle2.get("end_reason", "")),
                    "cost_delta_cycle2_minus_cycle1": float(delta),
                    "better_cycle": str(better_cycle),
                    "cycle1_log_file": str(
                        cfg.arena_games_dir_path() / f"game_{int(game_id)}_{MODEL_VS_MODEL_LABEL}.csv"
                    ),
                    "cycle2_log_file": str(
                        cfg.arena_games_dir_path() / f"game_{int(game_id)}_{MODEL_VS_TRIVIAL_LABEL}.csv"
                    ),
                }
            )

        out_csv = cfg.arena_results_csv_path()
        _write_results_csv(out_csv, rows)
        return out_csv
    finally:
        _safe_close_simulator(bundle.simulator)
