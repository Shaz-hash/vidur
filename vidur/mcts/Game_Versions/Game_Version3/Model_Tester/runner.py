from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from ....game_types import AdversaryAction, ControllerAction
from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.eval_utils import NoopReplayWriter
from ..DNN.export_torchscript_gv2 import export_torchscript_artifacts_gv2
from ..DNN.value_models import AlphaZeroModel
from ..DNN.selfPlay import SelfPlayRunner
from ..logger.evaluation_pipeline_logger import (
    ArenaGameCycleFileLogger,
    ArenaModelActionDetailLogger,
    arena_state_snapshot_for_log,
)
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import (
    _build_env_and_simulator,
    _load_weights_into_model,
    _set_global_seeds,
)
from .config import ModelTesterConfig
from .trivial_controller import select_trivial_controller_action
from .trivial_adversary import make_noop_adversary_action, select_trivial_adversary_action


MODEL_ADV_VS_TRIVIAL_CTRL_LABEL = "model_adv_depth1_vs_trivial_ctrl"
MODEL_ADV_VS_MODEL_CTRL_LABEL = "model_adv_depth1_vs_model_ctrl_depth1"


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


@dataclass(frozen=True)
class _ExpandedActionSpace:
    player: str
    search_state: Any
    forbidden_stop_ids: set[int]
    actions_by_index: list[Any | None]
    valid_mask: list[bool]
    valid_indices: list[int]


def _safe_close_simulator(simulator: Any) -> None:
    for method_name in ("shutdown", "close", "stop"):
        fn = getattr(simulator, method_name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


def _attach_native_runtime(model: Any, runtime: Any, model_version: int) -> None:
    setattr(model, "_native_ts_runtime", runtime)
    setattr(model, "_native_ts_model_version", int(model_version))


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


def _other_player(player: str) -> str:
    return "controller" if str(player) == "adversary" else "adversary"


def _mask_to_list(mask: Any) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in list(mask)]


def _has_prefill_pending(bundle: _RunnerBundle, state: Any) -> bool:
    env = bundle.env
    req_map = env._req_map(state.simulator)
    for rid in list(getattr(state.stats, "active_request_ids", set()) or set()):
        req = req_map.get(int(rid))
        if req is None or bool(getattr(req, "completed", False)):
            continue
        prefill_done = bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)))
        if prefill_done:
            continue
        if int(env._remaining_prefill(req)) > 0:
            return True
    return False


def _expand_action_space(
    *,
    bundle: _RunnerBundle,
    state: Any,
    player: str,
    pending_adv_pre_ctrl_snapshot: Any | None,
    pending_adv_pre_ctrl_stats: Any | None,
) -> _ExpandedActionSpace:
    runner = bundle.runner
    search_state = state
    forbidden_stop_ids: set[int] = set()
    if str(player) == "adversary":
        search_state, forbidden_stop_ids = runner._build_root_decision_state_for_adversary(
            current_state=state,
            pre_controller_snapshot=pending_adv_pre_ctrl_snapshot,
            pre_controller_stats=pending_adv_pre_ctrl_stats,
        )

    actions_by_index, mask_t = bundle.mcts._actions_and_mask(
        search_state,
        str(player),
        forbidden_stop_ids=forbidden_stop_ids if str(player) == "adversary" else None,
    )
    valid_mask = _mask_to_list(mask_t)
    valid_indices = [
        int(i)
        for i, ok in enumerate(valid_mask)
        if bool(ok) and int(i) < len(actions_by_index) and actions_by_index[int(i)] is not None
    ]
    return _ExpandedActionSpace(
        player=str(player),
        search_state=search_state,
        forbidden_stop_ids=set(int(x) for x in forbidden_stop_ids),
        actions_by_index=list(actions_by_index),
        valid_mask=list(valid_mask),
        valid_indices=list(valid_indices),
    )


def _align_player_to_valid_actions(
    *,
    bundle: _RunnerBundle,
    state: Any,
    player: str,
    pending_adv_pre_ctrl_snapshot: Any | None,
    pending_adv_pre_ctrl_stats: Any | None,
) -> tuple[str, _ExpandedActionSpace]:
    current = _expand_action_space(
        bundle=bundle,
        state=state,
        player=str(player),
        pending_adv_pre_ctrl_snapshot=pending_adv_pre_ctrl_snapshot,
        pending_adv_pre_ctrl_stats=pending_adv_pre_ctrl_stats,
    )
    if current.valid_indices:
        return str(player), current

    alt_player = _other_player(str(player))
    alternate = _expand_action_space(
        bundle=bundle,
        state=state,
        player=str(alt_player),
        pending_adv_pre_ctrl_snapshot=pending_adv_pre_ctrl_snapshot,
        pending_adv_pre_ctrl_stats=pending_adv_pre_ctrl_stats,
    )
    if alternate.valid_indices:
        return str(alt_player), alternate

    return str(player), current


def _select_model_depth1_action(
    *,
    bundle: _RunnerBundle,
    cfg: ModelTesterConfig,
    expanded: _ExpandedActionSpace,
    model: AlphaZeroModel,
) -> tuple[AdversaryAction | ControllerAction | None, Dict[str, Any]]:
    player = str(expanded.player)
    valid_indices = list(expanded.valid_indices)
    actions_by_index = list(expanded.actions_by_index)
    if not valid_indices:
        return None, {
            "selection_mode": f"model_depth1_{player}_no_valid_action",
            "valid_action_count": 0,
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    alias_to_canon, canon_to_aliases, canonical_indices = bundle.mcts._canonicalize_action_indices(
        player=str(player),
        actions_by_index=actions_by_index,
        valid_indices=valid_indices,
    )

    decision_snapshot, decision_stats = bundle.mcts._snapshot_state_and_stats(expanded.search_state)
    parent_cost = float(bundle.mcts._state_cost(expanded.search_state))
    parent_time = float(expanded.search_state.simulator._time)

    best_idx: int | None = None
    best_tuple: tuple[float, float, float, float, float, float] | None = None
    candidate_rows: list[tuple[int, str, float, float, float, float, float]] = []

    for cidx in canonical_indices:
        action = actions_by_index[int(cidx)]
        if action is None:
            continue

        q_tuple = bundle.mcts._evaluate_depth1_action_q(
            decision_snapshot=decision_snapshot,
            decision_stats=decision_stats,
            parent_player=str(player),
            parent_cost=float(parent_cost),
            parent_time=float(parent_time),
            action=action,
            dnn_model=model,
            model_version=int(cfg.bootstrap_model_version),
        )
        q = float(q_tuple[0])
        reward, discount, bootstrap, child_cost = (
            float(q_tuple[1]),
            float(q_tuple[2]),
            float(q_tuple[3]),
            float(q_tuple[4]),
        )
        candidate_rows.append(
            (
                int(cidx),
                repr(action),
                float(q),
                float(reward),
                float(discount),
                float(bootstrap),
                float(child_cost),
            )
        )
        if best_tuple is None:
            best_idx = int(cidx)
            best_tuple = q_tuple
            continue

        best_q = float(best_tuple[0])
        if str(player) == "adversary":
            better = (q < best_q) or (q == best_q and int(cidx) < int(best_idx))
        else:
            better = (q > best_q) or (q == best_q and int(cidx) < int(best_idx))

        if better:
            best_idx = int(cidx)
            best_tuple = q_tuple

    if best_idx is None or best_tuple is None:
        return None, {
            "selection_mode": f"model_depth1_{player}_empty_canonical_set",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    q, reward, discount, bootstrap, child_cost, _child_time = best_tuple
    selection_mode = (
        "model_depth1_adv_argmin_q"
        if str(player) == "adversary"
        else "model_depth1_ctrl_argmax_q"
    )
    reverse = bool(str(player) != "adversary")
    candidate_rows_sorted = sorted(
        candidate_rows,
        key=lambda row: ((-row[2]) if reverse else row[2], row[0]),
    )
    candidate_rows_top5 = candidate_rows_sorted[:5]
    return actions_by_index[int(best_idx)], {
        "selection_mode": str(selection_mode),
        "valid_action_count": int(len(valid_indices)),
        "iterations_requested": 0,
        "iterations_used": int(len(canonical_indices)),
        "chosen_q_value": float(q),
        "chosen_reward": float(reward),
        "chosen_discount": float(discount),
        "chosen_bootstrap": float(bootstrap),
        "chosen_child_cost": float(child_cost),
        "candidate_ranking_mode": "q_desc" if reverse else "q_asc",
        "candidate_top5_action_reprs": [row[1] for row in candidate_rows_top5],
        "candidate_top5_q_values": [float(row[2]) for row in candidate_rows_top5],
        "candidate_top5_rewards": [float(row[3]) for row in candidate_rows_top5],
        "candidate_top5_discounts": [float(row[4]) for row in candidate_rows_top5],
        "candidate_top5_bootstraps": [float(row[5]) for row in candidate_rows_top5],
        "candidate_top5_child_costs": [float(row[6]) for row in candidate_rows_top5],
        "candidate_ranked_rows": [
            {
                "rank": int(rank),
                "action_repr": str(row[1]),
                "q_value": float(row[2]),
                "immediate_reward": float(row[3]),
                "discount": float(row[4]),
                "bootstrap_value": float(row[5]),
                "child_cost": float(row[6]),
            }
            for rank, row in enumerate(candidate_rows_sorted, start=1)
        ],
    }


def _select_cycle_action(
    *,
    bundle: _RunnerBundle,
    cfg: ModelTesterConfig,
    state: Any,
    player: str,
    pending_adv_pre_ctrl_snapshot: Any | None,
    pending_adv_pre_ctrl_stats: Any | None,
    model: AlphaZeroModel,
    adversary_policy: str,
    controller_policy: str,
) -> tuple[str, AdversaryAction | ControllerAction | None, Dict[str, Any]]:
    player, expanded = _align_player_to_valid_actions(
        bundle=bundle,
        state=state,
        player=str(player),
        pending_adv_pre_ctrl_snapshot=pending_adv_pre_ctrl_snapshot,
        pending_adv_pre_ctrl_stats=pending_adv_pre_ctrl_stats,
    )

    if str(player) == "adversary":
        if str(adversary_policy) == "model_depth1":
            action, info = _select_model_depth1_action(
                bundle=bundle,
                cfg=cfg,
                expanded=expanded,
                model=model,
            )
        else:
            action, info = select_trivial_adversary_action(
                heuristic=str(cfg.trivial_adversary_policy.heuristic),
                actions_by_index=expanded.actions_by_index,
                valid_indices=expanded.valid_indices,
            )
    else:
        if str(controller_policy) == "model_depth1":
            action, info = _select_model_depth1_action(
                bundle=bundle,
                cfg=cfg,
                expanded=expanded,
                model=model,
            )
        else:
            action, info = select_trivial_controller_action(
                runner=bundle.runner,
                state=state,
                heuristic=str(cfg.trivial_policy.heuristic),
                budget_tokens=int(cfg.trivial_policy.budget_tokens),
                eviction_rule=str(cfg.trivial_policy.eviction_rule),
                actions_by_index=expanded.actions_by_index,
                valid_indices=expanded.valid_indices,
            )

    return str(player), action, dict(info or {})


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

    player, expanded = _align_player_to_valid_actions(
        bundle=bundle,
        state=state,
        player=str(player),
        pending_adv_pre_ctrl_snapshot=None,
        pending_adv_pre_ctrl_stats=None,
    )
    if not expanded.valid_indices:
        state = runner.env.initial_state()
        player, _ = _align_player_to_valid_actions(
            bundle=bundle,
            state=state,
            player=str(cfg.start_player),
            pending_adv_pre_ctrl_snapshot=None,
            pending_adv_pre_ctrl_stats=None,
        )
        depth = int(cfg.start_root_depth)

    if history_events:
        for cycle_label in (MODEL_ADV_VS_TRIVIAL_CTRL_LABEL, MODEL_ADV_VS_MODEL_CTRL_LABEL):
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


def _run_policy_cycle(
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
    adversary_policy: str,
    controller_policy: str,
    cycle_file_logger: ArenaGameCycleFileLogger | None,
    model_action_detail_logger: ArenaModelActionDetailLogger | None,
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

    try:
        while (
            turns < int(cfg.arena_max_total_turns)
            and float(state.simulator._time) < deadline_t
        ):
            player, action, selection_info = _select_cycle_action(
                bundle=bundle,
                cfg=cfg,
                state=state,
                player=str(player),
                pending_adv_pre_ctrl_snapshot=pending_adv_pre_ctrl_snapshot,
                pending_adv_pre_ctrl_stats=pending_adv_pre_ctrl_stats,
                model=model,
                adversary_policy=str(adversary_policy),
                controller_policy=str(controller_policy),
            )
            if action is None:
                end_reason = "no_valid_action"
                break

            sim_before = float(state.simulator._time)
            player_before = str(player)
            depth_before = int(depth)
            turn_before = int(turns)
            phase_name = "arena_step"

            pre_controller_snapshot = None
            pre_controller_stats = None
            if str(player_before) == "controller":
                pre_controller_snapshot = state.simulator.snapshot_state()
                pre_controller_stats = state.stats.clone()

            generated = len((action.requests or [])) if isinstance(action, AdversaryAction) else 0

            if (
                model_action_detail_logger is not None
                and isinstance(selection_info.get("candidate_ranked_rows"), list)
                and selection_info.get("candidate_ranked_rows")
            ):
                model_action_detail_logger.write_ranked_actions(
                    game_id=int(game_id),
                    cycle_label=str(cycle_label),
                    phase=str(phase_name),
                    turn=int(turn_before),
                    depth=int(depth_before),
                    player_acted=str(player_before),
                    ranked_rows=list(selection_info.get("candidate_ranked_rows") or []),
                )

            if str(player_before) == "adversary":
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
                    phase=str(phase_name),
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
                    chosen_q_value=selection_info.get("chosen_q_value"),
                    chosen_reward=selection_info.get("chosen_reward"),
                    chosen_discount=selection_info.get("chosen_discount"),
                    chosen_bootstrap=selection_info.get("chosen_bootstrap"),
                    chosen_child_cost=selection_info.get("chosen_child_cost"),
                    candidate_ranking_mode=selection_info.get("candidate_ranking_mode"),
                    candidate_top5_action_reprs=selection_info.get("candidate_top5_action_reprs"),
                    candidate_top5_q_values=selection_info.get("candidate_top5_q_values"),
                    candidate_top5_rewards=selection_info.get("candidate_top5_rewards"),
                    candidate_top5_discounts=selection_info.get("candidate_top5_discounts"),
                    candidate_top5_bootstraps=selection_info.get("candidate_top5_bootstraps"),
                    candidate_top5_child_costs=selection_info.get("candidate_top5_child_costs"),
                )

        if not end_reason and float(state.simulator._time) >= deadline_t:
            end_reason = "time_limit"

        while (
            turns < int(cfg.arena_max_total_turns)
            and cleanup_steps < int(cfg.arena_max_controller_cleanup_steps)
            and float(state.simulator._time) < deadline_t
            and _has_prefill_pending(bundle, state)
        ):
            sim_before = float(state.simulator._time)
            player_before = str(player)
            depth_before = int(depth)
            turn_before = int(turns)
            phase_name = "cleanup_step"

            if str(player_before) == "adversary":
                noop = make_noop_adversary_action()
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
                        selection_mode="cleanup_adversary_noop",
                        valid_action_count=1,
                        iterations_requested=0,
                        iterations_used=0,
                    )
                continue

            expanded = _expand_action_space(
                bundle=bundle,
                state=state,
                player="controller",
                pending_adv_pre_ctrl_snapshot=pending_adv_pre_ctrl_snapshot,
                pending_adv_pre_ctrl_stats=pending_adv_pre_ctrl_stats,
            )
            if str(controller_policy) == "model_depth1":
                action, selection_info = _select_model_depth1_action(
                    bundle=bundle,
                    cfg=cfg,
                    expanded=expanded,
                    model=model,
                )
            else:
                action, selection_info = select_trivial_controller_action(
                    runner=runner,
                    state=state,
                    heuristic=str(cfg.trivial_policy.heuristic),
                    budget_tokens=int(cfg.trivial_policy.budget_tokens),
                    eviction_rule=str(cfg.trivial_policy.eviction_rule),
                    actions_by_index=expanded.actions_by_index,
                    valid_indices=expanded.valid_indices,
                )

            if action is None:
                end_reason = end_reason or "cleanup_no_valid_action"
                break

            if (
                model_action_detail_logger is not None
                and isinstance(selection_info.get("candidate_ranked_rows"), list)
                and selection_info.get("candidate_ranked_rows")
            ):
                model_action_detail_logger.write_ranked_actions(
                    game_id=int(game_id),
                    cycle_label=str(cycle_label),
                    phase=str(phase_name),
                    turn=int(turn_before),
                    depth=int(depth_before),
                    player_acted=str(player_before),
                    ranked_rows=list(selection_info.get("candidate_ranked_rows") or []),
                )

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
                    phase=str(phase_name),
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
                    chosen_q_value=selection_info.get("chosen_q_value"),
                    chosen_reward=selection_info.get("chosen_reward"),
                    chosen_discount=selection_info.get("chosen_discount"),
                    chosen_bootstrap=selection_info.get("chosen_bootstrap"),
                    chosen_child_cost=selection_info.get("chosen_child_cost"),
                    candidate_ranking_mode=selection_info.get("candidate_ranking_mode"),
                    candidate_top5_action_reprs=selection_info.get("candidate_top5_action_reprs"),
                    candidate_top5_q_values=selection_info.get("candidate_top5_q_values"),
                    candidate_top5_rewards=selection_info.get("candidate_top5_rewards"),
                    candidate_top5_discounts=selection_info.get("candidate_top5_discounts"),
                    candidate_top5_bootstraps=selection_info.get("candidate_top5_bootstraps"),
                    candidate_top5_child_costs=selection_info.get("candidate_top5_child_costs"),
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

        root_logger = getattr(runner.mcts, "_root_logger", None)
        if root_logger is not None:
            root_logger.log_root(
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
    finally:
        bundle.mcts.clear_search_state(drop_scratch=True)


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

    for p in cfg.arena_games_dir_path().glob("*.csv"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        if cfg.arena_results_csv_path().exists():
            cfg.arena_results_csv_path().unlink()
    except Exception:
        pass

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
    model_action_detail_logger = ArenaModelActionDetailLogger(cfg.arena_games_dir_path())

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

            cycle1 = _run_policy_cycle(
                bundle=bundle,
                cfg=cfg,
                base_snapshot=base_snapshot,
                base_stats=base_stats,
                base_player=str(base_player),
                base_depth=int(base_depth),
                game_id=int(game_id),
                root_id_base=0,
                model=bundle.model,
                cycle_label=MODEL_ADV_VS_TRIVIAL_CTRL_LABEL,
                adversary_policy="model_depth1",
                controller_policy="trivial",
                cycle_file_logger=cycle_logger,
                model_action_detail_logger=model_action_detail_logger,
            )
            cycle_logger.write_cycle_end(
                game_id=int(game_id),
                cycle_label=MODEL_ADV_VS_TRIVIAL_CTRL_LABEL,
                total_cost=float(cycle1["total_cost"]),
                slo_violations=int(cycle1["slo_violations"]),
                total_lateness=float(cycle1["total_lateness"]),
                end_reason=str(cycle1.get("end_reason", "")),
            )

            cycle2 = _run_policy_cycle(
                bundle=bundle,
                cfg=cfg,
                base_snapshot=base_snapshot,
                base_stats=base_stats,
                base_player=str(base_player),
                base_depth=int(base_depth),
                game_id=int(game_id),
                root_id_base=1_000_000,
                model=bundle.model,
                cycle_label=MODEL_ADV_VS_MODEL_CTRL_LABEL,
                adversary_policy="model_depth1",
                controller_policy="model_depth1",
                cycle_file_logger=cycle_logger,
                model_action_detail_logger=model_action_detail_logger,
            )
            cycle_logger.write_cycle_end(
                game_id=int(game_id),
                cycle_label=MODEL_ADV_VS_MODEL_CTRL_LABEL,
                total_cost=float(cycle2["total_cost"]),
                slo_violations=int(cycle2["slo_violations"]),
                total_lateness=float(cycle2["total_lateness"]),
                end_reason=str(cycle2.get("end_reason", "")),
            )

            delta = float(cycle2["total_cost"]) - float(cycle1["total_cost"])
            if delta < -1e-9:
                better_cycle = MODEL_ADV_VS_MODEL_CTRL_LABEL
            elif delta > 1e-9:
                better_cycle = MODEL_ADV_VS_TRIVIAL_CTRL_LABEL
            else:
                better_cycle = "tie"

            rows.append(
                {
                    "game_id": int(game_id),
                    "history_hops": int(hops),
                    "model_checkpoint": str(cfg.model_checkpoint_path),
                    "cycle1_label": MODEL_ADV_VS_TRIVIAL_CTRL_LABEL,
                    "cycle1_slo_violations": int(cycle1["slo_violations"]),
                    "cycle1_total_lateness": float(cycle1["total_lateness"]),
                    "cycle1_total_cost": float(cycle1["total_cost"]),
                    "cycle1_end_reason": str(cycle1.get("end_reason", "")),
                    "cycle2_label": MODEL_ADV_VS_MODEL_CTRL_LABEL,
                    "cycle2_slo_violations": int(cycle2["slo_violations"]),
                    "cycle2_total_lateness": float(cycle2["total_lateness"]),
                    "cycle2_total_cost": float(cycle2["total_cost"]),
                    "cycle2_end_reason": str(cycle2.get("end_reason", "")),
                    "cost_delta_cycle2_minus_cycle1": float(delta),
                    "better_cycle": str(better_cycle),
                    "cycle1_log_file": str(
                        cfg.arena_games_dir_path() / f"game_{int(game_id)}_{MODEL_ADV_VS_TRIVIAL_CTRL_LABEL}.csv"
                    ),
                    "cycle2_log_file": str(
                        cfg.arena_games_dir_path() / f"game_{int(game_id)}_{MODEL_ADV_VS_MODEL_CTRL_LABEL}.csv"
                    ),
                }
            )

        out_csv = cfg.arena_results_csv_path()
        _write_results_csv(out_csv, rows)
        return out_csv
    finally:
        _safe_close_simulator(bundle.simulator)
