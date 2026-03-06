from __future__ import annotations

import csv
import json
import random
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from vidur.simulator import Simulator

from ..DNN.history_root import HistoryRootGenerator
from ..alphaZeroParrallel import configure_simulation
from ..environment import (
    AdversaryAction,
    AdversaryRequestSpec,
    ControllerAction,
    VidurMCTSEnvironment,
    VidurMCTSState,
)
from ..launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions
from ..virtual_environment import VirtualVidurMCTSEnvironment
from ..virtual_simulator import VirtualSimulator
from .bellman import compute_base_step_time, q_value, state_cost
from .config import LinearPipelineConfig, WorkerResult, WorkerTask
from .features import extract_features, state_hash
from .model import LinearValueModel


@dataclass(frozen=True)
class _HistoryRow:
    root_player: str
    phase: str
    best_action_json: str


def _mask_to_list(mask: Any) -> List[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


def _build_env(cfg: LinearPipelineConfig, use_virtual_env: bool) -> Tuple[Any, Any]:
    sim_cfg = configure_simulation(cfg.sim.cli_args)
    setattr(sim_cfg.cluster_config.cache_config, "assume_infinite_kv", True)

    slo_options = RequestSLOOptions(
        prefill_slos=tuple(cfg.constraints.prefill_slos),
        decode_slos=tuple(cfg.constraints.decode_slos),
    )
    constraints = MCTSConstraintConfig(
        maximum_qps=cfg.constraints.maximum_qps,
        min_request_tokens=cfg.constraints.min_request_tokens,
        max_request_tokens=cfg.constraints.max_request_tokens,
        interval_request_size=cfg.constraints.interval_request_size,
        request_slo_options=slo_options,
        prefill_slowdown=cfg.constraints.prefill_slowdown,
        prefill_profile_path=cfg.constraints.prefill_profile_path,
    )
    explore_cfg = MCTSExploreConfig(
        simulation_depth=2,
        simulation_random_tries=1,
        exploration_constant=1.7,
        max_branching=cfg.collection.max_branching,
        controller_budget_combs=10,
    )

    if use_virtual_env:
        simulator = VirtualSimulator(sim_cfg, register_atexit=False)
        env = VirtualVidurMCTSEnvironment(
            base_simulator=simulator,
            constraints=constraints,
            explore_cfg=explore_cfg,
            native_enabled=False,
            native_strict_compat=True,
        )
    else:
        simulator = Simulator(sim_cfg, register_atexit=False)
        env = VidurMCTSEnvironment(
            base_simulator=simulator,
            constraints=constraints,
            explore_cfg=explore_cfg,
        )

    return simulator, env


def _actions_and_valid(env: Any, state: VidurMCTSState, player: str, max_samples: int) -> Tuple[List[Optional[object]], List[int]]:
    if player == "controller":
        actions_by_index, mask = env.sample_controller_actions(state, max_samples)
    else:
        actions_by_index, mask = env.sample_adversary_actions(state, max_samples)
    mask_list = _mask_to_list(mask)
    valid = [i for i, ok in enumerate(mask_list) if ok and i < len(actions_by_index) and actions_by_index[i] is not None]
    return actions_by_index, valid


def _apply_action_inplace(env: Any, state: VidurMCTSState, player: str, action: object) -> Tuple[VidurMCTSState, str]:
    if player == "controller":
        state = env.apply_controller_action_only(state, action, inplace=True)
        return state, "adversary"
    state = env.apply_adversary_action_only(state, action, inplace=True)
    return state, "controller"


def _advance_forced_until_branching(
    env: Any,
    state: VidurMCTSState,
    player: str,
    *,
    max_hops: int,
    max_samples: int,
) -> Tuple[VidurMCTSState, str]:
    for _ in range(int(max_hops)):
        actions_by_index, valid = _actions_and_valid(env, state, player, max_samples)
        if len(valid) != 1:
            return state, player
        idx = int(valid[0])
        action = actions_by_index[idx]
        assert action is not None
        state, player = _apply_action_inplace(env, state, player, action)
    raise RuntimeError(f"Exceeded max_hops={max_hops} while advancing forced chain")


def _is_explicit_history_row(row: _HistoryRow) -> bool:
    s = (row.best_action_json or "").strip()
    if not s:
        return False
    try:
        d = json.loads(s)
    except Exception:
        return False
    return isinstance(d, dict) and ("history_phase" in d)


def _parse_action_json(action_json: str) -> object:
    d = json.loads(action_json)
    typ = str(d.get("type", "")).strip().lower()

    if typ == "adversary":
        reqs: List[AdversaryRequestSpec] = []
        for r in (d.get("requests") or []):
            reqs.append(
                AdversaryRequestSpec(
                    prefill_tokens=int(r.get("prefill_tokens", 0)),
                    decode_tokens=int(r.get("decode_tokens", 0)),
                    prefill_slo=float(r.get("prefill_slo", 0.0)),
                    decode_slo=float(r.get("decode_slo", 0.0)),
                )
            )
        stop = [int(x) for x in (d.get("stop_decode_ids") or [])]
        return AdversaryAction(requests=reqs, stop_decode_ids=stop)

    if typ == "controller":
        tok_alloc = {int(k): int(v) for k, v in (d.get("token_allocations") or {}).items()}
        prefill_alloc = {int(k): int(v) for k, v in (d.get("prefill_allocations") or {}).items()}
        decode_alloc = {int(k): int(v) for k, v in (d.get("decode_allocations") or {}).items()}
        sel = d.get("selected_request_ids")
        selected_ids = None if sel is None else [int(x) for x in sel]
        mapping = d.get("mapping")
        mapping_t = tuple(int(x) for x in mapping) if mapping is not None else None

        return ControllerAction(
            token_budget=int(d.get("token_budget", 0)),
            selected_request_ids=selected_ids,
            token_allocations=tok_alloc,
            prefill_allocations=prefill_alloc,
            decode_allocations=decode_alloc,
            heuristic=d.get("heuristic"),
            strategy=d.get("strategy"),
            mapping=mapping_t,
        )

    raise ValueError(f"Unknown action type in best_action_json: {typ!r}")


def _load_history_rows(path: Path) -> List[_HistoryRow]:
    rows: List[_HistoryRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                _HistoryRow(
                    root_player=str(d.get("root_player", "") or "").strip(),
                    phase=str(d.get("phase", "") or "").strip(),
                    best_action_json=str(d.get("best_action_json", "") or "").strip(),
                )
            )
    if not rows:
        raise RuntimeError(f"No rows found in history CSV: {path}")
    return rows


def _replay_history_to_state_after_last_action(
    env: Any,
    rows: List[_HistoryRow],
    *,
    root_player: str,
    align_branching_roots: bool,
    max_forced_hops: int,
    max_samples: int,
) -> Tuple[VidurMCTSState, str]:
    state = env.initial_state()
    player = str(root_player)

    action_rows = [r for r in rows if (r.best_action_json or "").strip()]
    if not action_rows:
        return state, player

    for i, row in enumerate(action_rows):
        is_hist = _is_explicit_history_row(row)
        if align_branching_roots and (not is_hist):
            state, player = _advance_forced_until_branching(
                env,
                state,
                player,
                max_hops=max_forced_hops,
                max_samples=max_samples,
            )

        if row.root_player and row.root_player != player:
            raise RuntimeError(
                f"Replay mismatch row={i}: row.root_player={row.root_player}, player={player}"
            )

        action = _parse_action_json(row.best_action_json)
        state, player = _apply_action_inplace(env, state, player, action)

        if i < len(action_rows) - 1 and align_branching_roots and (not is_hist):
            state, player = _advance_forced_until_branching(
                env,
                state,
                player,
                max_hops=max_forced_hops,
                max_samples=max_samples,
            )

    return state, player


def _advance_to_controller_branching(
    env: Any,
    state: VidurMCTSState,
    player: str,
    *,
    max_hops: int,
    max_samples: int,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[VidurMCTSState, str, bool]:
    """
    Move forward deterministically until we hit a controller branching state.
    Returns (state, player, terminal).
    """
    for _ in range(int(max_hops)):
        actions_by_index, valid = _actions_and_valid(env, state, player, max_samples)
        if not valid:
            return state, player, True

        if player == "controller" and len(valid) >= 2:
            return state, player, False

        acting_player = str(player)
        before_request_ids: set[int] = set()
        if acting_player == "adversary":
            try:
                before_request_ids = set(env._build_request_lookup(state.simulator).keys())
            except Exception:
                before_request_ids = set()

        if player == "adversary":
            idx = int(max(valid))  # max-request deterministic action in current indexing
        else:
            idx = int(valid[0])

        action = actions_by_index[idx]
        assert action is not None
        state, player = _apply_action_inplace(env, state, player, action)
        if step_callback is not None:
            step_callback(
                {
                    "phase": "forced_transition",
                    "acting_player": acting_player,
                    "action_index": int(idx),
                    "action_repr": repr(action),
                    "num_valid_actions": int(len(valid)),
                    "state_after": state,
                    "before_request_ids": before_request_ids,
                    "next_player": str(player),
                }
            )

    raise RuntimeError(f"Exceeded max_hops={max_hops} while seeking controller branching state")


def _load_model_for_worker(model_ckpt_path: Path, device: torch.device) -> LinearValueModel:
    payload = torch.load(model_ckpt_path, map_location="cpu")
    model = LinearValueModel(num_features=int(payload.get("num_features", 30)))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def _clone_from_state_snapshot(env: Any, snapshot: Any, stats_template: Any) -> VidurMCTSState:
    if hasattr(env, "clone_state_from_snapshot"):
        return env.clone_state_from_snapshot(snapshot, stats_template)

    # Fallback for compatibility.
    s = env.initial_state()
    s.simulator.restore_state(snapshot)
    s.stats = stats_template.clone()
    return s


def _collect_branching_samples(
    *,
    cfg: LinearPipelineConfig,
    env: Any,
    model: LinearValueModel,
    root_state: VidurMCTSState,
    root_player: str,
    worker_id: int,
    history_hop: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))

    train_target = int(cfg.collection.train_samples_per_worker)
    eval_target = int(cfg.collection.eval_samples_per_worker)

    base_step_time = compute_base_step_time(env, cfg.bellman.base_step_tokens)

    device = next(model.parameters()).device

    state, player, terminal = _advance_to_controller_branching(
        env,
        root_state,
        root_player,
        max_hops=cfg.collection.max_forced_hops,
        max_samples=cfg.collection.enum_max_samples,
    )

    queue: deque[Tuple[VidurMCTSState, str, str]] = deque()
    seen: set[str] = set()
    branching_snapshots: List[Tuple[Any, Any, str, str]] = []

    def maybe_enqueue(st: VidurMCTSState, pl: str) -> bool:
        if pl != "controller":
            return False
        acts, valid = _actions_and_valid(env, st, pl, cfg.collection.enum_max_samples)
        if len(valid) < 2:
            return False
        h = state_hash(env, st, pl)
        if h in seen:
            return False
        seen.add(h)
        queue.append((st, pl, h))
        snapshot = st.simulator.snapshot_state()
        branching_snapshots.append((snapshot, st.stats.clone(), pl, h))
        return True

    if not terminal:
        maybe_enqueue(state, player)

    train_feats: List[np.ndarray] = []
    train_targets: List[float] = []
    train_pred_before: List[float] = []
    train_meta: List[Dict[str, Any]] = []

    eval_feats: List[np.ndarray] = []
    eval_targets: List[float] = []
    eval_pred_before: List[float] = []
    eval_meta: List[Dict[str, Any]] = []

    progress_every = 250
    next_train_report = progress_every
    next_eval_report = progress_every

    def _report_progress(*, force: bool = False) -> None:
        nonlocal next_train_report, next_eval_report
        t_count = len(train_feats)
        e_count = len(eval_feats)
        if force or t_count >= next_train_report or e_count >= next_eval_report:
            print(
                f"[linear worker {int(worker_id)} hop={int(history_hop)}] "
                f"collected train={t_count}/{train_target} eval={e_count}/{eval_target}",
                flush=True,
            )
            while next_train_report <= t_count:
                next_train_report += progress_every
            while next_eval_report <= e_count:
                next_eval_report += progress_every

    max_expansions = (train_target + eval_target) * 20
    expansions = 0

    while (len(train_feats) < train_target or len(eval_feats) < eval_target) and expansions < max_expansions:
        expansions += 1

        if not queue:
            if not branching_snapshots:
                break
            snap, stats_t, pl, h = branching_snapshots[rng.randrange(len(branching_snapshots))]
            st = _clone_from_state_snapshot(env, snap, stats_t)
            queue.append((st, pl, h))

        current_state, current_player, h = queue.popleft()
        if current_player != "controller":
            continue

        actions_by_index, valid = _actions_and_valid(
            env,
            current_state,
            current_player,
            cfg.collection.enum_max_samples,
        )
        if len(valid) < 2:
            continue

        feat_s = extract_features(env, current_state)
        with torch.no_grad():
            pred_before = float(
                model.predict(torch.from_numpy(feat_s).to(device=device, dtype=torch.float32)).item()
            )

        sim_time_s = float(getattr(current_state.simulator, "_time", 0.0))
        cost_s = state_cost(env, current_state)

        q_values: List[float] = []
        child_payloads: List[Tuple[VidurMCTSState, str, bool]] = []

        for aidx in valid:
            action = actions_by_index[aidx]
            assert action is not None

            child_state = env.apply_controller_action_only(current_state, action, inplace=False)
            child_player = "adversary"

            child_state, child_player, child_terminal = _advance_to_controller_branching(
                env,
                child_state,
                child_player,
                max_hops=cfg.collection.max_total_steps_per_state,
                max_samples=cfg.collection.enum_max_samples,
            )

            cost_next = state_cost(env, child_state)
            sim_time_next = float(getattr(child_state.simulator, "_time", 0.0))
            dt = max(0.0, sim_time_next - sim_time_s)

            feat_next = extract_features(env, child_state)
            with torch.no_grad():
                v_next = float(
                    model.predict(torch.from_numpy(feat_next).to(device=device, dtype=torch.float32)).item()
                )

            q = q_value(
                cost_s=cost_s,
                cost_next=cost_next,
                v_next=v_next,
                discount_factor=cfg.bellman.discount_factor,
                delta_time=dt,
                base_step_time=base_step_time,
            )
            q_values.append(float(q))
            child_payloads.append((child_state, child_player, child_terminal))

        if not q_values:
            continue

        best_j = int(np.argmax(np.asarray(q_values, dtype=np.float32)))
        best_action_idx = int(valid[best_j])
        max_q = float(q_values[best_j])

        meta = {
            "worker_id": int(worker_id),
            "history_hop": int(history_hop),
            "state_hash": h,
            "sim_time": sim_time_s,
            "best_action_idx": best_action_idx,
            "max_q": max_q,
            "cost_s": float(cost_s),
            "num_valid_actions": int(len(valid)),
        }

        if len(eval_feats) < eval_target:
            eval_feats.append(feat_s)
            eval_targets.append(max_q)
            eval_pred_before.append(pred_before)
            eval_meta.append(meta)
            _report_progress()
        elif len(train_feats) < train_target:
            train_feats.append(feat_s)
            train_targets.append(max_q)
            train_pred_before.append(pred_before)
            train_meta.append(meta)
            _report_progress()

        for child_state, child_player, child_terminal in child_payloads:
            if child_terminal:
                continue
            maybe_enqueue(child_state, child_player)

    def _stack_or_empty(rows: List[np.ndarray]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 30), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    _report_progress(force=True)

    return {
        "train_features": _stack_or_empty(train_feats),
        "train_targets": np.asarray(train_targets, dtype=np.float32),
        "train_pred_before": np.asarray(train_pred_before, dtype=np.float32),
        "train_meta": train_meta,
        "eval_features": _stack_or_empty(eval_feats),
        "eval_targets": np.asarray(eval_targets, dtype=np.float32),
        "eval_pred_before": np.asarray(eval_pred_before, dtype=np.float32),
        "eval_meta": eval_meta,
    }


def _save_meta_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "worker_id",
        "history_hop",
        "state_hash",
        "sim_time",
        "best_action_idx",
        "max_q",
        "cost_s",
        "num_valid_actions",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _generate_root_state(task: WorkerTask, env: Any) -> Tuple[VidurMCTSState, str]:
    root_player = str(task.cfg.collection.root_player)
    state = env.initial_state()
    player = root_player

    history_csv = str(task.cfg.collection.history_csv or "").strip()
    if history_csv:
        rows = _load_history_rows(Path(history_csv))
        state, player = _replay_history_to_state_after_last_action(
            env,
            rows,
            root_player=root_player,
            align_branching_roots=bool(task.cfg.collection.align_branching_roots),
            max_forced_hops=int(task.cfg.collection.max_forced_hops),
            max_samples=int(task.cfg.collection.enum_max_samples),
        )

    hgen = HistoryRootGenerator(
        env=env,
        max_branching=int(task.cfg.collection.max_branching),
        iter_logger=None,
        root_logger=None,
    )

    state, player, _depth, _next_node_id, _last_node_id = hgen.generate_history_root(
        state=state,
        player=player,
        depth=0,
        nontrivial_hops=int(task.history_hop),
        game_id=int(task.round_idx) * 1000 + int(task.worker_id),
        root_id_for_logs=int(task.worker_id),
        seed=int(task.seed),
        log_history=False,
        max_total_steps=int(task.cfg.collection.max_forced_hops),
        log_node_id_start=0,
        log_parent_id_start=None,
    )

    return state, player


def run_collection_worker(task: WorkerTask, result_q: Any) -> None:
    try:
        random.seed(int(task.seed))
        np.random.seed(int(task.seed) % (2**32 - 1))
        torch.manual_seed(int(task.seed))

        device = torch.device(task.cfg.device)

        _, env = _build_env(task.cfg, use_virtual_env=bool(task.cfg.collection.use_virtual_env))
        model = _load_model_for_worker(Path(task.model_ckpt_path), device)

        root_state, root_player = _generate_root_state(task, env)

        shard = _collect_branching_samples(
            cfg=task.cfg,
            env=env,
            model=model,
            root_state=root_state,
            root_player=root_player,
            worker_id=task.worker_id,
            history_hop=task.history_hop,
            seed=task.seed,
        )

        out_dir = Path(task.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_npz = out_dir / f"worker_{task.worker_id:02d}_train.npz"
        eval_npz = out_dir / f"worker_{task.worker_id:02d}_eval.npz"
        train_meta_csv = out_dir / f"worker_{task.worker_id:02d}_train_meta.csv"
        eval_meta_csv = out_dir / f"worker_{task.worker_id:02d}_eval_meta.csv"

        np.savez_compressed(
            train_npz,
            features=shard["train_features"],
            targets=shard["train_targets"],
            pred_before=shard["train_pred_before"],
        )
        np.savez_compressed(
            eval_npz,
            features=shard["eval_features"],
            targets=shard["eval_targets"],
            pred_before=shard["eval_pred_before"],
        )

        _save_meta_csv(train_meta_csv, shard["train_meta"])
        _save_meta_csv(eval_meta_csv, shard["eval_meta"])

        result_q.put(
            WorkerResult(
                worker_id=int(task.worker_id),
                history_hop=int(task.history_hop),
                train_npz=str(train_npz),
                eval_npz=str(eval_npz),
                train_meta_csv=str(train_meta_csv),
                eval_meta_csv=str(eval_meta_csv),
                train_count=int(shard["train_features"].shape[0]),
                eval_count=int(shard["eval_features"].shape[0]),
                ok=True,
            )
        )
    except Exception:
        result_q.put(
            WorkerResult(
                worker_id=int(task.worker_id),
                history_hop=int(task.history_hop),
                train_npz="",
                eval_npz="",
                train_meta_csv="",
                eval_meta_csv="",
                train_count=0,
                eval_count=0,
                ok=False,
                error=traceback.format_exc(),
            )
        )
