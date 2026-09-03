"""Run Model_Tester with MCTS-valued model actions without editing Model_Tester.

This script monkeypatches the private model-depth1 selector at runtime. The arena
runner, logging, history handling, cleanup, and trivial policies still come from
``Game_Version3.Model_Tester``.
"""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from typing import Any

from vidur.Game_Version3.Model_Tester import runner as tester_runner
from vidur.Game_Version3.Model_Tester.config import (
    DEFAULT_MODEL_TESTER_CONFIG,
)
from vidur.Game_Version3.mcts import MCTSConfig, VidurMCTS

from vidur.bellman_v4_adv.arena_mcts_value_runner2 import (
    _limit_native_threads,
    _make_mcts_config,
    _mcts_root_value,
    _next_player,
    _patch_tree_mcts_runtime_issues,
)


_RUNTIME_ARGS: argparse.Namespace | None = None
_RUNTIME_POOL: mp.pool.Pool | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_BUNDLE: Any | None = None


def _snapshot_state_and_stats(state: Any) -> tuple[Any, Any]:
    sim = state.simulator
    snapshot = sim.snapshot_state_fast() if hasattr(sim, "snapshot_state_fast") else sim.snapshot_state()
    return snapshot, state.stats.clone()


def _restore_state(tree: VidurMCTS, snapshot: Any, stats: Any) -> Any:
    return tree.scratch_restore(snapshot, stats)


def _candidate_worker_init(args_dict: dict[str, Any], cfg: Any) -> None:
    global _WORKER_ARGS, _WORKER_BUNDLE
    _WORKER_ARGS = argparse.Namespace(**dict(args_dict))
    _limit_native_threads(int(_WORKER_ARGS.worker_threads))
    _WORKER_BUNDLE = tester_runner._build_bundle(cfg)


def _candidate_worker(task: dict[str, Any]) -> dict[str, Any]:
    args = _WORKER_ARGS
    bundle = _WORKER_BUNDLE
    if args is None or bundle is None:
        # Fallback for one-off pools/tests that do not use the initializer.
        args = argparse.Namespace(**dict(task["args"]))
        _limit_native_threads(int(args.worker_threads))
        bundle = tester_runner._build_bundle(task["cfg"])
    action = task["action"]
    canon_idx = int(task["canon_idx"])
    row_idx = int(task["row_idx"])
    player = str(task["player"])
    parent_cost = float(task["parent_cost"])
    parent_time = float(task["parent_time"])
    decision_snapshot = task["decision_snapshot"]
    decision_stats = task["decision_stats"]

    tree_cfg = _make_mcts_config(args, iterations=int(args.per_child_mcts_iterations), seed=int(args.seed) + canon_idx + 10_000)
    tree = VidurMCTS(env=bundle.env, mctsConfig=tree_cfg)
    _patch_tree_mcts_runtime_issues(tree)

    try:
        parent_state = _restore_state(tree, decision_snapshot, decision_stats)
        child_state = tree.apply_action(parent_state, player, action)
        child_player = _next_player(player)

        child_cost = float(tree.get_state_cost(child_state))
        reward = float(parent_cost) - float(child_cost)
        child_time = float(child_state.simulator._time)
        discount_time = getattr(child_state.stats, "transition_discount_time", None)
        if discount_time is None:
            discount_time = child_time
        discount = float(tree.time_discount(float(discount_time), float(parent_time)))

        child_tree = VidurMCTS(env=bundle.env, mctsConfig=tree_cfg)
        _patch_tree_mcts_runtime_issues(child_tree)
        try:
            result = child_tree.search_dnn(
                dnn_model=bundle.model,
                rootState=child_state,
                root_player=str(child_player),
                game_id=int(task["game_id"]),
                root_id=int(task["root_id"]) + row_idx,
                root_node_id_override=int(task["root_node_id_base"]) + row_idx,
                root_depth=int(task["root_depth"]) + 1,
                mcts_iter=int(args.per_child_mcts_iterations),
                model_version=int(args.model_version),
                use_model_bootstrap=bool(int(args.model_version) > 0)
                and not bool(getattr(args, "disable_model_bootstrap", False)),
                root_phase="arena_mcts_model_action_child",
                cycle_label=str(task.get("cycle_label", "")),
            )
            child_mcts_value, actual_iters = _mcts_root_value(child_tree, result)
        finally:
            try:
                child_tree.close()
            except Exception:
                pass

        q_value = float(reward) + float(discount) * float(child_mcts_value)
        return {
            "canon_idx": int(canon_idx),
            "action_repr": repr(action),
            "q_value": float(q_value),
            "reward": float(reward),
            "discount": float(discount),
            "bootstrap_value": float(child_mcts_value),
            "child_cost": float(child_cost),
            "child_time": float(child_time),
            "actual_iters": int(actual_iters),
        }
    finally:
        try:
            tree.close()
        except Exception:
            pass
        if _WORKER_BUNDLE is None:
            try:
                tester_runner._safe_close_simulator(bundle.simulator)
            except Exception:
                pass
        gc.collect()


def _serial_candidate(
    *,
    args: argparse.Namespace,
    bundle: Any,
    action: Any,
    canon_idx: int,
    row_idx: int,
    player: str,
    parent_cost: float,
    parent_time: float,
    decision_snapshot: Any,
    decision_stats: Any,
    game_id: int,
    root_id: int,
    root_node_id_base: int,
    root_depth: int,
    cycle_label: str,
) -> dict[str, Any]:
    tree_cfg = _make_mcts_config(args, iterations=int(args.per_child_mcts_iterations), seed=int(args.seed) + int(canon_idx) + 10_000)
    tree = VidurMCTS(env=bundle.env, mctsConfig=tree_cfg)
    _patch_tree_mcts_runtime_issues(tree)
    try:
        parent_state = _restore_state(tree, decision_snapshot, decision_stats)
        child_state = tree.apply_action(parent_state, player, action)
        child_player = _next_player(player)
        child_cost = float(tree.get_state_cost(child_state))
        reward = float(parent_cost) - float(child_cost)
        child_time = float(child_state.simulator._time)
        discount_time = getattr(child_state.stats, "transition_discount_time", None)
        if discount_time is None:
            discount_time = child_time
        discount = float(tree.time_discount(float(discount_time), float(parent_time)))

        child_tree = VidurMCTS(env=bundle.env, mctsConfig=tree_cfg)
        _patch_tree_mcts_runtime_issues(child_tree)
        try:
            result = child_tree.search_dnn(
                dnn_model=bundle.model,
                rootState=child_state,
                root_player=str(child_player),
                game_id=int(game_id),
                root_id=int(root_id) + int(row_idx),
                root_node_id_override=int(root_node_id_base) + int(row_idx),
                root_depth=int(root_depth) + 1,
                mcts_iter=int(args.per_child_mcts_iterations),
                model_version=int(args.model_version),
                use_model_bootstrap=bool(int(args.model_version) > 0)
                and not bool(getattr(args, "disable_model_bootstrap", False)),
                root_phase="arena_mcts_model_action_child",
                cycle_label=str(cycle_label),
            )
            child_mcts_value, actual_iters = _mcts_root_value(child_tree, result)
        finally:
            try:
                child_tree.close()
            except Exception:
                pass

        q_value = float(reward) + float(discount) * float(child_mcts_value)
        return {
            "canon_idx": int(canon_idx),
            "action_repr": repr(action),
            "q_value": float(q_value),
            "reward": float(reward),
            "discount": float(discount),
            "bootstrap_value": float(child_mcts_value),
            "child_cost": float(child_cost),
            "child_time": float(child_time),
            "actual_iters": int(actual_iters),
        }
    finally:
        try:
            tree.close()
        except Exception:
            pass



def _safe_label(value: Any) -> str:
    text = str(value or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _mcts_visit_log_paths(
    args: argparse.Namespace,
    *,
    game_id: int,
    cycle_label: str,
    phase: str,
    turn: int,
    player: str,
) -> tuple[Path, Path]:
    base = Path(getattr(args, "mcts_visit_log_dir", "") or Path(args.output_dir) / "mcts_visit_logs")
    stem = (
        f"game_{int(game_id)}_"
        f"{_safe_label(cycle_label)}_"
        f"turn_{int(turn):06d}_"
        f"{_safe_label(phase)}_"
        f"{_safe_label(player)}"
    )
    return base / f"{stem}_root.csv", base / f"{stem}_children.csv"


def _set_runtime_selection_context(**kwargs: Any) -> None:
    args = _RUNTIME_ARGS
    if args is None:
        return

    game_id_raw = kwargs.get("game_id")
    turn_raw = kwargs.get("turn")
    depth_raw = kwargs.get("depth")
    game_id = int(game_id_raw) if game_id_raw is not None else 0
    turn = int(turn_raw) if turn_raw is not None else 0
    depth = int(depth_raw) if depth_raw is not None else 0

    setattr(args, "current_game_id", int(game_id))
    setattr(args, "current_turn", int(turn))
    setattr(args, "current_phase", str(kwargs.get("phase", "")))
    setattr(args, "current_cycle_label", str(kwargs.get("cycle_label", "")))
    setattr(args, "current_root_depth", int(depth))
    # Unique enough for logs, but still stable across runs for the same game/turn.
    setattr(args, "current_root_id", int(game_id) * 10_000 + int(turn))
    setattr(args, "current_player", str(kwargs.get("player", "")))


def _select_model_mcts_shared_root_action(
    *,
    args: argparse.Namespace,
    bundle: Any,
    expanded: Any,
    player: str,
    actions_by_index: list[Any | None],
    valid_indices: list[int],
    canonical_indices: list[int],
    game_id: int,
    root_id: int,
    root_depth: int,
    cycle_label: str,
    phase: str,
    turn: int,
) -> tuple[Any | None, dict[str, Any]]:
    iterations = int(getattr(args, "shared_root_mcts_iterations", getattr(args, "per_child_mcts_iterations", 100)))
    iterations = max(iterations, len(canonical_indices))

    tree_cfg = _make_mcts_config(
        args,
        iterations=int(iterations),
        seed=int(args.seed) + int(root_id) + int(turn) + (0 if str(player) == "controller" else 1_000_000),
    )
    if bool(getattr(args, "write_mcts_visit_logs", False)):
        root_log_path, child_log_path = _mcts_visit_log_paths(
            args,
            game_id=int(game_id),
            cycle_label=str(cycle_label),
            phase=str(phase),
            turn=int(turn),
            player=str(player),
        )
        tree_cfg.log_flag = True
        tree_cfg.log_path = root_log_path
        tree_cfg.tree_log_path = child_log_path

    tree = VidurMCTS(env=bundle.env, mctsConfig=tree_cfg)
    _patch_tree_mcts_runtime_issues(tree)
    try:
        result = tree.search_dnn(
            dnn_model=bundle.model,
            rootState=expanded.search_state,
            root_player=str(player),
            game_id=int(game_id),
            root_id=int(root_id),
            root_node_id_override=int(root_id),
            root_depth=int(root_depth),
            mcts_iter=int(iterations),
            model_version=int(args.model_version),
            use_model_bootstrap=bool(int(args.model_version) > 0)
            and not bool(getattr(args, "disable_model_bootstrap", False)),
            root_phase="arena_mcts_shared_root",
            cycle_label=str(cycle_label),
            turn=int(turn),
        )

        root = tree._root
        if root is None:
            return None, {
                "selection_mode": f"model_mcts_shared_root_{player}_missing_root",
                "valid_action_count": int(len(valid_indices)),
                "iterations_requested": int(iterations),
                "iterations_used": 0,
            }

        rows: list[dict[str, Any]] = []
        for canon_idx in sorted(int(x) for x in canonical_indices):
            action = None
            if int(canon_idx) < len(root.actions_by_index):
                action = root.actions_by_index[int(canon_idx)]
            if action is None and int(canon_idx) < len(actions_by_index):
                action = actions_by_index[int(canon_idx)]

            child = root.children.get(int(canon_idx))
            if child is None or int(getattr(child, "visits", 0) or 0) <= 0:
                q_value = float("-inf") if str(player) == "controller" else float("inf")
                rows.append(
                    {
                        "canon_idx": int(canon_idx),
                        "action_repr": repr(action),
                        "q_value": float(q_value),
                        "reward": 0.0,
                        "discount": 1.0,
                        "bootstrap_value": float(q_value),
                        "child_cost": 0.0,
                        "child_time": 0.0,
                        "actual_iters": 0,
                        "child_node_id": "",
                    }
                )
                continue

            q_value = float(child.mean_value())
            rows.append(
                {
                    "canon_idx": int(canon_idx),
                    "action_repr": repr(action if action is not None else child.parent_action),
                    "q_value": float(q_value),
                    "reward": float(getattr(child, "reward", 0.0)),
                    "discount": float(getattr(child, "edge_discount", 1.0)),
                    # In shared-root mode this is the MCTS child Q estimate after UCT allocation.
                    "bootstrap_value": float(q_value),
                    "child_cost": float(getattr(child, "state_cost", 0.0)),
                    "child_time": float(getattr(child, "sim_time", 0.0)),
                    "actual_iters": int(getattr(child, "visits", 0) or 0),
                    "child_node_id": int(getattr(child, "node_id", -1)),
                }
            )
    finally:
        try:
            tree.close()
        except Exception:
            pass

    finite_rows = [r for r in rows if math.isfinite(float(r["q_value"]))]
    if not finite_rows:
        return None, {
            "selection_mode": f"model_mcts_shared_root_{player}_no_visited_children",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": int(iterations),
            "iterations_used": int(getattr(result, "root_visits", 0) or 0),
        }

    reverse = str(player) == "controller"
    rows_sorted = sorted(
        finite_rows,
        key=lambda r: ((-float(r["q_value"])) if reverse else float(r["q_value"]), int(r["canon_idx"])),
    )
    best = rows_sorted[0]
    best_idx = int(best["canon_idx"])
    top5 = rows_sorted[:5]
    selection_mode = "model_mcts_shared_root_ctrl_argmax_q" if str(player) == "controller" else "model_mcts_shared_root_adv_argmin_q"
    root_visits = int(getattr(root, "visits", 0) or 0)
    return actions_by_index[best_idx], {
        "selection_mode": selection_mode,
        "valid_action_count": int(len(valid_indices)),
        "iterations_requested": int(iterations),
        "iterations_used": int(root_visits),
        "chosen_q_value": float(best["q_value"]),
        "chosen_reward": float(best["reward"]),
        "chosen_discount": float(best["discount"]),
        "chosen_bootstrap": float(best["bootstrap_value"]),
        "chosen_child_cost": float(best["child_cost"]),
        "candidate_ranking_mode": "q_desc" if reverse else "q_asc",
        "candidate_top5_action_reprs": [str(r["action_repr"]) for r in top5],
        "candidate_top5_q_values": [float(r["q_value"]) for r in top5],
        "candidate_top5_rewards": [float(r["reward"]) for r in top5],
        "candidate_top5_discounts": [float(r["discount"]) for r in top5],
        "candidate_top5_bootstraps": [float(r["bootstrap_value"]) for r in top5],
        "candidate_top5_child_costs": [float(r["child_cost"]) for r in top5],
        "candidate_ranked_rows": [
            {
                "rank": int(rank),
                "action_repr": str(r["action_repr"]),
                "q_value": float(r["q_value"]),
                "immediate_reward": float(r["reward"]),
                "discount": float(r["discount"]),
                "bootstrap_value": float(r["bootstrap_value"]),
                "child_cost": float(r["child_cost"]),
                "mcts_child_visits": int(r["actual_iters"]),
                "mcts_child_node_id": r.get("child_node_id", ""),
            }
            for rank, r in enumerate(rows_sorted, start=1)
        ],
    }

def _select_model_mcts_action(
    *,
    bundle: Any,
    cfg: Any,
    expanded: Any,
    model: Any,
) -> tuple[Any | None, dict[str, Any]]:
    del model
    args = _RUNTIME_ARGS
    if args is None:
        raise RuntimeError("arena_mcts_model_tester runtime args were not initialised")

    player = str(expanded.player)
    valid_indices = list(expanded.valid_indices)
    actions_by_index = list(expanded.actions_by_index)
    if not valid_indices:
        return None, {
            "selection_mode": f"model_mcts_{player}_no_valid_action",
            "valid_action_count": 0,
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    probe_cfg = _make_mcts_config(args, iterations=1, seed=int(args.seed))
    probe = VidurMCTS(env=bundle.env, mctsConfig=probe_cfg)
    _patch_tree_mcts_runtime_issues(probe)
    try:
        _alias, _aliases, canonical_indices = probe.canonicalize_action_indices(
            player=str(player),
            actions_by_index=actions_by_index,
            valid_indices=valid_indices,
        )
        decision_snapshot, decision_stats = _snapshot_state_and_stats(expanded.search_state)
        parent_cost = float(probe.get_state_cost(expanded.search_state))
        parent_time = float(expanded.search_state.simulator._time)
    finally:
        probe.close()

    if not canonical_indices:
        return None, {
            "selection_mode": f"model_mcts_{player}_empty_canonical_set",
            "valid_action_count": len(valid_indices),
            "iterations_requested": int(args.per_child_mcts_iterations),
            "iterations_used": 0,
        }

    if len(canonical_indices) == 1:
        only_idx = int(canonical_indices[0])
        return actions_by_index[only_idx], {
            "selection_mode": f"model_mcts_{player}_single_canonical_action",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
            "chosen_q_value": None,
            "chosen_reward": None,
            "chosen_discount": None,
            "chosen_bootstrap": None,
            "chosen_child_cost": None,
            "candidate_ranking_mode": "single",
            "candidate_top5_action_reprs": [repr(actions_by_index[only_idx])],
            "candidate_top5_q_values": [],
            "candidate_top5_rewards": [],
            "candidate_top5_discounts": [],
            "candidate_top5_bootstraps": [],
            "candidate_top5_child_costs": [],
            "candidate_ranked_rows": [],
        }

    game_id = int(getattr(args, "current_game_id", 0))
    root_id = int(getattr(args, "current_root_id", 0))
    root_depth = int(getattr(args, "current_root_depth", 0))
    root_node_id_base = int(root_id) * 10_000
    cycle_label = str(getattr(args, "current_cycle_label", ""))
    turn = int(getattr(args, "current_turn", -1))
    phase = str(getattr(args, "current_phase", "arena_step"))

    if str(getattr(args, "mcts_selection_mode", "per_child")) == "shared_root":
        return _select_model_mcts_shared_root_action(
            args=args,
            bundle=bundle,
            expanded=expanded,
            player=str(player),
            actions_by_index=actions_by_index,
            valid_indices=valid_indices,
            canonical_indices=canonical_indices,
            game_id=int(game_id),
            root_id=int(root_id),
            root_depth=int(root_depth),
            cycle_label=str(cycle_label),
            phase=str(phase),
            turn=int(turn),
        )

    if int(args.num_processes) > 1 and len(canonical_indices) > 1:
        worker_count = min(72, int(args.num_processes), len(canonical_indices))
        tasks = []
        for row_idx, canon_idx in enumerate(sorted(canonical_indices), start=1):
            tasks.append(
                {
                    "args": dict(vars(args)),
                    "cfg": cfg,
                    "action": actions_by_index[int(canon_idx)],
                    "canon_idx": int(canon_idx),
                    "row_idx": int(row_idx),
                    "player": str(player),
                    "parent_cost": float(parent_cost),
                    "parent_time": float(parent_time),
                    "decision_snapshot": decision_snapshot,
                    "decision_stats": decision_stats,
                    "game_id": int(game_id),
                    "root_id": int(root_id),
                    "root_node_id_base": int(root_node_id_base),
                    "root_depth": int(root_depth),
                    "cycle_label": str(cycle_label),
                }
            )
        rows: list[dict[str, Any]] = []
        pool = _RUNTIME_POOL
        if pool is not None:
            for row in pool.imap_unordered(_candidate_worker, tasks):
                rows.append(row)
        else:
            ctx = mp.get_context(str(args.mp_start_method))
            with ctx.Pool(
                processes=int(worker_count),
                initializer=_candidate_worker_init,
                initargs=(dict(vars(args)), cfg),
            ) as pool:
                for row in pool.imap_unordered(_candidate_worker, tasks):
                    rows.append(row)
    else:
        rows = []
        for row_idx, canon_idx in enumerate(sorted(canonical_indices), start=1):
            rows.append(
                _serial_candidate(
                    args=args,
                    bundle=bundle,
                    action=actions_by_index[int(canon_idx)],
                    canon_idx=int(canon_idx),
                    row_idx=int(row_idx),
                    player=str(player),
                    parent_cost=float(parent_cost),
                    parent_time=float(parent_time),
                    decision_snapshot=decision_snapshot,
                    decision_stats=decision_stats,
                    game_id=int(game_id),
                    root_id=int(root_id),
                    root_node_id_base=int(root_node_id_base),
                    root_depth=int(root_depth),
                    cycle_label=str(cycle_label),
                )
            )

    if not rows:
        return None, {
            "selection_mode": f"model_mcts_{player}_no_candidates",
            "valid_action_count": len(valid_indices),
            "iterations_requested": int(args.per_child_mcts_iterations),
            "iterations_used": 0,
        }

    reverse = player == "controller"
    rows_sorted = sorted(rows, key=lambda r: ((-float(r["q_value"])) if reverse else float(r["q_value"]), int(r["canon_idx"])))
    best = rows_sorted[0]
    best_idx = int(best["canon_idx"])
    top5 = rows_sorted[:5]
    selection_mode = "model_mcts_ctrl_argmax_q" if player == "controller" else "model_mcts_adv_argmin_q"
    return actions_by_index[best_idx], {
        "selection_mode": selection_mode,
        "valid_action_count": int(len(valid_indices)),
        "iterations_requested": int(args.per_child_mcts_iterations),
        "iterations_used": int(len(canonical_indices) * int(args.per_child_mcts_iterations)),
        "chosen_q_value": float(best["q_value"]),
        "chosen_reward": float(best["reward"]),
        "chosen_discount": float(best["discount"]),
        "chosen_bootstrap": float(best["bootstrap_value"]),
        "chosen_child_cost": float(best["child_cost"]),
        "candidate_ranking_mode": "q_desc" if reverse else "q_asc",
        "candidate_top5_action_reprs": [str(r["action_repr"]) for r in top5],
        "candidate_top5_q_values": [float(r["q_value"]) for r in top5],
        "candidate_top5_rewards": [float(r["reward"]) for r in top5],
        "candidate_top5_discounts": [float(r["discount"]) for r in top5],
        "candidate_top5_bootstraps": [float(r["bootstrap_value"]) for r in top5],
        "candidate_top5_child_costs": [float(r["child_cost"]) for r in top5],
        "candidate_ranked_rows": [
            {
                "rank": int(rank),
                "action_repr": str(r["action_repr"]),
                "q_value": float(r["q_value"]),
                "immediate_reward": float(r["reward"]),
                "discount": float(r["discount"]),
                "bootstrap_value": float(r["bootstrap_value"]),
                "child_cost": float(r["child_cost"]),
            }
            for rank, r in enumerate(rows_sorted, start=1)
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Model_Tester arena game with MCTS-valued model actions.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-version", type=int, required=True)
    parser.add_argument("--disable-model-bootstrap", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--game-id-start", type=int, default=12_000_000)
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--mcts-selection-mode", choices=("per_child", "shared_root"), default="per_child")
    parser.add_argument("--shared-root-mcts-iterations", type=int, default=10_000)
    parser.add_argument("--per-child-mcts-iterations", type=int, default=100)
    parser.add_argument("--write-mcts-visit-logs", action="store_true")
    parser.add_argument("--mcts-visit-log-dir", default="")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--mp-start-method", choices=("spawn", "forkserver", "fork"), default="spawn")
    parser.add_argument("--trivial-budget-tokens", type=int, default=512)
    parser.add_argument("--arena-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--arena-max-total-turns", type=int, default=4096)
    parser.add_argument("--arena-max-controller-cleanup-steps", type=int, default=1024)
    parser.add_argument("--history-hops-min", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_min)
    parser.add_argument("--history-hops-max", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_max)
    parser.add_argument("--history-seed", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_seed)
    parser.add_argument(
        "--history-hops-unique",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_unique,
    )
    parser.add_argument(
        "--history-hops-force-zero",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_force_zero,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uct-c", type=float, default=1.4)
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--discount-time-denom", type=float, default=0.015725797204323228)
    parser.add_argument("--environment-lang", choices=("python",), default="python")
    parser.add_argument("--no-arena-game-logs", action="store_true")
    parser.add_argument("--write-model-action-detail-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    global _RUNTIME_ARGS, _RUNTIME_POOL
    args = _parse_args()
    _RUNTIME_ARGS = args
    _limit_native_threads(int(args.worker_threads))

    trivial_policy = replace(
        DEFAULT_MODEL_TESTER_CONFIG.trivial_policy,
        budget_tokens=int(args.trivial_budget_tokens),
    )
    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind="classical_joblib",
        model_checkpoint_path=str(args.model_path),
        output_dir=str(args.output_dir),
        num_games=int(args.num_games),
        game_id_start=int(args.game_id_start),
        environment_lang="python",
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        history_hops_unique=bool(args.history_hops_unique),
        history_hops_force_zero=bool(args.history_hops_force_zero),
        arena_time_limit_sec=float(args.arena_time_limit_sec),
        arena_max_total_turns=int(args.arena_max_total_turns),
        arena_max_controller_cleanup_steps=int(args.arena_max_controller_cleanup_steps),
        bootstrap_model_version=int(args.model_version),
        write_arena_game_logs=not bool(args.no_arena_game_logs),
        write_model_action_detail_logs=bool(args.write_model_action_detail_logs),
        arena_num_processes=1,
        arena_worker_threads=1,
        arena_mp_start_method="spawn",
        trivial_policy=trivial_policy,
        skip_model_ctrl_cycle=False,
    )

    pool: mp.pool.Pool | None = None
    if int(args.num_processes) > 1:
        ctx = mp.get_context(str(args.mp_start_method))
        worker_count = min(72, int(args.num_processes))
        pool = ctx.Pool(
            processes=int(worker_count),
            initializer=_candidate_worker_init,
            initargs=(dict(vars(args)), cfg),
        )
        _RUNTIME_POOL = pool

    old_selector = tester_runner._select_model_depth1_action
    old_context_hook = getattr(tester_runner, "_model_action_selection_context_hook", None)
    try:
        tester_runner._select_model_depth1_action = _select_model_mcts_action
        tester_runner._model_action_selection_context_hook = _set_runtime_selection_context
        out_csv = tester_runner.run_model_vs_trivial_tester(cfg)
        print(f"[arena-mcts-game] completed {out_csv}")
    finally:
        tester_runner._select_model_depth1_action = old_selector
        tester_runner._model_action_selection_context_hook = old_context_hook
        if pool is not None:
            pool.close()
            pool.join()
            _RUNTIME_POOL = None


if __name__ == "__main__":
    main()
