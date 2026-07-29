"""MCTS-valued arena-state analysis for v4-Adv classical models.

This script intentionally does not modify ``Model_Tester``. It builds the same
GV3 virtual environment and selects one clean arena-start state.

Default mode is ``per_child``: for every canonical action at the parent, apply
that action once to create the child state, then run a fresh fixed-budget MCTS
from that child state. This makes the bootstrap-vs-MCTS comparison fair because
each parent child receives exactly the same MCTS budget.
"""

from __future__ import annotations

import argparse
import csv
import gc
import multiprocessing as mp
import os
import random
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib

from vidur.mcts.Game_Versions.Game_Version3.Model_Tester.config import (
    DEFAULT_MODEL_TESTER_CONFIG,
)
from vidur.mcts.Game_Versions.Game_Version3.mcts import MCTSConfig, VidurMCTS
from vidur.mcts.Game_Versions.Game_Version3.multiProcessUtils import (
    _build_env_and_simulator,
    _set_global_seeds,
)


class ZeroBootstrapModel:
    """Minimal model used only for smoke-testing tree mechanics."""

    @property
    def model_name(self) -> str:
        return "zero_bootstrap"

    def infer_from_inputs(self, inputs: Any, player: str, *, device: Any = None) -> tuple[float, list[float]]:
        del inputs, player, device
        return 0.0, []


def _next_player(player: str) -> str:
    return "controller" if str(player) == "adversary" else "adversary"


def _load_model(model_path: Path | None, *, feature_dim: int) -> Any:
    if model_path is None:
        return ZeroBootstrapModel()

    model = joblib.load(model_path)
    if callable(getattr(model, "infer_from_inputs", None)):
        try:
            from vidur.mcts.Game_Versions.Game_Version3.DNN.infer import enable_inputs_extras

            enable_inputs_extras()
        except Exception:
            pass
        return model

    from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper

    try:
        from vidur.mcts.Game_Versions.Game_Version3.DNN.infer import enable_inputs_extras

        enable_inputs_extras()
    except Exception:
        pass
    return V4AdvHGBWrapper(model, feature_dim=int(feature_dim), model_tag=f"wrapped:{model_path.name}")


def _patch_tree_mcts_runtime_issues(mcts: VidurMCTS) -> None:
    """Temporary compatibility patch for current local ``mcts.py``.

    Keep this local to the experiment script so the source file remains
    untouched until the permanent patch is applied.
    """

    if not hasattr(mcts, "_node_counter"):
        setattr(mcts, "_node_counter", int(getattr(mcts, "_node_id_counter", 0)))

    def get_transition_reward(self: VidurMCTS, parent_cost: float, child_cost: float) -> float:
        return float(parent_cost) - float(child_cost)

    mcts.get_transition_reward = types.MethodType(get_transition_reward, mcts)


def _make_mcts_config(args: argparse.Namespace, *, iterations: int, seed: int) -> MCTSConfig:
    cfg = MCTSConfig()
    cfg.rng = random.Random(int(seed))
    cfg.num_simulations = int(iterations)
    cfg.uct_c = float(args.uct_c)
    cfg.discount_factor = float(args.discount_factor)
    cfg._discount_time_denom = float(args.discount_time_denom)
    cfg.log_flag = False
    return cfg


def _limit_native_threads(num_threads: int) -> None:
    threads = max(1, int(num_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=threads)
    except Exception:
        pass


def _build_env(args: argparse.Namespace) -> tuple[Any, Any]:
    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        environment_lang=str(args.environment_lang),
        use_virtual_env=True,
        model_device="cpu",
        history_hops_min=0,
        history_hops_max=0,
        history_hops_unique=False,
        history_hops_force_zero=True,
        arena_time_limit_sec=float(args.arena_time_limit_sec),
    )
    pipeline_cfg = cfg.to_pipeline_cfg()
    pipeline_cfg.validate()
    _set_global_seeds(
        int(pipeline_cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(pipeline_cfg.game_v2.reproducibility.torch_deterministic),
    )
    simulator, env, _unused, _explore_cfg = _build_env_and_simulator(
        pipeline_cfg,
        use_virtual_env=bool(pipeline_cfg.use_virtual_env),
    )
    return simulator, env


def _valid_canonical_root_actions(
    mcts: VidurMCTS,
    state: Any,
    player: str,
) -> tuple[str, list[Any | None], list[bool], list[int]]:
    actions, mask_t = mcts.get_actions_and_mask(state, str(player))
    valid_mask = [bool(x) for x in mask_t.tolist()]
    valid_indices = [
        int(i)
        for i, ok in enumerate(valid_mask)
        if bool(ok) and int(i) < len(actions) and actions[int(i)] is not None
    ]
    if not valid_indices:
        alt_player = _next_player(player)
        actions, mask_t = mcts.get_actions_and_mask(state, alt_player)
        valid_mask = [bool(x) for x in mask_t.tolist()]
        valid_indices = [
            int(i)
            for i, ok in enumerate(valid_mask)
            if bool(ok) and int(i) < len(actions) and actions[int(i)] is not None
        ]
        player = alt_player

    _alias, _aliases, canonical_indices = mcts.canonicalize_action_indices(
        player=str(player),
        actions_by_index=list(actions),
        valid_indices=list(valid_indices),
    )
    return str(player), list(actions), list(valid_mask), list(canonical_indices)


def _bootstrap_state_value(
    mcts: VidurMCTS,
    model: Any,
    state: Any,
    *,
    player: str,
    model_version: int,
) -> float:
    return float(
        mcts._bootstrap_value(
            dnn_model=model,
            state=state,
            player=str(player),
            model_version=int(model_version),
            use_model_bootstrap=bool(int(model_version) > 0),
        )
    )


def _mcts_root_value(tree: VidurMCTS, result: Any) -> tuple[float, int]:
    """Return a controller-valued MCTS state estimate and actual root visits."""

    root = tree._root
    if root is None:
        return float(getattr(result, "best_action_value", 0.0)), 0

    # Root mean is the value backed up to this state over the fixed budget. It is
    # the cleanest single state-value statistic for comparing against bootstrap.
    visits = int(getattr(root, "visits", 0) or 0)
    if visits > 0:
        return float(root.mean_value()), visits

    return float(getattr(result, "best_action_value", 0.0)), visits


def _write_analysis_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "game_id",
        "hop",
        "parent_node_id",
        "parent_player",
        "valid_canon_child_id",
        "child_node_id",
        "action_repr",
        "child_mcts_iterations",
        "child_model_bootstrap_value",
        "child_mcts_value",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _run_shared_root_analysis(
    *,
    args: argparse.Namespace,
    model: Any,
    tree: VidurMCTS,
    state: Any,
    player: str,
    canonical_indices: list[int],
) -> list[dict[str, Any]]:
    """Old diagnostic mode: one parent-root MCTS with UCT visit allocation."""

    effective_iterations = max(int(args.mcts_iterations), len(canonical_indices))
    tree.search_dnn(
        dnn_model=model,
        rootState=state,
        root_player=str(player),
        game_id=int(args.game_id),
        root_id=int(args.root_id),
        root_node_id_override=int(args.parent_node_id),
        root_depth=int(args.root_depth),
        mcts_iter=int(effective_iterations),
        model_version=int(args.model_version),
        use_model_bootstrap=bool(int(args.model_version) > 0),
        root_phase="arena_mcts_value_analysis",
        cycle_label="mcts_value_analysis",
    )

    root = tree._root
    if root is None:
        raise RuntimeError("MCTS root was not created")

    rows: list[dict[str, Any]] = []
    for canon_idx in sorted(canonical_indices):
        child = root.children.get(int(canon_idx))
        if child is None:
            rows.append(
                {
                    "game_id": int(args.game_id),
                    "hop": int(args.hop),
                    "parent_node_id": int(root.node_id),
                    "parent_player": str(root.player),
                    "valid_canon_child_id": int(canon_idx),
                    "child_node_id": "",
                    "action_repr": repr(root.actions_by_index[int(canon_idx)]),
                    "child_mcts_iterations": 0,
                    "child_model_bootstrap_value": "",
                    "child_mcts_value": "",
                }
            )
            continue

        child_state = tree.scratch_restore(child.cached_sim_snapshot, child.cached_stats)
        rows.append(
            {
                "game_id": int(args.game_id),
                "hop": int(args.hop),
                "parent_node_id": int(root.node_id),
                "parent_player": str(root.player),
                "valid_canon_child_id": int(canon_idx),
                "child_node_id": int(child.node_id),
                "action_repr": repr(child.parent_action),
                "child_mcts_iterations": int(child.visits),
                "child_model_bootstrap_value": _bootstrap_state_value(
                    tree,
                    model,
                    child_state,
                    player=str(child.player),
                    model_version=int(args.model_version),
                ),
                "child_mcts_value": float(child.mean_value()),
            }
        )
    return rows


def _run_per_child_analysis(
    *,
    args: argparse.Namespace,
    model: Any,
    env: Any,
    tree: VidurMCTS,
    state: Any,
    player: str,
    actions_by_index: list[Any | None],
    canonical_indices: list[int],
) -> list[dict[str, Any]]:
    """Run a fresh fixed-budget MCTS from every canonical parent child."""

    if int(args.num_processes) > 1 and len(canonical_indices) > 1:
        worker_count = min(72, int(args.num_processes), len(canonical_indices))
        tasks = [
            {
                "args": dict(vars(args)),
                "canon_idx": int(canon_idx),
                "row_idx": int(row_idx),
            }
            for row_idx, canon_idx in enumerate(sorted(canonical_indices), start=1)
        ]
        ctx = mp.get_context(str(args.mp_start_method))
        rows: list[dict[str, Any]] = []
        with ctx.Pool(processes=int(worker_count), maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(_per_child_worker, tasks):
                rows.append(row)
                print(
                    "[arena-mcts] child done "
                    f"canon={row['valid_canon_child_id']} "
                    f"iters={row['child_mcts_iterations']}",
                    flush=True,
                )
        return sorted(rows, key=lambda row: int(row["valid_canon_child_id"]))

    sim = state.simulator
    parent_snapshot = sim.snapshot_state_fast() if hasattr(sim, "snapshot_state_fast") else sim.snapshot_state()
    parent_stats = state.stats.clone()
    parent_node_id = int(args.parent_node_id)
    rows: list[dict[str, Any]] = []

    for row_idx, canon_idx in enumerate(sorted(canonical_indices), start=1):
        action = actions_by_index[int(canon_idx)]
        if action is None:
            continue

        child_node_id = parent_node_id + int(row_idx)
        parent_state = tree.scratch_restore(parent_snapshot, parent_stats)
        child_state = tree.apply_action(parent_state, str(player), action)
        child_player = _next_player(player)

        child_bootstrap = _bootstrap_state_value(
            tree,
            model,
            child_state,
            player=str(child_player),
            model_version=int(args.model_version),
        )

        child_cfg = _make_mcts_config(
            args,
            iterations=int(args.per_child_mcts_iterations),
            seed=int(args.seed) + int(canon_idx) + 10_000,
        )
        child_tree = VidurMCTS(env=env, mctsConfig=child_cfg)
        _patch_tree_mcts_runtime_issues(child_tree)
        try:
            result = child_tree.search_dnn(
                dnn_model=model,
                rootState=child_state,
                root_player=str(child_player),
                game_id=int(args.game_id),
                root_id=int(args.root_id) + int(row_idx),
                root_node_id_override=int(child_node_id),
                root_depth=int(args.root_depth) + 1,
                mcts_iter=int(args.per_child_mcts_iterations),
                model_version=int(args.model_version),
                use_model_bootstrap=bool(int(args.model_version) > 0),
                root_phase="arena_child_mcts_value_analysis",
                cycle_label="mcts_value_analysis",
            )
            mcts_value, actual_iters = _mcts_root_value(child_tree, result)
        finally:
            try:
                child_tree.close()
            except Exception:
                pass

        rows.append(
            {
                "game_id": int(args.game_id),
                "hop": int(args.hop),
                "parent_node_id": int(parent_node_id),
                "parent_player": str(player),
                "valid_canon_child_id": int(canon_idx),
                "child_node_id": int(child_node_id),
                "action_repr": repr(action),
                "child_mcts_iterations": int(actual_iters),
                "child_model_bootstrap_value": float(child_bootstrap),
                "child_mcts_value": float(mcts_value),
            }
        )

    return rows


def _per_child_worker(task: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**dict(task["args"]))
    canon_idx = int(task["canon_idx"])
    row_idx = int(task["row_idx"])
    _limit_native_threads(int(args.worker_threads))

    model_path = Path(args.model_path).expanduser() if args.model_path else None
    model = _load_model(model_path, feature_dim=int(args.feature_dim))
    simulator, env = _build_env(args)
    tree_cfg = _make_mcts_config(args, iterations=int(args.mcts_iterations), seed=int(args.seed))
    tree = VidurMCTS(env=env, mctsConfig=tree_cfg)
    _patch_tree_mcts_runtime_issues(tree)

    try:
        if int(args.hop) != 0:
            raise NotImplementedError("parallel worker currently supports clean arena-start states only: --hop 0")

        state = env.initial_state()
        player = str(args.start_player)
        player, actions, _valid_mask, canonical_indices = _valid_canonical_root_actions(tree, state, player)
        if canon_idx not in set(int(x) for x in canonical_indices):
            raise RuntimeError(f"canon_idx={canon_idx} not valid for rebuilt parent state")

        action = actions[int(canon_idx)]
        if action is None:
            raise RuntimeError(f"canon_idx={canon_idx} resolved to None action")

        parent_node_id = int(args.parent_node_id)
        child_node_id = parent_node_id + int(row_idx)
        child_state = tree.apply_action(state, str(player), action)
        child_player = _next_player(player)

        child_bootstrap = _bootstrap_state_value(
            tree,
            model,
            child_state,
            player=str(child_player),
            model_version=int(args.model_version),
        )

        child_cfg = _make_mcts_config(
            args,
            iterations=int(args.per_child_mcts_iterations),
            seed=int(args.seed) + int(canon_idx) + 10_000,
        )
        child_tree = VidurMCTS(env=env, mctsConfig=child_cfg)
        _patch_tree_mcts_runtime_issues(child_tree)
        try:
            result = child_tree.search_dnn(
                dnn_model=model,
                rootState=child_state,
                root_player=str(child_player),
                game_id=int(args.game_id),
                root_id=int(args.root_id) + int(row_idx),
                root_node_id_override=int(child_node_id),
                root_depth=int(args.root_depth) + 1,
                mcts_iter=int(args.per_child_mcts_iterations),
                model_version=int(args.model_version),
                use_model_bootstrap=bool(int(args.model_version) > 0),
                root_phase="arena_child_mcts_value_analysis",
                cycle_label="mcts_value_analysis",
            )
            mcts_value, actual_iters = _mcts_root_value(child_tree, result)
        finally:
            try:
                child_tree.close()
            except Exception:
                pass

        return {
            "game_id": int(args.game_id),
            "hop": int(args.hop),
            "parent_node_id": int(parent_node_id),
            "parent_player": str(player),
            "valid_canon_child_id": int(canon_idx),
            "child_node_id": int(child_node_id),
            "action_repr": repr(action),
            "child_mcts_iterations": int(actual_iters),
            "child_model_bootstrap_value": float(child_bootstrap),
            "child_mcts_value": float(mcts_value),
        }
    finally:
        try:
            tree.close()
        except Exception:
            pass
        for method in ("shutdown", "close", "stop"):
            fn = getattr(simulator, method, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass
        gc.collect()


def run_one_state(args: argparse.Namespace) -> Path:
    model_path = Path(args.model_path).expanduser() if args.model_path else None
    model = _load_model(model_path, feature_dim=int(args.feature_dim))
    simulator, env = _build_env(args)
    tree_cfg = _make_mcts_config(args, iterations=int(args.mcts_iterations), seed=int(args.seed))

    tree = VidurMCTS(env=env, mctsConfig=tree_cfg)
    _patch_tree_mcts_runtime_issues(tree)

    try:
        if int(args.hop) != 0:
            raise NotImplementedError("This wrapper currently supports clean arena-start states only: --hop 0")

        state = env.initial_state()
        player = str(args.start_player)
        player, actions, _valid_mask, canonical_indices = _valid_canonical_root_actions(tree, state, player)
        if not canonical_indices:
            raise RuntimeError(f"No valid canonical root actions for player={player}")

        if str(args.mode) == "shared_root":
            rows = _run_shared_root_analysis(
                args=args,
                model=model,
                tree=tree,
                state=state,
                player=player,
                canonical_indices=canonical_indices,
            )
            iteration_note = f"requested_iterations={int(args.mcts_iterations)}"
        else:
            rows = _run_per_child_analysis(
                args=args,
                model=model,
                env=env,
                tree=tree,
                state=state,
                player=player,
                actions_by_index=actions,
                canonical_indices=canonical_indices,
            )
            iteration_note = f"per_child_iterations={int(args.per_child_mcts_iterations)}"

        out_dir = Path(args.output_dir).expanduser()
        out_path = out_dir / "state_mcts_analysis.csv"
        _write_analysis_csv(out_path, rows)
        print(f"[arena-mcts] wrote {out_path}")
        print(
            "[arena-mcts] "
            f"game_id={int(args.game_id)} hop={int(args.hop)} player={player} "
            f"canonical_children={len(canonical_indices)} mode={str(args.mode)} {iteration_note}",
            flush=True,
        )
        return out_path
    finally:
        try:
            tree.close()
        except Exception:
            pass
        for method in ("shutdown", "close", "stop"):
            fn = getattr(simulator, method, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass
        gc.collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one arena-start state with standalone GV3 tree MCTS.")
    parser.add_argument("--model-path", default="", help="Wrapped or bare joblib model. Empty uses zero bootstrap smoke model.")
    parser.add_argument("--model-version", type=int, default=1)
    parser.add_argument("--feature-dim", type=int, default=226)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--game-id", type=int, default=12_000_000)
    parser.add_argument("--hop", type=int, default=0)
    parser.add_argument("--root-id", type=int, default=0)
    parser.add_argument("--parent-node-id", type=int, default=0)
    parser.add_argument("--root-depth", type=int, default=0)
    parser.add_argument("--start-player", choices=("adversary", "controller"), default="adversary")
    parser.add_argument("--mode", choices=("per_child", "shared_root"), default="per_child")
    parser.add_argument("--mcts-iterations", type=int, default=25, help="Iterations for --mode shared_root.")
    parser.add_argument("--per-child-mcts-iterations", type=int, default=100)
    parser.add_argument("--num-processes", type=int, default=1, help="Parallel workers for --mode per_child, capped at 72.")
    parser.add_argument("--worker-threads", type=int, default=1, help="BLAS/OpenMP threads per worker.")
    parser.add_argument("--mp-start-method", choices=("spawn", "forkserver", "fork"), default="spawn")
    parser.add_argument("--uct-c", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--discount-time-denom", type=float, default=0.015725797204323228)
    parser.add_argument("--arena-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--environment-lang", choices=("python", "native"), default="python")
    return parser.parse_args()


def main() -> None:
    run_one_state(_parse_args())


if __name__ == "__main__":
    main()
