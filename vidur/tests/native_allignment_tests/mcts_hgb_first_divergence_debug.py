from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from common import import_native_cpp, make_args, prepare_python_roots


@dataclass(frozen=True)
class ChildStats:
    visits: int
    value_sum: float
    mean_value: float
    reward: float
    discount: float
    state_cost: float
    sim_time: float


@dataclass(frozen=True)
class SearchStats:
    root_visits: int
    root_value_sum: float
    root_mean_value: float
    best_action_index: int
    children: dict[int, ChildStats]


def _state_payload(env: Any, state: Any) -> dict[str, Any]:
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    return nlt._native_state_payload(env, state)


def _prepare_root() -> tuple[Any, Any, Any, Any, Any, int, Any, dict[str, Any], Any, Any, int]:
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
    from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
    from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner
    import joblib
    from state_inference_allignment_test import DEFAULT_HGB_MODEL_PATH, _export_hgb_to_native_text
    from vidur.Game_Version3.DNN import infer as infer_module

    native = import_native_cpp(force_build=False)
    args = make_args(
        "mcts_alignment",
        num_roots=int(os.environ.get("MCTS_HGB_DEBUG_NUM_ROOTS", "32")),
        history_hops_min=0,
        history_hops_max=100,
        history_seed=202613,
        frontier_parity_roots=1,
        bellman_q_tolerance=1e-4,
        model_version=47,
        checkpoint_path=str(DEFAULT_HGB_MODEL_PATH),
    )
    cfg_python, simulator, env, explore_cfg, python_roots = prepare_python_roots(args)

    payload = _cfg_payload(cfg_python, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = True
    payload["native_search_mode"] = "full_tree"
    payload["root_dirichlet_noise_enabled"] = False
    payload["pb_c_base"] = float(cfg_python.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["max_forced_hops"] = int(cfg_python.max_forced_hops_per_root)

    infer_module.enable_inputs_extras()
    hgb_model = joblib.load(DEFAULT_HGB_MODEL_PATH)
    native_export_path = _export_hgb_to_native_text(
        hgb_model,
        Path(args.output_dir) / "v4_adv_hgb_native_export.tsv",
    )
    hgb_runtime = native.NewFeatures226HGBRuntime()
    hgb_runtime.load_model_export(str(native_export_path))

    runner = SelfPlayRunner(
        env=env,
        mcts=nlt.VidurMCTS(env=env, explore_cfg=explore_cfg),
        model=hgb_model,
        writer=NoopReplayWriter(),
        device_for_features=torch.device("cpu"),
        game_v2_cfg=cfg_python.game_v2,
    )

    target_root_id_env = os.environ.get("MCTS_HGB_DEBUG_ROOT_ID")
    target_root_id = int(target_root_id_env) if target_root_id_env not in (None, "") else None
    selected = None
    for pr in python_roots:
        root_state = pr.root_state
        root_player = str(pr.root_player)
        root_depth = int(pr.root_depth)
        root_state, root_player, root_depth = runner._advance_to_branching_root(
            root_state,
            root_player,
            root_depth,
            max_hops=int(cfg_python.max_forced_hops_per_root),
        )
        if root_player == "adversary":
            root_state, _ = runner._build_root_decision_state_for_adversary(
                current_state=root_state,
                pre_controller_snapshot=pr.pre_controller_snapshot,
                pre_controller_stats=pr.pre_controller_stats,
            )

        if root_player == "controller":
            actions, mask = env.sample_controller_actions(root_state)
        else:
            actions, mask = env.sample_adversary_actions(root_state)
        valid_count = sum(bool(mask[i]) and actions[i] is not None for i in range(len(actions)))
        if valid_count <= 1:
            continue
        if target_root_id is not None and int(pr.root_id) != target_root_id:
            continue
        selected = (pr, root_state, root_player, root_depth, valid_count)
        break

    if selected is None:
        suffix = f" for root_id={target_root_id}" if target_root_id is not None else ""
        raise RuntimeError(f"no multi-action root found{suffix}")

    return native, args, cfg_python, env, runner, int(args.history_seed), selected, payload, simulator, hgb_runtime, 47


def _run_python(
    *,
    cfg_python: Any,
    env: Any,
    runner: Any,
    args: Any,
    pr: Any,
    root_state: Any,
    root_player: str,
    root_depth: int,
    iterations: int,
) -> SearchStats:
    from vidur.Game_Version3.mcts import MCTSConfig, VidurMCTS

    full_cfg = MCTSConfig()
    full_cfg.rng = random.Random(int(args.history_seed) + int(pr.root_id))
    full_cfg.num_simulations = int(iterations)
    full_cfg.mcts_iterations = int(iterations)
    full_cfg.uct_c = float(cfg_python.game_v2.mcts_search.pb_c_init)
    full_cfg.discount_factor = float(cfg_python.game_v2.mcts_search.discount_factor)
    full_cfg._discount_time_denom = float(
        cfg_python.game_v2.mcts_search.discount_time_denominator_sec or 0.015725797204323228
    )
    full_cfg.log_flag = False

    py_mcts = VidurMCTS(env=env, mctsConfig=full_cfg)
    result = py_mcts.search_dnn(
        dnn_model=runner.model,
        rootState=root_state.fork(flag=False),
        root_player=root_player,
        game_id=0,
        root_id=int(pr.root_id),
        root_node_id_override=pr.root_node_id_override,
        root_depth=int(root_depth),
        mcts_iter=int(iterations),
        model_version=47,
        use_model_bootstrap=True,
        one_step_value_mode=False,
    )
    root = py_mcts._root
    if root is None:
        raise RuntimeError("python root missing")

    return SearchStats(
        root_visits=int(root.visits),
        root_value_sum=float(root.value_sum),
        root_mean_value=float(root.mean_value()),
        best_action_index=int(result.best_action_index if result.best_action_index is not None else -1),
        children={
            int(idx): ChildStats(
                visits=int(child.visits),
                value_sum=float(child.value_sum),
                mean_value=float(child.mean_value()),
                reward=float(child.reward),
                discount=float(child.edge_discount),
                state_cost=float(child.state_cost),
                sim_time=float(child.sim_time),
            )
            for idx, child in root.children.items()
        },
    )


def _run_native(
    *,
    native: Any,
    env: Any,
    payload: dict[str, Any],
    pr: Any,
    root_state: Any,
    root_player: str,
    root_depth: int,
    iterations: int,
    seed: int,
) -> SearchStats:
    runtime = payload["__hgb_runtime"]
    out = native.search_mcts_hgb226(
        runtime,
        47,
        _state_payload(env, root_state),
        payload,
        int(iterations),
        root_player,
        int(pr.root_node_id_override or pr.root_id),
        int(root_depth),
        0,
        int(pr.root_id),
        int(seed) + int(pr.root_id),
        False,
        False,
        "",
        "",
    )

    root_visits = int(out.get("root_visits", 0) or 0)
    root_value_sum = float(out.get("root_value_sum", 0.0) or 0.0)
    children: dict[int, ChildStats] = {}
    for raw in list(out.get("children", []) or []):
        idx = int(raw.get("index", -1))
        visits = int(raw.get("visits", 0) or 0)
        value_sum = float(raw.get("value_sum", 0.0) or 0.0)
        children[idx] = ChildStats(
            visits=visits,
            value_sum=value_sum,
            mean_value=(value_sum / visits) if visits > 0 else 0.0,
            reward=float(raw.get("reward", 0.0) or 0.0),
            discount=float(raw.get("discount", raw.get("edge_discount", 1.0)) or 1.0),
            state_cost=float(raw.get("state_cost", 0.0) or 0.0),
            sim_time=float(raw.get("sim_time", 0.0) or 0.0),
        )
    return SearchStats(
        root_visits=root_visits,
        root_value_sum=root_value_sum,
        root_mean_value=(root_value_sum / root_visits) if root_visits > 0 else 0.0,
        best_action_index=int(out.get("best_action_index", -1)),
        children=children,
    )


def _same(a: SearchStats, b: SearchStats, *, tol: float = 1e-12) -> bool:
    if a.root_visits != b.root_visits:
        return False
    if abs(a.root_value_sum - b.root_value_sum) > tol:
        return False
    if set(a.children) != set(b.children):
        return False
    for idx in sorted(a.children):
        ca = a.children[idx]
        cb = b.children[idx]
        if ca.visits != cb.visits:
            return False
        for name in ("value_sum", "mean_value", "reward", "discount", "state_cost", "sim_time"):
            if abs(float(getattr(ca, name)) - float(getattr(cb, name))) > tol:
                return False
    return True


def _diff_summary(a: SearchStats, b: SearchStats) -> list[tuple[float, int, ChildStats | None, ChildStats | None]]:
    rows = []
    for idx in sorted(set(a.children) | set(b.children)):
        ca = a.children.get(idx)
        cb = b.children.get(idx)
        if ca is None or cb is None:
            rows.append((float("inf"), idx, ca, cb))
            continue
        score = abs(ca.visits - cb.visits) + abs(ca.value_sum - cb.value_sum)
        rows.append((score, idx, ca, cb))
    rows.sort(reverse=True, key=lambda x: x[0])
    return rows


def _delta(prev: SearchStats, cur: SearchStats) -> tuple[int, float]:
    changed = []
    for idx in sorted(set(prev.children) | set(cur.children)):
        a = prev.children.get(idx)
        b = cur.children.get(idx)
        av = a.visits if a is not None else 0
        bv = b.visits if b is not None else 0
        if av != bv:
            aval = a.value_sum if a is not None else 0.0
            bval = b.value_sum if b is not None else 0.0
            changed.append((idx, bval - aval))
    if len(changed) != 1:
        return -999, float("nan")
    return changed[0]


def main() -> None:
    native, args, cfg_python, env, runner, seed, selected, payload, _simulator, hgb_runtime, model_version = _prepare_root()
    payload["__hgb_runtime"] = hgb_runtime
    pr, root_state, root_player, root_depth, valid_count = selected
    print(
        f"HGB root_id={int(pr.root_id)} player={root_player} depth={root_depth} "
        f"valid_count={valid_count} seed={int(args.history_seed) + int(pr.root_id)}"
    )

    cache: dict[int, tuple[SearchStats, SearchStats]] = {}

    def run(n: int) -> tuple[SearchStats, SearchStats]:
        if n not in cache:
            py = _run_python(
                cfg_python=cfg_python,
                env=env,
                runner=runner,
                args=args,
                pr=pr,
                root_state=root_state,
                root_player=root_player,
                root_depth=int(root_depth),
                iterations=n,
            )
            nat = _run_native(
                native=native,
                env=env,
                payload=payload,
                pr=pr,
                root_state=root_state,
                root_player=root_player,
                root_depth=int(root_depth),
                iterations=n,
                seed=seed,
            )
            cache[n] = (py, nat)
        return cache[n]

    checkpoints = [300, 500, 1000, 1500, 2000, 5000, 10000]
    for n in checkpoints:
        py, nat = run(n)
        print(
            f"n={n:5d} same={_same(py, nat)} "
            f"py_root={py.root_mean_value:.15f} native_root={nat.root_mean_value:.15f} "
            f"diff={abs(py.root_mean_value - nat.root_mean_value):.15f} "
            f"py_best={py.best_action_index} native_best={nat.best_action_index}"
        )

    lo = 0
    hi = 10000
    if _same(*run(hi)):
        print("no divergence by 10000")
        return
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _same(*run(mid)):
            lo = mid
        else:
            hi = mid

    print(f"first_divergent_iteration={hi} last_matching_iteration={lo}")
    py_prev, nat_prev = run(lo)
    py_cur, nat_cur = run(hi)
    py_delta_idx, py_delta_val = _delta(py_prev, py_cur)
    nat_delta_idx, nat_delta_val = _delta(nat_prev, nat_cur)
    print(f"python_delta_child={py_delta_idx} python_delta_value_sum={py_delta_val:.17g}")
    print(f"native_delta_child={nat_delta_idx} native_delta_value_sum={nat_delta_val:.17g}")
    print(
        f"root_before py={py_prev.root_mean_value:.17g} native={nat_prev.root_mean_value:.17g} "
        f"after py={py_cur.root_mean_value:.17g} native={nat_cur.root_mean_value:.17g}"
    )

    print("top_child_diffs_at_first_divergence:")
    for score, idx, py_child, nat_child in _diff_summary(py_cur, nat_cur)[:12]:
        print(
            f"  idx={idx} score={score:.17g} "
            f"py={py_child} native={nat_child}"
        )


if __name__ == "__main__":
    main()
