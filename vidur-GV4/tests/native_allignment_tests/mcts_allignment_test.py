from __future__ import annotations

import csv
import json
from pathlib import Path

from common import import_native_cpp, make_args, prepare_python_roots


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_zero_bootstrap_native_root_values_align_with_python_depth1() -> None:
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
    from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
    from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner, SingleRootRun
    import torch

    native = import_native_cpp()
    args = make_args(
        "mcts_alignment",
        num_roots=10,
        history_hops_min=0,
        history_hops_max=100,
        history_seed=202613,
        frontier_parity_roots=1,
        bellman_q_tolerance=1e-4,
        model_version=0,
    )
    cfg_python, simulator, env, explore_cfg, python_roots = prepare_python_roots(args)

    payload = _cfg_payload(cfg_python, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = False
    runtime = native.NativeTorchScriptInferRuntimeGV2("cpu")

    runner = SelfPlayRunner(
        env=env,
        mcts=nlt.VidurMCTS(env=env, explore_cfg=explore_cfg),
        model=nlt.AlphaZeroModel(spec=nlt.make_dnn_spec(cfg=cfg_python.game_v2)).to(torch.device("cpu")),
        writer=NoopReplayWriter(),
        device_for_features=torch.device("cpu"),
        game_v2_cfg=cfg_python.game_v2,
    )

    rows = []
    child_rows = []
    failures = []
    for pr in python_roots[:10]:
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

        py_result = runner.mcts.search_dnn(
            dnn_model=runner.model,
            rootState=root_state.fork(flag=False),
            root_player=root_player,
            game_id=0,
            root_id=int(pr.root_id),
            root_node_id_override=pr.root_node_id_override,
            root_depth=int(root_depth),
            model_version=0,
            use_model_bootstrap=False,
            one_step_value_mode=True,
        )
        native_out = native.search_mcts_dnn_torchscript(
            runtime,
            0,
            nlt._native_state_payload(env, root_state),
            payload,
            1,
            root_player,
            int(pr.root_node_id_override or pr.root_id),
            int(root_depth),
            0,
            int(pr.root_id),
            int(args.history_seed) + int(pr.root_id),
            False,
            False,
            "",
            "",
        )

        py_best = int(py_result.best_action_index if py_result.best_action_index is not None else -1)
        native_best = int(native_out.get("best_action_index", -1))
        py_value = float(py_result.best_action_value)
        py_action_values = [float(x) for x in list(py_result.action_values or [])]
        native_visits = int(native_out.get("root_visits", 0) or 0)
        native_value = (float(native_out.get("root_value_sum", 0.0)) / float(native_visits)) if native_visits > 0 else 0.0
        native_action_values = [float(x) for x in list(native_out.get("root_action_values", []) or [])]
        native_best_action_value = (
            native_action_values[native_best]
            if 0 <= native_best < len(native_action_values)
            else float("nan")
        )
        native_child_visits = {
            int(c.get("index", -1)): int(c.get("visits", 0) or 0)
            for c in list(native_out.get("children", []) or [])
            if int(c.get("index", -1)) >= 0
        }
        native_child_value_sum = {
            int(c.get("index", -1)): float(c.get("value_sum", 0.0) or 0.0)
            for c in list(native_out.get("children", []) or [])
            if int(c.get("index", -1)) >= 0
        }
        max_child_n = max(len(py_action_values), len(native_action_values))
        for child_idx in range(max_child_n):
            py_child_value = py_action_values[child_idx] if child_idx < len(py_action_values) else float("nan")
            native_child_value = native_action_values[child_idx] if child_idx < len(native_action_values) else float("nan")
            n_visits = int(native_child_visits.get(child_idx, 0))
            native_child_mean = (
                float(native_child_value_sum.get(child_idx, 0.0)) / float(n_visits)
                if n_visits > 0
                else 0.0
            )
            child_rows.append(
                {
                    "id": int(pr.root_id),
                    "root_player": root_player,
                    "python_root_value": py_value,
                    "native_root_value": native_value,
                    "child_index": int(child_idx),
                    "python_child_value": py_child_value,
                    "native_child_value": native_child_value,
                    "python_child_visits": 1 if child_idx == py_best else 0,
                    "native_child_visits": n_visits,
                    "native_child_mean_value": native_child_mean,
                    "python_selected": bool(child_idx == py_best),
                    "native_selected": bool(child_idx == native_best),
                    "value_abs_diff": abs(float(py_child_value) - float(native_child_value))
                    if py_child_value == py_child_value and native_child_value == native_child_value
                    else float("nan"),
                }
            )
        py_selected_native_value = (
            native_action_values[py_best]
            if 0 <= py_best < len(native_action_values)
            else float("nan")
        )
        value_diff = abs(py_value - native_value)
        tie_equivalent = (
            py_best == native_best
            or abs(float(py_selected_native_value) - float(native_best_action_value)) <= float(args.bellman_q_tolerance)
        )
        passed = bool(tie_equivalent and value_diff <= float(args.bellman_q_tolerance))
        if not passed:
            failures.append(int(pr.root_id))
        rows.append(
            {
                "root_id": int(pr.root_id),
                "root_player": root_player,
                "history_hops": int(pr.history_hops),
                "python_best_index": py_best,
                "native_best_index": native_best,
                "python_root_value": py_value,
                "native_root_value": native_value,
                "native_best_action_value": native_best_action_value,
                "py_selected_native_value": py_selected_native_value,
                "abs_diff": value_diff,
                "tie_equivalent": tie_equivalent,
                "native_root_action_values": json.dumps(native_action_values),
                "native_mcts_root_prior": json.dumps([float(x) for x in list(native_out.get("mcts_root_prior", []) or [])]),
                "native_keys": json.dumps(sorted(native_out.keys())),
                "passed": passed,
            }
        )
        if hasattr(runner.mcts, "clear_search_state"):
            runner.mcts.clear_search_state(drop_scratch=False)

    _write(Path(args.output_dir) / "zero_bootstrap_root_value_alignment.csv", rows)
    _write(Path(args.output_dir) / "zero_bootstrap_child_value_alignment.csv", child_rows)
    assert not failures, f"native zero-bootstrap root-value alignment failed for roots={failures[:20]}"



def test_full_root_mcts_10000_child_visit_distribution_csv() -> None:
    """Run a real root MCTS search and log child visit distributions.

    Native full-tree mode is expected to mirror plain Python `mcts.py` semantics:
    UCT selection, one random untried child expansion per simulation, raw
    Bellman rewards, and child mean values as root action values.  Exact visit
    equality still depends on RNG stream parity.
    """
    import math
    import random
    import torch

    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
    from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
    from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner
    from vidur.Game_Version3.mcts import VidurMCTS as FullTreeMCTS, MCTSConfig as FullTreeMCTSConfig

    native = import_native_cpp(force_build=True)
    iterations = 10000
    args = make_args(
        "mcts_alignment",
        num_roots=8,
        history_hops_min=0,
        history_hops_max=100,
        history_seed=202613,
        frontier_parity_roots=1,
        bellman_q_tolerance=1e-4,
        model_version=0,
    )
    cfg_python, simulator, env, explore_cfg, python_roots = prepare_python_roots(args)

    payload = _cfg_payload(cfg_python, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = False
    payload["native_search_mode"] = "full_tree"
    payload["root_dirichlet_noise_enabled"] = False
    payload["pb_c_base"] = float(cfg_python.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["max_forced_hops"] = int(cfg_python.max_forced_hops_per_root)

    runtime = native.NativeTorchScriptInferRuntimeGV2("cpu")
    runner = SelfPlayRunner(
        env=env,
        mcts=nlt.VidurMCTS(env=env, explore_cfg=explore_cfg),
        model=nlt.AlphaZeroModel(spec=nlt.make_dnn_spec(cfg=cfg_python.game_v2)).to(torch.device("cpu")),
        writer=NoopReplayWriter(),
        device_for_features=torch.device("cpu"),
        game_v2_cfg=cfg_python.game_v2,
    )

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
        if valid_count > 1:
            selected = (pr, root_state, root_player, root_depth, valid_count)
            break

    assert selected is not None, "no multi-action root found for full-root MCTS visit distribution test"
    pr, root_state, root_player, root_depth, valid_count = selected

    full_cfg = FullTreeMCTSConfig()
    full_cfg.rng = random.Random(int(args.history_seed) + int(pr.root_id))
    full_cfg.num_simulations = iterations
    full_cfg.mcts_iterations = iterations
    full_cfg.uct_c = float(cfg_python.game_v2.mcts_search.pb_c_init)
    full_cfg.discount_factor = float(cfg_python.game_v2.mcts_search.discount_factor)
    full_cfg._discount_time_denom = float(
        cfg_python.game_v2.mcts_search.discount_time_denominator_sec or 0.015725797204323228
    )
    full_cfg.log_flag = False

    py_mcts = FullTreeMCTS(env=env, mctsConfig=full_cfg)
    py_result = py_mcts.search_dnn(
        dnn_model=runner.model,
        rootState=root_state.fork(flag=False),
        root_player=root_player,
        game_id=0,
        root_id=int(pr.root_id),
        root_node_id_override=pr.root_node_id_override,
        root_depth=int(root_depth),
        mcts_iter=iterations,
        model_version=0,
        use_model_bootstrap=False,
        one_step_value_mode=False,
    )
    py_root = py_mcts._root
    assert py_root is not None

    native_out = native.search_mcts_dnn_torchscript(
        runtime,
        0,
        nlt._native_state_payload(env, root_state),
        payload,
        iterations,
        root_player,
        int(pr.root_node_id_override or pr.root_id),
        int(root_depth),
        0,
        int(pr.root_id),
        int(args.history_seed) + int(pr.root_id),
        False,
        False,
        "",
        "",
    )

    py_action_values = [float(x) for x in list(py_result.action_values or [])]
    native_action_values = [float(x) for x in list(native_out.get("root_action_values", []) or [])]
    py_alias_to_canon = {int(k): int(v) for k, v in (py_root.action_alias_to_canonical or {}).items()}
    native_alias_to_canon = {
        int(k): int(v) for k, v in dict(native_out.get("action_alias_to_canonical", {}) or {}).items()
    }
    native_child_by_index = {
        int(c.get("index", -1)): c
        for c in list(native_out.get("children", []) or [])
        if int(c.get("index", -1)) >= 0
    }

    py_root_value = float(py_root.mean_value())
    native_root_visits = int(native_out.get("root_visits", 0) or 0)
    native_root_value = (
        float(native_out.get("root_value_sum", 0.0)) / float(native_root_visits)
        if native_root_visits > 0
        else 0.0
    )

    child_rows = []
    max_child_n = max(len(py_action_values), len(native_action_values))
    for child_idx in range(max_child_n):
        py_canon = int(py_alias_to_canon.get(child_idx, child_idx))
        native_canon = int(native_alias_to_canon.get(child_idx, child_idx))
        py_child = py_root.children.get(py_canon)
        native_child = native_child_by_index.get(native_canon)
        py_visits = int(py_child.visits) if py_child is not None else 0
        native_visits = int(native_child.get("visits", 0) or 0) if native_child is not None else 0
        py_value = py_action_values[child_idx] if child_idx < len(py_action_values) else float("nan")
        native_value = native_action_values[child_idx] if child_idx < len(native_action_values) else float("nan")
        child_rows.append(
            {
                "id": int(pr.root_id),
                "root_player": root_player,
                "history_hops": int(pr.history_hops),
                "root_valid_action_count": int(valid_count),
                "iterations": int(iterations),
                "python_root_value": py_root_value,
                "native_root_value": native_root_value,
                "python_root_visits": int(py_root.visits),
                "native_root_visits": native_root_visits,
                "child_index": int(child_idx),
                "python_canonical_child_index": int(py_canon),
                "native_canonical_child_index": int(native_canon),
                "python_child_value": py_value,
                "native_child_value": native_value,
                "python_child_visits": py_visits,
                "native_child_visits": native_visits,
                "python_child_raw_mean_value": float(py_child.mean_value()) if py_child is not None else float("nan"),
                "native_child_raw_mean_value": (
                    float(native_child.get("value_sum", 0.0) or 0.0) / float(native_visits)
                    if native_child is not None and native_visits > 0
                    else float("nan")
                ),
                "python_selected": bool(child_idx == int(py_result.best_action_index or -1)),
                "native_selected": bool(child_idx == int(native_out.get("best_action_index", -1))),
                "value_abs_diff": abs(py_value - native_value)
                if math.isfinite(py_value) and math.isfinite(native_value)
                else float("nan"),
            }
        )

    _write(Path(args.output_dir) / "full_mcts_10000_child_visit_alignment.csv", child_rows)
    assert int(py_root.visits) == iterations, f"python root visits={py_root.visits}, expected={iterations}"
    assert native_root_visits == iterations, f"native root visits={native_root_visits}, expected={iterations}"
    assert sum(1 for c in py_root.children.values() if int(c.visits) > 0) > 1, "python visits did not distribute"
    assert sum(1 for c in native_child_by_index.values() if int(c.get("visits", 0) or 0) > 0) > 1, "native visits did not distribute"



def test_full_root_mcts_10000_hgb226_bootstrap_child_visit_distribution_csv() -> None:
    """Run full-root MCTS with the default v4-Adv HGB226 value bootstrap.

    This mirrors ``test_full_root_mcts_10000_child_visit_distribution_csv`` but
    enables model bootstrap on both sides:
      - Python uses the joblib V4AdvHGBWrapper via ``infer_from_inputs``.
      - Native uses the exported TSV HGB tree runtime via ``search_mcts_hgb226``.

    The output CSV intentionally has the same child-by-child schema as the
    zero-bootstrap 10k test so visit/value drift is easy to inspect.
    """
    import math
    import random

    import joblib
    import torch

    from state_inference_allignment_test import DEFAULT_HGB_MODEL_PATH, _export_hgb_to_native_text
    from vidur.Game_Version3.DNN import infer as infer_module
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
    from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
    from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner
    from vidur.Game_Version3.mcts import VidurMCTS as FullTreeMCTS, MCTSConfig as FullTreeMCTSConfig

    native = import_native_cpp(force_build=True)
    iterations = 10000
    model_version = 47
    args = make_args(
        "mcts_alignment",
        num_roots=8,
        history_hops_min=0,
        history_hops_max=100,
        history_seed=202613,
        frontier_parity_roots=1,
        bellman_q_tolerance=1e-4,
        model_version=model_version,
        checkpoint_path=str(DEFAULT_HGB_MODEL_PATH),
    )
    cfg_python, simulator, env, explore_cfg, python_roots = prepare_python_roots(args)

    if not DEFAULT_HGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"default HGB226 model not found: {DEFAULT_HGB_MODEL_PATH}")

    # Required for V4AdvHGBWrapper: it derives its 226D feature vector from
    # ModelInputs.extras, which build_model_inputs attaches only after opt-in.
    infer_module.enable_inputs_extras()
    hgb_model = joblib.load(DEFAULT_HGB_MODEL_PATH)
    if not callable(getattr(hgb_model, "infer_from_inputs", None)):
        raise TypeError(f"expected HGB wrapper with infer_from_inputs, got {type(hgb_model)!r}")

    out_dir = Path(args.output_dir)
    native_export_path = _export_hgb_to_native_text(
        hgb_model,
        out_dir / "v4_adv_hgb_native_export.tsv",
    )
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(native_export_path))

    payload = _cfg_payload(cfg_python, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = True
    payload["native_search_mode"] = "full_tree"
    payload["root_dirichlet_noise_enabled"] = False
    payload["pb_c_base"] = float(cfg_python.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["max_forced_hops"] = int(cfg_python.max_forced_hops_per_root)

    runner = SelfPlayRunner(
        env=env,
        mcts=nlt.VidurMCTS(env=env, explore_cfg=explore_cfg),
        model=hgb_model,
        writer=NoopReplayWriter(),
        device_for_features=torch.device("cpu"),
        game_v2_cfg=cfg_python.game_v2,
    )

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
        if valid_count > 1:
            selected = (pr, root_state, root_player, root_depth, valid_count)
            break

    assert selected is not None, "no multi-action root found for HGB226 bootstrap MCTS visit distribution test"
    pr, root_state, root_player, root_depth, valid_count = selected

    full_cfg = FullTreeMCTSConfig()
    full_cfg.rng = random.Random(int(args.history_seed) + int(pr.root_id))
    full_cfg.num_simulations = iterations
    full_cfg.mcts_iterations = iterations
    full_cfg.uct_c = float(cfg_python.game_v2.mcts_search.pb_c_init)
    full_cfg.discount_factor = float(cfg_python.game_v2.mcts_search.discount_factor)
    full_cfg._discount_time_denom = float(
        cfg_python.game_v2.mcts_search.discount_time_denominator_sec or 0.015725797204323228
    )
    full_cfg.log_flag = False

    py_mcts = FullTreeMCTS(env=env, mctsConfig=full_cfg)
    py_result = py_mcts.search_dnn(
        dnn_model=hgb_model,
        rootState=root_state.fork(flag=False),
        root_player=root_player,
        game_id=0,
        root_id=int(pr.root_id),
        root_node_id_override=pr.root_node_id_override,
        root_depth=int(root_depth),
        mcts_iter=iterations,
        model_version=model_version,
        use_model_bootstrap=True,
        one_step_value_mode=False,
    )
    py_root = py_mcts._root
    assert py_root is not None

    native_out = native.search_mcts_hgb226(
        runtime,
        model_version,
        nlt._native_state_payload(env, root_state),
        payload,
        iterations,
        root_player,
        int(pr.root_node_id_override or pr.root_id),
        int(root_depth),
        0,
        int(pr.root_id),
        int(args.history_seed) + int(pr.root_id),
        False,
        False,
        "",
        "",
    )

    py_action_values = [float(x) for x in list(py_result.action_values or [])]
    native_action_values = [float(x) for x in list(native_out.get("root_action_values", []) or [])]
    py_alias_to_canon = {int(k): int(v) for k, v in (py_root.action_alias_to_canonical or {}).items()}
    native_alias_to_canon = {
        int(k): int(v) for k, v in dict(native_out.get("action_alias_to_canonical", {}) or {}).items()
    }
    native_child_by_index = {
        int(c.get("index", -1)): c
        for c in list(native_out.get("children", []) or [])
        if int(c.get("index", -1)) >= 0
    }

    py_root_value = float(py_root.mean_value())
    native_root_visits = int(native_out.get("root_visits", 0) or 0)
    native_root_value = (
        float(native_out.get("root_value_sum", 0.0)) / float(native_root_visits)
        if native_root_visits > 0
        else 0.0
    )

    child_rows = []
    max_child_n = max(len(py_action_values), len(native_action_values))
    for child_idx in range(max_child_n):
        py_canon = int(py_alias_to_canon.get(child_idx, child_idx))
        native_canon = int(native_alias_to_canon.get(child_idx, child_idx))
        py_child = py_root.children.get(py_canon)
        native_child = native_child_by_index.get(native_canon)
        py_visits = int(py_child.visits) if py_child is not None else 0
        native_visits = int(native_child.get("visits", 0) or 0) if native_child is not None else 0
        py_value = py_action_values[child_idx] if child_idx < len(py_action_values) else float("nan")
        native_value = native_action_values[child_idx] if child_idx < len(native_action_values) else float("nan")
        child_rows.append(
            {
                "id": int(pr.root_id),
                "root_player": root_player,
                "history_hops": int(pr.history_hops),
                "root_valid_action_count": int(valid_count),
                "iterations": int(iterations),
                "python_root_value": py_root_value,
                "native_root_value": native_root_value,
                "python_root_visits": int(py_root.visits),
                "native_root_visits": native_root_visits,
                "child_index": int(child_idx),
                "python_canonical_child_index": int(py_canon),
                "native_canonical_child_index": int(native_canon),
                "python_child_value": py_value,
                "native_child_value": native_value,
                "python_child_visits": py_visits,
                "native_child_visits": native_visits,
                "python_child_raw_mean_value": float(py_child.mean_value()) if py_child is not None else float("nan"),
                "native_child_raw_mean_value": (
                    float(native_child.get("value_sum", 0.0) or 0.0) / float(native_visits)
                    if native_child is not None and native_visits > 0
                    else float("nan")
                ),
                "python_selected": bool(child_idx == int(py_result.best_action_index or -1)),
                "native_selected": bool(child_idx == int(native_out.get("best_action_index", -1))),
                "value_abs_diff": abs(py_value - native_value)
                if math.isfinite(py_value) and math.isfinite(native_value)
                else float("nan"),
            }
        )

    _write(out_dir / "full_mcts_10000_hgb226_child_visit_alignment.csv", child_rows)
    assert int(py_root.visits) == iterations, f"python root visits={py_root.visits}, expected={iterations}"
    assert native_root_visits == iterations, f"native root visits={native_root_visits}, expected={iterations}"
    assert sum(1 for c in py_root.children.values() if int(c.visits) > 0) > 1, "python visits did not distribute"
    assert sum(1 for c in native_child_by_index.values() if int(c.get("visits", 0) or 0) > 0) > 1, "native visits did not distribute"



def test_mcts_overall_root_alignment_hgb226_csv() -> None:
    """Run HGB226-bootstrap MCTS on 10 roots and write root-level timing/value alignment."""
    import math
    import os
    import random
    import time

    import joblib
    import torch

    from state_inference_allignment_test import DEFAULT_HGB_MODEL_PATH, _export_hgb_to_native_text
    from vidur.Game_Version3.DNN import infer as infer_module
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
    from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
    from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner
    from vidur.Game_Version3.mcts import VidurMCTS as FullTreeMCTS, MCTSConfig as FullTreeMCTSConfig

    native = import_native_cpp(force_build=False)
    iterations = int(os.environ.get("MCTS_ALIGNMENT_OVERALL_ITERATIONS", "10000"))
    model_version = 47
    target_roots = int(os.environ.get("MCTS_ALIGNMENT_OVERALL_ROOTS", "10"))
    history_seed = int(os.environ.get("MCTS_ALIGNMENT_OVERALL_HISTORY_SEED", "202613"))
    source_roots = int(os.environ.get("MCTS_ALIGNMENT_OVERALL_SOURCE_ROOTS", str(max(32, target_roots * 4))))
    output_name = os.environ.get("MCTS_ALIGNMENT_OVERALL_OUTPUT", "mcts_overall_root_allignment.csv")
    args = make_args(
        "mcts_alignment",
        num_roots=source_roots,
        history_hops_min=0,
        history_hops_max=100,
        history_seed=history_seed,
        frontier_parity_roots=1,
        bellman_q_tolerance=1e-4,
        model_version=model_version,
        checkpoint_path=str(DEFAULT_HGB_MODEL_PATH),
    )
    cfg_python, simulator, env, explore_cfg, python_roots = prepare_python_roots(args)

    if not DEFAULT_HGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"default HGB226 model not found: {DEFAULT_HGB_MODEL_PATH}")

    infer_module.enable_inputs_extras()
    hgb_model = joblib.load(DEFAULT_HGB_MODEL_PATH)
    if not callable(getattr(hgb_model, "infer_from_inputs", None)):
        raise TypeError(f"expected HGB wrapper with infer_from_inputs, got {type(hgb_model)!r}")

    out_dir = Path(args.output_dir)
    native_export_path = _export_hgb_to_native_text(
        hgb_model,
        out_dir / "v4_adv_hgb_native_export.tsv",
    )
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(native_export_path))

    payload = _cfg_payload(cfg_python, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = True
    payload["native_search_mode"] = "full_tree"
    payload["root_dirichlet_noise_enabled"] = False
    payload["pb_c_base"] = float(cfg_python.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["max_forced_hops"] = int(cfg_python.max_forced_hops_per_root)

    runner = SelfPlayRunner(
        env=env,
        mcts=nlt.VidurMCTS(env=env, explore_cfg=explore_cfg),
        model=hgb_model,
        writer=NoopReplayWriter(),
        device_for_features=torch.device("cpu"),
        game_v2_cfg=cfg_python.game_v2,
    )

    selected_roots = []
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
        if valid_count > 1:
            selected_roots.append((pr, root_state, root_player, root_depth, valid_count))
        if len(selected_roots) >= target_roots:
            break

    assert len(selected_roots) == target_roots, f"found only {len(selected_roots)} multi-action roots"

    rows = []
    for pr, root_state, root_player, root_depth, _valid_count in selected_roots:
        full_cfg = FullTreeMCTSConfig()
        full_cfg.rng = random.Random(int(args.history_seed) + int(pr.root_id))
        full_cfg.num_simulations = iterations
        full_cfg.mcts_iterations = iterations
        full_cfg.uct_c = float(cfg_python.game_v2.mcts_search.pb_c_init)
        full_cfg.discount_factor = float(cfg_python.game_v2.mcts_search.discount_factor)
        full_cfg._discount_time_denom = float(
            cfg_python.game_v2.mcts_search.discount_time_denominator_sec or 0.015725797204323228
        )
        full_cfg.log_flag = False

        py_mcts = FullTreeMCTS(env=env, mctsConfig=full_cfg)
        py_start = time.perf_counter()
        py_result = py_mcts.search_dnn(
            dnn_model=hgb_model,
            rootState=root_state.fork(flag=False),
            root_player=root_player,
            game_id=0,
            root_id=int(pr.root_id),
            root_node_id_override=pr.root_node_id_override,
            root_depth=int(root_depth),
            mcts_iter=iterations,
            model_version=model_version,
            use_model_bootstrap=True,
            one_step_value_mode=False,
        )
        py_time = time.perf_counter() - py_start
        py_root = py_mcts._root
        assert py_root is not None

        native_start = time.perf_counter()
        native_out = native.search_mcts_hgb226(
            runtime,
            model_version,
            nlt._native_state_payload(env, root_state),
            payload,
            iterations,
            root_player,
            int(pr.root_node_id_override or pr.root_id),
            int(root_depth),
            0,
            int(pr.root_id),
            int(args.history_seed) + int(pr.root_id),
            False,
            False,
            "",
            "",
        )
        native_time = time.perf_counter() - native_start

        py_root_value = float(py_root.mean_value())
        native_root_visits = int(native_out.get("root_visits", 0) or 0)
        native_root_value = (
            float(native_out.get("root_value_sum", 0.0)) / float(native_root_visits)
            if native_root_visits > 0
            else 0.0
        )
        python_best_idx = int(py_result.best_action_index) if py_result.best_action_index is not None else -1
        native_best_idx = int(native_out.get("best_action_index", -1))
        best_canon = int((py_root.action_alias_to_canonical or {}).get(python_best_idx, python_best_idx))
        best_child = py_root.children.get(best_canon)
        best_reward = float(getattr(best_child, "reward", float("nan"))) if best_child is not None else float("nan")
        best_bootstrap = float(best_child.mean_value()) if best_child is not None else float("nan")

        rows.append(
            {
                "root_id": int(pr.root_id),
                "root_player": root_player,
                "mcts_iterations": int(iterations),
                "python_root_value": py_root_value,
                "native_root_value": native_root_value,
                "abs_difference": abs(py_root_value - native_root_value),
                "best_action_index": python_best_idx,
                "python_best_action_index": python_best_idx,
                "native_best_action_index": native_best_idx,
                "best_action_index_matches": bool(python_best_idx == native_best_idx),
                "best_action_reward": best_reward,
                "best_action_bootstrap": best_bootstrap,
                "mcts_total_iteration_time_native": float(native_time),
                "mcts_total_iteration_time_python": float(py_time),
            }
        )
        if not math.isfinite(py_root_value) or not math.isfinite(native_root_value):
            raise AssertionError(f"non-finite root value for root_id={pr.root_id}")
        assert int(py_root.visits) == iterations, f"python root visits={py_root.visits}, expected={iterations}"
        assert native_root_visits == iterations, f"native root visits={native_root_visits}, expected={iterations}"

    _write(out_dir / output_name, rows)


if __name__ == "__main__":
    test_zero_bootstrap_native_root_values_align_with_python_depth1()
    test_full_root_mcts_10000_child_visit_distribution_csv()
    test_full_root_mcts_10000_hgb226_bootstrap_child_visit_distribution_csv()
    test_mcts_overall_root_alignment_hgb226_csv()
    print("mcts_allignment_test passed")
