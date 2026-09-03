
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import joblib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import import_native_cpp, make_args, prepare_python_roots
from state_inference_allignment_test import DEFAULT_HGB_MODEL_PATH, _export_hgb_to_native_text
from vidur.Game_Version3.DNN import infer as infer_module
from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner
from vidur.Game_Version3.mcts import MCTSConfig, VidurMCTS
from vidur.Game_Version3.tests import native_logger_tests as nlt

TARGET_ROOT_ID = int(os.environ.get("MCTS_LEAF_DEBUG_ROOT_ID", "8"))
TARGET_CALL = int(os.environ.get("MCTS_LEAF_DEBUG_CALL", "827"))
ITERATIONS = int(os.environ.get("MCTS_LEAF_DEBUG_ITERATIONS", str(TARGET_CALL)))

def main() -> None:
    native = import_native_cpp(force_build=False)
    args = make_args(
        "mcts_alignment",
        num_roots=32,
        history_hops_min=0,
        history_hops_max=100,
        history_seed=202613,
        frontier_parity_roots=1,
        bellman_q_tolerance=1e-4,
        model_version=47,
        checkpoint_path=str(DEFAULT_HGB_MODEL_PATH),
    )
    cfg_python, simulator, env, explore_cfg, python_roots = prepare_python_roots(args)
    infer_module.enable_inputs_extras()
    hgb_model = joblib.load(DEFAULT_HGB_MODEL_PATH)

    payload = _cfg_payload(cfg_python, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = True
    payload["native_search_mode"] = "full_tree"
    payload["root_dirichlet_noise_enabled"] = False
    payload["pb_c_base"] = float(cfg_python.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(cfg_python.game_v2.mcts_search.pb_c_init)
    payload["max_forced_hops"] = int(cfg_python.max_forced_hops_per_root)

    native_export_path = _export_hgb_to_native_text(
        hgb_model,
        Path(args.output_dir) / "v4_adv_hgb_native_export.tsv",
    )
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(native_export_path))

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
        if int(pr.root_id) == TARGET_ROOT_ID:
            selected = (pr, root_state, root_player, root_depth)
            break
    if selected is None:
        raise RuntimeError("target root missing")
    pr, root_state, root_player, root_depth = selected

    full_cfg = MCTSConfig()
    full_cfg.rng = random.Random(int(args.history_seed) + int(pr.root_id))
    full_cfg.num_simulations = ITERATIONS
    full_cfg.mcts_iterations = ITERATIONS
    full_cfg.uct_c = float(cfg_python.game_v2.mcts_search.pb_c_init)
    full_cfg.discount_factor = float(cfg_python.game_v2.mcts_search.discount_factor)
    full_cfg._discount_time_denom = float(
        cfg_python.game_v2.mcts_search.discount_time_denominator_sec or 0.015725797204323228
    )
    full_cfg.log_flag = False

    py_mcts = VidurMCTS(env=env, mctsConfig=full_cfg)
    original_rollout = py_mcts._rollout_value
    captured = {}
    counter = {"n": 0}

    def wrapped_rollout(state, player, **kwargs):
        counter["n"] += 1
        value = original_rollout(state, player, **kwargs)
        if counter["n"] == TARGET_CALL:
            captured["state"] = state.fork(flag=False)
            captured["player"] = player
            captured["value"] = float(value)
            captured["sim_time"] = float(state.simulator._time)
            captured["active"] = list(getattr(state.stats, "active_request_ids", []))
            captured["violated"] = list(getattr(state.stats, "violated_request_ids", []))
        return value

    py_mcts._rollout_value = wrapped_rollout
    py_mcts.search_dnn(
        dnn_model=hgb_model,
        rootState=root_state.fork(flag=False),
        root_player=root_player,
        game_id=0,
        root_id=int(pr.root_id),
        root_node_id_override=pr.root_node_id_override,
        root_depth=int(root_depth),
        mcts_iter=ITERATIONS,
        model_version=47,
        use_model_bootstrap=True,
        one_step_value_mode=False,
    )
    if not captured:
        raise RuntimeError("leaf not captured")

    payload_state = nlt._native_state_payload(env, captured["state"])
    native_dbg = runtime.infer_from_state(payload_state, payload, -1)

    from vidur.Game_Version3.DNN import infer as infer_module2
    inputs = infer_module2.build_model_inputs(
        state=captured["state"],
        player=captured["player"],
        device=torch.device("cpu"),
        build_action_mask_flag=False,
    )
    py_value_direct, _ = hgb_model.infer_from_inputs(inputs, captured["player"])
    py_value_direct = float(py_value_direct)

    if not inputs.extras:
        raise RuntimeError("inputs extras missing")
    py_features = [float(x) for x in hgb_model._features_from_extras(inputs.extras).reshape(-1).tolist()]
    native_features = [float(x) for x in native_dbg["features"]]
    diffs = [(abs(a-b), i, a, b) for i, (a,b) in enumerate(zip(py_features, native_features))]
    diffs.sort(reverse=True)

    decode_reqs = []
    for rid, req in payload_state.get("request_states", {}).items():
        if req.get("is_prefill_complete") and not req.get("completed"):
            decode_reqs.append((int(rid), int(req.get("num_processed_tokens", 0)), int(req.get("num_prefill_tokens", 0))))

    print("root", int(pr.root_id), root_player, "call", counter["n"])
    print("leaf_player", captured["player"], "sim_time", captured["sim_time"])
    print("active", captured["active"])
    print("violated", captured["violated"])
    print("decode_reqs", sorted(decode_reqs))
    print("python_rollout_value", captured["value"])
    print("python_direct_value", py_value_direct)
    print("native_value", float(native_dbg["value"]))
    print("native_raw", float(native_dbg["raw_value"]))
    print("max_feature_diff", diffs[0])
    print("top_feature_diffs")
    for row in diffs[:20]:
        if row[0] <= 0:
            break
        print(row)

if __name__ == "__main__":
    main()
