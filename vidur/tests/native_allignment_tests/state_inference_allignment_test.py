from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from common import REPO_ROOT, import_native_cpp, make_args, prepare_python_roots


DEFAULT_NUM_STATES = 100
DEFAULT_HISTORY_HOPS_MIN = 0
DEFAULT_HISTORY_HOPS_MAX = 100
DEFAULT_HISTORY_SEED = 202617
DEFAULT_VALUE_TOLERANCE = 1e-5
DEFAULT_POLICY_TOLERANCE = 1e-6
DEFAULT_FEATURE_TOLERANCE = 1e-6
DEFAULT_MODEL_KIND = "v4_adv_hgb"
DEFAULT_HGB_MODEL_REL = Path(
    "simulator_output/GV3_Agent/BellmanConvergence/"
    "cached_state_local_v4_adv_226/V_iter_xl_50/hgb_sq_47leaf_850iter/"
    "Model_Version47/v4_adv_hgb_wrapper.joblib"
)
DEFAULT_HGB_MODEL_PATH = REPO_ROOT / DEFAULT_HGB_MODEL_REL


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, float):
                    value = f"{value:.12g}" if math.isfinite(value) else str(value)
                elif value is None:
                    value = ""
                out[key] = value
            writer.writerow(out)


def _flat_float_tensor(t: torch.Tensor | None) -> list[float]:
    if t is None:
        return []
    return [float(x) for x in t.detach().to("cpu", dtype=torch.float32).reshape(-1).tolist()]


def _flat_bool_tensor(t: torch.Tensor | None) -> list[bool]:
    if t is None:
        return []
    return [bool(x) for x in t.detach().to("cpu", dtype=torch.bool).reshape(-1).tolist()]


def _model_inputs_to_native_dict(inputs: Any) -> dict[str, Any]:
    prefill_req_features = inputs.prefill_req_features
    decode_req_features = inputs.decode_req_features
    req_features = inputs.req_features
    return {
        "global_features": _flat_float_tensor(inputs.global_features),
        "action_mask": _flat_bool_tensor(inputs.action_mask),
        "prefill_req_features": _flat_float_tensor(prefill_req_features),
        "decode_req_features": _flat_float_tensor(decode_req_features),
        "prefill_req_mask": _flat_bool_tensor(inputs.prefill_req_mask),
        "decode_req_mask": _flat_bool_tensor(inputs.decode_req_mask),
        "prefill_req_n": int(prefill_req_features.shape[1]),
        "prefill_req_d": int(prefill_req_features.shape[2]),
        "decode_req_n": int(decode_req_features.shape[1]),
        "decode_req_d": int(decode_req_features.shape[2]),
        "req_features": _flat_float_tensor(req_features),
        "req_mask": _flat_bool_tensor(inputs.req_mask),
        "req_n": int(req_features.shape[1]) if req_features is not None else 0,
        "req_d": int(req_features.shape[2]) if req_features is not None else 0,
    }


def _native_inputs_to_model_inputs(inputs: dict[str, Any]) -> Any:
    from vidur.Game_Version3.DNN.types import ModelInputs

    def f32(values: Any, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(list(values or []), dtype=torch.float32).reshape(*shape)

    def mask(values: Any, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor([bool(x) for x in list(values or [])], dtype=torch.bool).reshape(*shape)

    pn = int(inputs["prefill_req_n"])
    pd = int(inputs["prefill_req_d"])
    dn = int(inputs["decode_req_n"])
    dd = int(inputs["decode_req_d"])
    rn = int(inputs["req_n"])
    rd = int(inputs["req_d"])
    return ModelInputs(
        prefill_req_features=f32(inputs["prefill_req_features"], (1, pn, pd)),
        decode_req_features=f32(inputs["decode_req_features"], (1, dn, dd)),
        global_features=f32(inputs["global_features"], (1, -1)),
        prefill_req_mask=mask(inputs["prefill_req_mask"], (1, pn)),
        decode_req_mask=mask(inputs["decode_req_mask"], (1, dn)),
        action_mask=mask(inputs["action_mask"], (1, -1)),
        req_features=f32(inputs["req_features"], (1, rn, rd)),
        req_mask=mask(inputs["req_mask"], (1, rn)),
    )


def _max_abs_diff_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")
    if a.numel() == 0:
        return 0.0
    return float((a.detach().to("cpu") - b.detach().to("cpu")).abs().max().item())


def _max_abs_diff_list(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _load_or_make_python_model(cfg: Any, checkpoint_path: Path, *, seed: int, out_dir: Path) -> tuple[Any, Path]:
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    torch.manual_seed(int(seed))
    random.seed(int(seed))
    model = nlt.AlphaZeroModel(spec=nlt.make_dnn_spec(cfg=cfg.game_v2)).to(torch.device("cpu"))
    model.eval()

    if checkpoint_path.exists():
        nlt._load_weights_into_model(model, checkpoint_path)
        model.eval()
        return model, checkpoint_path

    marker = out_dir / "untrained_deterministic_model_marker.pt"
    torch.save(
        {
            "seed": int(seed),
            "state_dict": model.state_dict(),
            "note": "No checkpoint was available; this deterministic random model is used only for native/Python inference alignment.",
        },
        marker,
    )
    return model, marker


def _state_signature(env: Any, prepared_root: Any) -> str:
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    try:
        sig = nlt._python_signature(env, prepared_root)
    except Exception:
        desc = env.describe_state(prepared_root.root_state)
        sig = {
            "root_id": int(prepared_root.root_id),
            "root_player": str(prepared_root.root_player),
            "root_depth": int(prepared_root.root_depth),
            "history_hops": int(prepared_root.history_hops),
            "sim_time": desc.get("sim_time", ""),
            "active_request_ids": desc.get("active_request_ids", []),
            "completed_request_ids": desc.get("completed_request_ids", []),
        }
    return json.dumps(sig, sort_keys=True)




def test_state_local_226d_features_do_not_fallback_to_historical_requests_when_active_empty() -> None:
    """Empty active_request_ids means no in-system requests for 226D inference."""
    from types import SimpleNamespace

    from vidur.bellman_v4_adv.build_state_local_features_adv import extract_features_one_record

    record = {
        "simulator_snapshot": {
            "time": 10.0,
            "request_states": {
                "1": {
                    "id": 1,
                    "arrived_at": 0.0,
                    "num_prefill_tokens": 512,
                    "remaining_prefill_tokens": 512,
                    "num_decode_tokens": 864,
                    "num_processed_tokens": 0,
                    "prefill_slo_time": 0.1,
                    "decode_slo_time": 0.05,
                },
                "2": {
                    "id": 2,
                    "arrived_at": 0.0,
                    "num_prefill_tokens": 512,
                    "remaining_prefill_tokens": 0,
                    "num_decode_tokens": 864,
                    "num_processed_tokens": 512,
                    "num_processed_decode_tokens": 0,
                    "prefill_slo_time": 0.1,
                    "decode_slo_time": 0.05,
                },
            },
        },
        "stats": SimpleNamespace(
            active_request_ids=set(),
            violated_request_ids={1, 2},
            per_request_prefill_lateness={1: 9.9},
            per_request_decode_lateness={2: 9.0},
            decode_next_deadline_by_id={},
            decode_tokens_counted={},
        ),
    }

    feat = extract_features_one_record(record)
    assert int(feat.shape[0]) == 226
    # First six globals are active-prefill count, active-decode count, active-total
    # count, remaining prefill, remaining decode, and generated active decode.
    assert [float(x) for x in feat[:6]] == [0.0] * 6


def _resolve_model_path(path_like: str | Path | None, default_path: Path) -> Path:
    if path_like is None or str(path_like).strip() == "":
        return default_path
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _export_hgb_to_native_text(model: Any, out_path: Path) -> Path:
    """Export sklearn HistGradientBoostingRegressor internals for native C++ traversal."""
    import numpy as np

    hgb = getattr(model, "hgb", model)
    predictors = list(getattr(hgb, "_predictors", []) or [])
    if not predictors:
        raise TypeError("expected sklearn HistGradientBoostingRegressor with _predictors")
    baseline = float(np.ravel(getattr(hgb, "_baseline_prediction"))[0])
    feature_dim = int(getattr(model, "feature_dim", 226))
    model_tag = str(getattr(model, "model_tag", "v4_adv_hgb"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import os
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write("hgb226_v1\n")
        f.write(f"model_tag\t{model_tag}\n")
        f.write(f"feature_dim\t{feature_dim}\n")
        f.write(f"baseline\t{baseline:.17g}\n")
        for tree_idx, pred in enumerate(predictors):
            tree = pred[0]
            nodes = tree.nodes
            # v4 HGB models are purely numerical trees. If this ever changes,
            # native traversal must add categorical bitset support explicitly.
            if "is_categorical" in nodes.dtype.names and bool(np.any(nodes["is_categorical"])):
                raise RuntimeError("native HGB export does not support categorical splits")
            f.write(f"tree\t{tree_idx}\t{len(nodes)}\n")
            for n in nodes:
                f.write(
                    "node"
                    f"\t{float(n['value']):.17g}"
                    f"\t{int(n['feature_idx'])}"
                    f"\t{float(n['num_threshold']):.17g}"
                    f"\t{1 if bool(n['missing_go_to_left']) else 0}"
                    f"\t{int(n['left'])}"
                    f"\t{int(n['right'])}"
                    f"\t{1 if bool(n['is_leaf']) else 0}\n"
                )
            f.write("end_tree\n")
    tmp.replace(out_path)
    return out_path


def run_hgb_state_inference_alignment(
    *,
    num_states: int = DEFAULT_NUM_STATES,
    history_hops_min: int = DEFAULT_HISTORY_HOPS_MIN,
    history_hops_max: int = DEFAULT_HISTORY_HOPS_MAX,
    history_seed: int = DEFAULT_HISTORY_SEED,
    value_tolerance: float = DEFAULT_VALUE_TOLERANCE,
    feature_tolerance: float = DEFAULT_FEATURE_TOLERANCE,
    hgb_model_path: str | Path | None = None,
    force_build: bool = False,
) -> Path:
    """Validate native 226D feature extraction + native HGB traversal against Python."""
    import joblib
    import numpy as np

    from vidur.Game_Version3.DNN import infer as infer_module
    from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.Game_Version3.tests.feature_conversion_tests import _make_action_mask_fn

    model_path = _resolve_model_path(hgb_model_path, DEFAULT_HGB_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"default HGB model not found: {model_path}")

    native = import_native_cpp(force_build=bool(force_build))
    model = joblib.load(model_path)
    if not callable(getattr(model, "infer_from_inputs", None)):
        raise TypeError(f"HGB model must implement infer_from_inputs: {type(model)!r}")
    if not hasattr(model, "hgb") or not callable(getattr(getattr(model, "hgb", None), "predict", None)):
        raise TypeError(f"expected V4AdvHGBWrapper-like model with .hgb.predict: {type(model)!r}")

    infer_module.enable_inputs_extras()
    args = make_args(
        "state_inference_alignment",
        num_roots=int(num_states),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        history_seed=int(history_seed),
        feature_tolerance=float(feature_tolerance),
        model_version=47,
        checkpoint_path=str(model_path),
    )
    out_dir = Path(args.output_dir)
    cfg_python, simulator, env, _explore_cfg, python_roots = prepare_python_roots(args)
    action_mask_fn = _make_action_mask_fn(env)

    native_export_path = _export_hgb_to_native_text(model, out_dir / "v4_adv_hgb_native_export.tsv")
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(native_export_path))

    payload = _cfg_payload(cfg_python, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)

    rows: list[dict[str, Any]] = []
    failures: list[int] = []
    feature_dim = int(getattr(model, "feature_dim", 226))
    for pr in python_roots[: int(num_states)]:
        state = pr.root_state
        player = str(pr.root_player)
        inputs = infer_module.build_model_inputs(
            state,
            player,
            torch.device("cpu"),
            build_action_mask_flag=True,
            action_mask_fn=action_mask_fn,
        )
        if not getattr(inputs, "extras", None):
            raise RuntimeError("HGB alignment test expected ModelInputs.extras to be attached")

        wrapper_value, _priors = model.infer_from_inputs(inputs, player, device=torch.device("cpu"))
        python_features = np.asarray(model._features_from_extras(inputs.extras), dtype=np.float32).reshape(-1)
        if int(python_features.size) != feature_dim:
            raise RuntimeError(f"HGB feature dim mismatch: got {python_features.size}, expected {feature_dim}")
        raw_direct = float(np.asarray(model.hgb.predict(python_features.reshape(1, -1)), dtype=np.float64)[0])
        direct_value = float(min(raw_direct, 0.0))

        native_out = runtime.infer_from_state(
            nlt._native_state_payload(env, state),
            payload,
            -1,
        )
        native_features = np.asarray(list(native_out["features"]), dtype=np.float32).reshape(-1)
        native_raw = float(native_out["raw_value"])
        native_value = float(native_out["value"])
        max_feature_diff = float(np.max(np.abs(python_features - native_features))) if python_features.size == native_features.size else float("inf")
        value_abs_diff = abs(float(wrapper_value) - native_value)
        raw_abs_diff = abs(raw_direct - native_raw)
        direct_abs_diff = abs(direct_value - native_value)
        passed = bool(
            max_feature_diff <= float(feature_tolerance)
            and value_abs_diff <= float(value_tolerance)
            and direct_abs_diff <= float(value_tolerance)
        )
        if not passed:
            failures.append(int(pr.root_id))

        rows.append(
            {
                "root_id": int(pr.root_id),
                "history_hops": int(pr.history_hops),
                "root_depth": int(pr.root_depth),
                "player": player,
                "state_signature": _state_signature(env, pr),
                "model_kind": "v4_adv_hgb",
                "model_path": str(model_path),
                "native_export_path": str(native_export_path),
                "model_tag": str(getattr(model, "model_tag", "")),
                "feature_dim": int(python_features.size),
                "native_feature_dim": int(native_features.size),
                "python_wrapper_value": float(wrapper_value),
                "direct_hgb_value": float(direct_value),
                "raw_direct_hgb_value": float(raw_direct),
                "native_value": native_value,
                "native_raw_value": native_raw,
                "value_abs_diff": float(value_abs_diff),
                "direct_abs_diff": float(direct_abs_diff),
                "raw_abs_diff": float(raw_abs_diff),
                "max_feature_diff": float(max_feature_diff),
                "native_decode_time_at_max": float(native_out["decode_time_at_max"]),
                "native_inference_supported": True,
                "native_note": "native 226D feature extraction + exported sklearn HGB tree traversal",
                "passed": passed,
            }
        )

    csv_path = out_dir / "state_inference_alignment.csv"
    _write_csv(csv_path, rows)
    if failures:
        raise AssertionError(f"native HGB inference alignment failed for root_id(s)={failures[:20]}; wrote {csv_path}")
    return csv_path

def run_state_inference_alignment(
    *,
    num_states: int = DEFAULT_NUM_STATES,
    history_hops_min: int = DEFAULT_HISTORY_HOPS_MIN,
    history_hops_max: int = DEFAULT_HISTORY_HOPS_MAX,
    history_seed: int = DEFAULT_HISTORY_SEED,
    value_tolerance: float = DEFAULT_VALUE_TOLERANCE,
    policy_tolerance: float = DEFAULT_POLICY_TOLERANCE,
    feature_tolerance: float = DEFAULT_FEATURE_TOLERANCE,
    checkpoint_path: str | None = None,
    hgb_model_path: str | Path | None = None,
    model_kind: str = DEFAULT_MODEL_KIND,
    force_build: bool = False,
) -> Path:
    if str(model_kind) == "v4_adv_hgb":
        return run_hgb_state_inference_alignment(
            num_states=int(num_states),
            history_hops_min=int(history_hops_min),
            history_hops_max=int(history_hops_max),
            history_seed=int(history_seed),
            value_tolerance=float(value_tolerance),
            feature_tolerance=float(feature_tolerance),
            hgb_model_path=hgb_model_path,
            force_build=bool(force_build),
        )
    if str(model_kind) != "torchscript":
        raise ValueError(f"unknown model_kind={model_kind!r}; expected 'v4_adv_hgb' or 'torchscript'")

    from vidur.Game_Version3.DNN.infer import build_model_inputs
    from vidur.Game_Version3.DNN.native_selfplay import (
        _cfg_payload,
        attach_execution_predictor_payload,
        export_torchscript_pair,
    )
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.Game_Version3.tests.feature_conversion_tests import _make_action_mask_fn

    native = import_native_cpp(force_build=bool(force_build))
    args = make_args(
        "state_inference_alignment",
        num_roots=int(num_states),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        history_seed=int(history_seed),
        feature_tolerance=float(feature_tolerance),
        model_version=1,
        checkpoint_path=checkpoint_path,
    )
    out_dir = Path(args.output_dir)
    cfg_python, simulator, env, _explore_cfg, python_roots = prepare_python_roots(args)
    action_mask_fn = _make_action_mask_fn(env)

    model, weights_path = _load_or_make_python_model(
        cfg_python,
        Path(args.checkpoint_path),
        seed=int(history_seed) + 17,
        out_dir=out_dir,
    )
    torchscript_spec = export_torchscript_pair(
        model=model,
        model_version=int(args.model_version),
        weights_path=weights_path,
        out_dir=out_dir / "torchscript",
    )

    runtime = native.NativeTorchScriptInferRuntimeGV2("cpu")
    runtime.load_models({int(args.model_version): torchscript_spec})

    payload = _cfg_payload(cfg_python, torchscript_model_spec=torchscript_spec)
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = False

    rows: list[dict[str, Any]] = []
    failures: list[int] = []
    for pr in python_roots[: int(num_states)]:
        state = pr.root_state
        player = str(pr.root_player)

        py_inputs = build_model_inputs(
            state,
            player,
            torch.device("cpu"),
            build_action_mask_flag=True,
            action_mask_fn=action_mask_fn,
        )
        py_inputs_native_dict = _model_inputs_to_native_dict(py_inputs)

        native_out = native.search_mcts_dnn_torchscript(
            runtime,
            0,
            nlt._native_state_payload(env, state),
            payload,
            1,
            player,
            int(pr.root_node_id_override or pr.root_id),
            int(pr.root_depth),
            0,
            int(pr.root_id),
            int(history_seed) + int(pr.root_id),
            False,
            False,
            "",
            "",
        )
        native_inputs_dict = dict(native_out["root_inputs"])
        native_inputs_as_py = _native_inputs_to_model_inputs(native_inputs_dict)

        with torch.inference_mode():
            py_value_on_py_inputs, py_policy_on_py_inputs = model.infer_from_inputs(
                py_inputs,
                player,
                device=torch.device("cpu"),
            )
            py_value_on_native_inputs, py_policy_on_native_inputs = model.infer_from_inputs(
                native_inputs_as_py,
                player,
                device=torch.device("cpu"),
            )

        native_value_on_py_inputs, native_policy_on_py_inputs = runtime.infer_from_inputs(
            py_inputs_native_dict,
            player,
            int(args.model_version),
        )
        native_value_on_native_inputs, native_policy_on_native_inputs = runtime.infer_from_inputs(
            native_inputs_dict,
            player,
            int(args.model_version),
        )

        feature_diffs = {
            "prefill_feature_diff": _max_abs_diff_tensor(
                py_inputs.prefill_req_features,
                native_inputs_as_py.prefill_req_features,
            ),
            "decode_feature_diff": _max_abs_diff_tensor(
                py_inputs.decode_req_features,
                native_inputs_as_py.decode_req_features,
            ),
            "global_feature_diff": _max_abs_diff_tensor(
                py_inputs.global_features,
                native_inputs_as_py.global_features,
            ),
            "req_feature_diff": _max_abs_diff_tensor(
                py_inputs.req_features,
                native_inputs_as_py.req_features,
            ),
        }
        prefill_mask_passed = bool(torch.equal(py_inputs.prefill_req_mask, native_inputs_as_py.prefill_req_mask))
        decode_mask_passed = bool(torch.equal(py_inputs.decode_req_mask, native_inputs_as_py.decode_req_mask))
        req_mask_passed = bool(torch.equal(py_inputs.req_mask, native_inputs_as_py.req_mask))
        action_mask_passed = bool(torch.equal(py_inputs.action_mask, native_inputs_as_py.action_mask))
        request_masks_passed = bool(prefill_mask_passed and decode_mask_passed and req_mask_passed)
        action_mask_diff_count = int(
            (py_inputs.action_mask.detach().to("cpu") != native_inputs_as_py.action_mask.detach().to("cpu"))
            .to(torch.int64)
            .sum()
            .item()
        )
        max_feature_diff = max(float(x) for x in feature_diffs.values())

        value_diff_same_py_inputs = abs(float(py_value_on_py_inputs) - float(native_value_on_py_inputs))
        value_diff_same_native_inputs = abs(float(py_value_on_native_inputs) - float(native_value_on_native_inputs))
        value_diff_end_to_end = abs(float(py_value_on_py_inputs) - float(native_value_on_native_inputs))
        policy_diff_same_py_inputs = _max_abs_diff_list(
            [float(x) for x in py_policy_on_py_inputs],
            [float(x) for x in native_policy_on_py_inputs],
        )
        policy_diff_same_native_inputs = _max_abs_diff_list(
            [float(x) for x in py_policy_on_native_inputs],
            [float(x) for x in native_policy_on_native_inputs],
        )

        passed = bool(
            request_masks_passed
            and max_feature_diff <= float(feature_tolerance)
            and value_diff_same_py_inputs <= float(value_tolerance)
            and value_diff_same_native_inputs <= float(value_tolerance)
            and value_diff_end_to_end <= float(value_tolerance)
            and policy_diff_same_py_inputs <= float(policy_tolerance)
            and policy_diff_same_native_inputs <= float(policy_tolerance)
        )
        if not passed:
            failures.append(int(pr.root_id))

        rows.append(
            {
                "root_id": int(pr.root_id),
                "history_hops": int(pr.history_hops),
                "root_depth": int(pr.root_depth),
                "player": player,
                "state_signature": _state_signature(env, pr),
                "checkpoint_path": str(weights_path),
                "torchscript_spec": str(torchscript_spec),
                "python_value_python_inputs": float(py_value_on_py_inputs),
                "native_value_python_inputs": float(native_value_on_py_inputs),
                "python_value_native_inputs": float(py_value_on_native_inputs),
                "native_value_native_inputs": float(native_value_on_native_inputs),
                "abs_diff_same_python_inputs": value_diff_same_py_inputs,
                "abs_diff_same_native_inputs": value_diff_same_native_inputs,
                "abs_diff_end_to_end": value_diff_end_to_end,
                "policy_diff_same_python_inputs": policy_diff_same_py_inputs,
                "policy_diff_same_native_inputs": policy_diff_same_native_inputs,
                "request_masks_passed": request_masks_passed,
                "prefill_mask_passed": prefill_mask_passed,
                "decode_mask_passed": decode_mask_passed,
                "req_mask_passed": req_mask_passed,
                "action_mask_passed": action_mask_passed,
                "action_mask_diff_count": action_mask_diff_count,
                "max_feature_diff": max_feature_diff,
                **feature_diffs,
                "passed": passed,
            }
        )

    csv_path = out_dir / "state_inference_alignment.csv"
    _write_csv(csv_path, rows)
    if failures:
        raise AssertionError(
            f"native/Python state inference alignment failed for root_id(s)={failures[:20]}; wrote {csv_path}"
        )
    return csv_path


def test_state_inference_alignment_python_vs_native() -> None:
    run_state_inference_alignment()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python and native GV3 TorchScript state inference.")
    parser.add_argument("--num-states", type=int, default=DEFAULT_NUM_STATES)
    parser.add_argument("--history-hops-min", type=int, default=DEFAULT_HISTORY_HOPS_MIN)
    parser.add_argument("--history-hops-max", type=int, default=DEFAULT_HISTORY_HOPS_MAX)
    parser.add_argument("--history-seed", type=int, default=DEFAULT_HISTORY_SEED)
    parser.add_argument("--value-tolerance", type=float, default=DEFAULT_VALUE_TOLERANCE)
    parser.add_argument("--policy-tolerance", type=float, default=DEFAULT_POLICY_TOLERANCE)
    parser.add_argument("--feature-tolerance", type=float, default=DEFAULT_FEATURE_TOLERANCE)
    parser.add_argument("--model-kind", choices=("v4_adv_hgb", "torchscript"), default=DEFAULT_MODEL_KIND)
    parser.add_argument("--hgb-model-path", type=str, default=str(DEFAULT_HGB_MODEL_PATH))
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--force-build", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    csv_path = run_state_inference_alignment(
        num_states=int(args.num_states),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        value_tolerance=float(args.value_tolerance),
        policy_tolerance=float(args.policy_tolerance),
        feature_tolerance=float(args.feature_tolerance),
        checkpoint_path=args.checkpoint_path,
        hgb_model_path=args.hgb_model_path,
        model_kind=str(args.model_kind),
        force_build=bool(args.force_build),
    )
    print(f"state inference alignment passed; wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
