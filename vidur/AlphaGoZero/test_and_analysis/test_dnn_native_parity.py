"""Reusable Python/native parity checks for AlphaGoZero residual DNN artifacts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch

from vidur.AlphaGoZero.dnn_models import (
    ADVERSARY_ACTION_DIM,
    CONTROLLER_ACTION_DIM,
    VALUE_FEATURE_DIM,
    MarkovValueDeepSet,
    PolicyRankMLP,
    ValueResidualMLP,
    export_dnn_to_native,
)
from vidur.AlphaGoZero.markov_value_features import build_markov_value_features


def _native_module():
    from vidur.Game_Version3_Cpp import mcts_native_gv2

    return mcts_native_gv2


def _load_or_create(path: Path | None, model: torch.nn.Module) -> torch.nn.Module:
    if path is None:
        return model.eval()
    loaded = joblib.load(path)
    if not isinstance(loaded, type(model)):
        raise TypeError(f"{path} contains {type(loaded).__name__}, expected {type(model).__name__}")
    return loaded.cpu().eval()


def _value_parity(
    native: object,
    model: ValueResidualMLP,
    features: np.ndarray,
    export_path: Path,
) -> dict[str, float | int]:
    export_dnn_to_native(model, export_path, model_tag="parity_value")
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(export_path))
    python_values = model.predict(features).astype(np.float64)
    native_values = np.asarray(
        [runtime.predict_from_features(row.tolist()) for row in features],
        dtype=np.float64,
    )
    native_batch = np.asarray(
        runtime.predict_batch_from_flat_features(
            features.reshape(-1).tolist(),
            int(features.shape[0]),
            int(features.shape[1]),
        ),
        dtype=np.float64,
    )
    error = np.abs(python_values - native_values)
    batch_error = np.abs(python_values - native_batch)
    return {
        "rows": int(features.shape[0]),
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error)),
        "batch_max_abs_error": float(np.max(batch_error)),
        "batch_mean_abs_error": float(np.mean(batch_error)),
    }


def _markov_value_parity(
    native: object,
    model: MarkovValueDeepSet,
    export_path: Path,
) -> dict[str, float | int | str]:
    # Valid GV3 states exercise serialization, native feature construction,
    # and inference together instead of comparing arbitrary non-state tensors.
    from vidur.AlphaGoZero.test_and_analysis.test_markov_value_features import (
        representative_state,
        successor_states,
    )

    decode_only, with_prefill = successor_states()
    payloads = [representative_state(), decode_only, with_prefill]
    features = [build_markov_value_features(payload) for payload in payloads]

    export_dnn_to_native(model, export_path, model_tag="parity_markov_value")
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(export_path))
    python_values = model.predict_structured(features).astype(np.float64)
    native_values = np.asarray(
        [
            runtime.infer_from_state(payload, {}, index + 1)["raw_value"]
            for index, payload in enumerate(payloads)
        ],
        dtype=np.float64,
    )
    error = np.abs(python_values - native_values)
    return {
        "feature_schema": "markov_v2",
        "rows": len(payloads),
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error)),
    }


def _policy_parity(
    native: object,
    model: PolicyRankMLP,
    states: np.ndarray,
    actions: np.ndarray,
    export_path: Path,
) -> dict[str, float | int]:
    export_dnn_to_native(model, export_path, model_tag=f"parity_{model.role}_policy")
    runtime = native.NativeHGBModelRuntime()
    runtime.load_model_export(str(export_path))

    rows = np.concatenate((states, actions), axis=1).astype(np.float32, copy=False)
    python_logits = model.predict(rows).astype(np.float64)
    native_single = np.asarray(
        [runtime.predict_from_features(row.tolist()) for row in rows],
        dtype=np.float64,
    )
    native_batch = np.asarray(
        runtime.predict_batch_from_flat_features(
            rows.reshape(-1).tolist(),
            int(rows.shape[0]),
            int(rows.shape[1]),
        ),
        dtype=np.float64,
    )

    single_error = np.abs(python_logits - native_single)
    batch_error = np.abs(python_logits - native_batch)
    python_top3 = np.argsort(-python_logits, kind="stable")[:3].astype(int).tolist()
    native_top3 = np.argsort(-native_batch, kind="stable")[:3].astype(int).tolist()
    return {
        "rows": int(rows.shape[0]),
        "single_max_abs_error": float(np.max(single_error)),
        "batch_max_abs_error": float(np.max(batch_error)),
        "single_mean_abs_error": float(np.mean(single_error)),
        "batch_mean_abs_error": float(np.mean(batch_error)),
        "python_top1": int(np.argmax(python_logits)),
        "native_top1": int(np.argmax(native_batch)),
        "python_top3": python_top3,
        "native_top3": native_top3,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    native = _native_module()

    if args.value_model is None:
        value: torch.nn.Module = ValueResidualMLP(role="controller").eval()
    else:
        value = joblib.load(args.value_model)
        if not isinstance(value, (ValueResidualMLP, MarkovValueDeepSet)):
            raise TypeError(
                f"{args.value_model} contains {type(value).__name__}, expected "
                "ValueResidualMLP or MarkovValueDeepSet"
            )
        value = value.cpu().eval()
    controller = _load_or_create(
        args.controller_policy_model,
        PolicyRankMLP(CONTROLLER_ACTION_DIM, role="controller"),
    )
    adversary = _load_or_create(
        args.adversary_policy_model,
        PolicyRankMLP(ADVERSARY_ACTION_DIM, role="adversary"),
    )
    assert isinstance(value, (ValueResidualMLP, MarkovValueDeepSet))
    assert isinstance(controller, PolicyRankMLP)
    assert isinstance(adversary, PolicyRankMLP)

    value_features = rng.normal(0.0, 0.5, size=(int(args.rows), VALUE_FEATURE_DIM)).astype(np.float32)
    shared_controller_state = rng.normal(0.0, 0.5, size=(1, VALUE_FEATURE_DIM)).astype(np.float32)
    shared_adversary_state = rng.normal(0.0, 0.5, size=(1, VALUE_FEATURE_DIM)).astype(np.float32)
    controller_states = np.repeat(shared_controller_state, int(args.rows), axis=0)
    adversary_states = np.repeat(shared_adversary_state, int(args.rows), axis=0)
    controller_actions = rng.normal(
        0.0,
        0.5,
        size=(int(args.rows), CONTROLLER_ACTION_DIM),
    ).astype(np.float32)
    adversary_actions = rng.normal(
        0.0,
        0.5,
        size=(int(args.rows), ADVERSARY_ACTION_DIM),
    ).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="agz_dnn_parity_") as tmp:
        tmp_path = Path(tmp)
        value_result = (
            _markov_value_parity(native, value, tmp_path / "value.tsv")
            if isinstance(value, MarkovValueDeepSet)
            else _value_parity(native, value, value_features, tmp_path / "value.tsv")
        )
        result = {
            "value": value_result,
            "controller_policy": _policy_parity(
                native,
                controller,
                controller_states,
                controller_actions,
                tmp_path / "controller.tsv",
            ),
            "adversary_policy": _policy_parity(
                native,
                adversary,
                adversary_states,
                adversary_actions,
                tmp_path / "adversary.tsv",
            ),
        }

    tolerance = float(args.tolerance)
    assert float(result["value"]["max_abs_error"]) <= tolerance, result
    if "batch_max_abs_error" in result["value"]:
        assert float(result["value"]["batch_max_abs_error"]) <= tolerance, result
    for role in ("controller_policy", "adversary_policy"):
        assert float(result[role]["single_max_abs_error"]) <= tolerance, result
        assert float(result[role]["batch_max_abs_error"]) <= tolerance, result
        assert int(result[role]["python_top1"]) == int(result[role]["native_top1"]), result
        assert list(result[role]["python_top3"]) == list(result[role]["native_top3"]), result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-model", type=Path)
    parser.add_argument("--controller-policy-model", type=Path)
    parser.add_argument("--adversary-policy-model", type=Path)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
