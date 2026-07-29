"""Parity and sufficiency tests for the GV3 Markov value-state contract."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from vidur.AlphaGoZero.dnn_models import (
    ADVERSARY_ACTION_DIM,
    CONTROLLER_ACTION_DIM,
    MarkovValueDeepSet,
    PolicyRankMLP,
    export_dnn_to_native,
)
from vidur.AlphaGoZero.markov_value_features import (
    GLOBAL_FEATURE_NAMES,
    LAUNCH_FEATURE_NAMES,
    REQUEST_FEATURE_NAMES,
    MarkovValueFeatures,
    build_markov_value_features,
    features_from_replay_row,
)
from vidur.AlphaGoZero.xl_coordinator import _source_root_key, _write_role_partition


def representative_state() -> dict[str, Any]:
    return {
        "sim_time": 1.0,
        "requests": [
            {
                "request_id": 0,
                "arrived_at": 0.2,
                "queued_at": 0.25,
                "num_prefill_tokens": 1024,
                "num_processed_prefill_tokens": 256,
                "num_decode_tokens": 216,
                "num_processed_decode_tokens": 0,
                "prefill_slo_time": 1.0,
                "decode_slo_time": 0.05,
                "prefill_deadline": 1.2,
                "decode_next_deadline": -1.0,
                "prefill_completed_at": -1.0,
                "prefill_lateness": 0.0,
                "decode_lateness": 0.0,
                "is_prefill_complete": False,
            },
            {
                "request_id": 1,
                "arrived_at": 0.1,
                "queued_at": 0.1,
                "num_prefill_tokens": 512,
                "num_processed_prefill_tokens": 512,
                "num_decode_tokens": 216,
                "num_processed_decode_tokens": 10,
                "prefill_slo_time": 1.0,
                "decode_slo_time": 0.05,
                "prefill_deadline": 1.1,
                "decode_next_deadline": 1.03,
                "prefill_completed_at": 0.8,
                "prefill_lateness": 0.02,
                "decode_lateness": 0.01,
                "is_prefill_complete": True,
            },
        ],
        "stats": {
            "active_request_ids": [0, 1],
            "violated_request_ids": [1],
            "prefill_lateness_finalized_ids": [1],
            "per_request_prefill_lateness_by_id": {"1": 0.02},
            "per_request_decode_lateness_by_id": {"1": 0.01},
            "decode_next_deadline_by_id": {"1": 1.03},
            "decode_tokens_counted_by_id": {"1": 10},
            "recent_launches": [(0.4, 3, 3072), (0.9, 2, 1536)],
            "next_adv_tick": 1.2,
            "pending_adv_tick": False,
            "missed_adv_source": 1,
            "decode_credit_balance": 432,
            "decode_credit_available": 432,
        },
    }


def successor_states() -> tuple[dict[str, Any], dict[str, Any]]:
    decode_only = representative_state()
    decode_only["sim_time"] += 0.013
    decode_only["requests"][1]["num_processed_decode_tokens"] += 1
    decode_only["stats"]["decode_tokens_counted_by_id"]["1"] += 1
    with_prefill = copy.deepcopy(decode_only)
    with_prefill["requests"][0]["num_processed_prefill_tokens"] += 256
    return decode_only, with_prefill


def _native_module() -> Any:
    from vidur.Game_Version3_Cpp import mcts_native_gv2

    return mcts_native_gv2


def _native_features(runtime: Any, payload: dict[str, Any]) -> MarkovValueFeatures:
    raw = runtime.build_markov_features_from_state(payload)
    request_count = int(raw["request_count"])
    launch_count = int(raw["launch_count"])
    return MarkovValueFeatures(
        global_features=np.asarray(raw["global_features"], dtype=np.float32),
        request_features=np.asarray(raw["request_features"], dtype=np.float32).reshape(
            request_count, len(REQUEST_FEATURE_NAMES)
        ),
        launch_features=np.asarray(raw["launch_features"], dtype=np.float32).reshape(
            launch_count, len(LAUNCH_FEATURE_NAMES)
        ),
        request_ids=tuple(int(value) for value in raw["request_ids"]),
    )


def write_parity_csv(
    output_csv: Path,
    *,
    seed: int = 2026,
    value_model_path: Path | None = None,
) -> dict[str, float | int]:
    torch.manual_seed(int(seed))
    runtime = _native_module().NewFeatures226HGBRuntime()
    decode_only, with_prefill = successor_states()
    payloads = {
        "base": representative_state(),
        "decode_only_successor": decode_only,
        "prefill_successor": with_prefill,
    }
    python_features = {
        case: build_markov_value_features(payload) for case, payload in payloads.items()
    }
    native_features = {
        case: _native_features(runtime, payload) for case, payload in payloads.items()
    }

    rows: list[dict[str, Any]] = []
    max_feature_error = 0.0
    for case in payloads:
        groups = (
            (
                "global",
                GLOBAL_FEATURE_NAMES,
                python_features[case].global_features.reshape(-1),
                native_features[case].global_features.reshape(-1),
            ),
            (
                "request",
                REQUEST_FEATURE_NAMES,
                python_features[case].request_features,
                native_features[case].request_features,
            ),
            (
                "launch",
                LAUNCH_FEATURE_NAMES,
                python_features[case].launch_features,
                native_features[case].launch_features,
            ),
        )
        for scope, names, python_values, native_values in groups:
            py_rows = np.asarray(python_values, dtype=np.float32).reshape(-1, len(names))
            native_rows = np.asarray(native_values, dtype=np.float32).reshape(-1, len(names))
            if py_rows.shape != native_rows.shape:
                raise AssertionError(f"{case}/{scope} shape mismatch")
            for token_index in range(py_rows.shape[0]):
                for feature_index, name in enumerate(names):
                    py_value = float(py_rows[token_index, feature_index])
                    native_value = float(native_rows[token_index, feature_index])
                    error = abs(py_value - native_value)
                    max_feature_error = max(max_feature_error, error)
                    rows.append(
                        {
                            "case": case,
                            "scope": scope,
                            "token_index": token_index,
                            "feature_index": feature_index,
                            "feature_name": name,
                            "python_value": py_value,
                            "native_value": native_value,
                            "abs_diff": error,
                        }
                    )

    if value_model_path is None:
        model = MarkovValueDeepSet(role="controller")
    else:
        model = joblib.load(Path(value_model_path))
        if not isinstance(model, MarkovValueDeepSet):
            raise TypeError(f"{value_model_path} is not a MarkovValueDeepSet")
    model.eval()
    with tempfile.TemporaryDirectory(prefix="agz_markov_parity_") as tmp:
        export_path = Path(tmp) / "value.tsv"
        export_dnn_to_native(model, export_path, model_tag="markov_parity")
        runtime.load_model_export(str(export_path))
        ordered_cases = list(payloads)
        python_values = model.predict_structured(
            [python_features[case] for case in ordered_cases]
        ).astype(np.float64)
        native_values = np.asarray(
            [
                runtime.infer_from_state(payloads[case], {}, index + 1)["raw_value"]
                for index, case in enumerate(ordered_cases)
            ],
            dtype=np.float64,
        )
    value_errors = np.abs(python_values - native_values)
    for index, case in enumerate(ordered_cases):
        rows.append(
            {
                "case": case,
                "scope": "value_inference",
                "token_index": 0,
                "feature_index": 0,
                "feature_name": "controller_cost_to_go",
                "python_value": float(python_values[index]),
                "native_value": float(native_values[index]),
                "abs_diff": float(value_errors[index]),
            }
        )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "csv_rows": len(rows),
        "max_feature_abs_diff": max_feature_error,
        "max_value_abs_diff": float(np.max(value_errors)),
    }


class MarkovValueFeatureTest(unittest.TestCase):
    def test_python_native_csv_parity_and_batch_value_parity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agz_markov_csv_") as tmp:
            output = Path(tmp) / "parity.csv"
            summary = write_parity_csv(output)
            self.assertTrue(output.is_file())
            self.assertGreater(int(summary["csv_rows"]), 100)
            self.assertLessEqual(float(summary["max_feature_abs_diff"]), 1e-7)
            self.assertLessEqual(float(summary["max_value_abs_diff"]), 1e-4)

    def test_publication_parity_dispatch_accepts_markov_value_artifact(self) -> None:
        from vidur.AlphaGoZero.test_and_analysis.test_dnn_native_parity import run

        with tempfile.TemporaryDirectory(prefix="agz_markov_publish_parity_") as tmp:
            root = Path(tmp)
            value_path = root / "value.joblib"
            controller_path = root / "controller.joblib"
            adversary_path = root / "adversary.joblib"
            joblib.dump(MarkovValueDeepSet(role="controller"), value_path)
            joblib.dump(
                PolicyRankMLP(CONTROLLER_ACTION_DIM, role="controller"),
                controller_path,
            )
            joblib.dump(
                PolicyRankMLP(ADVERSARY_ACTION_DIM, role="adversary"),
                adversary_path,
            )
            result = run(
                argparse.Namespace(
                    value_model=value_path,
                    controller_policy_model=controller_path,
                    adversary_policy_model=adversary_path,
                    rows=16,
                    seed=2026,
                    tolerance=1e-4,
                )
            )
            self.assertEqual(result["value"]["feature_schema"], "markov_v2")
            self.assertLessEqual(float(result["value"]["max_abs_error"]), 1e-4)
            self.assertEqual(
                result["controller_policy"]["python_top3"],
                result["controller_policy"]["native_top3"],
            )

    def test_exact_division_and_asinh_not_buckets(self) -> None:
        features = build_markov_value_features(representative_state())
        self.assertAlmostEqual(float(features.global_features[0]), 2.0 / 120.0, places=7)
        self.assertAlmostEqual(float(features.global_features[10]), math.asinh(1.0), places=7)
        self.assertAlmostEqual(float(features.request_features[0, 3]), 768.0 / 4096.0, places=7)
        self.assertAlmostEqual(float(features.launch_features[0, 0]), math.asinh(0.6), places=7)

    def test_equal_duration_prefill_and_decode_successors_do_not_alias(self) -> None:
        decode_only, with_prefill = successor_states()
        decode_features = build_markov_value_features(decode_only)
        prefill_features = build_markov_value_features(with_prefill)
        self.assertNotEqual(
            float(decode_features.global_features[3]),
            float(prefill_features.global_features[3]),
        )
        self.assertAlmostEqual(
            float(decode_features.request_features[0, 3] - prefill_features.request_features[0, 3]),
            256.0 / 4096.0,
            places=7,
        )

    def test_deepset_is_request_and_launch_permutation_invariant(self) -> None:
        payload = representative_state()
        base = build_markov_value_features(payload)
        permuted_payload = copy.deepcopy(payload)
        permuted_payload["requests"].reverse()
        permuted_payload["stats"]["recent_launches"].reverse()
        permuted = build_markov_value_features(permuted_payload)
        torch.manual_seed(9)
        model = MarkovValueDeepSet(role="controller").eval()
        values = model.predict_structured([base, permuted])
        self.assertLessEqual(abs(float(values[0]) - float(values[1])), 1e-5)

    def test_replay_round_trip_preserves_all_structured_values(self) -> None:
        features = build_markov_value_features(representative_state())
        restored = features_from_replay_row(features.replay_fields())
        np.testing.assert_array_equal(features.global_features, restored.global_features)
        np.testing.assert_array_equal(features.request_features, restored.request_features)
        np.testing.assert_array_equal(features.launch_features, restored.launch_features)

    def test_invalid_active_request_is_rejected_and_stale_launch_is_ignored(self) -> None:
        missing_request = representative_state()
        missing_request["requests"] = missing_request["requests"][:1]
        with self.assertRaisesRegex(ValueError, "has no request record"):
            build_markov_value_features(missing_request)
        stale_launch = representative_state()
        stale_launch["stats"]["recent_launches"] = [(-0.1, 1, 128)]
        filtered = build_markov_value_features(stale_launch)
        self.assertEqual(filtered.launch_features.shape, (0, len(LAUNCH_FEATURE_NAMES)))
        self.assertEqual(float(filtered.global_features[15]), 0.0)
        self.assertEqual(float(filtered.global_features[16]), 0.0)

    def test_coordinator_role_partition_preserves_markov_columns(self) -> None:
        features = build_markov_value_features(representative_state())
        source_row = {
            "game_id": "17",
            "turn_number": "4",
            "depth_number": "0",
            "player": "controller",
            "feature_complete": "1",
            "state_features_json": "[0.0]",
            "value_feature_complete": "1",
            **features.replay_fields(),
        }
        with tempfile.TemporaryDirectory(prefix="agz_markov_partition_") as tmp:
            root = Path(tmp)
            replay = root / "source_replay.csv"
            policy = root / "source_policy.csv"
            with replay.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(source_row))
                writer.writeheader()
                writer.writerow(source_row)
            with policy.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("game_id", "turn_number", "depth_number", "player"),
                )
                writer.writeheader()

            part = _write_role_partition(
                root,
                worker_id="worker1",
                source_shard_id="shard1",
                replay=replay,
                policy_rows=policy,
                manifest={
                    "model_version": 100,
                    "controller_model_version": 100,
                    "adversary_model_version": 100,
                },
                role="controller",
                selected_keys={_source_root_key(source_row)},
                extra={"source_worker_id": "worker1"},
            )
            self.assertIsNotNone(part)
            partition_replay = Path(part) / "replay_target_runtime_feature_complete.csv"
            with partition_replay.open("r", encoding="utf-8", newline="") as handle:
                retained = next(csv.DictReader(handle))
            restored = features_from_replay_row(retained)
            np.testing.assert_array_equal(features.global_features, restored.global_features)
            np.testing.assert_array_equal(features.request_features, restored.request_features)
            np.testing.assert_array_equal(features.launch_features, restored.launch_features)


if __name__ == "__main__":
    unittest.main()
