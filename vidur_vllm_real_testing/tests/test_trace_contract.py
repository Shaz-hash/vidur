from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from vidur.AlphaGoZero.markov_value_features import build_markov_value_features
from vidur_vllm_real_testing.canonicalization import CanonicalizationError, PrefillProfile
from vidur_vllm_real_testing.prepare_trace import prepare
from vidur_vllm_real_testing.trace_contract import (
    CANONICAL_COLUMNS,
    CanonicalTraceRequest,
    assert_canonical_column_contract,
    canonicalize_trace,
    load_raw_trace,
    validate_gv3_launch_windows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "vidur_vllm_real_testing"
PREFILL_PROFILE = REPO_ROOT / "simulator_output" / "prefill_profile.csv"


class TraceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = PrefillProfile.load(PREFILL_PROFILE)

    def test_example_preserves_actual_and_adds_canonical_fields(self) -> None:
        rows, derived_prefill, derived_decode = load_raw_trace(
            PACKAGE_DIR / "traces" / "example_raw_trace.csv",
            profile=self.profile,
        )
        self.assertEqual((derived_prefill, derived_decode), (0, 0))
        canonical = canonicalize_trace(rows, profile=self.profile)
        first = canonical[0]
        self.assertEqual(first.actual_prefill_tokens, 764)
        self.assertEqual(first.canonical_prefill_tokens, 768)
        self.assertEqual(first.prefill_rounding_delta_tokens, 4)
        self.assertEqual(first.actual_prefill_slo_s, 0.25)
        self.assertAlmostEqual(first.canonical_prefill_slo_s, 3 * 0.07016849423102292)
        self.assertEqual(first.actual_decode_tokens, first.canonical_decode_tokens)

    def test_prepared_trace_is_sorted_and_manifest_is_checksummed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "canonical.csv"
            manifest_path = root / "manifest.json"
            result = prepare(
                input_path=PACKAGE_DIR / "traces" / "example_raw_trace.csv",
                output_path=output,
                manifest_path=manifest_path,
                prefill_profile_path=PREFILL_PROFILE,
            )
            self.assertEqual(len(result.rows), 5)
            self.assertEqual(
                [row.arrived_at_s for row in result.rows],
                sorted(row.arrived_at_s for row in result.rows),
            )
            with output.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), CANONICAL_COLUMNS)
                self.assertEqual(len(list(reader)), 5)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["canonical"]["row_count"], 5)
            self.assertEqual(manifest["profiles"]["prefill_sha256"], self.profile.sha256)
            self.assertEqual(manifest["model_contract"]["value_feature_schema"], "markov_v2")

    def test_missing_slos_fail_unless_explicitly_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.csv"
            path.write_text(
                "arrived_at_s,num_prefill_tokens,num_decode_tokens\n0.0,128,864\n",
                encoding="utf-8",
            )
            with self.assertRaises(CanonicalizationError):
                load_raw_trace(path, profile=self.profile)
            rows, prefills, decodes = load_raw_trace(
                path, profile=self.profile, derive_missing_slos=True
            )
            self.assertEqual((len(rows), prefills, decodes), (1, 1, 1))

    def test_duplicate_request_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.csv"
            path.write_text(
                "request_id,arrived_at_s,num_prefill_tokens,num_decode_tokens,prefill_slo_s,decode_slo_s\n"
                "same,0.0,128,864,0.05,0.05\n"
                "same,1.2,128,864,0.05,0.05\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CanonicalizationError, "duplicate request_id"):
                load_raw_trace(path, profile=self.profile)

    def test_eighth_request_in_one_second_fails(self) -> None:
        rows = tuple(self._canonical_row(index, 0.0) for index in range(8))
        with self.assertRaisesRegex(CanonicalizationError, "arrivals"):
            validate_gv3_launch_windows(rows)

    def test_canonical_request_is_accepted_by_markov_v2(self) -> None:
        rows, _, _ = load_raw_trace(
            PACKAGE_DIR / "traces" / "example_raw_trace.csv",
            profile=self.profile,
        )
        request = canonicalize_trace(rows, profile=self.profile)[0]
        payload = {
            "sim_time": request.arrived_at_s,
            "requests": [
                {
                    "request_id": 0,
                    "num_prefill_tokens": request.canonical_prefill_tokens,
                    "num_processed_prefill_tokens": 0,
                    "num_decode_tokens": request.canonical_decode_tokens,
                    "num_processed_decode_tokens": 0,
                    "is_prefill_complete": False,
                    "arrived_at": request.arrived_at_s,
                    "queued_at": request.arrived_at_s,
                    "prefill_slo_time": request.canonical_prefill_slo_s,
                    "decode_slo_time": request.canonical_decode_slo_s,
                    "prefill_deadline": request.arrived_at_s + request.canonical_prefill_slo_s,
                }
            ],
            "stats": {
                "active_request_ids": [0],
                "violated_request_ids": [],
                "prefill_lateness_finalized_ids": [],
                "next_adv_tick": 0.2,
                "decode_credit_balance": 0,
                "decode_credit_available": 0,
                "recent_launches": [
                    {
                        "timestamp": request.arrived_at_s,
                        "count": 1,
                        "prefill_tokens": request.canonical_prefill_tokens,
                    }
                ],
            },
        }
        features = build_markov_value_features(payload)
        self.assertEqual(features.request_features.shape, (1, 24))
        self.assertAlmostEqual(
            float(features.request_features[0, 1]),
            request.canonical_prefill_tokens / 4096.0,
            places=7,
        )

    def test_schema_required_fields_match_csv_contract(self) -> None:
        assert_canonical_column_contract()
        schema = json.loads(
            (PACKAGE_DIR / "schemas" / "canonical_trace_row.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(tuple(schema["required"]), CANONICAL_COLUMNS)

    @staticmethod
    def _canonical_row(index: int, arrived_at_s: float) -> CanonicalTraceRequest:
        return CanonicalTraceRequest(
            schema_version="gv3_vllm_trace_v1",
            request_id=f"req-{index}",
            source_row=index + 2,
            arrived_at_s=arrived_at_s,
            actual_prefill_tokens=128,
            canonical_prefill_tokens=128,
            prefill_rounding_delta_tokens=0,
            actual_decode_tokens=864,
            canonical_decode_tokens=864,
            actual_prefill_slo_s=0.05,
            canonical_prefill_slo_s=0.05,
            prefill_profile_time_s=0.015,
            actual_decode_slo_s=0.05,
            canonical_decode_slo_s=0.05,
            prompt_mode="synthetic_token_ids",
            prompt_ref="",
            seed=index,
            ignore_eos=True,
        )


if __name__ == "__main__":
    unittest.main()
