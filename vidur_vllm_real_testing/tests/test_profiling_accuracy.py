from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from vidur_vllm_real_testing.profiling_accuracy.compare_results import comparison_rows
from vidur_vllm_real_testing.profiling_accuracy.config import ProfilingConfig, load_config
from vidur_vllm_real_testing.profiling_accuracy.vllm_records import aggregate_vllm_records
from vidur.entities.execution_time_predictor_request import ExecutionTimePredictorRequest
from vidur.execution_time_predictor.sklearn_execution_time_predictor import _MemmapTable
from vidur.execution_time_predictor.sklearn_execution_time_predictor_batch import (
    SklearnExecutionTimePredictorBatch,
)
from vidur_vllm_real_testing.profiling_accuracy.stats import percentile, timing_stats
from vidur_vllm_real_testing.profiling_accuracy.token_space import (
    decode_batch_space,
    mlp_token_space,
    prefill_chunk_space,
)


class ProfilingAccuracyTest(unittest.TestCase):
    def _config(self) -> ProfilingConfig:
        temp_dir = Path(tempfile.mkdtemp())
        profile = temp_dir / "prefill.csv"
        profile.write_text(
            "prefill_tokens,prefill_time_seconds\n128,0.01\n4096,0.2\n",
            encoding="utf-8",
        )
        return ProfilingConfig(
            prefill_profile_path=str(profile), output_root=str(temp_dir / "output")
        )

    def test_config_builds_cross_product_and_never_calibrates(self) -> None:
        config = self._config()
        config.validate()
        self.assertEqual(config.cases(), ((1, 128), (1, 4096), (2, 128), (2, 4096)))
        self.assertIsNone(config.to_dict()["calibration"])
        self.assertTrue(config.vllm_enable_chunked_prefill)

    def test_request_counts_and_token_budget_are_environment_configurable(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "VIDUR_PROFILE_REQUEST_COUNTS": "1,2,4",
                "VIDUR_PROFILE_MAX_BATCHED_TOKENS": "16384",
            },
        ):
            config = load_config()
        self.assertEqual(config.request_counts, (1, 2, 4))
        self.assertEqual(config.vllm_max_num_batched_tokens, 16384)

    def test_two_request_max_case_exposes_attention_clamp(self) -> None:
        config = self._config()
        self.assertGreater(2 * max(config.prefill_sizes()), config.max_prefill_chunk_size)

    def test_token_space_matches_vidur_limits(self) -> None:
        values = mlp_token_space(8192)
        self.assertEqual(max(values), 8192)
        self.assertEqual(min(values), 1)
        self.assertEqual(len(values), len(set(values)))

    def test_attention_space_covers_configured_limits_without_duplicates(self) -> None:
        chunks = prefill_chunk_space(4096)
        self.assertEqual(chunks.count(1024), 1)
        self.assertEqual(chunks[-1], 4096)
        batches = decode_batch_space(256)
        self.assertIn(128, batches)
        self.assertEqual(batches[-1], 256)


    def test_timing_statistics(self) -> None:
        stats = timing_stats([1, 2, 3, 4])
        self.assertEqual(stats["median"], 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)

    def test_vllm_aggregation_ignores_engine_internal_warmups(self) -> None:
        records = [
            {
                "scheduled_tokens": {"_warmup_0_": 2, "_warmup_1_": 2},
                "transformer_blocks_ms": 99.0,
                "full_model_forward_ms": 100.0,
            },
            {
                "scheduled_tokens": {"request-0": 128},
                "transformer_blocks_ms": 10.0,
                "full_model_forward_ms": 11.0,
            },
            {
                "scheduled_tokens": {"request-1": 128},
                "transformer_blocks_ms": 12.0,
                "full_model_forward_ms": 13.0,
            },
        ]
        rows = aggregate_vllm_records(records, warmups=1, repetitions=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_count"], 1)
        self.assertEqual(rows[0]["prefill_tokens_per_request"], 128)
        self.assertEqual(rows[0]["vllm_transformer_blocks_ms_median"], 12.0)

    def test_vllm_aggregation_preserves_exact_two_request_shape(self) -> None:
        records = [
            {
                "scheduled_tokens": {"profile_0_0_0": 256, "profile_0_0_1": 256},
                "transformer_blocks_ms": 20.0,
                "full_model_forward_ms": 21.0,
            },
            {
                "scheduled_tokens": {"profile_0_1_0": 256, "profile_0_1_1": 256},
                "transformer_blocks_ms": 22.0,
                "full_model_forward_ms": 23.0,
            },
            {
                "scheduled_tokens": {"profile_0_2_0": 256, "profile_0_2_1": 256},
                "transformer_blocks_ms": 24.0,
                "full_model_forward_ms": 25.0,
            },
        ]
        rows = aggregate_vllm_records(records, warmups=1, repetitions=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_count"], 2)
        self.assertEqual(rows[0]["prefill_tokens_per_request"], 256)
        self.assertEqual(rows[0]["total_prefill_tokens"], 512)
        self.assertEqual(rows[0]["vllm_transformer_blocks_ms_median"], 23.0)

    def test_prefill_memmap_rejects_and_clamps_oversized_chunk(self) -> None:
        table = _MemmapTable(
            mmap=np.zeros((129, 128), dtype=np.float64),
            kind="prefill",
            max_tokens=8192,
            max_batch_size=256,
            kv_gran=64,
            prefill_gran=32,
        )
        self.assertNotIn((0, 4160), table)
        self.assertEqual(table.nearest((0, 4160)), (0, 4096))
        self.assertEqual(table[(0, 4096)], 0.0)

    def test_two_request_prefill_uses_rounded_l2_aggregate(self) -> None:
        requests = [
            ExecutionTimePredictorRequest(
                num_processed_tokens=0,
                num_tokens_to_process=4096,
                is_prefill_complete=False,
            )
            for _ in range(2)
        ]
        batch = SklearnExecutionTimePredictorBatch(
            requests,
            kv_cache_prediction_granularity=64,
            prefill_chunk_size_prediction_granularity=32,
        )
        self.assertEqual(batch.total_num_tokens, 8192)
        self.assertEqual(batch.prefill_agg_chunk_size, 5824)

    def test_comparison_uses_transformer_block_boundary(self) -> None:
        actual = {
            (1, 128): {
                "vllm_transformer_blocks_ms_median": "10",
                "vllm_transformer_blocks_ms_p95": "11",
                "vllm_full_model_forward_ms_median": "12",
                "vllm_version": "0.26.0",
                "flashinfer_version": "0.6.14",
                "attention_backend": "FLASHINFER",
                "timing_boundary": "blocks",
            }
        }
        predicted = {
            (1, 128): {
                "vidur_model_ms": "9",
                "vidur_attention_lookup_clamped": "False",
            }
        }
        row = comparison_rows(actual, predicted)[0]
        self.assertEqual(row["vidur_minus_vllm_percent"], -10.0)
        self.assertEqual(row["absolute_percent_error"], 10.0)
        self.assertFalse(row["calibration_applied"])


if __name__ == "__main__":
    unittest.main()
