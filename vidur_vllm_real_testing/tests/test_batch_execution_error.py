from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import unittest

from vidur_vllm_real_testing.batch_execution_comparison import (
    build_batch_shape,
    comparison_rows,
    predictor_requests,
    read_batch_audit,
    write_comparison_csv,
)
from vidur_vllm_real_testing.profiling_accuracy.config import load_config
from vidur_vllm_real_testing.profiling_accuracy.predict_vidur_batches import (
    _create_predictor,
)
from vidur_vllm_real_testing.scheduler_contract import (
    LiveRequestSnapshot,
    LiveStateSnapshot,
    RequestPhase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/comparison_test.csv"
)
EXPECTED_CACHE = (
    REPO_ROOT
    / "simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/vidur_predictor_cache"
).resolve()


@dataclass
class _Execution:
    model_time_ms: float


class _Predictor:
    def __init__(self, model_time_ms: float = 22.5) -> None:
        self.model_time_ms = model_time_ms
        self.calls: list[tuple[object, ...]] = []

    def get_execution_time(self, requests: list[object], pipeline_stage: int) -> _Execution:
        if pipeline_stage != 0:
            raise AssertionError("Task 1.1.3 requires TP1/PP1 stage zero")
        self.calls.append(tuple(requests))
        return _Execution(self.model_time_ms)


def _request(
    request_id: str,
    *,
    phase: RequestPhase,
    prefill_tokens: int,
    prefill_remaining: int,
    decode_tokens: int = 864,
    decode_remaining: int = 864,
    computed: int = 0,
    output: int = 0,
) -> LiveRequestSnapshot:
    return LiveRequestSnapshot(
        request_id=request_id,
        phase=phase,
        arrival_time_s=0.0,
        actual_prefill_tokens=prefill_tokens,
        actual_prefill_remaining=prefill_remaining,
        canonical_prefill_tokens=prefill_tokens,
        canonical_prefill_remaining=prefill_remaining,
        actual_decode_tokens=decode_tokens,
        actual_decode_remaining=decode_remaining,
        canonical_decode_tokens=decode_tokens,
        canonical_decode_remaining=decode_remaining,
        actual_prefill_slo_s=0.1,
        canonical_prefill_slo_s=0.1,
        actual_decode_slo_s=0.05,
        canonical_decode_slo_s=0.05,
        num_computed_tokens=computed,
        num_output_tokens=output,
        queue_name="waiting" if phase is RequestPhase.PREFILL else "running",
    )


def _shape() -> dict[str, object]:
    snapshot = LiveStateSnapshot.build(
        (
            _request(
                "prefill-0",
                phase=RequestPhase.PREFILL,
                prefill_tokens=512,
                prefill_remaining=384,
                computed=128,
            ),
            _request(
                "decode-0",
                phase=RequestPhase.DECODE,
                prefill_tokens=1024,
                prefill_remaining=0,
                decode_remaining=854,
                computed=1034,
                output=10,
            ),
        ),
        max_num_scheduled_tokens=8192,
        captured_monotonic_s=999.0,
    )
    return build_batch_shape(snapshot, {"prefill-0": 256, "decode-0": 1})


class BatchExecutionErrorTests(unittest.TestCase):
    def test_exact_request_shape_is_preserved(self) -> None:
        shape = _shape()
        self.assertEqual(shape["total_prefill_size"], 256)
        self.assertEqual(shape["prefill_request_count"], 1)
        self.assertEqual(shape["decode_request_count"], 1)
        self.assertEqual(shape["total_scheduled_tokens"], 257)
        self.assertEqual(
            shape["prefill_request_stats"],
            [
                {
                    "request_id": "prefill-0",
                    "total_prefill_size": 512,
                    "completed_prefill_size_before": 128,
                    "remaining_prefill_size_before": 384,
                    "context_tokens_before": 128,
                    "scheduled_prefill_tokens": 256,
                }
            ],
        )
        self.assertEqual(
            shape["decode_request_stats"][0]["context_tokens_before"], 1034
        )

    def test_predictor_requests_match_gv3_batch_contract(self) -> None:
        requests = predictor_requests(_shape())
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            (
                requests[0].num_processed_tokens,
                requests[0].num_tokens_to_process,
                requests[0].is_prefill_complete,
            ),
            (128, 256, False),
        )
        self.assertEqual(
            (
                requests[1].num_processed_tokens,
                requests[1].num_tokens_to_process,
                requests[1].is_prefill_complete,
            ),
            (1034, 1, True),
        )

    def test_comparison_uses_gpu_duration_and_validates_clock(self) -> None:
        predictor = _Predictor(model_time_ms=22.5)
        rows = comparison_rows(
            [
                {
                    "batch_shape": _shape(),
                    "timing": {
                        "source": "gpu_forward",
                        "authoritative_duration_s": 0.025,
                    },
                    "state_before": {"sim_time": 1.0},
                    "state_after": {"sim_time": 1.025},
                }
            ],
            predictor=predictor,
            predictor_cache_dir=EXPECTED_CACHE,
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["recorded_batch_time_from_vidur"], 0.0225)
        self.assertAlmostEqual(rows[0]["absolute_percent_error"], 10.0)
        self.assertTrue(rows[0]["within_10_percent"])
        self.assertFalse(rows[0]["calibration_applied"])

    def test_clock_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(AssertionError, "logical clock advanced"):
            comparison_rows(
                [
                    {
                        "batch_shape": _shape(),
                        "timing": {
                            "source": "gpu_forward",
                            "authoritative_duration_s": 0.025,
                        },
                        "state_before": {"sim_time": 1.0},
                        "state_after": {"sim_time": 1.030},
                    }
                ],
                predictor=_Predictor(),
                predictor_cache_dir=EXPECTED_CACHE,
            )


def run_real_comparison(
    *,
    batch_audit: Path,
    output: Path,
    assert_margin: bool,
) -> dict[str, object]:
    config = load_config()
    actual_cache = config.predictor_cache_dir.resolve()
    if actual_cache != EXPECTED_CACHE:
        raise AssertionError(
            f"Task 1.1.3 must use {EXPECTED_CACHE}, not {actual_cache}"
        )
    predictor = _create_predictor("require_cache")
    rows = comparison_rows(
        read_batch_audit(batch_audit),
        predictor=predictor,
        predictor_cache_dir=actual_cache,
    )
    write_comparison_csv(rows, output)
    failures = [row for row in rows if not bool(row["within_10_percent"])]
    summary = {
        "batch_count": len(rows),
        "within_10_percent_count": len(rows) - len(failures),
        "outside_10_percent_count": len(failures),
        "maximum_absolute_percent_error": max(
            float(row["absolute_percent_error"]) for row in rows
        ),
        "mean_absolute_percent_error": sum(
            float(row["absolute_percent_error"]) for row in rows
        )
        / len(rows),
        "comparison_csv": str(output.resolve()),
        "predictor_cache": str(actual_cache),
        "calibration_applied": False,
        "logical_clock_matches_measured_duration": True,
    }
    if assert_margin and failures:
        failed_ids = [int(row["batch_index"]) for row in failures]
        raise AssertionError(
            f"{len(failures)}/{len(rows)} batches exceed 10% error: {failed_ids}"
        )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Task 1.1.3 batch timing test")
    parser.add_argument("--batch-audit", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assert-margin", action="store_true")
    args = parser.parse_args(argv)
    summary = run_real_comparison(
        batch_audit=args.batch_audit,
        output=args.output,
        assert_margin=bool(args.assert_margin),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
