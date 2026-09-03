from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vidur_vllm_real_testing.real_trace_benchmark import (
    BlockingInterval,
    RequestObservation,
    TraceRow,
    blocked_between,
    request_metrics,
    scheduler_blocking_total,
)


class RealTraceBenchmarkTests(unittest.TestCase):
    def test_in_progress_planning_freezes_logical_arrivals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "scheduler.jsonl"
            log_path.write_text(
                json.dumps({"controller_blocking_total_s": 3.0}) + "\n"
            )
            planning_path = Path(f"{log_path}.planning")
            planning_path.write_text(
                json.dumps({
                    "started_monotonic_s": 15.0,
                    "blocking_total_before_s": 3.0,
                })
            )
            with patch(
                "vidur_vllm_real_testing.real_trace_benchmark.time.monotonic",
                return_value=20.0,
            ):
                self.assertEqual(scheduler_blocking_total(log_path), 8.0)

    def test_blocking_overlap_is_request_scoped(self) -> None:
        intervals = [
            BlockingInterval(10.1, 10.3, frozenset({"a"})),
            BlockingInterval(10.4, 10.5, frozenset({"b"})),
        ]
        self.assertAlmostEqual(
            blocked_between(intervals, "a", 10.0, 10.6), 0.2
        )
        self.assertAlmostEqual(
            blocked_between(intervals, "b", 10.0, 10.6), 0.1
        )

    def test_corrected_slo_excludes_controller_blocking(self) -> None:
        row = TraceRow(
            "a", 0.0, 128, 3, 0.1, 0.05, "prompt.json", 1, True
        )
        observation = RequestObservation(
            row=row,
            submitted_monotonic_s=10.0,
            token_monotonic_s=(10.25, 10.36, 10.42, 10.48),
            http_completed_monotonic_s=10.43,
            dispatch_lag_s=0.0,
            response_completion_tokens=3,
            error=None,
        )
        intervals = [
            BlockingInterval(10.05, 10.20, frozenset({"a"})),
            BlockingInterval(10.30, 10.36, frozenset({"a"})),
        ]
        metrics = request_metrics(observation, intervals)
        self.assertEqual(metrics["tokens_observed"], 3)
        self.assertEqual(metrics["physical_output_tokens_observed"], 4)
        self.assertAlmostEqual(float(metrics["raw_ttft_s"]), 0.25)
        self.assertAlmostEqual(
            float(metrics["corrected_ttft_s"]), 0.10
        )
        self.assertAlmostEqual(
            float(metrics["raw_total_lateness_s"]), 0.23
        )
        self.assertAlmostEqual(
            float(metrics["corrected_total_lateness_s"]), 0.02
        )


if __name__ == "__main__":
    unittest.main()
