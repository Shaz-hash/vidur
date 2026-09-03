from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vidur_vllm_real_testing.persistent_run_summary import summarize


class PersistentRunSummaryTests(unittest.TestCase):
    def test_final_state_defines_authoritative_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batches.jsonl"
            rows = [
                {"state_after": {"sim_time": 1.0, "stats": {}}, "timing": {"source": "gpu_forward"}},
                {
                    "state_after": {
                        "sim_time": 2.5,
                        "stats": {
                            "requests_generated": 3,
                            "requests_completed": 3,
                            "active_request_ids": [],
                            "stopped_decode_request_ids": [1],
                            "dropped_request_ids": [2],
                            "slo_violations": 2,
                            "slo_lateness_sum": 0.75,
                        },
                    },
                    "timing": {"source": "gpu_forward"},
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = summarize(path, policy="sjf256")
            self.assertEqual(result["batch_count"], 2)
            self.assertEqual(result["total_slo_cost"], 2.75)
            self.assertEqual(result["active_request_count"], 0)
            self.assertEqual(result["timing_source"], "gpu_forward")


if __name__ == "__main__":
    unittest.main()
