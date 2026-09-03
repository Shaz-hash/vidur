from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from vidur.AlphaGoZero.test_and_analysis.candidate_prefill_q_trend import analyze_block


class CandidatePrefillQTrendTest(unittest.TestCase):
    def test_explicit_pending_prefill_filter_and_rankings(self) -> None:
        fields = [
            "player_acted",
            "prefill_remaining_by_id",
            "canonical_action_count",
            "action_repr",
            "candidate_top5_action_reprs",
            "candidate_top5_q_values",
            "candidate_top5_visits",
            "candidate_top5_priors",
        ]
        decode = "ControllerAction(prefill_allocations={}, decode_allocations={1: 1})"
        prefill = "ControllerAction(prefill_allocations={2: 128}, decode_allocations={1: 1})"
        rows = [
            {
                "player_acted": "controller",
                "prefill_remaining_by_id": "{}",
                "canonical_action_count": "4",
                "action_repr": decode,
                "candidate_top5_action_reprs": json.dumps([decode, prefill]),
                "candidate_top5_q_values": json.dumps([-0.1, -0.2]),
                "candidate_top5_visits": json.dumps([900, 100]),
                "candidate_top5_priors": json.dumps([0.9, 0.1]),
            },
            {
                "player_acted": "controller",
                "prefill_remaining_by_id": json.dumps({"2": 128}),
                "canonical_action_count": "4",
                "action_repr": prefill,
                "candidate_top5_action_reprs": json.dumps([prefill, decode]),
                "candidate_top5_q_values": json.dumps([-0.1, -0.3]),
                "candidate_top5_visits": json.dumps([700, 300]),
                "candidate_top5_priors": json.dumps([0.2, 0.8]),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            arena = Path(directory) / "arena_games"
            arena.mkdir()
            path = arena / "game_1_model_ctrl_depth1.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            summary = analyze_block(Path(directory))

        self.assertEqual(summary["controller_rows"], 2)
        self.assertEqual(summary["no_pending_rows"], 1)
        self.assertEqual(summary["pending_prefill_rows"], 1)
        self.assertEqual(summary["chosen_prefill_pct"], 100.0)
        self.assertEqual(summary["q_favors_prefill_pct"], 100.0)
        self.assertEqual(summary["prior_favors_prefill_pct"], 0.0)
        self.assertEqual(summary["top_visit_prefill_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
