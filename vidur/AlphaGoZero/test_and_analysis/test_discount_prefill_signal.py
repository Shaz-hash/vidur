from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from vidur.AlphaGoZero.test_and_analysis.discount_prefill_signal import (
    DEFAULT_DISCOUNT_DENOM_SEC,
    analyze,
)


class DiscountPrefillSignalTest(unittest.TestCase):
    def test_higher_gamma_preserves_identical_delayed_cost(self) -> None:
        fields = [
            "phase",
            "player_acted",
            "sim_time_before",
            "sim_time_after",
            "total_cost",
            "chosen_reward",
            "prefill_remaining_by_id",
            "action_repr",
            "candidate_top5_action_reprs",
            "candidate_top5_q_values",
            "candidate_top5_rewards",
            "candidate_top5_discounts",
        ]
        decode = "ControllerAction(prefill_allocations={}, decode_allocations={1: 1})"
        prefill = "ControllerAction(prefill_allocations={2: 128}, decode_allocations={1: 1})"
        rows = [
            {
                "phase": "arena_step",
                "player_acted": "controller",
                "sim_time_before": 0.0,
                "sim_time_after": 0.1,
                "total_cost": 0.0,
                "chosen_reward": 0.0,
                "prefill_remaining_by_id": json.dumps({"2": 128}),
                "action_repr": decode,
                "candidate_top5_action_reprs": json.dumps([prefill, decode]),
                "candidate_top5_q_values": json.dumps([-0.2, -0.1]),
                "candidate_top5_rewards": json.dumps([0.0, 0.0]),
                "candidate_top5_discounts": json.dumps([0.9, 0.9]),
            },
            {
                "phase": "arena_step",
                "player_acted": "adversary",
                "sim_time_before": 0.1,
                "sim_time_after": 0.1 + DEFAULT_DISCOUNT_DENOM_SEC,
                "total_cost": 1.0,
                "chosen_reward": -1.0,
                "prefill_remaining_by_id": json.dumps({"2": 128}),
                "action_repr": "AdversaryAction()",
                "candidate_top5_action_reprs": "[]",
                "candidate_top5_q_values": "[]",
                "candidate_top5_rewards": "[]",
                "candidate_top5_discounts": "[]",
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
            result = analyze(
                Path(directory),
                source_gamma=0.9,
                gammas=(0.9, 0.98, 0.995),
                discount_denom_sec=DEFAULT_DISCOUNT_DENOM_SEC,
                value_rmse=0.75,
            )

        self.assertEqual(result["pending_controller_states"], 1)
        self.assertEqual(result["chosen_decode_states"], 1)
        self.assertEqual(result["decode_immediate_reward_zero_pct"], 100.0)
        self.assertLess(
            result["gamma_0_9_decode_realized_return_abs_median"],
            result["gamma_0_98_decode_realized_return_abs_median"],
        )
        self.assertLess(
            result["gamma_0_98_decode_realized_return_abs_median"],
            result["gamma_0_995_decode_realized_return_abs_median"],
        )
        self.assertEqual(
            result["gamma_0_9_decode_abs_return_below_value_rmse_pct"], 100.0
        )
        self.assertEqual(
            result["gamma_0_995_decode_abs_return_below_value_rmse_pct"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
