from __future__ import annotations

import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from vidur.AlphaGoZero.adaptive_rollout import (
    activate_manifest_horizon,
    calculate_rollout_horizon,
    read_runtime_search_config,
    round_to_nearest_tick,
)
from vidur.AlphaGoZero import xl_coordinator


class AdaptiveRolloutTests(unittest.TestCase):
    def test_rounds_half_up_to_adversary_tick(self) -> None:
        self.assertEqual(round_to_nearest_tick(2.46, 0.2), 2.4)
        self.assertEqual(round_to_nearest_tick(2.78, 0.2), 2.8)
        self.assertEqual(round_to_nearest_tick(2.50, 0.2), 2.6)

    def test_calculation_caps_and_rounds(self) -> None:
        gamma = 0.98
        threshold = 0.09
        reference = 0.015725797204323228

        def p95_for_horizon(horizon: float) -> float:
            return threshold / math.pow(gamma, horizon / reference)

        low = calculate_rollout_horizon(
            p95_for_horizon(2.46),
            value_error_threshold=threshold,
            discount_factor=gamma,
            reference_step_sec=reference,
            max_horizon_sec=3.0,
            tick_sec=0.2,
        )
        high = calculate_rollout_horizon(
            p95_for_horizon(2.78),
            value_error_threshold=threshold,
            discount_factor=gamma,
            reference_step_sec=reference,
            max_horizon_sec=3.0,
            tick_sec=0.2,
        )
        capped = calculate_rollout_horizon(
            4.5,
            value_error_threshold=threshold,
            discount_factor=gamma,
            reference_step_sec=reference,
            max_horizon_sec=3.0,
            tick_sec=0.2,
        )
        accurate = calculate_rollout_horizon(
            0.05,
            value_error_threshold=threshold,
            discount_factor=gamma,
            reference_step_sec=reference,
            max_horizon_sec=3.0,
            tick_sec=0.2,
        )

        self.assertAlmostEqual(low.calculated_horizon_sec, 2.46, places=10)
        self.assertEqual(low.rounded_horizon_sec, 2.4)
        self.assertAlmostEqual(high.calculated_horizon_sec, 2.78, places=10)
        self.assertEqual(high.rounded_horizon_sec, 2.8)
        self.assertEqual(capped.calculated_horizon_sec, 3.0)
        self.assertEqual(capped.rounded_horizon_sec, 3.0)
        self.assertEqual(accurate.calculated_horizon_sec, 0.0)
        self.assertEqual(accurate.rounded_horizon_sec, 0.0)

    def test_manifest_activation_targets_next_candidate(self) -> None:
        manifest = {
            "next_rollout_horizon_rounded_sec": 2.4,
            "rollout_max_horizon_sec": 3.0,
            "rollout_horizon_source_controller_p95_abs_error": 2.1,
            "rollout_value_error_threshold": 0.09,
            "rollout_discount_factor": 0.98,
            "rollout_reference_step_sec": 0.015725797204323228,
            "rollout_horizon_tick_sec": 0.2,
            "next_rollout_horizon_raw_sec": 2.46,
            "next_rollout_horizon_calculated_sec": 2.46,
            "next_rollout_discounted_error": 0.09,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = activate_manifest_horizon(
                root,
                candidate_version=101,
                manifest=manifest,
            )
            loaded = read_runtime_search_config(root)

        self.assertEqual(payload["source_candidate_version"], 101)
        self.assertEqual(payload["target_candidate_version"], 102)
        self.assertEqual(loaded["active_rollout_horizon_sec"], 2.4)


    def test_spot_scheduler_tracks_active_adaptive_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(xl_coordinator, "active_rollout_horizon_sec", return_value=2.4):
                config = xl_coordinator._refresh_spot_scheduler(
                    root,
                    selfplay_limit=1200,
                    eval_limit=600,
                    selfplay_lease_sec=300,
                    worker_heartbeat_timeout_sec=300,
                )
        self.assertEqual(config["selfplay_total_parallel_games"], 1200)
        self.assertEqual(config["eval_total_parallel_games"], 600)
        self.assertEqual(
            config["selfplay_config"]["rollout_horizon_sec"],
            2.4,
        )

if __name__ == "__main__":
    unittest.main()
