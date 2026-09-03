"""Tests for delayed terminal bootstrap with an early replay sample window."""

from __future__ import annotations

import csv
import importlib
import math
import tempfile
import unittest
from pathlib import Path

from vidur.AlphaGoZero.replay_runtime import (
    AlphaGoZeroReplayRecorder,
    discounted_trajectory_targets,
)


CYCLE = "model_adv_depth1_vs_model_ctrl_depth1"


class DelayedTerminalBootstrapTests(unittest.TestCase):
    def _record(
        self,
        recorder: AlphaGoZeroReplayRecorder,
        *,
        turn: int,
        before: float,
        after: float,
        reward: float,
        discount: float,
    ) -> None:
        recorder.record_transition(
            game_id=17,
            cycle_label=CYCLE,
            phase="arena_step",
            turn=turn,
            depth=1,
            player="controller",
            sim_time_before=before,
            sim_time_after=after,
            total_cost_after=0.0,
            selection_info={
                "canonical_action_count": 3,
                "valid_action_count": 3,
                "chosen_reward": reward,
                "chosen_discount": discount,
            },
        )

    def test_post_window_rewards_affect_retained_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.csv"
            recorder = AlphaGoZeroReplayRecorder(
                path,
                sample_window_sec=5.0,
                discount_factor=0.99,
            )
            transitions = [
                (10.0, 11.0, -1.0, 0.9),
                (14.9, 15.0, -2.0, 0.8),
                (15.1, 16.0, -3.0, 0.7),
                (20.0, 22.0, -4.0, 0.6),
            ]
            for turn, transition in enumerate(transitions, start=1):
                self._record(
                    recorder,
                    turn=turn,
                    before=transition[0],
                    after=transition[1],
                    reward=transition[2],
                    discount=transition[3],
                )

            written = recorder.finish_cycle(
                game_id=17,
                cycle_label=CYCLE,
                terminal_bootstrap_value=5.0,
            )
            self.assertEqual(written, 2)

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            expected = discounted_trajectory_targets(
                [-1.0, -2.0, -3.0, -4.0],
                [0.9, 0.8, 0.7, 0.6],
                5.0,
            )
            self.assertEqual([float(row["time_at_state"]) for row in rows], [10.0, 14.9])
            self.assertAlmostEqual(float(rows[0]["target_value"]), expected[0])
            self.assertAlmostEqual(float(rows[1]["target_value"]), expected[1])
            self.assertEqual(float(rows[0]["trajectory_start_time"]), 10.0)
            self.assertEqual(float(rows[0]["trajectory_terminal_time"]), 22.0)
            self.assertEqual(float(rows[0]["replay_sample_window_sec"]), 5.0)
            self.assertEqual(float(rows[0]["terminal_bootstrap_value"]), 5.0)
            self.assertEqual(rows[0]["target_backup_backend"], "python")

            early_only = discounted_trajectory_targets(
                [-1.0, -2.0],
                [0.9, 0.8],
                5.0,
            )
            self.assertNotAlmostEqual(expected[0], early_only[0])

    def test_zero_window_preserves_legacy_emit_all_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.csv"
            recorder = AlphaGoZeroReplayRecorder(path, sample_window_sec=0.0)
            self._record(recorder, turn=1, before=7.0, after=8.0, reward=-1.0, discount=0.9)
            self._record(recorder, turn=2, before=20.0, after=21.0, reward=-2.0, discount=0.8)
            self.assertEqual(
                recorder.finish_cycle(
                    game_id=17,
                    cycle_label=CYCLE,
                    terminal_bootstrap_value=0.0,
                ),
                2,
            )

    def test_invalid_backup_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            discounted_trajectory_targets([1.0], [], 0.0)
        with self.assertRaises(ValueError):
            discounted_trajectory_targets([1.0], [1.1], 0.0)
        with self.assertRaises(ValueError):
            discounted_trajectory_targets([1.0], [0.9], math.nan)
        with self.assertRaises(ValueError):
            AlphaGoZeroReplayRecorder(Path("unused.csv"), sample_window_sec=-1.0)

    def test_native_backup_matches_python_when_native_module_is_available(self) -> None:
        try:
            native = importlib.import_module("mcts_native_gv2")
        except ImportError:
            self.skipTest("native module is not built")
        native_fn = getattr(native, "discounted_trajectory_targets", None)
        if native_fn is None:
            self.skipTest("native module predates delayed-bootstrap support")
        rewards = [-0.25, -3.0, 1.5, -0.75]
        discounts = [0.991, 0.75, 0.999, 0.4]
        expected = discounted_trajectory_targets(rewards, discounts, -7.25)
        actual = list(native_fn(rewards, discounts, -7.25))
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=14)


if __name__ == "__main__":
    unittest.main()
