"""Regression tests for fixed-trace launch history and strict GV3 legality."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from vidur.AlphaGoZero.markov_value_features import (
    LAUNCH_FEATURE_NAMES,
    REQUEST_FEATURE_NAMES,
    MarkovValueFeatures,
    build_markov_value_features,
)
from vidur.Game_Version3.Model_Tester.Synthetic_Trace_Tester.fixed_trace_eval_runner import (
    FixedTraceEvalConfig,
    FixedTraceEvalRunner,
)
from vidur.Game_Version3.tests import native_logger_tests as nlt


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRACE_ROOT = (
    _REPO_ROOT
    / "simulator_output"
    / "GV3_Agent"
    / "Synthetic_Trace_Tester"
    / "arena_like_trace_20s"
)
_LEGAL_TRACE = _TRACE_ROOT / "gv3_legal_homogeneous_20s_decode864.csv"
_OLD_DECODE216_TRACE = _TRACE_ROOT / "arena_like_diverse_20s_decode216.csv"


def _native_features(payload: dict[str, Any]) -> MarkovValueFeatures:
    from vidur.Game_Version3_Cpp import mcts_native_gv2

    runtime = mcts_native_gv2.NewFeatures226HGBRuntime()
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


class FixedTraceMarkovContractTest(unittest.TestCase):
    def _runner(self, trace_csv: Path, output_dir: Path) -> FixedTraceEvalRunner:
        return FixedTraceEvalRunner(
            FixedTraceEvalConfig(
                trace_csv=trace_csv,
                output_dir=output_dir,
                policy="sjf512",
                time_limit_sec=20.0,
                token_policy="raw",
                sjf_budget_tokens=256,
                record_launch_history=True,
                strict_gv3_trace=True,
            )
        )

    def test_legal_launch_is_recorded_and_python_native_features_match(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gv3_trace_contract_") as tmp:
            runner = self._runner(_LEGAL_TRACE, Path(tmp))
            try:
                state = runner.bundle.env.initial_state()
                state = runner.bundle.env.prepare_trace_controller_turn(state, inplace=True)
                state, info = runner._inject_due_rows(state, due_time=0.0)

                self.assertEqual(info["recorded_launches"], [(0.0, 7, 7168)])
                self.assertEqual(
                    runner.bundle.env._v2_get_recent_launches(state),
                    [(0.0, 7, 7168)],
                )

                payload = nlt._native_state_payload(runner.bundle.env, state)
                self.assertEqual(payload["stats"]["recent_launches"], [(0.0, 7, 7168)])
                python_features = build_markov_value_features(payload)
                native_features = _native_features(payload)
                np.testing.assert_allclose(
                    python_features.global_features,
                    native_features.global_features,
                    rtol=0.0,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    python_features.request_features,
                    native_features.request_features,
                    rtol=0.0,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    python_features.launch_features,
                    native_features.launch_features,
                    rtol=0.0,
                    atol=1e-7,
                )
                self.assertEqual(tuple(python_features.request_ids), tuple(native_features.request_ids))
                self.assertEqual(python_features.launch_features.shape, (1, 3))
                self.assertAlmostEqual(float(python_features.global_features[15]), 1.0)
                self.assertAlmostEqual(float(python_features.global_features[16]), 1.0)

                state.simulator._set_time(0.2)
                pending_payload = nlt._native_state_payload(runner.bundle.env, state)
                self.assertTrue(pending_payload["stats"]["pending_adv_tick"])
                pending_python_features = build_markov_value_features(pending_payload)
                pending_native_features = _native_features(pending_payload)
                self.assertEqual(float(pending_python_features.global_features[11]), 1.0)
                self.assertEqual(float(pending_native_features.global_features[11]), 1.0)
                actions, mask = runner.bundle.env.sample_adversary_actions(state)
                valid_new_launches = sum(
                    1
                    for action, valid in zip(actions, mask)
                    if bool(valid) and action is not None and bool(action.requests)
                )
                self.assertEqual(valid_new_launches, 0)
            finally:
                runner.close()

    def test_decode216_trace_is_rejected_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gv3_trace_reject_") as tmp:
            runner = self._runner(_OLD_DECODE216_TRACE, Path(tmp))
            try:
                state = runner.bundle.env.initial_state()
                state = runner.bundle.env.prepare_trace_controller_turn(state, inplace=True)
                with self.assertRaisesRegex(ValueError, "native adversary decode size"):
                    runner._inject_due_rows(state, due_time=0.0)
                self.assertEqual(int(state.stats.requests_generated), 0)
                self.assertFalse(state.stats.active_request_ids)
                self.assertFalse(runner.bundle.env._v2_get_recent_launches(state))
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
