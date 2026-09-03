from __future__ import annotations

import unittest

from vidur_vllm_real_testing.gv3_adapter import (
    GV3LiveAdapterConfig,
    GV3PersistentAdapter,
)
from vidur_vllm_real_testing.tests.test_gv3_live_adapter import (
    observation,
    request,
    snapshot,
)


class GV3ProductionAdapterTests(unittest.TestCase):
    def test_controller_eviction_updates_persistent_drop_state(self) -> None:
        calls = 0

        def planner(payload: object, live: object) -> object:
            nonlocal calls
            calls += 1
            details = (
                {"terminated_request_ids": ["req-1"]}
                if calls == 1
                else {}
            )
            return {
                "allocations": [
                    {
                        "request_id": "req-0",
                        "phase": "prefill",
                        "num_tokens": 128,
                    }
                ],
                "details": details,
            }

        adapter = GV3PersistentAdapter(controller=planner)
        first = request()
        evicted = request(request_id="req-1")
        plan = adapter.plan(snapshot(first, evicted))
        self.assertEqual(plan.details["terminated_request_ids"], ["req-1"])

        payload = adapter.state_payload()
        dropped = payload["requests"][1]
        self.assertTrue(dropped["completed"])
        self.assertTrue(dropped["dropped"])
        self.assertEqual(payload["stats"]["active_request_ids"], [0])
        self.assertEqual(payload["stats"]["completed_request_ids"], [1])
        self.assertEqual(payload["stats"]["dropped_request_ids"], [1])
        self.assertEqual(payload["stats"]["requests_completed"], 1)
        self.assertEqual(payload["stats"]["slo_lateness_sum"], 3.0)

        # vLLM removes the terminated request before the next decision.
        adapter.plan(snapshot(first, captured=2.0))

    def test_auto_drop_matches_native_terminal_and_credit_bookkeeping(self) -> None:
        adapter = GV3PersistentAdapter(
            fallback_policy="sjf-256",
            config=GV3LiveAdapterConfig(auto_drop_lateness_s=0.01, drop_cost=3.0),
        )
        initial = request(prefill_slo=0.02)
        adapter.plan(snapshot(initial))
        adapter.on_batch_completed(
            observation(
                row=initial,
                duration=0.04,
                scheduled_tokens=124,
                computed_after=124,
                output_after=0,
            )
        )

        payload = adapter.state_payload()
        state_request = payload["requests"][0]
        self.assertTrue(state_request["dropped"])
        self.assertTrue(state_request["completed"])
        self.assertFalse(state_request["violated"])
        self.assertEqual(payload["stats"]["requests_completed"], 1)
        self.assertEqual(payload["stats"]["slo_violations"], 0)
        self.assertAlmostEqual(payload["stats"]["slo_lateness_sum"], 3.0)
        self.assertEqual(payload["stats"]["decode_credit_balance"], 0)
        self.assertEqual(payload["stats"]["decode_tokens_counted_by_id"], {})

    def test_pending_drop_is_returned_before_live_state_drift_validation(self) -> None:
        adapter = GV3PersistentAdapter(
            fallback_policy="sjf-256",
            config=GV3LiveAdapterConfig(auto_drop_lateness_s=0.01, drop_cost=3.0),
        )
        initial = request(
            actual_prefill=374,
            canonical_prefill=384,
            actual_prefill_remaining=374,
            canonical_prefill_remaining=384,
            decode_total=44,
            decode_remaining=44,
            prefill_slo=0.02,
        )
        adapter.plan(snapshot(initial))
        adapter.on_batch_completed(
            observation(
                row=initial,
                duration=0.04,
                scheduled_tokens=128,
                computed_after=128,
                output_after=0,
            )
        )
        still_live = request(
            actual_prefill=374,
            canonical_prefill=384,
            actual_prefill_remaining=246,
            canonical_prefill_remaining=256,
            decode_total=44,
            decode_remaining=44,
            num_computed=128,
            prefill_slo=0.02,
        )
        plan = adapter.plan(snapshot(still_live, captured=2.0))
        self.assertEqual(plan.allocations, ())
        self.assertEqual(plan.details["terminated_request_ids"], ["req-0"])

    def test_controller_mapping_accepts_existing_phase_enum(self) -> None:
        seen: dict[str, object] = {}

        def planner(payload: object, live: object) -> object:
            seen["payload"] = payload
            return {
                "allocations": [
                    {"request_id": "req-0", "phase": request().phase, "num_tokens": 128}
                ]
            }

        adapter = GV3PersistentAdapter(controller=planner)
        plan = adapter.plan(snapshot(request()))
        self.assertEqual(plan.allocations[0].phase.value, "prefill")
        self.assertIn("payload", seen)

    def test_non_gv3_decode_cap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "864-token"):
            GV3PersistentAdapter(
                fallback_policy="sjf-256",
                config=GV3LiveAdapterConfig(max_decode_tokens_per_request=128),
            )


if __name__ == "__main__":
    unittest.main()
