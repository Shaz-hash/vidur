from __future__ import annotations

import unittest

from vidur_vllm_real_testing.canonicalization import CanonicalizationError
from vidur_vllm_real_testing.gv3_live_adapter import (
    BatchExecutionObservation,
    BatchRequestProgress,
    GV3LiveAdapterConfig,
    GV3PersistentAdapter,
)
from vidur_vllm_real_testing.scheduler_contract import (
    LiveRequestSnapshot,
    LiveStateSnapshot,
    RequestPhase,
)


def request(
    *,
    request_id: str = "req-0",
    phase: RequestPhase = RequestPhase.PREFILL,
    actual_prefill: int = 124,
    canonical_prefill: int = 128,
    actual_prefill_remaining: int | None = None,
    canonical_prefill_remaining: int | None = None,
    decode_total: int = 216,
    decode_remaining: int | None = None,
    num_computed: int = 0,
    num_output: int = 0,
    arrival: float = 0.0,
    prefill_slo: float = 0.2,
    decode_slo: float = 0.05,
) -> LiveRequestSnapshot:
    return LiveRequestSnapshot(
        request_id=request_id,
        phase=phase,
        arrival_time_s=arrival,
        actual_prefill_tokens=actual_prefill,
        actual_prefill_remaining=(
            actual_prefill if actual_prefill_remaining is None else actual_prefill_remaining
        ),
        canonical_prefill_tokens=canonical_prefill,
        canonical_prefill_remaining=(
            canonical_prefill
            if canonical_prefill_remaining is None
            else canonical_prefill_remaining
        ),
        actual_decode_tokens=decode_total,
        actual_decode_remaining=(decode_total if decode_remaining is None else decode_remaining),
        canonical_decode_tokens=decode_total,
        canonical_decode_remaining=(decode_total if decode_remaining is None else decode_remaining),
        actual_prefill_slo_s=prefill_slo,
        canonical_prefill_slo_s=prefill_slo,
        actual_decode_slo_s=decode_slo,
        canonical_decode_slo_s=decode_slo,
        num_computed_tokens=num_computed,
        num_output_tokens=num_output,
        queue_name="running",
    )


def snapshot(*rows: LiveRequestSnapshot, captured: float = 1.0) -> LiveStateSnapshot:
    return LiveStateSnapshot.build(
        rows,
        max_num_scheduled_tokens=4608,
        captured_monotonic_s=captured,
    )


def observation(
    *,
    row: LiveRequestSnapshot,
    duration: float,
    scheduled_tokens: int,
    computed_after: int,
    output_after: int,
    finished: bool = False,
) -> BatchExecutionObservation:
    return BatchExecutionObservation(
        scheduled_monotonic_s=10.0,
        completed_monotonic_s=10.0 + duration,
        duration_s=duration,
        scheduled_tokens_by_request={row.request_id: scheduled_tokens},
        request_progress=(
            BatchRequestProgress(
                request_id=row.request_id,
                phase_before=row.phase,
                num_computed_tokens_before=row.num_computed_tokens,
                num_output_tokens_before=row.num_output_tokens,
                num_computed_tokens_after=computed_after,
                num_output_tokens_after=output_after,
                finished_after=finished,
            ),
        ),
    )


class GV3PersistentAdapterTests(unittest.TestCase):
    def test_request_visible_between_batches_waits_for_canonical_arrival(self) -> None:
        adapter = GV3PersistentAdapter(fallback_policy="sjf-256")
        initial = request(actual_prefill=512, canonical_prefill=512)
        adapter.plan(snapshot(initial))
        adapter.on_batch_completed(
            observation(
                row=initial,
                duration=0.1,
                scheduled_tokens=256,
                computed_after=256,
                output_after=0,
            )
        )

        current = request(
            actual_prefill=512,
            canonical_prefill=512,
            actual_prefill_remaining=256,
            canonical_prefill_remaining=256,
            num_computed=256,
        )
        future = request(request_id="req-1", arrival=0.12)
        before_arrival = adapter.plan(snapshot(current, future, captured=2.0))
        self.assertEqual(
            [allocation.request_id for allocation in before_arrival.allocations],
            ["req-0"],
        )
        self.assertEqual(len(adapter.state_payload()["requests"]), 1)

        adapter.on_batch_completed(
            observation(
                row=current,
                duration=0.03,
                scheduled_tokens=256,
                computed_after=512,
                output_after=0,
            )
        )
        decode = request(
            phase=RequestPhase.DECODE,
            actual_prefill=512,
            canonical_prefill=512,
            actual_prefill_remaining=0,
            canonical_prefill_remaining=0,
            num_computed=512,
        )
        after_arrival = adapter.plan(snapshot(decode, future, captured=3.0))
        self.assertIn(
            "req-1",
            [allocation.request_id for allocation in after_arrival.allocations],
        )
        self.assertEqual(len(adapter.state_payload()["requests"]), 2)

    def test_nearest_down_prefill_keeps_actual_tail_schedulable(self) -> None:
        adapter = GV3PersistentAdapter(fallback_policy="sjf-256")
        initial = request(actual_prefill=396, canonical_prefill=384)
        adapter.plan(snapshot(initial))
        adapter.on_batch_completed(
            observation(
                row=initial,
                duration=0.04,
                scheduled_tokens=384,
                computed_after=384,
                output_after=0,
            )
        )
        state_request = adapter.state_payload()["requests"][0]
        self.assertFalse(state_request["is_prefill_complete"])
        self.assertEqual(state_request["num_prefill_tokens"], 512)
        self.assertEqual(state_request["num_processed_prefill_tokens"], 384)

        tail = request(
            actual_prefill=396,
            canonical_prefill=384,
            actual_prefill_remaining=12,
            canonical_prefill_remaining=128,
            num_computed=384,
        )
        tail_plan = adapter.plan(snapshot(tail, captured=2.0))
        self.assertEqual(tail_plan.allocations[0].num_tokens, 128)
        adapter.on_batch_completed(
            observation(
                row=tail,
                duration=0.01,
                scheduled_tokens=12,
                computed_after=396,
                output_after=0,
            )
        )
        self.assertTrue(
            adapter.state_payload()["requests"][0]["is_prefill_complete"]
        )

    def test_prefill_output_does_not_consume_decode_credit(self) -> None:
        adapter = GV3PersistentAdapter(fallback_policy="sjf-256")
        initial = request()
        plan = adapter.plan(snapshot(initial))
        self.assertEqual(plan.allocations[0].num_tokens, 128)

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
        self.assertEqual(state_request["num_processed_prefill_tokens"], 128)
        self.assertEqual(state_request["num_processed_decode_tokens"], 0)
        self.assertEqual(payload["stats"]["decode_credit_balance"], 216)
        self.assertAlmostEqual(state_request["prefill_completed_at"], 0.04)

        decode = request(
            phase=RequestPhase.DECODE,
            actual_prefill_remaining=0,
            canonical_prefill_remaining=0,
            decode_remaining=216,
            num_computed=124,
            num_output=0,
        )
        decode_plan = adapter.plan(snapshot(decode, captured=2.0))
        self.assertEqual(decode_plan.policy, "gv3-real-decode-fast-forward")
        self.assertEqual(len(decode_plan.allocations), 1)

    def test_decode_fast_forward_repeats_and_advances_past_tick(self) -> None:
        adapter = GV3PersistentAdapter(fallback_policy="sjf-256")
        initial = request()
        adapter.plan(snapshot(initial))
        adapter.on_batch_completed(
            observation(
                row=initial,
                duration=0.18,
                scheduled_tokens=124,
                computed_after=124,
                output_after=0,
            )
        )
        first_decode = request(
            phase=RequestPhase.DECODE,
            actual_prefill_remaining=0,
            canonical_prefill_remaining=0,
            decode_remaining=216,
            num_computed=124,
            num_output=0,
        )
        plan = adapter.plan(snapshot(first_decode, captured=2.0))
        self.assertEqual(plan.policy, "gv3-real-decode-fast-forward")
        adapter.on_batch_completed(
            observation(
                row=first_decode,
                duration=0.03,
                scheduled_tokens=1,
                computed_after=125,
                output_after=1,
            )
        )
        self.assertAlmostEqual(adapter.sim_time, 0.21)

        second_decode = request(
            phase=RequestPhase.DECODE,
            actual_prefill_remaining=0,
            canonical_prefill_remaining=0,
            decode_remaining=215,
            num_computed=125,
            num_output=1,
        )
        next_plan = adapter.plan(snapshot(second_decode, captured=3.0))
        self.assertEqual(next_plan.policy, "gv3-real-decode-fast-forward")
        self.assertAlmostEqual(adapter.state_payload()["stats"]["last_adv_tick"], 0.2)
        self.assertAlmostEqual(adapter.state_payload()["stats"]["next_adv_tick"], 0.4)

    def test_lateness_uses_completed_batch_end_time(self) -> None:
        adapter = GV3PersistentAdapter(fallback_policy="sjf-256")
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
        self.assertAlmostEqual(payload["requests"][0]["prefill_lateness"], 0.02)
        self.assertEqual(payload["stats"]["slo_violations"], 1)

        decode = request(
            phase=RequestPhase.DECODE,
            actual_prefill_remaining=0,
            canonical_prefill_remaining=0,
            decode_remaining=216,
            num_computed=124,
            num_output=0,
            prefill_slo=0.02,
        )
        adapter.plan(snapshot(decode, captured=2.0))
        adapter.on_batch_completed(
            observation(
                row=decode,
                duration=0.06,
                scheduled_tokens=1,
                computed_after=125,
                output_after=1,
            )
        )
        payload = adapter.state_payload()
        self.assertAlmostEqual(payload["requests"][0]["decode_lateness"], 0.01)
        self.assertAlmostEqual(payload["stats"]["slo_lateness_sum"], 0.03)

    def test_queue_snapshot_cannot_silently_replace_persistent_progress(self) -> None:
        adapter = GV3PersistentAdapter(fallback_policy="sjf-256")
        initial = request()
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
        with self.assertRaisesRegex(CanonicalizationError, "prefill drift"):
            adapter.plan(snapshot(initial, captured=2.0))

    def test_midstream_initialization_is_rejected_by_default(self) -> None:
        adapter = GV3PersistentAdapter(fallback_policy="sjf-256")
        midstream = request(
            actual_prefill_remaining=64,
            canonical_prefill_remaining=68,
            num_computed=60,
        )
        with self.assertRaisesRegex(CanonicalizationError, "must start before"):
            adapter.plan(snapshot(midstream))

    def test_native_payload_contains_markov_ledger_fields(self) -> None:
        seen: dict[str, object] = {}

        def planner(payload: object, live: LiveStateSnapshot) -> object:
            seen["payload"] = payload
            return {
                "state_fingerprint": live.fingerprint,
                "policy": "test-controller",
                "allocations": [
                    {"request_id": "req-0", "phase": "prefill", "num_tokens": 128}
                ],
            }

        adapter = GV3PersistentAdapter(controller=planner)
        adapter.plan(snapshot(request()))
        payload = seen["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(payload["sim_time"], 0.0)
        self.assertEqual(payload["stats"]["next_adv_tick"], 0.2)
        self.assertEqual(payload["stats"]["decode_credit_balance"], 0)
        self.assertEqual(payload["requests"][0]["prefill_deadline"], 0.2)

    def test_invalid_batch_duration_is_rejected(self) -> None:
        adapter = GV3PersistentAdapter(
            fallback_policy="sjf-256",
            config=GV3LiveAdapterConfig(),
        )
        initial = request()
        adapter.plan(snapshot(initial))
        with self.assertRaisesRegex(CanonicalizationError, "duration"):
            adapter.on_batch_completed(
                observation(
                    row=initial,
                    duration=0.0,
                    scheduled_tokens=124,
                    computed_after=124,
                    output_after=0,
                )
            )


if __name__ == "__main__":
    unittest.main()

