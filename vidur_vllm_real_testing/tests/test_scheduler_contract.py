from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from vidur_vllm_real_testing.canonicalization import CanonicalizationError
from vidur_vllm_real_testing.scheduler_contract import (
    CanonicalAllocation,
    LiveRequestSnapshot,
    LiveStateSnapshot,
    RequestPhase,
    SJFPlanner,
    SchedulePlan,
    SchedulerMode,
    TraceArrivalGroupBarrier,
    TraceMetadataRegistry,
    validate_and_project_plan,
    exclusively_matches_request_prefix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "vidur_vllm_real_testing"


def request(
    request_id: str,
    *,
    phase: RequestPhase,
    canonical_prefill_remaining: int = 0,
    actual_prefill_remaining: int = 0,
    arrival_time_s: float = 0.0,
) -> LiveRequestSnapshot:
    return LiveRequestSnapshot(
        request_id=request_id,
        phase=phase,
        arrival_time_s=arrival_time_s,
        actual_prefill_tokens=768,
        actual_prefill_remaining=actual_prefill_remaining,
        canonical_prefill_tokens=768,
        canonical_prefill_remaining=canonical_prefill_remaining,
        actual_decode_tokens=864,
        actual_decode_remaining=864,
        canonical_decode_tokens=864,
        canonical_decode_remaining=864,
        actual_prefill_slo_s=0.2,
        canonical_prefill_slo_s=0.2,
        actual_decode_slo_s=0.05,
        canonical_decode_slo_s=0.05,
        num_computed_tokens=0,
        num_output_tokens=0,
        queue_name="waiting",
    )


def snapshot(*requests: LiveRequestSnapshot, budget: int = 4608) -> LiveStateSnapshot:
    return LiveStateSnapshot.build(
        requests,
        max_num_scheduled_tokens=budget,
        captured_monotonic_s=10.0,
    )


class SchedulerContractTests(unittest.TestCase):
    def test_warmup_prefix_requires_a_nonempty_exclusive_request_set(self) -> None:
        prefix = "cmpl-vidur-warmup-"
        self.assertTrue(
            exclusively_matches_request_prefix(
                ["cmpl-vidur-warmup-000000-0", "cmpl-vidur-warmup-000001-0"],
                prefix,
            )
        )
        self.assertFalse(exclusively_matches_request_prefix([], prefix))
        self.assertFalse(
            exclusively_matches_request_prefix(
                ["cmpl-vidur-warmup-000000-0", "cmpl-splitwise-conv-000000-0"],
                prefix,
            )
        )

    def test_scheduler_mode_aliases_are_explicit(self) -> None:
        self.assertIs(SchedulerMode.parse("sjf256"), SchedulerMode.SJF_256)
        self.assertIs(SchedulerMode.parse("SJF_512"), SchedulerMode.SJF_512)
        self.assertIs(SchedulerMode.parse("baseline"), SchedulerMode.STOCK)

    def test_sjf256_includes_all_decodes_and_shortest_prefills(self) -> None:
        state = snapshot(
            request("decode-b", phase=RequestPhase.DECODE, arrival_time_s=0.2),
            request("prefill-long", phase=RequestPhase.PREFILL, canonical_prefill_remaining=700, actual_prefill_remaining=700),
            request("decode-a", phase=RequestPhase.DECODE, arrival_time_s=0.1),
            request("prefill-short", phase=RequestPhase.PREFILL, canonical_prefill_remaining=100, actual_prefill_remaining=100),
            request("prefill-medium", phase=RequestPhase.PREFILL, canonical_prefill_remaining=500, actual_prefill_remaining=500),
        )
        plan = SJFPlanner(256).plan(state)
        self.assertEqual(
            [(row.request_id, row.phase, row.num_tokens) for row in plan.allocations],
            [
                ("decode-a", RequestPhase.DECODE, 1),
                ("decode-b", RequestPhase.DECODE, 1),
                ("prefill-short", RequestPhase.PREFILL, 100),
                ("prefill-medium", RequestPhase.PREFILL, 156),
            ],
        )

    def test_sjf512_uses_the_same_policy_with_larger_budget(self) -> None:
        state = snapshot(
            request("short", phase=RequestPhase.PREFILL, canonical_prefill_remaining=100, actual_prefill_remaining=100),
            request("medium", phase=RequestPhase.PREFILL, canonical_prefill_remaining=500, actual_prefill_remaining=500),
        )
        plan = SJFPlanner(512).plan(state)
        self.assertEqual(
            [(row.request_id, row.num_tokens) for row in plan.allocations],
            [("short", 100), ("medium", 412)],
        )

    def test_canonical_prefill_tail_is_projected_to_actual_tail(self) -> None:
        state = snapshot(
            request(
                "rounded",
                phase=RequestPhase.PREFILL,
                canonical_prefill_remaining=128,
                actual_prefill_remaining=124,
            )
        )
        plan = SchedulePlan(
            state_fingerprint=state.fingerprint,
            allocations=(CanonicalAllocation("rounded", RequestPhase.PREFILL, 128),),
            policy="test",
        )
        validated = validate_and_project_plan(plan, state)
        self.assertEqual(validated.actual_token_budget, 124)
        self.assertTrue(validated.allocations[0].truncated_to_actual_tail)

    def test_stale_plan_is_rejected(self) -> None:
        before = snapshot(request("a", phase=RequestPhase.DECODE))
        after = snapshot(
            replace(
                before.requests[0],
                num_output_tokens=1,
                actual_decode_remaining=863,
                canonical_decode_remaining=863,
            )
        )
        plan = SJFPlanner(256).plan(before)
        with self.assertRaisesRegex(CanonicalizationError, "stale scheduler plan"):
            validate_and_project_plan(plan, after)

    def test_phase_mismatch_and_multi_token_decode_are_rejected(self) -> None:
        state = snapshot(request("decode", phase=RequestPhase.DECODE))
        wrong_phase = SchedulePlan(
            state.fingerprint,
            (CanonicalAllocation("decode", RequestPhase.PREFILL, 1),),
            "test",
        )
        with self.assertRaisesRegex(CanonicalizationError, "plan phase"):
            validate_and_project_plan(wrong_phase, state)
        multi_decode = SchedulePlan(
            state.fingerprint,
            (CanonicalAllocation("decode", RequestPhase.DECODE, 2),),
            "test",
        )
        with self.assertRaisesRegex(CanonicalizationError, "one token"):
            validate_and_project_plan(multi_decode, state)

    def test_empty_plan_for_live_work_is_rejected(self) -> None:
        state = snapshot(request("decode", phase=RequestPhase.DECODE))
        plan = SchedulePlan(state.fingerprint, (), "bad")
        with self.assertRaisesRegex(CanonicalizationError, "selected no work"):
            validate_and_project_plan(plan, state)

    def test_fingerprint_is_deterministic_and_ignores_capture_clock(self) -> None:
        row = request("a", phase=RequestPhase.DECODE)
        first = LiveStateSnapshot.build(
            [row], max_num_scheduled_tokens=10, captured_monotonic_s=1.0
        )
        second = LiveStateSnapshot.build(
            [row], max_num_scheduled_tokens=10, captured_monotonic_s=999.0
        )
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_trace_registry_loads_checked_in_canonical_trace(self) -> None:
        registry = TraceMetadataRegistry.load(
            PACKAGE_DIR / "traces" / "splitwise_conv_20s_english_canonical.csv"
        )
        self.assertEqual(len(registry), 31)
        row = registry.require("splitwise-conv-000000")
        self.assertGreater(row.canonical_prefill_tokens, 0)
        wrapped = registry.require("cmpl-splitwise-conv-000000-0")
        self.assertEqual(wrapped.request_id, row.request_id)
        wrapped_v026 = registry.require(
            "cmpl-splitwise-conv-000000-0-95437f26"
        )
        self.assertEqual(wrapped_v026.request_id, row.request_id)
        with self.assertRaisesRegex(CanonicalizationError, "no canonical trace metadata"):
            registry.require("missing")
        with self.assertRaisesRegex(CanonicalizationError, "no canonical trace metadata"):
            registry.require("cmpl-missing-0")
        with self.assertRaisesRegex(CanonicalizationError, "no canonical trace metadata"):
            registry.require("cmpl-splitwise-conv-000000-not-an-index-95437f26")

    def test_equal_time_trace_arrivals_are_released_atomically(self) -> None:
        registry = TraceMetadataRegistry.load(
            PACKAGE_DIR
            / "traces"
            / "gv3_legal_20s_exact_grid_mew1_a100_canonical.csv"
        )
        barrier = TraceArrivalGroupBarrier(registry, timeout_s=1.0)

        missing = barrier.pending_missing(
            ["cmpl-exact-grid-000000-0"],
            now_s=10.0,
        )
        self.assertEqual(
            missing[0.0],
            tuple(f"exact-grid-{index:06d}" for index in range(1, 7)),
        )

        first_group = [
            f"cmpl-exact-grid-{index:06d}-0" for index in range(7)
        ]
        self.assertEqual(
            barrier.pending_missing(first_group, now_s=10.1),
            {},
        )
        self.assertEqual(
            barrier.pending_missing(first_group[:1], now_s=20.0),
            {},
        )

    def test_atomic_arrival_barrier_fails_if_a_peer_never_arrives(self) -> None:
        registry = TraceMetadataRegistry.load(
            PACKAGE_DIR
            / "traces"
            / "gv3_legal_20s_exact_grid_mew1_a100_canonical.csv"
        )
        barrier = TraceArrivalGroupBarrier(registry, timeout_s=0.5)
        visible = ["cmpl-exact-grid-000008-0"]
        self.assertIn(2.4, barrier.pending_missing(visible, now_s=1.0))
        with self.assertRaisesRegex(
            CanonicalizationError,
            "timed out waiting for atomic trace-arrival group",
        ):
            barrier.pending_missing(visible, now_s=1.6)

    def test_trace_registry_rejects_wrong_schema(self) -> None:
        source = PACKAGE_DIR / "traces" / "splitwise_conv_20s_english_canonical.csv"
        with tempfile.TemporaryDirectory() as temp_dir:
            changed = Path(temp_dir) / "changed.csv"
            changed.write_text(
                source.read_text(encoding="utf-8").replace(
                    "gv3_vllm_trace_v1", "wrong", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CanonicalizationError, "schema_version"):
                TraceMetadataRegistry.load(changed)


if __name__ == "__main__":
    unittest.main()
