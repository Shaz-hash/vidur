from __future__ import annotations

from pathlib import Path
import unittest

from vidur_vllm_real_testing.canonicalization import CanonicalizationError
from vidur_vllm_real_testing.scheduler_contract import RequestPhase, TraceMetadataRegistry
from vidur_vllm_real_testing.vllm_live_state import build_live_state_snapshot


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE = REPO_ROOT / "vidur_vllm_real_testing" / "traces" / "example_raw_trace.csv"
CANONICAL_TRACE = (
    REPO_ROOT
    / "vidur_vllm_real_testing"
    / "traces"
    / "splitwise_conv_20s_english_canonical.csv"
)


class FakeRequest:
    def __init__(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        max_tokens: int,
        computed_tokens: int,
        output_tokens: int,
    ) -> None:
        self.request_id = request_id
        self.num_prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.num_computed_tokens = computed_tokens
        self.num_output_tokens = output_tokens


class VllmLiveStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = TraceMetadataRegistry.load(CANONICAL_TRACE)
        cls.metadata = cls.registry.require("splitwise-conv-000000")

    def test_partial_prefill_preserves_actual_and_canonical_tails(self) -> None:
        processed = max(1, self.metadata.actual_prefill_tokens // 2)
        request = FakeRequest(
            self.metadata.request_id,
            prompt_tokens=self.metadata.actual_prefill_tokens,
            max_tokens=self.metadata.actual_decode_tokens + 1,
            computed_tokens=processed,
            output_tokens=0,
        )
        state = build_live_state_snapshot(
            running=[],
            waiting=[request],
            registry=self.registry,
            max_num_scheduled_tokens=4608,
            captured_monotonic_s=1.0,
        )
        row = state.requests[0]
        self.assertIs(row.phase, RequestPhase.PREFILL)
        self.assertEqual(
            row.actual_prefill_remaining,
            self.metadata.actual_prefill_tokens - processed,
        )
        self.assertEqual(
            row.canonical_prefill_remaining,
            self.metadata.canonical_prefill_tokens - processed,
        )

    def test_actual_tail_remains_schedulable_after_rounding_down(self) -> None:
        metadata = self.registry.require("splitwise-conv-000001")
        self.assertGreater(
            metadata.actual_prefill_tokens, metadata.canonical_prefill_tokens
        )
        request = FakeRequest(
            metadata.request_id,
            prompt_tokens=metadata.actual_prefill_tokens,
            max_tokens=metadata.actual_decode_tokens + 1,
            computed_tokens=metadata.canonical_prefill_tokens,
            output_tokens=0,
        )
        state = build_live_state_snapshot(
            running=[request],
            waiting=[],
            registry=self.registry,
            max_num_scheduled_tokens=4608,
        )
        row = state.requests[0]
        self.assertIs(row.phase, RequestPhase.PREFILL)
        self.assertEqual(row.canonical_prefill_remaining, 128)

    def test_prefill_sampled_output_is_not_gv3_decode_progress(self) -> None:
        request = FakeRequest(
            self.metadata.request_id,
            prompt_tokens=self.metadata.actual_prefill_tokens,
            max_tokens=self.metadata.actual_decode_tokens + 1,
            computed_tokens=self.metadata.actual_prefill_tokens,
            output_tokens=1,
        )
        state = build_live_state_snapshot(
            running=[request],
            waiting=[],
            registry=self.registry,
            max_num_scheduled_tokens=4608,
        )
        row = state.requests[0]
        self.assertIs(row.phase, RequestPhase.DECODE)
        self.assertEqual(row.num_output_tokens, 0)
        self.assertEqual(
            row.actual_decode_remaining,
            self.metadata.actual_decode_tokens,
        )

    def test_completed_prefill_becomes_decode(self) -> None:
        request = FakeRequest(
            self.metadata.request_id,
            prompt_tokens=self.metadata.actual_prefill_tokens,
            max_tokens=self.metadata.actual_decode_tokens + 1,
            computed_tokens=self.metadata.actual_prefill_tokens,
            output_tokens=8,
        )
        state = build_live_state_snapshot(
            running=[request],
            waiting=[],
            registry=self.registry,
            max_num_scheduled_tokens=4608,
        )
        row = state.requests[0]
        self.assertIs(row.phase, RequestPhase.DECODE)
        self.assertEqual(row.actual_decode_remaining, self.metadata.actual_decode_tokens - 7)

    def test_real_prompt_length_mismatch_is_fatal(self) -> None:
        request = FakeRequest(
            self.metadata.request_id,
            prompt_tokens=self.metadata.actual_prefill_tokens + 1,
            max_tokens=self.metadata.actual_decode_tokens + 1,
            computed_tokens=0,
            output_tokens=0,
        )
        with self.assertRaisesRegex(CanonicalizationError, "prompt length"):
            build_live_state_snapshot(
                running=[],
                waiting=[request],
                registry=self.registry,
                max_num_scheduled_tokens=4608,
            )

    def test_duplicate_queue_membership_is_fatal(self) -> None:
        request = FakeRequest(
            self.metadata.request_id,
            prompt_tokens=self.metadata.actual_prefill_tokens,
            max_tokens=self.metadata.actual_decode_tokens + 1,
            computed_tokens=0,
            output_tokens=0,
        )
        with self.assertRaisesRegex(CanonicalizationError, "more than one vLLM queue"):
            build_live_state_snapshot(
                running=[request],
                waiting=[request],
                registry=self.registry,
                max_num_scheduled_tokens=4608,
            )


if __name__ == "__main__":
    unittest.main()
