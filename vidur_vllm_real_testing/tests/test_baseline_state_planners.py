from __future__ import annotations

import unittest

from vidur_vllm_real_testing.baseline_state_planners import sjf256, sjf512
from vidur_vllm_real_testing.scheduler_contract import (
    LiveRequestSnapshot,
    LiveStateSnapshot,
    RequestPhase,
)


def _request(request_id: str, phase: RequestPhase, remaining: int) -> LiveRequestSnapshot:
    return LiveRequestSnapshot(
        request_id=request_id,
        phase=phase,
        arrival_time_s=0.0,
        actual_prefill_tokens=remaining if phase is RequestPhase.PREFILL else 128,
        actual_prefill_remaining=remaining if phase is RequestPhase.PREFILL else 0,
        canonical_prefill_tokens=remaining if phase is RequestPhase.PREFILL else 128,
        canonical_prefill_remaining=remaining if phase is RequestPhase.PREFILL else 0,
        actual_decode_tokens=864,
        actual_decode_remaining=remaining if phase is RequestPhase.DECODE else 864,
        canonical_decode_tokens=864,
        canonical_decode_remaining=remaining if phase is RequestPhase.DECODE else 864,
        actual_prefill_slo_s=0.1,
        canonical_prefill_slo_s=0.1,
        actual_decode_slo_s=0.05,
        canonical_decode_slo_s=0.05,
        num_computed_tokens=128 if phase is RequestPhase.DECODE else 0,
        num_output_tokens=0,
        queue_name="running" if phase is RequestPhase.DECODE else "waiting",
    )


class BaselineStatePlannerTests(unittest.TestCase):
    def test_sjf_state_planners_preserve_decode_and_prefill_budgets(self) -> None:
        snapshot = LiveStateSnapshot.build(
            (
                _request("decode", RequestPhase.DECODE, 864),
                _request("short", RequestPhase.PREFILL, 128),
                _request("long", RequestPhase.PREFILL, 512),
            ),
            max_num_scheduled_tokens=8192,
            captured_monotonic_s=1.0,
        )
        plan256 = sjf256({}, snapshot)
        plan512 = sjf512({}, snapshot)
        self.assertEqual(plan256.policy, "sjf-256")
        self.assertEqual(plan512.policy, "sjf-512")
        self.assertEqual(sum(row.num_tokens for row in plan256.allocations), 257)
        self.assertEqual(sum(row.num_tokens for row in plan512.allocations), 513)


if __name__ == "__main__":
    unittest.main()
