"""Profiling helpers layered on the production persistent adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .gv3_live_adapter import BatchExecutionObservation, GV3PersistentAdapter
from .scheduler_contract import (
    CanonicalAllocation,
    LiveStateSnapshot,
    RequestPhase,
    SchedulePlan,
)


def full_prefill_state_planner(
    state_payload: Mapping[str, Any],
    snapshot: LiveStateSnapshot,
) -> SchedulePlan:
    """Schedule one isolated prefill in one batch for profile comparison."""

    del state_payload
    prefills = [
        request
        for request in snapshot.requests
        if request.phase is RequestPhase.PREFILL
        and request.canonical_prefill_remaining > 0
    ]
    if len(prefills) != 1:
        raise RuntimeError(
            f"prefill profiler requires exactly one pending prefill, got {len(prefills)}"
        )
    request = prefills[0]
    return SchedulePlan(
        state_fingerprint=snapshot.fingerprint,
        allocations=(
            CanonicalAllocation(
                request.request_id,
                RequestPhase.PREFILL,
                request.canonical_prefill_remaining,
            ),
        ),
        policy="gv3-prefill-profile-full-batch",
        details={"canonical_prefill_tokens": request.canonical_prefill_remaining},
    )


class GV3PrefillProfilingAdapter(GV3PersistentAdapter):
    """Persistent adapter that appends completed-batch timing observations."""

    def on_batch_completed(self, observation: BatchExecutionObservation) -> None:
        super().on_batch_completed(observation)
        output = os.environ.get("VIDUR_VLLM_GV3_BATCH_OBSERVATIONS", "").strip()
        if not output:
            return
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "scheduled_monotonic_s": observation.scheduled_monotonic_s,
            "completed_monotonic_s": observation.completed_monotonic_s,
            "duration_s": observation.duration_s,
            "scheduled_tokens_by_request": dict(
                observation.scheduled_tokens_by_request
            ),
            "request_progress": [
                {
                    "request_id": progress.request_id,
                    "phase_before": progress.phase_before.value,
                    "num_computed_tokens_before": progress.num_computed_tokens_before,
                    "num_output_tokens_before": progress.num_output_tokens_before,
                    "num_computed_tokens_after": progress.num_computed_tokens_after,
                    "num_output_tokens_after": progress.num_output_tokens_after,
                    "finished_after": progress.finished_after,
                }
                for progress in observation.request_progress
            ],
            "sim_time_after": self.sim_time,
            "decode_credit_balance_after": self.decode_credit_balance,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

