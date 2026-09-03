"""Convert vLLM 0.13 request objects into immutable GV3 scheduler snapshots."""

from __future__ import annotations

import time
from typing import Any, Iterable

from .canonicalization import CanonicalizationError
from .scheduler_contract import (
    LiveRequestSnapshot,
    LiveStateSnapshot,
    RequestPhase,
    TraceMetadataRegistry,
)

DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS = 1


def logical_decode_tokens_from_vllm(
    num_output_tokens: int,
    *,
    implicit_prefill_output_tokens: int = DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS,
) -> int:
    """Convert vLLM output count to explicit GV3 decode progress.

    vLLM samples its first output token in the final prefill model step. GV3
    completes prefill without advancing decode, so that physical token is a
    transport-only offset rather than a canonical decode action.
    """

    offset = int(implicit_prefill_output_tokens)
    if offset not in {0, 1}:
        raise ValueError("implicit_prefill_output_tokens must be 0 or 1")
    return max(0, int(num_output_tokens) - offset)


def snapshot_vllm_request(
    request: Any,
    *,
    queue_name: str,
    registry: TraceMetadataRegistry,
    implicit_prefill_output_tokens: int = DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS,
) -> LiveRequestSnapshot | None:
    request_id = str(request.request_id)
    metadata = registry.require(request_id)
    actual_prompt_tokens = int(request.num_prompt_tokens)
    if actual_prompt_tokens != metadata.actual_prefill_tokens:
        raise CanonicalizationError(
            f"request {request_id}: vLLM prompt length {actual_prompt_tokens} does not "
            f"match trace length {metadata.actual_prefill_tokens}"
        )
    output_offset = int(implicit_prefill_output_tokens)
    if output_offset not in {0, 1}:
        raise ValueError("implicit_prefill_output_tokens must be 0 or 1")
    max_tokens = int(request.max_tokens)
    expected_max_tokens = metadata.actual_decode_tokens + output_offset
    if max_tokens != expected_max_tokens:
        raise CanonicalizationError(
            f"request {request_id}: vLLM max_tokens {max_tokens} does not match trace "
            f"physical decode budget {expected_max_tokens} "
            f"({metadata.actual_decode_tokens} GV3 decodes + {output_offset} "
            "implicit prefill output)"
        )

    num_computed = max(0, int(request.num_computed_tokens))
    physical_output_tokens = max(0, int(request.num_output_tokens))
    output_tokens = logical_decode_tokens_from_vllm(
        physical_output_tokens,
        implicit_prefill_output_tokens=output_offset,
    )
    actual_prefill_processed = min(actual_prompt_tokens, num_computed)
    actual_prefill_remaining = max(0, actual_prompt_tokens - actual_prefill_processed)
    if actual_prefill_remaining > 0:
        canonical_prefill_remaining = max(
            0,
            metadata.canonical_prefill_tokens - actual_prefill_processed,
        )
        # Nearest-grid canonicalization may round a real prompt down. Expose
        # its final physical tail as one canonical 128-token prefill chunk;
        # plan projection truncates it back to the exact physical remainder.
        if canonical_prefill_remaining == 0 and actual_prefill_remaining > 0:
            canonical_prefill_remaining = 128
        phase = RequestPhase.PREFILL
    else:
        canonical_prefill_remaining = 0
        phase = RequestPhase.DECODE

    actual_decode_remaining = max(0, metadata.actual_decode_tokens - output_tokens)
    canonical_decode_remaining = max(0, metadata.canonical_decode_tokens - output_tokens)
    if phase is RequestPhase.DECODE and actual_decode_remaining <= 0:
        return None

    return LiveRequestSnapshot(
        request_id=request_id,
        phase=phase,
        arrival_time_s=metadata.arrived_at_s,
        actual_prefill_tokens=metadata.actual_prefill_tokens,
        actual_prefill_remaining=actual_prefill_remaining,
        canonical_prefill_tokens=metadata.canonical_prefill_tokens,
        canonical_prefill_remaining=canonical_prefill_remaining,
        actual_decode_tokens=metadata.actual_decode_tokens,
        actual_decode_remaining=actual_decode_remaining,
        canonical_decode_tokens=metadata.canonical_decode_tokens,
        canonical_decode_remaining=canonical_decode_remaining,
        actual_prefill_slo_s=metadata.actual_prefill_slo_s,
        canonical_prefill_slo_s=metadata.canonical_prefill_slo_s,
        actual_decode_slo_s=metadata.actual_decode_slo_s,
        canonical_decode_slo_s=metadata.canonical_decode_slo_s,
        num_computed_tokens=num_computed,
        num_output_tokens=output_tokens,
        queue_name=str(queue_name),
    )


def build_live_state_snapshot(
    *,
    running: Iterable[Any],
    waiting: Iterable[Any],
    registry: TraceMetadataRegistry,
    max_num_scheduled_tokens: int,
    captured_monotonic_s: float | None = None,
    implicit_prefill_output_tokens: int = DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS,
) -> LiveStateSnapshot:
    requests: list[LiveRequestSnapshot] = []
    seen: set[str] = set()
    for queue_name, queue in (("running", running), ("waiting", waiting)):
        for request in queue:
            request_id = str(request.request_id)
            if request_id in seen:
                raise CanonicalizationError(
                    f"request {request_id}: present in more than one vLLM queue"
                )
            seen.add(request_id)
            snapshot = snapshot_vllm_request(
                request,
                queue_name=queue_name,
                registry=registry,
                implicit_prefill_output_tokens=implicit_prefill_output_tokens,
            )
            if snapshot is not None:
                requests.append(snapshot)
    return LiveStateSnapshot.build(
        requests,
        max_num_scheduled_tokens=int(max_num_scheduled_tokens),
        captured_monotonic_s=(
            time.monotonic()
            if captured_monotonic_s is None
            else float(captured_monotonic_s)
        ),
    )
