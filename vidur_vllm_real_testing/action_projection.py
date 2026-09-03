"""Project canonical GV3 token allocations onto real vLLM request tails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .canonicalization import CanonicalizationError


@dataclass(frozen=True)
class ProjectedAllocation:
    request_id: str
    canonical_allocation: int
    actual_remaining_before: int
    actual_allocation: int
    truncated_to_actual_tail: bool


def project_token_allocations(
    canonical_allocations: Mapping[str, int],
    actual_remaining_by_request_id: Mapping[str, int],
) -> tuple[ProjectedAllocation, ...]:
    """Apply ``min(canonical allocation, actual remaining)`` with hard checks."""

    output: list[ProjectedAllocation] = []
    for request_id in sorted(canonical_allocations):
        canonical = int(canonical_allocations[request_id])
        if canonical != canonical_allocations[request_id] or canonical <= 0:
            raise CanonicalizationError(
                f"request {request_id}: canonical allocation must be a positive integer"
            )
        if request_id not in actual_remaining_by_request_id:
            raise CanonicalizationError(
                f"request {request_id}: canonical action references no live vLLM request"
            )
        actual_remaining = int(actual_remaining_by_request_id[request_id])
        if (
            actual_remaining != actual_remaining_by_request_id[request_id]
            or actual_remaining <= 0
        ):
            raise CanonicalizationError(
                f"request {request_id}: actual remaining tokens must be a positive integer"
            )
        actual = min(canonical, actual_remaining)
        output.append(
            ProjectedAllocation(
                request_id=request_id,
                canonical_allocation=canonical,
                actual_remaining_before=actual_remaining,
                actual_allocation=actual,
                truncated_to_actual_tail=actual < canonical,
            )
        )
    return tuple(output)
