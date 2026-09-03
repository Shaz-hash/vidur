"""Fast logical KV-block accounting for the GV4 engine.

The ledger owns no hidden allocator state. Request ownership and per-rank replica
counters in ``state.py`` remain authoritative, which makes cloning and a later
native array implementation straightforward.
"""

from __future__ import annotations

from collections.abc import Sequence

from .state import (
    BatchAllocation,
    ReplicaState,
    RequestState,
    UNASSIGNED_REPLICA,
)


__all__ = [
    "KVCapacityError",
    "KVLedgerError",
    "additional_blocks_for_work",
    "blocks_for_tokens",
    "can_reserve_blocks",
    "commit_batch_blocks",
    "free_logical_blocks",
    "release_request_blocks",
    "reserve_batch_blocks",
]


class KVLedgerError(ValueError):
    """Raised when a KV-ledger operation violates the state contract."""


class KVCapacityError(KVLedgerError):
    """Raised when every required rank cannot satisfy a reservation."""


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise KVLedgerError(f"{name} must be a nonnegative integer")


def _check_replica_shape(replica: ReplicaState) -> int:
    """Return rank count after inexpensive structural checks."""

    rank_count = len(replica.rank_kv_capacity_blocks)
    if rank_count == 0 or not (
        len(replica.rank_kv_committed_blocks)
        == len(replica.rank_kv_reserved_blocks)
        == rank_count
    ):
        raise KVLedgerError("replica KV arrays must have equal nonzero length")
    return rank_count


def _request_for_allocation(
    requests: Sequence[RequestState], allocation: BatchAllocation
) -> RequestState:
    "Return the request referenced by a batch allocation, or raise if invalid."
    request_id = allocation.request_id
    if not 0 <= request_id < len(requests):
        raise KVLedgerError(f"allocation references unknown request {request_id}")
    request = requests[request_id]
    if request.request_id != request_id:
        raise KVLedgerError("request array is not indexed by request ID")
    return request


def _check_request_owner(request: RequestState, replica: ReplicaState) -> None:
    "Raise if a request and replica disagree on ownership."
    if request.owner_replica_id == UNASSIGNED_REPLICA:
        raise KVLedgerError("an unassigned request cannot own KV blocks")
    if request.owner_replica_id != replica.replica_id:
        raise KVLedgerError("request and replica ownership disagree")


def blocks_for_tokens(token_count: int, block_size_tokens: int) -> int:
    """Return exact logical blocks needed for ``token_count`` resident tokens."""

    _nonnegative_int("token_count", token_count)
    if isinstance(block_size_tokens, bool) or not isinstance(block_size_tokens, int):
        raise KVLedgerError("block_size_tokens must be a positive integer")
    if block_size_tokens <= 0:
        raise KVLedgerError("block_size_tokens must be a positive integer")
    return (token_count + block_size_tokens - 1) // block_size_tokens


def additional_blocks_for_work(
    request: RequestState,
    *,
    prefill_tokens: int = 0,
    decode_tokens: int = 0,
    block_size_tokens: int,
) -> int:
    """Calculate exact new blocks needed by one pre-admission allocation.

    Existing committed and reserved blocks are reused first. This naturally lets
    a decode run at full global capacity when it still fits in the request's final
    partial block.
    """

    _nonnegative_int("prefill_tokens", prefill_tokens)
    _nonnegative_int("decode_tokens", decode_tokens)
    if prefill_tokens == 0 and decode_tokens == 0:
        raise KVLedgerError("an allocation must contain prefill or decode work")
    if prefill_tokens and decode_tokens:
        raise KVLedgerError("one request cannot prefill and decode in one batch")
    if prefill_tokens > request.remaining_prefill_tokens:
        raise KVLedgerError("prefill allocation exceeds remaining request work")
    if decode_tokens > request.remaining_decode_tokens:
        raise KVLedgerError("decode allocation exceeds remaining request work")

    resident_after = request.resident_tokens + prefill_tokens + decode_tokens
    needed_after = blocks_for_tokens(resident_after, block_size_tokens)
    currently_owned = request.committed_kv_blocks + request.reserved_kv_blocks
    return max(0, needed_after - currently_owned)


def free_logical_blocks(replica: ReplicaState) -> int:
    """Return the least free block (min available blocks) count across all ranks of a replica."""

    rank_count = _check_replica_shape(replica)
    capacities = replica.rank_kv_capacity_blocks
    committed = replica.rank_kv_committed_blocks
    reserved = replica.rank_kv_reserved_blocks

    least_free = capacities[0] - committed[0] - reserved[0]
    for rank_index in range(1, rank_count):
        free = capacities[rank_index] - committed[rank_index] - reserved[rank_index]
        if free < least_free:
            least_free = free
    if least_free < 0:
        raise KVLedgerError("replica KV usage exceeds physical capacity")
    return least_free


def can_reserve_blocks(
    replica: ReplicaState,
    required_blocks: int,
    *,
    releasable_blocks: int = 0,
) -> bool:
    """Check an all-rank reservation without mutating the replica.

    ``releasable_blocks`` represents already validated terminal evictions in an
    action resolver's scratch calculation. Reserved in-flight blocks are never
    releasable through this argument.
    """

    _nonnegative_int("required_blocks", required_blocks)
    _nonnegative_int("releasable_blocks", releasable_blocks)
    rank_count = _check_replica_shape(replica)
    capacities = replica.rank_kv_capacity_blocks
    committed = replica.rank_kv_committed_blocks
    reserved = replica.rank_kv_reserved_blocks

    for rank_index in range(rank_count):
        if releasable_blocks > committed[rank_index]:
            raise KVLedgerError("releasable blocks exceed committed KV ownership")
        used_after = (
            committed[rank_index]
            - releasable_blocks
            + reserved[rank_index]
            + required_blocks
        )
        if used_after > capacities[rank_index]:
            return False
    return True


def reserve_batch_blocks(
    replica: ReplicaState,
    requests: Sequence[RequestState],
    allocations: tuple[BatchAllocation, ...],
    *,
    block_size_tokens: int,
) -> int:
    """Atomically reserve exact KV blocks for a resolved microbatch.

    All allocations and all ranks are validated before any counter is changed.
    Call this before linking requests to the new in-flight microbatch. The return
    value is the total logical blocks reserved on every rank.
    """

    if not allocations:
        raise KVLedgerError("cannot reserve KV for an empty batch")
    _check_replica_shape(replica)

    total_new_blocks = 0
    previous_request_id = -1
    for allocation in allocations:
        allocation.assert_valid()
        if allocation.request_id <= previous_request_id:
            raise KVLedgerError("batch request IDs must be sorted and unique")
        previous_request_id = allocation.request_id

        request = _request_for_allocation(requests, allocation)
        _check_request_owner(request, replica)
        if request.lifecycle.is_terminal:
            raise KVLedgerError("terminal requests cannot reserve KV blocks")
        if request.has_inflight_work:
            raise KVLedgerError("request already has in-flight work")
        if request.reserved_kv_blocks:
            raise KVLedgerError("request already owns reserved KV blocks")

        expected_blocks = additional_blocks_for_work(
            request,
            prefill_tokens=allocation.prefill_tokens,
            decode_tokens=allocation.decode_tokens,
            block_size_tokens=block_size_tokens,
        )
        if allocation.new_kv_blocks != expected_blocks:
            raise KVLedgerError(
                "allocation new_kv_blocks differs from exact KV requirement"
            )
        total_new_blocks += expected_blocks

    if not can_reserve_blocks(replica, total_new_blocks):
        raise KVCapacityError(
            f"replica {replica.replica_id} cannot reserve "
            f"{total_new_blocks} logical KV blocks"
        )

    for allocation in allocations:
        requests[allocation.request_id].reserved_kv_blocks += allocation.new_kv_blocks
    if total_new_blocks:
        reserved = replica.rank_kv_reserved_blocks
        for rank_index in range(len(reserved)):
            reserved[rank_index] += total_new_blocks
    return total_new_blocks


def commit_batch_blocks(
    replica: ReplicaState,
    requests: Sequence[RequestState],
    allocations: tuple[BatchAllocation, ...],
) -> int:
    """Move a completed batch's KV blocks from reserved to committed.

    Total KV usage does not change. The transition engine separately commits the
    corresponding token counters and removes the batch from the in-flight ring.
    """

    if not allocations:
        raise KVLedgerError("cannot commit KV for an empty batch")
    rank_count = _check_replica_shape(replica)

    total_blocks = 0
    previous_request_id = -1
    for allocation in allocations:
        allocation.assert_valid()
        if allocation.request_id <= previous_request_id:
            raise KVLedgerError("batch request IDs must be sorted and unique")
        previous_request_id = allocation.request_id
        request = _request_for_allocation(requests, allocation)
        _check_request_owner(request, replica)
        if request.reserved_kv_blocks != allocation.new_kv_blocks:
            raise KVLedgerError("request KV reservation differs from batch allocation")
        total_blocks += allocation.new_kv_blocks

    rank_reserved = replica.rank_kv_reserved_blocks
    for rank_index in range(rank_count):
        if rank_reserved[rank_index] < total_blocks:
            raise KVLedgerError("replica has fewer reserved blocks than the batch")

    for allocation in allocations:
        request = requests[allocation.request_id]
        request.reserved_kv_blocks -= allocation.new_kv_blocks
        request.committed_kv_blocks += allocation.new_kv_blocks

    if total_blocks:
        rank_committed = replica.rank_kv_committed_blocks
        for rank_index in range(rank_count):
            rank_reserved[rank_index] -= total_blocks
            rank_committed[rank_index] += total_blocks
    return total_blocks


def release_request_blocks(request: RequestState, replica: ReplicaState) -> int:
    """Release all committed blocks for a non-in-flight terminal removal.

    Call this before changing the request to a terminal lifecycle. Returning the
    released logical count lets the transition engine log the exact KV delta.
    """

    rank_count = _check_replica_shape(replica)
    _check_request_owner(request, replica)
    if request.has_inflight_work:
        raise KVLedgerError("in-flight request KV cannot be released")
    if request.reserved_kv_blocks:
        raise KVLedgerError("non-in-flight request unexpectedly owns reserved KV")

    released = request.committed_kv_blocks
    rank_committed = replica.rank_kv_committed_blocks
    for rank_index in range(rank_count):
        if rank_committed[rank_index] < released:
            raise KVLedgerError("replica has fewer committed blocks than the request")

    request.committed_kv_blocks = 0
    if released:
        for rank_index in range(rank_count):
            rank_committed[rank_index] -= released
    return released
