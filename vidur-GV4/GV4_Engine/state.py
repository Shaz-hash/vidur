"""Compact, branchable runtime state for the GV4 game engine.

Only authoritative data lives here. KV allocation, pipeline timing, action
resolution, transitions, and snapshot I/O are separate modules. This keeps the
MCTS clone path small and gives the future native state a direct array layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import GV4EngineConfig


__all__ = [
    "BatchAllocation",
    "GV4State",
    "GV4StateError",
    "InflightMicrobatchState",
    "LaunchRecord",
    "NO_ID",
    "ObjectiveState",
    "Player",
    "ReplicaState",
    "RequestLifecycle",
    "RequestState",
    "TerminalReason",
    "UNASSIGNED_REPLICA",
    "UNSET_TIME",
]


NO_ID = -1
UNASSIGNED_REPLICA = -1
UNSET_TIME = -1.0


class GV4StateError(ValueError):
    """Raised when runtime state violates the GV4 contract."""


class Player(IntEnum):
    ADVERSARY = 0
    CONTROLLER = 1
    ROUTER = 2


class RequestLifecycle(IntEnum):
    WAITING_PREFILL = 0
    INFLIGHT_PREFILL = 1
    WAITING_DECODE = 2
    INFLIGHT_DECODE = 3
    STOP_PENDING = 4
    DROP_PENDING = 5
    COMPLETED = 6
    STOPPED = 7
    DROPPED = 8
    INFLIGHT_RECOMPUTE = 9
    PREEMPT_PENDING = 10

    @property
    def is_inflight(self) -> bool:
        return self in (
            RequestLifecycle.INFLIGHT_PREFILL,
            RequestLifecycle.INFLIGHT_DECODE,
            RequestLifecycle.INFLIGHT_RECOMPUTE,
            RequestLifecycle.PREEMPT_PENDING,
            RequestLifecycle.STOP_PENDING,
            RequestLifecycle.DROP_PENDING,
        )

    @property
    def is_terminal(self) -> bool:
        return self in (
            RequestLifecycle.COMPLETED,
            RequestLifecycle.STOPPED,
            RequestLifecycle.DROPPED,
        )


class TerminalReason(IntEnum):
    NONE = 0
    NATURAL_COMPLETION = 1
    ADVERSARY_STOP = 2
    CONTROLLER_EVICTION = 3
    AUTOMATIC_SLO_DROP = 4
    DECODE_CREDIT_EXHAUSTED = 5


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GV4StateError(f"{name} must be a nonnegative integer")


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise GV4StateError(f"{name} must be finite and >= 0")


def _finite_or_unset(name: str, value: float) -> None:
    if value != UNSET_TIME:
        _finite_nonnegative(name, value)


@dataclass(frozen=True, slots=True)
class LaunchRecord:
    """One adversary launch retained in the sliding launch window."""

    launch_time: float
    request_count: int
    prefill_tokens: int

    def assert_valid(self) -> None:
        _finite_nonnegative("launch_time", self.launch_time)
        if self.request_count <= 0:
            raise GV4StateError("launch request_count must be positive")
        if self.prefill_tokens <= 0:
            raise GV4StateError("launch prefill_tokens must be positive")



@dataclass(frozen=True, slots=True)
class BatchAllocation:
    """One kind of request work and its newly reserved logical KV blocks."""

    request_id: int
    prefill_tokens: int = 0
    decode_tokens: int = 0
    new_kv_blocks: int = 0
    recompute_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens + self.recompute_tokens

    def assert_valid(self) -> None:
        _nonnegative_int("allocation request_id", self.request_id)
        _nonnegative_int("allocation prefill_tokens", self.prefill_tokens)
        _nonnegative_int("allocation decode_tokens", self.decode_tokens)
        _nonnegative_int("allocation recompute_tokens", self.recompute_tokens)
        _nonnegative_int("allocation new_kv_blocks", self.new_kv_blocks)
        work_kinds = sum(
            value > 0
            for value in (
                self.prefill_tokens,
                self.decode_tokens,
                self.recompute_tokens,
            )
        )
        if work_kinds != 1:
            raise GV4StateError(
                "a batch allocation must contain exactly one kind of work"
            )


@dataclass(slots=True)
class InflightMicrobatchState:
    """An admitted batch and its precomputed pipeline calendar."""

    microbatch_id: int
    replica_id: int
    raw_action_index: int
    canonical_action_index: int
    allocations: tuple[BatchAllocation, ...]
    stage_ready_times: tuple[float, ...]
    stage_start_times: tuple[float, ...]
    stage_finish_times: tuple[float, ...]
    completion_applied: bool = False

    @property
    def final_completion_time(self) -> float:
        return self.stage_finish_times[-1]

    @property
    def total_prefill_tokens(self) -> int:
        return sum(item.prefill_tokens for item in self.allocations)

    @property
    def total_decode_tokens(self) -> int:
        return sum(item.decode_tokens for item in self.allocations)

    @property
    def total_recompute_tokens(self) -> int:
        return sum(item.recompute_tokens for item in self.allocations)

    def assert_valid(self, *, expected_stage_count: int | None = None) -> None:
        for name, value in (
            ("microbatch_id", self.microbatch_id),
            ("microbatch replica_id", self.replica_id),
            ("raw_action_index", self.raw_action_index),
            ("canonical_action_index", self.canonical_action_index),
        ):
            _nonnegative_int(name, value)

        if not isinstance(self.allocations, tuple) or not self.allocations:
            raise GV4StateError("microbatch allocations must be a nonempty tuple")
        request_ids = tuple(item.request_id for item in self.allocations)
        if request_ids != tuple(sorted(set(request_ids))):
            raise GV4StateError("microbatch request IDs must be sorted and unique")
        for allocation in self.allocations:
            allocation.assert_valid()

        arrays = (
            self.stage_ready_times,
            self.stage_start_times,
            self.stage_finish_times,
        )
        if any(not isinstance(values, tuple) for values in arrays):
            raise GV4StateError("microbatch stage-time arrays must be tuples")
        stage_count = len(self.stage_ready_times)
        if stage_count == 0 or any(len(values) != stage_count for values in arrays):
            raise GV4StateError("microbatch stage-time arrays must have equal length")
        if expected_stage_count is not None and stage_count != expected_stage_count:
            raise GV4StateError("microbatch PP width differs from replica PP width")

        previous_finish = None
        for stage, (ready, start, finish) in enumerate(zip(*arrays)):
            _finite_nonnegative(f"stage {stage} ready time", ready)
            _finite_nonnegative(f"stage {stage} start time", start)
            _finite_nonnegative(f"stage {stage} finish time", finish)
            if ready > start or start >= finish:
                raise GV4StateError(
                    f"stage {stage} must satisfy ready <= start < finish"
                )
            if previous_finish is not None and ready < previous_finish:
                raise GV4StateError(
                    f"stage {stage} is ready before the prior stage finishes"
                )
            previous_finish = finish

    def clone(self) -> "InflightMicrobatchState":
        # All tuple fields are immutable and safe to share between branches.
        return InflightMicrobatchState(
            self.microbatch_id,
            self.replica_id,
            self.raw_action_index,
            self.canonical_action_index,
            self.allocations,
            self.stage_ready_times,
            self.stage_start_times,
            self.stage_finish_times,
            self.completion_applied,
        )


@dataclass(slots=True)
class RequestState:
    """Logical request progress, SLO state, and physical KV residency."""

    request_id: int
    owner_replica_id: int
    lifecycle: RequestLifecycle
    arrival_time: float
    prefill_deadline: float
    decode_token_slo_sec: float
    original_prefill_tokens: int
    original_decode_tokens: int
    decode_credit_minted: bool = False
    committed_prefill_tokens: int = 0 ## tokens that have been fully processed and committed
    reserved_prefill_tokens: int = 0 ## tokens that are reserved for inflight work
    committed_decode_tokens: int = 0
    reserved_decode_tokens: int = 0 ## Should always be 1 if request is in the batch!
    committed_kv_blocks: int = 0
    reserved_kv_blocks: int = 0
    inflight_microbatch_id: int = NO_ID
    next_decode_deadline: float = UNSET_TIME
    prefill_lateness_sec: float = 0.0
    decode_lateness_sec: float = 0.0
    violation_recorded: bool = False
    terminal_reason: TerminalReason = TerminalReason.NONE
    terminal_requested_at: float = UNSET_TIME
    terminal_time: float = UNSET_TIME
    # Omitted in old fixtures means all committed context is physically resident.
    kv_computed_tokens: int = NO_ID
    reserved_recompute_tokens: int = 0

    def __post_init__(self) -> None:
        if self.kv_computed_tokens == NO_ID:
            self.kv_computed_tokens = (
                0 if self.lifecycle.is_terminal else self.logical_context_tokens
            )

    @property
    def remaining_prefill_tokens(self) -> int:
        return (
            self.original_prefill_tokens
            - self.committed_prefill_tokens
            - self.reserved_prefill_tokens
        )

    @property
    def remaining_decode_tokens(self) -> int:
        return (
            self.original_decode_tokens
            - self.committed_decode_tokens
            - self.reserved_decode_tokens
        )

    @property
    def logical_context_tokens(self) -> int:
        """Tokens whose model work completed, whether their KV remains or not."""

        return self.committed_prefill_tokens + self.committed_decode_tokens

    @property
    def remaining_recompute_tokens(self) -> int:
        """Committed context whose evicted KV still needs to be rebuilt."""

        return (
            self.logical_context_tokens
            - self.kv_computed_tokens
            - self.reserved_recompute_tokens
        )

    @property
    def is_decode_phase(self) -> bool:
        return self.committed_prefill_tokens == self.original_prefill_tokens

    @property
    def resident_tokens(self) -> int:
        """Physical KV tokens, including capacity reserved by in-flight work."""

        return (
            self.kv_computed_tokens
            + self.reserved_recompute_tokens
            + self.reserved_prefill_tokens
            + self.reserved_decode_tokens
        )

    @property
    def has_inflight_work(self) -> bool:
        return self.inflight_microbatch_id != NO_ID

    def age_at(self, now: float) -> float:
        return max(0.0, now - self.arrival_time)

    def prefill_slack_at(self, now: float) -> float:
        return self.prefill_deadline - now

    def tokens_used_in_final_kv_block(self, block_size_tokens: int) -> int:
        _nonnegative_int("resident tokens", self.resident_tokens)
        if block_size_tokens <= 0:
            raise GV4StateError("block_size_tokens must be positive")
        if self.resident_tokens == 0:
            return 0
        remainder = self.resident_tokens % block_size_tokens
        return remainder or block_size_tokens

    def tokens_until_next_kv_block(self, block_size_tokens: int) -> int:
        used = self.tokens_used_in_final_kv_block(block_size_tokens)
        if used in (0, block_size_tokens):
            return 0
        return block_size_tokens - used

    def assert_valid(self, *, block_size_tokens: int | None = None) -> None:
        _nonnegative_int("request_id", self.request_id)
        if self.owner_replica_id < UNASSIGNED_REPLICA:
            raise GV4StateError("owner_replica_id must be -1 or a replica ID")
        if not isinstance(self.lifecycle, RequestLifecycle):
            raise GV4StateError("request lifecycle must be RequestLifecycle")
        if not isinstance(self.terminal_reason, TerminalReason):
            raise GV4StateError("terminal reason must be TerminalReason")
        if not isinstance(self.violation_recorded, bool):
            raise GV4StateError("violation_recorded must be bool")
        if not isinstance(self.decode_credit_minted, bool):
            raise GV4StateError("decode_credit_minted must be bool")

        _finite_nonnegative("request arrival_time", self.arrival_time)
        _finite_nonnegative("request prefill_deadline", self.prefill_deadline)
        if self.prefill_deadline < self.arrival_time:
            raise GV4StateError("prefill deadline cannot precede request arrival")
        if not math.isfinite(self.decode_token_slo_sec) or self.decode_token_slo_sec <= 0:
            raise GV4StateError("decode_token_slo_sec must be finite and > 0")
        if self.original_prefill_tokens <= 0 or self.original_decode_tokens <= 0:
            raise GV4StateError("original prefill and decode tokens must be positive")

        for name, value in (
            ("committed_prefill_tokens", self.committed_prefill_tokens),
            ("reserved_prefill_tokens", self.reserved_prefill_tokens),
            ("committed_decode_tokens", self.committed_decode_tokens),
            ("reserved_decode_tokens", self.reserved_decode_tokens),
            ("kv_computed_tokens", self.kv_computed_tokens),
            ("reserved_recompute_tokens", self.reserved_recompute_tokens),
            ("committed_kv_blocks", self.committed_kv_blocks),
            ("reserved_kv_blocks", self.reserved_kv_blocks),
        ):
            _nonnegative_int(name, value)
        if self.remaining_prefill_tokens < 0 or self.remaining_decode_tokens < 0:
            raise GV4StateError("committed plus reserved work exceeds original work")
        if self.remaining_recompute_tokens < 0:
            raise GV4StateError("computed plus reserved KV exceeds logical context")
        if (self.reserved_prefill_tokens or self.reserved_decode_tokens) and (
            self.remaining_recompute_tokens or self.reserved_recompute_tokens
        ):
            raise GV4StateError("new work cannot run before KV recomputation finishes")

        if self.decode_credit_minted:
            if (
                self.committed_prefill_tokens != self.original_prefill_tokens
                or self.reserved_prefill_tokens
            ):
                raise GV4StateError(
                    "decode credit can be minted only after prefill completion"
                )
        elif self.committed_decode_tokens or self.reserved_decode_tokens:
            raise GV4StateError("decode work requires minted decode credit")

        _finite_nonnegative("prefill_lateness_sec", self.prefill_lateness_sec)
        _finite_nonnegative("decode_lateness_sec", self.decode_lateness_sec)
        _finite_or_unset("next_decode_deadline", self.next_decode_deadline)
        _finite_or_unset("terminal_requested_at", self.terminal_requested_at)
        _finite_or_unset("terminal_time", self.terminal_time)

        if self.lifecycle.is_inflight != self.has_inflight_work:
            raise GV4StateError("lifecycle and in-flight batch link disagree")
        if self.has_inflight_work and not (
            self.reserved_prefill_tokens
            or self.reserved_decode_tokens
            or self.reserved_recompute_tokens
        ):
            raise GV4StateError("an in-flight request must reserve work")
        if not self.has_inflight_work and (
            self.reserved_prefill_tokens
            or self.reserved_decode_tokens
            or self.reserved_recompute_tokens
            or self.reserved_kv_blocks
        ):
            raise GV4StateError("a non-in-flight request cannot retain reservations")

        if self.lifecycle == RequestLifecycle.WAITING_PREFILL:
            if self.remaining_prefill_tokens == 0:
                raise GV4StateError("WAITING_PREFILL requires unfinished prefill")
        elif self.lifecycle == RequestLifecycle.INFLIGHT_PREFILL:
            if (
                self.reserved_prefill_tokens == 0
                or self.reserved_decode_tokens
                or self.reserved_recompute_tokens
            ):
                raise GV4StateError("INFLIGHT_PREFILL must reserve only prefill work")
        elif self.lifecycle == RequestLifecycle.WAITING_DECODE:
            if self.remaining_prefill_tokens or self.remaining_decode_tokens == 0:
                raise GV4StateError("WAITING_DECODE requires unfinished decode only")
        elif self.lifecycle == RequestLifecycle.INFLIGHT_DECODE:
            if (
                self.reserved_decode_tokens != 1
                or self.reserved_prefill_tokens
                or self.reserved_recompute_tokens
            ):
                raise GV4StateError(
                    "INFLIGHT_DECODE must reserve exactly one decode token"
                )
        elif self.lifecycle == RequestLifecycle.INFLIGHT_RECOMPUTE:
            if (
                self.reserved_recompute_tokens == 0
                or self.reserved_prefill_tokens
                or self.reserved_decode_tokens
            ):
                raise GV4StateError(
                    "INFLIGHT_RECOMPUTE must reserve only recomputation"
                )
        elif self.lifecycle == RequestLifecycle.PREEMPT_PENDING:
            # The admitted allocation drains normally before its KV is released.
            work_kinds = sum(
                value > 0
                for value in (
                    self.reserved_prefill_tokens,
                    self.reserved_decode_tokens,
                    self.reserved_recompute_tokens,
                )
            )
            if work_kinds != 1:
                raise GV4StateError(
                    "PREEMPT_PENDING must retain exactly one in-flight work kind"
                )
        elif self.lifecycle == RequestLifecycle.COMPLETED:
            if self.remaining_prefill_tokens or self.remaining_decode_tokens:
                raise GV4StateError("COMPLETED requires all work to be committed")

        decode_is_active = (
            self.lifecycle
            in (
                RequestLifecycle.WAITING_DECODE,
                RequestLifecycle.INFLIGHT_DECODE,
            )
            or (
                self.lifecycle == RequestLifecycle.INFLIGHT_RECOMPUTE
                and self.is_decode_phase
            )
            or (
                self.lifecycle == RequestLifecycle.PREEMPT_PENDING
                and (
                    self.reserved_decode_tokens > 0
                    or (
                        self.reserved_recompute_tokens > 0
                        and self.is_decode_phase
                    )
                )
            )
            or self.reserved_decode_tokens > 0
        )
        if decode_is_active and self.next_decode_deadline == UNSET_TIME:
            raise GV4StateError("active decode requires next_decode_deadline")

        if self.lifecycle.is_terminal:
            if (
                self.committed_kv_blocks
                or self.reserved_kv_blocks
                or self.kv_computed_tokens
                or self.reserved_recompute_tokens
            ):
                raise GV4StateError("terminal requests cannot retain physical KV")
            if self.terminal_reason == TerminalReason.NONE:
                raise GV4StateError("terminal requests require a terminal reason")
            if self.terminal_time == UNSET_TIME:
                raise GV4StateError("terminal requests require terminal_time")
        elif self.terminal_time != UNSET_TIME:
            raise GV4StateError("non-terminal requests cannot have terminal_time")

        pending = self.lifecycle in (
            RequestLifecycle.STOP_PENDING,
            RequestLifecycle.DROP_PENDING,
        )
        if pending:
            if (
                self.terminal_reason == TerminalReason.NONE
                or self.terminal_requested_at == UNSET_TIME
            ):
                raise GV4StateError("pending terminal request lacks reason or time")
        elif not self.lifecycle.is_terminal and (
            self.terminal_reason != TerminalReason.NONE
            or self.terminal_requested_at != UNSET_TIME
        ):
            raise GV4StateError("active request carries terminal bookkeeping")

        if self.owner_replica_id == UNASSIGNED_REPLICA and (
            self.has_inflight_work
            or self.committed_prefill_tokens
            or self.committed_decode_tokens
            or self.kv_computed_tokens
            or self.reserved_recompute_tokens
            or self.committed_kv_blocks
        ):
            raise GV4StateError("unassigned request cannot own work or KV")

        if block_size_tokens is not None and not self.lifecycle.is_terminal:
            if block_size_tokens <= 0:
                raise GV4StateError("block_size_tokens must be positive")
            minimum_blocks = (
                self.resident_tokens + block_size_tokens - 1
            ) // block_size_tokens
            if self.committed_kv_blocks + self.reserved_kv_blocks < minimum_blocks:
                raise GV4StateError("request owns fewer KV blocks than resident tokens")

    def clone(self) -> "RequestState":
        return RequestState(
            request_id=self.request_id,
            owner_replica_id=self.owner_replica_id,
            lifecycle=self.lifecycle,
            arrival_time=self.arrival_time,
            prefill_deadline=self.prefill_deadline,
            decode_token_slo_sec=self.decode_token_slo_sec,
            original_prefill_tokens=self.original_prefill_tokens,
            original_decode_tokens=self.original_decode_tokens,
            decode_credit_minted=self.decode_credit_minted,
            committed_prefill_tokens=self.committed_prefill_tokens,
            reserved_prefill_tokens=self.reserved_prefill_tokens,
            committed_decode_tokens=self.committed_decode_tokens,
            reserved_decode_tokens=self.reserved_decode_tokens,
            committed_kv_blocks=self.committed_kv_blocks,
            reserved_kv_blocks=self.reserved_kv_blocks,
            inflight_microbatch_id=self.inflight_microbatch_id,
            next_decode_deadline=self.next_decode_deadline,
            prefill_lateness_sec=self.prefill_lateness_sec,
            decode_lateness_sec=self.decode_lateness_sec,
            violation_recorded=self.violation_recorded,
            terminal_reason=self.terminal_reason,
            terminal_requested_at=self.terminal_requested_at,
            terminal_time=self.terminal_time,
            kv_computed_tokens=self.kv_computed_tokens,
            reserved_recompute_tokens=self.reserved_recompute_tokens,
        )


@dataclass(slots=True)
class ObjectiveState:
    """Cumulative counters and costs needed for exact transition rewards."""

    requests_generated: int = 0
    requests_completed: int = 0
    requests_stopped: int = 0
    requests_dropped: int = 0
    slo_violations: int = 0
    prefill_lateness_sec: float = 0.0
    decode_lateness_sec: float = 0.0
    terminal_cost: float = 0.0
    total_cost: float = 0.0

    def assert_valid(self) -> None:
        for name, value in (
            ("requests_generated", self.requests_generated),
            ("requests_completed", self.requests_completed),
            ("requests_stopped", self.requests_stopped),
            ("requests_dropped", self.requests_dropped),
            ("slo_violations", self.slo_violations),
        ):
            _nonnegative_int(name, value)
        for name, value in (
            ("prefill_lateness_sec", self.prefill_lateness_sec),
            ("decode_lateness_sec", self.decode_lateness_sec),
            ("terminal_cost", self.terminal_cost),
            ("total_cost", self.total_cost),
        ):
            _finite_nonnegative(name, value)

    def clone(self) -> "ObjectiveState":
        return ObjectiveState(
            self.requests_generated,
            self.requests_completed,
            self.requests_stopped,
            self.requests_dropped,
            self.slo_violations,
            self.prefill_lateness_sec,
            self.decode_lateness_sec,
            self.terminal_cost,
            self.total_cost,
        )


@dataclass(slots=True)
class ReplicaState:
    """Per-replica KV counters, compact PP calendar, and in-flight ring."""

    replica_id: int
    rank_ids: tuple[int, ...]
    rank_kv_capacity_blocks: tuple[int, ...]
    rank_kv_committed_blocks: list[int]
    rank_kv_reserved_blocks: list[int]
    stage_tail_finish_times: list[float]
    stage_last_microbatch_ids: list[int]
    inflight_microbatches: list[InflightMicrobatchState]

    @property
    def inflight_count(self) -> int:
        return len(self.inflight_microbatches)

    @property
    def pipeline_parallel_size(self) -> int:
        return len(self.stage_tail_finish_times)

    def free_kv_blocks_by_rank(self) -> tuple[int, ...]:
        return tuple(
            capacity - committed - reserved
            for capacity, committed, reserved in zip(
                self.rank_kv_capacity_blocks,
                self.rank_kv_committed_blocks,
                self.rank_kv_reserved_blocks,
            )
        )

    def find_microbatch(self, microbatch_id: int) -> InflightMicrobatchState | None:
        for batch in self.inflight_microbatches:
            if batch.microbatch_id == microbatch_id:
                return batch
        return None

    def assert_valid(self) -> None:
        _nonnegative_int("replica_id", self.replica_id)
        if not isinstance(self.rank_ids, tuple):
            raise GV4StateError("replica rank_ids must be a tuple")
        if not isinstance(self.rank_kv_capacity_blocks, tuple):
            raise GV4StateError("replica rank capacities must be a tuple")

        rank_count = len(self.rank_ids)
        rank_arrays = (
            self.rank_kv_capacity_blocks,
            self.rank_kv_committed_blocks,
            self.rank_kv_reserved_blocks,
        )
        if rank_count == 0 or any(len(values) != rank_count for values in rank_arrays):
            raise GV4StateError("replica rank arrays must have equal nonzero length")
        if tuple(sorted(set(self.rank_ids))) != self.rank_ids:
            raise GV4StateError("replica rank IDs must be sorted and unique")

        if not self.stage_tail_finish_times or len(
            self.stage_tail_finish_times
        ) != len(self.stage_last_microbatch_ids):
            raise GV4StateError("replica stage arrays must have equal nonzero length")

        for capacity, committed, reserved in zip(*rank_arrays):
            if capacity <= 0:
                raise GV4StateError("rank KV capacity must be positive")
            _nonnegative_int("committed KV blocks", committed)
            _nonnegative_int("reserved KV blocks", reserved)
            if committed + reserved > capacity:
                raise GV4StateError("rank KV usage exceeds capacity")
        for tail in self.stage_tail_finish_times:
            _finite_nonnegative("stage tail finish time", tail)
        if any(batch_id < NO_ID for batch_id in self.stage_last_microbatch_ids):
            raise GV4StateError("stage last-batch IDs must be -1 or nonnegative")

        batch_ids = [batch.microbatch_id for batch in self.inflight_microbatches]
        if batch_ids != sorted(set(batch_ids)):
            raise GV4StateError("in-flight batches must have sorted unique IDs")

        prior_batch = None
        for batch in self.inflight_microbatches:
            ## Essentially we are checking that the finish time of the batch at the end of the pipeline to have
            ## finish time less than or equal to the tail finish time of the stage.
            batch.assert_valid(expected_stage_count=self.pipeline_parallel_size)
            if batch.replica_id != self.replica_id:
                raise GV4StateError("microbatch is stored under the wrong replica")
            if batch.completion_applied:
                raise GV4StateError("completed batch must leave the in-flight ring")
            for stage, finish in enumerate(batch.stage_finish_times):
                if finish > self.stage_tail_finish_times[stage]:
                    raise GV4StateError("stage tail precedes scheduled work")
                if (
                    prior_batch is not None
                    and batch.stage_start_times[stage]
                    < prior_batch.stage_finish_times[stage]
                ):
                    raise GV4StateError("FIFO stages cannot overlap batches")
            prior_batch = batch

    def clone(self) -> "ReplicaState":
        return ReplicaState(
            self.replica_id,
            self.rank_ids,
            self.rank_kv_capacity_blocks,
            self.rank_kv_committed_blocks.copy(),
            self.rank_kv_reserved_blocks.copy(),
            self.stage_tail_finish_times.copy(),
            self.stage_last_microbatch_ids.copy(),
            [batch.clone() for batch in self.inflight_microbatches],
        )


@dataclass(slots=True)
class GV4State:
    """Complete branchable state; immutable configuration remains external."""

    state_schema_version: str
    config_manifest_sha256: str
    now: float
    next_player: Player
    next_adversary_tick: float
    next_request_id: int
    next_microbatch_id: int
    tie_break_counter: int
    rng_seed: int
    rng_counter: int
    launch_history: list[LaunchRecord]
    decode_credits_available: int
    decode_credits_reserved: int
    decode_credits_minted_total: int
    decode_tokens_committed_total: int
    requests: list[RequestState]
    replicas: list[ReplicaState]
    objective: ObjectiveState

    @classmethod
    def initial(
        cls,
        config: "GV4EngineConfig",
        *,
        now: float = 0.0,
        next_player: Player = Player.ADVERSARY,
    ) -> "GV4State":
        """Create an empty runtime state from immutable configured dimensions."""

        _finite_nonnegative("initial time", now)
        rank_capacities = config.rank_kv_block_capacities()
        replicas = []
        for placement in config.topology.replica_placements:
            rank_ids = placement.rank_ids
            rank_count = len(rank_ids)
            stage_count = config.topology.pipeline_parallel_size
            replicas.append(
                ReplicaState(
                    replica_id=placement.replica_id,
                    rank_ids=rank_ids,
                    rank_kv_capacity_blocks=tuple(
                        rank_capacities[rank_id] for rank_id in rank_ids
                    ),
                    rank_kv_committed_blocks=[0] * rank_count,
                    rank_kv_reserved_blocks=[0] * rank_count,
                    stage_tail_finish_times=[now] * stage_count,
                    stage_last_microbatch_ids=[NO_ID] * stage_count,
                    inflight_microbatches=[],
                )
            )

        state = cls(
            state_schema_version=config.layout.state_schema_version,
            config_manifest_sha256=config.manifest_sha256(),
            now=now,
            next_player=next_player,
            next_adversary_tick=now,
            next_request_id=0,
            next_microbatch_id=0,
            tie_break_counter=0,
            rng_seed=config.global_seed,
            rng_counter=0,
            launch_history=[],
            decode_credits_available=0,
            decode_credits_reserved=0,
            decode_credits_minted_total=0,
            decode_tokens_committed_total=0,
            requests=[],
            replicas=replicas,
            objective=ObjectiveState(),
        )
        if config.enable_debug_asserts:
            state.assert_valid(config)
        return state

    def request(self, request_id: int) -> RequestState:
        """Return a request in O(1); IDs are stable array positions."""

        if not 0 <= request_id < len(self.requests):
            raise GV4StateError(f"unknown request ID {request_id}")
        request = self.requests[request_id]
        if request.request_id != request_id:
            raise GV4StateError("request array is not indexed by request ID")
        return request

    def replica(self, replica_id: int) -> ReplicaState:
        """Return a replica in O(1); IDs are stable array positions."""

        if not 0 <= replica_id < len(self.replicas):
            raise GV4StateError(f"unknown replica ID {replica_id}")
        replica = self.replicas[replica_id]
        if replica.replica_id != replica_id:
            raise GV4StateError("replica array is not indexed by replica ID")
        return replica

    def clone(self) -> "GV4State":
        """Copy mutable branch data and share only immutable tuples/records."""

        return GV4State(
            self.state_schema_version,
            self.config_manifest_sha256,
            self.now,
            self.next_player,
            self.next_adversary_tick,
            self.next_request_id,
            self.next_microbatch_id,
            self.tie_break_counter,
            self.rng_seed,
            self.rng_counter,
            self.launch_history.copy(),
            self.decode_credits_available,
            self.decode_credits_reserved,
            self.decode_credits_minted_total,
            self.decode_tokens_committed_total,
            [request.clone() for request in self.requests],
            [replica.clone() for replica in self.replicas],
            self.objective.clone(),
        )

    def assert_valid(self, config: "GV4EngineConfig") -> None:
        """Run expensive cross-record checks in tests and debug transitions."""

        if self.state_schema_version != config.layout.state_schema_version:
            raise GV4StateError("state schema does not match configuration")
        if self.config_manifest_sha256 != config.manifest_sha256():
            raise GV4StateError("state belongs to a different configuration")
        if not isinstance(self.next_player, Player):
            raise GV4StateError("next_player must be Player")
        _finite_nonnegative("state time", self.now)
        _finite_nonnegative("next adversary tick", self.next_adversary_tick)
        if self.next_adversary_tick + config.timing.epsilon < self.now:
            raise GV4StateError("next adversary tick is behind simulator time")

        for name, value in (
            ("next_request_id", self.next_request_id),
            ("next_microbatch_id", self.next_microbatch_id),
            ("tie_break_counter", self.tie_break_counter),
            ("rng_seed", self.rng_seed),
            ("rng_counter", self.rng_counter),
            ("decode_credits_available", self.decode_credits_available),
            ("decode_credits_reserved", self.decode_credits_reserved),
            ("decode_credits_minted_total", self.decode_credits_minted_total),
            ("decode_tokens_committed_total", self.decode_tokens_committed_total),
        ):
            _nonnegative_int(name, value)

        if self.next_request_id != len(self.requests):
            raise GV4StateError("next_request_id must equal append-only request count")
        if len(self.requests) > config.layout.max_requests:
            raise GV4StateError("request array exceeds native layout capacity")
        if len(self.replicas) != config.topology.num_replicas:
            raise GV4StateError("replica count differs from configuration")

        self._assert_launch_history(config)
        batches = self._assert_replicas(config)
        self._assert_requests_and_reservations(config, batches)
        self._assert_objective()

    def _assert_launch_history(self, config: "GV4EngineConfig") -> None:
        if len(self.launch_history) > config.layout.max_launch_history_entries:
            raise GV4StateError("launch history exceeds native layout capacity")
        prior_time = -1.0
        for record in self.launch_history:
            record.assert_valid()
            if record.launch_time < prior_time:
                raise GV4StateError("launch history must be ordered by time")
            prior_time = record.launch_time

    def _assert_replicas(
        self, config: "GV4EngineConfig"
    ) -> dict[int, InflightMicrobatchState]:
        rank_capacities = config.rank_kv_block_capacities()
        batches: dict[int, InflightMicrobatchState] = {}

        for replica_id, replica in enumerate(self.replicas):
            if replica.replica_id != replica_id:
                raise GV4StateError("replicas must remain in replica-ID order")
            replica.assert_valid()
            placement = config.topology.replica_placements[replica_id]
            if replica.rank_ids != placement.rank_ids:
                raise GV4StateError("replica rank placement differs from config")
            expected = tuple(rank_capacities[rank] for rank in replica.rank_ids)
            if replica.rank_kv_capacity_blocks != expected:
                raise GV4StateError("replica KV capacities differ from config")
            if replica.pipeline_parallel_size != config.topology.pipeline_parallel_size:
                raise GV4StateError("replica PP width differs from config")
            if replica.inflight_count > config.scheduler.max_inflight_microbatches:
                raise GV4StateError("replica exceeds in-flight batch capacity")
            for batch in replica.inflight_microbatches:
                if batch.microbatch_id in batches:
                    raise GV4StateError("microbatch ID is not globally unique")
                batches[batch.microbatch_id] = batch

        if batches and self.next_microbatch_id <= max(batches):
            raise GV4StateError("next_microbatch_id does not exceed active IDs")
        return batches

    def _assert_requests_and_reservations(
        self,
        config: "GV4EngineConfig",
        batches: dict[int, InflightMicrobatchState],
    ) -> None:
        committed_by_replica = [0] * len(self.replicas)
        reserved_by_replica = [0] * len(self.replicas)
        reserved_decode_tokens = 0
        minted_requests = 0
        committed_decode_tokens = 0

        for request_id, request in enumerate(self.requests):
            if request.request_id != request_id:
                raise GV4StateError("requests must remain in request-ID order")
            request.assert_valid(block_size_tokens=config.kv_cache.block_size_tokens)
            if (
                request.original_prefill_tokens
                > config.request.max_prefill_tokens_per_request
            ):
                raise GV4StateError("request prefill token count exceeds config")
            if not (
                config.request.min_decode_tokens_per_request
                <= request.original_decode_tokens
                <= config.request.max_decode_tokens_per_request
            ):
                raise GV4StateError("request decode token count exceeds config")
            if request.owner_replica_id >= len(self.replicas):
                raise GV4StateError("request owns an unknown replica")

            minted_requests += int(request.decode_credit_minted)
            committed_decode_tokens += request.committed_decode_tokens

            if request.owner_replica_id != UNASSIGNED_REPLICA:
                committed_by_replica[request.owner_replica_id] += (
                    request.committed_kv_blocks
                )
                reserved_by_replica[request.owner_replica_id] += (
                    request.reserved_kv_blocks
                )

            if not request.has_inflight_work:
                continue
            batch = batches.get(request.inflight_microbatch_id)
            if batch is None:
                raise GV4StateError("request references a missing microbatch")
            allocation = next(
                (
                    item
                    for item in batch.allocations
                    if item.request_id == request.request_id
                ),
                None,
            )
            if allocation is None or batch.replica_id != request.owner_replica_id:
                raise GV4StateError("request and batch links disagree")
            if (
                allocation.prefill_tokens != request.reserved_prefill_tokens
                or allocation.decode_tokens != request.reserved_decode_tokens
                or allocation.recompute_tokens != request.reserved_recompute_tokens
                or allocation.new_kv_blocks != request.reserved_kv_blocks
            ):
                raise GV4StateError("request reservations differ from allocation")
            reserved_decode_tokens += allocation.decode_tokens

        for batch in batches.values():
            for allocation in batch.allocations:
                request = self.request(allocation.request_id)
                if request.inflight_microbatch_id != batch.microbatch_id:
                    raise GV4StateError("batch lacks a matching request back-link")

        for replica in self.replicas:
            committed = committed_by_replica[replica.replica_id]
            reserved = reserved_by_replica[replica.replica_id]
            if any(value != committed for value in replica.rank_kv_committed_blocks):
                raise GV4StateError("committed KV ledger disagrees with requests")
            if any(value != reserved for value in replica.rank_kv_reserved_blocks):
                raise GV4StateError("reserved KV ledger disagrees with requests")

        if self.decode_credits_reserved != reserved_decode_tokens:
            raise GV4StateError("decode credit reservations disagree with batches")

        expected_minted = (
            minted_requests
            * config.credits.decode_credit_mint_per_prefill_completion
        )
        if self.decode_credits_minted_total != expected_minted:
            raise GV4StateError("decode credit mint total disagrees with requests")
        if self.decode_tokens_committed_total != committed_decode_tokens:
            raise GV4StateError("committed decode total disagrees with requests")
        if (
            self.decode_credits_available
            + self.decode_credits_reserved
            + self.decode_tokens_committed_total
            != self.decode_credits_minted_total
        ):
            raise GV4StateError("decode credit conservation invariant failed")
        if self.decode_credits_available == 0 and any(
            request.lifecycle in (
                RequestLifecycle.WAITING_DECODE,
                RequestLifecycle.INFLIGHT_DECODE,
            )
            or (
                request.lifecycle == RequestLifecycle.INFLIGHT_RECOMPUTE
                and request.is_decode_phase
            )
            or (
                request.lifecycle == RequestLifecycle.PREEMPT_PENDING
                and (
                    request.reserved_decode_tokens > 0
                    or (
                        request.reserved_recompute_tokens > 0
                        and request.is_decode_phase
                    )
                )
            )
            for request in self.requests
        ):
            raise GV4StateError(
                "zero available decode credit cannot leave an active decode request"
            )

    def _assert_objective(self) -> None:
        self.objective.assert_valid()
        if self.objective.requests_generated != len(self.requests):
            raise GV4StateError("generated counter disagrees with requests")
        expected_completed = sum(
            request.lifecycle == RequestLifecycle.COMPLETED
            for request in self.requests
        )
        expected_stopped = sum(
            request.lifecycle == RequestLifecycle.STOPPED
            for request in self.requests
        )
        expected_dropped = sum(
            request.lifecycle == RequestLifecycle.DROPPED
            for request in self.requests
        )
        expected_violations = sum(
            request.violation_recorded for request in self.requests
        )
        if self.objective.requests_completed != expected_completed:
            raise GV4StateError("completed counter disagrees with requests")
        if self.objective.requests_stopped != expected_stopped:
            raise GV4StateError("stopped counter disagrees with requests")
        if self.objective.requests_dropped != expected_dropped:
            raise GV4StateError("dropped counter disagrees with requests")
        if self.objective.slo_violations != expected_violations:
            raise GV4StateError("violation counter disagrees with requests")
