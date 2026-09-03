"""Fast FIFO pipeline timing for admitted GV4 microbatches.

The caller predicts one service duration per PP stage and one communication
duration per PP boundary. This module only places that work on the replica's
compact calendar. It does not choose requests, reserve KV, spend credits, or
commit completed work.

Stage service durations must already include GPU compute and TP collectives
inside that PP stage. Boundary durations contain only inter-stage PP transfer.
Keeping those values separate prevents PP communication from being counted
twice.
"""

from __future__ import annotations

import math

from .config import SchedulerConfig, TimingConfig
from .state import BatchAllocation, InflightMicrobatchState, ReplicaState


__all__ = [
    "PipelineCalendarError",
    "PipelineUnavailableError",
    "admit_microbatch",
    "build_microbatch_calendar",
    "can_admit_microbatch",
    "next_pipeline_admission_time",
]


class PipelineCalendarError(ValueError):
    """Raised when pipeline timing or calendar state violates the contract."""


class PipelineUnavailableError(PipelineCalendarError):
    """Raised when a structurally valid replica cannot admit work yet."""


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PipelineCalendarError(f"{name} must be a nonnegative integer")


def _finite_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineCalendarError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0.0:
        raise PipelineCalendarError(f"{name} must be a finite nonnegative number")


def _check_replica_calendar(
    replica: ReplicaState,
    scheduler: SchedulerConfig,
) -> int:
    """Return PP width after inexpensive hot-path structural checks."""

    stage_count = len(replica.stage_tail_finish_times)
    _nonnegative_int("replica_id", replica.replica_id)
    if stage_count == 0 or len(replica.stage_last_microbatch_ids) != stage_count:
        raise PipelineCalendarError(
            "replica stage-tail and last-microbatch arrays must have equal width"
        )
    for stage, tail in enumerate(replica.stage_tail_finish_times):
        _finite_nonnegative(f"stage {stage} tail finish time", tail)
    if any(batch_id < -1 for batch_id in replica.stage_last_microbatch_ids):
        raise PipelineCalendarError(
            "stage last-microbatch IDs must be -1 or nonnegative"
        )
    if replica.inflight_count > scheduler.max_inflight_microbatches:
        raise PipelineCalendarError("replica exceeds its in-flight microbatch limit")
    if scheduler.inter_stage_queue_capacity < scheduler.max_inflight_microbatches:
        raise PipelineCalendarError(
            "explicit PP backpressure is required when inter-stage capacity is "
            "smaller than the in-flight limit"
        )
    return stage_count


def _admission_blocker(
    replica: ReplicaState,
    *,
    admitted_at: float,
    scheduler: SchedulerConfig,
    timing: TimingConfig,
) -> str | None:
    "Return a string explaining why the replica cannot admit work, or None if it can."
    _finite_nonnegative("admitted_at", admitted_at)
    _check_replica_calendar(replica, scheduler)

    if replica.inflight_count >= scheduler.max_inflight_microbatches:
        return "the replica has no free in-flight microbatch slot"
    if replica.stage_tail_finish_times[0] > admitted_at + timing.epsilon:
        return "pipeline stage 0 is still busy"
    return None


def can_admit_microbatch(
    replica: ReplicaState,
    *,
    admitted_at: float,
    scheduler: SchedulerConfig,
    timing: TimingConfig,
) -> bool:
    """Return whether stage 0 and the bounded in-flight ring can admit work."""

    return (
        _admission_blocker(
            replica,
            admitted_at=admitted_at,
            scheduler=scheduler,
            timing=timing,
        )
        is None
    )


def next_pipeline_admission_time(
    replica: ReplicaState,
    *,
    now: float,
    scheduler: SchedulerConfig,
    timing: TimingConfig,
) -> float:
    """Return the next time allowed by pipeline occupancy alone.

    Request legality, KV availability, and credits may move the controller's
    actual decision later. The transition engine must process completions at the
    returned time before testing admission again.
    """

    _finite_nonnegative("now", now)
    _check_replica_calendar(replica, scheduler)

    candidate = max(now, replica.stage_tail_finish_times[0])
    if replica.inflight_count == scheduler.max_inflight_microbatches:
        if not replica.inflight_microbatches:
            raise PipelineCalendarError("full in-flight count has no batch record")
        # FIFO on the final stage makes the first ring entry complete first.
        candidate = max(
            candidate,
            replica.inflight_microbatches[0].final_completion_time,
        )
    return round(candidate, timing.time_round_digits)


def _check_allocations(allocations: tuple[BatchAllocation, ...]) -> None:
    if not isinstance(allocations, tuple) or not allocations:
        raise PipelineCalendarError("microbatch allocations must be a nonempty tuple")

    previous_request_id = -1
    for allocation in allocations:
        try:
            allocation.assert_valid()
        except ValueError as error:
            raise PipelineCalendarError(str(error)) from error
        if allocation.request_id <= previous_request_id:
            raise PipelineCalendarError(
                "microbatch request IDs must be sorted and unique"
            )
        previous_request_id = allocation.request_id


def _check_durations(
    *,
    stage_count: int,
    stage_service_times: tuple[float, ...],
    pp_communication_times: tuple[float, ...],
) -> None:
    if not isinstance(stage_service_times, tuple):
        raise PipelineCalendarError("stage_service_times must be a tuple")
    if not isinstance(pp_communication_times, tuple):
        raise PipelineCalendarError("pp_communication_times must be a tuple")
    if len(stage_service_times) != stage_count:
        raise PipelineCalendarError("one service duration is required per PP stage")
    if len(pp_communication_times) != stage_count - 1:
        raise PipelineCalendarError(
            "one communication duration is required per PP boundary"
        )

    for stage, duration in enumerate(stage_service_times):
        _finite_nonnegative(f"stage {stage} service duration", duration)
        if duration == 0.0:
            raise PipelineCalendarError("stage service durations must be positive")
    for boundary, duration in enumerate(pp_communication_times):
        _finite_nonnegative(f"PP boundary {boundary} duration", duration)


def _check_new_microbatch_id(
    replica: ReplicaState,
    microbatch_id: int,
) -> None:
    _nonnegative_int("microbatch_id", microbatch_id)
    latest_scheduled_id = max(replica.stage_last_microbatch_ids)
    if microbatch_id <= latest_scheduled_id:
        raise PipelineCalendarError(
            "new microbatch ID must be greater than every scheduled batch ID"
        )


def build_microbatch_calendar(
    replica: ReplicaState,
    *,
    microbatch_id: int,
    raw_action_index: int,
    canonical_action_index: int,
    allocations: tuple[BatchAllocation, ...],
    admitted_at: float,
    stage_service_times: tuple[float, ...],
    pp_communication_times: tuple[float, ...],
    scheduler: SchedulerConfig,
    timing: TimingConfig,
) -> InflightMicrobatchState:
    """Calculate an admitted batch's complete PP timetable without mutation."""

    blocker = _admission_blocker(
        replica,
        admitted_at=admitted_at,
        scheduler=scheduler,
        timing=timing,
    )
    if blocker is not None:
        raise PipelineUnavailableError(blocker)

    stage_count = replica.pipeline_parallel_size
    _check_new_microbatch_id(replica, microbatch_id)
    _nonnegative_int("raw_action_index", raw_action_index)
    _nonnegative_int("canonical_action_index", canonical_action_index)
    _check_allocations(allocations)
    _check_durations(
        stage_count=stage_count,
        stage_service_times=stage_service_times,
        pp_communication_times=pp_communication_times,
    )

    ready_times: list[float] = []
    start_times: list[float] = []
    finish_times: list[float] = []
    ready = admitted_at

    for stage in range(stage_count):
        start = max(ready, replica.stage_tail_finish_times[stage])
        finish = round(
            start + stage_service_times[stage],
            timing.time_round_digits,
        )
        if finish <= start:
            raise PipelineCalendarError(
                "time rounding removed a positive stage service duration"
            )

        ready_times.append(ready)
        start_times.append(start)
        finish_times.append(finish)

        if stage + 1 < stage_count:
            ready = round(
                finish + pp_communication_times[stage],
                timing.time_round_digits,
            )

    batch = InflightMicrobatchState(
        microbatch_id=microbatch_id,
        replica_id=replica.replica_id,
        raw_action_index=raw_action_index,
        canonical_action_index=canonical_action_index,
        allocations=allocations,
        stage_ready_times=tuple(ready_times),
        stage_start_times=tuple(start_times),
        stage_finish_times=tuple(finish_times),
    )
    return batch


def admit_microbatch(
    replica: ReplicaState,
    *,
    microbatch_id: int,
    raw_action_index: int,
    canonical_action_index: int,
    allocations: tuple[BatchAllocation, ...],
    admitted_at: float,
    stage_service_times: tuple[float, ...],
    pp_communication_times: tuple[float, ...],
    scheduler: SchedulerConfig,
    timing: TimingConfig,
) -> InflightMicrobatchState:
    """Atomically calculate and append one microbatch to a replica calendar."""

    batch = build_microbatch_calendar(
        replica,
        microbatch_id=microbatch_id,
        raw_action_index=raw_action_index,
        canonical_action_index=canonical_action_index,
        allocations=allocations,
        admitted_at=admitted_at,
        stage_service_times=stage_service_times,
        pp_communication_times=pp_communication_times,
        scheduler=scheduler,
        timing=timing,
    )

    # Every operation that can reject the admission has already completed.
    replica.stage_tail_finish_times[:] = batch.stage_finish_times
    replica.stage_last_microbatch_ids[:] = [microbatch_id] * len(
        batch.stage_finish_times
    )
    replica.inflight_microbatches.append(batch)
    return batch
