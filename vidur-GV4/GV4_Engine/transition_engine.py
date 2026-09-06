"""Atomic state transitions and boundary advancement for the GV4 engine.

Action selection is deliberately absent. This module applies already-resolved
actions, owns all mutable KV/credit/request bookkeeping, and commits completed
pipeline work in deterministic ``(completion_time, microbatch_id)`` order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from .action_resolver import (
    CanonicalAdversaryAction,
    CanonicalControllerAction,
    ControllerTransitionKind,
    PrefillTimeEstimator,
    resolve_adversary_action,
    resolve_controller_action,
)
from .config import GV4EngineConfig
from .kv_ledger import (
    commit_batch_blocks,
    preempt_request_blocks,
    release_request_blocks,
    reserve_batch_blocks,
)
from .pipeline_calendar import admit_microbatch, build_microbatch_calendar
from .state import (
    GV4State,
    LaunchRecord,
    Player,
    RequestLifecycle,
    RequestState,
    TerminalReason,
    UNASSIGNED_REPLICA,
    UNSET_TIME,
)


__all__ = [
    "LaunchPrefillTimeEstimator",
    "TransitionError",
    "TransitionOutcome",
    "advance_to",
    "apply_adversary_action",
    "apply_controller_action",
    "next_internal_completion_time",
    "next_wait_boundary_time",
]


LaunchPrefillTimeEstimator = Callable[[int], float]


class TransitionError(ValueError):
    """Raised when an action or time transition violates the GV4 contract."""


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    """Result plus the exact reward/discount metadata needed by MCTS."""

    state: GV4State
    transition_kind: str
    elapsed_sec: float
    objective_before: float
    objective_after: float
    edge_reward: float
    discount: float


def _outcome(
    state: GV4State,
    config: GV4EngineConfig,
    *,
    transition_kind: str,
    started_at: float,
    objective_before: float,
) -> TransitionOutcome:
    elapsed = max(0.0, state.now - started_at)
    objective_after = state.objective.total_cost
    return TransitionOutcome(
        state=state,
        transition_kind=transition_kind,
        elapsed_sec=elapsed,
        objective_before=objective_before,
        objective_after=objective_after,
        edge_reward=objective_before - objective_after,
        discount=config.reward.discount_for_elapsed(elapsed),
    )


def _assert_state(state: GV4State, config: GV4EngineConfig) -> None:
    if config.enable_debug_asserts:
        state.assert_valid(config)


def _refresh_objective(state: GV4State, config: GV4EngineConfig) -> None:
    """Rebuild cheap aggregate counters from authoritative request records."""

    completed = stopped = dropped = violations = 0
    prefill_lateness = decode_lateness = terminal_cost = total_cost = 0.0

    for request in state.requests:
        if request.lifecycle == RequestLifecycle.COMPLETED:
            completed += 1
        elif request.lifecycle == RequestLifecycle.STOPPED:
            stopped += 1
        elif request.lifecycle == RequestLifecycle.DROPPED:
            dropped += 1

        is_drop = request.lifecycle in (
            RequestLifecycle.DROP_PENDING,
            RequestLifecycle.DROPPED,
        )
        if is_drop:
            terminal_cost += config.cost.terminal_drop_cost
            total_cost += config.cost.terminal_drop_cost
            continue

        prefill_lateness += request.prefill_lateness_sec
        decode_lateness += request.decode_lateness_sec
        if request.violation_recorded:
            violations += 1
            request_lateness = (
                request.prefill_lateness_sec + request.decode_lateness_sec
            )
            total_cost += config.cost.violation_base_cost + min(
                request_lateness, config.cost.lateness_cap_sec
            )

    objective = state.objective
    objective.requests_generated = len(state.requests)
    objective.requests_completed = completed
    objective.requests_stopped = stopped
    objective.requests_dropped = dropped
    objective.slo_violations = violations
    objective.prefill_lateness_sec = prefill_lateness
    objective.decode_lateness_sec = decode_lateness
    objective.terminal_cost = terminal_cost
    objective.total_cost = total_cost


def _record_prefill_lateness(
    request: RequestState,
    at_time: float,
    config: GV4EngineConfig,
) -> None:
    lateness = max(0.0, at_time - request.prefill_deadline)
    if lateness > request.prefill_lateness_sec:
        request.prefill_lateness_sec = lateness
    if request.prefill_lateness_sec > config.timing.epsilon:
        request.violation_recorded = True


def _prune_launch_history(
    state: GV4State,
    config: GV4EngineConfig,
    at_time: float,
) -> None:
    """Discard launch records outside the open adversary window."""

    cutoff = at_time - config.timing.launch_window_sec
    epsilon = config.timing.epsilon
    state.launch_history[:] = [
        record
        for record in state.launch_history
        if record.launch_time > cutoff + epsilon
    ]


def _reserve_decode_credits(state: GV4State, tokens: int) -> None:
    if tokens > state.decode_credits_available:
        raise TransitionError("resolved batch exceeds available decode credits")
    state.decode_credits_available -= tokens
    state.decode_credits_reserved += tokens


def _consume_decode_reservations(state: GV4State, tokens: int) -> None:
    if tokens > state.decode_credits_reserved:
        raise TransitionError("completed batch exceeds reserved decode credits")
    state.decode_credits_reserved -= tokens
    state.decode_tokens_committed_total += tokens


def _mint_decode_credits(
    state: GV4State, request: RequestState, config: GV4EngineConfig
) -> None:
    if request.decode_credit_minted:
        raise TransitionError("request attempted to mint decode credit twice")
    minted = config.credits.decode_credit_mint_per_prefill_completion
    request.decode_credit_minted = True
    state.decode_credits_available += minted
    state.decode_credits_minted_total += minted


def _release_and_finish(
    state: GV4State,
    request: RequestState,
    *,
    lifecycle: RequestLifecycle,
    reason: TerminalReason,
    at_time: float,
) -> None:
    if request.has_inflight_work:
        raise TransitionError("cannot physically remove an in-flight request")
    if request.owner_replica_id != UNASSIGNED_REPLICA:
        release_request_blocks(request, state.replica(request.owner_replica_id))
    request.lifecycle = lifecycle
    request.terminal_reason = reason
    request.terminal_time = at_time
    request.next_decode_deadline = UNSET_TIME


def _mark_drop(
    state: GV4State,
    request: RequestState,
    *,
    reason: TerminalReason,
    at_time: float,
) -> None:
    """Charge terminal cost now; defer physical release only when in flight."""

    if request.lifecycle in (RequestLifecycle.DROP_PENDING, RequestLifecycle.DROPPED):
        return
    request.prefill_lateness_sec = 0.0
    request.decode_lateness_sec = 0.0
    request.violation_recorded = False
    request.terminal_reason = reason
    request.terminal_requested_at = at_time
    if request.has_inflight_work:
        request.lifecycle = RequestLifecycle.DROP_PENDING
        return
    _release_and_finish(
        state,
        request,
        lifecycle=RequestLifecycle.DROPPED,
        reason=reason,
        at_time=at_time,
    )


def _mark_stop(
    state: GV4State,
    request: RequestState,
    at_time: float,
    *,
    reason: TerminalReason = TerminalReason.ADVERSARY_STOP,
) -> None:
    request.terminal_reason = reason
    request.terminal_requested_at = at_time
    if request.has_inflight_work:
        request.lifecycle = RequestLifecycle.STOP_PENDING
        return
    _release_and_finish(
        state,
        request,
        lifecycle=RequestLifecycle.STOPPED,
        reason=reason,
        at_time=at_time,
    )


def _mark_preempt(state: GV4State, request: RequestState) -> int:
    """Apply preemption now, or defer it until admitted work completes."""

    if request.lifecycle.is_terminal or request.lifecycle in (
        RequestLifecycle.STOP_PENDING,
        RequestLifecycle.DROP_PENDING,
        RequestLifecycle.PREEMPT_PENDING,
    ):
        raise TransitionError("request cannot be preempted in its current lifecycle")
    if request.has_inflight_work:
        request.lifecycle = RequestLifecycle.PREEMPT_PENDING
        return 0

    released = preempt_request_blocks(
        request, state.replica(request.owner_replica_id)
    )
    request.lifecycle = (
        RequestLifecycle.WAITING_DECODE
        if request.is_decode_phase
        else RequestLifecycle.WAITING_PREFILL
    )
    return released


def _stop_active_decodes_after_credit_exhaustion(
    state: GV4State, at_time: float
) -> None:
    """Stop every existing decode generation after the final credit is issued."""

    if state.decode_credits_available != 0:
        return
    for request in state.requests:
        if request.lifecycle in (
            RequestLifecycle.WAITING_DECODE,
            RequestLifecycle.INFLIGHT_DECODE,
        ) or (
            request.lifecycle == RequestLifecycle.INFLIGHT_RECOMPUTE
            and request.is_decode_phase
        ) or (
            request.lifecycle == RequestLifecycle.PREEMPT_PENDING
            and (
                request.reserved_decode_tokens > 0
                or (
                    request.reserved_recompute_tokens > 0
                    and request.is_decode_phase
                )
            )
        ):
            _mark_stop(
                state,
                request,
                at_time,
                reason=TerminalReason.DECODE_CREDIT_EXHAUSTED,
            )


def _finish_naturally(
    state: GV4State, request: RequestState, at_time: float
) -> None:
    _release_and_finish(
        state,
        request,
        lifecycle=RequestLifecycle.COMPLETED,
        reason=TerminalReason.NATURAL_COMPLETION,
        at_time=at_time,
    )


def _complete_microbatch(
    state: GV4State,
    config: GV4EngineConfig,
    replica_id: int,
    microbatch_id: int,
) -> None:
    "Final commit point for a batch after it has finished last stage in the pipeline"
    replica = state.replica(replica_id)
    batch = replica.find_microbatch(microbatch_id)
    if batch is None or batch.completion_applied:
        raise TransitionError("microbatch completion is missing or already applied")
    completion_time = batch.final_completion_time

    commit_batch_blocks(replica, state.requests, batch.allocations)
    _consume_decode_reservations(state, batch.total_decode_tokens)

    for allocation in batch.allocations:
        request = state.request(allocation.request_id)
        pending_lifecycle = request.lifecycle
        preempt_after_completion = (
            pending_lifecycle == RequestLifecycle.PREEMPT_PENDING
        )
        request.committed_prefill_tokens += allocation.prefill_tokens
        request.reserved_prefill_tokens -= allocation.prefill_tokens
        request.committed_decode_tokens += allocation.decode_tokens
        request.reserved_decode_tokens -= allocation.decode_tokens
        request.kv_computed_tokens += allocation.total_tokens
        request.reserved_recompute_tokens -= allocation.recompute_tokens
        request.inflight_microbatch_id = -1

        if pending_lifecycle == RequestLifecycle.DROP_PENDING:
            _release_and_finish(
                state,
                request,
                lifecycle=RequestLifecycle.DROPPED,
                reason=request.terminal_reason,
                at_time=completion_time,
            )
            continue
        if pending_lifecycle == RequestLifecycle.STOP_PENDING:
            if allocation.decode_tokens and request.next_decode_deadline != UNSET_TIME:
                token_lateness = max(0.0, completion_time - request.next_decode_deadline)
                ## Stop applies for the decode requests
                request.decode_lateness_sec += token_lateness
                request.violation_recorded |= token_lateness > config.timing.epsilon
            _release_and_finish(
                state,
                request,
                lifecycle=RequestLifecycle.STOPPED,
                reason=request.terminal_reason,
                at_time=completion_time,
            )
            continue

        if allocation.recompute_tokens:
            request.lifecycle = (
                RequestLifecycle.WAITING_DECODE
                if request.is_decode_phase
                else RequestLifecycle.WAITING_PREFILL
            )
        elif allocation.prefill_tokens:
            if request.remaining_prefill_tokens:
                request.lifecycle = RequestLifecycle.WAITING_PREFILL
            else:
                _record_prefill_lateness(request, completion_time, config)
                request.lifecycle = RequestLifecycle.WAITING_DECODE
                request.next_decode_deadline = round(
                    completion_time + request.decode_token_slo_sec,
                    config.timing.time_round_digits,
                )
                _mint_decode_credits(state, request, config)
        else:
            if request.next_decode_deadline == UNSET_TIME:
                raise TransitionError("decode completion lacks a token deadline")
            token_lateness = max(0.0, completion_time - request.next_decode_deadline)
            request.decode_lateness_sec += token_lateness
            request.violation_recorded |= token_lateness > config.timing.epsilon
            if request.remaining_decode_tokens:
                request.lifecycle = RequestLifecycle.WAITING_DECODE
                request.next_decode_deadline = round(
                    completion_time + request.decode_token_slo_sec,
                    config.timing.time_round_digits,
                )
            else:
                _finish_naturally(state, request, completion_time)

        if preempt_after_completion:
            if request.lifecycle.is_terminal:
                raise TransitionError(
                    "a naturally completed request cannot remain preempt-pending"
                )
            preempt_request_blocks(request, replica)
            request.lifecycle = (
                RequestLifecycle.WAITING_DECODE
                if request.is_decode_phase
                else RequestLifecycle.WAITING_PREFILL
            )

    batch.completion_applied = True
    replica.inflight_microbatches.remove(batch)
    _prune_launch_history(state, config, completion_time)
    _refresh_objective(state, config)


def _apply_automatic_drops(
    state: GV4State,
    config: GV4EngineConfig,
    at_time: float,
) -> None:
    for request in state.requests:
        if request.lifecycle in (
            RequestLifecycle.WAITING_PREFILL,
            RequestLifecycle.INFLIGHT_PREFILL,
        ) or (
            request.lifecycle == RequestLifecycle.INFLIGHT_RECOMPUTE
            and not request.is_decode_phase
        ) or (
            request.lifecycle == RequestLifecycle.PREEMPT_PENDING
            and not request.is_decode_phase
            and request.reserved_decode_tokens == 0
        ):
            _record_prefill_lateness(request, at_time, config)

    for request in state.requests:
        if request.lifecycle.is_terminal or request.lifecycle in (
            RequestLifecycle.STOP_PENDING,
            RequestLifecycle.DROP_PENDING,
        ):
            continue
        total_lateness = request.prefill_lateness_sec + request.decode_lateness_sec
        if total_lateness >= config.cost.automatic_drop_lateness_sec:
            _mark_drop(
                state,
                request,
                reason=TerminalReason.AUTOMATIC_SLO_DROP,
                at_time=at_time,
            )
    _refresh_objective(state, config)


def _advance_to_inplace(
    state: GV4State,
    config: GV4EngineConfig,
    target_time: float,
) -> None:
    if not math.isfinite(target_time) or target_time < 0.0:
        raise TransitionError("target_time must be finite and nonnegative")
    if target_time + config.timing.epsilon < state.now:
        raise TransitionError("cannot move simulator time backwards")
    if target_time > state.next_adversary_tick + config.timing.epsilon:
        raise TransitionError("cannot advance past an unprocessed adversary tick")

    completions: list[tuple[float, int, int]] = []
    for replica in state.replicas:
        for batch in replica.inflight_microbatches:
            if batch.final_completion_time <= target_time + config.timing.epsilon:
                completions.append(
                    (batch.final_completion_time, batch.microbatch_id, replica.replica_id)
                )
    completions.sort()

    for completion_time, microbatch_id, replica_id in completions:
        state.now = completion_time
        _complete_microbatch(state, config, replica_id, microbatch_id)

    state.now = round(target_time, config.timing.time_round_digits)
    _prune_launch_history(state, config, state.now)
    _apply_automatic_drops(state, config, state.now)


def advance_to(
    state: GV4State,
    config: GV4EngineConfig,
    target_time: float,
    *,
    inplace: bool = False,
) -> TransitionOutcome:
    """Advance through internal completions, then apply drops at ``target_time``."""

    _assert_state(state, config)
    target = state if inplace else state.clone()
    started_at = target.now
    objective_before = target.objective.total_cost
    _advance_to_inplace(target, config, target_time)
    _assert_state(target, config)
    return _outcome(
        target,
        config,
        transition_kind="ADVANCE",
        started_at=started_at,
        objective_before=objective_before,
    )


def next_internal_completion_time(state: GV4State) -> float | None:
    """Return the earliest final-stage completion across all replicas."""

    earliest: float | None = None
    for replica in state.replicas:
        for batch in replica.inflight_microbatches:
            completion = batch.final_completion_time
            if earliest is None or completion < earliest:
                earliest = completion
    return earliest


def next_wait_boundary_time(state: GV4State, config: GV4EngineConfig) -> float:
    """Find the next future clock that can change controller action legality."""

    epsilon = config.timing.epsilon
    candidates: list[float] = []
    if state.next_adversary_tick > state.now + epsilon:
        candidates.append(state.next_adversary_tick)
    for replica in state.replicas:
        stage_zero_free = replica.stage_tail_finish_times[0]
        if stage_zero_free > state.now + epsilon:
            candidates.append(stage_zero_free)
        for batch in replica.inflight_microbatches:
            if batch.final_completion_time > state.now + epsilon:
                candidates.append(batch.final_completion_time)
    if not candidates:
        raise TransitionError("WAIT has no deterministic future enabling boundary")
    return round(min(candidates), config.timing.time_round_digits)


def apply_controller_action(
    state: GV4State,
    config: GV4EngineConfig,
    action: CanonicalControllerAction,
    *,
    stage_service_times: tuple[float, ...] | None = None,
    pp_communication_times: tuple[float, ...] | None = None,
    prefill_time_estimator: PrefillTimeEstimator | None = None,
    inplace: bool = False,
) -> TransitionOutcome:
    """Validate and atomically apply one canonical controller transition."""

    _assert_state(state, config)
    target = state if inplace else state.clone()
    started_at = target.now
    objective_before = target.objective.total_cost
    resolved = action.action
    if action.canonical_action_index < 0:
        raise TransitionError("canonical action index must be nonnegative")

    current = resolve_controller_action(
        target,
        config,
        replica_id=resolved.replica_id,
        raw_action_index=resolved.raw_action_index,
        prefill_time_estimator=prefill_time_estimator,
    )
    if current is None or current.canonical_key != resolved.canonical_key:
        raise TransitionError("controller action is stale or illegal for this state")

    service_times = stage_service_times or ()
    communication_times = pp_communication_times or ()
    if resolved.transition_kind == ControllerTransitionKind.BATCH:
        # Validate the complete calendar before changing KV, credits, or requests.
        build_microbatch_calendar(
            target.replica(resolved.replica_id),
            microbatch_id=target.next_microbatch_id,
            raw_action_index=resolved.raw_action_index,
            canonical_action_index=action.canonical_action_index,
            allocations=resolved.allocations,
            admitted_at=target.now,
            stage_service_times=service_times,
            pp_communication_times=communication_times,
            scheduler=config.scheduler,
            timing=config.timing,
        )
    elif stage_service_times is not None or pp_communication_times is not None:
        raise TransitionError("non-batch controller actions cannot carry stage timing")

    for request_id in resolved.evicted_request_ids:
        _mark_drop(
            target,
            target.request(request_id),
            reason=TerminalReason.CONTROLLER_EVICTION,
            at_time=target.now,
        )

    pending_ids = set(resolved.pending_preemption_request_ids)
    released_by_preemption = 0
    for request_id in resolved.preempted_request_ids:
        request = target.request(request_id)
        if request.has_inflight_work != (request_id in pending_ids):
            raise TransitionError("resolved preemption timing is stale")
        released_by_preemption += _mark_preempt(target, request)
    if released_by_preemption != resolved.preempted_kv_blocks:
        raise TransitionError("resolved preemption KV release is stale")

    if resolved.transition_kind == ControllerTransitionKind.BATCH:
        replica = target.replica(resolved.replica_id)
        reserve_batch_blocks(
            replica,
            target.requests,
            resolved.allocations,
            block_size_tokens=config.kv_cache.block_size_tokens,
        )
        decode_tokens = sum(item.decode_tokens for item in resolved.allocations)
        _reserve_decode_credits(target, decode_tokens)

        microbatch_id = target.next_microbatch_id
        for allocation in resolved.allocations:
            request = target.request(allocation.request_id)
            request.reserved_prefill_tokens = allocation.prefill_tokens
            request.reserved_decode_tokens = allocation.decode_tokens
            request.reserved_recompute_tokens = allocation.recompute_tokens
            request.inflight_microbatch_id = microbatch_id
            if allocation.recompute_tokens:
                request.lifecycle = RequestLifecycle.INFLIGHT_RECOMPUTE
            elif allocation.prefill_tokens:
                request.lifecycle = RequestLifecycle.INFLIGHT_PREFILL
            else:
                request.lifecycle = RequestLifecycle.INFLIGHT_DECODE

        admit_microbatch(
            replica,
            microbatch_id=microbatch_id,
            raw_action_index=resolved.raw_action_index,
            canonical_action_index=action.canonical_action_index,
            allocations=resolved.allocations,
            admitted_at=target.now,
            stage_service_times=service_times,
            pp_communication_times=communication_times,
            scheduler=config.scheduler,
            timing=config.timing,
        )
        if decode_tokens and target.decode_credits_available == 0:
            _stop_active_decodes_after_credit_exhaustion(target, target.now)
        target.next_microbatch_id += 1
    elif resolved.transition_kind == ControllerTransitionKind.WAIT:
        _advance_to_inplace(target, config, next_wait_boundary_time(target, config))

    target.next_player = Player.ADVERSARY
    _refresh_objective(target, config)
    _assert_state(target, config)
    return _outcome(
        target,
        config,
        transition_kind=resolved.transition_kind.name,
        started_at=started_at,
        objective_before=objective_before,
    )


def apply_adversary_action(
    state: GV4State,
    config: GV4EngineConfig,
    action: CanonicalAdversaryAction,
    *,
    prefill_time_estimator: LaunchPrefillTimeEstimator,
    inplace: bool = False,
) -> TransitionOutcome:
    """Apply one canonical adversary action and expose launches at the same tick."""

    _assert_state(state, config)
    target = state if inplace else state.clone()
    started_at = target.now
    objective_before = target.objective.total_cost
    resolved = action.action
    if action.canonical_action_index < 0:
        raise TransitionError("canonical action index must be nonnegative")
    current = resolve_adversary_action(
        target,
        config,
        raw_action_index=resolved.raw_action_index,
    )
    if current is None or current.canonical_key != resolved.canonical_key:
        raise TransitionError("adversary action is stale or illegal for this state")

    # Before the tick this is the forced no-op used by strict alternation.
    if target.now + config.timing.epsilon < target.next_adversary_tick:
        target.next_player = Player.CONTROLLER
        _assert_state(target, config)
        return _outcome(
            target,
            config,
            transition_kind="FORCED_NOOP",
            started_at=started_at,
            objective_before=objective_before,
        )

    prefill_duration = 0.0
    if resolved.launch_count:
        assert resolved.prefill_tokens is not None
        prefill_duration = float(prefill_time_estimator(resolved.prefill_tokens))
        if not math.isfinite(prefill_duration) or prefill_duration <= 0.0:
            raise TransitionError("prefill_time_estimator returned an invalid duration")

    _prune_launch_history(target, config, target.now)

    for request_id in resolved.stop_request_ids:
        _mark_stop(target, target.request(request_id), target.now)

    if resolved.launch_count:
        assert resolved.prefill_tokens is not None
        owner = (
            0
            if config.topology.num_replicas == 1
            and config.routing.implicit_single_replica_assignment
            else UNASSIGNED_REPLICA
        )
        deadline = round(
            target.now + config.slo.prefill_slowdown_factor * prefill_duration,
            config.timing.time_round_digits,
        )
        for _ in range(resolved.launch_count):
            request_id = target.next_request_id
            target.requests.append(
                RequestState(
                    request_id=request_id,
                    owner_replica_id=owner,
                    lifecycle=RequestLifecycle.WAITING_PREFILL,
                    arrival_time=target.now,
                    prefill_deadline=deadline,
                    decode_token_slo_sec=config.slo.decode_token_slo_sec,
                    original_prefill_tokens=resolved.prefill_tokens,
                    original_decode_tokens=(
                        config.request.max_decode_tokens_per_request
                    ),
                )
            )
            target.next_request_id += 1
        target.launch_history.append(
            LaunchRecord(
                launch_time=target.now,
                request_count=resolved.launch_count,
                prefill_tokens=resolved.launch_count * resolved.prefill_tokens,
            )
        )

    target.next_adversary_tick = round(
        target.now + config.timing.adversary_tick_sec,
        config.timing.time_round_digits,
    )
    target.next_player = Player.CONTROLLER
    _refresh_objective(target, config)
    _assert_state(target, config)
    return _outcome(
        target,
        config,
        transition_kind="ADVERSARY",
        started_at=started_at,
        objective_before=objective_before,
    )
