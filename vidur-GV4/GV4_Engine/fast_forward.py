"""Deterministic decode-only progression between external GV4 decisions."""

from __future__ import annotations

from collections.abc import Callable

from .action_resolver import (
    CanonicalAdversaryAction,
    CanonicalControllerAction,
    ControllerTransitionKind,
    ResolvedControllerAction,
    resolve_adversary_action,
    resolve_controller_action,
)
from .config import GV4EngineConfig
from .pipeline_calendar import can_admit_microbatch
from .state import GV4State, Player, RequestLifecycle
from .transition_engine import (
    advance_to,
    apply_adversary_action,
    apply_controller_action,
    next_wait_boundary_time,
)


BatchTimingProvider = Callable[
    [GV4State, ResolvedControllerAction],
    tuple[tuple[float, ...], tuple[float, ...]],
]


class FastForwardError(RuntimeError):
    """Raised when deterministic fast-forward cannot make safe progress."""


def _has_active_prefill(state: GV4State) -> bool:
    return any(
        not request.lifecycle.is_terminal
        and (
            request.remaining_prefill_tokens > 0
            or request.reserved_prefill_tokens > 0
        )
        for request in state.requests
    )


def _has_inflight_decode(state: GV4State) -> bool:
    return any(request.reserved_decode_tokens > 0 for request in state.requests)


def _forced_adversary_noop(
    state: GV4State, config: GV4EngineConfig
) -> None:
    resolved = resolve_adversary_action(state, config, raw_action_index=0)
    if resolved is None:
        raise FastForwardError("pre-tick adversary no-op is unexpectedly masked")
    action = CanonicalAdversaryAction(0, resolved, (0,))
    apply_adversary_action(
        state,
        config,
        action,
        prefill_time_estimator=lambda _tokens: 0.0,
        inplace=True,
    )


def _next_decode_batch(
    state: GV4State, config: GV4EngineConfig
) -> tuple[CanonicalControllerAction | None, bool]:
    """Find the first stable replica batch; report a free but KV-blocked replica."""

    blocked_at_free_stage = False
    for replica in state.replicas:
        has_waiter = any(
            request.owner_replica_id == replica.replica_id
            and request.lifecycle == RequestLifecycle.WAITING_DECODE
            for request in state.requests
        )
        if not has_waiter:
            continue
        if not can_admit_microbatch(
            replica,
            admitted_at=state.now,
            scheduler=config.scheduler,
            timing=config.timing,
        ):
            continue

        resolved = resolve_controller_action(
            state,
            config,
            replica_id=replica.replica_id,
            raw_action_index=0,
        )
        if resolved is None:
            raise FastForwardError("decode-only raw action is unexpectedly masked")
        if resolved.transition_kind == ControllerTransitionKind.BATCH:
            if resolved.total_prefill_tokens:
                raise FastForwardError("decode fast-forward resolved prefill work")
            return CanonicalControllerAction(0, resolved, (0,)), False
        blocked_at_free_stage = True

    return None, blocked_at_free_stage


def jump_idle_to_next_tick(
    state: GV4State,
    config: GV4EngineConfig,
    *,
    inplace: bool = False,
) -> GV4State:
    """Move a fully idle state directly to its next adversary tick."""

    target = state if inplace else state.clone()
    if any(not request.lifecycle.is_terminal for request in target.requests):
        raise FastForwardError("idle jump received an active request")
    if any(replica.inflight_microbatches for replica in target.replicas):
        raise FastForwardError("idle jump received in-flight work")
    advance_to(target, config, target.next_adversary_tick, inplace=True)
    target.next_player = Player.ADVERSARY
    return target


def fast_forward_decode_only_to_next_tick(
    state: GV4State,
    config: GV4EngineConfig,
    *,
    timing_provider: BatchTimingProvider,
    inplace: bool = False,
) -> GV4State:
    """Run forced decode batches without crossing the next adversary tick.

    A waiting decode that is KV-blocked at a free stage is not forced: the
    function returns a controller state so MCTS can choose an eviction policy.
    """

    target = state if inplace else state.clone()
    epsilon = config.timing.epsilon
    zero_time_steps = 0

    if target.now + epsilon >= target.next_adversary_tick:
        target.next_player = Player.ADVERSARY
        return target
    if _has_active_prefill(target):
        return target

    while target.now + epsilon < target.next_adversary_tick:
        if target.next_player == Player.ADVERSARY:
            _forced_adversary_noop(target, config)
            zero_time_steps += 1

        action, kv_blocked = _next_decode_batch(target, config)
        if action is not None:
            service, communication = timing_provider(target, action.action)
            apply_controller_action(
                target,
                config,
                action,
                stage_service_times=tuple(service),
                pp_communication_times=tuple(communication),
                inplace=True,
            )
            zero_time_steps += 1
        elif kv_blocked:
            return target
        elif _has_inflight_decode(target):
            boundary = next_wait_boundary_time(target, config)
            advance_to(target, config, boundary, inplace=True)
            zero_time_steps = 0
        else:
            return jump_idle_to_next_tick(target, config, inplace=True)

        if zero_time_steps > config.timing.max_zero_time_transitions_per_boundary:
            raise FastForwardError("too many zero-time fast-forward transitions")

    target.next_player = Player.ADVERSARY
    return target
