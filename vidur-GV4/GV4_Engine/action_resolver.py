"""Pure validation and canonicalization of GV4 player actions.

The resolver never mutates :class:`GV4State`. It converts a raw policy index
into the exact request, token, and KV effects that the transition engine may
apply. Keeping this layer pure makes action expansion safe for MCTS and gives
the future native implementation a small deterministic contract to mirror.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
import math

from .config import GV4EngineConfig
from .kv_ledger import additional_blocks_for_work, free_logical_blocks
from .pipeline_calendar import can_admit_microbatch
from .state import (
    BatchAllocation,
    GV4State,
    Player,
    RequestLifecycle,
    RequestState,
)


__all__ = [
    "ActionResolutionError",
    "CanonicalAdversaryAction",
    "CanonicalControllerAction",
    "ControllerTransitionKind",
    "PrefillTimeEstimator",
    "ResolvedAdversaryAction",
    "ResolvedControllerAction",
    "resolve_adversary_action",
    "resolve_adversary_actions",
    "resolve_controller_action",
    "resolve_controller_actions",
]


PrefillTimeEstimator = Callable[[RequestState, int], float]


class ActionResolutionError(ValueError):
    """Raised when action resolution cannot satisfy its deterministic contract."""


class ControllerTransitionKind(IntEnum):
    WAIT = 0
    EVICT_ONLY = 1
    BATCH = 2


@dataclass(frozen=True, slots=True)
class ResolvedControllerAction:
    """One legal raw controller action reduced to its physical effects."""

    raw_action_index: int
    replica_id: int
    eviction_rule: str
    prefill_budget: int
    ordering_heuristic: str
    transition_kind: ControllerTransitionKind
    evicted_request_ids: tuple[int, ...]
    allocations: tuple[BatchAllocation, ...]
    released_kv_blocks: int
    reserved_kv_blocks: int
    rank_kv_delta: tuple[tuple[int, int], ...]

    @property
    def canonical_key(self) -> tuple[object, ...]:
        work = tuple(
            (
                allocation.request_id,
                0 if allocation.prefill_tokens else 1,
                allocation.prefill_tokens,
                allocation.decode_tokens,
            )
            for allocation in self.allocations
        )
        return (
            int(self.transition_kind),
            self.evicted_request_ids,
            work,
            self.rank_kv_delta,
        )

    @property
    def total_prefill_tokens(self) -> int:
        return sum(item.prefill_tokens for item in self.allocations)

    @property
    def total_decode_tokens(self) -> int:
        return sum(item.decode_tokens for item in self.allocations)


@dataclass(frozen=True, slots=True)
class CanonicalControllerAction:
    """One MCTS edge and all raw policy indices that produce that edge."""

    canonical_action_index: int
    action: ResolvedControllerAction
    equivalent_raw_indices: tuple[int, ...]

    @property
    def representative_raw_index(self) -> int:
        return self.action.raw_action_index


@dataclass(frozen=True, slots=True)
class ResolvedAdversaryAction:
    """A strictly legal adversary launch/stop action."""

    raw_action_index: int
    launch_count: int
    prefill_tokens: int | None
    stop_rule: str
    stop_request_ids: tuple[int, ...]

    @property
    def canonical_key(self) -> tuple[object, ...]:
        """Describe only the state-changing effects of this raw action."""

        return (
            self.launch_count,
            self.prefill_tokens,
            tuple(sorted(self.stop_request_ids)),
        )


@dataclass(frozen=True, slots=True)
class CanonicalAdversaryAction:
    """One MCTS edge and all raw adversary indices that produce that edge."""

    canonical_action_index: int
    action: ResolvedAdversaryAction
    equivalent_raw_indices: tuple[int, ...]

    @property
    def representative_raw_index(self) -> int:
        return self.action.raw_action_index


def _prefill_lateness(request: RequestState, now: float) -> float:
    return max(request.prefill_lateness_sec, now - request.prefill_deadline, 0.0)


def _decode_lateness(request: RequestState) -> float:
    return max(0.0, request.prefill_lateness_sec + request.decode_lateness_sec)


class _ControllerContext:
    """Caches state-derived orderings while all raw actions are expanded."""

    __slots__ = (
        "state",
        "config",
        "replica_id",
        "replica",
        "prefill_time_estimator",
        "prefills",
        "decodes",
        "targets_by_rule",
        "orders",
    )

    def __init__(
        self,
        state: GV4State,
        config: GV4EngineConfig,
        replica_id: int,
        prefill_time_estimator: PrefillTimeEstimator | None,
    ) -> None:
        if not 0 <= replica_id < len(state.replicas):
            raise ActionResolutionError(f"unknown replica ID {replica_id}")
        self.state = state
        self.config = config
        self.replica_id = replica_id
        self.replica = state.replica(replica_id)
        self.prefill_time_estimator = prefill_time_estimator
        self.prefills = tuple(
            request
            for request in state.requests
            if request.owner_replica_id == replica_id
            and request.lifecycle == RequestLifecycle.WAITING_PREFILL
        )
        self.decodes = tuple(
            request
            for request in state.requests
            if request.owner_replica_id == replica_id
            and request.lifecycle == RequestLifecycle.WAITING_DECODE
        )
        self.targets_by_rule: dict[str, tuple[int, ...]] = {}
        self.orders: dict[tuple[tuple[int, ...], str], tuple[RequestState, ...]] = {}

    @property
    def has_waiting_work(self) -> bool:
        return bool(self.prefills or self.decodes)

    def eviction_targets(self, rule: str) -> tuple[int, ...]:
        cached = self.targets_by_rule.get(rule)
        if cached is not None:
            return cached

        resident_prefills = tuple(
            request for request in self.prefills if request.committed_kv_blocks > 0
        )
        resident_decodes = tuple(
            request for request in self.decodes if request.committed_kv_blocks > 0
        )
        now = self.state.now

        if rule == "evict_none":
            targets: tuple[int, ...] = ()
        elif rule == "evict_largest_prefill" and resident_prefills:
            request = max(
                resident_prefills,
                key=lambda item: (item.remaining_prefill_tokens, -item.request_id),
            )
            targets = (request.request_id,)
        elif rule == "evict_earliest_prefill_deadline" and resident_prefills:
            request = min(
                resident_prefills,
                key=lambda item: (item.prefill_deadline, item.request_id),
            )
            targets = (request.request_id,)
        elif rule == "evict_prefill_missed_deadline":
            targets = tuple(
                request.request_id
                for request in resident_prefills
                if _prefill_lateness(request, now) > self.config.timing.epsilon
            )
        elif rule == "evict_prefill_lateness_over_0p5":
            targets = tuple(
                request.request_id
                for request in resident_prefills
                if _prefill_lateness(request, now) > 0.5
            )
        elif rule == "evict_longest_decode" and resident_decodes:
            request = max(
                resident_decodes,
                key=lambda item: (item.committed_decode_tokens, -item.request_id),
            )
            targets = (request.request_id,)
        elif rule == "evict_decode_lateness_over_0p5":
            targets = tuple(
                request.request_id
                for request in resident_decodes
                if _decode_lateness(request) > 0.5
            )
        elif rule == "evict_prefill_highest_lateness" and resident_prefills:
            request = max(
                resident_prefills,
                key=lambda item: (_prefill_lateness(item, now), -item.request_id),
            )
            targets = (
                (request.request_id,)
                if _prefill_lateness(request, now) > self.config.timing.epsilon
                else ()
            )
        elif rule == "evict_decode_highest_lateness" and resident_decodes:
            request = max(
                resident_decodes,
                key=lambda item: (_decode_lateness(item), -item.request_id),
            )
            targets = (
                (request.request_id,)
                if _decode_lateness(request) > self.config.timing.epsilon
                else ()
            )
        else:
            targets = ()

        targets = tuple(sorted(targets))
        self.targets_by_rule[rule] = targets
        return targets

    def ordered_prefills(
        self, heuristic: str, evicted_ids: tuple[int, ...]
    ) -> tuple[RequestState, ...]:
        key = (evicted_ids, heuristic)
        cached = self.orders.get(key)
        if cached is not None:
            return cached

        evicted = set(evicted_ids)
        requests = [
            request for request in self.prefills if request.request_id not in evicted
        ]
        if heuristic == "SJF":
            requests.sort(
                key=lambda item: (item.remaining_prefill_tokens, item.request_id)
            )
        elif heuristic == "EDF":
            requests.sort(key=lambda item: (item.prefill_deadline, item.request_id))
        elif heuristic == "LJF":
            requests.sort(
                key=lambda item: (-item.remaining_prefill_tokens, item.request_id)
            )
        elif heuristic == "LST":
            estimator = self.prefill_time_estimator
            if estimator is None and requests:
                raise ActionResolutionError(
                    "LST resolution requires a prefill_time_estimator"
                )

            def least_slack(request: RequestState) -> tuple[float, int]:
                assert estimator is not None
                estimate = float(estimator(request, request.remaining_prefill_tokens))
                if not math.isfinite(estimate) or estimate < 0.0:
                    raise ActionResolutionError(
                        "prefill_time_estimator returned an invalid duration"
                    )
                return (
                    request.prefill_deadline - self.state.now - estimate,
                    request.request_id,
                )

            requests.sort(key=least_slack)
        else:
            raise ActionResolutionError(f"unsupported ordering heuristic {heuristic!r}")

        ordered = tuple(requests)
        self.orders[key] = ordered
        return ordered


def _fit_prefill_to_kv(
    request: RequestState,
    desired_tokens: int,
    free_blocks: int,
    block_size_tokens: int,
) -> tuple[int, int]:
    """Return the largest desired prefix and its exact new-block demand."""

    owned_blocks = request.committed_kv_blocks + request.reserved_kv_blocks
    maximum_resident = (owned_blocks + free_blocks) * block_size_tokens
    tokens = min(desired_tokens, max(0, maximum_resident - request.resident_tokens))
    if tokens <= 0:
        return 0, 0
    blocks = additional_blocks_for_work(
        request,
        prefill_tokens=tokens,
        block_size_tokens=block_size_tokens,
    )
    return tokens, blocks


def _resolve_controller_raw(
    context: _ControllerContext,
    raw_action_index: int,
) -> ResolvedControllerAction | None:
    state = context.state
    config = context.config
    action_config = config.controller_actions
    try:
        eviction_rule, budget, heuristic = action_config.raw_action_components(
            raw_action_index
        )
    except ValueError as error:
        raise ActionResolutionError(str(error)) from error

    if state.next_player != Player.CONTROLLER:
        return None
    if not can_admit_microbatch(
        context.replica,
        admitted_at=state.now,
        scheduler=config.scheduler,
        timing=config.timing,
    ):
        if raw_action_index != 0:
            return None
        return ResolvedControllerAction(
            raw_action_index=0,
            replica_id=context.replica_id,
            eviction_rule=eviction_rule,
            prefill_budget=budget,
            ordering_heuristic=heuristic,
            transition_kind=ControllerTransitionKind.WAIT,
            evicted_request_ids=(),
            allocations=(),
            released_kv_blocks=0,
            reserved_kv_blocks=0,
            rank_kv_delta=tuple(
                (rank_id, 0) for rank_id in context.replica.rank_ids
            ),
        )

    # Preserve GV3's strict duplicate masks around empty and zero-budget actions.
    if not context.has_waiting_work and raw_action_index != 0:
        return None
    if budget == 0 and heuristic != action_config.ordering_heuristics[0]:
        return None

    evicted_ids = context.eviction_targets(eviction_rule)
    if eviction_rule != "evict_none" and not evicted_ids:
        return None

    evicted = set(evicted_ids)
    ordered_prefills = context.ordered_prefills(heuristic, evicted_ids)
    total_prefill = sum(item.remaining_prefill_tokens for item in ordered_prefills)
    if budget > 0:
        if total_prefill == 0:
            return None
        minimum_positive = next(
            value for value in action_config.prefill_budget_options if value > 0
        )
        if budget > total_prefill and not (
            total_prefill < minimum_positive and budget == minimum_positive
        ):
            return None

    released_blocks = sum(
        state.request(request_id).committed_kv_blocks for request_id in evicted_ids
    )
    free_blocks = free_logical_blocks(context.replica) + released_blocks
    block_size = config.kv_cache.block_size_tokens
    tokens_left = config.scheduler.max_batch_tokens
    sequences_left = config.scheduler.max_sequences
    desired_left = min(budget, tokens_left)

    allocations: list[BatchAllocation] = []
    for request in ordered_prefills:
        if desired_left <= 0 or sequences_left <= 0:
            break
        desired = min(request.remaining_prefill_tokens, desired_left)
        tokens, blocks = _fit_prefill_to_kv(
            request, desired, free_blocks, block_size
        )
        if tokens <= 0:
            continue
        allocations.append(
            BatchAllocation(
                request.request_id,
                prefill_tokens=tokens,
                new_kv_blocks=blocks,
            )
        )
        desired_left -= tokens
        tokens_left -= tokens
        sequences_left -= 1
        free_blocks -= blocks

    funded_decode_slots = state.decode_credits_available
    decode_candidates = [
        request
        for request in context.decodes
        if request.request_id not in evicted
    ]
    zero_block_decodes: list[RequestState] = []
    boundary_decodes: list[RequestState] = []
    for request in decode_candidates:
        blocks = additional_blocks_for_work(
            request,
            decode_tokens=1,
            block_size_tokens=block_size,
        )
        (zero_block_decodes if blocks == 0 else boundary_decodes).append(request)

    for request in (*zero_block_decodes, *boundary_decodes):
        # Credit limits adversary-funded work; it is not a controller choice.
        if funded_decode_slots <= 0 or tokens_left <= 0 or sequences_left <= 0:
            break
        blocks = additional_blocks_for_work(
            request,
            decode_tokens=1,
            block_size_tokens=block_size,
        )
        if blocks > free_blocks:
            continue
        allocations.append(
            BatchAllocation(
                request.request_id,
                decode_tokens=1,
                new_kv_blocks=blocks,
            )
        )
        funded_decode_slots -= 1
        tokens_left -= 1
        sequences_left -= 1
        free_blocks -= blocks

    allocations.sort(key=lambda item: item.request_id)
    allocation_tuple = tuple(allocations)
    reserved_blocks = sum(item.new_kv_blocks for item in allocation_tuple)

    if allocation_tuple:
        kind = ControllerTransitionKind.BATCH
    elif evicted_ids:
        kind = ControllerTransitionKind.EVICT_ONLY
    else:
        kind = ControllerTransitionKind.WAIT
        if raw_action_index != 0:
            return None

    net_blocks = reserved_blocks - released_blocks
    rank_delta = tuple((rank_id, net_blocks) for rank_id in context.replica.rank_ids)
    return ResolvedControllerAction(
        raw_action_index=raw_action_index,
        replica_id=context.replica_id,
        eviction_rule=eviction_rule,
        prefill_budget=budget,
        ordering_heuristic=heuristic,
        transition_kind=kind,
        evicted_request_ids=evicted_ids,
        allocations=allocation_tuple,
        released_kv_blocks=released_blocks,
        reserved_kv_blocks=reserved_blocks,
        rank_kv_delta=rank_delta,
    )


def resolve_controller_action(
    state: GV4State,
    config: GV4EngineConfig,
    *,
    replica_id: int,
    raw_action_index: int,
    prefill_time_estimator: PrefillTimeEstimator | None = None,
) -> ResolvedControllerAction | None:
    """Resolve one raw controller index; return ``None`` when strictly masked."""

    context = _ControllerContext(state, config, replica_id, prefill_time_estimator)
    return _resolve_controller_raw(context, raw_action_index)


def resolve_controller_actions(
    state: GV4State,
    config: GV4EngineConfig,
    *,
    replica_id: int,
    prefill_time_estimator: PrefillTimeEstimator | None = None,
) -> tuple[
    tuple[ResolvedControllerAction | None, ...],
    tuple[CanonicalControllerAction, ...],
]:
    """Resolve the fixed raw space and merge equivalent physical transitions."""

    context = _ControllerContext(state, config, replica_id, prefill_time_estimator)
    raw_actions: list[ResolvedControllerAction | None] = []
    grouped: dict[tuple[object, ...], list[ResolvedControllerAction]] = {}

    for raw_index in range(config.controller_actions.raw_action_count):
        action = _resolve_controller_raw(context, raw_index)
        raw_actions.append(action)
        if action is not None:
            grouped.setdefault(action.canonical_key, []).append(action)

    canonical: list[CanonicalControllerAction] = []
    for canonical_index, aliases in enumerate(grouped.values()):
        representative = min(aliases, key=lambda item: item.raw_action_index)
        canonical.append(
            CanonicalControllerAction(
                canonical_action_index=canonical_index,
                action=representative,
                equivalent_raw_indices=tuple(
                    item.raw_action_index
                    for item in sorted(aliases, key=lambda item: item.raw_action_index)
                ),
            )
        )
    return tuple(raw_actions), tuple(canonical)


def _adversary_stop_ids(
    state: GV4State,
    rule: str,
) -> tuple[int, ...]:
    decodes = [
        request
        for request in state.requests
        if request.lifecycle
        in (RequestLifecycle.WAITING_DECODE, RequestLifecycle.INFLIGHT_DECODE)
    ]
    if rule == "stop_none":
        return ()
    if not decodes:
        return ()
    if rule == "stop_longest_decode":
        request = max(
            decodes,
            key=lambda item: (item.committed_decode_tokens, -item.request_id),
        )
        return (request.request_id,)
    if rule == "stop_shortest_decode":
        request = min(
            decodes,
            key=lambda item: (item.committed_decode_tokens, item.request_id),
        )
        return (request.request_id,)
    if rule == "stop_all_decodes_over_512":
        return tuple(
            request.request_id
            for request in decodes
            if request.committed_decode_tokens > 512
        )
    if rule == "stop_all_decodes_over_216":
        return tuple(
            request.request_id
            for request in decodes
            if request.committed_decode_tokens > 216
        )
    raise ActionResolutionError(f"unsupported stop rule {rule!r}")


def resolve_adversary_action(
    state: GV4State,
    config: GV4EngineConfig,
    *,
    raw_action_index: int,
) -> ResolvedAdversaryAction | None:
    """Resolve one adversary index with strict tick/window/stop masks."""

    try:
        launch_count, prefill_tokens, stop_rule = (
            config.adversary_actions.raw_action_components(raw_action_index)
        )
    except ValueError as error:
        raise ActionResolutionError(str(error)) from error

    if state.next_player != Player.ADVERSARY:
        return None
    if state.now + config.timing.epsilon < state.next_adversary_tick:
        return (
            ResolvedAdversaryAction(raw_action_index, 0, None, "stop_none", ())
            if raw_action_index == 0
            else None
        )

    stop_ids = _adversary_stop_ids(state, stop_rule)
    if stop_rule != "stop_none" and not stop_ids:
        return None

    cutoff = state.now - config.timing.launch_window_sec
    used_count = sum(
        record.request_count
        for record in state.launch_history
        if record.launch_time > cutoff + config.timing.epsilon
    )
    used_prefill = sum(
        record.prefill_tokens
        for record in state.launch_history
        if record.launch_time > cutoff + config.timing.epsilon
    )
    if launch_count:
        assert prefill_tokens is not None
        prefill_cap = (
            config.request.target_prefill_tokens_per_request_window_average
            * config.timing.max_requests_per_launch_window
        )
        if used_count + launch_count > config.timing.max_requests_per_launch_window:
            return None
        if used_prefill + launch_count * prefill_tokens > prefill_cap:
            return None
        if len(state.requests) + launch_count > config.layout.max_requests:
            return None
        if config.routing.enabled:
            unassigned = sum(
                request.owner_replica_id < 0 and not request.lifecycle.is_terminal
                for request in state.requests
            )
            if unassigned + launch_count > config.layout.max_unassigned_requests:
                return None

    return ResolvedAdversaryAction(
        raw_action_index=raw_action_index,
        launch_count=launch_count,
        prefill_tokens=prefill_tokens,
        stop_rule=stop_rule,
        stop_request_ids=stop_ids,
    )


def resolve_adversary_actions(
    state: GV4State,
    config: GV4EngineConfig,
) -> tuple[
    tuple[ResolvedAdversaryAction | None, ...],
    tuple[CanonicalAdversaryAction, ...],
]:
    """Resolve the fixed raw space and merge equivalent physical transitions."""

    raw_actions: list[ResolvedAdversaryAction | None] = []
    grouped: dict[tuple[object, ...], list[ResolvedAdversaryAction]] = {}

    for raw_index in range(config.adversary_actions.raw_action_count):
        action = resolve_adversary_action(
            state,
            config,
            raw_action_index=raw_index,
        )
        raw_actions.append(action)
        if action is not None:
            grouped.setdefault(action.canonical_key, []).append(action)

    canonical: list[CanonicalAdversaryAction] = []
    for canonical_index, aliases in enumerate(grouped.values()):
        representative = min(aliases, key=lambda item: item.raw_action_index)
        canonical.append(
            CanonicalAdversaryAction(
                canonical_action_index=canonical_index,
                action=representative,
                equivalent_raw_indices=tuple(
                    item.raw_action_index
                    for item in sorted(aliases, key=lambda item: item.raw_action_index)
                ),
            )
        )
    return tuple(raw_actions), tuple(canonical)
