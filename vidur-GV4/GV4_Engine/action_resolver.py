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
    PREEMPT_ONLY = 3
    EVICT_AND_PREEMPT = 4


@dataclass(frozen=True, slots=True)
class ResolvedControllerAction:
    """One legal raw controller action reduced to its physical effects."""

    raw_action_index: int
    replica_id: int
    preemption_rule: str
    eviction_rule: str
    prefill_budget: int
    ordering_heuristic: str
    transition_kind: ControllerTransitionKind
    evicted_request_ids: tuple[int, ...]
    preempted_request_ids: tuple[int, ...]
    pending_preemption_request_ids: tuple[int, ...]
    allocations: tuple[BatchAllocation, ...]
    released_kv_blocks: int
    preempted_kv_blocks: int
    reserved_kv_blocks: int
    rank_kv_delta: tuple[tuple[int, int], ...]

    @property
    def canonical_key(self) -> tuple[object, ...]:
        work = tuple(
            (
                allocation.request_id,
                (
                    0
                    if allocation.prefill_tokens
                    else 1
                    if allocation.decode_tokens
                    else 2
                ),
                allocation.prefill_tokens,
                allocation.decode_tokens,
                allocation.recompute_tokens,
            )
            for allocation in self.allocations
        )
        return (
            int(self.transition_kind),
            self.evicted_request_ids,
            self.preempted_request_ids,
            self.pending_preemption_request_ids,
            work,
            self.rank_kv_delta,
        )

    @property
    def total_prefill_tokens(self) -> int:
        return sum(item.prefill_tokens for item in self.allocations)

    @property
    def total_decode_tokens(self) -> int:
        return sum(item.decode_tokens for item in self.allocations)

    @property
    def total_recompute_tokens(self) -> int:
        return sum(item.recompute_tokens for item in self.allocations)

    @property
    def total_prefill_class_tokens(self) -> int:
        """Tokens charged to the shared prefill/recomputation budget."""

        return self.total_prefill_tokens + self.total_recompute_tokens


@dataclass(frozen=True, slots=True)
class _PreemptionCandidate:
    """One request and the state it will have when preemption takes effect."""

    request: RequestState
    release_time: float
    released_blocks: int
    recompute_tokens: int
    recovery_deadline: float
    pending: bool


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


def _prefill_class_tokens(request: RequestState) -> int:
    """Return reconstruction work first, otherwise ordinary prefill work."""

    return request.remaining_recompute_tokens or request.remaining_prefill_tokens


def _prefill_class_deadline(request: RequestState) -> float:
    if request.remaining_recompute_tokens and request.is_decode_phase:
        return request.next_decode_deadline
    return request.prefill_deadline


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
        "preemption_targets_by_rule",
        "preemption_candidates",
        "recompute_estimates",
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
        self.preemption_targets_by_rule: dict[
            tuple[str, tuple[int, ...]], tuple[int, ...]
        ] = {}
        self.preemption_candidates: tuple[_PreemptionCandidate, ...] | None = None
        self.recompute_estimates: dict[int, float] = {}
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

    def _build_preemption_candidates(self) -> tuple[_PreemptionCandidate, ...]:
        if self.preemption_candidates is not None:
            return self.preemption_candidates

        candidates: list[_PreemptionCandidate] = []
        for request in self.state.requests:
            if request.owner_replica_id != self.replica_id:
                continue
            if request.lifecycle.is_terminal or request.lifecycle in (
                RequestLifecycle.STOP_PENDING,
                RequestLifecycle.DROP_PENDING,
                RequestLifecycle.PREEMPT_PENDING,
            ):
                continue

            if request.has_inflight_work:
                batch = self.replica.find_microbatch(request.inflight_microbatch_id)
                if batch is None:
                    raise ActionResolutionError(
                        "in-flight preemption candidate has no microbatch"
                    )
                allocation = next(
                    (
                        item
                        for item in batch.allocations
                        if item.request_id == request.request_id
                    ),
                    None,
                )
                if allocation is None:
                    raise ActionResolutionError(
                        "in-flight preemption candidate has no allocation"
                    )
                # A final decode token completes the request; there is nothing to resume.
                if allocation.decode_tokens and request.remaining_decode_tokens == 0:
                    continue
                release_time = batch.final_completion_time
                logical_after = (
                    request.logical_context_tokens
                    + allocation.prefill_tokens
                    + allocation.decode_tokens
                )
                released_blocks = (
                    request.committed_kv_blocks + request.reserved_kv_blocks
                )
                prefill_after = (
                    request.committed_prefill_tokens + allocation.prefill_tokens
                )
                if prefill_after < request.original_prefill_tokens:
                    deadline = request.prefill_deadline
                elif allocation.prefill_tokens or allocation.decode_tokens:
                    deadline = release_time + request.decode_token_slo_sec
                else:
                    deadline = request.next_decode_deadline
                pending = True
            else:
                release_time = self.state.now
                logical_after = request.logical_context_tokens
                released_blocks = request.committed_kv_blocks
                deadline = (
                    request.next_decode_deadline
                    if request.is_decode_phase
                    else request.prefill_deadline
                )
                pending = False

            if logical_after > 0 and released_blocks > 0:
                candidates.append(
                    _PreemptionCandidate(
                        request=request,
                        release_time=release_time,
                        released_blocks=released_blocks,
                        recompute_tokens=logical_after,
                        recovery_deadline=deadline,
                        pending=pending,
                    )
                )

        self.preemption_candidates = tuple(candidates)
        return self.preemption_candidates

    def _recompute_time(self, candidate: _PreemptionCandidate) -> float:
        cached = self.recompute_estimates.get(candidate.request.request_id)
        if cached is not None:
            return cached
        if self.prefill_time_estimator is None:
            raise ActionResolutionError(
                "timing-based preemption requires a prefill_time_estimator"
            )
        estimate = float(
            self.prefill_time_estimator(candidate.request, candidate.recompute_tokens)
        )
        if not math.isfinite(estimate) or estimate < 0.0:
            raise ActionResolutionError(
                "prefill_time_estimator returned an invalid duration"
            )
        self.recompute_estimates[candidate.request.request_id] = estimate
        return estimate

    def preemption_targets(
        self, rule: str, excluded_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        key = (rule, excluded_ids)
        cached = self.preemption_targets_by_rule.get(key)
        if cached is not None:
            return cached
        if rule == "preempt_none":
            self.preemption_targets_by_rule[key] = ()
            return ()

        excluded = set(excluded_ids)
        candidates = [
            item
            for item in self._build_preemption_candidates()
            if item.request.request_id not in excluded
        ]
        if not candidates:
            self.preemption_targets_by_rule[key] = ()
            return ()

        if rule == "preempt_min_recompute":
            selected = min(
                candidates,
                key=lambda item: (
                    item.recompute_tokens,
                    -item.released_blocks,
                    item.request.request_id,
                ),
            )
        elif rule == "preempt_largest_kv":
            selected = min(
                candidates,
                key=lambda item: (
                    -item.released_blocks,
                    item.recompute_tokens,
                    item.request.request_id,
                ),
            )
        elif rule == "preempt_max_recovery_slack":
            selected = max(
                candidates,
                key=lambda item: (
                    item.recovery_deadline
                    - item.release_time
                    - self._recompute_time(item),
                    item.released_blocks,
                    -item.request.request_id,
                ),
            )
        elif rule == "preempt_best_relief_cost":
            epsilon = self.config.timing.epsilon

            def relief_cost(item: _PreemptionCandidate) -> tuple[float, int, int]:
                recompute_time = self._recompute_time(item)
                lateness = max(
                    0.0,
                    item.release_time + recompute_time - item.recovery_deadline,
                )
                new_violation_cost = (
                    self.config.cost.violation_base_cost
                    if lateness > epsilon and not item.request.violation_recorded
                    else 0.0
                )
                recovery_cost = (
                    recompute_time
                    + new_violation_cost
                    + min(lateness, self.config.cost.lateness_cap_sec)
                )
                score = item.released_blocks / max(epsilon, recovery_cost)
                return score, item.released_blocks, -item.request.request_id

            selected = max(candidates, key=relief_cost)
        else:
            raise ActionResolutionError(f"unsupported preemption rule {rule!r}")

        targets = (selected.request.request_id,)
        self.preemption_targets_by_rule[key] = targets
        return targets

    def ordered_prefill_class(
        self, heuristic: str, excluded_ids: tuple[int, ...]
    ) -> tuple[RequestState, ...]:
        key = (excluded_ids, heuristic)
        cached = self.orders.get(key)
        if cached is not None:
            return cached

        excluded = set(excluded_ids)
        requests = [
            request
            for request in (*self.prefills, *self.decodes)
            if request.request_id not in excluded
            and _prefill_class_tokens(request) > 0
            and (
                request.remaining_recompute_tokens > 0
                or request.lifecycle == RequestLifecycle.WAITING_PREFILL
            )
        ]
        if heuristic == "SJF":
            requests.sort(key=lambda item: (_prefill_class_tokens(item), item.request_id))
        elif heuristic == "EDF":
            requests.sort(key=lambda item: (_prefill_class_deadline(item), item.request_id))
        elif heuristic == "LJF":
            requests.sort(key=lambda item: (-_prefill_class_tokens(item), item.request_id))
        elif heuristic == "LST":
            estimator = self.prefill_time_estimator
            if estimator is None and requests:
                raise ActionResolutionError(
                    "LST resolution requires a prefill_time_estimator"
                )

            def least_slack(request: RequestState) -> tuple[float, int]:
                assert estimator is not None
                estimate = float(estimator(request, _prefill_class_tokens(request)))
                if not math.isfinite(estimate) or estimate < 0.0:
                    raise ActionResolutionError(
                        "prefill_time_estimator returned an invalid duration"
                    )
                return (
                    _prefill_class_deadline(request) - self.state.now - estimate,
                    request.request_id,
                )

            requests.sort(key=least_slack)
        else:
            raise ActionResolutionError(f"unsupported ordering heuristic {heuristic!r}")

        ordered = tuple(requests)
        self.orders[key] = ordered
        return ordered


def _fit_prefill_class_to_kv(
    request: RequestState,
    desired_tokens: int,
    free_blocks: int,
    block_size_tokens: int,
) -> tuple[int, int, bool]:
    """Fit ordinary prefill or reconstruction into the same KV budget."""

    owned_blocks = request.committed_kv_blocks + request.reserved_kv_blocks
    maximum_resident = (owned_blocks + free_blocks) * block_size_tokens
    tokens = min(desired_tokens, max(0, maximum_resident - request.resident_tokens))
    if tokens <= 0:
        return 0, 0, False
    is_recompute = request.remaining_recompute_tokens > 0
    if is_recompute:
        blocks = additional_blocks_for_work(
            request,
            recompute_tokens=tokens,
            block_size_tokens=block_size_tokens,
        )
    else:
        blocks = additional_blocks_for_work(
            request,
            prefill_tokens=tokens,
            block_size_tokens=block_size_tokens,
        )
    return tokens, blocks, is_recompute


def _resolve_controller_raw(
    context: _ControllerContext,
    raw_action_index: int,
) -> ResolvedControllerAction | None:
    state = context.state
    config = context.config
    action_config = config.controller_actions
    try:
        preemption_rule, eviction_rule, budget, heuristic = (
            action_config.raw_action_components(raw_action_index)
        )
    except ValueError as error:
        raise ActionResolutionError(str(error)) from error

    if state.next_player != Player.CONTROLLER:
        return None

    if not config.scheduler.request_preemption_enabled:
        preemption_rule = "preempt_none"

    # Memory decisions remain legal while stage zero is busy. Only the batch
    # portion of an action depends on pipeline admission.
    pipeline_open = can_admit_microbatch(
        context.replica,
        admitted_at=state.now,
        scheduler=config.scheduler,
        timing=config.timing,
    )
    if budget == 0 and heuristic != action_config.ordering_heuristics[0]:
        return None

    evicted_ids = context.eviction_targets(eviction_rule)
    if eviction_rule != "evict_none" and not evicted_ids:
        return None
    preempted_ids = context.preemption_targets(preemption_rule, evicted_ids)
    if preemption_rule != "preempt_none" and not preempted_ids:
        return None

    pending_preemption_ids = tuple(
        request_id
        for request_id in preempted_ids
        if state.request(request_id).has_inflight_work
    )
    immediate_preemption_ids = tuple(
        request_id
        for request_id in preempted_ids
        if not state.request(request_id).has_inflight_work
    )
    excluded_ids = tuple(sorted((*evicted_ids, *preempted_ids)))

    ordered_prefill_class = context.ordered_prefill_class(heuristic, excluded_ids)
    total_prefill_class = sum(
        _prefill_class_tokens(item) for item in ordered_prefill_class
    )

    if not pipeline_open:
        # Avoid aliases whose scheduling fields cannot take effect at this time.
        if budget != 0 or heuristic != action_config.ordering_heuristics[0]:
            return None
    elif budget > 0:
        if total_prefill_class == 0:
            return None
        minimum_positive = next(
            value for value in action_config.prefill_budget_options if value > 0
        )
        if budget > total_prefill_class and not (
            total_prefill_class < minimum_positive and budget == minimum_positive
        ):
            return None

    evicted_blocks = sum(
        state.request(request_id).committed_kv_blocks for request_id in evicted_ids
    )
    preempted_blocks = sum(
        state.request(request_id).committed_kv_blocks
        for request_id in immediate_preemption_ids
    )
    released_blocks = evicted_blocks + preempted_blocks
    free_blocks = free_logical_blocks(context.replica) + released_blocks
    block_size = config.kv_cache.block_size_tokens
    tokens_left = config.scheduler.max_batch_tokens
    sequences_left = config.scheduler.max_sequences
    desired_left = min(budget, tokens_left)

    allocations: list[BatchAllocation] = []
    if pipeline_open:
        for request in ordered_prefill_class:
            if desired_left <= 0 or sequences_left <= 0:
                break
            desired = min(_prefill_class_tokens(request), desired_left)
            tokens, blocks, is_recompute = _fit_prefill_class_to_kv(
                request, desired, free_blocks, block_size
            )
            if tokens <= 0:
                continue
            allocations.append(
                BatchAllocation(
                    request.request_id,
                    prefill_tokens=0 if is_recompute else tokens,
                    recompute_tokens=tokens if is_recompute else 0,
                    new_kv_blocks=blocks,
                )
            )
            desired_left -= tokens
            tokens_left -= tokens
            sequences_left -= 1
            free_blocks -= blocks

        funded_decode_slots = state.decode_credits_available
        excluded = set(excluded_ids)
        decode_candidates = [
            request
            for request in context.decodes
            if request.request_id not in excluded
            and request.remaining_recompute_tokens == 0
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
    elif evicted_ids and preempted_ids:
        kind = ControllerTransitionKind.EVICT_AND_PREEMPT
    elif evicted_ids:
        kind = ControllerTransitionKind.EVICT_ONLY
    elif preempted_ids:
        kind = ControllerTransitionKind.PREEMPT_ONLY
    else:
        kind = ControllerTransitionKind.WAIT
        if raw_action_index != 0:
            return None

    net_blocks = reserved_blocks - released_blocks
    rank_delta = tuple((rank_id, net_blocks) for rank_id in context.replica.rank_ids)
    return ResolvedControllerAction(
        raw_action_index=raw_action_index,
        replica_id=context.replica_id,
        preemption_rule=preemption_rule,
        eviction_rule=eviction_rule,
        prefill_budget=budget,
        ordering_heuristic=heuristic,
        transition_kind=kind,
        evicted_request_ids=evicted_ids,
        preempted_request_ids=preempted_ids,
        pending_preemption_request_ids=pending_preemption_ids,
        allocations=allocation_tuple,
        released_kv_blocks=released_blocks,
        preempted_kv_blocks=preempted_blocks,
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
        if (
            request.lifecycle
            in (RequestLifecycle.WAITING_DECODE, RequestLifecycle.INFLIGHT_DECODE)
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
        )
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
