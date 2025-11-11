from __future__ import annotations

import random
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

from vidur.entities import Request
from vidur.events.global_schedule_event import GlobalScheduleEvent
from vidur.events.replica_schedule_event import ReplicaScheduleEvent
from vidur.events.request_arrival_event import RequestArrivalEvent
from vidur.simulator import Simulator
from vidur.utils.slo_manager import SLOManager
from vidur.types import EventType
from .config import MCTSConstraintConfig, MCTSExploreConfig
from .prefill_calibrator import PrefillProfile


## These are the dimensions along which adversary will take an action. Some combination of these within constraints
@dataclass
class AdversaryRequestSpec:
    prefill_tokens: int
    decode_tokens: int
    prefill_slo: float
    decode_slo: float


## Selected combination by adversary on its turn
@dataclass
class AdversaryAction:
    requests: List[AdversaryRequestSpec] = field(default_factory=list)


## These are the dimensions along which the controller will take an action.
@dataclass
class ControllerAction:
    token_budget: int
    selected_request_ids: Optional[List[int]] = None
    token_allocations: Dict[int, int] = field(default_factory=dict)
    prefill_allocations: Dict[int, int] = field(default_factory=dict)
    decode_allocations: Dict[int, int] = field(default_factory=dict)


@dataclass
class _ControllerBudgetTracker:
    allocations: Dict[int, Tuple[int, int]]
    baseline_prefill: Dict[int, int]
    baseline_decode: Dict[int, int]

    def is_satisfied(self, lookup: Dict[int, Request]) -> bool:
        for rid, (prefill_budget, decode_budget) in self.allocations.items():
            req = lookup.get(rid)
            if req is None:
                continue
            gained_prefill = (
                req.num_processed_prefill_tokens - self.baseline_prefill.get(rid, 0)
            )
            if gained_prefill < prefill_budget:
                return False
            gained_decode = (
                req.num_processed_decode_tokens - self.baseline_decode.get(rid, 0)
            )
            if gained_decode < decode_budget:
                return False
        return True

## MCTS node carrying useful states for propogating upwards + helping in exploitation vs exploration goal
@dataclass
class VidurGameStats:
    """Lightweight bookkeeping attached to a simulator snapshot."""

    requests_generated: int = 0
    requests_completed: int = 0
    slo_violations: int = 0
    slo_lateness_sum: float = 0.0
    recent_arrivals: List[float] = field(default_factory=list)  # absolute times
    completed_request_ids: Set[int] = field(default_factory=set)

    def clone(self) -> "VidurGameStats":
        return VidurGameStats(
            requests_generated=self.requests_generated,
            requests_completed=self.requests_completed,
            slo_violations=self.slo_violations,
            slo_lateness_sum=self.slo_lateness_sum,
            recent_arrivals=list(self.recent_arrivals),
            completed_request_ids=set(self.completed_request_ids),
        )

## Essentially checkpoints the state so it can return back to it to run a different simulation
@dataclass
class VidurMCTSState:
    simulator: Simulator
    stats: VidurGameStats

    def fork(self) -> "VidurMCTSState":
        return VidurMCTSState(self.simulator.fork(), self.stats.clone())

## 
class VidurMCTSEnvironment:
    """Wraps a Vidur simulator to provide MCTS-compatible transitions."""

    def __init__(
        self,
        base_simulator: Simulator,
        constraints: MCTSConstraintConfig,
        explore_cfg: MCTSExploreConfig,
    ) -> None:
        self._base = base_simulator
        self._constraints = constraints
        self._cfg = explore_cfg
        self._slo_manager = SLOManager(base_simulator._config.slo_config)
        self._rng = random.Random(base_simulator._config.seed)
        self._prefill_profile = PrefillProfile.load_or_generate(
            base_simulator._config,
            step=constraints.interval_request_size,
            slowdown=constraints.prefill_slowdown,
            path=constraints.prefill_profile_path,
            max_tokens=constraints.max_request_tokens,
        )
        if self._constraints.max_request_tokens is None:
            self._constraints.max_request_tokens = self._prefill_profile.max_tokens

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def initial_state(self) -> VidurMCTSState:
        return VidurMCTSState(self._base.fork(), VidurGameStats())

    # ------------------------------------------------------------------ #
    # Action generation helpers
    # ------------------------------------------------------------------ #
    def sample_adversary_actions(
        self, state: VidurMCTSState, max_samples: int
    ) -> List[AdversaryAction]:
        """Generate candidate adversary actions obeying QPS/token bounds."""
        actions: List[AdversaryAction] = []
        max_new = self._available_qps_budget(state)
        if max_new <= 0:
            return [AdversaryAction([])]

        for _ in range(max_samples):
            num_requests = self._rng.randint(0, max_new)
            specs: List[AdversaryRequestSpec] = []
            for _ in range(num_requests):
                total_budget = self._max_request_tokens_allowed()
                decode = self._sample_token_size(
                    min_size=self._constraints.interval_request_size,
                    max_size=total_budget - self._constraints.interval_request_size,
                )
                max_prefill = max(self._constraints.interval_request_size, total_budget - decode)
                prefill = self._sample_token_size(min_size=self._constraints.interval_request_size , max_size=max_prefill)
                remaining = max(0, total_budget - prefill)
                if remaining < self._constraints.interval_request_size:
                    decode = remaining
                else:
                    decode = min(decode, remaining)
                if decode < 0:
                    decode = 0

                slo_opts = self._constraints.request_slo_options
                prefill_slo = self._prefill_profile.lookup(prefill)
                if slo_opts.prefill_slos:
                    prefill_slo *= self._rng.choice(slo_opts.prefill_slos)
                decode_slo = self._rng.choice(slo_opts.decode_slos) / 1000.0
                specs.append(
                    AdversaryRequestSpec(
                        prefill_tokens=prefill,
                        decode_tokens=decode,
                        prefill_slo=prefill_slo,
                        decode_slo=decode_slo,
                    )
                )
            actions.append(AdversaryAction(specs))

        if not actions:
            actions.append(AdversaryAction([]))
        return actions

    def sample_controller_actions(
        self, state: VidurMCTSState, max_samples: int
    ) -> List[ControllerAction]:
        """Generate candidate controller actions by sampling token budgets and request priorities."""
        self._drain_arrivals(state.simulator)
        # Cache lookup and waiting IDs once per call.
        token_budget_options = [b for b in self._enumerate_token_budgets(state) if b > 0]
        request_lookup = self._build_request_lookup(state.simulator)
        waiting_ids = sorted(request_lookup.keys())
        if not token_budget_options or not waiting_ids:
            fallback = self._build_minimal_controller_action(
                request_lookup, waiting_ids, self._constraints.interval_request_size
            )
            if fallback is not None:
                return [fallback]
            return [ControllerAction(token_budget=0, selected_request_ids=None)]

        actions: List[ControllerAction] = []
        for budget in token_budget_options:
            selected_count = min(len(waiting_ids), self._cfg.max_branching)
            if selected_count == 0:
                continue
            if selected_count >= len(waiting_ids):
                selected = list(waiting_ids)
            else:
                selected = self._rng.sample(waiting_ids, selected_count) ## ?? : I Need to modfify this because at the moment this is selecting all requests 
            selected.sort()
            variants = self._generate_allocation_variants(request_lookup, budget, selected)
            if not variants:
                continue
            # Respect overall branching cap to avoid producing an excessive number of actions.
            remaining = max(0, self._cfg.max_branching - len(actions))
            if remaining <= 0:
                break
            if len(variants) > remaining:
                # Randomly sample to keep distribution similar without exceeding cap.
                variants = self._rng.sample(variants, remaining)
            actions.extend(variants)

        actions = [act for act in actions if act.token_budget > 0]
        if actions:
            return actions

        fallback = self._build_minimal_controller_action(
            request_lookup, waiting_ids, self._constraints.interval_request_size
        )
        if fallback is not None:
            return [fallback]
        return [ControllerAction(token_budget=0, selected_request_ids=None)]

    def _max_request_tokens_allowed(self) -> int:
        if self._constraints.max_request_tokens is not None:
            return self._constraints.max_request_tokens
        return self._prefill_profile.max_tokens

    def _generate_allocation_variants(
        self,
        request_lookup: Dict[int, Request],
        token_budget: int,
        selected_ids: List[int],
    ) -> List[ControllerAction]:
        # print("FOR SIMULATION : TOKEN BUDGET = ", token_budget , " SELECTED REQUESTS = ", selected_ids)
        if token_budget <= 0 or not selected_ids:
            return []

        step = self._constraints.interval_request_size
        prefill_caps: Dict[int, int] = {}
        decode_candidates: List[int] = []

        for rid in selected_ids:
            req = request_lookup.get(rid)
            if req is None:
                continue

            remaining_prefill = max(0, req.num_prefill_tokens - req.num_processed_prefill_tokens)
            remaining_decode = max(0, req.num_decode_tokens - req.num_processed_decode_tokens)

            # Avoid repeated getattr lookups per request.
            prefill_done = getattr(req, "_is_prefill_complete", req.is_prefill_complete)
            # print("Request with id : ", rid , " is done with the prefill status : ", prefill_done)
            if remaining_prefill > 0 and not prefill_done:
                prefill_caps[rid] = remaining_prefill
                continue

            if prefill_done and remaining_decode > 0:
                decode_candidates.append(rid)
                continue

            if remaining_prefill > 0:
                prefill_caps[rid] = remaining_prefill

        if not prefill_caps and not decode_candidates:
            return []

        variants: List[ControllerAction] = []
        num_variants = max(1, self._cfg.controller_budget_combs)

        decode_alloc_base: Dict[int, int] = {
            rid: 1 for rid in decode_candidates
        }
        decode_total = sum(decode_alloc_base.values())

        total_prefill_capacity = sum(prefill_caps.values())
        effective_prefill_budget = min(token_budget, total_prefill_capacity)

        for _ in range(num_variants):
            allocations: Dict[int, int] = {}
            prefill_alloc: Dict[int, int] = {}

            remaining_prefill = effective_prefill_budget
            if prefill_caps and remaining_prefill > 0:
                caps = dict(prefill_caps)
                # Allocate full chunks first.
                while (
                    remaining_prefill >= step
                    and any(cap >= step for cap in caps.values())
                ):
                    chunk_candidates = [rid for rid, cap in caps.items() if cap >= step]
                    if not chunk_candidates:
                        break
                    rid = self._rng.choice(chunk_candidates)
                    alloc_amount = min(step, caps[rid], remaining_prefill)
                    if alloc_amount <= 0:
                        break
                    prefill_alloc[rid] = prefill_alloc.get(rid, 0) + alloc_amount
                    caps[rid] -= alloc_amount
                    remaining_prefill -= alloc_amount

                # Allocate any remaining budget (less than a chunk) to a request that can accept it.
                if remaining_prefill > 0:
                    remainder_candidates = [
                        rid for rid, cap in caps.items() if cap > 0
                    ]
                    if remainder_candidates:
                        rid = self._rng.choice(remainder_candidates)
                        alloc_amount = min(remaining_prefill, caps[rid])
                        if alloc_amount > 0:
                            prefill_alloc[rid] = prefill_alloc.get(rid, 0) + alloc_amount
                            caps[rid] -= alloc_amount
                            remaining_prefill -= alloc_amount

            prefill_alloc = {
                rid: tokens for rid, tokens in prefill_alloc.items() if tokens > 0
            }

            total_tokens = sum(prefill_alloc.values()) + decode_total
            if total_tokens <= 0:
                continue

            allocations.update(prefill_alloc)
            allocations.update(decode_alloc_base)
            prioritized_ids = sorted(set(prefill_alloc.keys()) | set(decode_alloc_base.keys()))

            variants.append(
                ControllerAction(
                    token_budget=total_tokens,
                    selected_request_ids=prioritized_ids,
                    token_allocations=allocations,
                    prefill_allocations=prefill_alloc,
                    decode_allocations=dict(decode_alloc_base),
                )
            )

        if not variants:
            fallback = self._build_minimal_controller_action(
                request_lookup, selected_ids, token_budget
            )
            if fallback is not None:
                variants.append(fallback)

        return variants

    def _build_minimal_controller_action(
        self,
        request_lookup: Dict[int, Request],
        candidate_ids: Sequence[int],
        token_budget: int,
    ) -> Optional[ControllerAction]:
        step = self._constraints.interval_request_size
        for rid in candidate_ids:
            req = request_lookup.get(rid)
            if req is None:
                continue

            remaining_prefill = max(
                0, req.num_prefill_tokens - req.num_processed_prefill_tokens
            )
            remaining_decode = max(
                0, req.num_decode_tokens - req.num_processed_decode_tokens
            )

            prefill_done = getattr(req, "_is_prefill_complete", req.is_prefill_complete)

            if remaining_prefill > 0 and not prefill_done:
                alloc = step if remaining_prefill >= step else remaining_prefill
                if alloc > 0:
                    return ControllerAction(
                        token_budget=alloc,
                        selected_request_ids=[rid],
                        token_allocations={rid: alloc},
                        prefill_allocations={rid: alloc},
                        decode_allocations={},
                    )

            if prefill_done and remaining_decode > 0:
                return ControllerAction(
                    token_budget=1,
                    selected_request_ids=[rid],
                    token_allocations={rid: 1},
                    prefill_allocations={},
                    decode_allocations={rid: 1},
                )
        return None

    # ------------------------------------------------------------------ #
    # Transition dynamics
    # ------------------------------------------------------------------ #
    def apply_adversary_action_only(
        self, state: VidurMCTSState, action: AdversaryAction, *, inplace: bool = False
    ) -> VidurMCTSState:
        """Apply adversary action.

        When ``inplace`` is False (default), returns a forked state (safe for tree expansion).
        When ``inplace`` is True, mutates and returns ``state`` (intended for rollout trials).
        """
        target_state = state if inplace else state.fork()
        self._apply_adversary_action(target_state, action)
        self._drain_arrivals(target_state.simulator)
        return target_state

    def apply_controller_action_only(
        self, state: VidurMCTSState, action: ControllerAction, *, inplace: bool = False
    ) -> VidurMCTSState:
        """Apply controller action.

        When ``inplace`` is False (default), returns a forked state (safe for tree expansion).
        When ``inplace`` is True, mutates and returns ``state`` (intended for rollout trials).
        Temporary scheduler budget overrides and hidden-requests are still snapshot/restored
        per call, regardless of ``inplace``.
        """
        new_state = state if inplace else state.fork()
        self._drain_arrivals(new_state.simulator)

        (
            tracker,
            scheduler_budget_snapshot,
            activated_replicas,
            hidden_requests,
        ) = self._configure_controller_action(new_state.simulator, action)
        try:
            if tracker is not None:
                sim_time = new_state.simulator._time
                new_state.simulator._add_event(GlobalScheduleEvent(sim_time))
                for replica_id in activated_replicas:
                    new_state.simulator._add_event(
                        ReplicaScheduleEvent(sim_time, replica_id)
                    )
                self._advance_simulation(new_state, tracker)
            self._update_stats(new_state)
        finally:
            self._restore_scheduler_budget_state(
                new_state.simulator, scheduler_budget_snapshot
            )
            self._restore_hidden_requests(new_state.simulator, hidden_requests)
        return new_state

    def apply_actions(
        self,
        state: VidurMCTSState,
        adversary_action: AdversaryAction,
        controller_action: ControllerAction,
    ) -> VidurMCTSState:
        """Apply both players' actions and advance the simulator."""
        intermediate = self.apply_adversary_action_only(state, adversary_action)
        return self.apply_controller_action_only(intermediate, controller_action)

    # ------------------------------------------------------------------ #
    # Objective evaluation
    # ------------------------------------------------------------------ #
    def evaluate_objective(self, state: VidurMCTSState) -> Tuple[int, float]:
        st = state.stats
        simulator = state.simulator
        request_lookup = self._build_request_lookup(simulator)

        violations = st.slo_violations
        lateness_sum = st.slo_lateness_sum

        for rid, req in request_lookup.items():
            if rid in st.completed_request_ids:
                continue
            lateness = self._compute_lateness(req, simulator._time)
            if lateness > 0:
                violations += 1
                lateness_sum += lateness

        avg_lateness = lateness_sum / max(violations, 1) if violations else 0.0
        return violations, avg_lateness

    def describe_state(self, state: VidurMCTSState) -> Dict[str, Any]:
        violations, avg_lateness = self.evaluate_objective(state)
        simulator = state.simulator
        request_lookup = self._build_request_lookup(simulator)
        waiting_ids = self._collect_waiting_request_ids(simulator)
        return {
            "sim_time": simulator._time,
            "requests_in_system": len(request_lookup),
            "requests_generated": state.stats.requests_generated,
            "requests_completed": state.stats.requests_completed,
            "slo_violations": violations,
            "avg_lateness": avg_lateness,
            "waiting_request_ids": waiting_ids,
            "completed_request_ids": list(state.stats.completed_request_ids),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _apply_adversary_action(
        self, state: VidurMCTSState, action: AdversaryAction
    ) -> None:
        sim = state.simulator
        time_now = sim._time ## ?? : Check how does this time functions in the simulator and whether it matches with our goal or not 
        for spec in action.requests: 
            req = Request(
                arrived_at=time_now,
                num_prefill_tokens=spec.prefill_tokens,
                num_decode_tokens=spec.decode_tokens,
                block_hash_ids=None,
                block_size=None,
            )
            self._slo_manager.set_slos(req)
            req.prefill_slo_time = spec.prefill_slo
            req.decode_slo_time = spec.decode_slo
            setattr(req, "_desired_prefill_slo_time", spec.prefill_slo)
            setattr(req, "_desired_decode_slo_time", spec.decode_slo)
            req.completion_slo_time = -1

            sim._add_event(RequestArrivalEvent(time_now, req))
            state.stats.requests_generated += 1
            state.stats.recent_arrivals.append(time_now)

        # Maintain arrival history within 1 second window for QPS constraint
        window_start = time_now - 1.0
        state.stats.recent_arrivals = [
            t for t in state.stats.recent_arrivals if t >= window_start
        ]


    def _prioritize_requests(self, replica_scheduler, selected_ids: Sequence[int]) -> None:
        waiting_queue = getattr(replica_scheduler, "_waiting_queue", None)
        if waiting_queue is None or not hasattr(waiting_queue, "to_list"):
            return

        current: List[Request] = waiting_queue.to_list()
        waiting_queue._request_queue = []  # type: ignore[attr-defined]
        waiting_queue._num_prefill_tokens = 0  # type: ignore[attr-defined]

        selected_set = set(selected_ids)
        ordered = [req for req in current if req.id in selected_set] + [
            req for req in current if req.id not in selected_set
        ]
        for req in ordered:
            waiting_queue.push(req)

    def _restrict_running_set(
        self,
        replica_scheduler,
        targeted_ids: Iterable[int],
    ) -> List[Request]:
        running = getattr(replica_scheduler, "_running", None)
        if running is None:
            return []

        targeted = set(targeted_ids)
        kept: List[Request] = []
        hidden: List[Request] = []
        for req in running:
            if req.id in targeted:
                kept.append(req)
            else:
                hidden.append(req)

        setattr(replica_scheduler, "_running", kept)

        scheduled_set = getattr(replica_scheduler, "scheduled_req_ids", None)
        if isinstance(scheduled_set, set):
            for req in hidden:
                scheduled_set.discard(req.id)

        return hidden

    def _restrict_waiting_queue(
        self,
        replica_scheduler,
        targeted_ids: Iterable[int],
    ) -> List[Request]:
        waiting_queue = getattr(replica_scheduler, "_waiting_queue", None)
        if waiting_queue is None or not hasattr(waiting_queue, "to_list"):
            return []

        current: List[Request] = waiting_queue.to_list()
        targeted_set = set(targeted_ids)
        hidden: List[Request] = []

        waiting_queue._request_queue = []  # type: ignore[attr-defined]
        waiting_queue._num_prefill_tokens = 0  # type: ignore[attr-defined]

        for req in current:
            if req.id in targeted_set:
                waiting_queue.push(req)
            else:
                hidden.append(req)
        return hidden

    def _scheduler_has_request(self, replica_scheduler, request_id: int) -> bool:
        requests_map = getattr(replica_scheduler, "_requests", {})
        if request_id in requests_map or str(request_id) in requests_map:
            return True
        running = getattr(replica_scheduler, "_running", [])
        for req in running:
            if req.id == request_id:
                return True
        waiting_queue = getattr(replica_scheduler, "_waiting_queue", None)
        if waiting_queue is not None and hasattr(waiting_queue, "to_list"):
            for req in waiting_queue.to_list():
                if req.id == request_id:
                    return True
        return False

    def _configure_controller_action(
        self, simulator: Simulator, action: ControllerAction
    ) -> Tuple[
        Optional[_ControllerBudgetTracker],
        Dict[Any, Dict[str, Any]],
        Set[Any],
        Dict[Any, List[Request]],
    ]:
        scheduler_snapshot = self._snapshot_scheduler_budget_state(simulator)

        request_lookup = self._build_request_lookup(simulator)
        if not request_lookup:
            action.token_allocations.clear()
            action.prefill_allocations.clear()
            action.decode_allocations.clear()
            return None, scheduler_snapshot, set(), {}

        max_budget = self._max_feasible_budget(simulator)
        requested_budget = max(0, int(action.token_budget or 0))
        if requested_budget <= 0:
            action.token_allocations.clear()
            action.prefill_allocations.clear()
            action.decode_allocations.clear()
            return None, scheduler_snapshot, set(), {}

        token_budget = min(requested_budget, max_budget)
        action.token_budget = token_budget

        selected_ids = list(action.selected_request_ids or [])
        if not selected_ids:
            selected_ids = sorted(request_lookup.keys())
        selected_ids = [rid for rid in selected_ids if rid in request_lookup]
        if not selected_ids:
            action.token_allocations.clear()
            action.prefill_allocations.clear()
            action.decode_allocations.clear()
            return None, scheduler_snapshot, set(), {}
        action.selected_request_ids = selected_ids

        allocations = dict(action.token_allocations)
        if not allocations:
            variants = self._generate_allocation_variants(
                request_lookup, token_budget, selected_ids
            )
            if not variants:
                action.token_allocations.clear()
                action.prefill_allocations.clear()
                action.decode_allocations.clear()
                return None, scheduler_snapshot, set(), {}
            chosen = variants[0]
            action.token_budget = chosen.token_budget
            token_budget = chosen.token_budget
            action.token_allocations = allocations = dict(chosen.token_allocations)
            action.prefill_allocations = dict(chosen.prefill_allocations)
            action.decode_allocations = dict(chosen.decode_allocations)

        if not action.prefill_allocations:
            action.prefill_allocations = {}
        if not action.decode_allocations:
            action.decode_allocations = {}

        tracker_allocations: Dict[int, Tuple[int, int]] = {}
        baseline_prefill: Dict[int, int] = {}
        baseline_decode: Dict[int, int] = {}

        for rid in list(allocations.keys()):
            req = request_lookup.get(rid)
            if req is None:
                allocations.pop(rid, None)
                action.prefill_allocations.pop(rid, None)
                action.decode_allocations.pop(rid, None)
                continue

            remaining_prefill = max(
                0, req.num_prefill_tokens - req.num_processed_prefill_tokens
            )
            prefill_tokens = min(
                max(0, action.prefill_allocations.get(rid, 0)), remaining_prefill
            )

            remaining_decode = max(
                0, req.num_decode_tokens - req.num_processed_decode_tokens
            )
            decode_tokens = min(
                max(0, action.decode_allocations.get(rid, 0)), remaining_decode
            )

            if prefill_tokens <= 0 and decode_tokens <= 0:
                allocations.pop(rid, None)
                action.prefill_allocations.pop(rid, None)
                action.decode_allocations.pop(rid, None)
                continue

            tracker_allocations[rid] = (prefill_tokens, decode_tokens)
            baseline_prefill[rid] = req.num_processed_prefill_tokens
            baseline_decode[rid] = req.num_processed_decode_tokens
            if prefill_tokens > 0:
                action.prefill_allocations[rid] = prefill_tokens
            else:
                action.prefill_allocations.pop(rid, None)
            if decode_tokens > 0:
                action.decode_allocations[rid] = decode_tokens
            else:
                action.decode_allocations.pop(rid, None)
            allocations[rid] = prefill_tokens + decode_tokens

        if not tracker_allocations:
            action.token_allocations.clear()
            action.prefill_allocations.clear()
            action.decode_allocations.clear()
            return None, scheduler_snapshot, set(), {}

        action.token_allocations = allocations
        action.token_budget = sum(allocations.values())

        activated_replicas: Set[Any] = set()
        hidden_state: Dict[Any, Dict[str, List[Request]]] = {}
        for replica_scheduler in simulator._scheduler._replica_schedulers.values():
            targeted_ids = [
                rid for rid in tracker_allocations.keys()
                if self._scheduler_has_request(replica_scheduler, rid)
            ]
            if targeted_ids:
                hidden_waiting = self._restrict_waiting_queue(replica_scheduler, targeted_ids)
                hidden_running = self._restrict_running_set(replica_scheduler, targeted_ids)
                replica_id = getattr(replica_scheduler, "replica_id", None) or getattr(
                    replica_scheduler, "_replica_id", None
                )
                if replica_id is not None and (hidden_waiting or hidden_running):
                    hidden_state[replica_id] = {
                        "waiting": hidden_waiting,
                        "running": hidden_running,
                    }
                self._prioritize_requests(replica_scheduler, targeted_ids)

        for replica_scheduler in simulator._scheduler._replica_schedulers.values():
            overrides = {
                rid: allocations[rid]
                for rid in allocations.keys()
                if self._scheduler_has_request(replica_scheduler, rid)
            }
            if hasattr(replica_scheduler, "set_token_budget_overrides"):
                replica_scheduler.set_token_budget_overrides(overrides)

            if overrides:
                total_tokens = sum(overrides.values())
                scheduler_cfg = getattr(replica_scheduler, "_config", None)
                if scheduler_cfg is not None and hasattr(
                    scheduler_cfg, "chunk_size"
                ):
                    scheduler_cfg.chunk_size = max(total_tokens, 1)
                activated_replicas.add(getattr(replica_scheduler, "replica_id", None) or getattr(replica_scheduler, "_replica_id", None))

        tracker = _ControllerBudgetTracker(
            allocations=tracker_allocations,
            baseline_prefill=baseline_prefill,
            baseline_decode=baseline_decode,
        )
        action.selected_request_ids = sorted(tracker_allocations.keys())
        activated_replicas = {replica_id for replica_id in activated_replicas if replica_id is not None}
        return tracker, scheduler_snapshot, activated_replicas, hidden_state




    def _advance_simulation(
        self,
        state: VidurMCTSState,
        tracker: _ControllerBudgetTracker,
    ) -> None:
        sim = state.simulator

        steps = 0
        max_steps = max(1, self._cfg.simulation_depth * 10)
        while sim._event_queue and steps < max_steps:
            next_event = sim._event_queue[0]
            if (
                next_event.event_type == EventType.REQUEST_ARRIVAL
                and next_event._time > sim._time
            ):
                break
            event = heapq.heappop(sim._event_queue)
            sim._set_time(event._time)
            new_events = event.handle_event(sim._scheduler, sim._cluster_metric_store)
            for new_event in new_events:
                sim._add_event(new_event)
            steps += 1
            lookup = self._build_request_lookup(sim)
            if tracker.is_satisfied(lookup):
                self._prune_pending_replica_schedule_events(sim)
                break

    def _update_stats(self, state: VidurMCTSState) -> None:
        sim = state.simulator
        stats = state.stats
        for replica_scheduler in sim._scheduler._replica_schedulers.values():
            for request in list(replica_scheduler._requests.values()):
                if request.completed and request.id not in stats.completed_request_ids:
                    stats.requests_completed += 1
                    lateness = self._compute_lateness(request, sim._time)
                    if lateness > 0:
                        stats.slo_violations += 1
                        stats.slo_lateness_sum += lateness
                    stats.completed_request_ids.add(request.id)

    def _compute_lateness(self, request: Request, sim_time: float) -> float:
        total_lateness = 0.0

        arrived_at = getattr(request, "_arrived_at", request.arrived_at)
        prefill_completed_at = getattr(request, "_prefill_completed_at", None)
        if not getattr(request, "_is_prefill_complete", request.is_prefill_complete):
            prefill_completed_at = None
        elif prefill_completed_at in (None, 0):
            prefill_completed_at = None

        prefill_slo = getattr(request, "_prefill_slo_time", None)
        if prefill_slo is not None:
            deadline = arrived_at + prefill_slo
            actual = prefill_completed_at if prefill_completed_at is not None else sim_time
            total_lateness += max(0.0, actual - deadline)

        decode_slo = getattr(request, "_decode_slo_time", None)
        if decode_slo is not None and decode_slo >= 0:
            has_decode_tokens = getattr(
                request, "_num_decode_tokens", request.num_decode_tokens
            ) > 0
            if has_decode_tokens and prefill_completed_at is not None:
                decode_tokens_done = request.num_processed_decode_tokens
                total_decode_tokens = getattr(
                    request, "_num_decode_tokens", request.num_decode_tokens
                )
                baseline_tokens = 1 if total_decode_tokens > 0 else 0
                actual_tokens = max(decode_tokens_done - baseline_tokens, 0)

                latest_iter_end = getattr(
                    request, "_latest_iteration_completed_at", None
                )
                if latest_iter_end in (None, 0):
                    latest_iter_end = None

                decode_lateness = 0.0

                if actual_tokens > 0:
                    produced_deadline = (
                        prefill_completed_at + actual_tokens * decode_slo
                    )
                    actual_decode = (
                        latest_iter_end if latest_iter_end is not None else sim_time
                    )
                    decode_lateness = max(
                        decode_lateness, max(0.0, actual_decode - produced_deadline)
                    )
                else:
                    first_deadline = prefill_completed_at + decode_slo
                    decode_lateness = max(
                        decode_lateness, max(0.0, sim_time - first_deadline)
                    )

                remaining_tokens = max(total_decode_tokens - actual_tokens, 0)
                if remaining_tokens > 0:
                    next_deadline = prefill_completed_at + (
                        actual_tokens + 1
                    ) * decode_slo
                    decode_lateness = max(
                        decode_lateness, max(0.0, sim_time - next_deadline)
                    )

                total_lateness += decode_lateness

        return total_lateness

    # ------------------------------------------------------------------ #
    # Utility functions
    # ------------------------------------------------------------------ #
    def _build_request_lookup(self, simulator: Simulator) -> Dict[int, Request]:
        lookup: Dict[int, Request] = {}
        for replica_scheduler in simulator._scheduler._replica_schedulers.values():
            waiting = getattr(replica_scheduler, "_waiting_queue", None)
            if waiting and hasattr(waiting, "to_list"):
                for req in waiting.to_list():
                    lookup[req.id] = req
            for req in getattr(replica_scheduler, "_running", []):
                lookup[req.id] = req
            for req in getattr(replica_scheduler, "_requests", {}).values():
                if not req.completed:
                    lookup[req.id] = req
        return lookup

    def _prune_pending_replica_schedule_events(self, simulator: Simulator) -> None:
        if not simulator._event_queue:
            return
        filtered_events = [
            event
            for event in simulator._event_queue
            if not (
                event.event_type == EventType.REPLICA_SCHEDULE
                and event._time == simulator._time
            )
        ]
        if len(filtered_events) != len(simulator._event_queue):
            simulator._event_queue = filtered_events
            heapq.heapify(simulator._event_queue)

    def _snapshot_scheduler_budget_state(
        self, simulator: Simulator
    ) -> Dict[Any, Dict[str, Any]]:
        snapshot: Dict[Any, Dict[str, Any]] = {}
        replica_schedulers = getattr(
            simulator._scheduler, "_replica_schedulers", {}
        )
        for replica_id, scheduler in replica_schedulers.items():
            chunk_size = None
            scheduler_cfg = getattr(scheduler, "_config", None)
            if scheduler_cfg is not None and hasattr(scheduler_cfg, "chunk_size"):
                chunk_size = scheduler_cfg.chunk_size

            overrides_snapshot = None
            if hasattr(scheduler, "_token_budget_overrides"):
                overrides_attr = getattr(scheduler, "_token_budget_overrides")
                if overrides_attr is not None:
                    overrides_snapshot = dict(overrides_attr)
                else:
                    overrides_snapshot = {}

            snapshot[replica_id] = {
                "chunk_size": chunk_size,
                "overrides": overrides_snapshot,
                "can_set_overrides": hasattr(scheduler, "set_token_budget_overrides"),
            }
        return snapshot

    def _restore_scheduler_budget_state(
        self,
        simulator: Simulator,
        snapshot: Dict[Any, Dict[str, Any]],
    ) -> None:
        replica_schedulers = getattr(
            simulator._scheduler, "_replica_schedulers", {}
        )
        for replica_id, saved in snapshot.items():
            scheduler = replica_schedulers.get(replica_id)
            if scheduler is None:
                continue

            chunk_size = saved.get("chunk_size")
            scheduler_cfg = getattr(scheduler, "_config", None)
            if (
                chunk_size is not None
                and scheduler_cfg is not None
                and hasattr(scheduler_cfg, "chunk_size")
            ):
                scheduler_cfg.chunk_size = chunk_size

            if saved.get("can_set_overrides") and hasattr(
                scheduler, "set_token_budget_overrides"
            ):
                overrides_snapshot = saved.get("overrides")
                if overrides_snapshot is not None:
                    scheduler.set_token_budget_overrides(dict(overrides_snapshot))
                else:
                    scheduler.set_token_budget_overrides({})

    def _restore_hidden_requests(
        self,
        simulator: Simulator,
        hidden_state: Dict[Any, Dict[str, List[Request]]],
    ) -> None:
        if not hidden_state:
            return
        for replica_id, groups in hidden_state.items():
            scheduler = simulator._scheduler._replica_schedulers.get(replica_id)
            if scheduler is None:
                continue

            waiting_queue = getattr(scheduler, "_waiting_queue", None)
            hidden_waiting = groups.get("waiting", [])
            if waiting_queue is not None and hasattr(waiting_queue, "push"):
                for req in hidden_waiting:
                    waiting_queue.push(req)

            hidden_running = groups.get("running", [])
            if hidden_running:
                running = getattr(scheduler, "_running", None)
                if running is None:
                    setattr(scheduler, "_running", list(hidden_running))
                else:
                    running.extend(hidden_running)

    def _drain_arrivals(self, simulator: Simulator) -> None:
        while simulator._event_queue:
            next_event = simulator._event_queue[0]
            if next_event.event_type not in (
                EventType.REQUEST_ARRIVAL,
                EventType.GLOBAL_SCHEDULE,
            ):
                break
            event = heapq.heappop(simulator._event_queue)
            simulator._set_time(event._time)
            new_events = event.handle_event(
                simulator._scheduler, simulator._cluster_metric_store
            )
            if event.event_type == EventType.REQUEST_ARRIVAL:
                request = getattr(event, "_request", None)
                if request is not None:
                    desired_prefill = getattr(
                        request, "_desired_prefill_slo_time", None
                    )
                    desired_decode = getattr(
                        request, "_desired_decode_slo_time", None
                    )
                    if desired_prefill is not None:
                        request.prefill_slo_time = desired_prefill
                        delattr(request, "_desired_prefill_slo_time")
                    if desired_decode is not None:
                        request.decode_slo_time = desired_decode
                        delattr(request, "_desired_decode_slo_time")
            for new_event in new_events:
                simulator._add_event(new_event)

    def _available_qps_budget(self, state: VidurMCTSState) -> int:
        window_start = state.simulator._time - 1.0
        arrivals_in_window = len([t for t in state.stats.recent_arrivals if t >= window_start])
        return max(0, self._constraints.maximum_qps - arrivals_in_window)

    def _collect_waiting_request_ids(self, simulator: Simulator) -> List[int]:
        request_ids: List[int] = []
        for replica_scheduler in simulator._scheduler._replica_schedulers.values():
            waiting = getattr(replica_scheduler, "_waiting_queue", None)
            if waiting and hasattr(waiting, "to_list"):
                request_ids.extend(req.id for req in waiting.to_list())
            running = getattr(replica_scheduler, "_running", [])
            request_ids.extend(req.id for req in running)
        return sorted(set(request_ids))

    def _sample_token_size(self, min_size: Optional[int] = None, max_size: Optional[int] = None) -> int:
        step = self._constraints.interval_request_size
        lo = max(step, min_size or self._constraints.min_request_tokens)
        hi = min(self._constraints.max_request_tokens, max_size or self._constraints.max_request_tokens)
        lo = max(step, step * ((lo + step - 1) // step))
        hi = max(lo, step * (hi // step))
        if hi < lo:
            hi = lo
        return self._rng.randrange(lo, hi + 1, step)

    def _enumerate_token_budgets(self, state: VidurMCTSState) -> List[int]:
        max_budget = self._max_feasible_budget(state.simulator)
        step = self._constraints.interval_request_size
        budgets = list(range(step, max_budget + 1, step))
        if len(budgets) > self._cfg.max_branching:
            budgets = self._rng.sample(budgets, self._cfg.max_branching)
        return sorted(budgets)


    ## ?? What about the decode lengths of both waiting requests ? every included/scheduled request in the budget has the token budget of 1 
    def _max_feasible_budget(self, simulator: Simulator) -> int:
        total_tokens = 0
        for replica_scheduler in simulator._scheduler._replica_schedulers.values():
            waiting = getattr(replica_scheduler, "_waiting_queue", None)
            if waiting and hasattr(waiting, "get_num_prefill_tokens"):
                total_tokens += waiting.get_num_prefill_tokens()

            running = getattr(replica_scheduler, "_running", [])
            for req in running:
                if req.is_prefill_complete or getattr(req, "has_started_decode", False):
                    total_tokens += 1
                else:
                    remaining_prefill = max(0, req.num_prefill_tokens - req.num_processed_prefill_tokens)
                    total_tokens += remaining_prefill

        if total_tokens == 0:
            return 0

        cache_config = simulator._config.cluster_config.cache_config
        cache_tokens = cache_config.block_size * (cache_config.num_blocks or 1)
        return min(total_tokens, cache_tokens)
