from __future__ import annotations

import random
import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

from vidur.entities import Request
from vidur.events.request_arrival_event import RequestArrivalEvent
from vidur.simulator import Simulator
from vidur.utils.slo_manager import SLOManager
from .config import MCTSConstraintConfig, MCTSExploreConfig


## These are the dimensions along which adversary will take an action. Some combination of these within constraints
@dataclass
class AdversaryRequestSpec:
    prefill_tokens: int
    decode_tokens: int
    prefill_slo: float
    decode_slo: float
    completion_slo: float


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
                total_budget = self._constraints.max_request_tokens
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
                specs.append(
                    AdversaryRequestSpec(
                        prefill_tokens=prefill,
                        decode_tokens=decode,
                        prefill_slo=self._rng.choice(slo_opts.prefill_slos),
                        decode_slo=self._rng.choice(slo_opts.decode_slos),
                        completion_slo=self._rng.choice(slo_opts.completion_slos),
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
        token_budget_options = self._enumerate_token_budgets(state)
        request_lookup = self._build_request_lookup(state.simulator)
        waiting_ids = sorted(request_lookup.keys())

        if not token_budget_options or not waiting_ids:
            return [ControllerAction(token_budget=0, selected_request_ids=None)]

        actions: List[ControllerAction] = []
        for budget in token_budget_options:
            selected_count = min(len(waiting_ids), self._cfg.max_branching)
            selected = sorted(self._rng.sample(waiting_ids, selected_count))
            actions.extend(self._generate_allocation_variants(request_lookup, budget, selected))

        return actions or [ControllerAction(token_budget=0, selected_request_ids=None)]

    def _generate_allocation_variants(
        self,
        request_lookup: Dict[int, Request],
        token_budget: int,
        selected_ids: List[int],
    ) -> List[ControllerAction]:
        if token_budget <= 0 or not selected_ids:
            return [ControllerAction(token_budget=0, selected_request_ids=list(selected_ids), token_allocations={})]

        step = self._constraints.interval_request_size
        decode_candidates: List[int] = []
        prefill_candidates: List[Tuple[int, int]] = []

        for rid in selected_ids:
            req = request_lookup.get(rid)
            if not req:
                continue
            if req.is_prefill_complete or getattr(req, "has_started_decode", False): 
                decode_candidates.append(rid)
            else:
                remaining_prefill = max(0, req.num_prefill_tokens - req.num_processed_prefill_tokens)
                remaining_prefill -= remaining_prefill % step ## ?? What if a request has a prefill size not equal to the requestb interval size ? it might not be every scheduled ? For now I guess we can ignore this as adversary will pick multiple of sample_interval_size. (To be continued..)
                if remaining_prefill > 0:
                    prefill_candidates.append((rid, remaining_prefill))

        self._rng.shuffle(decode_candidates)
        decode_selected = decode_candidates[: min(len(decode_candidates), token_budget)]
        base_alloc = {rid: 1 for rid in decode_selected}
        remaining_tokens = token_budget - len(base_alloc)

        if remaining_tokens <= 0 or not prefill_candidates:
            return [
                ControllerAction(
                    token_budget=token_budget,
                    selected_request_ids=list(selected_ids),
                    token_allocations=base_alloc,
                )
            ]

        variants: List[ControllerAction] = []
        for _ in range(self._cfg.controller_budget_combs):
            allocations = dict(base_alloc)
            caps = {rid: cap for rid, cap in prefill_candidates}
            rem = remaining_tokens
            while rem >= step and caps:
                candidates = [rid for rid, cap in caps.items() if allocations.get(rid, 0) + step <= cap]
                if not candidates:
                    break
                rid = self._rng.choice(candidates)
                allocations[rid] = allocations.get(rid, 0) + step
                rem -= step
                if allocations[rid] >= caps[rid]:
                    caps.pop(rid)

            variants.append(
                ControllerAction(
                    token_budget=token_budget,
                    selected_request_ids=list(selected_ids),
                    token_allocations=allocations,
                )
            )

        return variants

    # ------------------------------------------------------------------ #
    # Transition dynamics
    # ------------------------------------------------------------------ #
    def apply_actions(
        self,
        state: VidurMCTSState,
        adversary_action: AdversaryAction,
        controller_action: ControllerAction,
    ) -> VidurMCTSState:
        """Apply both players' actions and advance the simulator."""
        new_state = state.fork()

        self._apply_adversary_action(new_state, adversary_action)
        self._apply_controller_action(new_state, controller_action)
        self._advance_simulation(new_state)
        self._update_stats(new_state)
        return new_state

    # ------------------------------------------------------------------ #
    # Objective evaluation
    # ------------------------------------------------------------------ #
    def evaluate_objective(self, state: VidurMCTSState) -> Tuple[int, float]:
        st = state.stats
        if st.requests_completed == 0:
            return st.slo_violations, 0.0
        avg_lateness = st.slo_lateness_sum / max(st.requests_completed, 1)
        return st.slo_violations, avg_lateness

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
            req.decode_slo_time = spec.decode_slo
            req.completion_slo_time = spec.completion_slo
            # Prefill SLO: reuse SLO manager to compute baseline TTFT 
            ## ?? : This may not be accurate (to be continued later ignore this for now)...
            self._slo_manager.set_slos(req) 
            if spec.prefill_slo >= 0:
                req.prefill_slo_time = spec.prefill_slo

            sim._add_event(RequestArrivalEvent(time_now, req))
            state.stats.requests_generated += 1
            state.stats.recent_arrivals.append(time_now)

        # Maintain arrival history within 1 second window for QPS constraint
        window_start = time_now - 1.0
        state.stats.recent_arrivals = [
            t for t in state.stats.recent_arrivals if t >= window_start
        ]


    def _apply_controller_action(
        self, state: VidurMCTSState, action: ControllerAction
    ) -> None:
        sim = state.simulator
        max_budget = self._max_feasible_budget(sim)
        if action.token_budget:
            token_budget = max(
                self._constraints.interval_request_size,
                min(action.token_budget, max_budget),
            )
        else:
            token_budget = 0

        # Gather all requests by id across schedulers
        request_lookup = self._build_request_lookup(sim)

        if token_budget == 0:
            action.token_allocations = {}
            for replica_scheduler in sim._scheduler._replica_schedulers.values():
                if hasattr(replica_scheduler, "set_token_budget_overrides"):
                    replica_scheduler.set_token_budget_overrides({})
            return

        selected_ids = action.selected_request_ids or []
        if not selected_ids:
            action.token_allocations = {}
            for replica_scheduler in sim._scheduler._replica_schedulers.values():
                if hasattr(replica_scheduler, "set_token_budget_overrides"):
                    replica_scheduler.set_token_budget_overrides({})
                if token_budget:
                    replica_scheduler._config.chunk_size = token_budget
            return
        allocations = dict(action.token_allocations)
        if not allocations:
            variants = self._generate_allocation_variants(request_lookup, token_budget, selected_ids)
            allocations = variants[0].token_allocations if variants else {}
            action.token_allocations = allocations

        for replica_scheduler in sim._scheduler._replica_schedulers.values():
            waiting = getattr(replica_scheduler, "_waiting_queue", None)
            if waiting and hasattr(waiting, "to_list"):
                existing_ids = {req.id for req in waiting.to_list()}
            else:
                existing_ids = set()

            for req in list(getattr(replica_scheduler, "_running", [])):
                if waiting and hasattr(waiting, "push") and req.id not in existing_ids:
                    waiting.push(req)
                    existing_ids.add(req.id)
            replica_scheduler._running = []
            replica_scheduler.scheduled_req_ids.clear()

            scheduler_alloc = {
                rid: tokens
                for rid, tokens in allocations.items()
                if rid in replica_scheduler._requests or str(rid) in replica_scheduler._requests
            }
            if hasattr(replica_scheduler, "set_token_budget_overrides"):
                replica_scheduler.set_token_budget_overrides(scheduler_alloc)
            if token_budget:
                replica_scheduler._config.chunk_size = token_budget
            if action.selected_request_ids:
                self._prioritize_requests(replica_scheduler, action.selected_request_ids)





    ## We might have 
    def _prioritize_requests(self, replica_scheduler, selected_ids: Sequence[int]) -> None:
        ## We need to make the running requests to 0 so that scheduler only picks from the waiting requests only


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





    def _advance_simulation(self, state: VidurMCTSState) -> None:
        sim = state.simulator

        steps = self._cfg.simulation_depth
        for _ in range(steps):
            if not (
                sim._event_queue
                or sim._request_generator.get_next_request_arrival_time() is not None
            ):
                break

            next_event_time = sim._event_queue[0]._time if sim._event_queue else None
            next_arrival_time = sim._request_generator.get_next_request_arrival_time()
            ## ?? : Why are we depending on the request generator's output ? why not adversary ? where is the link between the adversary and the request generator (to be continued later... ignore for now)

            if (next_arrival_time is not None) and (
                next_event_time is None or next_arrival_time <= next_event_time
            ):
                sim._add_event(
                    RequestArrivalEvent(
                        next_arrival_time,
                        sim._request_generator.get_next_request(),
                    )
                )
                continue

            event = heapq.heappop(sim._event_queue)
            sim._set_time(event._time)
            new_events = event.handle_event(sim._scheduler, sim._cluster_metric_store)
            for new_event in new_events:
                sim._add_event(new_event)

    def _update_stats(self, state: VidurMCTSState) -> None:
        sim = state.simulator
        stats = state.stats
        for replica_scheduler in sim._scheduler._replica_schedulers.values():
            for request in list(replica_scheduler._requests.values()):
                if request.completed and request.id not in stats.completed_request_ids:
                    stats.requests_completed += 1
                    lateness = self._compute_lateness(request)
                    if lateness > 0:
                        stats.slo_violations += 1
                        stats.slo_lateness_sum += lateness
                    stats.completed_request_ids.add(request.id)

    def _compute_lateness(self, request: Request) -> float:
        lateness = 0.0

        if getattr(request, "prefill_slo_time", None):
            deadline = request.arrived_at + request.prefill_slo_time
            actual = request.prefill_completed_at or request.scheduled_at
            if actual is not None:
                lateness = max(lateness, max(0.0, actual - deadline))


        ## ?? : Decode deadline is slightly different concept : it involves measuring the lateness of each individual token of the request if that request is in the DECODE PHASE. When the request is in the decode phase , for the first token the deadline is prefill completed + decode slo deadline . For the rest of the tokens the deadline is last decode completed time + decode slo time . Using this , we need to compute the lateness and the SLO violations. So modify the function accordingly
        if request.decode_slo_time >= 0:
            base = max(request.prefill_completed_at, request.scheduled_at, request.arrived_at)
            decode_deadline = base + request.decode_slo_time
            completion = request.completed_at or request.scheduled_at or base
            lateness = max(lateness, max(0.0, completion - decode_deadline))

        if request.completion_slo_time >= 0 and request.arrived_at is not None:
            completion_deadline = request.arrived_at + request.completion_slo_time
            completion = request.completed_at or request.scheduled_at
            lateness = max(lateness, max(0.0, completion - completion_deadline))

        return lateness

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
        return lookup
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
