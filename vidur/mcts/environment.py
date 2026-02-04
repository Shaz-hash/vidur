
## بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ 

from __future__ import annotations

import random
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set
from itertools import combinations_with_replacement ## Used in the _populate_Prefill_Reqs_Table only 
import math

from vidur.entities import Request
from vidur.events.global_schedule_event import GlobalScheduleEvent
from vidur.events.replica_schedule_event import ReplicaScheduleEvent
from vidur.events.request_arrival_event import RequestArrivalEvent
from vidur.simulator import Simulator
from vidur.utils.slo_manager import SLOManager
from vidur.types import EventType
from .launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig
from .prefill_calibrator import PrefillProfile

# TODO : Remove this import after debugging
import time ## For Debugging



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
    stop_decode_ids: List[int] = field(default_factory=list)


## These are the dimensions along which the controller will take an action.
@dataclass
class ControllerAction:
    token_budget: int
    selected_request_ids: Optional[List[int]] = None
    token_allocations: Dict[int, int] = field(default_factory=dict)
    prefill_allocations: Dict[int, int] = field(default_factory=dict)
    decode_allocations: Dict[int, int] = field(default_factory=dict)
    heuristic: Optional[str] = None          # NEW
    strategy: Optional[str] = None           # NEW
    mapping: Optional[Tuple[int, ...]] = None   # NEW: prefill mapping vector



@dataclass
class _ControllerBudgetTracker:
    allocations: Dict[int, Tuple[int, int]]
    baseline_prefill: Dict[int, int]
    baseline_decode: Dict[int, int]
    tracked_requests: Dict[int, Request]  # NEW

    def is_satisfied(self) -> bool:
        for rid, (prefill_budget, decode_budget) in self.allocations.items():
            req = self.tracked_requests.get(rid)
            if req is None:
                continue
            gained_prefill = req.num_processed_prefill_tokens - self.baseline_prefill.get(rid, 0)
            if gained_prefill < prefill_budget:
                return False
            gained_decode = req.num_processed_decode_tokens - self.baseline_decode.get(rid, 0)
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

    # Per-request lateness tracking
    per_request_prefill_lateness: Dict[int, float] = field(default_factory=dict)  # monotone (max)
    per_request_decode_lateness: Dict[int, float] = field(default_factory=dict)   # cumulative sum

    # Decode tracking state
    decode_tokens_counted: Dict[int, int] = field(default_factory=dict)
    decode_next_deadline_by_id: Dict[int, float] = field(default_factory=dict)

    # Prefill bookkeeping: once prefill is complete, we can stop recomputing prefill lateness
    prefill_lateness_finalized: Set[int] = field(default_factory=set)

    violated_request_ids: Set[int] = field(default_factory=set)

    # Used for the Adversary to track arrivals within the  prefill reqs containing batches
    last_prefill_batch_time: Optional[float] = None

    def clone(self) -> "VidurGameStats":
        return VidurGameStats(
            requests_generated=self.requests_generated,
            requests_completed=self.requests_completed,
            slo_violations=self.slo_violations,
            slo_lateness_sum=self.slo_lateness_sum,
            recent_arrivals=list(self.recent_arrivals),
            completed_request_ids=set(self.completed_request_ids),
            per_request_prefill_lateness=dict(self.per_request_prefill_lateness),
            per_request_decode_lateness=dict(self.per_request_decode_lateness),
            decode_tokens_counted=dict(self.decode_tokens_counted),
            decode_next_deadline_by_id=dict(self.decode_next_deadline_by_id),
            prefill_lateness_finalized=set(self.prefill_lateness_finalized),
            violated_request_ids=set(self.violated_request_ids),
            last_prefill_batch_time=self.last_prefill_batch_time,
        )

## Essentially checkpoints the state so it can return back to it to run a different simulation
@dataclass
class VidurMCTSState:
    simulator: Simulator
    stats: VidurGameStats
    # TODO : Remove this flag after testing 
    def fork(self , flag: Optional[bool] = None) -> "VidurMCTSState":
        return VidurMCTSState(self.simulator.fork(flag=flag), self.stats.clone())

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
        self.all_possible_Prefill_reqs_table: Dict[int, List[int]] = None
        self.controller_all_possible_prefill_budgets: List[int] = None
        self.all_possible_Controller_actions: List[any] = None
        self.all_possible_Controller_States: Dict[Tuple[Tuple[int, ...], str, str], Dict[str, Any]] = {}
        self._base_snapshot = base_simulator.snapshot_state()
        self._history_root_snapshot = None

        # TODO : later this variable will encode max length + prefill + decode lengths
        if self._constraints.max_request_tokens is None:
            self._constraints.max_request_tokens = self._prefill_profile.max_tokens


    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def initial_state(self) -> VidurMCTSState:
        # Always recreate a fresh simulator from the frozen root snapshot.
        # This ensures request IDs and entity counters are identical for every
        # replay from the root, regardless of what other snapshots did.
        sim = Simulator(
            self._base._config,
            register_atexit=False,
            execution_time_predictor=getattr(self._base, "_execution_time_predictor", None),
        )
        from vidur.metrics.noop_metrics_store import NoOpClusterMetricsStore
        sim._cluster_metric_store = NoOpClusterMetricsStore()

        sim.restore_state(self._base_snapshot)
        return VidurMCTSState(sim, VidurGameStats())


    def snapshot_history_root(self, state: VidurMCTSState) -> None:
        """Capture a frozen simulator snapshot for the MCTS history root."""
        self._history_root_snapshot = state.simulator.snapshot_state()

    def clone_history_root_state(self, stats_template: VidurGameStats) -> VidurMCTSState:
        """Return a fresh state cloned from the history-root snapshot.

        Falls back to the base initial_state() if no history snapshot exists.
        """
        if self._history_root_snapshot is None:
            # No history prefix used; just start from the base snapshot.
            return self.initial_state()

        sim = Simulator(
            self._base._config,
            register_atexit=False,
            execution_time_predictor=getattr(
                self._base, "_execution_time_predictor", None
            ),
        )
        sim.restore_state(self._history_root_snapshot)
        return VidurMCTSState(sim, stats_template.clone())

    def clone_state_from_snapshot(self, snapshot: Any, stats_template: VidurGameStats) -> VidurMCTSState:
        sim = Simulator(
            self._base._config,
            register_atexit=False,
            execution_time_predictor=getattr(self._base, "_execution_time_predictor", None),
        )
        from vidur.metrics.noop_metrics_store import NoOpClusterMetricsStore
        sim._cluster_metric_store = NoOpClusterMetricsStore()

        sim.restore_state(snapshot)
        return VidurMCTSState(sim, stats_template.clone())



    # ------------------------------------------------------------------ #
    # Action sampling for Players
    # ------------------------------------------------------------------ #

    def sample_adversary_actions(
        self, state: VidurMCTSState, max_samples: int
    ) -> Tuple[List[Optional[AdversaryAction]], List[bool]]:
        """
        6 deterministic adversary actions.

        Index i (0..5) => send (i+1) requests, each with:
        - prefill_tokens = 3072 (or constraints.max_request_tokens)
        - decode_tokens  = 1000 (fixed)

        Gating:
        - Only allow sending a batch if >= 1.0s has passed since the last prefill batch time
            (tracked by state.stats.last_prefill_batch_time, which _apply_adversary_action updates).
        - If not allowed yet: return a single no-op at index 0 (mask[0]=True) to avoid branching.
        """
        self._drain_arrivals(state.simulator)

        NUM_ACTIONS = 6
        actions_by_index: List[Optional[AdversaryAction]] = [None] * NUM_ACTIONS
        mask: List[bool] = [False] * NUM_ACTIONS

        sim_time = float(state.simulator._time)

        last = getattr(state.stats, "last_prefill_batch_time", None)
        can_send = (last is None) or (sim_time >= float(last) + 1.0 - 1e-9)

        # not time yet -> forced no-op (single valid action)
        if not can_send:
            actions_by_index[0] = AdversaryAction(requests=[], stop_decode_ids=[])
            mask[0] = True
            return actions_by_index, mask

        # build real batch actions
        prefill_size = int(getattr(self._constraints, "max_request_tokens", 3072) or 3072)
        DECODE_TOKENS_FIXED = 5000

        slo_opts = self._constraints.request_slo_options

        base_prefill = float(self._prefill_profile.lookup(prefill_size))
        prefill_slo_time = base_prefill
    
        if getattr(slo_opts, "decode_slos", None):
            decode_slo_time = float(slo_opts.decode_slos[0]) / 1000.0
        else:
            decode_slo_time = 0.0

        for i in range(NUM_ACTIONS):
            count = i + 1
            specs: List[AdversaryRequestSpec] = []
            for _ in range(count):
                specs.append(
                    AdversaryRequestSpec(
                        prefill_tokens=prefill_size,
                        decode_tokens=DECODE_TOKENS_FIXED,
                        prefill_slo=float(prefill_slo_time),
                        decode_slo=float(decode_slo_time),
                    )
                )

            actions_by_index[i] = AdversaryAction(requests=specs, stop_decode_ids=[])
            mask[i] = True

        return actions_by_index, mask





    def sample_controller_actions(
        self,
        state: VidurMCTSState,
        max_samples: int,
        use_state_cache: bool = True,
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        """
        Deterministic controller action indexing (24 actions total):

        Budgets (6):  step * [1..6]   (fallback step=512 => [512..3072])
        Heuristics (4): ["SJF", "EDF", "LST", "LJF"]

        Indexing:
        index = budget_idx * 4 + heur_idx

        budget_idx: 0..5 corresponds to budgets: [step,2step,3step,4step,5step,6step]
        heur_idx:   0..3 corresponds to heuristics order above

        Each action deterministically:
        - sorts current *prefill* requests using the chosen heuristic
        - allocates prefill tokens greedily in that order up to the chosen budget
        - allocates decode tokens as 1 for every decode-eligible request (same as before)
        - fills ControllerAction.{token_allocations,prefill_allocations,decode_allocations}

        Masking rule:
        Let S = sum of remaining prefill tokens across all prefill requests.
        - If S > 0: budgets > S are masked False (so their 4 heuristic actions are invalid).
        - If S == 0: allow only the first budget group (budget_idx==0) as valid (so you still have valid actions for decode-only / no-op).

        Returns:
        actions_by_index: length 24, entries are ControllerAction or None
        mask: length 24, bool validity for each index
        """

        self._drain_arrivals(state.simulator)
        sim_time = state.simulator._time
        step = int(self._constraints.interval_request_size or 512)

        # Fixed 6 budgets (fallback: 512..3072 if step==512)
        budgets: List[int] = [step * i for i in range(1, 7)]

        # Build lookup once
        request_lookup = self._build_request_lookup(state.simulator)
        waiting_ids_all = sorted(request_lookup.keys())

        # Always return fixed-size outputs
        NUM_HEUR = 4
        NUM_BUDGETS = 6
        NUM_ACTIONS = NUM_HEUR * NUM_BUDGETS  # 24
        actions_by_index: List[Optional[ControllerAction]] = [None] * NUM_ACTIONS
        mask: List[bool] = [False] * NUM_ACTIONS

        # If nothing exists, return a single no-op at index 0 (valid)
        if not waiting_ids_all:
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True
            return actions_by_index, mask

        # Helpers
        def remaining_prefill(req: Request) -> int:
            return max(0, req.num_prefill_tokens - req.num_processed_prefill_tokens)

        def prefill_done(req: Request) -> bool:
            return getattr(req, "_is_prefill_complete", req.is_prefill_complete)

        # Prefill candidates only
        prefill_ids: List[int] = []
        for rid in waiting_ids_all:
            req = request_lookup.get(rid)
            if req is None:
                continue
            if remaining_prefill(req) > 0 and not prefill_done(req):
                prefill_ids.append(rid)

        # Decode candidates: always included in every action (1 decode token each)
        decode_candidates: List[int] = []
        for rid in waiting_ids_all:
            req = request_lookup.get(rid)
            if req is None:
                continue
            remaining_decode = max(0, req.num_decode_tokens - req.num_processed_decode_tokens)
            if prefill_done(req) and remaining_decode > 0:
                decode_candidates.append(rid)

        decode_candidates = sorted(decode_candidates)

        # Compute total remaining prefill tokens S for masking budgets
        total_remaining_prefill = 0
        for rid in prefill_ids:
            total_remaining_prefill += remaining_prefill(request_lookup[rid])


        # If there is no remaining prefill in the whole system, all 24 actions are equivalent
        # (they will only allocate decode tokens or do nothing). To avoid pointless branching,
        # force a single valid action (index 0).
        if total_remaining_prefill == 0:
            decode_base = {rid: 1 for rid in decode_candidates}
            token_alloc = dict(decode_base)
            selected = sorted(token_alloc.keys())
            a = ControllerAction(
                token_budget=len(decode_base),
                selected_request_ids=selected if selected else None,
                token_allocations=token_alloc,
                prefill_allocations={},
                decode_allocations=decode_base,
                heuristic="SJF",          # consistent with index 0 template
                strategy="Fixed",
            )
            actions_by_index = [None] * 24
            mask = [False] * 24
            actions_by_index[0] = a
            mask[0] = True
            return actions_by_index, mask

        # Heuristic order functions (deterministic)
        def order_sjf(ids: List[int]) -> List[int]:
            return sorted(ids, key=lambda rid: remaining_prefill(request_lookup[rid]))

        def order_edf(ids: List[int]) -> List[int]:
            return sorted(
                ids,
                key=lambda rid: (
                    getattr(request_lookup[rid], "arrived_at", 0.0)
                    + getattr(request_lookup[rid], "prefill_slo_time", 0.0)
                ),
            )

        def order_lst(ids: List[int]) -> List[int]:
            def slack(rid: int) -> float:
                req = request_lookup[rid]
                remaining_slo = (
                    getattr(req, "prefill_slo_time", 0.0)
                    - max(0.0, sim_time - getattr(req, "arrived_at", 0.0))
                )
                est = self._prefill_profile.lookup(remaining_prefill(req))
                return remaining_slo - est

            return sorted(ids, key=slack)

        def order_slowdown(ids: List[int]) -> List[int]:
            def ratio(rid: int) -> float:
                req = request_lookup[rid]
                waited = max(0.0, sim_time - getattr(req, "arrived_at", 0.0))
                est_full = self._prefill_profile.lookup(getattr(req, "num_prefill_tokens", 0)) or 1e-9
                return waited / est_full

            return sorted(ids, key=ratio, reverse=True)

        def order_ljf(ids: List[int]) -> List[int]:
            return sorted(ids, key=lambda rid: remaining_prefill(request_lookup[rid]), reverse=True)


        heuristics = [
            ("SJF", order_sjf),
            ("EDF", order_edf),
            ("LST", order_lst),
            ("LJF", order_ljf),
        ]

        def build_action(ordered_prefill: List[int], prefill_budget: int, heur_name: str) -> ControllerAction:
            # Greedy deterministic allocation across ordered prefill requests
            remaining_budget = max(0, int(prefill_budget))
            pre: Dict[int, int] = {}

            for rid in ordered_prefill:
                if remaining_budget <= 0:
                    break
                cap = remaining_prefill(request_lookup[rid])
                if cap <= 0:
                    continue
                alloc = min(cap, remaining_budget)
                if alloc > 0:
                    pre[rid] = alloc
                    remaining_budget -= alloc

            # Decode: 1 token each
            dec = {rid: 1 for rid in decode_candidates}

            token_alloc = {**pre, **dec}
            selected = sorted(set(token_alloc.keys()))
            total_budget = sum(token_alloc.values())

            a = ControllerAction(
                token_budget=total_budget,
                selected_request_ids=selected if selected else None,
                token_allocations=token_alloc,
                prefill_allocations=pre,
                decode_allocations=dec,
                heuristic=heur_name,
                strategy="Fixed",  # optional label
            )
            return a

        # Build the fixed indexed action list + mask
        for b_idx, b in enumerate(budgets):
            # Budget masking based on total remaining prefill
            if total_remaining_prefill > 0:
                budget_valid = b <= total_remaining_prefill
            else:
                # No prefill left: allow only the first budget group so we still have valid actions
                budget_valid = (b_idx == 0)

            for h_idx, (h_name, order_fn) in enumerate(heuristics):
                idx = b_idx * NUM_HEUR + h_idx

                if not budget_valid:
                    mask[idx] = False
                    actions_by_index[idx] = None
                    continue

                mask[idx] = True
                ordered_prefill = order_fn(prefill_ids) if prefill_ids else []
                actions_by_index[idx] = build_action(ordered_prefill, b, h_name)

        # Safety: ensure at least one valid action exists
        if not any(mask):
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True

        return actions_by_index, mask



    def _max_request_tokens_allowed(self) -> int:
        if self._constraints.max_request_tokens is not None:
            return self._constraints.max_request_tokens
        return self._prefill_profile.max_tokens


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


    #TODO: Remove the comments used for profiling after testing
    def apply_controller_action_only(
        self, state: VidurMCTSState, action: ControllerAction, *, inplace: bool = False
    ) -> VidurMCTSState:
        """Apply controller action.

        When ``inplace`` is False (default), returns a forked state (safe for tree expansion).
        When ``inplace`` is True, mutates and returns ``state`` (intended for rollout trials).
        Temporary scheduler budget overrides and hidden-requests are still snapshot/restored
        per call, regardless of ``inplace``.
        """
        # t0 = time.perf_counter()
        new_state = state if inplace else state.fork()
        from vidur.metrics.noop_metrics_store import NoOpClusterMetricsStore
        new_state.simulator._cluster_metric_store = NoOpClusterMetricsStore()
        # t1 = time.perf_counter()
        # print(f"[PROFILE] FORK phase={t1 - t0:.6f}s")
        
        # t0 = time.perf_counter()
        self._drain_arrivals(new_state.simulator)
        # t1 = time.perf_counter()
        # print(f"[PROFILE] DRAIN ARRIVALS phase={t1 - t0:.6f}s")
        
        # t0 = time.perf_counter()
        (
            tracker,
            scheduler_budget_snapshot,
            activated_replicas,
            hidden_requests,
        ) = self._configure_controller_action(new_state.simulator, action)
        # t1 = time.perf_counter()
        # print(f"[PROFILE] CONTROLLER ACTION SETUP phase={t1 - t0:.6f}s")
        try:
            if tracker is not None:
                # t0 = time.perf_counter()
                # sim_time = new_state.simulator._time
                # new_state.simulator._add_event(GlobalScheduleEvent(sim_time))
                # t1 = time.perf_counter()
                # print(f"[PROFILE] CONTROLLER ACTION SCHEDULE PHASE={t1 - t0:.6f}s")
                # t0 = time.perf_counter()
                # for replica_id in activated_replicas:
                #     new_state.simulator._add_event(
                #         ReplicaScheduleEvent(sim_time, replica_id)
                #     )
                # t1 = time.perf_counter()
                # print(f"[PROFILE] CONTROLLER ACTION REPLICA SCHEDULE PHASE={t1 - t0:.6f}s")
                # t0 = time.perf_counter()
                # self._advance_simulation(new_state, tracker)

                self._advance_simulation_fast(new_state, tracker, activated_replicas)
                # Only run this when controller did NO prefill work this step (decode-only trivial period)
                if not action.prefill_allocations:
                    self._maybe_fast_forward_decode_only_to_next_adv_second(new_state)
                # t1 = time.perf_counter()
                # print(f"[PROFILE] CONTROLLER ACTION ADVANCE SIM PHASE={t1 - t0:.6f}s")
            # t0 = time.perf_counter()
            self._update_stats(new_state)
            # t1 = time.perf_counter()
            # print(f"[PROFILE] CONTROLLER ACTION RESTORE PHASE={t1 - t0:.6f}s")
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
        violations = st.slo_violations
        # avg_lateness = (
        #     st.slo_lateness_sum / max(violations, 1) if violations else 0.0
        # )
        total_lateness = float(st.slo_lateness_sum)
        return violations, total_lateness


    def describe_state(self, state: VidurMCTSState) -> Dict[str, Any]:
        violations, total_lateness = self.evaluate_objective(state)
        simulator = state.simulator
        request_lookup = self._build_request_lookup(simulator)
        waiting_ids = self._collect_waiting_request_ids(simulator)
        return {
            "sim_time": simulator._time,
            "requests_in_system": len(request_lookup),
            "requests_generated": state.stats.requests_generated,
            "requests_completed": state.stats.requests_completed,
            "slo_violations": violations,
            "total_lateness": total_lateness, # NOTE: now total lateness (sum), not average TODO : make sure this defination becomes consistent everywhere
            "waiting_request_ids": waiting_ids,
            "completed_request_ids": list(state.stats.completed_request_ids),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _apply_adversary_action(
        self, state: VidurMCTSState, action: AdversaryAction
    ) -> None:
        
        if (not action.requests) and (not action.stop_decode_ids):
            return
        sim = state.simulator
        time_now = sim._time 

        # Bucket logical arrival time down to the nearest lowest whole second e.g. if time now is 1.2s --> arrival time is 1.0s
        if action.requests:
            if state.stats.last_prefill_batch_time is None:
                arrival_time = math.floor(time_now)  # or use time_now if you want fractional base
            else:
                arrival_time = state.stats.last_prefill_batch_time + 1.0

            state.stats.last_prefill_batch_time = arrival_time
        else:
            # no new prefill requests; don't touch last_prefill_batch_time
            arrival_time = math.floor(time_now)  # unused since no requests

        # sync the global counter to THIS simulator’s current max request id
        lookup = self._build_request_lookup(sim)
        Request._id = max(lookup.keys()) if lookup else -1

        for spec in action.requests: 
            req = Request(
                arrived_at=arrival_time,
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
            state.stats.recent_arrivals.append(arrival_time)

        # Optionally stop decode on selected requests
        if action.stop_decode_ids:
            request_lookup = self._build_request_lookup(state.simulator)
            for rid in action.stop_decode_ids:
                req = request_lookup.get(rid)
                if req is None:
                    continue
                # Force decode completion at current processed length
                req.num_decode_tokens = max(req.num_processed_decode_tokens, 0)

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

        # IMPORTANT: do NOT modify scheduled_req_ids here.
        # BatchEndEvent / on_batch_end relies on scheduled_req_ids to be
        # consistent with batches already in flight; we only want to hide
        # non-targeted requests from running, not rewrite scheduler history.

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
        # print("Controller Action is : ", action)
        scheduler_snapshot = self._snapshot_scheduler_budget_state(simulator)

        request_lookup = self._build_request_lookup(simulator)   
        ## If no request in the system then no action
        if not request_lookup:
            action.token_allocations.clear()
            action.prefill_allocations.clear()
            action.decode_allocations.clear()
            return None, scheduler_snapshot, set(), {}

        max_budget = self._max_feasible_budget(simulator)
        requested_budget = max(0, int(action.token_budget or 0))
        ## If no token budget then no action 
        if requested_budget <= 0:
            action.token_allocations.clear()
            action.prefill_allocations.clear()
            action.decode_allocations.clear()
            return None, scheduler_snapshot, set(), {}

        ## Replacing the token budget with minimum possible budget in the system if necessary
        token_budget = min(requested_budget, max_budget)
        action.token_budget = token_budget

        selected_ids = list(action.selected_request_ids or [])
        if not selected_ids:
            selected_ids = sorted(request_lookup.keys())

        ## Ensuring the selected ids match with the ids in the system
        selected_ids = [rid for rid in selected_ids if rid in request_lookup]
        if not selected_ids:
            action.token_allocations.clear()
            action.prefill_allocations.clear()
            action.decode_allocations.clear()
            return None, scheduler_snapshot, set(), {}

        action.selected_request_ids = selected_ids
        allocations = dict(action.token_allocations)

        ## If no allocations provided for all requests then no action aswell
        if not allocations:
            return None, scheduler_snapshot, set(), {}
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

        tracked_requests = {rid: request_lookup[rid] for rid in tracker_allocations.keys() if rid in request_lookup}

        tracker = _ControllerBudgetTracker(
            allocations=tracker_allocations,
            baseline_prefill=baseline_prefill,
            baseline_decode=baseline_decode,
            tracked_requests=tracked_requests,
        )
        action.selected_request_ids = sorted(tracker_allocations.keys())
        activated_replicas = {replica_id for replica_id in activated_replicas if replica_id is not None}
        return tracker, scheduler_snapshot, activated_replicas, hidden_state




    def _advance_simulation(self, state: VidurMCTSState, tracker: _ControllerBudgetTracker) -> None:
        from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
        from vidur.types import EventType

        sim = state.simulator

        drop_end_events = (
            (not sim._config.metrics_config.write_metrics)
            and type(sim._scheduler).on_prefill_end is BaseGlobalScheduler.on_prefill_end
            and type(sim._scheduler).on_request_end is BaseGlobalScheduler.on_request_end
        )

        if tracker.is_satisfied():
            self._prune_pending_replica_schedule_events(sim)
            return

        steps = 0
        max_steps = max(1, self._cfg.simulation_depth * 10)

        heappop = heapq.heappop
        add_event = sim._add_event
        scheduler = sim._scheduler
        metrics = sim._cluster_metric_store

        while sim._event_queue and steps < max_steps:
            next_event = sim._event_queue[0]
            if next_event.event_type == EventType.REQUEST_ARRIVAL and next_event._time > sim._time:
                break

            event = heappop(sim._event_queue)
            sim._set_time(event._time)

            new_events = event.handle_event(scheduler, metrics)

            if drop_end_events and event.event_type == EventType.BATCH_END:
                new_events = [e for e in new_events if e.event_type not in (EventType.PREFILL_END, EventType.REQUEST_END)]

            for e in new_events:
                add_event(e)
            steps += 1

            if event.event_type == EventType.BATCH_END and tracker.is_satisfied():
                self._prune_pending_replica_schedule_events(sim)
                break

    def _advance_simulation_fast(
        self,
        state: VidurMCTSState,
        tracker: _ControllerBudgetTracker,
        activated_replicas: Set[Any],
    ) -> None:
        """
        Fast-path controller advance for the common MCTS case:
        - single replica
        - single pipeline stage
        - no future REQUEST_ARRIVAL events pending

        Executes the same semantics as the event chain:
        ReplicaScheduleEvent -> BatchStageArrivalEvent -> ReplicaStageScheduleEvent
        -> BatchStageEndEvent -> BatchEndEvent (+reschedule)
        without using the event heap / BaseEvent.handle_event().
        """
        sim = state.simulator

        # If there are any future request arrivals, event ordering matters -> fall back.
        if any(
            (e.event_type == EventType.REQUEST_ARRIVAL and float(e._time) > float(sim._time))
            for e in getattr(sim, "_event_queue", [])
        ):
            self._advance_simulation(state, tracker)
            return

        if tracker.is_satisfied():
            self._prune_pending_replica_schedule_events(sim)
            return

        # Only safe for single-stage replicas (your config).
        for rid in activated_replicas:
            rs = sim._scheduler.get_replica_scheduler(rid)
            if int(getattr(rs, "_num_stages", 1)) != 1:
                self._advance_simulation(state, tracker)
                return

        # Remove any stale ReplicaScheduleEvent at current time; we drive scheduling manually.
        self._prune_pending_replica_schedule_events(sim)

        max_batches = max(1, self._cfg.simulation_depth * 100)
        batches_executed = 0

        global_sched = sim._scheduler
        get_replica_sched = global_sched.get_replica_scheduler
        get_stage_sched = global_sched.get_replica_stage_scheduler
        set_time = sim._set_time

        blocked_ids: set[int] = set()
        # print("Tracker is : ", tracker.allocations)
        while batches_executed < max_batches and not tracker.is_satisfied():
            made_progress = False

            for replica_id in activated_replicas:
                replica_scheduler = get_replica_sched(replica_id)

                # Mimic ReplicaScheduleEvent: keep trying if scheduler requeued but produced no batch.
                if not replica_scheduler.can_schedule():
                    continue

                while replica_scheduler.can_schedule():
                    out = replica_scheduler.on_schedule(sim._time)
                    batch = getattr(out, "batch", None)

                    if batch is None:
                        # In event-version: if requeued_requests exist, it would schedule another
                        # ReplicaScheduleEvent at the same time; so retry once more here.
                        requeued = getattr(out, "requeued_requests", None) or []
                        if requeued:
                            continue
                        break

                    # batch.on_schedule(time)
                    batch.on_schedule(sim._time)

                    # One stage: directly run the stage then batch_end at end_time
                    stage_scheduler = get_stage_sched(replica_id, 0)
                    stage_scheduler.add_batch(batch)
                    # print("hello" , batch.num_tokens)

                    b, batch_stage, _exec_time = stage_scheduler.on_schedule()
                    # if len(batch.num_tokens) == 1 and batch.num_tokens[0] in (512, 1024, 1536, 2048, 2560, 3072):
                    # print("overall batch :", batch)
                    # print("DEBUG tokens", batch.num_tokens,
                    #     "stage_total", batch_stage.execution_time,
                    #     "stage_model", batch_stage.model_execution_time)
                    # print("\n")
                    # print("DEBUG req->tokens", [(r.id, n) for r, n in zip(batch.requests, batch.num_tokens)])
                    # print("DEBUG overrides(before)", getattr(replica_scheduler, "_token_budget_overrides", None))
                    # print("DEBUG chunk_size", getattr(getattr(replica_scheduler, "_config", None), "chunk_size", None))
                    # print("\n")
                    

                    if b is None or batch_stage is None:
                        # Unexpected; preserve correctness
                        self._advance_simulation(state, tracker)
                        return

                    start_time = float(sim._time)
                    batch_stage.on_schedule(start_time)

                    end_time = start_time + float(batch_stage.execution_time)
                    set_time(end_time)

                    # BatchStageEndEvent semantics
                    stage_scheduler.on_stage_end()
                    batch_stage.on_stage_end(end_time)

                    # BatchEndEvent semantics
                    batch.on_batch_end(end_time)
                    global_sched.on_batch_end(batch)
                    replica_scheduler.on_batch_end(batch)
                    
                    batches_executed += 1
                    # print("Batches executed so far: ", batches_executed)
                    made_progress = True

                    # With 1 stage, we can't schedule another concurrent batch anyway.
                    break

                if tracker.is_satisfied() or batches_executed >= max_batches:
                    break

            if not made_progress:
                break

        if tracker.is_satisfied():
            self._prune_pending_replica_schedule_events(sim)


    def _maybe_fast_forward_decode_only_to_next_adv_second(self, state: VidurMCTSState) -> None:
        sim = state.simulator
        stats = state.stats

        last = getattr(stats, "last_prefill_batch_time", None)
        if last is None:
            return

        sim_t = float(sim._time)
        target_t = float(last) + 1.0
        if sim_t >= target_t - 1e-9:
            return  # adversary already allowed

        # full-system lookup (includes "hidden" requests since they remain in _requests)
        reqs = self._build_request_lookup(sim)
        if not reqs:
            return

        decode_active: list[Request] = []
        for r in reqs.values():
            remaining_prefill = max(0, int(r.num_prefill_tokens) - int(r.num_processed_prefill_tokens))
            prefill_done = bool(getattr(r, "_is_prefill_complete", r.is_prefill_complete))
            if remaining_prefill > 0 and not prefill_done:
                return  # still prefill in system -> do NOT fast-forward

            remaining_decode = max(0, int(r.num_decode_tokens) - int(r.num_processed_decode_tokens))
            if prefill_done and remaining_decode > 0:
                decode_active.append(r)

        if not decode_active:
            return

        # Make time jump explicit
        sim._set_time(target_t)

        # Reset decode deadlines so “skipped” time doesn’t create artificial decode lateness
        for r in decode_active:
            rid = int(r.id)
            decode_slo = getattr(r, "_decode_slo_time", None)
            if decode_slo is None:
                continue
            stats.decode_next_deadline_by_id[rid] = float(target_t) + float(decode_slo)








    def _update_stats(self, state: VidurMCTSState) -> None:
        sim = state.simulator
        stats = state.stats
        sim_time = float(sim._time)

        for replica_scheduler in sim._scheduler._replica_schedulers.values():
            for request in list(replica_scheduler._requests.values()):
                rid = int(request.id)

                # -------------------------
                # Prefill lateness (monotone max; finalized after prefill completes)
                # -------------------------
                if rid not in stats.prefill_lateness_finalized:
                    prefill_slo = getattr(request, "_prefill_slo_time", None)
                    if prefill_slo is not None:
                        arrived_at = float(getattr(request, "_arrived_at", request.arrived_at))
                        deadline = arrived_at + float(prefill_slo)

                        is_prefill_complete = bool(
                            getattr(request, "_is_prefill_complete", request.is_prefill_complete)
                        )
                        prefill_completed_at = getattr(request, "_prefill_completed_at", None)

                        if is_prefill_complete and prefill_completed_at not in (None, 0):
                            actual = float(prefill_completed_at)
                        else:
                            actual = sim_time

                        prefill_late = max(0.0, actual - deadline)

                        prev_prefill = float(stats.per_request_prefill_lateness.get(rid, 0.0))
                        if prefill_late > prev_prefill:
                            stats.slo_lateness_sum += (prefill_late - prev_prefill)
                            stats.per_request_prefill_lateness[rid] = prefill_late

                        if is_prefill_complete:
                            # after completion, prefill lateness is final; no need to recompute again
                            stats.prefill_lateness_finalized.add(rid)

                # -------------------------
                # Decode lateness (cumulative per new decode token)
                # Deadline rule you requested:
                #   - first deadline = prefill_completed_at + decode_slo
                #   - after each token: next_deadline = sim_time + decode_slo
                # -------------------------
                decode_slo = getattr(request, "_decode_slo_time", None)
                has_decode_tokens = int(getattr(request, "_num_decode_tokens", request.num_decode_tokens)) > 0

                is_prefill_complete = bool(
                    getattr(request, "_is_prefill_complete", request.is_prefill_complete)
                )
                prefill_completed_at = getattr(request, "_prefill_completed_at", None)

                if (
                    decode_slo is not None
                    and float(decode_slo) >= 0.0
                    and has_decode_tokens
                    and is_prefill_complete
                    and prefill_completed_at not in (None, 0)
                ):
                    # init first decode deadline if missing
                    if rid not in stats.decode_next_deadline_by_id:
                        stats.decode_next_deadline_by_id[rid] = float(prefill_completed_at) + float(decode_slo)

                    done = int(request.num_processed_decode_tokens)
                    counted = int(stats.decode_tokens_counted.get(rid, 0))
                    new_tokens = done - counted

                    if new_tokens:
                        # You asked for this assert:
                        assert new_tokens == 1, f"Expected 1 new decode token for req {rid}, got {new_tokens}"

                        deadline = float(stats.decode_next_deadline_by_id[rid])
                        token_late = max(0.0, sim_time - deadline)

                        # accumulate lateness for this request + global sum
                        stats.per_request_decode_lateness[rid] = float(
                            stats.per_request_decode_lateness.get(rid, 0.0)
                        ) + float(token_late)
                        stats.slo_lateness_sum += float(token_late)

                        # advance counters + set next deadline relative to *now* (your rule)
                        stats.decode_tokens_counted[rid] = done
                        stats.decode_next_deadline_by_id[rid] = sim_time + float(decode_slo)

                # -------------------------
                # Violations (once per request)
                # -------------------------
                total_lateness = float(stats.per_request_prefill_lateness.get(rid, 0.0)) + float(
                    stats.per_request_decode_lateness.get(rid, 0.0)
                )
                if total_lateness > 0.0 and rid not in stats.violated_request_ids:
                    stats.violated_request_ids.add(rid)
                    stats.slo_violations += 1

        # Track completions (same as your current logic)
        for replica_scheduler in sim._scheduler._replica_schedulers.values():
            for request in list(replica_scheduler._requests.values()):
                if request.completed and request.id not in stats.completed_request_ids:
                    stats.requests_completed += 1
                    stats.completed_request_ids.add(request.id)
            

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

    ## Needed to stop next batch execution because Batch_end event triggers next batch schedule event
    ## Hence neeeded so that controller' actions end cleanly
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


    def _populate_Prefill_Reqs_Table(self) -> None:
        
        ## Populates the adversary requests options (that are order agnostic) depending on Min and Max request length of prefill sizes and request interval step size. 

        function_path: str = "enviorment.py function _populate_prefill_reqs_table()"

        ## Step 1 : Getting the neccesary variables for combos :
        step = int(self._constraints.interval_request_size)
        min_tok_limit = max(step, int(self._constraints.min_request_tokens))
        max_tok_limit = int(self._max_request_tokens_allowed())
        qps = int(self._constraints.maximum_qps or 0)

        assert qps > 0 , f" Invalid QPS : {qps} passed to {function_path}. Must be greater than 0"
        assert step > 0, f" Invalid STEP : {step} passed to {function_path}. Must be greater than 0"

        # Step 2 : Normalising limits to multiple of the Step Size :
        lo = ((min_tok_limit + step - 1) // step ) * step ## Ceiling applied 
        hi = (max_tok_limit // step) * step ## Flooring applied 

        assert hi >= lo , f" Maximum Request Size {max_tok_limit} has been passed as smaller size than smaller {min_tok_limit} in {function_path}"

        # Step 3 : Possible Size options :
        size_options = list(range(lo , hi + 1, step)) # produces [step , 2xstep , .... hi x step]
        
        # Step 4 : Generate Combos :
        temp_table: Dict[int, List[int]] = {}
        idx = 0 
        for combo in combinations_with_replacement(size_options, qps):
            temp_table[idx] = list(combo) ;  idx += 1 

        self.all_possible_Prefill_reqs_table = temp_table    


    ## THESE FUNCTIONS ARE TO PRE-COMPUTE NECESSARY STATES :
    def _generate_prefill_sequences_with_budget(
        self, token_budget: int
    ) -> List[List[int]]:
        vals = list(self.controller_all_possible_prefill_budgets or [])
        L = len(vals)
        result: List[List[int]] = []
        current: List[int] = []

        def backtrack(pos: int, remaining: int) -> None:
            if pos == L:
                if current:
                    result.append(current.copy())
                return
            if current:
                seq = current + [0] * (L - pos)
                result.append(seq)
            if remaining <= 0:
                return
            for v in vals:
                if v <= remaining:
                    current.append(v)
                    backtrack(pos + 1, remaining - v)
                    current.pop()

        backtrack(0, token_budget)
        return result

    def precompute_controller_state_space(self, token_budget: int) -> None:
        if self.all_possible_Controller_States:
            return

        # ensure budgets list is filled (same logic you already use)
        if self.controller_all_possible_prefill_budgets is None:
            self.controller_all_possible_prefill_budgets = []
            step = self._constraints.interval_request_size
            min_tok = max(step, self._constraints.min_request_tokens)
            max_tok = self._max_request_tokens_allowed()
            lo = (min_tok + step - 1) // step
            hi = max_tok // step
            for i in range(lo, hi + 1):
                self.controller_all_possible_prefill_budgets.append(i * step)

        mappings = self._generate_prefill_sequences_with_budget(token_budget)
        heuristic_names = ["SJF", "EDF", "LST", "LJF"]
        strategies = ["All Allocation", "Max Allocation"]

        for m in mappings:
            mt = tuple(m)
            for h in heuristic_names:
                for s in strategies:
                    key = (mt, h, s)
                    if key not in self.all_possible_Controller_States:
                        self.all_possible_Controller_States[key] = {
                            "sample_visits": 0,
                            "mcts_visits": 0,
                            "cumulative_cost": 0.0,
                            "mean_cost": 0.0,
                            "cumulative_delta_cost": 0.0,
                            "mean_delta_cost": 0.0,
                            "last_cost": 0.0,
                            "last_delta_cost": 0.0,
                            "last_slo_violations": 0,
                            "last_avg_lateness": 0.0,
                            "total_decode_tokens": 0,
                            "last_decode_tokens": 0,
                        }


    def controller_states_fully_visited(self) -> bool:
        if not self.all_possible_Controller_States:
            return False
        return all(
            s.get("sample_visits", 0) >= 1
            for s in self.all_possible_Controller_States.values()
        )


    def update_controller_state_metrics(
        self,
        action: ControllerAction,
        total_cost: float,
        delta_cost: float,
        slo_violations: int,
        avg_lateness: float,
    ) -> None:
        if action.mapping is None or not action.heuristic or not action.strategy:
            return
        key = (tuple(action.mapping), action.heuristic, action.strategy)
        state = self.all_possible_Controller_States.get(key)
        if state is None:
            state = {
                "sample_visits": 0,
                "mcts_visits": 0,
                "cumulative_cost": 0.0,
                "mean_cost": 0.0,
                "cumulative_delta_cost": 0.0,
                "mean_delta_cost": 0.0,
                "last_cost": 0.0,
                "last_delta_cost": 0.0,
                "last_slo_violations": 0,
                "last_avg_lateness": 0.0,
                "total_decode_tokens": 0,
                "last_decode_tokens": 0,
            }
            self.all_possible_Controller_States[key] = state

        # Absolute cost stats for this state
        state["mcts_visits"] += 1
        state["cumulative_cost"] += total_cost
        state["mean_cost"] = state["cumulative_cost"] / max(1, state["mcts_visits"])
        state["last_cost"] = total_cost

        # Incremental cost stats (this decision only)
        state["cumulative_delta_cost"] += delta_cost
        state["mean_delta_cost"] = state["cumulative_delta_cost"] / max(1, state["mcts_visits"])
        state["last_delta_cost"] = delta_cost

        # SLO info
        state["last_slo_violations"] = slo_violations
        state["last_avg_lateness"] = avg_lateness

        # Decode tokens allocated by this controller decision
        decode_tokens = sum(action.decode_allocations.values())
        state["total_decode_tokens"] += decode_tokens
        state["last_decode_tokens"] = decode_tokens


