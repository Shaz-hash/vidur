
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
    # NEW: per-request max lateness and permanent violation flags
    per_request_max_lateness: Dict[int, float] = field(default_factory=dict)
    violated_request_ids: Set[int] = field(default_factory=set)

    def clone(self) -> "VidurGameStats":
        return VidurGameStats(
            requests_generated=self.requests_generated,
            requests_completed=self.requests_completed,
            slo_violations=self.slo_violations,
            slo_lateness_sum=self.slo_lateness_sum,
            recent_arrivals=list(self.recent_arrivals),
            completed_request_ids=set(self.completed_request_ids),
            per_request_max_lateness=dict(self.per_request_max_lateness),
            violated_request_ids=set(self.violated_request_ids),
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
        self.all_possible_Prefill_reqs_table: Dict[int, List[int]] = None
        self.controller_all_possible_prefill_budgets: List[int] = None
        self.all_possible_Controller_actions: List[any] = None
        self.all_possible_Controller_States: Dict[Tuple[Tuple[int, ...], str, str], Dict[str, Any]] = {}
        # key = (mapping_tuple, heuristic_name, allocation_strategy_name)

        # Take a frozen snapshot of the root simulator once, so that every
        # initial MCTS state starts from the exact same simulator + ID counters.
        self._base_snapshot = base_simulator.snapshot_state()

        if self._constraints.max_request_tokens is None:
            self._constraints.max_request_tokens = self._prefill_profile.max_tokens


    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def initial_state(self) -> VidurMCTSState:
        # Always recreate a fresh simulator from the frozen root snapshot.
        # This ensures request IDs and entity counters are identical for every
        # replay from the root, regardless of what other simulators did.
        sim = Simulator(
            self._base._config,
            register_atexit=False,
            execution_time_predictor=getattr(self._base, "_execution_time_predictor", None),
        )
        sim.restore_state(self._base_snapshot)
        return VidurMCTSState(sim, VidurGameStats())

    # ------------------------------------------------------------------ #
    # Action generation helpers
    # ------------------------------------------------------------------ #
    def sample_adversary_actions(
        self, state: VidurMCTSState, max_samples: int
    ) -> List[AdversaryAction]:
        """Generate adversary actions with fixed QPS and decode-stop options.

        - Generates exactly = QPS new prefill requests, all with the same prefill size
          chosen from [min..max] by step.
        - For decode-eligible requests whose processed decode tokens are a multiple of step,
          the adversary may choose to stop decode (set remaining decode to 0). The action
          enumerates subsets of such requests (limited by max_samples or max_branching) assuming prefill & Decode SLOs are fixed to 1 option.
        """
        function_path: str = "enviroment.py function sample_adversary_actions()"
        actions: List[AdversaryAction] = []
        qps_budget = self._available_qps_budget(state)
        max_qps = int(self._constraints.maximum_qps or 0)
        step = int(self._constraints.interval_request_size)
        slo_opts = self._constraints.request_slo_options
        cap = max_samples if (max_samples and max_samples > 0) else self._cfg.max_branching
        cap = max(1, int(cap))

        # Decode subsets: random unique subsets up to cap (including empty)
        def build_decode_subsets_random(boundary_decode: List[int], limit: int) -> List[List[int]]:
            if not boundary_decode:
                return [[]]
            n = len(boundary_decode)
            total = 1 << n
            if total <= limit:
                out: List[List[int]] = [[]]
                for mask in range(1, total):
                    subset = [boundary_decode[i] for i in range(n) if (mask >> i) & 1]
                    out.append(subset)
                return out[:limit]
            # Sample unique subsets randomly (include empty)
            seen: Set[Tuple[int, ...]] = {tuple()}
            out: List[List[int]] = [[]]
            trials = 0
            while len(out) < limit and trials < limit * 10:
                trials += 1
                subset = [rid for rid in boundary_decode if self._rng.random() < 0.5]
                key = tuple(sorted(subset))
                if key not in seen:
                    seen.add(key)
                    out.append(list(key))
            return out


         # Build specs converter once
       
        # Configures requests properly for the Simulator
        def specs_from_prefill_combo(combo: List[int]) -> List[AdversaryRequestSpec]:
            specs: List[AdversaryRequestSpec] = []
            MAX_REQUEST_LENGTH: int = 10240 ## ?? Note : Have this variable later passed from the config instead
            #total_budget = self._max_request_tokens_allowed()

            for prefill_size in combo:
                remaining_for_decode = max(0, MAX_REQUEST_LENGTH - prefill_size) ## ?? Note : Decode SLOS will be needed to take randomly here when we move from to multiple options next week IA
                prefill_slo = self._prefill_profile.lookup(prefill_size)
                if getattr(slo_opts, "prefill_slos", None):
                    prefill_slo *= (
                        slo_opts.prefill_slos[0]
                        if len(slo_opts.prefill_slos) == 1
                        else self._rng.choice(slo_opts.prefill_slos)
                    )
                if getattr(slo_opts, "decode_slos", None):
                    dec_ms = (
                        slo_opts.decode_slos[0]
                        if len(slo_opts.decode_slos) == 1
                        else self._rng.choice(slo_opts.decode_slos)
                    )
                    decode_slo = float(dec_ms) / 1000.0
                else:
                    decode_slo = 0.0
                specs.append(
                    AdversaryRequestSpec(
                        prefill_tokens=prefill_size,
                        decode_tokens=remaining_for_decode,
                        prefill_slo=prefill_slo,
                        decode_slo=decode_slo,
                    )
                )
            return specs


        assert max_qps > 0 , f" Max QPS ! > 0 {max_qps} in {function_path}"
        ## Since we are making Adversary always send the max QPS budget every second :
        if(qps_budget >= max_qps): ## This means we can go for the multiple actions now at the adversary here 

            ## Step 1 : Check whether we have all possible requests combinations produced already : 
            if (self.all_possible_Prefill_reqs_table is None):
                ## Step 1.1 : Fill the table first quickly to cache the results for the future:
                self._populate_Prefill_Reqs_Table()
            

             # Prefill combos: random sample up to cap from cache
            all_combos = (
                list(self.all_possible_Prefill_reqs_table.values())
                if self.all_possible_Prefill_reqs_table else []
            )

            assert len(all_combos) > 0, f"All combinations after creating tables still are {len(all_combos)}? in {function_path}"
            
            if len(all_combos) > cap:
                prefill_combos = self._rng.sample(all_combos, cap)
            else:
                prefill_combos = all_combos

            ## Step 2 : Find how many decodes eligible to stop are here 
            request_lookup = self._build_request_lookup(state.simulator)
            boundary_decode: List[int] = []

            for rid, req in request_lookup.items():
                if getattr(req, "_is_prefill_complete", req.is_prefill_complete):
                    rem_dec = max(0, req.num_decode_tokens - req.num_processed_decode_tokens) 
                    proc = max(0, req.num_processed_decode_tokens)
                    if rem_dec > 0 and proc >= step and (proc % step == 0):
                        boundary_decode.append(rid)

            ## Step 2.1 : Get the necessary random sample of the eligible decodes :
            decode_subsets = build_decode_subsets_random(boundary_decode, cap)


            ## Step 3 : Building the combinations for the Adv actions : 
            
            pairs_total = len(prefill_combos) * len(decode_subsets)            

            if pairs_total <= cap:
                ## Add every possibile combination now :
                specs_cache = [specs_from_prefill_combo(c) for c in prefill_combos]
                for i in range(len(prefill_combos)):
                    specs = specs_cache[i]
                    for stop_ids in decode_subsets:
                        actions.append(AdversaryAction(requests=list(specs), stop_decode_ids=list(stop_ids)))
            else :
                ## Sampling k unique linear iondecies without materialising product here :
                sampled = self._rng.sample(range(pairs_total), cap) ## picks any caps numbers from 0 to pairs_total - 1
                # Precompute specs only for the combos we actually picked
                combo_idxs = {idx // len(decode_subsets) for idx in sampled}
                specs_cache = {i: specs_from_prefill_combo(prefill_combos[i]) for i in combo_idxs}

                for idx in sampled:
                    i = idx // len(decode_subsets)
                    j = idx % len(decode_subsets)
                    actions.append(
                        AdversaryAction(
                            requests=list(specs_cache[i]),          # copy to avoid aliasing
                            stop_decode_ids=list(decode_subsets[j])
                        )
                    )
        
        else :
            ## Return Fallback here with no Prefill release but decode related actions are possible here :

            request_lookup = self._build_request_lookup(state.simulator)
            boundary_decode: List[int] = []


            for rid, req in request_lookup.items():
                if getattr(req, "_is_prefill_complete", req.is_prefill_complete):
                    rem_dec = max(0, req.num_decode_tokens - req.num_processed_decode_tokens) 
                    proc = max(0, req.num_processed_decode_tokens)
                    if rem_dec > 0 and proc >= step and (proc % step == 0):
                        boundary_decode.append(rid)

            ## Step 1 : Get the necessary random sample of the eligible decodes :
            decode_subsets = build_decode_subsets_random(boundary_decode, cap)
            for stop_ids in decode_subsets:
                actions.append(AdversaryAction(requests=[], stop_decode_ids=list(stop_ids)))           
        
        return actions

    def sample_controller_actions(
        self, state: VidurMCTSState, max_samples: int, use_state_cache: bool = True
    ) -> List[ControllerAction]:
        """Generate controller actions using 4 heuristics × 2 allocation strategies × budgets.

        Heuristics (ordering prefill candidates):
        - SJF (ascending remaining prefill)
        - EDF (ascending absolute deadline)
        - LST (ascending slack = remaining SLO time - estimated remaining process time)
        - Slowdown (descending slowdown ratio)

        Allocation strategies:
        - single: allocate whole budget to the first request
        - progressive: split among top requests by 50%/ceil and rules described

        TODO :
        --> log times in here to see how much sample controller action is taking here 
        --> Connect the MAX Token request size from config to here aswell
        --> Connect the visit_limit directly to the configuration 
        """
        
        MAX_REQUEST_LENGTH: int = 10240
        VISIT_LIMIT = 1
        self._drain_arrivals(state.simulator)
        sim_time = state.simulator._time
        step = self._constraints.interval_request_size

        # Build lookup once
        request_lookup = self._build_request_lookup(state.simulator)
        waiting_ids_all = sorted(request_lookup.keys())
        if not waiting_ids_all:
            return [ControllerAction(token_budget=0, selected_request_ids=None)]

        # Prefill candidates only
        def remaining_prefill(req: Request) -> int:
            return max(0, req.num_prefill_tokens - req.num_processed_prefill_tokens)

        def prefill_done(req: Request) -> bool:
            return getattr(req, "_is_prefill_complete", req.is_prefill_complete)

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
            rem_dec = max(0, req.num_decode_tokens - req.num_processed_decode_tokens) ##?? NOTE: This is not needed in here 
            if prefill_done(req) and rem_dec > 0:
                decode_candidates.append(rid)

        # If no prefill candidates, action is to include all decode candidates only
        if not prefill_ids:
            if decode_candidates:
                decode_base = {rid: 1 for rid in decode_candidates}
                sel = sorted(decode_candidates)
                tot = len(decode_candidates)
                return [
                    ControllerAction(
                        token_budget=tot,
                        selected_request_ids=sel,
                        token_allocations=dict(decode_base),
                        prefill_allocations={},
                        decode_allocations=decode_base,
                    )
                ]
            # Nothing to do
            return [ControllerAction(token_budget=0, selected_request_ids=None)]

        # Budgets from min..max inclusive by step
        min_tok = max(step, self._constraints.min_request_tokens)
        max_tok = self._max_request_tokens_allowed()

        ## First check if the budgets are all calculated or not yet 
        if (self.controller_all_possible_prefill_budgets == None):
            self.controller_all_possible_prefill_budgets = [] ## NOTE: Only needed once and we can store this 
            lo = (min_tok + step - 1) // step
            hi = max_tok // step
            for i in range(lo, hi + 1):
                self.controller_all_possible_prefill_budgets.append(i * step)
        

        # Ordering helpers
        def order_sjf(ids: List[int]) -> List[int]:
            return sorted(ids, key=lambda rid: remaining_prefill(request_lookup[rid]))

        def order_edf(ids: List[int]) -> List[int]:
            # absolute deadline = arrival + prefill SLO time
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
            # Descending: prioritize highest slowdown
            return sorted(ids, key=ratio, reverse=True)

        # heuristics = [order_sjf, order_edf, order_lst, order_slowdown]
        # New
        heuristics = [
            ("SJF", order_sjf),
            ("EDF", order_edf),
            ("LST", order_lst),
            ("Slowdown", order_slowdown),
        ]
        def ceil_to_step(x: int) -> int:
            return step * ((x + step - 1) // step)

        # Allocation Strategies 
        def build_alloc_single(
            ordered: List[int],
            budget: int,
        ) -> Tuple[ControllerAction, List[int]]:
            # Length of the prefill pattern we care about
            L = len(self.controller_all_possible_prefill_budgets)
            mapping: List[int] = [0] * L

            if budget <= 0 or not ordered or L == 0:
                return ControllerAction(token_budget=0, selected_request_ids=None), mapping

            remaining_budget = budget
            pre: Dict[int, int] = {}

            # Allocate greedily across the first L requests in `ordered`
            k = min(L, len(ordered))
            for idx in range(k):
                rid = ordered[idx]
                cap = remaining_prefill(request_lookup[rid])
                if cap <= 0 or remaining_budget <= 0:
                    continue
                alloc_here = min(cap, remaining_budget)
                pre[rid] = alloc_here
                mapping[idx] = alloc_here
                remaining_budget -= alloc_here
                if remaining_budget <= 0:
                    break

            if not pre and not decode_candidates:
                return ControllerAction(token_budget=0, selected_request_ids=None), mapping

            # Decode part (unchanged): 1 decode token per decode candidate
            decode_base = {did: 1 for did in decode_candidates}
            sel_set = set(pre.keys()) | set(decode_candidates)
            sel = sorted(sel_set)
            tot = sum(pre.values()) + len(decode_base)

            action = ControllerAction(
                token_budget=tot,
                selected_request_ids=sel,
                token_allocations={**pre, **decode_base},
                prefill_allocations=pre,
                decode_allocations=decode_base,
            )
            return action, mapping

        def build_alloc_progressive(
            ordered: List[int],
            budget: int
        ) -> Tuple[ControllerAction, List[int]]: 
            L = len(self.controller_all_possible_prefill_budgets)
            mapping: List[int] = [0] * L

            if budget <= 0 or not ordered or L == 0:
                return ControllerAction(token_budget=0, selected_request_ids=None), mapping

            remaining = budget
            pre: Dict[int, int] = {}

            k = min(L, len(ordered))
            for idx in range(k):
                if remaining <= 0:
                    break
                rid = ordered[idx]
                cap = remaining_prefill(request_lookup[rid])
                if cap <= 0:
                    continue

                # For the first request use 50% of original budget, then 50% of remaining
                half_base = budget if idx == 0 else remaining
                half = ceil_to_step(half_base // 2)

                # Desired allocation per your rule:
                # max(min(50% base, prefill remaining), step), then clip to remaining budget
                desired = max(
                    step,
                    min(half, cap),
                )
                alloc_here = min(desired, cap, remaining)

                if alloc_here <= 0:
                    continue

                pre[rid] = alloc_here
                mapping[idx] = alloc_here
                remaining -= alloc_here

            if not pre and not decode_candidates:
                return ControllerAction(token_budget=0, selected_request_ids=None), mapping

            decode_base = {did: 1 for did in decode_candidates}
            sel_set = set(pre.keys()) | set(decode_candidates)
            sel = sorted(sel_set)
            tot = sum(pre.values()) + len(decode_base)

            action = ControllerAction(
                token_budget=tot,
                selected_request_ids=sel,
                token_allocations={**pre, **decode_base},
                prefill_allocations=pre,
                decode_allocations=decode_base,
            )
            return action, mapping
        ##-----
        # actions: List[ControllerAction] = []

        # for heur_name, order_fn in heuristics:
        #     ordered = order_fn(prefill_ids)
        #     for b in self.controller_all_possible_prefill_budgets:
        #         # Single allocation (“All Allocation”)
        #         a1, l1 = build_alloc_single(ordered, b)
        #         # Progressive allocation (“Max Allocation”)
        #         a2, l2 = build_alloc_progressive(ordered, b)
        #         m1 = tuple(l1)
        #         m2 = tuple(l2)
        #         a1.heuristic = heur_name
        #         a1.strategy = "All Allocation"
        #         a1.mapping = m1

        #         a2.heuristic = heur_name
        #         a2.strategy = "Max Allocation"
        #         a2.mapping = m2


        #         if use_state_cache:
        #             # print("Creating actual Action")
        #             mapping1 = tuple(l1)
        #             key1 = (mapping1, heur_name, "All Allocation")
        #             if key1 not in self.all_possible_Controller_States:
        #                 self.all_possible_Controller_States[key1] = {
        #                     "visits": 0,
        #                     "objective_cost": 0.0,
        #                     "ucb_score": 0.0,
        #                     "slo_violations": 0,
        #                     "avg_lateness": 0.0,
        #                 }
        #             # else :
        #             #     print(f"KEY WAS FOUND IN THE STATE {(key1)}")
        #             state1 = self.all_possible_Controller_States[key1]
        #             if a1.token_budget > 0 and state1["sample_visits"] < VISIT_LIMIT:
        #                 state1["sample_visits"] += 1
        #                 actions.append(a1)

        #             mapping2 = tuple(l2)
        #             key2 = (mapping2, heur_name, "Max Allocation")
        #             if key2 not in self.all_possible_Controller_States:
        #                 self.all_possible_Controller_States[key2] = {
        #                     "visits": 0,
        #                     "objective_cost": 0.0,
        #                     "ucb_score": 0.0,
        #                     "slo_violations": 0,
        #                     "avg_lateness": 0.0,
        #                 }
        #             # else :
        #             #     print(f"KEY WAS FOUND IN THE STATE {(key2)}")
        #             state2 = self.all_possible_Controller_States[key2]
        #             if a2.token_budget > 0 and state2["sample_visits"] < VISIT_LIMIT:
        #                 state2["sample_visits"] += 1
        #                 actions.append(a2)
        #         else:
        #             # Rollout mode: ignore Controller_States and visit caps
        #             # print("Running Simulation!")
        #             if a1.token_budget > 0:
        #                 actions.append(a1)
        #             if a2.token_budget > 0:
        #                 actions.append(a2)
                        
        actions: List[ControllerAction] = []

        # Enumerate all heuristics × budgets × strategies,
        # without using the controller state cache or visit limits.
        for heur_name, order_fn in heuristics:
            ordered = order_fn(prefill_ids)
            for b in self.controller_all_possible_prefill_budgets:
                # Single allocation (“All Allocation”)
                a1, l1 = build_alloc_single(ordered, b)
                a1.heuristic = heur_name
                a1.strategy = "All Allocation"
                a1.mapping = tuple(l1)
                if a1.token_budget > 0:
                    actions.append(a1)

                # Progressive allocation (“Max Allocation”)
                a2, l2 = build_alloc_progressive(ordered, b)
                a2.heuristic = heur_name
                a2.strategy = "Max Allocation"
                a2.mapping = tuple(l2)
                if a2.token_budget > 0:
                    actions.append(a2)
        # Ensure we always return something
        if not actions:
            print(f"NO ACTION PRODUCED !!! ERROR POSSIBLY ON THE STATE : {state}")
            return [ControllerAction(token_budget=0, selected_request_ids=None)]
        return actions

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
        # t0 = time.perf_counter()
        target_state = state if inplace else state.fork()
        # t1 = time.perf_counter()
        self._apply_adversary_action(target_state, action)
        # t2 = time.perf_counter()
        self._drain_arrivals(target_state.simulator)
        # t3 = time.perf_counter()
        # print(
        #     f"[PROFILE] FORK phase={t1 - t0:.4f}s: "
        #     f"APPLYING ADVERSARY ACTION={t2 - t1:.4f}s "
        #     f"DRAIN ARRIVALS={t3 - t2:.4f}s"
        # )
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
        violations = st.slo_violations
        avg_lateness = (
            st.slo_lateness_sum / max(violations, 1) if violations else 0.0
        )
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
        time_now = sim._time 

        # Bucket logical arrival time down to the nearest lowest whole second e.g. if time now is 1.2s --> arrival time is 1.0s
        arrival_time = math.floor(time_now)

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

        # 1) For every request we know about, update per-request max lateness
        #    and permanent violation flags.
        for replica_scheduler in sim._scheduler._replica_schedulers.values():
            for request in list(replica_scheduler._requests.values()):
                rid = request.id
                lateness = self._compute_lateness(request, sim._time)

                # Monotone lateness: accumulate only the *increase* in max lateness
                prev_max = stats.per_request_max_lateness.get(rid, 0.0)
                if lateness > prev_max:
                    stats.slo_lateness_sum += (lateness - prev_max)
                    stats.per_request_max_lateness[rid] = lateness

                # Monotone violations: once late, always counted as a violation
                if lateness > 0 and rid not in stats.violated_request_ids:
                    stats.violated_request_ids.add(rid)
                    stats.slo_violations += 1

        # 2) Track completions (this stays monotone as before)
        for replica_scheduler in sim._scheduler._replica_schedulers.values():
            for request in list(replica_scheduler._requests.values()):
                if request.completed and request.id not in stats.completed_request_ids:
                    stats.requests_completed += 1
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
        heuristic_names = ["SJF", "EDF", "LST", "Slowdown"]
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


