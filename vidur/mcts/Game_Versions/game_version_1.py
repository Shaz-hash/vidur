### بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ 


from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from vidur.entities import Request
from ..game_types import AdversaryAction, AdversaryRequestSpec, ControllerAction


class GameVersion1Rules:
    """
    Behavior-preserving extraction of current v1 game rules:
    - Adversary can send every 1.0s, with 6 fixed action sizes (1..6 requests).
    - Controller action space is 24 fixed actions: 6 budgets x 4 heuristics.
    - Decode-only fast-forward target is last_prefill_batch_time + 1.0.
    """

    name = "game_version_1"
    send_interval_sec = 1.0
    adversary_num_actions = 6
    fixed_decode_tokens = 5000
    supports_native_controller_sampler = True  # current native sampler parity is v1 only

    def sample_adversary_actions(
        self,
        *,
        sim_time: float,
        last_prefill_batch_time: Optional[float],
        prefill_tokens: int, 
        prefill_slo_time: float,
        decode_slo_time: float
    ) -> Tuple[List[Optional[AdversaryAction]], List[bool]]:
            
        num_actions = int(self.adversary_num_actions)
        actions_by_index: List[Optional[AdversaryAction]] = [None] * num_actions
        mask: List[bool] = [False] * num_actions

        send_flag = ( last_prefill_batch_time is None) or (float(sim_time) >= float(last_prefill_batch_time) + self.send_interval_sec - 1e-9)

        if not send_flag : 
            actions_by_index[0] = AdversaryAction(requests=[], stop_decode_ids=[])
            mask[0] = True
            return actions_by_index, mask
        

        for i in range(num_actions):

            num_requests = i + 1
            spec = [
                AdversaryRequestSpec(
                    prefill_tokens=int(prefill_tokens),
                    decode_tokens=int(self.fixed_decode_tokens),
                    prefill_slo=float(prefill_slo_time),
                    decode_slo=float(decode_slo_time)
                )
                for _ in range(num_requests)
            ]
            actions_by_index[i] = AdversaryAction(requests=spec, stop_decode_ids=[])
            mask[i] = True

        return actions_by_index, mask



    def compute_adversary_arrival_time(
        self,
        *,
        sim_time: float,
        last_prefill_batch_time: Optional[float],
        has_requests: bool 
    ) -> float:
        if not has_requests:
            return float(math.floor(sim_time))
        
        if last_prefill_batch_time is None:
            return float(math.floor(sim_time))

        return float(last_prefill_batch_time) + float(self.send_interval_sec)


    def next_last_prefill_batch_time(
        self,
        *,
        sim_time: float,
        last_prefill_batch_time: Optional[float],
        has_requests: bool 
    ) -> Optional[float]:
        if not has_requests:
            return last_prefill_batch_time
        return self.compute_adversary_arrival_time(sim_time=sim_time, last_prefill_batch_time=last_prefill_batch_time, has_requests=has_requests)


    def next_adversary_release_time(
        self,
        *,
        last_prefill_batch_time: Optional[float],
    ) -> Optional[float]:
        if last_prefill_batch_time is None:
            return None
        return float(last_prefill_batch_time) + float(self.send_interval_sec)




    def arrival_window_start(self, *, sim_time: float) -> float:
        return float(sim_time) - float(self.send_interval_sec) 


    def sample_controller_actions(
        self, 
        *,
        request_lookup: Dict[int, Request],
        sim_time: float,
        prefill_step: int, 
        prefill_eta_table_lookup_fn: Callable[[int], float],
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:

        num_heur = 4
        budgets: List[int] = [int(prefill_step) * i for i in range(1, 7)]
        num_actions = num_heur * len(budgets)
        actions_by_index: List[Optional[ControllerAction]] = [None] * num_actions
        mask: List[bool] = [False] * num_actions

        if not request_lookup:
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True
            return actions_by_index, mask

        prefill_ids: List[int] = []
        decode_candidates: List[int] = []
        rem_pref_by_id: Dict[int, int] = {}
        arrived_by_id: Dict[int, float] = {}
        prefill_slo_by_id: Dict[int, float] = {}
        total_remaining_prefill = 0

        for rid in sorted(request_lookup.keys()):
            req = request_lookup[rid]
            rem_pref = max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))
            rem_dec = max(0, int(req.num_decode_tokens) - int(req.num_processed_decode_tokens))
            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))

            arrived_by_id[rid] = float(getattr(req, "arrived_at", 0.0))
            prefill_slo_by_id[rid] = float(getattr(req, "prefill_slo_time", 0.0))

            if (not prefill_done) and rem_pref > 0:
                prefill_ids.append(rid)
                rem_pref_by_id[rid] = rem_pref
                total_remaining_prefill += rem_pref
            elif prefill_done and rem_dec > 0:
                decode_candidates.append(rid)


        decode_candidates = sorted(decode_candidates)

        if total_remaining_prefill == 0:
            decode_alloc = { rid : 1 for rid in decode_candidates }
            token_alloc = dict(decode_alloc)
            selected = sorted(token_alloc.keys()) if token_alloc else None
            actions_by_index[0] = ControllerAction(
                token_budget=int(sum(token_alloc.values())),
                selected_request_ids=selected,
                token_allocations=token_alloc,
                prefill_allocations={},
                decode_allocations=decode_alloc,
                heuristic="SJF",
                strategy="Fixed",
            )
            mask[0] = True
            return actions_by_index, mask
        
        def order_sjf(ids: List[int]) -> List[int]:
            return sorted(ids, key=lambda rid: (rem_pref_by_id[rid], rid))

        def order_edf(ids: List[int]) -> List[int]:
            return sorted(ids, key=lambda rid: (arrived_by_id[rid] + prefill_slo_by_id[rid], rid))

        def order_lst(ids: List[int]) -> List[int]:
            def slack(rid: int) -> float:
                remaining_slo = prefill_slo_by_id[rid] - max(0.0, float(sim_time) - arrived_by_id[rid])
                est = float(prefill_eta_table_lookup_fn(int(rem_pref_by_id[rid])))
                return remaining_slo - est
            return sorted(ids, key=lambda rid: (slack(rid), rid))


        def order_ljf(ids: List[int]) -> List[int]:
            return sorted(ids, key=lambda rid: (-rem_pref_by_id[rid], rid))

        heuristics = [
            ("SJF", order_sjf),
            ("EDF", order_edf),
            ("LST", order_lst),
            ("LJF", order_ljf),
        ]

        def build_action(ordered_prefill: List[int], prefill_budget: int, heur_name: str) -> ControllerAction:
            remaining_budget = max(0, int(prefill_budget))
            pre: Dict[int, int] = {}

            for rid in ordered_prefill:
                if remaining_budget <= 0:
                    break
                cap = int(rem_pref_by_id[rid])
                if cap <= 0:
                    continue
                alloc = cap if cap < remaining_budget else remaining_budget
                pre[rid] = alloc
                remaining_budget -= alloc

            dec = {rid: 1 for rid in decode_candidates}
            token_alloc = dict(dec)
            token_alloc.update(pre)
            selected = sorted(token_alloc.keys()) if token_alloc else None

            return ControllerAction(
                token_budget=int(sum(token_alloc.values())),
                selected_request_ids=selected,
                token_allocations=token_alloc,
                prefill_allocations=pre,
                decode_allocations=dec,
                heuristic=heur_name,
                strategy="Fixed",
            )

        for b_idx, budget in enumerate(budgets):
            budget_valid = (int(budget) <= int(total_remaining_prefill)) if total_remaining_prefill > 0 else (b_idx == 0)
            for h_idx, (h_name, order_fn) in enumerate(heuristics):
                idx = b_idx * num_heur + h_idx
                if not budget_valid:
                    actions_by_index[idx] = None
                    mask[idx] = False
                    continue
                ordered = order_fn(prefill_ids) if prefill_ids else []
                actions_by_index[idx] = build_action(ordered, int(budget), h_name)
                mask[idx] = True

        if not any(mask):
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True

        return actions_by_index, mask






















