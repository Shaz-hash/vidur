from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from vidur.entities import Request

from ...game_types import AdversaryAction, AdversaryRequestSpec, ControllerAction
from .config import GameVersion2Config


@dataclass(frozen=True)
class _ReqView:
    rid: int
    completed: bool
    prefill_done: bool
    rem_prefill: int
    rem_decode: int
    decode_processed: int
    arrived_at: float
    prefill_slo: float
    prefill_deadline: float
    prefill_lateness: float
    total_lateness: float


def adversary_action_space_size(cfg: GameVersion2Config) -> int:
    """
    Fixed flattened adversary action space size.
    launch=0: one template-agnostic branch per stop-rule.
    launch>0: (max_launch_count * num_templates * num_stop_rules).
    """
    n_stop = len(cfg.adversary_action.stop_rule_names)
    n_templates = len(cfg.request.allowed_prefill_tokens)
    n_counts = int(cfg.adversary_action.max_launch_count_per_tick)  # excludes zero
    return n_stop + (n_counts * n_templates * n_stop)


def controller_action_space_size(cfg: GameVersion2Config) -> int:
    """
    Fixed flattened controller action space size.
    Cartesian product: eviction x budget x heuristic.
    """
    return (
        len(cfg.controller_action.eviction_rule_names)
        * len(cfg.controller_action.prefill_budget_options)
        * len(cfg.controller_action.ordering_heuristics)
    )


class GV2PlayerActionSampler:
    """
    Pure action generation + masking.
    No simulator mutation here.

    - Adversary: 3-head factorization flattened into fixed indexed list.
    - Controller: 3-head factorization flattened into fixed indexed list.
    """

    def __init__(self, cfg: GameVersion2Config) -> None:
        cfg.validate()
        self.cfg = cfg
        self._eps = float(cfg.timing.eps)

        self._stop_rules = tuple(cfg.adversary_action.stop_rule_names)
        self._templates = tuple(sorted(int(x) for x in cfg.request.allowed_prefill_tokens))
        self._max_launch = int(cfg.adversary_action.max_launch_count_per_tick)

        self._evict_rules = tuple(cfg.controller_action.eviction_rule_names)
        self._budgets = tuple(int(x) for x in cfg.controller_action.prefill_budget_options)
        self._heuristics = tuple(cfg.controller_action.ordering_heuristics)

        self._prefill_window_cap_tokens = int(
            cfg.request.target_prefill_tokens_per_request_avg_window
            * cfg.timing.max_requests_per_launch_window
        )

        # Heuristic canonical for budget=0 duplicate suppression.
        self._canonical_zero_budget_heuristic = self._heuristics[0] if self._heuristics else "SJF"

    # ----------------------------
    # Shared helpers
    # ----------------------------

    def _remaining_prefill(self, req: Request) -> int:
        return max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))

    def _remaining_decode(self, req: Request) -> int:
        return max(0, int(req.num_decode_tokens) - int(req.num_processed_decode_tokens))

    def _build_req_views(
        self,
        *,
        sim_time: float,
        request_lookup: Dict[int, Request],
        per_request_prefill_lateness: Optional[Dict[int, float]] = None,
        per_request_decode_lateness: Optional[Dict[int, float]] = None,
    ) -> Dict[int, _ReqView]:
        views: Dict[int, _ReqView] = {}
        pref_lat = per_request_prefill_lateness or {}
        dec_lat = per_request_decode_lateness or {}

        t = float(sim_time)
        for rid_raw, req in request_lookup.items():
            rid = int(rid_raw)
            completed = bool(getattr(req, "completed", False))
            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            rem_pref = self._remaining_prefill(req)
            rem_dec = self._remaining_decode(req)
            decode_processed = int(req.num_processed_decode_tokens)

            if completed:
                continue
            if (not prefill_done) and rem_pref <= 0:
                continue
            if prefill_done and rem_dec <= 0:
                continue

            arrived_at = float(getattr(req, "arrived_at", 0.0))
            prefill_slo = float(getattr(req, "prefill_slo_time", 0.0))
            prefill_deadline = arrived_at + prefill_slo
            prefill_late_now = max(0.0, t - prefill_deadline)

            # Prefer stats if available, fallback to instantaneous proxy.
            prefill_lateness = float(pref_lat.get(rid, prefill_late_now))
            decode_lateness = float(dec_lat.get(rid, 0.0))
            total_lateness = max(0.0, prefill_lateness) + max(0.0, decode_lateness)

            views[rid] = _ReqView(
                rid=rid,
                completed=False,
                prefill_done=prefill_done,
                rem_prefill=rem_pref,
                rem_decode=rem_dec,
                decode_processed=decode_processed,
                arrived_at=arrived_at,
                prefill_slo=prefill_slo,
                prefill_deadline=prefill_deadline,
                prefill_lateness=prefill_lateness,
                total_lateness=total_lateness,
            )
        return views

    def _window_usage(
        self,
        *,
        anchor_time: float,
        recent_launches: Sequence[object],
    ) -> Tuple[int, int]:
        """
        Returns (request_count_in_window, prefill_tokens_in_window)
        from entries in (anchor_time - launch_window_sec, anchor_time].
        Accepts entry shapes:
          - tuple/list: (timestamp, count, prefill_tokens)
          - dict: {"timestamp":..., "count":..., "prefill_tokens":...}
        """
        w = float(self.cfg.timing.launch_window_sec)
        lo = float(anchor_time) - w
        hi = float(anchor_time) + self._eps

        total_count = 0
        total_prefill = 0

        for item in recent_launches:
            ts = None
            cnt = 0
            ptk = 0

            if isinstance(item, (tuple, list)) and len(item) >= 3:
                ts = float(item[0])
                cnt = int(item[1])
                ptk = int(item[2])
            elif isinstance(item, dict):
                if "timestamp" in item:
                    ts = float(item["timestamp"])
                elif "time" in item:
                    ts = float(item["time"])
                cnt = int(item.get("count", item.get("requests", 0)))
                ptk = int(item.get("prefill_tokens", item.get("tokens", 0)))

            if ts is None:
                continue
            if (ts + self._eps) < lo:
                continue
            if ts > hi:
                continue

            total_count += max(0, cnt)
            total_prefill += max(0, ptk)

        return total_count, total_prefill

    # ----------------------------
    # Adversary sampling
    # ----------------------------
    # TODO : Can be improved via O(1) via state handling
    def _decode_active_ids(self, req_views: Dict[int, _ReqView]) -> List[int]:
        return sorted(
            rid for rid, rv in req_views.items()
            if rv.prefill_done and rv.rem_decode > 0
        )
    # TODO: Can be improved via single loop
    def _stop_ids_for_rule(
        self,
        *,
        rule: str,
        req_views: Dict[int, _ReqView],
        forbidden_stop_ids: Optional[Set[int]] = None,
    ) -> List[int]:
        forbidden = set(int(x) for x in (forbidden_stop_ids or set()))
        decode_ids = [rid for rid in self._decode_active_ids(req_views) if rid not in forbidden]
        if not decode_ids:
            return []

        if rule == "stop_none":
            return []

        if rule == "stop_longest_decode":
            best = max(decode_ids, key=lambda rid: (req_views[rid].decode_processed, -rid))
            return [int(best)]

        if rule == "stop_shortest_decode":
            best = min(decode_ids, key=lambda rid: (req_views[rid].decode_processed, rid))
            return [int(best)]

        if rule == "stop_all_decodes_over_512":
            return sorted([rid for rid in decode_ids if req_views[rid].decode_processed > 512])

        if rule == "stop_all_decodes_over_216":
            return sorted([rid for rid in decode_ids if req_views[rid].decode_processed > 216])

        return []


    def sample_adversary_actions(
        self,
        *,
        sim_time: float,
        decision_tick: float,
        request_lookup: Dict[int, Request],
        recent_launches: Sequence[object],
        prefill_slo_lookup_fn: Callable[[int], float],
        decode_slo_time: float,
        per_request_prefill_lateness: Optional[Dict[int, float]] = None,
        per_request_decode_lateness: Optional[Dict[int, float]] = None,
        forbidden_stop_ids: Optional[Set[int]] = None, # --> to consider controller selectected ids if controller cause adv to miss deadline
    ) -> Tuple[List[Optional[AdversaryAction]], List[bool]]:
        """
        Returns fixed-size flattened action list + mask.
        Flatten order:
          launch=0, prefill_template=ignored, stop_rule over all stop rules
          then for launch in 1..max_launch:
            for prefill_tokens in templates:
              for stop_rule in stop_rules
        """
        req_views = self._build_req_views(
            sim_time=sim_time,
            request_lookup=request_lookup,
            per_request_prefill_lateness=per_request_prefill_lateness,
            per_request_decode_lateness=per_request_decode_lateness,
        )
        stop_ids_by_rule = {
            r: self._stop_ids_for_rule(
                rule=r,
                req_views=req_views,
                forbidden_stop_ids=forbidden_stop_ids,
            )
            for r in self._stop_rules
        }

        n = adversary_action_space_size(self.cfg)
        actions_by_index: List[Optional[AdversaryAction]] = [None] * n
        mask: List[bool] = [False] * n

        # If we haven't reached decision tick, expose only a strict no-op.
        strict_pre_tick = float(sim_time) + self._eps < float(decision_tick)

        used_count, used_prefill = self._window_usage(
            anchor_time=float(decision_tick),
            recent_launches=recent_launches,
        )
        req_cap = int(self.cfg.timing.max_requests_per_launch_window)
        prefill_cap = int(self._prefill_window_cap_tokens)
        decode_tokens_per_new_req = int(self.cfg.request.max_decode_tokens_per_request)

        idx = 0

        # launch_count = 0 branch (template ignored)
        for stop_rule in self._stop_rules:
            stop_ids = stop_ids_by_rule.get(stop_rule, [])
            valid = True

            if strict_pre_tick:
                valid = (stop_rule == "stop_none")

            # Optional duplicate suppression: stop rule actions requiring decode ids
            # are invalid when there are no eligible decodes.
            if stop_rule != "stop_none" and not stop_ids:
                valid = False

            actions_by_index[idx] = AdversaryAction(requests=[], stop_decode_ids=list(stop_ids))
            mask[idx] = bool(valid)
            idx += 1

        # launch_count >= 1 branches
        for launch_count in range(1, self._max_launch + 1):
            for prefill_tokens in self._templates:
                new_prefill_total = int(launch_count) * int(prefill_tokens)

                launch_valid = True
                if strict_pre_tick:
                    launch_valid = False
                if used_count + launch_count > req_cap:
                    launch_valid = False
                if used_prefill + new_prefill_total > prefill_cap:
                    launch_valid = False

                for stop_rule in self._stop_rules:
                    stop_ids = stop_ids_by_rule.get(stop_rule, [])

                    valid = launch_valid
                    if stop_rule != "stop_none" and not stop_ids:
                        # keep rule semantics strict
                        valid = False

                    reqs: List[AdversaryRequestSpec] = []
                    if valid:
                        template_prefill_slo = float(prefill_slo_lookup_fn(int(prefill_tokens)))
                        reqs = [
                            AdversaryRequestSpec(
                                prefill_tokens=int(prefill_tokens),
                                decode_tokens=int(decode_tokens_per_new_req),
                                prefill_slo=template_prefill_slo,
                                decode_slo=float(decode_slo_time),
                            )
                            for _ in range(int(launch_count))
                        ]

                    actions_by_index[idx] = AdversaryAction(
                        requests=reqs if valid else [],
                        stop_decode_ids=list(stop_ids) if valid or stop_rule != "stop_none" else [],
                    )
                    mask[idx] = bool(valid)
                    idx += 1

        assert idx == n
        return actions_by_index, mask

    # ----------------------------
    # Controller sampling
    # ----------------------------
    # TODO: Can be computed via single loop
    def _eviction_targets(
        self,
        *,
        rule: str,
        req_views: Dict[int, _ReqView],
        violated_request_ids: Optional[Set[int]] = None,
    ) -> List[int]:
        violated = violated_request_ids or set()

        prefill_ids = [rid for rid, rv in req_views.items() if (not rv.prefill_done and rv.rem_prefill > 0)]
        decode_ids = [rid for rid, rv in req_views.items() if (rv.prefill_done and rv.rem_decode > 0)]

        if rule == "evict_none":
            return []

        if rule == "evict_largest_prefill":
            if not prefill_ids:
                return []
            rid = max(prefill_ids, key=lambda x: (req_views[x].rem_prefill, -x))
            return [int(rid)]

        if rule == "evict_earliest_prefill_deadline":
            if not prefill_ids:
                return []
            rid = min(prefill_ids, key=lambda x: (req_views[x].prefill_deadline, x))
            return [int(rid)]

        if rule == "evict_prefill_missed_deadline":
            out = [rid for rid in prefill_ids if req_views[rid].prefill_lateness > self._eps]
            return sorted(out)

        if rule == "evict_prefill_lateness_over_0p5":
            out = [rid for rid in prefill_ids if req_views[rid].prefill_lateness > 0.5]
            return sorted(out)

        if rule == "evict_longest_decode":
            if not decode_ids:
                return []
            # largest processed decode; tie -> smallest rid
            rid = max(decode_ids, key=lambda x: (req_views[x].decode_processed, -x))
            return [int(rid)]

        if rule == "evict_decode_lateness_over_0p5":
            out = [rid for rid in decode_ids if req_views[rid].total_lateness > 0.5]
            return sorted(out)

        if rule == "evict_prefill_highest_lateness":
            if not prefill_ids:
                return []
            rid = max(prefill_ids, key=lambda x: (req_views[x].prefill_lateness, -x))
            return [int(rid)] if req_views[rid].prefill_lateness > self._eps else []

        if rule == "evict_decode_highest_lateness":
            if not decode_ids:
                return []
            rid = max(decode_ids, key=lambda x: (req_views[x].total_lateness, -x))
            return [int(rid)] if req_views[rid].total_lateness > self._eps else []

        return []

    # TODO : The speed can be improved in ordering because for the eviction rules that target decode tokens, the prefill ordering can be done on the same remaining prefill set. Similarly, eviction rules that target prefill tokens can share the same decode ordering. Currently, this is computed separately for each branch which leads to some redundant computation.
    def sample_controller_actions(
        self,
        *,
        sim_time: float,
        request_lookup: Dict[int, Request],
        prefill_eta_lookup_fn: Callable[[int], float],
        decode_next_deadline_by_id: Optional[Dict[int, float]] = None,  # kept for forward compatibility
        per_request_prefill_lateness: Optional[Dict[int, float]] = None,
        per_request_decode_lateness: Optional[Dict[int, float]] = None,
        violated_request_ids: Optional[Set[int]] = None,
        decode_credit_balance: Optional[int] = None,
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        del decode_next_deadline_by_id  # currently not required for rules below

        req_views = self._build_req_views(
            sim_time=sim_time,
            request_lookup=request_lookup,
            per_request_prefill_lateness=per_request_prefill_lateness,
            per_request_decode_lateness=per_request_decode_lateness,
        )

        n = controller_action_space_size(self.cfg)
        actions_by_index: List[Optional[ControllerAction]] = [None] * n
        mask: List[bool] = [False] * n

        # Fast no-request fallback.
        if not req_views:
            # keep fixed-space contract: mark only first action as valid no-op
            if n > 0:
                actions_by_index[0] = ControllerAction(
                    token_budget=0,
                    selected_request_ids=None,
                    token_allocations={},
                    prefill_allocations={},
                    decode_allocations={},
                    heuristic=None,
                    strategy="GV2|evict_none",
                    mapping=(0, 0, 0),
                )
                mask[0] = True
            return actions_by_index, mask

        # Precompute per-eviction branch.
        branch_cache: Dict[str, Dict[str, object]] = {}

        for ev_rule in self._evict_rules:
            evict_ids = set(self._eviction_targets(
                rule=ev_rule,
                req_views=req_views,
                violated_request_ids=violated_request_ids,
            ))

            # Strict masking: non-none rule that matches nothing is invalid branch.
            branch_has_effect = (len(evict_ids) > 0)
            branch_valid = True
            if ev_rule != "evict_none" and not branch_has_effect and self.cfg.adversary_action.strict_masking:
                branch_valid = False

            rem_prefill_ids: List[int] = []
            rem_decode_ids: List[int] = []
            rem_pref_by_id: Dict[int, int] = {}
            prefill_deadline_by_id: Dict[int, float] = {}
            arrived_by_id: Dict[int, float] = {}
            prefill_slo_by_id: Dict[int, float] = {}

            for rid, rv in req_views.items():
                if rid in evict_ids:
                    continue
                if not rv.prefill_done and rv.rem_prefill > 0:
                    rem_prefill_ids.append(rid)
                    rem_pref_by_id[rid] = rv.rem_prefill
                    prefill_deadline_by_id[rid] = rv.prefill_deadline
                    arrived_by_id[rid] = rv.arrived_at
                    prefill_slo_by_id[rid] = rv.prefill_slo
                elif rv.prefill_done and rv.rem_decode > 0:
                    rem_decode_ids.append(rid)

            rem_prefill_ids.sort()
            rem_decode_ids.sort()

            total_prefill = sum(rem_pref_by_id.values())

            def order_prefill(heur: str) -> List[int]:
                if heur == "SJF":
                    return sorted(rem_prefill_ids, key=lambda x: (rem_pref_by_id[x], x))
                if heur == "EDF":
                    return sorted(rem_prefill_ids, key=lambda x: (prefill_deadline_by_id[x], x))
                if heur == "LST":
                    def slack(rid: int) -> float:
                        remaining_slo = prefill_slo_by_id[rid] - max(0.0, float(sim_time) - arrived_by_id[rid])
                        est = float(prefill_eta_lookup_fn(int(rem_pref_by_id[rid])))
                        return remaining_slo - est
                    return sorted(rem_prefill_ids, key=lambda x: (slack(x), x))
                if heur == "LJF":
                    return sorted(rem_prefill_ids, key=lambda x: (-rem_pref_by_id[x], x))
                return list(rem_prefill_ids)

            ordered_by_heur = {h: order_prefill(h) for h in self._heuristics}

            branch_cache[ev_rule] = {
                "valid": branch_valid,
                "evict_ids": tuple(sorted(evict_ids)),
                "decode_ids": tuple(rem_decode_ids),
                "total_prefill": int(total_prefill),
                "rem_pref_by_id": rem_pref_by_id,
                "ordered_by_heur": ordered_by_heur,
            }

        idx = 0
        for e_idx, ev_rule in enumerate(self._evict_rules):
            b = branch_cache[ev_rule]
            branch_valid = bool(b["valid"])
            decode_ids = list(b["decode_ids"])  # type: ignore[index]
            total_prefill = int(b["total_prefill"])  # type: ignore[index]
            rem_pref_by_id = b["rem_pref_by_id"]  # type: ignore[index]
            ordered_by_heur = b["ordered_by_heur"]  # type: ignore[index]

            for b_idx, budget in enumerate(self._budgets):
                for h_idx, heur in enumerate(self._heuristics):
                    valid = branch_valid

                    # Suppress duplicated semantics for budget==0 across heuristics.
                    if budget == 0 and heur != self._canonical_zero_budget_heuristic:
                        valid = False

                    if budget < 0:
                        valid = False
                    if budget > total_prefill:
                        valid = False
                    if total_prefill == 0 and budget > 0:
                        valid = False

                    # decode_alloc = {int(rid): 1 for rid in decode_ids}
                    max_decode = len(decode_ids) if decode_credit_balance is None else max(0, int(decode_credit_balance))
                    decode_ids_limited = decode_ids[:max_decode]
                    decode_alloc = {int(rid): 1 for rid in decode_ids_limited}

                    prefill_alloc: Dict[int, int] = {}

                    if valid and budget > 0:
                        remaining = int(budget)
                        ordered_pref = list(ordered_by_heur.get(heur, []))
                        for rid in ordered_pref:
                            if remaining <= 0:
                                break
                            cap = int(rem_pref_by_id[rid])
                            if cap <= 0:
                                continue
                            alloc = cap if cap < remaining else remaining
                            prefill_alloc[int(rid)] = int(alloc)
                            remaining -= int(alloc)

                    token_alloc = dict(decode_alloc)
                    token_alloc.update(prefill_alloc)

                    # If branch is valid but action does nothing, keep only canonical zero action.
                    if valid and not token_alloc:
                        valid = (budget == 0 and heur == self._canonical_zero_budget_heuristic and ev_rule == "evict_none")

                    action = ControllerAction(
                        token_budget=int(sum(token_alloc.values())),
                        selected_request_ids=(sorted(token_alloc.keys()) if token_alloc else None),
                        token_allocations=token_alloc,
                        prefill_allocations=prefill_alloc,
                        decode_allocations=decode_alloc,
                        heuristic=(heur if budget > 0 else None),
                        strategy=f"GV2|{ev_rule}",
                        mapping=(int(e_idx), int(b_idx), int(h_idx)),
                    )

                    actions_by_index[idx] = action
                    mask[idx] = bool(valid)
                    idx += 1

        assert idx == n
        return actions_by_index, mask
