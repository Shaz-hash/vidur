# # (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .environment import VidurGameStats, VidurMCTSState
from .game_types import ControllerAction

if TYPE_CHECKING:
    from .config import GameVersion2Config


class GV2RuntimeOps:
    def __init__(self, env: object, cfg: "GameVersion2Config") -> None:
        self._e = env
        self._cfg = cfg

    def _v2_round(self, t: float) -> float:
        return round(float(t), int(self._cfg.timing.time_round_digits))

    def _v2_quantize_up(self, t: float) -> float:
        step = float(self._e._MINI_TICK_SEC)
        k = math.ceil((float(t) - self._e._EPS) / step)
        return self._v2_round(k * step)

    def _v2_quantize_down(self, t: float) -> float:
        step = float(self._e._MINI_TICK_SEC)
        k = math.floor((float(t) + self._e._EPS) / step)
        return self._v2_round(k * step)

    def _v2_meta_get(self, stats: VidurGameStats, key: int, default: float) -> float:
        return float(stats.decode_next_deadline_by_id.get(int(key), float(default)))

    def _v2_meta_set(self, stats: VidurGameStats, key: int, value: float) -> None:
        stats.decode_next_deadline_by_id[int(key)] = float(value)

    def _v2_init_clock(self, state: VidurMCTSState, sim_time: float) -> None:
        stats = state.stats
        if int(self._e._META_NEXT_ADV_TICK) not in stats.decode_next_deadline_by_id:
            self._v2_meta_set(stats, self._e._META_NEXT_ADV_TICK, self._v2_quantize_down(sim_time))
        if int(self._e._META_LAST_ADV_TICK) not in stats.decode_next_deadline_by_id:
            self._v2_meta_set(stats, self._e._META_LAST_ADV_TICK, -1.0)
        if int(self._e._META_SENT_WINDOW_START) not in stats.decode_next_deadline_by_id:
            self._v2_meta_set(stats, self._e._META_SENT_WINDOW_START, -1.0)
        if int(self._e._META_CTRL_IDLE_UNTIL) not in stats.decode_next_deadline_by_id:
            self._v2_meta_set(stats, self._e._META_CTRL_IDLE_UNTIL, -1.0)
        if int(self._e._META_MISSED_ADV_SOURCE) not in stats.decode_next_deadline_by_id:
            self._v2_meta_set(stats, self._e._META_MISSED_ADV_SOURCE, 0.0)


    def _v2_current_adv_tick(self, state: VidurMCTSState) -> float:
        sim_time = float(state.simulator._time)
        self._v2_init_clock(state, sim_time)
        tick = self._v2_meta_get(state.stats, self._e._META_NEXT_ADV_TICK, self._v2_quantize_down(sim_time))
        return self._v2_round(tick)

    def _v2_next_adv_tick(self, state: VidurMCTSState) -> float:
        sim_time = float(state.simulator._time)
        self._v2_init_clock(state, sim_time)
        return self._v2_round(
            self._v2_meta_get(state.stats, self._e._META_NEXT_ADV_TICK, self._v2_quantize_down(sim_time))
        )

    def _v2_has_pending_adv_tick(self, state: VidurMCTSState) -> bool:
        next_tick = self._v2_next_adv_tick(state)
        sim_time = float(state.simulator._time)
        return next_tick <= (sim_time + self._e._EPS)

    def _v2_sent_window_start(self, state: VidurMCTSState) -> Optional[int]:
        v = self._v2_meta_get(state.stats, self._e._META_SENT_WINDOW_START, -1.0)
        if v < 0:
            return None
        return int(math.floor(v + self._e._EPS))

    def _v2_set_sent_window_start(self, state: VidurMCTSState, window_start: Optional[int]) -> None:
        self._v2_meta_set(
            state.stats,
            self._e._META_SENT_WINDOW_START,
            -1.0 if window_start is None else float(window_start),
        )

    def _v2_controller_idle_until(self, state: VidurMCTSState) -> Optional[float]:
        v = self._v2_meta_get(state.stats, self._e._META_CTRL_IDLE_UNTIL, -1.0)
        if v < 0:
            return None
        return self._v2_round(v)

    def _v2_set_controller_idle_until(self, state: VidurMCTSState, idle_until: Optional[float]) -> None:
        self._v2_meta_set(
            state.stats,
            self._e._META_CTRL_IDLE_UNTIL,
            -1.0 if idle_until is None else float(idle_until),
        )

    def _v2_has_active_prefill(self, state: VidurMCTSState) -> bool:
        req_map = self._e._req_map(state.simulator)
        for rid in state.stats.active_request_ids:
            req = req_map.get(int(rid))
            if req is None or bool(getattr(req, "completed", False)):
                continue
            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            rem_pref = self._e._remaining_prefill(req)
            if rem_pref > 0 and (not prefill_done):
                return True
        return False

    def _v2_controller_noop_space(self) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        actions: List[Optional[ControllerAction]] = [None] * self._e._ctrl_action_space
        mask: List[bool] = [False] * self._e._ctrl_action_space
        actions[0] = ControllerAction(
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
        return actions, mask

    def _v2_decode_credit_balance_raw(self, state: VidurMCTSState) -> int:
        return int(state.stats.decode_tokens_counted.get(self._e._META_DECODE_CREDIT_BAL, 0))

    def _v2_decode_credit_balance(self, state: VidurMCTSState) -> int:
        # available budget for scheduling
        return max(0, self._v2_decode_credit_balance_raw(state))

    def _v2_set_decode_credit_balance(self, state: VidurMCTSState, value: int) -> None:
        # keep signed value (can be negative)
        state.stats.decode_tokens_counted[self._e._META_DECODE_CREDIT_BAL] = int(value)

    def _v2_add_decode_credit(self, state: VidurMCTSState, delta: int) -> None:
        if delta <= 0:
            return
        self._v2_set_decode_credit_balance(
            state,
            self._v2_decode_credit_balance_raw(state) + int(delta),
        )

    def _v2_consume_decode_credit(self, state: VidurMCTSState, need: int) -> int:
        if need <= 0:
            return 0
        raw = self._v2_decode_credit_balance_raw(state)
        avail = max(0, raw)
        take = min(avail, int(need))
        self._v2_set_decode_credit_balance(state, raw - take)
        return take

    def _v2_get_recent_launches(self, state: VidurMCTSState) -> List[Tuple[float, int, int]]:
        launches: List[Tuple[float, int, int]] = []
        for item in state.stats.recent_arrivals:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                launches.append((float(item[0]), int(item[1]), int(item[2])))
            elif isinstance(item, (int, float)):
                launches.append((float(item), 1, int(self._e._max_request_tokens_allowed())))
        return launches

    def _v2_set_recent_launches(self, state: VidurMCTSState, launches: List[Tuple[float, int, int]]) -> None:
        state.stats.recent_arrivals = [
            (self._v2_round(float(t)), int(c), int(p))
            for (t, c, p) in launches
            if int(c) > 0 and int(p) >= 0
        ]

    def _v2_prune_recent_launches(self, state: VidurMCTSState, anchor_time: float) -> List[Tuple[float, int, int]]:
        window = float(self._cfg.timing.launch_window_sec)
        lo = float(anchor_time) - window
        pruned = [x for x in self._v2_get_recent_launches(state) if float(x[0]) + self._e._EPS >= lo]
        self._v2_set_recent_launches(state, pruned)
        return pruned

    def _v2_window_usage(self, anchor_time: float, launches: List[Tuple[float, int, int]]) -> Tuple[int, int]:
        lo = float(anchor_time) - float(self._cfg.timing.launch_window_sec)
        hi = float(anchor_time) + self._e._EPS
        c = 0
        p = 0
        for ts, cnt, pref in launches:
            if ts + self._e._EPS < lo:
                continue
            if ts > hi:
                continue
            c += max(0, int(cnt))
            p += max(0, int(pref))
        return c, p

    def _v2_has_active_decode(self, state: VidurMCTSState) -> bool:
        return len(self._v2_collect_decode_ids(state)) > 0

    def _v2_collect_decode_ids(self, state: VidurMCTSState) -> List[int]:
        req_map = self._e._req_map(state.simulator)
        cap = int(self._cfg.request.max_decode_tokens_per_request)
        out: List[int] = []
        for rid in sorted(state.stats.active_request_ids):
            req = req_map.get(int(rid))
            if req is None or bool(getattr(req, "completed", False)):
                continue
            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            if not prefill_done:
                continue
            rem_dec = self._e._remaining_decode(req)
            if rem_dec <= 0:
                continue
            if int(getattr(req, "num_processed_decode_tokens", 0)) >= cap:
                continue
            out.append(int(rid))

        if self._cfg.credits.enforce_nonnegative_decode_credits:
            bal = self._v2_decode_credit_balance(state)
            if bal <= 0:
                return []
            return out[:bal]


        return out

    def _v2_enforce_decode_caps(self, state: VidurMCTSState) -> None:
        req_map = self._e._req_map(state.simulator)
        cap = int(self._cfg.request.max_decode_tokens_per_request)
        touched: List[int] = []
        for rid in list(state.stats.active_request_ids):
            req = req_map.get(int(rid))
            if req is None:
                continue
            done = int(getattr(req, "num_processed_decode_tokens", 0))
            if done < cap:
                continue
            req._num_decode_tokens = min(int(getattr(req, "_num_decode_tokens", req.num_decode_tokens)), cap)
            if done >= int(req._num_decode_tokens):
                try:
                    req._completed = True
                    req._completed_at = float(state.simulator._time)
                except Exception:
                    pass
            touched.append(int(rid))
        self._e._update_active_ids_for(state, touched)

    def _v2_finalize_decodes_to_credit_budget(self, state: VidurMCTSState) -> None:
        """
        Enforce request-level decode-credit invariant after stats update:
        keep at most `decode_credit_balance` active decode requests.
        If overflow exists, complete requests in priority:
        1) largest processed decode tokens first
        2) tie-break by largest request id first
        """
        if not self._cfg.credits.enforce_nonnegative_decode_credits:
            return

        # First apply hard per-request decode cap logic.
        self._v2_enforce_decode_caps(state)

        bal = max(0, int(self._v2_decode_credit_balance(state)))
        req_map = self._e._req_map(state.simulator)
        cap = int(self._cfg.request.max_decode_tokens_per_request)

        decode_active: List[Tuple[int, int]] = []  # (processed_decode_tokens, rid)
        for rid in list(state.stats.active_request_ids):
            rid = int(rid)
            req = req_map.get(rid)
            if req is None or bool(getattr(req, "completed", False)):
                continue

            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            if not prefill_done:
                continue

            done = int(getattr(req, "num_processed_decode_tokens", 0))
            decode_goal = min(int(getattr(req, "_num_decode_tokens", req.num_decode_tokens)), cap)

            # Still decode-active only.
            if done < decode_goal:
                decode_active.append((done, rid))

        overflow = len(decode_active) - bal
        if overflow <= 0:
            return

        # Complete largest decode first; tie: largest id first.
        decode_active.sort(key=lambda x: (x[0], x[1]), reverse=True)
        to_complete = [rid for _, rid in decode_active[:overflow]]

        stats = state.stats
        for rid in to_complete:
            req = req_map.get(rid)
            if req is not None:
                try:
                    # Snap decode target to current progress and complete.
                    req._num_decode_tokens = max(int(req.num_processed_decode_tokens), 0)
                    req._completed = True
                    req._completed_at = float(state.simulator._time)
                except Exception:
                    pass

            if rid not in stats.completed_request_ids:
                stats.completed_request_ids.add(rid)
                stats.requests_completed += 1

            stats.active_request_ids.discard(rid)
            stats.stopped_decode_request_ids.add(rid)
            stats.decode_next_deadline_by_id.pop(rid, None)
            stats.decode_tokens_counted.pop(rid, None)

        self._e._update_active_ids_for(state, to_complete)


    def _v2_eviction_rule_from_action(self, action: ControllerAction) -> str:
        if action.mapping is not None and len(action.mapping) >= 1:
            e_idx = int(action.mapping[0])
            rules = self._cfg.controller_action.eviction_rule_names
            if 0 <= e_idx < len(rules):
                return str(rules[e_idx])
        s = str(action.strategy or "")
        if s.startswith("GV2|"):
            return s.split("|", 1)[1]
        return "evict_none"

    def _v2_drop_request(self, state: VidurMCTSState, rid: int, *, by_controller: bool) -> None:
        del by_controller
        rid = int(rid)
        stats = state.stats
        if rid in stats.completed_request_ids:
            return

        # Remove previously accrued per-request lateness from objective bucket.
        prev_pref = float(stats.per_request_prefill_lateness.pop(rid, 0.0))
        prev_dec = float(stats.per_request_decode_lateness.pop(rid, 0.0))
        prev_total = max(0.0, prev_pref + prev_dec)
        if prev_total > 0.0:
            stats.slo_lateness_sum = max(0.0, float(stats.slo_lateness_sum) - prev_total)

        # If this request had already contributed +1 violation term, remove it.
        if rid in stats.violated_request_ids:
            stats.violated_request_ids.discard(rid)
            if stats.slo_violations > 0:
                stats.slo_violations -= 1

        # Finalize request in simulator
        req = self._e._req_map(state.simulator).get(rid)
        if req is not None:
            try:
                req._num_prefill_tokens = int(getattr(req, "num_processed_prefill_tokens", 0))
                req._is_prefill_complete = True
                req._num_decode_tokens = int(getattr(req, "num_processed_decode_tokens", 0))
                req._completed = True
                req._completed_at = float(state.simulator._time)
            except Exception:
                pass

        stats.active_request_ids.discard(rid)
        stats.completed_request_ids.add(rid)
        stats.dropped_request_ids.add(rid)
        stats.requests_completed += 1

        # Reclaim remaining per-request decode credit budget on drop:
        # reclaim = max(0, mint_per_req - counted_decode_for_rid)
        mint = int(self._cfg.credits.decode_credit_mint_per_prefill_complete)
        if rid in stats.decode_tokens_counted:
            counted_for_rid = max(0, int(stats.decode_tokens_counted.get(rid, 0)))
            reclaim = max(0, mint - counted_for_rid)
            if reclaim > 0:
                self._v2_set_decode_credit_balance(
                    state,
                    self._v2_decode_credit_balance_raw(state) - reclaim,
                )

        # Cleanup decode tracking
        stats.decode_next_deadline_by_id.pop(rid, None)
        stats.decode_tokens_counted.pop(rid, None)
        stats.prefill_lateness_finalized.discard(rid)

        # Dropped request contributes terminal cost only.
        stats.slo_lateness_sum += float(self._cfg.cost.drop_cost)



    def _v2_apply_controller_eviction(self, state: VidurMCTSState, action: ControllerAction) -> None:
        rule = self._v2_eviction_rule_from_action(action)
        if rule == "evict_none":
            return

        req_lookup = self._e._build_request_lookup(state.simulator, state=state)
        req_views = self._e._action_sampler._build_req_views(
            sim_time=float(state.simulator._time),
            request_lookup=req_lookup,
            per_request_prefill_lateness=dict(state.stats.per_request_prefill_lateness),
            per_request_decode_lateness=dict(state.stats.per_request_decode_lateness),
        )
        targets = self._e._action_sampler._eviction_targets(
            rule=rule,
            req_views=req_views,
            violated_request_ids=set(state.stats.violated_request_ids),
        )
        for rid in targets:
            self._v2_drop_request(state, int(rid), by_controller=True)
            
        if self._cfg.credits.enforce_nonnegative_decode_credits:
            self._v2_finalize_decodes_to_credit_budget(state)


    def _v2_run_one_decode_only_batch(self, state: VidurMCTSState, decode_ids: List[int]) -> bool:
        if not decode_ids:
            return False

        decode_alloc = {int(rid): 1 for rid in decode_ids}
        action = ControllerAction(
            token_budget=len(decode_alloc),
            selected_request_ids=sorted(decode_alloc.keys()),
            token_allocations=dict(decode_alloc),
            prefill_allocations={},
            decode_allocations=dict(decode_alloc),
            heuristic=None,
            strategy="GV2|decode_only_ff",
            mapping=None,
        )

        req_map = self._e._req_map(state.simulator)
        active_ids = state.stats.active_request_ids
        decode_credit_limit = (
            self._v2_decode_credit_balance(state)
            if self._cfg.credits.enforce_nonnegative_decode_credits
            else None
        )

        credit_before = self._v2_decode_credit_balance_raw(state)
        _, _, _, req_ids, _, num_tokens, pred_reqs = self._e._normalize_and_collect_batch(
            action,
            req_map=req_map,
            active_ids=active_ids,
            decode_credit_limit=decode_credit_limit,
        )
        if not pred_reqs:
            return False

        st = float(state.simulator._time)
        et_pred = state.simulator._execution_time_predictor.get_execution_time(pred_reqs, 0)
        et = st + float(et_pred.total_time)
        state.simulator._set_time(et)

        batch_exec = {
            "request_ids": list(req_ids),
            "num_tokens": list(num_tokens),
            "start_time": st,
            "end_time": et,
            "stage_total_time": float(et_pred.total_time),
            "stage_model_time": float(et_pred.model_time),
        }
        self._e._update_requests_and_stats(state, batch_exec=batch_exec)
        
        credit_after = self._v2_decode_credit_balance_raw(state)
        self._e._record_internal_event(
            state,
            phase="internal:decode_ff_batch",
            start_time=float(st),
            end_time=float(et),
            reason="decode_only_fast_forward",
            request_ids=[int(x) for x in req_ids],
            num_tokens=[int(x) for x in num_tokens],
            stage_total_time=float(et_pred.total_time),
            decode_credit_before=int(credit_before),
            decode_credit_after=int(credit_after),
        )
        # self._v2_enforce_decode_caps(state)
        return True

    def _v2_missed_adv_source(self, state: VidurMCTSState) -> int:
        v = self._v2_meta_get(state.stats, self._e._META_MISSED_ADV_SOURCE, 0.0)
        return int(round(v))

    def _v2_set_missed_adv_source(self, state: VidurMCTSState, source: int) -> None:
        self._v2_meta_set(state.stats, self._e._META_MISSED_ADV_SOURCE, float(int(source)))
        

    def _maybe_fast_forward_decode_only_to_next_adv_second(self, state: VidurMCTSState) -> None:
        max_loops = 10_000
        loops = 0
        while loops < max_loops:
            loops += 1

            if self._v2_has_pending_adv_tick(state):
                return
            if self._v2_has_active_prefill(state):
                return

            decode_ids = self._v2_collect_decode_ids(state)
            if not decode_ids:
                next_tick = self._v2_next_adv_tick(state)
                now = float(state.simulator._time)
                if now + self._e._EPS < next_tick:
                    old_t = float(now)
                    new_t = float(next_tick)
                    state.simulator._set_time(new_t)
                    self._e._record_internal_event(
                        state,
                        phase="internal:jump_to_adv_tick",
                        start_time=old_t,
                        end_time=new_t,
                        reason="no_decode_active_jump_to_tick",
                        request_ids=[],
                        num_tokens=[],
                        stage_total_time=max(0.0, new_t - old_t),
                        decode_credit_before=int(self._v2_decode_credit_balance_raw(state)),
                        decode_credit_after=int(self._v2_decode_credit_balance_raw(state)),
                    )

                return

            progressed = self._v2_run_one_decode_only_batch(state, decode_ids)
            if not progressed:
                return
