# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import random
import json
from typing import Any, Optional, Tuple, Callable
import torch

from ....environment import VidurMCTSState
from ....game_types import ControllerAction, AdversaryAction
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger, DNNMCTSRootSummaryLogger


## TODO: Use this class later for the verification of a given trace
def _mask_to_list(mask) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


class HistoryRootGenerator:
    """
    Builds a "history root" state by advancing from a given state:
    - forced single-action chains are applied but do NOT count toward nontrivial_hops
    - branching steps (valid_actions > 1) pick a random valid action and DO count
    - returns a (state, player_to_act, depth) positioned at a branching (or terminal) node

    Optional: logs each applied step to the MCTS iteration logger with phase="history-...".
    """

    def __init__(
        self,
        *,
        env: Any,
        iter_logger: Optional[DNNMCTSIterationLogger] = None,
        root_logger: Optional[DNNMCTSRootSummaryLogger] = None,
        iter_logger_native: Optional[DNNMCTSIterationLogger] = None,
        root_logger_native: Optional[DNNMCTSRootSummaryLogger] = None,
    ) -> None:
        self.env = env
        self.iter_logger = iter_logger
        self.root_logger = root_logger
        self.iter_logger_native = iter_logger_native
        self.root_logger_native = root_logger_native



    def _actions_valid(self, state: VidurMCTSState, player: str) -> tuple[list[Optional[object]], list[int], list[bool]]:
        if player == "controller":
            actions_by_index, mask = self.env.sample_controller_actions(state)
        else:
            actions_by_index, mask = self.env.sample_adversary_actions(state)

        mask_list = _mask_to_list(mask)
        valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
        return actions_by_index, valid, mask_list

    def _phase_label(self, player: str, action: object, n_valid: int) -> str:
        if int(n_valid) == 1:
            return "forced_step:single-child"
        if player == "adversary":
            reqs = getattr(action, "requests", None) or []
            stop_ids = getattr(action, "stop_decode_ids", None) or []
            if (not reqs) and (not stop_ids):
                return "trivial-multiple-child"
        return "multiple-child"

    def _apply(self, state: VidurMCTSState, player: str, action: object) -> tuple[VidurMCTSState, str]:
        if player == "adversary":
            state = self.env.apply_adversary_action_only(state, action, inplace=True)
            return state, "controller"
        state = self.env.apply_controller_action_only(state, action, inplace=True)
        return state, "adversary"



    def _history_action_json(self, action: object, phase: str) -> str:
        if isinstance(action, ControllerAction):
            payload = {
                "type": "controller",
                "token_budget": int(action.token_budget),
                "selected_request_ids": [int(x) for x in (action.selected_request_ids or [])],
                "token_allocations": {str(int(k)): int(v) for k, v in (action.token_allocations or {}).items()},
                "prefill_allocations": {str(int(k)): int(v) for k, v in (action.prefill_allocations or {}).items()},
                "decode_allocations": {str(int(k)): int(v) for k, v in (action.decode_allocations or {}).items()},
                "heuristic": action.heuristic,
                "strategy": action.strategy,
                "history_phase": phase,
            }
            return json.dumps(payload, ensure_ascii=False)

        if isinstance(action, AdversaryAction):
            payload = {
                "type": "adversary",
                "requests": [
                    {
                        "prefill_tokens": int(r.prefill_tokens),
                        "decode_tokens": int(r.decode_tokens),
                        "prefill_slo": float(r.prefill_slo),
                        "decode_slo": float(r.decode_slo),
                    }
                    for r in (action.requests or [])
                ],
                "stop_decode_ids": [int(x) for x in (action.stop_decode_ids or [])],
                "history_phase": phase,
            }
            return json.dumps(payload, ensure_ascii=False)

        return json.dumps({"type": "unknown", "repr": repr(action), "history_phase": phase}, ensure_ascii=False)



    def _log_step(
        self,
        *,
        game_id: int,
        root_id: int,
        root_depth: int,
        node_depth: int,
        node_id: int,
        parent_node_id: int | None,
        player_acted: str,
        next_player: str,
        action_index: int,
        action: object,
        n_valid: int,
        phase: str,
        state_after: VidurMCTSState,
    ) -> None:
        violations, lateness_sum = self.env.evaluate_objective(state_after)
        objective_cost = float(violations) + float(lateness_sum)
        snap = self.env.describe_state(state_after)

        sim_time = float(getattr(state_after.simulator, "_time", 0.0))
        try:
            if str(player_acted).strip().lower() == "adversary":
                decision_state_time = float(self.env._v2_current_adv_tick(state_after))
            else:
                decision_state_time = sim_time
        except Exception:
            decision_state_time = sim_time

        last_adv_raw = snap.get("last_adv_tick", "")
        try:
            state_last_adv_tick = None if last_adv_raw in ("", None) else float(last_adv_raw)
        except Exception:
            state_last_adv_tick = None

        decode_counted: dict[int, int] = {}
        for k, v in (snap.get("decode_tokens_counted_by_id") or {}).items():
            try:
                ik = int(k)
                if ik >= 0:
                    decode_counted[ik] = int(v)
            except Exception:
                pass

        active_ids = [int(x) for x in (snap.get("active_request_ids") or [])]
        completed_ids = [int(x) for x in (snap.get("completed_request_ids") or [])]

        adv_deadlines_json = "{}"
        if player_acted == "adversary":
            req_specs = getattr(action, "requests", None) or []
            if req_specs:
                lookup = self.env._build_request_lookup(state_after.simulator)
                if lookup:
                    k = len(req_specs)
                    new_ids = sorted(lookup.keys())[-k:]  # assumes newest have largest ids
                    out = {}
                    for rid in new_ids:
                        req = lookup.get(rid)
                        if req is None:
                            continue
                        queued_at = float(getattr(req, "queued_at", getattr(req, "arrived_at", 0.0)) or 0.0)
                        slo = float(getattr(req, "prefill_slo_time", 0.0) or 0.0)
                        out[int(rid)] = queued_at + slo
                    adv_deadlines_json = json.dumps(out)

        for _ilog in (self.iter_logger, self.iter_logger_native):
            if _ilog is None:
                continue
            _ilog.log_expand(
                game_id=int(game_id),
                root_id=int(root_id),
                sim_iteration=-1,  # history row
                root_depth=int(root_depth),
                root_node_id=0,
                root_player=str(player_acted),
                node_depth=int(node_depth),
                parent_node_id=None if parent_node_id is None else int(parent_node_id),
                node_id=int(node_id),
                player_to_act=str(next_player),
                player_acted_to_create_this_node=str(player_acted),
                action_index=int(action_index),
                action_repr=repr(action),
                prior=1.0,
                reward=0.0,
                nn_called=False,
                num_valid_actions=int(n_valid),
                unique_actions=int(n_valid),
                nn_value_controller=None,
                objective_cost=float(objective_cost),
                adversary_prefill_deadlines_by_id_json=adv_deadlines_json,
                state_snapshot=snap,
                phase=f"history-{phase}",
            )

        for _rlog in (self.root_logger, self.root_logger_native):
            if _rlog is None:
                continue
            _rlog.log_root(
                game_id=int(game_id),
                root_id=int(root_id),
                root_depth=int(root_depth),
                root_node_id=int(node_id),
                root_player=str(player_acted),
                num_simulations=0,  # marks history rows
                model_root_value_controller=0.0,
                model_root_prior=[],
                normalized_root_prior=[],
                valid_action_mask=[],
                mcts_root_value_controller=0.0,
                mcts_root_prior=[],
                best_action_index=int(action_index),
                best_action_repr=repr(action),
                best_action_json=self._history_action_json(action, phase),
                phase=f"history-{phase}",
                sim_time=float(sim_time),
                decision_state_time=float(decision_state_time),
                state_pending_adv_tick=bool(snap.get("pending_adv_tick", False)),
                state_last_adv_tick=state_last_adv_tick,
                state_active_ids=active_ids,
                state_completed_request_ids=completed_ids,
                state_decode_credit_balance=int(snap.get("decode_credit_balance", 0)),
                state_decode_tokens_counted_by_id=decode_counted,
                slo_violations=int(violations),
                total_lateness=float(lateness_sum),
                total_cost=float(objective_cost),
            )

    def advance_to_branching_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        game_id: int,
        root_id: int,
        log_node_id: int,
        log_parent_id: int | None,
        max_hops: int = 10000,
        log_steps: bool = False,
        step_callback: Callable[[dict], None] | None = None, ## For arena logging when generating its history
    ) -> tuple[VidurMCTSState, str, int, int, int | None, int]:
        """
        Apply forced single-action steps until valid_actions != 1.
        Returns (state, player_to_act, depth, next_log_node_id, last_log_parent_id, forced_steps_applied).
        """
        forced_steps_applied = 0
        root_depth = int(depth)

        for _ in range(int(max_hops)):
            actions_by_index, valid, _mask = self._actions_valid(state, player)
            n_valid = len(valid)

            if n_valid != 1:
                return state, player, int(depth), int(log_node_id), log_parent_id, int(forced_steps_applied)

            idx = int(valid[0])
            action = actions_by_index[idx]
            assert action is not None

            sim_before = float(getattr(state.simulator, "_time", 0.0))
            acted = player
            state, next_player = self._apply(state, player, action)
            root_depth = int(depth)
            depth = int(depth) + 1
            forced_steps_applied += 1

            # For Arena Eval Logging :
            if step_callback is not None:
                viol_h, late_h = self.env.evaluate_objective(state)
                step_callback(
                    {
                        "phase": self._phase_label(acted, action, n_valid),  # use len(valid) in nontrivial block
                        "player_acted": str(acted),
                        "player_to_act_next": str(next_player),
                        "action_repr": repr(action),
                        "sim_time_before": float(sim_before),
                        "sim_time_after": float(state.simulator._time),
                        "depth_before": int(root_depth),
                        "depth_after": int(depth),
                        "slo_violations": int(viol_h),
                        "total_lateness": float(late_h),
                        "total_cost": float(viol_h) + float(late_h),
                    }
                )

            # For MCTS_ITER LOGGING :
            if log_steps:
                self._log_step(
                    game_id=game_id,
                    root_id=root_id,
                    root_depth=root_depth,
                    node_depth=int(depth),
                    node_id=int(log_node_id),
                    parent_node_id=log_parent_id,
                    player_acted=acted,
                    next_player=next_player,
                    action_index=idx,
                    action=action,
                    n_valid=n_valid,
                    phase=self._phase_label(acted, action, n_valid),
                    state_after=state,
                )
                log_parent_id = int(log_node_id)
                log_node_id += 1

            player = next_player

        return state, player, int(depth), int(log_node_id), log_parent_id, int(forced_steps_applied)

    def generate_history_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        nontrivial_hops: int,
        game_id: int,
        root_id_for_logs: int,
        seed: int = 0,
        log_history: bool = True,
        max_total_steps: int = 20000,
        log_node_id_start: int,
        log_parent_id_start: int | None = None,
        step_callback: Callable[[dict], None] | None = None, ## For arena logging when generating its history
    ) -> tuple[VidurMCTSState, str, int, int, int | None]:
        """
        Take `nontrivial_hops` branching decisions (valid_actions > 1),
        skipping forced chains in between.
        """
        target = int(nontrivial_hops)
        max_total_steps_i = max(1, int(max_total_steps))

        log_node_id = int(log_node_id_start)
        log_parent_id: int | None = None if log_parent_id_start is None else int(log_parent_id_start)

        if target <= 0:
            return state, player, int(depth), int(log_node_id), log_parent_id

        rng = random.Random(int(seed))
        total_steps = 0
        done = 0

        while done < target:
            remaining = int(max_total_steps_i) - int(total_steps)
            if remaining <= 0:
                print(
                    "[HistoryRootGenerator] max_total_steps reached; "
                    "returning current state without additional nontrivial hops"
                )
                return state, player, int(depth), int(log_node_id), log_parent_id

            state, player, depth, log_node_id, log_parent_id, forced_steps = self.advance_to_branching_root(
                state,
                player,
                depth,
                game_id=game_id,
                root_id=root_id_for_logs,
                max_hops=min(2000, int(remaining)),
                log_node_id=log_node_id,
                log_parent_id=log_parent_id,
                log_steps=log_history,
                step_callback=step_callback,
            )
            total_steps += int(forced_steps)

            remaining = int(max_total_steps_i) - int(total_steps)
            if remaining <= 0:
                print(
                    "[HistoryRootGenerator] max_total_steps reached; "
                    "returning current state without additional nontrivial hops"
                )
                return state, player, int(depth), int(log_node_id), log_parent_id

            actions_by_index, valid, _mask = self._actions_valid(state, player)
            if not valid:
                return state, player, int(depth), int(log_node_id), log_parent_id
            if len(valid) == 1:
                print("Error in history root creation: expected branching node but got single child")
                continue

            idx = int(rng.choice(valid))
            action = actions_by_index[idx]
            assert action is not None

            acted = player
            sim_before = float(getattr(state.simulator, "_time", 0.0))
            state, next_player = self._apply(state, player, action)

            root_depth = int(depth)
            depth = int(depth) + 1
            total_steps += 1

            # For Arena Eval Logging :
            if step_callback is not None:
                viol_h, late_h = self.env.evaluate_objective(state)
                step_callback(
                    {
                        "phase": self._phase_label(acted, action, len(valid)), # use len(valid) in nontrivial block
                        "player_acted": str(acted),
                        "player_to_act_next": str(next_player),
                        "action_repr": repr(action),
                        "sim_time_before": float(sim_before),
                        "sim_time_after": float(state.simulator._time),
                        "depth_before": int(root_depth),
                        "depth_after": int(depth),
                        "slo_violations": int(viol_h),
                        "total_lateness": float(late_h),
                        "total_cost": float(viol_h) + float(late_h),
                    }
                )

            # For MCTS_ITER LOGGING :
            if log_history:
                self._log_step(
                    game_id=game_id,
                    root_id=root_id_for_logs,
                    root_depth=root_depth,
                    node_depth=int(depth),
                    node_id=int(log_node_id),
                    parent_node_id=log_parent_id,
                    player_acted=acted,
                    next_player=next_player,
                    action_index=idx,
                    action=action,
                    n_valid=len(valid),
                    phase=self._phase_label(acted, action, len(valid)),
                    state_after=state,
                )
                log_parent_id = int(log_node_id)
                log_node_id += 1

            player = next_player
            done += 1

        remaining = int(max_total_steps_i) - int(total_steps)
        if remaining > 0:
            state, player, depth, log_node_id, log_parent_id, _ = self.advance_to_branching_root(
                state,
                player,
                depth,
                game_id=game_id,
                root_id=root_id_for_logs,
                max_hops=min(2000, int(remaining)),
                log_node_id=log_node_id,
                log_parent_id=log_parent_id,
                log_steps=log_history,
                step_callback=step_callback,
            )

        return state, player, int(depth), int(log_node_id), log_parent_id
