# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

from dataclasses import dataclass, field
import random
import json
from typing import Any, Optional, Tuple, Callable, Sequence, Iterator
import torch

from ....environment import VidurMCTSState
from ....game_types import ControllerAction, AdversaryAction
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger, DNNMCTSRootSummaryLogger


## TODO: Use this class later for the verification of a given trace
def _mask_to_list(mask) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


@dataclass
class _HistoryFrontierNode:
    state_snapshot: Any
    state_stats: Any
    player: str
    depth: int
    history_hops: int
    log_node_id: int | None = None
    pre_controller_snapshot: Any | None = None
    pre_controller_stats: Any | None = None
    untried_action_indices: list[int] = field(default_factory=list)


@dataclass
class _HistoryRootBatchSession:
    start_state: VidurMCTSState
    start_player: str
    start_depth: int
    min_history_hops: int
    max_history_hops: int
    max_total_steps: int
    max_children_per_expand: int | None
    target_roots: int
    next_root_id: int
    rng: random.Random
    anchor_budget: int
    max_iterations: int
    game_id: int = 0
    trace_root_id: int = 0
    next_log_node_id: int = 0
    history_trace_logger: Any | None = None
    emitted_roots: int = 0
    anchors_built: int = 0
    iterations: int = 0
    exhausted: bool = False
    seen_signatures: set[tuple[Any, ...]] = field(default_factory=set)
    active_path: list[_HistoryFrontierNode] = field(default_factory=list)
    shared_seen_signatures: Any | None = None
    shared_seen_lock: Any | None = None
    allow_duplicate_fallback: bool = True


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
        self._scratch_state: VidurMCTSState | None = None



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
        iter_loggers: Sequence[Any] | None = None,
        root_loggers: Sequence[Any] | None = None,
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

        active_iter_loggers = (
            (self.iter_logger, self.iter_logger_native)
            if iter_loggers is None
            else tuple(iter_loggers)
        )
        active_root_loggers = (
            (self.root_logger, self.root_logger_native)
            if root_loggers is None
            else tuple(root_loggers)
        )

        for _ilog in active_iter_loggers:
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

        for _rlog in active_root_loggers:
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

    def _snapshot_pair(self, state: VidurMCTSState) -> tuple[Any, Any]:
        sim = state.simulator
        if hasattr(sim, "snapshot_state_fast"):
            snap = sim.snapshot_state_fast()
        else:
            snap = sim.snapshot_state()
        snap = self._prune_completed_requests_from_snapshot(snap)
        return snap, state.stats.clone()

    def _prune_completed_requests_from_snapshot(self, snapshot: Any) -> Any:
        if not isinstance(snapshot, dict):
            return snapshot

        request_states_raw = snapshot.get("request_states", None)
        if not isinstance(request_states_raw, dict) or not request_states_raw:
            return snapshot

        completed_ids: set[int] = set()
        for rid, req_state in request_states_raw.items():
            if isinstance(req_state, dict) and bool(req_state.get("completed", False)):
                try:
                    completed_ids.add(int(rid))
                except Exception:
                    continue

        if not completed_ids:
            return snapshot

        pruned = dict(snapshot)
        pruned["request_states"] = {
            rid: req_state
            for rid, req_state in request_states_raw.items()
            if int(rid) not in completed_ids
        }

        waiting_ids = snapshot.get("waiting_ids", None)
        if isinstance(waiting_ids, list):
            pruned["waiting_ids"] = [
                int(rid) for rid in waiting_ids if int(rid) not in completed_ids
            ]

        running_ids = snapshot.get("running_ids", None)
        if isinstance(running_ids, list):
            pruned["running_ids"] = [
                int(rid) for rid in running_ids if int(rid) not in completed_ids
            ]

        overrides = snapshot.get("overrides", None)
        if isinstance(overrides, dict):
            pruned["overrides"] = {
                k: v for k, v in overrides.items() if int(k) not in completed_ids
            }

        return pruned

    def clear_scratch_state(self) -> None:
        self._scratch_state = None

    def _restore_frontier_state(self, node: _HistoryFrontierNode) -> VidurMCTSState:
        state = self._scratch_state
        if state is None:
            state = self.env.initial_state()
            self._scratch_state = state

        sim = state.simulator
        snapshot = node.state_snapshot
        if (
            isinstance(snapshot, dict)
            and snapshot.get("__mode__") == "mcts_fast"
            and hasattr(sim, "restore_state_fast")
        ):
            sim.restore_state_fast(snapshot)
        else:
            sim.restore_state(snapshot)
        state.stats = node.state_stats.clone()

        sync_fn = getattr(self.env, "_sync_active_ids_once", None)
        if callable(sync_fn):
            sync_fn(state)
        init_clock_fn = getattr(self.env, "_v2_init_clock", None)
        if callable(init_clock_fn):
            init_clock_fn(state, float(getattr(sim, "_time", 0.0)))
        return state

    def _frontier_node_from_state(
        self,
        *,
        state: VidurMCTSState,
        player: str,
        depth: int,
        history_hops: int,
        log_node_id: int | None,
        pre_controller_snapshot: Any | None,
        pre_controller_stats: Any | None,
    ) -> _HistoryFrontierNode:
        snapshot, stats = self._snapshot_pair(state)
        return _HistoryFrontierNode(
            state_snapshot=snapshot,
            state_stats=stats,
            player=str(player),
            depth=int(depth),
            history_hops=int(history_hops),
            log_node_id=None if log_node_id is None else int(log_node_id),
            pre_controller_snapshot=pre_controller_snapshot,
            pre_controller_stats=pre_controller_stats,
            untried_action_indices=[],
        )

    def _log_history_trace_step(
        self,
        *,
        session: _HistoryRootBatchSession | None,
        parent_node_id: int | None,
        depth_before: int,
        player_acted: str,
        next_player: str,
        action_index: int,
        action: object,
        n_valid: int,
        state_after: VidurMCTSState,
    ) -> int | None:
        if session is None or session.history_trace_logger is None:
            return parent_node_id

        node_id = int(session.next_log_node_id)
        session.next_log_node_id = int(session.next_log_node_id) + 1
        self._log_step(
            game_id=int(session.game_id),
            root_id=int(session.trace_root_id),
            root_depth=int(depth_before),
            node_depth=int(depth_before) + 1,
            node_id=int(node_id),
            parent_node_id=parent_node_id,
            player_acted=str(player_acted),
            next_player=str(next_player),
            action_index=int(action_index),
            action=action,
            n_valid=int(n_valid),
            phase=self._phase_label(str(player_acted), action, int(n_valid)),
            state_after=state_after,
            iter_loggers=(session.history_trace_logger,),
            root_loggers=(),
        )
        return int(node_id)

    def _apply_with_parent_context(
        self,
        state: VidurMCTSState,
        player: str,
        action: object,
    ) -> tuple[VidurMCTSState, str, Any | None, Any | None]:
        pre_ctrl_snapshot = None
        pre_ctrl_stats = None

        if player == "controller":
            pre_ctrl_snapshot, pre_ctrl_stats = self._snapshot_pair(state)

        state, next_player = self._apply(state, player, action)

        if player != "controller":
            return state, next_player, None, None

        src_fn = getattr(self.env, "_v2_missed_adv_source", None)
        miss_src = int(src_fn(state)) if callable(src_fn) else 0
        if miss_src == 1 and pre_ctrl_snapshot is not None and pre_ctrl_stats is not None:
            return state, next_player, pre_ctrl_snapshot, pre_ctrl_stats

        return state, next_player, None, None

    def _advance_to_branching_with_context(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        pre_controller_snapshot: Any | None,
        pre_controller_stats: Any | None,
        max_forced_steps: int,
        session: _HistoryRootBatchSession | None = None,
        log_parent_id: int | None = None,
    ) -> tuple[VidurMCTSState, str, int, Any | None, Any | None, int, int | None]:
        forced = 0
        pending_snap = pre_controller_snapshot
        pending_stats = pre_controller_stats
        last_log_node_id = log_parent_id

        while forced < int(max_forced_steps):
            actions_by_index, valid, _mask = self._actions_valid(state, player)
            if len(valid) != 1:
                break

            idx = int(valid[0])
            action = actions_by_index[idx]
            if action is None:
                break

            acted = player
            depth_before = int(depth)
            state, player, pending_snap, pending_stats = self._apply_with_parent_context(
                state,
                player,
                action,
            )
            depth = int(depth) + 1
            forced += 1
            last_log_node_id = self._log_history_trace_step(
                session=session,
                parent_node_id=last_log_node_id,
                depth_before=int(depth_before),
                player_acted=str(acted),
                next_player=str(player),
                action_index=int(idx),
                action=action,
                n_valid=len(valid),
                state_after=state,
            )

        return state, player, int(depth), pending_snap, pending_stats, int(forced), last_log_node_id

    def _roll_to_target_hops(
        self,
        *,
        initial_state: VidurMCTSState,
        start_player: str,
        start_depth: int,
        target_hops: int,
        rng: random.Random,
        max_total_steps: int,
        session: _HistoryRootBatchSession | None = None,
    ) -> _HistoryFrontierNode:
        state = initial_state.fork(flag=False)
        player = str(start_player)
        depth = int(start_depth)
        hops = 0
        steps = 0
        log_node_id: int | None = None
        pre_ctrl_snapshot = None
        pre_ctrl_stats = None

        step_cap = max(1, int(max_total_steps))
        target = max(0, int(target_hops))

        while hops < target and steps < step_cap:
            remaining = max(0, step_cap - steps)
            if remaining <= 0:
                break

            state, player, depth, pre_ctrl_snapshot, pre_ctrl_stats, forced, log_node_id = self._advance_to_branching_with_context(
                state,
                player,
                depth,
                pre_controller_snapshot=pre_ctrl_snapshot,
                pre_controller_stats=pre_ctrl_stats,
                max_forced_steps=min(2000, int(remaining)),
                session=session,
                log_parent_id=log_node_id,
            )
            steps += int(forced)
            if steps >= step_cap:
                break

            actions_by_index, valid, _mask = self._actions_valid(state, player)
            if not valid:
                break

            if len(valid) == 1:
                idx = int(valid[0])
                is_nontrivial = False
            else:
                idx = int(rng.choice(valid))
                is_nontrivial = True

            action = actions_by_index[idx]
            if action is None:
                break

            acted = player
            depth_before = int(depth)
            state, player, pre_ctrl_snapshot, pre_ctrl_stats = self._apply_with_parent_context(
                state,
                player,
                action,
            )
            depth = int(depth) + 1
            steps += 1
            log_node_id = self._log_history_trace_step(
                session=session,
                parent_node_id=log_node_id,
                depth_before=int(depth_before),
                player_acted=str(acted),
                next_player=str(player),
                action_index=int(idx),
                action=action,
                n_valid=len(valid),
                state_after=state,
            )
            if is_nontrivial:
                hops += 1

        remaining = max(0, step_cap - steps)
        if remaining > 0:
            state, player, depth, pre_ctrl_snapshot, pre_ctrl_stats, _forced, log_node_id = self._advance_to_branching_with_context(
                state,
                player,
                depth,
                pre_controller_snapshot=pre_ctrl_snapshot,
                pre_controller_stats=pre_ctrl_stats,
                max_forced_steps=min(2000, int(remaining)),
                session=session,
                log_parent_id=log_node_id,
            )

        return self._frontier_node_from_state(
            state=state,
            player=str(player),
            depth=int(depth),
            history_hops=int(hops),
            log_node_id=log_node_id,
            pre_controller_snapshot=pre_ctrl_snapshot,
            pre_controller_stats=pre_ctrl_stats,
        )

    def _node_signature(self, node: _HistoryFrontierNode) -> tuple[Any, ...]:
        state = self._restore_frontier_state(node)
        desc = self.env.describe_state(state)
        active_ids = tuple(int(x) for x in (desc.get("active_request_ids") or []))
        completed_ids = tuple(int(x) for x in (desc.get("completed_request_ids") or []))
        return (
            str(node.player),
            int(node.history_hops),
            round(float(desc.get("sim_time", 0.0)), 9),
            bool(desc.get("pending_adv_tick", False)),
            int(desc.get("decode_credit_balance", 0)),
            active_ids,
            completed_ids,
        )

    def _prepare_untried_actions_for_node(
        self,
        *,
        node: _HistoryFrontierNode,
        rng: random.Random,
        max_history_hops: int,
        max_children_per_expand: int | None,
        remaining_need: int,
    ) -> None:
        node.untried_action_indices = []
        if int(node.history_hops) >= int(max_history_hops):
            return

        state = self._restore_frontier_state(node)
        actions_by_index, valid, _mask = self._actions_valid(state, node.player)
        del actions_by_index
        if len(valid) <= 1:
            return

        valid_indices = [int(i) for i in valid]
        cap = len(valid_indices)
        if max_children_per_expand is not None and int(max_children_per_expand) > 0:
            cap = min(cap, int(max_children_per_expand))
        if int(remaining_need) > 0:
            cap = min(cap, max(1, int(remaining_need)))

        if cap < len(valid_indices):
            sampled = rng.sample(valid_indices, cap)
        else:
            sampled = list(valid_indices)
        rng.shuffle(sampled)
        node.untried_action_indices = sampled

    def _expand_one_child(
        self,
        *,
        node: _HistoryFrontierNode,
        action_index: int,
        max_total_steps: int,
        session: _HistoryRootBatchSession | None = None,
    ) -> _HistoryFrontierNode | None:
        parent = self._restore_frontier_state(node)
        actions_by_index, valid, _mask = self._actions_valid(parent, node.player)
        valid_set = set(int(i) for i in valid)
        idx = int(action_index)
        if idx not in valid_set:
            return None

        action = actions_by_index[idx]
        if action is None:
            return None

        depth_before = int(node.depth)
        child, next_player, pre_ctrl_snapshot, pre_ctrl_stats = self._apply_with_parent_context(
            parent,
            node.player,
            action,
        )

        child_depth = int(node.depth) + 1
        child_hops = int(node.history_hops) + 1
        log_node_id = self._log_history_trace_step(
            session=session,
            parent_node_id=node.log_node_id,
            depth_before=int(depth_before),
            player_acted=str(node.player),
            next_player=str(next_player),
            action_index=int(idx),
            action=action,
            n_valid=len(valid),
            state_after=child,
        )

        child, next_player, child_depth, pre_ctrl_snapshot, pre_ctrl_stats, _forced, log_node_id = self._advance_to_branching_with_context(
            child,
            next_player,
            child_depth,
            pre_controller_snapshot=pre_ctrl_snapshot,
            pre_controller_stats=pre_ctrl_stats,
            max_forced_steps=min(2000, max(1, int(max_total_steps))),
            session=session,
            log_parent_id=log_node_id,
        )

        return self._frontier_node_from_state(
            state=child,
            player=str(next_player),
            depth=int(child_depth),
            history_hops=int(child_hops),
            log_node_id=log_node_id,
            pre_controller_snapshot=pre_ctrl_snapshot,
            pre_controller_stats=pre_ctrl_stats,
        )

    def _as_root_record(
        self,
        *,
        node: _HistoryFrontierNode,
        root_id: int,
    ) -> dict[str, Any]:
        history_signature = self._node_signature(node)
        root_state = self._restore_frontier_state(node).fork(flag=False)
        pre_ctrl_snapshot = node.pre_controller_snapshot if node.player == "adversary" else None
        pre_ctrl_stats = node.pre_controller_stats if node.player == "adversary" else None
        return {
            "root_state": root_state,
            "root_player": str(node.player),
            "root_depth": int(node.depth),
            "root_id": int(root_id),
            "root_node_id_override": None,
            "history_log_node_id": node.log_node_id,
            "pre_controller_snapshot": pre_ctrl_snapshot,
            "pre_controller_stats": pre_ctrl_stats,
            "history_hops": int(node.history_hops),
            "history_signature": history_signature,
        }

    def _create_root_batch_session(
        self,
        *,
        initial_state: Optional[VidurMCTSState],
        start_player: str,
        start_depth: int,
        num_roots: int,
        start_root_id: int,
        game_id: int,
        nontrivial_hops: int,
        seed: int,
        max_total_steps: int,
        min_history_hops: Optional[int],
        max_history_hops: Optional[int],
        max_children_per_expand: Optional[int],
        initial_seen_signatures: Optional[Sequence[tuple[Any, ...]]],
        history_trace_logger: Any | None = None,
    ) -> _HistoryRootBatchSession:
        target_roots = max(0, int(num_roots))
        min_hops = int(nontrivial_hops if min_history_hops is None else min_history_hops)
        max_hops = int(min_hops if max_history_hops is None else max_history_hops)
        if max_hops < min_hops:
            max_hops = int(min_hops)

        return _HistoryRootBatchSession(
            start_state=initial_state.fork(flag=False) if initial_state is not None else self.env.initial_state(),
            start_player=str(start_player),
            start_depth=int(start_depth),
            min_history_hops=int(min_hops),
            max_history_hops=int(max_hops),
            max_total_steps=int(max_total_steps),
            max_children_per_expand=max_children_per_expand,
            target_roots=int(target_roots),
            next_root_id=int(start_root_id),
            rng=random.Random(int(seed)),
            anchor_budget=max(4, int(target_roots) * 3),
            max_iterations=max(200, int(target_roots) * 50),
            game_id=int(game_id),
            trace_root_id=int(start_root_id),
            next_log_node_id=0,
            history_trace_logger=history_trace_logger,
            seen_signatures=set(initial_seen_signatures or ()),
        )

    def _prepare_session_node(
        self,
        *,
        node: _HistoryFrontierNode,
        session: _HistoryRootBatchSession,
    ) -> None:
        remaining_need = max(0, int(session.target_roots) - int(session.emitted_roots))
        self._prepare_untried_actions_for_node(
            node=node,
            rng=session.rng,
            max_history_hops=int(session.max_history_hops),
            max_children_per_expand=session.max_children_per_expand,
            remaining_need=int(remaining_need),
        )

    def _try_claim_signature(
        self,
        *,
        session: _HistoryRootBatchSession,
        signature: tuple[Any, ...],
    ) -> bool:
        if signature in session.seen_signatures:
            return False

        shared_seen = session.shared_seen_signatures
        shared_lock = session.shared_seen_lock
        if shared_seen is None:
            session.seen_signatures.add(signature)
            return True

        claimed = False
        if shared_lock is None:
            if signature not in shared_seen:
                shared_seen[signature] = 1
                claimed = True
        else:
            with shared_lock:
                if signature not in shared_seen:
                    shared_seen[signature] = 1
                    claimed = True

        session.seen_signatures.add(signature)
        return bool(claimed)

    def _make_anchor_node(self, session: _HistoryRootBatchSession) -> _HistoryFrontierNode | None:
        if int(session.anchors_built) >= int(session.anchor_budget):
            session.exhausted = True
            return None
        if int(session.iterations) >= int(session.max_iterations):
            session.exhausted = True
            return None

        anchor = self._roll_to_target_hops(
            initial_state=session.start_state,
            start_player=str(session.start_player),
            start_depth=int(session.start_depth),
            target_hops=int(session.min_history_hops),
            rng=session.rng,
            max_total_steps=int(session.max_total_steps),
            session=session,
        )
        session.anchors_built += 1
        session.iterations += 1
        self._prepare_session_node(node=anchor, session=session)
        if anchor.untried_action_indices:
            session.active_path.append(anchor)
        return anchor

    def _next_unique_root_node(self, session: _HistoryRootBatchSession) -> _HistoryFrontierNode | None:
        while not bool(session.exhausted):
            if int(session.iterations) >= int(session.max_iterations):
                session.exhausted = True
                return None

            if not session.active_path:
                anchor = self._make_anchor_node(session)
                if anchor is None:
                    return None

                sig = self._node_signature(anchor)
                if not self._try_claim_signature(session=session, signature=sig):
                    continue
                return anchor

            node = session.active_path[-1]
            if not node.untried_action_indices:
                session.active_path.pop()
                continue

            action_index = int(node.untried_action_indices.pop())
            child = self._expand_one_child(
                node=node,
                action_index=action_index,
                max_total_steps=int(session.max_total_steps),
                session=session,
            )
            session.iterations += 1
            if child is None:
                continue

            self._prepare_session_node(node=child, session=session)
            if child.untried_action_indices:
                session.active_path.append(child)

            sig = self._node_signature(child)
            if not self._try_claim_signature(session=session, signature=sig):
                continue
            return child

        return None

    def _next_root_node_allow_duplicates(self, session: _HistoryRootBatchSession) -> _HistoryFrontierNode | None:
        node = self._next_unique_root_node(session)
        if node is not None:
            return node
        if not bool(session.allow_duplicate_fallback):
            return None

        if int(session.emitted_roots) >= int(session.target_roots):
            return None

        fallback_hops = int(session.rng.randint(int(session.min_history_hops), int(session.max_history_hops)))
        return self._roll_to_target_hops(
            initial_state=session.start_state,
            start_player=str(session.start_player),
            start_depth=int(session.start_depth),
            target_hops=int(fallback_hops),
            rng=session.rng,
            max_total_steps=int(session.max_total_steps),
            session=session,
        )

    def generate_roots_batch_iter(
        self,
        *,
        initial_state: Optional[VidurMCTSState],
        start_player: str,
        start_depth: int,
        num_roots: int,
        game_id: int,
        start_root_id: int,
        nontrivial_hops: int = 0,
        seed: int = 0,
        max_total_steps: int = 20000,
        log_history: bool = False,
        min_history_hops: Optional[int] = None,
        max_history_hops: Optional[int] = None,
        max_children_per_expand: Optional[int] = None,
        batch_size: int = 64,
        initial_seen_signatures: Optional[Sequence[tuple[Any, ...]]] = None,
        shared_seen_signatures: Any | None = None,
        shared_seen_lock: Any | None = None,
        allow_duplicate_fallback: bool = True,
        history_trace_logger: Any | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        del log_history

        session = self._create_root_batch_session(
            initial_state=initial_state,
            start_player=str(start_player),
            start_depth=int(start_depth),
            num_roots=int(num_roots),
            start_root_id=int(start_root_id),
            game_id=int(game_id),
            nontrivial_hops=int(nontrivial_hops),
            seed=int(seed),
            max_total_steps=int(max_total_steps),
            min_history_hops=min_history_hops,
            max_history_hops=max_history_hops,
            max_children_per_expand=max_children_per_expand,
            initial_seen_signatures=initial_seen_signatures,
            history_trace_logger=history_trace_logger,
        )
        session.shared_seen_signatures = shared_seen_signatures
        session.shared_seen_lock = shared_seen_lock
        session.allow_duplicate_fallback = bool(allow_duplicate_fallback)
        if int(session.target_roots) <= 0:
            return

        root_batch_size = max(1, int(batch_size))
        while int(session.emitted_roots) < int(session.target_roots):
            batch: list[dict[str, Any]] = []
            while len(batch) < root_batch_size and int(session.emitted_roots) < int(session.target_roots):
                node = self._next_root_node_allow_duplicates(session)
                if node is None:
                    break
                batch.append(self._as_root_record(node=node, root_id=int(session.next_root_id)))
                session.next_root_id += 1
                session.emitted_roots += 1

            if not batch:
                break
            yield batch

    def generate_roots_batch(
        self,
        *,
        initial_state: Optional[VidurMCTSState],
        start_player: str,
        start_depth: int,
        num_roots: int,
        game_id: int,
        start_root_id: int,
        nontrivial_hops: int = 0,
        seed: int = 0,
        max_total_steps: int = 20000,
        log_history: bool = False,
        min_history_hops: Optional[int] = None,
        max_history_hops: Optional[int] = None,
        max_children_per_expand: Optional[int] = None,
        initial_seen_signatures: Optional[Sequence[tuple[Any, ...]]] = None,
        shared_seen_signatures: Any | None = None,
        shared_seen_lock: Any | None = None,
        allow_duplicate_fallback: bool = True,
        history_trace_logger: Any | None = None,
    ) -> list[dict[str, Any]]:
        roots: list[dict[str, Any]] = []
        for batch in self.generate_roots_batch_iter(
            initial_state=initial_state,
            start_player=str(start_player),
            start_depth=int(start_depth),
            num_roots=int(num_roots),
            game_id=int(game_id),
            start_root_id=int(start_root_id),
            nontrivial_hops=int(nontrivial_hops),
            seed=int(seed),
            max_total_steps=int(max_total_steps),
            log_history=bool(log_history),
            min_history_hops=min_history_hops,
            max_history_hops=max_history_hops,
            max_children_per_expand=max_children_per_expand,
            batch_size=max(1, min(int(num_roots), 64)),
            initial_seen_signatures=initial_seen_signatures,
            shared_seen_signatures=shared_seen_signatures,
            shared_seen_lock=shared_seen_lock,
            allow_duplicate_fallback=bool(allow_duplicate_fallback),
            history_trace_logger=history_trace_logger,
        ):
            roots.extend(batch)
        return roots
