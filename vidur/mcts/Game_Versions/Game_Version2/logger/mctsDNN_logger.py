# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
dnn_logging.py

MuZero-style logging helpers for Vidur DNN MCTS.

Two logs:
1) Iteration-level log: one row per MCTS simulation (within a root).
   - includes game_id/root_id/root_node_id/root_depth
   - includes node/action details + reward/value
   - includes simulator snapshot fields (same ones you already log)

2) Root-summary log: one row per root after N simulations.
   - includes model priors/value at root
   - includes MCTS-derived policy/value at root
   - includes chosen/best action index
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


def _j(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)

# TODO: make these two functions shared utility somewhere because infer.py also uses them
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _compact_action_repr(s: str, max_len: int = 72) -> str:
    t = " ".join(str(s or "").split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."



def _topk_actions_json(
    probs: Sequence[float],
    valid_mask: Sequence[bool],
    action_repr_by_index: Optional[Dict[int, str]],
    k: int = 5,
) -> str:
    if not probs:
        return "[]"

    mask = list(valid_mask) if valid_mask is not None else []
    cand: list[tuple[int, float]] = []

    for i, p in enumerate(probs):
        if mask:
            if i >= len(mask) or not bool(mask[i]):
                continue
        cand.append((int(i), float(p)))

    if not cand:
        cand = [(int(i), float(p)) for i, p in enumerate(probs)]

    cand.sort(key=lambda x: (x[1], -x[0]), reverse=True)

    out = []
    amap = action_repr_by_index or {}
    for i, p in cand[: max(0, int(k))]:
        raw = amap.get(int(i), "")
        label = _compact_action_repr(raw) if str(raw or "").strip() else f"idx={int(i)}"
        out.append(
            {
                "i": int(i),
                "p": round(float(p), 6),
                "a": label,
            }
        )
    return _j(out)



class DNNMCTSIterationLogger:
    """
    One row per MCTS simulation step (per root).

    This is your “old _MCTSLogger”, but extended with:
      - game_id
      - root_id (self-play step index)
      - root_depth/root_node_id/root_player
      - sim_iteration (0..N-1 within this root)

    You can log at different phases, but I recommend logging only the “expand/eval”
    point once per simulation to keep file size manageable.
    """

    FIELDS = [
        "game_id",
        "root_id",
        "sim_iteration",

        "root_depth",
        "root_node_id",
        "root_player",

        "phase",            # e.g. "expand", "backup", "select" (use "expand" initially)
        "node_depth",
        "parent_node_id",
        "node_id",

        "player_acted_to_create_this_node",
        "player_to_act_in_this_node",
        

        # action info (incoming edge into node or chosen edge)
        "action_index",
        "action_repr",
        "prior",
        "model_prior_json",
        "normalized_prior_json",
        "reward",
        # "action_cost_softcap",

        # To see/debug whether DNN was called to filter non trivial actions
        "nn_called",
        "num_valid_actions",
        "unique_actions",

        # values
        "nn_value_controller",
        # "mcts_value_controller",

        # absolute cost snapshot (optional but useful)
        "objective_cost",

        "model_top5_actions_json",
        "mcts_top5_actions_json",


        # simulator snapshot fields (same as your old logger)
        "sim_time",
        "decision_state_time",
        "start_time",
        "end_time",
        "stage_total_time",
        "requests_in_system",
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "total_lateness",
        "avg_lateness",
        "state_active_ids",
        "state_waiting_ids",
        "state_completed_request_ids",
        "state_dropped_request_ids",
        "state_stopped_decode_request_ids",
        "state_pending_adv_tick",
        "state_last_adv_tick",
        "state_decode_credit_balance",
        "state_decode_tokens_counted_by_id",
        "state_violated_request_ids",
        "state_per_request_prefill_lateness_by_id",
        "state_per_request_decode_lateness_by_id",

        # action payloads (optional, but useful)
        "adversary_requests",
        "adversary_prefill_slos",
        "adversary_prefill_deadlines_by_id",
        "adversary_decode_slos",
        "controller_token_budget",
        "controller_selected_ids",
        "controller_allocations",
        "controller_prefill_allocations",
        "controller_decode_allocations",
        "controller_prefill_total",
        "controller_decode_total",
        "controller_heuristic",
        "controller_strategy",
    ]

    PUCT_FIELDS = [
        "game_id",
        "root_id",
        "sim_iteration",
        "root_node_id",
        "parent_node_id",
        "node_id",
        "player_acted_to_create_this_node",
        "reward",
        "node_dnn_value",
        "children_created_json",
        "dedup_children_json",
        "ancestor_chain_json",
        "selection_trace_json",
        "minmax_min",
        "minmax_max",
    ]

    #INTERNAL FEILDS FOR FAST FORWARD DECODING :
    INTERNAL_FIELDS = [
        "game_id",
        "root_id",
        "sim_iteration",
        "root_depth",
        "root_node_id",
        "root_player",
        "event_seq",
        "event_id",
        "phase",
        "node_id",
        "parent_node_id",
        "start_time",
        "end_time",
        "sim_time",
        "stage_total_time",
        "reason",
        "request_ids_json",
        "num_tokens_json",
        "decode_credit_before",
        "decode_credit_after",
        "requests_in_system",
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "total_lateness",
        "avg_lateness",
        "state_active_ids",
        "state_waiting_ids",
        "state_completed_request_ids",
        "state_dropped_request_ids",
        "state_stopped_decode_request_ids",
        "state_pending_adv_tick",
        "state_last_adv_tick",
        "state_decode_credit_balance",
        "state_decode_tokens_counted_by_id",
        "state_violated_request_ids",
        "state_per_request_prefill_lateness_by_id",
        "state_per_request_decode_lateness_by_id",
    ]


    def __init__(self, path: Optional[Union[str, Path]], *, flush_every: int = 1) -> None:
        self._path = Path(path) if path else None
        self._flush_every = max(1, int(flush_every))
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows = 0
        self._puct_file = None
        self._puct_writer: Optional[csv.DictWriter] = None
        self._internal_file = None
        self._internal_writer: Optional[csv.DictWriter] = None


    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        if self._puct_file is not None:
            self._puct_file.close()
        if self._internal_file is not None:
            self._internal_file.close()
        self._file = None
        self._writer = None
        self._puct_file = None
        self._puct_writer = None
        self._internal_file = None
        self._internal_writer = None

    def _ensure(self) -> None:
        if not self._path:
            return
        if self._writer is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def _ensure_puct(self) -> None:
        if not self._path:
            return
        if self._puct_writer is not None:
            return
        p = self._path.with_name(self._path.stem + "_puct.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        self._puct_file = p.open("w", newline="")
        self._puct_writer = csv.DictWriter(self._puct_file, fieldnames=self.PUCT_FIELDS)
        self._puct_writer.writeheader()

    def _ensure_internal(self) -> None:
        if not self._path:
            return
        if self._internal_writer is not None:
            return
        p = self._path.with_name(self._path.stem + "_internal.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        self._internal_file = p.open("w", newline="")
        self._internal_writer = csv.DictWriter(self._internal_file, fieldnames=self.INTERNAL_FIELDS)
        self._internal_writer.writeheader()


    def log_expand(
        self,
        *,
        game_id: int,
        root_id: int,
        sim_iteration: int,
        root_depth: int,
        root_node_id: int,
        root_player: str,

        node_depth: int,
        parent_node_id: Optional[int],
        node_id: int,
        player_to_act: str,
        player_acted_to_create_this_node: str,

        action_index: Optional[int],
        action_repr: str,
        prior: float,
        model_prior_json: str = "[]",
        normalized_prior_json: str = "[]",

        reward: float,
        # action_cost_softcap: float,

        # mcts_value_controller: float,

        # To see/debug whether DNN was called
        nn_called: bool,
        num_valid_actions: int,
        unique_actions: int,
        nn_value_controller: Optional[float],

        objective_cost: float,

        state_snapshot: Dict[str, Any],
        adversary_action_json: str = "[]",
        adversary_prefill_slos_json: str = "[]",
        adversary_prefill_deadlines_by_id_json: str = "{}",
        adversary_decode_slos_json: str = "[]",
        controller_token_budget: Union[int, str] = "",
        controller_selected_ids_json: str = "[]",
        controller_allocations_json: str = "{}",
        controller_prefill_allocations_json: str = "{}",
        controller_decode_allocations_json: str = "{}",
        controller_prefill_total: int = 0,
        controller_decode_total: int = 0,
        controller_heuristic: str = "",
        controller_strategy: str = "",
        phase: str = "expand",
        decision_state_time: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        stage_total_time: Optional[float] = None,
    ) -> None:
        if not self._path:
            return
        self._ensure()
        assert self._writer is not None

        sim_time = _safe_float(state_snapshot.get("sim_time", 0.0))
        st = sim_time if start_time is None else _safe_float(start_time, sim_time)
        et = sim_time if end_time is None else _safe_float(end_time, sim_time)
        dt = (
            _safe_float(stage_total_time, max(0.0, et - st))
            if stage_total_time is not None
            else max(0.0, et - st)
        )

        row = {
            "game_id": game_id,
            "root_id": root_id,
            "sim_iteration": sim_iteration,

            "root_depth": root_depth,
            "root_node_id": root_node_id,
            "root_player": root_player,

            "phase": phase,
            "node_depth": node_depth,
            "parent_node_id": "" if parent_node_id is None else str(parent_node_id),
            "node_id": node_id,

            "player_acted_to_create_this_node": player_acted_to_create_this_node,
            "player_to_act_in_this_node": player_to_act,
        

            "action_index": "" if action_index is None else int(action_index),
            "action_repr": action_repr,
            "prior": _safe_float(prior),
            "model_prior_json": str(model_prior_json),
            "normalized_prior_json": str(normalized_prior_json),

            "reward": _safe_float(reward),
            # "action_cost_softcap": _safe_float(action_cost_softcap),

            "nn_called": bool(nn_called),
            "num_valid_actions": int(num_valid_actions),
            "unique_actions": int(unique_actions),
            "nn_value_controller": "" if nn_value_controller is None else _safe_float(nn_value_controller),
            
            # "mcts_value_controller": _safe_float(mcts_value_controller),

            "objective_cost": _safe_float(objective_cost),

            "sim_time": sim_time,
            "decision_state_time": (
                sim_time if decision_state_time is None else _safe_float(decision_state_time, sim_time)
            ),
            "start_time": st,
            "end_time": et,
            "stage_total_time": dt,
            "requests_in_system": state_snapshot.get("requests_in_system", 0),
            "requests_generated": state_snapshot.get("requests_generated", 0),
            "requests_completed": state_snapshot.get("requests_completed", 0),
            "slo_violations": state_snapshot.get("slo_violations", 0),
            "total_lateness": state_snapshot.get("total_lateness", state_snapshot.get("avg_lateness", 0.0)),
            "avg_lateness": state_snapshot.get("avg_lateness", 0.0),
            "state_active_ids": _j(state_snapshot.get("active_request_ids", [])),
            "state_waiting_ids": _j(state_snapshot.get("waiting_request_ids", [])),
            "state_completed_request_ids": _j(state_snapshot.get("completed_request_ids", [])),
            "state_dropped_request_ids": _j(state_snapshot.get("dropped_request_ids", [])),
            "state_stopped_decode_request_ids": _j(state_snapshot.get("stopped_decode_request_ids", [])),
            "state_pending_adv_tick": bool(state_snapshot.get("pending_adv_tick", False)),
            "state_last_adv_tick": state_snapshot.get("last_adv_tick", ""),
            "state_decode_credit_balance": _safe_int(
                state_snapshot.get("decode_credit_balance", state_snapshot.get("decode_credit_available", 0))
            ),
            "state_decode_tokens_counted_by_id": _j(state_snapshot.get("decode_tokens_counted_by_id", {})),
            "state_violated_request_ids": _j(state_snapshot.get("violated_request_ids", [])),
            "state_per_request_prefill_lateness_by_id": _j(state_snapshot.get("per_request_prefill_lateness_by_id", {})),
            "state_per_request_decode_lateness_by_id": _j(state_snapshot.get("per_request_decode_lateness_by_id", {})),

            "adversary_requests": adversary_action_json,
            "adversary_prefill_slos": adversary_prefill_slos_json,
            "adversary_prefill_deadlines_by_id": adversary_prefill_deadlines_by_id_json,
            "adversary_decode_slos": adversary_decode_slos_json,

            "controller_token_budget": controller_token_budget,
            "controller_selected_ids": controller_selected_ids_json,
            "controller_allocations": controller_allocations_json,
            "controller_prefill_allocations": controller_prefill_allocations_json,
            "controller_decode_allocations": controller_decode_allocations_json,
            "controller_prefill_total": _safe_int(controller_prefill_total),
            "controller_decode_total": _safe_int(controller_decode_total),
            "controller_heuristic": controller_heuristic,
            "controller_strategy": controller_strategy,
        }

        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._file.flush()  # type: ignore[union-attr]

    def puct_log_expand(
        self,
        *,
        game_id: int,
        root_id: int,
        sim_iteration: int,
        root_node_id: int,
        parent_node_id: Optional[int],
        node_id: int,
        player_acted_to_create_this_node: str,
        reward: float,
        node_dnn_value: Optional[float],
        children_created: list,
        dedup_children: list,
        ancestor_chain: list,
        selection_trace: list,
        minmax_min: float,
        minmax_max: float,
    ) -> None:
        if not self._path:
            return
        self._ensure_puct()
        assert self._puct_writer is not None
        row = {
            "game_id": int(game_id),
            "root_id": int(root_id),
            "sim_iteration": int(sim_iteration),
            "root_node_id": int(root_node_id),
            "parent_node_id": "" if parent_node_id is None else int(parent_node_id),
            "node_id": int(node_id),
            "player_acted_to_create_this_node": str(player_acted_to_create_this_node),
            "reward": _safe_float(reward),
            "node_dnn_value": "" if node_dnn_value is None else _safe_float(node_dnn_value),
            "children_created_json": _j(children_created),
            "dedup_children_json": _j(dedup_children),
            "ancestor_chain_json": _j(ancestor_chain),
            "selection_trace_json": _j(selection_trace),
            "minmax_min": "" if math.isinf(minmax_min) else float(minmax_min),
            "minmax_max": "" if math.isinf(minmax_max) else float(minmax_max),
        }
        self._puct_writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._puct_file.flush()  # type: ignore[union-attr]

    def log_internal_step(
        self,
        *,
        game_id: int,
        root_id: int,
        sim_iteration: int,
        root_depth: int,
        root_node_id: int,
        root_player: str,
        event_seq: int,
        event_id: int,
        phase: str,
        node_id: Optional[int],
        parent_node_id: Optional[int],
        start_time: float,
        end_time: float,
        stage_total_time: float,
        reason: str,
        request_ids: Sequence[int],
        num_tokens: Sequence[int],
        decode_credit_before: Optional[int],
        decode_credit_after: Optional[int],
        state_snapshot: Dict[str, Any],
    ) -> None:
        if not self._path:
            return
        self._ensure_internal()
        assert self._internal_writer is not None

        row = {
            "game_id": int(game_id),
            "root_id": int(root_id),
            "sim_iteration": int(sim_iteration),
            "root_depth": int(root_depth),
            "root_node_id": int(root_node_id),
            "root_player": str(root_player),
            "event_seq": int(event_seq),
            "event_id": int(event_id),
            "phase": str(phase),
            "node_id": "" if node_id is None else int(node_id),
            "parent_node_id": "" if parent_node_id is None else int(parent_node_id),
            "start_time": _safe_float(start_time),
            "end_time": _safe_float(end_time),
            "sim_time": _safe_float(state_snapshot.get("sim_time", end_time)),
            "stage_total_time": _safe_float(stage_total_time),
            "reason": str(reason),
            "request_ids_json": _j(list(request_ids)),
            "num_tokens_json": _j(list(num_tokens)),
            "decode_credit_before": "" if decode_credit_before is None else int(decode_credit_before),
            "decode_credit_after": "" if decode_credit_after is None else int(decode_credit_after),
            "requests_in_system": state_snapshot.get("requests_in_system", 0),
            "requests_generated": state_snapshot.get("requests_generated", 0),
            "requests_completed": state_snapshot.get("requests_completed", 0),
            "slo_violations": state_snapshot.get("slo_violations", 0),
            "total_lateness": state_snapshot.get("total_lateness", state_snapshot.get("avg_lateness", 0.0)),
            "avg_lateness": state_snapshot.get("avg_lateness", 0.0),
            "state_active_ids": _j(state_snapshot.get("active_request_ids", [])),
            "state_waiting_ids": _j(state_snapshot.get("waiting_request_ids", [])),
            "state_completed_request_ids": _j(state_snapshot.get("completed_request_ids", [])),
            "state_dropped_request_ids": _j(state_snapshot.get("dropped_request_ids", [])),
            "state_stopped_decode_request_ids": _j(state_snapshot.get("stopped_decode_request_ids", [])),
            "state_pending_adv_tick": bool(state_snapshot.get("pending_adv_tick", False)),
            "state_last_adv_tick": state_snapshot.get("last_adv_tick", ""),
            "state_decode_credit_balance": _safe_int(
                state_snapshot.get("decode_credit_balance", state_snapshot.get("decode_credit_available", 0))
            ),
            "state_decode_tokens_counted_by_id": _j(state_snapshot.get("decode_tokens_counted_by_id", {})),
            "state_violated_request_ids": _j(state_snapshot.get("violated_request_ids", [])),
            "state_per_request_prefill_lateness_by_id": _j(state_snapshot.get("per_request_prefill_lateness_by_id", {})),
            "state_per_request_decode_lateness_by_id": _j(state_snapshot.get("per_request_decode_lateness_by_id", {})),
        }

        self._internal_writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._internal_file.flush()  # type: ignore[union-attr]

        # Mirror internal event into the main iteration CSV for chronological tracing.
        
        mirror_phases = {"internal:decode_ff_batch", "internal:jump_to_adv_tick"}
        if str(phase) in mirror_phases:
            self.log_expand(
                game_id=game_id,
                root_id=root_id,
                sim_iteration=sim_iteration,
                root_depth=root_depth,
                root_node_id=root_node_id,
                root_player=root_player,
                node_depth=root_depth,
                parent_node_id=parent_node_id,
                node_id=(root_node_id if node_id is None else int(node_id)),
                player_to_act="",
                player_acted_to_create_this_node="",
                action_index=None,
                action_repr="",
                prior=0.0,
                model_prior_json="[]",
                normalized_prior_json="[]",
                reward=0.0,
                nn_called=False,
                num_valid_actions=0,
                unique_actions=0,
                nn_value_controller=None,
                objective_cost=_safe_float(state_snapshot.get("objective_cost", 0.0)),
                state_snapshot=state_snapshot,
                adversary_action_json="[]",
                adversary_prefill_slos_json="[]",
                adversary_prefill_deadlines_by_id_json="{}",
                adversary_decode_slos_json="[]",
                controller_token_budget="",
                controller_selected_ids_json="[]",
                controller_allocations_json="{}",
                controller_prefill_allocations_json="{}",
                controller_decode_allocations_json="{}",
                controller_prefill_total=0,
                controller_decode_total=0,
                controller_heuristic="",
                controller_strategy="",
                phase=str(phase),
                start_time=_safe_float(start_time),
                end_time=_safe_float(end_time),
                stage_total_time=_safe_float(stage_total_time),
            )


class DNNMCTSRootSummaryLogger:
    """
    One row per root AFTER you finish N simulations.

    This is the best place to log:
      - model_root_value
      - model_prior vector
      - mcts_root_value estimate
      - mcts_prior (visit distribution)
      - best_action_index (argmax visits or sampled action index)
      - mask (valid actions)
    """

    FIELDS = [
        "game_id",
        "root_id",
        "root_depth",
        "root_node_id",
        "root_player",
        "num_simulations",

        "model_root_value_controller",
        "model_root_prior_json",
        "normalized_root_prior_json",
        "valid_action_mask_json",
        "model_top5_actions_json",

        "mcts_root_value_controller",
        "mcts_root_prior_json",
        "mcts_top5_actions_json",
        "best_action_index",
        "best_action_mcts_prob",
        "best_action_model_prob",
        "best_action_repr",
        "best_action_json",
        "phase",
        "cycle_label",
        "sim_time",
        "decision_state_time",
        "state_pending_adv_tick",
        "state_last_adv_tick",
        "state_active_ids",
        "state_completed_request_ids",
        "state_decode_credit_balance",
        "state_decode_tokens_counted_by_id",
        "slo_violations",
        "total_lateness",
        "total_cost",
    ]

    def __init__(self, path: Optional[Union[str, Path]], *, flush_every: int = 1) -> None:
        self._path = Path(path) if path else None
        self._flush_every = max(1, int(flush_every))
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows = 0

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    def _ensure(self) -> None:
        if not self._path:
            return
        if self._writer is not None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        expected_header = ",".join(self.FIELDS)
        needs_header = True

        if self._path.exists():
            try:
                if self._path.stat().st_size > 0:
                    needs_header = False
                    with self._path.open("r", encoding="utf-8") as f:
                        existing_header = f.readline().rstrip("\n")

                    if existing_header != expected_header:
                        raise ValueError(
                            f"Existing root log header mismatch at {self._path}.\n"
                            f"Expected: {expected_header}\n"
                            f"Found:    {existing_header}\n"
                            "Delete/rename the old log or update logger fields."
                        )
            except OSError:
                needs_header = True

        self._file = self._path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if needs_header:
            self._writer.writeheader()


    def log_root(
        self,
        *,
        game_id: int,
        root_id: int,
        root_depth: int,
        root_node_id: int,
        root_player: str,
        num_simulations: int,

        model_root_value_controller: float,
        model_root_prior: Sequence[float],
        normalized_root_prior: Sequence[float],
        valid_action_mask: Sequence[bool],

        mcts_root_value_controller: float,
        mcts_root_prior: Sequence[float],

        action_repr_by_index: Optional[Dict[int, str]] = None,

        # best_action_index: int,
        best_action_index: Optional[int],
        best_action_repr: str = "",
        best_action_json: str = "",
        phase: str = "train_root",
        cycle_label: str = "",
        sim_time: float = 0.0,
        decision_state_time: Optional[float] = None,
        state_pending_adv_tick: bool = False,
        state_last_adv_tick: Optional[float] = None,
        state_active_ids: Sequence[int] = (),
        state_completed_request_ids: Sequence[int] = (),
        state_decode_credit_balance: int = 0,
        state_decode_tokens_counted_by_id: Optional[Dict[int, int]] = None,
        slo_violations: int = 0,
        total_lateness: float = 0.0,
        total_cost: float = 0.0,
    ) -> None:
        if not self._path:
            return
        self._ensure()
        assert self._writer is not None

        # best_mcts = float(mcts_root_prior[best_action_index]) if best_action_index < len(mcts_root_prior) else 0.0
        # best_model = float(model_root_prior[best_action_index]) if best_action_index < len(model_root_prior) else 0.0

        idx_ok = best_action_index is not None and 0 <= int(best_action_index) < len(mcts_root_prior)
        best_mcts = float(mcts_root_prior[int(best_action_index)]) if idx_ok else 0.0
        best_model = float(model_root_prior[int(best_action_index)]) if (idx_ok and int(best_action_index) < len(model_root_prior)) else 0.0


        row = {
            "game_id": int(game_id),
            "root_id": int(root_id),
            "root_depth": int(root_depth),
            "root_node_id": int(root_node_id),
            "root_player": str(root_player),
            "num_simulations": int(num_simulations),

            "model_root_value_controller": _safe_float(model_root_value_controller),

            # Keep legacy columns present in schema but empty to avoid wide logs.
            "model_root_prior_json": "",
            "normalized_root_prior_json": "",
            "valid_action_mask_json": "",
            "model_top5_actions_json": _topk_actions_json(
                model_root_prior, valid_action_mask, action_repr_by_index, k=5
            ),

            "mcts_root_value_controller": _safe_float(mcts_root_value_controller),
            # Keep legacy column present in schema but empty to avoid wide logs.
            "mcts_root_prior_json": "",
            "mcts_top5_actions_json": _topk_actions_json(
                mcts_root_prior, valid_action_mask, action_repr_by_index, k=5
            ),


            # "model_root_prior_json": _j(list(model_root_prior)),
            # "normalized_root_prior_json": _j(list(normalized_root_prior)),
            # "valid_action_mask_json": _j(list(valid_action_mask)),

            # "mcts_root_value_controller": _safe_float(mcts_root_value_controller),
            # "mcts_root_prior_json": _j(list(mcts_root_prior)),
            # "best_action_index": int(best_action_index),
            "best_action_index": "" if best_action_index is None else int(best_action_index),
            "best_action_mcts_prob": _safe_float(best_mcts),
            "best_action_model_prob": _safe_float(best_model),
            "best_action_repr": str(best_action_repr or ""),
            "best_action_json": str(best_action_json or ""),
            "phase": str(phase),
            "cycle_label": str(cycle_label),
            "sim_time": float(sim_time),
            "decision_state_time": float(sim_time if decision_state_time is None else decision_state_time),
            "state_pending_adv_tick": bool(state_pending_adv_tick),
            "state_last_adv_tick": "" if state_last_adv_tick is None else float(state_last_adv_tick),
            "state_active_ids": _j([int(x) for x in state_active_ids]),
            "state_completed_request_ids": _j([int(x) for x in state_completed_request_ids]),
            "state_decode_credit_balance": int(state_decode_credit_balance),
            "state_decode_tokens_counted_by_id": _j({
                str(int(k)): int(v)
                for k, v in (state_decode_tokens_counted_by_id or {}).items()
                if int(k) >= 0
            }),
            "slo_violations": int(slo_violations),
            "total_lateness": float(total_lateness),
            "total_cost": float(total_cost),
        }

        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._file.flush()  # type: ignore[union-attr]
