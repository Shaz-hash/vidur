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

        # simulator snapshot fields (same as your old logger)
        "sim_time",
        "requests_in_system",
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "avg_lateness",
        "state_waiting_ids",
        "state_completed_request_ids",

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
        self._file = self._path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

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
    ) -> None:
        if not self._path:
            return
        self._ensure()
        assert self._writer is not None

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

            "sim_time": state_snapshot.get("sim_time", 0.0),
            "requests_in_system": state_snapshot.get("requests_in_system", 0),
            "requests_generated": state_snapshot.get("requests_generated", 0),
            "requests_completed": state_snapshot.get("requests_completed", 0),
            "slo_violations": state_snapshot.get("slo_violations", 0),
            "avg_lateness": state_snapshot.get("avg_lateness", 0.0),
            "state_waiting_ids": _j(state_snapshot.get("waiting_request_ids", [])),
            "state_completed_request_ids": _j(state_snapshot.get("completed_request_ids", [])),

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

        "mcts_root_value_controller",
        "mcts_root_prior_json",
        "best_action_index",
        "best_action_mcts_prob",
        "best_action_model_prob",
        "best_action_repr",
        "best_action_json",
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
        best_action_index: int,
        best_action_repr: str = "",
        best_action_json: str = "",
    ) -> None:
        if not self._path:
            return
        self._ensure()
        assert self._writer is not None

        best_mcts = float(mcts_root_prior[best_action_index]) if best_action_index < len(mcts_root_prior) else 0.0
        best_model = float(model_root_prior[best_action_index]) if best_action_index < len(model_root_prior) else 0.0

        row = {
            "game_id": int(game_id),
            "root_id": int(root_id),
            "root_depth": int(root_depth),
            "root_node_id": int(root_node_id),
            "root_player": str(root_player),
            "num_simulations": int(num_simulations),

            "model_root_value_controller": _safe_float(model_root_value_controller),
            "model_root_prior_json": _j(list(model_root_prior)),
            "normalized_root_prior_json": _j(list(normalized_root_prior)),
            "valid_action_mask_json": _j(list(valid_action_mask)),

            "mcts_root_value_controller": _safe_float(mcts_root_value_controller),
            "mcts_root_prior_json": _j(list(mcts_root_prior)),
            "best_action_index": int(best_action_index),
            "best_action_mcts_prob": _safe_float(best_mcts),
            "best_action_model_prob": _safe_float(best_model),
            "best_action_repr": str(best_action_repr or ""),
            "best_action_json": str(best_action_json or ""),
        }

        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._file.flush()  # type: ignore[union-attr]
