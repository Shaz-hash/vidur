from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .storage import read_parquet_records


@dataclass(frozen=True)
class CompatExportPaths:
    mcts_root_compat_csv: str
    mcts_iter_compat_csv: str


def _safe_json_list(cell: str) -> List[int]:
    try:
        val = json.loads(cell or "[]")
    except Exception:
        return []
    if not isinstance(val, list):
        return []
    out: List[int] = []
    for x in val:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _build_numeric_node_ids(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    ordered = sorted(
        nodes,
        key=lambda n: (
            int(n.get("game_id", 0)),
            int(n.get("root_id", 0)),
            int(n.get("branching_depth", 0)),
            str(n.get("node_id", "")),
        ),
    )
    out: Dict[str, int] = {}
    for idx, n in enumerate(ordered):
        out[str(n["node_id"])] = int(idx)
    return out


def _compute_root_node_by_id(nodes: List[Dict[str, Any]]) -> Dict[str, str]:
    node_by_id: Dict[str, Dict[str, Any]] = {str(n.get("node_id", "")): n for n in nodes}
    root_for: Dict[str, str] = {}
    for node_id in node_by_id.keys():
        cur = node_id
        guard = 0
        while guard < 100000:
            guard += 1
            n = node_by_id.get(cur)
            if n is None:
                break
            parent_id = str(n.get("parent_node_id", "") or "")
            if not parent_id:
                root_for[node_id] = cur
                break
            cur = parent_id
        if node_id not in root_for:
            root_for[node_id] = node_id
    return root_for


def export_compat_csvs(*, merged_dir: Path, out_dir: Path) -> CompatExportPaths:
    out_dir.mkdir(parents=True, exist_ok=True)

    states = read_parquet_records(merged_dir / "states.parquet")
    actions = read_parquet_records(merged_dir / "actions.parquet")
    transitions = read_parquet_records(merged_dir / "transitions.parquet")
    nodes = read_parquet_records(merged_dir / "nodes.parquet")

    state_by_id = {str(r["state_id"]): r for r in states}
    action_by_id = {str(r["action_id"]): r for r in actions}
    node_by_id = {str(r["node_id"]): r for r in nodes}
    trans_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for t in transitions:
        trans_by_key[(str(t["state_id"]), str(t["action_id"]), str(t["next_state_id"]))] = t

    node_num = _build_numeric_node_ids(nodes)
    root_node_by_id = _compute_root_node_by_id(nodes)

    iter_path = out_dir / "mcts_iter_compat.csv"
    root_path = out_dir / "mcts_root_compat.csv"

    iter_fields = [
        "game_id",
        "root_id",
        "root_node_id",
        "root_player",
        "node_id",
        "parent_node_id",
        "node_depth",
        "sim_time",
        "advanced_sim_time",
        "player_acted_to_create_this_node",
        "player_to_act_in_this_node",
        "action_index",
        "action_repr",
        "phase",
        "nn_called",
        "model_prior_json",
        "normalized_prior_json",
        "state_waiting_ids",
        "state_completed_request_ids",
        "adversary_prefill_deadlines_by_id",
        "slo_violations",
        "avg_lateness",
        "objective_cost",
    ]

    root_fields = [
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
        "phase",
        "cycle_label",
        "sim_time",
        "slo_violations",
        "total_lateness",
        "total_cost",
    ]

    rows_for_iter: List[Dict[str, Any]] = []
    rows_for_root: List[Dict[str, Any]] = []

    nodes_sorted = sorted(
        nodes,
        key=lambda n: (
            int(n.get("game_id", 0)),
            int(n.get("root_id", 0)),
            int(n.get("branching_depth", 0)),
            str(n.get("node_id", "")),
        ),
    )

    # Root player per trace comes from the first acted row in that trace.
    root_player_by_trace: Dict[str, str] = {}
    for n in nodes_sorted:
        incoming_action_id = str(n.get("incoming_action_id", ""))
        if not incoming_action_id:
            continue
        trace = str(n.get("trace_id", ""))
        action = action_by_id.get(incoming_action_id)
        if action is None:
            continue
        root_player_by_trace.setdefault(trace, str(action.get("actor", "")))

    for n in nodes_sorted:
        incoming_action_id = str(n.get("incoming_action_id", ""))
        if not incoming_action_id:
            continue

        state_id = str(n.get("state_id", ""))
        state = state_by_id.get(state_id)
        action = action_by_id.get(incoming_action_id)
        parent_id = str(n.get("parent_node_id", ""))
        parent = node_by_id.get(parent_id) if parent_id else None
        if state is None or action is None:
            continue

        parent_state_id = str(parent.get("state_id", "")) if parent is not None else ""
        tkey = (parent_state_id, incoming_action_id, state_id)
        trans = trans_by_key.get(tkey)
        deadlines_json = "{}" if trans is None else str(trans.get("adversary_prefill_deadlines_by_id_json", "{}"))
        sim_time_after_action = float(state.get("sim_time", 0.0))
        sim_time_after_advance = float(state.get("sim_time", 0.0))
        if trans is not None:
            sim_time_after_action = float(trans.get("sim_time_after_action", sim_time_after_action))
            sim_time_after_advance = float(trans.get("sim_time_after_advance", sim_time_after_advance))

        trace_id = str(n.get("trace_id", ""))
        root_player = root_player_by_trace.get(trace_id, str(action.get("actor", "")))
        waiting_json = str(state.get("waiting_request_ids_json", "[]"))
        completed_json = str(state.get("completed_request_ids_json", "[]"))

        rows_for_iter.append(
            {
                "game_id": int(n.get("game_id", 0)),
                "root_id": int(n.get("root_id", 0)),
                "root_node_id": int(node_num.get(root_node_by_id.get(str(n["node_id"]), str(n["node_id"])), 0)),
                "root_player": root_player,
                "node_id": int(node_num[str(n["node_id"])]),
                "parent_node_id": "" if not parent_id else int(node_num.get(parent_id, 0)),
                "node_depth": int(n.get("branching_depth", 0)),
                "sim_time": sim_time_after_action,
                "advanced_sim_time": sim_time_after_advance,
                "player_acted_to_create_this_node": str(action.get("actor", "")),
                "player_to_act_in_this_node": str(state.get("player_to_act", "")),
                "action_index": int(action.get("canonical_index", -1)),
                "action_repr": str(action.get("action_repr", "")),
                "phase": "expand",
                "nn_called": False,
                "model_prior_json": "[]",
                "normalized_prior_json": "[]",
                "state_waiting_ids": waiting_json,
                "state_completed_request_ids": completed_json,
                "adversary_prefill_deadlines_by_id": deadlines_json,
                "slo_violations": int(state.get("slo_violations", 0)),
                "avg_lateness": float(state.get("total_lateness", 0.0)),
                "objective_cost": float(state.get("total_cost", 0.0)),
            }
        )

        waiting_ids = _safe_json_list(waiting_json)
        rows_for_root.append(
            {
                "game_id": int(n.get("game_id", 0)),
                "root_id": int(node_num[str(n["node_id"])]),
                "root_depth": int(n.get("branching_depth", 0)),
                "root_node_id": int(node_num[str(n["node_id"])]),
                "root_player": str(action.get("actor", "")),
                "num_simulations": 1,
                "model_root_value_controller": 0.0,
                "model_root_prior_json": "[]",
                "normalized_root_prior_json": "[]",
                "valid_action_mask_json": json.dumps([True for _ in waiting_ids], ensure_ascii=False),
                "mcts_root_value_controller": 0.0,
                "mcts_root_prior_json": "[]",
                "best_action_index": int(action.get("canonical_index", -1)),
                "best_action_mcts_prob": 1.0,
                "best_action_model_prob": 0.0,
                "best_action_repr": str(action.get("action_repr", "")),
                "best_action_json": str(action.get("action_json", "{}")),
                "phase": "train_root_applied",
                "cycle_label": "linear_sampler",
                "sim_time": float(state.get("sim_time", 0.0)),
                "slo_violations": int(state.get("slo_violations", 0)),
                "total_lateness": float(state.get("total_lateness", 0.0)),
                "total_cost": float(state.get("total_cost", 0.0)),
            }
        )

    with iter_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=iter_fields)
        w.writeheader()
        for row in rows_for_iter:
            w.writerow(row)

    with root_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=root_fields)
        w.writeheader()
        for row in rows_for_root:
            w.writerow(row)

    return CompatExportPaths(
        mcts_root_compat_csv=str(root_path),
        mcts_iter_compat_csv=str(iter_path),
    )
