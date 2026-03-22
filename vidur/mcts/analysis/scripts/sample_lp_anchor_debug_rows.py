from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from vidur.mcts.linear.sampler.storage import read_parquet_records


ITER_FIELDS: Sequence[str] = (
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
    "state_position",
)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _load_tables(merged_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "anchors": read_parquet_records(merged_dir / "anchors.parquet"),
        "states": read_parquet_records(merged_dir / "states.parquet"),
        "actions": read_parquet_records(merged_dir / "actions.parquet"),
        "transitions": read_parquet_records(merged_dir / "transitions.parquet"),
        "nodes": read_parquet_records(merged_dir / "nodes.parquet"),
    }


def _pick_random_anchor_state_ids(
    anchors: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    by_player: Dict[str, List[str]] = {"controller": [], "adversary": []}

    seen: Dict[str, set[str]] = {"controller": set(), "adversary": set()}
    for row in anchors:
        player = str(row.get("player_to_act", "")).strip()
        if player not in by_player:
            continue
        sid = str(row.get("anchor_state_id", "")).strip()
        if not sid or sid in seen[player]:
            continue
        by_player[player].append(sid)
        seen[player].add(sid)

    out: Dict[str, str] = {}
    for player in ("adversary", "controller"):
        choices = by_player[player]
        if not choices:
            continue
        out[player] = rng.choice(choices)
    return out


def _build_node_context(nodes: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # Pick one stable representative node per state for game/root context.
    grouped: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        sid = str(n.get("state_id", "")).strip()
        if not sid:
            continue
        cur_key = (
            _to_int(n.get("game_id", 0)),
            _to_int(n.get("root_id", 0)),
            _to_int(n.get("step_idx", 0)),
            str(n.get("node_id", "")),
        )
        prev = grouped.get(sid)
        if prev is None:
            grouped[sid] = dict(n)
            grouped[sid]["_key"] = cur_key
            continue
        if cur_key < prev["_key"]:
            grouped[sid] = dict(n)
            grouped[sid]["_key"] = cur_key
    for sid in list(grouped.keys()):
        grouped[sid].pop("_key", None)
    return grouped


def _iter_anchor_rows(
    *,
    anchor_state_id: str,
    root_player: str,
    state_by_id: Mapping[str, Mapping[str, Any]],
    action_by_id: Mapping[str, Mapping[str, Any]],
    transitions_by_state: Mapping[str, List[Mapping[str, Any]]],
    node_ctx_by_state: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    anchor_state = state_by_id.get(anchor_state_id)
    if anchor_state is None:
        return out

    anchor_ctx = node_ctx_by_state.get(anchor_state_id, {})
    anchor_node_id = str(anchor_state_id)
    root_node_id = str(anchor_state_id)
    game_id = _to_int(anchor_ctx.get("game_id", 0))
    root_id = _to_int(anchor_ctx.get("root_id", 0))

    if anchor_ctx:
        anchor_node_id = str(anchor_ctx.get("node_id", anchor_node_id))
        root_node_id = str(anchor_ctx.get("root_id", root_node_id))

    out.append(
        {
            "game_id": game_id,
            "root_id": root_id,
            "root_node_id": root_node_id,
            "root_player": root_player,
            "node_id": anchor_node_id,
            "parent_node_id": "",
            "node_depth": _to_int(anchor_state.get("branching_depth", 0)),
            "sim_time": _to_float(anchor_state.get("sim_time", 0.0)),
            "advanced_sim_time": _to_float(anchor_state.get("sim_time", 0.0)),
            "player_acted_to_create_this_node": "",
            "player_to_act_in_this_node": str(anchor_state.get("player_to_act", "")),
            "action_index": "",
            "action_repr": "",
            "phase": "anchor",
            "nn_called": False,
            "model_prior_json": "[]",
            "normalized_prior_json": "[]",
            "state_waiting_ids": str(anchor_state.get("waiting_request_ids_json", "[]")),
            "state_completed_request_ids": str(anchor_state.get("completed_request_ids_json", "[]")),
            "adversary_prefill_deadlines_by_id": "{}",
            "slo_violations": _to_int(anchor_state.get("slo_violations", 0)),
            "avg_lateness": _to_float(anchor_state.get("total_lateness", 0.0)),
            "objective_cost": _to_float(anchor_state.get("total_cost", 0.0)),
            "state_position": "anchor",
        }
    )

    transitions = list(transitions_by_state.get(anchor_state_id, []))
    transitions.sort(
        key=lambda t: (
            _to_int(action_by_id.get(str(t.get("action_id", "")), {}).get("canonical_index", 10**9)),
            str(t.get("transition_id", "")),
        )
    )

    for t in transitions:
        action_id = str(t.get("action_id", ""))
        next_state_id = str(t.get("next_state_id", ""))
        action = action_by_id.get(action_id, {})
        child_state = state_by_id.get(next_state_id, {})
        child_ctx = node_ctx_by_state.get(next_state_id, {})

        node_id = str(next_state_id)
        if child_ctx:
            node_id = str(child_ctx.get("node_id", node_id))

        out.append(
            {
                "game_id": game_id,
                "root_id": root_id,
                "root_node_id": root_node_id,
                "root_player": root_player,
                "node_id": node_id,
                "parent_node_id": anchor_node_id,
                "node_depth": _to_int(child_state.get("branching_depth", _to_int(t.get("next_branching_depth", 0)))),
                "sim_time": _to_float(t.get("sim_time_after_action", child_state.get("sim_time", 0.0))),
                "advanced_sim_time": _to_float(t.get("sim_time_after_advance", child_state.get("sim_time", 0.0))),
                "player_acted_to_create_this_node": str(action.get("actor", t.get("actor", ""))),
                "player_to_act_in_this_node": str(child_state.get("player_to_act", "")),
                "action_index": _to_int(action.get("canonical_index", -1)),
                "action_repr": str(action.get("action_repr", "")),
                "phase": "canonical",
                "nn_called": False,
                "model_prior_json": "[]",
                "normalized_prior_json": "[]",
                "state_waiting_ids": str(child_state.get("waiting_request_ids_json", "[]")),
                "state_completed_request_ids": str(child_state.get("completed_request_ids_json", "[]")),
                "adversary_prefill_deadlines_by_id": str(t.get("adversary_prefill_deadlines_by_id_json", "{}")),
                "slo_violations": _to_int(child_state.get("slo_violations", 0)),
                "avg_lateness": _to_float(child_state.get("total_lateness", 0.0)),
                "objective_cost": _to_float(child_state.get("total_cost", 0.0)),
                "state_position": "canonical",
            }
        )
    return out


def build_debug_rows(merged_dir: Path, *, seed: int) -> List[Dict[str, Any]]:
    tables = _load_tables(merged_dir)
    anchors = tables["anchors"]
    states = tables["states"]
    actions = tables["actions"]
    transitions = tables["transitions"]
    nodes = tables["nodes"]

    picks = _pick_random_anchor_state_ids(anchors, seed=seed)
    if not picks:
        raise RuntimeError("no controller/adversary anchors found in anchors.parquet")

    state_by_id = {str(r.get("state_id", "")): r for r in states}
    action_by_id = {str(r.get("action_id", "")): r for r in actions}
    transitions_by_state: Dict[str, List[Dict[str, Any]]] = {}
    for t in transitions:
        sid = str(t.get("state_id", ""))
        if sid:
            transitions_by_state.setdefault(sid, []).append(t)
    node_ctx_by_state = _build_node_context(nodes)

    rows: List[Dict[str, Any]] = []
    for player in ("adversary", "controller"):
        sid = picks.get(player)
        if not sid:
            continue
        rows.extend(
            _iter_anchor_rows(
                anchor_state_id=sid,
                root_player=player,
                state_by_id=state_by_id,
                action_by_id=action_by_id,
                transitions_by_state=transitions_by_state,
                node_ctx_by_state=node_ctx_by_state,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Sample one adversary and one controller anchor from merged LP tables and "
            "write mcts_iter-style debug rows (anchor + canonical) with state_position."
        )
    )
    ap.add_argument("--merged-dir", type=str, required=True, help="Path to round_XXX/merged")
    ap.add_argument(
        "--out-csv",
        type=str,
        default="simulator_output/linear_lp_solution/lp_anchor_debug_samples.csv",
        help="Output CSV path",
    )
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for anchor sampling")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    merged_dir = Path(args.merged_dir)
    if not merged_dir.exists():
        raise FileNotFoundError(f"merged dir does not exist: {merged_dir}")

    rows = build_debug_rows(merged_dir, seed=int(args.seed))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ITER_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Also write a tiny JSON summary for quick context when eyeballing.
    summary = {
        "num_rows": int(len(rows)),
        "num_anchor_rows": int(sum(1 for r in rows if str(r.get("state_position", "")) == "anchor")),
        "num_canonical_rows": int(sum(1 for r in rows if str(r.get("state_position", "")) == "canonical")),
        "out_csv": str(out_csv),
    }
    summary_path = out_csv.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(
        f"[lp.debug] wrote rows={len(rows)} csv={out_csv} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
