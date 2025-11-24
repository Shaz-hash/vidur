"""
High-level algorithm for controllerBestAction.py

1. Locate the Parallel Launch directory:
   - repo_root = this_file.parents[3]
   - par_dir = repo_root / "simulator_output" / "Parrallel_Launch"
   - For each run, use:
       main CSV:  test_mcts_trace_P#.csv
       tree CSV:  test_mcts_trace_tree_P#.csv

2. For each run P#:
   a) Read main CSV (test_mcts_trace_P#.csv) into a list of dicts.
   b) Read tree CSV (test_mcts_trace_tree_P#.csv), find the last iteration
      (max iteration value), and build a mapping:
         node_id -> (visits, parent_visits, node_cost, ucb_score)
      using only rows from that last iteration.

   c) In main CSV, find the last row with phase == "INITIAL_HISTORY_GEN"
      (if any) and write it as the first row in analysis.csv, with
      visits/parent_visits/node_cost/ucb_score set to 0/empty.

   d) From all tree-phase rows in main CSV (phase == "tree"), group children
      by parent_node_id and player_to_act:
         parent_node_id -> {"adversary": [...], "controller": [...]}

      - Adversary candidate:
          parent P where:
            * there are exactly 10 children with player_to_act == "adversary"
            * at least one of those has non-empty adversary_requests
          depth for this group = min(child.depth)

      - Controller candidate:
          parent P where:
            * there are exactly 48 children with player_to_act == "controller"
          depth for this group = min(child.depth)

      Choose the topmost layer:
        * If any adversary candidates exist, pick the one with smallest depth.
        * Else if any controller candidates exist, pick the one with smallest depth.
        * Else: no useful layer; write only history row.

   e) If adversary-case was chosen:
        - Write all adversary children (10 rows) under that parent:
            * For each child row, attach visits/parent_visits/node_cost/ucb_score
              from tree CSV (last iteration), and write to analysis.csv.
        - For each adversary child, also write all of its controller children
          (phase == "tree", player_to_act == "controller", parent_node_id == child.node_id),
          again attaching tree CSV stats from last iteration.

      If controller-case was chosen:
        - Write only those controller children (up to 48) for that parent,
          with tree CSV stats attached.

3. For each run, create:
   simulator_output/Parrallel_Launch/analysis/P#/analysis.csv
   with columns:
     iteration,phase,depth,parent_node_id,node_id,player_to_act,next_player,
     visits,parent_visits,node_cost,ucb_score,
     sim_time,requests_in_system,requests_generated,requests_completed,
     slo_violations,avg_lateness,objective_cost,
     state_waiting_ids,state_completed_request_ids,
     adversary_requests,adversary_prefill_slos,adversary_decode_slos,
     controller_token_budget,controller_selected_ids,
     controller_allocations,controller_prefill_allocations,
     controller_decode_allocations,controller_prefill_total,
     controller_decode_total,controller_heuristic,controller_strategy
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ANALYSIS_FIELDS = [
    "iteration",
    "phase",
    "depth",
    "parent_node_id",
    "node_id",
    "player_to_act",
    "next_player",
    # Tree stats:
    "visits",
    "parent_visits",
    "node_cost",
    "ucb_score",
    # Main state fields:
    "sim_time",
    "requests_in_system",
    "requests_generated",
    "requests_completed",
    "slo_violations",
    "avg_lateness",
    "objective_cost",
    "state_waiting_ids",
    "state_completed_request_ids",
    "adversary_requests",
    "adversary_prefill_slos",
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


# def load_csv_rows(path: Path) -> List[Dict[str, str]]:
#     if not path.exists():
#         return []
#     with path.open("r", newline="") as f:
#         reader = csv.DictReader(f)
#         return list(reader)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    def _clean_lines(f):
        for line in f:
            # Remove any embedded NULs before csv.DictReader sees the line
            yield line.replace("\x00", "")

    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(_clean_lines(f))
        return list(reader)



def build_tree_stats_by_node(tree_path: Path) -> Dict[str, Dict[str, str]]:
    """
    From test_mcts_trace_tree_P#.csv:
      - find the maximum iteration
      - for that iteration, build: node_id -> row dict
    """
    rows = load_csv_rows(tree_path)
    if not rows:
        return {}

    # Filter rows with numeric iteration
    valid = []
    for r in rows:
        it = r.get("iteration", "").strip()
        if it.isdigit():
            valid.append(r)
    if not valid:
        return {}

    max_iter = max(int(r["iteration"]) for r in valid)
    last_rows = [r for r in valid if int(r["iteration"]) == max_iter]

    by_node: Dict[str, Dict[str, str]] = {}
    for r in last_rows:
        node_id = str(r.get("node_id", ""))
        by_node[node_id] = r
    return by_node


def get_last_history_row(main_rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Return the last row with phase == 'INITIAL_HISTORY_GEN', or None.
    """
    history_rows = [r for r in main_rows if r.get("phase") == "INITIAL_HISTORY_GEN"]
    return history_rows[-1] if history_rows else None


def group_tree_children(main_rows: List[Dict[str, str]]):
    """
    Group tree-phase rows by parent_node_id and player_to_act.
    Returns:
      children_by_parent: parent_id -> {"adversary": [...], "controller": [...]}
      all_tree_rows: list of tree rows (for later filtering)
    """
    children_by_parent: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: {"adversary": [], "controller": []}
    )
    tree_rows: List[Dict[str, str]] = []

    for r in main_rows:
        if r.get("phase") != "tree":
            continue
        player = r.get("player_to_act", "")
        parent = str(r.get("parent_node_id", ""))
        tree_rows.append(r)
        if player == "adversary":
            children_by_parent[parent]["adversary"].append(r)
        elif player == "controller":
            children_by_parent[parent]["controller"].append(r)

    return children_by_parent, tree_rows


def has_nonempty_adversary_requests(row: Dict[str, str]) -> bool:
    val = row.get("adversary_requests", "").strip()
    return bool(val) and val != "[]"


def has_nontrivial_controller_action(row: Dict[str, str]) -> bool:
    """Return True if this controller tree row actually allocates tokens."""
    tb = (row.get("controller_token_budget") or "").strip()
    try:
        budget = int(tb)
    except ValueError:
        budget = 0

    if budget > 0:
        return True

    pre = (row.get("controller_prefill_allocations") or "").strip()
    dec = (row.get("controller_decode_allocations") or "").strip()

    if pre and pre not in ("{}", "[]"):
        return True
    if dec and dec not in ("{}", "[]"):
        return True
    return False



def find_top_layer(
    main_rows: List[Dict[str, str]],
) -> Tuple[Optional[str], Optional[str], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Determine the topmost useful layer.

    Returns:
      (mode, parent_id, selected_children, all_tree_rows)
      where mode is "adversary", "controller", or None.
    """
    children_by_parent, tree_rows = group_tree_children(main_rows)

    adv_candidates: List[Tuple[float, str, List[Dict[str, str]]]] = []
    ctrl_candidates: List[Tuple[float, str, List[Dict[str, str]]]] = []

    for parent, grp in children_by_parent.items():
        adv_children = grp.get("adversary", [])
        ctrl_children = grp.get("controller", [])

        # Adversary fully expanded: exactly 10 children, and at least one generates new requests
        if len(adv_children) == 10 and any(
            has_nonempty_adversary_requests(r) for r in adv_children
        ):
            try:
                depth = min(float(r.get("depth", "0") or 0.0) for r in adv_children)
            except ValueError:
                depth = 0.0
            adv_candidates.append((depth, parent, adv_children))

        # Controller fully expanded: 48 children, and at least one with a real allocation
        if len(ctrl_children) == 48 and any(
            has_nontrivial_controller_action(r) for r in ctrl_children
        ):
            try:
                depth = min(float(r.get("depth", "0") or 0.0) for r in ctrl_children)
            except ValueError:
                depth = 0.0
            ctrl_candidates.append((depth, parent, ctrl_children))

    if not adv_candidates and not ctrl_candidates:
        return None, None, [], tree_rows

    combined: List[Tuple[float, str, str, List[Dict[str, str]]]] = []
    for depth, parent, children in adv_candidates:
        combined.append((depth, "adversary", parent, children))
    for depth, parent, children in ctrl_candidates:
        combined.append((depth, "controller", parent, children))

    # Pick the earliest layer across BOTH adversary and controller
    depth, mode, parent_id, children = min(combined, key=lambda t: t[0])
    return mode, parent_id, children, tree_rows

def build_analysis_row(
    main_row: Dict[str, str],
    tree_row: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Merge fields from main CSV row and (optional) tree CSV row into analysis row.
    """
    visits = parent_visits = node_cost = ucb_score = ""
    if tree_row is not None:
        visits = tree_row.get("visits", "").strip()
        parent_visits = tree_row.get("parent_visits", "").strip()
        node_cost = tree_row.get("node_cost", "").strip()
        ucb_score = tree_row.get("ucb_score", "").strip()

    out = {field: "" for field in ANALYSIS_FIELDS}

    # Core fields from main_row
    for key in [
        "iteration",
        "phase",
        "depth",
        "parent_node_id",
        "node_id",
        "player_to_act",
        "next_player",
        "sim_time",
        "requests_in_system",
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "avg_lateness",
        "objective_cost",
        "state_waiting_ids",
        "state_completed_request_ids",
        "adversary_requests",
        "adversary_prefill_slos",
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
    ]:
        if key in main_row:
            out[key] = main_row.get(key, "")

    # Tree stats
    out["visits"] = visits or "0"
    out["parent_visits"] = parent_visits or "0"
    out["node_cost"] = node_cost or "0"
    out["ucb_score"] = ucb_score or "0"

    return out


def analyze_run(main_csv: Path, tree_csv: Path, out_csv: Path) -> None:
    main_rows = load_csv_rows(main_csv)
    if not main_rows:
        return

    tree_stats_by_node = build_tree_stats_by_node(tree_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_FIELDS)
        writer.writeheader()

        # 1) Last history row (INITIAL_HISTORY_GEN), if any
        history_row = get_last_history_row(main_rows)
        if history_row is not None:
            analysis_row = build_analysis_row(history_row, tree_row=None)
            writer.writerow(analysis_row)

        # 2) Find topmost useful layer (adversary or controller)
        mode, parent_id, selected_children, tree_rows = find_top_layer(main_rows)
        if mode is None or not selected_children:
            return  # nothing more to write

        # Sort children deterministically (by node_id as int when possible)
        def sort_key(r: Dict[str, str]):
            nid = r.get("node_id", "")
            try:
                return int(nid)
            except ValueError:
                return nid

        selected_children = sorted(selected_children, key=sort_key)

        if mode == "adversary":
            # For each adversary child, write adversary row + its controller children
            # (controller children from the immediately deeper layer).
            # Build quick index for tree_rows by (parent_node_id, player_to_act).
            children_index: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(
                list
            )
            for r in tree_rows:
                if r.get("phase") != "tree":
                    continue
                parent = str(r.get("parent_node_id", ""))
                player = r.get("player_to_act", "")
                children_index[(parent, player)].append(r)

            for adv_row in selected_children:
                adv_node_id = str(adv_row.get("node_id", ""))
                adv_tree_row = tree_stats_by_node.get(adv_node_id)
                writer.writerow(build_analysis_row(adv_row, adv_tree_row))

                # Controller children under this adversary node
                ctrl_children = children_index.get((adv_node_id, "controller"), [])
                ctrl_children = sorted(ctrl_children, key=sort_key)
                for ctrl_row in ctrl_children:
                    ctrl_node_id = str(ctrl_row.get("node_id", ""))
                    ctrl_tree_row = tree_stats_by_node.get(ctrl_node_id)
                    writer.writerow(build_analysis_row(ctrl_row, ctrl_tree_row))

        elif mode == "controller":
            # Only controller children for this parent, no adversary children.
            for ctrl_row in selected_children:
                ctrl_node_id = str(ctrl_row.get("node_id", ""))
                ctrl_tree_row = tree_stats_by_node.get(ctrl_node_id)
                writer.writerow(build_analysis_row(ctrl_row, ctrl_tree_row))


def main() -> None:
    this_file = Path(__file__).resolve()
    # .../vidur/vidur/mcts/analysis/controllerBestAction.py
    # repo_root is the outer "vidur" that contains simulator_output/
    repo_root = this_file.parents[3]
    par_dir = repo_root / "simulator_output" / "Parrallel_Launch"
    analysis_root = par_dir / "analysis"

    if not par_dir.exists():
        print(f"[WARN] Parallel launch directory not found: {par_dir}")
        return

    # Find all test_mcts_trace_P*.csv runs
    main_files = sorted(par_dir.glob("test_mcts_trace_P*.csv"))
    if not main_files:
        print(f"[WARN] No test_mcts_trace_P*.csv files in {par_dir}")
        return

    for main_csv in main_files:
        stem = main_csv.stem  # e.g., "test_mcts_trace_P1"
        # Extract run id after last "_P"
        if "_P" not in stem:
            continue
        run_id = stem.split("_P", 1)[1]
        tree_csv = par_dir / f"test_mcts_trace_tree_P{run_id}.csv"
        if not tree_csv.exists():
            print(f"[WARN] Tree CSV missing for run {run_id}: {tree_csv}")
            continue

        out_dir = analysis_root / f"P{run_id}"
        out_csv = out_dir / "analysis.csv"
        print(f"[INFO] Analyzing run P{run_id} -> {out_csv}")
        analyze_run(main_csv, tree_csv, out_csv)


if __name__ == "__main__":
    main()
