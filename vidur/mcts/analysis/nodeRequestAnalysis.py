"""
nodeRequestSLOAnalysis.py

For a given process (e.g. P7) and controller node_id (e.g. 107), this script:

1. Loads test_mcts_trace_{PROCESS_NAME}.csv.
2. Finds the target tree node and its parent.
3. Walks the ancestor path up to the parent node, and along it:
   - Infers per-request generation info from adversary tree rows:
       * arrival_time (floored sim_time at generation)
       * prefill_slo
       * total prefill tokens
       * total decode tokens
   - Accumulates controller prefill allocations for each request
     up to (and including) the parent node.
4. At the target node, reads controller_selected_ids and
   controller_prefill_allocations, and:
   - Classifies decode-phase requests as:
       decode_ids = selected_ids - keys(controller_prefill_allocations)
   - Computes per-request:
       Remaining prefill BEFORE = total_prefill - prefill_alloc_before
       Remaining prefill AFTER  = total_prefill - (prefill_alloc_before + alloc_at_node)
       SLO lateness BEFORE = parent_sim_time - (arrival_time + prefill_slo)
       SLO lateness AFTER  = node_sim_time   - (arrival_time + prefill_slo)
   for all requests present in state_waiting_ids at the node.

Output: a human-readable report for that node under:
  simulator_output/Parrallel_Launch/analysis/nodeAnalysis/SLO_{PROCESS_NAME}_{TARGET_NODE_ID}.txt
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# === EDIT THESE FOR EACH ANALYSIS ===
PROCESS_NAME = "P7"       # e.g. "P7"
TARGET_NODE_ID = "107"    # e.g. "107"
# ====================================


def parse_json_field(raw: str, default: Any) -> Any:
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def parse_int_list(raw: str) -> List[int]:
    vals = parse_json_field(raw, [])
    try:
        return [int(x) for x in vals]
    except Exception:
        return []


def load_trace(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_node_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Map node_id -> last occurrence of that node_id for 'tree' and 'INITIAL_HISTORY_GEN' phases.
    """
    node_by_id: Dict[str, Dict[str, str]] = {}
    for r in rows:
        phase = r.get("phase", "")
        if phase not in ("tree", "INITIAL_HISTORY_GEN"):
            continue
        nid = str(r.get("node_id", "")).strip()
        if nid:
            node_by_id[nid] = r
    return node_by_id


def build_ancestor_path(
    node_by_id: Dict[str, Dict[str, str]], parent_node_id: str
) -> List[Dict[str, str]]:
    """
    Follow parent_node_id links from the given parent up to the root/history.
    Return the path from topmost ancestor to parent.
    """
    chain_ids: List[str] = []
    cur = parent_node_id
    while cur and cur in node_by_id:
        chain_ids.append(cur)
        parent = node_by_id[cur].get("parent_node_id", "")
        if not parent or parent == "root":
            break
        cur = str(parent)
    chain_ids.reverse()
    return [node_by_id[cid] for cid in chain_ids]


def accumulate_generation_and_prefill_before_parent(
    path_nodes: List[Dict[str, str]],
    parent_node_id: str,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, int]]:
    """
    Along the ancestor path, for nodes up to and including parent, infer:

      req_info[req_id] = {
          "arrival_time": float,
          "prefill_slo": float,
          "prefill_tokens": float,
          "decode_tokens": float,
      }

      prefill_alloc_before[req_id] = total prefill tokens allocated
                                     by controllers up to and including parent.
    """
    req_info: Dict[int, Dict[str, float]] = {}
    prefill_alloc_before: Dict[int, int] = defaultdict(int)

    known_ids: set[int] = set()

    for row in path_nodes:
        nid = str(row.get("node_id", "")).strip()

        # 1) Adversary: infer new requests and their SLOs/tokens
        if row.get("player_to_act") == "adversary":
            waiting_ids = set(parse_int_list(row.get("state_waiting_ids", "")))
            adv_reqs = parse_json_field(row.get("adversary_requests", ""), [])
            new_ids = [rid for rid in waiting_ids if rid not in known_ids]
            new_ids_sorted = sorted(new_ids)
            n = min(len(new_ids_sorted), len(adv_reqs))

            try:
                sim_time = float(row.get("sim_time", "0.0") or 0.0)
            except ValueError:
                sim_time = 0.0
            arrival_time = float(int(sim_time))  # floor

            for idx in range(n):
                rid = new_ids_sorted[idx]
                spec = adv_reqs[idx] or {}
                prefill_tokens = int(spec.get("prefill_tokens", 0) or 0)
                decode_tokens = int(spec.get("decode_tokens", 0) or 0)
                prefill_slo = float(spec.get("prefill_slo", 0.0) or 0.0)
                req_info[rid] = {
                    "arrival_time": arrival_time,
                    "prefill_slo": prefill_slo,
                    "prefill_tokens": float(prefill_tokens),
                    "decode_tokens": float(decode_tokens),
                }

            known_ids |= waiting_ids

        # 2) Controller: accumulate prefill allocations
        if row.get("player_to_act") == "controller":
            pre_alloc_raw = row.get("controller_prefill_allocations", "") or ""
            pre_alloc = parse_json_field(pre_alloc_raw, {})
            if isinstance(pre_alloc, dict):
                for k, v in pre_alloc.items():
                    try:
                        rid = int(k)
                        tokens = int(v)
                    except (ValueError, TypeError):
                        continue
                    prefill_alloc_before[rid] += tokens

        if nid == parent_node_id:
            break

    return req_info, prefill_alloc_before


def main() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3]
    par_dir = repo_root / "simulator_output" / "Parrallel_Launch"

    trace_csv = par_dir / f"test_mcts_trace_{PROCESS_NAME}.csv"
    if not trace_csv.exists():
        print(f"[ERROR] Trace CSV not found: {trace_csv}")
        return

    rows = load_trace(trace_csv)
    node_by_id = build_node_index(rows)

    target_id = str(TARGET_NODE_ID)
    if target_id not in node_by_id:
        print(f"[ERROR] Node {target_id} not found in tree trace.")
        return

    target_row = node_by_id[target_id]
    parent_id = str(target_row.get("parent_node_id", "")).strip()
    if not parent_id or parent_id not in node_by_id:
        print(f"[ERROR] Parent node {parent_id!r} for node {target_id} not found.")
        return
    parent_row = node_by_id[parent_id]

    # Build ancestor path to parent; accumulate generation and prefill BEFORE parent
    path_nodes = build_ancestor_path(node_by_id, parent_id)
    req_info, prefill_alloc_before = accumulate_generation_and_prefill_before_parent(
        path_nodes, parent_id
    )

    # Times
    try:
        parent_time = float(parent_row.get("sim_time", "0.0") or 0.0)
    except ValueError:
        parent_time = 0.0
    try:
        node_time = float(target_row.get("sim_time", "0.0") or 0.0)
    except ValueError:
        node_time = 0.0

    # Requests present at node (system state after controller action)
    waiting_after = parse_int_list(target_row.get("state_waiting_ids", ""))

    # Controller action at node
    selected_ids = parse_int_list(target_row.get("controller_selected_ids", ""))
    pre_alloc_node_raw = target_row.get("controller_prefill_allocations", "") or ""
    pre_alloc_node = parse_json_field(pre_alloc_node_raw, {})
    prefill_ids_node = sorted({int(k) for k in pre_alloc_node.keys()} if isinstance(pre_alloc_node, dict) else [])

    # Decode-phase IDs at node: selected but not receiving prefill here
    decode_ids = sorted(set(selected_ids) - set(prefill_ids_node))

    # Build per-request before/after stats for waiting requests
    per_request_lines: List[str] = []
    for rid in sorted(waiting_after):
        info = req_info.get(rid)
        if not info:
            # Unknown generation; skip detailed stats
            continue

        total_prefill = int(info.get("prefill_tokens", 0.0))
        arrival_time = float(info.get("arrival_time", 0.0))
        prefill_slo = float(info.get("prefill_slo", 0.0))

        before_alloc = prefill_alloc_before.get(rid, 0)
        after_alloc = before_alloc
        if isinstance(pre_alloc_node, dict):
            try:
                after_alloc += int(pre_alloc_node.get(str(rid), 0) or 0)
            except ValueError:
                pass

        rem_prefill_before = max(total_prefill - before_alloc, 0)
        rem_prefill_after = max(total_prefill - after_alloc, 0)

        lateness_before = parent_time - (arrival_time + prefill_slo)
        lateness_after = node_time - (arrival_time + prefill_slo)

        line = (
            f"  Request {rid:3d}: "
            f"Remaining prefill BEFORE = {rem_prefill_before:6d}, "
            f"SLO lateness BEFORE = {lateness_before: .6f} s\n"
            f"                Remaining prefill AFTER  = {rem_prefill_after:6d}, "
            f"SLO lateness AFTER  = {lateness_after: .6f} s"
        )
        per_request_lines.append(line)

    # Output
    node_analysis_dir = par_dir / "analysis" / "nodeAnalysis"
    node_analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = node_analysis_dir / f"SLO_{PROCESS_NAME}_{TARGET_NODE_ID}.txt"

    with out_path.open("w") as out:
        out.write(f"Process: {PROCESS_NAME}\n")
        out.write(f"Target controller node_id: {TARGET_NODE_ID}\n")
        out.write(f"Parent node_id: {parent_id}\n\n")

        out.write(f"Parent sim_time: {parent_time:.6f} s\n")
        out.write(f"Node   sim_time: {node_time:.6f} s\n\n")

        out.write("Decode-phase requests at node (selected but not receiving prefill here):\n")
        out.write(f"  Count: {len(decode_ids)}\n")
        if decode_ids:
            out.write("  IDs:   " + ", ".join(str(r) for r in decode_ids) + "\n")
        out.write("\n")

        out.write("Per-request prefill + SLO status (for requests in system at node):\n")
        if not per_request_lines:
            out.write("  (no detailed info; no matching generation logs for these IDs)\n")
        else:
            out.write("\n".join(per_request_lines))
            out.write("\n")

    print(f"[INFO] Wrote SLO node analysis to {out_path}")


if __name__ == "__main__":
    main()
