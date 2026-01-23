"""
Extract a single root->leaf trace from mcts_iter.csv with:
  - adversary action_index == 5 AND it sent 6 requests, and
  - minimum slo_violations (best), and
  - longest path length (tie-break), and
  - random choice if still tied (seeded).

Writes a CSV in this folder with the same headers as mcts_iter.csv.

NOTE:
This script outputs the trace starting at the FIRST adversary node on the
root->leaf path that matches (action_index==5 and >=6 requests). The output
trace therefore begins with that adversary action node (not the overall root).

Run:
  python3 -m vidur.mcts.tests.DNN_TESTS.test_extract_adv5_min_viol_trace \
    --log_path vidur/simulator_output/mcts_dnn_logs/mcts_iter.csv \
    --seed 0
"""

from __future__ import annotations

import argparse
import ast
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _as_int(x: str) -> Optional[int]:
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _default_log_path() -> Path:
    candidates = [
        Path("vidur/simulator_output/mcts_dnn_logs/mcts_iter.csv"),
        Path("simulator_output/mcts_dnn_logs/mcts_iter.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def adversary_sent_prefill(row: Dict[str, str]) -> bool:
    if (row.get("player_acted_to_create_this_node") or "").strip() != "adversary":
        return False
    action_repr = (row.get("action_repr") or "").strip()
    if "AdversaryAction" not in action_repr:
        return False
    if "requests=[]" in action_repr:
        return False
    return "requests=[" in action_repr


def _adversary_request_count(row: Dict[str, str]) -> int:
    if not adversary_sent_prefill(row):
        return 0
    action_repr = (row.get("action_repr") or "").strip()
    n = action_repr.count("AdversaryRequestSpec(")
    if n > 0:
        return int(n)

    # fallback: parse adversary_requests column if present
    adv_ids = (row.get("adversary_requests") or "").strip()
    if adv_ids:
        try:
            v = ast.literal_eval(adv_ids)
            if isinstance(v, list):
                return int(len(v))
        except Exception:
            return 0
    return 0


def adversary_sent_prefill_action_index_5_six_requests(row: Dict[str, str]) -> bool:
    if not adversary_sent_prefill(row):
        return False
    if _as_int(row.get("action_index", "")) != 5:
        return False
    return _adversary_request_count(row) >= 6


def load_rows(log_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"No headers found in {log_path}")
    return fieldnames, rows


def build_tree(
    rows: List[Dict[str, str]],
    *,
    game_id: Optional[int],
    root_id: Optional[int],
):
    # Build trees per (game_id, root_id). node_id is NOT unique across multiple runs.
    groups: Dict[Tuple[int, int], List[Dict[str, str]]] = {}
    for r in rows:
        gid = _as_int(r.get("game_id", ""))
        rid = _as_int(r.get("root_id", ""))
        if gid is None or rid is None:
            continue
        if game_id is not None and gid != game_id:
            continue
        if root_id is not None and rid != root_id:
            continue
        groups.setdefault((gid, rid), []).append(r)

    out: Dict[Tuple[int, int], Tuple[Dict[int, Dict[str, str]], Dict[int, Optional[int]], Dict[int, List[int]]]] = {}

    def better(existing: Dict[str, str], candidate: Dict[str, str]) -> bool:
        # prefer a row where adversary actually sent requests (best action_repr)
        if not adversary_sent_prefill(existing) and adversary_sent_prefill(candidate):
            return True
        # otherwise keep first-seen by sim_iteration
        ei = _as_int(existing.get("sim_iteration", "")) or 10**18
        ci = _as_int(candidate.get("sim_iteration", "")) or 10**18
        return ci < ei

    for key, grp in groups.items():
        node_row: Dict[int, Dict[str, str]] = {}
        parent_of: Dict[int, Optional[int]] = {}
        children_of: Dict[int, List[int]] = {}

        for r in grp:
            nid = _as_int(r.get("node_id", ""))
            if nid is None:
                continue

            if nid not in node_row or better(node_row[nid], r):
                node_row[nid] = r

            pid = _as_int(r.get("parent_node_id", ""))
            parent_of.setdefault(nid, pid)

            if pid is not None:
                children_of.setdefault(pid, [])
                if nid not in children_of[pid]:
                    children_of[pid].append(nid)

        out[key] = (node_row, parent_of, children_of)

    return out


def path_to_root(leaf_id: int, parent_of: Dict[int, Optional[int]]) -> List[int]:
    seen = set()
    out: List[int] = []
    cur = leaf_id
    while True:
        if cur in seen:
            break
        seen.add(cur)
        out.append(cur)
        p = parent_of.get(cur)
        if p is None:
            break
        cur = p
    out.reverse()
    return out


def path_violations(path: List[int], node_row: Dict[int, Dict[str, str]]) -> int:
    # slo_violations is cumulative in our logs; using max along the path is robust.
    v_max: Optional[int] = None
    for nid in path:
        r = node_row.get(nid)
        if not r:
            continue
        v = _as_int(r.get("slo_violations", ""))
        if v is None:
            continue
        v_max = v if v_max is None else max(v_max, v)
    return v_max if v_max is not None else 10**9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_path", type=str, default=None)
    ap.add_argument("--out_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game_id", type=int, default=None)
    ap.add_argument("--root_id", type=int, default=None)
    args = ap.parse_args()

    log_path = Path(args.log_path) if args.log_path else _default_log_path()
    if not log_path.exists():
        raise SystemExit(f"log_path not found: {log_path}")

    fieldnames, rows = load_rows(log_path)

    trees = build_tree(rows, game_id=args.game_id, root_id=args.root_id)
    if not trees:
        raise SystemExit("No rows/nodes found after filtering.")

    scored: List[Tuple[int, int, int, Tuple[int, int], List[int]]] = []
    # (violations, -len, adv5_six_count, (gid,rid), path)
    for key, (node_row, parent_of, children_of) in trees.items():
        if not node_row:
            continue
        leaves = [nid for nid in node_row.keys() if nid not in children_of]
        for leaf in leaves:
            path = path_to_root(leaf, parent_of)
            start_idx: Optional[int] = None
            for i, nid in enumerate(path):
                r = node_row.get(nid)
                if r and adversary_sent_prefill_action_index_5_six_requests(r):
                    start_idx = i
                    break
            if start_idx is None:
                continue

            # output trace starts at the first matching adversary node
            subpath = path[start_idx:]

            adv5_six = 0
            for nid in subpath:
                r = node_row.get(nid)
                if r and adversary_sent_prefill_action_index_5_six_requests(r):
                    adv5_six += 1

            viol = path_violations(subpath, node_row)
            scored.append((viol, -len(subpath), adv5_six, key, subpath))

    if not scored:
        raise SystemExit(
            "No root->leaf trace contains an adversary prefill send with action_index=5 and >=6 requests."
        )

    # 1) minimize violations
    min_viol = min(v for v, _, _, _, _ in scored)
    cand1 = [(v, nlen, c, key, p) for (v, nlen, c, key, p) in scored if v == min_viol]

    # 2) tie-break: longest path
    best_nlen = min(nlen for _, nlen, _, _, _ in cand1)  # most negative => longest
    cand2 = [(v, nlen, c, key, p) for (v, nlen, c, key, p) in cand1 if nlen == best_nlen]

    # 3) tie-break: largest adv5_count
    best_c = max(c for _, _, c, _, _ in cand2)
    cand3 = [(v, nlen, c, key, p) for (v, nlen, c, key, p) in cand2 if c == best_c]

    rng = random.Random(int(args.seed))
    chosen_viol, chosen_nlen, chosen_adv5_count, chosen_key, chosen_path = rng.choice(cand3)
    chosen_node_row, _chosen_parent_of, _chosen_children_of = trees[chosen_key]

    trace_rows: List[Dict[str, str]] = [
        chosen_node_row[nid] for nid in chosen_path if nid in chosen_node_row
    ]

    out_dir = Path(__file__).resolve().parent
    out_path = Path(args.out_path) if args.out_path else (out_dir / "extracted_trace_adv5_min_viol.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trace_rows:
            w.writerow(r)

    print(f"[OK] log_path={log_path}")
    print(f"[OK] wrote trace to {out_path}")
    print(f"[OK] game_id={chosen_key[0]} root_id={chosen_key[1]}")
    print(f"[OK] violations={chosen_viol} trace_len={len(trace_rows)} adv_action_index_5_sends={chosen_adv5_count}")
    if trace_rows:
        print(f"[OK] first_row.node_id={_as_int(trace_rows[0].get('node_id',''))} action_index={_as_int(trace_rows[0].get('action_index',''))}")
    print("[TRACE] node_ids:", " -> ".join(str(_as_int(r.get('node_id','')) or '?') for r in trace_rows))


if __name__ == "__main__":
    main()
