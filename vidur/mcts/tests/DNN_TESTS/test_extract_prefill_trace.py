"""
Pick the "longest / richest" trace from mcts_iter.csv and export it.

Definition of "best trace":
1) Max number of adversary actions that sent new prefill requests (requests != [])
2) If tie: max path length (nodes)
3) If tie: random choice (seeded)

Writes: extracted_trace.csv in this folder, with same headers as mcts_iter.csv.

Run:
  python3 -m vidur.mcts.tests.DNN_TESTS.test_extract_prefill_trace \
    --log_path vidur/simulator_output/mcts_dnn_logs/mcts_iter.csv \
    --seed 0
"""

from __future__ import annotations

import argparse
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


def load_rows(log_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"No headers found in {log_path}")
    return fieldnames, rows


def build_tree(rows: List[Dict[str, str]], *, game_id: Optional[int], root_id: Optional[int]):
    filtered: List[Dict[str, str]] = []
    for r in rows:
        gid = _as_int(r.get("game_id", ""))
        rid = _as_int(r.get("root_id", ""))
        if game_id is not None and gid != game_id:
            continue
        if root_id is not None and rid != root_id:
            continue
        filtered.append(r)

    node_row: Dict[int, Dict[str, str]] = {}
    parent_of: Dict[int, Optional[int]] = {}
    children_of: Dict[int, List[int]] = {}

    def better(existing: Dict[str, str], candidate: Dict[str, str]) -> bool:
        # prefer a row where adversary actually sent requests (best action_repr)
        if not adversary_sent_prefill(existing) and adversary_sent_prefill(candidate):
            return True
        # otherwise keep first-seen by sim_iteration
        ei = _as_int(existing.get("sim_iteration", "")) or 10**18
        ci = _as_int(candidate.get("sim_iteration", "")) or 10**18
        return ci < ei

    for r in filtered:
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

    roots = [nid for nid, pid in parent_of.items() if pid is None]
    return node_row, parent_of, children_of, roots


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

    node_row, parent_of, children_of, roots = build_tree(rows, game_id=args.game_id, root_id=args.root_id)
    if not node_row:
        raise SystemExit("No rows/nodes found after filtering.")

    leaves = [nid for nid in node_row.keys() if nid not in children_of]
    if not leaves:
        raise SystemExit("No leaves found in the log-derived tree.")

    # Evaluate every leaf path and pick the "best"
    scored: List[Tuple[int, int, List[int]]] = []  # (adv_prefill_sends, path_len, path)
    for leaf in leaves:
        path = path_to_root(leaf, parent_of)
        adv_sends = 0
        for nid in path:
            r = node_row.get(nid)
            if r and adversary_sent_prefill(r):
                adv_sends += 1
        scored.append((adv_sends, len(path), path))

    max_adv = max(a for a, _, _ in scored)
    best_adv = [(a, l, p) for (a, l, p) in scored if a == max_adv]
    max_len = max(l for _, l, _ in best_adv)
    best = [(a, l, p) for (a, l, p) in best_adv if l == max_len]

    rng = random.Random(int(args.seed))
    chosen_adv, chosen_len, chosen_path = rng.choice(best)

    trace_rows: List[Dict[str, str]] = [node_row[nid] for nid in chosen_path if nid in node_row]

    out_dir = Path(__file__).resolve().parent
    out_path = Path(args.out_path) if args.out_path else (out_dir / "extracted_trace.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trace_rows:
            w.writerow(r)

    print(f"[OK] log_path={log_path}")
    print(f"[OK] wrote trace to {out_path}")
    print(f"[OK] chosen trace length={chosen_len} adversary_prefill_sends={chosen_adv}")
    print("[TRACE] node_ids:", " -> ".join(str(_as_int(r.get('node_id','')) or '?') for r in trace_rows))


if __name__ == "__main__":
    main()
