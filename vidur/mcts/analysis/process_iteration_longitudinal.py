"""
Process-level longitudinal analysis for Parallel Launch MCTS runs.

For each run P# in simulator_output/Parrallel_Launch/analysis/P#/analysis.csv:
  1) Select a set of controller node_ids to track:
     - If there exists any adversary node with visits >= threshold, take ALL of
       its controller children (rows whose parent_node_id == adversary node_id).
     - Otherwise, take all controller rows in the analysis.csv (phase == "tree",
       player_to_act == "controller").
     - Then group by controller_token_budget and keep only the node_id with the
       highest visits per budget (ties broken by smallest numeric node_id).

  2) For each selected node_id, scan the corresponding
     test_mcts_trace_tree_P#.csv and extract (iteration, node_id, node_cost, visits)
     at every multiple of --stride and at the final iteration.
     Controller token budget is taken from analysis.csv.

  3) Write per-run output:
     simulator_output/Parrallel_Launch/analysis/P#/process_iteration_longitudnal.csv

  4) Verify that for the final iteration, (visits, node_cost) match the values
     in analysis.csv for those node_ids (best-effort; logs mismatches).

This script intentionally streams the potentially huge tree CSV and does not
load it fully into memory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "":
            return default
        return int(float(s))  # tolerate "0.0"
    except Exception:
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _node_id_sort_key(node_id: str) -> Tuple[int, str]:
    try:
        return (0, int(node_id))
    except Exception:
        return (1, node_id)


def select_controller_node_ids(
    analysis_csv: Path,
    *,
    adversary_visit_threshold: int,
) -> Tuple[Set[str], Dict[str, str], Dict[str, Tuple[int, float]]]:
    """
    Returns:
      - selected_node_ids: set of controller node_ids after budget-max filtering
      - node_budget: node_id -> controller_token_budget (string)
      - analysis_stats: node_id -> (visits, node_cost) from analysis.csv
    """
    rows = list(_iter_csv_rows(analysis_csv))
    if not rows:
        return set(), {}, {}

    # Find adversary nodes with many visits.
    high_adv_nodes: List[str] = []
    for r in rows:
        if r.get("player_to_act") != "adversary":
            continue
        visits = _safe_int(r.get("visits"), 0)
        if visits >= adversary_visit_threshold:
            nid = str(r.get("node_id", "")).strip()
            if nid:
                high_adv_nodes.append(nid)

    # Candidate controller rows.
    controller_rows: List[Dict[str, str]] = []
    if high_adv_nodes:
        adv_set = set(high_adv_nodes)
        for r in rows:
            if r.get("player_to_act") != "controller":
                continue
            parent = str(r.get("parent_node_id", "")).strip()
            if parent in adv_set:
                controller_rows.append(r)
    else:
        for r in rows:
            if r.get("phase") != "tree":
                continue
            if r.get("player_to_act") != "controller":
                continue
            controller_rows.append(r)

    node_budget: Dict[str, str] = {}
    analysis_stats: Dict[str, Tuple[int, float]] = {}
    for r in controller_rows:
        nid = str(r.get("node_id", "")).strip()
        if not nid:
            continue
        node_budget[nid] = str(r.get("controller_token_budget", "")).strip()
        analysis_stats[nid] = (_safe_int(r.get("visits"), 0), _safe_float(r.get("node_cost"), 0.0))

    # Keep highest-visits node per budget.
    best_by_budget: Dict[str, Tuple[int, str]] = {}  # budget -> (visits, node_id)
    for r in controller_rows:
        nid = str(r.get("node_id", "")).strip()
        if not nid:
            continue
        budget = str(r.get("controller_token_budget", "")).strip()
        visits = _safe_int(r.get("visits"), 0)
        cur = best_by_budget.get(budget)
        if cur is None:
            best_by_budget[budget] = (visits, nid)
            continue
        cur_visits, cur_nid = cur
        if visits > cur_visits:
            best_by_budget[budget] = (visits, nid)
        elif visits == cur_visits:
            if _node_id_sort_key(nid) < _node_id_sort_key(cur_nid):
                best_by_budget[budget] = (visits, nid)

    selected_node_ids = {nid for (_, nid) in best_by_budget.values()}
    # Ensure budgets for selected ids exist.
    node_budget = {nid: node_budget.get(nid, "") for nid in selected_node_ids}
    analysis_stats = {nid: analysis_stats.get(nid, (0, 0.0)) for nid in selected_node_ids}
    return selected_node_ids, node_budget, analysis_stats


def extract_tree_stats_longitudinal(
    tree_csv: Path,
    node_ids: Set[str],
    *,
    stride: int,
) -> Tuple[int, Dict[Tuple[str, int], Tuple[int, float]]]:
    """
    Stream the tree CSV once and collect:
      (node_id, iteration) -> (visits, node_cost)
    for iterations divisible by stride and for the final iteration.

    Returns:
      max_iter, data
    """
    stride = max(1, int(stride))
    max_iter = -1
    data: Dict[Tuple[str, int], Tuple[int, float]] = {}
    last_iter_rows: Dict[str, Tuple[int, float]] = {}

    if not tree_csv.exists():
        return -1, {}

    with tree_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = str(row.get("node_id", "")).strip()
            if nid not in node_ids:
                # Still track max_iter.
                it = _safe_int(row.get("iteration"), -1)
                if it > max_iter:
                    max_iter = it
                    last_iter_rows.clear()
                elif it == max_iter:
                    pass
                continue

            it = _safe_int(row.get("iteration"), -1)
            if it < 0:
                continue
            visits = _safe_int(row.get("visits"), 0)
            node_cost = _safe_float(row.get("node_cost"), 0.0)

            if it % stride == 0:
                data[(nid, it)] = (visits, node_cost)

            if it > max_iter:
                max_iter = it
                last_iter_rows = {nid: (visits, node_cost)}
            elif it == max_iter:
                last_iter_rows[nid] = (visits, node_cost)

    # Ensure last iteration is included even if not divisible by stride.
    if max_iter >= 0:
        for nid, stats in last_iter_rows.items():
            data[(nid, max_iter)] = stats

    return max_iter, data


def write_longitudinal_csv(
    out_csv: Path,
    node_budgets: Dict[str, str],
    max_iter: int,
    data: Dict[Tuple[str, int], Tuple[int, float]],
    *,
    stride: int,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["iteration", "node_id", "controller_token_budget", "visits", "node_cost"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        # Determine which iterations to emit (all collected iterations).
        iterations = sorted({it for (_, it) in data.keys()})
        # If we streamed and collected only stride-multiples and last, this is correct.
        # Ensure last iteration is included.
        if max_iter >= 0 and max_iter not in iterations:
            iterations.append(max_iter)
            iterations.sort()

        for it in iterations:
            for nid in sorted(node_budgets.keys(), key=_node_id_sort_key):
                stats = data.get((nid, it))
                if stats is None:
                    continue
                visits, node_cost = stats
                w.writerow(
                    {
                        "iteration": it,
                        "node_id": nid,
                        "controller_token_budget": node_budgets.get(nid, ""),
                        "visits": visits,
                        "node_cost": node_cost,
                    }
                )


def verify_last_iteration_matches_analysis(
    *,
    run_id: str,
    max_iter: int,
    node_ids: Set[str],
    analysis_stats: Dict[str, Tuple[int, float]],
    tree_data: Dict[Tuple[str, int], Tuple[int, float]],
) -> bool:
    ok = True
    for nid in sorted(node_ids, key=_node_id_sort_key):
        a_vis, a_cost = analysis_stats.get(nid, (0, 0.0))
        t = tree_data.get((nid, max_iter))
        if t is None:
            print(
                f"[WARN] P{run_id}: no tree stats for node_id={nid} at last iteration={max_iter}",
                file=sys.stderr,
            )
            ok = False
            continue
        t_vis, t_cost = t
        if a_vis != t_vis or abs(a_cost - t_cost) > 1e-9:
            print(
                f"[WARN] P{run_id}: mismatch at last iter for node_id={nid}: "
                f"analysis(visits={a_vis}, cost={a_cost}) vs tree(visits={t_vis}, cost={t_cost})",
                file=sys.stderr,
            )
            ok = False
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build per-process longitudinal node stats for Parallel Launch MCTS runs."
    )
    parser.add_argument(
        "--par_dir",
        type=str,
        default="simulator_output/Parrallel_Launch",
        help="Parallel launch directory containing test_mcts_trace_P*.csv files.",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Analysis directory (defaults to <par_dir>/analysis).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=200,
        help="Iteration stride to sample longitudinal stats (also includes last iteration).",
    )
    parser.add_argument(
        "--adv_visits_threshold",
        type=int,
        default=10000,
        help="Adversary visits threshold to switch selection mode.",
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of run IDs to process (e.g. 1 4 10). Default: all found in analysis_dir.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="process_iteration_longitudnal.csv",
        help="Output CSV name written inside each analysis/P#/ directory.",
    )

    args = parser.parse_args(argv)

    par_dir = Path(args.par_dir)
    analysis_root = Path(args.analysis_dir) if args.analysis_dir else (par_dir / "analysis")

    if not analysis_root.exists():
        print(f"[ERROR] analysis_dir not found: {analysis_root}", file=sys.stderr)
        return 2

    if args.run_ids:
        run_ids = [str(x).lstrip("P") for x in args.run_ids]
    else:
        discovered = [p.name.lstrip("P") for p in analysis_root.glob("P*") if p.is_dir()]
        run_ids = sorted(discovered, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))

    if not run_ids:
        print(f"[WARN] No runs found under {analysis_root}", file=sys.stderr)
        return 0

    any_fail = False
    for rid in run_ids:
        analysis_csv = analysis_root / f"P{rid}" / "analysis.csv"
        if not analysis_csv.exists():
            print(f"[WARN] Missing analysis CSV for P{rid}: {analysis_csv}", file=sys.stderr)
            continue

        node_ids, node_budgets, analysis_stats = select_controller_node_ids(
            analysis_csv, adversary_visit_threshold=args.adv_visits_threshold
        )
        if not node_ids:
            print(f"[WARN] P{rid}: no controller node_ids selected", file=sys.stderr)
            continue

        tree_csv = par_dir / f"test_mcts_trace_tree_P{rid}.csv"
        max_iter, tree_data = extract_tree_stats_longitudinal(
            tree_csv, node_ids, stride=args.stride
        )
        if max_iter < 0:
            print(f"[WARN] P{rid}: missing or empty tree CSV: {tree_csv}", file=sys.stderr)
            continue

        out_csv = analysis_root / f"P{rid}" / args.out_name
        write_longitudinal_csv(
            out_csv,
            node_budgets=node_budgets,
            max_iter=max_iter,
            data=tree_data,
            stride=args.stride,
        )

        ok = verify_last_iteration_matches_analysis(
            run_id=rid,
            max_iter=max_iter,
            node_ids=node_ids,
            analysis_stats=analysis_stats,
            tree_data=tree_data,
        )
        if not ok:
            any_fail = True

        print(
            f"[INFO] P{rid}: wrote {out_csv} "
            f"(selected_nodes={len(node_ids)}, last_iter={max_iter})"
        )

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
