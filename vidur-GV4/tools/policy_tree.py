from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _as_float(v: str, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _as_int(v: str, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def _load_json(v: str, fallback):
    if v is None or v == "":
        return fallback
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return fallback


@dataclass
class Node:
    node_id: str
    parent_id: Optional[str]
    depth: float
    actor: Optional[str]
    next_player: Optional[str]
    action_summary: str
    base_cost: Optional[float] = None  # mean of terminal rollout costs for this node
    value: Optional[float] = None      # propagated minimax value
    best_child: Optional[str] = None


def _summarize_action(row: Dict[str, str]) -> str:
    actor = row.get("player_to_act")
    if actor == "controller":
        budget = _as_int(row.get("controller_token_budget"))
        sel = _load_json(row.get("controller_selected_ids"), [])
        pre = _load_json(row.get("controller_prefill_allocations"), {})
        dec = _load_json(row.get("controller_decode_allocations"), {})
        pre_t = sum(int(v) for v in pre.values())
        dec_t = sum(int(v) for v in dec.values())
        return (
            f"Controller action | budget={budget:,} | prefill={pre_t:,} | decode={dec_t:,}\n"
            f"selected: {sel[:8]}{'…' if len(sel)>8 else ''}"
        )
    if actor == "adversary":
        specs = _load_json(row.get("adversary_requests"), [])
        if not specs:
            return "Adversary: no-op"
        tp = sum(int(s.get("prefill_tokens", 0)) for s in specs)
        td = sum(int(s.get("decode_tokens", 0)) for s in specs)
        return f"Adversary action | requests={len(specs)} | prefill={tp:,} | decode={td:,}"
    return ""


def discover_latest_iteration(csv_path: Path) -> Optional[int]:
    latest = None
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("phase") != "tree":
                continue
            it = _as_int(row.get("iteration"), None)  # type: ignore[arg-type]
            if it is None:
                continue
            if latest is None or it > latest:
                latest = it
    return latest


def load_tree_rows(csv_path: Path, max_iteration: Optional[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    root_added = False
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            phase = row.get("phase", "")
            itr = _as_int(row.get("iteration"))
            if max_iteration is not None and itr > max_iteration:
                continue
            if phase == "root":
                if not root_added:
                    rows.append(row)
                    root_added = True
                continue
            if phase != "tree":
                continue
            rows.append(row)
    return rows


def aggregate_rollout_costs(csv_path: Path, max_iteration: Optional[int]) -> Dict[str, List[float]]:
    """Return terminal rollout costs grouped by base tree node id.

    We parse rollout node_id formatted like rollout_<iter>_<treeNode>_<trial>_...,
    keeping the last cost per (iter, treeNode, trial).
    """
    per_trial: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("phase") != "rollout":
                continue
            itr = row.get("iteration")
            itr_val = None
            try:
                itr_val = int(float(itr)) if itr not in (None, "") else None
            except ValueError:
                pass
            if max_iteration is not None and itr_val is not None and itr_val > max_iteration:
                continue

            node_label = (row.get("node_id") or "").strip()
            base = None
            trial_key: Optional[Tuple[str, str, str]] = None

            if node_label.startswith("rollout_"):
                parts = node_label.split("_")
                if len(parts) >= 5:
                    base = parts[2]
                    trial_key = (parts[1], parts[2], parts[3])
            else:
                parent_id = (row.get("parent_node_id") or "").strip()
                if parent_id.isdigit():
                    base = parent_id
                    trial_key = (str(itr_val or 0), parent_id, node_label)

            if base is None or not base.isdigit():
                continue

            cost = _as_float(row.get("objective_cost"))
            per_trial[base][trial_key or ("", base, "")] = cost

    result: Dict[str, List[float]] = {}
    for base, trials in per_trial.items():
        result[base] = list(trials.values())
    return result


def build_tree(rows: Iterable[Dict[str, str]], rollout_costs: Dict[str, List[float]]) -> Tuple[Dict[str, Node], Dict[str, List[str]]]:
    nodes: Dict[str, Node] = {}
    children: Dict[str, List[str]] = defaultdict(list)

    for row in rows:
        nid = (row.get("node_id") or "").strip()
        if not nid:
            continue
        pid = (row.get("parent_node_id") or "").strip() or None
        if pid == "0":
            pid = "root"

        node = Node(
            node_id=nid,
            parent_id=pid,
            depth=_as_float(row.get("depth")),
            actor=row.get("player_to_act"),
            next_player=row.get("next_player"),
            action_summary=_summarize_action(row),
        )
        rc = rollout_costs.get(nid)
        if rc:
            node.base_cost = sum(rc) / len(rc)
        nodes[nid] = node
        if pid:
            children[pid].append(nid)

    # stable order for layout
    for sib in children.values():
        sib.sort(key=lambda x: (int(x) if x.isdigit() else x))
    return nodes, children


def minimax_propagate(nodes: Dict[str, Node], children: Dict[str, List[str]]) -> None:
    # order nodes by depth descending
    ordered = sorted(nodes.values(), key=lambda n: n.depth, reverse=True)
    for node in ordered:
        offs = children.get(node.node_id, [])
        # compute child values first (they are deeper because of sorting)
        child_vals: List[Tuple[str, float]] = []
        for cid in offs:
            cv = nodes[cid].value
            if cv is None:
                # if child has its own base cost, use it
                if nodes[cid].base_cost is not None:
                    cv = nodes[cid].base_cost
                else:
                    continue
            child_vals.append((cid, cv))

        if child_vals:
            if node.next_player == "controller":
                cid, val = min(child_vals, key=lambda t: t[1])
            else:
                cid, val = max(child_vals, key=lambda t: t[1])
            node.value = val
            node.best_child = cid
        else:
            if node.base_cost is not None:
                node.value = node.base_cost


def render_svg(nodes: Dict[str, Node], children: Dict[str, List[str]], title: str, out_path: Path) -> Path:
    # position by integer depth groups
    depth_groups: Dict[int, List[Node]] = defaultdict(list)
    for n in nodes.values():
        depth_groups[int(n.depth)].append(n)
    for grp in depth_groups.values():
        grp.sort(key=lambda n: (int(n.node_id) if n.node_id.isdigit() else n.node_id))

    positions: Dict[str, Tuple[float, float]] = {}
    for d in sorted(depth_groups.keys()):
        grp = depth_groups[d]
        for idx, n in enumerate(grp):
            positions[n.node_id] = (d, -idx)

    def to_px(p: Tuple[float, float]) -> Tuple[float, float]:
        x, y = p
        return (x + 1) * 180, (abs(y) + 1) * 160

    # canvas size
    xs = [p[0] for p in positions.values()] or [0]
    ys = [p[1] for p in positions.values()] or [0]
    width = int((max(xs) + 2) * 200)
    height = int((abs(min(ys)) + 2) * 180)

    # edges
    edge_elems: List[str] = []
    for pid, kids in children.items():
        for cid in kids:
            if pid not in positions or cid not in positions:
                continue
            x0, y0 = to_px(positions[pid])
            x1, y1 = to_px(positions[cid])
            is_best = nodes[pid].best_child == cid
            color = "#d62728" if is_best else "#bfbfbf"
            width_px = 3 if is_best else 1.5
            edge_elems.append(
                f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{width_px}" />'
            )

    # nodes
    node_elems: List[str] = []
    for n in nodes.values():
        if n.node_id not in positions:
            continue
        cx, cy = to_px(positions[n.node_id])
        fill = "#1f77b4" if n.next_player == "controller" else "#ff7f0e"
        value = "NA" if n.value is None else f"{n.value:.3f}"
        base = "NA" if n.base_cost is None else f"{n.base_cost:.3f}"
        node_elems.append(
            f'<g><circle cx="{cx}" cy="{cy}" r="24" fill="{fill}" stroke="#222" stroke-width="2" />'
            f'<text x="{cx}" y="{cy+40}" text-anchor="middle" font-size="12" fill="#111">{n.node_id} | v={value} | base={base}</text>'
            f'</g>'
        )

    legend = (
        '<g transform="translate(20,30)">'
        '<rect width="18" height="18" fill="#1f77b4" stroke="#222" />'
        '<text x="26" y="14" font-size="12">Controller to act next</text>'
        '<rect y="24" width="18" height="18" fill="#ff7f0e" stroke="#222" />'
        '<text x="26" y="38" font-size="12">Adversary to act next</text>'
        '<line x1="0" y1="54" x2="18" y2="54" stroke="#d62728" stroke-width="3" />'
        '<text x="26" y="58" font-size="12">Chosen child</text>'
        '</g>'
    )

    svg = (
        f"<svg width='{width}' height='{height}' style='background:#fff;border:1px solid #ddd'>"
        f"{legend}{''.join(edge_elems)}{''.join(node_elems)}</svg>"
    )
    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
  </head>
  <body style="font-family:Segoe UI, Arial, sans-serif;background:#f5f5f5;color:#222;">
    <h2 style="text-align:center;">{title}</h2>
    <div style="max-width:1200px;margin:0 auto;">{svg}</div>
  </body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a minimax policy tree from an MCTS CSV log and render an HTML SVG. "
            "Objective cost is violations + avg_lateness for both players."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("vidur/backup_FOR_comparison/test_mcts_trace.csv"),
        help="Path to MCTS CSV log.",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Use tree rows up to this iteration (defaults to latest tree iteration).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("simulator_output/policy_tree.html"),
        help="Output HTML file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    itr = args.iteration or discover_latest_iteration(args.csv)
    rows = load_tree_rows(args.csv, itr)
    rollouts = aggregate_rollout_costs(args.csv, None)
    nodes, children = build_tree(rows, rollouts)
    minimax_propagate(nodes, children)
    title = f"Policy Tree (iteration {itr if itr is not None else 'all'})"
    out = render_svg(nodes, children, title, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

