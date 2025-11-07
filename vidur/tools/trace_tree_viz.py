from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


try:  # Plotly is optional; fall back to SVG-only rendering if unavailable.
    import plotly.graph_objects as go  # type: ignore
except ImportError:
    go = None


@dataclass
class TraceNode:
    """Lightweight container for logged MCTS nodes."""

    iteration: int
    phase: str
    depth: float
    parent_id: Optional[str]
    node_id: str
    actor: Optional[str]
    next_player: Optional[str]
    sim_time: float
    requests_in_system: int
    requests_generated: int
    requests_completed: int
    slo_violations: int
    avg_lateness: float
    objective_cost: float
    waiting_ids: List[int]
    completed_ids: List[int]
    action_summary: str
    action_short_summary: str
    rollout_costs: List[float] = field(default_factory=list)
    sim_cost: float = 0.0


@dataclass
class TreeGraph:
    nodes: Dict[str, TraceNode]
    children: Dict[str, List[str]]
    edge_summaries: Dict[Tuple[str, str], Dict[str, str]]


def _as_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _load_json(value: str, fallback):
    if value is None or value == "":
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _discover_latest_iteration(csv_path: Path) -> Optional[int]:
    latest_any: Optional[int] = None
    latest_tree: Optional[int] = None
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            itr = row.get("iteration")
            try:
                itr_val = int(float(itr)) if itr is not None and itr != "" else None
            except ValueError:
                itr_val = None
            if itr_val is None:
                continue
            phase = row.get("phase", "")
            if (latest_any is None) or (itr_val > latest_any):
                latest_any = itr_val
            if phase == "tree":
                if (latest_tree is None) or (itr_val > latest_tree):
                    latest_tree = itr_val
    return latest_tree if latest_tree is not None else latest_any


def _load_rollout_costs(
    csv_path: Path,
    max_iteration: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Aggregate rollout costs per tree node by capturing the final cost of each trial."""

    latest_trial_cost: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("phase") != "rollout":
                continue
            itr = _as_int(row.get("iteration"))
            if max_iteration is not None and itr is not None and itr > max_iteration:
                continue

            node_label = (row.get("node_id") or "").strip()
            base_node_id: Optional[str] = None
            trial_key: Optional[Tuple[str, str, str]] = None

            if node_label.startswith("rollout_"):
                parts = node_label.split("_")
                if len(parts) >= 5:
                    base_node_id = parts[2]
                    trial_key = (parts[1], parts[2], parts[3])
            else:
                parent_id = (row.get("parent_node_id") or "").strip()
                if parent_id.isdigit():
                    base_node_id = parent_id
                    trial_key = (
                        str(itr) if itr is not None else "",
                        parent_id,
                        row.get("node_id", ""),
                    )

            if base_node_id is None or not base_node_id.isdigit():
                continue

            cost_value = _as_float(row.get("objective_cost"))
            trial_lookup = latest_trial_cost[base_node_id]
            trial_lookup[
                trial_key or ("", base_node_id, "")
            ] = cost_value  # later rows overwrite earlier entries

    aggregated: Dict[str, List[float]] = {}
    for node_id, trial_costs in latest_trial_cost.items():
        aggregated[node_id] = list(trial_costs.values())
    return aggregated


def _summarize_adversary(row: Dict[str, str]) -> str:
    specs: List[Dict[str, float]] = _load_json(row.get("adversary_requests"), [])
    if not specs:
        return "Adversary: no-op"
    total_prefill = sum(int(spec.get("prefill_tokens", 0)) for spec in specs)
    total_decode = sum(int(spec.get("decode_tokens", 0)) for spec in specs)
    slo_range = (
        min(spec.get("prefill_slo", 0.0) for spec in specs),
        max(spec.get("prefill_slo", 0.0) for spec in specs),
    )
    sample = ", ".join(
        f"{spec.get('prefill_tokens', 0)}/{spec.get('decode_tokens', 0)}"
        f" (SLO {spec.get('prefill_slo', 0):.3f}/{spec.get('decode_slo', 0):.3f})"
        for spec in specs[:4]
    )
    more = "…" if len(specs) > 4 else ""
    return (
        "Adversary action<br>"
        f"requests={len(specs)} | prefill={total_prefill:,} | decode={total_decode:,}<br>"
        f"prefill SLO range={slo_range[0]:.3f}-{slo_range[1]:.3f}<br>"
        f"samples: {sample}{more}"
    )


def _fmt_allocation(alloc: Dict[str, int], label: str) -> str:
    if not alloc:
        return ""
    items = list(alloc.items())
    items.sort(key=lambda item: int(item[0]))
    preview = ", ".join(f"{rid}:{tokens}" for rid, tokens in items[:6])
    if len(items) > 6:
        preview += ", …"
    total = sum(int(v) for _, v in items)
    return f"{label} ({total:,}): {preview}"


def _summarize_controller(row: Dict[str, str]) -> str:
    budget = _as_int(row.get("controller_token_budget"))
    selected = _load_json(row.get("controller_selected_ids"), [])
    prefill_alloc = _load_json(row.get("controller_prefill_allocations"), {})
    decode_alloc = _load_json(row.get("controller_decode_allocations"), {})
    prefill_total = (
        _as_int(row.get("controller_prefill_total"))
        or sum(int(v) for v in prefill_alloc.values())
    )
    decode_total = (
        _as_int(row.get("controller_decode_total"))
        or sum(int(v) for v in decode_alloc.values())
    )
    lines = [
        "Controller action",
        f"budget={budget:,} | prefill={prefill_total:,} | decode={decode_total:,}",
        f"selected ({len(selected)}): {selected[:8]}{'…' if len(selected) > 8 else ''}",
    ]
    alloc_prefill = _fmt_allocation(prefill_alloc, "prefill allocs")
    alloc_decode = _fmt_allocation(decode_alloc, "decode allocs")
    if alloc_prefill:
        lines.append(alloc_prefill)
    if alloc_decode:
        lines.append(alloc_decode)
    return "<br>".join(lines)


def _summarize_action(row: Dict[str, str]) -> str:
    actor = row.get("player_to_act")
    if actor == "adversary":
        return _summarize_adversary(row)
    if actor == "controller":
        return _summarize_controller(row)
    return "No action"


def _summarize_action_short(row: Dict[str, str]) -> str:
    actor = row.get("player_to_act")
    if actor == "adversary":
        specs: List[Dict[str, float]] = _load_json(row.get("adversary_requests"), [])
        total_prefill = sum(int(spec.get("prefill_tokens", 0)) for spec in specs)
        total_decode = sum(int(spec.get("decode_tokens", 0)) for spec in specs)
        return (
            f"adv req={len(specs)} pf={total_prefill:,} de={total_decode:,}"
            if specs
            else "adv noop"
        )
    if actor == "controller":
        budget = _as_int(row.get("controller_token_budget"))
        prefill_alloc = _load_json(row.get("controller_prefill_allocations"), {})
        decode_alloc = _load_json(row.get("controller_decode_allocations"), {})
        prefill_total = (
            _as_int(row.get("controller_prefill_total"))
            or sum(int(v) for v in prefill_alloc.values())
        )
        decode_total = (
            _as_int(row.get("controller_decode_total"))
            or sum(int(v) for v in decode_alloc.values())
        )
        return (
            f"ctl bud={budget:,} pf={prefill_total:,} de={decode_total:,}"
            if budget or prefill_total or decode_total
            else "ctl noop"
        )
    return ""


def load_trace_records(
    csv_path: Path,
    iteration: int,
    *,
    include_rollouts: bool = False,
    max_depth: Optional[float] = None,
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    root_added = False
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase = row.get("phase", "")
            itr = _as_int(row.get("iteration"))
            if phase == "root":
                if not root_added:
                    records.append(row)
                    root_added = True
                continue
            if not include_rollouts and phase not in {"tree"}:
                continue
            if itr is None or itr > iteration:
                continue
            depth = _as_float(row.get("depth"))
            if max_depth is not None and depth > max_depth:
                continue
            records.append(row)
    return records


def build_tree(
    records: Iterable[Dict[str, str]],
    rollout_costs: Optional[Dict[str, List[float]]] = None,
) -> TreeGraph:
    nodes: Dict[str, TraceNode] = {}
    children: Dict[str, List[str]] = defaultdict(list)
    edge_summaries: Dict[Tuple[str, str], Dict[str, str]] = {}

    for row in records:
        node_id = (row.get("node_id") or "").strip()
        if not node_id:
            continue
        parent_id = (row.get("parent_node_id") or "").strip() or None
        if parent_id == "0":
            parent_id = "root"

        rollout_list = []
        if rollout_costs:
            rollout_list = rollout_costs.get(node_id, [])
        node = TraceNode(
            iteration=_as_int(row.get("iteration")),
            phase=row.get("phase", ""),
            depth=_as_float(row.get("depth")),
            parent_id=parent_id,
            node_id=node_id,
            actor=row.get("player_to_act"),
            next_player=row.get("next_player"),
            sim_time=_as_float(row.get("sim_time")),
            requests_in_system=_as_int(row.get("requests_in_system")),
            requests_generated=_as_int(row.get("requests_generated")),
            requests_completed=_as_int(row.get("requests_completed")),
            slo_violations=_as_int(row.get("slo_violations")),
            avg_lateness=_as_float(row.get("avg_lateness")),
            objective_cost=_as_float(row.get("objective_cost")),
            waiting_ids=_load_json(row.get("state_waiting_ids"), []),
            completed_ids=_load_json(row.get("state_completed_request_ids"), []),
            action_summary=_summarize_action(row),
            action_short_summary=_summarize_action_short(row),
            rollout_costs=list(rollout_list),
            sim_cost=_as_float(row.get("objective_cost")),
        )
        if rollout_list:
            node.sim_cost = sum(rollout_list) / len(rollout_list)
        nodes[node_id] = node

        if parent_id:
            children[parent_id].append(node_id)
            edge_summaries[(parent_id, node_id)] = {
                "detail": node.action_summary,
                "short": node.action_short_summary,
            }

    # Ensure child lists are stable for reproducible layouts.
    for siblings in children.values():
        siblings.sort()

    return TreeGraph(nodes=nodes, children=children, edge_summaries=edge_summaries)


def _compute_best_child_map(tree: TreeGraph) -> Dict[str, str]:
    best_edges: Dict[str, str] = {}
    for node_id, node in tree.nodes.items():
        offspring = tree.children.get(node_id)
        if not offspring or not node.next_player:
            continue
        if node.next_player == "adversary":
            best_child = max(offspring, key=lambda cid: tree.nodes[cid].sim_cost)
        else:
            best_child = min(offspring, key=lambda cid: tree.nodes[cid].sim_cost)
        best_edges[node_id] = best_child
    return best_edges


def _compute_best_edges(tree: TreeGraph) -> Dict[Tuple[str, str], str]:
    mapping = _compute_best_child_map(tree)
    return {(parent, child): tree.nodes[parent].next_player or "" for parent, child in mapping.items()}


def _layout_nodes(tree: TreeGraph) -> Dict[str, Tuple[float, float]]:
    depth_groups: Dict[int, List[str]] = defaultdict(list)
    for node_id, node in tree.nodes.items():
        depth_groups[int(node.depth)].append(node_id)
    for group in depth_groups.values():
        group.sort()

    positions: Dict[str, Tuple[float, float]] = {}
    for depth in sorted(depth_groups):
        nodes_at_depth = depth_groups[depth]
        for idx, node_id in enumerate(nodes_at_depth):
            x = depth
            y = -idx
            positions[node_id] = (x, y)
    return positions


def _render_with_plotly(
    tree: TreeGraph,
    *,
    title: str,
    output_path: Path,
) -> Path:
    positions = _layout_nodes(tree)
    best_edges = _compute_best_edges(tree)

    best_edge_x: List[float] = []
    best_edge_y: List[float] = []
    other_edge_x: List[float] = []
    other_edge_y: List[float] = []
    edge_hover_x: List[float] = []
    edge_hover_y: List[float] = []
    edge_hover_text: List[str] = []

    for (parent_id, children_ids) in tree.children.items():
        for child_id in children_ids:
            if parent_id not in positions or child_id not in positions:
                continue
            x0, y0 = positions[parent_id]
            x1, y1 = positions[child_id]
            xs = [x0, x1, None]
            ys = [y0, y1, None]
            if (parent_id, child_id) in best_edges:
                best_edge_x.extend(xs)
                best_edge_y.extend(ys)
            else:
                other_edge_x.extend(xs)
                other_edge_y.extend(ys)
            edge_hover_x.append((x0 + x1) / 2)
            edge_hover_y.append((y0 + y1) / 2)
            edge_hover_text.append(
                tree.edge_summaries.get((parent_id, child_id), {}).get("detail", "")
            )

    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_hover: List[str] = []
    node_color: List[str] = []

    color_map = {"adversary": "#ff7f0e", "controller": "#1f77b4"}

    for node_id, node in tree.nodes.items():
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_x.append(x)
        node_y.append(y)
        label = f"{node_id}"
        node_text.append(label)
        hover = [
            f"<b>Node {node_id}</b>",
            f"phase={node.phase}",
            f"depth={node.depth}",
            f"actor={node.actor} | next={node.next_player}",
            f"tree objective={node.objective_cost:.3f}",
            f"simulated cost={node.sim_cost:.3f}",
            f"sim_time={node.sim_time:.3f}",
            f"requests_in_system={node.requests_in_system}",
            f"requests_generated={node.requests_generated}",
            f"requests_completed={node.requests_completed}",
            f"SLO violations={node.slo_violations}",
            f"avg lateness={node.avg_lateness:.6f}",
            f"waiting_ids={node.waiting_ids}",
        ]
        node_hover.append("<br>".join(hover))
        node_color.append(color_map.get(node.next_player, "#7f7f7f"))

    fig = go.Figure()
    if other_edge_x:
        fig.add_trace(
            go.Scatter(
                x=other_edge_x,
                y=other_edge_y,
                mode="lines",
                line=dict(color="#d9d9d9", width=1),
                hoverinfo="skip",
                name="explored actions",
            )
        )
    if best_edge_x:
        fig.add_trace(
            go.Scatter(
                x=best_edge_x,
                y=best_edge_y,
                mode="lines",
                line=dict(color="#d62728", width=3),
                hoverinfo="skip",
                name="best response",
            )
        )
    if edge_hover_text:
        fig.add_trace(
            go.Scatter(
                x=edge_hover_x,
                y=edge_hover_y,
                mode="markers",
                marker=dict(size=6, color="rgba(0,0,0,0)"),
                hoverinfo="text",
                text=edge_hover_text,
                name="actions",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="bottom center",
            marker=dict(size=16, color=node_color, line=dict(width=1, color="#333")),
            hoverinfo="text",
            hovertext=node_hover,
            name="states",
        )
        )
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def _render_with_svg(
    tree: TreeGraph,
    *,
    title: str,
    output_path: Path,
) -> Path:
    positions = _layout_nodes(tree)
    best_edges = _compute_best_edges(tree)

    if not positions:
        raise ValueError("No positions computed; nothing to render.")

    xs = [coord[0] for coord in positions.values()]
    ys = [coord[1] for coord in positions.values()]
    width = (max(xs, default=0) + 2) * 160
    height = (abs(min(ys, default=0)) + 2) * 160

    def to_pixel(coord: Tuple[float, float]) -> Tuple[float, float]:
        x, y = coord
        return (x + 1) * 160, (abs(y) + 1) * 140

    svg_elements: List[str] = []
    color_map = {"adversary": "#ff7f0e", "controller": "#1f77b4"}

    for (parent_id, children_ids) in tree.children.items():
        for child_id in children_ids:
            if parent_id not in positions or child_id not in positions:
                continue
            x0, y0 = to_pixel(positions[parent_id])
            x1, y1 = to_pixel(positions[child_id])
            is_best = (parent_id, child_id) in best_edges
            edge_color = "#d62728" if is_best else "#bfbfbf"
            detail = tree.edge_summaries.get((parent_id, child_id), {}).get("detail", "")
            short = tree.edge_summaries.get((parent_id, child_id), {}).get("short", "")
            svg_elements.append(
                "<g>"
                f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
                f'stroke="{edge_color}" stroke-width="{3 if is_best else 1.5}" />'
                f"<title>{detail}</title>"
                "</g>"
            )
            if short:
                label_x = (x0 + x1) / 2
                label_y = (y0 + y1) / 2 - 10
                svg_elements.append(
                    f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                    f'font-size="12" fill="#333">{short}</text>'
                )

    for node_id, node in tree.nodes.items():
        if node_id not in positions:
            continue
        cx, cy = to_pixel(positions[node_id])
        color = color_map.get(node.next_player, "#7f7f7f")
        hover = (
            f"Node {node_id}\\nphase={node.phase}\\nactor={node.actor} next={node.next_player}"
            f"\\nobjective={node.objective_cost:.3f}\\nsim_cost={node.sim_cost:.3f}"
            f"\\nsim_time={node.sim_time:.3f}"
            f"\\nreq_in_system={node.requests_in_system}\\nslo_violations={node.slo_violations}"
            f"\\navg_lateness={node.avg_lateness:.6f}"
        )
        svg_elements.append(
            "<g>"
            f'<circle cx="{cx}" cy="{cy}" r="22" fill="{color}" stroke="#222" stroke-width="2" />'
            f"<title>{hover}</title>"
            "</g>"
        )
        svg_elements.append(
            f'<text x="{cx}" y="{cy+38}" text-anchor="middle" font-size="12" fill="#111">'
            f"{node_id} | cost={node.sim_cost:.2f}</text>"
        )

    legend = (
        '<g transform="translate(20,30)">'
        '<rect width="18" height="18" fill="#ff7f0e" stroke="#222" />'
        '<text x="26" y="14" font-size="12">Adversary to act next</text>'
        '<rect y="24" width="18" height="18" fill="#1f77b4" stroke="#222" />'
        '<text x="26" y="38" font-size="12">Controller to act next</text>'
        '<line x1="0" y1="54" x2="18" y2="54" stroke="#d62728" stroke-width="3" />'
        '<text x="26" y="58" font-size="12">Best-response edge</text>'
        '</g>'
    )

    svg_body = "\n".join([legend] + svg_elements)
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      font-family: 'Segoe UI', sans-serif;
      background-color: #f5f5f5;
    }}
    svg {{
      background-color: #ffffff;
      border: 1px solid #ddd;
    }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <svg width="{width}" height="{height}">
    {svg_body}
  </svg>
  <p>Hover over nodes and edges to view full statistics.</p>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_tree(
    tree: TreeGraph,
    *,
    title: str,
    output_html: Path,
) -> Path:
    if go is not None:
        return _render_with_plotly(tree, title=title, output_path=output_html)
    output_path = (
        output_html if output_html.suffix else output_html.with_suffix(".html")
    )
    return _render_with_svg(tree, title=title, output_path=output_path)


def _best_path_sequence(tree: TreeGraph) -> List[str]:
    best_child = _compute_best_child_map(tree)
    if "root" not in tree.nodes:
        return []
    sequence = ["root"]
    seen = set(sequence)
    current = "root"
    while current in best_child:
        nxt = best_child[current]
        if nxt in seen:
            break
        sequence.append(nxt)
        seen.add(nxt)
        current = nxt
    return sequence


def render_best_path(
    tree: TreeGraph,
    *,
    title: str,
    output_path: Path,
) -> Path:
    sequence = _best_path_sequence(tree)
    if len(sequence) <= 1:
        raise ValueError("Unable to determine a best-response chain from the trace.")

    items: List[str] = []
    for depth, node_id in enumerate(sequence):
        node = tree.nodes.get(node_id)
        if node is None:
            continue
        summary = tree.edge_summaries.get(
            (sequence[depth - 1], node_id), {}
        ) if depth > 0 else {"detail": "Start", "short": ""}
        action = summary.get("detail", "Start")
        items.append(
            f"""
            <div class="step">
              <div class="badge {'adversary' if node.actor=='adversary' else 'controller'}">
                Depth {depth} • Player {node.actor or 'N/A'}
              </div>
              <div class="card">
                <h3>Node {node_id}</h3>
                <p><strong>Simulated cost:</strong> {node.sim_cost:.3f}</p>
                <p><strong>Tree objective snapshot:</strong> {node.objective_cost:.3f}</p>
                <p><strong>SLO violations:</strong> {node.slo_violations} | <strong>Avg lateness:</strong> {node.avg_lateness:.6f}</p>
                <p><strong>Requests in system:</strong> {node.requests_in_system}</p>
                <p><strong>Action:</strong><br>{action}</p>
              </div>
            </div>
            """
        )

    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{
        font-family: "Segoe UI", Arial, sans-serif;
        background: #f5f5f5;
        color: #222;
      }}
      .container {{
        max-width: 900px;
        margin: 20px auto;
        padding: 0 20px;
      }}
      .step {{
        position: relative;
        margin: 30px 0;
        padding-left: 30px;
      }}
      .step::before {{
        content: "";
        position: absolute;
        left: 14px;
        top: 0;
        bottom: -30px;
        width: 2px;
        background: #d0d0d0;
      }}
      .step:last-child::before {{
        bottom: 40px;
      }}
      .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        margin-bottom: 8px;
        color: #fff;
      }}
      .badge.adversary {{
        background: #ff7f0e;
      }}
      .badge.controller {{
        background: #1f77b4;
      }}
      .card {{
        background: #fff;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      }}
      h2 {{
        text-align: center;
      }}
      h3 {{
        margin: 0 0 10px 0;
      }}
      p {{
        margin: 4px 0;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h2>{title}</h2>
      <p>This timeline shows the best-response chain inferred from the trace: each player’s move leads to the child state with the most favorable objective (controller minimizes cost, adversary maximizes cost).</p>
      {''.join(items)}
    </div>
  </body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def describe_best_path(tree: TreeGraph) -> List[str]:
    """Generate a textual outline of best actions from root downward."""
    sequence = _best_path_sequence(tree)
    if len(sequence) <= 1:
        return []
    outline: List[str] = []
    for depth in range(len(sequence) - 1):
        parent = sequence[depth]
        child_id = sequence[depth + 1]
        child = tree.nodes.get(child_id)
        if not child:
            continue
        outline.append(
            f"depth {depth}: {parent} -> {child_id} "
            f"({child.actor}) cost={child.sim_cost:.3f}"
        )
    return outline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize an MCTS trace CSV either as a tree (all explored nodes) or as "
            "a best-response chain (one path showing alternating adversary/controller actions)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the simulator_output/test_mcts_trace.csv file.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Search iteration to visualize (defaults to the last iteration present in the CSV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulator_output/mcts_tree.html"),
        help="Where to write the interactive HTML visualization.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional depth cutoff (tree depth value from the log).",
    )
    parser.add_argument(
        "--include-rollouts",
        action="store_true",
        help="Include rollout rows as well as tree expansion rows.",
    )
    parser.add_argument(
        "--mode",
        choices=("tree", "best-path"),
        default="tree",
        help="tree = full visualization, best-path = static timeline of the best response chain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iteration = args.iteration
    if iteration is None:
        iteration = _discover_latest_iteration(args.csv)
        if iteration is None:
            raise SystemExit(
                "Could not determine a valid iteration from the trace CSV."
            )
        print(
            f"[trace_tree_viz] No --iteration provided; using latest iteration {iteration}."
        )
    records = load_trace_records(
        args.csv,
        iteration,
        include_rollouts=args.include_rollouts,
        max_depth=args.max_depth,
    )
    rollout_costs = _load_rollout_costs(args.csv)
    if not records:
        raise SystemExit(
            f"No rows found for iteration {iteration}. "
            "Did you pass the correct --iteration or depth filter?"
        )
    tree = build_tree(records, rollout_costs=rollout_costs)
    title = f"MCTS iteration {iteration} best-response {args.mode}"
    if args.mode == "tree":
        output_path = render_tree(
            tree,
            title=title,
            output_html=args.output,
        )
    else:
        output_path = render_best_path(
            tree,
            title=title,
            output_path=args.output,
        )
    outline = describe_best_path(tree)
    print(f"Wrote visualization to {output_path}")
    if outline:
        print("\nBest-response outline:")
        for line in outline:
            print("  ", line)


if __name__ == "__main__":
    main()
