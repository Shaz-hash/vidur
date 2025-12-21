"""
Plot longitudinal visit curves for Parallel Launch MCTS runs.

This script reads, for each process directory:
  simulator_output/Parrallel_Launch/analysis/P#/process_iteration_longitudnal.csv

and produces a clean line graph:
  - x-axis: MCTS iteration
  - y-axis: visits
  - one colored line per controller token budget
  - point markers shown every N iterations (default 2000) and at the last iteration

Legend budget labels are bucketed by rounding the controller token budget down to the
nearest multiple of 512 (e.g., 579 -> 512).

Output is written next to the CSV (same directory) as:
  process_iteration_visits.svg

Sample commands:
  # Plot all runs found under simulator_output/Parrallel_Launch/analysis/
  python3 -m vidur.mcts.analysis.plot_process_iteration_longitudinal

  # Plot a single run and customize marker spacing
  python3 -m vidur.mcts.analysis.plot_process_iteration_longitudinal --run_ids 10 --marker_stride 2000

  # Use a custom Parallel Launch directory
  python3 -m vidur.mcts.analysis.plot_process_iteration_longitudinal --par_dir simulator_output/Parrallel_Launch_Q10
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _read_process_iteration_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _budget_bucket(budget: str, multiple: int = 512) -> int:
    b = _safe_int(budget, 0)
    if multiple <= 0:
        return b
    return (b // multiple) * multiple


def _discover_run_ids(analysis_root: Path) -> List[str]:
    discovered = [p.name.lstrip("P") for p in analysis_root.glob("P*") if p.is_dir()]
    return sorted(discovered, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))


def _pick_marker_iters(iters: List[int], *, marker_stride: int) -> set[int]:
    if not iters:
        return set()
    marker_stride = max(1, int(marker_stride))
    max_it = max(iters)
    out = {it for it in iters if it % marker_stride == 0}
    out.add(max_it)
    return out


def plot_one_run(
    process_csv: Path,
    *,
    marker_stride: int,
    budget_multiple: int,
    out_name: str,
) -> None:
    rows = _read_process_iteration_csv(process_csv)
    if not rows:
        print(f"[WARN] Empty or missing CSV: {process_csv}", file=sys.stderr)
        return

    # Group points by budget bucket.
    points_by_budget: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
    all_iters: List[int] = []

    for r in rows:
        it = _safe_int(r.get("iteration"), -1)
        visits = _safe_int(r.get("visits"), 0)
        bud = _budget_bucket(r.get("controller_token_budget", ""), multiple=budget_multiple)
        if it < 0:
            continue
        points_by_budget[bud].append((it, visits))
        all_iters.append(it)

    if not points_by_budget:
        print(f"[WARN] No usable rows in {process_csv}", file=sys.stderr)
        return

    # Sort each series and dedupe by iteration (keep max visits if duplicated).
    series: Dict[int, List[Tuple[int, int]]] = {}
    for bud, pts in points_by_budget.items():
        by_it: Dict[int, int] = {}
        for it, v in pts:
            prev = by_it.get(it)
            by_it[it] = v if prev is None else max(prev, v)
        series[bud] = sorted(by_it.items(), key=lambda t: t[0])

    marker_iters = _pick_marker_iters(all_iters, marker_stride=marker_stride)

    budgets_sorted = sorted(series.keys())

    # Title hint requested by the user:
    # "Visits vs Iterations (Input queue size <= y)" where:
    #   y = max over controller budgets of (requests_in_system - controller_decode_total)
    # Values are sourced from the sibling analysis.csv in the same P#/ directory.
    y_hint = _compute_input_queue_hint(process_csv, rows, budget_multiple=budget_multiple)

    # Render a self-contained SVG (no external plotting deps).
    run_dir = process_csv.parent
    run_id = run_dir.name
    out_path = run_dir / out_name
    svg = render_svg(
        title=_format_title(run_id, y_hint=y_hint),
        x_label="Iteration",
        y_label="Visits",
        legend_title=f"Controller token budget (floored to {budget_multiple})",
        budgets_sorted=budgets_sorted,
        series=series,
        marker_iters=marker_iters,
    )
    out_path.write_text(svg, encoding="utf-8")
    print(f"[INFO] Wrote plot: {out_path}")


def _format_title(run_id: str, *, y_hint: Optional[int]) -> str:
    if y_hint is None:
        return f"Visits vs Iteration ({run_id})"
    return f"Visits vs Iteration (Input queue size <= {y_hint}) ({run_id})"


def _compute_input_queue_hint(
    process_csv: Path,
    process_rows: List[Dict[str, str]],
    *,
    budget_multiple: int,
) -> Optional[int]:
    """
    Compute the value 'y' for the plot title:

      For each controller budget option:
        input_queue = requests_in_system - controller_decode_total

    We use node_ids present in process_iteration_longitudnal.csv to look up the
    corresponding rows in the sibling analysis.csv (same directory).

    Returns:
      max(input_queue) across budgets (a safe upper bound), or None if unavailable.
    """
    analysis_csv = process_csv.parent / "analysis.csv"
    if not analysis_csv.exists():
        return None

    node_ids = {str(r.get("node_id", "")).strip() for r in process_rows}
    node_ids.discard("")
    if not node_ids:
        return None

    by_node: Dict[str, Dict[str, str]] = {}
    with analysis_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            nid = str(r.get("node_id", "")).strip()
            if nid:
                by_node[nid] = r

    budgets = {
        _budget_bucket(r.get("controller_token_budget", ""), multiple=budget_multiple)
        for r in process_rows
    }
    budgets.discard(0)

    max_hint: Optional[int] = None
    for bud in budgets:
        nid_for_budget: Optional[str] = None
        for r in process_rows:
            rb = _budget_bucket(r.get("controller_token_budget", ""), multiple=budget_multiple)
            if rb == bud:
                nid_for_budget = str(r.get("node_id", "")).strip()
                if nid_for_budget:
                    break

        if not nid_for_budget:
            continue
        arow = by_node.get(nid_for_budget)
        if not arow:
            continue

        req_in_system = _safe_int(arow.get("requests_in_system"), 0)
        decode_total = _safe_int(arow.get("controller_decode_total"), 0)
        hint = req_in_system - decode_total
        if max_hint is None or hint > max_hint:
            max_hint = hint

    return max_hint


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exp = math.floor(math.log10(raw_step))
    frac = raw_step / (10**exp)
    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exp)


def render_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    budgets_sorted: List[int],
    series: Dict[int, List[Tuple[int, int]]],
    marker_iters: set[int],
) -> str:
    # Canvas + layout
    width, height = 1200, 650
    margin_left, margin_right, margin_top, margin_bottom = 90, 280, 70, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Data ranges
    all_x: List[int] = []
    all_y: List[int] = []
    for pts in series.values():
        for x, y in pts:
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        raise ValueError("No points to plot")

    x_min = min(0, min(all_x))
    x_max = max(all_x)
    y_min = 0
    y_max = max(all_y)
    if y_max <= 0:
        y_max = 1

    def x_to_px(x: int) -> float:
        if x_max == x_min:
            return float(margin_left)
        return margin_left + (x - x_min) * plot_w / (x_max - x_min)

    def y_to_px(y: int) -> float:
        return margin_top + plot_h - (y - y_min) * plot_h / (y_max - y_min)

    # Ticks/grid
    x_tick_target = 6
    y_tick_target = 6
    x_step = _nice_step((x_max - x_min) / max(1, x_tick_target))
    y_step = _nice_step((y_max - y_min) / max(1, y_tick_target))

    def frange(start: float, stop: float, step: float) -> Iterable[float]:
        v = start
        # add small epsilon so we include the stop when close
        while v <= stop + 1e-9:
            yield v
            v += step

    # Palette + markers
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    marker_kinds = ["circle", "square", "triangle", "diamond", "cross", "x"]

    def marker_svg(kind: str, cx: float, cy: float, color: str) -> str:
        size = 5.0
        if kind == "circle":
            return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{color}" />'
        if kind == "square":
            return f'<rect x="{(cx-size):.2f}" y="{(cy-size):.2f}" width="{(2*size):.2f}" height="{(2*size):.2f}" fill="{color}" />'
        if kind == "triangle":
            p1 = (cx, cy - size * 1.2)
            p2 = (cx - size, cy + size)
            p3 = (cx + size, cy + size)
            return (
                f'<polygon points="{p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}" '
                f'fill="{color}" />'
            )
        if kind == "diamond":
            p1 = (cx, cy - size * 1.2)
            p2 = (cx - size, cy)
            p3 = (cx, cy + size * 1.2)
            p4 = (cx + size, cy)
            return (
                f'<polygon points="{p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f} {p4[0]:.2f},{p4[1]:.2f}" '
                f'fill="{color}" />'
            )
        if kind == "cross":
            s = size * 1.2
            return (
                f'<line x1="{(cx-s):.2f}" y1="{cy:.2f}" x2="{(cx+s):.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="2"/>'
                f'<line x1="{cx:.2f}" y1="{(cy-s):.2f}" x2="{cx:.2f}" y2="{(cy+s):.2f}" stroke="{color}" stroke-width="2"/>'
            )
        if kind == "x":
            s = size * 1.1
            return (
                f'<line x1="{(cx-s):.2f}" y1="{(cy-s):.2f}" x2="{(cx+s):.2f}" y2="{(cy+s):.2f}" stroke="{color}" stroke-width="2"/>'
                f'<line x1="{(cx-s):.2f}" y1="{(cy+s):.2f}" x2="{(cx+s):.2f}" y2="{(cy-s):.2f}" stroke="{color}" stroke-width="2"/>'
            )
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{color}" />'

    # SVG header
    parts: List[str] = []
    def esc(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')

    # Title
    parts.append(
        f'<text x="{width/2:.2f}" y="35" text-anchor="middle" font-family="Arial" font-size="20" fill="#111">{esc(title)}</text>'
    )

    # Axes + grid (Y)
    for yv in frange(0.0, float(y_max), y_step):
        py = y_to_px(int(yv))
        parts.append(
            f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left+plot_w}" y2="{py:.2f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left-10}" y="{py+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#333">{int(yv)}</text>'
        )

    # Axes + grid (X)
    for xv in frange(float(x_min), float(x_max), x_step):
        px = x_to_px(int(xv))
        parts.append(
            f'<line x1="{px:.2f}" y1="{margin_top}" x2="{px:.2f}" y2="{margin_top+plot_h}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{margin_top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{int(xv)}</text>'
        )

    # Axis lines
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#222" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#222" stroke-width="2"/>'
    )

    # Axis labels
    parts.append(
        f'<text x="{margin_left+plot_w/2:.2f}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111">{esc(x_label)}</text>'
    )
    parts.append(
        f'<text x="22" y="{margin_top+plot_h/2:.2f}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111" transform="rotate(-90 22 {margin_top+plot_h/2:.2f})">{esc(y_label)}</text>'
    )

    # Plot series
    for i, bud in enumerate(budgets_sorted):
        pts = series[bud]
        color = colors[i % len(colors)]
        marker = marker_kinds[i % len(marker_kinds)]

        # Polyline
        path = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in pts)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')

        # Markers
        for x, y in pts:
            if x in marker_iters:
                parts.append(marker_svg(marker, x_to_px(x), y_to_px(y), color))

    # Legend
    legend_x = margin_left + plot_w + 20
    legend_y = margin_top + 10
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-family="Arial" font-size="14" fill="#111">{esc(legend_title)}</text>'
    )
    ly = legend_y + 20
    for i, bud in enumerate(budgets_sorted):
        color = colors[i % len(colors)]
        marker = marker_kinds[i % len(marker_kinds)]
        parts.append(marker_svg(marker, legend_x + 8, ly - 4, color))
        parts.append(
            f'<text x="{legend_x+20}" y="{ly}" font-family="Arial" font-size="13" fill="#111">{esc(f"budget≈{bud}")}</text>'
        )
        ly += 22

    parts.append("</svg>")
    return "\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot visits vs iteration per token budget.")
    p.add_argument(
        "--par_dir",
        type=str,
        default="simulator_output/Parrallel_Launch",
        help="Parallel launch directory (contains analysis/).",
    )
    p.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Analysis directory (defaults to <par_dir>/analysis).",
    )
    p.add_argument(
        "--run_ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional run IDs to plot (e.g. 1 4 10). Default: all discovered.",
    )
    p.add_argument(
        "--in_name",
        type=str,
        default="process_iteration_longitudnal.csv",
        help="Input CSV name inside each analysis/P#/ directory.",
    )
    p.add_argument(
        "--out_name",
        type=str,
        default="process_iteration_visits.svg",
        help="Output image name written next to the input CSV.",
    )
    p.add_argument(
        "--marker_stride",
        type=int,
        default=2000,
        help="Show point markers every N iterations (also shows the last iteration).",
    )
    p.add_argument(
        "--budget_multiple",
        type=int,
        default=512,
        help="Legend bucketing: floor budget to nearest multiple of this value.",
    )

    args = p.parse_args(argv)

    par_dir = Path(args.par_dir)
    analysis_root = Path(args.analysis_dir) if args.analysis_dir else (par_dir / "analysis")
    if not analysis_root.exists():
        print(f"[ERROR] analysis_dir not found: {analysis_root}", file=sys.stderr)
        return 2

    if args.run_ids:
        run_ids = [str(x).lstrip("P") for x in args.run_ids]
    else:
        run_ids = _discover_run_ids(analysis_root)

    if not run_ids:
        print(f"[WARN] No runs found under {analysis_root}", file=sys.stderr)
        return 0

    for rid in run_ids:
        process_csv = analysis_root / f"P{rid}" / args.in_name
        if not process_csv.exists():
            print(f"[WARN] Missing {args.in_name} for P{rid}: {process_csv}", file=sys.stderr)
            continue
        plot_one_run(
            process_csv,
            marker_stride=args.marker_stride,
            budget_multiple=args.budget_multiple,
            out_name=args.out_name,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
