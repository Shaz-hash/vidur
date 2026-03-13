#!/usr/bin/env python3
"""
Plot arena SLO cost vs training samples from arena logs.

Data source:
  simulator_output/mcts_dnn_logs/gen_XXXXXX/arena_results.csv

For each generation, this script reads the FIRST data row from arena_results.csv
and uses one of:
  y = best_as_adv_cost      (series=best)
  y = candidate_as_adv_cost (series=candidate)

X-axis uses sample count progression where each generation contributes a fixed
number of samples:
  x = (gen - start_gen + 1) * samples_per_gen

Example:
  python3 -m vidur.mcts.analysis.scripts.plot_slo_vs_training_samples \
    --logs-root simulator_output/mcts_dnn_logs \
    --start-gen 6 \
    --end-gen 180 \
    --samples-per-gen 1250 \
    --series best \
    --baseline 4.876256857
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


@dataclass
class Point:
    gen: int
    samples: int
    slo_cost: float


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _series_to_column(series: str) -> str:
    s = str(series).strip().lower()
    if s == "candidate":
        return "candidate_as_adv_cost"
    return "best_as_adv_cost"


def _read_first_row_cost(csv_path: Path, *, cost_column: str) -> Optional[float]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
        if first is None:
            return None
        if cost_column not in first:
            return None
        return _safe_float(first.get(cost_column), default=0.0)


def _collect_points(
    logs_root: Path,
    *,
    start_gen: int,
    end_gen: int,
    samples_per_gen: int,
    cost_column: str,
) -> List[Point]:
    out: List[Point] = []
    for gen in range(int(start_gen), int(end_gen) + 1):
        csv_path = logs_root / f"gen_{gen:06d}" / "arena_results.csv"
        y = _read_first_row_cost(csv_path, cost_column=cost_column)
        if y is None:
            continue
        x = (int(gen) - int(start_gen) + 1) * int(samples_per_gen)
        out.append(Point(gen=int(gen), samples=int(x), slo_cost=float(y)))
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot SLO trend from arena_results.csv across generations.")
    ap.add_argument(
        "--logs-root",
        type=Path,
        default=Path("simulator_output/mcts_dnn_logs"),
        help="Root dir containing gen_XXXXXX folders.",
    )
    ap.add_argument("--start-gen", type=int, default=6)
    ap.add_argument("--end-gen", type=int, default=111)
    ap.add_argument("--samples-per-gen", type=int, default=1250)
    ap.add_argument(
        "--series",
        type=str,
        choices=["candidate", "best"],
        default="best",
        help="Which arena series to plot from first row: candidate_as_adv_cost or best_as_adv_cost.",
    )
    ap.add_argument("--baseline", type=float, default=4.876256857)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output plot path.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=180,
    )
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    cost_column = _series_to_column(args.series)

    out_path: Path
    if args.out is None:
        out_path = Path(
            f"simulator_output/mcts_dnn_logs/slo_vs_training_samples_{args.series}_"
            f"gen_{int(args.start_gen):06d}_{int(args.end_gen):06d}.png"
        )
    else:
        out_path = Path(args.out)

    points = _collect_points(
        args.logs_root,
        start_gen=int(args.start_gen),
        end_gen=int(args.end_gen),
        samples_per_gen=int(args.samples_per_gen),
        cost_column=cost_column,
    )
    if not points:
        raise RuntimeError(
            f"No data points found under {args.logs_root} for generations "
            f"{args.start_gen}..{args.end_gen}"
        )

    xs = [p.samples for p in points]
    ys = [p.slo_cost for p in points]
    gens = [p.gen for p in points]

    below = [p for p in points if p.slo_cost < float(args.baseline)]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(
        xs,
        ys,
        color="#1f77b4",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=f"Model's SLO Cost ({cost_column})",
    )
    ax.axhline(
        y=float(args.baseline),
        color="red",
        linestyle="-",
        linewidth=1.8,
        label=f"SJF-1024 ({args.baseline:.9f})",
    )

    if below:
        bx = [p.samples for p in below]
        by = [p.slo_cost for p in below]
        ax.scatter(
            bx,
            by,
            color="#2ca02c",
            s=42,
            zorder=4,
            label=f"Below trivial strategy (n={len(below)})",
        )

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("SLO Violations Cost")
    ax.set_title(
        f"SLO Cost vs Training Samples ({args.series}, gen_{args.start_gen:06d}..gen_{args.end_gen:06d})"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    ax.text(
        0.01,
        0.99,
        f"Below trivial strategy count: {len(below)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#999999"),
    )

    # Keep x ticks readable.
    if len(xs) > 1:
        step = max(1, len(xs) // 12)
        tick_x = [xs[i] for i in range(0, len(xs), step)]
        tick_lbl = [f"{x:,}" for x in tick_x]
        ax.set_xticks(tick_x)
        ax.set_xticklabels(tick_lbl, rotation=25, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))

    print(f"Saved plot: {out_path}")
    print(f"Points plotted: {len(points)}")
    print(f"Generations used: min={min(gens)} max={max(gens)}")
    print(f"Series: {args.series} ({cost_column})")
    print(f"Below baseline ({args.baseline:.9f}): {len(below)} points")


if __name__ == "__main__":
    main()
