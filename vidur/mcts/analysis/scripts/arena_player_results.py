#!/usr/bin/env python3
# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
Plot per-generation arena winner counts for candidate and best players.

Data source per generation:
  simulator_output/mcts_dnn_logs/gen_XXXXXX/candidate_arena_performance.csv
  simulator_output/mcts_dnn_logs/gen_XXXXXX/best_arena_performance.csv

Y-axis: number of games the target player won in that generation.
X-axis: generation index (e.g., 7..383).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


@dataclass
class GenPoint:
    gen: int
    wins: int


def _count_wins(csv_path: Path, *, winner_name: str) -> Optional[int]:
    if not csv_path.exists():
        return None
    winner_name = str(winner_name).strip().lower()
    wins = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("winner", "")).strip().lower() == winner_name:
                wins += 1
    return int(wins)


def _collect_points(
    logs_root: Path,
    *,
    start_gen: int,
    end_gen: int,
    filename: str,
    winner_name: str,
) -> List[GenPoint]:
    points: List[GenPoint] = []
    for gen in range(int(start_gen), int(end_gen) + 1):
        csv_path = logs_root / f"gen_{gen:06d}" / filename
        wins = _count_wins(csv_path, winner_name=winner_name)
        if wins is None:
            continue
        points.append(GenPoint(gen=int(gen), wins=int(wins)))
    return points


def _plot_wins(
    points: List[GenPoint],
    *,
    title: str,
    line_label: str,
    out_path: Path,
    max_games: int,
    color: str,
    dpi: int,
) -> None:
    if not points:
        raise RuntimeError(f"No points to plot for {line_label}")

    xs = [p.gen for p in points]
    ys = [p.wins for p in points]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9)
    ax.scatter(xs, ys, color=color, s=30, zorder=3, label=line_label)
    ax.axhline(
        y=int(max_games),
        color="red",
        linewidth=1.8,
        linestyle="-",
        label=f"Max arena games = {int(max_games)}",
    )
    ax.set_xlabel("Training Iteration (Generation)")
    ax.set_ylabel("Number of Games Won")
    ax.set_title(title)
    ax.set_ylim(0, max(int(max_games), max(ys)) + 1)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    if len(xs) > 1:
        step = max(1, len(xs) // 14)
        tick_x = [xs[i] for i in range(0, len(xs), step)]
        if xs[-1] not in tick_x:
            tick_x.append(xs[-1])
        ax.set_xticks(tick_x)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot per-generation arena winners for candidate and best.")
    ap.add_argument(
        "--logs-root",
        type=Path,
        default=Path("simulator_output/mcts_dnn_logs"),
        help="Root directory containing gen_XXXXXX folders.",
    )
    ap.add_argument("--start-gen", type=int, default=7)
    ap.add_argument("--end-gen", type=int, default=383)
    ap.add_argument("--max-games", type=int, default=20, help="Horizontal reference line value.")
    ap.add_argument(
        "--out-candidate",
        type=Path,
        default=Path("simulator_output/candidate_arena_performance.png"),
        help="Output PNG for candidate wins plot.",
    )
    ap.add_argument(
        "--out-best",
        type=Path,
        default=Path("simulator_output/best_arena_performance.png"),
        help="Output PNG for best wins plot.",
    )
    ap.add_argument("--dpi", type=int, default=180)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.start_gen) > int(args.end_gen):
        raise ValueError("start-gen cannot be greater than end-gen")

    candidate_points = _collect_points(
        Path(args.logs_root),
        start_gen=int(args.start_gen),
        end_gen=int(args.end_gen),
        filename="candidate_arena_performance.csv",
        winner_name="candidate",
    )
    best_points = _collect_points(
        Path(args.logs_root),
        start_gen=int(args.start_gen),
        end_gen=int(args.end_gen),
        filename="best_arena_performance.csv",
        winner_name="best",
    )

    _plot_wins(
        candidate_points,
        title=f"Candidate Arena Wins by Generation ({int(args.start_gen)}-{int(args.end_gen)})",
        line_label="Candidate Wins",
        out_path=Path(args.out_candidate),
        max_games=int(args.max_games),
        color="#1f77b4",
        dpi=int(args.dpi),
    )
    _plot_wins(
        best_points,
        title=f"Best Arena Wins by Generation ({int(args.start_gen)}-{int(args.end_gen)})",
        line_label="Best Wins",
        out_path=Path(args.out_best),
        max_games=int(args.max_games),
        color="#ff7f0e",
        dpi=int(args.dpi),
    )

    print(f"[arena_player_results] wrote {Path(args.out_candidate)} (points={len(candidate_points)})")
    print(f"[arena_player_results] wrote {Path(args.out_best)} (points={len(best_points)})")


if __name__ == "__main__":
    main()

