#!/usr/bin/env python3
"""Plot paired AlphaGoZero controller and trivial-policy arena costs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _model_version(path: Path) -> int | None:
    for parent in (path, *path.parents):
        match = re.fullmatch(r"eval_(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"game_id", "history_hops", "cycle1_total_cost", "cycle2_total_cost"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return sorted(rows, key=lambda row: int(row["game_id"]))


def plot(input_csv: Path, output_png: Path, summary_json: Path) -> dict[str, object]:
    rows = _read_rows(input_csv)
    if not rows:
        raise ValueError(f"{input_csv} has no game rows")

    game_ids = [int(row["game_id"]) for row in rows]
    hops = [int(row["history_hops"]) for row in rows]
    sjf_costs = np.asarray([float(row["cycle1_total_cost"]) for row in rows])
    model_costs = np.asarray([float(row["cycle2_total_cost"]) for row in rows])
    eps = 1e-9
    model_wins = int(np.sum(model_costs < sjf_costs - eps))
    sjf_wins = int(np.sum(sjf_costs < model_costs - eps))
    ties = int(len(rows) - model_wins - sjf_wins)
    version = _model_version(input_csv)
    model_label = f"V{version} model" if version is not None else "model controller"

    x = np.arange(len(rows), dtype=np.float64)
    width = 0.42
    fig, ax = plt.subplots(figsize=(22, 7.5), constrained_layout=True)
    ax.bar(x - width / 2, model_costs, width, label=model_label, color="#1769a5")
    ax.bar(x + width / 2, sjf_costs, width, label="SJF 256", color="#ff7f0e")

    max_cost = float(max(np.max(model_costs), np.max(sjf_costs)))
    label_offset = max(0.15, max_cost * 0.008)
    for index, (hop, model_cost, sjf_cost) in enumerate(zip(hops, model_costs, sjf_costs)):
        ax.text(
            x[index],
            max(model_cost, sjf_cost) + label_offset,
            f"h{hop}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#8a4b08",
            rotation=90,
        )

    count = len(rows)
    title = (
        f"{model_label} vs SJF-256 over {count} games\n"
        f"mean model cost={np.mean(model_costs):.2f}, mean SJF-256 cost={np.mean(sjf_costs):.2f} | "
        f"model wins {model_wins}/{count}, SJF wins {sjf_wins}/{count}, ties {ties}/{count}"
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("total cost (lower is better)")
    ax.set_xlabel("game id (last 3 digits); labels show history hop")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{game_id % 1000:03d}" for game_id in game_ids], rotation=90, fontsize=7)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_ylim(0.0, max_cost + max(2.0, max_cost * 0.09))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    summary: dict[str, object] = {
        "input_csv": str(input_csv.resolve()),
        "output_png": str(output_png.resolve()),
        "model_version": version,
        "games": count,
        "mean_model_total_cost": float(np.mean(model_costs)),
        "mean_sjf256_total_cost": float(np.mean(sjf_costs)),
        "mean_model_minus_sjf256_cost": float(np.mean(model_costs - sjf_costs)),
        "model_wins": model_wins,
        "sjf256_wins": sjf_wins,
        "ties": ties,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    output = args.output or args.input_csv.with_name("sjf256_cost_comparison.png")
    summary_path = args.summary or output.with_suffix(".summary.json")
    print(json.dumps(plot(args.input_csv, output, summary_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
