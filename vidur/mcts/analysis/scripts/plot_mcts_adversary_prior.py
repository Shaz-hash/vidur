#!/usr/bin/env python3
import csv
import io
import json

import matplotlib.pyplot as plt

LOG_SNIPPET = """game_id,num_simulations,mcts_root_prior_json
6000,4000,"[0.00825, 0.01275, 0.01475, 0.0265, 0.0655, 0.87225]"
"""

TITLE = "MCTS produced Prior for Adversary after training model on only 400 Samples."
OUT_PATH = "mcts_adversary_prior_bar.png"


def main():
    row = next(csv.DictReader(io.StringIO(LOG_SNIPPET)))
    probs = [float(x) for x in json.loads(row["mcts_root_prior_json"])]

    x = list(range(1, 7))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(x, probs, color=colors, edgecolor="black")

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xlabel("Number of Prefill Requests Generated (Size 3072)")
    ax.set_ylabel("MCTS Probability")
    ax.set_title(TITLE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.98, p + 0.015),
            f"{p:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
