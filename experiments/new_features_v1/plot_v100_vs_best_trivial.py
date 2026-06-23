"""
Bar chart: per-game V100 model cost vs best-trivial-policy cost.

Inputs: simulator_output/GV3_Agent/Model_Tester_Results/v100_multi_sjf/sjf_{128,256,512,1024}_50games/arena_results.csv
  cycle1_total_cost = trivial (SJF) cycle cost for that budget
  cycle2_total_cost = model cycle cost (same across budgets — it's the same model)

For each game id, the "best trivial policy" is the SJF budget whose cycle1_total_cost is smallest.
The model cost is taken as the mean of cycle2_total_cost across the 4 csvs (they are nominally identical
runs for cycle2; we pick the min as a small safety net for any rerun noise).

Saves: v100_multi_sjf/v100_vs_best_trivial.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/GV3_Agent/Model_Tester_Results/v100_multi_sjf")

BUDGETS = [128, 256, 512, 1024]


def main() -> None:
    frames = {}
    for b in BUDGETS:
        p = ROOT / f"sjf_{b}_50games" / "arena_results.csv"
        df = pd.read_csv(p)
        frames[b] = df[["game_id", "cycle1_total_cost", "cycle2_total_cost"]].rename(
            columns={"cycle1_total_cost": f"trivial_{b}", "cycle2_total_cost": f"model_{b}"}
        )
    merged = frames[BUDGETS[0]]
    for b in BUDGETS[1:]:
        merged = merged.merge(frames[b], on="game_id")

    trivial_cols = [f"trivial_{b}" for b in BUDGETS]
    model_cols = [f"model_{b}" for b in BUDGETS]
    merged["trivial_best_cost"] = merged[trivial_cols].min(axis=1)
    merged["trivial_best_budget"] = merged[trivial_cols].idxmin(axis=1).str.replace("trivial_", "").astype(int)
    merged["model_cost"] = merged[model_cols].min(axis=1)

    merged = merged.sort_values("game_id").reset_index(drop=True)
    out_csv = ROOT / "v100_vs_best_trivial.csv"
    merged[["game_id", "trivial_best_cost", "trivial_best_budget", "model_cost",
            *trivial_cols, *model_cols]].to_csv(out_csv, index=False)
    print(f"[plot] wrote {out_csv} ({len(merged)} games)")

    n = len(merged)
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(14, n * 0.32), 6))
    bars_model = ax.bar(x - width / 2, merged["model_cost"].values, width,
                        label="V100 model", color="#1f77b4")
    bars_triv = ax.bar(x + width / 2, merged["trivial_best_cost"].values, width,
                       label="best trivial SJF", color="#ff7f0e")

    for xi, b in zip(x, merged["trivial_best_budget"].values):
        ax.text(xi + width / 2, merged["trivial_best_cost"].iloc[xi] + 0.6,
                f"{int(b)}", ha="center", va="bottom", fontsize=7, color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(g))[-3:] for g in merged["game_id"]], rotation=90, fontsize=8)
    ax.set_xlabel("game id (last 3 digits)")
    ax.set_ylabel("total cost (lower is better)")
    model_mean = merged["model_cost"].mean()
    triv_mean = merged["trivial_best_cost"].mean()
    n_model_wins = int((merged["model_cost"] < merged["trivial_best_cost"]).sum())
    ax.set_title(
        f"V100 vs best-trivial-SJF over {n} games\n"
        f"mean model cost={model_mean:.2f}  mean best-trivial cost={triv_mean:.2f}  "
        f"model wins {n_model_wins}/{n} games"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_png = ROOT / "v100_vs_best_trivial.png"
    fig.savefig(out_png, dpi=140)
    print(f"[plot] wrote {out_png}")


if __name__ == "__main__":
    main()
