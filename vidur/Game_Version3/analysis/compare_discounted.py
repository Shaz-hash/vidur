#!/usr/bin/env python3
"""Per-game discounted-return comparison: V50 model vs SJF-128/256/512 trivials.

For each game id, compute G_0 (discounted sum of immediate rewards starting at
the first row) for the model trajectory and each of the three SJF trivial
trajectories, then pick the best (highest = least negative) trivial and decide
the per-game winner.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    from .reward import get_discounted_sum_reward
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vidur.Game_Version3.analysis.reward import get_discounted_sum_reward


REPO_ROOT = Path("/home/shazer/Desktop/Research/Vidur/vidur-classical-search")
RESULTS_ROOT = REPO_ROOT / "simulator_output/GV3_Agent/Model_Tester_Results"

MODEL_PRIMARY = RESULTS_ROOT / "v50_47x850_a2_sjf512_50games" / "arena_games"
MODEL_FALLBACK = RESULTS_ROOT / "v50_47x850_a2_sjf512_50games_full" / "arena_games"

SJF_DIRS = {
    "sjf128": RESULTS_ROOT / "v50_47x850_a2_sjf128_50games_cyc1" / "arena_games",
    "sjf256": RESULTS_ROOT / "v50_47x850_a2_sjf256_50games_cyc1" / "arena_games",
    "sjf512": RESULTS_ROOT / "v50_47x850_a2_sjf512_50games_full" / "arena_games",
}

OUTPUT_CSV = RESULTS_ROOT / "v50_47x850_a2_sjf512_50games_full" / "full_arena_discounted.csv"

GAME_IDS = list(range(12000000, 12000050))


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def total_step_discounted_rewards(rows: list[dict]) -> float:
    """Sum of G_t (discounted future return) over every action row.

    Equivalent to the loop:

        total = 0.0
        for row in rows:
            total += get_discounted_sum_reward(rows, rows.index(row))

    but uses enumerate (O(N) instead of O(N^2)) and is correct even when two
    rows compare equal as dicts (rows.index would always return the first
    match in that case). Skips arena_end summary rows since they are not
    actions.
    """
    total = 0.0
    for i, row in enumerate(rows):
        phase = row.get("phase", "") or ""
        if phase == "arena_end" or phase.startswith("history_step"):
            continue
        total += get_discounted_sum_reward(rows, i)
    return total


# def _game_score(path: Path) -> float:
#     rows = _load_rows(path)
#     if not rows:
#         return 0.0
#     return total_step_discounted_rewards(rows)

def _game_score(path: Path) -> float:
    rows = _load_rows(path)
    if not rows:
        return 0.0
    first_arena = next(
        (i for i, row in enumerate(rows) if row.get("phase") == "arena_step"),
        0,
    )
    return get_discounted_sum_reward(rows, first_arena)


def _model_path(game_id: int) -> Path:
    name = f"game_{game_id}_model_adv_depth1_vs_model_ctrl_depth1.csv"
    primary = MODEL_PRIMARY / name
    fallback = MODEL_FALLBACK / name
    # Prefer the file with more rows (the primary dir occasionally contains a
    # stub with only the adversary's opening row).
    if primary.exists() and fallback.exists():
        if sum(1 for _ in primary.open()) >= sum(1 for _ in fallback.open()):
            # print("Primary Path is : ", primary)
            return primary
        # print("Fallback Path is : ", fallback)
        return fallback
    if primary.exists():
        # print("Primary Path is : ", primary)
        return primary
    return fallback


def _trivial_path(sjf_dir: Path, game_id: int) -> Path:
    return sjf_dir / f"game_{game_id}_model_adv_depth1_vs_trivial_ctrl.csv"


def main() -> None:
    rows_out: list[dict] = []
    for gid in GAME_IDS:
        model_p = _model_path(gid)
        if not model_p.exists():
            print(f"[skip] missing model trajectory for {gid}: {model_p}")
            continue

        model_score = _game_score(model_p)

        trivial_scores: dict[str, float] = {}
        for tag, sjf_dir in SJF_DIRS.items():
            tp = _trivial_path(sjf_dir, gid)
            if not tp.exists():
                print(f"[warn] missing {tag} trivial for {gid}: {tp}")
                continue
            trivial_scores[tag] = _game_score(tp)

        if not trivial_scores:
            print(f"[skip] no trivial trajectories for {gid}")
            continue

        best_tag = max(trivial_scores, key=trivial_scores.get)
        best_val = trivial_scores[best_tag]

        if model_score > best_val:
            winner = "model"
        elif model_score < best_val:
            winner = best_tag
        else:
            winner = "tie"

        rows_out.append(
            {
                "game_number": gid,
                "model_discounted_sum": f"{model_score:.6f}",
                "best_trivial_discounted_sum": f"{best_val:.6f}",
                "best_trivial_policy": best_tag,
                "winner": winner,
                "sjf128_discounted_sum": f"{trivial_scores.get('sjf128', float('nan')):.6f}",
                "sjf256_discounted_sum": f"{trivial_scores.get('sjf256', float('nan')):.6f}",
                "sjf512_discounted_sum": f"{trivial_scores.get('sjf512', float('nan')):.6f}",
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_number",
        "model_discounted_sum",
        "best_trivial_discounted_sum",
        "best_trivial_policy",
        "winner",
        "sjf128_discounted_sum",
        "sjf256_discounted_sum",
        "sjf512_discounted_sum",
    ]
    with OUTPUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    n = len(rows_out)
    n_model = sum(1 for r in rows_out if r["winner"] == "model")
    n_trivial = sum(1 for r in rows_out if r["winner"] not in ("model", "tie"))
    n_tie = sum(1 for r in rows_out if r["winner"] == "tie")
    print(f"\nWrote {OUTPUT_CSV} ({n} rows)")
    print(f"WINS: model={n_model} best_trivial={n_trivial} ties={n_tie}")
    by_policy: dict[str, int] = {}
    for r in rows_out:
        if r["winner"] not in ("model", "tie"):
            by_policy[r["winner"]] = by_policy.get(r["winner"], 0) + 1
    if by_policy:
        print(f"trivial-win breakdown: {by_policy}")


if __name__ == "__main__":
    main()
