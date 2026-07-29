#!/usr/bin/env python3



import argparse
import csv
import json
from pathlib import Path



def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_action_reward(rows: list[dict], row_index: int) -> float:
    row = rows[row_index]

    # Model-scored rows already log the Bellman immediate reward.
    chosen_reward = row.get("chosen_reward", "")
    if chosen_reward not in (None, ""):
        return _safe_float(chosen_reward)

    # Trivial/history rows do not log chosen_reward, so reconstruct:
    # reward = parent_cost - child_cost = -(child_cost - parent_cost)
    child_cost = _safe_float(row.get("total_cost"))

    if row_index <= 0:
        parent_cost = 0.0
    else:
        parent_cost = _safe_float(rows[row_index - 1].get("total_cost"))

    return parent_cost - child_cost


def get_discounted_sum_reward(rows: list[dict], row_index: int) -> float:
    discount_factor = 0.995
    step_time = 0.015725797204323228

    row_time = _safe_float(rows[row_index].get("sim_time_before"))
    reward_sum = 0.0

    for i in range(row_index, len(rows)):
        row_i = rows[i]

        # Skip non-action summary rows.
        if row_i.get("phase") == "arena_end":
            continue

        reward = _row_action_reward(rows, i)
        sim_time_before = _safe_float(row_i.get("sim_time_before"), row_time)

        reward_sum += reward * discount_factor ** (
            (sim_time_before - row_time) / step_time
        )

    return reward_sum




def print_game(path: Path, show_candidates: bool = False) -> None:
    print("\n" + "=" * 120)
    print(f"GAME FILE: {path.name}")
    print("=" * 120)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("empty file")
        return

    first = rows[0]
    print(f"game_id={first.get('game_id', '')} cycle={first.get('cycle_label', '')} rows={len(rows)}")
    print("Reward Boostrap Calculation Starting .... ")
    # row_step_rewards = 0
    # row_step_rewards = get_discounted_sum_reward(rows, 0)
    
    for row in rows:

        row_step_reward = get_discounted_sum_reward(rows, rows.index(row))        
        row_step_rewards += row_step_reward

    print("-" * 120)
    print("Total Bootstrap Reward for this Game: ", row_step_rewards)
    

       


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Arena result directory containing arena_games/.",
    )
    parser.add_argument("--game-id", type=str, default="", help="Optional game id filter, e.g. 50001000.")
    parser.add_argument("--cycle", type=str, default="", help="Optional cycle substring filter.")
    parser.add_argument("--show-candidates", action="store_true", help="Print top-5 candidate action details.")
    args = parser.parse_args()

    arena_dir = args.base_dir / "arena_games"
    paths = sorted(arena_dir.glob("game_*.csv"))

    if args.game_id:
        paths = [p for p in paths if f"game_{args.game_id}_" in p.name]
    if args.cycle:
        paths = [p for p in paths if args.cycle in p.name]

    if not paths:
        raise SystemExit(f"No game CSVs found under {arena_dir}")

    for path in paths:
        print_game(path, show_candidates=args.show_candidates)


if __name__ == "__main__":
    main()
