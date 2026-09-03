"""Print discounted immediate-cost returns from an arena game CSV.

Default behavior computes, for turn 1:

    cost_1 + discount_1 * (cost_2 + discount_2 * (...))

where cost_i is the incremental objective-cost increase between consecutive
arena rows. Rows with no MCTS search are still included in the recurrence.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

DEFAULT_ARENA_CSV = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/AlphaGoZero/phase1_local_smoke/"
    "iter1000_hop0_gid17000001/jobs/game_17000001_hop_0/arena_games/"
    "game_17000001_model_adv_depth1_vs_model_ctrl_depth1.csv"
)

DISCOUNT_FACTOR = 0.995
DISCOUNT_TIME_DENOM_SEC = 0.015725797204323228


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _time_discount(time_after: float, time_before: float) -> float:
    dt = max(0.0, float(time_after) - float(time_before))
    return float(DISCOUNT_FACTOR ** (dt / max(DISCOUNT_TIME_DENOM_SEC, 1e-9)))


def _read_arena_steps(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    return [row for row in rows if str(row.get("phase", "")) == "arena_step"]


def _build_transitions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    prev_total_cost = 0.0
    for row in rows:
        total_cost = _to_float(row.get("total_cost"), prev_total_cost)
        immediate_cost = float(total_cost - prev_total_cost)
        if str(row.get("chosen_discount", "")) != "":
            discount = _to_float(row.get("chosen_discount"), 1.0)
        else:
            discount = _time_discount(
                _to_float(row.get("sim_time_after")),
                _to_float(row.get("sim_time_before")),
            )
        transitions.append(
            {
                "turn": _to_int(row.get("turn")),
                "depth": _to_int(row.get("depth")),
                "player": str(row.get("player_acted", "")),
                "valid_action_count": _to_int(row.get("valid_action_count")),
                "canonical_action_count": _to_int(row.get("canonical_action_count"), _to_int(row.get("valid_action_count"))),
                "immediate_cost": immediate_cost,
                "discount": float(discount),
                "time_before": _to_float(row.get("sim_time_before")),
                "time_after": _to_float(row.get("sim_time_after")),
                "total_cost": total_cost,
            }
        )
        prev_total_cost = total_cost
    return transitions


def _discounted_cost_returns(transitions: list[dict[str, Any]], terminal_bootstrap_cost: float = 0.0) -> list[float]:
    returns = [0.0] * len(transitions)
    running = float(terminal_bootstrap_cost)
    for idx in range(len(transitions) - 1, -1, -1):
        step = transitions[idx]
        running = float(step["immediate_cost"] + step["discount"] * running)
        returns[idx] = running
    return returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute discounted immediate-cost returns from an arena CSV.")
    parser.add_argument("--arena-csv", type=Path, default=DEFAULT_ARENA_CSV)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--terminal-bootstrap-cost", type=float, default=0.0)
    parser.add_argument("--print-first", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_arena_steps(args.arena_csv)
    transitions = _build_transitions(rows)
    returns = _discounted_cost_returns(
        transitions,
        terminal_bootstrap_cost=float(args.terminal_bootstrap_cost),
    )

    matching_indices = [idx for idx, step in enumerate(transitions) if int(step["turn"]) == int(args.turn)]
    if not matching_indices:
        raise SystemExit(f"turn {args.turn} not found in {args.arena_csv}")
    idx = matching_indices[0]
    step = transitions[idx]
    print(f"arena_csv={args.arena_csv}")
    print(f"num_arena_steps={len(transitions)}")
    print(f"turn={args.turn}")
    print(f"turn_index={idx}")
    print(f"turn_player={step['player']}")
    print(f"turn_immediate_cost={step['immediate_cost']}")
    print(f"turn_discount={step['discount']}")
    print(f"discounted_immediate_cost_return_from_turn={returns[idx]}")
    print()
    print("first_rows:")
    header = "idx,turn,player,canon_actions,immediate_cost,discount,return,time_before,time_after,total_cost"
    print(header)
    for j, row in enumerate(transitions[: max(0, int(args.print_first))]):
        print(
            f"{j},{row['turn']},{row['player']},{row['canonical_action_count']},"
            f"{row['immediate_cost']},{row['discount']},{returns[j]},"
            f"{row['time_before']},{row['time_after']},{row['total_cost']}"
        )


if __name__ == "__main__":
    main()
