"""Build a smoke replay-target CSV from a model-vs-model arena game log.

This is intentionally log-based for Phase 1 validation. Production replay should
write raw simulator states at game runtime, but this script validates target
semantics using the same complete trajectory that appears in arena CSV logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

DISCOUNT_FACTOR = 0.98
DISCOUNT_TIME_DENOM_SEC = 0.015725797204323228

FIELDS = [
    "game_id",
    "phase",
    "turn_number",
    "depth_number",
    "player",
    "canonical_action_count",
    "immediate_cost",
    "immediate_reward",
    "discount",
    "temperature",
    "value_by_model_at_state",
    "mcts_root_value",
    "target_value",
    "model_prior_top5",
    "target_mcts_distribution_top5",
    "time_at_state",
    "time_after_state",
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _json_list(value: Any) -> list[float]:
    if value is None or value == "":
        return []
    try:
        data = json.loads(value)
        if not isinstance(data, list):
            return []
        return [float(x) for x in data]
    except Exception:
        return []


def _time_discount(t_child: float, t_parent: float) -> float:
    dt = max(0.0, float(t_child) - float(t_parent))
    return float(DISCOUNT_FACTOR ** (dt / max(DISCOUNT_TIME_DENOM_SEC, 1e-9)))


def _read_steps(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    return [r for r in rows if str(r.get("phase", "")) == "arena_step"]


def _transition_reward(row: dict[str, Any], prev_total_cost: float) -> tuple[float, float]:
    current_total_cost = _float(row.get("total_cost"), prev_total_cost)
    immediate_cost = current_total_cost - float(prev_total_cost)
    if str(row.get("chosen_reward", "")) != "":
        immediate_reward = _float(row.get("chosen_reward"), -immediate_cost)
    else:
        # Native MCTS uses controller-value convention: reward is parent_cost - child_cost.
        immediate_reward = -float(immediate_cost)
    return float(immediate_cost), float(immediate_reward)


def _transition_discount(row: dict[str, Any]) -> float:
    if str(row.get("chosen_discount", "")) != "":
        return _float(row.get("chosen_discount"), 1.0)
    return _time_discount(_float(row.get("sim_time_after")), _float(row.get("sim_time_before")))


def _final_bootstrap(rows: list[dict[str, Any]]) -> float:
    # For the last searched action, chosen_bootstrap is V_old(s_T), the model value
    # of the final child state after that action. This matches the requested
    # truncated-return bootstrap for the smoke run.
    for row in reversed(rows):
        if str(row.get("chosen_bootstrap", "")) != "":
            return _float(row.get("chosen_bootstrap"), 0.0)
    return 0.0


def build_target_rows(arena_csv: Path, *, min_canonical_actions: int) -> list[dict[str, Any]]:
    steps = _read_steps(arena_csv)
    if not steps:
        return []

    prev_cost = 0.0
    transitions: list[dict[str, Any]] = []
    for row in steps:
        immediate_cost, immediate_reward = _transition_reward(row, prev_cost)
        discount = _transition_discount(row)
        current_cost = _float(row.get("total_cost"), prev_cost)
        transitions.append(
            {
                "row": row,
                "immediate_cost": immediate_cost,
                "immediate_reward": immediate_reward,
                "discount": discount,
            }
        )
        prev_cost = current_cost

    next_value = _final_bootstrap(steps)
    for item in reversed(transitions):
        item["target_value"] = float(item["immediate_reward"] + item["discount"] * next_value)
        next_value = float(item["target_value"])

    output: list[dict[str, Any]] = []
    for item in transitions:
        row = item["row"]
        canon_n = _int(row.get("canonical_action_count"), _int(row.get("valid_action_count")))
        if canon_n <= int(min_canonical_actions):
            continue
        mcts_probs = _json_list(row.get("candidate_top5_mcts_probs"))
        if not mcts_probs:
            visits = _json_list(row.get("candidate_top5_visits"))
            denom = max(1.0, _float(row.get("iterations_used"), sum(visits) or 1.0))
            mcts_probs = [float(v) / denom for v in visits]
        output.append(
            {
                "game_id": row.get("game_id", ""),
                "phase": row.get("phase", ""),
                "turn_number": row.get("turn", ""),
                "depth_number": row.get("depth", ""),
                "player": row.get("player_acted", ""),
                "canonical_action_count": canon_n,
                "immediate_cost": item["immediate_cost"],
                "immediate_reward": item["immediate_reward"],
                "discount": item["discount"],
                "temperature": _float(row.get("mcts_action_temperature", row.get("policy_prior_temperature")), 1.0),
                "value_by_model_at_state": _float(row.get("model_value_at_state"), 0.0),
                "mcts_root_value": _float(row.get("mcts_root_value"), 0.0),
                "target_value": item["target_value"],
                "model_prior_top5": json.dumps(_json_list(row.get("candidate_top5_priors"))),
                "target_mcts_distribution_top5": json.dumps(mcts_probs),
                "time_at_state": _float(row.get("sim_time_before"), 0.0),
                "time_after_state": _float(row.get("sim_time_after"), 0.0),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AlphaGoZero replay target smoke CSV from an arena game log.")
    parser.add_argument("--arena-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--min-canonical-actions", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_target_rows(args.arena_csv, min_canonical_actions=int(args.min_canonical_actions))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
