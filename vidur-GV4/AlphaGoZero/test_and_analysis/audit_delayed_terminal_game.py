"""Audit delayed-terminal AlphaGoZero replay targets against a full game trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from vidur.Game_Version3.config import GameVersion2Config
from vidur.AlphaGoZero.replay_runtime import (
    DISCOUNT_TIME_DENOM_SEC,
    discounted_trajectory_targets,
)

ALLOWED_PREFILL_TOKENS = {
    int(value) for value in GameVersion2Config().request.allowed_prefill_tokens
}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float(default)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _json_list(value: str) -> list[int]:
    parsed = json.loads(value or "[]")
    return [int(item) for item in parsed] if isinstance(parsed, list) else []


def _json_int_map(value: str) -> dict[int, int]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        return {}
    return {int(key): int(item) for key, item in parsed.items()}


def _repr_int_map(action_repr: str, field: str) -> dict[int, int]:
    match = re.search(rf"{re.escape(field)}=\{{([^}}]*)\}}", action_repr)
    if not match:
        return {}
    result: dict[int, int] = {}
    for pair in match.group(1).split(","):
        if ":" in pair:
            key, value = pair.split(":", 1)
            result[int(key.strip())] = int(value.strip())
    return result


def _semantic_audit(rows: list[dict[str, str]], tolerance: float) -> dict[str, object]:
    prior_time = _float(rows[0], "sim_time_before")
    prior_cost = 0.0
    prior_violations = 0
    prior_completed: set[int] = set()
    prior_active: set[int] = set()
    all_seen: set[int] = set()
    controller_rows = 0
    adversary_rows = 0
    for index, row in enumerate(rows):
        turn = int(float(row["turn"]))
        depth = int(float(row["depth"]))
        actor = row["player_acted"]
        before = _float(row, "sim_time_before")
        after = _float(row, "sim_time_after")
        cost = _float(row, "total_cost")
        violations = int(float(row["slo_violations"]))
        lateness = _float(row, "total_lateness")
        active = set(_json_list(row["active_request_ids"]))
        completed = set(_json_list(row["completed_request_ids"]))
        prefill = set(_json_int_map(row["prefill_remaining_by_id"]))

        if turn != index or depth != index:
            raise AssertionError(f"non-consecutive turn/depth at row {index}")
        if actor not in {"controller", "adversary"}:
            raise AssertionError(f"invalid actor at row {index}: {actor}")
        if row["player_to_act_next"] == actor:
            raise AssertionError(f"actor did not alternate at row {index}")
        if abs(before - prior_time) > tolerance or after + tolerance < before:
            raise AssertionError(f"non-monotonic simulator time at row {index}")
        if cost + tolerance < prior_cost:
            raise AssertionError(f"non-monotonic objective cost at row {index}")
        if abs(cost - (violations + lateness)) > tolerance:
            raise AssertionError(f"objective identity failed at row {index}")
        if not prior_completed.issubset(completed):
            raise AssertionError(f"completed request disappeared at row {index}")
        if active & completed or not prefill.issubset(active):
            raise AssertionError(f"invalid request lifecycle at row {index}")
        if int(float(row["decode_credit_balance"])) < 0:
            raise AssertionError(f"negative decode credit at row {index}")

        action = row["action_repr"]
        if actor == "controller":
            controller_rows += 1
            allocations = _repr_int_map(action, "token_allocations")
            prefill_allocations = _repr_int_map(action, "prefill_allocations")
            decode_allocations = _repr_int_map(action, "decode_allocations")
            selected_match = re.search(r"selected_request_ids=\[([^\]]*)\]", action)
            selected = {
                int(item.strip())
                for item in (selected_match.group(1).split(",") if selected_match else [])
                if item.strip()
            }
            budget_match = re.search(r"token_budget=(\d+)", action)
            budget = int(budget_match.group(1)) if budget_match else 0
            if set(allocations) != selected or sum(allocations.values()) != budget:
                raise AssertionError(f"controller budget mismatch at row {index}")
            for request_id, amount in allocations.items():
                if prefill_allocations.get(request_id, 0) + decode_allocations.get(request_id, 0) != amount:
                    raise AssertionError(f"controller allocation split mismatch at row {index}")
            if prior_active and not selected.issubset(prior_active):
                raise AssertionError(f"controller selected unknown request at row {index}")
        else:
            adversary_rows += 1
            if abs(after - before) > tolerance:
                raise AssertionError(f"adversary action advanced simulator time at row {index}")
            tokens = {int(item) for item in re.findall(r"prefill_tokens=(\d+)", action)}
            if not tokens.issubset(ALLOWED_PREFILL_TOKENS):
                raise AssertionError(f"invalid adversary prefill size at row {index}")

        all_seen.update(active)
        all_seen.update(completed)
        prior_time = after
        prior_cost = cost
        prior_violations = violations
        prior_completed = completed
        prior_active = active

    if all_seen and all_seen != set(range(max(all_seen) + 1)):
        raise AssertionError("request IDs are not globally contiguous")
    return {
        "semantic_rows_checked": len(rows),
        "controller_rows_checked": controller_rows,
        "adversary_rows_checked": adversary_rows,
        "request_ids_checked": len(all_seen),
        "final_objective_cost": prior_cost,
        "final_slo_violations": prior_violations,
    }


def audit(
    arena_csv: Path,
    replay_csv: Path,
    *,
    discount_factor: float,
    sample_window_sec: float,
    tolerance: float,
) -> dict[str, object]:
    arena_rows = [
        row for row in _read(arena_csv)
        if row.get("phase") == "arena_step"
    ]
    replay_rows = _read(replay_csv)
    if not arena_rows or not replay_rows:
        raise AssertionError("arena and replay CSVs must both contain rows")

    previous_cost = 0.0
    rewards: list[float] = []
    discounts: list[float] = []
    max_discount_error = 0.0
    for row in arena_rows:
        total_cost = _float(row, "total_cost")
        chosen_reward = row.get("chosen_reward", "")
        reward = (
            float(chosen_reward)
            if chosen_reward not in ("", None)
            else -(total_cost - previous_cost)
        )
        previous_cost = total_cost
        dt = max(
            0.0,
            _float(row, "sim_time_after") - _float(row, "sim_time_before"),
        )
        expected_discount = discount_factor ** (dt / DISCOUNT_TIME_DENOM_SEC)
        chosen_discount = row.get("chosen_discount", "")
        discount = (
            float(chosen_discount)
            if chosen_discount not in ("", None)
            else expected_discount
        )
        max_discount_error = max(
            max_discount_error, abs(discount - expected_discount)
        )
        rewards.append(reward)
        discounts.append(discount)

    terminal_value = _float(replay_rows[0], "terminal_bootstrap_value")
    targets = discounted_trajectory_targets(rewards, discounts, terminal_value)
    by_turn = {int(float(row["turn"])): index for index, row in enumerate(arena_rows)}
    max_target_error = 0.0
    for row in replay_rows:
        turn = int(float(row["turn_number"]))
        max_target_error = max(
            max_target_error,
            abs(_float(row, "target_value") - targets[by_turn[turn]]),
        )

    start_time = _float(replay_rows[0], "trajectory_start_time")
    terminal_time = _float(replay_rows[0], "trajectory_terminal_time")
    max_sample_offset = max(
        _float(row, "time_at_state") - start_time for row in replay_rows
    )
    post_window_transitions = sum(
        _float(row, "sim_time_before") > start_time + sample_window_sec + 1e-12
        for row in arena_rows
    )
    truncated_index = max(
        index
        for index, row in enumerate(arena_rows)
        if _float(row, "sim_time_before") <= start_time + sample_window_sec + 1e-12
    )
    truncated_targets = discounted_trajectory_targets(
        rewards[: truncated_index + 1],
        discounts[: truncated_index + 1],
        terminal_value,
    )
    post_window_target_effect = abs(targets[0] - truncated_targets[0])

    search_rows = [
        row for row in arena_rows if int(float(row.get("iterations_used") or 0)) > 0
    ]
    controller_searches = sum(
        row.get("player_acted") == "controller" for row in search_rows
    )
    adversary_searches = sum(
        row.get("player_acted") == "adversary" for row in search_rows
    )
    result: dict[str, object] = {
        "arena_rows": len(arena_rows),
        "replay_rows": len(replay_rows),
        "searches": len(search_rows),
        "controller_searches": controller_searches,
        "adversary_searches": adversary_searches,
        "trajectory_start_time": start_time,
        "trajectory_terminal_time": terminal_time,
        "trajectory_duration_sec": terminal_time - start_time,
        "max_replay_sample_offset_sec": max_sample_offset,
        "post_window_transitions": post_window_transitions,
        "terminal_bootstrap_value": terminal_value,
        "first_full_target": targets[0],
        "first_truncated_target": truncated_targets[0],
        "post_window_target_effect": post_window_target_effect,
        "max_discount_error": max_discount_error,
        "max_target_error": max_target_error,
        **_semantic_audit(arena_rows, tolerance),
    }
    if terminal_time + tolerance < start_time + 12.0:
        raise AssertionError(result)
    if max_sample_offset > sample_window_sec + tolerance:
        raise AssertionError(result)
    if post_window_transitions <= 0 or post_window_target_effect <= tolerance:
        raise AssertionError(result)
    if max_discount_error > tolerance or max_target_error > tolerance:
        raise AssertionError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena-csv", type=Path, required=True)
    parser.add_argument("--replay-csv", type=Path, required=True)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--sample-window-sec", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    result = audit(
        args.arena_csv,
        args.replay_csv,
        discount_factor=args.discount_factor,
        sample_window_sec=args.sample_window_sec,
        tolerance=args.tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
