"""Measure how time discounting changes pending-prefill learning signal.

The report has two complementary parts:

1. Recompute realized, no-bootstrap returns from identical arena trajectories
   under several discount factors.
2. Decompose logged root-child Q as reward + discount * continuation and
   change only the root-edge discount for same-state prefill/decode siblings.

The second calculation is a local sensitivity test, not a substitute for
retraining the value model under the alternative discount.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_DISCOUNT_DENOM_SEC = 0.015725797204323228


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _structured(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(value)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return default


def _has_pending_prefill(row: dict[str, Any]) -> bool:
    remaining = _structured(row.get("prefill_remaining_by_id"), {})
    return isinstance(remaining, dict) and any(
        _finite(value) > 0.0 for value in remaining.values()
    )


def _action_has_prefill(action: Any) -> bool:
    text = str(action)
    return "prefill_allocations={" in text and "prefill_allocations={}" not in text


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def _rate(count: int, total: int) -> float:
    return 100.0 * float(count) / float(total) if total else math.nan


def _discount(gamma: float, exponent: float) -> float:
    return float(float(gamma) ** max(0.0, float(exponent)))


def _edge_exponent(discount: float, source_gamma: float) -> float | None:
    value = float(discount)
    if not (0.0 < value <= 1.0):
        return None
    if abs(value - 1.0) <= 1e-15:
        return 0.0
    denominator = math.log(float(source_gamma))
    if abs(denominator) <= 1e-15:
        return None
    exponent = math.log(value) / denominator
    return max(0.0, exponent) if math.isfinite(exponent) else None


def _arena_files(block: Path) -> list[Path]:
    arena = block / "arena_games" if (block / "arena_games").is_dir() else block
    return sorted(arena.glob("*model_ctrl_depth1.csv"))


def _read_transitions(path: Path, discount_denom_sec: float) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    previous_total_cost = 0.0
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("phase", "")) != "arena_step":
                continue
            total_cost = _finite(raw.get("total_cost"), previous_total_cost)
            cost_delta = total_cost - previous_total_cost
            previous_total_cost = total_cost
            chosen_reward = raw.get("chosen_reward")
            reward = (
                _finite(chosen_reward)
                if chosen_reward is not None and str(chosen_reward).strip()
                else -float(cost_delta)
            )
            time_before = _finite(raw.get("sim_time_before"))
            time_after = _finite(raw.get("sim_time_after"), time_before)
            transitions.append(
                {
                    "row": raw,
                    "reward": float(reward),
                    "exponent": max(0.0, time_after - time_before)
                    / max(1e-12, float(discount_denom_sec)),
                }
            )
    return transitions


def _realized_returns(
    transitions: list[dict[str, Any]], gammas: tuple[float, ...]
) -> dict[float, list[float]]:
    results = {gamma: [0.0] * len(transitions) for gamma in gammas}
    for gamma in gammas:
        running = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            transition = transitions[index]
            running = float(
                transition["reward"]
                + _discount(gamma, transition["exponent"]) * running
            )
            results[gamma][index] = running
    return results


def analyze(
    block: Path,
    *,
    source_gamma: float,
    gammas: tuple[float, ...],
    discount_denom_sec: float,
    value_rmse: float | None,
) -> dict[str, Any]:
    files = _arena_files(block)
    pending_states = 0
    chosen_prefill = 0
    chosen_decode = 0
    decode_immediate_zero = 0
    realized_decode: dict[float, list[float]] = {gamma: [] for gamma in gammas}
    realized_decode_future: dict[float, list[float]] = {gamma: [] for gamma in gammas}

    sibling_states = 0
    immediate_prefill = 0
    immediate_decode = 0
    immediate_tie = 0
    continuation_prefill = 0
    source_q_decode = 0
    q_gaps: dict[float, list[float]] = {gamma: [] for gamma in gammas}
    source_decode_flips: dict[float, int] = {gamma: 0 for gamma in gammas}
    source_prefill_flips: dict[float, int] = {gamma: 0 for gamma in gammas}
    reward_gaps: list[float] = []
    continuation_gaps: list[float] = []

    for path in files:
        transitions = _read_transitions(path, discount_denom_sec)
        returns = _realized_returns(transitions, gammas)
        for index, transition in enumerate(transitions):
            row = transition["row"]
            if row.get("player_acted") != "controller" or not _has_pending_prefill(row):
                continue
            pending_states += 1
            selected_prefill = _action_has_prefill(row.get("action_repr"))
            chosen_prefill += int(selected_prefill)
            chosen_decode += int(not selected_prefill)
            if not selected_prefill:
                decode_immediate_zero += int(abs(transition["reward"]) <= 1e-12)
                for gamma in gammas:
                    value = returns[gamma][index]
                    realized_decode[gamma].append(value)
                    realized_decode_future[gamma].append(value - transition["reward"])

            actions = list(_structured(row.get("candidate_top5_action_reprs"), []))
            q_values = list(_structured(row.get("candidate_top5_q_values"), []))
            rewards = list(_structured(row.get("candidate_top5_rewards"), []))
            discounts = list(_structured(row.get("candidate_top5_discounts"), []))
            count = min(len(actions), len(q_values), len(rewards), len(discounts))
            if count <= 0:
                continue
            actions = actions[:count]
            q_values = [_finite(value, math.nan) for value in q_values[:count]]
            rewards = [_finite(value, math.nan) for value in rewards[:count]]
            discounts = [_finite(value, math.nan) for value in discounts[:count]]
            valid = [
                item
                for item in range(count)
                if all(math.isfinite(values[item]) for values in (q_values, rewards, discounts))
                and discounts[item] > 0.0
            ]
            prefill = [item for item in valid if _action_has_prefill(actions[item])]
            decode = [item for item in valid if item not in prefill]
            if not prefill or not decode:
                continue

            prefill_index = max(prefill, key=q_values.__getitem__)
            decode_index = max(decode, key=q_values.__getitem__)
            exponents = {
                item: _edge_exponent(discounts[item], source_gamma)
                for item in (prefill_index, decode_index)
            }
            if any(exponents[item] is None for item in exponents):
                continue
            continuations = {
                item: (q_values[item] - rewards[item]) / discounts[item]
                for item in (prefill_index, decode_index)
            }

            sibling_states += 1
            reward_gap = rewards[prefill_index] - rewards[decode_index]
            continuation_gap = continuations[prefill_index] - continuations[decode_index]
            reward_gaps.append(reward_gap)
            continuation_gaps.append(continuation_gap)
            immediate_prefill += int(reward_gap > 1e-12)
            immediate_decode += int(reward_gap < -1e-12)
            immediate_tie += int(abs(reward_gap) <= 1e-12)
            continuation_prefill += int(continuation_gap > 0.0)
            source_gap = q_values[prefill_index] - q_values[decode_index]
            source_was_decode = source_gap <= 0.0
            source_q_decode += int(source_was_decode)

            for gamma in gammas:
                q_prefill = rewards[prefill_index] + _discount(
                    gamma, float(exponents[prefill_index])
                ) * continuations[prefill_index]
                q_decode = rewards[decode_index] + _discount(
                    gamma, float(exponents[decode_index])
                ) * continuations[decode_index]
                gap = q_prefill - q_decode
                q_gaps[gamma].append(gap)
                source_decode_flips[gamma] += int(source_was_decode and gap > 0.0)
                source_prefill_flips[gamma] += int(not source_was_decode and gap <= 0.0)

    output: dict[str, Any] = {
        "arena_block": str(block),
        "game_files": len(files),
        "source_gamma": source_gamma,
        "discount_denom_sec": discount_denom_sec,
        "pending_controller_states": pending_states,
        "chosen_prefill_pct": _rate(chosen_prefill, pending_states),
        "chosen_decode_states": chosen_decode,
        "decode_immediate_reward_zero_pct": _rate(
            decode_immediate_zero, chosen_decode
        ),
        "same_state_sibling_rows": sibling_states,
        "immediate_reward_favors_prefill_pct": _rate(
            immediate_prefill, sibling_states
        ),
        "immediate_reward_favors_decode_pct": _rate(immediate_decode, sibling_states),
        "immediate_reward_tie_pct": _rate(immediate_tie, sibling_states),
        "implied_continuation_favors_prefill_pct": _rate(
            continuation_prefill, sibling_states
        ),
        "reward_gap_prefill_minus_decode_mean": _mean(reward_gaps),
        "reward_gap_prefill_minus_decode_median": _median(reward_gaps),
        "continuation_gap_prefill_minus_decode_mean": _mean(continuation_gaps),
        "continuation_gap_prefill_minus_decode_median": _median(continuation_gaps),
        "source_q_favors_decode_rows": source_q_decode,
        "source_q_favors_prefill_rows": sibling_states - source_q_decode,
    }
    for gamma in gammas:
        label = str(gamma).replace(".", "_")
        decode_returns = realized_decode[gamma]
        future = realized_decode_future[gamma]
        gaps = q_gaps[gamma]
        output[f"gamma_{label}_decode_realized_return_mean"] = _mean(decode_returns)
        output[f"gamma_{label}_decode_realized_return_median"] = _median(decode_returns)
        output[f"gamma_{label}_decode_realized_return_abs_median"] = _median(
            [abs(value) for value in decode_returns]
        )
        output[f"gamma_{label}_decode_realized_return_abs_p25"] = _percentile(
            [abs(value) for value in decode_returns], 0.25
        )
        output[f"gamma_{label}_decode_realized_return_abs_p75"] = _percentile(
            [abs(value) for value in decode_returns], 0.75
        )
        output[f"gamma_{label}_decode_future_component_abs_median"] = _median(
            [abs(value) for value in future]
        )
        if value_rmse is not None:
            output[f"gamma_{label}_decode_abs_return_below_value_rmse_pct"] = _rate(
                sum(abs(value) < float(value_rmse) for value in decode_returns),
                len(decode_returns),
            )
        output[f"gamma_{label}_edge_only_q_favors_prefill_pct"] = _rate(
            sum(value > 0.0 for value in gaps), len(gaps)
        )
        output[f"gamma_{label}_edge_only_q_gap_mean"] = _mean(gaps)
        output[f"gamma_{label}_edge_only_q_gap_median"] = _median(gaps)
        output[f"gamma_{label}_source_decode_flip_to_prefill_pct"] = _rate(
            source_decode_flips[gamma], source_q_decode
        )
        output[f"gamma_{label}_source_prefill_flip_to_decode_pct"] = _rate(
            source_prefill_flips[gamma], sibling_states - source_q_decode
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify pending-prefill signal attenuation under time discounting."
    )
    parser.add_argument("arena_block", type=Path)
    parser.add_argument("--source-gamma", type=float, default=0.90)
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.90, 0.98, 0.995]
    )
    parser.add_argument(
        "--discount-denom-sec", type=float, default=DEFAULT_DISCOUNT_DENOM_SEC
    )
    parser.add_argument("--value-rmse", type=float)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    gammas = tuple(dict.fromkeys(float(value) for value in args.gammas))
    if any(not (0.0 < value <= 1.0) for value in gammas):
        raise ValueError("all discount factors must be in (0, 1]")
    if not (0.0 < float(args.source_gamma) < 1.0):
        raise ValueError("source gamma must be in (0, 1)")
    result = analyze(
        args.arena_block.expanduser().resolve(),
        source_gamma=float(args.source_gamma),
        gammas=gammas,
        discount_denom_sec=float(args.discount_denom_sec),
        value_rmse=args.value_rmse,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    print(payload, end="")
    if args.json:
        path = args.json.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
