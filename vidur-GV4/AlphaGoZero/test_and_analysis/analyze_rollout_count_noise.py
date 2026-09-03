#!/usr/bin/env python3
"""Measure policy-rollout Monte Carlo noise for direct root actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 50, 100, 200])
    return parser.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def _summary(
    *,
    sample_id: int,
    root_id: int,
    action_index: int,
    action_category: str,
    leaf_evaluation_id: int,
    rollout_count: int,
    trajectories: list[dict[str, float]],
) -> dict[str, Any]:
    prefix = trajectories[:rollout_count]
    returns = [row["total_return"] for row in prefix]
    rewards = [row["reward_return"] for row in prefix]
    bootstraps = [row["bootstrap_return"] for row in prefix]
    mean_return, std_return = _mean_std(returns)
    standard_error = std_return / math.sqrt(rollout_count)
    return {
        "sample_id": sample_id,
        "root_id": root_id,
        "root_action_index": action_index,
        "action_category": action_category,
        "leaf_evaluation_id": leaf_evaluation_id,
        "rollout_count": rollout_count,
        "mean_return": mean_return,
        "std_return": std_return,
        "standard_error_return": standard_error,
        "normal_95_ci_low_return": mean_return - 1.96 * standard_error,
        "normal_95_ci_high_return": mean_return + 1.96 * standard_error,
        "mean_cost": -mean_return,
        "std_cost": std_return,
        "standard_error_cost": standard_error,
        "mean_reward_return": statistics.fmean(rewards),
        "mean_bootstrap_return": statistics.fmean(bootstraps),
    }


def main() -> None:
    args = _parse_args()
    counts = sorted(set(args.counts))
    if not counts or counts[0] < 2:
        raise ValueError("counts must contain values >= 2")
    max_count = counts[-1]

    groups: dict[tuple[int, int, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "root_id": -1,
            "search_iteration": math.inf,
            "trajectories": {},
        }
    )
    with args.trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["sample_id"]),
                int(row["root_action_index"]),
                row["rollout_category"],
                int(row["leaf_evaluation_id"]),
            )
            group = groups[key]
            group["root_id"] = int(row["root_id"])
            group["search_iteration"] = min(
                group["search_iteration"], int(row["search_iteration"])
            )
            rollout_id = int(row["rollout_id"])
            trajectory = group["trajectories"].setdefault(
                rollout_id,
                {
                    "tree_path_steps": 0,
                    "reward_return": float(
                        row["trajectory_reward_return_from_root"]
                    ),
                    "bootstrap_return": float(
                        row["trajectory_bootstrap_return_from_root"]
                    ),
                    "total_return": float(row["trajectory_total_return_from_root"]),
                },
            )
            current_values = (
                float(row["trajectory_reward_return_from_root"]),
                float(row["trajectory_bootstrap_return_from_root"]),
                float(row["trajectory_total_return_from_root"]),
            )
            expected_values = (
                trajectory["reward_return"],
                trajectory["bootstrap_return"],
                trajectory["total_return"],
            )
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(current_values, expected_values)
            ):
                raise AssertionError(f"inconsistent trajectory values for {key}")
            if row["step_phase"] == "tree_path":
                trajectory["tree_path_steps"] += 1

    direct_by_action: dict[tuple[int, int, str], tuple[int, dict[str, Any]]] = {}
    for (sample_id, action_index, category, leaf_id), group in groups.items():
        trajectories = group["trajectories"]
        if len(trajectories) < max_count:
            continue
        if any(
            trajectory["tree_path_steps"] != 1
            for trajectory in trajectories.values()
        ):
            continue
        action_key = (sample_id, action_index, category)
        candidate = (leaf_id, group)
        previous = direct_by_action.get(action_key)
        if previous is None or (
            group["search_iteration"], leaf_id
        ) < (
            previous[1]["search_iteration"], previous[0]
        ):
            direct_by_action[action_key] = candidate

    if len(direct_by_action) < 2:
        raise AssertionError(
            f"expected at least two direct root-action batches, found "
            f"{len(direct_by_action)}"
        )

    action_rows: list[dict[str, Any]] = []
    direct_batches: dict[str, dict[str, Any]] = {}
    for (sample_id, action_index, category), (leaf_id, group) in sorted(
        direct_by_action.items()
    ):
        trajectories = group["trajectories"]
        sorted_ids = sorted(trajectories)
        expected_ids = list(range(max_count))
        if sorted_ids[:max_count] != expected_ids:
            raise AssertionError(
                f"rollout IDs for action {action_index} are not 0..{max_count - 1}"
            )
        ordered = [trajectories[rollout_id] for rollout_id in expected_ids]
        for trajectory in ordered:
            if not math.isclose(
                trajectory["reward_return"] + trajectory["bootstrap_return"],
                trajectory["total_return"],
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise AssertionError("reward + bootstrap does not equal total return")
            if not all(
                math.isfinite(trajectory[field])
                for field in ("reward_return", "bootstrap_return", "total_return")
            ):
                raise AssertionError("non-finite trajectory return")

        direct_batches[category] = {
            "sample_id": sample_id,
            "root_id": group["root_id"],
            "action_index": action_index,
            "leaf_evaluation_id": leaf_id,
            "search_iteration": group["search_iteration"],
            "trajectories": ordered,
        }
        for rollout_count in counts:
            action_rows.append(
                _summary(
                    sample_id=sample_id,
                    root_id=group["root_id"],
                    action_index=action_index,
                    action_category=category,
                    leaf_evaluation_id=leaf_id,
                    rollout_count=rollout_count,
                    trajectories=ordered,
                )
            )

    decode = direct_batches.get("decode_only")
    prefill_categories = sorted(
        category for category in direct_batches if category.startswith("prefill_")
    )
    if decode is None or not prefill_categories:
        raise AssertionError(
            "direct batches must include decode_only and at least one prefill action"
        )
    prefill_category = prefill_categories[0]
    prefill = direct_batches[prefill_category]

    gap_rows: list[dict[str, Any]] = []
    for rollout_count in counts:
        decode_returns = [
            row["total_return"] for row in decode["trajectories"][:rollout_count]
        ]
        prefill_returns = [
            row["total_return"] for row in prefill["trajectories"][:rollout_count]
        ]
        decode_mean, decode_std = _mean_std(decode_returns)
        prefill_mean, prefill_std = _mean_std(prefill_returns)
        gap = prefill_mean - decode_mean
        gap_standard_error = math.sqrt(
            decode_std * decode_std / rollout_count
            + prefill_std * prefill_std / rollout_count
        )
        gap_rows.append(
            {
                "rollout_count": rollout_count,
                "decode_mean_return": decode_mean,
                "decode_mean_cost": -decode_mean,
                f"{prefill_category}_mean_return": prefill_mean,
                f"{prefill_category}_mean_cost": -prefill_mean,
                "prefill_minus_decode_mean_return_gap": gap,
                "gap_standard_error": gap_standard_error,
                "normal_95_ci_low_gap": gap - 1.96 * gap_standard_error,
                "normal_95_ci_high_gap": gap + 1.96 * gap_standard_error,
                "preferred_action_by_mean": (
                    prefill_category if gap > 0.0 else "decode_only"
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_csv = args.output_dir / "rollout_count_noise_summary.csv"
    with action_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(action_rows[0]))
        writer.writeheader()
        writer.writerows(action_rows)
    gap_csv = args.output_dir / "rollout_count_action_gap.csv"
    with gap_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gap_rows[0]))
        writer.writeheader()
        writer.writerows(gap_rows)

    first_count = counts[0]
    last_count = counts[-1]
    noise_reduction: dict[str, Any] = {}
    for category in ("decode_only", prefill_category):
        first = next(
            row
            for row in action_rows
            if row["action_category"] == category
            and row["rollout_count"] == first_count
        )
        last = next(
            row
            for row in action_rows
            if row["action_category"] == category
            and row["rollout_count"] == last_count
        )
        noise_reduction[category] = {
            "standard_error_at_first_count": first["standard_error_return"],
            "standard_error_at_last_count": last["standard_error_return"],
            "observed_standard_error_reduction_factor": (
                first["standard_error_return"] / last["standard_error_return"]
            ),
        }

    report = {
        "trace": str(args.trace),
        "counts": counts,
        "direct_batches": {
            category: {
                key: value
                for key, value in batch.items()
                if key != "trajectories"
            }
            for category, batch in direct_batches.items()
        },
        "noise_reduction": noise_reduction,
        "theoretical_independent_sample_reduction_factor": math.sqrt(
            last_count / first_count
        ),
        "action_rows": action_rows,
        "gap_rows": gap_rows,
    }
    json_path = args.output_dir / "rollout_count_noise_summary.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
