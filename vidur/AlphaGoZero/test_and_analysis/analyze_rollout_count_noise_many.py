#!/usr/bin/env python3
"""Aggregate direct-action policy-rollout noise across multiple roots."""

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
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 50, 100, 200])
    parser.add_argument("--expected-states", type=int)
    return parser.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    counts = sorted(set(args.counts))
    if not counts or counts[0] < 2:
        raise ValueError("counts must contain values >= 2")
    max_count = counts[-1]

    groups: dict[tuple[int, int, int, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "root_id": -1,
            "pending_prefill_tokens": -1,
            "pending_prefill_requests": -1,
            "search_iteration": math.inf,
            "trajectories": {},
        }
    )
    for trace_index, trace_path in enumerate(args.trace):
        with trace_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    trace_index,
                    int(row["sample_id"]),
                    int(row["root_action_index"]),
                    row["rollout_category"],
                    int(row["leaf_evaluation_id"]),
                )
                group = groups[key]
                group["root_id"] = int(row["root_id"])
                group["pending_prefill_tokens"] = int(row["pending_prefill_tokens"])
                group["pending_prefill_requests"] = int(row["pending_prefill_requests"])
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
                values = (
                    float(row["trajectory_reward_return_from_root"]),
                    float(row["trajectory_bootstrap_return_from_root"]),
                    float(row["trajectory_total_return_from_root"]),
                )
                expected = (
                    trajectory["reward_return"],
                    trajectory["bootstrap_return"],
                    trajectory["total_return"],
                )
                if any(
                    not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-12)
                    for actual, wanted in zip(values, expected)
                ):
                    raise AssertionError(f"inconsistent trajectory values for {key}")
                if row["step_phase"] == "tree_path":
                    trajectory["tree_path_steps"] += 1
    direct_actions: dict[tuple[int, int, str], dict[str, Any]] = {}
    for (
        _trace_index,
        sample_id,
        action_index,
        category,
        leaf_id,
    ), group in groups.items():
        trajectories = group["trajectories"]
        if len(trajectories) < max_count:
            continue
        if any(
            trajectory["tree_path_steps"] != 1
            for trajectory in trajectories.values()
        ):
            continue
        action_key = (sample_id, action_index, category)
        previous = direct_actions.get(action_key)
        if previous is not None and (
            previous["search_iteration"], previous["leaf_evaluation_id"]
        ) <= (group["search_iteration"], leaf_id):
            continue

        rollout_ids = sorted(trajectories)
        expected_ids = list(range(max_count))
        if rollout_ids[:max_count] != expected_ids:
            raise AssertionError(
                f"rollout IDs for {action_key} are not 0..{max_count - 1}"
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
        direct_actions[action_key] = {
            **{
                key: value
                for key, value in group.items()
                if key != "trajectories"
            },
            "leaf_evaluation_id": leaf_id,
            "trajectories": ordered,
        }

    actions_by_state: dict[int, list[tuple[int, str, dict[str, Any]]]] = defaultdict(
        list
    )
    for (sample_id, action_index, category), action in direct_actions.items():
        actions_by_state[sample_id].append((action_index, category, action))
    if args.expected_states is not None and len(actions_by_state) != args.expected_states:
        raise AssertionError(
            f"expected {args.expected_states} states, found {len(actions_by_state)}"
        )

    action_rows: list[dict[str, Any]] = []
    summaries: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for sample_id, actions in sorted(actions_by_state.items()):
        for action_index, category, action in sorted(actions):
            for rollout_count in counts:
                prefix = action["trajectories"][:rollout_count]
                returns = [trajectory["total_return"] for trajectory in prefix]
                rewards = [trajectory["reward_return"] for trajectory in prefix]
                bootstraps = [
                    trajectory["bootstrap_return"] for trajectory in prefix
                ]
                mean_return, std_return = _mean_std(returns)
                standard_error = std_return / math.sqrt(rollout_count)
                row = {
                    "sample_id": sample_id,
                    "root_id": action["root_id"],
                    "pending_prefill_tokens": action["pending_prefill_tokens"],
                    "pending_prefill_requests": action["pending_prefill_requests"],
                    "root_action_index": action_index,
                    "action_category": category,
                    "leaf_evaluation_id": action["leaf_evaluation_id"],
                    "search_iteration": action["search_iteration"],
                    "rollout_count": rollout_count,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "standard_error_return": standard_error,
                    "normal_95_ci_low_return": mean_return - 1.96 * standard_error,
                    "normal_95_ci_high_return": mean_return + 1.96 * standard_error,
                    "mean_cost": -mean_return,
                    "mean_reward_return": statistics.fmean(rewards),
                    "mean_bootstrap_return": statistics.fmean(bootstraps),
                }
                action_rows.append(row)
                summaries[
                    (sample_id, action_index, category, rollout_count)
                ] = row

    state_rows: list[dict[str, Any]] = []
    for sample_id, actions in sorted(actions_by_state.items()):
        decode_actions = [
            (action_index, category, action)
            for action_index, category, action in actions
            if category == "decode_only"
        ]
        prefill_actions = [
            (action_index, category, action)
            for action_index, category, action in actions
            if category.startswith("prefill_")
        ]
        for rollout_count in counts:
            all_candidates = [
                summaries[(sample_id, action_index, category, rollout_count)]
                for action_index, category, _ in actions
            ]
            overall_best = max(all_candidates, key=lambda row: row["mean_return"])
            base = {
                "sample_id": sample_id,
                "root_id": all_candidates[0]["root_id"],
                "pending_prefill_tokens": all_candidates[0][
                    "pending_prefill_tokens"
                ],
                "pending_prefill_requests": all_candidates[0][
                    "pending_prefill_requests"
                ],
                "direct_action_count": len(actions),
                "rollout_count": rollout_count,
                "overall_best_action_index": overall_best["root_action_index"],
                "overall_best_category": overall_best["action_category"],
                "comparison_status": "comparable",
            }
            if not decode_actions or not prefill_actions:
                state_rows.append(
                    {
                        **base,
                        "comparison_status": (
                            "missing_decode"
                            if not decode_actions
                            else "missing_prefill"
                        ),
                        "decode_action_index": "",
                        "decode_mean_return": "",
                        "decode_mean_cost": "",
                        "decode_standard_error": "",
                        "best_prefill_action_index": "",
                        "best_prefill_category": "",
                        "best_prefill_mean_return": "",
                        "best_prefill_mean_cost": "",
                        "best_prefill_standard_error": "",
                        "prefill_minus_decode_mean_return_gap": "",
                        "gap_standard_error": "",
                        "normal_95_ci_low_gap": "",
                        "normal_95_ci_high_gap": "",
                        "preferred_action_by_mean": "",
                        "significance": "",
                    }
                )
                continue

            decode = max(
                (
                    summaries[(sample_id, action_index, category, rollout_count)]
                    for action_index, category, _ in decode_actions
                ),
                key=lambda row: row["mean_return"],
            )
            prefill = max(
                (
                    summaries[(sample_id, action_index, category, rollout_count)]
                    for action_index, category, _ in prefill_actions
                ),
                key=lambda row: row["mean_return"],
            )
            gap = prefill["mean_return"] - decode["mean_return"]
            gap_standard_error = math.sqrt(
                prefill["standard_error_return"] ** 2
                + decode["standard_error_return"] ** 2
            )
            ci_low = gap - 1.96 * gap_standard_error
            ci_high = gap + 1.96 * gap_standard_error
            state_rows.append(
                {
                    **base,
                    "decode_action_index": decode["root_action_index"],
                    "decode_mean_return": decode["mean_return"],
                    "decode_mean_cost": decode["mean_cost"],
                    "decode_standard_error": decode["standard_error_return"],
                    "best_prefill_action_index": prefill["root_action_index"],
                    "best_prefill_category": prefill["action_category"],
                    "best_prefill_mean_return": prefill["mean_return"],
                    "best_prefill_mean_cost": prefill["mean_cost"],
                    "best_prefill_standard_error": prefill[
                        "standard_error_return"
                    ],
                    "prefill_minus_decode_mean_return_gap": gap,
                    "gap_standard_error": gap_standard_error,
                    "normal_95_ci_low_gap": ci_low,
                    "normal_95_ci_high_gap": ci_high,
                    "preferred_action_by_mean": (
                        prefill["action_category"] if gap > 0.0 else "decode_only"
                    ),
                    "significance": (
                        "prefill_significant"
                        if ci_low > 0.0
                        else "decode_significant"
                        if ci_high < 0.0
                        else "uncertain"
                    ),
                }
            )

    reference_preference = {
        row["sample_id"]: row["preferred_action_by_mean"]
        for row in state_rows
        if row["rollout_count"] == max_count
        and row["comparison_status"] == "comparable"
    }
    aggregate_rows: list[dict[str, Any]] = []
    for rollout_count in counts:
        rows = [
            row for row in state_rows if row["rollout_count"] == rollout_count
        ]
        comparable = [
            row for row in rows if row["comparison_status"] == "comparable"
        ]
        gaps = [
            float(row["prefill_minus_decode_mean_return_gap"])
            for row in comparable
        ]
        pref_better = [
            row for row in comparable if row["preferred_action_by_mean"] != "decode_only"
        ]
        decode_better = [
            row for row in comparable if row["preferred_action_by_mean"] == "decode_only"
        ]
        aggregate_rows.append(
            {
                "rollout_count": rollout_count,
                "total_states": len(rows),
                "comparable_states": len(comparable),
                "missing_decode_states": sum(
                    row["comparison_status"] == "missing_decode" for row in rows
                ),
                "prefill_better_states": len(pref_better),
                "prefill_better_fraction": len(pref_better) / len(comparable),
                "decode_better_states": len(decode_better),
                "decode_better_fraction": len(decode_better) / len(comparable),
                "prefill_significant_states": sum(
                    row["significance"] == "prefill_significant"
                    for row in comparable
                ),
                "decode_significant_states": sum(
                    row["significance"] == "decode_significant"
                    for row in comparable
                ),
                "uncertain_states": sum(
                    row["significance"] == "uncertain" for row in comparable
                ),
                "mean_prefill_minus_decode_return_gap": statistics.fmean(gaps),
                "median_prefill_minus_decode_return_gap": statistics.median(gaps),
                "mean_gap_standard_error": statistics.fmean(
                    float(row["gap_standard_error"]) for row in comparable
                ),
                "preference_matches_200_count": sum(
                    row["preferred_action_by_mean"]
                    == reference_preference[row["sample_id"]]
                    for row in comparable
                ),
                "preference_matches_200_fraction": sum(
                    row["preferred_action_by_mean"]
                    == reference_preference[row["sample_id"]]
                    for row in comparable
                )
                / len(comparable),
                "overall_best_is_decode_states": sum(
                    row["overall_best_category"] == "decode_only" for row in rows
                ),
                "overall_best_is_prefill_states": sum(
                    str(row["overall_best_category"]).startswith("prefill_")
                    for row in rows
                ),
            }
        )

    noise_rows: list[dict[str, Any]] = []
    first_count = counts[0]
    for sample_id, action_index, category in sorted(direct_actions):
        first = summaries[(sample_id, action_index, category, first_count)]
        last = summaries[(sample_id, action_index, category, max_count)]
        noise_rows.append(
            {
                "sample_id": sample_id,
                "root_action_index": action_index,
                "action_category": category,
                "standard_error_at_first_count": first[
                    "standard_error_return"
                ],
                "standard_error_at_last_count": last["standard_error_return"],
                "observed_standard_error_reduction_factor": (
                    first["standard_error_return"]
                    / last["standard_error_return"]
                    if last["standard_error_return"] > 0.0
                    else math.inf
                ),
            }
        )

    reduction_factors = [
        row["observed_standard_error_reduction_factor"]
        for row in noise_rows
        if math.isfinite(row["observed_standard_error_reduction_factor"])
    ]
    report = {
        "traces": [str(path) for path in args.trace],
        "counts": counts,
        "states": len(actions_by_state),
        "direct_canonical_actions": len(direct_actions),
        "theoretical_independent_sample_reduction_factor": math.sqrt(
            max_count / first_count
        ),
        "observed_standard_error_reduction_factor_mean": statistics.fmean(
            reduction_factors
        ),
        "observed_standard_error_reduction_factor_median": statistics.median(
            reduction_factors
        ),
        "aggregate_rows": aggregate_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "rollout_count_noise_by_action.csv", action_rows)
    _write_csv(
        args.output_dir / "rollout_count_decode_vs_best_prefill_by_state.csv",
        state_rows,
    )
    _write_csv(args.output_dir / "rollout_count_aggregate.csv", aggregate_rows)
    _write_csv(
        args.output_dir / "rollout_count_noise_reduction_by_action.csv",
        noise_rows,
    )
    (args.output_dir / "rollout_count_noise_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
