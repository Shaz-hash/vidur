from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare simulator and real-vLLM prefill times")
    parser.add_argument("--simulator-profile", type=Path, required=True)
    parser.add_argument("--real-profile", type=Path, required=True)
    parser.add_argument("--legacy-profile", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    simulator = pd.read_csv(args.simulator_profile).rename(
        columns={"prefill_time_seconds": "new_simulator_s"}
    )
    real = pd.read_csv(args.real_profile)[
        ["prefill_tokens", "observed_median_s"]
    ].rename(columns={"observed_median_s": "real_vllm_s"})
    comparison = real.merge(simulator, on="prefill_tokens", how="left", validate="one_to_one")

    if args.legacy_profile and args.legacy_profile.exists():
        legacy = pd.read_csv(args.legacy_profile).rename(
            columns={"prefill_time_seconds": "legacy_simulator_s"}
        )
        comparison = comparison.merge(legacy, on="prefill_tokens", how="left")

    if comparison["new_simulator_s"].isna().any():
        missing = comparison.loc[comparison["new_simulator_s"].isna(), "prefill_tokens"].tolist()
        raise ValueError(f"new simulator profile is missing token sizes: {missing}")

    comparison["new_error_s"] = comparison["new_simulator_s"] - comparison["real_vllm_s"]
    comparison["new_abs_percent_error"] = (
        comparison["new_error_s"].abs() / comparison["real_vllm_s"] * 100.0
    )
    comparison["real_vs_new_difference_percent"] = (
        (comparison["real_vllm_s"] - comparison["new_simulator_s"])
        / comparison["new_simulator_s"]
        * 100.0
    )

    if "legacy_simulator_s" in comparison:
        comparison["legacy_abs_percent_error"] = (
            (comparison["legacy_simulator_s"] - comparison["real_vllm_s"]).abs()
            / comparison["real_vllm_s"]
            * 100.0
        )
        comparison["abs_percent_error_improvement"] = (
            comparison["legacy_abs_percent_error"]
            - comparison["new_abs_percent_error"]
        )

    errors = comparison["new_abs_percent_error"].to_numpy(dtype=float)
    summary = {
        "rows": int(len(comparison)),
        "mean_absolute_percent_error": float(np.mean(errors)),
        "median_absolute_percent_error": float(np.median(errors)),
        "p95_absolute_percent_error": float(np.percentile(errors, 95)),
        "max_absolute_percent_error": float(np.max(errors)),
        "within_5_percent": int(np.sum(errors <= 5.0)),
        "within_10_percent": int(np.sum(errors <= 10.0)),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
