from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _load_prefill_times(path: Path, value_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"prefill_tokens", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[["prefill_tokens", value_column]].copy()


def _scale_profile(source: Path, destination: Path, factor: float) -> list[str]:
    frame = pd.read_csv(source, low_memory=False)
    timing_columns = [column for column in frame.columns if column.startswith("time_stats.")]
    if not timing_columns:
        raise ValueError(f"no time_stats columns found in {source}")
    frame.loc[:, timing_columns] = frame.loc[:, timing_columns].astype(float) * factor
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return timing_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply one auditable vLLM/Sarathi backend factor to GPU operation profiles"
    )
    parser.add_argument("--raw-compute-root", type=Path, required=True)
    parser.add_argument("--output-compute-root", type=Path, required=True)
    parser.add_argument("--raw-prefill-profile", type=Path, required=True)
    parser.add_argument("--real-profile", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    args = parser.parse_args()

    raw = _load_prefill_times(args.raw_prefill_profile, "prefill_time_seconds").rename(
        columns={"prefill_time_seconds": "raw_simulator_s"}
    )
    real = _load_prefill_times(args.real_profile, "observed_median_s").rename(
        columns={"observed_median_s": "real_vllm_s"}
    )
    matched = real.merge(raw, on="prefill_tokens", how="inner", validate="one_to_one")
    if matched.empty:
        raise ValueError("real and raw simulator profiles have no common prefill sizes")
    if (matched[["real_vllm_s", "raw_simulator_s"]] <= 0).any().any():
        raise ValueError("calibration timings must be positive")

    ratios = matched["real_vllm_s"] / matched["raw_simulator_s"]
    factor = float(np.median(ratios.to_numpy(dtype=float)))
    if not 0.5 <= factor <= 1.5:
        raise ValueError(f"refusing implausible backend calibration factor: {factor}")

    scaled_columns: dict[str, list[str]] = {}
    for name in ("mlp.csv", "attention.csv"):
        scaled_columns[name] = _scale_profile(
            args.raw_compute_root / name,
            args.output_compute_root / name,
            factor,
        )

    metadata = {
        "method": "median(real_vllm_s / raw_simulator_s)",
        "scope": "single global multiplier applied to every time_stats column",
        "factor": factor,
        "calibration_points": [
            {
                "prefill_tokens": int(row.prefill_tokens),
                "real_vllm_s": float(row.real_vllm_s),
                "raw_simulator_s": float(row.raw_simulator_s),
                "ratio": float(row.real_vllm_s / row.raw_simulator_s),
            }
            for row in matched.itertuples(index=False)
        ],
        "scaled_columns": scaled_columns,
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Keep the calibration evidence beside the transformed profile.
    shutil.copy2(args.output_metadata, args.output_compute_root / "calibration.json")
    print(f"backend calibration factor={factor:.12f}")


if __name__ == "__main__":
    main()
