from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import math
import sys
from pathlib import Path


def _load_profile(path: Path) -> tuple[list[int], list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tokens = [int(row["prefill_tokens"]) for row in rows]
    times = [float(row["prefill_time_seconds"]) for row in rows]
    return tokens, times


def _load_native_module(repo_root: Path):
    candidates = [
        repo_root / ".venv" / "lib" / "python3.10" / "site-packages" / "numpy.libs",
        repo_root.parent / "vidur" / ".venv" / "lib" / "python3.10" / "site-packages" / "numpy.libs",
    ]
    for root in candidates:
        quadmath = glob.glob(str(root / "libquadmath*.so.0.0.0"))
        gfortran = glob.glob(str(root / "libgfortran*.so.5.0.0"))
        if quadmath and gfortran:
            ctypes.CDLL(quadmath[0], mode=ctypes.RTLD_GLOBAL)
            ctypes.CDLL(gfortran[0], mode=ctypes.RTLD_GLOBAL)
            break
    from vidur.Game_Version3_Cpp import mcts_native_gv2

    return mcts_native_gv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the deployable mew1 A100 simulator artifacts")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()

    profile_root = Path(__file__).resolve().parent
    repo_root = profile_root.parents[2]
    artifacts = profile_root / "artifacts"
    tokens, times = _load_profile(artifacts / "prefill_profile.csv")

    assert tokens == list(range(128, 4097, 128)), "unexpected prefill token grid"
    assert all(value > 0.0 and math.isfinite(value) for value in times), "invalid prefill time"
    assert all(left < right for left, right in zip(times, times[1:])), "prefill profile is not monotonic"

    validation = json.loads(
        (artifacts / "validation" / "prefill_comparison.json").read_text(encoding="utf-8")
    )
    assert int(validation["rows"]) == 7
    assert float(validation["max_absolute_percent_error"]) <= 10.0

    calibration = json.loads((artifacts / "calibration.json").read_text(encoding="utf-8"))
    factor = float(calibration["factor"])
    assert 0.5 <= factor <= 1.5
    assert len(calibration["calibration_points"]) == 7

    cache_files = [path for path in (artifacts / "cache").iterdir() if path.is_file()]
    assert cache_files, "deployable predictor cache is empty"

    native_queries = 0
    native_max_abs_diff = 0.0
    if not args.skip_native:
        native = _load_native_module(repo_root)
        queries = sorted(
            set(tokens + [1, 127, 129, 255, 257, 511, 513, 768, 1536, 3072, 4096])
        )
        native_values = [
            float(value)
            for value in native.debug_prefill_profile_lookup(tokens, times, queries)
        ]
        python_values = [
            times[min(range(len(tokens)), key=lambda index: abs(tokens[index] - query))]
            for query in queries
        ]
        differences = [
            abs(python_value - native_value)
            for python_value, native_value in zip(python_values, native_values)
        ]
        native_queries = len(queries)
        native_max_abs_diff = max(differences, default=0.0)
        assert native_max_abs_diff <= 1e-12

    print(
        json.dumps(
            {
                "passed": True,
                "profile_rows": len(tokens),
                "calibration_factor": factor,
                "mean_absolute_percent_error": validation["mean_absolute_percent_error"],
                "max_absolute_percent_error": validation["max_absolute_percent_error"],
                "cache_file_count": len(cache_files),
                "native_queries": native_queries,
                "native_max_abs_diff": native_max_abs_diff,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
