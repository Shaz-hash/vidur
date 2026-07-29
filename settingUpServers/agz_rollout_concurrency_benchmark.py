#!/usr/bin/env python3
"""Benchmark native rollout MCTS under the worker's real process concurrency."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def run(args: argparse.Namespace) -> dict[str, object]:
    model_root = args.model_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "controller_value": model_root / "controller_value/dnn_value_markov_deepset_192_v2/model.joblib",
        "adversary_value": model_root / "adversary_value/dnn_value_markov_deepset_192_v2/model.joblib",
        "controller_policy": model_root / "controller_prior/dnn_policy_rank_192_v1/model.joblib",
        "adversary_policy": model_root / "adversary_prior/dnn_policy_rank_192_v1/model.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    processes: list[tuple[subprocess.Popen[str], object, Path]] = []
    started = time.perf_counter()
    for index in range(int(args.processes)):
        process_dir = output_dir / f"process_{index:03d}"
        process_dir.mkdir(parents=True, exist_ok=True)
        log = (process_dir / "run.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "vidur.AlphaGoZero.test_and_analysis.test_dnn_mcts_parity",
            "--controller-value-model", str(paths["controller_value"]),
            "--adversary-value-model", str(paths["adversary_value"]),
            "--controller-policy-model", str(paths["controller_policy"]),
            "--adversary-policy-model", str(paths["adversary_policy"]),
            "--iterations", str(args.iterations),
            "--rollout-count", str(args.rollout_count),
            "--rollout-parallel-threads", str(args.rollout_parallel_threads),
            "--rollout-horizon-sec", str(args.rollout_horizon_sec),
            "--rollout-policy-temperature", "1.0",
            "--discount-factor", str(args.discount_factor),
            "--root-player", args.root_player,
            "--seed", str(int(args.seed) + index),
            "--output-dir", str(process_dir),
            "--native-only",
        ]
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        processes.append((process, log, process_dir))

    peak_rss_bytes = 0
    while any(process.poll() is None for process, _log, _dir in processes):
        peak_rss_bytes = max(
            peak_rss_bytes,
            sum(_rss_bytes(process.pid) for process, _log, _dir in processes if process.poll() is None),
        )
        time.sleep(0.25)
    wall_seconds = time.perf_counter() - started

    elapsed: list[float] = []
    failures: list[dict[str, object]] = []
    for index, (process, log, process_dir) in enumerate(processes):
        log.close()
        result_path = process_dir / "result.json"
        if process.returncode != 0 or not result_path.is_file():
            failures.append({"index": index, "returncode": process.returncode, "log": str(process_dir / "run.log")})
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        elapsed.append(float(payload["native_elapsed_s"]))

    result: dict[str, object] = {
        "processes": int(args.processes),
        "iterations": int(args.iterations),
        "rollout_count": int(args.rollout_count),
        "rollout_parallel_threads": int(args.rollout_parallel_threads),
        "rollout_horizon_sec": float(args.rollout_horizon_sec),
        "wall_seconds": wall_seconds,
        "successful_processes": len(elapsed),
        "failed_processes": failures,
        "searches_per_second": len(elapsed) / wall_seconds,
        "native_elapsed_min_s": min(elapsed) if elapsed else None,
        "native_elapsed_median_s": statistics.median(elapsed) if elapsed else None,
        "native_elapsed_p95_s": _percentile(elapsed, 0.95) if elapsed else None,
        "native_elapsed_max_s": max(elapsed) if elapsed else None,
        "peak_combined_rss_gib": peak_rss_bytes / (1024 ** 3),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rollout-count", type=int, default=10)
    parser.add_argument("--rollout-parallel-threads", type=int, default=1)
    parser.add_argument("--rollout-horizon-sec", type=float, default=1.0)
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--root-player", choices=("controller", "adversary"), default="controller")
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
