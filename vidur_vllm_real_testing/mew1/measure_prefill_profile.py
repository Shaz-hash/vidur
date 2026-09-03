#!/usr/bin/env python3
"""Measure isolated Llama-3-8B prefills through the persistent GV3 hook."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import statistics
from typing import Any


def _read_profile(path: Path) -> dict[int, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            int(row["prefill_tokens"]): float(row["prefill_time_seconds"])
            for row in csv.DictReader(handle)
        }


def _write_trace(
    path: Path,
    jobs: list[tuple[int, int, bool]],
    profile: dict[int, float],
) -> None:
    columns = (
        "schema_version",
        "request_id",
        "arrived_at_s",
        "actual_prefill_tokens",
        "canonical_prefill_tokens",
        "actual_decode_tokens",
        "canonical_decode_tokens",
        "actual_prefill_slo_s",
        "canonical_prefill_slo_s",
        "actual_decode_slo_s",
        "canonical_decode_slo_s",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for request_index, (_, tokens, _) in enumerate(jobs):
            writer.writerow(
                {
                    "schema_version": "vidur-vllm-trace-v1",
                    "request_id": str(request_index),
                    "arrived_at_s": 0.0,
                    "actual_prefill_tokens": tokens,
                    "canonical_prefill_tokens": tokens,
                    "actual_decode_tokens": 1,
                    "canonical_decode_tokens": 1,
                    "actual_prefill_slo_s": 3.0 * profile[tokens],
                    "canonical_prefill_slo_s": 3.0 * profile[tokens],
                    "actual_decode_slo_s": 0.05,
                    "canonical_decode_slo_s": 0.05,
                }
            )


def _load_observations(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    source = root / "source/vidur-classical-search"
    profile_path = source / "simulator_output/prefill_profile.csv"
    profile = _read_profile(profile_path)
    tokens = [int(value) for value in args.tokens]
    unknown = sorted(set(tokens) - set(profile))
    if unknown:
        raise ValueError(f"tokens absent from frozen profile: {unknown}")

    jobs: list[tuple[int, int, bool]] = [
        (index, int(args.warmup_tokens), True) for index in range(args.warmups)
    ]
    measured = [
        (0, token_count, False)
        for token_count in tokens
        for _ in range(args.repetitions)
    ]
    random.Random(args.seed).shuffle(measured)
    jobs.extend(
        (index + args.warmups, token_count, False)
        for index, (_, token_count, _) in enumerate(measured)
    )

    run_dir = Path(args.output).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "prefill_profile_trace.csv"
    observations_path = run_dir / "batch_observations.jsonl"
    observations_path.unlink(missing_ok=True)
    _write_trace(trace_path, jobs, profile)

    os.environ["VIDUR_VLLM_SCHEDULER_MODE"] = "controller"
    os.environ["VIDUR_VLLM_CANONICAL_TRACE"] = str(trace_path)
    os.environ["VIDUR_VLLM_GV3_PLANNER"] = (
        "vidur_vllm_real_testing.prefill_profile_adapter:GV3PrefillProfilingAdapter"
    )
    os.environ["VIDUR_VLLM_GV3_STATE_PLANNER"] = (
        "vidur_vllm_real_testing.prefill_profile_adapter:full_prefill_state_planner"
    )
    os.environ["VIDUR_VLLM_GV3_BATCH_OBSERVATIONS"] = str(observations_path)
    # This profiler ends after the physical token sampled with prefill and does
    # not exercise canonical GV3 decode actions.
    os.environ["VIDUR_VLLM_GV3_IMPLICIT_PREFILL_OUTPUT_TOKENS"] = "0"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    model_dir = source / "vidur_vllm_real_testing/models/llama3_8b_dummy"
    tokenizer_dir = source / "vidur_vllm_real_testing/tokenizer/llama3_8b"
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(tokenizer_dir),
        load_format="dummy",
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=4352,
        max_num_batched_tokens=4608,
        max_num_seqs=8,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        scheduling_policy="fcfs",
        scheduler_cls=(
            "vidur_vllm_real_testing.gv3_persistent_scheduler:GV3PersistentScheduler"
        ),
        disable_log_stats=True,
        seed=args.seed,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    for _, token_count, _ in jobs:
        prompt = {"prompt_token_ids": [1000] * token_count}
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].outputs[0].token_ids) != 1:
            raise RuntimeError(f"unexpected vLLM output for {token_count} tokens")

    observations = _load_observations(observations_path)
    if len(observations) != len(jobs):
        raise RuntimeError(
            f"expected {len(jobs)} completed batches, observed {len(observations)}"
        )

    durations: dict[int, list[float]] = {token_count: [] for token_count in tokens}
    raw_rows: list[dict[str, Any]] = []
    for job, observation in zip(jobs, observations):
        request_index, token_count, warmup = job
        scheduled = dict(observation["scheduled_tokens_by_request"])
        if scheduled != {str(request_index): token_count}:
            raise RuntimeError(
                f"request {request_index}: expected one {token_count}-token prefill, "
                f"observed {scheduled}"
            )
        duration = float(observation["duration_s"])
        raw_rows.append(
            {
                "request_id": request_index,
                "prefill_tokens": token_count,
                "warmup": int(warmup),
                "observed_batch_s": duration,
                "profile_s": profile[token_count],
                "ratio_observed_to_profile": duration / profile[token_count],
            }
        )
        if not warmup:
            durations[token_count].append(duration)

    raw_path = run_dir / "prefill_measurements_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows: list[dict[str, Any]] = []
    for token_count in tokens:
        values = durations[token_count]
        median = statistics.median(values)
        expected = profile[token_count]
        summary_rows.append(
            {
                "prefill_tokens": token_count,
                "samples": len(values),
                "observed_median_s": median,
                "observed_mean_s": statistics.fmean(values),
                "observed_min_s": min(values),
                "observed_max_s": max(values),
                "profile_s": expected,
                "median_absolute_error_s": median - expected,
                "median_relative_error": (median - expected) / expected,
                "ratio_observed_to_profile": median / expected,
            }
        )
    summary_path = run_dir / "prefill_profile_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "status": "passed",
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model_architecture": "Llama-3-8B exact config with vLLM dummy weights",
        "scheduler": "GV3PersistentScheduler",
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "summary_csv": str(summary_path),
        "raw_csv": str(raw_path),
        "rows": summary_rows,
    }
    (run_dir / "prefill_profile_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/shaz/vidur")
    parser.add_argument(
        "--output",
        default="/home/shaz/vidur/runs/prefill_profile_gpu2",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024, 2048, 3072, 4096],
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

