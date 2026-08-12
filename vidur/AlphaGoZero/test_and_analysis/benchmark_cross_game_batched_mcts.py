"""Benchmark independent-game MCTS inference batching without pipeline changes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

from vidur.AlphaGoZero.test_and_analysis.trace_root_puct_evolution import (
    _make_cfg_payload,
)
from vidur.Game_Version3.DNN.native_selfplay import (
    _cfg_payload,
    attach_execution_predictor_payload,
)
from vidur.tests.native_allignment_tests.common import (
    import_native_cpp,
    make_args,
    prepare_python_roots,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS = (
    ROOT
    / "simulator_output/GV3_Agent/PUCT_Evolution_PUCT_0p05_v103"
)
DEFAULT_VALUE = (
    DEFAULT_ANALYSIS
    / "models/Model_Version103/controller_value"
    / "dnn_value_markov_deepset_192_v2/native_model.tsv"
)
DEFAULT_CONTROLLER_POLICY = (
    DEFAULT_ANALYSIS
    / "models/Model_Version103/controller_prior"
    / "dnn_policy_markov_deepset_192_v3/native_model.tsv"
)
DEFAULT_ADVERSARY_POLICY = (
    DEFAULT_ANALYSIS
    / "models/Model_Version104/adversary_prior"
    / "dnn_policy_markov_deepset_192_v3/native_model.tsv"
)
DEFAULT_TASKS = DEFAULT_ANALYSIS / "results/selected_root_tasks.json"
DEFAULT_OUTPUT = (
    ROOT
    / "simulator_output/GV3_Agent"
    / "Cross_Game_Batched_MCTS_Benchmark"
)


def _load_task(path: Path) -> dict[str, Any]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not tasks:
        raise RuntimeError(f"no root tasks in {path}")
    task = dict(tasks[0])
    if int(task.get("pending_prefill_requests", 0)) <= 0:
        raise RuntimeError("benchmark root does not contain pending prefills")
    return task


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    generation_args = make_args(
        "cross_game_batch_benchmark",
        num_roots=1,
        history_hops_min=0,
        history_hops_max=0,
        history_seed=int(args.seed),
        frontier_parity_roots=0,
    )
    pipeline_cfg, simulator, _, _, _ = prepare_python_roots(generation_args)
    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload.update(
        {
            "discount_factor": float(args.discount_factor),
            "use_model_bootstrap": True,
            "use_policy_prior": True,
            "native_search_mode": "full_tree_rollout",
            "rollout_count": int(args.rollout_count),
            "rollout_parallel_threads": 1,
            "rollout_policy_parallel_threads": 1,
            "rollout_horizon_sec": float(args.rollout_horizon_sec),
            "rollout_policy_temperature": 1.0,
            "rollout_probability_quantum": 1e-6,
            "rollout_max_actions": 4096,
            "rollout_optimized_execution": True,
            "capture_rollout_trace": False,
            "capture_root_puct_trace": False,
            "root_dirichlet_noise_enabled": False,
            "puct_c": float(args.puct_c),
            "policy_prior_temperature": 1.0,
            "prior_min_prob": 1e-8,
            "max_forced_hops": int(pipeline_cfg.max_forced_hops_per_root),
            "value_feature_schema": "markov_v2",
            "policy_feature_schema": "markov_v2",
        }
    )
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    native = import_native_cpp(build_if_missing=False)
    value_runtime = native.NewFeatures226HGBRuntime()
    value_runtime.load_model_export(str(args.value_model.resolve()))
    controller_runtime = native.NativeHGBModelRuntime()
    controller_runtime.load_model_export(
        str(args.controller_policy_model.resolve())
    )
    adversary_runtime = native.NativeHGBModelRuntime()
    adversary_runtime.load_model_export(
        str(args.adversary_policy_model.resolve())
    )

    task = _load_task(args.root_tasks)
    payload = _config_payload(args)
    results: list[dict[str, Any]] = []
    for game_count in args.game_counts:
        result = dict(
            native.benchmark_cross_game_mcts_hgb226(
                value_runtime,
                controller_runtime,
                adversary_runtime,
                task["state_payload"],
                payload,
                int(game_count),
                int(args.iterations),
                "controller",
                int(task["root_node_id"]),
                int(task["root_depth"]),
                int(task["sample_id"]),
                int(args.seed),
                int(args.worker_threads),
                int(args.inference_threads),
                int(args.max_batch_requests),
                int(args.max_batch_wait_us),
            )
        )
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    summary = {
        "root": {
            "source": str(args.root_tasks.resolve()),
            "sample_id": int(task["sample_id"]),
            "source_root_id": int(task["source_root_id"]),
            "sim_time": float(task["sim_time"]),
            "pending_prefill_requests": int(task["pending_prefill_requests"]),
            "pending_prefill_tokens": int(task["pending_prefill_tokens"]),
            "action_labels": task.get("action_labels", {}),
        },
        "configuration": {
            "iterations_per_game": int(args.iterations),
            "rollout_count": int(args.rollout_count),
            "rollout_horizon_sec": float(args.rollout_horizon_sec),
            "discount_factor": float(args.discount_factor),
            "puct_c": float(args.puct_c),
            "worker_threads": int(args.worker_threads),
            "inference_threads": int(args.inference_threads),
            "max_batch_requests": int(args.max_batch_requests),
            "max_batch_wait_us": int(args.max_batch_wait_us),
            "value_model": str(args.value_model.resolve()),
            "controller_policy_model": str(
                args.controller_policy_model.resolve()
            ),
            "adversary_policy_model": str(
                args.adversary_policy_model.resolve()
            ),
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if results:
        with (args.output_dir / "timings.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-counts", type=int, nargs="+", default=[100, 1000]
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rollout-count", type=int, default=4)
    parser.add_argument("--rollout-horizon-sec", type=float, default=3.0)
    parser.add_argument("--discount-factor", type=float, default=0.98)
    parser.add_argument("--puct-c", type=float, default=1.25)
    parser.add_argument("--worker-threads", type=int, default=32)
    parser.add_argument("--inference-threads", type=int, default=32)
    parser.add_argument("--max-batch-requests", type=int, default=32)
    parser.add_argument("--max-batch-wait-us", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--root-tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--value-model", type=Path, default=DEFAULT_VALUE)
    parser.add_argument(
        "--controller-policy-model",
        type=Path,
        default=DEFAULT_CONTROLLER_POLICY,
    )
    parser.add_argument(
        "--adversary-policy-model",
        type=Path,
        default=DEFAULT_ADVERSARY_POLICY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
