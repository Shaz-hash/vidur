"""Validate and execute Task 1.1.2 on one explicitly selected mew1 GPU."""

from __future__ import annotations

import argparse
import csv
import getpass
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Iterable

from .canonicalization import PREFILL_ROUNDING_CEILING
from .native_dnn_mcts_state_planner import (
    ProductionNativeDNNMCTSPlanner,
    load_frozen_native_cfg,
)
from .prompt_materialization import verify_prompt_artifacts
from .scheduler_contract import TraceMetadataRegistry
from .task_1_1_2_trace import DEFAULT_TOKENIZER_DIR, prepare_task_1_1_2_trace
from .test_traces_on_GPU_with_AlphaGOZERO_models_config import (
    AlphaGoZeroGPUTraceConfig,
    GV3_IN_DISTRIBUTION_PREFILL_TOKENS,
    TraceType,
    load_config,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def _read_canonical_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_prepared_trace(
    config: AlphaGoZeroGPUTraceConfig,
    *,
    prepared_root: str | Path,
    tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
) -> dict[str, object]:
    config.validate()
    root = Path(prepared_root).expanduser().resolve()
    canonical = root / "canonical_trace.csv"
    raw = root / "raw_trace.csv"
    manifest_path = root / "trace_manifest.json"
    config_path = root / "experiment_config.json"
    for path in (canonical, raw, manifest_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    registry = TraceMetadataRegistry.load(canonical)
    rows = _read_canonical_rows(canonical)
    if len(rows) != len(registry):
        raise RuntimeError("canonical trace registry lost requests")
    verified_prompts = verify_prompt_artifacts(
        raw_trace_path=raw,
        tokenizer_dir=tokenizer_dir,
    )
    if verified_prompts != len(rows):
        raise RuntimeError("prompt and canonical trace counts differ")

    trace_type = TraceType.parse(config.trace_type)
    rounded_rows = 0
    for row in rows:
        actual = int(row["actual_prefill_tokens"])
        canonical_tokens = int(row["canonical_prefill_tokens"])
        if trace_type is TraceType.IN_DISTRIBUTION:
            if actual != canonical_tokens:
                raise RuntimeError("in-distribution physical and canonical lengths differ")
            if actual not in GV3_IN_DISTRIBUTION_PREFILL_TOKENS:
                raise RuntimeError(f"in-distribution request uses unsupported size {actual}")
        else:
            expected = ((actual + 127) // 128) * 128
            if canonical_tokens != expected:
                raise RuntimeError(
                    f"out-distribution ceiling mismatch: {actual} -> {canonical_tokens}, "
                    f"expected {expected}"
                )
            rounded_rows += int(actual != canonical_tokens)
    if trace_type is TraceType.OUT_DISTRIBUTION and rounded_rows == 0:
        raise RuntimeError("out-distribution trace contains no non-grid physical prompts")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rounding = (
        PREFILL_ROUNDING_CEILING
        if trace_type is TraceType.OUT_DISTRIBUTION
        else "nearest_128_half_up"
    )
    if manifest["canonicalization"]["prefill_rounding"] != expected_rounding:
        raise RuntimeError("trace manifest records the wrong prefill rounding mode")
    if manifest["task_1_1_2"]["simultaneous_arrival_groups"] <= 0:
        raise RuntimeError("trace does not exercise simultaneous request arrivals")
    if manifest["task_1_1_2"]["controller_wall_time_in_clock"] is not False:
        raise RuntimeError("trace manifest incorrectly includes controller wall time")
    if manifest["task_1_1_2"]["calibration"] is not None:
        raise RuntimeError("Task 1.1.2 must never apply calibration")
    return {
        "canonical_trace": str(canonical),
        "request_count": len(rows),
        "verified_prompts": verified_prompts,
        "rounded_rows": rounded_rows,
        "trace_type": trace_type.value,
        "simultaneous_arrival_groups": manifest["task_1_1_2"][
            "simultaneous_arrival_groups"
        ],
    }


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", ""))


def runtime_preflight(
    config: AlphaGoZeroGPUTraceConfig,
    *,
    require_free_gpu: bool,
) -> dict[str, object]:
    config.validate()
    if socket.gethostname().split(".")[0] != "mew1" or getpass.getuser() != "shaz":
        raise RuntimeError("Task 1.1.2 runtime is restricted to shaz@mew1")
    versions = {
        "vllm": _module_version("vllm"),
        "flashinfer": _module_version("flashinfer"),
        "torch": _module_version("torch"),
    }
    expected = {
        "vllm": config.vllm_version,
        "flashinfer": config.flashinfer_version,
    }
    for name, version in expected.items():
        if versions[name] != version:
            raise RuntimeError(f"{name} version mismatch: {versions[name]} != {version}")
    if not versions["torch"].startswith(config.torch_version):
        raise RuntimeError(
            f"torch version mismatch: {versions['torch']} does not start with {config.torch_version}"
        )

    nvcc = Path(config.cuda_home) / "bin/nvcc"
    compiler = subprocess.run(
        [str(nvcc), "--version"], check=True, capture_output=True, text=True
    ).stdout
    if f"V{config.cuda_compiler_version}" not in compiler:
        raise RuntimeError(
            f"CUDA compiler mismatch: expected V{config.cuda_compiler_version} at {nvcc}"
        )

    config.validate_model_artifacts()
    native_cfg = load_frozen_native_cfg(Path(config.native_mcts_config_path))
    if str(native_cfg.get("value_feature_schema")) != "markov_v2":
        raise RuntimeError("native MCTS config is not Markov-v2")
    if str(native_cfg.get("policy_feature_schema")) != "markov_v2":
        raise RuntimeError("native MCTS policy config is not Markov-v2")

    old_environment = os.environ.copy()
    try:
        os.environ.update(
            config.to_environment(
                canonical_trace=Path("/dev/null"),
                scheduler_log=Path("/dev/null"),
            )
        )
        planner = ProductionNativeDNNMCTSPlanner.from_environment()
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
    if not bool(planner.value_runtime.loaded):
        raise RuntimeError("controller value DNN did not load in the native runtime")
    if not bool(planner.controller_prior_runtime.is_markov_policy):
        raise RuntimeError("controller policy is not a native Markov DNN")
    if not bool(planner.adversary_prior_runtime.is_markov_policy):
        raise RuntimeError("adversary policy is not a native Markov DNN")

    command = [
        "nvidia-smi",
        "-i",
        str(config.gpu_index),
        "--query-compute-apps=pid",
        "--format=csv,noheader",
    ]
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    gpu_pids = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if require_free_gpu and gpu_pids:
        raise RuntimeError(
            f"mew1 GPU {config.gpu_index} is occupied by PIDs {gpu_pids}; refusing to disturb it"
        )
    return {
        "gpu_index": config.gpu_index,
        "gpu_process_ids": gpu_pids,
        "versions": versions,
        "cuda_compiler": config.cuda_compiler_version,
        "native_cfg": str(Path(config.native_mcts_config_path).resolve()),
        "promoted_versions": {
            "controller": planner.controller_model_version,
            "adversary": planner.adversary_model_version,
        },
        "native_dnn_models_loaded": True,
        "calibration": None,
    }


def run_benchmark(
    config: AlphaGoZeroGPUTraceConfig,
    *,
    prepared_root: str | Path,
    output_dir: str | Path,
    mode: str = "controller",
) -> None:
    if mode not in {"controller", "sjf256", "sjf512"}:
        raise ValueError("mode must be controller, sjf256, or sjf512")
    trace_result = validate_prepared_trace(config, prepared_root=prepared_root)
    runtime_preflight(config, require_free_gpu=True)
    if mode == "controller":
        config.validate_model_artifacts()

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    scheduler_log = output / "scheduler.jsonl"
    environment = os.environ.copy()
    environment.update(
        config.to_environment(
            canonical_trace=Path(str(trace_result["canonical_trace"])),
            scheduler_log=scheduler_log,
        )
    )
    environment.update(
        {
            "VIDUR_MEW1_ROOT": config.remote_root,
            "VIDUR_REAL_SOURCE_ROOT": config.remote_source_root,
            "VIDUR_REAL_ENV_DIR": config.remote_environment,
            "VIDUR_REAL_HF_HOME": config.remote_hf_home,
            "VIDUR_REAL_CUDA_HOME": config.cuda_home,
            "VIDUR_VLLM_ATTENTION_BACKEND": config.attention_backend,
            "VIDUR_VLLM_MAX_NUM_BATCHED_TOKENS": str(config.max_num_batched_tokens),
            "VIDUR_VLLM_MAX_NUM_SEQS": str(config.max_num_seqs),
            "VIDUR_VLLM_GPU_MEMORY_UTILIZATION": str(config.gpu_memory_utilization),
        }
    )
    launcher = PACKAGE_DIR / "mew1/run_real_policy_benchmark.sh"
    subprocess.run(
        ["bash", str(launcher), mode, str(output)],
        check=True,
        env=environment,
    )


def main(argv: Iterable[str] | None = None) -> None:

    """
        Does the following in the normal intended way :
        1. Loads config from the test_traces_on_GPU_with_AlphaGOZERO_models_config.py
        2. prepares trace 
        3. validates trace, GPU envi, and DNN bundle
        4. launch vllm server for experiment

        for example :
            python -m vidur_vllm_real_testing.task_1_1_2_runner prepare \
            --output-root /path/to/prepared_trace

            python -m vidur_vllm_real_testing.task_1_1_2_runner validate \
            --prepared-root /path/to/prepared_trace \
            --runtime \
            --require-free-gpu

            python -m vidur_vllm_real_testing.task_1_1_2_runner run \
            --prepared-root /path/to/prepared_trace \
            --output-dir /path/to/results \
            --mode controller

    """



    parser = argparse.ArgumentParser(description="Task 1.1.2 trace/GPU pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", required=True, type=Path)
    prepare_parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--prepared-root", required=True, type=Path)
    validate_parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    validate_parser.add_argument("--runtime", action="store_true")
    validate_parser.add_argument("--require-free-gpu", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--prepared-root", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument(
        "--mode", choices=("controller", "sjf256", "sjf512"), default="controller"
    )

    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "prepare":
        artifact = prepare_task_1_1_2_trace(
            config,
            output_root=args.output_root,
            tokenizer_dir=args.tokenizer_dir,
        )
        result: object = {
            "canonical_trace": str(artifact.canonical_trace),
            "manifest": str(artifact.manifest),
            "request_count": artifact.request_count,
        }
    elif args.command == "validate":
        result = {
            "trace": validate_prepared_trace(
                config,
                prepared_root=args.prepared_root,
                tokenizer_dir=args.tokenizer_dir,
            )
        }
        if args.runtime:
            result["runtime"] = runtime_preflight(
                config,
                require_free_gpu=bool(args.require_free_gpu),
            )
    else:
        run_benchmark(
            config,
            prepared_root=args.prepared_root,
            output_dir=args.output_dir,
            mode=args.mode,
        )
        result = {"status": "complete", "output_dir": str(args.output_dir)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
