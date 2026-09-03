from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import flashinfer
import torch
import vllm
from vllm import LLM

from .config import load_config
from .vllm_records import aggregate_vllm_records
from .vllm_block_timing_worker import TIMING_LOG_ENV


def _read_records(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Measure equal-size prefill batches in latest vLLM"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=config.vllm_warmups)
    parser.add_argument("--repetitions", type=int, default=config.vllm_repetitions)
    parser.add_argument("--only-tokens", type=str)
    parser.add_argument("--only-request-counts", type=str)
    args = parser.parse_args(argv)
    if args.raw_log.exists():
        args.raw_log.unlink()
    args.raw_log.parent.mkdir(parents=True, exist_ok=True)
    os.environ[TIMING_LOG_ENV] = str(args.raw_log.resolve())

    tokens = config.prefill_sizes()
    request_counts = config.request_counts
    if args.only_tokens:
        selected = {int(value) for value in args.only_tokens.split(",") if value.strip()}
        tokens = tuple(value for value in tokens if value in selected)
    if args.only_request_counts:
        selected = {
            int(value) for value in args.only_request_counts.split(",") if value.strip()
        }
        request_counts = tuple(value for value in request_counts if value in selected)
    if not tokens or not request_counts:
        raise ValueError("the selected vLLM case set is empty")

    engine = LLM(
        model=config.vllm_model_name,
        runner="generate",
        skip_tokenizer_init=True,
        tensor_parallel_size=config.tensor_parallel_size,
        dtype="half",
        seed=config.random_seed,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        attention_backend=config.attention_backend,
        max_model_len=config.max_tokens_per_request,
        max_num_batched_tokens=config.vllm_max_num_batched_tokens,
        max_num_seqs=config.vllm_max_num_seqs,
        enable_chunked_prefill=config.vllm_enable_chunked_prefill,
        enable_prefix_caching=False,
        async_scheduling=config.vllm_async_scheduling,
        worker_cls=(
            "vidur_vllm_real_testing.profiling_accuracy."
            "vllm_block_timing_worker.TransformerBlockTimedWorker"
        ),
        disable_log_stats=True,
    )
    try:
        rpc_results = engine.collective_rpc(
            "profile_equal_prefill_batches",
            kwargs={
                "cases": [
                    (request_count, token_count)
                    for request_count in request_counts
                    for token_count in tokens
                ],
                "warmups": args.warmups,
                "repetitions": args.repetitions,
            },
        )
    finally:
        engine.llm_engine.engine_core.shutdown()

    if len(rpc_results) != 1:
        raise RuntimeError(f"TP1 profiling expected one worker, got {len(rpc_results)}")
    records = list(rpc_results[0])
    args.raw_log.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    rows = aggregate_vllm_records(
        records,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    expected_shapes = {
        (request_count, token_count)
        for request_count in request_counts
        for token_count in tokens
    }
    measured_shapes = {
        (int(row["request_count"]), int(row["prefill_tokens_per_request"]))
        for row in rows
    }
    if measured_shapes != expected_shapes:
        raise RuntimeError(
            f"measured shapes differ from plan: missing={expected_shapes - measured_shapes} "
            f"extra={measured_shapes - expected_shapes}"
        )
    for row in rows:
        row.update(
            {
                "vllm_version": vllm.__version__,
                "flashinfer_version": flashinfer.__version__,
                "torch_version": torch.__version__,
                "attention_backend": config.attention_backend,
                "model_source": config.vllm_model_name,
                "timing_boundary": "first_decoder_layer_start_to_last_decoder_layer_end",
                "batch_construction": "vllm_model_runner_dummy_prefill_rpc",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} vLLM batch measurements to {args.output}")


if __name__ == "__main__":
    main()
