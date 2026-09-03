from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

from vidur.config import SimulationConfig
from vidur.entities.execution_time_predictor_request import (
    ExecutionTimePredictorRequest,
)
from vidur.simulator import Simulator
from vidur.execution_time_predictor.sklearn_execution_time_predictor_batch import (
    SklearnExecutionTimePredictorBatch,
)

from .config import load_config


def _simulator_args(cache_mode: str) -> list[str]:
    config = load_config()
    return [
        "predict_vidur_batches",
        "--replica_config_model_name",
        config.vidur_model_name,
        "--replica_config_device",
        config.simulator_device,
        "--replica_config_network_device",
        config.network_device,
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_tensor_parallel_size",
        str(config.tensor_parallel_size),
        "--replica_config_num_pipeline_stages",
        str(config.pipeline_parallel_size),
        "--global_scheduler_config_type",
        "round_robin",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap",
        "512",
        "--execution_time_predictor_config_type",
        "random_forest",
        "--random_forest_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(config.max_tokens_per_request),
        "--random_forest_execution_time_predictor_config_prediction_max_batch_size",
        str(config.max_batch_size),
        "--random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(config.max_prefill_chunk_size),
        "--random_forest_execution_time_predictor_config_compute_input_file",
        str(config.raw_profile_dir / "mlp.csv"),
        "--random_forest_execution_time_predictor_config_attention_input_file",
        str(config.raw_profile_dir / "attention.csv"),
        "--random_forest_execution_time_predictor_config_cache_dir",
        str(config.predictor_cache_dir),
        "--random_forest_execution_time_predictor_config_cache_mode",
        cache_mode,
        "--random_forest_execution_time_predictor_config_num_training_job_threads",
        str(config.predictor_training_threads),
        "--no-snapshot_rng_state",
    ]


def _create_predictor(cache_mode: str):
    original_argv = sys.argv
    try:
        sys.argv = _simulator_args(cache_mode)
        simulation_config = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv
    if hasattr(simulation_config.request_generator_config, "num_requests"):
        simulation_config.request_generator_config.num_requests = 0
    simulation_config.metrics_config.write_metrics = False
    simulation_config.metrics_config.enable_chrome_trace = False
    simulation_config.metrics_config.write_json_trace = False
    simulator = Simulator(simulation_config, register_atexit=False)
    return simulator._execution_time_predictor


def _predict_rows(predictor) -> list[dict[str, object]]:
    config = load_config()
    rows: list[dict[str, object]] = []
    for request_count, tokens in config.cases():
        requests = [
            ExecutionTimePredictorRequest(
                num_processed_tokens=0,
                num_tokens_to_process=tokens,
                is_prefill_complete=False,
            )
            for _ in range(request_count)
        ]
        batch = SklearnExecutionTimePredictorBatch(
            requests,
            predictor._config.kv_cache_prediction_granularity,
            predictor._config.prefill_chunk_size_prediction_granularity,
        )
        execution = predictor.get_execution_time(requests, 0)
        components = execution.to_dict()
        total_tokens = request_count * tokens
        row: dict[str, object] = {
            "request_count": request_count,
            "prefill_tokens_per_request": tokens,
            "total_prefill_tokens": total_tokens,
            "vidur_model_ms": execution.model_time_ms,
            "vidur_total_ms": execution.total_time * 1000.0,
            "vidur_num_layers": execution.num_layers,
            "vidur_attention_aggregate_chunk_tokens": batch.prefill_agg_chunk_size,
            "vidur_attention_lookup_chunk_tokens": min(
                batch.prefill_agg_chunk_size, config.max_prefill_chunk_size
            ),
            "vidur_attention_lookup_clamped": (
                batch.prefill_agg_chunk_size > config.max_prefill_chunk_size
            ),
            "vidur_compute_lookup_total_tokens": min(
                total_tokens, config.max_tokens_per_request
            ),
            "profile_device": config.device_name,
            "calibration_factor": 1.0,
            "calibration_applied": False,
        }
        row.update({f"vidur_per_layer_{key}_ms": value for key, value in components.items()})
        rows.append(row)
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and query the raw Vidur predictor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-mode", choices=("use_cache", "require_cache"), default="use_cache"
    )
    args = parser.parse_args(argv)
    config = load_config()
    for required in (config.raw_profile_dir / "mlp.csv", config.raw_profile_dir / "attention.csv"):
        if not required.is_file():
            raise FileNotFoundError(required)
    config.predictor_cache_dir.mkdir(parents=True, exist_ok=True)
    rows = _predict_rows(_create_predictor(args.cache_mode))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} raw Vidur predictions to {args.output}")


if __name__ == "__main__":
    main()
