from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List

from vidur.config import SimulationConfig
from vidur.entities.batch import Batch
from vidur.entities.request import Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.types.replica_id import ReplicaId


# Same defaults used in alphaZeroParrallel.py
DEFAULT_SIM_CLI_ARGS = [
    "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
    "--replica_config_device", "h100",
    "--replica_config_network_device", "h100_dgx",
    "--cluster_config_num_replicas", "1",
    "--replica_config_tensor_parallel_size", "1",
    "--replica_config_num_pipeline_stages", "1",
    "--global_scheduler_config_type", "round_robin",
    "--replica_scheduler_config_type", "vllm_v1",
    "--vllm_v1_scheduler_config_batch_size_cap", "512",
    "--no-snapshot_rng_state",
]


def configure_simulation(sim_args: Iterable[str]) -> SimulationConfig:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(sim_args)
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv

    # Keep lightweight
    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False
    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]

    return cfg


def parse_decode_counts(raw: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        v = int(token)
        if v < 0:
            raise ValueError(f"decode count must be >= 0, got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("decode_counts cannot be empty")
    return vals


def make_prefill_request(replica_id: ReplicaId, prefill_tokens: int, decode_tokens: int) -> Request:
    req = Request(
        arrived_at=0.0,
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=decode_tokens,
        block_hash_ids=None,
        block_size=None,
    )
    req.assign_replica(replica_id)
    return req


def make_decode_ready_request(replica_id: ReplicaId, prefill_tokens: int, decode_tokens: int) -> Request:
    # req = Request(
    #     arrived_at=0.0,
    #     num_prefill_tokens=prefill_tokens,
    #     num_decode_tokens=decode_tokens,
    #     block_hash_ids=None,
    #     block_size=None,
    # )
    # req.assign_replica(replica_id)
    req = Request(
        arrived_at=0.0,
        num_prefill_tokens=3072,
        num_decode_tokens=5000,
        block_hash_ids=None,
        block_size=None,
    )
    req.assign_replica(ReplicaId(0))
    req._num_processed_tokens = 3072
    req._is_prefill_complete = True


    # Mark as decode-phase request (prefill already complete)
    req._num_processed_tokens = req.num_prefill_tokens
    req._is_prefill_complete = True
    req._scheduled = True
    req._scheduled_at = 0.0
    req._prefill_completed_at = 0.0
    return req


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSV: total_prefill, number_of_decodes_in_batch, batch_execution_time"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="simulator_output/mcts_dnn_logs/prefill_decode_batch_times.csv",
    )
    parser.add_argument("--prefill_tokens", type=int, default=3072)
    parser.add_argument("--decode_counts", type=str, default="10,20,30,40,50")
    parser.add_argument("--decode_tokens_per_request", type=int, default=5000)
    parser.add_argument("--decode_tokens_scheduled", type=int, default=1)

    # Any unknown args are forwarded to SimulationConfig CLI parser
    args, sim_cli_args = parser.parse_known_args()

    if args.prefill_tokens <= 0:
        raise ValueError("--prefill_tokens must be > 0")
    if args.decode_tokens_per_request <= 0:
        raise ValueError("--decode_tokens_per_request must be > 0")
    if args.decode_tokens_scheduled <= 0:
        raise ValueError("--decode_tokens_scheduled must be > 0")

    decode_counts = parse_decode_counts(args.decode_counts)
    if not sim_cli_args:
        sim_cli_args = list(DEFAULT_SIM_CLI_ARGS)

    sim_cfg = configure_simulation(sim_cli_args)

    predictor = ExecutionTimePredictorRegistry.get(
        sim_cfg.execution_time_predictor_config.get_type(),
        predictor_config=sim_cfg.execution_time_predictor_config,
        replica_config=sim_cfg.cluster_config.replica_config,
        cache_config=sim_cfg.cluster_config.cache_config,
    )

    replica_id = ReplicaId(0)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "total_prefill",
        "number_of_decodes_in_batch",
        "batch_execution_time",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n_decode in decode_counts:
            requests = [
                make_prefill_request(replica_id, args.prefill_tokens, args.decode_tokens_per_request)
            ]
            num_tokens = [args.prefill_tokens]

            for _ in range(n_decode):
                requests.append(
                    make_decode_ready_request(replica_id, args.prefill_tokens, args.decode_tokens_per_request)
                )
                num_tokens.append(args.decode_tokens_scheduled)

            batch = Batch(replica_id=replica_id, requests=requests, num_tokens=num_tokens)
            et = predictor.get_batch_execution_time(batch, pipeline_stage=0)

            writer.writerow(
                {
                    "total_prefill": int(batch.num_prefill_tokens),
                    "number_of_decodes_in_batch": int(n_decode),
                    "batch_execution_time": float(et.total_time),
                }
            )

    print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
