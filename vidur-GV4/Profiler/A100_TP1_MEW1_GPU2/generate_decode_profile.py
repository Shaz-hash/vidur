from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

from vidur.config import SimulationConfig
from vidur.entities.execution_time_predictor_request import (
    ExecutionTimePredictorRequest,
)
from vidur.simulator import Simulator


def _parse_tokens(value: str) -> list[int]:
    tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not tokens or any(token <= 0 for token in tokens):
        raise argparse.ArgumentTypeError("tokens must be positive comma-separated integers")
    return tokens


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a TP1 decode timing table")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=_parse_tokens, required=True)
    args, simulator_args = parser.parse_known_args(argv)

    original_argv = sys.argv
    try:
        sys.argv = ["generate_decode_profile"] + simulator_args
        config = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv

    if hasattr(config.request_generator_config, "num_requests"):
        config.request_generator_config.num_requests = 0
    config.metrics_config.write_metrics = False
    config.metrics_config.enable_chrome_trace = False
    config.metrics_config.write_json_trace = False

    simulator = Simulator(config, register_atexit=False)
    predictor = simulator._execution_time_predictor

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decode tokens", "decode_time_seconds"])
        for context_tokens in args.tokens:
            request = ExecutionTimePredictorRequest(
                num_processed_tokens=context_tokens,
                num_tokens_to_process=1,
                is_prefill_complete=True,
            )
            execution_time = predictor.get_execution_time([request], 0)
            writer.writerow([context_tokens, float(execution_time.total_time)])

    print(f"Generated decode profile -> {args.output}")


if __name__ == "__main__":
    main()
