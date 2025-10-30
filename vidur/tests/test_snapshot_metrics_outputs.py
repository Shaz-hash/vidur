import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds
from vidur.entities import Batch, BatchStage, Request, Replica
from vidur.entities.execution_time import ExecutionTime
from vidur.events import BaseEvent


# ---------------------------------------------------------------------------
# Shared helpers copied from test_snapshot_equivalence.py
# ---------------------------------------------------------------------------

def make_config() -> SimulationConfig:
    args = [
        "test_snapshot_metrics_outputs.py",
        "--time_limit",
        "10800",
        "--replica_config_model_name",
        "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device",
        "h100",
        "--replica_config_network_device",
        "h100_dgx",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_tensor_parallel_size",
        "1",
        "--replica_config_num_pipeline_stages",
        "1",
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        "128",
        "--length_generator_config_type",
        "trace",
        "--trace_request_length_generator_config_trace_file",
        "./data/processed_traces/mooncake_conversation_trace.csv",
        "--interval_generator_config_type",
        "poisson",
        "--poisson_request_interval_generator_config_qps",
        "2.0",
        "--global_scheduler_config_type",
        "round_robin",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--vllm_v1_scheduler_config_chunk_size",
        "512",
        "--vllm_v1_scheduler_config_batch_size_cap",
        "512",
        "--cache_config_enable_prefix_caching",
        "--metrics_config_write_json_trace",
        "--slo_config_decode_time_slo", 
        "1",
        "--slo_config_completion_time_slo",
        "10"

    ]

    old_argv = sys.argv
    sys.argv = args
    try:
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = old_argv
    return cfg


def step_events(sim: Simulator, n: int) -> None:
    import heapq

    steps = 0
    while steps < n and (
        sim._event_queue
        or sim._request_generator.get_next_request_arrival_time() is not None
    ):
        next_event_time = sim._event_queue[0]._time if sim._event_queue else None
        next_request_arrival_time = sim._request_generator.get_next_request_arrival_time()

        if (next_request_arrival_time is not None) and (
            next_event_time is None or next_request_arrival_time <= next_event_time
        ):
            from vidur.events import RequestArrivalEvent

            sim._add_event(
                RequestArrivalEvent(
                    next_request_arrival_time,
                    sim._request_generator.get_next_request(),
                )
            )
            continue

        event = sim._event_queue[0]
        heapq.heappop(sim._event_queue)
        sim._set_time(event._time)
        new_events = event.handle_event(sim._scheduler, sim._cluster_metric_store)
        sim._add_events(new_events)

        if sim._config.metrics_config.write_json_trace:
            sim._event_trace.append(event.to_dict())

        if sim._config.metrics_config.enable_chrome_trace:
            chrome_trace = event.to_chrome_trace()
            if chrome_trace:
                sim._event_chrome_trace.append(chrome_trace)

        steps += 1


# ---------------------------------------------------------------------------
# New helpers for this script
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("simulator_output/snapshot_trace_outputs")

def reset_entity_ids():
    Request._id = Batch._id = BatchStage._id = ExecutionTime._id = Replica._id = -1
    BaseEvent._id = 0


def prepare_config_with_output(subdir: str) -> SimulationConfig:
    cfg = make_config()
    out_dir = OUTPUT_ROOT / subdir
    os.makedirs(out_dir, exist_ok=True)
    # Override auto-generated timestamped directory so all artifacts land here.
    cfg.metrics_config.output_dir = str(out_dir)
    cfg.metrics_config.save_table_to_wandb = False
    cfg.metrics_config.store_plots = False
    cfg.metrics_config.keep_individual_batch_metrics = True
    cfg.metrics_config.store_batch_metrics = True
    return cfg


def run_full_simulation(cfg: SimulationConfig) -> str:
    """Run a simulator end-to-end and force metrics to disk."""
    set_seeds(cfg.seed)
    reset_entity_ids()
    sim = Simulator(cfg, register_atexit=False)
    sim.run()
    sim._write_output()
    return cfg.metrics_config.output_dir


def run_from_snapshot(snapshot, subdir: str) -> str:
    cfg = prepare_config_with_output(subdir)
    set_seeds(cfg.seed)
    sim = Simulator(cfg, register_atexit=False)
    sim.restore_state(snapshot)
    sim.run()
    sim._write_output()
    return cfg.metrics_config.output_dir


def combine_metrics(branch_dir: str, sources: List[Tuple[str, str]]) -> None:
    # base_path = Path(baseline_dir)
    # branch_path = Path(branch_dir)

    csv_names = set()
    for _, src in sources:
        src_path = Path(src)
        csv_names.update(file.name for file in src_path.glob("*.csv"))

    for filename in csv_names:
        frames: List[pd.DataFrame] = []
        for label, src in sources:
            file_path = Path(src) / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                if not df.empty:
                    df = df.copy()
                    df["__source__"] = label
                    frames.append(df)

        combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined_path = Path(branch_dir) / f"combined_{filename}"
        combined_df.to_csv(combined_path, index=False)


def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Baseline run.
    baseline_cfg = prepare_config_with_output("baseline")
    baseline_dir = run_full_simulation(baseline_cfg)

    # 2. Create snapshot mid-run.
    seed_cfg = prepare_config_with_output("pre_snapshot")
    set_seeds(seed_cfg.seed)
    reset_entity_ids()
    seed_sim = Simulator(seed_cfg, register_atexit=False)
    step_events(seed_sim, n=500)
    seed_sim._write_output()
    snapshot = seed_sim.snapshot_state()

    # 3. Two forked runs from the snapshot.
    branch_a_dir = run_from_snapshot(snapshot, "branch_A")
    branch_b_dir = run_from_snapshot(snapshot, "branch_B")

    # 4. Combine baseline + pre-snapshot + branch metrics per CSV.
    combine_metrics(
        branch_a_dir,
        [
            ("baseline", baseline_dir),
            ("pre_snapshot", seed_cfg.metrics_config.output_dir),
            ("branch_A", branch_a_dir),
        ],
    )
    combine_metrics(
        branch_b_dir,
        [
            ("baseline", baseline_dir),
            ("pre_snapshot", seed_cfg.metrics_config.output_dir),
            ("branch_B", branch_b_dir),
        ],
    )

    print("Baseline metrics written to:", baseline_dir)
    print("Snapshot branch A metrics written to:", branch_a_dir)
    print("Snapshot branch B metrics written to:", branch_b_dir)
    print("Combined CSVs with baseline appended are stored alongside each branch output.")


if __name__ == "__main__":
    main()
