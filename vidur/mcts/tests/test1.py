# vidur/vidur/mcts/tests/test_sim_snapshot_equivalence.py

import os
import shutil
import sys
from pathlib import Path
from typing import List

import pandas as pd
from pandas.api.types import is_numeric_dtype

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds
from vidur.entities import Batch, BatchStage, Request, Replica
from vidur.entities.execution_time import ExecutionTime
from vidur.events import BaseEvent

# ---------------------------------------------------------------------------
# 1) Build the same config you use via CLI (testing_configs.txt)
# ---------------------------------------------------------------------------

def make_config() -> SimulationConfig:
    """
    Build a SimulationConfig equivalent to the TESTING CONFIG in testing_configs.txt.

    You can adjust these args to match whatever config you want to validate.
    """
    args = [
            "test_sim_snapshot_equivalence.py",
            "--time_limit", "10800",
            "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
            "--replica_config_device", "a100",
            "--replica_config_network_device", "a100_dgx",
            "--cluster_config_num_replicas", "1",
            "--replica_config_tensor_parallel_size", "2",
            "--replica_config_num_pipeline_stages", "1",
            "--request_generator_config_type", "synthetic",
            "--synthetic_request_generator_config_num_requests", "128",
            "--length_generator_config_type", "trace",
            "--trace_request_length_generator_config_trace_file",
            "./data/processed_traces/mooncake_conversation_trace.csv",
            "--interval_generator_config_type", "poisson",
            "--poisson_request_interval_generator_config_qps", "2.0",
            "--global_scheduler_config_type", "round_robin",
            "--replica_scheduler_config_type", "vllm_v1",
            "--vllm_v1_scheduler_config_chunk_size", "512",
            "--vllm_v1_scheduler_config_batch_size_cap", "512",
            "--metrics_config_write_metrics",
            "--metrics_config_store_operation_metrics",
            "--metrics_config_keep_individual_batch_metrics",
            "--cache_config_enable_prefix_caching",
            "--slo_config_decode_time_slo", "1",
            "--slo_config_completion_time_slo", "10",
            "--metrics_config_write_json_trace",
        ]


    old_argv = sys.argv
    sys.argv = args
    try:
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = old_argv
    return cfg

# ---------------------------------------------------------------------------
# 2) Helpers to drive the simulator deterministically
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("simulator_output/mcts_snapshot_tests")
SNAPSHOT_STEPS = 50  # number of event-handles before we snapshot/restore
SNAPSHOT_COMPLETED_BATCHES = 8  # e.g., snapshot after 8 completed batches

def reset_entity_ids() -> None:
    """Reset global ID counters so two runs have identical Request/Batch IDs."""
    Request._id = Batch._id = BatchStage._id = ExecutionTime._id = Replica._id = -1
    BaseEvent._id = 0

def step_events(sim: Simulator, n: int) -> None:
    """
    Advance the simulator by exactly n event-handles (or until idle),
    mirroring the main run loop semantics.
    """
    import heapq

    steps = 0
    while steps < n and (
        sim._event_queue
        or sim._request_generator.get_next_request_arrival_time() is not None
    ):
        next_event_time = sim._event_queue[0]._time if sim._event_queue else None
        next_request_arrival_time = sim._request_generator.get_next_request_arrival_time()

        # Inject arrival if it comes first
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

        # One normal event
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

def step_until_completed_batches(sim: Simulator, target_batches: int) -> None:
    """
    Advance the simulator until exactly `target_batches` BatchEndEvent events
    have been handled (or until the sim goes idle).
    """
    import heapq
    from vidur.events.batch_end_event import BatchEndEvent
    from vidur.events import RequestArrivalEvent

    completed = 0
    while completed < target_batches and (
        sim._event_queue
        or sim._request_generator.get_next_request_arrival_time() is not None
    ):
        next_event_time = sim._event_queue[0]._time if sim._event_queue else None
        next_arrival = sim._request_generator.get_next_request_arrival_time()

        # Inject arrivals just like the main run loop
        if (next_arrival is not None) and (
            next_event_time is None or next_arrival <= next_event_time
        ):
            sim._add_event(
                RequestArrivalEvent(next_arrival, sim._request_generator.get_next_request())
            )
            continue

        event = sim._event_queue[0]
        heapq.heappop(sim._event_queue)
        sim._set_time(event._time)
        new_events = event.handle_event(sim._scheduler, sim._cluster_metric_store)
        sim._add_events(new_events)

        if isinstance(event, BatchEndEvent):
            completed += 1

        if sim._config.metrics_config.write_json_trace:
            sim._event_trace.append(event.to_dict())

        if sim._config.metrics_config.enable_chrome_trace:
            ct = event.to_chrome_trace()
            if ct:
                sim._event_chrome_trace.append(ct)

def prepare_config_with_output(subdir: str) -> SimulationConfig:
    """
    Build a config and override metrics output_dir to a fixed subdir
    so we can compare CSVs across runs.
    """
    cfg = make_config()
    out_dir = OUTPUT_ROOT / subdir
    os.makedirs(out_dir, exist_ok=True)

    # Override auto timestamped directory
    cfg.metrics_config.output_dir = str(out_dir)
    cfg.metrics_config.save_table_to_wandb = False
    cfg.metrics_config.store_plots = False

    # Make sure we actually write the metrics we want to compare
    cfg.metrics_config.write_metrics = True
    cfg.metrics_config.store_request_metrics = True
    cfg.metrics_config.store_batch_metrics = True
    cfg.metrics_config.keep_individual_batch_metrics = True
    cfg.metrics_config.store_operation_metrics = True

    return cfg

def run_baseline() -> str:
    """Run full simulation without snapshot, return output_dir."""
    cfg = prepare_config_with_output("baseline")
    set_seeds(cfg.seed)
    reset_entity_ids()
    sim = Simulator(cfg, register_atexit=False)
    sim.run()
    sim._write_output()
    return cfg.metrics_config.output_dir

def run_with_mid_snapshot() -> str:
    """
    Run simulation but after SNAPSHOT_STEPS events:
      - take a snapshot,
      - restore into a fresh Simulator,
      - complete the run from that restored state.
    Return the output_dir of the restored run.
    """
    cfg = prepare_config_with_output("with_snapshot")
    set_seeds(cfg.seed)
    reset_entity_ids()
    # Seed run up to snapshot point
    seed_sim = Simulator(cfg, register_atexit=False)
    # step_events(seed_sim, SNAPSHOT_STEPS)
    step_until_completed_batches(seed_sim, SNAPSHOT_COMPLETED_BATCHES)
    snapshot = seed_sim.snapshot_state()

    # *** NEW: reset only Replica IDs before constructing the restored simulator ***
    Replica._id = -1

    # Continue from an immediately restored instance
    sim_restored = Simulator(cfg, register_atexit=False)
    sim_restored.restore_state(snapshot)
    sim_restored.run()
    sim_restored._write_output()

    return cfg.metrics_config.output_dir

# ---------------------------------------------------------------------------
# 3) CSV comparison helpers
# ---------------------------------------------------------------------------

def load_and_normalize_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Sort by all columns to ignore row order differences
    cols: List[str] = sorted(df.columns.tolist())
    df = df[cols].sort_values(by=cols).reset_index(drop=True)
    return df


def assert_batch_metrics_equal(baseline_dir: str, snapshot_dir: str) -> None:
    path_a = Path(baseline_dir) / "batch_metrics.csv"
    path_b = Path(snapshot_dir) / "batch_metrics.csv"
    assert path_a.exists(), f"{path_a} missing"
    assert path_b.exists(), f"{path_b} missing"

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    print("Baseline Snapshot Number : ", SNAPSHOT_COMPLETED_BATCHES)
    # Keep only post‑snapshot batches in baseline
    # (i.e., drop the first SNAPSHOT_COMPLETED_BATCHES completed batches)
    df_a_tail = df_a[df_a["Batch Id"] >= SNAPSHOT_COMPLETED_BATCHES].copy()
    df_a_tail = df_a_tail.sort_values("Batch Id").reset_index(drop=True)

    # Snapshot run should contain exactly these post‑snapshot batches
    df_b_sorted = df_b.sort_values("Batch Id").reset_index(drop=True)
    df_b_sorted["Batch Id"] += SNAPSHOT_COMPLETED_BATCHES

    assert len(df_a_tail) == len(df_b_sorted), (
        f"batch count mismatch after snapshot: "
        f"{len(df_a_tail)} baseline vs {len(df_b_sorted)} snapshot"
    )

    # Check that Batch Ids line up 1:1
    assert (df_a_tail["Batch Id"].values == df_b_sorted["Batch Id"].values).all(), (
        f"Batch Ids differ:\n"
        f"baseline: {df_a_tail['Batch Id'].tolist()}\n"
        f"snapshot: {df_b_sorted['Batch Id'].tolist()}"
    )

    # Compare all other columns with tolerance, ignore replica if you don’t care
    ignore_cols = {"replica"}  # optional
    cols = [c for c in df_a_tail.columns if c not in ignore_cols]

    for col in cols:
        pd.testing.assert_series_equal(
            df_a_tail[col].reset_index(drop=True),
            df_b_sorted[col].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-5,
            atol=1e-6,
            obj=f"column {col}",
        )

def assert_request_metrics_equal(baseline_dir: str, snapshot_dir: str) -> None:
    path_a = Path(baseline_dir) / "request_metrics.csv"
    path_b = Path(snapshot_dir) / "request_metrics.csv"
    assert path_a.exists(), f"{path_a} missing"
    assert path_b.exists(), f"{path_b} missing"

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    id_col = "Request Id"
    assert id_col in df_a.columns and id_col in df_b.columns

    # Drop columns we don't care about / are history‑dependent
    drop_cols = {"replica", "request_inter_arrival_delay"}
    for df in (df_a, df_b):
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Only compare requests whose snapshot metrics are "complete":
    # e.g. have a non‑null num_tokens (you can also use request_arrived_at)
    complete_mask = df_b["request_num_tokens"].notna()
    df_b_sub = df_b[complete_mask].copy()

    # Align baseline to the same set of request_ids
    df_a_sub = df_a[df_a[id_col].isin(df_b_sub[id_col])].copy()

    df_a_sub = df_a_sub.set_index(id_col).sort_index()
    df_b_sub = df_b_sub.set_index(id_col).sort_index()

    base_ids = set(df_a_sub.index)
    snap_ids = set(df_b_sub.index)
    missing_in_baseline = snap_ids - base_ids
    missing_in_snapshot = base_ids - snap_ids
    if missing_in_baseline or missing_in_snapshot:
        msg = []
        if missing_in_baseline:
            msg.append(f"Missing in baseline: {sorted(missing_in_baseline)[:10]}")
        if missing_in_snapshot:
            msg.append(f"Missing in snapshot: {sorted(missing_in_snapshot)[:10]}")
        raise AssertionError("Request id mismatch between baseline and snapshot.\n" + "\n".join(msg))

    common_ids = sorted(base_ids & snap_ids)
    common_cols = [c for c in df_a_sub.columns if c in df_b_sub.columns]

    mismatches: dict[int, list[str]] = {}

    for rid in common_ids:
        row_a = df_a_sub.loc[rid]
        row_b = df_b_sub.loc[rid]
        diff_cols: list[str] = []
        for col in common_cols:
            a_val = row_a[col]
            b_val = row_b[col]
            if is_numeric_dtype(df_a_sub[col]) and is_numeric_dtype(df_b_sub[col]):
                if pd.isna(a_val) and pd.isna(b_val):
                    continue
                if pd.isna(a_val) or pd.isna(b_val):
                    diff_cols.append(col)
                    continue
                # small tolerance for float noise
                if not (abs(a_val - b_val) <= 1e-6 or abs(a_val - b_val) <= 1e-5 * max(abs(a_val), abs(b_val))):
                    diff_cols.append(col)
            else:
                if pd.isna(a_val) and pd.isna(b_val):
                    continue
                if a_val != b_val:
                    diff_cols.append(col)
        if diff_cols:
            mismatches[rid] = diff_cols

    if mismatches:
        lines = ["Request metrics mismatch for the following request_ids:"]
        for rid in sorted(mismatches):
            cols_str = ", ".join(sorted(mismatches[rid]))
            lines.append(f"  {rid}: {cols_str}")
        raise AssertionError("\n".join(lines))
    else:
        print("✅ request_metrics rows match for all post-snapshot requests (within tolerance)")

def assert_metrics_equal(dir_a: str, dir_b: str, filename: str) -> None:
    if filename == "batch_metrics":
        assert_batch_metrics_equal(dir_a, dir_b)
        return
    if filename == "request_metrics":
        assert_request_metrics_equal(dir_a, dir_b)
        return

    path_a = Path(dir_a) / f"{filename}.csv"
    path_b = Path(dir_b) / f"{filename}.csv"
    assert path_a.exists(), f"{path_a} missing"
    assert path_b.exists(), f"{path_b} missing"

    df_a = load_and_normalize_csv(path_a)
    df_b = load_and_normalize_csv(path_b)

    pd.testing.assert_frame_equal(
        df_a, df_b,
        check_dtype=False,
        check_like=True,
        rtol=1e-5,
        atol=1e-6,
    )




# ---------------------------------------------------------------------------
# 4) Main test logic
# ---------------------------------------------------------------------------

def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_dir = run_baseline()
    snapshot_dir = run_with_mid_snapshot()

    # Compare request- and batch-level metrics
    assert_metrics_equal(baseline_dir, snapshot_dir, "request_metrics")
    assert_metrics_equal(baseline_dir, snapshot_dir, "batch_metrics")

    print("✅ request_metrics.csv match between baseline and snapshot+restore run")
    print("✅ batch_metrics.csv match between baseline and snapshot+restore run")
    print("Baseline metrics dir:", baseline_dir)
    print("Snapshot+restore metrics dir:", snapshot_dir)

if __name__ == "__main__":
    main()
