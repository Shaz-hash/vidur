"""
INFINITE_KV_CACHE_Manager.py

Integration tests for the "InfiniteKVCacheManager" fast-path.

Goals:
1) Finite KV vs Infinite KV:
   Run the SAME deterministic workload twice:
     - once with normal (finite) KV manager
     - once with InfiniteKVCacheManager
   and compare request/batch metrics for equivalence (within tolerance).

2) Snapshot/Restore correctness (Infinite KV):
   Run a baseline Infinite-KV simulation to completion, then run a second
   Infinite-KV simulation where we snapshot mid-run, restore into a new
   Simulator, and finish. Compare final per-request logical state.

How to run:
  python3 -m vidur.mcts.tests.INFINITE_KV_CACHE_Manager

IMPORTANT:
  Your vllm_v1_replica_scheduler must select InfiniteKVCacheManager via a config
  toggle (recommended):
      use_infinite = getattr(self._cache_config, "assume_infinite_kv", False)
  If your scheduler hardcodes use_infinite=True, this test cannot run the
  "finite" baseline.
"""

from __future__ import annotations

import json
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds

from vidur.entities import Batch, BatchStage, Request, Replica
from vidur.entities.execution_time import ExecutionTime
from vidur.events import BaseEvent

from vidur.types.execution_time_predictor_cache_mode import ExecutionTimePredictorCacheMode
from pathlib import Path


# ----------------------------
# Test knobs (edit as needed)
# ----------------------------

OUTPUT_ROOT = Path("simulator_output/infinite_kv_cache_tests")

# Workload
NUM_REQUESTS = 200
QPS = 2.0  # uniform arrivals => deterministic
PREFILL_TOKENS = 2048
DECODE_TOKENS = 256

# Scheduler / KV
CHUNK_SIZE = 512
BATCH_SIZE_CAP = 256
CACHE_BLOCK_SIZE = 16
CACHE_NUM_BLOCKS = 4096  # pick high enough to avoid preemption in "finite" run

# Snapshot point (count completed batches before snapshot)
SNAPSHOT_COMPLETED_BATCHES = 8

# Comparison tolerances
RTOL = 1e-5
ATOL = 1e-6


# ----------------------------
# Helpers
# ----------------------------

def reset_entity_ids() -> None:
    """Reset global ID counters so two runs have identical Request/Batch IDs."""
    Request._id = Batch._id = BatchStage._id = ExecutionTime._id = Replica._id = -1
    BaseEvent._id = 0


def _load_and_sort_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    # stable sort by all columns to ignore row order differences
    cols = sorted(df.columns.tolist())
    return df[cols].sort_values(by=cols).reset_index(drop=True)


def assert_csv_equal(path_a: Path, path_b: Path) -> None:
    assert path_a.exists(), f"missing: {path_a}"
    assert path_b.exists(), f"missing: {path_b}"
    df_a = _load_and_sort_csv(path_a)
    df_b = _load_and_sort_csv(path_b)
    pd.testing.assert_frame_equal(df_a, df_b, check_dtype=False, rtol=RTOL, atol=ATOL)


def _iter_requests(sim: Simulator):
    # Replica schedulers own all requests after assignment.
    for replica_scheduler in sim._scheduler._replica_schedulers.values():
        if hasattr(replica_scheduler, "_requests"):
            for req in replica_scheduler._requests.values():
                yield req

    # Also include anything still in global queue (should be empty at end).
    for req in getattr(sim._scheduler, "_request_queue", []):
        yield req


def collect_request_states(sim: Simulator) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for req in _iter_requests(sim):
        out[int(req.id)] = req.snapshot_state()
    return out


def _cmp_any(a: Any, b: Any, path: str) -> List[str]:
    diffs: List[str] = []

    # float tolerance
    if isinstance(a, float) and isinstance(b, float):
        if not (abs(a - b) <= ATOL or abs(a - b) <= RTOL * max(abs(a), abs(b), 1.0)):
            diffs.append(f"{path}: {a} != {b}")
        return diffs

    # primitives
    if isinstance(a, (int, str, bool, type(None))) or isinstance(b, (int, str, bool, type(None))):
        if a != b:
            diffs.append(f"{path}: {a} != {b}")
        return diffs

    # lists/tuples
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(_cmp_any(x, y, f"{path}[{i}]"))
        return diffs

    # dicts
    if isinstance(a, dict) and isinstance(b, dict):
        ka = set(a.keys())
        kb = set(b.keys())
        if ka != kb:
            diffs.append(f"{path}: keys differ (a-b={sorted(ka-kb)[:5]}, b-a={sorted(kb-ka)[:5]})")
            return diffs
        for k in ka:
            diffs.extend(_cmp_any(a[k], b[k], f"{path}.{k}"))
        return diffs

    # fallback
    if a != b:
        diffs.append(f"{path}: {a} != {b}")
    return diffs


def assert_request_states_equal(a: Dict[int, dict], b: Dict[int, dict]) -> None:
    ida = set(a.keys())
    idb = set(b.keys())
    if ida != idb:
        raise AssertionError(
            "Request id sets differ:\n"
            f"  only_in_a={sorted(ida-idb)[:10]}\n"
            f"  only_in_b={sorted(idb-ida)[:10]}"
        )

    all_diffs: List[str] = []
    for rid in sorted(ida):
        diffs = _cmp_any(a[rid], b[rid], f"request[{rid}]")
        if diffs:
            all_diffs.extend(diffs[:25])  # cap spam
            if len(all_diffs) >= 200:
                break

    if all_diffs:
        raise AssertionError("Request state mismatch (showing first diffs):\n" + "\n".join(all_diffs))


def make_config(*, out_dir: Path, assume_infinite_kv: bool) -> SimulationConfig:
    """
    Build a deterministic config:
    - synthetic generator + fixed length
    - uniform inter-arrival times
    """
    args = [
        "INFINITE_KV_CACHE_Manager.py",
        "--time_limit",
        "100000",
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
        "--global_scheduler_config_type",
        "round_robin",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--vllm_v1_scheduler_config_chunk_size",
        str(CHUNK_SIZE),
        "--vllm_v1_scheduler_config_batch_size_cap",
        str(BATCH_SIZE_CAP),
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        str(NUM_REQUESTS),
        "--length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        str(PREFILL_TOKENS),
        "--fixed_request_length_generator_config_decode_tokens",
        str(DECODE_TOKENS),
        "--interval_generator_config_type",
        "uniform",
        "--uniform_request_interval_generator_config_qps",
        str(QPS),
        "--cache_config_block_size",
        str(CACHE_BLOCK_SIZE),
        "--cache_config_num_blocks",
        str(CACHE_NUM_BLOCKS),
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--metrics_config_keep_individual_batch_metrics",
    ]

    old_argv = sys.argv
    sys.argv = args
    try:
        cfg = SimulationConfig.create_from_cli_args()
        cfg.execution_time_predictor_config.cache_dir = str(Path(__file__).resolve().parents[3] / "cache")

        # If you want to guarantee it NEVER trains:
        cfg.execution_time_predictor_config.cache_mode = ExecutionTimePredictorCacheMode.REQUIRE_CACHE
    finally:
        sys.argv = old_argv

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_config.output_dir = str(out_dir)
    cfg.metrics_config.save_table_to_wandb = False
    cfg.metrics_config.store_plots = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False

    # # Execution-time predictor: reuse cached models/predictions + keep it small
    # et = cfg.execution_time_predictor_config
    # et.cache_dir = str(Path("simulator_output/execution_time_predictor_cache"))
    # et.cache_mode = "use_cache"          # or "require_cache" after you warm it once
    # et.prediction_max_tokens_per_request = 8192
    # et.prediction_max_batch_size = 128


    # Toggle InfiniteKVCacheManager (scheduler should use getattr(...)).
    setattr(cfg.cluster_config.cache_config, "assume_infinite_kv", bool(assume_infinite_kv))

    return cfg


def run_full(cfg: SimulationConfig) -> Tuple[Simulator, Path]:
    set_seeds(cfg.seed)
    reset_entity_ids()
    sim = Simulator(cfg, register_atexit=False)
    _assert_kv_manager_selection(
        sim,
        bool(getattr(cfg.cluster_config.cache_config, "assume_infinite_kv", False)),
    )
    sim.run()
    sim._write_output()
    return sim, Path(cfg.metrics_config.output_dir)


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
        sim._event_queue or sim._request_generator.get_next_request_arrival_time() is not None
    ):
        next_event_time = sim._event_queue[0]._time if sim._event_queue else None
        next_arrival = sim._request_generator.get_next_request_arrival_time()

        # Inject arrivals just like the main run loop
        if (next_arrival is not None) and (
            next_event_time is None or next_arrival <= next_event_time
        ):
            sim._add_event(RequestArrivalEvent(next_arrival, sim._request_generator.get_next_request()))
            continue

        event = sim._event_queue[0]
        heapq.heappop(sim._event_queue)
        sim._set_time(event._time)
        new_events = event.handle_event(sim._scheduler, sim._cluster_metric_store)
        sim._add_events(new_events)

        if isinstance(event, BatchEndEvent):
            completed += 1

def backfill_replica_request_arrivals(sim: Simulator) -> None:
    """
    Recreate per-replica 'on_request_arrival' bookkeeping after restore,
    for requests that were already scheduled before the snapshot time.

    This fixes NaNs like request_arrived_at / inter_arrival_delay in request_metrics.csv.
    """
    cms = sim._cluster_metric_store

    seen: set[int] = set()
    scheduled: list[Request] = []

    for replica_scheduler in sim._scheduler._replica_schedulers.values():
        for req in getattr(replica_scheduler, "_requests", {}).values():
            rid = int(req.id)
            if rid in seen:
                continue
            seen.add(rid)

            # only backfill if it was already scheduled at/before snapshot time
            if not getattr(req, "_scheduled", False):
                continue
            if float(getattr(req, "_scheduled_at", 0.0)) > float(sim._time):
                continue

            # must have a replica_id to map to the right ReplicaMetricsStore
            if getattr(req, "_replica_id", None) is None:
                continue

            scheduled.append(req)

    # ensure deterministic ordering so inter-arrival-delay is reproduced
    scheduled.sort(key=lambda r: (float(getattr(r, "_scheduled_at", 0.0)), int(r.id)))

    for req in scheduled:
        store = cms._get_replica_metrics_store(req.replica_id)
        store.on_request_arrival(req)


def run_with_mid_snapshot(cfg: SimulationConfig, *, snapshot_batches: int) -> Tuple[Simulator, Path]:
    """
    Run up to `snapshot_batches` completed batches, snapshot, restore into a new
    Simulator, then finish.
    """
    set_seeds(cfg.seed)
    reset_entity_ids()
    print("_here2!")
    seed_sim = Simulator(cfg, register_atexit=False)
    _assert_kv_manager_selection(
        seed_sim,
        bool(getattr(cfg.cluster_config.cache_config, "assume_infinite_kv", False)),
    )
    print("_here!")
    step_until_completed_batches(seed_sim, snapshot_batches)
    snapshot = seed_sim.snapshot_state()
    del seed_sim
    import gc; gc.collect()

    # Keep replica ids stable across "fresh Simulator" creations in-process.
    Replica._id = -1

    sim_restored = Simulator(cfg, register_atexit=False)
    sim_restored.restore_state(snapshot)
    print("here2!")
    backfill_replica_request_arrivals(sim_restored)
    print("here222!")
    _assert_kv_manager_selection(
        sim_restored,
        bool(getattr(cfg.cluster_config.cache_config, "assume_infinite_kv", False)),
    )
    sim_restored.run()
    sim_restored._write_output()
    return sim_restored, Path(cfg.metrics_config.output_dir)



def _assert_kv_manager_selection(sim: Simulator, expect_infinite: bool) -> None:
    """
    Guardrail: ensure the scheduler actually respected cfg.cache_config.assume_infinite_kv.
    If your scheduler hardcodes the choice, the A/B test becomes meaningless.
    """
    names: List[str] = []
    for replica_scheduler in sim._scheduler._replica_schedulers.values():
        kvm = getattr(replica_scheduler, "_kv_cache_manager", None)
        if kvm is None:
            continue
        names.append(type(kvm).__name__)

    if not names:
        raise AssertionError("Could not find any replica _kv_cache_manager to validate.")

    any_infinite = any("InfiniteKVCacheManager" in n for n in names)
    if expect_infinite and not any_infinite:
        raise AssertionError(
            "Config expected InfiniteKVCacheManager, but replica schedulers did not use it.\n"
            f"Found KV manager types: {names}\n"
            "Fix: in vllm_v1_replica_scheduler.py, set:\n"
            "  use_infinite = getattr(self._cache_config, 'assume_infinite_kv', False)"
        )
    if (not expect_infinite) and any_infinite:
        raise AssertionError(
            "Config expected finite ReplicaKVCacheManager, but replica schedulers used InfiniteKVCacheManager.\n"
            f"Found KV manager types: {names}\n"
            "Fix: in vllm_v1_replica_scheduler.py, do NOT hardcode use_infinite=True.\n"
            "Use cfg.cache_config.assume_infinite_kv instead."
        )


def assert_batch_metrics_equal_after_snapshot(
    full_run_csv: Path,
    restored_csv: Path,
    *,
    skipped_batches: int,
    id_col: str = "Batch Id",
) -> None:
    df_full = pd.read_csv(full_run_csv)
    df_rest = pd.read_csv(restored_csv)

    df_full = df_full[df_full[id_col] >= skipped_batches].copy()
    df_full[id_col] = df_full[id_col] - skipped_batches

    cols = sorted(df_full.columns.tolist())
    df_full = df_full[cols].sort_values(by=cols).reset_index(drop=True)
    df_rest = df_rest[cols].sort_values(by=cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(df_full, df_rest, check_dtype=False, rtol=RTOL, atol=ATOL)


# ----------------------------
# Main test runner
# ----------------------------

def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Full-run equivalence: finite vs infinite
    cfg_finite = make_config(out_dir=OUTPUT_ROOT / "finite_kv", assume_infinite_kv=False)
    sim_finite, dir_finite = run_full(cfg_finite)
    req_finite = collect_request_states(sim_finite)

    cfg_inf = make_config(out_dir=OUTPUT_ROOT / "infinite_kv", assume_infinite_kv=True)
    sim_inf, dir_inf = run_full(cfg_inf)
    req_inf = collect_request_states(sim_inf)

    # Compare logical per-request state first (strong invariant).
    assert_request_states_equal(req_finite, req_inf)

    # Compare metrics CSVs as a secondary check (nice for debugging).
    assert_csv_equal(dir_finite / "request_metrics.csv", dir_inf / "request_metrics.csv")
    assert_csv_equal(dir_finite / "batch_metrics.csv", dir_inf / "batch_metrics.csv")

    print("✅ Finite vs Infinite KV: request/batch metrics match")
    print("Finite metrics dir:", dir_finite)
    print("Infinite metrics dir:", dir_inf)

    # 2) Snapshot/restore correctness for Infinite KV (compare final request state)
    cfg_inf_snap = make_config(out_dir=OUTPUT_ROOT / "infinite_kv_snapshot", assume_infinite_kv=True)
    sim_inf_restored, _ = run_with_mid_snapshot(cfg_inf_snap, snapshot_batches=SNAPSHOT_COMPLETED_BATCHES)
    print("here!")
    req_inf_restored = collect_request_states(sim_inf_restored)
    print("here1!")
    assert_request_states_equal(req_inf, req_inf_restored)
    print("here2!")
    # request metrics should match fully
    assert_csv_equal(dir_inf / "request_metrics.csv", (OUTPUT_ROOT / "infinite_kv_snapshot") / "request_metrics.csv")
    print("here3!")
    # batch metrics: restored run likely only contains rows after snapshot point
    assert_batch_metrics_equal_after_snapshot(
        dir_inf / "batch_metrics.csv",
        OUTPUT_ROOT / "infinite_kv_snapshot" / "batch_metrics.csv",
        skipped_batches=SNAPSHOT_COMPLETED_BATCHES,
    )


    print("✅ Infinite KV snapshot+restore: final per-request state matches baseline Infinite KV")


if __name__ == "__main__":
    main()
