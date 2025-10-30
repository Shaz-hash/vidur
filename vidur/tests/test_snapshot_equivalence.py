# tests/test_snapshot_equivalence.py
import os
import sys
from typing import List, Dict, Any, Tuple

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds

# import whatever you use to build SimulationConfig in vidur.main
# from vidur.config import SimulationConfig, ...  # <- your usual imports

# --- 1) Build the same config you pass via CLI ---------------------------
def make_config():
    args = [
        "test_snapshot_equivalence.py",  # dummy argv[0]
        "--time_limit", "10800",
        "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device", "h100",
        "--replica_config_network_device", "h100_dgx",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
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
        "--cache_config_enable_prefix_caching",          # turn ON
        "--metrics_config_write_json_trace",
        "--metrics_config_keep_individual_batch_metrics",
        "--metrics_config_write_metrics",
        "--metrics_config_store_operation_metrics"
    ]

    old_argv = sys.argv
    sys.argv = args
    try:
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = old_argv
    return cfg



# --- 2) A tiny stepper to advance exactly N events -----------------------
def step_events(sim: Simulator, n: int):
    """
    Advance the simulator by exactly n event-handles (or until idle).
    Mirrors the main run loop for a single iteration to keep semantics consistent.
    """
    import heapq
    steps = 0
    while steps < n and (sim._event_queue or sim._request_generator.get_next_request_arrival_time() is not None):
        next_event_time = sim._event_queue[0]._time if sim._event_queue else None
        next_request_arrival_time = sim._request_generator.get_next_request_arrival_time()

        # inject arrival if it comes first
        if (next_request_arrival_time is not None) and (next_event_time is None or next_request_arrival_time <= next_event_time):
            from vidur.events import RequestArrivalEvent
            sim._add_event(RequestArrivalEvent(next_request_arrival_time, sim._request_generator.get_next_request()))
            continue

        # one normal event
        event = sim._event_queue[0]
        heapq.heappop(sim._event_queue)
        sim._set_time(event._time)
        new_events = event.handle_event(sim._scheduler, sim._cluster_metric_store)
        sim._add_events(new_events)

        if sim._config.metrics_config.write_json_trace:
            sim._event_trace.append(event.to_dict())

        if sim._config.metrics_config.enable_chrome_trace:
            ct = event.to_chrome_trace()
            if ct:
                sim._event_chrome_trace.append(ct)

        steps += 1

# --- 3) Helpers to compare runs -----------------------------------------
def extract_completion_times(trace: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    From event_trace dicts, map request_id -> completed_at (using RequestEndEvent).
    """
    out: Dict[int, float] = {}
    for ev in trace:
        if ev.get("type") == "RequestEndEvent":
            rid = int(ev["data"]["request_id"])
            t = float(ev["data"]["time"])
            out[rid] = t
    return out

def compare_event_traces(t1: List[Dict[str, Any]], t2: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if len(t1) != len(t2):
        return False, f"event_trace length mismatch: {len(t1)} vs {len(t2)}"

    for i, (a, b) in enumerate(zip(t1, t2)):
        # compare key fields; allow extra metadata differences if any
        if a.get("type") != b.get("type"):
            return False, f"[{i}] type mismatch: {a.get('type')} vs {b.get('type')}"
        if a.get("data") != b.get("data"):
            return False, f"[{i}] data mismatch: {a.get('data')} vs {b.get('data')}"
    return True, "OK"

# --- 4) The test itself --------------------------------------------------
def main():
    cfg = make_config()
    set_seeds(cfg.seed) 

    # sim0 = Simulator(cfg, register_atexit=False)
    # snap0 = sim0.snapshot_state()
    # sim0a = sim0.fork()
    # sim0b = Simulator(cfg, register_atexit=False); sim0b.restore_state(snap0)
    # sim0a.run(); sim0b.run()

    # ok, msg = compare_event_traces(sim0a._event_trace, sim0b._event_trace)
    # assert ok, f"event traces diverged: {msg}"
    # print("✅ Snapshot/restore deterministic replay: PASS")
    

    # A: start a simulation, run a few events, snapshot/fork mid-run
    sim = Simulator(cfg, register_atexit=False)

    # run a small prefix of events (tweak 200–2000 depending on QPS/trace)
    step_events(sim, n=500)
    

    # snapshot + fork
    snap = sim.snapshot_state()
    simA = sim.fork()  # uses the snapshot internally
    # For parity, restore another fresh instance from the same snapshot
    simB = Simulator(cfg, register_atexit=False)
    simB.restore_state(snap)

    # complete both runs
    simA.run()
    simB.run()


    # --- compare event traces (needs write_json_trace=True) ---
    tA = simA._event_trace
    tB = simB._event_trace
    ok, msg = compare_event_traces(tA, tB)
    assert ok, f"Event traces diverged: {msg}"

    # --- compare completion times ---
    cA = extract_completion_times(tA)
    cB = extract_completion_times(tB)
    assert cA == cB, f"Per-request completion times differ.\nA-only: {cA.keys()-cB.keys()}\nB-only: {cB.keys()-cA.keys()}"

    print("✅ Snapshot/restore deterministic replay: PASS")

    simA._write_output()
    simB._write_output()

    # mA = extract_batch_metrics_from_trace(simA._event_trace)
    # mB = extract_batch_metrics_from_trace(simB._event_trace)

    # write_batch_metrics_csv(mA, "tests/out/simA_batch_metrics.csv")
    # write_batch_metrics_csv(mB, "tests/out/simB_batch_metrics.csv")

    # (Optional) Branching sanity: inject one extra request into B then finish.
    # You can uncomment this to see divergence only from the extra request.
    # from vidur.entities.request import Request
    # extra = Request(arrived_at=simB._time, num_prefill_tokens=32, num_decode_tokens=16,
    #                 block_hash_ids=None, block_size=None, session_id=None)
    # simB._scheduler.add_request(extra)
    # simB.run()
    # # now traces must differ; that's expected

if __name__ == "__main__":
    main()
