#!/usr/bin/env python3

# # (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
"""

python -m vidur.mcts.profile_sim_memory \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_device h100 \
  --replica_config_network_device h100_dgx \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --global_scheduler_config_type round_robin \
  --replica_scheduler_config_type vllm_v1 \


"""

import os
import sys
from pathlib import Path

import psutil

import time 

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from .run_mcts import configure_simulation # you already have this in run_mcts.py


def rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def main(argv=None) -> None:
    print(f"[MEM] start: {rss_mb():.1f} MB")

    t0 = time.perf_counter()

    # reuse same CLI args you would pass to run_mcts.py after the MCTS flags
    sim_cfg = configure_simulation(argv or [])
    et_cfg = sim_cfg.execution_time_predictor_config
    et_cfg.prediction_max_tokens_per_request = 262144
    et_cfg.prediction_max_batch_size = 512
    assume_infinite_kv = True
    setattr(sim_cfg.cluster_config.cache_config, "assume_infinite_kv", bool(assume_infinite_kv))
    print(f"[MEM] after configure_simulation: {rss_mb():.1f} MB")

    sim = Simulator(sim_cfg, register_atexit=False)
    t1 = time.perf_counter()

    print(f"[MEM] after Simulator init: {rss_mb():.1f} MB")

    snap = sim.snapshot_state()
    t2 = time.perf_counter()
    print(f"[MEM] after snapshot_state: {rss_mb():.1f} MB")

    sim2 = Simulator(sim_cfg, register_atexit=False,
                     execution_time_predictor=sim._execution_time_predictor)
    
    t3 = time.perf_counter()
    sim2.restore_state(snap)
    t4 = time.perf_counter()


    print(
            f"[PROFILE] KV RESULTS : \n"
            f"Snapshot={t2 - t1:.6f}s\n"
            f"Snapshot={t3 - t2:.6f}s\n"
            f"Restore={t4 - t3:.6f}s\n"
        )


    print(f"[MEM] after restore_state into sim2: {rss_mb():.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
