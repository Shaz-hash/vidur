#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import psutil

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from .run_mcts import configure_simulation # you already have this in run_mcts.py


def rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def main(argv=None) -> None:
    print(f"[MEM] start: {rss_mb():.1f} MB")

    # reuse same CLI args you would pass to run_mcts.py after the MCTS flags
    sim_cfg = configure_simulation(argv or [])
    et_cfg = sim_cfg.execution_time_predictor_config
    et_cfg.prediction_max_tokens_per_request = 8192
    et_cfg.prediction_max_batch_size = 64
    print(f"[MEM] after configure_simulation: {rss_mb():.1f} MB")

    sim = Simulator(sim_cfg, register_atexit=False)
    print(f"[MEM] after Simulator init: {rss_mb():.1f} MB")

    snap = sim.snapshot_state()
    print(f"[MEM] after snapshot_state: {rss_mb():.1f} MB")

    sim2 = Simulator(sim_cfg, register_atexit=False,
                     execution_time_predictor=sim._execution_time_predictor)
    sim2.restore_state(snap)
    print(f"[MEM] after restore_state into sim2: {rss_mb():.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
