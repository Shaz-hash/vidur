# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import os
from dataclasses import replace

from .config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig
from .multiProcessUtils import run_parallel_self_improvement as run_local_self_improvement
from .Network import run_parallel_self_improvement as run_network_self_improvement
from .Network import run_worker_agent


def main() -> None:
    cfg: MultipleProcessTrainingConfig = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
    env_role = str(os.environ.get("GV2_NETWORK_NODE_ROLE", "")).strip().lower()
    if env_role:
        cfg = replace(cfg, network=replace(cfg.network, node_role=env_role, enabled=(env_role != "local")))
    cfg.validate()
    if bool(cfg.network.enabled):
        if str(cfg.network.node_role) == "worker":
            run_worker_agent(cfg)
            return
        if str(cfg.network.node_role) == "head":
            run_network_self_improvement(cfg)
            return
    run_local_self_improvement(cfg)


if __name__ == "__main__":
    main()
