# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
alphaZero.py
run command :
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.Game_Versions.Game_Version2.alphaZero

For native :
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.alphaZero

Single entrypoint that owns configuration for:
- Vidur simulator config (CLI args passed to SimulationConfig)
- MCTS constraints + explore cfg
- DNN model
- MCTS-DNN logging paths
- Replay dataset writing (shards)
- Self-play run settings

For now: runs ONE root search (5000 sims) and writes ONE dataset sample.
Training hookup comes next.
"""

from __future__ import annotations


import csv
import math
import os
import random
import re
import time
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Keep CPU thread pools at 1 per process to avoid oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .config import DEFAULT_GAME_V2_CONFIG, GameVersion2Config
from vidur.config import SimulationConfig
from vidur.simulator import Simulator

from ...environment import VidurMCTSEnvironment
from .mctsDNN import VidurMCTS

# environment.py currently imports configs from launch_mcts_job.py
from ...launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions

from .DNN.models import AlphaZeroModel
from .DNN.replay_write import ReplayWriter, ReplayWriterConfig
from .DNN.selfPlay import SelfPlayRunner, SingleRootRun
from .DNN.dnn_spec import make_dnn_spec

from .DNN.replay_dataset import load_manifest, collate_mixed_samples
from .DNN.trainer import Trainer, TrainerConfig

from .virtual_environment import VirtualVidurMCTSEnvironment
from ...virtual_simulator import VirtualSimulator


# -----------------------------
# Helper functions
# -----------------------------

def _set_global_seeds(seed: int, *, torch_deterministic: bool) -> None:
    s = int(seed)
    random.seed(s)
    if np is not None:
        np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if bool(torch_deterministic):
        # Required by CUDA>=10.2 for deterministic CuBLAS paths.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    else:
        # Seeded but allow non-deterministic kernels (faster, avoids CuBLAS constraint errors).
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def _build_constraints_and_explore(cfg: "AlphaZeroConfig") -> tuple[MCTSConstraintConfig, MCTSExploreConfig]:
    gv2 = cfg.game_v2
    gv2.validate()
    legacy = gv2.legacy_mcts

    decode_slos = tuple(float(x) for x in (legacy.decode_slos or (50.0,)))
    slo_options = RequestSLOOptions(
        prefill_slos=(3.0,),   # compatibility placeholder (unused by current GV2 env path)
        decode_slos=decode_slos,
    )

    constraints = MCTSConstraintConfig(
        maximum_qps=int(gv2.timing.max_requests_per_launch_window),
        min_request_tokens=int(gv2.derived_min_request_tokens()),
        max_request_tokens=int(gv2.derived_max_request_tokens()),
        interval_request_size=int(gv2.derived_interval_request_size()),
        request_slo_options=slo_options,
        prefill_slowdown=float(legacy.prefill_slowdown),
        prefill_profile_path=str(legacy.prefill_profile_path),
    )

    search = gv2.mcts_search
    explore_cfg = MCTSExploreConfig()  # legacy fields unused in GV2 python path

    setattr(explore_cfg, "prior_value_mode", str(search.prior_value_mode))
    setattr(explore_cfg, "root_dirichlet_noise_enabled", bool(search.root_dirichlet_noise_enabled))
    setattr(explore_cfg, "root_dirichlet_alpha", float(search.root_dirichlet_alpha))
    setattr(explore_cfg, "root_dirichlet_epsilon", float(search.root_dirichlet_epsilon))

    setattr(explore_cfg, "discount_factor", float(search.discount_factor))
    if search.discount_time_denominator_sec is not None:
        setattr(
            explore_cfg,
            "discount_time_denominator_sec",
            float(search.discount_time_denominator_sec),
        )

    setattr(explore_cfg, "reward_knee", float(search.reward_knee))
    setattr(explore_cfg, "reward_max_penalty", float(search.reward_max_penalty))
    if search.reward_tail_alpha is not None:
        setattr(explore_cfg, "reward_tail_alpha", float(search.reward_tail_alpha))



    return constraints, explore_cfg


def _build_env_and_simulator(
    cfg: "AlphaZeroConfig",
    *,
    use_virtual_env: bool,
) -> tuple[object, object, MCTSConstraintConfig, MCTSExploreConfig]:
    sim_cfg = configure_simulation(cfg.sim.cli_args)
    setattr(sim_cfg.cluster_config.cache_config, "assume_infinite_kv", True)
    constraints, explore_cfg = _build_constraints_and_explore(cfg)
    if use_virtual_env:
        simulator = VirtualSimulator(sim_cfg, register_atexit=False)
        env = VirtualVidurMCTSEnvironment(
            base_simulator=simulator,
            constraints=constraints,
            explore_cfg=explore_cfg,
            game_v2_cfg=cfg.game_v2,
    )
    else:
        simulator = Simulator(sim_cfg, register_atexit=False)
        env = VidurMCTSEnvironment(
            base_simulator=simulator,
            constraints=constraints,
            explore_cfg=explore_cfg,
            game_v2_cfg=cfg.game_v2,

        )
    return simulator, env, constraints, explore_cfg




# -----------------------------
# Config groups
# -----------------------------

@dataclass(frozen=True)
class SimulationCLIGroup:
    """
    These are the SAME args you’d pass to:
      python -m vidur.simulator ...
    but we route them through SimulationConfig.create_from_cli_args().
    """
    cli_args: Sequence[str]



@dataclass(frozen=True)
class ModelGroup:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LoggingGroup:
    mcts_iter_log: str = "simulator_output/mcts_dnn_logs/mcts_iter.csv"
    mcts_root_log: str = "simulator_output/mcts_dnn_logs/mcts_root.csv"
    flush_every: int = 1


@dataclass(frozen=True)
class DatasetGroup:
    out_dir: str = "simulator_output/mcts_dnn_dataset/train"
    shard_size: int = 512


@dataclass(frozen=True)
class RunGroup:
    game_id: int = 0
    root_id: int = 0
    root_depth: int = 0
    root_player: str = "adversary"  # or "controller"
    iterations: int = 10
    feature_version: int = 1


@dataclass(frozen=True)
class AlphaZeroConfig:
    sim: SimulationCLIGroup
    model: ModelGroup
    logging: LoggingGroup
    dataset: DatasetGroup
    run: RunGroup
    game_v2: GameVersion2Config = dc_field(default_factory=lambda: DEFAULT_GAME_V2_CONFIG)
   
# -----------------------------
# Helpers
# -----------------------------

def configure_simulation(sim_args: Iterable[str]) -> SimulationConfig:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(sim_args)
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv

    # keep lightweight
    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False

    # MCTS adversary generates requests; keep generator quiet
    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]

    return cfg


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    # Keep PyTorch CPU execution single-threaded for predictable per-process scaling.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    use_virtual_env = True

    # ---- Edit values here (single source of truth) ----
    cfg = AlphaZeroConfig(
        sim=SimulationCLIGroup(
            cli_args=[
                "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
                "--replica_config_device", "a100",
                "--replica_config_network_device", "a100_dgx",
                "--cluster_config_num_replicas", "1",
                "--replica_config_tensor_parallel_size", "1",
                "--replica_config_num_pipeline_stages", "1",
                "--global_scheduler_config_type", "round_robin",
                "--replica_scheduler_config_type", "vllm_v1",
                "--vllm_v1_scheduler_config_batch_size_cap", "512",
                "--execution_time_predictor_config_type", "random_forest",
                "--random_forest_execution_time_predictor_config_prediction_max_tokens_per_request", "8192",
                "--random_forest_execution_time_predictor_config_prediction_max_batch_size", "256",
                "--random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size", "4096",
                "--no-snapshot_rng_state"
            ]
        ),
        game_v2=DEFAULT_GAME_V2_CONFIG,
        model=ModelGroup(
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        logging=LoggingGroup(
            mcts_iter_log="simulator_output/mcts_dnn_logs/mcts_iter.csv",
            mcts_root_log="simulator_output/mcts_dnn_logs/mcts_root.csv",
            flush_every=1,
        ),
        dataset=DatasetGroup(
            out_dir="simulator_output/mcts_dnn_dataset/train",
            shard_size=512,
        ),
        run=RunGroup(
            game_id=0,
            root_id=0,
            root_depth=0,
            root_player="adversary",
            iterations=4000,
            feature_version=1,
        ),
    )

    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )


    ## MODEL TRAINING PARMS FOR SELF-IMPROVEMENT LOOP:
    num_generations = 1
    roots_per_generation = 1
    adv_iterations_per_root = 4000
    cont_iterations_per_root = 4000
    # train_steps_per_generation = 200   
    max_batch_size = 256
    train_log_csv = Path("simulator_output/mcts_dnn_logs/train_metrics.csv")
    ckpt_dir = Path("simulator_output/mcts_dnn_checkpoints")


    # ---- Build simulator/env/mcts/model/writer ----
    simulator, env, _, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=use_virtual_env)

    gv2_dnn_spec = make_dnn_spec(cfg=getattr(env, "_gv2_cfg", None))

    model = AlphaZeroModel(
        spec=gv2_dnn_spec,
    ).to(torch.device(cfg.model.device))

    model.eval()
    run_model = model
    

    writer = ReplayWriter(
        ReplayWriterConfig(out_dir=Path(cfg.dataset.out_dir), shard_size=cfg.dataset.shard_size)
    )

    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        log_path=cfg.logging.mcts_iter_log,
        tree_log_path=cfg.logging.mcts_root_log,
        logger_flush_every=cfg.logging.flush_every,
        verbose=True,
        complete_log=True,
    )


    runner = SelfPlayRunner(
        env=env,
        mcts=mcts,
        model=run_model,
        writer=writer,
        device_for_features=torch.device("cpu"),
        game_v2_cfg=cfg.game_v2,
    )



    ## 

    # try:
        # runner.run_single_root(
        #     SingleRootRun(
        #         game_id=cfg.run.game_id,
        #         root_id=cfg.run.root_id,
        #         root_depth=cfg.run.root_depth,
        #         root_player=cfg.run.root_player,
        #         iterations=cfg.run.iterations,
        #         feature_version=cfg.run.feature_version,
        #     )
        # )

        # runner.run_n_roots(
        #     game_id=cfg.run.game_id,
        #     num_roots=10,                 # how many root positions to collect
        #     iterations_per_root=cfg.run.iterations,     # MCTS sims per root
        #     start_root_id=cfg.run.root_id,
        #     start_root_depth=cfg.run.root_depth,
        #     start_player=cfg.run.root_player,  # usually "adversary"
        #     feature_version=cfg.run.feature_version,
        # )
        # writer.close()

    # finally:
    #     mcts.close()

    try:

        # runner.run_single_root(
        #     SingleRootRun(
        #         game_id=cfg.run.game_id,
        #         root_id=cfg.run.root_id,
        #         root_depth=cfg.run.root_depth,
        #         root_player=cfg.run.root_player,
        #         iterations=cfg.run.iterations,
        #         feature_version=cfg.run.feature_version,
        #     )
        # )


        runner.run_n_roots(
            game_id=cfg.run.game_id,
            num_roots=1,
            adv_iterations_per_root=int(cfg.run.iterations),
            cont_iterations_per_root=int(cfg.run.iterations),
            max_batch_size=int(max_batch_size),
            start_root_id=cfg.run.root_id,
            start_root_depth=int(cfg.run.root_depth),
            start_player=str(cfg.run.root_player),
            feature_version=cfg.run.feature_version,
        )


        # selfImprovementPolicy(
        #     cfg=cfg,
        #     env=env,
        #     mcts=mcts,
        #     model=model,
        #     num_generations=num_generations,
        #     roots_per_generation=roots_per_generation,
        #     adv_iterations_per_root=adv_iterations_per_root,
        #     cont_iterations_per_root=cont_iterations_per_root,
        #     history_nontrivial_hops=history_nontrivial_hops,
        #     max_batch_size=max_batch_size,
        #     train_steps_per_generation=train_steps_per_generation,
        #     ckpt_dir=ckpt_dir,
        #     train_log_csv=train_log_csv,
        #     device_for_features=torch.device("cpu"),
        # )
    finally:
        writer.close()
        mcts.close()
        



if __name__ == "__main__":
    main()
