# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
alphaZero.py
run command :
python3 -m vidur.mcts.alphaZero

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
import random
import re
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

from .environment import VidurMCTSEnvironment
from .mctsDNN import VidurMCTS

# environment.py currently imports configs from launch_mcts_job.py
from .launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions

from .DNN.models import AlphaZeroModel
from .DNN.replay_write import ReplayWriter, ReplayWriterConfig
from .DNN.selfPlay import SelfPlayRunner, SingleRootRun

from .DNN.replay_dataset import load_manifest, collate_mixed_samples
from .DNN.trainer import Trainer, TrainerConfig

from .virtual_environment import VirtualVidurMCTSEnvironment
from .virtual_simulator import VirtualSimulator



# ----- 
# FUNCTIONS for SELF IMPROVEMENT TRAINING LOOP\
# -----


TRAIN_LOG_FIELDS = [
    "time",
    "event",              # train/eval/load_ckpt
    "gen",
    "trainer_step",
    "dataset_dir",
    "num_samples",
    "num_controller",
    "num_adversary",
    "loss",
    "policy_loss",
    "value_loss",
    "controller_policy_loss",
    "controller_value_loss",
    "controller_count",
    "adversary_policy_loss",
    "adversary_value_loss",
    "adversary_count",
    "saved_best",
    "ckpt_path",
    "best_path",
    "resume_ckpt",
]


def _append_train_log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    full = {k: row.get(k, "") for k in TRAIN_LOG_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(full)


def _next_generation_index(dataset_base: Path) -> int:
    if not dataset_base.exists():
        return 0
    best = -1
    for p in dataset_base.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^gen_(\d+)$", p.name)
        if not m:
            continue
        best = max(best, int(m.group(1)))
    return best + 1


def _maybe_resume_best(
    *,
    trainer: Trainer,
    model: torch.nn.Module,
    ckpt_dir: Path,
    best_path: Path,
) -> Optional[Path]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Prefer best.pt
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=trainer.device)
        model.load_state_dict(ckpt["model_state"])
        # resume optimizer + step if possible (so training continues smoothly)
        try:
            trainer.opt.load_state_dict(ckpt["optimizer_state"])
            trainer.step = int(ckpt.get("step", 0))
            trainer.best_eval_loss = float(ckpt.get("best_eval_loss", math.inf))
        except Exception:
            # Still fine: weights loaded, optimizer reset
            pass
        return best_path

    # Fallback: latest ckpt_gen_*.pt if present
    ckpts = sorted(ckpt_dir.glob("ckpt_gen_*_step_*.pt"))
    if ckpts:
        last = max(ckpts, key=lambda p: p.stat().st_mtime)
        ckpt = torch.load(last, map_location=trainer.device)
        model.load_state_dict(ckpt["model_state"])
        try:
            trainer.opt.load_state_dict(ckpt["optimizer_state"])
            trainer.step = int(ckpt.get("step", 0))
            trainer.best_eval_loss = float(ckpt.get("best_eval_loss", math.inf))
        except Exception:
            pass
        return last

    return None


def selfImprovementPolicy(
    *,
    cfg: "AlphaZeroConfig",
    env: VidurMCTSEnvironment | VirtualVidurMCTSEnvironment,
    mcts: VidurMCTS,
    model: AlphaZeroModel,
    num_generations: int,
    roots_per_generation: int,
    # iterations_per_root: int,
    adv_iterations_per_root: int,
    cont_iterations_per_root: int,
    history_nontrivial_hops: int = 0,
    max_batch_size: int = 72,
    train_steps_per_generation: int,
    ckpt_dir: Path,
    train_log_csv: Path,
    device_for_features: torch.device = torch.device("cpu"),
) -> None:
    best_path = ckpt_dir / "best.pt"

    trainer = Trainer(
        model=model,
        cfg=TrainerConfig(
            lr=1e-3,
            weight_decay=1e-4,
            policy_weight=1.0,
            value_weight=1.0,
            grad_clip_norm=5.0,
        ),
        device=torch.device(cfg.model.device),
    )

    resume_ckpt = _maybe_resume_best(trainer=trainer, model=trainer.model, ckpt_dir=ckpt_dir, best_path=best_path)
    _append_train_log_row(
        train_log_csv,
        {
            "time": time.time(),
            "event": "load_ckpt",
            "gen": "",
            "trainer_step": int(trainer.step),
            "dataset_dir": "",
            "num_samples": "",
            "num_controller": "",
            "num_adversary": "",
            "loss": "",
            "policy_loss": "",
            "value_loss": "",
            "controller_policy_loss": "",
            "controller_value_loss": "",
            "controller_count": "",
            "adversary_policy_loss": "",
            "adversary_value_loss": "",
            "adversary_count": "",
            "saved_best": "",
            "ckpt_path": "",
            "best_path": str(best_path),
            "resume_ckpt": str(resume_ckpt) if resume_ckpt else "",
        },
    )

    dataset_base = Path(cfg.dataset.out_dir)
    gen0 = _next_generation_index(dataset_base)

    for j in range(int(num_generations)):
        gen = gen0 + j
        gen_dataset_dir = dataset_base / f"gen_{gen:06d}"

        gen_writer = ReplayWriter(
            ReplayWriterConfig(out_dir=gen_dataset_dir, shard_size=cfg.dataset.shard_size)
        )
        runner = SelfPlayRunner(
            env=env,
            mcts=mcts,
            model=trainer.model,   # always use the current weights
            writer=gen_writer,
            device_for_features=device_for_features,
        )

        # 1) self-play: generate roots_per_generation samples
        trainer.model.eval()
        runner.run_n_roots(
            game_id=cfg.run.game_id + gen,
            num_roots=roots_per_generation,
            adv_iterations_per_root=adv_iterations_per_root,
            cont_iterations_per_root=cont_iterations_per_root,
            max_batch_size=max_batch_size,
            start_root_id=0,
            start_root_depth=0,
            start_player="adversary",
            history_nontrivial_hops=history_nontrivial_hops,
            feature_version=cfg.run.feature_version,
        )
        gen_writer.close()

        # 2) load ALL samples from this generation (train on exactly these)
        samples = []
        for entry in load_manifest(gen_dataset_dir / "manifest.jsonl"):
            samples.extend(torch.load(entry.path, map_location="cpu"))
        if not samples:
            raise RuntimeError(f"No samples written for gen={gen} in {gen_dataset_dir}")

        num_controller = sum(1 for s in samples if s.get("player") == "controller")
        num_adversary = sum(1 for s in samples if s.get("player") == "adversary")

        eval_batch = collate_mixed_samples(samples, device=trainer.device)

        # 3) train multiple optimizer steps on this batch (few samples => multiple epochs)
        # for _ in range(int(train_steps_per_generation)):
        #     train_metrics = trainer.train_step(batch)
        #     _append_train_log_row(
        #         train_log_csv,
        #         {
        #             "time": time.time(),
        #             "event": "train",
        #             "gen": gen,
        #             "trainer_step": int(trainer.step),
        #             "dataset_dir": str(gen_dataset_dir),
        #             "num_samples": int(len(samples)),
        #             "num_controller": int(num_controller),
        #             "num_adversary": int(num_adversary),
        #             **train_metrics,
        #             "saved_best": "",
        #             "ckpt_path": "",
        #             "best_path": str(best_path),
        #             "resume_ckpt": str(resume_ckpt) if resume_ckpt else "",
        #         },
        #     )

        # Eval on full latest-generation dataset (stable metric)
        eval_batch = collate_mixed_samples(samples, device=trainer.device)

        # Train with random minibatches from latest generation (replay-style)
        rng = random.Random(1000 + int(gen))   # deterministic per gen; change seed if you want
        train_minibatch_size = 32

        for _ in range(int(train_steps_per_generation)):
            minibatch_samples = rng.choices(samples, k=train_minibatch_size)  # with replacement
            train_batch = collate_mixed_samples(minibatch_samples, device=trainer.device)

            train_metrics = trainer.train_step(train_batch)
            _append_train_log_row(
                train_log_csv,
                {
                    "time": time.time(),
                    "event": "train",
                    "gen": gen,
                    "trainer_step": int(trainer.step),
                    "dataset_dir": str(gen_dataset_dir),
                    "num_samples": int(len(samples)),
                    "num_controller": int(num_controller),
                    "num_adversary": int(num_adversary),
                    **train_metrics,
                    "saved_best": "",
                    "ckpt_path": "",
                    "best_path": str(best_path),
                    "resume_ckpt": str(resume_ckpt) if resume_ckpt else "",
                },
            )


        # 4) eval + checkpoint
        eval_metrics = trainer.eval_step(eval_batch)
        saved_best = trainer.maybe_save_best(eval_metrics["loss"], best_path)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ckpt_gen_{gen:06d}_step_{trainer.step:06d}.pt"
        trainer.save_checkpoint(ckpt_path)

        _append_train_log_row(
            train_log_csv,
            {
                "time": time.time(),
                "event": "eval",
                "gen": gen,
                "trainer_step": int(trainer.step),
                "dataset_dir": str(gen_dataset_dir),
                "num_samples": int(len(samples)),
                "num_controller": int(num_controller),
                "num_adversary": int(num_adversary),
                **eval_metrics,
                "saved_best": bool(saved_best),
                "ckpt_path": str(ckpt_path),
                "best_path": str(best_path),
                "resume_ckpt": str(resume_ckpt) if resume_ckpt else "",
            },
        )



# -----------------------------
# Helper functions
# -----------------------------
def _append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)



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
class MCTSConstraintsGroup:
    maximum_qps: int = 10
    interval_request_size: int = 512
    min_request_tokens: int = 512
    max_request_tokens: int = 3072
    prefill_profile_path: str = "simulator_output/prefill_profile.csv"
    prefill_slowdown: float = 3.0

    prefill_slos: Sequence[float] = (3.0,)
    decode_slos: Sequence[float] = (50.0,)


@dataclass(frozen=True)
class MCTSExploreGroup:
    simulation_depth: int = 2
    simulation_random_tries: int = 1
    exploration_constant: float = 1.7
    max_branching: int = 10
    controller_budget_combs: int = 10
    controller_min_prior_threshold : float = 0.01
    adversary_min_prior_threshold : float = 0.1




@dataclass(frozen=True)
class ModelGroup:
    num_actions_controller: int = 24
    num_actions_adversary: int = 6
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
    constraints: MCTSConstraintsGroup
    explore: MCTSExploreGroup
    model: ModelGroup
    logging: LoggingGroup
    dataset: DatasetGroup
    run: RunGroup


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

    use_virtual_env = True

    # ---- Edit values here (single source of truth) ----
    cfg = AlphaZeroConfig(
        sim=SimulationCLIGroup(
            cli_args=[
                "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
                "--replica_config_device", "h100",
                "--replica_config_network_device", "h100_dgx",
                "--cluster_config_num_replicas", "1",
                "--replica_config_tensor_parallel_size", "1",
                "--replica_config_num_pipeline_stages", "1",
                "--global_scheduler_config_type", "round_robin",
                "--replica_scheduler_config_type", "vllm_v1",
                "--vllm_v1_scheduler_config_batch_size_cap", "512",
                "--no-snapshot_rng_state"
            ]
        ),
        constraints=MCTSConstraintsGroup(
            maximum_qps=5,
            interval_request_size=512,
            min_request_tokens=512,
            max_request_tokens=3072,
            prefill_profile_path="simulator_output/prefill_profile.csv",
            prefill_slowdown=3.0,
            prefill_slos=(3.0,),
            decode_slos=(50.0,),
        ),
        explore=MCTSExploreGroup(
            simulation_depth=2,
            simulation_random_tries=1,
            exploration_constant=1.7,
            max_branching=10,
            controller_budget_combs=10,
            controller_min_prior_threshold=0.01,
            adversary_min_prior_threshold=0.1,
        ),
        model=ModelGroup(
            num_actions_controller=24,
            num_actions_adversary=6,
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
            iterations=8000,
            feature_version=1,
        ),
    )


    ## MODEL TRAINING PARMS FOR SELF-IMPROVEMENT LOOP:
    num_generations = 1
    history_nontrivial_hops = 0
    roots_per_generation = 1
    adv_iterations_per_root = 8000
    cont_iterations_per_root = 8000
    train_steps_per_generation = 200   
    max_batch_size = 256
    train_log_csv = Path("simulator_output/mcts_dnn_logs/train_metrics.csv")
    ckpt_dir = Path("simulator_output/mcts_dnn_checkpoints")


    # ---- Build simulator/env/mcts/model/writer ----
    sim_cfg = configure_simulation(cfg.sim.cli_args)
    setattr(sim_cfg.cluster_config.cache_config, "assume_infinite_kv", True)
    # simulator = Simulator(sim_cfg, register_atexit=False)
    # print("Simulator initialized.")
    # print(simulator._execution_time_predictor.to_dict())

    if use_virtual_env:
        simulator = VirtualSimulator(sim_cfg, register_atexit=False)
    else:
        simulator = Simulator(sim_cfg, register_atexit=False)


    slo_options = RequestSLOOptions(
        prefill_slos=tuple(cfg.constraints.prefill_slos),
        decode_slos=tuple(cfg.constraints.decode_slos),
    )

    constraints = MCTSConstraintConfig(
        maximum_qps=cfg.constraints.maximum_qps,
        min_request_tokens=cfg.constraints.min_request_tokens,
        max_request_tokens=cfg.constraints.max_request_tokens,
        interval_request_size=cfg.constraints.interval_request_size,
        request_slo_options=slo_options,
        prefill_slowdown=cfg.constraints.prefill_slowdown,
        prefill_profile_path=cfg.constraints.prefill_profile_path,
    )

    explore_cfg = MCTSExploreConfig(
        simulation_depth=cfg.explore.simulation_depth,
        simulation_random_tries=cfg.explore.simulation_random_tries,
        exploration_constant=cfg.explore.exploration_constant,
        max_branching=cfg.explore.max_branching,
        controller_budget_combs=cfg.explore.controller_budget_combs,
    )

    setattr(explore_cfg, "controller_min_prior_threshold", float(cfg.explore.controller_min_prior_threshold))
    setattr(explore_cfg, "adversary_min_prior_threshold", float(cfg.explore.adversary_min_prior_threshold))


    if use_virtual_env:
        env = VirtualVidurMCTSEnvironment(
            base_simulator=simulator,
            constraints=constraints,
            explore_cfg=explore_cfg,
        )
    else:
        env = VidurMCTSEnvironment(
            base_simulator=simulator,
            constraints=constraints,
            explore_cfg=explore_cfg,
        )

    model = AlphaZeroModel(
        num_actions_controller=cfg.model.num_actions_controller,
        num_actions_adversary=cfg.model.num_actions_adversary,
    ).to(torch.device(cfg.model.device))
    model.eval()

    writer = ReplayWriter(
        ReplayWriterConfig(out_dir=Path(cfg.dataset.out_dir), shard_size=cfg.dataset.shard_size)
    )

    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        log_path=cfg.logging.mcts_iter_log,
        tree_log_path=cfg.logging.mcts_root_log,
        logger_flush_every=cfg.logging.flush_every,
    )

    runner = SelfPlayRunner(
        env=env,
        mcts=mcts,
        model=model,
        writer=writer,
        device_for_features=torch.device("cpu"),
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

        runner.run_single_root(
            SingleRootRun(
                game_id=cfg.run.game_id,
                root_id=cfg.run.root_id,
                root_depth=cfg.run.root_depth,
                root_player=cfg.run.root_player,
                iterations=cfg.run.iterations,
                feature_version=cfg.run.feature_version,
            )
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
        mcts.close()



if __name__ == "__main__":
    main()


