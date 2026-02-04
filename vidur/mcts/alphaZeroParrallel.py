# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
alphaZeroParrallel.py
run command :
python3 -m vidur.mcts.alphaZeroParrallel

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

import multiprocessing as mp
import os 

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


def _selfplay_worker_main(
    worker_id: int,
    task_q: "mp.Queue",
    result_q: "mp.Queue",
    cfg: "AlphaZeroConfig",
    adv_iterations_per_root: int,
    cont_iterations_per_root: int,
    max_batch_size: int,
    history_nontrivial_hops: int,
) -> None:
    # Important: avoid CPU oversubscription when you run many processes
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Optional: pin to core (Linux)
    # try:
    #     os.sched_setaffinity(0, {worker_id})
    # except Exception:
    #     pass

    # Build simulator/env ONCE per process (big speed win vs rebuilding every gen)
    sim_cfg = configure_simulation(cfg.sim.cli_args)
    setattr(sim_cfg.cluster_config.cache_config, "assume_infinite_kv", True)
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

    env = VidurMCTSEnvironment(base_simulator=simulator, constraints=constraints, explore_cfg=explore_cfg)

    # Build model once per process; reload weights each generation
    model = AlphaZeroModel(
        num_actions_controller=cfg.model.num_actions_controller,
        num_actions_adversary=cfg.model.num_actions_adversary,
    ).to(torch.device("cpu"))
    model.eval()

    while True:
        task = task_q.get()
        if task is None:
            break

        # Unpack task
        gen = int(task["gen"])
        out_dir = Path(task["out_dir"])
        weights_path = Path(task["weights_path"])
        game_id = int(task["game_id"])
        start_root_id = int(task["start_root_id"])
        history_seed = int(task["history_seed"])
        roots = int(task["num_roots"])

        # Load frozen weights for this generation
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Per-worker writer
        writer = ReplayWriter(ReplayWriterConfig(out_dir=out_dir, shard_size=cfg.dataset.shard_size))

        # IMPORTANT: per-worker MCTS instance (don’t share across processes)
        # Also IMPORTANT: logs must be unique per worker, or disable by passing None.
        # If you want logs:
        #   iter_log = out_dir / "mcts_iter.csv"
        #   root_log = out_dir / "mcts_root.csv"
        # Else (faster, no contention):
        # iter_log = None
        # root_log = None

        # Put per-worker logs in the global logs dir (not inside dataset dir)
        logs_base = Path(cfg.logging.mcts_iter_log).parent  # simulator_output/mcts_dnn_logs
        logs_dir = logs_base / f"gen_{gen:06d}"
        logs_dir.mkdir(parents=True, exist_ok=True)

        iter_log = logs_dir / f"mcts_iter_p{worker_id:02d}.csv"
        root_log = logs_dir / f"mcts_root_p{worker_id:02d}.csv"



        mcts = VidurMCTS(env=env, explore_cfg=explore_cfg, log_path=iter_log, tree_log_path=root_log, logger_flush_every=cfg.logging.flush_every)
        runner = SelfPlayRunner(env=env, mcts=mcts, model=model, writer=writer, device_for_features=torch.device("cpu"))

        # Make history different per worker/gen by changing game_id and/or seed.
        # Right now run_n_roots only takes history_nontrivial_hops, so “seed” comes from game_id/root_id_for_logs.
        # If you want explicit control, add a `history_seed` arg to run_n_roots and forward it to HistoryRootGenerator.
        runner.run_n_roots(
            game_id=game_id,
            num_roots=roots,
            adv_iterations_per_root=adv_iterations_per_root,
            cont_iterations_per_root=cont_iterations_per_root,
            max_batch_size=max_batch_size,
            start_root_id=start_root_id,
            start_root_depth=0,
            start_player="adversary",
            history_nontrivial_hops=history_nontrivial_hops,
            feature_version=cfg.run.feature_version,
        )

        writer.close()
        mcts.close()

        result_q.put(
            {
                "worker_id": worker_id,
                "gen": gen,
                "out_dir": str(out_dir),
            }
        )










def selfImprovementPolicy(
    *,
    cfg: "AlphaZeroConfig",
    model: AlphaZeroModel,
    num_selfPlay_workers: int = 1,
    num_generations: int,
    roots_per_generation: int,
    # iterations_per_root: int,
    adv_iterations_per_root: int,
    cont_iterations_per_root: int,
    history_nontrivial_hops: int | Sequence[int] = 0,
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
    
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()

    def _hops_for_worker(wid: int) -> int:
        if isinstance(history_nontrivial_hops, int):
            return int(history_nontrivial_hops)
        hops_list = list(history_nontrivial_hops)
        if len(hops_list) != int(num_selfPlay_workers):
            raise ValueError(
                f"history_nontrivial_hops must have len == num_selfPlay_workers "
                f"({len(hops_list)} != {int(num_selfPlay_workers)})"
            )
        return int(hops_list[wid])


    workers = []
    for wid in range(int(num_selfPlay_workers)):
        p = ctx.Process(
            target=_selfplay_worker_main,
            args=(
                wid,
                task_q,
                result_q,
                cfg,
                adv_iterations_per_root,
                cont_iterations_per_root,
                max_batch_size,
                _hops_for_worker(wid),
            ),
        )
        p.start()
        workers.append(p)


    dataset_base = Path(cfg.dataset.out_dir)
    gen0 = _next_generation_index(dataset_base)

    for j in range(int(num_generations)):
        gen = gen0 + j
        gen_dataset_dir = dataset_base / f"gen_{gen:06d}"

        # 0) freeze current weights for self-play workers
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        weights_path = ckpt_dir / f"selfplay_weights_gen_{gen:06d}.pt"
        state = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
        torch.save({"model_state": state}, weights_path)

        # 1) dispatch tasks
        gen_dataset_dir.mkdir(parents=True, exist_ok=True)

        # split roots across workers
        per = int(math.ceil(roots_per_generation / float(num_selfPlay_workers)))
        tasks_sent = 0
        for wid in range(int(num_selfPlay_workers)):
            start = wid * per
            end = min(roots_per_generation, (wid + 1) * per)
            n_roots = max(0, end - start)
            if n_roots == 0:
                continue

            out_dir = gen_dataset_dir / f"proc_{wid:02d}"
            # ensure unique game_id per worker so history randomness differs
            worker_game_id = int(cfg.run.game_id) + gen * 1000 + wid
            # ensure root_id ranges don’t collide (optional, but helps debugging)
            worker_start_root_id = start

            task_q.put(
                {
                    "gen": gen,
                    "out_dir": str(out_dir),
                    "weights_path": str(weights_path),
                    "game_id": worker_game_id,
                    "start_root_id": worker_start_root_id,
                    "history_seed": (gen * 100000 + wid),
                    "num_roots": n_roots,
                }
            )
            tasks_sent += 1

        # 2) wait for all worker results
        results = []
        for _ in range(tasks_sent):
            results.append(result_q.get())


        # 2) load ALL samples from this generation (train on exactly these)
        samples = []
        for proc_dir in sorted(gen_dataset_dir.glob("proc_*")):
            manifest = proc_dir / "manifest.jsonl"
            if not manifest.exists():
                continue
            for entry in load_manifest(manifest):
                samples.extend(torch.load(entry.path, map_location="cpu"))

        ## -- Safety check for samples :
        if not samples:
            raise RuntimeError(f"No samples written for gen={gen} in {gen_dataset_dir}")


        num_controller = sum(1 for s in samples if s.get("player") == "controller")
        num_adversary = sum(1 for s in samples if s.get("player") == "adversary")


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

    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"selfplay worker died: pid={p.pid} exitcode={p.exitcode}")


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
            iterations=1000,
            feature_version=1,
        ),
    )


    ## MODEL TRAINING PARMS FOR SELF-IMPROVEMENT LOOP:
    num_selfPlay_workers = 8
    num_generations = 100
    history_nontrivial_hops = [0, 5 , 10 , 15 , 20, 25, 30, 35]  # per worker
    roots_per_generation = 400
    adv_iterations_per_root = 5000
    cont_iterations_per_root = 5000
    train_steps_per_generation = 600   
    max_batch_size = 256
    train_log_csv = Path("simulator_output/mcts_dnn_logs/train_metrics.csv")
    ckpt_dir = Path("simulator_output/mcts_dnn_checkpoints")

    model = AlphaZeroModel(
        num_actions_controller=cfg.model.num_actions_controller,
        num_actions_adversary=cfg.model.num_actions_adversary,
    ).to(torch.device(cfg.model.device))
    model.eval()



    ## 

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

        selfImprovementPolicy(
            cfg=cfg,
            model=model,
            num_generations=num_generations,
            num_selfPlay_workers=num_selfPlay_workers,
            roots_per_generation=roots_per_generation,
            adv_iterations_per_root=adv_iterations_per_root,
            cont_iterations_per_root=cont_iterations_per_root,
            history_nontrivial_hops=history_nontrivial_hops,
            max_batch_size=max_batch_size,
            train_steps_per_generation=train_steps_per_generation,
            ckpt_dir=ckpt_dir,
            train_log_csv=train_log_csv,
            device_for_features=torch.device("cpu"),
        )
    except Exception as e:
        print(f"ERROR during self-improvement policy: {e}")
        raise
    



if __name__ == "__main__":
    main()


