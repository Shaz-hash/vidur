# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import queue
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Avoid CPU oversubscription per process
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

## TODO !!! Bring these into the Game Version2 if needed else remove them, Investigate their requirement
from ...environment import VidurMCTSEnvironment
from ...launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions
from ...virtual_simulator import VirtualSimulator

from .config import MultipleProcessTrainingConfig
from .virtual_environment import VirtualVidurMCTSEnvironment
from .mctsDNN import VidurMCTS
from .DNN.dnn_spec import make_dnn_spec
from .DNN.models import AlphaZeroModel
from .DNN.replay_write import ReplayWriter, ReplayWriterConfig
from .DNN.selfPlay import SelfPlayRunner
from .DNN.replay_buffer import BestModelReplayBuffer
from .DNN.replay_dataset import collate_mixed_samples
from .DNN.trainer import Trainer, TrainerConfig
from .DNN.export_torchscript_gv2 import export_torchscript_artifacts_gv2


# Imports for the game evaluationn 
## TODO : might bring these into game version 2 logger
from .DNN.eval_utils import grade_arena_from_game_logs, extract_model_state, NoopReplayWriter
from .logger.evaluation_pipeline_logger import (
    ArenaGameCycleFileLogger,
    EvaluationMetricsLogger,
    generation_log_dir_from_iter_log,
)



def _set_global_seeds(seed: int, *, torch_deterministic: bool) -> None:
    s = int(seed)
    random.seed(s)
    if np is not None:
        np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    if bool(torch_deterministic):
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
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def configure_simulation(sim_args: Iterable[str]) -> SimulationConfig:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(sim_args)
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv

    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False

    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]

    return cfg

def _attach_native_runtime(model: Any, runtime: Any, model_version: int) -> None:
    setattr(model, "_native_ts_runtime", runtime)
    setattr(model, "_native_ts_model_version", int(model_version))


def _build_constraints_and_explore(
    cfg: MultipleProcessTrainingConfig,
) -> tuple[MCTSConstraintConfig, MCTSExploreConfig]:
    gv2 = cfg.game_v2
    gv2.validate()
    legacy = gv2.legacy_mcts

    decode_slos = tuple(float(x) for x in (legacy.decode_slos or (50.0,)))
    slo_options = RequestSLOOptions(
        prefill_slos=(3.0,),
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
    explore_cfg = MCTSExploreConfig()
    native_on = (str(cfg.environment_lang).strip().lower() == "native")

    setattr(explore_cfg, "native_mcts_enabled", native_on)
    setattr(explore_cfg, "torchscript_full_native_search", native_on)
    setattr(explore_cfg, "native_log_events", bool(cfg.native_log_events))
    setattr(explore_cfg, "native_profile", bool(cfg.native_profile))
    setattr(explore_cfg, "native_log_flush_every", int(cfg.native_log_flush_every))
    setattr(explore_cfg, "max_forced_hops", int(cfg.max_forced_hops_per_root))
    setattr(explore_cfg, "reuse_root_infer_inputs", False)

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
    cfg: MultipleProcessTrainingConfig,
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


def _partition_roots_contiguous(total_roots: int, workers: int) -> list[tuple[int, int]]:
    total_roots = int(total_roots)
    workers = int(workers)
    base = total_roots // workers
    rem = total_roots % workers

    out: list[tuple[int, int]] = []
    start = 0
    for wid in range(workers):
        count = base + (1 if wid < rem else 0)
        out.append((start, count))
        start += count
    return out


def _next_generation_index(dataset_base: Path) -> int:
    max_gen = -1
    for p in dataset_base.glob("gen_*"):
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("gen_"):
            continue
        tail = name[4:]
        if tail.isdigit():
            max_gen = max(max_gen, int(tail))
    return max_gen + 1

def _load_weights_into_model(model: torch.nn.Module, weights_path: Path) -> None:
    blob = torch.load(weights_path, map_location="cpu")
    model_state = extract_model_state(blob)
    model.load_state_dict(model_state, strict=True)
    model.eval()

def _load_trainer_from_checkpoint(trainer: Trainer, ckpt_path: Path) -> None:
    blob = torch.load(ckpt_path, map_location="cpu")
    model_state = extract_model_state(blob)
    trainer.model.load_state_dict(model_state, strict=True)

    # Optional: restore optimizer/step when checkpoint has them.
    opt_state = blob.get("optimizer_state", None)
    if opt_state is not None:
        try:
            trainer.opt.load_state_dict(opt_state)
        except Exception:
            pass

    if "step" in blob:
        try:
            trainer.step = int(blob["step"])
        except Exception:
            pass

    if "best_eval_loss" in blob:
        try:
            trainer.best_eval_loss = float(blob["best_eval_loss"])
        except Exception:
            pass

    trainer.model.eval()


def _build_arena_game_entries(
    *,
    cfg: MultipleProcessTrainingConfig,
    generation: int,
) -> list[dict]:
    ecfg = cfg.evaluation
    rng = random.Random(int(ecfg.random_seed_base) + int(generation))
    n = int(ecfg.num_games)
    max_hops = int(ecfg.max_history_hops)

    hops = [int(rng.randint(0, max_hops)) for _ in range(n)]
    if bool(ecfg.ensure_zero_hop_game) and n > 0:
        hops[0] = 0

    base_gid = int(cfg.run.game_id) + int(ecfg.game_id_offset) + int(generation) * 1_000_000
    out: list[dict] = []
    for i, h in enumerate(hops):
        out.append(
            {
                "game_id": int(base_gid + i),
                "history_hops": int(h),
                "player_to_act": str(ecfg.start_player),
                "depth": int(ecfg.start_root_depth),
                "root_id": int(i),
            }
        )
    return out



# TODO : Might better to bring this part into the logger
_TRAIN_LOG_HEADER = [
    "time",
    "event",
    "generation",
    "train_step",
    "train_step_in_generation",
    "loss",
    "policy_loss",
    "value_loss",
    "controller_policy_loss",
    "controller_value_loss",
    "controller_count",
    "adversary_policy_loss",
    "adversary_value_loss",
    "adversary_count",
    "replay_total_samples",
    "added_from_generation",
    "dataset_dir",
    "checkpoint_path",
    "latest_path",
]


def _append_train_log_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_TRAIN_LOG_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _TRAIN_LOG_HEADER})


def _collect_results(
    result_q: mp.Queue,
    *,
    expected: int,
    task_kind: str,
    timeout_sec: int,
) -> list[dict]:
    msgs: list[dict] = []
    for _ in range(int(expected)):
        try:
            msg = result_q.get(timeout=int(timeout_sec))
        except queue.Empty as e:
            raise RuntimeError(
                f"Timed out waiting for worker result: task_kind={task_kind}, expected={expected}, got={len(msgs)}"
            ) from e

        msg_kind = str(msg.get("task_kind", ""))
        if msg_kind != str(task_kind):
            raise RuntimeError(f"Unexpected result kind='{msg_kind}' while waiting for '{task_kind}': {msg}")

        if not bool(msg.get("ok", False)):
            tb = str(msg.get("traceback", "")).strip()
            err = str(msg.get("error", "unknown error"))
            wid = msg.get("worker_id", "?")
            if tb:
                raise RuntimeError(f"{task_kind} worker {wid} failed: {err}\n{tb}")
            raise RuntimeError(f"{task_kind} worker {wid} failed: {err}")

        msgs.append(msg)
    return msgs


def _selfplay_worker_main(
    worker_id: int,
    total_workers: int,
    cfg: MultipleProcessTrainingConfig,
    task_q: mp.Queue,
    result_q: mp.Queue,
    ready_q: mp.Queue,
) -> None:
    del total_workers

    try:
        _set_global_seeds(
            int(cfg.game_v2.reproducibility.global_seed) + int(worker_id),
            torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
        )
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        _, env, _, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=bool(cfg.use_virtual_env))

        spec = make_dnn_spec(cfg=cfg.game_v2)
        model = AlphaZeroModel(spec=spec).to(torch.device(cfg.model.device))
        model.eval()

        native_runtime = None
        if bool(getattr(explore_cfg, "native_mcts_enabled", False)):
            from ... import mcts_native_gv2 as _mcts_native_gv2
            native_runtime = _mcts_native_gv2.NativeTorchScriptInferRuntimeGV2(str(cfg.model.device), -50.0, 100.0)
                
        arena_candidate_model = AlphaZeroModel(spec=spec).to(torch.device(cfg.model.device))
        arena_candidate_model.eval()

        arena_best_model = AlphaZeroModel(spec=spec).to(torch.device(cfg.model.device))
        arena_best_model.eval()

        ready_q.put({"worker_id": int(worker_id), "ready": True})
    except Exception as exc:
        ready_q.put(
            {
                "worker_id": int(worker_id),
                "ready": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_q.get()
        if task is None:
            break

        task_kind = str(task.get("task_kind", "selfplay"))

        if task_kind == "selfplay":
            writer = None
            mcts = None
            try:
                gen = int(task["generation"])
                out_dir = Path(task["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)

                _load_weights_into_model(model, Path(task["weights_path"]))

                # For native :
                if native_runtime is not None:
                    mv = int(task["native_model_version"])
                    mspec = str(task["native_model_spec"])
                    native_runtime.load_models({mv: mspec})
                    _attach_native_runtime(model, native_runtime, mv)


                task_seed = int(task.get("task_seed", 0))
                _set_global_seeds(
                    task_seed,
                    torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
                )

                writer = ReplayWriter(
                    ReplayWriterConfig(
                        out_dir=out_dir,
                        shard_size=int(cfg.dataset.shard_size),
                    )
                )

                logs_base = Path(cfg.logging.mcts_iter_log).parent
                logs_dir = logs_base / f"gen_{gen:06d}"
                logs_dir.mkdir(parents=True, exist_ok=True)

                iter_log = logs_dir / f"mcts_iter_p{int(worker_id):02d}.csv"
                root_log = logs_dir / f"mcts_root_p{int(worker_id):02d}.csv"

                mcts = VidurMCTS(
                    env=env,
                    explore_cfg=explore_cfg,
                    log_path=iter_log,
                    tree_log_path=root_log,
                    logger_flush_every=int(cfg.logging.flush_every),
                    verbose=False,
                    complete_log=False,
                )

                runner = SelfPlayRunner(
                    env=env,
                    mcts=mcts,
                    model=model,
                    writer=writer,
                    device_for_features=torch.device(cfg.model.device),
                    game_v2_cfg=cfg.game_v2,
                )

                runner.run_n_roots(
                    game_id=int(task["game_id"]),
                    num_roots=int(task["num_roots"]),
                    adv_iterations_per_root=int(task["adv_iterations_per_root"]),
                    cont_iterations_per_root=int(task["cont_iterations_per_root"]),
                    max_batch_size=int(task["max_batch_size"]),
                    start_root_id=int(task["start_root_id"]),
                    start_root_depth=int(task["start_root_depth"]),
                    start_player=str(task["start_player"]),
                    feature_version=int(task["feature_version"]),
                    sample_from_mcts_policy=bool(task["sample_from_mcts_policy"]),
                    selfplay_policy_temperature=float(task["selfplay_policy_temperature"]),
                    action_seed_base=int(task["action_seed_base"]),
                    history_nontrivial_hops=int(task["history_nontrivial_hops"]),
                    history_seed=int(task["history_seed"]),
                    max_forced_hops_per_root=int(task["max_forced_hops_per_root"]),
                    history_max_total_steps=int(task["history_max_total_steps"]),
                    log_history_rows=bool(task["log_history_rows"]),
                )

                result_q.put(
                    {
                        "ok": True,
                        "task_kind": "selfplay",
                        "worker_id": int(worker_id),
                        "generation": gen,
                        "out_dir": str(out_dir),
                    }
                )
            except Exception as exc:
                result_q.put(
                    {
                        "ok": False,
                        "task_kind": "selfplay",
                        "worker_id": int(worker_id),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                if mcts is not None:
                    try:
                        mcts.close()
                    except Exception:
                        pass
            continue

        if task_kind == "arena":
            arena_mcts = None
            try:
                gen = int(task["generation"])
                entries = list(task.get("arena_entries", []))

                _load_weights_into_model(arena_candidate_model, Path(task["candidate_weights_path"]))
                _load_weights_into_model(arena_best_model, Path(task["best_weights_path"]))

                # Loading model + weights for the native :
                if native_runtime is not None:
                    cv = int(task["candidate_model_version"])
                    bv = int(task["best_model_version"])
                    cs = str(task["candidate_model_spec"])
                    bs = str(task["best_model_spec"])
                    native_runtime.load_models({cv: cs, bv: bs})
                    _attach_native_runtime(arena_candidate_model, native_runtime, cv)
                    _attach_native_runtime(arena_best_model, native_runtime, bv)


                task_seed = int(task.get("task_seed", 0))
                _set_global_seeds(
                    task_seed,
                    torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
                )

                arena_mcts = VidurMCTS(
                    env=env,
                    explore_cfg=explore_cfg,
                    log_path=None,
                    tree_log_path=None,
                    logger_flush_every=int(cfg.logging.flush_every),
                    verbose=False,
                    complete_log=False,
                )

                runner = SelfPlayRunner(
                    env=env,
                    mcts=arena_mcts,
                    model=arena_candidate_model,
                    writer=NoopReplayWriter(),
                    device_for_features=torch.device(cfg.model.device),
                    game_v2_cfg=cfg.game_v2,
                )

                arena_games_dir = Path(task["arena_games_dir"])
                cycle_logger = ArenaGameCycleFileLogger(arena_games_dir)

                per_game = []
                for entry in entries:
                    out = runner.run_arena_game(
                        game_id=int(entry["game_id"]),
                        candidate_model=arena_candidate_model,
                        best_model=arena_best_model,
                        history_nontrivial_hops=int(entry["history_hops"]),
                        adv_iterations_per_root=int(task["adv_iterations_per_root"]),
                        cont_iterations_per_root=int(task["cont_iterations_per_root"]),
                        arena_time_limit_sec=float(task["arena_time_limit_sec"]),
                        arena_max_controller_cleanup_steps=int(task["arena_max_controller_cleanup_steps"]),
                        arena_max_total_turns=int(task["arena_max_total_turns"]),
                        feature_version=int(task["feature_version"]),
                        tie_points=float(task["tie_points"]),
                        start_player=str(entry.get("player_to_act", "adversary")),
                        start_root_depth=int(entry.get("depth", 0)),
                        history_root_id_for_logs=int(entry.get("root_id", 0)),
                        cycle_file_logger=cycle_logger,
                    )

                    cycle_a = out["cycle_a"]
                    cycle_b = out["cycle_b"]

                    cycle_logger.write_cycle_end(
                        game_id=int(entry["game_id"]),
                        cycle_label="candidate_as_adversary",
                        total_cost=float(cycle_a["total_cost"]),
                        slo_violations=int(cycle_a["slo_violations"]),
                        total_lateness=float(cycle_a["total_lateness"]),
                        end_reason=str(cycle_a.get("end_reason", "")),
                    )
                    cycle_logger.write_cycle_end(
                        game_id=int(entry["game_id"]),
                        cycle_label="best_as_adversary",
                        total_cost=float(cycle_b["total_cost"]),
                        slo_violations=int(cycle_b["slo_violations"]),
                        total_lateness=float(cycle_b["total_lateness"]),
                        end_reason=str(cycle_b.get("end_reason", "")),
                    )


                    per_game.append(
                        {
                            "game_id": int(out["game_id"]),
                            "winner": str(out["winner"]),
                            "candidate_points": float(out["candidate_points"]),
                            "best_points": float(out["best_points"]),
                            "candidate_as_adv_cost": float(out["candidate_as_adv_cost"]),
                            "best_as_adv_cost": float(out["best_as_adv_cost"]),
                        }
                    )

                result_q.put(
                    {
                        "ok": True,
                        "task_kind": "arena",
                        "worker_id": int(worker_id),
                        "generation": int(gen),
                        "num_games": int(len(per_game)),
                        "results": per_game,
                    }
                )
            except Exception as exc:
                result_q.put(
                    {
                        "ok": False,
                        "task_kind": "arena",
                        "worker_id": int(worker_id),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                if arena_mcts is not None:
                    try:
                        arena_mcts.close()
                    except Exception:
                        pass
            continue

        result_q.put(
            {
                "ok": False,
                "task_kind": task_kind,
                "worker_id": int(worker_id),
                "error": f"unknown task_kind={task_kind}",
            }
        )


def run_parallel_self_improvement(cfg: MultipleProcessTrainingConfig) -> None:
    cfg.validate()

    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    dataset_base = Path(cfg.dataset.out_dir)
    dataset_base.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_log_csv = Path(cfg.train_metrics_csv)

    eval_metrics_logger = EvaluationMetricsLogger(
        Path(cfg.evaluation_logging.metrics_csv),
        flush_every=int(cfg.logging.flush_every),
    )

    spec = make_dnn_spec(cfg=cfg.game_v2)
    model = AlphaZeroModel(spec=spec)

    trainer_h = cfg.game_v2.trainer
    trainer = Trainer(
        model=model,
        cfg=TrainerConfig(
            lr=float(trainer_h.lr),
            weight_decay=float(trainer_h.weight_decay),
            policy_weight=float(trainer_h.policy_weight),
            value_weight=float(trainer_h.value_weight),
            grad_clip_norm=float(trainer_h.grad_clip_norm),
            checkpoint_every=int(trainer_h.checkpoint_every),
            eval_every=int(trainer_h.eval_every),
            invalid_logit=float(trainer_h.invalid_logit),
        ),
        device=torch.device(cfg.model.device),
    )

    replay_buffer = BestModelReplayBuffer(
        capacity_samples=int(cfg.replay_capacity_samples),
        max_cached_shards=int(cfg.replay_max_cached_shards),
        seed=int(cfg.replay_seed),
    )

    # best_ckpt_path = ckpt_dir / "best.pt"
    # if not best_ckpt_path.exists():
    #     init_state = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
    #     torch.save({"model_state": init_state}, best_ckpt_path)

    best_ckpt_path = ckpt_dir / "best.pt"
    if not best_ckpt_path.exists():
        # Create first best checkpoint from current trainer init.
        trainer.save_checkpoint(best_ckpt_path)

    # Always resume trainer from best model checkpoint (not latest).
    _load_trainer_from_checkpoint(trainer, best_ckpt_path)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    ready_q = ctx.Queue()

    workers: list[mp.Process] = []
    try:
        for wid in range(int(cfg.num_processes)):
            p = ctx.Process(
                target=_selfplay_worker_main,
                args=(wid, int(cfg.num_processes), cfg, task_q, result_q, ready_q),
                daemon=False,
            )
            p.start()
            workers.append(p)

        for _ in range(int(cfg.num_processes)):
            try:
                msg = ready_q.get(timeout=int(cfg.worker_result_timeout_sec))
            except queue.Empty as e:
                raise RuntimeError("Timed out waiting for worker ready signal") from e

            if not bool(msg.get("ready", False)):
                err = str(msg.get("error", "unknown"))
                tb = str(msg.get("traceback", ""))
                wid = msg.get("worker_id", "?")
                raise RuntimeError(f"worker {wid} init failed: {err}\n{tb}")

        gen_start = _next_generation_index(dataset_base)

        for local_gen in range(int(cfg.num_generations)):
            gen = int(gen_start + local_gen)
            gen_dataset_dir = dataset_base / f"gen_{gen:06d}"
            gen_dataset_dir.mkdir(parents=True, exist_ok=True)

            weights_path = ckpt_dir / f"selfplay_weights_gen_{gen:06d}.pt"
            model_state_cpu = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
            torch.save({"model_state": model_state_cpu}, weights_path)

            native_on = bool(str(cfg.environment_lang).strip().lower() == "native")
            native_model_version = int(gen * 100 + 1)
            native_model_spec = ""

            if native_on:
                ts_dir = Path(cfg.native_torchscript_dir) / f"gen_{gen:06d}" / "selfplay"
                ts = export_torchscript_artifacts_gv2(
                    checkpoint_path=weights_path,
                    out_dir=ts_dir,
                    model_version=native_model_version,
                    spec=spec,
                    device="cpu",
                )
                native_model_spec = f"{ts.controller_path}||{ts.adversary_path}"



            splits = _partition_roots_contiguous(
                total_roots=int(cfg.roots_per_generation),
                workers=int(cfg.num_processes),
            )

            tasks_sent = 0
            for wid, (start, count) in enumerate(splits):
                if count <= 0:
                    continue

                out_dir = gen_dataset_dir / f"proc_{wid:02d}"
                worker_game_id = int(cfg.run.game_id) + gen * 1_000_000 + wid
                worker_start_root_id = int(cfg.run.root_id) + int(start)

                history_seed = int(cfg.history_seed) + gen * 10_000 + wid
                action_seed_base = int(cfg.action_seed_base) + gen * 1_000_000 + wid * 1000
                task_seed = int(cfg.game_v2.reproducibility.global_seed) + gen * 10_000 + wid

                task_q.put(
                    {
                        "task_kind": "selfplay",
                        "generation": gen,
                        "native_model_version": int(native_model_version),
                        "native_model_spec": str(native_model_spec),
                        "out_dir": str(out_dir),
                        "weights_path": str(weights_path),
                        "game_id": worker_game_id,
                        "num_roots": int(count),
                        "start_root_id": worker_start_root_id,
                        "start_root_depth": int(cfg.run.root_depth),
                        "start_player": str(cfg.run.root_player),
                        "feature_version": int(cfg.run.feature_version),
                        "adv_iterations_per_root": int(cfg.adv_iterations_per_root),
                        "cont_iterations_per_root": int(cfg.cont_iterations_per_root),
                        "max_batch_size": int(cfg.max_batch_size),
                        "history_nontrivial_hops": int(cfg.history_hops_per_worker[wid]),
                        "history_seed": history_seed,
                        "sample_from_mcts_policy": bool(cfg.sample_from_mcts_policy),
                        "selfplay_policy_temperature": float(cfg.selfplay_policy_temperature),
                        "action_seed_base": action_seed_base,
                        "max_forced_hops_per_root": int(cfg.max_forced_hops_per_root),
                        "history_max_total_steps": int(cfg.history_max_total_steps),
                        "log_history_rows": bool(cfg.log_history_rows),
                        "task_seed": task_seed,
                    }
                )
                tasks_sent += 1

            if tasks_sent <= 0:
                raise RuntimeError(f"No self-play tasks sent for generation {gen}")

            _collect_results(
                result_q,
                expected=tasks_sent,
                task_kind="selfplay",
                timeout_sec=int(cfg.worker_result_timeout_sec),
            )

            added_from_gen = replay_buffer.add_generation_dir(gen_dataset_dir)
            if replay_buffer.total_samples <= 0:
                raise RuntimeError(
                    f"Replay buffer empty after generation={gen}, added_from_gen={added_from_gen}, dir={gen_dataset_dir}"
                )

            for step_in_gen in range(int(cfg.train_steps_per_generation)):
                samples = replay_buffer.sample_batch(int(cfg.train_batch_size))
                batch_by_player = collate_mixed_samples(samples, device=trainer.device)
                metrics = trainer.train_step(batch_by_player)

                _append_train_log_row(
                    train_log_csv,
                    {
                        "time": time.time(),
                        "event": "train_step",
                        "generation": int(gen),
                        "train_step": int(metrics["step"]),
                        "train_step_in_generation": int(step_in_gen),
                        "loss": float(metrics["loss"]),
                        "policy_loss": float(metrics["policy_loss"]),
                        "value_loss": float(metrics["value_loss"]),
                        "controller_policy_loss": float(metrics["controller_policy_loss"]),
                        "controller_value_loss": float(metrics["controller_value_loss"]),
                        "controller_count": float(metrics["controller_count"]),
                        "adversary_policy_loss": float(metrics["adversary_policy_loss"]),
                        "adversary_value_loss": float(metrics["adversary_value_loss"]),
                        "adversary_count": float(metrics["adversary_count"]),
                        "replay_total_samples": int(replay_buffer.total_samples),
                        "added_from_generation": int(added_from_gen),
                        "dataset_dir": str(gen_dataset_dir),
                        "checkpoint_path": "",
                        "latest_path": "",
                    },
                )

            gen_ckpt_path = ckpt_dir / f"gen_{gen:06d}.pt"
            trainer.save_checkpoint(gen_ckpt_path)

            latest_path = ckpt_dir / "latest.pt"
            shutil.copyfile(gen_ckpt_path, latest_path)

            _append_train_log_row(
                train_log_csv,
                {
                    "time": time.time(),
                    "event": "checkpoint",
                    "generation": int(gen),
                    "train_step": int(trainer.step),
                    "train_step_in_generation": "",
                    "loss": "",
                    "policy_loss": "",
                    "value_loss": "",
                    "controller_policy_loss": "",
                    "controller_value_loss": "",
                    "controller_count": "",
                    "adversary_policy_loss": "",
                    "adversary_value_loss": "",
                    "adversary_count": "",
                    "replay_total_samples": int(replay_buffer.total_samples),
                    "added_from_generation": int(added_from_gen),
                    "dataset_dir": str(gen_dataset_dir),
                    "checkpoint_path": str(gen_ckpt_path),
                    "latest_path": str(latest_path),
                },
            )


            ## Game Evaluation begins here :
            if bool(cfg.evaluation.enabled):
                ecfg = cfg.evaluation

                gen_logs_dir = generation_log_dir_from_iter_log(cfg.logging.mcts_iter_log, gen)
                arena_games_dir = gen_logs_dir / "arena_games"
                arena_games_dir.mkdir(parents=True, exist_ok=True)

                arena_entries = _build_arena_game_entries(cfg=cfg, generation=gen)

                candidate_model_version = int(gen * 100 + 2)
                best_model_version = int(gen * 100 + 3)
                candidate_model_spec = ""
                best_model_spec = ""

                if native_on:
                    ts_dir = Path(cfg.native_torchscript_dir) / f"gen_{gen:06d}" / "arena"
                    ts_c = export_torchscript_artifacts_gv2(
                        checkpoint_path=gen_ckpt_path, out_dir=ts_dir, model_version=candidate_model_version, spec=spec, device="cpu"
                    )
                    ts_b = export_torchscript_artifacts_gv2(
                        checkpoint_path=best_ckpt_path, out_dir=ts_dir, model_version=best_model_version, spec=spec, device="cpu"
                    )
                    candidate_model_spec = f"{ts_c.controller_path}||{ts_c.adversary_path}"
                    best_model_spec = f"{ts_b.controller_path}||{ts_b.adversary_path}"


                splits_eval = _partition_roots_contiguous(
                    total_roots=len(arena_entries),
                    workers=int(cfg.num_processes),
                )

                arena_tasks_sent = 0
                for wid, (start, count) in enumerate(splits_eval):
                    if count <= 0:
                        continue
                    subset = arena_entries[start : start + count]
                    arena_task_seed = int(cfg.game_v2.reproducibility.global_seed) + gen * 100_000 + wid + 777

                    task_q.put(
                        {
                            "task_kind": "arena",
                            "generation": int(gen),
                            "candidate_model_version": int(candidate_model_version),
                            "best_model_version": int(best_model_version),
                            "candidate_model_spec": str(candidate_model_spec),
                            "best_model_spec": str(best_model_spec),
                            "arena_entries": subset,
                            "arena_games_dir": str(arena_games_dir),
                            "candidate_weights_path": str(gen_ckpt_path),
                            "best_weights_path": str(best_ckpt_path),
                            "adv_iterations_per_root": int(ecfg.arena_iters_adversary),
                            "cont_iterations_per_root": int(ecfg.arena_iters_controller),
                            "arena_time_limit_sec": float(ecfg.arena_time_limit_sec),
                            "arena_max_controller_cleanup_steps": int(ecfg.arena_max_controller_cleanup_steps_safety),
                            "arena_max_total_turns": int(ecfg.arena_max_total_turns_safety),
                            "feature_version": int(ecfg.feature_version),
                            "tie_points": float(ecfg.tie_points),
                            "task_seed": int(arena_task_seed),
                        }
                    )
                    arena_tasks_sent += 1

                if arena_tasks_sent <= 0:
                    raise RuntimeError(f"No arena tasks sent for generation {gen}")

                _collect_results(
                    result_q,
                    expected=arena_tasks_sent,
                    task_kind="arena",
                    timeout_sec=int(cfg.worker_result_timeout_sec),
                )

                arena_results_csv = gen_logs_dir / "arena_results.csv"
                arena_metrics = grade_arena_from_game_logs(
                    game_log_dir=arena_games_dir,
                    out_csv=arena_results_csv,
                    tie_points=float(ecfg.tie_points),
                    win_threshold=float(ecfg.arena_win_threshold),
                )

                best_before = str(best_ckpt_path)
                promoted = bool(arena_metrics["passed"])
                if promoted:
                    shutil.copyfile(gen_ckpt_path, best_ckpt_path)

                eval_metrics_logger.log_generation(
                    generation=int(gen),
                    candidate_checkpoint=str(gen_ckpt_path),
                    best_checkpoint_before=str(best_before),
                    best_checkpoint_after=str(best_ckpt_path),
                    num_games=int(arena_metrics["num_eval_roots"]),
                    candidate_points=float(arena_metrics["candidate_points"]),
                    best_points=float(arena_metrics["best_points"]),
                    total_points=float(arena_metrics["total_points"]),
                    candidate_win_rate=float(arena_metrics["candidate_win_rate"]),
                    win_threshold=float(arena_metrics["arena_win_threshold"]),
                    passed=bool(arena_metrics["passed"]),
                    promoted_to_best=bool(promoted),
                    arena_results_csv=str(arena_results_csv),
                    arena_games_dir=str(arena_games_dir),
                )


    finally:

        try:
            eval_metrics_logger.close()
        except Exception:
            pass

        for _ in workers:
            try:
                task_q.put(None)
            except Exception:
                pass

        for p in workers:
            try:
                p.join(timeout=30)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.join(timeout=5)
                except Exception:
                    pass
