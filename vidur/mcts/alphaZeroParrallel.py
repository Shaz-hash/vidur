# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
alphaZeroParrallel.py

build the native if needed using the following command :

cd /home/shazer/Desktop/Research/Vidur/vidur/vidur/mcts/native
PY=/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$($PY -m pybind11 --cmakedir)"

cmake --build build -j
cp build/mcts_native*.so ../


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

# TODO: We need a worker pool for the selfImprovementPolicy and Evaluator that can run multiple MCTS-DNN arenas in parallel. 
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
import traceback
import queue
import faulthandler
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

from .environment import VidurMCTSEnvironment
from .mctsDNN import VidurMCTS
from .virtual_environment import VirtualVidurMCTSEnvironment
from .virtual_simulator import VirtualSimulator

# environment.py currently imports configs from launch_mcts_job.py
from .launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions

from .DNN.models import AlphaZeroModel
from .DNN.replay_write import ReplayWriter, ReplayWriterConfig
from .DNN.selfPlay import SelfPlayRunner, SingleRootRun

from .DNN.replay_dataset import load_manifest, collate_mixed_samples
from .DNN.trainer import Trainer, TrainerConfig
from .DNN.replay_buffer import BestModelReplayBuffer

from .logger.eval_logger import EvalArenaGenerationLogger
from .logger.replay_logger import ReplayBufferLogger
from .DNN.evaluator import EvaluatorConfig

from .DNN.eval_utils import (
    grade_arena_from_game_logs,
    extract_model_state,
    NoopReplayWriter,
    write_arena_cycle_end_csv,
)

try:
    from .native.python.infer_client import (
        TorchScriptInferClient,
        TorchScriptModelPaths,
        TorchScriptServiceModelAdapter,
        NativeTorchScriptModelAdapter,
        build_native_torchscript_runtime,
        build_native_infer_service_runtime,
        start_torchscript_infer_service,
    )
except Exception:  # pragma: no cover - optional runtime dependency path
    TorchScriptInferClient = None
    TorchScriptModelPaths = None
    TorchScriptServiceModelAdapter = None
    NativeTorchScriptModelAdapter = None
    build_native_torchscript_runtime = None
    build_native_infer_service_runtime = None
    start_torchscript_infer_service = None



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
    "arena_candidate_points",
    "arena_best_points",
    "arena_total_points",
    "arena_candidate_win_rate",
    "arena_win_threshold",
    "arena_passed",
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


def _build_constraints_and_explore(cfg: "AlphaZeroConfig") -> tuple[MCTSConstraintConfig, MCTSExploreConfig]:
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
    setattr(
        explore_cfg,
        "controller_min_prior_threshold",
        float(cfg.explore.controller_min_prior_threshold),
    )
    setattr(
        explore_cfg,
        "adversary_min_prior_threshold",
        float(cfg.explore.adversary_min_prior_threshold),
    )
    setattr(
    explore_cfg,
    "root_dirichlet_noise_enabled",
    bool(cfg.explore.root_dirichlet_noise_enabled),
    )
    setattr(
        explore_cfg,
        "root_dirichlet_alpha",
        float(cfg.explore.root_dirichlet_alpha),
    )
    setattr(
        explore_cfg,
        "root_dirichlet_epsilon",
        float(cfg.explore.root_dirichlet_epsilon),
    )
    setattr(
        explore_cfg,
        "native_mcts_enabled",
        bool(getattr(getattr(cfg, "native", None), "enabled", False))
        and str(getattr(getattr(cfg, "native", None), "backend", "python")) == "cpp_virtual",
    )
    setattr(
        explore_cfg,
        "torchscript_full_native_search",
        bool(getattr(getattr(cfg, "native", None), "torchscript_full_native_search", False)),
    )

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
            native_enabled=_use_cpp_virtual_backend(cfg),
            native_strict_compat=bool(getattr(getattr(cfg, "native", None), "strict_compat", True)),
        )
    else:
        simulator = Simulator(sim_cfg, register_atexit=False)
        env = VidurMCTSEnvironment(
            base_simulator=simulator,
            constraints=constraints,
            explore_cfg=explore_cfg,
        )

    return simulator, env, constraints, explore_cfg



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

def _build_arena_root_entries(gen: int , evaluator_cfg: EvaluatorConfig , cfg: "AlphaZeroConfig") -> list[dict]:
    rng = random.Random(int(evaluator_cfg.random_seed_base) + int(gen))
    n = int(evaluator_cfg.num_random_games)
    max_hops = int(evaluator_cfg.max_history_depth)
    hops = [0] + [int(rng.randint(0, max_hops)) for _ in range(max(0, n - 1))]
    base_gid = int(cfg.run.game_id) + 900000 + int(gen) * 1000
    return [
        {"game_id": base_gid + i, "history_hops": h, "player_to_act": "adversary", "depth": 0, "root_id": i}
        for i, h in enumerate(hops)
    ]


def _load_ckpt_model_state_cpu(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    return {k: v.detach().cpu() for k, v in ckpt["model_state"].items()}

def _restore_trainer_from_ckpt(*, trainer: Trainer, path: Path) -> None:
    ckpt = torch.load(path, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state"])
    try:
        trainer.opt.load_state_dict(ckpt["optimizer_state"])
    except Exception:
        pass
    trainer.best_eval_loss = float(ckpt.get("best_eval_loss", trainer.best_eval_loss))



def _infer_best_generation_from_train_log(path: Path) -> int:
    if not path.exists():
        return -1

    best_gen = -1
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("event", "")).strip() != "eval_arena":
                continue
            saved = str(row.get("saved_best", "")).strip().lower()
            if saved not in {"true", "1", "yes"}:
                continue
            try:
                best_gen = max(best_gen, int(row.get("gen", "")))
            except Exception:
                pass
    return best_gen


def _csv_bool(v: object) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _use_cpp_virtual_backend(cfg: "AlphaZeroConfig") -> bool:
    ncfg = getattr(cfg, "native", None)
    if ncfg is None:
        return False
    return bool(getattr(ncfg, "enabled", False)) and str(getattr(ncfg, "backend", "python")) == "cpp_virtual"


def _infer_service_enabled(cfg: "AlphaZeroConfig") -> bool:
    ncfg = getattr(cfg, "native", None)
    if ncfg is None:
        return False
    return bool(getattr(ncfg, "enabled", False)) and str(getattr(ncfg, "infer_mode", "python")) == "torchscript_service"


def _infer_cpp_runtime_enabled(cfg: "AlphaZeroConfig") -> bool:
    ncfg = getattr(cfg, "native", None)
    if ncfg is None:
        return False
    return bool(getattr(ncfg, "enabled", False)) and str(getattr(ncfg, "infer_mode", "python")) == "torchscript_cpp"


def _wait_for_infer_service(addr: str, *, timeout_s: float = 30.0) -> None:
    if TorchScriptInferClient is None:
        raise RuntimeError("TorchScriptInferClient is unavailable; cannot use infer_mode=torchscript_service")

    deadline = time.time() + float(timeout_s)
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            c = TorchScriptInferClient(addr=str(addr))
            if c.ping():
                c.close()
                return
            c.close()
        except Exception as exc:  # pragma: no cover - startup retry path
            last_err = exc
        time.sleep(0.2)
    raise RuntimeError(f"inference service did not become ready at {addr}: {last_err!r}")


def _export_torchscript_artifacts_for_checkpoint(
    *,
    checkpoint_path: Path,
    out_dir: Path,
    model_version: int,
    device: str = "cpu",
    num_actions_controller: int,
    num_actions_adversary: int,
) -> tuple[Path, Path, Path]:
    from .DNN.export_torchscript import export_torchscript_artifacts

    out = export_torchscript_artifacts(
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        model_version=int(model_version),
        device=str(device),
        num_actions_controller=int(num_actions_controller),
        num_actions_adversary=int(num_actions_adversary),
    )
    return out.controller_path, out.adversary_path, out.meta_path

def _bootstrap_replay_from_train_log(
    *,
    replay_buffer: BestModelReplayBuffer,
    train_log_csv: Path,
    best_generation: int,
) -> tuple[int, int]:
    """
    Load replay from generations AFTER current best where arena failed.
    Returns: (num_generations_loaded, raw_samples_added)
    """
    if best_generation < 0 or not train_log_csv.exists():
        return 0, 0

    rows: list[tuple[int, bool, Path | None]] = []
    with train_log_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("event", "")).strip() != "eval_arena":
                continue
            try:
                gen = int(row.get("gen", ""))
            except Exception:
                continue
            saved_best = _csv_bool(row.get("saved_best", ""))
            ds = str(row.get("dataset_dir", "")).strip()
            dataset_dir = Path(ds) if ds else None
            rows.append((gen, saved_best, dataset_dir))

    rows.sort(key=lambda x: x[0])

    loaded_gens = 0
    raw_added = 0
    seen_gens: set[int] = set()

    for gen, saved_best, dataset_dir in rows:
        if gen <= int(best_generation):
            continue
        if saved_best:
            continue
        if gen in seen_gens:
            continue
        if dataset_dir is None or not dataset_dir.exists():
            continue

        raw_added += int(replay_buffer.add_generation_dir(dataset_dir))
        seen_gens.add(gen)
        loaded_gens += 1

    return loaded_gens, raw_added


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
    # history_nontrivial_hops: int,
    default_history_nontrivial_hops: int,
    use_virtual_env: bool,
    total_workers: int = 1,
) -> None:
    # Important: avoid CPU oversubscription when many workers are active.
    # Native TorchScript on CPU can otherwise spawn many threads per worker.
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print(f"[worker {worker_id}] torch threads set to intra=1 interop=1", flush=True)
    except Exception as e:
        print(f"[worker {worker_id}] thread-limit setup failed: {e}", flush=True)
    # Pin each worker to a dedicated logical CPU from current allowed set (Linux)
    try:
        avail_cpus = sorted(os.sched_getaffinity(0))
        if avail_cpus:
            cpu = avail_cpus[worker_id % len(avail_cpus)]
            os.sched_setaffinity(0, {cpu})
            print(f"[worker {worker_id}] pinned to CPU {cpu}")
    except Exception as e:
        print(f"[worker {worker_id}] affinity pin failed: {e}")

    try:
        fault_log = Path(cfg.logging.mcts_iter_log).parent / f"worker_{int(worker_id):02d}_faulthandler.log"
        fault_log.parent.mkdir(parents=True, exist_ok=True)
        _fault_fh = open(fault_log, "a", buffering=1)
        faulthandler.enable(file=_fault_fh, all_threads=True)
    except Exception as e:
        print(f"[worker {worker_id}] faulthandler setup failed: {e}")

    # Native C++ tracing controls (used by mcts_native via env vars)
    try:
        ncfg = getattr(cfg, "native", None)
        if bool(getattr(ncfg, "native_trace", False)):
            os.environ["VIDUR_NATIVE_TRACE"] = "1"
            os.environ["VIDUR_NATIVE_TRACE_EVERY"] = str(int(getattr(ncfg, "native_trace_every", 500)))
            trace_dir = str(getattr(ncfg, "native_trace_dir", "")).strip()
            if trace_dir:
                os.environ["VIDUR_NATIVE_TRACE_DIR"] = trace_dir
            print(
                f"[worker {worker_id}] native trace enabled "
                f"(every={os.environ.get('VIDUR_NATIVE_TRACE_EVERY')}, "
                f"dir={os.environ.get('VIDUR_NATIVE_TRACE_DIR', '/tmp')})",
                flush=True,
            )
        else:
            os.environ.pop("VIDUR_NATIVE_TRACE", None)
            os.environ.pop("VIDUR_NATIVE_TRACE_EVERY", None)
            os.environ.pop("VIDUR_NATIVE_TRACE_DIR", None)
    except Exception as e:
        print(f"[worker {worker_id}] native trace env setup failed: {e}")



    # Build simulator/env ONCE per process (big speed win vs rebuilding every gen)
    _, env, _, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=use_virtual_env)
    multiprocess_safety_mode = bool(getattr(getattr(cfg, "native", None), "multiprocess_safety_mode", True))
    safe_multi = multiprocess_safety_mode and int(total_workers) > 1
    if safe_multi:
        if bool(getattr(explore_cfg, "native_mcts_enabled", False)):
            setattr(explore_cfg, "native_mcts_enabled", False)
            print(
                f"[worker {worker_id}] safety: disabled native_mcts_enabled for multiprocess stability",
                flush=True,
            )
        if hasattr(env, "_native_enabled") and bool(getattr(env, "_native_enabled", False)):
            try:
                setattr(env, "_native_enabled", False)
                print(
                    f"[worker {worker_id}] safety: disabled native controller sampler for multiprocess stability",
                    flush=True,
                )
            except Exception:
                pass

    # Build model once per process; reload weights each generation
    model = AlphaZeroModel(
        num_actions_controller=cfg.model.num_actions_controller,
        num_actions_adversary=cfg.model.num_actions_adversary,
    ).to(torch.device("cpu"))
    model.eval()

    use_service_infer = _infer_service_enabled(cfg)
    use_cpp_runtime_infer = _infer_cpp_runtime_enabled(cfg)
    infer_client = None
    native_ts_runtime = None
    native_service_runtime = None
    runtime_device = str(cfg.native.infer_service_device)
    if use_service_infer:
        if TorchScriptInferClient is None:
            raise RuntimeError("infer_mode=torchscript_service requested but infer client is unavailable")
        infer_client = TorchScriptInferClient(addr=str(cfg.native.infer_service_addr))
    if use_cpp_runtime_infer:
        if build_native_torchscript_runtime is None:
            raise RuntimeError("infer_mode=torchscript_cpp requested but native runtime builder is unavailable")
        if runtime_device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                f"torchscript_cpp requested device={runtime_device} but CUDA is unavailable "
                "(torch.cuda.is_available() is False). Fix CUDA/runtime setup or switch infer_mode."
            )
        native_ts_runtime = build_native_torchscript_runtime(
            device=runtime_device,
        )
        print(
            f"[worker {worker_id}] torchscript_cpp runtime device={runtime_device}",
            flush=True,
        )
    if use_service_infer and build_native_infer_service_runtime is not None:
        try:
            native_service_runtime = build_native_infer_service_runtime(
                addr=str(cfg.native.infer_service_addr),
            )
            if native_service_runtime.ping():
                print(
                    f"[worker {worker_id}] native infer-service runtime connected addr={cfg.native.infer_service_addr}",
                    flush=True,
                )
        except Exception as e:
            native_service_runtime = None
            print(
                f"[worker {worker_id}] native infer-service runtime unavailable: {e}",
                flush=True,
            )


    arena_candidate_model = AlphaZeroModel(
        num_actions_controller=cfg.model.num_actions_controller,
        num_actions_adversary=cfg.model.num_actions_adversary,
    ).to(torch.device("cpu"))
    arena_candidate_model.eval()

    arena_best_model = AlphaZeroModel(
        num_actions_controller=cfg.model.num_actions_controller,
        num_actions_adversary=cfg.model.num_actions_adversary,
    ).to(torch.device("cpu"))
    arena_best_model.eval()

    _loaded_candidate_sig: Optional[tuple[Path, int, int]] = None
    _loaded_best_sig: Optional[tuple[Path, int, int]] = None
    _warned_service_callback_bridge = False

    def _weights_sig(path: Path) -> tuple[Path, int, int]:
        # Path alone is not enough (best.pt path stays constant while contents change).
        rp = path.resolve()
        try:
            st = path.stat()
            return (rp, int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            return (rp, -1, -1)



    while True:

        task = task_q.get()
        if task is None:
            break

        task_kind = str(task.get("task_kind", "selfplay"))
        try:
            if task_kind == "selfplay":
                # Unpack task
                gen = int(task["gen"])
                round_idx = int(task.get("round_idx", 0))
                out_dir = Path(task["out_dir"])
                weights_path = Path(task["weights_path"])
                game_id = int(task["game_id"])
                start_root_id = int(task["start_root_id"])
                history_seed = int(task["history_seed"])
                roots = int(task["num_roots"])
                sample_from_policy = bool(task.get("sample_from_mcts_policy", False))
                policy_temp = float(task.get("selfplay_policy_temperature", 1.0))
                action_seed_base = int(task.get("action_seed_base", task.get("history_seed", 0)))

                run_model = model

                # Load frozen weights for this generation
                ckpt = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(ckpt["model_state"])
                model.eval()

                infer_mode = str(task.get("infer_mode", "python"))
                if safe_multi and infer_mode == "torchscript_cpp":
                    infer_mode = "python"
                native_mcts_on = bool(getattr(explore_cfg, "native_mcts_enabled", False))
                if (
                    infer_mode == "torchscript_service"
                    and native_mcts_on
                    and native_service_runtime is not None
                    and NativeTorchScriptModelAdapter is not None
                ):
                    ts_controller_raw = str(task.get("ts_controller_path", "")).strip()
                    ts_adversary_raw = str(task.get("ts_adversary_path", "")).strip()
                    if not ts_controller_raw or not ts_adversary_raw:
                        raise RuntimeError("torchscript_service task missing ts_controller_path/ts_adversary_path")
                    model_version = int(task["model_version"])
                    native_service_runtime.load_models(
                        int(model_version),
                        str(ts_controller_raw),
                        str(ts_adversary_raw),
                    )
                    run_model = NativeTorchScriptModelAdapter(
                        runtime=native_service_runtime,
                        model_version=int(model_version),
                        fallback_model=model,
                        fallback_to_python=bool(task.get("fallback_to_python_infer", True)),
                    )
                elif (
                    infer_mode == "torchscript_service"
                    and infer_client is not None
                    and TorchScriptModelPaths is not None
                    and TorchScriptServiceModelAdapter is not None
                ):
                    ts_controller_raw = str(task.get("ts_controller_path", "")).strip()
                    ts_adversary_raw = str(task.get("ts_adversary_path", "")).strip()
                    if not ts_controller_raw or not ts_adversary_raw:
                        raise RuntimeError("torchscript_service task missing ts_controller_path/ts_adversary_path")
                    ts_controller = Path(ts_controller_raw)
                    ts_adversary = Path(ts_adversary_raw)
                    model_version = int(task["model_version"])
                    infer_client.ensure_models_loaded(
                        TorchScriptModelPaths(
                            controller_path=str(ts_controller),
                            adversary_path=str(ts_adversary),
                            model_version=int(model_version),
                        )
                    )
                    run_model = TorchScriptServiceModelAdapter(
                        client=infer_client,
                        model_version=int(model_version),
                        fallback_model=model,
                        fallback_to_python=bool(task.get("fallback_to_python_infer", True)),
                    )
                    if native_mcts_on and not _warned_service_callback_bridge:
                        print(
                            f"[worker {worker_id}] torchscript_service fallback path uses "
                            "C++->Python callback per simulation (high overhead).",
                            flush=True,
                        )
                        _warned_service_callback_bridge = True
                elif (
                    infer_mode == "torchscript_cpp"
                    and native_ts_runtime is not None
                    and NativeTorchScriptModelAdapter is not None
                ):
                    ts_controller_raw = str(task.get("ts_controller_path", "")).strip()
                    ts_adversary_raw = str(task.get("ts_adversary_path", "")).strip()
                    if not ts_controller_raw or not ts_adversary_raw:
                        raise RuntimeError("torchscript_cpp task missing ts_controller_path/ts_adversary_path")
                    model_version = int(task["model_version"])
                    native_ts_runtime.load_models(
                        int(model_version),
                        str(ts_controller_raw),
                        str(ts_adversary_raw),
                    )
                    run_model = NativeTorchScriptModelAdapter(
                        runtime=native_ts_runtime,
                        model_version=int(model_version),
                        fallback_model=model,
                        fallback_to_python=bool(task.get("fallback_to_python_infer", True)),
                    )

                # Per-worker writer
                writer = ReplayWriter(ReplayWriterConfig(out_dir=out_dir, shard_size=cfg.dataset.shard_size))

                # Put per-worker logs in the global logs dir (not inside dataset dir)
                logs_base = Path(cfg.logging.mcts_iter_log).parent  # simulator_output/mcts_dnn_logs
                logs_dir = logs_base / f"gen_{gen:06d}"
                logs_dir.mkdir(parents=True, exist_ok=True)

                iter_log = logs_dir / f"mcts_iter_p{worker_id:02d}.csv"
                root_log = logs_dir / f"mcts_root_p{worker_id:02d}_gen{round_idx:02d}.csv"

                mcts = VidurMCTS(
                    env=env,
                    explore_cfg=explore_cfg,
                    log_path=iter_log,
                    tree_log_path=root_log,
                    logger_flush_every=cfg.logging.flush_every,
                )
                runner = SelfPlayRunner(
                    env=env,
                    mcts=mcts,
                    model=run_model,
                    writer=writer,
                    device_for_features=torch.device("cpu"),
                )

                hops_for_task = int(task.get("history_nontrivial_hops", default_history_nontrivial_hops))
                runner.run_n_roots(
                    game_id=game_id,
                    num_roots=roots,
                    adv_iterations_per_root=adv_iterations_per_root,
                    cont_iterations_per_root=cont_iterations_per_root,
                    max_batch_size=max_batch_size,
                    start_root_id=start_root_id,
                    start_root_depth=0,
                    start_player="adversary",
                    history_nontrivial_hops=hops_for_task,
                    feature_version=cfg.run.feature_version,
                    history_seed=history_seed,
                    sample_from_mcts_policy=sample_from_policy,
                    selfplay_policy_temperature=policy_temp,
                    action_seed_base=action_seed_base,
                    max_forced_hops_per_root=int(task.get("max_forced_hops_per_root", 1024)),
                    history_max_total_steps=int(task.get("history_max_total_steps", 20000)),
                )

                writer.close()
                mcts.close()

                result_q.put(
                    {
                        "ok": True,
                        "task_kind": "selfplay",
                        "worker_id": worker_id,
                        "gen": gen,
                        "round_idx": round_idx,
                        "out_dir": str(out_dir),
                    }
                )
                continue

            if task_kind == "arena":
                gen = int(task["gen"])
                arena_roots = list(task["arena_roots"])
                arena_games_dir = Path(task["arena_games_dir"])

                candidate_weights_path = Path(task["candidate_weights_path"])
                best_weights_path = Path(task["best_weights_path"])

                candidate_sig = _weights_sig(candidate_weights_path)
                if _loaded_candidate_sig != candidate_sig:
                    ckpt_c = torch.load(candidate_weights_path, map_location="cpu")
                    arena_candidate_model.load_state_dict(extract_model_state(ckpt_c), strict=True)
                    arena_candidate_model.eval()
                    _loaded_candidate_sig = candidate_sig

                best_sig = _weights_sig(best_weights_path)
                if _loaded_best_sig != best_sig:
                    ckpt_b = torch.load(best_weights_path, map_location="cpu")
                    arena_best_model.load_state_dict(extract_model_state(ckpt_b), strict=True)
                    arena_best_model.eval()
                    _loaded_best_sig = best_sig

                candidate_model_for_arena = arena_candidate_model
                best_model_for_arena = arena_best_model

                infer_mode = str(task.get("infer_mode", "python"))
                if safe_multi and infer_mode == "torchscript_cpp":
                    infer_mode = "python"
                native_mcts_on = bool(getattr(explore_cfg, "native_mcts_enabled", False))
                if (
                    infer_mode == "torchscript_service"
                    and native_mcts_on
                    and native_service_runtime is not None
                    and NativeTorchScriptModelAdapter is not None
                ):
                    required_keys = [
                        "candidate_ts_controller_path",
                        "candidate_ts_adversary_path",
                        "best_ts_controller_path",
                        "best_ts_adversary_path",
                    ]
                    missing = [k for k in required_keys if not str(task.get(k, "")).strip()]
                    if missing:
                        raise RuntimeError(f"torchscript_service arena task missing fields: {missing}")
                    candidate_model_version = int(task["candidate_model_version"])
                    best_model_version = int(task["best_model_version"])
                    native_service_runtime.load_models(
                        candidate_model_version,
                        str(task["candidate_ts_controller_path"]),
                        str(task["candidate_ts_adversary_path"]),
                    )
                    native_service_runtime.load_models(
                        best_model_version,
                        str(task["best_ts_controller_path"]),
                        str(task["best_ts_adversary_path"]),
                    )

                    fallback_to_python = bool(task.get("fallback_to_python_infer", True))
                    candidate_model_for_arena = NativeTorchScriptModelAdapter(
                        runtime=native_service_runtime,
                        model_version=candidate_model_version,
                        fallback_model=arena_candidate_model,
                        fallback_to_python=fallback_to_python,
                    )
                    best_model_for_arena = NativeTorchScriptModelAdapter(
                        runtime=native_service_runtime,
                        model_version=best_model_version,
                        fallback_model=arena_best_model,
                        fallback_to_python=fallback_to_python,
                    )
                elif (
                    infer_mode == "torchscript_service"
                    and infer_client is not None
                    and TorchScriptModelPaths is not None
                    and TorchScriptServiceModelAdapter is not None
                ):
                    required_keys = [
                        "candidate_ts_controller_path",
                        "candidate_ts_adversary_path",
                        "best_ts_controller_path",
                        "best_ts_adversary_path",
                    ]
                    missing = [k for k in required_keys if not str(task.get(k, "")).strip()]
                    if missing:
                        raise RuntimeError(f"torchscript_service arena task missing fields: {missing}")
                    cand_paths = TorchScriptModelPaths(
                        controller_path=str(task["candidate_ts_controller_path"]),
                        adversary_path=str(task["candidate_ts_adversary_path"]),
                        model_version=int(task["candidate_model_version"]),
                    )
                    best_paths = TorchScriptModelPaths(
                        controller_path=str(task["best_ts_controller_path"]),
                        adversary_path=str(task["best_ts_adversary_path"]),
                        model_version=int(task["best_model_version"]),
                    )
                    infer_client.ensure_models_loaded(cand_paths)
                    infer_client.ensure_models_loaded(best_paths)

                    fallback_to_python = bool(task.get("fallback_to_python_infer", True))
                    candidate_model_for_arena = TorchScriptServiceModelAdapter(
                        client=infer_client,
                        model_version=int(cand_paths.model_version),
                        fallback_model=arena_candidate_model,
                        fallback_to_python=fallback_to_python,
                    )
                    best_model_for_arena = TorchScriptServiceModelAdapter(
                        client=infer_client,
                        model_version=int(best_paths.model_version),
                        fallback_model=arena_best_model,
                        fallback_to_python=fallback_to_python,
                    )
                    if native_mcts_on and not _warned_service_callback_bridge:
                        print(
                            f"[worker {worker_id}] torchscript_service fallback path uses "
                            "C++->Python callback per simulation (high overhead).",
                            flush=True,
                        )
                        _warned_service_callback_bridge = True
                elif (
                    infer_mode == "torchscript_cpp"
                    and native_ts_runtime is not None
                    and NativeTorchScriptModelAdapter is not None
                ):
                    required_keys = [
                        "candidate_ts_controller_path",
                        "candidate_ts_adversary_path",
                        "best_ts_controller_path",
                        "best_ts_adversary_path",
                    ]
                    missing = [k for k in required_keys if not str(task.get(k, "")).strip()]
                    if missing:
                        raise RuntimeError(f"torchscript_cpp arena task missing fields: {missing}")

                    candidate_model_version = int(task["candidate_model_version"])
                    best_model_version = int(task["best_model_version"])

                    native_ts_runtime.load_models(
                        candidate_model_version,
                        str(task["candidate_ts_controller_path"]),
                        str(task["candidate_ts_adversary_path"]),
                    )
                    native_ts_runtime.load_models(
                        best_model_version,
                        str(task["best_ts_controller_path"]),
                        str(task["best_ts_adversary_path"]),
                    )

                    fallback_to_python = bool(task.get("fallback_to_python_infer", True))
                    candidate_model_for_arena = NativeTorchScriptModelAdapter(
                        runtime=native_ts_runtime,
                        model_version=candidate_model_version,
                        fallback_model=arena_candidate_model,
                        fallback_to_python=fallback_to_python,
                    )
                    best_model_for_arena = NativeTorchScriptModelAdapter(
                        runtime=native_ts_runtime,
                        model_version=best_model_version,
                        fallback_model=arena_best_model,
                        fallback_to_python=fallback_to_python,
                    )

                adv_iters = int(task["adv_iterations_per_root"])
                cont_iters = int(task["cont_iterations_per_root"])
                max_adv_moves = int(task["arena_max_adversary_moves"])
                max_cleanup = int(task["arena_max_controller_cleanup_steps"])
                max_turns = int(task["arena_max_total_turns"])
                feature_version = int(task["feature_version"])
                tie_points = float(task["tie_points"])

                noop_writer = NoopReplayWriter()

                for entry in arena_roots:
                    game_id = int(entry["game_id"])
                    history_hops = int(entry["history_hops"])
                    start_player = str(entry.get("player_to_act", "adversary"))
                    start_depth = int(entry.get("depth", 0))
                    history_root_id = int(entry.get("root_id", 0))

                    # debug full log (single file with both cycles)
                    debug_root_log = arena_games_dir / f"game_{game_id}.csv"

                    mcts = VidurMCTS(
                        env=env,
                        explore_cfg=explore_cfg,
                        log_path=None,
                        tree_log_path=debug_root_log,
                        logger_flush_every=cfg.logging.flush_every,
                    )
                    runner = SelfPlayRunner(
                        env=env,
                        mcts=mcts,
                        model=candidate_model_for_arena,
                        writer=noop_writer,
                        device_for_features=torch.device("cpu"),
                    )

                    try:
                        out = runner.run_arena_game(
                            game_id=game_id,
                            candidate_model=candidate_model_for_arena,
                            best_model=best_model_for_arena,
                            history_nontrivial_hops=history_hops,
                            adv_iterations_per_root=adv_iters,
                            cont_iterations_per_root=cont_iters,
                            arena_max_adversary_moves=max_adv_moves,
                            arena_max_controller_cleanup_steps=max_cleanup,
                            arena_max_total_turns=max_turns,
                            feature_version=feature_version,
                            tie_points=tie_points,
                            start_player=start_player,
                            start_root_depth=start_depth,
                            history_root_id_for_logs=history_root_id,
                        )
                    finally:
                        mcts.close()

                    cycle_a = dict(out["cycle_a"])
                    cycle_b = dict(out["cycle_b"])

                    write_arena_cycle_end_csv(
                        arena_games_dir / f"game_{game_id}_adv_candidate_ctrl_best.csv",
                        game_id=game_id,
                        cycle_label="candidate_as_adversary",
                        total_cost=float(out["candidate_as_adv_cost"]),
                        slo_violations=int(cycle_a.get("slo_violations", 0)),
                        total_lateness=float(cycle_a.get("total_lateness", 0.0)),
                    )
                    write_arena_cycle_end_csv(
                        arena_games_dir / f"game_{game_id}_adv_best_ctrl_candidate.csv",
                        game_id=game_id,
                        cycle_label="best_as_adversary",
                        total_cost=float(out["best_as_adv_cost"]),
                        slo_violations=int(cycle_b.get("slo_violations", 0)),
                        total_lateness=float(cycle_b.get("total_lateness", 0.0)),
                    )

                result_q.put(
                    {
                        "ok": True,
                        "task_kind": "arena",
                        "worker_id": worker_id,
                        "gen": gen,
                        "num_games": len(arena_roots),
                    }
                )
                continue

            result_q.put(
                {
                    "ok": False,
                    "task_kind": str(task_kind),
                    "worker_id": worker_id,
                    "error": f"unknown task_kind={task_kind}",
                }
            )
        except Exception as exc:
            result_q.put(
                {
                    "ok": False,
                    "task_kind": str(task_kind),
                    "worker_id": worker_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

def selfImprovementPolicy(
    *,
    cfg: "AlphaZeroConfig",
    evaluator_cfg: EvaluatorConfig,
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
    use_virtual_env: bool = False,
    replay_capacity_samples: int = 12_000,
    replay_max_cached_shards: int = 8,
    replay_seed: int = 2026,
) -> None:



    def _sample_worker_hops_for_generation(gen: int) -> list[int]:
        rng = random.Random(91337 + int(gen))
        max_hops = int(evaluator_cfg.max_history_depth)
        n_workers = int(num_selfPlay_workers)

        hops = [int(rng.randint(0, max_hops)) for _ in range(n_workers)]
        if n_workers > 0:
            hops[0] = 0  # always assign 0-hop history to proc 0
        return hops


    best_path = ckpt_dir / "best.pt"

    trainer = Trainer(
        model=model,
        cfg=TrainerConfig(
            lr=3e-4,
            weight_decay=2e-4,
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
            "arena_candidate_points": "",
            "arena_best_points": "",
            "arena_total_points": "",
            "arena_candidate_win_rate": "",
            "arena_win_threshold": "",
            "arena_passed": "",
        },
    )

    sim_cli_args = list(getattr(cfg.sim, "cli_args"))
    if not best_path.exists():
        trainer.save_checkpoint(best_path)    


    replay_buffer = BestModelReplayBuffer(
        capacity_samples=int(replay_capacity_samples),
        max_cached_shards=int(replay_max_cached_shards),
        seed=int(replay_seed),
    )

    

    replay_log_csv = train_log_csv.parent / "replay_logs.csv"
    replay_logger = ReplayBufferLogger(replay_log_csv, flush_every=1)
    best_model_generation = _infer_best_generation_from_train_log(train_log_csv)

    boot_gens, boot_raw = _bootstrap_replay_from_train_log(
        replay_buffer=replay_buffer,
        train_log_csv=train_log_csv,
        best_generation=best_model_generation,
        )

    ## Limiting threads :
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    infer_service_proc = None
    infer_mode = str(getattr(cfg.native, "infer_mode", "python"))
    using_infer_service = _infer_service_enabled(cfg)
    using_cpp_runtime_infer = _infer_cpp_runtime_enabled(cfg)
    export_ts_artifacts = (
        bool(getattr(cfg.native, "export_torchscript", False))
        or using_infer_service
        or using_cpp_runtime_infer
    )

    if using_infer_service:
        if start_torchscript_infer_service is None:
            raise RuntimeError("infer_mode=torchscript_service requested but infer service entrypoint is unavailable")
        infer_service_proc = start_torchscript_infer_service(
            addr=str(cfg.native.infer_service_addr),
            device=str(cfg.native.infer_service_device),
            max_batch=int(cfg.native.infer_max_batch),
            max_wait_us=int(cfg.native.infer_max_wait_us),
            impl=str(getattr(cfg.native, "infer_service_impl", "cpp")),
        )
        _wait_for_infer_service(str(cfg.native.infer_service_addr), timeout_s=45.0)


    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()

    # def _hops_for_worker(wid: int) -> int:
    #     if isinstance(history_nontrivial_hops, int):
    #         return int(history_nontrivial_hops)
    #     hops_list = list(history_nontrivial_hops)
    #     if len(hops_list) != int(num_selfPlay_workers):
    #         raise ValueError(
    #             f"history_nontrivial_hops must have len == num_selfPlay_workers "
    #             f"({len(hops_list)} != {int(num_selfPlay_workers)})"
    #         )
    #     return int(hops_list[wid])


    def _hops_for_worker(wid: int) -> int:
        if isinstance(history_nontrivial_hops, int):
            return int(history_nontrivial_hops)

        hops_list = list(history_nontrivial_hops)
        if not hops_list:
            return 0

        # No strict length requirement; repeat cyclically if workers > provided hops.
        return int(hops_list[wid % len(hops_list)])


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
                use_virtual_env,
                int(num_selfPlay_workers),
            ),
        )
        p.start()
        workers.append(p)

    def _collect_results(
        *,
        expected: int,
        task_kind: str,
        gen: int,
        round_idx: int | None = None,
        poll_timeout_s: float = 30.0,
    ) -> list[dict]:
        msgs: list[dict] = []
        wait_loops = 0
        while len(msgs) < int(expected):
            try:
                msg = result_q.get(timeout=float(poll_timeout_s))
            except queue.Empty:
                dead = [p for p in workers if p.exitcode not in (None, 0)]
                if dead:
                    details = ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in dead)
                    raise RuntimeError(
                        f"{task_kind} wait failed: worker process died while waiting for results: {details}"
                    )
                wait_loops += 1
                if (wait_loops % 2) == 0:
                    alive = sum(1 for p in workers if p.is_alive())
                    ridx = f", round={round_idx}" if round_idx is not None else ""
                    print(
                        f"[main] waiting for {task_kind} results gen={gen}{ridx}: "
                        f"received={len(msgs)}/{expected}, alive_workers={alive}",
                        flush=True,
                    )
                continue

            msg_kind = str(msg.get("task_kind", ""))
            if msg_kind != str(task_kind):
                raise RuntimeError(
                    f"received unexpected task result kind='{msg_kind}' while waiting for '{task_kind}': {msg}"
                )
            if not bool(msg.get("ok", False)):
                tb = str(msg.get("traceback", "")).strip()
                err = str(msg.get("error", "unknown error"))
                wid = msg.get("worker_id", "?")
                if tb:
                    raise RuntimeError(f"{task_kind} worker {wid} failed: {err}\n{tb}")
                raise RuntimeError(f"{task_kind} worker {wid} failed: {err}")
            msgs.append(msg)
        return msgs


    dataset_base = Path(cfg.dataset.out_dir)
    gen0 = _next_generation_index(dataset_base)

    for j in range(int(num_generations)):

        # Capturing the best gen model weights :
        best_state_cpu = _load_ckpt_model_state_cpu(best_path)

        gen = gen0 + j
        gen_dataset_dir = dataset_base / f"gen_{gen:06d}"
        logs_base = Path(cfg.logging.mcts_iter_log).parent
        gen_logs_dir = logs_base / f"gen_{gen:06d}"
        gen_logs_dir.mkdir(parents=True, exist_ok=True)

        worker_hops_for_gen = _sample_worker_hops_for_generation(gen)

        # 0) freeze current weights for self-play workers
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        weights_path = ckpt_dir / f"selfplay_weights_gen_{gen:06d}.pt"
        state = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
        torch.save({"model_state": state}, weights_path)

        selfplay_model_version = int(gen) * 100 + 1
        selfplay_ts_controller_path = None
        selfplay_ts_adversary_path = None
        if export_ts_artifacts:
            ts_controller, ts_adversary, _ = _export_torchscript_artifacts_for_checkpoint(
                checkpoint_path=weights_path,
                out_dir=ckpt_dir,
                model_version=selfplay_model_version,
                device="cpu",
                num_actions_controller=int(cfg.model.num_actions_controller),
                num_actions_adversary=int(cfg.model.num_actions_adversary),
            )
            selfplay_ts_controller_path = ts_controller
            selfplay_ts_adversary_path = ts_adversary


        # split roots across workers
        # per = int(math.ceil(roots_per_generation / float(num_selfPlay_workers)))
        # tasks_sent = 0
        # for wid in range(int(num_selfPlay_workers)):
        #     start = wid * per
        #     end = min(roots_per_generation, (wid + 1) * per)
        #     n_roots = max(0, end - start)
        #     if n_roots == 0:
        #         continue

        #     out_dir = gen_dataset_dir / f"proc_{wid:02d}"
        #     # ensure unique game_id per worker so history randomness differs
        #     worker_game_id = int(cfg.run.game_id) + gen * 1000 + wid
        #     # ensure root_id ranges don’t collide (optional, but helps debugging)
        #     worker_start_root_id = start

        #     task_q.put(
        #         {
        #             "gen": gen,
        #             "out_dir": str(out_dir),
        #             "weights_path": str(weights_path),
        #             "game_id": worker_game_id,
        #             "start_root_id": worker_start_root_id,
        #             "history_seed": (gen * 100000 + wid),
        #             "num_roots": n_roots,
        #             "history_nontrivial_hops": int(worker_hops_for_gen[wid]),
        #         }
        #     )
        #     tasks_sent += 1


        # Run self-play collection multiple times per generation (same frozen best/candidate weights)
        arena_per_gen = max(1, int(getattr(evaluator_cfg, "arena_per_gen", 1)))
        all_results = []

        for round_idx in range(arena_per_gen):
            per = int(math.ceil(roots_per_generation / float(num_selfPlay_workers)))
            tasks_sent = 0

            for wid in range(int(num_selfPlay_workers)):
                start = wid * per
                end = min(roots_per_generation, (wid + 1) * per)
                n_roots = max(0, end - start)
                if n_roots == 0:
                    continue

                out_dir = gen_dataset_dir / f"proc_{wid:02d}"

                # keep ids/seeds unique across rounds
                worker_game_id = int(cfg.run.game_id) + gen * 1_000_000 + round_idx * 10_000 + wid
                worker_start_root_id = round_idx * int(roots_per_generation) + start

                task_q.put(
                    {
                        "gen": gen,
                        "round_idx": int(round_idx),
                        "out_dir": str(out_dir),
                        "weights_path": str(weights_path),
                        "game_id": worker_game_id,
                        "start_root_id": worker_start_root_id,
                        "history_seed": (gen * 1_000_000 + round_idx * 1_000 + wid),
                        "num_roots": n_roots,
                        "history_nontrivial_hops": int(worker_hops_for_gen[wid]),
                        "sample_from_mcts_policy": bool(getattr(cfg.run, "sample_from_mcts_policy", False)),
                        "selfplay_policy_temperature": float(getattr(cfg.run, "selfplay_policy_temperature", 1.0)),
                        "action_seed_base":(gen * 1_000_000 + round_idx * 1_000 + wid) ,
                        "max_forced_hops_per_root": int(getattr(cfg.run, "max_forced_hops_per_root", 1024)),
                        "history_max_total_steps": int(getattr(cfg.run, "history_max_total_steps", 20000)),
                        "infer_mode": str(infer_mode),
                        "fallback_to_python_infer": bool(getattr(cfg.native, "fallback_to_python_infer", True)),
                        "model_version": int(selfplay_model_version),
                        "ts_controller_path": str(selfplay_ts_controller_path) if selfplay_ts_controller_path else "",
                        "ts_adversary_path": str(selfplay_ts_adversary_path) if selfplay_ts_adversary_path else "",
                    }
                )
                tasks_sent += 1

            if tasks_sent > 0:
                all_results.extend(
                    _collect_results(
                        expected=tasks_sent,
                        task_kind="selfplay",
                        gen=int(gen),
                        round_idx=int(round_idx),
                    )
                )

        # # 2) wait for all worker results
        # results = []
        # for _ in range(tasks_sent):
        #     results.append(result_q.get())


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

        added_from_gen = replay_buffer.add_generation_dir(gen_dataset_dir)
        if replay_buffer.total_samples <= 0:
            raise RuntimeError(
                f"Replay buffer empty after gen={gen}. "
                f"added_from_gen={added_from_gen}, gen_dir={gen_dataset_dir}"
            )

        num_controller = sum(1 for s in samples if s.get("player") == "controller")
        num_adversary = sum(1 for s in samples if s.get("player") == "adversary")


        # Eval on full latest-generation dataset (stable metric)
        eval_batch = collate_mixed_samples(samples, device=trainer.device)

        # Train with random minibatches from replay buffer
        train_minibatch_size = 32
        replay_buffer.reseed(1000 + int(gen))  # deterministic per generation

        # Dynamic train steps: base_steps * round(replay_size / roots_per_generation)
        ratio = float(replay_buffer.total_samples) / float(max(1, int(roots_per_generation)))
        step_multiplier = max(1, int(math.floor(ratio + 0.5)))  # nearest positive integer
        effective_train_steps = int(train_steps_per_generation) * step_multiplier

        replay_gen_ids_csv = replay_buffer.active_generation_ids_csv(max_items=256)
        replay_logger.log_row(
            candidate_generation=int(gen),
            best_model_generation=int(best_model_generation),
            replay_samples_size=int(replay_buffer.total_samples),
            effective_training_steps=int(effective_train_steps),
            replay_generation_ids=replay_gen_ids_csv,
        )

        for _ in range(int(effective_train_steps)):
            minibatch_samples = replay_buffer.sample_batch(train_minibatch_size)
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
                    "num_samples": int(replay_buffer.total_samples),  # now training pool size
                    "num_controller": int(num_controller),
                    "num_adversary": int(num_adversary),
                    **train_metrics,
                    "saved_best": "",
                    "ckpt_path": "",
                    "best_path": str(best_path),
                    "resume_ckpt": str(resume_ckpt) if resume_ckpt else "",
                },
            )




        # 4) train-set eval (keep your existing behavior)
        eval_metrics = trainer.eval_step(eval_batch)
        _append_train_log_row(
            train_log_csv,
            {
                "time": time.time(),
                "event": "eval_trainset",
                "gen": gen,
                "trainer_step": int(trainer.step),
                "dataset_dir": str(gen_dataset_dir),
                "num_samples": int(len(samples)),
                "num_controller": int(num_controller),
                "num_adversary": int(num_adversary),
                **eval_metrics,
                "saved_best": "",
                "ckpt_path": "",
                "best_path": str(best_path),
                "resume_ckpt": str(resume_ckpt) if resume_ckpt else "",
            },
        )

         # Save candidate checkpoint (pre-arena decision)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ckpt_gen_{gen:06d}_step_{trainer.step:06d}.pt"
        trainer.save_checkpoint(ckpt_path)

        arena_candidate_version = int(gen) * 100 + 2
        arena_best_version = int(gen) * 100 + 3
        candidate_ts_controller_path = None
        candidate_ts_adversary_path = None
        best_ts_controller_path = None
        best_ts_adversary_path = None
        if export_ts_artifacts:
            c_ctrl, c_adv, _ = _export_torchscript_artifacts_for_checkpoint(
                checkpoint_path=ckpt_path,
                out_dir=ckpt_dir,
                model_version=arena_candidate_version,
                device="cpu",
                num_actions_controller=int(cfg.model.num_actions_controller),
                num_actions_adversary=int(cfg.model.num_actions_adversary),
            )
            b_ctrl, b_adv, _ = _export_torchscript_artifacts_for_checkpoint(
                checkpoint_path=best_path,
                out_dir=ckpt_dir,
                model_version=arena_best_version,
                device="cpu",
                num_actions_controller=int(cfg.model.num_actions_controller),
                num_actions_adversary=int(cfg.model.num_actions_adversary),
            )
            candidate_ts_controller_path = c_ctrl
            candidate_ts_adversary_path = c_adv
            best_ts_controller_path = b_ctrl
            best_ts_adversary_path = b_adv

        # 5) arena evaluation against current best 
   
        arena_root_entries = _build_arena_root_entries(gen, evaluator_cfg, cfg)
        if not arena_root_entries:
            raise RuntimeError("arena root payload is empty")

        arena_games_dir = gen_logs_dir / "arena_games"
        arena_games_dir.mkdir(parents=True, exist_ok=True)

        per_arena = int(math.ceil(len(arena_root_entries) / float(num_selfPlay_workers)))
        arena_tasks_sent = 0

        for wid in range(int(num_selfPlay_workers)):
            lo = wid * per_arena
            hi = min(len(arena_root_entries), (wid + 1) * per_arena)
            if hi <= lo:
                continue

            task_q.put(
                {
                    "task_kind": "arena",
                    "gen": gen,
                    "arena_roots": arena_root_entries[lo:hi],
                    "arena_games_dir": str(arena_games_dir),
                    "candidate_weights_path": str(ckpt_path),
                    "best_weights_path": str(best_path),
                    "adv_iterations_per_root": int(evaluator_cfg.adv_iterations_per_root),
                    "cont_iterations_per_root": int(evaluator_cfg.cont_iterations_per_root),
                    "arena_max_adversary_moves": int(evaluator_cfg.arena_max_adversary_moves),
                    "arena_max_controller_cleanup_steps": int(evaluator_cfg.arena_max_controller_cleanup_steps),
                    "arena_max_total_turns": int(evaluator_cfg.arena_max_total_turns),
                    "feature_version": int(cfg.run.feature_version),
                    "tie_points": float(evaluator_cfg.tie_points),
                    "infer_mode": str(infer_mode),
                    "fallback_to_python_infer": bool(getattr(cfg.native, "fallback_to_python_infer", True)),
                    "candidate_model_version": int(arena_candidate_version),
                    "best_model_version": int(arena_best_version),
                    "candidate_ts_controller_path": str(candidate_ts_controller_path) if candidate_ts_controller_path else "",
                    "candidate_ts_adversary_path": str(candidate_ts_adversary_path) if candidate_ts_adversary_path else "",
                    "best_ts_controller_path": str(best_ts_controller_path) if best_ts_controller_path else "",
                    "best_ts_adversary_path": str(best_ts_adversary_path) if best_ts_adversary_path else "",
                }
            )
            arena_tasks_sent += 1

        arena_msgs = _collect_results(
            expected=arena_tasks_sent,
            task_kind="arena",
            gen=int(gen),
            round_idx=None,
        )

        arena_metrics = grade_arena_from_game_logs(
            game_log_dir=arena_games_dir,
            out_csv=gen_logs_dir / "arena_results.csv",
            tie_points=float(evaluator_cfg.tie_points),
            win_threshold=float(evaluator_cfg.arena_win_threshold),
        )
        arena_passed = bool(arena_metrics["passed"])

        if arena_passed:
            trainer.save_checkpoint(best_path)
            replay_buffer.reset_for_new_best()
            best_model_generation = int(gen)
        else:
            _restore_trainer_from_ckpt(trainer=trainer, path=best_path)

        _append_train_log_row(
            train_log_csv,
            {
                "time": time.time(),
                "event": "eval_arena",
                "gen": gen,
                "trainer_step": int(trainer.step),
                "dataset_dir": str(gen_dataset_dir),
                "num_samples": int(len(samples)),
                "num_controller": int(num_controller),
                "num_adversary": int(num_adversary),
                "saved_best": bool(arena_passed),
                "ckpt_path": str(ckpt_path),
                "best_path": str(best_path),
                "resume_ckpt": str(resume_ckpt) if resume_ckpt else "",
                "arena_candidate_points": float(arena_metrics["candidate_points"]),
                "arena_best_points": float(arena_metrics["best_points"]),
                "arena_total_points": float(arena_metrics["total_points"]),
                "arena_candidate_win_rate": float(arena_metrics["candidate_win_rate"]),
                "arena_win_threshold": float(arena_metrics["arena_win_threshold"]),
                "arena_passed": bool(arena_passed),
            },
        )       

    replay_logger.close() 
    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"selfplay worker died: pid={p.pid} exitcode={p.exitcode}")

    if infer_service_proc is not None:
        if TorchScriptInferClient is not None:
            try:
                c = TorchScriptInferClient(addr=str(cfg.native.infer_service_addr))
                c.shutdown()
                c.close()
            except Exception:
                pass
        infer_service_proc.terminate()
        infer_service_proc.join(timeout=5.0)



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
    root_dirichlet_noise_enabled: bool = False
    root_dirichlet_alpha: float = 0.6
    root_dirichlet_epsilon: float = 0.25


@dataclass(frozen=True)
class ModelGroup:
    num_actions_controller: int = 24
    num_actions_adversary: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class NativeRuntimeGroup:
    enabled: bool = False
    backend: str = "python"  # python | cpp_virtual
    strict_compat: bool = True
    multiprocess_safety_mode: bool = False
    native_trace: bool = False
    native_trace_every: int = 500
    native_trace_dir: str = "simulator_output/mcts_dnn_logs"
    infer_mode: str = "python"  # python | torchscript_service | torchscript_cpp
    infer_service_impl: str = "cpp"  # cpp | python
    torchscript_full_native_search: bool = False
    infer_service_addr: str = "127.0.0.1:50201"
    infer_service_device: str = "cuda:0"
    infer_max_batch: int = 256
    infer_max_wait_us: int = 2000
    export_torchscript: bool = False
    fallback_to_python_infer: bool = True


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
    sample_from_mcts_policy: bool = False
    selfplay_policy_temperature: float = 1.0
    max_forced_hops_per_root: int = 1024
    history_max_total_steps: int = 20000


@dataclass(frozen=True)
class AlphaZeroConfig:
    sim: SimulationCLIGroup
    constraints: MCTSConstraintsGroup
    explore: MCTSExploreGroup
    model: ModelGroup
    logging: LoggingGroup
    dataset: DatasetGroup
    run: RunGroup
    native: NativeRuntimeGroup = dc_field(default_factory=NativeRuntimeGroup)


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
            controller_min_prior_threshold=0.01,
            adversary_min_prior_threshold=0.1,
            root_dirichlet_noise_enabled=True,
            root_dirichlet_alpha=0.6,
            root_dirichlet_epsilon=0.35,
        ),
        model=ModelGroup(
            num_actions_controller=24,
            num_actions_adversary=6,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        native=NativeRuntimeGroup(
            enabled=True,
            backend="cpp_virtual",  # python | cpp_virtual
            strict_compat=True,
            multiprocess_safety_mode=False,
            native_trace=False,
            native_trace_every=250,
            native_trace_dir="simulator_output/mcts_dnn_logs",
            infer_mode="torchscript_service",
            infer_service_impl="cpp",
            torchscript_full_native_search=True,
            infer_service_addr="127.0.0.1:50201",
            infer_service_device="cuda:0",
            infer_max_batch=256,
            infer_max_wait_us=500,
            export_torchscript=False,
            fallback_to_python_infer=False,
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
            sample_from_mcts_policy=False,
            selfplay_policy_temperature=0,
            max_forced_hops_per_root=512,
            history_max_total_steps=12000,
        ),
    )


    # TODO: Remove useless feilds and move the config class to eval_utils
    evaluator_cfg = EvaluatorConfig(
        num_random_games=10,
        max_history_depth=10,  # no history for now (can add later if you want to test it in the arena)
        random_seed_base=12345,
        adv_iterations_per_root=4000,
        cont_iterations_per_root=10000,
        arena_num_processes=10,
        # arena_iters_adversary=2000,
        # arena_iters_controller=2000,
        arena_max_adversary_moves=1,
        arena_max_controller_cleanup_steps=128,
        arena_max_total_turns=512,
        arena_win_threshold=0.52,
        tie_points=0.5,
        debug_sample_games=5,
        debug_flush_every=1,
        arena_per_gen=1,
    )



    ## MODEL TRAINING PARMS FOR SELF-IMPROVEMENT LOOP:
    num_selfPlay_workers = 15
    num_generations = 200
    history_nontrivial_hops = [0, 5 , 10 , 15 , 20, 25, 30, 35]  # per worker
    roots_per_generation = 500
    adv_iterations_per_root = 4000
    cont_iterations_per_root = 12000
    train_steps_per_generation = 75 
    max_batch_size = 256
    train_log_csv = Path("simulator_output/mcts_dnn_logs/train_metrics.csv")
    ckpt_dir = Path("simulator_output/mcts_dnn_checkpoints")
    use_virtual_env = True
    replay_capacity_samples = 4000
    replay_max_cached_shards = 1024
    replay_seed = 2026


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
            evaluator_cfg=evaluator_cfg,
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
            use_virtual_env=use_virtual_env,
            replay_capacity_samples=replay_capacity_samples,
            replay_max_cached_shards=replay_max_cached_shards,
            replay_seed=replay_seed,
        )
    except Exception as e:
        print(f"ERROR during self-improvement policy: {e}")
        raise
    



if __name__ == "__main__":
    main()
