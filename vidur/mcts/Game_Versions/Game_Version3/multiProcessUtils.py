# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import gc
import logging
import math
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

# The entrypoint sets main-process thread env before importing this module.
# Worker processes switch back to single-threaded mode explicitly before self-play.

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from vidur.config import SimulationConfig
from vidur.config.config import MetricsConfig
from vidur.simulator import Simulator

from ...environment import VidurMCTSEnvironment
from ...launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions
from ...virtual_simulator import VirtualSimulator

from .config import MultipleProcessTrainingConfig
from .virtual_environment import VirtualVidurMCTSEnvironment
from .mctsDNN import VidurMCTS
from .DNN.dnn_spec import make_dnn_spec
from .DNN.value_models import AlphaZeroModel
from .DNN.replay_write import ReplayWriter, ReplayWriterConfig
from .DNN.selfPlay import SelfPlayRunner
from .DNN.native_selfplay import run_native_selfplay_to_writers
from .DNN.replay_buffer import BestModelReplayBuffer
from .DNN.replay_dataset import collate_mixed_samples, load_manifest
from .DNN.trainer import Trainer, TrainerConfig
from .logger.evaluation_pipeline_logger import EvaluationMetricsLogger, EvalRootPredictionLogger


def _suppress_noisy_selfplay_loggers() -> None:
    logging.getLogger("vidur.execution_time_predictor.sklearn_execution_time_predictor").setLevel(
        logging.WARNING
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


def _set_torch_intraop_threads(num_threads: int) -> int:
    target = max(1, int(num_threads))
    try:
        torch.set_num_threads(target)
    except Exception:
        pass
    try:
        return int(torch.get_num_threads())
    except Exception:
        return target


def _set_runtime_cpu_thread_env(num_threads: int) -> int:
    target = max(1, int(num_threads))
    value = str(target)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value
    return target


def configure_simulation(sim_args: Iterable[str]) -> SimulationConfig:
    original_argv = sys.argv
    original_metrics_post_init = MetricsConfig.__post_init__
    original_write_config_to_file = SimulationConfig.write_config_to_file
    try:
        # Self-play workers do not use Vidur's standalone simulator metrics output.
        # Suppress per-process timestamped directory creation and config.json writes
        # while reconstructing the simulation config from CLI args.
        MetricsConfig.__post_init__ = lambda self: None  # type: ignore[assignment]
        SimulationConfig.write_config_to_file = lambda self: None  # type: ignore[assignment]
        sys.argv = [original_argv[0]] + list(sim_args)
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        MetricsConfig.__post_init__ = original_metrics_post_init
        SimulationConfig.write_config_to_file = original_write_config_to_file
        sys.argv = original_argv

    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False

    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]

    return cfg


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

    setattr(explore_cfg, "native_mcts_enabled", False)
    setattr(explore_cfg, "torchscript_full_native_search", False)
    setattr(explore_cfg, "native_log_events", False)
    setattr(explore_cfg, "native_profile", False)
    setattr(explore_cfg, "native_log_flush_every", 1)
    setattr(explore_cfg, "max_forced_hops", int(cfg.max_forced_hops_per_root))
    setattr(explore_cfg, "reuse_root_infer_inputs", False)

    setattr(explore_cfg, "prior_value_mode", str(search.prior_value_mode))
    setattr(explore_cfg, "root_dirichlet_noise_enabled", bool(search.root_dirichlet_noise_enabled))
    setattr(explore_cfg, "root_dirichlet_alpha", float(search.root_dirichlet_alpha))
    setattr(explore_cfg, "root_dirichlet_epsilon", float(search.root_dirichlet_epsilon))
    setattr(explore_cfg, "pb_c_base", float(search.pb_c_base))
    setattr(explore_cfg, "pb_c_init", float(search.pb_c_init))

    setattr(explore_cfg, "discount_factor", float(search.discount_factor))
    if search.discount_time_denominator_sec is not None:
        setattr(explore_cfg, "discount_time_denominator_sec", float(search.discount_time_denominator_sec))

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
    setattr(
        sim_cfg,
        "mcts_restore_pool_free_cap",
        int(cfg.virtual_simulator_max_free_request_pool),
    )
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


def _history_signature_key(sig: Any) -> Any:
    if isinstance(sig, str):
        return sig
    if isinstance(sig, (list, tuple)):
        return tuple(_history_signature_key(x) for x in sig)
    return sig


def _build_selfplay_cycle_task_payloads(
    cfg: MultipleProcessTrainingConfig,
    *,
    gen: int,
    cycle_index: int,
    weights_path: Path,
    gen_train_dir: Path,
    gen_eval_dir: Path,
    hop_ranges: Sequence[tuple[int, int]],
    seen_history_signatures_by_worker: dict[int, set[tuple[Any, ...]]],
) -> list[dict[str, Any]]:
    splits = _partition_roots_contiguous(
        total_roots=int(cfg.roots_per_generation),
        workers=int(cfg.num_processes),
    )
    roots_per_cycle = int(cfg.roots_per_generation)
    cycle_root_base = int(cfg.run.root_id) + int(cycle_index) * int(roots_per_cycle)

    task_payloads: list[dict[str, Any]] = []
    for wid, (start, count) in enumerate(splits):
        if count <= 0:
            continue

        worker_game_id = (
            int(cfg.run.game_id)
            + int(gen) * 1_000_000
            + int(cycle_index) * 100_000
            + int(wid)
        )
        worker_start_root_id = int(cycle_root_base) + int(start)

        history_seed = (
            int(cfg.history_seed)
            + int(gen) * 100_000
            + int(cycle_index) * 10_000
            + int(wid)
        )
        action_seed_base = (
            int(cfg.action_seed_base)
            + int(gen) * 10_000_000
            + int(cycle_index) * 100_000
            + int(wid) * 1000
        )
        task_seed = (
            int(cfg.game_v2.reproducibility.global_seed)
            + int(gen) * 100_000
            + int(cycle_index) * 10_000
            + int(wid)
        )

        hop_min, hop_max = hop_ranges[wid]
        eval_split_seed = (
            int(cfg.eval_split_seed_base)
            + int(gen) * 1_000_000
            + int(cycle_index) * 10_000
            + int(wid)
        )

        task_payloads.append(
            {
                "interval_id": int(wid),
                "task_kind": "selfplay",
                "generation": int(gen),
                "cycle_index": int(cycle_index),
                "gen_train_dir_base": str(gen_train_dir),
                "gen_eval_dir_base": str(gen_eval_dir),
                "weights_path": str(weights_path),
                "model_version": int(gen),
                "game_id_base": int(worker_game_id),
                "num_roots": int(count),
                "start_root_id": int(worker_start_root_id),
                "start_root_depth": int(cfg.run.root_depth),
                "start_player": str(cfg.run.root_player),
                "feature_version": int(cfg.run.feature_version),
                "adv_iterations_per_root": int(cfg.adv_iterations_per_root),
                "cont_iterations_per_root": int(cfg.cont_iterations_per_root),
                "max_batch_size": int(cfg.max_batch_size),
                "history_nontrivial_hops": int(hop_min),
                "history_hops_min": int(hop_min),
                "history_hops_max": int(hop_max),
                "history_seed_base": int(history_seed),
                "sample_from_mcts_policy": bool(cfg.sample_from_mcts_policy),
                "selfplay_policy_temperature": float(cfg.selfplay_policy_temperature),
                "action_seed_base_base": int(action_seed_base),
                "max_forced_hops_per_root": int(cfg.max_forced_hops_per_root),
                "history_max_total_steps": int(cfg.history_max_total_steps),
                "history_root_batch_size": int(cfg.history_root_batch_size),
                "log_history_rows": bool(cfg.log_history_rows),
                "eval_split_ratio": float(cfg.eval_split_ratio),
                "eval_split_seed_base": int(eval_split_seed),
                "task_seed_base": int(task_seed),
                "history_seen_signatures": list(
                    seen_history_signatures_by_worker.get(int(wid), set())
                ),
            }
        )
    return task_payloads


def _eligible_interval_deficit(interval_state: dict[str, Any]) -> int:
    target = int(interval_state["target_roots"])
    completed = int(interval_state["completed_unique_roots"])
    inflight = int(interval_state["inflight_requested_roots"])
    return max(0, target - completed - inflight)


def _make_selfplay_chunk_task(
    *,
    interval_state: dict[str, Any],
    chunk_roots: int,
    task_instance_id: int,
) -> dict[str, Any]:
    interval_id = int(interval_state["interval_id"])
    chunk_index = int(interval_state["chunk_index"])
    interval_state["chunk_index"] = int(chunk_index + 1)

    start_root_id = int(interval_state["next_root_id"])
    interval_state["next_root_id"] = int(start_root_id + int(chunk_roots))

    chunk_seed_offset = int(chunk_index) * 1_000_003 + int(task_instance_id) * 101
    gen_train_dir_base = Path(interval_state["gen_train_dir_base"])
    gen_eval_dir_base = Path(interval_state["gen_eval_dir_base"])

    return {
        "worker_id": int(interval_id),
        "interval_id": int(interval_id),
        "task_kind": "selfplay",
        "generation": int(interval_state["generation"]),
        "cycle_index": int(interval_state["cycle_index"]),
        "task_instance_id": int(task_instance_id),
        "chunk_index": int(chunk_index),
        "out_dir_train": str(
            gen_train_dir_base / f"proc_{int(interval_id):02d}_cycle_{int(interval_state['cycle_index']):02d}_task_{int(task_instance_id):05d}"
        ),
        "out_dir_eval": str(
            gen_eval_dir_base / f"proc_{int(interval_id):02d}_cycle_{int(interval_state['cycle_index']):02d}_task_{int(task_instance_id):05d}"
        ),
        "weights_path": str(interval_state["weights_path"]),
        "model_version": int(interval_state["model_version"]),
        "game_id": int(interval_state["game_id_base"]) + int(chunk_index),
        "num_roots": int(chunk_roots),
        "start_root_id": int(start_root_id),
        "start_root_depth": int(interval_state["start_root_depth"]),
        "start_player": str(interval_state["start_player"]),
        "feature_version": int(interval_state["feature_version"]),
        "adv_iterations_per_root": int(interval_state["adv_iterations_per_root"]),
        "cont_iterations_per_root": int(interval_state["cont_iterations_per_root"]),
        "max_batch_size": int(interval_state["max_batch_size"]),
        "history_nontrivial_hops": int(interval_state["history_nontrivial_hops"]),
        "history_hops_min": int(interval_state["history_hops_min"]),
        "history_hops_max": int(interval_state["history_hops_max"]),
        "history_seed": int(interval_state["history_seed_base"]) + int(chunk_seed_offset),
        "sample_from_mcts_policy": bool(interval_state["sample_from_mcts_policy"]),
        "selfplay_policy_temperature": float(interval_state["selfplay_policy_temperature"]),
        "action_seed_base": int(interval_state["action_seed_base_base"]) + int(chunk_seed_offset) * 17,
        "max_forced_hops_per_root": int(interval_state["max_forced_hops_per_root"]),
        "history_max_total_steps": int(interval_state["history_max_total_steps"]),
        "history_root_batch_size": int(interval_state["history_root_batch_size"]),
        "log_history_rows": bool(interval_state["log_history_rows"]),
        "eval_split_ratio": float(interval_state["eval_split_ratio"]),
        "eval_split_seed": int(interval_state["eval_split_seed_base"]) + int(chunk_seed_offset) * 31,
        "task_seed": int(interval_state["task_seed_base"]) + int(chunk_seed_offset) * 43,
        "history_seen_signatures": list(interval_state["completed_history_signatures"]),
        "suppress_worker_progress_logs": bool(interval_state.get("suppress_worker_progress_logs", False)),
        "allow_duplicate_history_fallback": bool(interval_state["allow_duplicate_history_fallback"]),
        "shared_history_signatures": interval_state["shared_seen_proxy"],
        "shared_history_lock": interval_state["shared_lock_proxy"],
    }


def _run_selfplay_cycle(
    ctx: mp.context.BaseContext,
    cfg: MultipleProcessTrainingConfig,
    *,
    gen: int,
    cycle_index: int,
    task_payloads: Sequence[dict[str, Any]],
) -> list[dict]:
    if len(task_payloads) <= 0:
        raise RuntimeError(f"No self-play tasks sent for generation {gen}, cycle {cycle_index}")

    manager = ctx.Manager()
    result_q = ctx.Queue()
    workers: list[mp.Process] = []
    active_workers: dict[int, mp.Process] = {}
    active_task_meta: dict[int, dict[str, Any]] = {}
    msgs: list[dict] = []
    max_active = max(1, int(getattr(cfg, "max_concurrent_selfplay_workers", cfg.num_processes)))
    max_per_interval = max(1, int(getattr(cfg, "max_workers_per_interval", 1)))
    chunk_roots_default = max(1, int(getattr(cfg, "selfplay_dynamic_chunk_roots", 128)))
    zero_progress_patience = max(1, int(getattr(cfg, "selfplay_zero_progress_interval_patience", 3)))
    rss_limit_bytes = int(float(getattr(cfg, "selfplay_launch_rss_limit_gb", 100.0)) * float(1024 ** 3))
    poll_sec = max(0.1, float(getattr(cfg, "selfplay_launch_poll_sec", 2.0)))
    timeout_sec = max(1, int(cfg.worker_result_timeout_sec))
    last_progress_ts = time.time()
    rss_pause_logged = False
    total_launched_tasks = 0
    available_slots = list(range(int(max_active)))

    interval_states: dict[int, dict[str, Any]] = {}
    for spec in task_payloads:
        interval_id = int(spec["interval_id"])
        shared_seen_proxy = manager.dict()
        initial_sigs = list(spec.get("history_seen_signatures", []) or [])
        for sig in initial_sigs:
            shared_seen_proxy[_history_signature_key(sig)] = 1
        interval_states[int(interval_id)] = {
            **dict(spec),
            "interval_id": int(interval_id),
            "target_roots": int(spec["num_roots"]),
            "completed_unique_roots": 0,
            "inflight_requested_roots": 0,
            "active_workers": 0,
            "next_root_id": int(spec["start_root_id"]),
            "chunk_index": 0,
            "zero_progress_completions": 0,
            "exhausted": False,
            "completed_history_signatures": set(_history_signature_key(sig) for sig in initial_sigs),
            "shared_seen_proxy": shared_seen_proxy,
            "shared_lock_proxy": manager.Lock(),
            "allow_duplicate_history_fallback": bool(getattr(cfg, "history_allow_duplicate_root_fallback", False)),
        }

    def _has_eligible_interval() -> bool:
        for state in interval_states.values():
            if bool(state["exhausted"]):
                continue
            if int(state["active_workers"]) >= int(max_per_interval):
                continue
            if _eligible_interval_deficit(state) <= 0:
                continue
            return True
        return False

    def _pick_interval_state() -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for state in interval_states.values():
            if bool(state["exhausted"]):
                continue
            if int(state["active_workers"]) >= int(max_per_interval):
                continue
            if _eligible_interval_deficit(state) <= 0:
                continue
            candidates.append(state)
        if not candidates:
            return None
        candidates.sort(
            key=lambda s: (
                -_eligible_interval_deficit(s),
                int(s["active_workers"]),
                int(s["zero_progress_completions"]),
                int(s["interval_id"]),
            )
        )
        return candidates[0]

    try:
        while _has_eligible_interval() or active_workers:
            launched_this_round = False
            while available_slots and _has_eligible_interval():
                manager_process = getattr(manager, "_process", None)
                manager_pid = getattr(manager_process, "pid", None)
                current_rss = _sum_python_rss_bytes(
                    active_worker_pids=(
                        pid
                        for pid in (
                            [manager_pid]
                            + [p.pid for p in active_workers.values() if getattr(p, "pid", None)]
                        )
                        if pid
                    )
                )
                allow_spawn = current_rss < int(rss_limit_bytes)
                if not allow_spawn:
                    if not rss_pause_logged:
                        print(
                            f"[GV3 gen={int(gen):06d} cycle={int(cycle_index):02d}] "
                            f"self-play launch paused: rss={_format_gib(current_rss)} "
                            f"limit={_format_gib(rss_limit_bytes)} active_workers={int(len(active_workers))}",
                            flush=True,
                        )
                        rss_pause_logged = True
                    break

                interval_state = _pick_interval_state()
                if interval_state is None:
                    break
                remaining = _eligible_interval_deficit(interval_state)
                if int(remaining) <= 0:
                    break
                chunk_roots = min(int(chunk_roots_default), int(remaining))
                worker_slot = int(available_slots.pop(0))
                total_launched_tasks += 1
                task = _make_selfplay_chunk_task(
                    interval_state=interval_state,
                    chunk_roots=int(chunk_roots),
                    task_instance_id=int(total_launched_tasks),
                )
                p = ctx.Process(
                    target=_selfplay_worker_main,
                    args=(worker_slot, cfg, task, result_q),
                    daemon=False,
                )
                p.start()
                workers.append(p)
                active_workers[int(worker_slot)] = p
                active_task_meta[int(worker_slot)] = {
                    "interval_id": int(interval_state["interval_id"]),
                    "requested_roots": int(chunk_roots),
                }
                interval_state["active_workers"] = int(interval_state["active_workers"]) + 1
                interval_state["inflight_requested_roots"] = (
                    int(interval_state["inflight_requested_roots"]) + int(chunk_roots)
                )
                launched_this_round = True
                rss_pause_logged = False
                last_progress_ts = time.time()

            if not active_workers:
                if _has_eligible_interval():
                    if time.time() - float(last_progress_ts) >= float(timeout_sec):
                        raise RuntimeError(
                            f"Timed out waiting to launch self-play work: "
                            f"generation={int(gen)}, cycle={int(cycle_index)}, "
                            f"active_workers=0, eligible_intervals_pending=1"
                        )
                    time.sleep(float(poll_sec))
                continue

            try:
                msg = result_q.get(timeout=max(1, int(math.ceil(poll_sec))))
            except queue.Empty:
                if time.time() - float(last_progress_ts) >= float(timeout_sec):
                    raise RuntimeError(
                        f"Timed out waiting for worker result: task_kind=selfplay, "
                        f"launched={int(total_launched_tasks)}, got={int(len(msgs))}, "
                        f"active_workers={int(len(active_workers))}"
                    )
                continue

            msg = _consume_worker_result(msg, task_kind="selfplay")
            slot = int(msg.get("worker_slot_id", -1))
            interval_id = int(msg.get("interval_id", msg.get("worker_id", -1)))
            proc = active_workers.pop(int(slot), None)
            meta = active_task_meta.pop(int(slot), {})
            if proc is not None:
                try:
                    proc.join(timeout=5)
                except Exception:
                    pass
            if int(slot) >= 0:
                available_slots.append(int(slot))
                available_slots.sort()

            interval_state = interval_states[int(interval_id)]
            requested_roots = int(meta.get("requested_roots", dict(msg.get("run_stats", {}) or {}).get("num_roots_requested", 0)))
            run_stats = dict(msg.get("run_stats", {}) or {})
            emitted_history_signatures = [
                _history_signature_key(sig)
                for sig in list(run_stats.get("history_signatures", []) or [])
            ]
            unique_chunk_roots = int(len(emitted_history_signatures))
            interval_state["active_workers"] = max(0, int(interval_state["active_workers"]) - 1)
            interval_state["inflight_requested_roots"] = max(
                0,
                int(interval_state["inflight_requested_roots"]) - int(requested_roots),
            )
            interval_state["completed_unique_roots"] = (
                int(interval_state["completed_unique_roots"]) + int(unique_chunk_roots)
            )
            if emitted_history_signatures:
                interval_state["zero_progress_completions"] = 0
                for sig in emitted_history_signatures:
                    interval_state["completed_history_signatures"].add(sig)
            else:
                interval_state["zero_progress_completions"] = int(interval_state["zero_progress_completions"]) + 1
                if (
                    int(interval_state["active_workers"]) == 0
                    and _eligible_interval_deficit(interval_state) > 0
                    and int(interval_state["zero_progress_completions"]) >= int(zero_progress_patience)
                ):
                    interval_state["exhausted"] = True
                    print(
                        f"[GV3 gen={int(gen):06d} cycle={int(cycle_index):02d}] "
                        f"interval {int(interval_id):02d} marked exhausted after "
                        f"{int(interval_state['zero_progress_completions'])} zero-yield chunks: "
                        f"completed_unique_roots={int(interval_state['completed_unique_roots'])}/"
                        f"{int(interval_state['target_roots'])}",
                        flush=True,
                    )

            msgs.append(msg)
            last_progress_ts = time.time()

        unfinished = [
            state for state in interval_states.values()
            if int(state["completed_unique_roots"]) < int(state["target_roots"])
        ]
        if unfinished:
            parts = [
                f"i{int(s['interval_id']):02d}={int(s['completed_unique_roots'])}/{int(s['target_roots'])}"
                for s in unfinished
            ]
            print(
                f"[GV3 gen={int(gen):06d} cycle={int(cycle_index):02d}] "
                f"interval deficits after dynamic scheduling: {', '.join(parts)}",
                flush=True,
            )
        return msgs
    finally:
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
        try:
            manager.shutdown()
        except Exception:
            pass


def _next_generation_index(dataset_base: Path) -> int:
    max_gen = -1
    for p in dataset_base.glob("gen_*"):
        if not p.is_dir():
            continue
        tail = p.name[4:] if p.name.startswith("gen_") else ""
        if tail.isdigit():
            max_gen = max(max_gen, int(tail))
    return max_gen + 1


def _compute_train_steps_this_generation(
    cfg: MultipleProcessTrainingConfig,
    replay_total_samples: int,
) -> int:
    target_epochs = float(getattr(cfg, "train_target_epochs_per_generation", 0.0))
    if target_epochs > 0.0:
        batch_size = max(1, int(cfg.train_batch_size))
        samples = max(1, int(replay_total_samples))
        return int(math.ceil(target_epochs * float(samples) / float(batch_size)))

    base_steps = max(1, int(cfg.train_steps_per_generation))
    roots_ref = max(1, int(cfg.roots_per_generation))
    ratio = float(max(1, int(replay_total_samples))) / float(roots_ref)
    return int(math.ceil(base_steps * ratio))


def _load_weights_into_model(model: torch.nn.Module, weights_path: Path) -> None:
    blob = torch.load(weights_path, map_location="cpu")
    model_state = blob.get("model_state", blob)
    model.load_state_dict(model_state, strict=True)
    model.eval()


def _load_trainer_from_checkpoint(trainer: Trainer, ckpt_path: Path) -> None:
    blob = torch.load(ckpt_path, map_location="cpu")
    model_state = blob.get("model_state", blob)
    trainer.model.load_state_dict(model_state, strict=True)

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


def _read_process_rss_bytes(pid: int) -> int:
    try:
        text = Path(f"/proc/{int(pid)}/status").read_text()
    except Exception:
        return 0

    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024
                except Exception:
                    return 0
    return 0


def _sum_python_rss_bytes(*, active_worker_pids: Iterable[int]) -> int:
    total = _read_process_rss_bytes(os.getpid())
    seen = {int(os.getpid())}
    for pid in active_worker_pids:
        ipid = int(pid)
        if ipid <= 0 or ipid in seen:
            continue
        seen.add(ipid)
        total += _read_process_rss_bytes(ipid)
    return int(total)


def _format_gib(num_bytes: int) -> str:
    return f"{float(num_bytes) / float(1024 ** 3):.2f} GiB"


def _consume_worker_result(msg: dict, *, task_kind: str) -> dict:
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

    return msg


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _weighted_avg(total: float, weight: float) -> float:
    if float(weight) <= 0.0:
        return float("nan")
    return float(total) / float(weight)


def _aggregate_metric_rows(rows: list[dict]) -> dict[str, float]:
    total_count = 0.0
    controller_count = 0.0
    adversary_count = 0.0

    sum_loss = 0.0
    sum_value_loss = 0.0
    sum_value_mse = 0.0
    sum_value_mae = 0.0

    sum_controller_value_loss = 0.0
    sum_controller_value_mse = 0.0
    sum_controller_value_mae = 0.0
    sum_adversary_value_loss = 0.0
    sum_adversary_value_mse = 0.0
    sum_adversary_value_mae = 0.0

    for r in rows:
        c = max(0.0, _safe_float(r.get("controller_count", 0.0), 0.0))
        a = max(0.0, _safe_float(r.get("adversary_count", 0.0), 0.0))
        t = c + a

        if t > 0.0:
            total_count += t
            sum_loss += _safe_float(r.get("loss", float("nan"))) * t
            sum_value_loss += _safe_float(r.get("value_loss", float("nan"))) * t
            sum_value_mse += _safe_float(r.get("value_mse_error", float("nan"))) * t
            sum_value_mae += _safe_float(r.get("value_mae_error", float("nan"))) * t

        if c > 0.0:
            controller_count += c
            sum_controller_value_loss += _safe_float(r.get("controller_value_loss", float("nan"))) * c
            sum_controller_value_mse += _safe_float(r.get("controller_value_mse_error", float("nan"))) * c
            sum_controller_value_mae += _safe_float(r.get("controller_value_mae_error", float("nan"))) * c

        if a > 0.0:
            adversary_count += a
            sum_adversary_value_loss += _safe_float(r.get("adversary_value_loss", float("nan"))) * a
            sum_adversary_value_mse += _safe_float(r.get("adversary_value_mse_error", float("nan"))) * a
            sum_adversary_value_mae += _safe_float(r.get("adversary_value_mae_error", float("nan"))) * a

    return {
        "samples_needed": int(round(total_count)),
        "controller_samples_needed": int(round(controller_count)),
        "adversary_samples_needed": int(round(adversary_count)),
        "loss_for_selection": _weighted_avg(sum_loss, total_count),
        "value_loss": _weighted_avg(sum_value_loss, total_count),
        "value_mse_error": _weighted_avg(sum_value_mse, total_count),
        "value_mae_error": _weighted_avg(sum_value_mae, total_count),
        "controller_value_loss": _weighted_avg(sum_controller_value_loss, controller_count),
        "controller_value_mse_error": _weighted_avg(sum_controller_value_mse, controller_count),
        "controller_value_mae_error": _weighted_avg(sum_controller_value_mae, controller_count),
        "adversary_value_loss": _weighted_avg(sum_adversary_value_loss, adversary_count),
        "adversary_value_mse_error": _weighted_avg(sum_adversary_value_mse, adversary_count),
        "adversary_value_mae_error": _weighted_avg(sum_adversary_value_mae, adversary_count),
    }


def _load_generation_samples(dataset_dir: Path) -> list[dict]:
    out: list[dict] = []
    dataset_dir = Path(dataset_dir)

    for proc_dir in sorted(dataset_dir.glob("proc_*")):
        manifest = proc_dir / "manifest.jsonl"
        if not manifest.exists():
            continue

        for entry in load_manifest(manifest, allow_empty=True):
            shard_path = Path(entry.path)
            if not shard_path.is_absolute():
                shard_path = (proc_dir / shard_path).resolve()
            shard = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard, list):
                raise TypeError(f"Expected list shard at {shard_path}, got {type(shard)}")
            out.extend(shard)

    return out


def _scan_dataset_partition_stats(dataset_dir: Path) -> dict[str, Any]:
    stats = {
        "samples_total": 0,
        "controller_samples": 0,
        "adversary_samples": 0,
        "root_ids": set(),
    }
    dataset_dir = Path(dataset_dir)

    for proc_dir in sorted(dataset_dir.glob("proc_*")):
        manifest = proc_dir / "manifest.jsonl"
        if not manifest.exists():
            continue

        for entry in load_manifest(manifest, allow_empty=True):
            shard_path = Path(entry.path)
            if not shard_path.is_absolute():
                shard_path = (proc_dir / shard_path).resolve()
            try:
                shard = torch.load(shard_path, map_location="cpu")
            except Exception as exc:
                print(
                    f"[GV3 dataset scan] skipping unreadable shard: path={shard_path}, error={exc}",
                    flush=True,
                )
                continue
            if not isinstance(shard, list):
                print(
                    f"[GV3 dataset scan] skipping non-list shard: path={shard_path}, type={type(shard)}",
                    flush=True,
                )
                continue

            for sample in shard:
                stats["samples_total"] += 1
                player = str(sample.get("player", ""))
                if player == "controller":
                    stats["controller_samples"] += 1
                elif player == "adversary":
                    stats["adversary_samples"] += 1

                root_id = sample.get("root_id", None)
                if root_id is not None:
                    stats["root_ids"].add(int(root_id))

    return stats


def _scan_generation_dataset_stats(
    *,
    train_dir: Path,
    eval_dir: Path,
) -> dict[str, int]:
    train_stats = _scan_dataset_partition_stats(train_dir)
    eval_stats = _scan_dataset_partition_stats(eval_dir)
    unique_root_ids = set(train_stats["root_ids"])
    unique_root_ids.update(eval_stats["root_ids"])

    return {
        "num_unique_roots": int(len(unique_root_ids)),
        "train_samples_total": int(train_stats["samples_total"]),
        "eval_samples_total": int(eval_stats["samples_total"]),
        "controller_train_samples": int(train_stats["controller_samples"]),
        "controller_eval_samples": int(eval_stats["controller_samples"]),
        "adversary_train_samples": int(train_stats["adversary_samples"]),
        "adversary_eval_samples": int(eval_stats["adversary_samples"]),
    }


def _evaluate_samples(
    trainer: Trainer,
    *,
    samples: list[dict],
    batch_size: int,
) -> dict[str, float]:
    if not samples:
        return _aggregate_metric_rows([])

    rows: list[dict] = []
    bsz = max(1, int(batch_size))
    for i in range(0, len(samples), bsz):
        chunk = samples[i : i + bsz]
        mixed = collate_mixed_samples(
            chunk,
            device=trainer.device,
            include_policy_tensors=False,
            include_legacy_fallback=False,
            include_ids=False,
        )
        rows.append(trainer.eval_step(mixed))

    return _aggregate_metric_rows(rows)


def _build_eval_prediction_rows(
    trainer: Trainer,
    *,
    samples: list[dict],
    batch_size: int,
    model_version: int,
) -> list[dict[str, Any]]:
    if not samples:
        return []

    rows: list[dict[str, Any]] = []
    bsz = max(1, int(batch_size))
    trainer.model.eval()

    with torch.no_grad():
        for i in range(0, len(samples), bsz):
            chunk = samples[i : i + bsz]
            mixed = collate_mixed_samples(
                chunk,
                device=trainer.device,
                include_policy_tensors=False,
                include_legacy_fallback=False,
                include_ids=True,
            )

            for player in ("controller", "adversary"):
                batch = mixed.get(player)
                if batch is None:
                    continue

                _policy_logits, value_raw = trainer.model.forward(
                    player=player,
                    prefill_req_features=batch.get("prefill_req_features"),
                    decode_req_features=batch.get("decode_req_features"),
                    global_features=batch["global_features"],
                    prefill_req_mask=batch.get("prefill_req_mask"),
                    decode_req_mask=batch.get("decode_req_mask"),
                    req_features=batch.get("req_features"),
                    req_mask=batch.get("req_mask"),
                    action_mask=None,
                )

                pred_values = trainer.model.value_scalar_from_logits(value_raw).view(-1).detach().cpu().tolist()
                true_values = batch["target_value"].view(-1).detach().cpu().tolist()
                root_ids = batch["ids"]["root_id"].view(-1).detach().cpu().tolist()

                for root_id, pred_value, true_value in zip(root_ids, pred_values, true_values):
                    rows.append(
                        {
                            "model_version": int(model_version),
                            "root_number": int(root_id),
                            "root_player_type": str(player),
                            "model_predicted_value": float(pred_value),
                            "true_tree_search_value": float(true_value),
                        }
                    )

    rows.sort(key=lambda r: (int(r["root_number"]), str(r["root_player_type"])))
    return rows


def _selfplay_worker_main(
    worker_id: int,
    cfg: MultipleProcessTrainingConfig,
    task: dict,
    result_q: mp.Queue,
) -> None:
    writer_train = None
    writer_eval = None
    mcts = None
    try:
        _suppress_noisy_selfplay_loggers()
        _set_global_seeds(
            int(cfg.game_v2.reproducibility.global_seed) + int(worker_id),
            torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
        )
        _set_runtime_cpu_thread_env(1)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        spec = make_dnn_spec(cfg=cfg.game_v2)
        model = AlphaZeroModel(spec=spec).to(torch.device(cfg.model.device))
        model.eval()
        task_kind = str(task.get("task_kind", "selfplay"))
        if task_kind != "selfplay":
            result_q.put(
                {
                    "ok": False,
                    "task_kind": task_kind,
                    "worker_id": int(worker_id),
                    "error": f"unknown task_kind={task_kind}",
                }
            )
            return

        gen = int(task["generation"])
        cycle_index = int(task.get("cycle_index", 0))
        interval_id = int(task.get("interval_id", worker_id))
        task_instance_id = int(task.get("task_instance_id", 0))
        chunk_index = int(task.get("chunk_index", 0))
        out_dir_train = Path(task["out_dir_train"])
        out_dir_eval = Path(task["out_dir_eval"])
        out_dir_train.mkdir(parents=True, exist_ok=True)
        out_dir_eval.mkdir(parents=True, exist_ok=True)

        _load_weights_into_model(model, Path(task["weights_path"]))

        task_seed = int(task.get("task_seed", 0))
        _set_global_seeds(
            task_seed,
            torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
        )

        writer_train = ReplayWriter(
            ReplayWriterConfig(
                out_dir=out_dir_train,
                shard_size=int(cfg.dataset.shard_size),
            )
        )
        writer_eval = ReplayWriter(
            ReplayWriterConfig(
                out_dir=out_dir_eval,
                shard_size=int(cfg.dataset.shard_size),
            )
        )

        progress_prefix = (
            f"[GV3 selfplay gen={int(gen):06d} cycle={int(cycle_index):02d} "
            f"interval={int(interval_id):02d} chunk={int(chunk_index):03d} slot={int(worker_id):02d}]"
        )
        if bool(task.get("suppress_worker_progress_logs", False)):
            progress_prefix = ""

        if str(getattr(cfg, "environment_lang", "python")).lower() == "native":
            if progress_prefix:
                print(
                    f"{progress_prefix} native self-play starting: roots={int(task['num_roots'])}, "
                    f"hop_range=[{int(task['history_hops_min'])}, {int(task['history_hops_max'])}], "
                    f"eval_ratio={float(task.get('eval_split_ratio', 0.0)):.3f}",
                    flush=True,
                )
            run_stats = run_native_selfplay_to_writers(
                cfg=cfg,
                task=task,
                model=model,
                writer_train=writer_train,
                writer_eval=writer_eval,
            )
            if writer_train is not None:
                writer_train.close()
                writer_train = None
            if writer_eval is not None:
                writer_eval.close()
                writer_eval = None
            if progress_prefix:
                print(
                    f"{progress_prefix} native self-play finished: "
                    f"roots_generated={int(run_stats.get('num_roots_generated', 0))}, "
                    f"unique_roots={int(run_stats.get('num_unique_roots', 0))}, "
                    f"train_samples={int(run_stats.get('train_samples_total', 0))}, "
                    f"eval_samples={int(run_stats.get('eval_samples_total', 0))}",
                    flush=True,
                )
            result_q.put(
                {
                    "ok": True,
                    "task_kind": "selfplay",
                    "worker_id": int(interval_id),
                    "worker_slot_id": int(worker_id),
                    "interval_id": int(interval_id),
                    "generation": int(gen),
                    "cycle_index": int(cycle_index),
                    "task_instance_id": int(task_instance_id),
                    "out_dir_train": str(out_dir_train),
                    "out_dir_eval": str(out_dir_eval),
                    "run_stats": run_stats,
                }
            )
            return

        _, env, _, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=bool(cfg.use_virtual_env))

        logs_base = Path(cfg.logging.mcts_iter_log).parent
        logs_dir = logs_base / f"gen_{gen:06d}"
        logs_dir.mkdir(parents=True, exist_ok=True)

        iter_log = (
            logs_dir
            / f"mcts_iter_i{int(interval_id):02d}_c{int(cycle_index):02d}_t{int(task_instance_id):05d}_w{int(worker_id):02d}.csv"
        )
        root_log = (
            logs_dir
            / f"mcts_root_i{int(interval_id):02d}_c{int(cycle_index):02d}_t{int(task_instance_id):05d}_w{int(worker_id):02d}.csv"
        )

        mcts = VidurMCTS(
            env=env,
            explore_cfg=explore_cfg,
            rng=random.Random(int(task["action_seed_base"])),
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
            writer=writer_train,
            eval_writer=writer_eval,
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
            history_hops_min=int(task["history_hops_min"]),
            history_hops_max=int(task["history_hops_max"]),
            history_seed=int(task["history_seed"]),
            max_forced_hops_per_root=int(task["max_forced_hops_per_root"]),
            history_max_total_steps=int(task["history_max_total_steps"]),
            history_root_batch_size=int(task.get("history_root_batch_size", 64)),
            log_history_rows=bool(task["log_history_rows"]),
            progress_prefix=progress_prefix,
            model_version=int(task.get("model_version", 0)),
            eval_split_ratio=float(task.get("eval_split_ratio", 0.0)),
            eval_split_seed=int(task.get("eval_split_seed", 0)),
            history_seen_signatures=task.get("history_seen_signatures", None),
            shared_history_signatures=task.get("shared_history_signatures", None),
            shared_history_lock=task.get("shared_history_lock", None),
            allow_duplicate_history_fallback=bool(task.get("allow_duplicate_history_fallback", True)),
        )

        run_stats = dict(getattr(runner, "last_run_stats", {}) or {})

        if writer_train is not None:
            writer_train.close()
            writer_train = None
        if writer_eval is not None:
            writer_eval.close()
            writer_eval = None
        if mcts is not None:
            mcts.close()
            mcts = None

        result_q.put(
            {
                "ok": True,
                "task_kind": "selfplay",
                "worker_id": int(interval_id),
                "worker_slot_id": int(worker_id),
                "interval_id": int(interval_id),
                "generation": int(gen),
                "cycle_index": int(cycle_index),
                "task_instance_id": int(task_instance_id),
                "out_dir_train": str(out_dir_train),
                "out_dir_eval": str(out_dir_eval),
                "run_stats": run_stats,
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
        if writer_train is not None:
            try:
                writer_train.close()
            except Exception:
                pass
        if writer_eval is not None:
            try:
                writer_eval.close()
            except Exception:
                pass
        if mcts is not None:
            try:
                mcts.close()
            except Exception:
                pass


def _aggregate_worker_run_stats(msgs: list[dict]) -> dict[str, int]:
    agg = {
        "num_roots_requested": 0,
        "num_roots_generated": 0,
        "num_unique_roots": 0,
        "controller_train_samples": 0,
        "controller_eval_samples": 0,
        "adversary_train_samples": 0,
        "adversary_eval_samples": 0,
        "train_samples_total": 0,
        "eval_samples_total": 0,
    }
    for m in msgs:
        rs = dict(m.get("run_stats", {}) or {})
        for k in list(agg.keys()):
            agg[k] += int(rs.get(k, 0))
    return agg


def run_parallel_self_improvement(cfg: MultipleProcessTrainingConfig) -> None:
    cfg.validate()

    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )

    _set_runtime_cpu_thread_env(int(getattr(cfg, "train_num_threads", 1)))
    _set_torch_intraop_threads(int(getattr(cfg, "train_num_threads", 1)))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    dataset_base = Path(cfg.dataset.out_dir)
    dataset_base.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            value_weight=float(trainer_h.value_weight),
            value_loss_alpha=float(trainer_h.value_loss_alpha),
            grad_clip_norm=float(trainer_h.grad_clip_norm),
            checkpoint_every=int(trainer_h.checkpoint_every),
            eval_every=int(trainer_h.eval_every),
        ),
        device=torch.device(cfg.model.device),
    )

    replay_buffer = BestModelReplayBuffer(
        capacity_samples=int(cfg.replay_capacity_samples),
        max_cached_shards=int(cfg.replay_max_cached_shards),
        seed=int(cfg.replay_seed),
    )

    best_ckpt_path = ckpt_dir / "best.pt"
    if not best_ckpt_path.exists():
        trainer.save_checkpoint(best_ckpt_path)

    _load_trainer_from_checkpoint(trainer, best_ckpt_path)

    ctx = mp.get_context("spawn")
    try:
        gen_start = _next_generation_index(dataset_base)
        reuse_dataset_generation_remaining = int(getattr(cfg, "reuse_dataset_generation", -1))
        reuse_dataset_generation_only_once = bool(
            getattr(cfg, "reuse_dataset_generation_only_once", False)
        )

        for local_gen in range(int(cfg.num_generations)):
            gen = int(gen_start + local_gen)

            _load_trainer_from_checkpoint(trainer, best_ckpt_path)

            gen_dataset_dir = dataset_base / f"gen_{gen:06d}"
            gen_train_dir = gen_dataset_dir / "train"
            gen_eval_dir = gen_dataset_dir / "eval"
            gen_train_dir.mkdir(parents=True, exist_ok=True)
            gen_eval_dir.mkdir(parents=True, exist_ok=True)
            gen_logs_dir = Path(cfg.logging.mcts_iter_log).parent / f"gen_{gen:06d}"
            gen_logs_dir.mkdir(parents=True, exist_ok=True)

            reuse_dataset_generation = int(reuse_dataset_generation_remaining)
            if reuse_dataset_generation >= 0:
                source_dataset_dir = dataset_base / f"gen_{reuse_dataset_generation:06d}"
                source_train_dir = source_dataset_dir / "train"
                source_eval_dir = source_dataset_dir / "eval"
                if not source_train_dir.exists():
                    raise FileNotFoundError(f"Reused train dataset dir not found: {source_train_dir}")
                if not source_eval_dir.exists():
                    raise FileNotFoundError(f"Reused eval dataset dir not found: {source_eval_dir}")

                if bool(getattr(cfg, "reuse_dataset_reset_replay_each_generation", True)):
                    replay_buffer.reset_for_new_best()

                generation_stats = _scan_generation_dataset_stats(
                    train_dir=source_train_dir,
                    eval_dir=source_eval_dir,
                )
                print(
                    f"[GV3 gen={int(gen):06d}] reusing dataset from gen={int(reuse_dataset_generation):06d}: "
                    f"unique_roots={int(generation_stats['num_unique_roots'])}, "
                    f"train_samples={int(generation_stats['train_samples_total'])}, "
                    f"eval_samples={int(generation_stats['eval_samples_total'])}",
                    flush=True,
                )
                train_source_dir = source_train_dir
                eval_source_dir = source_eval_dir
                if reuse_dataset_generation_only_once:
                    print(
                        f"[GV3 gen={int(gen):06d}] one-shot dataset reuse consumed; "
                        f"subsequent generations will return to fresh self-play.",
                        flush=True,
                    )
                    reuse_dataset_generation_remaining = -1
            else:
                replay_buffer.reset_for_new_best()
                weights_path = ckpt_dir / f"selfplay_weights_gen_{gen:06d}.pt"
                shutil.copyfile(best_ckpt_path, weights_path)
                hop_ranges = cfg.selfplay_hop_ranges_for_generation(gen)
                total_cycles = max(1, int(getattr(cfg, "sample_cycles_per_generation", 1)))
                seen_history_signatures_by_worker: dict[int, set[tuple[Any, ...]]] = {
                    int(wid): set() for wid in range(int(cfg.num_processes))
                }
                worker_msgs_all: list[dict] = []

                _set_runtime_cpu_thread_env(1)
                _set_torch_intraop_threads(1)

                for cycle_index in range(int(total_cycles)):
                    task_payloads = _build_selfplay_cycle_task_payloads(
                        cfg,
                        gen=int(gen),
                        cycle_index=int(cycle_index),
                        weights_path=weights_path,
                        gen_train_dir=gen_train_dir,
                        gen_eval_dir=gen_eval_dir,
                        hop_ranges=hop_ranges,
                        seen_history_signatures_by_worker=seen_history_signatures_by_worker,
                    )
                    worker_msgs = _run_selfplay_cycle(
                        ctx,
                        cfg,
                        gen=int(gen),
                        cycle_index=int(cycle_index),
                        task_payloads=task_payloads,
                    )
                    worker_msgs_all.extend(worker_msgs)

                    for msg in worker_msgs:
                        wid = int(msg.get("worker_id", 0))
                        run_stats = dict(msg.get("run_stats", {}) or {})
                        for sig in list(run_stats.get("history_signatures", []) or []):
                            seen_history_signatures_by_worker.setdefault(int(wid), set()).add(sig)

                    cycle_stats = _aggregate_worker_run_stats(worker_msgs)
                    print(
                        f"[GV3 gen={int(gen):06d}] self-play cycle {int(cycle_index) + 1}/{int(total_cycles)} complete: "
                        f"unique_roots={int(cycle_stats['num_unique_roots'])}/"
                        f"{int(cfg.roots_per_generation)}, "
                        f"train_samples={int(cycle_stats['train_samples_total'])}, "
                        f"eval_samples={int(cycle_stats['eval_samples_total'])}",
                        flush=True,
                    )

                generation_stats = _aggregate_worker_run_stats(worker_msgs_all)
                print(
                    f"[GV3 gen={int(gen):06d}] self-play complete: "
                    f"unique_roots={int(generation_stats['num_unique_roots'])}/"
                    f"{int(cfg.roots_per_generation) * int(total_cycles)}, "
                    f"train_samples={int(generation_stats['train_samples_total'])}, "
                    f"eval_samples={int(generation_stats['eval_samples_total'])}",
                    flush=True,
                )
                train_source_dir = gen_train_dir
                eval_source_dir = gen_eval_dir

            added_from_gen = replay_buffer.add_generation_dir(train_source_dir)
            if replay_buffer.total_samples <= 0:
                raise RuntimeError(
                    f"Replay buffer empty after generation={gen}, added_from_gen={added_from_gen}, dir={train_source_dir}"
                )

            _set_runtime_cpu_thread_env(int(getattr(cfg, "train_num_threads", 1)))
            train_steps_this_gen = _compute_train_steps_this_generation(cfg, replay_buffer.total_samples)
            actual_train_threads = _set_torch_intraop_threads(int(getattr(cfg, "train_num_threads", 1)))
            if (
                (
                    reuse_dataset_generation >= 0
                    and bool(getattr(cfg, "replay_preload_all_shards_on_reuse", False))
                )
                or (
                    reuse_dataset_generation < 0
                    and bool(getattr(cfg, "replay_preload_all_shards_each_generation", False))
                )
            ):
                preload_limit = int(getattr(cfg, "replay_preload_max_shards", 512))
                print(
                    f"[GV3 gen={int(gen):06d}] replay preload starting: "
                    f"num_shards={int(replay_buffer.num_shards)}, "
                    f"max_preload_shards={int(preload_limit)}",
                    flush=True,
                )
                preload_stats = replay_buffer.preload_all_shards(max_shards=preload_limit)
                print(
                    f"[GV3 gen={int(gen):06d}] replay preload complete: "
                    f"loaded_shards={int(preload_stats['loaded_shards'])}, "
                    f"cached_shards={int(preload_stats['cached_shards'])}, "
                    f"num_shards={int(preload_stats['num_shards'])}",
                    flush=True,
                )
            effective_epochs = (
                float(train_steps_this_gen) * float(int(cfg.train_batch_size))
            ) / float(max(1, int(replay_buffer.total_samples)))
            steps_per_epoch = max(
                1,
                int(
                    math.ceil(
                        float(max(1, int(replay_buffer.total_samples)))
                        / float(max(1, int(cfg.train_batch_size)))
                    )
                ),
            )
            epoch_count = max(
                1,
                int(math.ceil(float(train_steps_this_gen) / float(steps_per_epoch))),
            )
            print(
                f"[GV3 gen={int(gen):06d}] training starting: "
                f"replay_total_samples={int(replay_buffer.total_samples)}, "
                f"added_from_generation={int(added_from_gen)}, "
                f"train_steps={int(train_steps_this_gen)}, batch_size={int(cfg.train_batch_size)}, "
                f"target_epochs={float(cfg.train_target_epochs_per_generation):.3f}, "
                f"effective_epochs={float(effective_epochs):.3f}, "
                f"steps_per_epoch={int(steps_per_epoch)}, epochs={int(epoch_count)}, "
                f"progress_every_steps={int(getattr(cfg, 'train_progress_print_every_steps', 100))}, "
                f"train_threads={int(actual_train_threads)}",
                flush=True,
            )

            train_rows: list[dict] = []
            global_step = 0
            progress_print_every = max(
                1,
                int(getattr(cfg, "train_progress_print_every_steps", 100)),
            )
            progress_window_start = time.perf_counter()
            for epoch_idx in range(int(epoch_count)):
                epoch_rows: list[dict] = []
                progress_rows: list[dict] = []
                epoch_steps = min(
                    int(steps_per_epoch),
                    int(train_steps_this_gen) - int(global_step),
                )
                if epoch_steps <= 0:
                    break

                for _ in range(int(epoch_steps)):
                    samples = replay_buffer.sample_batch(int(cfg.train_batch_size))
                    batch_by_player = collate_mixed_samples(
                        samples,
                        device=trainer.device,
                        include_policy_tensors=False,
                        include_legacy_fallback=False,
                        include_ids=False,
                    )
                    row = trainer.train_step(batch_by_player)
                    train_rows.append(row)
                    epoch_rows.append(row)
                    progress_rows.append(row)
                    global_step += 1

                    if (
                        (int(global_step) % int(progress_print_every) == 0)
                        or (int(global_step) == int(train_steps_this_gen))
                    ):
                        progress_metrics = _aggregate_metric_rows(progress_rows)
                        epoch_step_idx = len(epoch_rows)
                        elapsed_sec = max(0.0, float(time.perf_counter() - progress_window_start))
                        steps_in_window = max(1, len(progress_rows))
                        print(
                            f"[GV3 gen={int(gen):06d}] training progress: "
                            f"epoch={int(epoch_idx) + 1}/{int(epoch_count)}, "
                            f"epoch_step={int(epoch_step_idx)}/{int(epoch_steps)}, "
                            f"global_step={int(global_step)}/{int(train_steps_this_gen)}, "
                            f"window_steps={int(steps_in_window)}, "
                            f"window_sec={float(elapsed_sec):.3f}, "
                            f"sec_per_step={float(elapsed_sec) / float(steps_in_window):.3f}, "
                            f"loss={float(progress_metrics['loss_for_selection']):.6f}, "
                            f"value_mse={float(progress_metrics['value_mse_error']):.6f}, "
                            f"value_mae={float(progress_metrics['value_mae_error']):.6f}",
                            flush=True,
                        )
                        progress_rows.clear()
                        progress_window_start = time.perf_counter()

                epoch_metrics = _aggregate_metric_rows(epoch_rows)
                print(
                    f"[GV3 gen={int(gen):06d}] epoch {int(epoch_idx) + 1}/{int(epoch_count)} complete: "
                    f"steps={int(global_step)}/{int(train_steps_this_gen)}, "
                    f"loss={float(epoch_metrics['loss_for_selection']):.6f}, "
                    f"value_mse={float(epoch_metrics['value_mse_error']):.6f}, "
                    f"value_mae={float(epoch_metrics['value_mae_error']):.6f}",
                    flush=True,
                )

            train_metrics = _aggregate_metric_rows(train_rows)

            eval_samples = _load_generation_samples(eval_source_dir)
            print(
                f"[GV3 gen={int(gen):06d}] evaluation starting: "
                f"eval_samples={int(len(eval_samples))}, batch_size={int(cfg.train_batch_size)}",
                flush=True,
            )
            eval_metrics = _evaluate_samples(
                trainer,
                samples=eval_samples,
                batch_size=int(cfg.train_batch_size),
            )
            eval_prediction_rows = _build_eval_prediction_rows(
                trainer,
                samples=eval_samples,
                batch_size=int(cfg.train_batch_size),
                model_version=int(gen),
            )
            EvalRootPredictionLogger(
                gen_logs_dir / "eval_root_value_predictions.csv"
            ).write_rows(eval_prediction_rows)

            gen_ckpt_path = ckpt_dir / f"gen_{gen:06d}.pt"
            trainer.save_checkpoint(gen_ckpt_path)

            latest_path = ckpt_dir / "latest.pt"
            shutil.copyfile(gen_ckpt_path, latest_path)

            select_loss = _safe_float(eval_metrics.get("loss_for_selection", float("nan")))
            if math.isnan(select_loss):
                select_loss = _safe_float(train_metrics.get("loss_for_selection", float("nan")))
            if not math.isnan(select_loss):
                trainer.best_eval_loss = float(select_loss)
            trainer.save_checkpoint(best_ckpt_path)

            common_row = {
                "generation": int(gen),
                "model_version": int(gen),
                "num_roots_required": int(cfg.roots_per_generation) * int(max(1, int(getattr(cfg, "sample_cycles_per_generation", 1)))),
                "num_unique_roots_created": int(generation_stats["num_unique_roots"]),
                "controller_training_samples": int(generation_stats["controller_train_samples"]),
                "controller_eval_samples": int(generation_stats["controller_eval_samples"]),
                "adversary_training_samples": int(generation_stats["adversary_train_samples"]),
                "adversary_eval_samples": int(generation_stats["adversary_eval_samples"]),
            }

            eval_metrics_logger.log_phase(
                **common_row,
                phase="training",
                samples_needed=int(train_metrics["samples_needed"]),
                loss_for_selection=float(train_metrics["loss_for_selection"]),
                value_loss=float(train_metrics["value_loss"]),
                controller_value_loss=float(train_metrics["controller_value_loss"]),
                adversary_value_loss=float(train_metrics["adversary_value_loss"]),
                value_mse_error=float(train_metrics["value_mse_error"]),
                value_mae_error=float(train_metrics["value_mae_error"]),
                controller_value_mse_error=float(train_metrics["controller_value_mse_error"]),
                controller_value_mae_error=float(train_metrics["controller_value_mae_error"]),
                adversary_value_mse_error=float(train_metrics["adversary_value_mse_error"]),
                adversary_value_mae_error=float(train_metrics["adversary_value_mae_error"]),
            )

            eval_metrics_logger.log_phase(
                **common_row,
                phase="eval",
                samples_needed=int(eval_metrics["samples_needed"]),
                loss_for_selection=float(eval_metrics["loss_for_selection"]),
                value_loss=float(eval_metrics["value_loss"]),
                controller_value_loss=float(eval_metrics["controller_value_loss"]),
                adversary_value_loss=float(eval_metrics["adversary_value_loss"]),
                value_mse_error=float(eval_metrics["value_mse_error"]),
                value_mae_error=float(eval_metrics["value_mae_error"]),
                controller_value_mse_error=float(eval_metrics["controller_value_mse_error"]),
                controller_value_mae_error=float(eval_metrics["controller_value_mae_error"]),
                adversary_value_mse_error=float(eval_metrics["adversary_value_mse_error"]),
                adversary_value_mae_error=float(eval_metrics["adversary_value_mae_error"]),
            )

            if reuse_dataset_generation < 0:
                replay_buffer.reset_for_new_best()
                del train_rows
                del eval_samples
                del eval_prediction_rows
                gc.collect()

    finally:
        try:
            eval_metrics_logger.close()
        except Exception:
            pass
