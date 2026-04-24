from __future__ import annotations

import math
import multiprocessing as mp
import os
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig
from ...multiProcessUtils import (
    _aggregate_worker_run_stats,
    _build_selfplay_cycle_task_payloads,
    _run_selfplay_cycle,
    _set_global_seeds,
    _set_runtime_cpu_thread_env,
)
from ..network_config import resolve_device
from .files import list_relative_files, utc_now_iso, write_json
from .types import NetworkSelfplayTask, NetworkTaskResult


def _task_cfg(task: NetworkSelfplayTask) -> MultipleProcessTrainingConfig:
    base = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
    return replace(
        base,
        model=replace(base.model, device=resolve_device(task.model_device)),
        dataset=replace(base.dataset, shard_size=int(task.shard_size)),
        logging=replace(
            base.logging,
            mcts_iter_log=str(Path(task.logs_dir) / "mcts_iter.csv"),
            mcts_root_log=str(Path(task.logs_dir) / "mcts_root.csv"),
        ),
        run=replace(
            base.run,
            game_id=int(task.game_id),
            root_id=int(task.start_root_id),
            root_depth=int(task.start_root_depth),
            root_player=str(task.start_player),
            feature_version=int(task.feature_version),
        ),
        num_processes=int(_resolve_worker_processes(task)),
        max_concurrent_selfplay_workers=int(_resolve_max_concurrent_workers(task)),
        max_workers_per_interval=int(task.max_workers_per_interval),
        selfplay_dynamic_chunk_roots=int(task.selfplay_dynamic_chunk_roots),
        selfplay_zero_progress_interval_patience=int(task.selfplay_zero_progress_interval_patience),
        selfplay_launch_rss_limit_gb=float(task.selfplay_launch_rss_limit_gb),
        selfplay_launch_poll_sec=float(task.selfplay_launch_poll_sec),
        roots_per_generation=int(task.num_roots),
        sample_cycles_per_generation=1,
        adv_iterations_per_root=int(task.adv_iterations_per_root),
        cont_iterations_per_root=int(task.cont_iterations_per_root),
        max_batch_size=int(task.max_batch_size),
        history_seed=int(task.history_seed),
        history_hops_min=int(task.history_hops_min),
        history_hops_max=int(task.history_hops_max),
        history_hops_per_worker=tuple(_worker_hop_ranges(task)),
        max_forced_hops_per_root=int(task.max_forced_hops_per_root),
        history_max_total_steps=int(task.history_max_total_steps),
        history_root_batch_size=int(task.history_root_batch_size),
        history_allow_duplicate_root_fallback=bool(task.allow_duplicate_history_fallback),
        log_history_rows=bool(task.log_history_rows),
        eval_split_ratio=float(task.eval_split_ratio),
        eval_split_seed_base=int(task.eval_split_seed),
        sample_from_mcts_policy=bool(task.sample_from_mcts_policy),
        selfplay_policy_temperature=float(task.selfplay_policy_temperature),
        action_seed_base=int(task.action_seed_base),
        use_virtual_env=bool(task.use_virtual_env),
        worker_result_timeout_sec=int(task.worker_result_timeout_sec),
    )


def _inclusive_hop_count(task: NetworkSelfplayTask) -> int:
    return max(1, int(task.history_hops_max) - int(task.history_hops_min) + 1)


def _resolve_worker_processes(task: NetworkSelfplayTask) -> int:
    explicit = int(task.worker_processes)
    if explicit > 0:
        requested = explicit
    else:
        cpu_count = int(os.cpu_count() or 1)
        requested = int(math.floor(float(cpu_count) * float(task.worker_cpu_fraction)))
    return max(1, min(int(requested), int(task.num_roots), _inclusive_hop_count(task)))


def _resolve_max_concurrent_workers(task: NetworkSelfplayTask) -> int:
    resolved_processes = _resolve_worker_processes(task)
    explicit = int(task.max_concurrent_workers)
    if explicit > 0:
        return max(1, min(explicit, resolved_processes))
    return resolved_processes


def _partition_inclusive_range(lo: int, hi: int, parts: int) -> list[tuple[int, int]]:
    lo_i = int(lo)
    hi_i = int(hi)
    if hi_i < lo_i:
        hi_i = lo_i
    count = int(hi_i - lo_i + 1)
    n = max(1, min(int(parts), int(count)))
    base, rem = divmod(int(count), int(n))
    out: list[tuple[int, int]] = []
    start = int(lo_i)
    for idx in range(int(n)):
        width = int(base + (1 if idx < rem else 0))
        end = int(start + width - 1)
        out.append((int(start), int(end)))
        start = int(end + 1)
    return out


def _worker_hop_ranges(task: NetworkSelfplayTask) -> list[tuple[int, int]]:
    return _partition_inclusive_range(
        int(task.history_hops_min),
        int(task.history_hops_max),
        _resolve_worker_processes(task),
    )


def _execute_multiprocess_selfplay(task: NetworkSelfplayTask) -> dict[str, Any]:
    cfg = _task_cfg(task)
    worker_hop_ranges = _worker_hop_ranges(task)
    train_dir = Path(task.out_dir_train)
    eval_dir = Path(task.out_dir_eval)
    logs_dir = Path(task.logs_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    _set_runtime_cpu_thread_env(1)
    _set_global_seeds(
        int(task.task_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )

    print(
        f"[GV3 network task={task.task_id}] multiprocess self-play starting: "
        f"roots={int(task.num_roots)}, worker_processes={int(cfg.num_processes)}, "
        f"max_concurrent={int(cfg.max_concurrent_selfplay_workers)}, "
        f"hop_range=[{int(task.history_hops_min)}, {int(task.history_hops_max)}], "
        f"worker_hop_ranges={worker_hop_ranges}, chunk_roots={int(cfg.selfplay_dynamic_chunk_roots)}",
        flush=True,
    )

    payloads = _build_selfplay_cycle_task_payloads(
        cfg,
        gen=int(task.generation),
        cycle_index=0,
        weights_path=Path(task.weights_path),
        gen_train_dir=train_dir,
        gen_eval_dir=eval_dir,
        hop_ranges=worker_hop_ranges,
        seen_history_signatures_by_worker={},
    )
    for payload in payloads:
        payload["model_version"] = int(task.model_version)

    ctx = mp.get_context("spawn")
    worker_msgs = _run_selfplay_cycle(
        ctx,
        cfg,
        gen=int(task.generation),
        cycle_index=0,
        task_payloads=payloads,
    )
    run_stats = _aggregate_worker_run_stats(worker_msgs)
    run_stats.update(
        {
            "worker_processes": int(cfg.num_processes),
            "max_concurrent_workers": int(cfg.max_concurrent_selfplay_workers),
            "max_workers_per_interval": int(cfg.max_workers_per_interval),
            "selfplay_dynamic_chunk_roots": int(cfg.selfplay_dynamic_chunk_roots),
            "worker_hop_ranges": [list(x) for x in worker_hop_ranges],
            "num_worker_messages": int(len(worker_msgs)),
        }
    )
    print(
        f"[GV3 network task={task.task_id}] multiprocess self-play complete: "
        f"roots_generated={int(run_stats['num_roots_generated'])}, "
        f"unique_roots={int(run_stats['num_unique_roots'])}, "
        f"train_samples={int(run_stats['train_samples_total'])}, "
        f"eval_samples={int(run_stats['eval_samples_total'])}",
        flush=True,
    )
    return run_stats


def execute_selfplay_task(task: NetworkSelfplayTask) -> NetworkTaskResult:
    received_at = utc_now_iso()
    started_at = received_at

    try:
        run_stats = _execute_multiprocess_selfplay(task)

        finished_at = utc_now_iso()
        result_dir = Path(task.result_dir)
        produced_files = list_relative_files(result_dir)
        result = NetworkTaskResult(
            ok=True,
            session_id=task.session_id,
            task_id=task.task_id,
            machine_name=task.machine_name,
            machine_ip=task.machine_ip,
            generation=int(task.generation),
            model_version=int(task.model_version),
            received_at_utc=received_at,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            sent_back_at_utc=finished_at,
            result_dir=str(result_dir),
            train_dir=str(Path(task.out_dir_train)),
            eval_dir=str(Path(task.out_dir_eval)),
            logs_dir=str(Path(task.logs_dir)),
            run_stats=dict(run_stats),
            produced_files=produced_files,
        )
        write_json(result_dir / "result.json", result.to_dict())
        return result

    except Exception as exc:
        finished_at = utc_now_iso()
        result = NetworkTaskResult(
            ok=False,
            session_id=task.session_id,
            task_id=task.task_id,
            machine_name=task.machine_name,
            machine_ip=task.machine_ip,
            generation=int(task.generation),
            model_version=int(task.model_version),
            received_at_utc=received_at,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            sent_back_at_utc=finished_at,
            result_dir=str(task.result_dir),
            train_dir=str(task.out_dir_train),
            eval_dir=str(task.out_dir_eval),
            logs_dir=str(task.logs_dir),
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_json(Path(task.result_dir) / "result.json", result.to_dict())
        return result
