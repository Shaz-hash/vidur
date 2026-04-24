from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..common.csv_logs import (
    ALLOCATION_COLUMNS,
    RECEIVED_COLUMNS,
    append_csv_row,
)
from ..common.files import (
    count_partition_samples,
    read_json,
    rewrite_manifest_paths_absolute,
    utc_now_iso,
    write_json,
)
from ..common.types import NetworkSelfplayTask, NetworkTaskResult
from ..network_config import (
    DEFAULT_NETWORK_CONFIG,
    NetworkMachineConfig,
    NetworkTaskDefaults,
    repo_root,
    selected_machines,
)
from .training import NetworkTrainingSummary, train_network_generation


def _safe_id(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "machine"


def _append_process_log(path: Path | None, text: str) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


def _log_line(path: Path | None, message: str) -> None:
    print(message, flush=True)
    _append_process_log(path, message + "\n")


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    process_log_path: Path | None = None,
    log_output_to_process_log: bool = True,
    log_command_to_process_log: bool = False,
) -> subprocess.CompletedProcess:
    cmd_line = "+ " + " ".join(shlex.quote(x) for x in cmd)
    print(cmd_line, flush=True)
    if log_command_to_process_log:
        _append_process_log(process_log_path, cmd_line + "\n")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_output_to_process_log:
            _append_process_log(process_log_path, line)
    returncode = proc.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, output="".join(output))
    return subprocess.CompletedProcess(cmd, returncode, stdout="".join(output))


def _ssh(
    machine: NetworkMachineConfig,
    remote_cmd: str,
    *,
    check: bool = True,
    process_log_path: Path | None = None,
    log_output_to_process_log: bool = True,
) -> subprocess.CompletedProcess:
    return _run(
        ["ssh", machine.ssh_host, remote_cmd],
        check=check,
        process_log_path=process_log_path,
        log_output_to_process_log=log_output_to_process_log,
    )


def _rsync_to(
    machine: NetworkMachineConfig,
    local_path: Path,
    remote_path: str,
    *,
    process_log_path: Path | None = None,
    log_output_to_process_log: bool = True,
) -> None:
    _run(
        ["rsync", "-az", str(local_path), f"{machine.ssh_host}:{remote_path}"],
        process_log_path=process_log_path,
        log_output_to_process_log=log_output_to_process_log,
    )


def _rsync_from(
    machine: NetworkMachineConfig,
    remote_path: str,
    local_path: Path,
    *,
    process_log_path: Path | None = None,
    log_output_to_process_log: bool = True,
) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    _run(
        ["rsync", "-az", f"{machine.ssh_host}:{remote_path.rstrip('/')}/", f"{local_path}/"],
        process_log_path=process_log_path,
        log_output_to_process_log=log_output_to_process_log,
    )


def load_machines(path: Path) -> list[NetworkMachineConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"machines config must contain a list: {path}")
    return [NetworkMachineConfig(**item) for item in raw]


def _partition_counts(total: int, parts: int) -> list[int]:
    n = max(1, int(parts))
    base, rem = divmod(max(0, int(total)), int(n))
    return [int(base + (1 if idx < rem else 0)) for idx in range(int(n))]


def _partition_inclusive_range(lo: int, hi: int, parts: int) -> list[tuple[int, int]]:
    lo_i = int(lo)
    hi_i = max(int(hi), int(lo_i))
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


def _build_task(
    *,
    machine: NetworkMachineConfig,
    task_defaults: NetworkTaskDefaults,
    session_id: str,
    generation: int,
    model_version: int,
    weights_remote_path: str,
    remote_result_dir: str,
    cycle_index: int,
    sample_cycles_per_generation: int,
    task_index: int,
    num_roots: int,
    start_root_id: int,
    history_hops_min: int,
    history_hops_max: int,
    history_seed: int,
    worker_model_device: str,
) -> NetworkSelfplayTask:
    machine_tag = _safe_id(machine.name)
    task_id = f"{session_id}_c{int(cycle_index):03d}_{machine_tag}_{task_index:03d}"
    return NetworkSelfplayTask(
        session_id=str(session_id),
        task_id=str(task_id),
        machine_name=str(machine.name),
        machine_ip=str(machine.public_ip),
        generation=int(generation),
        cycle_index=int(cycle_index),
        sample_cycles_per_generation=int(sample_cycles_per_generation),
        model_version=int(model_version),
        weights_path=str(weights_remote_path),
        result_dir=str(remote_result_dir),
        out_dir_train=str(Path(remote_result_dir) / "train"),
        out_dir_eval=str(Path(remote_result_dir) / "eval"),
        logs_dir=str(Path(remote_result_dir) / "logs"),
        game_id=int(task_defaults.game_id_base) + int(cycle_index) * 100_000 + int(task_index),
        num_roots=int(num_roots),
        start_root_id=int(start_root_id),
        start_root_depth=int(task_defaults.start_root_depth),
        start_player=str(task_defaults.start_player),
        feature_version=int(task_defaults.feature_version),
        adv_iterations_per_root=int(task_defaults.adv_iterations_per_root),
        cont_iterations_per_root=int(task_defaults.cont_iterations_per_root),
        max_batch_size=int(task_defaults.max_batch_size),
        history_nontrivial_hops=int(history_hops_min),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        history_hop_interval_width=int(task_defaults.history_hop_interval_width),
        history_seed=int(history_seed) + int(cycle_index) * 10_000 + int(task_index) * 1009,
        sample_from_mcts_policy=bool(task_defaults.sample_from_mcts_policy),
        selfplay_policy_temperature=float(task_defaults.selfplay_policy_temperature),
        action_seed_base=int(task_defaults.action_seed_base) + int(cycle_index) * 100_003 + int(task_index) * 10_003,
        max_forced_hops_per_root=int(task_defaults.max_forced_hops_per_root),
        history_max_total_steps=int(task_defaults.history_max_total_steps),
        history_root_batch_size=int(task_defaults.history_root_batch_size),
        log_history_rows=bool(task_defaults.log_history_rows),
        eval_split_ratio=float(task_defaults.eval_split_ratio),
        eval_split_seed=int(task_defaults.eval_split_seed) + int(cycle_index) * 200_003 + int(task_index) * 20_011,
        task_seed=int(task_defaults.task_seed_base) + int(cycle_index) * 300_007 + int(task_index) * 30_017,
        shard_size=int(task_defaults.shard_size),
        worker_cpu_fraction=float(task_defaults.worker_cpu_fraction),
        worker_processes=int(task_defaults.worker_processes),
        max_concurrent_workers=int(task_defaults.max_concurrent_workers),
        max_workers_per_interval=int(task_defaults.max_workers_per_interval),
        selfplay_dynamic_chunk_roots=int(task_defaults.selfplay_dynamic_chunk_roots),
        selfplay_zero_progress_interval_patience=int(task_defaults.selfplay_zero_progress_interval_patience),
        selfplay_launch_rss_limit_gb=float(task_defaults.selfplay_launch_rss_limit_gb),
        selfplay_launch_poll_sec=float(task_defaults.selfplay_launch_poll_sec),
        worker_result_timeout_sec=int(task_defaults.worker_result_timeout_sec),
        model_device=str(worker_model_device),
        use_virtual_env=bool(task_defaults.use_virtual_env),
        allow_duplicate_history_fallback=bool(task_defaults.allow_duplicate_history_fallback),
    )


def _allocation_row(task: NetworkSelfplayTask, sent_at: str, status: str) -> dict[str, Any]:
    return {
        "session_id": task.session_id,
        "task_id": task.task_id,
        "machine_name": task.machine_name,
        "machine_ip": task.machine_ip,
        "sent_at_utc": sent_at,
        "generation": int(task.generation),
        "model_version": int(task.model_version),
        "num_roots": int(task.num_roots),
        "history_hops_min": int(task.history_hops_min),
        "history_hops_max": int(task.history_hops_max),
        "history_seed": int(task.history_seed),
        "start_root_id": int(task.start_root_id),
        "game_id": int(task.game_id),
        "status": status,
    }


def _received_row(
    *,
    task: NetworkSelfplayTask,
    result: NetworkTaskResult | None,
    local_result_dir: Path,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    run_stats = dict(result.run_stats or {}) if result is not None else {}
    return {
        "session_id": task.session_id,
        "task_id": task.task_id,
        "machine_name": task.machine_name,
        "machine_ip": task.machine_ip,
        "received_at_utc": utc_now_iso(),
        "generation": int(task.generation),
        "model_version": int(task.model_version),
        "num_roots": int(task.num_roots),
        "history_hops_min": int(task.history_hops_min),
        "history_hops_max": int(task.history_hops_max),
        "history_seed": int(task.history_seed),
        "roots_generated": int(run_stats.get("num_roots_generated", 0)),
        "train_samples": int(run_stats.get("train_samples_total", 0)),
        "eval_samples": int(run_stats.get("eval_samples_total", 0)),
        "result_dir": str(local_result_dir),
        "status": status,
        "error": error,
    }


def _read_result(local_result_dir: Path) -> NetworkTaskResult | None:
    path = Path(local_result_dir) / "result.json"
    if not path.exists():
        return None
    return NetworkTaskResult.from_dict(read_json(path))


def _verify_local_result(local_result_dir: Path) -> dict[str, int]:
    train_dir = Path(local_result_dir) / "train"
    eval_dir = Path(local_result_dir) / "eval"
    rewrite_manifest_paths_absolute(train_dir)
    rewrite_manifest_paths_absolute(eval_dir)
    train = count_partition_samples(train_dir)
    eval_ = count_partition_samples(eval_dir)
    return {
        "train_samples": int(train["samples_total"]),
        "eval_samples": int(eval_["samples_total"]),
        "samples_total": int(train["samples_total"]) + int(eval_["samples_total"]),
        "train_shards": int(train["num_shards"]),
        "eval_shards": int(eval_["num_shards"]),
    }


def _copy_proc_dirs_to_generation(
    *,
    src_partition_dir: Path,
    dst_partition_dir: Path,
    task_id: str,
) -> int:
    src = Path(src_partition_dir)
    dst = Path(dst_partition_dir)
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    task_tag = _safe_id(task_id)
    for proc_dir in sorted(src.glob("proc_*")):
        if not proc_dir.is_dir():
            continue
        dst_name = f"proc_net_{task_tag}_{proc_dir.name[5:]}"
        dst_dir = dst / dst_name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(proc_dir, dst_dir)
        copied += 1
    if copied:
        rewrite_manifest_paths_absolute(dst)
    return copied


def _copy_logs_to_generation(
    *,
    src_logs_dir: Path,
    dst_gen_logs_dir: Path,
    task_id: str,
) -> int:
    src = Path(src_logs_dir)
    if not src.exists():
        return 0
    dst = Path(dst_gen_logs_dir) / f"network_{_safe_id(task_id)}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def _mirror_result_to_generation_layout(
    *,
    task: NetworkSelfplayTask,
    local_result_dir: Path,
    dataset_dir: Path,
    logs_dir: Path,
) -> dict[str, int | str]:
    gen_dataset_dir = Path(dataset_dir) / f"gen_{int(task.generation):06d}"
    gen_train_dir = gen_dataset_dir / "train"
    gen_eval_dir = gen_dataset_dir / "eval"
    gen_logs_dir = Path(logs_dir) / f"gen_{int(task.generation):06d}"

    train_proc_dirs = _copy_proc_dirs_to_generation(
        src_partition_dir=Path(local_result_dir) / "train",
        dst_partition_dir=gen_train_dir,
        task_id=task.task_id,
    )
    eval_proc_dirs = _copy_proc_dirs_to_generation(
        src_partition_dir=Path(local_result_dir) / "eval",
        dst_partition_dir=gen_eval_dir,
        task_id=task.task_id,
    )
    copied_logs = _copy_logs_to_generation(
        src_logs_dir=Path(local_result_dir) / "logs",
        dst_gen_logs_dir=gen_logs_dir,
        task_id=task.task_id,
    )
    return {
        "canonical_train_dir": str(gen_train_dir),
        "canonical_eval_dir": str(gen_eval_dir),
        "canonical_logs_dir": str(gen_logs_dir),
        "canonical_train_proc_dirs": int(train_proc_dirs),
        "canonical_eval_proc_dirs": int(eval_proc_dirs),
        "canonical_log_files": int(copied_logs),
    }


def _aggregate_result_stats(results: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "num_roots_requested",
        "num_roots_generated",
        "num_unique_roots",
        "controller_train_samples",
        "controller_eval_samples",
        "adversary_train_samples",
        "adversary_eval_samples",
        "train_samples_total",
        "eval_samples_total",
    )
    out = {k: 0 for k in keys}
    for result in results:
        stats = dict(result.get("run_stats", {}) or {})
        for key in keys:
            out[key] += int(stats.get(key, 0))
    return out


def dispatch_task(
    *,
    machine: NetworkMachineConfig,
    task: NetworkSelfplayTask,
    local_task_path: Path,
    local_weights_path: Path,
    local_result_dir: Path,
    remote_task_path: str,
    remote_weights_path: str,
    remote_result_dir: str,
    allocation_log_csv: Path,
    received_log_csv: Path,
    dataset_dir: Path,
    logs_dir: Path,
    process_log_path: Path | None,
) -> dict[str, Any]:
    sent_at = utc_now_iso()
    local_task_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(local_task_path, task.to_dict())
    append_csv_row(
        allocation_log_csv,
        ALLOCATION_COLUMNS,
        _allocation_row(task, sent_at, "sending"),
    )

    remote_mkdir = "mkdir -p " + " ".join(
        shlex.quote(x)
        for x in [
            str(Path(remote_task_path).parent),
            str(Path(remote_weights_path).parent),
            str(remote_result_dir),
        ]
    )
    _ssh(machine, remote_mkdir, process_log_path=process_log_path)
    _rsync_to(machine, local_weights_path, remote_weights_path, process_log_path=process_log_path)
    _rsync_to(machine, local_task_path, remote_task_path, process_log_path=process_log_path)
    append_csv_row(
        allocation_log_csv,
        ALLOCATION_COLUMNS,
        _allocation_row(task, sent_at, "sent"),
    )

    remote_process_log = (
        Path(machine.repo_dir) / "simulator_output" / "Game_Version3" / "mcts_dnn_logs" / "alphaZeroParrallel.out"
    )
    worker_cmd = (
        f"cd {shlex.quote(machine.repo_dir)} && "
        f"PYTHONPATH={shlex.quote(machine.repo_dir)} "
        f"{shlex.quote(machine.python)} -u -m "
        "vidur.mcts.Game_Versions.Game_Version3.Network.client.run_task "
        f"--task {shlex.quote(remote_task_path)}"
    )
    remote_inner = (
        "set -o pipefail; "
        f"mkdir -p {shlex.quote(str(remote_process_log.parent))}; "
        f"({worker_cmd}) 2>&1 | tee -a {shlex.quote(str(remote_process_log))}; "
        "status=${PIPESTATUS[0]}; "
        "exit $status"
    )
    remote_cmd = f"bash -lc {shlex.quote(remote_inner)}"

    remote_status = "ok"
    remote_error = ""
    completed = _ssh(
        machine,
        remote_cmd,
        check=False,
        process_log_path=process_log_path,
        log_output_to_process_log=False,
    )
    if completed.returncode != 0:
        remote_status = "failed"
        remote_error = f"remote command exited with code {completed.returncode}"

    try:
        _rsync_from(machine, remote_result_dir, local_result_dir, process_log_path=process_log_path)
    except Exception as exc:
        remote_status = "failed"
        remote_error = f"{remote_error}; pull failed: {exc}".strip("; ")

    result = _read_result(local_result_dir)
    if result is not None and not result.ok:
        remote_status = "failed"
        remote_error = result.error

    verify_stats: dict[str, int] = {}
    canonical_stats: dict[str, int | str] = {}
    if result is not None and result.ok:
        verify_stats = _verify_local_result(local_result_dir)
        canonical_stats = _mirror_result_to_generation_layout(
            task=task,
            local_result_dir=local_result_dir,
            dataset_dir=dataset_dir,
            logs_dir=logs_dir,
        )

    append_csv_row(
        received_log_csv,
        RECEIVED_COLUMNS,
        _received_row(
            task=task,
            result=result,
            local_result_dir=local_result_dir,
            status=remote_status,
            error=remote_error,
        ),
    )

    if remote_status != "ok":
        raise RuntimeError(f"Network task {task.task_id} failed: {remote_error}")

    _log_line(
        process_log_path,
        (
            f"[GV3 network server] dispatch complete: task={task.task_id}, "
            f"machine={machine.name}, samples_total={int(verify_stats.get('samples_total', 0))}, "
            f"train_samples={int(verify_stats.get('train_samples', 0))}, "
            f"eval_samples={int(verify_stats.get('eval_samples', 0))}"
        ),
    )
    return {
        "task_id": task.task_id,
        "machine": machine.name,
        "local_result_dir": str(local_result_dir),
        "run_stats": dict(result.run_stats or {}) if result is not None else {},
        **verify_stats,
        **canonical_stats,
    }


def _parse_args() -> argparse.Namespace:
    defaults = DEFAULT_NETWORK_CONFIG.task
    paths = DEFAULT_NETWORK_CONFIG.paths
    parser = argparse.ArgumentParser(description="Dispatch GV3 self-play tasks over SSH.")
    parser.add_argument("--machines-config", default=str(paths.machines_json))
    parser.add_argument("--machine", action="append", default=None, help="Machine name/SSH host/IP to use")
    parser.add_argument("--session-id", default=f"gv3_network_{utc_now_iso().replace(':', '').replace('-', '')}")
    parser.add_argument("--generation", type=int, default=defaults.generation)
    parser.add_argument("--num-generations", type=int, default=defaults.num_generations)
    parser.add_argument("--model-version", type=int, default=None)
    parser.add_argument("--weights-path", default=str(paths.default_weights_path))
    parser.add_argument("--total-roots", type=int, default=defaults.total_roots_per_generation)
    parser.add_argument("--roots-per-cycle", type=int, default=defaults.roots_per_cycle)
    parser.add_argument("--sample-cycles-per-generation", type=int, default=defaults.sample_cycles_per_generation)
    parser.add_argument("--num-roots-per-machine", type=int, default=defaults.num_roots_per_machine)
    parser.add_argument("--history-hops-min", type=int, default=defaults.history_hops_min)
    parser.add_argument("--history-hops-max", type=int, default=defaults.history_hops_max)
    parser.add_argument("--history-hop-interval-width", type=int, default=defaults.history_hop_interval_width)
    parser.add_argument("--history-seed", type=int, default=defaults.history_seed)
    parser.add_argument("--worker-model-device", default=defaults.worker_model_device)
    parser.add_argument("--worker-cpu-fraction", type=float, default=defaults.worker_cpu_fraction)
    parser.add_argument("--worker-processes", type=int, default=defaults.worker_processes)
    parser.add_argument("--max-concurrent-workers", type=int, default=defaults.max_concurrent_workers)
    parser.add_argument("--max-workers-per-interval", type=int, default=defaults.max_workers_per_interval)
    parser.add_argument("--selfplay-dynamic-chunk-roots", type=int, default=defaults.selfplay_dynamic_chunk_roots)
    parser.add_argument(
        "--selfplay-zero-progress-interval-patience",
        type=int,
        default=defaults.selfplay_zero_progress_interval_patience,
    )
    parser.add_argument("--selfplay-launch-rss-limit-gb", type=float, default=defaults.selfplay_launch_rss_limit_gb)
    parser.add_argument("--selfplay-launch-poll-sec", type=float, default=defaults.selfplay_launch_poll_sec)
    parser.add_argument("--worker-result-timeout-sec", type=int, default=defaults.worker_result_timeout_sec)
    parser.add_argument("--adv-iterations-per-root", type=int, default=defaults.adv_iterations_per_root)
    parser.add_argument("--cont-iterations-per-root", type=int, default=defaults.cont_iterations_per_root)
    parser.add_argument("--eval-split-ratio", type=float, default=defaults.eval_split_ratio)
    parser.add_argument("--shard-size", type=int, default=defaults.shard_size)
    parser.add_argument("--local-training-device", default=defaults.local_training_device)
    parser.add_argument("--skip-training", action="store_true", default=not bool(defaults.train_after_collection))
    parser.add_argument("--train-batch-size", type=int, default=defaults.local_train_batch_size)
    parser.add_argument("--train-target-epochs", type=float, default=defaults.local_train_target_epochs_per_generation)
    parser.add_argument(
        "--train-progress-every-steps",
        type=int,
        default=defaults.local_train_progress_print_every_steps,
    )
    parser.add_argument("--train-num-threads", type=int, default=defaults.local_train_num_threads)
    parser.add_argument("--replay-capacity-samples", type=int, default=defaults.local_replay_capacity_samples)
    parser.add_argument("--replay-max-cached-shards", type=int, default=defaults.local_replay_max_cached_shards)
    parser.add_argument("--replay-seed", type=int, default=defaults.local_replay_seed)
    parser.add_argument("--output-dir", default=str(paths.output_dir))
    parser.add_argument("--process-log-path", default=str(paths.process_log_path))
    parser.add_argument("--dataset-dir", default=str(paths.dataset_dir))
    parser.add_argument("--logs-dir", default=str(paths.logs_dir))
    parser.add_argument("--eval-metrics-csv", default=str(paths.eval_metrics_csv))
    parser.add_argument("--checkpoints-dir", default=str(paths.checkpoints_dir))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = replace(DEFAULT_NETWORK_CONFIG.paths, output_dir=Path(args.output_dir))
    task_defaults = replace(
        DEFAULT_NETWORK_CONFIG.task,
        total_roots_per_generation=int(args.total_roots),
        roots_per_cycle=int(args.roots_per_cycle),
        sample_cycles_per_generation=int(args.sample_cycles_per_generation),
        history_hop_interval_width=int(args.history_hop_interval_width),
        adv_iterations_per_root=int(args.adv_iterations_per_root),
        cont_iterations_per_root=int(args.cont_iterations_per_root),
        worker_cpu_fraction=float(args.worker_cpu_fraction),
        worker_processes=int(args.worker_processes),
        max_concurrent_workers=int(args.max_concurrent_workers),
        max_workers_per_interval=int(args.max_workers_per_interval),
        selfplay_dynamic_chunk_roots=int(args.selfplay_dynamic_chunk_roots),
        selfplay_zero_progress_interval_patience=int(args.selfplay_zero_progress_interval_patience),
        selfplay_launch_rss_limit_gb=float(args.selfplay_launch_rss_limit_gb),
        selfplay_launch_poll_sec=float(args.selfplay_launch_poll_sec),
        worker_result_timeout_sec=int(args.worker_result_timeout_sec),
        eval_split_ratio=float(args.eval_split_ratio),
        shard_size=int(args.shard_size),
    )
    generation = int(args.generation)
    model_version = int((generation + 1) if args.model_version is None else args.model_version)
    local_weights = Path(args.weights_path).expanduser()
    if not local_weights.is_absolute():
        local_weights = repo_root() / local_weights
    if not local_weights.exists():
        raise FileNotFoundError(f"weights checkpoint not found: {local_weights}")
    process_log_path = Path(args.process_log_path).expanduser()
    if not process_log_path.is_absolute():
        process_log_path = repo_root() / process_log_path
    dataset_dir = Path(args.dataset_dir).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root() / dataset_dir
    logs_dir = Path(args.logs_dir).expanduser()
    if not logs_dir.is_absolute():
        logs_dir = repo_root() / logs_dir
    eval_metrics_csv = Path(args.eval_metrics_csv).expanduser()
    if not eval_metrics_csv.is_absolute():
        eval_metrics_csv = repo_root() / eval_metrics_csv
    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = repo_root() / checkpoints_dir

    machines = selected_machines(load_machines(Path(args.machines_config)), args.machine)
    if not machines:
        raise RuntimeError("No machines selected")

    cycles = max(1, int(args.sample_cycles_per_generation))
    if int(args.roots_per_cycle) > 0:
        roots_per_cycle = int(args.roots_per_cycle)
    elif int(args.total_roots) > 0:
        roots_per_cycle = int((int(args.total_roots) + int(cycles) - 1) // int(cycles))
    else:
        roots_per_cycle = int(args.num_roots_per_machine) * int(len(machines))
    total_roots_required = int(roots_per_cycle) * int(cycles)
    machine_hop_ranges = _partition_inclusive_range(
        int(args.history_hops_min),
        int(args.history_hops_max),
        len(machines),
    )
    if len(machine_hop_ranges) < len(machines):
        machine_hop_ranges.extend([machine_hop_ranges[-1]] * (len(machines) - len(machine_hop_ranges)))

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    selfplay_weights_path = checkpoints_dir / f"selfplay_weights_gen_{int(generation):06d}.pt"
    shutil.copyfile(local_weights, selfplay_weights_path)
    _log_line(
        process_log_path,
        (
            f"[GV3 network server] session starting: session_id={args.session_id}, "
            f"generation={int(generation)}, model_version={int(model_version)}, "
            f"machines={','.join(m.name for m in machines)}, "
            f"roots_per_cycle={int(roots_per_cycle)}, cycles={int(cycles)}, "
            f"total_roots_required={int(total_roots_required)}, "
            f"adv_iterations_per_root={int(task_defaults.adv_iterations_per_root)}, "
            f"cont_iterations_per_root={int(task_defaults.cont_iterations_per_root)}, "
            f"worker_cpu_fraction={float(task_defaults.worker_cpu_fraction):.3f}, "
            f"weights={local_weights}"
        ),
    )
    results: list[dict[str, Any]] = []

    for cycle_index in range(int(cycles)):
        machine_root_counts = _partition_counts(int(roots_per_cycle), len(machines))
        cycle_root_base = int(task_defaults.start_root_id) + int(cycle_index) * int(roots_per_cycle)
        next_root_id = int(cycle_root_base)
        for idx, machine in enumerate(machines):
            task_id_preview = f"{args.session_id}_c{int(cycle_index):03d}_{_safe_id(machine.name)}_{idx:03d}"
            remote_base = Path(machine.repo_dir) / "simulator_output" / "Game_Version3" / "network"
            remote_task_dir = remote_base / "tasks" / args.session_id / task_id_preview
            remote_result_dir = remote_base / "results" / args.session_id / task_id_preview
            remote_task_path = str(remote_task_dir / "task.json")
            remote_weights_path = str(remote_task_dir / "weights.pt")

            machine_roots = int(machine_root_counts[idx])
            machine_hops_min, machine_hops_max = machine_hop_ranges[idx]
            task = _build_task(
                machine=machine,
                task_defaults=task_defaults,
                session_id=str(args.session_id),
                generation=int(generation),
                model_version=int(model_version),
                weights_remote_path=remote_weights_path,
                remote_result_dir=str(remote_result_dir),
                cycle_index=int(cycle_index),
                sample_cycles_per_generation=int(cycles),
                task_index=int(idx),
                num_roots=int(machine_roots),
                start_root_id=int(next_root_id),
                history_hops_min=int(machine_hops_min),
                history_hops_max=int(machine_hops_max),
                history_seed=int(args.history_seed),
                worker_model_device=str(args.worker_model_device),
            )
            next_root_id += int(machine_roots)

            local_task_path = paths.server_tasks_dir / args.session_id / task.task_id / "task.json"
            local_result_dir = paths.received_dir / args.session_id / task.task_id
            result = dispatch_task(
                machine=machine,
                task=task,
                local_task_path=local_task_path,
                local_weights_path=local_weights,
                local_result_dir=local_result_dir,
                remote_task_path=remote_task_path,
                remote_weights_path=remote_weights_path,
                remote_result_dir=str(remote_result_dir),
                allocation_log_csv=paths.allocation_log_csv,
                received_log_csv=paths.received_log_csv,
                dataset_dir=dataset_dir,
                logs_dir=logs_dir,
                process_log_path=process_log_path,
            )
            results.append(result)
        cycle_results = results[-len(machines) :]
        cycle_stats = _aggregate_result_stats(cycle_results)
        _log_line(
            process_log_path,
            (
                f"[GV3 network server] collection cycle {int(cycle_index) + 1}/{int(cycles)} received: "
                f"roots_generated={int(cycle_stats.get('num_roots_generated', 0))}, "
                f"unique_roots={int(cycle_stats.get('num_unique_roots', 0))}, "
                f"train_samples={int(cycle_stats.get('train_samples_total', 0))}, "
                f"eval_samples={int(cycle_stats.get('eval_samples_total', 0))}"
            ),
        )

    aggregate_stats = _aggregate_result_stats(results)
    training_summary: NetworkTrainingSummary | None = None
    if not bool(args.skip_training):
        training_summary = train_network_generation(
            generation=int(generation),
            model_version=int(model_version),
            total_roots_required=int(total_roots_required),
            roots_per_cycle=int(roots_per_cycle),
            sample_cycles_per_generation=int(cycles),
            dataset_dir=dataset_dir,
            logs_dir=logs_dir,
            eval_metrics_csv=eval_metrics_csv,
            checkpoints_dir=checkpoints_dir,
            initial_checkpoint_path=local_weights,
            collection_stats=aggregate_stats,
            local_training_device=str(args.local_training_device),
            train_batch_size=int(args.train_batch_size),
            train_target_epochs=float(args.train_target_epochs),
            train_progress_every_steps=int(args.train_progress_every_steps),
            train_num_threads=int(args.train_num_threads),
            replay_capacity_samples=int(args.replay_capacity_samples),
            replay_max_cached_shards=int(args.replay_max_cached_shards),
            replay_seed=int(args.replay_seed),
            log_line=lambda message: _log_line(process_log_path, message),
        )
    _log_line(
        process_log_path,
        (
            f"[GV3 network server] session complete: session_id={args.session_id}, tasks={len(results)}, "
            f"roots_generated={int(aggregate_stats.get('num_roots_generated', 0))}, "
            f"unique_roots={int(aggregate_stats.get('num_unique_roots', 0))}, "
            f"train_samples={int(aggregate_stats.get('train_samples_total', 0))}, "
            f"eval_samples={int(aggregate_stats.get('eval_samples_total', 0))}"
            + (
                f", train_steps={int(training_summary.train_steps)}, checkpoint={training_summary.checkpoint_path}"
                if training_summary is not None
                else ", training=skipped"
            )
        ),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "generation": int(generation),
                "roots_per_cycle": int(roots_per_cycle),
                "sample_cycles_per_generation": int(cycles),
                "total_roots_required": int(total_roots_required),
                "aggregate_stats": aggregate_stats,
                "training_summary": training_summary.__dict__ if training_summary is not None else None,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
