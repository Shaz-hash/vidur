from __future__ import annotations

import argparse
import json
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
) -> subprocess.CompletedProcess:
    cmd_line = "+ " + " ".join(shlex.quote(x) for x in cmd)
    _log_line(process_log_path, cmd_line)
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
    task_index: int,
    num_roots: int,
    history_hops_min: int,
    history_hops_max: int,
    history_seed: int,
    worker_model_device: str,
) -> NetworkSelfplayTask:
    machine_tag = _safe_id(machine.name)
    task_id = f"{session_id}_{machine_tag}_{task_index:03d}"
    return NetworkSelfplayTask(
        session_id=str(session_id),
        task_id=str(task_id),
        machine_name=str(machine.name),
        machine_ip=str(machine.public_ip),
        generation=int(generation),
        model_version=int(model_version),
        weights_path=str(weights_remote_path),
        result_dir=str(remote_result_dir),
        out_dir_train=str(Path(remote_result_dir) / "train"),
        out_dir_eval=str(Path(remote_result_dir) / "eval"),
        logs_dir=str(Path(remote_result_dir) / "logs"),
        game_id=int(task_defaults.game_id_base) + int(task_index),
        num_roots=int(num_roots),
        start_root_id=int(task_defaults.start_root_id) + int(task_index) * int(num_roots),
        start_root_depth=int(task_defaults.start_root_depth),
        start_player=str(task_defaults.start_player),
        feature_version=int(task_defaults.feature_version),
        adv_iterations_per_root=int(task_defaults.adv_iterations_per_root),
        cont_iterations_per_root=int(task_defaults.cont_iterations_per_root),
        max_batch_size=int(task_defaults.max_batch_size),
        history_nontrivial_hops=int(history_hops_min),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        history_seed=int(history_seed) + int(task_index) * 1009,
        sample_from_mcts_policy=bool(task_defaults.sample_from_mcts_policy),
        selfplay_policy_temperature=float(task_defaults.selfplay_policy_temperature),
        action_seed_base=int(task_defaults.action_seed_base) + int(task_index) * 10_003,
        max_forced_hops_per_root=int(task_defaults.max_forced_hops_per_root),
        history_max_total_steps=int(task_defaults.history_max_total_steps),
        history_root_batch_size=int(task_defaults.history_root_batch_size),
        log_history_rows=bool(task_defaults.log_history_rows),
        eval_split_ratio=float(task_defaults.eval_split_ratio),
        eval_split_seed=int(task_defaults.eval_split_seed) + int(task_index) * 20_011,
        task_seed=int(task_defaults.task_seed_base) + int(task_index) * 30_017,
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
    process_log_path: Path | None,
) -> dict[str, Any]:
    sent_at = utc_now_iso()
    _log_line(
        process_log_path,
        (
            f"[GV3 network server] dispatch starting: task={task.task_id}, "
            f"machine={machine.name}, roots={int(task.num_roots)}, "
            f"history_range=[{int(task.history_hops_min)}, {int(task.history_hops_max)}], "
            f"model_version={int(task.model_version)}"
        ),
    )
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
    worker_header = (
        f"[GV3 network worker] task={task.task_id} machine={machine.name} "
        f"roots={int(task.num_roots)} model_version={int(task.model_version)} "
        f"history_range=[{int(task.history_hops_min)}, {int(task.history_hops_max)}] start={utc_now_iso()}"
    )
    worker_footer_prefix = f"[GV3 network worker] task={task.task_id} machine={machine.name}"
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
        f"printf '%s\\n' {shlex.quote(worker_header)} >> {shlex.quote(str(remote_process_log))}; "
        f"({worker_cmd}) 2>&1 | tee -a {shlex.quote(str(remote_process_log))}; "
        "status=${PIPESTATUS[0]}; "
        f"printf '%s exit=%s finish=%s\\n' {shlex.quote(worker_footer_prefix)} "
        f"\"$status\" \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" >> {shlex.quote(str(remote_process_log))}; "
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
    if result is not None and result.ok:
        verify_stats = _verify_local_result(local_result_dir)

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
        **verify_stats,
    }


def _parse_args() -> argparse.Namespace:
    defaults = DEFAULT_NETWORK_CONFIG.task
    paths = DEFAULT_NETWORK_CONFIG.paths
    parser = argparse.ArgumentParser(description="Dispatch GV3 self-play tasks over SSH.")
    parser.add_argument("--machines-config", default=str(paths.machines_json))
    parser.add_argument("--machine", action="append", default=None, help="Machine name/SSH host/IP to use")
    parser.add_argument("--session-id", default=f"gv3_network_{utc_now_iso().replace(':', '').replace('-', '')}")
    parser.add_argument("--generation", type=int, default=defaults.generation)
    parser.add_argument("--model-version", type=int, default=None)
    parser.add_argument("--weights-path", default=str(paths.default_weights_path))
    parser.add_argument("--total-roots", type=int, default=defaults.total_roots_per_generation)
    parser.add_argument("--num-roots-per-machine", type=int, default=defaults.num_roots_per_machine)
    parser.add_argument("--history-hops-min", type=int, default=defaults.history_hops_min)
    parser.add_argument("--history-hops-max", type=int, default=defaults.history_hops_max)
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
    parser.add_argument("--output-dir", default=str(paths.output_dir))
    parser.add_argument("--process-log-path", default=str(paths.process_log_path))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = replace(DEFAULT_NETWORK_CONFIG.paths, output_dir=Path(args.output_dir))
    task_defaults = replace(
        DEFAULT_NETWORK_CONFIG.task,
        total_roots_per_generation=int(args.total_roots),
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
    model_version = int(generation if args.model_version is None else args.model_version)
    local_weights = Path(args.weights_path).expanduser()
    if not local_weights.is_absolute():
        local_weights = repo_root() / local_weights
    if not local_weights.exists():
        raise FileNotFoundError(f"weights checkpoint not found: {local_weights}")
    process_log_path = Path(args.process_log_path).expanduser()
    if not process_log_path.is_absolute():
        process_log_path = repo_root() / process_log_path

    machines = selected_machines(load_machines(Path(args.machines_config)), args.machine)
    if not machines:
        raise RuntimeError("No machines selected")

    if int(args.total_roots) > 0:
        machine_root_counts = _partition_counts(int(args.total_roots), len(machines))
        machine_hop_ranges = _partition_inclusive_range(
            int(args.history_hops_min),
            int(args.history_hops_max),
            len(machines),
        )
        if len(machine_hop_ranges) < len(machines):
            machine_hop_ranges.extend([machine_hop_ranges[-1]] * (len(machines) - len(machine_hop_ranges)))
    else:
        machine_root_counts = [int(args.num_roots_per_machine) for _ in machines]
        machine_hop_ranges = [
            (int(args.history_hops_min), int(args.history_hops_max))
            for _ in machines
        ]

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _log_line(
        process_log_path,
        (
            f"[GV3 network server] session starting: session_id={args.session_id}, "
            f"generation={int(generation)}, model_version={int(model_version)}, "
            f"machines={','.join(m.name for m in machines)}, "
            f"total_roots={int(args.total_roots)}, "
            f"adv_iterations_per_root={int(task_defaults.adv_iterations_per_root)}, "
            f"cont_iterations_per_root={int(task_defaults.cont_iterations_per_root)}, "
            f"worker_cpu_fraction={float(task_defaults.worker_cpu_fraction):.3f}, "
            f"weights={local_weights}"
        ),
    )
    results: list[dict[str, Any]] = []

    for idx, machine in enumerate(machines):
        task_id_preview = f"{args.session_id}_{_safe_id(machine.name)}_{idx:03d}"
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
            task_index=int(idx),
            num_roots=int(machine_roots),
            history_hops_min=int(machine_hops_min),
            history_hops_max=int(machine_hops_max),
            history_seed=int(args.history_seed),
            worker_model_device=str(args.worker_model_device),
        )

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
            process_log_path=process_log_path,
        )
        results.append(result)

    _log_line(
        process_log_path,
        f"[GV3 network server] session complete: session_id={args.session_id}, tasks={len(results)}",
    )
    _log_line(process_log_path, json.dumps({"ok": True, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
