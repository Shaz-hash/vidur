from __future__ import annotations

import argparse
import json
import shlex
import subprocess
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


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True)


def _ssh(machine: NetworkMachineConfig, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["ssh", machine.ssh_host, remote_cmd], check=check)


def _rsync_to(machine: NetworkMachineConfig, local_path: Path, remote_path: str) -> None:
    _run(["rsync", "-az", str(local_path), f"{machine.ssh_host}:{remote_path}"])


def _rsync_from(machine: NetworkMachineConfig, remote_path: str, local_path: Path) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    _run(["rsync", "-az", f"{machine.ssh_host}:{remote_path.rstrip('/')}/", f"{local_path}/"])


def load_machines(path: Path) -> list[NetworkMachineConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"machines config must contain a list: {path}")
    return [NetworkMachineConfig(**item) for item in raw]


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
    proc_name = f"proc_{machine_tag}_{task_index:03d}"
    return NetworkSelfplayTask(
        session_id=str(session_id),
        task_id=str(task_id),
        machine_name=str(machine.name),
        machine_ip=str(machine.public_ip),
        generation=int(generation),
        model_version=int(model_version),
        weights_path=str(weights_remote_path),
        result_dir=str(remote_result_dir),
        out_dir_train=str(Path(remote_result_dir) / "train" / proc_name),
        out_dir_eval=str(Path(remote_result_dir) / "eval" / proc_name),
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
    _ssh(machine, remote_mkdir)
    _rsync_to(machine, local_weights_path, remote_weights_path)
    _rsync_to(machine, local_task_path, remote_task_path)
    append_csv_row(
        allocation_log_csv,
        ALLOCATION_COLUMNS,
        _allocation_row(task, sent_at, "sent"),
    )

    remote_cmd = (
        f"cd {shlex.quote(machine.repo_dir)} && "
        f"PYTHONPATH={shlex.quote(machine.repo_dir)} "
        f"{shlex.quote(machine.python)} -m "
        "vidur.mcts.Game_Versions.Game_Version3.Network.client.run_task "
        f"--task {shlex.quote(remote_task_path)}"
    )

    remote_status = "ok"
    remote_error = ""
    completed = _ssh(machine, remote_cmd, check=False)
    if completed.returncode != 0:
        remote_status = "failed"
        remote_error = f"remote command exited with code {completed.returncode}"

    try:
        _rsync_from(machine, remote_result_dir, local_result_dir)
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
    parser.add_argument("--num-roots-per-machine", type=int, default=defaults.num_roots_per_machine)
    parser.add_argument("--history-hops-min", type=int, default=defaults.history_hops_min)
    parser.add_argument("--history-hops-max", type=int, default=defaults.history_hops_max)
    parser.add_argument("--history-seed", type=int, default=defaults.history_seed)
    parser.add_argument("--worker-model-device", default=defaults.worker_model_device)
    parser.add_argument("--eval-split-ratio", type=float, default=defaults.eval_split_ratio)
    parser.add_argument("--shard-size", type=int, default=defaults.shard_size)
    parser.add_argument("--output-dir", default=str(paths.output_dir))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = replace(DEFAULT_NETWORK_CONFIG.paths, output_dir=Path(args.output_dir))
    task_defaults = replace(
        DEFAULT_NETWORK_CONFIG.task,
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

    machines = selected_machines(load_machines(Path(args.machines_config)), args.machine)
    if not machines:
        raise RuntimeError("No machines selected")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for idx, machine in enumerate(machines):
        task_id_preview = f"{args.session_id}_{_safe_id(machine.name)}_{idx:03d}"
        remote_base = Path(machine.repo_dir) / "simulator_output" / "Game_Version3" / "network"
        remote_task_dir = remote_base / "tasks" / args.session_id / task_id_preview
        remote_result_dir = remote_base / "results" / args.session_id / task_id_preview
        remote_task_path = str(remote_task_dir / "task.json")
        remote_weights_path = str(remote_task_dir / "weights.pt")

        task = _build_task(
            machine=machine,
            task_defaults=task_defaults,
            session_id=str(args.session_id),
            generation=int(generation),
            model_version=int(model_version),
            weights_remote_path=remote_weights_path,
            remote_result_dir=str(remote_result_dir),
            task_index=int(idx),
            num_roots=int(args.num_roots_per_machine),
            history_hops_min=int(args.history_hops_min),
            history_hops_max=int(args.history_hops_max),
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
        )
        results.append(result)

    print(json.dumps({"ok": True, "results": results}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
