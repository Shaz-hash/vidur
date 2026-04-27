from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..common.csv_logs import CLEANUP_COLUMNS, append_csv_row
from ..common.files import utc_now_iso
from ..network_config import (
    DEFAULT_NETWORK_CONFIG,
    NetworkMachineConfig,
    namespace_path_defaults,
    resolve_output_name,
    selected_machines,
)


def load_machines(path: Path) -> list[NetworkMachineConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"machines config must contain a list: {path}")
    return [NetworkMachineConfig(**item) for item in raw]


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True)


def _ssh(machine: NetworkMachineConfig, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["ssh", machine.ssh_host, remote_cmd], check=check)


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_received_rows(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _machine_for_row(
    row: dict[str, str],
    machines: list[NetworkMachineConfig],
) -> NetworkMachineConfig | None:
    row_name = str(row.get("machine_name", "")).strip()
    row_ip = str(row.get("machine_ip", "")).strip()
    for machine in machines:
        if row_name and row_name in {machine.name, machine.ssh_host}:
            return machine
        if row_ip and row_ip == machine.public_ip:
            return machine
    return None


def _select_received_rows(
    rows: list[dict[str, str]],
    *,
    machines: list[NetworkMachineConfig],
    session_id: str | None,
    generation: int | None,
    model_version: int | None,
) -> list[tuple[dict[str, str], NetworkMachineConfig]]:
    selected: list[tuple[dict[str, str], NetworkMachineConfig]] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        if session_id is not None and str(row.get("session_id", "")) != str(session_id):
            continue
        if generation is not None and _as_int(row.get("generation")) != int(generation):
            continue
        if model_version is not None and _as_int(row.get("model_version")) != int(model_version):
            continue
        machine = _machine_for_row(row, machines)
        if machine is None:
            continue
        task_id = str(row.get("task_id", "")).strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        selected.append((row, machine))
    return selected


def _ensure_within(path: Path, root: Path) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve()
    if resolved_root in resolved_path.parents:
        return resolved_path
    raise ValueError(f"refusing to delete path outside {resolved_root}: {resolved_path}")


def _remote_dirs(
    row: dict[str, str],
    machine: NetworkMachineConfig,
    *,
    remote_output_name: str,
) -> tuple[str, str]:
    session_id = str(row.get("session_id", "")).strip()
    task_id = str(row.get("task_id", "")).strip()
    if not session_id or not task_id:
        raise ValueError(f"received log row is missing session_id/task_id: {row}")
    remote_base = Path(machine.repo_dir) / "simulator_output" / str(remote_output_name) / "network"
    remote_result_dir = remote_base / "results" / session_id / task_id
    remote_task_dir = remote_base / "tasks" / session_id / task_id
    return str(remote_result_dir), str(remote_task_dir)


def _delete_local_received(local_result_dir: Path, received_root: Path, *, dry_run: bool) -> str:
    safe_path = _ensure_within(local_result_dir, received_root)
    if not safe_path.exists():
        return "local_missing"
    if dry_run:
        return "local_would_delete"
    shutil.rmtree(safe_path)
    return "local_deleted"


def _delete_remote_dirs(
    machine: NetworkMachineConfig,
    remote_dirs: list[str],
    *,
    dry_run: bool,
) -> str:
    if not remote_dirs:
        return "remote_skipped"
    if dry_run:
        return "remote_would_delete"
    remote_cmd = "rm -rf -- " + " ".join(shlex.quote(x) for x in remote_dirs)
    completed = _ssh(machine, remote_cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"remote cleanup failed on {machine.name}: exit={completed.returncode}")
    return "remote_deleted"


def cleanup_consumed_generation_samples(
    *,
    session_id: str | None = None,
    generation: int | None = None,
    model_version: int | None = None,
    machine_names: list[str] | None = None,
    output_dir: Path | None = None,
    machines_config: Path | None = None,
    output_name: str | None = None,
    remote_output_name: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    if session_id is None and generation is None:
        raise ValueError("cleanup requires at least one scope: session_id or generation")

    cfg = DEFAULT_NETWORK_CONFIG
    if not bool(cfg.cleanup.cleanup_after_generation_done):
        return {
            "ok": True,
            "cleanup_enabled": False,
            "dry_run": bool(dry_run),
            "selected": 0,
            "results": [],
        }

    paths = cfg.paths
    if output_name is not None and output_dir is None:
        paths = replace(paths, output_dir=namespace_path_defaults(output_name)["output_dir"])
    elif output_dir is not None:
        paths = replace(paths, output_dir=Path(output_dir))
    if remote_output_name is None:
        if output_name is not None:
            remote_output_name = output_name
        elif output_dir is not None:
            out_path = Path(output_dir)
            remote_output_name = out_path.parent.name if out_path.name == "network" else out_path.name
        else:
            remote_output_name = "Game_Version3"
    remote_output_name = resolve_output_name(remote_output_name)
    machines_path = Path(machines_config or paths.machines_json)
    machines = selected_machines(load_machines(machines_path), machine_names)
    received_rows = _read_received_rows(paths.received_log_csv)
    selected = _select_received_rows(
        received_rows,
        machines=machines,
        session_id=session_id,
        generation=generation,
        model_version=model_version,
    )

    results: list[dict[str, Any]] = []
    cleanup_at = utc_now_iso()
    for row, machine in selected:
        remote_result_dir, remote_task_dir = _remote_dirs(
            row,
            machine,
            remote_output_name=remote_output_name,
        )
        local_result_dir = Path(row.get("result_dir", ""))
        if not local_result_dir.is_absolute():
            local_result_dir = paths.output_dir / local_result_dir

        remote_delete_dirs: list[str] = []
        if bool(cfg.cleanup.remove_remote_results):
            remote_delete_dirs.append(remote_result_dir)
        if bool(cfg.cleanup.remove_remote_tasks):
            remote_delete_dirs.append(remote_task_dir)

        status_parts: list[str] = []
        error = ""
        status = "ok"
        try:
            if bool(cfg.cleanup.remove_local_received):
                status_parts.append(
                    _delete_local_received(
                        local_result_dir,
                        paths.received_dir,
                        dry_run=bool(dry_run),
                    )
                )
            if bool(cfg.cleanup.remove_remote_results) or bool(cfg.cleanup.remove_remote_tasks):
                status_parts.append(
                    _delete_remote_dirs(machine, remote_delete_dirs, dry_run=bool(dry_run))
                )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            if bool(cfg.cleanup.fail_on_remote_cleanup_error):
                raise

        result_row = {
            "session_id": row.get("session_id", ""),
            "task_id": row.get("task_id", ""),
            "machine_name": row.get("machine_name", ""),
            "machine_ip": row.get("machine_ip", ""),
            "cleanup_at_utc": cleanup_at,
            "generation": row.get("generation", ""),
            "model_version": row.get("model_version", ""),
            "local_result_dir": str(local_result_dir),
            "remote_result_dir": remote_result_dir,
            "remote_task_dir": remote_task_dir,
            "remove_local_received": bool(cfg.cleanup.remove_local_received),
            "remove_remote_results": bool(cfg.cleanup.remove_remote_results),
            "remove_remote_tasks": bool(cfg.cleanup.remove_remote_tasks),
            "dry_run": bool(dry_run),
            "status": status if not status_parts else f"{status}:{','.join(status_parts)}",
            "error": error,
        }
        append_csv_row(paths.cleanup_log_csv, CLEANUP_COLUMNS, result_row)
        results.append(result_row)

    return {
        "ok": True,
        "cleanup_enabled": True,
        "dry_run": bool(dry_run),
        "selected": len(results),
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    defaults = DEFAULT_NETWORK_CONFIG.paths
    parser = argparse.ArgumentParser(
        description="Delete consumed GV3 network samples after local training/evaluation is done."
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--model-version", type=int, default=None)
    parser.add_argument("--machine", action="append", default=None, help="Machine name/SSH host/IP to clean")
    parser.add_argument("--machines-config", default=str(defaults.machines_json))
    parser.add_argument(
        "--output-name",
        default=None,
        help="Simulator output namespace. Defaults to Game_Version3; use Game_Version3_Native for isolated native runs.",
    )
    parser.add_argument(
        "--remote-output-name",
        default=None,
        help="Remote simulator output namespace. Defaults to --output-name, or Game_Version3.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Preview cleanup without deleting files")
    parser.add_argument(
        "--confirm-delete",
        action="store_true",
        help="Actually delete selected local and remote folders. Without this flag the command is a dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dry_run = bool(args.dry_run) or not bool(args.confirm_delete)
    result = cleanup_consumed_generation_samples(
        session_id=args.session_id,
        generation=args.generation,
        model_version=args.model_version,
        machine_names=args.machine,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        machines_config=Path(args.machines_config),
        output_name=args.output_name,
        remote_output_name=args.remote_output_name,
        dry_run=dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
