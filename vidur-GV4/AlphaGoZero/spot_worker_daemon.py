"""Spot worker that pulls self-play or evaluation work from the XL coordinator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import psutil

from vidur.AlphaGoZero.config import (
    AGZ_SPOT_CONTROLLER_RETRY_ATTEMPTS,
    AGZ_SPOT_CONTROLLER_RETRY_INITIAL_SEC,
    AGZ_SPOT_CONTROLLER_RETRY_MAX_SEC,
    AGZ_SPOT_PUBLISH_RETRY_ATTEMPTS,
    AGZ_SPOT_WORKER_ERROR_BACKOFF_SEC,
    REPO_ROOT,
)
from vidur.AlphaGoZero.durable_transfer import (
    atomic_write_json,
    remote_verify_and_publish,
    rsync_dir_to,
    sha256_file,
    utc_now,
    verify_sha256sums,
    write_sha256sums,
)
from vidur.AlphaGoZero.spot_work_protocol import (
    SELFPLAY_CONFIG_FIELDS,
    selfplay_config_sha256,
    validate_selfplay_config,
)
from vidur.AlphaGoZero.worker_daemon import build_parser as build_selfplay_parser
from vidur.AlphaGoZero.worker_daemon import run_worker


GIB = 1024**3
_EVENT_LOG_LOCK = threading.Lock()


def _record_worker_event(
    args: argparse.Namespace,
    event: str,
    **fields: Any,
) -> None:
    path = Path(args.output_root) / "spot_worker_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time_utc": utc_now(), "event": str(event), **fields}
    try:
        with _EVENT_LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        # Logging must never turn a recoverable transport error into a crash.
        pass


def _retry_delay(args: argparse.Namespace, attempt: int) -> float:
    initial = max(
        0.0,
        float(
            getattr(
                args,
                "controller_retry_initial_sec",
                AGZ_SPOT_CONTROLLER_RETRY_INITIAL_SEC,
            )
        ),
    )
    maximum = max(
        initial,
        float(
            getattr(
                args,
                "controller_retry_max_sec",
                AGZ_SPOT_CONTROLLER_RETRY_MAX_SEC,
            )
        ),
    )
    return min(maximum, initial * (2 ** max(0, int(attempt) - 1)))


def _apply_controller_selfplay_config(
    args: argparse.Namespace,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    raw = assignment.get("selfplay_config")
    fingerprint = str(assignment.get("selfplay_config_sha256", ""))
    if not isinstance(raw, dict) or not fingerprint:
        raise ValueError("self-play assignment is missing coordinator configuration")
    config = validate_selfplay_config(dict(raw))
    calculated = selfplay_config_sha256(config)
    if fingerprint != calculated:
        raise ValueError(
            "self-play assignment configuration fingerprint mismatch: "
            f"expected={fingerprint}, calculated={calculated}"
        )
    for field in SELFPLAY_CONFIG_FIELDS:
        setattr(args, field, config[field])
    args.assignment_id = str(assignment.get("assignment_id", ""))
    args.assignment_config_sha256 = calculated
    parent_count = int(config["parent_state_count"])
    parent_dir = Path(str(config["parent_dataset_dir"]))
    if parent_count > 0 and not parent_dir.is_dir():
        raise FileNotFoundError(
            f"coordinator-assigned parent dataset is unavailable: {parent_dir}"
        )
    return config


def _verify_eval_command(assignment: dict[str, Any]) -> None:
    command = assignment.get("command")
    fingerprint = str(assignment.get("command_sha256", ""))
    if not isinstance(command, list) or not fingerprint:
        raise ValueError("evaluation assignment is missing its command fingerprint")
    payload = json.dumps(
        command, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != fingerprint:
        raise ValueError("evaluation assignment command fingerprint mismatch")


def _read_cgroup_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _cgroup_memory_headroom_bytes() -> int | None:
    candidates = (
        (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.current"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, usage_path in candidates:
        limit = _read_cgroup_int(limit_path)
        usage = _read_cgroup_int(usage_path)
        if limit is None or usage is None:
            continue
        # Some cgroup-v1 hosts expose an effectively unlimited sentinel.
        if limit >= 1 << 60:
            continue
        return max(0, limit - usage)
    return None


def _cgroup_cpu_quota_count() -> int | None:
    try:
        fields = Path("/sys/fs/cgroup/cpu.max").read_text(
            encoding="utf-8"
        ).split()
    except OSError:
        fields = []
    if len(fields) == 2 and fields[0] != "max":
        try:
            quota, period = int(fields[0]), int(fields[1])
            if quota > 0 and period > 0:
                return max(1, quota // period)
        except ValueError:
            pass

    quota = _read_cgroup_int(
        Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    )
    period = _read_cgroup_int(
        Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    )
    if quota is not None and period and quota > 0:
        return max(1, quota // period)
    return None


def _automatic_reserved_cpus(cpu_count: int) -> int:
    count = max(1, int(cpu_count))
    if count == 1:
        return 0
    return min(count - 1, max(2, int(math.ceil(count / 16.0))))


def detect_worker_resources(args: argparse.Namespace) -> dict[str, int]:
    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_cpus = int(os.cpu_count() or 1)
    quota_cpus = _cgroup_cpu_quota_count()
    cpu_count = max(
        1,
        min(
            int(affinity_cpus),
            int(quota_cpus) if quota_cpus is not None else int(affinity_cpus),
        ),
    )

    configured_reserve = int(getattr(args, "resource_reserve_cpus", -1))
    reserved_cpus = (
        _automatic_reserved_cpus(cpu_count)
        if configured_reserve < 0
        else min(cpu_count - 1, max(0, configured_reserve))
    )
    cpu_threads_per_game = max(
        1,
        int(getattr(args, "worker_threads", 1)),
        int(getattr(args, "rollout_parallel_threads", 1)),
    )
    cpu_slots = max(0, cpu_count - reserved_cpus) // cpu_threads_per_game

    host_available_memory = max(0, int(psutil.virtual_memory().available))
    cgroup_headroom = _cgroup_memory_headroom_bytes()
    available_memory = (
        min(host_available_memory, int(cgroup_headroom))
        if cgroup_headroom is not None
        else host_available_memory
    )
    memory_gib_per_game = max(
        0.001, float(getattr(args, "resource_memory_gib_per_game", 1.0))
    )
    memory_bytes_per_game = max(1, int(memory_gib_per_game * GIB))
    memory_slots = available_memory // memory_bytes_per_game

    configured_cap = int(getattr(args, "parallel_games", 0))
    advertised_capacity = min(max(0, cpu_count - reserved_cpus), memory_slots)
    effective_capacity = min(cpu_slots, memory_slots)
    if configured_cap > 0:
        effective_capacity = min(effective_capacity, configured_cap)
    return {
        "cpu_count": int(cpu_count),
        "reserved_cpu_count": int(reserved_cpus),
        "cpu_threads_per_game": int(cpu_threads_per_game),
        "cpu_slots": int(cpu_slots),
        "available_memory_bytes": int(available_memory),
        "memory_bytes_per_game": int(memory_bytes_per_game),
        "memory_slots": int(memory_slots),
        "advertised_capacity": max(0, int(advertised_capacity)),
        "effective_capacity": max(0, int(effective_capacity)),
    }


def _controller_call(args: argparse.Namespace, command: list[str]) -> dict[str, Any]:
    remote = [
        str(args.controller_python),
        "-m",
        "vidur.AlphaGoZero.spot_work_cli",
        "--root",
        str(args.xl_output_root),
        *command,
    ]
    shell_command = f"cd {shlex.quote(str(args.controller_repo))} && {shlex.join(remote)}"
    attempts = max(
        1,
        int(
            getattr(
                args,
                "controller_retry_attempts",
                AGZ_SPOT_CONTROLLER_RETRY_ATTEMPTS,
            )
        ),
    )
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                ["ssh", str(args.xl_host), shell_command],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(30, int(args.controller_timeout_sec)),
            )
            lines = [
                line for line in completed.stdout.splitlines() if line.strip()
            ]
            if not lines:
                raise RuntimeError("controller returned no JSON")
            result = json.loads(lines[-1])
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"controller returned a non-object response: {result!r}"
                )
            if attempt > 1:
                _record_worker_event(
                    args,
                    "controller_call_recovered",
                    command=str(command[0] if command else ""),
                    attempt=attempt,
                )
            return result
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            _record_worker_event(
                args,
                "controller_call_retry",
                command=str(command[0] if command else ""),
                attempt=attempt,
                max_attempts=attempts,
                error=repr(exc),
            )
            if attempt < attempts:
                time.sleep(_retry_delay(args, attempt))
    assert last_error is not None
    raise RuntimeError(
        f"controller call failed after {attempts} attempts: {command!r}"
    ) from last_error


def _artifact_is_current(artifact: dict[str, Any]) -> bool:
    directory = Path(str(artifact["source_dir"]))
    for row in artifact.get("files", []):
        path = directory / str(row["relative_path"])
        if not path.is_file() or int(path.stat().st_size) != int(row["bytes"]):
            return False
        if sha256_file(path) != str(row["sha256"]):
            return False
    return True


def _stage_artifacts(args: argparse.Namespace, artifacts: list[dict[str, Any]]) -> None:
    unique = {str(artifact["source_dir"]): artifact for artifact in artifacts}
    for source_dir, artifact in sorted(unique.items()):
        if _artifact_is_current(artifact):
            continue
        destination = Path(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "--partial",
                "--delay-updates",
                "--timeout=120",
                f"{args.xl_host}:{source_dir.rstrip('/')}/",
                str(destination).rstrip("/") + "/",
            ],
            check=True,
        )
        if not _artifact_is_current(artifact):
            raise RuntimeError(f"model artifact checksum mismatch after staging: {source_dir}")


def _install_selfplay_bundle(args: argparse.Namespace, assignment: dict[str, Any]) -> None:
    _stage_artifacts(args, list(assignment["model_artifacts"]))
    current = Path(args.output_root) / "models" / "current_model.json"
    atomic_write_json(current, dict(assignment["model_bundle"]))


def _selfplay_heartbeat_loop(
    args: argparse.Namespace,
    assignment: dict[str, Any],
    stop: threading.Event,
    preempt_requested: threading.Event,
) -> None:
    while not stop.wait(max(5.0, float(args.heartbeat_sec))):
        try:
            response = _controller_call(
                args,
                [
                    "selfplay-heartbeat",
                    "--worker-id",
                    str(args.worker_id),
                    "--assignment-id",
                    str(assignment["assignment_id"]),
                    "--lease-token",
                    str(assignment["lease_token"]),
                    "--extend-sec",
                    str(int(assignment.get("lease_sec", args.assignment_lease_sec))),
                ],
            )
            if response.get("extended") is not True:
                raise RuntimeError(f"self-play lease heartbeat rejected: {response}")
            if response.get("preempt_requested") is True:
                atomic_write_json(
                    Path(args.output_root) / "last_selfplay_preemption.json",
                    {
                        "assignment_id": str(assignment["assignment_id"]),
                        "eval_batch_id": str(response.get("eval_batch_id", "")),
                        "reason": str(response.get("preempt_reason", "")),
                        "requested_at_utc": utc_now(),
                    },
                )
                preempt_requested.set()
                return
        except Exception as exc:  # noqa: BLE001 - game processes keep running.
            atomic_write_json(
                Path(args.output_root) / "last_selfplay_heartbeat_error.json",
                {"error": repr(exc), "time_utc": utc_now()},
            )


def _run_selfplay_wave(args: argparse.Namespace, assignment: dict[str, Any]) -> None:
    stop = threading.Event()
    preempt_requested = threading.Event()
    heartbeat: threading.Thread | None = None
    if assignment.get("lease_token"):
        heartbeat = threading.Thread(
            target=_selfplay_heartbeat_loop,
            args=(args, assignment, stop, preempt_requested),
            daemon=True,
        )
        heartbeat.start()

    status = "failed"
    release_error: BaseException | None = None
    try:
        _install_selfplay_bundle(args, assignment)
        wave_args = argparse.Namespace(**vars(args))
        resolved_config = _apply_controller_selfplay_config(wave_args, assignment)
        wave_args.parallel_games = max(1, int(assignment["parallel_games"]))
        wave_args.max_games = max(1, int(assignment["games"]))
        wave_args.upload_after_game = True
        wave_args.flush_at_end = False
        atomic_write_json(
            Path(args.output_root) / "last_applied_selfplay_config.json",
            {
                "assignment_id": str(assignment["assignment_id"]),
                "selfplay_config_sha256": str(assignment["selfplay_config_sha256"]),
                "selfplay_config": resolved_config,
                "applied_at_utc": utc_now(),
            },
        )
        was_preempted = run_worker(wave_args, preempt_event=preempt_requested)
        status = "preempted" if was_preempted else "completed"
    finally:
        stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout=10.0)
        if assignment.get("lease_token"):
            try:
                response = _controller_call(
                    args,
                    [
                        "selfplay-release",
                        "--worker-id",
                        str(args.worker_id),
                        "--assignment-id",
                        str(assignment["assignment_id"]),
                        "--lease-token",
                        str(assignment["lease_token"]),
                        "--status",
                        status,
                    ],
                )
                if response.get("released") is not True:
                    raise RuntimeError(f"self-play lease release rejected: {response}")
            except BaseException as exc:  # noqa: BLE001 - preserve the wave result.
                release_error = exc
                atomic_write_json(
                    Path(args.output_root) / "last_selfplay_release_error.json",
                    {"error": repr(exc), "time_utc": utc_now()},
                )
        if status in {"completed", "preempted"} and release_error is not None:
            raise release_error


def _assignment_work_dir(args: argparse.Namespace, task_id: str) -> Path:
    return Path(args.output_root) / "spot_eval_runs" / str(task_id)


def _write_single_game_metadata(
    work_dir: Path,
    assignment: dict[str, Any],
    *,
    elapsed_sec: float,
) -> None:
    arena_results = work_dir / "arena_results.csv"
    with arena_results.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(
            f"Spot task {assignment['task_id']} produced {len(rows)} arena rows"
        )
    game_id = int(float(rows[0]["game_id"]))
    if game_id != int(assignment["game_id"]):
        raise RuntimeError(
            f"Spot task {assignment['task_id']} game_id={game_id}, "
            f"expected={assignment['game_id']}"
        )
    history_hops = int(float(rows[0].get("history_hops", 0) or 0))
    planned_fields = ["game_index", "game_id", "history_hops", "job_output_dir"]
    planned = {
        "game_index": 0,
        "game_id": game_id,
        "history_hops": history_hops,
        "job_output_dir": str(work_dir),
    }
    with (work_dir / "planned_games.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=planned_fields)
        writer.writeheader()
        writer.writerow(planned)

    status_fields = [
        "timestamp",
        "game_index",
        "game_id",
        "history_hops",
        "status",
        "pid",
        "returncode",
        "elapsed_sec",
        "job_output_dir",
        "log_file",
        "arena_results_csv",
        "model_artifacts_cleaned",
    ]
    status = {
        **planned,
        "timestamp": utc_now(),
        "status": "ok",
        "pid": os.getpid(),
        "returncode": 0,
        "elapsed_sec": f"{float(elapsed_sec):.6f}",
        "log_file": str(work_dir / "spot_eval.log"),
        "arena_results_csv": str(arena_results),
        "model_artifacts_cleaned": "spot_staged",
    }
    with (work_dir / "job_status.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=status_fields)
        writer.writeheader()
        writer.writerow(status)


def _run_eval_assignment(args: argparse.Namespace, assignment: dict[str, Any]) -> Path:
    _verify_eval_command(assignment)
    task_id = str(assignment["task_id"])
    work_dir = _assignment_work_dir(args, task_id)
    if (
        (work_dir / "spot_assignment.json").is_file()
        and (work_dir / "spot_result_manifest.json").is_file()
    ):
        valid, _errors = verify_sha256sums(work_dir)
        if valid:
            return work_dir
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(work_dir) if item == "{WORK_DIR}" else str(item)
        for item in assignment["command"]
    ]
    if len(command) >= 3 and Path(command[0]).name.startswith("python"):
        command[0] = sys.executable
    atomic_write_json(
        work_dir / "spot_assignment.json",
        {
            "assignment": assignment,
            "resolved_command": command,
            "worker_id": str(args.worker_id),
            "started_at_utc": utc_now(),
        },
    )
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started_at = time.monotonic()
    with (work_dir / "spot_eval.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    _write_single_game_metadata(
        work_dir,
        assignment,
        elapsed_sec=time.monotonic() - started_at,
    )
    missing = [
        filename for filename in assignment.get("expected_files", [])
        if not (work_dir / str(filename)).is_file()
    ]
    if missing:
        raise RuntimeError(f"evaluation task {task_id} missing outputs: {missing}")
    atomic_write_json(
        work_dir / "spot_result_manifest.json",
        {
            "schema_version": 1,
            "task_id": task_id,
            "batch_id": str(assignment["batch_id"]),
            "block_name": str(assignment["block_name"]),
            "game_id": int(assignment["game_id"]),
            "worker_id": str(args.worker_id),
            "lease_token": str(assignment["lease_token"]),
            "completed_at_utc": utc_now(),
        },
    )
    write_sha256sums(work_dir, include_manifest=True)
    return work_dir


def _publish_eval_result(
    args: argparse.Namespace,
    assignment: dict[str, Any],
    work_dir: Path,
) -> None:
    task_id = str(assignment["task_id"])
    base = str(args.xl_output_root).rstrip("/") + "/spot_work"
    uploading = f"{base}/result_uploading/{args.worker_id}/{task_id}"
    result = f"{base}/results/{task_id}"
    rejected = f"{base}/result_rejected/{args.worker_id}/{task_id}"
    attempts = max(
        1,
        int(
            getattr(
                args,
                "publish_retry_attempts",
                AGZ_SPOT_PUBLISH_RETRY_ATTEMPTS,
            )
        ),
    )
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            rsync_dir_to(work_dir, str(args.xl_host), uploading)
            remote_verify_and_publish(
                host=str(args.xl_host),
                uploading_dir=uploading,
                incoming_dir=result,
                rejected_dir=rejected,
            )
            break
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            last_error = exc
            _record_worker_event(
                args,
                "eval_result_publish_retry",
                task_id=task_id,
                attempt=attempt,
                max_attempts=attempts,
                error=repr(exc),
            )
            if attempt < attempts:
                time.sleep(_retry_delay(args, attempt))
    else:
        assert last_error is not None
        raise RuntimeError(
            f"result publication failed after {attempts} attempts for {task_id}"
        ) from last_error
    response = _controller_call(
        args,
        [
            "complete",
            "--worker-id",
            str(args.worker_id),
            "--task-id",
            task_id,
            "--lease-token",
            str(assignment["lease_token"]),
        ],
    )
    if response.get("accepted") is not True:
        raise RuntimeError(f"controller rejected completed task {task_id}: {response}")


def _recover_pending_eval_results(args: argparse.Namespace) -> tuple[int, int]:
    root = Path(args.output_root) / "spot_eval_runs"
    if not root.is_dir():
        return 0, 0
    recovered = 0
    pending = 0
    for work_dir in sorted(root.iterdir()):
        if not work_dir.is_dir():
            continue
        assignment_path = work_dir / "spot_assignment.json"
        manifest_path = work_dir / "spot_result_manifest.json"
        if not assignment_path.is_file() or not manifest_path.is_file():
            continue
        valid, errors = verify_sha256sums(work_dir)
        if not valid:
            _record_worker_event(
                args,
                "pending_eval_result_invalid",
                task_id=work_dir.name,
                errors=errors,
            )
            continue
        try:
            assignment_record = json.loads(
                assignment_path.read_text(encoding="utf-8")
            )
            assignment = dict(assignment_record["assignment"])
            _publish_eval_result(args, assignment, work_dir)
        except Exception as exc:  # noqa: BLE001 - retain output for another retry.
            pending += 1
            _record_worker_event(
                args,
                "pending_eval_result_retry_deferred",
                task_id=work_dir.name,
                error=repr(exc),
            )
            continue
        recovered += 1
        _record_worker_event(
            args, "pending_eval_result_recovered", task_id=work_dir.name
        )
        if not bool(args.keep_game_runs):
            shutil.rmtree(work_dir, ignore_errors=True)
    return recovered, pending


def _report_eval_failure(
    args: argparse.Namespace,
    assignment: dict[str, Any],
    error: BaseException,
) -> None:
    _controller_call(
        args,
        [
            "fail",
            "--worker-id",
            str(args.worker_id),
            "--task-id",
            str(assignment["task_id"]),
            "--lease-token",
            str(assignment["lease_token"]),
            "--error",
            repr(error)[-8000:],
            "--max-attempts",
            str(int(args.eval_max_attempts)),
        ],
    )


def _heartbeat_loop(
    args: argparse.Namespace,
    assignments: list[dict[str, Any]],
    stop: threading.Event,
) -> None:
    leases = {
        str(assignment["task_id"]): str(assignment["lease_token"])
        for assignment in assignments
    }
    while not stop.wait(max(5.0, float(args.heartbeat_sec))):
        try:
            response = _controller_call(
                args,
                [
                    "heartbeat",
                    "--worker-id",
                    str(args.worker_id),
                    "--leases-json",
                    json.dumps(leases, sort_keys=True),
                    "--extend-sec",
                    str(int(args.assignment_lease_sec)),
                ],
            )
            rejected = list(response.get("rejected", []))
            if rejected:
                raise RuntimeError(f"controller rejected active leases: {rejected}")
        except Exception as exc:  # noqa: BLE001 - running games retain their leases.
            atomic_write_json(
                Path(args.output_root) / "last_heartbeat_error.json",
                {"error": repr(exc), "time_utc": utc_now()},
            )


def _run_eval_wave(args: argparse.Namespace, assignments: list[dict[str, Any]]) -> None:
    artifacts: list[dict[str, Any]] = []
    for assignment in assignments:
        artifacts.extend(list(assignment.get("model_artifacts", [])))
    _stage_artifacts(args, artifacts)

    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(args, assignments, stop),
        daemon=True,
    )
    heartbeat.start()
    try:
        with ThreadPoolExecutor(
            # XL already capped this lease wave by live CPU and RAM resources.
            max_workers=max(1, len(assignments)),
            thread_name_prefix="agz_spot_eval",
        ) as executor:
            futures = {
                executor.submit(_run_eval_assignment, args, assignment): assignment
                for assignment in assignments
            }
            for future in as_completed(futures):
                assignment = futures[future]
                try:
                    work_dir = future.result()
                except Exception as exc:  # noqa: BLE001 - each lease is independent.
                    _record_worker_event(
                        args,
                        "eval_game_failed",
                        task_id=str(assignment["task_id"]),
                        error=repr(exc),
                    )
                    try:
                        _report_eval_failure(args, assignment, exc)
                    except Exception as report_exc:  # noqa: BLE001
                        _record_worker_event(
                            args,
                            "eval_failure_report_deferred",
                            task_id=str(assignment["task_id"]),
                            error=repr(report_exc),
                        )
                    continue
                try:
                    _publish_eval_result(args, assignment, work_dir)
                except Exception as exc:  # noqa: BLE001 - do not discard success.
                    _record_worker_event(
                        args,
                        "eval_result_publish_deferred",
                        task_id=str(assignment["task_id"]),
                        work_dir=str(work_dir),
                        error=repr(exc),
                    )
                    continue
                if not bool(args.keep_game_runs):
                    shutil.rmtree(work_dir, ignore_errors=True)
    finally:
        stop.set()
        heartbeat.join(timeout=10.0)


def run_spot_worker(args: argparse.Namespace) -> None:
    waves = 0
    while int(args.max_waves) <= 0 or waves < int(args.max_waves):
        try:
            _recover_pending_eval_results(args)
        except Exception as exc:  # noqa: BLE001 - worker must stay available.
            _record_worker_event(args, "pending_result_scan_failed", error=repr(exc))
        resources = detect_worker_resources(args)
        try:
            assignment = _controller_call(
                args,
                [
                "request",
                "--worker-id",
                str(args.worker_id),
                "--capacity",
                str(int(resources["advertised_capacity"])),
                "--cpu-count",
                str(int(resources["cpu_count"])),
                "--reserved-cpu-count",
                str(int(resources["reserved_cpu_count"])),
                "--available-memory-bytes",
                str(int(resources["available_memory_bytes"])),
                "--cpu-threads-per-game",
                str(int(resources["cpu_threads_per_game"])),
                "--memory-bytes-per-game",
                str(int(resources["memory_bytes_per_game"])),
                ],
            )
        except Exception as exc:  # noqa: BLE001 - transient control-plane outage.
            _record_worker_event(args, "work_request_deferred", error=repr(exc))
            time.sleep(
                max(
                    0.1,
                    float(
                        getattr(
                            args,
                            "worker_error_backoff_sec",
                            AGZ_SPOT_WORKER_ERROR_BACKOFF_SEC,
                        )
                    ),
                )
            )
            continue
        mode = str(assignment.get("mode", "idle"))
        atomic_write_json(
            Path(args.output_root) / "last_spot_assignment.json",
            {"response": assignment, "received_at_utc": utc_now()},
        )
        if mode in {"eval", "selfplay"}:
            try:
                if mode == "eval":
                    _run_eval_wave(args, list(assignment["assignments"]))
                else:
                    _run_selfplay_wave(args, assignment)
            except Exception as exc:  # noqa: BLE001 - keep daemon alive.
                _record_worker_event(
                    args,
                    "work_wave_deferred",
                    mode=mode,
                    error=repr(exc),
                )
                time.sleep(
                    max(
                        0.1,
                        float(
                            getattr(
                                args,
                                "worker_error_backoff_sec",
                                AGZ_SPOT_WORKER_ERROR_BACKOFF_SEC,
                            )
                        ),
                    )
                )
            waves += 1
            continue
        if mode != "idle":
            raise RuntimeError(f"unsupported controller work mode: {mode!r}")
        time.sleep(max(float(args.poll_sec), float(assignment.get("retry_after_sec", 1.0))))


def build_parser() -> argparse.ArgumentParser:
    parser = build_selfplay_parser()
    parser.description = "Run a pull-based AlphaGoZero Spot worker."
    parser.add_argument(
        "--controller-repo",
        default="/home/ubuntu/vidur-classical-search",
    )
    parser.add_argument(
        "--controller-python",
        default="/home/ubuntu/vidur-classical-search/.venv/bin/python3",
    )
    parser.add_argument("--controller-timeout-sec", type=int, default=120)
    parser.add_argument(
        "--controller-retry-attempts",
        type=int,
        default=AGZ_SPOT_CONTROLLER_RETRY_ATTEMPTS,
    )
    parser.add_argument(
        "--controller-retry-initial-sec",
        type=float,
        default=AGZ_SPOT_CONTROLLER_RETRY_INITIAL_SEC,
    )
    parser.add_argument(
        "--controller-retry-max-sec",
        type=float,
        default=AGZ_SPOT_CONTROLLER_RETRY_MAX_SEC,
    )
    parser.add_argument(
        "--publish-retry-attempts",
        type=int,
        default=AGZ_SPOT_PUBLISH_RETRY_ATTEMPTS,
    )
    parser.add_argument(
        "--worker-error-backoff-sec",
        type=float,
        default=AGZ_SPOT_WORKER_ERROR_BACKOFF_SEC,
    )
    parser.add_argument("--assignment-lease-sec", type=int, default=3600)
    parser.add_argument("--heartbeat-sec", type=float, default=10.0)
    parser.add_argument("--eval-max-attempts", type=int, default=5)
    parser.add_argument("--max-waves", type=int, default=0, help="0 means run forever")
    parser.add_argument(
        "--resource-reserve-cpus",
        type=int,
        default=-1,
        help="-1 reserves max(2, ceil(cpus/16)); non-negative overrides it",
    )
    parser.add_argument("--resource-memory-gib-per-game", type=float, default=1.0)
    parser.set_defaults(
        parallel_games=0,
        upload_after_game=True,
        flush_at_end=False,
    )
    return parser


def main() -> None:
    run_spot_worker(build_parser().parse_args())


if __name__ == "__main__":
    main()
