"""Deploy and supervise one manifest-pinned GV4 AlphaGoZero experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Mapping, Sequence

from ..config import ExperimentConfig, load_experiment_config
from ..model_bundle import load_model_bundle
from .cluster import (
    ClusterSpec,
    HostSpec,
    copy_to_host,
    load_cluster,
    process_is_running,
    run_command,
    run_shell,
    start_detached,
    stop_detached,
    sync_tree_to_host,
)
from .durable_transfer import sync_current_model


DEPLOYMENT_SCHEMA_VERSION = "gv4_deployment_v1"
_VALIDATION_PREFIX = "GV4_DEPLOY_VALIDATION="
SOURCE_EXCLUDES = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "build_aws",
    "cache",
    "network",
    "simulator_output",
)

__all__ = [
    "DEPLOYMENT_SCHEMA_VERSION",
    "ProcessStatus",
    "build_native",
    "cluster_status",
    "coordinator_command",
    "deploy_all",
    "deployment_environment",
    "stage_current_model",
    "stage_manifests",
    "start_services",
    "stop_services",
    "sync_source",
    "validate_hosts",
    "worker_command",
    "write_deployment_record",
]


@dataclass(frozen=True, slots=True)
class ProcessStatus:
    host_id: str
    role: str
    running: bool
    pid_path: str
    log_path: str


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _unique_hosts(hosts: Sequence[HostSpec]) -> tuple[HostSpec, ...]:
    result: list[HostSpec] = []
    seen: set[tuple[str, str]] = set()
    for host in hosts:
        identity = (host.address, host.repo_root)
        if identity not in seen:
            result.append(host)
            seen.add(identity)
    return tuple(result)


def _remote_manifest_paths(host: HostSpec) -> tuple[str, str]:
    root = host.experiment_root.rstrip("/")
    return f"{root}/experiment_config.json", f"{root}/cluster.json"


def _process_paths(host: HostSpec, process_name: str) -> tuple[str, str]:
    root = host.experiment_root.rstrip("/")
    return (
        f"{root}/processes/{process_name}.pid",
        f"{root}/logs/{process_name}.log",
    )


def deployment_environment(
    experiment: ExperimentConfig,
    host: HostSpec,
) -> dict[str, str]:
    """Return the complete environment shared by one launched process."""

    environment = dict(experiment.deployment.environment)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = host.repo_root
    return environment


def coordinator_command(host: HostSpec) -> list[str]:
    config_path, cluster_path = _remote_manifest_paths(host)
    return [
        host.python_executable,
        f"{host.package_root}/AlphaGoZeroGV4/process_entrypoint.py",
        "AlphaGoZeroGV4.distributed_operations.xl_coordinator",
        "--experiment-config",
        config_path,
        "--cluster",
        cluster_path,
        "--root",
        host.experiment_root,
    ]


def worker_command(host: HostSpec) -> list[str]:
    config_path, cluster_path = _remote_manifest_paths(host)
    return [
        host.python_executable,
        f"{host.package_root}/AlphaGoZeroGV4/process_entrypoint.py",
        "AlphaGoZeroGV4.distributed_operations.worker_daemon",
        "--experiment-config",
        config_path,
        "--cluster",
        cluster_path,
        "--worker-id",
        host.host_id,
        "--root",
        host.experiment_root,
    ]


def write_deployment_record(
    destination: str | Path,
    *,
    experiment_config_path: str | Path,
    cluster_path: str | Path,
    source_root: str | Path,
    current_model_path: str | Path | None = None,
) -> Path:
    """Record the exact inputs staged by one deployment invocation."""

    source = Path(source_root).expanduser().resolve()
    experiment = load_experiment_config(experiment_config_path)
    cluster = load_cluster(cluster_path)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    model: dict[str, Any] | None = None
    if current_model_path is not None:
        bundle = load_model_bundle(
            current_model_path,
            config=experiment.resolve_engine_config(),
            device="cpu",
        )
        model = {
            "bundle_version": bundle.bundle_version,
            "manifest_sha256": bundle.manifest_sha256,
            "model_versions": bundle.model_versions,
        }
    record = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment.experiment_id,
        "experiment_manifest_sha256": experiment.manifest_sha256(),
        "experiment_config_file_sha256": _sha256(experiment_config_path),
        "cluster_manifest_sha256": _sha256(cluster_path),
        "cluster_name": cluster.name,
        "source_root": str(source),
        "git_commit": (commit.stdout or "").strip() if commit.returncode == 0 else "",
        "git_dirty": status.returncode != 0 or bool((status.stdout or "").strip()),
        "current_model": model,
    }
    path = Path(destination).expanduser().resolve()
    _atomic_json(path, record)
    return path


def sync_source(
    source_root: str | Path,
    cluster: ClusterSpec,
    *,
    exclude_names: Sequence[str] = SOURCE_EXCLUDES,
) -> None:
    """Merge code onto each distinct host without deleting host-local data."""

    for host in _unique_hosts(cluster.all_hosts):
        sync_tree_to_host(
            source_root,
            host,
            host.repo_root,
            exclude_names=exclude_names,
        )


def stage_manifests(
    experiment_config_path: str | Path,
    cluster_path: str | Path,
    cluster: ClusterSpec,
    *,
    deployment_record: str | Path | None = None,
) -> None:
    """Copy immutable manifests to conventional paths on every host."""

    for host in cluster.all_hosts:
        config_target, cluster_target = _remote_manifest_paths(host)
        run_shell(host, f"mkdir -p {shlex.quote(host.experiment_root)}")
        copy_to_host(experiment_config_path, host, config_target)
        copy_to_host(cluster_path, host, cluster_target)
        if deployment_record is not None:
            copy_to_host(
                deployment_record,
                host,
                f"{host.experiment_root.rstrip('/')}/deployment_manifest.json",
            )


def stage_current_model(
    current_model_path: str | Path,
    experiment: ExperimentConfig,
    cluster: ClusterSpec,
) -> dict[str, str]:
    """Validate and atomically expose one model bundle on every host."""

    engine = experiment.resolve_engine_config()
    return {
        host.host_id: sync_current_model(
            current_model_path,
            host,
            config=engine,
        )
        for host in cluster.all_hosts
    }


def _native_required(
    experiment: ExperimentConfig,
    cluster: ClusterSpec,
    host: HostSpec,
) -> bool:
    if host.role == "worker" and experiment.self_play.backend == "native":
        return True
    arena_host_ids = {item.host_id for item in cluster.arena_hosts}
    if (
        experiment.distributed_evaluation.enabled
        and host.host_id in arena_host_ids
        and experiment.arena.backend == "native"
    ):
        return True
    if (
        host.role == "coordinator"
        and not experiment.distributed_evaluation.enabled
        and experiment.arena.backend == "native"
    ):
        return True
    return bool(
        host.role == "coordinator"
        and experiment.sjf is not None
        and experiment.sjf.backend == "native"
    )


def build_native(experiment: ExperimentConfig, cluster: ClusterSpec) -> None:
    """Build the parity-tested GV4 extension only on hosts that need it."""

    required = _unique_hosts(
        tuple(
            host
            for host in cluster.all_hosts
            if _native_required(experiment, cluster, host)
        )
    )
    for host in required:
        source = f"{host.package_root}/GV4_Cpp"
        build = f"{source}/build"
        run_command(
            host,
            [
                "cmake",
                "-S",
                source,
                "-B",
                build,
                f"-DPython_EXECUTABLE={host.python_executable}",
            ],
            cwd=host.repo_root,
        )
        run_command(
            host,
            [
                "cmake",
                "--build",
                build,
                "-j",
                str(experiment.deployment.native_build_jobs),
            ],
            cwd=host.repo_root,
        )


def _validation_command(host: HostSpec) -> list[str]:
    config_path, cluster_path = _remote_manifest_paths(host)
    return [
        host.python_executable,
        f"{host.package_root}/AlphaGoZeroGV4/process_entrypoint.py",
        "AlphaGoZeroGV4.distributed_operations.deploy",
        "--experiment-config",
        config_path,
        "--cluster",
        cluster_path,
        "validate-host",
        "--host-id",
        host.host_id,
    ]


def _validate_local_host(
    experiment: ExperimentConfig,
    cluster: ClusterSpec,
    host_id: str,
) -> dict[str, Any]:
    host = next(
        (candidate for candidate in cluster.all_hosts if candidate.host_id == host_id),
        None,
    )
    if host is None:
        raise KeyError(f"unknown host {host_id!r}")
    engine = experiment.resolve_engine_config()
    pointer = Path(host.experiment_root) / "models" / "current_model.json"
    bundle = load_model_bundle(pointer, config=engine, device="cpu")
    if _native_required(experiment, cluster, host):
        from GV4_Cpp import gv4_native

        if gv4_native is None:
            raise RuntimeError("GV4 native module did not load")
    return {
        "host_id": host.host_id,
        "config_manifest_sha256": engine.manifest_sha256(),
        "bundle_version": bundle.bundle_version,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "native_required": _native_required(experiment, cluster, host),
    }


def validate_hosts(
    experiment: ExperimentConfig,
    cluster: ClusterSpec,
) -> dict[str, dict[str, Any]]:
    """Fail before launch if config, models, imports, or native builds differ."""

    results: dict[str, dict[str, Any]] = {}
    for host in cluster.all_hosts:
        completed = run_command(
            host,
            _validation_command(host),
            cwd=host.repo_root,
            environment=deployment_environment(experiment, host),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"host {host.host_id} validation failed:\n{completed.stdout or ''}"
            )
        payload = next(
            (
                line.removeprefix(_VALIDATION_PREFIX)
                for line in reversed((completed.stdout or "").splitlines())
                if line.startswith(_VALIDATION_PREFIX)
            ),
            "",
        )
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"host {host.host_id} returned invalid validation JSON")
        results[host.host_id] = value
    return results


def start_services(
    experiment: ExperimentConfig,
    cluster: ClusterSpec,
) -> dict[str, int]:
    """Start one guarded coordinator and one guarded daemon per worker."""

    started: dict[str, int] = {}
    coordinator = cluster.coordinator
    coordinator_pid, coordinator_log = _process_paths(coordinator, "coordinator")
    if not process_is_running(coordinator, coordinator_pid):
        started[coordinator.host_id] = start_detached(
            coordinator,
            coordinator_command(coordinator),
            cwd=coordinator.repo_root,
            environment=deployment_environment(experiment, coordinator),
            pid_path=coordinator_pid,
            log_path=coordinator_log,
        )
    for worker in cluster.workers:
        name = f"worker_{worker.host_id}"
        pid_path, log_path = _process_paths(worker, name)
        if process_is_running(worker, pid_path):
            continue
        started[worker.host_id] = start_detached(
            worker,
            worker_command(worker),
            cwd=worker.repo_root,
            environment=deployment_environment(experiment, worker),
            pid_path=pid_path,
            log_path=log_path,
        )
    return started


def stop_services(cluster: ClusterSpec) -> dict[str, bool]:
    """Stop only PIDs owned by this experiment, workers before coordinator."""

    stopped: dict[str, bool] = {}
    for worker in cluster.workers:
        pid_path, _ = _process_paths(worker, f"worker_{worker.host_id}")
        stopped[worker.host_id] = stop_detached(worker, pid_path)
    coordinator = cluster.coordinator
    pid_path, _ = _process_paths(coordinator, "coordinator")
    stopped[coordinator.host_id] = stop_detached(coordinator, pid_path)
    return stopped


def cluster_status(cluster: ClusterSpec) -> tuple[ProcessStatus, ...]:
    statuses: list[ProcessStatus] = []
    coordinator = cluster.coordinator
    pid_path, log_path = _process_paths(coordinator, "coordinator")
    statuses.append(
        ProcessStatus(
            coordinator.host_id,
            coordinator.role,
            process_is_running(coordinator, pid_path),
            pid_path,
            log_path,
        )
    )
    for worker in cluster.workers:
        pid_path, log_path = _process_paths(worker, f"worker_{worker.host_id}")
        statuses.append(
            ProcessStatus(
                worker.host_id,
                worker.role,
                process_is_running(worker, pid_path),
                pid_path,
                log_path,
            )
        )
    return tuple(statuses)


def deploy_all(
    *,
    experiment_config_path: str | Path,
    cluster_path: str | Path,
    source_root: str | Path,
    current_model_path: str | Path,
    deployment_record: str | Path,
) -> dict[str, Any]:
    """Synchronize, validate, and start a complete experiment."""

    experiment = load_experiment_config(experiment_config_path)
    cluster = load_cluster(cluster_path)
    sync_source(source_root, cluster)
    record = write_deployment_record(
        deployment_record,
        experiment_config_path=experiment_config_path,
        cluster_path=cluster_path,
        source_root=source_root,
        current_model_path=current_model_path,
    )
    stage_manifests(
        experiment_config_path,
        cluster_path,
        cluster,
        deployment_record=record,
    )
    models = stage_current_model(current_model_path, experiment, cluster)
    build_native(experiment, cluster)
    validation = validate_hosts(experiment, cluster)
    started = start_services(experiment, cluster)
    return {
        "deployment_record": str(record),
        "models": models,
        "validation": validation,
        "started": started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--cluster", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("sync-source")
    source.add_argument("--source-root", type=Path, required=True)

    stage = commands.add_parser("stage")
    stage.add_argument("--source-root", type=Path, required=True)
    stage.add_argument("--current-model", type=Path)
    stage.add_argument("--deployment-record", type=Path)

    commands.add_parser("build-native")
    commands.add_parser("validate")
    commands.add_parser("start")
    commands.add_parser("stop")
    commands.add_parser("status")

    local = commands.add_parser("validate-host")
    local.add_argument("--host-id", required=True)

    complete = commands.add_parser("all")
    complete.add_argument("--source-root", type=Path, required=True)
    complete.add_argument("--current-model", type=Path, required=True)
    complete.add_argument("--deployment-record", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    experiment = load_experiment_config(args.experiment_config)
    cluster = load_cluster(args.cluster)

    if args.command == "validate-host":
        print(
            _VALIDATION_PREFIX
            + json.dumps(
                _validate_local_host(experiment, cluster, args.host_id),
                sort_keys=True,
            )
        )
        return
    if args.command == "sync-source":
        sync_source(args.source_root, cluster)
        return
    if args.command == "stage":
        record = args.deployment_record or (
            args.experiment_config.parent / "deployment_manifest.json"
        )
        write_deployment_record(
            record,
            experiment_config_path=args.experiment_config,
            cluster_path=args.cluster,
            source_root=args.source_root,
            current_model_path=args.current_model,
        )
        stage_manifests(
            args.experiment_config,
            args.cluster,
            cluster,
            deployment_record=record,
        )
        if args.current_model is not None:
            stage_current_model(args.current_model, experiment, cluster)
        return
    if args.command == "build-native":
        build_native(experiment, cluster)
        return
    if args.command == "validate":
        print(json.dumps(validate_hosts(experiment, cluster), indent=2, sort_keys=True))
        return
    if args.command == "start":
        validate_hosts(experiment, cluster)
        print(json.dumps(start_services(experiment, cluster), indent=2, sort_keys=True))
        return
    if args.command == "stop":
        print(json.dumps(stop_services(cluster), indent=2, sort_keys=True))
        return
    if args.command == "status":
        print(
            json.dumps(
                [asdict(item) for item in cluster_status(cluster)],
                indent=2,
                sort_keys=True,
            )
        )
        return
    record = args.deployment_record or (
        args.experiment_config.parent / "deployment_manifest.json"
    )
    print(
        json.dumps(
            deploy_all(
                experiment_config_path=args.experiment_config,
                cluster_path=args.cluster,
                source_root=args.source_root,
                current_model_path=args.current_model,
                deployment_record=record,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
