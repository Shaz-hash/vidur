"""Cluster inventory plus small local/SSH process and transfer helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Mapping, Sequence


CLUSTER_SCHEMA_VERSION = "gv4_cluster_v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

__all__ = [
    "CLUSTER_SCHEMA_VERSION",
    "ClusterSpec",
    "HostSpec",
    "copy_from_host",
    "copy_to_host",
    "load_cluster",
    "process_is_running",
    "run_command",
    "run_shell",
    "start_detached",
    "stop_detached",
    "sync_tree_to_host",
    "write_cluster",
]


@dataclass(frozen=True, slots=True)
class HostSpec:
    """One machine and the paths used by this experiment on that machine."""

    host_id: str
    address: str
    ordinal: int
    role: str
    repo_root: str
    experiment_root: str
    python_executable: str = "python3"

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.host_id):
            raise ValueError("host_id is not safe for paths or process labels")
        if not self.address.strip():
            raise ValueError("host address cannot be empty")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("host ordinal must be a nonnegative integer")
        if self.role not in {"coordinator", "worker", "evaluation"}:
            raise ValueError("host role must be coordinator, worker, or evaluation")
        for name in ("repo_root", "experiment_root", "python_executable"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        for name in ("repo_root", "experiment_root"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be an absolute path")

    @property
    def is_local(self) -> bool:
        return self.address in {"local", "localhost", "127.0.0.1", "::1"}

    @property
    def package_root(self) -> str:
        return f"{self.repo_root.rstrip('/')}/vidur-GV4"


@dataclass(frozen=True, slots=True)
class ClusterSpec:
    """Exactly one coordinator and a nonempty set of unique workers."""

    name: str
    coordinator: HostSpec
    workers: tuple[HostSpec, ...]
    evaluation_hosts: tuple[HostSpec, ...] = ()

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.name):
            raise ValueError("cluster name is invalid")
        if self.coordinator.role != "coordinator":
            raise ValueError("cluster coordinator has the wrong role")
        if not self.workers:
            raise ValueError("cluster must contain at least one worker")
        if any(host.role != "worker" for host in self.workers):
            raise ValueError("all workers must have role='worker'")
        if any(host.role != "evaluation" for host in self.evaluation_hosts):
            raise ValueError("all evaluation hosts must have role='evaluation'")
        hosts = self.all_hosts
        ids = [host.host_id for host in hosts]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster host IDs must be unique")
        worker_ordinals = [host.ordinal for host in self.workers]
        if len(worker_ordinals) != len(set(worker_ordinals)):
            raise ValueError("worker ordinals must be unique")
        if any(ordinal <= 0 for ordinal in worker_ordinals):
            raise ValueError("worker ordinals must be positive")

    @property
    def all_hosts(self) -> tuple[HostSpec, ...]:
        return (self.coordinator, *self.workers, *self.evaluation_hosts)

    @property
    def arena_hosts(self) -> tuple[HostSpec, ...]:
        return self.evaluation_hosts or self.workers

    def worker(self, worker_id: str) -> HostSpec:
        for host in self.workers:
            if host.host_id == worker_id:
                return host
        raise KeyError(f"unknown worker {worker_id!r}")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_cluster(path: str | Path, cluster: ClusterSpec) -> Path:
    destination = Path(path).expanduser().resolve()
    _atomic_json(
        destination,
        {
            "schema_version": CLUSTER_SCHEMA_VERSION,
            "name": cluster.name,
            "coordinator": asdict(cluster.coordinator),
            "workers": [asdict(host) for host in cluster.workers],
            "evaluation_hosts": [asdict(host) for host in cluster.evaluation_hosts],
        },
    )
    return destination


def load_cluster(path: str | Path) -> ClusterSpec:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read cluster manifest {source}: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != CLUSTER_SCHEMA_VERSION
    ):
        raise ValueError("unsupported cluster manifest")
    return ClusterSpec(
        name=str(value.get("name", "")),
        coordinator=HostSpec(**value["coordinator"]),
        workers=tuple(HostSpec(**item) for item in value.get("workers", ())),
        evaluation_hosts=tuple(
            HostSpec(**item) for item in value.get("evaluation_hosts", ())
        ),
    )


def _remote_script(
    command: Sequence[str],
    *,
    cwd: str | None,
    environment: Mapping[str, str] | None,
) -> str:
    parts: list[str] = ["set -e"]
    if cwd:
        parts.append(f"cd {shlex.quote(cwd)}")
    rendered = shlex.join([str(item) for item in command])
    if environment:
        assignments = " ".join(
            f"{name}={shlex.quote(str(value))}"
            for name, value in sorted(environment.items())
        )
        rendered = f"env {assignments} {rendered}"
    parts.append(rendered)
    return "\n".join(parts)


def run_shell(
    host: HostSpec,
    script: str,
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = (
        ["bash", "-lc", script] if host.is_local else ["ssh", host.address, script]
    )
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )


def run_command(
    host: HostSpec,
    command: Sequence[str],
    *,
    cwd: str | None = None,
    environment: Mapping[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run argv directly when local and through a safely quoted SSH script otherwise."""

    if host.is_local:
        process_environment = os.environ.copy()
        process_environment.update(dict(environment or {}))
        return subprocess.run(
            [str(item) for item in command],
            cwd=cwd,
            env=process_environment,
            check=check,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
        )
    return run_shell(
        host,
        _remote_script(command, cwd=cwd, environment=environment),
        check=check,
        capture_output=capture_output,
    )


def copy_to_host(source: str | Path, host: HostSpec, destination: str) -> None:
    """Copy a file or merge a directory; never delete unrelated remote files."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if host.is_local:
        target = Path(destination).expanduser().resolve()
        if source_path == target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target)
        return
    parent = str(Path(destination).parent)
    run_shell(host, f"mkdir -p {shlex.quote(parent)}")
    if source_path.is_dir():
        run_shell(host, f"mkdir -p {shlex.quote(destination)}")
        source_text = str(source_path).rstrip("/") + "/"
        target_text = f"{host.address}:{destination.rstrip('/')}/"
    else:
        source_text = str(source_path)
        target_text = f"{host.address}:{destination}"
    subprocess.run(["rsync", "-az", "--partial", source_text, target_text], check=True)


def copy_from_host(host: HostSpec, source: str, destination: str | Path) -> None:
    target = Path(destination).expanduser().resolve()
    if host.is_local:
        origin = Path(source).expanduser().resolve()
        if origin == target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target, dirs_exist_ok=True)
        else:
            shutil.copy2(origin, target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source_text = f"{host.address}:{source.rstrip('/')}/"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", "--partial", source_text, str(target) + "/"],
        check=True,
    )


def sync_tree_to_host(
    source: str | Path,
    host: HostSpec,
    destination: str,
    *,
    exclude_names: Sequence[str] = (),
) -> None:
    """Merge a source tree without deleting host-local files or generated caches."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise NotADirectoryError(source_path)
    excluded = {str(name) for name in exclude_names}
    if host.is_local:
        target = Path(destination).expanduser().resolve()
        if source_path == target:
            return
        if source_path in target.parents:
            raise ValueError("source tree cannot be synchronized inside itself")
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source_path,
            target,
            dirs_exist_ok=True,
            ignore=lambda _directory, names: [
                name for name in names if name in excluded or name.endswith(".pyc")
            ],
        )
        return

    run_shell(host, f"mkdir -p {shlex.quote(destination)}")
    command = ["rsync", "-az", "--partial"]
    for name in sorted(excluded):
        command.extend(("--exclude", f"{name}/"))
    command.extend(("--exclude", "*.pyc"))
    command.extend(
        (
            str(source_path).rstrip("/") + "/",
            f"{host.address}:{destination.rstrip('/')}/",
        )
    )
    subprocess.run(command, check=True)


def start_detached(
    host: HostSpec,
    command: Sequence[str],
    *,
    cwd: str,
    environment: Mapping[str, str],
    pid_path: str,
    log_path: str,
) -> int:
    """Start one guarded daemon and return the PID written on its host."""

    rendered = _remote_script(command, cwd=cwd, environment=environment).splitlines()[
        -1
    ]
    script = "\n".join(
        (
            "set -e",
            f"mkdir -p {shlex.quote(str(Path(pid_path).parent))} {shlex.quote(str(Path(log_path).parent))}",
            f"if [ -f {shlex.quote(pid_path)} ] && kill -0 $(cat {shlex.quote(pid_path)}) 2>/dev/null; then exit 17; fi",
            f"cd {shlex.quote(cwd)}",
            f"nohup {rendered} >> {shlex.quote(log_path)} 2>&1 < /dev/null &",
            f"echo $! > {shlex.quote(pid_path)}",
            f"cat {shlex.quote(pid_path)}",
        )
    )
    completed = run_shell(host, script)
    return int((completed.stdout or "").strip().splitlines()[-1])


def process_is_running(host: HostSpec, pid_path: str) -> bool:
    command = (
        f"test -f {shlex.quote(pid_path)} && "
        f"kill -0 $(cat {shlex.quote(pid_path)}) 2>/dev/null"
    )
    return run_shell(host, command, check=False).returncode == 0


def stop_detached(host: HostSpec, pid_path: str, *, timeout_sec: float = 10.0) -> bool:
    """Request TERM, wait briefly, and leave unrelated processes untouched."""

    if not process_is_running(host, pid_path):
        return False
    quoted = shlex.quote(pid_path)
    run_shell(host, f"kill -TERM $(cat {quoted})", check=False)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not process_is_running(host, pid_path):
            run_shell(host, f"rm -f {quoted}", check=False)
            return True
        time.sleep(0.1)
    return False
