
"""Cluster inventory and SSH helpers for AlphaGoZero GV3."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
REMOTE_OUTPUT_ROOT = os.environ.get(
    "AGZ_REMOTE_OUTPUT_ROOT",
    f"{REMOTE_REPO}/simulator_output/GV3_Agent/AlphaGoZero",
)
LOCAL_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HostSpec:
    worker_id: str
    host: str
    ordinal: int
    role: str = "worker"


@dataclass(frozen=True)
class ClusterSpec:
    """A coordinator plus the workers that may serve one experiment."""

    name: str
    xl: HostSpec
    workers: tuple[HostSpec, ...]


DEFAULT_XL = HostSpec(worker_id="xl", host="bellman-classical-xl", ordinal=0, role="xl")
DEFAULT_WORKERS = (
    HostSpec("worker1", "bellman-classical-worker-1", 1),
    HostSpec("worker2", "bellman-classical-worker-2", 2),
    HostSpec("worker3", "bellman-classical-worker-3", 3),
    HostSpec("worker4", "bellman-classical-worker-4", 4),
    HostSpec("worker5", "bellman-classical-worker-5", 5),
    HostSpec("worker6", "bellman-classical-worker-6", 6),
    HostSpec("worker7", "bellman-classical-worker-7", 7),
    HostSpec("worker8", "bellman-classical-worker-8", 8),
)
EXP2_XL = HostSpec(worker_id="xl", host="bellman-classical-exp2-xl", ordinal=0, role="xl")
EXP2_WORKERS = tuple(
    HostSpec(f"worker{index}", f"bellman-classical-exp2-worker-{index}", index)
    for index in range(1, 9)
)
EXP3_XL = HostSpec(worker_id="xl", host="bellman-classical-exp3-xl", ordinal=0, role="xl")
EXP3_WORKERS = tuple(
    HostSpec(f"worker{index}", f"bellman-classical-exp3-worker-{index}", index)
    for index in range(1, 9)
)
SPOT_XL = HostSpec(worker_id="xl", host="localhost", ordinal=0, role="xl")
SPOT_WORKERS: tuple[HostSpec, ...] = ()

CLUSTERS = {
    "default": ClusterSpec("default", DEFAULT_XL, DEFAULT_WORKERS),
    "exp2": ClusterSpec("exp2", EXP2_XL, EXP2_WORKERS),
    "exp3": ClusterSpec("exp3", EXP3_XL, EXP3_WORKERS),
    "spot": ClusterSpec("spot", SPOT_XL, SPOT_WORKERS),
}


def get_cluster(name: str | None = None) -> ClusterSpec:
    """Resolve the selected cluster without changing the legacy default."""
    cluster_name = str(name or os.environ.get("AGZ_CLUSTER", "default")).strip().lower()
    try:
        return CLUSTERS[cluster_name]
    except KeyError as exc:
        available = ", ".join(sorted(CLUSTERS))
        raise ValueError(f"unknown AGZ cluster {cluster_name!r}; expected one of: {available}") from exc


ACTIVE_CLUSTER = get_cluster()
XL = ACTIVE_CLUSTER.xl
WORKERS = list(ACTIVE_CLUSTER.workers)


def parse_workers(raw: str | None) -> list[HostSpec]:
    if not raw or raw.strip().lower() == "all":
        return list(WORKERS)
    wanted = [x.strip() for x in raw.split(",") if x.strip()]
    by_id = {w.worker_id: w for w in WORKERS}
    by_host = {w.host: w for w in WORKERS}
    out: list[HostSpec] = []
    for item in wanted:
        if item in by_id:
            out.append(by_id[item])
        elif item in by_host:
            out.append(by_host[item])
        elif item.isdigit() and f"worker{int(item)}" in by_id:
            out.append(by_id[f"worker{int(item)}"])
        else:
            raise ValueError(f"unknown worker {item!r}")
    return out


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def ssh(host: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["ssh", host, command], check=check)


def rsync_to(local_path: str | Path, host: str, remote_path: str) -> None:
    run(["rsync", "-az", "--delete", str(local_path).rstrip("/") + "/", f"{host}:{remote_path.rstrip('/')}/"])


def rsync_from(host: str, remote_path: str, local_path: str | Path) -> None:
    Path(local_path).mkdir(parents=True, exist_ok=True)
    run(["rsync", "-az", f"{host}:{remote_path.rstrip('/')}/", str(local_path).rstrip("/") + "/"])
