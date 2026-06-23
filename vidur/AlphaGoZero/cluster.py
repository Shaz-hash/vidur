
"""Cluster inventory and SSH helpers for AlphaGoZero GV3."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
REMOTE_OUTPUT_ROOT = f"{REMOTE_REPO}/simulator_output/GV3_Agent/AlphaGoZero"
LOCAL_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HostSpec:
    worker_id: str
    host: str
    ordinal: int
    role: str = "worker"


XL = HostSpec(worker_id="xl", host="bellman-classical-xl", ordinal=0, role="xl")
WORKERS = [
    HostSpec("worker1", "bellman-classical-worker-1", 1),
    HostSpec("worker2", "bellman-classical-worker-2", 2),
    HostSpec("worker3", "bellman-classical-worker-3", 3),
    HostSpec("worker4", "bellman-classical-worker-4", 4),
    HostSpec("worker5", "bellman-classical-worker-5", 5),
    HostSpec("worker6", "bellman-classical-worker-6", 6),
    HostSpec("worker7", "bellman-classical-worker-7", 7),
    HostSpec("worker8", "bellman-classical-worker-8", 8),
]


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
