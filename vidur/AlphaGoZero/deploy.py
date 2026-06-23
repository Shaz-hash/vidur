"""Deployment helper for GV3 AlphaGoZero workers and Classical XL.

This script intentionally wraps only boring SSH/rsync operations. It does not
hide state changes: every remote command is printed before execution.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from vidur.AlphaGoZero.cluster import LOCAL_REPO, REMOTE_OUTPUT_ROOT, REMOTE_REPO, WORKERS, XL, HostSpec, parse_workers

REMOTE_PYTHON = f"{REMOTE_REPO}/.venv/bin/python3"


WORKER_PARENT_DATASETS = {
    "worker1": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_01_bellman-classical-worker-1_hops_151_300",
    "worker2": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_02_bellman-classical-worker-2_hops_301_450",
    "worker3": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_03_bellman-classical-worker-3_hops_451_600",
    "worker4": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_04_bellman-classical-worker-4_hops_601_750",
    "worker5": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_00_bellman-classical-worker-5_hops_0_100",
    "worker6": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_01_bellman-classical-worker-6_hops_100_200",
    "worker7": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_02_bellman-classical-worker-7_hops_200_300",
    "worker8": f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_03_bellman-classical-worker-8_hops_300_400",
}


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    cp = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode != 0 and cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n", flush=True)
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args, output=cp.stdout)
    return cp


def _ssh(host: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, command], check=check)


def _rsync(local: str | Path, remote: str) -> None:
    _run([
        "rsync", "-az", "--partial", "--delay-updates", "--timeout=60",
        str(local), remote,
    ])


def _hosts(args: argparse.Namespace) -> list[HostSpec]:
    hosts: list[HostSpec] = []
    if bool(args.include_xl):
        hosts.append(XL)
    hosts.extend(parse_workers(args.workers))
    seen: set[str] = set()
    out: list[HostSpec] = []
    for h in hosts:
        if h.host in seen:
            continue
        seen.add(h.host)
        out.append(h)
    return out


def install_key(args: argparse.Namespace) -> None:
    key = Path(args.key_path).expanduser().resolve()
    if not key.is_file():
        raise FileNotFoundError(key)
    config = Path(args.ssh_config_path).expanduser().resolve()
    for h in _hosts(args):
        _ssh(h.host, "mkdir -p ~/.ssh && chmod 700 ~/.ssh")
        _rsync(key, f"{h.host}:~/.ssh/{key.name}")
        if config.is_file() and bool(args.upload_ssh_config):
            _rsync(config, f"{h.host}:~/.ssh/config")
            _ssh(h.host, f"chmod 600 ~/.ssh/{shlex.quote(key.name)} ~/.ssh/config")
        else:
            _ssh(h.host, f"chmod 600 ~/.ssh/{shlex.quote(key.name)}")


def sync_code(args: argparse.Namespace) -> None:
    src = str(LOCAL_REPO).rstrip("/") + "/"
    excludes = [
        "--exclude", ".git/",
        "--exclude", "simulator_output/",
        "--exclude", "__pycache__/",
        "--exclude", "*.pyc",
        "--exclude", ".venv",
        "--exclude", ".venv/",
        "--exclude", "cache",
        "--exclude", "cache/",
        "--exclude", "vidur/Game_Version3_Cpp/build/",
    ]
    for h in _hosts(args):
        _ssh(h.host, f"mkdir -p {shlex.quote(REMOTE_REPO)}")
        _run([
            "rsync", "-az", "--partial", "--delay-updates", "--timeout=120",
            *excludes,
            src,
            f"{h.host}:{REMOTE_REPO.rstrip('/')}/",
        ])


def build_native(args: argparse.Namespace) -> None:
    for h in _hosts(args):
        cmd = (
            f"cd {shlex.quote(REMOTE_REPO)} && "
            "rm -rf vidur/Game_Version3_Cpp/build/CMakeCache.txt vidur/Game_Version3_Cpp/build/CMakeFiles && "
            f"cmake -S vidur/Game_Version3_Cpp -B vidur/Game_Version3_Cpp/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE={shlex.quote(REMOTE_PYTHON)} && "
            "cmake --build vidur/Game_Version3_Cpp/build -j$(nproc)"
        )
        _ssh(h.host, cmd)


def launch_xl_ingest(args: argparse.Namespace) -> None:
    log = f"{REMOTE_OUTPUT_ROOT}/xl_coordinator.log"
    cmd = (
        f"mkdir -p {shlex.quote(REMOTE_OUTPUT_ROOT)} && "
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"nohup {shlex.quote(REMOTE_PYTHON)} -m vidur.AlphaGoZero.xl_coordinator --output-root {shlex.quote(REMOTE_OUTPUT_ROOT)} --loop --poll-sec {float(args.poll_sec)} "
        f"< /dev/null > {shlex.quote(log)} 2>&1 & echo $!"
    )
    print(_ssh(XL.host, cmd).stdout, end="")


def launch_worker_smoke(args: argparse.Namespace) -> None:
    for h in parse_workers(args.workers):
        worker_root = f"{REMOTE_OUTPUT_ROOT}/worker_smoke/{h.worker_id}"
        log = f"{worker_root}/worker_daemon.log"
        game_id_start = int(args.game_id_start) + h.ordinal * 10_000
        cmd = (
            f"mkdir -p {shlex.quote(worker_root)} && "
            f"cd {shlex.quote(REMOTE_REPO)} && "
            f"nohup {shlex.quote(REMOTE_PYTHON)} -m vidur.AlphaGoZero.worker_daemon "
            f"--worker-id {shlex.quote(h.worker_id)} "
            f"--run-id smoke_{shlex.quote(h.worker_id)} "
            f"--output-root {shlex.quote(worker_root)} "
            f"--xl-host {shlex.quote(XL.host)} "
            f"--xl-output-root {shlex.quote(REMOTE_OUTPUT_ROOT)} "
            f"--game-id-start {game_id_start} "
            f"--max-games 1 --parallel-games 1 --buffer-threshold 1 --flush-at-end --upload-after-game "
            f"--iterations {int(args.iterations)} --history-hops {int(args.history_hops)} "
            f"--ack-timeout-sec {int(args.ack_timeout_sec)} "
            f"< /dev/null > {shlex.quote(log)} 2>&1 & echo $!"
        )
        print(f"[{h.worker_id}]", _ssh(h.host, cmd).stdout, end="")



def _remote_parent_count(host: str, dataset_dir: str, root_player_filter: str) -> int:
    script = (
        "from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import count_samples; "
        f"print(count_samples({dataset_dir!r}, root_player_filter={root_player_filter!r}))"
    )
    cmd = f"cd {shlex.quote(REMOTE_REPO)} && {shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(script)}"
    cp = _ssh(host, cmd)
    out = (cp.stdout or "").strip().splitlines()[-1]
    return int(out)


def launch_worker_large(args: argparse.Namespace) -> None:
    for h in parse_workers(args.workers):
        worker_root = f"{REMOTE_OUTPUT_ROOT}/worker_large/{h.worker_id}"
        log = f"{worker_root}/worker_daemon.log"
        game_id_start = int(args.game_id_start) + h.ordinal * 10_000_000
        parent_dir = str(args.parent_dataset_dir or WORKER_PARENT_DATASETS[h.worker_id])
        parent_count = int(args.parent_state_count)
        if parent_count <= 0:
            parent_count = _remote_parent_count(h.host, parent_dir, str(args.parent_root_player_filter))
        worker_cmd = (
            f"{shlex.quote(REMOTE_PYTHON)} -m vidur.AlphaGoZero.worker_daemon "
            f"--worker-id {shlex.quote(h.worker_id)} "
            f"--run-id {shlex.quote(args.run_id)} "
            f"--output-root {shlex.quote(worker_root)} "
            f"--xl-host {shlex.quote(XL.host)} "
            f"--xl-output-root {shlex.quote(REMOTE_OUTPUT_ROOT)} "
            f"--game-id-start {game_id_start} "
            f"--max-games {int(args.max_games)} "
            f"--parallel-games {int(args.parallel_games)} "
            f"--buffer-threshold {int(args.buffer_threshold)} "
            f"--upload-after-game --upload-poll-sec {float(args.upload_poll_sec)} "
            f"--iterations {int(args.iterations)} --history-hops {int(args.history_hops)} "
            f"--ack-timeout-sec {int(args.ack_timeout_sec)} "
            f"--parent-dataset-dir {shlex.quote(parent_dir)} "
            f"--parent-state-count {int(parent_count)} "
            f"--parent-root-player-filter {shlex.quote(args.parent_root_player_filter)}"
        )
        cmd = (
            f"mkdir -p {shlex.quote(worker_root)} && "
            f"cd {shlex.quote(REMOTE_REPO)} && "
            f"setsid -f sh -c {shlex.quote(worker_cmd)} < /dev/null > {shlex.quote(log)} 2>&1 && "
            f"echo launched"
        )
        cp = _ssh(h.host, cmd)
        print(f"[{h.worker_id}] {cp.stdout.strip()} parent_count={parent_count} root={worker_root}")


def large_status(args: argparse.Namespace) -> None:
    for h in parse_workers(args.workers):
        worker_root = f"{REMOTE_OUTPUT_ROOT}/worker_large/{h.worker_id}"
        cmd = (
            f"printf 'host={h.host}\n'; "
            "pgrep -af 'vidur.AlphaGoZero.worker_daemon' || true; "
            f"printf '\nworker replay:\n'; tail -5 {shlex.quote(worker_root)}/replay_buffer.csv 2>/dev/null || true; "
            f"printf '\nmodel communication:\n'; tail -5 {shlex.quote(worker_root)}/'model communication.csv' 2>/dev/null || true; "
            f"printf '\nready shards:\n'; find {shlex.quote(worker_root)}/ready -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l; "
            f"printf '\nrecent log:\n'; tail -20 {shlex.quote(worker_root)}/worker_daemon.log 2>/dev/null || true"
        )
        cp = _ssh(h.host, cmd, check=False)
        print(cp.stdout or "")

def smoke_status(args: argparse.Namespace) -> None:
    for h in parse_workers(args.workers):
        worker_root = f"{REMOTE_OUTPUT_ROOT}/worker_smoke/{h.worker_id}"
        cmd = (
            f"printf 'host={h.host}\\n'; "
            f"tail -20 {shlex.quote(worker_root)}/worker_daemon.log 2>/dev/null || true; "
            f"printf '\\nworker replay:\\n'; tail -5 {shlex.quote(worker_root)}/replay_buffer.csv 2>/dev/null || true; "
            f"printf '\\nxl ack:\\n'; ls -1 {shlex.quote(REMOTE_OUTPUT_ROOT)}/acks/{shlex.quote(h.worker_id)} 2>/dev/null || true"
        )
        cp = _ssh(h.host, cmd, check=False)
        print(cp.stdout or "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy/smoke helper for GV3 AlphaGoZero.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_hosts = argparse.ArgumentParser(add_help=False)
    common_hosts.add_argument("--workers", default="all", help="all, comma-separated worker ids, host aliases, or ordinals")
    common_hosts.add_argument("--include-xl", action=argparse.BooleanOptionalAction, default=False)

    p_key = sub.add_parser("install-key", parents=[common_hosts])
    p_key.add_argument("--key-path", default="~/.ssh/shaz-pr1.pem")
    p_key.add_argument("--ssh-config-path", default="~/.ssh/config")
    p_key.add_argument("--upload-ssh-config", action=argparse.BooleanOptionalAction, default=True)
    p_key.set_defaults(func=install_key)

    p_sync = sub.add_parser("sync-code", parents=[common_hosts])
    p_sync.set_defaults(func=sync_code)

    p_build = sub.add_parser("build-native", parents=[common_hosts])
    p_build.set_defaults(func=build_native)

    p_xl = sub.add_parser("launch-xl-ingest")
    p_xl.add_argument("--poll-sec", type=float, default=10.0)
    p_xl.set_defaults(func=launch_xl_ingest)

    p_smoke = sub.add_parser("launch-worker-smoke", parents=[common_hosts])
    p_smoke.add_argument("--iterations", type=int, default=1000)
    p_smoke.add_argument("--history-hops", type=int, default=0)
    p_smoke.add_argument("--game-id-start", type=int, default=19_000_000)
    p_smoke.add_argument("--ack-timeout-sec", type=int, default=0)
    p_smoke.set_defaults(func=launch_worker_smoke)

    p_large = sub.add_parser("launch-worker-large", parents=[common_hosts])
    p_large.add_argument("--run-id", default="agz_large_hgb63_a2_v100")
    p_large.add_argument("--iterations", type=int, default=1000)
    p_large.add_argument("--history-hops", type=int, default=0)
    p_large.add_argument("--game-id-start", type=int, default=20_000_000)
    p_large.add_argument("--max-games", type=int, default=0, help="0 means run forever")
    p_large.add_argument("--parallel-games", type=int, default=60)
    p_large.add_argument("--buffer-threshold", type=int, default=10_000)
    p_large.add_argument("--upload-poll-sec", type=float, default=30.0)
    p_large.add_argument("--ack-timeout-sec", type=int, default=0)
    p_large.add_argument("--parent-dataset-dir", default="")
    p_large.add_argument("--parent-state-count", type=int, default=0)
    p_large.add_argument("--parent-root-player-filter", choices=("controller", "adversary", "any"), default="any")
    p_large.set_defaults(func=launch_worker_large)

    p_status_large = sub.add_parser("large-status", parents=[common_hosts])
    p_status_large.set_defaults(func=large_status)

    p_status = sub.add_parser("smoke-status", parents=[common_hosts])
    p_status.set_defaults(func=smoke_status)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
