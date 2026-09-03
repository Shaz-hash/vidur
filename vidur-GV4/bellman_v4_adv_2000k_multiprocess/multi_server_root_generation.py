"""Coordinate multi-server GV3 parent-root generation jobs.

This script is intentionally a thin SSH/rsync coordinator. Each remote machine
runs the existing rootGeneration.py locally, using its own multiprocessing
workers. The coordinator only syncs code, starts jobs, polls status, stops jobs,
and optionally pulls outputs back.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_LOCAL_REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS_PER_SERVER = 450_000
DEFAULT_TOTAL_ROOTS = 2_250_000
DEFAULT_MIN_SELECTED_RATIO = 0.40
DEFAULT_TARGET_ABS_THRESHOLD = 1.0
DEFAULT_HISTORY_SIGNATURE_CACHE_SIZE = 10_000
DEFAULT_HISTORY_MAX_TOTAL_STEPS = 20_000
DEFAULT_CANDIDATE_BATCH_SIZE = 256
DEFAULT_GENERATION_BATCH_SIZE = 8
DEFAULT_SHARD_SIZE = 128
DEFAULT_WORKER_ROOTS_PER_TASK = 64
DEFAULT_MAX_PROCESSES_PER_INTERVAL = 4
DEFAULT_NUM_PROCESSES = 64
DEFAULT_MAX_CANDIDATE_MULTIPLIER = 49
DEFAULT_SEED = 2027

DEFAULT_HOSTS = [
    "bellman-classical",
    "bellman-classical-worker-1",
    "bellman-classical-worker-2",
    "bellman-classical-worker-3",
    "bellman-classical-worker-4",
    "bellman-classical-worker-5",
    "bellman-classical-worker-6",
    "bellman-classical-worker-7",
    "bellman-classical-worker-8",
]


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    host: str
    remote_repo: str
    output_dir: str
    num_roots: int
    max_candidate_roots: int
    num_processes: int
    history_hops_min: int
    history_hops_max: int
    seed: int
    candidate_batch_size: int = DEFAULT_CANDIDATE_BATCH_SIZE
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE
    shard_size: int = DEFAULT_SHARD_SIZE
    worker_roots_per_task: int = DEFAULT_WORKER_ROOTS_PER_TASK
    max_processes_per_interval: int = DEFAULT_MAX_PROCESSES_PER_INTERVAL
    history_signature_cache_size: int = DEFAULT_HISTORY_SIGNATURE_CACHE_SIZE
    min_large_abs_target_ratio: float = DEFAULT_MIN_SELECTED_RATIO
    target_abs_threshold: float = DEFAULT_TARGET_ABS_THRESHOLD
    history_max_total_steps: int = DEFAULT_HISTORY_MAX_TOTAL_STEPS
    progress_every: int = 1000


def _repo_root() -> Path:
    return DEFAULT_LOCAL_REPO


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    text = " ".join(shlex.quote(x) for x in cmd)
    print(f"[cmd] {text}", flush=True)
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _ssh(host: str, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, remote_cmd], check=check)


def _rsync_to(local_path: Path, host: str, remote_path: str) -> None:
    _run(["rsync", "-az", str(local_path), f"{host}:{remote_path}"])


def _rsync_from(host: str, remote_path: str, local_path: Path) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    _run(["rsync", "-az", f"{host}:{remote_path.rstrip('/')}/", f"{local_path}/"])


def split_hop_intervals(hop_min: int, hop_max: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if hop_max < hop_min:
        raise ValueError("hop_max must be >= hop_min")
    total = hop_max - hop_min + 1
    intervals: list[tuple[int, int]] = []
    cursor = hop_min
    for idx in range(parts):
        width = total // parts + (1 if idx < total % parts else 0)
        start = cursor
        end = cursor + width - 1
        intervals.append((start, end))
        cursor = end + 1
    return intervals


def parse_hosts(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_HOSTS)
    hosts = [x.strip() for x in raw.split(",") if x.strip()]
    if not hosts:
        raise ValueError("--hosts produced no hosts")
    return hosts


def build_jobs(args: argparse.Namespace, *, smoke: bool = False) -> list[JobSpec]:
    hosts = parse_hosts(args.hosts)
    if smoke:
        hosts = [args.smoke_host]

    intervals = split_hop_intervals(int(args.hop_min), int(args.hop_max), max(1, len(hosts)))
    jobs: list[JobSpec] = []
    for idx, host in enumerate(hosts):
        hmin, hmax = intervals[idx]
        roots = int(args.smoke_roots if smoke else args.roots_per_server)
        max_candidates = int(args.smoke_max_candidates if smoke else args.max_candidates_per_server)
        if max_candidates <= 0:
            max_candidates = roots * int(args.max_candidate_multiplier)

        num_processes = int(args.smoke_num_processes if smoke else args.num_processes)
        output_base = str(args.remote_output_base).rstrip("/")
        suffix = (
            f"{args.smoke_label}_{host}_hops_{hmin}_{hmax}"
            if smoke
            else f"server_{idx:02d}_{host}_hops_{hmin}_{hmax}"
        )
        output_dir = f"{str(args.remote_repo).rstrip('/')}/{output_base}/{EXPERIMENT_NAME}/{suffix}"
        jobs.append(
            JobSpec(
                job_id=f"server_{idx:02d}" if not smoke else "smoke_00",
                host=str(host),
                remote_repo=str(args.remote_repo),
                output_dir=output_dir,
                num_roots=roots,
                max_candidate_roots=max_candidates,
                num_processes=num_processes,
                history_hops_min=int(hmin),
                history_hops_max=int(hmax),
                seed=int(args.seed) + idx * 10_000,
                candidate_batch_size=int(args.candidate_batch_size),
                generation_batch_size=int(args.generation_batch_size),
                shard_size=int(args.shard_size),
                worker_roots_per_task=int(args.smoke_worker_roots_per_task if smoke else args.worker_roots_per_task),
                max_processes_per_interval=int(args.max_processes_per_interval),
                history_signature_cache_size=int(args.history_signature_cache_size),
                min_large_abs_target_ratio=float(args.min_large_abs_target_ratio),
                target_abs_threshold=float(args.target_abs_threshold),
                history_max_total_steps=int(args.history_max_total_steps),
                progress_every=int(args.progress_every),
            )
        )
    return jobs


def root_generation_command(job: JobSpec) -> list[str]:
    python = f"{job.remote_repo.rstrip('/')}/.venv/bin/python3"
    return [
        python,
        "-m",
        "vidur.Game_Version3.ModelSearchBed.analysis_testing.rootGeneration",
        "--output-dir",
        job.output_dir,
        "--num-roots",
        str(job.num_roots),
        "--max-candidate-roots",
        str(job.max_candidate_roots),
        "--num-processes",
        str(job.num_processes),
        "--candidate-batch-size",
        str(job.candidate_batch_size),
        "--generation-batch-size",
        str(job.generation_batch_size),
        "--shard-size",
        str(job.shard_size),
        "--worker-roots-per-task",
        str(job.worker_roots_per_task),
        "--max-processes-per-interval",
        str(job.max_processes_per_interval),
        "--history-signature-cache-size",
        str(job.history_signature_cache_size),
        "--min-large-abs-target-ratio",
        str(job.min_large_abs_target_ratio),
        "--target-abs-threshold",
        str(job.target_abs_threshold),
        "--history-hops-min",
        str(job.history_hops_min),
        "--history-hops-max",
        str(job.history_hops_max),
        "--history-max-total-steps",
        str(job.history_max_total_steps),
        "--seed",
        str(job.seed),
        "--progress-every",
        str(job.progress_every),
    ]


def sync_host(host: str, remote_repo: str) -> None:
    local_repo = _repo_root()
    remote_repo = remote_repo.rstrip("/")

    _ssh(
        host,
        "mkdir -p "
        + " ".join(
            shlex.quote(p)
            for p in [
                f"{remote_repo}/vidur",
                f"{remote_repo}/simulator_output",
            ]
        ),
    )

    # Sync the self-contained GV3 tree used by root generation, plus the shared
    # trace validators imported by GV3 tests.
    _rsync_to(local_repo / "vidur/Game_Version3", host, f"{remote_repo}/vidur/")
    _rsync_to(local_repo / "vidur/tests", host, f"{remote_repo}/vidur/")
    _rsync_to(
        local_repo / "vidur/bellman_v4_adv_2000k_multiprocess",
        host,
        f"{remote_repo}/vidur/",
    )

    # Keep profile lookup tables aligned; they are required by the simulator.
    for rel in ("simulator_output/prefill_profile.csv", "simulator_output/decode_profile.csv"):
        src = local_repo / rel
        if src.exists():
            _rsync_to(src, host, f"{remote_repo}/{rel}")

    compile_cmd = (
        f"cd {shlex.quote(remote_repo)} && "
        f"{shlex.quote(remote_repo + '/.venv/bin/python3')} -m py_compile "
        "vidur/Game_Version3/ModelSearchBed/analysis_testing/rootGeneration.py "
        "vidur/Game_Version3/ModelSearchBed/root_storage.py "
        "vidur/Game_Version3/DNN/history_root.py "
        "vidur/Game_Version3/mctsDNN.py "
        "vidur/Game_Version3/tests/history_node_tests.py "
        "vidur/tests/game_engine_tests.py "
        "vidur/tests/run_game_engine_trace_tests.py "
        "vidur/bellman_v4_adv_2000k_multiprocess/multi_server_root_generation.py"
    )
    out = _ssh(host, compile_cmd)
    if out.stdout.strip():
        print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def sync_jobs(jobs: Iterable[JobSpec]) -> None:
    seen: set[tuple[str, str]] = set()
    for job in jobs:
        key = (job.host, job.remote_repo)
        if key in seen:
            continue
        seen.add(key)
        print(f"[sync] host={job.host} repo={job.remote_repo}", flush=True)
        sync_host(job.host, job.remote_repo)


def launch_job(job: JobSpec, *, force: bool = False) -> None:
    job_json = json.dumps(asdict(job), indent=2, sort_keys=True)
    cmd = root_generation_command(job)
    quoted_cmd = " ".join(shlex.quote(x) for x in cmd)
    out_dir = shlex.quote(job.output_dir)
    job_json_q = shlex.quote(job_json)

    remove_existing = f"rm -rf {out_dir}; " if force else ""
    remote = (
        "set -e; "
        f"cd {shlex.quote(job.remote_repo)}; "
        f"{remove_existing}"
        f"mkdir -p {out_dir}; "
        f"printf '%s\\n' {job_json_q} > {out_dir}/job_config.json; "
        f"printf '%s\\n' {shlex.quote(quoted_cmd)} > {out_dir}/command.txt; "
        f"nohup {quoted_cmd} > {out_dir}/root_generation.log 2>&1 < /dev/null & "
        f"echo $! > {out_dir}/root_generation.pid; "
        f"echo launched pid=$(cat {out_dir}/root_generation.pid) out={out_dir}"
    )
    out = _ssh(job.host, remote)
    print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def launch_jobs(jobs: Iterable[JobSpec], *, force: bool = False) -> None:
    for job in jobs:
        print(
            f"[launch] {job.host} {job.job_id} roots={job.num_roots} "
            f"hops={job.history_hops_min}-{job.history_hops_max} out={job.output_dir}",
            flush=True,
        )
        launch_job(job, force=force)


def remote_status(job: JobSpec, *, tail_lines: int = 20) -> dict[str, object]:
    out_dir = shlex.quote(job.output_dir)
    remote = (
        f"OUT={out_dir}; "
        "PID=$(cat \"$OUT/root_generation.pid\" 2>/dev/null || true); "
        "RUNNING=0; "
        "if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then RUNNING=1; fi; "
        "ROOTS=\"\"; SELECTED=\"\"; RATIO=\"\"; SHARDS=\"\"; "
        "if [ -f \"$OUT/summary.json\" ]; then "
        f"{shlex.quote(job.remote_repo + '/.venv/bin/python3')} - <<'PY' \"$OUT/summary.json\"\n"
        "import json,sys\n"
        "p=sys.argv[1]\n"
        "d=json.load(open(p))\n"
        "s=d.get('stats',{})\n"
        "print('SUMMARY_ROOTS=%s' % s.get('roots_stored',''))\n"
        "print('SUMMARY_SELECTED=%s' % s.get('nonzero_roots_stored',''))\n"
        "print('SUMMARY_RATIO=%s' % d.get('nonzero_ratio',''))\n"
        "print('SUMMARY_SHARDS=%s' % s.get('shards_written',''))\n"
        "PY\n"
        "fi; "
        "echo STATUS_PID=$PID; echo STATUS_RUNNING=$RUNNING; "
        "echo STATUS_OUT=$OUT; "
        "echo STATUS_LOG_TAIL_BEGIN; "
        f"tail -n {int(tail_lines)} \"$OUT/root_generation.log\" 2>/dev/null || true; "
        "echo STATUS_LOG_TAIL_END"
    )
    completed = _ssh(job.host, remote, check=False)
    stdout = completed.stdout
    result: dict[str, object] = {
        "job_id": job.job_id,
        "host": job.host,
        "output_dir": job.output_dir,
        "returncode": completed.returncode,
        "raw": stdout,
    }
    for line in stdout.splitlines():
        if line.startswith("STATUS_PID="):
            result["pid"] = line.split("=", 1)[1]
        elif line.startswith("STATUS_RUNNING="):
            result["running"] = line.split("=", 1)[1] == "1"
        elif line.startswith("SUMMARY_ROOTS="):
            result["roots_stored"] = line.split("=", 1)[1]
        elif line.startswith("SUMMARY_SELECTED="):
            result["selected_roots"] = line.split("=", 1)[1]
        elif line.startswith("SUMMARY_RATIO="):
            result["selected_ratio"] = line.split("=", 1)[1]
        elif line.startswith("SUMMARY_SHARDS="):
            result["shards"] = line.split("=", 1)[1]
    return result


def print_status(jobs: Iterable[JobSpec], *, tail_lines: int = 20) -> None:
    rows: list[dict[str, object]] = []
    for job in jobs:
        status = remote_status(job, tail_lines=tail_lines)
        rows.append(status)
        print("=" * 100)
        print(
            f"{job.host} {job.job_id} running={status.get('running')} "
            f"pid={status.get('pid', '')} roots={status.get('roots_stored', '')} "
            f"selected={status.get('selected_roots', '')} ratio={status.get('selected_ratio', '')}"
        )
        print(str(status.get("raw", "")).split("STATUS_LOG_TAIL_BEGIN", 1)[-1].split("STATUS_LOG_TAIL_END", 1)[0])

    print("=" * 100)
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["job_id", "host", "running", "pid", "roots_stored", "selected_roots", "selected_ratio", "shards", "output_dir"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def stop_jobs(jobs: Iterable[JobSpec]) -> None:
    for job in jobs:
        out_dir = shlex.quote(job.output_dir)
        remote = (
            f"OUT={out_dir}; "
            "PID=$(cat \"$OUT/root_generation.pid\" 2>/dev/null || true); "
            "if [ -n \"$PID\" ]; then kill -TERM \"$PID\" 2>/dev/null || true; fi; "
            "pkill -TERM -f \"$OUT\" 2>/dev/null || true; "
            "echo stopped out=$OUT pid=$PID"
        )
        out = _ssh(job.host, remote, check=False)
        print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def collect_jobs(jobs: Iterable[JobSpec], local_collect_dir: Path) -> None:
    for job in jobs:
        dst = local_collect_dir / job.job_id
        print(f"[collect] {job.host}:{job.output_dir} -> {dst}", flush=True)
        _rsync_from(job.host, job.output_dir, dst)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hosts", default=None, help="Comma-separated SSH hosts for full launch.")
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument(
        "--remote-output-base",
        default="simulator_output/GV3_Agent/ModelSearchBed",
        help="Path relative to remote repo for generated dataset outputs.",
    )
    parser.add_argument("--hop-min", type=int, default=0)
    parser.add_argument("--hop-max", type=int, default=750)
    parser.add_argument("--roots-per-server", type=int, default=DEFAULT_ROOTS_PER_SERVER)
    parser.add_argument("--max-candidates-per-server", type=int, default=0)
    parser.add_argument("--max-candidate-multiplier", type=int, default=DEFAULT_MAX_CANDIDATE_MULTIPLIER)
    parser.add_argument("--num-processes", type=int, default=DEFAULT_NUM_PROCESSES)
    parser.add_argument("--candidate-batch-size", type=int, default=DEFAULT_CANDIDATE_BATCH_SIZE)
    parser.add_argument("--generation-batch-size", type=int, default=DEFAULT_GENERATION_BATCH_SIZE)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--worker-roots-per-task", type=int, default=DEFAULT_WORKER_ROOTS_PER_TASK)
    parser.add_argument("--max-processes-per-interval", type=int, default=DEFAULT_MAX_PROCESSES_PER_INTERVAL)
    parser.add_argument("--history-signature-cache-size", type=int, default=DEFAULT_HISTORY_SIGNATURE_CACHE_SIZE)
    parser.add_argument("--min-large-abs-target-ratio", type=float, default=DEFAULT_MIN_SELECTED_RATIO)
    parser.add_argument("--target-abs-threshold", type=float, default=DEFAULT_TARGET_ABS_THRESHOLD)
    parser.add_argument("--history-max-total-steps", type=int, default=DEFAULT_HISTORY_MAX_TOTAL_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--smoke-host", default="bellman-classical")
    parser.add_argument("--smoke-roots", type=int, default=100)
    parser.add_argument("--smoke-max-candidates", type=int, default=2000)
    parser.add_argument("--smoke-num-processes", type=int, default=16)
    parser.add_argument("--smoke-worker-roots-per-task", type=int, default=25)
    parser.add_argument("--smoke-label", default="smoke_latest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("sync", "launch", "status", "tail", "stop", "collect", "smoke"):
        p = sub.add_parser(name)
        add_common_args(p)
        if name == "launch":
            p.add_argument("--force", action="store_true", help="Remove remote output dirs before launch.")
        if name == "status":
            p.add_argument("--tail-lines", type=int, default=20)
        if name == "tail":
            p.add_argument("--job-index", type=int, default=0)
            p.add_argument("--tail-lines", type=int, default=80)
        if name == "collect":
            p.add_argument("--local-collect-dir", default=str(_repo_root() / "simulator_output/GV3_Agent/ModelSearchBed/collected_2250k_roots"))
        if name == "smoke":
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-sync", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = args.cmd == "smoke"
    jobs = build_jobs(args, smoke=smoke)

    if args.cmd == "sync":
        sync_jobs(jobs)
    elif args.cmd == "launch":
        launch_jobs(jobs, force=bool(args.force))
    elif args.cmd == "status":
        print_status(jobs, tail_lines=int(args.tail_lines))
    elif args.cmd == "tail":
        idx = int(args.job_index)
        if idx < 0 or idx >= len(jobs):
            raise IndexError(f"--job-index out of range: {idx}")
        status = remote_status(jobs[idx], tail_lines=int(args.tail_lines))
        print(status.get("raw", ""))
    elif args.cmd == "stop":
        stop_jobs(jobs)
    elif args.cmd == "collect":
        collect_jobs(jobs, Path(args.local_collect_dir).expanduser())
    elif args.cmd == "smoke":
        if not bool(args.no_sync):
            sync_jobs(jobs)
        launch_jobs(jobs, force=bool(args.force))
        time.sleep(5)
        print_status(jobs, tail_lines=40)
    else:
        raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
