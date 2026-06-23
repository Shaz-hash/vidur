"""Coordinate per-server GV3 Adv child-transition cache generation.

This is the next phase after the distributed root dataset is generated. Each
remote machine builds child transitions for its own root partition only. The
actual child-generation implementation is the standalone GV3 script:

    vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv

The coordinator only starts/stops/polls jobs over SSH.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_HOSTS = [
    "bellman-classical",
    "bellman-classical-worker-1",
    "bellman-classical-worker-2",
    "bellman-classical-worker-3",
    "bellman-classical-worker-4",
]


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    host: str
    remote_repo: str
    root_dataset_dir: str
    child_cache_dir: str
    num_processes: int
    parents_per_task: int
    transition_shard_size: int
    validate_first_n: int
    seed: int


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _ssh(host: str, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, remote_cmd], check=check)


def parse_hosts(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_HOSTS)
    hosts = [x.strip() for x in raw.split(",") if x.strip()]
    if not hosts:
        raise ValueError("--hosts produced no hosts")
    return hosts


def split_hop_intervals(hop_min: int, hop_max: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if hop_max < hop_min:
        raise ValueError("hop_max must be >= hop_min")
    total = hop_max - hop_min + 1
    out: list[tuple[int, int]] = []
    cursor = hop_min
    for idx in range(parts):
        width = total // parts + (1 if idx < total % parts else 0)
        start = cursor
        end = cursor + width - 1
        out.append((start, end))
        cursor = end + 1
    return out


def build_jobs(args: argparse.Namespace) -> list[JobSpec]:
    hosts = parse_hosts(args.hosts)
    intervals = split_hop_intervals(int(args.hop_min), int(args.hop_max), len(hosts))
    jobs: list[JobSpec] = []
    base = f"{str(args.remote_repo).rstrip('/')}/{str(args.remote_output_base).strip('/')}"
    for idx, (host, (hmin, hmax)) in enumerate(zip(hosts, intervals)):
        suffix = f"server_{idx:02d}_{host}_hops_{hmin}_{hmax}"
        root_dir = f"{base}/{args.experiment_name}/{suffix}"
        child_dir = str(args.child_cache_dir).format(
            remote_repo=str(args.remote_repo).rstrip("/"),
            remote_output_base=str(args.remote_output_base).strip("/"),
            experiment_name=str(args.experiment_name),
            suffix=suffix,
            root_dataset_dir=root_dir,
            host=host,
            job_id=f"server_{idx:02d}",
            hop_min=hmin,
            hop_max=hmax,
        )
        jobs.append(
            JobSpec(
                job_id=f"server_{idx:02d}",
                host=host,
                remote_repo=str(args.remote_repo),
                root_dataset_dir=root_dir,
                child_cache_dir=child_dir,
                num_processes=int(args.num_processes),
                parents_per_task=int(args.parents_per_task),
                transition_shard_size=int(args.transition_shard_size),
                validate_first_n=int(args.validate_first_n),
                seed=int(args.seed) + idx * 10_000,
            )
        )
    return jobs


def command_for(job: JobSpec, *, max_roots: int | None = None) -> list[str]:
    cmd = [
        f"{job.remote_repo.rstrip('/')}/.venv/bin/python3",
        "-m",
        "vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv",
        "--dataset-dir",
        job.root_dataset_dir,
        "--output-dir",
        job.child_cache_dir,
        "--root-player-filter",
        "controller",
        "--transition-shard-size",
        str(job.transition_shard_size),
        "--num-processes",
        str(job.num_processes),
        "--parents-per-task",
        str(job.parents_per_task),
        "--validate-first-n",
        str(job.validate_first_n),
        "--seed",
        str(job.seed),
        "--overwrite",
    ]
    if max_roots is not None and int(max_roots) > 0:
        cmd.extend(["--max-roots", str(int(max_roots))])
    return cmd


def launch_job(job: JobSpec, *, max_roots: int | None = None) -> None:
    out_dir = shlex.quote(job.child_cache_dir)
    quoted_cmd = " ".join(shlex.quote(x) for x in command_for(job, max_roots=max_roots))
    payload = shlex.quote(json.dumps(asdict(job), indent=2, sort_keys=True))
    remote = (
        "set -e; "
        f"cd {shlex.quote(job.remote_repo)}; "
        f"rm -rf {out_dir}; mkdir -p {out_dir}; "
        f"printf '%s\\n' {payload} > {out_dir}/job_config.json; "
        f"printf '%s\\n' {shlex.quote(quoted_cmd)} > {out_dir}/command.txt; "
        f"nohup {quoted_cmd} > {out_dir}/child_cache_generation.log 2>&1 < /dev/null & "
        f"echo $! > {out_dir}/child_cache_generation.pid; "
        f"echo launched pid=$(cat {out_dir}/child_cache_generation.pid) out={out_dir}"
    )
    out = _ssh(job.host, remote)
    print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def status_job(job: JobSpec, *, tail_lines: int = 20) -> dict[str, object]:
    out_dir = shlex.quote(job.child_cache_dir)
    remote = (
        f"OUT={out_dir}; "
        "PID=$(cat \"$OUT/child_cache_generation.pid\" 2>/dev/null || true); "
        "RUNNING=0; if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then RUNNING=1; fi; "
        "if [ -f \"$OUT/summary.json\" ]; then "
        f"{shlex.quote(job.remote_repo.rstrip('/') + '/.venv/bin/python3')} - <<'PY' \"$OUT/summary.json\"\n"
        "import json,sys\n"
        "d=json.load(open(sys.argv[1]))\n"
        "print('SUMMARY_PARENTS=%s' % d.get('num_parent_states',''))\n"
        "print('SUMMARY_TRANSITIONS=%s' % d.get('num_transitions',''))\n"
        "print('SUMMARY_SHARDS=%s' % d.get('num_shards',''))\n"
        "PY\n"
        "fi; "
        "echo STATUS_PID=$PID; echo STATUS_RUNNING=$RUNNING; echo STATUS_OUT=$OUT; "
        "echo STATUS_LOG_TAIL_BEGIN; "
        f"tail -n {int(tail_lines)} \"$OUT/child_cache_generation.log\" 2>/dev/null || true; "
        "echo STATUS_LOG_TAIL_END"
    )
    completed = _ssh(job.host, remote, check=False)
    result: dict[str, object] = {"job_id": job.job_id, "host": job.host, "output_dir": job.child_cache_dir, "raw": completed.stdout}
    for line in completed.stdout.splitlines():
        if line.startswith("STATUS_PID="):
            result["pid"] = line.split("=", 1)[1]
        elif line.startswith("STATUS_RUNNING="):
            result["running"] = line.split("=", 1)[1] == "1"
        elif line.startswith("SUMMARY_PARENTS="):
            result["parents"] = line.split("=", 1)[1]
        elif line.startswith("SUMMARY_TRANSITIONS="):
            result["transitions"] = line.split("=", 1)[1]
        elif line.startswith("SUMMARY_SHARDS="):
            result["shards"] = line.split("=", 1)[1]
    return result


def print_status(jobs: list[JobSpec], *, tail_lines: int) -> None:
    rows = [status_job(job, tail_lines=tail_lines) for job in jobs]
    for row in rows:
        print("=" * 100)
        print(f"{row['host']} {row['job_id']} running={row.get('running')} pid={row.get('pid','')} parents={row.get('parents','')} transitions={row.get('transitions','')} shards={row.get('shards','')}")
        print(str(row.get("raw", "")).split("STATUS_LOG_TAIL_BEGIN", 1)[-1].split("STATUS_LOG_TAIL_END", 1)[0])
    print("=" * 100)
    writer = csv.DictWriter(sys.stdout, fieldnames=["job_id", "host", "running", "pid", "parents", "transitions", "shards", "output_dir"])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def stop_jobs(jobs: list[JobSpec]) -> None:
    for job in jobs:
        out_dir = shlex.quote(job.child_cache_dir)
        remote = (
            f"OUT={out_dir}; PID=$(cat \"$OUT/child_cache_generation.pid\" 2>/dev/null || true); "
            "if [ -n \"$PID\" ]; then kill -TERM \"$PID\" 2>/dev/null || true; fi; "
            "pkill -TERM -f \"$OUT\" 2>/dev/null || true; "
            "echo stopped out=$OUT pid=$PID"
        )
        out = _ssh(job.host, remote, check=False)
        print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hosts", default=None)
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--hop-min", type=int, default=0)
    parser.add_argument("--hop-max", type=int, default=750)
    parser.add_argument("--child-cache-dir", default="{root_dataset_dir}_child_transitions_adv")
    parser.add_argument("--num-processes", type=int, default=64)
    parser.add_argument("--parents-per-task", type=int, default=1000)
    parser.add_argument("--transition-shard-size", type=int, default=4096)
    parser.add_argument("--validate-first-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "status", "stop", "smoke"):
        p = sub.add_parser(name)
        add_args(p)
        if name == "status":
            p.add_argument("--tail-lines", type=int, default=20)
        if name == "smoke":
            p.add_argument("--smoke-host", default="bellman-classical")
            p.add_argument("--smoke-root-dir", required=True)
            p.add_argument("--smoke-output-dir", required=True)
            p.add_argument("--smoke-max-roots", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "smoke":
        job = JobSpec(
            job_id="smoke_00",
            host=args.smoke_host,
            remote_repo=args.remote_repo,
            root_dataset_dir=args.smoke_root_dir,
            child_cache_dir=args.smoke_output_dir,
            num_processes=min(2, int(args.num_processes)),
            parents_per_task=1,
            transition_shard_size=min(512, int(args.transition_shard_size)),
            validate_first_n=2,
            seed=int(args.seed),
        )
        launch_job(job, max_roots=int(args.smoke_max_roots))
        time.sleep(5)
        print_status([job], tail_lines=40)
        return

    jobs = build_jobs(args)
    if args.cmd == "launch":
        for job in jobs:
            print(f"[launch] {job.host} {job.job_id} root={job.root_dataset_dir} child={job.child_cache_dir}", flush=True)
            launch_job(job)
    elif args.cmd == "status":
        print_status(jobs, tail_lines=int(args.tail_lines))
    elif args.cmd == "stop":
        stop_jobs(jobs)


if __name__ == "__main__":
    main()
