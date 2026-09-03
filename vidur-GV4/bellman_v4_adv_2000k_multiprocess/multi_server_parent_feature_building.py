"""Coordinate per-server parent feature building for distributed GV3 roots."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_HOSTS = ["bellman-classical", "bellman-classical-worker-1", "bellman-classical-worker-2", "bellman-classical-worker-3", "bellman-classical-worker-4"]


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    host: str
    remote_repo: str
    root_dataset_dir: str
    feature_dir: str
    num_processes: int
    limit_shards: int = 0


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _ssh(host: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, cmd], check=check)


def parse_hosts(raw: str | None) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else list(DEFAULT_HOSTS)


def split_hop_intervals(hop_min: int, hop_max: int, parts: int) -> list[tuple[int, int]]:
    total = hop_max - hop_min + 1
    out = []
    cur = hop_min
    for i in range(parts):
        width = total // parts + (1 if i < total % parts else 0)
        out.append((cur, cur + width - 1))
        cur += width
    return out


def build_jobs(args: argparse.Namespace) -> list[JobSpec]:
    hosts = parse_hosts(args.hosts)
    base = f"{str(args.remote_repo).rstrip('/')}/{str(args.remote_output_base).strip('/')}"
    jobs = []
    for idx, (host, (hmin, hmax)) in enumerate(zip(hosts, split_hop_intervals(args.hop_min, args.hop_max, len(hosts)))):
        suffix = f"server_{idx:02d}_{host}_hops_{hmin}_{hmax}"
        root_dir = f"{base}/{args.experiment_name}/{suffix}"
        feat_dir = str(args.feature_dir).format(root_dataset_dir=root_dir, suffix=suffix, host=host, job_id=f"server_{idx:02d}")
        jobs.append(JobSpec(f"server_{idx:02d}", host, args.remote_repo, root_dir, feat_dir, args.num_processes, args.limit_shards))
    return jobs


def command_for(job: JobSpec) -> list[str]:
    cmd = [
        f"{job.remote_repo.rstrip('/')}/.venv/bin/python3",
        "-m", "vidur.bellman_v4_adv.build_state_local_features_adv",
        "--manifest", f"{job.root_dataset_dir}/manifest.jsonl",
        "--shard-dir", job.root_dataset_dir,
        "--out-features", f"{job.feature_dir}/parent_features.npy",
        "--out-targets", f"{job.feature_dir}/parent_targets.npy",
        "--out-meta", f"{job.feature_dir}/parent_features.meta.json",
        "--root-player-filter", "controller",
        "--num-processes", str(job.num_processes),
    ]
    if int(job.limit_shards) > 0:
        cmd.extend(["--limit-shards", str(job.limit_shards)])
    return cmd


def launch_job(job: JobSpec) -> None:
    out_dir = shlex.quote(job.feature_dir)
    cmd = " ".join(shlex.quote(x) for x in command_for(job))
    payload = shlex.quote(json.dumps(asdict(job), indent=2, sort_keys=True))
    remote = (
        "set -e; "
        f"cd {shlex.quote(job.remote_repo)}; rm -rf {out_dir}; mkdir -p {out_dir}; "
        f"printf '%s\\n' {payload} > {out_dir}/job_config.json; "
        f"printf '%s\\n' {shlex.quote(cmd)} > {out_dir}/command.txt; "
        f"nohup {cmd} > {out_dir}/parent_feature_build.log 2>&1 < /dev/null & "
        f"echo $! > {out_dir}/parent_feature_build.pid; echo launched pid=$(cat {out_dir}/parent_feature_build.pid) out={out_dir}"
    )
    out = _ssh(job.host, remote)
    print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def status_job(job: JobSpec, tail_lines: int) -> dict[str, object]:
    out_dir = shlex.quote(job.feature_dir)
    remote = (
        f"OUT={out_dir}; PID=$(cat \"$OUT/parent_feature_build.pid\" 2>/dev/null || true); "
        "RUNNING=0; if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then RUNNING=1; fi; "
        "ROWS=''; DIM=''; if [ -f \"$OUT/parent_features.meta.json\" ]; then "
        f"{shlex.quote(job.remote_repo.rstrip('/') + '/.venv/bin/python3')} - <<'PY' \"$OUT/parent_features.meta.json\"\n"
        "import json,sys\nd=json.load(open(sys.argv[1])); print('SUMMARY_ROWS=%s' % d.get('num_records','')); print('SUMMARY_DIM=%s' % d.get('feature_dim',''))\nPY\n"
        "fi; echo STATUS_PID=$PID; echo STATUS_RUNNING=$RUNNING; echo STATUS_OUT=$OUT; echo STATUS_LOG_TAIL_BEGIN; "
        f"tail -n {int(tail_lines)} \"$OUT/parent_feature_build.log\" 2>/dev/null || true; echo STATUS_LOG_TAIL_END"
    )
    res = _ssh(job.host, remote, check=False)
    row = {"job_id": job.job_id, "host": job.host, "output_dir": job.feature_dir, "raw": res.stdout}
    for line in res.stdout.splitlines():
        if line.startswith("STATUS_PID="): row["pid"] = line.split("=",1)[1]
        elif line.startswith("STATUS_RUNNING="): row["running"] = line.split("=",1)[1] == "1"
        elif line.startswith("SUMMARY_ROWS="): row["rows"] = line.split("=",1)[1]
        elif line.startswith("SUMMARY_DIM="): row["dim"] = line.split("=",1)[1]
    return row


def print_status(jobs: list[JobSpec], tail_lines: int) -> None:
    rows = [status_job(j, tail_lines) for j in jobs]
    for row in rows:
        print("="*100)
        print(f"{row['host']} {row['job_id']} running={row.get('running')} pid={row.get('pid','')} rows={row.get('rows','')} dim={row.get('dim','')}")
        print(str(row.get('raw','')).split('STATUS_LOG_TAIL_BEGIN',1)[-1].split('STATUS_LOG_TAIL_END',1)[0])
    print("="*100)
    w = csv.DictWriter(sys.stdout, fieldnames=["job_id","host","running","pid","rows","dim","output_dir"])
    w.writeheader()
    for row in rows: w.writerow({k: row.get(k, "") for k in w.fieldnames})


def stop_jobs(jobs: list[JobSpec]) -> None:
    for job in jobs:
        out_dir = shlex.quote(job.feature_dir)
        out = _ssh(job.host, f"OUT={out_dir}; PID=$(cat \"$OUT/parent_feature_build.pid\" 2>/dev/null || true); if [ -n \"$PID\" ]; then kill -TERM \"$PID\" 2>/dev/null || true; fi; pkill -TERM -f \"$OUT\" 2>/dev/null || true; echo stopped out=$OUT pid=$PID", check=False)
        print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hosts", default=None)
    p.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    p.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--hop-min", type=int, default=0)
    p.add_argument("--hop-max", type=int, default=750)
    p.add_argument("--feature-dir", default="{root_dataset_dir}_parent_features_adv")
    p.add_argument("--num-processes", type=int, default=64)
    p.add_argument("--limit-shards", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "status", "stop"):
        p = sub.add_parser(name)
        add_args(p)
        if name == "status": p.add_argument("--tail-lines", type=int, default=20)
    args = parser.parse_args()
    jobs = build_jobs(args)
    if args.cmd == "launch":
        for j in jobs: launch_job(j)
    elif args.cmd == "status":
        print_status(jobs, args.tail_lines)
    elif args.cmd == "stop":
        stop_jobs(jobs)

if __name__ == "__main__":
    main()
