"""Coordinate adversary parent/state feature building for workers 5-8.

This builds the 226D state features for adversary roots generated on workers
5-8, then assembles those per-worker shards onto worker 5 for adversary policy
training.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon"
DEFAULT_TARGET_HOST = "bellman-classical-worker-5"
DEFAULT_ASSEMBLY_DIR = "{base}/{experiment_name}/assembled_parent_child_features_adv"
DEFAULT_FEATURE_DIR = "{root_dataset_dir}_parent_features_adv"
STATE_FEATURE_DIM = 226


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    server_index: int
    hops_min: int
    hops_max: int

    @property
    def job_id(self) -> str:
        return f"server_{self.server_index:02d}"

    @property
    def root_dir_name(self) -> str:
        return f"{self.job_id}_{self.host}_hops_{self.hops_min}_{self.hops_max}"


WORKERS: tuple[WorkerSpec, ...] = (
    WorkerSpec("worker5", "bellman-classical-worker-5", 0, 0, 100),
    WorkerSpec("worker6", "bellman-classical-worker-6", 1, 100, 200),
    WorkerSpec("worker7", "bellman-classical-worker-7", 2, 200, 300),
    WorkerSpec("worker8", "bellman-classical-worker-8", 3, 300, 400),
)


@dataclass(frozen=True)
class JobSpec:
    worker_id: str
    job_id: str
    host: str
    remote_repo: str
    root_dataset_dir: str
    feature_dir: str
    num_processes: int
    limit_shards: int = 0


def _run(cmd: list[str], *, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _ssh(host: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, cmd], check=check)


def _rsync(src: str, dst: str) -> subprocess.CompletedProcess[str]:
    return _run(["rsync", "-az", src, dst])


def _base(remote_repo: str, remote_output_base: str) -> str:
    return f"{str(remote_repo).rstrip('/')}/{str(remote_output_base).strip('/')}"


def _assembly_dir(args: argparse.Namespace) -> str:
    base = _base(args.remote_repo, args.remote_output_base)
    return str(args.assembly_dir).format(
        base=base,
        experiment_name=args.experiment_name,
        remote_repo=str(args.remote_repo).rstrip("/"),
    )


def parse_workers(raw: str | None) -> list[WorkerSpec]:
    if raw is None or not str(raw).strip():
        return list(WORKERS)
    wanted = {x.strip() for x in str(raw).split(",") if x.strip()}
    selected = [
        w
        for w in WORKERS
        if w.worker_id in wanted or w.host in wanted or w.job_id in wanted or str(w.server_index) in wanted
    ]
    if not selected:
        raise SystemExit(f"no adversary workers selected from {raw!r}")
    return selected


def build_jobs(args: argparse.Namespace) -> list[JobSpec]:
    base = _base(args.remote_repo, args.remote_output_base)
    jobs: list[JobSpec] = []
    for worker in parse_workers(args.workers):
        root_dir = f"{base}/{args.experiment_name}/{worker.root_dir_name}"
        feat_dir = str(args.feature_dir).format(
            root_dataset_dir=root_dir,
            suffix=worker.root_dir_name,
            host=worker.host,
            worker_id=worker.worker_id,
            job_id=worker.job_id,
        )
        jobs.append(
            JobSpec(
                worker_id=worker.worker_id,
                job_id=worker.job_id,
                host=worker.host,
                remote_repo=args.remote_repo,
                root_dataset_dir=root_dir,
                feature_dir=feat_dir,
                num_processes=args.num_processes,
                limit_shards=args.limit_shards,
            )
        )
    return jobs


def command_for(job: JobSpec) -> list[str]:
    cmd = [
        f"{job.remote_repo.rstrip('/')}/.venv/bin/python3",
        "-m",
        "vidur.bellman_v4_adv.build_state_local_features_adv",
        "--manifest",
        f"{job.root_dataset_dir}/manifest.jsonl",
        "--shard-dir",
        job.root_dataset_dir,
        "--out-features",
        f"{job.feature_dir}/parent_features.npy",
        "--out-targets",
        f"{job.feature_dir}/parent_targets.npy",
        "--out-meta",
        f"{job.feature_dir}/parent_features.meta.json",
        "--root-player-filter",
        "adversary",
        "--num-processes",
        str(job.num_processes),
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
        f"cd {shlex.quote(job.remote_repo)}; "
        f"rm -rf {out_dir}; mkdir -p {out_dir}; "
        f"printf '%s\\n' {payload} > {out_dir}/job_config.json; "
        f"printf '%s\\n' {shlex.quote(cmd)} > {out_dir}/command.txt; "
        f"nohup {cmd} > {out_dir}/parent_feature_build.log 2>&1 < /dev/null & "
        f"echo $! > {out_dir}/parent_feature_build.pid; "
        f"echo launched worker={job.worker_id} pid=$(cat {out_dir}/parent_feature_build.pid) out={out_dir}"
    )
    out = _ssh(job.host, remote)
    print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def status_job(job: JobSpec, tail_lines: int) -> dict[str, Any]:
    out_dir = shlex.quote(job.feature_dir)
    python = shlex.quote(job.remote_repo.rstrip("/") + "/.venv/bin/python3")
    remote = (
        f"OUT={out_dir}; PID=$(cat \"$OUT/parent_feature_build.pid\" 2>/dev/null || true); "
        "RUNNING=0; if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then RUNNING=1; fi; "
        "if [ -f \"$OUT/parent_features.meta.json\" ]; then "
        f"{python} - <<'PY' \"$OUT/parent_features.meta.json\"\n"
        "import json,sys\n"
        "d=json.load(open(sys.argv[1]));\n"
        "print('SUMMARY_ROWS=%s' % d.get('num_records',''))\n"
        "print('SUMMARY_DIM=%s' % d.get('feature_dim',''))\n"
        "PY\n"
        "fi; "
        "echo STATUS_PID=$PID; echo STATUS_RUNNING=$RUNNING; echo STATUS_OUT=$OUT; echo STATUS_LOG_TAIL_BEGIN; "
        f"tail -n {int(tail_lines)} \"$OUT/parent_feature_build.log\" 2>/dev/null || true; "
        "echo STATUS_LOG_TAIL_END"
    )
    res = _ssh(job.host, remote, check=False)
    row: dict[str, Any] = {
        "worker_id": job.worker_id,
        "job_id": job.job_id,
        "host": job.host,
        "output_dir": job.feature_dir,
        "raw": res.stdout,
    }
    for line in res.stdout.splitlines():
        if line.startswith("STATUS_PID="):
            row["pid"] = line.split("=", 1)[1]
        elif line.startswith("STATUS_RUNNING="):
            row["running"] = line.split("=", 1)[1] == "1"
        elif line.startswith("SUMMARY_ROWS="):
            row["rows"] = line.split("=", 1)[1]
        elif line.startswith("SUMMARY_DIM="):
            row["dim"] = line.split("=", 1)[1]
    return row


def print_status(jobs: list[JobSpec], tail_lines: int) -> None:
    rows = [status_job(j, tail_lines) for j in jobs]
    for row in rows:
        print("=" * 100)
        print(
            f"{row['host']} {row['worker_id']} {row['job_id']} "
            f"running={row.get('running')} pid={row.get('pid','')} "
            f"rows={row.get('rows','')} dim={row.get('dim','')}"
        )
        print(str(row.get("raw", "")).split("STATUS_LOG_TAIL_BEGIN", 1)[-1].split("STATUS_LOG_TAIL_END", 1)[0])
    print("=" * 100)
    w = csv.DictWriter(sys.stdout, fieldnames=["worker_id", "job_id", "host", "running", "pid", "rows", "dim", "output_dir"])
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in w.fieldnames})


def stop_jobs(jobs: list[JobSpec]) -> None:
    for job in jobs:
        out_dir = shlex.quote(job.feature_dir)
        out = _ssh(
            job.host,
            (
                f"OUT={out_dir}; PID=$(cat \"$OUT/parent_feature_build.pid\" 2>/dev/null || true); "
                "if [ -n \"$PID\" ]; then kill -TERM \"$PID\" 2>/dev/null || true; fi; "
                "pkill -TERM -f \"$OUT\" 2>/dev/null || true; "
                f"echo stopped worker={job.worker_id} out=$OUT pid=$PID"
            ),
            check=False,
        )
        print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def _remote_meta(host: str, remote_repo: str, feature_dir: str) -> dict[str, Any]:
    python = shlex.quote(remote_repo.rstrip("/") + "/.venv/bin/python3")
    meta = shlex.quote(f"{feature_dir.rstrip('/')}/parent_features.meta.json")
    remote = (
        f"{python} - <<'PY' {meta}\n"
        "import json,sys\n"
        "d=json.load(open(sys.argv[1]));\n"
        "print(json.dumps({'rows': int(d.get('num_records', 0)), 'dim': int(d.get('feature_dim', 0))}))\n"
        "PY"
    )
    out = _ssh(host, remote)
    return json.loads(out.stdout.strip().splitlines()[-1])


def sync_assembly(jobs: list[JobSpec], args: argparse.Namespace) -> None:
    target_host = str(args.target_host)
    assembly_dir = _assembly_dir(args).rstrip("/")
    entries: list[dict[str, Any]] = []
    _ssh(target_host, f"mkdir -p {shlex.quote(assembly_dir + '/parent')}")

    with tempfile.TemporaryDirectory(prefix="adv_parent_features_assemble_") as tmp_s:
        tmp = Path(tmp_s)
        for job in jobs:
            dest_dir = f"{assembly_dir}/parent/{job.worker_id}_{job.job_id}"
            if job.host == target_host:
                remote = (
                    f"mkdir -p {shlex.quote(dest_dir)}; "
                    f"rsync -az {shlex.quote(job.feature_dir.rstrip('/') + '/')} {shlex.quote(dest_dir.rstrip('/') + '/')}"
                )
                _ssh(target_host, remote)
            else:
                local_dir = tmp / f"{job.worker_id}_{job.job_id}"
                local_dir.mkdir(parents=True, exist_ok=True)
                _rsync(f"{job.host}:{job.feature_dir.rstrip('/')}/", str(local_dir) + "/")
                _rsync(str(local_dir) + "/", f"{target_host}:{dest_dir.rstrip('/')}/")

            meta = _remote_meta(target_host, job.remote_repo, dest_dir)
            rows = int(meta.get("rows", 0))
            dim = int(meta.get("dim", 0))
            entries.append(
                {
                    "worker_id": job.worker_id,
                    "job_id": job.job_id,
                    "source_host": job.host,
                    "source_dir": job.feature_dir,
                    "local_dir": dest_dir,
                    "rows": rows,
                    "dim": dim,
                    "valid": bool(rows > 0 and dim == STATE_FEATURE_DIM),
                }
            )

        manifest = {
            "experiment_name": args.experiment_name,
            "target_host": target_host,
            "assembly_dir": assembly_dir,
            "root_player_filter": "adversary",
            "feature_dim": STATE_FEATURE_DIM,
            "parent": entries,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest_path = tmp / "assembly_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _rsync(str(manifest_path), f"{target_host}:{assembly_dir}/assembly_manifest.json")

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def status_assembly(args: argparse.Namespace) -> None:
    target_host = str(args.target_host)
    assembly_dir = _assembly_dir(args).rstrip("/")
    remote = (
        f"if [ ! -f {shlex.quote(assembly_dir + '/assembly_manifest.json')} ]; then "
        f"echo missing {shlex.quote(assembly_dir + '/assembly_manifest.json')}; exit 0; fi; "
        f"cat {shlex.quote(assembly_dir + '/assembly_manifest.json')}"
    )
    out = _ssh(target_host, remote, check=False)
    print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workers", default=None, help="Comma-separated worker ids/hosts/job ids. Defaults to worker5-worker8.")
    p.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    p.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    p.add_argument("--num-processes", type=int, default=64)
    p.add_argument("--limit-shards", type=int, default=0)


def add_assembly_args(p: argparse.ArgumentParser) -> None:
    add_args(p)
    p.add_argument("--target-host", default=DEFAULT_TARGET_HOST)
    p.add_argument("--assembly-dir", default=DEFAULT_ASSEMBLY_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "status", "stop"):
        p = sub.add_parser(name)
        add_args(p)
        if name == "status":
            p.add_argument("--tail-lines", type=int, default=20)
    p_sync = sub.add_parser("sync-assembly")
    add_assembly_args(p_sync)
    p_status = sub.add_parser("status-assembly")
    add_assembly_args(p_status)

    args = parser.parse_args()
    jobs = build_jobs(args)
    if args.cmd == "launch":
        for j in jobs:
            launch_job(j)
    elif args.cmd == "status":
        print_status(jobs, args.tail_lines)
    elif args.cmd == "stop":
        stop_jobs(jobs)
    elif args.cmd == "sync-assembly":
        sync_assembly(jobs, args)
    elif args.cmd == "status-assembly":
        status_assembly(args)


if __name__ == "__main__":
    main()
