"""Fast remote root-count status for multi-server GV3 root generation.

The root generator writes shard-level manifest records. During a run, some
worker outputs may still be under ``_worker_parts`` and not yet merged into the
top-level manifest, so this script intentionally scans all manifest files and
deduplicates by root-id ranges when possible.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_TARGET_ROOTS = 2_250_000
DEFAULT_HOSTS = [
    "bellman-classical",
    "bellman-classical-worker-1",
    "bellman-classical-worker-2",
    "bellman-classical-worker-3",
    "bellman-classical-worker-4",
]


REMOTE_COUNTER = r"""
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
host_label = sys.argv[2]

files = 0
shards = 0
roots = 0
nonzero = 0
zero = 0
root_ranges = set()
fallback_keys = set()
job_dirs = set()

if base.exists():
    for p in base.rglob("manifest*.jsonl"):
        files += 1
        try:
            rel_parts = p.relative_to(base).parts
        except ValueError:
            rel_parts = p.parts
        if rel_parts:
            job_dirs.add(rel_parts[0])
        try:
            fh = p.open()
        except OSError:
            continue
        with fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                n = int(rec.get("num_records", 0) or 0)
                nz = int(rec.get("nonzero_records", 0) or 0)
                z = int(rec.get("zero_records", 0) or 0)
                if n <= 0:
                    continue

                rid_min = rec.get("root_id_min")
                rid_max = rec.get("root_id_max")
                if rid_min is not None and rid_max is not None:
                    key = (int(rid_min), int(rid_max), n)
                    if key in root_ranges:
                        continue
                    root_ranges.add(key)
                else:
                    # Unique enough for worker-part manifests where shard names
                    # repeat under different task dirs.
                    key = (str(p.relative_to(base)), line_no, rec.get("shard_path", ""))
                    if key in fallback_keys:
                        continue
                    fallback_keys.add(key)

                shards += 1
                roots += n
                nonzero += nz
                zero += z

print(json.dumps({
    "host": host_label,
    "base": str(base),
    "exists": base.exists(),
    "job_dirs": len(job_dirs),
    "manifest_files": files,
    "shards": shards,
    "roots": roots,
    "nonzero": nonzero,
    "zero": zero,
    "ratio": (nonzero / roots) if roots else 0.0,
}, sort_keys=True))
"""


@dataclass(frozen=True)
class HostResult:
    host: str
    ok: bool
    data: dict[str, object] | None
    error: str = ""


def parse_hosts(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_HOSTS)
    hosts = [x.strip() for x in raw.split(",") if x.strip()]
    if not hosts:
        raise ValueError("--hosts produced no hosts")
    return hosts


def remote_base(remote_repo: str, experiment_name: str) -> str:
    return (
        f"{remote_repo.rstrip('/')}/simulator_output/GV3_Agent/"
        f"ModelSearchBed/{experiment_name}"
    )


def count_host(host: str, *, remote_repo: str, experiment_name: str, timeout: int) -> HostResult:
    script_b64 = base64.b64encode(REMOTE_COUNTER.encode("utf-8")).decode("ascii")
    base = remote_base(remote_repo, experiment_name)
    remote_cmd = (
        f"python3 -c {shlex.quote('import base64; exec(base64.b64decode(' + repr(script_b64) + ').decode())')} "
        f"{shlex.quote(base)} {shlex.quote(host)}"
    )
    try:
        proc = subprocess.run(
            ["ssh", host, remote_cmd],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return HostResult(host=host, ok=False, data=None, error=f"timeout after {timeout}s: {exc}")

    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return HostResult(host=host, ok=False, data=None, error=out)
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception as exc:
        return HostResult(host=host, ok=False, data=None, error=f"failed to parse output: {exc}; output={out}")
    return HostResult(host=host, ok=True, data=data)


def print_table(results: list[HostResult], *, target_total: int) -> int:
    headers = ["host", "roots", "nonzero", "ratio", "shards", "manifests", "job_dirs", "status"]
    rows: list[list[str]] = []
    total_roots = 0
    total_nonzero = 0
    total_shards = 0
    total_files = 0

    for result in results:
        if not result.ok or result.data is None:
            rows.append([result.host, "-", "-", "-", "-", "-", "-", "ERROR"])
            continue
        data = result.data
        roots = int(data.get("roots", 0) or 0)
        nz = int(data.get("nonzero", 0) or 0)
        shards = int(data.get("shards", 0) or 0)
        files = int(data.get("manifest_files", 0) or 0)
        total_roots += roots
        total_nonzero += nz
        total_shards += shards
        total_files += files
        rows.append(
            [
                str(data.get("host", result.host)),
                f"{roots:,}",
                f"{nz:,}",
                f"{(nz / roots if roots else 0.0):.4f}",
                f"{shards:,}",
                f"{files:,}",
                str(data.get("job_dirs", 0)),
                "OK" if data.get("exists", False) else "MISSING",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.rjust(widths[idx]) if idx else cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))

    overall_ratio = (total_nonzero / total_roots) if total_roots else 0.0
    progress = (total_roots / target_total) if target_total else 0.0
    print()
    print(f"total_roots={total_roots:,} / {target_total:,} ({progress:.2%})")
    print(f"total_nonzero={total_nonzero:,} ratio={overall_ratio:.4f}")
    print(f"total_shards={total_shards:,} manifest_files={total_files:,}")

    errors = [r for r in results if not r.ok]
    if errors:
        print()
        print("errors:")
        for r in errors:
            print(f"- {r.host}: {r.error[:500]}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", default=None, help="Comma-separated SSH hosts. Defaults to the 5 generation workers.")
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_ROOTS)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-parallel", type=int, default=5)
    args = parser.parse_args(argv)

    hosts = parse_hosts(args.hosts)
    max_parallel = max(1, min(int(args.max_parallel), len(hosts)))
    with futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futs = [
            pool.submit(
                count_host,
                host,
                remote_repo=str(args.remote_repo),
                experiment_name=str(args.experiment_name),
                timeout=int(args.timeout),
            )
            for host in hosts
        ]
        results = [f.result() for f in futs]

    results.sort(key=lambda r: hosts.index(r.host))
    return print_table(results, target_total=int(args.target_total))


if __name__ == "__main__":
    raise SystemExit(main())
