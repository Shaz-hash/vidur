#!/usr/bin/env python3
"""Fast remote status for MCTS value-function policy target generation.

Counts durable progress from each worker's ``partials/done_batch_*.json`` files.
This is intentionally lightweight: it does not read the large target CSVs unless
needed, because every completed batch writes a small JSON summary with state,
row, and error counts.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import shlex
import subprocess
from dataclasses import dataclass


DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_MODEL_LABEL = "worker1_200k_alpha2__hgb_sq_63leaf_1050iter_a2__v47__"
DEFAULT_MCTS_ITERATIONS = 10_000
DEFAULT_TARGET_DIR_SUFFIX = "effectcanon_full"
DEFAULT_RUN_SUFFIX = ""
DEFAULT_WORKERS = {
    "worker1": "bellman-classical-worker-1",
    "worker2": "bellman-classical-worker-2",
    "worker3": "bellman-classical-worker-3",
    "worker4": "bellman-classical-worker-4",
}


REMOTE_COUNTER = r"""
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
worker = sys.argv[2]
host = sys.argv[3]

manifest_path = out_dir / "run_manifest.json"
final_manifest_path = out_dir / "manifest.json"
partials = out_dir / "partials"

run_manifest = {}
if manifest_path.exists():
    try:
        run_manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        run_manifest = {"_parse_error": repr(exc)}

final_manifest = {}
if final_manifest_path.exists():
    try:
        final_manifest = json.loads(final_manifest_path.read_text())
    except Exception as exc:
        final_manifest = {"_parse_error": repr(exc)}

completed_batches = 0
states_done = 0
states_success = 0
states_failed = 0
target_rows = 0
time_rows = 0
batch_errors = 0
min_batch = None
max_batch = None
latest_mtime = 0.0

if partials.exists():
    for p in partials.glob("done_batch_*.json"):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            batch_errors += 1
            continue
        completed_batches += 1
        idx = int(rec.get("batch_index", -1) or -1)
        min_batch = idx if min_batch is None else min(min_batch, idx)
        max_batch = idx if max_batch is None else max(max_batch, idx)
        states = int(rec.get("num_states", 0) or 0)
        success = int(rec.get("num_success", 0) or 0)
        errors = int(rec.get("num_errors", 0) or 0)
        states_done += states
        states_success += success
        states_failed += errors
        target_rows += int(rec.get("num_target_rows", 0) or 0)
        time_rows += success
        try:
            latest_mtime = max(latest_mtime, p.stat().st_mtime)
        except OSError:
            pass

error_jsonl_files = 0
error_jsonl_rows = 0
if partials.exists():
    for p in partials.glob("errors_batch_*.jsonl"):
        error_jsonl_files += 1
        try:
            with p.open() as f:
                error_jsonl_rows += sum(1 for _ in f)
        except OSError:
            pass

target_files = len(list(partials.glob("targets_batch_*.csv"))) if partials.exists() else 0
time_files = len(list(partials.glob("times_batch_*.csv"))) if partials.exists() else 0

total_batches = int(run_manifest.get("num_batches", 0) or 0)
total_states = int(run_manifest.get("total_available_states", 0) or 0)
batch_size = int(run_manifest.get("batch_size", 0) or 0)
num_processes = int(run_manifest.get("num_processes", 0) or 0)
mcts_iterations = int(run_manifest.get("mcts_iterations", 0) or 0)
uct_c = float(run_manifest.get("uct_c", 0.0) or 0.0)

print(json.dumps({
    "worker": worker,
    "host": host,
    "output_dir": str(out_dir),
    "exists": out_dir.exists(),
    "run_manifest_exists": manifest_path.exists(),
    "final_manifest_exists": final_manifest_path.exists(),
    "targets_csv_exists": (out_dir / "targets.csv").exists(),
    "target_time_csv_exists": (out_dir / "target_time.csv").exists(),
    "total_batches": total_batches,
    "completed_batches": completed_batches,
    "batch_progress": (completed_batches / total_batches) if total_batches else 0.0,
    "total_states": total_states,
    "states_done": states_done,
    "states_success": states_success,
    "states_failed": states_failed,
    "state_progress": (states_success / total_states) if total_states else 0.0,
    "target_rows": target_rows,
    "time_rows": time_rows,
    "target_files": target_files,
    "time_files": time_files,
    "batch_json_parse_errors": batch_errors,
    "error_jsonl_files": error_jsonl_files,
    "error_jsonl_rows": error_jsonl_rows,
    "min_batch": min_batch,
    "max_batch": max_batch,
    "latest_done_mtime": latest_mtime,
    "batch_size": batch_size,
    "num_processes": num_processes,
    "mcts_iterations": mcts_iterations,
    "uct_c": uct_c,
    "final_num_target_rows": int(final_manifest.get("num_target_rows", 0) or 0),
    "final_num_time_rows": int(final_manifest.get("num_time_rows", 0) or 0),
}, sort_keys=True))
"""


@dataclass(frozen=True)
class WorkerResult:
    worker: str
    host: str
    ok: bool
    data: dict[str, object] | None = None
    error: str = ""


def parse_workers(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return list(DEFAULT_WORKERS.items())
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            worker, host = [x.strip() for x in item.split("=", 1)]
        else:
            worker = item
            host = DEFAULT_WORKERS.get(worker)
            if host is None:
                raise ValueError(f"unknown worker {worker!r}; use worker=host for custom hosts")
        out.append((worker, host))
    if not out:
        raise ValueError("--workers produced no workers")
    return out


def target_base(
    remote_repo: str,
    experiment_name: str,
    model_label: str,
    mcts_iterations: int,
    target_dir_suffix: str = "",
) -> str:
    suffix = str(target_dir_suffix).strip()
    iter_leaf = f"{model_label}_iter{int(mcts_iterations)}"
    if suffix:
        iter_leaf = f"{iter_leaf}_{suffix}"
    return (
        f"{remote_repo.rstrip('/')}/simulator_output/GV3_Agent/ModelSearchBed/"
        f"{experiment_name}/mcts_value_function_visit_targets/"
        f"{iter_leaf}"
    )


def worker_output_dir(
    *,
    remote_repo: str,
    experiment_name: str,
    model_label: str,
    mcts_iterations: int,
    target_dir_suffix: str,
    worker: str,
    run_suffix: str,
) -> str:
    suffix = str(run_suffix).strip()
    leaf = worker if not suffix else f"{worker}_{suffix}"
    return f"{target_base(remote_repo, experiment_name, model_label, mcts_iterations, target_dir_suffix)}/{leaf}"


def count_worker(
    worker: str,
    host: str,
    *,
    remote_repo: str,
    experiment_name: str,
    model_label: str,
    mcts_iterations: int,
    target_dir_suffix: str,
    run_suffix: str,
    timeout: int,
) -> WorkerResult:
    script_b64 = base64.b64encode(REMOTE_COUNTER.encode("utf-8")).decode("ascii")
    out_dir = worker_output_dir(
        remote_repo=remote_repo,
        experiment_name=experiment_name,
        model_label=model_label,
        mcts_iterations=mcts_iterations,
        target_dir_suffix=target_dir_suffix,
        worker=worker,
        run_suffix=run_suffix,
    )
    remote_cmd = (
        f"python3 -c {shlex.quote('import base64; exec(base64.b64decode(' + repr(script_b64) + ').decode())')} "
        f"{shlex.quote(out_dir)} {shlex.quote(worker)} {shlex.quote(host)}"
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
        return WorkerResult(worker=worker, host=host, ok=False, error=f"timeout after {timeout}s: {exc}")

    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return WorkerResult(worker=worker, host=host, ok=False, error=out)
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception as exc:
        return WorkerResult(worker=worker, host=host, ok=False, error=f"failed to parse output: {exc}; output={out}")
    return WorkerResult(worker=worker, host=host, ok=True, data=data)


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "-"


def print_table(results: list[WorkerResult], *, json_output: bool = False) -> int:
    if json_output:
        print(json.dumps([r.data if r.ok else {"worker": r.worker, "host": r.host, "error": r.error} for r in results], indent=2, sort_keys=True))
        return 0 if all(r.ok for r in results) else 1

    headers = [
        "worker",
        "host",
        "states",
        "state%",
        "batches",
        "batch%",
        "target_rows",
        "errors",
        "files",
        "procs",
        "status",
    ]
    rows: list[list[str]] = []
    total_states = 0
    total_done = 0
    total_targets = 0
    total_errors = 0
    total_batches = 0
    total_batches_done = 0

    for result in results:
        if not result.ok or result.data is None:
            rows.append([result.worker, result.host, "-", "-", "-", "-", "-", "-", "-", "-", "ERROR"])
            continue
        data = result.data
        states_done = int(data.get("states_success", 0) or 0)
        states_total = int(data.get("total_states", 0) or 0)
        batches_done = int(data.get("completed_batches", 0) or 0)
        batches_total = int(data.get("total_batches", 0) or 0)
        target_rows = int(data.get("target_rows", 0) or 0)
        errors = int(data.get("states_failed", 0) or 0) + int(data.get("error_jsonl_rows", 0) or 0) + int(data.get("batch_json_parse_errors", 0) or 0)
        final = bool(data.get("final_manifest_exists")) and bool(data.get("targets_csv_exists"))
        exists = bool(data.get("exists"))
        status = "DONE" if final else ("RUNNING" if exists and batches_done < batches_total else "PARTIAL" if exists else "MISSING")
        total_states += states_total
        total_done += states_done
        total_targets += target_rows
        total_errors += errors
        total_batches += batches_total
        total_batches_done += batches_done
        rows.append(
            [
                str(data.get("worker", result.worker)),
                str(data.get("host", result.host)),
                f"{states_done:,}/{states_total:,}",
                _fmt_pct(data.get("state_progress", 0.0)),
                f"{batches_done:,}/{batches_total:,}",
                _fmt_pct(data.get("batch_progress", 0.0)),
                f"{target_rows:,}",
                f"{errors:,}",
                f"{int(data.get('target_files', 0) or 0):,}/{int(data.get('time_files', 0) or 0):,}",
                str(data.get("num_processes", "-")),
                status,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.rjust(widths[idx]) if idx not in (0, 1, 10) else cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))

    state_progress = (total_done / total_states) if total_states else 0.0
    batch_progress = (total_batches_done / total_batches) if total_batches else 0.0
    print()
    print(f"total_states={total_done:,}/{total_states:,} ({state_progress:.2%})")
    print(f"total_batches={total_batches_done:,}/{total_batches:,} ({batch_progress:.2%})")
    print(f"total_target_rows={total_targets:,} total_errors={total_errors:,}")

    errors = [r for r in results if not r.ok]
    if errors:
        print()
        print("errors:")
        for r in errors:
            print(f"- {r.worker}/{r.host}: {r.error[:500]}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default=None, help="Comma-separated workers. Use worker or worker=host. Defaults to worker1..worker4.")
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--mcts-iterations", type=int, default=DEFAULT_MCTS_ITERATIONS)
    parser.add_argument(
        "--target-dir-suffix",
        default=DEFAULT_TARGET_DIR_SUFFIX,
        help="Suffix after {model_label}_iter{mcts_iterations}, e.g. effectcanon_full. Empty string means no suffix.",
    )
    parser.add_argument("--run-suffix", default=DEFAULT_RUN_SUFFIX, help="Suffix after worker id, e.g. fixed_rerun1. Empty string means plain worker dirs.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    workers = parse_workers(args.workers)
    max_parallel = max(1, min(int(args.max_parallel), len(workers)))
    with futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futs = [
            pool.submit(
                count_worker,
                worker,
                host,
                remote_repo=str(args.remote_repo),
                experiment_name=str(args.experiment_name),
                model_label=str(args.model_label),
                mcts_iterations=int(args.mcts_iterations),
                target_dir_suffix=str(args.target_dir_suffix),
                run_suffix=str(args.run_suffix),
                timeout=int(args.timeout),
            )
            for worker, host in workers
        ]
        results = [f.result() for f in futs]

    order = {worker: idx for idx, (worker, _host) in enumerate(workers)}
    results.sort(key=lambda r: order.get(r.worker, 999))
    return print_table(results, json_output=bool(args.json))


if __name__ == "__main__":
    raise SystemExit(main())
