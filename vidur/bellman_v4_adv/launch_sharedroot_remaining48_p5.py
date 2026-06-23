"""Resume shared-root MCTS arena games with lower concurrency.

This keeps the same game IDs and history hops from the original 50-game
manifest, skips games already completed in the original run, and writes the
remaining games to a separate output directory. Each game still runs in its own
process so process exit returns memory to the OS.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path("/home/ubuntu/vidur-classical-search")
PYTHON = REPO / ".venv/bin/python3"
MODEL = (
    REPO
    / "simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4_adv_226"
    / "V_iter_xl_50/hgb_sq_47leaf_850iter/Model_Version47/v4_adv_hgb_wrapper.joblib"
)
SOURCE_OUT = (
    REPO
    / "simulator_output/GV3_Agent/Model_Tester_Results"
    / "mcts_sharedroot_hgb47_v47_sjf512_50games_hops0_100_seed2026_unique_iter10000_iso10_no_details_canonfix"
)
OUT = (
    REPO
    / "simulator_output/GV3_Agent/Model_Tester_Results"
    / "mcts_sharedroot_hgb47_v47_sjf512_remaining48_hops0_100_seed2026_unique_iter10000_iso5_no_details_canonfix"
)
CONCURRENCY = 5


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def completed_game_ids() -> set[int]:
    completed: set[int] = set()
    for row in _read_csv(SOURCE_OUT / "arena_results.csv"):
        game_id = row.get("game_id")
        if game_id is None or str(game_id).strip() == "":
            continue
        if row.get("cycle1_end_reason") and row.get("cycle2_end_reason"):
            completed.add(int(game_id))
    return completed


def load_remaining_jobs() -> list[dict[str, object]]:
    manifest = _read_csv(SOURCE_OUT / "manifest.csv")
    if not manifest:
        raise FileNotFoundError(SOURCE_OUT / "manifest.csv")

    skip = completed_game_ids()
    jobs: list[dict[str, object]] = []
    for row in manifest:
        game_id = int(row["game_id"])
        if game_id in skip:
            continue
        hop = int(row["history_hops"])
        idx = int(row["index"])
        job_dir = OUT / "jobs" / f"game_{game_id}_hop_{hop}"
        jobs.append(
            {
                "index": idx,
                "game_id": game_id,
                "history_hops": hop,
                "status": "pending",
                "pid": "",
                "returncode": "",
                "elapsed_sec": "",
                "job_dir": str(job_dir),
                "process": None,
                "start_time": None,
            }
        )
    return jobs


def write_manifest(jobs: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "game_id", "history_hops", "job_dir"])
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "index": job["index"],
                    "game_id": job["game_id"],
                    "history_hops": job["history_hops"],
                    "job_dir": job["job_dir"],
                }
            )


def command_for(job: dict[str, object]) -> list[str]:
    hop = int(job["history_hops"])
    gid = int(job["game_id"])
    return [
        str(PYTHON),
        "-m",
        "vidur.bellman_v4_adv.arena_mcts_model_tester",
        "--model-path",
        str(MODEL),
        "--model-version",
        "47",
        "--output-dir",
        str(job["job_dir"]),
        "--game-id-start",
        str(gid),
        "--num-games",
        "1",
        "--mcts-selection-mode",
        "shared_root",
        "--shared-root-mcts-iterations",
        "10000",
        "--trivial-budget-tokens",
        "512",
        "--arena-time-limit-sec",
        "5.0",
        "--history-hops-min",
        str(hop),
        "--history-hops-max",
        str(hop),
        "--no-history-hops-force-zero",
        "--worker-threads",
        "1",
        "--seed",
        "2026",
    ]


def write_status(jobs: list[dict[str, object]]) -> None:
    fields = ["index", "game_id", "history_hops", "status", "pid", "returncode", "elapsed_sec", "job_dir"]
    with (OUT / "launcher_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({k: job.get(k, "") for k in fields})


def collect_finished(job: dict[str, object]) -> None:
    job_dir = Path(str(job["job_dir"]))
    parent_games = OUT / "arena_games"
    parent_games.mkdir(parents=True, exist_ok=True)
    for src in sorted((job_dir / "arena_games").glob("*.csv")):
        if src.name.endswith("_model_action_details.csv"):
            continue
        shutil.copy2(src, parent_games / src.name)


def merge_results() -> None:
    result_files = sorted((OUT / "jobs").glob("game_*_hop_*/arena_results.csv"))
    if not result_files:
        return
    out_path = OUT / "arena_results.csv"
    with out_path.open("w", newline="") as fout:
        writer = None
        merged = 0
        for path in result_files:
            with path.open(newline="") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=list(reader.fieldnames or []))
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    merged += 1
    print(f"[launcher] merged rows={merged} files={len(result_files)} -> {out_path}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs").mkdir(parents=True, exist_ok=True)
    (OUT / "arena_games").mkdir(parents=True, exist_ok=True)

    jobs = load_remaining_jobs()
    write_manifest(jobs)
    print(f"[launcher] source={SOURCE_OUT}", flush=True)
    print(f"[launcher] out={OUT}", flush=True)
    print(f"[launcher] jobs={len(jobs)} concurrency={CONCURRENCY}", flush=True)

    next_idx = 0
    running: list[dict[str, object]] = []
    completed = 0
    failed = 0
    start_all = time.time()

    while completed + failed < len(jobs):
        while next_idx < len(jobs) and len(running) < CONCURRENCY:
            job = jobs[next_idx]
            job_dir = Path(str(job["job_dir"]))
            job_dir.mkdir(parents=True, exist_ok=True)
            log_f = (job_dir / "run.log").open("w")
            proc = subprocess.Popen(
                command_for(job),
                cwd=str(REPO),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            job["process"] = proc
            job["log_f"] = log_f
            job["pid"] = proc.pid
            job["status"] = "running"
            job["start_time"] = time.time()
            running.append(job)
            print(
                "[launcher] started idx={} game={} hop={} pid={}".format(
                    job["index"], job["game_id"], job["history_hops"], proc.pid
                ),
                flush=True,
            )
            next_idx += 1

        time.sleep(10)
        still_running: list[dict[str, object]] = []
        for job in running:
            proc = job["process"]
            if proc is None:
                continue
            rc = proc.poll()
            if rc is None:
                job["elapsed_sec"] = int(time.time() - float(job["start_time"]))
                still_running.append(job)
                continue

            job["returncode"] = int(rc)
            job["elapsed_sec"] = int(time.time() - float(job["start_time"]))
            try:
                job["log_f"].close()
            except Exception:
                pass

            if int(rc) == 0:
                job["status"] = "done"
                completed += 1
                collect_finished(job)
                print(
                    "[launcher] done idx={} game={} hop={} elapsed={}s".format(
                        job["index"], job["game_id"], job["history_hops"], job["elapsed_sec"]
                    ),
                    flush=True,
                )
            else:
                job["status"] = "failed"
                failed += 1
                print(
                    "[launcher] FAILED idx={} game={} hop={} rc={}".format(
                        job["index"], job["game_id"], job["history_hops"], rc
                    ),
                    flush=True,
                )

        running = still_running
        write_status(jobs)
        merge_results()
        print(
            "[launcher] progress done={} failed={} running={} pending={} elapsed={}s".format(
                completed, failed, len(running), len(jobs) - next_idx, int(time.time() - start_all)
            ),
            flush=True,
        )

    write_status(jobs)
    merge_results()
    print(f"[launcher] complete done={completed} failed={failed} out={OUT}", flush=True)


if __name__ == "__main__":
    main()
