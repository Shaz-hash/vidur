"""Launch shared-root MCTS arena games as isolated one-game processes.

Each game runs in its own Python process so process exit returns simulator,
model, sklearn, and MCTS allocator memory to the OS. The launcher keeps a fixed
number of games active and merges per-game arena_results.csv files.
"""

from __future__ import annotations

import csv
import random
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
OUT = (
    REPO
    / "simulator_output/GV3_Agent/Model_Tester_Results"
    / "mcts_sharedroot_hgb47_v47_sjf512_50games_hops0_100_seed2026_unique_iter10000_iso10_no_details_canonfix"
)
GAME_ID_START = 12_000_000
NUM_GAMES = 50
CONCURRENCY = 10
HOP_MIN = 0
HOP_MAX = 100
HISTORY_SEED = 2026


def sample_hops() -> list[int]:
    rng = random.Random(HISTORY_SEED)
    hops = rng.sample(list(range(HOP_MIN, HOP_MAX + 1)), k=NUM_GAMES)
    if 0 in hops:
        zero_idx = hops.index(0)
        hops[0], hops[zero_idx] = hops[zero_idx], hops[0]
    else:
        hops[0] = 0
    return [int(x) for x in hops]


def write_manifest(hops: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "game_id", "history_hops", "job_dir"])
        writer.writeheader()
        for i, hop in enumerate(hops):
            gid = GAME_ID_START + i
            writer.writerow(
                {
                    "index": i,
                    "game_id": gid,
                    "history_hops": hop,
                    "job_dir": str(OUT / "jobs" / f"game_{gid}_hop_{hop}"),
                }
            )


def command_for(i: int, hop: int, job_dir: Path) -> list[str]:
    gid = GAME_ID_START + i
    return [
        str(PYTHON),
        "-m",
        "vidur.bellman_v4_adv.arena_mcts_model_tester",
        "--model-path",
        str(MODEL),
        "--model-version",
        "47",
        "--output-dir",
        str(job_dir),
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


def write_status(rows: list[dict]) -> None:
    fields = ["index", "game_id", "history_hops", "status", "pid", "returncode", "elapsed_sec", "job_dir"]
    with (OUT / "launcher_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def collect_finished(job: dict) -> None:
    job_dir = Path(job["job_dir"])
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
    hops = sample_hops()
    write_manifest(hops)

    jobs: list[dict] = []
    for i, hop in enumerate(hops):
        gid = GAME_ID_START + i
        job_dir = OUT / "jobs" / f"game_{gid}_hop_{hop}"
        job_dir.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "index": i,
                "game_id": gid,
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

    next_idx = 0
    running: list[dict] = []
    completed = 0
    failed = 0
    start_all = time.time()
    while completed + failed < len(jobs):
        while next_idx < len(jobs) and len(running) < CONCURRENCY:
            job = jobs[next_idx]
            job_dir = Path(job["job_dir"])
            log_f = (job_dir / "run.log").open("w")
            proc = subprocess.Popen(
                command_for(int(job["index"]), int(job["history_hops"]), job_dir),
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
        still_running: list[dict] = []
        for job in running:
            proc = job["process"]
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
