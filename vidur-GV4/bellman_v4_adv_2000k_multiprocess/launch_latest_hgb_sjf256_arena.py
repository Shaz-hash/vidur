#!/usr/bin/env python3
"""Launch latest multiserver HGB models in CPP arena vs SJF-256.

This is intentionally a thin launcher around arena_mcts_value_runnerCPP.py.  It
reads latest_manifest.csv and starts one top-level arena launcher per model.
Each top-level launcher then manages its own per-game workers.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path("/home/ubuntu/vidur-classical-search")
DEFAULT_LATEST_ROOT = (
    DEFAULT_REPO_ROOT
    / "simulator_output/GV3_Agent/bellman_multiserver_HGB/latest_version"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_REPO_ROOT
    / "simulator_output/GV3_Agent/Model_Tester_Results/HGB_MultiServer_Games/SJF-256"
)


def _uct_label(value: float) -> str:
    text = f"{value:g}".replace(".", "")
    return f"uct{text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch latest HGB CPP arena games vs trivial SJF-256."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--latest-root", type=Path, default=DEFAULT_LATEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--uct-c", type=float, required=True)
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--num-parallel-games", type=int, default=5)
    parser.add_argument("--game-id-start", type=int, default=12_000_000)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=100)
    parser.add_argument("--history-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arena-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--mcts-iterations", type=int, default=10_000)
    parser.add_argument("--trivial-budget-tokens", type=int, default=256)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    latest_root = args.latest_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest = latest_root / "latest_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"latest manifest not found: {manifest}")

    suffix = args.suffix.strip() or f"{_uct_label(float(args.uct_c))}_rerun1"
    rows: list[dict[str, str]] = []
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("worker") or not row.get("config_name") or not row.get("latest_version"):
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no model rows found in {manifest}")

    output_root.mkdir(parents=True, exist_ok=True)
    launch_csv = output_root / f"launched_latest_models_{suffix}.csv"
    launched: list[dict[str, str | int | float]] = []

    for row in rows:
        worker = str(row["worker"])
        config_name = str(row["config_name"])
        version = int(row["latest_version"])
        model_path = latest_root / worker / config_name / f"Model_Version{version}" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        out_dir = (
            output_root
            / f"{worker}__{config_name}__v{version}"
            / f"sjf256_cpp_visitselect_50games_iter10000_p5_{suffix}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "launcher.out"

        cmd = [
            str(repo_root / ".venv/bin/python3"),
            "-m",
            "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP",
            "--model-path",
            str(model_path),
            "--model-version",
            str(version),
            "--output-dir",
            str(out_dir),
            "--game-id-start",
            str(int(args.game_id_start)),
            "--num-games",
            str(int(args.num_games)),
            "--num-parallel-games",
            str(int(args.num_parallel_games)),
            "--shared-root-mcts-iterations",
            str(int(args.mcts_iterations)),
            "--worker-threads",
            "1",
            "--trivial-budget-tokens",
            str(int(args.trivial_budget_tokens)),
            "--arena-time-limit-sec",
            str(float(args.arena_time_limit_sec)),
            "--arena-max-total-turns",
            "4096",
            "--arena-max-controller-cleanup-steps",
            "1024",
            "--history-hops-min",
            str(int(args.history_hops_min)),
            "--history-hops-max",
            str(int(args.history_hops_max)),
            "--history-hops-unique",
            "--history-hops-force-zero",
            "--history-seed",
            str(int(args.history_seed)),
            "--seed",
            str(int(args.seed)),
            "--uct-c",
            str(float(args.uct_c)),
        ]

        pid = -1
        if args.dry_run:
            print(" ".join(cmd))
        else:
            log_f = open(log_path, "ab")
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pid = int(proc.pid)
            print(f"launched pid={pid} {worker}/{config_name} v{version} -> {out_dir}", flush=True)

        launched.append(
            {
                "worker": worker,
                "config_name": config_name,
                "version": version,
                "uct_c": float(args.uct_c),
                "pid": pid,
                "model_path": str(model_path),
                "output_dir": str(out_dir),
                "log_path": str(log_path),
            }
        )

    with open(launch_csv, "w", newline="") as f:
        fieldnames = [
            "worker",
            "config_name",
            "version",
            "uct_c",
            "pid",
            "model_path",
            "output_dir",
            "log_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(launched)
    print(f"wrote {launch_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
