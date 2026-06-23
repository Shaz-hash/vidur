"""Small local wrapper for native C++ GV3 value+prior arena self-play.

This is Phase 1 glue: Python coordinates paths and process launch, while the
existing native arena runner performs MCTS/game execution and writes arena CSVs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from vidur.AlphaGoZero.config import Phase1SmokeConfig, REPO_ROOT


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path_arg(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _default_output_dir(cfg: Phase1SmokeConfig, *, iterations: int, hops: int, game_id: int) -> Path:
    return cfg.output_root / f"iter{int(iterations)}_hop{int(hops)}_gid{int(game_id)}"


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _job_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "jobs" / f"game_{int(args.game_id)}_hop_{int(args.history_hops)}"


def _copy_job_outputs_to_top_level(output_dir: Path, job_dir: Path) -> None:
    result_src = job_dir / "arena_results.csv"
    if result_src.is_file():
        shutil.copy2(result_src, output_dir / "arena_results.csv")

    games_src = job_dir / "arena_games"
    if games_src.is_dir():
        games_dst = output_dir / "arena_games"
        games_dst.mkdir(parents=True, exist_ok=True)
        for src in sorted(games_src.glob("*.csv")):
            shutil.copy2(src, games_dst / src.name)


def _arena_game_csvs(output_dir: Path) -> list[str]:
    job_logs = sorted((output_dir / "jobs").glob("game_*_hop_*/arena_games/*.csv"))
    if job_logs:
        return [str(p) for p in job_logs]
    return [str(p) for p in sorted((output_dir / "arena_games").glob("*.csv"))]


def build_command(args: argparse.Namespace, *, command_output_dir: Path, launcher_worker: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP",
        "--model-path",
        str(args.value_model_path),
        "--model-version",
        str(int(args.model_version)),
        "--feature-dim",
        str(int(args.feature_dim)),
        "--output-dir",
        str(command_output_dir),
        "--game-id-start",
        str(int(args.game_id)),
        "--num-games",
        "1" if launcher_worker else str(int(args.num_games)),
        "--num-parallel-games",
        "1" if launcher_worker else str(int(args.parallel_games)),
        "--launcher-poll-sec",
        str(float(args.launcher_poll_sec)),
        "--shared-root-mcts-iterations",
        str(int(args.iterations)),
        "--worker-threads",
        str(int(args.worker_threads)),
        "--trivial-budget-tokens",
        str(int(args.trivial_budget_tokens)),
        "--arena-time-limit-sec",
        str(float(args.arena_time_limit_sec)),
        "--history-hops-min",
        str(int(args.history_hops)),
        "--history-hops-max",
        str(int(args.history_hops)),
        "--no-history-hops-unique",
        "--seed",
        str(int(args.seed)),
        "--uct-c",
        str(float(args.uct_c)),
        "--puct-c",
        str(float(args.puct_c)),
        "--policy-prior-temperature",
        str(float(args.policy_prior_temperature)),
        "--prior-min-prob",
        str(float(args.prior_min_prob)),
        "--root-dirichlet-noise-enabled" if bool(args.root_dirichlet_noise_enabled) else "--no-root-dirichlet-noise-enabled",
        "--root-dirichlet-alpha",
        str(float(args.root_dirichlet_alpha)),
        "--root-dirichlet-epsilon",
        str(float(args.root_dirichlet_epsilon)),
        "--agz-sample-initial-moves" if bool(args.agz_sample_initial_moves) else "--no-agz-sample-initial-moves",
        "--agz-sample-initial-move-count",
        str(int(args.agz_sample_initial_move_count)),
        "--agz-mcts-action-temperature",
        str(float(args.agz_mcts_action_temperature)),
        "--controller-prior-model-path",
        str(args.controller_prior_model_path),
        "--adversary-prior-model-path",
        str(args.adversary_prior_model_path),
    ]
    if launcher_worker:
        cmd.append("--launcher-worker")
    if int(args.history_hops) == 0:
        cmd.append("--history-hops-force-zero")
    else:
        cmd.append("--no-history-hops-force-zero")
    if bool(args.skip_model_ctrl_cycle):
        cmd.append("--skip-model-ctrl-cycle")
    if bool(args.only_model_ctrl_cycle):
        cmd.append("--only-model-ctrl-cycle")
    if bool(args.force_build_native):
        cmd.append("--force-build-native")
    if bool(args.write_mcts_visit_logs):
        cmd.append("--write-mcts-visit-logs")
    cmd.extend(["--agz-replay-target-csv", str(args.output_dir / "replay_target_runtime.csv")])
    return cmd


def parse_args() -> argparse.Namespace:
    cfg = Phase1SmokeConfig()
    parser = argparse.ArgumentParser(description="Run one local native GV3 value+prior MCTS arena smoke game.")
    parser.add_argument("--value-model-path", type=_path_arg, default=cfg.value_model_path)
    parser.add_argument("--controller-prior-model-path", type=_path_arg, default=cfg.controller_prior_model_path)
    parser.add_argument("--adversary-prior-model-path", type=_path_arg, default=cfg.adversary_prior_model_path)
    parser.add_argument("--model-version", type=int, default=cfg.model_version)
    parser.add_argument("--feature-dim", type=int, default=cfg.feature_dim)
    parser.add_argument("--game-id", type=int, default=cfg.game_id)
    parser.add_argument("--num-games", type=int, default=cfg.num_games)
    parser.add_argument("--parallel-games", type=int, default=cfg.parallel_games)
    parser.add_argument("--history-hops", type=int, default=cfg.history_hops)
    parser.add_argument("--iterations", type=int, default=cfg.mcts_iterations)
    parser.add_argument("--arena-time-limit-sec", type=float, default=cfg.arena_time_limit_sec)
    parser.add_argument("--trivial-budget-tokens", type=int, default=cfg.trivial_budget_tokens)
    parser.add_argument("--puct-c", type=float, default=cfg.puct_c)
    parser.add_argument("--uct-c", type=float, default=cfg.uct_c)
    parser.add_argument("--policy-prior-temperature", type=float, default=cfg.policy_prior_temperature)
    parser.add_argument("--prior-min-prob", type=float, default=cfg.prior_min_prob)
    parser.add_argument("--root-dirichlet-noise-enabled", action=argparse.BooleanOptionalAction, default=cfg.root_dirichlet_noise_enabled)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=cfg.root_dirichlet_alpha)
    parser.add_argument("--root-dirichlet-epsilon", type=float, default=cfg.root_dirichlet_epsilon)
    parser.add_argument("--agz-sample-initial-moves", action=argparse.BooleanOptionalAction, default=cfg.agz_sample_initial_moves)
    parser.add_argument("--agz-sample-initial-move-count", type=int, default=cfg.agz_sample_initial_move_count)
    parser.add_argument("--agz-mcts-action-temperature", type=float, default=cfg.agz_mcts_action_temperature)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--worker-threads", type=int, default=cfg.worker_threads)
    parser.add_argument("--launcher-poll-sec", type=float, default=1.0)
    parser.add_argument("--output-dir", type=_path_arg, default=None)
    parser.add_argument("--skip-model-ctrl-cycle", action="store_true")
    parser.add_argument("--only-model-ctrl-cycle", action="store_true", default=True)
    parser.add_argument("--include-trivial-cycle", action="store_false", dest="only_model_ctrl_cycle")
    parser.add_argument("--force-build-native", action="store_true")
    parser.add_argument("--write-mcts-visit-logs", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = _default_output_dir(
            cfg,
            iterations=int(args.iterations),
            hops=int(args.history_hops),
            game_id=int(args.game_id),
        )
    else:
        args.output_dir = Path(args.output_dir).expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    _require_file(args.value_model_path, "value model")
    _require_file(args.controller_prior_model_path, "controller prior model")
    _require_file(args.adversary_prior_model_path, "adversary prior model")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher_worker = int(args.num_games) == 1
    command_output_dir = _job_output_dir(args) if launcher_worker else args.output_dir
    command_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, command_output_dir=command_output_dir, launcher_worker=launcher_worker)
    summary_path = args.output_dir / "phase1_run_summary.json"
    start = time.time()
    summary = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(args.output_dir),
        "job_output_dir": str(command_output_dir),
        "command": cmd,
        "iterations": int(args.iterations),
        "history_hops": int(args.history_hops),
        "policy_prior_temperature": float(args.policy_prior_temperature),
        "root_dirichlet_noise_enabled": bool(args.root_dirichlet_noise_enabled),
        "root_dirichlet_alpha": float(args.root_dirichlet_alpha),
        "root_dirichlet_epsilon": float(args.root_dirichlet_epsilon),
        "agz_sample_initial_moves": bool(args.agz_sample_initial_moves),
        "agz_sample_initial_move_count": int(args.agz_sample_initial_move_count),
        "agz_mcts_action_temperature": float(args.agz_mcts_action_temperature),
        "trivial_budget_tokens": int(args.trivial_budget_tokens),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[agz-phase1] launching local game out={args.output_dir}", flush=True)
    print("[agz-phase1] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start
    if proc.returncode == 0 and launcher_worker:
        _copy_job_outputs_to_top_level(args.output_dir, command_output_dir)

    arena_games = _arena_game_csvs(args.output_dir)
    summary.update(
        {
            "status": "ok" if proc.returncode == 0 else "failed",
            "finished_at_utc": _utc_now(),
            "returncode": int(proc.returncode),
            "elapsed_sec": float(elapsed),
            "arena_results_csv": str(args.output_dir / "arena_results.csv"),
            "arena_game_csvs": arena_games,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[agz-phase1] finished rc={proc.returncode} elapsed={elapsed:.3f}s", flush=True)
    print(f"[agz-phase1] summary={summary_path}", flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
