
"""Worker-side AlphaGoZero replay generation daemon.

This is deliberately process-per-game to avoid simulator/native memory bloat.
The worker can keep generating while completed shards upload asynchronously.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidur.AlphaGoZero.config import Phase1SmokeConfig, REPO_ROOT
from vidur.AlphaGoZero.durable_transfer import (
    append_csv_file,
    append_csv_row,
    atomic_write_json,
    finalize_shard,
    local_time_24h,
    replay_counts,
    remote_verify_and_publish,
    rsync_dir_to,
    utc_now,
    wait_for_ack,
)

WORKER_REPLAY_FIELDS = [
    "states_generated",
    "controller_states",
    "adversary_states",
    "games_executed_so_far",
    "model_iteration_version",
    "time_24h",
]
MODEL_COMM_FIELDS = [
    "model_received_time",
    "model_version",
    "states_generated_by_model_version_current_buffer",
]


@dataclass
class WorkerState:
    states: int = 0
    controller: int = 0
    adversary: int = 0
    games: int = 0
    shard_index: int = 0
    model_version: int = 100


@dataclass
class RunningGame:
    game_id: int
    out_dir: Path
    replay_csv: Path
    log_path: Path
    proc: subprocess.Popen[Any]
    model_version: int


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _load_state(path: Path, default_model_version: int) -> WorkerState:
    if not path.exists():
        return WorkerState(model_version=int(default_model_version))
    data = json.loads(path.read_text(encoding="utf-8"))
    return WorkerState(
        states=int(data.get("states", 0)),
        controller=int(data.get("controller", 0)),
        adversary=int(data.get("adversary", 0)),
        games=int(data.get("games", 0)),
        shard_index=int(data.get("shard_index", 0)),
        model_version=int(data.get("model_version", default_model_version)),
    )


def _save_state(path: Path, st: WorkerState) -> None:
    atomic_write_json(path, {
        "states": st.states,
        "controller": st.controller,
        "adversary": st.adversary,
        "games": st.games,
        "shard_index": st.shard_index,
        "model_version": st.model_version,
        "updated_at_utc": utc_now(),
    })


def _ensure_active_dir(root: Path) -> Path:
    active = root / "active"
    active.mkdir(parents=True, exist_ok=True)
    return active


def _worker_buffer_path(root: Path) -> Path:
    return root / "replay_buffer.csv"


def _model_comm_path(root: Path) -> Path:
    # Keep the user-facing filename exactly as requested.
    return root / "model communication.csv"


def _append_worker_status(root: Path, st: WorkerState) -> None:
    append_csv_row(_worker_buffer_path(root), WORKER_REPLAY_FIELDS, {
        "states_generated": int(st.states),
        "controller_states": int(st.controller),
        "adversary_states": int(st.adversary),
        "games_executed_so_far": int(st.games),
        "model_iteration_version": int(st.model_version),
        "time_24h": local_time_24h(),
    })


def _append_model_comm(root: Path, st: WorkerState) -> None:
    append_csv_row(_model_comm_path(root), MODEL_COMM_FIELDS, {
        "model_received_time": local_time_24h(),
        "model_version": int(st.model_version),
        "states_generated_by_model_version_current_buffer": int(st.states),
    })


def _current_model_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, int]:
    cfg = Phase1SmokeConfig()
    # Initial implementation uses configured file paths. Worker model pulling can
    # later atomically update current_model.json to override these paths.
    current = Path(args.output_root) / "models" / "current_model.json"
    if current.exists():
        data = json.loads(current.read_text(encoding="utf-8"))
        return (
            Path(data["value_model_path"]),
            Path(data["controller_prior_model_path"]),
            Path(data["adversary_prior_model_path"]),
            int(data["model_version"]),
        )
    return (
        Path(args.value_model_path or cfg.value_model_path),
        Path(args.controller_prior_model_path or cfg.controller_prior_model_path),
        Path(args.adversary_prior_model_path or cfg.adversary_prior_model_path),
        int(args.model_version),
    )


def _select_parent_args(args: argparse.Namespace, rng: random.Random, used: set[int]) -> list[str]:
    dataset = str(args.parent_dataset_dir or "")
    if not dataset:
        return []
    max_id = int(args.parent_state_count)
    if max_id <= 0:
        if int(args.parent_state_id) >= 0:
            sid = int(args.parent_state_id)
        else:
            return []
    else:
        for _ in range(max(10, max_id * 2)):
            sid = int(rng.randrange(max_id))
            if sid not in used:
                used.add(sid)
                break
        else:
            used.clear()
            sid = int(rng.randrange(max_id))
            used.add(sid)
    return [
        "--parent-dataset-dir", dataset,
        "--parent-state-id", str(int(sid)),
        "--parent-root-player-filter", str(args.parent_root_player_filter),
    ]


def _build_game_command(
    args: argparse.Namespace,
    *,
    game_id: int,
    out_dir: Path,
    used_parent_ids: set[int],
    rng: random.Random,
) -> tuple[list[str], Path, int]:
    value_model, ctrl_prior, adv_prior, model_version = _current_model_paths(args)
    replay_csv = out_dir / "replay_target_runtime.csv"
    cmd = [
        sys.executable,
        "-m", "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP",
        "--launcher-worker",
        "--model-path", str(value_model),
        "--model-version", str(int(model_version)),
        "--feature-dim", str(int(args.feature_dim)),
        "--output-dir", str(out_dir),
        "--game-id-start", str(int(game_id)),
        "--num-games", "1",
        "--num-parallel-games", "1",
        "--shared-root-mcts-iterations", str(int(args.iterations)),
        "--worker-threads", str(int(args.worker_threads)),
        "--trivial-budget-tokens", str(int(args.trivial_budget_tokens)),
        "--arena-time-limit-sec", str(float(args.arena_time_limit_sec)),
        "--history-hops-min", str(int(args.history_hops)),
        "--history-hops-max", str(int(args.history_hops)),
        "--no-history-hops-unique",
        "--seed", str(int(args.seed) + int(game_id)),
        "--uct-c", str(float(args.uct_c)),
        "--puct-c", str(float(args.puct_c)),
        "--policy-prior-temperature", str(float(args.policy_prior_temperature)),
        "--prior-min-prob", str(float(args.prior_min_prob)),
        "--root-dirichlet-noise-enabled" if bool(args.root_dirichlet_noise_enabled) else "--no-root-dirichlet-noise-enabled",
        "--root-dirichlet-alpha", str(float(args.root_dirichlet_alpha)),
        "--root-dirichlet-epsilon", str(float(args.root_dirichlet_epsilon)),
        "--agz-sample-initial-moves" if bool(args.agz_sample_initial_moves) else "--no-agz-sample-initial-moves",
        "--agz-sample-initial-move-count", str(int(args.agz_sample_initial_move_count)),
        "--agz-mcts-action-temperature", str(float(args.agz_mcts_action_temperature)),
        "--controller-prior-model-path", str(ctrl_prior),
        "--adversary-prior-model-path", str(adv_prior),
        "--only-model-ctrl-cycle",
        "--agz-replay-target-csv", str(replay_csv),
    ]
    if int(args.history_hops) == 0:
        cmd.append("--history-hops-force-zero")
    else:
        cmd.append("--no-history-hops-force-zero")
    cmd.extend(_select_parent_args(args, rng, used_parent_ids))
    return cmd, replay_csv, int(model_version)


def _launch_one_game(
    args: argparse.Namespace,
    *,
    game_id: int,
    out_dir: Path,
    used_parent_ids: set[int],
    rng: random.Random,
) -> RunningGame:
    cmd, replay_csv, model_version = _build_game_command(
        args,
        game_id=int(game_id),
        out_dir=out_dir,
        used_parent_ids=used_parent_ids,
        rng=rng,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "launch_command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    log = out_dir / "game_process.log"
    f = log.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), text=True, stdout=f, stderr=subprocess.STDOUT)
    finally:
        f.close()
    return RunningGame(
        game_id=int(game_id),
        out_dir=out_dir,
        replay_csv=replay_csv,
        log_path=log,
        proc=proc,
        model_version=int(model_version),
    )


def _run_one_game(args: argparse.Namespace, *, game_id: int, out_dir: Path, st: WorkerState, used_parent_ids: set[int], rng: random.Random) -> Path:
    running = _launch_one_game(args, game_id=game_id, out_dir=out_dir, used_parent_ids=used_parent_ids, rng=rng)
    rc = running.proc.wait()
    if int(rc) != 0:
        raise RuntimeError(f"game {game_id} failed rc={rc}; see {running.log_path}")
    st.model_version = int(running.model_version)
    return running.replay_csv


def _merge_game_into_active(
    args: argparse.Namespace,
    *,
    game_id: int,
    game_out: Path,
    replay_csv: Path,
    st: WorkerState,
    model_version: int | None = None,
) -> dict[str, int]:
    active = _ensure_active_dir(Path(args.output_root))
    active_replay = active / "replay_target_runtime.csv"
    if model_version is None:
        model_version = int(st.model_version)
    extra = {
        "worker_id": str(args.worker_id),
        "run_id": str(args.run_id),
        "model_version": int(model_version),
        "game_id_source": int(game_id),
    }
    added = append_csv_file(active_replay, replay_csv, extra=extra)
    policy_csv = Path(game_out) / "replay_policy_rows.csv"
    if policy_csv.exists():
        append_csv_file(active / "replay_policy_rows.csv", policy_csv, extra=extra)
    counts = replay_counts(replay_csv)
    st.states += int(counts["states"])
    st.controller += int(counts["controller"])
    st.adversary += int(counts["adversary"])
    st.games += 1
    # Keep lightweight game output references in active shard for traceability.
    active_games = active / "game_outputs"
    active_games.mkdir(parents=True, exist_ok=True)
    manifest = {
        "game_id": int(game_id),
        "game_output_dir": str(game_out),
        "replay_rows": int(added),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(active_games / f"game_{int(game_id)}.json", manifest)
    return counts


def _cleanup_game_output(args: argparse.Namespace, game_out: Path) -> None:
    if bool(getattr(args, "keep_game_runs", False)):
        return
    shutil.rmtree(Path(game_out), ignore_errors=True)


def _freeze_if_needed(args: argparse.Namespace, st: WorkerState, *, force: bool = False) -> Path | None:
    root = Path(args.output_root)
    active = root / "active"
    if not active.exists() or not (active / "replay_target_runtime.csv").exists():
        return None
    if not force and int(st.states) < int(args.buffer_threshold):
        return None
    shard_id = f"{args.worker_id}_{int(st.shard_index):06d}"
    shard = finalize_shard(
        active_dir=active,
        ready_root=root / "ready",
        shard_id=shard_id,
        worker_id=str(args.worker_id),
        model_version=int(st.model_version),
        games_executed=int(st.games),
    )
    st.shard_index += 1
    st.states = 0
    st.controller = 0
    st.adversary = 0
    st.games = 0
    _save_state(root / "worker_state.json", st)
    _append_worker_status(root, st)
    return shard


def _upload_ready_shards(args: argparse.Namespace) -> None:
    ready = Path(args.output_root) / "ready"
    if not ready.exists():
        return
    for shard in sorted(p for p in ready.iterdir() if p.is_dir()):
        shard_id = shard.name
        upload = f"{args.xl_output_root.rstrip('/')}/incoming_uploading/{args.worker_id}/{shard_id}"
        incoming = f"{args.xl_output_root.rstrip('/')}/incoming/{args.worker_id}/{shard_id}"
        rejected = f"{args.xl_output_root.rstrip('/')}/rejected/{args.worker_id}/{shard_id}"
        ack = f"{args.xl_output_root.rstrip('/')}/acks/{args.worker_id}/{shard_id}.accepted"
        rsync_dir_to(shard, str(args.xl_host), upload)
        remote_verify_and_publish(host=str(args.xl_host), uploading_dir=upload, incoming_dir=incoming, rejected_dir=rejected)
        got_ack = wait_for_ack(host=str(args.xl_host), ack_path=ack, timeout_sec=int(args.ack_timeout_sec), poll_sec=5.0)
        if got_ack.accepted:
            shutil.rmtree(shard)


def _uploader_loop(args: argparse.Namespace, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            _upload_ready_shards(args)
        except Exception as exc:  # noqa: BLE001 - uploader must not kill generation.
            err_dir = Path(args.output_root) / "upload_errors"
            err_dir.mkdir(parents=True, exist_ok=True)
            append_csv_row(err_dir / "upload_errors.csv", ["time_24h", "error"], {
                "time_24h": local_time_24h(),
                "error": repr(exc),
            })
        stop_event.wait(float(args.upload_poll_sec))


def run_worker(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    st = _load_state(root / "worker_state.json", int(args.model_version))
    _append_model_comm(root, st)
    rng = random.Random(int(args.seed) + abs(hash(str(args.worker_id))) % 1000000)
    used_parent_ids: set[int] = set()
    running: list[RunningGame] = []
    launched = 0

    uploader_stop = threading.Event()
    uploader_thread: threading.Thread | None = None
    if bool(args.upload_after_game):
        uploader_thread = threading.Thread(target=_uploader_loop, args=(args, uploader_stop), daemon=True)
        uploader_thread.start()

    try:
        while True:
            can_launch_more = int(args.max_games) <= 0 or launched < int(args.max_games)
            while can_launch_more and len(running) < max(1, int(args.parallel_games)):
                _value_model, _ctrl_prior, _adv_prior, current_model_version = _current_model_paths(args)
                if int(current_model_version) != int(st.model_version):
                    st.model_version = int(current_model_version)
                    _save_state(root / "worker_state.json", st)
                    _append_model_comm(root, st)
                game_id = int(args.game_id_start) + int(st.shard_index) * 1_000_000 + launched
                game_out = root / "runs" / f"game_{game_id}"
                running.append(_launch_one_game(args, game_id=game_id, out_dir=game_out, used_parent_ids=used_parent_ids, rng=rng))
                launched += 1
                can_launch_more = int(args.max_games) <= 0 or launched < int(args.max_games)

            if not running and not can_launch_more:
                break

            completed_any = False
            for game in list(running):
                rc = game.proc.poll()
                if rc is None:
                    continue
                running.remove(game)
                completed_any = True
                if int(rc) != 0:
                    append_csv_row(root / "game_errors.csv", ["game_id", "model_version", "return_code", "log_path", "time_24h"], {
                        "game_id": int(game.game_id),
                        "model_version": int(game.model_version),
                        "return_code": int(rc),
                        "log_path": str(game.log_path),
                        "time_24h": local_time_24h(),
                    })
                    if not bool(args.continue_on_game_error):
                        raise RuntimeError(f"game {game.game_id} failed rc={rc}; see {game.log_path}")
                    continue

                st.model_version = int(game.model_version)
                _merge_game_into_active(
                    args,
                    game_id=int(game.game_id),
                    game_out=game.out_dir,
                    replay_csv=game.replay_csv,
                    st=st,
                    model_version=int(game.model_version),
                )
                _cleanup_game_output(args, game.out_dir)
                _save_state(root / "worker_state.json", st)
                _append_worker_status(root, st)
                _freeze_if_needed(args, st)

            if not completed_any:
                time.sleep(float(args.poll_sec))

        if bool(args.flush_at_end):
            _freeze_if_needed(args, st, force=True)
        if bool(args.upload_after_game):
            _upload_ready_shards(args)
    finally:
        uploader_stop.set()
        if uploader_thread is not None:
            uploader_thread.join(timeout=10.0)
        for game in running:
            if game.proc.poll() is None:
                game.proc.terminate()


def parse_args() -> argparse.Namespace:
    cfg = Phase1SmokeConfig()
    p = argparse.ArgumentParser(description="Run AlphaGoZero worker self-play generation.")
    p.add_argument("--worker-id", required=True)
    p.add_argument("--run-id", default="agz_run")
    p.add_argument("--output-root", type=_path_arg, default=Path("/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero/worker_buffer"))
    p.add_argument("--xl-host", default="bellman-classical-xl")
    p.add_argument("--xl-output-root", default="/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero")
    p.add_argument("--value-model-path", default=str(cfg.value_model_path))
    p.add_argument("--controller-prior-model-path", default=str(cfg.controller_prior_model_path))
    p.add_argument("--adversary-prior-model-path", default=str(cfg.adversary_prior_model_path))
    p.add_argument("--model-version", type=int, default=cfg.model_version)
    p.add_argument("--feature-dim", type=int, default=cfg.feature_dim)
    p.add_argument("--game-id-start", type=int, default=18_000_000)
    p.add_argument("--max-games", type=int, default=0, help="0 means run forever")
    p.add_argument("--parallel-games", type=int, default=60)
    p.add_argument("--poll-sec", type=float, default=1.0)
    p.add_argument("--buffer-threshold", type=int, default=10_000)
    p.add_argument("--iterations", type=int, default=cfg.mcts_iterations)
    p.add_argument("--history-hops", type=int, default=0)
    p.add_argument("--arena-time-limit-sec", type=float, default=5.0)
    p.add_argument("--trivial-budget-tokens", type=int, default=256)
    p.add_argument("--puct-c", type=float, default=cfg.puct_c)
    p.add_argument("--uct-c", type=float, default=1.4)
    p.add_argument("--policy-prior-temperature", type=float, default=1.0)
    p.add_argument("--prior-min-prob", type=float, default=1e-8)
    p.add_argument("--root-dirichlet-noise-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--root-dirichlet-alpha", type=float, default=0.03)
    p.add_argument("--root-dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--agz-sample-initial-moves", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--agz-sample-initial-move-count", type=int, default=30)
    p.add_argument("--agz-mcts-action-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--worker-threads", type=int, default=1)
    p.add_argument("--parent-dataset-dir", default="")
    p.add_argument("--parent-state-id", type=int, default=-1)
    p.add_argument("--parent-state-count", type=int, default=0)
    p.add_argument("--parent-root-player-filter", choices=("controller", "adversary", "any"), default="any")
    p.add_argument("--upload-after-game", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--upload-poll-sec", type=float, default=30.0)
    p.add_argument("--flush-at-end", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ack-timeout-sec", type=int, default=0)
    p.add_argument("--continue-on-game-error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-game-runs", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    run_worker(parse_args())


if __name__ == "__main__":
    main()
