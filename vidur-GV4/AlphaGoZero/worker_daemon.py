
"""Worker-side AlphaGoZero replay generation daemon.

This is deliberately process-per-game to avoid simulator/native memory bloat.
The worker can keep generating while completed shards upload asynchronously.
"""

from __future__ import annotations

import argparse
import csv
import inspect
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

from vidur.AlphaGoZero.adaptive_rollout import active_rollout_horizon_sec
from vidur.AlphaGoZero.config import Phase1SmokeConfig, REPO_ROOT
from vidur.AlphaGoZero.replay_runtime import AlphaGoZeroReplayRecorder
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
    "controller_model_iteration_version",
    "adversary_model_iteration_version",
    "time_24h",
]
MODEL_COMM_FIELDS = [
    "model_received_time",
    "model_version",
    "controller_model_version",
    "adversary_model_version",
    "states_generated_by_model_version_current_buffer",
]
NATIVE_GAME_ID_MAX = 2_147_483_647
_UPLOAD_READY_LOCK = threading.Lock()


@dataclass
class WorkerState:
    states: int = 0
    controller: int = 0
    adversary: int = 0
    games: int = 0
    shard_index: int = 0
    next_game_sequence: int = 0
    model_version: int = 100
    controller_model_version: int = 100
    adversary_model_version: int = 100


@dataclass(frozen=True)
class CurrentModelPaths:
    controller_value_model_path: Path
    adversary_value_model_path: Path
    controller_prior_model_path: Path
    adversary_prior_model_path: Path
    model_version: int
    controller_model_version: int
    adversary_model_version: int


@dataclass
class RunningGame:
    game_id: int
    out_dir: Path
    replay_csv: Path
    log_path: Path
    proc: subprocess.Popen[Any]
    model_version: int
    controller_model_version: int
    adversary_model_version: int


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _load_state(path: Path, default_model_version: int) -> WorkerState:
    if not path.exists():
        return WorkerState(
            model_version=int(default_model_version),
            controller_model_version=int(default_model_version),
            adversary_model_version=int(default_model_version),
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return WorkerState(
        states=int(data.get("states", 0)),
        controller=int(data.get("controller", 0)),
        adversary=int(data.get("adversary", 0)),
        games=int(data.get("games", 0)),
        shard_index=int(data.get("shard_index", 0)),
        next_game_sequence=int(data.get("next_game_sequence", 0)),
        model_version=int(data.get("model_version", default_model_version)),
        controller_model_version=int(data.get("controller_model_version", data.get("model_version", default_model_version))),
        adversary_model_version=int(data.get("adversary_model_version", data.get("model_version", default_model_version))),
    )


def _save_state(path: Path, st: WorkerState) -> None:
    atomic_write_json(path, {
        "states": st.states,
        "controller": st.controller,
        "adversary": st.adversary,
        "games": st.games,
        "shard_index": st.shard_index,
        "next_game_sequence": st.next_game_sequence,
        "model_version": st.model_version,
        "controller_model_version": st.controller_model_version,
        "adversary_model_version": st.adversary_model_version,
        "updated_at_utc": utc_now(),
    })


def _ensure_active_dir(root: Path) -> Path:
    active = root / "active"
    active.mkdir(parents=True, exist_ok=True)
    return active


def _allocate_game_id(args: argparse.Namespace, st: WorkerState, state_path: Path) -> int:
    game_id = int(args.game_id_start) + int(st.next_game_sequence)
    if game_id < 0 or game_id > NATIVE_GAME_ID_MAX:
        raise RuntimeError(
            f"worker game-id sequence exhausted native int32 range: game_id={game_id}; "
            "choose a lower --game-id-start"
        )
    st.next_game_sequence += 1
    # Persist before launch so daemon restarts never reuse an in-flight game ID.
    _save_state(state_path, st)
    return game_id


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
        "controller_model_iteration_version": int(st.controller_model_version),
        "adversary_model_iteration_version": int(st.adversary_model_version),
        "time_24h": local_time_24h(),
    })


def _append_model_comm(root: Path, st: WorkerState) -> None:
    append_csv_row(_model_comm_path(root), MODEL_COMM_FIELDS, {
        "model_received_time": local_time_24h(),
        "model_version": int(st.model_version),
        "controller_model_version": int(st.controller_model_version),
        "adversary_model_version": int(st.adversary_model_version),
        "states_generated_by_model_version_current_buffer": int(st.states),
    })


def _current_model_paths(args: argparse.Namespace) -> CurrentModelPaths:
    cfg = Phase1SmokeConfig()
    current = Path(args.output_root) / "models" / "current_model.json"
    if current.exists():
        data = json.loads(current.read_text(encoding="utf-8"))
        model_family = str(data.get("model_family", "hgb") or "hgb").lower()
        if model_family == "dnn" and data.get("native_ready") is not True:
            raise RuntimeError("refusing to launch an EXP2 DNN bundle without native_ready=true")
        legacy_version = int(data.get("model_version", 100) or 100)
        controller_version = int(data.get("controller_model_version", legacy_version) or legacy_version)
        adversary_version = int(data.get("adversary_model_version", legacy_version) or legacy_version)
        legacy_value_path = data.get("value_model_path") or str(cfg.value_model_path)
        result = CurrentModelPaths(
            controller_value_model_path=Path(data.get("controller_value_model_path") or legacy_value_path),
            adversary_value_model_path=Path(data.get("adversary_value_model_path") or legacy_value_path),
            controller_prior_model_path=Path(data["controller_prior_model_path"]),
            adversary_prior_model_path=Path(data["adversary_prior_model_path"]),
            model_version=int(max(legacy_version, controller_version, adversary_version)),
            controller_model_version=int(controller_version),
            adversary_model_version=int(adversary_version),
        )
        missing = [
            str(path)
            for path in (
                result.controller_value_model_path,
                result.adversary_value_model_path,
                result.controller_prior_model_path,
                result.adversary_prior_model_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"current model bundle is incomplete: {missing}")
        return result
    model_version = int(args.model_version)
    value_path = Path(args.value_model_path or cfg.value_model_path)
    return CurrentModelPaths(
        controller_value_model_path=Path(getattr(args, "controller_value_model_path", "") or value_path),
        adversary_value_model_path=Path(getattr(args, "adversary_value_model_path", "") or value_path),
        controller_prior_model_path=Path(args.controller_prior_model_path or cfg.controller_prior_model_path),
        adversary_prior_model_path=Path(args.adversary_prior_model_path or cfg.adversary_prior_model_path),
        model_version=int(model_version),
        controller_model_version=int(model_version),
        adversary_model_version=int(model_version),
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



def _rollout_horizon_sec_for_game(args: argparse.Namespace) -> float:
    # Spot assignments are complete, fingerprint-verified controller configs.
    # Fixed workers instead receive the adaptive runtime file in their root.
    if str(getattr(args, "assignment_config_sha256", "")):
        return float(args.rollout_horizon_sec)
    return float(active_rollout_horizon_sec(Path(args.output_root)))


def _build_game_command(
    args: argparse.Namespace,
    *,
    game_id: int,
    out_dir: Path,
    used_parent_ids: set[int],
    rng: random.Random,
) -> tuple[list[str], Path, CurrentModelPaths]:
    rollout_horizon_sec = _rollout_horizon_sec_for_game(args)
    model_paths = _current_model_paths(args)
    replay_csv = out_dir / "replay_target_runtime.csv"
    cmd = [
        sys.executable,
        "-m", "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP",
        "--launcher-worker",
        "--model-path", str(model_paths.controller_value_model_path),
        "--model-version", str(int(model_paths.model_version)),
        "--feature-dim", str(int(args.feature_dim)),
        "--output-dir", str(out_dir),
        "--game-id-start", str(int(game_id)),
        "--num-games", "1",
        "--num-parallel-games", "1",
        "--shared-root-mcts-iterations", str(int(args.iterations)),
        "--discount-factor", str(float(args.discount_factor)),
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
        "--root-dirichlet-total-concentration", str(float(args.root_dirichlet_total_concentration)),
        "--root-dirichlet-epsilon", str(float(args.root_dirichlet_epsilon)),
        "--agz-sample-initial-moves" if bool(args.agz_sample_initial_moves) else "--no-agz-sample-initial-moves",
        "--agz-sample-initial-move-count", str(int(args.agz_sample_initial_move_count)),
        "--agz-mcts-action-temperature", str(float(args.agz_mcts_action_temperature)),
        "--controller-prior-model-path", str(model_paths.controller_prior_model_path),
        "--adversary-prior-model-path", str(model_paths.adversary_prior_model_path),
        "--role-controller-value-model-path", str(model_paths.controller_value_model_path),
        "--role-controller-prior-model-path", str(model_paths.controller_prior_model_path),
        "--role-adversary-value-model-path", str(model_paths.adversary_value_model_path),
        "--role-adversary-prior-model-path", str(model_paths.adversary_prior_model_path),
        "--native-search-mode", str(args.native_search_mode),
        "--rollout-count", str(int(args.rollout_count)),
        "--rollout-parallel-threads", str(int(args.rollout_parallel_threads)),
        "--rollout-horizon-sec", str(float(rollout_horizon_sec)),
        "--rollout-policy-temperature", str(float(args.rollout_policy_temperature)),
        "--rollout-probability-quantum", str(float(args.rollout_probability_quantum)),
        "--rollout-max-actions", str(int(args.rollout_max_actions)),
        "--only-model-ctrl-cycle",
        "--agz-replay-target-csv", str(replay_csv),
        "--agz-replay-sample-window-sec", str(float(args.replay_sample_window_sec)),
    ]
    if int(args.history_hops) == 0:
        cmd.append("--history-hops-force-zero")
    else:
        cmd.append("--no-history-hops-force-zero")
    cmd.extend(_select_parent_args(args, rng, used_parent_ids))
    return cmd, replay_csv, model_paths


def _launch_one_game(
    args: argparse.Namespace,
    *,
    game_id: int,
    out_dir: Path,
    used_parent_ids: set[int],
    rng: random.Random,
) -> RunningGame:
    cmd, replay_csv, model_paths = _build_game_command(
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
        model_version=int(model_paths.model_version),
        controller_model_version=int(model_paths.controller_model_version),
        adversary_model_version=int(model_paths.adversary_model_version),
    )


def _run_one_game(args: argparse.Namespace, *, game_id: int, out_dir: Path, st: WorkerState, used_parent_ids: set[int], rng: random.Random) -> Path:
    running = _launch_one_game(args, game_id=game_id, out_dir=out_dir, used_parent_ids=used_parent_ids, rng=rng)
    rc = running.proc.wait()
    if int(rc) != 0:
        raise RuntimeError(f"game {game_id} failed rc={rc}; see {running.log_path}")
    st.model_version = int(running.model_version)
    st.controller_model_version = int(running.controller_model_version)
    st.adversary_model_version = int(running.adversary_model_version)
    return running.replay_csv


def _merge_game_into_active(
    args: argparse.Namespace,
    *,
    game_id: int,
    game_out: Path,
    replay_csv: Path,
    st: WorkerState,
    model_version: int | None = None,
    controller_model_version: int | None = None,
    adversary_model_version: int | None = None,
) -> dict[str, int]:
    active = _ensure_active_dir(Path(args.output_root))
    active_replay = active / "replay_target_runtime.csv"
    if model_version is None:
        model_version = int(st.model_version)
    if controller_model_version is None:
        controller_model_version = int(st.controller_model_version)
    if adversary_model_version is None:
        adversary_model_version = int(st.adversary_model_version)
    extra = {
        "worker_id": str(args.worker_id),
        "run_id": str(args.run_id),
        "model_version": int(model_version),
        "controller_model_version": int(controller_model_version),
        "adversary_model_version": int(adversary_model_version),
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
        "mcts_iterations": int(args.iterations),
        "puct_c": float(args.puct_c),
        "root_dirichlet_alpha": float(args.root_dirichlet_alpha),
        "root_dirichlet_total_concentration": float(args.root_dirichlet_total_concentration),
        "root_dirichlet_epsilon": float(args.root_dirichlet_epsilon),
        "assignment_id": str(getattr(args, "assignment_id", "")),
        "selfplay_config_sha256": str(getattr(args, "assignment_config_sha256", "")),
        "game_output_dir": str(game_out),
        "replay_rows": int(added),
        "model_version": int(model_version),
        "controller_model_version": int(controller_model_version),
        "adversary_model_version": int(adversary_model_version),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(active_games / f"game_{int(game_id)}.json", manifest)
    return counts


def _cleanup_game_output(args: argparse.Namespace, game_out: Path) -> None:
    if bool(getattr(args, "keep_game_runs", False)):
        return
    shutil.rmtree(Path(game_out), ignore_errors=True)


def _eval_pause_paths(root: Path) -> tuple[Path, Path]:
    control = Path(root) / "control"
    return control / "eval_pause.request", control / "eval_pause.ack.json"


def _eval_pause_requested(root: Path) -> bool:
    request, _ack = _eval_pause_paths(root)
    return request.is_file()


def _pause_selfplay_if_requested(
    args: argparse.Namespace,
    running: list[RunningGame],
) -> bool:
    root = Path(args.output_root)
    request, ack = _eval_pause_paths(root)
    if not request.is_file():
        ack.unlink(missing_ok=True)
        return False

    terminated = list(running)
    for game in terminated:
        if game.proc.poll() is None:
            game.proc.terminate()
    deadline = time.monotonic() + 10.0
    for game in terminated:
        if game.proc.poll() is None:
            try:
                game.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                game.proc.kill()
                game.proc.wait()
        _cleanup_game_output(args, game.out_dir)
    running.clear()

    if not ack.is_file():
        ack.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            ack,
            {
                "status": "paused",
                "worker_id": str(args.worker_id),
                "terminated_inflight_games": int(len(terminated)),
                "time_utc": utc_now(),
            },
        )
    time.sleep(float(args.poll_sec))
    return True


def _validate_runtime_compatibility(args: argparse.Namespace) -> None:
    parameters = inspect.signature(AlphaGoZeroReplayRecorder.__init__).parameters
    if "discount_factor" not in parameters:
        raise RuntimeError(
            "incompatible AlphaGoZeroReplayRecorder deployment: "
            "worker runner requires discount_factor support"
        )
    if not (0.0 < float(args.discount_factor) <= 1.0):
        raise ValueError("discount_factor must be in (0, 1]")


def _preserve_last_game_error(root: Path, game: RunningGame, return_code: int) -> Path:
    try:
        tail = game.log_path.read_text(encoding="utf-8", errors="replace")[-16_000:]
    except Exception:
        tail = ""
    path = Path(root) / "last_game_error.log"
    path.write_text(
        f"game_id={int(game.game_id)} return_code={int(return_code)} "
        f"time_utc={utc_now()}\n{tail}",
        encoding="utf-8",
    )
    return path


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
        controller_model_version=int(st.controller_model_version),
        adversary_model_version=int(st.adversary_model_version),
        games_executed=int(st.games),
        metadata={
            "mcts_iterations": int(args.iterations),
            "puct_c": float(args.puct_c),
            "root_dirichlet_noise_enabled": bool(args.root_dirichlet_noise_enabled),
            "root_dirichlet_alpha": float(args.root_dirichlet_alpha),
            "root_dirichlet_total_concentration": float(args.root_dirichlet_total_concentration),
            "root_dirichlet_epsilon": float(args.root_dirichlet_epsilon),
            "agz_sample_initial_moves": bool(args.agz_sample_initial_moves),
            "agz_sample_initial_move_count": int(args.agz_sample_initial_move_count),
            "agz_mcts_action_temperature": float(args.agz_mcts_action_temperature),
            "assignment_id": str(getattr(args, "assignment_id", "")),
            "selfplay_config_sha256": str(getattr(args, "assignment_config_sha256", "")),
        },
    )
    st.shard_index += 1
    st.states = 0
    st.controller = 0
    st.adversary = 0
    st.games = 0
    _save_state(root / "worker_state.json", st)
    _append_worker_status(root, st)
    return shard


def _ready_shards_in_upload_order(
    args: argparse.Namespace,
    ready: Path,
    current: CurrentModelPaths,
) -> list[Path]:
    del args, current
    return sorted((path for path in Path(ready).iterdir() if path.is_dir()), key=lambda path: path.name)


def _upload_ready_shards_serial(args: argparse.Namespace) -> None:
    ready = Path(args.output_root) / "ready"
    if not ready.exists():
        return
    current = _current_model_paths(args)
    for shard in _ready_shards_in_upload_order(args, ready, current):
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


def _upload_ready_shards(args: argparse.Namespace) -> None:
    with _UPLOAD_READY_LOCK:
        _upload_ready_shards_serial(args)


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


def run_worker(
    args: argparse.Namespace,
    *,
    preempt_event: threading.Event | None = None,
) -> bool:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    _validate_runtime_compatibility(args)
    state_path = root / "worker_state.json"
    st = _load_state(state_path, int(args.model_version))
    _append_model_comm(root, st)
    rng = random.Random(int(args.seed) + abs(hash(str(args.worker_id))) % 1000000)
    used_parent_ids: set[int] = set()
    running: list[RunningGame] = []
    launched = 0
    consecutive_game_errors = 0

    uploader_stop = threading.Event()
    uploader_thread: threading.Thread | None = None
    preempted = False
    if bool(args.upload_after_game):
        uploader_thread = threading.Thread(target=_uploader_loop, args=(args, uploader_stop), daemon=True)
        uploader_thread.start()

    try:
        while True:
            if _pause_selfplay_if_requested(args, running):
                continue
            stop_requested = preempt_event is not None and preempt_event.is_set()
            can_launch_more = (
                not stop_requested
                and (int(args.max_games) <= 0 or launched < int(args.max_games))
            )
            while can_launch_more and len(running) < max(1, int(args.parallel_games)):
                if _eval_pause_requested(root):
                    break
                current_paths = _current_model_paths(args)
                if (
                    int(current_paths.controller_model_version) != int(st.controller_model_version)
                    or int(current_paths.adversary_model_version) != int(st.adversary_model_version)
                    or int(current_paths.model_version) != int(st.model_version)
                ):
                    st.model_version = int(current_paths.model_version)
                    st.controller_model_version = int(current_paths.controller_model_version)
                    st.adversary_model_version = int(current_paths.adversary_model_version)
                    _save_state(state_path, st)
                    _append_model_comm(root, st)
                game_id = _allocate_game_id(args, st, state_path)
                game_out = root / "runs" / f"game_{game_id}"
                running.append(_launch_one_game(args, game_id=game_id, out_dir=game_out, used_parent_ids=used_parent_ids, rng=rng))
                launched += 1
                can_launch_more = (
                    (preempt_event is None or not preempt_event.is_set())
                    and (
                        int(args.max_games) <= 0
                        or launched < int(args.max_games)
                    )
                )

            if preempt_event is not None and preempt_event.is_set() and not running:
                preempted = True
                _freeze_if_needed(args, st, force=True)
                if bool(args.upload_after_game):
                    _upload_ready_shards(args)
                break
            if not running and not can_launch_more:
                break

            completed_any = False
            failed_this_poll = 0
            succeeded_this_poll = 0
            for game in list(running):
                rc = game.proc.poll()
                if rc is None:
                    continue
                running.remove(game)
                completed_any = True
                if int(rc) != 0:
                    preserved_log = _preserve_last_game_error(root, game, int(rc))
                    append_csv_row(root / "game_errors.csv", ["game_id", "model_version", "return_code", "log_path", "time_24h"], {
                        "game_id": int(game.game_id),
                        "model_version": int(game.model_version),
                        "return_code": int(rc),
                        "log_path": str(preserved_log),
                        "time_24h": local_time_24h(),
                    })
                    _cleanup_game_output(args, game.out_dir)
                    failed_this_poll += 1
                    if not bool(args.continue_on_game_error):
                        raise RuntimeError(f"game {game.game_id} failed rc={rc}; see {preserved_log}")
                    continue

                succeeded_this_poll += 1
                st.model_version = int(game.model_version)
                st.controller_model_version = int(game.controller_model_version)
                st.adversary_model_version = int(game.adversary_model_version)
                _merge_game_into_active(
                    args,
                    game_id=int(game.game_id),
                    game_out=game.out_dir,
                    replay_csv=game.replay_csv,
                    st=st,
                    model_version=int(game.model_version),
                    controller_model_version=int(game.controller_model_version),
                    adversary_model_version=int(game.adversary_model_version),
                )
                _cleanup_game_output(args, game.out_dir)
                _save_state(state_path, st)
                _append_worker_status(root, st)
                _freeze_if_needed(args, st)

            if succeeded_this_poll:
                consecutive_game_errors = 0
            elif failed_this_poll:
                consecutive_game_errors += int(failed_this_poll)
                if consecutive_game_errors >= int(args.max_consecutive_game_errors):
                    raise RuntimeError(
                        f"stopping after {consecutive_game_errors} consecutive game errors; "
                        f"see {root / 'last_game_error.log'}"
                    )
                time.sleep(float(args.game_error_backoff_sec))
            elif not completed_any:
                time.sleep(float(args.poll_sec))

            if preempt_event is not None and preempt_event.is_set():
                preempted = True
                _freeze_if_needed(args, st, force=True)
                if bool(args.upload_after_game):
                    _upload_ready_shards(args)
                break

        if bool(args.flush_at_end):
            _freeze_if_needed(args, st, force=True)
        if bool(args.upload_after_game):
            _upload_ready_shards(args)
        return preempted
    finally:
        for game in running:
            if game.proc.poll() is None:
                game.proc.terminate()
                try:
                    game.proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    game.proc.kill()
            _cleanup_game_output(args, game.out_dir)
        uploader_stop.set()
        if uploader_thread is not None:
            uploader_thread.join(timeout=10.0)


def build_parser() -> argparse.ArgumentParser:
    cfg = Phase1SmokeConfig()
    p = argparse.ArgumentParser(description="Run AlphaGoZero worker self-play generation.")
    p.add_argument("--worker-id", required=True)
    p.add_argument("--run-id", default="agz_run")
    p.add_argument("--output-root", type=_path_arg, default=Path("/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero/worker_buffer"))
    p.add_argument("--xl-host", default="bellman-classical-xl")
    p.add_argument("--xl-output-root", default="/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero")
    p.add_argument("--value-model-path", default=str(cfg.value_model_path))
    p.add_argument("--controller-value-model-path", default="")
    p.add_argument("--adversary-value-model-path", default="")
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
    p.add_argument("--discount-factor", type=float, default=cfg.discount_factor)
    p.add_argument("--history-hops", type=int, default=0)
    p.add_argument("--arena-time-limit-sec", type=float, default=cfg.arena_time_limit_sec)
    p.add_argument("--replay-sample-window-sec", type=float, default=cfg.replay_sample_window_sec)
    p.add_argument("--trivial-budget-tokens", type=int, default=256)
    p.add_argument("--puct-c", type=float, default=cfg.puct_c)
    p.add_argument("--uct-c", type=float, default=1.4)
    p.add_argument("--policy-prior-temperature", type=float, default=1.0)
    p.add_argument("--prior-min-prob", type=float, default=1e-8)
    p.add_argument("--root-dirichlet-noise-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--root-dirichlet-alpha", type=float, default=cfg.root_dirichlet_alpha)
    p.add_argument(
        "--root-dirichlet-total-concentration",
        type=float,
        default=cfg.root_dirichlet_total_concentration,
    )
    p.add_argument("--root-dirichlet-epsilon", type=float, default=cfg.root_dirichlet_epsilon)
    p.add_argument("--agz-sample-initial-moves", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--agz-sample-initial-move-count", type=int, default=cfg.agz_sample_initial_move_count)
    p.add_argument("--agz-mcts-action-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--worker-threads", type=int, default=1)
    p.add_argument("--native-search-mode", choices=("full_tree", "full_tree_rollout"), default=cfg.native_search_mode)
    p.add_argument("--rollout-count", type=int, default=cfg.rollout_count)
    p.add_argument("--rollout-parallel-threads", type=int, default=cfg.rollout_parallel_threads)
    p.add_argument("--rollout-horizon-sec", type=float, default=cfg.rollout_horizon_sec)
    p.add_argument("--rollout-policy-temperature", type=float, default=cfg.rollout_policy_temperature)
    p.add_argument("--rollout-probability-quantum", type=float, default=cfg.rollout_probability_quantum)
    p.add_argument("--rollout-max-actions", type=int, default=cfg.rollout_max_actions)
    p.add_argument("--parent-dataset-dir", default="")
    p.add_argument("--parent-state-id", type=int, default=-1)
    p.add_argument("--parent-state-count", type=int, default=0)
    p.add_argument("--parent-root-player-filter", choices=("controller", "adversary", "any"), default="any")
    p.add_argument("--upload-after-game", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--upload-poll-sec", type=float, default=30.0)
    p.add_argument("--flush-at-end", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ack-timeout-sec", type=int, default=0)
    p.add_argument("--continue-on-game-error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-consecutive-game-errors", type=int, default=120)
    p.add_argument("--game-error-backoff-sec", type=float, default=30.0)
    p.add_argument("--keep-game-runs", action=argparse.BooleanOptionalAction, default=False)
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> None:
    run_worker(parse_args())


if __name__ == "__main__":
    main()
