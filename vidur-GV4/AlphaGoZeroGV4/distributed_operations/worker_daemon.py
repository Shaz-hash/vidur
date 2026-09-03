"""Supervise isolated GV4 self-play games and publish durable replay shards."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from GV4_Engine.config import GV4EngineConfig

from ..config import ExperimentConfig, ExperimentPaths, load_experiment_config
from ..model_bundle import load_model_bundle, sha256_file
from ..training_and_evaluation.replay_dataset import load_replay_partition
from .cluster import ClusterSpec, HostSpec, load_cluster
from .durable_transfer import (
    freeze_replay_shard,
    publish_replay_shard,
    read_ack,
    retire_acknowledged_shard,
    validate_replay_shard,
)


WORKER_STATE_SCHEMA_VERSION = "gv4_worker_state_v1"

__all__ = [
    "BundleSnapshot",
    "WorkerDaemon",
    "WorkerState",
    "build_game_command",
    "validate_game_output",
]


@dataclass(frozen=True, slots=True)
class BundleSnapshot:
    manifest_path: Path
    manifest_sha256: str
    bundle_version: int
    artifact_versions: dict[str, int]
    role_versions: dict[str, int]


@dataclass(slots=True)
class WorkerState:
    worker_id: str
    next_game_sequence: int = 0
    next_shard_sequence: int = 0
    games_completed: int = 0
    games_failed: int = 0
    roots_generated: int = 0
    published_at: dict[str, float] = field(default_factory=dict)
    last_error: str = ""


@dataclass(slots=True)
class _RunningGame:
    game_id: int
    output_dir: Path
    pid: int
    process: subprocess.Popen[str] | None = None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return value


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _state_payload(state: WorkerState) -> dict[str, Any]:
    return {"schema_version": WORKER_STATE_SCHEMA_VERSION, **asdict(state)}


def _load_state(path: Path, worker_id: str) -> WorkerState:
    if not path.exists():
        return WorkerState(worker_id=worker_id)
    value = _read_object(path)
    if value.pop("schema_version", None) != WORKER_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported worker state schema")
    state = WorkerState(**value)
    if state.worker_id != worker_id:
        raise ValueError("worker state belongs to another worker")
    if (
        min(
            state.next_game_sequence,
            state.next_shard_sequence,
            state.games_completed,
            state.games_failed,
            state.roots_generated,
        )
        < 0
    ):
        raise ValueError("worker state counters cannot be negative")
    return state


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def build_game_command(
    experiment: ExperimentConfig,
    *,
    python_executable: str,
    output_dir: Path,
    game_id: int,
    seed: int,
    history_hops: int,
    bundle: BundleSnapshot,
) -> list[str]:
    """Translate the frozen self-play config into one runner invocation."""

    play = experiment.self_play
    search = play.search
    entrypoint = Path(__file__).resolve().parents[1] / "process_entrypoint.py"
    return [
        python_executable,
        str(entrypoint),
        "AlphaGoZeroGV4.runner",
        "--engine-config-factory",
        experiment.engine_config_factory,
        "--backend",
        play.backend,
        "--output-dir",
        str(output_dir),
        "--game-id",
        str(game_id),
        "--cycle-label",
        "self_play",
        "--seed",
        str(seed),
        "--history-hops",
        str(history_hops),
        "--horizon-sec",
        str(play.horizon_sec),
        "--max-actions",
        str(play.max_actions),
        "--mcts-iterations",
        str(search.iterations),
        "--puct-c",
        str(search.puct_c),
        "--policy-prior-temperature",
        str(search.policy_prior_temperature),
        "--prior-min-probability",
        str(search.prior_min_probability),
        "--root-dirichlet-alpha",
        str(search.root_dirichlet_alpha),
        "--root-dirichlet-epsilon",
        str(search.root_dirichlet_epsilon),
        "--root-dirichlet-total-concentration",
        str(search.root_dirichlet_total_concentration),
        "--selection-temperature",
        str(play.selection_temperature),
        "--rollout-count",
        str(search.rollout_count),
        "--rollout-horizon-sec",
        str(search.rollout_horizon_sec),
        "--rollout-seed",
        str(search.rollout_seed + seed),
        "--rollout-policy-temperature",
        str(search.rollout_policy_temperature),
        "--rollout-probability-quantum",
        str(search.rollout_probability_quantum),
        "--rollout-max-actions",
        str(search.rollout_max_actions),
        "--model-bundle",
        str(bundle.manifest_path),
        "--model-device",
        play.model_device,
        _bool_flag("use-policy-prior", search.use_policy_prior),
        _bool_flag("use-model-bootstrap", search.use_model_bootstrap),
        "--bootstrap-mode",
        play.bootstrap_mode,
    ]


def validate_game_output(
    output_dir: str | Path,
    *,
    game_id: int,
    config: GV4EngineConfig,
    bundle: BundleSnapshot,
) -> int:
    """Validate process result, replay rows, model lineage, and game identity."""

    root = Path(output_dir).expanduser().resolve()
    result = _read_object(root / "game_result.json")
    if int(result.get("game_id", -1)) != game_id:
        raise ValueError("game result has the wrong game ID")
    metadata = result.get("engine_metadata")
    if not isinstance(metadata, Mapping) or (
        metadata.get("config_manifest_sha256") != config.manifest_sha256()
    ):
        raise ValueError("game result uses another engine config")
    if result.get("backend") not in {"python", "native"}:
        raise ValueError("game result has no valid backend")
    if result.get("model_versions") != bundle.role_versions:
        raise ValueError("game result changed model versions during play")

    partition = load_replay_partition(root / "replay_manifest.json", config)
    if partition.manifest.game_id != game_id:
        raise ValueError("replay manifest has the wrong game ID")
    if partition.manifest.state_rows <= 0:
        raise ValueError("self-play game produced no trainable roots")
    replay_manifest = _read_object(partition.manifest.manifest_path)
    if replay_manifest.get("model_versions") != bundle.artifact_versions:
        raise ValueError("replay model lineage differs from the launch snapshot")
    replay_result = result.get("replay")
    if not isinstance(replay_result, Mapping) or (
        replay_result.get("state_sha256") != partition.manifest.state_sha256
        or replay_result.get("action_sha256") != partition.manifest.action_sha256
    ):
        raise ValueError("game result and replay checksums disagree")
    return partition.manifest.state_rows


class WorkerDaemon:
    """One restart-safe worker supervisor; `run_once` is intentionally testable."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        cluster: ClusterSpec,
        worker: HostSpec,
        *,
        root: str | Path | None = None,
    ) -> None:
        if worker.role != "worker" or cluster.worker(worker.host_id) != worker:
            raise ValueError("worker is not part of this cluster")
        self.experiment = experiment
        self.engine_config = experiment.resolve_engine_config()
        self.cluster = cluster
        self.worker = worker
        self.paths = ExperimentPaths(Path(root or worker.experiment_root))
        self.worker_root = self.paths.worker(worker.host_id)
        self.runs = self.worker_root / "game_runs"
        self.replay_root = self.worker_root / "replay"
        self.active = self.replay_root / "active"
        self.ready = self.replay_root / "ready"
        self.failed = self.worker_root / "failed_games"
        self.completed = self.worker_root / "completed_games"
        self.state_path = self.worker_root / "worker_state.json"
        for path in (
            self.runs,
            self.ready,
            self.failed,
            self.completed,
            self.active / "games",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.state = _load_state(self.state_path, worker.host_id)
        self.running: dict[int, _RunningGame] = {}
        self._last_ack_poll_at = 0.0
        self._bundle_pointer_digest = ""
        self._bundle_snapshot: BundleSnapshot | None = None
        self._recover_runs()

    def _save_state(self) -> None:
        _atomic_json(self.state_path, _state_payload(self.state))

    def _current_bundle(self) -> BundleSnapshot:
        pointer = self.paths.current_model
        pointer_digest = sha256_file(pointer)
        if pointer_digest == self._bundle_pointer_digest and self._bundle_snapshot:
            return self._bundle_snapshot
        loaded = load_model_bundle(
            pointer,
            config=self.engine_config,
            device=self.experiment.self_play.model_device,
        )
        snapshot = BundleSnapshot(
            manifest_path=loaded.manifest_path,
            manifest_sha256=loaded.manifest_sha256,
            bundle_version=loaded.bundle_version,
            artifact_versions=loaded.model_versions,
            role_versions=loaded.role_versions,
        )
        self._bundle_pointer_digest = pointer_digest
        self._bundle_snapshot = snapshot
        return snapshot

    def _recover_runs(self) -> None:
        for output_dir in sorted(self.runs.glob("game_*")):
            launch_path = output_dir / "launch.json"
            try:
                launch = _read_object(launch_path)
                game_id = int(launch["game_id"])
                pid = int(launch["pid"])
            except (KeyError, TypeError, ValueError, OSError):
                self._reject_game(output_dir, "invalid launch record")
                continue
            if _pid_alive(pid):
                self.running[game_id] = _RunningGame(game_id, output_dir, pid)
            else:
                self._finish_game(_RunningGame(game_id, output_dir, pid))

    def _allocate_game(self) -> tuple[int, int, int]:
        sequence = self.state.next_game_sequence
        if sequence >= self.experiment.worker.game_id_stride:
            raise RuntimeError("worker exhausted its disjoint game-ID range")
        game_id = self.worker.ordinal * self.experiment.worker.game_id_stride + sequence
        self.state.next_game_sequence += 1
        self._save_state()
        seed = self.experiment.self_play.seed + game_id
        rng = random.Random(seed ^ 0x4756_3457)
        hops = rng.randint(
            self.experiment.self_play.history_hops_min,
            self.experiment.self_play.history_hops_max,
        )
        return game_id, seed, hops

    def _launch_game(self) -> _RunningGame:
        game_id, seed, history_hops = self._allocate_game()
        bundle = self._current_bundle()
        output_dir = self.runs / f"game_{game_id}"
        output_dir.mkdir()
        command = build_game_command(
            self.experiment,
            python_executable=self.worker.python_executable,
            output_dir=output_dir,
            game_id=game_id,
            seed=seed,
            history_hops=history_hops,
            bundle=bundle,
        )
        environment = os.environ.copy()
        environment.update(dict(self.experiment.deployment.environment))
        environment.update(
            {
                "OMP_NUM_THREADS": str(self.experiment.worker.process_threads),
                "MKL_NUM_THREADS": str(self.experiment.worker.process_threads),
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            self.worker.repo_root,
                            environment.get("PYTHONPATH", ""),
                        ),
                    )
                ),
            }
        )
        log_path = output_dir / "game_process.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=self.worker.repo_root,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _atomic_json(
            output_dir / "launch.json",
            {
                "game_id": game_id,
                "seed": seed,
                "history_hops": history_hops,
                "pid": process.pid,
                "command": command,
                "engine_config_sha256": self.engine_config.manifest_sha256(),
                "bundle_manifest": str(bundle.manifest_path),
                "bundle_manifest_sha256": bundle.manifest_sha256,
                "bundle_version": bundle.bundle_version,
                "artifact_versions": bundle.artifact_versions,
                "role_versions": bundle.role_versions,
            },
        )
        running = _RunningGame(game_id, output_dir, process.pid, process)
        self.running[game_id] = running
        return running

    def _snapshot_from_launch(self, output_dir: Path) -> BundleSnapshot:
        value = _read_object(output_dir / "launch.json")
        return BundleSnapshot(
            manifest_path=Path(value["bundle_manifest"]),
            manifest_sha256=str(value["bundle_manifest_sha256"]),
            bundle_version=int(value["bundle_version"]),
            artifact_versions={
                str(key): int(item)
                for key, item in dict(value["artifact_versions"]).items()
            },
            role_versions={
                str(key): int(item)
                for key, item in dict(value["role_versions"]).items()
            },
        )

    def _reject_game(self, output_dir: Path, reason: str) -> None:
        self.state.games_failed += 1
        self.state.last_error = reason
        if output_dir.exists():
            destination = self.failed / f"{output_dir.name}_{time.time_ns()}"
            os.replace(output_dir, destination)
            (destination / "failure.txt").write_text(reason + "\n", encoding="utf-8")
        self._save_state()

    def _finish_game(self, game: _RunningGame) -> None:
        try:
            bundle = self._snapshot_from_launch(game.output_dir)
            roots = validate_game_output(
                game.output_dir,
                game_id=game.game_id,
                config=self.engine_config,
                bundle=bundle,
            )
            destination = self.active / "games" / game.output_dir.name
            if destination.exists():
                raise FileExistsError(destination)
            if self.experiment.worker.keep_game_runs:
                shutil.copytree(
                    game.output_dir,
                    self.completed / game.output_dir.name,
                    dirs_exist_ok=True,
                )
            os.replace(game.output_dir, destination)
            self.state.games_completed += 1
            self.state.roots_generated += roots
            self.state.last_error = ""
            self._save_state()
        except Exception as error:
            self._reject_game(game.output_dir, str(error))

    def _reap_games(self) -> None:
        for game_id, game in tuple(self.running.items()):
            running = (
                game.process.poll() is None
                if game.process is not None
                else _pid_alive(game.pid)
            )
            if running:
                continue
            self.running.pop(game_id)
            self._finish_game(game)

    def _active_usage(self) -> tuple[int, int, int]:
        games = roots = total_bytes = 0
        for manifest_path in self.active.glob("games/game_*/replay_manifest.json"):
            value = _read_object(manifest_path)
            games += 1
            roots += int(value.get("state_rows", 0))
        for path in self.active.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size
        return games, roots, total_bytes

    def _freeze_if_needed(self, *, force: bool = False) -> None:
        games, roots, size = self._active_usage()
        if not games:
            return
        limits = self.experiment.worker
        if not force and (
            games < limits.shard_max_games
            and roots < limits.shard_max_roots
            and size < limits.shard_max_bytes
        ):
            return
        shard_id = f"shard_{self.state.next_shard_sequence:012d}"
        freeze_replay_shard(
            self.active,
            self.ready,
            worker_id=self.worker.host_id,
            shard_id=shard_id,
            config=self.engine_config,
        )
        self.state.next_shard_sequence += 1
        self._save_state()
        (self.active / "games").mkdir(parents=True)

    def _ready_shards(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in sorted(self.ready.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        )

    def _transfer_shards(self) -> None:
        coordinator = self.cluster.coordinator
        now = time.time()
        poll_acknowledgements = (
            now - self._last_ack_poll_at >= self.experiment.worker.ack_poll_sec
        )
        for path in self._ready_shards():
            shard = validate_replay_shard(path, self.engine_config)
            acknowledgement = (
                read_ack(
                    coordinator,
                    worker_id=shard.worker_id,
                    shard_id=shard.shard_id,
                    expected_digest=shard.digest,
                )
                if poll_acknowledgements
                else None
            )
            if acknowledgement is not None:
                retire_acknowledged_shard(path, coordinator, config=self.engine_config)
                self.state.published_at.pop(shard.shard_id, None)
                self._save_state()
                continue
            if not self.experiment.worker.upload_enabled:
                continue
            last_attempt = self.state.published_at.get(shard.shard_id, 0.0)
            if now - last_attempt < self.experiment.worker.retry_sec:
                continue
            publish_replay_shard(path, coordinator, config=self.engine_config)
            self.state.published_at[shard.shard_id] = now
            self._save_state()
        if poll_acknowledgements:
            self._last_ack_poll_at = now

    def run_once(self, *, allow_launch: bool = True) -> None:
        self._reap_games()
        self._freeze_if_needed()
        try:
            self._transfer_shards()
        except Exception as error:
            self.state.last_error = f"transfer: {error}"
            self._save_state()

        if not allow_launch:
            return
        ready_count = len(self._ready_shards())
        if ready_count >= self.experiment.worker.max_ready_shards:
            return
        while len(self.running) < self.experiment.worker.parallel_games:
            try:
                self._launch_game()
            except Exception as error:
                self.state.games_failed += 1
                self.state.last_error = f"launch: {error}"
                self._save_state()
                break

    def run_forever(self, *, max_game_attempts: int = 0) -> None:
        stop = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        final_sequence = (
            0
            if max_game_attempts <= 0
            else self.state.next_game_sequence + max_game_attempts
        )
        while not stop:
            allow_launch = not final_sequence or (
                self.state.next_game_sequence < final_sequence
            )
            self.run_once(allow_launch=allow_launch)
            if not allow_launch and not self.running:
                break
            time.sleep(self.experiment.worker.poll_sec)
        self._reap_games()
        self._freeze_if_needed(force=True)
        self._transfer_shards()

    def status(self) -> dict[str, Any]:
        games, roots, size = self._active_usage()
        return {
            **_state_payload(self.state),
            "running_game_ids": sorted(self.running),
            "active_games": games,
            "active_roots": roots,
            "active_bytes": size,
            "ready_shards": [path.name for path in self._ready_shards()],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-game-attempts", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    experiment = load_experiment_config(args.experiment_config)
    cluster = load_cluster(args.cluster)
    daemon = WorkerDaemon(
        experiment,
        cluster,
        cluster.worker(args.worker_id),
        root=args.root,
    )
    if args.status:
        print(json.dumps(daemon.status(), indent=2, sort_keys=True))
        return
    if args.max_game_attempts < 0:
        raise ValueError("max-game-attempts cannot be negative")
    daemon.run_forever(max_game_attempts=args.max_game_attempts)


if __name__ == "__main__":
    main()
