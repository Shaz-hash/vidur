"""Ingest GV4 replay and supervise one train/evaluate/promote subprocess."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from GV4_Engine.config import GV4EngineConfig

from ..config import ExperimentConfig, ExperimentPaths, load_experiment_config
from ..training_and_evaluation.agz_train_eval_promote import (
    ArenaEvaluator,
    TrainEvalPromoteConfig,
    run_train_eval_promote,
)
from .cluster import ClusterSpec, load_cluster
from .durable_transfer import (
    ValidatedReplayShard,
    sync_current_model,
    validate_replay_shard,
    write_ack,
)


COORDINATOR_STATE_SCHEMA_VERSION = "gv4_coordinator_state_v1"

__all__ = [
    "Coordinator",
    "CoordinatorState",
    "TrainingGate",
    "run_training_cycle",
]


@dataclass(slots=True)
class CoordinatorState:
    accepted: dict[str, dict[str, Any]] = field(default_factory=dict)
    next_accept_sequence: int = 0
    roots_since_training: int = 0
    cycles_completed: int = 0
    training: dict[str, Any] | None = None
    last_error: str = ""


@dataclass(frozen=True, slots=True)
class TrainingGate:
    ready: bool
    reason: str
    total_roots: int
    controller_roots: int
    adversary_roots: int
    roots_since_training: int


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


def _load_state(path: Path) -> CoordinatorState:
    if not path.exists():
        return CoordinatorState()
    value = _read_object(path)
    if value.pop("schema_version", None) != COORDINATOR_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported coordinator state schema")
    state = CoordinatorState(**value)
    if (
        min(
            state.next_accept_sequence,
            state.roots_since_training,
            state.cycles_completed,
        )
        < 0
    ):
        raise ValueError("coordinator counters cannot be negative")
    return state


def _state_payload(state: CoordinatorState) -> dict[str, Any]:
    return {"schema_version": COORDINATOR_STATE_SCHEMA_VERSION, **asdict(state)}


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


def _entry(shard: ValidatedReplayShard, path: Path, sequence: int) -> dict[str, Any]:
    return {
        "worker_id": shard.worker_id,
        "shard_id": shard.shard_id,
        "digest": shard.digest,
        "path": str(path),
        "status": "active",
        "accept_sequence": sequence,
        "games": shard.games,
        "state_rows": shard.state_rows,
        "action_rows": shard.action_rows,
        "controller_roots": shard.controller_roots,
        "adversary_roots": shard.adversary_roots,
        "created_at_utc": shard.created_at_utc,
    }


def run_training_cycle(
    experiment: ExperimentConfig,
    *,
    root: str | Path,
    status_path: str | Path,
    cluster: ClusterSpec | None = None,
    experiment_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Child-process entry point that owns exactly one training cycle."""

    paths = ExperimentPaths(Path(root))
    status = Path(status_path).expanduser().resolve()
    try:
        engine_config = experiment.resolve_engine_config()
        experiment_engine_hash = engine_config.manifest_sha256()
        arena_evaluator: ArenaEvaluator | None = None
        if experiment.distributed_evaluation.enabled:
            if cluster is None or experiment_config_path is None:
                raise ValueError(
                    "distributed evaluation requires cluster and config manifests"
                )
            from .distributed_eval import evaluate_candidate_distributed

            def distributed_arena(
                engine_config: GV4EngineConfig,
                incumbent: Any,
                candidate: Any,
                arena: Any,
                _timing_provider_factory: Any,
                _logger: Any,
            ) -> Any:
                if engine_config.manifest_sha256() != experiment_engine_hash:
                    raise ValueError(
                        "training and distributed arena engine configs differ"
                    )
                if arena != experiment.arena:
                    raise ValueError("training and distributed arena settings differ")
                return evaluate_candidate_distributed(
                    experiment,
                    cluster,
                    incumbent,
                    candidate,
                    experiment_config_path=experiment_config_path,
                    output_dir=status.parent / "output" / "distributed_arena",
                )

            arena_evaluator = distributed_arena
        result = run_train_eval_promote(
            engine_config,
            TrainEvalPromoteConfig(
                replay_root=paths.global_replay,
                replay_index_dir=paths.coordinator / "replay_index",
                models_root=paths.models,
                current_model_path=paths.current_model,
                output_dir=status.parent / "output",
                trainer=experiment.trainer,
                arena=experiment.arena,
                promotion=experiment.promotion,
                sjf=experiment.sjf,
            ),
            arena_evaluator=arena_evaluator,
        )
        payload = {"status": "complete", "result": result.summary()}
    except Exception as error:
        payload = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    _atomic_json(status, payload)
    return payload


class Coordinator:
    """Restart-safe ingestion and training coordinator with a testable `run_once`."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        cluster: ClusterSpec,
        *,
        root: str | Path | None = None,
        config_path: str | Path | None = None,
        cluster_path: str | Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.engine_config: GV4EngineConfig = experiment.resolve_engine_config()
        self.cluster = cluster
        self.host = cluster.coordinator
        self.paths = ExperimentPaths(Path(root or self.host.experiment_root))
        self.config_path = Path(config_path or self.paths.config_manifest).resolve()
        self.cluster_path = Path(cluster_path or self.paths.cluster_manifest).resolve()
        self.state_path = self.paths.coordinator / "coordinator_state.json"
        self.gate_path = self.paths.coordinator / "training_gate.json"
        self.partitions = self.paths.global_replay / "partitions"
        self.rejected = self.paths.coordinator / "rejected"
        for path in (
            self.paths.incoming,
            self.paths.incoming_uploading,
            self.paths.acknowledgements,
            self.partitions,
            self.rejected,
            self.paths.training_output,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.state = _load_state(self.state_path)
        self.training_process: subprocess.Popen[str] | None = None
        self._reconcile_partitions()

    def _save_state(self) -> None:
        _atomic_json(self.state_path, _state_payload(self.state))

    def _active_entries(self) -> list[dict[str, Any]]:
        return [
            value
            for value in self.state.accepted.values()
            if value.get("status") == "active"
        ]

    def _role_counts(self) -> tuple[int, int]:
        active = self._active_entries()
        return (
            sum(int(item["controller_roots"]) for item in active),
            sum(int(item["adversary_roots"]) for item in active),
        )

    def _reconcile_partitions(self) -> None:
        changed = False
        for path in sorted(self.partitions.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            shard = validate_replay_shard(path, self.engine_config)
            existing = self.state.accepted.get(shard.identity)
            if existing is not None:
                if existing.get("digest") != shard.digest:
                    raise ValueError("accepted shard identity changed digest")
                continue
            record = _entry(shard, path, self.state.next_accept_sequence)
            self.state.next_accept_sequence += 1
            self.state.accepted[shard.identity] = record
            self.state.roots_since_training += shard.state_rows
            write_ack(self.paths.root, shard, accepted_path=str(path))
            changed = True
        if changed:
            self._save_state()

    def _already_accepted(self, shard: ValidatedReplayShard) -> bool:
        record = self.state.accepted.get(shard.identity)
        if record is None:
            return False
        if record.get("digest") != shard.digest:
            raise ValueError("duplicate shard identity has another digest")
        write_ack(
            self.paths.root,
            shard,
            accepted_path=str(record.get("path", "pruned")),
        )
        return True

    def _ingest(self, incoming: Path) -> bool:
        shard = validate_replay_shard(incoming, self.engine_config)
        if self._already_accepted(shard):
            shutil.rmtree(incoming)
            return False
        destination = self.partitions / f"{shard.worker_id}__{shard.shard_id}"
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(incoming, destination)
        accepted = replace(shard, path=destination)
        self.state.accepted[accepted.identity] = _entry(
            accepted, destination, self.state.next_accept_sequence
        )
        self.state.next_accept_sequence += 1
        self.state.roots_since_training += accepted.state_rows
        self.state.last_error = ""
        self._save_state()
        write_ack(self.paths.root, accepted, accepted_path=str(destination))
        return True

    def ingest_once(self) -> int:
        accepted = 0
        processed = 0
        maximum = self.experiment.coordinator.max_ingest_per_pass
        for incoming in sorted(self.paths.incoming.iterdir()):
            if not incoming.is_dir() or incoming.name.startswith("."):
                continue
            if maximum and processed >= maximum:
                break
            processed += 1
            try:
                accepted += int(self._ingest(incoming))
            except Exception as error:
                self.state.last_error = f"ingest {incoming.name}: {error}"
                self._save_state()
                destination = self.rejected / f"{incoming.name}_{time.time_ns()}"
                os.replace(incoming, destination)
                (destination / "rejection.txt").write_text(
                    str(error) + "\n", encoding="utf-8"
                )
        return accepted

    def training_gate(self) -> TrainingGate:
        controller, adversary = self._role_counts()
        total = controller + adversary
        limits = self.experiment.coordinator
        if self.state.training is not None:
            reason = "training cycle is already running"
        elif not limits.training_enabled:
            reason = "training is disabled"
        elif self.state.roots_since_training < limits.train_trigger_new_roots:
            reason = "not enough new replay"
        elif controller < limits.min_controller_roots:
            reason = "not enough controller replay"
        elif adversary < limits.min_adversary_roots:
            reason = "not enough adversary replay"
        elif not self.paths.current_model.is_file():
            reason = "current model pointer is missing"
        else:
            reason = "ready"
        gate = TrainingGate(
            ready=reason == "ready",
            reason=reason,
            total_roots=total,
            controller_roots=controller,
            adversary_roots=adversary,
            roots_since_training=self.state.roots_since_training,
        )
        _atomic_json(self.gate_path, asdict(gate))
        return gate

    def _training_command(self, status_path: Path) -> list[str]:
        entrypoint = f"{self.host.package_root}/AlphaGoZeroGV4/process_entrypoint.py"
        return [
            self.host.python_executable,
            entrypoint,
            "AlphaGoZeroGV4.distributed_operations.xl_coordinator",
            "--experiment-config",
            str(self.config_path),
            "--cluster",
            str(self.cluster_path),
            "--root",
            str(self.paths.root),
            "--run-training-cycle",
            "--training-status",
            str(status_path),
        ]

    def _launch_training(self) -> None:
        cycle = self.state.cycles_completed + 1
        cycle_dir = self.paths.training_output / f"cycle_{cycle:06d}_{time.time_ns()}"
        cycle_dir.mkdir(parents=True)
        status_path = cycle_dir / "cycle_status.json"
        log_path = cycle_dir / "cycle_process.log"
        command = self._training_command(status_path)
        environment = os.environ.copy()
        environment.update(dict(self.experiment.deployment.environment))
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (
                    self.host.repo_root,
                    environment.get("PYTHONPATH", ""),
                ),
            )
        )
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=self.host.repo_root,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.training_process = process
        self.state.training = {
            "pid": process.pid,
            "status_path": str(status_path),
            "log_path": str(log_path),
            "roots_at_start": self.state.roots_since_training,
            "started_at": time.time(),
        }
        self._save_state()

    def _broadcast_current_model(self) -> None:
        if not self.experiment.coordinator.broadcast_promotions:
            return
        errors: list[str] = []
        for worker in self.cluster.workers:
            try:
                sync_current_model(
                    self.paths.current_model,
                    worker,
                    config=self.engine_config,
                )
            except Exception as error:
                errors.append(f"{worker.host_id}: {error}")
        if errors:
            self.state.last_error = "model broadcast: " + "; ".join(errors)

    def _reconcile_training(self) -> None:
        record = self.state.training
        if record is None:
            return
        process_running = (
            self.training_process.poll() is None
            if self.training_process is not None
            else _pid_alive(int(record.get("pid", -1)))
        )
        status_path = Path(str(record["status_path"]))
        if process_running and not status_path.exists():
            return
        if not status_path.exists():
            self.state.last_error = "training process exited without a status file"
            self.state.training = None
            self.training_process = None
            self._save_state()
            return
        status = _read_object(status_path)
        if status.get("status") == "complete":
            consumed = int(record.get("roots_at_start", 0))
            self.state.roots_since_training = max(
                0, self.state.roots_since_training - consumed
            )
            self.state.cycles_completed += 1
            self.state.last_error = ""
            self._broadcast_current_model()
        else:
            self.state.last_error = f"training failed: {status.get('error', 'unknown')}"
        self.state.training = None
        self.training_process = None
        self._save_state()

    def _prune(self) -> int:
        if self.state.training is not None:
            return 0
        active = sorted(
            self._active_entries(), key=lambda item: int(item["accept_sequence"])
        )
        controller, adversary = self._role_counts()
        limit = self.experiment.coordinator.max_replay_roots
        removed = 0
        while controller + adversary > limit and active:
            victim_index = next(
                (
                    index
                    for index, item in enumerate(active)
                    if controller - int(item["controller_roots"])
                    >= self.experiment.coordinator.min_controller_roots
                    and adversary - int(item["adversary_roots"])
                    >= self.experiment.coordinator.min_adversary_roots
                ),
                None,
            )
            if victim_index is None:
                break
            victim = active.pop(victim_index)
            path = Path(victim["path"])
            if path.is_dir():
                shutil.rmtree(path)
            victim["status"] = "pruned"
            controller -= int(victim["controller_roots"])
            adversary -= int(victim["adversary_roots"])
            removed += 1
        if removed:
            self._save_state()
        return removed

    def run_once(self, *, allow_training: bool = True) -> TrainingGate:
        self._reconcile_training()
        self.ingest_once()
        self._prune()
        gate = self.training_gate()
        if gate.ready and allow_training:
            self._launch_training()
            gate = self.training_gate()
        return gate

    def run_forever(self, *, max_cycles: int = 0) -> None:
        stop = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        target = 0 if max_cycles <= 0 else self.state.cycles_completed + max_cycles
        while not stop:
            allow_training = not target or self.state.cycles_completed < target
            self.run_once(allow_training=allow_training)
            if target and self.state.cycles_completed >= target:
                break
            time.sleep(self.experiment.coordinator.poll_sec)

    def status(self) -> dict[str, Any]:
        return {
            **_state_payload(self.state),
            "training_gate": asdict(self.training_gate()),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--run-training-cycle", action="store_true")
    parser.add_argument("--training-status", type=Path)
    parser.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    experiment = load_experiment_config(args.experiment_config)
    cluster = load_cluster(args.cluster)
    root = args.root or cluster.coordinator.experiment_root
    if args.run_training_cycle:
        if args.training_status is None:
            raise ValueError("--training-status is required for a training child")
        payload = run_training_cycle(
            experiment,
            root=root,
            status_path=args.training_status,
            cluster=cluster,
            experiment_config_path=args.experiment_config,
        )
        if payload["status"] != "complete":
            raise SystemExit(1)
        return
    coordinator = Coordinator(
        experiment,
        cluster,
        root=root,
        config_path=args.experiment_config,
        cluster_path=args.cluster,
    )
    if args.status:
        print(json.dumps(coordinator.status(), indent=2, sort_keys=True))
        return
    if args.max_cycles < 0:
        raise ValueError("max-cycles cannot be negative")
    coordinator.run_forever(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
