"""Portable, validated configuration for the GV4 AlphaGoZero pipeline.

Game semantics remain in :mod:`GV4_Engine.config`.  This module owns only the
orchestration choices around that engine: self-play, replay transport, training,
evaluation, promotion, and deployment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

from GV4_Engine.config import GV4EngineConfig

from .engine_runtime import BackendName, SearchConfig
from .training_and_evaluation.arena import ArenaConfig
from .training_and_evaluation.baselines import SJFPolicyConfig
from .training_and_evaluation.promotion import PromotionPolicy
from .training_and_evaluation.sjf_runner import SJFBenchmarkConfig
from .training_and_evaluation.trainer import TrainerConfig


EXPERIMENT_CONFIG_SCHEMA_VERSION = "gv4_agz_experiment_v1"
_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

__all__ = [
    "CoordinatorConfig",
    "DeploymentConfig",
    "DistributedEvaluationConfig",
    "EXPERIMENT_CONFIG_SCHEMA_VERSION",
    "ExperimentConfig",
    "ExperimentPaths",
    "SelfPlayConfig",
    "WorkerConfig",
    "load_experiment_config",
    "resolve_engine_config",
    "write_experiment_config",
]


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_float(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Settings used by every isolated self-play game process."""

    backend: BackendName = "native"
    search: SearchConfig = field(
        default_factory=lambda: SearchConfig(
            iterations=1_000,
            puct_c=2.5,
            use_policy_prior=True,
            use_model_bootstrap=True,
        )
    )
    seed: int = 2026
    history_hops_min: int = 0
    history_hops_max: int = 20
    horizon_sec: float = 20.0
    max_actions: int = 100_000
    selection_temperature: float = 1.0
    bootstrap_mode: str = "model"
    model_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.backend not in {"python", "native"}:
            raise ValueError("self_play.backend must be python or native")
        _nonnegative_int("self_play.seed", self.seed)
        _nonnegative_int("self_play.history_hops_min", self.history_hops_min)
        _nonnegative_int("self_play.history_hops_max", self.history_hops_max)
        if self.history_hops_min > self.history_hops_max:
            raise ValueError("history_hops_min cannot exceed history_hops_max")
        _positive_float("self_play.horizon_sec", self.horizon_sec)
        _positive_int("self_play.max_actions", self.max_actions)
        if (
            not math.isfinite(self.selection_temperature)
            or self.selection_temperature < 0.0
        ):
            raise ValueError("selection_temperature must be nonnegative and finite")
        if self.bootstrap_mode not in {"neutral_zero", "model"}:
            raise ValueError("bootstrap_mode must be neutral_zero or model")
        if not self.model_device.strip():
            raise ValueError("model_device cannot be empty")


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Process supervision, local shard, and upload limits for one worker."""

    parallel_games: int = 1
    process_threads: int = 1
    game_id_stride: int = 1_000_000_000
    shard_max_games: int = 64
    shard_max_roots: int = 100_000
    shard_max_bytes: int = 1 << 30
    max_ready_shards: int = 4
    poll_sec: float = 1.0
    retry_sec: float = 10.0
    ack_poll_sec: float = 5.0
    upload_enabled: bool = True
    keep_game_runs: bool = False

    def __post_init__(self) -> None:
        for name in (
            "parallel_games",
            "process_threads",
            "game_id_stride",
            "shard_max_games",
            "shard_max_roots",
            "shard_max_bytes",
            "max_ready_shards",
        ):
            _positive_int(f"worker.{name}", getattr(self, name))
        for name in ("poll_sec", "retry_sec", "ack_poll_sec"):
            _positive_float(f"worker.{name}", getattr(self, name))


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    """Replay bounds and the gate for launching one training cycle."""

    max_replay_roots: int = 35_000_000
    train_trigger_new_roots: int = 600_000
    min_controller_roots: int = 250_000
    min_adversary_roots: int = 25_000
    max_ingest_per_pass: int = 0
    poll_sec: float = 2.0
    training_enabled: bool = True
    broadcast_promotions: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_replay_roots",
            "train_trigger_new_roots",
            "min_controller_roots",
            "min_adversary_roots",
        ):
            _positive_int(f"coordinator.{name}", getattr(self, name))
        _nonnegative_int("coordinator.max_ingest_per_pass", self.max_ingest_per_pass)
        _positive_float("coordinator.poll_sec", self.poll_sec)
        if self.min_controller_roots + self.min_adversary_roots > self.max_replay_roots:
            raise ValueError("minimum role replay exceeds max_replay_roots")


@dataclass(frozen=True, slots=True)
class DistributedEvaluationConfig:
    """How a paired arena is divided across fixed evaluation hosts."""

    enabled: bool = False
    games_per_chunk: int = 10
    max_parallel_chunks_per_host: int = 1
    keep_remote_outputs: bool = False

    def __post_init__(self) -> None:
        _positive_int("distributed_eval.games_per_chunk", self.games_per_chunk)
        _positive_int(
            "distributed_eval.max_parallel_chunks_per_host",
            self.max_parallel_chunks_per_host,
        )


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Remote command defaults; concrete hosts live in the cluster manifest."""

    native_build_jobs: int = 4
    environment: tuple[tuple[str, str], ...] = (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
    )

    def __post_init__(self) -> None:
        _positive_int("deployment.native_build_jobs", self.native_build_jobs)
        names: set[str] = set()
        for name, value in self.environment:
            if not name or "=" in name or name in names:
                raise ValueError("deployment environment names must be unique")
            if not isinstance(value, str):
                raise ValueError("deployment environment values must be strings")
            names.add(name)


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Conventional paths below one host-specific experiment root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def config_manifest(self) -> Path:
        return self.root / "experiment_config.json"

    @property
    def cluster_manifest(self) -> Path:
        return self.root / "cluster.json"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def current_model(self) -> Path:
        return self.models / "current_model.json"

    @property
    def coordinator(self) -> Path:
        return self.root / "coordinator"

    @property
    def global_replay(self) -> Path:
        return self.coordinator / "global_replay"

    @property
    def incoming(self) -> Path:
        return self.coordinator / "incoming"

    @property
    def incoming_uploading(self) -> Path:
        return self.coordinator / "incoming_uploading"

    @property
    def acknowledgements(self) -> Path:
        return self.coordinator / "acks"

    @property
    def training_output(self) -> Path:
        return self.coordinator / "training"

    def worker(self, worker_id: str) -> Path:
        if not _EXPERIMENT_ID.fullmatch(worker_id):
            raise ValueError("worker_id is not path-safe")
        return self.root / "workers" / worker_id


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Single source of truth for one GV4 AlphaGoZero experiment."""

    experiment_id: str
    engine_config_factory: str
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)
    sjf: SJFBenchmarkConfig | None = None
    distributed_evaluation: DistributedEvaluationConfig = field(
        default_factory=DistributedEvaluationConfig
    )
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def __post_init__(self) -> None:
        if not _EXPERIMENT_ID.fullmatch(self.experiment_id):
            raise ValueError(
                "experiment_id must contain only letters, digits, '.', '_', or '-'"
            )
        module, separator, function = self.engine_config_factory.partition(":")
        if not separator or not module or not function:
            raise ValueError("engine_config_factory must have form module:function")
        if self.arena.games < self.promotion.min_games:
            raise ValueError("arena.games must cover promotion.min_games")

    def resolve_engine_config(self) -> GV4EngineConfig:
        config = resolve_engine_config(self.engine_config_factory)
        self.validate(config)
        return config

    def validate(self, engine_config: GV4EngineConfig) -> None:
        engine_config.validate()
        if engine_config.topology.num_replicas != 1:
            raise ValueError("initial GV4 AlphaGoZero supports exactly one replica")
        if self.self_play.bootstrap_mode == "model" and not (
            self.self_play.search.use_model_bootstrap
        ):
            raise ValueError("model bootstrap mode requires model-backed MCTS leaves")
        if not self.self_play.search.use_policy_prior:
            raise ValueError("AlphaGoZero self-play requires policy priors")
        if not self.arena.search.use_policy_prior or not (
            self.arena.search.use_model_bootstrap
        ):
            raise ValueError("arena evaluation requires policy and value models")
        if self.sjf is not None and not self.sjf.search.use_model_bootstrap:
            raise ValueError("SJF evaluation requires a model adversary value")

    def paths(self, root: str | Path) -> ExperimentPaths:
        return ExperimentPaths(Path(root))

    def to_manifest_dict(
        self, engine_config: GV4EngineConfig | None = None
    ) -> dict[str, Any]:
        engine = engine_config or self.resolve_engine_config()
        self.validate(engine)
        return {
            "schema_version": EXPERIMENT_CONFIG_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "engine_config_factory": self.engine_config_factory,
            "engine_config_sha256": engine.manifest_sha256(),
            "engine_manifest_schema_version": (engine.layout.manifest_schema_version),
            "settings": _plain(
                {
                    "self_play": asdict(self.self_play),
                    "worker": asdict(self.worker),
                    "coordinator": asdict(self.coordinator),
                    "trainer": asdict(self.trainer),
                    "arena": asdict(self.arena),
                    "promotion": asdict(self.promotion),
                    "sjf": None if self.sjf is None else asdict(self.sjf),
                    "distributed_evaluation": asdict(self.distributed_evaluation),
                    "deployment": asdict(self.deployment),
                }
            ),
        }

    def manifest_sha256(self, engine_config: GV4EngineConfig | None = None) -> str:
        payload = json.dumps(
            self.to_manifest_dict(engine_config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def resolve_engine_config(factory_specification: str) -> GV4EngineConfig:
    """Call one zero-argument engine factory and validate its result."""

    module_name, separator, function_name = factory_specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("engine config factory must have form module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    value = factory()
    if not isinstance(value, GV4EngineConfig):
        raise TypeError("engine config factory did not return GV4EngineConfig")
    value.validate()
    return value


def write_experiment_config(
    path: str | Path,
    config: ExperimentConfig,
    *,
    engine_config: GV4EngineConfig | None = None,
) -> Path:
    """Atomically publish a portable experiment manifest."""

    destination = Path(path).expanduser().resolve()
    _atomic_json(destination, config.to_manifest_dict(engine_config))
    return destination


def _search(value: Mapping[str, Any]) -> SearchConfig:
    return SearchConfig(**dict(value))


def _arena(value: Mapping[str, Any]) -> ArenaConfig:
    fields = dict(value)
    fields["search"] = _search(fields["search"])
    return ArenaConfig(**fields)


def _sjf(value: Mapping[str, Any]) -> SJFBenchmarkConfig:
    fields = dict(value)
    fields["search"] = _search(fields["search"])
    fields["policy"] = SJFPolicyConfig(**dict(fields["policy"]))
    return SJFBenchmarkConfig(**fields)


def load_experiment_config(
    path: str | Path,
    *,
    verify_engine: bool = True,
) -> ExperimentConfig:
    """Load the manifest and fail if its engine factory changed underneath it."""

    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read experiment config {source}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("experiment config must be a JSON object")
    if value.get("schema_version") != EXPERIMENT_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported GV4 AlphaGoZero experiment schema")
    settings = value.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("experiment config has no settings object")

    self_play = dict(settings["self_play"])
    self_play["search"] = _search(self_play["search"])
    deployment = dict(settings["deployment"])
    deployment["environment"] = tuple(
        tuple(item) for item in deployment.get("environment", ())
    )
    raw_sjf = settings.get("sjf")
    result = ExperimentConfig(
        experiment_id=str(value.get("experiment_id", "")),
        engine_config_factory=str(value.get("engine_config_factory", "")),
        self_play=SelfPlayConfig(**self_play),
        worker=WorkerConfig(**dict(settings["worker"])),
        coordinator=CoordinatorConfig(**dict(settings["coordinator"])),
        trainer=TrainerConfig(**dict(settings["trainer"])),
        arena=_arena(settings["arena"]),
        promotion=PromotionPolicy(**dict(settings["promotion"])),
        sjf=None if raw_sjf is None else _sjf(raw_sjf),
        distributed_evaluation=DistributedEvaluationConfig(
            **dict(settings["distributed_evaluation"])
        ),
        deployment=DeploymentConfig(**deployment),
    )
    if verify_engine:
        engine = result.resolve_engine_config()
        expected = str(value.get("engine_config_sha256", ""))
        if engine.manifest_sha256() != expected:
            raise ValueError("engine factory no longer matches experiment manifest")
        if engine.layout.manifest_schema_version != value.get(
            "engine_manifest_schema_version"
        ):
            raise ValueError("engine manifest schema changed")
    return result
