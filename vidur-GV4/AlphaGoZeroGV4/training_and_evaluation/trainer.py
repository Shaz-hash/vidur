"""Train one complete GV4 candidate bundle from indexed self-play replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, cast

from GV4_Engine.config import GV4EngineConfig

from ..dnn_models import DEFAULT_VALUE_SCALE, fit_policy_dnn, fit_value_dnn
from ..model_bundle import LoadedModelBundle, publish_model_bundle
from .indexed_replay import ReplayIndex
from .replay_dataset import Role, materialize_training_data


__all__ = ["TrainerConfig", "TrainingResult", "train_candidate"]


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Bounded sampling and optimizer settings for one AGZ training cycle."""

    max_roots_per_role: int = 250_000
    epochs: int = 5
    value_batch_size: int = 4096
    policy_root_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    huber_delta: float = 0.1
    value_scale: float = DEFAULT_VALUE_SCALE
    torch_threads: int = 24
    device: str | None = None
    seed: int = 2026

    def __post_init__(self) -> None:
        for name in (
            "max_roots_per_role",
            "epochs",
            "value_batch_size",
            "policy_root_batch_size",
            "torch_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("learning_rate", "huber_delta", "value_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative and finite")


@dataclass(slots=True)
class TrainingResult:
    """Published candidate plus compact metrics needed by the coordinator."""

    bundle: LoadedModelBundle
    parent_bundle_version: int
    candidate_bundle_version: int
    available_roots: dict[Role, int]
    sampled_roots: dict[Role, int]
    sampled_actions: dict[Role, int]
    metrics: dict[str, dict[str, float | int]]
    elapsed_sec: float

    def summary(self) -> dict[str, Any]:
        return {
            "parent_bundle_version": self.parent_bundle_version,
            "candidate_bundle_version": self.candidate_bundle_version,
            "candidate_manifest": str(self.bundle.manifest_path),
            "candidate_manifest_sha256": self.bundle.manifest_sha256,
            "available_roots": dict(self.available_roots),
            "sampled_roots": dict(self.sampled_roots),
            "sampled_actions": dict(self.sampled_actions),
            "metrics": self.metrics,
            "elapsed_sec": self.elapsed_sec,
        }


def _checkpoint(bundle: LoadedModelBundle, artifact_name: str) -> Path:
    artifact = bundle.artifacts[artifact_name]
    return bundle.manifest_path.parent / artifact.checkpoint_path


def train_candidate(
    replay_index: ReplayIndex,
    engine_config: GV4EngineConfig,
    incumbent: LoadedModelBundle,
    *,
    candidate_version: int,
    destination: str | Path,
    training: TrainerConfig = TrainerConfig(),
) -> TrainingResult:
    """Warm-start and publish controller/adversary value and policy models."""

    engine_config.validate()
    if replay_index.config.manifest_sha256() != engine_config.manifest_sha256():
        raise ValueError("replay index and trainer use different GV4 configs")
    if incumbent.config_manifest_sha256 != engine_config.manifest_sha256():
        raise ValueError("incumbent bundle and trainer use different GV4 configs")
    if candidate_version <= incumbent.bundle_version:
        raise ValueError("candidate version must be newer than the incumbent")

    started = time.perf_counter()
    models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    sampled_roots: dict[Role, int] = {}
    sampled_actions: dict[Role, int] = {}
    available = replay_index.role_counts

    for role_index, role_name in enumerate(("controller", "adversary")):
        role = cast(Role, role_name)
        roots = replay_index.sample(
            role,
            training.max_roots_per_role,
            seed=training.seed + role_index,
        )
        if not roots:
            raise ValueError(f"replay contains no {role} training roots")
        data = materialize_training_data(roots, role=role)
        sampled_roots[role] = data.root_count
        sampled_actions[role] = data.action_count

        value_name = f"{role}_value"
        value_model, value_metrics = fit_value_dnn(
            data.states,
            data.value_targets,
            config=engine_config,
            role=role,
            initial_model_path=_checkpoint(incumbent, value_name),
            seed=training.seed + 10 + role_index,
            epochs=training.epochs,
            batch_size=training.value_batch_size,
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
            torch_threads=training.torch_threads,
            huber_delta=training.huber_delta,
            value_scale=training.value_scale,
            device=training.device,
        )
        policy_name = f"{role}_policy"
        policy_model, policy_metrics = fit_policy_dnn(
            data.states,
            data.policy_action_features,
            data.policy_targets,
            data.policy_offsets,
            config=engine_config,
            role=role,
            initial_model_path=_checkpoint(incumbent, policy_name),
            seed=training.seed + 20 + role_index,
            epochs=training.epochs,
            root_batch_size=training.policy_root_batch_size,
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
            torch_threads=training.torch_threads,
            device=training.device,
        )
        models[value_name] = value_model
        models[policy_name] = policy_model
        metrics[value_name] = value_metrics
        metrics[policy_name] = policy_metrics

    destination = Path(destination).expanduser().resolve()
    bundle = publish_model_bundle(
        destination,
        bundle_version=candidate_version,
        config=engine_config,
        models=models,
        metadata={
            "eval_status": "candidate",
            "parent_bundle_version": incumbent.bundle_version,
            "parent_manifest_sha256": incumbent.manifest_sha256,
            "replay_index_dir": str(replay_index.index_dir),
            "replay_partitions": len(replay_index.manifests),
            "available_roots": available,
            "sampled_roots": sampled_roots,
            "sampled_actions": sampled_actions,
            "training_metrics": metrics,
        },
    )
    return TrainingResult(
        bundle=bundle,
        parent_bundle_version=incumbent.bundle_version,
        candidate_bundle_version=candidate_version,
        available_roots=available,
        sampled_roots=sampled_roots,
        sampled_actions=sampled_actions,
        metrics=metrics,
        elapsed_sec=time.perf_counter() - started,
    )
