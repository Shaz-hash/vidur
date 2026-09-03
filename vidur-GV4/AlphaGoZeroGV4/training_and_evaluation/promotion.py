"""Decide and atomically publish independent GV4 role promotions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

from GV4_Engine.config import GV4EngineConfig

from ..model_bundle import (
    LoadedModelBundle,
    load_model_bundle,
    publish_model_bundle,
    write_current_model_pointer,
)
from .arena import ArenaResult, RoleArenaStats


__all__ = [
    "PromotionDecision",
    "PromotionPolicy",
    "PromotionResult",
    "decide_promotion",
    "publish_promotion",
]


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Minimum paired evidence required for either role to advance."""

    min_games: int = 100
    controller_min_score_rate: float = 0.55
    adversary_min_score_rate: float = 0.55
    minimum_mean_improvement: float = 0.0

    def __post_init__(self) -> None:
        if self.min_games <= 0:
            raise ValueError("min_games must be positive")
        for name in ("controller_min_score_rate", "adversary_min_score_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.minimum_mean_improvement):
            raise ValueError("minimum_mean_improvement must be finite")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    incumbent_bundle_version: int
    candidate_bundle_version: int
    promote_controller: bool
    promote_adversary: bool
    controller_reason: str
    adversary_reason: str


@dataclass(slots=True)
class PromotionResult:
    decision: PromotionDecision
    promoted_bundle: LoadedModelBundle
    current_model_path: Path
    record_path: Path
    created_composite_bundle: bool

    def summary(self) -> dict[str, Any]:
        return {
            "decision": asdict(self.decision),
            "promoted_bundle_version": self.promoted_bundle.bundle_version,
            "promoted_manifest": str(self.promoted_bundle.manifest_path),
            "promoted_manifest_sha256": self.promoted_bundle.manifest_sha256,
            "promoted_role_versions": self.promoted_bundle.role_versions,
            "current_model_path": str(self.current_model_path),
            "record_path": str(self.record_path),
            "created_composite_bundle": self.created_composite_bundle,
        }


def _role_decision(
    stats: RoleArenaStats,
    *,
    min_games: int,
    min_score_rate: float,
    minimum_mean_improvement: float,
) -> tuple[bool, str]:
    if stats.games < min_games:
        return False, f"only {stats.games} paired games; need {min_games}"
    if stats.score_rate < min_score_rate:
        return False, f"score rate {stats.score_rate:.6g} is below {min_score_rate:.6g}"
    if stats.mean_improvement < minimum_mean_improvement:
        return (
            False,
            f"mean improvement {stats.mean_improvement:.6g} is below "
            f"{minimum_mean_improvement:.6g}",
        )
    return True, "arena thresholds passed"


def decide_promotion(
    arena: ArenaResult,
    policy: PromotionPolicy = PromotionPolicy(),
) -> PromotionDecision:
    """Make controller and adversary decisions from their isolated matchups."""

    controller, controller_reason = _role_decision(
        arena.controller,
        min_games=policy.min_games,
        min_score_rate=policy.controller_min_score_rate,
        minimum_mean_improvement=policy.minimum_mean_improvement,
    )
    adversary, adversary_reason = _role_decision(
        arena.adversary,
        min_games=policy.min_games,
        min_score_rate=policy.adversary_min_score_rate,
        minimum_mean_improvement=policy.minimum_mean_improvement,
    )
    return PromotionDecision(
        incumbent_bundle_version=arena.incumbent_bundle_version,
        candidate_bundle_version=arena.candidate_bundle_version,
        promote_controller=controller,
        promote_adversary=adversary,
        controller_reason=controller_reason,
        adversary_reason=adversary_reason,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _composite_bundle(
    models_root: Path,
    config: GV4EngineConfig,
    incumbent: LoadedModelBundle,
    candidate: LoadedModelBundle,
    decision: PromotionDecision,
) -> tuple[LoadedModelBundle, bool]:
    controller = candidate if decision.promote_controller else incumbent
    adversary = candidate if decision.promote_adversary else incumbent
    versions = {
        "controller_value": controller.artifacts["controller_value"].model_version,
        "controller_policy": controller.artifacts["controller_policy"].model_version,
        "adversary_value": adversary.artifacts["adversary_value"].model_version,
        "adversary_policy": adversary.artifacts["adversary_policy"].model_version,
    }
    models = {
        "controller_value": controller.models["controller_value"],
        "controller_policy": controller.models["controller_policy"],
        "adversary_value": adversary.models["adversary_value"],
        "adversary_policy": adversary.models["adversary_policy"],
    }
    destination = models_root / (
        f"Model_Version{candidate.bundle_version}_"
        f"C{versions['controller_value']}_A{versions['adversary_value']}"
    )
    if destination.exists():
        loaded = load_model_bundle(destination, config=config)
        if loaded.model_versions != versions:
            raise FileExistsError(
                f"incompatible composite bundle exists: {destination}"
            )
        return loaded, False
    return (
        publish_model_bundle(
            destination,
            bundle_version=candidate.bundle_version,
            config=config,
            models=models,
            model_versions=versions,
            metadata={
                "eval_status": "promoted_composite",
                "incumbent_manifest_sha256": incumbent.manifest_sha256,
                "candidate_manifest_sha256": candidate.manifest_sha256,
                "promotion_decision": asdict(decision),
            },
        ),
        True,
    )


def publish_promotion(
    models_root: str | Path,
    config: GV4EngineConfig,
    incumbent: LoadedModelBundle,
    candidate: LoadedModelBundle,
    decision: PromotionDecision,
    *,
    current_model_path: str | Path | None = None,
    record_path: str | Path | None = None,
) -> PromotionResult:
    """Publish one complete four-model view, never a half-updated role pair."""

    root = Path(models_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if decision.incumbent_bundle_version != incumbent.bundle_version:
        raise ValueError("promotion decision names a different incumbent")
    if decision.candidate_bundle_version != candidate.bundle_version:
        raise ValueError("promotion decision names a different candidate")
    if {
        incumbent.config_manifest_sha256,
        candidate.config_manifest_sha256,
    } != {config.manifest_sha256()}:
        raise ValueError("promotion bundles and engine config do not match")

    created_composite = False
    if decision.promote_controller and decision.promote_adversary:
        promoted = candidate
    elif not decision.promote_controller and not decision.promote_adversary:
        promoted = incumbent
    else:
        promoted, created_composite = _composite_bundle(
            root, config, incumbent, candidate, decision
        )

    pointer = (
        Path(current_model_path).expanduser().resolve()
        if current_model_path is not None
        else root / "current_model.json"
    )
    write_current_model_pointer(pointer, promoted)
    record = (
        Path(record_path).expanduser().resolve()
        if record_path is not None
        else root / "promotions" / f"candidate_{candidate.bundle_version}.json"
    )
    _atomic_json(
        record,
        {
            "decision": asdict(decision),
            "incumbent_manifest_sha256": incumbent.manifest_sha256,
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "promoted_manifest": str(promoted.manifest_path),
            "promoted_manifest_sha256": promoted.manifest_sha256,
            "promoted_role_versions": promoted.role_versions,
            "created_composite_bundle": created_composite,
        },
    )
    return PromotionResult(
        decision=decision,
        promoted_bundle=promoted,
        current_model_path=pointer,
        record_path=record,
        created_composite_bundle=created_composite,
    )
