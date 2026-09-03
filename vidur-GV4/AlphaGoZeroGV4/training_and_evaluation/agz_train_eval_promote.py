"""Thin coordinator for one GV4 AlphaGoZero train/evaluate/promote cycle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from GV4_Engine.config import GV4EngineConfig

from ..model_bundle import LoadedModelBundle, load_model_bundle
from .arena import ArenaConfig, ArenaResult, evaluate_candidate
from .evaluation_pipeline_logger import EvaluationPipelineLogger
from .indexed_replay import ReplayIndex, open_replay_index
from .promotion import (
    PromotionPolicy,
    PromotionResult,
    decide_promotion,
    publish_promotion,
)
from .sjf_runner import SJFBenchmarkConfig, SJFBenchmarkResult, evaluate_against_sjf
from .trainer import TrainerConfig, TrainingResult, train_candidate


TimingProviderFactory = Callable[[], Any]
ArenaEvaluator = Callable[
    [
        GV4EngineConfig,
        LoadedModelBundle,
        LoadedModelBundle,
        ArenaConfig,
        TimingProviderFactory | None,
        EvaluationPipelineLogger,
    ],
    ArenaResult,
]

__all__ = [
    "ArenaEvaluator",
    "TrainEvalPromoteConfig",
    "TrainEvalPromoteResult",
    "next_candidate_version",
    "run_train_eval_promote",
]


@dataclass(frozen=True, slots=True)
class TrainEvalPromoteConfig:
    replay_root: Path
    models_root: Path
    output_dir: Path
    current_model_path: Path | None = None
    replay_index_dir: Path | None = None
    candidate_version: int | None = None
    rebuild_replay_index: bool = False
    model_device: str = "cpu"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)
    sjf: SJFBenchmarkConfig | None = None

    def __post_init__(self) -> None:
        for name in ("replay_root", "models_root", "output_dir"):
            object.__setattr__(
                self, name, Path(getattr(self, name)).expanduser().resolve()
            )
        if self.current_model_path is not None:
            object.__setattr__(
                self,
                "current_model_path",
                Path(self.current_model_path).expanduser().resolve(),
            )
        if self.replay_index_dir is not None:
            object.__setattr__(
                self,
                "replay_index_dir",
                Path(self.replay_index_dir).expanduser().resolve(),
            )
        if self.candidate_version is not None and self.candidate_version < 0:
            raise ValueError("candidate_version must be nonnegative")

    @property
    def current_pointer(self) -> Path:
        return self.current_model_path or self.models_root / "current_model.json"


@dataclass(slots=True)
class TrainEvalPromoteResult:
    replay_index: ReplayIndex
    incumbent: LoadedModelBundle
    training: TrainingResult
    arena: ArenaResult
    promotion: PromotionResult
    sjf: SJFBenchmarkResult | None
    summary_path: Path
    elapsed_sec: float

    def summary(self) -> dict[str, Any]:
        return {
            "incumbent_bundle_version": self.incumbent.bundle_version,
            "replay_index_dir": str(self.replay_index.index_dir),
            "replay_role_counts": self.replay_index.role_counts,
            "training": self.training.summary(),
            "arena": self.arena.summary(),
            "promotion": self.promotion.summary(),
            "sjf": None if self.sjf is None else self.sjf.summary(),
            "elapsed_sec": self.elapsed_sec,
        }


def next_candidate_version(models_root: str | Path, incumbent_version: int) -> int:
    """Return the first unused sequential immutable candidate directory."""

    root = Path(models_root).expanduser().resolve()
    version = int(incumbent_version) + 1
    while (root / f"Model_Version{version}").exists():
        version += 1
    return version


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _evaluate_locally(
    engine_config: GV4EngineConfig,
    incumbent: LoadedModelBundle,
    candidate: LoadedModelBundle,
    arena: ArenaConfig,
    timing_provider_factory: TimingProviderFactory | None,
    logger: EvaluationPipelineLogger,
) -> ArenaResult:
    return evaluate_candidate(
        engine_config,
        incumbent,
        candidate,
        arena=arena,
        timing_provider_factory=timing_provider_factory,
        logger=logger,
    )


def run_train_eval_promote(
    engine_config: GV4EngineConfig,
    pipeline: TrainEvalPromoteConfig,
    *,
    timing_provider_factory: TimingProviderFactory | None = None,
    logger: EvaluationPipelineLogger | None = None,
    arena_evaluator: ArenaEvaluator | None = None,
) -> TrainEvalPromoteResult:
    """Execute one cycle while delegating every semantic decision downstream."""

    started = time.perf_counter()
    engine_config.validate()
    pipeline.models_root.mkdir(parents=True, exist_ok=True)
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or EvaluationPipelineLogger(pipeline.output_dir / "evaluation")

    replay_index = open_replay_index(
        pipeline.replay_root,
        engine_config,
        index_dir=pipeline.replay_index_dir,
        rebuild=pipeline.rebuild_replay_index,
    )
    incumbent = load_model_bundle(
        pipeline.current_pointer,
        config=engine_config,
        device=pipeline.model_device,
    )
    candidate_version = (
        pipeline.candidate_version
        if pipeline.candidate_version is not None
        else next_candidate_version(pipeline.models_root, incumbent.bundle_version)
    )
    training = train_candidate(
        replay_index,
        engine_config,
        incumbent,
        candidate_version=candidate_version,
        destination=pipeline.models_root / f"Model_Version{candidate_version}",
        training=pipeline.trainer,
    )
    logger.log_training(training)

    evaluator = arena_evaluator or _evaluate_locally
    arena = evaluator(
        engine_config,
        incumbent,
        training.bundle,
        pipeline.arena,
        timing_provider_factory,
        logger,
    )
    logger.log_arena(arena)
    decision = decide_promotion(arena, pipeline.promotion)
    promotion = publish_promotion(
        pipeline.models_root,
        engine_config,
        incumbent,
        training.bundle,
        decision,
        current_model_path=pipeline.current_pointer,
    )
    logger.log_promotion(promotion)

    sjf = None
    if pipeline.sjf is not None:
        sjf = evaluate_against_sjf(
            engine_config,
            promotion.promoted_bundle,
            benchmark=pipeline.sjf,
            timing_provider_factory=timing_provider_factory,
            logger=logger,
        )
        logger.log_sjf_summary(sjf)

    summary_path = pipeline.output_dir / f"cycle_{candidate_version}.json"
    result = TrainEvalPromoteResult(
        replay_index=replay_index,
        incumbent=incumbent,
        training=training,
        arena=arena,
        promotion=promotion,
        sjf=sjf,
        summary_path=summary_path,
        elapsed_sec=time.perf_counter() - started,
    )
    _atomic_json(summary_path, result.summary())
    return result


def _load_config_factory(specification: str) -> GV4EngineConfig:
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("config factory must have the form module:function")
    config = getattr(importlib.import_module(module_name), function_name)()
    if not isinstance(config, GV4EngineConfig):
        raise TypeError("config factory did not return GV4EngineConfig")
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-config-factory", required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--backend", choices=("python", "native"), default="native")
    parser.add_argument("--arena-games", type=int, default=100)
    parser.add_argument("--mcts-iterations", type=int, default=100)
    parser.add_argument("--training-epochs", type=int, default=5)
    parser.add_argument("--training-device")
    parser.add_argument("--run-sjf", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    from ..engine_runtime import SearchConfig

    search = SearchConfig(
        iterations=args.mcts_iterations,
        use_policy_prior=True,
        use_model_bootstrap=True,
    )
    arena = ArenaConfig(games=args.arena_games, backend=args.backend, search=search)
    sjf = (
        SJFBenchmarkConfig(games=args.arena_games, backend=args.backend, search=search)
        if args.run_sjf
        else None
    )
    result = run_train_eval_promote(
        _load_config_factory(args.engine_config_factory),
        TrainEvalPromoteConfig(
            replay_root=args.replay_root,
            models_root=args.models_root,
            output_dir=args.output_dir,
            candidate_version=args.candidate_version,
            trainer=TrainerConfig(
                epochs=args.training_epochs,
                device=args.training_device,
            ),
            arena=arena,
            sjf=sjf,
        ),
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
