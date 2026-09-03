"""Paired-seed arena evaluation for independent GV4 role promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Literal, Protocol

from GV4_Engine.config import GV4EngineConfig
from GV4_Engine.dnn_inference.inference import GV4DNNInference

from ..engine_runtime import (
    BackendName,
    EngineRuntime,
    SearchConfig,
    create_engine_runtime,
)
from ..model_bundle import LoadedModelBundle
from ..runner import GameCycleConfig, GameCycleResult, run_game_cycle


ArenaRole = Literal["controller", "adversary"]
Scenario = Literal["incumbent", "candidate_controller", "candidate_adversary"]
TimingProviderFactory = Callable[[], Any]

__all__ = [
    "ArenaConfig",
    "ArenaGameResult",
    "ArenaResult",
    "RoleArenaStats",
    "evaluate_candidate",
    "summarize_role_results",
]


class ArenaLogger(Protocol):
    def observer(
        self,
        *,
        game_id: int,
        cycle_label: str,
        controller_version: int,
        adversary_version: int,
        selection_mode: str,
        iterations_requested: int | None = None,
    ) -> Any: ...

    def log_game(
        self,
        result: GameCycleResult,
        *,
        scenario: str,
        paired_seed: int,
        controller_version: int,
        adversary_version: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    games: int = 100
    backend: BackendName = "native"
    search: SearchConfig = field(
        default_factory=lambda: SearchConfig(
            iterations=100,
            use_policy_prior=True,
            use_model_bootstrap=True,
        )
    )
    seed: int = 2026
    history_hops: int = 0
    horizon_sec: float = 20.0
    max_actions: int = 100_000
    tie_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name in ("games", "max_actions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "history_hops"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.backend not in {"python", "native"}:
            raise ValueError("backend must be python or native")
        if not math.isfinite(self.horizon_sec) or self.horizon_sec <= 0.0:
            raise ValueError("horizon_sec must be positive and finite")
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class ArenaGameResult:
    pair_index: int
    paired_seed: int
    scenario: Scenario
    controller_version: int
    adversary_version: int
    final_cost: float
    slo_violations: int
    prefill_lateness_sec: float
    decode_lateness_sec: float
    requests_completed: int
    requests_stopped: int
    requests_dropped: int
    actions_applied: int
    end_reason: str


@dataclass(frozen=True, slots=True)
class RoleArenaStats:
    role: ArenaRole
    games: int
    wins: int
    ties: int
    losses: int
    score_rate: float
    mean_improvement: float


@dataclass(frozen=True, slots=True)
class ArenaResult:
    incumbent_bundle_version: int
    candidate_bundle_version: int
    games: tuple[ArenaGameResult, ...]
    controller: RoleArenaStats
    adversary: RoleArenaStats

    def summary(self) -> dict[str, Any]:
        return {
            "incumbent_bundle_version": self.incumbent_bundle_version,
            "candidate_bundle_version": self.candidate_bundle_version,
            "controller": asdict(self.controller),
            "adversary": asdict(self.adversary),
            "games": [asdict(game) for game in self.games],
        }


def _mixed_inference(
    backend: BackendName,
    config: GV4EngineConfig,
    controller: LoadedModelBundle,
    adversary: LoadedModelBundle,
) -> Any:
    values = {
        "controller_value_model": controller.models["controller_value"],
        "controller_policy_model": controller.models["controller_policy"],
        "adversary_value_model": adversary.models["adversary_value"],
        "adversary_policy_model": adversary.models["adversary_policy"],
    }
    if backend == "python":
        return GV4DNNInference(config, **values)
    from GV4_Cpp import gv4_native
    from GV4_Cpp.runtime import config_from_python

    return gv4_native.InferenceRuntime(config_from_python(config), **values)


def _runtime(
    config: GV4EngineConfig,
    arena: ArenaConfig,
    *,
    seed: int,
    controller: LoadedModelBundle,
    adversary: LoadedModelBundle,
    timing_provider_factory: TimingProviderFactory | None,
) -> EngineRuntime:
    inference = _mixed_inference(arena.backend, config, controller, adversary)
    timing = None if timing_provider_factory is None else timing_provider_factory()
    return create_engine_runtime(
        arena.backend,
        config,
        search_config=arena.search,
        seed=seed,
        timing_provider=timing,
        python_inference=inference if arena.backend == "python" else None,
        native_inference=inference if arena.backend == "native" else None,
    )


def _play(
    config: GV4EngineConfig,
    arena: ArenaConfig,
    *,
    pair_index: int,
    scenario: Scenario,
    seed: int,
    controller: LoadedModelBundle,
    adversary: LoadedModelBundle,
    timing_provider_factory: TimingProviderFactory | None,
    logger: ArenaLogger | None,
) -> ArenaGameResult:
    game_id = pair_index * 3 + (
        0 if scenario == "incumbent" else 1 if scenario == "candidate_controller" else 2
    )
    controller_version = controller.role_versions["controller"]
    adversary_version = adversary.role_versions["adversary"]
    runtime = _runtime(
        config,
        arena,
        seed=seed,
        controller=controller,
        adversary=adversary,
        timing_provider_factory=timing_provider_factory,
    )
    observer = (
        None
        if logger is None
        else logger.observer(
            game_id=game_id,
            cycle_label=f"arena_{scenario}",
            controller_version=controller_version,
            adversary_version=adversary_version,
            selection_mode="mcts",
            iterations_requested=arena.search.iterations,
        )
    )
    try:
        result = run_game_cycle(
            runtime,
            GameCycleConfig(
                game_id=game_id,
                cycle_label=f"arena_{scenario}",
                seed=seed,
                history_hops=arena.history_hops,
                horizon_sec=arena.horizon_sec,
                max_actions=arena.max_actions,
                selection_temperature=0.0,
                bootstrap_mode="model",
            ),
            observer=observer,
            model_versions={
                "controller": controller_version,
                "adversary": adversary_version,
            },
        )
    finally:
        runtime.close()
    if logger is not None:
        logger.log_game(
            result,
            scenario=scenario,
            paired_seed=seed,
            controller_version=controller_version,
            adversary_version=adversary_version,
        )
    objective = result.final_state_summary.objective
    return ArenaGameResult(
        pair_index=pair_index,
        paired_seed=seed,
        scenario=scenario,
        controller_version=controller_version,
        adversary_version=adversary_version,
        final_cost=result.final_objective,
        slo_violations=objective.slo_violations,
        prefill_lateness_sec=objective.prefill_lateness_sec,
        decode_lateness_sec=objective.decode_lateness_sec,
        requests_completed=objective.requests_completed,
        requests_stopped=objective.requests_stopped,
        requests_dropped=objective.requests_dropped,
        actions_applied=result.actions_applied,
        end_reason=result.end_reason,
    )


def summarize_role_results(
    role: ArenaRole,
    baseline: list[float],
    candidate: list[float],
    tolerance: float,
) -> RoleArenaStats:
    # Controller improves by reducing cost; adversary improves by increasing it.
    improvements = [
        base - trial if role == "controller" else trial - base
        for base, trial in zip(baseline, candidate)
    ]
    wins = sum(value > tolerance for value in improvements)
    losses = sum(value < -tolerance for value in improvements)
    ties = len(improvements) - wins - losses
    games = len(improvements)
    return RoleArenaStats(
        role=role,
        games=games,
        wins=wins,
        ties=ties,
        losses=losses,
        score_rate=(wins + 0.5 * ties) / games,
        mean_improvement=sum(improvements) / games,
    )


def evaluate_candidate(
    config: GV4EngineConfig,
    incumbent: LoadedModelBundle,
    candidate: LoadedModelBundle,
    *,
    arena: ArenaConfig = ArenaConfig(),
    timing_provider_factory: TimingProviderFactory | None = None,
    logger: ArenaLogger | None = None,
) -> ArenaResult:
    """Evaluate each candidate role while holding the opposing role fixed."""

    config_hash = config.manifest_sha256()
    if {
        incumbent.config_manifest_sha256,
        candidate.config_manifest_sha256,
    } != {config_hash}:
        raise ValueError("arena bundles and engine config do not match")

    rows: list[ArenaGameResult] = []
    baseline_costs: list[float] = []
    controller_costs: list[float] = []
    adversary_costs: list[float] = []
    for pair_index in range(arena.games):
        seed = arena.seed + pair_index
        baseline = _play(
            config,
            arena,
            pair_index=pair_index,
            scenario="incumbent",
            seed=seed,
            controller=incumbent,
            adversary=incumbent,
            timing_provider_factory=timing_provider_factory,
            logger=logger,
        )
        controller_game = _play(
            config,
            arena,
            pair_index=pair_index,
            scenario="candidate_controller",
            seed=seed,
            controller=candidate,
            adversary=incumbent,
            timing_provider_factory=timing_provider_factory,
            logger=logger,
        )
        adversary_game = _play(
            config,
            arena,
            pair_index=pair_index,
            scenario="candidate_adversary",
            seed=seed,
            controller=incumbent,
            adversary=candidate,
            timing_provider_factory=timing_provider_factory,
            logger=logger,
        )
        rows.extend((baseline, controller_game, adversary_game))
        baseline_costs.append(baseline.final_cost)
        controller_costs.append(controller_game.final_cost)
        adversary_costs.append(adversary_game.final_cost)

    return ArenaResult(
        incumbent_bundle_version=incumbent.bundle_version,
        candidate_bundle_version=candidate.bundle_version,
        games=tuple(rows),
        controller=summarize_role_results(
            "controller", baseline_costs, controller_costs, arena.tie_tolerance
        ),
        adversary=summarize_role_results(
            "adversary", baseline_costs, adversary_costs, arena.tie_tolerance
        ),
    )
