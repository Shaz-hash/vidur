"""Evaluate a GV4 model controller against the deterministic SJF256 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Callable, Mapping, Protocol

from GV4_Engine.config import GV4EngineConfig
from GV4_Engine.history_root import HistoryRootGenerator

from ..engine_runtime import (
    BackendName,
    EngineRuntime,
    SearchConfig,
    SearchContext,
    StateSummary,
    create_engine_runtime,
)
from ..model_bundle import LoadedModelBundle
from ..runner import (
    GameCycleConfig,
    GameCycleResult,
    PlayedStep,
    run_game_cycle,
    select_root_action,
)
from .baselines import SJFPolicyConfig, select_sjf_controller_action


TimingProviderFactory = Callable[[], Any]

__all__ = [
    "BaselineCycleResult",
    "SJFBenchmarkConfig",
    "SJFBenchmarkResult",
    "SJFGameComparison",
    "evaluate_against_sjf",
    "run_sjf_cycle",
]


class SJFLogger(Protocol):
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

    def log_baseline_game(
        self,
        result: "BaselineCycleResult",
        *,
        scenario: str,
        paired_seed: int,
        controller_version: int,
        adversary_version: int,
    ) -> None: ...

    def log_sjf_comparison(self, result: "SJFGameComparison") -> None: ...


@dataclass(frozen=True, slots=True)
class SJFBenchmarkConfig:
    games: int = 100
    backend: BackendName = "native"
    search: SearchConfig = field(
        default_factory=lambda: SearchConfig(
            iterations=100,
            use_policy_prior=True,
            use_model_bootstrap=True,
        )
    )
    policy: SJFPolicyConfig = field(default_factory=SJFPolicyConfig)
    seed: int = 12_026
    history_hops: int = 0
    horizon_sec: float = 20.0
    max_actions: int = 100_000
    tie_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.games <= 0 or self.max_actions <= 0:
            raise ValueError("games and max_actions must be positive")
        if self.seed < 0 or self.history_hops < 0:
            raise ValueError("seed and history_hops must be nonnegative")
        if self.backend not in {"python", "native"}:
            raise ValueError("backend must be python or native")
        if not math.isfinite(self.horizon_sec) or self.horizon_sec <= 0.0:
            raise ValueError("horizon_sec must be positive and finite")
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class BaselineCycleResult:
    game_id: int
    cycle_label: str
    final_state_summary: StateSummary
    end_reason: str
    actions_applied: int
    searched_adversary_decisions: int
    sjf_decisions: int
    zero_budget_fallbacks: int
    final_time: float
    final_objective: float


@dataclass(frozen=True, slots=True)
class SJFGameComparison:
    pair_index: int
    paired_seed: int
    model_cost: float
    sjf_cost: float
    model_improvement: float
    outcome: str
    model_actions: int
    sjf_actions: int


@dataclass(frozen=True, slots=True)
class SJFBenchmarkResult:
    bundle_version: int
    games: tuple[SJFGameComparison, ...]
    model_wins: int
    ties: int
    sjf_wins: int
    model_score_rate: float
    mean_model_improvement: float

    def summary(self) -> dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "model_wins": self.model_wins,
            "ties": self.ties,
            "sjf_wins": self.sjf_wins,
            "model_score_rate": self.model_score_rate,
            "mean_model_improvement": self.mean_model_improvement,
            "games": [asdict(game) for game in self.games],
        }


def _emit(
    observer: Any,
    runtime: EngineRuntime,
    *,
    sequence: int,
    phase: str,
    decision_index: int | None,
    before: Any,
    edge: Any,
    search: Any = None,
) -> None:
    if observer is None:
        return
    observer(
        PlayedStep(
            sequence=sequence,
            phase=phase,
            decision_index=decision_index,
            edge=edge,
            search=search,
            state_before=runtime.summarize_state(before),
            state_after=runtime.summarize_state(edge.state),
        )
    )


def run_sjf_cycle(
    runtime: EngineRuntime,
    cycle: GameCycleConfig,
    *,
    policy: SJFPolicyConfig = SJFPolicyConfig(),
    observer: Any = None,
    model_versions: Mapping[str, int] | None = None,
) -> BaselineCycleResult:
    """Play direct SJF controller turns and normal model-MCTS adversary turns."""

    history = HistoryRootGenerator(runtime).generate(
        hops=cycle.history_hops,
        seed=cycle.seed,
    )
    state = history.state
    deadline = runtime.state_time(state) + cycle.horizon_sec
    actions_applied = adversary_decisions = sjf_decisions = fallbacks = 0
    decision_index = 0
    selection_rng = random.Random(cycle.seed ^ 0x534A_4632_3536)
    end_reason = ""

    while actions_applied < cycle.max_actions:
        if runtime.state_time(state) >= deadline:
            end_reason = "horizon_reached"
            break
        legal = runtime.canonical_actions(state)
        if not legal:
            end_reason = "no_legal_actions"
            break

        if len(legal) == 1:
            edge = runtime.apply_action(state, legal[0])
            _emit(
                observer,
                runtime,
                sequence=actions_applied,
                phase="forced",
                decision_index=None,
                before=state,
                edge=edge,
            )
        elif runtime.player_to_move(state) == "controller":
            selection = select_sjf_controller_action(legal, runtime.config, policy)
            edge = runtime.apply_action(state, selection.action)
            sjf_decisions += 1
            fallbacks += int(selection.used_zero_budget_fallback)
            _emit(
                observer,
                runtime,
                sequence=actions_applied,
                phase="searched",
                decision_index=decision_index,
                before=state,
                edge=edge,
            )
            decision_index += 1
        else:
            search = runtime.search(
                state,
                SearchContext(
                    game_id=cycle.game_id,
                    root_id=decision_index,
                    root_node_id=decision_index
                    * (runtime.search_config.iterations + 2),
                    root_depth=history.achieved_hops + actions_applied,
                    model_version=int((model_versions or {}).get("adversary", 0)),
                    cycle_label=cycle.cycle_label,
                ),
            )
            action = select_root_action(search, temperature=0.0, rng=selection_rng)
            edge = runtime.apply_action(state, action)
            adversary_decisions += 1
            _emit(
                observer,
                runtime,
                sequence=actions_applied,
                phase="searched",
                decision_index=decision_index,
                before=state,
                edge=edge,
                search=search,
            )
            decision_index += 1
        state = edge.state
        actions_applied += 1

    if not end_reason:
        end_reason = "max_actions_reached"
    return BaselineCycleResult(
        game_id=cycle.game_id,
        cycle_label=cycle.cycle_label,
        final_state_summary=runtime.summarize_state(state),
        end_reason=end_reason,
        actions_applied=actions_applied,
        searched_adversary_decisions=adversary_decisions,
        sjf_decisions=sjf_decisions,
        zero_budget_fallbacks=fallbacks,
        final_time=runtime.state_time(state),
        final_objective=runtime.objective_cost(state),
    )


def _runtime(
    config: GV4EngineConfig,
    bundle: LoadedModelBundle,
    benchmark: SJFBenchmarkConfig,
    *,
    seed: int,
    timing_provider_factory: TimingProviderFactory | None,
) -> EngineRuntime:
    inference = bundle.create_inference(benchmark.backend, config)
    timing = None if timing_provider_factory is None else timing_provider_factory()
    return create_engine_runtime(
        benchmark.backend,
        config,
        search_config=benchmark.search,
        seed=seed,
        timing_provider=timing,
        python_inference=inference if benchmark.backend == "python" else None,
        native_inference=inference if benchmark.backend == "native" else None,
    )


def evaluate_against_sjf(
    config: GV4EngineConfig,
    bundle: LoadedModelBundle,
    *,
    benchmark: SJFBenchmarkConfig = SJFBenchmarkConfig(),
    timing_provider_factory: TimingProviderFactory | None = None,
    logger: SJFLogger | None = None,
) -> SJFBenchmarkResult:
    """Compare model MCTS and SJF256 controllers against the same model adversary."""

    if bundle.config_manifest_sha256 != config.manifest_sha256():
        raise ValueError("SJF bundle and engine config do not match")
    role_versions = bundle.role_versions
    comparisons: list[SJFGameComparison] = []

    for pair_index in range(benchmark.games):
        seed = benchmark.seed + pair_index
        sjf_runtime = _runtime(
            config,
            bundle,
            benchmark,
            seed=seed,
            timing_provider_factory=timing_provider_factory,
        )
        sjf_observer = (
            None
            if logger is None
            else logger.observer(
                game_id=pair_index * 2,
                cycle_label="sjf256",
                controller_version=role_versions["controller"],
                adversary_version=role_versions["adversary"],
                selection_mode="sjf256",
                iterations_requested=benchmark.search.iterations,
            )
        )
        try:
            sjf = run_sjf_cycle(
                sjf_runtime,
                GameCycleConfig(
                    game_id=pair_index * 2,
                    cycle_label="sjf256",
                    seed=seed,
                    history_hops=benchmark.history_hops,
                    horizon_sec=benchmark.horizon_sec,
                    max_actions=benchmark.max_actions,
                    selection_temperature=0.0,
                    bootstrap_mode="model",
                ),
                policy=benchmark.policy,
                observer=sjf_observer,
                model_versions=role_versions,
            )
        finally:
            sjf_runtime.close()

        model_runtime = _runtime(
            config,
            bundle,
            benchmark,
            seed=seed,
            timing_provider_factory=timing_provider_factory,
        )
        model_observer = (
            None
            if logger is None
            else logger.observer(
                game_id=pair_index * 2 + 1,
                cycle_label="model_vs_sjf",
                controller_version=role_versions["controller"],
                adversary_version=role_versions["adversary"],
                selection_mode="mcts",
                iterations_requested=benchmark.search.iterations,
            )
        )
        try:
            model = run_game_cycle(
                model_runtime,
                GameCycleConfig(
                    game_id=pair_index * 2 + 1,
                    cycle_label="model_vs_sjf",
                    seed=seed,
                    history_hops=benchmark.history_hops,
                    horizon_sec=benchmark.horizon_sec,
                    max_actions=benchmark.max_actions,
                    selection_temperature=0.0,
                    bootstrap_mode="model",
                ),
                observer=model_observer,
                model_versions=role_versions,
            )
        finally:
            model_runtime.close()

        improvement = sjf.final_objective - model.final_objective
        outcome = (
            "model_win"
            if improvement > benchmark.tie_tolerance
            else "sjf_win" if improvement < -benchmark.tie_tolerance else "tie"
        )
        comparison = SJFGameComparison(
            pair_index=pair_index,
            paired_seed=seed,
            model_cost=model.final_objective,
            sjf_cost=sjf.final_objective,
            model_improvement=improvement,
            outcome=outcome,
            model_actions=model.actions_applied,
            sjf_actions=sjf.actions_applied,
        )
        comparisons.append(comparison)
        if logger is not None:
            logger.log_baseline_game(
                sjf,
                scenario="sjf256",
                paired_seed=seed,
                controller_version=role_versions["controller"],
                adversary_version=role_versions["adversary"],
            )
            logger.log_game(
                model,
                scenario="model_vs_sjf",
                paired_seed=seed,
                controller_version=role_versions["controller"],
                adversary_version=role_versions["adversary"],
            )
            logger.log_sjf_comparison(comparison)

    model_wins = sum(game.outcome == "model_win" for game in comparisons)
    ties = sum(game.outcome == "tie" for game in comparisons)
    count = len(comparisons)
    return SJFBenchmarkResult(
        bundle_version=bundle.bundle_version,
        games=tuple(comparisons),
        model_wins=model_wins,
        ties=ties,
        sjf_wins=count - model_wins - ties,
        model_score_rate=(model_wins + 0.5 * ties) / count,
        mean_model_improvement=sum(game.model_improvement for game in comparisons)
        / count,
    )
