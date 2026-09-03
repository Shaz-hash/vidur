"""Run one backend-neutral GV4 AlphaGoZero game cycle.

The worker layer will launch this module in an isolated process. This file owns
game orchestration only: random history, root search, visit-based selection,
forced one-action advancement, and replay finalization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import importlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable, Literal, Mapping, Sequence

from GV4_Engine.config import GV4EngineConfig
from GV4_Engine.history_root import HistoryRoot, HistoryRootGenerator

from .engine_runtime import (
    AppliedEdge,
    BackendName,
    CanonicalActionRef,
    EngineMetadata,
    EngineRuntime,
    SearchConfig,
    SearchContext,
    SearchResult,
    StateSummary,
    create_engine_runtime,
)
from .model_bundle import LoadedModelBundle, load_model_bundle
from .replay_runtime import GV4ReplayRecorder, ReplayWriteResult


BootstrapMode = Literal["neutral_zero", "model"]
StepObserver = Callable[["PlayedStep"], None]

__all__ = [
    "GameCycleConfig",
    "GameCycleResult",
    "PlayedStep",
    "RunnerError",
    "run_game_cycle",
    "select_root_action",
]


class RunnerError(RuntimeError):
    """Raised when a game cannot continue without violating its contract."""


@dataclass(frozen=True, slots=True)
class GameCycleConfig:
    """All orchestration choices for one root and one played cycle."""

    game_id: int
    cycle_label: str = "self_play"
    seed: int = 0
    history_hops: int = 0
    horizon_sec: float = 20.0
    max_actions: int = 100_000
    selection_temperature: float = 1.0
    bootstrap_mode: BootstrapMode = "neutral_zero"

    def __post_init__(self) -> None:
        for name in ("game_id", "seed", "history_hops"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.cycle_label:
            raise ValueError("cycle_label cannot be empty")
        if not math.isfinite(self.horizon_sec) or self.horizon_sec <= 0.0:
            raise ValueError("horizon_sec must be positive and finite")
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if (
            not math.isfinite(self.selection_temperature)
            or self.selection_temperature < 0.0
        ):
            raise ValueError("selection_temperature must be nonnegative")
        if self.bootstrap_mode not in {"neutral_zero", "model"}:
            raise ValueError("bootstrap_mode must be neutral_zero or model")


@dataclass(frozen=True, slots=True)
class PlayedStep:
    """One immutable logger event produced after an applied action."""

    sequence: int
    phase: Literal["searched", "forced"]
    decision_index: int | None
    edge: AppliedEdge
    search: SearchResult | None
    state_before: StateSummary
    state_after: StateSummary


@dataclass(frozen=True, slots=True)
class GameCycleResult:
    """Summary returned to a worker or local test harness."""

    game_id: int
    cycle_label: str
    backend: BackendName
    engine_metadata: EngineMetadata
    model_versions: dict[str, int]
    history: HistoryRoot
    final_state: Any = field(repr=False, compare=False)
    final_state_summary: StateSummary
    end_reason: str
    actions_applied: int
    searched_decisions: int
    forced_actions: int
    final_time: float
    final_objective: float
    bootstrap_kind: str
    bootstrap_value: float
    replay: ReplayWriteResult | None

    def summary(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "cycle_label": self.cycle_label,
            "backend": self.backend,
            "engine_metadata": asdict(self.engine_metadata),
            "model_versions": dict(self.model_versions),
            "history": {
                "seed": self.history.seed,
                "requested_hops": self.history.requested_hops,
                "achieved_hops": self.history.achieved_hops,
                "next_player": self.history.next_player,
                "final_time": self.history.final_time,
                "stop_reason": self.history.stop_reason,
                "steps": [asdict(step) for step in self.history.steps],
            },
            "end_reason": self.end_reason,
            "actions_applied": self.actions_applied,
            "searched_decisions": self.searched_decisions,
            "forced_actions": self.forced_actions,
            "final_time": self.final_time,
            "final_objective": self.final_objective,
            "final_state": asdict(self.final_state_summary),
            "bootstrap_kind": self.bootstrap_kind,
            "bootstrap_value": self.bootstrap_value,
            "replay": (
                None
                if self.replay is None
                else {
                    "state_path": str(self.replay.state_path),
                    "action_path": str(self.replay.action_path),
                    "manifest_path": str(self.replay.manifest_path),
                    "state_rows": self.replay.state_rows,
                    "action_rows": self.replay.action_rows,
                    "state_sha256": self.replay.state_sha256,
                    "action_sha256": self.replay.action_sha256,
                }
            ),
        }


def select_root_action(
    result: SearchResult,
    *,
    temperature: float,
    rng: random.Random,
) -> CanonicalActionRef:
    """Select from canonical root visits without reintroducing raw aliases."""

    stats = result.action_stats
    if not stats:
        raise RunnerError("MCTS returned no canonical root actions")

    if temperature == 0.0:
        sign = 1.0 if result.root_player == "controller" else -1.0
        return max(
            stats,
            key=lambda item: (
                item.visits,
                sign * item.mean_value,
                -item.action.representative_raw_index,
            ),
        ).action

    exponent = 1.0 / temperature
    weights = [
        float(item.visits) ** exponent if item.visits > 0 else 0.0 for item in stats
    ]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        weights = [1.0] * len(stats)
        total = float(len(stats))

    draw = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(stats, weights):
        cumulative += weight
        if draw < cumulative:
            return item.action
    return stats[-1].action


def _emit(
    observer: StepObserver | None,
    runtime: EngineRuntime,
    *,
    sequence: int,
    phase: Literal["searched", "forced"],
    decision_index: int | None,
    state_before: Any,
    edge: AppliedEdge,
    search: SearchResult | None = None,
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
            state_before=runtime.summarize_state(state_before),
            state_after=runtime.summarize_state(edge.state),
        )
    )


def _bootstrap(
    runtime: EngineRuntime,
    mode: BootstrapMode,
    end_reason: str,
    state: Any,
) -> tuple[str, float]:
    if end_reason == "no_legal_actions":
        return "terminal_zero", 0.0
    if mode == "model":
        return "model", float(runtime.bootstrap_value(state))
    return "neutral_zero", 0.0


def run_game_cycle(
    runtime: EngineRuntime,
    config: GameCycleConfig,
    *,
    initial_state: Any | None = None,
    replay: GV4ReplayRecorder | None = None,
    observer: StepObserver | None = None,
    model_versions: Mapping[str, int] | None = None,
) -> GameCycleResult:
    """Generate one history root and play until time or action safety cutoff."""

    history = HistoryRootGenerator(runtime).generate(
        hops=config.history_hops,
        seed=config.seed,
        initial_state=initial_state,
    )
    state = history.state
    deadline = runtime.state_time(state) + config.horizon_sec
    selection_rng = random.Random(config.seed ^ 0xA5A5_A5A5_5A5A_5A5A)

    actions_applied = 0
    forced_actions = 0
    decision_index = 0
    end_reason = ""

    while actions_applied < config.max_actions:
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
                state_before=state,
                edge=edge,
            )
            state = edge.state
            actions_applied += 1
            forced_actions += 1
            continue

        context = SearchContext(
            game_id=config.game_id,
            root_id=decision_index,
            # One iteration can create at most one node. Reserving a disjoint
            # range keeps IDs unique if several root trees share one log.
            root_node_id=decision_index * (runtime.search_config.iterations + 2),
            root_depth=history.achieved_hops + actions_applied,
            model_version=int(
                (model_versions or {}).get(runtime.player_to_move(state), 0)
            ),
            cycle_label=config.cycle_label,
        )
        search = runtime.search(state, context)
        selected = select_root_action(
            search,
            temperature=config.selection_temperature,
            rng=selection_rng,
        )
        selected_edge = runtime.apply_action(state, selected)
        _emit(
            observer,
            runtime,
            sequence=actions_applied,
            phase="searched",
            decision_index=decision_index,
            state_before=state,
            edge=selected_edge,
            search=search,
        )
        state = selected_edge.state
        actions_applied += 1

        # Forced turns have no useful policy target, but their Bellman edges
        # must remain part of the preceding searched decision.
        forced_chain: list[AppliedEdge] = []
        while (
            actions_applied < config.max_actions
            and runtime.state_time(state) < deadline
        ):
            forced_legal = runtime.canonical_actions(state)
            if len(forced_legal) != 1:
                break
            edge = runtime.apply_action(state, forced_legal[0])
            forced_chain.append(edge)
            _emit(
                observer,
                runtime,
                sequence=actions_applied,
                phase="forced",
                decision_index=decision_index,
                state_before=state,
                edge=edge,
            )
            state = edge.state
            actions_applied += 1
            forced_actions += 1

        if replay is not None:
            replay.record_decision(
                decision_index=decision_index,
                search=search,
                selected_action=selected,
                selected_edge=selected_edge,
                forced_edges=forced_chain,
            )
        decision_index += 1

    if not end_reason:
        end_reason = "max_actions_reached"

    bootstrap_kind, bootstrap_value = _bootstrap(
        runtime, config.bootstrap_mode, end_reason, state
    )
    replay_result = None
    if replay is not None:
        replay_result = replay.finish_cycle(
            bootstrap_kind=bootstrap_kind,
            bootstrap_value=bootstrap_value,
            end_reason=end_reason,
            final_time=runtime.state_time(state),
        )

    return GameCycleResult(
        game_id=config.game_id,
        cycle_label=config.cycle_label,
        backend=runtime.backend,
        engine_metadata=runtime.metadata,
        model_versions={
            str(role): int(version) for role, version in (model_versions or {}).items()
        },
        history=history,
        final_state=state,
        final_state_summary=runtime.summarize_state(state),
        end_reason=end_reason,
        actions_applied=actions_applied,
        searched_decisions=decision_index,
        forced_actions=forced_actions,
        final_time=runtime.state_time(state),
        final_objective=runtime.objective_cost(state),
        bootstrap_kind=bootstrap_kind,
        bootstrap_value=bootstrap_value,
        replay=replay_result,
    )


def _load_config_factory(specification: str) -> GV4EngineConfig:
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("config factory must have the form module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    config = factory()
    if not isinstance(config, GV4EngineConfig):
        raise TypeError("config factory did not return GV4EngineConfig")
    config.validate()
    return config


def _atomic_result(path: Path, result: GameCycleResult) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result.summary(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-config-factory", required=True)
    parser.add_argument("--backend", choices=("python", "native"), default="native")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--game-id", type=int, default=0)
    parser.add_argument("--cycle-label", default="uniform_self_play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-hops", type=int, default=0)
    parser.add_argument("--horizon-sec", type=float, default=20.0)
    parser.add_argument("--max-actions", type=int, default=100_000)
    parser.add_argument("--mcts-iterations", type=int, default=100)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--policy-prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-min-probability", type=float, default=1e-8)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--root-dirichlet-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--root-dirichlet-total-concentration", type=float, default=0.0
    )
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-count", type=int, default=0)
    parser.add_argument("--rollout-horizon-sec", type=float, default=0.4)
    parser.add_argument("--rollout-seed", type=int)
    parser.add_argument("--rollout-policy-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-probability-quantum", type=float, default=1e-6)
    parser.add_argument("--rollout-max-actions", type=int, default=4096)
    parser.add_argument("--model-bundle", type=Path)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument(
        "--use-policy-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-model-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=("neutral_zero", "model"),
        default="neutral_zero",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI used initially for uniform games and later by the worker daemon."""

    args = _parser().parse_args(argv)
    engine_config = _load_config_factory(args.engine_config_factory)
    bundle: LoadedModelBundle | None = None
    if args.model_bundle is not None:
        bundle = load_model_bundle(
            args.model_bundle,
            config=engine_config,
            device=args.model_device,
        )
    if (
        args.use_policy_prior
        or args.use_model_bootstrap
        or args.bootstrap_mode == "model"
    ) and bundle is None:
        raise RunnerError("model-backed search requires --model-bundle")

    search_config = SearchConfig(
        iterations=args.mcts_iterations,
        puct_c=args.puct_c,
        policy_prior_temperature=args.policy_prior_temperature,
        prior_min_probability=args.prior_min_probability,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_dirichlet_epsilon=args.root_dirichlet_epsilon,
        root_dirichlet_total_concentration=(
            args.root_dirichlet_total_concentration
        ),
        use_policy_prior=args.use_policy_prior,
        use_model_bootstrap=args.use_model_bootstrap,
        rollout_count=args.rollout_count,
        rollout_horizon_sec=args.rollout_horizon_sec,
        rollout_seed=args.seed if args.rollout_seed is None else args.rollout_seed,
        rollout_policy_temperature=args.rollout_policy_temperature,
        rollout_probability_quantum=args.rollout_probability_quantum,
        rollout_max_actions=args.rollout_max_actions,
    )
    cycle_config = GameCycleConfig(
        game_id=args.game_id,
        cycle_label=args.cycle_label,
        seed=args.seed,
        history_hops=args.history_hops,
        horizon_sec=args.horizon_sec,
        max_actions=args.max_actions,
        selection_temperature=args.selection_temperature,
        bootstrap_mode=args.bootstrap_mode,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "game_result.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"game result already exists: {result_path}")
    inference = (
        None if bundle is None else bundle.create_inference(args.backend, engine_config)
    )
    runtime = create_engine_runtime(
        args.backend,
        engine_config,
        search_config=search_config,
        seed=args.seed,
        python_inference=inference if args.backend == "python" else None,
        native_inference=inference if args.backend == "native" else None,
    )
    try:
        recorder = GV4ReplayRecorder(
            output_dir,
            game_id=cycle_config.game_id,
            cycle_label=cycle_config.cycle_label,
            engine_metadata=runtime.metadata,
            model_versions={} if bundle is None else bundle.model_versions,
            game_seed=cycle_config.seed,
            history_hops=cycle_config.history_hops,
            overwrite=args.overwrite,
        )
        result = run_game_cycle(
            runtime,
            cycle_config,
            replay=recorder,
            model_versions={} if bundle is None else bundle.role_versions,
        )
        _atomic_result(result_path, result)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
