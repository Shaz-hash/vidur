"""Seeded random GV4 roots shared by Python, native, tests, and self-play.

The generator deliberately knows nothing about requests, KV blocks, pipeline
stages, or MCTS. A small runtime adapter supplies legal canonical actions and
applies them. This keeps random-root generation stable when engine internals
change.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Protocol, Sequence


__all__ = [
    "HistoryRoot",
    "HistoryRootError",
    "HistoryRootGenerator",
    "HistoryStep",
]


class HistoryRootError(ValueError):
    """Raised when a runtime breaks the random-history contract."""


class _ActionView(Protocol):
    player: str
    canonical_action_index: int
    representative_raw_index: int
    equivalent_raw_indices: tuple[int, ...]


class _AppliedEdgeView(Protocol):
    state: Any
    player: str
    next_player: str
    started_at: float
    finished_at: float
    reward: float
    discount: float
    transition_kind: str


class _HistoryRuntime(Protocol):
    def initial_state(self) -> Any: ...

    def clone_state(self, state: Any) -> Any: ...

    def state_time(self, state: Any) -> float: ...

    def player_to_move(self, state: Any) -> str: ...

    def canonical_actions(self, state: Any) -> Sequence[_ActionView]: ...

    def apply_action(self, state: Any, action: _ActionView) -> _AppliedEdgeView: ...


@dataclass(frozen=True, slots=True)
class HistoryStep:
    """One canonical action used to reach a random root."""

    hop: int
    player: str
    canonical_action_index: int
    representative_raw_index: int
    equivalent_raw_indices: tuple[int, ...]
    started_at: float
    finished_at: float
    reward: float
    discount: float
    transition_kind: str


@dataclass(frozen=True, slots=True)
class HistoryRoot:
    """A generated root and the complete path used to create it."""

    state: Any
    seed: int
    requested_hops: int
    steps: tuple[HistoryStep, ...]
    next_player: str
    final_time: float
    stop_reason: str

    @property
    def achieved_hops(self) -> int:
        return len(self.steps)

    @property
    def complete(self) -> bool:
        return self.achieved_hops == self.requested_hops


class HistoryRootGenerator:
    """Apply exactly one uniformly sampled canonical action per history hop."""

    __slots__ = ("runtime",)

    def __init__(self, runtime: _HistoryRuntime) -> None:
        self.runtime = runtime

    def generate(
        self,
        *,
        hops: int,
        seed: int,
        initial_state: Any | None = None,
        horizon_time: float | None = None,
    ) -> HistoryRoot:
        """Return a deterministic random root without mutating the source state.

        A forced one-action turn still counts as one hop. Automatic time,
        pipeline, or completion work performed inside that action does not.
        Generation stops early only at the optional horizon or when the runtime
        exposes no legal canonical action.
        """

        if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
            raise HistoryRootError("hops must be a nonnegative integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise HistoryRootError("seed must be a nonnegative integer")
        if horizon_time is not None and (
            not math.isfinite(horizon_time) or horizon_time < 0.0
        ):
            raise HistoryRootError("horizon_time must be finite and nonnegative")

        source = (
            self.runtime.initial_state() if initial_state is None else initial_state
        )
        state = self.runtime.clone_state(source)
        rng = random.Random(seed)
        steps: list[HistoryStep] = []
        stop_reason = "requested_hops_reached"

        while len(steps) < hops:
            now = float(self.runtime.state_time(state))
            if not math.isfinite(now) or now < 0.0:
                raise HistoryRootError("runtime exposed an invalid simulation time")
            if horizon_time is not None and now >= horizon_time:
                stop_reason = "horizon_reached"
                break

            actions = tuple(self.runtime.canonical_actions(state))
            if not actions:
                stop_reason = "no_legal_actions"
                break

            # Sorting makes the seed independent of container iteration order.
            actions = tuple(
                sorted(
                    actions,
                    key=lambda item: (
                        int(item.canonical_action_index),
                        int(item.representative_raw_index),
                    ),
                )
            )
            action = actions[rng.randrange(len(actions))]
            expected_player = self.runtime.player_to_move(state)
            if action.player != expected_player:
                raise HistoryRootError("action player does not match the state turn")

            edge = self.runtime.apply_action(state, action)
            if edge.player != expected_player:
                raise HistoryRootError("applied edge reports the wrong acting player")
            if not all(
                math.isfinite(value)
                for value in (
                    edge.started_at,
                    edge.finished_at,
                    edge.reward,
                    edge.discount,
                )
            ):
                raise HistoryRootError("history action produced a non-finite edge")
            if edge.finished_at < edge.started_at:
                raise HistoryRootError("history action moved time backwards")
            if not 0.0 <= edge.discount <= 1.0:
                raise HistoryRootError("history action produced an invalid discount")
            if edge.next_player != self.runtime.player_to_move(edge.state):
                raise HistoryRootError("history edge reports the wrong next player")
            if not math.isclose(
                edge.finished_at,
                self.runtime.state_time(edge.state),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise HistoryRootError("history edge time does not match its state")

            steps.append(
                HistoryStep(
                    hop=len(steps) + 1,
                    player=edge.player,
                    canonical_action_index=int(action.canonical_action_index),
                    representative_raw_index=int(action.representative_raw_index),
                    equivalent_raw_indices=tuple(action.equivalent_raw_indices),
                    started_at=float(edge.started_at),
                    finished_at=float(edge.finished_at),
                    reward=float(edge.reward),
                    discount=float(edge.discount),
                    transition_kind=str(edge.transition_kind),
                )
            )
            state = edge.state

        return HistoryRoot(
            state=state,
            seed=seed,
            requested_hops=hops,
            steps=tuple(steps),
            next_player=self.runtime.player_to_move(state),
            final_time=float(self.runtime.state_time(state)),
            stop_reason=stop_reason,
        )
