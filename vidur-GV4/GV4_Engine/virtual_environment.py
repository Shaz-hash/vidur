"""Thin MCTS-facing facade for the compact GV4 transition engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .action_resolver import (
    CanonicalAdversaryAction,
    CanonicalControllerAction,
    ControllerTransitionKind,
    resolve_adversary_actions,
    resolve_controller_actions,
)
from .config import GV4EngineConfig
from .fast_forward import BatchTimingProvider, fast_forward_decode_only_to_next_tick
from .kv_ledger import free_logical_blocks
from .state import GV4State, Player
from .transition_engine import apply_adversary_action, apply_controller_action


PrefillTimeEstimator = Callable[[int], float]


class GV4VirtualVidurMCTSEnvironment:
    """Own callbacks and delegate all state changes to pure GV4 engine modules."""

    __slots__ = ("config", "_batch_timing", "_prefill_time")

    def __init__(
        self,
        config: GV4EngineConfig,
        *,
        batch_timing_provider: BatchTimingProvider,
        prefill_time_estimator: PrefillTimeEstimator,
    ) -> None:
        self.config = config
        self._batch_timing = batch_timing_provider
        self._prefill_time = prefill_time_estimator

    @classmethod
    def from_config(
        cls,
        config: GV4EngineConfig,
        *,
        max_timing_cache_entries: int = 65_536,
    ) -> "GV4VirtualVidurMCTSEnvironment":
        """Build Vidur's predictor/cache and return a ready GV4 environment."""

        timing = config.create_vidur_timing_provider(
            max_cache_entries=max_timing_cache_entries
        )
        return cls(
            config,
            batch_timing_provider=timing,
            prefill_time_estimator=timing.estimate_prefill_time,
        )

    def initial_state(
        self,
        *,
        now: float = 0.0,
        next_player: Player = Player.ADVERSARY,
    ) -> GV4State:
        return GV4State.initial(self.config, now=now, next_player=next_player)

    def _request_prefill_time(self, _request: object, tokens: int) -> float:
        return float(self._prefill_time(tokens))

    def sample_controller_actions(
        self, state: GV4State, *, replica_id: int = 0
    ) -> tuple[
        tuple[CanonicalControllerAction | None, ...],
        tuple[bool, ...],
    ]:
        """Return canonical edges indexed by the fixed raw policy dimension."""

        raw, canonical = resolve_controller_actions(
            state,
            self.config,
            replica_id=replica_id,
            prefill_time_estimator=self._request_prefill_time,
        )
        by_raw: list[CanonicalControllerAction | None] = [None] * len(raw)
        for edge in canonical:
            for raw_index in edge.equivalent_raw_indices:
                by_raw[raw_index] = edge
        return tuple(by_raw), tuple(action is not None for action in by_raw)

    def sample_adversary_actions(self, state: GV4State) -> tuple[
        tuple[CanonicalAdversaryAction | None, ...],
        tuple[bool, ...],
    ]:
        raw, canonical = resolve_adversary_actions(state, self.config)
        by_raw: list[CanonicalAdversaryAction | None] = [None] * len(raw)
        for edge in canonical:
            for raw_index in edge.equivalent_raw_indices:
                by_raw[raw_index] = edge
        return tuple(by_raw), tuple(action is not None for action in by_raw)

    def apply_controller_action_only(
        self,
        state: GV4State,
        action: CanonicalControllerAction,
        *,
        inplace: bool = False,
        fast_forward: bool = True,
    ) -> GV4State:
        has_memory_action = bool(
            action.action.evicted_request_ids
            or action.action.preempted_request_ids
        )
        service = communication = None
        if action.action.transition_kind == ControllerTransitionKind.BATCH:
            service, communication = self._batch_timing(state, action.action)
        result = apply_controller_action(
            state,
            self.config,
            action,
            stage_service_times=None if service is None else tuple(service),
            pp_communication_times=(
                None if communication is None else tuple(communication)
            ),
            prefill_time_estimator=self._request_prefill_time,
            inplace=inplace,
        ).state
        fully_idle = all(
            request.lifecycle.is_terminal for request in result.requests
        ) and all(not replica.inflight_microbatches for replica in result.replicas)
        if fast_forward and (not has_memory_action or fully_idle):
            result = fast_forward_decode_only_to_next_tick(
                result,
                self.config,
                timing_provider=self._batch_timing,
                inplace=True,
            )
        return result

    def apply_adversary_action_only(
        self,
        state: GV4State,
        action: CanonicalAdversaryAction,
        *,
        inplace: bool = False,
    ) -> GV4State:
        return apply_adversary_action(
            state,
            self.config,
            action,
            prefill_time_estimator=self._prefill_time,
            inplace=inplace,
        ).state

    def evaluate_objective(self, state: GV4State) -> tuple[int, float]:
        return state.objective.slo_violations, state.objective.total_cost

    def describe_state(self, state: GV4State) -> dict[str, Any]:
        lifecycle_counts: dict[str, int] = {}
        for request in state.requests:
            name = request.lifecycle.name
            lifecycle_counts[name] = lifecycle_counts.get(name, 0) + 1
        return {
            "now": state.now,
            "next_player": state.next_player.name,
            "next_adversary_tick": state.next_adversary_tick,
            "requests": lifecycle_counts,
            "decode_credits": {
                "available": state.decode_credits_available,
                "reserved": state.decode_credits_reserved,
                "committed": state.decode_tokens_committed_total,
            },
            "replicas": tuple(
                {
                    "replica_id": replica.replica_id,
                    "inflight_microbatches": replica.inflight_count,
                    "free_kv_blocks": free_logical_blocks(replica),
                    "stage_tail_finish_times": tuple(replica.stage_tail_finish_times),
                }
                for replica in state.replicas
            ),
            "objective": {
                "slo_violations": state.objective.slo_violations,
                "total_cost": state.objective.total_cost,
            },
        }


GV4VirtualEnvironment = GV4VirtualVidurMCTSEnvironment
