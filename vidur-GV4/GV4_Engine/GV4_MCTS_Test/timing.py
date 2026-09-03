"""Timing providers for production-profile and lightweight harness runs."""

from __future__ import annotations

from ..action_resolver import ResolvedControllerAction
from ..config import GV4EngineConfig
from ..state import GV4State
from ..vidur_timing_provider import VidurTimingProvider
from .config import GV4MCTSTestConfig


class DeterministicTimingProvider:
    """Small deterministic oracle used only to smoke-test the trace machinery."""

    __slots__ = ("_stage_count",)

    def __init__(self, stage_count: int) -> None:
        self._stage_count = stage_count

    def __call__(
        self, state: GV4State, action: ResolvedControllerAction
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        del state
        tokens = sum(item.total_tokens for item in action.allocations)
        sequences = len(action.allocations)
        base = 0.001 + tokens * 0.000002 + sequences * 0.00001
        service = tuple(
            base * (1.0 + stage * 0.05) for stage in range(self._stage_count)
        )
        communication = tuple(
            0.0002 + sequences * 0.000001 for _ in range(max(0, self._stage_count - 1))
        )
        return service, communication

    def estimate_prefill_time(self, tokens: int) -> float:
        if tokens <= 0:
            raise ValueError("prefill tokens must be positive")
        base = 0.001 + tokens * 0.000002 + 0.00001
        service = sum(base * (1.0 + stage * 0.05) for stage in range(self._stage_count))
        communication = max(0, self._stage_count - 1) * 0.000201
        return service + communication


def build_timing_provider(
    test: GV4MCTSTestConfig,
    engine: GV4EngineConfig,
) -> VidurTimingProvider | DeterministicTimingProvider:
    """Use real Vidur timings by default; deterministic mode is an explicit smoke mode."""

    if test.timing_mode == "deterministic":
        return DeterministicTimingProvider(engine.topology.pipeline_parallel_size)
    return engine.create_vidur_timing_provider()
