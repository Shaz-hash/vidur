"""Shared failure reporting for readable, grouped trace checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any

from ..config import GV4EngineConfig
from .trace_context import TraceNode


class GV4TraceValidationError(AssertionError):
    """Raised after all trace groups run and one or more checks failed."""


@dataclass(slots=True)
class ValidationReport:
    checks_by_group: Counter[str] = field(default_factory=Counter)
    coverage: Counter[str] = field(default_factory=Counter)
    failures: list[str] = field(default_factory=list)
    max_failures: int = 200

    def check(
        self,
        group: str,
        name: str,
        condition: bool,
        node: TraceNode | None = None,
        detail: str = "",
    ) -> None:
        self.checks_by_group[group] += 1
        if condition or len(self.failures) >= self.max_failures:
            return
        location = ""
        if node is not None:
            location = (
                f" run={node.key.run_id} root={node.key.root_id} "
                f"node={node.key.node_id} depth={node.depth}"
            )
        suffix = f": {detail}" if detail else ""
        self.failures.append(f"[{group}] {name}{location}{suffix}")

    def close(
        self,
        group: str,
        name: str,
        actual: float,
        expected: float,
        node: TraceNode | None = None,
        *,
        abs_tol: float = 1e-9,
        rel_tol: float = 1e-8,
    ) -> None:
        self.check(
            group,
            name,
            math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol),
            node,
            detail=f"actual={actual!r}, expected={expected!r}",
        )

    def cover(self, event: str, amount: int = 1) -> None:
        self.coverage[event] += amount

    @property
    def total_checks(self) -> int:
        return sum(self.checks_by_group.values())

    def assert_clean(self) -> None:
        if not self.failures:
            return
        shown = "\n".join(self.failures[:20])
        remainder = len(self.failures) - 20
        if remainder > 0:
            shown += f"\n... {remainder} additional failures"
        raise GV4TraceValidationError(shown)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "checks_by_group": dict(sorted(self.checks_by_group.items())),
            "coverage": dict(sorted(self.coverage.items())),
            "failure_count": len(self.failures),
            "failures": self.failures,
        }


@dataclass(frozen=True, slots=True)
class ValidationContext:
    config: GV4EngineConfig
    timing_provider: Any

    @property
    def epsilon(self) -> float:
        return self.config.timing.epsilon
