"""Simple controller baselines expressed through the GV4 action contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from GV4_Engine.config import GV4EngineConfig

from ..engine_runtime import CanonicalActionRef


__all__ = [
    "BaselineSelection",
    "SJFPolicyConfig",
    "select_sjf_controller_action",
]


@dataclass(frozen=True, slots=True)
class SJFPolicyConfig:
    """The historical SJF256 policy: no eviction, 256 prefill tokens, SJF."""

    prefill_budget: int = 256
    eviction_rule: str = "evict_none"
    ordering_heuristic: str = "SJF"


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    action: CanonicalActionRef
    requested_raw_index: int
    matched_raw_index: int
    used_zero_budget_fallback: bool


def _raw_index(config: GV4EngineConfig, policy: SJFPolicyConfig, budget: int) -> int:
    actions = config.controller_actions
    try:
        rule = actions.eviction_rule_names.index(policy.eviction_rule)
        budget_index = actions.prefill_budget_options.index(budget)
        ordering = actions.ordering_heuristics.index(policy.ordering_heuristic)
    except ValueError as error:
        raise ValueError(
            "SJF policy is not representable by this GV4 action schema"
        ) from error
    return actions.encode_raw_index(rule, budget_index, ordering)


def _find_alias(
    actions: Sequence[CanonicalActionRef], raw_index: int
) -> CanonicalActionRef | None:
    return next(
        (action for action in actions if raw_index in action.equivalent_raw_indices),
        None,
    )


def select_sjf_controller_action(
    actions: Sequence[CanonicalActionRef],
    config: GV4EngineConfig,
    policy: SJFPolicyConfig = SJFPolicyConfig(),
) -> BaselineSelection:
    """Select SJF256 by raw semantics, even when GV4 aliases that raw action."""

    legal = tuple(actions)
    if not legal:
        raise ValueError("SJF cannot select from an empty action set")
    if any(action.player != "controller" for action in legal):
        raise ValueError("SJF received a non-controller action")

    requested = _raw_index(config, policy, policy.prefill_budget)
    selected = _find_alias(legal, requested)
    if selected is not None:
        return BaselineSelection(selected, requested, requested, False)

    # No prefill may currently be schedulable. Budget zero is the intended
    # fallback and still admits mandatory decode work under GV4 semantics.
    fallback = _raw_index(config, policy, 0)
    selected = _find_alias(legal, fallback)
    if selected is None:
        raise ValueError("neither SJF budget nor its zero-budget fallback is legal")
    return BaselineSelection(selected, requested, fallback, True)
