"""Load a synchronous GV3 planner without coupling vLLM to one model runtime."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Mapping

from .scheduler_contract import (
    CanonicalAllocation,
    LiveStateSnapshot,
    Planner,
    RequestPhase,
    SchedulePlan,
)


class CallablePlanner:
    def __init__(self, callback: Callable[[LiveStateSnapshot], object]) -> None:
        self._callback = callback

    def plan(self, snapshot: LiveStateSnapshot) -> SchedulePlan:
        return coerce_schedule_plan(self._callback(snapshot), snapshot=snapshot)


def _phase(raw: object) -> RequestPhase:
    if isinstance(raw, RequestPhase):
        return raw
    return RequestPhase(str(raw).strip().lower())


def _allocation_from_mapping(raw: Mapping[str, object]) -> CanonicalAllocation:
    return CanonicalAllocation(
        request_id=str(raw["request_id"]),
        phase=_phase(raw["phase"]),
        num_tokens=int(raw["num_tokens"]),
    )


def coerce_schedule_plan(raw: object, *, snapshot: LiveStateSnapshot) -> SchedulePlan:
    if isinstance(raw, SchedulePlan):
        return raw
    if isinstance(raw, Mapping):
        allocations = tuple(
            _allocation_from_mapping(item)
            for item in list(raw.get("allocations", ()))
            if isinstance(item, Mapping)
        )
        return SchedulePlan(
            state_fingerprint=str(raw.get("state_fingerprint", snapshot.fingerprint)),
            allocations=allocations,
            policy=str(raw.get("policy", "gv3-controller")),
            details=(
                dict(raw.get("details", {}))
                if isinstance(raw.get("details", {}), Mapping)
                else None
            ),
        )

    # This path accepts the existing GV3 ControllerAction dataclass while
    # keeping the vLLM package independent of Game_Version3 imports.
    prefill = getattr(raw, "prefill_allocations", None)
    decode = getattr(raw, "decode_allocations", None)
    if isinstance(prefill, Mapping) and isinstance(decode, Mapping):
        allocations: list[CanonicalAllocation] = []
        for request_id, num_tokens in decode.items():
            allocations.append(
                CanonicalAllocation(str(request_id), RequestPhase.DECODE, int(num_tokens))
            )
        for request_id, num_tokens in prefill.items():
            allocations.append(
                CanonicalAllocation(str(request_id), RequestPhase.PREFILL, int(num_tokens))
            )
        return SchedulePlan(
            state_fingerprint=snapshot.fingerprint,
            allocations=tuple(allocations),
            policy="gv3-controller",
            details={"controller_action_repr": repr(raw)},
        )
    raise TypeError(
        "planner must return SchedulePlan, a plan mapping, or a GV3 ControllerAction"
    )


def load_planner(spec: str) -> Planner:
    """Load ``module:attribute``; classes may expose ``from_environment``."""

    module_name, separator, attribute_name = str(spec).strip().partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("planner must use module:attribute syntax")
    target = getattr(importlib.import_module(module_name), attribute_name)
    if inspect.isclass(target):
        factory = getattr(target, "from_environment", None)
        instance = factory() if callable(factory) else target()
    else:
        instance = target
    plan_method = getattr(instance, "plan", None)
    if callable(plan_method):
        return CallablePlanner(plan_method)
    if callable(instance):
        return CallablePlanner(instance)
    raise TypeError(f"planner {spec!r} is neither callable nor exposes plan(snapshot)")
