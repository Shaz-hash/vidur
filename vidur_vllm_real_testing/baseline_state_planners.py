"""Deterministic baseline planners for the persistent real-GPU GV3 adapter."""

from __future__ import annotations

from typing import Any, Mapping

from .scheduler_contract import LiveStateSnapshot, SJFPlanner, SchedulePlan


def sjf256(
    state_payload: Mapping[str, Any], snapshot: LiveStateSnapshot
) -> SchedulePlan:
    del state_payload
    return SJFPlanner(256).plan(snapshot)


def sjf512(
    state_payload: Mapping[str, Any], snapshot: LiveStateSnapshot
) -> SchedulePlan:
    del state_payload
    return SJFPlanner(512).plan(snapshot)
