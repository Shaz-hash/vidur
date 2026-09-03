"""Deterministic planners used only by the scheduler integration smoke."""

from __future__ import annotations

from dataclasses import replace

from .scheduler_contract import LiveStateSnapshot, SJFPlanner, SchedulePlan


def valid_sjf256(snapshot: LiveStateSnapshot) -> SchedulePlan:
    return SJFPlanner(256).plan(snapshot)


def stale_sjf256(snapshot: LiveStateSnapshot) -> SchedulePlan:
    return replace(SJFPlanner(256).plan(snapshot), state_fingerprint="stale")
