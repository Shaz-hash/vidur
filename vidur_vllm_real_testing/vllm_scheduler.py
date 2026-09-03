"""Pinned vLLM 0.13 scheduler hook for GV3, shadow, and SJF baselines."""

from __future__ import annotations

import itertools
import os
import time
from typing import Any, Iterable

from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import Scheduler

from .patch_vllm_scheduler import verify_patch
from .scheduler_contract import (
    LiveStateSnapshot,
    SchedulerMode,
    SJFPlanner,
    TraceMetadataRegistry,
    ValidatedSchedulePlan,
    exclusively_matches_request_prefix,
    plan_as_dict,
    validate_and_project_plan,
)
from .scheduler_logging import SchedulerDecisionLogger
from .scheduler_planner import load_planner
from .vllm_live_state import (
    DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS,
    build_live_state_snapshot,
)


def _ordered_requests(requests: Iterable[Any], request_ids: Iterable[str]) -> list[Any]:
    by_id = {str(request.request_id): request for request in requests}
    return [by_id[request_id] for request_id in request_ids if request_id in by_id]


def _new_queue(policy: SchedulingPolicy, requests: Iterable[Any]) -> Any:
    queue = create_request_queue(policy)
    for request in requests:
        queue.add_request(request)
    return queue


class GV3Scheduler(Scheduler):
    """Constrain stock scheduling with a validated per-request token plan.

    The base scheduler still owns KV allocation, preemption, encoder handling,
    connector metadata, and SchedulerOutput construction. This class controls
    only which live requests are visible for one call and their token caps.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        verify_patch()
        self._vidur_mode = SchedulerMode.parse(
            os.environ.get("VIDUR_VLLM_SCHEDULER_MODE", SchedulerMode.STOCK.value)
        )
        trace_path = os.environ.get("VIDUR_VLLM_CANONICAL_TRACE", "").strip()
        self._vidur_registry = (
            TraceMetadataRegistry.load(trace_path) if trace_path else None
        )
        log_path = os.environ.get("VIDUR_VLLM_SCHEDULER_LOG", "").strip()
        self._vidur_logger = SchedulerDecisionLogger(log_path or None)
        self._vidur_decision_ids = itertools.count(1)
        self._vidur_request_token_caps: dict[str, int] | None = None
        self._vidur_warmup_request_prefix = os.environ.get(
            "VIDUR_VLLM_WARMUP_REQUEST_PREFIX", ""
        ).strip()
        self._vidur_implicit_prefill_output_tokens = int(
            os.environ.get(
                "VIDUR_VLLM_GV3_IMPLICIT_PREFILL_OUTPUT_TOKENS",
                str(DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS),
            )
        )
        if self._vidur_implicit_prefill_output_tokens not in {0, 1}:
            raise ValueError(
                "VIDUR_VLLM_GV3_IMPLICIT_PREFILL_OUTPUT_TOKENS must be 0 or 1"
            )

        if self._vidur_mode is SchedulerMode.SJF_256:
            self._vidur_planner = SJFPlanner(256)
        elif self._vidur_mode is SchedulerMode.SJF_512:
            self._vidur_planner = SJFPlanner(512)
        elif self._vidur_mode in {
            SchedulerMode.SHADOW,
            SchedulerMode.ACTIVE_VALIDATION,
            SchedulerMode.CONTROLLER,
        }:
            planner_spec = os.environ.get("VIDUR_VLLM_GV3_PLANNER", "").strip()
            if not planner_spec:
                raise RuntimeError(
                    f"{self._vidur_mode.value} mode requires VIDUR_VLLM_GV3_PLANNER="
                    "module:attribute"
                )
            self._vidur_planner = load_planner(planner_spec)
        else:
            self._vidur_planner = None

        if self._vidur_mode is not SchedulerMode.STOCK:
            if self._vidur_registry is None:
                raise RuntimeError(
                    f"{self._vidur_mode.value} mode requires VIDUR_VLLM_CANONICAL_TRACE"
                )
            if self.policy is not SchedulingPolicy.FCFS:
                raise RuntimeError("GV3 scheduler modes require vLLM scheduling-policy=fcfs")
            if self.cache_config.enable_prefix_caching:
                raise RuntimeError("initial GV3 real-vLLM tests require prefix caching disabled")
            if not self.scheduler_config.enable_chunked_prefill:
                raise RuntimeError("GV3 scheduler modes require chunked prefill enabled")

    def _snapshot(self) -> LiveStateSnapshot:
        assert self._vidur_registry is not None
        return build_live_state_snapshot(
            running=self.running,
            waiting=self.waiting,
            registry=self._vidur_registry,
            max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            implicit_prefill_output_tokens=(
                self._vidur_implicit_prefill_output_tokens
            ),
        )

    def _has_only_warmup_requests(self) -> bool:
        requests = list(self.running) + list(self.waiting)
        return exclusively_matches_request_prefix(
            (str(request.request_id) for request in requests),
            self._vidur_warmup_request_prefix,
        )

    def _log(
        self,
        *,
        decision_id: int,
        snapshot: LiveStateSnapshot | None,
        validated: ValidatedSchedulePlan | None,
        planning_s: float,
        fallback_reason: str | None,
        output: Any,
        applied: bool,
    ) -> None:
        logged_monotonic_s = time.monotonic()
        self._vidur_logger.write(
            {
                "decision_id": int(decision_id),
                "mode": self._vidur_mode.value,
                "decision_started_monotonic_s": max(
                    0.0, logged_monotonic_s - float(planning_s)
                ),
                "decision_finished_monotonic_s": logged_monotonic_s,
                "live_request_ids": (
                    []
                    if snapshot is None
                    else [
                        self._vidur_registry.require(request.request_id).request_id
                        for request in snapshot.requests
                    ]
                ),
                "live_state_fingerprint": (
                    None if snapshot is None else snapshot.fingerprint
                ),
                "plan": plan_as_dict(validated),
                "planning_s": float(planning_s),
                "fallback_reason": fallback_reason,
                "applied": bool(applied),
                "actual_num_scheduled_tokens": dict(output.num_scheduled_tokens),
                "actual_total_num_scheduled_tokens": int(
                    output.total_num_scheduled_tokens
                ),
            },
            blocking_s=float(planning_s),
        )

    def _apply_validated(
        self,
        validated: ValidatedSchedulePlan,
        *,
        throttle_prefills: bool = False,
    ) -> Any:
        original_running = list(self.running)
        original_waiting = list(self.waiting)
        planned_ids = list(validated.request_ids)
        planned_set = set(planned_ids)

        selected_running = _ordered_requests(original_running, planned_ids)
        selected_waiting = _ordered_requests(original_waiting, planned_ids)
        hidden_running = [
            request
            for request in original_running
            if str(request.request_id) not in planned_set
        ]
        hidden_waiting = [
            request
            for request in original_waiting
            if str(request.request_id) not in planned_set
        ]

        original_max_tokens = self.max_num_scheduled_tokens
        original_max_requests = self.max_num_running_reqs
        original_threshold = self.scheduler_config.long_prefill_token_threshold
        self.running = selected_running
        self.waiting = _new_queue(self.policy, selected_waiting)
        self.max_num_scheduled_tokens = validated.actual_token_budget
        self.max_num_running_reqs = max(
            len(selected_running),
            original_max_requests - len(hidden_running),
        )
        self.scheduler_config.long_prefill_token_threshold = 0
        self._vidur_request_token_caps = validated.actual_tokens_by_request()

        output = None
        try:
            output = super().schedule(throttle_prefills)
        finally:
            current_running = list(self.running)
            current_waiting = list(self.waiting)
            current_running_by_id = {
                str(request.request_id): request for request in current_running
            }
            restored_running: list[Any] = []
            for request in original_running:
                request_id = str(request.request_id)
                if request_id in planned_set:
                    if request_id in current_running_by_id:
                        restored_running.append(current_running_by_id.pop(request_id))
                else:
                    restored_running.append(request)
            restored_running.extend(current_running_by_id.values())
            self.running = restored_running

            current_waiting_by_id = {
                str(request.request_id): request for request in current_waiting
            }
            original_waiting_ids = {
                str(request.request_id) for request in original_waiting
            }
            preempted = [
                request
                for request in current_waiting
                if str(request.request_id) not in original_waiting_ids
            ]
            restored_waiting = list(preempted)
            for request in original_waiting:
                request_id = str(request.request_id)
                if request_id in planned_set:
                    if request_id in current_waiting_by_id:
                        restored_waiting.append(current_waiting_by_id[request_id])
                else:
                    restored_waiting.append(request)
            self.waiting = _new_queue(self.policy, restored_waiting)

            self._vidur_request_token_caps = None
            self.max_num_scheduled_tokens = original_max_tokens
            self.max_num_running_reqs = original_max_requests
            self.scheduler_config.long_prefill_token_threshold = original_threshold

        assert output is not None
        expected = validated.actual_tokens_by_request()
        actual = {str(key): int(value) for key, value in output.num_scheduled_tokens.items()}
        if actual != expected:
            raise RuntimeError(
                "vLLM could not apply the validated GV3 plan exactly; "
                f"expected={expected}, actual={actual}"
            )
        return output

    def schedule(self, throttle_prefills: bool = False) -> Any:
        if self._vidur_mode is SchedulerMode.STOCK:
            return super().schedule(throttle_prefills)
        if self._has_only_warmup_requests():
            return super().schedule(throttle_prefills)
        decision_id = next(self._vidur_decision_ids)

        snapshot = self._snapshot()
        planning_started = time.monotonic()
        self._vidur_logger.begin_planning(planning_started)
        validated = None
        try:
            assert self._vidur_planner is not None
            plan = self._vidur_planner.plan(snapshot)
            # Rebuild after planning so asynchronous planners cannot apply a
            # decision to a state that changed while they were working.
            current_snapshot = self._snapshot()
            validated = validate_and_project_plan(plan, current_snapshot)
        except Exception as exc:
            planning_s = time.monotonic() - planning_started
            if self._vidur_mode in {
                SchedulerMode.SHADOW,
                SchedulerMode.ACTIVE_VALIDATION,
            }:
                output = super().schedule(throttle_prefills)
                self._log(
                    decision_id=decision_id,
                    snapshot=snapshot,
                    validated=None,
                    planning_s=planning_s,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                    output=output,
                    applied=False,
                )
                return output
            self._vidur_logger.cancel_planning()
            raise

        planning_s = time.monotonic() - planning_started
        if self._vidur_mode is SchedulerMode.SHADOW:
            output = super().schedule(throttle_prefills)
            self._log(
                decision_id=decision_id,
                snapshot=snapshot,
                validated=validated,
                planning_s=planning_s,
                fallback_reason=None,
                output=output,
                applied=False,
            )
            return output

        output = self._apply_validated(
            validated, throttle_prefills=throttle_prefills
        )
        self._log(
            decision_id=decision_id,
            snapshot=snapshot,
            validated=validated,
            planning_s=planning_s,
            fallback_reason=None,
            output=output,
            applied=True,
        )
        return output
