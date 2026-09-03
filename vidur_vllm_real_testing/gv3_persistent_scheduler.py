"""vLLM 0.13 scheduler lifecycle hook for :mod:`gv3_live_adapter`.

This class deliberately leaves stock KV allocation, preemption, and output
construction in vLLM.  It adds only two things: applying GV3 stop decisions
before scheduling and reporting completed real batches to the persistent
adapter with measured service time.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from typing import Any

from vllm.v1.request import RequestStatus

from .batch_execution_comparison import build_batch_shape

from .gv3_live_adapter import (
    BatchExecutionObservation,
    BatchRequestProgress,
)
from .scheduler_contract import (
    LiveStateSnapshot,
    RequestPhase,
    SchedulePlan,
    SchedulerMode,
    TraceArrivalGroupBarrier,
    ValidatedSchedulePlan,
    validate_and_project_plan,
)
from .vllm_live_state import logical_decode_tokens_from_vllm
from .vllm_gpu_timing_channel import GPUForwardTimingReader
from .vllm_scheduler import GV3Scheduler


class GV3PersistentScheduler(GV3Scheduler):
    """GV3Scheduler with completed-batch feedback to one persistent planner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vidur_batch_duration_source = os.environ.get(
            "VIDUR_VLLM_GV3_BATCH_DURATION_SOURCE", "engine_wall"
        ).strip()
        if self._vidur_batch_duration_source not in {"engine_wall", "gpu_forward"}:
            raise ValueError(
                "VIDUR_VLLM_GV3_BATCH_DURATION_SOURCE must be "
                "engine_wall or gpu_forward"
            )
        self._vidur_gpu_timing_reader: GPUForwardTimingReader | None = None
        if self._vidur_batch_duration_source == "gpu_forward":
            timing_path = os.environ.get("VIDUR_VLLM_GPU_TIMING_LOG", "").strip()
            if not timing_path:
                raise ValueError(
                    "VIDUR_VLLM_GPU_TIMING_LOG is required for gpu_forward timing"
                )
            self._vidur_gpu_timing_reader = GPUForwardTimingReader(timing_path)
        self._vidur_pending_execution: dict[str, Any] | None = None
        self._vidur_arrival_barrier = (
            None
            if self._vidur_registry is None
            else TraceArrivalGroupBarrier(
                self._vidur_registry,
                timeout_s=float(
                    os.environ.get("VIDUR_VLLM_ARRIVAL_GROUP_TIMEOUT_S", "10.0")
                ),
            )
        )

    def _lifecycle_target(self) -> object | None:
        planner = getattr(self, "_vidur_planner", None)
        callback = getattr(planner, "_callback", None)
        owner = getattr(callback, "__self__", None)
        if owner is not None:
            return owner
        return planner

    @staticmethod
    def _terminated_ids(plan: object) -> tuple[str, ...]:
        details = getattr(plan, "details", None)
        if not isinstance(details, dict):
            try:
                details = dict(details or {})
            except Exception:
                return ()
        return tuple(str(request_id) for request_id in details.get("terminated_request_ids", ()))

    def _record_pending_execution(
        self,
        *,
        snapshot: LiveStateSnapshot,
        output: Any,
    ) -> None:
        scheduled = {
            str(request_id): int(tokens)
            for request_id, tokens in dict(output.num_scheduled_tokens).items()
        }
        if not scheduled:
            self._vidur_pending_execution = None
            return
        self._vidur_pending_execution = {
            "snapshot": snapshot,
            "scheduled_tokens": scheduled,
            "scheduled_monotonic_s": time.monotonic(),
        }

    def schedule(self, throttle_prefills: bool = False) -> Any:
        if self._vidur_mode is SchedulerMode.STOCK:
            return super().schedule(throttle_prefills)
        if self._has_only_warmup_requests():
            return super(GV3Scheduler, self).schedule(throttle_prefills)
        if self._vidur_pending_execution is not None:
            raise RuntimeError("vLLM requested another schedule before the prior batch completed")

        decision_id = next(self._vidur_decision_ids)
        snapshot = self._snapshot()
        planning_started = time.monotonic()
        self._vidur_logger.begin_planning(planning_started)
        validated: ValidatedSchedulePlan | None = None
        try:
            assert self._vidur_planner is not None
            pending_arrivals = {}
            if (
                self._vidur_arrival_barrier is not None
                and self._vidur_mode
                in {SchedulerMode.CONTROLLER, SchedulerMode.SJF_256, SchedulerMode.SJF_512}
            ):
                pending_arrivals = self._vidur_arrival_barrier.pending_missing(
                    (request.request_id for request in snapshot.requests),
                    now_s=time.monotonic(),
                )
            if pending_arrivals:
                barrier_plan = SchedulePlan(
                    state_fingerprint=snapshot.fingerprint,
                    allocations=(),
                    policy="gv3-arrival-group-barrier",
                    details={
                        "pending_arrival_groups": {
                            str(arrived_at_s): list(missing_ids)
                            for arrived_at_s, missing_ids in pending_arrivals.items()
                        }
                    },
                )
                validated = ValidatedSchedulePlan(
                    source=barrier_plan,
                    allocations=(),
                )
                planning_s = time.monotonic() - planning_started
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
                self._record_pending_execution(snapshot=snapshot, output=output)
                return output

            planning_snapshot = snapshot
            terminated_ids: set[str] = set()
            zero_work_replans = 0
            max_zero_work_replans = max(1, len(snapshot.requests) + 1)
            while True:
                plan = self._vidur_planner.plan(planning_snapshot)
                terminated = self._terminated_ids(plan)
                before_ids = set(planning_snapshot.by_id())
                if terminated and self._vidur_mode is not SchedulerMode.SHADOW:
                    self.finish_requests(terminated, RequestStatus.FINISHED_STOPPED)
                    terminated_ids.update(terminated)

                current_snapshot = self._snapshot()
                if current_snapshot.fingerprint != plan.state_fingerprint:
                    live_ids = set(current_snapshot.by_id())
                    plan = replace(
                        plan,
                        state_fingerprint=current_snapshot.fingerprint,
                        allocations=tuple(
                            allocation
                            for allocation in plan.allocations
                            if allocation.request_id in live_ids
                        ),
                    )

                # Eviction is a zero-time GV3 transition. If it removes the
                # last admitted work while later trace requests are already
                # visible to vLLM, immediately replan so the adapter can
                # advance its virtual clock to the next arrival.
                if (
                    self._vidur_mode is not SchedulerMode.SHADOW
                    and terminated
                    and not plan.allocations
                    and current_snapshot.requests
                ):
                    after_ids = set(current_snapshot.by_id())
                    if len(after_ids) >= len(before_ids):
                        raise RuntimeError(
                            "eviction-only GV3 transition did not reduce live requests"
                        )
                    zero_work_replans += 1
                    if zero_work_replans > max_zero_work_replans:
                        raise RuntimeError(
                            "too many consecutive eviction-only GV3 transitions"
                        )
                    planning_snapshot = current_snapshot
                    continue
                break

            if terminated_ids:
                details = dict(plan.details or {})
                details["terminated_request_ids"] = sorted(terminated_ids)
                if zero_work_replans:
                    details["zero_work_replans"] = int(zero_work_replans)
                plan = replace(plan, details=details)
            validated = validate_and_project_plan(plan, current_snapshot)
        except Exception as exc:
            planning_s = time.monotonic() - planning_started
            if self._vidur_mode in {
                SchedulerMode.SHADOW,
                SchedulerMode.ACTIVE_VALIDATION,
            }:
                output = super(GV3Scheduler, self).schedule(throttle_prefills)
                self._log(
                    decision_id=decision_id,
                    snapshot=snapshot,
                    validated=None,
                    planning_s=planning_s,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                    output=output,
                    applied=False,
                )
                self._record_pending_execution(snapshot=snapshot, output=output)
                return output
            self._vidur_logger.cancel_planning()
            raise

        planning_s = time.monotonic() - planning_started
        if self._vidur_mode is SchedulerMode.SHADOW:
            output = super(GV3Scheduler, self).schedule(throttle_prefills)
            self._log(
                decision_id=decision_id,
                snapshot=snapshot,
                validated=validated,
                planning_s=planning_s,
                fallback_reason=None,
                output=output,
                applied=False,
            )
            self._record_pending_execution(snapshot=snapshot, output=output)
            return output

        assert validated is not None
        execution_snapshot = self._snapshot()
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
        self._record_pending_execution(snapshot=execution_snapshot, output=output)
        return output

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> Any:
        pending = self._vidur_pending_execution
        if pending is None:
            return super().update_from_output(scheduler_output, model_runner_output)

        expected = dict(pending["scheduled_tokens"])
        actual = {
            str(request_id): int(tokens)
            for request_id, tokens in dict(scheduler_output.num_scheduled_tokens).items()
        }
        if actual != expected:
            raise RuntimeError(
                f"completed scheduler output differs from recorded batch: {actual} != {expected}"
            )
        request_refs = {
            request_id: self.requests.get(request_id) for request_id in expected
        }
        completed_monotonic_s = time.monotonic()
        result = super().update_from_output(scheduler_output, model_runner_output)

        before_by_id = pending["snapshot"].by_id()
        progress: list[BatchRequestProgress] = []
        for request_id in expected:
            before = before_by_id.get(request_id)
            request = self.requests.get(request_id) or request_refs.get(request_id)
            if before is None or request is None:
                raise RuntimeError(
                    f"request {request_id}: missing lifecycle state for completed batch"
                )
            # schedule() advances this counter optimistically. Derive the
            # completed progress from the exact validated batch instead.
            progress.append(
                BatchRequestProgress(
                    request_id=request_id,
                    phase_before=before.phase,
                    num_computed_tokens_before=int(before.num_computed_tokens),
                    num_output_tokens_before=int(before.num_output_tokens),
                    num_computed_tokens_after=(
                        int(before.num_computed_tokens) + int(expected[request_id])
                    ),
                    num_output_tokens_after=logical_decode_tokens_from_vllm(
                        int(request.num_output_tokens),
                        implicit_prefill_output_tokens=(
                            self._vidur_implicit_prefill_output_tokens
                        ),
                    ),
                    finished_after=request_id not in self.requests,
                )
            )

        target = self._lifecycle_target()
        callback = getattr(target, "on_batch_completed", None)
        if callable(callback):
            state_payload = getattr(target, "state_payload", None)
            if not callable(state_payload):
                raise RuntimeError(
                    "persistent GV3 adapter does not expose state_payload"
                )
            state_before = state_payload()
            batch_shape = build_batch_shape(pending["snapshot"], expected)
            scheduled_monotonic_s = float(pending["scheduled_monotonic_s"])
            engine_batch_wall_s = max(
                0.0, completed_monotonic_s - scheduled_monotonic_s
            )
            gpu_forward_s = None
            if self._vidur_batch_duration_source == "gpu_forward":
                assert self._vidur_gpu_timing_reader is not None
                gpu_forward_s = self._vidur_gpu_timing_reader.read_for_batch(expected)
                authoritative_duration_s = gpu_forward_s
            else:
                authoritative_duration_s = engine_batch_wall_s
            callback(
                BatchExecutionObservation(
                    scheduled_monotonic_s=scheduled_monotonic_s,
                    completed_monotonic_s=completed_monotonic_s,
                    duration_s=authoritative_duration_s,
                    scheduled_tokens_by_request=expected,
                    request_progress=tuple(progress),
                    gpu_forward_s=gpu_forward_s,
                    engine_batch_wall_s=engine_batch_wall_s,
                )
            )
            log_path = getattr(self._vidur_logger, "path", None)
            if log_path is not None:
                audit_path = Path(f"{log_path}.batches.jsonl")
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "scheduled_tokens": expected,
                                "batch_shape": batch_shape,
                                "state_before": state_before,
                                "request_progress": [asdict(row) for row in progress],
                                "timing": {
                                    "source": self._vidur_batch_duration_source,
                                    "authoritative_duration_s": authoritative_duration_s,
                                    "gpu_forward_s": gpu_forward_s,
                                    "engine_batch_wall_s": engine_batch_wall_s,
                                },
                                "state_after": state_payload(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        self._vidur_pending_execution = None
        return result

