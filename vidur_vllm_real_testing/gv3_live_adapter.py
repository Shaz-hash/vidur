"""Persistent GV3 state bridge for real vLLM execution.

The adapter owns one canonical state for the lifetime of a benchmark.  Queue
snapshots validate that state; they do not replace it.  Real completed batch
durations advance the canonical clock, while native MCTS may fork the exported
payload and continue to use Vidur's frozen timing model for hypothetical work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import math
import os
from typing import Any, Callable, Mapping

from .canonicalization import CanonicalizationError
from .scheduler_contract import (
    CanonicalAllocation,
    LiveRequestSnapshot,
    LiveStateSnapshot,
    RequestPhase,
    SJFPlanner,
    SchedulePlan,
)


_EPS = 1e-9


@dataclass(frozen=True)
class BatchRequestProgress:
    request_id: str
    phase_before: RequestPhase
    num_computed_tokens_before: int
    num_output_tokens_before: int
    num_computed_tokens_after: int
    num_output_tokens_after: int
    finished_after: bool


@dataclass(frozen=True)
class BatchExecutionObservation:
    scheduled_monotonic_s: float
    completed_monotonic_s: float
    duration_s: float
    scheduled_tokens_by_request: Mapping[str, int]
    request_progress: tuple[BatchRequestProgress, ...]
    gpu_forward_s: float | None = None
    engine_batch_wall_s: float | None = None


@dataclass(frozen=True)
class GV3LiveAdapterConfig:
    adversary_tick_s: float = 0.2
    launch_window_s: float = 1.0
    decode_credit_mint: int = 216
    max_decode_tokens_per_request: int = 864
    auto_drop_lateness_s: float = 2.0
    time_round_digits: int = 10
    allow_midstream_initialization: bool = False

    def validate(self) -> None:
        if self.adversary_tick_s <= 0.0:
            raise ValueError("adversary_tick_s must be positive")
        if self.launch_window_s <= 0.0:
            raise ValueError("launch_window_s must be positive")
        if self.decode_credit_mint <= 0:
            raise ValueError("decode_credit_mint must be positive")
        if self.max_decode_tokens_per_request <= 0:
            raise ValueError("max_decode_tokens_per_request must be positive")
        if self.auto_drop_lateness_s <= 0.0:
            raise ValueError("auto_drop_lateness_s must be positive")


@dataclass
class _RequestLedger:
    request_id: str
    gv3_request_id: int
    arrived_at: float
    canonical_prefill_tokens: int
    canonical_decode_tokens: int
    actual_prefill_tokens: int
    actual_decode_tokens: int
    prefill_slo: float
    decode_slo: float
    processed_prefill: int = 0
    actual_processed_prefill: int = 0
    processed_decode: int = 0
    prefill_completed_at: float | None = None
    completed_at: float | None = None
    prefill_lateness: float = 0.0
    decode_lateness: float = 0.0
    decode_next_deadline: float | None = None
    decode_tokens_counted: int = 0
    prefill_lateness_finalized: bool = False
    completed: bool = False
    dropped: bool = False
    stopped_decode: bool = False
    violated: bool = False

    @property
    def prefill_complete(self) -> bool:
        return self.actual_processed_prefill >= self.actual_prefill_tokens

    @property
    def remaining_prefill(self) -> int:
        canonical = max(0, self.canonical_prefill_tokens - self.processed_prefill)
        actual = max(0, self.actual_prefill_tokens - self.actual_processed_prefill)
        if canonical == 0 and actual > 0:
            return max(128, actual)
        return canonical

    @property
    def remaining_decode(self) -> int:
        goal = min(self.canonical_decode_tokens, 864)
        return max(0, goal - self.processed_decode)

    @property
    def active(self) -> bool:
        return not self.completed and (self.remaining_prefill > 0 or self.remaining_decode > 0)

    def native_payload(self) -> dict[str, Any]:
        return {
            "request_id": int(self.gv3_request_id),
            "arrived_at": float(self.arrived_at),
            "queued_at": float(self.arrived_at),
            "num_prefill_tokens": int(self.processed_prefill + self.remaining_prefill),
            "num_processed_prefill_tokens": int(self.processed_prefill),
            "num_decode_tokens": int(self.canonical_decode_tokens),
            "num_processed_decode_tokens": int(self.processed_decode),
            "prefill_slo_time": float(self.prefill_slo),
            "decode_slo_time": float(self.decode_slo),
            "completion_slo_time": -1.0,
            "prefill_deadline": float(self.arrived_at + self.prefill_slo),
            "decode_next_deadline": (
                -1.0 if self.decode_next_deadline is None else float(self.decode_next_deadline)
            ),
            "prefill_completed_at": (
                -1.0 if self.prefill_completed_at is None else float(self.prefill_completed_at)
            ),
            "completed_at": -1.0 if self.completed_at is None else float(self.completed_at),
            "prefill_lateness": float(self.prefill_lateness),
            "decode_lateness": float(self.decode_lateness),
            "is_prefill_complete": bool(self.prefill_complete),
            "completed": bool(self.completed),
            "dropped": bool(self.dropped),
            "stopped_decode": bool(self.stopped_decode),
            "violated": bool(self.violated),
        }


StatePlanner = Callable[[Mapping[str, Any], LiveStateSnapshot], object]


def _load_state_planner(spec: str) -> StatePlanner:
    module_name, separator, attribute_name = str(spec).strip().partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("state planner must use module:attribute syntax")
    target = getattr(importlib.import_module(module_name), attribute_name)
    if inspect.isclass(target):
        factory = getattr(target, "from_environment", None)
        target = factory() if callable(factory) else target()
    method = getattr(target, "plan_state", None)
    if callable(method):
        return method
    if callable(target):
        return target
    raise TypeError(f"state planner {spec!r} is not callable")


class GV3PersistentAdapter:
    """One persistent real-execution state and one scheduling interface."""

    def __init__(
        self,
        *,
        controller: StatePlanner | None = None,
        fallback_policy: str | None = None,
        config: GV3LiveAdapterConfig = GV3LiveAdapterConfig(),
    ) -> None:
        config.validate()
        self.config = config
        self._controller = controller
        self._fallback_policy = None if fallback_policy is None else str(fallback_policy)
        self._requests: dict[str, _RequestLedger] = {}
        self._next_gv3_request_id = 0
        self._sim_time = 0.0
        self._next_adv_tick = 0.0
        self._last_adv_tick = -1.0
        self._decode_credit_balance = 0
        self._recent_launches: list[tuple[float, int, int]] = []
        self._requests_generated = 0
        self._requests_completed = 0
        self._slo_violations = 0
        self._slo_lateness_sum = 0.0
        self._pending_termination_ids: set[str] = set()
        self._initialized = False
        self._completed_batch_count = 0

    @classmethod
    def from_environment(cls) -> "GV3PersistentAdapter":
        state_planner_spec = os.environ.get("VIDUR_VLLM_GV3_STATE_PLANNER", "").strip()
        fallback = os.environ.get("VIDUR_VLLM_GV3_ADAPTER_POLICY", "").strip() or None
        controller = _load_state_planner(state_planner_spec) if state_planner_spec else None
        return cls(
            controller=controller,
            fallback_policy=fallback,
            config=GV3LiveAdapterConfig(
                adversary_tick_s=float(os.environ.get("VIDUR_VLLM_GV3_ADV_TICK_S", "0.2")),
                launch_window_s=float(os.environ.get("VIDUR_VLLM_GV3_LAUNCH_WINDOW_S", "1.0")),
                decode_credit_mint=int(os.environ.get("VIDUR_VLLM_GV3_DECODE_CREDIT_MINT", "216")),
                max_decode_tokens_per_request=int(
                    os.environ.get("VIDUR_VLLM_GV3_MAX_DECODE_TOKENS", "864")
                ),
                auto_drop_lateness_s=float(
                    os.environ.get("VIDUR_VLLM_GV3_AUTO_DROP_LATENESS_S", "2.0")
                ),
                allow_midstream_initialization=os.environ.get(
                    "VIDUR_VLLM_GV3_ALLOW_MIDSTREAM_INIT", "0"
                ).strip().lower() in {"1", "true", "yes"},
            ),
        )

    @property
    def sim_time(self) -> float:
        return float(self._sim_time)

    @property
    def decode_credit_balance(self) -> int:
        return int(self._decode_credit_balance)

    def _round_time(self, value: float) -> float:
        return round(float(value), int(self.config.time_round_digits))

    def _canonical_prefill_processed(self, row: LiveRequestSnapshot) -> int:
        actual_processed = max(0, row.actual_prefill_tokens - row.actual_prefill_remaining)
        if actual_processed >= row.actual_prefill_tokens:
            return int(row.canonical_prefill_tokens)
        return min(int(actual_processed), int(row.canonical_prefill_tokens))

    def _add_request(self, row: LiveRequestSnapshot) -> _RequestLedger:
        if row.request_id in self._requests:
            raise CanonicalizationError(f"request {row.request_id}: duplicate ledger insertion")
        processed_prefill = self._canonical_prefill_processed(row)
        actual_processed_prefill = max(
            0, row.actual_prefill_tokens - row.actual_prefill_remaining
        )
        processed_decode = max(0, row.actual_decode_tokens - row.actual_decode_remaining)
        if (
            not self.config.allow_midstream_initialization
            and (processed_prefill > 0 or processed_decode > 0)
        ):
            raise CanonicalizationError(
                f"request {row.request_id}: adapter must start before request execution; "
                f"observed prefill={processed_prefill}, decode={processed_decode}"
            )
        request = _RequestLedger(
            request_id=row.request_id,
            gv3_request_id=self._next_gv3_request_id,
            arrived_at=float(row.arrival_time_s),
            canonical_prefill_tokens=int(row.canonical_prefill_tokens),
            canonical_decode_tokens=int(row.canonical_decode_tokens),
            actual_prefill_tokens=int(row.actual_prefill_tokens),
            actual_decode_tokens=int(row.actual_decode_tokens),
            prefill_slo=float(row.canonical_prefill_slo_s),
            decode_slo=float(row.canonical_decode_slo_s),
            processed_prefill=int(processed_prefill),
            actual_processed_prefill=int(actual_processed_prefill),
            processed_decode=int(processed_decode),
        )
        self._next_gv3_request_id += 1
        self._requests[row.request_id] = request
        self._requests_generated += 1
        if request.prefill_complete:
            request.prefill_completed_at = float(self._sim_time)
            request.prefill_lateness_finalized = True
            request.decode_next_deadline = self._sim_time + request.decode_slo
            self._decode_credit_balance += self.config.decode_credit_mint
            request.decode_tokens_counted = int(processed_decode)
            self._decode_credit_balance -= int(processed_decode)
        return request

    def _record_new_launches(self, new_rows: list[LiveRequestSnapshot]) -> None:
        grouped: dict[float, tuple[int, int]] = {}
        for row in new_rows:
            count, prefill = grouped.get(float(row.arrival_time_s), (0, 0))
            grouped[float(row.arrival_time_s)] = (
                count + 1,
                prefill + int(row.canonical_prefill_tokens),
            )
        self._recent_launches.extend(
            (self._round_time(timestamp), int(count), int(prefill))
            for timestamp, (count, prefill) in sorted(grouped.items())
        )
        self._prune_launches()

    def _prune_launches(self) -> None:
        lower = self._sim_time - self.config.launch_window_s
        self._recent_launches = [
            row for row in self._recent_launches if row[0] + _EPS >= lower
        ]

    def _active_requests(self) -> list[_RequestLedger]:
        return [request for request in self._requests.values() if request.active]

    def _pending_prefills(self) -> list[_RequestLedger]:
        return [request for request in self._active_requests() if request.remaining_prefill > 0]

    def _active_decodes(self) -> list[_RequestLedger]:
        return [
            request
            for request in self._active_requests()
            if request.prefill_complete and request.remaining_decode > 0
        ]

    def _advance_idle_for_first_arrival(self, rows: tuple[LiveRequestSnapshot, ...]) -> None:
        if self._active_requests() or not rows:
            return
        earliest = min(float(row.arrival_time_s) for row in rows)
        if earliest > self._sim_time + _EPS:
            self._sim_time = self._round_time(earliest)

    def reconcile(self, snapshot: LiveStateSnapshot) -> None:
        self._advance_idle_for_first_arrival(snapshot.requests)
        live = snapshot.by_id()
        new_rows: list[LiveRequestSnapshot] = []
        for row in snapshot.requests:
            request = self._requests.get(row.request_id)
            if request is None:
                if row.arrival_time_s > self._sim_time + _EPS:
                    # Real requests may become visible between GPU batches.
                    # Keep them waiting until the persistent canonical clock
                    # reaches their trace arrival instead of scheduling early.
                    continue
                request = self._add_request(row)
                new_rows.append(row)
                continue
            if row.request_id in self._pending_termination_ids:
                # The scheduler applies these stop decisions immediately after
                # plan() returns. Do not compare a terminally truncated GV3
                # ledger against the still-live vLLM request first.
                continue
            expected_prefill = request.remaining_prefill
            expected_decode = request.remaining_decode
            if row.canonical_prefill_remaining != expected_prefill:
                raise CanonicalizationError(
                    f"request {row.request_id}: canonical prefill drift; "
                    f"ledger={expected_prefill}, vLLM={row.canonical_prefill_remaining}"
                )
            if row.canonical_decode_remaining != expected_decode:
                raise CanonicalizationError(
                    f"request {row.request_id}: canonical decode drift; "
                    f"ledger={expected_decode}, vLLM={row.canonical_decode_remaining}"
                )
        for request in self._active_requests():
            if request.request_id not in live and request.request_id not in self._pending_termination_ids:
                raise CanonicalizationError(
                    f"request {request.request_id}: disappeared from vLLM without a completed batch"
                )
        if new_rows:
            self._record_new_launches(new_rows)
        self._initialized = True
        self._consume_reached_adversary_ticks()

    def _consume_reached_adversary_ticks(self) -> None:
        while self._next_adv_tick <= self._sim_time + _EPS:
            self._last_adv_tick = self._round_time(self._next_adv_tick)
            self._next_adv_tick = self._round_time(
                self._next_adv_tick + self.config.adversary_tick_s
            )

    def _update_prefill_lateness(self, request: _RequestLedger) -> None:
        if request.prefill_lateness_finalized:
            return
        actual = (
            request.prefill_completed_at
            if request.prefill_complete and request.prefill_completed_at is not None
            else self._sim_time
        )
        lateness = max(0.0, float(actual) - (request.arrived_at + request.prefill_slo))
        if lateness > request.prefill_lateness:
            self._slo_lateness_sum += lateness - request.prefill_lateness
            request.prefill_lateness = lateness
        if request.prefill_complete:
            request.prefill_lateness_finalized = True

    def _drop_request(self, request: _RequestLedger) -> None:
        if request.completed:
            return
        request.dropped = True
        self._complete_request(request)
        request.decode_next_deadline = None
        self._pending_termination_ids.add(request.request_id)
        self._decode_credit_balance += max(
            0, self.config.decode_credit_mint - request.decode_tokens_counted
        )

    def _record_violation_or_drop(self, request: _RequestLedger) -> None:
        total = request.prefill_lateness + request.decode_lateness
        if total >= self.config.auto_drop_lateness_s and not request.completed:
            self._drop_request(request)
            return
        if total > 0.0 and not request.violated:
            request.violated = True
            self._slo_violations += 1

    def _complete_request(self, request: _RequestLedger) -> None:
        if request.completed:
            return
        request.completed = True
        request.completed_at = self._sim_time
        self._requests_completed += 1

    def _finalize_decode_overflow(self) -> None:
        active = self._active_decodes()
        overflow = len(active) - max(0, self._decode_credit_balance)
        if overflow <= 0:
            return
        active.sort(
            key=lambda request: (request.processed_decode, request.gv3_request_id),
            reverse=True,
        )
        for request in active[:overflow]:
            request.canonical_decode_tokens = request.processed_decode
            request.actual_decode_tokens = min(
                request.actual_decode_tokens, request.processed_decode
            )
            request.stopped_decode = True
            self._complete_request(request)
            request.decode_next_deadline = None
            self._pending_termination_ids.add(request.request_id)

    def _apply_progress(self, progress: BatchRequestProgress) -> None:
        request = self._requests.get(progress.request_id)
        if request is None:
            raise CanonicalizationError(
                f"completed batch references unknown request {progress.request_id}"
            )
        actual_prefill_after = min(
            request.actual_prefill_tokens,
            max(0, int(progress.num_computed_tokens_after)),
        )
        canonical_prefill_after = (
            request.canonical_prefill_tokens
            if actual_prefill_after >= request.actual_prefill_tokens
            else actual_prefill_after
        )
        decode_after = min(
            request.canonical_decode_tokens,
            max(0, int(progress.num_output_tokens_after)),
        )
        if canonical_prefill_after < request.processed_prefill:
            raise CanonicalizationError(f"request {request.request_id}: prefill progress regressed")
        if decode_after < request.processed_decode:
            raise CanonicalizationError(f"request {request.request_id}: decode progress regressed")

        prefill_was_complete = request.prefill_complete
        request.processed_prefill = int(canonical_prefill_after)
        request.actual_processed_prefill = int(actual_prefill_after)
        if request.prefill_complete and not prefill_was_complete:
            request.prefill_completed_at = self._sim_time
            self._decode_credit_balance += self.config.decode_credit_mint
            request.decode_next_deadline = self._sim_time + request.decode_slo

        decode_delta = int(decode_after - request.processed_decode)
        if decode_delta > 0:
            if not request.prefill_complete:
                raise CanonicalizationError(
                    f"request {request.request_id}: decode advanced before prefill completion"
                )
            allowed = min(decode_delta, max(0, self._decode_credit_balance))
            if allowed != decode_delta:
                raise CanonicalizationError(
                    f"request {request.request_id}: real decode exceeded GV3 credit balance"
                )
            deadline = request.decode_next_deadline
            if deadline is None:
                base = request.prefill_completed_at or self._sim_time
                deadline = base + request.decode_slo
            token_lateness = max(0.0, self._sim_time - deadline)
            request.decode_lateness += token_lateness * decode_delta
            self._slo_lateness_sum += token_lateness * decode_delta
            request.processed_decode = int(decode_after)
            request.decode_tokens_counted += decode_delta
            self._decode_credit_balance -= decode_delta
            request.decode_next_deadline = self._sim_time + request.decode_slo

        self._update_prefill_lateness(request)
        self._record_violation_or_drop(request)
        if request.remaining_prefill <= 0 and request.remaining_decode <= 0:
            self._complete_request(request)
            request.decode_next_deadline = None
        elif progress.finished_after and not request.completed:
            request.stopped_decode = True
            request.canonical_decode_tokens = request.processed_decode
            self._complete_request(request)
            request.decode_next_deadline = None

    def on_batch_completed(self, observation: BatchExecutionObservation) -> None:
        duration = float(observation.duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise CanonicalizationError(f"invalid completed GPU batch duration {duration!r}")
        expected = set(str(key) for key in observation.scheduled_tokens_by_request)
        actual = {row.request_id for row in observation.request_progress}
        if expected != actual:
            raise CanonicalizationError(
                f"batch progress IDs differ from scheduled IDs: scheduled={sorted(expected)}, "
                f"progress={sorted(actual)}"
            )
        self._sim_time = self._round_time(self._sim_time + duration)
        for progress in observation.request_progress:
            self._apply_progress(progress)
        for request in self._active_requests():
            self._update_prefill_lateness(request)
            self._record_violation_or_drop(request)
        self._finalize_decode_overflow()
        self._prune_launches()
        self._completed_batch_count += 1

    def _termination_details(self) -> list[str]:
        ids = sorted(self._pending_termination_ids)
        self._pending_termination_ids.clear()
        return ids

    def _decode_fast_forward_plan(self, snapshot: LiveStateSnapshot) -> SchedulePlan:
        self._finalize_decode_overflow()
        live = snapshot.by_id()
        active = sorted(self._active_decodes(), key=lambda request: request.gv3_request_id)
        allocations = tuple(
            CanonicalAllocation(request.request_id, RequestPhase.DECODE, 1)
            for request in active[: max(0, self._decode_credit_balance)]
            if request.request_id in live
        )
        return SchedulePlan(
            state_fingerprint=snapshot.fingerprint,
            allocations=allocations,
            policy="gv3-real-decode-fast-forward",
            details={
                "sim_time": self._sim_time,
                "next_adversary_tick": self._next_adv_tick,
                "terminated_request_ids": self._termination_details(),
            },
        )

    def _coerce_controller_result(
        self,
        raw: object,
        snapshot: LiveStateSnapshot,
    ) -> SchedulePlan:
        if isinstance(raw, SchedulePlan):
            return raw
        prefill = getattr(raw, "prefill_allocations", None)
        decode = getattr(raw, "decode_allocations", None)
        if isinstance(prefill, Mapping) and isinstance(decode, Mapping):
            reverse = {request.gv3_request_id: request.request_id for request in self._requests.values()}
            allocations = [
                CanonicalAllocation(reverse[int(request_id)], RequestPhase.DECODE, int(tokens))
                for request_id, tokens in decode.items()
            ]
            allocations.extend(
                CanonicalAllocation(reverse[int(request_id)], RequestPhase.PREFILL, int(tokens))
                for request_id, tokens in prefill.items()
            )
            evicted = [
                reverse[int(request_id)]
                for request_id in list(getattr(raw, "evicted_request_ids", ()) or ())
            ]
            return SchedulePlan(
                state_fingerprint=snapshot.fingerprint,
                allocations=tuple(allocations),
                policy="gv3-controller",
                details={"terminated_request_ids": evicted, "action_repr": repr(raw)},
            )
        if isinstance(raw, Mapping):
            allocations = tuple(
                CanonicalAllocation(
                    str(item["request_id"]),
                    RequestPhase(str(item["phase"])),
                    int(item["num_tokens"]),
                )
                for item in list(raw.get("allocations", ()))
            )
            return SchedulePlan(
                state_fingerprint=str(raw.get("state_fingerprint", snapshot.fingerprint)),
                allocations=allocations,
                policy=str(raw.get("policy", "gv3-controller")),
                details=dict(raw.get("details", {}) or {}),
            )
        raise TypeError("state planner returned an unsupported controller result")

    def _controller_plan(self, snapshot: LiveStateSnapshot) -> SchedulePlan:
        admitted_ids = {
            request.request_id
            for request in self._active_requests()
            if request.arrived_at <= self._sim_time + _EPS
        }
        admitted_snapshot = LiveStateSnapshot.build(
            (
                row
                for row in snapshot.requests
                if row.request_id in admitted_ids
            ),
            max_num_scheduled_tokens=snapshot.max_num_scheduled_tokens,
            captured_monotonic_s=snapshot.captured_monotonic_s,
        )
        if self._controller is not None:
            raw = self._controller(self.state_payload(), admitted_snapshot)
            plan = self._coerce_controller_result(raw, admitted_snapshot)
        elif self._fallback_policy in {"sjf-256", "sjf256"}:
            plan = SJFPlanner(256).plan(admitted_snapshot)
        elif self._fallback_policy in {"sjf-512", "sjf512"}:
            plan = SJFPlanner(512).plan(admitted_snapshot)
        else:
            raise RuntimeError(
                "GV3PersistentAdapter requires VIDUR_VLLM_GV3_STATE_PLANNER or an "
                "explicit test fallback policy"
            )
        details = dict(plan.details or {})
        controller_terminated = {
            str(request_id)
            for request_id in details.get("terminated_request_ids", ())
        }
        allocated_ids = {
            allocation.request_id for allocation in plan.allocations
        }
        overlap = sorted(controller_terminated & allocated_ids)
        if overlap:
            raise CanonicalizationError(
                f"controller both allocated and evicted requests: {overlap}"
            )
        for request_id in sorted(controller_terminated):
            request = self._requests.get(request_id)
            if request is None or not request.active:
                raise CanonicalizationError(
                    f"controller evicted non-active request {request_id}"
                )
            self._drop_request(request)

        details.setdefault("sim_time", self._sim_time)
        details.setdefault("next_adversary_tick", self._next_adv_tick)
        pending = self._termination_details()
        if pending:
            details["terminated_request_ids"] = sorted(
                set(details.get("terminated_request_ids", ())) | set(pending)
            )
        return SchedulePlan(
            state_fingerprint=snapshot.fingerprint,
            allocations=plan.allocations,
            policy=plan.policy,
            details=details,
        )

    def plan(self, snapshot: LiveStateSnapshot) -> SchedulePlan:
        self.reconcile(snapshot)
        if not self._pending_prefills() and self._active_decodes():
            return self._decode_fast_forward_plan(snapshot)
        if not self._active_requests():
            return SchedulePlan(
                state_fingerprint=snapshot.fingerprint,
                allocations=(),
                policy="gv3-real-idle",
                details={
                    "sim_time": self._sim_time,
                    "next_adversary_tick": self._next_adv_tick,
                    "terminated_request_ids": self._termination_details(),
                },
            )
        return self._controller_plan(snapshot)

    def state_payload(self) -> dict[str, Any]:
        active = sorted(
            request.gv3_request_id for request in self._requests.values() if request.active
        )
        completed = sorted(
            request.gv3_request_id for request in self._requests.values() if request.completed
        )
        dropped = sorted(
            request.gv3_request_id for request in self._requests.values() if request.dropped
        )
        stopped = sorted(
            request.gv3_request_id
            for request in self._requests.values()
            if request.stopped_decode
        )
        violated = sorted(
            request.gv3_request_id for request in self._requests.values() if request.violated
        )
        prefill_lateness = {
            request.gv3_request_id: request.prefill_lateness
            for request in self._requests.values()
            if request.prefill_lateness > 0.0
        }
        decode_lateness = {
            request.gv3_request_id: request.decode_lateness
            for request in self._requests.values()
            if request.decode_lateness > 0.0
        }
        decode_deadlines = {
            request.gv3_request_id: request.decode_next_deadline
            for request in self._requests.values()
            if request.decode_next_deadline is not None and request.active
        }
        decode_counted = {
            request.gv3_request_id: request.decode_tokens_counted
            for request in self._requests.values()
            if request.prefill_complete and not request.dropped
        }
        finalized = sorted(
            request.gv3_request_id
            for request in self._requests.values()
            if request.prefill_lateness_finalized
        )
        return {
            "sim_time": float(self._sim_time),
            "decision_state_time": float(self._sim_time),
            "next_request_id": int(self._next_gv3_request_id),
            "requests": [
                request.native_payload()
                for request in sorted(
                    self._requests.values(), key=lambda item: item.gv3_request_id
                )
            ],
            "stats": {
                "requests_generated": int(self._requests_generated),
                "requests_completed": int(self._requests_completed),
                "slo_violations": int(self._slo_violations),
                "slo_lateness_sum": float(self._slo_lateness_sum),
                "recent_arrivals": [list(row) for row in self._recent_launches],
                "recent_launches": [list(row) for row in self._recent_launches],
                "active_request_ids": active,
                "completed_request_ids": completed,
                "dropped_request_ids": dropped,
                "stopped_decode_request_ids": stopped,
                "violated_request_ids": violated,
                "per_request_prefill_lateness_by_id": prefill_lateness,
                "per_request_decode_lateness_by_id": decode_lateness,
                "decode_next_deadline_by_id": decode_deadlines,
                "decode_tokens_counted_by_id": decode_counted,
                "decode_credit_balance": int(self._decode_credit_balance),
                "decode_credit_available": max(0, int(self._decode_credit_balance)),
                "prefill_lateness_finalized_ids": finalized,
                "pending_adv_tick": False,
                "last_adv_tick": float(self._last_adv_tick),
                "next_adv_tick": float(self._next_adv_tick),
                "missed_adv_source": 0,
                "transition_discount_time": float(self._sim_time),
                "transition_final_time": float(self._sim_time),
            },
        }

