"""Production GV3 adapter with native-equivalent terminal bookkeeping.

This module is the supported planner entrypoint.  It specializes the persistent
live bridge with the exact GV3 drop/credit behavior from
``gv2_virtual_environment.cpp`` while retaining one state ledger across vLLM
scheduler calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

from .canonicalization import CanonicalizationError
from .gv3_live_adapter import (
    BatchRequestProgress,
    GV3LiveAdapterConfig as _BaseConfig,
    GV3PersistentAdapter as _BaseAdapter,
    _RequestLedger,
    _load_state_planner,
)
from .scheduler_contract import (
    CanonicalAllocation,
    LiveStateSnapshot,
    RequestPhase,
    SchedulePlan,
)


@dataclass(frozen=True)
class GV3LiveAdapterConfig(_BaseConfig):
    drop_cost: float = 3.0

    def validate(self) -> None:
        super().validate()
        if self.max_decode_tokens_per_request != 864:
            raise ValueError(
                "production GV3 adapter requires the trained 864-token decode cap"
            )
        if self.drop_cost < 0.0:
            raise ValueError("drop_cost must be nonnegative")


class GV3PersistentAdapter(_BaseAdapter):
    """Persistent real-execution state with exact GV3 credit/drop semantics."""

    config: GV3LiveAdapterConfig

    def __init__(
        self,
        *,
        controller: Any | None = None,
        fallback_policy: str | None = None,
        config: GV3LiveAdapterConfig = GV3LiveAdapterConfig(),
    ) -> None:
        super().__init__(
            controller=controller,
            fallback_policy=fallback_policy,
            config=config,
        )

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
                decode_credit_mint=int(
                    os.environ.get("VIDUR_VLLM_GV3_DECODE_CREDIT_MINT", "216")
                ),
                max_decode_tokens_per_request=int(
                    os.environ.get("VIDUR_VLLM_GV3_MAX_DECODE_TOKENS", "864")
                ),
                auto_drop_lateness_s=float(
                    os.environ.get("VIDUR_VLLM_GV3_AUTO_DROP_LATENESS_S", "2.0")
                ),
                time_round_digits=int(
                    os.environ.get("VIDUR_VLLM_GV3_TIME_ROUND_DIGITS", "10")
                ),
                allow_midstream_initialization=os.environ.get(
                    "VIDUR_VLLM_GV3_ALLOW_MIDSTREAM_INIT", "0"
                ).strip().lower()
                in {"1", "true", "yes"},
                drop_cost=float(os.environ.get("VIDUR_VLLM_GV3_DROP_COST", "3.0")),
            ),
        )

    @staticmethod
    def _has_credit_entry(request: _RequestLedger) -> bool:
        return bool(getattr(request, "decode_credit_entry", False))

    @staticmethod
    def _set_credit_entry(request: _RequestLedger, value: bool) -> None:
        setattr(request, "decode_credit_entry", bool(value))

    def _add_request(self, row: Any) -> _RequestLedger:
        request = super()._add_request(row)
        has_entry = request.prefill_complete and request.remaining_decode > 0
        self._set_credit_entry(request, has_entry)
        if request.prefill_complete and not has_entry:
            self._decode_credit_balance -= self.config.decode_credit_mint
            request.decode_tokens_counted = 0
        return request

    def _drop_request(self, request: _RequestLedger) -> None:
        if request.completed:
            return
        total = request.prefill_lateness + request.decode_lateness

        # GV3 drop_request removes live lateness/violation accounting and
        # replaces it with the fixed terminal drop cost.
        self._slo_lateness_sum = max(0.0, self._slo_lateness_sum - total)
        if request.violated:
            request.violated = False
            self._slo_violations = max(0, self._slo_violations - 1)

        # DecodeCreditLedger::reclaim_on_drop subtracts the unspent minted
        # credit. It does not add credit back to the global balance.
        if self._has_credit_entry(request):
            unspent = max(
                0,
                self.config.decode_credit_mint - request.decode_tokens_counted,
            )
            self._decode_credit_balance -= unspent
            self._set_credit_entry(request, False)
            request.decode_tokens_counted = 0

        request.canonical_prefill_tokens = request.processed_prefill
        request.actual_prefill_tokens = min(
            request.actual_prefill_tokens,
            request.actual_processed_prefill,
        )
        request.canonical_decode_tokens = request.processed_decode
        request.actual_decode_tokens = min(
            request.actual_decode_tokens,
            request.processed_decode,
        )
        request.prefill_lateness = 0.0
        request.decode_lateness = 0.0
        request.prefill_lateness_finalized = False
        request.dropped = True
        self._complete_request(request)
        request.decode_next_deadline = None
        self._pending_termination_ids.add(request.request_id)
        self._slo_lateness_sum += self.config.drop_cost

    def _record_violation_or_drop(self, request: _RequestLedger) -> None:
        total = request.prefill_lateness + request.decode_lateness
        if total >= self.config.auto_drop_lateness_s and not request.completed:
            self._drop_request(request)
            return
        if total > 0.0 and not request.violated:
            request.violated = True
            self._slo_violations += 1

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
                request.actual_decode_tokens,
                request.processed_decode,
            )
            request.stopped_decode = True
            self._complete_request(request)
            request.decode_next_deadline = None
            self._set_credit_entry(request, False)
            request.decode_tokens_counted = 0
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
            request.decode_next_deadline = self._sim_time + request.decode_slo
            if request.remaining_decode > 0:
                self._decode_credit_balance += self.config.decode_credit_mint
                self._set_credit_entry(request, True)

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

    def _coerce_controller_result(
        self,
        raw: object,
        snapshot: LiveStateSnapshot,
    ) -> SchedulePlan:
        if isinstance(raw, Mapping):
            allocations = tuple(
                CanonicalAllocation(
                    str(item["request_id"]),
                    item["phase"]
                    if isinstance(item["phase"], RequestPhase)
                    else RequestPhase(str(item["phase"]).strip().lower()),
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
        return super()._coerce_controller_result(raw, snapshot)

    def state_payload(self) -> dict[str, Any]:
        payload = super().state_payload()
        payload["stats"]["decode_tokens_counted_by_id"] = {
            request.gv3_request_id: request.decode_tokens_counted
            for request in self._requests.values()
            if self._has_credit_entry(request)
        }
        return payload
