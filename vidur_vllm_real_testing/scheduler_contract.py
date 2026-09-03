"""Pure scheduling contracts shared by the vLLM hook and CPU tests."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .action_projection import ProjectedAllocation, project_token_allocations
from .canonicalization import CanonicalizationError, TRACE_SCHEMA_VERSION


class SchedulerMode(str, Enum):
    STOCK = "stock"
    SHADOW = "shadow"
    ACTIVE_VALIDATION = "active-validation"
    CONTROLLER = "controller"
    SJF_256 = "sjf-256"
    SJF_512 = "sjf-512"

    @classmethod
    def parse(cls, raw: str) -> "SchedulerMode":
        normalized = str(raw).strip().lower().replace("_", "-")
        aliases = {
            "baseline": cls.STOCK,
            "controller-sync": cls.CONTROLLER,
            "controller-sync-validation": cls.ACTIVE_VALIDATION,
            "sjf256": cls.SJF_256,
            "sjf512": cls.SJF_512,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"unsupported scheduler mode {raw!r}; expected one of {choices}") from exc


class RequestPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class TraceRequestMetadata:
    request_id: str
    arrived_at_s: float
    actual_prefill_tokens: int
    canonical_prefill_tokens: int
    actual_decode_tokens: int
    canonical_decode_tokens: int
    actual_prefill_slo_s: float
    canonical_prefill_slo_s: float
    actual_decode_slo_s: float
    canonical_decode_slo_s: float


class TraceMetadataRegistry:
    """Request metadata keyed by the exact request IDs submitted to vLLM."""

    def __init__(self, rows: Mapping[str, TraceRequestMetadata]) -> None:
        self._rows = dict(rows)
        grouped: dict[float, list[str]] = {}
        for request_id, row in self._rows.items():
            grouped.setdefault(float(row.arrived_at_s), []).append(request_id)
        self._arrival_groups = {
            arrived_at_s: tuple(sorted(request_ids))
            for arrived_at_s, request_ids in grouped.items()
        }

    @classmethod
    def load(cls, path: str | Path) -> "TraceMetadataRegistry":
        trace_path = Path(path).expanduser().resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        rows: dict[str, TraceRequestMetadata] = {}
        with trace_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "schema_version",
                "request_id",
                "arrived_at_s",
                "actual_prefill_tokens",
                "canonical_prefill_tokens",
                "actual_decode_tokens",
                "canonical_decode_tokens",
                "actual_prefill_slo_s",
                "canonical_prefill_slo_s",
                "actual_decode_slo_s",
                "canonical_decode_slo_s",
            }
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise CanonicalizationError(
                    f"canonical scheduler trace is missing columns: {missing}"
                )
            for source_row, raw in enumerate(reader, start=2):
                if str(raw["schema_version"]).strip() != TRACE_SCHEMA_VERSION:
                    raise CanonicalizationError(
                        f"row {source_row}: unsupported schema_version "
                        f"{raw['schema_version']!r}"
                    )
                request_id = str(raw["request_id"]).strip()
                if not request_id or request_id in rows:
                    raise CanonicalizationError(
                        f"row {source_row}: invalid or duplicate request_id {request_id!r}"
                    )
                rows[request_id] = TraceRequestMetadata(
                    request_id=request_id,
                    arrived_at_s=float(raw["arrived_at_s"]),
                    actual_prefill_tokens=int(raw["actual_prefill_tokens"]),
                    canonical_prefill_tokens=int(raw["canonical_prefill_tokens"]),
                    actual_decode_tokens=int(raw["actual_decode_tokens"]),
                    canonical_decode_tokens=int(raw["canonical_decode_tokens"]),
                    actual_prefill_slo_s=float(raw["actual_prefill_slo_s"]),
                    canonical_prefill_slo_s=float(raw["canonical_prefill_slo_s"]),
                    actual_decode_slo_s=float(raw["actual_decode_slo_s"]),
                    canonical_decode_slo_s=float(raw["canonical_decode_slo_s"]),
                )
        return cls(rows)

    def require(self, request_id: str) -> TraceRequestMetadata:
        raw_request_id = str(request_id)
        try:
            return self._rows[raw_request_id]
        except KeyError as exc:
            # vLLM OpenAI completions wrap an explicit request ID as
            # cmpl-<request-id>-<prompt-index> before scheduling it. Resolve
            # only an exact known trace ID after removing that wrapper.
            if raw_request_id.startswith("cmpl-"):
                wrapped = raw_request_id[5:]
                base, separator, prompt_index = wrapped.rpartition("-")
                if separator and prompt_index.isdigit() and base in self._rows:
                    return self._rows[base]

                # vLLM 0.26 appends an opaque request suffix after the
                # prompt index: cmpl-<trace-id>-<prompt-index>-<suffix>.
                without_suffix, suffix_separator, suffix = wrapped.rpartition("-")
                base, index_separator, prompt_index = without_suffix.rpartition("-")
                if (
                    suffix_separator
                    and suffix
                    and suffix.isalnum()
                    and index_separator
                    and prompt_index.isdigit()
                    and base in self._rows
                ):
                    return self._rows[base]
            raise CanonicalizationError(
                f"live vLLM request {request_id!r} has no canonical trace metadata"
            ) from exc

    def __len__(self) -> int:
        return len(self._rows)

    def request_ids_at(self, arrived_at_s: float) -> tuple[str, ...]:
        return self._arrival_groups.get(float(arrived_at_s), ())


class TraceArrivalGroupBarrier:
    """Release equal-time trace arrivals to the scheduler atomically."""

    def __init__(
        self,
        registry: TraceMetadataRegistry,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        if float(timeout_s) <= 0.0:
            raise ValueError("arrival-group timeout must be positive")
        self._registry = registry
        self._timeout_s = float(timeout_s)
        self._released: set[float] = set()
        self._waiting_since: dict[float, float] = {}

    def pending_missing(
        self,
        visible_request_ids: Iterable[str],
        *,
        now_s: float,
    ) -> dict[float, tuple[str, ...]]:
        visible_by_arrival: dict[float, set[str]] = {}
        for live_request_id in visible_request_ids:
            metadata = self._registry.require(str(live_request_id))
            arrived_at_s = float(metadata.arrived_at_s)
            if arrived_at_s in self._released:
                continue
            visible_by_arrival.setdefault(arrived_at_s, set()).add(
                metadata.request_id
            )

        pending: dict[float, tuple[str, ...]] = {}
        for arrived_at_s, visible_ids in sorted(visible_by_arrival.items()):
            expected_ids = set(self._registry.request_ids_at(arrived_at_s))
            missing_ids = tuple(sorted(expected_ids - visible_ids))
            if not missing_ids:
                self._released.add(arrived_at_s)
                self._waiting_since.pop(arrived_at_s, None)
                continue
            started_s = self._waiting_since.setdefault(arrived_at_s, float(now_s))
            waited_s = max(0.0, float(now_s) - started_s)
            if waited_s > self._timeout_s:
                raise CanonicalizationError(
                    "timed out waiting for atomic trace-arrival group "
                    f"at t={arrived_at_s}: missing={list(missing_ids)}"
                )
            pending[arrived_at_s] = missing_ids
        return pending


def exclusively_matches_request_prefix(
    request_ids: Iterable[str], prefix: str
) -> bool:
    ids = tuple(str(request_id) for request_id in request_ids)
    return bool(prefix and ids and all(request_id.startswith(prefix) for request_id in ids))

@dataclass(frozen=True)
class LiveRequestSnapshot:
    request_id: str
    phase: RequestPhase
    arrival_time_s: float
    actual_prefill_tokens: int
    actual_prefill_remaining: int
    canonical_prefill_tokens: int
    canonical_prefill_remaining: int
    actual_decode_tokens: int
    actual_decode_remaining: int
    canonical_decode_tokens: int
    canonical_decode_remaining: int
    actual_prefill_slo_s: float
    canonical_prefill_slo_s: float
    actual_decode_slo_s: float
    canonical_decode_slo_s: float
    num_computed_tokens: int
    num_output_tokens: int
    queue_name: str

    @property
    def actual_schedulable_tokens(self) -> int:
        if self.phase is RequestPhase.PREFILL:
            return self.actual_prefill_remaining
        return 1 if self.actual_decode_remaining > 0 else 0

    @property
    def canonical_schedulable_tokens(self) -> int:
        if self.phase is RequestPhase.PREFILL:
            return self.canonical_prefill_remaining
        return 1 if self.canonical_decode_remaining > 0 else 0


@dataclass(frozen=True)
class LiveStateSnapshot:
    requests: tuple[LiveRequestSnapshot, ...]
    max_num_scheduled_tokens: int
    captured_monotonic_s: float
    fingerprint: str

    @classmethod
    def build(
        cls,
        requests: Iterable[LiveRequestSnapshot],
        *,
        max_num_scheduled_tokens: int,
        captured_monotonic_s: float,
    ) -> "LiveStateSnapshot":
        ordered = tuple(sorted(requests, key=lambda req: req.request_id))
        payload = {
            "max_num_scheduled_tokens": int(max_num_scheduled_tokens),
            "requests": [asdict(request) for request in ordered],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: value.value if isinstance(value, Enum) else value,
        ).encode("utf-8")
        return cls(
            requests=ordered,
            max_num_scheduled_tokens=int(max_num_scheduled_tokens),
            captured_monotonic_s=float(captured_monotonic_s),
            fingerprint=hashlib.sha256(encoded).hexdigest(),
        )

    def by_id(self) -> dict[str, LiveRequestSnapshot]:
        return {request.request_id: request for request in self.requests}


@dataclass(frozen=True)
class CanonicalAllocation:
    request_id: str
    phase: RequestPhase
    num_tokens: int


@dataclass(frozen=True)
class SchedulePlan:
    state_fingerprint: str
    allocations: tuple[CanonicalAllocation, ...]
    policy: str
    details: Mapping[str, Any] | None = None

    @property
    def canonical_token_budget(self) -> int:
        return sum(allocation.num_tokens for allocation in self.allocations)


@dataclass(frozen=True)
class AppliedAllocation:
    request_id: str
    phase: RequestPhase
    canonical_tokens: int
    actual_tokens: int
    truncated_to_actual_tail: bool


@dataclass(frozen=True)
class ValidatedSchedulePlan:
    source: SchedulePlan
    allocations: tuple[AppliedAllocation, ...]

    @property
    def actual_token_budget(self) -> int:
        return sum(allocation.actual_tokens for allocation in self.allocations)

    @property
    def actual_prefill_budget(self) -> int:
        return sum(
            allocation.actual_tokens
            for allocation in self.allocations
            if allocation.phase is RequestPhase.PREFILL
        )

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(allocation.request_id for allocation in self.allocations)

    def actual_tokens_by_request(self) -> dict[str, int]:
        return {
            allocation.request_id: allocation.actual_tokens
            for allocation in self.allocations
        }


class Planner(Protocol):
    def plan(self, snapshot: LiveStateSnapshot) -> SchedulePlan:
        ...


class SJFPlanner:
    """GV3 SJF: all one-token decodes plus a shortest-prefill budget."""

    def __init__(self, prefill_budget_tokens: int) -> None:
        budget = int(prefill_budget_tokens)
        if budget not in {256, 512}:
            raise ValueError("SJF prefill budget must be 256 or 512")
        self.prefill_budget_tokens = budget

    def plan(self, snapshot: LiveStateSnapshot) -> SchedulePlan:
        allocations: list[CanonicalAllocation] = []
        decodes = sorted(
            (
                request
                for request in snapshot.requests
                if request.phase is RequestPhase.DECODE
                and request.canonical_decode_remaining > 0
            ),
            key=lambda request: (request.arrival_time_s, request.request_id),
        )
        allocations.extend(
            CanonicalAllocation(request.request_id, RequestPhase.DECODE, 1)
            for request in decodes
        )

        remaining_budget = self.prefill_budget_tokens
        prefills = sorted(
            (
                request
                for request in snapshot.requests
                if request.phase is RequestPhase.PREFILL
                and request.canonical_prefill_remaining > 0
            ),
            key=lambda request: (
                request.canonical_prefill_remaining,
                request.arrival_time_s,
                request.request_id,
            ),
        )
        for request in prefills:
            if remaining_budget <= 0:
                break
            allocation = min(request.canonical_prefill_remaining, remaining_budget)
            allocations.append(
                CanonicalAllocation(
                    request.request_id,
                    RequestPhase.PREFILL,
                    int(allocation),
                )
            )
            remaining_budget -= int(allocation)

        return SchedulePlan(
            state_fingerprint=snapshot.fingerprint,
            allocations=tuple(allocations),
            policy=f"sjf-{self.prefill_budget_tokens}",
            details={
                "heuristic": "SJF",
                "prefill_budget_tokens": self.prefill_budget_tokens,
                "decode_count": len(decodes),
            },
        )


def _project_allocations(
    plan: SchedulePlan,
    requests_by_id: Mapping[str, LiveRequestSnapshot],
) -> tuple[AppliedAllocation, ...]:
    canonical = {allocation.request_id: allocation.num_tokens for allocation in plan.allocations}
    actual_remaining = {
        request_id: requests_by_id[request_id].actual_schedulable_tokens
        for request_id in canonical
        if request_id in requests_by_id
    }
    projected: Sequence[ProjectedAllocation] = project_token_allocations(
        canonical,
        actual_remaining,
    )
    projected_by_id = {row.request_id: row for row in projected}
    return tuple(
        AppliedAllocation(
            request_id=allocation.request_id,
            phase=allocation.phase,
            canonical_tokens=allocation.num_tokens,
            actual_tokens=projected_by_id[allocation.request_id].actual_allocation,
            truncated_to_actual_tail=projected_by_id[
                allocation.request_id
            ].truncated_to_actual_tail,
        )
        for allocation in plan.allocations
    )


def validate_and_project_plan(
    plan: SchedulePlan,
    snapshot: LiveStateSnapshot,
) -> ValidatedSchedulePlan:
    """Reject stale, illegal, or non-representable plans before vLLM mutation."""

    if plan.state_fingerprint != snapshot.fingerprint:
        raise CanonicalizationError(
            "stale scheduler plan: live-state fingerprint changed before application"
        )
    requests_by_id = snapshot.by_id()
    seen: set[str] = set()
    for allocation in plan.allocations:
        request_id = str(allocation.request_id)
        if request_id in seen:
            raise CanonicalizationError(f"request {request_id}: duplicate plan allocation")
        seen.add(request_id)
        if request_id not in requests_by_id:
            raise CanonicalizationError(
                f"request {request_id}: scheduler plan references no live request"
            )
        if int(allocation.num_tokens) != allocation.num_tokens or allocation.num_tokens <= 0:
            raise CanonicalizationError(
                f"request {request_id}: allocation must be a positive integer"
            )
        request = requests_by_id[request_id]
        if allocation.phase is not request.phase:
            raise CanonicalizationError(
                f"request {request_id}: plan phase {allocation.phase.value} does not match "
                f"live phase {request.phase.value}"
            )
        if allocation.phase is RequestPhase.DECODE and allocation.num_tokens != 1:
            raise CanonicalizationError(
                f"request {request_id}: vLLM decode allocation must be one token"
            )
        if allocation.num_tokens > request.canonical_schedulable_tokens:
            raise CanonicalizationError(
                f"request {request_id}: allocation exceeds canonical remaining work"
            )

    applied = _project_allocations(plan, requests_by_id)
    total = sum(allocation.actual_tokens for allocation in applied)
    if snapshot.requests and total <= 0:
        raise CanonicalizationError("scheduler plan selected no work for a non-empty state")
    if total > snapshot.max_num_scheduled_tokens:
        raise CanonicalizationError(
            f"projected token budget {total} exceeds vLLM limit "
            f"{snapshot.max_num_scheduled_tokens}"
        )

    return ValidatedSchedulePlan(source=plan, allocations=applied)


def plan_as_dict(plan: SchedulePlan | ValidatedSchedulePlan | None) -> object:
    if plan is None:
        return None
    if isinstance(plan, ValidatedSchedulePlan):
        return {
            "policy": plan.source.policy,
            "state_fingerprint": plan.source.state_fingerprint,
            "allocations": [asdict(allocation) for allocation in plan.allocations],
            "details": dict(plan.source.details or {}),
        }
    return {
        "policy": plan.policy,
        "state_fingerprint": plan.state_fingerprint,
        "allocations": [asdict(allocation) for allocation in plan.allocations],
        "details": dict(plan.details or {}),
    }
