"""Deterministic GV4 state and canonical-action feature extraction.

This module is the Python reference serializer for training, inference, and the
future native implementation. It intentionally depends on NumPy, but not on
PyTorch or AlphaGoZero. Variable-size collections remain row matrices; a model
may pad and mask them only when forming a minibatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..action_resolver import (
    CanonicalAdversaryAction,
    CanonicalControllerAction,
    ControllerTransitionKind,
)
from ..config import GV4EngineConfig
from ..state import (
    GV4State,
    InflightMicrobatchState,
    Player,
    RequestLifecycle,
    RequestState,
    UNASSIGNED_REPLICA,
    UNSET_TIME,
)


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]

WORKLOAD_WINDOW_MULTIPLIER: Final[float] = 20.0

GLOBAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "active_request_count",
    "active_prefill_count",
    "active_decode_count",
    "remaining_prefill_tokens",
    "remaining_decode_tokens",
    "committed_prefill_tokens",
    "committed_decode_tokens",
    "committed_context_tokens",
    "decode_credit_balance",
    "decode_credits_reserved",
    "next_adversary_tick_delta",
    "logical_tokens_free_fraction",
    "waiting_prefill_count",
    "inflight_prefill_count",
    "waiting_decode_count",
    "inflight_decode_count",
    "stop_pending_count",
    "drop_pending_count",
    "active_violation_fraction",
    "active_prefill_violation_fraction",
    "active_decode_violation_fraction",
    "waiting_prefill_violation_fraction",
    "inflight_prefill_violation_fraction",
    "waiting_decode_violation_fraction",
    "inflight_decode_violation_fraction",
    "stop_pending_violation_fraction",
    "drop_pending_violation_fraction",
)

LAUNCH_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "launch_age",
    "request_count",
    "prefill_tokens",
)

CONTROLLER_HEADER_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "transition_wait",
    "transition_evict_only",
    "transition_batch",
    "evicted_prefill_count",
    "evicted_decode_count",
    "decode_request_count",
    "total_allocated_prefill_tokens",
)

CONTROLLER_REQUEST_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "allocated_prefill",
    "evicted",
    "allocated_prefill_tokens",
    "remaining_prefill_tokens",
    "total_prefill_tokens",
    "arrival_age",
    "current_lateness",
    "violated",
    "new_reserved_kv_blocks",
    "already_inflight",
)

ADVERSARY_HEADER_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "launched_request_count",
    "prefill_tokens_per_launched_request",
    "total_launched_prefill_tokens",
    "stopped_decode_count",
    "stopped_inflight_decode_count",
)

ADVERSARY_REQUEST_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "decode_total",
    "decode_committed",
    "decode_remaining",
    "current_lateness",
    "decode_deadline_present",
    "decode_deadline_delta",
    "violated",
    "reserved_decode_token",
    "inflight",
)

_NONTERMINAL_LIFECYCLES: Final[tuple[RequestLifecycle, ...]] = (
    RequestLifecycle.WAITING_PREFILL,
    RequestLifecycle.INFLIGHT_PREFILL,
    RequestLifecycle.WAITING_DECODE,
    RequestLifecycle.INFLIGHT_DECODE,
    RequestLifecycle.STOP_PENDING,
    RequestLifecycle.DROP_PENDING,
)


class DNNFeatureError(ValueError):
    """Raised when state cannot be represented by the GV4 feature contract."""


def _request_feature_names(stage_count: int) -> tuple[str, ...]:
    return (
        "decode_phase",
        "prefill_total",
        "prefill_committed",
        "prefill_remaining",
        "decode_total",
        "decode_committed",
        "decode_remaining",
        "committed_context",
        "arrival_age",
        "current_lateness",
        "prefill_deadline_delta",
        "decode_deadline_present",
        "decode_deadline_delta",
        "violated",
        *(f"lifecycle_{item.name.lower()}" for item in _NONTERMINAL_LIFECYCLES),
        "reserved_tokens",
        "partial_block_used_fraction",
        "tokens_until_next_block_fraction",
        "has_inflight_work",
        *(f"active_pipeline_stage_{index}" for index in range(stage_count)),
        "pipeline_wait_or_transfer",
        "inflight_prefill_tokens",
        "inflight_decode_token",
        "pending_stop",
        "pending_drop",
        "pending_termination_age",
    )


def _replica_feature_names(stage_count: int) -> tuple[str, ...]:
    return (
        "committed_logical_kv_blocks",
        "reserved_logical_kv_blocks",
        "free_logical_kv_blocks",
        "min_rank_free_fraction",
        "mean_rank_free_fraction",
        "max_rank_free_fraction",
        "inflight_microbatch_count",
        *(f"stage_{index}_free" for index in range(stage_count)),
    )


def _microbatch_feature_names(stage_count: int) -> tuple[str, ...]:
    return (
        *(f"active_stage_{index}" for index in range(stage_count)),
        "waiting_or_transfer",
        "prefill_request_count",
        "decode_request_count",
        "prefill_tokens",
        "decode_tokens",
        "prefill_reserved_kv_blocks",
        "decode_reserved_kv_blocks",
        "violated_prefill_request_count",
        "violated_decode_request_count",
    )


@dataclass(frozen=True, slots=True)
class GV4FeatureLayout:
    """Ordered names and dimensions for one resolved topology."""

    schema_version: str
    pipeline_stage_count: int
    global_names: tuple[str, ...]
    request_names: tuple[str, ...]
    launch_names: tuple[str, ...]
    replica_names: tuple[str, ...]
    microbatch_names: tuple[str, ...]
    controller_header_names: tuple[str, ...]
    controller_request_names: tuple[str, ...]
    adversary_header_names: tuple[str, ...]
    adversary_request_names: tuple[str, ...]

    @classmethod
    def from_config(cls, config: GV4EngineConfig) -> "GV4FeatureLayout":
        stages = config.topology.pipeline_parallel_size
        return cls(
            schema_version=config.layout.feature_schema_version,
            pipeline_stage_count=stages,
            global_names=GLOBAL_FEATURE_NAMES,
            request_names=_request_feature_names(stages),
            launch_names=LAUNCH_FEATURE_NAMES,
            replica_names=_replica_feature_names(stages),
            microbatch_names=_microbatch_feature_names(stages),
            controller_header_names=CONTROLLER_HEADER_FEATURE_NAMES,
            controller_request_names=CONTROLLER_REQUEST_FEATURE_NAMES,
            adversary_header_names=ADVERSARY_HEADER_FEATURE_NAMES,
            adversary_request_names=ADVERSARY_REQUEST_FEATURE_NAMES,
        )


@dataclass(frozen=True, slots=True)
class GV4FeatureScales:
    """Config-derived normalization constants shared by every feature builder."""

    window_request_cap: float
    window_prefill_cap: float
    active_request_scale: float
    system_prefill_scale: float
    system_decode_scale: float
    request_prefill_scale: float
    request_decode_scale: float
    decode_credit_scale: float
    adversary_time_scale: float
    launch_age_scale: float
    lateness_scale: float
    block_token_scale: float
    controller_prefill_action_scale: float
    controller_kv_block_scale: float
    system_logical_blocks: float
    system_logical_tokens: float

    @classmethod
    def from_config(cls, config: GV4EngineConfig) -> "GV4FeatureScales":
        window_requests = float(config.timing.max_requests_per_launch_window)
        window_prefill = float(
            config.request.target_prefill_tokens_per_request_window_average
            * config.timing.max_requests_per_launch_window
        )
        active_requests = WORKLOAD_WINDOW_MULTIPLIER * window_requests
        controller_prefill = float(
            max(config.controller_actions.prefill_budget_options)
        )
        if controller_prefill <= 0.0:
            raise DNNFeatureError(
                "at least one positive controller prefill budget is required"
            )

        rank_capacities = config.rank_kv_block_capacities()
        logical_blocks = 0
        for placement in config.topology.replica_placements:
            logical_blocks += min(
                rank_capacities[index] for index in placement.rank_ids
            )
        if logical_blocks <= 0:
            raise DNNFeatureError("configured replicas have no logical KV capacity")

        block_tokens = float(config.kv_cache.block_size_tokens)
        return cls(
            window_request_cap=window_requests,
            window_prefill_cap=window_prefill,
            active_request_scale=active_requests,
            system_prefill_scale=WORKLOAD_WINDOW_MULTIPLIER * window_prefill,
            system_decode_scale=(
                active_requests
                * float(config.request.target_decode_tokens_per_request_average)
            ),
            request_prefill_scale=float(config.request.max_prefill_tokens_per_request),
            request_decode_scale=float(config.request.max_decode_tokens_per_request),
            decode_credit_scale=float(
                config.request.target_decode_tokens_per_request_average
            ),
            adversary_time_scale=float(config.timing.adversary_tick_sec),
            launch_age_scale=float(config.timing.launch_window_sec),
            lateness_scale=float(config.cost.lateness_cap_sec),
            block_token_scale=block_tokens,
            controller_prefill_action_scale=controller_prefill,
            controller_kv_block_scale=float(
                math.ceil(controller_prefill / block_tokens)
            ),
            system_logical_blocks=float(logical_blocks),
            system_logical_tokens=float(logical_blocks) * block_tokens,
        )


@dataclass(frozen=True, slots=True)
class GV4StateFeatures:
    """One complete scheduler state with variable rows kept unpadded."""

    schema_version: str
    config_manifest_sha256: str
    global_features: FloatArray
    request_rows: FloatArray
    request_replica_offsets: IntArray
    launch_rows: FloatArray
    replica_rows: FloatArray
    microbatch_rows: FloatArray
    microbatch_replica_offsets: IntArray


@dataclass(frozen=True, slots=True)
class GV4ControllerActionFeatures:
    """Physical effects of one legal canonical controller edge."""

    header: FloatArray
    affected_request_rows: FloatArray


@dataclass(frozen=True, slots=True)
class GV4AdversaryActionFeatures:
    """Physical effects of one legal canonical adversary edge."""

    header: FloatArray
    affected_request_rows: FloatArray


def _float_vector(values: object, width: int, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if result.shape != (width,):
        raise DNNFeatureError(f"{label} has shape {result.shape}, expected ({width},)")
    if not np.isfinite(result).all():
        raise DNNFeatureError(f"{label} contains a non-finite value")
    result.setflags(write=False)
    return result


def _float_matrix(rows: object, width: int, label: str) -> FloatArray:
    result = np.asarray(rows, dtype=np.float32)
    if result.size == 0:
        result = np.empty((0, width), dtype=np.float32)
    else:
        result = result.reshape(-1, width)
    if not np.isfinite(result).all():
        raise DNNFeatureError(f"{label} contains a non-finite value")
    result.setflags(write=False)
    return result


def _offset_array(values: object, final_count: int, label: str) -> IntArray:
    result = np.asarray(values, dtype=np.int32).reshape(-1)
    if result.size == 0 or result[0] != 0 or result[-1] != final_count:
        raise DNNFeatureError(f"{label} does not bound all rows")
    if np.any(result[1:] < result[:-1]):
        raise DNNFeatureError(f"{label} must be nondecreasing")
    result.setflags(write=False)
    return result


def _fraction(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else float(numerator) / float(denominator)


def _active_stage(
    batch: InflightMicrobatchState,
    now: float,
    epsilon: float,
) -> int:
    """Return the currently executing PP stage without exposing its times."""

    for index, (start, finish) in enumerate(
        zip(batch.stage_start_times, batch.stage_finish_times)
    ):
        if now + epsilon >= start and now < finish - epsilon:
            return index
    return -1


def _decode_phase(request: RequestState) -> bool:
    if request.lifecycle in (
        RequestLifecycle.WAITING_DECODE,
        RequestLifecycle.INFLIGHT_DECODE,
    ):
        return True
    if request.lifecycle in (
        RequestLifecycle.STOP_PENDING,
        RequestLifecycle.DROP_PENDING,
    ):
        return request.reserved_decode_tokens > 0
    return False


def _current_lateness(request: RequestState, now: float) -> float:
    if _decode_phase(request):
        return request.prefill_lateness_sec + request.decode_lateness_sec
    return max(request.prefill_lateness_sec, now - request.prefill_deadline, 0.0)


class GV4FeatureBuilder:
    """Build exact GV4 DNN inputs from one immutable config contract."""

    __slots__ = ("config", "layout", "scales", "_manifest_sha256")

    def __init__(self, config: GV4EngineConfig) -> None:
        self.config = config
        self.layout = GV4FeatureLayout.from_config(config)
        self.scales = GV4FeatureScales.from_config(config)
        self._manifest_sha256 = config.manifest_sha256()

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    def schema_metadata(self) -> dict[str, object]:
        """Return metadata that model and replay artifacts must persist."""

        return {
            "schema_version": self.layout.schema_version,
            "config_manifest_sha256": self._manifest_sha256,
            "pipeline_stage_count": self.layout.pipeline_stage_count,
            "feature_names": {
                "global": list(self.layout.global_names),
                "request": list(self.layout.request_names),
                "launch": list(self.layout.launch_names),
                "replica": list(self.layout.replica_names),
                "microbatch": list(self.layout.microbatch_names),
                "controller_header": list(self.layout.controller_header_names),
                "controller_request": list(self.layout.controller_request_names),
                "adversary_header": list(self.layout.adversary_header_names),
                "adversary_request": list(self.layout.adversary_request_names),
            },
            "scales": asdict(self.scales),
        }

    def _check_state(self, state: GV4State) -> None:
        if state.state_schema_version != self.config.layout.state_schema_version:
            raise DNNFeatureError("state schema does not match feature configuration")
        if state.config_manifest_sha256 != self._manifest_sha256:
            raise DNNFeatureError("state belongs to a different engine manifest")
        if not math.isfinite(state.now) or state.now < 0.0:
            raise DNNFeatureError("state time must be finite and nonnegative")
        if state.next_adversary_tick + self.config.timing.epsilon < state.now:
            raise DNNFeatureError("next adversary tick is behind state time")
        if len(state.replicas) != self.config.topology.num_replicas:
            raise DNNFeatureError("state replica count differs from configuration")

    def _live_requests_by_replica(self, state: GV4State) -> list[list[RequestState]]:
        grouped = [[] for _ in state.replicas]
        epsilon = self.config.timing.epsilon
        for request in state.requests:
            if request.lifecycle.is_terminal:
                continue
            if request.arrival_time > state.now + epsilon:
                raise DNNFeatureError(
                    f"request {request.request_id} has a future arrival time"
                )
            if request.owner_replica_id == UNASSIGNED_REPLICA:
                continue
            if not 0 <= request.owner_replica_id < len(grouped):
                raise DNNFeatureError(
                    f"request {request.request_id} has an invalid replica owner"
                )
            grouped[request.owner_replica_id].append(request)
        return grouped

    def _batch_maps(self, state: GV4State) -> tuple[
        dict[int, tuple[InflightMicrobatchState, int]],
        dict[int, tuple[InflightMicrobatchState, int]],
    ]:
        by_id: dict[int, tuple[InflightMicrobatchState, int]] = {}
        by_request: dict[int, tuple[InflightMicrobatchState, int]] = {}
        epsilon = self.config.timing.epsilon
        for replica in state.replicas:
            for batch in replica.inflight_microbatches:
                if batch.microbatch_id in by_id:
                    raise DNNFeatureError("duplicate in-flight microbatch ID")
                view = (batch, _active_stage(batch, state.now, epsilon))
                by_id[batch.microbatch_id] = view
                for allocation in batch.allocations:
                    if allocation.request_id in by_request:
                        raise DNNFeatureError(
                            "request appears in two in-flight batches"
                        )
                    by_request[allocation.request_id] = view
        return by_id, by_request

    def _request_row(
        self,
        request: RequestState,
        state: GV4State,
        batch_view: tuple[InflightMicrobatchState, int] | None,
    ) -> list[float]:
        scales = self.scales
        decode_phase = _decode_phase(request)
        if request.has_inflight_work:
            if (
                batch_view is None
                or batch_view[0].microbatch_id != request.inflight_microbatch_id
            ):
                raise DNNFeatureError(
                    f"request {request.request_id} lacks its in-flight batch"
                )
            active_stage = batch_view[1]
        else:
            if batch_view is not None:
                raise DNNFeatureError(
                    f"request {request.request_id} has an unexpected batch allocation"
                )
            active_stage = -1

        decode_deadline_present = request.next_decode_deadline != UNSET_TIME
        decode_deadline_delta = (
            math.asinh(
                (request.next_decode_deadline - state.now)
                / request.decode_token_slo_sec
            )
            if decode_deadline_present
            else 0.0
        )
        lifecycle_bits = [
            float(request.lifecycle == lifecycle)
            for lifecycle in _NONTERMINAL_LIFECYCLES
        ]
        stage_bits = [
            float(active_stage == index)
            for index in range(self.layout.pipeline_stage_count)
        ]
        pending = request.lifecycle in (
            RequestLifecycle.STOP_PENDING,
            RequestLifecycle.DROP_PENDING,
        )
        if pending:
            if request.terminal_requested_at == UNSET_TIME:
                raise DNNFeatureError("pending request lacks terminal request time")
            if request.terminal_requested_at > state.now + self.config.timing.epsilon:
                raise DNNFeatureError(
                    "pending request has a future terminal request time"
                )
            pending_age = math.asinh(
                (state.now - request.terminal_requested_at) / scales.lateness_scale
            )
        else:
            pending_age = 0.0

        return [
            float(decode_phase),
            request.original_prefill_tokens / scales.request_prefill_scale,
            request.committed_prefill_tokens / scales.request_prefill_scale,
            request.remaining_prefill_tokens / scales.request_prefill_scale,
            request.original_decode_tokens / scales.request_decode_scale,
            request.committed_decode_tokens / scales.request_decode_scale,
            request.remaining_decode_tokens / scales.request_decode_scale,
            (request.committed_prefill_tokens + request.committed_decode_tokens)
            / (scales.request_prefill_scale + scales.request_decode_scale),
            math.asinh((state.now - request.arrival_time) / scales.launch_age_scale),
            math.asinh(_current_lateness(request, state.now) / scales.launch_age_scale),
            math.asinh(
                (request.prefill_deadline - state.now) / scales.launch_age_scale
            ),
            float(decode_deadline_present),
            decode_deadline_delta,
            float(request.violation_recorded),
            *lifecycle_bits,
            (request.reserved_prefill_tokens + request.reserved_decode_tokens)
            / (scales.request_prefill_scale + scales.request_decode_scale),
            request.tokens_used_in_final_kv_block(
                self.config.kv_cache.block_size_tokens
            )
            / scales.block_token_scale,
            request.tokens_until_next_kv_block(self.config.kv_cache.block_size_tokens)
            / scales.block_token_scale,
            float(request.has_inflight_work),
            *stage_bits,
            float(request.has_inflight_work and active_stage < 0),
            request.reserved_prefill_tokens / scales.request_prefill_scale,
            float(request.reserved_decode_tokens),
            float(request.lifecycle == RequestLifecycle.STOP_PENDING),
            float(request.lifecycle == RequestLifecycle.DROP_PENDING),
            pending_age,
        ]

    def _replica_usage(self, replica_id: int, state: GV4State) -> tuple[int, int, int]:
        replica = state.replica(replica_id)
        if len(set(replica.rank_kv_committed_blocks)) != 1:
            raise DNNFeatureError("logical committed KV is not mirrored across ranks")
        if len(set(replica.rank_kv_reserved_blocks)) != 1:
            raise DNNFeatureError("logical reserved KV is not mirrored across ranks")
        committed = replica.rank_kv_committed_blocks[0]
        reserved = replica.rank_kv_reserved_blocks[0]
        free = min(replica.free_kv_blocks_by_rank())
        return committed, reserved, free

    def _global_row(
        self,
        state: GV4State,
        live_requests: list[RequestState],
    ) -> list[float]:
        scales = self.scales
        prefills = [request for request in live_requests if not _decode_phase(request)]
        decodes = [request for request in live_requests if _decode_phase(request)]
        lifecycle_counts = {
            lifecycle: sum(request.lifecycle == lifecycle for request in live_requests)
            for lifecycle in _NONTERMINAL_LIFECYCLES
        }
        violated = [request for request in live_requests if request.violation_recorded]
        live_count = len(live_requests)
        free_logical_blocks = sum(
            self._replica_usage(replica.replica_id, state)[2]
            for replica in state.replicas
        )

        def violation_fraction(lifecycle: RequestLifecycle) -> float:
            return _fraction(
                sum(
                    request.violation_recorded and request.lifecycle == lifecycle
                    for request in live_requests
                ),
                live_count,
            )

        return [
            live_count / scales.active_request_scale,
            len(prefills) / scales.active_request_scale,
            len(decodes) / scales.active_request_scale,
            sum(item.remaining_prefill_tokens for item in live_requests)
            / scales.system_prefill_scale,
            sum(item.remaining_decode_tokens for item in live_requests)
            / scales.system_decode_scale,
            sum(item.committed_prefill_tokens for item in live_requests)
            / scales.system_prefill_scale,
            sum(item.committed_decode_tokens for item in live_requests)
            / scales.system_decode_scale,
            sum(
                item.committed_prefill_tokens + item.committed_decode_tokens
                for item in live_requests
            )
            / scales.system_logical_tokens,
            math.asinh(state.decode_credits_available / scales.decode_credit_scale),
            math.asinh(state.decode_credits_reserved / scales.decode_credit_scale),
            math.asinh(
                (state.next_adversary_tick - state.now) / scales.adversary_time_scale
            ),
            (free_logical_blocks * scales.block_token_scale)
            / scales.system_logical_tokens,
            *(
                lifecycle_counts[lifecycle] / scales.active_request_scale
                for lifecycle in _NONTERMINAL_LIFECYCLES
            ),
            _fraction(len(violated), live_count),
            _fraction(sum(item.violation_recorded for item in prefills), live_count),
            _fraction(sum(item.violation_recorded for item in decodes), live_count),
            *(violation_fraction(lifecycle) for lifecycle in _NONTERMINAL_LIFECYCLES),
        ]

    def _launch_rows(self, state: GV4State) -> list[list[float]]:
        rows: list[list[float]] = []
        cutoff = state.now - self.config.timing.launch_window_sec
        epsilon = self.config.timing.epsilon
        prior_time = -1.0
        for record in state.launch_history:
            if record.launch_time < prior_time:
                raise DNNFeatureError("launch history is not chronological")
            prior_time = record.launch_time
            if record.launch_time > state.now + epsilon:
                raise DNNFeatureError("launch history contains a future-dated record")
            if record.launch_time <= cutoff + epsilon:
                continue
            rows.append(
                [
                    math.asinh(
                        (state.now - record.launch_time) / self.scales.launch_age_scale
                    ),
                    record.request_count / self.scales.window_request_cap,
                    record.prefill_tokens / self.scales.window_prefill_cap,
                ]
            )
        return rows

    def _replica_row(
        self,
        replica_id: int,
        state: GV4State,
        active_stages: tuple[int, ...],
    ) -> list[float]:
        replica = state.replica(replica_id)
        committed, reserved, free = self._replica_usage(replica_id, state)
        free_fractions = [
            available / capacity
            for available, capacity in zip(
                replica.free_kv_blocks_by_rank(), replica.rank_kv_capacity_blocks
            )
        ]
        occupied = [False] * self.layout.pipeline_stage_count
        for stage in active_stages:
            if stage >= 0:
                occupied[stage] = True
        return [
            committed / self.scales.system_logical_blocks,
            reserved / self.scales.system_logical_blocks,
            free / self.scales.system_logical_blocks,
            min(free_fractions),
            sum(free_fractions) / len(free_fractions),
            max(free_fractions),
            replica.inflight_count / self.config.scheduler.max_inflight_microbatches,
            *(float(not value) for value in occupied),
        ]

    def _microbatch_row(
        self,
        batch: InflightMicrobatchState,
        active_stage: int,
        state: GV4State,
    ) -> list[float]:
        allocations = batch.allocations
        prefill = [item for item in allocations if item.prefill_tokens]
        decode = [item for item in allocations if item.decode_tokens]
        stage_bits = [
            float(active_stage == index)
            for index in range(self.layout.pipeline_stage_count)
        ]
        return [
            *stage_bits,
            float(active_stage < 0),
            len(prefill) / self.scales.active_request_scale,
            len(decode) / self.scales.active_request_scale,
            sum(item.prefill_tokens for item in prefill)
            / self.scales.system_prefill_scale,
            sum(item.decode_tokens for item in decode)
            / self.scales.system_decode_scale,
            sum(item.new_kv_blocks for item in prefill)
            / self.scales.system_logical_blocks,
            sum(item.new_kv_blocks for item in decode)
            / self.scales.system_logical_blocks,
            sum(state.request(item.request_id).violation_recorded for item in prefill)
            / self.scales.active_request_scale,
            sum(state.request(item.request_id).violation_recorded for item in decode)
            / self.scales.active_request_scale,
        ]

    def build_state(self, state: GV4State) -> GV4StateFeatures:
        """Serialize one state without padding, truncation, or future-time leakage."""

        self._check_state(state)
        grouped_requests = self._live_requests_by_replica(state)
        batch_by_id, batch_by_request = self._batch_maps(state)
        live_requests = [item for group in grouped_requests for item in group]

        request_rows: list[list[float]] = []
        request_offsets = [0]
        for requests in grouped_requests:
            for request in requests:
                request_rows.append(
                    self._request_row(
                        request,
                        state,
                        batch_by_request.get(request.request_id),
                    )
                )
            request_offsets.append(len(request_rows))

        replica_rows: list[list[float]] = []
        microbatch_rows: list[list[float]] = []
        microbatch_offsets = [0]
        for replica in state.replicas:
            active_stages = tuple(
                batch_by_id[batch.microbatch_id][1]
                for batch in replica.inflight_microbatches
            )
            replica_rows.append(
                self._replica_row(replica.replica_id, state, active_stages)
            )
            for batch in replica.inflight_microbatches:
                microbatch_rows.append(
                    self._microbatch_row(
                        batch,
                        batch_by_id[batch.microbatch_id][1],
                        state,
                    )
                )
            microbatch_offsets.append(len(microbatch_rows))

        layout = self.layout
        return GV4StateFeatures(
            schema_version=layout.schema_version,
            config_manifest_sha256=self._manifest_sha256,
            global_features=_float_vector(
                self._global_row(state, live_requests),
                len(layout.global_names),
                "global features",
            ),
            request_rows=_float_matrix(
                request_rows, len(layout.request_names), "request rows"
            ),
            request_replica_offsets=_offset_array(
                request_offsets, len(request_rows), "request replica offsets"
            ),
            launch_rows=_float_matrix(
                self._launch_rows(state), len(layout.launch_names), "launch rows"
            ),
            replica_rows=_float_matrix(
                replica_rows, len(layout.replica_names), "replica rows"
            ),
            microbatch_rows=_float_matrix(
                microbatch_rows,
                len(layout.microbatch_names),
                "microbatch rows",
            ),
            microbatch_replica_offsets=_offset_array(
                microbatch_offsets,
                len(microbatch_rows),
                "microbatch replica offsets",
            ),
        )

    def _action_request(self, state: GV4State, request_id: int) -> RequestState:
        try:
            request = state.request(request_id)
        except (IndexError, ValueError) as error:
            raise DNNFeatureError(f"action references request {request_id}") from error
        if request.lifecycle.is_terminal:
            raise DNNFeatureError("canonical action references a terminal request")
        return request

    def build_controller_action(
        self,
        state: GV4State,
        edge: CanonicalControllerAction,
    ) -> GV4ControllerActionFeatures:
        """Serialize one canonical controller edge from its parent state."""

        self._check_state(state)
        if state.next_player != Player.CONTROLLER:
            raise DNNFeatureError("controller features require a controller state")
        if not isinstance(edge, CanonicalControllerAction):
            raise TypeError("edge must be CanonicalControllerAction")
        action = edge.action

        evicted = [
            self._action_request(state, request_id)
            for request_id in action.evicted_request_ids
        ]
        if any(request.owner_replica_id != action.replica_id for request in evicted):
            raise DNNFeatureError("controller eviction targets another replica")
        decode_allocations = [item for item in action.allocations if item.decode_tokens]
        transition_bits = [
            float(action.transition_kind == kind) for kind in ControllerTransitionKind
        ]
        header = [
            *transition_bits,
            sum(not _decode_phase(request) for request in evicted)
            / self.scales.active_request_scale,
            sum(_decode_phase(request) for request in evicted)
            / self.scales.active_request_scale,
            len(decode_allocations) / self.scales.active_request_scale,
            action.total_prefill_tokens / self.scales.controller_prefill_action_scale,
        ]

        affected: dict[int, tuple[RequestState, int, int, bool]] = {}
        for allocation in action.allocations:
            if not allocation.prefill_tokens:
                continue
            request = self._action_request(state, allocation.request_id)
            if request.owner_replica_id != action.replica_id:
                raise DNNFeatureError("controller allocation targets another replica")
            if request.request_id in affected:
                raise DNNFeatureError("duplicate request allocation in one edge")
            affected[request.request_id] = (
                request,
                allocation.prefill_tokens,
                allocation.new_kv_blocks,
                False,
            )
        for request in evicted:
            if request.request_id in affected:
                raise DNNFeatureError("one edge cannot allocate and evict one request")
            affected[request.request_id] = (request, 0, 0, True)

        rows = []
        for request_id in sorted(affected):
            request, allocated_tokens, new_blocks, is_evicted = affected[request_id]
            rows.append(
                [
                    float(allocated_tokens > 0),
                    float(is_evicted),
                    allocated_tokens / self.scales.controller_prefill_action_scale,
                    request.remaining_prefill_tokens
                    / self.scales.request_prefill_scale,
                    request.original_prefill_tokens / self.scales.request_prefill_scale,
                    math.asinh(
                        (state.now - request.arrival_time)
                        / self.scales.launch_age_scale
                    ),
                    math.asinh(
                        _current_lateness(request, state.now)
                        / self.scales.launch_age_scale
                    ),
                    float(request.violation_recorded),
                    new_blocks / self.scales.controller_kv_block_scale,
                    float(request.has_inflight_work),
                ]
            )

        return GV4ControllerActionFeatures(
            header=_float_vector(
                header,
                len(self.layout.controller_header_names),
                "controller action header",
            ),
            affected_request_rows=_float_matrix(
                rows,
                len(self.layout.controller_request_names),
                "controller affected-request rows",
            ),
        )

    def build_adversary_action(
        self,
        state: GV4State,
        edge: CanonicalAdversaryAction,
    ) -> GV4AdversaryActionFeatures:
        """Serialize one canonical adversary edge from its parent state."""

        self._check_state(state)
        if state.next_player != Player.ADVERSARY:
            raise DNNFeatureError("adversary features require an adversary state")
        if not isinstance(edge, CanonicalAdversaryAction):
            raise TypeError("edge must be CanonicalAdversaryAction")
        action = edge.action
        if (action.launch_count > 0) != (action.prefill_tokens is not None):
            raise DNNFeatureError("adversary launch payload is inconsistent")
        prefill_tokens = action.prefill_tokens or 0
        stopped = [
            self._action_request(state, request_id)
            for request_id in sorted(action.stop_request_ids)
        ]
        if any(not _decode_phase(request) for request in stopped):
            raise DNNFeatureError("adversary stop target is not a decode request")

        header = [
            action.launch_count / self.scales.window_request_cap,
            prefill_tokens / self.scales.request_prefill_scale,
            (action.launch_count * prefill_tokens) / self.scales.window_prefill_cap,
            len(stopped) / self.scales.window_request_cap,
            sum(request.has_inflight_work for request in stopped)
            / self.scales.window_request_cap,
        ]
        rows = []
        for request in stopped:
            deadline_present = request.next_decode_deadline != UNSET_TIME
            rows.append(
                [
                    request.original_decode_tokens / self.scales.request_decode_scale,
                    request.committed_decode_tokens / self.scales.request_decode_scale,
                    request.remaining_decode_tokens / self.scales.request_decode_scale,
                    math.asinh(
                        _current_lateness(request, state.now)
                        / self.scales.launch_age_scale
                    ),
                    float(deadline_present),
                    (
                        math.asinh(
                            (request.next_decode_deadline - state.now)
                            / request.decode_token_slo_sec
                        )
                        if deadline_present
                        else 0.0
                    ),
                    float(request.violation_recorded),
                    float(request.reserved_decode_tokens),
                    float(request.has_inflight_work),
                ]
            )

        return GV4AdversaryActionFeatures(
            header=_float_vector(
                header,
                len(self.layout.adversary_header_names),
                "adversary action header",
            ),
            affected_request_rows=_float_matrix(
                rows,
                len(self.layout.adversary_request_names),
                "adversary affected-request rows",
            ),
        )


__all__ = [
    "ADVERSARY_HEADER_FEATURE_NAMES",
    "ADVERSARY_REQUEST_FEATURE_NAMES",
    "CONTROLLER_HEADER_FEATURE_NAMES",
    "CONTROLLER_REQUEST_FEATURE_NAMES",
    "DNNFeatureError",
    "GLOBAL_FEATURE_NAMES",
    "GV4AdversaryActionFeatures",
    "GV4ControllerActionFeatures",
    "GV4FeatureBuilder",
    "GV4FeatureLayout",
    "GV4FeatureScales",
    "GV4StateFeatures",
    "LAUNCH_FEATURE_NAMES",
    "WORKLOAD_WINDOW_MULTIPLIER",
]
