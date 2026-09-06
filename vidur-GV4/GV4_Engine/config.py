"""Immutable configuration contract for the GV4 virtual game engine.

This module contains only values that affect legal actions, simulator state,
transitions, timing, rewards, or Python/native layout compatibility. Search,
network-training, and distributed-pipeline settings belong to their respective
AlphaGoZero configuration and should compose this engine manifest rather than
being duplicated here.

The dataclasses deliberately fail closed. Predictor inputs and usable KV bytes
must be supplied explicitly; this module never guesses GPU memory capacity or
silently substitutes a different hardware profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence


if TYPE_CHECKING:
    from .vidur_timing_provider import VidurTimingProvider


__all__ = [
    "AdversaryActionConfig",
    "ControllerActionConfig",
    "CostConfig",
    "DecodeCreditConfig",
    "GV4ConfigError",
    "GV4EngineConfig",
    "KVCacheConfig",
    "LayerRange",
    "ModelConfig",
    "NativeLayoutConfig",
    "ReplicaPlacement",
    "RequestConfig",
    "RewardConfig",
    "RoutingConfig",
    "SLOConfig",
    "SchedulerConfig",
    "TimingConfig",
    "TopologyConfig",
    "VidurPredictorConfig",
]


class GV4ConfigError(ValueError):
    """Raised when a resolved GV4 engine manifest is inconsistent."""


_DTYPE_BYTES: Mapping[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
}

_SUPPORTED_EVICTION_RULES = frozenset(
    {
        "evict_none",
        "evict_largest_prefill",
        "evict_earliest_prefill_deadline",
        "evict_prefill_missed_deadline",
        "evict_prefill_lateness_over_0p5",
        "evict_longest_decode",
        "evict_decode_lateness_over_0p5",
        "evict_prefill_highest_lateness",
        "evict_decode_highest_lateness",
    }
)
_SUPPORTED_PREEMPTION_RULES = frozenset(
    {
        "preempt_none",
        "preempt_min_recompute",
        "preempt_largest_kv",
        "preempt_max_recovery_slack",
        "preempt_best_relief_cost",
    }
)
_SUPPORTED_ORDERING_HEURISTICS = frozenset({"SJF", "EDF", "LST", "LJF"})
_SUPPORTED_STOP_RULES = frozenset(
    {
        "stop_none",
        "stop_longest_decode",
        "stop_shortest_decode",
        "stop_all_decodes_over_512",
        "stop_all_decodes_over_216",
    }
)
_SUPPORTED_VIDUR_CACHE_MODES = frozenset({"ignore_cache", "use_cache", "require_cache"})


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GV4ConfigError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GV4ConfigError(f"{name} must be a nonnegative integer")


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise GV4ConfigError(f"{name} must be finite and > 0")


def _nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise GV4ConfigError(f"{name} must be finite and >= 0")


def _nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GV4ConfigError(f"{name} must be a nonempty string")


def _unique(name: str, values: Sequence[Any]) -> None:
    if len(set(values)) != len(values):
        raise GV4ConfigError(f"{name} must not contain duplicates")


def _validate_layer_partition(
    name: str,
    ranges: Sequence["LayerRange"],
    *,
    expected_layers: int | None = None,
) -> None:
    if not ranges:
        raise GV4ConfigError(f"{name} cannot be empty")
    if ranges[0].begin != 0:
        raise GV4ConfigError(f"{name} must begin at layer 0")
    for previous, current in zip(ranges, ranges[1:]):
        if previous.end != current.begin:
            raise GV4ConfigError(f"{name} must be contiguous without gaps or overlap")
    if expected_layers is not None and ranges[-1].end != expected_layers:
        raise GV4ConfigError(
            f"{name} ends at {ranges[-1].end}, expected {expected_layers}"
        )


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _to_plain_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _to_plain_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


@dataclass(frozen=True, slots=True, order=True)
class LayerRange:
    """Half-open model-layer interval assigned to one PP stage."""

    begin: int
    end: int

    def __post_init__(self) -> None:
        _nonnegative_int("layer_range.begin", self.begin)
        _positive_int("layer_range.end", self.end)
        if self.end <= self.begin:
            raise GV4ConfigError("layer_range.end must be greater than begin")

    @property
    def count(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class ReplicaPlacement:
    """Ordered TP rank membership for every PP stage of one replica.
    Essentially marks every Replica's PP with its TP rank/GPU ids
    """

    replica_id: int
    stage_rank_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _nonnegative_int("replica_id", self.replica_id)
        normalized = tuple(
            tuple(int(rank) for rank in stage) for stage in self.stage_rank_ids
        )
        object.__setattr__(self, "stage_rank_ids", normalized)
        if not normalized:
            raise GV4ConfigError("replica placement must contain at least one PP stage")
        flat: list[int] = []
        for stage_index, ranks in enumerate(normalized):
            if not ranks:
                raise GV4ConfigError(
                    f"replica {self.replica_id} stage {stage_index} has no TP ranks"
                )
            for rank in ranks:
                _nonnegative_int("rank_id", rank)
                flat.append(rank)
        _unique(f"replica {self.replica_id} rank IDs", flat)

    @property
    def rank_ids(self) -> tuple[int, ...]:
        return tuple(rank for stage in self.stage_rank_ids for rank in stage)


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    """Homogeneous data-, tensor-, and pipeline-parallel placement.
    \n
    Initialises the following :
     - num_replicas
     - tensor_parallel_size
     - pipeline_parallel_size
     - stage_layer_ranges
     - replica_placements
     - require_equal_stage_layers (to ensure that all PP stages have equal number of layers)
     \n
     Performs the following checks :
     - stage_layer_ranges length must equal pipeline_parallel_size
     - replica_placements length must equal num_replicas
     - rank IDs must be contiguous from zero onwards and should be unique across all replicas
     - all replicas should have the same number of layers in each PP stage
    """

    num_replicas: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    stage_layer_ranges: tuple[LayerRange, ...]
    replica_placements: tuple[ReplicaPlacement, ...]
    require_equal_stage_layers: bool = True

    def __post_init__(self) -> None:
        _positive_int("topology.num_replicas", self.num_replicas)
        _positive_int("topology.tensor_parallel_size", self.tensor_parallel_size)
        _positive_int("topology.pipeline_parallel_size", self.pipeline_parallel_size)
        ranges = tuple(self.stage_layer_ranges)
        placements = tuple(self.replica_placements)
        object.__setattr__(self, "stage_layer_ranges", ranges)
        object.__setattr__(self, "replica_placements", placements)

        if len(ranges) != self.pipeline_parallel_size:
            raise GV4ConfigError(
                "stage_layer_ranges length must equal pipeline_parallel_size"
            )
        _validate_layer_partition("topology.stage_layer_ranges", ranges)
        if (
            self.require_equal_stage_layers
            and len({item.count for item in ranges}) != 1
        ):
            raise GV4ConfigError(
                "unequal PP layer ranges require require_equal_stage_layers=False"
            )

        if len(placements) != self.num_replicas:
            raise GV4ConfigError(
                "replica_placements length must equal topology.num_replicas"
            )
        if tuple(item.replica_id for item in placements) != tuple(
            range(self.num_replicas)
        ):
            raise GV4ConfigError("replica IDs must be ordered and contiguous from zero")

        all_ranks: list[int] = []
        for placement in placements:
            if len(placement.stage_rank_ids) != self.pipeline_parallel_size:
                raise GV4ConfigError(
                    f"replica {placement.replica_id} PP stage count does not match topology"
                )
            for stage_index, ranks in enumerate(placement.stage_rank_ids):
                if len(ranks) != self.tensor_parallel_size:
                    raise GV4ConfigError(
                        f"replica {placement.replica_id} stage {stage_index} has "
                        f"{len(ranks)} ranks, expected {self.tensor_parallel_size}"
                    )
            all_ranks.extend(placement.rank_ids)

        _unique("global rank IDs", all_ranks)
        expected_ranks = list(range(self.total_ranks))
        if sorted(all_ranks) != expected_ranks:
            raise GV4ConfigError(
                "rank IDs must be contiguous from zero for the compact native layout"
            )

    @classmethod
    def contiguous(
        cls,
        *,
        num_replicas: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        num_layers: int,
    ) -> "TopologyConfig":
        _positive_int("num_replicas", num_replicas)
        _positive_int("tensor_parallel_size", tensor_parallel_size)
        _positive_int("pipeline_parallel_size", pipeline_parallel_size)
        _positive_int("num_layers", num_layers)
        if num_layers % pipeline_parallel_size != 0:
            raise GV4ConfigError(
                "equal stage placement requires num_layers divisible by PP; "
                "construct TopologyConfig explicitly for a profiled uneven partition"
            )
        layers_per_stage = num_layers // pipeline_parallel_size
        ranges = tuple(
            LayerRange(index * layers_per_stage, (index + 1) * layers_per_stage)
            for index in range(pipeline_parallel_size)
        )
        next_rank = 0
        placements: list[ReplicaPlacement] = []
        for replica_id in range(num_replicas):
            stages: list[tuple[int, ...]] = []
            for _ in range(pipeline_parallel_size):
                stages.append(tuple(range(next_rank, next_rank + tensor_parallel_size)))
                next_rank += tensor_parallel_size
            placements.append(ReplicaPlacement(replica_id, tuple(stages)))
        return cls(
            num_replicas=num_replicas,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            stage_layer_ranges=ranges,
            replica_placements=tuple(placements),
        )

    @property
    def ranks_per_replica(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def total_ranks(self) -> int:
        return self.num_replicas * self.ranks_per_replica

    @property
    def layers_on_stages(self) -> tuple[int, ...]:
        return tuple(item.count for item in self.stage_layer_ranges)

    def stage_index_for_rank(self, rank_id: int) -> int:
        for placement in self.replica_placements:
            for stage_index, ranks in enumerate(placement.stage_rank_ids):
                if rank_id in ranks:
                    return stage_index
        raise GV4ConfigError(f"rank {rank_id} is not part of this topology")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model dimensions that influence execution and KV-cache geometry. \n
    We can derive the following from the model config params below for example :
    KV bytes/token/rank =
                            2
                            * bytes per KV element
                            * layers on this PP stage
                            * head dimension
                            * KV heads on this TP rank

    For example for the Llama 3-8B with TP/PP1 :
    2 * 2 bytes * 32 layers * 128 head_dim * 8 KV heads
        = 131,072 bytes per token per rank (additional 2 is for key and value)
    """

    model_id: str
    model_revision: str
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    head_dim: int = 128
    num_kv_heads: int = 8
    vocab_size: int = 128_256
    weight_dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        _nonempty_string("model.model_id", self.model_id)
        _nonempty_string("model.model_revision", self.model_revision)
        if self.model_revision.strip().lower() in {"main", "master", "latest"}:
            raise GV4ConfigError("model_revision must identify an immutable revision")
        for name, value in (
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("head_dim", self.head_dim),
            ("num_kv_heads", self.num_kv_heads),
            ("vocab_size", self.vocab_size),
        ):
            _positive_int(f"model.{name}", value)
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise GV4ConfigError(
                "model.hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise GV4ConfigError(
                "num_attention_heads must be divisible by num_kv_heads"
            )
        for name, dtype in (
            ("weight_dtype", self.weight_dtype),
            ("kv_dtype", self.kv_dtype),
        ):
            if dtype not in _DTYPE_BYTES:
                raise GV4ConfigError(
                    f"unsupported {name} {dtype!r}; set an explicitly supported dtype"
                )

    @property
    def kv_element_bytes(self) -> int:
        return _DTYPE_BYTES[self.kv_dtype]


@dataclass(frozen=True, slots=True)
class VidurPredictorConfig:
    """Inputs needed to build or load Vidur's timing tables once at startup.

    Model identity comes from ``ModelConfig``; TP and PP come from
    ``TopologyConfig``; KV block size comes from ``KVCacheConfig``. Keeping those
    values out of this record prevents two conflicting sources of truth.
    """

    device: str
    network_device: str
    cache_dir: str
    cache_mode: str = "use_cache"
    prediction_max_tokens_per_request: int = 8192
    prediction_max_batch_size: int = 256
    prediction_max_prefill_chunk_size: int = 4096
    kv_cache_prediction_granularity: int = 64
    prefill_chunk_size_prediction_granularity: int = 32
    prefill_profile_step_tokens: int = 128
    num_training_job_threads: int = 1

    def __post_init__(self) -> None:
        _nonempty_string("vidur_predictor.device", self.device)
        _nonempty_string("vidur_predictor.network_device", self.network_device)
        _nonempty_string("vidur_predictor.cache_dir", self.cache_dir)

        cache_mode = self.cache_mode.strip().lower()
        if cache_mode not in _SUPPORTED_VIDUR_CACHE_MODES:
            raise GV4ConfigError(
                "vidur_predictor.cache_mode must be ignore_cache, use_cache, "
                "or require_cache"
            )
        object.__setattr__(self, "cache_mode", cache_mode)
        object.__setattr__(
            self,
            "cache_dir",
            str(Path(self.cache_dir).expanduser().resolve()),
        )

        for name, value in (
            (
                "prediction_max_tokens_per_request",
                self.prediction_max_tokens_per_request,
            ),
            ("prediction_max_batch_size", self.prediction_max_batch_size),
            (
                "prediction_max_prefill_chunk_size",
                self.prediction_max_prefill_chunk_size,
            ),
            ("kv_cache_prediction_granularity", self.kv_cache_prediction_granularity),
            (
                "prefill_chunk_size_prediction_granularity",
                self.prefill_chunk_size_prediction_granularity,
            ),
            ("prefill_profile_step_tokens", self.prefill_profile_step_tokens),
            ("num_training_job_threads", self.num_training_job_threads),
        ):
            _positive_int(f"vidur_predictor.{name}", value)

        if (
            self.prediction_max_tokens_per_request
            % self.kv_cache_prediction_granularity
        ):
            raise GV4ConfigError(
                "prediction token maximum must be divisible by KV granularity"
            )
        if (
            self.prediction_max_prefill_chunk_size
            % self.prefill_chunk_size_prediction_granularity
        ):
            raise GV4ConfigError(
                "prefill prediction maximum must be divisible by its granularity"
            )
        if (
            self.prefill_profile_step_tokens
            % self.prefill_chunk_size_prediction_granularity
        ):
            raise GV4ConfigError(
                "prefill profile step must align to predictor prefill granularity"
            )


@dataclass(frozen=True, slots=True)
class KVCacheConfig:
    """Configure the hard per-rank KV-cache capacity.

    `kv_budget_bytes_per_rank` is the memory available after model weights,
    runtime buffers, and other non-KV allocations have been reserved, but
    before applying the GV4 safety margin.

    GV4 applies `memory_safety_margin_fraction` once and treats the remaining
    memory as a hard allocation limit:

        hard_limit = kv_budget * (1 - memory_safety_margin_fraction)

    The KV ledger converts this hard byte limit into whole KV blocks. Any
    controller action requiring more blocks than the remaining hard capacity
    is illegal. GV4 v1 has no separate allocator watermark, preemption,
    prefix caching, or KV offload.
    """

    kv_budget_bytes_per_rank: tuple[int, ...]
    block_size_tokens: int = 16
    memory_safety_margin_fraction: float = 0.10
    block_preallocation_granularity: int = 64
    prefix_caching_enabled: bool = False
    cpu_offload_enabled: bool = False
    disk_offload_enabled: bool = False

    def __post_init__(self) -> None:
        budgets = tuple(int(value) for value in self.kv_budget_bytes_per_rank)
        object.__setattr__(self, "kv_budget_bytes_per_rank", budgets)

        if not budgets:
            raise GV4ConfigError("kv_budget_bytes_per_rank cannot be empty")

        for rank_id, value in enumerate(budgets):
            _positive_int(f"kv_budget_bytes_per_rank[{rank_id}]", value)

        _positive_int("kv.block_size_tokens", self.block_size_tokens)
        _positive_int(
            "kv.block_preallocation_granularity",
            self.block_preallocation_granularity,
        )

        margin = float(self.memory_safety_margin_fraction)
        if not math.isfinite(margin) or not 0.0 <= margin < 1.0:
            raise GV4ConfigError(
                "kv.memory_safety_margin_fraction must be finite and in [0, 1)"
            )

        if self.prefix_caching_enabled:
            raise GV4ConfigError("prefix caching is deferred in GV4 state schema v1")

        if self.cpu_offload_enabled or self.disk_offload_enabled:
            raise GV4ConfigError("KV offload is deferred in GV4 state schema v1")

    def hard_kv_byte_limits_per_rank(self) -> tuple[int, ...]:
        usable_fraction = 1.0 - self.memory_safety_margin_fraction
        return tuple(
            math.floor(budget * usable_fraction)
            for budget in self.kv_budget_bytes_per_rank
        )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Per-replica batch and compact PP-calendar limits."""

    max_batch_tokens: int = 4096 + 512
    max_sequences: int = 256  ## number of unique requests in a batch
    max_prefill_chunk_tokens: int = 4096
    max_inflight_microbatches: int = 1
    inter_stage_queue_capacity: int = 1
    decode_tokens_per_request_per_batch: int = 1
    fifo_stages: bool = True
    deterministic_service_times: bool = True
    request_preemption_enabled: bool = True
    inflight_eviction_enabled: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("max_batch_tokens", self.max_batch_tokens),
            ("max_sequences", self.max_sequences),
            ("max_prefill_chunk_tokens", self.max_prefill_chunk_tokens),
            ("max_inflight_microbatches", self.max_inflight_microbatches),
            ("inter_stage_queue_capacity", self.inter_stage_queue_capacity),
            (
                "decode_tokens_per_request_per_batch",
                self.decode_tokens_per_request_per_batch,
            ),
        ):
            _positive_int(f"scheduler.{name}", value)
        if self.inter_stage_queue_capacity < self.max_inflight_microbatches:
            raise GV4ConfigError(
                "inter_stage_queue_capacity must be >= max_inflight_microbatches "
                "until explicit PP backpressure is implemented"
            )
        if self.decode_tokens_per_request_per_batch != 1:
            raise GV4ConfigError("GV4 v1 admits at most one decode token per request")
        if not self.fifo_stages:
            raise GV4ConfigError("non-FIFO PP scheduling is deferred in GV4 v1")
        if not self.deterministic_service_times:
            raise GV4ConfigError("GV4 v1 requires deterministic stage service times")
        if not isinstance(self.request_preemption_enabled, bool):
            raise GV4ConfigError("request_preemption_enabled must be bool")
        if self.inflight_eviction_enabled:
            raise GV4ConfigError("in-flight terminal eviction is illegal in GV4 v1")
        if self.max_prefill_chunk_tokens > self.max_batch_tokens:
            raise GV4ConfigError(
                "max_prefill_chunk_tokens cannot exceed max_batch_tokens"
            )


@dataclass(frozen=True, slots=True)
class TimingConfig:
    adversary_tick_sec: float = 0.2
    launch_window_sec: float = 1.0
    max_requests_per_launch_window: int = 7
    epsilon: float = 1e-6
    time_round_digits: int = 10
    max_zero_time_transitions_per_boundary: int = 512

    def __post_init__(self) -> None:
        _positive_finite("timing.adversary_tick_sec", self.adversary_tick_sec)
        _positive_finite("timing.launch_window_sec", self.launch_window_sec)
        _positive_int(
            "timing.max_requests_per_launch_window",
            self.max_requests_per_launch_window,
        )
        _positive_finite("timing.epsilon", self.epsilon)
        _nonnegative_int("timing.time_round_digits", self.time_round_digits)
        _positive_int(
            "timing.max_zero_time_transitions_per_boundary",
            self.max_zero_time_transitions_per_boundary,
        )


@dataclass(frozen=True, slots=True)
class DecodeCreditConfig:
    """Pooled adversary decode budget minted by completed prefills."""

    decode_credit_mint_per_prefill_completion: int = 216
    enforce_nonnegative_decode_credits: bool = True

    def __post_init__(self) -> None:
        _positive_int(
            "credits.decode_credit_mint_per_prefill_completion",
            self.decode_credit_mint_per_prefill_completion,
        )
        if not self.enforce_nonnegative_decode_credits:
            raise GV4ConfigError("decode credits must be guarded against overspending")


@dataclass(frozen=True, slots=True)
class RequestConfig:
    max_prefill_tokens_per_request: int = 4096
    min_decode_tokens_per_request: int = 1
    max_decode_tokens_per_request: int = 864
    target_decode_tokens_per_request_average: int = 216
    target_prefill_tokens_per_request_window_average: int = 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("max_prefill_tokens_per_request", self.max_prefill_tokens_per_request),
            ("min_decode_tokens_per_request", self.min_decode_tokens_per_request),
            ("max_decode_tokens_per_request", self.max_decode_tokens_per_request),
            (
                "target_decode_tokens_per_request_average",
                self.target_decode_tokens_per_request_average,
            ),
            (
                "target_prefill_tokens_per_request_window_average",
                self.target_prefill_tokens_per_request_window_average,
            ),
        ):
            _positive_int(f"request.{name}", value)
        if self.min_decode_tokens_per_request > self.max_decode_tokens_per_request:
            raise GV4ConfigError("minimum decode tokens cannot exceed maximum")
        if (
            self.target_decode_tokens_per_request_average
            > self.max_decode_tokens_per_request
        ):
            raise GV4ConfigError("target decode average exceeds request maximum")
        if (
            self.target_prefill_tokens_per_request_window_average
            > self.max_prefill_tokens_per_request
        ):
            raise GV4ConfigError("target prefill average exceeds request maximum")


@dataclass(frozen=True, slots=True)
class SLOConfig:
    prefill_slowdown_factor: float = 3.0
    decode_token_slo_sec: float = 0.050

    def __post_init__(self) -> None:
        _positive_finite("slo.prefill_slowdown_factor", self.prefill_slowdown_factor)
        _positive_finite("slo.decode_token_slo_sec", self.decode_token_slo_sec)


@dataclass(frozen=True, slots=True)
class CostConfig:
    violation_base_cost: float = 1.0
    lateness_cap_sec: float = 2.0
    terminal_drop_cost: float = 3.0
    automatic_drop_lateness_sec: float = 2.0

    def __post_init__(self) -> None:
        _nonnegative_finite("cost.violation_base_cost", self.violation_base_cost)
        _positive_finite("cost.lateness_cap_sec", self.lateness_cap_sec)
        _positive_finite("cost.terminal_drop_cost", self.terminal_drop_cost)
        _positive_finite(
            "cost.automatic_drop_lateness_sec", self.automatic_drop_lateness_sec
        )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Time-varying discount contract shared with MCTS and replay targets."""

    discount_factor: float = 0.98
    discount_reference_step_sec: float = 0.015725797204323228
    objective_mode: str = "minimize_cost"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.discount_factor)) or not (
            0.0 < float(self.discount_factor) <= 1.0
        ):
            raise GV4ConfigError("reward.discount_factor must be in (0, 1]")
        _positive_finite(
            "reward.discount_reference_step_sec", self.discount_reference_step_sec
        )
        if self.objective_mode != "minimize_cost":
            raise GV4ConfigError("GV4 v1 uses the minimize_cost value convention")

    def discount_for_elapsed(self, elapsed_sec: float) -> float:
        _nonnegative_finite("elapsed_sec", elapsed_sec)
        exponent = float(elapsed_sec) / self.discount_reference_step_sec
        return self.discount_factor**exponent


@dataclass(frozen=True, slots=True)
class ControllerActionConfig:
    preemption_rule_names: tuple[str, ...] = (
        "preempt_none",
        "preempt_min_recompute",
        "preempt_largest_kv",
        "preempt_max_recovery_slack",
        "preempt_best_relief_cost",
    )
    eviction_rule_names: tuple[str, ...] = (
        "evict_none",
        "evict_largest_prefill",
        "evict_earliest_prefill_deadline",
        "evict_prefill_missed_deadline",
        "evict_prefill_lateness_over_0p5",
        "evict_longest_decode",
        "evict_decode_lateness_over_0p5",
        "evict_prefill_highest_lateness",
        "evict_decode_highest_lateness",
    )
    prefill_budget_options: tuple[int, ...] = (
        0,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
    )
    ordering_heuristics: tuple[str, ...] = ("SJF", "EDF", "LST", "LJF")

    def __post_init__(self) -> None:
        preemption_rules = tuple(self.preemption_rule_names)
        rules = tuple(self.eviction_rule_names)
        budgets = tuple(int(value) for value in self.prefill_budget_options)
        heuristics = tuple(self.ordering_heuristics)
        object.__setattr__(self, "preemption_rule_names", preemption_rules)
        object.__setattr__(self, "eviction_rule_names", rules)
        object.__setattr__(self, "prefill_budget_options", budgets)
        object.__setattr__(self, "ordering_heuristics", heuristics)
        if not preemption_rules or preemption_rules[0] != "preempt_none":
            raise GV4ConfigError(
                "controller preemption rules must begin with preempt_none"
            )
        if not rules or rules[0] != "evict_none":
            raise GV4ConfigError("controller eviction rules must begin with evict_none")
        if not budgets or budgets[0] != 0:
            raise GV4ConfigError("controller prefill budgets must begin with zero")
        if not heuristics:
            raise GV4ConfigError("controller ordering heuristics cannot be empty")
        _unique("controller preemption rules", preemption_rules)
        _unique("controller eviction rules", rules)
        _unique("controller prefill budgets", budgets)
        _unique("controller ordering heuristics", heuristics)
        unsupported_preemption = (
            set(preemption_rules) - _SUPPORTED_PREEMPTION_RULES
        )
        unsupported_rules = set(rules) - _SUPPORTED_EVICTION_RULES
        unsupported_heuristics = set(heuristics) - _SUPPORTED_ORDERING_HEURISTICS
        if unsupported_preemption:
            raise GV4ConfigError(
                "unsupported preemption rules: "
                f"{sorted(unsupported_preemption)}"
            )
        if unsupported_rules:
            raise GV4ConfigError(
                f"unsupported eviction rules: {sorted(unsupported_rules)}"
            )
        if unsupported_heuristics:
            raise GV4ConfigError(
                f"unsupported ordering heuristics: {sorted(unsupported_heuristics)}"
            )
        if any(value < 0 for value in budgets):
            raise GV4ConfigError("controller prefill budgets must be nonnegative")
        if tuple(sorted(budgets)) != budgets:
            raise GV4ConfigError("controller prefill budgets must be ascending")

    @property
    def raw_action_count(self) -> int:
        return (
            len(self.preemption_rule_names)
            * len(self.eviction_rule_names)
            * len(self.prefill_budget_options)
            * len(self.ordering_heuristics)
        )

    def encode_raw_index(
        self,
        eviction_rule_index: int,
        prefill_budget_index: int,
        ordering_heuristic_index: int,
        *,
        preemption_rule_index: int = 0,
    ) -> int:
        if not 0 <= preemption_rule_index < len(self.preemption_rule_names):
            raise GV4ConfigError("preemption_rule_index is out of range")
        if not 0 <= eviction_rule_index < len(self.eviction_rule_names):
            raise GV4ConfigError("eviction_rule_index is out of range")
        if not 0 <= prefill_budget_index < len(self.prefill_budget_options):
            raise GV4ConfigError("prefill_budget_index is out of range")
        if not 0 <= ordering_heuristic_index < len(self.ordering_heuristics):
            raise GV4ConfigError("ordering_heuristic_index is out of range")
        return (
            (
                preemption_rule_index * len(self.eviction_rule_names)
                + eviction_rule_index
            )
            * len(self.prefill_budget_options)
            + prefill_budget_index
        ) * len(self.ordering_heuristics) + ordering_heuristic_index

    def decode_raw_index(self, raw_index: int) -> tuple[int, int, int, int]:
        if not 0 <= raw_index < self.raw_action_count:
            raise GV4ConfigError("controller raw action index is out of range")
        rule_budget_index, heuristic_index = divmod(
            raw_index, len(self.ordering_heuristics)
        )
        preemption_eviction_index, budget_index = divmod(
            rule_budget_index, len(self.prefill_budget_options)
        )
        preemption_index, eviction_index = divmod(
            preemption_eviction_index, len(self.eviction_rule_names)
        )
        return preemption_index, eviction_index, budget_index, heuristic_index

    def raw_action_components(self, raw_index: int) -> tuple[str, str, int, str]:
        (
            preemption_index,
            eviction_index,
            budget_index,
            heuristic_index,
        ) = self.decode_raw_index(raw_index)
        return (
            self.preemption_rule_names[preemption_index],
            self.eviction_rule_names[eviction_index],
            self.prefill_budget_options[budget_index],
            self.ordering_heuristics[heuristic_index],
        )


@dataclass(frozen=True, slots=True)
class AdversaryActionConfig:
    max_launch_count_per_tick: int = 7
    prefill_token_templates: tuple[int, ...] = (
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
    )
    stop_rule_names: tuple[str, ...] = (
        "stop_none",
        "stop_longest_decode",
        "stop_shortest_decode",
        "stop_all_decodes_over_512",
        "stop_all_decodes_over_216",
    )
    strict_masking: bool = True

    def __post_init__(self) -> None:
        templates = tuple(int(value) for value in self.prefill_token_templates)
        stop_rules = tuple(self.stop_rule_names)
        object.__setattr__(self, "prefill_token_templates", templates)
        object.__setattr__(self, "stop_rule_names", stop_rules)
        _nonnegative_int(
            "adversary.max_launch_count_per_tick", self.max_launch_count_per_tick
        )
        if not templates or any(value <= 0 for value in templates):
            raise GV4ConfigError("adversary prefill templates must be positive")
        if tuple(sorted(templates)) != templates:
            raise GV4ConfigError("adversary prefill templates must be ascending")
        if not stop_rules or stop_rules[0] != "stop_none":
            raise GV4ConfigError("adversary stop rules must begin with stop_none")
        _unique("adversary prefill templates", templates)
        _unique("adversary stop rules", stop_rules)
        unsupported = set(stop_rules) - _SUPPORTED_STOP_RULES
        if unsupported:
            raise GV4ConfigError(
                f"unsupported adversary stop rules: {sorted(unsupported)}"
            )
        if not self.strict_masking:
            raise GV4ConfigError("GV4 requires strict adversary action masking")

    @property
    def raw_action_count(self) -> int:
        # Launch count zero omits the meaningless prefill-template dimension.
        return len(self.stop_rule_names) * (
            1 + self.max_launch_count_per_tick * len(self.prefill_token_templates)
        )

    def encode_raw_index(
        self,
        launch_count: int,
        prefill_template_index: int | None,
        stop_rule_index: int,
    ) -> int:
        if not 0 <= launch_count <= self.max_launch_count_per_tick:
            raise GV4ConfigError("launch_count is out of range")
        if not 0 <= stop_rule_index < len(self.stop_rule_names):
            raise GV4ConfigError("stop_rule_index is out of range")
        if launch_count == 0:
            if prefill_template_index is not None:
                raise GV4ConfigError("launch_count zero has no prefill template")
            return stop_rule_index
        if prefill_template_index is None or not (
            0 <= prefill_template_index < len(self.prefill_token_templates)
        ):
            raise GV4ConfigError("prefill_template_index is out of range")
        launch_template_index = (launch_count - 1) * len(
            self.prefill_token_templates
        ) + prefill_template_index
        return len(self.stop_rule_names) + (
            launch_template_index * len(self.stop_rule_names) + stop_rule_index
        )

    def decode_raw_index(self, raw_index: int) -> tuple[int, int | None, int]:
        if not 0 <= raw_index < self.raw_action_count:
            raise GV4ConfigError("adversary raw action index is out of range")
        stop_rule_count = len(self.stop_rule_names)
        if raw_index < stop_rule_count:
            return 0, None, raw_index
        launch_template_index, stop_rule_index = divmod(
            raw_index - stop_rule_count, stop_rule_count
        )
        launch_zero_based, template_index = divmod(
            launch_template_index, len(self.prefill_token_templates)
        )
        return launch_zero_based + 1, template_index, stop_rule_index

    def raw_action_components(self, raw_index: int) -> tuple[int, int | None, str]:
        launch_count, template_index, stop_rule_index = self.decode_raw_index(raw_index)
        prefill_tokens = (
            None
            if template_index is None
            else self.prefill_token_templates[template_index]
        )
        return launch_count, prefill_tokens, self.stop_rule_names[stop_rule_index]


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Sequential global routing schema; scheduling remains per replica."""

    enabled: bool = False
    allow_hold: bool = True
    implicit_single_replica_assignment: bool = True
    one_request_per_zero_time_decision: bool = True
    assignment_order: str = "arrival_then_request_id"
    max_zero_time_assignments_per_boundary: int = 512

    def __post_init__(self) -> None:
        if not self.one_request_per_zero_time_decision:
            raise GV4ConfigError("combinatorial multi-request routing is forbidden")
        if self.assignment_order != "arrival_then_request_id":
            raise GV4ConfigError("GV4 v1 supports only arrival_then_request_id routing")
        _positive_int(
            "routing.max_zero_time_assignments_per_boundary",
            self.max_zero_time_assignments_per_boundary,
        )

    def raw_action_count(self, num_replicas: int) -> int:
        _positive_int("num_replicas", num_replicas)
        if not self.enabled:
            return 0
        return num_replicas + int(self.allow_hold)


@dataclass(frozen=True, slots=True)
class NativeLayoutConfig:
    """Versioned fixed-array bounds shared by Python snapshots and native code."""

    manifest_schema_version: str = "gv4_engine_manifest_v7"
    state_schema_version: str = "gv4_state_v5"
    action_schema_version: str = "gv4_actions_v3"
    feature_schema_version: str = "gv4_markov_v5"
    native_layout_version: str = "gv4_native_layout_v5"
    max_requests: int = 512
    max_unassigned_requests: int = 512
    max_replicas: int = 8
    max_tensor_parallel_size: int = 8
    max_pipeline_parallel_size: int = 8
    max_inflight_microbatches_per_replica: int = 64
    max_launch_history_entries: int = 64
    max_controller_raw_actions: int = 2048
    max_adversary_raw_actions: int = 1024
    max_router_raw_actions: int = 16

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_schema_version", self.manifest_schema_version),
            ("state_schema_version", self.state_schema_version),
            ("action_schema_version", self.action_schema_version),
            ("feature_schema_version", self.feature_schema_version),
            ("native_layout_version", self.native_layout_version),
        ):
            _nonempty_string(f"layout.{name}", value)
        if not self.feature_schema_version.startswith("gv4_markov_"):
            raise GV4ConfigError("legacy feature schemas are not valid for GV4")
        for name, value in (
            ("max_requests", self.max_requests),
            ("max_unassigned_requests", self.max_unassigned_requests),
            ("max_replicas", self.max_replicas),
            ("max_tensor_parallel_size", self.max_tensor_parallel_size),
            ("max_pipeline_parallel_size", self.max_pipeline_parallel_size),
            (
                "max_inflight_microbatches_per_replica",
                self.max_inflight_microbatches_per_replica,
            ),
            ("max_launch_history_entries", self.max_launch_history_entries),
            ("max_controller_raw_actions", self.max_controller_raw_actions),
            ("max_adversary_raw_actions", self.max_adversary_raw_actions),
            ("max_router_raw_actions", self.max_router_raw_actions),
        ):
            _positive_int(f"layout.{name}", value)
        if self.max_unassigned_requests > self.max_requests:
            raise GV4ConfigError("max_unassigned_requests cannot exceed max_requests")


@dataclass(frozen=True, slots=True)
class GV4EngineConfig:
    """Resolved immutable engine manifest used by every GV4 implementation."""

    model: ModelConfig
    topology: TopologyConfig
    vidur_predictor: VidurPredictorConfig
    kv_cache: KVCacheConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    credits: DecodeCreditConfig = field(default_factory=DecodeCreditConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    controller_actions: ControllerActionConfig = field(
        default_factory=ControllerActionConfig
    )
    adversary_actions: AdversaryActionConfig = field(
        default_factory=AdversaryActionConfig
    )
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    layout: NativeLayoutConfig = field(default_factory=NativeLayoutConfig)
    global_seed: int = 6
    snapshot_rng_state: bool = True
    enable_debug_asserts: bool = True
    fail_fast_on_invalid_state: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        topology = self.topology
        model = self.model
        predictor = self.vidur_predictor

        _nonnegative_int("global_seed", self.global_seed)
        if not self.snapshot_rng_state:
            raise GV4ConfigError("stochastic GV4 state must snapshot its RNG state")

        _validate_layer_partition(
            "topology.stage_layer_ranges",
            topology.stage_layer_ranges,
            expected_layers=model.num_layers,
        )

        tp = topology.tensor_parallel_size
        if model.hidden_size % tp != 0:
            raise GV4ConfigError("model hidden size must be divisible by TP")
        if model.num_attention_heads % tp != 0:
            raise GV4ConfigError("attention heads must be divisible by TP")
        self.kv_heads_on_tensor_parallel_rank()

        if len(self.kv_cache.kv_budget_bytes_per_rank) != topology.total_ranks:
            raise GV4ConfigError(
                "kv_budget_bytes_per_rank length must equal total configured ranks"
            )
        if any(capacity <= 0 for capacity in self.rank_kv_block_capacities()):
            raise GV4ConfigError(
                "every rank must have capacity for at least one KV block"
            )

        if self.scheduler.max_sequences > predictor.prediction_max_batch_size:
            raise GV4ConfigError(
                "scheduler max_sequences exceeds the Vidur predictor batch range"
            )
        if (
            self.scheduler.max_prefill_chunk_tokens
            > predictor.prediction_max_prefill_chunk_size
        ):
            raise GV4ConfigError(
                "scheduler max prefill chunk exceeds the Vidur predictor range"
            )
        if (
            self.scheduler.max_batch_tokens
            > predictor.prediction_max_tokens_per_request
        ):
            raise GV4ConfigError(
                "scheduler max batch tokens exceeds the Vidur predictor token range"
            )
        if (
            self.request.max_prefill_tokens_per_request
            > self.scheduler.max_prefill_chunk_tokens
        ):
            raise GV4ConfigError(
                "request prefill maximum exceeds scheduler max prefill chunk"
            )
        if (
            self.request.max_prefill_tokens_per_request
            % predictor.prefill_profile_step_tokens
        ):
            raise GV4ConfigError(
                "request prefill maximum must be divisible by the prefill "
                "profile step"
            )
        decode_mint = self.credits.decode_credit_mint_per_prefill_completion
        if decode_mint != self.request.target_decode_tokens_per_request_average:
            raise GV4ConfigError(
                "decode credit mint must equal the target decode-token average"
            )
        if decode_mint > self.request.max_decode_tokens_per_request:
            raise GV4ConfigError("decode credit mint exceeds request decode maximum")

        interval = predictor.prefill_chunk_size_prediction_granularity
        aligned_values: Iterable[tuple[str, int]] = (
            (f"adversary prefill template {value}", value)
            for value in self.adversary_actions.prefill_token_templates
        )
        for name, value in aligned_values:
            if value % interval != 0:
                raise GV4ConfigError(
                    f"{name} is not aligned to predictor interval {interval}"
                )
            if value % predictor.prefill_profile_step_tokens != 0:
                raise GV4ConfigError(
                    f"{name} is not aligned to prefill profile step "
                    f"{predictor.prefill_profile_step_tokens}"
                )
        for value in self.controller_actions.prefill_budget_options:
            if value and value % interval != 0:
                raise GV4ConfigError(
                    f"controller prefill budget {value} is not aligned to "
                    f"predictor interval {interval}"
                )
        if (
            max(self.adversary_actions.prefill_token_templates)
            > self.request.max_prefill_tokens_per_request
        ):
            raise GV4ConfigError("adversary prefill template exceeds request maximum")
        if (
            max(self.controller_actions.prefill_budget_options)
            > self.scheduler.max_prefill_chunk_tokens
        ):
            raise GV4ConfigError(
                "controller prefill budget exceeds scheduler chunk maximum"
            )
        if (
            self.adversary_actions.max_launch_count_per_tick
            > self.timing.max_requests_per_launch_window
        ):
            raise GV4ConfigError(
                "one adversary launch action exceeds launch-window capacity"
            )

        if topology.num_replicas > 1 and not self.routing.enabled:
            raise GV4ConfigError("multiple replicas require explicit routing")
        if (
            topology.num_replicas == 1
            and not self.routing.enabled
            and not self.routing.implicit_single_replica_assignment
        ):
            raise GV4ConfigError(
                "a disabled one-replica router requires implicit assignment"
            )

        layout = self.layout
        for actual, maximum, label in (
            (topology.num_replicas, layout.max_replicas, "replicas"),
            (
                topology.tensor_parallel_size,
                layout.max_tensor_parallel_size,
                "tensor-parallel size",
            ),
            (
                topology.pipeline_parallel_size,
                layout.max_pipeline_parallel_size,
                "pipeline-parallel size",
            ),
            (
                self.scheduler.max_inflight_microbatches,
                layout.max_inflight_microbatches_per_replica,
                "in-flight microbatches",
            ),
            (
                self.controller_actions.raw_action_count,
                layout.max_controller_raw_actions,
                "controller raw actions",
            ),
            (
                self.adversary_actions.raw_action_count,
                layout.max_adversary_raw_actions,
                "adversary raw actions",
            ),
            (
                self.routing.raw_action_count(topology.num_replicas),
                layout.max_router_raw_actions,
                "router raw actions",
            ),
        ):
            if actual > maximum:
                raise GV4ConfigError(
                    f"configured {label} ({actual}) exceed "
                    f"native layout bound ({maximum})"
                )
        if layout.max_requests < self.timing.max_requests_per_launch_window:
            raise GV4ConfigError("native request capacity is below one launch window")
        if (
            layout.max_launch_history_entries
            < self.timing.max_requests_per_launch_window
        ):
            raise GV4ConfigError("launch-history layout cannot hold one full window")
        if (
            self.routing.max_zero_time_assignments_per_boundary
            < layout.max_unassigned_requests
        ):
            raise GV4ConfigError(
                "routing zero-time guard must cover every unassigned request slot"
            )

    def kv_heads_on_tensor_parallel_rank(self) -> int:
        """Return physical KV heads per rank, including replicated-GQA heads."""

        tp = self.topology.tensor_parallel_size
        kv_heads = self.model.num_kv_heads
        if kv_heads >= tp:
            if kv_heads % tp != 0:
                raise GV4ConfigError("KV heads must be divisible by TP")
            return kv_heads // tp
        if tp % kv_heads != 0:
            raise GV4ConfigError(
                "when TP exceeds KV heads, TP must be divisible by KV heads"
            )
        return 1

    def kv_bytes_per_token_by_stage(self) -> tuple[int, ...]:
        kv_heads_per_rank = self.kv_heads_on_tensor_parallel_rank()
        return tuple(
            2
            * self.model.kv_element_bytes
            * layer_range.count
            * self.model.head_dim
            * kv_heads_per_rank
            for layer_range in self.topology.stage_layer_ranges
        )

    def rank_kv_block_capacities(self) -> tuple[int, ...]:
        bytes_per_token = self.kv_bytes_per_token_by_stage()
        capacities: list[int] = []
        for rank_id, usable_bytes in enumerate(
            self.kv_cache.hard_kv_byte_limits_per_rank()
        ):
            stage_index = self.topology.stage_index_for_rank(rank_id)
            bytes_per_block = (
                bytes_per_token[stage_index] * self.kv_cache.block_size_tokens
            )
            capacities.append(usable_bytes // bytes_per_block)
        return tuple(capacities)

    def create_vidur_timing_provider(
        self,
        *,
        max_cache_entries: int = 65_536,
    ) -> "VidurTimingProvider":
        """Load/build Vidur timing artifacts once and return the GV4 adapter."""

        from .vidur_timing_provider import VidurTimingProvider

        return VidurTimingProvider.from_config(
            self, max_cache_entries=max_cache_entries
        )

    def replica_logical_block_capacities(self) -> tuple[int, ...]:
        rank_capacities = self.rank_kv_block_capacities()
        return tuple(
            min(rank_capacities[rank_id] for rank_id in placement.rank_ids)
            for placement in self.topology.replica_placements
        )

    def derived_dimensions(self) -> dict[str, Any]:
        return {
            "ranks_per_replica": self.topology.ranks_per_replica,
            "total_ranks": self.topology.total_ranks,
            "layers_on_stages": list(self.topology.layers_on_stages),
            "kv_heads_on_tensor_parallel_rank": (
                self.kv_heads_on_tensor_parallel_rank()
            ),
            "kv_bytes_per_token_by_stage": list(self.kv_bytes_per_token_by_stage()),
            "kv_blocks_per_rank": list(self.rank_kv_block_capacities()),
            "logical_kv_blocks_per_replica": list(
                self.replica_logical_block_capacities()
            ),
            "controller_raw_action_count": (self.controller_actions.raw_action_count),
            "adversary_raw_action_count": (self.adversary_actions.raw_action_count),
            "router_raw_action_count": self.routing.raw_action_count(
                self.topology.num_replicas
            ),
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.layout.manifest_schema_version,
            "engine": _to_plain_data(self),
            "derived": self.derived_dimensions(),
        }

    def to_manifest_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_manifest_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )

    def manifest_sha256(self) -> str:
        canonical = self.to_manifest_json(indent=None).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
