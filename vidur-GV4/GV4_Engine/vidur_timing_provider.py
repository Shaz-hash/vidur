"""Adapt GV4 batches to Vidur's cached execution-time predictor."""

from __future__ import annotations

from collections import OrderedDict
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Protocol

from fasteners import InterProcessReaderWriterLock
from vidur.entities.execution_time_predictor_request import (
    ExecutionTimePredictorRequest,
)

from .action_resolver import ControllerTransitionKind, ResolvedControllerAction
from .config import GV4EngineConfig
from .state import GV4State


TimingResult = tuple[tuple[float, ...], tuple[float, ...]]
RequestShape = tuple[int, int, bool]
BatchShape = tuple[RequestShape, ...]


@dataclass(frozen=True, slots=True)
class PrefillProfile:
    """In-memory end-to-end prefill estimates sampled on a fixed token grid."""

    path: Path
    step_tokens: int
    max_tokens: int
    end_to_end_seconds: tuple[float, ...]

    def estimate(self, tokens: int) -> float:
        if tokens <= 0:
            raise VidurTimingProviderError(
                "prefill estimate requires a positive token count"
            )
        if tokens > self.max_tokens:
            raise VidurTimingProviderError("prefill estimate exceeds profile range")

        # Non-grid residual chunks use the next measured point conservatively.
        row_index = (tokens + self.step_tokens - 1) // self.step_tokens - 1
        return self.end_to_end_seconds[row_index]


class VidurTimingProviderError(RuntimeError):
    """Raised when a GV4 batch cannot be represented by the Vidur profile."""


class _ExecutionTime(Protocol):
    @property
    def model_time(self) -> float: ...

    @property
    def pipeline_parallel_communication_time(self) -> float: ...


class VidurExecutionTimePredictor(Protocol):
    """The small part of Vidur's predictor API required by GV4."""

    def get_execution_time(
        self,
        requests: list[ExecutionTimePredictorRequest],
        pipeline_stage: int,
    ) -> _ExecutionTime: ...


def build_vidur_predictor(config: GV4EngineConfig) -> VidurExecutionTimePredictor:
    """Construct Vidur's predictor directly from the resolved GV4 config."""

    from vidur.config import (
        CacheConfig as VidurCacheConfig,
        RandomForestExecutionTimePredictorConfig,
        ReplicaConfig,
    )
    from vidur.execution_time_predictor import ExecutionTimePredictorRegistry

    settings = config.vidur_predictor
    predictor_config = RandomForestExecutionTimePredictorConfig(
        cache_dir=settings.cache_dir,
        cache_mode=settings.cache_mode,
        prediction_max_tokens_per_request=(settings.prediction_max_tokens_per_request),
        prediction_max_batch_size=settings.prediction_max_batch_size,
        prediction_max_prefill_chunk_size=(settings.prediction_max_prefill_chunk_size),
        kv_cache_prediction_granularity=(settings.kv_cache_prediction_granularity),
        prefill_chunk_size_prediction_granularity=(
            settings.prefill_chunk_size_prediction_granularity
        ),
        num_training_job_threads=settings.num_training_job_threads,
        skip_cpu_overhead_modeling=True,
    )
    replica_config = ReplicaConfig(
        model_name=config.model.model_id,
        device=settings.device,
        network_device=settings.network_device,
        tensor_parallel_size=config.topology.tensor_parallel_size,
        num_pipeline_stages=config.topology.pipeline_parallel_size,
    )
    cache_config = VidurCacheConfig(block_size=config.kv_cache.block_size_tokens)
    return ExecutionTimePredictorRegistry.get(
        predictor_config.get_type(),
        predictor_config=predictor_config,
        replica_config=replica_config,
        cache_config=cache_config,
    )


def prefill_profile_path(config: GV4EngineConfig) -> Path:
    """Return the deterministic profile path for this TP/PP hardware setup."""

    predictor = config.vidur_predictor

    def safe(value: str) -> str:
        component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
        if not component:
            raise VidurTimingProviderError("profile filename component is empty")
        return component

    topology = config.topology
    filename = (
        f"prefill_profile_TP{topology.tensor_parallel_size}"
        f"_PP{topology.pipeline_parallel_size}"
        f"_{safe(predictor.device)}_{safe(predictor.network_device)}.csv"
    )
    return Path(predictor.cache_dir) / filename


def _profile_columns(stage_count: int) -> tuple[str, ...]:
    columns = ["prefill_request_size_tokens"]
    for stage in range(stage_count):
        columns.append(f"stage_{stage}_computation_time_sec")
        if stage < stage_count - 1:
            columns.append(f"stage_{stage}_{stage + 1}_pp_communication_time_sec")
    columns.append("end_to_end_prefill_time_sec")
    return tuple(columns)


def _predict_stage_times(
    predictor: VidurExecutionTimePredictor,
    requests: list[ExecutionTimePredictorRequest],
    stage_count: int,
) -> TimingResult:
    """Split Vidur model time into TP-inclusive service and PP transfer."""

    stage_service: list[float] = []
    pp_communication: list[float] = []
    for stage in range(stage_count):
        execution = predictor.get_execution_time(requests, stage)
        model_seconds = float(execution.model_time)
        pp_seconds = float(execution.pipeline_parallel_communication_time) * 1e-3
        service_seconds = model_seconds - pp_seconds

        if not math.isfinite(service_seconds) or service_seconds <= 0.0:
            raise VidurTimingProviderError("Vidur returned invalid stage service time")
        if not math.isfinite(pp_seconds) or pp_seconds < 0.0:
            raise VidurTimingProviderError(
                "Vidur returned invalid PP communication time"
            )

        stage_service.append(service_seconds)
        if stage < stage_count - 1:
            pp_communication.append(pp_seconds)
        elif pp_seconds != 0.0:
            raise VidurTimingProviderError(
                "Vidur returned PP communication for the final stage"
            )

    return tuple(stage_service), tuple(pp_communication)


def _load_prefill_profile(
    config: GV4EngineConfig,
    path: Path,
) -> PrefillProfile | None:
    """Load a complete compatible CSV, or return None for regeneration."""

    stage_count = config.topology.pipeline_parallel_size
    step = config.vidur_predictor.prefill_profile_step_tokens
    maximum = config.request.max_prefill_tokens_per_request
    columns = _profile_columns(stage_count)
    expected_tokens = tuple(range(step, maximum + 1, step))

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != columns:
                return None
            rows = list(reader)
    except (OSError, csv.Error):
        return None

    if len(rows) != len(expected_tokens):
        return None

    totals: list[float] = []
    try:
        for expected, row in zip(expected_tokens, rows):
            if None in row or int(row[columns[0]]) != expected:
                return None

            component_total = 0.0
            for stage in range(stage_count):
                service = float(row[f"stage_{stage}_computation_time_sec"])
                if not math.isfinite(service) or service <= 0.0:
                    return None
                component_total += service
                if stage < stage_count - 1:
                    communication = float(
                        row[f"stage_{stage}_{stage + 1}" "_pp_communication_time_sec"]
                    )
                    if not math.isfinite(communication) or communication < 0.0:
                        return None
                    component_total += communication

            total = float(row["end_to_end_prefill_time_sec"])
            if (
                not math.isfinite(total)
                or total <= 0.0
                or not math.isclose(total, component_total, rel_tol=1e-9, abs_tol=1e-12)
            ):
                return None
            totals.append(total)
    except (KeyError, TypeError, ValueError):
        return None

    return PrefillProfile(path, step, maximum, tuple(totals))


def _generate_prefill_rows(
    config: GV4EngineConfig,
    predictor: VidurExecutionTimePredictor,
) -> list[dict[str, int | float]]:
    stage_count = config.topology.pipeline_parallel_size
    step = config.vidur_predictor.prefill_profile_step_tokens
    maximum = config.request.max_prefill_tokens_per_request
    rows: list[dict[str, int | float]] = []

    for tokens in range(step, maximum + 1, step):
        requests = [
            ExecutionTimePredictorRequest(
                num_processed_tokens=0,
                num_tokens_to_process=tokens,
                is_prefill_complete=False,
            )
        ]
        service, communication = _predict_stage_times(predictor, requests, stage_count)
        row: dict[str, int | float] = {"prefill_request_size_tokens": tokens}
        for stage, duration in enumerate(service):
            row[f"stage_{stage}_computation_time_sec"] = duration
            if stage < stage_count - 1:
                row[f"stage_{stage}_{stage + 1}_pp_communication_time_sec"] = (
                    communication[stage]
                )
        row["end_to_end_prefill_time_sec"] = sum(service) + sum(communication)
        rows.append(row)
    return rows


def _write_prefill_profile(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, int | float]],
) -> None:
    """Atomically publish a complete CSV so readers never see a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as error:
        raise VidurTimingProviderError(
            f"failed to write prefill profile {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ensure_prefill_profile(
    config: GV4EngineConfig,
    predictor: VidurExecutionTimePredictor,
) -> PrefillProfile:
    """Load the derived SLO profile, generating it once when absent or invalid."""

    path = prefill_profile_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = InterProcessReaderWriterLock(f"{path}.lock")

    with lock.read_lock():
        profile = _load_prefill_profile(config, path)
    if profile is not None:
        return profile

    with lock.write_lock():
        profile = _load_prefill_profile(config, path)
        if profile is not None:
            return profile
        rows = _generate_prefill_rows(config, predictor)
        _write_prefill_profile(
            path,
            _profile_columns(config.topology.pipeline_parallel_size),
            rows,
        )
        profile = _load_prefill_profile(config, path)
        if profile is None:
            raise VidurTimingProviderError(
                f"generated prefill profile failed validation: {path}"
            )
        return profile


class VidurTimingProvider:
    """Return stage-service and PP-boundary times for one resolved GV4 batch.

    The injected Vidur predictor owns its fitted models and memmap cache tables.
    This adapter only constructs Vidur request features and separates its timing:

    * stage service = GPU kernels + TP collectives
    * PP boundary = send/receive communication to the next stage

    Vidur reports ``model_time`` in seconds, with PP communication included, but
    reports ``pipeline_parallel_communication_time`` in milliseconds. Subtracting
    the converted PP value prevents the GV4 calendar from counting it twice.
    CPU overhead from ``ExecutionTime.total_time`` is deliberately never used.
    """

    __slots__ = (
        "_cache",
        "_config",
        "_max_cache_entries",
        "_prefill_profile",
        "_predictor",
        "_stage_count",
    )

    def __init__(
        self,
        config: GV4EngineConfig,
        predictor: VidurExecutionTimePredictor,
        *,
        prefill_profile: PrefillProfile | None = None,
        max_cache_entries: int = 65_536,
    ) -> None:
        if max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be positive")

        self._config = config
        self._predictor = predictor
        self._stage_count = config.topology.pipeline_parallel_size
        self._max_cache_entries = max_cache_entries
        self._prefill_profile = prefill_profile
        self._cache: OrderedDict[BatchShape, TimingResult] = OrderedDict()
        self._validate_predictor_topology()

    @classmethod
    def from_config(
        cls,
        config: GV4EngineConfig,
        *,
        max_cache_entries: int = 65_536,
    ) -> "VidurTimingProvider":
        predictor = build_vidur_predictor(config)
        return cls(
            config,
            predictor,
            prefill_profile=ensure_prefill_profile(config, predictor),
            max_cache_entries=max_cache_entries,
        )

    def __call__(
        self, state: GV4State, action: ResolvedControllerAction
    ) -> TimingResult:
        shape = self._batch_shape(state, action)
        return self._timing_for_shape(shape)

    def estimate_prefill_time(self, tokens: int) -> float:
        """Predict one fresh request's end-to-end prefill time in seconds."""

        if self._prefill_profile is not None:
            return self._prefill_profile.estimate(tokens)
        if tokens <= 0:
            raise VidurTimingProviderError("prefill estimate requires positive tokens")
        maximum = self._config.vidur_predictor.prediction_max_prefill_chunk_size
        if tokens > maximum:
            raise VidurTimingProviderError("prefill estimate exceeds predictor range")
        service, communication = self._timing_for_shape(((0, tokens, False),))
        return sum(service) + sum(communication)

    def _timing_for_shape(self, shape: BatchShape) -> TimingResult:
        cached = self._cache.get(shape)
        if cached is not None:
            self._cache.move_to_end(shape)
            return cached

        timing = self._predict(shape)
        self._cache[shape] = timing
        if len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)
        return timing

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def prefill_profile_file(self) -> Path | None:
        return None if self._prefill_profile is None else self._prefill_profile.path

    def _batch_shape(
        self, state: GV4State, action: ResolvedControllerAction
    ) -> BatchShape:
        if action.transition_kind != ControllerTransitionKind.BATCH:
            raise VidurTimingProviderError("timing requires a resolved batch action")
        if not action.allocations:
            raise VidurTimingProviderError("batch action has no allocations")

        predictor = self._config.vidur_predictor
        if len(action.allocations) > predictor.prediction_max_batch_size:
            raise VidurTimingProviderError("batch exceeds the profiled batch size")

        shape: list[RequestShape] = []
        prefill_square_sum = 0
        for allocation in action.allocations:
            request = state.request(allocation.request_id)
            if request.owner_replica_id != action.replica_id:
                raise VidurTimingProviderError(
                    "batch contains a request from another replica"
                )
            if (
                request.reserved_prefill_tokens
                or request.reserved_decode_tokens
                or request.reserved_recompute_tokens
            ):
                raise VidurTimingProviderError(
                    "in-flight request cannot enter another batch"
                )

            if allocation.recompute_tokens:
                if allocation.recompute_tokens > request.remaining_recompute_tokens:
                    raise VidurTimingProviderError(
                        "recompute allocation exceeds missing KV context"
                    )
                processed = request.kv_computed_tokens
                is_decode = False
            else:
                if request.remaining_recompute_tokens:
                    raise VidurTimingProviderError(
                        "new work requires fully reconstructed KV context"
                    )
                is_decode = allocation.decode_tokens > 0
                if is_decode != request.is_decode_phase:
                    raise VidurTimingProviderError(
                        "allocation phase disagrees with request progress"
                    )
                processed = request.logical_context_tokens

            shape.append((processed, allocation.total_tokens, is_decode))
            prefill_tokens = allocation.prefill_tokens + allocation.recompute_tokens
            prefill_square_sum += prefill_tokens**2

        max_chunk = predictor.prediction_max_prefill_chunk_size
        if prefill_square_sum > max_chunk * max_chunk:
            raise VidurTimingProviderError(
                "aggregate prefill chunk exceeds the profiled predictor boundary"
            )

        # Vidur's batch features are aggregate and therefore request-order invariant.
        return tuple(sorted(shape, key=lambda item: (item[2], item[0], item[1])))

    def _predict(self, shape: BatchShape) -> TimingResult:
        requests = [
            ExecutionTimePredictorRequest(
                num_processed_tokens=processed,
                num_tokens_to_process=tokens,
                is_prefill_complete=is_decode,
            )
            for processed, tokens, is_decode in shape
        ]

        return _predict_stage_times(self._predictor, requests, self._stage_count)

    def _validate_predictor_topology(self) -> None:
        """Catch a mismatched stock Vidur predictor before the first MCTS query."""

        replica = getattr(self._predictor, "_replica_config", None)
        if replica is None:  # Allows small protocol-compatible test predictors.
            return

        expected = self._config.topology
        actual_tp = getattr(replica, "tensor_parallel_size", None)
        actual_pp = getattr(replica, "num_pipeline_stages", None)
        if actual_tp != expected.tensor_parallel_size or actual_pp != self._stage_count:
            raise VidurTimingProviderError(
                "Vidur predictor TP/PP topology does not match the GV4 config"
            )

        layer_counts = {item.count for item in expected.stage_layer_ranges}
        predictor_layers = getattr(
            self._predictor, "_num_layers_per_pipeline_stage", None
        )
        if len(layer_counts) != 1 or predictor_layers != next(iter(layer_counts)):
            raise VidurTimingProviderError(
                "stock Vidur predictor requires the configured equal PP layer partition"
            )
