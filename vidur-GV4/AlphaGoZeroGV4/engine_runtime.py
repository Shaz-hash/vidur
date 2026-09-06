"""One small runtime contract over the Python and native GV4 engines.

AlphaGoZero orchestration consumes the records in this module instead of
inspecting Python or pybind state objects. Engine-specific conversion remains
inside the two adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import importlib
import math
from pathlib import Path
import random
import sys
from typing import Any, Literal, Protocol

from GV4_Engine.config import GV4EngineConfig
from GV4_Engine.dnn_inference.dnn_features import GV4FeatureBuilder
from GV4_Engine.dnn_inference.inference import GV4DNNInference
from GV4_Engine.state import Player
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment


PlayerName = Literal["adversary", "controller"]
BackendName = Literal["python", "native"]
FloatMatrix = tuple[tuple[float, ...], ...]


__all__ = [
    "ActionFeatureSnapshot",
    "AppliedEdge",
    "BatchAllocationSummary",
    "CanonicalActionRef",
    "EngineMetadata",
    "EngineRuntime",
    "LaunchSummary",
    "MicrobatchSummary",
    "NativeEngineRuntime",
    "ObjectiveSummary",
    "PythonEngineRuntime",
    "ReplicaSummary",
    "RequestSummary",
    "RootActionStats",
    "SearchConfig",
    "SearchContext",
    "SearchResult",
    "StateFeatureSnapshot",
    "StateSummary",
    "create_engine_runtime",
]


class EngineRuntimeError(RuntimeError):
    """Raised when one backend cannot satisfy the shared runtime contract."""


@dataclass(frozen=True, slots=True)
class EngineMetadata:
    """Immutable identities attached to replay, models, and evaluation logs."""

    backend: BackendName
    config_manifest_sha256: str
    manifest_schema_version: str
    state_schema_version: str
    action_schema_version: str
    feature_schema_version: str
    native_layout_version: str
    time_epsilon: float


@dataclass(frozen=True, slots=True)
class LaunchSummary:
    launch_time: float
    request_count: int
    prefill_tokens: int


@dataclass(frozen=True, slots=True)
class RequestSummary:
    """Logger-facing request data with no Python/native state dependency."""

    request_id: int
    owner_replica_id: int
    lifecycle: str
    arrival_time: float
    prefill_deadline: float
    decode_token_slo_sec: float
    original_prefill_tokens: int
    original_decode_tokens: int
    committed_prefill_tokens: int
    reserved_prefill_tokens: int
    committed_decode_tokens: int
    reserved_decode_tokens: int
    logical_context_tokens: int
    kv_computed_tokens: int
    reserved_recompute_tokens: int
    remaining_recompute_tokens: int
    committed_kv_blocks: int
    reserved_kv_blocks: int
    inflight_microbatch_id: int
    next_decode_deadline: float
    prefill_lateness_sec: float
    decode_lateness_sec: float
    violation_recorded: bool
    terminal_reason: str
    terminal_requested_at: float
    terminal_time: float


@dataclass(frozen=True, slots=True)
class BatchAllocationSummary:
    request_id: int
    prefill_tokens: int
    decode_tokens: int
    recompute_tokens: int
    new_kv_blocks: int


@dataclass(frozen=True, slots=True)
class MicrobatchSummary:
    microbatch_id: int
    replica_id: int
    raw_action_index: int
    canonical_action_index: int
    allocations: tuple[BatchAllocationSummary, ...]
    stage_ready_times: tuple[float, ...]
    stage_start_times: tuple[float, ...]
    stage_finish_times: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReplicaSummary:
    replica_id: int
    rank_ids: tuple[int, ...]
    rank_kv_capacity_blocks: tuple[int, ...]
    rank_kv_committed_blocks: tuple[int, ...]
    rank_kv_reserved_blocks: tuple[int, ...]
    rank_kv_free_blocks: tuple[int, ...]
    stage_tail_finish_times: tuple[float, ...]
    stage_last_microbatch_ids: tuple[int, ...]
    inflight_microbatches: tuple[MicrobatchSummary, ...]


@dataclass(frozen=True, slots=True)
class ObjectiveSummary:
    requests_generated: int
    requests_completed: int
    requests_stopped: int
    requests_dropped: int
    slo_violations: int
    prefill_lateness_sec: float
    decode_lateness_sec: float
    terminal_cost: float
    total_cost: float


@dataclass(frozen=True, slots=True)
class StateSummary:
    """Complete read-only logging view of either backend's state."""

    state_schema_version: str
    config_manifest_sha256: str
    now: float
    next_player: PlayerName
    next_adversary_tick: float
    next_request_id: int
    next_microbatch_id: int
    decode_credits_available: int
    decode_credits_reserved: int
    decode_credits_minted_total: int
    decode_tokens_committed_total: int
    launch_history: tuple[LaunchSummary, ...]
    requests: tuple[RequestSummary, ...]
    replicas: tuple[ReplicaSummary, ...]
    objective: ObjectiveSummary


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Backend-independent MCTS and optional rollout settings."""

    iterations: int = 100
    puct_c: float = 1.0
    use_policy_prior: bool = False
    use_model_bootstrap: bool = False
    policy_prior_temperature: float = 1.0
    prior_min_probability: float = 1e-8
    root_dirichlet_alpha: float = 0.0
    root_dirichlet_epsilon: float = 0.0
    root_dirichlet_total_concentration: float = 0.0
    rollout_count: int = 0
    rollout_horizon_sec: float = 0.4
    rollout_seed: int = 0
    rollout_policy_temperature: float = 1.0
    rollout_probability_quantum: float = 1e-6
    rollout_max_actions: int = 4096

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("MCTS iterations must be positive")
        if not math.isfinite(self.puct_c) or self.puct_c <= 0.0:
            raise ValueError("puct_c must be positive and finite")
        if (
            not math.isfinite(self.policy_prior_temperature)
            or self.policy_prior_temperature <= 0.0
        ):
            raise ValueError("policy_prior_temperature must be positive")
        if (
            not math.isfinite(self.prior_min_probability)
            or self.prior_min_probability < 0.0
        ):
            raise ValueError("prior_min_probability must be nonnegative")
        if not 0.0 <= self.root_dirichlet_epsilon <= 1.0:
            raise ValueError("root_dirichlet_epsilon must be in [0, 1]")
        if self.root_dirichlet_alpha < 0.0:
            raise ValueError("root_dirichlet_alpha must be nonnegative")
        if self.root_dirichlet_total_concentration < 0.0:
            raise ValueError("root_dirichlet_total_concentration must be nonnegative")
        if self.rollout_count < 0:
            raise ValueError("rollout_count cannot be negative")
        if (
            not math.isfinite(self.rollout_horizon_sec)
            or self.rollout_horizon_sec < 0.0
        ):
            raise ValueError("rollout_horizon_sec must be nonnegative")
        if self.rollout_seed < 0:
            raise ValueError("rollout_seed must be nonnegative")
        if (
            not math.isfinite(self.rollout_policy_temperature)
            or self.rollout_policy_temperature <= 0.0
        ):
            raise ValueError("rollout_policy_temperature must be positive")
        if (
            not math.isfinite(self.rollout_probability_quantum)
            or self.rollout_probability_quantum < 0.0
        ):
            raise ValueError("rollout_probability_quantum must be nonnegative")
        if self.rollout_max_actions <= 0:
            raise ValueError("rollout_max_actions must be positive")


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Identifiers that affect logging but not game semantics."""

    game_id: int
    root_id: int
    root_depth: int
    root_node_id: int = 0
    model_version: int = 0
    cycle_label: str = "self_play"


@dataclass(frozen=True, slots=True)
class CanonicalActionRef:
    """Backend-neutral identity plus the private engine action object."""

    backend: BackendName
    player: PlayerName
    canonical_action_index: int
    representative_raw_index: int
    equivalent_raw_indices: tuple[int, ...]
    payload: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class StateFeatureSnapshot:
    """Serializable form of the structured GV4 state features."""

    schema_version: str
    config_manifest_sha256: str
    global_features: tuple[float, ...]
    request_rows: FloatMatrix
    request_replica_offsets: tuple[int, ...]
    launch_rows: FloatMatrix
    replica_rows: FloatMatrix
    microbatch_rows: FloatMatrix
    microbatch_replica_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionFeatureSnapshot:
    """Serializable form of one canonical action feature object."""

    header: tuple[float, ...]
    affected_request_rows: FloatMatrix


@dataclass(frozen=True, slots=True)
class RootActionStats:
    """MCTS statistics and features for one canonical root edge."""

    action: CanonicalActionRef
    visits: int
    value_sum: float
    mean_value: float
    prior: float | None
    features: ActionFeatureSnapshot


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The complete backend-neutral output of one root search."""

    root_node_id: int
    root_player: PlayerName
    next_player: PlayerName
    best_action: CanonicalActionRef | None
    best_action_value: float
    root_value: float
    action_stats: tuple[RootActionStats, ...]
    raw_valid_mask: tuple[bool, ...]
    state_features: StateFeatureSnapshot
    used_bootstrap: bool
    used_rollout: bool
    diagnostics: dict[str, Any]

    def stats_for(self, action: CanonicalActionRef) -> RootActionStats:
        for item in self.action_stats:
            if item.action.representative_raw_index == action.representative_raw_index:
                return item
        raise KeyError(
            f"root action {action.representative_raw_index} is not in the search"
        )


@dataclass(frozen=True, slots=True)
class AppliedEdge:
    """One selected action and its exact controller-valued Bellman edge."""

    state: Any = field(repr=False, compare=False)
    action: CanonicalActionRef
    player: PlayerName
    next_player: PlayerName
    started_at: float
    finished_at: float
    objective_before: float
    objective_after: float
    reward: float
    discount: float
    transition_kind: str


class EngineRuntime(Protocol):
    backend: BackendName
    config: GV4EngineConfig
    search_config: SearchConfig

    @property
    def metadata(self) -> EngineMetadata: ...

    def initial_state(self) -> Any: ...

    def clone_state(self, state: Any) -> Any: ...

    def state_time(self, state: Any) -> float: ...

    def objective_cost(self, state: Any) -> float: ...

    def player_to_move(self, state: Any) -> PlayerName: ...

    def summarize_state(self, state: Any) -> StateSummary: ...

    def canonical_actions(self, state: Any) -> tuple[CanonicalActionRef, ...]: ...

    def apply_action(self, state: Any, action: CanonicalActionRef) -> AppliedEdge: ...

    def search(self, state: Any, context: SearchContext) -> SearchResult: ...

    def bootstrap_value(self, state: Any) -> float: ...

    def close(self) -> None: ...


def _player_name(value: Any) -> PlayerName:
    text = value if isinstance(value, str) else getattr(value, "name", str(value))
    name = str(text).rsplit(".", 1)[-1].lower()
    if name not in {"adversary", "controller"}:
        raise EngineRuntimeError(f"unsupported GV4 player {value!r}")
    return name  # type: ignore[return-value]


def _enum_name(value: Any) -> str:
    """Normalize Python and pybind enum values to one uppercase spelling."""

    text = value if isinstance(value, str) else getattr(value, "name", str(value))
    return str(text).rsplit(".", 1)[-1].upper()


def _request_summary(request: Any) -> RequestSummary:
    return RequestSummary(
        request_id=int(request.request_id),
        owner_replica_id=int(request.owner_replica_id),
        lifecycle=_enum_name(request.lifecycle),
        arrival_time=float(request.arrival_time),
        prefill_deadline=float(request.prefill_deadline),
        decode_token_slo_sec=float(request.decode_token_slo_sec),
        original_prefill_tokens=int(request.original_prefill_tokens),
        original_decode_tokens=int(request.original_decode_tokens),
        committed_prefill_tokens=int(request.committed_prefill_tokens),
        reserved_prefill_tokens=int(request.reserved_prefill_tokens),
        committed_decode_tokens=int(request.committed_decode_tokens),
        reserved_decode_tokens=int(request.reserved_decode_tokens),
        logical_context_tokens=int(request.logical_context_tokens),
        kv_computed_tokens=int(request.kv_computed_tokens),
        reserved_recompute_tokens=int(request.reserved_recompute_tokens),
        remaining_recompute_tokens=int(request.remaining_recompute_tokens),
        committed_kv_blocks=int(request.committed_kv_blocks),
        reserved_kv_blocks=int(request.reserved_kv_blocks),
        inflight_microbatch_id=int(request.inflight_microbatch_id),
        next_decode_deadline=float(request.next_decode_deadline),
        prefill_lateness_sec=float(request.prefill_lateness_sec),
        decode_lateness_sec=float(request.decode_lateness_sec),
        violation_recorded=bool(request.violation_recorded),
        terminal_reason=_enum_name(request.terminal_reason),
        terminal_requested_at=float(request.terminal_requested_at),
        terminal_time=float(request.terminal_time),
    )


def _microbatch_summary(batch: Any) -> MicrobatchSummary:
    allocations = tuple(
        BatchAllocationSummary(
            request_id=int(item.request_id),
            prefill_tokens=int(item.prefill_tokens),
            decode_tokens=int(item.decode_tokens),
            recompute_tokens=int(item.recompute_tokens),
            new_kv_blocks=int(item.new_kv_blocks),
        )
        for item in batch.allocations
    )
    return MicrobatchSummary(
        microbatch_id=int(batch.microbatch_id),
        replica_id=int(batch.replica_id),
        raw_action_index=int(batch.raw_action_index),
        canonical_action_index=int(batch.canonical_action_index),
        allocations=allocations,
        stage_ready_times=tuple(float(value) for value in batch.stage_ready_times),
        stage_start_times=tuple(float(value) for value in batch.stage_start_times),
        stage_finish_times=tuple(float(value) for value in batch.stage_finish_times),
    )


def _replica_summary(replica: Any) -> ReplicaSummary:
    capacities = tuple(int(value) for value in replica.rank_kv_capacity_blocks)
    committed = tuple(int(value) for value in replica.rank_kv_committed_blocks)
    reserved = tuple(int(value) for value in replica.rank_kv_reserved_blocks)
    if not (len(capacities) == len(committed) == len(reserved)):
        raise EngineRuntimeError("replica KV arrays have different lengths")
    return ReplicaSummary(
        replica_id=int(replica.replica_id),
        rank_ids=tuple(int(value) for value in replica.rank_ids),
        rank_kv_capacity_blocks=capacities,
        rank_kv_committed_blocks=committed,
        rank_kv_reserved_blocks=reserved,
        rank_kv_free_blocks=tuple(
            capacity - used - pending
            for capacity, used, pending in zip(capacities, committed, reserved)
        ),
        stage_tail_finish_times=tuple(
            float(value) for value in replica.stage_tail_finish_times
        ),
        stage_last_microbatch_ids=tuple(
            int(value) for value in replica.stage_last_microbatch_ids
        ),
        inflight_microbatches=tuple(
            _microbatch_summary(batch) for batch in replica.inflight_microbatches
        ),
    )


def _state_replicas(state: Any) -> tuple[Any, ...]:
    # Python stores a list; the current one-replica native layout stores one field.
    if hasattr(state, "replicas"):
        return tuple(state.replicas)
    if hasattr(state, "replica"):
        return (state.replica,)
    raise EngineRuntimeError("engine state exposes no replica data")


def _objective_summary(objective: Any) -> ObjectiveSummary:
    return ObjectiveSummary(
        requests_generated=int(objective.requests_generated),
        requests_completed=int(objective.requests_completed),
        requests_stopped=int(objective.requests_stopped),
        requests_dropped=int(objective.requests_dropped),
        slo_violations=int(objective.slo_violations),
        prefill_lateness_sec=float(objective.prefill_lateness_sec),
        decode_lateness_sec=float(objective.decode_lateness_sec),
        terminal_cost=float(objective.terminal_cost),
        total_cost=float(objective.total_cost),
    )


def _finite_vector(values: Any, label: str) -> tuple[float, ...]:
    raw = values.tolist() if hasattr(values, "tolist") else list(values)
    result = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in result):
        raise EngineRuntimeError(f"{label} contains a non-finite value")
    return result


def _finite_matrix(values: Any, label: str) -> FloatMatrix:
    if all(hasattr(values, name) for name in ("values", "rows", "columns")):
        row_count = int(values.rows)
        width = int(values.columns)
        if row_count < 0 or width < 0:
            raise EngineRuntimeError(f"{label} has negative native dimensions")
        flat = _finite_vector(values.values, label)
        if len(flat) != row_count * width:
            raise EngineRuntimeError(f"{label} has inconsistent native dimensions")
        if width == 0:
            return tuple(() for _ in range(row_count))
        return tuple(
            tuple(flat[row * width : (row + 1) * width]) for row in range(row_count)
        )

    raw = values.tolist() if hasattr(values, "tolist") else list(values)
    rows = tuple(tuple(float(value) for value in row) for row in raw)
    if not all(math.isfinite(value) for row in rows for value in row):
        raise EngineRuntimeError(f"{label} contains a non-finite value")
    return rows


def _state_features(features: Any) -> StateFeatureSnapshot:
    return StateFeatureSnapshot(
        schema_version=str(features.schema_version),
        config_manifest_sha256=str(features.config_manifest_sha256),
        global_features=_finite_vector(features.global_features, "global features"),
        request_rows=_finite_matrix(features.request_rows, "request features"),
        request_replica_offsets=tuple(
            int(value) for value in features.request_replica_offsets
        ),
        launch_rows=_finite_matrix(features.launch_rows, "launch features"),
        replica_rows=_finite_matrix(features.replica_rows, "replica features"),
        microbatch_rows=_finite_matrix(features.microbatch_rows, "microbatch features"),
        microbatch_replica_offsets=tuple(
            int(value) for value in features.microbatch_replica_offsets
        ),
    )


def _action_features(features: Any) -> ActionFeatureSnapshot:
    return ActionFeatureSnapshot(
        header=_finite_vector(features.header, "action header"),
        affected_request_rows=_finite_matrix(
            features.affected_request_rows, "affected-request features"
        ),
    )


def _diagnostics(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


class _BaseEngineRuntime:
    backend: BackendName

    def __init__(self, config: GV4EngineConfig, search_config: SearchConfig) -> None:
        config.validate()
        if config.topology.num_replicas != 1:
            raise NotImplementedError("GV4 AlphaGoZero currently supports one replica")
        self.config = config
        self.search_config = search_config
        layout = config.layout
        self._metadata = EngineMetadata(
            backend=self.backend,
            config_manifest_sha256=config.manifest_sha256(),
            manifest_schema_version=layout.manifest_schema_version,
            state_schema_version=layout.state_schema_version,
            action_schema_version=layout.action_schema_version,
            feature_schema_version=layout.feature_schema_version,
            native_layout_version=layout.native_layout_version,
            time_epsilon=float(config.timing.epsilon),
        )

    @property
    def metadata(self) -> EngineMetadata:
        return self._metadata

    def clone_state(self, state: Any) -> Any:
        return state.clone()

    def state_time(self, state: Any) -> float:
        return float(state.now)

    def objective_cost(self, state: Any) -> float:
        return float(state.objective.total_cost)

    def player_to_move(self, state: Any) -> PlayerName:
        return _player_name(state.next_player)

    def summarize_state(self, state: Any) -> StateSummary:
        """Copy state once for logs without exposing a mutable engine object."""

        return StateSummary(
            state_schema_version=str(state.state_schema_version),
            config_manifest_sha256=str(state.config_manifest_sha256),
            now=float(state.now),
            next_player=self.player_to_move(state),
            next_adversary_tick=float(state.next_adversary_tick),
            next_request_id=int(state.next_request_id),
            next_microbatch_id=int(state.next_microbatch_id),
            decode_credits_available=int(state.decode_credits_available),
            decode_credits_reserved=int(state.decode_credits_reserved),
            decode_credits_minted_total=int(state.decode_credits_minted_total),
            decode_tokens_committed_total=int(state.decode_tokens_committed_total),
            launch_history=tuple(
                LaunchSummary(
                    launch_time=float(item.launch_time),
                    request_count=int(item.request_count),
                    prefill_tokens=int(item.prefill_tokens),
                )
                for item in state.launch_history
            ),
            requests=tuple(_request_summary(item) for item in state.requests),
            replicas=tuple(_replica_summary(item) for item in _state_replicas(state)),
            objective=_objective_summary(state.objective),
        )

    def _action_ref(self, payload: Any, player: PlayerName) -> CanonicalActionRef:
        aliases = tuple(int(value) for value in payload.equivalent_raw_indices)
        return CanonicalActionRef(
            backend=self.backend,
            player=player,
            canonical_action_index=int(payload.canonical_action_index),
            representative_raw_index=int(payload.representative_raw_index),
            equivalent_raw_indices=aliases,
            payload=payload,
        )

    def _finish_edge(
        self,
        before: Any,
        after: Any,
        action: CanonicalActionRef,
        transition_kind: str,
    ) -> AppliedEdge:
        started_at = self.state_time(before)
        finished_at = self.state_time(after)
        if not math.isfinite(started_at) or not math.isfinite(finished_at):
            raise EngineRuntimeError("engine action produced a non-finite time")
        if finished_at + self.config.timing.epsilon < started_at:
            raise EngineRuntimeError("engine action moved simulation time backwards")
        objective_before = self.objective_cost(before)
        objective_after = self.objective_cost(after)
        if not math.isfinite(objective_before) or not math.isfinite(objective_after):
            raise EngineRuntimeError("engine action produced a non-finite objective")
        elapsed = max(0.0, finished_at - started_at)
        discount = float(self.config.reward.discount_for_elapsed(elapsed))
        if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise EngineRuntimeError("engine action produced an invalid discount")
        return AppliedEdge(
            state=after,
            action=action,
            player=action.player,
            next_player=self.player_to_move(after),
            started_at=started_at,
            finished_at=finished_at,
            objective_before=objective_before,
            objective_after=objective_after,
            reward=objective_before - objective_after,
            discount=discount,
            transition_kind=transition_kind,
        )

    def _check_action(self, state: Any, action: CanonicalActionRef) -> None:
        if action.backend != self.backend:
            raise EngineRuntimeError("canonical action belongs to another backend")
        if action.player != self.player_to_move(state):
            raise EngineRuntimeError("canonical action does not match the state turn")


def _import_python_mcts(name: str) -> Any:
    # The historical folder contains a hyphen and therefore needs importlib.
    repository_root = Path(__file__).resolve().parents[2]
    root_text = str(repository_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(f"vidur-GV4.{name}")


class PythonEngineRuntime(_BaseEngineRuntime):
    """Adapter over the Python reference engine and MCTS."""

    backend: BackendName = "python"

    def __init__(
        self,
        config: GV4EngineConfig,
        *,
        timing_provider: Any,
        search_config: SearchConfig,
        seed: int,
        inference: GV4DNNInference | None = None,
    ) -> None:
        super().__init__(config, search_config)
        self.environment = GV4VirtualVidurMCTSEnvironment(
            config,
            batch_timing_provider=timing_provider,
            prefill_time_estimator=timing_provider.estimate_prefill_time,
        )
        self.inference = inference
        self.feature_builder = GV4FeatureBuilder(config)

        base = _import_python_mcts("mcts_value_prior")
        mcts_config = base.MCTSConfig()
        mcts_config.mcts_iterations = search_config.iterations
        mcts_config.puct_c = search_config.puct_c
        mcts_config.use_policy_prior = search_config.use_policy_prior
        mcts_config.policy_prior_temperature = search_config.policy_prior_temperature
        mcts_config.prior_min_prob = search_config.prior_min_probability
        mcts_config.root_dirichlet_alpha = search_config.root_dirichlet_alpha
        mcts_config.root_dirichlet_epsilon = search_config.root_dirichlet_epsilon
        mcts_config.root_dirichlet_total_concentration = (
            search_config.root_dirichlet_total_concentration
        )
        mcts_config.rng = random.Random(seed)
        mcts_config.rollout_count = search_config.rollout_count
        mcts_config.rollout_horizon_sec = search_config.rollout_horizon_sec
        mcts_config.rollout_seed = search_config.rollout_seed
        mcts_config.rollout_policy_temperature = (
            search_config.rollout_policy_temperature
        )
        mcts_config.rollout_probability_quantum = (
            search_config.rollout_probability_quantum
        )
        mcts_config.rollout_max_actions = search_config.rollout_max_actions

        mcts_class = base.VidurMCTS
        if search_config.rollout_count:
            rollout = _import_python_mcts("mcts_value_prior_rollout")
            mcts_class = rollout.VidurMCTSPolicyRollout
        self._mcts = mcts_class(self.environment, mcts_config)

    def initial_state(self) -> Any:
        return self.environment.initial_state(now=0.0, next_player=Player.ADVERSARY)

    def canonical_actions(self, state: Any) -> tuple[CanonicalActionRef, ...]:
        player = self.player_to_move(state)
        if player == "controller":
            by_raw, _ = self.environment.sample_controller_actions(state, replica_id=0)
        else:
            by_raw, _ = self.environment.sample_adversary_actions(state)

        unique: dict[int, CanonicalActionRef] = {}
        for payload in by_raw:
            if payload is None:
                continue
            representative = int(payload.representative_raw_index)
            unique.setdefault(representative, self._action_ref(payload, player))
        return tuple(unique[index] for index in sorted(unique))

    def apply_action(self, state: Any, action: CanonicalActionRef) -> AppliedEdge:
        self._check_action(state, action)
        if action.player == "controller":
            result = self.environment.apply_controller_action_only(
                state, action.payload, inplace=False, fast_forward=True
            )
            kind = action.payload.action.transition_kind.name
        else:
            forced = state.now + self.config.timing.epsilon < state.next_adversary_tick
            result = self.environment.apply_adversary_action_only(
                state, action.payload, inplace=False
            )
            kind = "FORCED_NOOP" if forced else "ADVERSARY"
        return self._finish_edge(state, result, action, kind)

    def _feature_for_action(
        self, state: Any, action: CanonicalActionRef
    ) -> ActionFeatureSnapshot:
        if action.player == "controller":
            features = self.feature_builder.build_controller_action(
                state, action.payload
            )
        else:
            features = self.feature_builder.build_adversary_action(
                state, action.payload
            )
        return _action_features(features)

    def search(self, state: Any, context: SearchContext) -> SearchResult:
        actions = self.canonical_actions(state)
        result = self._mcts.search_dnn(
            self.inference,
            state,
            self.player_to_move(state),
            game_id=context.game_id,
            root_id=context.root_id,
            root_node_id_override=context.root_node_id,
            root_depth=context.root_depth,
            mcts_iter=self.search_config.iterations,
            model_version=context.model_version,
            use_model_bootstrap=self.search_config.use_model_bootstrap,
            cycle_label=context.cycle_label,
        )
        root = self._mcts._root
        if root is None:
            raise EngineRuntimeError("Python MCTS did not retain its root")

        stats: list[RootActionStats] = []
        for action in actions:
            child = root.children.get(action.representative_raw_index)
            visits = 0 if child is None else int(child.visits)
            value_sum = 0.0 if child is None else float(child.value_sum)
            mean_value = 0.0 if child is None else float(child.mean_value())
            stats.append(
                RootActionStats(
                    action=action,
                    visits=visits,
                    value_sum=value_sum,
                    mean_value=mean_value,
                    prior=float(
                        root.action_priors.get(action.representative_raw_index, 0.0)
                    ),
                    features=self._feature_for_action(state, action),
                )
            )

        by_representative = {
            action.representative_raw_index: action for action in actions
        }
        best_action = (
            None
            if result.best_action_index is None
            else by_representative.get(int(result.best_action_index))
        )
        rollout_stats = getattr(self._mcts, "rollout_stats", None)
        return SearchResult(
            root_node_id=int(result.root_node_id),
            root_player=self.player_to_move(state),
            next_player=_player_name(result.next_player),
            best_action=best_action,
            best_action_value=float(result.best_action_value),
            root_value=float(root.mean_value()),
            action_stats=tuple(stats),
            raw_valid_mask=tuple(bool(value) for value in result.valid_mask),
            state_features=_state_features(self.feature_builder.build_state(state)),
            used_bootstrap=bool(result.used_bootstrap),
            used_rollout=bool(
                self.search_config.rollout_count
                and self.search_config.rollout_horizon_sec > 0.0
            ),
            diagnostics={"rollout": _diagnostics(rollout_stats)},
        )

    def bootstrap_value(self, state: Any) -> float:
        if self.inference is None:
            raise EngineRuntimeError("Python model bootstrap has no inference runtime")
        return float(
            self.inference.predict_value(
                state,
                player=self.player_to_move(state),
            )
        )

    def close(self) -> None:
        self._mcts.close()


class NativeEngineRuntime(_BaseEngineRuntime):
    """Adapter over the C++ engine, feature builder, and MCTS binding."""

    backend: BackendName = "native"

    def __init__(
        self,
        config: GV4EngineConfig,
        *,
        timing_provider: Any,
        search_config: SearchConfig,
        inference: Any | None = None,
    ) -> None:
        super().__init__(config, search_config)
        if search_config.root_dirichlet_epsilon:
            raise NotImplementedError(
                "native GV4 root Dirichlet noise is not bound yet"
            )

        package_root = Path(__file__).resolve().parents[1]
        package_text = str(package_root)
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
        native_package = importlib.import_module("GV4_Cpp")
        native_module = importlib.import_module("GV4_Cpp.gv4_native")

        self._native = native_module
        self.environment = native_package.environment_from_python(
            config, timing_provider
        )
        self.inference = inference
        native_config = native_package.config_from_python(config)
        self.feature_builder = native_module.FeatureBuilder(native_config)

    def initial_state(self) -> Any:
        return self.environment.initial_state(0.0, self._native.Player.ADVERSARY)

    def canonical_actions(self, state: Any) -> tuple[CanonicalActionRef, ...]:
        player = self.player_to_move(state)
        if player == "controller":
            payloads = self.environment.sample_controller_actions(
                state
            ).canonical_actions
        else:
            payloads = self.environment.sample_adversary_actions(
                state
            ).canonical_actions
        actions = tuple(self._action_ref(payload, player) for payload in payloads)
        return tuple(sorted(actions, key=lambda item: item.representative_raw_index))

    def apply_action(self, state: Any, action: CanonicalActionRef) -> AppliedEdge:
        self._check_action(state, action)
        if action.player == "controller":
            result = self.environment.apply_controller_action_only(
                state, action.payload, True
            )
            kind = str(action.payload.action.transition_kind).rsplit(".", 1)[-1]
        else:
            forced = state.now + self.config.timing.epsilon < state.next_adversary_tick
            result = self.environment.apply_adversary_action_only(state, action.payload)
            kind = "FORCED_NOOP" if forced else "ADVERSARY"
        return self._finish_edge(state, result, action, kind.upper())

    def _feature_for_action(
        self, state: Any, action: CanonicalActionRef
    ) -> ActionFeatureSnapshot:
        if action.player == "controller":
            features = self.feature_builder.build_controller_action(
                state, action.payload
            )
        else:
            features = self.feature_builder.build_adversary_action(
                state, action.payload
            )
        return _action_features(features)

    def search(self, state: Any, context: SearchContext) -> SearchResult:
        actions = self.canonical_actions(state)
        result = dict(
            self._native.run_policy_rollout_mcts(
                self.environment,
                state,
                state.next_player,
                inference=self.inference,
                iterations=self.search_config.iterations,
                puct_c=self.search_config.puct_c,
                policy_prior_temperature=(self.search_config.policy_prior_temperature),
                prior_min_probability=self.search_config.prior_min_probability,
                rollout_count=self.search_config.rollout_count,
                rollout_horizon_sec=self.search_config.rollout_horizon_sec,
                rollout_seed=self.search_config.rollout_seed,
                rollout_policy_temperature=(
                    self.search_config.rollout_policy_temperature
                ),
                rollout_probability_quantum=(
                    self.search_config.rollout_probability_quantum
                ),
                rollout_max_actions=self.search_config.rollout_max_actions,
                use_policy_prior=self.search_config.use_policy_prior,
                use_model_bootstrap=self.search_config.use_model_bootstrap,
                root_node_id=context.root_node_id,
                root_depth=context.root_depth,
            )
        )
        raw_stats = {
            int(row["representative_raw_index"]): row
            for row in result["root_action_stats"]
        }
        uniform_prior = 1.0 / len(actions) if actions else 0.0
        stats: list[RootActionStats] = []
        for action in actions:
            row = raw_stats.get(action.representative_raw_index)
            visits = 0 if row is None else int(row["visits"])
            value_sum = 0.0 if row is None else float(row["value_sum"])
            mean_value = 0.0 if row is None else float(row["mean_value"])
            stats.append(
                RootActionStats(
                    action=action,
                    visits=visits,
                    value_sum=value_sum,
                    mean_value=mean_value,
                    prior=(
                        None if self.search_config.use_policy_prior else uniform_prior
                    ),
                    features=self._feature_for_action(state, action),
                )
            )

        by_representative = {
            action.representative_raw_index: action for action in actions
        }
        best_index = int(result["best_action_index"])
        best_action = None if best_index < 0 else by_representative.get(best_index)
        total_visits = sum(item.visits for item in stats)
        root_value = (
            sum(item.value_sum for item in stats) / total_visits
            if total_visits
            else 0.0
        )
        return SearchResult(
            root_node_id=int(result["root_node_id"]),
            root_player=_player_name(result["root_player"]),
            next_player=_player_name(result["next_player"]),
            best_action=best_action,
            best_action_value=float(result["best_action_value"]),
            root_value=float(root_value),
            action_stats=tuple(stats),
            raw_valid_mask=tuple(bool(value) for value in result["valid_mask"]),
            state_features=_state_features(self.feature_builder.build_state(state)),
            used_bootstrap=bool(result["used_bootstrap"]),
            used_rollout=bool(result["used_rollout"]),
            diagnostics={"rollout": dict(result["rollout_stats"])},
        )

    def bootstrap_value(self, state: Any) -> float:
        if self.inference is None:
            raise EngineRuntimeError("native model bootstrap has no inference runtime")
        return float(self.inference.predict_value(state, state.next_player))

    def close(self) -> None:
        return None


def create_engine_runtime(
    backend: BackendName,
    config: GV4EngineConfig,
    *,
    search_config: SearchConfig,
    seed: int,
    timing_provider: Any | None = None,
    python_inference: GV4DNNInference | None = None,
    native_inference: Any | None = None,
) -> EngineRuntime:
    """Construct one validated runtime; the caller thereafter stays backend-blind."""

    timing = timing_provider or config.create_vidur_timing_provider()
    if backend == "python":
        return PythonEngineRuntime(
            config,
            timing_provider=timing,
            search_config=search_config,
            seed=seed,
            inference=python_inference,
        )
    if backend == "native":
        return NativeEngineRuntime(
            config,
            timing_provider=timing,
            search_config=search_config,
            inference=native_inference,
        )
    raise ValueError(f"unsupported GV4 backend {backend!r}")
