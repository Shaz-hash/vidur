"""Strict readers for immutable GV4 self-play replay partitions.

The game recorder writes variable-size structured features.  This module keeps
that structure intact and validates it before training sees a sample.  It does
not know how requests, actions, or pipeline stages are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np

from GV4_Engine.config import GV4EngineConfig

from ..dnn_models import GV4ModelSpec
from ..engine_runtime import ActionFeatureSnapshot, StateFeatureSnapshot
from ..model_bundle import sha256_file
from ..replay_runtime import REPLAY_SCHEMA_VERSION


Role = Literal["controller", "adversary"]
RootKey = tuple[int, str, int]

__all__ = [
    "ReplayActionSample",
    "ReplayDatasetError",
    "ReplayPartition",
    "ReplayRootSample",
    "RoleTrainingData",
    "ValidatedReplayManifest",
    "discover_replay_manifests",
    "load_replay_partition",
    "load_root_at",
    "load_roots_at",
    "materialize_training_data",
    "replay_root_key",
    "validate_replay_manifest",
]


class ReplayDatasetError(ValueError):
    """Raised when replay cannot be interpreted unambiguously."""


@dataclass(frozen=True, slots=True)
class ValidatedReplayManifest:
    """Paths and identities from one verified completion manifest."""

    manifest_path: Path
    manifest_sha256: str
    state_path: Path
    action_path: Path
    state_sha256: str
    action_sha256: str
    state_rows: int
    action_rows: int
    config_manifest_sha256: str
    feature_schema_version: str
    game_id: int
    cycle_label: str


@dataclass(frozen=True, slots=True)
class ReplayActionSample:
    canonical_action_index: int
    representative_raw_index: int
    equivalent_raw_indices: tuple[int, ...]
    visits: int
    visit_probability: float
    value_sum: float
    mean_value: float
    prior: float | None
    selected: bool
    features: ActionFeatureSnapshot


@dataclass(frozen=True, slots=True)
class ReplayRootSample:
    """One value target and its complete canonical policy target."""

    source_manifest: Path
    game_id: int
    cycle_label: str
    decision_index: int
    root_node_id: int
    player: Role
    target_value: float
    state_features: StateFeatureSnapshot
    actions: tuple[ReplayActionSample, ...]

    @property
    def key(self) -> RootKey:
        return (self.game_id, self.cycle_label, self.decision_index)


@dataclass(frozen=True, slots=True)
class ReplayPartition:
    manifest: ValidatedReplayManifest
    roots: tuple[ReplayRootSample, ...]


@dataclass(frozen=True, slots=True)
class RoleTrainingData:
    """Arrays/sequences in the exact shape expected by the GV4 trainers."""

    role: Role
    states: tuple[StateFeatureSnapshot, ...]
    value_targets: np.ndarray
    policy_action_features: tuple[ActionFeatureSnapshot, ...]
    policy_targets: np.ndarray
    policy_offsets: tuple[tuple[int, int], ...]

    @property
    def root_count(self) -> int:
        return len(self.states)

    @property
    def action_count(self) -> int:
        return len(self.policy_action_features)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayDatasetError(f"cannot read replay JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayDatasetError(f"replay JSON is not an object: {path}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ReplayDatasetError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ReplayDatasetError(f"{label} must be a nonnegative integer") from error
    if result < 0 or isinstance(value, float) and value != result:
        raise ReplayDatasetError(f"{label} must be a nonnegative integer")
    return result


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReplayDatasetError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ReplayDatasetError(f"{label} must be finite")
    return result


def _role(value: Any) -> Role:
    result = str(value).lower()
    if result not in {"controller", "adversary"}:
        raise ReplayDatasetError(f"invalid replay player {value!r}")
    return result  # type: ignore[return-value]


def replay_root_key(row: Mapping[str, Any]) -> RootKey:
    return (
        _nonnegative_int(row.get("game_id"), "game_id"),
        str(row.get("cycle_label", "")),
        _nonnegative_int(row.get("decision_index"), "decision_index"),
    )


def _file_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReplayDatasetError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ReplayDatasetError(
                    f"replay row at {path}:{line_number} is not an object"
                )
            yield value


def discover_replay_manifests(root: str | Path) -> tuple[Path, ...]:
    """Return only explicitly completed replay manifests, in stable order."""

    source = Path(root).expanduser().resolve()
    if source.is_file():
        return (source,) if source.name == "replay_manifest.json" else ()
    return tuple(sorted(source.rglob("replay_manifest.json")))


def validate_replay_manifest(
    path: str | Path,
    config: GV4EngineConfig,
) -> ValidatedReplayManifest:
    """Verify completion, schemas, config identity, files, hashes, and counts."""

    config.validate()
    manifest_path = Path(path).expanduser().resolve()
    value = _read_json_object(manifest_path)
    if value.get("status") != "complete":
        raise ReplayDatasetError(f"replay is not complete: {manifest_path}")
    if value.get("replay_schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayDatasetError("replay schema version does not match GV4")

    config_hash = str(value.get("config_manifest_sha256", ""))
    feature_schema = str(value.get("feature_schema_version", ""))
    if config_hash != config.manifest_sha256():
        raise ReplayDatasetError("replay was generated by a different GV4 config")
    if feature_schema != config.layout.feature_schema_version:
        raise ReplayDatasetError("replay uses a different feature schema")

    metadata = value.get("engine_metadata")
    if not isinstance(metadata, Mapping):
        raise ReplayDatasetError("replay manifest has no engine metadata")
    expected_metadata = {
        "config_manifest_sha256": config_hash,
        "manifest_schema_version": config.layout.manifest_schema_version,
        "state_schema_version": config.layout.state_schema_version,
        "action_schema_version": config.layout.action_schema_version,
        "feature_schema_version": feature_schema,
        "native_layout_version": config.layout.native_layout_version,
    }
    for name, expected in expected_metadata.items():
        if str(metadata.get(name, "")) != str(expected):
            raise ReplayDatasetError(f"replay engine metadata differs at {name}")

    state_path = manifest_path.parent / "replay_states.jsonl"
    action_path = manifest_path.parent / "replay_actions.jsonl"
    if not state_path.is_file() or not action_path.is_file():
        raise ReplayDatasetError(f"replay files are missing beside {manifest_path}")
    expected_state_hash = str(value.get("state_sha256", ""))
    expected_action_hash = str(value.get("action_sha256", ""))
    if sha256_file(state_path) != expected_state_hash:
        raise ReplayDatasetError(f"state replay checksum mismatch: {state_path}")
    if sha256_file(action_path) != expected_action_hash:
        raise ReplayDatasetError(f"action replay checksum mismatch: {action_path}")

    game_id = _nonnegative_int(value.get("game_id"), "manifest game_id")
    cycle_label = str(value.get("cycle_label", ""))
    if not cycle_label:
        raise ReplayDatasetError("replay cycle label cannot be empty")
    return ValidatedReplayManifest(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        state_path=state_path,
        action_path=action_path,
        state_sha256=expected_state_hash,
        action_sha256=expected_action_hash,
        state_rows=_nonnegative_int(value.get("state_rows"), "state_rows"),
        action_rows=_nonnegative_int(value.get("action_rows"), "action_rows"),
        config_manifest_sha256=config_hash,
        feature_schema_version=feature_schema,
        game_id=game_id,
        cycle_label=cycle_label,
    )


def _vector(value: Any, width: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReplayDatasetError(f"{label} must be a vector")
    result = tuple(_finite_float(item, label) for item in value)
    if len(result) != width:
        raise ReplayDatasetError(f"{label} has width {len(result)}; expected {width}")
    return result


def _matrix(value: Any, width: int, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReplayDatasetError(f"{label} must be a matrix")
    return tuple(_vector(row, width, label) for row in value)


def _offsets(value: Any, row_count: int, replicas: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReplayDatasetError(f"{label} must be a vector")
    result = tuple(_nonnegative_int(item, label) for item in value)
    if len(result) != replicas + 1 or not result or result[0] != 0:
        raise ReplayDatasetError(f"{label} has an invalid replica shape")
    if any(right < left for left, right in zip(result, result[1:])):
        raise ReplayDatasetError(f"{label} must be nondecreasing")
    if result[-1] != row_count:
        raise ReplayDatasetError(f"{label} does not cover every row")
    return result


def _state_features(value: Any, spec: GV4ModelSpec) -> StateFeatureSnapshot:
    if not isinstance(value, Mapping):
        raise ReplayDatasetError("state_features must be an object")
    if str(value.get("schema_version", "")) != spec.feature_schema_version:
        raise ReplayDatasetError("state row uses a different feature schema")
    if str(value.get("config_manifest_sha256", "")) != spec.config_manifest_sha256:
        raise ReplayDatasetError("state row uses a different config manifest")

    requests = _matrix(value.get("request_rows"), spec.request_dim, "request_rows")
    microbatches = _matrix(
        value.get("microbatch_rows"), spec.microbatch_dim, "microbatch_rows"
    )
    replicas = _matrix(value.get("replica_rows"), spec.replica_dim, "replica_rows")
    if len(replicas) != spec.num_replicas:
        raise ReplayDatasetError("replica_rows must contain one row per replica")
    return StateFeatureSnapshot(
        schema_version=spec.feature_schema_version,
        config_manifest_sha256=spec.config_manifest_sha256,
        global_features=_vector(
            value.get("global_features"), spec.global_dim, "global_features"
        ),
        request_rows=requests,
        request_replica_offsets=_offsets(
            value.get("request_replica_offsets"),
            len(requests),
            spec.num_replicas,
            "request_replica_offsets",
        ),
        launch_rows=_matrix(value.get("launch_rows"), spec.launch_dim, "launch_rows"),
        replica_rows=replicas,
        microbatch_rows=microbatches,
        microbatch_replica_offsets=_offsets(
            value.get("microbatch_replica_offsets"),
            len(microbatches),
            spec.num_replicas,
            "microbatch_replica_offsets",
        ),
    )


def _action_features(
    value: Any,
    spec: GV4ModelSpec,
    role: Role,
) -> ActionFeatureSnapshot:
    if not isinstance(value, Mapping):
        raise ReplayDatasetError("action_features must be an object")
    header_width, row_width = spec.action_dimensions(role)
    return ActionFeatureSnapshot(
        header=_vector(value.get("header"), header_width, "action header"),
        affected_request_rows=_matrix(
            value.get("affected_request_rows"), row_width, "affected request rows"
        ),
    )


def _parse_action(
    row: Mapping[str, Any],
    *,
    key: RootKey,
    role: Role,
    spec: GV4ModelSpec,
) -> ReplayActionSample:
    if row.get("replay_schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayDatasetError("action row uses a different replay schema")
    if replay_root_key(row) != key or _role(row.get("player")) != role:
        raise ReplayDatasetError("action row belongs to another replay root")
    aliases_raw = row.get("equivalent_raw_indices")
    if not isinstance(aliases_raw, Sequence) or isinstance(aliases_raw, (str, bytes)):
        raise ReplayDatasetError("equivalent_raw_indices must be a list")
    aliases = tuple(
        _nonnegative_int(value, "raw action alias") for value in aliases_raw
    )
    representative = _nonnegative_int(
        row.get("representative_raw_index"), "representative_raw_index"
    )
    if (
        not aliases
        or representative not in aliases
        or len(set(aliases)) != len(aliases)
    ):
        raise ReplayDatasetError("raw action aliases are incomplete or duplicated")
    prior_raw = row.get("prior")
    prior = None if prior_raw is None else _finite_float(prior_raw, "action prior")
    if prior is not None and prior < 0.0:
        raise ReplayDatasetError("action prior cannot be negative")
    probability = _finite_float(row.get("visit_probability"), "visit_probability")
    if probability < 0.0:
        raise ReplayDatasetError("visit_probability cannot be negative")
    selected = row.get("selected")
    if not isinstance(selected, bool):
        raise ReplayDatasetError("selected must be a JSON boolean")
    return ReplayActionSample(
        canonical_action_index=_nonnegative_int(
            row.get("canonical_action_index"), "canonical_action_index"
        ),
        representative_raw_index=representative,
        equivalent_raw_indices=aliases,
        visits=_nonnegative_int(row.get("visits"), "visits"),
        visit_probability=probability,
        value_sum=_finite_float(row.get("value_sum"), "value_sum"),
        mean_value=_finite_float(row.get("mean_value"), "mean_value"),
        prior=prior,
        selected=selected,
        features=_action_features(row.get("action_features"), spec, role),
    )


def _parse_root(
    state_row: Mapping[str, Any],
    action_rows: Sequence[Mapping[str, Any]],
    *,
    manifest: ValidatedReplayManifest,
    spec: GV4ModelSpec,
) -> ReplayRootSample:
    if state_row.get("replay_schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayDatasetError("state row uses a different replay schema")
    key = replay_root_key(state_row)
    if key[:2] != (manifest.game_id, manifest.cycle_label):
        raise ReplayDatasetError("state row identity differs from its manifest")
    role = _role(state_row.get("player"))
    actions = tuple(
        _parse_action(row, key=key, role=role, spec=spec) for row in action_rows
    )
    expected_actions = _nonnegative_int(
        state_row.get("canonical_action_count"), "canonical_action_count"
    )
    if expected_actions != len(actions) or not actions:
        raise ReplayDatasetError("state/action row count differs")
    canonical_ids = {item.canonical_action_index for item in actions}
    representatives = {item.representative_raw_index for item in actions}
    if len(canonical_ids) != len(actions) or len(representatives) != len(actions):
        raise ReplayDatasetError("one replay root repeats a canonical action")
    if sum(item.selected for item in actions) != 1:
        raise ReplayDatasetError("one replay root must select exactly one action")
    probability_sum = sum(item.visit_probability for item in actions)
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ReplayDatasetError("policy visit probabilities do not sum to one")

    target = _finite_float(state_row.get("target_value"), "target_value")
    if target > 1e-6:
        raise ReplayDatasetError("controller-valued replay target cannot be positive")
    return ReplayRootSample(
        source_manifest=manifest.manifest_path,
        game_id=key[0],
        cycle_label=key[1],
        decision_index=key[2],
        root_node_id=_nonnegative_int(state_row.get("root_node_id"), "root_node_id"),
        player=role,
        target_value=target,
        state_features=_state_features(state_row.get("state_features"), spec),
        actions=actions,
    )


def load_replay_partition(
    manifest_path: str | Path,
    config: GV4EngineConfig,
) -> ReplayPartition:
    manifest = validate_replay_manifest(manifest_path, config)
    state_rows = tuple(_file_rows(manifest.state_path))
    action_rows = tuple(_file_rows(manifest.action_path))
    if (
        len(state_rows) != manifest.state_rows
        or len(action_rows) != manifest.action_rows
    ):
        raise ReplayDatasetError("replay row count differs from its manifest")

    grouped: dict[RootKey, list[dict[str, Any]]] = {}
    for row in action_rows:
        grouped.setdefault(replay_root_key(row), []).append(row)
    spec = GV4ModelSpec.from_config(config)
    roots: list[ReplayRootSample] = []
    seen: set[RootKey] = set()
    for row in state_rows:
        key = replay_root_key(row)
        if key in seen:
            raise ReplayDatasetError(f"duplicate replay root {key}")
        seen.add(key)
        roots.append(
            _parse_root(row, grouped.pop(key, ()), manifest=manifest, spec=spec)
        )
    if grouped:
        raise ReplayDatasetError("action replay contains roots with no state row")
    return ReplayPartition(manifest=manifest, roots=tuple(roots))


def _read_span_from(stream: Any, path: Path, offset: int, length: int) -> bytes:
    if offset < 0 or length <= 0:
        raise ReplayDatasetError("replay byte span is invalid")
    stream.seek(offset)
    value = stream.read(length)
    if len(value) != length:
        raise ReplayDatasetError(f"replay byte span exceeds {path}")
    return value


def load_root_at(
    manifest: ValidatedReplayManifest,
    config: GV4EngineConfig,
    *,
    state_offset: int,
    state_length: int,
    action_offset: int,
    action_length: int,
) -> ReplayRootSample:
    """Materialize one indexed root without scanning its replay partition."""

    return load_roots_at(
        manifest,
        config,
        ((state_offset, state_length, action_offset, action_length),),
    )[0]


def load_roots_at(
    manifest: ValidatedReplayManifest,
    config: GV4EngineConfig,
    spans: Sequence[tuple[int, int, int, int]],
) -> tuple[ReplayRootSample, ...]:
    """Materialize several roots while opening each replay file only once."""

    spec = GV4ModelSpec.from_config(config)
    roots: list[ReplayRootSample] = []
    with (
        manifest.state_path.open("rb") as state_stream,
        manifest.action_path.open("rb") as action_stream,
    ):
        for state_offset, state_length, action_offset, action_length in spans:
            try:
                state_row = json.loads(
                    _read_span_from(
                        state_stream,
                        manifest.state_path,
                        state_offset,
                        state_length,
                    )
                )
                action_rows = [
                    json.loads(line)
                    for line in _read_span_from(
                        action_stream,
                        manifest.action_path,
                        action_offset,
                        action_length,
                    ).splitlines()
                ]
            except json.JSONDecodeError as error:
                raise ReplayDatasetError(
                    "indexed replay span contains invalid JSON"
                ) from error
            if not isinstance(state_row, dict) or not all(
                isinstance(row, dict) for row in action_rows
            ):
                raise ReplayDatasetError(
                    "indexed replay span is not an object sequence"
                )
            roots.append(
                _parse_root(
                    state_row,
                    action_rows,
                    manifest=manifest,
                    spec=spec,
                )
            )
    return tuple(roots)


def materialize_training_data(
    roots: Iterable[ReplayRootSample],
    *,
    role: Role,
) -> RoleTrainingData:
    """Flatten policy actions while retaining one state per policy root."""

    selected = tuple(root for root in roots if root.player == role)
    if not selected:
        raise ReplayDatasetError(f"no {role} replay roots were selected")
    actions: list[ActionFeatureSnapshot] = []
    probabilities: list[float] = []
    offsets: list[tuple[int, int]] = []
    for root in selected:
        begin = len(actions)
        actions.extend(item.features for item in root.actions)
        probabilities.extend(item.visit_probability for item in root.actions)
        offsets.append((begin, len(actions)))
    return RoleTrainingData(
        role=role,
        states=tuple(root.state_features for root in selected),
        value_targets=np.asarray(
            [root.target_value for root in selected], dtype=np.float32
        ),
        policy_action_features=tuple(actions),
        policy_targets=np.asarray(probabilities, dtype=np.float32),
        policy_offsets=tuple(offsets),
    )
