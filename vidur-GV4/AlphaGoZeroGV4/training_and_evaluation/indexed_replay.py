"""Disk-backed random access over immutable GV4 replay partitions.

The index stores byte ranges, not decoded feature tensors.  A training cycle can
therefore sample roots without loading every historical JSONL file into memory.
The expensive checksum and row validation happens only when the index is built.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from GV4_Engine.config import GV4EngineConfig

from ..model_bundle import sha256_file
from .replay_dataset import (
    ReplayDatasetError,
    ReplayRootSample,
    Role,
    RootKey,
    ValidatedReplayManifest,
    discover_replay_manifests,
    load_roots_at,
    replay_root_key,
    validate_replay_manifest,
)


INDEX_SCHEMA_VERSION = "gv4_replay_index_v1"
_ROLE_TO_CODE = {"controller": 0, "adversary": 1}
_CODE_TO_ROLE: dict[int, Role] = {0: "controller", 1: "adversary"}
_ADDRESS_DTYPE = np.dtype(
    [
        ("partition", "<u4"),
        ("state_offset", "<u8"),
        ("state_length", "<u8"),
        ("action_offset", "<u8"),
        ("action_length", "<u8"),
        ("action_count", "<u4"),
        ("role", "u1"),
        ("game_id", "<u8"),
        ("decision_index", "<u4"),
        ("root_node_id", "<u8"),
    ]
)

__all__ = [
    "INDEX_SCHEMA_VERSION",
    "ReplayAddress",
    "ReplayIndex",
    "ReplayIndexError",
    "build_replay_index",
    "open_replay_index",
]


class ReplayIndexError(RuntimeError):
    """Raised when replay cannot be indexed without ambiguity."""


@dataclass(frozen=True, slots=True)
class ReplayAddress:
    """One root's immutable location in a replay state/action file pair."""

    partition: int
    state_offset: int
    state_length: int
    action_offset: int
    action_length: int
    action_count: int
    role: Role
    game_id: int
    decision_index: int
    root_node_id: int

    @property
    def span(self) -> tuple[int, int, int, int]:
        return (
            self.state_offset,
            self.state_length,
            self.action_offset,
            self.action_length,
        )


class ReplayIndex:
    """Memory-mapped addresses plus validated immutable partition metadata."""

    def __init__(
        self,
        *,
        replay_root: Path,
        index_dir: Path,
        config: GV4EngineConfig,
        manifests: Sequence[ValidatedReplayManifest],
        addresses: np.ndarray,
    ) -> None:
        self.replay_root = replay_root
        self.index_dir = index_dir
        self.config = config
        self.manifests = tuple(manifests)
        self.addresses = addresses

    def __len__(self) -> int:
        return int(len(self.addresses))

    @property
    def role_counts(self) -> dict[Role, int]:
        codes = self.addresses["role"]
        return {
            role: int(np.count_nonzero(codes == code))
            for role, code in _ROLE_TO_CODE.items()
        }

    def sample_addresses(
        self,
        role: Role,
        count: int,
        *,
        seed: int,
    ) -> tuple[ReplayAddress, ...]:
        """Sample distinct roots for one player using a repeatable seed."""

        if role not in _ROLE_TO_CODE:
            raise ValueError("role must be controller or adversary")
        if isinstance(count, bool) or count < 0:
            raise ValueError("sample count must be nonnegative")
        matching = np.flatnonzero(self.addresses["role"] == _ROLE_TO_CODE[role])
        if not count or not len(matching):
            return ()
        take = min(int(count), int(len(matching)))
        selected = np.random.default_rng(seed).choice(
            matching, size=take, replace=False
        )
        return tuple(_address(self.addresses[int(index)]) for index in selected)

    def sample(
        self,
        role: Role,
        count: int,
        *,
        seed: int,
    ) -> tuple[ReplayRootSample, ...]:
        """Materialize sampled roots, opening each touched partition once."""

        addresses = self.sample_addresses(role, count, seed=seed)
        grouped: dict[int, list[tuple[int, ReplayAddress]]] = {}
        for output_index, address in enumerate(addresses):
            grouped.setdefault(address.partition, []).append((output_index, address))

        roots: list[ReplayRootSample | None] = [None] * len(addresses)
        for partition, entries in grouped.items():
            if not 0 <= partition < len(self.manifests):
                raise ReplayIndexError("index references an unknown replay partition")
            entries.sort(key=lambda item: item[1].state_offset)
            loaded = load_roots_at(
                self.manifests[partition],
                self.config,
                tuple(item.span for _, item in entries),
            )
            for (output_index, _), root in zip(entries, loaded):
                roots[output_index] = root
        if any(root is None for root in roots):
            raise ReplayIndexError("index failed to materialize every sampled root")
        return tuple(root for root in roots if root is not None)


def _json_row(line: bytes, path: Path, offset: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ReplayIndexError(f"invalid JSON in {path} at byte {offset}") from error
    if not isinstance(value, dict):
        raise ReplayIndexError(f"replay row in {path} is not an object")
    return value


def _role(value: Any) -> Role:
    role = str(value).lower()
    if role not in _ROLE_TO_CODE:
        raise ReplayIndexError(f"invalid replay player {value!r}")
    return role  # type: ignore[return-value]


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ReplayIndexError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ReplayIndexError(f"{label} must be a nonnegative integer") from error
    if result < 0 or isinstance(value, float) and value != result:
        raise ReplayIndexError(f"{label} must be a nonnegative integer")
    return result


def _action_groups(path: Path) -> Iterator[tuple[RootKey, Role, int, int, int]]:
    """Yield contiguous action spans; recorder order makes this a streaming pass."""

    pending: tuple[int, bytes, dict[str, Any]] | None = None
    with path.open("rb") as stream:
        while True:
            if pending is None:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    return
                row = _json_row(line, path, offset)
            else:
                offset, line, row = pending
                pending = None

            key = replay_root_key(row)
            role = _role(row.get("player"))
            length = len(line)
            count = 1
            while True:
                next_offset = stream.tell()
                next_line = stream.readline()
                if not next_line:
                    break
                next_row = _json_row(next_line, path, next_offset)
                if replay_root_key(next_row) != key:
                    pending = (next_offset, next_line, next_row)
                    break
                length += len(next_line)
                count += 1
            yield key, role, offset, length, count
            if not next_line:
                return


def _scan_partition(
    partition: int,
    manifest: ValidatedReplayManifest,
    config: GV4EngineConfig,
) -> list[tuple[int, ...]]:
    groups = iter(_action_groups(manifest.action_path))
    addresses: list[tuple[int, ...]] = []
    action_rows = 0
    with manifest.state_path.open("rb") as stream:
        decision = 0
        while True:
            state_offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = _json_row(line, manifest.state_path, state_offset)
            key = replay_root_key(row)
            if key != (manifest.game_id, manifest.cycle_label, decision):
                raise ReplayIndexError(
                    "state replay decisions must be contiguous and match the manifest"
                )
            try:
                action_key, action_role, action_offset, action_length, count = next(
                    groups
                )
            except StopIteration as error:
                raise ReplayIndexError(
                    "state root has no canonical action rows"
                ) from error
            role = _role(row.get("player"))
            if action_key != key or action_role != role:
                raise ReplayIndexError("state and action replay roots are out of order")
            expected_count = _nonnegative_int(
                row.get("canonical_action_count"), "canonical_action_count"
            )
            if count != expected_count or count <= 0:
                raise ReplayIndexError("canonical action count differs from its state")
            root_node_id = _nonnegative_int(row.get("root_node_id"), "root_node_id")
            addresses.append(
                (
                    partition,
                    state_offset,
                    len(line),
                    action_offset,
                    action_length,
                    count,
                    _ROLE_TO_CODE[role],
                    manifest.game_id,
                    decision,
                    root_node_id,
                )
            )
            action_rows += count
            decision += 1

    try:
        next(groups)
    except StopIteration:
        pass
    else:
        raise ReplayIndexError("action replay has roots with no state row")
    if len(addresses) != manifest.state_rows or action_rows != manifest.action_rows:
        raise ReplayIndexError("indexed row totals differ from the replay manifest")

    # Parse every feature once at build time. Later reads can trust the byte index
    # while still rechecking each sampled root's local shape and identity.
    for begin in range(0, len(addresses), 512):
        chunk = addresses[begin : begin + 512]
        load_roots_at(
            manifest,
            config,
            tuple((row[1], row[2], row[3], row[4]) for row in chunk),
        )
    return addresses


def _manifest_record(manifest: ValidatedReplayManifest) -> dict[str, Any]:
    value = asdict(manifest)
    for name in ("manifest_path", "state_path", "action_path"):
        value[name] = str(value[name])
    return value


def _manifest_from_record(value: Mapping[str, Any]) -> ValidatedReplayManifest:
    fields = dict(value)
    for name in ("manifest_path", "state_path", "action_path"):
        fields[name] = Path(str(fields[name]))
    try:
        return ValidatedReplayManifest(**fields)
    except TypeError as error:
        raise ReplayIndexError("indexed manifest metadata is invalid") from error


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_signatures(replay_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for manifest_path in discover_replay_manifests(replay_root):
        state_path = manifest_path.parent / "replay_states.jsonl"
        action_path = manifest_path.parent / "replay_actions.jsonl"
        if not state_path.is_file() or not action_path.is_file():
            raise ReplayIndexError(f"replay files are missing beside {manifest_path}")
        result.append(
            {
                "manifest": {
                    **_file_signature(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "state": _file_signature(state_path),
                "action": _file_signature(action_path),
            }
        )
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_replay_index(
    replay_root: str | Path,
    config: GV4EngineConfig,
    *,
    index_dir: str | Path | None = None,
) -> Path:
    """Validate all immutable partitions and atomically publish their index."""

    config.validate()
    root = Path(replay_root).expanduser().resolve()
    output = (
        Path(index_dir).expanduser().resolve()
        if index_dir is not None
        else (root.parent if root.is_file() else root) / ".gv4_replay_index"
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = discover_replay_manifests(root)
    if not paths:
        raise ReplayIndexError(f"no completed replay manifests under {root}")

    manifests: list[ValidatedReplayManifest] = []
    rows: list[tuple[int, ...]] = []
    for partition, path in enumerate(paths):
        try:
            manifest = validate_replay_manifest(path, config)
            rows.extend(_scan_partition(partition, manifest, config))
        except ReplayDatasetError as error:
            raise ReplayIndexError(
                f"invalid replay partition {path}: {error}"
            ) from error
        manifests.append(manifest)

    addresses = np.asarray(rows, dtype=_ADDRESS_DTYPE)
    _atomic_array(output / "addresses.npy", addresses)
    counts = {
        role: int(np.count_nonzero(addresses["role"] == code))
        for role, code in _ROLE_TO_CODE.items()
    }
    _atomic_json(
        output / "index.json",
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "config_manifest_sha256": config.manifest_sha256(),
            "feature_schema_version": config.layout.feature_schema_version,
            "replay_root": str(root),
            "source_signatures": _source_signatures(root),
            "manifests": [_manifest_record(item) for item in manifests],
            "root_count": len(addresses),
            "role_counts": counts,
            "address_dtype": _ADDRESS_DTYPE.descr,
        },
    )
    return output


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_current(
    metadata: Mapping[str, Any] | None,
    root: Path,
    config: GV4EngineConfig,
) -> bool:
    if metadata is None:
        return False
    try:
        return (
            metadata.get("schema_version") == INDEX_SCHEMA_VERSION
            and metadata.get("config_manifest_sha256") == config.manifest_sha256()
            and metadata.get("feature_schema_version")
            == config.layout.feature_schema_version
            and metadata.get("source_signatures") == _source_signatures(root)
        )
    except (OSError, ReplayIndexError):
        return False


def open_replay_index(
    replay_root: str | Path,
    config: GV4EngineConfig,
    *,
    index_dir: str | Path | None = None,
    rebuild: bool = False,
) -> ReplayIndex:
    """Open a fresh index, rebuilding it under an inter-process file lock."""

    config.validate()
    root = Path(replay_root).expanduser().resolve()
    output = (
        Path(index_dir).expanduser().resolve()
        if index_dir is not None
        else (root.parent if root.is_file() else root) / ".gv4_replay_index"
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "index.json"
    address_path = output / "addresses.npy"
    with (output / "index.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        metadata = _read_metadata(metadata_path)
        if (
            rebuild
            or not address_path.is_file()
            or not _is_current(metadata, root, config)
        ):
            build_replay_index(root, config, index_dir=output)
            metadata = _read_metadata(metadata_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    if metadata is None:
        raise ReplayIndexError("replay index metadata was not published")
    manifests_raw = metadata.get("manifests")
    if not isinstance(manifests_raw, list):
        raise ReplayIndexError("replay index has no partition metadata")
    manifests = tuple(_manifest_from_record(item) for item in manifests_raw)
    try:
        addresses = np.load(address_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ReplayIndexError("cannot load replay address index") from error
    if addresses.dtype != _ADDRESS_DTYPE or addresses.ndim != 1:
        raise ReplayIndexError("replay address index has an incompatible layout")
    if len(addresses) != int(metadata.get("root_count", -1)):
        raise ReplayIndexError("replay address count differs from index metadata")
    return ReplayIndex(
        replay_root=root,
        index_dir=output,
        config=config,
        manifests=manifests,
        addresses=addresses,
    )


def _address(row: np.void) -> ReplayAddress:
    code = int(row["role"])
    if code not in _CODE_TO_ROLE:
        raise ReplayIndexError("index contains an invalid player code")
    return ReplayAddress(
        partition=int(row["partition"]),
        state_offset=int(row["state_offset"]),
        state_length=int(row["state_length"]),
        action_offset=int(row["action_offset"]),
        action_length=int(row["action_length"]),
        action_count=int(row["action_count"]),
        role=_CODE_TO_ROLE[code],
        game_id=int(row["game_id"]),
        decision_index=int(row["decision_index"]),
        root_node_id=int(row["root_node_id"]),
    )
