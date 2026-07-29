"""Indexed, without-replacement replay sampling for AlphaGoZero training.

The legacy sampler opens every cache and considers every state/action row on
each training cycle.  This module builds one immutable, memory-mappable index
per replay partition.  A training cycle samples global row addresses once,
maps them to partition-local rows, and reads only those rows.
"""

from __future__ import annotations

import bisect
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vidur.AlphaGoZero.markov_value_features import (
    GLOBAL_DIM as MARKOV_GLOBAL_DIM,
    LAUNCH_DIM as MARKOV_LAUNCH_DIM,
    MARKOV_VALUE_SCHEMA,
    REQUEST_DIM as MARKOV_REQUEST_DIM,
    MarkovValueFeatures,
    features_from_replay_row,
)


INDEX_VERSION = 1
KEY_SEP = "\x1f"
VALUE_FEATURE_DIM = 226
CONTROLLER_ACTION_DIM = 43
ADVERSARY_ACTION_DIM = 7
POLICY_ALPHA = 1.0


@dataclass(frozen=True)
class ReplayIndexDescriptor:
    cache_dir: str
    source_path: str
    policy_path: str
    role: str
    rows: int
    policy_roots: int
    action_rows: int


@dataclass(frozen=True)
class ReplayAddress:
    order: int
    cache_dir: str
    local_row: int


class IndexedRoots(dict):
    """Selected policy roots plus their direct partition/local-row locators."""

    def __init__(self) -> None:
        super().__init__()
        self.locators: dict[tuple[str, str, int, int, int, str], tuple[str, int]] = {}


def _read_json_list(raw: str) -> list[float]:
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except Exception:
        return []
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            return []
        if not math.isfinite(number):
            return []
        result.append(number)
    return result


def _root_key(row: dict[str, str]) -> tuple[str, str, int, int, int, str]:
    return (
        str(row.get("accepted_shard_id", "")),
        str(row.get("source_worker_id", "")),
        int(float(row.get("game_id", 0) or 0)),
        int(float(row.get("turn_number", 0) or 0)),
        int(float(row.get("depth_number", 0) or 0)),
        str(row.get("player", "")).lower(),
    )


def key_to_string(key: tuple[str, str, int, int, int, str]) -> str:
    return KEY_SEP.join(
        (
            str(key[0]),
            str(key[1]),
            str(int(key[2])),
            str(int(key[3])),
            str(int(key[4])),
            str(key[5]),
        )
    )


def key_from_string(raw: str) -> tuple[str, str, int, int, int, str]:
    fields = str(raw).split(KEY_SEP)
    if len(fields) != 6:
        raise ValueError(f"invalid indexed replay key: {raw!r}")
    return (fields[0], fields[1], int(fields[2]), int(fields[3]), int(fields[4]), fields[5])


def _signature(path: Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def cache_dir_for_state(source: Path, value_feature_schema: str) -> Path:
    schema = str(value_feature_schema).replace("/", "_")
    return Path(source).parent / f".replay_index_{schema}_v{INDEX_VERSION}"


def _metadata_is_fresh(
    cache_dir: Path,
    state_path: Path,
    policy_path: Path,
    value_feature_schema: str,
) -> bool:
    metadata_path = Path(cache_dir) / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            int(metadata.get("version", -1)) == int(INDEX_VERSION)
            and str(metadata.get("value_feature_schema", "")) == str(value_feature_schema)
            and dict(metadata.get("state_signature", {})) == _signature(state_path)
            and dict(metadata.get("policy_signature", {})) == _signature(policy_path)
        )
    except Exception:
        return False


def _save_array(directory: Path, name: str, value: np.ndarray) -> None:
    path = Path(directory) / f"{name}.npy"
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _load_descriptor(cache_dir: Path) -> ReplayIndexDescriptor:
    metadata = json.loads((Path(cache_dir) / "metadata.json").read_text(encoding="utf-8"))
    return ReplayIndexDescriptor(
        cache_dir=str(Path(cache_dir)),
        source_path=str(metadata["source_path"]),
        policy_path=str(metadata["policy_path"]),
        role=str(metadata["role"]),
        rows=int(metadata["rows"]),
        policy_roots=int(metadata["policy_roots"]),
        action_rows=int(metadata["action_rows"]),
    )


def ensure_partition_index(
    state_path: Path,
    *,
    value_feature_schema: str,
) -> tuple[ReplayIndexDescriptor, bool, float]:
    """Build or validate the row-aligned index for one immutable partition."""

    started = time.perf_counter()
    state_path = Path(state_path)
    policy_path = state_path.parent / "replay_policy_rows.csv"
    if not state_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError(f"replay partition is incomplete: {state_path.parent}")
    cache_dir = cache_dir_for_state(state_path, value_feature_schema)
    lock_path = cache_dir.with_name(cache_dir.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _metadata_is_fresh(cache_dir, state_path, policy_path, value_feature_schema):
            return _load_descriptor(cache_dir), False, float(time.perf_counter() - started)

        keys: list[str] = []
        roles: list[str] = []
        state_features: list[np.ndarray] = []
        targets: list[float] = []
        value_globals: list[np.ndarray] = []
        request_offsets = [0]
        request_features: list[np.ndarray] = []
        launch_offsets = [0]
        launch_features: list[np.ndarray] = []

        with state_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True"}:
                    continue
                role = str(row.get("player", row.get("root_player", ""))).lower()
                if role not in {"controller", "adversary"}:
                    continue
                legacy = _read_json_list(str(row.get("state_features_json", "")))
                if len(legacy) != VALUE_FEATURE_DIM:
                    continue
                try:
                    target = float(row.get("target_value", 0.0) or 0.0)
                except Exception:
                    continue
                if not math.isfinite(target):
                    continue
                if str(value_feature_schema) == MARKOV_VALUE_SCHEMA:
                    if str(row.get("value_feature_complete", "0")).strip() not in {"1", "true", "True"}:
                        continue
                    try:
                        structured = features_from_replay_row(row)
                    except ValueError:
                        continue
                    value_globals.append(structured.global_features)
                    request_features.extend(structured.request_features)
                    launch_features.extend(structured.launch_features)
                keys.append(key_to_string(_root_key(row)))
                roles.append(role)
                state_features.append(np.asarray(legacy, dtype=np.float32))
                targets.append(target)
                request_offsets.append(len(request_features))
                launch_offsets.append(len(launch_features))

        distinct_roles = sorted(set(roles))
        if len(distinct_roles) != 1:
            raise RuntimeError(
                f"indexed replay requires role-specific partitions, got roles={distinct_roles} "
                f"in {state_path}"
            )
        role = distinct_roles[0]
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"duplicate replay root key in {state_path}")
        key_to_row = {key: index for index, key in enumerate(keys)}
        action_dim = CONTROLLER_ACTION_DIM if role == "controller" else ADVERSARY_ACTION_DIM
        actions_by_row: dict[int, list[tuple[int, np.ndarray, int]]] = {}
        policy_rows_read = 0
        with policy_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                policy_rows_read += 1
                if str(row.get("player", "")).lower() != role:
                    continue
                local_row = key_to_row.get(key_to_string(_root_key(row)))
                if local_row is None:
                    continue
                action = _read_json_list(str(row.get("action_features_json", "")))
                if len(action) != action_dim:
                    continue
                try:
                    canon_index = int(float(row.get("canon_action_index", -1) or -1))
                    visits = int(float(row.get("visit_count", 0) or 0))
                except Exception:
                    continue
                actions_by_row.setdefault(local_row, []).append(
                    (canon_index, np.asarray(action, dtype=np.float32), visits)
                )

        policy_offsets = [0]
        policy_eligible_rows: list[int] = []
        action_indices: list[int] = []
        visits: list[int] = []
        action_features: list[np.ndarray] = []
        for local_row in range(len(keys)):
            actions = sorted(actions_by_row.get(local_row, ()), key=lambda item: item[0])
            if actions:
                policy_eligible_rows.append(local_row)
            for canon_index, action, visit_count in actions:
                action_indices.append(int(canon_index))
                visits.append(max(0, int(visit_count)))
                action_features.append(action)
            policy_offsets.append(len(action_indices))

        state_array = (
            np.vstack(state_features).astype(np.float32, copy=False)
            if state_features
            else np.empty((0, VALUE_FEATURE_DIM), dtype=np.float32)
        )
        global_array = (
            np.vstack(value_globals).astype(np.float32, copy=False)
            if value_globals
            else np.empty((0, MARKOV_GLOBAL_DIM), dtype=np.float32)
        )
        request_array = (
            np.vstack(request_features).astype(np.float32, copy=False)
            if request_features
            else np.empty((0, MARKOV_REQUEST_DIM), dtype=np.float32)
        )
        launch_array = (
            np.vstack(launch_features).astype(np.float32, copy=False)
            if launch_features
            else np.empty((0, MARKOV_LAUNCH_DIM), dtype=np.float32)
        )
        action_array = (
            np.vstack(action_features).astype(np.float32, copy=False)
            if action_features
            else np.empty((0, action_dim), dtype=np.float32)
        )

        temporary = cache_dir.with_name(cache_dir.name + f".tmp.{os.getpid()}.{time.time_ns()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            _save_array(temporary, "keys", np.asarray(keys, dtype=np.str_))
            _save_array(temporary, "state_features", state_array)
            _save_array(temporary, "targets", np.asarray(targets, dtype=np.float32))
            _save_array(temporary, "value_globals", global_array)
            _save_array(temporary, "request_offsets", np.asarray(request_offsets, dtype=np.int64))
            _save_array(temporary, "request_features", request_array)
            _save_array(temporary, "launch_offsets", np.asarray(launch_offsets, dtype=np.int64))
            _save_array(temporary, "launch_features", launch_array)
            _save_array(temporary, "policy_offsets", np.asarray(policy_offsets, dtype=np.int64))
            _save_array(
                temporary,
                "policy_eligible_rows",
                np.asarray(policy_eligible_rows, dtype=np.int64),
            )
            _save_array(temporary, "action_indices", np.asarray(action_indices, dtype=np.int32))
            _save_array(temporary, "visits", np.asarray(visits, dtype=np.int32))
            _save_array(temporary, "action_features", action_array)
            metadata = {
                "version": int(INDEX_VERSION),
                "value_feature_schema": str(value_feature_schema),
                "source_path": str(state_path),
                "policy_path": str(policy_path),
                "state_signature": _signature(state_path),
                "policy_signature": _signature(policy_path),
                "role": role,
                "rows": int(len(keys)),
                "policy_roots": int(len(policy_eligible_rows)),
                "action_rows": int(len(action_indices)),
                "policy_rows_read": int(policy_rows_read),
                "action_dim": int(action_dim),
            }
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            os.replace(temporary, cache_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return _load_descriptor(cache_dir), True, float(time.perf_counter() - started)


def ensure_partition_index_worker(
    state_path: str,
    value_feature_schema: str,
) -> tuple[ReplayIndexDescriptor, bool, float]:
    return ensure_partition_index(
        Path(state_path),
        value_feature_schema=str(value_feature_schema),
    )


def ensure_partition_indexes(
    state_paths: Iterable[Path],
    *,
    value_feature_schema: str,
    workers: int,
) -> tuple[list[ReplayIndexDescriptor], dict[str, float | int]]:
    started = time.perf_counter()
    paths = [Path(path) for path in state_paths if Path(path).is_file()]
    descriptors: dict[str, ReplayIndexDescriptor] = {}
    missing: list[Path] = []
    for path in paths:
        policy_path = path.parent / "replay_policy_rows.csv"
        cache_dir = cache_dir_for_state(path, value_feature_schema)
        if policy_path.is_file() and _metadata_is_fresh(
            cache_dir,
            path,
            policy_path,
            value_feature_schema,
        ):
            descriptors[str(path)] = _load_descriptor(cache_dir)
        else:
            missing.append(path)

    rebuilt = 0
    if missing:
        process_count = min(max(1, int(workers)), len(missing))
        if process_count == 1:
            for path in missing:
                descriptor, did_rebuild, _ = ensure_partition_index(
                    path,
                    value_feature_schema=value_feature_schema,
                )
                descriptors[str(path)] = descriptor
                rebuilt += int(did_rebuild)
        else:
            with ProcessPoolExecutor(max_workers=process_count) as executor:
                futures = {
                    executor.submit(
                        ensure_partition_index_worker,
                        str(path),
                        str(value_feature_schema),
                    ): path
                    for path in missing
                }
                for future in as_completed(futures):
                    path = futures[future]
                    descriptor, did_rebuild, _ = future.result()
                    descriptors[str(path)] = descriptor
                    rebuilt += int(did_rebuild)
    ordered = [descriptors[str(path)] for path in paths if str(path) in descriptors]
    return ordered, {
        "indexed_cache_files": int(len(ordered)),
        "indexed_cache_files_rebuilt": int(rebuilt),
        "indexed_cache_ensure_elapsed_s": float(time.perf_counter() - started),
    }


def sample_addresses(
    descriptors: Iterable[ReplayIndexDescriptor],
    *,
    role: str,
    sample_size: int,
    seed: int,
    policy_only: bool = False,
) -> tuple[list[ReplayAddress], int]:
    """Uniformly sample unique global rows and map them to local cache rows."""

    eligible = [descriptor for descriptor in descriptors if descriptor.role == str(role)]
    counts = [
        int(descriptor.policy_roots if policy_only else descriptor.rows)
        for descriptor in eligible
    ]
    cumulative: list[int] = []
    total = 0
    for count in counts:
        total += max(0, int(count))
        cumulative.append(total)
    chosen_count = min(max(0, int(sample_size)), total)
    if chosen_count <= 0:
        return [], total
    rng = random.Random(int(seed))
    global_rows = rng.sample(range(total), chosen_count)
    result: list[ReplayAddress | None] = [None] * chosen_count
    policy_selections: dict[int, list[tuple[int, int]]] = {}
    for order, global_row in enumerate(global_rows):
        partition_index = bisect.bisect_right(cumulative, int(global_row))
        previous = 0 if partition_index == 0 else cumulative[partition_index - 1]
        local = int(global_row) - int(previous)
        descriptor = eligible[partition_index]
        if policy_only:
            policy_selections.setdefault(partition_index, []).append((order, local))
            continue
        result[order] = ReplayAddress(
            order=int(order),
            cache_dir=str(descriptor.cache_dir),
            local_row=int(local),
        )

    # Resolve policy-root indirection one partition at a time. Retaining a
    # memmap per partition exhausts the process file limit as replay grows.
    for partition_index, selections in policy_selections.items():
        descriptor = eligible[partition_index]
        policy_rows_path = Path(descriptor.cache_dir) / "policy_eligible_rows.npy"
        with policy_rows_path.open("rb") as handle:
            rows = np.load(handle, allow_pickle=False)
        for order, local in selections:
            result[order] = ReplayAddress(
                order=int(order),
                cache_dir=str(descriptor.cache_dir),
                local_row=int(rows[local]),
            )

    if any(address is None for address in result):
        raise RuntimeError("indexed replay address resolution was incomplete")
    return [address for address in result if address is not None], total


def address_fingerprint(addresses: Iterable[ReplayAddress]) -> str:
    digest = hashlib.sha256()
    for address in addresses:
        digest.update(str(address.cache_dir).encode("utf-8"))
        digest.update(b"\0")
        digest.update(int(address.local_row).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _group_addresses(addresses: Iterable[ReplayAddress]) -> list[tuple[str, list[tuple[int, int]]]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for address in addresses:
        grouped.setdefault(address.cache_dir, []).append((int(address.order), int(address.local_row)))
    return list(grouped.items())


def _balanced_batches(
    groups: list[tuple[str, list[tuple[int, int]]]],
    workers: int,
) -> list[list[tuple[str, list[tuple[int, int]]]]]:
    count = min(max(1, int(workers)), max(1, len(groups)))
    batches: list[list[tuple[str, list[tuple[int, int]]]]] = [[] for _ in range(count)]
    loads = [0] * count
    for group in sorted(groups, key=lambda item: len(item[1]), reverse=True):
        target = min(range(count), key=loads.__getitem__)
        batches[target].append(group)
        loads[target] += len(group[1])
    return [batch for batch in batches if batch]


def _load_selected_states_worker(
    groups: list[tuple[str, list[tuple[int, int]]]],
    value_feature_schema: str,
    include_target: bool,
) -> list[tuple[Any, ...]]:
    output: list[tuple[Any, ...]] = []
    for cache_dir_raw, selected in groups:
        cache_dir = Path(cache_dir_raw)
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        role = str(metadata["role"])
        keys = np.load(cache_dir / "keys.npy", mmap_mode="r", allow_pickle=False)
        states = np.load(cache_dir / "state_features.npy", mmap_mode="r", allow_pickle=False)
        targets = (
            np.load(cache_dir / "targets.npy", mmap_mode="r", allow_pickle=False)
            if include_target
            else None
        )
        globals_array = requests = request_offsets = launches = launch_offsets = None
        if str(value_feature_schema) == MARKOV_VALUE_SCHEMA and include_target:
            globals_array = np.load(cache_dir / "value_globals.npy", mmap_mode="r", allow_pickle=False)
            request_offsets = np.load(cache_dir / "request_offsets.npy", mmap_mode="r", allow_pickle=False)
            requests = np.load(cache_dir / "request_features.npy", mmap_mode="r", allow_pickle=False)
            launch_offsets = np.load(cache_dir / "launch_offsets.npy", mmap_mode="r", allow_pickle=False)
            launches = np.load(cache_dir / "launch_features.npy", mmap_mode="r", allow_pickle=False)
        for order, local_row in selected:
            structured: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            target = 0.0
            if include_target:
                assert targets is not None
                target = float(targets[local_row])
                if str(value_feature_schema) == MARKOV_VALUE_SCHEMA:
                    assert globals_array is not None
                    assert request_offsets is not None and requests is not None
                    assert launch_offsets is not None and launches is not None
                    request_begin = int(request_offsets[local_row])
                    request_end = int(request_offsets[local_row + 1])
                    launch_begin = int(launch_offsets[local_row])
                    launch_end = int(launch_offsets[local_row + 1])
                    structured = (
                        np.asarray(globals_array[local_row], dtype=np.float32).copy(),
                        np.asarray(requests[request_begin:request_end], dtype=np.float32).copy(),
                        np.asarray(launches[launch_begin:launch_end], dtype=np.float32).copy(),
                    )
            output.append(
                (
                    int(order),
                    str(keys[local_row]),
                    np.asarray(states[local_row], dtype=np.float32).copy(),
                    float(target),
                    role,
                    structured,
                    str(cache_dir),
                    int(local_row),
                )
            )
    return output


def materialize_value_samples(
    addresses: Iterable[ReplayAddress],
    *,
    value_feature_schema: str,
    workers: int,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    groups = _group_addresses(addresses)
    batches = _balanced_batches(groups, workers)
    records: list[tuple[Any, ...]] = []
    if len(batches) <= 1:
        records = _load_selected_states_worker(
            batches[0] if batches else [],
            str(value_feature_schema),
            True,
        )
    else:
        with ProcessPoolExecutor(max_workers=len(batches)) as executor:
            futures = [
                executor.submit(
                    _load_selected_states_worker,
                    batch,
                    str(value_feature_schema),
                    True,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda item: int(item[0]))
    samples: list[dict[str, Any]] = []
    for _order, raw_key, state, target, role, structured, _cache_dir, _local_row in records:
        if str(value_feature_schema) == MARKOV_VALUE_SCHEMA:
            if structured is None:
                raise RuntimeError("indexed Markov sample is missing structured features")
            global_features, request_features, launch_features = structured
            value_features: np.ndarray | MarkovValueFeatures = MarkovValueFeatures(
                global_features=global_features,
                request_features=request_features,
                launch_features=launch_features,
                request_ids=tuple(range(int(request_features.shape[0]))),
            )
        else:
            value_features = state
        samples.append(
            {
                "_key": key_from_string(raw_key),
                "_state_features": state,
                "_value_features": value_features,
                "_target_value": float(target),
                "player": str(role),
            }
        )
    _ = started
    return samples


def materialize_policy_roots(
    addresses: Iterable[ReplayAddress],
    *,
    value_feature_schema: str,
    workers: int,
) -> IndexedRoots:
    groups = _group_addresses(addresses)
    batches = _balanced_batches(groups, workers)
    records: list[tuple[Any, ...]] = []
    if len(batches) <= 1:
        records = _load_selected_states_worker(
            batches[0] if batches else [],
            str(value_feature_schema),
            False,
        )
    else:
        with ProcessPoolExecutor(max_workers=len(batches)) as executor:
            futures = [
                executor.submit(
                    _load_selected_states_worker,
                    batch,
                    str(value_feature_schema),
                    False,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda item: int(item[0]))
    roots = IndexedRoots()
    for _order, raw_key, state, _target, _role, _structured, cache_dir, local_row in records:
        key = key_from_string(raw_key)
        roots[key] = state
        roots.locators[key] = (str(cache_dir), int(local_row))
    return roots


def materialize_policy_arrays(
    roots: IndexedRoots,
    *,
    action_dim: int,
    root_cap: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]],
    dict[str, int | float],
]:
    """Read only selected action ranges and create the existing training arrays."""

    selected_keys = list(roots.keys())[: max(0, int(root_cap))]
    grouped: dict[str, list[tuple[tuple[str, str, int, int, int, str], int]]] = {}
    for key in selected_keys:
        cache_dir, local_row = roots.locators[key]
        grouped.setdefault(cache_dir, []).append((key, int(local_row)))

    root_records: list[
        tuple[
            tuple[str, str, int, int, int, str],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    for cache_dir_raw, selected in grouped.items():
        cache_dir = Path(cache_dir_raw)
        offsets = np.load(cache_dir / "policy_offsets.npy", mmap_mode="r", allow_pickle=False)
        indices = np.load(cache_dir / "action_indices.npy", mmap_mode="r", allow_pickle=False)
        visits = np.load(cache_dir / "visits.npy", mmap_mode="r", allow_pickle=False)
        features = np.load(cache_dir / "action_features.npy", mmap_mode="r", allow_pickle=False)
        if int(features.shape[1]) != int(action_dim):
            raise RuntimeError(
                f"indexed action dimension {features.shape[1]} != expected {action_dim}: {cache_dir}"
            )
        for key, local_row in selected:
            begin = int(offsets[local_row])
            end = int(offsets[local_row + 1])
            if end <= begin:
                continue
            root_records.append(
                (
                    key,
                    np.asarray(indices[begin:end], dtype=np.int32).copy(),
                    np.asarray(visits[begin:end], dtype=np.int32).copy(),
                    np.asarray(features[begin:end], dtype=np.float32).copy(),
                )
            )
    root_records.sort(key=lambda item: item[0])
    total_actions = sum(int(record[1].shape[0]) for record in root_records)
    X = np.empty((total_actions, VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32)
    y = np.empty(total_actions, dtype=np.float32)
    probabilities = np.empty(total_actions, dtype=np.float32)
    offsets_out: list[tuple[int, int]] = []
    position = 0
    for key, action_indices, visit_counts, action_features in root_records:
        order = np.argsort(action_indices, kind="stable")
        action_features = action_features[order]
        visit_values = np.maximum(0, visit_counts[order]).astype(np.float64, copy=False)
        total = float(np.sum(visit_values))
        probs = (
            visit_values / total
            if total > 0.0
            else np.full(visit_values.size, 1.0 / float(visit_values.size), dtype=np.float64)
        )
        logits = np.log(visit_values + float(POLICY_ALPHA))
        logits -= float(np.mean(logits))
        count = int(action_features.shape[0])
        end = position + count
        X[position:end, :VALUE_FEATURE_DIM] = roots[key]
        X[position:end, VALUE_FEATURE_DIM:] = action_features
        y[position:end] = logits.astype(np.float32, copy=False)
        probabilities[position:end] = probs.astype(np.float32, copy=False)
        offsets_out.append((position, end))
        position = end
    return (X, y, probabilities, offsets_out), {
        "roots_with_actions": int(len(root_records)),
        "action_rows": int(total_actions),
    }


def indexed_sample_training_data(
    state_paths: Iterable[Path],
    *,
    value_feature_schema: str,
    seed: int,
    controller_value_rows: int,
    adversary_value_rows: int,
    controller_policy_roots: int,
    adversary_policy_roots: int,
    cache_workers: int,
    extraction_workers: int,
) -> tuple[
    list[dict[str, Any]],
    IndexedRoots,
    IndexedRoots,
    dict[str, int],
    dict[str, float | int],
]:
    started = time.perf_counter()
    descriptors, cache_timings = ensure_partition_indexes(
        state_paths,
        value_feature_schema=value_feature_schema,
        workers=cache_workers,
    )
    controller_values, controller_total = sample_addresses(
        descriptors,
        role="controller",
        sample_size=int(controller_value_rows),
        seed=int(seed) + 104,
    )
    adversary_values, adversary_total = sample_addresses(
        descriptors,
        role="adversary",
        sample_size=int(adversary_value_rows),
        seed=int(seed) + 105,
    )
    controller_policy, controller_policy_total = sample_addresses(
        descriptors,
        role="controller",
        sample_size=int(controller_policy_roots),
        seed=int(seed) + 102,
        policy_only=True,
    )
    adversary_policy, adversary_policy_total = sample_addresses(
        descriptors,
        role="adversary",
        sample_size=int(adversary_policy_roots),
        seed=int(seed) + 103,
        policy_only=True,
    )
    selected_at = time.perf_counter()
    controller_sample = materialize_value_samples(
        controller_values,
        value_feature_schema=value_feature_schema,
        workers=extraction_workers,
    )
    adversary_sample = materialize_value_samples(
        adversary_values,
        value_feature_schema=value_feature_schema,
        workers=extraction_workers,
    )
    controller_roots = materialize_policy_roots(
        controller_policy,
        value_feature_schema=value_feature_schema,
        workers=extraction_workers,
    )
    adversary_roots = materialize_policy_roots(
        adversary_policy,
        value_feature_schema=value_feature_schema,
        workers=extraction_workers,
    )
    materialized_at = time.perf_counter()
    counts = {
        "states": int(controller_total + adversary_total),
        "controller": int(controller_total),
        "adversary": int(adversary_total),
        "controller_policy_roots": int(controller_policy_total),
        "adversary_policy_roots": int(adversary_policy_total),
    }
    timings: dict[str, float | int] = {
        **cache_timings,
        "state_cache_files": int(cache_timings["indexed_cache_files"]),
        "state_cache_files_rebuilt": int(cache_timings["indexed_cache_files_rebuilt"]),
        "state_cache_files_missing_at_load": 0,
        "state_cache_ensure_elapsed_s": float(cache_timings["indexed_cache_ensure_elapsed_s"]),
        "state_cache_sample_elapsed_s": float(materialized_at - selected_at),
        "state_cache_total_elapsed_s": float(materialized_at - started),
        "indexed_address_selection_elapsed_s": float(selected_at - started - float(cache_timings["indexed_cache_ensure_elapsed_s"])),
        "indexed_materialize_elapsed_s": float(materialized_at - selected_at),
        "indexed_extraction_workers": int(extraction_workers),
        "indexed_unique_value_addresses": int(len(controller_values) + len(adversary_values)),
        "indexed_unique_policy_addresses": int(len(controller_policy) + len(adversary_policy)),
        "indexed_controller_value_sample_sha256": address_fingerprint(controller_values),
        "indexed_adversary_value_sample_sha256": address_fingerprint(adversary_values),
        "indexed_controller_policy_sample_sha256": address_fingerprint(controller_policy),
        "indexed_adversary_policy_sample_sha256": address_fingerprint(adversary_policy),
    }
    return (
        controller_sample + adversary_sample,
        controller_roots,
        adversary_roots,
        counts,
        timings,
    )
