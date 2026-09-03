"""XL-side AlphaGoZero HGB train/eval/promotion driver.

This script is intentionally conservative: it trains only from replay rows marked
feature_complete=1 and writes candidate artifacts/metrics atomically. Evaluation
and promotion orchestration is isolated here so the ingest loop can launch it as
an asynchronous subprocess.
"""

from __future__ import annotations

import hashlib
import argparse
import csv
import gc
import json
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from vidur.AlphaGoZero.config import (
    ADVERSARY_POLICY_SAMPLE_CAP,
    ADVERSARY_POLICY_ROOT_SAMPLE_TARGET,
    AGZ_DNN_EPOCHS,
    AGZ_DNN_INITIAL_LR,
    AGZ_DNN_POLICY_ROOT_BATCH_SIZE,
    AGZ_DNN_TORCH_THREADS_PER_MODEL,
    AGZ_DNN_UPDATE_LR,
    AGZ_DNN_VALUE_BATCH_SIZE,
    AGZ_BENCHMARK_GAMES,
    AGZ_EVAL_GAMES,
    AGZ_EVAL_PUCT_C,
    AGZ_EVAL_ROLLOUT_COUNT,
    AGZ_EVAL_ROLLOUT_PARALLEL_THREADS,
    AGZ_REPLAY_EXTRACTION_WORKERS,
    AGZ_REPLAY_INDEX_BUILD_WORKERS,
    AGZ_REPLAY_SAMPLER,
    AGZ_SJF_PUCT_C,
    AGZ_SJF_ROLLOUT_COUNT,
    AGZ_SJF_ROLLOUT_PARALLEL_THREADS,
    AGZ_MODEL_FAMILY,
    AGZ_POLICY_FEATURE_SCHEMA,
    AGZ_NATIVE_SEARCH_MODE,
    AGZ_ROLLOUT_COUNT,
    AGZ_ROLLOUT_HORIZON_SEC,
    AGZ_ROLLOUT_MAX_ACTIONS,
    AGZ_ROLLOUT_POLICY_TEMPERATURE,
    AGZ_ROLLOUT_PROBABILITY_QUANTUM,
    AGZ_VALUE_FEATURE_SCHEMA,
    AGZ_DISCOUNT_FACTOR,
    AGZ_EVAL_MCTS_ITERATIONS,
    AGZ_MAX_ADVERSARY_VALUE_STATES,
    XL_CONTROLLER_POLICY_SAMPLE_CAP,
    DEFAULT_ADVERSARY_PRIOR_MODEL_PATH,
    DEFAULT_CONTROLLER_PRIOR_MODEL_PATH,
    DEFAULT_VALUE_MODEL_PATH,
    MIN_CONTROLLER_STATES_FOR_EVAL,
    MIN_ADVERSARY_STATES_FOR_EVAL,
    POLICY_CACHE_BUILD_WORKERS,
    POLICY_METRICS_WORKERS,
    POLICY_ROOT_OVERSAMPLE_FACTOR,
    PROMOTION_WIN_RATE_THRESHOLD,
    ROLE_PROMOTION_WIN_THRESHOLD,
)
from vidur.AlphaGoZero.cluster import REMOTE_OUTPUT_ROOT, REMOTE_REPO, WORKERS
from vidur.AlphaGoZero.distributed_eval import (
    cpu_safe_hosts,
    default_role_hosts,
    default_sjf_hosts,
    merge_arena_block,
    distributed_eval_enabled,
    memory_safe_hosts,
    merge_split_sjf_cycles,
    pause_selfplay_for_eval,
    plan_independent_block_chunks,
    plan_role_chunks,
    plan_single_block_chunks,
    run_distributed_chunks,
    spot_pull_eval_enabled,
)
from vidur.AlphaGoZero.adaptive_rollout import (
    active_rollout_horizon_sec,
    configured_rollout_horizon,
    runtime_search_config_path,
)
from vidur.AlphaGoZero.durable_transfer import append_csv_row, atomic_write_json, local_time_24h, utc_now
from vidur.AlphaGoZero.spot_distributed_eval import run_spot_pull_commands
from vidur.AlphaGoZero.markov_value_features import (
    GLOBAL_DIM as MARKOV_GLOBAL_DIM,
    LAUNCH_DIM as MARKOV_LAUNCH_DIM,
    MARKOV_VALUE_SCHEMA,
    REQUEST_DIM as MARKOV_REQUEST_DIM,
    MarkovValueFeatures,
    features_from_replay_row,
)
from vidur.bellman_v4_adv.arena_mcts_value_runnerCPP import _export_hgb_to_native_text

HGB_CONFIG_NAME = "hgb_sq_63leaf_1050iter_a2"
HGB_POLICY_CONFIG_NAME = "hgb_policy_63leaf_1050iter"
if AGZ_VALUE_FEATURE_SCHEMA not in {"legacy_226", MARKOV_VALUE_SCHEMA}:
    raise ValueError(f"unsupported AGZ_VALUE_FEATURE_SCHEMA={AGZ_VALUE_FEATURE_SCHEMA!r}")
if AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA and AGZ_MODEL_FAMILY != "dnn":
    raise ValueError("markov_v2 value features require AGZ_MODEL_FAMILY=dnn")
if AGZ_POLICY_FEATURE_SCHEMA not in {"legacy_226", MARKOV_VALUE_SCHEMA}:
    raise ValueError(f"unsupported AGZ_POLICY_FEATURE_SCHEMA={AGZ_POLICY_FEATURE_SCHEMA!r}")
if AGZ_POLICY_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA:
    if AGZ_MODEL_FAMILY != "dnn":
        raise ValueError("markov_v2 policy features require AGZ_MODEL_FAMILY=dnn")
    if AGZ_VALUE_FEATURE_SCHEMA != MARKOV_VALUE_SCHEMA:
        raise ValueError("markov_v2 policy features require markov_v2 indexed state caches")
    if AGZ_REPLAY_SAMPLER != "indexed_v1":
        raise ValueError("markov_v2 policy features require AGZ_REPLAY_SAMPLER=indexed_v1")

CONFIG_NAME = (
    "dnn_value_markov_deepset_192_v2"
    if AGZ_MODEL_FAMILY == "dnn" and AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA
    else ("dnn_value_residual_192_v1" if AGZ_MODEL_FAMILY == "dnn" else HGB_CONFIG_NAME)
)
POLICY_CONFIG_NAME = (
    "dnn_policy_markov_deepset_192_v3"
    if AGZ_MODEL_FAMILY == "dnn" and AGZ_POLICY_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA
    else ("dnn_policy_rank_192_v1" if AGZ_MODEL_FAMILY == "dnn" else HGB_POLICY_CONFIG_NAME)
)
VALUE_FEATURE_DIM = 226
CTRL_ACTION_DIM = 43
ADV_ACTION_DIM = 7
POLICY_ALPHA = 1.0
DEFAULT_HGB_OPENMP_THREADS = 16
HGB_OPENMP_THREADS_ENV = "AGZ_HGB_OPENMP_THREADS"

TRAIN_MODEL_FIELDS = [
    "model_config", "model_version", "time_of_training_24h", "sampled_controller_states", "sampled_adversary_states",
    "value_mse", "value_rmse", "value_max_abs_error", "value_p95_abs_error",
    "controller_value_mse", "controller_value_rmse", "controller_value_max_abs_error", "controller_value_p95_abs_error",
    "adversary_value_mse", "adversary_value_rmse", "adversary_value_max_abs_error", "adversary_value_p95_abs_error",
    "controller_policy_mse", "controller_policy_cross_entropy", "controller_policy_top1", "controller_policy_top3",
    "adversary_policy_mse", "adversary_policy_cross_entropy", "adversary_policy_top1", "adversary_policy_top3",
    "rollout_horizon_used_sec", "rollout_horizon_source_controller_p95_abs_error",
    "rollout_value_error_threshold", "rollout_discount_factor", "rollout_reference_step_sec",
    "rollout_max_horizon_sec", "rollout_horizon_tick_sec", "next_rollout_horizon_raw_sec",
    "next_rollout_horizon_calculated_sec", "next_rollout_horizon_rounded_sec",
    "next_rollout_discounted_error",
]


TESTING_ADVERSARY_MODE = "Testing Adversary"
TESTING_CONTROLLER_MODE = "Testing Controller"

def _progress(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "time_utc": utc_now()}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True), flush=True)


EVAL_GAME_FIELDS = [
    "promoted_controller_model_version",
    "promoted_adversary_model_version",
    "candidate_model_version",
    "new_controller_model_version",
    "new_adversary_model_version",
    "candidate_win_ratio_in_100_games",
    "candidate_controller_promoted",
    "candidate_adversary_promoted",
    "candidate_promoted",
    "time_of_eval",
    "game_mode",
    "game_id",
    "game_hop_number",
    "total_cost_when_promoted_adv_and_promoted_controller",
    "total_cost_when_candidate_adv_and_promoted_controller",
    "total_cost_when_promoted_adv_and_candidate_controller",
    "total_cost_when_candidate_adv_and_candidate_controller",
]


@dataclass(frozen=True)
class ModelBundle:
    model_version: int
    controller_model_version: int
    adversary_model_version: int
    controller_value_model_path: Path
    adversary_value_model_path: Path
    controller_prior_model_path: Path
    adversary_prior_model_path: Path

    @property
    def value_model_path(self) -> Path:
        # Legacy callers pass a single model path; use the controller value by default.
        return self.controller_value_model_path

    def to_json(self) -> dict[str, Any]:
        model_family = "dnn" if "dnn_" in str(self.controller_value_model_path).lower() else "hgb"
        return {
            "model_version": int(max(self.controller_model_version, self.adversary_model_version, self.model_version)),
            "controller_model_version": int(self.controller_model_version),
            "adversary_model_version": int(self.adversary_model_version),
            "model_family": model_family,
            "native_ready": True,
            "incremental_update": bool(model_family == "dnn" and int(self.model_version) > 100),
            "target_perspective": "controller",
            "value_model_path": str(self.controller_value_model_path),
            "controller_value_model_path": str(self.controller_value_model_path),
            "adversary_value_model_path": str(self.adversary_value_model_path),
            "controller_prior_model_path": str(self.controller_prior_model_path),
            "adversary_prior_model_path": str(self.adversary_prior_model_path),
            "updated_at_utc": utc_now(),
        }


def _read_json_list(raw: str) -> list[float]:
    if not raw:
        return []
    try:
        vals = json.loads(raw)
    except Exception:
        return []
    if not isinstance(vals, list):
        return []
    out: list[float] = []
    for v in vals:
        try:
            x = float(v)
            if math.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return out


def _root_key(row: dict[str, str]) -> tuple[str, str, int, int, int, str]:
    return (
        str(row.get("accepted_shard_id", "")),
        str(row.get("source_worker_id", "")),
        int(float(row.get("game_id", 0) or 0)),
        int(float(row.get("turn_number", 0) or 0)),
        int(float(row.get("depth_number", 0) or 0)),
        str(row.get("player", "")),
    )


def _load_feature_state_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True"}:
                continue
            features = _read_json_list(row.get("state_features_json", ""))
            if len(features) != VALUE_FEATURE_DIM:
                continue
            item = dict(row)
            item["_key"] = _root_key(row)
            item["_state_features"] = features
            item["_target_value"] = float(row.get("target_value", 0.0) or 0.0)
            rows.append(item)
    return rows


def _load_policy_rows(path: Path) -> dict[tuple[str, str, int, int, int, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str, int, int, int, str], list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            player = str(row.get("player", ""))
            feats = _read_json_list(row.get("action_features_json", ""))
            if player == "controller" and len(feats) != CTRL_ACTION_DIM:
                continue
            if player == "adversary" and len(feats) != ADV_ACTION_DIM:
                continue
            item = dict(row)
            item["_key"] = _root_key(row)
            item["_action_features"] = feats
            item["_visit_count"] = int(float(row.get("visit_count", 0) or 0))
            item["_mcts_visit_prob"] = float(row.get("mcts_visit_prob", 0.0) or 0.0)
            out.setdefault(item["_key"], []).append(item)
    for rows in out.values():
        rows.sort(key=lambda r: int(float(r.get("canon_action_index", -1) or -1)))
    return out


def _replay_path_snapshot(root: Path) -> tuple[list[Path], list[Path]]:
    """Return one consistent snapshot of fully committed replay partitions."""

    root = Path(root)
    partitions = root / "global_replay" / "partitions"
    feature_paths: list[Path] = []
    policy_paths: list[Path] = []
    # The coordinator writes the manifest last via atomic rename. Files visible
    # before that point belong to an in-progress partition and must not be read.
    for manifest_path in sorted(partitions.glob("*/*/partition_manifest.json")):
        partition = manifest_path.parent
        feature_path = partition / "replay_target_runtime_feature_complete.csv"
        policy_path = partition / "replay_policy_rows.csv"
        if not feature_path.is_file() or not policy_path.is_file():
            continue
        feature_paths.append(feature_path)
        policy_paths.append(policy_path)
    if feature_paths:
        return feature_paths, policy_paths

    legacy_feature = root / "global_replay" / "replay_target_runtime_feature_complete.csv"
    legacy_policy = root / "global_replay" / "replay_policy_rows.csv"
    if legacy_feature.is_file() and legacy_policy.is_file():
        return [legacy_feature], [legacy_policy]
    return [], []


def _feature_replay_paths(root: Path) -> list[Path]:
    return _replay_path_snapshot(root)[0]


def _policy_replay_paths(root: Path) -> list[Path]:
    return _replay_path_snapshot(root)[1]


STATE_CACHE_VERSION = 2
POLICY_CACHE_VERSION = 1
KEY_SEP = "\x1f"


def _key_to_string(key: tuple[str, str, int, int, int, str]) -> str:
    return KEY_SEP.join((str(key[0]), str(key[1]), str(int(key[2])), str(int(key[3])), str(int(key[4])), str(key[5])))


def _key_from_string(raw: str) -> tuple[str, str, int, int, int, str]:
    parts = str(raw).split(KEY_SEP)
    if len(parts) != 6:
        raise ValueError(f"invalid cache key: {raw!r}")
    return (parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), parts[5])


def _cache_paths(source: Path, suffix: str) -> tuple[Path, Path]:
    source = Path(source)
    return source.with_name(source.name + f".{suffix}.npz"), source.with_name(source.name + f".{suffix}.json")


def _source_signature(source: Path) -> dict[str, int]:
    st = Path(source).stat()
    return {"source_size": int(st.st_size), "source_mtime_ns": int(st.st_mtime_ns)}


def _cache_is_fresh(source: Path, meta_path: Path, *, kind: str, version: int) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    sig = _source_signature(source)
    return (
        str(meta.get("kind")) == str(kind)
        and int(meta.get("version", -1)) == int(version)
        and int(meta.get("source_size", -1)) == int(sig["source_size"])
        and int(meta.get("source_mtime_ns", -1)) == int(sig["source_mtime_ns"])
    )


def _write_cache_metadata(source: Path, meta_path: Path, *, kind: str, version: int, extra: dict[str, Any]) -> None:
    meta = {
        "kind": str(kind),
        "version": int(version),
        "source_path": str(source),
        **_source_signature(source),
        "created_at_utc": utc_now(),
        **extra,
    }
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, meta_path)


def _state_cache_suffix() -> str:
    schema = str(AGZ_VALUE_FEATURE_SCHEMA).replace("/", "_")
    return f"state_cache_{schema}_v{STATE_CACHE_VERSION}"


def _ensure_state_cache(path: Path) -> tuple[Path, bool, float]:
    t0 = time.time()
    path = Path(path)
    cache_path, meta_path = _cache_paths(path, _state_cache_suffix())
    if cache_path.exists() and _cache_is_fresh(path, meta_path, kind="state", version=STATE_CACHE_VERSION):
        return cache_path, False, float(time.time() - t0)

    keys: list[str] = []
    players: list[int] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    value_globals: list[np.ndarray] = []
    request_offsets = [0]
    request_features: list[np.ndarray] = []
    launch_offsets = [0]
    launch_features: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item = _minimal_state_row(row)
            if item is None:
                continue
            player = str(item["player"])
            if player == "controller":
                player_code = 0
            elif player == "adversary":
                player_code = 1
            else:
                continue
            keys.append(_key_to_string(item["_key"]))
            players.append(player_code)
            features.append(np.asarray(item["_state_features"], dtype=np.float32))
            targets.append(float(item["_target_value"]))
            if AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA:
                value = item["_value_features"]
                if not isinstance(value, MarkovValueFeatures):
                    raise TypeError("Markov replay row did not produce structured features")
                value_globals.append(value.global_features)
                request_features.extend(value.request_features)
                launch_features.extend(value.launch_features)
            request_offsets.append(len(request_features))
            launch_offsets.append(len(launch_features))

    feature_arr = np.vstack(features).astype(np.float32, copy=False) if features else np.empty((0, VALUE_FEATURE_DIM), dtype=np.float32)
    global_arr = np.vstack(value_globals).astype(np.float32, copy=False) if value_globals else np.empty((0, MARKOV_GLOBAL_DIM), dtype=np.float32)
    request_arr = np.vstack(request_features).astype(np.float32, copy=False) if request_features else np.empty((0, MARKOV_REQUEST_DIM), dtype=np.float32)
    launch_arr = np.vstack(launch_features).astype(np.float32, copy=False) if launch_features else np.empty((0, MARKOV_LAUNCH_DIM), dtype=np.float32)
    tmp_cache = cache_path.with_name(cache_path.name + ".tmp")
    with tmp_cache.open("wb") as f:
        np.savez(
            f,
            keys=np.asarray(keys, dtype=np.str_),
            players=np.asarray(players, dtype=np.int8),
            features=feature_arr,
            targets=np.asarray(targets, dtype=np.float32),
            value_feature_schema=np.asarray([AGZ_VALUE_FEATURE_SCHEMA], dtype=np.str_),
            value_globals=global_arr,
            request_offsets=np.asarray(request_offsets, dtype=np.int64),
            request_features=request_arr,
            launch_offsets=np.asarray(launch_offsets, dtype=np.int64),
            launch_features=launch_arr,
        )
    os.replace(tmp_cache, cache_path)
    _write_cache_metadata(
        path,
        meta_path,
        kind="state",
        version=STATE_CACHE_VERSION,
        extra={
            "rows": int(len(keys)),
            "policy_state_feature_dim": int(VALUE_FEATURE_DIM),
            "value_feature_schema": str(AGZ_VALUE_FEATURE_SCHEMA),
            "request_rows": int(request_arr.shape[0]),
            "launch_rows": int(launch_arr.shape[0]),
        },
    )
    return cache_path, True, float(time.time() - t0)


def _ensure_state_cache_worker(path: str) -> tuple[str, bool, float]:
    cache_path, did_rebuild, elapsed = _ensure_state_cache(Path(path))
    return str(cache_path), bool(did_rebuild), float(elapsed)


def _policy_arrays_from_grouped(
    grouped: dict[str, list[tuple[int, np.ndarray, int]]],
    *,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    action_indices: list[int] = []
    visits_out: list[int] = []
    features: list[np.ndarray] = []
    pos = 0
    for key in sorted(grouped.keys()):
        actions = sorted(grouped[key], key=lambda x: x[0])
        keys.append(key)
        starts.append(pos)
        for idx, feats, visits in actions:
            action_indices.append(int(idx))
            visits_out.append(int(visits))
            features.append(np.asarray(feats, dtype=np.float32))
            pos += 1
        ends.append(pos)
    feature_arr = (
        np.vstack(features).astype(np.float32, copy=False)
        if features
        else np.empty((0, int(action_dim)), dtype=np.float32)
    )
    return (
        np.asarray(keys, dtype=np.str_),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(action_indices, dtype=np.int32),
        np.asarray(visits_out, dtype=np.int32),
        feature_arr,
    )


def _ensure_policy_cache(path: Path) -> tuple[Path, bool, float]:
    t0 = time.time()
    path = Path(path)
    cache_path, meta_path = _cache_paths(path, "policy_cache_v1")
    if cache_path.exists() and _cache_is_fresh(path, meta_path, kind="policy", version=POLICY_CACHE_VERSION):
        return cache_path, False, float(time.time() - t0)

    ctrl: dict[str, list[tuple[int, np.ndarray, int]]] = {}
    adv: dict[str, list[tuple[int, np.ndarray, int]]] = {}
    rows_read = 0
    rows_kept = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_read += 1
            player = str(row.get("player", "")).lower()
            if player == "controller":
                action_dim = CTRL_ACTION_DIM
                grouped = ctrl
            elif player == "adversary":
                action_dim = ADV_ACTION_DIM
                grouped = adv
            else:
                continue
            feats = _read_json_list(row.get("action_features_json", ""))
            if len(feats) != int(action_dim):
                continue
            try:
                idx = int(float(row.get("canon_action_index", -1) or -1))
            except Exception:
                idx = -1
            try:
                visits = int(float(row.get("visit_count", 0) or 0))
            except Exception:
                visits = 0
            grouped.setdefault(_key_to_string(_root_key(row)), []).append((idx, np.asarray(feats, dtype=np.float32), visits))
            rows_kept += 1

    ck, cs, ce, cai, cv, cf = _policy_arrays_from_grouped(ctrl, action_dim=CTRL_ACTION_DIM)
    ak, ass, ae, aai, av, af = _policy_arrays_from_grouped(adv, action_dim=ADV_ACTION_DIM)
    tmp_cache = cache_path.with_name(cache_path.name + ".tmp")
    with tmp_cache.open("wb") as f:
        np.savez(
            f,
            ctrl_keys=ck,
            ctrl_starts=cs,
            ctrl_ends=ce,
            ctrl_action_indices=cai,
            ctrl_visits=cv,
            ctrl_features=cf,
            adv_keys=ak,
            adv_starts=ass,
            adv_ends=ae,
            adv_action_indices=aai,
            adv_visits=av,
            adv_features=af,
        )
    os.replace(tmp_cache, cache_path)
    _write_cache_metadata(
        path,
        meta_path,
        kind="policy",
        version=POLICY_CACHE_VERSION,
        extra={
            "rows_read": int(rows_read),
            "rows_kept": int(rows_kept),
            "controller_roots": int(len(ck)),
            "controller_action_rows": int(cai.size),
            "adversary_roots": int(len(ak)),
            "adversary_action_rows": int(aai.size),
            "controller_action_dim": int(CTRL_ACTION_DIM),
            "adversary_action_dim": int(ADV_ACTION_DIM),
        },
    )
    return cache_path, True, float(time.time() - t0)


def _ensure_policy_cache_worker(path: str) -> tuple[str, bool, float]:
    cache_path, did_rebuild, elapsed = _ensure_policy_cache(Path(path))
    return str(cache_path), bool(did_rebuild), float(elapsed)


def _policy_root_sample_target(cap: int, *, explicit_target: int = 0) -> int:
    if int(explicit_target) > 0:
        return max(int(cap), int(explicit_target))
    factor = max(1.0, float(POLICY_ROOT_OVERSAMPLE_FACTOR))
    return max(int(cap), int(math.ceil(float(cap) * factor)))


def _available_policy_root_requirement(available_roots: int, sample_cap: int) -> int:
    """Use every available actionable root up to the configured maximum."""

    return min(max(0, int(available_roots)), max(0, int(sample_cap)))


def _cap_actions_by_root_order(
    actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]],
    roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
    cap: int | None,
) -> dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]]:
    if cap is None or int(cap) <= 0 or len(actions_by_key) <= int(cap):
        return actions_by_key
    limited: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]] = {}
    for key in roots:
        actions = actions_by_key.get(key)
        if actions is None:
            continue
        limited[key] = actions
        if len(limited) >= int(cap):
            break
    return limited


def _reservoir_slot(sample_len: int, *, seen: int, max_rows: int, rng: random.Random) -> int | None:
    if max_rows <= 0:
        return None
    if int(sample_len) < int(max_rows):
        return int(sample_len)
    j = rng.randrange(int(seen))
    return int(j) if j < int(max_rows) else None


def _candidate_sampling_seed(seed: int, candidate_version: int) -> int:
    return int(seed) + int(candidate_version)


def _value_role_sample_targets(total_rows: int, adversary_cap: int) -> tuple[int, int]:
    total = max(0, int(total_rows))
    adversary = min(total, max(0, int(adversary_cap)))
    controller = total - adversary
    return controller, adversary


def _stream_sample_state_rows_cached(
    paths: list[Path],
    *,
    seed: int,
    max_value_rows: int,
    max_controller_policy_roots: int,
    max_adversary_policy_roots: int,
    max_controller_value_rows: int | None = None,
    max_adversary_value_rows: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int, int, int, str], np.ndarray],
    dict[tuple[str, str, int, int, int, str], np.ndarray],
    dict[str, int],
    dict[str, float | int],
]:
    t_total = time.time()
    t_ensure = time.time()
    cache_paths: list[Path] = []
    rebuilt = 0
    missing_or_stale_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        cache_path, meta_path = _cache_paths(path, _state_cache_suffix())
        if cache_path.exists() and _cache_is_fresh(
            path, meta_path, kind="state", version=STATE_CACHE_VERSION
        ):
            cache_paths.append(cache_path)
        else:
            missing_or_stale_paths.append(path)
    build_workers = max(1, int(POLICY_CACHE_BUILD_WORKERS))
    if missing_or_stale_paths and build_workers > 1:
        workers = min(build_workers, len(missing_or_stale_paths))
        results_by_source: dict[Path, tuple[Path, bool, float]] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_ensure_state_cache_worker, str(path)): path
                for path in missing_or_stale_paths
            }
            for future in as_completed(futures):
                source = futures[future]
                cache_path_str, did_rebuild, elapsed = future.result()
                results_by_source[source] = (Path(cache_path_str), bool(did_rebuild), float(elapsed))
        for path in missing_or_stale_paths:
            cache_path, did_rebuild, _ = results_by_source[path]
            cache_paths.append(cache_path)
            rebuilt += int(did_rebuild)
    else:
        for path in missing_or_stale_paths:
            cache_path, did_rebuild, _ = _ensure_state_cache(path)
            cache_paths.append(cache_path)
            rebuilt += int(did_rebuild)
    ensure_elapsed = float(time.time() - t_ensure)

    rng_value = random.Random(int(seed) + 101)
    rng_ctrl = random.Random(int(seed) + 102)
    rng_adv = random.Random(int(seed) + 103)
    rng_controller_value = random.Random(int(seed) + 104)
    rng_adversary_value = random.Random(int(seed) + 105)
    value_sample: list[dict[str, Any]] = []
    controller_value_sample: list[dict[str, Any]] = []
    adversary_value_sample: list[dict[str, Any]] = []
    ctrl_sample: list[dict[str, Any]] = []
    adv_sample: list[dict[str, Any]] = []
    counts = {"states": 0, "controller": 0, "adversary": 0}
    seen_value = 0
    seen_controller_value = 0
    seen_adversary_value = 0
    seen_ctrl = 0
    seen_adv = 0
    role_value_caps = max_controller_value_rows is not None and max_adversary_value_rows is not None

    def make_item(
        keys: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        players: np.ndarray,
        value_globals: np.ndarray,
        request_offsets: np.ndarray,
        request_features: np.ndarray,
        launch_offsets: np.ndarray,
        launch_features: np.ndarray,
        idx: int,
    ) -> dict[str, Any]:
        player = "controller" if int(players[idx]) == 0 else "adversary"
        state_features = np.asarray(features[idx], dtype=np.float32).copy()
        if AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA:
            request_begin = int(request_offsets[idx])
            request_end = int(request_offsets[idx + 1])
            launch_begin = int(launch_offsets[idx])
            launch_end = int(launch_offsets[idx + 1])
            value_features: np.ndarray | MarkovValueFeatures = MarkovValueFeatures(
                global_features=np.asarray(value_globals[idx], dtype=np.float32).copy(),
                request_features=np.asarray(request_features[request_begin:request_end], dtype=np.float32).copy(),
                launch_features=np.asarray(launch_features[launch_begin:launch_end], dtype=np.float32).copy(),
                request_ids=tuple(range(request_end - request_begin)),
            )
        else:
            value_features = state_features
        return {
            "_key": _key_from_string(str(keys[idx])),
            "_state_features": state_features,
            "_value_features": value_features,
            "_target_value": float(targets[idx]),
            "player": player,
        }

    t_sample = time.time()
    missing_at_state_load = 0
    for cache_path in cache_paths:
        if not cache_path.exists():
            missing_at_state_load += 1
            continue
        with np.load(cache_path, allow_pickle=False) as data:
            keys = data["keys"]
            players = data["players"]
            features = data["features"]
            targets = data["targets"]
            cached_schema = str(data["value_feature_schema"][0])
            if cached_schema != AGZ_VALUE_FEATURE_SCHEMA:
                raise RuntimeError(f"state cache schema {cached_schema!r} != {AGZ_VALUE_FEATURE_SCHEMA!r}")
            value_globals = data["value_globals"]
            request_offsets = data["request_offsets"]
            request_features = data["request_features"]
            launch_offsets = data["launch_offsets"]
            launch_features = data["launch_features"]
            for i in range(int(keys.shape[0])):
                player_code = int(players[i])
                counts["states"] += 1
                seen_value += 1
                value_bucket = value_sample
                if not role_value_caps:
                    value_slot = _reservoir_slot(
                        len(value_sample),
                        seen=seen_value,
                        max_rows=int(max_value_rows),
                        rng=rng_value,
                    )
                elif player_code == 0:
                    seen_controller_value += 1
                    value_bucket = controller_value_sample
                    value_slot = _reservoir_slot(
                        len(controller_value_sample),
                        seen=seen_controller_value,
                        max_rows=int(max_controller_value_rows),
                        rng=rng_controller_value,
                    )
                elif player_code == 1:
                    seen_adversary_value += 1
                    value_bucket = adversary_value_sample
                    value_slot = _reservoir_slot(
                        len(adversary_value_sample),
                        seen=seen_adversary_value,
                        max_rows=int(max_adversary_value_rows),
                        rng=rng_adversary_value,
                    )
                else:
                    value_slot = None
                need_item = value_slot is not None
                ctrl_slot: int | None = None
                adv_slot: int | None = None
                if player_code == 0:
                    counts["controller"] += 1
                    seen_ctrl += 1
                    ctrl_slot = _reservoir_slot(
                        len(ctrl_sample),
                        seen=seen_ctrl,
                        max_rows=int(max_controller_policy_roots),
                        rng=rng_ctrl,
                    )
                    need_item = need_item or ctrl_slot is not None
                elif player_code == 1:
                    counts["adversary"] += 1
                    seen_adv += 1
                    adv_slot = _reservoir_slot(
                        len(adv_sample),
                        seen=seen_adv,
                        max_rows=int(max_adversary_policy_roots),
                        rng=rng_adv,
                    )
                    need_item = need_item or adv_slot is not None
                if not need_item:
                    continue
                item = make_item(
                    keys,
                    features,
                    targets,
                    players,
                    value_globals,
                    request_offsets,
                    request_features,
                    launch_offsets,
                    launch_features,
                    i,
                )
                if value_slot is not None:
                    if value_slot == len(value_bucket):
                        value_bucket.append(item)
                    else:
                        value_bucket[value_slot] = item
                if ctrl_slot is not None:
                    ctrl_item = (
                        item
                        if str(item["player"]) == "controller"
                        else make_item(
                            keys,
                            features,
                            targets,
                            players,
                            value_globals,
                            request_offsets,
                            request_features,
                            launch_offsets,
                            launch_features,
                            i,
                        )
                    )
                    if ctrl_slot == len(ctrl_sample):
                        ctrl_sample.append(ctrl_item)
                    else:
                        ctrl_sample[ctrl_slot] = ctrl_item
                if adv_slot is not None:
                    adv_item = (
                        item
                        if str(item["player"]) == "adversary"
                        else make_item(
                            keys,
                            features,
                            targets,
                            players,
                            value_globals,
                            request_offsets,
                            request_features,
                            launch_offsets,
                            launch_features,
                            i,
                        )
                    )
                    if adv_slot == len(adv_sample):
                        adv_sample.append(adv_item)
                    else:
                        adv_sample[adv_slot] = adv_item
    if role_value_caps:
        value_sample = controller_value_sample + adversary_value_sample
    sample_elapsed = float(time.time() - t_sample)
    ctrl_roots = {r["_key"]: r["_state_features"] for r in ctrl_sample}
    adv_roots = {r["_key"]: r["_state_features"] for r in adv_sample}
    timings: dict[str, float | int] = {
        "state_cache_files": int(len(cache_paths)),
        "state_cache_files_rebuilt": int(rebuilt),
        "state_cache_files_missing_at_load": int(missing_at_state_load),
        "state_cache_ensure_elapsed_s": float(ensure_elapsed),
        "state_cache_sample_elapsed_s": float(sample_elapsed),
        "state_cache_total_elapsed_s": float(time.time() - t_total),
    }
    return value_sample, ctrl_roots, adv_roots, counts, timings


def _append_cached_actions(
    actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]],
    selected: dict[str, tuple[str, str, int, int, int, str]],
    *,
    keys: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    action_indices: np.ndarray,
    visits: np.ndarray,
    features: np.ndarray,
) -> tuple[int, int]:
    roots_seen = 0
    action_rows = 0
    for root_idx, raw_key in enumerate(keys):
        key = str(raw_key)
        selected_key = selected.get(key)
        if selected_key is None:
            continue
        start = int(starts[root_idx])
        end = int(ends[root_idx])
        if end <= start:
            continue
        roots_seen += 1
        action_rows += end - start
        bucket = actions_by_key.setdefault(selected_key, [])
        for row_idx in range(start, end):
            bucket.append((
                int(action_indices[row_idx]),
                np.asarray(features[row_idx], dtype=np.float32).copy(),
                int(visits[row_idx]),
            ))
    return roots_seen, action_rows


def _collect_policy_training_arrays_for_players_cached(
    paths: list[Path],
    controller_roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
    adversary_roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
    *,
    controller_root_cap: int | None = None,
    adversary_root_cap: int | None = None,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]],
    tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]],
    dict[str, float | int],
]:
    t_total = time.time()
    t_ensure = time.time()
    cache_paths: list[Path] = []
    rebuilt = 0
    missing_or_stale_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        cache_path, meta_path = _cache_paths(path, "policy_cache_v1")
        if cache_path.exists() and _cache_is_fresh(path, meta_path, kind="policy", version=POLICY_CACHE_VERSION):
            cache_paths.append(cache_path)
        else:
            missing_or_stale_paths.append(path)

    build_workers = max(1, int(POLICY_CACHE_BUILD_WORKERS))
    parallel_build_elapsed = 0.0
    if missing_or_stale_paths:
        if build_workers <= 1:
            for path in missing_or_stale_paths:
                cache_path, did_rebuild, _ = _ensure_policy_cache(path)
                cache_paths.append(cache_path)
                rebuilt += int(did_rebuild)
        else:
            workers = min(int(build_workers), len(missing_or_stale_paths))
            t_parallel = time.time()
            results_by_source: dict[Path, tuple[Path, bool, float]] = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_ensure_policy_cache_worker, str(path)): path
                    for path in missing_or_stale_paths
                }
                for future in as_completed(futures):
                    source_path = futures[future]
                    cache_path_str, did_rebuild, elapsed = future.result()
                    results_by_source[source_path] = (Path(cache_path_str), bool(did_rebuild), float(elapsed))
            parallel_build_elapsed = float(time.time() - t_parallel)
            for path in missing_or_stale_paths:
                cache_path, did_rebuild, _ = results_by_source[path]
                cache_paths.append(cache_path)
                rebuilt += int(did_rebuild)
    ensure_elapsed = float(time.time() - t_ensure)

    ctrl_selected = {_key_to_string(k): k for k in controller_roots}
    adv_selected = {_key_to_string(k): k for k in adversary_roots}
    ctrl_actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]] = {}
    adv_actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]] = {}
    ctrl_roots_seen = 0
    adv_roots_seen = 0
    ctrl_action_rows = 0
    adv_action_rows = 0
    cached_action_rows_seen = 0
    cached_roots_seen = 0

    t_lookup = time.time()
    missing_at_policy_load = 0
    for cache_path in cache_paths:
        if not cache_path.exists():
            missing_at_policy_load += 1
            continue
        with np.load(cache_path, allow_pickle=False) as data:
            cached_action_rows_seen += int(data["ctrl_action_indices"].shape[0]) + int(data["adv_action_indices"].shape[0])
            cached_roots_seen += int(data["ctrl_keys"].shape[0]) + int(data["adv_keys"].shape[0])
            roots, rows = _append_cached_actions(
                ctrl_actions_by_key,
                ctrl_selected,
                keys=data["ctrl_keys"],
                starts=data["ctrl_starts"],
                ends=data["ctrl_ends"],
                action_indices=data["ctrl_action_indices"],
                visits=data["ctrl_visits"],
                features=data["ctrl_features"],
            )
            ctrl_roots_seen += roots
            ctrl_action_rows += rows
            roots, rows = _append_cached_actions(
                adv_actions_by_key,
                adv_selected,
                keys=data["adv_keys"],
                starts=data["adv_starts"],
                ends=data["adv_ends"],
                action_indices=data["adv_action_indices"],
                visits=data["adv_visits"],
                features=data["adv_features"],
            )
            adv_roots_seen += roots
            adv_action_rows += rows
    lookup_elapsed = float(time.time() - t_lookup)
    ctrl_roots_available = int(len(ctrl_actions_by_key))
    adv_roots_available = int(len(adv_actions_by_key))
    ctrl_actions_by_key = _cap_actions_by_root_order(ctrl_actions_by_key, controller_roots, controller_root_cap)
    adv_actions_by_key = _cap_actions_by_root_order(adv_actions_by_key, adversary_roots, adversary_root_cap)

    t_materialize = time.time()
    ctrl_arrays = _materialize_policy_training_arrays(
        controller_roots,
        ctrl_actions_by_key,
        action_dim=CTRL_ACTION_DIM,
    )
    adv_arrays = _materialize_policy_training_arrays(
        adversary_roots,
        adv_actions_by_key,
        action_dim=ADV_ACTION_DIM,
    )
    materialize_elapsed = float(time.time() - t_materialize)
    timings: dict[str, float | int] = {
        "policy_cache_files": int(len(cache_paths)),
        "policy_cache_files_rebuilt": int(rebuilt),
        "policy_cache_build_workers": int(build_workers),
        "policy_cache_files_submitted_for_build": int(len(missing_or_stale_paths)),
        "policy_cache_parallel_build_elapsed_s": float(parallel_build_elapsed),
        "policy_cache_files_missing_at_load": int(missing_at_policy_load),
        "policy_cache_ensure_elapsed_s": float(ensure_elapsed),
        "policy_cache_lookup_elapsed_s": float(lookup_elapsed),
        "policy_cache_total_elapsed_s": float(time.time() - t_total),
        "policy_cache_roots_seen": int(cached_roots_seen),
        "policy_cache_action_rows_seen": int(cached_action_rows_seen),
        "policy_replay_rows_scanned": int(cached_action_rows_seen),
        "policy_replay_scan_elapsed_s": float(lookup_elapsed),
        "policy_array_materialize_elapsed_s": float(materialize_elapsed),
        "controller_policy_roots_with_actions_available": int(ctrl_roots_available),
        "adversary_policy_roots_with_actions_available": int(adv_roots_available),
        "controller_policy_roots_with_actions": int(len(ctrl_actions_by_key)),
        "adversary_policy_roots_with_actions": int(len(adv_actions_by_key)),
        "controller_policy_action_rows": int(ctrl_arrays[0].shape[0]),
        "adversary_policy_action_rows": int(adv_arrays[0].shape[0]),
    }
    return ctrl_arrays, adv_arrays, timings


def _reservoir_add(
    sample: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    seen: int,
    max_rows: int,
    rng: random.Random,
) -> None:
    if max_rows <= 0:
        return
    if len(sample) < int(max_rows):
        sample.append(item)
        return
    j = rng.randrange(int(seen))
    if j < int(max_rows):
        sample[j] = item


def _minimal_state_row(row: dict[str, str]) -> dict[str, Any] | None:
    if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True", ""}:
        return None
    features = _read_json_list(row.get("state_features_json", ""))
    if len(features) != VALUE_FEATURE_DIM:
        return None
    if AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA:
        if str(row.get("value_feature_complete", "0")).strip() not in {"1", "true", "True"}:
            return None
        try:
            value_features: np.ndarray | MarkovValueFeatures = features_from_replay_row(row)
        except ValueError:
            return None
    else:
        value_features = np.asarray(features, dtype=np.float32)
    try:
        target = float(row.get("target_value", 0.0) or 0.0)
    except Exception:
        target = 0.0
    player = str(row.get("player", row.get("root_player", ""))).lower()
    return {
        "_key": _root_key(row),
        "_state_features": np.asarray(features, dtype=np.float32),
        "_value_features": value_features,
        "_target_value": float(target),
        "player": player,
    }


def _stream_sample_state_rows(
    paths: list[Path],
    *,
    seed: int,
    max_value_rows: int,
    max_controller_policy_roots: int,
    max_adversary_policy_roots: int,
    max_controller_value_rows: int | None = None,
    max_adversary_value_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, int, int, str], np.ndarray], dict[tuple[str, str, int, int, int, str], np.ndarray], dict[str, int]]:
    rng_value = random.Random(int(seed) + 101)
    rng_ctrl = random.Random(int(seed) + 102)
    rng_adv = random.Random(int(seed) + 103)
    rng_controller_value = random.Random(int(seed) + 104)
    rng_adversary_value = random.Random(int(seed) + 105)
    value_sample: list[dict[str, Any]] = []
    controller_value_sample: list[dict[str, Any]] = []
    adversary_value_sample: list[dict[str, Any]] = []
    ctrl_sample: list[dict[str, Any]] = []
    adv_sample: list[dict[str, Any]] = []
    counts = {"states": 0, "controller": 0, "adversary": 0}
    seen_value = 0
    seen_controller_value = 0
    seen_adversary_value = 0
    seen_ctrl = 0
    seen_adv = 0
    role_value_caps = max_controller_value_rows is not None and max_adversary_value_rows is not None
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item = _minimal_state_row(row)
                if item is None:
                    continue
                counts["states"] += 1
                player = str(item["player"])
                if player == "controller":
                    counts["controller"] += 1
                elif player == "adversary":
                    counts["adversary"] += 1
                seen_value += 1
                if not role_value_caps:
                    _reservoir_add(
                        value_sample,
                        item,
                        seen=seen_value,
                        max_rows=int(max_value_rows),
                        rng=rng_value,
                    )
                elif player == "controller":
                    seen_controller_value += 1
                    _reservoir_add(
                        controller_value_sample,
                        item,
                        seen=seen_controller_value,
                        max_rows=int(max_controller_value_rows),
                        rng=rng_controller_value,
                    )
                elif player == "adversary":
                    seen_adversary_value += 1
                    _reservoir_add(
                        adversary_value_sample,
                        item,
                        seen=seen_adversary_value,
                        max_rows=int(max_adversary_value_rows),
                        rng=rng_adversary_value,
                    )
                if player == "controller":
                    seen_ctrl += 1
                    _reservoir_add(
                        ctrl_sample,
                        item,
                        seen=seen_ctrl,
                        max_rows=int(max_controller_policy_roots),
                        rng=rng_ctrl,
                    )
                elif player == "adversary":
                    seen_adv += 1
                    _reservoir_add(
                        adv_sample,
                        item,
                        seen=seen_adv,
                        max_rows=int(max_adversary_policy_roots),
                        rng=rng_adv,
                    )
    if role_value_caps:
        value_sample = controller_value_sample + adversary_value_sample
    ctrl_roots = {r["_key"]: r["_state_features"] for r in ctrl_sample}
    adv_roots = {r["_key"]: r["_state_features"] for r in adv_sample}
    return value_sample, ctrl_roots, adv_roots, counts


def _collect_policy_training_arrays(
    paths: list[Path],
    roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
    *,
    player: str,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if not roots:
        return (
            np.empty((0, VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("player", "")).lower() != str(player):
                    continue
                key = _root_key(row)
                if key not in roots:
                    continue
                feats = _read_json_list(row.get("action_features_json", ""))
                if len(feats) != int(action_dim):
                    continue
                try:
                    idx = int(float(row.get("canon_action_index", -1) or -1))
                except Exception:
                    idx = -1
                try:
                    visits = int(float(row.get("visit_count", 0) or 0))
                except Exception:
                    visits = 0
                actions_by_key.setdefault(key, []).append((idx, np.asarray(feats, dtype=np.float32), visits))

    total_actions = sum(len(v) for v in actions_by_key.values())
    if total_actions <= 0:
        return (
            np.empty((0, VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    X = np.empty((int(total_actions), VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32)
    y = np.empty(int(total_actions), dtype=np.float32)
    probs_out = np.empty(int(total_actions), dtype=np.float32)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for key in sorted(actions_by_key.keys()):
        actions = sorted(actions_by_key[key], key=lambda x: x[0])
        visits = np.asarray([max(0, int(a[2])) for a in actions], dtype=np.float64)
        total = float(np.sum(visits))
        probs = visits / total if total > 0.0 else np.full(visits.size, 1.0 / float(visits.size), dtype=np.float64)
        logits = np.log(visits + float(POLICY_ALPHA))
        logits = logits - float(np.mean(logits))
        start = pos
        sf = roots[key]
        for action, logit, prob in zip(actions, logits, probs):
            X[pos, :VALUE_FEATURE_DIM] = sf
            X[pos, VALUE_FEATURE_DIM:] = action[1]
            y[pos] = float(logit)
            probs_out[pos] = float(prob)
            pos += 1
        offsets.append((start, pos))
    return X[:pos], y[:pos], probs_out[:pos], offsets


def _materialize_policy_training_arrays(
    roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
    actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]],
    *,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    total_actions = sum(len(v) for v in actions_by_key.values())
    if total_actions <= 0:
        return (
            np.empty((0, VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    X = np.empty((int(total_actions), VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32)
    y = np.empty(int(total_actions), dtype=np.float32)
    probs_out = np.empty(int(total_actions), dtype=np.float32)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for key in sorted(actions_by_key.keys()):
        actions = sorted(actions_by_key[key], key=lambda x: x[0])
        visits = np.asarray([max(0, int(a[2])) for a in actions], dtype=np.float64)
        total = float(np.sum(visits))
        probs = visits / total if total > 0.0 else np.full(visits.size, 1.0 / float(visits.size), dtype=np.float64)
        logits = np.log(visits + float(POLICY_ALPHA))
        logits = logits - float(np.mean(logits))
        start = pos
        sf = roots[key]
        for action, logit, prob in zip(actions, logits, probs):
            X[pos, :VALUE_FEATURE_DIM] = sf
            X[pos, VALUE_FEATURE_DIM:] = action[1]
            y[pos] = float(logit)
            probs_out[pos] = float(prob)
            pos += 1
        offsets.append((start, pos))
    return X[:pos], y[:pos], probs_out[:pos], offsets


def _collect_policy_training_arrays_for_players(
    paths: list[Path],
    controller_roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
    adversary_roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]],
    tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]],
    dict[str, float | int],
]:
    ctrl_actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]] = {}
    adv_actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]] = {}
    t_scan = time.time()
    rows_scanned = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_scanned += 1
                player = str(row.get("player", "")).lower()
                if player == "controller":
                    roots = controller_roots
                    action_dim = CTRL_ACTION_DIM
                    actions_by_key = ctrl_actions_by_key
                elif player == "adversary":
                    roots = adversary_roots
                    action_dim = ADV_ACTION_DIM
                    actions_by_key = adv_actions_by_key
                else:
                    continue
                key = _root_key(row)
                if key not in roots:
                    continue
                feats = _read_json_list(row.get("action_features_json", ""))
                if len(feats) != int(action_dim):
                    continue
                try:
                    idx = int(float(row.get("canon_action_index", -1) or -1))
                except Exception:
                    idx = -1
                try:
                    visits = int(float(row.get("visit_count", 0) or 0))
                except Exception:
                    visits = 0
                actions_by_key.setdefault(key, []).append((idx, np.asarray(feats, dtype=np.float32), visits))

    scan_elapsed = float(time.time() - t_scan)
    t_materialize = time.time()
    ctrl_arrays = _materialize_policy_training_arrays(
        controller_roots,
        ctrl_actions_by_key,
        action_dim=CTRL_ACTION_DIM,
    )
    adv_arrays = _materialize_policy_training_arrays(
        adversary_roots,
        adv_actions_by_key,
        action_dim=ADV_ACTION_DIM,
    )
    materialize_elapsed = float(time.time() - t_materialize)
    timings: dict[str, float | int] = {
        "policy_replay_rows_scanned": int(rows_scanned),
        "policy_replay_scan_elapsed_s": float(scan_elapsed),
        "policy_array_materialize_elapsed_s": float(materialize_elapsed),
        "controller_policy_roots_with_actions": int(len(ctrl_actions_by_key)),
        "adversary_policy_roots_with_actions": int(len(adv_actions_by_key)),
        "controller_policy_action_rows": int(ctrl_arrays[0].shape[0]),
        "adversary_policy_action_rows": int(adv_arrays[0].shape[0]),
    }
    return ctrl_arrays, adv_arrays, timings


def _value_arrays_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray | list[MarkovValueFeatures], np.ndarray]:
    if not rows:
        empty_x: np.ndarray | list[MarkovValueFeatures]
        empty_x = [] if AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA else np.empty(
            (0, VALUE_FEATURE_DIM), dtype=np.float32
        )
        return empty_x, np.empty(0, dtype=np.float32)
    y = np.asarray([float(row["_target_value"]) for row in rows], dtype=np.float32)
    if AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA:
        samples = [row["_value_features"] for row in rows]
        if not all(isinstance(sample, MarkovValueFeatures) for sample in samples):
            raise TypeError("Markov value sample contains a legacy feature vector")
        return samples, y
    X = np.empty((len(rows), VALUE_FEATURE_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        X[i] = row["_value_features"]
    return X, y


def _sample_rows(rows: list[dict[str, Any]], *, seed: int, max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= int(max_rows):
        return list(rows)
    rng = random.Random(int(seed))
    return sorted(rng.sample(rows, int(max_rows)), key=lambda r: r["_key"])


def _value_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred.astype(np.float64) - y.astype(np.float64)
    abs_err = np.abs(err)
    mse = float(np.mean(err * err)) if err.size else 0.0
    return {
        "mse": mse,
        "rmse": float(math.sqrt(max(0.0, mse))),
        "max_abs_error": float(np.max(abs_err)) if abs_err.size else 0.0,
        "p95_abs_error": float(np.percentile(abs_err, 95)) if abs_err.size else 0.0,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    z = x.astype(np.float64) - float(np.max(x))
    e = np.exp(z)
    s = float(np.sum(e))
    if not math.isfinite(s) or s <= 0.0:
        return np.full(x.size, 1.0 / float(x.size), dtype=np.float64)
    return e / s


def _policy_arrays(
    state_rows: list[dict[str, Any]],
    policy_by_key: dict[tuple[str, str, int, int, int, str], list[dict[str, Any]]],
    *,
    player: str,
    sample_roots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    roots = [r for r in state_rows if str(r.get("player")) == player and r["_key"] in policy_by_key]
    roots = _sample_rows(roots, seed=seed, max_rows=int(sample_roots))
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    p_rows: list[float] = []
    offsets: list[tuple[int, int]] = []
    for root in roots:
        actions = policy_by_key.get(root["_key"], [])
        if not actions:
            continue
        visits = np.asarray([max(0, int(a["_visit_count"])) for a in actions], dtype=np.float64)
        if visits.size == 0:
            continue
        total = float(np.sum(visits))
        probs = visits / total if total > 0.0 else np.full(visits.size, 1.0 / float(visits.size), dtype=np.float64)
        logs = np.log(visits + float(POLICY_ALPHA))
        centered = logs - float(np.mean(logs))
        start = len(X_rows)
        sf = list(root["_state_features"])
        for action, logit, prob in zip(actions, centered, probs):
            X_rows.append(sf + list(action["_action_features"]))
            y_rows.append(float(logit))
            p_rows.append(float(prob))
        offsets.append((start, len(X_rows)))
    if not X_rows:
        return np.empty((0, VALUE_FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), []
    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(p_rows, dtype=np.float32),
        offsets,
    )


def _policy_metrics(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    probs: np.ndarray,
    offsets: list[tuple[int, int]],
    structured_states: list[MarkovValueFeatures] | None = None,
) -> dict[str, float]:
    if X.shape[0] == 0:
        return {"mse": 0.0, "cross_entropy": 0.0, "top1": 0.0, "top3": 0.0}
    pred = (
        model.predict_structured(structured_states, X, offsets).astype(np.float64)
        if structured_states is not None
        else model.predict(X).astype(np.float64)
    )
    mse = float(np.mean((pred - y.astype(np.float64)) ** 2))
    ce_sum = 0.0
    top1 = 0
    top3 = 0
    roots = 0
    eps = 1e-12
    for s, e in offsets:
        if e <= s:
            continue
        p = probs[s:e].astype(np.float64)
        if p.sum() <= 0.0:
            p = np.full(e - s, 1.0 / float(e - s), dtype=np.float64)
        else:
            p = p / p.sum()
        q = _softmax(pred[s:e])
        best = int(np.argmax(p))
        order = np.argsort(-pred[s:e], kind="stable")
        top1 += int(order[0] == best)
        top3 += int(best in set(int(x) for x in order[: min(3, order.size)]))
        ce_sum += -float(np.sum(p * np.log(np.maximum(q, eps))))
        roots += 1
    denom = float(max(1, roots))
    return {"mse": mse, "cross_entropy": float(ce_sum / denom), "top1": float(top1 / denom), "top3": float(top3 / denom)}


_POLICY_METRICS_PROCESS_DATA: dict[str, tuple[Any, ...]] = {}
_POLICY_METRICS_PROCESS_READY = False


def _policy_metric_root_ranges(
    root_count: int,
    *,
    chunks: int,
) -> list[tuple[int, int]]:
    if int(root_count) <= 0:
        return []
    count = min(int(root_count), max(1, int(chunks)))
    return [
        (
            (index * int(root_count)) // count,
            ((index + 1) * int(root_count)) // count,
        )
        for index in range(count)
    ]


def _policy_metrics_process_chunk(
    role: str,
    root_begin: int,
    root_end: int,
) -> dict[str, float | int | str]:
    global _POLICY_METRICS_PROCESS_READY

    if not _POLICY_METRICS_PROCESS_READY:
        try:
            import torch

            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        except ImportError:
            pass
        _POLICY_METRICS_PROCESS_READY = True

    model, X, y, probs, offsets, structured_states = _POLICY_METRICS_PROCESS_DATA[
        str(role)
    ]
    selected_offsets = offsets[int(root_begin):int(root_end)]
    if not selected_offsets:
        return {
            "role": str(role),
            "squared_error_sum": 0.0,
            "action_count": 0,
            "cross_entropy_sum": 0.0,
            "top1_count": 0,
            "top3_count": 0,
            "root_count": 0,
        }

    action_begin = int(selected_offsets[0][0])
    action_end = int(selected_offsets[-1][1])
    local_offsets = [
        (int(begin) - action_begin, int(end) - action_begin)
        for begin, end in selected_offsets
    ]
    X_chunk = X[action_begin:action_end]
    states_chunk = (
        structured_states[int(root_begin):int(root_end)]
        if structured_states is not None
        else None
    )
    with threadpool_limits(limits=1):
        pred = (
            model.predict_structured(
                states_chunk,
                X_chunk,
                local_offsets,
            ).astype(np.float64)
            if states_chunk is not None
            else model.predict(X_chunk).astype(np.float64)
        )

    target = y[action_begin:action_end].astype(np.float64)
    err = pred - target
    squared_error_sum = float(np.sum(err * err, dtype=np.float64))
    cross_entropy_sum = 0.0
    top1_count = 0
    top3_count = 0
    root_count = 0
    eps = 1e-12
    local_probs = probs[action_begin:action_end]
    for begin, end in local_offsets:
        if end <= begin:
            continue
        p = local_probs[begin:end].astype(np.float64)
        if p.sum() <= 0.0:
            p = np.full(end - begin, 1.0 / float(end - begin), dtype=np.float64)
        else:
            p = p / p.sum()
        q = _softmax(pred[begin:end])
        best = int(np.argmax(p))
        order = np.argsort(-pred[begin:end], kind="stable")
        top1_count += int(order[0] == best)
        top3_count += int(best in set(int(index) for index in order[: min(3, order.size)]))
        cross_entropy_sum += -float(np.sum(p * np.log(np.maximum(q, eps))))
        root_count += 1
    return {
        "role": str(role),
        "squared_error_sum": squared_error_sum,
        "action_count": int(action_end - action_begin),
        "cross_entropy_sum": float(cross_entropy_sum),
        "top1_count": int(top1_count),
        "top3_count": int(top3_count),
        "root_count": int(root_count),
    }


def _aggregate_policy_metric_parts(
    parts: list[dict[str, float | int | str]],
) -> dict[str, float]:
    action_count = sum(int(part["action_count"]) for part in parts)
    root_count = sum(int(part["root_count"]) for part in parts)
    if action_count <= 0:
        return {"mse": 0.0, "cross_entropy": 0.0, "top1": 0.0, "top3": 0.0}
    root_denom = float(max(1, root_count))
    return {
        "mse": float(
            sum(float(part["squared_error_sum"]) for part in parts)
            / float(action_count)
        ),
        "cross_entropy": float(
            sum(float(part["cross_entropy_sum"]) for part in parts) / root_denom
        ),
        "top1": float(
            sum(int(part["top1_count"]) for part in parts) / root_denom
        ),
        "top3": float(
            sum(int(part["top3_count"]) for part in parts) / root_denom
        ),
    }


def _policy_metrics_parallel_pair(
    controller_model: Any,
    Xc: np.ndarray,
    yc: np.ndarray,
    pc: np.ndarray,
    offc: list[tuple[int, int]],
    adversary_model: Any,
    Xa: np.ndarray,
    ya: np.ndarray,
    pa: np.ndarray,
    offa: list[tuple[int, int]],
    controller_states: list[MarkovValueFeatures] | None = None,
    adversary_states: list[MarkovValueFeatures] | None = None,
    *,
    max_workers: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float | int | str]]:
    started = time.time()
    total_roots = int(len(offc) + len(offa))
    workers = min(max(1, int(max_workers)), max(1, total_roots))
    if workers <= 1 or total_roots <= 1:
        controller_metrics = _policy_metrics(
            controller_model, Xc, yc, pc, offc, controller_states
        )
        adversary_metrics = _policy_metrics(
            adversary_model, Xa, ya, pa, offa, adversary_states
        )
        return controller_metrics, adversary_metrics, {
            "policy_metrics_mode": "serial",
            "policy_metrics_workers": 1,
            "policy_metrics_tasks": 2,
            "policy_metrics_wall_elapsed_s": float(time.time() - started),
        }

    try:
        context = multiprocessing.get_context("fork")
    except ValueError as exc:
        raise RuntimeError(
            "parallel policy metrics require the Linux fork multiprocessing context"
        ) from exc

    task_target = min(total_roots, workers * 4)
    controller_tasks = max(
        1 if offc else 0,
        int(round(task_target * len(offc) / float(total_roots))),
    )
    adversary_tasks = max(1 if offa else 0, task_target - controller_tasks)
    tasks = [
        ("controller", begin, end)
        for begin, end in _policy_metric_root_ranges(
            len(offc), chunks=controller_tasks
        )
    ]
    tasks.extend(
        ("adversary", begin, end)
        for begin, end in _policy_metric_root_ranges(
            len(offa), chunks=adversary_tasks
        )
    )

    global _POLICY_METRICS_PROCESS_DATA
    _POLICY_METRICS_PROCESS_DATA = {
        "controller": (
            controller_model,
            Xc,
            yc,
            pc,
            offc,
            controller_states,
        ),
        "adversary": (
            adversary_model,
            Xa,
            ya,
            pa,
            offa,
            adversary_states,
        ),
    }
    parts_by_role: dict[str, list[dict[str, float | int | str]]] = {
        "controller": [],
        "adversary": [],
    }
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_policy_metrics_process_chunk, role, begin, end)
                for role, begin, end in tasks
            ]
            for future in as_completed(futures):
                part = future.result()
                parts_by_role[str(part["role"])].append(part)
    finally:
        _POLICY_METRICS_PROCESS_DATA = {}

    return (
        _aggregate_policy_metric_parts(parts_by_role["controller"]),
        _aggregate_policy_metric_parts(parts_by_role["adversary"]),
        {
            "policy_metrics_mode": "multiprocess_fork_shared",
            "policy_metrics_workers": int(workers),
            "policy_metrics_tasks": int(len(tasks)),
            "policy_metrics_wall_elapsed_s": float(time.time() - started),
        },
    )


def _fit_value(X: np.ndarray, y: np.ndarray, *, seed: int, apply_thread_limit: bool = True) -> Any:
    est = HistGradientBoostingRegressor(
        loss="squared_error",
        max_leaf_nodes=63,
        max_iter=1050,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=int(seed),
        early_stopping=False,
    )
    weights = (1.0 + 2.0 * np.abs(y)).astype(np.float32)
    if apply_thread_limit:
        with _hgb_threadpool_context():
            est.fit(X, y, sample_weight=weights)
    else:
        est.fit(X, y, sample_weight=weights)
    return est


def _fit_value_with_timing(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    apply_thread_limit: bool = True,
) -> tuple[Any, float]:
    t0 = time.time()
    model = _fit_value(X, y, seed=int(seed), apply_thread_limit=bool(apply_thread_limit))
    return model, float(time.time() - t0)


def _fit_value_models_parallel(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    version: int,
) -> tuple[Any, Any, dict[str, float | int | str]]:
    t0 = time.time()
    hgb_openmp_threads = _hgb_openmp_thread_limit()
    with _hgb_threadpool_context():
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agz_value_fit") as executor:
            ctrl_future = executor.submit(
                _fit_value_with_timing,
                X,
                y,
                seed=int(seed) + int(version) + 11,
                apply_thread_limit=False,
            )
            adv_future = executor.submit(
                _fit_value_with_timing,
                X,
                y,
                seed=int(seed) + int(version) + 17,
                apply_thread_limit=False,
            )
            ctrl_model, ctrl_elapsed = ctrl_future.result()
            adv_model, adv_elapsed = adv_future.result()
    timings: dict[str, float | int | str] = {
        "value_fit_mode": "threaded_controller_adversary",
        "hgb_openmp_threads": "unlimited" if hgb_openmp_threads is None else int(hgb_openmp_threads),
        "controller_value_fit_elapsed_s": float(ctrl_elapsed),
        "adversary_value_fit_elapsed_s": float(adv_elapsed),
        "value_fit_wall_elapsed_s": float(time.time() - t0),
    }
    return ctrl_model, adv_model, timings


def _hgb_openmp_thread_limit() -> int | None:
    raw = os.environ.get(HGB_OPENMP_THREADS_ENV)
    if raw is None:
        return int(DEFAULT_HGB_OPENMP_THREADS)
    value = str(raw).strip().lower()
    if value in {"", "0", "none", "auto", "default", "unlimited"}:
        return None
    try:
        return max(1, int(value))
    except ValueError:
        return int(DEFAULT_HGB_OPENMP_THREADS)


def _hgb_threadpool_context():
    limit = _hgb_openmp_thread_limit()
    if limit is None:
        return nullcontext()
    return threadpool_limits(limits=int(limit), user_api="openmp")


def _fit_policy(X: np.ndarray, y: np.ndarray, *, seed: int, apply_thread_limit: bool = True) -> Any:
    est = HistGradientBoostingRegressor(
        loss="squared_error",
        max_leaf_nodes=63,
        max_iter=1050,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=int(seed),
        early_stopping=False,
    )
    if apply_thread_limit:
        with _hgb_threadpool_context():
            est.fit(X, y)
    else:
        est.fit(X, y)
    return est


def _fit_policy_with_timing(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    apply_thread_limit: bool = True,
) -> tuple[Any, float]:
    t0 = time.time()
    model = _fit_policy(X, y, seed=int(seed), apply_thread_limit=bool(apply_thread_limit))
    return model, float(time.time() - t0)


def _fit_policy_models_parallel(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xa: np.ndarray,
    ya: np.ndarray,
    *,
    seed: int,
    version: int,
) -> tuple[Any, Any, dict[str, float | int | str]]:
    t0 = time.time()
    hgb_openmp_threads = _hgb_openmp_thread_limit()
    with _hgb_threadpool_context():
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agz_policy_fit") as executor:
            ctrl_future = executor.submit(
                _fit_policy_with_timing,
                Xc,
                yc,
                seed=int(seed) + int(version) + 101,
                apply_thread_limit=False,
            )
            adv_future = executor.submit(
                _fit_policy_with_timing,
                Xa,
                ya,
                seed=int(seed) + int(version) + 201,
                apply_thread_limit=False,
            )
            ctrl_model, ctrl_elapsed = ctrl_future.result()
            adv_model, adv_elapsed = adv_future.result()
    timings: dict[str, float | int | str] = {
        "policy_fit_mode": "threaded_controller_adversary",
        "hgb_openmp_threads": "unlimited" if hgb_openmp_threads is None else int(hgb_openmp_threads),
        "controller_policy_fit_elapsed_s": float(ctrl_elapsed),
        "adversary_policy_fit_elapsed_s": float(adv_elapsed),
        "policy_fit_wall_elapsed_s": float(time.time() - t0),
    }
    return ctrl_model, adv_model, timings




def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_incremental_dnn_parent(
    path: Path,
    *,
    expected_type: type[Any],
    label: str,
) -> tuple[Path, str]:
    from vidur.AlphaGoZero.dnn_models import is_dnn_model

    parent_path = Path(path)
    if not parent_path.is_file():
        raise RuntimeError(f"missing DNN training parent for {label}: {parent_path}")
    parent = joblib.load(parent_path)
    if not is_dnn_model(parent) or not isinstance(parent, expected_type):
        raise RuntimeError(
            f"DNN training parent for {label} is not {expected_type.__name__}: "
            f"{parent_path} ({type(parent).__name__})"
        )
    if not getattr(parent, "optimizer_state", None):
        raise RuntimeError(f"DNN training parent for {label} has no optimizer state: {parent_path}")
    return parent_path, _sha256_file(parent_path)


def _fit_dnn_value_models_parallel(
    X: np.ndarray | list[MarkovValueFeatures],
    y: np.ndarray,
    *,
    seed: int,
    version: int,
    training_parent: ModelBundle,
) -> tuple[Any, Any, dict[str, float | int | str]]:
    from vidur.AlphaGoZero.dnn_models import (
        MarkovValueDeepSet,
        ValueResidualMLP,
        fit_markov_value_dnn,
        fit_value_dnn,
    )

    markov = AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA
    expected_type = MarkovValueDeepSet if markov else ValueResidualMLP
    fit_function = fit_markov_value_dnn if markov else fit_value_dnn
    controller_parent, controller_sha = _require_incremental_dnn_parent(
        training_parent.controller_value_model_path,
        expected_type=expected_type,
        label="controller_value",
    )
    adversary_parent, adversary_sha = _require_incremental_dnn_parent(
        training_parent.adversary_value_model_path,
        expected_type=expected_type,
        label="adversary_value",
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agz_dnn_value_fit") as executor:
        controller_future = executor.submit(
            fit_function,
            X,
            y,
            role="controller",
            initial_model_path=controller_parent,
            seed=int(seed) + int(version) + 11,
            epochs=int(AGZ_DNN_EPOCHS),
            batch_size=int(AGZ_DNN_VALUE_BATCH_SIZE),
            lr=float(AGZ_DNN_UPDATE_LR),
            torch_threads=int(AGZ_DNN_TORCH_THREADS_PER_MODEL),
        )
        adversary_future = executor.submit(
            fit_function,
            X,
            y,
            role="adversary",
            initial_model_path=adversary_parent,
            seed=int(seed) + int(version) + 17,
            epochs=int(AGZ_DNN_EPOCHS),
            batch_size=int(AGZ_DNN_VALUE_BATCH_SIZE),
            lr=float(AGZ_DNN_UPDATE_LR),
            torch_threads=int(AGZ_DNN_TORCH_THREADS_PER_MODEL),
        )
        controller_model, controller_timing = controller_future.result()
        adversary_model, adversary_timing = adversary_future.result()

    controller_model.training_metadata.update({
        "parent_model_version": int(training_parent.controller_model_version),
        "parent_sha256": controller_sha,
        "parent_path": str(controller_parent),
        "incremental_update": True,
        "target_perspective": "controller",
    })
    adversary_model.training_metadata.update({
        "parent_model_version": int(training_parent.adversary_model_version),
        "parent_sha256": adversary_sha,
        "parent_path": str(adversary_parent),
        "incremental_update": True,
        "target_perspective": "controller",
    })
    return controller_model, adversary_model, {
        "value_fit_mode": "incremental_dnn_threaded_controller_adversary",
        "controller_value_fit_elapsed_s": float(controller_timing["elapsed_s"]),
        "adversary_value_fit_elapsed_s": float(adversary_timing["elapsed_s"]),
        "value_fit_wall_elapsed_s": float(time.time() - started),
        "controller_parent_version": int(training_parent.controller_model_version),
        "adversary_parent_version": int(training_parent.adversary_model_version),
    }


def _fit_dnn_policy_models_parallel(
    Xc: np.ndarray,
    pc: np.ndarray,
    offc: list[tuple[int, int]],
    Xa: np.ndarray,
    pa: np.ndarray,
    offa: list[tuple[int, int]],
    controller_states: list[MarkovValueFeatures] | None = None,
    adversary_states: list[MarkovValueFeatures] | None = None,
    *,
    seed: int,
    version: int,
    training_parent: ModelBundle,
) -> tuple[Any, Any, dict[str, float | int | str]]:
    from vidur.AlphaGoZero.dnn_models import (
        MarkovPolicyRankDeepSet,
        PolicyRankMLP,
        fit_markov_policy_dnn,
        fit_policy_dnn,
    )

    markov_policy = AGZ_POLICY_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA
    expected_type = MarkovPolicyRankDeepSet if markov_policy else PolicyRankMLP
    fit_fn = fit_markov_policy_dnn if markov_policy else fit_policy_dnn
    if markov_policy and (controller_states is None or adversary_states is None):
        raise RuntimeError("Markov policy training is missing structured policy roots")

    controller_parent, controller_sha = _require_incremental_dnn_parent(
        training_parent.controller_prior_model_path,
        expected_type=expected_type,
        label="controller_prior",
    )
    adversary_parent, adversary_sha = _require_incremental_dnn_parent(
        training_parent.adversary_prior_model_path,
        expected_type=expected_type,
        label="adversary_prior",
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agz_dnn_policy_fit") as executor:
        controller_future = executor.submit(
            fit_fn,
            *([controller_states, Xc] if markov_policy else [Xc]),
            pc,
            offc,
            role="controller",
            action_dim=CTRL_ACTION_DIM,
            initial_model_path=controller_parent,
            seed=int(seed) + int(version) + 101,
            epochs=int(AGZ_DNN_EPOCHS),
            root_batch_size=int(AGZ_DNN_POLICY_ROOT_BATCH_SIZE),
            lr=float(AGZ_DNN_UPDATE_LR),
            torch_threads=int(AGZ_DNN_TORCH_THREADS_PER_MODEL),
        )
        adversary_future = executor.submit(
            fit_fn,
            *([adversary_states, Xa] if markov_policy else [Xa]),
            pa,
            offa,
            role="adversary",
            action_dim=ADV_ACTION_DIM,
            initial_model_path=adversary_parent,
            seed=int(seed) + int(version) + 201,
            epochs=int(AGZ_DNN_EPOCHS),
            root_batch_size=int(AGZ_DNN_POLICY_ROOT_BATCH_SIZE),
            lr=float(AGZ_DNN_UPDATE_LR),
            torch_threads=int(AGZ_DNN_TORCH_THREADS_PER_MODEL),
        )
        controller_model, controller_timing = controller_future.result()
        adversary_model, adversary_timing = adversary_future.result()

    controller_model.training_metadata.update({
        "parent_model_version": int(training_parent.controller_model_version),
        "parent_sha256": controller_sha,
        "parent_path": str(controller_parent),
        "incremental_update": True,
    })
    adversary_model.training_metadata.update({
        "parent_model_version": int(training_parent.adversary_model_version),
        "parent_sha256": adversary_sha,
        "parent_path": str(adversary_parent),
        "incremental_update": True,
    })
    return controller_model, adversary_model, {
        "policy_fit_mode": "incremental_dnn_threaded_controller_adversary",
        "controller_policy_fit_elapsed_s": float(controller_timing["elapsed_s"]),
        "adversary_policy_fit_elapsed_s": float(adversary_timing["elapsed_s"]),
        "policy_fit_wall_elapsed_s": float(time.time() - started),
        "controller_parent_version": int(training_parent.controller_model_version),
        "adversary_parent_version": int(training_parent.adversary_model_version),
    }

def _candidate_manifest_versions(root: Path) -> list[int]:
    models_dir = Path(root) / "models"
    versions: list[int] = []
    for manifest_path in models_dir.glob("Model_Version*/candidate_manifest.json") if models_dir.exists() else []:
        suffix = manifest_path.parent.name.removeprefix("Model_Version")
        version = 0
        if suffix.isdigit():
            version = int(suffix)
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                version = int(manifest.get("model_version", 0) or 0)
            except Exception:
                version = 0
        if version > 0:
            versions.append(version)
    return versions


def _next_candidate_version(root: Path) -> int:
    state_path = root / "xl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    versions = [int(state.get("promoted_model_version", 100) or 100)]
    for key in ("last_candidate_model_version", "last_training_cycle_completed_model_version"):
        try:
            version = int(state.get(key, 0) or 0)
        except Exception:
            version = 0
        if version > 0:
            versions.append(version)
    versions.extend(_candidate_manifest_versions(root))
    return max(versions) + 1


def _retire_incomplete_candidate_dir(path: Path) -> None:
    path = Path(path)
    if not path.exists() or (path / "candidate_manifest.json").exists():
        return
    failed_root = path.parent / "incomplete_model_dirs"
    failed_root.mkdir(parents=True, exist_ok=True)
    dest = failed_root / f"{path.name}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    shutil.move(str(path), str(dest))


def _mark_training_complete(root: Path, version: int) -> None:
    state_path = root / "xl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state["candidate_counter"] = int(state.get("candidate_counter", 0)) + 1
    state["last_candidate_model_version"] = int(version)
    state["last_model_fit_completed_at_utc"] = utc_now()
    atomic_write_json(state_path, state)


def _mark_training_cycle_complete(root: Path, version: int) -> None:
    state_path = root / "xl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    current_path = root / "training" / "current_training.json"
    try:
        current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else {}
    except Exception:
        current = {}
    consumed = int(dict(current.get("gate", {}) or {}).get("new_states_since_last_training", 0) or 0)
    available = int(state.get("new_states_since_last_training", 0) or 0)
    state["new_states_since_last_training"] = max(0, available - consumed)
    state["states_consumed_by_last_training_cycle"] = int(consumed)
    state["last_training_completed_at_utc"] = utc_now()
    state["last_training_cycle_completed_model_version"] = int(version)
    atomic_write_json(state_path, state)


def _mark_current_training_finalized(root: Path, version: int) -> None:
    path = root / "training" / "current_training.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data["cycle_finalized_at_utc"] = utc_now()
    data["cycle_finalized_model_version"] = int(version)
    atomic_write_json(path, data)


def _load_state(root: Path) -> dict[str, Any]:
    path = Path(root) / "xl_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_write_json(Path(root) / "xl_state.json", state)


def _default_promoted_bundle() -> ModelBundle:
    return ModelBundle(
        model_version=100,
        controller_model_version=100,
        adversary_model_version=100,
        controller_value_model_path=Path(DEFAULT_VALUE_MODEL_PATH),
        adversary_value_model_path=Path(DEFAULT_VALUE_MODEL_PATH),
        controller_prior_model_path=Path(DEFAULT_CONTROLLER_PRIOR_MODEL_PATH),
        adversary_prior_model_path=Path(DEFAULT_ADVERSARY_PRIOR_MODEL_PATH),
    )


def _current_promoted_bundle(root: Path) -> ModelBundle:
    current = Path(root) / "models" / "current_model.json"
    if not current.exists():
        return _default_promoted_bundle()
    data = json.loads(current.read_text(encoding="utf-8"))
    legacy_version = int(data.get("model_version", 100) or 100)
    controller_version = int(data.get("controller_model_version", legacy_version) or legacy_version)
    adversary_version = int(data.get("adversary_model_version", legacy_version) or legacy_version)
    legacy_value_path = data.get("value_model_path") or str(DEFAULT_VALUE_MODEL_PATH)
    return ModelBundle(
        model_version=int(max(legacy_version, controller_version, adversary_version)),
        controller_model_version=int(controller_version),
        adversary_model_version=int(adversary_version),
        controller_value_model_path=Path(data.get("controller_value_model_path") or legacy_value_path),
        adversary_value_model_path=Path(data.get("adversary_value_model_path") or legacy_value_path),
        controller_prior_model_path=Path(data["controller_prior_model_path"]),
        adversary_prior_model_path=Path(data["adversary_prior_model_path"]),
    )


def _candidate_bundle_from_output(version: int, out: Path) -> ModelBundle:
    return ModelBundle(
        model_version=int(version),
        controller_model_version=int(version),
        adversary_model_version=int(version),
        controller_value_model_path=out / "controller_value" / CONFIG_NAME / "model.joblib",
        adversary_value_model_path=out / "adversary_value" / CONFIG_NAME / "model.joblib",
        controller_prior_model_path=out / "controller_prior" / POLICY_CONFIG_NAME / "model.joblib",
        adversary_prior_model_path=out / "adversary_prior" / POLICY_CONFIG_NAME / "model.joblib",
    )



def _published_dnn_candidate_bundle(root: Path, version: int) -> ModelBundle | None:
    """Return a parity-validated, fully published DNN training checkpoint."""
    out = Path(root) / "models" / f"Model_Version{int(version)}"
    manifest_path = out / "candidate_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(manifest.get("model_version", 0) or 0) != int(version):
        return None
    if str(manifest.get("model_family", "")).lower() != "dnn":
        return None
    if not bool(manifest.get("native_ready", False)):
        return None
    bundle = _candidate_bundle_from_output(int(version), out)
    required = (
        bundle.controller_value_model_path,
        bundle.adversary_value_model_path,
        bundle.controller_prior_model_path,
        bundle.adversary_prior_model_path,
    )
    return bundle if all(path.is_file() for path in required) else None


def _latest_dnn_training_parent_bundle(
    root: Path,
    *,
    next_version: int,
    promoted: ModelBundle,
) -> ModelBundle:
    """Continue optimization from the latest checkpoint, not the self-play best."""
    controller_version = int(promoted.controller_model_version)
    adversary_version = int(promoted.adversary_model_version)
    controller_value = promoted.controller_value_model_path
    adversary_value = promoted.adversary_value_model_path
    controller_prior = promoted.controller_prior_model_path
    adversary_prior = promoted.adversary_prior_model_path

    for version in sorted(set(_candidate_manifest_versions(root)), reverse=True):
        if int(version) >= int(next_version):
            continue
        candidate = _published_dnn_candidate_bundle(root, int(version))
        if candidate is None:
            continue
        if int(version) >= controller_version:
            controller_version = int(version)
            controller_value = candidate.controller_value_model_path
            controller_prior = candidate.controller_prior_model_path
        if int(version) >= adversary_version:
            adversary_version = int(version)
            adversary_value = candidate.adversary_value_model_path
            adversary_prior = candidate.adversary_prior_model_path
        break

    return ModelBundle(
        model_version=max(controller_version, adversary_version),
        controller_model_version=controller_version,
        adversary_model_version=adversary_version,
        controller_value_model_path=controller_value,
        adversary_value_model_path=adversary_value,
        controller_prior_model_path=controller_prior,
        adversary_prior_model_path=adversary_prior,
    )

def _run(cmd: list[str], *, cwd: Path | None = None, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=log, stderr=subprocess.STDOUT, check=True, text=True)


def _arena_cmd(
    *,
    output_dir: Path,
    model_path: Path,
    model_version: int,
    controller_prior_path: Path,
    adversary_prior_path: Path,
    num_games: int,
    parallel_games: int,
    game_id_start: int,
    iterations: int,
    history_seed: int,
    history_hops_min: int,
    history_hops_max: int,
    only_model_ctrl_cycle: bool,
    skip_model_ctrl_cycle: bool = False,
    rollout_count: int | None = None,
    rollout_parallel_threads: int | None = None,
    write_arena_game_logs: bool = True,
    role_controller: ModelBundle | None = None,
    role_adversary: ModelBundle | None = None,
    puct_c: float | None = None,
) -> list[str]:
    if only_model_ctrl_cycle and skip_model_ctrl_cycle:
        raise ValueError("arena command cannot both skip and exclusively run the model cycle")
    resolved_rollout_count = int(
        rollout_count
        if rollout_count is not None
        else (
            AGZ_EVAL_ROLLOUT_COUNT
            if only_model_ctrl_cycle
            else AGZ_SJF_ROLLOUT_COUNT
        )
    )
    resolved_rollout_threads = int(
        rollout_parallel_threads
        if rollout_parallel_threads is not None
        else (
            AGZ_EVAL_ROLLOUT_PARALLEL_THREADS
            if only_model_ctrl_cycle
            else AGZ_SJF_ROLLOUT_PARALLEL_THREADS
        )
    )
    resolved_puct_c = float(
        puct_c
        if puct_c is not None
        else (
            AGZ_EVAL_PUCT_C
            if only_model_ctrl_cycle
            else AGZ_SJF_PUCT_C
        )
    )
    cmd = [
        sys.executable,
        "-m",
        "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP",
        "--model-path",
        str(model_path),
        "--model-version",
        str(int(model_version)),
        "--feature-dim",
        "226",
        "--output-dir",
        str(output_dir),
        "--game-id-start",
        str(int(game_id_start)),
        "--num-games",
        str(int(num_games)),
        "--num-parallel-games",
        str(int(parallel_games)),
        "--shared-root-mcts-iterations",
        str(int(iterations)),
        "--discount-factor",
        str(float(AGZ_DISCOUNT_FACTOR)),
        "--worker-threads",
        "1",
        "--trivial-budget-tokens",
        "256",
        "--arena-time-limit-sec",
        "5.0",
        "--history-hops-min",
        str(int(history_hops_min)),
        "--history-hops-max",
        str(int(history_hops_max)),
        "--history-seed",
        str(int(history_seed)),
        "--history-hops-unique",
        "--no-history-hops-force-zero",
        "--seed",
        str(int(history_seed) + int(game_id_start)),
        "--puct-c",
        str(resolved_puct_c),
        "--policy-prior-temperature",
        "1.0",
        "--no-root-dirichlet-noise-enabled",
        "--no-agz-sample-initial-moves",
        "--controller-prior-model-path",
        str(controller_prior_path),
        "--adversary-prior-model-path",
        str(adversary_prior_path),
        "--native-search-mode",
        str(AGZ_NATIVE_SEARCH_MODE),
        "--rollout-count",
        str(int(resolved_rollout_count)),
        "--rollout-parallel-threads",
        str(int(resolved_rollout_threads)),
        "--rollout-horizon-sec",
        str(float(AGZ_ROLLOUT_HORIZON_SEC)),
        "--rollout-policy-temperature",
        str(float(AGZ_ROLLOUT_POLICY_TEMPERATURE)),
        "--rollout-probability-quantum",
        str(float(AGZ_ROLLOUT_PROBABILITY_QUANTUM)),
        "--rollout-max-actions",
        str(int(AGZ_ROLLOUT_MAX_ACTIONS)),
    ]
    if only_model_ctrl_cycle:
        cmd.append("--only-model-ctrl-cycle")
    if skip_model_ctrl_cycle:
        cmd.append("--skip-model-ctrl-cycle")
    if not bool(write_arena_game_logs):
        cmd.append("--no-arena-game-logs")
    if role_controller is not None and role_adversary is not None:
        cmd.extend([
            "--role-controller-value-model-path",
            str(role_controller.controller_value_model_path),
            "--role-controller-prior-model-path",
            str(role_controller.controller_prior_model_path),
            "--role-adversary-value-model-path",
            str(role_adversary.adversary_value_model_path),
            "--role-adversary-prior-model-path",
            str(role_adversary.adversary_prior_model_path),
        ])
    return cmd


def _run_arena(
    *,
    root: Path,
    output_dir: Path,
    model_path: Path,
    model_version: int,
    controller_prior_path: Path,
    adversary_prior_path: Path,
    num_games: int,
    parallel_games: int,
    game_id_start: int,
    iterations: int,
    history_seed: int,
    history_hops_min: int,
    history_hops_max: int,
    only_model_ctrl_cycle: bool,
    skip_model_ctrl_cycle: bool = False,
    rollout_count: int | None = None,
    rollout_parallel_threads: int | None = None,
    write_arena_game_logs: bool = True,
    role_controller: ModelBundle | None = None,
    role_adversary: ModelBundle | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _arena_cmd(
        output_dir=output_dir,
        model_path=model_path,
        model_version=int(model_version),
        controller_prior_path=controller_prior_path,
        adversary_prior_path=adversary_prior_path,
        num_games=int(num_games),
        parallel_games=int(parallel_games),
        game_id_start=int(game_id_start),
        iterations=int(iterations),
        history_seed=int(history_seed),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        only_model_ctrl_cycle=bool(only_model_ctrl_cycle),
        skip_model_ctrl_cycle=bool(skip_model_ctrl_cycle),
        rollout_count=rollout_count,
        rollout_parallel_threads=rollout_parallel_threads,
        write_arena_game_logs=bool(write_arena_game_logs),
        role_controller=role_controller,
        role_adversary=role_adversary,
    )
    (output_dir / "launch_command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    _run(cmd, cwd=Path(__file__).resolve().parents[2], log_path=output_dir / "arena_launcher.log")
    return output_dir / "arena_results.csv"


def _read_planned_hops(output_dir: Path) -> dict[int, int]:
    path = Path(output_dir) / "planned_games.csv"
    out: dict[int, int] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(float(row.get("game_id", 0) or 0))] = int(float(row.get("history_hops", 0) or 0))
    return out


def _read_cycle2_costs(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not Path(path).exists():
        return out
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gid = int(float(row.get("game_id", 0) or 0))
            out[gid] = float(row.get("cycle2_total_cost", 0.0) or 0.0)
    return out


def _role_history(state: dict[str, Any], role: str, current_version: int) -> list[int]:
    history = []
    role_key = f"{role}_promoted_model_history"
    for item in state.get(role_key, []) or []:
        try:
            version = int(item)
        except Exception:
            continue
        if version > 0 and version not in history:
            history.append(version)
    if int(current_version) > 0 and int(current_version) not in history:
        history.append(int(current_version))
    return history


def _compose_role_bundle(
    *,
    current: ModelBundle,
    candidate: ModelBundle,
    use_candidate_controller: bool,
    use_candidate_adversary: bool,
) -> ModelBundle:
    """Build a role combination without mutating promotion state."""

    controller_version = int(
        candidate.controller_model_version
        if use_candidate_controller
        else current.controller_model_version
    )
    adversary_version = int(
        candidate.adversary_model_version
        if use_candidate_adversary
        else current.adversary_model_version
    )
    return ModelBundle(
        model_version=int(
            max(
                controller_version,
                adversary_version,
                current.model_version,
                candidate.model_version,
            )
        ),
        controller_model_version=controller_version,
        adversary_model_version=adversary_version,
        controller_value_model_path=(
            candidate.controller_value_model_path
            if use_candidate_controller
            else current.controller_value_model_path
        ),
        adversary_value_model_path=(
            candidate.adversary_value_model_path
            if use_candidate_adversary
            else current.adversary_value_model_path
        ),
        controller_prior_model_path=(
            candidate.controller_prior_model_path
            if use_candidate_controller
            else current.controller_prior_model_path
        ),
        adversary_prior_model_path=(
            candidate.adversary_prior_model_path
            if use_candidate_adversary
            else current.adversary_prior_model_path
        ),
    )


def _promote_candidate_roles(
    root: Path,
    *,
    current: ModelBundle,
    candidate: ModelBundle,
    promote_controller: bool,
    promote_adversary: bool,
) -> ModelBundle:
    new_controller_version = int(candidate.controller_model_version if promote_controller else current.controller_model_version)
    new_adversary_version = int(candidate.adversary_model_version if promote_adversary else current.adversary_model_version)
    bundle = ModelBundle(
        model_version=int(max(new_controller_version, new_adversary_version, current.model_version, candidate.model_version)),
        controller_model_version=int(new_controller_version),
        adversary_model_version=int(new_adversary_version),
        controller_value_model_path=(
            candidate.controller_value_model_path if promote_controller else current.controller_value_model_path
        ),
        adversary_value_model_path=(
            candidate.adversary_value_model_path if promote_adversary else current.adversary_value_model_path
        ),
        controller_prior_model_path=(
            candidate.controller_prior_model_path if promote_controller else current.controller_prior_model_path
        ),
        adversary_prior_model_path=(
            candidate.adversary_prior_model_path if promote_adversary else current.adversary_prior_model_path
        ),
    )

    models_dir = Path(root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(models_dir / "current_model.json", bundle.to_json())

    state = _load_state(root)
    state["controller_promoted_model_version"] = int(bundle.controller_model_version)
    state["adversary_promoted_model_version"] = int(bundle.adversary_model_version)
    state["promoted_model_version"] = int(max(bundle.controller_model_version, bundle.adversary_model_version))

    controller_history = _role_history(state, "controller", current.controller_model_version)
    adversary_history = _role_history(state, "adversary", current.adversary_model_version)
    if promote_controller and int(candidate.controller_model_version) not in controller_history:
        controller_history.append(int(candidate.controller_model_version))
    if promote_adversary and int(candidate.adversary_model_version) not in adversary_history:
        adversary_history.append(int(candidate.adversary_model_version))
    state["controller_promoted_model_history"] = controller_history
    state["adversary_promoted_model_history"] = adversary_history

    promoted_history = []
    for version in [*controller_history, *adversary_history]:
        if int(version) > 0 and int(version) not in promoted_history:
            promoted_history.append(int(version))
    state["promoted_model_history"] = promoted_history

    controller_count = int(state.get("controller_promotions_completed", 0) or 0) + int(bool(promote_controller))
    adversary_count = int(state.get("adversary_promotions_completed", 0) or 0) + int(bool(promote_adversary))
    state["controller_promotions_completed"] = int(controller_count)
    state["adversary_promotions_completed"] = int(adversary_count)
    state["promotions_completed"] = int(max(controller_count, adversary_count))
    state["role_promotions_completed_total"] = int(controller_count + adversary_count)
    if promote_controller:
        state["last_controller_promotion_at_utc"] = utc_now()
    if promote_adversary:
        state["last_adversary_promotion_at_utc"] = utc_now()
    if promote_controller or promote_adversary:
        state["last_promotion_at_utc"] = utc_now()
    # The coordinator owns xl_state.json because ingestion remains active while
    # evaluation runs. Promotion is durably represented by current_model.json
    # and the completed candidate manifest, which the coordinator reconciles.
    return bundle


def _promote_candidate(root: Path, candidate: ModelBundle) -> None:
    current = _current_promoted_bundle(root)
    _promote_candidate_roles(
        root,
        current=current,
        candidate=candidate,
        promote_controller=True,
        promote_adversary=True,
    )


def _broadcast_current_bundle_to_workers(root: Path, bundle: ModelBundle, *, candidate_version: int | None = None) -> None:
    root = Path(root)
    if candidate_version is not None:
        model_dir = root / "models" / f"Model_Version{int(candidate_version)}"
    else:
        model_dir = root / "models" / f"Model_Version{int(bundle.model_version)}"
    remote_root = str(root).rstrip("/")
    for worker in WORKERS:
        worker_current_dir = f"{remote_root}/worker_large/{worker.worker_id}/models"
        if model_dir.exists():
            remote_model_dir = f"{remote_root}/models/{model_dir.name}"
            _run(["ssh", worker.host, f"mkdir -p {remote_model_dir!r} {worker_current_dir!r}"])
            _run([
                "rsync",
                "-az",
                "--partial",
                "--delay-updates",
                "--timeout=120",
                str(model_dir).rstrip("/") + "/",
                f"{worker.host}:{remote_model_dir.rstrip('/')}/",
            ])
        else:
            _run(["ssh", worker.host, f"mkdir -p {worker_current_dir!r}"])
        payload = json.dumps(bundle.to_json(), sort_keys=True)
        _run(["ssh", worker.host, f"printf '%s\n' {payload!r} > {worker_current_dir!r}/current_model.json"])


    _broadcast_runtime_search_config_to_workers(root)


def _broadcast_runtime_search_config_to_workers(root: Path) -> None:
    root = Path(root)
    runtime_path = runtime_search_config_path(root)
    if not runtime_path.exists():
        return
    remote_root = str(root).rstrip("/")
    for worker in WORKERS:
        worker_root = f"{remote_root}/worker_large/{worker.worker_id}"
        _run(["ssh", worker.host, f"mkdir -p {worker_root!r}"])
        _run([
            "rsync",
            "-az",
            "--partial",
            "--delay-updates",
            "--timeout=120",
            str(runtime_path),
            f"{worker.host}:{worker_root}/runtime_search_config.json",
        ])


def _broadcast_candidate_to_workers(root: Path, candidate: ModelBundle) -> None:
    _broadcast_current_bundle_to_workers(root, candidate, candidate_version=int(candidate.model_version))

def _ensure_eval_details_header(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = list(EVAL_GAME_FIELDS)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=expected).writeheader()
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        has_rows = any(True for _ in reader)
    if header == expected:
        return
    if has_rows:
        backup = path.with_suffix(path.suffix + f".schema_mismatch_{int(time.time())}.bak")
        shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=expected).writeheader()

def _role_arena_command(
    *,
    output_dir: Path,
    controller: ModelBundle,
    adversary: ModelBundle,
    eval_games: int,
    eval_parallel: int,
    game_id_start: int,
    iterations: int,
    history_seed: int,
    max_hop: int,
) -> list[str]:
    return _arena_cmd(
        output_dir=output_dir,
        model_path=controller.controller_value_model_path,
        model_version=int(controller.controller_model_version),
        controller_prior_path=controller.controller_prior_model_path,
        adversary_prior_path=adversary.adversary_prior_model_path,
        num_games=int(eval_games),
        parallel_games=int(eval_parallel),
        game_id_start=int(game_id_start),
        iterations=int(iterations),
        history_seed=int(history_seed),
        history_hops_min=0,
        history_hops_max=int(max_hop),
        only_model_ctrl_cycle=True,
        rollout_count=int(AGZ_EVAL_ROLLOUT_COUNT),
        puct_c=float(AGZ_EVAL_PUCT_C),
        role_controller=controller,
        role_adversary=adversary,
    )


def _run_role_eval_blocks(
    *,
    root: Path,
    eval_dir: Path,
    promoted: ModelBundle,
    candidate: ModelBundle,
    eval_games: int,
    eval_parallel: int,
    iterations: int,
    adversary_start_gid: int,
    controller_start_gid: int,
    adversary_seed: int,
    controller_seed: int,
    max_hop: int,
) -> tuple[Path, Path, Path, Path]:
    adv_baseline_dir = eval_dir / "promoted_adversary_vs_promoted_controller_for_adversary"
    adv_candidate_dir = eval_dir / "candidate_adversary_vs_promoted_controller"
    ctrl_baseline_dir = eval_dir / "promoted_adversary_vs_promoted_controller_for_controller"
    ctrl_candidate_dir = eval_dir / "promoted_adversary_vs_candidate_controller"
    specs = {
        adv_baseline_dir.name: (adv_baseline_dir, promoted, promoted, adversary_start_gid, adversary_seed),
        adv_candidate_dir.name: (adv_candidate_dir, promoted, candidate, adversary_start_gid, adversary_seed),
        ctrl_baseline_dir.name: (ctrl_baseline_dir, promoted, promoted, controller_start_gid, controller_seed),
        ctrl_candidate_dir.name: (ctrl_candidate_dir, candidate, promoted, controller_start_gid, controller_seed),
    }
    if spot_pull_eval_enabled() or distributed_eval_enabled():
        commands = {
            name: _role_arena_command(
                output_dir=spec[0],
                controller=spec[1],
                adversary=spec[2],
                eval_games=eval_games,
                eval_parallel=eval_parallel,
                game_id_start=spec[3],
                iterations=iterations,
                history_seed=spec[4],
                max_hop=max_hop,
            )
            for name, spec in specs.items()
        }
        final_dirs = {name: spec[0] for name, spec in specs.items()}
        if spot_pull_eval_enabled():
            results = run_spot_pull_commands(
                root=root,
                block_commands=commands,
                final_dirs=final_dirs,
                expected_games_by_block={name: int(eval_games) for name in specs},
                merge_block=merge_arena_block,
            )
            return (
                results[adv_baseline_dir.name],
                results[adv_candidate_dir.name],
                results[ctrl_baseline_dir.name],
                results[ctrl_candidate_dir.name],
            )
        scratch = root / ".distributed_eval_scratch" / eval_dir.name / "role"
        hosts = memory_safe_hosts(default_role_hosts(), active_blocks=len(commands))
        hosts = cpu_safe_hosts(
            hosts,
            phase="role",
            total_games=int(eval_games) * len(commands),
        )
        chunks = plan_role_chunks(commands, scratch_root=scratch, hosts=hosts)
        results = run_distributed_chunks(
            chunks,
            final_dirs=final_dirs,
            expected_games_by_block={name: int(eval_games) for name in specs},
        )
        return (
            results[adv_baseline_dir.name],
            results[adv_candidate_dir.name],
            results[ctrl_baseline_dir.name],
            results[ctrl_candidate_dir.name],
        )

    local_results: dict[str, Path] = {}
    for name, (output_dir, controller, adversary, game_id_start, seed) in specs.items():
        local_results[name] = _run_arena(
            root=root,
            output_dir=output_dir,
            model_path=controller.controller_value_model_path,
            model_version=int(controller.controller_model_version),
            controller_prior_path=controller.controller_prior_model_path,
            adversary_prior_path=adversary.adversary_prior_model_path,
            num_games=int(eval_games),
            parallel_games=int(eval_parallel),
            game_id_start=int(game_id_start),
            iterations=int(iterations),
            history_seed=int(seed),
            history_hops_min=0,
            history_hops_max=int(max_hop),
            only_model_ctrl_cycle=True,
            role_controller=controller,
            role_adversary=adversary,
        )
    return (
        local_results[adv_baseline_dir.name],
        local_results[adv_candidate_dir.name],
        local_results[ctrl_baseline_dir.name],
        local_results[ctrl_candidate_dir.name],
    )


def _sjf_cycle_plan(
    *,
    output_dir: Path,
    candidate: ModelBundle,
    benchmark_games: int,
    eval_parallel: int,
    iterations: int,
    history_seed: int,
) -> tuple[dict[str, list[str]], dict[str, Path]]:
    """Build the two independently executable halves of one SJF benchmark."""

    output_dir = Path(output_dir)
    trivial_dir = output_dir / "cycle1_trivial"
    model_dir = output_dir / "cycle2_model"
    common = dict(
        model_path=candidate.controller_value_model_path,
        model_version=int(candidate.controller_model_version),
        controller_prior_path=candidate.controller_prior_model_path,
        adversary_prior_path=candidate.adversary_prior_model_path,
        num_games=int(benchmark_games),
        parallel_games=int(eval_parallel),
        game_id_start=80_000_000 + int(candidate.model_version) * 10_000,
        iterations=int(iterations),
        history_seed=int(history_seed) + 909,
        history_hops_min=0,
        history_hops_max=max(100, int(benchmark_games) + 5),
        write_arena_game_logs=True,
        role_controller=candidate,
        role_adversary=candidate,
        rollout_count=int(AGZ_SJF_ROLLOUT_COUNT),
        rollout_parallel_threads=int(AGZ_SJF_ROLLOUT_PARALLEL_THREADS),
        puct_c=float(AGZ_SJF_PUCT_C),
    )
    commands = {
        trivial_dir.name: _arena_cmd(
            output_dir=trivial_dir,
            only_model_ctrl_cycle=False,
            skip_model_ctrl_cycle=True,
            **common,
        ),
        model_dir.name: _arena_cmd(
            output_dir=model_dir,
            only_model_ctrl_cycle=True,
            skip_model_ctrl_cycle=False,
            **common,
        ),
    }
    return commands, {
        trivial_dir.name: trivial_dir,
        model_dir.name: model_dir,
    }


def _run_sjf_benchmark(
    *,
    root: Path,
    eval_dir: Path,
    candidate: ModelBundle,
    benchmark_games: int,
    eval_parallel: int,
    iterations: int,
    history_seed: int,
) -> Path:
    sjf_dir = eval_dir / "SJF_256_Game"
    trivial_dir = sjf_dir / "cycle1_trivial"
    model_dir = sjf_dir / "cycle2_model"
    common = dict(
        model_path=candidate.controller_value_model_path,
        model_version=int(candidate.controller_model_version),
        controller_prior_path=candidate.controller_prior_model_path,
        adversary_prior_path=candidate.adversary_prior_model_path,
        num_games=int(benchmark_games),
        parallel_games=int(eval_parallel),
        game_id_start=80_000_000 + int(candidate.model_version) * 10_000,
        iterations=int(iterations),
        history_seed=int(history_seed) + 909,
        history_hops_min=0,
        history_hops_max=max(100, int(benchmark_games) + 5),
        write_arena_game_logs=True,
        role_controller=candidate,
        role_adversary=candidate,
        rollout_count=int(AGZ_SJF_ROLLOUT_COUNT),
        rollout_parallel_threads=int(AGZ_SJF_ROLLOUT_PARALLEL_THREADS),
        puct_c=float(AGZ_SJF_PUCT_C),
    )
    trivial_command = _arena_cmd(
        output_dir=trivial_dir,
        only_model_ctrl_cycle=False,
        skip_model_ctrl_cycle=True,
        **common,
    )
    model_command = _arena_cmd(
        output_dir=model_dir,
        only_model_ctrl_cycle=True,
        skip_model_ctrl_cycle=False,
        **common,
    )
    commands = {trivial_dir.name: trivial_command, model_dir.name: model_command}
    if spot_pull_eval_enabled():
        run_spot_pull_commands(
            root=root,
            block_commands=commands,
            final_dirs={trivial_dir.name: trivial_dir, model_dir.name: model_dir},
            expected_games_by_block={
                name: int(benchmark_games) for name in commands
            },
            merge_block=merge_arena_block,
        )
    elif distributed_eval_enabled():
        scratch = root / ".distributed_eval_scratch" / eval_dir.name / "sjf"
        total_cycle_jobs = 2 * int(benchmark_games)
        hosts = memory_safe_hosts(default_sjf_hosts(total_cycle_jobs), active_blocks=2)
        hosts = cpu_safe_hosts(
            hosts,
            phase="sjf",
            total_games=total_cycle_jobs,
        )
        commands = {trivial_dir.name: trivial_command, model_dir.name: model_command}
        chunks = plan_independent_block_chunks(
            commands,
            scratch_root=scratch,
            hosts=hosts,
        )
        run_distributed_chunks(
            chunks,
            final_dirs={trivial_dir.name: trivial_dir, model_dir.name: model_dir},
            expected_games_by_block={
                name: int(benchmark_games) for name in commands
            },
        )
    else:
        _run(
            trivial_command,
            cwd=Path(__file__).resolve().parents[2],
            log_path=trivial_dir / "arena_launcher.log",
        )
        _run(
            model_command,
            cwd=Path(__file__).resolve().parents[2],
            log_path=model_dir / "arena_launcher.log",
        )
    return merge_split_sjf_cycles(
        trivial_dir=trivial_dir,
        model_dir=model_dir,
        output_dir=sjf_dir,
        expected_games=int(benchmark_games),
    )


def _run_spot_role_eval_with_speculative_sjf(
    *,
    root: Path,
    eval_dir: Path,
    promoted: ModelBundle,
    candidate: ModelBundle,
    eval_games: int,
    eval_parallel: int,
    benchmark_games: int,
    iterations: int,
    adversary_start_gid: int,
    controller_start_gid: int,
    adversary_seed: int,
    controller_seed: int,
    history_seed: int,
    max_hop: int,
) -> tuple[Path, Path, Path, Path, dict[str, Path]]:
    """Run role evaluation and all possible promoted SJF combinations together."""

    adv_baseline_dir = eval_dir / "promoted_adversary_vs_promoted_controller_for_adversary"
    adv_candidate_dir = eval_dir / "candidate_adversary_vs_promoted_controller"
    ctrl_baseline_dir = eval_dir / "promoted_adversary_vs_promoted_controller_for_controller"
    ctrl_candidate_dir = eval_dir / "promoted_adversary_vs_candidate_controller"
    role_specs = {
        adv_baseline_dir.name: (
            adv_baseline_dir,
            promoted,
            promoted,
            adversary_start_gid,
            adversary_seed,
        ),
        adv_candidate_dir.name: (
            adv_candidate_dir,
            promoted,
            candidate,
            adversary_start_gid,
            adversary_seed,
        ),
        ctrl_baseline_dir.name: (
            ctrl_baseline_dir,
            promoted,
            promoted,
            controller_start_gid,
            controller_seed,
        ),
        ctrl_candidate_dir.name: (
            ctrl_candidate_dir,
            candidate,
            promoted,
            controller_start_gid,
            controller_seed,
        ),
    }
    commands = {
        name: _role_arena_command(
            output_dir=spec[0],
            controller=spec[1],
            adversary=spec[2],
            eval_games=int(eval_games),
            eval_parallel=int(eval_parallel),
            game_id_start=int(spec[3]),
            iterations=int(iterations),
            history_seed=int(spec[4]),
            max_hop=int(max_hop),
        )
        for name, spec in role_specs.items()
    }
    final_dirs = {name: spec[0] for name, spec in role_specs.items()}
    expected_games_by_block = {name: int(eval_games) for name in role_specs}

    scenario_bundles = {
        "controller_candidate": _compose_role_bundle(
            current=promoted,
            candidate=candidate,
            use_candidate_controller=True,
            use_candidate_adversary=False,
        ),
        "adversary_candidate": _compose_role_bundle(
            current=promoted,
            candidate=candidate,
            use_candidate_controller=False,
            use_candidate_adversary=True,
        ),
        "both_candidates": _compose_role_bundle(
            current=promoted,
            candidate=candidate,
            use_candidate_controller=True,
            use_candidate_adversary=True,
        ),
    }
    speculative_root = eval_dir / "SJF_256_Speculative"
    scenario_cycle_blocks: dict[str, dict[str, str]] = {}
    for scenario, bundle in scenario_bundles.items():
        scenario_dir = speculative_root / scenario
        cycle_commands, cycle_dirs = _sjf_cycle_plan(
            output_dir=scenario_dir,
            candidate=bundle,
            benchmark_games=int(benchmark_games),
            eval_parallel=int(eval_parallel),
            iterations=int(iterations),
            history_seed=int(history_seed),
        )
        scenario_cycle_blocks[scenario] = {}
        for cycle_name, command in cycle_commands.items():
            block_name = f"sjf__{scenario}__{cycle_name}"
            commands[block_name] = command
            final_dirs[block_name] = cycle_dirs[cycle_name]
            expected_games_by_block[block_name] = int(benchmark_games)
            scenario_cycle_blocks[scenario][cycle_name] = block_name

    merged = run_spot_pull_commands(
        root=root,
        block_commands=commands,
        final_dirs=final_dirs,
        expected_games_by_block=expected_games_by_block,
        merge_block=merge_arena_block,
    )
    speculative_results: dict[str, Path] = {}
    for scenario, cycle_blocks in scenario_cycle_blocks.items():
        scenario_dir = speculative_root / scenario
        speculative_results[scenario] = merge_split_sjf_cycles(
            trivial_dir=final_dirs[cycle_blocks["cycle1_trivial"]],
            model_dir=final_dirs[cycle_blocks["cycle2_model"]],
            output_dir=scenario_dir,
            expected_games=int(benchmark_games),
        )

    atomic_write_json(
        speculative_root / "manifest.json",
        {
            "mode": "spot_role_and_speculative_sjf_single_batch_v1",
            "candidate_model_version": int(candidate.model_version),
            "promoted_controller_model_version": int(promoted.controller_model_version),
            "promoted_adversary_model_version": int(promoted.adversary_model_version),
            "promotion_inputs": "role_arena_results_only",
            "scenarios": {
                name: {
                    "bundle": scenario_bundles[name].to_json(),
                    "arena_results_csv": str(path),
                }
                for name, path in speculative_results.items()
            },
        },
    )
    return (
        merged[adv_baseline_dir.name],
        merged[adv_candidate_dir.name],
        merged[ctrl_baseline_dir.name],
        merged[ctrl_candidate_dir.name],
        speculative_results,
    )


def _selected_speculative_sjf_scenario(
    *,
    promote_controller: bool,
    promote_adversary: bool,
) -> str:
    if promote_controller and promote_adversary:
        return "both_candidates"
    if promote_controller:
        return "controller_candidate"
    if promote_adversary:
        return "adversary_candidate"
    return ""


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _publish_speculative_sjf_as_standard(
    *,
    eval_dir: Path,
    scenario: str,
    source_results: Path,
) -> Path:
    source_dir = Path(source_results).parent
    target_dir = Path(eval_dir) / "SJF_256_Game"
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir, copy_function=_hardlink_or_copy)
    atomic_write_json(
        target_dir / "speculative_selection.json",
        {
            "scenario": str(scenario),
            "source_dir": str(source_dir),
            "arena_results_csv": str(target_dir / "arena_results.csv"),
        },
    )
    return target_dir / "arena_results.csv"


def _role_candidate_promoted(
    *,
    wins: int,
    games_compared: int,
    expected_games: int,
    strict_win_threshold: int,
) -> bool:
    return bool(
        int(games_compared) >= int(expected_games)
        and int(wins) > int(strict_win_threshold)
    )


def _evaluate_and_maybe_promote_unpaused(
    root: Path,
    *,
    candidate: ModelBundle,
    eval_games: int,
    eval_parallel: int,
    benchmark_games: int,
    iterations: int,
    history_seed: int,
) -> dict[str, Any]:
    promoted = _current_promoted_bundle(root)
    eval_number = int(candidate.model_version)
    eval_dir = Path(root) / "eval_of_models" / f"eval_{eval_number:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    start_gid = 70_000_000 + int(candidate.model_version) * 10_000
    max_hop = max(100, int(eval_games) + 5)

    def run_role_arena(
        *,
        output_dir: Path,
        controller: ModelBundle,
        adversary: ModelBundle,
        game_id_start: int,
        seed: int,
    ) -> Path:
        return _run_arena(
            root=root,
            output_dir=output_dir,
            model_path=controller.controller_value_model_path,
            model_version=int(controller.controller_model_version),
            controller_prior_path=controller.controller_prior_model_path,
            adversary_prior_path=adversary.adversary_prior_model_path,
            num_games=int(eval_games),
            parallel_games=int(eval_parallel),
            game_id_start=int(game_id_start),
            iterations=int(iterations),
            history_seed=int(seed),
            history_hops_min=0,
            history_hops_max=int(max_hop),
            only_model_ctrl_cycle=True,
            role_controller=controller,
            role_adversary=adversary,
        )

    adversary_start_gid = int(start_gid)
    controller_start_gid = int(start_gid) + max(5000, int(eval_games) * 2)
    adversary_seed = int(history_seed)
    controller_seed = int(history_seed) + 1_000_003

    adv_baseline_dir = eval_dir / "promoted_adversary_vs_promoted_controller_for_adversary"
    adv_candidate_dir = eval_dir / "candidate_adversary_vs_promoted_controller"
    ctrl_baseline_dir = eval_dir / "promoted_adversary_vs_promoted_controller_for_controller"
    ctrl_candidate_dir = eval_dir / "promoted_adversary_vs_candidate_controller"

    (
        adv_baseline_results,
        adv_candidate_results,
        ctrl_baseline_results,
        ctrl_candidate_results,
    ) = _run_role_eval_blocks(
        root=root,
        eval_dir=eval_dir,
        promoted=promoted,
        candidate=candidate,
        eval_games=int(eval_games),
        eval_parallel=int(eval_parallel),
        iterations=int(iterations),
        adversary_start_gid=adversary_start_gid,
        controller_start_gid=controller_start_gid,
        adversary_seed=adversary_seed,
        controller_seed=controller_seed,
        max_hop=max_hop,
    )

    adv_baseline_costs = _read_cycle2_costs(adv_baseline_results)
    adv_candidate_costs = _read_cycle2_costs(adv_candidate_results)
    ctrl_baseline_costs = _read_cycle2_costs(ctrl_baseline_results)
    ctrl_candidate_costs = _read_cycle2_costs(ctrl_candidate_results)

    adversary_hops = _read_planned_hops(adv_baseline_dir)
    controller_hops = _read_planned_hops(ctrl_baseline_dir)
    adversary_games = sorted(set(adv_baseline_costs).intersection(adv_candidate_costs))
    controller_games = sorted(set(ctrl_baseline_costs).intersection(ctrl_candidate_costs))

    eps = 1e-9
    testing_adversary_wins = sum(
        1
        for gid in adversary_games
        if float(adv_candidate_costs[gid]) > float(adv_baseline_costs[gid]) + eps
    )
    testing_controller_wins = sum(
        1
        for gid in controller_games
        if float(ctrl_candidate_costs[gid]) < float(ctrl_baseline_costs[gid]) - eps
    )
    testing_adversary_win_ratio = float(testing_adversary_wins / max(1, len(adversary_games)))
    testing_controller_win_ratio = float(testing_controller_wins / max(1, len(controller_games)))
    role_win_threshold = int(ROLE_PROMOTION_WIN_THRESHOLD)
    required_wins_per_role = int(role_win_threshold) + 1
    expected_games_per_role = int(eval_games)
    candidate_adversary_promoted = _role_candidate_promoted(
        wins=int(testing_adversary_wins),
        games_compared=len(adversary_games),
        expected_games=int(expected_games_per_role),
        strict_win_threshold=int(role_win_threshold),
    )
    candidate_controller_promoted = _role_candidate_promoted(
        wins=int(testing_controller_wins),
        games_compared=len(controller_games),
        expected_games=int(expected_games_per_role),
        strict_win_threshold=int(role_win_threshold),
    )
    did_promote = bool(candidate_controller_promoted or candidate_adversary_promoted)
    new_controller_version = int(candidate.controller_model_version if candidate_controller_promoted else promoted.controller_model_version)
    new_adversary_version = int(candidate.adversary_model_version if candidate_adversary_promoted else promoted.adversary_model_version)
    eval_time = local_time_24h()
    details = eval_dir / "eval_game_details.csv"
    _ensure_eval_details_header(details)

    for gid in adversary_games:
        append_csv_row(details, EVAL_GAME_FIELDS, {
            "promoted_controller_model_version": int(promoted.controller_model_version),
            "promoted_adversary_model_version": int(promoted.adversary_model_version),
            "candidate_model_version": int(candidate.model_version),
            "new_controller_model_version": int(new_controller_version),
            "new_adversary_model_version": int(new_adversary_version),
            "candidate_win_ratio_in_100_games": float(testing_adversary_win_ratio),
            "candidate_controller_promoted": int(candidate_controller_promoted),
            "candidate_adversary_promoted": int(candidate_adversary_promoted),
            "candidate_promoted": int(did_promote),
            "time_of_eval": eval_time,
            "game_mode": TESTING_ADVERSARY_MODE,
            "game_id": int(gid),
            "game_hop_number": int(adversary_hops.get(gid, 0)),
            "total_cost_when_promoted_adv_and_promoted_controller": float(adv_baseline_costs[gid]),
            "total_cost_when_candidate_adv_and_promoted_controller": float(adv_candidate_costs[gid]),
            "total_cost_when_promoted_adv_and_candidate_controller": "",
            "total_cost_when_candidate_adv_and_candidate_controller": "",
        })
    for gid in controller_games:
        append_csv_row(details, EVAL_GAME_FIELDS, {
            "promoted_controller_model_version": int(promoted.controller_model_version),
            "promoted_adversary_model_version": int(promoted.adversary_model_version),
            "candidate_model_version": int(candidate.model_version),
            "new_controller_model_version": int(new_controller_version),
            "new_adversary_model_version": int(new_adversary_version),
            "candidate_win_ratio_in_100_games": float(testing_controller_win_ratio),
            "candidate_controller_promoted": int(candidate_controller_promoted),
            "candidate_adversary_promoted": int(candidate_adversary_promoted),
            "candidate_promoted": int(did_promote),
            "time_of_eval": eval_time,
            "game_mode": TESTING_CONTROLLER_MODE,
            "game_id": int(gid),
            "game_hop_number": int(controller_hops.get(gid, 0)),
            "total_cost_when_promoted_adv_and_promoted_controller": float(ctrl_baseline_costs[gid]),
            "total_cost_when_candidate_adv_and_promoted_controller": "",
            "total_cost_when_promoted_adv_and_candidate_controller": float(ctrl_candidate_costs[gid]),
            "total_cost_when_candidate_adv_and_candidate_controller": "",
        })

    candidate_wins = max(int(testing_adversary_wins), int(testing_controller_wins))
    candidate_win_ratio = max(float(testing_adversary_win_ratio), float(testing_controller_win_ratio))
    result = {
        "eval_dir": str(eval_dir),
        "arena_dirs": {
            "promoted_adversary_vs_promoted_controller_for_adversary": str(adv_baseline_dir),
            "candidate_adversary_vs_promoted_controller": str(adv_candidate_dir),
            "promoted_adversary_vs_promoted_controller_for_controller": str(ctrl_baseline_dir),
            "promoted_adversary_vs_candidate_controller": str(ctrl_candidate_dir),
        },
        "promoted_controller_model_version": int(promoted.controller_model_version),
        "promoted_adversary_model_version": int(promoted.adversary_model_version),
        "candidate_model_version": int(candidate.model_version),
        "new_controller_model_version": int(new_controller_version),
        "new_adversary_model_version": int(new_adversary_version),
        "promoted_model_version": int(max(promoted.controller_model_version, promoted.adversary_model_version)),
        "new_promoted_model_version": int(max(new_controller_version, new_adversary_version)),
        "candidate_win_ratio": float(candidate_win_ratio),
        "candidate_wins": int(candidate_wins),
        "games_compared": int(len(adversary_games) + len(controller_games)),
        "candidate_promoted": bool(did_promote),
        "candidate_controller_promoted": bool(candidate_controller_promoted),
        "candidate_adversary_promoted": bool(candidate_adversary_promoted),
        "promotion_win_rate_threshold": float(PROMOTION_WIN_RATE_THRESHOLD),
        "promotion_rule": f"role_independent_current_baseline_{int(eval_games)}_each_gt_{int(ROLE_PROMOTION_WIN_THRESHOLD)}",
        "role_promotion_win_threshold": int(role_win_threshold),
        "promotion_required_wins_per_role": int(required_wins_per_role),
        "expected_games_per_role": int(expected_games_per_role),
        "testing_adversary_games_compared": int(len(adversary_games)),
        "testing_adversary_wins": int(testing_adversary_wins),
        "testing_adversary_win_ratio": float(testing_adversary_win_ratio),
        "testing_controller_games_compared": int(len(controller_games)),
        "testing_controller_wins": int(testing_controller_wins),
        "testing_controller_win_ratio": float(testing_controller_win_ratio),
    }
    atomic_write_json(eval_dir / "eval_summary.json", result)

    if did_promote:
        promoted_after = _promote_candidate_roles(
            root,
            current=promoted,
            candidate=candidate,
            promote_controller=bool(candidate_controller_promoted),
            promote_adversary=bool(candidate_adversary_promoted),
        )
        _broadcast_current_bundle_to_workers(root, promoted_after, candidate_version=int(candidate.model_version))
        sjf_results = _run_sjf_benchmark(
            root=root,
            eval_dir=eval_dir,
            candidate=promoted_after,
            benchmark_games=int(benchmark_games),
            eval_parallel=int(eval_parallel),
            iterations=int(iterations),
            history_seed=int(history_seed),
        )
        result["sjf_results_csv"] = str(sjf_results)
        atomic_write_json(eval_dir / "eval_summary.json", result)
    return result


def evaluate_and_maybe_promote(
    root: Path,
    *,
    candidate: ModelBundle,
    eval_games: int,
    eval_parallel: int,
    benchmark_games: int,
    iterations: int,
    history_seed: int,
) -> dict[str, Any]:
    with pause_selfplay_for_eval(root):
        return _evaluate_and_maybe_promote_unpaused(
            root,
            candidate=candidate,
            eval_games=eval_games,
            eval_parallel=eval_parallel,
            benchmark_games=benchmark_games,
            iterations=iterations,
            history_seed=history_seed,
        )


def train_candidate(root: Path, *, seed: int = 2026, max_value_states: int = 500_000, eval_games: int = AGZ_EVAL_GAMES, eval_parallel: int = 60, benchmark_games: int = AGZ_BENCHMARK_GAMES, mcts_iterations: int = AGZ_EVAL_MCTS_ITERATIONS) -> dict[str, Any]:
    root = Path(root)
    dataset_preparation_started = time.time()
    version = _next_candidate_version(root)
    sampling_seed = _candidate_sampling_seed(seed, version)
    _progress(
        "train_candidate_start",
        root=str(root),
        seed=int(seed),
        candidate_version=int(version),
        sampling_seed=int(sampling_seed),
        max_value_states=int(max_value_states),
        eval_games=int(eval_games),
        eval_parallel=int(eval_parallel),
        benchmark_games=int(benchmark_games),
        mcts_iterations=int(mcts_iterations),
    )
    replay_paths, policy_paths = _replay_path_snapshot(root)
    _progress(
        "train_candidate_replay_paths_ready",
        feature_replay_paths=int(len(replay_paths)),
        policy_replay_paths=int(len(policy_paths)),
    )
    if not replay_paths:
        raise RuntimeError(f"no feature-complete replay partitions or legacy replay found under {root / 'global_replay'}")
    if not policy_paths:
        raise RuntimeError(f"no policy replay partitions or legacy replay found under {root / 'global_replay'}")
    use_indexed_replay = str(AGZ_REPLAY_SAMPLER) == "indexed_v1"
    controller_policy_root_sample_target = (
        int(XL_CONTROLLER_POLICY_SAMPLE_CAP)
        if use_indexed_replay
        else _policy_root_sample_target(int(XL_CONTROLLER_POLICY_SAMPLE_CAP))
    )
    adversary_policy_root_sample_target = (
        int(ADVERSARY_POLICY_SAMPLE_CAP)
        if use_indexed_replay
        else _policy_root_sample_target(
            int(ADVERSARY_POLICY_SAMPLE_CAP),
            explicit_target=int(ADVERSARY_POLICY_ROOT_SAMPLE_TARGET),
        )
    )
    controller_value_state_target, adversary_value_state_target = _value_role_sample_targets(
        int(max_value_states),
        int(AGZ_MAX_ADVERSARY_VALUE_STATES),
    )
    _progress(
        "train_candidate_state_sampling_start",
        candidate_version=int(version),
        sampling_seed=int(sampling_seed),
        controller_policy_root_sample_target=int(controller_policy_root_sample_target),
        adversary_policy_root_sample_target=int(adversary_policy_root_sample_target),
        controller_value_state_target=int(controller_value_state_target),
        adversary_value_state_target=int(adversary_value_state_target),
        replay_sampler=str(AGZ_REPLAY_SAMPLER),
    )
    if use_indexed_replay:
        from vidur.AlphaGoZero.indexed_replay import indexed_sample_training_data

        sampled, controller_roots, adversary_roots, replay_counts_sampled, state_cache_timings = indexed_sample_training_data(
            replay_paths,
            value_feature_schema=str(AGZ_VALUE_FEATURE_SCHEMA),
            seed=int(sampling_seed),
            controller_value_rows=int(controller_value_state_target),
            adversary_value_rows=int(adversary_value_state_target),
            controller_policy_roots=int(controller_policy_root_sample_target),
            adversary_policy_roots=int(adversary_policy_root_sample_target),
            cache_workers=int(AGZ_REPLAY_INDEX_BUILD_WORKERS),
            extraction_workers=int(AGZ_REPLAY_EXTRACTION_WORKERS),
        )
    else:
        sampled, controller_roots, adversary_roots, replay_counts_sampled, state_cache_timings = _stream_sample_state_rows_cached(
            replay_paths,
            seed=int(sampling_seed),
            max_value_rows=int(max_value_states),
            max_controller_policy_roots=int(controller_policy_root_sample_target),
            max_adversary_policy_roots=int(adversary_policy_root_sample_target),
            max_controller_value_rows=int(controller_value_state_target),
            max_adversary_value_rows=int(adversary_value_state_target),
        )
    _progress(
        "train_candidate_state_sampling_complete",
        value_rows_sampled=int(len(sampled)),
        controller_policy_roots_sampled=int(len(controller_roots)),
        adversary_policy_roots_sampled=int(len(adversary_roots)),
        controller_value_states_sampled=int(sum(str(row.get("player")) == "controller" for row in sampled)),
        adversary_value_states_sampled=int(sum(str(row.get("player")) == "adversary" for row in sampled)),
        replay_controller_rows_sampled=int(replay_counts_sampled.get("controller", 0)),
        replay_adversary_rows_sampled=int(replay_counts_sampled.get("adversary", 0)),
        state_cache_total_elapsed_s=float(state_cache_timings.get("state_cache_total_elapsed_s", 0.0)),
        state_cache_files=int(state_cache_timings.get("state_cache_files", 0)),
        state_cache_files_rebuilt=int(state_cache_timings.get("state_cache_files_rebuilt", 0)),
    )
    if not sampled:
        raise RuntimeError(f"no feature-complete replay rows found in {len(replay_paths)} replay file(s)")
    if int(replay_counts_sampled["controller"]) < int(MIN_CONTROLLER_STATES_FOR_EVAL):
        raise RuntimeError(
            f"not enough controller feature rows: {replay_counts_sampled['controller']} < {MIN_CONTROLLER_STATES_FOR_EVAL}"
        )
    if int(replay_counts_sampled["adversary"]) < int(MIN_ADVERSARY_STATES_FOR_EVAL):
        raise RuntimeError(
            f"not enough adversary feature rows: {replay_counts_sampled['adversary']} < {MIN_ADVERSARY_STATES_FOR_EVAL}"
        )

    t0 = time.time()
    _progress(
        "train_candidate_policy_cache_lookup_start",
        policy_paths=int(len(policy_paths)),
        controller_roots=int(len(controller_roots)),
        adversary_roots=int(len(adversary_roots)),
    )
    controller_policy_states: list[MarkovValueFeatures] | None = None
    adversary_policy_states: list[MarkovValueFeatures] | None = None
    if use_indexed_replay:
        from vidur.AlphaGoZero.indexed_replay import (
            IndexedRoots,
            materialize_markov_policy_arrays,
            materialize_policy_arrays,
        )

        if not isinstance(controller_roots, IndexedRoots) or not isinstance(adversary_roots, IndexedRoots):
            raise TypeError("indexed replay did not return direct policy-root locators")
        if AGZ_POLICY_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA:
            (
                controller_policy_states,
                Xc,
                yc,
                pc,
                offc,
            ), controller_policy_index = materialize_markov_policy_arrays(
                controller_roots,
                action_dim=int(CTRL_ACTION_DIM),
                root_cap=int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
            )
            (
                adversary_policy_states,
                Xa,
                ya,
                pa,
                offa,
            ), adversary_policy_index = materialize_markov_policy_arrays(
                adversary_roots,
                action_dim=int(ADV_ACTION_DIM),
                root_cap=int(ADVERSARY_POLICY_SAMPLE_CAP),
            )
        else:
            (Xc, yc, pc, offc), controller_policy_index = materialize_policy_arrays(
                controller_roots,
                action_dim=int(CTRL_ACTION_DIM),
                root_cap=int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
            )
            (Xa, ya, pa, offa), adversary_policy_index = materialize_policy_arrays(
                adversary_roots,
                action_dim=int(ADV_ACTION_DIM),
                root_cap=int(ADVERSARY_POLICY_SAMPLE_CAP),
            )
        selected_action_rows = int(controller_policy_index["action_rows"]) + int(adversary_policy_index["action_rows"])
        policy_scan_timings = {
            "policy_cache_files": int(state_cache_timings.get("indexed_cache_files", 0)),
            "policy_cache_files_rebuilt": int(state_cache_timings.get("indexed_cache_files_rebuilt", 0)),
            "policy_cache_files_missing_at_load": 0,
            "policy_cache_ensure_elapsed_s": 0.0,
            "policy_cache_lookup_elapsed_s": 0.0,
            "policy_cache_total_elapsed_s": 0.0,
            "policy_cache_roots_seen": int(replay_counts_sampled.get("controller_policy_roots", 0)) + int(replay_counts_sampled.get("adversary_policy_roots", 0)),
            "policy_cache_action_rows_seen": int(selected_action_rows),
            "policy_replay_rows_scanned": int(selected_action_rows),
            "policy_replay_scan_elapsed_s": 0.0,
            "policy_array_materialize_elapsed_s": 0.0,
            "controller_policy_roots_with_actions_available": int(len(controller_roots)),
            "adversary_policy_roots_with_actions_available": int(len(adversary_roots)),
            "controller_policy_roots_with_actions": int(controller_policy_index["roots_with_actions"]),
            "adversary_policy_roots_with_actions": int(adversary_policy_index["roots_with_actions"]),
            "controller_policy_action_rows": int(controller_policy_index["action_rows"]),
            "adversary_policy_action_rows": int(adversary_policy_index["action_rows"]),
        }
    else:
        (Xc, yc, pc, offc), (Xa, ya, pa, offa), policy_scan_timings = _collect_policy_training_arrays_for_players_cached(
            policy_paths,
            controller_roots,
            adversary_roots,
            controller_root_cap=int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
            adversary_root_cap=int(ADVERSARY_POLICY_SAMPLE_CAP),
        )
    _progress(
        "train_candidate_policy_cache_lookup_complete",
        controller_policy_rows=int(Xc.shape[0]),
        adversary_policy_rows=int(Xa.shape[0]),
        controller_policy_roots_with_actions=int(policy_scan_timings.get("controller_policy_roots_with_actions", 0)),
        adversary_policy_roots_with_actions=int(policy_scan_timings.get("adversary_policy_roots_with_actions", 0)),
        controller_policy_roots_with_actions_available=int(policy_scan_timings.get("controller_policy_roots_with_actions_available", 0)),
        adversary_policy_roots_with_actions_available=int(policy_scan_timings.get("adversary_policy_roots_with_actions_available", 0)),
        policy_cache_total_elapsed_s=float(policy_scan_timings.get("policy_cache_total_elapsed_s", 0.0)),
        policy_cache_files=int(policy_scan_timings.get("policy_cache_files", 0)),
        policy_cache_files_rebuilt=int(policy_scan_timings.get("policy_cache_files_rebuilt", 0)),
    )
    if Xc.shape[0] == 0 or Xa.shape[0] == 0:
        raise RuntimeError(f"policy rows missing: controller_rows={Xc.shape[0]} adversary_rows={Xa.shape[0]}")
    controller_policy_roots_required = _available_policy_root_requirement(
        int(policy_scan_timings.get("controller_policy_roots_with_actions_available", 0)),
        int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
    )
    adversary_policy_roots_required = _available_policy_root_requirement(
        int(policy_scan_timings.get("adversary_policy_roots_with_actions_available", 0)),
        int(ADVERSARY_POLICY_SAMPLE_CAP),
    )
    if controller_policy_roots_required <= 0 or adversary_policy_roots_required <= 0:
        raise RuntimeError(
            "policy roots with actions missing: "
            f"controller={controller_policy_roots_required} adversary={adversary_policy_roots_required}"
        )
    if int(policy_scan_timings["controller_policy_roots_with_actions"]) < int(controller_policy_roots_required):
        _progress(
            "train_candidate_failed",
            reason="not_enough_controller_policy_roots_with_actions",
            controller_policy_roots_with_actions=int(policy_scan_timings["controller_policy_roots_with_actions"]),
            controller_policy_sample_cap=int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
            controller_policy_roots_required=int(controller_policy_roots_required),
            controller_policy_roots_sampled=int(len(controller_roots)),
            controller_policy_roots_available=int(policy_scan_timings.get("controller_policy_roots_with_actions_available", 0)),
        )
        raise RuntimeError(
            "not enough controller policy roots with actions: "
            f"{policy_scan_timings['controller_policy_roots_with_actions']} < {controller_policy_roots_required} "
            f"(sampled={len(controller_roots)}, available={policy_scan_timings.get('controller_policy_roots_with_actions_available')}, "
            f"target={controller_policy_root_sample_target}, cap={XL_CONTROLLER_POLICY_SAMPLE_CAP})"
        )
    if int(policy_scan_timings["adversary_policy_roots_with_actions"]) < int(adversary_policy_roots_required):
        _progress(
            "train_candidate_failed",
            reason="not_enough_adversary_policy_roots_with_actions",
            adversary_policy_roots_with_actions=int(policy_scan_timings["adversary_policy_roots_with_actions"]),
            adversary_policy_sample_cap=int(ADVERSARY_POLICY_SAMPLE_CAP),
            adversary_policy_roots_required=int(adversary_policy_roots_required),
            adversary_policy_roots_sampled=int(len(adversary_roots)),
            adversary_policy_roots_available=int(policy_scan_timings.get("adversary_policy_roots_with_actions_available", 0)),
        )
        raise RuntimeError(
            "not enough adversary policy roots with actions: "
            f"{policy_scan_timings['adversary_policy_roots_with_actions']} < {adversary_policy_roots_required} "
            f"(sampled={len(adversary_roots)}, available={policy_scan_timings.get('adversary_policy_roots_with_actions_available')}, "
            f"target={adversary_policy_root_sample_target}, cap={ADVERSARY_POLICY_SAMPLE_CAP})"
        )

    Xv, yv = _value_arrays_from_rows(sampled)
    dataset_preparation_elapsed_s = float(time.time() - dataset_preparation_started)
    _progress(
        "train_candidate_dataset_preparation_complete",
        candidate_version=int(version),
        replay_sampler=str(AGZ_REPLAY_SAMPLER),
        elapsed_s=float(dataset_preparation_elapsed_s),
        under_eight_minutes=bool(dataset_preparation_elapsed_s < 480.0),
    )
    promoted = _current_promoted_bundle(root)
    training_parent = (
        _latest_dnn_training_parent_bundle(root, next_version=int(version), promoted=promoted)
        if AGZ_MODEL_FAMILY == "dnn"
        else promoted
    )
    _progress(
        "train_candidate_parent_selected",
        version=int(version),
        controller_training_parent_version=int(training_parent.controller_model_version),
        adversary_training_parent_version=int(training_parent.adversary_model_version),
        controller_promoted_selfplay_version=int(promoted.controller_model_version),
        adversary_promoted_selfplay_version=int(promoted.adversary_model_version),
    )
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / f"Model_Version{int(version)}"
    if out.exists() and (out / "candidate_manifest.json").exists():
        raise RuntimeError(f"candidate output already exists with manifest: {out}")
    tmp_out = models_dir / f".Model_Version{int(version)}.tmp_{os.getpid()}_{time.strftime('%Y%m%d_%H%M%S')}"
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=False)
    candidate_published = False

    try:
        _progress("train_candidate_value_fit_start", version=int(version), value_rows=int(yv.shape[0]))
        if AGZ_MODEL_FAMILY == "dnn":
            controller_value_model, adversary_value_model, value_fit_timings = _fit_dnn_value_models_parallel(
                Xv,
                yv,
                seed=int(seed),
                version=int(version),
                training_parent=training_parent,
            )
        else:
            controller_value_model, adversary_value_model, value_fit_timings = _fit_value_models_parallel(
                Xv,
                yv,
                seed=int(seed),
                version=int(version),
            )
        _progress(
            "train_candidate_value_fit_complete",
            version=int(version),
            value_fit_mode=str(value_fit_timings.get("value_fit_mode", "")),
            controller_value_fit_elapsed_s=float(value_fit_timings.get("controller_value_fit_elapsed_s", 0.0)),
            adversary_value_fit_elapsed_s=float(value_fit_timings.get("adversary_value_fit_elapsed_s", 0.0)),
            value_fit_wall_elapsed_s=float(value_fit_timings.get("value_fit_wall_elapsed_s", 0.0)),
        )
        if AGZ_VALUE_FEATURE_SCHEMA == MARKOV_VALUE_SCHEMA:
            controller_value_pred = controller_value_model.predict_structured(Xv).astype(np.float32)
            adversary_value_pred = adversary_value_model.predict_structured(Xv).astype(np.float32)
        else:
            controller_value_pred = controller_value_model.predict(Xv).astype(np.float32)
            adversary_value_pred = adversary_value_model.predict(Xv).astype(np.float32)
        controller_value_metrics = _value_metrics(yv, controller_value_pred)
        adversary_value_metrics = _value_metrics(yv, adversary_value_pred)
        value_metrics = controller_value_metrics

        controller_value_dir = tmp_out / "controller_value" / CONFIG_NAME
        adversary_value_dir = tmp_out / "adversary_value" / CONFIG_NAME
        controller_value_dir.mkdir(parents=True, exist_ok=True)
        adversary_value_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(controller_value_model, controller_value_dir / "model.joblib", compress=3)
        joblib.dump(adversary_value_model, adversary_value_dir / "model.joblib", compress=3)
        if AGZ_MODEL_FAMILY == "dnn":
            from vidur.AlphaGoZero.dnn_models import export_dnn_to_native, write_artifact_metadata

            export_dnn_to_native(
                controller_value_model,
                controller_value_dir / "native_model.tsv",
                model_tag=f"agz_controller_value_v{version}",
            )
            export_dnn_to_native(
                adversary_value_model,
                adversary_value_dir / "native_model.tsv",
                model_tag=f"agz_adversary_value_v{version}",
            )
            write_artifact_metadata(controller_value_model, controller_value_dir / "metadata.json")
            write_artifact_metadata(adversary_value_model, adversary_value_dir / "metadata.json")
        else:
            _export_hgb_to_native_text(
                controller_value_model,
                controller_value_dir / "native_model.tsv",
                feature_dim_override=226,
                model_tag_override=f"agz_controller_value_v{version}",
            )
            _export_hgb_to_native_text(
                adversary_value_model,
                adversary_value_dir / "native_model.tsv",
                feature_dim_override=226,
                model_tag_override=f"agz_adversary_value_v{version}",
            )

        _progress(
            "train_candidate_policy_fit_start",
            version=int(version),
            controller_policy_rows=int(Xc.shape[0]),
            adversary_policy_rows=int(Xa.shape[0]),
        )
        if AGZ_MODEL_FAMILY == "dnn":
            ctrl_model, adv_model, policy_fit_timings = _fit_dnn_policy_models_parallel(
                Xc,
                pc,
                offc,
                Xa,
                pa,
                offa,
                controller_policy_states,
                adversary_policy_states,
                seed=int(seed),
                version=int(version),
                training_parent=training_parent,
            )
        else:
            ctrl_model, adv_model, policy_fit_timings = _fit_policy_models_parallel(
                Xc,
                yc,
                Xa,
                ya,
                seed=int(seed),
                version=int(version),
            )
        _progress(
            "train_candidate_policy_fit_complete",
            version=int(version),
            policy_fit_mode=str(policy_fit_timings.get("policy_fit_mode", "")),
            controller_policy_fit_elapsed_s=float(policy_fit_timings.get("controller_policy_fit_elapsed_s", 0.0)),
            adversary_policy_fit_elapsed_s=float(policy_fit_timings.get("adversary_policy_fit_elapsed_s", 0.0)),
            policy_fit_wall_elapsed_s=float(policy_fit_timings.get("policy_fit_wall_elapsed_s", 0.0)),
        )
        _progress(
            "train_candidate_policy_metrics_start",
            version=int(version),
            configured_workers=int(POLICY_METRICS_WORKERS),
            controller_policy_roots=int(len(offc)),
            adversary_policy_roots=int(len(offa)),
        )
        ctrl_metrics, adv_metrics, policy_metric_timings = _policy_metrics_parallel_pair(
            ctrl_model,
            Xc,
            yc,
            pc,
            offc,
            adv_model,
            Xa,
            ya,
            pa,
            offa,
            controller_policy_states,
            adversary_policy_states,
            max_workers=int(POLICY_METRICS_WORKERS),
        )
        _progress(
            "train_candidate_policy_metrics_complete",
            version=int(version),
            policy_metrics_mode=str(policy_metric_timings["policy_metrics_mode"]),
            policy_metrics_workers=int(policy_metric_timings["policy_metrics_workers"]),
            policy_metrics_tasks=int(policy_metric_timings["policy_metrics_tasks"]),
            policy_metrics_wall_elapsed_s=float(
                policy_metric_timings["policy_metrics_wall_elapsed_s"]
            ),
        )

        ctrl_dir = tmp_out / "controller_prior" / POLICY_CONFIG_NAME
        adv_dir = tmp_out / "adversary_prior" / POLICY_CONFIG_NAME
        ctrl_dir.mkdir(parents=True, exist_ok=True)
        adv_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(ctrl_model, ctrl_dir / "model.joblib", compress=3)
        joblib.dump(adv_model, adv_dir / "model.joblib", compress=3)
        if AGZ_MODEL_FAMILY == "dnn":
            from vidur.AlphaGoZero.dnn_models import export_dnn_to_native, write_artifact_metadata

            export_dnn_to_native(
                ctrl_model,
                ctrl_dir / "native_model.tsv",
                model_tag=f"agz_controller_prior_v{version}",
            )
            export_dnn_to_native(
                adv_model,
                adv_dir / "native_model.tsv",
                model_tag=f"agz_adversary_prior_v{version}",
            )
            write_artifact_metadata(ctrl_model, ctrl_dir / "metadata.json")
            write_artifact_metadata(adv_model, adv_dir / "metadata.json")
        else:
            _export_hgb_to_native_text(ctrl_model, ctrl_dir / "native_model.tsv", feature_dim_override=269, model_tag_override=f"agz_controller_prior_v{version}")
            _export_hgb_to_native_text(adv_model, adv_dir / "native_model.tsv", feature_dim_override=233, model_tag_override=f"agz_adversary_prior_v{version}")

        dnn_parity: dict[str, Any] = {}
        if AGZ_MODEL_FAMILY == "dnn":
            for role, value_path in (
                ("controller", controller_value_dir / "model.joblib"),
                ("adversary", adversary_value_dir / "model.joblib"),
            ):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "vidur.AlphaGoZero.test_and_analysis.test_dnn_native_parity",
                        "--value-model",
                        str(value_path),
                        "--controller-policy-model",
                        str(ctrl_dir / "model.joblib"),
                        "--adversary-policy-model",
                        str(adv_dir / "model.joblib"),
                        "--rows",
                        "128",
                        "--tolerance",
                        "1e-4",
                    ],
                    cwd=str(Path(__file__).resolve().parents[2]),
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if completed.returncode != 0:
                    parity_output = str(completed.stdout or "").strip()
                    _progress(
                        "train_candidate_dnn_parity_failed",
                        version=int(version),
                        role=str(role),
                        returncode=int(completed.returncode),
                        output=parity_output,
                    )
                    raise RuntimeError(
                        f"DNN Python/native parity failed for {role} value model "
                        f"(exit {completed.returncode}): {parity_output}"
                    )
                dnn_parity[role] = json.loads(completed.stdout)
        rollout_horizon_used = active_rollout_horizon_sec(root)
        next_rollout_horizon = configured_rollout_horizon(
            controller_value_metrics["p95_abs_error"]
        )
        metrics = {
            "model_config": CONFIG_NAME,
            "model_version": int(version),
            "time_of_training_24h": local_time_24h(),
            "sampled_controller_states": int(sum(1 for r in sampled if str(r.get("player")) == "controller")),
            "sampled_adversary_states": int(sum(1 for r in sampled if str(r.get("player")) == "adversary")),
            "value_mse": value_metrics["mse"],
            "value_rmse": value_metrics["rmse"],
            "value_max_abs_error": value_metrics["max_abs_error"],
            "value_p95_abs_error": value_metrics["p95_abs_error"],
            "controller_value_mse": controller_value_metrics["mse"],
            "controller_value_rmse": controller_value_metrics["rmse"],
            "controller_value_max_abs_error": controller_value_metrics["max_abs_error"],
            "controller_value_p95_abs_error": controller_value_metrics["p95_abs_error"],
            "adversary_value_mse": adversary_value_metrics["mse"],
            "adversary_value_rmse": adversary_value_metrics["rmse"],
            "adversary_value_max_abs_error": adversary_value_metrics["max_abs_error"],
            "adversary_value_p95_abs_error": adversary_value_metrics["p95_abs_error"],
            "controller_policy_mse": ctrl_metrics["mse"],
            "controller_policy_cross_entropy": ctrl_metrics["cross_entropy"],
            "controller_policy_top1": ctrl_metrics["top1"],
            "controller_policy_top3": ctrl_metrics["top3"],
            "adversary_policy_mse": adv_metrics["mse"],
            "adversary_policy_cross_entropy": adv_metrics["cross_entropy"],
            "adversary_policy_top1": adv_metrics["top1"],
            "adversary_policy_top3": adv_metrics["top3"],
            "rollout_horizon_used_sec": float(rollout_horizon_used),
            "rollout_horizon_source_controller_p95_abs_error": float(
                next_rollout_horizon.controller_p95_abs_error
            ),
            "rollout_value_error_threshold": float(
                next_rollout_horizon.value_error_threshold
            ),
            "rollout_discount_factor": float(next_rollout_horizon.discount_factor),
            "rollout_reference_step_sec": float(next_rollout_horizon.reference_step_sec),
            "rollout_max_horizon_sec": float(next_rollout_horizon.max_horizon_sec),
            "rollout_horizon_tick_sec": float(next_rollout_horizon.tick_sec),
            "next_rollout_horizon_raw_sec": float(next_rollout_horizon.raw_horizon_sec),
            "next_rollout_horizon_calculated_sec": float(
                next_rollout_horizon.calculated_horizon_sec
            ),
            "next_rollout_horizon_rounded_sec": float(
                next_rollout_horizon.rounded_horizon_sec
            ),
            "next_rollout_discounted_error": float(
                next_rollout_horizon.discounted_error_at_rounded_horizon
            ),
        }
        metrics_extra = {
            "mcts_iterations": int(mcts_iterations),
            "replay_sampler": str(AGZ_REPLAY_SAMPLER),
            "sampling_seed": int(sampling_seed),
            "indexed_address_selection_elapsed_s": float(state_cache_timings.get("indexed_address_selection_elapsed_s", 0.0)),
            "indexed_materialize_elapsed_s": float(state_cache_timings.get("indexed_materialize_elapsed_s", 0.0)),
            "dataset_preparation_elapsed_s": float(dataset_preparation_elapsed_s),
            "indexed_controller_value_sample_sha256": str(state_cache_timings.get("indexed_controller_value_sample_sha256", "")),
            "indexed_adversary_value_sample_sha256": str(state_cache_timings.get("indexed_adversary_value_sample_sha256", "")),
            "indexed_controller_policy_sample_sha256": str(state_cache_timings.get("indexed_controller_policy_sample_sha256", "")),
            "indexed_adversary_policy_sample_sha256": str(state_cache_timings.get("indexed_adversary_policy_sample_sha256", "")),
            "state_cache_files": int(state_cache_timings["state_cache_files"]),
            "state_cache_files_rebuilt": int(state_cache_timings["state_cache_files_rebuilt"]),
            "state_cache_files_missing_at_load": int(state_cache_timings.get("state_cache_files_missing_at_load", 0)),
            "state_cache_ensure_elapsed_s": float(state_cache_timings["state_cache_ensure_elapsed_s"]),
            "state_cache_sample_elapsed_s": float(state_cache_timings["state_cache_sample_elapsed_s"]),
            "state_cache_total_elapsed_s": float(state_cache_timings["state_cache_total_elapsed_s"]),
            "policy_root_oversample_factor": float(POLICY_ROOT_OVERSAMPLE_FACTOR),
            "controller_policy_root_sample_target": int(controller_policy_root_sample_target),
            "adversary_policy_root_sample_target": int(adversary_policy_root_sample_target),
            "controller_policy_roots_sampled": int(len(controller_roots)),
            "adversary_policy_roots_sampled": int(len(adversary_roots)),
            "controller_policy_roots_required": int(controller_policy_roots_required),
            "adversary_policy_roots_required": int(adversary_policy_roots_required),
            "controller_policy_sample_cap": int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
            "adversary_policy_sample_cap": int(ADVERSARY_POLICY_SAMPLE_CAP),
            "value_fit_mode": str(value_fit_timings["value_fit_mode"]),
            "controller_value_fit_elapsed_s": float(value_fit_timings["controller_value_fit_elapsed_s"]),
            "adversary_value_fit_elapsed_s": float(value_fit_timings["adversary_value_fit_elapsed_s"]),
            "value_fit_wall_elapsed_s": float(value_fit_timings["value_fit_wall_elapsed_s"]),
            "policy_fit_mode": str(policy_fit_timings["policy_fit_mode"]),
            "model_family": str(AGZ_MODEL_FAMILY),
            "value_feature_schema": str(AGZ_VALUE_FEATURE_SCHEMA),
            "hgb_openmp_threads": policy_fit_timings.get("hgb_openmp_threads", ""),
            "policy_cache_files": int(policy_scan_timings["policy_cache_files"]),
            "policy_cache_files_rebuilt": int(policy_scan_timings["policy_cache_files_rebuilt"]),
            "policy_cache_files_missing_at_load": int(policy_scan_timings.get("policy_cache_files_missing_at_load", 0)),
            "policy_cache_ensure_elapsed_s": float(policy_scan_timings["policy_cache_ensure_elapsed_s"]),
            "policy_cache_lookup_elapsed_s": float(policy_scan_timings["policy_cache_lookup_elapsed_s"]),
            "policy_cache_total_elapsed_s": float(policy_scan_timings["policy_cache_total_elapsed_s"]),
            "policy_cache_roots_seen": int(policy_scan_timings["policy_cache_roots_seen"]),
            "policy_cache_action_rows_seen": int(policy_scan_timings["policy_cache_action_rows_seen"]),
            "policy_replay_rows_scanned": int(policy_scan_timings["policy_replay_rows_scanned"]),
            "policy_replay_scan_elapsed_s": float(policy_scan_timings["policy_replay_scan_elapsed_s"]),
            "policy_array_materialize_elapsed_s": float(policy_scan_timings["policy_array_materialize_elapsed_s"]),
            "controller_policy_roots_with_actions_available": int(policy_scan_timings.get("controller_policy_roots_with_actions_available", policy_scan_timings["controller_policy_roots_with_actions"])),
            "adversary_policy_roots_with_actions_available": int(policy_scan_timings.get("adversary_policy_roots_with_actions_available", policy_scan_timings["adversary_policy_roots_with_actions"])),
            "controller_policy_roots_with_actions": int(policy_scan_timings["controller_policy_roots_with_actions"]),
            "adversary_policy_roots_with_actions": int(policy_scan_timings["adversary_policy_roots_with_actions"]),
            "controller_policy_action_rows": int(policy_scan_timings["controller_policy_action_rows"]),
            "adversary_policy_action_rows": int(policy_scan_timings["adversary_policy_action_rows"]),
            "controller_policy_fit_elapsed_s": float(policy_fit_timings["controller_policy_fit_elapsed_s"]),
            "adversary_policy_fit_elapsed_s": float(policy_fit_timings["adversary_policy_fit_elapsed_s"]),
            "policy_fit_wall_elapsed_s": float(policy_fit_timings["policy_fit_wall_elapsed_s"]),
            "policy_metrics_mode": str(policy_metric_timings["policy_metrics_mode"]),
            "policy_metrics_workers": int(policy_metric_timings["policy_metrics_workers"]),
            "policy_metrics_tasks": int(policy_metric_timings["policy_metrics_tasks"]),
            "policy_metrics_wall_elapsed_s": float(
                policy_metric_timings["policy_metrics_wall_elapsed_s"]
            ),
        }
        manifest = {
            **metrics,
            **metrics_extra,
            "value_model_path": str(out / "controller_value" / CONFIG_NAME / "model.joblib"),
            "controller_value_model_path": str(out / "controller_value" / CONFIG_NAME / "model.joblib"),
            "adversary_value_model_path": str(out / "adversary_value" / CONFIG_NAME / "model.joblib"),
            "controller_prior_model_path": str(out / "controller_prior" / POLICY_CONFIG_NAME / "model.joblib"),
            "adversary_prior_model_path": str(out / "adversary_prior" / POLICY_CONFIG_NAME / "model.joblib"),
            "fit_elapsed_s": float(time.time() - t0),
            "created_at_utc": utc_now(),
            "model_family": str(AGZ_MODEL_FAMILY),
            "native_ready": bool(AGZ_MODEL_FAMILY != "dnn" or dnn_parity),
            "dnn_parity": dnn_parity,
            "target_perspective": "controller",
            "controller_parent_model_version": int(training_parent.controller_model_version),
            "adversary_parent_model_version": int(training_parent.adversary_model_version),
            "controller_value_parent_sha256": str(getattr(controller_value_model, "training_metadata", {}).get("parent_sha256", "")),
            "adversary_value_parent_sha256": str(getattr(adversary_value_model, "training_metadata", {}).get("parent_sha256", "")),
            "incremental_update": bool(AGZ_MODEL_FAMILY == "dnn"),
            "training_lineage": "latest_published_candidate_checkpoint",
            "selfplay_lineage": "current_promoted_role_bundle",
            "controller_promoted_selfplay_model_version": int(promoted.controller_model_version),
            "adversary_promoted_selfplay_model_version": int(promoted.adversary_model_version),
            "failed_candidate_updates_accumulate": True,
            "promotion_win_rate_threshold": float(PROMOTION_WIN_RATE_THRESHOLD),
            "role_promotion_win_threshold": int(ROLE_PROMOTION_WIN_THRESHOLD),
            "promotion_required_wins_per_role": int(ROLE_PROMOTION_WIN_THRESHOLD) + 1,
            "promotion_rule": f"role_independent_current_baseline_{int(eval_games)}_each_gt_{int(ROLE_PROMOTION_WIN_THRESHOLD)}",
            "eval_status": "running",
        }
        atomic_write_json(tmp_out / "candidate_manifest.json", manifest)
        _retire_incomplete_candidate_dir(out)
        tmp_out.rename(out)
        candidate_published = True
        _progress("train_candidate_published", version=int(version), output_dir=str(out))

        append_csv_row(root / "train_model.csv", TRAIN_MODEL_FIELDS, metrics)
        candidate = _candidate_bundle_from_output(int(version), out)
        del sampled
        del controller_roots
        del adversary_roots
        del Xv, yv, controller_value_pred, adversary_value_pred
        del Xc, yc, pc, offc
        del Xa, ya, pa, offa
        del controller_value_model, adversary_value_model, ctrl_model, adv_model
        gc.collect()
        try:
            _progress(
                "train_candidate_eval_start",
                version=int(version),
                eval_games=int(eval_games),
                eval_parallel=int(eval_parallel),
                benchmark_games=int(benchmark_games),
            )
            eval_result = evaluate_and_maybe_promote(
                root,
                candidate=candidate,
                eval_games=int(eval_games),
                eval_parallel=int(eval_parallel),
                benchmark_games=int(benchmark_games),
                iterations=int(mcts_iterations),
                history_seed=int(seed) + int(version),
            )
        except Exception as exc:
            _progress("train_candidate_eval_failed", version=int(version), error=repr(exc))
            manifest.update({"eval_status": "failed", "eval_error": repr(exc), "updated_at_utc": utc_now()})
            atomic_write_json(out / "candidate_manifest.json", manifest)
            raise
        _progress("train_candidate_eval_complete", version=int(version), **eval_result)
        manifest.update({"eval_status": "complete", **eval_result})
        atomic_write_json(out / "candidate_manifest.json", manifest)
        return {**metrics, **eval_result}
    except Exception as exc:
        _progress("train_candidate_failed", version=int(version), error=repr(exc), candidate_published=bool(candidate_published))
        if not candidate_published:
            shutil.rmtree(tmp_out, ignore_errors=True)
        raise


def evaluate_existing_candidate(
    root: Path,
    *,
    version: int,
    seed: int,
    eval_games: int,
    eval_parallel: int,
    benchmark_games: int,
    mcts_iterations: int,
) -> dict[str, Any]:
    root = Path(root)
    model_dir = root / "models" / f"Model_Version{int(version)}"
    manifest_path = model_dir / "candidate_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("eval_status", "")) == "complete":
        raise RuntimeError(f"candidate v{version} evaluation is already complete")
    promoted = _current_promoted_bundle(root)
    if int(promoted.controller_model_version) >= int(version) or int(promoted.adversary_model_version) >= int(version):
        raise RuntimeError(
            "refusing to reevaluate a candidate whose role was already promoted: "
            f"controller={promoted.controller_model_version} adversary={promoted.adversary_model_version}"
        )
    manifest.update({"eval_status": "running", "eval_error": "", "updated_at_utc": utc_now()})
    atomic_write_json(manifest_path, manifest)
    candidate = _candidate_bundle_from_output(int(version), model_dir)
    _progress(
        "existing_candidate_eval_start",
        version=int(version),
        eval_games=int(eval_games),
        benchmark_games=int(benchmark_games),
        distributed=bool(distributed_eval_enabled()),
    )
    try:
        result = evaluate_and_maybe_promote(
            root,
            candidate=candidate,
            eval_games=int(eval_games),
            eval_parallel=int(eval_parallel),
            benchmark_games=int(benchmark_games),
            iterations=int(mcts_iterations),
            history_seed=int(seed) + int(version),
        )
    except Exception as exc:
        manifest.update({"eval_status": "failed", "eval_error": repr(exc), "updated_at_utc": utc_now()})
        atomic_write_json(manifest_path, manifest)
        _progress("existing_candidate_eval_failed", version=int(version), error=repr(exc))
        raise
    manifest.update({"eval_status": "complete", "eval_error": "", **result, "updated_at_utc": utc_now()})
    atomic_write_json(manifest_path, manifest)
    _progress("existing_candidate_eval_complete", version=int(version), **result)
    return result


def resume_sjf_for_existing_evaluation(
    root: Path,
    *,
    version: int,
    seed: int,
    eval_parallel: int,
    benchmark_games: int,
    mcts_iterations: int,
) -> dict[str, Any]:
    """Finish only SJF after role evaluation and promotion already succeeded."""

    root = Path(root)
    eval_dir = root / "eval_of_models" / f"eval_{int(version):06d}"
    summary_path = eval_dir / "eval_summary.json"
    manifest_path = (
        root / "models" / f"Model_Version{int(version)}" / "candidate_manifest.json"
    )
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            summary_path if not summary_path.is_file() else manifest_path
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(summary.get("candidate_model_version", 0)) != int(version):
        raise RuntimeError("evaluation summary candidate version mismatch")
    if not bool(summary.get("candidate_promoted", False)):
        raise RuntimeError("SJF resume is only valid after a role promotion")

    promoted = _current_promoted_bundle(root)
    expected_controller = int(summary.get("new_controller_model_version", 0))
    expected_adversary = int(summary.get("new_adversary_model_version", 0))
    if (
        int(promoted.controller_model_version) != expected_controller
        or int(promoted.adversary_model_version) != expected_adversary
    ):
        raise RuntimeError(
            "current promoted bundle no longer matches the completed role evaluation: "
            f"current=({promoted.controller_model_version},{promoted.adversary_model_version}) "
            f"expected=({expected_controller},{expected_adversary})"
        )

    manifest.update(
        {"eval_status": "running", "eval_error": "", "updated_at_utc": utc_now()}
    )
    atomic_write_json(manifest_path, manifest)
    _progress(
        "existing_sjf_resume_start",
        version=int(version),
        benchmark_games=int(benchmark_games),
    )
    try:
        with pause_selfplay_for_eval(root):
            sjf_results = _run_sjf_benchmark(
                root=root,
                eval_dir=eval_dir,
                candidate=promoted,
                benchmark_games=int(benchmark_games),
                eval_parallel=int(eval_parallel),
                iterations=int(mcts_iterations),
                history_seed=int(seed) + int(version),
            )
    except Exception as exc:
        manifest.update(
            {
                "eval_status": "failed",
                "eval_error": repr(exc),
                "updated_at_utc": utc_now(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        _progress("existing_sjf_resume_failed", version=int(version), error=repr(exc))
        raise

    result = {**summary, "sjf_results_csv": str(sjf_results)}
    manifest.update(
        {
            "eval_status": "complete",
            "eval_error": "",
            **result,
            "updated_at_utc": utc_now(),
        }
    )
    atomic_write_json(manifest_path, manifest)
    _progress(
        "existing_sjf_resume_complete",
        version=int(version),
        sjf_results_csv=str(sjf_results),
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one AGZ candidate HGB bundle from XL feature-complete replay.")
    p.add_argument("--output-root", type=Path, default=Path("/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero"))
    p.add_argument("--eval-existing-version", type=int, default=0)
    p.add_argument("--resume-sjf-version", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-value-states", type=int, default=int(os.environ.get("AGZ_MAX_VALUE_STATES", "500000")))
    p.add_argument("--eval-games", type=int, default=int(AGZ_EVAL_GAMES))
    p.add_argument("--eval-parallel", type=int, default=60)
    p.add_argument("--benchmark-games", type=int, default=int(AGZ_BENCHMARK_GAMES))
    p.add_argument("--mcts-iterations", type=int, default=int(AGZ_EVAL_MCTS_ITERATIONS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.resume_sjf_version) > 0:
        result = resume_sjf_for_existing_evaluation(
            Path(args.output_root).expanduser(),
            version=int(args.resume_sjf_version),
            seed=int(args.seed),
            eval_parallel=int(args.eval_parallel),
            benchmark_games=int(args.benchmark_games),
            mcts_iterations=int(args.mcts_iterations),
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    if int(args.eval_existing_version) > 0:
        result = evaluate_existing_candidate(
            Path(args.output_root).expanduser(),
            version=int(args.eval_existing_version),
            seed=int(args.seed),
            eval_games=int(args.eval_games),
            eval_parallel=int(args.eval_parallel),
            benchmark_games=int(args.benchmark_games),
            mcts_iterations=int(args.mcts_iterations),
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    metrics = train_candidate(
        Path(args.output_root).expanduser(),
        seed=int(args.seed),
        max_value_states=int(args.max_value_states),
        eval_games=int(args.eval_games),
        eval_parallel=int(args.eval_parallel),
        benchmark_games=int(args.benchmark_games),
        mcts_iterations=int(args.mcts_iterations),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
