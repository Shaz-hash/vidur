"""Cliff-aware classical controller value surrogate (ModelInputs schema).

This backend extracts features purely from GV3 `ModelInputs` (the same tensors
used by the neural value model) so the training distribution and the live MCTS
bootstrap distribution match exactly. Specifically:

  - Training time:   record -> state_loader(record) -> build_model_inputs ->
                     extract_cliff_features_from_inputs(inputs)
  - Bootstrap time:  state at MCTS leaf -> build_model_inputs(leaf_state, ...) ->
                     extract_cliff_features_from_inputs(inputs)

The previous experiment showed the per-iteration supervised fit is fine but the
same-version Bellman residual blows up - that is precisely what happens when
bootstrap features land outside the training distribution. Using `ModelInputs`
on both ends fixes the distribution gap.

On top of that shared base, we keep two GV3-specific design choices:

  1. Top-K "dangerous" request slots picked from prefill+decode rows by
     slack-to-drop and lateness, in fixed positions, so a single critical
     request cannot be averaged away.
  2. HistGradientBoosting with absolute-error loss (median regression) plus a
     dedicated tail-residual booster trained only on |y| >= 0.5 rows so the
     long left tail is not dominated by the zero-target majority.

The model stays under the 150k scalar parameter budget (see fit_seconds
metadata after training).
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


CLIFF_FEATURE_CACHE_VERSION = "cliff_aware_modelinputs_v2"
# When set to True via thread-local config, build_cliff_feature_matrix_from_records
# also appends action-aggregate features to every row. The cache key picks up an
# extra suffix so augmented and base caches don't collide.
_CLIFF_USE_ACTION_FEATURES = False
_CLIFF_ACTION_TOP_K = 4
DEFAULT_MODEL_BUDGET = 150_000

# Top-K dangerous request slots taken from the union of prefill and decode
# rows in the ModelInputs schema. Each slot keeps a fixed-size feature
# sub-vector so single critical requests survive into model splits without
# being averaged away.
TOP_K_DANGER_SLOTS = 16
DANGER_PER_SLOT_DIM = 9  # see _slot_features

# ModelInputs row schema (DNN/infer.py): [remaining_norm, age_norm,
# lateness_norm, slack_to_drop_norm, violated_bit]
ROW_SCHEMA_DIM = 5
PREFILL_ROW_LIMIT = 10
DECODE_ROW_LIMIT = 50

# Buckets for "how many active requests are within X of the SLO drop cliff".
SLACK_BUCKETS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
LATENESS_BUCKETS = (0.0, 0.05, 0.25, 0.50, 1.0)

# Player indicator bits to skip from the global feature vector. The same skip
# the existing bellman_shaped "model_inputs" path uses so train-time and
# bootstrap-time roots/child states agree on player encoding.
GLOBAL_PLAYER_BIT_INDICES = (0, 1)


# ---------------------------------------------------------------------------
# Worker globals (used during multi-process feature extraction)
# ---------------------------------------------------------------------------


_FEATURE_WORKER_STATE_LOADER: Any | None = None
_FEATURE_WORKER_PLAYER = "controller"


def _limit_native_threads(num_threads: int) -> None:
    threads = max(1, int(num_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=threads)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _tensor_to_2d(value: Any) -> np.ndarray:
    arr = value.detach().to("cpu").float().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        return np.asarray(arr.reshape(1, -1), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def _tensor_to_1d(value: Any, *, fallback_len: int) -> np.ndarray:
    if value is None:
        return np.ones((int(fallback_len),), dtype=np.float32)
    arr = value.detach().to("cpu").float().numpy()
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr.reshape(-1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-slot feature builder
# ---------------------------------------------------------------------------


def _slot_features(
    row: np.ndarray,
    *,
    is_prefill: bool,
) -> tuple[list[float], float]:
    """Return per-slot features and a danger score for sorting (smaller = worse)."""

    remaining_norm = float(row[0]) if row.shape[0] > 0 else 0.0
    age_norm = float(row[1]) if row.shape[0] > 1 else 0.0
    lateness_norm = float(row[2]) if row.shape[0] > 2 else 0.0
    slack_norm = float(row[3]) if row.shape[0] > 3 else 0.0
    violated = float(row[4]) if row.shape[0] > 4 else 0.0

    # lateness_norm and slack_to_drop_norm both come pre-normalized in the
    # ModelInputs schema, so we keep them in their native [0, 1+] range.
    lateness_to_drop_proxy = max(0.0, 1.0 - lateness_norm)
    feats = [
        remaining_norm,
        age_norm,
        lateness_norm,
        slack_norm,
        violated,
        lateness_to_drop_proxy,
        1.0 if is_prefill else 0.0,
        # Compound danger flags help boosted trees split early.
        1.0 if (slack_norm <= 0.0) else 0.0,
        1.0 if (lateness_norm > 0.0) else 0.0,
    ]
    assert len(feats) == DANGER_PER_SLOT_DIM, (
        f"DANGER_PER_SLOT_DIM mismatch: {len(feats)} vs {DANGER_PER_SLOT_DIM}"
    )

    # Danger score: violated rows rank above unviolated; among unviolated,
    # smaller slack and larger lateness rank as more dangerous.
    if violated >= 0.5:
        score = -1e3 + (1.0 - lateness_norm)
    else:
        score = -lateness_norm * 2.0 + slack_norm
    return feats, float(score)


def _row_iter(
    prefill: np.ndarray,
    prefill_mask: np.ndarray,
    decode: np.ndarray,
    decode_mask: np.ndarray,
):
    for i in range(int(prefill.shape[0])):
        m = float(prefill_mask[i]) if i < prefill_mask.shape[0] else 1.0
        if m >= 0.5:
            yield prefill[i], True
    for i in range(int(decode.shape[0])):
        m = float(decode_mask[i]) if i < decode_mask.shape[0] else 1.0
        if m >= 0.5:
            yield decode[i], False


# ---------------------------------------------------------------------------
# Cliff feature builder (consumes ModelInputs)
# ---------------------------------------------------------------------------


def extract_cliff_features_from_inputs(inputs: Any) -> tuple[list[float], list[str]]:
    """Build a cliff-aware feature vector from one GV3 ModelInputs object.

    The same function is used at training time (after state_loader rebuilds
    the GV3 state and DNN/infer.build_model_inputs converts to tensors) and
    at MCTS bootstrap time (where the search loop already has ModelInputs).
    Keeping a single feature path means train and bootstrap distributions
    match exactly which is required for Bellman residual convergence.
    """

    values: list[float] = []
    names: list[str] = []

    # ---------- Global features pass-through (skip player bits) ----------
    global_arr = _tensor_to_2d(inputs.global_features).reshape(-1)
    for idx, val in enumerate(global_arr.tolist()):
        if idx in GLOBAL_PLAYER_BIT_INDICES:
            continue
        names.append(f"global_{idx:03d}")
        values.append(float(val))

    # ---------- Per-row prefill / decode tensors ----------
    prefill = _tensor_to_2d(inputs.prefill_req_features)
    decode = _tensor_to_2d(inputs.decode_req_features)
    prefill_mask = _tensor_to_1d(inputs.prefill_req_mask, fallback_len=int(prefill.shape[0]))
    decode_mask = _tensor_to_1d(inputs.decode_req_mask, fallback_len=int(decode.shape[0]))

    # Pad/truncate to the canonical row limits so the schema is fixed-length
    # regardless of how the underlying tensor was shaped.
    if prefill.shape[0] < PREFILL_ROW_LIMIT:
        pad = np.zeros((PREFILL_ROW_LIMIT - prefill.shape[0], ROW_SCHEMA_DIM), dtype=np.float32)
        prefill = np.concatenate([prefill, pad], axis=0) if prefill.size else pad
        prefill_mask = np.concatenate(
            [prefill_mask, np.zeros((PREFILL_ROW_LIMIT - prefill_mask.shape[0],), dtype=np.float32)],
            axis=0,
        )
    if decode.shape[0] < DECODE_ROW_LIMIT:
        pad = np.zeros((DECODE_ROW_LIMIT - decode.shape[0], ROW_SCHEMA_DIM), dtype=np.float32)
        decode = np.concatenate([decode, pad], axis=0) if decode.size else pad
        decode_mask = np.concatenate(
            [decode_mask, np.zeros((DECODE_ROW_LIMIT - decode_mask.shape[0],), dtype=np.float32)],
            axis=0,
        )

    # ---------- Aggregate per-row stats ----------
    def _aggregate(rows: np.ndarray, mask: np.ndarray, prefix: str) -> None:
        active = mask >= 0.5
        n_active = int(np.sum(active))
        names.append(f"{prefix}_active_count")
        values.append(float(n_active))
        # For each row dim, compute count/sum/mean/min/max/p50/p90 over masked
        # rows. These match the bellman_shaped path's masked aggregates and
        # let trees cover both "many requests in trouble" and "single bad
        # request" regimes.
        if n_active > 0:
            view = rows[active]
        else:
            view = np.zeros((0, rows.shape[1]), dtype=np.float32)
        for d in range(rows.shape[1]):
            col = view[:, d] if view.shape[0] else np.zeros((0,), dtype=np.float32)
            if col.size:
                stats = (
                    float(np.sum(col)),
                    float(np.mean(col)),
                    float(np.min(col)),
                    float(np.max(col)),
                    float(np.percentile(col, 50)),
                    float(np.percentile(col, 90)),
                )
            else:
                stats = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for stat_name, stat_value in zip(("sum", "mean", "min", "max", "p50", "p90"), stats):
                names.append(f"{prefix}_d{d:02d}_{stat_name}")
                values.append(float(stat_value))

    _aggregate(prefill, prefill_mask, "prefill")
    _aggregate(decode, decode_mask, "decode")

    # ---------- Cliff bucket counts ----------
    # Slack-to-drop and lateness from the row schema (idx 3 and 2).
    def _bucket_counts(rows: np.ndarray, mask: np.ndarray, prefix: str) -> None:
        active = mask >= 0.5
        slack = rows[active, 3] if active.any() else np.zeros((0,), dtype=np.float32)
        lateness = rows[active, 2] if active.any() else np.zeros((0,), dtype=np.float32)
        violated = rows[active, 4] if active.any() else np.zeros((0,), dtype=np.float32)
        for thresh in SLACK_BUCKETS:
            count = int(np.sum(slack <= float(thresh))) if slack.size else 0
            names.append(f"{prefix}_slack_le_{int(thresh*1000):04d}")
            values.append(float(count))
        for thresh in LATENESS_BUCKETS:
            count = int(np.sum(lateness > float(thresh))) if lateness.size else 0
            names.append(f"{prefix}_late_gt_{int(thresh*1000):04d}")
            values.append(float(count))
        names.append(f"{prefix}_violated_count")
        values.append(float(int(np.sum(violated >= 0.5)) if violated.size else 0))
        names.append(f"{prefix}_max_lateness")
        values.append(float(np.max(lateness)) if lateness.size else 0.0)
        names.append(f"{prefix}_min_slack")
        values.append(float(np.min(slack)) if slack.size else 0.0)
        names.append(f"{prefix}_lateness_sum")
        values.append(float(np.sum(lateness)) if lateness.size else 0.0)

    _bucket_counts(prefill, prefill_mask, "prefill")
    _bucket_counts(decode, decode_mask, "decode")

    # ---------- Top-K dangerous slots ----------
    rows: list[tuple[list[float], float]] = []
    for row, is_prefill in _row_iter(prefill, prefill_mask, decode, decode_mask):
        feats, score = _slot_features(row, is_prefill=is_prefill)
        rows.append((feats, float(score)))
    rows.sort(key=lambda item: float(item[1]))

    for slot in range(TOP_K_DANGER_SLOTS):
        if slot < len(rows):
            slot_feats, _ = rows[slot]
            present = 1.0
        else:
            slot_feats = [0.0] * DANGER_PER_SLOT_DIM
            present = 0.0
        names.append(f"slot_{slot:02d}_present")
        values.append(present)
        for j, val in enumerate(slot_feats):
            names.append(f"slot_{slot:02d}_d{j:02d}")
            values.append(float(val))

    return values, names


# ---------------------------------------------------------------------------
# Build feature matrix from records (uses state_loader to get ModelInputs)
# ---------------------------------------------------------------------------


def build_cliff_feature_matrix_from_records(
    records: list[dict[str, Any]],
    *,
    state_loader: Any,
    player: str = "controller",
) -> tuple[np.ndarray, list[str]]:
    if state_loader is None:
        raise ValueError("state_loader is required for cliff_aware feature extraction")
    from ..DNN import infer as dnn_infer

    use_act = bool(_CLIFF_USE_ACTION_FEATURES)
    top_k_act = int(_CLIFF_ACTION_TOP_K)
    if use_act:
        from .action_features import (
            _action_feature_names as _act_names_fn,
            _zero_action_feature_vector as _act_zero_fn,
            extract_action_features_from_state as _extract_act,
        )
        from .forecast_features import (
            extract_forecast_features_from_state as _extract_forecast,
            FORECAST_FEATURE_NAMES as _FORECAST_NAMES,
        )
        action_names = _act_names_fn(top_k_act)
        zero_action = _act_zero_fn(top_k_act)

    rows: list[list[float]] = []
    feature_names: list[str] | None = None
    mcts = getattr(state_loader, "mcts", None)
    env = getattr(state_loader, "env", None)
    for index, record in enumerate(records, start=1):
        state = state_loader(record)
        inputs = dnn_infer.build_model_inputs(
            state,
            str(record.get("root_player", player)),
            device="cpu",
            build_action_mask_flag=False,
        )
        row, names = extract_cliff_features_from_inputs(inputs)
        if use_act:
            try:
                # Parents are already controller-to-act, so no noop adversary needed.
                a_feats, a_names = _extract_act(
                    state, mcts=mcts, env=env, top_k=top_k_act,
                )
            except Exception:
                a_feats, a_names = list(zero_action), list(action_names)
            try:
                f_feats, f_names = _extract_forecast(state, env=env)
            except Exception:
                f_feats = [0.0] * len(_FORECAST_NAMES)
                f_names = list(_FORECAST_NAMES)
            row = list(row) + [float(x) for x in a_feats] + [float(x) for x in f_feats]
            names = list(names) + list(a_names) + list(f_names)
        if feature_names is None:
            feature_names = list(names)
        elif len(row) != len(feature_names):
            raise RuntimeError(
                f"feature row length {len(row)} != schema length {len(feature_names)}"
            )
        rows.append(row)
        if index % 4096 == 0:
            if mcts is not None:
                clear_fn = getattr(mcts, "clear_search_state", None)
                if callable(clear_fn):
                    clear_fn(drop_scratch=True)
            gc.collect()
    if not rows:
        raise ValueError("cannot build features for empty records")
    return np.asarray(rows, dtype=np.float32), list(feature_names or [])


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _records_feature_fingerprint(records: list[dict[str, Any]], *, player: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(CLIFF_FEATURE_CACHE_VERSION.encode("utf-8"))
    h.update(str(player).encode("utf-8"))
    h.update(str(len(records)).encode("utf-8"))
    for record in records:
        for key in ("root_id", "root_depth", "root_player", "root_node_id_override"):
            h.update(str(record.get(key, "")).encode("utf-8", errors="replace"))
            h.update(b"\0")
    return h.hexdigest()


def _cache_paths(cache_dir: Path, split_name: str, fingerprint: str) -> tuple[Path, Path, Path]:
    stem = f"{CLIFF_FEATURE_CACHE_VERSION}_{split_name}_{fingerprint}"
    return (
        cache_dir / f"{stem}.npy",
        cache_dir / f"{stem}.names.json",
        cache_dir / f"{stem}.meta.json",
    )


def _load_cache(
    cache_dir: Path, split_name: str, fingerprint: str, expected_count: int
) -> tuple[np.ndarray, list[str]] | None:
    matrix_path, names_path, _meta_path = _cache_paths(cache_dir, split_name, fingerprint)
    if not matrix_path.exists() or not names_path.exists():
        return None
    matrix = np.load(matrix_path, allow_pickle=False)
    if int(matrix.shape[0]) != int(expected_count):
        return None
    names = json.loads(names_path.read_text(encoding="utf-8"))
    return np.asarray(matrix, dtype=np.float32), [str(x) for x in names]


def _write_cache(
    cache_dir: Path,
    split_name: str,
    fingerprint: str,
    matrix: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, names_path, meta_path = _cache_paths(cache_dir, split_name, fingerprint)
    tmp_matrix = matrix_path.with_name(matrix_path.name + ".tmp")
    tmp_names = names_path.with_name(names_path.name + ".tmp")
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_matrix.open("wb") as f:
        np.save(f, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
    tmp_names.write_text(json.dumps(list(feature_names)), encoding="utf-8")
    tmp_meta.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True), encoding="utf-8")
    tmp_matrix.replace(matrix_path)
    tmp_names.replace(names_path)
    tmp_meta.replace(meta_path)


# ---------------------------------------------------------------------------
# Multi-process feature extraction
# ---------------------------------------------------------------------------


def _init_feature_worker(model_cfg: Any, extra_config: dict[str, Any], player: str) -> None:
    global _FEATURE_WORKER_STATE_LOADER, _FEATURE_WORKER_PLAYER
    global _CLIFF_USE_ACTION_FEATURES, _CLIFF_ACTION_TOP_K
    from .self_model_test import RootStateLoader

    _limit_native_threads(int((extra_config or {}).get("cliff_worker_threads", 1)))
    _FEATURE_WORKER_STATE_LOADER = RootStateLoader(model_cfg)
    _FEATURE_WORKER_PLAYER = str(player)
    cfg = extra_config or {}
    _CLIFF_USE_ACTION_FEATURES = bool(cfg.get("use_action_features", False))
    _CLIFF_ACTION_TOP_K = int(cfg.get("cliff_action_top_k", 4))
    if _CLIFF_USE_ACTION_FEATURES:
        # Bump cache version inside the worker so written files match the parent.
        global CLIFF_FEATURE_CACHE_VERSION
        CLIFF_FEATURE_CACHE_VERSION = "cliff_aware_modelinputs_v3_actaug"


def _feature_worker(item: tuple[int, list[dict[str, Any]]]) -> tuple[int, np.ndarray, list[str]]:
    chunk_index, records = item
    if _FEATURE_WORKER_STATE_LOADER is None:
        raise RuntimeError("cliff feature worker state loader not initialised")
    matrix, names = build_cliff_feature_matrix_from_records(
        records,
        state_loader=_FEATURE_WORKER_STATE_LOADER,
        player=str(_FEATURE_WORKER_PLAYER),
    )
    mcts = getattr(_FEATURE_WORKER_STATE_LOADER, "mcts", None)
    clear_fn = getattr(mcts, "clear_search_state", None)
    if callable(clear_fn):
        clear_fn(drop_scratch=True)
    gc.collect()
    return int(chunk_index), matrix, names


def _build_cliff_matrix_parallel(
    records: list[dict[str, Any]],
    *,
    model_cfg: Any,
    extra_config: dict[str, Any],
    player: str,
    num_processes: int,
    chunk_records: int,
    start_method: str,
    maxtasks_per_child: int | None,
) -> tuple[np.ndarray, list[str]]:
    if not records:
        raise ValueError("cannot build features for empty records")
    chunk_size = max(1, int(chunk_records))
    chunks: list[tuple[int, list[dict[str, Any]]]] = []
    for chunk_index, start in enumerate(range(0, len(records), chunk_size)):
        chunks.append((int(chunk_index), list(records[start : start + chunk_size])))
    process_count = min(max(1, int(num_processes)), len(chunks))
    if process_count <= 1:
        from .self_model_test import RootStateLoader

        loader = RootStateLoader(model_cfg)
        try:
            return build_cliff_feature_matrix_from_records(records, state_loader=loader, player=str(player))
        finally:
            try:
                loader.close()
            except Exception:
                pass
    method = str(start_method or "spawn")
    if method not in mp.get_all_start_methods():
        raise RuntimeError(f"multiprocessing start method {method!r} unavailable")
    print(
        f"[cliff_features] building {len(records)} rows with {process_count} processes, "
        f"{len(chunks)} chunks, chunk_records={chunk_size}, start_method={method}",
        flush=True,
    )
    ctx = mp.get_context(method)
    results: dict[int, tuple[np.ndarray, list[str]]] = {}
    completed = 0
    with ctx.Pool(
        processes=int(process_count),
        initializer=_init_feature_worker,
        initargs=(model_cfg, dict(extra_config or {}), str(player)),
        maxtasksperchild=int(maxtasks_per_child) if maxtasks_per_child else None,
    ) as pool:
        for ci, mat, names in pool.imap_unordered(_feature_worker, chunks):
            results[int(ci)] = (mat, names)
            completed += 1
            if completed == 1 or completed % 16 == 0 or completed == len(chunks):
                print(
                    f"[cliff_features] completed_chunks={completed}/{len(chunks)}",
                    flush=True,
                )
    ordered = [results[i] for i in range(len(chunks))]
    feature_names = list(ordered[0][1])
    matrices = [item[0] for item in ordered]
    return np.concatenate(matrices, axis=0).astype(np.float32, copy=False), feature_names


def build_or_load_cliff_feature_matrix(
    records: list[dict[str, Any]],
    *,
    state_loader: Any,
    model_cfg: Any,
    extra_config: dict[str, Any],
    output_dir: Path,
    split_name: str,
    player: str = "controller",
) -> tuple[np.ndarray, list[str]]:
    cache_enabled = bool(extra_config.get("cliff_feature_cache_enabled", True))
    cache_dir = Path(
        extra_config.get(
            "cliff_feature_cache_dir",
            str(output_dir.parent / "_cliff_feature_cache"),
        )
    )
    fingerprint = _records_feature_fingerprint(records, player=str(player))
    if cache_enabled:
        cached = _load_cache(cache_dir, str(split_name), fingerprint, len(records))
        if cached is not None:
            matrix, names = cached
            print(
                f"[cliff_features] cache hit split={split_name} rows={matrix.shape[0]} "
                f"features={matrix.shape[1]} path={cache_dir}",
                flush=True,
            )
            return matrix, names

    num_processes = int(extra_config.get("cliff_feature_num_processes", 1))
    chunk_records = int(extra_config.get("cliff_feature_chunk_records", 1024))
    start_method = str(extra_config.get("cliff_feature_mp_start_method", "spawn"))
    maxtasks_raw = extra_config.get("cliff_feature_maxtasks_per_child", None)

    t0 = time.time()
    if num_processes > 1 and len(records) > max(1, int(chunk_records)):
        matrix, names = _build_cliff_matrix_parallel(
            records,
            model_cfg=model_cfg,
            extra_config=extra_config,
            player=str(player),
            num_processes=int(num_processes),
            chunk_records=int(chunk_records),
            start_method=str(start_method),
            maxtasks_per_child=maxtasks_raw,
        )
    else:
        matrix, names = build_cliff_feature_matrix_from_records(
            records, state_loader=state_loader, player=str(player)
        )
    seconds = time.time() - t0
    print(
        f"[cliff_features] built split={split_name} rows={matrix.shape[0]} "
        f"features={matrix.shape[1]} seconds={seconds:.2f}",
        flush=True,
    )

    if cache_enabled:
        _write_cache(
            cache_dir,
            str(split_name),
            fingerprint,
            matrix,
            names,
            {
                "cache_version": CLIFF_FEATURE_CACHE_VERSION,
                "split_name": str(split_name),
                "rows": int(matrix.shape[0]),
                "features": int(matrix.shape[1]),
                "seconds": float(seconds),
                "feature_num_processes": int(num_processes),
                "feature_chunk_records": int(chunk_records),
            },
        )
        print(
            f"[cliff_features] cache written split={split_name} path={cache_dir}",
            flush=True,
        )
    return matrix, names


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliffAwareControllerValueModel:
    """Cliff-aware classical controller value surrogate."""

    estimators: dict[str, Any]
    feature_names: list[str]
    feature_config: dict[str, Any]
    trainable_params: int
    blend_weight_hgb: float = 0.5
    zero_threshold: float = 0.70
    cliff_residual_scale: float = 1.0
    tail_residual_scale: float = 1.0
    tail_residual_pred_threshold: float = -0.5
    polyak_alpha: float = 1.0
    bootstrap_clip_min: float = -50.0
    bootstrap_clip_max: float = 0.0
    model_name: str = "cliff_aware_controller_value"
    uses_neural_network: bool = False
    uses_target_leakage: bool = False
    cached_predictions: dict[str, list[float]] = field(
        default_factory=dict, repr=False, compare=False
    )
    runtime_prediction_cache: dict[bytes, float] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ----- prediction core -----

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        extra = np.asarray(self.estimators["extra"].predict(x), dtype=np.float64)
        hgb = np.asarray(self.estimators["hgb"].predict(x), dtype=np.float64)
        w = float(self.blend_weight_hgb)
        base = (1.0 - w) * extra + w * hgb
        base = np.minimum(base, 0.0)

        zero_gate = self.estimators.get("zero_gate")
        if zero_gate is not None:
            proba = np.asarray(zero_gate.predict_proba(x), dtype=np.float64)
            classes = list(getattr(zero_gate, "classes_", []))
            zero_idx = None
            for candidate in (1, True):
                if candidate in classes:
                    zero_idx = classes.index(candidate)
                    break
            if zero_idx is not None:
                zero_prob = proba[:, int(zero_idx)]
                base = base.copy()
                base[zero_prob >= float(self.zero_threshold)] = 0.0

        cliff_residual = self.estimators.get("cliff_residual")
        if cliff_residual is not None and float(self.cliff_residual_scale) != 0.0:
            base = np.minimum(
                base
                + float(self.cliff_residual_scale)
                * np.asarray(cliff_residual.predict(x), dtype=np.float64),
                0.0,
            )

        tail = self.estimators.get("tail_residual")
        if tail is not None and float(self.tail_residual_scale) != 0.0:
            tail_mask = base <= float(self.tail_residual_pred_threshold)
            if np.any(tail_mask):
                tail_pred = np.asarray(tail.predict(x[tail_mask]), dtype=np.float64)
                base[tail_mask] = np.minimum(
                    base[tail_mask] + float(self.tail_residual_scale) * tail_pred,
                    0.0,
                )

        return base

    # ----- MCTS / bellman_convergence bootstrap entry -----

    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: Any | None = None,
    ) -> tuple[float, list[float]]:
        del player, device
        row, _names = extract_cliff_features_from_inputs(inputs)
        target_len = len(self.feature_names)
        if len(row) != target_len:
            # The two extractors should agree, but stay safe.
            if len(row) < target_len:
                row = row + [0.0] * (target_len - len(row))
            else:
                row = row[:target_len]
        x = np.asarray([row], dtype=np.float32)
        key = x.tobytes()
        cached = self.runtime_prediction_cache.get(key)
        if cached is not None:
            return float(cached), []
        extra = self.estimators.get("extra")
        if extra is not None and hasattr(extra, "n_jobs"):
            try:
                extra.n_jobs = 1
            except Exception:
                pass
        value = float(self.predict_matrix(x)[0])
        # Clip bootstrap predictions to a stable range to prevent occasional
        # large negative outliers from amplifying through Bellman iterations.
        # Approximate VI with function approximation needs a contraction in
        # max-norm; clipping at the source keeps the iteration well-behaved.
        clip_min = float(self.bootstrap_clip_min)
        clip_max = float(self.bootstrap_clip_max)
        if value < clip_min:
            value = clip_min
        elif value > clip_max:
            value = clip_max
        if len(self.runtime_prediction_cache) > 50_000:
            self.runtime_prediction_cache.clear()
        self.runtime_prediction_cache[key] = value
        return value, []


# ---------------------------------------------------------------------------
# Param accounting
# ---------------------------------------------------------------------------


def _tree_node_budget(estimator: Any) -> int:
    estimators = getattr(estimator, "estimators_", None)
    if estimators is not None:
        total = 0
        for item in np.ravel(estimators):
            tree = getattr(item, "tree_", None)
            if tree is not None:
                total += int(tree.node_count)
        if total > 0:
            return int(total)
    max_iter = int(getattr(estimator, "max_iter", 0) or 0)
    max_leaf_nodes = int(getattr(estimator, "max_leaf_nodes", 0) or 0)
    if max_iter > 0 and max_leaf_nodes > 0:
        return int(max_iter * max(1, 2 * max_leaf_nodes - 1))
    return 0


def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    abs_err = np.abs(err)
    mse = float(np.mean(err * err))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(abs_err)),
        "p50_abs_error": float(np.percentile(abs_err, 50)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "max_abs_error": float(np.max(abs_err)),
    }


def _error_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true64 = np.asarray(y_true, dtype=np.float64)
    y_pred64 = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_pred64 - y_true64)
    abs_t = np.abs(y_true64)
    out: dict[str, Any] = {
        "num_abs_error_gt_0p05": int(np.sum(abs_err > 0.05)),
        "num_abs_error_gt_0p10": int(np.sum(abs_err > 0.10)),
        "num_abs_error_gt_0p50": int(np.sum(abs_err > 0.50)),
        "num_abs_error_gt_1p00": int(np.sum(abs_err > 1.00)),
    }
    buckets = {
        "target_abs_eq_0": abs_t <= 1e-12,
        "target_abs_le_0p05": abs_t <= 0.05,
        "target_abs_0p05_1": (abs_t > 0.05) & (abs_t < 1.0),
        "target_abs_1_2": (abs_t >= 1.0) & (abs_t < 2.0),
        "target_abs_2_3": (abs_t >= 2.0) & (abs_t < 3.0),
        "target_abs_ge_3": abs_t >= 3.0,
    }
    for name, mask in buckets.items():
        count = int(np.sum(mask))
        prefix = f"bucket_{name}"
        out[f"{prefix}_count"] = count
        if count <= 0:
            continue
        bucket_errors = abs_err[mask]
        out[f"{prefix}_mae"] = float(np.mean(bucket_errors))
        out[f"{prefix}_p95_abs_error"] = float(np.percentile(bucket_errors, 95))
        out[f"{prefix}_max_abs_error"] = float(np.max(bucket_errors))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _target_array(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(r["target_value"]) for r in records], dtype=np.float32)


def train_cliff_aware_model(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    cfg: Any,
    output_dir: str | Path,
    state_loader: Any | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extra_cfg: dict[str, Any] = dict(getattr(cfg, "extra_config", {}) or {})
    max_params = int(extra_cfg.get("max_trainable_params", DEFAULT_MODEL_BUDGET))
    seed = int(extra_cfg.get("random_state", 2027))

    t0 = time.time()
    x_train, feature_names = build_or_load_cliff_feature_matrix(
        list(train_records),
        state_loader=state_loader,
        model_cfg=cfg,
        extra_config=extra_cfg,
        output_dir=output,
        split_name="train",
        player="controller",
    )
    x_eval, _ = build_or_load_cliff_feature_matrix(
        list(eval_records),
        state_loader=state_loader,
        model_cfg=cfg,
        extra_config=extra_cfg,
        output_dir=output,
        split_name="eval",
        player="controller",
    )
    feature_seconds = time.time() - t0
    y_train = _target_array(list(train_records))
    y_eval = _target_array(list(eval_records))

    estimators: dict[str, Any] = {}
    # Tree node budget allocation under 150k scalars (ARM box: 96 cores). The
    # earlier 74k-node configuration left the V1 max_abs_error at ~2.5 which
    # then amplified through Bellman iterations. Capacity has been pushed up
    # roughly 2x in the residual stages so the tail predictor has enough
    # headroom to absorb cliff outliers without exceeding 150k scalars.
    estimators["extra"] = ExtraTreesRegressor(
        n_estimators=int(extra_cfg.get("extra_n_estimators", 24)),
        max_leaf_nodes=int(extra_cfg.get("extra_max_leaf_nodes", 128)),
        min_samples_leaf=int(extra_cfg.get("extra_min_samples_leaf", 16)),
        max_features=float(extra_cfg.get("extra_max_features", 0.85)),
        bootstrap=False,
        random_state=seed,
        n_jobs=int(extra_cfg.get("n_jobs", -1)),
    )
    estimators["hgb"] = HistGradientBoostingRegressor(
        loss=str(extra_cfg.get("hgb_loss", "absolute_error")),
        max_iter=int(extra_cfg.get("hgb_max_iter", 400)),
        max_leaf_nodes=int(extra_cfg.get("hgb_max_leaf_nodes", 21)),
        learning_rate=float(extra_cfg.get("hgb_learning_rate", 0.05)),
        l2_regularization=float(extra_cfg.get("hgb_l2_regularization", 0.0)),
        max_bins=int(extra_cfg.get("hgb_max_bins", 255)),
        early_stopping=False,
        random_state=seed + 17,
    )
    estimators["zero_gate"] = HistGradientBoostingClassifier(
        max_iter=int(extra_cfg.get("zero_gate_max_iter", 200)),
        max_leaf_nodes=int(extra_cfg.get("zero_gate_max_leaf_nodes", 15)),
        learning_rate=float(extra_cfg.get("zero_gate_learning_rate", 0.05)),
        l2_regularization=float(extra_cfg.get("zero_gate_l2_regularization", 0.0)),
        max_bins=int(extra_cfg.get("zero_gate_max_bins", 255)),
        early_stopping=False,
        random_state=seed + 29,
    )
    estimators["cliff_residual"] = HistGradientBoostingRegressor(
        loss=str(extra_cfg.get("cliff_residual_loss", "absolute_error")),
        max_iter=int(extra_cfg.get("cliff_residual_max_iter", 800)),
        max_leaf_nodes=int(extra_cfg.get("cliff_residual_max_leaf_nodes", 31)),
        learning_rate=float(extra_cfg.get("cliff_residual_learning_rate", 0.03)),
        l2_regularization=float(extra_cfg.get("cliff_residual_l2_regularization", 0.0)),
        max_bins=int(extra_cfg.get("cliff_residual_max_bins", 255)),
        early_stopping=False,
        random_state=seed + 41,
    )
    estimators["tail_residual"] = HistGradientBoostingRegressor(
        loss=str(extra_cfg.get("tail_residual_loss", "absolute_error")),
        max_iter=int(extra_cfg.get("tail_residual_max_iter", 500)),
        max_leaf_nodes=int(extra_cfg.get("tail_residual_max_leaf_nodes", 47)),
        learning_rate=float(extra_cfg.get("tail_residual_learning_rate", 0.04)),
        l2_regularization=float(extra_cfg.get("tail_residual_l2_regularization", 0.0)),
        max_bins=int(extra_cfg.get("tail_residual_max_bins", 255)),
        early_stopping=False,
        random_state=seed + 53,
    )

    fit_start = time.time()

    estimators["extra"].fit(x_train, y_train)
    gc.collect()
    estimators["hgb"].fit(x_train, y_train)
    gc.collect()

    blend_w = float(extra_cfg.get("blend_weight_hgb", 0.5))
    base_extra_train = np.asarray(estimators["extra"].predict(x_train), dtype=np.float64)
    base_hgb_train = np.asarray(estimators["hgb"].predict(x_train), dtype=np.float64)
    base_train = np.minimum((1.0 - blend_w) * base_extra_train + blend_w * base_hgb_train, 0.0)

    zero_eps = float(extra_cfg.get("zero_label_eps", 0.05))
    zero_labels = np.asarray(np.abs(y_train) <= zero_eps, dtype=np.int8)
    estimators["zero_gate"].fit(x_train, zero_labels)
    gc.collect()

    zero_threshold = float(extra_cfg.get("zero_threshold", 0.70))
    proba_train = np.asarray(estimators["zero_gate"].predict_proba(x_train), dtype=np.float64)
    classes = list(getattr(estimators["zero_gate"], "classes_", []))
    zero_idx = None
    for candidate in (1, True):
        if candidate in classes:
            zero_idx = classes.index(candidate)
            break
    if zero_idx is not None:
        zero_prob_train = proba_train[:, int(zero_idx)]
        base_train = base_train.copy()
        base_train[zero_prob_train >= zero_threshold] = 0.0

    cliff_residual_target = np.asarray(y_train, dtype=np.float64) - base_train
    cliff_weights = (
        1.0
        + float(extra_cfg.get("cliff_residual_error_weight", 4.0))
        * np.minimum(
            1.0,
            np.abs(cliff_residual_target)
            / max(1e-9, float(extra_cfg.get("cliff_residual_error_scale", 0.30))),
        )
        + float(extra_cfg.get("cliff_residual_large_weight", 2.0))
        * (np.abs(y_train) >= float(extra_cfg.get("cliff_residual_large_target_abs", 1.0)))
    )
    estimators["cliff_residual"].fit(
        x_train,
        cliff_residual_target.astype(np.float32),
        sample_weight=cliff_weights.astype(np.float32),
    )
    gc.collect()

    tail_target_abs = float(extra_cfg.get("tail_residual_target_abs", 0.5))
    tail_mask_train = np.abs(y_train) >= tail_target_abs
    if int(np.sum(tail_mask_train)) > 0:
        base_after_cliff_train = np.minimum(
            base_train
            + np.asarray(estimators["cliff_residual"].predict(x_train), dtype=np.float64),
            0.0,
        )
        tail_residual_target = (np.asarray(y_train, dtype=np.float64) - base_after_cliff_train)[tail_mask_train]
        x_tail_train = x_train[tail_mask_train]
        tail_weights = 1.0 + float(extra_cfg.get("tail_residual_error_weight", 6.0)) * np.minimum(
            1.0,
            np.abs(tail_residual_target)
            / max(1e-9, float(extra_cfg.get("tail_residual_error_scale", 0.30))),
        )
        estimators["tail_residual"].fit(
            x_tail_train,
            tail_residual_target.astype(np.float32),
            sample_weight=tail_weights.astype(np.float32),
        )
    else:
        estimators.pop("tail_residual", None)
    gc.collect()

    fit_seconds = time.time() - fit_start

    trainable_params = int(sum(_tree_node_budget(est) for est in estimators.values()))
    if trainable_params > max_params:
        raise ValueError(
            f"cliff_aware model has estimated {trainable_params} scalar params, "
            f"above budget {max_params}"
        )

    model = CliffAwareControllerValueModel(
        estimators=estimators,
        feature_names=feature_names,
        feature_config={
            "feature_source": "model_inputs",
            "cache_version": CLIFF_FEATURE_CACHE_VERSION,
        },
        trainable_params=int(trainable_params),
        blend_weight_hgb=float(blend_w),
        zero_threshold=float(zero_threshold),
        cliff_residual_scale=float(extra_cfg.get("cliff_residual_scale", 1.0)),
        tail_residual_scale=float(extra_cfg.get("tail_residual_scale", 1.0)),
        tail_residual_pred_threshold=float(extra_cfg.get("tail_residual_pred_threshold", -0.5)),
        polyak_alpha=float(extra_cfg.get("polyak_alpha", 1.0)),
        bootstrap_clip_min=float(extra_cfg.get("bootstrap_clip_min", -5.0)),
        bootstrap_clip_max=float(extra_cfg.get("bootstrap_clip_max", 0.0)),
        model_name=str(getattr(cfg, "model_name", "cliff_aware_controller_value")),
    )

    train_pred = model.predict_matrix(x_train)
    eval_pred = model.predict_matrix(x_eval)
    train_metrics = _error_metrics(y_train, train_pred)
    eval_metrics = _error_metrics(y_eval, eval_pred)
    train_diag = _error_diagnostics(y_train, train_pred)
    eval_diag = _error_diagnostics(y_eval, eval_pred)

    extra_est = estimators.get("extra")
    if extra_est is not None and hasattr(extra_est, "n_jobs"):
        try:
            extra_est.n_jobs = 1
        except Exception:
            pass

    model = CliffAwareControllerValueModel(
        estimators=model.estimators,
        feature_names=model.feature_names,
        feature_config=model.feature_config,
        trainable_params=model.trainable_params,
        blend_weight_hgb=model.blend_weight_hgb,
        zero_threshold=model.zero_threshold,
        cliff_residual_scale=model.cliff_residual_scale,
        tail_residual_scale=model.tail_residual_scale,
        tail_residual_pred_threshold=model.tail_residual_pred_threshold,
        polyak_alpha=model.polyak_alpha,
        bootstrap_clip_min=model.bootstrap_clip_min,
        bootstrap_clip_max=model.bootstrap_clip_max,
        model_name=model.model_name,
        cached_predictions={
            "train": [float(x) for x in train_pred.tolist()],
            "eval": [float(x) for x in eval_pred.tolist()],
        },
    )

    model_path = output / "cliff_aware_controller_value.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model.model_name,
        "backend": "cliff_aware_modelinputs",
        "trainable_params_estimate": int(model.trainable_params),
        "feature_count": int(len(feature_names)),
        "uses_neural_network": False,
        "uses_target_leakage": False,
        "num_train_records": int(len(train_records)),
        "num_eval_records": int(len(eval_records)),
        "feature_seconds": float(feature_seconds),
        "fit_seconds": float(fit_seconds),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "train_diagnostics": train_diag,
        "eval_diagnostics": eval_diag,
        "feature_config": dict(model.feature_config),
        "description": (
            "Cliff-aware non-neural controller value surrogate built on the GV3 "
            "ModelInputs schema. HistGradientBoosting (absolute error) + zero "
            "gate + tail residual booster, with top-K dangerous-request slots."
        ),
    }
    metadata_path = output / "cliff_aware_controller_value.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "model": model,
        "best_checkpoint_path": model_path,
        "train_metrics": [
            {
                "backend": "cliff_aware_modelinputs",
                "epoch": 0,
                "model_name": str(model.model_name),
                "trainable_params": int(model.trainable_params),
                "num_train_records": int(len(train_records)),
                "num_eval_records": int(len(eval_records)),
                "feature_seconds": float(feature_seconds),
                "fit_seconds": float(fit_seconds),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"eval_{k}": float(v) for k, v in eval_metrics.items()},
                **{f"train_diag_{k}": float(v) for k, v in train_diag.items() if isinstance(v, (int, float))},
                **{f"eval_diag_{k}": float(v) for k, v in eval_diag.items() if isinstance(v, (int, float))},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Inference helpers wired into DNN/infer.py
# ---------------------------------------------------------------------------


def predict_cliff_aware_values(
    *,
    model: CliffAwareControllerValueModel,
    records: list[dict[str, Any]],
    state_loader: Any | None = None,
    split_name: str | None = None,
) -> list[float]:
    if not isinstance(model, CliffAwareControllerValueModel):
        raise TypeError(f"expected CliffAwareControllerValueModel, got {type(model)!r}")
    if split_name is not None:
        cached = model.cached_predictions.get(str(split_name))
        if cached is not None and len(cached) == len(records):
            return [float(x) for x in cached]
        if str(split_name).startswith("train"):
            cached = model.cached_predictions.get("train")
            if cached is not None and len(cached) == len(records):
                return [float(x) for x in cached]
        if str(split_name).startswith("eval"):
            cached = model.cached_predictions.get("eval")
            if cached is not None and len(cached) == len(records):
                return [float(x) for x in cached]
    if state_loader is None:
        raise ValueError("state_loader is required when predicting from records without cache")
    matrix, _names = build_cliff_feature_matrix_from_records(
        list(records), state_loader=state_loader, player="controller"
    )
    return [float(v) for v in model.predict_matrix(matrix).tolist()]


def predict_cliff_aware_value_from_inputs(
    *,
    model: CliffAwareControllerValueModel,
    inputs: Any,
    player: str,
    device: Any | None = None,
) -> float:
    value, _priors = model.infer_from_inputs(inputs, player, device=device)
    return float(value)


def predict_cliff_aware_values_from_inputs_batch(
    *,
    model: CliffAwareControllerValueModel,
    inputs_list: Sequence[Any],
) -> list[float]:
    if not isinstance(model, CliffAwareControllerValueModel):
        raise TypeError(f"expected CliffAwareControllerValueModel, got {type(model)!r}")
    rows: list[list[float]] = []
    target_len = len(model.feature_names)
    for inputs in inputs_list:
        row, _names = extract_cliff_features_from_inputs(inputs)
        if len(row) != target_len:
            if len(row) < target_len:
                row = row + [0.0] * (target_len - len(row))
            else:
                row = row[:target_len]
        rows.append(row)
    if not rows:
        return []
    x = np.asarray(rows, dtype=np.float32)
    preds = np.asarray(model.predict_matrix(x), dtype=np.float64)
    clip_min = float(model.bootstrap_clip_min)
    clip_max = float(model.bootstrap_clip_max)
    preds = np.clip(preds, clip_min, clip_max)
    return [float(v) for v in preds.tolist()]
