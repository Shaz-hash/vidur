"""Action-conditional feature builder for the GV3 controller value model.

Why this module exists:
    The value-target distribution at GV3 controller roots is bimodal for
    visually-similar states. Two states with nearly identical aggregate
    request statistics can have target_value 0 (controller has a saving
    action) or target_value <= -3 (no saving action exists). The cliff_aware
    feature pipeline aggregates state-level signals only, so the resulting
    NN/tree predictions saturate at ~1.5 mean abs error on those rows.

    Action-conditional features fix that floor by handing the model the
    distribution of immediate rewards across all valid controller actions.
    For each record we run mctsDNN's canonical action enumeration and apply
    each action via apply_controller_action_only, recording leaf_cost,
    reward, and timing. The aggregated per-record vector includes
        min/max/mean reward, count of zero-reward actions, top-K best
        actions (reward, leaf_cost, fast-forward delta).

    These features are computable both at training time (record ->
    state_loader -> mctsDNN actions) and at MCTS bootstrap time (state ->
    mctsDNN actions). They keep the train and bootstrap distributions
    identical, like the cliff_aware path.

Cost:
    Building one action-feature row requires N_canonical state forks +
    apply_controller_action_only calls. Roughly 5-7x the per-record cost of
    cliff_aware features. We pay it once per dataset and cache the result
    on disk; subsequent Bellman iterations reuse the cache.
"""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ACTION_FEATURE_CACHE_VERSION = "action_features_v1"
DEFAULT_TOP_K_ACTIONS = 4
PER_TOP_ACTION_DIMS = 5  # (reward, leaf_cost, time_delta, ev_idx_norm, valid)


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
# Per-record extractor (uses mctsDNN canonical action expansion)
# ---------------------------------------------------------------------------


def _action_feature_names(top_k: int) -> list[str]:
    base = [
        "act_n_canonical",
        "act_n_valid",
        "act_min_reward",
        "act_max_reward",
        "act_mean_reward",
        "act_sum_reward",
        "act_std_reward",
        "act_min_leaf_cost",
        "act_max_leaf_cost",
        "act_mean_leaf_cost",
        "act_min_discount",
        "act_max_discount",
        "act_mean_discount",
        "act_min_time_delta",
        "act_max_time_delta",
        "act_mean_time_delta",
        # cliff distance counts across actions
        "act_n_reward_zero",
        "act_n_reward_lt_neg0p1",
        "act_n_reward_lt_neg1",
        "act_n_reward_lt_neg3",
        "act_n_leaf_cost_zero",
        "act_n_leaf_cost_lt_1",
        "act_n_leaf_cost_lt_3",
        # immediate-reward bin histogram (5 bins)
        "act_hist_reward_eq_0",
        "act_hist_reward_neg0_to_neg0p5",
        "act_hist_reward_neg0p5_to_neg1",
        "act_hist_reward_neg1_to_neg2",
        "act_hist_reward_neg2_or_worse",
    ]
    for i in range(int(top_k)):
        base.extend(
            [
                f"act_top{i:02d}_reward",
                f"act_top{i:02d}_leaf_cost",
                f"act_top{i:02d}_time_delta",
                f"act_top{i:02d}_discount",
                f"act_top{i:02d}_present",
            ]
        )
    return base


def _zero_action_feature_vector(top_k: int) -> list[float]:
    return [0.0] * len(_action_feature_names(top_k))


def extract_action_feature_vector(
    record: dict[str, Any],
    *,
    state_loader: Any,
    top_k: int = DEFAULT_TOP_K_ACTIONS,
) -> tuple[list[float], list[str]]:
    """Build a per-record action-conditional feature vector.

    Uses mctsDNN canonical action enumeration to walk every distinct
    controller action available at the state, apply each one, and record
    immediate reward + child cost + timing. Returns a fixed-length feature
    vector compatible with the column names in `_action_feature_names`.
    """

    names = _action_feature_names(int(top_k))

    if state_loader is None:
        return _zero_action_feature_vector(int(top_k)), names

    mcts = getattr(state_loader, "mcts", None)
    env = getattr(state_loader, "env", None)
    if mcts is None or env is None:
        return _zero_action_feature_vector(int(top_k)), names

    state = state_loader(record)
    root_player = str(record.get("root_player", "controller"))
    if root_player != "controller":
        return _zero_action_feature_vector(int(top_k)), names

    # mimic mctsDNN's depth1 search up to action enumeration.
    mcts.clear_search_state(drop_scratch=False)
    decision_state, _ = mcts._decision_state(None, state, "controller") if False else (state, None)
    # mctsDNN._decision_state requires a node; pass state directly when root.
    root_cost = float(mcts._state_cost(decision_state))
    root_time = float(decision_state.simulator._time)

    actions_by_index, mask_t = mcts._actions_and_mask(
        decision_state,
        "controller",
        forbidden_stop_ids=None,
    )
    valid_mask = [bool(x) for x in mask_t.tolist()]
    valid_indices = [
        i for i, ok in enumerate(valid_mask) if ok and actions_by_index[i] is not None
    ]
    if not valid_indices:
        return _zero_action_feature_vector(int(top_k)), names

    _alias_to_canon, _canon_to_aliases, canonical_indices = mcts._canonicalize_action_indices(
        player="controller",
        actions_by_index=actions_by_index,
        valid_indices=valid_indices,
    )

    decision_snapshot, decision_stats = mcts._snapshot_state_and_stats(decision_state)

    rewards: list[float] = []
    leaf_costs: list[float] = []
    leaf_times: list[float] = []
    discounts: list[float] = []
    time_deltas: list[float] = []

    for cidx in canonical_indices:
        action = actions_by_index[cidx]
        if action is None:
            continue
        leaf_state = mcts._scratch_restore(decision_snapshot, decision_stats)
        leaf_state = mcts._env.apply_controller_action_only(
            leaf_state,
            action,
            inplace=True,
            fast_forward=False,
        )
        leaf_cost = float(mcts._state_cost(leaf_state))
        reward = float(mcts._transition_reward(root_cost, leaf_cost))
        leaf_time = float(leaf_state.simulator._time)
        discount_time_attr = getattr(leaf_state.stats, "transition_discount_time", None)
        discount_time = float(discount_time_attr) if discount_time_attr is not None else leaf_time
        discount = float(mcts._time_discount(discount_time, root_time))
        rewards.append(reward)
        leaf_costs.append(leaf_cost)
        leaf_times.append(leaf_time)
        discounts.append(discount)
        time_deltas.append(max(0.0, leaf_time - root_time))

    if not rewards:
        return _zero_action_feature_vector(int(top_k)), names

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    leaf_costs_arr = np.asarray(leaf_costs, dtype=np.float64)
    discounts_arr = np.asarray(discounts, dtype=np.float64)
    time_deltas_arr = np.asarray(time_deltas, dtype=np.float64)

    values: list[float] = []

    def add(v: Any) -> None:
        try:
            values.append(float(v))
        except Exception:
            values.append(0.0)

    add(int(len(canonical_indices)))
    add(int(len(rewards_arr)))
    add(float(np.min(rewards_arr)))
    add(float(np.max(rewards_arr)))
    add(float(np.mean(rewards_arr)))
    add(float(np.sum(rewards_arr)))
    add(float(np.std(rewards_arr)) if rewards_arr.size > 1 else 0.0)
    add(float(np.min(leaf_costs_arr)))
    add(float(np.max(leaf_costs_arr)))
    add(float(np.mean(leaf_costs_arr)))
    add(float(np.min(discounts_arr)))
    add(float(np.max(discounts_arr)))
    add(float(np.mean(discounts_arr)))
    add(float(np.min(time_deltas_arr)))
    add(float(np.max(time_deltas_arr)))
    add(float(np.mean(time_deltas_arr)))
    add(int((np.abs(rewards_arr) <= 1e-9).sum()))
    add(int((rewards_arr < -0.1).sum()))
    add(int((rewards_arr < -1.0).sum()))
    add(int((rewards_arr < -3.0).sum()))
    add(int((np.abs(leaf_costs_arr) <= 1e-9).sum()))
    add(int((leaf_costs_arr < 1.0).sum()))
    add(int((leaf_costs_arr < 3.0).sum()))
    # histogram of immediate rewards
    add(int((np.abs(rewards_arr) <= 1e-9).sum()))
    add(int(((rewards_arr < 0.0) & (rewards_arr >= -0.5)).sum()))
    add(int(((rewards_arr < -0.5) & (rewards_arr >= -1.0)).sum()))
    add(int(((rewards_arr < -1.0) & (rewards_arr >= -2.0)).sum()))
    add(int((rewards_arr < -2.0).sum()))

    # top-K actions by reward (largest first; tie -> smallest leaf_cost)
    order = np.lexsort((leaf_costs_arr, -rewards_arr))
    for i in range(int(top_k)):
        if i < len(order):
            j = int(order[i])
            add(rewards_arr[j])
            add(leaf_costs_arr[j])
            add(time_deltas_arr[j])
            add(discounts_arr[j])
            add(1.0)
        else:
            add(0.0)
            add(0.0)
            add(0.0)
            add(0.0)
            add(0.0)

    if len(values) != len(names):
        raise RuntimeError(
            f"action feature vector length mismatch: {len(values)} != {len(names)}"
        )
    return values, names


# ---------------------------------------------------------------------------
# Build matrix from records (sequential)
# ---------------------------------------------------------------------------


def build_action_feature_matrix_from_records(
    records: list[dict[str, Any]],
    *,
    state_loader: Any,
    top_k: int = DEFAULT_TOP_K_ACTIONS,
) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    feature_names: list[str] | None = None
    for index, record in enumerate(records, start=1):
        row, names = extract_action_feature_vector(
            record, state_loader=state_loader, top_k=int(top_k)
        )
        if feature_names is None:
            feature_names = list(names)
        elif len(row) != len(feature_names):
            raise RuntimeError(
                f"action feature row length {len(row)} != schema length {len(feature_names)}"
            )
        rows.append(row)
        if index % 4096 == 0:
            mcts = getattr(state_loader, "mcts", None)
            clear_fn = getattr(mcts, "clear_search_state", None)
            if callable(clear_fn):
                clear_fn(drop_scratch=True)
            gc.collect()
    if not rows:
        raise ValueError("cannot build features for empty records")
    return np.asarray(rows, dtype=np.float32), list(feature_names or [])


# ---------------------------------------------------------------------------
# Multi-process build with cache
# ---------------------------------------------------------------------------


_ACTION_WORKER_LOADER: Any | None = None
_ACTION_WORKER_TOP_K = DEFAULT_TOP_K_ACTIONS


def _action_records_fingerprint(records: list[dict[str, Any]], *, top_k: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(ACTION_FEATURE_CACHE_VERSION.encode("utf-8"))
    h.update(f"top_k={top_k}".encode("utf-8"))
    h.update(str(len(records)).encode("utf-8"))
    for record in records:
        for key in ("root_id", "root_depth", "root_player", "root_node_id_override"):
            h.update(str(record.get(key, "")).encode("utf-8", errors="replace"))
            h.update(b"\0")
    return h.hexdigest()


def _action_cache_paths(
    cache_dir: Path, split_name: str, fingerprint: str
) -> tuple[Path, Path, Path]:
    stem = f"{ACTION_FEATURE_CACHE_VERSION}_{split_name}_{fingerprint}"
    return (
        cache_dir / f"{stem}.npy",
        cache_dir / f"{stem}.names.json",
        cache_dir / f"{stem}.meta.json",
    )


def _load_action_cache(
    cache_dir: Path, split_name: str, fingerprint: str, expected: int
) -> tuple[np.ndarray, list[str]] | None:
    matrix_path, names_path, _ = _action_cache_paths(cache_dir, split_name, fingerprint)
    if not matrix_path.exists() or not names_path.exists():
        return None
    matrix = np.load(matrix_path, allow_pickle=False)
    if int(matrix.shape[0]) != int(expected):
        return None
    names = json.loads(names_path.read_text(encoding="utf-8"))
    return np.asarray(matrix, dtype=np.float32), [str(x) for x in names]


def _write_action_cache(
    cache_dir: Path,
    split_name: str,
    fingerprint: str,
    matrix: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, names_path, meta_path = _action_cache_paths(
        cache_dir, split_name, fingerprint
    )
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


def _init_action_worker(
    model_cfg: Any, extra_config: dict[str, Any], top_k: int
) -> None:
    global _ACTION_WORKER_LOADER
    global _ACTION_WORKER_TOP_K
    from .self_model_test import RootStateLoader

    _limit_native_threads(int((extra_config or {}).get("action_worker_threads", 1)))
    _ACTION_WORKER_LOADER = RootStateLoader(model_cfg)
    _ACTION_WORKER_TOP_K = int(top_k)


def _action_worker(
    item: tuple[int, list[dict[str, Any]]],
) -> tuple[int, np.ndarray, list[str]]:
    chunk_index, records = item
    if _ACTION_WORKER_LOADER is None:
        raise RuntimeError("action feature worker state loader not initialised")
    matrix, names = build_action_feature_matrix_from_records(
        records,
        state_loader=_ACTION_WORKER_LOADER,
        top_k=int(_ACTION_WORKER_TOP_K),
    )
    mcts = getattr(_ACTION_WORKER_LOADER, "mcts", None)
    clear_fn = getattr(mcts, "clear_search_state", None)
    if callable(clear_fn):
        clear_fn(drop_scratch=True)
    gc.collect()
    return int(chunk_index), matrix, names


def _build_action_matrix_parallel(
    records: list[dict[str, Any]],
    *,
    model_cfg: Any,
    extra_config: dict[str, Any],
    top_k: int,
    num_processes: int,
    chunk_records: int,
    start_method: str,
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
            return build_action_feature_matrix_from_records(
                records, state_loader=loader, top_k=int(top_k)
            )
        finally:
            try:
                loader.close()
            except Exception:
                pass
    method = str(start_method or "spawn")
    if method not in mp.get_all_start_methods():
        raise RuntimeError(f"multiprocessing start method {method!r} unavailable")
    print(
        f"[action_features] building {len(records)} rows with {process_count} processes, "
        f"{len(chunks)} chunks, chunk_records={chunk_size}, start_method={method}",
        flush=True,
    )
    ctx = mp.get_context(method)
    results: dict[int, tuple[np.ndarray, list[str]]] = {}
    completed = 0
    with ctx.Pool(
        processes=int(process_count),
        initializer=_init_action_worker,
        initargs=(model_cfg, dict(extra_config or {}), int(top_k)),
    ) as pool:
        for ci, mat, names in pool.imap_unordered(_action_worker, chunks):
            results[int(ci)] = (mat, names)
            completed += 1
            if completed == 1 or completed % 16 == 0 or completed == len(chunks):
                print(
                    f"[action_features] completed_chunks={completed}/{len(chunks)}",
                    flush=True,
                )
    ordered = [results[i] for i in range(len(chunks))]
    feature_names = list(ordered[0][1])
    matrices = [item[0] for item in ordered]
    return (
        np.concatenate(matrices, axis=0).astype(np.float32, copy=False),
        feature_names,
    )


def build_or_load_action_feature_matrix(
    records: list[dict[str, Any]],
    *,
    state_loader: Any,
    model_cfg: Any,
    extra_config: dict[str, Any],
    output_dir: Path,
    split_name: str,
    top_k: int = DEFAULT_TOP_K_ACTIONS,
) -> tuple[np.ndarray, list[str]]:
    cache_enabled = bool(extra_config.get("action_feature_cache_enabled", True))
    cache_dir = Path(
        extra_config.get(
            "action_feature_cache_dir",
            str(output_dir.parent / "_action_feature_cache"),
        )
    )
    fingerprint = _action_records_fingerprint(records, top_k=int(top_k))
    if cache_enabled:
        cached = _load_action_cache(cache_dir, str(split_name), fingerprint, len(records))
        if cached is not None:
            matrix, names = cached
            print(
                f"[action_features] cache hit split={split_name} rows={matrix.shape[0]} "
                f"features={matrix.shape[1]} path={cache_dir}",
                flush=True,
            )
            return matrix, names

    num_processes = int(extra_config.get("action_feature_num_processes", 1))
    chunk_records = int(extra_config.get("action_feature_chunk_records", 1024))
    start_method = str(extra_config.get("action_feature_mp_start_method", "spawn"))

    t0 = time.time()
    if num_processes > 1 and len(records) > max(1, int(chunk_records)):
        matrix, names = _build_action_matrix_parallel(
            records,
            model_cfg=model_cfg,
            extra_config=extra_config,
            top_k=int(top_k),
            num_processes=int(num_processes),
            chunk_records=int(chunk_records),
            start_method=str(start_method),
        )
    else:
        matrix, names = build_action_feature_matrix_from_records(
            records, state_loader=state_loader, top_k=int(top_k)
        )
    seconds = time.time() - t0
    print(
        f"[action_features] built split={split_name} rows={matrix.shape[0]} "
        f"features={matrix.shape[1]} seconds={seconds:.2f}",
        flush=True,
    )

    if cache_enabled:
        _write_action_cache(
            cache_dir,
            str(split_name),
            fingerprint,
            matrix,
            names,
            {
                "cache_version": ACTION_FEATURE_CACHE_VERSION,
                "split_name": str(split_name),
                "rows": int(matrix.shape[0]),
                "features": int(matrix.shape[1]),
                "seconds": float(seconds),
                "feature_num_processes": int(num_processes),
                "feature_chunk_records": int(chunk_records),
                "top_k": int(top_k),
            },
        )
        print(
            f"[action_features] cache written split={split_name} path={cache_dir}",
            flush=True,
        )
    return matrix, names


# ---------------------------------------------------------------------------
# ModelInputs path: GV3 ModelInputs do not carry per-action info, so the
# bootstrap-time caller must derive it from the live state. This helper
# expects the caller to wrap a state and return the same feature row.
# ---------------------------------------------------------------------------


def extract_action_features_from_state(
    state: Any,
    *,
    mcts: Any,
    env: Any,
    top_k: int = DEFAULT_TOP_K_ACTIONS,
) -> tuple[list[float], list[str]]:
    """Variant for live-state callers (e.g. MCTS bootstrap)."""

    names = _action_feature_names(int(top_k))
    if mcts is None or env is None or state is None:
        return _zero_action_feature_vector(int(top_k)), names

    mcts.clear_search_state(drop_scratch=False)
    decision_state = state
    root_cost = float(mcts._state_cost(decision_state))
    root_time = float(decision_state.simulator._time)

    actions_by_index, mask_t = mcts._actions_and_mask(
        decision_state,
        "controller",
        forbidden_stop_ids=None,
    )
    valid_mask = [bool(x) for x in mask_t.tolist()]
    valid_indices = [
        i for i, ok in enumerate(valid_mask) if ok and actions_by_index[i] is not None
    ]
    if not valid_indices:
        return _zero_action_feature_vector(int(top_k)), names

    _alias_to_canon, _canon_to_aliases, canonical_indices = mcts._canonicalize_action_indices(
        player="controller",
        actions_by_index=actions_by_index,
        valid_indices=valid_indices,
    )
    decision_snapshot, decision_stats = mcts._snapshot_state_and_stats(decision_state)

    rewards: list[float] = []
    leaf_costs: list[float] = []
    leaf_times: list[float] = []
    discounts: list[float] = []
    time_deltas: list[float] = []
    for cidx in canonical_indices:
        action = actions_by_index[cidx]
        if action is None:
            continue
        leaf_state = mcts._scratch_restore(decision_snapshot, decision_stats)
        leaf_state = mcts._env.apply_controller_action_only(
            leaf_state, action, inplace=True, fast_forward=False
        )
        leaf_cost = float(mcts._state_cost(leaf_state))
        reward = float(mcts._transition_reward(root_cost, leaf_cost))
        leaf_time = float(leaf_state.simulator._time)
        discount_time_attr = getattr(leaf_state.stats, "transition_discount_time", None)
        discount_time = float(discount_time_attr) if discount_time_attr is not None else leaf_time
        discount = float(mcts._time_discount(discount_time, root_time))
        rewards.append(reward)
        leaf_costs.append(leaf_cost)
        leaf_times.append(leaf_time)
        discounts.append(discount)
        time_deltas.append(max(0.0, leaf_time - root_time))

    if not rewards:
        return _zero_action_feature_vector(int(top_k)), names

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    leaf_costs_arr = np.asarray(leaf_costs, dtype=np.float64)
    discounts_arr = np.asarray(discounts, dtype=np.float64)
    time_deltas_arr = np.asarray(time_deltas, dtype=np.float64)

    values: list[float] = []

    def add(v: Any) -> None:
        try:
            values.append(float(v))
        except Exception:
            values.append(0.0)

    add(int(len(canonical_indices)))
    add(int(len(rewards_arr)))
    add(float(np.min(rewards_arr)))
    add(float(np.max(rewards_arr)))
    add(float(np.mean(rewards_arr)))
    add(float(np.sum(rewards_arr)))
    add(float(np.std(rewards_arr)) if rewards_arr.size > 1 else 0.0)
    add(float(np.min(leaf_costs_arr)))
    add(float(np.max(leaf_costs_arr)))
    add(float(np.mean(leaf_costs_arr)))
    add(float(np.min(discounts_arr)))
    add(float(np.max(discounts_arr)))
    add(float(np.mean(discounts_arr)))
    add(float(np.min(time_deltas_arr)))
    add(float(np.max(time_deltas_arr)))
    add(float(np.mean(time_deltas_arr)))
    add(int((np.abs(rewards_arr) <= 1e-9).sum()))
    add(int((rewards_arr < -0.1).sum()))
    add(int((rewards_arr < -1.0).sum()))
    add(int((rewards_arr < -3.0).sum()))
    add(int((np.abs(leaf_costs_arr) <= 1e-9).sum()))
    add(int((leaf_costs_arr < 1.0).sum()))
    add(int((leaf_costs_arr < 3.0).sum()))
    add(int((np.abs(rewards_arr) <= 1e-9).sum()))
    add(int(((rewards_arr < 0.0) & (rewards_arr >= -0.5)).sum()))
    add(int(((rewards_arr < -0.5) & (rewards_arr >= -1.0)).sum()))
    add(int(((rewards_arr < -1.0) & (rewards_arr >= -2.0)).sum()))
    add(int((rewards_arr < -2.0).sum()))

    order = np.lexsort((leaf_costs_arr, -rewards_arr))
    for i in range(int(top_k)):
        if i < len(order):
            j = int(order[i])
            add(rewards_arr[j])
            add(leaf_costs_arr[j])
            add(time_deltas_arr[j])
            add(discounts_arr[j])
            add(1.0)
        else:
            add(0.0)
            add(0.0)
            add(0.0)
            add(0.0)
            add(0.0)

    return values, names
