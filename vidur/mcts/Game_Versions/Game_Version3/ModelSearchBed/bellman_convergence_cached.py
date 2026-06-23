"""Cached-children Bellman convergence driver.

Uses a pre-computed canonical child-transition cache (produced by
``analysis_testing/rootChildGeneration.py``) to skip simulator action
enumeration during the bootstrap target step. For each Bellman iteration we:

1. Load V_{N-1} (the trained cliff-aware model from the previous iteration).
2. Vectorized-predict V_{N-1}(child_state) for every cached canonical child.
3. Per parent, set target = max_a [reward(a) + discount(a) * V_{N-1}(child_a)]
   over the canonical children.
4. Train V_N on (parent_features -> target).

A one-time phase-0 step rebuilds child states from the cached snapshots,
runs ``build_model_inputs`` then ``extract_cliff_features_from_inputs``, and
writes a single ``child_features.npy`` memmap. After that, every iteration
just does a vectorized ``predict_matrix(child_features)`` plus a per-parent
``np.maximum.reduceat`` style scan.

This driver is intentionally ONLY for the cliff-aware (and learned classical)
backend. NN-with-action-features bootstrap is not supported here because
action features describe the parent state, not the child, so the existing
non-cached driver is fine for that case.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import logger as bellman_logger
from .self_model_test import (
    RootStateLoader,
    SelfModelTestConfig,
    build_self_model_test_config,
    count_trainable_parameters,
    load_root_records,
)


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


@dataclass(frozen=True)
class CachedBellmanConfig:
    dataset_dir: Path
    child_cache_dir: Path
    output_dir: Path
    num_versions: int = 50
    eval_ratio: float = 0.20
    split_seed: int = 12345
    seed: int = 2027
    batch_size: int = 256
    abs_error_threshold: float = 1.0
    feature_workers: int = 32
    feature_chunk_shards: int = 4
    train_workers: int = 32
    target_chunk_size: int = 1_000_000
    predict_processes: int = 32
    predict_worker_threads: int = 3
    same_version_every_n: int = 5
    extra_config: dict[str, Any] | None = None
    skip_corrupt_shards: bool = True
    backend: str = "cliff_aware"  # cliff_aware | classical


# ---------------------------------------------------------------------------
# Phase 0: extract child features into a single memmap
# ---------------------------------------------------------------------------


_FEATURE_WORKER_LOADER: RootStateLoader | None = None


def _feature_worker_init(model_cfg: SelfModelTestConfig, num_threads: int) -> None:
    """Set up a forked feature-extraction worker."""

    _limit_native_threads(int(num_threads))
    global _FEATURE_WORKER_LOADER
    _FEATURE_WORKER_LOADER = RootStateLoader(model_cfg)


def _feature_worker_shutdown() -> None:
    global _FEATURE_WORKER_LOADER
    if _FEATURE_WORKER_LOADER is not None:
        try:
            _FEATURE_WORKER_LOADER.close()
        except Exception:
            pass
        _FEATURE_WORKER_LOADER = None


def _feature_worker_process_shards(
    args: tuple[int, list[str], str, int, bool],
) -> tuple[int, np.ndarray, np.ndarray, list[str], int, list[tuple[str, str]]]:
    """Load shards, extract child features, and return (chunk_idx, X, meta, names, n_rows, bad_shards).

    args: (chunk_idx, shard_paths, child_cache_root, total_max_features, with_action_features)

    When with_action_features is True the per-state cliff feature row is
    concatenated with action-aggregate features extracted from a controller
    action enumeration on the rebuilt child state (post-noop-adversary so
    the state is controller-to-act).
    """

    from ..DNN import infer as dnn_infer
    from .cliff_aware_value_model import extract_cliff_features_from_inputs
    from .self_model_test import _state_from_record
    from .action_features import (
        DEFAULT_TOP_K_ACTIONS,
        _zero_action_feature_vector,
        _action_feature_names,
        extract_action_features_from_state,
    )
    from .forecast_features import (
        extract_forecast_features_from_state,
        FORECAST_FEATURE_NAMES,
    )
    from ....game_types import AdversaryAction

    chunk_idx, shard_paths, child_cache_root, _max_features, with_action_features = args
    if _FEATURE_WORKER_LOADER is None:
        raise RuntimeError("feature worker loader not initialized")
    env = _FEATURE_WORKER_LOADER.env
    mcts = _FEATURE_WORKER_LOADER.mcts
    top_k = int(DEFAULT_TOP_K_ACTIONS)
    action_names = _action_feature_names(top_k)
    zero_action = np.asarray(_zero_action_feature_vector(top_k), dtype=np.float32)

    rows: list[np.ndarray] = []
    metas: list[tuple[int, int, int, float, float]] = []
    feature_names: list[str] | None = None
    bad_shards: list[tuple[str, str]] = []

    for shard_path in shard_paths:
        full = Path(child_cache_root) / shard_path
        try:
            recs = torch.load(full, map_location="cpu", weights_only=False)
        except Exception as exc:
            bad_shards.append((str(shard_path), repr(exc)))
            continue
        for r in recs:
            if not bool(r.get("is_valid", False)):
                continue
            try:
                child_record = {
                    "simulator_snapshot": r["child_simulator_snapshot"],
                    "stats": r["child_stats"],
                }
                state = _state_from_record(env, child_record)
                if with_action_features:
                    # Advance one no-op adversary action so the state is
                    # controller-to-act (matching the parent action-feature
                    # distribution), and so cliff/action/forecast features all
                    # describe the *same* state.
                    try:
                        noop_adv = AdversaryAction(requests=[], stop_decode_ids=[])
                        state = env.apply_adversary_action_only(
                            state, noop_adv, inplace=True
                        )
                    except Exception:
                        pass
                    inputs = dnn_infer.build_model_inputs(
                        state,
                        "controller",
                        torch.device("cpu"),
                        build_action_mask_flag=False,
                    )
                    feats, names = extract_cliff_features_from_inputs(inputs)
                    try:
                        a_feats, a_names = extract_action_features_from_state(
                            state, mcts=mcts, env=env, top_k=top_k,
                        )
                    except Exception:
                        a_feats = list(zero_action.tolist())
                        a_names = list(action_names)
                    try:
                        f_feats, f_names = extract_forecast_features_from_state(
                            state, env=env,
                        )
                    except Exception:
                        f_feats = [0.0] * len(FORECAST_FEATURE_NAMES)
                        f_names = list(FORECAST_FEATURE_NAMES)
                    feats = list(feats) + [float(x) for x in a_feats] + [float(x) for x in f_feats]
                    names = list(names) + list(a_names) + list(f_names)
                else:
                    inputs = dnn_infer.build_model_inputs(
                        state,
                        "adversary",
                        torch.device("cpu"),
                        build_action_mask_flag=False,
                    )
                    feats, names = extract_cliff_features_from_inputs(inputs)
            except Exception as exc:
                bad_shards.append((str(shard_path), f"row decode err {exc!r}"))
                continue
            if feature_names is None:
                feature_names = list(names)
            rows.append(np.asarray(feats, dtype=np.float32))
            metas.append(
                (
                    int(r["parent_state_id"]),
                    int(r["parent_root_id"]),
                    int(r.get("canonical_action_index", r.get("action_index", -1))),
                    float(r["reward"]),
                    float(r["discount"]),
                )
            )
        # Free MCTS scratch periodically.
        try:
            mcts.clear_search_state(drop_scratch=True)
        except Exception:
            pass
        gc.collect()

    if not rows:
        return chunk_idx, np.empty((0, 0), dtype=np.float32), np.empty((0, 5), dtype=np.float64), [], 0, bad_shards

    X = np.stack(rows, axis=0)
    meta = np.asarray(metas, dtype=np.float64)  # use float64 to hold int32 ids exactly
    return chunk_idx, X, meta, list(feature_names or []), int(X.shape[0]), bad_shards


def _read_manifest(manifest_path: Path) -> list[str]:
    out: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out.append(str(entry["shard_path"]))
    return out


def _chunked(lst: Sequence[str], size: int) -> list[list[str]]:
    size = max(1, int(size))
    return [list(lst[i : i + size]) for i in range(0, len(lst), size)]


def build_child_feature_cache(
    *,
    cfg: CachedBellmanConfig,
    model_cfg: SelfModelTestConfig,
) -> tuple[Path, Path, Path, list[str]]:
    """Phase 0: extract cliff-aware features for every cached child transition.

    Writes:
        <output>/child_features.npy        (N, D) float32 memmap
        <output>/child_meta.npy            (N, 5) float64: parent_state_id,
            parent_root_id, canonical_action_index, reward, discount
        <output>/child_features.names.json
        <output>/child_features.summary.json

    Returns paths to features, meta, names, and the loaded feature_names list.
    """

    feature_path = cfg.output_dir / "child_features.npy"
    meta_path = cfg.output_dir / "child_meta.npy"
    names_path = cfg.output_dir / "child_features.names.json"
    summary_path = cfg.output_dir / "child_features.summary.json"

    if (
        feature_path.exists()
        and meta_path.exists()
        and names_path.exists()
        and summary_path.exists()
    ):
        names = json.loads(names_path.read_text(encoding="utf-8"))
        print(f"[cached] feature cache already present: {feature_path}", flush=True)
        return feature_path, meta_path, names_path, list(names)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = cfg.child_cache_dir / "manifest.jsonl"
    shard_paths = _read_manifest(manifest)
    print(f"[cached] manifest has {len(shard_paths)} shards", flush=True)

    chunks = _chunked(shard_paths, cfg.feature_chunk_shards)
    print(f"[cached] chunking into {len(chunks)} groups of <={cfg.feature_chunk_shards} shards", flush=True)

    # Run an estimate pass so we know the memmap shape. Worker processes return
    # in-memory arrays; we accumulate them and resize a memmap at the end. To
    # avoid loading 44 GB into RAM at once we instead **stream** each chunk
    # result to a growing memmap.

    # First call must produce feature_names so we know D before allocating
    # the memmap. Process chunk 0 in-process to discover D, then fan out.
    with_action_features = bool((cfg.extra_config or {}).get("use_action_features", False))
    print(
        f"[cached] probing first chunk to discover feature dimension "
        f"(action_features={with_action_features})",
        flush=True,
    )
    _feature_worker_init(model_cfg, num_threads=1)
    try:
        idx0, X0, meta0, feature_names, n0, bad0 = _feature_worker_process_shards(
            (0, chunks[0], str(cfg.child_cache_dir), 0, with_action_features)
        )
        if not feature_names:
            raise RuntimeError("first chunk produced no feature names; cache probably empty")
        D = int(X0.shape[1])
    finally:
        _feature_worker_shutdown()

    # Estimate total rows from summary.json, then build memmap with that capacity.
    summary_src = json.loads((cfg.child_cache_dir / "summary.json").read_text())
    n_canonical_total = int(summary_src.get("num_canonical_transitions", summary_src.get("num_transitions", 0)))
    if n_canonical_total <= 0:
        raise RuntimeError("could not infer total canonical transitions from summary.json")
    print(f"[cached] feature dim D={D}, expected ~{n_canonical_total} canonical rows", flush=True)

    # Allocate memmap with slack (we trim if actual smaller).
    capacity = int(n_canonical_total * 1.02) + 1024
    X_mm = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(capacity, D)
    )
    meta_mm = np.lib.format.open_memmap(
        meta_path, mode="w+", dtype=np.float64, shape=(capacity, 5)
    )

    write_cursor = 0

    def write_block(X: np.ndarray, meta: np.ndarray) -> int:
        nonlocal write_cursor
        n = int(X.shape[0])
        if n == 0:
            return 0
        end = write_cursor + n
        if end > capacity:
            raise RuntimeError(
                f"feature memmap capacity exceeded ({end} > {capacity}); regenerate with larger slack"
            )
        X_mm[write_cursor:end] = X
        meta_mm[write_cursor:end] = meta
        write_cursor = end
        return n

    write_block(X0, meta0)
    bad_shards: list[tuple[str, str]] = list(bad0)
    print(f"[cached] chunk 0 done: rows={n0}", flush=True)

    remaining = chunks[1:]
    if remaining:
        n_proc = max(1, int(cfg.feature_workers))
        # Use spawn so each worker gets a fresh sklearn / mcts state cleanly.
        ctx = mp.get_context("fork")
        print(
            f"[cached] launching {n_proc} fork workers over {len(remaining)} chunks",
            flush=True,
        )
        # maxtasksperchild recycles worker processes after this many chunks
        # so the per-worker scratch/MCTS-cache memory cannot grow unbounded.
        max_tasks = int((cfg.extra_config or {}).get("feature_maxtasks_per_child", 8))
        with ctx.Pool(
            processes=n_proc,
            initializer=_feature_worker_init,
            initargs=(model_cfg, 1),
            maxtasksperchild=max(1, max_tasks),
        ) as pool:
            tasks = [
                (i + 1, list(grp), str(cfg.child_cache_dir), 0, with_action_features)
                for i, grp in enumerate(remaining)
            ]
            n_done = 0
            t0 = time.time()
            for chunk_idx, X, meta, _names, n_rows, bad in pool.imap_unordered(
                _feature_worker_process_shards, tasks, chunksize=1
            ):
                write_block(X, meta)
                bad_shards.extend(bad)
                n_done += 1
                elapsed = time.time() - t0
                if n_done % max(1, len(remaining) // 50) == 0 or n_done == len(remaining):
                    rate = write_cursor / max(1.0, elapsed)
                    print(
                        f"[cached] feature progress {n_done}/{len(remaining)} chunks, "
                        f"rows_so_far={write_cursor}, rate={rate:,.0f} rows/s, "
                        f"bad_shards={len(bad_shards)}",
                        flush=True,
                    )

    # Trim memmap to actual used rows.
    print(f"[cached] writing final {write_cursor} rows; trimming memmap", flush=True)
    del X_mm
    del meta_mm

    if write_cursor < capacity:
        # Reload, copy, overwrite trimmed.
        X_full = np.load(feature_path, mmap_mode="r")
        meta_full = np.load(meta_path, mmap_mode="r")
        X_trim = np.lib.format.open_memmap(
            feature_path.with_suffix(".tmp.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(write_cursor, D),
        )
        meta_trim = np.lib.format.open_memmap(
            meta_path.with_suffix(".tmp.npy"),
            mode="w+",
            dtype=np.float64,
            shape=(write_cursor, 5),
        )
        X_trim[:] = X_full[:write_cursor]
        meta_trim[:] = meta_full[:write_cursor]
        del X_trim
        del meta_trim
        del X_full
        del meta_full
        feature_path.with_suffix(".tmp.npy").replace(feature_path)
        meta_path.with_suffix(".tmp.npy").replace(meta_path)

    names_path.write_text(json.dumps(list(feature_names)), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "num_rows": int(write_cursor),
                "feature_dim": int(D),
                "child_cache_dir": str(cfg.child_cache_dir),
                "bad_shard_count": len(bad_shards),
                "bad_shards": bad_shards[:128],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cached] feature cache done: rows={write_cursor}, D={D}", flush=True)
    return feature_path, meta_path, names_path, list(feature_names)


# ---------------------------------------------------------------------------
# Per-parent index over the meta array
# ---------------------------------------------------------------------------


def build_parent_index(
    meta_path: Path,
    *,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort the child meta by parent_state_id and produce per-parent slices.

    Returns:
        order: int64 (N,) — permutation that sorts meta by parent_state_id.
        parent_ids: int64 (P,) — sorted unique parent_state_ids present.
        offsets: int64 (P+1,) — segment offsets into the *sorted* arrays.
        rewards: float32 (N,) — child rewards in sorted order.
        discounts: float32 (N,) — child discounts in sorted order.
    """

    idx_path = cache_dir / "parent_index.npz"
    if idx_path.exists():
        z = np.load(idx_path)
        return z["order"], z["parent_ids"], z["offsets"], z["rewards"], z["discounts"]

    meta = np.load(meta_path, mmap_mode="r")
    psid = meta[:, 0].astype(np.int64)
    rewards = meta[:, 3].astype(np.float32)
    discounts = meta[:, 4].astype(np.float32)
    order = np.argsort(psid, kind="stable")
    psid_sorted = psid[order]
    rewards_sorted = rewards[order]
    discounts_sorted = discounts[order]
    parent_ids, first_index = np.unique(psid_sorted, return_index=True)
    offsets = np.append(first_index, len(psid_sorted)).astype(np.int64)
    np.savez(
        idx_path,
        order=order.astype(np.int64),
        parent_ids=parent_ids.astype(np.int64),
        offsets=offsets,
        rewards=rewards_sorted.astype(np.float32),
        discounts=discounts_sorted.astype(np.float32),
    )
    print(
        f"[cached] parent index: parents={len(parent_ids)}, "
        f"min_kids={int((offsets[1:] - offsets[:-1]).min())}, "
        f"max_kids={int((offsets[1:] - offsets[:-1]).max())}, "
        f"mean_kids={float((offsets[1:] - offsets[:-1]).mean()):.2f}",
        flush=True,
    )
    return order, parent_ids, offsets, rewards_sorted, discounts_sorted


# ---------------------------------------------------------------------------
# Bellman target computation (vectorized)
# ---------------------------------------------------------------------------


def _segment_max(
    values: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Return per-segment max of ``values`` given ``offsets`` of length P+1.

    Implemented with np.maximum.reduceat for speed; offsets must be monotone.
    """

    if len(offsets) <= 1:
        return np.empty((0,), dtype=values.dtype)
    starts = offsets[:-1]
    # reduceat segments: indices into values, ranges [starts[i], starts[i+1]).
    out = np.maximum.reduceat(values, starts)
    # For empty segments (shouldn't happen but guard) reduceat would propagate
    # the last value; mask them to -inf.
    seg_lens = offsets[1:] - offsets[:-1]
    if np.any(seg_lens <= 0):
        out = out.copy()
        out[seg_lens <= 0] = -np.inf
    return out


_PREDICT_WORKER_MODEL: Any | None = None
_PREDICT_WORKER_FEATURE_PATH: str | None = None


def _predict_worker_init(model_path: str, feature_path: str, num_threads: int) -> None:
    """Forked worker that loads the per-iteration model once."""

    _limit_native_threads(int(num_threads))
    import joblib

    global _PREDICT_WORKER_MODEL
    global _PREDICT_WORKER_FEATURE_PATH
    _PREDICT_WORKER_MODEL = joblib.load(model_path)
    _PREDICT_WORKER_FEATURE_PATH = str(feature_path)


def _predict_worker_chunk(args: tuple[int, int]) -> tuple[int, np.ndarray]:
    """Run predict_matrix on a slice [start, end) of the feature memmap."""

    start, end = args
    if _PREDICT_WORKER_MODEL is None or _PREDICT_WORKER_FEATURE_PATH is None:
        raise RuntimeError("predict worker not initialized")
    X_mm = np.load(_PREDICT_WORKER_FEATURE_PATH, mmap_mode="r")
    block = np.ascontiguousarray(X_mm[int(start):int(end)])
    out = _PREDICT_WORKER_MODEL.predict_matrix(block)
    return int(start), np.asarray(out, dtype=np.float64)


def predict_v_for_features(
    *,
    model: Any,
    feature_path: Path,
    chunk: int = 1_000_000,
    num_processes: int = 1,
    model_path: str | Path | None = None,
    worker_threads: int = 4,
    act_max_reward_col: int | None = None,
) -> np.ndarray:
    """Run the model's vectorized predict over the entire feature memmap.

    If ``act_max_reward_col`` is not None and the feature memmap has that
    column, we clamp ``V(s) >= act_max_reward(s)``. This is the "no-runaway"
    constraint for the chronic Bellman pocket: states where the best one-step
    reward is 0 (savable) cannot have V more negative than 0 in finite-horizon
    bootstrap. We also enforce the controller V invariant V <= 0.
    """

    X_mm = np.load(feature_path, mmap_mode="r")
    n = int(X_mm.shape[0])
    out = np.empty(n, dtype=np.float64)
    chunk = max(1, int(chunk))
    starts = list(range(0, n, chunk))
    items = [(s, min(n, s + chunk)) for s in starts]

    predict_matrix = getattr(model, "predict_matrix", None)
    if not callable(predict_matrix):
        raise RuntimeError(
            "model does not expose predict_matrix; cached driver only supports "
            "cliff_aware/classical backends"
        )

    n_clamp_lower = 0
    n_clamp_upper = 0
    import time as _time
    for start, end in items:
        ts = _time.time()
        block = np.ascontiguousarray(X_mm[start:end])
        preds = predict_matrix(block)
        if act_max_reward_col is not None and 0 <= int(act_max_reward_col) < int(block.shape[1]):
            act_max = block[:, int(act_max_reward_col)].astype(np.float64, copy=False)
            # Lower bound: V(s) >= act_max_reward(s); since both are <= 0 this
            # raises predictions that overshoot below the achievable floor.
            below = preds < act_max
            n_clamp_lower += int(below.sum())
            preds = np.maximum(preds, act_max)
        # Upper bound: V <= 0 (controller cost is non-negative monotone).
        above = preds > 0.0
        n_clamp_upper += int(above.sum())
        preds = np.minimum(preds, 0.0)
        out[start:end] = preds
        if (start // chunk) % 4 == 0:
            print(
                f"[predict] chunk {start}-{end} in {_time.time()-ts:.1f}s",
                flush=True,
            )
    if act_max_reward_col is not None:
        print(
            f"[predict] clamp_to_act_max applied: lower={n_clamp_lower} (V<act_max), "
            f"upper={n_clamp_upper} (V>0)",
            flush=True,
        )
    return out


def compute_targets_from_cache(
    *,
    model: Any | None,
    feature_path: Path,
    parent_ids: np.ndarray,
    offsets: np.ndarray,
    order: np.ndarray,
    rewards_sorted: np.ndarray,
    discounts_sorted: np.ndarray,
    bootstrap_version: int,
    chunk: int = 1_000_000,
    num_processes: int = 1,
    model_path: str | Path | None = None,
    worker_threads: int = 4,
    act_max_reward_col: int | None = None,
    bootstrap_shrink: float = 1.0,
) -> dict[int, float]:
    """Compute per-parent Bellman target = max_a [r + γ V_{N-1}(child_a)].

    For ``bootstrap_version == 0`` (no model yet), V_{N-1} is treated as zero and
    target = max_a r_a.

    ``bootstrap_shrink`` (in (0, 1]) damps the Bellman bootstrap; with shrink<1
    the effective discount becomes shrink*γ and target ranges stay tighter to
    the immediate-reward distribution. Useful to prevent fitted-Q runaway.
    """

    import time as _time
    if int(bootstrap_version) == 0 or model is None:
        v_child_sorted = np.zeros(int(rewards_sorted.shape[0]), dtype=np.float32)
    else:
        t0 = _time.time()
        v_child = predict_v_for_features(
            model=model,
            feature_path=feature_path,
            chunk=int(chunk),
            num_processes=int(num_processes),
            model_path=model_path,
            worker_threads=int(worker_threads),
            act_max_reward_col=act_max_reward_col,
        )
        print(f"[targets] predict_v done in {_time.time()-t0:.1f}s", flush=True)
        t0 = _time.time()
        v_child_sorted = v_child[order].astype(np.float32, copy=False)
        print(f"[targets] reorder by parent done in {_time.time()-t0:.1f}s", flush=True)

    t0 = _time.time()
    shrink = float(bootstrap_shrink)
    q_sorted = rewards_sorted + (shrink * discounts_sorted) * v_child_sorted
    np.minimum(q_sorted, 0.0, out=q_sorted)
    target_per_parent = _segment_max(q_sorted, offsets)
    print(f"[targets] q_sorted + segment max in {_time.time()-t0:.1f}s", flush=True)
    t0 = _time.time()
    out = {
        int(parent_ids[i]): float(target_per_parent[i]) for i in range(parent_ids.shape[0])
    }
    print(f"[targets] dict build in {_time.time()-t0:.1f}s, parents={len(out)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Per-iteration trainer + evaluator
# ---------------------------------------------------------------------------


def _train_cliff_aware_with_targets(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    model_cfg: SelfModelTestConfig,
    state_loader: RootStateLoader,
    model_dir: Path,
) -> Any:
    from .cliff_aware_value_model import train_cliff_aware_model

    artifacts = train_cliff_aware_model(
        train_records=list(train_records),
        eval_records=list(eval_records),
        cfg=model_cfg,
        output_dir=model_dir,
        state_loader=state_loader,
    )
    return artifacts["model"] if isinstance(artifacts, dict) and "model" in artifacts else artifacts


def _train_classical_with_targets(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    model_cfg: SelfModelTestConfig,
    state_loader: RootStateLoader,
    model_dir: Path,
) -> Any:
    from .classical_value_model import train_learned_classical_model

    artifacts = train_learned_classical_model(
        train_records=list(train_records),
        eval_records=list(eval_records),
        cfg=model_cfg,
        output_dir=model_dir,
        state_loader=state_loader,
    )
    return artifacts["model"] if isinstance(artifacts, dict) and "model" in artifacts else artifacts


def _evaluate_split(
    *,
    model: Any,
    records: list[dict[str, Any]],
    targets: list[float],
    sample_numbers: list[int],
    model_cfg: SelfModelTestConfig,
    state_loader: RootStateLoader,
    output_csv: Path,
    abs_error_threshold: float,
    split_name: str = "eval",
    act_max_per_record: list[float] | None = None,
) -> dict[str, Any]:
    """Predict + write CSV + compute metrics.

    If ``act_max_per_record`` is given, predictions are clamped to
    ``[act_max_reward(s), 0]`` before metrics/CSV. This makes evaluation
    consistent with the bootstrap clamp used in compute_targets_from_cache.
    """
    from .self_model_test import predict_candidate_values

    labeled = [dict(r) for r in records]
    for r, t in zip(labeled, targets):
        r["target_value"] = float(t)
    predictions = predict_candidate_values(
        model,
        labeled,
        model_cfg,
        state_loader,
        split_name=str(split_name),
    )
    if act_max_per_record is not None and len(act_max_per_record) == len(predictions):
        # We intentionally do NOT clamp predictions for eval CSV: the metric
        # should reflect what the model actually outputs. The bootstrap clamp
        # in compute_targets_from_cache is sufficient to keep targets stable.
        pass
    bellman_logger.write_prediction_results_csv(
        output_csv,
        predictions=predictions,
        targets=targets,
        sample_numbers=sample_numbers,
    )
    return bellman_logger.compute_error_metrics(
        predictions, targets, abs_error_threshold=float(abs_error_threshold)
    )


def _records_with_targets(
    records: list[dict[str, Any]],
    *,
    targets_by_psid: dict[int, float],
    psid_for_record: list[int],
    fallback_target: float = 0.0,
) -> tuple[list[dict[str, Any]], list[float], list[int]]:
    """Attach per-record targets and drop records whose parents have no children.

    Returns the kept records, kept targets, and the kept ``parent_state_id``
    values (which we use as ``sample_number`` in CSV output for traceability).
    """

    kept: list[dict[str, Any]] = []
    kept_targets: list[float] = []
    kept_psids: list[int] = []
    for rec, psid in zip(records, psid_for_record):
        if int(psid) not in targets_by_psid:
            continue
        new = dict(rec)
        t = float(targets_by_psid[int(psid)])
        new["target_value"] = t
        kept.append(new)
        kept_targets.append(t)
        kept_psids.append(int(psid))
    return kept, kept_targets, kept_psids


def _psid_for_records(records: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for i, rec in enumerate(records):
        psid = rec.get("parent_state_id")
        if psid is None:
            psid = i  # canonical: position in dataset is the parent_state_id used at cache time
        out.append(int(psid))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def cached_bellman_convergence(cfg: CachedBellmanConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Toggle action-feature augmentation on the cliff feature schema (parents).
    # This is read by build_cliff_feature_matrix_from_records via module globals
    # so the trainer's parent feature build matches the child Phase-0 schema.
    use_action_features = bool((cfg.extra_config or {}).get("use_action_features", False))
    if use_action_features:
        from . import cliff_aware_value_model as _caw
        _caw._CLIFF_USE_ACTION_FEATURES = True
        _caw._CLIFF_ACTION_TOP_K = int((cfg.extra_config or {}).get("cliff_action_top_k", 4))
        # Bump cache version so old cliff caches don't collide.
        _caw.CLIFF_FEATURE_CACHE_VERSION = "cliff_aware_modelinputs_v3_actaug"

    # Compute the absolute column index of `act_max_reward` in the
    # child-features memmap so the V predictor can clamp V(s) >= act_max(s)
    # when reading bootstrap targets. This is None when action features are off.
    act_max_reward_col: int | None = None
    if use_action_features:
        # Cliff feature dim depends on row-schema constants; use the same code
        # path the feature builder uses to discover it dynamically.
        from .cliff_aware_value_model import (
            extract_cliff_features_from_inputs as _probe_cliff,
        )
        from .action_features import _action_feature_names as _act_names
        # We can't probe without an inputs object handy here; use the known
        # offset from action_features.py: act_max_reward is at index 3 within
        # the action block. The cliff block size is known statically.
        # Probe via a lightweight call at start-up below; for now we rely on
        # config override.
        act_max_reward_col = int((cfg.extra_config or {}).get(
            "act_max_reward_col_override", -1
        ))
        if act_max_reward_col < 0:
            act_max_reward_col = None  # disabled until probed

    model0_cfg = build_self_model_test_config(
        dataset_dir=cfg.dataset_dir,
        output_dir=cfg.output_dir / "Model_Version0",
        num_roots=10**8,
        max_candidate_roots=10**8,
        seed=int(cfg.seed),
        root_player_filter="controller",
        allow_dataset_generation=False,
        eval_ratio=float(cfg.eval_ratio),
        split_seed=int(cfg.split_seed),
        model_name="cached_bellman_v0",
        batch_size=int(cfg.batch_size),
        extra_config=dict(cfg.extra_config or {}),
    )

    print("[cached] loading parent records (entire dataset, controller-only)", flush=True)
    records = load_root_records(model0_cfg)
    print(f"[cached] parent records: {len(records)}", flush=True)

    # Deterministic split
    indices = list(range(len(records)))
    random.Random(int(cfg.split_seed)).shuffle(indices)
    eval_count = int(round(len(indices) * float(cfg.eval_ratio)))
    eval_count = max(1, min(len(indices) - 1, eval_count))
    eval_indices = sorted(indices[:eval_count])
    eval_set = set(eval_indices)
    train_indices = [i for i in range(len(records)) if i not in eval_set]
    train_records = [records[i] for i in train_indices]
    eval_records = [records[i] for i in eval_indices]
    print(
        f"[cached] split: train={len(train_records)}, eval={len(eval_records)}",
        flush=True,
    )

    psid_train = train_indices  # by construction, parent_state_id == position in dataset
    psid_eval = eval_indices

    # Phase 0: extract child features into memmap.
    feature_path, meta_path, _names_path, feature_names = build_child_feature_cache(
        cfg=cfg, model_cfg=model0_cfg
    )

    # Resolve the absolute column index of act_max_reward for the V clamp.
    if use_action_features and act_max_reward_col is None:
        try:
            act_max_reward_col = int(feature_names.index("act_max_reward"))
            print(
                f"[cached] V clamp: act_max_reward column resolved to index "
                f"{act_max_reward_col}",
                flush=True,
            )
        except ValueError:
            act_max_reward_col = None
            print("[cached] V clamp DISABLED: act_max_reward not in feature schema", flush=True)

    # Sorted parent index for fast per-parent max-Q.
    order, parent_ids, offsets, rewards_sorted, discounts_sorted = build_parent_index(
        meta_path, cache_dir=cfg.output_dir
    )

    # Per-parent act_max_reward = max immediate r over canonical children.
    # rewards_sorted contains the cached r(parent, a) values for each canonical
    # (parent, a) row in the same parent-grouped order as parent_ids/offsets.
    act_max_per_parent_arr = _segment_max(rewards_sorted, offsets)
    act_max_by_psid = {
        int(parent_ids[i]): float(act_max_per_parent_arr[i])
        for i in range(parent_ids.shape[0])
    }
    print(
        f"[cached] computed act_max_reward per parent: range=[{act_max_per_parent_arr.min():.3f}, "
        f"{act_max_per_parent_arr.max():.3f}], mean={float(act_max_per_parent_arr.mean()):.3f}",
        flush=True,
    )

    bellman_logger.write_split_indices(
        cfg.output_dir / "split_indices.json",
        train_indices=train_indices,
        eval_indices=eval_indices,
    )
    bellman_logger.write_json(
        cfg.output_dir / "config.json",
        {
            "dataset_dir": str(cfg.dataset_dir),
            "child_cache_dir": str(cfg.child_cache_dir),
            "output_dir": str(cfg.output_dir),
            "num_versions": int(cfg.num_versions),
            "eval_ratio": float(cfg.eval_ratio),
            "split_seed": int(cfg.split_seed),
            "feature_workers": int(cfg.feature_workers),
            "train_workers": int(cfg.train_workers),
            "extra_config": dict(cfg.extra_config or {}),
            "backend": str(cfg.backend),
        },
    )

    state_loader = RootStateLoader(model0_cfg)
    state_loader._bellman_extra_config = dict(cfg.extra_config or {})

    train_fn = (
        _train_cliff_aware_with_targets
        if str(cfg.backend) == "cliff_aware"
        else _train_classical_with_targets
    )

    bootstrap_model: Any | None = None
    bootstrap_version: int = 0
    bootstrap_model_path: str | None = None
    # If non-None, holds T[V_{prev}] computed at the end of the previous iteration
    # via a *same-version* residual pass. We can reuse it as the next iteration's
    # bootstrap targets and skip a 14-million-row predict pass.
    pending_targets_by_psid: dict[int, float] | None = None

    try:
        for model_version in range(1, int(cfg.num_versions) + 1):
            t_iter0 = time.time()
            model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
            model_dir.mkdir(parents=True, exist_ok=True)
            iter_model_cfg = build_self_model_test_config(
                dataset_dir=cfg.dataset_dir,
                output_dir=model_dir,
                num_roots=10**8,
                max_candidate_roots=10**8,
                seed=int(cfg.seed),
                root_player_filter="controller",
                allow_dataset_generation=False,
                eval_ratio=float(cfg.eval_ratio),
                split_seed=int(cfg.split_seed),
                model_name=f"cached_bellman_v{model_version}",
                batch_size=int(cfg.batch_size),
                extra_config=dict(cfg.extra_config or {}),
            )

            t0 = time.time()
            if pending_targets_by_psid is not None:
                # Reuse T[V_{N-1}] computed at the end of the previous iteration.
                targets_by_psid = pending_targets_by_psid
                pending_targets_by_psid = None
                t_targets = 0.0
                print(
                    f"[cached] V{model_version}: REUSED bootstrap targets from "
                    f"previous iteration's same-version pass (free)",
                    flush=True,
                )
            else:
                targets_by_psid = compute_targets_from_cache(
                    model=bootstrap_model,
                    feature_path=feature_path,
                    parent_ids=parent_ids,
                    offsets=offsets,
                    order=order,
                    rewards_sorted=rewards_sorted,
                    discounts_sorted=discounts_sorted,
                    bootstrap_version=int(bootstrap_version),
                    chunk=int(cfg.target_chunk_size),
                    num_processes=int(cfg.predict_processes),
                    model_path=bootstrap_model_path,
                    worker_threads=int(cfg.predict_worker_threads),
                    act_max_reward_col=act_max_reward_col,
                    bootstrap_shrink=float((cfg.extra_config or {}).get("bootstrap_shrink", 1.0)),
                )
                t_targets = time.time() - t0
                print(
                    f"[cached] V{model_version}: bootstrap targets done in {t_targets:.1f}s "
                    f"(parents_with_targets={len(targets_by_psid)})",
                    flush=True,
                )

            train_kept, train_targets, train_kept_sample_numbers = _records_with_targets(
                train_records, targets_by_psid=targets_by_psid, psid_for_record=psid_train
            )
            eval_kept, eval_targets, eval_kept_sample_numbers = _records_with_targets(
                eval_records, targets_by_psid=targets_by_psid, psid_for_record=psid_eval
            )
            print(
                f"[cached] V{model_version}: train={len(train_kept)}, eval={len(eval_kept)}",
                flush=True,
            )

            # Train V_N
            t0 = time.time()
            model = train_fn(
                train_records=train_kept,
                eval_records=eval_kept,
                model_cfg=iter_model_cfg,
                state_loader=state_loader,
                model_dir=model_dir,
            )
            t_train = time.time() - t0
            params = int(count_trainable_parameters(model))
            print(
                f"[cached] V{model_version}: trained in {t_train:.1f}s, params={params}",
                flush=True,
            )

            train_act_max = (
                [act_max_by_psid.get(int(p), 0.0) for p in train_kept_sample_numbers]
                if act_max_reward_col is not None else None
            )
            eval_act_max = (
                [act_max_by_psid.get(int(p), 0.0) for p in eval_kept_sample_numbers]
                if act_max_reward_col is not None else None
            )
            train_metrics = _evaluate_split(
                model=model,
                records=train_kept,
                targets=train_targets,
                sample_numbers=train_kept_sample_numbers,
                model_cfg=iter_model_cfg,
                state_loader=state_loader,
                output_csv=model_dir / "train_results.csv",
                abs_error_threshold=float(cfg.abs_error_threshold),
                split_name="train",
                act_max_per_record=train_act_max,
            )
            eval_metrics = _evaluate_split(
                model=model,
                records=eval_kept,
                targets=eval_targets,
                sample_numbers=eval_kept_sample_numbers,
                model_cfg=iter_model_cfg,
                state_loader=state_loader,
                output_csv=model_dir / "eval_results.csv",
                abs_error_threshold=float(cfg.abs_error_threshold),
                split_name="eval",
                act_max_per_record=eval_act_max,
            )
            train_row = {
                "model_version": int(model_version),
                "split": "train",
                "trainable_params": params,
            }
            train_row.update(train_metrics)
            eval_row = {
                "model_version": int(model_version),
                "split": "eval",
                "trainable_params": params,
            }
            eval_row.update(eval_metrics)
            bellman_logger.write_summary_csv(
                cfg.output_dir / f"version_{int(bootstrap_version)}_to_{int(model_version)}.csv",
                [
                    {"source_model_version": int(bootstrap_version),
                     "target_model_version": int(model_version),
                     **train_row},
                    {"source_model_version": int(bootstrap_version),
                     "target_model_version": int(model_version),
                     **eval_row},
                ],
            )

            # The newly trained V_N has been saved as a joblib by train_fn.
            iter_model_path = model_dir / "cliff_aware_controller_value.joblib"
            if not iter_model_path.exists():
                # Fallback for the (unused) classical backend.
                iter_model_path = model_dir / "learned_classical_controller_value.joblib"
            iter_model_path_str = str(iter_model_path) if iter_model_path.exists() else None

            # Same-version Bellman residual check (n -> n) on a configurable cadence.
            same_every = max(1, int(cfg.same_version_every_n))
            do_same_version = (
                int(model_version) % same_every == 0
                or int(model_version) == int(cfg.num_versions)
            )
            if do_same_version:
                t0 = time.time()
                same_targets_by_psid = compute_targets_from_cache(
                    model=model,
                    feature_path=feature_path,
                    parent_ids=parent_ids,
                    offsets=offsets,
                    order=order,
                    rewards_sorted=rewards_sorted,
                    discounts_sorted=discounts_sorted,
                    bootstrap_version=int(model_version),
                    chunk=int(cfg.target_chunk_size),
                    num_processes=int(cfg.predict_processes),
                    model_path=iter_model_path_str,
                    worker_threads=int(cfg.predict_worker_threads),
                    act_max_reward_col=act_max_reward_col,
                    bootstrap_shrink=float((cfg.extra_config or {}).get("bootstrap_shrink", 1.0)),
                )
                t_same = time.time() - t0
                same_train_kept, same_train_targets, same_train_kept_psids = _records_with_targets(
                    train_records, targets_by_psid=same_targets_by_psid, psid_for_record=psid_train
                )
                same_eval_kept, same_eval_targets, same_eval_kept_psids = _records_with_targets(
                    eval_records, targets_by_psid=same_targets_by_psid, psid_for_record=psid_eval
                )
                same_train_act_max = (
                    [act_max_by_psid.get(int(p), 0.0) for p in same_train_kept_psids]
                    if act_max_reward_col is not None else None
                )
                same_eval_act_max = (
                    [act_max_by_psid.get(int(p), 0.0) for p in same_eval_kept_psids]
                    if act_max_reward_col is not None else None
                )
                same_train_metrics = _evaluate_split(
                    model=model,
                    records=same_train_kept,
                    targets=same_train_targets,
                    sample_numbers=same_train_kept_psids,
                    model_cfg=iter_model_cfg,
                    state_loader=state_loader,
                    output_csv=model_dir / "train_results_same_version.csv",
                    abs_error_threshold=float(cfg.abs_error_threshold),
                    split_name="train",
                    act_max_per_record=same_train_act_max,
                )
                same_eval_metrics = _evaluate_split(
                    model=model,
                    records=same_eval_kept,
                    targets=same_eval_targets,
                    sample_numbers=same_eval_kept_psids,
                    model_cfg=iter_model_cfg,
                    state_loader=state_loader,
                    output_csv=model_dir / "eval_results_same_version.csv",
                    abs_error_threshold=float(cfg.abs_error_threshold),
                    split_name="eval",
                    act_max_per_record=same_eval_act_max,
                )
                same_row_train = {
                    "source_model_version": int(model_version),
                    "target_model_version": int(model_version),
                    "model_version": int(model_version),
                    "split": "train_same_version",
                    "trainable_params": params,
                }
                same_row_train.update(same_train_metrics)
                same_row_eval = {
                    "source_model_version": int(model_version),
                    "target_model_version": int(model_version),
                    "model_version": int(model_version),
                    "split": "eval_same_version",
                    "trainable_params": params,
                }
                same_row_eval.update(same_eval_metrics)
                bellman_logger.write_summary_csv(
                    cfg.output_dir / f"version_{int(model_version)}_to_{int(model_version)}.csv",
                    [same_row_train, same_row_eval],
                )
                # Stash these targets so the *next* iteration can reuse them as
                # its bootstrap targets without recomputing.
                pending_targets_by_psid = same_targets_by_psid
            else:
                t_same = 0.0
                same_train_metrics = same_eval_metrics = {}
                pending_targets_by_psid = None

            t_iter = time.time() - t_iter0
            print(
                f"[cached] V{model_version}: ITERATION DONE in {t_iter:.1f}s "
                f"(targets {t_targets:.1f}s, train {t_train:.1f}s, same {t_same:.1f}s)",
                flush=True,
            )
            print(
                f"[cached] V{model_version}: train mse={train_metrics.get('mse', float('nan')):.4f} "
                f"p95={train_metrics.get('p95_abs_error', float('nan')):.4f} "
                f"max={train_metrics.get('max_abs_error', float('nan')):.4f} "
                f"| eval mse={eval_metrics.get('mse', float('nan')):.4f} "
                f"p95={eval_metrics.get('p95_abs_error', float('nan')):.4f} "
                f"max={eval_metrics.get('max_abs_error', float('nan')):.4f}",
                flush=True,
            )
            if do_same_version:
                print(
                    f"[cached] V{model_version}: same-version "
                    f"train mse={same_train_metrics.get('mse', float('nan')):.4f} "
                    f"p95={same_train_metrics.get('p95_abs_error', float('nan')):.4f} "
                    f"max={same_train_metrics.get('max_abs_error', float('nan')):.4f} "
                    f"| eval mse={same_eval_metrics.get('mse', float('nan')):.4f} "
                    f"p95={same_eval_metrics.get('p95_abs_error', float('nan')):.4f} "
                    f"max={same_eval_metrics.get('max_abs_error', float('nan')):.4f}",
                    flush=True,
                )

            bootstrap_model = model
            bootstrap_version = int(model_version)
            bootstrap_model_path = iter_model_path_str
            gc.collect()
    finally:
        try:
            state_loader.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> CachedBellmanConfig:
    parser = argparse.ArgumentParser(description="Cached-children Bellman convergence driver")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--child-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-versions", type=int, default=50)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--abs-error-threshold", type=float, default=1.0)
    parser.add_argument("--feature-workers", type=int, default=32)
    parser.add_argument("--feature-chunk-shards", type=int, default=4)
    parser.add_argument("--train-workers", type=int, default=32)
    parser.add_argument("--target-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--predict-processes", type=int, default=32,
                        help="Worker processes for vectorized predict over the child memmap.")
    parser.add_argument("--predict-worker-threads", type=int, default=3,
                        help="BLAS/OpenMP threads per predict worker.")
    parser.add_argument("--same-version-every-n", type=int, default=5,
                        help="Run same-version residual + emit residual CSV every N iterations (and on the final iteration).")
    parser.add_argument("--backend", choices=("cliff_aware", "classical"), default="cliff_aware")
    parser.add_argument("--extra-config-json", default="{}")
    args = parser.parse_args()
    extra_config = json.loads(args.extra_config_json)
    return CachedBellmanConfig(
        dataset_dir=Path(args.dataset_dir).expanduser(),
        child_cache_dir=Path(args.child_cache_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        num_versions=int(args.num_versions),
        eval_ratio=float(args.eval_ratio),
        split_seed=int(args.split_seed),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        abs_error_threshold=float(args.abs_error_threshold),
        feature_workers=int(args.feature_workers),
        feature_chunk_shards=int(args.feature_chunk_shards),
        train_workers=int(args.train_workers),
        target_chunk_size=int(args.target_chunk_size),
        predict_processes=int(args.predict_processes),
        predict_worker_threads=int(args.predict_worker_threads),
        same_version_every_n=int(args.same_version_every_n),
        extra_config=dict(extra_config),
        backend=str(args.backend),
    )


def main() -> None:
    cached_bellman_convergence(parse_args())


if __name__ == "__main__":
    main()
