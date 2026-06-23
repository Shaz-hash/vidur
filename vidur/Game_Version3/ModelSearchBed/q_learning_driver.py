"""Action-conditional Q-learning driver. Pre-computes per-row action features
and runs an FQI loop where each iteration:

  - target_k = reward_k + γ_k * V_{N-1}(s'_k)
  - train Q_N on per-row [parent_features, per_action_features] -> target_k
  - V_N target per parent = max_a Q_N(s, a)
  - train V_N (HGB) on parent_features -> V_N target
  - V_N is saved as the iteration's ``cliff_aware_controller_value.joblib`` so
    the existing CSV evaluators consume it.

This driver re-uses every memmap and feature cache built by
``bellman_convergence_cached``, plus a one-time pass to write
``per_row_action.npy`` and ``parent_features.npy`` (parents indexed by psid).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import logger as bellman_logger
from .self_model_test import (
    RootStateLoader,
    SelfModelTestConfig,
    build_self_model_test_config,
    load_root_records,
)
from .per_action_features import (
    extract_per_action_features,
    per_action_feature_names,
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


@dataclass(frozen=True)
class QConfig:
    cached_run_dir: Path        # existing v4-style run with child_features.npy etc
    output_dir: Path
    num_versions: int = 25
    eval_ratio: float = 0.20
    split_seed: int = 12345
    seed: int = 2027
    abs_error_threshold: float = 1.0
    q_hgb_max_iter: int = 400
    q_hgb_max_leaf_nodes: int = 31
    q_hgb_learning_rate: float = 0.05
    v_hgb_max_iter: int = 250
    v_hgb_max_leaf_nodes: int = 21
    v_hgb_learning_rate: float = 0.05
    bootstrap_shrink: float = 1.0
    polyak_alpha: float = 1.0  # 1.0 = no smoothing
    tail_weight_scale: float = 4.0  # sample weight = 1 + scale * |y|/3 clamped
    extra_config: dict[str, Any] | None = None


def _read_manifest(child_cache_dir: Path) -> list[str]:
    out = []
    with (child_cache_dir / "manifest.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["shard_path"])
    return out


def precompute_per_row_action_features(
    child_cache_dir: Path,
    output_dir: Path,
) -> Path:
    """Walk every shard, parse action_repr, write per_row_action.npy memmap.

    Output shape: (num_rows, len(per_action_feature_names())).
    """

    out_path = output_dir / "per_row_action.npy"
    summary_path = output_dir / "per_row_action.summary.json"
    if out_path.exists() and summary_path.exists():
        print(f"[qlearn] per-row action features already present: {out_path}", flush=True)
        return out_path

    import torch

    shards = _read_manifest(child_cache_dir)
    summary = json.loads((child_cache_dir / "summary.json").read_text())
    n_total = int(summary.get("num_canonical_transitions", summary.get("num_transitions", 0)))
    if n_total <= 0:
        raise RuntimeError("could not determine row count from summary.json")
    D = len(per_action_feature_names())
    print(f"[qlearn] allocating per-row action memmap shape=({n_total}, {D})", flush=True)
    capacity = int(n_total * 1.02) + 1024
    output_dir.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(capacity, D)
    )

    cursor = 0
    bad: list[tuple[str, str]] = []
    t0 = time.time()
    for i, shard in enumerate(shards):
        full = child_cache_dir / shard
        try:
            recs = torch.load(full, map_location="cpu", weights_only=False)
        except Exception as exc:
            bad.append((shard, repr(exc)))
            continue
        block: list[list[float]] = []
        for r in recs:
            if not bool(r.get("is_valid", False)):
                continue
            try:
                feats = extract_per_action_features(str(r.get("action_repr", "")))
            except Exception:
                feats = [0.0] * D
            block.append(feats)
        if not block:
            continue
        arr = np.asarray(block, dtype=np.float32)
        end = cursor + arr.shape[0]
        if end > capacity:
            raise RuntimeError(f"per_row_action capacity overflow {end}>{capacity}")
        mm[cursor:end] = arr
        cursor = end
        if (i + 1) % 200 == 0 or (i + 1) == len(shards):
            rate = cursor / max(1.0, time.time() - t0)
            print(
                f"[qlearn] per-row actions: shard {i+1}/{len(shards)}, "
                f"rows={cursor}/{n_total}, rate={rate:,.0f} rows/s, bad={len(bad)}",
                flush=True,
            )

    if cursor < capacity:
        del mm
        full_arr = np.load(out_path, mmap_mode="r")
        trim = np.lib.format.open_memmap(
            out_path.with_suffix(".tmp.npy"),
            mode="w+", dtype=np.float32, shape=(cursor, D),
        )
        trim[:] = full_arr[:cursor]
        del trim
        del full_arr
        out_path.with_suffix(".tmp.npy").replace(out_path)

    summary_path.write_text(json.dumps({
        "num_rows": int(cursor),
        "feature_dim": int(D),
        "feature_names": per_action_feature_names(),
        "bad_shards": bad[:64],
    }, indent=2))
    print(f"[qlearn] per-row action features done: rows={cursor}, D={D}", flush=True)
    return out_path


def precompute_parent_features_indexed(
    cached_run_dir: Path,
    output_dir: Path,
    n_parents: int,
) -> Path:
    """Build parent_features.npy of shape (n_parents, D) indexed by parent_state_id 0..n.

    The cached driver stored per-split parent feature matrices in the
    _cliff_feature_cache (for train and eval splits separately). Easier to
    re-use the v4 child features memmap *for parents*: parent index i is
    represented in the cliff feature cache via fingerprinted blobs. To keep
    this self-contained we instead pull rows out of the existing per-split
    matrices using the split_indices file.
    """

    out_path = output_dir / "parent_features.npy"
    if out_path.exists():
        print(f"[qlearn] parent_features already present: {out_path}", flush=True)
        return out_path

    split_idx_path = cached_run_dir / "split_indices.json"
    if not split_idx_path.exists():
        raise FileNotFoundError(split_idx_path)
    si = json.loads(split_idx_path.read_text())
    train_idx = list(si["train_indices"])
    eval_idx = list(si["eval_indices"])

    # Find the on-disk feature caches under the cached run.
    cache_dir = cached_run_dir / "_cliff_feature_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)
    train_npy = sorted(cache_dir.glob("*_train_*.npy"))
    eval_npy = sorted(cache_dir.glob("*_eval_*.npy"))
    if not train_npy or not eval_npy:
        raise FileNotFoundError(f"cliff feature .npy files not in {cache_dir}")
    train_mat = np.load(train_npy[0], mmap_mode="r")
    eval_mat = np.load(eval_npy[0], mmap_mode="r")
    if train_mat.shape[1] != eval_mat.shape[1]:
        raise ValueError(f"train/eval feature dim mismatch: {train_mat.shape[1]} vs {eval_mat.shape[1]}")
    D = int(train_mat.shape[1])
    print(f"[qlearn] indexing parent_features: train={len(train_idx)}, eval={len(eval_idx)}, D={D}", flush=True)
    if len(train_idx) != train_mat.shape[0] or len(eval_idx) != eval_mat.shape[0]:
        raise ValueError("split indices length mismatch with feature matrices")

    out_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_parents, D))
    for k, psid in enumerate(train_idx):
        out_mm[int(psid)] = train_mat[k]
    for k, psid in enumerate(eval_idx):
        out_mm[int(psid)] = eval_mat[k]
    del out_mm
    print(f"[qlearn] parent_features done at {out_path}", flush=True)
    return out_path


class PolyakEnsemble:
    """Simple ensemble that averages predictions of two regressors.

    Provides a `predict_matrix` and `predict` that do alpha * new + (1-alpha) * prev.
    Saved/loaded via joblib.
    """

    def __init__(self, new_model: Any, prev_model: Any | None, alpha: float = 0.5):
        self.new_model = new_model
        self.prev_model = prev_model
        self.alpha = float(alpha)

    def predict(self, X):
        a = self.alpha
        new = np.asarray(self.new_model.predict(X), dtype=np.float64)
        if self.prev_model is None:
            out = new
        else:
            prev = np.asarray(self.prev_model.predict(X), dtype=np.float64)
            out = a * new + (1.0 - a) * prev
        return np.minimum(out, 0.0)

    def predict_matrix(self, X):
        return self.predict(X)


def _segment_max(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    if len(offsets) <= 1:
        return np.empty((0,), dtype=values.dtype)
    starts = offsets[:-1]
    out = np.maximum.reduceat(values, starts)
    seg_lens = offsets[1:] - offsets[:-1]
    if np.any(seg_lens <= 0):
        out = out.copy()
        out[seg_lens <= 0] = -np.inf
    return out


def build_per_row_input_matrix(
    *,
    parent_features: np.ndarray,         # (P, D_p), indexed by parent_state_id
    per_row_action: np.ndarray,          # (N, D_pa)
    parent_state_id_per_row: np.ndarray, # (N,)
    out_path: Path,
) -> Path:
    """Materialize the joined (N, D_p + D_pa) matrix on disk so HGB can stream it.

    We do this once at the start of training and reuse across iterations.
    """

    if out_path.exists():
        print(f"[qlearn] joined per-row input already present: {out_path}", flush=True)
        return out_path

    P, Dp = parent_features.shape
    N, Dpa = per_row_action.shape
    print(f"[qlearn] joining per-row input: parents={P} rows={N} D=Dp{Dp}+Dpa{Dpa}", flush=True)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, Dp + Dpa))
    chunk = 1_000_000
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        ids = parent_state_id_per_row[s:e].astype(np.int64)
        mm[s:e, :Dp] = parent_features[ids]
        mm[s:e, Dp:] = per_row_action[s:e]
        if (s // chunk) % 4 == 0:
            print(f"[qlearn] joined rows {s}..{e}", flush=True)
    del mm
    return out_path


def q_learning_run(cfg: QConfig) -> None:
    """Main FQI loop."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Discover the cached run we're reusing ----------
    feat_path = cfg.cached_run_dir / "child_features.npy"
    meta_path = cfg.cached_run_dir / "child_meta.npy"
    split_path = cfg.cached_run_dir / "split_indices.json"
    parent_idx_path = cfg.cached_run_dir / "parent_index.npz"
    if not feat_path.exists() or not meta_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            f"missing cached artifacts in {cfg.cached_run_dir}: "
            f"need child_features.npy, child_meta.npy, split_indices.json"
        )

    si = json.loads(split_path.read_text())
    train_indices = [int(x) for x in si["train_indices"]]
    eval_indices = [int(x) for x in si["eval_indices"]]
    n_parents = max(max(train_indices), max(eval_indices)) + 1
    print(f"[qlearn] n_parents={n_parents}, train={len(train_indices)}, eval={len(eval_indices)}", flush=True)

    # Per-row action features
    per_row_action_path = precompute_per_row_action_features(
        child_cache_dir=cfg.cached_run_dir.parent.parent
            / "model_search_roots_controller_350k_abs1_ratio40_child_transitions_recycled",
        output_dir=cfg.output_dir,
    )

    # Parent feature matrix indexed by parent_state_id
    parent_feat_path = precompute_parent_features_indexed(
        cached_run_dir=cfg.cached_run_dir,
        output_dir=cfg.output_dir,
        n_parents=n_parents,
    )

    # Load metadata
    meta = np.load(meta_path, mmap_mode="r")
    parent_state_id_per_row = meta[:, 0].astype(np.int64)
    rewards_per_row = meta[:, 3].astype(np.float32)
    discounts_per_row = meta[:, 4].astype(np.float32)
    n_rows = int(meta.shape[0])
    print(f"[qlearn] n_rows={n_rows}", flush=True)

    # parent_index for V max-Q aggregation (sorted by parent_state_id)
    z = np.load(parent_idx_path)
    order = z["order"].astype(np.int64)
    parent_ids = z["parent_ids"].astype(np.int64)
    offsets = z["offsets"].astype(np.int64)
    rewards_sorted = z["rewards"].astype(np.float32)
    discounts_sorted = z["discounts"].astype(np.float32)

    # Per-row joined input matrix (Dp + Dpa)
    per_row_action = np.load(per_row_action_path, mmap_mode="r")
    parent_features = np.load(parent_feat_path, mmap_mode="r")
    Dp = int(parent_features.shape[1])
    Dpa = int(per_row_action.shape[1])
    print(f"[qlearn] feature dims: parent={Dp}, per-action={Dpa}, joined={Dp + Dpa}", flush=True)

    joined_path = cfg.output_dir / "per_row_input.npy"
    build_per_row_input_matrix(
        parent_features=parent_features,
        per_row_action=per_row_action,
        parent_state_id_per_row=parent_state_id_per_row,
        out_path=joined_path,
    )
    X_per_row = np.load(joined_path, mmap_mode="r")

    # ---------- FQI loop ----------
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib

    bootstrap_shrink = float(cfg.bootstrap_shrink)
    v_prev_path: str | None = None  # joblib of state-only V from previous iter

    # Q-only mode: skip V regressor entirely; bootstrap V(s') via act_max_reward at child.
    extra = dict(cfg.extra_config or {})
    q_only_mode = bool(extra.get("q_only_bootstrap", False))
    # Locate act_max_reward column in the child feature memmap.
    act_max_col_q = int(extra.get("act_max_reward_col", -1))
    if q_only_mode and act_max_col_q < 0:
        # try resolve from child_features.names.json
        try:
            names_path = cfg.cached_run_dir / "child_features.names.json"
            child_names = json.loads(names_path.read_text())
            act_max_col_q = int(child_names.index("act_max_reward"))
        except Exception:
            act_max_col_q = -1
    if q_only_mode:
        if act_max_col_q < 0:
            raise RuntimeError("q_only_bootstrap requires act_max_reward column resolvable")
        print(f"[qlearn] Q-only bootstrap mode ON, act_max_reward col={act_max_col_q}", flush=True)

    for vi in range(1, int(cfg.num_versions) + 1):
        t_iter = time.time()
        model_dir = cfg.output_dir / f"Model_Version{vi}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Compute targets per row ----------
        t0 = time.time()
        if q_only_mode:
            # V_proxy at child = act_max_reward(child) (analytical, simulator-derived).
            # Then iterate by adding γ × Q_max-aggregation across iterations:
            # V_{N-1}(s') = max_a' Q_{N-1}(s', a') would require grandchildren we
            # don't have, so we approximate using a bounded chain:
            # target_iter1 = r + γ × act_max(child)
            # target_iter2 = r + γ × max(act_max(child), Q_iter1_at_child_state) — but
            # Q at child_state requires per-action features for the child, which we
            # don't have; so we use act_max which is a fixed analytical lower bound.
            # This converges Q to a 2-step lookahead value, much tighter than
            # state-only V regression.
            X_child_mm = np.load(feat_path, mmap_mode="r")
            v_at_child = np.empty(n_rows, dtype=np.float32)
            chunk = 1_000_000
            for s in range(0, n_rows, chunk):
                e = min(n_rows, s + chunk)
                v_at_child[s:e] = X_child_mm[s:e, act_max_col_q].astype(np.float32, copy=False)
            del X_child_mm
            np.minimum(v_at_child, 0.0, out=v_at_child)
            print(f"[qlearn] V{vi}: V_at_child = act_max_reward(child), no model bootstrap", flush=True)
        elif vi == 1 or v_prev_path is None:
            v_at_child = np.zeros(n_rows, dtype=np.float32)
        else:
            v_prev = joblib.load(v_prev_path)
            X_child_mm = np.load(feat_path, mmap_mode="r")
            v_at_child = np.empty(n_rows, dtype=np.float32)
            chunk = 1_000_000
            for s in range(0, n_rows, chunk):
                e = min(n_rows, s + chunk)
                block = np.ascontiguousarray(X_child_mm[s:e])
                v_at_child[s:e] = v_prev.predict(block).astype(np.float32, copy=False)
                if (s // chunk) % 4 == 0:
                    print(f"[qlearn] V_prev predict rows {s}..{e}", flush=True)
            np.minimum(v_at_child, 0.0, out=v_at_child)
            del X_child_mm
            del v_prev
        targets_per_row = rewards_per_row + (bootstrap_shrink * discounts_per_row) * v_at_child
        np.minimum(targets_per_row, 0.0, out=targets_per_row)
        # Train/eval split per row by parent_state_id
        train_set = set(train_indices)
        is_train_row = np.fromiter(
            (int(p) in train_set for p in parent_state_id_per_row),
            dtype=bool, count=n_rows
        )
        train_row_idx = np.where(is_train_row)[0]
        eval_row_idx = np.where(~is_train_row)[0]
        print(
            f"[qlearn] V{vi}: target compute done in {time.time()-t0:.1f}s, "
            f"train_rows={len(train_row_idx)}, eval_rows={len(eval_row_idx)}",
            flush=True,
        )

        # ---------- Train Q_N on per-row features ----------
        t0 = time.time()
        # Materialize the training matrix in memory: Q_train_X
        # 14M rows × 523 dims × 4 bytes = ~30 GB. Acceptable on 185 GB box.
        print(f"[qlearn] V{vi}: materializing Q train matrix ({len(train_row_idx)} rows × {Dp+Dpa})", flush=True)
        Q_train_X = X_per_row[train_row_idx].astype(np.float32, copy=False)
        Q_train_y = targets_per_row[train_row_idx].astype(np.float32, copy=False)
        Q_eval_X = X_per_row[eval_row_idx].astype(np.float32, copy=False)
        Q_eval_y = targets_per_row[eval_row_idx].astype(np.float32, copy=False)
        print(f"[qlearn] V{vi}: Q_train shape={Q_train_X.shape}", flush=True)

        q_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=int(cfg.q_hgb_max_iter),
            max_leaf_nodes=int(cfg.q_hgb_max_leaf_nodes),
            learning_rate=float(cfg.q_hgb_learning_rate),
            max_bins=255,
            early_stopping=False,
            random_state=int(cfg.seed) + vi,
        )
        q_model.fit(Q_train_X, Q_train_y)
        gc.collect()
        print(f"[qlearn] V{vi}: Q trained in {time.time()-t0:.1f}s", flush=True)

        # ---------- Q evaluation per row + V_N target = max_a Q ----------
        t0 = time.time()
        q_train_pred = q_model.predict(Q_train_X)
        q_eval_pred = q_model.predict(Q_eval_X)
        # Train/eval Q metrics
        q_train_err = np.abs(q_train_pred - Q_train_y)
        q_eval_err = np.abs(q_eval_pred - Q_eval_y)
        q_train_metrics = {
            "mse": float(np.mean(q_train_err ** 2)),
            "rmse": float(math.sqrt(float(np.mean(q_train_err ** 2)))),
            "mae": float(np.mean(q_train_err)),
            "p50_abs_error": float(np.percentile(q_train_err, 50)),
            "p95_abs_error": float(np.percentile(q_train_err, 95)),
            "max_abs_error": float(np.max(q_train_err)),
        }
        q_eval_metrics = {
            "mse": float(np.mean(q_eval_err ** 2)),
            "rmse": float(math.sqrt(float(np.mean(q_eval_err ** 2)))),
            "mae": float(np.mean(q_eval_err)),
            "p50_abs_error": float(np.percentile(q_eval_err, 50)),
            "p95_abs_error": float(np.percentile(q_eval_err, 95)),
            "max_abs_error": float(np.max(q_eval_err)),
        }
        print(
            f"[qlearn] V{vi}: Q eval mse={q_eval_metrics['mse']:.5f} "
            f"p95={q_eval_metrics['p95_abs_error']:.4f} max={q_eval_metrics['max_abs_error']:.4f}",
            flush=True,
        )
        del Q_train_X, Q_eval_X
        gc.collect()

        # Predict Q on every row to compute per-parent V_N target = max_a Q.
        all_q = np.empty(n_rows, dtype=np.float32)
        chunk = 1_000_000
        for s in range(0, n_rows, chunk):
            e = min(n_rows, s + chunk)
            block = np.ascontiguousarray(X_per_row[s:e])
            all_q[s:e] = q_model.predict(block).astype(np.float32, copy=False)
        np.minimum(all_q, 0.0, out=all_q)
        # max_a Q per parent: reorder by sorted index, then segment_max
        all_q_sorted = all_q[order]
        v_n_target_per_parent = _segment_max(all_q_sorted, offsets)
        del all_q, all_q_sorted
        gc.collect()
        print(f"[qlearn] V{vi}: V_target compute done in {time.time()-t0:.1f}s", flush=True)

        # ---------- Train V_N on parent features → V_target ----------
        t0 = time.time()
        # Train/eval split for V (parents)
        train_psids = np.asarray(train_indices, dtype=np.int64)
        eval_psids = np.asarray(eval_indices, dtype=np.int64)
        # Map parent_id -> v_n_target
        v_target_by_psid = {int(parent_ids[i]): float(v_n_target_per_parent[i]) for i in range(parent_ids.shape[0])}
        # Build V train/eval arrays
        V_train_X = parent_features[train_psids]
        V_train_y = np.asarray([v_target_by_psid.get(int(p), 0.0) for p in train_psids], dtype=np.float32)
        V_eval_X = parent_features[eval_psids]
        V_eval_y = np.asarray([v_target_by_psid.get(int(p), 0.0) for p in eval_psids], dtype=np.float32)

        v_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=int(cfg.v_hgb_max_iter),
            max_leaf_nodes=int(cfg.v_hgb_max_leaf_nodes),
            learning_rate=float(cfg.v_hgb_learning_rate),
            max_bins=255,
            early_stopping=False,
            random_state=int(cfg.seed) + vi + 11,
        )
        # Tail-aware sample weights: rows with larger |target| get more weight
        # so the regressor commits to fitting them, not the bulk-zero region.
        tail_w = 1.0 + float(cfg.tail_weight_scale) * np.minimum(1.0, np.abs(V_train_y) / 3.0)
        v_model.fit(V_train_X, V_train_y, sample_weight=tail_w.astype(np.float32))
        # Polyak-wrap with previous V model to dampen iter-to-iter noise.
        polyak_alpha = float(cfg.polyak_alpha)
        v_prev_for_polyak = None
        if vi >= 2 and v_prev_path is not None and polyak_alpha < 1.0:
            try:
                v_prev_for_polyak = joblib.load(v_prev_path)
            except Exception as e:
                print(f"[qlearn] V{vi}: Polyak prev load failed ({e!r})", flush=True)
        v_polyak = PolyakEnsemble(v_model, v_prev_for_polyak, alpha=polyak_alpha)
        v_train_pred = v_polyak.predict(V_train_X)
        v_eval_pred = v_polyak.predict(V_eval_X)
        v_train_err = np.abs(v_train_pred - V_train_y)
        v_eval_err = np.abs(v_eval_pred - V_eval_y)
        v_train_metrics = {
            "mse": float(np.mean(v_train_err ** 2)),
            "rmse": float(math.sqrt(float(np.mean(v_train_err ** 2)))),
            "mae": float(np.mean(v_train_err)),
            "p50_abs_error": float(np.percentile(v_train_err, 50)),
            "p95_abs_error": float(np.percentile(v_train_err, 95)),
            "max_abs_error": float(np.max(v_train_err)),
        }
        v_eval_metrics = {
            "mse": float(np.mean(v_eval_err ** 2)),
            "rmse": float(math.sqrt(float(np.mean(v_eval_err ** 2)))),
            "mae": float(np.mean(v_eval_err)),
            "p50_abs_error": float(np.percentile(v_eval_err, 50)),
            "p95_abs_error": float(np.percentile(v_eval_err, 95)),
            "max_abs_error": float(np.max(v_eval_err)),
        }
        print(
            f"[qlearn] V{vi}: V eval mse={v_eval_metrics['mse']:.5f} "
            f"p95={v_eval_metrics['p95_abs_error']:.4f} max={v_eval_metrics['max_abs_error']:.4f}",
            flush=True,
        )
        # Save the Polyak-wrapped V model so next iter bootstraps from a smoothed
        # function. This is what other code paths see at predict time.
        v_model_path = model_dir / "v_model.joblib"
        joblib.dump(v_polyak, v_model_path)
        v_prev_path = str(v_model_path)
        # Also save Q model
        q_model_path = model_dir / "q_model.joblib"
        joblib.dump(q_model, q_model_path)
        print(
            f"[qlearn] V{vi}: trained Q+V in {time.time()-t0:.1f}s, saved {v_model_path}",
            flush=True,
        )

        # ---------- CSVs ----------
        with (model_dir / "v_train_results.csv").open("w") as f:
            f.write("sample_number,model_prediction,true_MCTS_DNN_value\n")
            for i, p in enumerate(train_psids):
                f.write(f"{int(p)},{float(v_train_pred[i]):.6f},{float(V_train_y[i]):.6f}\n")
        with (model_dir / "v_eval_results.csv").open("w") as f:
            f.write("sample_number,model_prediction,true_MCTS_DNN_value\n")
            for i, p in enumerate(eval_psids):
                f.write(f"{int(p)},{float(v_eval_pred[i]):.6f},{float(V_eval_y[i]):.6f}\n")

        # Same-version: V_N(s) vs T[V_N](s). Since V_target = max_a Q, that IS T[V_N]
        # for one-step Bellman. So same-version max equals max(V_N(s) - V_target(s))
        # which is exactly v_eval_metrics['max_abs_error']. We log it for clarity.
        bellman_logger.write_summary_csv(
            cfg.output_dir / f"version_{vi-1}_to_{vi}.csv",
            [
                {"source_model_version": vi - 1, "target_model_version": vi, "split": "train",
                 "model_version": vi, **v_train_metrics, **{f"q_{k}": v for k, v in q_train_metrics.items()}},
                {"source_model_version": vi - 1, "target_model_version": vi, "split": "eval",
                 "model_version": vi, **v_eval_metrics, **{f"q_{k}": v for k, v in q_eval_metrics.items()}},
            ],
        )

        # Save V_N as the iteration's "value model" using the cliff_aware filename
        # so existing arena tooling can pick it up.
        try:
            joblib.dump(v_model, model_dir / "cliff_aware_controller_value.joblib")
        except Exception:
            pass

        print(
            f"[qlearn] V{vi}: ITERATION DONE in {time.time()-t_iter:.1f}s | "
            f"V eval p95={v_eval_metrics['p95_abs_error']:.4f} max={v_eval_metrics['max_abs_error']:.4f} | "
            f"Q eval p95={q_eval_metrics['p95_abs_error']:.4f} max={q_eval_metrics['max_abs_error']:.4f}",
            flush=True,
        )


def parse_args() -> QConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--cached-run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-versions", type=int, default=25)
    p.add_argument("--bootstrap-shrink", type=float, default=1.0)
    p.add_argument("--q-hgb-max-iter", type=int, default=400)
    p.add_argument("--q-hgb-max-leaf-nodes", type=int, default=31)
    p.add_argument("--q-hgb-lr", type=float, default=0.05)
    p.add_argument("--v-hgb-max-iter", type=int, default=250)
    p.add_argument("--v-hgb-max-leaf-nodes", type=int, default=21)
    p.add_argument("--v-hgb-lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--polyak-alpha", type=float, default=1.0,
                   help="V smoothing across iterations: 1.0 = no smoothing.")
    p.add_argument("--tail-weight-scale", type=float, default=4.0,
                   help="V sample weight = 1 + scale*min(1, |y|/3).")
    args = p.parse_args()
    return QConfig(
        cached_run_dir=Path(args.cached_run_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        num_versions=int(args.num_versions),
        bootstrap_shrink=float(args.bootstrap_shrink),
        q_hgb_max_iter=int(args.q_hgb_max_iter),
        q_hgb_max_leaf_nodes=int(args.q_hgb_max_leaf_nodes),
        q_hgb_learning_rate=float(args.q_hgb_lr),
        v_hgb_max_iter=int(args.v_hgb_max_iter),
        v_hgb_max_leaf_nodes=int(args.v_hgb_max_leaf_nodes),
        v_hgb_learning_rate=float(args.v_hgb_lr),
        seed=int(args.seed),
        polyak_alpha=float(args.polyak_alpha),
        tail_weight_scale=float(args.tail_weight_scale),
    )


def main() -> None:
    q_learning_run(parse_args())


if __name__ == "__main__":
    main()
