"""
Bellman value iteration for the v4 (224-d state-local) classical model.

For each iteration n in [start..end]:
  1. If n == 1: target_1(parent) = max_a min_b (reward_a + 0) = act_max_reward,
     computed from children (no bootstrap). Since V_0=0 and adversary actions do
     not add reward/time, all adversary branches under a controller action have
     the same immediate value.
  2. Else: load V_{n-1} (joblib), predict V_{n-1}(child) on every (valid) child, then
     target_n(parent) = max_a min_b [
       reward_a + gamma_a * clip_max(V_{n-1}(child_{a,b}), 0)
     ] over is_valid==1 controller/adversary children in both child caches (D1 + D2).
  3. Train HGB on (parent_features, target) with sample_weight = 1 + tail_alpha * |target|.
  4. Save V_n joblib + train_results.csv + eval_results.csv (forward, n-1 -> n).
  5. **Same-version CSV optimization**: target_{n+1}(parent) computed at the START of the
     next iteration is identical to same_version_target_n(parent) (both use V_n on the
     child side). So at the start of iter n+1, after we compute target_{n+1}, we have all
     the data needed to write Model_Version{n}/{train,eval}_results_same_version.csv:
         y_true = target_{n+1}  (just computed for the next iter's training)
         y_pred = V_n(parent)   (cached in memory from iter n's forward predictions)
     This avoids a second per-iter pass over the 30M children.
  6. Save metadata.json with all metrics.

Outputs (under <out_dir>/Model_Version{n}/):
  model.joblib
  train_results.csv, eval_results.csv
  train_results_same_version.csv, eval_results_same_version.csv (written one iter late)
  metadata.json (forward metrics; same-version metrics added at iter n+1)

Children get clamped at <=0 before bootstrap (mirrors the no-positive-V invariant we
already enforce on parent predictions).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _set_threads(n: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass


def _hgb_n_params(model) -> int:
    total = 0
    for est in model._predictors:
        for tree in est:
            try:
                total += int(np.sum(tree.nodes["is_leaf"])) * 3
            except Exception:
                total += int(tree.nodes.size) * 3
    return total


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "n": int(y_true.size),
        "mse": float(np.mean(err * err)),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "mae": float(np.mean(abs_err)),
        "p50_abs": float(np.quantile(abs_err, 0.5)),
        "p90_abs": float(np.quantile(abs_err, 0.9)),
        "p95_abs": float(np.quantile(abs_err, 0.95)),
        "p99_abs": float(np.quantile(abs_err, 0.99)),
        "max_abs": float(abs_err.max()),
        "n_over_05": int(np.sum(abs_err > 0.5)),
        "n_over_1": int(np.sum(abs_err > 1.0)),
        "n_over_2": int(np.sum(abs_err > 2.0)),
        "frac_over_05": float(np.mean(abs_err > 0.5)),
    }


def _write_predictions_csv(path: Path, idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    abs_err = np.abs(y_pred - y_true)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["psid", "y_true", "y_pred", "abs_err"])
        for i in range(len(idx)):
            w.writerow([int(idx[i]), float(y_true[i]), float(y_pred[i]), float(abs_err[i])])


def _load_child_cache(cache_dir: Path):
    """Load Adv child cache arrays.

    ``parent_index.npz`` stores metadata in original child-row order plus an
    ``order`` permutation that sorts rows by parent id. The Bellman backup uses
    that sorted view to do max_controller min_adversary aggregation per parent.
    """
    feats = np.load(cache_dir / "child_features.npy", mmap_mode="r")
    pi = np.load(cache_dir / "parent_index.npz")
    required = (
        "controller_action_indices",
        "adversary_action_indices",
    )
    missing = [name for name in required if name not in pi.files]
    if missing:
        raise RuntimeError(
            f"{cache_dir / 'parent_index.npz'} is not an Adv child index; "
            f"missing keys={missing}. Rebuild with build_child_features_v4Adv.py"
        )
    return {
        "features": feats,
        "order": pi["order"],
        "parent_ids": pi["parent_ids"],
        "offsets": pi["offsets"],
        "rewards": pi["rewards"],
        "discounts": pi["discounts"],
        "is_valid": pi["is_valid"],
        "controller_action_indices": pi["controller_action_indices"],
        "adversary_action_indices": pi["adversary_action_indices"],
    }


def _hier_predict_block(hier, X_block: np.ndarray) -> np.ndarray:
    """Gated predict for a V4HierWrapper-shaped object: 0 if p_pos>tau else min(r,0)."""
    p_pos = hier.cls.predict_proba(X_block)[:, 1]
    r = hier.reg.predict(X_block)
    yp = np.where(p_pos > float(hier.tau), 0.0, np.minimum(r, 0.0))
    return yp.astype(np.float32)


def _is_hier(obj) -> bool:
    return hasattr(obj, "cls") and hasattr(obj, "reg") and hasattr(obj, "tau")


def _predict_chunked(model, X: np.ndarray, chunk: int = 524288) -> np.ndarray:
    """Predict V on a (mmap-able) feature array in chunks, clamping to <= 0.
    Handles either a sklearn HGBRegressor or a V4HierWrapper-shaped object.
    """
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    hier = _is_hier(model)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        block = np.asarray(X[s:e])  # bring into RAM
        if hier:
            yp = _hier_predict_block(model, block)
        else:
            yp = model.predict(block).astype(np.float32)
            np.minimum(yp, 0.0, out=yp)
        out[s:e] = yp
    return out


def _bellman_minimax_per_parent(
    cache: dict,
    child_v: np.ndarray,
    parent_id_to_pos_offset: dict,
    parent_id_to_pos_count: dict,
    parent_target: np.ndarray,
) -> None:
    """
    For every parent in this Adv cache, compute:

        max_controller_action min_adversary_action [
            reward + discount * V(final_child)
        ]

    V(child) is already clamped to <= 0 by the predictor. Reward/discount are
    attached to the controller edge parent -> intermediate; the adversary action
    only changes the final child state used for bootstrap.

    The caller starts parent_target = -inf and we max-update it. This preserves
    the existing D1/D2 merge behavior while making each cache internally minimax.
    """
    _ = parent_id_to_pos_count
    order = cache["order"]
    rewards_sorted = cache["rewards"][order]
    disc_sorted = cache["discounts"][order]
    is_valid_sorted = cache["is_valid"][order]
    ctrl_sorted = cache["controller_action_indices"][order]
    cv_sorted = child_v[order]

    candidates = rewards_sorted + disc_sorted * cv_sorted
    np.minimum(candidates, 0.0, out=candidates)

    parent_ids = cache["parent_ids"]
    offsets = cache["offsets"]
    for i, pid in enumerate(parent_ids):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        if s == e:
            continue
        slot = parent_id_to_pos_offset.get(int(pid))
        if slot is None:
            continue

        valid = is_valid_sorted[s:e]
        if not np.any(valid):
            continue

        seg_candidates = candidates[s:e][valid]
        seg_ctrl = ctrl_sorted[s:e][valid]

        controller_values: list[float] = []
        for ctrl_action in np.unique(seg_ctrl):
            ctrl_mask = seg_ctrl == ctrl_action
            if not np.any(ctrl_mask):
                continue
            controller_values.append(float(np.min(seg_candidates[ctrl_mask])))

        if not controller_values:
            continue

        seg_max = float(np.max(np.asarray(controller_values, dtype=np.float32)))
        if seg_max > parent_target[slot]:
            parent_target[slot] = seg_max


def _validate_iter1_targets(
    *,
    out_dir: Path,
    stored_targets: np.ndarray,
    computed_targets: np.ndarray,
    atol: float,
) -> None:
    """Fail fast if repaired parent labels disagree with child-cache immediate rewards."""

    diff = computed_targets.astype(np.float32) - stored_targets.astype(np.float32)
    abs_diff = np.abs(diff)
    bad = abs_diff > float(atol)
    worst_order = np.argsort(abs_diff)[-20:][::-1]
    report = {
        "atol": float(atol),
        "n": int(stored_targets.size),
        "n_bad": int(np.sum(bad)),
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "p50_abs_diff": float(np.quantile(abs_diff, 0.50)) if abs_diff.size else 0.0,
        "p90_abs_diff": float(np.quantile(abs_diff, 0.90)) if abs_diff.size else 0.0,
        "p95_abs_diff": float(np.quantile(abs_diff, 0.95)) if abs_diff.size else 0.0,
        "p99_abs_diff": float(np.quantile(abs_diff, 0.99)) if abs_diff.size else 0.0,
        "worst": [
            {
                "global_index": int(i),
                "stored_target": float(stored_targets[i]),
                "max_child_reward": float(computed_targets[i]),
                "abs_diff": float(abs_diff[i]),
            }
            for i in worst_order[:20]
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v1_target_consistency.json").write_text(json.dumps(report, indent=2))
    print(f"[bell] V1 target consistency: {report}", flush=True)
    if report["n_bad"] > 0:
        raise RuntimeError(
            "child-cache rewards are inconsistent with repaired parent targets; "
            f"n_bad={report['n_bad']} max_abs_diff={report['max_abs_diff']:.6f}. "
            f"See {out_dir / 'v1_target_consistency.json'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-d1", required=True)
    parser.add_argument("--targets-d1", required=True)
    parser.add_argument("--features-d2", required=True)
    parser.add_argument("--targets-d2", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--child-cache-d1", required=True)
    parser.add_argument("--child-cache-d2", required=True)
    parser.add_argument("--n-d1", type=int, default=292713,
                        help="how many parent rows in D1 (used to map child parent_state_id -> global row)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-iter", type=int, default=1)
    parser.add_argument("--end-iter", type=int, default=50)
    parser.add_argument("--max-leaf-nodes", type=int, default=47)
    parser.add_argument("--max-iter", type=int, default=850)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--tail-alpha", type=float, default=2.0)
    parser.add_argument("--threads-per", type=int, default=48)
    parser.add_argument("--predict-chunk", type=int, default=524288)
    parser.add_argument("--init-v0-zero", action="store_true",
                        help="Force iter 1 to use V_0=0 bootstrap (i.e. target_1 = max_a reward_a). "
                             "If false and start-iter>1, will load V_{start-1} from out-dir.")
    parser.add_argument("--v1-target-consistency-atol", type=float, default=1e-4,
                        help="Fail before training if max_a child_reward differs from stored parent targets by more than this.")
    parser.add_argument("--skip-v1-target-consistency-check", action="store_true",
                        help="Skip the repaired-parent-target vs child-cache-reward preflight check.")
    args = parser.parse_args()

    _set_threads(args.threads_per)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bell] loading parent features…", flush=True)
    f1 = np.load(args.features_d1, mmap_mode="r")
    t1 = np.load(args.targets_d1, mmap_mode="r")
    f2 = np.load(args.features_d2, mmap_mode="r")
    t2 = np.load(args.targets_d2, mmap_mode="r")
    n_d1 = int(f1.shape[0])
    n_d2 = int(f2.shape[0])
    assert n_d1 == args.n_d1, f"n_d1 mismatch: features {n_d1} vs --n-d1 {args.n_d1}"
    print(f"[bell] X_d1={f1.shape} X_d2={f2.shape} (concat in-place)", flush=True)
    X = np.concatenate([np.asarray(f1), np.asarray(f2)], axis=0)
    n_total = X.shape[0]
    print(f"[bell] X={X.shape}", flush=True)

    split = json.loads(Path(args.split_json).read_text())
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    eval_idx = np.asarray(split["eval_indices"], dtype=np.int64)

    print(f"[bell] loading child caches…", flush=True)
    cache_d1 = _load_child_cache(Path(args.child_cache_d1))
    cache_d2 = _load_child_cache(Path(args.child_cache_d2))
    print(f"[bell] d1 children={cache_d1['features'].shape[0]} d2 children={cache_d2['features'].shape[0]}",
          flush=True)

    # parent-id -> global row index map.
    # For D1: parent_id == global row (rows 0..n_d1-1).
    # For D2: parent_id (within d2) == global_row - n_d1 (rows n_d1..n_total-1).
    parent_pos_map_d1 = {int(pid): int(pid) for pid in cache_d1["parent_ids"]}
    parent_pos_map_d2 = {int(pid): int(pid) + n_d1 for pid in cache_d2["parent_ids"]}
    parent_count_d1 = {int(pid): 1 for pid in parent_pos_map_d1}
    parent_count_d2 = {int(pid): 1 for pid in parent_pos_map_d2}

    # ======================================================================
    # Helper to compute (target across all parents) given a callable v_child(features) -> v
    # ======================================================================
    def _compute_targets_bellman(v_child_fn) -> np.ndarray:
        """Returns parent_target shape (n_total,) initialised to 0 for parents that have
        no children (shouldn't happen in our datasets) and to max-Bellman value otherwise.
        Children are clamped via v_child_fn, expected to return arrays already <= 0.
        """
        parent_target = np.full(n_total, -np.inf, dtype=np.float32)

        # D1
        t0 = time.time()
        cv1 = v_child_fn(cache_d1["features"])
        print(f"[bell]   v_child(D1) elapsed={time.time() - t0:.1f}s shape={cv1.shape}", flush=True)
        _bellman_minimax_per_parent(cache_d1, cv1, parent_pos_map_d1, parent_count_d1, parent_target)
        del cv1

        # D2
        t0 = time.time()
        cv2 = v_child_fn(cache_d2["features"])
        print(f"[bell]   v_child(D2) elapsed={time.time() - t0:.1f}s shape={cv2.shape}", flush=True)
        _bellman_minimax_per_parent(cache_d2, cv2, parent_pos_map_d2, parent_count_d2, parent_target)
        del cv2

        # any parent left with -inf (no valid children) -> 0
        n_no_child = int(np.sum(~np.isfinite(parent_target)))
        if n_no_child > 0:
            print(f"[bell]   {n_no_child} parents had no valid child; setting target=0", flush=True)
            parent_target[~np.isfinite(parent_target)] = 0.0
        # safety: clamp to <= 0 (rewards are <=0, V is clamped <=0, so max should be <=0)
        np.minimum(parent_target, 0.0, out=parent_target)
        return parent_target

    precomputed_iter1_target: np.ndarray | None = None
    if not args.skip_v1_target_consistency_check:
        print("[bell] preflight: checking max_a child_reward against stored parent targets", flush=True)

        def v0_for_preflight(features_arr):
            return np.zeros(features_arr.shape[0], dtype=np.float32)

        precomputed_iter1_target = _compute_targets_bellman(v0_for_preflight)
        stored_targets = np.concatenate([np.asarray(t1), np.asarray(t2)], axis=0).astype(np.float32)
        _validate_iter1_targets(
            out_dir=out_dir,
            stored_targets=stored_targets,
            computed_targets=precomputed_iter1_target,
            atol=float(args.v1_target_consistency_atol),
        )

    # ======================================================================
    # Iteration loop
    # Same-version optimization: target_{n+1} (computed at start of iter n+1, using V_n
    # on children) IS the same-version target for V_n. So at iter n+1 we have everything
    # we need to write Model_Version{n}/{train,eval}_results_same_version.csv: the just-
    # computed target_{n+1} as y_true, and V_n(parent) cached from iter n's forward step.
    # We carry V_n's forward parent-predictions across iterations to enable this.
    # ======================================================================
    overall_t0 = time.time()
    iter_summary: list[dict[str, Any]] = []

    # cached forward parent-preds from previous iteration (V_{n-1} on parents).
    # When iter n starts, after computing target_n (which used V_{n-1} on children),
    # we have what we need to write Model_Version{n-1}/*_same_version.csv:
    #   y_true = target_n
    #   y_pred = prev_yp_full  (= V_{n-1}(parent) from iter n-1)
    prev_yp_full: np.ndarray | None = None  # shape (n_total,) or None

    for n in range(args.start_iter, args.end_iter + 1):
        sub = out_dir / f"Model_Version{n}"
        sub.mkdir(parents=True, exist_ok=True)

        print(f"\n[bell] ===== iteration n={n} =====", flush=True)
        it_t0 = time.time()

        # ----- compute target for THIS iteration (uses V_{n-1}) -----
        if n == 1:
            print(f"[bell] iter1 target = max_a r_a (V_0 := 0)", flush=True)
            if precomputed_iter1_target is not None:
                target = precomputed_iter1_target.copy()
            else:
                def v0(features_arr):
                    return np.zeros(features_arr.shape[0], dtype=np.float32)
                target = _compute_targets_bellman(v0)
        else:
            prev_path = out_dir / f"Model_Version{n-1}" / "model.joblib"
            print(f"[bell] iter{n} loading V_{n-1} from {prev_path}", flush=True)
            v_prev = joblib.load(prev_path)
            def v_pred(features_arr, _m=v_prev):
                return _predict_chunked(_m, features_arr, chunk=args.predict_chunk)
            target = _compute_targets_bellman(v_pred)
            del v_prev

        # ----- write SAME-VERSION CSVs for the previous iter (n-1) using just-computed target_n
        if n > args.start_iter and prev_yp_full is not None:
            prev_sub = out_dir / f"Model_Version{n-1}"
            ys_train = target[train_idx]
            ys_eval = target[eval_idx]
            yp_train_prev = prev_yp_full[train_idx]
            yp_eval_prev = prev_yp_full[eval_idx]
            m_train_same_prev = _metrics(ys_train, yp_train_prev)
            m_eval_same_prev = _metrics(ys_eval, yp_eval_prev)
            print(f"[bell] iter{n} same-version for V_{n-1}: train {m_train_same_prev}", flush=True)
            print(f"[bell] iter{n} same-version for V_{n-1}:  eval {m_eval_same_prev}", flush=True)
            _write_predictions_csv(prev_sub / "train_results_same_version.csv", train_idx,
                                   ys_train, yp_train_prev)
            _write_predictions_csv(prev_sub / "eval_results_same_version.csv", eval_idx,
                                   ys_eval, yp_eval_prev)
            # patch metadata.json for V_{n-1} with the now-known same-version metrics
            meta_prev_path = prev_sub / "metadata.json"
            if meta_prev_path.exists():
                meta_prev = json.loads(meta_prev_path.read_text())
                meta_prev["metrics_train_same_version"] = m_train_same_prev
                meta_prev["metrics_eval_same_version"] = m_eval_same_prev
                meta_prev_path.write_text(json.dumps(meta_prev, indent=2))
            # update existing iter_summary entry for V_{n-1}
            for entry in iter_summary:
                if entry.get("iter") == n - 1:
                    entry["metrics_train_same_version"] = m_train_same_prev
                    entry["metrics_eval_same_version"] = m_eval_same_prev
                    break

        # ----- train V_n -----
        y_train = target[train_idx]
        y_eval = target[eval_idx]
        sample_weight = (1.0 + args.tail_alpha * np.abs(y_train)).astype(np.float32)
        est = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            random_state=0,
            early_stopping=False,
        )
        print(f"[bell] iter{n} fitting HGB max_leaves={args.max_leaf_nodes} max_iter={args.max_iter} alpha={args.tail_alpha}",
              flush=True)
        fit_t0 = time.time()
        est.fit(X[train_idx], y_train, sample_weight=sample_weight)
        fit_elapsed = time.time() - fit_t0
        n_params = _hgb_n_params(est)
        print(f"[bell] iter{n} fit_elapsed={fit_elapsed:.1f}s n_params={n_params:,}", flush=True)

        # ----- forward CSVs (n-1 -> n): how well V_n fits target_n -----
        # Predict on the FULL parent set (so we can carry yp into next iter for same-version)
        yp_full = np.minimum(_predict_chunked(est, X, chunk=args.predict_chunk), 0.0)
        yp_train = yp_full[train_idx]
        yp_eval = yp_full[eval_idx]
        m_train = _metrics(y_train, yp_train)
        m_eval = _metrics(y_eval, yp_eval)
        print(f"[bell] iter{n} FORWARD train: {m_train}", flush=True)
        print(f"[bell] iter{n} FORWARD  eval: {m_eval}", flush=True)
        _write_predictions_csv(sub / "train_results.csv", train_idx, y_train, yp_train)
        _write_predictions_csv(sub / "eval_results.csv", eval_idx, y_eval, yp_eval)

        # save V_n
        joblib.dump(est, sub / "model.joblib", compress=3)
        # roll forward
        prev_yp_full = yp_full

        meta = {
            "iter": n,
            "n_params_estimate": int(n_params),
            "max_leaf_nodes": args.max_leaf_nodes,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
            "tail_alpha": args.tail_alpha,
            "fit_elapsed_s": fit_elapsed,
            "iter_total_elapsed_s": time.time() - it_t0,
            "metrics_train_forward": m_train,
            "metrics_eval_forward": m_eval,
            # same-version metrics will be filled in at iter n+1 (lazy)
            "metrics_train_same_version": None,
            "metrics_eval_same_version": None,
        }
        (sub / "metadata.json").write_text(json.dumps(meta, indent=2))
        iter_summary.append(meta)
        # rolling summary
        (out_dir / "iter_summary.json").write_text(json.dumps(iter_summary, indent=2))

        # tsv: only forward maxes and p95 are known immediately; same-version columns are
        # tagged "(n-1)" because the values written at iter n correspond to V_{n-1}.
        with open(out_dir / "iter_summary.tsv", "a") as f:
            if n == args.start_iter:
                f.write(
                    "iter\tparams\tfit_s\tfwd_train_max\tfwd_eval_max\tfwd_train_p95\tfwd_eval_p95\t"
                    "same_train_max_for_iter_n-1\tsame_eval_max_for_iter_n-1\t"
                    "same_train_p95_for_iter_n-1\tsame_eval_p95_for_iter_n-1\t"
                    "fwd_eval_n>0.5\tsame_eval_n>0.5_for_iter_n-1\n"
                )
            if n > args.start_iter:
                ms_t = iter_summary[-2]["metrics_train_same_version"] or {}
                ms_e = iter_summary[-2]["metrics_eval_same_version"] or {}
                same_train_max = ms_t.get("max_abs", float("nan"))
                same_eval_max = ms_e.get("max_abs", float("nan"))
                same_train_p95 = ms_t.get("p95_abs", float("nan"))
                same_eval_p95 = ms_e.get("p95_abs", float("nan"))
                same_eval_n05 = ms_e.get("n_over_05", -1)
            else:
                same_train_max = same_eval_max = same_train_p95 = same_eval_p95 = float("nan")
                same_eval_n05 = -1
            f.write(
                f"{n}\t{n_params}\t{fit_elapsed:.1f}\t"
                f"{m_train['max_abs']:.4f}\t{m_eval['max_abs']:.4f}\t{m_train['p95_abs']:.4f}\t{m_eval['p95_abs']:.4f}\t"
                f"{same_train_max:.4f}\t{same_eval_max:.4f}\t{same_train_p95:.4f}\t{same_eval_p95:.4f}\t"
                f"{m_eval['n_over_05']}\t{same_eval_n05}\n"
            )

        print(
            f"[bell] iter{n} DONE in {time.time() - it_t0:.1f}s "
            f"(fwd eval max={m_eval['max_abs']:.3f})",
            flush=True,
        )

    # Final: V_end has no V_{end+1} target to compute same-version against. Skip.
    print(f"\n[bell] NOTE: V_{args.end_iter}'s same-version CSVs are not produced "
          f"(would require iter {args.end_iter + 1}'s target).", flush=True)
    print(f"\n[bell] OVERALL elapsed={time.time() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
