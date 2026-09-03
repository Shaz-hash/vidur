"""
XL capacity sweep at variable tail_alpha for V1 supervised on v4 features.

Same three hgb_sq configurations as train_v1_supervised_xl.py (near 120k params),
trained at a single chosen tail_alpha value passed via --tail-alpha.

Usage:
  python -m vidur.bellman_v4_adv.train_v1_supervised_xl_alpha \
      --features-d1 ... --targets-d1 ... \
      --features-d2 ... --targets-d2 ... \
      --split-json ... --output-dir ... --tail-alpha 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor


def _set_threads(n: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass


def _hgb_n_params(model: HistGradientBoostingRegressor) -> int:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-d1", required=True)
    parser.add_argument("--targets-d1", required=True)
    parser.add_argument("--features-d2", required=True)
    parser.add_argument("--targets-d2", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-alpha", type=float, required=True)
    parser.add_argument("--threads-per", type=int, default=8)
    args = parser.parse_args()

    _set_threads(args.threads_per)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[xl-a] tail_alpha={args.tail_alpha} loading features…")
    f1 = np.load(args.features_d1)
    t1 = np.load(args.targets_d1)
    f2 = np.load(args.features_d2)
    t2 = np.load(args.targets_d2)
    X = np.concatenate([f1, f2], axis=0)
    y = np.concatenate([t1, t2], axis=0)
    print(f"[xl-a] X={X.shape} y={y.shape}")

    split = json.loads(Path(args.split_json).read_text())
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    eval_idx = np.asarray(split["eval_indices"], dtype=np.int64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_eval = X[eval_idx]
    y_eval = y[eval_idx]
    print(f"[xl-a] train n={len(train_idx)} eval n={len(eval_idx)}")

    sample_weight = (1.0 + args.tail_alpha * np.abs(y_train)).astype(np.float32)

    models = {
        "hgb_sq_31leaf_1280iter": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=1280,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=0,
            early_stopping=False,
        ),
        "hgb_sq_63leaf_630iter": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=630,
            max_leaf_nodes=63,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=0,
            early_stopping=False,
        ),
        "hgb_sq_47leaf_850iter": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=850,
            max_leaf_nodes=47,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=0,
            early_stopping=False,
        ),
    }

    summary: list[dict[str, Any]] = []
    for name, est in models.items():
        print(f"\n[xl-a] === fitting {name} alpha={args.tail_alpha} ===")
        t0 = time.time()
        est.fit(X_train, y_train, sample_weight=sample_weight)
        fit_elapsed = time.time() - t0
        n_params = _hgb_n_params(est)
        print(f"[xl-a] {name}: fit elapsed={fit_elapsed:.1f}s n_params={n_params:,}")

        t0 = time.time()
        yp_train = np.minimum(est.predict(X_train), 0.0)
        yp_eval = np.minimum(est.predict(X_eval), 0.0)
        pred_elapsed = time.time() - t0

        m_train = _metrics(y_train, yp_train)
        m_eval = _metrics(y_eval, yp_eval)
        print(f"[xl-a] {name} TRAIN {m_train}")
        print(f"[xl-a] {name} EVAL  {m_eval}")

        sub = out_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        _write_predictions_csv(sub / "train_results.csv", train_idx, y_train, yp_train)
        _write_predictions_csv(sub / "eval_results.csv", eval_idx, y_eval, yp_eval)
        meta = {
            "model": name,
            "n_params_estimate": int(n_params),
            "fit_elapsed_s": fit_elapsed,
            "pred_elapsed_s": pred_elapsed,
            "tail_alpha": args.tail_alpha,
            "metrics_train": m_train,
            "metrics_eval": m_eval,
        }
        (sub / "metadata.json").write_text(json.dumps(meta, indent=2))
        joblib.dump(est, sub / "model.joblib", compress=3)
        summary.append({"name": name, **meta})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[xl-a] SUMMARY:")
    for s in summary:
        print(
            f"  {s['name']:30s} alpha={args.tail_alpha:.1f} params={s['n_params_estimate']:>8,d} "
            f"train_max={s['metrics_train']['max_abs']:.3f} eval_max={s['metrics_eval']['max_abs']:.3f} "
            f"train_p95={s['metrics_train']['p95_abs']:.4f} eval_p95={s['metrics_eval']['p95_abs']:.4f} "
            f"train_mae={s['metrics_train']['mae']:.4f} eval_mae={s['metrics_eval']['mae']:.4f} "
            f"eval_n>0.5={s['metrics_eval']['n_over_05']}"
        )


if __name__ == "__main__":
    main()
