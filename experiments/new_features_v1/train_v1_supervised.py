"""
Iteration-1 (V1) supervised trainer on the new state-local features.

Label: target_value (== act_max_reward at iteration 1, by construction).
Input: features built by build_state_local_features.py for combined Dataset 1+2.

Trains and evaluates several classical models under the 120k param budget:
  - HistGradientBoosting with absolute_error loss
  - HistGradientBoosting with squared_error loss
  - ExtraTreesRegressor (small)

Writes per-model:
  - predictions on train + eval (CSV)
  - aggregate metrics + n_params estimate (JSON)
  - the joblib model
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
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor


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
    # approx leaves * 3 (split feat, threshold, value)
    total = 0
    for est in model._predictors:
        for tree in est:
            try:
                total += int(np.sum(tree.nodes["is_leaf"])) * 3
            except Exception:
                # Fallback: count node count.
                total += int(tree.nodes.size) * 3
    return total


def _et_n_params(model: ExtraTreesRegressor) -> int:
    total = 0
    for est in model.estimators_:
        total += int(est.tree_.node_count) * 3
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


def _train_model(name: str, est: Any, X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None) -> tuple[Any, float]:
    t0 = time.time()
    if sample_weight is not None:
        est.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        est.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[train] {name}: fit elapsed={elapsed:.1f}s")
    return est, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-d1", required=True)
    parser.add_argument("--targets-d1", required=True)
    parser.add_argument("--features-d2", required=True)
    parser.add_argument("--targets-d2", required=True)
    parser.add_argument("--split-json", required=True, help="cached_v20_combined_650k/split_indices_combined.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--tail-alpha", type=float, default=4.0, help="sample_weight = 1 + alpha*|target|")
    parser.add_argument("--threads-per", type=int, default=4)
    args = parser.parse_args()

    _set_threads(args.threads_per)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[train] loading features…")
    f1 = np.load(args.features_d1)
    t1 = np.load(args.targets_d1)
    f2 = np.load(args.features_d2)
    t2 = np.load(args.targets_d2)
    print(f"[train] d1={f1.shape}/{t1.shape} d2={f2.shape}/{t2.shape}")
    X = np.concatenate([f1, f2], axis=0)
    y = np.concatenate([t1, t2], axis=0)
    print(f"[train] combined X={X.shape} y={y.shape}")

    split = json.loads(Path(args.split_json).read_text())
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    eval_idx = np.asarray(split["eval_indices"], dtype=np.int64)
    print(f"[train] train n={len(train_idx)} eval n={len(eval_idx)}")
    assert max(train_idx.max(), eval_idx.max()) < len(X), "split index oob"

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_eval = X[eval_idx]
    y_eval = y[eval_idx]

    print(f"[train] target ranges: y_train min={y_train.min():.4f} max={y_train.max():.4f} mean={y_train.mean():.4f} std={y_train.std():.4f}")
    print(f"[train] target ranges: y_eval  min={y_eval.min():.4f}  max={y_eval.max():.4f}  mean={y_eval.mean():.4f}  std={y_eval.std():.4f}")

    sample_weight = None
    if args.tail_alpha > 0:
        sample_weight = (1.0 + args.tail_alpha * np.abs(y_train)).astype(np.float32)
        print(f"[train] tail_alpha={args.tail_alpha} sw mean={sample_weight.mean():.3f} max={sample_weight.max():.3f}")

    models = {
        "hgb_abs_27leaf_900iter": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=900,
            max_leaf_nodes=27,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=0,
            early_stopping=False,
        ),
        "hgb_abs_31leaf_700iter": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=700,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=0,
            early_stopping=False,
        ),
        "hgb_sq_31leaf_700iter": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=700,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=0,
            early_stopping=False,
        ),
        "et_64leaf_24trees": ExtraTreesRegressor(
            n_estimators=24,
            max_leaf_nodes=64,
            min_samples_leaf=20,
            n_jobs=args.n_jobs,
            random_state=0,
        ),
    }

    summary: list[dict[str, Any]] = []
    for name, est in models.items():
        print(f"\n[train] === fitting {name} ===")
        est, fit_elapsed = _train_model(name, est, X_train, y_train, sample_weight)
        # n_params
        if isinstance(est, HistGradientBoostingRegressor):
            n_params = _hgb_n_params(est)
        else:
            n_params = _et_n_params(est)
        print(f"[train] {name}: n_params={n_params:,}")
        # predictions
        t0 = time.time()
        yp_train = est.predict(X_train)
        yp_eval = est.predict(X_eval)
        # clamp to <= 0 (V is non-positive cost)
        yp_train = np.minimum(yp_train, 0.0)
        yp_eval = np.minimum(yp_eval, 0.0)
        pred_elapsed = time.time() - t0
        m_train = _metrics(y_train, yp_train)
        m_eval = _metrics(y_eval, yp_eval)
        passed_train = (
            m_train["mse"] < 0.05 and m_train["rmse"] < 0.05 and m_train["mae"] < 0.05
            and m_train["p95_abs"] < 0.05 and m_train["max_abs"] < 0.5
        )
        passed_eval = (
            m_eval["mse"] < 0.05 and m_eval["rmse"] < 0.05 and m_eval["mae"] < 0.05
            and m_eval["p95_abs"] < 0.05 and m_eval["max_abs"] < 0.5
        )
        print(f"[train] {name} TRAIN {m_train}")
        print(f"[train] {name} EVAL  {m_eval}")
        print(f"[train] {name} pass_train={passed_train} pass_eval={passed_eval} pred_elapsed={pred_elapsed:.1f}s")

        # write predictions + meta
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
            "pass_train": bool(passed_train),
            "pass_eval": bool(passed_eval),
        }
        (sub / "metadata.json").write_text(json.dumps(meta, indent=2))
        joblib.dump(est, sub / "model.joblib", compress=3)
        summary.append({"name": name, **meta})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[train] SUMMARY:")
    for s in summary:
        print(
            f"  {s['name']:36s} params={s['n_params_estimate']:>8,d} "
            f"train_max={s['metrics_train']['max_abs']:.3f} eval_max={s['metrics_eval']['max_abs']:.3f} "
            f"train_p95={s['metrics_train']['p95_abs']:.3f} eval_p95={s['metrics_eval']['p95_abs']:.3f} "
            f"train_mae={s['metrics_train']['mae']:.4f} eval_mae={s['metrics_eval']['mae']:.4f}"
        )


if __name__ == "__main__":
    main()
