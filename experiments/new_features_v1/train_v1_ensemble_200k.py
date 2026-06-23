"""
200k-budget V1 ensemble of two complementary HGB regressors averaged.

Sub-model A: ~100k params, weight = 1 + 2*|y| (focuses on tail magnitudes).
Sub-model B: ~100k params, weight = 1 + 50*(|y|<0.05) (focuses on cliff escapees).

Final prediction: clip( 0.5 * (yA + yB), -inf, 0 ).
Each sub-model's params capped near 100k to stay under the 200k total budget.

This addresses the dual-rare-event problem: a single HGB optimised with one weighting
either over-predicts cliff escapees or under-predicts cliff fallers. Averaging two
specialists smooths both.
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
        "frac_over_05": float(np.mean(abs_err > 0.5)),
    }


def _passes_gate(m: dict[str, float]) -> bool:
    return (
        m["mse"] < 0.05
        and m["rmse"] < 0.05
        and m["mae"] < 0.05
        and m["p95_abs"] < 0.05
        and m["max_abs"] < 0.5
    )


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
    parser.add_argument("--leaves-a", type=int, default=23, help="leaves for sub-model A (tail)")
    parser.add_argument("--iters-a", type=int, default=1450, help="iters for sub-model A (~100k params)")
    parser.add_argument("--alpha-a", type=float, default=2.0)
    parser.add_argument("--beta-a", type=float, default=0.0)
    parser.add_argument("--leaves-b", type=int, default=23)
    parser.add_argument("--iters-b", type=int, default=1450)
    parser.add_argument("--alpha-b", type=float, default=0.0)
    parser.add_argument("--beta-b", type=float, default=50.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--threads-per", type=int, default=8)
    args = parser.parse_args()

    _set_threads(args.threads_per)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ens] loading features…")
    f1 = np.load(args.features_d1)
    t1 = np.load(args.targets_d1)
    f2 = np.load(args.features_d2)
    t2 = np.load(args.targets_d2)
    X = np.concatenate([f1, f2], axis=0)
    y = np.concatenate([t1, t2], axis=0)
    split = json.loads(Path(args.split_json).read_text())
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    eval_idx = np.asarray(split["eval_indices"], dtype=np.int64)
    X_train, y_train = X[train_idx], y[train_idx]
    X_eval, y_eval = X[eval_idx], y[eval_idx]
    print(f"[ens] X={X.shape} train n={len(train_idx)} eval n={len(eval_idx)}")

    def _w(alpha: float, beta: float) -> np.ndarray:
        w = 1.0 + alpha * np.abs(y_train)
        if beta > 0:
            w = w + beta * (np.abs(y_train) < 0.05).astype(np.float32)
        return w.astype(np.float32)

    sw_a = _w(args.alpha_a, args.beta_a)
    sw_b = _w(args.alpha_b, args.beta_b)

    print(f"[ens] sub-A: leaves={args.leaves_a} iters={args.iters_a} alpha={args.alpha_a} beta={args.beta_a}")
    print(f"[ens]        sw_a min/median/mean/max: {sw_a.min():.3f}/{np.median(sw_a):.3f}/{sw_a.mean():.3f}/{sw_a.max():.3f}")
    print(f"[ens] sub-B: leaves={args.leaves_b} iters={args.iters_b} alpha={args.alpha_b} beta={args.beta_b}")
    print(f"[ens]        sw_b min/median/mean/max: {sw_b.min():.3f}/{np.median(sw_b):.3f}/{sw_b.mean():.3f}/{sw_b.max():.3f}")

    estA = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=args.iters_a,
        max_leaf_nodes=args.leaves_a,
        learning_rate=args.lr,
        l2_regularization=args.l2,
        random_state=0,
        early_stopping=False,
    )
    estB = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=args.iters_b,
        max_leaf_nodes=args.leaves_b,
        learning_rate=args.lr,
        l2_regularization=args.l2,
        random_state=1,
        early_stopping=False,
    )
    t0 = time.time()
    estA.fit(X_train, y_train, sample_weight=sw_a)
    t_a = time.time() - t0
    nA = _hgb_n_params(estA)
    print(f"[ens] A fit {t_a:.1f}s, params={nA:,}")
    t0 = time.time()
    estB.fit(X_train, y_train, sample_weight=sw_b)
    t_b = time.time() - t0
    nB = _hgb_n_params(estB)
    print(f"[ens] B fit {t_b:.1f}s, params={nB:,}")
    n_total = nA + nB
    print(f"[ens] TOTAL params={n_total:,}")

    yA_train = estA.predict(X_train)
    yB_train = estB.predict(X_train)
    yA_eval = estA.predict(X_eval)
    yB_eval = estB.predict(X_eval)

    # ensemble = clamped average
    yp_train = np.minimum(0.5 * (yA_train + yB_train), 0.0)
    yp_eval = np.minimum(0.5 * (yA_eval + yB_eval), 0.0)

    m_train = _metrics(y_train, yp_train)
    m_eval = _metrics(y_eval, yp_eval)
    gate_train = _passes_gate(m_train)
    gate_eval = _passes_gate(m_eval)
    print(f"[ens] TRAIN gate={gate_train} max={m_train['max_abs']:.3f} p95={m_train['p95_abs']:.4f} "
          f"mae={m_train['mae']:.4f} mse={m_train['mse']:.4f} n>0.5={m_train['n_over_05']}")
    print(f"[ens] EVAL  gate={gate_eval}  max={m_eval['max_abs']:.3f} p95={m_eval['p95_abs']:.4f} "
          f"mae={m_eval['mae']:.4f} mse={m_eval['mse']:.4f} n>0.5={m_eval['n_over_05']}")

    # also report sub-models alone for comparison
    yA_train_clamped = np.minimum(yA_train, 0.0)
    yB_train_clamped = np.minimum(yB_train, 0.0)
    yA_eval_clamped = np.minimum(yA_eval, 0.0)
    yB_eval_clamped = np.minimum(yB_eval, 0.0)
    m_a_eval = _metrics(y_eval, yA_eval_clamped)
    m_b_eval = _metrics(y_eval, yB_eval_clamped)
    print(f"[ens] sub-A alone EVAL: max={m_a_eval['max_abs']:.3f} n>0.5={m_a_eval['n_over_05']}")
    print(f"[ens] sub-B alone EVAL: max={m_b_eval['max_abs']:.3f} n>0.5={m_b_eval['n_over_05']}")

    _write_predictions_csv(out_dir / "train_results.csv", train_idx, y_train, yp_train)
    _write_predictions_csv(out_dir / "eval_results.csv", eval_idx, y_eval, yp_eval)
    meta = {
        "leaves_a": args.leaves_a, "iters_a": args.iters_a, "alpha_a": args.alpha_a, "beta_a": args.beta_a,
        "leaves_b": args.leaves_b, "iters_b": args.iters_b, "alpha_b": args.alpha_b, "beta_b": args.beta_b,
        "lr": args.lr, "l2": args.l2,
        "n_params_a": int(nA), "n_params_b": int(nB), "n_params_total": int(n_total),
        "fit_a_s": t_a, "fit_b_s": t_b,
        "metrics_train": m_train, "metrics_eval": m_eval,
        "metrics_a_eval_alone": m_a_eval, "metrics_b_eval_alone": m_b_eval,
        "gate_train": gate_train, "gate_eval": gate_eval,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    joblib.dump({"estA": estA, "estB": estB}, out_dir / "model.joblib", compress=3)


if __name__ == "__main__":
    main()
