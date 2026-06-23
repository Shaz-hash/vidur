"""
200k capacity sweep for V1 supervised on v4 features.

Aims at the 200,000 effective scalar param budget. Counts ~ 3 * sum_of_leaves.

Sweep over (capacity shape) x (tail_alpha) x (loss):
  shape: 31 leaf x 2150 iter (199,950)
         47 leaf x 1410 iter (198,810)
         63 leaf x 1050 iter (198,450)
         95 leaf x  700 iter (199,500)
        127 leaf x  525 iter (200,025)
  tail_alpha: 2.0, 4.0, 8.0
  loss: squared_error, absolute_error

Reports eval max_abs / p95_abs / MAE / MSE / RMSE for each config and selects the
winner that satisfies MSE/RMSE/MAE/p95 < 0.05 and max_abs < 0.5 on both train and eval.
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


SHAPES: list[tuple[str, int, int]] = [
    # (name, max_leaf_nodes, max_iter)  approx params = 3 * leaves * iter
    ("L31_I2150", 31, 2150),
    ("L47_I1410", 47, 1410),
    ("L63_I1050", 63, 1050),
    ("L95_I0700", 95, 700),
    ("L127_I0525", 127, 525),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-d1", required=True)
    parser.add_argument("--targets-d1", required=True)
    parser.add_argument("--features-d2", required=True)
    parser.add_argument("--targets-d2", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--shapes",
        default=",".join(s[0] for s in SHAPES),
        help="Comma-separated list of shape names to fit (default: all).",
    )
    parser.add_argument(
        "--alphas",
        default="2.0,4.0,8.0",
        help="Comma-separated tail_alpha values for sample_weight = 1 + alpha * |y|",
    )
    parser.add_argument(
        "--losses",
        default="squared_error",
        help="Comma-separated HGB losses (e.g. squared_error,absolute_error).",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--threads-per", type=int, default=8)
    args = parser.parse_args()

    _set_threads(args.threads_per)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[200k] loading features…")
    f1 = np.load(args.features_d1)
    t1 = np.load(args.targets_d1)
    f2 = np.load(args.features_d2)
    t2 = np.load(args.targets_d2)
    X = np.concatenate([f1, f2], axis=0)
    y = np.concatenate([t1, t2], axis=0)
    print(f"[200k] X={X.shape} y={y.shape}")

    split = json.loads(Path(args.split_json).read_text())
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    eval_idx = np.asarray(split["eval_indices"], dtype=np.int64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_eval = X[eval_idx]
    y_eval = y[eval_idx]
    print(f"[200k] train n={len(train_idx)} eval n={len(eval_idx)}")

    selected_shapes = [s for s in SHAPES if s[0] in set(args.shapes.split(","))]
    alphas = [float(a) for a in args.alphas.split(",") if a]
    losses = [l.strip() for l in args.losses.split(",") if l.strip()]

    summary: list[dict[str, Any]] = []
    for loss in losses:
        for alpha in alphas:
            sw = (1.0 + alpha * np.abs(y_train)).astype(np.float32)
            for shape_name, leaves, iters in selected_shapes:
                cfg = f"{loss}_{shape_name}_a{alpha:g}".replace(".", "p")
                print(f"\n[200k] === fitting {cfg}  leaves={leaves} iters={iters} ===")
                est = HistGradientBoostingRegressor(
                    loss=loss,
                    max_iter=iters,
                    max_leaf_nodes=leaves,
                    learning_rate=args.learning_rate,
                    l2_regularization=args.l2,
                    random_state=0,
                    early_stopping=False,
                )
                t0 = time.time()
                est.fit(X_train, y_train, sample_weight=sw)
                fit_elapsed = time.time() - t0
                n_params = _hgb_n_params(est)
                print(f"[200k] {cfg}: fit elapsed={fit_elapsed:.1f}s n_params={n_params:,}")

                t0 = time.time()
                yp_train = np.minimum(est.predict(X_train), 0.0)
                yp_eval = np.minimum(est.predict(X_eval), 0.0)
                pred_elapsed = time.time() - t0

                m_train = _metrics(y_train, yp_train)
                m_eval = _metrics(y_eval, yp_eval)
                gate_train = _passes_gate(m_train)
                gate_eval = _passes_gate(m_eval)
                print(
                    f"[200k] {cfg} TRAIN gate={gate_train} max={m_train['max_abs']:.3f} "
                    f"p95={m_train['p95_abs']:.4f} mae={m_train['mae']:.4f} mse={m_train['mse']:.4f}"
                )
                print(
                    f"[200k] {cfg} EVAL  gate={gate_eval}  max={m_eval['max_abs']:.3f} "
                    f"p95={m_eval['p95_abs']:.4f} mae={m_eval['mae']:.4f} mse={m_eval['mse']:.4f}"
                )

                sub = out_dir / cfg
                sub.mkdir(parents=True, exist_ok=True)
                _write_predictions_csv(sub / "train_results.csv", train_idx, y_train, yp_train)
                _write_predictions_csv(sub / "eval_results.csv", eval_idx, y_eval, yp_eval)
                meta = {
                    "config": cfg,
                    "loss": loss,
                    "leaves": leaves,
                    "iters": iters,
                    "tail_alpha": alpha,
                    "learning_rate": args.learning_rate,
                    "l2": args.l2,
                    "n_params_estimate": int(n_params),
                    "fit_elapsed_s": fit_elapsed,
                    "pred_elapsed_s": pred_elapsed,
                    "metrics_train": m_train,
                    "metrics_eval": m_eval,
                    "gate_train": gate_train,
                    "gate_eval": gate_eval,
                }
                (sub / "metadata.json").write_text(json.dumps(meta, indent=2))
                joblib.dump(est, sub / "model.joblib", compress=3)
                summary.append({"name": cfg, **meta})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[200k] SUMMARY (sorted by eval_max_abs ascending):")
    for s in sorted(summary, key=lambda x: x["metrics_eval"]["max_abs"]):
        print(
            f"  {s['name']:42s} params={s['n_params_estimate']:>8,d} "
            f"train_max={s['metrics_train']['max_abs']:.3f} eval_max={s['metrics_eval']['max_abs']:.3f} "
            f"train_p95={s['metrics_train']['p95_abs']:.4f} eval_p95={s['metrics_eval']['p95_abs']:.4f} "
            f"eval_mae={s['metrics_eval']['mae']:.4f} eval_n>0.5={s['metrics_eval']['n_over_05']} "
            f"gate_eval={s['gate_eval']}"
        )


if __name__ == "__main__":
    main()
