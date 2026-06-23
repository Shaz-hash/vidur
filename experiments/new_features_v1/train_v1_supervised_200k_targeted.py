"""
200k targeted sweep — focuses on cliff-region rare-event modeling.

Adds:
  - alpha=0 (uniform weighting) so the dominant y=0 rows aren't suppressed.
  - low-y-boost weighting: weight = 1 + alpha*|y| + beta*(|y|<0.05) (extra weight on near-zero rows).
  - higher l2 / lower learning_rate combos.

Sweeps:
  shape: L31 x 2150 (best so far) and L23 x 2900 (~200k with smaller leaves, longer chain)
  weight: alpha=0, alpha=2 + beta=10, alpha=2 + beta=50
  loss: squared_error, absolute_error
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


def _build_weights(y: np.ndarray, alpha: float, near_zero_boost: float) -> np.ndarray:
    w = 1.0 + alpha * np.abs(y)
    if near_zero_boost > 0.0:
        # Near-zero is |y| < 0.05 (the gate threshold)
        w = w + near_zero_boost * (np.abs(y) < 0.05).astype(np.float32)
    return w.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-d1", required=True)
    parser.add_argument("--targets-d1", required=True)
    parser.add_argument("--features-d2", required=True)
    parser.add_argument("--targets-d2", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated list of <shape>:<alpha>:<beta>:<loss>:<lr>:<l2> tuples; if empty, runs default set.",
    )
    parser.add_argument("--threads-per", type=int, default=8)
    args = parser.parse_args()

    _set_threads(args.threads_per)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[200kT] loading features…")
    f1 = np.load(args.features_d1)
    t1 = np.load(args.targets_d1)
    f2 = np.load(args.features_d2)
    t2 = np.load(args.targets_d2)
    X = np.concatenate([f1, f2], axis=0)
    y = np.concatenate([t1, t2], axis=0)
    print(f"[200kT] X={X.shape} y={y.shape}")

    split = json.loads(Path(args.split_json).read_text())
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    eval_idx = np.asarray(split["eval_indices"], dtype=np.int64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_eval = X[eval_idx]
    y_eval = y[eval_idx]
    print(f"[200kT] train n={len(train_idx)} eval n={len(eval_idx)}")

    SHAPE_MAP = {
        "L31_I2150": (31, 2150),
        "L23_I2900": (23, 2900),  # narrower trees for sparser leaf coverage
        "L47_I1410": (47, 1410),
        "L19_I3500": (19, 3500),  # very narrow + very long
    }

    # Default config list: explore alpha=0 (uniform), beta-boost, deeper l2, lower lr
    if args.configs:
        configs = []
        for spec in args.configs.split(","):
            spec = spec.strip()
            if not spec:
                continue
            parts = spec.split(":")
            shape, alpha_s, beta_s, loss = parts[0], parts[1], parts[2], parts[3]
            lr = float(parts[4]) if len(parts) > 4 else 0.05
            l2 = float(parts[5]) if len(parts) > 5 else 1.0
            configs.append({
                "shape": shape,
                "alpha": float(alpha_s),
                "beta": float(beta_s),
                "loss": loss,
                "lr": lr,
                "l2": l2,
            })
    else:
        configs = [
            # alpha=0 uniform (let dominant y=0 voice through)
            {"shape": "L31_I2150", "alpha": 0.0, "beta": 0.0, "loss": "squared_error", "lr": 0.05, "l2": 1.0},
            # small alpha + strong near-zero boost
            {"shape": "L31_I2150", "alpha": 1.0, "beta": 5.0, "loss": "squared_error", "lr": 0.05, "l2": 1.0},
            {"shape": "L31_I2150", "alpha": 2.0, "beta": 10.0, "loss": "squared_error", "lr": 0.05, "l2": 1.0},
            # narrower trees (better isolation, more iters)
            {"shape": "L23_I2900", "alpha": 2.0, "beta": 10.0, "loss": "squared_error", "lr": 0.05, "l2": 1.0},
            {"shape": "L23_I2900", "alpha": 0.0, "beta": 0.0, "loss": "squared_error", "lr": 0.05, "l2": 1.0},
            # absolute_error with no weighting (median fitting → robust)
            {"shape": "L31_I2150", "alpha": 0.0, "beta": 0.0, "loss": "absolute_error", "lr": 0.05, "l2": 1.0},
            # heavy l2 + lower lr to suppress overfitting
            {"shape": "L31_I2150", "alpha": 2.0, "beta": 10.0, "loss": "squared_error", "lr": 0.025, "l2": 5.0},
        ]

    summary: list[dict[str, Any]] = []
    for cfg in configs:
        leaves, iters = SHAPE_MAP[cfg["shape"]]
        alpha = cfg["alpha"]
        beta = cfg["beta"]
        loss = cfg["loss"]
        lr = cfg["lr"]
        l2 = cfg["l2"]
        sw = _build_weights(y_train, alpha, beta)
        name = f"{loss}_{cfg['shape']}_a{alpha:g}_b{beta:g}_lr{lr:g}_l2{l2:g}".replace(".", "p")
        print(f"\n[200kT] === fitting {name} (leaves={leaves} iters={iters}) ===")
        print(f"[200kT] sw min/median/mean/max: {sw.min():.3f}/{np.median(sw):.3f}/{sw.mean():.3f}/{sw.max():.3f}")

        est = HistGradientBoostingRegressor(
            loss=loss,
            max_iter=iters,
            max_leaf_nodes=leaves,
            learning_rate=lr,
            l2_regularization=l2,
            random_state=0,
            early_stopping=False,
        )
        t0 = time.time()
        est.fit(X_train, y_train, sample_weight=sw)
        fit_elapsed = time.time() - t0
        n_params = _hgb_n_params(est)
        print(f"[200kT] {name}: fit elapsed={fit_elapsed:.1f}s n_params={n_params:,}")

        t0 = time.time()
        yp_train = np.minimum(est.predict(X_train), 0.0)
        yp_eval = np.minimum(est.predict(X_eval), 0.0)
        pred_elapsed = time.time() - t0

        m_train = _metrics(y_train, yp_train)
        m_eval = _metrics(y_eval, yp_eval)
        gate_train = _passes_gate(m_train)
        gate_eval = _passes_gate(m_eval)
        print(
            f"[200kT] {name} TRAIN gate={gate_train} max={m_train['max_abs']:.3f} "
            f"p95={m_train['p95_abs']:.4f} mae={m_train['mae']:.4f} mse={m_train['mse']:.4f} "
            f"n>0.5={m_train['n_over_05']}"
        )
        print(
            f"[200kT] {name} EVAL  gate={gate_eval}  max={m_eval['max_abs']:.3f} "
            f"p95={m_eval['p95_abs']:.4f} mae={m_eval['mae']:.4f} mse={m_eval['mse']:.4f} "
            f"n>0.5={m_eval['n_over_05']}"
        )

        sub = out_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        _write_predictions_csv(sub / "train_results.csv", train_idx, y_train, yp_train)
        _write_predictions_csv(sub / "eval_results.csv", eval_idx, y_eval, yp_eval)
        meta = {
            "config": name,
            **cfg,
            "leaves": leaves,
            "iters": iters,
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
        summary.append({"name": name, **meta})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[200kT] SUMMARY (sorted by eval_max_abs ascending):")
    for s in sorted(summary, key=lambda x: x["metrics_eval"]["max_abs"]):
        print(
            f"  {s['name']:60s} params={s['n_params_estimate']:>8,d} "
            f"train_max={s['metrics_train']['max_abs']:.3f} eval_max={s['metrics_eval']['max_abs']:.3f} "
            f"train_n>0.5={s['metrics_train']['n_over_05']:>4d} eval_n>0.5={s['metrics_eval']['n_over_05']:>4d} "
            f"gate_eval={s['gate_eval']}"
        )


if __name__ == "__main__":
    main()
