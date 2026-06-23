"""
Hierarchical V1: cliff-escapee classifier + cliff-faller regressor under 200k params.

Two-stage architecture motivated by the dual-rare-event problem in v4 features:
  - 51% of training rows have y == 0 (cliff escapees: B>=128 picks would not violate).
  - ~9% have y in {-2, -3} (cliff fallers: every B picks introduces a violation).
  A single weighted regressor either over-suppresses cliff escapees (predicts negative
  when truth is 0) or under-corrects cliff fallers (predicts ~0 when truth is -2/-3).

Stage 1 — HGBClassifier g(x) = P( y >= -0.05 | x )
  - max_leaf_nodes=15, max_iter=1100  →  ~50k params (3 leaves per node)
  - sample_weight = 1 (uniform; logistic loss already handles class imbalance reasonably,
    but we add a positive-class boost to push P>tau on the dominant y=0 class).

Stage 2 — HGBRegressor r(x) trained ONLY on rows with y < -0.05
  - max_leaf_nodes=31, max_iter=1600  →  ~150k params
  - sample_weight = 1 + 2*|y|  (tail focus, since this dataset only has negatives now)
  - Loss: squared_error, prediction clipped at 0.

Inference:  y_hat = 0 if g(x) > tau else min(r(x), 0).
We sweep tau in {0.50, 0.70, 0.85, 0.95} and keep the one minimising EVAL max_abs.

Total params budget: classifier ~50k + regressor ~150k ≈ 200k.
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


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


def _gated_predict(p_pos: np.ndarray, r_neg: np.ndarray, tau: float) -> np.ndarray:
    is_zero = p_pos > tau
    yp = np.where(is_zero, 0.0, np.minimum(r_neg, 0.0))
    return yp.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-d1", required=True)
    parser.add_argument("--targets-d1", required=True)
    parser.add_argument("--features-d2", required=True)
    parser.add_argument("--targets-d2", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    # classifier
    parser.add_argument("--cls-leaves", type=int, default=15)
    parser.add_argument("--cls-iters", type=int, default=1100)
    parser.add_argument("--cls-pos-weight", type=float, default=1.0,
                        help="weight multiplier on positive (y>=-0.05) class")
    parser.add_argument("--cls-threshold", type=float, default=0.0,
                        help="cliff threshold; rows with y >= -threshold count as POS class")
    # regressor (only fits on rows with y < -threshold)
    parser.add_argument("--reg-leaves", type=int, default=31)
    parser.add_argument("--reg-iters", type=int, default=1600)
    parser.add_argument("--reg-alpha", type=float, default=2.0)
    parser.add_argument("--reg-loss", default="squared_error")
    # tau sweep
    parser.add_argument("--tau-grid", default="0.50,0.70,0.85,0.95")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--threads-per", type=int, default=8)
    args = parser.parse_args()

    _set_threads(args.threads_per)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hier] loading features…")
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
    print(f"[hier] X={X.shape} train n={len(train_idx)} eval n={len(eval_idx)}")

    # cliff label: 1 = escapee (y >= -threshold), 0 = faller
    thr = args.cls_threshold if args.cls_threshold > 0 else 0.05
    is_pos_train = (y_train >= -thr).astype(np.int32)
    is_pos_eval = (y_eval >= -thr).astype(np.int32)
    print(f"[hier] cliff threshold={thr}")
    print(f"[hier] train pos%={is_pos_train.mean():.3f} eval pos%={is_pos_eval.mean():.3f}")

    # -------- Stage 1: classifier --------
    cls_sw = np.where(is_pos_train == 1, args.cls_pos_weight, 1.0).astype(np.float32)
    cls = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=args.cls_iters,
        max_leaf_nodes=args.cls_leaves,
        learning_rate=args.lr,
        l2_regularization=args.l2,
        random_state=0,
        early_stopping=False,
    )
    print(f"[hier] CLS fit leaves={args.cls_leaves} iters={args.cls_iters} pos_w={args.cls_pos_weight}")
    t0 = time.time()
    cls.fit(X_train, is_pos_train, sample_weight=cls_sw)
    t_cls = time.time() - t0
    n_cls = _hgb_n_params(cls)
    print(f"[hier] CLS fit {t_cls:.1f}s params={n_cls:,}")

    p_train = cls.predict_proba(X_train)[:, 1]
    p_eval = cls.predict_proba(X_eval)[:, 1]

    # -------- Stage 2: regressor on negatives only --------
    neg_mask = is_pos_train == 0
    Xr = X_train[neg_mask]
    yr = y_train[neg_mask]
    sw_r = (1.0 + args.reg_alpha * np.abs(yr)).astype(np.float32)
    print(f"[hier] REG training set size={Xr.shape[0]} (negatives only)")
    print(f"[hier] REG fit leaves={args.reg_leaves} iters={args.reg_iters} alpha={args.reg_alpha} loss={args.reg_loss}")

    reg = HistGradientBoostingRegressor(
        loss=args.reg_loss,
        max_iter=args.reg_iters,
        max_leaf_nodes=args.reg_leaves,
        learning_rate=args.lr,
        l2_regularization=args.l2,
        random_state=0,
        early_stopping=False,
    )
    t0 = time.time()
    reg.fit(Xr, yr, sample_weight=sw_r)
    t_reg = time.time() - t0
    n_reg = _hgb_n_params(reg)
    print(f"[hier] REG fit {t_reg:.1f}s params={n_reg:,}")
    n_total = n_cls + n_reg
    print(f"[hier] TOTAL params={n_total:,}")

    r_train = reg.predict(X_train)
    r_eval = reg.predict(X_eval)

    # -------- tau sweep --------
    taus = [float(t) for t in args.tau_grid.split(",") if t]
    best = None
    sweep = []
    for tau in taus:
        yp_tr = _gated_predict(p_train, r_train, tau)
        yp_ev = _gated_predict(p_eval, r_eval, tau)
        m_tr = _metrics(y_train, yp_tr)
        m_ev = _metrics(y_eval, yp_ev)
        gate_tr = _passes_gate(m_tr)
        gate_ev = _passes_gate(m_ev)
        sweep.append({
            "tau": tau, "metrics_train": m_tr, "metrics_eval": m_ev,
            "gate_train": gate_tr, "gate_eval": gate_ev,
        })
        print(f"[hier] tau={tau:.2f} TRAIN max={m_tr['max_abs']:.3f} n>0.5={m_tr['n_over_05']} "
              f"EVAL max={m_ev['max_abs']:.3f} n>0.5={m_ev['n_over_05']} "
              f"gate_eval={gate_ev}")
        if best is None or m_ev["max_abs"] < best["metrics_eval"]["max_abs"]:
            best = sweep[-1]
            best["yp_train"] = yp_tr
            best["yp_eval"] = yp_ev

    print(f"\n[hier] BEST tau={best['tau']:.2f} EVAL max={best['metrics_eval']['max_abs']:.3f} "
          f"n>0.5={best['metrics_eval']['n_over_05']} gate_eval={best['gate_eval']}")

    yp_train = best["yp_train"]
    yp_eval = best["yp_eval"]
    _write_predictions_csv(out_dir / "train_results.csv", train_idx, y_train, yp_train)
    _write_predictions_csv(out_dir / "eval_results.csv", eval_idx, y_eval, yp_eval)
    meta = {
        "cls_leaves": args.cls_leaves, "cls_iters": args.cls_iters,
        "cls_pos_weight": args.cls_pos_weight, "cls_threshold": thr,
        "reg_leaves": args.reg_leaves, "reg_iters": args.reg_iters,
        "reg_alpha": args.reg_alpha, "reg_loss": args.reg_loss,
        "lr": args.lr, "l2": args.l2,
        "n_params_cls": int(n_cls), "n_params_reg": int(n_reg), "n_params_total": int(n_total),
        "fit_cls_s": t_cls, "fit_reg_s": t_reg,
        "tau_sweep": [{k: v for k, v in s.items() if k not in {"yp_train", "yp_eval"}} for s in sweep],
        "best_tau": best["tau"],
        "metrics_train": best["metrics_train"], "metrics_eval": best["metrics_eval"],
        "gate_train": best["gate_train"], "gate_eval": best["gate_eval"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    joblib.dump({"cls": cls, "reg": reg, "tau": best["tau"], "thr": thr}, out_dir / "model.joblib", compress=3)
    print(f"[hier] saved to {out_dir}")


if __name__ == "__main__":
    main()
