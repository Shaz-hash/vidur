"""
200k V1 multiclass bucket classifier:
  - Round y to integer cliff bucket in {-7,-6,-5,-4,-3,-2,-1,0}.
  - Train HGBClassifier to predict bucket given x.
  - Inference: predicted bucket -> bucket-mean (computed from training data).
  - Optionally add a small within-bucket residual regressor that adjusts.

The problem class is dominated by the integer-cliff structure of targets, where each
"reward" event drops y by 1.0 and within-bucket residual std is < 0.13 for buckets
near zero. So if classification is perfect, max_abs ≤ ~0.5 by construction.

Two configs run:
  A) classifier-only: y_hat = mean(y | bucket_pred)
  B) classifier + within-bucket residual regressor (small)

Param budgets:
  A: ~199,500 in classifier (8 classes × ~25k each ≈ 200k)
  B: classifier ~150k + residual regressor ~50k
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
    parser.add_argument("--cls-iters", type=int, default=600)
    parser.add_argument("--use-residual", action="store_true",
                        help="Add small per-bucket residual regressor")
    parser.add_argument("--res-leaves", type=int, default=15)
    parser.add_argument("--res-iters", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--threads-per", type=int, default=16)
    args = parser.parse_args()

    _set_threads(args.threads_per)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bkt] loading features…")
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
    print(f"[bkt] X={X.shape} train n={len(train_idx)} eval n={len(eval_idx)}")

    # bucket = round(y) clipped to [-7, 0]
    b_train = np.clip(np.round(y_train).astype(np.int64), -7, 0)
    b_eval = np.clip(np.round(y_eval).astype(np.int64), -7, 0)
    bucket_classes, bucket_counts = np.unique(b_train, return_counts=True)
    print(f"[bkt] train bucket histogram: {list(zip(bucket_classes.tolist(), bucket_counts.tolist()))}")
    # bucket-mean lookup from training y
    bucket_mean = {b: float(y_train[b_train == b].mean()) for b in bucket_classes.tolist()}
    print(f"[bkt] bucket means: {bucket_mean}")

    # -------- Stage 1: multiclass classifier --------
    cls = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=args.cls_iters,
        max_leaf_nodes=args.cls_leaves,
        learning_rate=args.lr,
        l2_regularization=args.l2,
        random_state=0,
        early_stopping=False,
    )
    print(f"[bkt] CLS fit leaves={args.cls_leaves} iters={args.cls_iters} (multiclass {len(bucket_classes)})")
    t0 = time.time()
    cls.fit(X_train, b_train)
    t_cls = time.time() - t0
    n_cls = _hgb_n_params(cls)
    print(f"[bkt] CLS fit {t_cls:.1f}s params={n_cls:,}")

    # predicted bucket (argmax)
    pb_train = cls.predict(X_train)
    pb_eval = cls.predict(X_eval)
    cls_acc_train = float(np.mean(pb_train == b_train))
    cls_acc_eval = float(np.mean(pb_eval == b_eval))
    print(f"[bkt] CLS accuracy: train={cls_acc_train:.4f} eval={cls_acc_eval:.4f}")

    # confusion stats by bucket on eval
    print(f"[bkt] EVAL confusion by true bucket (true_b -> pred_b distribution):")
    for b in sorted(bucket_classes.tolist()):
        m = b_eval == b
        if m.sum() == 0:
            continue
        sub = pb_eval[m]
        u, c = np.unique(sub, return_counts=True)
        order = np.argsort(-c)
        top = [(int(u[i]), int(c[i]), float(c[i]) / m.sum()) for i in order[:3]]
        print(f"  true={b:>3d} (n={m.sum():>6d}): top preds = {top}")

    # bucket-mean prediction
    bm_arr = np.array([bucket_mean[int(b)] for b in pb_train])
    bm_arr_e = np.array([bucket_mean[int(b)] for b in pb_eval])

    n_total = n_cls

    if args.use_residual:
        # within-bucket residual regressor: predict (y - bucket_mean(round(y)))
        # so residual targets in [-0.5, 0.5], small and homoscedastic.
        # train on rows where pb_train == b_train (correctly classified) so
        # at inference residual model's input bucket == true bucket of training rows.
        # Actually simpler: fit residual r(x) = y - bucket_mean(b_train_TRUE)
        # then add r(x) regardless. Residual model sees the residual signal.
        true_bm = np.array([bucket_mean[int(b)] for b in b_train])
        res_target = (y_train - true_bm).astype(np.float64)
        print(f"[bkt] residual target: mean={res_target.mean():.4f} std={res_target.std():.4f} "
              f"min={res_target.min():.3f} max={res_target.max():.3f}")
        res = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=args.res_iters,
            max_leaf_nodes=args.res_leaves,
            learning_rate=args.lr,
            l2_regularization=args.l2,
            random_state=0,
            early_stopping=False,
        )
        t0 = time.time()
        res.fit(X_train, res_target)
        t_res = time.time() - t0
        n_res = _hgb_n_params(res)
        print(f"[bkt] RES fit {t_res:.1f}s params={n_res:,}")
        n_total += n_res
        r_train = res.predict(X_train)
        r_eval = res.predict(X_eval)
    else:
        r_train = np.zeros_like(bm_arr)
        r_eval = np.zeros_like(bm_arr_e)
        n_res = 0
        t_res = 0.0
        res = None

    # final prediction
    yp_train = bm_arr + r_train
    yp_eval = bm_arr_e + r_eval
    # clamp ≤ 0
    yp_train = np.minimum(yp_train, 0.0)
    yp_eval = np.minimum(yp_eval, 0.0)

    m_train = _metrics(y_train, yp_train)
    m_eval = _metrics(y_eval, yp_eval)
    gate_train = _passes_gate(m_train)
    gate_eval = _passes_gate(m_eval)
    print(f"[bkt] TRAIN gate={gate_train} max={m_train['max_abs']:.3f} p95={m_train['p95_abs']:.4f} "
          f"mae={m_train['mae']:.4f} mse={m_train['mse']:.4f} n>0.5={m_train['n_over_05']}")
    print(f"[bkt] EVAL  gate={gate_eval}  max={m_eval['max_abs']:.3f} p95={m_eval['p95_abs']:.4f} "
          f"mae={m_eval['mae']:.4f} mse={m_eval['mse']:.4f} n>0.5={m_eval['n_over_05']}")
    print(f"[bkt] TOTAL params={n_total:,}")

    _write_predictions_csv(out_dir / "train_results.csv", train_idx, y_train, yp_train)
    _write_predictions_csv(out_dir / "eval_results.csv", eval_idx, y_eval, yp_eval)
    meta = {
        "cls_leaves": args.cls_leaves, "cls_iters": args.cls_iters,
        "use_residual": args.use_residual,
        "res_leaves": args.res_leaves, "res_iters": args.res_iters,
        "lr": args.lr, "l2": args.l2,
        "n_params_cls": int(n_cls), "n_params_res": int(n_res), "n_params_total": int(n_total),
        "fit_cls_s": t_cls, "fit_res_s": t_res,
        "cls_acc_train": cls_acc_train, "cls_acc_eval": cls_acc_eval,
        "bucket_mean": bucket_mean,
        "metrics_train": m_train, "metrics_eval": m_eval,
        "gate_train": gate_train, "gate_eval": gate_eval,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    saved = {"cls": cls, "bucket_mean": bucket_mean}
    if res is not None:
        saved["res"] = res
    joblib.dump(saved, out_dir / "model.joblib", compress=3)
    print(f"[bkt] saved to {out_dir}")


if __name__ == "__main__":
    main()
