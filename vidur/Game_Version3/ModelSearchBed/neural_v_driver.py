"""FQI driver with NN V regressor (replacing HGB V).

Reuses the per_row_input.npy + parent_features.npy + child_features.npy
memmaps from the existing q_learning_driver runs.

Per iteration:
  1. Compute targets per row: target = r + γ * V_{N-1}(s'_child).
     V_{N-1}(s') comes from the previous NN V model. For V1, V_{N-1}=0.
  2. Train Q on per-row [parent_features, per_action_features] -> target.
     Q is HGB (handles 14M rows well).
  3. Compute V_N target per parent = max_a Q_N(s, a).
  4. Train NN V on parent_features -> V_N target.
     Architecture: 498 -> 256 -> 128 -> 64 -> 1 with -softplus output (V <= 0).
     Loss: weighted MSE with sample_weight = 1 + tail_scale*min(1,|y|/3).
  5. Save NN V as next bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import logger as bellman_logger


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
class NNVConfig:
    cached_run_dir: Path        # has child_features.npy, child_meta.npy, parent_index.npz, split_indices.json
    fqi_run_dir: Path           # has per_row_input.npy, parent_features.npy, per_row_action.npy
    output_dir: Path
    num_versions: int = 25
    seed: int = 2027
    abs_error_threshold: float = 1.0
    bootstrap_shrink: float = 0.6
    # Q model hparams (HGB)
    q_hgb_max_iter: int = 400
    q_hgb_max_leaf_nodes: int = 31
    q_hgb_learning_rate: float = 0.05
    # NN V hparams
    v_hidden_dims: tuple[int, ...] = (256, 128, 64)
    v_lr: float = 1e-3
    v_epochs: int = 120
    v_batch_size: int = 4096
    v_tail_weight_scale: float = 50.0
    v_weight_decay: float = 1e-5
    v_dropout: float = 0.0
    # Misc
    target_chunk_size: int = 1_000_000
    extra_config: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# NN V model
# ---------------------------------------------------------------------------


def _build_nn_v(in_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.0):
    import torch.nn as nn

    layers: list[nn.Module] = []
    last = int(in_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        last = int(h)
    # final scalar
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


class NNVPredictor:
    """Wraps a torch module + per-feature normalization + V<=0 enforcement.

    Provides .predict(x) -> np.ndarray and .save / .load for joblib.
    """

    def __init__(
        self,
        module,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        self.module = module
        self.feat_mean = np.asarray(feat_mean, dtype=np.float32)
        self.feat_std = np.asarray(feat_std, dtype=np.float32)
        self.in_dim = int(in_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.dropout = float(dropout)

    def _norm(self, X: np.ndarray) -> np.ndarray:
        std = np.where(self.feat_std > 1e-9, self.feat_std, 1.0)
        return (X.astype(np.float32) - self.feat_mean) / std

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        Xn = self._norm(X)
        self.module.eval()
        with torch.no_grad():
            t = torch.from_numpy(Xn).float()
            out = self.module(t).view(-1).cpu().numpy()
        # Enforce V <= 0 only at predict time (training is unconstrained).
        out = np.minimum(out, 0.0)
        return out.astype(np.float32, copy=False)

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)


def _save_nn_predictor(path: Path, predictor: NNVPredictor) -> None:
    import torch
    state = {
        "state_dict": predictor.module.state_dict(),
        "feat_mean": predictor.feat_mean,
        "feat_std": predictor.feat_std,
        "in_dim": predictor.in_dim,
        "hidden_dims": list(predictor.hidden_dims),
        "dropout": predictor.dropout,
    }
    torch.save(state, str(path))


def _load_nn_predictor(path: Path) -> NNVPredictor:
    import torch
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    module = _build_nn_v(int(state["in_dim"]), tuple(state["hidden_dims"]), float(state["dropout"]))
    module.load_state_dict(state["state_dict"])
    module.eval()
    return NNVPredictor(
        module=module,
        feat_mean=state["feat_mean"],
        feat_std=state["feat_std"],
        in_dim=int(state["in_dim"]),
        hidden_dims=tuple(state["hidden_dims"]),
        dropout=float(state["dropout"]),
    )


def _train_nn_v(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    cfg: NNVConfig,
    init_state: dict | None = None,
) -> tuple[NNVPredictor, dict, dict]:
    """Fit an NN V regressor and return (predictor, train_metrics, eval_metrics)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cpu"  # box has no GPU
    in_dim = int(X_train.shape[1])

    # Normalization stats from train data
    feat_mean = X_train.astype(np.float64).mean(axis=0).astype(np.float32)
    feat_std = X_train.astype(np.float64).std(axis=0).astype(np.float32)
    feat_std_safe = np.where(feat_std > 1e-9, feat_std, 1.0)

    Xn_train = ((X_train.astype(np.float32) - feat_mean) / feat_std_safe).astype(np.float32)
    Xn_eval = ((X_eval.astype(np.float32) - feat_mean) / feat_std_safe).astype(np.float32)
    yt = y_train.astype(np.float32)
    ye = y_eval.astype(np.float32)

    # Sample weights
    tail_w = 1.0 + float(cfg.v_tail_weight_scale) * np.minimum(1.0, np.abs(yt) / 3.0)
    tail_w = tail_w.astype(np.float32)

    # Build module
    module = _build_nn_v(in_dim, cfg.v_hidden_dims, cfg.v_dropout)
    if init_state is not None:
        try:
            module.load_state_dict(init_state)
        except Exception:
            pass
    module.to(device)
    n_params = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    print(f"[nnv] V model params: {n_params}", flush=True)

    optim = torch.optim.AdamW(module.parameters(), lr=float(cfg.v_lr),
                              weight_decay=float(cfg.v_weight_decay))

    # Mini-batch loop on numpy index permutation to avoid copying full tensor.
    n = int(Xn_train.shape[0])
    bs = int(cfg.v_batch_size)
    rng = np.random.default_rng(int(cfg.seed))
    epochs = int(cfg.v_epochs)
    best_eval_max = float("inf")
    best_state = None
    log_every = max(1, epochs // 10)

    for ep in range(epochs):
        module.train()
        perm = rng.permutation(n)
        ep_loss = 0.0
        n_seen = 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = torch.from_numpy(Xn_train[idx]).to(device)
            yb = torch.from_numpy(yt[idx]).to(device)
            wb = torch.from_numpy(tail_w[idx]).to(device)
            v_hat = module(xb).view(-1)
            # Soft penalty for violating V <= 0
            penalty = torch.relu(v_hat).pow(2).mean()
            err = v_hat - yb
            loss = (wb * err * err).mean() + 0.1 * penalty
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            optim.step()
            ep_loss += float(loss.item()) * float(idx.shape[0])
            n_seen += int(idx.shape[0])
        ep_loss /= max(1, n_seen)

        # Eval
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            module.eval()
            with torch.no_grad():
                p_train = []
                for s in range(0, n, bs):
                    xb = torch.from_numpy(Xn_train[s:s + bs]).to(device)
                    v_hat = module(xb).view(-1)
                    v_hat = torch.minimum(v_hat, torch.zeros_like(v_hat))
                    p_train.append(v_hat.cpu().numpy())
                p_train_arr = np.concatenate(p_train)
                p_eval = []
                for s in range(0, Xn_eval.shape[0], bs):
                    xb = torch.from_numpy(Xn_eval[s:s + bs]).to(device)
                    v_hat = module(xb).view(-1)
                    v_hat = torch.minimum(v_hat, torch.zeros_like(v_hat))
                    p_eval.append(v_hat.cpu().numpy())
                p_eval_arr = np.concatenate(p_eval)
            e_train = np.abs(p_train_arr - yt)
            e_eval = np.abs(p_eval_arr - ye)
            print(
                f"[nnv] ep={ep+1}/{epochs} loss={ep_loss:.5f} | "
                f"train mse={float(np.mean(e_train**2)):.5f} p95={float(np.percentile(e_train,95)):.4f} "
                f"max={float(e_train.max()):.4f} | "
                f"eval mse={float(np.mean(e_eval**2)):.5f} p95={float(np.percentile(e_eval,95)):.4f} "
                f"max={float(e_eval.max()):.4f}",
                flush=True,
            )
            cur_eval_max = float(e_eval.max())
            if cur_eval_max < best_eval_max:
                best_eval_max = cur_eval_max
                best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    # Restore best state
    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    # Final metrics
    bs2 = bs
    with torch.no_grad():
        p_train = []
        for s in range(0, n, bs2):
            xb = torch.from_numpy(Xn_train[s:s + bs2]).to(device)
            v_hat = module(xb).view(-1)
            v_hat = torch.minimum(v_hat, torch.zeros_like(v_hat))
            p_train.append(v_hat.cpu().numpy())
        p_train_arr = np.concatenate(p_train)
        p_eval = []
        for s in range(0, Xn_eval.shape[0], bs2):
            xb = torch.from_numpy(Xn_eval[s:s + bs2]).to(device)
            v_hat = module(xb).view(-1)
            v_hat = torch.minimum(v_hat, torch.zeros_like(v_hat))
            p_eval.append(v_hat.cpu().numpy())
        p_eval_arr = np.concatenate(p_eval)
    e_train = np.abs(p_train_arr - yt)
    e_eval = np.abs(p_eval_arr - ye)

    train_metrics = {
        "mse": float(np.mean(e_train ** 2)),
        "rmse": float(math.sqrt(float(np.mean(e_train ** 2)))),
        "mae": float(e_train.mean()),
        "p50_abs_error": float(np.percentile(e_train, 50)),
        "p95_abs_error": float(np.percentile(e_train, 95)),
        "max_abs_error": float(e_train.max()),
    }
    eval_metrics = {
        "mse": float(np.mean(e_eval ** 2)),
        "rmse": float(math.sqrt(float(np.mean(e_eval ** 2)))),
        "mae": float(e_eval.mean()),
        "p50_abs_error": float(np.percentile(e_eval, 50)),
        "p95_abs_error": float(np.percentile(e_eval, 95)),
        "max_abs_error": float(e_eval.max()),
    }
    predictor = NNVPredictor(
        module=module,
        feat_mean=feat_mean,
        feat_std=feat_std,
        in_dim=in_dim,
        hidden_dims=cfg.v_hidden_dims,
        dropout=cfg.v_dropout,
    )
    return predictor, train_metrics, eval_metrics


# ---------------------------------------------------------------------------
# Helpers (segment max etc) — duplicated from q_learning_driver to avoid coupling
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def neural_v_run(cfg: NNVConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    feat_path_child = cfg.cached_run_dir / "child_features.npy"
    meta_path = cfg.cached_run_dir / "child_meta.npy"
    split_path = cfg.cached_run_dir / "split_indices.json"
    parent_idx_path = cfg.cached_run_dir / "parent_index.npz"
    parent_feat_path = cfg.fqi_run_dir / "parent_features.npy"
    per_row_input_path = cfg.fqi_run_dir / "per_row_input.npy"

    for p in (feat_path_child, meta_path, split_path, parent_idx_path, parent_feat_path, per_row_input_path):
        if not p.exists():
            raise FileNotFoundError(f"missing required artifact: {p}")

    si = json.loads(split_path.read_text())
    train_indices = np.asarray([int(x) for x in si["train_indices"]], dtype=np.int64)
    eval_indices = np.asarray([int(x) for x in si["eval_indices"]], dtype=np.int64)
    n_parents = max(int(train_indices.max()), int(eval_indices.max())) + 1
    print(f"[nnv] n_parents={n_parents}, train={len(train_indices)}, eval={len(eval_indices)}", flush=True)

    meta = np.load(meta_path, mmap_mode="r")
    parent_state_id_per_row = meta[:, 0].astype(np.int64)
    rewards_per_row = meta[:, 3].astype(np.float32)
    discounts_per_row = meta[:, 4].astype(np.float32)
    n_rows = int(meta.shape[0])
    print(f"[nnv] n_rows={n_rows}", flush=True)

    z = np.load(parent_idx_path)
    order = z["order"].astype(np.int64)
    parent_ids = z["parent_ids"].astype(np.int64)
    offsets = z["offsets"].astype(np.int64)

    parent_features = np.load(parent_feat_path, mmap_mode="r")
    Dp = int(parent_features.shape[1])
    print(f"[nnv] parent feat dim={Dp}", flush=True)
    X_per_row = np.load(per_row_input_path, mmap_mode="r")

    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib

    bootstrap_shrink = float(cfg.bootstrap_shrink)
    v_prev_path: str | None = None
    v_prev: NNVPredictor | None = None

    for vi in range(1, int(cfg.num_versions) + 1):
        t_iter = time.time()
        model_dir = cfg.output_dir / f"Model_Version{vi}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Compute Q targets per row ----------
        t0 = time.time()
        if v_prev is None:
            v_at_child = np.zeros(n_rows, dtype=np.float32)
        else:
            X_child_mm = np.load(feat_path_child, mmap_mode="r")
            v_at_child = np.empty(n_rows, dtype=np.float32)
            chunk = int(cfg.target_chunk_size)
            for s in range(0, n_rows, chunk):
                e = min(n_rows, s + chunk)
                block = np.ascontiguousarray(X_child_mm[s:e])
                v_at_child[s:e] = v_prev.predict(block)
            del X_child_mm
            np.minimum(v_at_child, 0.0, out=v_at_child)
        targets_per_row = rewards_per_row + (bootstrap_shrink * discounts_per_row) * v_at_child
        np.minimum(targets_per_row, 0.0, out=targets_per_row)

        # Train/eval split per row by parent_state_id membership
        train_set = set(int(x) for x in train_indices)
        is_train_row = np.fromiter(
            (int(p) in train_set for p in parent_state_id_per_row),
            dtype=bool, count=n_rows,
        )
        train_row_idx = np.where(is_train_row)[0]
        eval_row_idx = np.where(~is_train_row)[0]
        print(
            f"[nnv] V{vi}: target compute {time.time()-t0:.1f}s, "
            f"train_rows={len(train_row_idx)}, eval_rows={len(eval_row_idx)}",
            flush=True,
        )

        # ---------- Train Q (HGB) on per-row joined features ----------
        t0 = time.time()
        Q_train_X = X_per_row[train_row_idx].astype(np.float32, copy=False)
        Q_train_y = targets_per_row[train_row_idx].astype(np.float32, copy=False)
        Q_eval_X = X_per_row[eval_row_idx].astype(np.float32, copy=False)
        Q_eval_y = targets_per_row[eval_row_idx].astype(np.float32, copy=False)

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
        q_train_pred = q_model.predict(Q_train_X)
        q_eval_pred = q_model.predict(Q_eval_X)
        e = np.abs(q_eval_pred - Q_eval_y)
        print(
            f"[nnv] V{vi}: Q trained in {time.time()-t0:.1f}s | Q eval p95={np.percentile(e,95):.3f} "
            f"max={e.max():.3f}",
            flush=True,
        )
        del Q_train_X, Q_eval_X
        gc.collect()

        # ---------- V_target per parent = max_a Q(s, a) ----------
        t0 = time.time()
        all_q = np.empty(n_rows, dtype=np.float32)
        chunk = int(cfg.target_chunk_size)
        for s in range(0, n_rows, chunk):
            e = min(n_rows, s + chunk)
            block = np.ascontiguousarray(X_per_row[s:e])
            all_q[s:e] = q_model.predict(block).astype(np.float32, copy=False)
        np.minimum(all_q, 0.0, out=all_q)
        all_q_sorted = all_q[order]
        v_n_target_per_parent = _segment_max(all_q_sorted, offsets)
        del all_q, all_q_sorted
        gc.collect()
        print(f"[nnv] V{vi}: V_target (max_a Q) built in {time.time()-t0:.1f}s", flush=True)

        # ---------- Train NN V on parent_features -> V_target ----------
        v_target_by_psid = {int(parent_ids[i]): float(v_n_target_per_parent[i])
                            for i in range(parent_ids.shape[0])}
        train_psids = train_indices
        eval_psids = eval_indices
        V_train_X = parent_features[train_psids]
        V_train_y = np.asarray([v_target_by_psid.get(int(p), 0.0) for p in train_psids], dtype=np.float32)
        V_eval_X = parent_features[eval_psids]
        V_eval_y = np.asarray([v_target_by_psid.get(int(p), 0.0) for p in eval_psids], dtype=np.float32)

        # Warm-start NN from previous V state if present
        init_state = None
        if v_prev is not None:
            init_state = {k: v.detach().cpu().clone() for k, v in v_prev.module.state_dict().items()}

        t0 = time.time()
        predictor, v_train_metrics, v_eval_metrics = _train_nn_v(
            X_train=V_train_X.astype(np.float32),
            y_train=V_train_y,
            X_eval=V_eval_X.astype(np.float32),
            y_eval=V_eval_y,
            cfg=cfg,
            init_state=init_state,
        )
        print(f"[nnv] V{vi}: NN V trained in {time.time()-t0:.1f}s", flush=True)
        print(
            f"[nnv] V{vi}: V eval mse={v_eval_metrics['mse']:.5f} "
            f"p95={v_eval_metrics['p95_abs_error']:.4f} "
            f"max={v_eval_metrics['max_abs_error']:.4f}",
            flush=True,
        )

        # Save
        v_path = model_dir / "v_model.pt"
        _save_nn_predictor(v_path, predictor)
        v_prev = predictor
        v_prev_path = str(v_path)

        # CSV outputs
        v_train_pred = predictor.predict(V_train_X.astype(np.float32))
        v_eval_pred = predictor.predict(V_eval_X.astype(np.float32))
        with (model_dir / "v_train_results.csv").open("w") as f:
            f.write("sample_number,model_prediction,true_MCTS_DNN_value\n")
            for i, p in enumerate(train_psids):
                f.write(f"{int(p)},{float(v_train_pred[i]):.6f},{float(V_train_y[i]):.6f}\n")
        with (model_dir / "v_eval_results.csv").open("w") as f:
            f.write("sample_number,model_prediction,true_MCTS_DNN_value\n")
            for i, p in enumerate(eval_psids):
                f.write(f"{int(p)},{float(v_eval_pred[i]):.6f},{float(V_eval_y[i]):.6f}\n")
        bellman_logger.write_summary_csv(
            cfg.output_dir / f"version_{vi-1}_to_{vi}.csv",
            [
                {"source_model_version": vi - 1, "target_model_version": vi, "split": "train",
                 "model_version": vi, **v_train_metrics},
                {"source_model_version": vi - 1, "target_model_version": vi, "split": "eval",
                 "model_version": vi, **v_eval_metrics},
            ],
        )
        # Save Q joblib too for debugging
        try:
            joblib.dump(q_model, model_dir / "q_model.joblib")
        except Exception:
            pass

        print(
            f"[nnv] V{vi}: ITERATION DONE in {time.time()-t_iter:.1f}s | "
            f"V eval p95={v_eval_metrics['p95_abs_error']:.4f} max={v_eval_metrics['max_abs_error']:.4f}",
            flush=True,
        )


def parse_args() -> NNVConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--cached-run-dir", required=True)
    p.add_argument("--fqi-run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-versions", type=int, default=25)
    p.add_argument("--bootstrap-shrink", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--q-hgb-max-iter", type=int, default=400)
    p.add_argument("--q-hgb-max-leaf-nodes", type=int, default=31)
    p.add_argument("--q-hgb-lr", type=float, default=0.05)
    p.add_argument("--v-hidden-dims", type=str, default="256,128,64")
    p.add_argument("--v-lr", type=float, default=1e-3)
    p.add_argument("--v-epochs", type=int, default=120)
    p.add_argument("--v-batch-size", type=int, default=4096)
    p.add_argument("--v-tail-weight-scale", type=float, default=50.0)
    p.add_argument("--v-weight-decay", type=float, default=1e-5)
    p.add_argument("--v-dropout", type=float, default=0.0)
    args = p.parse_args()
    return NNVConfig(
        cached_run_dir=Path(args.cached_run_dir).expanduser(),
        fqi_run_dir=Path(args.fqi_run_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        num_versions=int(args.num_versions),
        bootstrap_shrink=float(args.bootstrap_shrink),
        seed=int(args.seed),
        q_hgb_max_iter=int(args.q_hgb_max_iter),
        q_hgb_max_leaf_nodes=int(args.q_hgb_max_leaf_nodes),
        q_hgb_learning_rate=float(args.q_hgb_lr),
        v_hidden_dims=tuple(int(x) for x in str(args.v_hidden_dims).split(",") if x.strip()),
        v_lr=float(args.v_lr),
        v_epochs=int(args.v_epochs),
        v_batch_size=int(args.v_batch_size),
        v_tail_weight_scale=float(args.v_tail_weight_scale),
        v_weight_decay=float(args.v_weight_decay),
        v_dropout=float(args.v_dropout),
    )


def main() -> None:
    neural_v_run(parse_args())


if __name__ == "__main__":
    main()
