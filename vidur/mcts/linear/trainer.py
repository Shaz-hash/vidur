from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .config import LinearPipelineConfig
from .features import compute_feature_stats
from .model import LinearValueModel


@dataclass(frozen=True)
class TrainResult:
    epoch_rows: List[Dict[str, float]]
    final_metrics: Dict[str, float]


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _eval_model(
    model: LinearValueModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if x.shape[0] == 0:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}

    model.eval()
    preds: List[np.ndarray] = []

    with torch.no_grad():
        n = x.shape[0]
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device=device, dtype=torch.float32)
            yb = model(xb).detach().cpu().numpy().astype(np.float32)
            preds.append(yb)

    pred = np.concatenate(preds, axis=0)
    target = y.astype(np.float32)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    r2 = float(_r2_score(target, pred))
    return {"mse": mse, "mae": mae, "r2": r2}


def train_value_model(
    *,
    cfg: LinearPipelineConfig,
    model: LinearValueModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    device: torch.device,
) -> TrainResult:
    if train_x.ndim != 2 or train_x.shape[1] != model.num_features:
        raise ValueError(f"train_x shape mismatch: {train_x.shape}")
    if train_y.ndim != 1:
        raise ValueError(f"train_y shape mismatch: {train_y.shape}")

    mean, std = compute_feature_stats(train_x)
    model = model.to(device)
    model.set_feature_stats(
        torch.from_numpy(mean).to(device=device, dtype=torch.float32),
        torch.from_numpy(std).to(device=device, dtype=torch.float32),
    )

    opt_name = str(cfg.training.optimizer).strip().lower()
    if opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )
    else:
        opt = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )

    batch_size = int(cfg.training.batch_size)
    epochs = int(cfg.training.epochs)

    epoch_rows: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        n = train_x.shape[0]
        perm = np.random.permutation(n)

        loss_sum = 0.0
        seen = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(train_x[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(train_y[idx]).to(device=device, dtype=torch.float32)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bsz = int(idx.shape[0])
            loss_sum += float(loss.item()) * bsz
            seen += bsz

        train_mse = float(loss_sum / max(1, seen))
        eval_metrics = _eval_model(model, eval_x, eval_y, device=device, batch_size=batch_size)
        train_metrics = _eval_model(model, train_x, train_y, device=device, batch_size=batch_size)

        epoch_rows.append(
            {
                "epoch": float(epoch),
                "train_mse": float(train_mse),
                "train_mae": float(train_metrics["mae"]),
                "train_r2": float(train_metrics["r2"]),
                "eval_mse": float(eval_metrics["mse"]),
                "eval_mae": float(eval_metrics["mae"]),
                "eval_r2": float(eval_metrics["r2"]),
            }
        )

    final = epoch_rows[-1] if epoch_rows else {
        "epoch": 0.0,
        "train_mse": 0.0,
        "train_mae": 0.0,
        "train_r2": 0.0,
        "eval_mse": 0.0,
        "eval_mae": 0.0,
        "eval_r2": 0.0,
    }

    return TrainResult(epoch_rows=epoch_rows, final_metrics=final)
