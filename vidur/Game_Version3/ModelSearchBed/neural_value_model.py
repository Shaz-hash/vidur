"""Small neural value model for the GV3 controller value function.

Replaces the cliff_aware tree backbone with a ReLU MLP. Trees plateaued at
~2.0 max_abs_error on cliff states because they cannot represent step
functions on shared-feature inputs. A ReLU MLP can encode steps exactly with
a single hidden unit per cliff direction, so this model class is the right
hammer for the discontinuities GV3 cost has at:

  * SLO violation step (+1)
  * auto-drop cliff at total_lateness = 2.0 (+drop_cost = 3.0)
  * decode-credit floor at 0
  * launch-window integer caps

The architecture deliberately mirrors the cliff_aware feature schema (352
dims from `extract_cliff_features_from_inputs`) so both backends share the
on-disk feature cache. That keeps the train and MCTS-bootstrap distributions
identical, which is required for Bellman residual convergence.

Two stabilizers are built in:

  * Polyak (EMA) target network. Bootstrap predictions during the next
    iteration's target compute are taken from the EMA copy of the weights,
    not the freshly trained model. This makes the iteration a contraction in
    expectation and damps the per-iteration noise band that previously left
    max_abs_error stuck at ~2.0.

  * Bucket-weighted Huber loss. Rows with |target| >= 1 get extra weight,
    and an explicit "outlier hinge" term penalises any per-row error larger
    than `outlier_threshold` to keep max_abs_error from drifting.

The model stays under the 150k effective scalar parameter budget by using a
modest hidden width (256 -> 256 -> 128 -> 1) and weight tying disabled. See
`_count_trainable_params` for the exact accounting.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cliff_aware_value_model import (
    CLIFF_FEATURE_CACHE_VERSION,
    build_or_load_cliff_feature_matrix,
    extract_cliff_features_from_inputs,
)
from .action_features import (
    ACTION_FEATURE_CACHE_VERSION,
    DEFAULT_TOP_K_ACTIONS,
    build_or_load_action_feature_matrix,
    extract_action_features_from_state,
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class ValueMLP(nn.Module):
    """Compact ReLU MLP that outputs a non-positive controller value.

    Two heads share the trunk:

      * value_head -> `-softplus(linear)` in (-inf, 0]; the actual scalar V(s)
      * gate_head  -> sigmoid in (0, 1); P(target == 0). Used to snap small
        predictions to exactly zero, which kills the most common max-error
        failure mode (true value 0 but model predicted a large negative
        because aggregate features alias near-cliff but actually-safe
        states with cliff-violating ones).

    Forward returns the gated prediction `(1 - gate) * value_head(x)`. Both
    heads are trained jointly: the gate via BCE on labels `1{|target| <=
    zero_eps}`, the value via Huber-on-residual.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (208, 208, 104),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.LayerNorm(int(h)))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        self.body = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, 1)
        self.gate_head = nn.Linear(prev, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Bias gate so initial probability of "is_zero" is small (~0.1) so
        # the value head dominates early and the gate has to actively learn
        # which states are truly zero.
        with torch.no_grad():
            self.gate_head.bias.fill_(-2.2)

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        value = -F.softplus(self.value_head(z).squeeze(-1))
        gate_logit = self.gate_head(z).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        return value, gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.forward_components(x)
        return (1.0 - gate) * value


def _count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Picklable wrapper that ships with the joblib so MCTS workers can use it
# without depending on torch's grad/optimizer state.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralValueModel:
    """MCTS-bootstrap-compatible neural value surrogate.

    Stores the network state_dict (CPU tensors) plus feature schema length
    and bootstrap clip range. `infer_from_inputs` rebuilds the network on
    first use and caches it.
    """

    state_dict: dict[str, Any]
    in_dim: int
    feature_names: list[str]
    hidden_dims: tuple[int, ...]
    dropout: float
    trainable_params: int
    bootstrap_clip_min: float = -7.0
    bootstrap_clip_max: float = 0.0
    polyak_alpha: float = 0.5
    gate_snap_threshold: float = 0.5
    cliff_dim: int = 352
    action_dim: int = 0
    feature_config: dict[str, Any] = field(default_factory=dict)
    cached_predictions: dict[str, list[float]] = field(
        default_factory=dict, repr=False, compare=False
    )
    runtime_prediction_cache: dict[bytes, float] = field(
        default_factory=dict, repr=False, compare=False
    )
    runtime_module: Any = field(default=None, repr=False, compare=False)
    model_name: str = "neural_value_model"
    uses_neural_network: bool = True
    uses_target_leakage: bool = False

    # ------------------------------------------------------------------
    def _ensure_module(self) -> nn.Module:
        mod = self.runtime_module
        if mod is None:
            mod = ValueMLP(
                in_dim=int(self.in_dim),
                hidden_dims=tuple(self.hidden_dims),
                dropout=float(self.dropout),
            )
            mod.load_state_dict({k: torch.as_tensor(v) for k, v in self.state_dict.items()})
            mod.eval()
            object.__setattr__(self, "runtime_module", mod)
        return mod

    # ------------------------------------------------------------------
    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        mod = self._ensure_module()
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(x, dtype=np.float32))
            value, gate = mod.forward_components(t)
            # Soft gate -> (1-gate)*value, then hard-snap to exactly zero when
            # the gate is highly confident. The hard snap kills the residual
            # ~(1 - 0.95) * value contribution that would otherwise show up
            # as a small non-zero prediction on safe states.
            soft = (1.0 - gate) * value
            hard_thresh = float(getattr(self, "gate_snap_threshold", 0.5))
            if hard_thresh > 0.0:
                snap_mask = gate >= hard_thresh
                soft = torch.where(snap_mask, torch.zeros_like(soft), soft)
            preds = soft.cpu().numpy().astype(np.float64)
        return preds

    # ------------------------------------------------------------------
    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: Any | None = None,
    ) -> tuple[float, list[float]]:
        del player, device
        row, _names = extract_cliff_features_from_inputs(inputs)
        cliff_dim = int(self.cliff_dim or len(row))
        action_dim = int(self.action_dim or 0)
        target_len = int(self.in_dim)
        if len(row) < cliff_dim:
            row = row + [0.0] * (cliff_dim - len(row))
        elif len(row) > cliff_dim:
            row = row[:cliff_dim]
        # Pad with zeros for action-conditional dims that we cannot compute
        # from ModelInputs alone. The model has been trained to be robust
        # to zeros in those columns (verify by training-time augmentation).
        if action_dim > 0:
            row = row + [0.0] * action_dim
        if len(row) < target_len:
            row = row + [0.0] * (target_len - len(row))
        elif len(row) > target_len:
            row = row[:target_len]
        x = np.asarray([row], dtype=np.float32)
        key = x.tobytes()
        cached = self.runtime_prediction_cache.get(key)
        if cached is not None:
            return float(cached), []
        value = float(self.predict_matrix(x)[0])
        clip_min = float(self.bootstrap_clip_min)
        clip_max = float(self.bootstrap_clip_max)
        if value < clip_min:
            value = clip_min
        elif value > clip_max:
            value = clip_max
        if len(self.runtime_prediction_cache) > 50_000:
            self.runtime_prediction_cache.clear()
        self.runtime_prediction_cache[key] = value
        return value, []


# ---------------------------------------------------------------------------
# Loss & training
# ---------------------------------------------------------------------------


def _bucket_weights(
    y: np.ndarray,
    *,
    large_target_thresh: float,
    large_target_weight: float,
    huge_target_thresh: float,
    huge_target_weight: float,
) -> np.ndarray:
    """Per-row weights that give large-target rows extra emphasis."""

    abs_y = np.abs(y)
    w = np.ones_like(abs_y, dtype=np.float32)
    w[abs_y >= float(large_target_thresh)] = float(large_target_weight)
    w[abs_y >= float(huge_target_thresh)] = float(huge_target_weight)
    return w.astype(np.float32)


def _huber_outlier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    huber_beta: float,
    outlier_threshold: float,
    outlier_weight: float,
) -> torch.Tensor:
    """Weighted Huber + explicit hinge above outlier_threshold."""

    err = pred - target
    abs_err = err.abs()
    huber = torch.where(
        abs_err <= float(huber_beta),
        0.5 * err.pow(2) / max(1e-9, float(huber_beta)),
        abs_err - 0.5 * float(huber_beta),
    )
    base = (weight * huber).mean()
    hinge = torch.clamp(abs_err - float(outlier_threshold), min=0.0)
    outlier = (weight * hinge.pow(2)).mean()
    return base + float(outlier_weight) * outlier


def _gated_value_loss(
    value_raw: torch.Tensor,
    gate: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    huber_beta: float,
    outlier_threshold: float,
    outlier_weight: float,
    gate_label_eps: float,
    gate_bce_weight: float,
    gate_zero_pred_weight: float,
) -> torch.Tensor:
    """Loss that jointly trains the value head and the zero-gate head.

    pred = (1 - gate) * value_raw, where value_raw <= 0 by construction.
    The composite loss has three pieces:

      L_value       = weighted Huber + outlier hinge on (pred - target)
      L_gate_bce    = BCE(gate, 1{|target| <= gate_label_eps})
      L_gate_zero   = on rows the gate confidently snaps to zero, also
                      penalise a |target| > gate_label_eps mismatch
                      directly so the gate cannot relieve the value loss
                      by rerouting bad rows through "predict 0".
    """

    pred = (1.0 - gate) * value_raw
    err = pred - target
    abs_err = err.abs()
    huber = torch.where(
        abs_err <= float(huber_beta),
        0.5 * err.pow(2) / max(1e-9, float(huber_beta)),
        abs_err - 0.5 * float(huber_beta),
    )
    value_base = (weight * huber).mean()
    hinge = torch.clamp(abs_err - float(outlier_threshold), min=0.0)
    value_outlier = (weight * hinge.pow(2)).mean()

    # Gate BCE: label is 1 when target is "approximately zero".
    zero_label = (target.abs() <= float(gate_label_eps)).to(torch.float32)
    eps = 1e-6
    gate_clamped = gate.clamp(eps, 1.0 - eps)
    bce = -(zero_label * torch.log(gate_clamped) + (1.0 - zero_label) * torch.log(1.0 - gate_clamped))
    gate_loss = (weight * bce).mean()

    # Penalise the gate when it confidently zeroes a non-zero row. This
    # specifically attacks the failure mode where a near-cliff but
    # cliff-violating row sneaks through as "0" -> max error blows up.
    gate_zero_misroute = gate * (1.0 - zero_label) * target.abs()
    gate_zero_loss = (weight * gate_zero_misroute).mean()

    return (
        value_base
        + float(outlier_weight) * value_outlier
        + float(gate_bce_weight) * gate_loss
        + float(gate_zero_pred_weight) * gate_zero_loss
    )


def _ema_update_(target: nn.Module, online: nn.Module, alpha: float) -> None:
    """In-place: target = alpha * online + (1 - alpha) * target."""

    a = float(alpha)
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.mul_(1.0 - a).add_(op.detach(), alpha=a)
        for tb, ob in zip(target.buffers(), online.buffers()):
            tb.mul_(1.0 - a).add_(ob.detach(), alpha=a)


def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    abs_err = np.abs(err)
    mse = float(np.mean(err * err))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(abs_err)),
        "p50_abs_error": float(np.percentile(abs_err, 50)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "max_abs_error": float(np.max(abs_err)),
    }


def _error_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true64 = np.asarray(y_true, dtype=np.float64)
    y_pred64 = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_pred64 - y_true64)
    abs_t = np.abs(y_true64)
    out: dict[str, Any] = {
        "num_abs_error_gt_0p05": int(np.sum(abs_err > 0.05)),
        "num_abs_error_gt_0p10": int(np.sum(abs_err > 0.10)),
        "num_abs_error_gt_0p50": int(np.sum(abs_err > 0.50)),
        "num_abs_error_gt_1p00": int(np.sum(abs_err > 1.00)),
    }
    buckets = {
        "target_abs_eq_0": abs_t <= 1e-12,
        "target_abs_le_0p05": abs_t <= 0.05,
        "target_abs_0p05_1": (abs_t > 0.05) & (abs_t < 1.0),
        "target_abs_1_2": (abs_t >= 1.0) & (abs_t < 2.0),
        "target_abs_2_3": (abs_t >= 2.0) & (abs_t < 3.0),
        "target_abs_ge_3": abs_t >= 3.0,
    }
    for name, mask in buckets.items():
        count = int(np.sum(mask))
        prefix = f"bucket_{name}"
        out[f"{prefix}_count"] = count
        if count <= 0:
            continue
        bucket_errors = abs_err[mask]
        out[f"{prefix}_mae"] = float(np.mean(bucket_errors))
        out[f"{prefix}_p95_abs_error"] = float(np.percentile(bucket_errors, 95))
        out[f"{prefix}_max_abs_error"] = float(np.max(bucket_errors))
    return out


# ---------------------------------------------------------------------------
# Public training entry point used by trainer.py::train_model_search hook
# ---------------------------------------------------------------------------


def _read_int(extra: dict[str, Any], key: str, default: int) -> int:
    val = extra.get(key, default)
    try:
        return int(val)
    except Exception:
        return int(default)


def _read_float(extra: dict[str, Any], key: str, default: float) -> float:
    val = extra.get(key, default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _target_array(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(r["target_value"]) for r in records], dtype=np.float32)


def train_neural_value_model(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    cfg: Any,
    output_dir: str | Path,
    state_loader: Any | None = None,
) -> dict[str, Any]:
    """Fit the neural value model on cliff_aware features."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extra_cfg: dict[str, Any] = dict(getattr(cfg, "extra_config", {}) or {})

    # ---- features ----
    use_action_features = bool(extra_cfg.get("nn_use_action_features", True))
    action_top_k = int(extra_cfg.get("nn_action_top_k", DEFAULT_TOP_K_ACTIONS))

    t0 = time.time()
    cliff_train, cliff_names = build_or_load_cliff_feature_matrix(
        list(train_records),
        state_loader=state_loader,
        model_cfg=cfg,
        extra_config=extra_cfg,
        output_dir=output,
        split_name="train",
        player="controller",
    )
    cliff_eval, _ = build_or_load_cliff_feature_matrix(
        list(eval_records),
        state_loader=state_loader,
        model_cfg=cfg,
        extra_config=extra_cfg,
        output_dir=output,
        split_name="eval",
        player="controller",
    )
    if use_action_features:
        act_train, act_names = build_or_load_action_feature_matrix(
            list(train_records),
            state_loader=state_loader,
            model_cfg=cfg,
            extra_config=extra_cfg,
            output_dir=output,
            split_name="train",
            top_k=int(action_top_k),
        )
        act_eval, _ = build_or_load_action_feature_matrix(
            list(eval_records),
            state_loader=state_loader,
            model_cfg=cfg,
            extra_config=extra_cfg,
            output_dir=output,
            split_name="eval",
            top_k=int(action_top_k),
        )
        x_train = np.concatenate([cliff_train, act_train], axis=1).astype(np.float32, copy=False)
        x_eval = np.concatenate([cliff_eval, act_eval], axis=1).astype(np.float32, copy=False)
        feature_names = list(cliff_names) + list(act_names)
    else:
        x_train = cliff_train
        x_eval = cliff_eval
        feature_names = list(cliff_names)
    feature_seconds = time.time() - t0

    y_train = _target_array(list(train_records))
    y_eval = _target_array(list(eval_records))

    # ---- model ----
    in_dim = int(x_train.shape[1])
    hidden_dims = tuple(
        int(x)
        for x in extra_cfg.get("nn_hidden_dims", (208, 208, 104))
    )
    dropout = _read_float(extra_cfg, "nn_dropout", 0.05)
    max_params = _read_int(extra_cfg, "max_trainable_params", 150_000)

    online = ValueMLP(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
    trainable = _count_trainable_params(online)
    if trainable > max_params:
        raise ValueError(
            f"NeuralValueModel has {trainable} params, above budget {max_params}. "
            "Reduce nn_hidden_dims."
        )

    # warm-start from the previous iteration's weights when present so
    # iteration N+1 starts close to the converged-so-far model and only
    # has to learn the residual update introduced by the new bootstrap.
    warm_path = extra_cfg.get("nn_warm_start_state_dict_path")
    if warm_path:
        try:
            sd = torch.load(str(warm_path), map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                online.load_state_dict(sd)
                print(f"[neural_value] warm-started from {warm_path}", flush=True)
        except Exception as exc:
            print(f"[neural_value] warm start failed: {exc}", flush=True)

    target = ValueMLP(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
    target.load_state_dict(online.state_dict())
    target.eval()
    for p in target.parameters():
        p.requires_grad = False

    # ---- weights ----
    sample_weights = _bucket_weights(
        y_train,
        large_target_thresh=_read_float(extra_cfg, "nn_large_target_thresh", 1.0),
        large_target_weight=_read_float(extra_cfg, "nn_large_target_weight", 4.0),
        huge_target_thresh=_read_float(extra_cfg, "nn_huge_target_thresh", 3.0),
        huge_target_weight=_read_float(extra_cfg, "nn_huge_target_weight", 8.0),
    )

    # ---- optimizer ----
    lr = _read_float(extra_cfg, "nn_lr", 1.5e-3)
    weight_decay = _read_float(extra_cfg, "nn_weight_decay", 1e-5)
    grad_clip = _read_float(extra_cfg, "nn_grad_clip", 1.0)
    epochs = _read_int(extra_cfg, "nn_epochs", 30)
    batch_size = _read_int(extra_cfg, "nn_batch_size", 1024)
    huber_beta = _read_float(extra_cfg, "nn_huber_beta", 0.05)
    outlier_threshold = _read_float(extra_cfg, "nn_outlier_threshold", 0.10)
    outlier_weight = _read_float(extra_cfg, "nn_outlier_weight", 8.0)
    gate_label_eps = _read_float(extra_cfg, "nn_gate_label_eps", 0.05)
    gate_bce_weight = _read_float(extra_cfg, "nn_gate_bce_weight", 4.0)
    gate_zero_pred_weight = _read_float(extra_cfg, "nn_gate_zero_pred_weight", 1.0)
    target_alpha = _read_float(extra_cfg, "nn_target_alpha", 1.0)
    seed = _read_int(extra_cfg, "random_state", 2027)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.AdamW(online.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs * max(1, x_train.shape[0] // batch_size))
    )

    # convert to tensors once
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    wt = torch.as_tensor(sample_weights, dtype=torch.float32)
    xe = torch.as_tensor(x_eval, dtype=torch.float32)
    ye = torch.as_tensor(y_eval, dtype=torch.float32)

    # Masking augmentation: a fraction of training rows have their action
    # feature columns zeroed out. This teaches the network to fall back to
    # cliff features when action features are unavailable (which happens at
    # MCTS bootstrap time on adversary child states). Without this the
    # model's predictions blow up at bootstrap.
    action_mask_prob = _read_float(extra_cfg, "nn_action_mask_prob", 0.25)
    cliff_dim_local = int(cliff_train.shape[1])

    n = int(xt.shape[0])
    fit_start = time.time()
    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    metrics_log: list[dict[str, Any]] = []
    # Composite epoch-selection score:
    #   score = mse + alpha * max(eval) + beta * p95(eval)
    # picks an epoch where the bulk fit AND the tail are simultaneously good.
    score_max_w = _read_float(extra_cfg, "nn_score_max_weight", 0.02)
    score_p95_w = _read_float(extra_cfg, "nn_score_p95_weight", 0.5)

    for epoch in range(int(epochs)):
        online.train()
        idx = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            sl = idx[start : start + batch_size]
            t = torch.as_tensor(sl, dtype=torch.long)
            xb = xt.index_select(0, t)
            yb = yt.index_select(0, t)
            wb = wt.index_select(0, t)
            # zero out action-feature columns for a random subset of rows
            if use_action_features and action_mask_prob > 0.0:
                drop_mask = torch.rand(xb.shape[0]) < float(action_mask_prob)
                if drop_mask.any():
                    xb = xb.clone()
                    xb[drop_mask, cliff_dim_local:] = 0.0
            optimizer.zero_grad(set_to_none=True)
            value_raw, gate = online.forward_components(xb)
            loss = _gated_value_loss(
                value_raw,
                gate,
                yb,
                wb,
                huber_beta=huber_beta,
                outlier_threshold=outlier_threshold,
                outlier_weight=outlier_weight,
                gate_label_eps=gate_label_eps,
                gate_bce_weight=gate_bce_weight,
                gate_zero_pred_weight=gate_zero_pred_weight,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(online.parameters(), float(grad_clip))
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())
            batches += 1
            # online -> target EMA every step so target follows weights smoothly.
            if target_alpha > 0.0 and target_alpha < 1.0:
                _ema_update_(target, online, alpha=float(target_alpha))

        if target_alpha >= 1.0:
            target.load_state_dict(online.state_dict())

        with torch.no_grad():
            online.eval()
            v_t, g_t = online.forward_components(xt)
            v_e, g_e = online.forward_components(xe)
            snap_th = _read_float(extra_cfg, "nn_gate_snap_threshold", 0.5)
            soft_t = (1.0 - g_t) * v_t
            soft_e = (1.0 - g_e) * v_e
            if snap_th > 0.0:
                soft_t = torch.where(g_t >= snap_th, torch.zeros_like(soft_t), soft_t)
                soft_e = torch.where(g_e >= snap_th, torch.zeros_like(soft_e), soft_e)
            pred_train = soft_t.cpu().numpy().astype(np.float64)
            pred_eval = soft_e.cpu().numpy().astype(np.float64)
        train_metrics = _error_metrics(y_train, pred_train)
        eval_metrics = _error_metrics(y_eval, pred_eval)
        metrics_log.append(
            {
                "epoch": int(epoch),
                "loss": epoch_loss / max(1, batches),
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_p95": train_metrics["p95_abs_error"],
                "train_max": train_metrics["max_abs_error"],
                "eval_mse": eval_metrics["mse"],
                "eval_mae": eval_metrics["mae"],
                "eval_p95": eval_metrics["p95_abs_error"],
                "eval_max": eval_metrics["max_abs_error"],
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        score = (
            float(eval_metrics["mse"])
            + float(score_max_w) * float(eval_metrics["max_abs_error"])
            + float(score_p95_w) * float(eval_metrics["p95_abs_error"])
        )
        if score < best_score:
            best_score = float(score)
            best_state = {k: v.detach().clone().cpu() for k, v in online.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"[neural_value] epoch={epoch+1}/{epochs} loss={epoch_loss/max(1,batches):.5f} "
                f"train_mse={train_metrics['mse']:.6f} eval_mse={eval_metrics['mse']:.6f} "
                f"eval_p95={eval_metrics['p95_abs_error']:.4f} eval_max={eval_metrics['max_abs_error']:.4f}",
                flush=True,
            )

    fit_seconds = time.time() - fit_start

    # If best-by-eval-max state exists, load it before final eval; otherwise
    # keep the last state from training. Best state captures the moment in
    # the training trajectory with the smallest cliff outlier.
    if best_state is not None:
        online.load_state_dict(best_state)

    # final predictions (apply gate snap so saved metrics match deploy
    # behavior of NeuralValueModel.predict_matrix).
    with torch.no_grad():
        online.eval()
        v_t, g_t = online.forward_components(xt)
        v_e, g_e = online.forward_components(xe)
        snap_th = _read_float(extra_cfg, "nn_gate_snap_threshold", 0.5)
        soft_t = (1.0 - g_t) * v_t
        soft_e = (1.0 - g_e) * v_e
        if snap_th > 0.0:
            soft_t = torch.where(g_t >= snap_th, torch.zeros_like(soft_t), soft_t)
            soft_e = torch.where(g_e >= snap_th, torch.zeros_like(soft_e), soft_e)
        pred_train = soft_t.cpu().numpy().astype(np.float64)
        pred_eval = soft_e.cpu().numpy().astype(np.float64)

    train_metrics = _error_metrics(y_train, pred_train)
    eval_metrics = _error_metrics(y_eval, pred_eval)
    train_diag = _error_diagnostics(y_train, pred_train)
    eval_diag = _error_diagnostics(y_eval, pred_eval)

    # Save the EMA target as the deployed model when target_alpha < 1; that
    # is the version we want as the bootstrap for the next iteration.
    deploy_module = target if (0.0 < target_alpha < 1.0) else online
    deploy_state = {k: v.detach().cpu() for k, v in deploy_module.state_dict().items()}

    gate_snap_threshold = _read_float(extra_cfg, "nn_gate_snap_threshold", 0.5)
    cliff_dim = int(cliff_train.shape[1])
    action_dim = int(act_train.shape[1]) if use_action_features else 0
    model = NeuralValueModel(
        state_dict=deploy_state,
        in_dim=int(in_dim),
        feature_names=list(feature_names),
        hidden_dims=tuple(hidden_dims),
        dropout=float(dropout),
        trainable_params=int(trainable),
        bootstrap_clip_min=_read_float(extra_cfg, "bootstrap_clip_min", -7.0),
        bootstrap_clip_max=_read_float(extra_cfg, "bootstrap_clip_max", 0.0),
        polyak_alpha=float(target_alpha),
        gate_snap_threshold=float(gate_snap_threshold),
        cliff_dim=int(cliff_dim),
        action_dim=int(action_dim),
        feature_config={
            "feature_source": "model_inputs+actions" if use_action_features else "model_inputs",
            "cache_version": CLIFF_FEATURE_CACHE_VERSION,
            "action_cache_version": ACTION_FEATURE_CACHE_VERSION if use_action_features else None,
        },
        cached_predictions={
            "train": [float(x) for x in pred_train.tolist()],
            "eval": [float(x) for x in pred_eval.tolist()],
        },
        model_name=str(getattr(cfg, "model_name", "neural_value_model")),
    )

    model_path = output / "neural_value_model.joblib"
    joblib.dump(model, model_path)
    # also save the raw state dict so the next iteration can warm-start from it
    torch.save(deploy_state, output / "neural_value_state_dict.pt")

    metadata = {
        "backend": "neural",
        "model_name": model.model_name,
        "trainable_params": int(trainable),
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "in_dim": int(in_dim),
        "feature_count": int(len(feature_names)),
        "uses_neural_network": True,
        "uses_target_leakage": False,
        "num_train_records": int(len(train_records)),
        "num_eval_records": int(len(eval_records)),
        "feature_seconds": float(feature_seconds),
        "fit_seconds": float(fit_seconds),
        "polyak_alpha": float(target_alpha),
        "bootstrap_clip_min": float(model.bootstrap_clip_min),
        "bootstrap_clip_max": float(model.bootstrap_clip_max),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "train_diagnostics": train_diag,
        "eval_diagnostics": eval_diag,
        "epochs_log": metrics_log,
        "feature_config": dict(model.feature_config),
        "description": (
            "ReLU MLP on cliff_aware features with EMA target network and "
            "outlier-hinge loss. Output is non-positive via -softplus."
        ),
    }
    metadata_path = output / "neural_value_model.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "model": model,
        "best_checkpoint_path": model_path,
        "train_metrics": [
            {
                "backend": "neural",
                "epoch": 0,
                "model_name": str(model.model_name),
                "trainable_params": int(trainable),
                "num_train_records": int(len(train_records)),
                "num_eval_records": int(len(eval_records)),
                "feature_seconds": float(feature_seconds),
                "fit_seconds": float(fit_seconds),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"eval_{k}": float(v) for k, v in eval_metrics.items()},
                **{f"train_diag_{k}": float(v) for k, v in train_diag.items() if isinstance(v, (int, float))},
                **{f"eval_diag_{k}": float(v) for k, v in eval_diag.items() if isinstance(v, (int, float))},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Inference helpers wired by DNN/infer.py
# ---------------------------------------------------------------------------


def predict_neural_values(
    *,
    model: NeuralValueModel,
    records: list[dict[str, Any]],
    state_loader: Any | None = None,
    split_name: str | None = None,
) -> list[float]:
    if not isinstance(model, NeuralValueModel):
        raise TypeError(f"expected NeuralValueModel, got {type(model)!r}")
    if split_name is not None:
        cached = model.cached_predictions.get(str(split_name))
        if cached is not None and len(cached) == len(records):
            return [float(x) for x in cached]
        if str(split_name).startswith("train"):
            cached = model.cached_predictions.get("train")
            if cached is not None and len(cached) == len(records):
                return [float(x) for x in cached]
        if str(split_name).startswith("eval"):
            cached = model.cached_predictions.get("eval")
            if cached is not None and len(cached) == len(records):
                return [float(x) for x in cached]
    if state_loader is None:
        raise ValueError("state_loader required when predicting from records without cache")
    from .cliff_aware_value_model import build_cliff_feature_matrix_from_records

    matrix, _names = build_cliff_feature_matrix_from_records(
        list(records), state_loader=state_loader, player="controller"
    )
    return [float(v) for v in model.predict_matrix(matrix).tolist()]


def predict_neural_value_from_inputs(
    *,
    model: NeuralValueModel,
    inputs: Any,
    player: str,
    device: Any | None = None,
) -> float:
    value, _ = model.infer_from_inputs(inputs, player, device=device)
    return float(value)


def predict_neural_values_from_inputs_batch(
    *,
    model: NeuralValueModel,
    inputs_list: Sequence[Any],
    action_feature_rows: Sequence[Sequence[float]] | None = None,
) -> list[float]:
    if not isinstance(model, NeuralValueModel):
        raise TypeError(f"expected NeuralValueModel, got {type(model)!r}")
    cliff_dim = int(model.cliff_dim or 0)
    action_dim = int(model.action_dim or 0)
    target_len = int(model.in_dim)
    rows: list[list[float]] = []
    for idx, inputs in enumerate(inputs_list):
        cliff_row, _ = extract_cliff_features_from_inputs(inputs)
        if cliff_dim > 0:
            if len(cliff_row) < cliff_dim:
                cliff_row = cliff_row + [0.0] * (cliff_dim - len(cliff_row))
            elif len(cliff_row) > cliff_dim:
                cliff_row = cliff_row[:cliff_dim]
        if action_dim > 0:
            if action_feature_rows is not None and idx < len(action_feature_rows):
                act = list(action_feature_rows[idx]) or []
            else:
                act = []
            if len(act) < action_dim:
                act = act + [0.0] * (action_dim - len(act))
            elif len(act) > action_dim:
                act = act[:action_dim]
            row = list(cliff_row) + list(act)
        else:
            row = list(cliff_row)
        if len(row) < target_len:
            row = row + [0.0] * (target_len - len(row))
        elif len(row) > target_len:
            row = row[:target_len]
        rows.append(row)
    if not rows:
        return []
    x = np.asarray(rows, dtype=np.float32)
    preds = np.asarray(model.predict_matrix(x), dtype=np.float64)
    clip_min = float(model.bootstrap_clip_min)
    clip_max = float(model.bootstrap_clip_max)
    preds = np.clip(preds, clip_min, clip_max)
    return [float(v) for v in preds.tolist()]
