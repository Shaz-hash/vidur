# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
trainer.py

Single-process trainer for AlphaZeroModel using root samples:
- policy target: MCTS visit distribution
- value target: MCTS root value (controller perspective)

Works with mixed-player batches by doing two forward passes (one per player head).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..config import DEFAULT_GAME_V2_CONFIG
from .model_search_nn import (
    HorizonValueNet,
    error_metrics as model_search_error_metrics,
    root_record_to_features,
    records_to_duration_tensor,
    records_to_feature_tensor,
    records_to_target_tensor,
)
from .value_models import AlphaZeroModel


_T = DEFAULT_GAME_V2_CONFIG.trainer

@dataclass
class TrainerConfig:
    lr: float = _T.lr
    weight_decay: float = _T.weight_decay
    policy_weight: float = _T.policy_weight
    value_weight: float = _T.value_weight
    value_only: bool = _T.value_only
    grad_clip_norm: float = _T.grad_clip_norm
    checkpoint_every: int = _T.checkpoint_every
    eval_every: int = _T.eval_every
    invalid_logit: float = _T.invalid_logit


class Trainer:
    def __init__(
        self,
        model: AlphaZeroModel,
        *,
        cfg: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.step: int = 0
        self.best_eval_loss: float = math.inf

    def _policy_loss(
        self,
        policy_logits: torch.Tensor,     # [B, A]
        target_policy: torch.Tensor,     # [B, A]
        action_mask: torch.Tensor,       # [B, A] bool
    ) -> torch.Tensor:
        # mask invalid logits with large negative finite value (avoid -inf * 0 issues)
        logits = policy_logits.masked_fill(~action_mask, self.cfg.invalid_logit)
        log_probs = F.log_softmax(logits, dim=-1)
        # cross-entropy with distribution target
        return -(target_policy * log_probs).sum(dim=-1).mean()

    def _value_loss(
        self,
        value_raw: torch.Tensor,         # [B, 1]
        target_value: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        pred_value = self.model.value_normalized_from_raw(value_raw).view(-1)
        target_norm = self.model.value_target_to_model(target_value).view(-1)
        return F.smooth_l1_loss(pred_value, target_norm)

    def _value_errors(
        self,
        value_raw: torch.Tensor,         # [B, 1]
        target_value: torch.Tensor,      # [B]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_value_real = self.model.value_scalar_from_logits(value_raw).view(-1)
        target_value_real = target_value.view(-1)
        err = pred_value_real - target_value_real
        mse = torch.mean(err * err)
        mae = torch.mean(torch.abs(err))
        return mse, mae


    def train_step(self, batch_by_player: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, float]:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        total_policy = torch.tensor(0.0, device=self.device)
        total_value = torch.tensor(0.0, device=self.device)
        total_value_mse = torch.tensor(0.0, device=self.device)
        total_value_mae = torch.tensor(0.0, device=self.device)
        total_count = 0

        player_stats: Dict[str, Dict[str, float]] = {
            "controller": {
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "value_mse_error": float("nan"),
                "value_mae_error": float("nan"),
                "count": 0.0,
            },
            "adversary": {
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "value_mse_error": float("nan"),
                "value_mae_error": float("nan"),
                "count": 0.0,
            },
        }


        for player in ["controller", "adversary"]:
            batch = batch_by_player.get(player)
            if batch is None:
                continue

            global_features = batch["global_features"]
            prefill_req_features = batch.get("prefill_req_features")
            decode_req_features = batch.get("decode_req_features")
            prefill_req_mask = batch.get("prefill_req_mask")
            decode_req_mask = batch.get("decode_req_mask")

            # keep legacy fallback fields for compatibility
            req_features = batch.get("req_features")
            req_mask = batch.get("req_mask")

            target_value = batch["target_value"]
            action_mask = batch.get("action_mask")
            target_policy = batch.get("target_policy")

            bsz = int(global_features.shape[0])
            total_count += bsz

            policy_logits, value_raw = self.model.forward(
                player=player,
                prefill_req_features=prefill_req_features,
                decode_req_features=decode_req_features,
                global_features=global_features,
                prefill_req_mask=prefill_req_mask,
                decode_req_mask=decode_req_mask,
                req_features=req_features,
                req_mask=req_mask,
                action_mask=None,  # keep masking in loss
            )

            if bool(self.cfg.value_only) or float(self.cfg.policy_weight) <= 0.0:
                p_loss = torch.zeros((), device=self.device)
            else:
                if action_mask is None or target_policy is None:
                    raise RuntimeError("Policy tensors missing for non-value-only training batch")
                p_loss = self._policy_loss(policy_logits, target_policy, action_mask)
            v_loss = self._value_loss(value_raw, target_value)
            v_mse, v_mae = self._value_errors(value_raw, target_value)

            player_stats[player]["policy_loss"] = float(p_loss.detach().cpu())
            player_stats[player]["value_loss"] = float(v_loss.detach().cpu())
            player_stats[player]["value_mse_error"] = float(v_mse.detach().cpu())
            player_stats[player]["value_mae_error"] = float(v_mae.detach().cpu())
            player_stats[player]["count"] = float(bsz)


            total_policy = total_policy + p_loss * bsz
            total_value = total_value + v_loss * bsz
            total_value_mse = total_value_mse + v_mse * bsz
            total_value_mae = total_value_mae + v_mae * bsz

        if total_count == 0:
            raise RuntimeError("Empty mixed batch (no controller or adversary samples)")

        policy_loss = total_policy / total_count
        value_loss = total_value / total_count
        value_mse_error = total_value_mse / total_count
        value_mae_error = total_value_mae / total_count
        if bool(self.cfg.value_only):
            loss = self.cfg.value_weight * value_loss
        else:
            loss = self.cfg.policy_weight * policy_loss + self.cfg.value_weight * value_loss

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(
                "Non-finite training loss before backward: "
                f"loss={float(loss.detach().cpu())}, "
                f"value_loss={float(value_loss.detach().cpu())}, "
                f"policy_loss={float(policy_loss.detach().cpu())}"
            )

        loss.backward()
        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            except TypeError:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.grad_clip_norm,
                )
                if not bool(torch.isfinite(grad_norm).item()):
                    raise RuntimeError(f"Non-finite gradient norm before optimizer step: {grad_norm}")
        self.opt.step()

        self.step += 1
        return {
            "step": float(self.step),
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "value_mse_error": float(value_mse_error.detach().cpu()),
            "value_mae_error": float(value_mae_error.detach().cpu()),
            "controller_policy_loss": player_stats["controller"]["policy_loss"],
            "controller_value_loss": player_stats["controller"]["value_loss"],
            "controller_value_mse_error": player_stats["controller"]["value_mse_error"],
            "controller_value_mae_error": player_stats["controller"]["value_mae_error"],
            "controller_count": player_stats["controller"]["count"],
            "adversary_policy_loss": player_stats["adversary"]["policy_loss"],
            "adversary_value_loss": player_stats["adversary"]["value_loss"],
            "adversary_value_mse_error": player_stats["adversary"]["value_mse_error"],
            "adversary_value_mae_error": player_stats["adversary"]["value_mae_error"],
            "adversary_count": player_stats["adversary"]["count"],
        }


    @torch.no_grad()
    def eval_step(self, batch_by_player: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, float]:
        self.model.eval()

        total_policy = 0.0
        total_value = 0.0
        total_value_mse = 0.0
        total_value_mae = 0.0
        total_count = 0

        player_stats: Dict[str, Dict[str, float]] = {
            "controller": {
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "value_mse_error": float("nan"),
                "value_mae_error": float("nan"),
                "count": 0.0,
            },
            "adversary": {
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "value_mse_error": float("nan"),
                "value_mae_error": float("nan"),
                "count": 0.0,
            },
        }

        for player in ["controller", "adversary"]:
            batch = batch_by_player.get(player)
            if batch is None:
                continue

            global_features = batch["global_features"]
            prefill_req_features = batch.get("prefill_req_features")
            decode_req_features = batch.get("decode_req_features")
            prefill_req_mask = batch.get("prefill_req_mask")
            decode_req_mask = batch.get("decode_req_mask")

            # keep legacy fallback fields for compatibility
            req_features = batch.get("req_features")
            req_mask = batch.get("req_mask")

            target_value = batch["target_value"]
            action_mask = batch.get("action_mask")
            target_policy = batch.get("target_policy")

            bsz = int(global_features.shape[0])
            total_count += bsz

            policy_logits, value_raw = self.model.forward(
                player=player,
                prefill_req_features=prefill_req_features,
                decode_req_features=decode_req_features,
                global_features=global_features,
                prefill_req_mask=prefill_req_mask,
                decode_req_mask=decode_req_mask,
                req_features=req_features,
                req_mask=req_mask,
                action_mask=None,  # keep masking in loss
            )

            if bool(self.cfg.value_only) or float(self.cfg.policy_weight) <= 0.0:
                p_loss = 0.0
            else:
                if action_mask is None or target_policy is None:
                    raise RuntimeError("Policy tensors missing for non-value-only eval batch")
                p_loss = self._policy_loss(policy_logits, target_policy, action_mask).item()
            v_loss = self._value_loss(value_raw, target_value).item()
            v_mse_t, v_mae_t = self._value_errors(value_raw, target_value)
            v_mse = float(v_mse_t.item())
            v_mae = float(v_mae_t.item())

            player_stats[player]["policy_loss"] = float(p_loss)
            player_stats[player]["value_loss"] = float(v_loss)
            player_stats[player]["value_mse_error"] = float(v_mse)
            player_stats[player]["value_mae_error"] = float(v_mae)
            player_stats[player]["count"] = float(bsz)

            total_policy += p_loss * bsz
            total_value += v_loss * bsz
            total_value_mse += v_mse * bsz
            total_value_mae += v_mae * bsz

        if total_count == 0:
            raise RuntimeError("Empty eval batch")

        policy_loss = total_policy / total_count
        value_loss = total_value / total_count
        value_mse_error = total_value_mse / total_count
        value_mae_error = total_value_mae / total_count
        if bool(self.cfg.value_only):
            loss = self.cfg.value_weight * value_loss
        else:
            loss = self.cfg.policy_weight * policy_loss + self.cfg.value_weight * value_loss

        return {
            "step": float(self.step),
            "loss": float(loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "value_mse_error": float(value_mse_error),
            "value_mae_error": float(value_mae_error),

            "controller_policy_loss": player_stats["controller"]["policy_loss"],
            "controller_value_loss": player_stats["controller"]["value_loss"],
            "controller_value_mse_error": player_stats["controller"]["value_mse_error"],
            "controller_value_mae_error": player_stats["controller"]["value_mae_error"],
            "controller_count": player_stats["controller"]["count"],

            "adversary_policy_loss": player_stats["adversary"]["policy_loss"],
            "adversary_value_loss": player_stats["adversary"]["value_loss"],
            "adversary_value_mse_error": player_stats["adversary"]["value_mse_error"],
            "adversary_value_mae_error": player_stats["adversary"]["value_mae_error"],
            "adversary_count": player_stats["adversary"]["count"],
        }


    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.opt.state_dict(),
                "best_eval_loss": self.best_eval_loss,
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def maybe_save_best(self, eval_loss: float, best_path: Path) -> bool:
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = float(eval_loss)
            self.save_checkpoint(best_path)
            return True
        return False


def _model_search_extra(cfg: Any, key: str, default: Any) -> Any:
    extra = getattr(cfg, "extra_config", {}) or {}
    if isinstance(extra, dict):
        return extra.get(key, default)
    return default


def _feature_slice_file_worker(
    records: list[dict[str, Any]],
    start: int,
    end: int,
    shard_path: str,
) -> None:
    """Build one feature shard in a child process and write it to disk."""

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    tensor = torch.tensor(
        [root_record_to_features(records[idx]) for idx in range(int(start), int(end))],
        dtype=torch.float32,
    )
    path = Path(shard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "start": int(start),
            "end": int(end),
            "features": tensor,
        },
        path,
    )


def _records_to_feature_tensor_parallel(
    records: list[dict[str, Any]],
    *,
    num_processes: int,
    chunk_size: int,
    start_method: str,
    shard_dir: Path,
) -> torch.Tensor:
    """Build ModelSearch feature tensors in parallel over record index ranges."""

    record_count = len(records)
    if record_count <= 0:
        return torch.empty((0, 0), dtype=torch.float32)

    worker_count = max(1, min(int(num_processes), int(record_count)))
    if worker_count <= 1:
        return records_to_feature_tensor(records)

    chunk = max(1, int(chunk_size))
    ranges = [(start, min(start + chunk, record_count)) for start in range(0, record_count, chunk)]
    ctx = mp.get_context(str(start_method))
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    for wave_start in range(0, len(ranges), worker_count):
        procs: list[mp.Process] = []
        wave = ranges[wave_start : wave_start + worker_count]
        for offset, (start, end) in enumerate(wave):
            part_id = wave_start + offset
            shard_path = shard_dir / f"features_{int(part_id):05d}.pt"
            proc = ctx.Process(
                target=_feature_slice_file_worker,
                args=(records, int(start), int(end), str(shard_path)),
            )
            proc.start()
            procs.append(proc)
            shard_paths.append(shard_path)
        for proc in procs:
            proc.join()
        failed = [proc for proc in procs if int(proc.exitcode or 0) != 0]
        if failed:
            raise RuntimeError(
                "feature worker failed: "
                + ", ".join(f"pid={proc.pid} exitcode={proc.exitcode}" for proc in failed)
            )

    tensors = []
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu")
        tensors.append(payload["features"].to(dtype=torch.float32))
    if not tensors:
        return torch.empty((0, 0), dtype=torch.float32)
    features = torch.cat(tensors, dim=0)
    shutil.rmtree(shard_dir, ignore_errors=True)
    return features


def _build_model_search_feature_tensor(
    records: list[dict[str, Any]],
    *,
    cfg: Any,
    split_name: str,
) -> torch.Tensor:
    worker_count = int(_model_search_extra(cfg, "model_search_feature_num_processes", 1))
    chunk_size = int(_model_search_extra(cfg, "model_search_feature_chunk_size", 4096))
    start_method = str(_model_search_extra(cfg, "model_search_feature_start_method", "fork"))
    if worker_count <= 1:
        return records_to_feature_tensor(records)

    print(
        "[model_search_nn] "
        f"building {split_name} features with {worker_count} workers "
        f"records={len(records)} chunk_size={chunk_size} start_method={start_method}",
        flush=True,
    )
    return _records_to_feature_tensor_parallel(
        records,
        num_processes=worker_count,
        chunk_size=chunk_size,
        start_method=start_method,
        shard_dir=Path(getattr(cfg, "output_dir", ".")) / "_feature_shards" / str(split_name),
    )


@torch.no_grad()
def _predict_model_search_tensor(
    model: HorizonValueNet,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    preds: list[torch.Tensor] = []
    for start in range(0, int(features.shape[0]), int(batch_size)):
        batch = features[start : start + int(batch_size)].to(device)
        preds.append(model(batch).detach().cpu())
    return torch.cat(preds, dim=0) if preds else torch.empty((0,), dtype=torch.float32)


def train_model_search(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    cfg: Any,
    state_loader: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Train the neural ModelSearchBed controller-value regressor.

    The implementation uses only stored simulator snapshots/stats for features.
    It does not use selected-action fields as model inputs and does not call
    MCTS/search to recompute labels.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / getattr(cfg, "checkpoint_name", "best_model.pt")

    device_name = str(
        _model_search_extra(
            cfg,
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    device = torch.device(device_name)
    batch_size = int(getattr(cfg, "batch_size", 256))
    num_epochs = int(_model_search_extra(cfg, "num_epochs", 50))
    lr = float(_model_search_extra(cfg, "learning_rate", 1.2e-3))
    weight_decay = float(_model_search_extra(cfg, "weight_decay", 1e-4))
    selected_weight = float(_model_search_extra(cfg, "selected_weight", 12.0))
    nonzero_weight = float(_model_search_extra(cfg, "nonzero_weight", 2.5))
    magnitude_weight = float(_model_search_extra(cfg, "magnitude_weight", 2.5))
    zero_weight = float(_model_search_extra(cfg, "zero_weight", 4.0))
    l1_weight = float(_model_search_extra(cfg, "l1_weight", 0.05))
    topk_weight = float(_model_search_extra(cfg, "topk_weight", 0.15))
    residual_l2_weight = float(_model_search_extra(cfg, "residual_l2_weight", 0.002))
    hidden_dim = int(_model_search_extra(cfg, "hidden_dim", 112))
    dropout_p = float(_model_search_extra(cfg, "dropout_p", 0.02))
    residual_bound = float(_model_search_extra(cfg, "residual_bound", 12.0))
    duration_loss_weight = float(_model_search_extra(cfg, "duration_loss_weight", 0.0))

    if num_epochs <= 0:
        raise ValueError("num_epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    del state_loader

    train_features = _build_model_search_feature_tensor(
        train_records,
        cfg=cfg,
        split_name="train",
    )
    train_targets = records_to_target_tensor(train_records)
    train_durations = records_to_duration_tensor(train_records)
    eval_features = _build_model_search_feature_tensor(
        eval_records,
        cfg=cfg,
        split_name="eval",
    )
    eval_targets = records_to_target_tensor(eval_records)

    model = HorizonValueNet(
        input_dim=int(train_features.shape[1]),
        hidden_dim=int(hidden_dim),
        dropout_p=float(dropout_p),
        residual_bound=float(residual_bound),
    )
    model.set_normalizer(train_features)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if int(params) > 150_000:
        raise ValueError(f"ModelSearch NN has {params} trainable params, above 150000")
    model.to(device)

    train_features_device = train_features.to(device)
    train_targets_device = train_targets.to(device)
    train_durations_device = train_durations.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hard_epochs = int(_model_search_extra(cfg, "hard_epochs", 0))
    hard_threshold = float(_model_search_extra(cfg, "hard_threshold", 0.04))
    hard_repeat = int(_model_search_extra(cfg, "hard_repeat", 64))
    hard_loss_multiplier = float(_model_search_extra(cfg, "hard_loss_multiplier", 6.0))
    max_error_selection_weight = float(_model_search_extra(cfg, "max_error_selection_weight", 0.1))
    total_epochs = int(num_epochs) + max(0, int(hard_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(total_epochs)),
        eta_min=float(lr) * 0.08,
    )

    metrics: list[dict[str, Any]] = []
    best_eval_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    train_pred = _predict_model_search_tensor(
        model,
        train_features,
        batch_size=max(int(batch_size), 4096),
        device=device,
    )
    eval_pred = _predict_model_search_tensor(
        model,
        eval_features,
        batch_size=max(int(batch_size), 4096),
        device=device,
    )
    train_m = model_search_error_metrics(train_targets, train_pred)
    eval_m = model_search_error_metrics(eval_targets, eval_pred)
    best_eval_score = float(eval_m["mse"]) + float(max_error_selection_weight) * float(eval_m["max_abs_error"]) ** 2
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(
        {
            "model_state": best_state,
            "trainable_params": int(params),
            "feature_dim": int(train_features.shape[1]),
            "epoch": 0,
            "eval_metrics": eval_m,
            "train_metrics": train_m,
            "eval_selection_score": float(best_eval_score),
        },
        best_path,
    )
    metrics.append(
        {
            "epoch": 0,
            "backend": "horizon_state_mlp",
            "model_name": str(getattr(cfg, "model_name", "nn_agent_candidate")),
            "trainable_params": int(params),
            "train_loss": float(train_m["mse"]),
            "train_mse": float(train_m["mse"]),
            "train_rmse": float(train_m["rmse"]),
            "train_mae": float(train_m["mae"]),
            "train_p95_abs_error": float(train_m["p95_abs_error"]),
            "train_max_abs_error": float(train_m["max_abs_error"]),
            "eval_mse": float(eval_m["mse"]),
            "eval_rmse": float(eval_m["rmse"]),
            "eval_mae": float(eval_m["mae"]),
            "eval_p95_abs_error": float(eval_m["p95_abs_error"]),
            "eval_max_abs_error": float(eval_m["max_abs_error"]),
            "eval_selection_score": float(best_eval_score),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "saved_best": True,
            "hard_stage": False,
            "hard_count": 0,
        }
    )
    print(
        "[model_search_nn] "
        "epoch=000/pretrain "
        f"train_mae={train_m['mae']:.6f} train_p95={train_m['p95_abs_error']:.6f} "
        f"eval_mse={eval_m['mse']:.6f} eval_rmse={eval_m['rmse']:.6f} "
        f"eval_mae={eval_m['mae']:.6f} eval_p95={eval_m['p95_abs_error']:.6f} "
        f"eval_max={eval_m['max_abs_error']:.6f} best=True",
        flush=True,
    )

    for epoch in range(1, int(total_epochs) + 1):
        model.train()
        hard_stage = bool(epoch > int(num_epochs))
        hard_count = 0
        if hard_stage:
            train_pred_for_hard = _predict_model_search_tensor(
                model,
                train_features,
                batch_size=max(int(batch_size), 4096),
                device=device,
            )
            hard_mask = (train_pred_for_hard - train_targets).abs() > float(hard_threshold)
            hard_idx_cpu = torch.nonzero(hard_mask, as_tuple=False).view(-1)
            hard_count = int(hard_idx_cpu.numel())
            if hard_count > 0:
                repeated = hard_idx_cpu.repeat_interleave(max(1, int(hard_repeat))).to(device)
                random_count = min(int(train_features_device.shape[0]), max(int(batch_size), int(repeated.numel())))
                random_idx = torch.randperm(int(train_features_device.shape[0]), device=device)[:random_count]
                order = torch.cat([repeated, random_idx], dim=0)
                order = order[torch.randperm(int(order.numel()), device=device)]
            else:
                order = torch.randperm(int(train_features_device.shape[0]), device=device)
        else:
            order = torch.randperm(int(train_features_device.shape[0]), device=device)
        loss_sum = 0.0
        sample_count = 0

        for start in range(0, int(order.numel()), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch_features = train_features_device.index_select(0, idx)
            target = train_targets_device.index_select(0, idx)
            duration_target = train_durations_device.index_select(0, idx)
            prediction, residual, prior_value, duration_pred = model.forward_parts(batch_features)
            abs_target = target.abs()
            nonzero_mask = abs_target > 1e-6
            weights = (
                1.0
                + float(zero_weight) * (~nonzero_mask).to(dtype=target.dtype)
                + float(selected_weight) * (abs_target >= 1.0).to(dtype=target.dtype)
                + float(nonzero_weight) * (abs_target > 0.02).to(dtype=target.dtype)
                + float(magnitude_weight) * abs_target.clamp_max(8.0)
            )
            if hard_stage:
                weights = weights * float(hard_loss_multiplier)
            err = prediction - target
            loss = (weights * err.pow(2)).mean()
            if float(l1_weight) > 0.0:
                loss = loss + float(l1_weight) * (weights * err.abs()).mean()
            if float(topk_weight) > 0.0:
                k = max(1, int(math.ceil(float(err.numel()) * 0.10)))
                tail = torch.topk(err.abs(), k=k, largest=True).values
                loss = loss + float(topk_weight) * tail.pow(2).mean()
            if float(residual_l2_weight) > 0.0:
                loss = loss + float(residual_l2_weight) * residual.pow(2).mean()
            if float(duration_loss_weight) > 0.0:
                duration_err = (duration_pred - duration_target) / 0.05
                loss = loss + float(duration_loss_weight) * duration_err.pow(2).mean()
            loss = loss + 1e-7 * prior_value.pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            count = int(idx.numel())
            loss_sum += float(loss.detach().cpu()) * float(count)
            sample_count += count

        scheduler.step()

        train_pred = _predict_model_search_tensor(
            model,
            train_features,
            batch_size=max(int(batch_size), 4096),
            device=device,
        )
        eval_pred = _predict_model_search_tensor(
            model,
            eval_features,
            batch_size=max(int(batch_size), 4096),
            device=device,
        )
        train_m = model_search_error_metrics(train_targets, train_pred)
        eval_m = model_search_error_metrics(eval_targets, eval_pred)
        eval_mse = float(eval_m["mse"])
        eval_score = float(eval_mse) + float(max_error_selection_weight) * float(eval_m["max_abs_error"]) ** 2
        saved_best = bool(eval_score < best_eval_score)
        if saved_best:
            best_eval_score = eval_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "trainable_params": int(params),
                    "feature_dim": int(train_features.shape[1]),
                    "epoch": int(epoch),
                    "eval_metrics": eval_m,
                    "train_metrics": train_m,
                    "eval_selection_score": float(eval_score),
                },
                best_path,
            )

        row = {
            "epoch": int(epoch),
            "backend": "horizon_state_mlp",
            "model_name": str(getattr(cfg, "model_name", "nn_agent_candidate")),
            "trainable_params": int(params),
            "train_loss": float(loss_sum / max(1, sample_count)),
            "train_mse": float(train_m["mse"]),
            "train_rmse": float(train_m["rmse"]),
            "train_mae": float(train_m["mae"]),
            "train_p95_abs_error": float(train_m["p95_abs_error"]),
            "train_max_abs_error": float(train_m["max_abs_error"]),
            "eval_mse": float(eval_m["mse"]),
            "eval_rmse": float(eval_m["rmse"]),
            "eval_mae": float(eval_m["mae"]),
            "eval_p95_abs_error": float(eval_m["p95_abs_error"]),
            "eval_max_abs_error": float(eval_m["max_abs_error"]),
            "eval_selection_score": float(eval_score),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "saved_best": bool(saved_best),
            "hard_stage": bool(hard_stage),
            "hard_count": int(hard_count),
        }
        metrics.append(row)
        print(
            "[model_search_nn] "
            f"epoch={epoch:03d}/{int(total_epochs):03d} "
            f"train_mae={train_m['mae']:.6f} train_p95={train_m['p95_abs_error']:.6f} "
            f"eval_mse={eval_m['mse']:.6f} eval_rmse={eval_m['rmse']:.6f} "
            f"eval_mae={eval_m['mae']:.6f} eval_p95={eval_m['p95_abs_error']:.6f} "
            f"eval_max={eval_m['max_abs_error']:.6f} "
            f"hard={hard_count if hard_stage else 0} best={saved_best}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    model.to("cpu")
    model.eval()
    # Final Bellman CSV evaluation uses the same records immediately after
    # training. Keep CPU feature tensors on the model so inference does not
    # rebuild Python features for train/eval splits.
    model._model_search_cached_features = {
        "train": train_features.detach().cpu(),
        "eval": eval_features.detach().cpu(),
    }
    return {
        "model": model,
        "best_checkpoint_path": best_path,
        "train_metrics": metrics,
    }
