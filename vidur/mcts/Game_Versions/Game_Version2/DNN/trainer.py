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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..config import DEFAULT_GAME_V2_CONFIG
from .models import AlphaZeroModel


_T = DEFAULT_GAME_V2_CONFIG.trainer

@dataclass
class TrainerConfig:
    lr: float = _T.lr
    weight_decay: float = _T.weight_decay
    policy_weight: float = _T.policy_weight
    value_weight: float = _T.value_weight
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


    def train_step(self, batch_by_player: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, float]:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        total_policy = torch.tensor(0.0, device=self.device)
        total_value = torch.tensor(0.0, device=self.device)
        total_count = 0

        player_stats: Dict[str, Dict[str, float]] = {
            "controller": {"policy_loss": float("nan"), "value_loss": float("nan"), "count": 0.0},
            "adversary": {"policy_loss": float("nan"), "value_loss": float("nan"), "count": 0.0},
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

            action_mask = batch["action_mask"]
            target_policy = batch["target_policy"]
            target_value = batch["target_value"]

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

            p_loss = self._policy_loss(policy_logits, target_policy, action_mask)
            v_loss = self._value_loss(value_raw, target_value)

            player_stats[player]["policy_loss"] = float(p_loss.detach().cpu())
            player_stats[player]["value_loss"] = float(v_loss.detach().cpu())
            player_stats[player]["count"] = float(bsz)


            total_policy = total_policy + p_loss * bsz
            total_value = total_value + v_loss * bsz

        if total_count == 0:
            raise RuntimeError("Empty mixed batch (no controller or adversary samples)")

        policy_loss = total_policy / total_count
        value_loss = total_value / total_count
        loss = self.cfg.policy_weight * policy_loss + self.cfg.value_weight * value_loss

        loss.backward()
        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()

        self.step += 1
        return {
            "step": float(self.step),
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "controller_policy_loss": player_stats["controller"]["policy_loss"],
            "controller_value_loss": player_stats["controller"]["value_loss"],
            "controller_count": player_stats["controller"]["count"],
            "adversary_policy_loss": player_stats["adversary"]["policy_loss"],
            "adversary_value_loss": player_stats["adversary"]["value_loss"],
            "adversary_count": player_stats["adversary"]["count"],
        }


    @torch.no_grad()
    def eval_step(self, batch_by_player: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, float]:
        self.model.eval()

        total_policy = 0.0
        total_value = 0.0
        total_count = 0

        player_stats: Dict[str, Dict[str, float]] = {
            "controller": {"policy_loss": float("nan"), "value_loss": float("nan"), "count": 0.0},
            "adversary": {"policy_loss": float("nan"), "value_loss": float("nan"), "count": 0.0},
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

            action_mask = batch["action_mask"]
            target_policy = batch["target_policy"]
            target_value = batch["target_value"]

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

            p_loss = self._policy_loss(policy_logits, target_policy, action_mask).item()
            v_loss = self._value_loss(value_raw, target_value).item()

            player_stats[player]["policy_loss"] = float(p_loss)
            player_stats[player]["value_loss"] = float(v_loss)
            player_stats[player]["count"] = float(bsz)

            total_policy += p_loss * bsz
            total_value += v_loss * bsz

        if total_count == 0:
            raise RuntimeError("Empty eval batch")

        policy_loss = total_policy / total_count
        value_loss = total_value / total_count
        loss = self.cfg.policy_weight * policy_loss + self.cfg.value_weight * value_loss

        return {
            "step": float(self.step),
            "loss": float(loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),

            "controller_policy_loss": player_stats["controller"]["policy_loss"],
            "controller_value_loss": player_stats["controller"]["value_loss"],
            "controller_count": player_stats["controller"]["count"],

            "adversary_policy_loss": player_stats["adversary"]["policy_loss"],
            "adversary_value_loss": player_stats["adversary"]["value_loss"],
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
