from __future__ import annotations

import torch
from torch import nn


class LinearValueModel(nn.Module):
    def __init__(self, num_features: int = 30, zero_init: bool = True) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.value = nn.Linear(self.num_features, 1, bias=True)
        self.register_buffer("feat_mean", torch.zeros(self.num_features, dtype=torch.float32))
        self.register_buffer("feat_std", torch.ones(self.num_features, dtype=torch.float32))

        if zero_init:
            nn.init.zeros_(self.value.weight)
            nn.init.zeros_(self.value.bias)

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.detach().to(dtype=torch.float32, device=self.feat_mean.device)
        std = std.detach().to(dtype=torch.float32, device=self.feat_std.device)
        if mean.numel() != self.num_features or std.numel() != self.num_features:
            raise ValueError(
                f"feature stats size mismatch: got mean={mean.numel()}, std={std.numel()}, expected={self.num_features}"
            )
        self.feat_mean.copy_(mean.view(-1))
        self.feat_std.copy_(std.view(-1).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.size(-1) != self.num_features:
            raise ValueError(
                f"expected x shape [B,{self.num_features}], got {tuple(x.shape)}"
            )
        z = (x - self.feat_mean) / self.feat_std.clamp_min(1e-6)
        out = self.value(z).squeeze(-1)
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        y = self.forward(x)
        if was_training:
            self.train()
        return y
