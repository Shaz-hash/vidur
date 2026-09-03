from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..environment import VidurMCTSState
from .dnn_spec import DEFAULT_DNN_SPEC, DNNGameSpec
from .types import ModelInputs, Player

DEFAULT_SPEC = DEFAULT_DNN_SPEC

N_PREFILL_REQ: int = int(DEFAULT_SPEC.n_prefill_req)
D_PREFILL_REQ: int = int(DEFAULT_SPEC.d_prefill_req)
N_DECODE_REQ: int = int(DEFAULT_SPEC.n_decode_req)
D_DECODE_REQ: int = int(DEFAULT_SPEC.d_decode_req)
D_GLOBAL: int = int(DEFAULT_SPEC.d_global)

N_REQ: int = int(DEFAULT_SPEC.n_req_total)
D_REQ: int = int(DEFAULT_SPEC.d_prefill_req)

NUM_ACTIONS_CONTROLLER: int = int(DEFAULT_SPEC.num_actions_controller)
NUM_ACTIONS_ADVERSARY: int = int(DEFAULT_SPEC.num_actions_adversary)

V_MIN: float = DEFAULT_SPEC.v_min
V_MAX: float = DEFAULT_SPEC.v_max
V_LINEAR_MIN: float = DEFAULT_SPEC.v_linear_min
V_NORM_MIN: float = DEFAULT_SPEC.v_norm_min
V_NORM_MAX: float = DEFAULT_SPEC.v_norm_max
V_LINEAR_NORM_MIN: float = DEFAULT_SPEC.v_linear_norm_min
V_TAIL_COMPRESS_POWER: float = DEFAULT_SPEC.v_tail_compress_power


def normalize_value_real(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, V_MIN, V_MAX)
    linear_scale = abs(V_LINEAR_NORM_MIN) / abs(V_LINEAR_MIN)

    y_linear = x * linear_scale

    tail_real_span = V_LINEAR_MIN - V_MIN
    tail_norm_span = V_NORM_MIN - V_LINEAR_NORM_MIN
    t = ((V_LINEAR_MIN - x) / tail_real_span).clamp(0.0, 1.0)
    y_tail = V_LINEAR_NORM_MIN + tail_norm_span * torch.pow(t, V_TAIL_COMPRESS_POWER)

    return torch.where(x >= V_LINEAR_MIN, y_linear, y_tail).clamp(V_NORM_MIN, V_NORM_MAX)


def denormalize_value_model(y: torch.Tensor) -> torch.Tensor:
    y = torch.clamp(y, V_NORM_MIN, V_NORM_MAX)
    linear_scale = abs(V_LINEAR_NORM_MIN) / abs(V_LINEAR_MIN)

    x_linear = y / linear_scale

    tail_real_span = V_LINEAR_MIN - V_MIN
    tail_norm_span = V_LINEAR_NORM_MIN - V_NORM_MIN
    t = ((V_LINEAR_NORM_MIN - y) / tail_norm_span).clamp(0.0, 1.0)
    x_tail = V_LINEAR_MIN - tail_real_span * torch.pow(t, 1.0 / V_TAIL_COMPRESS_POWER)

    return torch.where(y >= V_LINEAR_NORM_MIN, x_linear, x_tail).clamp(V_MIN, V_MAX)


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def _sanitize_model_features(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # All GV3 model features are normalized into [0, 1]. Native/padded masked
    # rows can contain stale values; zero them before any encoder sees them.
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if mask is not None:
        x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
    return x


class _FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlphaZeroModel(nn.Module):
    """
    GV3 value-focused model.

    The public API intentionally matches the legacy AlphaZero-style model so the
    existing search code can keep calling it without core MCTS changes.
    """

    def __init__(
        self,
        *,
        spec: Optional[DNNGameSpec] = None,
        num_actions_controller: Optional[int] = None,
        num_actions_adversary: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.spec = spec or DEFAULT_SPEC
        self.spec.validate()

        self.n_prefill_req = int(self.spec.n_prefill_req)
        self.d_prefill_req = int(self.spec.d_prefill_req)
        self.n_decode_req = int(self.spec.n_decode_req)
        self.d_decode_req = int(self.spec.d_decode_req)
        self.d_global = int(self.spec.d_global)

        self.num_actions_controller = int(
            num_actions_controller if num_actions_controller is not None else self.spec.num_actions_controller
        )
        self.num_actions_adversary = int(
            num_actions_adversary if num_actions_adversary is not None else self.spec.num_actions_adversary
        )

        d_model = int(self.spec.d_req_emb)
        d_hidden = int(self.spec.d_trunk)
        n_heads = int(self.spec.num_attention_heads)
        n_layers = int(self.spec.num_attention_layers)
        dropout_p = float(self.spec.dropout_p)

        self.prefill_encoder = _FeatureMLP(self.d_prefill_req, d_model)
        self.decode_encoder = _FeatureMLP(self.d_decode_req, d_model)
        self.global_encoder = _FeatureMLP(self.d_global, d_model)

        self.player_embedding = nn.Embedding(2, d_model)
        self.type_embedding = nn.Embedding(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=max(d_hidden, d_model * 2),
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
            )

        self.summary_head = nn.Sequential(
            nn.Linear(d_model * 4, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.controller_value_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )
        self.adversary_value_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

        self._infer_perf_enabled = True
        self._infer_perf_every = 10000
        self.reset_infer_perf()

    def _player_index(self, player: Player) -> int:
        if player == "controller":
            return 0
        if player == "adversary":
            return 1
        raise ValueError(f"Unknown player={player!r}")

    def _split_fallback_inputs(
        self,
        *,
        req_features: Optional[torch.Tensor],
        req_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if req_features is None:
            raise ValueError("Provide either split req tensors or req_features fallback")
        if req_features.dim() != 3:
            raise ValueError(f"req_features must be [B,N,D], got {tuple(req_features.shape)}")

        bsz, n_all, d_all = req_features.shape
        if d_all < max(self.d_prefill_req, self.d_decode_req):
            raise ValueError(
                f"req_features last dim {d_all} smaller than required "
                f"{max(self.d_prefill_req, self.d_decode_req)}"
            )

        p_take = min(self.n_prefill_req, n_all)
        prefill_req_features = req_features[:, :p_take, : self.d_prefill_req]
        if p_take < self.n_prefill_req:
            pad = req_features.new_zeros((bsz, self.n_prefill_req - p_take, self.d_prefill_req))
            prefill_req_features = torch.cat([prefill_req_features, pad], dim=1)

        d_start = self.n_prefill_req
        d_end = min(d_start + self.n_decode_req, n_all)
        decode_req_features = req_features[:, d_start:d_end, : self.d_decode_req]
        d_take = decode_req_features.size(1)
        if d_take < self.n_decode_req:
            pad = req_features.new_zeros((bsz, self.n_decode_req - d_take, self.d_decode_req))
            decode_req_features = torch.cat([decode_req_features, pad], dim=1)

        prefill_req_mask = None
        decode_req_mask = None
        if req_mask is not None:
            pm = req_mask[:, :p_take]
            if p_take < self.n_prefill_req:
                pm_pad = torch.zeros((bsz, self.n_prefill_req - p_take), dtype=torch.bool, device=req_mask.device)
                pm = torch.cat([pm, pm_pad], dim=1)
            prefill_req_mask = pm

            dm = req_mask[:, d_start:d_end]
            if d_take < self.n_decode_req:
                dm_pad = torch.zeros((bsz, self.n_decode_req - d_take), dtype=torch.bool, device=req_mask.device)
                dm = torch.cat([dm, dm_pad], dim=1)
            decode_req_mask = dm

        return prefill_req_features, decode_req_features, prefill_req_mask, decode_req_mask

    def _encode_state(
        self,
        *,
        player: Player,
        prefill_req_features: torch.Tensor,
        decode_req_features: torch.Tensor,
        global_features: torch.Tensor,
        prefill_req_mask: Optional[torch.Tensor],
        decode_req_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz = global_features.size(0)
        device = global_features.device

        if prefill_req_mask is None:
            prefill_req_mask = torch.ones((bsz, self.n_prefill_req), dtype=torch.bool, device=device)
        if decode_req_mask is None:
            decode_req_mask = torch.ones((bsz, self.n_decode_req), dtype=torch.bool, device=device)

        prefill_req_features = _sanitize_model_features(prefill_req_features, prefill_req_mask)
        decode_req_features = _sanitize_model_features(decode_req_features, decode_req_mask)
        global_features = torch.nan_to_num(
            global_features,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        p_tok = self.prefill_encoder(prefill_req_features)
        d_tok = self.decode_encoder(decode_req_features)
        g_tok = self.global_encoder(global_features)

        p_type = self.type_embedding(torch.full((self.n_prefill_req,), 1, dtype=torch.long, device=device))
        d_type = self.type_embedding(torch.full((self.n_decode_req,), 2, dtype=torch.long, device=device))
        cls_type = self.type_embedding(torch.zeros((1,), dtype=torch.long, device=device))

        p_tok = p_tok + p_type.unsqueeze(0)
        d_tok = d_tok + d_type.unsqueeze(0)

        player_idx = torch.full((bsz,), self._player_index(player), dtype=torch.long, device=device)
        cls_tok = g_tok + self.player_embedding(player_idx) + cls_type.expand(bsz, -1)
        cls_tok = cls_tok.unsqueeze(1)

        tokens = torch.cat([cls_tok, p_tok, d_tok], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((bsz, 1), dtype=torch.bool, device=device),
                ~prefill_req_mask,
                ~decode_req_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)

        cls_out = encoded[:, 0, :]
        p_out = encoded[:, 1 : 1 + self.n_prefill_req, :]
        d_out = encoded[:, 1 + self.n_prefill_req :, :]

        p_pool = _masked_mean(p_out, prefill_req_mask)
        d_pool = _masked_mean(d_out, decode_req_mask)

        summary = torch.cat([cls_out, p_pool, d_pool, g_tok], dim=-1)
        return self.summary_head(summary)

    def _dummy_policy_logits(
        self,
        *,
        player: Player,
        batch_size: int,
        device: torch.device,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        width = self.num_actions_controller if player == "controller" else self.num_actions_adversary
        logits = torch.zeros((batch_size, width), dtype=torch.float32, device=device)
        if action_mask is not None:
            if action_mask.shape != logits.shape:
                raise ValueError(
                    f"action_mask shape {tuple(action_mask.shape)} != logits shape {tuple(logits.shape)}"
                )
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits

    def forward(
        self,
        *,
        player: Player,
        global_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        prefill_req_features: Optional[torch.Tensor] = None,
        decode_req_features: Optional[torch.Tensor] = None,
        prefill_req_mask: Optional[torch.Tensor] = None,
        decode_req_mask: Optional[torch.Tensor] = None,
        req_features: Optional[torch.Tensor] = None,
        req_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prefill_req_features is None or decode_req_features is None:
            prefill_req_features, decode_req_features, prefill_req_mask, decode_req_mask = self._split_fallback_inputs(
                req_features=req_features,
                req_mask=req_mask,
            )

        if global_features.dim() != 2 or global_features.size(1) != self.d_global:
            raise ValueError(
                f"global_features must be [B,{self.d_global}], got {tuple(global_features.shape)}"
            )
        if prefill_req_features.dim() != 3 or prefill_req_features.size(1) != self.n_prefill_req or prefill_req_features.size(2) != self.d_prefill_req:
            raise ValueError(
                f"prefill_req_features must be [B,{self.n_prefill_req},{self.d_prefill_req}], got {tuple(prefill_req_features.shape)}"
            )
        if decode_req_features.dim() != 3 or decode_req_features.size(1) != self.n_decode_req or decode_req_features.size(2) != self.d_decode_req:
            raise ValueError(
                f"decode_req_features must be [B,{self.n_decode_req},{self.d_decode_req}], got {tuple(decode_req_features.shape)}"
            )

        h = self._encode_state(
            player=player,
            prefill_req_features=prefill_req_features,
            decode_req_features=decode_req_features,
            global_features=global_features,
            prefill_req_mask=prefill_req_mask,
            decode_req_mask=decode_req_mask,
        )

        if player == "controller":
            value_raw = self.controller_value_head(h)
        elif player == "adversary":
            value_raw = self.adversary_value_head(h)
        else:
            raise ValueError(f"Unknown player={player!r}; expected 'controller' or 'adversary'")

        policy_logits = self._dummy_policy_logits(
            player=player,
            batch_size=global_features.size(0),
            device=global_features.device,
            action_mask=action_mask,
        )
        return policy_logits, value_raw

    @torch.inference_mode()
    def infer_from_inputs(
        self,
        inputs: ModelInputs,
        player: Player,
        *,
        device: Optional[torch.device] = None,
    ) -> Tuple[float, list[float]]:
        dev = device or next(self.parameters()).device
        t0 = time.perf_counter()

        prefill_req_features = inputs.prefill_req_features.to(dev)
        decode_req_features = inputs.decode_req_features.to(dev)
        global_features = inputs.global_features.to(dev)
        prefill_req_mask = inputs.prefill_req_mask.to(dev) if inputs.prefill_req_mask is not None else None
        decode_req_mask = inputs.decode_req_mask.to(dev) if inputs.decode_req_mask is not None else None
        action_mask = inputs.action_mask.to(dev) if inputs.action_mask is not None else None
        req_features = inputs.req_features.to(dev) if inputs.req_features is not None else None
        req_mask = inputs.req_mask.to(dev) if inputs.req_mask is not None else None

        policy_logits, value_raw = self.forward(
            player=player,
            prefill_req_features=prefill_req_features,
            decode_req_features=decode_req_features,
            global_features=global_features,
            prefill_req_mask=prefill_req_mask,
            decode_req_mask=decode_req_mask,
            action_mask=action_mask,
            req_features=req_features,
            req_mask=req_mask,
        )

        value = self.value_scalar_from_logits(value_raw).squeeze(0).item()
        priors_t = torch.nan_to_num(F.softmax(policy_logits, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        priors = priors_t.squeeze(0).to("cpu").tolist()

        self._infer_perf["forward"] += max(0.0, time.perf_counter() - t0)
        self._infer_perf["calls"] += 1
        return float(value), list(priors)

    @torch.inference_mode()
    def infer_from_state(
        self,
        state: VidurMCTSState,
        player: Player,
        *,
        build_inputs: Callable[[VidurMCTSState, Player, torch.device], ModelInputs],
        device: Optional[torch.device] = None,
    ) -> Tuple[float, list[float]]:
        dev = device or next(self.parameters()).device
        inputs = build_inputs(state, player, dev)
        return self.infer_from_inputs(inputs, player, device=dev)

    def scale_value(self, value_real: torch.Tensor) -> torch.Tensor:
        return normalize_value_real(value_real)

    def unscale_value(self, value_scaled: torch.Tensor) -> torch.Tensor:
        return denormalize_value_model(value_scaled)

    def value_target_to_model(self, value_real: torch.Tensor) -> torch.Tensor:
        return normalize_value_real(value_real)

    def value_target_to_support(self, value_real: torch.Tensor) -> torch.Tensor:
        return self.value_target_to_model(value_real)

    def value_normalized_from_raw(self, value_raw: torch.Tensor) -> torch.Tensor:
        return -torch.sigmoid(value_raw).squeeze(-1)

    def value_scalar_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        value_norm = self.value_normalized_from_raw(value_logits)
        return denormalize_value_model(value_norm)

    def reset_infer_perf(self) -> None:
        self._infer_perf = {
            "forward": 0.0,
            "calls": 0,
        }

    def _print_infer_perf(self, prefix: str = "[INFER_PERF]") -> None:
        p = self._infer_perf
        calls = max(1, int(p.get("calls", 0)))
        avg_ms = 1000.0 * float(p.get("forward", 0.0)) / float(calls)
        print(f"{prefix} calls={calls} total={float(p.get('forward', 0.0)):.3f}s avg={avg_ms:.3f}ms")
