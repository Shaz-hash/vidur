

## (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)


"""
AlphaZero-style policy/value network for Vidur MCTS.

Design goals (for correct bridging with MCTS):
- Node = simulator state + player-to-act (in MCTS code).
- Edge = action; each edge stores a *prior* from the NN policy π(a|s).
- This file only defines the NN and inference helpers.
- Feature extraction from VidurMCTSState is intentionally NOT implemented here yet.

Important: NN value is from the *controller perspective*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# NOTE: models.py is inside vidur/vidur/mcts/DNN/, so environment is one level up.
from ..environment import VidurMCTSState
from .dnn_spec import DEFAULT_DNN_SPEC, DNNGameSpec
from .types import ModelInputs, Player


# =============================================================================
# Configuration constants (can later be moved into a config object) from dnn_spec.py for easy access by model and training code.
# =============================================================================

DEFAULT_SPEC = DEFAULT_DNN_SPEC

N_PREFILL_REQ: int = int(DEFAULT_SPEC.n_prefill_req)
D_PREFILL_REQ: int = int(DEFAULT_SPEC.d_prefill_req)
N_DECODE_REQ: int = int(DEFAULT_SPEC.n_decode_req)
D_DECODE_REQ: int = int(DEFAULT_SPEC.d_decode_req)
D_GLOBAL: int = int(DEFAULT_SPEC.d_global)

# Backward-compat aliases (for old imports)
N_REQ: int = int(DEFAULT_SPEC.n_req_total)
D_REQ: int = int(DEFAULT_SPEC.d_prefill_req)

NUM_ACTIONS_CONTROLLER: int = int(DEFAULT_SPEC.num_actions_controller)
NUM_ACTIONS_ADVERSARY: int = int(DEFAULT_SPEC.num_actions_adversary)

D_REQ_EMB: int = int(DEFAULT_SPEC.d_req_emb)
D_GLOBAL_EMB: int = int(DEFAULT_SPEC.d_global_emb)
D_TRUNK: int = int(DEFAULT_SPEC.d_trunk)
D_COND_EMB: int = int(DEFAULT_SPEC.d_cond_emb)


# Value range (real units, controller perspective)
V_MIN: float = DEFAULT_SPEC.v_min
V_MAX: float = DEFAULT_SPEC.v_max

# Scalar-head normalization:
#   [-48, 0]   -> [-0.98, 0.0] linearly
#   [-50, -48] -> [-1.0, -0.98] with compressed quadratic tail
V_LINEAR_MIN: float = DEFAULT_SPEC.v_linear_min
V_NORM_MIN: float = DEFAULT_SPEC.v_norm_min
V_NORM_MAX: float = DEFAULT_SPEC.v_norm_max
V_LINEAR_NORM_MIN: float = DEFAULT_SPEC.v_linear_norm_min
V_TAIL_COMPRESS_POWER: float = DEFAULT_SPEC.v_tail_compress_power

# MuZero-style value support (optional, but you already started it) * Note : Penalty and Max SLO cost in real units in seconds
# HARD_MISS_PENALTY: float = 5
# MAX_SLO_COST: float = 10
# GAMMA: float = 0.98
# VALUE_SCALE: float = 0.5  # scale between real and scaled units

# # SUPPORT_SIZE: int = math.ceil(((MAX_SLO_COST + HARD_MISS_PENALTY) / (1.0 - GAMMA)) / VALUE_SCALE)
# SUPPORT_SIZE: int = math.ceil(((1) / (1.0 - GAMMA)) / VALUE_SCALE)



##------------------------------
# HELPER FUNCTIONS
##------------------------------

def mlp(in_dim: int, hidden: list[int], out_dim: int, act: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """Creates a multi-layer perceptron (MLP) with the specified architecture.

    Args:
        in_dim (int): Input dimension.
        hidden (list[int]): List of hidden layer sizes.
        out_dim (int): Output dimension.
        act (nn.Module): Activation function to use between layers.

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    layers : list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def masked_mean (x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Computes the mean of a tensor along a specified dimension, optionally masking elements.
    x: [B, N, D]
    mask: [B, N] (boolean) True for valid elements, False for masked elements

    Args:
        x (torch.Tensor): Input tensor.
        mask (Optional[torch.Tensor]): Optional boolean mask to apply.
        dim (int): Dimension along which to compute the mean.

    Returns:
        torch.Tensor: The computed mean tensor.
    """
    if mask is None:
        return x.mean(dim=dim)
    m = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B, N, 1]
    denom = m.sum(dim=dim, keepdim=False).clamp(min=1.0)  # [B, 1] or [B]
    return (x * m).sum(dim=dim) / denom.unsqueeze(-1) if denom.dim() == 1 else (x * m).sum(dim=dim) / denom



def masked_max (x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Computes the maximum of a tensor along a specified dimension, optionally masking elements.
    x: [B, N, D]
    safe masked max. Invalid (masked) positions get -inf; if all invalid then output becomes 0

    Args:
        x (torch.Tensor): Input tensor.
        mask (Optional[torch.Tensor]): 
        dim (int): Dimension along which to compute the maximum.

    Returns:
        torch.Tensor: The computed maximum tensor.
    """
    if mask is None:
        return x.max(dim=dim).values
    x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    out = x_masked.max(dim=dim).values
    return torch.nan_to_num(out, neginf=0.0, posinf=0.0)


def masked_min(x, mask, dim):
    "Similar to masked_max but for min"
    if mask is None:
        return x.min(dim=dim).values
    # mask: True=valid
    x2 = x.masked_fill(~mask.unsqueeze(-1), float("inf"))
    out = x2.min(dim=dim).values
    out = out.masked_fill(torch.isinf(out), 0.0)
    return out




#------------------------------
# Value Normalization Helpers for AlphaZeroModel
#------------------------------

def normalize_value_real(x: torch.Tensor) -> torch.Tensor:
    """
    Maps real value in [V_MIN, V_MAX] into model space [-1, 0].

    For controller values:
      - [-48, 0] is nearly all of the dynamic range and stays linear
      - [-50, -48] is compressed into a small tail near -1
    """
    x = torch.clamp(x, V_MIN, V_MAX)
    linear_scale = abs(V_LINEAR_NORM_MIN) / abs(V_LINEAR_MIN)

    y_linear = x * linear_scale

    tail_real_span = V_LINEAR_MIN - V_MIN  # 2.0
    tail_norm_span = V_NORM_MIN - V_LINEAR_NORM_MIN  # -0.02
    t = ((V_LINEAR_MIN - x) / tail_real_span).clamp(0.0, 1.0)
    y_tail = V_LINEAR_NORM_MIN + tail_norm_span * torch.pow(t, V_TAIL_COMPRESS_POWER)

    return torch.where(x >= V_LINEAR_MIN, y_linear, y_tail).clamp(V_NORM_MIN, V_NORM_MAX)


def denormalize_value_model(y: torch.Tensor) -> torch.Tensor:
    """
    Inverse of normalize_value_real().
    """
    y = torch.clamp(y, V_NORM_MIN, V_NORM_MAX)
    linear_scale = abs(V_LINEAR_NORM_MIN) / abs(V_LINEAR_MIN)

    x_linear = y / linear_scale

    tail_real_span = V_LINEAR_MIN - V_MIN  # 2.0
    tail_norm_span = V_LINEAR_NORM_MIN - V_NORM_MIN  # 0.02
    t = ((V_LINEAR_NORM_MIN - y) / tail_norm_span).clamp(0.0, 1.0)
    x_tail = V_LINEAR_MIN - tail_real_span * torch.pow(t, 1.0 / V_TAIL_COMPRESS_POWER)

    return torch.where(y >= V_LINEAR_NORM_MIN, x_linear, x_tail).clamp(V_MIN, V_MAX)





class AlphaZeroModel(nn.Module):

    """
    Policy/value network with:
      - shared trunk
      - separate policy heads per player
      - single scalar value head in normalized space

    Forward returns:
      policy_logits: [B, num_actions(player)]
      value_raw:     [B, 1] raw scalar head output (convert via value_scalar_from_logits)
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

        # Shapes
        self.n_prefill_req = int(self.spec.n_prefill_req)
        self.d_prefill_req = int(self.spec.d_prefill_req)
        self.n_decode_req = int(self.spec.n_decode_req)
        self.d_decode_req = int(self.spec.d_decode_req)
        self.d_global = int(self.spec.d_global)

        # Flat output dims
        self.num_actions_controller = int(
            num_actions_controller if num_actions_controller is not None else self.spec.num_actions_controller
        )
        self.num_actions_adversary = int(
            num_actions_adversary if num_actions_adversary is not None else self.spec.num_actions_adversary
        )

        # Encoders (separate DeepSets)
        self.norm_prefill = nn.Identity()
        self.norm_decode = nn.Identity()
        self.norm_global = nn.Identity()

        self.prefill_encoder = nn.Linear(self.d_prefill_req, D_REQ_EMB)
        self.decode_encoder = nn.Linear(self.d_decode_req, D_REQ_EMB)
        self.global_encoder = nn.Linear(self.d_global, D_GLOBAL_EMB)

        # pooled(prefill)=3*emb, pooled(decode)=3*emb, global=emb
        trunk_in = D_REQ_EMB * 6 + D_GLOBAL_EMB
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, D_TRUNK),
            nn.ReLU(),
        )

        # -------- Adversary factor heads --------
        self.adv_launch_head = nn.Linear(D_TRUNK, int(self.spec.adv_launch_size))
        self.adv_launch_emb = nn.Embedding(int(self.spec.adv_launch_size), D_COND_EMB)
        self.adv_template_head = nn.Sequential(
            nn.Linear(D_TRUNK + D_COND_EMB, D_TRUNK),
            nn.ReLU(),
            nn.Linear(D_TRUNK, int(self.spec.adv_template_size)),
        )
        self.adv_template_emb = nn.Embedding(int(self.spec.adv_template_size), D_COND_EMB)
        self.adv_stop_head = nn.Sequential(
            nn.Linear(D_TRUNK + 2 * D_COND_EMB, D_TRUNK),
            nn.ReLU(),
            nn.Linear(D_TRUNK, int(self.spec.adv_stop_size)),
        )

        # -------- Controller factor heads --------
        self.ctrl_evict_head = nn.Linear(D_TRUNK, int(self.spec.ctrl_evict_size))
        self.ctrl_evict_emb = nn.Embedding(int(self.spec.ctrl_evict_size), D_COND_EMB)

        self.ctrl_budget_head = nn.Sequential(
            nn.Linear(D_TRUNK + D_COND_EMB, D_TRUNK),
            nn.ReLU(),
            nn.Linear(D_TRUNK, int(self.spec.ctrl_budget_size)),
        )
        self.ctrl_budget_emb = nn.Embedding(int(self.spec.ctrl_budget_size), D_COND_EMB)

        self.ctrl_heur_head = nn.Sequential(
            nn.Linear(D_TRUNK + 2 * D_COND_EMB, D_TRUNK),
            nn.ReLU(),
            nn.Linear(D_TRUNK, int(self.spec.ctrl_heur_size)),
        )

        self.value_head = nn.Linear(D_TRUNK, 1)

        self._infer_perf_enabled = True
        self._infer_perf_sync_cuda = True
        self._infer_perf_every = 10000
        self.reset_infer_perf()

    def _pool_set(
        self,
        x: torch.Tensor,                  # [B, N, D]
        mask: Optional[torch.Tensor],     # [B, N]
    ) -> torch.Tensor:
        x_mean = masked_mean(x, mask, dim=1)
        x_max = masked_max(x, mask, dim=1)
        x_min = masked_min(x, mask, dim=1)
        return torch.cat([x_mean, x_max, x_min], dim=-1)  # [B, 3D]

    def _encode_trunk(
        self,
        prefill_req_features: torch.Tensor,      # [B, Np, Dp]
        decode_req_features: torch.Tensor,       # [B, Nd, Dd]
        global_features: torch.Tensor,           # [B, Dg]
        prefill_req_mask: Optional[torch.Tensor],
        decode_req_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz = prefill_req_features.size(0)

        p = self.norm_prefill(prefill_req_features)
        d = self.norm_decode(decode_req_features)
        g = self.norm_global(global_features)

        p_enc = self.prefill_encoder(p.reshape(bsz * self.n_prefill_req, self.d_prefill_req)).reshape(
            bsz, self.n_prefill_req, D_REQ_EMB
        )
        d_enc = self.decode_encoder(d.reshape(bsz * self.n_decode_req, self.d_decode_req)).reshape(
            bsz, self.n_decode_req, D_REQ_EMB
        )
        g_enc = self.global_encoder(g)

        pooled_p = self._pool_set(p_enc, prefill_req_mask)
        pooled_d = self._pool_set(d_enc, decode_req_mask)

        trunk_in = torch.cat([pooled_p, pooled_d, g_enc], dim=-1)
        h = self.trunk(trunk_in)
        return h  # [B, D_TRUNK]

    def _compose_adv_flat_logits(self, h: torch.Tensor) -> torch.Tensor:
        bsz = h.size(0)
        flat = h.new_zeros((bsz, self.num_actions_adversary))

        launch_logits = self.adv_launch_head(h)  # [B, L]
        L = int(self.spec.adv_launch_size)
        T = int(self.spec.adv_template_size)
        S = int(self.spec.adv_stop_size)

        for l in range(L):
            l_idx = torch.full((bsz,), l, dtype=torch.long, device=h.device)
            l_emb = self.adv_launch_emb(l_idx)
            h_l = torch.cat([h, l_emb], dim=-1)

            template_logits = self.adv_template_head(h_l)  # [B, T]

            if l == 0:
                # template ignored in flattening; use template 0 as sentinel condition
                t0_idx = torch.zeros((bsz,), dtype=torch.long, device=h.device)
                t0_emb = self.adv_template_emb(t0_idx)
                h_l_t0 = torch.cat([h_l, t0_emb], dim=-1)
                stop_logits = self.adv_stop_head(h_l_t0)  # [B, S]
                for s in range(S):
                    idx = self.spec.adv_flat_index(0, 0, s)
                    flat[:, idx] = launch_logits[:, 0] + stop_logits[:, s]
            else:
                for t in range(T):
                    t_idx = torch.full((bsz,), t, dtype=torch.long, device=h.device)
                    t_emb = self.adv_template_emb(t_idx)
                    h_l_t = torch.cat([h_l, t_emb], dim=-1)
                    stop_logits = self.adv_stop_head(h_l_t)  # [B, S]
                    for s in range(S):
                        idx = self.spec.adv_flat_index(l, t, s)
                        flat[:, idx] = launch_logits[:, l] + template_logits[:, t] + stop_logits[:, s]

        return flat

    def _compose_ctrl_flat_logits(self, h: torch.Tensor) -> torch.Tensor:
        bsz = h.size(0)
        flat = h.new_zeros((bsz, self.num_actions_controller))

        evict_logits = self.ctrl_evict_head(h)  # [B, E]
        E = int(self.spec.ctrl_evict_size)
        B = int(self.spec.ctrl_budget_size)
        H = int(self.spec.ctrl_heur_size)

        for e in range(E):
            e_idx = torch.full((bsz,), e, dtype=torch.long, device=h.device)
            e_emb = self.ctrl_evict_emb(e_idx)
            h_e = torch.cat([h, e_emb], dim=-1)

            budget_logits = self.ctrl_budget_head(h_e)  # [B, B]

            for b in range(B):
                b_idx = torch.full((bsz,), b, dtype=torch.long, device=h.device)
                b_emb = self.ctrl_budget_emb(b_idx)
                h_e_b = torch.cat([h_e, b_emb], dim=-1)

                heur_logits = self.ctrl_heur_head(h_e_b)  # [B, H]

                for hh in range(H):
                    idx = self.spec.ctrl_flat_index(e, b, hh)
                    flat[:, idx] = evict_logits[:, e] + budget_logits[:, b] + heur_logits[:, hh]

        return flat

    ## BATCH POLICY LOGIT CREATING METHOD:

    def _compose_ctrl_flat_logits_batch(self, h: torch.Tensor) -> torch.Tensor:
        """
        Batched equivalent of _compose_ctrl_flat_logits.
        Produces flat controller logits with index order matching:
        idx = (e * B + b) * H + hh
        """
        bsz = h.size(0)
        device = h.device

        E = int(self.spec.ctrl_evict_size)
        B = int(self.spec.ctrl_budget_size)
        H = int(self.spec.ctrl_heur_size)

        # [B, E]
        evict_logits = self.ctrl_evict_head(h)

        # Build h_e for all evict choices at once: [B, E, D_TRUNK + D_COND_EMB]
        e_ids = torch.arange(E, device=device, dtype=torch.long)
        e_emb = self.ctrl_evict_emb(e_ids)  # [E, D_COND_EMB]
        h_e = torch.cat(
            [
                h.unsqueeze(1).expand(-1, E, -1),
                e_emb.unsqueeze(0).expand(bsz, -1, -1),
            ],
            dim=-1,
        )

        # Budget logits conditioned on evict for all e at once: [B, E, B]
        budget_logits = self.ctrl_budget_head(
            h_e.reshape(bsz * E, -1)
        ).reshape(bsz, E, B)

        # Build h_e_b for all (e,b) at once: [B, E, B, D_TRUNK + 2*D_COND_EMB]
        b_ids = torch.arange(B, device=device, dtype=torch.long)
        b_emb = self.ctrl_budget_emb(b_ids)  # [B, D_COND_EMB]
        h_e_b = torch.cat(
            [
                h_e.unsqueeze(2).expand(-1, -1, B, -1),
                b_emb.unsqueeze(0).unsqueeze(0).expand(bsz, E, -1, -1),
            ],
            dim=-1,
        )

        # Heuristic logits conditioned on (e,b) for all combos: [B, E, B, H]
        heur_logits = self.ctrl_heur_head(
            h_e_b.reshape(bsz * E * B, -1)
        ).reshape(bsz, E, B, H)

        # Compose joint logits in log-space and flatten to controller action space.
        flat4 = (
            evict_logits.unsqueeze(-1).unsqueeze(-1) +
            budget_logits.unsqueeze(-1) +
            heur_logits
        )  # [B, E, B, H]

        flat = flat4.reshape(bsz, E * B * H)

        if flat.size(1) != self.num_actions_controller:
            raise RuntimeError(
                f"controller flat size mismatch: got {flat.size(1)}, expected {self.num_actions_controller}"
            )
        return flat


    def _compose_adv_flat_logits_batch(self, h: torch.Tensor) -> torch.Tensor:
        """
        Batched equivalent of _compose_adv_flat_logits.
        Keeps original semantics:
        - launch=0 ignores template in flattening and uses template=0 sentinel for stop conditioning
        - launch>=1 uses full (launch, template, stop)
        """
        bsz = h.size(0)
        device = h.device

        L = int(self.spec.adv_launch_size)
        T = int(self.spec.adv_template_size)
        S = int(self.spec.adv_stop_size)

        # [B, L]
        launch_logits = self.adv_launch_head(h)

        # Build h_l for all launches: [B, L, D_TRUNK + D_COND_EMB]
        l_ids = torch.arange(L, device=device, dtype=torch.long)
        l_emb = self.adv_launch_emb(l_ids)  # [L, D_COND_EMB]
        h_l = torch.cat(
            [
                h.unsqueeze(1).expand(-1, L, -1),
                l_emb.unsqueeze(0).expand(bsz, -1, -1),
            ],
            dim=-1,
        )

        # Template logits for all launches: [B, L, T]
        template_logits = self.adv_template_head(
            h_l.reshape(bsz * L, -1)
        ).reshape(bsz, L, T)

        # Build h_l_t for all (launch, template): [B, L, T, D_TRUNK + 2*D_COND_EMB]
        t_ids = torch.arange(T, device=device, dtype=torch.long)
        t_emb = self.adv_template_emb(t_ids)  # [T, D_COND_EMB]
        h_l_t = torch.cat(
            [
                h_l.unsqueeze(2).expand(-1, -1, T, -1),
                t_emb.unsqueeze(0).unsqueeze(0).expand(bsz, L, -1, -1),
            ],
            dim=-1,
        )

        # Stop logits for all (launch, template): [B, L, T, S]
        stop_logits = self.adv_stop_head(
            h_l_t.reshape(bsz * L * T, -1)
        ).reshape(bsz, L, T, S)

        # launch=0 branch: template ignored in flattening; use template=0 sentinel
        base = launch_logits[:, 0].unsqueeze(-1) + stop_logits[:, 0, 0, :]  # [B, S]

        # launch>=1 branch
        if L > 1:
            rest = (
                launch_logits[:, 1:].unsqueeze(-1).unsqueeze(-1) +
                template_logits[:, 1:, :].unsqueeze(-1) +
                stop_logits[:, 1:, :, :]
            )  # [B, L-1, T, S]
            flat = torch.cat([base, rest.reshape(bsz, -1)], dim=1)
        else:
            flat = base

        if flat.size(1) != self.num_actions_adversary:
            raise RuntimeError(
                f"adversary flat size mismatch: got {flat.size(1)}, expected {self.num_actions_adversary}"
            )
        return flat
        


    def forward(
        self,
        *,
        player: Player,
        global_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,

        # New GV2 split inputs:
        prefill_req_features: Optional[torch.Tensor] = None,
        decode_req_features: Optional[torch.Tensor] = None,
        prefill_req_mask: Optional[torch.Tensor] = None,
        decode_req_mask: Optional[torch.Tensor] = None,

        # Backward-compat old path:
        req_features: Optional[torch.Tensor] = None,
        req_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preferred:
          prefill_req_features [B,Np,Dp], decode_req_features [B,Nd,Dd], global_features [B,Dg]
        Temporary fallback:
          req_features [B,N,D] splits into prefill/decode by first Np / next Nd slices.
        """

        if prefill_req_features is None or decode_req_features is None:
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

            # prefill slice
            p_take = min(self.n_prefill_req, n_all)
            prefill_req_features = req_features[:, :p_take, : self.d_prefill_req]
            if p_take < self.n_prefill_req:
                pad = req_features.new_zeros((bsz, self.n_prefill_req - p_take, self.d_prefill_req))
                prefill_req_features = torch.cat([prefill_req_features, pad], dim=1)

            # decode slice
            d_start = self.n_prefill_req
            d_end = min(d_start + self.n_decode_req, n_all)
            decode_req_features = req_features[:, d_start:d_end, : self.d_decode_req]
            d_take = decode_req_features.size(1)
            if d_take < self.n_decode_req:
                pad = req_features.new_zeros((bsz, self.n_decode_req - d_take, self.d_decode_req))
                decode_req_features = torch.cat([decode_req_features, pad], dim=1)

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

        # strict checks (split path)
        if prefill_req_features.dim() != 3 or prefill_req_features.size(1) != self.n_prefill_req or prefill_req_features.size(2) != self.d_prefill_req:
            raise ValueError(
                f"prefill_req_features must be [B,{self.n_prefill_req},{self.d_prefill_req}], got {tuple(prefill_req_features.shape)}"
            )
        if decode_req_features.dim() != 3 or decode_req_features.size(1) != self.n_decode_req or decode_req_features.size(2) != self.d_decode_req:
            raise ValueError(
                f"decode_req_features must be [B,{self.n_decode_req},{self.d_decode_req}], got {tuple(decode_req_features.shape)}"
            )
        if global_features.dim() != 2 or global_features.size(1) != self.d_global:
            raise ValueError(
                f"global_features must be [B,{self.d_global}], got {tuple(global_features.shape)}"
            )

        h = self._encode_trunk(
            prefill_req_features=prefill_req_features,
            decode_req_features=decode_req_features,
            global_features=global_features,
            prefill_req_mask=prefill_req_mask,
            decode_req_mask=decode_req_mask,
        )

        if player == "adversary":
            # policy_logits = self._compose_adv_flat_logits(h)
            policy_logits = self._compose_adv_flat_logits_batch(h)
        elif player == "controller":
            # policy_logits = self._compose_ctrl_flat_logits(h)
            policy_logits = self._compose_ctrl_flat_logits_batch(h)
        else:
            raise ValueError(f"Unknown player={player!r}; expected 'controller' or 'adversary'")

        if action_mask is not None:
            if action_mask.shape != policy_logits.shape:
                raise ValueError(
                    f"action_mask shape {tuple(action_mask.shape)} != logits shape {tuple(policy_logits.shape)}"
                )
            policy_logits = policy_logits.masked_fill(~action_mask, float("-inf"))

        value_raw = self.value_head(h)  # [B,1]
        return policy_logits, value_raw


    # -------------------------------------------------------------------------
    # Inference helpers (bridge to MCTS)
    # -------------------------------------------------------------------------

    # @torch.no_grad()
    @torch.inference_mode()
    # def infer_from_inputs(
    #     self,
    #     inputs: ModelInputs,
    #     player: Player,
    #     *,
    #     device: Optional[torch.device] = None,
    # ) -> Tuple[float, list[float]]:
    #     """
    #     Returns:
    #       value_controller: float
    #       priors: list[float] aligned with deterministic action indexing for that player
    #               (i.e., priors[i] = π(a_i | s) for action index i)
    #     """
    #     dev = device or next(self.parameters()).device

    #     req_features = inputs.req_features.to(dev)
    #     global_features = inputs.global_features.to(dev)
    #     req_mask = inputs.req_mask.to(dev) if inputs.req_mask is not None else None
    #     action_mask = inputs.action_mask.to(dev) if inputs.action_mask is not None else None

    #     policy_logits, value_logits = self.forward(
    #         req_features=req_features,
    #         global_features=global_features,
    #         player=player,
    #         req_mask=req_mask,
    #         action_mask=action_mask,
    #     )

    #     # Convert value support logits -> scalar in real units (controller perspective)
    #     value = self.value_scalar_from_logits(value_logits).squeeze(0).item()

    #     # Convert logits -> normalized priors (softmax over valid actions)
    #     priors = F.softmax(policy_logits, dim=-1).squeeze(0).to("cpu").tolist()
    #     return value, priors


    @torch.inference_mode()
    def infer_from_inputs(
        self,
        inputs: ModelInputs,
        player: Player,
        *,
        device: Optional[torch.device] = None,
    ) -> Tuple[float, list[float]]:
        dev = device or next(self.parameters()).device

        profile = bool(getattr(self, "_infer_perf_enabled", False))
        sync_cuda = bool(getattr(self, "_infer_perf_sync_cuda", True)) and (dev.type == "cuda")

        def _sync():
            if sync_cuda:
                torch.cuda.synchronize(dev)

        if profile:
            _sync()
            t_all = time.perf_counter()

            _sync()
            t = time.perf_counter()

        
        prefill_req_features = inputs.prefill_req_features.to(dev)
        decode_req_features = inputs.decode_req_features.to(dev)
        global_features = inputs.global_features.to(dev)

        prefill_req_mask = inputs.prefill_req_mask.to(dev) if inputs.prefill_req_mask is not None else None
        decode_req_mask = inputs.decode_req_mask.to(dev) if inputs.decode_req_mask is not None else None
        action_mask = inputs.action_mask.to(dev) if inputs.action_mask is not None else None

        req_features = inputs.req_features.to(dev) if inputs.req_features is not None else None
        req_mask = inputs.req_mask.to(dev) if inputs.req_mask is not None else None


        if profile:
            _sync()
            self._infer_perf["to_device"] += time.perf_counter() - t

            _sync()
            t = time.perf_counter()

        policy_logits, value_raw = self.forward(
            player=player,
            prefill_req_features=prefill_req_features,
            decode_req_features=decode_req_features,
            global_features=global_features,
            prefill_req_mask=prefill_req_mask,
            decode_req_mask=decode_req_mask,
            action_mask=action_mask,
            req_features=req_features,   # fallback
            req_mask=req_mask,           # fallback
        )


        if profile:
            _sync()
            self._infer_perf["forward"] += time.perf_counter() - t

            _sync()
            t = time.perf_counter()

        value = self.value_scalar_from_logits(value_raw).squeeze(0).item()

        if profile:
            _sync()
            self._infer_perf["value_decode"] += time.perf_counter() - t

            _sync()
            t = time.perf_counter()

        priors_t = F.softmax(policy_logits, dim=-1).squeeze(0)

        if profile:
            _sync()
            self._infer_perf["softmax"] += time.perf_counter() - t

            _sync()
            t = time.perf_counter()

        priors = priors_t.to("cpu").tolist()

        if profile:
            _sync()
            self._infer_perf["to_cpu_list"] += time.perf_counter() - t

            _sync()
            self._infer_perf["total"] += time.perf_counter() - t_all
            self._infer_perf["calls"] += 1

            # every = int(getattr(self, "_infer_perf_every", 10000))
            # if every > 0 and (self._infer_perf["calls"] % every == 0):
            #     self._print_infer_perf(prefix=f"[INFER_PERF player={player}]")

        return value, priors




    # TODO: I dont think this is needed even in MCTS so remove it afterwards 
    @torch.no_grad()
    def infer_from_state(
        self,
        state: VidurMCTSState,
        player: Player,
        *,
        build_inputs: Callable[[VidurMCTSState, Player, torch.device], ModelInputs],
        device: Optional[torch.device] = None,
    ) -> Tuple[float, list[float]]:
        """
        This is the method MCTS should call.

        You provide `build_inputs(state, player, device)` elsewhere (e.g. mcts/DNN/infer.py),
        so environment.py stays clean.
        """
        dev = device or next(self.parameters()).device
        inputs = build_inputs(state, player, dev)
        return self.infer_from_inputs(inputs, player, device=dev)


        


    ##==============================    
    # HELPERS FOR TRAINING & INFERENCE with scalar normalization conversions
    ##==============================


    def scale_value(self, value_real: torch.Tensor) -> torch.Tensor:
        """
        Converts real value to normalized model value.
        """
        return normalize_value_real(value_real)

    def unscale_value(self, value_scaled: torch.Tensor) -> torch.Tensor:
        """
        Converts normalized model value to real value.
        """
        return denormalize_value_model(value_scaled)

    # def value_target_to_support(self, value_real: torch.Tensor) -> torch.Tensor:
    #     """
    #     v_real: [B] or [B, T]
    #     Converts real value targets to support logits        
    #     """
    #     value_scaled = self.scale_value(value_real)
    #     return scalar_to_support(value_scaled, SUPPORT_SIZE)

    # def value_scalar_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
    #     """
    #     value_logits: [B, 2*support_size + 1] predicted logits in SCALED units 
    #     Converts value logits to real scalar values
    #     """
    #     value_scaled = support_to_scalar(value_logits, SUPPORT_SIZE)
    #     return self.unscale_value(value_scaled)

    def value_target_to_model(self, value_real: torch.Tensor) -> torch.Tensor:
        return normalize_value_real(value_real)

    # Backward-compatible alias used by older trainer code paths.
    def value_target_to_support(self, value_real: torch.Tensor) -> torch.Tensor:
        return self.value_target_to_model(value_real)

    def value_normalized_from_raw(self, value_raw: torch.Tensor) -> torch.Tensor:
        return -torch.sigmoid(value_raw).squeeze(-1)

    def value_scalar_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        # Backward-compatible name; value_logits is now the raw scalar-head output.
        value_norm = self.value_normalized_from_raw(value_logits)
        return denormalize_value_model(value_norm)




    def reset_infer_perf(self) -> None:
        self._infer_perf = {
            "calls": 0,
            "total": 0.0,
            "to_device": 0.0,
            "forward": 0.0,
            "value_decode": 0.0,
            "softmax": 0.0,
            "to_cpu_list": 0.0,
        }

    def _print_infer_perf(self, prefix: str = "[INFER_PERF]") -> None:
        p = self._infer_perf
        c = max(1, int(p["calls"]))
        tot = max(1e-12, float(p["total"]))

        def pct(x: float) -> float:
            return 100.0 * float(x) / tot

        # print(
        #     f"{prefix}\n"
        #     f"  calls={c}\n"
        #     f"  total={p['total']:.3f}s total_per_call={p['total']/c:.6f}s\n"
        #     f"  to_device={p['to_device']:.3f}s ({pct(p['to_device']):.1f}%)\n"
        #     f"  forward={p['forward']:.3f}s ({pct(p['forward']):.1f}%)\n"
        #     f"  value_decode={p['value_decode']:.3f}s ({pct(p['value_decode']):.1f}%)\n"
        #     f"  softmax={p['softmax']:.3f}s ({pct(p['softmax']):.1f}%)\n"
        #     f"  to_cpu_list={p['to_cpu_list']:.3f}s ({pct(p['to_cpu_list']):.1f}%)"
        # )
























