

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
from .types import ModelInputs, Player




# =============================================================================
# Configuration constants (can later be moved into a config object)
# =============================================================================

# Input tensor shapes (your planned feature schema)
N_REQ: int = 20         # max number of (prefill) requests represented
D_REQ: int = 3           # features per request
D_GLOBAL: int = 9        # global features

# Action space sizes (you said you'll make deterministic indexing in environment.py)
NUM_ACTIONS_CONTROLLER: int = 24
NUM_ACTIONS_ADVERSARY: int = 6  # placeholder; update once adversary action indexing is finalized



# Embedding sizes
D_REQ_EMB: int = 16
D_GLOBAL_EMB: int = 16
D_TRUNK: int = 32

# Value range (real units, controller perspective)
V_MIN: float = -50.0
V_MAX: float = 0.0

# Legacy support constants kept for compatibility with older helpers/imports.
NUM_BINS: int = 401
V_STEP: float = (V_MAX - V_MIN) / (NUM_BINS - 1)

# Scalar-head normalization:
#   [-48, 0]   -> [-0.98, 0.0] linearly
#   [-50, -48] -> [-1.0, -0.98] with compressed quadratic tail
V_LINEAR_MIN: float = -48.0
V_NORM_MIN: float = -1.0
V_NORM_MAX: float = 0.0
V_LINEAR_NORM_MIN: float = -0.98
V_TAIL_COMPRESS_POWER: float = 2.0


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

# def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
#     """
#     Converts scalar values to a categorical distribution over a discrete support i.e. with (2 * support_size + 1) categories.
#     x: [B] or [B, 1] or [B, T]
#     returns: [..., 2*support_size+1] distribution logits target (soft one-hot)
#     Uses MuZero scaling: y = sign(x)*(sqrt(|x|+1)-1) + 0.001*x


#     Args:
#         x (torch.Tensor): Input tensor of shape [B] containing scalar values.
#         support_size (int): Size of the discrete support.

#     Returns:
#         torch.Tensor: Categorical distribution tensor of shape [B, support_size].
#     """
#     # Ensure Shape [.... , 1] for generality
#     if x.dim() == 1:
#         x = x.unsqueeze(-1)  # [B, 1]

#     # Scale (compress) value range
#     x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + 0.001 * x

#     # Clamp to support 
#     x = torch.clamp(x, -support_size, support_size)

#     floor = torch.floor(x)
#     prob = x - floor # fractional part

#     # One/Two hot distribution
#     # Output shape : [..., 2*support_size + 1]
#     last_dim = 2 * support_size + 1
#     out_shape = list(x.shape[:-1]) + [last_dim]
#     dist = torch.zeros(out_shape, device=x.device, dtype=x.dtype)

#     # Put mass on floor bin 
#     idx0 = (floor + support_size).long()  # shift to [0, 2*support_size]
#     idx1 = (idx0 + 1).clamp(0, last_dim - 1)

#     p0 = (1.0 - prob)
#     p1 = prob

#     dist.scatter_add_(-1, idx0, p0)
#     dist.scatter_add_(-1, idx1, p1)
#     return dist


# def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
#     """
#     Converts a categorical distribution over a discrete support back to scalar values.
#     logits: [..., 2*support_size + 1]
#     returns: [...] scalar values

#     Args:
#         logits (torch.Tensor): Input tensor of shape [..., 2*support_size + 1] containing logits.
#         support_size (int): Size of the discrete support.

#     Returns:
#         torch.Tensor: Scalar tensor of shape [...].
#     """
#     probs = F.softmax(logits, dim= -1)  # [..., 2*support_size + 1]
#     support_values = torch.arange(-support_size, support_size + 1, device= logits.device, dtype= probs.dtype)  # [2*support_size + 1]
    
#     # broadcast support_values to match probs shape
#     x = (probs * support_values).sum(dim= -1)
#     # Inverse scaling (MuZero)
#     # x = sign(x) * ( ((sqrt(1+4*0.001*(|x|+1+0.001)) - 1) / (2*0.001))^2 - 1 )
#     eps = 0.001
#     x = torch.sign(x) * (
#         (
#             (torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0)
#             / (2.0 * eps)
#         )
#         ** 2
#         - 1.0
#     )
#     return x


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
        num_actions_controller: int = NUM_ACTIONS_CONTROLLER,
        num_actions_adversary: int = NUM_ACTIONS_ADVERSARY,
    ) -> None:
        super().__init__()

        self.num_actions_controller = int(num_actions_controller)
        self.num_actions_adversary = int(num_actions_adversary)

        self.norm_req = nn.Identity()
        self.norm_global = nn.Identity()

        self.req_encoder = nn.Linear(D_REQ, D_REQ_EMB)           # 3 -> 16
        self.global_encoder = nn.Linear(D_GLOBAL, D_GLOBAL_EMB)  # 9 -> 16

        trunk_in = D_REQ_EMB * 3 + D_GLOBAL_EMB  # mean + max + min + global = 16*4 = 64
        self.trunk = nn.Linear(trunk_in, D_TRUNK)  # 64 -> 32

        # Two policy heads (one per player)
        self.policy_head_controller = nn.Linear(D_TRUNK, self.num_actions_controller)  # 32 -> 24
        self.policy_head_adversary = nn.Linear(D_TRUNK, self.num_actions_adversary)    # 32 -> 6

        # Value head (single bounded scalar)
        self.value_head = nn.Linear(D_TRUNK, 1)


        self._infer_perf_enabled = True
        self._infer_perf_sync_cuda = True   # accurate GPU timing; adds overhead
        self._infer_perf_every = 10000
        self.reset_infer_perf()



    def forward(
        self,
        req_features: torch.Tensor,
        global_features: torch.Tensor,
        *,
        player: Player,
        req_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        req_features:   [B, N_REQ, D_REQ]
        global_features:[B, D_GLOBAL]
        req_mask:       [B, N_REQ] bool (True = valid)
        action_mask:    [B, num_actions(player)] bool (True = valid)
        """
        if req_features.dim() != 3 or req_features.size(1) != N_REQ or req_features.size(2) != D_REQ:
            raise ValueError(f"req_features must be [B,{N_REQ},{D_REQ}], got {tuple(req_features.shape)}")
        if global_features.dim() != 2 or global_features.size(1) != D_GLOBAL:
            raise ValueError(f"global_features must be [B,{D_GLOBAL}], got {tuple(global_features.shape)}")

        bsz = req_features.size(0)

        req_features = self.norm_req(req_features)
        global_features = self.norm_global(global_features)

        req_encoded = self.req_encoder(req_features.reshape(bsz * N_REQ, D_REQ)).reshape(bsz, N_REQ, D_REQ_EMB)

        req_mean = masked_mean(req_encoded, req_mask, dim=1)  # [B,64]
        req_max = masked_max(req_encoded, req_mask, dim=1)    # [B,64]
        req_min = masked_min(req_encoded, req_mask, dim=1)  # [B,64]

        global_encoded = self.global_encoder(global_features) # [B,64]

        h = self.trunk(torch.cat([req_mean, req_max, req_min, global_encoded], dim=-1))  # [B,128]

        if player == "controller":
            policy_logits = self.policy_head_controller(h)  # [B, A_c]
        elif player == "adversary":
            policy_logits = self.policy_head_adversary(h)   # [B, A_a]
        else:
            raise ValueError(f"Unknown player={player!r}; expected 'controller' or 'adversary'")

        if action_mask is not None:
            if action_mask.shape != policy_logits.shape:
                raise ValueError(f"action_mask shape {tuple(action_mask.shape)} != logits shape {tuple(policy_logits.shape)}")
            policy_logits = policy_logits.masked_fill(~action_mask, float("-inf"))

        value_raw = self.value_head(h)  # [B, 1]
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

        req_features = inputs.req_features.to(dev)
        global_features = inputs.global_features.to(dev)
        req_mask = inputs.req_mask.to(dev) if inputs.req_mask is not None else None
        action_mask = inputs.action_mask.to(dev) if inputs.action_mask is not None else None

        if profile:
            _sync()
            self._infer_perf["to_device"] += time.perf_counter() - t

            _sync()
            t = time.perf_counter()

        policy_logits, value_raw = self.forward(
            req_features=req_features,
            global_features=global_features,
            player=player,
            req_mask=req_mask,
            action_mask=action_mask,
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
























