

## (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

import math 
from abc import ABC, abstractmethod


from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F




##==============================
# HARDCODED PARAMETERS
##==============================

N_REQ : int = 20  ## --> Number of requests in a batch
D_REQ : int = 5   ## --> Number of features per request
D_GLOBAL : int = 4  ## --> Number of global features
NUM_ACTIONS : int = 24  ## --> Number of possible actions
# Value Scaling : the bin size for value head support is calculated as (Max SLO Cost for a move : 5seconds + HARDMISS_PENALTY)/ (1-gamma) 
HARD_MISS_PENALTY : int = 15  # seconds
MAX_SLO_COST : int = 5  # seconds
GAMMA : float = 0.98  # discount factor
VALUE_SCALE : int = 20
SUPPORT_SIZE = math.ceil(((MAX_SLO_COST + HARD_MISS_PENALTY) / (1 - GAMMA)) / VALUE_SCALE)  # Assuming support size is equal to value scale for simplicity


##------------------------------
# HELPER FUNCTIONS
##------------------------------

def mlp(in_dim: int, hidden: list[int], out_dim: int, act=nn.ReLU) -> nn.Sequential:
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
    if mask is not None:
        return x.mean(dim= dim)
    
    m = mask.float().unsqueeze(-1)  # [B, N, 1]
    denom = m.sum(dim= dim, keepdim= True).clamp(min= 1.0)  
    return (x * m).sum(dim= dim) / denom


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
    if mask is not None:
        return x.max(dim= dim).values

    x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    return torch.nan_to_num(out,neginf=0.0)



#------------------------------
# Support Helpers for AlphaZeroModel
#------------------------------

def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Converts scalar values to a categorical distribution over a discrete support i.e. with (2 * support_size + 1) categories.
    x: [B] or [B, 1] or [B, T]
    returns: [..., 2*support_size+1] distribution logits target (soft one-hot)
    Uses MuZero scaling: y = sign(x)*(sqrt(|x|+1)-1) + 0.001*x


    Args:
        x (torch.Tensor): Input tensor of shape [B] containing scalar values.
        support_size (int): Size of the discrete support.

    Returns:
        torch.Tensor: Categorical distribution tensor of shape [B, support_size].
    """
    # Ensure Shape [.... , 1] for generality
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # [B, 1]

    # Scale (compress) value range
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + 0.001 * x

    # Clamp to support 
    x = torch.clamp(x, -support_size, support_size)

    floor = torch.floor(x)
    prob = x - floor # fractional part

    # One/Two hot distribution
    # Output shape : [..., 2*support_size + 1]
    last_dim = 2 * support_size + 1
    out_shape = list(x.shape[:-1]) + [last_dim]
    logits = torch.zeros(out_shape, device= x.device)  # 

    # Put mass on floor bin 
    idx0 = (floor + support_size).long()  # shift to [0, 2*support_size]
    logits.scatter_add_(-1, idx0.unsqueeze(-1), (1 - prob).unsqueeze(-1))

    # Put remaining mass on next bin (floor+1)
    idx1 = idx0 + 1

    # if idx1 is out of bounds, clamp to last bin a.k.a drop the excess mass
    valid = (idx1 >= 0) & (idx1 < last_dim)
    prob1 = torch.where(valid, prob, torch.zeros_like(prob))
    idx1 = torch.where(valid, idx1, idx0)  # if invalid, put mass back to idx0
    logits.scatter_add_(-1, idx1.unsqueeze(-1), prob1.unsqueeze(-1))

    return logits


def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Converts a categorical distribution over a discrete support back to scalar values.
    logits: [..., 2*support_size + 1]
    returns: [...] scalar values

    Args:
        logits (torch.Tensor): Input tensor of shape [..., 2*support_size + 1] containing logits.
        support_size (int): Size of the discrete support.

    Returns:
        torch.Tensor: Scalar tensor of shape [...].
    """
    probs = F.softmax(logits, dim= -1)  # [..., 2*support_size + 1]
    support_values = torch.arange(-support_size, support_size + 1, device= logits.device, dtype= probs.dtype)  # [2*support_size + 1]
    
    # broadcast support_values to match probs shape
    x = (probs * support_values).sum(dim= -1)
    # Inverse scaling (MuZero)
    # x = sign(x) * ( ((sqrt(1+4*0.001*(|x|+1+0.001)) - 1) / (2*0.001))^2 - 1 )
    eps = 0.001
    x = torch.sign(x) * ((((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps))) - 1) / (2 * eps)) ** 2 - 1)

    return x


class AlphaZeroModel(nn.Module):

    """
    Inputs: 
        req_features : torch.Tensor : [B, N_REQ : 20, D_REQ] (pad with zeros if < 20 requests)
        global_features : torch.Tensor : [B, D_GLOBAL]
        req_mask : torch.Tensor : [B, N_REQ] (boolean) True for valid requests, False for padded requests
        action_mask : torch.Tensor : [B, NUM_ACTIONS : 24] (boolean) True for valid actions, False for invalid actions
    Outputs:
        policy_logits : torch.Tensor : [B, NUM_ACTIONS : 24] (logits for each action)
        value : [B] scalar in *real units* after applying VALUE_SCALE : 20
    """


    def __init__(self):

        super().__init__() # initialize the nn.Module

        # Normalisation over each request's features and global features vector
        self.norm_req = nn.LayerNorm(D_REQ)
        self.norm_global = nn.LayerNorm(D_GLOBAL)

        # Request Encoder MLP (DEEPSet style)
        self.req_encoder = mlp(in_dim= D_REQ, hidden= [128,128], out_dim= 64)

        # Global Encoder MLP
        self.global_encoder = mlp(in_dim= D_GLOBAL, hidden= [128], out_dim= 64)

        # Trunk after pooling (mean + max + global)
        turnk_in = 64 * 3  # mean pooled reqs + max pooled reqs + global encoded
        self.trunk = mlp(trunk_in, hidden = [128, 128], out_dim= 128)

        ## HEADS => Policy Head and Value Head
        self.policy_head = nn.Linear(128, NUM_ACTIONS)
        self.value_head = nn.Linear(128, 2 * support_size + 1)  # outputs logits over support

        
        

    def forward(
        self,
        req_features: torch.Tensor,
        global_features: torch.Tensor,
        req_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AlphaZeroModel.

        Args:
            req_features (torch.Tensor): Request features tensor of shape [B, N_REQ : 20 e.g., D_REQ].
            global_features (torch.Tensor): Global features tensor of shape [B, D_GLOBAL].
            req_mask (Optional[torch.Tensor]): Optional request mask of shape [B, N_REQ].
            action_mask (Optional[torch.Tensor]): Optional action mask of shape [B, NUM_ACTIONS].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy logits and value tensor.
        """

        assert req_features.dim() == 3 and req_features.size(1) == N_REQ and req_features.size(2) == D_REQ, f"req_features must be of shape [B, N_REQ, D_REQ], got {req_features.shape}"
        B, N , D = req_features.shape
        assert N == N_REQ and D == D_REQ, f"Expected req_features shape [B, {N_REQ}, {D_REQ}], got {req_features.shape}" 
        assert global_features.dim() == 2 and global_features.size(1) == D_GLOBAL, f"global_features must be of shape [B, D_GLOBAL], got {global_features.shape}"
        

        # Normalise : 
        req_features = self.norm_req(req_features)  # [B, N_REQ, D_REQ]
        global_features = self.norm_global(global_features)  # [B, D_GLOBAL]

        # Encode requests individually (shared weights)
        req_encoded = self.req_encoder(req_features.reshape(B * N , D)).reshape(B, N , -1)  # [B, N_REQ, 64]

        # Pool over the set (order-invariant)
        req_mean = masked_mean(req_encoded, req_mask, dim= 1)  # [B, 64]
        req_max = masked_max(req_encoded, req_mask, dim=1)  # [B, 64]

        # Encode global features
        global_encoded = self.global_encoder(global_features)  # [B, 64]

        # Concatenate pooled reqs and global encoding i.e. combine + trunk 
        h = self.trunk(torch.cat([req_mean, req_max, global_encoded], dim= -1))  # [B, 192] -> [B, 128]
    
        # Policy Head
        policy_logits = self.policy_head(h)  # [B, NUM_ACTIONS]
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, float('-inf'))
        
        # Value Head
        # value = self.value_head(h).squeeze(-1) * VALUE_SCALE  # [B]
        value_logits = self.value_head(h)  # [B, 2*support_size + 1]

        return policy_logits, value_logits


    ##==============================    
    # HELPERS FOR TRAINING & INFERENCE with SUPPORT/SCALAR conversions
    ##==============================


    def scale_value(self, value_real: torch.Tensor) -> torch.Tensor:
        """
        Converts real value  to scaled value
        """
        return value_real / VALUE_SCALE

    def unscale_value(self, value_scaled: torch.Tensor) -> torch.Tensor:
        """
        Converts scaled value to real value
        """
        return value_scaled * VALUE_SCALE

    def value_target_to_support(self, value_real: torch.Tensor) -> torch.Tensor:
        """
        v_real: [B] or [B, T]
        Converts real value targets to support logits        
        """
        valzue_scaled = self.scale_value(value_real)
        return scalar_to_support(value_scaled, SUPPORT_SIZE)

    def value_scalar_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        """
        value_logits: [B, 2*support_size + 1] predicted logits in SCALED units 
        Converts value logits to real scalar values
        """
        value_scaled = support_to_scalar(value_logits, SUPPORT_SIZE)
        return self.unscale_value(value_scaled)
































