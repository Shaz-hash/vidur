
## (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import torch

from ....environment import VidurMCTSState

# =============================================================================
# Types aka interfaces / Protocols
# =============================================================================

@dataclass(frozen=True)
class ModelInputs:
    # Preferred split inputs
    prefill_req_features: torch.Tensor          # [1, N_PREFILL_REQ, D_PREFILL_REQ]
    decode_req_features: torch.Tensor           # [1, N_DECODE_REQ,  D_DECODE_REQ]
    global_features: torch.Tensor               # [1, D_GLOBAL]

    prefill_req_mask: Optional[torch.Tensor] = None   # [1, N_PREFILL_REQ]
    decode_req_mask: Optional[torch.Tensor] = None    # [1, N_DECODE_REQ]
    action_mask: Optional[torch.Tensor] = None        # [1, A(player)]

    # Optional legacy aliases
    req_features: Optional[torch.Tensor] = None
    req_mask: Optional[torch.Tensor] = None

    # Opt-in side channel for non-tensor feature builders (e.g. classical v4
    # HGB wrappers that need the raw simulator_snapshot+stats to produce the
    # 224-d state-local features). Only populated when build_model_inputs is
    # called with the extras flag enabled. The torch DNN ignores this field.
    extras: Optional[Dict[str, Any]] = None


class DNNModel(Protocol):
    def infer_from_state(
        self,
        state: VidurMCTSState,
        player: Player,
        *,
        build_inputs: Callable[[VidurMCTSState, Player, torch.device], ModelInputs],
        device: Optional[torch.device] = None,
    ) -> Tuple[float, list[float]]:
        """Return (value_controller, priors_for_actions)."""
        ...

Player = str  # expected: "controller" or "adversary"