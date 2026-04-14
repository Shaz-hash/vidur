
## (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

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