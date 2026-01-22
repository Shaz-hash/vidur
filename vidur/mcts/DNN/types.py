
## (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import torch

from ..environment import VidurMCTSState


# =============================================================================
# Types aka interfaces / Protocols
# =============================================================================

@dataclass(frozen=True)
class ModelInputs:
    """
    What the feature-builder should produce from (state, player).
    Shapes are single-batch (B=1) for inference in MCTS.
    """
    req_features: torch.Tensor          # [1, N_REQ, D_REQ]
    global_features: torch.Tensor       # [1, D_GLOBAL]
    req_mask: Optional[torch.Tensor] = None     # [1, N_REQ] bool, True = valid
    action_mask: Optional[torch.Tensor] = None  # [1, num_actions(player)] bool, True = valid


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