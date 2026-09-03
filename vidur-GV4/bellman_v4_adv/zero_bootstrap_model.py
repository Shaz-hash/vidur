from __future__ import annotations

from typing import Any

import numpy as np


class ZeroBootstrapModel:
    """Minimal classical_joblib-compatible model for no-bootstrap MCTS runs."""

    def infer_from_inputs(self, inputs: Any, player: str, device: Any = None) -> np.ndarray:
        batch = getattr(inputs, "features", None)
        if batch is None:
            try:
                n = len(inputs)
            except TypeError:
                n = 1
        else:
            n = int(getattr(batch, "shape", [len(batch)])[0])
        return np.zeros(int(n), dtype=np.float32)
