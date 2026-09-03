"""Lean production entry point for the promoted native DNN MCTS planner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .native_dnn_mcts_planner import (
    NativeDNNMCTSPlannerConfig,
    PromotedNativeDNNMCTSPlanner,
)


def load_frozen_native_cfg(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen native MCTS config: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "vidur_vllm_native_mcts_cfg_v1":
        raise ValueError(f"unsupported frozen native MCTS config schema: {path}")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise TypeError(f"frozen native MCTS config payload must be an object: {path}")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed = hashlib.sha256(encoded).hexdigest()
    if observed != str(document.get("payload_sha256", "")):
        raise ValueError(f"frozen native MCTS config checksum mismatch: {path}")
    if not payload.get("execution_predictor_component_tables"):
        raise ValueError(f"frozen config has no execution predictor tables: {path}")
    return payload


class ProductionNativeDNNMCTSPlanner(PromotedNativeDNNMCTSPlanner):
    """Promoted DNN planner with no simulator/training-stack startup dependency."""

    @classmethod
    def from_environment(cls) -> "ProductionNativeDNNMCTSPlanner":
        config = NativeDNNMCTSPlannerConfig.from_environment()
        default = Path(__file__).resolve().parent / "artifacts" / "native_mcts_cfg.json"
        path = Path(
            os.environ.get("VIDUR_VLLM_GV3_NATIVE_CFG", str(default))
        ).expanduser()
        return cls(config, cfg_payload=load_frozen_native_cfg(path))

