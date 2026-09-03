"""Freeze the simulator-native MCTS configuration for lean vLLM deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .native_dnn_mcts_planner import (
    _build_native_cfg,
    NativeDNNMCTSPlannerConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = NativeDNNMCTSPlannerConfig(model_bundle=args.model_bundle)
    payload = _build_native_cfg(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    document = {
        "schema": "vidur_vllm_native_mcts_cfg_v1",
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "payload_sha256": document["payload_sha256"],
                "component_table_count": len(
                    payload.get("execution_predictor_component_tables", {})
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

