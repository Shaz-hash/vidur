from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Iterable

import torch

from .config import load_config


PACKAGES = (
    "vllm",
    "flashinfer-python",
    "torch",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-runtime",
)


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the raw profiling manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config()
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in PACKAGES},
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ).splitlines(),
        "config": config.to_dict(),
        "measurement": {
            "vllm_boundary": (
                "CUDA event at first decoder layer entry through CUDA event at "
                "last decoder layer exit"
            ),
            "vidur_boundary": "ExecutionTime.model_time",
            "batch_construction": (
                "vLLM model-runner _dummy_run invoked through worker RPC with "
                "an asserted exact equal-size request shape"
            ),
            "cpu_and_scheduler_overhead_included": False,
            "embedding_and_final_softmax_included": False,
            "calibration_applied": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
