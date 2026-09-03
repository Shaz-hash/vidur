"""Strict file channel for CUDA-forward batch timings.

vLLM serializes ``ModelRunnerOutput`` between the worker and scheduler
processes, so arbitrary attributes attached by a custom worker do not survive.
This channel transports only the measured duration and validates it against
the exact scheduled request/token map before GV3 consumes it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping


def normalized_scheduled_tokens(values: Mapping[object, object]) -> dict[str, int]:
    return {str(request_id): int(tokens) for request_id, tokens in values.items()}


def append_gpu_forward_timing(
    path: str | Path,
    *,
    scheduled_tokens: Mapping[object, object],
    gpu_forward_s: float,
) -> None:
    duration_s = float(gpu_forward_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"invalid CUDA model-forward duration {duration_s!r}")
    payload = {
        "scheduled_tokens": normalized_scheduled_tokens(scheduled_tokens),
        "gpu_forward_s": duration_s,
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


class GPUForwardTimingReader:
    """Consume each timing record exactly once and verify its batch identity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0

    def read_for_batch(self, scheduled_tokens: Mapping[object, object]) -> float:
        expected = normalized_scheduled_tokens(scheduled_tokens)
        while True:
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    handle.seek(self._offset)
                    line = handle.readline()
                    if not line:
                        raise RuntimeError(
                            f"missing GPU-forward timing record at byte {self._offset}"
                        )
                    self._offset = handle.tell()
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"GPU-forward timing channel does not exist: {self.path}"
                ) from exc

            try:
                payload = json.loads(line)
                actual = normalized_scheduled_tokens(payload["scheduled_tokens"])
                duration_s = float(payload["gpu_forward_s"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid GPU-forward timing record: {line!r}") from exc
            if not math.isfinite(duration_s) or duration_s <= 0.0:
                raise RuntimeError(
                    f"invalid GPU-forward timing duration {duration_s!r}"
                )
            if actual == expected:
                return duration_s
            if actual and all(
                request_id.startswith("_warmup_") for request_id in actual
            ):
                continue
            raise RuntimeError(
                "GPU-forward timing batch mismatch: "
                f"worker={actual}, scheduler={expected}"
            )
