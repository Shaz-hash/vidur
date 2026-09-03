"""vLLM worker that reports CUDA compute time for each real batch."""

from __future__ import annotations

from collections.abc import Callable
import math
import os
from pathlib import Path
from typing import Any

import torch
from vllm.v1.worker.gpu_worker import Worker

from .vllm_gpu_timing_channel import append_gpu_forward_timing


GPU_TIMING_PATH_ENV = "VIDUR_VLLM_GPU_TIMING_LOG"
GPU_TIMING_SCOPE_ENV = "VIDUR_VLLM_GPU_TIMING_SCOPE"
WARMUP_PREFIX_ENV = "VIDUR_VLLM_WARMUP_REQUEST_PREFIX"
TIMING_SCOPE_FULL_MODEL = "full_model_forward"
TIMING_SCOPE_TRANSFORMER_BLOCKS = "transformer_blocks"


def _find_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("layers",),
    )
    for attributes in candidates:
        value: Any = model
        for attribute in attributes:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if isinstance(value, torch.nn.ModuleList) and value:
            return value
    raise RuntimeError(
        f"cannot locate transformer layers on vLLM model {type(model).__name__}"
    )


class TimedGPUWorker(Worker):
    """Measure either full forward or the Vidur-modeled transformer stack."""

    def init_device(self) -> None:
        super().init_device()
        timing_path = os.environ.get(GPU_TIMING_PATH_ENV, "").strip()
        if not timing_path:
            raise RuntimeError(f"{GPU_TIMING_PATH_ENV} is required")
        self._vidur_gpu_timing_path = Path(timing_path)
        self._vidur_warmup_prefix = os.environ.get(
            WARMUP_PREFIX_ENV, "cmpl-vidur-warmup-"
        )
        self._vidur_timing_scope = os.environ.get(
            GPU_TIMING_SCOPE_ENV,
            TIMING_SCOPE_FULL_MODEL,
        ).strip()
        if self._vidur_timing_scope not in {
            TIMING_SCOPE_FULL_MODEL,
            TIMING_SCOPE_TRANSFORMER_BLOCKS,
        }:
            raise RuntimeError(
                f"{GPU_TIMING_SCOPE_ENV} must be {TIMING_SCOPE_FULL_MODEL} or "
                f"{TIMING_SCOPE_TRANSFORMER_BLOCKS}"
            )
        self._vidur_forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._vidur_block_starts: list[torch.cuda.Event] = []
        self._vidur_block_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        model = self.model_runner.model
        if self._vidur_timing_scope == TIMING_SCOPE_TRANSFORMER_BLOCKS:
            layers = _find_transformer_layers(model)

            def before_first_layer(
                module: torch.nn.Module,
                inputs: tuple[Any, ...],
            ) -> None:
                del module, inputs
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                self._vidur_block_starts.append(start)

            def after_last_layer(
                module: torch.nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
            ) -> None:
                del module, inputs, output
                if not self._vidur_block_starts:
                    raise RuntimeError(
                        "last transformer layer ran without the first layer"
                    )
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._vidur_block_events.append(
                    (self._vidur_block_starts.pop(0), end)
                )

            layers[0].register_forward_pre_hook(before_first_layer)
            layers[-1].register_forward_hook(after_last_layer)

        original_forward: Callable[..., Any] = model.forward

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return original_forward(*args, **kwargs)
            finally:
                end.record()
                self._vidur_forward_events.append((start, end))

        model.forward = timed_forward

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        self._vidur_forward_events.clear()
        self._vidur_block_starts.clear()
        self._vidur_block_events.clear()
        output = super().execute_model(scheduler_output)
        scheduled_tokens = dict(scheduler_output.num_scheduled_tokens)
        if scheduled_tokens and not self._vidur_forward_events:
            raise RuntimeError(
                "the vLLM model executed a non-empty batch without invoking "
                "the timed model forward"
            )
        if not self._vidur_forward_events:
            return output

        if self._vidur_timing_scope == TIMING_SCOPE_TRANSFORMER_BLOCKS:
            if not self._vidur_block_events or self._vidur_block_starts:
                raise RuntimeError(
                    "the scheduled batch did not cross a complete transformer stack"
                )
            selected_events = self._vidur_block_events
        else:
            selected_events = self._vidur_forward_events
        selected_events[-1][1].synchronize()
        elapsed_s = sum(
            start.elapsed_time(end) for start, end in selected_events
        ) / 1000.0
        if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
            raise RuntimeError(f"invalid CUDA model-forward duration {elapsed_s!r}")
        is_warmup = bool(scheduled_tokens) and all(
            str(request_id).startswith(self._vidur_warmup_prefix)
            for request_id in scheduled_tokens
        )
        if scheduled_tokens and not is_warmup:
            append_gpu_forward_timing(
                self._vidur_gpu_timing_path,
                scheduled_tokens=scheduled_tokens,
                gpu_forward_s=elapsed_s,
            )
        return output
