"""vLLM worker that times exactly the transformer-layer stack with CUDA events."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from vllm.v1.worker.gpu_worker import Worker


TIMING_LOG_ENV = "VIDUR_PROFILE_VLLM_TIMING_LOG"


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


def _append_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


class TransformerBlockTimedWorker(Worker):
    """Measure the same repeated decoder-block boundary modeled by Vidur."""

    def init_device(self) -> None:
        super().init_device()
        value = os.environ.get(TIMING_LOG_ENV, "").strip()
        if not value:
            raise RuntimeError(f"{TIMING_LOG_ENV} is required")
        self._vidur_timing_log = Path(value)
        self._vidur_block_starts: list[torch.cuda.Event] = []
        self._vidur_block_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._vidur_full_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.model_runner.model
        layers = _find_transformer_layers(model)

        def before_first_layer(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            del module, inputs
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._vidur_block_starts.append(start)

        def after_last_layer(
            module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> None:
            del module, inputs, output
            if not self._vidur_block_starts:
                raise RuntimeError("last transformer layer ran without the first layer")
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._vidur_block_pairs.append((self._vidur_block_starts.pop(0), end))

        layers[0].register_forward_pre_hook(before_first_layer)
        layers[-1].register_forward_hook(after_last_layer)

        original_forward = model.forward

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return original_forward(*args, **kwargs)
            finally:
                end.record()
                self._vidur_full_pairs.append((start, end))

        model.forward = timed_forward

    @torch.inference_mode()
    def profile_equal_prefill_batches(
        self,
        cases: list[tuple[int, int]],
        warmups: int,
        repetitions: int,
    ) -> list[dict[str, object]]:
        """Run exact synthetic prefill shapes through vLLM's model runner.

        The offline scheduler is intentionally bypassed here: it may split two
        waiting prefills across engine steps. ``_dummy_run`` is vLLM's own
        model-runner profiling path and lets this benchmark force the physical
        GPU batch shape while retaining vLLM's kernels, attention backend, and
        KV-cache preparation.
        """
        if warmups < 0 or repetitions <= 0:
            raise ValueError("warmups must be non-negative and repetitions positive")
        model_runner = self.model_runner
        original_max_num_reqs = model_runner.max_num_reqs
        records: list[dict[str, object]] = []
        try:
            for case_index, (request_count, tokens) in enumerate(cases):
                if request_count <= 0 or tokens <= 0:
                    raise ValueError(f"invalid prefill shape {(request_count, tokens)}")
                if request_count > original_max_num_reqs:
                    raise ValueError(
                        f"request count {request_count} exceeds vLLM cap "
                        f"{original_max_num_reqs}"
                    )
                total_tokens = request_count * tokens
                if total_tokens > model_runner.max_num_tokens:
                    raise ValueError(
                        f"batch has {total_tokens} tokens but vLLM cap is "
                        f"{model_runner.max_num_tokens}"
                    )

                # vLLM divides dummy-run tokens evenly over max_num_reqs.
                # Setting this temporary limit makes the generated physical
                # shape exactly request_count x tokens.
                model_runner.max_num_reqs = request_count
                for run_index in range(warmups + repetitions):
                    self._vidur_block_starts.clear()
                    self._vidur_block_pairs.clear()
                    self._vidur_full_pairs.clear()
                    model_runner._dummy_run(total_tokens, skip_eplb=True)
                    if not self._vidur_block_pairs or not self._vidur_full_pairs:
                        raise RuntimeError("dummy prefill missed a timing boundary")
                    if self._vidur_block_starts:
                        raise RuntimeError("dummy prefill left an unmatched timing event")
                    self._vidur_full_pairs[-1][1].synchronize()
                    block_ms = sum(
                        start.elapsed_time(end)
                        for start, end in self._vidur_block_pairs
                    )
                    full_ms = sum(
                        start.elapsed_time(end)
                        for start, end in self._vidur_full_pairs
                    )
                    if not all(
                        math.isfinite(value) and value > 0
                        for value in (block_ms, full_ms)
                    ):
                        raise RuntimeError(
                            f"invalid CUDA timings block={block_ms} full={full_ms}"
                        )
                    scheduled = {
                        f"profile_{case_index}_{run_index}_{request_index}": tokens
                        for request_index in range(request_count)
                    }
                    records.append(
                        {
                            "scheduled_tokens": scheduled,
                            "request_count": request_count,
                            "total_scheduled_tokens": total_tokens,
                            "transformer_blocks_ms": block_ms,
                            "full_model_forward_ms": full_ms,
                            "forward_calls": len(self._vidur_full_pairs),
                            "block_stack_calls": len(self._vidur_block_pairs),
                            "pid": os.getpid(),
                            "source": "vllm_model_runner_dummy_prefill",
                        }
                    )
        finally:
            model_runner.max_num_reqs = original_max_num_reqs
        return records

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        self._vidur_block_starts.clear()
        self._vidur_block_pairs.clear()
        self._vidur_full_pairs.clear()
        output = super().execute_model(scheduler_output)
        scheduled = {
            str(request_id): int(tokens)
            for request_id, tokens in scheduler_output.num_scheduled_tokens.items()
        }
        if not scheduled:
            return output
        if not self._vidur_block_pairs or not self._vidur_full_pairs:
            raise RuntimeError("a scheduled batch did not cross both timing boundaries")
        if self._vidur_block_starts:
            raise RuntimeError("an unmatched transformer-layer timing event remains")

        self._vidur_full_pairs[-1][1].synchronize()
        block_ms = sum(start.elapsed_time(end) for start, end in self._vidur_block_pairs)
        full_ms = sum(start.elapsed_time(end) for start, end in self._vidur_full_pairs)
        if not all(math.isfinite(value) and value > 0 for value in (block_ms, full_ms)):
            raise RuntimeError(f"invalid CUDA timings block={block_ms} full={full_ms}")
        _append_json(
            self._vidur_timing_log,
            {
                "scheduled_tokens": scheduled,
                "request_count": len(scheduled),
                "total_scheduled_tokens": sum(scheduled.values()),
                "transformer_blocks_ms": block_ms,
                "full_model_forward_ms": full_ms,
                "forward_calls": len(self._vidur_full_pairs),
                "block_stack_calls": len(self._vidur_block_pairs),
                "pid": os.getpid(),
            },
        )
        return output
