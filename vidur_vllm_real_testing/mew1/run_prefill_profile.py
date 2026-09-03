#!/usr/bin/env python3
"""Pinned vLLM-0.13 API entrypoint for the prefill profiler."""

from __future__ import annotations

import vllm


_VLLM_LLM = vllm.LLM


class _NormalizedSchedulerPathLLM:
    def __new__(cls, *args: object, **kwargs: object) -> object:
        scheduler_cls = kwargs.get("scheduler_cls")
        if isinstance(scheduler_cls, str) and ":" in scheduler_cls:
            module, _, name = scheduler_cls.partition(":")
            kwargs["scheduler_cls"] = f"{module}.{name}"
        return _VLLM_LLM(*args, **kwargs)


vllm.LLM = _NormalizedSchedulerPathLLM

from vidur_vllm_real_testing.mew1.measure_prefill_profile import main


if __name__ == "__main__":
    main()

