#!/usr/bin/env python3
"""Run the prefill profiler against the pinned vLLM and trace contracts."""

from __future__ import annotations

import csv
from pathlib import Path

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

from vidur_vllm_real_testing import canonicalization
from vidur_vllm_real_testing.mew1 import measure_prefill_profile


def _write_contract_trace(
    path: Path,
    jobs: list[tuple[int, int, bool]],
    profile: dict[int, float],
) -> None:
    columns = (
        "schema_version",
        "request_id",
        "arrived_at_s",
        "actual_prefill_tokens",
        "canonical_prefill_tokens",
        "actual_decode_tokens",
        "canonical_decode_tokens",
        "actual_prefill_slo_s",
        "canonical_prefill_slo_s",
        "actual_decode_slo_s",
        "canonical_decode_slo_s",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for request_index, (_, tokens, _) in enumerate(jobs):
            writer.writerow(
                {
                    "schema_version": canonicalization.TRACE_SCHEMA_VERSION,
                    "request_id": str(request_index),
                    "arrived_at_s": 0.0,
                    "actual_prefill_tokens": tokens,
                    "canonical_prefill_tokens": tokens,
                    "actual_decode_tokens": 1,
                    "canonical_decode_tokens": 1,
                    "actual_prefill_slo_s": 3.0 * profile[tokens],
                    "canonical_prefill_slo_s": 3.0 * profile[tokens],
                    "actual_decode_slo_s": 0.05,
                    "canonical_decode_slo_s": 0.05,
                }
            )


measure_prefill_profile._write_trace = _write_contract_trace


if __name__ == "__main__":
    measure_prefill_profile.main()
