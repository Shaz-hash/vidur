"""CPU-only integration smoke against vLLM's real KV scheduler implementation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _TextOnlyRegistry:
    @staticmethod
    def supports_multimodal_inputs(model_config: object) -> bool:
        return False


def _vllm_config() -> Any:
    from vllm.config import CacheConfig, SchedulerConfig

    scheduler = SchedulerConfig(
        max_model_len=8192,
        is_encoder_decoder=False,
        max_num_batched_tokens=4608,
        max_num_seqs=512,
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
        policy="fcfs",
    )
    cache = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        enable_prefix_caching=False,
    )
    cache.num_gpu_blocks = 4096
    return SimpleNamespace(
        scheduler_config=scheduler,
        cache_config=cache,
        lora_config=None,
        kv_events_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        observability_config=SimpleNamespace(kv_cache_metrics=None),
        model_config=SimpleNamespace(is_encoder_decoder=False, max_model_len=8192),
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
    )


def _kv_cache_config() -> Any:
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=4096,
        kv_cache_tensors=[KVCacheTensor(size=4096 * spec.page_size_bytes, shared_by=["layer.0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )


def _trace_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _request(row: dict[str, str]) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    prompt_tokens = int(row["actual_prefill_tokens"])
    return Request(
        request_id=str(row["request_id"]),
        prompt_token_ids=[1] * prompt_tokens,
        sampling_params=SamplingParams(
            max_tokens=int(row["actual_decode_tokens"]) + 1,
            ignore_eos=True,
        ),
        pooling_params=None,
        eos_token_id=None,
        arrival_time=float(row["arrived_at_s"]),
    )


def _scheduler(cls: type[Any]) -> Any:
    return cls(
        vllm_config=_vllm_config(),
        kv_cache_config=_kv_cache_config(),
        structured_output_manager=object(),
        block_size=16,
        mm_registry=_TextOnlyRegistry(),
    )


def _add_mixed_requests(scheduler: Any, rows: list[dict[str, str]]) -> None:
    from vllm.v1.request import RequestStatus

    decode = _request(rows[0])
    blocks = scheduler.kv_cache_manager.allocate_slots(
        decode,
        decode.num_prompt_tokens,
    )
    if blocks is None:
        raise RuntimeError("scheduler smoke could not allocate decode prompt KV")
    decode.num_computed_tokens = decode.num_prompt_tokens
    decode.append_output_token_ids(2)
    decode.status = RequestStatus.RUNNING
    scheduler.running.append(decode)
    scheduler.requests[decode.request_id] = decode
    for row in rows[1:]:
        scheduler.add_request(_request(row))


def _run_once(
    mode: str,
    trace: Path,
    rows: list[dict[str, str]],
    *,
    mixed: bool = False,
    planner: str | None = None,
) -> dict[str, Any]:
    from vllm.v1.core.sched.scheduler import Scheduler

    from .vllm_scheduler import GV3Scheduler

    os.environ["VIDUR_VLLM_SCHEDULER_MODE"] = mode
    os.environ["VIDUR_VLLM_CANONICAL_TRACE"] = str(trace)
    if planner is None:
        os.environ.pop("VIDUR_VLLM_GV3_PLANNER", None)
    else:
        os.environ["VIDUR_VLLM_GV3_PLANNER"] = planner
    cls = Scheduler if mode == "base-stock" else GV3Scheduler
    scheduler = _scheduler(cls)
    if mixed:
        _add_mixed_requests(scheduler, rows)
    else:
        for row in rows:
            scheduler.add_request(_request(row))
    output = scheduler.schedule()
    return {
        "tokens": {str(key): int(value) for key, value in output.num_scheduled_tokens.items()},
        "total": int(output.total_num_scheduled_tokens),
        "running": [str(request.request_id) for request in scheduler.running],
        "waiting": [str(request.request_id) for request in scheduler.waiting],
    }


def _expected_sjf_actual_total(rows: list[dict[str, str]], budget: int) -> int:
    remaining = int(budget)
    actual_total = 0
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["canonical_prefill_tokens"]),
            float(item["arrived_at_s"]),
            str(item["request_id"]),
        ),
    ):
        if remaining <= 0:
            break
        canonical = min(int(row["canonical_prefill_tokens"]), remaining)
        actual_total += min(canonical, int(row["actual_prefill_tokens"]))
        remaining -= canonical
    return actual_total


def run_smoke(trace: str | Path) -> dict[str, object]:
    trace_path = Path(trace).expanduser().resolve()
    rows = _trace_rows(trace_path)[:3]
    if len(rows) < 3:
        raise RuntimeError("scheduler smoke requires at least three trace rows")

    stock_base = _run_once("base-stock", trace_path, rows)
    stock_hook = _run_once("stock", trace_path, rows)
    if stock_base != stock_hook:
        raise AssertionError(
            f"stock scheduler changed through custom hook: base={stock_base}, hook={stock_hook}"
        )

    sjf256 = _run_once("sjf-256", trace_path, rows)
    sjf512 = _run_once("sjf-512", trace_path, rows)
    if sjf256["total"] != _expected_sjf_actual_total(rows, 256):
        raise AssertionError(f"unexpected SJF-256 output: {sjf256}")
    if sjf512["total"] != _expected_sjf_actual_total(rows, 512):
        raise AssertionError(f"unexpected SJF-512 output: {sjf512}")

    mixed_stock_base = _run_once("base-stock", trace_path, rows, mixed=True)
    mixed_stock_hook = _run_once("stock", trace_path, rows, mixed=True)
    if mixed_stock_base != mixed_stock_hook:
        raise AssertionError(
            "stock scheduler changed for a mixed decode/prefill batch: "
            f"base={mixed_stock_base}, hook={mixed_stock_hook}"
        )
    mixed_sjf256 = _run_once("sjf-256", trace_path, rows, mixed=True)
    mixed_sjf512 = _run_once("sjf-512", trace_path, rows, mixed=True)
    expected_mixed_256 = 1 + _expected_sjf_actual_total(rows[1:], 256)
    expected_mixed_512 = 1 + _expected_sjf_actual_total(rows[1:], 512)
    if mixed_sjf256["total"] != expected_mixed_256:
        raise AssertionError(f"unexpected mixed SJF-256 output: {mixed_sjf256}")
    if mixed_sjf512["total"] != expected_mixed_512:
        raise AssertionError(f"unexpected mixed SJF-512 output: {mixed_sjf512}")

    valid_planner = "vidur_vllm_real_testing.scheduler_smoke_planners:valid_sjf256"
    stale_planner = "vidur_vllm_real_testing.scheduler_smoke_planners:stale_sjf256"
    controller = _run_once(
        "controller", trace_path, rows, mixed=True, planner=valid_planner
    )
    if controller != mixed_sjf256:
        raise AssertionError(
            f"controller planner did not match its SJF-256 reference: {controller}"
        )
    shadow = _run_once("shadow", trace_path, rows, mixed=True, planner=valid_planner)
    if shadow != mixed_stock_hook:
        raise AssertionError(f"shadow mode changed stock scheduling: {shadow}")
    active_fallback = _run_once(
        "active-validation", trace_path, rows, mixed=True, planner=stale_planner
    )
    if active_fallback != mixed_stock_hook:
        raise AssertionError(
            f"active-validation did not fall back to stock on stale plan: {active_fallback}"
        )
    report = {
        "status": "passed",
        "stock_equivalence": stock_hook,
        "sjf256": sjf256,
        "sjf512": sjf512,
        "mixed_stock_equivalence": mixed_stock_hook,
        "mixed_sjf256": mixed_sjf256,
        "mixed_sjf512": mixed_sjf512,
        "controller": controller,
        "shadow": shadow,
        "active_validation_stale_fallback": active_fallback,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()
    run_smoke(args.trace)


if __name__ == "__main__":
    main()
