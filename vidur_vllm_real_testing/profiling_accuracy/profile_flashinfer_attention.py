from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
from math import ceil
from pathlib import Path
from typing import Callable, Iterable

import flashinfer
import torch

from .config import load_config
from .stats import timing_stats
from .token_space import decode_batch_space, kv_cache_space, prefill_chunk_space


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128


@dataclass(frozen=True)
class AttentionCase:
    prefill_chunk_size: int
    kv_cache_size: int
    batch_size: int
    is_prefill: bool


def _measure_cuda(
    operation: Callable[[], object], *, warmups: int, repetitions: int
) -> dict[str, float]:
    output: object = None
    for _ in range(warmups):
        output = operation()
    torch.cuda.synchronize()
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    if output is None:
        # append_paged_kv_cache returns None by design.
        pass
    return timing_stats(start.elapsed_time(end) for start, end in events)


def _add_stats(row: dict[str, object], name: str, stats: dict[str, float]) -> None:
    for key, value in stats.items():
        row[f"time_stats.{name}.{key}"] = value


def _zero_stats(row: dict[str, object], name: str) -> None:
    _add_stats(row, name, {key: 0.0 for key in ("min", "max", "mean", "median", "std")})


def _metadata(
    case: AttentionCase, *, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_tokens = case.prefill_chunk_size if case.is_prefill else 1
    total_length = case.kv_cache_size + query_tokens
    pages_per_sequence = ceil(total_length / block_size)
    qo_indptr = torch.arange(
        0,
        (case.batch_size + 1) * query_tokens,
        query_tokens,
        device=device,
        dtype=torch.int32,
    )
    kv_indptr = torch.arange(
        0,
        (case.batch_size + 1) * pages_per_sequence,
        pages_per_sequence,
        device=device,
        dtype=torch.int32,
    )
    kv_indices = torch.arange(
        case.batch_size * pages_per_sequence, device=device, dtype=torch.int32
    )
    kv_last_page_len = torch.full(
        (case.batch_size,),
        total_length % block_size or block_size,
        device=device,
        dtype=torch.int32,
    )
    batch_indices = torch.arange(
        case.batch_size, device=device, dtype=torch.int32
    ).repeat_interleave(query_tokens)
    positions = torch.cat(
        [
            torch.arange(
                case.kv_cache_size,
                total_length,
                device=device,
                dtype=torch.int32,
            )
            for _ in range(case.batch_size)
        ]
    )
    return (
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        batch_indices,
        positions,
    )


def _profile_case(
    case: AttentionCase,
    *,
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    kv_cache: torch.Tensor,
    block_size: int,
    warmups: int,
    repetitions: int,
) -> dict[str, object]:
    device = kv_cache.device
    dtype = kv_cache.dtype
    query_tokens = case.prefill_chunk_size if case.is_prefill else 1
    total_query_tokens = case.batch_size * query_tokens
    q = torch.empty(
        (total_query_tokens, NUM_Q_HEADS, HEAD_SIZE), device=device, dtype=dtype
    )
    k = torch.empty(
        (total_query_tokens, NUM_KV_HEADS, HEAD_SIZE), device=device, dtype=dtype
    )
    v = torch.empty_like(k)
    (
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        batch_indices,
        positions,
    ) = _metadata(case, block_size=block_size, device=device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        block_size,
        causal=True,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )

    def append() -> None:
        flashinfer.append_paged_kv_cache(
            k,
            v,
            batch_indices,
            positions,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            kv_layout="NHD",
        )

    append_stats = _measure_cuda(
        append, warmups=warmups, repetitions=repetitions
    )
    attention_stats = _measure_cuda(
        lambda: wrapper.run(q, kv_cache),
        warmups=warmups,
        repetitions=repetitions,
    )
    row: dict[str, object] = {
        "n_embd": NUM_Q_HEADS * HEAD_SIZE,
        "n_q_head": NUM_Q_HEADS,
        "n_kv_head": NUM_KV_HEADS,
        "block_size": block_size,
        "num_tensor_parallel_workers": 1,
        "max_model_len": 8192,
        "batch_size": case.batch_size,
        "prefill_chunk_size": case.prefill_chunk_size,
        "kv_cache_size": case.kv_cache_size,
        "is_prefill": case.is_prefill,
        "attention_backend": f"flashinfer-{flashinfer.__version__}",
        "profiler_implementation": "flashinfer-direct-paged-kv",
    }
    _zero_stats(row, "attn_input_reshape")
    _add_stats(row, "attn_kv_cache_save", append_stats)
    if case.is_prefill:
        _add_stats(row, "attn_prefill", attention_stats)
        _zero_stats(row, "attn_decode")
    else:
        _zero_stats(row, "attn_prefill")
        _add_stats(row, "attn_decode", attention_stats)
    _zero_stats(row, "attn_output_reshape")
    return row


def _cases(mode: str, max_tokens: int, max_chunk: int, max_batch: int) -> list[AttentionCase]:
    result: list[AttentionCase] = []
    if mode in {"prefill", "all"}:
        for chunk in prefill_chunk_space(max_chunk):
            for kv_size in range(0, max_tokens - chunk + 1, chunk):
                result.append(AttentionCase(chunk, kv_size, 1, True))
    if mode in {"decode", "all"}:
        for batch_size in decode_batch_space(max_batch):
            for kv_size in kv_cache_space(max_tokens):
                if kv_size > 0:
                    result.append(AttentionCase(0, kv_size, batch_size, False))
    return result


def main(argv: Iterable[str] | None = None) -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Profile latest FlashInfer for Vidur")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefill", "decode", "all"), default="all")
    parser.add_argument("--max-tokens", type=int, default=config.max_tokens_per_request)
    parser.add_argument("--max-chunk", type=int, default=config.max_prefill_chunk_size)
    parser.add_argument("--max-batch", type=int, default=config.max_batch_size)
    parser.add_argument("--warmups", type=int, default=config.operation_warmups)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional smoke-test case limit")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    dtype = torch.float16
    cases = _cases(args.mode, args.max_tokens, args.max_chunk, args.max_batch)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("no attention cases selected")
    max_pages = max(
        case.batch_size
        * ceil(
            (case.kv_cache_size + (case.prefill_chunk_size if case.is_prefill else 1))
            / config.block_size
        )
        for case in cases
    )
    kv_cache = torch.empty(
        (
            max_pages,
            2,
            config.block_size,
            NUM_KV_HEADS,
            HEAD_SIZE,
        ),
        device=device,
        dtype=dtype,
    )
    workspace = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter | None = None
        for index, case in enumerate(cases, start=1):
            row = _profile_case(
                case,
                wrapper=wrapper,
                kv_cache=kv_cache,
                block_size=config.block_size,
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if index % 100 == 0 or index == len(cases):
                print(f"profiled {index}/{len(cases)} attention cases", flush=True)
    print(f"wrote {len(cases)} attention profile rows to {args.output}")


if __name__ == "__main__":
    main()
