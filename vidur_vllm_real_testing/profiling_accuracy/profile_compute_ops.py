from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as functional
from vllm import _custom_ops as custom_ops

from .config import load_config
from .stats import timing_stats
from .token_space import mlp_token_space


HIDDEN_SIZE = 4096
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
MLP_HIDDEN_SIZE = 14336
VOCAB_SIZE = 128256
NUM_LAYERS = 32


def _measure_cuda(
    operation: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmups: int,
    repetitions: int,
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
    return timing_stats(start.elapsed_time(end) for start, end in events)


def _add_stats(row: dict[str, object], name: str, stats: dict[str, float]) -> None:
    for key, value in stats.items():
        row[f"time_stats.{name}.{key}"] = value


def _profile_one_token_count(
    num_tokens: int,
    *,
    warmups: int,
    repetitions: int,
    weights: dict[str, torch.Tensor],
) -> dict[str, object]:
    device = torch.device("cuda")
    dtype = torch.float16
    hidden = torch.empty((num_tokens, HIDDEN_SIZE), device=device, dtype=dtype)
    residual = torch.empty_like(hidden)
    norm_output = torch.empty_like(hidden)
    positions = torch.arange(num_tokens, device=device, dtype=torch.long)
    query = torch.empty(
        (num_tokens, NUM_Q_HEADS * HEAD_SIZE), device=device, dtype=dtype
    )
    key = torch.empty(
        (num_tokens, NUM_KV_HEADS * HEAD_SIZE), device=device, dtype=dtype
    )

    row: dict[str, object] = {
        "n_head": NUM_Q_HEADS,
        "n_kv_head": NUM_KV_HEADS,
        "n_embd": HIDDEN_SIZE,
        "n_expanded_embd": MLP_HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "use_gated_mlp": True,
        "num_tokens": num_tokens,
        "num_tensor_parallel_workers": 1,
        "profiler_implementation": "vllm-0.26.0-primitives",
    }

    operations: list[tuple[str, Callable[[], object]]] = [
        (
            "input_layernorm",
            lambda: custom_ops.rms_norm(
                norm_output, hidden, weights["norm"], 1e-5
            ),
        ),
        (
            "attn_pre_proj",
            lambda: functional.linear(hidden, weights["qkv"]),
        ),
        (
            "attn_rope",
            lambda: custom_ops.rotary_embedding(
                positions,
                query,
                key,
                HEAD_SIZE,
                weights["rope_cache"],
                True,
            ),
        ),
        (
            "attn_post_proj",
            lambda: functional.linear(hidden, weights["o_proj"]),
        ),
        (
            "post_attention_layernorm",
            lambda: custom_ops.rms_norm(
                norm_output, hidden, weights["norm"], 1e-5
            ),
        ),
        (
            "mlp_up_proj",
            lambda: functional.linear(hidden, weights["gate_up"]),
        ),
    ]
    for name, operation in operations:
        _add_stats(
            row,
            name,
            _measure_cuda(operation, warmups=warmups, repetitions=repetitions),
        )

    gate_up = torch.empty(
        (num_tokens, 2 * MLP_HIDDEN_SIZE), device=device, dtype=dtype
    )
    activation_output = torch.empty(
        (num_tokens, MLP_HIDDEN_SIZE), device=device, dtype=dtype
    )
    _add_stats(
        row,
        "mlp_act",
        _measure_cuda(
            lambda: torch.ops._C.silu_and_mul(activation_output, gate_up),
            warmups=warmups,
            repetitions=repetitions,
        ),
    )
    mlp_hidden = torch.empty(
        (num_tokens, MLP_HIDDEN_SIZE), device=device, dtype=dtype
    )
    _add_stats(
        row,
        "mlp_down_proj",
        _measure_cuda(
            lambda: functional.linear(mlp_hidden, weights["down"]),
            warmups=warmups,
            repetitions=repetitions,
        ),
    )
    _add_stats(
        row,
        "add",
        _measure_cuda(
            lambda: hidden + residual,
            warmups=warmups,
            repetitions=repetitions,
        ),
    )
    return row


def main(argv: Iterable[str] | None = None) -> None:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Profile Vidur's operation decomposition with latest vLLM primitives"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=config.max_tokens_per_request)
    parser.add_argument("--warmups", type=int, default=config.operation_warmups)
    parser.add_argument(
        "--repetitions", type=int, default=config.operation_repetitions
    )
    parser.add_argument(
        "--only-tokens",
        type=str,
        help="Optional comma-separated token counts for a smoke run",
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    device = torch.device("cuda")
    dtype = torch.float16
    inverse_frequency = 1.0 / (
        500000.0
        ** (torch.arange(0, HEAD_SIZE, 2, dtype=torch.float32) / HEAD_SIZE)
    )
    frequencies = torch.outer(
        torch.arange(config.max_tokens_per_request, dtype=torch.float32),
        inverse_frequency,
    )
    rope_cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(
        device=device, dtype=dtype
    )
    weights = {
        "qkv": torch.empty(
            ((NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_SIZE, HIDDEN_SIZE),
            device=device,
            dtype=dtype,
        ),
        "o_proj": torch.empty(
            (HIDDEN_SIZE, HIDDEN_SIZE), device=device, dtype=dtype
        ),
        "gate_up": torch.empty(
            (2 * MLP_HIDDEN_SIZE, HIDDEN_SIZE), device=device, dtype=dtype
        ),
        "down": torch.empty(
            (HIDDEN_SIZE, MLP_HIDDEN_SIZE), device=device, dtype=dtype
        ),
        "norm": torch.ones((HIDDEN_SIZE,), device=device, dtype=dtype),
        "rope_cache": rope_cache,
    }
    if args.only_tokens:
        token_counts = sorted(
            {int(value) for value in args.only_tokens.split(",") if value.strip()},
            reverse=True,
        )
    else:
        token_counts = mlp_token_space(args.max_tokens)

    rows = [
        _profile_one_token_count(
            token_count,
            warmups=args.warmups,
            repetitions=args.repetitions,
            weights=weights,
        )
        for token_count in token_counts
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} compute profile rows to {args.output}")


if __name__ == "__main__":
    main()
