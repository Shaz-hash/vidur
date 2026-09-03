from __future__ import annotations

from .stats import percentile, timing_stats


def _shape(record: dict[str, object]) -> tuple[int, int]:
    values = list(dict(record["scheduled_tokens"]).values())
    if not values or len(set(values)) != 1:
        raise ValueError(f"expected equal-size prefill requests, got {values}")
    return len(values), int(values[0])


def aggregate_vllm_records(
    records: list[dict[str, object]], *, warmups: int, repetitions: int
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for record in records:
        request_ids = tuple(dict(record["scheduled_tokens"]))
        if request_ids and all(value.startswith("_warmup_") for value in request_ids):
            continue
        grouped.setdefault(_shape(record), []).append(record)
    rows: list[dict[str, object]] = []
    for (request_count, tokens), group in sorted(grouped.items()):
        expected = warmups + repetitions
        if len(group) != expected:
            raise RuntimeError(
                f"shape {(request_count, tokens)} produced {len(group)} batches; "
                f"expected {expected}. This usually means vLLM split a prefill."
            )
        active = group[warmups:]
        block_values = [float(row["transformer_blocks_ms"]) for row in active]
        full_values = [float(row["full_model_forward_ms"]) for row in active]
        block_stats = timing_stats(block_values)
        full_stats = timing_stats(full_values)
        rows.append(
            {
                "request_count": request_count,
                "prefill_tokens_per_request": tokens,
                "total_prefill_tokens": request_count * tokens,
                "active_repetitions": repetitions,
                "vllm_transformer_blocks_ms_min": block_stats["min"],
                "vllm_transformer_blocks_ms_median": block_stats["median"],
                "vllm_transformer_blocks_ms_mean": block_stats["mean"],
                "vllm_transformer_blocks_ms_p95": percentile(block_values, 95),
                "vllm_transformer_blocks_ms_max": block_stats["max"],
                "vllm_full_model_forward_ms_median": full_stats["median"],
                "vllm_full_minus_blocks_ms_median": (
                    full_stats["median"] - block_stats["median"]
                ),
            }
        )
    return rows
