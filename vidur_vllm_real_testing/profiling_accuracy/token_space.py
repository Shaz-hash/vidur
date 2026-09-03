from __future__ import annotations


def mlp_token_space(max_tokens: int) -> list[int]:
    values = (
        [1, 2, 4]
        + list(range(8, 1024, 8))
        + list(range(1024, 2 * 1024 + 1, 16))
        + list(range(2 * 1024, 4 * 1024 + 1, 32))
        + list(range(4 * 1024, 8 * 1024 + 1, 64))
    )
    return sorted({value for value in values if value <= max_tokens}, reverse=True)


def prefill_chunk_space(max_chunk_size: int) -> list[int]:
    values = (
        list(range(32, 128 + 1, 32))
        + list(range(256, 1024 + 1, 128))
        + list(range(1024, 4 * 1024 + 1, 512))
    )
    return sorted({value for value in values if value <= max_chunk_size})


def kv_cache_space(max_tokens: int) -> list[int]:
    values = (
        list(range(0, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, max_tokens + 1, 256))
    )
    return sorted({value for value in values if value <= max_tokens})


def decode_batch_space(max_batch_size: int) -> list[int]:
    values = list(range(1, 128 + 1)) + list(range(128, 1024 + 1, 8))
    return sorted({value for value in values if value <= max_batch_size})
