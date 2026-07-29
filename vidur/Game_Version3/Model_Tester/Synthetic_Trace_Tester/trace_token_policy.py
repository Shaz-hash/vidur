from __future__ import annotations

from collections.abc import Sequence

from .trace_types import EffectiveTraceRequest, TraceRequest


def bucket_prefill_to_allowed(value: int, allowed: Sequence[int]) -> int:
    allowed_i = sorted(int(x) for x in allowed)
    if not allowed_i:
        raise ValueError("allowed prefill token list is empty")
    v = int(value)
    return int(min(allowed_i, key=lambda x: (abs(int(x) - v), int(x))))


def effective_trace_request(
    req: TraceRequest,
    *,
    policy: str,
    allowed_prefill_tokens: Sequence[int],
    max_prefill_tokens: int,
    min_decode_tokens: int,
    max_decode_tokens: int,
) -> EffectiveTraceRequest:
    p = str(policy or "clip").strip().lower()
    raw_prefill = int(req.num_prefill_tokens)
    raw_decode = int(req.num_decode_tokens)

    if p == "clip":
        eff_prefill = max(1, min(raw_prefill, int(max_prefill_tokens)))
        eff_decode = max(int(min_decode_tokens), min(raw_decode, int(max_decode_tokens)))
    elif p == "bucket_gv3":
        clipped_prefill = max(1, min(raw_prefill, int(max_prefill_tokens)))
        eff_prefill = bucket_prefill_to_allowed(clipped_prefill, allowed_prefill_tokens)
        eff_decode = max(int(min_decode_tokens), min(raw_decode, int(max_decode_tokens)))
    else:
        raise ValueError(f"Unsupported synthetic trace token policy: {policy!r}")

    return EffectiveTraceRequest(
        trace_row_id=int(req.trace_row_id),
        arrived_at=float(req.arrived_at),
        original_prefill_tokens=int(raw_prefill),
        original_decode_tokens=int(raw_decode),
        effective_prefill_tokens=int(eff_prefill),
        effective_decode_tokens=int(eff_decode),
    )
