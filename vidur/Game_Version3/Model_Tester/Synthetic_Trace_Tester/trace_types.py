from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceRequest:
    trace_row_id: int
    arrived_at: float
    num_prefill_tokens: int
    num_decode_tokens: int


@dataclass(frozen=True)
class EffectiveTraceRequest:
    trace_row_id: int
    arrived_at: float
    original_prefill_tokens: int
    original_decode_tokens: int
    effective_prefill_tokens: int
    effective_decode_tokens: int


@dataclass(frozen=True)
class RandomTraceHistoryResult:
    trace_csv: Path
    summary_csv: Path
    input_trace_csv: Path
    rows_read: int
    rows_released: int
    rows_launched: int
    rows_pending: int
    turns: int
    final_sim_time: float
    requests_generated: int
    requests_completed: int
    slo_violations: int
    total_lateness: float
    end_reason: str
