from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .trace_types import TraceRequest


REQUIRED_COLUMNS = ("arrived_at", "num_prefill_tokens", "num_decode_tokens")


def load_trace_csv(path: str | Path, *, limit: int | None = None) -> list[TraceRequest]:
    trace_path = Path(path).expanduser()
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    out: list[TraceRequest] = []
    with trace_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"{trace_path} missing required columns: {missing}")

        for row_id, row in enumerate(reader):
            arrived_at = float(row["arrived_at"])
            prefill = int(float(row["num_prefill_tokens"]))
            decode = int(float(row["num_decode_tokens"]))
            if arrived_at < 0.0:
                continue
            if prefill <= 0 or decode <= 0:
                continue
            out.append(
                TraceRequest(
                    trace_row_id=int(row_id),
                    arrived_at=float(arrived_at),
                    num_prefill_tokens=int(prefill),
                    num_decode_tokens=int(decode),
                )
            )
            if limit is not None and len(out) >= int(limit):
                break

    out.sort(key=lambda r: (float(r.arrived_at), int(r.trace_row_id)))
    return out


def rebase_trace(trace: Iterable[TraceRequest], *, start_time: float | None = None) -> list[TraceRequest]:
    rows = list(trace)
    if not rows:
        return []
    base = float(rows[0].arrived_at if start_time is None else start_time)
    return [
        TraceRequest(
            trace_row_id=int(r.trace_row_id),
            arrived_at=max(0.0, float(r.arrived_at) - base),
            num_prefill_tokens=int(r.num_prefill_tokens),
            num_decode_tokens=int(r.num_decode_tokens),
        )
        for r in rows
    ]
