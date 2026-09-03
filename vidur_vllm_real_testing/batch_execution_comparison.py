"""Compare real vLLM batches with the uncalibrated Vidur predictor."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .scheduler_contract import LiveStateSnapshot, RequestPhase


COMPARISON_COLUMNS = (
    "batch_index",
    "sim_time_before",
    "sim_time_after",
    "logical_time_advance_s",
    "total_prefill_size",
    "decode_request_count",
    "prefill_request_count",
    "total_scheduled_tokens",
    "prefill_request_stats",
    "decode_request_stats",
    "recorded_batch_time_from_trace",
    "recorded_batch_time_from_vidur",
    "vidur_minus_trace_seconds",
    "vidur_minus_trace_percent",
    "absolute_percent_error",
    "within_10_percent",
    "timing_source",
    "predictor_cache_dir",
    "calibration_applied",
)


def build_batch_shape(
    snapshot: LiveStateSnapshot,
    scheduled_tokens: Mapping[str, int],
) -> dict[str, Any]:
    """Describe the exact vLLM batch at the pre-execution boundary."""

    by_id = snapshot.by_id()
    prefill_stats: list[dict[str, Any]] = []
    decode_stats: list[dict[str, Any]] = []
    normalized = {
        str(request_id): int(tokens)
        for request_id, tokens in scheduled_tokens.items()
    }
    for request_id, tokens in sorted(normalized.items()):
        if tokens <= 0:
            raise ValueError(f"request {request_id}: scheduled tokens must be positive")
        try:
            request = by_id[request_id]
        except KeyError as exc:
            raise ValueError(
                f"request {request_id}: scheduled request is absent from snapshot"
            ) from exc
        if request.phase is RequestPhase.PREFILL:
            if tokens > int(request.actual_prefill_remaining):
                raise ValueError(
                    f"request {request_id}: scheduled {tokens} prefill tokens but only "
                    f"{request.actual_prefill_remaining} remain"
                )
            prefill_stats.append(
                {
                    "request_id": request_id,
                    "total_prefill_size": int(request.actual_prefill_tokens),
                    "completed_prefill_size_before": int(
                        request.actual_prefill_tokens
                        - request.actual_prefill_remaining
                    ),
                    "remaining_prefill_size_before": int(
                        request.actual_prefill_remaining
                    ),
                    "context_tokens_before": int(request.num_computed_tokens),
                    "scheduled_prefill_tokens": tokens,
                }
            )
        elif request.phase is RequestPhase.DECODE:
            if tokens > int(request.actual_decode_remaining):
                raise ValueError(
                    f"request {request_id}: scheduled {tokens} decode tokens but only "
                    f"{request.actual_decode_remaining} remain"
                )
            decode_stats.append(
                {
                    "request_id": request_id,
                    "context_tokens_before": int(request.num_computed_tokens),
                    "total_decode_tokens": int(request.actual_decode_tokens),
                    "completed_decode_tokens_before": int(
                        request.num_output_tokens
                    ),
                    "remaining_decode_tokens_before": int(
                        request.actual_decode_remaining
                    ),
                    "scheduled_decode_tokens": tokens,
                }
            )
        else:
            raise ValueError(f"request {request_id}: unsupported phase {request.phase}")

    total_prefill = sum(
        int(row["scheduled_prefill_tokens"]) for row in prefill_stats
    )
    total_decode = sum(
        int(row["scheduled_decode_tokens"]) for row in decode_stats
    )
    if not total_prefill and not total_decode:
        raise ValueError("batch has no scheduled work")
    return {
        "total_prefill_size": total_prefill,
        "decode_request_count": len(decode_stats),
        "prefill_request_count": len(prefill_stats),
        "total_scheduled_tokens": total_prefill + total_decode,
        "prefill_request_stats": prefill_stats,
        "decode_request_stats": decode_stats,
    }


def predictor_requests(
    batch_shape: Mapping[str, Any],
) -> list[Any]:
    from vidur.entities.execution_time_predictor_request import (
        ExecutionTimePredictorRequest,
    )

    requests: list[Any] = []
    for row in batch_shape["prefill_request_stats"]:
        requests.append(
            ExecutionTimePredictorRequest(
                num_processed_tokens=int(row["context_tokens_before"]),
                num_tokens_to_process=int(row["scheduled_prefill_tokens"]),
                is_prefill_complete=False,
            )
        )
    for row in batch_shape["decode_request_stats"]:
        requests.append(
            ExecutionTimePredictorRequest(
                num_processed_tokens=int(row["context_tokens_before"]),
                num_tokens_to_process=int(row["scheduled_decode_tokens"]),
                is_prefill_complete=True,
            )
        )
    return requests


def predict_batch_seconds(predictor: object, batch_shape: Mapping[str, Any]) -> float:
    requests = predictor_requests(batch_shape)
    execution = predictor.get_execution_time(requests, 0)
    return float(execution.model_time_ms) / 1000.0


def comparison_rows(
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    predictor: object,
    predictor_cache_dir: str | Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cache_path = str(Path(predictor_cache_dir).expanduser().resolve())
    for batch_index, audit in enumerate(audit_rows):
        batch_shape = dict(audit["batch_shape"])
        timing = dict(audit["timing"])
        state_before = dict(audit["state_before"])
        state_after = dict(audit["state_after"])
        actual_s = float(timing["authoritative_duration_s"])
        if actual_s <= 0.0:
            raise ValueError(f"batch {batch_index}: non-positive measured duration")
        predicted_s = predict_batch_seconds(predictor, batch_shape)
        sim_before = float(state_before["sim_time"])
        sim_after = float(state_after["sim_time"])
        logical_advance = sim_after - sim_before
        if abs(logical_advance - actual_s) > 5e-10:
            raise AssertionError(
                f"batch {batch_index}: logical clock advanced {logical_advance:.12g}s, "
                f"not measured duration {actual_s:.12g}s"
            )
        difference_s = predicted_s - actual_s
        signed_percent = 100.0 * difference_s / actual_s
        absolute_percent = abs(signed_percent)
        result.append(
            {
                "batch_index": batch_index,
                "sim_time_before": sim_before,
                "sim_time_after": sim_after,
                "logical_time_advance_s": logical_advance,
                "total_prefill_size": int(batch_shape["total_prefill_size"]),
                "decode_request_count": int(batch_shape["decode_request_count"]),
                "prefill_request_count": int(batch_shape["prefill_request_count"]),
                "total_scheduled_tokens": int(batch_shape["total_scheduled_tokens"]),
                "prefill_request_stats": json.dumps(
                    batch_shape["prefill_request_stats"], sort_keys=True
                ),
                "decode_request_stats": json.dumps(
                    batch_shape["decode_request_stats"], sort_keys=True
                ),
                "recorded_batch_time_from_trace": actual_s,
                "recorded_batch_time_from_vidur": predicted_s,
                "vidur_minus_trace_seconds": difference_s,
                "vidur_minus_trace_percent": signed_percent,
                "absolute_percent_error": absolute_percent,
                "within_10_percent": absolute_percent <= 10.0 + 1e-9,
                "timing_source": str(timing["source"]),
                "predictor_cache_dir": cache_path,
                "calibration_applied": False,
            }
        )
    return result


def read_batch_audit(path: str | Path) -> list[dict[str, Any]]:
    audit_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with audit_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                "batch_shape",
                "state_before",
                "state_after",
                "timing",
            } - set(row)
            if missing:
                raise ValueError(
                    f"{audit_path}:{line_number}: missing Task 1.1.3 fields {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"batch audit is empty: {audit_path}")
    return rows


def write_comparison_csv(rows: list[Mapping[str, Any]], output: str | Path) -> Path:
    if not rows:
        raise ValueError("cannot write an empty comparison")
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path
