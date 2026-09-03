#!/usr/bin/env python3
"""Join real-vLLM scheduler actions with persistent GV3 SLO state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .scheduler_log_to_csv import CSV_FIELDS, _flatten, _json


SLO_FIELDS = (
    "has_batch_state",
    "batch_sim_time_after",
    "requests_generated_cumulative",
    "requests_completed_cumulative",
    "active_request_count_after",
    "dropped_request_count_cumulative",
    "stopped_decode_request_count_cumulative",
    "gv3_slo_violations_increment",
    "gv3_slo_lateness_increment",
    "gv3_total_cost_increment",
    "gv3_slo_violations_cumulative",
    "gv3_slo_lateness_cumulative",
    "gv3_total_cost_cumulative",
    "decode_credit_balance",
    "violated_request_ids",
    "dropped_request_ids",
    "prefill_lateness_by_request",
    "decode_lateness_by_request",
    "completed_request_ids_this_batch",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def convert(scheduler_path: Path, batch_path: Path, output_path: Path) -> int:
    scheduler_rows = _read_jsonl(scheduler_path)
    batch_rows = _read_jsonl(batch_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_index = 0
    previous_violations = 0
    previous_lateness = 0.0
    previous_completed: set[int] = set()

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS + SLO_FIELDS)
        writer.writeheader()
        for scheduler_record in scheduler_rows:
            row = _flatten(scheduler_record)
            total_scheduled = int(
                scheduler_record.get("actual_total_num_scheduled_tokens") or 0
            )
            batch: dict[str, Any] | None = None
            if total_scheduled > 0:
                if batch_index >= len(batch_rows):
                    raise ValueError("scheduler log has more executed batches than batch log")
                batch = batch_rows[batch_index]
                batch_index += 1
                expected = {
                    str(key): int(value)
                    for key, value in dict(
                        scheduler_record.get("actual_num_scheduled_tokens") or {}
                    ).items()
                }
                actual = {
                    str(key): int(value)
                    for key, value in dict(batch.get("scheduled_tokens") or {}).items()
                }
                if actual != expected:
                    raise ValueError(
                        f"batch {batch_index}: scheduler/batch token mismatch: "
                        f"{expected} != {actual}"
                    )

            if batch is None:
                row.update({field: "" for field in SLO_FIELDS})
                row["has_batch_state"] = False
                writer.writerow(row)
                continue

            state = dict(batch.get("state_after") or {})
            stats = dict(state.get("stats") or {})
            violations = int(stats.get("slo_violations") or 0)
            lateness = float(stats.get("slo_lateness_sum") or 0.0)
            total_cost = float(violations) + lateness
            completed = {int(value) for value in stats.get("completed_request_ids", ())}
            row.update(
                {
                    "has_batch_state": True,
                    "batch_sim_time_after": state.get("sim_time"),
                    "requests_generated_cumulative": stats.get("requests_generated"),
                    "requests_completed_cumulative": stats.get("requests_completed"),
                    "active_request_count_after": len(stats.get("active_request_ids", ())),
                    "dropped_request_count_cumulative": len(
                        stats.get("dropped_request_ids", ())
                    ),
                    "stopped_decode_request_count_cumulative": len(
                        stats.get("stopped_decode_request_ids", ())
                    ),
                    "gv3_slo_violations_increment": violations - previous_violations,
                    "gv3_slo_lateness_increment": lateness - previous_lateness,
                    "gv3_total_cost_increment": (
                        violations - previous_violations + lateness - previous_lateness
                    ),
                    "gv3_slo_violations_cumulative": violations,
                    "gv3_slo_lateness_cumulative": lateness,
                    "gv3_total_cost_cumulative": total_cost,
                    "decode_credit_balance": stats.get("decode_credit_balance"),
                    "violated_request_ids": _json(stats.get("violated_request_ids")),
                    "dropped_request_ids": _json(stats.get("dropped_request_ids")),
                    "prefill_lateness_by_request": _json(
                        stats.get("per_request_prefill_lateness_by_id")
                    ),
                    "decode_lateness_by_request": _json(
                        stats.get("per_request_decode_lateness_by_id")
                    ),
                    "completed_request_ids_this_batch": _json(
                        sorted(completed - previous_completed)
                    ),
                }
            )
            writer.writerow(row)
            previous_violations = violations
            previous_lateness = lateness
            previous_completed = completed

    if batch_index != len(batch_rows):
        raise ValueError(
            f"batch log has {len(batch_rows) - batch_index} unmatched executed batches"
        )
    return len(scheduler_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scheduler_log", type=Path)
    parser.add_argument("batch_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = convert(args.scheduler_log, args.batch_log, args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
