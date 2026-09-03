"""Audit static Markov-v2 request features derived from a canonical real trace."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from vidur.AlphaGoZero.markov_value_features import build_markov_value_features

from .canonicalization import ADVERSARY_TICK_S, TRACE_SCHEMA_VERSION
from .trace_contract import CANONICAL_COLUMNS


AUDIT_COLUMNS = (
    "request_id",
    "arrived_at_s",
    "actual_prefill_tokens",
    "canonical_prefill_tokens",
    "actual_decode_tokens",
    "canonical_decode_tokens",
    "actual_prefill_slo_s",
    "canonical_prefill_slo_s",
    "actual_decode_slo_s",
    "canonical_decode_slo_s",
    "markov_prefill_total_div_4096",
    "markov_prefill_remaining_div_4096",
    "markov_decode_total_div_864",
    "markov_decode_remaining_div_864",
    "markov_asinh_prefill_slo_div_1s",
    "markov_asinh_decode_slo_div_0p05s",
    "prompt_mode",
    "prompt_ref",
)


def _next_tick_after(timestamp: float) -> float:
    tick_index = math.floor(timestamp / ADVERSARY_TICK_S + 1e-9) + 1
    return tick_index * ADVERSARY_TICK_S


def _initial_markov_payload(row: dict[str, str], numeric_id: int) -> dict[str, object]:
    arrival = float(row["arrived_at_s"])
    prefill = int(row["canonical_prefill_tokens"])
    decode = int(row["canonical_decode_tokens"])
    prefill_slo = float(row["canonical_prefill_slo_s"])
    decode_slo = float(row["canonical_decode_slo_s"])
    return {
        "sim_time": arrival,
        "requests": [
            {
                "request_id": numeric_id,
                "num_prefill_tokens": prefill,
                "num_processed_prefill_tokens": 0,
                "num_decode_tokens": decode,
                "num_processed_decode_tokens": 0,
                "is_prefill_complete": False,
                "arrived_at": arrival,
                "queued_at": arrival,
                "prefill_slo_time": prefill_slo,
                "decode_slo_time": decode_slo,
                "prefill_deadline": arrival + prefill_slo,
            }
        ],
        "stats": {
            "active_request_ids": [numeric_id],
            "violated_request_ids": [],
            "prefill_lateness_finalized_ids": [],
            "next_adv_tick": _next_tick_after(arrival),
            "decode_credit_balance": 0,
            "decode_credit_available": 0,
            "recent_launches": [
                {"timestamp": arrival, "count": 1, "prefill_tokens": prefill}
            ],
        },
    }


def build_static_feature_audit(
    *,
    canonical_trace_path: str | Path,
    output_path: str | Path,
) -> int:
    source = Path(canonical_trace_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    audit_rows: list[dict[str, object]] = []
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANONICAL_COLUMNS:
            raise ValueError("canonical trace columns do not match gv3_vllm_trace_v1")
        for index, row in enumerate(reader):
            if row["schema_version"] != TRACE_SCHEMA_VERSION:
                raise ValueError("unsupported canonical trace schema")
            features = build_markov_value_features(_initial_markov_payload(row, index))
            request = features.request_features[0]
            audit_rows.append(
                {
                    "request_id": row["request_id"],
                    "arrived_at_s": row["arrived_at_s"],
                    "actual_prefill_tokens": row["actual_prefill_tokens"],
                    "canonical_prefill_tokens": row["canonical_prefill_tokens"],
                    "actual_decode_tokens": row["actual_decode_tokens"],
                    "canonical_decode_tokens": row["canonical_decode_tokens"],
                    "actual_prefill_slo_s": row["actual_prefill_slo_s"],
                    "canonical_prefill_slo_s": row["canonical_prefill_slo_s"],
                    "actual_decode_slo_s": row["actual_decode_slo_s"],
                    "canonical_decode_slo_s": row["canonical_decode_slo_s"],
                    "markov_prefill_total_div_4096": float(request[1]),
                    "markov_prefill_remaining_div_4096": float(request[3]),
                    "markov_decode_total_div_864": float(request[4]),
                    "markov_decode_remaining_div_864": float(request[6]),
                    "markov_asinh_prefill_slo_div_1s": float(request[10]),
                    "markov_asinh_decode_slo_div_0p05s": float(request[12]),
                    "prompt_mode": row["prompt_mode"],
                    "prompt_ref": row["prompt_ref"],
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)
    return len(audit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit static Markov-v2 trace features.")
    parser.add_argument("--canonical-trace", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_static_feature_audit(
        canonical_trace_path=args.canonical_trace,
        output_path=args.output,
    )
    print(f"wrote static Markov-v2 feature audit for {count} requests")


if __name__ == "__main__":
    main()
