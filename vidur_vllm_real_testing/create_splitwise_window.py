"""Create a strict raw trace template from a bounded Splitwise time window."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .canonicalization import (
    CANONICAL_DECODE_SLO_S,
    CanonicalizationConfig,
    PrefillProfile,
    canonicalize_decode_tokens,
    canonicalize_prefill_slo,
    canonicalize_prefill_tokens,
)
from .prepare_trace import DEFAULT_PREFILL_PROFILE
from .trace_contract import RAW_COLUMNS


def create_splitwise_window(
    *,
    input_path: str | Path,
    output_path: str | Path,
    end_time_s: float,
    prefill_profile_path: str | Path = DEFAULT_PREFILL_PROFILE,
) -> int:
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    profile = PrefillProfile.load(prefill_profile_path)
    config = CanonicalizationConfig()
    rows: list[dict[str, object]] = []
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"arrived_at", "num_prefill_tokens", "num_decode_tokens"}
        if not expected.issubset(set(reader.fieldnames or ())):
            raise ValueError("input is not a processed Splitwise trace")
        for source_index, row in enumerate(reader):
            arrived_at = float(row["arrived_at"])
            if arrived_at > end_time_s:
                break
            prefill = int(row["num_prefill_tokens"])
            decode = int(row["num_decode_tokens"])
            canonical_prefill = canonicalize_prefill_tokens(prefill, config)
            canonicalize_decode_tokens(decode, config)
            _, prefill_slo = canonicalize_prefill_slo(canonical_prefill, profile, config)
            rows.append(
                {
                    "request_id": f"splitwise-conv-{source_index:06d}",
                    "arrived_at_s": arrived_at,
                    "num_prefill_tokens": prefill,
                    "num_decode_tokens": decode,
                    "prefill_slo_s": prefill_slo,
                    "decode_slo_s": CANONICAL_DECODE_SLO_S,
                    "prompt_mode": "synthetic_token_ids",
                    "prompt_ref": "",
                    "seed": source_index,
                    "ignore_eos": "true",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a bounded Splitwise raw trace.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--end-time-s", type=float, default=20.0)
    parser.add_argument("--prefill-profile", default=str(DEFAULT_PREFILL_PROFILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = create_splitwise_window(
        input_path=args.input,
        output_path=args.output,
        end_time_s=args.end_time_s,
        prefill_profile_path=args.prefill_profile,
    )
    print(f"created {count} Splitwise raw trace rows")


if __name__ == "__main__":
    main()
