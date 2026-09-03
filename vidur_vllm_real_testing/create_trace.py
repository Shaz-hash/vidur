"""Create the strict raw-vLLM trace contract from existing three-column traces."""

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


LEGACY_COLUMNS = ("arrived_at", "num_prefill_tokens", "num_decode_tokens")


def create_from_legacy(
    *,
    input_path: str | Path,
    output_path: str | Path,
    prefill_profile_path: str | Path = DEFAULT_PREFILL_PROFILE,
    request_id_prefix: str = "req",
) -> int:
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    profile = PrefillProfile.load(prefill_profile_path)
    config = CanonicalizationConfig()
    output_rows: list[dict[str, object]] = []

    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(LEGACY_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"legacy trace is missing columns: {missing}")
        for index, row in enumerate(reader):
            arrived_at = float(row["arrived_at"])
            prefill = int(row["num_prefill_tokens"])
            decode = int(row["num_decode_tokens"])
            canonical_prefill = canonicalize_prefill_tokens(prefill, config)
            canonicalize_decode_tokens(decode, config)
            _, prefill_slo = canonicalize_prefill_slo(canonical_prefill, profile, config)
            output_rows.append(
                {
                    "request_id": f"{request_id_prefix}-{index:06d}",
                    "arrived_at_s": arrived_at,
                    "num_prefill_tokens": prefill,
                    "num_decode_tokens": decode,
                    "prefill_slo_s": prefill_slo,
                    "decode_slo_s": CANONICAL_DECODE_SLO_S,
                    "prompt_mode": "synthetic_token_ids",
                    "prompt_ref": "",
                    "seed": index,
                    "ignore_eos": "true",
                }
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a legacy Vidur trace to the vLLM raw schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefill-profile", default=str(DEFAULT_PREFILL_PROFILE))
    parser.add_argument("--request-id-prefix", default="req")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = create_from_legacy(
        input_path=args.input,
        output_path=args.output,
        prefill_profile_path=args.prefill_profile,
        request_id_prefix=args.request_id_prefix,
    )
    print(f"created {count} raw trace rows")


if __name__ == "__main__":
    main()
