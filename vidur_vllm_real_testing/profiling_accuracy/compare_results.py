from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

from .stats import percentile


KEY = ("request_count", "prefill_tokens_per_request")


def _read(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = tuple(int(row[column]) for column in KEY)
        if key in result:
            raise ValueError(f"duplicate key {key} in {path}")
        result[key] = row
    return result


def comparison_rows(
    actual: dict[tuple[int, int], dict[str, str]],
    predicted: dict[tuple[int, int], dict[str, str]],
) -> list[dict[str, object]]:
    if set(actual) != set(predicted):
        raise ValueError(
            f"case mismatch: actual-only={set(actual) - set(predicted)}, "
            f"Vidur-only={set(predicted) - set(actual)}"
        )
    rows: list[dict[str, object]] = []
    for key in sorted(actual):
        actual_ms = float(actual[key]["vllm_transformer_blocks_ms_median"])
        vidur_ms = float(predicted[key]["vidur_model_ms"])
        if actual_ms <= 0 or vidur_ms <= 0:
            raise ValueError(f"non-positive timing for {key}")
        signed_error_ms = vidur_ms - actual_ms
        rows.append(
            {
                "request_count": key[0],
                "prefill_tokens_per_request": key[1],
                "total_prefill_tokens": key[0] * key[1],
                "vllm_transformer_blocks_ms_median": actual_ms,
                "vllm_transformer_blocks_ms_p95": float(
                    actual[key]["vllm_transformer_blocks_ms_p95"]
                ),
                "vllm_full_model_forward_ms_median": float(
                    actual[key]["vllm_full_model_forward_ms_median"]
                ),
                "vidur_model_ms": vidur_ms,
                "vidur_minus_vllm_ms": signed_error_ms,
                "vidur_minus_vllm_percent": signed_error_ms / actual_ms * 100.0,
                "absolute_error_ms": abs(signed_error_ms),
                "absolute_percent_error": abs(signed_error_ms) / actual_ms * 100.0,
                "vidur_over_vllm_ratio": vidur_ms / actual_ms,
                "vidur_attention_lookup_clamped": predicted[key][
                    "vidur_attention_lookup_clamped"
                ].lower()
                == "true",
                "vllm_version": actual[key]["vllm_version"],
                "flashinfer_version": actual[key]["flashinfer_version"],
                "attention_backend": actual[key]["attention_backend"],
                "timing_boundary": actual[key]["timing_boundary"],
                "calibration_applied": False,
            }
        )
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    def summarize(selected: list[dict[str, object]]) -> dict[str, float | int]:
        ape = [float(row["absolute_percent_error"]) for row in selected]
        signed = [float(row["vidur_minus_vllm_percent"]) for row in selected]
        absolute_ms = [float(row["absolute_error_ms"]) for row in selected]
        signed_ms = [float(row["vidur_minus_vllm_ms"]) for row in selected]
        return {
            "cases": len(selected),
            "mean_absolute_error_ms": statistics.fmean(absolute_ms),
            "median_absolute_error_ms": statistics.median(absolute_ms),
            "p95_absolute_error_ms": percentile(absolute_ms, 95),
            "max_absolute_error_ms": max(absolute_ms),
            "mean_signed_error_ms": statistics.fmean(signed_ms),
            "mean_absolute_percent_error": statistics.fmean(ape),
            "median_absolute_percent_error": statistics.median(ape),
            "p95_absolute_percent_error": percentile(ape, 95),
            "max_absolute_percent_error": max(ape),
            "mean_signed_percent_error": statistics.fmean(signed),
        }

    result: dict[str, object] = {"overall": summarize(rows), "by_request_count": {}}
    counts = sorted({int(row["request_count"]) for row in rows})
    for count in counts:
        selected = [row for row in rows if int(row["request_count"]) == count]
        result["by_request_count"][str(count)] = summarize(selected)
    unclamped = [
        row for row in rows if not bool(row["vidur_attention_lookup_clamped"])
    ]
    clamped = [row for row in rows if bool(row["vidur_attention_lookup_clamped"])]
    if unclamped:
        result["unclamped"] = summarize(unclamped)
    if clamped:
        result["clamped"] = summarize(clamped)
    result["calibration_applied"] = False
    result["timing_boundary"] = (
        "vLLM CUDA events from first decoder layer start through last decoder "
        "layer end versus Vidur ExecutionTime.model_time"
    )
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare raw Vidur and vLLM timings")
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--vidur", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = comparison_rows(_read(args.actual), _read(args.vidur))
    if any(not math.isfinite(float(row["absolute_percent_error"])) for row in rows):
        raise ValueError("comparison produced a non-finite error")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(
        json.dumps(_summary(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
