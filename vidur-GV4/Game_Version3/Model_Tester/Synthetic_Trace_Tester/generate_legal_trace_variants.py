#!/usr/bin/env python3
"""Generate deterministic GV3-legal variants of an existing burst trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .trace_loader import load_trace_csv


ALLOWED_PREFILL_TOKENS = (128, 256, 512, 1024, 1536, 2048, 3072, 4096)
DECODE_TOKENS_PER_REQUEST = 864
ADVERSARY_TICK_SEC = 0.2
MAX_REQUESTS_PER_LAUNCH = 7
MAX_PREFILL_TOKENS_PER_LAUNCH = 7168


@dataclass(frozen=True)
class Burst:
    request_count: int
    prefill_tokens: int
    decode_tokens: int

    @property
    def total_prefill_tokens(self) -> int:
        return int(self.request_count * self.prefill_tokens)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--prefix",
        default="gv3_legal_homogeneous_20s_decode864_variant",
    )
    return parser.parse_args()


def _load_bursts(path: Path) -> tuple[list[float], list[Burst]]:
    grouped: dict[float, list[tuple[int, int]]] = defaultdict(list)
    for row in load_trace_csv(path):
        grouped[float(row.arrived_at)].append(
            (int(row.num_prefill_tokens), int(row.num_decode_tokens))
        )

    times = sorted(grouped)
    bursts: list[Burst] = []
    for arrived_at in times:
        rows = grouped[arrived_at]
        prefill_values = {prefill for prefill, _decode in rows}
        decode_values = {decode for _prefill, decode in rows}
        if len(prefill_values) != 1 or len(decode_values) != 1:
            raise ValueError(
                f"source launch at {arrived_at} is not homogeneous: "
                f"prefill={sorted(prefill_values)} decode={sorted(decode_values)}"
            )
        burst = Burst(
            request_count=len(rows),
            prefill_tokens=next(iter(prefill_values)),
            decode_tokens=next(iter(decode_values)),
        )
        _validate_burst(arrived_at, burst)
        bursts.append(burst)

    if len(times) < 2:
        raise ValueError("source trace must contain at least two launch windows")
    for previous, current in zip(times, times[1:]):
        if current - previous <= 1.0 + 1e-9:
            raise ValueError(
                "source launch spacing must exceed the one-second GV3 window: "
                f"{previous} -> {current}"
            )
    return times, bursts


def _validate_burst(arrived_at: float, burst: Burst) -> None:
    nearest_tick = round(arrived_at / ADVERSARY_TICK_SEC) * ADVERSARY_TICK_SEC
    if abs(arrived_at - nearest_tick) > 1e-8:
        raise ValueError(f"launch {arrived_at} is off the adversary tick grid")
    if burst.prefill_tokens not in ALLOWED_PREFILL_TOKENS:
        raise ValueError(f"unsupported prefill size: {burst.prefill_tokens}")
    if burst.decode_tokens != DECODE_TOKENS_PER_REQUEST:
        raise ValueError(f"unsupported decode size: {burst.decode_tokens}")
    if burst.request_count > MAX_REQUESTS_PER_LAUNCH:
        raise ValueError(f"launch count exceeds cap: {burst.request_count}")
    if burst.total_prefill_tokens > MAX_PREFILL_TOKENS_PER_LAUNCH:
        raise ValueError(
            f"launch prefill exceeds cap: {burst.total_prefill_tokens}"
        )


def _write_variant(
    *,
    source_trace: Path,
    output_dir: Path,
    prefix: str,
    variant_index: int,
    variant_seed: int,
    times: list[float],
    bursts: list[Burst],
) -> dict[str, object]:
    csv_path = output_dir / f"{prefix}_{variant_index:02d}.csv"
    summary_path = output_dir / f"{prefix}_{variant_index:02d}_window_summary.csv"
    manifest_path = output_dir / f"{prefix}_{variant_index:02d}_manifest.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "arrived_at",
                "num_prefill_tokens",
                "num_decode_tokens",
            ),
        )
        writer.writeheader()
        for arrived_at, burst in zip(times, bursts):
            _validate_burst(arrived_at, burst)
            for _ in range(burst.request_count):
                writer.writerow(
                    {
                        "arrived_at": arrived_at,
                        "num_prefill_tokens": burst.prefill_tokens,
                        "num_decode_tokens": burst.decode_tokens,
                    }
                )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "window_idx",
                "arrived_at",
                "request_count",
                "total_prefill_tokens",
                "prefill_pattern",
                "decode_tokens_per_request",
            ),
        )
        writer.writeheader()
        for window_idx, (arrived_at, burst) in enumerate(zip(times, bursts)):
            writer.writerow(
                {
                    "window_idx": window_idx,
                    "arrived_at": arrived_at,
                    "request_count": burst.request_count,
                    "total_prefill_tokens": burst.total_prefill_tokens,
                    "prefill_pattern": " ".join(
                        [str(burst.prefill_tokens)] * burst.request_count
                    ),
                    "decode_tokens_per_request": burst.decode_tokens,
                }
            )

    total_requests = sum(burst.request_count for burst in bursts)
    total_prefill = sum(burst.total_prefill_tokens for burst in bursts)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "trace_path": str(csv_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "source_trace": str(source_trace.resolve()),
        "variant_index": variant_index,
        "seed": variant_seed,
        "sha256": digest,
        "duration_sec": 20.0,
        "num_launch_windows": len(times),
        "launch_spacing_sec": times[1] - times[0],
        "total_requests": total_requests,
        "total_prefill_tokens": total_prefill,
        "decode_tokens_per_request": DECODE_TOKENS_PER_REQUEST,
        "allowed_prefill_tokens": list(ALLOWED_PREFILL_TOKENS),
        "max_requests_per_launch_window": MAX_REQUESTS_PER_LAUNCH,
        "max_prefill_tokens_per_launch_window": MAX_PREFILL_TOKENS_PER_LAUNCH,
        "homogeneous_prefill_per_launch": True,
        "adversary_tick_sec": ADVERSARY_TICK_SEC,
        "constraints_satisfied": True,
        "generation": (
            "Deterministic permutation of the source trace's complete burst "
            "multiset; timestamps and aggregate workload are unchanged."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = _parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    source_trace = args.source_trace.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    times, source_bursts = _load_bursts(source_trace)

    seen_orders: set[tuple[Burst, ...]] = {tuple(source_bursts)}
    manifests: list[dict[str, object]] = []
    for variant_index in range(1, args.count + 1):
        variant_seed = int(args.seed) + variant_index
        shuffled = list(source_bursts)
        random.Random(variant_seed).shuffle(shuffled)
        while tuple(shuffled) in seen_orders:
            variant_seed += int(args.count)
            random.Random(variant_seed).shuffle(shuffled)
        seen_orders.add(tuple(shuffled))
        manifests.append(
            _write_variant(
                source_trace=source_trace,
                output_dir=output_dir,
                prefix=str(args.prefix),
                variant_index=variant_index,
                variant_seed=variant_seed,
                times=times,
                bursts=shuffled,
            )
        )

    suite_path = output_dir / f"{args.prefix}_suite_manifest.json"
    suite_path.write_text(
        json.dumps(
            {
                "source_trace": str(source_trace),
                "base_seed": int(args.seed),
                "variant_count": len(manifests),
                "variants": manifests,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(suite_path)


if __name__ == "__main__":
    main()
