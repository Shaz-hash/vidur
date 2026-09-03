"""Summarize authoritative GV3 cost from a persistent real-GPU batch audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_last_batch(path: str | Path) -> tuple[int, dict[str, Any]]:
    count = 0
    last: dict[str, Any] | None = None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            last = json.loads(line)
            count += 1
    if last is None:
        raise ValueError(f"empty persistent batch audit: {path}")
    return count, last


def summarize(path: str | Path, *, policy: str) -> dict[str, Any]:
    batch_count, last = read_last_batch(path)
    state = dict(last["state_after"])
    stats = dict(state["stats"])
    violations = int(stats.get("slo_violations") or 0)
    lateness = float(stats.get("slo_lateness_sum") or 0.0)
    return {
        "policy": str(policy),
        "batch_count": batch_count,
        "final_sim_time_s": float(state["sim_time"]),
        "requests_generated": int(stats.get("requests_generated") or 0),
        "requests_completed": int(stats.get("requests_completed") or 0),
        "active_request_count": len(stats.get("active_request_ids", ())),
        "stopped_decode_request_count": len(
            stats.get("stopped_decode_request_ids", ())
        ),
        "dropped_request_count": len(stats.get("dropped_request_ids", ())),
        "slo_violations": violations,
        "slo_lateness_s": lateness,
        "total_slo_cost": violations + lateness,
        "timing_source": str(dict(last["timing"])["source"]),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args(argv)
    result = summarize(args.batch_log, policy=args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
