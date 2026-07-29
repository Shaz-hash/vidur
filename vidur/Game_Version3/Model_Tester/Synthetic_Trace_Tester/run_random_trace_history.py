from __future__ import annotations

import argparse
from pathlib import Path

from .random_trace_history_runner import RandomTraceHistoryConfig, run_random_trace_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a GV3-compatible random-action history from a processed trace CSV."
    )
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--time-limit-sec", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--max-trace-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--token-policy", choices=("clip", "bucket_gv3"), default="clip")
    parser.add_argument("--rebase-to-zero", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv).expanduser()
    summary_csv = Path(args.summary_csv).expanduser() if args.summary_csv else output_csv.with_name(output_csv.stem + "_summary.csv")
    cfg = RandomTraceHistoryConfig(
        trace_csv=Path(args.trace_csv).expanduser(),
        output_csv=output_csv,
        summary_csv=summary_csv,
        time_limit_sec=float(args.time_limit_sec),
        max_steps=int(args.max_steps),
        max_trace_rows=int(args.max_trace_rows),
        seed=int(args.seed),
        token_policy=str(args.token_policy),
        rebase_to_zero=bool(args.rebase_to_zero),
    )
    result = run_random_trace_history(cfg)
    print(f"synthetic trace history written: {result.trace_csv}")
    print(f"summary written: {result.summary_csv}")
    print(
        "summary: "
        f"turns={result.turns} final_sim_time={result.final_sim_time:.6f} "
        f"released={result.rows_released} launched={result.rows_launched} "
        f"generated={result.requests_generated} completed={result.requests_completed} "
        f"end_reason={result.end_reason}"
    )


if __name__ == "__main__":
    main()
