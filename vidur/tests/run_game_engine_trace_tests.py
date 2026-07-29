#!/usr/bin/env python3
"""Run game-engine trace invariants against one or more CSV trace files.
Command :
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur-classical-search \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/tests/run_game_engine_trace_tests.py \
  /home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/GV3_Agent/rootChildAdvTraceSmoke/root_child_adv_trace_parent_000021_ctrl_0008_adv_0005.csv


"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vidur.tests import game_engine_tests as ge_tests  # noqa: E402


DEFAULT_TRACE = (
    REPO_ROOT
    / "simulator_output/GV3_Agent/rootChildAdvTraceSmoke/"
    / "root_child_adv_trace_parent_000021_ctrl_0008_adv_0005.csv"
)


def _iter_paths(raw_paths: Iterable[str]) -> list[Path]:
    paths = [Path(p).expanduser() for p in raw_paths]
    return paths if paths else [DEFAULT_TRACE]


def _choose_trace_builder(path: Path, rows: list[ge_tests.Row], mode: str) -> tuple[list[list[ge_tests.Row]], bool, bool]:
    name = path.name.lower()
    if mode == "synthetic-trace":
        return ge_tests.build_ordered_root_traces(rows), True, True
    if mode == "ordered" or (mode == "auto" and ("root_child_adv_trace" in name or "extracted_trace" in name)):
        return ge_tests.build_ordered_root_traces(rows), True, False
    if mode == "extracted-style" or (mode == "auto" and "mcts_iter" in name):
        return ge_tests.build_extracted_style_leaf_traces(rows), True, False
    return ge_tests.build_leaf_traces(rows), False, False


def run_file(path: Path, *, mode: str, failed_dir: Path) -> tuple[int, int]:
    rows, fieldnames, raw_by_rownum = ge_tests.load_rows(str(path))
    traces, is_extracted, synthetic_trace_mode = _choose_trace_builder(path, rows, mode)
    prefill_profile: dict[int, float] = {}
    unsupported_hits: set[str] = set()

    adv_actions = 0
    for trace_index, trace in enumerate(traces):
        try:
            adv_actions += ge_tests.run_trace(
                trace,
                prefill_profile=prefill_profile,
                assume_extracted_trace=is_extracted,
                synthetic_trace_mode=synthetic_trace_mode,
                unsupported_hits=unsupported_hits,
            )
        except ge_tests.TestFailure:
            failed_out = failed_dir / f"{path.stem}_failed_trace_{trace_index:04d}.csv"
            ge_tests._write_failed_trace_csv(
                trace,
                fieldnames=fieldnames,
                raw_by_rownum=raw_by_rownum,
                out_path=failed_out,
            )
            print(f"[FAIL-TRACE] wrote failed trace to {failed_out}")
            raise

    if unsupported_hits:
        skipped = ", ".join(sorted(unsupported_hits))
        print(f"[WARN] unsupported checks skipped for {path}: {skipped}")
    return len(traces), adv_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_csv",
        nargs="*",
        help=f"Trace CSV(s) to validate. Defaults to {DEFAULT_TRACE}",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "ordered", "extracted-style", "leaf", "synthetic-trace"),
        default="auto",
        help="How to split CSV rows into test traces.",
    )
    parser.add_argument(
        "--failed-dir",
        default=str(REPO_ROOT / "simulator_output/GV3_Agent/game_engine_test_failures"),
        help="Directory where the failing trace slice is written on failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _iter_paths(args.trace_csv)
    failed_dir = Path(args.failed_dir).expanduser()
    failed_dir.mkdir(parents=True, exist_ok=True)

    total_traces = 0
    total_adv_actions = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        traces, adv_actions = run_file(path, mode=str(args.mode), failed_dir=failed_dir)
        total_traces += traces
        total_adv_actions += adv_actions
        print(f"PASSED {path}: traces={traces} adv_actions_checked={adv_actions}")

    print(f"ALL PASSED: files={len(paths)} traces={total_traces} adv_actions_checked={total_adv_actions}")


if __name__ == "__main__":
    main()
