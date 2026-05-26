"""Append a selected root action to an existing parent trace CSV and validate it.

This is for debugging existing per-root files such as:
    root_015221_trace.csv
    root_015221_selected_action.csv

The selected-action CSV identifies the best action index. The real child
post-state is loaded from an existing child-transition cache, then converted
into a DNNMCTSIterationLogger-compatible row and appended to the parent trace.
The original parent trace CSV is not modified; a combined CSV is written under
`--output-dir` and passed to the existing GV3 trace validator.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .rootChildGeneration import RootChildGenerationConfig, _make_state_loader
    from .rootChildTraceTesting import _append_child_trace_row, _child_state_from_cached_row, _write_trace_csv
    from ...tests.history_node_tests import _validate_trace_csv
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parents[6]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.analysis_testing.rootChildGeneration import (
        RootChildGenerationConfig,
        _make_state_loader,
    )
    from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.analysis_testing.rootChildTraceTesting import (
        _append_child_trace_row,
        _child_state_from_cached_row,
        _write_trace_csv,
    )
    from vidur.mcts.Game_Versions.Game_Version3.tests.history_node_tests import _validate_trace_csv


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser()
    with p.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_selected_action(path: str | Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one selected-action row in {path}, found {len(rows)}")
    return rows[0]


def _to_int(value: Any, *, name: str) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception as exc:
        raise RuntimeError(f"invalid integer {name}={value!r}") from exc


def find_cached_child_transition(
    child_cache_dir: str | Path,
    *,
    parent_root_id: int,
    action_index: int,
) -> dict[str, Any]:
    cache_dir = Path(child_cache_dir).expanduser()
    manifest = cache_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"child cache manifest not found: {manifest}")

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            rows = torch.load(cache_dir / str(entry["shard_path"]), map_location="cpu", weights_only=False)
            for row in rows:
                if int(row.get("parent_root_id", -1)) != int(parent_root_id):
                    continue
                if int(row.get("action_index", -1)) != int(action_index):
                    continue
                return row

    raise RuntimeError(
        "selected child transition not found in cache: "
        f"parent_root_id={parent_root_id}, action_index={action_index}, cache={cache_dir}"
    )


def append_selected_action_trace(
    *,
    dataset_dir: str | Path,
    child_cache_dir: str | Path,
    parent_trace_csv: str | Path,
    selected_action_csv: str | Path,
    output_dir: str | Path,
    seed: int = 2027,
) -> dict[str, Any]:
    parent_trace_csv = Path(parent_trace_csv).expanduser()
    selected_action_csv = Path(selected_action_csv).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_rows = _read_csv_rows(parent_trace_csv)
    if not parent_rows:
        raise RuntimeError(f"parent trace has no rows: {parent_trace_csv}")

    selected = _read_selected_action(selected_action_csv)
    root_id = _to_int(selected.get("root_id"), name="root_id")
    action_index = _to_int(selected.get("best_action_index"), name="best_action_index")

    child_row = find_cached_child_transition(
        child_cache_dir,
        parent_root_id=root_id,
        action_index=action_index,
    )

    loader_cfg = RootChildGenerationConfig(
        dataset_dir=Path(dataset_dir).expanduser(),
        output_dir=output_dir / "_loader_tmp",
        max_roots=1,
        root_player_filter="controller",
        overwrite=True,
        seed=int(seed),
    )
    state_loader = _make_state_loader(loader_cfg)
    try:
        child_state = _child_state_from_cached_row(state_loader, child_row)
        child_trace_row = _append_child_trace_row(
            trace_rows=parent_rows,
            child_row=child_row,
            child_state=child_state,
            state_loader=state_loader,
        )
    finally:
        state_loader.close()

    out_csv = output_dir / f"{parent_trace_csv.stem}_with_selected_action_{action_index:04d}.csv"
    _write_trace_csv(out_csv, parent_rows + [child_trace_row])

    failed_csv = ""
    error = ""
    try:
        trace_count, adv_actions_checked = _validate_trace_csv(out_csv)
        passed = True
    except Exception as exc:
        passed = False
        trace_count = 0
        adv_actions_checked = 0
        error = repr(exc)
        failed_path = out_csv.with_name(out_csv.stem + "_failed.csv")
        failed_path.write_text(out_csv.read_text(encoding="utf-8"), encoding="utf-8")
        failed_csv = str(failed_path)

    summary = {
        "parent_trace_csv": str(parent_trace_csv),
        "selected_action_csv": str(selected_action_csv),
        "child_cache_dir": str(Path(child_cache_dir).expanduser()),
        "output_csv": str(out_csv),
        "failed_csv": failed_csv,
        "root_id": int(root_id),
        "selected_action_index": int(action_index),
        "trace_chains_checked": int(trace_count),
        "adv_actions_checked": int(adv_actions_checked),
        "passed": bool(passed),
        "error": error,
    }

    summary_path = output_dir / "selected_action_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "selected_action_trace_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)

    if not passed:
        raise RuntimeError(f"selected-action trace validation failed: {error}; output_csv={out_csv}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append selected child action to an existing root trace and validate it.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--child-cache-dir", required=True)
    parser.add_argument("--parent-trace-csv", required=True)
    parser.add_argument("--selected-action-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = append_selected_action_trace(
        dataset_dir=args.dataset_dir,
        child_cache_dir=args.child_cache_dir,
        parent_trace_csv=args.parent_trace_csv,
        selected_action_csv=args.selected_action_csv,
        output_dir=args.output_dir,
        seed=int(args.seed),
    )
    print(
        "selected-action trace passed: "
        f"root_id={summary['root_id']} "
        f"action_index={summary['selected_action_index']} "
        f"output_csv={summary['output_csv']}"
    )


if __name__ == "__main__":
    main()
