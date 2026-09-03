#!/usr/bin/env python3
"""Exhaustively trace-test valid controller actions from stored GV3 parent states.

This harness intentionally loads parent states from an existing stored root
dataset. It does not synthesize a parent from a fresh simulator. For each chosen
stored parent:

1. Load the stored simulator snapshot and stats.
2. Load the stored `history_trace_logs` for that parent.
3. Enumerate every controller action marked valid by `sample_controller_actions`.
4. Apply each valid action to a cloned parent state.
5. Append the resulting controller transition row to the parent history trace.
6. Validate the resulting trace CSV.

Default command:

PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur-classical-search \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  -m vidur.Game_Version3.tests.sample_controller_action_transition_tests
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..ModelSearchBed.analysis_testing.rootChildGeneration import (
    RootChildGenerationConfig,
    _make_state_loader,
    load_samples,
)
from ..ModelSearchBed.analysis_testing.rootChildTraceTesting import (
    _controller_payload_from_repr,
    _load_parent_trace_rows,
    _logger_row,
    _safe_float,
    _safe_int,
    _write_trace_csv,
)
from .history_node_tests import _validate_trace_csv


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACE_CSV = (
    REPO_ROOT
    / "simulator_output/GV3_Agent/rootChildAdvTraceSmoke/"
    "root_child_adv_trace_parent_000021_ctrl_0008_adv_0005.csv"
)
DEFAULT_CLASSICAL_DATASET_DIR = (
    REPO_ROOT
    / "simulator_output/GV3_Agent/rootChildTraceParentSubset/stored_roots"
)
DEFAULT_MAIN_DATASET_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur/"
    "simulator_output/GV3_Agent/rootChildTraceParentSubset/stored_roots"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "simulator_output/GV3_Agent/sampleControllerActionTraceTests"
)


@dataclass(frozen=True)
class SampleControllerActionTraceConfig:
    dataset_dir: Path
    output_dir: Path
    source_trace_csv: Path | None = DEFAULT_TRACE_CSV
    parent_state_id: int | None = None
    max_actions: int | None = None
    require_stored_parent_trace: bool = True
    overwrite: bool = True
    seed: int = 2027


def _default_dataset_dir() -> Path:
    if DEFAULT_CLASSICAL_DATASET_DIR.exists():
        return DEFAULT_CLASSICAL_DATASET_DIR
    return DEFAULT_MAIN_DATASET_DIR


def _clean_output_dir(cfg: SampleControllerActionTraceConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.overwrite:
        return
    for pattern in (
        "controller_action_trace_*.csv",
        "controller_action_trace_*_failed.csv",
        "history_node_tests_failed_trace.csv",
        "summary.csv",
        "summary.json",
        "config.json",
    ):
        for path in cfg.output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _infer_parent_state_id(trace_csv: Path | None, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    if trace_csv is None:
        raise ValueError("parent_state_id is required when source_trace_csv is not provided")
    match = re.search(r"parent_(\d+)", str(trace_csv.name))
    if not match:
        raise ValueError(f"could not infer parent_state_id from trace path: {trace_csv}")
    return int(match.group(1))


def _load_parent_record(
    dataset_dir: Path,
    *,
    parent_state_id: int,
) -> dict[str, Any]:
    for sample_id, record in load_samples(
        dataset_dir,
        root_player_filter="controller",
        max_roots=int(parent_state_id) + 1,
    ):
        if int(sample_id) == int(parent_state_id):
            return record
    raise RuntimeError(f"parent_state_id={parent_state_id} not found in {dataset_dir}")


def _state_cost(env: Any, state: Any) -> float:
    violations, lateness = env.evaluate_objective(state)
    return float(violations) + float(lateness)


def _clone_parent_state(parent_state: Any) -> Any:
    fork = getattr(parent_state, "fork", None)
    if callable(fork):
        try:
            return fork(flag=False)
        except TypeError:
            return fork()
    raise TypeError(f"parent state does not expose fork(): {type(parent_state)!r}")


def _last_parent_trace_values(parent_trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not parent_trace_rows:
        raise ValueError("parent_trace_rows is empty")
    last = parent_trace_rows[-1]
    return {
        "game_id": _safe_int(last.get("game_id"), 0),
        "root_id": _safe_int(last.get("root_id"), 0),
        "root_depth": _safe_int(last.get("root_depth"), _safe_int(last.get("node_depth"), 0)),
        "root_node_id": _safe_int(last.get("root_node_id"), 0),
        "root_player": str(last.get("root_player") or "adversary"),
        "parent_node_id": _safe_int(last.get("node_id"), 0),
        "parent_depth": _safe_int(last.get("node_depth"), 0),
        "sim_iteration": _safe_int(last.get("sim_iteration"), 0),
        "sim_time": _safe_float(last.get("sim_time"), 0.0),
    }


def _build_controller_transition_row(
    *,
    parent_trace_rows: list[dict[str, Any]],
    action_index: int,
    action: Any,
    child_state: Any,
    env: Any,
    parent_cost: float,
    child_cost: float,
    num_valid_actions: int,
    unique_actions: int,
) -> dict[str, Any]:
    parent = _last_parent_trace_values(parent_trace_rows)
    child_snapshot = env.describe_state(child_state)
    child_time = _safe_float(child_snapshot.get("sim_time"), parent["sim_time"])
    action_repr = repr(action)
    reward = -(max(0.0, float(child_cost) - float(parent_cost)))

    return _logger_row(
        game_id=int(parent["game_id"]),
        root_id=int(parent["root_id"]),
        sim_iteration=int(parent["sim_iteration"]) + 1,
        root_depth=int(parent["root_depth"]) + 1,
        root_node_id=int(parent["root_node_id"]),
        root_player=str(parent["root_player"]),
        phase="sample-controller-action-transition",
        node_depth=int(parent["parent_depth"]) + 1,
        parent_node_id=int(parent["parent_node_id"]),
        node_id=int(parent["parent_node_id"]) + 1,
        player_acted="controller",
        player_to_act="adversary",
        action_index=int(action_index),
        action_repr=action_repr,
        reward=float(reward),
        objective_cost=float(child_cost),
        state_snapshot=child_snapshot,
        num_valid_actions=int(num_valid_actions),
        unique_actions=int(unique_actions),
        decision_state_time=float(parent["sim_time"]),
        start_time=float(parent["sim_time"]),
        end_time=float(child_time),
        stage_total_time=max(0.0, float(child_time) - float(parent["sim_time"])),
        extra_payload=_controller_payload_from_repr(action_repr),
    )


def _valid_controller_actions(env: Any, parent_state: Any) -> list[tuple[int, Any]]:
    actions_by_index, mask = env.sample_controller_actions(parent_state)
    out: list[tuple[int, Any]] = []
    for idx, ok in enumerate(mask):
        if bool(ok) and actions_by_index[int(idx)] is not None:
            out.append((int(idx), actions_by_index[int(idx)]))
    return out


def run_sample_controller_action_trace_test(
    cfg: SampleControllerActionTraceConfig,
) -> dict[str, Any]:
    _clean_output_dir(cfg)
    _write_json(cfg.output_dir / "config.json", asdict(cfg))

    parent_state_id = _infer_parent_state_id(cfg.source_trace_csv, cfg.parent_state_id)
    record = _load_parent_record(cfg.dataset_dir, parent_state_id=parent_state_id)
    if str(record.get("root_player", "")) != "controller":
        raise RuntimeError(
            f"parent_state_id={parent_state_id} is not a controller root: "
            f"root_player={record.get('root_player')!r}"
        )
    if cfg.require_stored_parent_trace and not record.get("history_trace_logs"):
        raise RuntimeError(f"parent_state_id={parent_state_id} has no stored history_trace_logs")

    loader_cfg = RootChildGenerationConfig(
        dataset_dir=cfg.dataset_dir,
        output_dir=cfg.output_dir / "_loader_tmp",
        max_roots=int(parent_state_id) + 1,
        root_player_filter="controller",
        overwrite=True,
        seed=int(cfg.seed),
    )
    state_loader = _make_state_loader(loader_cfg)

    rows_for_summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parent_root_id = int(record.get("root_id", -1))

    try:
        parent_state = state_loader(record)
        parent_cost = _state_cost(state_loader.env, parent_state)
        parent_trace_rows, trace_source = _load_parent_trace_rows(
            record=record,
            parent_state=parent_state,
            state_loader=state_loader,
            allow_synthetic_parent_trace=not bool(cfg.require_stored_parent_trace),
        )
        if not parent_trace_rows:
            raise RuntimeError(f"parent trace unavailable for parent_state_id={parent_state_id}")

        valid_actions = _valid_controller_actions(state_loader.env, parent_state)
        if cfg.max_actions is not None:
            valid_actions = valid_actions[: int(cfg.max_actions)]
        if not valid_actions:
            raise RuntimeError(f"no valid controller actions for parent_state_id={parent_state_id}")

        unique_actions = len({repr(action) for _, action in valid_actions})
        for trace_index, (action_index, action) in enumerate(valid_actions):
            child_state = state_loader.env.apply_controller_action_only(
                _clone_parent_state(parent_state),
                action,
                inplace=True,
            )
            child_cost = _state_cost(state_loader.env, child_state)
            transition_row = _build_controller_transition_row(
                parent_trace_rows=parent_trace_rows,
                action_index=action_index,
                action=action,
                child_state=child_state,
                env=state_loader.env,
                parent_cost=parent_cost,
                child_cost=child_cost,
                num_valid_actions=len(valid_actions),
                unique_actions=unique_actions,
            )
            combined_rows = list(parent_trace_rows) + [transition_row]
            trace_csv = (
                cfg.output_dir
                / f"controller_action_trace_parent_{parent_state_id:06d}_action_{action_index:04d}.csv"
            )
            _write_trace_csv(trace_csv, combined_rows)

            summary_row = {
                "trace_index": trace_index,
                "parent_state_id": parent_state_id,
                "parent_root_id": parent_root_id,
                "action_index": int(action_index),
                "action_repr": repr(action),
                "parent_cost": float(parent_cost),
                "child_cost": float(child_cost),
                "reward": -(max(0.0, float(child_cost) - float(parent_cost))),
                "trace_source": trace_source,
                "trace_csv": str(trace_csv),
                "trace_chains_checked": 0,
                "adv_actions_checked": 0,
                "passed": False,
                "error": "",
            }
            try:
                trace_count, adv_actions_checked = _validate_trace_csv(trace_csv)
                summary_row.update(
                    {
                        "trace_chains_checked": int(trace_count),
                        "adv_actions_checked": int(adv_actions_checked),
                        "passed": True,
                    }
                )
            except Exception as exc:
                failed_copy = trace_csv.with_name(trace_csv.stem + "_failed.csv")
                shutil.copy2(trace_csv, failed_copy)
                summary_row["error"] = repr(exc)
                failures.append(dict(summary_row))
            rows_for_summary.append(summary_row)
    finally:
        state_loader.close()

    fieldnames = [
        "trace_index",
        "parent_state_id",
        "parent_root_id",
        "action_index",
        "action_repr",
        "parent_cost",
        "child_cost",
        "reward",
        "trace_source",
        "trace_csv",
        "trace_chains_checked",
        "adv_actions_checked",
        "passed",
        "error",
    ]
    with (cfg.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_for_summary:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    summary = {
        "dataset_dir": str(cfg.dataset_dir),
        "source_trace_csv": "" if cfg.source_trace_csv is None else str(cfg.source_trace_csv),
        "output_dir": str(cfg.output_dir),
        "parent_state_id": int(parent_state_id),
        "parent_root_id": int(parent_root_id),
        "parent_cost": float(rows_for_summary[0]["parent_cost"]) if rows_for_summary else None,
        "valid_controller_actions_tested": int(len(rows_for_summary)),
        "traces_passed": int(sum(1 for row in rows_for_summary if row.get("passed"))),
        "failures": failures,
        "passed": not failures and bool(rows_for_summary),
    }
    _write_json(cfg.output_dir / "summary.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} controller action trace validation(s) failed; see {cfg.output_dir}")
    return summary


def parse_args() -> SampleControllerActionTraceConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(_default_dataset_dir()))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source-trace-csv", default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--parent-state-id", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--allow-synthetic-parent-trace", action="store_true")
    parser.add_argument("--reuse-existing-output", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    source_trace = None if str(args.source_trace_csv).strip() == "" else Path(args.source_trace_csv).expanduser()
    return SampleControllerActionTraceConfig(
        dataset_dir=Path(args.dataset_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        source_trace_csv=source_trace,
        parent_state_id=args.parent_state_id,
        max_actions=args.max_actions,
        require_stored_parent_trace=not bool(args.allow_synthetic_parent_trace),
        overwrite=not bool(args.reuse_existing_output),
        seed=int(args.seed),
    )


def main() -> None:
    cfg = parse_args()
    summary = run_sample_controller_action_trace_test(cfg)
    print(
        "Controller sampled-action trace test passed: "
        f"traces={summary['traces_passed']}/{summary['valid_controller_actions_tested']}, "
        f"parent_state_id={summary['parent_state_id']}, "
        f"output_dir={cfg.output_dir}"
    )


if __name__ == "__main__":
    main()
