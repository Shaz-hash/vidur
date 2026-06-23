#!/usr/bin/env python3
"""
Create and validate controller->adversary child traces for the Adv transition cache.

Default smoke command:

PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur-classical-search \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  -m experiments.new_features_v1.test_and_analysis.rootChildSelectedActionTraceTesting \
  --max-traces 5

Default output:
/home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/GV3_Agent/rootChildAdvTraceSmoke
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import (  # noqa: E402
    RootChildGenerationConfig,
    _make_state_loader,
    generate_child_states,
    load_samples,
    test_child_states,
)
from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.analysis_testing.rootChildTraceTesting import (  # noqa: E402
    _controller_payload_from_repr,
    _json_dumps,
    _load_parent_trace_rows,
    _logger_row,
    _safe_float,
    _safe_int,
    _write_trace_csv,
)
from vidur.mcts.Game_Versions.Game_Version3.tests.history_node_tests import _validate_trace_csv  # noqa: E402


@dataclass(frozen=True)
class AdvSelectedActionTraceConfig:
    dataset_dir: Path
    output_dir: Path
    parent_state_id: int | None = None
    max_traces: int = 5
    require_stored_parent_trace: bool = True
    overwrite: bool = True
    seed: int = 2027


def _default_dataset_dir() -> Path:
    plain_subset = Path(
        "/home/shazer/Desktop/Research/Vidur/vidur/"
        "simulator_output/GV3_Agent/rootChildTraceParentSubset/stored_roots"
    )
    if (plain_subset / "manifest.jsonl").exists():
        return plain_subset
    return REPO_ROOT / "simulator_output/GV3_Agent/rootChildTraceParentSubset/stored_roots"


def _default_output_dir() -> Path:
    return REPO_ROOT / "simulator_output/GV3_Agent/rootChildAdvTraceSmoke"


def _clean_output_dir(cfg: AdvSelectedActionTraceConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.overwrite:
        return
    for pattern in (
        "root_child_adv_trace_*.csv",
        "root_child_adv_trace_*_failed.csv",
        "history_node_tests_failed_trace.csv",
        "trace_test_summary.csv",
        "trace_test_summary.json",
        "adv_selected_action_trace_config.json",
    ):
        for path in cfg.output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_from_snapshot(state_loader: Any, snapshot: Any, stats: Any) -> Any:
    clone_fn = getattr(state_loader.env, "clone_state_from_snapshot", None)
    if callable(clone_fn):
        return clone_fn(snapshot, stats)
    state = state_loader.env.initial_state()
    state.simulator.restore_state(snapshot)
    state.stats = stats.clone()
    return state


_REQ_RE = re.compile(
    r"AdversaryRequestSpec\("
    r"prefill_tokens=(?P<prefill_tokens>\d+),\s*"
    r"decode_tokens=(?P<decode_tokens>\d+),\s*"
    r"prefill_slo=(?P<prefill_slo>[-+0-9.eE]+),\s*"
    r"decode_slo=(?P<decode_slo>[-+0-9.eE]+)"
    r"\)"
)


def _adversary_payload_from_repr(action_repr: str) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for match in _REQ_RE.finditer(str(action_repr)):
        requests.append(
            {
                "prefill_tokens": int(match.group("prefill_tokens")),
                "decode_tokens": int(match.group("decode_tokens")),
                "prefill_slo": float(match.group("prefill_slo")),
                "decode_slo": float(match.group("decode_slo")),
            }
        )
    return {
        "adversary_requests": _json_dumps(requests),
        "adversary_prefill_slos": _json_dumps([float(r["prefill_slo"]) for r in requests]),
        "adversary_prefill_deadlines_by_id": "{}",
        "adversary_decode_slos": _json_dumps([float(r["decode_slo"]) for r in requests]),
    }


def _append_adv_transition_rows(
    *,
    trace_rows: list[dict[str, Any]],
    child_row: dict[str, Any],
    intermediate_state: Any,
    child_state: Any,
    state_loader: Any,
) -> list[dict[str, Any]]:
    last = trace_rows[-1]
    parent_node_id = _safe_int(last.get("node_id"), 0)
    parent_depth = _safe_int(last.get("node_depth"), 1)
    parent_sim_time = _safe_float(last.get("sim_time"), 0.0)

    intermediate_snapshot = state_loader.env.describe_state(intermediate_state)
    child_snapshot = state_loader.env.describe_state(child_state)
    intermediate_time = _safe_float(
        intermediate_snapshot.get("sim_time"),
        _safe_float(child_row.get("intermediate_time"), parent_sim_time),
    )
    child_time = _safe_float(
        child_snapshot.get("sim_time"),
        _safe_float(child_row.get("child_time"), intermediate_time),
    )

    controller_node_id = parent_node_id + 1
    adversary_node_id = controller_node_id + 1
    controller_action_repr = str(child_row.get("controller_action_repr", child_row.get("action_repr", "")))
    adversary_action_repr = str(child_row.get("adversary_action_repr", ""))

    controller_row = _logger_row(
        game_id=_safe_int(last.get("game_id"), 0),
        root_id=_safe_int(last.get("root_id"), _safe_int(child_row.get("parent_root_id"), 0)),
        sim_iteration=_safe_int(last.get("sim_iteration"), -1) + 1,
        root_depth=_safe_int(last.get("root_depth"), parent_depth) + 1,
        root_node_id=_safe_int(last.get("root_node_id"), 0),
        root_player=str(last.get("root_player") or "adversary"),
        phase="history-child-transition-adv-controller",
        node_depth=parent_depth + 1,
        parent_node_id=parent_node_id,
        node_id=controller_node_id,
        player_acted="controller",
        player_to_act="adversary",
        action_index=int(child_row.get("controller_action_index", child_row.get("action_index", -1))),
        action_repr=controller_action_repr,
        reward=float(child_row.get("controller_reward", child_row.get("reward", 0.0))),
        objective_cost=float(child_row.get("intermediate_cost", 0.0)),
        state_snapshot=intermediate_snapshot,
        num_valid_actions=_safe_int((child_row.get("row") or {}).get("num_controller_valid_actions"), 0),
        unique_actions=_safe_int((child_row.get("row") or {}).get("num_controller_canonical_actions"), 0),
        decision_state_time=parent_sim_time,
        start_time=parent_sim_time,
        end_time=intermediate_time,
        stage_total_time=max(0.0, intermediate_time - parent_sim_time),
        extra_payload=_controller_payload_from_repr(controller_action_repr),
    )

    adversary_row = _logger_row(
        game_id=_safe_int(last.get("game_id"), 0),
        root_id=_safe_int(last.get("root_id"), _safe_int(child_row.get("parent_root_id"), 0)),
        sim_iteration=_safe_int(last.get("sim_iteration"), -1) + 2,
        root_depth=_safe_int(last.get("root_depth"), parent_depth) + 2,
        root_node_id=_safe_int(last.get("root_node_id"), 0),
        root_player=str(last.get("root_player") or "adversary"),
        phase="history-child-transition-adv-adversary",
        node_depth=parent_depth + 2,
        parent_node_id=controller_node_id,
        node_id=adversary_node_id,
        player_acted="adversary",
        player_to_act="controller",
        action_index=int(child_row.get("adversary_action_index", -1)),
        action_repr=adversary_action_repr,
        reward=0.0,
        objective_cost=float(child_row.get("child_cost", 0.0)),
        state_snapshot=child_snapshot,
        num_valid_actions=_safe_int((child_row.get("row") or {}).get("num_adversary_valid_actions"), 0),
        unique_actions=_safe_int((child_row.get("row") or {}).get("num_adversary_valid_actions"), 0),
        decision_state_time=intermediate_time,
        start_time=intermediate_time,
        end_time=child_time,
        stage_total_time=max(0.0, child_time - intermediate_time),
        extra_payload=_adversary_payload_from_repr(adversary_action_repr),
    )
    return [controller_row, adversary_row]


def _load_parent_record(cfg: AdvSelectedActionTraceConfig) -> tuple[int, dict[str, Any]]:
    if cfg.parent_state_id is not None:
        target = int(cfg.parent_state_id)
        for parent_state_id, record in load_samples(
            cfg.dataset_dir,
            root_player_filter="controller",
            max_roots=target + 1,
        ):
            if int(parent_state_id) == target:
                return int(parent_state_id), record
        raise RuntimeError(f"parent_state_id={target} not found in {cfg.dataset_dir}")

    for parent_state_id, record in load_samples(cfg.dataset_dir, root_player_filter="controller"):
        if record.get("history_trace_logs"):
            return int(parent_state_id), record
    raise RuntimeError(f"no controller parent with history_trace_logs found in {cfg.dataset_dir}")


def create_adv_trace_smoke_csvs(cfg: AdvSelectedActionTraceConfig) -> dict[str, Any]:
    _clean_output_dir(cfg)
    _write_json(cfg.output_dir / "adv_selected_action_trace_config.json", asdict(cfg))

    parent_state_id, record = _load_parent_record(cfg)
    loader_cfg = RootChildGenerationConfig(
        dataset_dir=cfg.dataset_dir,
        output_dir=cfg.output_dir / "_loader_tmp",
        max_roots=parent_state_id + 1,
        root_player_filter="controller",
        overwrite=True,
        seed=int(cfg.seed),
    )
    state_loader = _make_state_loader(loader_cfg)

    rows_for_summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    passed = 0

    try:
        parent_state = state_loader(record)
        children_all = generate_child_states(
            parent_state_id=parent_state_id,
            root_record=record,
            root_parent_state=parent_state,
            mcts=state_loader.mcts,
            include_alias_rows=False,
        )
        valid_children = [row for row in children_all if int(row.get("adversary_action_index", -1)) >= 0]
        nonempty_adversary_children = [
            row
            for row in valid_children
            if "AdversaryRequestSpec" in str(row.get("adversary_action_repr", ""))
        ]
        selection_source = (
            "nonempty_adversary"
            if len(nonempty_adversary_children) >= int(cfg.max_traces)
            else "all_valid_adversary"
        )
        children = (
            nonempty_adversary_children
            if selection_source == "nonempty_adversary"
            else valid_children
        )[: int(cfg.max_traces)]
        if len(children) < int(cfg.max_traces):
            raise RuntimeError(
                f"only generated {len(children)} adversary-expanded children for parent_state_id={parent_state_id}"
            )

        test_child_states(
            children,
            parent_records_by_id={parent_state_id: record},
            state_loader=state_loader,
            max_checks=len(children),
        )

        parent_trace_rows, trace_source = _load_parent_trace_rows(
            record=record,
            parent_state=parent_state,
            state_loader=state_loader,
            allow_synthetic_parent_trace=not bool(cfg.require_stored_parent_trace),
        )
        if not parent_trace_rows:
            raise RuntimeError(f"parent trace unavailable; source={trace_source}")

        for i, child_row in enumerate(children):
            intermediate_state = _state_from_snapshot(
                state_loader,
                child_row["intermediate_simulator_snapshot"],
                child_row["intermediate_stats"],
            )
            child_state = _state_from_snapshot(
                state_loader,
                child_row["child_simulator_snapshot"],
                child_row["child_stats"],
            )
            appended_rows = _append_adv_transition_rows(
                trace_rows=parent_trace_rows,
                child_row=child_row,
                intermediate_state=intermediate_state,
                child_state=child_state,
                state_loader=state_loader,
            )
            combined_rows = parent_trace_rows + appended_rows
            trace_csv = (
                cfg.output_dir
                / (
                    f"root_child_adv_trace_parent_{parent_state_id:06d}_"
                    f"ctrl_{int(child_row.get('controller_action_index', -1)):04d}_"
                    f"adv_{int(child_row.get('adversary_action_index', -1)):04d}.csv"
                )
            )
            _write_trace_csv(trace_csv, combined_rows)

            try:
                trace_count, adv_actions_checked = _validate_trace_csv(trace_csv)
                passed += 1
                summary_row = {
                    "trace_index": i,
                    "parent_state_id": parent_state_id,
                    "parent_root_id": int(child_row.get("parent_root_id", -1)),
                    "controller_action_index": int(child_row.get("controller_action_index", -1)),
                    "adversary_action_index": int(child_row.get("adversary_action_index", -1)),
                    "trace_source": trace_source,
                    "trace_csv": str(trace_csv),
                    "trace_chains_checked": int(trace_count),
                    "adv_actions_checked": int(adv_actions_checked),
                    "passed": True,
                    "error": "",
                }
            except Exception as exc:
                failed_copy = trace_csv.with_name(trace_csv.stem + "_failed.csv")
                shutil.copy2(trace_csv, failed_copy)
                failure = {
                    "trace_index": i,
                    "parent_state_id": parent_state_id,
                    "parent_root_id": int(child_row.get("parent_root_id", -1)),
                    "controller_action_index": int(child_row.get("controller_action_index", -1)),
                    "adversary_action_index": int(child_row.get("adversary_action_index", -1)),
                    "trace_csv": str(trace_csv),
                    "failed_trace_csv": str(failed_copy),
                    "error": repr(exc),
                }
                failures.append(failure)
                summary_row = dict(failure)
                summary_row.update(
                    {
                        "trace_source": trace_source,
                        "trace_chains_checked": 0,
                        "adv_actions_checked": 0,
                        "passed": False,
                    }
                )
            rows_for_summary.append(summary_row)
    finally:
        state_loader.close()

    fieldnames = [
        "trace_index",
        "parent_state_id",
        "parent_root_id",
        "controller_action_index",
        "adversary_action_index",
        "trace_source",
        "trace_csv",
        "trace_chains_checked",
        "adv_actions_checked",
        "passed",
        "error",
    ]
    with (cfg.output_dir / "trace_test_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_for_summary:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    summary = {
        "dataset_dir": str(cfg.dataset_dir),
        "output_dir": str(cfg.output_dir),
        "parent_state_id": int(parent_state_id),
        "parent_root_id": int(record.get("root_id", -1)),
        "children_generated_for_parent": int(len(children_all)),
        "selection_source": selection_source,
        "nonempty_adversary_children": int(len(nonempty_adversary_children)),
        "traces_written": int(len(rows_for_summary)),
        "traces_passed": int(passed),
        "failures": failures,
        "passed": not failures and passed == len(rows_for_summary),
    }
    _write_json(cfg.output_dir / "trace_test_summary.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} Adv trace validation(s) failed; see {cfg.output_dir}")
    return summary


def parse_args() -> AdvSelectedActionTraceConfig:
    parser = argparse.ArgumentParser(description="Create Adv selected-action trace smoke CSVs.")
    parser.add_argument("--dataset-dir", default=str(_default_dataset_dir()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--parent-state-id", type=int, default=None)
    parser.add_argument("--max-traces", type=int, default=5)
    parser.add_argument("--allow-synthetic-parent-trace", action="store_true")
    parser.add_argument("--reuse-existing-output", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if int(args.max_traces) <= 0:
        raise ValueError("--max-traces must be > 0")
    return AdvSelectedActionTraceConfig(
        dataset_dir=Path(args.dataset_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        parent_state_id=args.parent_state_id,
        max_traces=int(args.max_traces),
        require_stored_parent_trace=not bool(args.allow_synthetic_parent_trace),
        overwrite=not bool(args.reuse_existing_output),
        seed=int(args.seed),
    )


def main() -> None:
    cfg = parse_args()
    summary = create_adv_trace_smoke_csvs(cfg)
    print(
        "Adv selected-action trace smoke passed: "
        f"traces={summary['traces_passed']}/{summary['traces_written']}, "
        f"parent_state_id={summary['parent_state_id']}, "
        f"output_dir={cfg.output_dir}"
    )


if __name__ == "__main__":
    main()
