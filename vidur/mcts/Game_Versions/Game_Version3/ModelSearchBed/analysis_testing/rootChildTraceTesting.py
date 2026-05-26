"""Build parent+child trace CSVs from cached child transitions and validate them.

The intended strict path is:
    1. Load a cached child transition.
    2. Load its parent root record.
    3. Take the parent record's stored `history_trace_logs`.
    4. Append one logger-compatible child transition row.
    5. Run the existing GV3 history trace validator on the resulting CSV.

The large controller-root dataset was generated without parent trace logs to
save space. For smoke testing that cache, this script can synthesize a minimal
single-adversary parent row for simple one-request roots. That fallback is only
for validating the child-transition trace plumbing; it is not a replacement for
stored full parent history traces.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

try:
    from .rootChildGeneration import (
        RootChildGenerationConfig,
        _make_state_loader,
        load_samples,
    )
    from ...config import GameVersion2Config
    from ...logger.mctsDNN_logger import DNNMCTSIterationLogger
    from ...tests.history_node_tests import _validate_trace_csv
except ImportError:
    import sys

    repo_root = Path(__file__).resolve().parents[6]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.analysis_testing.rootChildGeneration import (
        RootChildGenerationConfig,
        _make_state_loader,
        load_samples,
    )
    from vidur.mcts.Game_Versions.Game_Version3.config import GameVersion2Config
    from vidur.mcts.Game_Versions.Game_Version3.logger.mctsDNN_logger import DNNMCTSIterationLogger
    from vidur.mcts.Game_Versions.Game_Version3.tests.history_node_tests import _validate_trace_csv


@dataclass(frozen=True)
class RootChildTraceTestingConfig:
    dataset_dir: Path
    child_cache_dir: Path
    output_dir: Path
    max_children: int = 20
    canonical_only: bool = True
    allow_synthetic_parent_trace: bool = True
    overwrite: bool = True
    seed: int = 2027


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_dataset_dir() -> Path:
    return (
        _repo_root()
        / "simulator_output"
        / "GV3_Agent"
        / "model_search_roots_controller_350k_abs1_ratio40"
    )


def _default_child_cache_dir() -> Path:
    return (
        _repo_root()
        / "simulator_output"
        / "GV3_Agent"
        / "model_search_roots_controller_350k_abs1_ratio40_child_transitions_smoke_codex"
    )


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "GV3_Agent" / "rootChildTraceTests"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _empty_logger_row() -> dict[str, Any]:
    return {field: "" for field in DNNMCTSIterationLogger.FIELDS}


def _clean_output_dir(cfg: RootChildTraceTestingConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.overwrite:
        return

    for pattern in (
        "root_child_trace_*.csv",
        "trace_test_summary.csv",
        "trace_test_summary.json",
        "root_child_trace_testing_config.json",
        "history_node_tests_failed_trace.csv",
    ):
        for path in cfg.output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DNNMCTSIterationLogger.FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DNNMCTSIterationLogger.FIELDS})


def load_child_transitions(
    child_cache_dir: str | Path,
    *,
    max_children: int,
    canonical_only: bool,
) -> list[dict[str, Any]]:
    """Load cached child transitions from child-transition shards."""

    cache_dir = Path(child_cache_dir).expanduser()
    manifest = cache_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"child cache manifest not found: {manifest}")

    out: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            shard_path = cache_dir / str(entry["shard_path"])
            rows = torch.load(shard_path, map_location="cpu", weights_only=False)
            for row in rows:
                if canonical_only and int(row["action_index"]) != int(row["canonical_action_index"]):
                    continue
                out.append(row)
                if len(out) >= int(max_children):
                    return out

    return out


def load_parent_records(
    dataset_dir: str | Path,
    parent_state_ids: set[int],
) -> dict[int, dict[str, Any]]:
    """Load the parent records needed by selected child transitions."""

    if not parent_state_ids:
        return {}

    max_parent_id = max(int(x) for x in parent_state_ids)
    records: dict[int, dict[str, Any]] = {}
    for parent_state_id, record in load_samples(
        dataset_dir,
        root_player_filter="controller",
        max_roots=max_parent_id + 1,
    ):
        if int(parent_state_id) in parent_state_ids:
            records[int(parent_state_id)] = record
            if len(records) == len(parent_state_ids):
                break

    missing = sorted(parent_state_ids - set(records))
    if missing:
        raise RuntimeError(f"missing parent records for parent_state_ids={missing[:20]}")
    return records


def _extract_literal_from_repr(action_repr: str, field_name: str) -> Any:
    match = re.search(rf"{re.escape(field_name)}=([^,)]+(?:\}}|\])?)", action_repr)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return None


def _extract_int_from_repr(action_repr: str, field_name: str) -> int | None:
    match = re.search(rf"{re.escape(field_name)}=(-?\d+)", action_repr)
    if not match:
        return None
    return int(match.group(1))


def _extract_str_from_repr(action_repr: str, field_name: str) -> str:
    match = re.search(rf"{re.escape(field_name)}='([^']*)'", action_repr)
    return "" if not match else str(match.group(1))


def _controller_payload_from_repr(action_repr: str) -> dict[str, Any]:
    token_alloc = _extract_literal_from_repr(action_repr, "token_allocations") or {}
    prefill_alloc = _extract_literal_from_repr(action_repr, "prefill_allocations") or {}
    decode_alloc = _extract_literal_from_repr(action_repr, "decode_allocations") or {}
    selected_ids = _extract_literal_from_repr(action_repr, "selected_request_ids") or []
    token_budget = _extract_int_from_repr(action_repr, "token_budget")
    if token_budget is None:
        token_budget = sum(int(v) for v in token_alloc.values())

    return {
        "controller_token_budget": int(token_budget),
        "controller_selected_ids": _json_dumps([int(x) for x in selected_ids]),
        "controller_allocations": _json_dumps({str(int(k)): int(v) for k, v in token_alloc.items()}),
        "controller_prefill_allocations": _json_dumps({str(int(k)): int(v) for k, v in prefill_alloc.items()}),
        "controller_decode_allocations": _json_dumps({str(int(k)): int(v) for k, v in decode_alloc.items()}),
        "controller_prefill_total": sum(int(v) for v in prefill_alloc.values()),
        "controller_decode_total": sum(int(v) for v in decode_alloc.values()),
        "controller_heuristic": _extract_str_from_repr(action_repr, "heuristic"),
        "controller_strategy": _extract_str_from_repr(action_repr, "strategy"),
    }


def _logger_row(
    *,
    game_id: int,
    root_id: int,
    sim_iteration: int,
    root_depth: int,
    root_node_id: int,
    root_player: str,
    phase: str,
    node_depth: int,
    parent_node_id: int | None,
    node_id: int,
    player_acted: str,
    player_to_act: str,
    action_index: int,
    action_repr: str,
    reward: float,
    objective_cost: float,
    state_snapshot: dict[str, Any],
    num_valid_actions: int = 0,
    unique_actions: int = 0,
    decision_state_time: float | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    stage_total_time: float | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sim_time = _safe_float(state_snapshot.get("sim_time"), 0.0)
    st = sim_time if start_time is None else _safe_float(start_time, sim_time)
    et = sim_time if end_time is None else _safe_float(end_time, sim_time)
    dt = max(0.0, et - st) if stage_total_time is None else _safe_float(stage_total_time)

    row = _empty_logger_row()
    row.update(
        {
            "game_id": int(game_id),
            "root_id": int(root_id),
            "sim_iteration": int(sim_iteration),
            "root_depth": int(root_depth),
            "root_node_id": int(root_node_id),
            "root_player": str(root_player),
            "phase": str(phase),
            "node_depth": int(node_depth),
            "parent_node_id": "" if parent_node_id is None else str(int(parent_node_id)),
            "node_id": int(node_id),
            "player_acted_to_create_this_node": str(player_acted),
            "player_to_act_in_this_node": str(player_to_act),
            "action_index": int(action_index),
            "action_repr": str(action_repr),
            "prior": 1.0,
            "model_prior_json": "[]",
            "normalized_prior_json": "[]",
            "reward": float(reward),
            "nn_called": False,
            "num_valid_actions": int(num_valid_actions),
            "unique_actions": int(unique_actions),
            "nn_value_controller": "",
            "objective_cost": float(objective_cost),
            "model_top5_actions_json": "[]",
            "mcts_top5_actions_json": "[]",
            "sim_time": sim_time,
            "decision_state_time": sim_time if decision_state_time is None else float(decision_state_time),
            "start_time": st,
            "end_time": et,
            "stage_total_time": dt,
            "requests_in_system": state_snapshot.get("requests_in_system", 0),
            "requests_generated": state_snapshot.get("requests_generated", 0),
            "requests_completed": state_snapshot.get("requests_completed", 0),
            "slo_violations": state_snapshot.get("slo_violations", 0),
            "total_lateness": state_snapshot.get("total_lateness", state_snapshot.get("avg_lateness", 0.0)),
            "avg_lateness": state_snapshot.get("avg_lateness", 0.0),
            "state_active_ids": _json_dumps(state_snapshot.get("active_request_ids", [])),
            "state_waiting_ids": _json_dumps(state_snapshot.get("waiting_request_ids", [])),
            "state_completed_request_ids": _json_dumps(state_snapshot.get("completed_request_ids", [])),
            "state_dropped_request_ids": _json_dumps(state_snapshot.get("dropped_request_ids", [])),
            "state_stopped_decode_request_ids": _json_dumps(state_snapshot.get("stopped_decode_request_ids", [])),
            "state_pending_adv_tick": bool(state_snapshot.get("pending_adv_tick", False)),
            "state_last_adv_tick": state_snapshot.get("last_adv_tick", ""),
            "state_decode_credit_balance": state_snapshot.get(
                "decode_credit_balance",
                state_snapshot.get("decode_credit_available", 0),
            ),
            "state_decode_tokens_counted_by_id": _json_dumps(state_snapshot.get("decode_tokens_counted_by_id", {})),
            "state_violated_request_ids": _json_dumps(state_snapshot.get("violated_request_ids", [])),
            "state_per_request_prefill_lateness_by_id": _json_dumps(
                state_snapshot.get("per_request_prefill_lateness_by_id", {})
            ),
            "state_per_request_decode_lateness_by_id": _json_dumps(
                state_snapshot.get("per_request_decode_lateness_by_id", {})
            ),
            "adversary_requests": "[]",
            "adversary_prefill_slos": "[]",
            "adversary_prefill_deadlines_by_id": "{}",
            "adversary_decode_slos": "[]",
            "controller_token_budget": "",
            "controller_selected_ids": "[]",
            "controller_allocations": "{}",
            "controller_prefill_allocations": "{}",
            "controller_decode_allocations": "{}",
            "controller_prefill_total": 0,
            "controller_decode_total": 0,
            "controller_heuristic": "",
            "controller_strategy": "",
        }
    )
    if extra_payload:
        row.update(extra_payload)
    return row


def _adversary_action_index_for_single_launch(prefill_tokens: int) -> int:
    cfg = GameVersion2Config()
    stop_rules = list(cfg.adversary_action.stop_rule_names)
    templates = sorted(int(x) for x in cfg.request.allowed_prefill_tokens)
    stop_idx = stop_rules.index("stop_none") if "stop_none" in stop_rules else 0
    template_idx = templates.index(int(prefill_tokens))
    n_stop = len(stop_rules)
    return int(n_stop + template_idx * n_stop + stop_idx)


def _simple_synthetic_parent_trace_row(
    *,
    record: dict[str, Any],
    parent_state: Any,
    state_loader: Any,
) -> dict[str, Any] | None:
    """Create one adversary row for very simple one-request roots.

    This is only for smoke caches that lack stored `history_trace_logs`.
    """

    snapshot = record.get("simulator_snapshot") or {}
    request_states = snapshot.get("request_states") or {}
    if len(request_states) != 1:
        return None

    rid_raw, req = next(iter(request_states.items()))
    rid = int(rid_raw)
    if bool(req.get("completed", False)) or bool(req.get("preempted", False)):
        return None
    if int(req.get("num_processed_tokens", 0)) != 0:
        return None
    if int(req.get("remaining_prefill_tokens", -1)) != int(req.get("num_prefill_tokens", -2)):
        return None

    prefill_tokens = int(req["num_prefill_tokens"])
    decode_tokens = int(req["num_decode_tokens"])
    prefill_slo = float(req.get("prefill_slo_time", 0.0))
    decode_slo = float(req.get("decode_slo_time", 0.05))
    arrived_at = float(req.get("arrived_at", req.get("queued_at", 0.0)))
    deadline = float(req.get("queued_at", arrived_at)) + prefill_slo
    action_index = _adversary_action_index_for_single_launch(prefill_tokens)
    action_repr = (
        "AdversaryAction(requests=["
        "AdversaryRequestSpec("
        f"prefill_tokens={prefill_tokens}, "
        f"decode_tokens={decode_tokens}, "
        f"prefill_slo={prefill_slo}, "
        f"decode_slo={decode_slo}"
        ")], stop_decode_ids=[])"
    )

    state_snapshot = state_loader.env.describe_state(parent_state)
    parent_cost = float(state_loader.mcts._state_cost(parent_state))
    adv_request_payload = [
        {
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_slo": prefill_slo,
            "decode_slo": decode_slo,
        }
    ]

    return _logger_row(
        game_id=0,
        root_id=int(record.get("root_id", rid)),
        sim_iteration=-1,
        root_depth=0,
        root_node_id=0,
        root_player="adversary",
        phase="history-synthetic-parent",
        node_depth=1,
        parent_node_id=None,
        node_id=0,
        player_acted="adversary",
        player_to_act="controller",
        action_index=action_index,
        action_repr=action_repr,
        reward=0.0,
        objective_cost=parent_cost,
        state_snapshot=state_snapshot,
        num_valid_actions=0,
        unique_actions=0,
        decision_state_time=arrived_at,
        start_time=arrived_at,
        end_time=float(state_snapshot.get("sim_time", arrived_at)),
        extra_payload={
            "adversary_requests": _json_dumps(adv_request_payload),
            "adversary_prefill_slos": _json_dumps([prefill_slo]),
            "adversary_prefill_deadlines_by_id": _json_dumps({str(rid): deadline}),
            "adversary_decode_slos": _json_dumps([decode_slo]),
        },
    )


def _load_parent_trace_rows(
    *,
    record: dict[str, Any],
    parent_state: Any,
    state_loader: Any,
    allow_synthetic_parent_trace: bool,
) -> tuple[list[dict[str, Any]], str]:
    rows = [dict(row) for row in (record.get("history_trace_logs") or [])]
    if rows:
        return rows, "stored"

    if not allow_synthetic_parent_trace:
        return [], "missing"

    synthetic = _simple_synthetic_parent_trace_row(
        record=record,
        parent_state=parent_state,
        state_loader=state_loader,
    )
    if synthetic is None:
        return [], "missing_not_synthesizable"
    return [synthetic], "synthetic"


def _child_state_from_cached_row(state_loader: Any, child_row: dict[str, Any]) -> Any:
    clone_fn = getattr(state_loader.env, "clone_state_from_snapshot", None)
    if callable(clone_fn):
        return clone_fn(child_row["child_simulator_snapshot"], child_row["child_stats"])

    state = state_loader.env.initial_state()
    state.simulator.restore_state(child_row["child_simulator_snapshot"])
    state.stats = child_row["child_stats"].clone()
    return state


def _append_child_trace_row(
    *,
    trace_rows: list[dict[str, Any]],
    child_row: dict[str, Any],
    child_state: Any,
    state_loader: Any,
) -> dict[str, Any]:
    last = trace_rows[-1]
    parent_node_id = _safe_int(last.get("node_id"), 0)
    child_node_id = parent_node_id + 1
    parent_depth = _safe_int(last.get("node_depth"), 1)
    child_depth = parent_depth + 1
    parent_sim_time = _safe_float(last.get("sim_time"), 0.0)
    child_snapshot = state_loader.env.describe_state(child_state)
    child_time = _safe_float(child_snapshot.get("sim_time"), _safe_float(child_row.get("child_time"), parent_sim_time))
    action_repr = str(child_row.get("action_repr", ""))

    return _logger_row(
        game_id=_safe_int(last.get("game_id"), 0),
        root_id=_safe_int(last.get("root_id"), _safe_int(child_row.get("parent_root_id"), 0)),
        sim_iteration=_safe_int(last.get("sim_iteration"), -1) + 1,
        root_depth=_safe_int(last.get("root_depth"), child_depth - 1) + 1,
        root_node_id=_safe_int(last.get("root_node_id"), 0),
        root_player=str(last.get("root_player") or "adversary"),
        phase="history-child-transition",
        node_depth=child_depth,
        parent_node_id=parent_node_id,
        node_id=child_node_id,
        player_acted="controller",
        player_to_act="adversary",
        action_index=int(child_row["action_index"]),
        action_repr=action_repr,
        reward=float(child_row.get("reward", 0.0)),
        objective_cost=float(child_row.get("child_cost", 0.0)),
        state_snapshot=child_snapshot,
        num_valid_actions=_safe_int((child_row.get("row") or {}).get("num_valid_actions"), 0),
        unique_actions=_safe_int((child_row.get("row") or {}).get("num_canonical_actions"), 0),
        decision_state_time=parent_sim_time,
        start_time=parent_sim_time,
        end_time=child_time,
        stage_total_time=max(0.0, child_time - parent_sim_time),
        extra_payload=_controller_payload_from_repr(action_repr),
    )


def create_and_validate_child_trace_csvs(cfg: RootChildTraceTestingConfig) -> dict[str, Any]:
    """Create combined parent+child trace CSVs and validate each one."""

    _clean_output_dir(cfg)
    _write_json(cfg.output_dir / "root_child_trace_testing_config.json", asdict(cfg))

    child_rows = load_child_transitions(
        cfg.child_cache_dir,
        max_children=int(cfg.max_children),
        canonical_only=bool(cfg.canonical_only),
    )
    if not child_rows:
        raise RuntimeError(f"no child transitions found under {cfg.child_cache_dir}")

    parent_ids = {int(row["parent_state_id"]) for row in child_rows}
    parent_records = load_parent_records(cfg.dataset_dir, parent_ids)

    loader_cfg = RootChildGenerationConfig(
        dataset_dir=cfg.dataset_dir,
        output_dir=cfg.output_dir / "_loader_tmp",
        max_roots=max(parent_ids) + 1,
        root_player_filter="controller",
        overwrite=True,
        seed=int(cfg.seed),
    )
    state_loader = _make_state_loader(loader_cfg)

    rows_for_summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    passed = 0

    try:
        for child_row in child_rows:
            parent_state_id = int(child_row["parent_state_id"])
            action_index = int(child_row["action_index"])
            record = parent_records[parent_state_id]
            parent_state = state_loader(record)
            parent_trace_rows, trace_source = _load_parent_trace_rows(
                record=record,
                parent_state=parent_state,
                state_loader=state_loader,
                allow_synthetic_parent_trace=bool(cfg.allow_synthetic_parent_trace),
            )

            if not parent_trace_rows:
                skipped.append(
                    {
                        "parent_state_id": parent_state_id,
                        "parent_root_id": int(child_row.get("parent_root_id", -1)),
                        "action_index": action_index,
                        "reason": trace_source,
                    }
                )
                continue

            child_state = _child_state_from_cached_row(state_loader, child_row)
            appended_child = _append_child_trace_row(
                trace_rows=parent_trace_rows,
                child_row=child_row,
                child_state=child_state,
                state_loader=state_loader,
            )
            combined_rows = parent_trace_rows + [appended_child]

            trace_csv = (
                cfg.output_dir
                / f"root_child_trace_parent_{parent_state_id:06d}_action_{action_index:04d}.csv"
            )
            _write_trace_csv(trace_csv, combined_rows)

            try:
                trace_count, adv_actions_checked = _validate_trace_csv(trace_csv)
                passed += 1
                row = {
                    "parent_state_id": parent_state_id,
                    "parent_root_id": int(child_row.get("parent_root_id", -1)),
                    "action_index": action_index,
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
                failures.append(
                    {
                        "parent_state_id": parent_state_id,
                        "parent_root_id": int(child_row.get("parent_root_id", -1)),
                        "action_index": action_index,
                        "trace_source": trace_source,
                        "trace_csv": str(trace_csv),
                        "failed_trace_csv": str(failed_copy),
                        "error": repr(exc),
                    }
                )
                row = {
                    "parent_state_id": parent_state_id,
                    "parent_root_id": int(child_row.get("parent_root_id", -1)),
                    "action_index": action_index,
                    "trace_source": trace_source,
                    "trace_csv": str(trace_csv),
                    "trace_chains_checked": 0,
                    "adv_actions_checked": 0,
                    "passed": False,
                    "error": repr(exc),
                }

            rows_for_summary.append(row)
    finally:
        state_loader.close()

    summary = {
        "dataset_dir": str(cfg.dataset_dir),
        "child_cache_dir": str(cfg.child_cache_dir),
        "output_dir": str(cfg.output_dir),
        "children_loaded": int(len(child_rows)),
        "traces_written": int(len(rows_for_summary)),
        "traces_passed": int(passed),
        "failures": failures,
        "skipped": skipped,
        "passed": not failures and bool(rows_for_summary),
    }

    _write_json(cfg.output_dir / "trace_test_summary.json", summary)
    with (cfg.output_dir / "trace_test_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "parent_state_id",
            "parent_root_id",
            "action_index",
            "trace_source",
            "trace_csv",
            "trace_chains_checked",
            "adv_actions_checked",
            "passed",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_for_summary:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    if failures:
        raise RuntimeError(f"{len(failures)} child trace validation(s) failed; see {cfg.output_dir}")
    if not rows_for_summary:
        raise RuntimeError(
            "no child trace CSVs were validated; parent traces were missing or not synthesizable"
        )
    return summary


def parse_args() -> RootChildTraceTestingConfig:
    parser = argparse.ArgumentParser(description="Validate cached GV3 root-child transition traces.")
    parser.add_argument("--dataset-dir", default=str(_default_dataset_dir()))
    parser.add_argument("--child-cache-dir", default=str(_default_child_cache_dir()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--max-children", type=int, default=20)
    parser.add_argument("--include-alias-rows", action="store_true")
    parser.add_argument(
        "--require-stored-parent-trace",
        action="store_true",
        help="Do not synthesize simple parent traces when stored history_trace_logs are missing.",
    )
    parser.add_argument("--reuse-existing-output", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if int(args.max_children) <= 0:
        raise ValueError("--max-children must be > 0")

    return RootChildTraceTestingConfig(
        dataset_dir=Path(args.dataset_dir).expanduser(),
        child_cache_dir=Path(args.child_cache_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        max_children=int(args.max_children),
        canonical_only=not bool(args.include_alias_rows),
        allow_synthetic_parent_trace=not bool(args.require_stored_parent_trace),
        overwrite=not bool(args.reuse_existing_output),
        seed=int(args.seed),
    )


def main() -> None:
    cfg = parse_args()
    summary = create_and_validate_child_trace_csvs(cfg)
    print(
        "root child trace tests passed: "
        f"traces={summary['traces_passed']}/{summary['traces_written']}, "
        f"skipped={len(summary['skipped'])}, "
        f"output_dir={cfg.output_dir}"
    )


if __name__ == "__main__":
    main()
