from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...tests import game_engine_tests as gv2_trace_tests
from ..DNN.history_root import HistoryRootGenerator
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, GameVersion2Config
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger
from ..multiProcessUtils import _build_env_and_simulator, _set_global_seeds


FRONTIER_FIELDS = [
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "history_log_node_id",
    "history_signature_json",
    "strict_signature_json",
    "duplicate_history_signature",
    "duplicate_strict_signature",
    "valid_action_count",
    "frontier_kind",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "tests"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _patch_gv2_validator_for_gv3() -> None:
    cfg = GameVersion2Config()
    gv2_trace_tests._GV2_CFG = cfg
    gv2_trace_tests.INTERVAL_EPS = float(cfg.timing.eps)
    gv2_trace_tests._ALLOWED_PREFILL_TOKENS = set(int(x) for x in cfg.request.allowed_prefill_tokens)
    gv2_trace_tests._ALLOWED_PREFILL_TOKENS_SORTED = sorted(gv2_trace_tests._ALLOWED_PREFILL_TOKENS)
    gv2_trace_tests._GV2_TICK_SEC = float(cfg.timing.adversary_tick_sec)
    gv2_trace_tests._GV2_DEADLINE_TOL = max(1e-6, 10.0 * gv2_trace_tests.INTERVAL_EPS)
    gv2_trace_tests._GV2_OBJECTIVE_TOL = max(1e-4, 20.0 * gv2_trace_tests.INTERVAL_EPS)
    gv2_trace_tests._GV2_DECODE_CAP = int(cfg.request.max_decode_tokens_per_request)
    gv2_trace_tests._GV2_DECODE_MINT = int(cfg.credits.decode_credit_mint_per_prefill_complete)
    gv2_trace_tests._GV2_CREDIT_TOL = 0


def _strict_signature(env: Any, record: dict[str, Any]) -> tuple[Any, ...]:
    desc = env.describe_state(record["root_state"])
    return (
        str(record["root_player"]),
        int(record["root_depth"]),
        int(record.get("history_hops", 0)),
        round(float(desc.get("sim_time", 0.0)), 9),
        bool(desc.get("pending_adv_tick", False)),
        "" if desc.get("last_adv_tick", "") in (None, "") else round(float(desc.get("last_adv_tick", 0.0)), 9),
        int(desc.get("decode_credit_balance", 0)),
        tuple(int(x) for x in (desc.get("active_request_ids") or [])),
        tuple(int(x) for x in (desc.get("waiting_request_ids") or [])),
        tuple(int(x) for x in (desc.get("completed_request_ids") or [])),
        tuple(int(x) for x in (desc.get("dropped_request_ids") or [])),
        tuple(int(x) for x in (desc.get("stopped_decode_request_ids") or [])),
        tuple(sorted((int(k), int(v)) for k, v in (desc.get("decode_tokens_counted_by_id") or {}).items())),
        tuple(sorted((int(k), round(float(v), 9)) for k, v in (desc.get("per_request_prefill_lateness_by_id") or {}).items())),
        tuple(sorted((int(k), round(float(v), 9)) for k, v in (desc.get("per_request_decode_lateness_by_id") or {}).items())),
    )


def _valid_count_for_frontier(env: Any, record: dict[str, Any]) -> int:
    state = record["root_state"].fork(flag=False)
    player = str(record["root_player"])
    if player == "controller":
        actions_by_index, mask = env.sample_controller_actions(state)
    else:
        actions_by_index, mask = env.sample_adversary_actions(state)
    mask_list = [bool(x) for x in mask]
    return sum(1 for i, ok in enumerate(mask_list) if bool(ok) and actions_by_index[int(i)] is not None)


def _write_frontier_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRONTIER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FRONTIER_FIELDS})


def _validate_trace_csv(trace_csv: Path) -> tuple[int, int]:
    _patch_gv2_validator_for_gv3()
    rows, fieldnames, raw_by_rownum = gv2_trace_tests.load_rows(str(trace_csv))
    traces = gv2_trace_tests.build_leaf_traces(rows)
    if not traces:
        raise RuntimeError(f"no trace chains built from {trace_csv}")

    prefill_profile: dict[int, float] = {}
    adv_actions_checked = 0
    for trace in traces:
        try:
            adv_actions_checked += gv2_trace_tests.run_trace(
                trace,
                prefill_profile=prefill_profile,
                assume_extracted_trace=False,
                unsupported_hits=set(),
            )
        except gv2_trace_tests.TestFailure:
            failed_out = trace_csv.parent / "history_node_tests_failed_trace.csv"
            gv2_trace_tests._write_failed_trace_csv(
                trace,
                fieldnames=fieldnames,
                raw_by_rownum=raw_by_rownum,
                out_path=failed_out,
            )
            raise

    return len(traces), int(adv_actions_checked)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GV3 optimized history frontier traces with GV2-style semantic tests."
    )
    parser.add_argument("--num-roots", type=int, default=32)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=32)
    parser.add_argument("--history-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-total-steps", type=int, default=20000)
    parser.add_argument("--max-children-per-expand", type=int, default=0)
    parser.add_argument("--allow-duplicate-fallback", action="store_true", default=False)
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    trace_csv = output_dir / "history_node_tests_mcts_iter.csv"
    frontier_csv = output_dir / "history_node_tests_frontiers.csv"

    cfg = replace(
        DEFAULT_MULTIPROCESS_TRAINING_CONFIG,
        environment_lang="python",
        use_virtual_env=True,
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
    )
    cfg.validate()
    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )

    _simulator, env, _constraints, _explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=True)
    trace_logger = DNNMCTSIterationLogger(trace_csv, flush_every=1)
    history = HistoryRootGenerator(env=env)

    try:
        records: list[dict[str, Any]] = []
        for batch in history.generate_roots_batch_iter(
            initial_state=None,
            start_player="adversary",
            start_depth=0,
            num_roots=int(args.num_roots),
            game_id=0,
            start_root_id=0,
            nontrivial_hops=int(args.history_hops_min),
            min_history_hops=int(args.history_hops_min),
            max_history_hops=int(args.history_hops_max),
            seed=int(args.history_seed),
            max_total_steps=int(args.max_total_steps),
            batch_size=int(args.batch_size),
            max_children_per_expand=(
                None if int(args.max_children_per_expand) <= 0 else int(args.max_children_per_expand)
            ),
            allow_duplicate_fallback=bool(args.allow_duplicate_fallback),
            history_trace_logger=trace_logger,
        ):
            records.extend(batch)
    finally:
        trace_logger.close()

    if len(records) != int(args.num_roots):
        raise RuntimeError(f"expected {int(args.num_roots)} frontier roots, generated {len(records)}")

    seen_history: set[str] = set()
    seen_strict: set[str] = set()
    frontier_rows: list[dict[str, Any]] = []
    duplicate_history: list[int] = []
    duplicate_strict: list[int] = []
    for record in records:
        root_id = int(record["root_id"])
        history_sig = record.get("history_signature")
        strict_sig = _strict_signature(env, record)
        history_key = _json_dumps(history_sig)
        strict_key = _json_dumps(strict_sig)
        hist_dup = history_key in seen_history
        strict_dup = strict_key in seen_strict
        if hist_dup:
            duplicate_history.append(root_id)
        if strict_dup:
            duplicate_strict.append(root_id)
        seen_history.add(history_key)
        seen_strict.add(strict_key)

        valid_count = _valid_count_for_frontier(env, record)
        frontier_rows.append(
            {
                "root_id": root_id,
                "root_player": str(record["root_player"]),
                "root_depth": int(record["root_depth"]),
                "history_hops": int(record.get("history_hops", 0)),
                "history_log_node_id": "" if record.get("history_log_node_id") is None else int(record["history_log_node_id"]),
                "history_signature_json": history_key,
                "strict_signature_json": strict_key,
                "duplicate_history_signature": str(bool(hist_dup)).lower(),
                "duplicate_strict_signature": str(bool(strict_dup)).lower(),
                "valid_action_count": int(valid_count),
                "frontier_kind": "terminal" if valid_count == 0 else ("forced" if valid_count == 1 else "branching"),
            }
        )

    _write_frontier_csv(frontier_csv, frontier_rows)

    if duplicate_history or duplicate_strict:
        raise RuntimeError(
            "frontier uniqueness failed: "
            f"history_duplicates={duplicate_history[:20]}, strict_duplicates={duplicate_strict[:20]} "
            f"(wrote {frontier_csv})"
        )

    forced_frontiers = [row["root_id"] for row in frontier_rows if row["frontier_kind"] == "forced"]
    if forced_frontiers:
        raise RuntimeError(
            f"frontier correctness failed: roots ended on forced single-action nodes {forced_frontiers[:20]}"
        )

    traces_checked, adv_actions_checked = _validate_trace_csv(trace_csv)
    print(
        "history node tests passed: "
        f"frontiers={len(records)}, traces={traces_checked}, adv_actions={adv_actions_checked}, "
        f"trace_csv={trace_csv}, frontier_csv={frontier_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
