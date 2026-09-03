from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from common import import_native_cpp, make_args


NUMERIC_FIELDS = {
    "prior",
    "reward",
    "nn_value_controller",
    "objective_cost",
    "sim_time",
    "decision_state_time",
    "start_time",
    "end_time",
    "stage_total_time",
    "total_lateness",
    "avg_lateness",
}

INTEGER_FIELDS = {
    "game_id",
    "root_id",
    "sim_iteration",
    "root_depth",
    "root_node_id",
    "node_depth",
    "parent_node_id",
    "node_id",
    "action_index",
    "num_valid_actions",
    "unique_actions",
    "requests_in_system",
    "requests_generated",
    "requests_completed",
    "slo_violations",
    "state_decode_credit_balance",
    "controller_token_budget",
    "controller_prefill_total",
    "controller_decode_total",
}

JSON_FIELDS = {
    "model_prior_json",
    "normalized_prior_json",
    "model_top5_actions_json",
    "mcts_top5_actions_json",
    "state_active_ids",
    "state_waiting_ids",
    "state_completed_request_ids",
    "state_dropped_request_ids",
    "state_stopped_decode_request_ids",
    "state_decode_tokens_counted_by_id",
    "state_violated_request_ids",
    "state_per_request_prefill_lateness_by_id",
    "state_per_request_decode_lateness_by_id",
    "adversary_requests",
    "adversary_prefill_slos",
    "adversary_prefill_deadlines_by_id",
    "adversary_decode_slos",
    "controller_selected_ids",
    "controller_allocations",
    "controller_prefill_allocations",
    "controller_decode_allocations",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate Python and native GV3 history traces for the same random roots, "
            "validate both with game-engine tests, and compare the CSV rows."
        )
    )
    p.add_argument("--num-roots", type=int, default=1)
    p.add_argument("--history-hops", type=int, default=1000)
    p.add_argument("--history-seed", type=int, default=202612)
    p.add_argument("--output-name", type=str, default="")
    p.add_argument("--feature-tolerance", type=float, default=1e-6)
    p.add_argument("--float-tolerance", type=float, default=1e-9)
    p.add_argument("--max-mismatches", type=int, default=100)
    return p.parse_args(argv)


def _make_exact_hop_cfg(args: argparse.Namespace, *, environment_lang: str) -> Any:
    from vidur.Game_Version3.config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG

    cfg = replace(
        DEFAULT_MULTIPROCESS_TRAINING_CONFIG,
        environment_lang=str(environment_lang),
        use_virtual_env=True,
        model=replace(DEFAULT_MULTIPROCESS_TRAINING_CONFIG.model, device="cpu"),
        history_hops_min=int(args.history_hops),
        history_hops_max=int(args.history_hops),
        # Exact non-zero hop tests intentionally do not include hop 0.
        history_hops_force_zero=False,
    )
    cfg.validate()
    return cfg


def _clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "python_history_mcts_iter.csv",
        "python_history_mcts_iter_internal.csv",
        "native_history_mcts_iter.csv",
        "native_history_mcts_iter_internal.csv",
        "native_frontiers.csv",
        "native_depth1_search.csv",
        "native_depth1_details.csv",
        "native_frontier_feature.csv",
        "native_frontier_feature_compare.csv",
        "trace_alignment_mismatches.csv",
        "history_node_tests_failed_trace.csv",
    ):
        p = out_dir / name
        if p.exists():
            p.unlink()


def _generate_python_trace(args: argparse.Namespace, cfg: Any, out_dir: Path) -> tuple[Any, list[Any], Path]:
    from vidur.Game_Version3.logger.mctsDNN_logger import DNNMCTSIterationLogger
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    nlt._set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )
    simulator, env, _constraints, explore_cfg = nlt._build_env_and_simulator(cfg, use_virtual_env=True)
    runner = nlt._make_runner(
        cfg,
        env=env,
        explore_cfg=explore_cfg,
        model=None,
        history_seed=int(args.history_seed),
        out_dir=out_dir,
    )

    trace_csv = out_dir / "python_history_mcts_iter.csv"
    trace_logger = DNNMCTSIterationLogger(trace_csv, flush_every=1)
    runner.history.iter_logger = trace_logger

    roots: list[Any] = []
    for batch in runner.history.generate_roots_batch_iter(
        initial_state=None,
        start_player="adversary",
        start_depth=0,
        num_roots=int(args.num_roots),
        game_id=0,
        start_root_id=0,
        nontrivial_hops=int(args.history_hops),
        min_history_hops=int(args.history_hops),
        max_history_hops=int(args.history_hops),
        seed=int(args.history_seed),
        max_total_steps=int(cfg.history_max_total_steps),
        log_history=True,
        batch_size=int(cfg.history_root_batch_size),
        allow_duplicate_fallback=True,
        history_trace_logger=trace_logger,
    ):
        roots.extend(batch)

    trace_logger.close()
    return simulator, roots, trace_csv


def _generate_native_trace(native: Any, simulator: Any, args: argparse.Namespace, cfg: Any, out_dir: Path) -> dict[str, Any]:
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    trace_csv = out_dir / "native_history_mcts_iter.csv"
    frontier_csv = out_dir / "native_frontiers.csv"
    depth1_search_csv = out_dir / "native_depth1_search.csv"
    depth1_details_csv = out_dir / "native_depth1_details.csv"

    payload = nlt._cfg_payload(cfg, torchscript_model_spec="")
    nlt.attach_execution_predictor_payload(payload, simulator)
    payload.update(
        {
            "native_history_trace_log_path": str(trace_csv),
            "native_frontier_log_path": str(frontier_csv),
            "native_depth1_search_log_path": str(depth1_search_csv),
            "native_depth1_details_log_path": str(depth1_details_csv),
            "use_model_bootstrap": False,
        }
    )

    runtime = native.NativeTorchScriptInferRuntimeGV2("cpu")
    return dict(
        native.generate_selfplay_samples_torchscript(
            runtime,
            0,
            nlt._empty_initial_state_payload(),
            payload,
            0,
            int(args.num_roots),
            0,
            0,
            "adversary",
            int(cfg.run.feature_version),
            int(cfg.adv_iterations_per_root),
            int(cfg.cont_iterations_per_root),
            int(args.history_hops),
            int(args.history_hops),
            int(args.history_seed),
            int(cfg.history_max_total_steps),
            int(cfg.max_forced_hops_per_root),
            0.0,
            int(cfg.eval_split_seed_base),
            int(cfg.action_seed_base),
            True,
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_json_maybe(cell: str) -> Any:
    s = (cell or "").strip()
    if s == "":
        return ""
    return json.loads(s)


def _normalize_json_value(value: Any, *, float_tolerance: float) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v, float_tolerance=float_tolerance) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_json_value(v, float_tolerance=float_tolerance) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        digits = 12 if float_tolerance <= 0 else max(0, int(abs(math.log10(float_tolerance))) + 2)
        return round(value, digits)
    return value


def _cells_equal(field: str, py: str, nt: str, *, float_tolerance: float) -> tuple[bool, str, str]:
    py_s = "" if py is None else str(py).strip()
    nt_s = "" if nt is None else str(nt).strip()

    if field in INTEGER_FIELDS:
        if py_s == "" and nt_s == "":
            return True, py_s, nt_s
        return int(float(py_s or 0)) == int(float(nt_s or 0)), py_s, nt_s

    if field in NUMERIC_FIELDS:
        if py_s == "" and nt_s == "":
            return True, py_s, nt_s
        py_f = float(py_s or 0.0)
        nt_f = float(nt_s or 0.0)
        return abs(py_f - nt_f) <= float(float_tolerance), f"{py_f:.17g}", f"{nt_f:.17g}"

    if field in JSON_FIELDS:
        try:
            py_obj = _normalize_json_value(_load_json_maybe(py_s), float_tolerance=float_tolerance)
            nt_obj = _normalize_json_value(_load_json_maybe(nt_s), float_tolerance=float_tolerance)
            return py_obj == nt_obj, json.dumps(py_obj, sort_keys=True), json.dumps(nt_obj, sort_keys=True)
        except Exception:
            return py_s == nt_s, py_s, nt_s

    return py_s == nt_s, py_s, nt_s


def _compare_trace_csvs(
    python_csv: Path,
    native_csv: Path,
    *,
    out_dir: Path,
    float_tolerance: float,
    max_mismatches: int,
) -> dict[str, Any]:
    py_rows = _read_csv(python_csv)
    nt_rows = _read_csv(native_csv)

    mismatch_rows: list[dict[str, Any]] = []
    if len(py_rows) != len(nt_rows):
        mismatch_rows.append(
            {
                "row_index": "",
                "field": "__row_count__",
                "python_value": len(py_rows),
                "native_value": len(nt_rows),
            }
        )

    py_fields = list(py_rows[0].keys()) if py_rows else []
    nt_fields = list(nt_rows[0].keys()) if nt_rows else []
    common_fields = [f for f in py_fields if f in nt_fields]
    if py_fields != nt_fields:
        mismatch_rows.append(
            {
                "row_index": "",
                "field": "__fieldnames__",
                "python_value": json.dumps(py_fields),
                "native_value": json.dumps(nt_fields),
            }
        )

    for idx, (py_row, nt_row) in enumerate(zip(py_rows, nt_rows)):
        for field in common_fields:
            ok, py_norm, nt_norm = _cells_equal(
                field,
                py_row.get(field, ""),
                nt_row.get(field, ""),
                float_tolerance=float_tolerance,
            )
            if ok:
                continue
            mismatch_rows.append(
                {
                    "row_index": idx,
                    "field": field,
                    "python_value": py_norm,
                    "native_value": nt_norm,
                }
            )
            if len(mismatch_rows) >= int(max_mismatches):
                break
        if len(mismatch_rows) >= int(max_mismatches):
            break

    mismatch_csv = out_dir / "trace_alignment_mismatches.csv"
    with mismatch_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "field", "python_value", "native_value"])
        writer.writeheader()
        writer.writerows(mismatch_rows)

    if mismatch_rows:
        first = mismatch_rows[0]
        raise AssertionError(
            f"python/native trace CSV mismatch: mismatches={len(mismatch_rows)} "
            f"first={first} wrote {mismatch_csv}"
        )

    return {
        "rows": len(py_rows),
        "fields": len(common_fields),
        "mismatch_csv": str(mismatch_csv),
    }


def run_trace_alignment(args: argparse.Namespace) -> dict[str, Any]:
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    output_name = args.output_name or f"trace_alignment_hops{int(args.history_hops)}_roots{int(args.num_roots)}"
    common_args = make_args(
        output_name,
        num_roots=int(args.num_roots),
        history_hops_min=int(args.history_hops),
        history_hops_max=int(args.history_hops),
        history_seed=int(args.history_seed),
        frontier_parity_roots=int(args.num_roots),
        feature_tolerance=float(args.feature_tolerance),
    )
    common_args.history_hops = int(args.history_hops)
    common_args.float_tolerance = float(args.float_tolerance)
    common_args.max_mismatches = int(args.max_mismatches)

    out_dir = Path(common_args.output_dir)
    _clean_output_dir(out_dir)

    native = import_native_cpp()
    cfg_python = _make_exact_hop_cfg(common_args, environment_lang="python")
    simulator, python_roots, python_csv = _generate_python_trace(common_args, cfg_python, out_dir)

    cfg_native = _make_exact_hop_cfg(common_args, environment_lang="native")
    native_out = _generate_native_trace(native, simulator, common_args, cfg_native, out_dir)
    native_csv = out_dir / "native_history_mcts_iter.csv"

    python_trace_count, python_adv_actions = nlt._validate_trace_csv(python_csv)
    native_trace_count, native_adv_actions = nlt._validate_trace_csv(native_csv)
    comparison = _compare_trace_csvs(
        python_csv,
        native_csv,
        out_dir=out_dir,
        float_tolerance=float(common_args.float_tolerance),
        max_mismatches=int(common_args.max_mismatches),
    )

    return {
        "output_dir": str(out_dir),
        "python_csv": str(python_csv),
        "native_csv": str(native_csv),
        "python_roots": len(python_roots),
        "native_samples": len(list(native_out.get("samples", []) or [])),
        "python_trace_count": python_trace_count,
        "native_trace_count": native_trace_count,
        "python_adv_actions_checked": python_adv_actions,
        "native_adv_actions_checked": native_adv_actions,
        **comparison,
    }


def test_native_trace_csv_alignment() -> None:
    stats = run_trace_alignment(_parse_args([]))
    assert int(stats["python_trace_count"]) > 0
    assert int(stats["native_trace_count"]) > 0
    assert int(stats["rows"]) > 0


if __name__ == "__main__":
    stats = run_trace_alignment(_parse_args())
    print(json.dumps(stats, indent=2, sort_keys=True))
    print("trace_allignment_test passed")
