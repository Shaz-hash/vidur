### (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

from ..sampler.storage import read_parquet_records
from .assembler import LPAssemblerConfig, _build_request_aggregates, assemble_lp_problem
from .features import extract_features_for_set
from .solver import solve_lp_problem


ITER_FIELDS: Sequence[str] = (
    "game_id",
    "root_id",
    "root_node_id",
    "root_player",
    "node_id",
    "parent_node_id",
    "node_depth",
    "sim_time",
    "advanced_sim_time",
    "player_acted_to_create_this_node",
    "player_to_act_in_this_node",
    "action_index",
    "action_repr",
    "phase",
    "nn_called",
    "model_prior_json",
    "normalized_prior_json",
    "state_waiting_ids",
    "state_completed_request_ids",
    "adversary_prefill_deadlines_by_id",
    "slo_violations",
    "avg_lateness",
    "objective_cost",
    "state_position",
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Assemble and solve linear LP from sampler merged tables")
    ap.add_argument("--sampler-round-dir", type=str, required=True, help="Path to sampler merged dir (round_XXX/merged)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for LP artifacts")
    ap.add_argument("--discount-factor", type=float, default=0.98)
    ap.add_argument("--weight-bound-abs", type=float, default=1000.0)
    ap.add_argument("--structural-margin-eps", type=float, default=1e-3)
    ap.add_argument("--feature-set", type=str, default="baseline_v1")
    ap.add_argument(
        "--objective-mode",
        type=str,
        default="minimax_directional_slack_eliminated",
        choices=["minimax_directional_slack_eliminated", "avg_anchor_value_gap"],
    )
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--violation-tol", type=float, default=1e-8)
    ap.add_argument(
        "--skip-default-evaluator",
        action="store_true",
        help="Write LP artifacts only and skip the post-solve evaluator rollout export.",
    )
    return ap


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _find_first_row_by_key(path: Path, *, key: str, value: str) -> dict | None:
    needle = str(value)
    if not needle or (not path.exists()):
        return None

    if pq is None:
        try:
            for row in read_parquet_records(path):
                if str(row.get(key, "")) == needle:
                    return dict(row)
            return None
        except Exception as exc:
            print(
                f"[linear.lp] warning: could not read optional parquet file {path}: {exc!r}",
                flush=True,
            )
            return None

    try:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches():
            obj = batch.to_pydict()
            keys = obj.get(key, [])
            n = len(keys)
            for i in range(n):
                if str(keys[i]) == needle:
                    return {k: obj[k][i] for k in obj.keys()}
    except Exception as exc:
        print(
            f"[linear.lp] warning: could not read optional parquet file {path}: {exc!r}",
            flush=True,
        )
        return None
    return None


def _pick_random_lp_state(problem, *, seed: int) -> tuple[object, str, str, str]:
    rows = list(problem.constraint_rows)
    if not rows:
        raise RuntimeError("no LP constraint rows found for random-state export")
    rng = random.Random(int(seed))
    row = rows[rng.randrange(len(rows))]
    choose_canonical = bool(rng.getrandbits(1))
    if choose_canonical and str(row.next_state_id):
        return row, str(row.next_state_id), "canonical", str(row.state_id)
    return row, str(row.state_id), "anchor", str(row.state_id)


def _build_mcts_iter_random_state_row(
    *,
    merged_dir: Path,
    constraint_row: object,
    sampled_state_id: str,
    state_position: str,
    root_state_id: str,
) -> dict:
    states_path = merged_dir / "states.parquet"
    nodes_path = merged_dir / "nodes.parquet"
    transitions_path = merged_dir / "transitions.parquet"
    actions_path = merged_dir / "actions.parquet"

    sampled_state = _find_first_row_by_key(states_path, key="state_id", value=sampled_state_id) or {}
    root_state = _find_first_row_by_key(states_path, key="state_id", value=root_state_id) or {}
    sampled_node = _find_first_row_by_key(nodes_path, key="state_id", value=sampled_state_id) or {}
    root_node = _find_first_row_by_key(nodes_path, key="state_id", value=root_state_id) or {}
    transition = _find_first_row_by_key(
        transitions_path,
        key="transition_id",
        value=str(getattr(constraint_row, "transition_id", "")),
    ) or {}
    action = _find_first_row_by_key(
        actions_path,
        key="action_id",
        value=str(transition.get("action_id", "")),
    ) or {}

    root_node_id = str(root_node.get("node_id", root_state_id))
    node_id = str(sampled_node.get("node_id", sampled_state_id))
    root_player = str(root_state.get("player_to_act", sampled_state.get("player_to_act", "")))

    if state_position == "canonical" and transition:
        sim_time = _to_float(transition.get("sim_time_after_action", sampled_state.get("sim_time", 0.0)))
        adv_time = _to_float(transition.get("sim_time_after_advance", sampled_state.get("sim_time", 0.0)))
        parent_node_id = str(root_node.get("node_id", root_state_id))
        player_acted = str(action.get("actor", transition.get("actor", "")))
        action_index: object = _to_int(action.get("canonical_index", -1))
        action_repr = str(action.get("action_repr", ""))
        adv_deadlines = str(transition.get("adversary_prefill_deadlines_by_id_json", "{}"))
    else:
        sim_time = _to_float(sampled_state.get("sim_time", 0.0))
        adv_time = sim_time
        parent_node_id = ""
        player_acted = ""
        action_index = ""
        action_repr = ""
        adv_deadlines = "{}"

    return {
        "game_id": _to_int(root_node.get("game_id", 0)),
        "root_id": _to_int(root_node.get("root_id", 0)),
        "root_node_id": root_node_id,
        "root_player": root_player,
        "node_id": node_id,
        "parent_node_id": parent_node_id,
        "node_depth": _to_int(sampled_state.get("branching_depth", 0)),
        "sim_time": sim_time,
        "advanced_sim_time": adv_time,
        "player_acted_to_create_this_node": player_acted,
        "player_to_act_in_this_node": str(sampled_state.get("player_to_act", "")),
        "action_index": action_index,
        "action_repr": action_repr,
        "phase": state_position,
        "nn_called": False,
        "model_prior_json": "[]",
        "normalized_prior_json": "[]",
        "state_waiting_ids": str(sampled_state.get("waiting_request_ids_json", "[]")),
        "state_completed_request_ids": str(sampled_state.get("completed_request_ids_json", "[]")),
        "adversary_prefill_deadlines_by_id": adv_deadlines,
        "slo_violations": _to_int(sampled_state.get("slo_violations", 0)),
        "avg_lateness": _to_float(sampled_state.get("total_lateness", 0.0)),
        "objective_cost": _to_float(sampled_state.get("total_cost", 0.0)),
        "state_position": state_position,
    }


def _write_single_row_csv(path: Path, fieldnames: Sequence[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _build_normalized_feature_row(
    *,
    merged_dir: Path,
    sampled_state_id: str,
    feature_set: str,
    feature_names: Sequence[str],
) -> dict:
    states_path = merged_dir / "states.parquet"
    requests_path = merged_dir / "requests.parquet"
    state_row = _find_first_row_by_key(states_path, key="state_id", value=sampled_state_id)
    if state_row is None:
        raise RuntimeError(f"sampled state not found in states.parquet: {sampled_state_id}")
    sim_time = _to_float(state_row.get("sim_time", 0.0))
    req_aggs = _build_request_aggregates(
        requests_path=requests_path,
        required_state_ids={str(sampled_state_id)},
        state_sim_time={str(sampled_state_id): sim_time},
    )
    agg = req_aggs.get(str(sampled_state_id))
    fmap = extract_features_for_set(
        feature_set=str(feature_set),
        state_row=state_row,
        req_agg=agg,
    )
    return {name: float(fmap.get(name, 0.0)) for name in feature_names}


def _write_weights_by_feature_json(path: Path, feature_names: Sequence[str], weights: np.ndarray) -> None:
    payload = {
        "num_features": int(len(feature_names)),
        "weights_by_feature": {
            str(name): float(weights[i]) for i, name in enumerate(feature_names)
        },
    }
    _write_json(path, payload)


def _run_default_evaluator(out_dir: Path) -> None:
    from ..evaluator.run import main as evaluator_main

    evaluator_main(
        [
            "--lp-solution-dir",
            str(out_dir),
            "--out-csv",
            str(out_dir / "linear_eval_game_h0.csv"),
            "--history-hop",
            "0",
            "--discount-factor",
            "0.98",
            "--max-steps",
            "1024",
            "--seed",
            "12345",
            "--root-player",
            "adversary",
            "--use-virtual-env",
        ]
    )


def _write_constraint_violations_csv(
    out_csv: Path,
    sample_errors: np.ndarray,
    nonneg_row_violations: np.ndarray,
    cap_row_violations: np.ndarray,
    ineq_violations_all: np.ndarray,
    players: list[str],
    tol: float,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    by_player_err = {"controller": [], "adversary": []}
    by_player_nonneg = {"controller": [], "adversary": []}
    by_player_cap = {"controller": [], "adversary": []}
    for i, p in enumerate(players):
        if p in by_player_err:
            by_player_err[p].append(float(sample_errors[i]))
            by_player_nonneg[p].append(float(nonneg_row_violations[i]))
            by_player_cap[p].append(float(cap_row_violations[i]))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "player",
                "num_samples",
                "max_error",
                "mean_error",
                "p95_error",
                "max_nonneg_violation",
                "mean_nonneg_violation",
                "p95_nonneg_violation",
                "max_cap_violation",
                "mean_cap_violation",
                "p95_cap_violation",
                "num_ineq_violated_gt_tol_all_rows",
                "frac_ineq_violated_gt_tol_all_rows",
                "max_ineq_violation_all_rows",
                "tolerance",
            ],
        )
        w.writeheader()
        all_ineq_viol = np.array(ineq_violations_all, dtype=np.float64)
        num_ineq_viol = int(np.sum(all_ineq_viol > tol)) if all_ineq_viol.size else 0
        frac_ineq_viol = float(num_ineq_viol / max(1, all_ineq_viol.size)) if all_ineq_viol.size else 0.0
        max_ineq_viol = float(np.max(all_ineq_viol)) if all_ineq_viol.size else 0.0

        for player in ("controller", "adversary"):
            err_vals = np.array(by_player_err[player], dtype=np.float64)
            nonneg_vals = np.array(by_player_nonneg[player], dtype=np.float64)
            cap_vals = np.array(by_player_cap[player], dtype=np.float64)
            if err_vals.size == 0:
                row = {
                    "player": player,
                    "num_samples": 0,
                    "max_error": 0.0,
                    "mean_error": 0.0,
                    "p95_error": 0.0,
                    "max_nonneg_violation": 0.0,
                    "mean_nonneg_violation": 0.0,
                    "p95_nonneg_violation": 0.0,
                    "max_cap_violation": 0.0,
                    "mean_cap_violation": 0.0,
                    "p95_cap_violation": 0.0,
                    "num_ineq_violated_gt_tol_all_rows": int(num_ineq_viol),
                    "frac_ineq_violated_gt_tol_all_rows": float(frac_ineq_viol),
                    "max_ineq_violation_all_rows": float(max_ineq_viol),
                    "tolerance": float(tol),
                }
            else:
                row = {
                    "player": player,
                    "num_samples": int(err_vals.size),
                    "max_error": float(np.max(err_vals)),
                    "mean_error": float(np.mean(err_vals)),
                    "p95_error": float(np.percentile(err_vals, 95)),
                    "max_nonneg_violation": float(np.max(nonneg_vals)),
                    "mean_nonneg_violation": float(np.mean(nonneg_vals)),
                    "p95_nonneg_violation": float(np.percentile(nonneg_vals, 95)),
                    "max_cap_violation": float(np.max(cap_vals)),
                    "mean_cap_violation": float(np.mean(cap_vals)),
                    "p95_cap_violation": float(np.percentile(cap_vals, 95)),
                    "num_ineq_violated_gt_tol_all_rows": int(num_ineq_viol),
                    "frac_ineq_violated_gt_tol_all_rows": float(frac_ineq_viol),
                    "max_ineq_violation_all_rows": float(max_ineq_viol),
                    "tolerance": float(tol),
                }
            w.writerow(row)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LPAssemblerConfig(
        sampler_round_dir=str(args.sampler_round_dir),
        discount_factor=float(args.discount_factor),
        weight_bound_abs=float(args.weight_bound_abs),
        structural_margin_eps=float(args.structural_margin_eps),
        feature_set=str(args.feature_set),
        objective_mode=str(args.objective_mode),
        seed=int(args.seed),
    )
    problem = assemble_lp_problem(cfg)
    solved = solve_lp_problem(problem)

    A = problem.A_ub
    b = problem.b_ub
    x_full = np.asarray(solved.full_solution, dtype=np.float64).reshape(-1)
    lhs = np.asarray(A @ x_full, dtype=np.float64).reshape(-1)
    residual = lhs - b
    violations = np.maximum(0.0, residual)
    players = [r.player for r in problem.constraint_rows]
    m = len(problem.constraint_rows)
    rows_per_sample = max(1, int(problem.rows_per_sample))
    if m > 0:
        first_row_residual = residual[0::rows_per_sample][:m]
        derived_sample_errors = np.maximum(0.0, -first_row_residual)
        nonneg_row_violations = np.maximum(0.0, first_row_residual)
        if bool(problem.has_cap_rows):
            cap_row_residual = residual[1::rows_per_sample][:m]
            cap_row_violations = np.maximum(0.0, cap_row_residual)
        else:
            cap_row_violations = np.zeros((m,), dtype=np.float64)
    else:
        derived_sample_errors = np.zeros((0,), dtype=np.float64)
        nonneg_row_violations = np.zeros((0,), dtype=np.float64)
        cap_row_violations = np.zeros((0,), dtype=np.float64)
    sample_errors = np.asarray(solved.sample_errors, dtype=np.float64).reshape(-1)
    if sample_errors.shape[0] != m:
        raise RuntimeError(
            f"sample error length mismatch: errors={sample_errors.shape[0]} constraints={m}"
        )

    dataset_summary = dict(problem.dataset_summary)
    dataset_summary.update(
        {
            "solver_status": int(solved.status),
            "solver_success": bool(solved.success),
            "solver_message": str(solved.message),
            "objective_value": float(solved.objective_value),
            "max_error": float(solved.max_error),
            "solver_nit": int(solved.nit),
            "solver_highs_status": str(solved.highs_status),
        }
    )
    _write_json(out_dir / "lp_dataset_summary.json", dataset_summary)

    problem_stats = dict(problem.problem_stats)
    problem_stats.update(
        {
            "residual_max": float(np.max(residual)) if residual.size else 0.0,
            "violation_max": float(np.max(violations)) if violations.size else 0.0,
            "violation_mean": float(np.mean(violations)) if violations.size else 0.0,
            "nonneg_violation_max": float(np.max(nonneg_row_violations)) if nonneg_row_violations.size else 0.0,
            "nonneg_violation_mean": float(np.mean(nonneg_row_violations)) if nonneg_row_violations.size else 0.0,
            "cap_violation_max": float(np.max(cap_row_violations)) if cap_row_violations.size else 0.0,
            "cap_violation_mean": float(np.mean(cap_row_violations)) if cap_row_violations.size else 0.0,
            "sample_error_max": float(np.max(sample_errors)) if sample_errors.size else 0.0,
            "sample_error_mean": float(np.mean(sample_errors)) if sample_errors.size else 0.0,
            "sample_error_derivation_delta_max": float(
                np.max(np.abs(sample_errors - derived_sample_errors))
            ) if sample_errors.size else 0.0,
            "rows_per_sample": int(rows_per_sample),
            "has_cap_rows": bool(problem.has_cap_rows),
            "objective_mode": str(problem.objective_mode),
        }
    )
    _write_json(out_dir / "lp_problem_stats.json", problem_stats)

    np.savez(
        out_dir / "lp_solution.npz",
        weights=solved.weights.astype(np.float64),
        policy_weights=solved.policy_weights.astype(np.float64),
        sample_errors=solved.sample_errors.astype(np.float64),
        max_error=np.array([float(solved.max_error)], dtype=np.float64),
        full_solution=solved.full_solution.astype(np.float64),
        objective_value=np.array([float(solved.objective_value)], dtype=np.float64),
        status=np.array([int(solved.status)], dtype=np.int64),
        success=np.array([1 if solved.success else 0], dtype=np.int64),
        message=np.array([str(solved.message)]),
        nit=np.array([int(solved.nit)], dtype=np.int64),
        highs_status=np.array([str(solved.highs_status)]),
    )

    feature_names = list(problem.feature_names)
    _write_json(out_dir / "lp_feature_names.json", {"feature_names": feature_names})
    _write_weights_by_feature_json(
        out_dir / "lp_weights_by_feature.json",
        feature_names=feature_names,
        weights=solved.weights.astype(np.float64),
    )
    _write_constraint_violations_csv(
        out_csv=out_dir / "lp_constraint_violations.csv",
        sample_errors=sample_errors,
        nonneg_row_violations=nonneg_row_violations,
        cap_row_violations=cap_row_violations,
        ineq_violations_all=violations,
        players=players,
        tol=float(args.violation_tol),
    )

    merged_dir = Path(args.sampler_round_dir)
    c_row, sampled_state_id, sampled_pos, root_state_id = _pick_random_lp_state(
        problem,
        seed=int(args.seed),
    )
    iter_row = _build_mcts_iter_random_state_row(
        merged_dir=merged_dir,
        constraint_row=c_row,
        sampled_state_id=sampled_state_id,
        state_position=sampled_pos,
        root_state_id=root_state_id,
    )
    _write_single_row_csv(
        out_dir / "mcts_iter_random_state.csv",
        fieldnames=ITER_FIELDS,
        row=iter_row,
    )

    feature_row = _build_normalized_feature_row(
        merged_dir=merged_dir,
        sampled_state_id=sampled_state_id,
        feature_set=str(args.feature_set),
        feature_names=feature_names,
    )
    _write_single_row_csv(
        out_dir / "normalised_features_random_state.csv",
        fieldnames=feature_names,
        row=feature_row,
    )

    if not bool(args.skip_default_evaluator):
        _run_default_evaluator(out_dir)

    print(
        f"[linear.lp] solved status={solved.status} success={solved.success} "
        f"objective={solved.objective_value:.6f} "
        f"objective_mode={problem.objective_mode} "
        f"constraints={A.shape[0]} feature_vars={problem.n_weight_vars} total_vars={A.shape[1]} "
        f"out_dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
