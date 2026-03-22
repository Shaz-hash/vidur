from __future__ import annotations

from pathlib import Path

import numpy as np

from vidur.mcts.linear.lp.assembler import LPAssemblerConfig, assemble_lp_problem
from vidur.mcts.linear.lp.features import (
    RequestAggregate,
    extract_features_for_set,
    feature_sign_constraints_for_set,
    feature_names_for_set,
    feature_weight_bounds_for_set,
    structural_preference_constraints_for_set,
    update_request_aggregate,
)
from vidur.mcts.linear.lp.run import main as lp_main
from vidur.mcts.linear.lp.solver import solve_lp_problem
from vidur.mcts.linear.sampler.storage import write_parquet_records


def _state(
    state_id: str,
    player: str,
    *,
    sim_time: float,
    req_in_system: int,
    slo_viol: int,
    total_lateness: float,
    total_cost: float,
    branching_depth: int,
) -> dict:
    return {
        "state_id": state_id,
        "worker_id": 0,
        "round_idx": 0,
        "root_seed": 123,
        "history_hop": 0,
        "player_to_act": player,
        "branching_depth": branching_depth,
        "sim_time": sim_time,
        "requests_in_system": req_in_system,
        "requests_generated": req_in_system,
        "requests_completed": 0,
        "slo_violations": slo_viol,
        "total_lateness": total_lateness,
        "total_cost": total_cost,
        "completed_request_ids_json": "[]",
        "waiting_request_ids_json": "[]",
        "terminal": False,
    }


def _anchor(state_id: str, player: str) -> dict:
    return {
        "anchor_state_id": state_id,
        "worker_id": 0,
        "round_idx": 0,
        "history_hop": 0,
        "player_to_act": player,
        "branching_depth": 0,
        "trace_len": 1,
        "attempt_index": 1,
    }


def _transition(tid: str, state_id: str, next_state_id: str, actor: str, cost_s: float, cost_next: float) -> dict:
    return {
        "transition_id": tid,
        "state_id": state_id,
        "action_id": f"a_{tid}",
        "next_state_id": next_state_id,
        "actor": actor,
        "sim_time_before": 0.0,
        "sim_time_after_action": 0.0,
        "sim_time_after_advance": 0.0,
        "delta_time_action": 0.0,
        "delta_time_advance": 0.0,
        "delta_time": 0.0,
        "cost_s": cost_s,
        "cost_next": cost_next,
        "reward": 0.0,
        "branching_depth": 0,
        "next_branching_depth": 1,
        "terminal": False,
        "adversary_prefill_deadlines_by_id_json": "{}",
    }


def _build_small_merged_dir(tmp_path: Path) -> Path:
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    states = [
        _state("s_ctrl", "controller", sim_time=1.0, req_in_system=3, slo_viol=1, total_lateness=0.3, total_cost=1.3, branching_depth=0),
        _state("s_adv1", "adversary", sim_time=2.0, req_in_system=4, slo_viol=2, total_lateness=0.5, total_cost=2.5, branching_depth=0),
        _state("s_adv2", "adversary", sim_time=3.0, req_in_system=2, slo_viol=1, total_lateness=0.1, total_cost=1.1, branching_depth=0),
        _state("s_n1", "controller", sim_time=1.5, req_in_system=2, slo_viol=0, total_lateness=0.2, total_cost=0.2, branching_depth=1),
        _state("s_n2", "adversary", sim_time=2.5, req_in_system=1, slo_viol=0, total_lateness=0.1, total_cost=0.1, branching_depth=1),
        _state("s_n3", "adversary", sim_time=3.5, req_in_system=1, slo_viol=0, total_lateness=0.1, total_cost=0.1, branching_depth=1),
    ]
    anchors = [
        _anchor("s_ctrl", "controller"),
        _anchor("s_adv1", "adversary"),
        _anchor("s_adv2", "adversary"),
    ]
    transitions = [
        _transition("t_ctrl", "s_ctrl", "s_n1", "controller", cost_s=2.0, cost_next=5.0),
        _transition("t_adv1", "s_adv1", "s_n2", "adversary", cost_s=7.0, cost_next=4.0),
        _transition("t_adv2", "s_adv2", "s_n3", "adversary", cost_s=9.0, cost_next=8.0),
    ]
    requests: list[dict] = []

    write_parquet_records(merged / "states.parquet", states, compression="zstd")
    write_parquet_records(merged / "anchors.parquet", anchors, compression="zstd")
    write_parquet_records(merged / "transitions.parquet", transitions, compression="zstd")
    write_parquet_records(merged / "requests.parquet", requests, compression="zstd")
    return merged


def test_baseline_v1_feature_names_and_slot_semantics() -> None:
    names = feature_names_for_set("baseline_v1")
    assert len(names) == 7
    assert names[0] == "bias"
    assert names[1] == "current_burst_size_norm"
    assert names[2] == "current_burst_remaining_chunks_norm"
    assert names[3] == "current_burst_deadline_urgency_norm"
    assert names[4] == "current_burst_lateness_sec_norm"
    assert names[5] == "decode_active_norm"
    assert names[6] == "decode_violated_count_norm"

    sim_time = 1.0
    agg = RequestAggregate()
    rows = [
        {
            "request_id": 1,
            "completed": False,
            "prefill_complete": False,
            "prefill_tokens_remaining": 1536,
            "decode_tokens_remaining": 5000,
            "prefill_deadline": 1.2,
            "prefill_lateness_now": 0.0,
            "prefill_violated_now": False,
            "decode_violated_now": False,
            "violated_now": False,
            "total_lateness_now": 0.0,
        },
        {
            "request_id": 2,
            "completed": False,
            "prefill_complete": False,
            "prefill_tokens_remaining": 3072,
            "decode_tokens_remaining": 5000,
            "prefill_deadline": 1.2,
            "prefill_lateness_now": 0.0,
            "prefill_violated_now": False,
            "decode_violated_now": False,
            "violated_now": False,
            "total_lateness_now": 0.0,
        },
        {
            "request_id": 5,
            "completed": False,
            "prefill_complete": True,
            "prefill_tokens_remaining": 0,
            "decode_tokens_remaining": 5000,
            "prefill_deadline": 1.2,
            "decode_deadline": 1.4,
            "prefill_lateness_now": 0.0,
            "decode_lateness_now": 0.0,
            "prefill_violated_now": False,
            "decode_violated_now": False,
            "violated_now": False,
            "total_lateness_now": 0.0,
        },
        {
            "request_id": 8,
            "completed": False,
            "prefill_complete": True,
            "prefill_tokens_remaining": 0,
            "decode_tokens_remaining": 100,
            "decode_deadline": 0.7,
            "prefill_deadline": 0.6,
            "prefill_lateness_now": 0.0,
            "decode_lateness_now": 0.3,
            "prefill_violated_now": False,
            "decode_violated_now": True,
            "violated_now": True,
            "total_lateness_now": 0.3,
        },
        {
            "request_id": 4,
            "completed": True,
            "prefill_complete": True,
            "prefill_tokens_remaining": 0,
            "decode_tokens_remaining": 500,
            "decode_deadline": 0.95,
            "prefill_deadline": 0.6,
            "prefill_lateness_now": 0.0,
            "decode_lateness_now": 0.3,
            "prefill_violated_now": False,
            "decode_violated_now": True,
            "violated_now": True,
            "total_lateness_now": 0.3,
        },
        {
            "request_id": 20,
            "completed": True,
            "prefill_complete": False,
            "prefill_tokens_remaining": 1000,
            "decode_tokens_remaining": 0,
            "prefill_deadline": 2.0,
            "prefill_lateness_now": 0.0,
            "prefill_violated_now": False,
            "decode_violated_now": False,
            "violated_now": False,
            "total_lateness_now": 0.0,
        },
    ]
    for row in rows:
        update_request_aggregate(agg, row, sim_time=sim_time)

    state_row = {
        "state_id": "s_test",
        "player_to_act": "controller",
        "sim_time": sim_time,
        "requests_in_system": 5,
    }
    feats = extract_features_for_set(feature_set="baseline_v1", state_row=state_row, req_agg=agg)
    vec = np.array([feats[name] for name in names], dtype=np.float64)
    assert vec.shape == (7,)

    assert abs(feats["current_burst_size_norm"] - (3.0 / 6.0)) < 1e-12
    assert abs(feats["current_burst_remaining_chunks_norm"] - (9.0 / 36.0)) < 1e-12
    assert abs(feats["current_burst_deadline_urgency_norm"] - 5.0) < 1e-12
    assert abs(feats["current_burst_lateness_sec_norm"] - 0.0) < 1e-12
    assert abs(feats["decode_active_norm"] - (2.0 / 200.0)) < 1e-12
    assert abs(feats["decode_violated_count_norm"] - (1.0 / 200.0)) < 1e-12


def test_baseline_v1_no_active_prefill_defaults() -> None:
    agg = RequestAggregate()
    update_request_aggregate(
        agg,
        {
            "request_id": 11,
            "completed": False,
            "prefill_complete": True,
            "prefill_tokens_remaining": 0,
            "decode_tokens_remaining": 256,
            "prefill_deadline": 0.5,
            "prefill_lateness_now": 0.0,
            "prefill_violated_now": False,
            "decode_violated_now": False,
            "violated_now": False,
            "total_lateness_now": 0.0,
        },
        sim_time=1.0,
    )
    feats = extract_features_for_set(
        feature_set="baseline_v1",
        state_row={"state_id": "s", "player_to_act": "adversary", "sim_time": 1.0, "requests_in_system": 1},
        req_agg=agg,
    )
    assert abs(feats["current_burst_size_norm"] - 0.0) < 1e-12
    assert abs(feats["current_burst_remaining_chunks_norm"] - 0.0) < 1e-12
    assert abs(feats["current_burst_deadline_urgency_norm"] - 0.0) < 1e-12
    assert abs(feats["current_burst_lateness_sec_norm"] - 0.0) < 1e-12
    assert abs(feats["decode_active_norm"] - (1.0 / 200.0)) < 1e-12
    assert abs(feats["decode_violated_count_norm"] - 0.0) < 1e-12


def test_baseline_v1_late_burst_features() -> None:
    agg = RequestAggregate()
    update_request_aggregate(
        agg,
        {
            "request_id": 7,
            "completed": False,
            "prefill_complete": False,
            "prefill_tokens_remaining": 512,
            "decode_tokens_remaining": 5000,
            "prefill_deadline": 0.8,
            "prefill_lateness_now": 0.2,
            "prefill_violated_now": True,
            "decode_violated_now": False,
            "violated_now": True,
            "total_lateness_now": 0.2,
        },
        sim_time=1.0,
    )
    feats = extract_features_for_set(
        feature_set="baseline_v1",
        state_row={"state_id": "s_late", "player_to_act": "controller", "sim_time": 1.0, "requests_in_system": 1},
        req_agg=agg,
    )
    assert abs(feats["current_burst_size_norm"] - (1.0 / 6.0)) < 1e-12
    assert abs(feats["current_burst_remaining_chunks_norm"] - (1.0 / 36.0)) < 1e-12
    assert abs(feats["current_burst_deadline_urgency_norm"] - 1000.0) < 1e-12
    assert abs(feats["current_burst_lateness_sec_norm"] - 0.2) < 1e-12


def test_baseline_v1_feature_weight_bounds_and_sign_constraints() -> None:
    bounds = feature_weight_bounds_for_set("baseline_v1", weight_bound_abs=10.0)
    assert bounds == [
        (-10.0, 10.0),
        (0.0, 10.0),
        (0.0, 10.0),
        (0.0, 10.0),
        (0.0, 10.0),
        (0.0, 10.0),
        (0.0, 10.0),
    ]
    assert feature_sign_constraints_for_set("baseline_v1") == {
        "current_burst_size_norm": "nonnegative",
        "current_burst_remaining_chunks_norm": "nonnegative",
        "current_burst_deadline_urgency_norm": "nonnegative",
        "current_burst_lateness_sec_norm": "nonnegative",
        "decode_active_norm": "nonnegative",
        "decode_violated_count_norm": "nonnegative",
    }


def test_baseline_v1_structural_preference_constraints() -> None:
    prefs = structural_preference_constraints_for_set("baseline_v1", margin_eps=1e-3)
    assert [p.name for p in prefs] == [
        "fresh_burst_vs_empty",
        "more_decode_load_worse",
    ]
    assert all(abs(float(p.margin) - 1e-3) < 1e-12 for p in prefs)


def test_lp_assembler_signs_delta_and_objective(tmp_path: Path) -> None:
    merged = _build_small_merged_dir(tmp_path)
    cfg = LPAssemblerConfig(
        sampler_round_dir=str(merged),
        discount_factor=0.98,
        weight_bound_abs=100.0,
        feature_set="baseline_v1",
        seed=123,
    )
    problem = assemble_lp_problem(cfg)
    d = len(problem.feature_names)
    m = len(problem.constraint_rows)
    assert problem.A_ub.shape[0] == 2 * m
    assert problem.A_ub.shape[1] == d + 1
    assert problem.n_weight_vars == d
    assert problem.n_error_vars == 0
    assert problem.t_var_index == d
    assert problem.objective_mode == "minimax_directional_slack_eliminated"
    assert problem.rows_per_sample == 2
    assert problem.has_cap_rows is True
    assert problem.bounds[:d] == tuple(feature_weight_bounds_for_set("baseline_v1", weight_bound_abs=100.0))
    assert problem.dataset_summary["n_constraints_ctrl"] == 1
    assert problem.dataset_summary["n_constraints_adv"] == 2
    assert problem.dataset_summary["n_constraints_structural"] == 2
    assert problem.dataset_summary["feature_sign_constraints"] == feature_sign_constraints_for_set("baseline_v1")
    assert problem.dataset_summary["structural_constraint_names"] == [
        "fresh_burst_vs_empty",
        "more_decode_load_worse",
    ]

    # Build expected feature vectors directly using feature extractor.
    all_states = {
        "s_ctrl": _state("s_ctrl", "controller", sim_time=1.0, req_in_system=3, slo_viol=1, total_lateness=0.3, total_cost=1.3, branching_depth=0),
        "s_adv1": _state("s_adv1", "adversary", sim_time=2.0, req_in_system=4, slo_viol=2, total_lateness=0.5, total_cost=2.5, branching_depth=0),
        "s_adv2": _state("s_adv2", "adversary", sim_time=3.0, req_in_system=2, slo_viol=1, total_lateness=0.1, total_cost=1.1, branching_depth=0),
        "s_n1": _state("s_n1", "controller", sim_time=1.5, req_in_system=2, slo_viol=0, total_lateness=0.2, total_cost=0.2, branching_depth=1),
        "s_n2": _state("s_n2", "adversary", sim_time=2.5, req_in_system=1, slo_viol=0, total_lateness=0.1, total_cost=0.1, branching_depth=1),
        "s_n3": _state("s_n3", "adversary", sim_time=3.5, req_in_system=1, slo_viol=0, total_lateness=0.1, total_cost=0.1, branching_depth=1),
    }
    names = feature_names_for_set("baseline_v1")
    phi = {
        sid: np.array(
            [extract_features_for_set(feature_set="baseline_v1", state_row=sr, req_agg=None).get(n, 0.0) for n in names],
            dtype=np.float64,
        )
        for sid, sr in all_states.items()
    }

    c_expected = np.zeros((d + 1,), dtype=np.float64)
    c_expected[problem.t_var_index] = 1.0
    np.testing.assert_allclose(problem.objective_c, c_expected, rtol=0, atol=1e-12)

    row_by_tid = {r.transition_id: i for i, r in enumerate(problem.constraint_rows)}
    A = problem.A_ub.toarray()
    b = problem.b_ub

    i_ctrl = row_by_tid["t_ctrl"]
    t_idx = problem.t_var_index
    a_ctrl = phi["s_ctrl"] - 0.98 * phi["s_n1"]

    np.testing.assert_allclose(A[2 * i_ctrl, :d], a_ctrl, rtol=0, atol=1e-10)
    assert abs(float(b[2 * i_ctrl]) - 3.0) < 1e-12  # delta = 5 - 2

    np.testing.assert_allclose(A[2 * i_ctrl + 1, :d], -a_ctrl, rtol=0, atol=1e-10)
    assert abs(float(A[2 * i_ctrl + 1, t_idx]) + 1.0) < 1e-12
    assert abs(float(b[2 * i_ctrl + 1]) + 3.0) < 1e-12

    i_adv1 = row_by_tid["t_adv1"]
    a_adv1 = phi["s_adv1"] - 0.98 * phi["s_n2"]

    # Adversary nonneg side: -a <= -delta  (delta=-3 => rhs=3)
    np.testing.assert_allclose(A[2 * i_adv1, :d], -a_adv1, rtol=0, atol=1e-10)
    assert abs(float(b[2 * i_adv1]) - 3.0) < 1e-12

    # Adversary cap side: a - t <= delta  (delta=-3)
    np.testing.assert_allclose(A[2 * i_adv1 + 1, :d], a_adv1, rtol=0, atol=1e-10)
    assert abs(float(A[2 * i_adv1 + 1, t_idx]) + 1.0) < 1e-12
    assert abs(float(b[2 * i_adv1 + 1]) + 3.0) < 1e-12

    # Structural constraints are appended after the Bellman rows.
    first_structural_row = 2 * m
    assert A.shape[0] == 2 * m + 2
    assert abs(float(b[first_structural_row]) + 1e-3) < 1e-12


def test_lp_solver_and_cli_smoke(tmp_path: Path) -> None:
    merged = _build_small_merged_dir(tmp_path)
    problem = assemble_lp_problem(
        LPAssemblerConfig(
            sampler_round_dir=str(merged),
            discount_factor=0.98,
            weight_bound_abs=10.0,
            feature_set="baseline_v1",
            seed=123,
        )
    )
    solved = solve_lp_problem(problem)
    m = len(problem.constraint_rows)
    d = len(problem.feature_names)
    assert solved.weights.shape[0] == d
    assert solved.policy_weights.shape[0] == d
    assert solved.sample_errors.shape[0] == m
    assert solved.full_solution.shape[0] == d + 1
    assert np.all(np.isfinite(solved.weights))
    assert np.all(solved.sample_errors >= -1e-10)
    assert np.max(solved.sample_errors) <= solved.max_error + 1e-6
    resid = problem.A_ub @ solved.full_solution - problem.b_ub
    first_resid = resid[0::2][:m]
    second_resid = resid[1::2][:m]
    derived = np.maximum(0.0, -first_resid)
    np.testing.assert_allclose(derived, solved.sample_errors, rtol=0, atol=1e-7)
    assert np.max(np.maximum(0.0, first_resid)) < 1e-5
    assert np.max(np.maximum(0.0, second_resid)) < 1e-5
    assert solved.status in {0, 1, 2, 3, 4}

    out_dir = tmp_path / "lp_out"
    lp_main(
        [
            "--sampler-round-dir",
            str(merged),
            "--out-dir",
            str(out_dir),
            "--discount-factor",
            "0.98",
            "--weight-bound-abs",
            "10.0",
            "--feature-set",
            "baseline_v1",
            "--skip-default-evaluator",
            "--seed",
            "123",
        ]
    )
    assert (out_dir / "lp_dataset_summary.json").exists()
    assert (out_dir / "lp_problem_stats.json").exists()
    assert (out_dir / "lp_solution.npz").exists()
    assert (out_dir / "lp_feature_names.json").exists()
    assert (out_dir / "lp_constraint_violations.csv").exists()

    with np.load(out_dir / "lp_solution.npz") as arr:
        assert arr["weights"].shape[0] == d
        assert arr["policy_weights"].shape[0] == d
        assert arr["sample_errors"].shape[0] == m
        assert arr["full_solution"].shape[0] == d + 1


def test_lp_assembler_avg_anchor_objective_and_constraints(tmp_path: Path) -> None:
    merged = _build_small_merged_dir(tmp_path)
    problem = assemble_lp_problem(
        LPAssemblerConfig(
            sampler_round_dir=str(merged),
            discount_factor=0.98,
            weight_bound_abs=100.0,
            feature_set="baseline_v1",
            objective_mode="avg_anchor_value_gap",
            seed=123,
        )
    )
    d = len(problem.feature_names)
    m = len(problem.constraint_rows)
    assert problem.A_ub.shape == (m + 2, d)
    assert problem.n_error_vars == 0
    assert problem.t_var_index == -1
    assert problem.objective_mode == "avg_anchor_value_gap"
    assert problem.rows_per_sample == 1
    assert problem.has_cap_rows is False
    assert problem.bounds == tuple(feature_weight_bounds_for_set("baseline_v1", weight_bound_abs=100.0))
    assert problem.dataset_summary["n_constraints_structural"] == 2

    all_states = {
        "s_ctrl": _state("s_ctrl", "controller", sim_time=1.0, req_in_system=3, slo_viol=1, total_lateness=0.3, total_cost=1.3, branching_depth=0),
        "s_adv1": _state("s_adv1", "adversary", sim_time=2.0, req_in_system=4, slo_viol=2, total_lateness=0.5, total_cost=2.5, branching_depth=0),
        "s_adv2": _state("s_adv2", "adversary", sim_time=3.0, req_in_system=2, slo_viol=1, total_lateness=0.1, total_cost=1.1, branching_depth=0),
        "s_n1": _state("s_n1", "controller", sim_time=1.5, req_in_system=2, slo_viol=0, total_lateness=0.2, total_cost=0.2, branching_depth=1),
        "s_n2": _state("s_n2", "adversary", sim_time=2.5, req_in_system=1, slo_viol=0, total_lateness=0.1, total_cost=0.1, branching_depth=1),
        "s_n3": _state("s_n3", "adversary", sim_time=3.5, req_in_system=1, slo_viol=0, total_lateness=0.1, total_cost=0.1, branching_depth=1),
    }
    names = feature_names_for_set("baseline_v1")
    phi = {
        sid: np.array(
            [extract_features_for_set(feature_set="baseline_v1", state_row=sr, req_agg=None).get(n, 0.0) for n in names],
            dtype=np.float64,
        )
        for sid, sr in all_states.items()
    }
    c_expected = 0.5 * (phi["s_adv1"] + phi["s_adv2"]) - phi["s_ctrl"]
    np.testing.assert_allclose(problem.objective_c, c_expected, rtol=0, atol=1e-12)

    row_by_tid = {r.transition_id: i for i, r in enumerate(problem.constraint_rows)}
    A = problem.A_ub.toarray()
    b = problem.b_ub

    i_ctrl = row_by_tid["t_ctrl"]
    np.testing.assert_allclose(A[i_ctrl, :d], phi["s_ctrl"] - 0.98 * phi["s_n1"], rtol=0, atol=1e-10)
    assert abs(float(b[i_ctrl]) - 3.0) < 1e-12

    i_adv1 = row_by_tid["t_adv1"]
    np.testing.assert_allclose(A[i_adv1, :d], -(phi["s_adv1"] - 0.98 * phi["s_n2"]), rtol=0, atol=1e-10)
    assert abs(float(b[i_adv1]) - 3.0) < 1e-12


def test_lp_solver_and_cli_avg_mode_smoke(tmp_path: Path) -> None:
    merged = _build_small_merged_dir(tmp_path)
    problem = assemble_lp_problem(
        LPAssemblerConfig(
            sampler_round_dir=str(merged),
            discount_factor=0.98,
            weight_bound_abs=10.0,
            feature_set="baseline_v1",
            objective_mode="avg_anchor_value_gap",
            seed=123,
        )
    )
    solved = solve_lp_problem(problem)
    d = len(problem.feature_names)
    m = len(problem.constraint_rows)
    assert solved.weights.shape[0] == d
    assert solved.full_solution.shape[0] == d
    assert solved.sample_errors.shape[0] == m
    assert abs(float(solved.max_error) - float(np.max(solved.sample_errors))) < 1e-8

    out_dir = tmp_path / "lp_out_avg"
    lp_main(
        [
            "--sampler-round-dir",
            str(merged),
            "--out-dir",
            str(out_dir),
            "--discount-factor",
            "0.98",
            "--weight-bound-abs",
            "10.0",
            "--feature-set",
            "baseline_v1",
            "--objective-mode",
            "avg_anchor_value_gap",
            "--skip-default-evaluator",
            "--seed",
            "123",
        ]
    )
    with np.load(out_dir / "lp_solution.npz") as arr:
        assert arr["weights"].shape[0] == d
        assert arr["sample_errors"].shape[0] == m
        assert arr["full_solution"].shape[0] == d
