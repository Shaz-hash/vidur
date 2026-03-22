from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

from ..sampler.storage import read_parquet_records
from .features import (
    RequestAggregate,
    extract_features_for_set,
    feature_sign_constraints_for_set,
    feature_names_for_set,
    feature_weight_bounds_for_set,
    structural_preference_constraints_for_set,
    update_request_aggregate,
)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class LPAssemblerConfig:
    sampler_round_dir: str
    discount_factor: float = 0.98
    weight_bound_abs: float = 100.0
    structural_margin_eps: float = 1e-3
    feature_set: str = "baseline_v1"
    objective_mode: str = "minimax_directional_slack_eliminated"
    seed: int = 12345


@dataclass(frozen=True)
class LPConstraintRow:
    player: str
    anchor_state_id: str
    transition_id: str
    state_id: str
    next_state_id: str
    delta_cost: float


@dataclass(frozen=True)
class LPProblem:
    feature_names: Sequence[str]
    objective_c: np.ndarray
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    bounds: Sequence[Tuple[float, float]]
    constraint_rows: Sequence[LPConstraintRow]
    dataset_summary: Mapping[str, Any]
    problem_stats: Mapping[str, Any]
    n_weight_vars: int
    n_error_vars: int
    t_var_index: int
    objective_mode: str
    rows_per_sample: int
    has_cap_rows: bool
    transition_error_indices: Mapping[str, int]


def _load_merged_tables(merged_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    anchors = read_parquet_records(merged_dir / "anchors.parquet")
    states = read_parquet_records(merged_dir / "states.parquet")
    transitions = read_parquet_records(merged_dir / "transitions.parquet")
    return {
        "anchors": anchors,
        "states": states,
        "transitions": transitions,
    }


def _build_request_aggregates(
    *,
    requests_path: Path,
    required_state_ids: set[str],
    state_sim_time: Mapping[str, float],
) -> Dict[str, RequestAggregate]:
    if not required_state_ids:
        return {}

    aggs: Dict[str, RequestAggregate] = {}

    def _update(state_id: str, req_row: Mapping[str, Any]) -> None:
        sim_time = _to_float(state_sim_time.get(state_id, 0.0))
        agg = aggs.get(state_id)
        if agg is None:
            agg = RequestAggregate()
            aggs[state_id] = agg
        update_request_aggregate(agg, req_row, sim_time=sim_time)

    if pq is None:
        for row in read_parquet_records(requests_path):
            sid = str(row.get("state_id", ""))
            if sid in required_state_ids:
                _update(sid, row)
        return aggs

    if not requests_path.exists():
        return aggs

    pf = pq.ParquetFile(requests_path)
    cols = [
        "state_id",
        "request_id",
        "prefill_tokens_remaining",
        "decode_tokens_remaining",
        "completed",
        "prefill_complete",
        "prefill_lateness_now",
        "decode_lateness_now",
        "prefill_violated_now",
        "decode_violated_now",
        "violated_now",
        "total_lateness_now",
        "prefill_deadline",
        "decode_deadline",
    ]
    for batch in pf.iter_batches(columns=cols, batch_size=200_000):
        obj = batch.to_pydict()
        n = len(obj.get("state_id", []))
        for i in range(n):
            sid = str(obj["state_id"][i])
            if sid not in required_state_ids:
                continue
            row = {k: obj[k][i] for k in cols if k in obj}
            _update(sid, row)

    return aggs


def _build_state_features(
    *,
    feature_set: str,
    state_rows_by_id: Mapping[str, Mapping[str, Any]],
    req_aggs_by_state_id: Mapping[str, RequestAggregate],
    required_state_ids: Iterable[str],
) -> Dict[str, np.ndarray]:
    names = feature_names_for_set(feature_set)
    out: Dict[str, np.ndarray] = {}
    for sid in sorted(set(str(x) for x in required_state_ids if str(x))):
        sr = state_rows_by_id.get(sid)
        if sr is None:
            continue
        agg = req_aggs_by_state_id.get(sid)
        fmap = extract_features_for_set(feature_set=feature_set, state_row=sr, req_agg=agg)
        vec = np.array([_to_float(fmap.get(name, 0.0)) for name in names], dtype=np.float64)
        out[sid] = vec
    return out


def assemble_lp_problem(cfg: LPAssemblerConfig) -> LPProblem:
    merged_dir = Path(cfg.sampler_round_dir)
    if not merged_dir.exists():
        raise FileNotFoundError(f"sampler round dir does not exist: {merged_dir}")

    tables = _load_merged_tables(merged_dir)
    anchors = tables["anchors"]
    states = tables["states"]
    transitions = tables["transitions"]

    state_rows_by_id: Dict[str, Dict[str, Any]] = {
        str(r.get("state_id", "")): r for r in states if str(r.get("state_id", ""))
    }

    anchor_player_by_state: Dict[str, str] = {}
    n_anchor_ctrl = 0
    n_anchor_adv = 0
    n_anchor_missing_state = 0
    for a in anchors:
        sid = str(a.get("anchor_state_id", ""))
        if not sid:
            continue
        p = str(a.get("player_to_act", "")).strip()
        if sid not in state_rows_by_id:
            n_anchor_missing_state += 1
            continue
        if p not in {"controller", "adversary"}:
            continue
        if sid in anchor_player_by_state:
            continue
        anchor_player_by_state[sid] = p
        if p == "controller":
            n_anchor_ctrl += 1
        else:
            n_anchor_adv += 1

    transitions_by_anchor: Dict[str, List[Dict[str, Any]]] = {}
    for t in transitions:
        sid = str(t.get("state_id", ""))
        if sid not in anchor_player_by_state:
            continue
        transitions_by_anchor.setdefault(sid, []).append(t)

    required_state_ids: set[str] = set(anchor_player_by_state.keys())
    for rows in transitions_by_anchor.values():
        for t in rows:
            nsid = str(t.get("next_state_id", ""))
            if nsid:
                required_state_ids.add(nsid)

    state_sim_time: Dict[str, float] = {
        sid: _to_float(state_rows_by_id[sid].get("sim_time", 0.0))
        for sid in required_state_ids
        if sid in state_rows_by_id
    }
    req_aggs_by_state_id = _build_request_aggregates(
        requests_path=merged_dir / "requests.parquet",
        required_state_ids=required_state_ids,
        state_sim_time=state_sim_time,
    )

    feature_names = feature_names_for_set(cfg.feature_set)
    state_features = _build_state_features(
        feature_set=cfg.feature_set,
        state_rows_by_id=state_rows_by_id,
        req_aggs_by_state_id=req_aggs_by_state_id,
        required_state_ids=required_state_ids,
    )
    d = len(feature_names)
    if d == 0:
        raise RuntimeError("feature dimension is zero")

    mode = str(cfg.objective_mode).strip() or "minimax_directional_slack_eliminated"
    valid_modes = {"minimax_directional_slack_eliminated", "avg_anchor_value_gap"}
    if mode not in valid_modes:
        raise ValueError(f"unsupported objective_mode={mode!r}; expected one of {sorted(valid_modes)}")

    # Build transition set first (one sample per transition).
    gamma = float(cfg.discount_factor)
    transition_terms: List[Tuple[str, str, str, str, np.ndarray, float]] = []
    n_transitions_missing_next_state = 0
    n_anchors_without_transitions = 0
    for sid, player in sorted(anchor_player_by_state.items()):
        trows = transitions_by_anchor.get(sid, [])
        if not trows:
            n_anchors_without_transitions += 1
            continue
        phi_s = state_features.get(sid)
        if phi_s is None:
            continue
        for t in trows:
            tid = str(t.get("transition_id", ""))
            nsid = str(t.get("next_state_id", ""))
            phi_n = state_features.get(nsid)
            if (not nsid) or (phi_n is None):
                n_transitions_missing_next_state += 1
                continue
            delta_cost = _to_float(t.get("cost_next", 0.0)) - _to_float(t.get("cost_s", 0.0))
            a = phi_s - gamma * phi_n
            transition_terms.append((sid, player, tid, nsid, a, float(delta_cost)))

    m = len(transition_terms)
    if m == 0:
        raise RuntimeError("assembled LP has zero constraints; check anchors/transitions input")

    n_weight_vars = int(d)
    n_error_vars = 0
    has_cap_rows = mode == "minimax_directional_slack_eliminated"
    rows_per_sample = 2 if has_cap_rows else 1
    if has_cap_rows:
        t_var_index = int(n_weight_vars)
        n_total_vars = int(n_weight_vars + 1)
        # Objective: minimize t (max per-sample directional slack).
        c = np.zeros((n_total_vars,), dtype=np.float64)
        c[t_var_index] = 1.0
    else:
        t_var_index = -1
        n_total_vars = int(n_weight_vars)
        adv_anchor_vecs = [
            state_features[sid]
            for sid, p in anchor_player_by_state.items()
            if p == "adversary" and sid in state_features
        ]
        ctrl_anchor_vecs = [
            state_features[sid]
            for sid, p in anchor_player_by_state.items()
            if p == "controller" and sid in state_features
        ]
        if not adv_anchor_vecs or not ctrl_anchor_vecs:
            raise RuntimeError(
                "avg_anchor_value_gap requires at least one controller and one adversary anchor with features"
            )
        mean_adv = np.mean(np.stack(adv_anchor_vecs, axis=0), axis=0)
        mean_ctrl = np.mean(np.stack(ctrl_anchor_vecs, axis=0), axis=0)
        c = (mean_adv - mean_ctrl).astype(np.float64, copy=False)

    rows_idx = array("I")
    cols_idx = array("I")
    vals = array("d")
    b_data = array("d")
    row_meta: List[LPConstraintRow] = []
    transition_error_indices: Dict[str, int] = {}
    n_constraints_ctrl = 0
    n_constraints_adv = 0

    for i, (sid, player, tid, nsid, a, b) in enumerate(transition_terms):
        transition_error_indices[tid] = int(i)

        if player == "controller":
            coef_first = a
            rhs_first = float(b)
            coef_second = -a
            rhs_second = float(-b)
        else:
            coef_first = -a
            rhs_first = float(-b)
            coef_second = a
            rhs_second = float(b)

        row_first = rows_per_sample * i

        nz_first = np.flatnonzero(coef_first)
        for j in nz_first.tolist():
            rows_idx.append(row_first)
            cols_idx.append(j)
            vals.append(float(coef_first[j]))
        b_data.append(rhs_first)

        if has_cap_rows:
            row_second = row_first + 1
            nz_second = np.flatnonzero(coef_second)
            for j in nz_second.tolist():
                rows_idx.append(row_second)
                cols_idx.append(j)
                vals.append(float(coef_second[j]))
            rows_idx.append(row_second)
            cols_idx.append(t_var_index)
            vals.append(-1.0)
            b_data.append(rhs_second)

        if player == "controller":
            n_constraints_ctrl += 1
        else:
            n_constraints_adv += 1

        row_meta.append(
            LPConstraintRow(
                player=player,
                anchor_state_id=sid,
                transition_id=tid,
                state_id=sid,
                next_state_id=nsid,
                delta_cost=float(b),
            )
        )

    wmax = float(cfg.weight_bound_abs)
    if not np.isfinite(wmax) or wmax <= 0.0:
        raise ValueError(f"weight_bound_abs must be finite and > 0, got {wmax}")
    bounds: List[Tuple[float, float]] = feature_weight_bounds_for_set(
        cfg.feature_set,
        weight_bound_abs=wmax,
    )
    if has_cap_rows:
        bounds.append((0.0, np.inf))  # t
    sign_constraints = feature_sign_constraints_for_set(cfg.feature_set)
    structural_constraints = structural_preference_constraints_for_set(
        cfg.feature_set,
        margin_eps=float(cfg.structural_margin_eps),
    )
    structural_row_start = rows_per_sample * m
    for offset, pref in enumerate(structural_constraints):
        row_idx = structural_row_start + offset
        better = np.array([_to_float(pref.better_features.get(name, 0.0)) for name in feature_names], dtype=np.float64)
        worse = np.array([_to_float(pref.worse_features.get(name, 0.0)) for name in feature_names], dtype=np.float64)
        coef = better - worse
        nz = np.flatnonzero(coef)
        for j in nz.tolist():
            rows_idx.append(row_idx)
            cols_idx.append(j)
            vals.append(float(coef[j]))
        b_data.append(float(-pref.margin))

    row_np = np.frombuffer(rows_idx, dtype=np.uint32).astype(np.int64, copy=False)
    col_np = np.frombuffer(cols_idx, dtype=np.uint32).astype(np.int64, copy=False)
    val_np = np.frombuffer(vals, dtype=np.float64)
    n_structural_constraints = len(structural_constraints)
    total_rows = rows_per_sample * m + n_structural_constraints
    A_ub = sp.coo_matrix(
        (val_np, (row_np, col_np)),
        shape=(total_rows, n_total_vars),
        dtype=np.float64,
    ).tocsr()
    b_ub = np.frombuffer(b_data, dtype=np.float64).copy()

    dataset_summary: Dict[str, Any] = {
        "feature_set": str(cfg.feature_set),
        "cost_mode": "delta",
        "objective_mode": str(mode),
        "discount_factor": float(cfg.discount_factor),
        "weight_bound_abs": float(cfg.weight_bound_abs),
        "structural_margin_eps": float(cfg.structural_margin_eps),
        "feature_sign_constraints": sign_constraints,
        "structural_constraint_names": [pref.name for pref in structural_constraints],
        "n_anchor_total": int(len(anchor_player_by_state)),
        "n_anchor_ctrl": int(n_anchor_ctrl),
        "n_anchor_adv": int(n_anchor_adv),
        "n_anchor_missing_state": int(n_anchor_missing_state),
        "n_anchor_without_transitions": int(n_anchors_without_transitions),
        "n_constraints_total": int(len(row_meta)),
        "n_constraints_ctrl": int(n_constraints_ctrl),
        "n_constraints_adv": int(n_constraints_adv),
        "n_constraints_structural": int(n_structural_constraints),
        "n_transitions_missing_next_state": int(n_transitions_missing_next_state),
        "feature_dim": int(d),
        "n_weight_vars": int(n_weight_vars),
        "n_error_vars": int(n_error_vars),
        "n_aux_vars_total": 1 if has_cap_rows else 0,
    }

    problem_stats: Dict[str, Any] = {
        "n_rows": int(A_ub.shape[0]),
        "n_cols": int(A_ub.shape[1]),
        "nnz": int(A_ub.nnz),
        "n_bounds": int(len(bounds)),
        "n_sign_constrained_weight_vars": int(len(sign_constraints)),
        "n_rows_structural_constraints": int(n_structural_constraints),
        "n_transition_samples": int(m),
        "n_rows_nonneg_sides": int(m),
        "n_rows_max_error_caps": int(m if has_cap_rows else 0),
    }

    return LPProblem(
        feature_names=tuple(feature_names),
        objective_c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=tuple(bounds),
        constraint_rows=tuple(row_meta),
        dataset_summary=dataset_summary,
        problem_stats=problem_stats,
        n_weight_vars=int(n_weight_vars),
        n_error_vars=int(n_error_vars),
        t_var_index=int(t_var_index),
        objective_mode=str(mode),
        rows_per_sample=int(rows_per_sample),
        has_cap_rows=bool(has_cap_rows),
        transition_error_indices=transition_error_indices,
    )
