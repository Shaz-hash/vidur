from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
from scipy.optimize import linprog

from .assembler import LPProblem


@dataclass(frozen=True)
class LPSolveResult:
    success: bool
    status: int
    message: str
    objective_value: float
    weights: np.ndarray
    policy_weights: np.ndarray
    sample_errors: np.ndarray
    max_error: float
    full_solution: np.ndarray
    nit: int
    highs_status: str
    raw: Mapping[str, Any]


def solve_lp_problem(problem: LPProblem) -> LPSolveResult:
    res = linprog(
        c=problem.objective_c,
        A_ub=problem.A_ub,
        b_ub=problem.b_ub,
        bounds=list(problem.bounds),
        method="highs",
    )
    x = np.array(res.x if res.x is not None else np.zeros_like(problem.objective_c), dtype=np.float64)
    objective = float(res.fun) if res.fun is not None else float(problem.objective_c @ x)
    highs_status = str(getattr(res, "highs_status", ""))

    n_w = int(problem.n_weight_vars)
    t_idx = int(problem.t_var_index)
    w = np.asarray(x[:n_w], dtype=np.float64)
    m = len(problem.constraint_rows)
    step = max(1, int(problem.rows_per_sample))
    if m > 0:
        lhs = np.asarray(problem.A_ub @ x, dtype=np.float64).reshape(-1)
        residual = lhs - problem.b_ub
        # First row for each sample is the non-negativity side.
        first_row_residual = residual[0::step][:m]
        e = np.maximum(0.0, -first_row_residual)
    else:
        e = np.zeros((0,), dtype=np.float64)
    if 0 <= t_idx < int(x.shape[0]):
        t = float(x[t_idx])
    else:
        t = float(np.max(e)) if e.size else 0.0

    raw: Dict[str, Any] = {
        "status": int(res.status),
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", 0) or 0),
        "highs_status": highs_status,
        "objective_mode": str(problem.objective_mode),
    }
    return LPSolveResult(
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        objective_value=float(objective),
        weights=w,
        policy_weights=w,
        sample_errors=e,
        max_error=t,
        full_solution=x,
        nit=int(getattr(res, "nit", 0) or 0),
        highs_status=highs_status,
        raw=raw,
    )
