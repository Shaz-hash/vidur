from __future__ import annotations

from typing import Any


def state_cost(env: Any, state: Any) -> float:
    violations, lateness_sum = env.evaluate_objective(state)
    return float(violations) + float(lateness_sum)


def compute_base_step_time(env: Any, base_step_tokens: int = 512) -> float:
    """
    Matches current MCTS_DNN calibration shape:
      step_time = prefill_profile.lookup(step_tokens) / prefill_slowdown
    with a safe fallback.
    """
    fallback = 0.0388862329
    try:
        step_tokens = int(getattr(getattr(env, "_constraints", None), "interval_request_size", base_step_tokens) or base_step_tokens)
        slowdown = float(getattr(getattr(env, "_constraints", None), "prefill_slowdown", 1.0) or 1.0)
        if slowdown <= 0:
            slowdown = 1.0
        scaled = float(getattr(env, "_prefill_profile").lookup(step_tokens) or 1e-9)
        v = scaled / slowdown
        if v <= 0:
            return fallback
        return float(v)
    except Exception:
        return fallback


def effective_discount(discount_factor: float, delta_time: float, base_step_time: float) -> float:
    denom = max(float(base_step_time), 1e-9)
    dt = max(0.0, float(delta_time))
    g = float(discount_factor)
    return float(g ** (dt / denom))


def transition_reward(cost_s: float, cost_next: float) -> float:
    # cost reduction should be positive reward
    return float(cost_s) - float(cost_next)


def q_value(
    *,
    cost_s: float,
    cost_next: float,
    v_next: float,
    discount_factor: float,
    delta_time: float,
    base_step_time: float,
) -> float:
    r = transition_reward(cost_s, cost_next)
    gamma_eff = effective_discount(discount_factor, delta_time, base_step_time)
    return float(r + gamma_eff * float(v_next))
