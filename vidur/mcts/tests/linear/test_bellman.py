from __future__ import annotations

from vidur.mcts.linear.bellman import effective_discount, q_value, transition_reward


def test_bellman_q_value_matches_manual() -> None:
    cost_s = 10.0
    cost_next = 12.5
    v_next = -3.0
    discount = 0.98
    dt = 0.1
    base = 0.05

    r = transition_reward(cost_s, cost_next)
    g = effective_discount(discount, dt, base)
    q_manual = r + g * v_next
    q = q_value(
        cost_s=cost_s,
        cost_next=cost_next,
        v_next=v_next,
        discount_factor=discount,
        delta_time=dt,
        base_step_time=base,
    )
    assert abs(q - q_manual) < 1e-9


def test_discount_monotonic_in_dt() -> None:
    g0 = effective_discount(0.98, 0.0, 0.05)
    g1 = effective_discount(0.98, 0.1, 0.05)
    g2 = effective_discount(0.98, 0.5, 0.05)
    assert g0 >= g1 >= g2
