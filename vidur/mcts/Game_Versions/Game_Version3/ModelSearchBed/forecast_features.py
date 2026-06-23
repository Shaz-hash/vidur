"""Forward-looking analytical features computed from a rebuilt GV3 state.

These features answer "in the next adversary tick, how many SLO crossings are
inevitable regardless of controller action?". They use only deterministic
engine arithmetic (per-request remaining work, prefill profile, decode
deadlines, tick interval) — no simulator rollout required.

The discriminator that matters most is the count of requests whose SLO target
will be crossed within the next tick. Two states with similar cliff-feature
"appearance" (low slack, multiple active requests) often differ by a multiple
of "violation count" in V because the timing relative to the tick window is
different. Cliff features alone don't expose that arithmetic.
"""

from __future__ import annotations
from typing import Any


FORECAST_TOP_K_SLOTS = 8   # top-K most-dangerous active requests
FORECAST_PER_SLOT_DIM = 9   # see _slot_per_request_features

FORECAST_GLOBAL_NAMES = [
    "fc_n_active",
    "fc_n_in_prefill",
    "fc_n_in_decode",
    "fc_tick_advance_est",
    # Prefill-side timing
    "fc_prefill_min_time_to_target",
    "fc_prefill_max_time_to_target",
    "fc_prefill_mean_time_to_target",
    "fc_prefill_min_slack_minus_eta",   # most negative = most doomed
    "fc_prefill_max_slack_minus_eta",
    "fc_n_prefill_doomed",              # eta > slack (this request can't be saved)
    "fc_n_prefill_savable",             # 0 < slack >= eta
    "fc_n_prefill_already_late",        # sim_time > target
    "fc_sum_remaining_prefill_tokens",
    "fc_prefill_eta_min",
    "fc_prefill_eta_max",
    "fc_prefill_eta_total",
    # Decode-side timing
    "fc_decode_min_time_to_deadline",
    "fc_decode_max_time_to_deadline",
    "fc_decode_mean_time_to_deadline",
    "fc_n_decode_inev_violation",       # deadline within tick window
    "fc_n_decode_already_late",
    "fc_n_decode_saveable_in_tick",
    "fc_sum_remaining_decode_tokens",
    # Aggregate forecast
    "fc_n_doomed_total",                # n_prefill_doomed + n_decode_inev
    "fc_n_savable_total",
    "fc_doomed_floor_estimate",         # rough lower bound on next-tick reward
]


def _slot_feature_names() -> list[str]:
    names = list(FORECAST_GLOBAL_NAMES)
    for i in range(FORECAST_TOP_K_SLOTS):
        names.extend([
            f"fc_slot{i:02d}_present",
            f"fc_slot{i:02d}_is_prefill",
            f"fc_slot{i:02d}_remaining_norm",   # remaining_tokens / 1024
            f"fc_slot{i:02d}_time_to_target",
            f"fc_slot{i:02d}_eta",
            f"fc_slot{i:02d}_slack_minus_eta",  # > 0 = savable, <= 0 = doomed
            f"fc_slot{i:02d}_already_late",     # binary
            f"fc_slot{i:02d}_doomed",           # binary: slack < eta
            f"fc_slot{i:02d}_savable",          # binary: 0 < slack and eta <= slack
        ])
    return names


FORECAST_FEATURE_NAMES = _slot_feature_names()


def _zero_forecast() -> list[float]:
    return [0.0] * len(FORECAST_FEATURE_NAMES)


def extract_forecast_features_from_state(
    state: Any,
    *,
    env: Any,
    tick_sec: float | None = None,
) -> tuple[list[float], list[str]]:
    """Compute forward-looking forecast features for a rebuilt GV3 state."""

    if state is None or env is None:
        return _zero_forecast(), list(FORECAST_FEATURE_NAMES)

    sim = state.simulator
    sim_time = float(sim._time)
    stats = state.stats
    active_ids = sorted(int(x) for x in getattr(stats, "active_request_ids", []) or [])

    if not active_ids:
        out = _zero_forecast()
        return out, list(FORECAST_FEATURE_NAMES)

    if tick_sec is None:
        try:
            tick_sec = float(env._gv2_cfg.timing.adversary_tick_sec)
        except Exception:
            tick_sec = 0.2

    # Time until next adversary tick (we're between ticks because controller
    # turns happen between adversary turns).
    eps = 1e-9
    n_full = max(0, int((sim_time + eps) // tick_sec))
    next_tick_time = (n_full + 1) * tick_sec
    tick_advance_est = max(0.0, float(next_tick_time - sim_time))

    req_map = env._req_map(sim)
    decode_dl_map = dict(getattr(stats, "decode_next_deadline_by_id", {}) or {})
    drop_grace = 0.0
    try:
        drop_grace = float(env._gv2_cfg.timing.drop_grace_sec)
    except Exception:
        pass

    profile = getattr(env, "_prefill_profile", None)
    def prefill_eta(toks: int) -> float:
        if toks <= 0 or profile is None:
            return 0.0
        try:
            return float(profile.lookup(int(toks)))
        except Exception:
            return 0.0

    prefill_times: list[float] = []     # slack to SLO target (signed)
    prefill_etas: list[float] = []
    prefill_remaining: list[int] = []
    decode_times: list[float] = []      # slack to next decode deadline (signed)
    decode_remaining: list[int] = []
    # Per-request slot rows for the top-K dangerous slots.
    slots: list[tuple[float, list[float]]] = []  # (danger_score_lower=worse, slot_features)
    n_in_prefill = 0
    n_in_decode = 0
    n_prefill_doomed = 0
    n_prefill_savable = 0
    n_prefill_already_late = 0
    n_decode_inev = 0
    n_decode_already_late = 0
    n_decode_savable = 0

    for rid in active_ids:
        req = req_map.get(int(rid))
        if req is None:
            continue
        prefill_done = bool(getattr(req, "_is_prefill_complete",
                                    getattr(req, "is_prefill_complete", False)))
        if not prefill_done:
            n_in_prefill += 1
            arrived = float(getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0)))
            prefill_slo = float(
                getattr(req, "_prefill_slo_time",
                getattr(req, "_desired_prefill_slo_time",
                getattr(req, "prefill_slo_time",
                getattr(req, "prefill_slo", 0.0)))) or 0.0
            )
            target = arrived + prefill_slo
            slack = target - sim_time
            rem = int(env._remaining_prefill(req))
            eta = prefill_eta(rem)
            slack_minus_eta = slack - eta
            prefill_times.append(slack)
            prefill_etas.append(eta)
            prefill_remaining.append(rem)
            already_late = 1.0 if slack <= 0.0 else 0.0
            if slack <= 0.0:
                n_prefill_already_late += 1
            doomed = (slack < eta)
            savable = (slack > 0.0 and eta <= slack)
            if doomed:
                n_prefill_doomed += 1
            if savable:
                n_prefill_savable += 1
            slot_feat = [
                1.0,                             # present
                1.0,                             # is_prefill
                float(rem) / 1024.0,
                float(slack),
                float(eta),
                float(slack_minus_eta),
                already_late,
                1.0 if doomed else 0.0,
                1.0 if savable else 0.0,
            ]
            slots.append((float(slack_minus_eta), slot_feat))
        else:
            n_in_decode += 1
            rem_dec = int(env._remaining_decode(req))
            next_dl = decode_dl_map.get(int(rid), None)
            slack = 0.0 if next_dl is None else float(next_dl) - sim_time
            decode_times.append(slack)
            decode_remaining.append(rem_dec)
            already_late = 1.0 if slack <= 0.0 else 0.0
            if slack <= 0.0:
                n_decode_already_late += 1
            inev = (slack <= tick_advance_est + eps)
            saveable = (slack > 0.0 and slack <= tick_advance_est + eps)
            if inev:
                n_decode_inev += 1
            if saveable:
                n_decode_savable += 1
            # For decode, "doomed" rule is approximate: if slack < 0, doomed.
            doomed = (slack < 0.0)
            savable = (slack > 0.0)
            slot_feat = [
                1.0,                             # present
                0.0,                             # is_prefill
                float(rem_dec) / 1024.0,
                float(slack),
                0.0,                             # decode "eta" placeholder
                float(slack),                    # slack_minus_eta = slack
                already_late,
                1.0 if doomed else 0.0,
                1.0 if savable else 0.0,
            ]
            slots.append((float(slack), slot_feat))

    def safe_min(xs):
        return float(min(xs)) if xs else 0.0

    def safe_max(xs):
        return float(max(xs)) if xs else 0.0

    def safe_mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    sum_rem_pre = float(sum(prefill_remaining))
    sum_eta_pre = float(sum(prefill_etas))
    eta_min_pre = safe_min(prefill_etas)
    eta_max_pre = safe_max(prefill_etas)
    sum_rem_dec = float(sum(decode_remaining))

    prefill_slack_minus_eta = [
        prefill_times[i] - prefill_etas[i] for i in range(len(prefill_times))
    ]

    n_doomed_total = n_prefill_doomed + n_decode_inev
    n_savable_total = n_prefill_savable + n_decode_savable
    # Floor: one violation = +1 to slo_violations + lateness up to drop_grace.
    doomed_floor = -float(n_doomed_total) * (1.0 + max(0.0, drop_grace))

    global_out = [
        float(len(active_ids)),
        float(n_in_prefill),
        float(n_in_decode),
        float(tick_advance_est),
        # Prefill stats
        safe_min(prefill_times),
        safe_max(prefill_times),
        safe_mean(prefill_times),
        safe_min(prefill_slack_minus_eta),
        safe_max(prefill_slack_minus_eta),
        float(n_prefill_doomed),
        float(n_prefill_savable),
        float(n_prefill_already_late),
        sum_rem_pre,
        eta_min_pre,
        eta_max_pre,
        sum_eta_pre,
        # Decode stats
        safe_min(decode_times),
        safe_max(decode_times),
        safe_mean(decode_times),
        float(n_decode_inev),
        float(n_decode_already_late),
        float(n_decode_savable),
        sum_rem_dec,
        # Aggregate
        float(n_doomed_total),
        float(n_savable_total),
        doomed_floor,
    ]
    assert len(global_out) == len(FORECAST_GLOBAL_NAMES), (
        f"global feature length mismatch {len(global_out)} vs {len(FORECAST_GLOBAL_NAMES)}"
    )

    # Sort slots ascending by danger score (lowest = most doomed first).
    slots.sort(key=lambda x: x[0])
    slot_out: list[float] = []
    for k in range(FORECAST_TOP_K_SLOTS):
        if k < len(slots):
            slot_out.extend(slots[k][1])
        else:
            slot_out.extend([0.0] * FORECAST_PER_SLOT_DIM)

    out = global_out + slot_out
    assert len(out) == len(FORECAST_FEATURE_NAMES), (
        f"forecast feature length mismatch {len(out)} vs {len(FORECAST_FEATURE_NAMES)}"
    )
    return out, list(FORECAST_FEATURE_NAMES)
