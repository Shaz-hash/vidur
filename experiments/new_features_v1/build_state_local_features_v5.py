"""
Build the v5 state-local feature schema = v4 (224d) + analytical lookahead block (20d).

The v5 schema adds 20 scalars derived from prefill_profile.csv-driven greedy chunked
prefill simulation. Per Q&A line 343 in TASK.md: "Analytical lookahead features built
from prefill_profile.csv: e.g., for each candidate token_budget B in {128, 256, 512,
1024}, simulate one tick of greedy chunked prefill using the profile's batch-time
table on the parent's pending prefills; report counts of prefills that miss SLO.
Parent-only — NOT children." (We compute identically on parent and child rows; this
is state-local and used identically in train and inference, satisfying the iteration-
time symmetry rule.)

For each B in [128, 256, 512, 1024]:
  - Greedily pick active prefill requests by ascending prefill slack until total
    selected prefill remaining tokens >= B (or no more requests). Cap each request's
    contribution to its remaining tokens.
  - tick_time = prefill_profile_lookup(min(B_used, total_capable))
  - tick_time = max(tick_time, decode_time_at_max_context) (since decode tail
    blocks the chunked tick).
  - For prefills NOT picked, slack drops by tick_time.
  - For each not-picked prefill: count where post-tick slack < 0 (would violate).
  - For each active decode: count where (decode_next_deadline - sim_time) - tick_time < 0.

Per B emit 5 scalars:
  - frac_prefills_picked     = n_picked / max(1, n_active_prefill)
  - tick_tokens_norm         = total_prefill_tokens_for_tick / 1024 (clipped 0..1)
  - tick_time_norm           = tick_time / 0.5  (decode batch ~0.013s, prefill 1024 ~0.098s)
  - n_post_tick_prefill_viol = (count) / 20
  - n_post_tick_decode_viol  = (count) / 100

4 budgets x 5 scalars = 20 new globals.

Total D = 21 (v4 globals) + 20 (lookahead) + 7*25 (prefill slots) + 7*4 (decode) = 244.

The 41 "globals" sit at the start of the vector for HGB-friendly column locality.
Slots remain identical to v4 so train pipelines can reuse parent-side metadata.
"""

from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ----------------------- denominators / constants (mirror infer.py) -----------------------

F_ACTIVE_PREFILL_COUNT_DEN: float = 20.0
F_ACTIVE_DECODE_COUNT_DEN: float = 100.0
F_ACTIVE_TOTAL_COUNT_DEN: float = 120.0
F_TOTAL_REMAINING_PREFILL_DEN: float = 20.0 * 4096.0
F_TOTAL_REMAINING_DECODE_DEN: float = 100.0 * 864.0
F_TOTAL_DECODE_GENERATED_ACTIVE_DEN: float = 100.0 * 864.0
F_VIOLATED_COUNT_DEN: float = 100.0
F_PREFILL_NEAR_DROP_DEN: float = 20.0
F_DECODE_NEAR_DROP_DEN: float = 100.0
F_RECENT_LAUNCH_COUNT_DEN: float = 7.0
F_RECENT_LAUNCH_PREFILL_DEN: float = 1024.0 * 7.0
F_DECODE_CREDIT_DEN: float = 100.0 * 216.0
F_NEAR_DROP_LATENESS_LOW_SEC: float = 0.5
F_NEAR_DROP_LATENESS_HIGH_SEC: float = 1.5
F_LAUNCH_EWMA_WINDOW_SEC: float = 1.0
F_LAUNCH_EWMA_ALPHA: float = 0.37
MAX_REQUESTS_PER_LAUNCH_WINDOW: int = 7
MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW: int = 1024 * 7
DECODE_REMAINING_DEN: float = 864.0


_DECODE_PROFILE_TOKENS: list[int] = []
_DECODE_PROFILE_TIMES: list[float] = []
_PREFILL_PROFILE_TOKENS: list[int] = []
_PREFILL_PROFILE_TIMES: list[float] = []
_DECODE_PROFILE_DEFAULT = REPO_ROOT / "simulator_output" / "decode_profile.csv"
_PREFILL_PROFILE_DEFAULT = REPO_ROOT / "simulator_output" / "prefill_profile.csv"

LOOKAHEAD_BUDGETS: list[int] = [128, 256, 512, 1024]
TICK_TIME_NORM_DEN: float = 0.5
LOOKAHEAD_PREFILL_VIOL_DEN: float = 20.0
LOOKAHEAD_DECODE_VIOL_DEN: float = 100.0
N_LOOKAHEAD_BUDGETS: int = len(LOOKAHEAD_BUDGETS)
D_LOOKAHEAD_PER_B: int = 5
D_LOOKAHEAD: int = N_LOOKAHEAD_BUDGETS * D_LOOKAHEAD_PER_B  # 20


def _load_decode_profile(path: Path | None = None) -> None:
    global _DECODE_PROFILE_TOKENS, _DECODE_PROFILE_TIMES
    if _DECODE_PROFILE_TOKENS:
        return
    p = path or Path(os.environ.get("DECODE_PROFILE_PATH", str(_DECODE_PROFILE_DEFAULT)))
    rows: list[tuple[int, float]] = []
    with open(p, newline="") as f:
        reader = _csv.reader(f)
        next(reader)
        for r in reader:
            if not r:
                continue
            rows.append((int(r[0]), float(r[1])))
    rows.sort()
    _DECODE_PROFILE_TOKENS = [t for t, _ in rows]
    _DECODE_PROFILE_TIMES = [t for _, t in rows]


def _load_prefill_profile(path: Path | None = None) -> None:
    global _PREFILL_PROFILE_TOKENS, _PREFILL_PROFILE_TIMES
    if _PREFILL_PROFILE_TOKENS:
        return
    p = path or Path(os.environ.get("PREFILL_PROFILE_PATH", str(_PREFILL_PROFILE_DEFAULT)))
    rows: list[tuple[int, float]] = []
    with open(p, newline="") as f:
        reader = _csv.reader(f)
        next(reader)
        for r in reader:
            if not r:
                continue
            rows.append((int(r[0]), float(r[1])))
    rows.sort()
    _PREFILL_PROFILE_TOKENS = [t for t, _ in rows]
    _PREFILL_PROFILE_TIMES = [t for _, t in rows]


def _decode_time_for_tokens(n_tokens: int) -> float:
    if not _DECODE_PROFILE_TOKENS:
        _load_decode_profile()
    if n_tokens <= _DECODE_PROFILE_TOKENS[0]:
        return _DECODE_PROFILE_TIMES[0]
    if n_tokens >= _DECODE_PROFILE_TOKENS[-1]:
        return _DECODE_PROFILE_TIMES[-1]
    best_i = 0
    best_d = abs(n_tokens - _DECODE_PROFILE_TOKENS[0])
    for i, t in enumerate(_DECODE_PROFILE_TOKENS):
        d = abs(n_tokens - t)
        if d < best_d or (d == best_d and t < _DECODE_PROFILE_TOKENS[best_i]):
            best_d = d
            best_i = i
    return _DECODE_PROFILE_TIMES[best_i]


def _prefill_time_for_tokens(n_tokens: int) -> float:
    """Lookup prefill batch time for the smallest profile bucket >= n_tokens, clamped at table ends."""
    if not _PREFILL_PROFILE_TOKENS:
        _load_prefill_profile()
    if n_tokens <= 0:
        return 0.0
    if n_tokens <= _PREFILL_PROFILE_TOKENS[0]:
        return _PREFILL_PROFILE_TIMES[0]
    if n_tokens >= _PREFILL_PROFILE_TOKENS[-1]:
        return _PREFILL_PROFILE_TIMES[-1]
    # smallest bucket >= n_tokens
    for i, t in enumerate(_PREFILL_PROFILE_TOKENS):
        if t >= n_tokens:
            return _PREFILL_PROFILE_TIMES[i]
    return _PREFILL_PROFILE_TIMES[-1]


PREFILL_SLACK_BUCKET_EDGES = [
    0.0133,
    0.0135,
    0.0137,
    0.0140,
    0.0145,
    0.015725797204323228,
    0.023274675327417962,
    0.031963770276289896,
    0.03888623299112536,
    0.06091750997076902,
    0.07016849423102292,
    0.08612442901238251,
    0.09850190759874299,
    0.19613847773632437,
    0.28408680179190937,
]
N_SLACK_BUCKETS = 17
N_LATE_BUCKETS = 4

N_PREFILL_SLOTS = MAX_REQUESTS_PER_LAUNCH_WINDOW
N_DECODE_SLOTS = MAX_REQUESTS_PER_LAUNCH_WINDOW

D_PREFILL_PER_SLOT = 1 + 1 + 1 + 1 + N_LATE_BUCKETS + N_SLACK_BUCKETS
D_DECODE_PER_SLOT = 4
D_GLOBAL_V4 = 21  # the 21 v4 globals
D_GLOBAL = D_GLOBAL_V4 + D_LOOKAHEAD  # 41
D_TOTAL = D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT + N_DECODE_SLOTS * D_DECODE_PER_SLOT  # 244


# ----------------------- helpers (mirror v4 builder) -----------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _norm01(x: float, denom: float) -> float:
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(x) / float(denom)))


def _slack_bucket_idx(slack: float) -> int:
    if slack <= 0.0:
        return 0
    for i, edge in enumerate(PREFILL_SLACK_BUCKET_EDGES):
        if slack <= edge:
            return i + 1
    return len(PREFILL_SLACK_BUCKET_EDGES) + 1


def _lateness_bucket_idx(late: float) -> int:
    if late < 0.5:
        return 0
    if late < 1.0:
        return 1
    if late < 1.5:
        return 2
    return 3


def _stats_attr(stats: Any, name: str, default: Any = None) -> Any:
    return getattr(stats, name, default)


def _stats_dict(stats: Any, name: str) -> dict:
    out = getattr(stats, name, None)
    return out if isinstance(out, dict) else {}


def _stats_set(stats: Any, name: str) -> set:
    out = getattr(stats, name, None)
    if out is None:
        return set()
    try:
        return set(int(x) for x in out)
    except Exception:
        return set()


def _is_prefill(req: dict) -> bool:
    if bool(req.get("completed", False)):
        return False
    if bool(req.get("is_prefill_complete", False)):
        return False
    return True


def _is_decode(req: dict) -> bool:
    if bool(req.get("completed", False)):
        return False
    if not bool(req.get("is_prefill_complete", False)):
        return False
    total = _safe_int(req.get("num_decode_tokens"))
    proc_total = _safe_int(req.get("num_processed_tokens"))
    total_pref = _safe_int(req.get("num_prefill_tokens"))
    done_decode = max(0, proc_total - total_pref)
    return total > done_decode


def _det_random_subset_indices(n_total: int, k: int, *, seed_key: str) -> list[int]:
    if n_total <= k:
        return list(range(n_total))
    h = hashlib.sha256(seed_key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=k, replace=False).tolist())


# ----------------------- per-record feature extraction -----------------------


def _compute_lookahead_block(
    enriched_prefill: list[dict],
    decode_reqs: list[dict],
    decode_deadlines: dict,
    sim_time: float,
    decode_time_at_max: float,
) -> list[float]:
    """For each B in LOOKAHEAD_BUDGETS, compute 5 scalars; return a flat list of D_LOOKAHEAD floats."""

    if enriched_prefill:
        sorted_pref = sorted(enriched_prefill, key=lambda p: (p["slack_clamped"], p["rid"]))
    else:
        sorted_pref = []
    n_active_prefill = max(1, len(sorted_pref))

    decode_deadline_list: list[float] = []
    for r in decode_reqs:
        rid = _safe_int(r.get("id"), -1)
        arrived_at = _safe_float(r.get("arrived_at"))
        decode_slo = _safe_float(r.get("decode_slo_time"))
        deadline = _safe_float(decode_deadlines.get(rid, arrived_at + decode_slo))
        decode_deadline_list.append(deadline - sim_time)

    out: list[float] = []
    for B in LOOKAHEAD_BUDGETS:
        remaining_budget = B
        n_picked = 0
        tokens_for_tick = 0
        picked_ids: set[int] = set()
        for p in sorted_pref:
            if remaining_budget <= 0:
                break
            take = min(int(p["remaining"]), remaining_budget)
            if take <= 0:
                continue
            tokens_for_tick += take
            remaining_budget -= take
            n_picked += 1
            picked_ids.add(int(p["rid"]))
            if n_picked >= MAX_REQUESTS_PER_LAUNCH_WINDOW:
                break

        # tick time = max(prefill_lookup(tokens), decode_time_at_max)
        if tokens_for_tick > 0:
            t_pref = _prefill_time_for_tokens(tokens_for_tick)
        else:
            t_pref = 0.0
        tick_time = max(t_pref, decode_time_at_max)

        n_post_tick_pref_viol = 0
        for p in sorted_pref:
            if int(p["rid"]) in picked_ids:
                continue
            if p["slack_clamped"] - tick_time < 0.0:
                n_post_tick_pref_viol += 1

        n_post_tick_dec_viol = 0
        for d_until_deadline in decode_deadline_list:
            if d_until_deadline - tick_time < 0.0:
                n_post_tick_dec_viol += 1

        out.append(_norm01(n_picked, MAX_REQUESTS_PER_LAUNCH_WINDOW))
        out.append(_norm01(tokens_for_tick, MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW))
        out.append(_norm01(tick_time, TICK_TIME_NORM_DEN))
        out.append(_norm01(n_post_tick_pref_viol, LOOKAHEAD_PREFILL_VIOL_DEN))
        out.append(_norm01(n_post_tick_dec_viol, LOOKAHEAD_DECODE_VIOL_DEN))

    return out


def extract_features_one_record(record: dict) -> np.ndarray:
    snapshot = record.get("simulator_snapshot") or {}
    stats = record.get("stats")
    sim_time = _safe_float(snapshot.get("time"))

    request_states = list((snapshot.get("request_states") or {}).values())
    active_id_set: set[int] = _stats_set(stats, "active_request_ids")
    if active_id_set:
        active_requests = [
            r for r in request_states
            if isinstance(r, dict) and _safe_int(r.get("id"), -1) in active_id_set
        ]
    else:
        active_requests = [r for r in request_states if isinstance(r, dict)]

    violated_set = _stats_set(stats, "violated_request_ids")
    per_req_prefill_late = _stats_dict(stats, "per_request_prefill_lateness")
    per_req_decode_late = _stats_dict(stats, "per_request_decode_lateness")
    decode_deadlines = _stats_dict(stats, "decode_next_deadline_by_id")

    prefill_reqs = [r for r in active_requests if _is_prefill(r)]
    decode_reqs = [r for r in active_requests if _is_decode(r)]

    num_prefill = len(prefill_reqs)
    num_decode = len(decode_reqs)
    num_active = len(active_requests)

    total_remaining_prefill = sum(
        max(0.0, _safe_float(r.get("remaining_prefill_tokens"))) for r in prefill_reqs
    )

    def _decode_remaining(r: dict) -> float:
        total_dec = max(0.0, _safe_float(r.get("num_decode_tokens")))
        proc_total = max(0.0, _safe_float(r.get("num_processed_tokens")))
        total_pref = max(0.0, _safe_float(r.get("num_prefill_tokens")))
        done_decode = max(0.0, proc_total - total_pref)
        return max(0.0, total_dec - done_decode)

    def _decode_done(r: dict) -> float:
        proc_total = max(0.0, _safe_float(r.get("num_processed_tokens")))
        total_pref = max(0.0, _safe_float(r.get("num_prefill_tokens")))
        return max(0.0, proc_total - total_pref)

    total_remaining_decode = sum(_decode_remaining(r) for r in decode_reqs)
    total_decode_generated_active = sum(
        max(0.0, _safe_float(r.get("num_processed_decode_tokens")))
        if "num_processed_decode_tokens" in r
        else _decode_done(r)
        for r in decode_reqs
    )

    active_ids = {_safe_int(r.get("id"), -1) for r in active_requests}
    num_violated_active = len(active_ids & violated_set)
    num_prefill_violated = sum(
        1 for r in prefill_reqs if _safe_int(r.get("id"), -1) in violated_set
    )
    num_decode_violated = sum(
        1 for r in decode_reqs if _safe_int(r.get("id"), -1) in violated_set
    )

    p_late_05_15 = 0
    p_late_15 = 0
    for r in prefill_reqs:
        rid = _safe_int(r.get("id"), -1)
        late = _safe_float(per_req_prefill_late.get(rid, 0.0))
        if late <= 0.0:
            arrived_at = _safe_float(r.get("arrived_at"))
            slo_t = _safe_float(r.get("prefill_slo_time"))
            late = max(0.0, sim_time - (arrived_at + slo_t))
        if late > F_NEAR_DROP_LATENESS_LOW_SEC and late < F_NEAR_DROP_LATENESS_HIGH_SEC:
            p_late_05_15 += 1
        elif late >= F_NEAR_DROP_LATENESS_HIGH_SEC:
            p_late_15 += 1

    d_late_05_15 = 0
    d_late_15 = 0
    for r in decode_reqs:
        rid = _safe_int(r.get("id"), -1)
        late_p = _safe_float(per_req_prefill_late.get(rid, 0.0))
        late_d = _safe_float(per_req_decode_late.get(rid, 0.0))
        late = max(late_p, late_d)
        if late > F_NEAR_DROP_LATENESS_LOW_SEC and late < F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_05_15 += 1
        elif late >= F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_15 += 1

    launch_count = 0.0
    launch_prefill = 0.0
    ewma = 0.0
    for item in (_stats_attr(stats, "recent_arrivals") or []):
        ts = None
        cnt = 0
        prefill = 0
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            ts = _safe_float(item[0])
            cnt = _safe_int(item[1])
            prefill = _safe_int(item[2])
        elif isinstance(item, dict):
            ts = _safe_float(item.get("timestamp", item.get("time")))
            cnt = _safe_int(item.get("count", item.get("requests", 0)))
            prefill = _safe_int(item.get("prefill_tokens", item.get("tokens", 0)))
        elif isinstance(item, (int, float)):
            ts = _safe_float(item)
            cnt = 1
        if ts is None:
            continue
        dt = max(0.0, sim_time - ts)
        if dt > F_LAUNCH_EWMA_WINDOW_SEC:
            continue
        launch_count += max(0.0, float(cnt))
        launch_prefill += max(0.0, float(prefill))
        ewma += max(0.0, float(cnt)) * math.exp(-F_LAUNCH_EWMA_ALPHA * dt)

    remaining_launch_request_headroom = max(0.0, MAX_REQUESTS_PER_LAUNCH_WINDOW - launch_count)
    remaining_launch_prefill_headroom = max(0.0, MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW - launch_prefill)

    decode_tokens_counted = _stats_dict(stats, "decode_tokens_counted")
    META_DECODE_CREDIT_BAL = -9_100_005
    decode_credit = max(0.0, _safe_float(decode_tokens_counted.get(META_DECODE_CREDIT_BAL, 0.0)))

    if prefill_reqs:
        min_prefill_slack = math.inf
        for r in prefill_reqs:
            arrived_at = _safe_float(r.get("arrived_at"))
            slo_t = _safe_float(r.get("prefill_slo_time"))
            slack = (arrived_at + slo_t) - sim_time
            if slack < min_prefill_slack:
                min_prefill_slack = slack
    else:
        min_prefill_slack = math.inf

    if decode_reqs:
        max_decode_context_tokens = 0
        for r in decode_reqs:
            num_pref = max(0, _safe_int(r.get("num_prefill_tokens")))
            proc_total = max(0, _safe_int(r.get("num_processed_tokens")))
            done_decode = max(0, proc_total - num_pref)
            ctx = num_pref + done_decode
            if ctx > max_decode_context_tokens:
                max_decode_context_tokens = ctx
        decode_time_at_max = _decode_time_for_tokens(int(max_decode_context_tokens))
    else:
        decode_time_at_max = 0.0

    if math.isinf(min_prefill_slack):
        edf_minus_batch_norm = 1.0
    else:
        edf_minus_batch_norm = max(0.0, min(1.0, min_prefill_slack - decode_time_at_max))

    n_active_edf_margin_gt_batch = 0
    for r in prefill_reqs:
        arrived_at = _safe_float(r.get("arrived_at"))
        slo_t = _safe_float(r.get("prefill_slo_time"))
        deadline = arrived_at + slo_t
        if (deadline - sim_time) - decode_time_at_max > 0.0:
            n_active_edf_margin_gt_batch += 1
    for r in decode_reqs:
        rid = _safe_int(r.get("id"), -1)
        arrived_at = _safe_float(r.get("arrived_at"))
        decode_slo = _safe_float(r.get("decode_slo_time"))
        deadline = _safe_float(decode_deadlines.get(rid, arrived_at + decode_slo))
        if (deadline - sim_time) - decode_time_at_max > 0.0:
            n_active_edf_margin_gt_batch += 1

    globals_v4 = [
        _norm01(num_prefill, F_ACTIVE_PREFILL_COUNT_DEN),
        _norm01(num_decode, F_ACTIVE_DECODE_COUNT_DEN),
        _norm01(num_active, F_ACTIVE_TOTAL_COUNT_DEN),
        _norm01(total_remaining_prefill, F_TOTAL_REMAINING_PREFILL_DEN),
        _norm01(total_remaining_decode, F_TOTAL_REMAINING_DECODE_DEN),
        _norm01(total_decode_generated_active, F_TOTAL_DECODE_GENERATED_ACTIVE_DEN),
        _norm01(num_violated_active, F_VIOLATED_COUNT_DEN),
        _norm01(p_late_05_15, F_PREFILL_NEAR_DROP_DEN),
        _norm01(p_late_15, F_PREFILL_NEAR_DROP_DEN),
        _norm01(d_late_05_15, F_DECODE_NEAR_DROP_DEN),
        _norm01(d_late_15, F_DECODE_NEAR_DROP_DEN),
        _norm01(launch_count, F_RECENT_LAUNCH_COUNT_DEN),
        _norm01(launch_prefill, F_RECENT_LAUNCH_PREFILL_DEN),
        _norm01(remaining_launch_request_headroom, MAX_REQUESTS_PER_LAUNCH_WINDOW),
        _norm01(remaining_launch_prefill_headroom, MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW),
        _norm01(ewma, F_RECENT_LAUNCH_COUNT_DEN),
        _norm01(decode_credit, F_DECODE_CREDIT_DEN),
        _norm01(num_prefill_violated, F_ACTIVE_PREFILL_COUNT_DEN),
        _norm01(num_decode_violated, F_ACTIVE_DECODE_COUNT_DEN),
        edf_minus_batch_norm,
        _norm01(n_active_edf_margin_gt_batch, F_ACTIVE_TOTAL_COUNT_DEN),
    ]
    assert len(globals_v4) == D_GLOBAL_V4

    enriched_prefill = []
    for r in prefill_reqs:
        rid = _safe_int(r.get("id"), -1)
        arrived_at = _safe_float(r.get("arrived_at"))
        slo_t = _safe_float(r.get("prefill_slo_time"))
        deadline = arrived_at + slo_t
        slack = deadline - sim_time
        slack_clamped = max(0.0, slack)
        late_stored = _safe_float(per_req_prefill_late.get(rid, 0.0))
        late_now = max(0.0, sim_time - deadline)
        late = max(late_stored, late_now)
        enriched_prefill.append({
            "rid": rid,
            "remaining": max(0.0, _safe_float(r.get("remaining_prefill_tokens"))),
            "total": max(0.0, _safe_float(r.get("num_prefill_tokens"))),
            "violated": rid in violated_set,
            "slack_clamped": slack_clamped,
            "lateness": late,
        })

    lookahead_block = _compute_lookahead_block(
        enriched_prefill, decode_reqs, decode_deadlines, sim_time, decode_time_at_max,
    )
    assert len(lookahead_block) == D_LOOKAHEAD

    all_violated = enriched_prefill and all(p["violated"] for p in enriched_prefill)
    if all_violated:
        enriched_prefill.sort(key=lambda p: p["rid"])
    else:
        enriched_prefill.sort(key=lambda p: (p["slack_clamped"], p["rid"]))

    selected_prefill = enriched_prefill[:N_PREFILL_SLOTS]

    prefill_block = np.zeros((N_PREFILL_SLOTS, D_PREFILL_PER_SLOT), dtype=np.float32)
    for i, p in enumerate(selected_prefill):
        prefill_block[i, 0] = 1.0
        prefill_block[i, 1] = _norm01(p["remaining"], 4096.0)
        prefill_block[i, 2] = _norm01(p["total"], 4096.0)
        prefill_block[i, 3] = 1.0 if p["violated"] else 0.0
        lb = _lateness_bucket_idx(p["lateness"])
        prefill_block[i, 4 + lb] = 1.0
        sb = _slack_bucket_idx(p["slack_clamped"])
        prefill_block[i, 4 + N_LATE_BUCKETS + sb] = 1.0

    non_violated_decode = [
        r for r in decode_reqs if _safe_int(r.get("id"), -1) not in violated_set
    ]
    non_violated_decode.sort(key=lambda r: _safe_int(r.get("id"), -1))
    n_nv = len(non_violated_decode)
    if n_nv > N_DECODE_SLOTS:
        seed_key = "|".join([
            str(record.get("root_id", -1)),
            f"{sim_time:.9f}",
            ",".join(str(_safe_int(r.get("id"), -1)) for r in non_violated_decode),
        ])
        idxs = _det_random_subset_indices(n_nv, N_DECODE_SLOTS, seed_key=seed_key)
        chosen = [non_violated_decode[i] for i in idxs]
    else:
        chosen = non_violated_decode

    decode_block = np.zeros((N_DECODE_SLOTS, D_DECODE_PER_SLOT), dtype=np.float32)
    for i, r in enumerate(chosen[:N_DECODE_SLOTS]):
        rem = _decode_remaining(r)
        done = _decode_done(r)
        decode_block[i, 0] = 1.0
        decode_block[i, 1] = _norm01(rem, DECODE_REMAINING_DEN)
        decode_block[i, 2] = 1.0 if done > 216.0 else 0.0
        decode_block[i, 3] = 1.0 if done > 512.0 else 0.0

    out = np.empty((D_TOTAL,), dtype=np.float32)
    out[:D_GLOBAL_V4] = np.asarray(globals_v4, dtype=np.float32)
    out[D_GLOBAL_V4:D_GLOBAL] = np.asarray(lookahead_block, dtype=np.float32)
    out[D_GLOBAL:D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT] = prefill_block.reshape(-1)
    out[D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT:] = decode_block.reshape(-1)
    return out


def feature_names() -> list[str]:
    names: list[str] = []
    names.extend([
        "g05_num_prefill", "g06_num_decode", "g07_num_active",
        "g08_total_remaining_prefill", "g09_total_remaining_decode",
        "g10_total_decode_generated_active", "g11_num_violated_active",
        "g12_p_late_05_15", "g13_p_late_15",
        "g14_d_late_05_15", "g15_d_late_15",
        "g16_launch_count", "g17_launch_prefill",
        "g18_remaining_launch_request_headroom", "g19_remaining_launch_prefill_headroom",
        "g20_ewma", "g21_decode_credit",
        "g_extra_prefill_violated_active", "g_extra_decode_violated_active",
        "g_edf_minus_batch_norm",
        "g_n_active_edf_margin_gt_batch_norm",
    ])
    for B in LOOKAHEAD_BUDGETS:
        names.append(f"la{B}_frac_picked")
        names.append(f"la{B}_tokens_norm")
        names.append(f"la{B}_tick_time_norm")
        names.append(f"la{B}_n_post_pref_viol")
        names.append(f"la{B}_n_post_dec_viol")
    for i in range(N_PREFILL_SLOTS):
        names.append(f"pref{i}_present")
        names.append(f"pref{i}_remaining_norm")
        names.append(f"pref{i}_total_norm")
        names.append(f"pref{i}_violated")
        for b in range(N_LATE_BUCKETS):
            names.append(f"pref{i}_late_b{b}")
        for b in range(N_SLACK_BUCKETS):
            names.append(f"pref{i}_slack_b{b}")
    for i in range(N_DECODE_SLOTS):
        names.append(f"dec{i}_present")
        names.append(f"dec{i}_remaining_norm")
        names.append(f"dec{i}_done_gt_216")
        names.append(f"dec{i}_done_gt_512")
    assert len(names) == D_TOTAL, f"feature names {len(names)} != D_TOTAL {D_TOTAL}"
    return names


def _process_shard(args: tuple[str, str, int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    shard_path, root_player_filter, start_idx, end_idx = args
    sys.path.insert(0, str(REPO_ROOT))
    records = torch.load(shard_path, weights_only=False)
    feats = []
    targets = []
    for rec in records:
        if root_player_filter and rec.get("root_player") != root_player_filter:
            continue
        feats.append(extract_features_one_record(rec))
        targets.append(_safe_float(rec.get("target_value", 0.0)))
    arr = np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, D_TOTAL), dtype=np.float32)
    tgt = np.asarray(targets, dtype=np.float32)
    return start_idx, len(feats), arr, tgt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--out-features", required=True)
    parser.add_argument("--out-targets", required=True)
    parser.add_argument("--out-meta", required=True)
    parser.add_argument("--root-player-filter", default="controller")
    parser.add_argument("--num-processes", type=int, default=32)
    parser.add_argument("--limit-shards", type=int, default=0)
    args = parser.parse_args()

    manifest_lines = Path(args.manifest).read_text().splitlines()
    if args.limit_shards > 0:
        manifest_lines = manifest_lines[: args.limit_shards]

    shard_jobs: list[tuple[str, str, int, int]] = []
    cursor = 0
    expected_total = 0
    for line in manifest_lines:
        m = json.loads(line)
        n = int(m.get("num_records", 0))
        n_kept_in_shard = int((m.get("player_counts") or {}).get(args.root_player_filter, n))
        shard_path = str(Path(args.shard_dir) / m["shard_path"])
        shard_jobs.append((shard_path, args.root_player_filter, cursor, cursor + n_kept_in_shard))
        cursor += n_kept_in_shard
        expected_total += n_kept_in_shard

    print(f"[buildv5] manifest={args.manifest} shards={len(shard_jobs)} expected_total_records={expected_total}")
    sys.stdout.flush()

    out_features = np.zeros((expected_total, D_TOTAL), dtype=np.float32)
    out_targets = np.zeros((expected_total,), dtype=np.float32)

    started = time.time()
    completed = 0
    last_log = started
    n_records_done = 0
    with ProcessPoolExecutor(max_workers=args.num_processes) as ex:
        futs = [ex.submit(_process_shard, job) for job in shard_jobs]
        for fut in as_completed(futs):
            start_idx, n_kept, arr, tgt = fut.result()
            out_features[start_idx:start_idx + n_kept] = arr
            out_targets[start_idx:start_idx + n_kept] = tgt
            n_records_done += n_kept
            completed += 1
            now = time.time()
            if now - last_log > 5.0 or completed == len(shard_jobs):
                rate = completed / max(1e-9, now - started)
                eta = (len(shard_jobs) - completed) / max(1e-9, rate)
                print(
                    f"[buildv5] shards_done={completed}/{len(shard_jobs)} "
                    f"records={n_records_done}/{expected_total} "
                    f"rate_shards_s={rate:.2f} eta_s={eta:.1f}",
                    flush=True,
                )
                last_log = now

    out_features_path = Path(args.out_features)
    out_features_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_features_path, out_features)
    np.save(args.out_targets, out_targets)
    meta = {
        "manifest": args.manifest,
        "shard_dir": args.shard_dir,
        "root_player_filter": args.root_player_filter,
        "num_records": expected_total,
        "feature_dim": D_TOTAL,
        "feature_names": feature_names(),
        "n_global": D_GLOBAL,
        "n_global_v4": D_GLOBAL_V4,
        "n_lookahead": D_LOOKAHEAD,
        "lookahead_budgets": LOOKAHEAD_BUDGETS,
        "n_prefill_slots": N_PREFILL_SLOTS,
        "d_prefill_per_slot": D_PREFILL_PER_SLOT,
        "n_decode_slots": N_DECODE_SLOTS,
        "d_decode_per_slot": D_DECODE_PER_SLOT,
        "slack_bucket_edges": PREFILL_SLACK_BUCKET_EDGES,
        "elapsed_s": time.time() - started,
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    print(f"[buildv5] DONE features={out_features.shape} targets={out_targets.shape}")


if __name__ == "__main__":
    main()
