"""Neural model-search backend for GV3 controller root values.

This module is intentionally separate from the production AlphaZero model. It
is used by ModelSearchBed experiments where the input is a stored root record
and the target is the controller-perspective value.

Important boundary: feature extraction here is state-only. It does not use the
stored best action, best reward, child cost, child time, or target value as
features, and it does not call MCTS/search to recompute labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from .model_search_duration_lookup import BATCH_DURATION_BY_SHAPE
except Exception:  # pragma: no cover - keeps imports safe before lookup generation.
    BATCH_DURATION_BY_SHAPE: dict[tuple[tuple[int, int, int], ...], float] = {}


MAX_REQUESTS = 8
REQUEST_FEATURE_DIM = 43
REQUEST_AGG_FEATURE_DIM = REQUEST_FEATURE_DIM * 4

DECODE_ACTION_TIME = 0.013306510239316609
DECODE_ACTION_TIME_P95 = 0.013416547224996217
DECODE_ACTION_TIME_MAX = 0.013534043940467921
SMALL_ACTION_TIME_PAD = 0.0005
PREFILL_ACTION_TIME_PAD = 0.00025
DECODE_PRIOR_MARGIN = 0.00005
META_DECODE_CREDIT_BAL = -9_100_005
DECODE_CREDIT_MINT_PER_PREFILL_COMPLETE = 216
DECODE_MEAN_TIME_BY_COUNT: dict[int, float] = {
    1: 0.013318445400138712,
    2: 0.013303527624545713,
    3: 0.01334938201719718,
    4: 0.013235282958736421,
    5: 0.013088235774012878,
    6: 0.012993111328978613,
    7: 0.012949050185374096,
}
PREFILL_TIME_BY_BUDGET: dict[int, float] = {
    0: 0.0,
    128: 0.015725797204323228,
    256: 0.023274675327417962,
    512: 0.03888623299112536,
    1024: 0.09850190759874299,
    1536: 0.14267405276361292,
    2048: 0.19613847773632437,
    3072: 0.28408680179190937,
    4096: 0.38669262762929524,
}
PREFILL_BUDGETS = (128, 256, 512, 1024, 1536, 2048, 3072, 4096)
HORIZON_STEPS = (
    0.0,
    0.004,
    0.008,
    0.012,
    DECODE_ACTION_TIME,
    DECODE_ACTION_TIME_P95,
    DECODE_ACTION_TIME_MAX,
    PREFILL_TIME_BY_BUDGET[128],
    PREFILL_TIME_BY_BUDGET[128] + SMALL_ACTION_TIME_PAD,
    PREFILL_TIME_BY_BUDGET[256],
    PREFILL_TIME_BY_BUDGET[256] + SMALL_ACTION_TIME_PAD,
    PREFILL_TIME_BY_BUDGET[512],
    0.08,
    PREFILL_TIME_BY_BUDGET[1024],
    PREFILL_TIME_BY_BUDGET[1536],
    PREFILL_TIME_BY_BUDGET[2048],
    PREFILL_TIME_BY_BUDGET[3072],
    PREFILL_TIME_BY_BUDGET[4096],
)
HORIZON_FEATURES_PER_STEP = 12
ACTION_RISK_FEATURE_DIM = 128
GLOBAL_FEATURE_DIM = 52
FEATURE_DIM = (
    MAX_REQUESTS * REQUEST_FEATURE_DIM
    + REQUEST_AGG_FEATURE_DIM
    + len(HORIZON_STEPS) * HORIZON_FEATURES_PER_STEP
    + ACTION_RISK_FEATURE_DIM
    + GLOBAL_FEATURE_DIM
)
ACTION_RISK_START = (
    MAX_REQUESTS * REQUEST_FEATURE_DIM
    + REQUEST_AGG_FEATURE_DIM
    + len(HORIZON_STEPS) * HORIZON_FEATURES_PER_STEP
)
ACTION_BEST_DELTA_FEATURE_INDEX = ACTION_RISK_START + 12
ACTION_NOMINAL_DELTA_FEATURE_INDEX = ACTION_RISK_START + 12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _bool_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _log1p_norm(value: float, denom: float) -> float:
    return math.log1p(max(0.0, float(value))) / float(denom)


def _decode_time_prior(decode_count: int) -> float:
    return float(DECODE_MEAN_TIME_BY_COUNT.get(int(decode_count), DECODE_ACTION_TIME)) + DECODE_PRIOR_MARGIN


def _decode_duration_for_infos(decode_infos: list[_ReqInfo]) -> float:
    """Deterministic decode-time surrogate from selected request shape.

    The GV3 random-forest predictor is deterministic but expensive to call in a
    hot feature loop. These piecewise constants are derived from the predictor's
    observed outputs for the fixed GV3 decode request patterns. They use only
    state/action shape: selected decode count and each request's processed-token
    position, not target values.
    """

    procs = sorted(max(0, int(x.processed_total)) for x in decode_infos)
    n = len(procs)
    if n <= 0:
        return 0.0

    if n == 1:
        p = procs[0]
        if p == 128:
            return 0.013461777940392494
        if p < 256:
            return 0.01331353560090065
        if p == 256:
            return 0.013302132487297058
        if p < 512:
            return 0.013289418071508408
        if p == 512:
            return 0.013322972692549229
        if p < 1024:
            return 0.013286666013300419
        if p == 1024:
            return 0.013359996490180492
        if p < 1536:
            return 0.013333812355995178
        if p == 1536:
            return 0.01330462284386158
        if p < 2048:
            return 0.013194260187447071
        if p == 2048:
            return 0.013330312445759773
        if p < 3072:
            return 0.013319696299731731
        if p < 4096:
            return 0.013391785323619843
        return 0.013353824615478516

    if n == 2:
        lo, hi = procs[0], procs[-1]
        if hi < 256:
            return 0.013368723914027214 if lo <= 128 and hi <= 129 else 0.013271993026137352
        if hi < 512:
            return 0.01335194706916809 if lo <= 256 and hi <= 257 else 0.013288245536386967
        if hi < 1024:
            return 0.013304236344993114 if lo <= 512 and hi <= 513 else 0.013252331875264645
        if hi < 1536:
            return 0.013297519646584988 if lo <= 1024 and hi <= 1025 else 0.013288311660289764
        if hi < 2048:
            return 0.013237782754004002
        if hi < 3072:
            return 0.013255215249955654
        return 0.013255215249955654

    if n == 3:
        hi = procs[-1]
        if hi < 256:
            return 0.01335507445037365
        if hi < 512:
            return 0.013416547328233719
        if hi < 1024:
            return 0.013297009281814098 if procs[0] >= 513 else 0.013274071738123894
        if hi < 1536:
            return 0.013369841501116753 if procs[0] >= 1025 else 0.013267614878714085
        return 0.013267614878714085

    if n == 4:
        hi = procs[-1]
        if hi < 256:
            return 0.013178576715290546
        if hi < 512:
            return 0.013178969733417034
        if hi < 1024:
            return 0.0131864408031106
        if hi < 1536:
            return 0.013208757154643536
        return 0.013208757154643536

    return float(DECODE_MEAN_TIME_BY_COUNT.get(int(n), DECODE_ACTION_TIME))


def _recent_arrival_features(stats: Any) -> tuple[float, float]:
    recent = list(getattr(stats, "recent_arrivals", []) or [])
    count = 0
    tokens = 0
    for item in recent:
        if isinstance(item, (tuple, list)):
            count += max(0, _safe_int(item[1], 0)) if len(item) > 1 else 1
            tokens += max(0, _safe_int(item[2], 0)) if len(item) > 2 else 0
        elif isinstance(item, dict):
            count += max(0, _safe_int(item.get("count", item.get("requests", 1)), 1))
            tokens += max(0, _safe_int(item.get("tokens", item.get("prefill_tokens", 0)), 0))
        else:
            count += 1
    return float(count), float(tokens)


def _decode_credit_balance(stats: Any) -> int:
    counted = getattr(stats, "decode_tokens_counted", {}) or {}
    return max(0, _safe_int(counted.get(META_DECODE_CREDIT_BAL, 0), 0))


@dataclass(frozen=True)
class _ReqInfo:
    rid: int
    present: float
    is_prefill: float
    is_decode: float
    prefill_total: int
    prefill_remaining: int
    prefill_done: int
    decode_total: int
    processed_total: int
    decode_done: int
    decode_remaining: int
    arrived_at: float
    queued_at: float
    prefill_deadline: float
    prefill_slack: float
    decode_deadline: float
    decode_slack: float
    prefill_slo: float
    decode_slo: float
    stat_pref_late: float
    stat_dec_late: float
    total_lateness: float
    violated: float
    dropped: float
    prefill_finalized: float
    scheduled: float
    preempted: float
    completed: float
    execution_time: float
    model_execution_time: float
    prefill_completed_at: float


def _prefill_eta(tokens: int) -> float:
    tokens_i = max(0, int(tokens))
    if tokens_i <= 0:
        return 0.0
    if tokens_i in PREFILL_TIME_BY_BUDGET:
        return float(PREFILL_TIME_BY_BUDGET[tokens_i])

    keys = sorted(PREFILL_TIME_BY_BUDGET)
    lower = max((x for x in keys if x <= tokens_i), default=0)
    upper = min((x for x in keys if x >= tokens_i), default=keys[-1])
    if lower == upper:
        return float(PREFILL_TIME_BY_BUDGET[lower])
    lo_v = float(PREFILL_TIME_BY_BUDGET[lower])
    hi_v = float(PREFILL_TIME_BY_BUDGET[upper])
    alpha = (tokens_i - lower) / float(upper - lower)
    return lo_v + alpha * (hi_v - lo_v)


def _ordered_prefill_infos(prefill_infos: list[_ReqInfo], heuristic: str) -> list[_ReqInfo]:
    if heuristic == "SJF":
        return sorted(prefill_infos, key=lambda x: (x.prefill_remaining, x.rid))
    if heuristic == "EDF":
        return sorted(prefill_infos, key=lambda x: (x.prefill_deadline, x.rid))
    if heuristic == "LST":
        return sorted(prefill_infos, key=lambda x: (x.prefill_slack - _prefill_eta(x.prefill_remaining), x.rid))
    if heuristic == "LJF":
        return sorted(prefill_infos, key=lambda x: (-x.prefill_remaining, x.rid))
    return sorted(prefill_infos, key=lambda x: int(x.rid))


def _allocate_prefill(
    prefill_infos: list[_ReqInfo],
    *,
    budget: int,
    heuristic: str,
) -> list[tuple[_ReqInfo, int]]:
    remaining = max(0, int(budget))
    allocations: list[tuple[_ReqInfo, int]] = []
    if remaining <= 0:
        return allocations
    for info in _ordered_prefill_infos(prefill_infos, heuristic):
        if remaining <= 0:
            break
        alloc = min(int(info.prefill_remaining), remaining)
        if alloc <= 0:
            continue
        allocations.append((info, int(alloc)))
        remaining -= int(alloc)
    return allocations


def _batch_shape_key(
    *,
    decode_infos: list[_ReqInfo],
    prefill_allocs: list[tuple[_ReqInfo, int]],
) -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, tuple[int, int, int]]] = []
    for info in decode_infos:
        rows.append((int(info.rid), (1, int(info.processed_total), 1)))
    for info, alloc in prefill_allocs:
        rows.append((int(info.rid), (0, int(info.processed_total), int(alloc))))
    return tuple(row for _rid, row in sorted(rows, key=lambda x: x[0]))


def _duration_for_batch_shape(
    *,
    decode_infos: list[_ReqInfo],
    prefill_allocs: list[tuple[_ReqInfo, int]],
) -> float | None:
    if not BATCH_DURATION_BY_SHAPE:
        return None
    key = _batch_shape_key(decode_infos=decode_infos, prefill_allocs=prefill_allocs)
    if not key:
        return 0.0
    value = BATCH_DURATION_BY_SHAPE.get(key)
    return None if value is None else float(value)


def _controller_batch_duration(
    *,
    decode_infos: list[_ReqInfo],
    prefill_allocs: list[tuple[_ReqInfo, int]],
) -> float:
    exact = _duration_for_batch_shape(decode_infos=decode_infos, prefill_allocs=prefill_allocs)
    if exact is not None:
        return float(exact)
    if decode_infos and not prefill_allocs:
        return _decode_duration_for_infos(decode_infos)
    if prefill_allocs:
        budget = sum(int(alloc) for _info, alloc in prefill_allocs)
        return _prefill_eta(int(budget)) + PREFILL_ACTION_TIME_PAD
    return 0.0


def _build_request_infos(record: dict[str, Any]) -> tuple[float, Any, list[_ReqInfo]]:
    snapshot = record["simulator_snapshot"]
    stats = record["stats"]
    sim_time = _safe_float(snapshot.get("time", 0.0))
    request_states = snapshot.get("request_states", {}) or {}
    active_ids = set(getattr(stats, "active_request_ids", set()) or set())
    decode_deadlines = getattr(stats, "decode_next_deadline_by_id", {}) or {}
    stat_prefill_lateness = getattr(stats, "per_request_prefill_lateness", {}) or {}
    stat_decode_lateness = getattr(stats, "per_request_decode_lateness", {}) or {}
    violated_ids = set(getattr(stats, "violated_request_ids", set()) or set())
    dropped_ids = set(getattr(stats, "dropped_request_ids", set()) or set())
    finalized_ids = set(getattr(stats, "prefill_lateness_finalized", set()) or set())

    infos: list[_ReqInfo] = []
    for raw_rid, req in request_states.items():
        rid = _safe_int(raw_rid, -1)
        if active_ids and rid not in active_ids:
            continue

        arrived_at = _safe_float(req.get("arrived_at", 0.0))
        queued_at = _safe_float(req.get("queued_at", arrived_at))
        prefill_total = max(0, _safe_int(req.get("num_prefill_tokens", 0)))
        prefill_remaining = max(0, _safe_int(req.get("remaining_prefill_tokens", 0)))
        prefill_done = max(0, prefill_total - prefill_remaining)
        decode_total = max(0, _safe_int(req.get("num_decode_tokens", 0)))
        processed_total = max(0, _safe_int(req.get("num_processed_tokens", 0)))
        decode_done = max(0, processed_total - prefill_total) if bool(req.get("is_prefill_complete", False)) else 0
        decode_done = min(decode_done, decode_total)
        decode_remaining = max(0, decode_total - decode_done)

        prefill_slo = _safe_float(req.get("prefill_slo_time", 0.0))
        decode_slo = _safe_float(req.get("decode_slo_time", 0.0))
        prefill_deadline = arrived_at + prefill_slo if prefill_slo >= 0.0 else 1e9
        prefill_slack = prefill_deadline - sim_time
        prefill_complete = bool(req.get("is_prefill_complete", False))
        is_prefill = 1.0 if (not prefill_complete and prefill_remaining > 0 and not bool(req.get("completed", False))) else 0.0
        is_decode = 1.0 if (prefill_complete and decode_remaining > 0 and not bool(req.get("completed", False))) else 0.0

        decode_deadline = _safe_float(decode_deadlines.get(rid, sim_time + decode_slo), sim_time + decode_slo)
        decode_slack = decode_deadline - sim_time
        stat_pref_late = _safe_float(stat_prefill_lateness.get(rid, 0.0))
        stat_dec_late = _safe_float(stat_decode_lateness.get(rid, 0.0))
        total_lateness = max(0.0, stat_pref_late) + max(0.0, stat_dec_late)

        infos.append(
            _ReqInfo(
                rid=rid,
                present=1.0,
                is_prefill=is_prefill,
                is_decode=is_decode,
                prefill_total=prefill_total,
                prefill_remaining=prefill_remaining,
                prefill_done=prefill_done,
                decode_total=decode_total,
                processed_total=processed_total,
                decode_done=decode_done,
                decode_remaining=decode_remaining,
                arrived_at=arrived_at,
                queued_at=queued_at,
                prefill_deadline=prefill_deadline,
                prefill_slack=prefill_slack,
                decode_deadline=decode_deadline,
                decode_slack=decode_slack,
                prefill_slo=prefill_slo,
                decode_slo=decode_slo,
                stat_pref_late=stat_pref_late,
                stat_dec_late=stat_dec_late,
                total_lateness=total_lateness,
                violated=1.0 if rid in violated_ids else 0.0,
                dropped=1.0 if rid in dropped_ids else 0.0,
                prefill_finalized=1.0 if rid in finalized_ids else 0.0,
                scheduled=_bool_float(req.get("scheduled", False)),
                preempted=_bool_float(req.get("preempted", False)),
                completed=_bool_float(req.get("completed", False)),
                execution_time=_safe_float(req.get("execution_time", 0.0)),
                model_execution_time=_safe_float(req.get("model_execution_time", 0.0)),
                prefill_completed_at=_safe_float(req.get("prefill_completed_at", 0.0)),
            )
        )
    return sim_time, stats, infos


def _request_row(info: _ReqInfo, sim_time: float) -> list[float]:
    pref_late_now = max(0.0, -float(info.prefill_slack))
    dec_late_now = max(0.0, -float(info.decode_slack))
    min_slack = min(float(info.prefill_slack), float(info.decode_slack))
    return [
        info.present,
        info.is_prefill,
        info.is_decode,
        info.scheduled,
        info.preempted,
        info.completed,
        info.violated,
        info.dropped,
        info.prefill_finalized,
        _log1p_norm(info.prefill_total, 9.0),
        _log1p_norm(info.prefill_remaining, 9.0),
        _log1p_norm(info.prefill_done, 9.0),
        _log1p_norm(info.decode_total, 9.0),
        _log1p_norm(info.decode_done, 9.0),
        _log1p_norm(info.decode_remaining, 9.0),
        _clip(sim_time - info.arrived_at, 0.0, 5.0),
        _clip(sim_time - info.queued_at, 0.0, 5.0),
        info.prefill_slack,
        _clip(info.prefill_slack, -1.0, 1.0),
        pref_late_now,
        info.decode_slack,
        _clip(info.decode_slack, -1.0, 1.0),
        dec_late_now,
        min_slack,
        info.prefill_slo,
        info.decode_slo,
        min(info.execution_time, 2.0),
        min(info.model_execution_time, 2.0),
        min(info.prefill_completed_at, 5.0),
        min(info.stat_pref_late, 20.0),
        min(info.stat_dec_late, 20.0),
        min(info.total_lateness, 20.0),
        _clip(2.0 - info.total_lateness, -5.0, 2.0),
        1.0 if info.prefill_slack <= DECODE_ACTION_TIME else 0.0,
        1.0 if info.prefill_slack <= PREFILL_TIME_BY_BUDGET[128] else 0.0,
        1.0 if info.prefill_slack <= PREFILL_TIME_BY_BUDGET[128] + SMALL_ACTION_TIME_PAD else 0.0,
        1.0 if info.prefill_slack <= PREFILL_TIME_BY_BUDGET[256] else 0.0,
        1.0 if info.decode_slack <= DECODE_ACTION_TIME_MAX else 0.0,
        1.0 if 0 < info.prefill_remaining <= 128 else 0.0,
        1.0 if 0 < info.prefill_remaining <= 256 else 0.0,
        1.0 if 0 < info.prefill_remaining <= 512 else 0.0,
        1.0 if 0 < info.prefill_remaining <= 1024 else 0.0,
        float(info.rid % 16) / 16.0,
    ]


def _future_delta(
    infos: list[_ReqInfo],
    *,
    sim_time: float,
    dt: float,
    decode_selected: bool,
    decode_selected_ids: set[int] | None = None,
    evicted_ids: set[int] | None = None,
) -> tuple[float, float, float, float, float]:
    """Approximate immediate cost increase after an action duration.

    This mirrors the stats update shape using only the stored root state and an
    action duration. It deliberately avoids action search and simulator replay.
    """

    del sim_time
    evicted = evicted_ids or set()
    prefill_delta = 0.0
    decode_delta = 0.0
    new_violations = 0.0
    near_drop_count = 0.0
    max_total_lateness = 0.0

    for info in infos:
        if info.rid in evicted:
            continue

        pref_after = float(info.stat_pref_late)
        dec_after = float(info.stat_dec_late)
        if info.prefill_finalized < 0.5 and info.is_prefill > 0.5:
            late = max(0.0, float(dt) - float(info.prefill_slack))
            inc = max(0.0, late - float(info.stat_pref_late))
            prefill_delta += inc
            pref_after = max(pref_after, late)

        selected_for_decode = bool(decode_selected) and (
            decode_selected_ids is None or int(info.rid) in decode_selected_ids
        )
        if selected_for_decode and info.is_decode > 0.5:
            late = max(0.0, float(dt) - float(info.decode_slack))
            decode_delta += late
            dec_after = dec_after + late

        total_after = max(0.0, pref_after) + max(0.0, dec_after)
        if total_after > 0.0 and info.violated < 0.5:
            new_violations += 1.0
        if total_after >= 1.5:
            near_drop_count += 1.0
        max_total_lateness = max(max_total_lateness, total_after)

    total_delta = float(prefill_delta + decode_delta + new_violations)
    return total_delta, prefill_delta, decode_delta, new_violations, max_total_lateness + near_drop_count


def _eviction_candidate_ids(infos: list[_ReqInfo], rule: str) -> set[int]:
    prefill = [x for x in infos if x.is_prefill > 0.5]
    decode = [x for x in infos if x.is_decode > 0.5]
    if rule == "evict_none":
        return set()
    if rule == "evict_largest_prefill" and prefill:
        return {max(prefill, key=lambda x: (x.prefill_remaining, -x.rid)).rid}
    if rule == "evict_earliest_prefill_deadline" and prefill:
        return {min(prefill, key=lambda x: (x.prefill_deadline, x.rid)).rid}
    if rule == "evict_prefill_missed_deadline":
        return {x.rid for x in prefill if x.stat_pref_late > 0.0 or x.prefill_slack < 0.0}
    if rule == "evict_prefill_lateness_over_0p5":
        return {x.rid for x in prefill if x.stat_pref_late > 0.5}
    if rule == "evict_longest_decode" and decode:
        return {max(decode, key=lambda x: (x.decode_remaining, -x.rid)).rid}
    if rule == "evict_decode_lateness_over_0p5":
        return {x.rid for x in decode if x.stat_dec_late > 0.5}
    if rule == "evict_prefill_highest_lateness" and prefill:
        x = max(prefill, key=lambda y: (y.stat_pref_late, -y.rid))
        return {x.rid} if x.stat_pref_late > 0.0 else set()
    if rule == "evict_decode_highest_lateness" and decode:
        x = max(decode, key=lambda y: (y.stat_dec_late, -y.rid))
        return {x.rid} if x.stat_dec_late > 0.0 else set()
    return set()


def _action_risk_features(infos: list[_ReqInfo], sim_time: float, stats: Any) -> list[float]:
    del sim_time
    prefill_infos = [x for x in infos if x.is_prefill > 0.5]
    decode_infos = sorted([x for x in infos if x.is_decode > 0.5], key=lambda x: int(x.rid))
    total_prefill = sum(int(x.prefill_remaining) for x in prefill_infos)
    decode_count = len(decode_infos)
    decode_credit = _decode_credit_balance(stats)
    sched_decode_count = min(int(decode_count), int(decode_credit))
    decode_selected_ids = {int(x.rid) for x in decode_infos[:sched_decode_count]}

    def add_action_row(
        *,
        branch_infos: list[_ReqInfo],
        decode_branch_infos: list[_ReqInfo],
        prefill_allocs: list[tuple[_ReqInfo, int]],
        evicted_ids: set[int],
        branch_code: float,
    ) -> None:
        if not decode_branch_infos and not prefill_allocs:
            return
        dt = _controller_batch_duration(
            decode_infos=decode_branch_infos,
            prefill_allocs=prefill_allocs,
        )
        decode_branch_ids = {int(x.rid) for x in decode_branch_infos}
        delta, pref, dec, viol, dropish = _future_delta(
            infos,
            sim_time=0.0,
            dt=dt,
            decode_selected=bool(decode_branch_infos),
            decode_selected_ids=decode_branch_ids,
            evicted_ids=evicted_ids,
        )
        evicted_lateness = sum(x.total_lateness for x in infos if x.rid in evicted_ids)
        # GV3 drop semantics remove the request's previously accrued lateness
        # and violation before adding the terminal drop cost.
        drop_penalty = 0.0
        for info in infos:
            if int(info.rid) not in evicted_ids:
                continue
            drop_penalty += 3.0 - max(0.0, float(info.total_lateness)) - (1.0 if info.violated > 0.5 else 0.0)
        budget = sum(int(alloc) for _info, alloc in prefill_allocs)
        candidate_rows.append(
            [
                1.0,
                dt,
                delta + drop_penalty,
                pref,
                dec,
                viol,
                dropish,
                branch_code + min(evicted_lateness, 20.0) / 200.0 + float(budget) / 40960.0,
            ]
        )

    candidate_rows: list[list[float]] = []
    evict_rules = (
        "evict_none",
        "evict_largest_prefill",
        "evict_earliest_prefill_deadline",
        "evict_prefill_missed_deadline",
        "evict_prefill_lateness_over_0p5",
        "evict_longest_decode",
        "evict_decode_lateness_over_0p5",
        "evict_prefill_highest_lateness",
        "evict_decode_highest_lateness",
    )
    heuristics = ("SJF", "EDF", "LST", "LJF")

    for rule_idx, rule in enumerate(evict_rules):
        evicted = _eviction_candidate_ids(infos, rule)
        if rule != "evict_none" and not evicted:
            continue
        branch_prefill_infos = [x for x in prefill_infos if int(x.rid) not in evicted]
        branch_decode_infos_all = [x for x in decode_infos if int(x.rid) not in evicted]
        branch_decode_count = min(len(branch_decode_infos_all), max(0, int(decode_credit)))
        branch_decode_infos = branch_decode_infos_all[:branch_decode_count]
        branch_total_prefill = sum(int(x.prefill_remaining) for x in branch_prefill_infos)
        branch_code = float(rule_idx) / 16.0

        # budget=0 is valid only when decode work remains; this matches the
        # sampler's "decode-only" branch.
        add_action_row(
            branch_infos=branch_prefill_infos,
            decode_branch_infos=branch_decode_infos,
            prefill_allocs=[],
            evicted_ids=evicted,
            branch_code=branch_code,
        )

        if branch_total_prefill <= 0:
            continue
        for budget in PREFILL_BUDGETS:
            if int(budget) > int(branch_total_prefill):
                continue
            for heur_idx, heuristic in enumerate(heuristics):
                prefill_allocs = _allocate_prefill(
                    branch_prefill_infos,
                    budget=int(budget),
                    heuristic=str(heuristic),
                )
                add_action_row(
                    branch_infos=branch_prefill_infos,
                    decode_branch_infos=branch_decode_infos,
                    prefill_allocs=prefill_allocs,
                    evicted_ids=evicted,
                    branch_code=branch_code + float(heur_idx) / 64.0,
                )

    if not candidate_rows:
        candidate_rows = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    rows = sorted(candidate_rows, key=lambda x: (x[2], x[1], x[7]))
    nominal_best = rows[0]
    worst = rows[-1]
    deltas = [x[2] for x in rows]
    durations = [x[1] for x in rows]

    robust_rows: list[list[float]] = []
    if sched_decode_count > 0:
        dt = _controller_batch_duration(
            decode_infos=decode_infos[:sched_decode_count],
            prefill_allocs=[],
        )
        delta, pref, dec, viol, dropish = _future_delta(
            infos,
            sim_time=0.0,
            dt=dt,
            decode_selected=True,
            decode_selected_ids=decode_selected_ids,
        )
        robust_rows.append([1.0, dt, delta, pref, dec, viol, dropish, 0.0])
    for budget in PREFILL_BUDGETS:
        if total_prefill <= 0 or int(budget) > int(total_prefill):
            continue
        prefill_allocs = _allocate_prefill(prefill_infos, budget=int(budget), heuristic="SJF")
        dt = _controller_batch_duration(
            decode_infos=decode_infos[:sched_decode_count],
            prefill_allocs=prefill_allocs,
        )
        delta, pref, dec, viol, dropish = _future_delta(
            infos,
            sim_time=0.0,
            dt=dt,
            decode_selected=sched_decode_count > 0,
            decode_selected_ids=decode_selected_ids,
        )
        robust_rows.append([1.0, dt, delta, pref, dec, viol, dropish, float(budget) / 4096.0])
    robust_best = sorted(robust_rows, key=lambda x: (x[2], x[1], x[7]))[0] if robust_rows else nominal_best

    features: list[float] = [
        len(rows) / 64.0,
        decode_count / 16.0,
        total_prefill / 4096.0,
        robust_best[1],
        robust_best[2] / 8.0,
        robust_best[3] / 8.0,
        robust_best[4] / 8.0,
        robust_best[5] / 8.0,
        robust_best[6] / 8.0,
        worst[1],
        worst[2] / 8.0,
        nominal_best[1],
        nominal_best[2] / 8.0,
        min(deltas),
        max(deltas),
        sum(deltas) / len(deltas),
        min(durations),
        max(durations),
        1.0 if robust_best[2] <= 1e-9 else 0.0,
        1.0 if robust_best[2] >= 1.0 else 0.0,
        1.0 if decode_count > 0 and robust_best[1] <= DECODE_ACTION_TIME_MAX + 1e-9 else 0.0,
        1.0 if total_prefill > 0 and decode_count <= 0 else 0.0,
    ]

    # Include the best few candidates explicitly. This gives the NN access to
    # action-feasibility/risk structure without using the stored selected action.
    for row in rows[:8]:
        features.extend(
            [
                row[0],
                row[1],
                row[2] / 8.0,
                row[3] / 8.0,
                row[4] / 8.0,
                row[5] / 8.0,
                row[6] / 8.0,
                row[7],
            ]
        )
    while len(features) < ACTION_RISK_FEATURE_DIM:
        features.append(0.0)
    return features[:ACTION_RISK_FEATURE_DIM]


def _horizon_features(infos: list[_ReqInfo], stats: Any) -> list[float]:
    out: list[float] = []
    decode_infos = sorted([x for x in infos if x.is_decode > 0.5], key=lambda x: int(x.rid))
    sched_decode_count = min(len(decode_infos), _decode_credit_balance(stats))
    decode_selected_ids = {int(x.rid) for x in decode_infos[:sched_decode_count]}
    for dt in HORIZON_STEPS:
        delta_decode, pref_d, dec_d, viol_d, dropish_d = _future_delta(
            infos,
            sim_time=0.0,
            dt=float(dt),
            decode_selected=True,
            decode_selected_ids=decode_selected_ids,
        )
        delta_nodecode, pref_nd, _dec_nd, viol_nd, dropish_nd = _future_delta(
            infos,
            sim_time=0.0,
            dt=float(dt),
            decode_selected=False,
        )
        late_pref_count = 0.0
        safe_pref_count = 0.0
        for info in infos:
            if info.is_prefill > 0.5:
                if info.prefill_slack <= float(dt):
                    late_pref_count += 1.0
                else:
                    safe_pref_count += 1.0
        out.extend(
            [
                float(dt),
                delta_decode / 8.0,
                delta_nodecode / 8.0,
                pref_d / 8.0,
                pref_nd / 8.0,
                dec_d / 8.0,
                viol_d / 8.0,
                viol_nd / 8.0,
                dropish_d / 8.0,
                dropish_nd / 8.0,
                late_pref_count / 16.0,
                safe_pref_count / 16.0,
            ]
        )
    return out


def root_record_to_features(record: dict[str, Any]) -> list[float]:
    """Build state-only numeric features for one stored controller root."""

    snapshot = record["simulator_snapshot"]
    sim_time, stats, infos = _build_request_infos(record)

    sort_keyed = sorted(
        infos,
        key=lambda x: (
            min(x.prefill_slack if x.is_prefill > 0.5 else 1e9, x.decode_slack if x.is_decode > 0.5 else 1e9),
            x.prefill_remaining,
            x.rid,
        ),
    )
    flattened: list[float] = []
    for info in sort_keyed[:MAX_REQUESTS]:
        flattened.extend(_request_row(info, sim_time))
    while len(flattened) < MAX_REQUESTS * REQUEST_FEATURE_DIM:
        flattened.extend([0.0] * REQUEST_FEATURE_DIM)

    rows = [_request_row(info, sim_time) for info in infos]
    if rows:
        aggregates: list[float] = []
        for col in zip(*rows):
            values = list(col)
            aggregates.extend([sum(values), sum(values) / len(values), min(values), max(values)])
    else:
        aggregates = [0.0] * REQUEST_AGG_FEATURE_DIM

    prefill_infos = [x for x in infos if x.is_prefill > 0.5]
    decode_infos = [x for x in infos if x.is_decode > 0.5]
    prefill_slacks = [x.prefill_slack for x in prefill_infos]
    decode_slacks = [x.decode_slack for x in decode_infos]
    total_prefill = sum(int(x.prefill_remaining) for x in prefill_infos)
    active_total_lateness = sum(float(x.total_lateness) for x in infos)
    dropped_count = len(getattr(stats, "dropped_request_ids", set()) or set())
    stopped_count = len(getattr(stats, "stopped_decode_request_ids", set()) or set())
    min_pref_slack = min(prefill_slacks) if prefill_slacks else 10.0
    min_dec_slack = min(decode_slacks) if decode_slacks else 10.0
    safe_decode_gap = min_pref_slack - _decode_time_prior(len(decode_infos)) if decode_infos else -10.0
    safe_128_gap = min_pref_slack - (PREFILL_TIME_BY_BUDGET[128] + PREFILL_ACTION_TIME_PAD) if prefill_infos else 10.0
    recent_count, recent_tokens = _recent_arrival_features(stats)
    decode_credit = _decode_credit_balance(stats)
    sched_decode_count = min(len(decode_infos), int(decode_credit))
    unsched_decode_count = max(0, len(decode_infos) - int(sched_decode_count))

    horizon = _horizon_features(infos, stats)
    action_risk = _action_risk_features(infos, sim_time, stats)
    parent_cost = _safe_float(getattr(stats, "slo_violations", 0)) + _safe_float(
        getattr(stats, "slo_lateness_sum", 0.0)
    )
    global_features = [
        min(sim_time, 5.0),
        _log1p_norm(_safe_int(record.get("history_hops", 0)), 6.0),
        _log1p_norm(_safe_int(record.get("root_depth", 0)), 8.0),
        len(infos) / 16.0,
        len(prefill_infos) / 16.0,
        len(decode_infos) / 16.0,
        len(snapshot.get("waiting_ids", []) or []) / 16.0,
        len(snapshot.get("running_ids", []) or []) / 16.0,
        _safe_float(getattr(stats, "slo_violations", 0)) / 20.0,
        min(_safe_float(getattr(stats, "slo_lateness_sum", 0.0)), 20.0) / 20.0,
        min(parent_cost, 40.0) / 40.0,
        _safe_float(getattr(stats, "requests_generated", 0)) / 100.0,
        _safe_float(getattr(stats, "requests_completed", 0)) / 100.0,
        _safe_float(getattr(stats, "last_prefill_batch_time", 0.0)),
        _safe_float(getattr(stats, "transition_discount_time", 0.0)),
        _safe_float(getattr(stats, "transition_final_time", 0.0)),
        recent_count / 16.0,
        _log1p_norm(recent_tokens, 10.0),
        total_prefill / 4096.0,
        _log1p_norm(total_prefill, 10.0),
        min_pref_slack,
        _clip(min_pref_slack, -1.0, 1.0),
        min_dec_slack,
        _clip(min_dec_slack, -1.0, 1.0),
        safe_decode_gap,
        _clip(safe_decode_gap, -1.0, 1.0),
        safe_128_gap,
        _clip(safe_128_gap, -1.0, 1.0),
        1.0 if decode_infos and safe_decode_gap >= 0.0 else 0.0,
        1.0 if decode_infos and safe_decode_gap < 0.0 else 0.0,
        1.0 if prefill_infos and safe_128_gap >= 0.0 else 0.0,
        1.0 if prefill_infos and safe_128_gap < 0.0 else 0.0,
        sum(1 for x in prefill_infos if x.prefill_slack <= DECODE_ACTION_TIME_MAX) / 16.0,
        sum(1 for x in prefill_infos if x.prefill_slack <= PREFILL_TIME_BY_BUDGET[128] + PREFILL_ACTION_TIME_PAD) / 16.0,
        sum(1 for x in prefill_infos if x.prefill_slack <= PREFILL_TIME_BY_BUDGET[256]) / 16.0,
        sum(1 for x in decode_infos if x.decode_slack <= DECODE_ACTION_TIME) / 16.0,
        min((x.total_lateness for x in infos), default=0.0),
        max((x.total_lateness for x in infos), default=0.0),
        sum(x.total_lateness for x in infos) / max(1, len(infos)),
        sum(1 for x in infos if x.total_lateness >= 1.5) / 16.0,
        int(decode_credit) / float(DECODE_CREDIT_MINT_PER_PREFILL_COMPLETE),
        int(sched_decode_count) / 16.0,
        int(unsched_decode_count) / 16.0,
        _clip(min_pref_slack - DECODE_ACTION_TIME_MAX, -0.1, 0.1),
        _clip(min_pref_slack - PREFILL_TIME_BY_BUDGET[128], -0.1, 0.1),
        _clip(min_pref_slack - (PREFILL_TIME_BY_BUDGET[256] + PREFILL_ACTION_TIME_PAD), -0.1, 0.1),
        _clip(min_dec_slack - DECODE_ACTION_TIME, -0.1, 0.1),
        sum(
            1
            for x in prefill_infos
            if min(
                abs(x.prefill_slack - DECODE_ACTION_TIME_MAX),
                abs(x.prefill_slack - PREFILL_TIME_BY_BUDGET[128]),
                abs(x.prefill_slack - PREFILL_TIME_BY_BUDGET[256]),
            )
            <= 0.001
        )
        / 16.0,
        int(dropped_count) / 16.0,
        int(stopped_count) / 16.0,
        min(active_total_lateness, 20.0) / 20.0,
        min(max(0.0, _safe_float(getattr(stats, "slo_lateness_sum", 0.0)) - active_total_lateness), 20.0) / 20.0,
    ]

    features = flattened + aggregates + horizon + action_risk + global_features
    if len(features) != FEATURE_DIM:
        raise RuntimeError(f"feature dim mismatch: {len(features)} != {FEATURE_DIM}")
    return features


def records_to_feature_tensor(records: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([root_record_to_features(record) for record in records], dtype=torch.float32)


def records_to_target_tensor(records: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([float(record["target_value"]) for record in records], dtype=torch.float32)


def records_to_duration_tensor(records: list[dict[str, Any]]) -> torch.Tensor:
    durations = []
    for record in records:
        sim_time = _safe_float(record.get("simulator_snapshot", {}).get("time", 0.0))
        child_time = _safe_float(record.get("best_child_time", sim_time), sim_time)
        durations.append(max(0.0, child_time - sim_time))
    return torch.tensor(durations, dtype=torch.float32)


def error_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    err = prediction.view(-1).detach().cpu() - target.view(-1).detach().cpu()
    abs_err = torch.abs(err)
    mse = float(torch.mean(err * err).item())
    if abs_err.numel() == 0:
        p50 = p95 = max_abs = float("nan")
    else:
        p50 = float(torch.quantile(abs_err, 0.50).item())
        p95 = float(torch.quantile(abs_err, 0.95).item())
        max_abs = float(torch.max(abs_err).item())
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(torch.mean(abs_err).item()),
        "p50_abs_error": p50,
        "p95_abs_error": p95,
        "max_abs_error": max_abs,
    }


class HorizonValueNet(nn.Module):
    """Compact bootstrap-compatible state-value network.

    The state-derived immediate-risk estimate is a basis signal and the network
    learns the future residual. This is the shape we need for deeper targets:

        reward + discount * bootstrap_value(child_state)

    For the depth-1/no-bootstrap dataset, the correct residual should be near
    zero. For deeper datasets, the residual can represent the discounted
    bootstrap contribution without changing the feature contract.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 112,
        dropout_p: float = 0.02,
        residual_bound: float = 12.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        self.residual_bound = float(residual_bound)
        self.register_buffer("feature_mean", torch.zeros(self.input_dim), persistent=True)
        self.register_buffer("feature_std", torch.ones(self.input_dim), persistent=True)
        self.in_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout_p)),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout_p)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 80),
            nn.LayerNorm(80),
            nn.SiLU(),
        )
        self.residual_head = nn.Linear(80, 1)
        self.duration_head = nn.Linear(80, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def set_normalizer(self, features: torch.Tensor) -> None:
        mean = features.mean(dim=0)
        std = features.std(dim=0).clamp_min(1e-6)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def forward_parts(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (features - self.feature_mean) / self.feature_std
        h = self.in_proj(x)
        h = h + self.block1(h)
        h = self.block2(h)
        residual = float(self.residual_bound) * torch.tanh(self.residual_head(h)).view(-1)
        duration = 0.5 * torch.sigmoid(self.duration_head(h)).view(-1)
        nominal_delta = features[:, ACTION_NOMINAL_DELTA_FEATURE_INDEX].view(-1) * 8.0
        prior_value = -nominal_delta
        value_raw = prior_value + residual
        value = torch.minimum(value_raw, torch.zeros_like(value_raw))
        return value, residual, prior_value, duration

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value, _residual, _prior_value, _duration = self.forward_parts(features)
        return value


@dataclass
class ModelSearchDataset:
    features: torch.Tensor
    targets: torch.Tensor

    def __len__(self) -> int:
        return int(self.targets.shape[0])
