from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from ..features import (
    _arrived_at,
    _build_request_lookup,
    _decode_slo,
    _prefill_complete,
    _prefill_slo,
    _remaining_decode,
    _remaining_prefill,
    _req_completed,
)
from ..lp.features import (
    RequestAggregate,
    extract_features_for_set,
    update_request_aggregate,
)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _request_rows_from_live_state(env: Any, state: Any) -> List[Dict[str, Any]]:
    lookup = _build_request_lookup(env, state)
    sim_time = _safe_float(getattr(state.simulator, "_time", 0.0), 0.0)
    stats = getattr(state, "stats", None)
    decode_deadline_by_id = getattr(stats, "decode_next_deadline_by_id", {}) if stats is not None else {}

    rows: List[Dict[str, Any]] = []
    for rid in sorted(int(x) for x in lookup.keys()):
        req = lookup[rid]
        completed = bool(_req_completed(req))
        prefill_complete = bool(_prefill_complete(req))
        prefill_remaining = float(_remaining_prefill(req))
        decode_remaining = float(_remaining_decode(req))

        arrived_at = float(_arrived_at(req))
        prefill_slo = float(_prefill_slo(req))
        decode_slo = float(_decode_slo(req))

        prefill_deadline = arrived_at + prefill_slo if prefill_slo > 0.0 else 0.0
        if isinstance(decode_deadline_by_id, dict) and int(rid) in decode_deadline_by_id:
            decode_deadline = _safe_float(decode_deadline_by_id[int(rid)], 0.0)
        elif decode_slo > 0.0:
            decode_deadline = sim_time + decode_slo
        else:
            decode_deadline = 0.0

        prefill_lateness_now = max(0.0, sim_time - prefill_deadline) if prefill_deadline > 0.0 else 0.0
        decode_lateness_now = max(0.0, sim_time - decode_deadline) if decode_deadline > 0.0 else 0.0
        total_lateness_now = prefill_lateness_now + decode_lateness_now

        rows.append(
            {
                "request_id": int(rid),
                "prefill_tokens_remaining": float(prefill_remaining),
                "decode_tokens_remaining": float(decode_remaining),
                "completed": bool(completed),
                "prefill_complete": bool(prefill_complete),
                "prefill_lateness_now": float(prefill_lateness_now),
                "decode_lateness_now": float(decode_lateness_now),
                "prefill_violated_now": bool(prefill_lateness_now > 0.0),
                "decode_violated_now": bool(decode_lateness_now > 0.0),
                "violated_now": bool(total_lateness_now > 0.0),
                "total_lateness_now": float(total_lateness_now),
                "prefill_deadline": float(prefill_deadline),
                "decode_deadline": float(decode_deadline),
            }
        )
    return rows


def extract_lp_baseline_v1_feature_map(env: Any, state: Any) -> Dict[str, float]:
    sim_time = _safe_float(getattr(state.simulator, "_time", 0.0), 0.0)
    describe = {}
    try:
        describe = dict(env.describe_state(state))
    except Exception:
        describe = {}

    req_rows = _request_rows_from_live_state(env, state)
    agg = RequestAggregate()
    for row in req_rows:
        update_request_aggregate(agg, row, sim_time=sim_time)

    state_row = {
        "state_id": "",
        "player_to_act": "",
        "sim_time": sim_time,
        "requests_in_system": _safe_int(describe.get("requests_in_system", len(req_rows)), len(req_rows)),
    }
    return extract_features_for_set(
        feature_set="baseline_v1",
        state_row=state_row,
        req_agg=agg,
    )


def extract_lp_baseline_v1_vector(
    env: Any,
    state: Any,
    *,
    feature_names: List[str],
) -> Tuple[np.ndarray, Dict[str, float]]:
    fmap = extract_lp_baseline_v1_feature_map(env, state)
    vec = np.asarray([float(fmap.get(name, 0.0)) for name in feature_names], dtype=np.float64)
    return vec, fmap

