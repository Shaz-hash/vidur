from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _req_id(req: Any) -> int:
    return _safe_int(getattr(req, "id", None), -1)


def _req_completed(req: Any) -> bool:
    return bool(getattr(req, "completed", False))


def _prefill_complete(req: Any) -> bool:
    return bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)))


def _total_prefill(req: Any) -> int:
    return _safe_int(getattr(req, "_num_prefill_tokens", getattr(req, "num_prefill_tokens", 0)), 0)


def _done_prefill(req: Any) -> int:
    return _safe_int(getattr(req, "num_processed_prefill_tokens", 0), 0)


def _total_decode(req: Any) -> int:
    return _safe_int(getattr(req, "_num_decode_tokens", getattr(req, "num_decode_tokens", 0)), 0)


def _done_decode(req: Any) -> int:
    return _safe_int(getattr(req, "num_processed_decode_tokens", 0), 0)


def _remaining_prefill(req: Any) -> int:
    return max(0, _total_prefill(req) - _done_prefill(req))


def _remaining_decode(req: Any) -> int:
    return max(0, _total_decode(req) - _done_decode(req))


def _prefill_slo(req: Any) -> float:
    return _safe_float(getattr(req, "_prefill_slo_time", getattr(req, "prefill_slo_time", 0.0)), 0.0)


def _decode_slo(req: Any) -> float:
    return _safe_float(getattr(req, "_decode_slo_time", getattr(req, "decode_slo_time", 0.0)), 0.0)


def _arrived_at(req: Any) -> float:
    return _safe_float(getattr(req, "queued_at", getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0))), 0.0)


def _is_prefill_active(req: Any) -> bool:
    if _req_completed(req):
        return False
    if _prefill_complete(req):
        return False
    return _remaining_prefill(req) > 0


def _is_decode_active(req: Any) -> bool:
    if _req_completed(req):
        return False
    if not _prefill_complete(req):
        return False
    return _remaining_decode(req) > 0


def _build_request_lookup(env: Any, state: Any) -> Dict[int, Any]:
    sim = state.simulator
    if hasattr(env, "_build_request_lookup"):
        try:
            out = env._build_request_lookup(sim, state)
            if isinstance(out, dict):
                return out
        except TypeError:
            try:
                out = env._build_request_lookup(sim)
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
        except Exception:
            pass

    # Fallback path mirrored from infer.py
    lookup: Dict[int, Any] = {}
    sched = getattr(sim, "_scheduler", None)
    if sched is None:
        return lookup

    rq = getattr(sched, "_request_queue", None)
    if rq is not None:
        for req in rq:
            lookup[_req_id(req)] = req

    replica_schedulers = getattr(sched, "_replica_schedulers", {}) or {}
    for rs in replica_schedulers.values():
        waiting = getattr(rs, "_waiting_queue", None)
        if waiting is not None and hasattr(waiting, "to_list"):
            for req in waiting.to_list():
                lookup[_req_id(req)] = req

        for req in getattr(rs, "_running", []) or []:
            lookup[_req_id(req)] = req

        req_map = getattr(rs, "_requests", None)
        if isinstance(req_map, dict):
            for req in req_map.values():
                if not _req_completed(req):
                    lookup[_req_id(req)] = req

    # sanitize invalid ids
    lookup = {rid: req for rid, req in lookup.items() if rid >= 0 and req is not None}
    return lookup


def _prefill_deadline(req: Any) -> Optional[float]:
    slo = _prefill_slo(req)
    if slo <= 0:
        return None
    return _arrived_at(req) + slo


def _decode_deadline(req: Any, state: Any) -> Optional[float]:
    rid = _req_id(req)
    if rid < 0:
        return None
    st = getattr(state, "stats", None)
    by_id = getattr(st, "decode_next_deadline_by_id", {}) if st is not None else {}
    if isinstance(by_id, dict) and rid in by_id:
        return _safe_float(by_id.get(rid), None)

    slo = _decode_slo(req)
    if slo <= 0:
        return None
    sim_time = _safe_float(getattr(state.simulator, "_time", 0.0), 0.0)
    return sim_time + slo


def _min_mean(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    return float(min(vals)), float(sum(vals) / len(vals))


def extract_features(env: Any, state: Any) -> np.ndarray:
    lookup = _build_request_lookup(env, state)
    reqs = list(lookup.values())

    sim_time = _safe_float(getattr(state.simulator, "_time", 0.0), 0.0)

    prefill_reqs = [r for r in reqs if _is_prefill_active(r)]
    decode_reqs = [r for r in reqs if _is_decode_active(r)]

    n_waiting_total = float(len(reqs))
    n_prefill_active = float(len(prefill_reqs))
    n_decode_active = float(len(decode_reqs))

    prefill_remaining = [_remaining_prefill(r) for r in prefill_reqs]
    decode_remaining = [_remaining_decode(r) for r in decode_reqs]

    sum_prefill_tokens_remaining = float(sum(prefill_remaining))
    sum_decode_tokens_remaining = float(sum(decode_remaining))
    max_prefill_tokens_remaining = float(max(prefill_remaining) if prefill_remaining else 0)
    max_decode_tokens_remaining = float(max(decode_remaining) if decode_remaining else 0)
    mean_prefill_tokens_remaining = float(sum(prefill_remaining) / len(prefill_remaining)) if prefill_remaining else 0.0
    mean_decode_tokens_remaining = float(sum(decode_remaining) / len(decode_remaining)) if decode_reqs else 0.0
    sum_total_tokens_remaining = float(sum_prefill_tokens_remaining + sum_decode_tokens_remaining)

    prefill_slacks: List[float] = []
    for req in prefill_reqs:
        dl = _prefill_deadline(req)
        if dl is not None:
            prefill_slacks.append(float(dl - sim_time))
    min_prefill_slack_sec, mean_prefill_slack_sec = _min_mean(prefill_slacks)
    prefill_overdues = [max(0.0, -x) for x in prefill_slacks]
    prefill_overdue_count = float(sum(1 for x in prefill_slacks if x < 0.0))
    prefill_overdue_lateness_sum_sec = float(sum(prefill_overdues))

    decode_slacks: List[float] = []
    for req in decode_reqs:
        dl = _decode_deadline(req, state)
        if dl is not None:
            decode_slacks.append(float(dl - sim_time))
    min_decode_slack_sec, mean_decode_slack_sec = _min_mean(decode_slacks)
    decode_overdues = [max(0.0, -x) for x in decode_slacks]
    decode_overdue_count = float(sum(1 for x in decode_slacks if x < 0.0))
    decode_overdue_lateness_sum_sec = float(sum(decode_overdues))

    stats = getattr(state, "stats", None)
    cum_slo_violations = float(_safe_int(getattr(stats, "slo_violations", 0), 0))
    cum_lateness_sum_sec = float(_safe_float(getattr(stats, "slo_lateness_sum", 0.0), 0.0))

    phase = float(sim_time % 1.0)
    time_to_next_adversary_inject_sec = float(max(0.0, min(1.0, 1.0 - phase)))

    n_decode = int(n_decode_active)
    decode_bucket_0 = 1.0 if n_decode == 0 else 0.0
    decode_bucket_1_2 = 1.0 if 1 <= n_decode <= 2 else 0.0
    decode_bucket_3_4 = 1.0 if 3 <= n_decode <= 4 else 0.0
    decode_bucket_5_8 = 1.0 if 5 <= n_decode <= 8 else 0.0
    decode_bucket_9_plus = 1.0 if n_decode >= 9 else 0.0

    n_prefill = int(n_prefill_active)
    prefill_bucket_0 = 1.0 if n_prefill == 0 else 0.0
    prefill_bucket_1_plus = 1.0 if n_prefill >= 1 else 0.0

    feat = np.array(
        [
            n_waiting_total,
            n_prefill_active,
            n_decode_active,
            sum_prefill_tokens_remaining,
            sum_decode_tokens_remaining,
            max_prefill_tokens_remaining,
            max_decode_tokens_remaining,
            mean_prefill_tokens_remaining,
            mean_decode_tokens_remaining,
            sum_total_tokens_remaining,
            min_prefill_slack_sec,
            mean_prefill_slack_sec,
            prefill_overdue_count,
            prefill_overdue_lateness_sum_sec,
            min_decode_slack_sec,
            mean_decode_slack_sec,
            decode_overdue_count,
            decode_overdue_lateness_sum_sec,
            cum_slo_violations,
            cum_lateness_sum_sec,
            sim_time,
            phase,
            time_to_next_adversary_inject_sec,
            decode_bucket_0,
            decode_bucket_1_2,
            decode_bucket_3_4,
            decode_bucket_5_8,
            decode_bucket_9_plus,
            prefill_bucket_0,
            prefill_bucket_1_plus,
        ],
        dtype=np.float32,
    )
    if feat.shape[0] != 30:
        raise AssertionError(f"expected 30 features, got {feat.shape[0]}")
    return feat


def compute_feature_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape={x.shape}")
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def normalize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def state_hash(env: Any, state: Any, player: str) -> str:
    lookup = _build_request_lookup(env, state)
    stats = getattr(state, "stats", None)
    payload = {
        "player": str(player),
        "sim_time": round(_safe_float(getattr(state.simulator, "_time", 0.0), 0.0), 6),
        "slo_viol": _safe_int(getattr(stats, "slo_violations", 0), 0),
        "slo_lat": round(_safe_float(getattr(stats, "slo_lateness_sum", 0.0), 0.0), 6),
        "reqs": [],
    }

    items = []
    for rid, req in lookup.items():
        items.append(
            {
                "id": int(rid),
                "completed": bool(_req_completed(req)),
                "prefill_complete": bool(_prefill_complete(req)),
                "prefill_remaining": int(_remaining_prefill(req)),
                "decode_remaining": int(_remaining_decode(req)),
                "done_prefill": int(_done_prefill(req)),
                "done_decode": int(_done_decode(req)),
                "prefill_slo": round(_prefill_slo(req), 6),
                "decode_slo": round(_decode_slo(req), 6),
                "arrived_at": round(_arrived_at(req), 6),
            }
        )
    payload["reqs"] = sorted(items, key=lambda d: d["id"])

    b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(b).hexdigest()
