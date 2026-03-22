from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Tuple


@dataclass
class PrefillBurstAggregate:
    request_count: int = 0
    active_prefill_count: int = 0
    prefill_remaining_sum: float = 0.0
    one_chunk_left_count: int = 0
    deadline_distance: float = float("inf")


@dataclass
class RequestAggregate:
    n_decode_active: int = 0
    n_decode_violated_active: int = 0
    prefill_bursts: Dict[float, PrefillBurstAggregate] = field(default_factory=dict)
    current_prefill_deadline_distance: float = 0.0
    current_burst_request_count: int = 0
    current_burst_active_prefill_count: int = 0
    current_burst_prefill_remaining_sum: float = 0.0
    current_burst_one_chunk_left_count: int = 0

    def finalize(self) -> None:
        active_bursts = [
            burst
            for burst in self.prefill_bursts.values()
            if burst.active_prefill_count > 0
        ]
        if not active_bursts:
            self.current_prefill_deadline_distance = 0.0
            self.current_burst_request_count = 0
            self.current_burst_active_prefill_count = 0
            self.current_burst_prefill_remaining_sum = 0.0
            self.current_burst_one_chunk_left_count = 0
            return

        burst = min(
            active_bursts,
            key=lambda x: (x.deadline_distance, -x.request_count),
        )
        if not math.isfinite(burst.deadline_distance):
            self.current_prefill_deadline_distance = 0.0
        else:
            self.current_prefill_deadline_distance = float(burst.deadline_distance)
        self.current_burst_request_count = int(burst.request_count)
        self.current_burst_active_prefill_count = int(burst.active_prefill_count)
        self.current_burst_prefill_remaining_sum = float(burst.prefill_remaining_sum)
        self.current_burst_one_chunk_left_count = int(burst.one_chunk_left_count)


@dataclass(frozen=True)
class StructuralPreferenceConstraint:
    name: str
    better_features: Mapping[str, float]
    worse_features: Mapping[str, float]
    margin: float


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"1", "true", "t", "yes", "y"}
    try:
        return bool(int(v))
    except Exception:
        return bool(v)


def update_request_aggregate(
    agg: RequestAggregate,
    req_row: Mapping[str, Any],
    *,
    sim_time: float,
) -> None:
    prefill_chunk_size = 512.0
    prefill_remaining = _to_float(req_row.get("prefill_tokens_remaining", 0.0))
    decode_remaining = _to_float(req_row.get("decode_tokens_remaining", 0.0))
    completed = _to_bool(req_row.get("completed", False))
    prefill_complete = _to_bool(req_row.get("prefill_complete", False))
    prefill_deadline = _to_float(req_row.get("prefill_deadline", 0.0))

    if completed:
        return

    if prefill_deadline > 0.0:
        burst_key = round(prefill_deadline, 9)
        burst = agg.prefill_bursts.get(burst_key)
        if burst is None:
            burst = PrefillBurstAggregate()
            agg.prefill_bursts[burst_key] = burst
        burst.request_count += 1
        deadline_distance = prefill_deadline - sim_time
        if deadline_distance < burst.deadline_distance:
            burst.deadline_distance = deadline_distance

    if not prefill_complete:
        chunks_left = int(math.ceil(max(prefill_remaining, 0.0) / prefill_chunk_size))
        if prefill_deadline > 0.0:
            burst = agg.prefill_bursts[round(prefill_deadline, 9)]
            burst.active_prefill_count += 1
            burst.prefill_remaining_sum += prefill_remaining
            if chunks_left == 1:
                burst.one_chunk_left_count += 1
    elif decode_remaining > 0.0:
        agg.n_decode_active += 1
        if _to_bool(req_row.get("decode_violated_now", False)):
            agg.n_decode_violated_active += 1


BASELINE_V1_FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "current_burst_size_norm",
    "current_burst_remaining_chunks_norm",
    "current_burst_deadline_urgency_norm",
    "current_burst_lateness_sec_norm",
    "decode_active_norm",
    "decode_violated_count_norm",
)

BASELINE_V1_SIGN_CONSTRAINTS: Mapping[str, str] = {
    "current_burst_size_norm": "nonnegative",
    "current_burst_remaining_chunks_norm": "nonnegative",
    "current_burst_deadline_urgency_norm": "nonnegative",
    "current_burst_lateness_sec_norm": "nonnegative",
    "decode_active_norm": "nonnegative",
    "decode_violated_count_norm": "nonnegative",
}


def extract_baseline_v1_features(
    state_row: Mapping[str, Any],
    req_agg: RequestAggregate | None,
) -> Dict[str, float]:
    agg = req_agg if req_agg is not None else RequestAggregate()
    agg.finalize()
    burst_size_denom = 6.0
    burst_chunk_norm_denom = 6.0 * 6.0
    decode_count_norm_denom = 200.0
    burst_request_count = float(agg.current_burst_request_count)
    deadline_distance = float(agg.current_prefill_deadline_distance)
    remaining_chunks_norm = float(agg.current_burst_prefill_remaining_sum) / (512.0 * burst_chunk_norm_denom)
    if burst_request_count > 0.0:
        deadline_urgency_norm = 1.0 / max(deadline_distance, 1e-3)
        lateness_sec_norm = max(0.0, -deadline_distance)
    else:
        deadline_urgency_norm = 0.0
        lateness_sec_norm = 0.0

    feats: Dict[str, float] = {
        "bias": 1.0,
        "current_burst_size_norm": burst_request_count / burst_size_denom,
        "current_burst_remaining_chunks_norm": remaining_chunks_norm,
        "current_burst_deadline_urgency_norm": deadline_urgency_norm,
        "current_burst_lateness_sec_norm": lateness_sec_norm,
        "decode_active_norm": float(agg.n_decode_active) / decode_count_norm_denom,
        "decode_violated_count_norm": float(agg.n_decode_violated_active) / decode_count_norm_denom,
    }
    return feats


def feature_names_for_set(feature_set: str) -> List[str]:
    if feature_set != "baseline_v1":
        raise ValueError(f"unsupported feature_set={feature_set!r}; expected 'baseline_v1'")
    return list(BASELINE_V1_FEATURE_NAMES)


def feature_weight_bounds_for_set(
    feature_set: str,
    *,
    weight_bound_abs: float,
) -> List[Tuple[float, float]]:
    if feature_set != "baseline_v1":
        raise ValueError(f"unsupported feature_set={feature_set!r}; expected 'baseline_v1'")
    wmax = float(weight_bound_abs)
    bounds_by_name: Dict[str, Tuple[float, float]] = {
        "bias": (-wmax, wmax),
        "current_burst_size_norm": (0.0, wmax),
        "current_burst_remaining_chunks_norm": (0.0, wmax),
        "current_burst_deadline_urgency_norm": (0.0, wmax),
        "current_burst_lateness_sec_norm": (0.0, wmax),
        "decode_active_norm": (0.0, wmax),
        "decode_violated_count_norm": (0.0, wmax),
    }
    return [bounds_by_name[name] for name in feature_names_for_set(feature_set)]


def feature_sign_constraints_for_set(feature_set: str) -> Dict[str, str]:
    if feature_set != "baseline_v1":
        raise ValueError(f"unsupported feature_set={feature_set!r}; expected 'baseline_v1'")
    return dict(BASELINE_V1_SIGN_CONSTRAINTS)


def structural_preference_constraints_for_set(
    feature_set: str,
    *,
    margin_eps: float,
) -> List[StructuralPreferenceConstraint]:
    if feature_set != "baseline_v1":
        raise ValueError(f"unsupported feature_set={feature_set!r}; expected 'baseline_v1'")
    eps = float(margin_eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"margin_eps must be finite and > 0, got {margin_eps}")

    return [
        StructuralPreferenceConstraint(
            name="fresh_burst_vs_empty",
            better_features={
                "bias": 1.0,
            },
            worse_features={
                "bias": 1.0,
                "current_burst_size_norm": 1.0 / 6.0,
                "current_burst_remaining_chunks_norm": 6.0 / 36.0,
                "current_burst_deadline_urgency_norm": 1.0,
            },
            margin=eps,
        ),
        StructuralPreferenceConstraint(
            name="more_decode_load_worse",
            better_features={
                "bias": 1.0,
            },
            worse_features={
                "bias": 1.0,
                "decode_active_norm": 0.1,
            },
            margin=eps,
        ),
    ]


def extract_features_for_set(
    *,
    feature_set: str,
    state_row: Mapping[str, Any],
    req_agg: RequestAggregate | None,
) -> Dict[str, float]:
    if feature_set != "baseline_v1":
        raise ValueError(f"unsupported feature_set={feature_set!r}; expected 'baseline_v1'")
    return extract_baseline_v1_features(state_row=state_row, req_agg=req_agg)
