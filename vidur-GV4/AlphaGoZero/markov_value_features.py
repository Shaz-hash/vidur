"""Markov-sufficient structured value features for GV3 AlphaGoZero.

Policy models intentionally continue to use the legacy 226D representation.
This module owns the versioned value-only state contract shared with native.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

MARKOV_VALUE_SCHEMA = "markov_v2"
GLOBAL_DIM = 19
REQUEST_DIM = 24
LAUNCH_DIM = 3

MAX_ACTIVE_REQUEST_SCALE = 120.0
MAX_PREFILL_REQUEST_SCALE = 20.0
MAX_DECODE_REQUEST_SCALE = 100.0
MAX_PREFILL_TOKENS = 4096.0
MAX_DECODE_TOKENS = 864.0
DECODE_CREDIT_MINT = 216.0
ADVERSARY_TICK_SEC = 0.2
LAUNCH_WINDOW_SEC = 1.0
LAUNCH_REQUEST_CAP = 7.0
LAUNCH_PREFILL_CAP = 7.0 * 1024.0
PREFILL_TIME_SCALE_SEC = 1.0
DECODE_TIME_SCALE_SEC = 0.05

META_NEXT_ADV_TICK = -9_100_001
META_DECODE_CREDIT_BALANCE = -9_100_005
META_MISSED_ADV_SOURCE = -9_100_006

GLOBAL_FEATURE_NAMES = (
    "active_count_div_120",
    "active_prefill_count_div_20",
    "active_decode_count_div_100",
    "remaining_prefill_div_81920",
    "remaining_decode_div_86400",
    "processed_prefill_div_81920",
    "processed_decode_div_86400",
    "processed_context_div_595200",
    "asinh_signed_decode_credit_div_216",
    "usable_decode_credit_div_216",
    "asinh_next_adv_tick_delta_div_0p2",
    "pending_adv_tick",
    "missed_adv_none",
    "missed_adv_controller_cross",
    "missed_adv_fast_forward_cross",
    "launch_request_count_div_7",
    "launch_prefill_tokens_div_7168",
    "active_violated_count_div_100",
    "active_prefill_finalized_count_div_20",
)

REQUEST_FEATURE_NAMES = (
    "decode_phase",
    "prefill_total_div_4096",
    "prefill_processed_div_4096",
    "prefill_remaining_div_4096",
    "decode_total_div_864",
    "decode_processed_div_864",
    "decode_remaining_div_864",
    "processed_context_div_4960",
    "asinh_arrival_age_div_1s",
    "asinh_queue_age_div_1s",
    "asinh_prefill_slo_div_1s",
    "asinh_prefill_deadline_delta_div_1s",
    "asinh_decode_slo_div_0p05s",
    "decode_deadline_present",
    "asinh_decode_deadline_delta_div_0p05s",
    "prefill_completed_at_present",
    "asinh_prefill_completion_age_div_1s",
    "asinh_prefill_lateness_div_1s",
    "asinh_decode_lateness_div_1s",
    "decode_tokens_counted_div_864",
    "decode_credit_ledger_entry_present",
    "violated",
    "prefill_lateness_finalized",
    "is_prefill_complete",
)

LAUNCH_FEATURE_NAMES = (
    "asinh_launch_age_div_1s",
    "request_count_div_7",
    "prefill_tokens_div_7168",
)


@dataclass(frozen=True)
class MarkovValueFeatures:
    global_features: np.ndarray
    request_features: np.ndarray
    launch_features: np.ndarray
    request_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.global_features.shape != (GLOBAL_DIM,):
            raise ValueError(f"global feature shape {self.global_features.shape} != {(GLOBAL_DIM,)}")
        if self.request_features.ndim != 2 or self.request_features.shape[1] != REQUEST_DIM:
            raise ValueError(f"request feature shape {self.request_features.shape}")
        if self.launch_features.ndim != 2 or self.launch_features.shape[1] != LAUNCH_DIM:
            raise ValueError(f"launch feature shape {self.launch_features.shape}")
        if self.request_features.shape[0] != len(self.request_ids):
            raise ValueError("request feature/id count mismatch")
        for name, values in (
            ("global", self.global_features),
            ("request", self.request_features),
            ("launch", self.launch_features),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} features contain non-finite values")

    def replay_fields(self) -> dict[str, str | int]:
        return {
            "value_feature_schema": MARKOV_VALUE_SCHEMA,
            "value_global_features_json": _compact_json(self.global_features.tolist()),
            "value_request_features_json": _compact_json(self.request_features.tolist()),
            "value_launch_features_json": _compact_json(self.launch_features.tolist()),
            "value_request_count": int(self.request_features.shape[0]),
            "value_launch_count": int(self.launch_features.shape[0]),
        }


def feature_schema_metadata() -> dict[str, Any]:
    return {
        "name": MARKOV_VALUE_SCHEMA,
        "global_dim": GLOBAL_DIM,
        "request_dim": REQUEST_DIM,
        "launch_dim": LAUNCH_DIM,
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "request_feature_names": list(REQUEST_FEATURE_NAMES),
        "launch_feature_names": list(LAUNCH_FEATURE_NAMES),
        "scales": {
            "active_requests": MAX_ACTIVE_REQUEST_SCALE,
            "prefill_requests": MAX_PREFILL_REQUEST_SCALE,
            "decode_requests": MAX_DECODE_REQUEST_SCALE,
            "prefill_tokens": MAX_PREFILL_TOKENS,
            "decode_tokens": MAX_DECODE_TOKENS,
            "decode_credit_mint": DECODE_CREDIT_MINT,
            "adversary_tick_sec": ADVERSARY_TICK_SEC,
            "launch_window_sec": LAUNCH_WINDOW_SEC,
            "launch_request_cap": LAUNCH_REQUEST_CAP,
            "launch_prefill_cap": LAUNCH_PREFILL_CAP,
            "prefill_time_sec": PREFILL_TIME_SCALE_SEC,
            "decode_time_sec": DECODE_TIME_SCALE_SEC,
        },
        "normalization": "division_or_asinh_raw_div_scale_no_clipping_no_buckets",
    }


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _mapping(value: Any, *, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _map_get(mapping: Mapping[Any, Any], key: int, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    text = str(key)
    if text in mapping:
        return mapping[text]
    return default


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _integer(value: Any, *, label: str) -> int:
    result = _finite(value, label=label)
    integer = int(result)
    if float(integer) != result:
        raise ValueError(f"{label} is not integral: {value!r}")
    return integer


def _nonnegative_int(value: Any, *, label: str) -> int:
    result = _integer(value, label=label)
    if result < 0:
        raise ValueError(f"{label} is negative: {result}")
    return result


def _boolean(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _asinh_scaled(value: float, scale: float) -> float:
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid asinh arguments value={value} scale={scale}")
    return math.asinh(value / scale)


def _id_set(stats: Mapping[Any, Any], name: str) -> set[int]:
    raw = stats.get(name, ())
    if raw is None:
        return set()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"stats.{name} must be a sequence")
    return {_integer(value, label=f"stats.{name}") for value in raw}


def _numeric_map(stats: Mapping[Any, Any], name: str) -> Mapping[Any, Any]:
    raw = stats.get(name, {})
    if raw is None:
        return {}
    return _mapping(raw, label=f"stats.{name}")


def _parse_launch(entry: Any, index: int) -> tuple[float, int, int]:
    if isinstance(entry, Mapping):
        timestamp = entry.get("timestamp", entry.get("time"))
        count = entry.get("count", entry.get("requests", 0))
        tokens = entry.get("prefill_tokens", entry.get("tokens", 0))
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) >= 3:
        timestamp, count, tokens = entry[0], entry[1], entry[2]
    else:
        raise ValueError(f"recent_launches[{index}] has invalid shape")
    return (
        _finite(timestamp, label=f"recent_launches[{index}].timestamp"),
        _nonnegative_int(count, label=f"recent_launches[{index}].count"),
        _nonnegative_int(tokens, label=f"recent_launches[{index}].prefill_tokens"),
    )


def build_markov_value_features(
    state_payload: Mapping[str, Any],
    *,
    launch_window_sec: float = LAUNCH_WINDOW_SEC,
) -> MarkovValueFeatures:
    state = _mapping(state_payload, label="state")
    stats = _mapping(state.get("stats", {}), label="state.stats")
    sim_time = _finite(state.get("sim_time", state.get("time")), label="state.sim_time")

    requests_raw = state.get("requests", ())
    if isinstance(requests_raw, (str, bytes)) or not isinstance(requests_raw, Sequence):
        raise ValueError("state.requests must be a sequence")
    requests_by_id: dict[int, Mapping[Any, Any]] = {}
    for index, raw in enumerate(requests_raw):
        request = _mapping(raw, label=f"state.requests[{index}]")
        rid = _integer(request.get("request_id", request.get("id")), label=f"request[{index}].id")
        if rid in requests_by_id:
            raise ValueError(f"duplicate request id {rid}")
        requests_by_id[rid] = request

    if "active_request_ids" not in stats:
        raise ValueError("stats.active_request_ids is required")
    active_ids_raw = stats.get("active_request_ids") or ()
    if isinstance(active_ids_raw, (str, bytes)) or not isinstance(active_ids_raw, Sequence):
        raise ValueError("stats.active_request_ids must be a sequence")
    active_ids = tuple(sorted(_integer(value, label="active_request_id") for value in active_ids_raw))
    if len(set(active_ids)) != len(active_ids):
        raise ValueError("stats.active_request_ids contains duplicates")

    violated_ids = _id_set(stats, "violated_request_ids")
    finalized_ids = _id_set(stats, "prefill_lateness_finalized_ids")
    prefill_lateness = _numeric_map(stats, "per_request_prefill_lateness_by_id")
    decode_lateness = _numeric_map(stats, "per_request_decode_lateness_by_id")
    decode_deadlines = _numeric_map(stats, "decode_next_deadline_by_id")
    counted_by_id = _numeric_map(stats, "decode_tokens_counted_by_id")

    request_rows: list[list[float]] = []
    prefill_count = 0
    decode_count = 0
    total_remaining_prefill = 0
    total_remaining_decode = 0
    total_processed_prefill = 0
    total_processed_decode = 0

    for rid in active_ids:
        if rid not in requests_by_id:
            raise ValueError(f"active request {rid} has no request record")
        request = requests_by_id[rid]
        if any(bool(request.get(name, False)) for name in ("completed", "dropped", "stopped_decode", "feature_only")):
            raise ValueError(f"active request {rid} is terminal or feature-only")

        prefill_total = _nonnegative_int(
            request.get("num_prefill_tokens", request.get("prefill_tokens", 0)),
            label=f"request {rid} prefill total",
        )
        prefill_processed = _nonnegative_int(
            request.get("num_processed_prefill_tokens", request.get("processed_prefill_tokens", 0)),
            label=f"request {rid} prefill processed",
        )
        decode_total = _nonnegative_int(
            request.get("num_decode_tokens", request.get("decode_tokens", 0)),
            label=f"request {rid} decode total",
        )
        decode_processed = _nonnegative_int(
            request.get("num_processed_decode_tokens", request.get("processed_decode_tokens", 0)),
            label=f"request {rid} decode processed",
        )
        if prefill_processed > prefill_total or decode_processed > decode_total:
            raise ValueError(f"request {rid} processed tokens exceed total")
        prefill_remaining = prefill_total - prefill_processed
        decode_remaining = decode_total - decode_processed
        is_prefill_complete = bool(request.get("is_prefill_complete", request.get("prefill_complete", False)))
        if is_prefill_complete != (prefill_remaining == 0):
            raise ValueError(
                f"request {rid} prefill-complete flag disagrees with remaining tokens"
            )
        decode_phase = is_prefill_complete and decode_remaining > 0
        if not decode_phase and prefill_remaining <= 0:
            raise ValueError(f"active request {rid} has no executable work")
        prefill_count += int(not decode_phase)
        decode_count += int(decode_phase)

        arrived_at = _finite(request.get("arrived_at", 0.0), label=f"request {rid} arrived_at")
        queued_at = _finite(request.get("queued_at", arrived_at), label=f"request {rid} queued_at")
        prefill_slo = _finite(
            request.get("prefill_slo_time", request.get("prefill_slo", 0.0)),
            label=f"request {rid} prefill_slo",
        )
        decode_slo = _finite(
            request.get("decode_slo_time", request.get("decode_slo", 0.0)),
            label=f"request {rid} decode_slo",
        )
        if prefill_slo < 0.0 or decode_slo < 0.0:
            raise ValueError(f"request {rid} has negative SLO")
        prefill_deadline = _finite(
            request.get("prefill_deadline", arrived_at + prefill_slo),
            label=f"request {rid} prefill_deadline",
        )

        decode_deadline_raw = _map_get(
            decode_deadlines,
            rid,
            request.get("decode_next_deadline", request.get("decode_deadline", -1.0)),
        )
        decode_deadline = _finite(decode_deadline_raw, label=f"request {rid} decode deadline")
        decode_deadline_present = decode_deadline >= 0.0
        if decode_phase and not decode_deadline_present:
            raise ValueError(f"decode-phase request {rid} has no decode deadline")

        completed_at = _finite(
            request.get("prefill_completed_at", -1.0),
            label=f"request {rid} prefill_completed_at",
        )
        completed_at_present = completed_at >= 0.0
        if decode_phase and not completed_at_present:
            raise ValueError(f"decode-phase request {rid} has no prefill completion time")

        pref_late = _finite(
            _map_get(prefill_lateness, rid, request.get("prefill_lateness", 0.0)),
            label=f"request {rid} prefill lateness",
        )
        dec_late = _finite(
            _map_get(decode_lateness, rid, request.get("decode_lateness", 0.0)),
            label=f"request {rid} decode lateness",
        )
        if pref_late < 0.0 or dec_late < 0.0:
            raise ValueError(f"request {rid} has negative lateness")

        ledger_raw = _map_get(counted_by_id, rid, None)
        ledger_present = ledger_raw is not None
        ledger_counted = (
            _nonnegative_int(ledger_raw, label=f"request {rid} credit counted")
            if ledger_present
            else 0
        )

        request_rows.append(
            [
                _boolean(decode_phase),
                prefill_total / MAX_PREFILL_TOKENS,
                prefill_processed / MAX_PREFILL_TOKENS,
                prefill_remaining / MAX_PREFILL_TOKENS,
                decode_total / MAX_DECODE_TOKENS,
                decode_processed / MAX_DECODE_TOKENS,
                decode_remaining / MAX_DECODE_TOKENS,
                (prefill_processed + decode_processed) / (MAX_PREFILL_TOKENS + MAX_DECODE_TOKENS),
                _asinh_scaled(sim_time - arrived_at, PREFILL_TIME_SCALE_SEC),
                _asinh_scaled(sim_time - queued_at, PREFILL_TIME_SCALE_SEC),
                _asinh_scaled(prefill_slo, PREFILL_TIME_SCALE_SEC),
                _asinh_scaled(prefill_deadline - sim_time, PREFILL_TIME_SCALE_SEC),
                _asinh_scaled(decode_slo, DECODE_TIME_SCALE_SEC),
                _boolean(decode_deadline_present),
                _asinh_scaled(decode_deadline - sim_time, DECODE_TIME_SCALE_SEC)
                if decode_deadline_present
                else 0.0,
                _boolean(completed_at_present),
                _asinh_scaled(sim_time - completed_at, PREFILL_TIME_SCALE_SEC)
                if completed_at_present
                else 0.0,
                _asinh_scaled(pref_late, PREFILL_TIME_SCALE_SEC),
                _asinh_scaled(dec_late, PREFILL_TIME_SCALE_SEC),
                ledger_counted / MAX_DECODE_TOKENS,
                _boolean(ledger_present),
                _boolean(rid in violated_ids or bool(request.get("violated", False))),
                _boolean(rid in finalized_ids),
                _boolean(is_prefill_complete),
            ]
        )
        total_remaining_prefill += prefill_remaining
        total_remaining_decode += decode_remaining
        total_processed_prefill += prefill_processed
        total_processed_decode += decode_processed

    launches_raw = stats.get("recent_launches", ())
    if launches_raw is None:
        launches_raw = ()
    if isinstance(launches_raw, (str, bytes)) or not isinstance(launches_raw, Sequence):
        raise ValueError("stats.recent_launches must be a sequence")
    launch_rows: list[list[float]] = []
    launch_request_count = 0
    launch_prefill_tokens = 0
    for index, raw in enumerate(launches_raw):
        timestamp, count, tokens = _parse_launch(raw, index)
        age = sim_time - timestamp
        if age < -1e-7:
            raise ValueError(f"recent_launches[{index}] is in the future")
        if age > float(launch_window_sec) + 1e-6:
            continue
        launch_rows.append(
            [
                _asinh_scaled(age, float(launch_window_sec)),
                count / LAUNCH_REQUEST_CAP,
                tokens / LAUNCH_PREFILL_CAP,
            ]
        )
        launch_request_count += count
        launch_prefill_tokens += tokens

    next_tick_raw = stats.get("next_adv_tick", _map_get(decode_deadlines, META_NEXT_ADV_TICK, None))
    if next_tick_raw is None:
        raise ValueError("stats.next_adv_tick is required")
    next_tick = _finite(next_tick_raw, label="stats.next_adv_tick")
    if next_tick < 0.0:
        raise ValueError("stats.next_adv_tick is missing")

    credit_raw = stats.get(
        "decode_credit_balance",
        _map_get(counted_by_id, META_DECODE_CREDIT_BALANCE, 0),
    )
    credit = _integer(credit_raw, label="stats.decode_credit_balance")
    usable_credit = _nonnegative_int(
        stats.get("decode_credit_available", max(0, credit)),
        label="stats.decode_credit_available",
    )
    missed_source = _integer(
        stats.get("missed_adv_source", _map_get(decode_deadlines, META_MISSED_ADV_SOURCE, 0)),
        label="stats.missed_adv_source",
    )
    if missed_source not in (0, 1, 2):
        raise ValueError(f"unsupported missed_adv_source {missed_source}")

    global_values = np.asarray(
        [
            len(active_ids) / MAX_ACTIVE_REQUEST_SCALE,
            prefill_count / MAX_PREFILL_REQUEST_SCALE,
            decode_count / MAX_DECODE_REQUEST_SCALE,
            total_remaining_prefill / (MAX_PREFILL_REQUEST_SCALE * MAX_PREFILL_TOKENS),
            total_remaining_decode / (MAX_DECODE_REQUEST_SCALE * MAX_DECODE_TOKENS),
            total_processed_prefill / (MAX_PREFILL_REQUEST_SCALE * MAX_PREFILL_TOKENS),
            total_processed_decode / (MAX_DECODE_REQUEST_SCALE * MAX_DECODE_TOKENS),
            (total_processed_prefill + total_processed_decode)
            / (MAX_ACTIVE_REQUEST_SCALE * (MAX_PREFILL_TOKENS + MAX_DECODE_TOKENS)),
            _asinh_scaled(float(credit), DECODE_CREDIT_MINT),
            usable_credit / DECODE_CREDIT_MINT,
            _asinh_scaled(next_tick - sim_time, ADVERSARY_TICK_SEC),
            _boolean(stats.get("pending_adv_tick", next_tick <= sim_time + 1e-9)),
            _boolean(missed_source == 0),
            _boolean(missed_source == 1),
            _boolean(missed_source == 2),
            launch_request_count / LAUNCH_REQUEST_CAP,
            launch_prefill_tokens / LAUNCH_PREFILL_CAP,
            sum(1 for rid in active_ids if rid in violated_ids) / MAX_DECODE_REQUEST_SCALE,
            sum(1 for rid in active_ids if rid in finalized_ids) / MAX_PREFILL_REQUEST_SCALE,
        ],
        dtype=np.float32,
    )
    request_values = np.asarray(request_rows, dtype=np.float32).reshape(-1, REQUEST_DIM)
    launch_values = np.asarray(launch_rows, dtype=np.float32).reshape(-1, LAUNCH_DIM)
    return MarkovValueFeatures(
        global_features=global_values,
        request_features=request_values,
        launch_features=launch_values,
        request_ids=active_ids,
    )


def features_from_replay_row(row: Mapping[str, Any]) -> MarkovValueFeatures:
    if str(row.get("value_feature_schema", "")).strip() != MARKOV_VALUE_SCHEMA:
        raise ValueError(f"replay row is not {MARKOV_VALUE_SCHEMA}")

    def parse(name: str) -> Any:
        raw = row.get(name, "")
        try:
            return json.loads(str(raw))
        except Exception as exc:
            raise ValueError(f"invalid {name}") from exc

    global_values = np.asarray(parse("value_global_features_json"), dtype=np.float32)
    request_values = np.asarray(parse("value_request_features_json"), dtype=np.float32).reshape(-1, REQUEST_DIM)
    launch_values = np.asarray(parse("value_launch_features_json"), dtype=np.float32).reshape(-1, LAUNCH_DIM)
    expected_requests = _integer(row.get("value_request_count", request_values.shape[0]), label="value_request_count")
    expected_launches = _integer(row.get("value_launch_count", launch_values.shape[0]), label="value_launch_count")
    if expected_requests != request_values.shape[0] or expected_launches != launch_values.shape[0]:
        raise ValueError("replay structured feature count mismatch")
    return MarkovValueFeatures(
        global_features=global_values,
        request_features=request_values,
        launch_features=launch_values,
        request_ids=tuple(range(expected_requests)),
    )
