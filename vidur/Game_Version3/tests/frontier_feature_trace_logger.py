from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as Fnn

from ..DNN.infer import build_model_inputs


META_DECODE_CREDIT_BAL = -9_100_005


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    try:
        return bool(value)
    except Exception:
        return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _tensor_json(tensor: torch.Tensor) -> str:
    return _json_dumps([float(x) for x in tensor.detach().cpu().reshape(-1).tolist()])


def _mask_json(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "[]"
    return _json_dumps([bool(x) for x in tensor.detach().cpu().reshape(-1).tolist()])


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _norm01(x: float, denom: float) -> float:
    if float(denom) <= 0.0:
        return 0.0
    return _clip(float(x) / float(denom), 0.0, 1.0)


def _centered01(x: float, radius: float) -> float:
    if float(radius) <= 0.0:
        return 0.5
    clipped = _clip(float(x), -float(radius), float(radius))
    return (clipped + float(radius)) / (2.0 * float(radius))


def _int_list(raw: Any) -> list[int]:
    return [int(x) for x in list(raw or [])]


def _int_float_map(raw: Any) -> dict[int, float]:
    return {int(k): float(v) for k, v in dict(raw or {}).items()}


def _int_int_map(raw: Any) -> dict[int, int]:
    return {int(k): int(v) for k, v in dict(raw or {}).items()}


def _request_lookup(env: Any, state: Any) -> dict[int, Any]:
    if hasattr(env, "_build_request_lookup"):
        return {int(k): v for k, v in env._build_request_lookup(state.simulator, state=state).items()}
    req_map = getattr(env, "_req_map", lambda sim: {})(state.simulator)
    active = set(int(x) for x in getattr(getattr(state, "stats", None), "active_request_ids", set()) or set())
    return {int(k): v for k, v in dict(req_map).items() if int(k) in active}


def _request_snapshot(req: Any) -> dict[str, Any]:
    arrived = _safe_float(getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0)), 0.0)
    queued = _safe_float(getattr(req, "queued_at", arrived), arrived)
    total_decode = _safe_int(getattr(req, "_num_decode_tokens", getattr(req, "num_decode_tokens", 0)), 0)
    return {
        "request_id": _safe_int(getattr(req, "id", -1), -1),
        "arrived_at": arrived,
        "queued_at": queued,
        "num_prefill_tokens": _safe_int(getattr(req, "num_prefill_tokens", 0), 0),
        "num_processed_prefill_tokens": _safe_int(getattr(req, "num_processed_prefill_tokens", 0), 0),
        "num_decode_tokens": total_decode,
        "num_processed_decode_tokens": _safe_int(getattr(req, "num_processed_decode_tokens", 0), 0),
        "is_prefill_complete": _safe_bool(
            getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)),
            False,
        ),
        "completed": _safe_bool(getattr(req, "completed", False), False),
        "prefill_slo_time": _safe_float(getattr(req, "_prefill_slo_time", getattr(req, "prefill_slo_time", 0.0)), 0.0),
        "decode_slo_time": _safe_float(getattr(req, "_decode_slo_time", getattr(req, "decode_slo_time", 0.0)), 0.0),
        "prefill_completed_at": _safe_float(getattr(req, "_prefill_completed_at", 0.0), 0.0),
    }


def build_frontier_state_row(
    *,
    env: Any,
    state: Any,
    root_id: int,
    root_player: str,
    root_depth: int,
    history_hops: int,
    history_log_node_id: int | None,
) -> dict[str, Any]:
    stats = getattr(state, "stats", None)
    requests = [_request_snapshot(req) for _, req in sorted(_request_lookup(env, state).items())]
    violations, lateness = env.evaluate_objective(state)
    desc = env.describe_state(state)
    return {
        "root_id": int(root_id),
        "root_player": str(root_player),
        "root_depth": int(root_depth),
        "history_hops": int(history_hops),
        "history_log_node_id": "" if history_log_node_id is None else int(history_log_node_id),
        "sim_time": _safe_float(getattr(state.simulator, "_time", 0.0), 0.0),
        "objective_cost": float(violations) + float(lateness),
        "slo_violations": int(violations),
        "slo_lateness_sum": float(lateness),
        "active_request_ids_json": _json_dumps(sorted(int(x) for x in getattr(stats, "active_request_ids", set()) or set())),
        "completed_request_ids_json": _json_dumps(sorted(int(x) for x in getattr(stats, "completed_request_ids", set()) or set())),
        "dropped_request_ids_json": _json_dumps(sorted(int(x) for x in getattr(stats, "dropped_request_ids", set()) or set())),
        "stopped_decode_request_ids_json": _json_dumps(sorted(int(x) for x in getattr(stats, "stopped_decode_request_ids", set()) or set())),
        "violated_request_ids_json": _json_dumps(sorted(int(x) for x in getattr(stats, "violated_request_ids", set()) or set())),
        "per_request_prefill_lateness_json": _json_dumps(getattr(stats, "per_request_prefill_lateness", {}) or {}),
        "per_request_decode_lateness_json": _json_dumps(getattr(stats, "per_request_decode_lateness", {}) or {}),
        "decode_next_deadline_by_id_json": _json_dumps(getattr(stats, "decode_next_deadline_by_id", {}) or {}),
        "decode_tokens_counted_by_id_json": _json_dumps(getattr(stats, "decode_tokens_counted", {}) or {}),
        "recent_arrivals_json": _json_dumps(getattr(stats, "recent_arrivals", []) or []),
        "pending_adv_tick": str(bool(desc.get("pending_adv_tick", False))).lower(),
        "last_adv_tick": desc.get("last_adv_tick", ""),
        "requests_json": _json_dumps(requests),
    }


def build_production_feature_row(
    *,
    env: Any,
    state: Any,
    root_id: int,
    root_player: str,
    root_depth: int,
    history_hops: int,
    action_mask_fn: Any,
) -> dict[str, Any]:
    inputs = build_model_inputs(
        state,
        str(root_player),
        torch.device("cpu"),
        build_action_mask_flag=True,
        action_mask_fn=action_mask_fn,
    )
    del env
    return {
        "root_id": int(root_id),
        "root_player": str(root_player),
        "root_depth": int(root_depth),
        "history_hops": int(history_hops),
        "prefill_req_features_json": _tensor_json(inputs.prefill_req_features),
        "decode_req_features_json": _tensor_json(inputs.decode_req_features),
        "global_features_json": _tensor_json(inputs.global_features),
        "prefill_req_mask_json": _mask_json(inputs.prefill_req_mask),
        "decode_req_mask_json": _mask_json(inputs.decode_req_mask),
        "req_features_json": _tensor_json(inputs.req_features),
        "req_mask_json": _mask_json(inputs.req_mask),
        "action_mask_json": _mask_json(inputs.action_mask),
    }


def _prefill_remaining(req: dict[str, Any]) -> int:
    return max(0, _safe_int(req.get("num_prefill_tokens"), 0) - _safe_int(req.get("num_processed_prefill_tokens"), 0))


def _decode_remaining(req: dict[str, Any]) -> int:
    return max(0, _safe_int(req.get("num_decode_tokens"), 0) - _safe_int(req.get("num_processed_decode_tokens"), 0))


def _prefill_deadline(req: dict[str, Any]) -> float | None:
    slo = _safe_float(req.get("prefill_slo_time"), 0.0)
    if slo <= 0.0:
        return None
    return _safe_float(req.get("queued_at"), _safe_float(req.get("arrived_at"), 0.0)) + slo


def _decode_deadline(req: dict[str, Any], decode_next_deadline: dict[int, float]) -> float | None:
    rid = _safe_int(req.get("request_id"), -1)
    deadline = _safe_float(decode_next_deadline.get(rid, 0.0), 0.0)
    if deadline > 0.0:
        return deadline
    prefill_completed_at = _safe_float(req.get("prefill_completed_at"), 0.0)
    decode_slo = _safe_float(req.get("decode_slo_time"), 0.0)
    if prefill_completed_at > 0.0 and decode_slo > 0.0:
        return prefill_completed_at + decode_slo
    return None


def _is_prefill_request(req: dict[str, Any]) -> bool:
    return (
        not _safe_bool(req.get("completed"), False)
        and not _safe_bool(req.get("is_prefill_complete"), False)
        and _prefill_remaining(req) > 0
    )


def _is_decode_request(req: dict[str, Any]) -> bool:
    return (
        not _safe_bool(req.get("completed"), False)
        and _safe_bool(req.get("is_prefill_complete"), False)
        and _decode_remaining(req) > 0
    )


def _request_total_lateness(
    req: dict[str, Any],
    *,
    sim_time: float,
    prefill_lateness: dict[int, float],
    decode_lateness: dict[int, float],
) -> float:
    rid = _safe_int(req.get("request_id"), -1)
    pref = float(prefill_lateness.get(rid, 0.0))
    dec = float(decode_lateness.get(rid, 0.0))
    if pref <= 0.0:
        deadline = _prefill_deadline(req)
        if deadline is not None:
            pref = max(0.0, float(sim_time) - float(deadline))
    return max(0.0, pref + dec)


def _recent_launch_summary(row: dict[str, str], *, sim_time: float, window_sec: float, alpha: float) -> tuple[float, float, float]:
    launch_count = 0.0
    launch_prefill = 0.0
    ewma = 0.0
    for item in list(_json_loads(row.get("recent_arrivals_json"), []) or []):
        ts = None
        cnt = 0
        prefill = 0
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            ts = float(item[0])
            cnt = int(item[1])
            prefill = int(item[2])
        elif isinstance(item, dict):
            ts = float(item.get("timestamp", item.get("time", 0.0)))
            cnt = int(item.get("count", item.get("requests", 0)))
            prefill = int(item.get("prefill_tokens", item.get("tokens", 0)))
        elif isinstance(item, (int, float)):
            ts = float(item)
            cnt = 1
        if ts is None:
            continue
        dt = max(0.0, float(sim_time) - float(ts))
        if dt > float(window_sec):
            continue
        launch_count += max(0.0, float(cnt))
        launch_prefill += max(0.0, float(prefill))
        ewma += max(0.0, float(cnt)) * math.exp(-float(alpha) * dt)
    return launch_count, launch_prefill, ewma


def expected_features_from_frontier_row(row: dict[str, str], cfg: Any) -> dict[str, Any]:
    F = cfg.features
    sim_time = _safe_float(row.get("sim_time"), 0.0)
    player = str(row.get("root_player", ""))
    requests = list(_json_loads(row.get("requests_json"), []) or [])
    active_ids = set(_int_list(_json_loads(row.get("active_request_ids_json"), [])))
    terminal_ids = (
        set(_int_list(_json_loads(row.get("completed_request_ids_json"), [])))
        | set(_int_list(_json_loads(row.get("dropped_request_ids_json"), [])))
        | set(_int_list(_json_loads(row.get("stopped_decode_request_ids_json"), [])))
    )
    violated_ids = set(_int_list(_json_loads(row.get("violated_request_ids_json"), [])))
    prefill_lateness = _int_float_map(_json_loads(row.get("per_request_prefill_lateness_json"), {}))
    decode_lateness = _int_float_map(_json_loads(row.get("per_request_decode_lateness_json"), {}))
    decode_next_deadline = _int_float_map(_json_loads(row.get("decode_next_deadline_by_id_json"), {}))
    decode_counted = _int_int_map(_json_loads(row.get("decode_tokens_counted_by_id_json"), {}))

    reqs = [dict(r) for r in requests if _safe_int(dict(r).get("request_id"), -1) in active_ids]
    prefill_reqs = [r for r in reqs if _is_prefill_request(r)]
    decode_reqs = [r for r in reqs if _is_decode_request(r)]

    def prefill_key(req: dict[str, Any]) -> tuple[float, float, int]:
        deadline = _prefill_deadline(req)
        time_left = float("inf") if deadline is None else float(deadline) - sim_time
        late = _request_total_lateness(req, sim_time=sim_time, prefill_lateness=prefill_lateness, decode_lateness=decode_lateness)
        return time_left, -late, _safe_int(req.get("request_id"), 0)

    def decode_key(req: dict[str, Any]) -> tuple[int, float, int, int, int]:
        rid = _safe_int(req.get("request_id"), 0)
        late = _request_total_lateness(req, sim_time=sim_time, prefill_lateness=prefill_lateness, decode_lateness=decode_lateness)
        return (
            -1 if rid in violated_ids else 0,
            -late,
            -_safe_int(req.get("num_processed_decode_tokens"), 0),
            -_decode_remaining(req),
            rid,
        )

    prefill_reqs.sort(key=prefill_key)
    decode_reqs.sort(key=decode_key)

    prefill_feat = torch.zeros((1, int(F.n_prefill_req), int(F.d_prefill_req)), dtype=torch.float32)
    decode_feat = torch.zeros((1, int(F.n_decode_req), int(F.d_decode_req)), dtype=torch.float32)
    prefill_mask = torch.zeros((1, int(F.n_prefill_req)), dtype=torch.bool)
    decode_mask = torch.zeros((1, int(F.n_decode_req)), dtype=torch.bool)

    prefill_slot_ids: list[int] = []
    decode_slot_ids: list[int] = []

    for i, req in enumerate(prefill_reqs[: int(F.n_prefill_req)]):
        rid = _safe_int(req.get("request_id"), -1)
        total_prefill = _safe_int(req.get("num_prefill_tokens"), 0)
        rem_prefill = _prefill_remaining(req)
        done_prefill = max(0, total_prefill - rem_prefill)
        age = max(0.0, sim_time - _safe_float(req.get("arrived_at"), 0.0))
        prefill_late = _safe_float(prefill_lateness.get(rid, 0.0), 0.0)
        deadline = _prefill_deadline(req)
        slack = 0.0 if deadline is None else float(deadline) - sim_time
        prefill_slo = _safe_float(req.get("prefill_slo_time"), 0.0)
        processed_frac = 0.0 if total_prefill <= 0 else _clip(float(done_prefill) / float(max(1, total_prefill)), 0.0, 1.0)
        values = [
            _norm01(rem_prefill, F.prefill_remaining_den),
            _norm01(total_prefill, F.prefill_total_den),
            processed_frac,
            _norm01(age, F.age_den_sec),
            _norm01(prefill_late, F.lateness_den_sec),
            _centered01(slack, F.slack_den_sec),
            _norm01(prefill_slo, F.prefill_slo_den_sec),
            1.0 if rid in violated_ids else 0.0,
            1.0 if prefill_late > float(F.near_drop_lateness_low_sec) else 0.0,
            1.0 if prefill_late >= float(F.near_drop_lateness_high_sec) else 0.0,
        ]
        prefill_feat[0, i, :] = torch.tensor(values, dtype=torch.float32)
        prefill_mask[0, i] = True
        prefill_slot_ids.append(int(rid))

    for i, req in enumerate(decode_reqs[: int(F.n_decode_req)]):
        rid = _safe_int(req.get("request_id"), -1)
        total_decode = _safe_int(req.get("num_decode_tokens"), 0)
        rem_decode = _decode_remaining(req)
        done_decode = max(0, total_decode - rem_decode)
        age = max(0.0, sim_time - _safe_float(req.get("arrived_at"), 0.0))
        late = _request_total_lateness(req, sim_time=sim_time, prefill_lateness=prefill_lateness, decode_lateness=decode_lateness)
        deadline = _decode_deadline(req, decode_next_deadline)
        slack = 0.0 if deadline is None else float(deadline) - sim_time
        decode_slo = _safe_float(req.get("decode_slo_time"), 0.0)
        processed_frac = 0.0 if total_decode <= 0 else _clip(float(done_decode) / float(max(1, total_decode)), 0.0, 1.0)
        values = [
            _norm01(rem_decode, F.decode_remaining_den),
            _norm01(total_decode, F.decode_total_den),
            _norm01(done_decode, F.decode_processed_den),
            processed_frac,
            _norm01(age, F.age_den_sec),
            _norm01(late, F.lateness_den_sec),
            _centered01(slack, F.slack_den_sec),
            _norm01(decode_slo, F.decode_slo_den_sec),
            1.0 if rid in violated_ids else 0.0,
            1.0 if late > float(F.near_drop_lateness_low_sec) else 0.0,
            1.0 if late >= float(F.near_drop_lateness_high_sec) else 0.0,
            1.0 if done_decode > 216 else 0.0,
            1.0 if done_decode > 512 else 0.0,
        ]
        decode_feat[0, i, :] = torch.tensor(values, dtype=torch.float32)
        decode_mask[0, i] = True
        decode_slot_ids.append(int(rid))

    active_feature_ids = {_safe_int(req.get("request_id"), -1) for req in reqs}
    num_violated_active = len([rid for rid in active_feature_ids if rid in violated_ids])

    p_late_05_15 = 0
    p_late_15 = 0
    for req in prefill_reqs:
        late = _safe_float(prefill_lateness.get(_safe_int(req.get("request_id"), -1), 0.0), 0.0)
        if late > float(F.near_drop_lateness_low_sec) and late < float(F.near_drop_lateness_high_sec):
            p_late_05_15 += 1
        elif late >= float(F.near_drop_lateness_high_sec):
            p_late_15 += 1

    d_late_05_15 = 0
    d_late_15 = 0
    for req in decode_reqs:
        late = _request_total_lateness(req, sim_time=sim_time, prefill_lateness=prefill_lateness, decode_lateness=decode_lateness)
        if late > float(F.near_drop_lateness_low_sec) and late < float(F.near_drop_lateness_high_sec):
            d_late_05_15 += 1
        elif late >= float(F.near_drop_lateness_high_sec):
            d_late_15 += 1

    num_prefill = len(prefill_reqs)
    num_decode = len(decode_reqs)
    num_active = len(reqs)
    total_remaining_prefill = sum(_prefill_remaining(r) for r in prefill_reqs)
    total_remaining_decode = sum(_decode_remaining(r) for r in decode_reqs)
    total_decode_generated_active = sum(_safe_int(r.get("num_processed_decode_tokens"), 0) for r in decode_reqs)
    slo_violations = _safe_int(row.get("slo_violations"), 0)
    slo_lateness_sum = _safe_float(row.get("slo_lateness_sum"), 0.0)
    objective_cost = float(slo_violations) + float(slo_lateness_sum)
    launch_count, launch_prefill, ewma = _recent_launch_summary(
        row,
        sim_time=sim_time,
        window_sec=float(F.launch_ewma_window_sec),
        alpha=float(F.launch_ewma_alpha),
    )
    max_requests_window = float(max(1, int(cfg.timing.max_requests_per_launch_window)))
    max_prefill_window = float(
        max(
            1,
            int(cfg.request.target_prefill_tokens_per_request_avg_window)
            * int(cfg.timing.max_requests_per_launch_window),
        )
    )
    remaining_launch_request_headroom = max(0.0, max_requests_window - float(launch_count))
    remaining_launch_prefill_headroom = max(0.0, max_prefill_window - float(launch_prefill))
    decode_credit = max(0.0, float(decode_counted.get(META_DECODE_CREDIT_BAL, 0)))

    global_feat = torch.tensor(
        [[
            1.0 if player == "controller" else 0.0,
            1.0 if player == "adversary" else 0.0,
            _norm01(objective_cost, F.objective_cost_den),
            _norm01(slo_violations, F.violated_count_den),
            _norm01(slo_lateness_sum, F.total_lateness_den),
            _norm01(num_prefill, F.active_prefill_count_den),
            _norm01(num_decode, F.active_decode_count_den),
            _norm01(num_active, F.active_total_count_den),
            _norm01(total_remaining_prefill, F.total_remaining_prefill_den),
            _norm01(total_remaining_decode, F.total_remaining_decode_den),
            _norm01(total_decode_generated_active, F.total_decode_generated_active_den),
            _norm01(num_violated_active, F.violated_count_den),
            _norm01(p_late_05_15, F.prefill_near_drop_den),
            _norm01(p_late_15, F.prefill_near_drop_den),
            _norm01(d_late_05_15, F.decode_near_drop_den),
            _norm01(d_late_15, F.decode_near_drop_den),
            _norm01(launch_count, F.recent_launch_count_den),
            _norm01(launch_prefill, F.recent_launch_prefill_den),
            _norm01(remaining_launch_request_headroom, max_requests_window),
            _norm01(remaining_launch_prefill_headroom, max_prefill_window),
            _norm01(ewma, F.recent_launch_count_den),
            _norm01(decode_credit, F.decode_credit_den),
            1.0 if num_prefill > 0 else 0.0,
            1.0 if num_decode > 0 else 0.0,
        ]],
        dtype=torch.float32,
    )

    legacy_width = max(int(prefill_feat.size(-1)), int(decode_feat.size(-1)))
    req_features = torch.cat(
        [
            Fnn.pad(prefill_feat, (0, legacy_width - int(prefill_feat.size(-1)))),
            Fnn.pad(decode_feat, (0, legacy_width - int(decode_feat.size(-1)))),
        ],
        dim=1,
    )
    req_mask = torch.cat([prefill_mask, decode_mask], dim=1)
    feature_ids = set(prefill_slot_ids) | set(decode_slot_ids)

    return {
        "prefill_req_features": prefill_feat,
        "decode_req_features": decode_feat,
        "global_features": global_feat,
        "prefill_req_mask": prefill_mask,
        "decode_req_mask": decode_mask,
        "req_features": req_features,
        "req_mask": req_mask,
        "prefill_slot_request_ids": prefill_slot_ids,
        "decode_slot_request_ids": decode_slot_ids,
        "active_request_ids": sorted(active_ids),
        "terminal_request_ids": sorted(terminal_ids),
        "feature_request_ids": sorted(feature_ids),
        "feature_ids_subset_active": feature_ids.issubset(active_ids),
        "feature_ids_disjoint_terminal": not bool(feature_ids & terminal_ids),
    }


def _tensor_from_json(value: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    data = _json_loads(value, [])
    tensor = torch.tensor(data, dtype=dtype)
    return tensor.reshape(shape)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return math.inf
    return float(torch.max(torch.abs(a.to(torch.float32) - b.to(torch.float32))).item())


def compare_feature_rows(
    *,
    frontier_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
    cfg: Any,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    by_root = {int(row["root_id"]): row for row in feature_rows}
    rows: list[dict[str, Any]] = []
    failures: list[int] = []
    F = cfg.features

    for frontier_row in frontier_rows:
        root_id = int(frontier_row["root_id"])
        feature_row = by_root[root_id]
        expected = expected_features_from_frontier_row(frontier_row, cfg)
        actual_prefill = _tensor_from_json(
            feature_row["prefill_req_features_json"],
            (1, int(F.n_prefill_req), int(F.d_prefill_req)),
            torch.float32,
        )
        actual_decode = _tensor_from_json(
            feature_row["decode_req_features_json"],
            (1, int(F.n_decode_req), int(F.d_decode_req)),
            torch.float32,
        )
        actual_global = _tensor_from_json(
            feature_row["global_features_json"],
            tuple(expected["global_features"].shape),
            torch.float32,
        )
        actual_prefill_mask = _tensor_from_json(
            feature_row["prefill_req_mask_json"],
            tuple(expected["prefill_req_mask"].shape),
            torch.bool,
        )
        actual_decode_mask = _tensor_from_json(
            feature_row["decode_req_mask_json"],
            tuple(expected["decode_req_mask"].shape),
            torch.bool,
        )
        actual_req_features = _tensor_from_json(
            feature_row["req_features_json"],
            tuple(expected["req_features"].shape),
            torch.float32,
        )
        actual_req_mask = _tensor_from_json(
            feature_row["req_mask_json"],
            tuple(expected["req_mask"].shape),
            torch.bool,
        )

        prefill_diff = _max_abs_diff(actual_prefill, expected["prefill_req_features"])
        decode_diff = _max_abs_diff(actual_decode, expected["decode_req_features"])
        global_diff = _max_abs_diff(actual_global, expected["global_features"])
        req_diff = _max_abs_diff(actual_req_features, expected["req_features"])
        prefill_mask_match = bool(torch.equal(actual_prefill_mask, expected["prefill_req_mask"]))
        decode_mask_match = bool(torch.equal(actual_decode_mask, expected["decode_req_mask"]))
        req_mask_match = bool(torch.equal(actual_req_mask, expected["req_mask"]))
        max_diff = max(prefill_diff, decode_diff, global_diff, req_diff)
        passed = (
            max_diff <= float(tolerance)
            and prefill_mask_match
            and decode_mask_match
            and req_mask_match
            and bool(expected["feature_ids_subset_active"])
            and bool(expected["feature_ids_disjoint_terminal"])
        )
        if not passed:
            failures.append(root_id)
        rows.append(
            {
                "root_id": root_id,
                "root_player": frontier_row.get("root_player", ""),
                "root_depth": frontier_row.get("root_depth", ""),
                "history_hops": frontier_row.get("history_hops", ""),
                "passed": passed,
                "max_feature_diff": max_diff,
                "prefill_diff": prefill_diff,
                "decode_diff": decode_diff,
                "global_diff": global_diff,
                "req_diff": req_diff,
                "prefill_mask_match": prefill_mask_match,
                "decode_mask_match": decode_mask_match,
                "req_mask_match": req_mask_match,
                "feature_ids_subset_active": bool(expected["feature_ids_subset_active"]),
                "feature_ids_disjoint_terminal": bool(expected["feature_ids_disjoint_terminal"]),
                "active_request_ids_json": _json_dumps(expected["active_request_ids"]),
                "terminal_request_ids_json": _json_dumps(expected["terminal_request_ids"]),
                "prefill_feature_request_ids_json": _json_dumps(expected["prefill_slot_request_ids"]),
                "decode_feature_request_ids_json": _json_dumps(expected["decode_slot_request_ids"]),
            }
        )

    return rows, failures


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, float):
                    value = f"{value:.10g}"
                out[key] = value
            writer.writerow(out)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

