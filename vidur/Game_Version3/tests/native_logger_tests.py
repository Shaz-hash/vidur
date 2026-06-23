from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.infer import build_model_inputs
from ..DNN.native_selfplay import (
    _cfg_payload,
    _empty_initial_state_payload,
    attach_execution_predictor_payload,
    export_torchscript_pair,
)
from ..DNN.selfPlay import SelfPlayRunner, SingleRootRun
from ..DNN.value_models import AlphaZeroModel
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import _build_env_and_simulator, _load_weights_into_model, _set_global_seeds
from .feature_conversion_tests import _make_action_mask_fn, _max_abs_diff
from .frontier_feature_trace_logger import compare_feature_rows, read_csv, write_csv
from .frontier_feature_trace_tests import COMPARE_FIELDS, FEATURE_FIELDS
from .history_node_tests import _validate_trace_csv
from .test_checking_depth1 import _NoopWriter, _details_for_root


META_NEXT_ADV_TICK = -9_100_001
META_LAST_ADV_TICK = -9_100_002
META_DECODE_CREDIT_BAL = -9_100_005
META_MISSED_ADV_SOURCE = -9_100_006


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vidur" / "Game_Version3").is_dir():
            return parent
    return p.parents[3]


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "tests" / "native_logger_tests"


def _default_checkpoint_path() -> Path:
    candidates = [
        _repo_root() / "simulator_output" / "Game_Version3_Fresh8_Hops200" / "mcts_dnn_checkpoints" / "best.pt",
        _repo_root() / "simulator_output" / "Game_Version3" / "mcts_dnn_checkpoints" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native GV3 correctness gate: logger, frontier signatures, features, and bootstrap Bellman parity."
    )
    parser.add_argument("--num-roots", type=int, default=4)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=8)
    parser.add_argument("--history-seed", type=int, default=2026)
    parser.add_argument("--frontier-parity-roots", type=int, default=1)
    parser.add_argument("--feature-tolerance", type=float, default=1e-6)
    parser.add_argument("--bellman-q-tolerance", type=float, default=1e-2)
    parser.add_argument("--strict-adversary-q-parity", action="store_true")
    parser.add_argument("--skip-bootstrap-bellman", action="store_true")
    parser.add_argument("--model-version", type=int, default=1)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--checkpoint-path", default=str(_default_checkpoint_path()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
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
                elif value is None:
                    value = ""
                out[key] = value
            writer.writerow(out)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        if value is None:
            return default
        return bool(value)
    except Exception:
        return default


def _rid(req: Any) -> int:
    return _safe_int(_safe_getattr(req, "id", _safe_getattr(req, "request_id", -1)), -1)


def _json_int_list(raw: Any) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [int(x) for x in list(raw or [])]


def _id_list(raw: Any) -> list[int]:
    try:
        return sorted(int(x) for x in list(raw or []))
    except Exception:
        return []


def _stats_map(stats: Any, name: str) -> dict[int, Any]:
    raw = _safe_getattr(stats, name, {}) or {}
    return {int(k): v for k, v in dict(raw).items()}


def _request_lookup(env: Any, state: Any) -> dict[int, Any]:
    if hasattr(env, "_build_request_lookup"):
        return {int(k): v for k, v in env._build_request_lookup(state.simulator, state=state).items()}
    return {int(k): v for k, v in env._req_map(state.simulator).items()}


def _prefill_deadline(req: Any) -> float:
    deadline = _safe_getattr(req, "prefill_deadline", None)
    if deadline is not None:
        return _safe_float(deadline, 0.0)
    queued = _safe_float(_safe_getattr(req, "queued_at", _safe_getattr(req, "_arrived_at", 0.0)), 0.0)
    slo = _safe_float(_safe_getattr(req, "_prefill_slo_time", _safe_getattr(req, "prefill_slo_time", 0.0)), 0.0)
    return queued + slo if slo > 0.0 else 0.0


def _snapshot_request_counter(state: Any) -> int | None:
    try:
        snap = state.simulator.snapshot_state_fast()
    except Exception:
        try:
            snap = state.simulator.snapshot_state()
        except Exception:
            return None
    counters = getattr(snap, "entity_counters", None)
    if isinstance(counters, dict):
        for key in ("Request", "request"):
            if key in counters:
                return int(counters[key])
    if isinstance(snap, dict):
        counters = snap.get("entity_counters") or {}
        if isinstance(counters, dict):
            for key in ("Request", "request"):
                if key in counters:
                    return int(counters[key])
    return None


def _native_state_payload(env: Any, state: Any) -> dict[str, Any]:
    sim = state.simulator
    stats = _safe_getattr(state, "stats", None)
    sim_time = _safe_float(_safe_getattr(sim, "_time", 0.0), 0.0)
    req_map = _request_lookup(env, state)

    requests: list[dict[str, Any]] = []
    for rid, req in sorted(req_map.items()):
        arrived = _safe_float(_safe_getattr(req, "_arrived_at", _safe_getattr(req, "arrived_at", 0.0)), 0.0)
        queued = _safe_float(_safe_getattr(req, "queued_at", arrived), arrived)
        is_prefill_complete = _safe_bool(
            _safe_getattr(req, "_is_prefill_complete", _safe_getattr(req, "is_prefill_complete", False)),
            False,
        )
        completed = _safe_bool(_safe_getattr(req, "completed", False), False)
        prefill_completed_at = _safe_float(
            _safe_getattr(req, "_prefill_completed_at", _safe_getattr(req, "prefill_completed_at", -1.0)),
            -1.0,
        )
        if not is_prefill_complete and prefill_completed_at <= 0.0:
            prefill_completed_at = -1.0
        completed_at = _safe_float(_safe_getattr(req, "completed_at", -1.0), -1.0)
        if not completed and completed_at <= 0.0:
            completed_at = -1.0
        decode_next_deadline = _safe_float(
            _stats_map(stats, "decode_next_deadline_by_id").get(int(rid), -1.0),
            _safe_float(_safe_getattr(req, "decode_next_deadline", -1.0), -1.0),
        )
        if not is_prefill_complete and decode_next_deadline <= 0.0:
            decode_next_deadline = -1.0
        requests.append(
            {
                "request_id": int(rid),
                "arrived_at": arrived,
                "queued_at": queued,
                "num_prefill_tokens": _safe_int(_safe_getattr(req, "num_prefill_tokens", 0), 0),
                "num_processed_prefill_tokens": _safe_int(
                    _safe_getattr(req, "num_processed_prefill_tokens", 0), 0
                ),
                "num_decode_tokens": _safe_int(
                    _safe_getattr(req, "_num_decode_tokens", _safe_getattr(req, "num_decode_tokens", 0)), 0
                ),
                "num_processed_decode_tokens": _safe_int(
                    _safe_getattr(req, "num_processed_decode_tokens", 0), 0
                ),
                "prefill_slo_time": _safe_float(
                    _safe_getattr(req, "_prefill_slo_time", _safe_getattr(req, "prefill_slo_time", 0.0)), 0.0
                ),
                "decode_slo_time": _safe_float(
                    _safe_getattr(req, "_decode_slo_time", _safe_getattr(req, "decode_slo_time", 0.0)), 0.0
                ),
                "prefill_deadline": _prefill_deadline(req),
                "decode_next_deadline": decode_next_deadline,
                "prefill_completed_at": prefill_completed_at,
                "completed_at": completed_at,
                "prefill_lateness": _safe_float(
                    _stats_map(stats, "per_request_prefill_lateness").get(int(rid), 0.0),
                    _safe_float(_safe_getattr(req, "prefill_lateness", 0.0), 0.0),
                ),
                "decode_lateness": _safe_float(
                    _stats_map(stats, "per_request_decode_lateness").get(int(rid), 0.0),
                    _safe_float(_safe_getattr(req, "decode_lateness", 0.0), 0.0),
                ),
                "is_prefill_complete": is_prefill_complete,
                "completed": completed,
                "dropped": _safe_bool(_safe_getattr(req, "dropped", False), False),
                "stopped_decode": _safe_bool(_safe_getattr(req, "stopped_decode", False), False),
                "violated": int(rid) in set(_id_list(_safe_getattr(stats, "violated_request_ids", []))),
            }
        )


    # HGB-v4 inference in Python rebuilds its 226D features from
    # ``simulator.snapshot_state_fast()["request_states"]`` plus stats.  The
    # env request lookup is enough for engine dynamics, but it can omit
    # finalized request records that are still present in the simulator
    # snapshot.  Preserve the snapshot request records for native inference so
    # C++ sees the same raw feature input as the Python wrapper.
    snapshot_request_states = {}
    try:
        snap_for_requests = sim.snapshot_state_fast() if hasattr(sim, "snapshot_state_fast") else sim.snapshot_state()
        if isinstance(snap_for_requests, dict):
            snapshot_request_states = dict(snap_for_requests.get("request_states") or {})
    except Exception:
        snapshot_request_states = {}

    if snapshot_request_states:
        snapshot_requests: list[dict[str, Any]] = []
        completed_ids_for_payload = set(_id_list(_safe_getattr(stats, "completed_request_ids", [])))
        dropped_ids_for_payload = set(_id_list(_safe_getattr(stats, "dropped_request_ids", [])))
        stopped_ids_for_payload = set(_id_list(_safe_getattr(stats, "stopped_decode_request_ids", [])))
        active_ids_for_payload = set(_id_list(_safe_getattr(stats, "active_request_ids", [])))
        finalized_ids_for_payload = completed_ids_for_payload | dropped_ids_for_payload | stopped_ids_for_payload
        for raw_key, raw_value in sorted(snapshot_request_states.items(), key=lambda kv: int(kv[0])):
            if not isinstance(raw_value, dict):
                continue
            rid = _safe_int(raw_value.get("request_id", raw_value.get("id", raw_key)), -1)
            if rid < 0:
                continue
            req = req_map.get(int(rid))

            def _raw_or_attr(names: tuple[str, ...], attr: str, default: Any) -> Any:
                for name in names:
                    if name in raw_value and raw_value.get(name) is not None:
                        return raw_value.get(name)
                if req is not None:
                    return _safe_getattr(req, attr, default)
                return default

            arrived = _safe_float(
                _raw_or_attr(("arrived_at", "_arrived_at"), "_arrived_at", 0.0),
                0.0,
            )
            queued = _safe_float(
                _raw_or_attr(("queued_at",), "queued_at", arrived),
                arrived,
            )
            num_prefill = _safe_int(
                _raw_or_attr(("num_prefill_tokens", "prefill_tokens"), "num_prefill_tokens", 0),
                0,
            )
            num_decode = _safe_int(
                _raw_or_attr(("num_decode_tokens", "decode_tokens", "_num_decode_tokens"), "_num_decode_tokens", 0),
                0,
            )
            total_processed = _safe_int(raw_value.get("num_processed_tokens", 0), 0)
            processed_prefill = _safe_int(
                raw_value.get("num_processed_prefill_tokens", raw_value.get("processed_prefill_tokens", None)),
                -1,
            )
            if processed_prefill < 0:
                processed_prefill = _safe_int(
                    _safe_getattr(req, "num_processed_prefill_tokens", min(total_processed, num_prefill)) if req is not None else min(total_processed, num_prefill),
                    min(total_processed, num_prefill),
                )
            processed_decode = _safe_int(
                raw_value.get("num_processed_decode_tokens", raw_value.get("processed_decode_tokens", None)),
                -1,
            )
            if processed_decode < 0:
                processed_decode = _safe_int(
                    _safe_getattr(req, "num_processed_decode_tokens", max(0, total_processed - num_prefill)) if req is not None else max(0, total_processed - num_prefill),
                    max(0, total_processed - num_prefill),
                )

            is_prefill_complete = _safe_bool(
                _raw_or_attr(("is_prefill_complete", "prefill_complete", "_is_prefill_complete"), "_is_prefill_complete", processed_prefill >= num_prefill),
                processed_prefill >= num_prefill,
            )
            completed = _safe_bool(_raw_or_attr(("completed",), "completed", False), False)
            if int(rid) in finalized_ids_for_payload:
                completed = True
            feature_only = (int(rid) not in active_ids_for_payload) and (int(rid) not in finalized_ids_for_payload)
            prefill_slo = _safe_float(
                _raw_or_attr(("prefill_slo_time", "prefill_slo", "_prefill_slo_time"), "_prefill_slo_time", 0.0),
                0.0,
            )
            decode_slo = _safe_float(
                _raw_or_attr(("decode_slo_time", "decode_slo", "_decode_slo_time"), "_decode_slo_time", 0.0),
                0.0,
            )
            prefill_deadline = _safe_float(raw_value.get("prefill_deadline", 0.0), 0.0)
            if prefill_deadline <= 0.0:
                if req is not None:
                    prefill_deadline = _prefill_deadline(req)
                elif prefill_slo > 0.0:
                    prefill_deadline = queued + prefill_slo
            prefill_completed_at = _safe_float(
                _raw_or_attr(("prefill_completed_at", "_prefill_completed_at"), "_prefill_completed_at", -1.0),
                -1.0,
            )
            if not is_prefill_complete and prefill_completed_at <= 0.0:
                prefill_completed_at = -1.0
            completed_at = _safe_float(_raw_or_attr(("completed_at",), "completed_at", -1.0), -1.0)
            if not completed and completed_at <= 0.0:
                completed_at = -1.0

            decode_next_deadline = _safe_float(
                _stats_map(stats, "decode_next_deadline_by_id").get(
                    int(rid),
                    _safe_float(raw_value.get("decode_next_deadline", raw_value.get("decode_deadline", -1.0)), -1.0),
                ),
                -1.0,
            )
            if not is_prefill_complete and decode_next_deadline <= 0.0:
                decode_next_deadline = -1.0

            snapshot_requests.append(
                {
                    "request_id": int(rid),
                    "id": int(rid),
                    "arrived_at": arrived,
                    "queued_at": queued,
                    "num_prefill_tokens": num_prefill,
                    "num_processed_prefill_tokens": max(0, processed_prefill),
                    "num_decode_tokens": num_decode,
                    "num_processed_decode_tokens": max(0, processed_decode),
                    "num_processed_tokens": max(0, processed_prefill) + max(0, processed_decode),
                    "remaining_prefill_tokens": max(0, num_prefill - max(0, processed_prefill)),
                    "prefill_slo_time": prefill_slo,
                    "decode_slo_time": decode_slo,
                    "prefill_deadline": prefill_deadline,
                    "decode_next_deadline": decode_next_deadline,
                    "prefill_completed_at": prefill_completed_at,
                    "completed_at": completed_at,
                    "prefill_lateness": _safe_float(
                        _stats_map(stats, "per_request_prefill_lateness").get(int(rid), raw_value.get("prefill_lateness", 0.0)),
                        0.0,
                    ),
                    "decode_lateness": _safe_float(
                        _stats_map(stats, "per_request_decode_lateness").get(int(rid), raw_value.get("decode_lateness", 0.0)),
                        0.0,
                    ),
                    "is_prefill_complete": is_prefill_complete,
                    "completed": completed,
                    "dropped": (int(rid) in dropped_ids_for_payload) or _safe_bool(raw_value.get("dropped", _safe_getattr(req, "dropped", False) if req is not None else False), False),
                    "stopped_decode": (int(rid) in stopped_ids_for_payload) or _safe_bool(raw_value.get("stopped_decode", _safe_getattr(req, "stopped_decode", False) if req is not None else False), False),
                    "violated": int(rid) in set(_id_list(_safe_getattr(stats, "violated_request_ids", []))),
                    "feature_only": bool(feature_only),
                }
            )
        if snapshot_requests:
            requests = snapshot_requests

    decode_deadlines = {
        int(k): float(v)
        for k, v in _stats_map(stats, "decode_next_deadline_by_id").items()
    }
    next_adv_tick = _safe_float(
        _safe_getattr(stats, "next_adv_tick", decode_deadlines.get(META_NEXT_ADV_TICK, 0.0)),
        _safe_float(decode_deadlines.get(META_NEXT_ADV_TICK, 0.0), 0.0),
    )
    last_adv_tick = _safe_float(
        _safe_getattr(stats, "last_adv_tick", decode_deadlines.get(META_LAST_ADV_TICK, -1.0)),
        _safe_float(decode_deadlines.get(META_LAST_ADV_TICK, -1.0), -1.0),
    )
    missed_adv_source = _safe_int(
        _safe_getattr(stats, "missed_adv_source", decode_deadlines.get(META_MISSED_ADV_SOURCE, 0.0)),
        _safe_int(decode_deadlines.get(META_MISSED_ADV_SOURCE, 0.0), 0),
    )
    decode_deadlines[META_NEXT_ADV_TICK] = float(next_adv_tick)
    decode_deadlines[META_LAST_ADV_TICK] = float(last_adv_tick)
    decode_deadlines[META_MISSED_ADV_SOURCE] = float(missed_adv_source)

    counted = {int(k): int(v) for k, v in _stats_map(stats, "decode_tokens_counted").items()}
    decode_credit_balance = _safe_int(
        _safe_getattr(stats, "decode_credit_balance", counted.get(META_DECODE_CREDIT_BAL, 0)),
        _safe_int(counted.get(META_DECODE_CREDIT_BAL, 0), 0),
    )
    counted[META_DECODE_CREDIT_BAL] = int(decode_credit_balance)

    recent_launches = []
    try:
        recent_launches = list(env._v2_get_recent_launches(state))
    except Exception:
        recent_launches = list(_safe_getattr(stats, "recent_arrivals", []) or [])

    known_request_ids = (
        [int(x) for x in req_map.keys()]
        + _id_list(_safe_getattr(stats, "active_request_ids", []))
        + _id_list(_safe_getattr(stats, "completed_request_ids", []))
        + _id_list(_safe_getattr(stats, "dropped_request_ids", []))
        + _id_list(_safe_getattr(stats, "stopped_decode_request_ids", []))
        + _id_list(_safe_getattr(stats, "violated_request_ids", []))
    )
    max_request_id = max(known_request_ids + [-1])
    # Vidur snapshots store the last assigned Request id in entity_counters.
    # Native needs the next id to assign, so guard with both sources.
    counter_last_request_id = _snapshot_request_counter(state)
    next_request_id = max_request_id + 1
    if counter_last_request_id is not None:
        next_request_id = max(next_request_id, int(counter_last_request_id) + 1)

    return {
        "sim_time": sim_time,
        "decision_state_time": sim_time,
        "next_request_id": int(next_request_id),
        "requests": requests,
        "stats": {
            "requests_generated": _safe_int(_safe_getattr(stats, "requests_generated", 0), 0),
            "requests_completed": _safe_int(_safe_getattr(stats, "requests_completed", 0), 0),
            "slo_violations": _safe_int(_safe_getattr(stats, "slo_violations", 0), 0),
            "slo_lateness_sum": _safe_float(_safe_getattr(stats, "slo_lateness_sum", 0.0), 0.0),
            "recent_arrivals": list(_safe_getattr(stats, "recent_arrivals", []) or []),
            "recent_launches": recent_launches,
            "active_request_ids": _id_list(_safe_getattr(stats, "active_request_ids", [])),
            "completed_request_ids": _id_list(_safe_getattr(stats, "completed_request_ids", [])),
            "dropped_request_ids": _id_list(_safe_getattr(stats, "dropped_request_ids", [])),
            "stopped_decode_request_ids": _id_list(_safe_getattr(stats, "stopped_decode_request_ids", [])),
            "violated_request_ids": _id_list(_safe_getattr(stats, "violated_request_ids", [])),
            "prefill_lateness_finalized_ids": _id_list(
                _safe_getattr(stats, "prefill_lateness_finalized", [])
            ),
            "decode_tokens_counted_by_id": counted,
            "per_request_prefill_lateness_by_id": {
                int(k): float(v) for k, v in _stats_map(stats, "per_request_prefill_lateness").items()
            },
            "per_request_decode_lateness_by_id": {
                int(k): float(v) for k, v in _stats_map(stats, "per_request_decode_lateness").items()
            },
            "decode_next_deadline_by_id": decode_deadlines,
            "next_adv_tick": float(next_adv_tick),
            "last_adv_tick": float(last_adv_tick),
            "pending_adv_tick": _safe_bool(_safe_getattr(stats, "pending_adv_tick", False), False),
            "missed_adv_source": int(missed_adv_source),
            "decode_credit_balance": int(decode_credit_balance),
            "decode_credit_available": max(0, int(decode_credit_balance)),
        },
    }


def _flat_tensor(values: Any, shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(list(values or []), dtype=dtype).reshape(*shape)


def _native_root_inputs_to_tensors(inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    pn = int(inputs["prefill_req_n"])
    pd = int(inputs["prefill_req_d"])
    dn = int(inputs["decode_req_n"])
    dd = int(inputs["decode_req_d"])
    rn = int(inputs["req_n"])
    rd = int(inputs["req_d"])
    return {
        "prefill_req_features": _flat_tensor(inputs["prefill_req_features"], (1, pn, pd), dtype=torch.float32),
        "decode_req_features": _flat_tensor(inputs["decode_req_features"], (1, dn, dd), dtype=torch.float32),
        "global_features": _flat_tensor(inputs["global_features"], (1, -1), dtype=torch.float32),
        "prefill_req_mask": _flat_tensor(inputs["prefill_req_mask"], (1, pn), dtype=torch.bool),
        "decode_req_mask": _flat_tensor(inputs["decode_req_mask"], (1, dn), dtype=torch.bool),
        "action_mask": _flat_tensor(inputs["action_mask"], (1, -1), dtype=torch.bool),
        "req_features": _flat_tensor(inputs["req_features"], (1, rn, rd), dtype=torch.float32),
        "req_mask": _flat_tensor(inputs["req_mask"], (1, rn), dtype=torch.bool),
    }


def _json_flat(values: Any, *, bool_values: bool = False) -> str:
    if bool_values:
        return json.dumps([bool(x) for x in list(values or [])], sort_keys=True)
    return json.dumps([float(x) for x in list(values or [])], sort_keys=True)


def _native_sample_feature_row(sample: dict[str, Any]) -> dict[str, Any]:
    inputs = dict(sample.get("inputs", {}) or {})
    return {
        "root_id": int(sample["root_id"]),
        "root_player": str(sample["player"]),
        "root_depth": int(sample["root_depth"]),
        "history_hops": int(sample.get("history_hops", 0)),
        "prefill_req_features_json": _json_flat(inputs.get("prefill_req_features", [])),
        "decode_req_features_json": _json_flat(inputs.get("decode_req_features", [])),
        "global_features_json": _json_flat(inputs.get("global_features", [])),
        "prefill_req_mask_json": _json_flat(inputs.get("prefill_req_mask", []), bool_values=True),
        "decode_req_mask_json": _json_flat(inputs.get("decode_req_mask", []), bool_values=True),
        "req_features_json": _json_flat(inputs.get("req_features", [])),
        "req_mask_json": _json_flat(inputs.get("req_mask", []), bool_values=True),
        "action_mask_json": _json_flat(inputs.get("action_mask", []), bool_values=True),
    }


def _python_signature(env: Any, prepared_root: Any) -> dict[str, Any]:
    desc = env.describe_state(prepared_root.root_state)
    total_lateness = _safe_float(desc.get("total_lateness", desc.get("slo_lateness_sum", 0.0)), 0.0)
    violations = _safe_int(desc.get("slo_violations", 0), 0)
    return {
        "root_id": int(prepared_root.root_id),
        "root_player": str(prepared_root.root_player),
        "root_depth": int(prepared_root.root_depth),
        "history_hops": int(prepared_root.history_hops),
        "sim_time": round(_safe_float(desc.get("sim_time", 0.0), 0.0), 9),
        "objective_cost": round(float(violations) + float(total_lateness), 9),
        "slo_violations": int(violations),
        "total_lateness": round(float(total_lateness), 9),
        "decode_credit_balance": _safe_int(desc.get("decode_credit_balance", 0), 0),
        "state_active_ids": tuple(int(x) for x in desc.get("active_request_ids", []) or []),
        "state_completed_request_ids": tuple(int(x) for x in desc.get("completed_request_ids", []) or []),
    }


def _native_signature(row: dict[str, str]) -> dict[str, Any]:
    return {
        "root_id": int(row["root_id"]),
        "root_player": str(row["root_player"]),
        "root_depth": int(row["root_depth"]),
        "history_hops": int(row["history_hops"]),
        "sim_time": round(_safe_float(row.get("sim_time", 0.0), 0.0), 9),
        "objective_cost": round(_safe_float(row.get("objective_cost", 0.0), 0.0), 9),
        "slo_violations": _safe_int(row.get("slo_violations", 0), 0),
        "total_lateness": round(_safe_float(row.get("total_lateness", 0.0), 0.0), 9),
        "decode_credit_balance": _safe_int(row.get("decode_credit_balance", 0), 0),
        "state_active_ids": tuple(_json_int_list(row.get("state_active_ids", "[]"))),
        "state_completed_request_ids": tuple(_json_int_list(row.get("state_completed_request_ids", "[]"))),
    }


def _make_cfg(args: argparse.Namespace, *, environment_lang: str) -> Any:
    cfg = replace(
        DEFAULT_MULTIPROCESS_TRAINING_CONFIG,
        environment_lang=str(environment_lang),
        use_virtual_env=True,
        model=replace(DEFAULT_MULTIPROCESS_TRAINING_CONFIG.model, device=str(args.model_device)),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
    )
    cfg.validate()
    return cfg


def _make_runner(cfg: Any, *, env: Any, explore_cfg: Any, model: Any, history_seed: int, out_dir: Path) -> SelfPlayRunner:
    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        rng=random.Random(int(history_seed)),
        log_path=str(out_dir / "python_mcts_iter.csv"),
        tree_log_path=str(out_dir / "python_mcts_root.csv"),
        logger_flush_every=1,
        verbose=False,
        complete_log=False,
    )
    return SelfPlayRunner(
        env=env,
        mcts=mcts,
        model=model,
        writer=_NoopWriter(),
        eval_writer=None,
        device_for_features=torch.device(str(cfg.model.device)),
        game_v2_cfg=cfg.game_v2,
    )


def _generate_python_roots(args: argparse.Namespace, cfg: Any, out_dir: Path) -> tuple[Any, Any, Any, list[Any]]:
    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )
    simulator, env, _constraints, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=True)
    runner = _make_runner(cfg, env=env, explore_cfg=explore_cfg, model=None, history_seed=int(args.history_seed), out_dir=out_dir)
    roots: list[Any] = []
    for batch in runner._iter_prepared_history_root_batches(
        game_id=0,
        num_roots=int(args.num_roots),
        start_root_id=0,
        start_root_depth=0,
        start_player="adversary",
        initial_state=None,
        history_nontrivial_hops=int(args.history_hops_min),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        history_max_total_steps=int(cfg.history_max_total_steps),
        log_history_rows=False,
        root_batch_size=int(cfg.history_root_batch_size),
        allow_duplicate_history_fallback=True,
    ):
        roots.extend(batch)
    return simulator, env, explore_cfg, roots


def _run_native_logger(
    *,
    native: Any,
    simulator: Any,
    args: argparse.Namespace,
    cfg: Any,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    trace_csv = out_dir / "history_mcts_iter.csv"
    frontier_csv = out_dir / "frontiers.csv"
    depth1_search_csv = out_dir / "depth1_search.csv"
    depth1_details_csv = out_dir / "depth1_details.csv"

    payload = _cfg_payload(cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload.update(
        {
            "native_history_trace_log_path": str(trace_csv),
            "native_frontier_log_path": str(frontier_csv),
            "native_depth1_search_log_path": str(depth1_search_csv),
            "native_depth1_details_log_path": str(depth1_details_csv),
            "use_model_bootstrap": False,
        }
    )

    runtime = native.NativeTorchScriptInferRuntimeGV2("cpu")
    native_out = native.generate_selfplay_samples_torchscript(
        runtime,
        0,
        _empty_initial_state_payload(),
        payload,
        0,
        int(args.num_roots),
        0,
        0,
        "adversary",
        int(cfg.run.feature_version),
        int(cfg.adv_iterations_per_root),
        int(cfg.cont_iterations_per_root),
        int(args.history_hops_min),
        int(args.history_hops_max),
        int(args.history_seed),
        int(cfg.history_max_total_steps),
        int(cfg.max_forced_hops_per_root),
        0.0,
        int(cfg.eval_split_seed_base),
        int(cfg.action_seed_base),
        True,
    )

    trace_count, adv_actions_checked = _validate_trace_csv(trace_csv)
    search_rows = _read_csv(depth1_search_csv)
    failed = [row.get("root_id", "") for row in search_rows if row.get("selection_passed") != "true"]
    if failed:
        raise RuntimeError(f"native depth1 selection failed for root_id(s): {failed[:20]}")

    frontier_rows = _read_csv(frontier_csv)
    detail_rows = _read_csv(depth1_details_csv)
    sample_feature_rows = [
        _native_sample_feature_row(dict(sample))
        for sample in list(native_out.get("samples", []) or [])
    ]
    stats = dict(native_out.get("stats", {}) or {})
    stats.update(
        {
            "trace_chains": trace_count,
            "adv_actions_checked": adv_actions_checked,
            "frontier_rows": len(frontier_rows),
            "depth1_search_rows": len(search_rows),
            "depth1_detail_rows": len(detail_rows),
            "sample_feature_rows": len(sample_feature_rows),
        }
    )
    return stats, frontier_rows, search_rows, detail_rows, sample_feature_rows


def _run_frontier_parity(
    *,
    env: Any,
    python_roots: list[Any],
    native_frontiers: list[dict[str, str]],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    n = min(int(args.frontier_parity_roots), len(python_roots), len(native_frontiers))
    rows: list[dict[str, Any]] = []
    failures: list[int] = []
    for idx in range(n):
        py = _python_signature(env, python_roots[idx])
        nt = _native_signature(native_frontiers[idx])
        keys = [
            "root_player",
            "root_depth",
            "history_hops",
            "sim_time",
            "objective_cost",
            "slo_violations",
            "total_lateness",
            "decode_credit_balance",
            "state_active_ids",
            "state_completed_request_ids",
        ]
        passed = all(py[key] == nt[key] for key in keys)
        if not passed:
            failures.append(int(idx))
        rows.append(
            {
                "root_id": int(idx),
                "passed": passed,
                "python_signature": json.dumps(py, sort_keys=True),
                "native_signature": json.dumps(nt, sort_keys=True),
            }
        )
    _write_csv(out_dir / "native_frontier_parity.csv", ["root_id", "passed", "python_signature", "native_signature"], rows)
    if failures:
        raise RuntimeError(
            f"native/python frontier parity failed for first {n} roots; failed rows={failures[:20]}"
        )
    return {"frontier_parity_roots": n}


def _run_feature_parity(
    *,
    native: Any,
    env: Any,
    cfg: Any,
    python_roots: list[Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    payload = _cfg_payload(cfg, torchscript_model_spec="")
    payload["use_model_bootstrap"] = False
    runtime = native.NativeTorchScriptInferRuntimeGV2("cpu")
    action_mask_fn = _make_action_mask_fn(env)
    tol = float(args.feature_tolerance)
    rows: list[dict[str, Any]] = []
    failures: list[int] = []

    for pr in python_roots:
        state = pr.root_state
        player = str(pr.root_player)
        native_out = native.search_mcts_dnn_torchscript(
            runtime,
            0,
            _native_state_payload(env, state),
            payload,
            1,
            player,
            int(pr.root_node_id_override or pr.root_id),
            int(pr.root_depth),
            0,
            int(pr.root_id),
            int(args.history_seed) + int(pr.root_id),
            False,
            False,
            "",
            "",
        )
        nt = _native_root_inputs_to_tensors(dict(native_out["root_inputs"]))
        py = build_model_inputs(
            state,
            player,
            torch.device("cpu"),
            build_action_mask_flag=True,
            action_mask_fn=action_mask_fn,
        )
        diffs = {
            "prefill_diff": _max_abs_diff(nt["prefill_req_features"], py.prefill_req_features),
            "decode_diff": _max_abs_diff(nt["decode_req_features"], py.decode_req_features),
            "global_diff": _max_abs_diff(nt["global_features"], py.global_features),
            "req_diff": _max_abs_diff(nt["req_features"], py.req_features),
        }
        mask_pass = all(
            bool(torch.equal(nt[key], getattr(py, key)))
            for key in ("prefill_req_mask", "decode_req_mask", "action_mask", "req_mask")
        )
        max_diff = max(float(v) for v in diffs.values())
        passed = bool(mask_pass and max_diff <= tol)
        if not passed:
            failures.append(int(pr.root_id))
        rows.append(
            {
                "root_id": int(pr.root_id),
                "root_player": player,
                "root_depth": int(pr.root_depth),
                "history_hops": int(pr.history_hops),
                "passed": passed,
                "mask_passed": mask_pass,
                "max_feature_diff": max_diff,
                **diffs,
            }
        )

    fields = [
        "root_id",
        "root_player",
        "root_depth",
        "history_hops",
        "passed",
        "mask_passed",
        "max_feature_diff",
        "prefill_diff",
        "decode_diff",
        "global_diff",
        "req_diff",
    ]
    _write_csv(out_dir / "native_feature_parity.csv", fields, rows)
    if failures:
        raise RuntimeError(f"native feature parity failed for root_id(s): {failures[:20]}")
    return {
        "feature_roots": len(rows),
        "max_feature_diff": max((float(r["max_feature_diff"]) for r in rows), default=0.0),
    }


def _run_native_trace_feature_check(
    *,
    cfg: Any,
    native_frontiers: list[dict[str, str]],
    native_feature_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    feature_csv = out_dir / "native_frontier_feature.csv"
    compare_csv = out_dir / "native_frontier_feature_compare.csv"
    write_csv(feature_csv, FEATURE_FIELDS, native_feature_rows)
    compare_rows, failures = compare_feature_rows(
        frontier_rows=native_frontiers,
        feature_rows=read_csv(feature_csv),
        cfg=cfg.game_v2,
        tolerance=float(args.feature_tolerance),
    )
    write_csv(compare_csv, COMPARE_FIELDS, compare_rows)
    if failures:
        raise RuntimeError(
            f"native trace-derived feature check failed for root_id(s): {failures[:20]} "
            f"(wrote {compare_csv})"
        )
    return {
        "feature_roots": len(compare_rows),
        "max_feature_diff": max((float(row["max_feature_diff"]) for row in compare_rows), default=0.0),
        "feature_csv": feature_csv,
        "feature_compare_csv": compare_csv,
    }


def _load_python_model(cfg: Any, checkpoint_path: Path) -> AlphaZeroModel:
    spec = make_dnn_spec(cfg=cfg.game_v2)
    model = AlphaZeroModel(spec=spec).to(torch.device(str(cfg.model.device)))
    _load_weights_into_model(model, checkpoint_path)
    model.eval()
    return model


def _run_bootstrap_bellman_parity(
    *,
    native: Any,
    simulator: Any,
    env: Any,
    explore_cfg: Any,
    cfg: Any,
    python_roots: list[Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found for bootstrap Bellman parity: {checkpoint_path}")

    model = _load_python_model(cfg, checkpoint_path)
    torchscript_spec = export_torchscript_pair(
        model=model,
        model_version=int(args.model_version),
        weights_path=checkpoint_path,
        out_dir=out_dir / "torchscript",
    )
    payload = _cfg_payload(cfg, torchscript_model_spec=torchscript_spec)
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = True

    runtime = native.NativeTorchScriptInferRuntimeGV2(str(args.model_device))
    runtime.load_models({int(args.model_version): torchscript_spec})

    runner = _make_runner(
        cfg,
        env=env,
        explore_cfg=explore_cfg,
        model=model,
        history_seed=int(args.history_seed),
        out_dir=out_dir,
    )
    rows: list[dict[str, Any]] = []
    failures: list[int] = []
    max_q_diff = 0.0
    max_controller_q_diff = 0.0
    max_adversary_q_diff = 0.0

    for pr in python_roots:
        root_state = pr.root_state
        root_player = str(pr.root_player)
        root_depth = int(pr.root_depth)
        root_state, root_player, root_depth = runner._advance_to_branching_root(
            root_state,
            root_player,
            root_depth,
            max_hops=int(cfg.max_forced_hops_per_root),
        )
        if root_player == "adversary":
            root_state, _ = runner._build_root_decision_state_for_adversary(
                current_state=root_state,
                pre_controller_snapshot=pr.pre_controller_snapshot,
                pre_controller_stats=pr.pre_controller_stats,
            )

        py_result = runner.run_single_root(
            SingleRootRun(
                game_id=0,
                root_id=int(pr.root_id),
                root_depth=int(root_depth),
                root_player=str(root_player),
                feature_version=int(cfg.run.feature_version),
                root_node_id_override=pr.root_node_id_override,
                model_version=int(args.model_version),
                use_model_bootstrap=True,
            ),
            root_state=root_state,
        )
        py_details, expected = _details_for_root(
            mcts=runner.mcts,
            model=model,
            root_player=str(root_player),
            root_id=int(pr.root_id),
            root_depth=int(root_depth),
            history_hops=int(pr.history_hops),
            model_version=int(args.model_version),
            selected_index=int(py_result.best_idx) if int(py_result.best_idx) >= 0 else None,
        )
        py_q_by_idx = {
            int(row["action_index"]): float(row["action_q_value"])
            for row in py_details
            if bool(row["valid"]) and row["action_q_value"] is not None
        }
        py_component_by_idx = {
            int(row["action_index"]): {
                "reward": float(row["action_reward_cost"]),
                "discount": float(row["action_discount"]),
                "bootstrap": float(row["action_bootstrap_value"]),
                "discounted_bootstrap": float(row["action_discount"]) * float(row["action_bootstrap_value"]),
            }
            for row in py_details
            if (
                bool(row["valid"])
                and row["action_reward_cost"] is not None
                and row["action_discount"] is not None
                and row["action_bootstrap_value"] is not None
            )
        }

        native_out = native.search_mcts_dnn_torchscript(
            runtime,
            int(args.model_version),
            _native_state_payload(env, root_state),
            payload,
            1,
            root_player,
            int(pr.root_node_id_override or pr.root_id),
            int(root_depth),
            0,
            int(pr.root_id),
            int(args.history_seed) + int(pr.root_id),
            False,
            False,
            "",
            "",
        )
        native_values = [float(v) for v in list(native_out.get("root_action_values", []))]
        native_mask = [bool(v) for v in list(native_out.get("root_nn_valid_mask", []))]
        native_rewards = [float(v) for v in list(native_out.get("root_action_rewards", []))]
        native_discounts = [float(v) for v in list(native_out.get("root_action_discounts", []))]
        native_bootstraps = [float(v) for v in list(native_out.get("root_action_bootstraps", []))]
        def _native_component(idx: int) -> dict[str, float | None]:
            if idx < 0:
                return {"reward": None, "discounted_bootstrap": None}
            if idx >= len(native_rewards) or idx >= len(native_discounts) or idx >= len(native_bootstraps):
                return {"reward": None, "discounted_bootstrap": None}
            return {
                "reward": native_rewards[idx],
                "discounted_bootstrap": native_discounts[idx] * native_bootstraps[idx],
            }

        diffs = [
            abs(float(py_q_by_idx[idx]) - float(native_values[idx]))
            for idx in py_q_by_idx
            if 0 <= idx < len(native_values) and idx < len(native_mask) and bool(native_mask[idx])
        ]
        q_diff = max(diffs) if diffs else 0.0
        max_q_diff = max(max_q_diff, q_diff)
        if root_player == "controller":
            max_controller_q_diff = max(max_controller_q_diff, q_diff)
        else:
            max_adversary_q_diff = max(max_adversary_q_diff, q_diff)

        py_expected_idx = -1 if expected is None else int(expected["action_index"])
        native_best_idx = int(native_out.get("best_action_index", -1))
        py_best_idx = int(py_result.best_idx)
        py_best_component = py_component_by_idx.get(py_best_idx, {})
        native_best_component = _native_component(native_best_idx)
        native_at_python_best_component = _native_component(py_best_idx)
        py_at_native_best_component = py_component_by_idx.get(native_best_idx, {})
        best_pass = bool(native_best_idx == py_expected_idx == int(py_result.best_idx))
        q_pass = bool(q_diff <= float(args.bellman_q_tolerance))
        enforce_q = bool(root_player == "controller" or args.strict_adversary_q_parity)
        passed = bool(best_pass and (q_pass or not enforce_q))
        if not passed:
            failures.append(int(pr.root_id))
        rows.append(
            {
                "root_id": int(pr.root_id),
                "root_player": str(root_player),
                "root_depth": int(root_depth),
                "history_hops": int(pr.history_hops),
                "passed": passed,
                "best_passed": best_pass,
                "q_passed": q_pass,
                "q_enforced": enforce_q,
                "python_best_index": int(py_result.best_idx),
                "python_expected_index": py_expected_idx,
                "native_best_index": native_best_idx,
                "python_best_reward_cost": py_best_component.get("reward"),
                "python_best_discounted_bootstrap": py_best_component.get("discounted_bootstrap"),
                "native_best_reward_cost": native_best_component.get("reward"),
                "native_best_discounted_bootstrap": native_best_component.get("discounted_bootstrap"),
                "native_at_python_best_reward_cost": native_at_python_best_component.get("reward"),
                "native_at_python_best_discounted_bootstrap": native_at_python_best_component.get("discounted_bootstrap"),
                "python_at_native_best_reward_cost": py_at_native_best_component.get("reward"),
                "python_at_native_best_discounted_bootstrap": py_at_native_best_component.get("discounted_bootstrap"),
                "max_q_abs_diff": q_diff,
                "valid_action_count": len(py_q_by_idx),
            }
        )

        if hasattr(runner.mcts, "clear_search_state"):
            runner.mcts.clear_search_state(drop_scratch=False)

    fields = [
        "root_id",
        "root_player",
        "root_depth",
        "history_hops",
        "passed",
        "best_passed",
        "q_passed",
        "q_enforced",
        "python_best_index",
        "python_expected_index",
        "native_best_index",
        "python_best_reward_cost",
        "python_best_discounted_bootstrap",
        "native_best_reward_cost",
        "native_best_discounted_bootstrap",
        "native_at_python_best_reward_cost",
        "native_at_python_best_discounted_bootstrap",
        "python_at_native_best_reward_cost",
        "python_at_native_best_discounted_bootstrap",
        "max_q_abs_diff",
        "valid_action_count",
    ]
    _write_csv(out_dir / "native_bellman_bootstrap_parity.csv", fields, rows)
    if failures:
        raise RuntimeError(f"native bootstrap Bellman parity failed for root_id(s): {failures[:20]}")
    return {
        "bellman_roots": len(rows),
        "max_q_diff": max_q_diff,
        "max_controller_q_diff": max_controller_q_diff,
        "max_adversary_q_diff": max_adversary_q_diff,
    }


def main() -> None:
    import vidur.mcts.mcts_native_gv2 as native

    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_python = _make_cfg(args, environment_lang="python")
    py_simulator, py_env, py_explore_cfg, py_roots = _generate_python_roots(args, cfg_python, out_dir)

    cfg_native = _make_cfg(args, environment_lang="native")
    native_stats, frontier_rows, search_rows, detail_rows, native_feature_rows = _run_native_logger(
        native=native,
        simulator=py_simulator,
        args=args,
        cfg=cfg_native,
        out_dir=out_dir,
    )

    frontier_stats = _run_frontier_parity(
        env=py_env,
        python_roots=py_roots,
        native_frontiers=frontier_rows,
        args=args,
        out_dir=out_dir,
    )
    feature_stats = _run_native_trace_feature_check(
        cfg=cfg_python,
        native_frontiers=frontier_rows,
        native_feature_rows=native_feature_rows,
        args=args,
        out_dir=out_dir,
    )
    bellman_stats: dict[str, Any] = {"bellman_roots": 0, "skipped": True}
    if not bool(args.skip_bootstrap_bellman):
        bellman_stats = _run_bootstrap_bellman_parity(
            native=native,
            simulator=py_simulator,
            env=py_env,
            explore_cfg=py_explore_cfg,
            cfg=cfg_python,
            python_roots=py_roots,
            args=args,
            out_dir=out_dir,
        )

    print(
        "native GV3 tests passed: "
        f"native_roots={native_stats.get('num_roots_generated', len(search_rows))}, "
        f"trace_chains={native_stats['trace_chains']}, "
        f"adv_actions_checked={native_stats['adv_actions_checked']}, "
        f"frontiers={len(frontier_rows)}, "
        f"depth1_roots={len(search_rows)}, "
        f"depth1_details={len(detail_rows)}, "
        f"frontier_parity_roots={frontier_stats['frontier_parity_roots']}, "
        f"feature_roots={feature_stats['feature_roots']}, "
        f"max_feature_diff={feature_stats['max_feature_diff']:.3g}, "
        f"bellman_roots={bellman_stats.get('bellman_roots', 0)}, "
        f"max_controller_q_diff={bellman_stats.get('max_controller_q_diff', 0.0):.3g}, "
        f"max_adversary_q_diff={bellman_stats.get('max_adversary_q_diff', 0.0):.3g}, "
        f"output_dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
