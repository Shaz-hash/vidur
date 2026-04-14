from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from vidur.mcts.Game_Versions.Game_Version2.config import GameVersion2Config


_GV2_CFG = GameVersion2Config()
_EPS = float(_GV2_CFG.timing.eps)

_ALLOWED_PREFILL_TOKENS: Set[int] = set(int(x) for x in _GV2_CFG.request.allowed_prefill_tokens)
_MAX_PREFILL_PER_REQ = int(_GV2_CFG.request.max_prefill_tokens_per_request)
_MIN_DECODE_PER_REQ = int(_GV2_CFG.request.min_decode_tokens_per_request)
_MAX_DECODE_PER_REQ = int(_GV2_CFG.request.max_decode_tokens_per_request)

_MAX_LAUNCH_PER_TICK = int(_GV2_CFG.adversary_action.max_launch_count_per_tick)
_LAUNCH_WIN_SEC = float(_GV2_CFG.timing.launch_window_sec)
_MAX_REQS_PER_WIN = int(_GV2_CFG.timing.max_requests_per_launch_window)

# Matches PlayerActionSampler._prefill_window_cap_tokens
_PREFILL_WIN_CAP_TOKENS = int(_GV2_CFG.request.target_prefill_tokens_per_request_avg_window) * int(
    _GV2_CFG.timing.max_requests_per_launch_window
)

_ALLOWED_WINNERS = {"candidate", "best", "tie"}


class TestFailure(AssertionError):
    pass


@dataclass
class ParsedAdversaryAction:
    requests: List[Dict[str, Any]]
    stop_decode_ids: List[int]


@dataclass
class ParsedControllerAction:
    prefill_allocations: Dict[int, int]


@dataclass
class AdvEvent:
    rownum: int
    game_id: int
    sim_time: float
    launch_count: int
    prefill_total: int
    context: str


@dataclass
class PrefillTracker:
    # None means unknown initial prefill remaining (cannot assert yet).
    rem_by_id: Dict[int, Optional[int]]
    pending_launch_prefills: deque[int]
    prev_active_ids: Set[int]


@dataclass
class EvalCycleTracker:
    rem_prefill_by_id: Dict[int, int]
    prev_active_ids: Set[int]
    prev_decode_tokens_by_id: Dict[int, int]
    initialized: bool


def _fail(test_name: str, file_path: str, rownum: int, msg: str) -> None:
    raise TestFailure(f"[{test_name}] file={file_path} row={rownum} ERROR: {msg}")


def _safe_int(x: object, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _json_loads_maybe(s: str):
    s = (s or "").strip()
    if s in ("", "null", "None"):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_int_list(cell: str) -> List[int]:
    obj = _json_loads_maybe(cell)
    if not isinstance(obj, list):
        return []
    out: List[int] = []
    for x in obj:
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


def _parse_int_map(cell: str) -> Dict[int, int]:
    obj = _json_loads_maybe(cell)
    if not isinstance(obj, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in obj.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            pass
    return out


def _parse_float_list(cell: str) -> List[float]:
    obj = _json_loads_maybe(cell)
    if not isinstance(obj, list):
        return []
    out: List[float] = []
    for x in obj:
        try:
            out.append(float(x))
        except Exception:
            out.append(float("nan"))
    return out


def _extract_int_field(text: str, name: str) -> Optional[int]:
    m = re.search(rf"{re.escape(name)}\s*=\s*(-?\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_float_field(text: str, name: str) -> Optional[float]:
    m = re.search(
        rf"{re.escape(name)}\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_stop_decode_ids_from_repr(action_repr: str) -> List[int]:
    m = re.search(r"stop_decode_ids=\[([^\]]*)\]", action_repr or "")
    if not m:
        return []
    s = m.group(1).strip()
    if not s:
        return []
    out: List[int] = []
    for p in s.split(","):
        try:
            out.append(int(p.strip()))
        except Exception:
            pass
    return out


def _parse_adversary_action(best_action_json: str, action_repr: str) -> Optional[ParsedAdversaryAction]:
    obj = _json_loads_maybe(best_action_json)
    if isinstance(obj, dict) and str(obj.get("type", "")).strip().lower() == "adversary":
        reqs_raw = obj.get("requests", [])
        stop_raw = obj.get("stop_decode_ids", [])
        reqs: List[Dict[str, Any]] = []
        if isinstance(reqs_raw, list):
            for r in reqs_raw:
                if not isinstance(r, dict):
                    continue
                reqs.append(
                    {
                        "prefill_tokens": _safe_int(r.get("prefill_tokens"), -1),
                        "decode_tokens": _safe_int(r.get("decode_tokens"), -1),
                        "prefill_slo": _safe_float(r.get("prefill_slo"), float("nan")),
                        "decode_slo": _safe_float(r.get("decode_slo"), float("nan")),
                    }
                )
        stop_ids: List[int] = []
        if isinstance(stop_raw, list):
            for x in stop_raw:
                try:
                    stop_ids.append(int(x))
                except Exception:
                    pass
        return ParsedAdversaryAction(requests=reqs, stop_decode_ids=stop_ids)

    if "AdversaryAction" not in (action_repr or ""):
        return None

    reqs: List[Dict[str, Any]] = []
    for m in re.finditer(r"AdversaryRequestSpec\((.*?)\)", action_repr or ""):
        body = m.group(1)
        pf = _extract_int_field(body, "prefill_tokens")
        dd = _extract_int_field(body, "decode_tokens")
        ps = _extract_float_field(body, "prefill_slo")
        ds = _extract_float_field(body, "decode_slo")
        if pf is None or dd is None:
            continue
        reqs.append(
            {
                "prefill_tokens": int(pf),
                "decode_tokens": int(dd),
                "prefill_slo": float(ps if ps is not None else float("nan")),
                "decode_slo": float(ds if ds is not None else float("nan")),
            }
        )
    stop_ids = _parse_stop_decode_ids_from_repr(action_repr or "")
    return ParsedAdversaryAction(requests=reqs, stop_decode_ids=stop_ids)


def _parse_controller_action(best_action_json: str, action_repr: str) -> Optional[ParsedControllerAction]:
    obj = _json_loads_maybe(best_action_json)
    if isinstance(obj, dict) and str(obj.get("type", "")).strip().lower() == "controller":
        raw = obj.get("prefill_allocations", {})
        out: Dict[int, int] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    rid = int(k)
                    tok = int(v)
                    if rid >= 0 and tok >= 0:
                        out[rid] = tok
                except Exception:
                    pass
        return ParsedControllerAction(prefill_allocations=out)

    if "ControllerAction" not in (action_repr or ""):
        return None

    m = re.search(r"prefill_allocations=\{([^}]*)\}", action_repr or "")
    if not m:
        return ParsedControllerAction(prefill_allocations={})

    out: Dict[int, int] = {}
    body = m.group(1).strip()
    if body:
        for kv in body.split(","):
            if ":" not in kv:
                continue
            a, b = kv.split(":", 1)
            try:
                rid = int(a.strip())
                tok = int(b.strip())
                if rid >= 0 and tok >= 0:
                    out[rid] = tok
            except Exception:
                pass
    return ParsedControllerAction(prefill_allocations=out)


def _check_adversary_action_constraints(
    *,
    action: ParsedAdversaryAction,
    test_name: str,
    file_path: str,
    rownum: int,
) -> Tuple[int, int]:
    launch_count = len(action.requests)
    prefill_total = 0

    if launch_count < 0 or launch_count > _MAX_LAUNCH_PER_TICK:
        _fail(
            test_name,
            file_path,
            rownum,
            f"launch_count out of bounds: got={launch_count}, max={_MAX_LAUNCH_PER_TICK}",
        )

    for i, req in enumerate(action.requests):
        pf = _safe_int(req.get("prefill_tokens"), -1)
        dd = _safe_int(req.get("decode_tokens"), -1)
        ps = _safe_float(req.get("prefill_slo"), float("nan"))
        ds = _safe_float(req.get("decode_slo"), float("nan"))

        if pf not in _ALLOWED_PREFILL_TOKENS:
            _fail(
                test_name,
                file_path,
                rownum,
                f"request[{i}] invalid prefill_tokens={pf}, allowed={sorted(_ALLOWED_PREFILL_TOKENS)}",
            )
        if pf <= 0 or pf > _MAX_PREFILL_PER_REQ:
            _fail(test_name, file_path, rownum, f"request[{i}] prefill_tokens out of bounds: {pf}")

        if dd < _MIN_DECODE_PER_REQ or dd > _MAX_DECODE_PER_REQ:
            _fail(
                test_name,
                file_path,
                rownum,
                f"request[{i}] decode_tokens out of bounds: got={dd}, expected in [{_MIN_DECODE_PER_REQ}, {_MAX_DECODE_PER_REQ}]",
            )

        if not math.isfinite(ps) or ps <= 0.0:
            _fail(test_name, file_path, rownum, f"request[{i}] invalid prefill_slo={ps}")
        if not math.isfinite(ds) or ds <= 0.0:
            _fail(test_name, file_path, rownum, f"request[{i}] invalid decode_slo={ds}")

        prefill_total += pf

    stop_ids = [int(x) for x in action.stop_decode_ids]
    if len(set(stop_ids)) != len(stop_ids):
        _fail(test_name, file_path, rownum, f"duplicate stop_decode_ids found: {stop_ids}")
    if any(x < 0 for x in stop_ids):
        _fail(test_name, file_path, rownum, f"negative stop_decode_ids are invalid: {stop_ids}")

    return launch_count, prefill_total


def _check_sliding_window_constraints(
    *,
    events_by_game: Dict[int, List[AdvEvent]],
    test_name: str,
    file_path: str,
) -> None:
    for game_id, events in events_by_game.items():
        events.sort(key=lambda e: (e.sim_time, e.rownum))
        q: deque[Tuple[float, int, int]] = deque()
        inwin_count = 0
        inwin_prefill = 0
        last_t: Optional[float] = None

        for ev in events:
            t = float(ev.sim_time)

            # If a stream jumps backwards in time, treat it as a new segment.
            if last_t is not None and (t + _EPS) < last_t:
                q.clear()
                inwin_count = 0
                inwin_prefill = 0
            last_t = t

            lo = t - _LAUNCH_WIN_SEC
            while q and (q[0][0] + _EPS) < lo:
                _, c0, p0 = q.popleft()
                inwin_count -= c0
                inwin_prefill -= p0

            next_count = inwin_count + ev.launch_count
            next_prefill = inwin_prefill + ev.prefill_total

            if next_count > _MAX_REQS_PER_WIN:
                _fail(
                    test_name,
                    file_path,
                    ev.rownum,
                    f"window request-count overflow: game={game_id}, context={ev.context}, "
                    f"inwin_before={inwin_count}, this={ev.launch_count}, cap={_MAX_REQS_PER_WIN}",
                )
            if next_prefill > _PREFILL_WIN_CAP_TOKENS:
                _fail(
                    test_name,
                    file_path,
                    ev.rownum,
                    f"window prefill overflow: game={game_id}, context={ev.context}, "
                    f"inwin_before={inwin_prefill}, this={ev.prefill_total}, cap={_PREFILL_WIN_CAP_TOKENS}",
                )

            q.append((t, ev.launch_count, ev.prefill_total))
            inwin_count = next_count
            inwin_prefill = next_prefill


def _check_root_file(path: str) -> int:
    required = {
        "game_id",
        "root_id",
        "root_player",
        "phase",
        "sim_time",
        "best_action_index",
        "best_action_repr",
        "best_action_json",
        "best_action_mcts_prob",
        "best_action_model_prob",
        "mcts_root_prior_json",
        "normalized_root_prior_json",
        "valid_action_mask_json",
    }

    rows_checked = 0
    adv_events_by_game: Dict[int, List[AdvEvent]] = defaultdict(list)
    prev_completed_by_game: Dict[int, Set[int]] = defaultdict(set)
    prefill_tracker_by_game: Dict[int, PrefillTracker] = defaultdict(
        lambda: PrefillTracker(rem_by_id={}, pending_launch_prefills=deque(), prev_active_ids=set())
    )

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        missing = sorted(required - cols)
        if missing:
            _fail("test_root_required_columns", path, 1, f"missing columns: {missing}")

        for rownum, row in enumerate(r, start=2):
            rows_checked += 1
            game_id = _safe_int(row.get("game_id"), 0)
            phase = (row.get("phase") or "").strip().lower()
            root_player = (row.get("root_player") or "").strip().lower()
            sim_time = _safe_float(row.get("sim_time"), 0.0)
            best_action_json = row.get("best_action_json") or ""
            best_action_repr = row.get("best_action_repr") or ""

            active_ids_now = set(_parse_int_list(row.get("state_active_ids") or "[]"))
            completed_ids_now = set(_parse_int_list(row.get("state_completed_request_ids") or "[]"))
            decode_map_now = _parse_int_map(row.get("state_decode_tokens_counted_by_id") or "{}")

            tracker = prefill_tracker_by_game[game_id]
            action_applied_in_row = bool(phase.endswith("applied"))
            parsed_ctrl = _parse_controller_action(best_action_json, best_action_repr) if action_applied_in_row else None
            parsed_adv = _parse_adversary_action(best_action_json, best_action_repr) if action_applied_in_row else None

            active_list_now = sorted(int(x) for x in active_ids_now)
            completed_list_now = sorted(int(x) for x in completed_ids_now)
            if len(active_list_now) != len(set(active_list_now)):
                _fail("test_root_active_ids_unique", path, rownum, f"duplicate active ids: {active_list_now}")
            if len(completed_list_now) != len(set(completed_list_now)):
                _fail("test_root_completed_ids_unique", path, rownum, f"duplicate completed ids: {completed_list_now}")
            overlap = active_ids_now & completed_ids_now
            if overlap:
                _fail(
                    "test_root_active_completed_disjoint",
                    path,
                    rownum,
                    f"ids present in both active and completed: {sorted(overlap)}",
                )
            for rid, tok in decode_map_now.items():
                if tok < 0:
                    _fail(
                        "test_root_decode_tokens_nonnegative",
                        path,
                        rownum,
                        f"state_decode_tokens_counted_by_id[{rid}]={tok} negative",
                    )
                if int(rid) not in active_ids_now:
                    _fail(
                        "test_root_decode_map_subset_active",
                        path,
                        rownum,
                        f"decode map id {rid} not present in active ids",
                    )

            if parsed_adv is not None:
                for req in parsed_adv.requests:
                    pf = _safe_int(req.get("prefill_tokens"), -1)
                    if pf > 0:
                        tracker.pending_launch_prefills.append(int(pf))

            # Bind newly appeared active IDs to pending adversary launches.
            new_ids = sorted(active_ids_now - tracker.prev_active_ids)
            for rid in new_ids:
                if tracker.pending_launch_prefills:
                    tracker.rem_by_id[int(rid)] = max(0, int(tracker.pending_launch_prefills.popleft()))
                else:
                    tracker.rem_by_id[int(rid)] = None

            if parsed_ctrl is not None:
                for rid, tok in parsed_ctrl.prefill_allocations.items():
                    rid_i = int(rid)
                    tok_i = int(tok)
                    if tok_i < 0:
                        _fail(
                            "test_root_prefill_alloc_nonnegative",
                            path,
                            rownum,
                            f"negative prefill allocation: rid={rid_i}, tok={tok_i}",
                        )
                    if rid_i not in active_ids_now:
                        _fail(
                            "test_root_prefill_alloc_subset_active",
                            path,
                            rownum,
                            f"prefill allocation rid={rid_i} not present in active ids",
                        )
                    cur = tracker.rem_by_id.get(rid_i, None)
                    if cur is None:
                        continue
                    if tok_i > int(cur):
                        _fail(
                            "test_root_prefill_alloc_not_over_remaining",
                            path,
                            rownum,
                            f"rid={rid_i} prefill allocation {tok_i} exceeds tracked remaining {cur}",
                        )
                    tracker.rem_by_id[rid_i] = max(0, int(cur) - tok_i)

            # Keep tracker only for currently active IDs.
            for rid in list(tracker.rem_by_id.keys()):
                if (rid not in active_ids_now) or (rid in completed_ids_now):
                    tracker.rem_by_id.pop(rid, None)

            # Invariant: decode may appear only after tracked prefill is complete.
            for rid in decode_map_now.keys():
                rid_i = int(rid)
                rem = tracker.rem_by_id.get(rid_i, None)
                if rem is not None and rem > 0:
                    _fail(
                        "test_root_decode_requires_prefill_complete",
                        path,
                        rownum,
                        f"rid={rid_i} appears in decode map but tracked prefill_remaining={rem} > 0",
                    )
                if rid_i in active_ids_now:
                    tracker.rem_by_id[rid_i] = 0

            best_idx = _safe_int(row.get("best_action_index"), -1)
            best_mcts_prob = _safe_float(row.get("best_action_mcts_prob"), 0.0)
            best_model_prob = _safe_float(row.get("best_action_model_prob"), 0.0)

            if not (0.0 - _EPS <= best_mcts_prob <= 1.0 + _EPS):
                _fail("test_root_prob_bounds", path, rownum, f"best_action_mcts_prob out of [0,1]: {best_mcts_prob}")
            if not (0.0 - _EPS <= best_model_prob <= 1.0 + _EPS):
                _fail("test_root_prob_bounds", path, rownum, f"best_action_model_prob out of [0,1]: {best_model_prob}")

            mcts_prior = _parse_float_list(row.get("mcts_root_prior_json") or "[]")
            norm_prior = _parse_float_list(row.get("normalized_root_prior_json") or "[]")
            valid_mask_obj = _json_loads_maybe(row.get("valid_action_mask_json") or "[]")
            valid_mask = valid_mask_obj if isinstance(valid_mask_obj, list) else []

            if mcts_prior and best_idx >= 0 and best_idx >= len(mcts_prior):
                _fail(
                    "test_root_best_action_index_range",
                    path,
                    rownum,
                    f"best_action_index={best_idx} out of range for mcts_root_prior size={len(mcts_prior)}",
                )

            if mcts_prior:
                s = sum(x for x in mcts_prior if math.isfinite(x))
                if abs(s - 1.0) > 5e-2:
                    _fail("test_root_mcts_prior_normalized", path, rownum, f"mcts_root_prior sum={s:.6f} (expected ~1)")
                if best_idx >= 0:
                    p_idx = mcts_prior[best_idx]
                    if math.isfinite(p_idx) and abs(p_idx - best_mcts_prob) > 5e-2:
                        _fail(
                            "test_root_best_action_prob_matches_prior",
                            path,
                            rownum,
                            f"best_action_mcts_prob={best_mcts_prob:.6f} differs from prior[{best_idx}]={p_idx:.6f}",
                        )

            if norm_prior and valid_mask and len(norm_prior) != len(valid_mask):
                _fail(
                    "test_root_prior_mask_shape",
                    path,
                    rownum,
                    f"normalized_root_prior length={len(norm_prior)} vs valid_action_mask length={len(valid_mask)} mismatch",
                )

            tracker.prev_active_ids = set(active_ids_now)

            # adversary constraints on applied roots
            if root_player == "adversary" and phase.endswith("applied"):
                parsed = _parse_adversary_action(
                    best_action_json,
                    best_action_repr,
                )
                if parsed is None:
                    _fail(
                        "test_root_adversary_action_parseable",
                        path,
                        rownum,
                        "could not parse adversary action from best_action_json/repr",
                    )

                launch_count, prefill_total = _check_adversary_action_constraints(
                    action=parsed,
                    test_name="test_root_adversary_action_constraints",
                    file_path=path,
                    rownum=rownum,
                )
                prev_completed = set(prev_completed_by_game.get(int(game_id), set()))
                bad_stop = sorted(int(rid) for rid in (parsed.stop_decode_ids or []) if int(rid) in prev_completed)
                if bad_stop:
                    _fail(
                        "test_root_adversary_stop_not_already_completed",
                        path,
                        rownum,
                        (
                            f"adversary stop_decode_ids include already-completed IDs from previous row: {bad_stop}; "
                            f"prev_completed={sorted(prev_completed)}"
                        ),
                    )
                adv_events_by_game[game_id].append(
                    AdvEvent(
                        rownum=rownum,
                        game_id=game_id,
                        sim_time=sim_time,
                        launch_count=launch_count,
                        prefill_total=prefill_total,
                        context="mcts_root",
                    )
                )

            prev_completed_by_game[int(game_id)] = set(int(x) for x in completed_ids_now)

    _check_sliding_window_constraints(
        events_by_game=adv_events_by_game,
        test_name="test_root_adversary_sliding_window_constraints",
        file_path=path,
    )
    return rows_checked


def _check_arena_game_file(path: str) -> int:
    required = {
        "game_id",
        "cycle_label",
        "phase",
        "turn",
        "depth",
        "player_acted",
        "player_to_act_next",
        "action_repr",
        "sim_time_before",
        "sim_time_after",
        "total_cost",
        "slo_violations",
        "total_lateness",
        "active_request_ids",
        "completed_request_ids",
        "decode_credit_balance",
        "decode_processed_tokens_by_id",
        "prefill_remaining_by_id",
        "end_reason",
    }

    rows_checked = 0
    adv_events_by_game: Dict[int, List[AdvEvent]] = defaultdict(list)
    
    prev_turn_by_stream: Dict[Tuple[str, str], int] = {}
    prev_time_by_stream: Dict[Tuple[str, str], float] = {}
    prev_completed_by_cycle: Dict[str, Set[int]] = defaultdict(set)
    cycle_tracker_by_cycle: Dict[str, EvalCycleTracker] = defaultdict(
        lambda: EvalCycleTracker(
            rem_prefill_by_id={},
            prev_active_ids=set(),
            prev_decode_tokens_by_id={},
            initialized=False,
        )
    )

    saw_end = False

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        missing = sorted(required - cols)
        if missing:
            _fail("test_eval_required_columns", path, 1, f"missing columns: {missing}")

        for rownum, row in enumerate(r, start=2):
            rows_checked += 1
            game_id = _safe_int(row.get("game_id"), 0)
            cycle = (row.get("cycle_label") or "").strip()
           
            phase = (row.get("phase") or "").strip().lower()
            player_acted = (row.get("player_acted") or "").strip().lower()

            turn_raw = (row.get("turn") or "").strip()
            sim_before_raw = (row.get("sim_time_before") or "").strip()
            sim_after_raw = (row.get("sim_time_after") or "").strip()

            has_turn = (turn_raw != "")
            has_times = (sim_before_raw != "") and (sim_after_raw != "")

            turn = int(float(turn_raw)) if has_turn else None
            sim_before = float(sim_before_raw) if has_times else None
            sim_after = float(sim_after_raw) if has_times else None

            if has_times:
                if not math.isfinite(sim_before) or sim_before < 0.0:
                    _fail("test_eval_time_values", path, rownum, f"invalid sim_time_before={sim_before}")
                if not math.isfinite(sim_after) or sim_after < 0.0:
                    _fail("test_eval_time_values", path, rownum, f"invalid sim_time_after={sim_after}")
                if sim_after + _EPS < sim_before:
                    _fail(
                        "test_eval_time_flow",
                        path,
                        rownum,
                        f"sim_time_after < sim_time_before ({sim_after} < {sim_before})",
                    )

            # Monotonic checks only for step/history rows with explicit turn/time.
            if cycle and phase != "arena_end":
                if phase.startswith("history_step:"):
                    phase_group = "history_step"
                elif phase == "arena_step":
                    phase_group = "arena_step"
                else:
                    phase_group = phase

                stream_key = (cycle, phase_group)

                if has_turn:
                    if stream_key in prev_turn_by_stream and turn < prev_turn_by_stream[stream_key]:
                        _fail(
                            "test_eval_turn_monotonic",
                            path,
                            rownum,
                            f"turn decreased in stream={stream_key}: prev={prev_turn_by_stream[stream_key]} now={turn}",
                        )
                    prev_turn_by_stream[stream_key] = turn

                if has_times:
                    if stream_key in prev_time_by_stream and sim_after + _EPS < prev_time_by_stream[stream_key]:
                        _fail(
                            "test_eval_time_monotonic_by_cycle",
                            path,
                            rownum,
                            f"sim_time_after decreased in stream={stream_key}: prev={prev_time_by_stream[stream_key]} now={sim_after}",
                        )
                    prev_time_by_stream[stream_key] = sim_after


            total_cost = _safe_float(row.get("total_cost"), 0.0)
            total_lateness = _safe_float(row.get("total_lateness"), 0.0)
            slo_viol = _safe_int(row.get("slo_violations"), 0)

            if not math.isfinite(total_cost):
                _fail("test_eval_cost_finite", path, rownum, f"invalid total_cost={total_cost}")
            if not math.isfinite(total_lateness):
                _fail("test_eval_lateness_finite", path, rownum, f"invalid total_lateness={total_lateness}")
            if slo_viol < 0:
                _fail("test_eval_slo_nonnegative", path, rownum, f"negative slo_violations={slo_viol}")

            active_raw = (row.get("active_request_ids") or "").strip()
            completed_raw = (row.get("completed_request_ids") or "").strip()
            decode_credit_raw = (row.get("decode_credit_balance") or "").strip()
            decode_raw = (row.get("decode_processed_tokens_by_id") or "").strip()
            prefill_raw = (row.get("prefill_remaining_by_id") or "").strip()

            has_state_snapshot = bool(active_raw or completed_raw or decode_raw or prefill_raw or decode_credit_raw)

            active_ids = _parse_int_list(active_raw or "[]")
            completed_ids = _parse_int_list(completed_raw or "[]")
            decode_credit_balance = _safe_int(decode_credit_raw if decode_credit_raw != "" else 0, 0)
            decode_map = _parse_int_map(decode_raw or "{}")
            prefill_map = _parse_int_map(prefill_raw or "{}")
            active_set = set(int(x) for x in active_ids)

            if len(set(active_ids)) != len(active_ids):
                _fail("test_eval_active_ids_unique", path, rownum, f"duplicate active ids: {active_ids}")
            if len(set(completed_ids)) != len(completed_ids):
                _fail("test_eval_completed_ids_unique", path, rownum, f"duplicate completed ids: {completed_ids}")

            overlap = set(active_ids) & set(completed_ids)
            if overlap:
                _fail(
                    "test_eval_active_completed_disjoint",
                    path,
                    rownum,
                    f"ids present in both active and completed: {sorted(overlap)}",
                )

            if has_state_snapshot and decode_credit_balance < 0:
                _fail(
                    "test_eval_decode_credit_nonnegative",
                    path,
                    rownum,
                    f"decode_credit_balance negative: {decode_credit_balance}",
                )

            for rid, v in decode_map.items():
                if v < 0:
                    _fail(
                        "test_eval_decode_processed_nonnegative",
                        path,
                        rownum,
                        f"decode_processed_tokens_by_id[{rid}]={v} negative",
                    )
                if rid not in set(active_ids):
                    _fail(
                        "test_eval_decode_map_subset_active",
                        path,
                        rownum,
                        f"decode map id {rid} not present in active ids",
                    )

            for rid, v in prefill_map.items():
                if v < 0:
                    _fail(
                        "test_eval_prefill_remaining_nonnegative",
                        path,
                        rownum,
                        f"prefill_remaining_by_id[{rid}]={v} negative",
                    )
                if rid not in set(active_ids):
                    _fail(
                        "test_eval_prefill_map_subset_active",
                        path,
                        rownum,
                        f"prefill map id {rid} not present in active ids",
                    )

            action_repr = row.get("action_repr") or ""
            parsed_adv_action = _parse_adversary_action("", action_repr) if "AdversaryAction" in action_repr else None
            parsed_ctrl_action = _parse_controller_action("", action_repr) if "ControllerAction" in action_repr else None

            if cycle and has_state_snapshot:
                tracker = cycle_tracker_by_cycle[cycle]

                if not tracker.initialized:
                    tracker.rem_prefill_by_id = {int(rid): int(prefill_map.get(int(rid), 0)) for rid in active_set}
                    tracker.prev_active_ids = set(active_set)
                    tracker.prev_decode_tokens_by_id = {int(rid): int(v) for rid, v in decode_map.items() if int(rid) in active_set}
                    tracker.initialized = True
                else:
                    prev_active = set(tracker.prev_active_ids)
                    new_ids = sorted(active_set - prev_active)

                    # Apply current row action effects to expected tracker state (row is post-action snapshot).
                    if player_acted == "adversary":
                        if parsed_adv_action is None:
                            _fail(
                                "test_eval_adversary_action_parseable",
                                path,
                                rownum,
                                "could not parse adversary action_repr",
                            )
                        launched = len(parsed_adv_action.requests)
                        if len(new_ids) != launched:
                            _fail(
                                "test_eval_new_ids_match_adv_launch_count",
                                path,
                                rownum,
                                f"new active ids count={len(new_ids)} does not match launched requests={launched}",
                            )
                        for rid, req in zip(new_ids, parsed_adv_action.requests):
                            pf = _safe_int(req.get("prefill_tokens"), -1)
                            tracker.rem_prefill_by_id[int(rid)] = max(0, int(pf))

                    if player_acted == "controller" and parsed_ctrl_action is not None:
                        for rid, tok in parsed_ctrl_action.prefill_allocations.items():
                            rid_i = int(rid)
                            tok_i = int(tok)
                            if rid_i not in active_set:
                                _fail(
                                    "test_eval_prefill_alloc_subset_active",
                                    path,
                                    rownum,
                                    f"prefill allocation rid={rid_i} not in active ids",
                                )
                            cur = int(tracker.rem_prefill_by_id.get(rid_i, 0))
                            if tok_i > cur:
                                _fail(
                                    "test_eval_prefill_alloc_not_over_remaining",
                                    path,
                                    rownum,
                                    f"rid={rid_i} prefill allocation {tok_i} exceeds expected remaining {cur}",
                                )
                            tracker.rem_prefill_by_id[rid_i] = max(0, cur - tok_i)

                    # Keep tracker only for currently active IDs.
                    for rid in list(tracker.rem_prefill_by_id.keys()):
                        if rid not in active_set:
                            tracker.rem_prefill_by_id.pop(rid, None)

                expected_prefill_map = {
                    int(rid): int(rem)
                    for rid, rem in tracker.rem_prefill_by_id.items()
                    if int(rid) in active_set and int(rem) > 0
                }
                if prefill_map != expected_prefill_map:
                    _fail(
                        "test_eval_prefill_map_matches_action_replay",
                        path,
                        rownum,
                        f"prefill map mismatch: expected={expected_prefill_map}, got={prefill_map}",
                    )

                # Decode may appear only once expected prefill is complete.
                for rid in decode_map.keys():
                    rid_i = int(rid)
                    rem_pref = int(tracker.rem_prefill_by_id.get(rid_i, 0))
                    if rem_pref > 0:
                        _fail(
                            "test_eval_decode_requires_prefill_complete",
                            path,
                            rownum,
                            f"rid={rid_i} has decode entry but expected prefill_remaining={rem_pref} > 0",
                        )

                # Decode processed tokens should be non-decreasing for persistent active IDs.
                prev_decode = tracker.prev_decode_tokens_by_id
                for rid, prev_tok in list(prev_decode.items()):
                    if rid not in active_set:
                        prev_decode.pop(rid, None)
                for rid, tok in decode_map.items():
                    rid_i = int(rid)
                    tok_i = int(tok)
                    prev_tok = prev_decode.get(rid_i, None)
                    if prev_tok is not None and tok_i < int(prev_tok):
                        _fail(
                            "test_eval_decode_processed_monotonic",
                            path,
                            rownum,
                            f"rid={rid_i} decode processed decreased: prev={prev_tok}, now={tok_i}",
                        )
                    prev_decode[rid_i] = tok_i

                tracker.prev_active_ids = set(active_set)

            if player_acted == "adversary" and parsed_adv_action is not None:
                launch_count, prefill_total = _check_adversary_action_constraints(
                    action=parsed_adv_action,
                    test_name="test_eval_adversary_action_constraints",
                    file_path=path,
                    rownum=rownum,
                )
                prev_completed = set(prev_completed_by_cycle.get(str(cycle), set()))
                bad_stop = sorted(
                    int(rid) for rid in (parsed_adv_action.stop_decode_ids or []) if int(rid) in prev_completed
                )
                if bad_stop:
                    _fail(
                        "test_eval_adversary_stop_not_already_completed",
                        path,
                        rownum,
                        (
                            f"adversary stop_decode_ids include already-completed IDs from previous row in cycle={cycle}: "
                            f"{bad_stop}; prev_completed={sorted(prev_completed)}"
                        ),
                    )
                adv_events_by_game[game_id].append(
                    AdvEvent(
                        rownum=rownum,
                        game_id=game_id,
                        sim_time=sim_before,
                        launch_count=launch_count,
                        prefill_total=prefill_total,
                        context=f"arena_game:{cycle}:{phase}",
                    )
                )

            if cycle and has_state_snapshot:
                prev_completed_by_cycle[str(cycle)] = set(int(x) for x in completed_ids)

            if phase == "arena_end":
                saw_end = True
                end_reason = (row.get("end_reason") or "").strip()
                if end_reason == "":
                    _fail("test_eval_end_reason_present", path, rownum, "arena_end row missing end_reason")

    if rows_checked == 0:
        _fail("test_eval_nonempty", path, 1, "arena game file is empty")
    if not saw_end:
        _fail("test_eval_has_end_row", path, rows_checked + 1, "no arena_end row found")

    _check_sliding_window_constraints(
        events_by_game=adv_events_by_game,
        test_name="test_eval_adversary_sliding_window_constraints",
        file_path=path,
    )
    return rows_checked


def _check_arena_results_file(path: str) -> int:
    required = {
        "game_id",
        "candidate_as_adv_cost",
        "best_as_adv_cost",
        "winner",
        "candidate_points",
        "best_points",
        "candidate_as_adv_file",
        "best_as_adv_file",
    }
    rows_checked = 0

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        missing = sorted(required - cols)
        if missing:
            _fail("test_arena_results_required_columns", path, 1, f"missing columns: {missing}")

        for rownum, row in enumerate(r, start=2):
            rows_checked += 1
            winner = (row.get("winner") or "").strip().lower()
            if winner not in _ALLOWED_WINNERS:
                _fail("test_arena_results_winner_domain", path, rownum, f"invalid winner={winner}")

            cp = _safe_float(row.get("candidate_points"), 0.0)
            bp = _safe_float(row.get("best_points"), 0.0)
            if not math.isfinite(cp) or not math.isfinite(bp):
                _fail("test_arena_results_points_finite", path, rownum, f"invalid points: cp={cp}, bp={bp}")
            if cp < -_EPS or bp < -_EPS:
                _fail("test_arena_results_points_nonnegative", path, rownum, f"negative points: cp={cp}, bp={bp}")
            if abs((cp + bp) - 1.0) > 1e-6:
                _fail("test_arena_results_points_sum", path, rownum, f"cp+bp must be 1.0, got {cp + bp}")

            if cp > bp + _EPS and winner != "candidate":
                _fail("test_arena_results_winner_matches_points", path, rownum, "winner mismatch: expected candidate")
            if bp > cp + _EPS and winner != "best":
                _fail("test_arena_results_winner_matches_points", path, rownum, "winner mismatch: expected best")
            if abs(cp - bp) <= _EPS and winner != "tie":
                _fail("test_arena_results_winner_matches_points", path, rownum, "winner mismatch: expected tie")

            ccost = _safe_float(row.get("candidate_as_adv_cost"), float("nan"))
            bcost = _safe_float(row.get("best_as_adv_cost"), float("nan"))
            if not math.isfinite(ccost) or ccost < 0.0:
                _fail("test_arena_results_costs", path, rownum, f"invalid candidate_as_adv_cost={ccost}")
            if not math.isfinite(bcost) or bcost < 0.0:
                _fail("test_arena_results_costs", path, rownum, f"invalid best_as_adv_cost={bcost}")

            cfile = (row.get("candidate_as_adv_file") or "").strip()
            bfile = (row.get("best_as_adv_file") or "").strip()
            if cfile and not os.path.exists(cfile):
                _fail("test_arena_results_referenced_files_exist", path, rownum, f"missing file: {cfile}")
            if bfile and not os.path.exists(bfile):
                _fail("test_arena_results_referenced_files_exist", path, rownum, f"missing file: {bfile}")

    if rows_checked == 0:
        _fail("test_arena_results_nonempty", path, 1, "arena_results.csv is empty")
    return rows_checked


def _check_eval_metrics_file(path: str) -> int:
    required = {
        "generation",
        "num_games",
        "candidate_points",
        "best_points",
        "total_points",
        "candidate_win_rate",
        "win_threshold",
        "passed",
        "promoted_to_best",
        "arena_results_csv",
        "arena_games_dir",
    }

    rows_checked = 0
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        missing = sorted(required - cols)
        if missing:
            _fail("test_eval_metrics_required_columns", path, 1, f"missing columns: {missing}")

        for rownum, row in enumerate(r, start=2):
            rows_checked += 1
            num_games = _safe_int(row.get("num_games"), 0)
            cp = _safe_float(row.get("candidate_points"), 0.0)
            bp = _safe_float(row.get("best_points"), 0.0)
            tp = _safe_float(row.get("total_points"), 0.0)
            wr = _safe_float(row.get("candidate_win_rate"), 0.0)
            th = _safe_float(row.get("win_threshold"), 0.0)
            passed = str(row.get("passed") or "").strip().lower() == "true"
            promoted = str(row.get("promoted_to_best") or "").strip().lower() == "true"

            if num_games <= 0:
                _fail("test_eval_metrics_num_games_positive", path, rownum, f"num_games must be >0, got {num_games}")
            if abs((cp + bp) - tp) > 1e-6:
                _fail("test_eval_metrics_points_identity", path, rownum, f"cp+bp != total_points ({cp}+{bp}!={tp})")
            if abs(tp - float(num_games)) > 1e-6:
                _fail(
                    "test_eval_metrics_total_points_vs_num_games",
                    path,
                    rownum,
                    f"total_points should equal num_games ({tp} != {num_games})",
                )

            expected_wr = cp / tp if tp > _EPS else 0.0
            if abs(wr - expected_wr) > 1e-6:
                _fail(
                    "test_eval_metrics_win_rate_formula",
                    path,
                    rownum,
                    f"candidate_win_rate mismatch: got={wr}, expected={expected_wr}",
                )
            if passed and wr + _EPS < th:
                _fail(
                    "test_eval_metrics_pass_threshold",
                    path,
                    rownum,
                    f"passed=True but win_rate < threshold ({wr} < {th})",
                )
            if promoted and not passed:
                _fail(
                    "test_eval_metrics_promote_implies_pass",
                    path,
                    rownum,
                    "promoted_to_best=True while passed=False",
                )

            results_csv = (row.get("arena_results_csv") or "").strip()
            games_dir = (row.get("arena_games_dir") or "").strip()
            if results_csv and not os.path.exists(results_csv):
                _fail("test_eval_metrics_references_exist", path, rownum, f"missing arena_results_csv: {results_csv}")
            if games_dir and not os.path.isdir(games_dir):
                _fail("test_eval_metrics_references_exist", path, rownum, f"missing arena_games_dir: {games_dir}")

    if rows_checked == 0:
        _fail("test_eval_metrics_nonempty", path, 1, "eval_metrics.csv is empty")
    return rows_checked


def _classify_csv_path(p: str) -> Optional[str]:
    pp = p.lower()
    base = os.path.basename(pp)

    if not pp.endswith(".csv"):
        return None
    if "mcts_root" in base:
        return "root"
    if base == "arena_results.csv":
        return "arena_results"
    if base == "eval_metrics.csv":
        return "eval_metrics"
    if "/arena_games/" in pp or "\\arena_games\\" in pp:
        return "arena_game"
    if base.startswith("game_"):
        return "arena_game"
    return None


def _expand_args_to_csv_paths(args: Sequence[str]) -> List[str]:
    out: List[str] = []
    for a in args:
        hits = glob.glob(a)
        if not hits and os.path.exists(a):
            hits = [a]

        for h in hits:
            if os.path.isdir(h):
                out.extend(glob.glob(os.path.join(h, "**", "*.csv"), recursive=True))
            else:
                out.append(h)

    out = sorted(set(os.path.abspath(x) for x in out if str(x).lower().endswith(".csv")))
    return out


def _discover_default_paths() -> List[str]:
    patterns = [
        "simulator_output/Game_Version2/mcts_dnn_logs/**/*.csv",
        "vidur/simulator_output/Game_Version2/mcts_dnn_logs/**/*.csv",
    ]
    out: List[str] = []
    for p in patterns:
        out.extend(glob.glob(p, recursive=True))
    out = sorted(set(os.path.abspath(x) for x in out if str(x).lower().endswith(".csv")))
    return out


def main() -> None:
    args = sys.argv[1:]
    if args:
        csv_paths = _expand_args_to_csv_paths(args)
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found from args: {args}")
    else:
        csv_paths = _discover_default_paths()
        if not csv_paths:
            raise FileNotFoundError("No default GV2 root/eval CSV files found")

    by_kind: Dict[str, List[str]] = defaultdict(list)
    for p in csv_paths:
        kind = _classify_csv_path(p)
        if kind:
            by_kind[kind].append(p)

    if not any(by_kind.values()):
        raise FileNotFoundError("No mcts_root / arena_game / arena_results / eval_metrics CSV files matched")

    for k in by_kind:
        by_kind[k] = sorted(set(by_kind[k]))

    print("=== GV2 Root/Eval Log Tests ===")
    print(
        f"Discovered files: root={len(by_kind.get('root', []))}, "
        f"arena_game={len(by_kind.get('arena_game', []))}, "
        f"arena_results={len(by_kind.get('arena_results', []))}, "
        f"eval_metrics={len(by_kind.get('eval_metrics', []))}"
    )

    total_rows = 0
    try:
        for p in by_kind.get("root", []):
            n = _check_root_file(p)
            total_rows += n
            print(f"✅ root passed: {p} rows={n}")

        for p in by_kind.get("arena_game", []):
            n = _check_arena_game_file(p)
            total_rows += n
            print(f"✅ arena_game passed: {p} rows={n}")

        for p in by_kind.get("arena_results", []):
            n = _check_arena_results_file(p)
            total_rows += n
            print(f"✅ arena_results passed: {p} rows={n}")

        for p in by_kind.get("eval_metrics", []):
            n = _check_eval_metrics_file(p)
            total_rows += n
            print(f"✅ eval_metrics passed: {p} rows={n}")

    except TestFailure as e:
        print("\n❌ Root/Eval log test failed:\n")
        print(str(e))
        sys.exit(1)

    print(f"\n✅ All root/eval tests passed. total_rows_checked={total_rows}")


if __name__ == "__main__":
    main()
