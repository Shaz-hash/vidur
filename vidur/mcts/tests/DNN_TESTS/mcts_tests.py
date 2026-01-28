# # (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

# RUN THE FILE WITH CMD : python3 -m vidur.mcts.tests.DNN_TESTS.mcts_tests

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Config (matches alphaZero defaults)
# -----------------------------
MIN_PREFILL_TOKENS = 512
MAX_PREFILL_TOKENS = 3072
PREFILL_SLOWDOWN = 3.0

# float tolerances
DEADLINE_TOL = 1e-5
INTERVAL_EPS = 1e-9

DECODE_SLO = 0.05  # 50ms
BUDGET_STEP = 512  # controller interval
MAX_REQUESTS_NORM = 200  # not used here, but kept for future

@dataclass
class RequestState:
    rid: int
    prefill_tokens_total: int
    remaining_prefill: int
    arrived_at: float
    prefill_deadline: float

    # lateness tracking (matches environment._update_stats semantics)
    prefill_lateness: float = 0.0
    prefill_finalized: bool = False # means that we are done with calculating prefill lateness since the request is done with prefill stage 
    prefill_completed_at: Optional[float] = None

    decode_slo: float = DECODE_SLO
    decode_next_deadline: Optional[float] = None
    decode_done: int = 0
    decode_counted: int = 0
    decode_lateness: float = 0.0

    violated: bool = False
    completed: bool = False  # from state_completed_request_ids

@dataclass
class ObjectiveState:
    slo_violations: int = 0
    slo_lateness_sum: float = 0.0

# Update TraceContext to carry objective + request states
@dataclass
class TraceContext:
    expected_actor: str
    last_adv_batch_time: Optional[float]
    max_seen_request_id: int
    requests: Dict[int, RequestState]
    obj: ObjectiveState


# -----------------------------
# Helpers
# -----------------------------
def _resolve_existing_path(candidates: List[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


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
        # try to normalize python-ish literals
        try:
            s2 = s.replace("None", "null").replace("True", "true").replace("False", "false")
            return json.loads(s2)
        except Exception:
            return None


def _parse_int_list(cell: str) -> List[int]:
    val = _json_loads_maybe(cell)
    if isinstance(val, list):
        return [_safe_int(x, default=0) for x in val]
    return []


def _parse_deadline_map(cell: str) -> Dict[int, float]:
    val = _json_loads_maybe(cell)
    if isinstance(val, dict):
        out: Dict[int, float] = {}
        for k, v in val.items():
            out[_safe_int(k, default=-1)] = _safe_float(v, default=float("nan"))
        out.pop(-1, None)
        return out
    return {}


def _count_adversary_specs(action_repr: str) -> int:
    return len(re.findall(r"AdversaryRequestSpec\(", action_repr or ""))


def _parse_prefill_tokens(action_repr: str) -> List[int]:
    toks = re.findall(r"prefill_tokens=(\d+)", action_repr or "")
    return [int(t) for t in toks]


def _parse_prefill_slos(action_repr: str) -> List[float]:
    vals = re.findall(r"prefill_slo=([0-9eE+\-\.]+)", action_repr or "")
    out: List[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out

def _parse_token_budget(action_repr: str) -> int:
    m = re.search(r"token_budget=(\d+)", action_repr or "")
    return int(m.group(1)) if m else 0

def _parse_int_map(action_repr: str, field: str) -> Dict[int, int]:
    m = re.search(rf"{re.escape(field)}=\{{([^}}]*)\}}", action_repr or "")
    if not m:
        return {}
    inside = m.group(1).strip()
    if inside == "":
        return {}
    out: Dict[int, int] = {}
    for part in inside.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[int(k.strip())] = int(v.strip())
    return out



def load_prefill_profile(path: str) -> Dict[int, float]:
    table: Dict[int, float] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = _safe_int(row.get("prefill_tokens"), default=-1)
            v = _safe_float(row.get("prefill_time_seconds"), default=float("nan"))
            if k >= 0 and math.isfinite(v):
                table[k] = float(v)
    if not table:
        raise RuntimeError(f"Prefill profile seems empty: {path}")
    return table


# -----------------------------
# Row + context
# -----------------------------
@dataclass
class Row:
    rownum: int
    game_id: int
    root_id: int
    root_node_id: int
    root_player: str

    node_id: int
    parent_node_id: Optional[int]
    node_depth: int

    sim_time: float
    player_acted: str
    player_to_act: str

    action_index: int
    action_repr: str

    state_waiting_ids: List[int]
    state_completed_ids: List[int]

    adv_deadlines_by_id: Dict[int, float]

    slo_violations: int
    avg_lateness: float
    objective_cost: float


class TestFailure(AssertionError):
    pass


def _fail(test_name: str, msg: str, row: Row, trace_id: str) -> None:
    prefix = (
        f"[{test_name}] trace={trace_id} rownum={row.rownum} node_id={row.node_id} "
        f"depth={row.node_depth} sim_time={row.sim_time:.6f}"
    )
    ar = row.action_repr or ""
    if len(ar) > 300:
        ar = ar[:300] + "..."
    detail = (
        f"{prefix}\n"
        f"  player_acted={row.player_acted} player_to_act={row.player_to_act} action_index={row.action_index}\n"
        f"  action_repr={ar}\n"
        f"  waiting_ids={row.state_waiting_ids}\n"
        f"  adv_deadlines_by_id_keys={sorted(row.adv_deadlines_by_id.keys())}\n"
        f"  ERROR: {msg}"
    )
    raise TestFailure(detail)


# -----------------------------
# Tests (adversary-only for now)
# -----------------------------
def test_player_turn_consistency(ctx: TraceContext, row: Row, trace_id: str) -> None:
    # Test 0
    if row.player_acted != ctx.expected_actor:
        _fail(
            "test_player_turn_consistency",
            f"expected actor {ctx.expected_actor!r}, got {row.player_acted!r}",
            row,
            trace_id,
        )
    if row.player_to_act:
        ctx.expected_actor = row.player_to_act


def test_adversary_interval_and_update_arrival(
    ctx: TraceContext,
    row: Row,
    *,
    adv_request_count: int,
    trace_id: str,
) -> float:
    # Test 1
    if adv_request_count <= 0:
        return float("nan")

    sim_t = row.sim_time
    inferred_arrival = float(math.floor(sim_t + 1e-9))

    if ctx.last_adv_batch_time is None:
        ctx.last_adv_batch_time = inferred_arrival
        return inferred_arrival

    if sim_t + INTERVAL_EPS < ctx.last_adv_batch_time + 1.0:
        _fail(
            "test_adversary_interval",
            f"adversary created requests at sim_time={sim_t:.6f} but last_adv_batch_time={ctx.last_adv_batch_time:.6f} (<1s interval)",
            row,
            trace_id,
        )

    ctx.last_adv_batch_time = inferred_arrival
    return inferred_arrival


def test_adversary_count_matches_action_index(row: Row, *, adv_request_count: int, trace_id: str) -> None:
    # Test 2
    if adv_request_count <= 0:
        return
    expected = row.action_index + 1
    if adv_request_count != expected:
        _fail(
            "test_adversary_count_matches_action_index",
            f"expected {expected} requests for action_index={row.action_index}, got {adv_request_count}",
            row,
            trace_id,
        )


def test_adversary_prefill_token_bounds(
    row: Row,
    *,
    adv_prefill_tokens: List[int],
    trace_id: str,
) -> None:
    # Test 3
    for tok in adv_prefill_tokens:
        if tok < MIN_PREFILL_TOKENS or tok > MAX_PREFILL_TOKENS:
            _fail(
                "test_adversary_prefill_token_bounds",
                f"prefill_tokens={tok} out of bounds [{MIN_PREFILL_TOKENS},{MAX_PREFILL_TOKENS}]",
                row,
                trace_id,
            )


def test_adversary_request_ids_sequential(
    ctx: TraceContext,
    row: Row,
    *,
    new_ids: List[int],
    trace_id: str,
) -> None:
    # Test 4
    if not new_ids:
        return
    start = ctx.max_seen_request_id + 1
    expected = list(range(start, start + len(new_ids)))
    if new_ids != expected:
        _fail(
            "test_adversary_request_ids_sequential",
            f"expected new ids {expected}, got {new_ids} (prev_max={ctx.max_seen_request_id})",
            row,
            trace_id,
        )
    ctx.max_seen_request_id = expected[-1]


def test_adversary_deadlines_correct(
    ctx: TraceContext,
    row: Row,
    *,
    arrival_time: float,
    new_ids: List[int],
    prefill_tokens: List[int],
    prefill_slos: List[float],
    deadline_by_id: Dict[int, float],
    prefill_profile: Dict[int, float],
    trace_id: str,
) -> None:
    # Test 5
    if not new_ids:
        return

    if len(prefill_tokens) != len(new_ids):
        _fail(
            "test_adversary_deadlines_correct",
            f"prefill_tokens parsed ({len(prefill_tokens)}) != new_ids ({len(new_ids)})",
            row,
            trace_id,
        )
    if prefill_slos and len(prefill_slos) != len(new_ids):
        _fail(
            "test_adversary_deadlines_correct",
            f"prefill_slo parsed ({len(prefill_slos)}) != new_ids ({len(new_ids)})",
            row,
            trace_id,
        )

    for idx, (rid, tok) in enumerate(zip(new_ids, prefill_tokens)):
        if tok not in prefill_profile:
            _fail(
                "test_adversary_deadlines_correct",
                f"prefill_profile missing entry for prefill_tokens={tok}",
                row,
                trace_id,
            )

        base = float(prefill_profile[tok])
        expected_slo = base * float(PREFILL_SLOWDOWN)
        expected_deadline = float(arrival_time) + expected_slo

        actual_deadline = deadline_by_id.get(rid, float("nan"))
        if not math.isfinite(actual_deadline):
            _fail(
                "test_adversary_deadlines_correct",
                f"missing/NaN deadline for new req_id={rid}",
                row,
                trace_id,
            )
        if abs(actual_deadline - expected_deadline) > DEADLINE_TOL:
            _fail(
                "test_adversary_deadlines_correct",
                f"deadline mismatch for req_id={rid}: expected {expected_deadline:.9f}, got {actual_deadline:.9f}",
                row,
                trace_id,
            )

        if prefill_slos:
            actual_slo = float(prefill_slos[idx])
            if abs(actual_slo - expected_slo) > DEADLINE_TOL:
                _fail(
                    "test_adversary_deadlines_correct",
                    f"prefill_slo mismatch for req_id={rid}: expected {expected_slo:.9f}, got {actual_slo:.9f}",
                    row,
                    trace_id,
                )

        # store for future controller-phase tests
        # ctx.requests.setdefault(rid, {})
        # ctx.requests[rid].update(
        #     {"prefill_tokens": float(tok), "prefill_deadline": float(expected_deadline)}
        # )
        # store for future controller-phase tests
        if rid not in ctx.requests:
            ctx.requests[rid] = RequestState(
                rid=rid,
                prefill_tokens_total=int(tok),
                remaining_prefill=int(tok),
                arrived_at=float(arrival_time),
                prefill_deadline=float(expected_deadline),
                decode_slo=float(DECODE_SLO),
            )
        # else:
        #     # optional sanity check
        #     rs = ctx.requests[rid]
        #     rs.prefill_deadline = float(expected_deadline)




# ----------------------------
# Tests Controller 
# -----------------------------


def test_controller_budget_if_waiting(row: Row, token_budget: int, trace_id: str) -> None:
    # Test 1
    if row.state_waiting_ids and token_budget <= 0:
        _fail(
            "test_controller_budget_if_waiting",
            f"waiting_ids present but token_budget={token_budget}",
            row,
            trace_id,
        )

def test_controller_prefill_if_needed(ctx: TraceContext, row: Row, prefill_total: int, trace_id: str) -> None:
    # Test 2
    has_prefill_work = False
    for rid in row.state_waiting_ids:
        rs = ctx.requests.get(rid)
        if rs and (not rs.completed) and rs.remaining_prefill > 0:
            has_prefill_work = True
            break

    if has_prefill_work and prefill_total <= 0:
        _fail(
            "test_controller_prefill_if_needed",
            "prefill work exists but prefill_total==0",
            row,
            trace_id,
        )
    if (not has_prefill_work) and prefill_total != 0:
        _fail(
            "test_controller_prefill_if_needed",
            f"no prefill work but prefill_total={prefill_total}",
            row,
            trace_id,
        )

def test_controller_alloc_ids_exist(ctx: TraceContext, row: Row, token_alloc: Dict[int, int], trace_id: str) -> None:
    # Test 3
    for rid in token_alloc.keys():
        if rid not in ctx.requests:
            _fail(
                "test_controller_alloc_ids_exist",
                f"allocated to rid={rid} not in request dict",
                row,
                trace_id,
            )

def test_controller_prefill_budget_grid(row: Row, prefill_total: int, trace_id: str) -> None:
    # Test 4
    if prefill_total == 0:
        return
    if prefill_total < MIN_PREFILL_TOKENS or prefill_total > MAX_PREFILL_TOKENS:
        _fail(
            "test_controller_prefill_budget_grid",
            f"prefill_total={prefill_total} out of [{MIN_PREFILL_TOKENS},{MAX_PREFILL_TOKENS}]",
            row,
            trace_id,
        )
    if prefill_total % BUDGET_STEP != 0:
        _fail(
            "test_controller_prefill_budget_grid",
            f"prefill_total={prefill_total} not multiple of {BUDGET_STEP}",
            row,
            trace_id,
        )

def test_controller_allocations_respect_remaining(
    ctx: TraceContext,
    row: Row,
    prefill_alloc: Dict[int, int],
    decode_alloc: Dict[int, int],
    token_alloc: Dict[int, int],
    trace_id: str,
) -> None:
    # Test 5
    # Prefill allocations
    for rid, amt in prefill_alloc.items():
        rs = ctx.requests.get(rid)
        if rs is None:
            _fail("test_controller_allocations_respect_remaining", f"prefill rid={rid} not in dict", row, trace_id)
        if amt <= 0:
            _fail("test_controller_allocations_respect_remaining", f"prefill rid={rid} nonpositive amt={amt}", row, trace_id)
        if amt < BUDGET_STEP:
            _fail("test_controller_allocations_respect_remaining", f"prefill rid={rid} amt={amt} < {BUDGET_STEP}", row, trace_id)
        if amt % BUDGET_STEP != 0:
            _fail("test_controller_allocations_respect_remaining", f"prefill rid={rid} amt={amt} not multiple of {BUDGET_STEP}", row, trace_id)
        if amt > rs.remaining_prefill:
            _fail(
                "test_controller_allocations_respect_remaining",
                f"prefill rid={rid} amt={amt} exceeds remaining {rs.remaining_prefill}",
                row,
                trace_id,
            )

    # Decode allocations: must be 1 and only for prefill-complete
    for rid, amt in decode_alloc.items():
        rs = ctx.requests.get(rid)
        if rs is None:
            _fail("test_controller_allocations_respect_remaining", f"decode rid={rid} not in dict", row, trace_id)
        if amt != 1:
            _fail("test_controller_allocations_respect_remaining", f"decode rid={rid} amt={amt} != 1", row, trace_id)
        if rs.remaining_prefill != 0:
            _fail(
                "test_controller_allocations_respect_remaining",
                f"decode rid={rid} but remaining_prefill={rs.remaining_prefill}",
                row,
                trace_id,
            )

    # Any prefill-complete request that appears in waiting_ids should only get 1 token
    for rid in row.state_waiting_ids:
        rs = ctx.requests.get(rid)
        if rs and (not rs.completed) and rs.remaining_prefill == 0:
            if token_alloc.get(rid, 0) != 1:
                _fail(
                    "test_controller_allocations_respect_remaining",
                    f"rid={rid} prefill_done but token_alloc={token_alloc.get(rid)} != 1",
                    row,
                    trace_id,
                )

def test_controller_time_delta(
    row: Row,
    prev_row: Optional[Row],
    prefill_profile: Dict[int, float],
    prefill_total: int,
    trace_id: str,
) -> None:
    # Test 6
    if prev_row is None:
        return
    dt = float(row.sim_time) - float(prev_row.sim_time)
    if dt <= 0.0:
        _fail("test_controller_time_delta", f"dt={dt:.9f} not >0", row, trace_id)

    # if prefill_total >= BUDGET_STEP:
    #     lo = prefill_profile.get(prefill_total)
    #     max_profile_tokens = max(prefill_profile.keys())
    #     hi_key = min(prefill_total + BUDGET_STEP, max_profile_tokens)
    #     hi = prefill_profile.get(hi_key)

    #     if lo is None or hi is None:
    #         _fail(
    #             "test_controller_time_delta",
    #             f"missing prefill_profile for {prefill_total} or {min(prefill_total + BUDGET_STEP, MAX_PREFILL_TOKENS)}",
    #             row,
    #             trace_id,
    #         )
    #     if dt + DEADLINE_TOL < lo:
    #         _fail("test_controller_time_delta", f"dt={dt:.9f} < prefill_time({prefill_total})={lo:.9f}", row, trace_id)
    #     if dt - DEADLINE_TOL > hi:
    #         _fail(
    #             "test_controller_time_delta",
    #             f"dt={dt:.9f} > prefill_time({min(prefill_total + BUDGET_STEP, MAX_PREFILL_TOKENS)})={hi:.9f}",
    #             row,
    #             trace_id,
    #         )
    
    if prefill_total >= BUDGET_STEP:
        min_profile_tokens = min(prefill_profile.keys())
        max_profile_tokens = max(prefill_profile.keys())

        lo_key = max(min_profile_tokens, prefill_total - BUDGET_STEP)
        hi_key = min(max_profile_tokens, prefill_total + BUDGET_STEP)

        lo = prefill_profile.get(lo_key)
        hi = prefill_profile.get(hi_key)

        if lo is None or hi is None:
            _fail(
                "test_controller_time_delta",
                f"missing prefill_profile for lo={lo_key} or hi={hi_key}",
                row,
                trace_id,
            )

        if dt + DEADLINE_TOL < lo:
            _fail(
                "test_controller_time_delta",
                f"dt={dt:.9f} < prefill_time({lo_key})={lo:.9f}",
                row,
                trace_id,
            )
        if dt - DEADLINE_TOL > hi:
            _fail(
                "test_controller_time_delta",
                f"dt={dt:.9f} > prefill_time({hi_key})={hi:.9f}",
                row,
                trace_id,
            )

    
    else:
        # decode-only: just bound it by prefill_time(1024) as you requested
        cap = prefill_profile.get(1024)
        if cap is not None and dt - DEADLINE_TOL > cap:
            _fail("test_controller_time_delta", f"decode-only dt={dt:.9f} > prefill_time(1024)={cap:.9f}", row, trace_id)


def _apply_controller_and_update_objective(
    ctx: TraceContext,
    row: Row,
    prefill_alloc: Dict[int, int],
    decode_alloc: Dict[int, int],
) -> None:
    sim_time = float(row.sim_time)

    # Apply prefill allocations (and set prefill_completed_at when it hits 0)
    for rid, amt in prefill_alloc.items():
        rs = ctx.requests[rid]
        rs.remaining_prefill = max(0, int(rs.remaining_prefill) - int(amt))
        if rs.remaining_prefill == 0 and rs.prefill_completed_at is None:
            rs.prefill_completed_at = sim_time

    # Apply decode allocations (each should be 1)
    for rid, amt in decode_alloc.items():
        ctx.requests[rid].decode_done += int(amt)

    # --- Prefill lateness: monotone max, finalize after prefill completes ---
    for rs in ctx.requests.values():
        if rs.prefill_finalized:
            continue

        deadline = float(rs.prefill_deadline)
        is_prefill_complete = (rs.remaining_prefill == 0 and rs.prefill_completed_at is not None)
        actual = float(rs.prefill_completed_at) if is_prefill_complete else sim_time

        prefill_late = max(0.0, actual - deadline)
        if prefill_late > rs.prefill_lateness:
            ctx.obj.slo_lateness_sum += (prefill_late - rs.prefill_lateness)
            rs.prefill_lateness = prefill_late

        if is_prefill_complete:
            rs.prefill_finalized = True

    # --- Decode lateness: per new decode token (deadline = prefill_completed_at + slo, then next = sim_time + slo) ---
    for rs in ctx.requests.values():
        if rs.remaining_prefill != 0 or rs.prefill_completed_at is None:
            continue

        if rs.decode_next_deadline is None:
            rs.decode_next_deadline = float(rs.prefill_completed_at) + float(rs.decode_slo)

        new_tokens = int(rs.decode_done) - int(rs.decode_counted)
        if new_tokens:
            if new_tokens != 1:
                raise AssertionError(f"Expected 1 new decode token for req {rs.rid}, got {new_tokens}")

            token_late = max(0.0, sim_time - float(rs.decode_next_deadline))
            rs.decode_lateness += float(token_late)
            ctx.obj.slo_lateness_sum += float(token_late)

            rs.decode_counted = int(rs.decode_done)
            rs.decode_next_deadline = sim_time + float(rs.decode_slo)

    # --- Violations: once per request when total lateness becomes >0 ---
    for rs in ctx.requests.values():
        total_late = float(rs.prefill_lateness) + float(rs.decode_lateness)
        if total_late > 0.0 and not rs.violated:
            rs.violated = True
            ctx.obj.slo_violations += 1

    # Mark completed ids (so we don't treat them as active later)
    for rid in row.state_completed_ids:
        if rid in ctx.requests:
            ctx.requests[rid].completed = True


def test_objective_matches_log(ctx: TraceContext, row: Row, trace_id: str) -> None:
    got_viol = int(ctx.obj.slo_violations)
    got_late = float(ctx.obj.slo_lateness_sum)          # SUM (not avg)
    got_obj = got_viol + got_late

    exp_viol = int(row.slo_violations)
    exp_obj = float(row.objective_cost)                 # column still named avg_lateness in CSV
    exp_late = max(0.0, exp_obj - float(exp_viol))

    if got_viol != exp_viol:
        _fail("test_objective_matches_log", f"slo_violations mismatch: expected {exp_viol}, got {got_viol}", row, trace_id)
    if abs(got_late - exp_late) > DEADLINE_TOL:
        _fail("test_objective_matches_log", f"lateness_sum mismatch: expected {exp_late:.9f}, got {got_late:.9f}", row, trace_id)
    if abs(got_obj - exp_obj) > DEADLINE_TOL:
        _fail("test_objective_matches_log", f"objective_cost mismatch: expected {exp_obj:.9f}, got {got_obj:.9f}", row, trace_id)


# -----------------------------
# Trace running
# -----------------------------
def run_trace(trace_rows: List[Row], *, prefill_profile: Dict[int, float]) -> int:
    trace_id = f"game={trace_rows[0].game_id} root={trace_rows[0].root_id} leaf={trace_rows[-1].node_id}"
    ctx = TraceContext(
        expected_actor=trace_rows[0].root_player,
        last_adv_batch_time=None,
        max_seen_request_id=-1,
        requests={},
        obj=ObjectiveState(),
    )

    adv_actions_checked = 0
    prev_row = None
    for row in trace_rows:
        test_player_turn_consistency(ctx, row, trace_id)

        ids_seen = set(row.state_waiting_ids) | set(row.state_completed_ids)

        if row.player_acted == "adversary":
            adv_request_count = _count_adversary_specs(row.action_repr)
            deadline_map = row.adv_deadlines_by_id

            # ensure deadlines-by-id logging matches how many requests were created
            if adv_request_count > 0 and len(deadline_map) != adv_request_count:
                _fail(
                    "test_adversary_deadlines_map_count",
                    f"adversary created {adv_request_count} requests but adversary_prefill_deadlines_by_id has {len(deadline_map)} entries",
                    row,
                    trace_id,
                )

            new_ids_sorted = sorted(deadline_map.keys()) if adv_request_count > 0 else []

            # incorporate existing ids (exclude this row's new ids) before checking sequential IDs
            existing_ids = ids_seen - set(new_ids_sorted)
            if existing_ids:
                ctx.max_seen_request_id = max(ctx.max_seen_request_id, max(existing_ids))

            if adv_request_count > 0:
                adv_actions_checked += 1

                arrival_time = test_adversary_interval_and_update_arrival(
                    ctx, row, adv_request_count=adv_request_count, trace_id=trace_id
                )
                test_adversary_count_matches_action_index(
                    row, adv_request_count=adv_request_count, trace_id=trace_id
                )

                prefill_tokens = _parse_prefill_tokens(row.action_repr)
                prefill_slos = _parse_prefill_slos(row.action_repr)

                test_adversary_prefill_token_bounds(
                    row, adv_prefill_tokens=prefill_tokens, trace_id=trace_id
                )
                test_adversary_request_ids_sequential(
                    ctx, row, new_ids=new_ids_sorted, trace_id=trace_id
                )
                test_adversary_deadlines_correct(
                    ctx,
                    row,
                    arrival_time=arrival_time,
                    new_ids=new_ids_sorted,
                    prefill_tokens=prefill_tokens,
                    prefill_slos=prefill_slos,
                    deadline_by_id=deadline_map,
                    prefill_profile=prefill_profile,
                    trace_id=trace_id,
                )

            # after validation, include new ids in max tracking
            if new_ids_sorted:
                ctx.max_seen_request_id = max(ctx.max_seen_request_id, max(new_ids_sorted))

        else:
            # controller row: incorporate any seen ids
            if ids_seen:
                ctx.max_seen_request_id = max(ctx.max_seen_request_id, max(ids_seen))
            token_budget = _parse_token_budget(row.action_repr)
            token_alloc = _parse_int_map(row.action_repr, "token_allocations")
            prefill_alloc = _parse_int_map(row.action_repr, "prefill_allocations")
            decode_alloc = _parse_int_map(row.action_repr, "decode_allocations")

            prefill_total = sum(prefill_alloc.values())
            decode_total = sum(decode_alloc.values())

            # Test 1..6
            test_controller_budget_if_waiting(row, token_budget, trace_id)
            test_controller_prefill_if_needed(ctx, row, prefill_total, trace_id)
            test_controller_alloc_ids_exist(ctx, row, token_alloc, trace_id)
            test_controller_prefill_budget_grid(row, prefill_total, trace_id)

            # extra sanity: token_budget must equal sum(token_allocations)
            if token_budget != sum(token_alloc.values()):
                _fail(
                    "test_controller_token_budget_sum",
                    f"token_budget {token_budget} != sum(token_alloc) {sum(token_alloc.values())}",
                    row,
                    trace_id,
                )

            test_controller_allocations_respect_remaining(ctx, row, prefill_alloc, decode_alloc, token_alloc, trace_id)
            test_controller_time_delta(row, prev_row, prefill_profile, prefill_total, trace_id)

            # Test 7: apply allocations + update lateness/violations/objective + compare with log
            _apply_controller_and_update_objective(ctx, row, prefill_alloc, decode_alloc)
            test_objective_matches_log(ctx, row, trace_id)

            prev_row = row
    

    return adv_actions_checked


# -----------------------------
# CSV loading + leaf-trace building
# -----------------------------
def load_rows(mcts_iter_path: str) -> List[Row]:
    out: List[Row] = []
    with open(mcts_iter_path, newline="") as f:
        r = csv.DictReader(f)
        for i, raw in enumerate(r, start=2):  # header is line 1
            parent_cell = (raw.get("parent_node_id") or "").strip()
            parent_node_id = None if parent_cell == "" else _safe_int(parent_cell, default=-1)
            if parent_node_id == -1:
                parent_node_id = None

            out.append(
                Row(
                    rownum=i,
                    game_id=_safe_int(raw.get("game_id"), 0),
                    root_id=_safe_int(raw.get("root_id"), 0),
                    root_node_id=_safe_int(raw.get("root_node_id"), 0),
                    root_player=(raw.get("root_player") or "").strip(),
                    node_id=_safe_int(raw.get("node_id"), 0),
                    parent_node_id=parent_node_id,
                    node_depth=_safe_int(raw.get("node_depth"), 0),
                    sim_time=_safe_float(raw.get("sim_time"), 0.0),
                    player_acted=(raw.get("player_acted_to_create_this_node") or "").strip(),
                    player_to_act=(raw.get("player_to_act_in_this_node") or "").strip(),
                    action_index=_safe_int(raw.get("action_index"), -1),
                    action_repr=(raw.get("action_repr") or ""),
                    state_waiting_ids=_parse_int_list(raw.get("state_waiting_ids") or ""),
                    state_completed_ids=_parse_int_list(raw.get("state_completed_request_ids") or ""),
                    adv_deadlines_by_id=_parse_deadline_map(
                        raw.get("adversary_prefill_deadlines_by_id") or ""
                    ),
                    slo_violations=_safe_int(raw.get("slo_violations"), 0),
                    avg_lateness=_safe_float(raw.get("avg_lateness"), 0.0),
                    objective_cost=_safe_float(raw.get("objective_cost"), 0.0),
                )
            )
    return out


def build_leaf_traces(rows: List[Row]) -> List[List[Row]]:
    by_root: Dict[Tuple[int, int], Dict[int, Row]] = {}
    parent_set_by_root: Dict[Tuple[int, int], set] = {}

    for row in rows:
        key = (row.game_id, row.root_id)
        by_root.setdefault(key, {})[row.node_id] = row
        parent_set_by_root.setdefault(key, set())
        if row.parent_node_id is not None:
            parent_set_by_root[key].add(int(row.parent_node_id))

    traces: List[List[Row]] = []
    for key, nodes in by_root.items():
        parents = parent_set_by_root.get(key, set())
        leaves = [nid for nid in nodes.keys() if nid not in parents]
        for leaf_id in leaves:
            chain: List[Row] = []
            cur = nodes[leaf_id]
            while True:
                chain.append(cur)
                pid = cur.parent_node_id
                if pid is None or pid not in nodes:
                    break
                cur = nodes[pid]
            chain.reverse()
            traces.append(chain)

    traces.sort(key=lambda t: (t[0].game_id, t[0].root_id, t[-1].node_depth, t[-1].node_id))
    return traces


def main() -> None:
    mcts_iter_path = _resolve_existing_path(
        [
            "vidur/simulator_output/mcts_dnn_logs/mcts_iter.csv",
            "simulator_output/mcts_dnn_logs/mcts_iter.csv",
        ]
    )
    prefill_profile_path = _resolve_existing_path(
        [
            "vidur/simulator_output/prefill_profile.csv",
            "simulator_output/prefill_profile.csv",
        ]
    )

    prefill_profile = load_prefill_profile(prefill_profile_path)
    rows = load_rows(mcts_iter_path)
    traces = build_leaf_traces(rows)

    print(f"Loaded {len(rows)} rows from {mcts_iter_path}")
    print(f"Built {len(traces)} leaf traces")

    total_adv_actions = 0
    try:
        for tr in traces:
            total_adv_actions += run_trace(tr, prefill_profile=prefill_profile)
    except TestFailure as e:
        print("\n❌ MCTS_DNN adversary log test failed:\n")
        print(str(e))
        sys.exit(1)

    print(
        f"\n✅ All adversary log tests passed. traces={len(traces)} adversary_actions_checked={total_adv_actions}"
    )


if __name__ == "__main__":
    main()
























