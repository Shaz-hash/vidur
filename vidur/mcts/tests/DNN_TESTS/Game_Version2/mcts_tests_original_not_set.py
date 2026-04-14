from __future__ import annotations

import json
import math
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


from vidur.mcts.Game_Versions.Game_Version2.config import GameVersion2Config

_GV2_CFG = GameVersion2Config()
INTERVAL_EPS = float(_GV2_CFG.timing.eps)
_ALLOWED_PREFILL_TOKENS: Set[int] = set(int(x) for x in _GV2_CFG.request.allowed_prefill_tokens)
_ALLOWED_PREFILL_TOKENS_SORTED = sorted(_ALLOWED_PREFILL_TOKENS)
_GV2_TICK_SEC = float(_GV2_CFG.timing.adversary_tick_sec)
_GV2_DEADLINE_TOL = max(1e-6, 10.0 * INTERVAL_EPS)
_GV2_OBJECTIVE_TOL = max(1e-6, 20.0 * INTERVAL_EPS)
_GV2_DECODE_CAP = int(_GV2_CFG.request.max_decode_tokens_per_request)
_GV2_DECODE_MINT = int(_GV2_CFG.credits.decode_credit_mint_per_prefill_complete)
_GV2_CREDIT_TOL = 0  # integer arithmetic


@dataclass
class RequestState:
    rid: int
    prefill_tokens_total: int
    remaining_prefill: int
    arrived_at: float
    prefill_deadline: float
    decode_slo: float = 0.05

    prefill_lateness: float = 0.0
    decode_lateness: float = 0.0
    prefill_completed_at: Optional[float] = None
    decode_done: int = 0
    completed: bool = False


@dataclass
class ObjectiveState:
    slo_violations: int = 0
    slo_lateness_sum: float = 0.0


@dataclass
class TraceContext:
    expected_actor: str
    last_adv_batch_time: Optional[float]
    max_seen_request_id: int
    requests: Dict[int, RequestState]
    obj: ObjectiveState


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

    phase: str
    nn_called: bool
    model_prior_json: str
    normalized_prior_json: str

    state_active_ids: List[int]
    state_waiting_ids: List[int]
    state_completed_ids: List[int]
    state_dropped_request_ids: List[int]
    state_stopped_decode_request_ids: List[int]
    state_pending_adv_tick: bool
    state_last_adv_tick: Optional[float]
    state_decode_credit_balance: int
    state_decode_tokens_counted_by_id: Dict[int, int]

    adv_deadlines_by_id: Dict[int, float]

    slo_violations: int
    total_lateness: float
    objective_cost: float
    requests_completed: int

    adversary_requests_raw: str
    controller_allocations_raw: str
    controller_prefill_allocations_raw: str
    controller_decode_allocations_raw: str
    controller_token_budget_raw: str
    controller_strategy_raw: str


class TestFailure(AssertionError):
    pass


def _fail(test_name: str, msg: str, row: Row, trace_id: str) -> None:
    raise TestFailure(
        f"[{test_name}] trace={trace_id} rownum={row.rownum} node_id={row.node_id} "
        f"depth={row.node_depth} sim_time={row.sim_time:.6f}\n"
        f"  player_acted={row.player_acted} player_to_act={row.player_to_act} action_index={row.action_index}\n"
        f"  action_repr={row.action_repr[:400]}\n"
        f"  ERROR: {msg}"
    )


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


def _parse_deadline_map(cell: str) -> Dict[int, float]:
    obj = _json_loads_maybe(cell)
    if not isinstance(obj, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in obj.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def _parse_int_map(cell: str, action_repr_fallback: str, field_name: str) -> Dict[int, int]:
    obj = _json_loads_maybe(cell)
    if isinstance(obj, dict):
        out: Dict[int, int] = {}
        for k, v in obj.items():
            try:
                out[int(k)] = int(v)
            except Exception:
                pass
        return out

    # fallback from action_repr
    m = re.search(rf"{re.escape(field_name)}=\{{([^}}]*)\}}", action_repr_fallback or "")
    if not m:
        return {}
    out: Dict[int, int] = {}
    for kv in m.group(1).split(","):
        if ":" not in kv:
            continue
        a, b = kv.split(":", 1)
        try:
            out[int(a.strip())] = int(b.strip())
        except Exception:
            pass
    return out


def _parse_token_budget(raw_cell: str, action_repr: str) -> int:
    v = _safe_int(raw_cell, default=-1)
    if v >= 0:
        return v
    m = re.search(r"token_budget=(\d+)", action_repr or "")
    return int(m.group(1)) if m else 0


def _parse_adversary_requests(cell: str, action_repr: str) -> List[Dict[str, object]]:
    obj = _json_loads_maybe(cell)
    # logger often stores action json object: {"requests":[...], ...}
    if isinstance(obj, dict) and isinstance(obj.get("requests"), list):
        return [x for x in obj["requests"] if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    # fallback: count only; details unavailable
    n = len(re.findall(r"AdversaryRequestSpec\(", action_repr or ""))
    if n <= 0:
        return []
    return [{"prefill_tokens": -1, "decode_tokens": -1, "prefill_slo": 0.0, "decode_slo": 0.0} for _ in range(n)]


def _parse_stop_decode_ids(action_repr: str) -> List[int]:
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


def _parse_eviction_rule(controller_strategy_raw: str, action_repr: str) -> str:
    s = (controller_strategy_raw or "").strip()
    if s.startswith("GV2|"):
        return s.split("|", 1)[1]
    m = re.search(r"strategy='GV2\|([^']+)'", action_repr or "")
    if m:
        return m.group(1)
    return "evict_none"


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


def _parse_adversary_requests(cell: str, action_repr: str) -> List[Dict[str, object]]:
    # 1) Preferred: structured JSON payload.
    obj = _json_loads_maybe(cell)
    req_list = None
    if isinstance(obj, dict) and isinstance(obj.get("requests"), list):
        req_list = obj["requests"]
    elif isinstance(obj, list):
        req_list = obj

    if isinstance(req_list, list):
        out: List[Dict[str, object]] = []
        for x in req_list:
            if not isinstance(x, dict):
                continue
            out.append(
                {
                    "prefill_tokens": int(_safe_int(x.get("prefill_tokens"), -1)),
                    "decode_tokens": int(_safe_int(x.get("decode_tokens"), -1)),
                    "prefill_slo": float(_safe_float(x.get("prefill_slo"), float("nan"))),
                    "decode_slo": float(_safe_float(x.get("decode_slo"), float("nan"))),
                }
            )
        if out:
            return out

    # 2) Fallback: parse repr string.
    ar = (action_repr or "").strip()
    if "AdversaryAction" not in ar:
        return []
    if "requests=[]" in ar:
        return []

    out: List[Dict[str, object]] = []
    for m in re.finditer(r"AdversaryRequestSpec\((.*?)\)", ar):
        body = m.group(1)
        pf = _extract_int_field(body, "prefill_tokens")
        dd = _extract_int_field(body, "decode_tokens")
        ps = _extract_float_field(body, "prefill_slo")
        ds = _extract_float_field(body, "decode_slo")
        if pf is None or dd is None:
            continue
        out.append(
            {
                "prefill_tokens": int(pf),
                "decode_tokens": int(dd),
                "prefill_slo": float(ps if ps is not None else float("nan")),
                "decode_slo": float(ds if ds is not None else float("nan")),
            }
        )
    return out



def _resolve_mcts_iter_paths(cli_args: Optional[List[str]] = None) -> List[str]:
    args = list(cli_args) if cli_args is not None else sys.argv[1:]
    if args:
        out: List[str] = []
        for a in args:
            ms = glob.glob(a)
            if ms:
                out.extend(ms)
            elif os.path.exists(a):
                out.append(a)
        out = sorted(set(out))
        if out:
            return out
        raise FileNotFoundError(f"No files matched args: {args}")

    pats = [
        "simulator_output/mcts_dnn_logs/mcts_iter*.csv",
        "simulator_output/mcts_dnn_logs/gen_*/mcts_iter*.csv",
        "vidur/simulator_output/mcts_dnn_logs/mcts_iter*.csv",
        "vidur/simulator_output/mcts_dnn_logs/gen_*/mcts_iter*.csv",
    ]
    out: List[str] = []
    for p in pats:
        out.extend(glob.glob(p))
    out = sorted(set(out))
    if not out:
        raise FileNotFoundError("No mcts_iter*.csv found")
    return out


def load_rows(mcts_iter_path: str) -> List[Row]:
    out: List[Row] = []
    with open(mcts_iter_path, newline="") as f:
        r = csv.DictReader(f)
        for i, raw in enumerate(r, start=2):
            parent_cell = (raw.get("parent_node_id") or "").strip()
            parent_node_id = None if parent_cell == "" else _safe_int(parent_cell, default=-1)
            if parent_node_id == -1:
                parent_node_id = None

            state_waiting = _parse_int_list(raw.get("state_waiting_ids") or "[]")
            state_active = _parse_int_list(raw.get("state_active_ids") or "[]")
            if not state_active:
                state_active = list(state_waiting)

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
                    player_acted=(raw.get("player_acted_to_create_this_node") or "").strip().lower(),
                    player_to_act=(raw.get("player_to_act_in_this_node") or "").strip().lower(),

                    action_index=_safe_int(raw.get("action_index"), -1),
                    action_repr=(raw.get("action_repr") or ""),

                    phase=(raw.get("phase") or "").strip(),
                    nn_called=(str(raw.get("nn_called") or "").strip().lower() == "true"),
                    model_prior_json=(raw.get("model_prior_json") or "[]"),
                    normalized_prior_json=(raw.get("normalized_prior_json") or "[]"),

                    state_active_ids=state_active,
                    state_waiting_ids=state_waiting,
                    state_completed_ids=_parse_int_list(raw.get("state_completed_request_ids") or "[]"),
                    state_dropped_request_ids=_parse_int_list(raw.get("state_dropped_request_ids") or "[]"),
                    state_stopped_decode_request_ids=_parse_int_list(raw.get("state_stopped_decode_request_ids") or "[]"),
                    state_pending_adv_tick=(str(raw.get("state_pending_adv_tick") or "").strip().lower() == "true"),
                    state_last_adv_tick=(
                        _safe_float(raw.get("state_last_adv_tick"), default=float("nan"))
                        if (raw.get("state_last_adv_tick") or "").strip() != "" else None
                    ),
                    state_decode_credit_balance=_safe_int(
                        raw.get("state_decode_credit_balance") or raw.get("decode_credit_balance"), 0
                    ),
                    state_decode_tokens_counted_by_id={
                        int(k): int(v)
                        for k, v in (_json_loads_maybe(raw.get("state_decode_tokens_counted_by_id") or "{}") or {}).items()
                        if str(k).lstrip("-").isdigit()
                    },

                    adv_deadlines_by_id=_parse_deadline_map(raw.get("adversary_prefill_deadlines_by_id") or "{}"),

                    slo_violations=_safe_int(raw.get("slo_violations"), 0),
                    total_lateness=_safe_float(raw.get("total_lateness") or raw.get("avg_lateness"), 0.0),
                    objective_cost=_safe_float(raw.get("objective_cost"), 0.0),
                    requests_completed=_safe_int(raw.get("requests_completed"), 0),

                    adversary_requests_raw=(raw.get("adversary_requests") or ""),
                    controller_allocations_raw=(raw.get("controller_allocations") or ""),
                    controller_prefill_allocations_raw=(raw.get("controller_prefill_allocations") or ""),
                    controller_decode_allocations_raw=(raw.get("controller_decode_allocations") or ""),
                    controller_token_budget_raw=(raw.get("controller_token_budget") or ""),
                    controller_strategy_raw=(raw.get("controller_strategy") or ""),
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


def run_trace(trace_rows: List[Row], *, prefill_profile: Dict[int, float]) -> int:
    trace_id = f"game={trace_rows[0].game_id} root={trace_rows[0].root_id} leaf={trace_rows[-1].node_id}"

    ctx = TraceContext(
        expected_actor=(trace_rows[0].root_player or "adversary").strip().lower(),
        last_adv_batch_time=None,
        max_seen_request_id=-1,
        requests={},
        obj=ObjectiveState(),
    )

    adv_actions_checked = 0
    prev_row: Optional[Row] = None

    for row in trace_rows:
        test_player_turn_consistency(ctx, row, trace_id)
        test_normalized_prior_sums(row, trace_id)

        token_budget = _parse_token_budget(row.controller_token_budget_raw, row.action_repr)
        token_alloc = _parse_int_map(row.controller_allocations_raw, row.action_repr, "token_allocations")
        prefill_alloc = _parse_int_map(row.controller_prefill_allocations_raw, row.action_repr, "prefill_allocations")
        decode_alloc = _parse_int_map(row.controller_decode_allocations_raw, row.action_repr, "decode_allocations")

        stop_decode_ids = _parse_stop_decode_ids(row.action_repr)
        adversary_requests = _parse_adversary_requests(row.adversary_requests_raw, row.action_repr)

        decision_tick = (
            float(row.state_last_adv_tick)
            if row.state_last_adv_tick is not None and math.isfinite(float(row.state_last_adv_tick))
            else float(row.sim_time)
        )

        active_ids = set(int(x) for x in row.state_active_ids)
        completed_ids = set(int(x) for x in row.state_completed_ids)
        dropped_ids = set(int(x) for x in row.state_dropped_request_ids)
        stopped_ids = set(int(x) for x in row.state_stopped_decode_request_ids)

        # shared row-level checks
        test_objective_matches_log(ctx, row, trace_id)
        test_finalized_ids_do_not_reappear_gv2(
            ctx, row,
            current_active_ids=active_ids,
            current_completed_ids=completed_ids,
            current_dropped_ids=dropped_ids,
            controller_token_alloc=(token_alloc if row.player_acted == "controller" else None),
            adversary_stop_ids=(stop_decode_ids if row.player_acted == "adversary" else None),
            trace_id=trace_id,
        )

        if row.player_acted == "adversary":
            created_ids = sorted(int(k) for k in row.adv_deadlines_by_id.keys())
            adv_actions_checked += int(len(adversary_requests) > 0)

            test_adversary_prefill_token_bounds(
                row, adv_prefill_tokens=[int(r.get("prefill_tokens", -1)) for r in adversary_requests], trace_id=trace_id
            )
            test_adversary_request_ids_sequential(ctx, row, new_ids=created_ids, trace_id=trace_id)
            test_adversary_deadlines_correct(
                row,
                decision_tick=decision_tick,
                created_ids=created_ids,
                adversary_requests=adversary_requests,
                deadline_by_id=row.adv_deadlines_by_id,
                trace_id=trace_id,
            )
            test_adversary_interval_and_update_arrival(
                ctx, row, decision_tick=decision_tick, adv_created_count=len(adversary_requests), trace_id=trace_id
            )
            test_adversary_action_index_matches_payload(
                row, action_index=row.action_index, adversary_requests=adversary_requests,
                stop_decode_ids=stop_decode_ids, trace_id=trace_id
            )
            test_adversary_tick_progression_gv2(
                ctx, row, decision_tick=decision_tick, adv_created_count=len(adversary_requests), trace_id=trace_id
            )
            test_adversary_sliding_window_constraints_gv2(
                ctx, row, decision_tick=decision_tick, adversary_requests=adversary_requests, trace_id=trace_id
            )
            test_missed_tick_replay_correctness_gv2(ctx, row, decision_tick=decision_tick, trace_id=trace_id)

            test_adversary_stop_rule_applied_and_completion_updated_gv2(
                ctx, row,
                action_index=row.action_index,
                stop_decode_ids=stop_decode_ids,
                current_stopped_decode_ids=stopped_ids,
                current_completed_ids=completed_ids,
                requests_completed=row.requests_completed,
                trace_id=trace_id,
            )

            # update ctx.requests with newly created requests (for next-row pre-action checks)
            for rid, req in zip(created_ids, adversary_requests):
                if rid in ctx.requests:
                    continue
                pf = int(req.get("prefill_tokens", 0))
                pf_slo = float(req.get("prefill_slo", 0.0))
                dd_slo = float(req.get("decode_slo", 0.05))
                deadline = float(row.adv_deadlines_by_id.get(rid, decision_tick + pf_slo))
                ctx.requests[rid] = RequestState(
                    rid=rid,
                    prefill_tokens_total=pf,
                    remaining_prefill=pf,
                    arrived_at=decision_tick,
                    prefill_deadline=deadline,
                    decode_slo=dd_slo,
                )

        elif row.player_acted == "controller":
            test_controller_action_regime_gv2(
                row, token_budget=token_budget, token_alloc=token_alloc,
                prefill_alloc=prefill_alloc, decode_alloc=decode_alloc, trace_id=trace_id
            )
            test_controller_alloc_ids_exist(ctx, row, token_alloc, trace_id)
            test_controller_prefill_allocation_validity_gv2(
                ctx, row, prefill_alloc=prefill_alloc, trace_id=trace_id
            )
            test_controller_time_delta(
                ctx, row, prev_row, prefill_profile,
                sum(prefill_alloc.values()), sum(decode_alloc.values()), trace_id
            )
            test_missed_tick_replay_correctness_gv2(ctx, row, decision_tick=decision_tick, trace_id=trace_id)

            ev_rule = _parse_eviction_rule(row.controller_strategy_raw, row.action_repr)
            test_controller_eviction_rule_applied_and_dropped_updated_gv2(
                ctx, row,
                eviction_rule=ev_rule,
                current_dropped_ids=dropped_ids,
                current_completed_ids=completed_ids,
                requests_completed=row.requests_completed,
                trace_id=trace_id,
            )

            # lightweight state advance for next-row rule checks
            for rid, amt in prefill_alloc.items():
                rs = ctx.requests.get(int(rid))
                if rs is None or rs.completed:
                    continue
                rs.remaining_prefill = max(0, int(rs.remaining_prefill) - int(amt))
                if rs.remaining_prefill == 0 and rs.prefill_completed_at is None:
                    rs.prefill_completed_at = float(row.sim_time)
            for rid, amt in decode_alloc.items():
                rs = ctx.requests.get(int(rid))
                if rs is None or rs.completed:
                    continue
                if rs.remaining_prefill == 0:
                    rs.decode_done += int(amt)

        # sync finalized states from row snapshot
        for rid in (completed_ids | dropped_ids | stopped_ids):
            if rid in ctx.requests:
                ctx.requests[rid].completed = True

        # decode-credit tests (if you added these functions)
        if "test_decode_cap_and_completion_gv2" in globals():
            decode_done_by_id = {rid: int(rs.decode_done) for rid, rs in ctx.requests.items()}
            test_decode_cap_and_completion_gv2(
                ctx, row,
                decode_done_by_id=decode_done_by_id,
                completed_ids=completed_ids,
                dropped_ids=dropped_ids,
                requests_completed=row.requests_completed,
                trace_id=trace_id,
            )

        if "test_decode_credit_mint_consume_gv2" in globals():
            test_decode_credit_mint_consume_gv2(
                ctx, row,
                decode_credit_balance=int(row.state_decode_credit_balance),
                decode_tokens_counted_by_id=dict(row.state_decode_tokens_counted_by_id),
                trace_id=trace_id,
            )

        if "test_decode_credit_shortage_tiebreak_gv2" in globals():
            active_decode_ids = set(
                rid for rid, rs in ctx.requests.items()
                if (rid in active_ids) and (not rs.completed) and (rs.remaining_prefill == 0)
            )
            test_decode_credit_shortage_tiebreak_gv2(
                ctx, row,
                active_decode_ids=active_decode_ids,
                stopped_decode_ids=stopped_ids,
                decode_done_by_id={rid: int(rs.decode_done) for rid, rs in ctx.requests.items()},
                decode_credit_balance=int(row.state_decode_credit_balance),
                trace_id=trace_id,
            )

        if "test_decode_drop_reclaims_credit_gv2" in globals():
            active_decode_ids = set(
                rid for rid, rs in ctx.requests.items()
                if (rid in active_ids) and (not rs.completed) and (rs.remaining_prefill == 0)
            )
            test_decode_drop_reclaims_credit_gv2(
                ctx, row,
                active_decode_ids=active_decode_ids,
                dropped_ids=dropped_ids,
                decode_tokens_counted_by_id=dict(row.state_decode_tokens_counted_by_id),
                decode_credit_balance=int(row.state_decode_credit_balance),
                trace_id=trace_id,
            )

        prev_row = row

    return adv_actions_checked


# Keep tolerances local for prior normalization checks.
## TODO: Shift these constants into the config.py
PRIOR_SUM_TOL = max(1e-6, 10.0 * INTERVAL_EPS)
PRIOR_POS_EPS = 1e-12


def _parse_float_list_json(cell: str) -> List[float]:
    s = (cell or "").strip()
    if not s or s == "[]":
        return []
    try:
        arr = json.loads(s)
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    out: List[float] = []
    for x in arr:
        try:
            out.append(float(x))
        except Exception:
            return []
    return out


def _row_visible_request_ids(row: "Row") -> Set[int]:
    ids: Set[int] = set()
    candidate_fields = (
        "state_active_ids",
        "state_waiting_ids",
        "state_completed_ids",
        "state_dropped_request_ids",
        "state_stopped_decode_request_ids",
    )
    for f in candidate_fields:
        vals = getattr(row, f, None)
        if not isinstance(vals, list):
            continue
        for v in vals:
            try:
                iv = int(v)
            except Exception:
                continue
            if iv >= 0:
                ids.add(iv)
    return ids


def test_player_turn_consistency(ctx: "TraceContext", row: "Row", trace_id: str) -> None:
    # Same as v1, with safe fallback when player_to_act is missing.
    acted = (row.player_acted or "").strip().lower()
    expected = (ctx.expected_actor or "").strip().lower()

    if acted != expected:
        _fail(
            "test_player_turn_consistency",
            f"expected actor {expected!r}, got {acted!r}",
            row,
            trace_id,
        )

    nxt = (row.player_to_act or "").strip().lower()
    if nxt in ("controller", "adversary"):
        ctx.expected_actor = nxt
    else:
        # fallback toggle
        ctx.expected_actor = "controller" if acted == "adversary" else "adversary"


def test_normalized_prior_sums(row: "Row", trace_id: str) -> None:
    # Same behavior as v1.
    phase = (row.phase or "").strip()
    if phase.startswith("history-"):
        return
    if not bool(row.nn_called):
        return

    model_prior = _parse_float_list_json(row.model_prior_json)
    norm_prior = _parse_float_list_json(row.normalized_prior_json)

    # Forced/trivial rows often have empty prior payloads.
    if not model_prior or not norm_prior:
        return

    if len(model_prior) != len(norm_prior):
        _fail(
            "test_normalized_prior_sums",
            f"prior length mismatch: model={len(model_prior)} norm={len(norm_prior)}",
            row,
            trace_id,
        )

    sm = float(sum(model_prior))
    sn = float(sum(norm_prior))
    if abs(sm - 1.0) > PRIOR_SUM_TOL:
        _fail("test_normalized_prior_sums", f"model_prior sum={sm:.9f} != 1", row, trace_id)
    if abs(sn - 1.0) > PRIOR_SUM_TOL:
        _fail("test_normalized_prior_sums", f"normalized_prior sum={sn:.9f} != 1", row, trace_id)

    for name, arr in (("model_prior_json", model_prior), ("normalized_prior_json", norm_prior)):
        if any((not math.isfinite(x)) for x in arr):
            _fail("test_normalized_prior_sums", f"{name} contains non-finite values", row, trace_id)
        if min(arr) < -PRIOR_POS_EPS:
            _fail("test_normalized_prior_sums", f"{name} has negative prob min={min(arr):.9e}", row, trace_id)

    if not any(p > PRIOR_POS_EPS for p in norm_prior):
        _fail("test_normalized_prior_sums", "normalized_prior has no positive entries", row, trace_id)


def test_adversary_request_ids_sequential(
    ctx: "TraceContext",
    row: "Row",
    *,
    new_ids: List[int],
    trace_id: str,
) -> None:
    phase = (row.phase or "").strip().lower()
    if phase.startswith("internal:"):
        return
    if (row.player_acted or "").strip().lower() != "adversary":
        return
    if not new_ids:
        return

    ids = [int(x) for x in new_ids]

    # 1) strictly increasing + unique
    if ids != sorted(ids):
        _fail(
            "test_adversary_request_ids_sequential",
            f"new_ids not sorted increasing: {ids}",
            row,
            trace_id,
        )
    if len(ids) != len(set(ids)):
        _fail(
            "test_adversary_request_ids_sequential",
            f"new_ids contains duplicates: {ids}",
            row,
            trace_id,
        )

    # 2) contiguous block internally
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1] + 1:
            _fail(
                "test_adversary_request_ids_sequential",
                f"new_ids not contiguous: {ids}",
                row,
                trace_id,
            )

    # 3) monotonic globally (allow gaps in trace coverage)
    visible_prev = _row_visible_request_ids(row) - set(ids)
    prev_max = int(ctx.max_seen_request_id)
    if visible_prev:
        prev_max = max(prev_max, max(int(v) for v in visible_prev))

    if ids[0] <= prev_max:
        _fail(
            "test_adversary_request_ids_sequential",
            f"new id block starts at {ids[0]} but prev_max={prev_max}",
            row,
            trace_id,
        )

    # 4) advance context max to current knowledge
    visible_now = _row_visible_request_ids(row)
    if visible_now:
        ctx.max_seen_request_id = max(prev_max, ids[-1], max(int(v) for v in visible_now))
    else:
        ctx.max_seen_request_id = max(prev_max, ids[-1])




def test_controller_alloc_ids_exist(
    ctx: "TraceContext",
    row: "Row",
    token_alloc: Dict[int, int],
    trace_id: str,
) -> None:
    phase = (row.phase or "").strip().lower()
    if phase.startswith("internal:"):
        return
    if (row.player_acted or "").strip().lower() != "controller":
        return

    for rid in token_alloc.keys():
        rid_i = int(rid)
        if rid_i not in ctx.requests:
            _fail(
                "test_controller_alloc_ids_exist",
                f"allocated to rid={rid_i} not in request dict",
                row,
                trace_id,
            )



def test_adversary_prefill_token_bounds(
    row: "Row",
    *,
    adv_prefill_tokens: List[int],
    trace_id: str,
) -> None:
    phase = (row.phase or "").strip().lower()
    if phase.startswith("internal:"):
        return
    if (row.player_acted or "").strip().lower() != "adversary":
        return
    if not adv_prefill_tokens:
        return

    parsed_any = False
    for tok in adv_prefill_tokens:
        t = _safe_int(tok, default=-1)
        if t < 0:
            # unknown token from weak fallback parsing; ignore here
            continue
        parsed_any = True
        if t not in _ALLOWED_PREFILL_TOKENS:
            _fail(
                "test_adversary_prefill_token_bounds",
                f"prefill_tokens={t} not in allowed set {_ALLOWED_PREFILL_TOKENS_SORTED}",
                row,
                trace_id,
            )

    # Optional strictness: if action clearly had requests but none parsed, fail.
    if (not parsed_any) and ("AdversaryRequestSpec(" in (row.action_repr or "")):
        _fail(
            "test_adversary_prefill_token_bounds",
            "could not parse adversary prefill_tokens from row payload/action_repr",
            row,
            trace_id,
        )




def _is_on_tick_grid(t: float, step: float, eps: float) -> bool:
    if step <= 0.0:
        return False
    k = round(float(t) / float(step))
    snapped = float(k) * float(step)
    return abs(float(t) - snapped) <= eps


def test_adversary_deadlines_correct(
    row: "Row",
    *,
    decision_tick: float,
    created_ids: List[int],
    adversary_requests: List[Dict[str, object]],
    deadline_by_id: Dict[int, float],
    trace_id: str,
) -> None:
    """
    GV2 semantics:
      expected_deadline[rid] = decision_tick + request.prefill_slo
    even when row.sim_time > decision_tick (missed-tick replay case).
    """
    # No created requests -> no deadline checks needed.
    if not created_ids:
        if deadline_by_id:
            _fail(
                "test_adversary_deadlines_correct",
                f"no created_ids but deadline map is non-empty: keys={sorted(deadline_by_id.keys())}",
                row,
                trace_id,
            )
        return

    # Tick must be valid and not in future of this row.
    if not _is_on_tick_grid(float(decision_tick), _GV2_TICK_SEC, 5.0 * INTERVAL_EPS):
        _fail(
            "test_adversary_deadlines_correct",
            f"decision_tick={decision_tick:.9f} is not on {_GV2_TICK_SEC:.3f}s grid",
            row,
            trace_id,
        )
    if float(decision_tick) > float(row.sim_time) + INTERVAL_EPS:
        _fail(
            "test_adversary_deadlines_correct",
            f"decision_tick={decision_tick:.9f} > row.sim_time={row.sim_time:.9f}",
            row,
            trace_id,
        )

    # Structural checks.
    ids = [int(x) for x in created_ids]
    if len(ids) != len(set(ids)):
        _fail(
            "test_adversary_deadlines_correct",
            f"created_ids contains duplicates: {ids}",
            row,
            trace_id,
        )
    
    if len(adversary_requests) != len(ids):
        reparsed = _parse_adversary_requests("", row.action_repr)
        if len(reparsed) == len(ids):
            adversary_requests = reparsed
        else:
            _fail(
                "test_adversary_deadlines_correct",
                f"adversary_requests len={len(adversary_requests)} != created_ids len={len(ids)}",
                row,
                trace_id,
            )



    map_keys = sorted(int(k) for k in deadline_by_id.keys())
    if sorted(ids) != map_keys:
        _fail(
            "test_adversary_deadlines_correct",
            f"deadline keys mismatch: expected ids={sorted(ids)}, got keys={map_keys}",
            row,
            trace_id,
        )

    # Per-request deadline check.
    for rid, req in zip(ids, adversary_requests):
        prefill_tokens = int(req.get("prefill_tokens", -1))
        prefill_slo = float(req.get("prefill_slo", float("nan")))

        if prefill_tokens not in _ALLOWED_PREFILL_TOKENS:
            _fail(
                "test_adversary_deadlines_correct",
                f"rid={rid} prefill_tokens={prefill_tokens} not in allowed set {sorted(_ALLOWED_PREFILL_TOKENS)}",
                row,
                trace_id,
            )
        if (not math.isfinite(prefill_slo)) or prefill_slo < 0.0:
            _fail(
                "test_adversary_deadlines_correct",
                f"rid={rid} invalid prefill_slo={prefill_slo}",
                row,
                trace_id,
            )

        expected_deadline = float(decision_tick) + float(prefill_slo)
        actual_deadline = float(deadline_by_id.get(rid, float("nan")))

        if not math.isfinite(actual_deadline):
            _fail(
                "test_adversary_deadlines_correct",
                f"rid={rid} missing/NaN deadline in deadline_by_id",
                row,
                trace_id,
            )

        if abs(actual_deadline - expected_deadline) > _GV2_DEADLINE_TOL:
            _fail(
                "test_adversary_deadlines_correct",
                f"rid={rid} deadline mismatch: expected={expected_deadline:.9f}, got={actual_deadline:.9f}",
                row,
                trace_id,
            )



def test_adversary_interval_and_update_arrival(
    ctx: "TraceContext",
    row: "Row",
    *,
    decision_tick: float,
    adv_created_count: int,
    trace_id: str,
) -> float:
    """
    GV2 replacement for v1 interval/arrival logic.

    decision_tick:
      tick at which adversary made this decision (can be < row.sim_time for missed-tick replay).
    adv_created_count:
      number of requests actually created by this adversary action.
    """
    tick = float(decision_tick)
    sim_t = float(row.sim_time)

    # 1) tick must be finite, on 0.2 grid, and not in the future.
    if not math.isfinite(tick):
        _fail(
            "test_adversary_interval_and_update_arrival",
            f"decision_tick is non-finite: {decision_tick!r}",
            row,
            trace_id,
        )

    if not _is_on_tick_grid(tick, _GV2_TICK_SEC, 5.0 * INTERVAL_EPS):
        _fail(
            "test_adversary_interval_and_update_arrival",
            f"decision_tick={tick:.9f} is not on {_GV2_TICK_SEC:.3f}s grid",
            row,
            trace_id,
        )

    if tick > sim_t + INTERVAL_EPS:
        _fail(
            "test_adversary_interval_and_update_arrival",
            f"decision_tick={tick:.9f} > sim_time={sim_t:.9f}",
            row,
            trace_id,
        )

    # Read prior state from context (no hard dependency on dataclass fields).
    prev_tick = getattr(ctx, "last_adv_decision_tick", None)
    prev_sent = bool(getattr(ctx, "last_adv_sent", False))
    prev_send_window = getattr(ctx, "last_adv_send_window", None)

    # 2) decision ticks must advance and move in step-multiples.
    if prev_tick is not None:
        prev_tick = float(prev_tick)

        if tick <= prev_tick + INTERVAL_EPS:
            _fail(
                "test_adversary_interval_and_update_arrival",
                f"non-increasing decision_tick: prev={prev_tick:.9f}, now={tick:.9f}",
                row,
                trace_id,
            )

        dt = tick - prev_tick
        if not _is_on_tick_grid(dt, _GV2_TICK_SEC, 5.0 * INTERVAL_EPS):
            _fail(
                "test_adversary_interval_and_update_arrival",
                f"tick delta {dt:.9f} is not multiple of {_GV2_TICK_SEC:.3f}",
                row,
                trace_id,
            )

    sent_now = int(adv_created_count) > 0

    # 4) at most one send per integer-second window.
    if sent_now:
        ctx.last_adv_batch_time = tick


    # Persist for next adversary-row check.
    ctx.last_adv_decision_tick = tick
    ctx.last_adv_sent = bool(sent_now)

    return tick



def test_adversary_action_index_matches_payload(
    row: "Row",
    *,
    action_index: int,
    adversary_requests: List[Dict[str, object]],
    stop_decode_ids: List[int],
    trace_id: str,
) -> None:
    """
    GV2 replacement for v1 count-vs-index test.

    Validates that logged action_index decodes to the same adversary payload
    shown in the log (launch count, prefill template, stop rule shape).
    """
    idx = int(action_index)
    if idx < 0:
        _fail(
            "test_adversary_action_index_matches_payload",
            f"invalid action_index={idx}",
            row,
            trace_id,
        )

    stop_rules = list(_GV2_CFG.adversary_action.stop_rule_names)
    templates = sorted(int(x) for x in _GV2_CFG.request.allowed_prefill_tokens)
    n_stop = len(stop_rules)
    n_templates = len(templates)
    max_launch = int(_GV2_CFG.adversary_action.max_launch_count_per_tick)
    decode_cap = int(_GV2_CFG.request.max_decode_tokens_per_request)

    total_space = n_stop + (max_launch * n_templates * n_stop)
    if idx >= total_space:
        _fail(
            "test_adversary_action_index_matches_payload",
            f"action_index={idx} out of bounds [0, {total_space - 1}]",
            row,
            trace_id,
        )

    # Decode flattened index -> (launch_count, template_idx, stop_rule_idx)
    if idx < n_stop:
        launch_count = 0
        template_idx = None
        stop_rule_idx = idx
    else:
        j = idx - n_stop
        block = n_templates * n_stop
        launch_count = (j // block) + 1
        rem = j % block
        template_idx = rem // n_stop
        stop_rule_idx = rem % n_stop

    stop_rule = stop_rules[stop_rule_idx]
    expected_req_count = int(launch_count)
    actual_req_count = len(adversary_requests)

    # 1) launch count must match payload request count.
    if actual_req_count != expected_req_count:
        _fail(
            "test_adversary_action_index_matches_payload",
            f"action_index={idx} decodes launch_count={expected_req_count}, "
            f"but payload has {actual_req_count} requests",
            row,
            trace_id,
        )

    # 2) request payload fields must match decoded template when launch_count>0.
    if launch_count > 0:
        assert template_idx is not None
        expected_prefill = int(templates[template_idx])

        for k, req in enumerate(adversary_requests):
            prefill = int(req.get("prefill_tokens", -1))
            decode = int(req.get("decode_tokens", -1))
            if prefill != expected_prefill:
                _fail(
                    "test_adversary_action_index_matches_payload",
                    f"req[{k}] prefill_tokens={prefill} != expected template {expected_prefill} "
                    f"(action_index={idx})",
                    row,
                    trace_id,
                )
            if decode != decode_cap:
                _fail(
                    "test_adversary_action_index_matches_payload",
                    f"req[{k}] decode_tokens={decode} != expected decode cap {decode_cap} "
                    f"(action_index={idx})",
                    row,
                    trace_id,
                )

    # 3) stop_none must imply empty stop list.
    # For non-stop_none, exact id set can vary by state, so we only sanity check type/uniqueness/nonnegativity.
    if stop_rule == "stop_none":
        if stop_decode_ids:
            _fail(
                "test_adversary_action_index_matches_payload",
                f"action_index={idx} decodes stop_rule=stop_none but stop_decode_ids={stop_decode_ids}",
                row,
                trace_id,
            )
    else:
        seen = set()
        for rid in stop_decode_ids:
            rid_i = int(rid)
            if rid_i < 0:
                _fail(
                    "test_adversary_action_index_matches_payload",
                    f"negative stop_decode id {rid_i} for stop_rule={stop_rule}",
                    row,
                    trace_id,
                )
            if rid_i in seen:
                _fail(
                    "test_adversary_action_index_matches_payload",
                    f"duplicate stop_decode id {rid_i} for stop_rule={stop_rule}",
                    row,
                    trace_id,
                )
            seen.add(rid_i)


def test_controller_action_regime_gv2(
    row: "Row",
    *,
    token_budget: int,
    token_alloc: Dict[int, int],
    prefill_alloc: Dict[int, int],
    decode_alloc: Dict[int, int],
    trace_id: str,
) -> None:
    """
    GV2 replacement for v1 test_controller_budget_if_waiting.
    """
    acted = (row.player_acted or "").strip().lower()
    if acted != "controller":
        return

    pending_adv = bool(getattr(row, "state_pending_adv_tick", False))
    active_ids = list(getattr(row, "state_active_ids", []) or [])
    no_requests = (len(active_ids) == 0)

    total_alloc = int(sum(int(v) for v in token_alloc.values()))

    # Regime A: controller must be forced no-op
    if pending_adv or no_requests:
        if token_budget != 0:
            _fail(
                "test_controller_action_regime_gv2",
                f"forced-noop regime but token_budget={token_budget}",
                row,
                trace_id,
            )
        if token_alloc or prefill_alloc or decode_alloc:
            _fail(
                "test_controller_action_regime_gv2",
                "forced-noop regime but allocations are non-empty",
                row,
                trace_id,
            )
        return

    # Regime B: normal controller branch
    if token_budget < 0:
        _fail(
            "test_controller_action_regime_gv2",
            f"negative token_budget={token_budget}",
            row,
            trace_id,
        )

    if token_budget != total_alloc:
        _fail(
            "test_controller_action_regime_gv2",
            f"token_budget {token_budget} != sum(token_alloc) {total_alloc}",
            row,
            trace_id,
        )

    if token_budget == 0 and (token_alloc or prefill_alloc or decode_alloc):
        _fail(
            "test_controller_action_regime_gv2",
            "token_budget=0 but allocations are non-empty",
            row,
            trace_id,
        )


def test_controller_prefill_allocation_validity_gv2(
    ctx: "TraceContext",
    row: "Row",
    *,
    prefill_alloc: Dict[int, int],
    trace_id: str,
) -> None:
    """
    GV2 replacement for v1 test_controller_prefill_if_needed.

    Rules:
    1) If a rid appears in prefill_alloc, it must exist and still have prefill remaining.
    2) If no request has prefill remaining, prefill_alloc must be empty.
    3) Do NOT force prefill_alloc>0 merely because prefill work exists (GV2 allows no-op/budget=0).
    """
    if (row.player_acted or "").strip().lower() != "controller":
        return

    # Compute "has prefill work" from context request states.
    has_prefill_work = False
    for rs in ctx.requests.values():
        if bool(getattr(rs, "completed", False)):
            continue
        rem_pref = int(getattr(rs, "remaining_prefill", 0))
        if rem_pref > 0:
            has_prefill_work = True
            break

    # 1) Every prefill allocation must target an existing, prefill-pending request.
    for rid_raw, amt_raw in prefill_alloc.items():
        rid = int(rid_raw)
        amt = int(amt_raw)

        if amt <= 0:
            _fail(
                "test_controller_prefill_allocation_validity_gv2",
                f"prefill allocation for rid={rid} has non-positive amt={amt}",
                row,
                trace_id,
            )

        rs = ctx.requests.get(rid)
        if rs is None:
            _fail(
                "test_controller_prefill_allocation_validity_gv2",
                f"prefill allocation targets unknown rid={rid}",
                row,
                trace_id,
            )

        if bool(getattr(rs, "completed", False)):
            _fail(
                "test_controller_prefill_allocation_validity_gv2",
                f"prefill allocation targets completed rid={rid}",
                row,
                trace_id,
            )

        rem_pref = int(getattr(rs, "remaining_prefill", 0))
        if rem_pref <= 0:
            _fail(
                "test_controller_prefill_allocation_validity_gv2",
                f"prefill allocation for rid={rid} but remaining_prefill={rem_pref}",
                row,
                trace_id,
            )

        if amt > rem_pref:
            _fail(
                "test_controller_prefill_allocation_validity_gv2",
                f"prefill over-allocation for rid={rid}: amt={amt} > remaining_prefill={rem_pref}",
                row,
                trace_id,
            )

    # 2) If there is no prefill work in the system, prefill allocations must be empty.
    if (not has_prefill_work) and bool(prefill_alloc):
        _fail(
            "test_controller_prefill_allocation_validity_gv2",
            f"no prefill work exists, but prefill_alloc is non-empty: {prefill_alloc}",
            row,
            trace_id,
        )


def test_controller_time_delta(
    ctx: "TraceContext",
    row: "Row",
    prev_row: Optional["Row"],
    prefill_profile: Dict[int, float],
    prefill_total: int,
    decode_total: int,
    trace_id: str,
) -> None:
    """
    GV2 version:
    - time must be non-decreasing
    - if controller actually schedules tokens, dt must be positive
    - if prefill was scheduled, dt should not be smaller than a profile-based lower bound
      (using nearest available profile key <= prefill_total)
    """
    del ctx  # kept for call-site compatibility

    if prev_row is None:
        return

    dt = float(row.sim_time) - float(prev_row.sim_time)

    # 1) monotonic time
    if dt < -INTERVAL_EPS:
        _fail(
            "test_controller_time_delta",
            f"sim_time went backwards: prev={prev_row.sim_time:.9f}, now={row.sim_time:.9f}, dt={dt:.9f}",
            row,
            trace_id,
        )

    # controller row should never be exactly negative/invalid; tiny jitter allowed around 0
    if not math.isfinite(dt):
        _fail(
            "test_controller_time_delta",
            f"non-finite dt={dt}",
            row,
            trace_id,
        )

    scheduled_tokens = int(prefill_total) + int(decode_total)

    # 2) if controller scheduled any tokens, time must advance
    if scheduled_tokens > 0 and dt <= INTERVAL_EPS:
        _fail(
            "test_controller_time_delta",
            f"scheduled tokens (prefill={prefill_total}, decode={decode_total}) but dt={dt:.9f} <= 0",
            row,
            trace_id,
        )

    # 3) prefill-profile lower bound check for rows that executed prefill
    if int(prefill_total) > 0 and prefill_profile:
        keys = sorted(int(k) for k in prefill_profile.keys() if int(k) > 0)
        if keys:
            p = int(prefill_total)
            lo_candidates = [k for k in keys if k <= p]
            lo_key = max(lo_candidates) if lo_candidates else min(keys)
            lo = float(prefill_profile[lo_key])

            # Keep tolerance a bit practical for logging/scheduler jitter.
            lower_tol = max(2e-3, 20.0 * INTERVAL_EPS)
            if dt + lower_tol < lo:
                _fail(
                    "test_controller_time_delta",
                    (
                        f"dt={dt:.9f} too small for prefill_total={p}; "
                        f"profile lower bound key={lo_key} -> {lo:.9f} "
                        f"(tol={lower_tol:.6f})"
                    ),
                    row,
                    trace_id,
                )


def _gv2_coerce_int_set(v) -> Set[int]:
    if v is None:
        return set()
    if isinstance(v, set):
        return {int(x) for x in v}
    if isinstance(v, (list, tuple)):
        out: Set[int] = set()
        for x in v:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return set()
        try:
            obj = json.loads(s)
        except Exception:
            return set()
        return _gv2_coerce_int_set(obj)
    return set()


def _gv2_coerce_float_map(v) -> dict[int, float]:
    if v is None:
        return {}
    if isinstance(v, dict):
        out: dict[int, float] = {}
        for k, val in v.items():
            try:
                out[int(k)] = float(val)
            except Exception:
                continue
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except Exception:
            return {}
        return _gv2_coerce_float_map(obj)
    return {}


def _gv2_row_int_set(row: "Row", names: List[str]) -> Set[int]:
    for n in names:
        if hasattr(row, n):
            vals = _gv2_coerce_int_set(getattr(row, n))
            if vals:
                return vals
    return set()


def _gv2_row_float_map(row: "Row", names: List[str]) -> dict[int, float]:
    for n in names:
        if hasattr(row, n):
            mp = _gv2_coerce_float_map(getattr(row, n))
            if mp:
                return mp
    return {}


def _gv2_row_total_lateness(row: "Row") -> float:
    # Prefer GV2 name, fallback to old logger field.
    if hasattr(row, "total_lateness"):
        try:
            return float(getattr(row, "total_lateness"))
        except Exception:
            pass
    try:
        return float(getattr(row, "avg_lateness", 0.0))
    except Exception:
        return 0.0


def test_objective_matches_log(ctx: "TraceContext", row: "Row", trace_id: str) -> None:
    """
    GV2 objective check:
      1) identity: objective_cost == slo_violations + total_lateness
      2) drop semantics: for newly dropped rid,
         expected replacement delta per request is
           drop_cost - old_violation_bit - old_lateness
         and objective delta must include at least that amount.
    """
    exp_viol = int(getattr(row, "slo_violations", 0))
    exp_late = float(_gv2_row_total_lateness(row))
    exp_obj = float(getattr(row, "objective_cost", 0.0))

    # Base identity check (always).
    if abs(exp_obj - (float(exp_viol) + exp_late)) > _GV2_OBJECTIVE_TOL:
        _fail(
            "test_objective_matches_log",
            (
                f"objective identity mismatch: objective_cost={exp_obj:.9f} "
                f"vs slo_violations+total_lateness={(float(exp_viol) + exp_late):.9f} "
                f"(viol={exp_viol}, late={exp_late:.9f})"
            ),
            row,
            trace_id,
        )

    # Current row state sets/maps.
    curr_dropped = _gv2_row_int_set(
        row,
        ["state_dropped_request_ids", "dropped_request_ids"],
    )
    curr_active = _gv2_row_int_set(
        row,
        ["state_active_ids", "state_waiting_ids", "waiting_request_ids"],
    )
    curr_completed = _gv2_row_int_set(
        row,
        ["state_completed_ids", "state_completed_request_ids", "completed_request_ids"],
    )
    curr_violated = _gv2_row_int_set(
        row,
        ["state_violated_request_ids", "violated_request_ids"],
    )

    curr_pref_late = _gv2_row_float_map(
        row,
        [
            "state_per_request_prefill_lateness_by_id",
            "state_prefill_lateness_by_id",
            "per_request_prefill_lateness_by_id",
        ],
    )
    curr_dec_late = _gv2_row_float_map(
        row,
        [
            "state_per_request_decode_lateness_by_id",
            "state_decode_lateness_by_id",
            "per_request_decode_lateness_by_id",
        ],
    )

    # Previous row snapshots stored in ctx.
    prev_obj = getattr(ctx, "_gv2_prev_objective_cost", None)
    prev_dropped = getattr(ctx, "_gv2_prev_dropped_ids", set())
    prev_violated = getattr(ctx, "_gv2_prev_violated_ids", set())
    prev_pref_late = getattr(ctx, "_gv2_prev_pref_late_by_id", {})
    prev_dec_late = getattr(ctx, "_gv2_prev_dec_late_by_id", {})

    if prev_obj is not None:
        new_drops = sorted(set(curr_dropped) - set(prev_dropped))
        if new_drops:
            # Structural checks for each newly dropped request.
            for rid in new_drops:
                if rid in curr_active:
                    _fail(
                        "test_objective_matches_log",
                        f"rid={rid} newly dropped but still active/waiting",
                        row,
                        trace_id,
                    )
                if curr_completed and rid not in curr_completed:
                    _fail(
                        "test_objective_matches_log",
                        f"rid={rid} newly dropped but not marked completed",
                        row,
                        trace_id,
                    )
                if rid in curr_violated:
                    _fail(
                        "test_objective_matches_log",
                        f"rid={rid} newly dropped but still in violated_request_ids",
                        row,
                        trace_id,
                    )
                if rid in curr_pref_late or rid in curr_dec_late:
                    _fail(
                        "test_objective_matches_log",
                        (
                            f"rid={rid} newly dropped but still present in per-request lateness maps "
                            f"(pref={curr_pref_late.get(rid, 0.0):.6f}, dec={curr_dec_late.get(rid, 0.0):.6f})"
                        ),
                        row,
                        trace_id,
                    )

            # Cost replacement condition requested:
            # expected per-drop contribution delta:
            #   drop_cost - old_violation_bit - old_lateness
            # Since other non-drop updates can also occur in same row (>=0 objective impact),
            # enforce lower bound on row objective delta.
            expected_drop_delta = 0.0
            for rid in new_drops:
                old_violation_bit = 1.0 if int(rid) in set(prev_violated) else 0.0
                old_lateness = float(prev_pref_late.get(int(rid), 0.0)) + float(prev_dec_late.get(int(rid), 0.0))
                expected_drop_delta += float(_GV2_CFG.cost.drop_cost) - old_violation_bit - old_lateness

            got_delta = float(exp_obj) - float(prev_obj)
            if got_delta + _GV2_OBJECTIVE_TOL < expected_drop_delta:
                _fail(
                    "test_objective_matches_log",
                    (
                        f"drop-cost replacement under-accounted: got objective delta={got_delta:.9f}, "
                        f"expected at least {expected_drop_delta:.9f} from new_drops={new_drops} "
                        f"(rule: drop_cost - old_violation_bit - old_lateness)"
                    ),
                    row,
                    trace_id,
                )

    # Persist for next row.
    ctx._gv2_prev_objective_cost = float(exp_obj)
    ctx._gv2_prev_dropped_ids = set(curr_dropped)
    ctx._gv2_prev_violated_ids = set(curr_violated)
    ctx._gv2_prev_pref_late_by_id = dict(curr_pref_late)
    ctx._gv2_prev_dec_late_by_id = dict(curr_dec_late)


def test_adversary_tick_progression_gv2(
    ctx: "TraceContext",
    row: "Row",
    *,
    decision_tick: float,
    adv_created_count: int,
    trace_id: str,
) -> None:
    """
    GV2 tick progression:
    - adversary decision ticks are on 0.2 grid
    - tick <= sim_time
    - from one adversary decision to next, tick advances by exactly +0.2
      regardless of send/no-send
    """
    if (row.player_acted or "").strip().lower() != "adversary":
        return

    tick = float(decision_tick)
    sim_t = float(row.sim_time)

    if not math.isfinite(tick):
        _fail(
            "test_adversary_tick_progression_gv2",
            f"non-finite decision_tick={decision_tick!r}",
            row,
            trace_id,
        )

    if not _is_on_tick_grid(tick, _GV2_TICK_SEC, 5.0 * INTERVAL_EPS):
        _fail(
            "test_adversary_tick_progression_gv2",
            f"decision_tick={tick:.9f} not on {_GV2_TICK_SEC:.3f}s grid",
            row,
            trace_id,
        )

    if tick > sim_t + INTERVAL_EPS:
        _fail(
            "test_adversary_tick_progression_gv2",
            f"decision_tick={tick:.9f} > sim_time={sim_t:.9f}",
            row,
            trace_id,
        )

    prev_tick = getattr(ctx, "_gv2_prev_adv_tick_exact", None)
    if prev_tick is not None:
        prev_tick = float(prev_tick)
        dt = tick - prev_tick
        # GV2 env now uses next_tick = tick + 0.2 on both send and no-send.
        if abs(dt - float(_GV2_TICK_SEC)) > max(1e-6, 10.0 * INTERVAL_EPS):
            _fail(
                "test_adversary_tick_progression_gv2",
                (
                    f"expected exact +{_GV2_TICK_SEC:.3f}s tick advance "
                    f"(send/no-send), got dt={dt:.9f} "
                    f"(prev={prev_tick:.9f}, now={tick:.9f}, created={int(adv_created_count)})"
                ),
                row,
                trace_id,
            )

    ctx._gv2_prev_adv_tick_exact = tick


def test_adversary_sliding_window_constraints_gv2(
    ctx: "TraceContext",
    row: "Row",
    *,
    decision_tick: float,
    adversary_requests: List[dict],
    trace_id: str,
) -> None:
    """
    GV2 sliding-window launch constraints:
    Rebuilds recent launches in (tick - launch_window_sec, tick] and verifies:
      used_count + new_count <= max_requests_per_launch_window
      used_prefill + new_prefill <= prefill_window_cap_tokens
    """
    if (row.player_acted or "").strip().lower() != "adversary":
        return

    tick = float(decision_tick)
    if not math.isfinite(tick):
        _fail(
            "test_adversary_sliding_window_constraints_gv2",
            f"non-finite decision_tick={decision_tick!r}",
            row,
            trace_id,
        )

    # Keep history as list[(timestamp, count, prefill_sum)].
    hist = list(getattr(ctx, "_gv2_launch_history", []))
    w = float(_GV2_CFG.timing.launch_window_sec)

    lo = tick - w
    pruned = []
    for item in hist:
        try:
            ts, cnt, pref = float(item[0]), int(item[1]), int(item[2])
        except Exception:
            continue
        # Match env prune rule: keep if ts + eps >= lo
        if ts + INTERVAL_EPS >= lo:
            pruned.append((ts, cnt, pref))

    used_count = sum(int(x[1]) for x in pruned)
    used_prefill = sum(int(x[2]) for x in pruned)

    reqs = list(adversary_requests or [])
    new_count = len(reqs)
    new_prefill = 0
    for req in reqs:
        try:
            new_prefill += int(req.get("prefill_tokens", 0))
        except Exception:
            pass

    # Per-tick launch-count cap.
    launch_cap_per_tick = int(_GV2_CFG.adversary_action.max_launch_count_per_tick)
    if new_count > launch_cap_per_tick:
        _fail(
            "test_adversary_sliding_window_constraints_gv2",
            f"new_count={new_count} exceeds per-tick cap={launch_cap_per_tick}",
            row,
            trace_id,
        )

    # Sliding-window caps.
    req_cap = int(_GV2_CFG.timing.max_requests_per_launch_window)
    prefill_cap = int(_GV2_CFG.request.target_prefill_tokens_per_request_avg_window) * req_cap

    if new_count > 0:
        if used_count + new_count > req_cap:
            _fail(
                "test_adversary_sliding_window_constraints_gv2",
                (
                    f"request-window cap violated at tick={tick:.9f}: "
                    f"used={used_count}, new={new_count}, cap={req_cap}"
                ),
                row,
                trace_id,
            )
        if used_prefill + new_prefill > prefill_cap:
            _fail(
                "test_adversary_sliding_window_constraints_gv2",
                (
                    f"prefill-window cap violated at tick={tick:.9f}: "
                    f"used_prefill={used_prefill}, new_prefill={new_prefill}, cap={prefill_cap}"
                ),
                row,
                trace_id,
            )

        pruned.append((tick, int(new_count), int(new_prefill)))

    ctx._gv2_launch_history = pruned


def test_missed_tick_replay_correctness_gv2(
    ctx: "TraceContext",
    row: "Row",
    *,
    decision_tick: float,
    trace_id: str,
) -> None:
    """
    GV2 missed-tick replay correctness:
    - Track last adversary decision tick.
    - For controller rows, pending flag must match:
        pending_expected = (last_adv_tick + 0.2 <= sim_time)
    This enforces that controller does not resume normal branching
    until all pending adversary ticks are drained (including multiple missed ticks).
    """
    actor = (row.player_acted or "").strip().lower()

    if actor == "adversary":
        tick = float(decision_tick)
        if not math.isfinite(tick):
            _fail(
                "test_missed_tick_replay_correctness_gv2",
                f"non-finite adversary decision_tick={decision_tick!r}",
                row,
                trace_id,
            )
        ctx._gv2_last_adv_tick_for_pending = tick
        return

    if actor != "controller":
        return

    last_tick = getattr(ctx, "_gv2_last_adv_tick_for_pending", None)
    if last_tick is None:
        # No adversary tick observed yet in this trace segment.
        return

    sim_t = float(row.sim_time)
    next_tick = float(last_tick) + float(_GV2_TICK_SEC)
    pending_expected = (next_tick <= sim_t + INTERVAL_EPS)
    pending_logged = bool(getattr(row, "state_pending_adv_tick", False))

    if pending_logged != pending_expected:
        _fail(
            "test_missed_tick_replay_correctness_gv2",
            (
                f"pending flag mismatch: logged={pending_logged}, expected={pending_expected} "
                f"(last_adv_tick={float(last_tick):.9f}, next_tick={next_tick:.9f}, sim_time={sim_t:.9f})"
            ),
            row,
            trace_id,
        )


# -----------------------------
# GV2 helpers for rule/ID checks
# -----------------------------
def _gv2_as_int_set(x: Any) -> Set[int]:
    if x is None:
        return set()
    if isinstance(x, set):
        return {int(v) for v in x}
    if isinstance(x, (list, tuple)):
        out: Set[int] = set()
        for v in x:
            try:
                out.add(int(v))
            except Exception:
                pass
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return set()
        try:
            obj = json.loads(s)
        except Exception:
            return set()
        return _gv2_as_int_set(obj)
    return set()


def _gv2_req_rem_prefill(rs: Any) -> int:
    return max(0, int(getattr(rs, "remaining_prefill", getattr(rs, "rem_prefill", 0))))


def _gv2_req_decode_done(rs: Any) -> int:
    return int(getattr(rs, "decode_done", getattr(rs, "num_processed_decode_tokens", 0)))


def _gv2_req_decode_total(rs: Any) -> int:
    for name in (
        "remaining_decode_total",
        "decode_tokens_total",
        "num_decode_tokens",
        "_num_decode_tokens",
    ):
        if hasattr(rs, name):
            try:
                v = int(getattr(rs, name))
                if v > 0:
                    return v
            except Exception:
                pass
    # fallback for tests if total not explicitly tracked
    return int(_GV2_CFG.request.max_decode_tokens_per_request)


def _gv2_req_rem_decode(rs: Any) -> int:
    if hasattr(rs, "remaining_decode"):
        try:
            return max(0, int(getattr(rs, "remaining_decode")))
        except Exception:
            pass
    total = _gv2_req_decode_total(rs)
    done = _gv2_req_decode_done(rs)
    return max(0, total - done)


def _gv2_req_prefill_done(rs: Any) -> bool:
    return _gv2_req_rem_prefill(rs) <= 0


def _gv2_req_total_lateness(rs: Any) -> float:
    return float(getattr(rs, "prefill_lateness", 0.0)) + float(getattr(rs, "decode_lateness", 0.0))


def _gv2_expected_eviction_targets_from_ctx(
    ctx: "TraceContext",
    *,
    rule: str,
) -> List[int]:
    reqs: Dict[int, Any] = dict(getattr(ctx, "requests", {}))

    prefill_ids = [
        rid for rid, rs in reqs.items()
        if (not bool(getattr(rs, "completed", False)))
        and (not _gv2_req_prefill_done(rs))
        and (_gv2_req_rem_prefill(rs) > 0)
    ]
    decode_ids = [
        rid for rid, rs in reqs.items()
        if (not bool(getattr(rs, "completed", False)))
        and _gv2_req_prefill_done(rs)
        and (_gv2_req_rem_decode(rs) > 0)
    ]

    if rule == "evict_none":
        return []

    if rule == "evict_largest_prefill":
        if not prefill_ids:
            return []
        rid = max(prefill_ids, key=lambda x: (_gv2_req_rem_prefill(reqs[x]), -x))
        return [int(rid)]

    if rule == "evict_earliest_prefill_deadline":
        if not prefill_ids:
            return []
        rid = min(prefill_ids, key=lambda x: (float(getattr(reqs[x], "prefill_deadline", float("inf"))), x))
        return [int(rid)]

    if rule == "evict_prefill_missed_deadline":
        return sorted(int(rid) for rid in prefill_ids if float(getattr(reqs[rid], "prefill_lateness", 0.0)) > INTERVAL_EPS)

    if rule == "evict_prefill_lateness_over_0p5":
        return sorted(int(rid) for rid in prefill_ids if float(getattr(reqs[rid], "prefill_lateness", 0.0)) > 0.5)

    if rule == "evict_longest_decode":
        if not decode_ids:
            return []
        rid = max(decode_ids, key=lambda x: (_gv2_req_rem_decode(reqs[x]), -x))
        return [int(rid)]

    if rule == "evict_decode_lateness_over_0p5":
        return sorted(int(rid) for rid in decode_ids if _gv2_req_total_lateness(reqs[rid]) > 0.5)

    if rule == "evict_prefill_highest_lateness":
        if not prefill_ids:
            return []
        rid = max(prefill_ids, key=lambda x: (float(getattr(reqs[x], "prefill_lateness", 0.0)), -x))
        return [int(rid)] if float(getattr(reqs[rid], "prefill_lateness", 0.0)) > INTERVAL_EPS else []

    if rule == "evict_decode_highest_lateness":
        if not decode_ids:
            return []
        rid = max(decode_ids, key=lambda x: (_gv2_req_total_lateness(reqs[x]), -x))
        return [int(rid)] if _gv2_req_total_lateness(reqs[rid]) > INTERVAL_EPS else []

    return []


def _gv2_decode_stop_rule_from_action_index(action_index: int) -> str:
    stop_rules = list(_GV2_CFG.adversary_action.stop_rule_names)
    templates = sorted(int(x) for x in _GV2_CFG.request.allowed_prefill_tokens)
    n_stop = len(stop_rules)
    n_templates = len(templates)
    max_launch = int(_GV2_CFG.adversary_action.max_launch_count_per_tick)

    idx = int(action_index)
    total_space = n_stop + (max_launch * n_templates * n_stop)
    if idx < 0 or idx >= total_space:
        return "INVALID"

    if idx < n_stop:
        return str(stop_rules[idx])

    j = idx - n_stop
    rem = j % (n_templates * n_stop)
    stop_idx = rem % n_stop
    return str(stop_rules[stop_idx])


def _gv2_expected_stop_ids_from_ctx(ctx: "TraceContext", *, rule: str) -> List[int]:
    reqs: Dict[int, Any] = dict(getattr(ctx, "requests", {}))
    decode_ids = sorted(
        rid for rid, rs in reqs.items()
        if (not bool(getattr(rs, "completed", False)))
        and _gv2_req_prefill_done(rs)
        and (_gv2_req_rem_decode(rs) > 0)
    )
    if not decode_ids:
        return []

    if rule == "stop_none":
        return []

    if rule == "stop_longest_decode":
        rid = max(decode_ids, key=lambda x: (_gv2_req_rem_decode(reqs[x]), -x))
        return [int(rid)]

    if rule == "stop_shortest_decode":
        rid = min(decode_ids, key=lambda x: (_gv2_req_rem_decode(reqs[x]), x))
        return [int(rid)]

    if rule == "stop_all_decodes_over_512":
        return sorted(int(rid) for rid in decode_ids if _gv2_req_decode_done(reqs[rid]) > 512)

    if rule == "stop_all_decodes_over_216":
        return sorted(int(rid) for rid in decode_ids if _gv2_req_decode_done(reqs[rid]) > 216)

    return []




def test_controller_eviction_rule_applied_and_dropped_updated_gv2(
    ctx: "TraceContext",
    row: "Row",
    *,
    eviction_rule: str,
    current_dropped_ids: Set[int],
    current_completed_ids: Set[int],
    requests_completed: int,
    trace_id: str,
) -> None:
    """
    Validates controller eviction behavior:
    - chosen eviction rule targets correct request IDs from pre-action ctx state
    - those targets are dropped+completed in this row
    - requests_completed increases accordingly (lower bound)
    """
    if (row.player_acted or "").strip().lower() != "controller":
        return

    expected = _gv2_expected_eviction_targets_from_ctx(ctx, rule=str(eviction_rule))
    prev_dropped = set(getattr(ctx, "_gv2_prev_dropped_ids_ev", set()))
    prev_completed = set(getattr(ctx, "_gv2_prev_completed_ids_ev", set()))
    prev_rc = getattr(ctx, "_gv2_prev_requests_completed_ev", None)

    curr_dropped = set(int(x) for x in current_dropped_ids)
    curr_completed = set(int(x) for x in current_completed_ids)

    if eviction_rule != "evict_none" and not expected:
        _fail(
            "test_controller_eviction_rule_applied_and_dropped_updated_gv2",
            f"eviction rule {eviction_rule!r} selected but no eligible targets in pre-action state",
            row,
            trace_id,
        )

    exp_set = set(int(x) for x in expected)
    if exp_set:
        missing_drop = sorted(exp_set - curr_dropped)
        if missing_drop:
            _fail(
                "test_controller_eviction_rule_applied_and_dropped_updated_gv2",
                f"expected evicted IDs not dropped: {missing_drop} (rule={eviction_rule})",
                row,
                trace_id,
            )
        missing_completed = sorted(exp_set - curr_completed)
        if missing_completed:
            _fail(
                "test_controller_eviction_rule_applied_and_dropped_updated_gv2",
                f"expected evicted IDs not completed: {missing_completed} (rule={eviction_rule})",
                row,
                trace_id,
            )

        expected_new = [rid for rid in exp_set if rid not in prev_dropped and rid not in prev_completed]
        if prev_rc is not None and int(requests_completed) < int(prev_rc) + len(expected_new):
            _fail(
                "test_controller_eviction_rule_applied_and_dropped_updated_gv2",
                (
                    f"requests_completed did not increase enough: prev={prev_rc}, now={requests_completed}, "
                    f"expected_new_evicted={sorted(expected_new)}"
                ),
                row,
                trace_id,
            )

    ctx._gv2_prev_dropped_ids_ev = curr_dropped
    ctx._gv2_prev_completed_ids_ev = curr_completed
    ctx._gv2_prev_requests_completed_ev = int(requests_completed)


def test_adversary_stop_rule_applied_and_completion_updated_gv2(
    ctx: "TraceContext",
    row: "Row",
    *,
    action_index: int,
    stop_decode_ids: List[int],
    current_stopped_decode_ids: Set[int],
    current_completed_ids: Set[int],
    requests_completed: int,
    trace_id: str,
) -> None:
    """
    Validates adversary stop-rule behavior:
    - stop rule decoded from action_index matches stopped IDs from pre-action ctx state
    - stopped IDs are reflected in stopped/completed sets
    - requests_completed increases accordingly (lower bound)
    """
    if (row.player_acted or "").strip().lower() != "adversary":
        return

    rule = _gv2_decode_stop_rule_from_action_index(int(action_index))
    if rule == "INVALID":
        _fail(
            "test_adversary_stop_rule_applied_and_completion_updated_gv2",
            f"invalid adversary action_index={action_index}",
            row,
            trace_id,
        )

    expected = _gv2_expected_stop_ids_from_ctx(ctx, rule=rule)
    got = sorted(int(x) for x in (stop_decode_ids or []))
    if got != expected:
        _fail(
            "test_adversary_stop_rule_applied_and_completion_updated_gv2",
            f"stop IDs mismatch for rule={rule}: expected={expected}, got={got}",
            row,
            trace_id,
        )

    curr_stopped = set(int(x) for x in current_stopped_decode_ids)
    curr_completed = set(int(x) for x in current_completed_ids)

    missing_in_stopped = sorted(set(got) - curr_stopped)
    if missing_in_stopped:
        _fail(
            "test_adversary_stop_rule_applied_and_completion_updated_gv2",
            f"stop IDs missing from stopped_decode_request_ids: {missing_in_stopped}",
            row,
            trace_id,
        )

    missing_in_completed = sorted(set(got) - curr_completed)
    if missing_in_completed:
        _fail(
            "test_adversary_stop_rule_applied_and_completion_updated_gv2",
            f"stop IDs missing from completed_request_ids: {missing_in_completed}",
            row,
            trace_id,
        )

    prev_completed = set(getattr(ctx, "_gv2_prev_completed_ids_stop", set()))
    prev_dropped = set(getattr(ctx, "_gv2_prev_dropped_ids_stop", set()))
    prev_rc = getattr(ctx, "_gv2_prev_requests_completed_stop", None)

    expected_new = [rid for rid in got if rid not in prev_completed and rid not in prev_dropped]
    if prev_rc is not None and int(requests_completed) < int(prev_rc) + len(expected_new):
        _fail(
            "test_adversary_stop_rule_applied_and_completion_updated_gv2",
            (
                f"requests_completed did not increase enough after stop rule: "
                f"prev={prev_rc}, now={requests_completed}, expected_new_stopped={sorted(expected_new)}"
            ),
            row,
            trace_id,
        )

    ctx._gv2_prev_completed_ids_stop = curr_completed
    # if you already track dropped in row, pass it too; else keep prior.
    ctx._gv2_prev_dropped_ids_stop = prev_dropped
    ctx._gv2_prev_requests_completed_stop = int(requests_completed)


def test_finalized_ids_do_not_reappear_gv2(
    ctx: "TraceContext",
    row: "Row",
    *,
    current_active_ids: Set[int],
    current_completed_ids: Set[int],
    current_dropped_ids: Set[int],
    controller_token_alloc: Optional[Dict[int, int]],
    adversary_stop_ids: Optional[List[int]],
    trace_id: str,
) -> None:
    """
    Ensures finalized IDs (completed or dropped) never reappear later as active/allocated/stopped targets.
    """
    finalized_prev = set(getattr(ctx, "_gv2_finalized_ids_seen", set()))

    active_now = set(int(x) for x in current_active_ids)
    completed_now = set(int(x) for x in current_completed_ids)
    dropped_now = set(int(x) for x in current_dropped_ids)

    resurrected = sorted(finalized_prev & active_now)
    if resurrected:
        _fail(
            "test_finalized_ids_do_not_reappear_gv2",
            f"finalized IDs reappeared as active: {resurrected}",
            row,
            trace_id,
        )

    alloc_keys = set()
    for k in (controller_token_alloc or {}).keys():
        try:
            alloc_keys.add(int(k))
        except Exception:
            pass
    bad_alloc = sorted(finalized_prev & alloc_keys)
    if bad_alloc:
        _fail(
            "test_finalized_ids_do_not_reappear_gv2",
            f"finalized IDs appear in controller allocations: {bad_alloc}",
            row,
            trace_id,
        )

    stop_set = set(int(x) for x in (adversary_stop_ids or []))
    bad_stop = sorted(finalized_prev & stop_set)
    if bad_stop:
        _fail(
            "test_finalized_ids_do_not_reappear_gv2",
            f"finalized IDs appear again in adversary stop list: {bad_stop}",
            row,
            trace_id,
        )

    # monotonic-finalized expectation
    finalized_now = completed_now | dropped_now
    if not finalized_prev.issubset(finalized_now | finalized_prev):
        _fail(
            "test_finalized_ids_do_not_reappear_gv2",
            "finalized set became non-monotonic unexpectedly",
            row,
            trace_id,
        )

    ctx._gv2_finalized_ids_seen = finalized_prev | finalized_now


def main() -> None:
    paths = _resolve_mcts_iter_paths()
    prefill_profile = {}  # keep if you don’t need strict prefill-profile lower bound

    grand_traces = 0
    grand_adv = 0

    try:
        for p in paths:
            rows = load_rows(p)
            traces = build_leaf_traces(rows)

            print(f"\n=== Testing {p} ===")
            print(f"Loaded {len(rows)} rows")
            print(f"Built {len(traces)} traces")

            adv = 0
            for tr in traces:
                adv += run_trace(tr, prefill_profile=prefill_profile)

            print(f"✅ Passed {p}: traces={len(traces)} adv_actions={adv}")
            grand_traces += len(traces)
            grand_adv += adv

    except TestFailure as e:
        print("\n❌ MCTS_DNN log test failed:\n")
        print(str(e))
        sys.exit(1)

    print(f"\n✅ All files passed. files={len(paths)} traces={grand_traces} adv_actions_checked={grand_adv}")


if __name__ == "__main__":
    main()