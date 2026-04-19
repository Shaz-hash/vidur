# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
GV2 infer.py

Builds ModelInputs with split DeepSets:
- prefill_req_features: [1, N_PREFILL_REQ, D_PREFILL_REQ]
- decode_req_features : [1, N_DECODE_REQ,  D_DECODE_REQ]
- global_features     : [1, D_GLOBAL]

Per-request feature schema (prefill/decode both, D=5):
0) remaining_*_norm
1) age_norm
2) lateness_norm
3) slack_to_drop_norm
4) violated_bit

Global features (D_GLOBAL=11):
0) system_load
1) active_prefill_count_norm
2) active_decode_count_norm
3) total_remaining_prefill_norm
4) total_decode_generated_active_norm
5) num_violated_norm
6) prefill_num_near_drop_norm_with_lateness_0p5
7) prefill_num_near_drop_norm_with_lateness_1p5
8) decode_num_near_drop_norm_with_lateness_0p5
9) decode_num_near_drop_norm_with_lateness_1p5
10) launch_count_ewma_norm
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as Fnn

from ....environment import VidurMCTSState
from ..config import DEFAULT_GAME_V2_CONFIG
from .dnn_spec import DEFAULT_DNN_SPEC
from .types import ModelInputs


# -----------------------------
# Spec/config single source of truth
# -----------------------------
_SPEC = DEFAULT_DNN_SPEC
_CFG = DEFAULT_GAME_V2_CONFIG
F = _CFG.features

N_PREFILL_REQ: int = int(_SPEC.n_prefill_req)
D_PREFILL_REQ: int = int(_SPEC.d_prefill_req)
N_DECODE_REQ: int = int(_SPEC.n_decode_req)
D_DECODE_REQ: int = int(_SPEC.d_decode_req)
D_GLOBAL: int = int(_SPEC.d_global)

NUM_ACTIONS_CONTROLLER: int = int(_SPEC.num_actions_controller)
NUM_ACTIONS_ADVERSARY: int = int(_SPEC.num_actions_adversary)


# -----------------------------
# HELPERS
# -----------------------------





def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _build_request_lookup(simulator) -> Dict[int, object]:
    """
    Best-effort request lookup from internal scheduler structures.
    Mirrors the environment’s approach but avoids importing env internals here.
    """
    lookup: Dict[int, object] = {}

    sched = getattr(simulator, "_scheduler", None)
    if sched is None:
        return lookup

    # Global queue (if present)
    rq = getattr(sched, "_request_queue", None)
    if rq is not None:
        for req in rq:
            lookup[getattr(req, "id")] = req

    replica_schedulers = getattr(sched, "_replica_schedulers", {}) or {}
    for _, rs in replica_schedulers.items():
        waiting = getattr(rs, "_waiting_queue", None)
        if waiting is not None and hasattr(waiting, "to_list"):
            for req in waiting.to_list():
                lookup[getattr(req, "id")] = req

        for req in getattr(rs, "_running", []) or []:
            lookup[getattr(req, "id")] = req

        req_map = getattr(rs, "_requests", None)
        if isinstance(req_map, dict):
            for req in req_map.values():
                if not getattr(req, "completed", False):
                    lookup[getattr(req, "id")] = req

    return lookup

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _arrived_at(req) -> float:
    return _safe_float(getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0)), 0.0)

def _prefill_remaining(req) -> int:
    return max(0, _safe_int(getattr(req, "num_prefill_tokens", 0), 0) - _safe_int(getattr(req, "num_processed_prefill_tokens", 0), 0))

def _decode_remaining(req) -> int:
    tot = _safe_int(getattr(req, "_num_decode_tokens", getattr(req, "num_decode_tokens", 0)), 0)
    done = _safe_int(getattr(req, "num_processed_decode_tokens", 0), 0)
    return max(0, tot - done)

def _request_total_lateness(state: VidurMCTSState, req, sim_time: float) -> float:
    rid = _safe_int(getattr(req, "id", -1), -1)
    st = getattr(state, "stats", None)
    if st is None:
        return 0.0
    pref = float(getattr(st, "per_request_prefill_lateness", {}).get(rid, 0.0))
    dec = float(getattr(st, "per_request_decode_lateness", {}).get(rid, 0.0))
    if pref <= 0.0:
        dl = _prefill_deadline(req)
        if dl is not None:
            pref = max(0.0, sim_time - float(dl))
    return max(0.0, pref + dec)

def _is_violated(state: VidurMCTSState, req) -> bool:
    rid = _safe_int(getattr(req, "id", -1), -1)
    violated = set(getattr(getattr(state, "stats", None), "violated_request_ids", set()) or set())
    return rid in violated

def _iter_recent_launch_counts(state: VidurMCTSState):
    recent = list(getattr(getattr(state, "stats", None), "recent_arrivals", []) or [])
    for item in recent:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            ts = float(item[0])
            cnt = int(item[1])
            yield ts, max(0, cnt)
        elif isinstance(item, dict):
            ts = float(item.get("timestamp", item.get("time", 0.0)))
            cnt = int(item.get("count", item.get("requests", 0)))
            yield ts, max(0, cnt)
        elif isinstance(item, (int, float)):
            yield float(item), 1


def _prefill_deadline(req) -> Optional[float]:
    """
    Deadline = queued_at/arrived_at + prefill_slo_time (if present).
    Uses private fields if needed because MCTS attaches SLOs that way.
    """
    slo = getattr(req, "_prefill_slo_time", None)
    if slo is None:
        return None
    arrived = getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0))
    queued = getattr(req, "queued_at", arrived)
    return _safe_float(queued) + _safe_float(slo)


def _is_prefill_request(req) -> bool:
    return (not getattr(req, "completed", False)) and (not getattr(req, "is_prefill_complete", False))


def _is_decode_request(req) -> bool:
    if getattr(req, "completed", False):
        return False
    if not getattr(req, "is_prefill_complete", False):
        return False
    # “decode remaining” check (best effort)
    total_decode = _safe_int(getattr(req, "_num_decode_tokens", getattr(req, "num_decode_tokens", 0)), 0)
    processed_decode = _safe_int(getattr(req, "num_processed_decode_tokens", 0), 0)
    return total_decode > processed_decode


def build_action_mask(
    state: VidurMCTSState,
    player: str,
    device: torch.device,
    *,
    action_mask_fn: Optional[Callable[[VidurMCTSState, str, int, torch.device], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Action mask builder.
    - If you provide action_mask_fn, it must return a bool tensor [1, num_actions(player)].
    - Otherwise: defaults to all-True (everything valid), until deterministic action indexing is implemented.
    """
    if player == "controller":
        n = NUM_ACTIONS_CONTROLLER
    elif player == "adversary":
        n = NUM_ACTIONS_ADVERSARY
    else:
        raise ValueError(f"Unknown player={player!r}")

    if action_mask_fn is None:
        return torch.ones((1, n), dtype=torch.bool, device=device)

    mask = action_mask_fn(state, player, n, device)
    if mask.dtype != torch.bool:
        raise TypeError("action_mask_fn must return dtype=torch.bool")
    if tuple(mask.shape) != (1, n):
        raise ValueError(f"action_mask_fn returned shape {tuple(mask.shape)}, expected {(1, n)}")
    return mask

META_DECODE_CREDIT_BAL = -9_100_005


def _norm01(x: float, denom: float) -> float:
    if denom <= 0.0:
        return 0.0
    return _clip(float(x) / float(denom), 0.0, 1.0)


def _centered01(x: float, radius: float) -> float:
    if radius <= 0.0:
        return 0.5
    clipped = _clip(float(x), -float(radius), float(radius))
    return (clipped + float(radius)) / (2.0 * float(radius))


def _decode_deadline(req, state: VidurMCTSState) -> Optional[float]:
    rid = _safe_int(getattr(req, "id", -1), -1)
    st = getattr(state, "stats", None)
    if st is None:
        return None
    deadline = _safe_float(getattr(st, "decode_next_deadline_by_id", {}).get(rid, 0.0), 0.0)
    if deadline > 0.0:
        return deadline
    decode_slo = _safe_float(getattr(req, "_decode_slo_time", getattr(req, "decode_slo_time", 0.0)), 0.0)
    prefill_completed_at = _safe_float(getattr(req, "_prefill_completed_at", 0.0), 0.0)
    if prefill_completed_at > 0.0 and decode_slo > 0.0:
        return prefill_completed_at + decode_slo
    return None


def _recent_launch_summary(state: VidurMCTSState, sim_time: float) -> tuple[float, float, float]:
    launch_count = 0.0
    launch_prefill = 0.0
    ewma = 0.0
    window_sec = float(F.launch_ewma_window_sec)
    alpha = float(F.launch_ewma_alpha)
    for item in list(getattr(getattr(state, "stats", None), "recent_arrivals", []) or []):
        ts = None
        cnt = 0
        prefill = 0
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            ts = float(item[0])
            cnt = int(item[1])
            prefill = int(item[2])
        elif isinstance(item, dict):
            if "timestamp" in item:
                ts = float(item["timestamp"])
            elif "time" in item:
                ts = float(item["time"])
            cnt = int(item.get("count", item.get("requests", 0)))
            prefill = int(item.get("prefill_tokens", item.get("tokens", 0)))
        elif isinstance(item, (int, float)):
            ts = float(item)
            cnt = 1

        if ts is None:
            continue

        dt = max(0.0, float(sim_time) - float(ts))
        if dt > window_sec:
            continue

        launch_count += max(0.0, float(cnt))
        launch_prefill += max(0.0, float(prefill))
        ewma += max(0.0, float(cnt)) * math.exp(-alpha * dt)

    return launch_count, launch_prefill, ewma


def _decode_credit_available(state: VidurMCTSState) -> float:
    counted = getattr(getattr(state, "stats", None), "decode_tokens_counted", {}) or {}
    raw = _safe_float(counted.get(META_DECODE_CREDIT_BAL, 0.0), 0.0)
    return max(0.0, raw)


# NOTE to Shazer: This function can't be batched across multiple processes , this func is used in the MCTS search mainly --> therefore in the  mcts search + arena

def build_model_inputs(
    state: VidurMCTSState,
    player: str,
    device: torch.device,
    *,
    build_action_mask_flag: bool = False,
    action_mask_fn: Optional[Callable[[VidurMCTSState, str, int, torch.device], torch.Tensor]] = None,
    debug: bool = False,
    debug_out_path: Optional[str] = None,
) -> ModelInputs:
    feat_device = torch.device("cpu")
    sim = state.simulator
    sim_time = _safe_float(getattr(sim, "_time", 0.0), 0.0)
    stats = getattr(state, "stats", None)

    lookup = _build_request_lookup(sim)
    reqs = list(lookup.values())

    prefill_reqs = [r for r in reqs if _is_prefill_request(r)]
    decode_reqs = [r for r in reqs if _is_decode_request(r)]

    def _prefill_key(r):
        dl = _prefill_deadline(r)
        time_left = (float("inf") if dl is None else float(dl) - sim_time)
        late = _request_total_lateness(state, r, sim_time)
        rid = _safe_int(getattr(r, "id", 0), 0)
        return (time_left, -late, rid)

    prefill_reqs.sort(key=_prefill_key)

    def _decode_priority(r):
        late = _request_total_lateness(state, r, sim_time)
        processed = _safe_int(getattr(r, "num_processed_decode_tokens", 0), 0)
        remaining = _decode_remaining(r)
        rid = _safe_int(getattr(r, "id", 0), 0)
        violated = 1 if _is_violated(state, r) else 0
        return (-violated, -late, -processed, -remaining, rid)

    if len(decode_reqs) > N_DECODE_REQ:
        decode_reqs.sort(key=_decode_priority)
        decode_reqs = decode_reqs[:N_DECODE_REQ]
    else:
        decode_reqs.sort(key=_decode_priority)

    prefill_feat = torch.zeros((1, N_PREFILL_REQ, D_PREFILL_REQ), dtype=torch.float32, device=feat_device)
    decode_feat = torch.zeros((1, N_DECODE_REQ, D_DECODE_REQ), dtype=torch.float32, device=feat_device)
    prefill_mask = torch.zeros((1, N_PREFILL_REQ), dtype=torch.bool, device=feat_device)
    decode_mask = torch.zeros((1, N_DECODE_REQ), dtype=torch.bool, device=feat_device)

    for i, req in enumerate(prefill_reqs[:N_PREFILL_REQ]):
        total_prefill = _safe_int(getattr(req, "num_prefill_tokens", 0), 0)
        rem_prefill = _prefill_remaining(req)
        done_prefill = max(0, total_prefill - rem_prefill)
        age = max(0.0, sim_time - _arrived_at(req))
        prefill_late = _safe_float(getattr(stats, "per_request_prefill_lateness", {}).get(_safe_int(getattr(req, "id", -1), -1), 0.0), 0.0)
        deadline = _prefill_deadline(req)
        slack = 0.0 if deadline is None else float(deadline) - sim_time
        prefill_slo = _safe_float(getattr(req, "_prefill_slo_time", getattr(req, "prefill_slo_time", 0.0)), 0.0)
        violated_bit = 1.0 if _is_violated(state, req) else 0.0
        processed_frac = 0.0 if total_prefill <= 0 else _clip(float(done_prefill) / float(max(1, total_prefill)), 0.0, 1.0)

        prefill_feat[0, i, 0] = _norm01(rem_prefill, F.prefill_remaining_den)
        prefill_feat[0, i, 1] = _norm01(total_prefill, F.prefill_total_den)
        prefill_feat[0, i, 2] = processed_frac
        prefill_feat[0, i, 3] = _norm01(age, F.age_den_sec)
        prefill_feat[0, i, 4] = _norm01(prefill_late, F.lateness_den_sec)
        prefill_feat[0, i, 5] = _centered01(slack, F.slack_den_sec)
        prefill_feat[0, i, 6] = _norm01(prefill_slo, F.prefill_slo_den_sec)
        prefill_feat[0, i, 7] = violated_bit
        prefill_feat[0, i, 8] = 1.0 if prefill_late > float(F.near_drop_lateness_low_sec) else 0.0
        prefill_feat[0, i, 9] = 1.0 if prefill_late >= float(F.near_drop_lateness_high_sec) else 0.0
        prefill_mask[0, i] = True

    for i, req in enumerate(decode_reqs[:N_DECODE_REQ]):
        total_decode = _safe_int(getattr(req, "_num_decode_tokens", getattr(req, "num_decode_tokens", 0)), 0)
        rem_decode = _decode_remaining(req)
        done_decode = max(0, total_decode - rem_decode)
        age = max(0.0, sim_time - _arrived_at(req))
        late = _request_total_lateness(state, req, sim_time)
        decode_deadline = _decode_deadline(req, state)
        decode_slack = 0.0 if decode_deadline is None else float(decode_deadline) - sim_time
        decode_slo = _safe_float(getattr(req, "_decode_slo_time", getattr(req, "decode_slo_time", 0.0)), 0.0)
        violated_bit = 1.0 if _is_violated(state, req) else 0.0
        processed_frac = 0.0 if total_decode <= 0 else _clip(float(done_decode) / float(max(1, total_decode)), 0.0, 1.0)

        decode_feat[0, i, 0] = _norm01(rem_decode, F.decode_remaining_den)
        decode_feat[0, i, 1] = _norm01(total_decode, F.decode_total_den)
        decode_feat[0, i, 2] = _norm01(done_decode, F.decode_processed_den)
        decode_feat[0, i, 3] = processed_frac
        decode_feat[0, i, 4] = _norm01(age, F.age_den_sec)
        decode_feat[0, i, 5] = _norm01(late, F.lateness_den_sec)
        decode_feat[0, i, 6] = _centered01(decode_slack, F.slack_den_sec)
        decode_feat[0, i, 7] = _norm01(decode_slo, F.decode_slo_den_sec)
        decode_feat[0, i, 8] = violated_bit
        decode_feat[0, i, 9] = 1.0 if late > float(F.near_drop_lateness_low_sec) else 0.0
        decode_feat[0, i, 10] = 1.0 if late >= float(F.near_drop_lateness_high_sec) else 0.0
        decode_feat[0, i, 11] = 1.0 if done_decode > 216 else 0.0
        decode_feat[0, i, 12] = 1.0 if done_decode > 512 else 0.0
        decode_mask[0, i] = True

    violated_ids = set(getattr(getattr(state, "stats", None), "violated_request_ids", set()) or set())
    active_ids = {_safe_int(getattr(r, "id", -1), -1) for r in reqs}
    num_violated_active = len([rid for rid in active_ids if rid in violated_ids])

    p_late_05_15 = 0
    p_late_15 = 0
    for r in prefill_reqs:
        late = _safe_float(getattr(stats, "per_request_prefill_lateness", {}).get(_safe_int(getattr(r, "id", -1), -1), 0.0), 0.0)
        if late > float(F.near_drop_lateness_low_sec) and late < float(F.near_drop_lateness_high_sec):
            p_late_05_15 += 1
        elif late >= float(F.near_drop_lateness_high_sec):
            p_late_15 += 1

    d_late_05_15 = 0
    d_late_15 = 0
    for r in decode_reqs:
        late = _request_total_lateness(state, r, sim_time)
        if late > float(F.near_drop_lateness_low_sec) and late < float(F.near_drop_lateness_high_sec):
            d_late_05_15 += 1
        elif late >= float(F.near_drop_lateness_high_sec):
            d_late_15 += 1

    num_prefill = len(prefill_reqs)
    num_decode = len(decode_reqs)
    num_active = len(reqs)
    total_remaining_prefill = sum(_prefill_remaining(r) for r in prefill_reqs)
    total_remaining_decode = sum(_decode_remaining(r) for r in decode_reqs)
    total_decode_generated_active = sum(_safe_int(getattr(r, "num_processed_decode_tokens", 0), 0) for r in decode_reqs)
    slo_violations = _safe_int(getattr(stats, "slo_violations", 0), 0)
    slo_lateness_sum = _safe_float(getattr(stats, "slo_lateness_sum", 0.0), 0.0)
    objective_cost = float(slo_violations) + float(slo_lateness_sum)
    launch_count, launch_prefill, ewma = _recent_launch_summary(state, sim_time)
    decode_credit = _decode_credit_available(state)

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
            _norm01(ewma, F.recent_launch_count_den),
            _norm01(decode_credit, F.decode_credit_den),
            1.0 if num_prefill > 0 else 0.0,
            1.0 if num_decode > 0 else 0.0,
        ]],
        dtype=torch.float32,
        device=feat_device,
    )

    if build_action_mask_flag:
        action_mask = build_action_mask(state, player, device, action_mask_fn=action_mask_fn)
    else:
        action_mask = None

    legacy_width = max(int(prefill_feat.size(-1)), int(decode_feat.size(-1)))
    req_features_legacy = torch.cat(
        [
            Fnn.pad(prefill_feat, (0, legacy_width - int(prefill_feat.size(-1)))),
            Fnn.pad(decode_feat, (0, legacy_width - int(decode_feat.size(-1)))),
        ],
        dim=1,
    )
    req_mask_legacy = torch.cat([prefill_mask, decode_mask], dim=1)

    return ModelInputs(
        prefill_req_features=prefill_feat,
        decode_req_features=decode_feat,
        global_features=global_feat,
        prefill_req_mask=prefill_mask,
        decode_req_mask=decode_mask,
        action_mask=action_mask,
        req_features=req_features_legacy,
        req_mask=req_mask_legacy,
    )




def _infer_debug_dump(
    *,
    player: str,
    sim_time: float,
    num_prefill: int,
    num_decode: int,
    missed_prefill: int,
    total_active: int,
    prefill_rate: float,
    backlog_over_rate: float,
    prefill_over_200: float,                 
    decode_over_200: float,  
    decode_violated_over_200: float,
    total_violated_over_200: float,                
    total_remaining_prefill_tokens: int,     
    remaining_prefill_norm: float,           
    req_debug_rows: List[Dict[str, float]],
    out_path: Optional[str] = None,
) -> None:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append(f"[INFER DEBUG] player={player} sim_time={sim_time:.6f}")
    lines.append(
        f"prefill={num_prefill} decode={num_decode} missed_prefill_deadline={missed_prefill} "
        f"total_active={total_active} prefill_rate={prefill_rate:.3f} backlog_over_rate={backlog_over_rate:.3f}"
    )
    lines.append(
        f"prefill/200={prefill_over_200:.6f} decode/200={decode_over_200:.6f} "
        f"remaining_prefill_tokens={total_remaining_prefill_tokens} "
        f"remaining_prefill_norm={remaining_prefill_norm:.6f}"
    )

    lines.append(
        f"decode_violated/200={decode_violated_over_200:.6f} "
        f"total_violated/200={total_violated_over_200:.6f}"
    )



    lines.append("-" * 100)

    for j, d in enumerate(req_debug_rows, start=1):
        lines.append(f"R{j}: req_id={int(d.get('req_id', -1))}")
        lines.append(
            f"  remaining_prefill={int(d.get('remaining_prefill', 0))} "
            f"rem_norm={d.get('rem_norm', 0.0):.6f}"
        )
        lines.append(
            f"  cached_prefill={int(d.get('cached_prefill', 0))} "
            f"cached_norm={d.get('cached_norm', 0.0):.6f}"
        )
        lines.append(
            f"  deadline={d.get('deadline', float('nan')):.6f} sim_time={sim_time:.6f} "
            f"time_left={d.get('time_left', 0.0):.6f}"
        )
        lines.append(
            f"  slo={d.get('slo', 0.0):.6f} slowdown={d.get('slowdown', 0.0):.6f} "
            f"base_total_exec={d.get('base_total_exec', 0.0):.6f}"
        )
        lines.append(
            f"  frac_remaining={d.get('frac_remaining', 0.0):.6f} "
            f"est_remaining_exec={d.get('est_remaining_exec', 0.0):.6f}"
        )
        lines.append(
            f"  slack={d.get('slack', 0.0):.6f} slack_ratio={d.get('slack_ratio', 0.0):.6f}"
        )


    text = "\n".join(lines) + "\n"

    if out_path:
        from pathlib import Path

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)













