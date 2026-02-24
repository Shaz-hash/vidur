
# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)


"""
infer.py

Purpose:
- Convert a VidurMCTSState (i.e., simulator + stats) into NN-ready tensors:
    req_features   [1, N_REQ, D_REQ]
    global_features[1, D_GLOBAL]
    req_mask       [1, N_REQ] (bool)
    action_mask    [1, num_actions(player)] (bool)

This file is intentionally separate from:
- models.py (pure NN: forward + heads)
- environment.py (sim transitions + action generation)

It keeps environment readable while still letting the model infer from simulator state.

Notes:
- This builds a *minimal* feature set that matches what you described earlier.
- You will likely refine/normalize these features later.
"""

# =============================================================================
# ModelInputs feature schema (infer.py)
# =============================================================================
# Shapes:
#   req_features    : [1, N_REQ, D_REQ]   (prefill requests only)
#   global_features : [1, D_GLOBAL]
#   req_mask        : [1, N_REQ] bool     (True = this slot is a real request)
#   action_mask     : [1, A(player)] bool (True = action is valid)
#
# Request ordering:
#   We take ONLY active prefill requests (not completed and not prefill-complete),
#   and sort them by urgency: (time_left_to_deadline, request_id). First N_REQ kept.
#
# Per-request features (D_REQ=3), for slot i:
#   f0 = remaining_prefill_norm
#        = (num_prefill_tokens - num_processed_prefill_tokens) / max_prefill_tokens
#          (max_prefill_tokens default = 3072)
#
#   f1 = cached_prefill_norm
#        = num_processed_prefill_tokens / max_prefill_tokens
#
#   f2 = slack_ratio (clipped to [-slack_clip, +slack_clip], slack_clip default = 5.0)
#        deadline = queued_at + prefill_slo_time
#        time_left = deadline - sim_time
#        base_total_exec = prefill_slo_time / prefill_slowdown   (prefill_slowdown default = 3.0)
#        frac_remaining = remaining_prefill / max(1, total_prefill_tokens)
#        est_remaining_exec = base_total_exec * frac_remaining
#        slack = time_left - est_remaining_exec
#        slack_ratio = slack / prefill_slo_time
#
# Global features (D_GLOBAL=7):
#   g0 = prefill_frac_total  = num_prefill_active / max(1, total_active_requests)
#   g1 = decode_frac_total   = num_decode_active  / max(1, total_active_requests)
#   g2 = prefill_missed_frac = missed_prefill_deadlines / max(1, num_prefill_active)
#   g3 = backlog_over_rate   = min(num_prefill_active / max(1, prefill_rate + 1), 25.0)
#        prefill_rate uses state.stats.maximum_qps if set, else fallback 5.0
#   g4 = prefill_over_200    = num_prefill_active / 200
#   g5 = decode_over_200     = num_decode_active  / 200
#   g6 = remaining_prefill_norm
#        = total_remaining_prefill_tokens / (3072 * 10)
#
# Note:
#   action_mask is currently all-True unless an action_mask_fn is provided.
# =============================================================================







from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..environment import VidurMCTSState
from .types import ModelInputs

# -----------------------------
# Shape constants (keep aligned with models.py)
# -----------------------------
N_REQ: int = 20
D_REQ: int = 3  # per-request features
D_GLOBAL: int = 9

# Placeholder action space sizes (keep aligned with your policy heads)
NUM_ACTIONS_CONTROLLER: int = 24
NUM_ACTIONS_ADVERSARY: int = 6

MAX_ACTIVE_REQUESTS: int = 200 ## Used for normalization of number of prefill requests in the system
REMAINING_PREFILL_DENOM: int = 3072 * 10 ## Used for normalization of remaining prefill tokens for all the requests in the system


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

# NOTE to Shazer: This function can't be batched across multiple processes , this func is used in the MCTS search mainly --> therefore in the  mcts search + arena
def build_model_inputs(
    state: VidurMCTSState,
    player: str,
    device: torch.device,
    *,
    build_action_mask_flag : bool = False,  
    action_mask_fn: Optional[Callable[[VidurMCTSState, str, int, torch.device], torch.Tensor]] = None,
    max_prefill_tokens: int = 3072,
    prefill_slowdown: float = 3.0,
    slack_clip: float = 5.0,
    debug: bool = False,
    debug_out_path: Optional[str] = None,

) -> ModelInputs:
    """
    Minimal feature spec (as you described):
    Per-request (prefill only), up to N_REQ:
      1) prefill_size
      2) prefill_deadline (absolute time)
      3) prefill_tokens_cached
      4) time_left_to_deadline (deadline - sim_time)

    Global:
      1) num_prefill_requests_in_system
      2) num_decode_requests_in_system
      3) num_prefill_deadline_missed (at current sim_time)

    Returns single-batch tensors (B=1) since MCTS calls per-node inference.
    """

    # Forcefully making device to CPU 
    feat_device = torch.device("cpu")

    sim = state.simulator
    sim_time = _safe_float(getattr(sim, "_time", 0.0), 0.0)

    lookup = _build_request_lookup(sim)
    requests = list(lookup.values())

    # Identify prefill requests and sort deterministically by urgency then id
    prefill_reqs: List[object] = [r for r in requests if _is_prefill_request(r)]

    req_debug_rows: List[Dict[str, float]] = []

    ## TODO: this should not be required because we assume all requests have deadlines
    def urgency_key(req) -> Tuple[float, int]:
        dl = _prefill_deadline(req)
        # If no deadline, push to the end
        if dl is None:
            dl = float("inf")
        time_left = dl - sim_time
        rid = _safe_int(getattr(req, "id", 0), 0)
        return (time_left, rid)

    prefill_reqs.sort(key=urgency_key)

    # Build per-request matrix
    req_feat = torch.zeros((1, N_REQ, D_REQ), dtype=torch.float32, device=feat_device)
    req_mask = torch.zeros((1, N_REQ), dtype=torch.bool, device=feat_device)

    eps = 1e-9
    max_prefill_tokens_f = float(max(1, int(max_prefill_tokens)))
    slowdown = float(max(prefill_slowdown, eps))

    for i, req in enumerate(prefill_reqs[:N_REQ]):
        total_pref = _safe_int(getattr(req, "num_prefill_tokens", 0), 0)
        done_pref = _safe_int(getattr(req, "num_processed_prefill_tokens", 0), 0)
        remaining_pref = max(0, total_pref - done_pref)

        cached = done_pref

        # Normalize sizes into [0,1]
        rem_norm = float(remaining_pref) / max_prefill_tokens_f
        cached_norm = float(cached) / max_prefill_tokens_f

        # Slack ratio: (time_left - est_remaining_exec_time) / prefill_slo_time
        slo = getattr(req, "_prefill_slo_time", None)
        if slo is None:
            slo = getattr(req, "prefill_slo_time", None)

        slack_ratio = 0.0
        if slo is not None and float(slo) > 0.0:
            queued_at = _safe_float(getattr(req, "queued_at", getattr(req, "arrived_at", 0.0)), 0.0)
            deadline = float(queued_at) + float(slo)
            time_left = deadline - sim_time

            # Approximate remaining exec time:
            # prefill_slo_time = base_exec_time * slowdown  => base_exec_time = prefill_slo_time / slowdown
            base_total_exec = float(slo) / slowdown
            frac_remaining = float(remaining_pref) / float(max(1, total_pref))
            est_remaining_exec = base_total_exec * frac_remaining

            slack = time_left - est_remaining_exec
            slack_ratio = slack / float(slo)

            # if slack_ratio > 1 :
            #     print("SLACK RATIO > 1: time_left=", time_left, " est_remaining_exec=", est_remaining_exec, " slo=", slo)
            #     print("  (queued_at=", queued_at, " sim_time=", sim_time, " remaining_pref=", remaining_pref, " total_pref=", total_pref, ")")
            # elif slack_ratio < -1:
            #     print("SLACK RATIO < -1: time_left=", time_left, " est_remaining_exec=", est_remaining_exec, " slo=", slo)
            #     print("  (queued_at=", queued_at, " sim_time=", sim_time, " remaining_pref=", remaining_pref, " total_pref=", total_pref, ")")

            # clip for stability
            if slack_ratio > slack_clip:
                slack_ratio = slack_clip
            elif slack_ratio < -slack_clip:
                slack_ratio = -slack_clip

        # Feature vector: [remaining_prefill_norm, cached_norm, slack_ratio]
        req_feat[0, i, 0] = rem_norm
        req_feat[0, i, 1] = cached_norm
        req_feat[0, i, 2] = float(slack_ratio)

        req_mask[0, i] = True


        if debug:
            queued_at = _safe_float(getattr(req, "queued_at", getattr(req, "arrived_at", 0.0)), 0.0)
            deadline = float("nan")
            time_left = 0.0
            slo_val = 0.0
            base_total_exec = 0.0
            frac_remaining = 0.0
            est_remaining_exec = 0.0
            slack = 0.0

            slo = getattr(req, "_prefill_slo_time", None)
            if slo is None:
                slo = getattr(req, "prefill_slo_time", None)

            if slo is not None and float(slo) > 0.0:
                slo_val = float(slo)
                deadline = float(queued_at) + slo_val
                time_left = float(deadline) - float(sim_time)
                base_total_exec = slo_val / slowdown
                frac_remaining = float(remaining_pref) / float(max(1, total_pref))
                est_remaining_exec = base_total_exec * frac_remaining
                slack = time_left - est_remaining_exec

            req_debug_rows.append(
                {
                    "req_id": float(_safe_int(getattr(req, "id", -1), -1)),
                    "remaining_prefill": float(remaining_pref),
                    "rem_norm": float(rem_norm),
                    "cached_prefill": float(cached),
                    "cached_norm": float(cached_norm),
                    "deadline": float(deadline),
                    "time_left": float(time_left),
                    "slo": float(slo_val),
                    "slowdown": float(slowdown),
                    "base_total_exec": float(base_total_exec),
                    "frac_remaining": float(frac_remaining),
                    "est_remaining_exec": float(est_remaining_exec),
                    "slack": float(slack),
                    "slack_ratio": float(slack_ratio),
                }
            )


    

    # ----------------
    # Global features (normalized)
    # ----------------
    total_active = len(requests)  # not completed (best-effort from lookup)
    denom_total = float(max(1, total_active))

    num_prefill = len(prefill_reqs)
    num_decode = sum(1 for r in requests if _is_decode_request(r))

    # prefill missed ratio: missed_prefill / total_prefill
    missed_prefill = 0
    for req in prefill_reqs:
        dl = _prefill_deadline(req)
        if dl is not None and sim_time > dl:
            missed_prefill += 1
    denom_prefill = float(max(1, num_prefill))

    prefill_frac_total = float(num_prefill) / denom_total
    decode_frac_total = float(num_decode) / denom_total
    prefill_missed_frac = float(missed_prefill) / denom_prefill

    # extra: backlog vs prefill rate (qps), clamped to 25
    prefill_rate = float(getattr(getattr(state, "stats", None), "maximum_qps", 0) or 0)
    # better: pass rate explicitly; fallback to constraints maximum_qps (your env uses it)
    if prefill_rate <= 0:
        prefill_rate = 5.0  # fallback
    backlog_over_rate = float(num_prefill) / float(max(1.0, prefill_rate + 1))
    backlog_over_rate = min(backlog_over_rate, 25.0)

    # NEW (requested):
    prefill_over_200 = float(num_prefill) / float(MAX_ACTIVE_REQUESTS)
    decode_over_200 = float(num_decode) / float(MAX_ACTIVE_REQUESTS)

    total_remaining_prefill_tokens = 0
    for req in prefill_reqs:
        total_pref = _safe_int(getattr(req, "num_prefill_tokens", 0), 0)
        done_pref = _safe_int(getattr(req, "num_processed_prefill_tokens", 0), 0)
        total_remaining_prefill_tokens += max(0, total_pref - done_pref)

    remaining_prefill_norm = float(total_remaining_prefill_tokens) / float(REMAINING_PREFILL_DENOM)


    # NEW: violated-request features (history-based, tracked by environment stats)
    violated_ids = set(getattr(getattr(state, "stats", None), "violated_request_ids", set()) or set())

    num_total_violated_active = 0
    num_decode_violated_active = 0

    for r in requests:
        rid = _safe_int(getattr(r, "id", None), -1)
        if rid in violated_ids:
            num_total_violated_active += 1
            if _is_decode_request(r):
                num_decode_violated_active += 1

    decode_violated_over_200 = float(num_decode_violated_active) / float(MAX_ACTIVE_REQUESTS)
    total_violated_over_200 = float(num_total_violated_active) / float(MAX_ACTIVE_REQUESTS)


    global_feat = torch.tensor(
        [[
            prefill_frac_total,
            decode_frac_total,
            prefill_missed_frac,
            backlog_over_rate,
            prefill_over_200,
            decode_over_200,
            decode_violated_over_200,   # NEW
            total_violated_over_200,    # NEW
            remaining_prefill_norm,
        ]],
        dtype=torch.float32,
        device=feat_device,
    )


    # action_mask = build_action_mask(state, player, device, action_mask_fn=action_mask_fn)
    if build_action_mask_flag:
        action_mask = build_action_mask(state, player, device, action_mask_fn=action_mask_fn)
    else :
        action_mask = None

    # if debug:
    #     _infer_debug_dump(
    #         player=player,
    #         sim_time=sim_time,
    #         num_prefill=int(num_prefill),
    #         num_decode=int(num_decode),
    #         missed_prefill=int(missed_prefill),
    #         total_active=int(total_active),
    #         prefill_rate=float(prefill_rate),
    #         backlog_over_rate=float(backlog_over_rate),
    #         prefill_over_200=float(prefill_over_200),
    #         decode_over_200=float(decode_over_200),
    #         decode_violated_over_200=float(decode_violated_over_200),
    #         total_violated_over_200=float(total_violated_over_200),
    #         total_remaining_prefill_tokens=int(total_remaining_prefill_tokens),
    #         remaining_prefill_norm=float(remaining_prefill_norm),
    #         req_debug_rows=req_debug_rows,
    #         out_path=debug_out_path,
    #     )


    return ModelInputs(
        req_features=req_feat,
        global_features=global_feat,
        req_mask=req_mask,
        action_mask=action_mask,
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

















