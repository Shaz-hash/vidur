"""
Build the new state-local feature schema described in TASK.md (sections A, B, C).

This pulls features straight from the stored root shards' `simulator_snapshot` +
`stats` rather than restoring full VidurMCTSState objects, mirroring the approach
in `classical_value_model.extract_state_feature_vector`.

The output is a single .npy of shape (N_records, D_FEATURES) and a small json
with feature names + dataset psid offsets so the trainer can split train/eval.

Schema (in order):

  - 21 globals (skip the 6 forbidden cols):
      g05_num_prefill, g06_num_decode, g07_num_active,
      g08_total_remaining_prefill, g09_total_remaining_decode,
      g10_total_decode_generated_active, g11_num_violated_active,
      g12_p_late_05_15, g13_p_late_15, g14_d_late_05_15, g15_d_late_15,
      g16_launch_count, g17_launch_prefill,
      g18_remaining_launch_request_headroom, g19_remaining_launch_prefill_headroom,
      g20_ewma, g21_decode_credit,
      g_extra_prefill_violated_active, g_extra_decode_violated_active,
      g_edf_minus_batch_norm  (clip(min_prefill_slack - decode_time_at_max, 0, 1)),
      g_n_active_edf_margin_gt_batch_norm  (#active reqs with per-req deadline margin
        > decode_time_at_max, divided by F.active_total_count_den=120)

  - 7 prefill slots, each emits 25 dims:
      [present, remaining_tokens, total_tokens, violated_bit,
       lateness_bucket(4-onehot), slack_bucket(17-onehot)]
      Slack bucket scheme: bucket 0 = exact zero, buckets 1..6 are 6 fine sub-buckets
      within the 128-token-profile band (~0.0157s) to resolve the decode-batch boundary,
      buckets 7..15 follow the original prefill_profile.csv edges, bucket 16 = > 3072-token-time.
      So per-slot: present(1) + rem(1) + tot(1) + viol(1) + late_oh(4) + slack_oh(17) = 25

  - 7 decode slots, each emits 4 dims:
      [present, decode_remaining_norm, done_gt_216, done_gt_512]
      (active and not yet violated; deterministic random pick across state if > 7)

Per-slot prefill = 25, x7 = 175
Per-slot decode  = 4 , x7 = 28
Globals          = 21
Total            = 21 + 175 + 28 = 224
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch  # only for torch.load; keep python-side feature extraction


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ----------------------- denominators / constants (mirror infer.py) -----------------------

# These mirror DEFAULT_GAME_V2_CONFIG.features.* and timing.* — hardcoded here to keep the
# build script independent of the simulator module graph during feature extraction.
F_ACTIVE_PREFILL_COUNT_DEN: float = 20.0
F_ACTIVE_DECODE_COUNT_DEN: float = 100.0
F_ACTIVE_TOTAL_COUNT_DEN: float = 120.0
F_TOTAL_REMAINING_PREFILL_DEN: float = 20.0 * 4096.0
F_TOTAL_REMAINING_DECODE_DEN: float = 100.0 * 864.0
F_TOTAL_DECODE_GENERATED_ACTIVE_DEN: float = 100.0 * 864.0
F_VIOLATED_COUNT_DEN: float = 100.0
F_PREFILL_NEAR_DROP_DEN: float = 20.0
F_DECODE_NEAR_DROP_DEN: float = 100.0
F_RECENT_LAUNCH_COUNT_DEN: float = 7.0
F_RECENT_LAUNCH_PREFILL_DEN: float = 1024.0 * 7.0
F_DECODE_CREDIT_DEN: float = 100.0 * 216.0
F_NEAR_DROP_LATENESS_LOW_SEC: float = 0.5
F_NEAR_DROP_LATENESS_HIGH_SEC: float = 1.5
F_LAUNCH_EWMA_WINDOW_SEC: float = 1.0
F_LAUNCH_EWMA_ALPHA: float = 0.37
MAX_REQUESTS_PER_LAUNCH_WINDOW: int = 7
MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW: int = 1024 * 7
DECODE_REMAINING_DEN: float = 864.0


# decode_profile.csv lookup: decode_tokens -> decode_time_seconds.
# Loaded lazily at first use; the file path is set via env var or defaults to
# simulator_output/decode_profile.csv at the repo root.
_DECODE_PROFILE_TOKENS: list[int] = []
_DECODE_PROFILE_TIMES: list[float] = []
_DECODE_PROFILE_DEFAULT = REPO_ROOT / "simulator_output" / "decode_profile.csv"


def _load_decode_profile(path: Path | None = None) -> None:
    """Populate the global decode-profile arrays. Idempotent."""
    global _DECODE_PROFILE_TOKENS, _DECODE_PROFILE_TIMES
    if _DECODE_PROFILE_TOKENS:
        return
    p = path or Path(os.environ.get("DECODE_PROFILE_PATH", str(_DECODE_PROFILE_DEFAULT)))
    rows: list[tuple[int, float]] = []
    with open(p, newline="") as f:
        import csv as _csv
        reader = _csv.reader(f)
        next(reader)  # header
        for r in reader:
            if not r:
                continue
            rows.append((int(r[0]), float(r[1])))
    rows.sort()
    _DECODE_PROFILE_TOKENS = [t for t, _ in rows]
    _DECODE_PROFILE_TIMES = [t for _, t in rows]


def _decode_time_for_tokens(n_tokens: int) -> float:
    """Look up decode batch time for the nearest decode_tokens row (ties → smaller)."""
    if not _DECODE_PROFILE_TOKENS:
        _load_decode_profile()
    if n_tokens <= _DECODE_PROFILE_TOKENS[0]:
        return _DECODE_PROFILE_TIMES[0]
    if n_tokens >= _DECODE_PROFILE_TOKENS[-1]:
        return _DECODE_PROFILE_TIMES[-1]
    # find nearest with ties → smaller bucket
    best_i = 0
    best_d = abs(n_tokens - _DECODE_PROFILE_TOKENS[0])
    for i, t in enumerate(_DECODE_PROFILE_TOKENS):
        d = abs(n_tokens - t)
        if d < best_d or (d == best_d and t < _DECODE_PROFILE_TOKENS[best_i]):
            best_d = d
            best_i = i
    return _DECODE_PROFILE_TIMES[best_i]


_SIM_DECODE_ENV: Any | None = None
_SIM_DECODE_MCTS: Any | None = None
_SIM_DECODE_SCRATCH_STATE: Any | None = None


def _decode_time_profile_fallback_from_rows(decode_reqs: list[dict[str, Any]]) -> float:
    """Old approximation: nearest profile time for the largest active decode context."""
    if not decode_reqs:
        return 0.0
    max_decode_context_tokens = 0
    for r in decode_reqs:
        num_pref = max(0, _safe_int(r.get("num_prefill_tokens")))
        proc_total = max(0, _safe_int(r.get("num_processed_tokens")))
        done_decode = max(0, proc_total - num_pref)
        max_decode_context_tokens = max(max_decode_context_tokens, num_pref + done_decode)
    return _decode_time_for_tokens(int(max_decode_context_tokens))


def _decode_probe_env() -> Any:
    """Build one reusable GV3 Python env per process for decode-only timing probes."""
    global _SIM_DECODE_ENV, _SIM_DECODE_MCTS
    if _SIM_DECODE_ENV is not None:
        return _SIM_DECODE_ENV

    from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.root_storage import (
        build_storage_config,
        make_local_env_and_mcts,
    )

    cfg = build_storage_config(
        output_dir=REPO_ROOT / "simulator_output" / "_decode_time_probe_unused",
        num_roots=1,
        max_candidate_roots=1,
        candidate_batch_size=1,
        min_nonzero_target_ratio=0.0,
        history_hops_min=0,
        history_hops_max=1,
        root_player_filter="controller",
        environment_lang="python",
        include_history_trace_logs=False,
        num_processes=1,
    )
    _SIM_DECODE_ENV, _SIM_DECODE_MCTS = make_local_env_and_mcts(cfg)
    return _SIM_DECODE_ENV


def _restore_decode_probe_state(record: dict[str, Any]) -> Any:
    """Restore a stored root snapshot into a reusable scratch state."""
    global _SIM_DECODE_SCRATCH_STATE

    env = _decode_probe_env()
    snapshot = record.get("simulator_snapshot")
    stats = record.get("stats")
    if snapshot is None:
        raise ValueError("record is missing simulator_snapshot")
    clone_fn = getattr(stats, "clone", None)
    if not callable(clone_fn):
        raise TypeError("record stats must provide clone()")

    if _SIM_DECODE_SCRATCH_STATE is None:
        _SIM_DECODE_SCRATCH_STATE = env.initial_state()

    sim = _SIM_DECODE_SCRATCH_STATE.simulator
    if (
        isinstance(snapshot, dict)
        and snapshot.get("__mode__") == "mcts_fast"
        and hasattr(sim, "restore_state_fast")
    ):
        sim.restore_state_fast(snapshot)
    else:
        sim.restore_state(snapshot)
    _SIM_DECODE_SCRATCH_STATE.stats = clone_fn()
    return _SIM_DECODE_SCRATCH_STATE


def _decode_time_using_simulator(record: dict[str, Any], decode_reqs: list[dict[str, Any]]) -> float:
    """Execute the decode-only no-op action and return the actual batch time.

    GV3 no-op semantics automatically schedules one decode token for every
    active decode-ready request. Running it with fast_forward=False measures only
    the controller batch execution time, not any transition-time jump.
    """
    if not decode_reqs:
        return 0.0

    try:
        from vidur.mcts.game_types import ControllerAction

        env = _decode_probe_env()
        state = _restore_decode_probe_state(record)
        start_time = float(getattr(state.simulator, "_time", 0.0))
        action = ControllerAction(
            token_budget=0,
            selected_request_ids=None,
            token_allocations={},
            prefill_allocations={},
            decode_allocations={},
            heuristic=None,
            strategy="GV2|evict_none",
            mapping=(0, 0, 0),
        )
        child_state = env.apply_controller_action_only(
            state,
            action,
            inplace=True,
            fast_forward=False,
        )
        end_time = float(getattr(child_state.simulator, "_time", start_time))
        return max(0.0, end_time - start_time)
    except Exception:
        if os.environ.get("STRICT_SIM_DECODE_TIME") == "1":
            raise
        return _decode_time_profile_fallback_from_rows(decode_reqs)


# Slack bucket boundaries from TASK.md.
# 6 fine sub-buckets within the 128-token prefill-profile band (~0.0157s) to give
# the model resolution around the ~0.01329s decode-batch execution boundary
# (motivated by V1 outlier analysis in request_edf_analysis.csv).
# Bucket 0 = exactly zero slack; final bucket = >3072-token-time slack.
PREFILL_SLACK_BUCKET_EDGES = [
    0.0133,
    0.0135,
    0.0137,
    0.0140,
    0.0145,
    0.015725797204323228,
    0.023274675327417962,
    0.031963770276289896,
    0.03888623299112536,
    0.06091750997076902,
    0.07016849423102292,
    0.08612442901238251,
    0.09850190759874299,
    0.19613847773632437,
    0.28408680179190937,
]
N_SLACK_BUCKETS = 17   # 0=zero, 1..15=ranges, 16=>last
N_LATE_BUCKETS = 4

N_PREFILL_SLOTS = MAX_REQUESTS_PER_LAUNCH_WINDOW  # 7
N_DECODE_SLOTS = MAX_REQUESTS_PER_LAUNCH_WINDOW   # 7

D_PREFILL_PER_SLOT = 1 + 1 + 1 + 1 + N_LATE_BUCKETS + N_SLACK_BUCKETS  # = 25
D_DECODE_PER_SLOT = 4
D_GLOBAL = 21  # 19 base + 2 EDF-vs-decode-batch globals
D_TOTAL = D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT + N_DECODE_SLOTS * D_DECODE_PER_SLOT


# ----------------------- helpers -----------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _norm01(x: float, denom: float) -> float:
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(x) / float(denom)))


def _slack_bucket_idx(slack: float) -> int:
    if slack <= 0.0:
        return 0
    for i, edge in enumerate(PREFILL_SLACK_BUCKET_EDGES):
        if slack <= edge:
            return i + 1
    return len(PREFILL_SLACK_BUCKET_EDGES) + 1  # last bucket index = N_SLACK_BUCKETS - 1


def _lateness_bucket_idx(late: float) -> int:
    # buckets: [0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, +inf)
    if late < 0.5:
        return 0
    if late < 1.0:
        return 1
    if late < 1.5:
        return 2
    return 3


def _stats_attr(stats: Any, name: str, default: Any = None) -> Any:
    return getattr(stats, name, default)


def _stats_dict(stats: Any, name: str) -> dict:
    out = getattr(stats, name, None)
    return out if isinstance(out, dict) else {}


def _stats_set(stats: Any, name: str) -> set:
    out = getattr(stats, name, None)
    if out is None:
        return set()
    try:
        return set(int(x) for x in out)
    except Exception:
        return set()


def _is_prefill(req: dict) -> bool:
    if bool(req.get("completed", False)):
        return False
    if bool(req.get("is_prefill_complete", False)):
        return False
    return True


def _is_decode(req: dict) -> bool:
    if bool(req.get("completed", False)):
        return False
    if not bool(req.get("is_prefill_complete", False)):
        return False
    total = _safe_int(req.get("num_decode_tokens"))
    proc_total = _safe_int(req.get("num_processed_tokens"))
    total_pref = _safe_int(req.get("num_prefill_tokens"))
    done_decode = max(0, proc_total - total_pref)
    return total > done_decode


def _det_random_subset_indices(n_total: int, k: int, *, seed_key: str) -> list[int]:
    """Deterministic k-of-N subset selection seeded by a hash of seed_key."""
    if n_total <= k:
        return list(range(n_total))
    h = hashlib.sha256(seed_key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=k, replace=False).tolist())


# ----------------------- per-record feature extraction -----------------------


def extract_features_one_record(record: dict) -> np.ndarray:
    """Return the (D_TOTAL,) float32 feature vector for one root record."""

    snapshot = record.get("simulator_snapshot") or {}
    stats = record.get("stats")
    sim_time = _safe_float(snapshot.get("time"))

    # gather active requests
    request_states = list((snapshot.get("request_states") or {}).values())
    active_id_set: set[int] = _stats_set(stats, "active_request_ids")
    if active_id_set:
        active_requests = [
            r for r in request_states
            if isinstance(r, dict) and _safe_int(r.get("id"), -1) in active_id_set
        ]
    else:
        active_requests = [r for r in request_states if isinstance(r, dict)]

    violated_set = _stats_set(stats, "violated_request_ids")
    per_req_prefill_late = _stats_dict(stats, "per_request_prefill_lateness")
    per_req_decode_late = _stats_dict(stats, "per_request_decode_lateness")
    decode_deadlines = _stats_dict(stats, "decode_next_deadline_by_id")

    prefill_reqs = [r for r in active_requests if _is_prefill(r)]
    decode_reqs = [r for r in active_requests if _is_decode(r)]

    # ------ globals -------
    num_prefill = len(prefill_reqs)
    num_decode = len(decode_reqs)
    num_active = len(active_requests)

    total_remaining_prefill = sum(
        max(0.0, _safe_float(r.get("remaining_prefill_tokens"))) for r in prefill_reqs
    )

    def _decode_remaining(r: dict) -> float:
        total_dec = max(0.0, _safe_float(r.get("num_decode_tokens")))
        proc_total = max(0.0, _safe_float(r.get("num_processed_tokens")))
        total_pref = max(0.0, _safe_float(r.get("num_prefill_tokens")))
        done_decode = max(0.0, proc_total - total_pref)
        return max(0.0, total_dec - done_decode)

    def _decode_done(r: dict) -> float:
        proc_total = max(0.0, _safe_float(r.get("num_processed_tokens")))
        total_pref = max(0.0, _safe_float(r.get("num_prefill_tokens")))
        return max(0.0, proc_total - total_pref)

    total_remaining_decode = sum(_decode_remaining(r) for r in decode_reqs)
    total_decode_generated_active = sum(
        max(0.0, _safe_float(r.get("num_processed_decode_tokens")))
        if "num_processed_decode_tokens" in r
        else _decode_done(r)
        for r in decode_reqs
    )

    active_ids = {_safe_int(r.get("id"), -1) for r in active_requests}
    num_violated_active = len(active_ids & violated_set)
    num_prefill_violated = sum(
        1 for r in prefill_reqs if _safe_int(r.get("id"), -1) in violated_set
    )
    num_decode_violated = sum(
        1 for r in decode_reqs if _safe_int(r.get("id"), -1) in violated_set
    )

    # late buckets
    p_late_05_15 = 0
    p_late_15 = 0
    for r in prefill_reqs:
        rid = _safe_int(r.get("id"), -1)
        late = _safe_float(per_req_prefill_late.get(rid, 0.0))
        # also fall back to "now - deadline" if not stored
        if late <= 0.0:
            arrived_at = _safe_float(r.get("arrived_at"))
            slo_t = _safe_float(r.get("prefill_slo_time"))
            late = max(0.0, sim_time - (arrived_at + slo_t))
        if late > F_NEAR_DROP_LATENESS_LOW_SEC and late < F_NEAR_DROP_LATENESS_HIGH_SEC:
            p_late_05_15 += 1
        elif late >= F_NEAR_DROP_LATENESS_HIGH_SEC:
            p_late_15 += 1

    d_late_05_15 = 0
    d_late_15 = 0
    for r in decode_reqs:
        rid = _safe_int(r.get("id"), -1)
        late_p = _safe_float(per_req_prefill_late.get(rid, 0.0))
        late_d = _safe_float(per_req_decode_late.get(rid, 0.0))
        late = max(late_p, late_d)
        if late > F_NEAR_DROP_LATENESS_LOW_SEC and late < F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_05_15 += 1
        elif late >= F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_15 += 1

    # launch window summary (mirror _recent_launch_summary)
    launch_count = 0.0
    launch_prefill = 0.0
    ewma = 0.0
    for item in (_stats_attr(stats, "recent_arrivals") or []):
        ts = None
        cnt = 0
        prefill = 0
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            ts = _safe_float(item[0])
            cnt = _safe_int(item[1])
            prefill = _safe_int(item[2])
        elif isinstance(item, dict):
            ts = _safe_float(item.get("timestamp", item.get("time")))
            cnt = _safe_int(item.get("count", item.get("requests", 0)))
            prefill = _safe_int(item.get("prefill_tokens", item.get("tokens", 0)))
        elif isinstance(item, (int, float)):
            ts = _safe_float(item)
            cnt = 1
        if ts is None:
            continue
        dt = max(0.0, sim_time - ts)
        if dt > F_LAUNCH_EWMA_WINDOW_SEC:
            continue
        launch_count += max(0.0, float(cnt))
        launch_prefill += max(0.0, float(prefill))
        ewma += max(0.0, float(cnt)) * math.exp(-F_LAUNCH_EWMA_ALPHA * dt)

    remaining_launch_request_headroom = max(0.0, MAX_REQUESTS_PER_LAUNCH_WINDOW - launch_count)
    remaining_launch_prefill_headroom = max(0.0, MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW - launch_prefill)

    # decode credit available (META_DECODE_CREDIT_BAL key in decode_tokens_counted)
    decode_tokens_counted = _stats_dict(stats, "decode_tokens_counted")
    META_DECODE_CREDIT_BAL = -9_100_005
    decode_credit = max(0.0, _safe_float(decode_tokens_counted.get(META_DECODE_CREDIT_BAL, 0.0)))

    # EDF-vs-decode-batch-time scalar global (TASK.md §A.1).
    # min_prefill_slack across active prefill requests (default +inf if none).
    if prefill_reqs:
        min_prefill_slack = math.inf
        for r in prefill_reqs:
            arrived_at = _safe_float(r.get("arrived_at"))
            slo_t = _safe_float(r.get("prefill_slo_time"))
            slack = (arrived_at + slo_t) - sim_time
            if slack < min_prefill_slack:
                min_prefill_slack = slack
    else:
        min_prefill_slack = math.inf

    # Simulator-measured decode-only batch time from this exact state.
    # This executes the GV3 strict-noop controller path with fast_forward=False,
    # which schedules one decode token per active decode-ready request.
    decode_time_at_max = _decode_time_using_simulator(record, decode_reqs)

    if math.isinf(min_prefill_slack):
        edf_minus_batch_norm = 1.0
    else:
        edf_minus_batch_norm = max(0.0, min(1.0, min_prefill_slack - decode_time_at_max))

    # Count of active requests whose own deadline-vs-batch margin is strictly positive.
    # For prefill phase: deadline = arrived_at + prefill_slo_time.
    # For decode phase: deadline = decode_next_deadline_by_id[rid] (fallback arrived_at + decode_slo_time).
    n_active_edf_margin_gt_batch = 0
    for r in prefill_reqs:
        arrived_at = _safe_float(r.get("arrived_at"))
        slo_t = _safe_float(r.get("prefill_slo_time"))
        deadline = arrived_at + slo_t
        if (deadline - sim_time) - decode_time_at_max > 0.0:
            n_active_edf_margin_gt_batch += 1
    for r in decode_reqs:
        rid = _safe_int(r.get("id"), -1)
        arrived_at = _safe_float(r.get("arrived_at"))
        decode_slo = _safe_float(r.get("decode_slo_time"))
        deadline = _safe_float(decode_deadlines.get(rid, arrived_at + decode_slo))
        if (deadline - sim_time) - decode_time_at_max > 0.0:
            n_active_edf_margin_gt_batch += 1

    globals_vec = [
        _norm01(num_prefill, F_ACTIVE_PREFILL_COUNT_DEN),
        _norm01(num_decode, F_ACTIVE_DECODE_COUNT_DEN),
        _norm01(num_active, F_ACTIVE_TOTAL_COUNT_DEN),
        _norm01(total_remaining_prefill, F_TOTAL_REMAINING_PREFILL_DEN),
        _norm01(total_remaining_decode, F_TOTAL_REMAINING_DECODE_DEN),
        _norm01(total_decode_generated_active, F_TOTAL_DECODE_GENERATED_ACTIVE_DEN),
        _norm01(num_violated_active, F_VIOLATED_COUNT_DEN),
        _norm01(p_late_05_15, F_PREFILL_NEAR_DROP_DEN),
        _norm01(p_late_15, F_PREFILL_NEAR_DROP_DEN),
        _norm01(d_late_05_15, F_DECODE_NEAR_DROP_DEN),
        _norm01(d_late_15, F_DECODE_NEAR_DROP_DEN),
        _norm01(launch_count, F_RECENT_LAUNCH_COUNT_DEN),
        _norm01(launch_prefill, F_RECENT_LAUNCH_PREFILL_DEN),
        _norm01(remaining_launch_request_headroom, MAX_REQUESTS_PER_LAUNCH_WINDOW),
        _norm01(remaining_launch_prefill_headroom, MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW),
        _norm01(ewma, F_RECENT_LAUNCH_COUNT_DEN),
        _norm01(decode_credit, F_DECODE_CREDIT_DEN),
        _norm01(num_prefill_violated, F_ACTIVE_PREFILL_COUNT_DEN),
        _norm01(num_decode_violated, F_ACTIVE_DECODE_COUNT_DEN),
        edf_minus_batch_norm,
        _norm01(n_active_edf_margin_gt_batch, F_ACTIVE_TOTAL_COUNT_DEN),
    ]
    assert len(globals_vec) == D_GLOBAL

    # ------ prefill slots -------
    # Compute prefill_slack_clamped + lateness; sort.
    enriched_prefill = []
    for r in prefill_reqs:
        rid = _safe_int(r.get("id"), -1)
        arrived_at = _safe_float(r.get("arrived_at"))
        slo_t = _safe_float(r.get("prefill_slo_time"))
        deadline = arrived_at + slo_t
        slack = deadline - sim_time
        slack_clamped = max(0.0, slack)
        late_stored = _safe_float(per_req_prefill_late.get(rid, 0.0))
        late_now = max(0.0, sim_time - deadline)
        late = max(late_stored, late_now)
        enriched_prefill.append({
            "rid": rid,
            "remaining": max(0.0, _safe_float(r.get("remaining_prefill_tokens"))),
            "total": max(0.0, _safe_float(r.get("num_prefill_tokens"))),
            "violated": rid in violated_set,
            "slack_clamped": slack_clamped,
            "lateness": late,
        })

    all_violated = enriched_prefill and all(p["violated"] for p in enriched_prefill)
    if all_violated:
        enriched_prefill.sort(key=lambda p: p["rid"])
    else:
        enriched_prefill.sort(key=lambda p: (p["slack_clamped"], p["rid"]))

    selected_prefill = enriched_prefill[:N_PREFILL_SLOTS]

    prefill_block = np.zeros((N_PREFILL_SLOTS, D_PREFILL_PER_SLOT), dtype=np.float32)
    for i, p in enumerate(selected_prefill):
        # present, remaining_tokens (norm by 4096), total_tokens (norm by 4096),
        # violated_bit, lateness one-hot(4), slack one-hot(12).
        prefill_block[i, 0] = 1.0
        prefill_block[i, 1] = _norm01(p["remaining"], 4096.0)
        prefill_block[i, 2] = _norm01(p["total"], 4096.0)
        prefill_block[i, 3] = 1.0 if p["violated"] else 0.0
        # lateness one-hot
        lb = _lateness_bucket_idx(p["lateness"])
        prefill_block[i, 4 + lb] = 1.0
        # slack one-hot
        sb = _slack_bucket_idx(p["slack_clamped"])
        prefill_block[i, 4 + N_LATE_BUCKETS + sb] = 1.0

    # ------ decode slots -------
    # active not-yet-violated decode requests; deterministic random pick of 7 if more.
    non_violated_decode = [
        r for r in decode_reqs if _safe_int(r.get("id"), -1) not in violated_set
    ]
    # sort by id to make slicing consistent
    non_violated_decode.sort(key=lambda r: _safe_int(r.get("id"), -1))
    n_nv = len(non_violated_decode)
    if n_nv > N_DECODE_SLOTS:
        # deterministic seed: root_id + sim_time + active ids tuple
        seed_key = "|".join([
            str(record.get("root_id", -1)),
            f"{sim_time:.9f}",
            ",".join(str(_safe_int(r.get("id"), -1)) for r in non_violated_decode),
        ])
        idxs = _det_random_subset_indices(n_nv, N_DECODE_SLOTS, seed_key=seed_key)
        chosen = [non_violated_decode[i] for i in idxs]
    else:
        chosen = non_violated_decode

    decode_block = np.zeros((N_DECODE_SLOTS, D_DECODE_PER_SLOT), dtype=np.float32)
    for i, r in enumerate(chosen[:N_DECODE_SLOTS]):
        rem = _decode_remaining(r)
        done = _decode_done(r)
        decode_block[i, 0] = 1.0
        decode_block[i, 1] = _norm01(rem, DECODE_REMAINING_DEN)
        decode_block[i, 2] = 1.0 if done > 216.0 else 0.0
        decode_block[i, 3] = 1.0 if done > 512.0 else 0.0

    out = np.empty((D_TOTAL,), dtype=np.float32)
    out[:D_GLOBAL] = np.asarray(globals_vec, dtype=np.float32)
    out[D_GLOBAL:D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT] = prefill_block.reshape(-1)
    out[D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT:] = decode_block.reshape(-1)
    return out


def feature_names() -> list[str]:
    names: list[str] = []
    names.extend([
        "g05_num_prefill", "g06_num_decode", "g07_num_active",
        "g08_total_remaining_prefill", "g09_total_remaining_decode",
        "g10_total_decode_generated_active", "g11_num_violated_active",
        "g12_p_late_05_15", "g13_p_late_15",
        "g14_d_late_05_15", "g15_d_late_15",
        "g16_launch_count", "g17_launch_prefill",
        "g18_remaining_launch_request_headroom", "g19_remaining_launch_prefill_headroom",
        "g20_ewma", "g21_decode_credit",
        "g_extra_prefill_violated_active", "g_extra_decode_violated_active",
        "g_edf_minus_batch_norm",
        "g_n_active_edf_margin_gt_batch_norm",
    ])
    for i in range(N_PREFILL_SLOTS):
        names.append(f"pref{i}_present")
        names.append(f"pref{i}_remaining_norm")
        names.append(f"pref{i}_total_norm")
        names.append(f"pref{i}_violated")
        for b in range(N_LATE_BUCKETS):
            names.append(f"pref{i}_late_b{b}")
        for b in range(N_SLACK_BUCKETS):
            names.append(f"pref{i}_slack_b{b}")
    for i in range(N_DECODE_SLOTS):
        names.append(f"dec{i}_present")
        names.append(f"dec{i}_remaining_norm")
        names.append(f"dec{i}_done_gt_216")
        names.append(f"dec{i}_done_gt_512")
    assert len(names) == D_TOTAL, f"feature names {len(names)} != D_TOTAL {D_TOTAL}"
    return names


# ----------------------- shard processing -----------------------


def _process_shard(args: tuple[str, str, int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Process one shard.
    Args: (shard_path, root_player_filter, shard_start_idx, shard_end_idx)
    Returns: (shard_start_idx, n_kept, features array (n_kept x D_TOTAL), target_value array (n_kept,))
    """
    shard_path, root_player_filter, start_idx, end_idx = args
    sys.path.insert(0, str(REPO_ROOT))
    records = torch.load(shard_path, weights_only=False)
    feats = []
    targets = []
    keep_mask: list[bool] = []
    for rec in records:
        if root_player_filter and rec.get("root_player") != root_player_filter:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        feats.append(extract_features_one_record(rec))
        targets.append(_safe_float(rec.get("target_value", 0.0)))
    arr = np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, D_TOTAL), dtype=np.float32)
    tgt = np.asarray(targets, dtype=np.float32)
    return start_idx, len(feats), arr, tgt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="path to manifest.jsonl")
    parser.add_argument("--shard-dir", required=True, help="dir containing shards (relative paths in manifest)")
    parser.add_argument("--out-features", required=True)
    parser.add_argument("--out-targets", required=True)
    parser.add_argument("--out-meta", required=True)
    parser.add_argument("--root-player-filter", default="controller")
    parser.add_argument("--num-processes", type=int, default=32)
    parser.add_argument("--limit-shards", type=int, default=0, help="0=all, else limit to first N shards")
    args = parser.parse_args()

    manifest_lines = Path(args.manifest).read_text().splitlines()
    if args.limit_shards > 0:
        manifest_lines = manifest_lines[: args.limit_shards]

    shard_jobs: list[tuple[str, str, int, int]] = []
    cursor = 0
    expected_total = 0
    for line in manifest_lines:
        m = json.loads(line)
        n = int(m.get("num_records", 0))
        # we filter on player_counts.controller
        n_kept_in_shard = int((m.get("player_counts") or {}).get(args.root_player_filter, n))
        shard_path = str(Path(args.shard_dir) / m["shard_path"])
        shard_jobs.append((shard_path, args.root_player_filter, cursor, cursor + n_kept_in_shard))
        cursor += n_kept_in_shard
        expected_total += n_kept_in_shard

    print(f"[build] manifest={args.manifest} shards={len(shard_jobs)} expected_total_records={expected_total}")
    sys.stdout.flush()

    out_features = np.zeros((expected_total, D_TOTAL), dtype=np.float32)
    out_targets = np.zeros((expected_total,), dtype=np.float32)

    started = time.time()
    completed = 0
    last_log = started
    n_records_done = 0
    with ProcessPoolExecutor(max_workers=args.num_processes) as ex:
        futs = [ex.submit(_process_shard, job) for job in shard_jobs]
        for fut in as_completed(futs):
            start_idx, n_kept, arr, tgt = fut.result()
            out_features[start_idx:start_idx + n_kept] = arr
            out_targets[start_idx:start_idx + n_kept] = tgt
            n_records_done += n_kept
            completed += 1
            now = time.time()
            if now - last_log > 5.0 or completed == len(shard_jobs):
                rate = completed / max(1e-9, now - started)
                eta = (len(shard_jobs) - completed) / max(1e-9, rate)
                print(
                    f"[build] shards_done={completed}/{len(shard_jobs)} "
                    f"records={n_records_done}/{expected_total} "
                    f"rate_shards_s={rate:.2f} eta_s={eta:.1f}",
                    flush=True,
                )
                last_log = now

    out_features_path = Path(args.out_features)
    out_features_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_features_path, out_features)
    np.save(args.out_targets, out_targets)
    meta = {
        "manifest": args.manifest,
        "shard_dir": args.shard_dir,
        "root_player_filter": args.root_player_filter,
        "num_records": expected_total,
        "feature_dim": D_TOTAL,
        "feature_names": feature_names(),
        "n_global": D_GLOBAL,
        "n_prefill_slots": N_PREFILL_SLOTS,
        "d_prefill_per_slot": D_PREFILL_PER_SLOT,
        "n_decode_slots": N_DECODE_SLOTS,
        "d_decode_per_slot": D_DECODE_PER_SLOT,
        "slack_bucket_edges": PREFILL_SLACK_BUCKET_EDGES,
        "elapsed_s": time.time() - started,
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    print(f"[build] DONE features={out_features.shape} targets={out_targets.shape}")
    print(f"[build] features_path={out_features_path}")
    print(f"[build] targets_path={args.out_targets}")
    print(f"[build] meta_path={args.out_meta}")


if __name__ == "__main__":
    main()
