"""
Diagnostic CSVs for the worst eval-set predictions of a single trained V1 model.

Writes 5 files into the model's directory:

  bellman_outliers_inspection.csv
    one row per outlier — psid, y_true, y_pred, abs_err, locator (dataset/shard/in_shard_idx),
    plus a few summary fields from the snapshot.

  raw_state_global.csv
    one row per outlier — every global field from the raw snapshot/stats *before* normalisation.

  normalised_state_global.csv
    one row per outlier — the 19 normalised globals exactly as fed to the model
    (read straight out of features_d{1,2}.npy).

  raw_state_request.csv
    one row per (outlier, active request) — the raw per-request snapshot fields
    (prefill + decode requests, all unfiltered). Includes type, selected_slot_idx,
    selection_rank.

  normalised_state_request.csv
    one row per (outlier, slot) — the per-slot vector exactly as fed to the model.
    Slots 0..6 = prefill (20 dims each), 7..13 = decode (4 dims each). Padded slots
    are emitted with present=0 and zeros in the rest of the vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiments.new_features_v1.build_state_local_features import (  # noqa: E402
    D_DECODE_PER_SLOT,
    D_GLOBAL,
    D_PREFILL_PER_SLOT,
    D_TOTAL,
    F_ACTIVE_DECODE_COUNT_DEN,
    F_ACTIVE_PREFILL_COUNT_DEN,
    F_ACTIVE_TOTAL_COUNT_DEN,
    F_DECODE_CREDIT_DEN,
    F_DECODE_NEAR_DROP_DEN,
    F_LAUNCH_EWMA_ALPHA,
    F_LAUNCH_EWMA_WINDOW_SEC,
    F_NEAR_DROP_LATENESS_HIGH_SEC,
    F_NEAR_DROP_LATENESS_LOW_SEC,
    F_PREFILL_NEAR_DROP_DEN,
    F_RECENT_LAUNCH_COUNT_DEN,
    F_RECENT_LAUNCH_PREFILL_DEN,
    F_TOTAL_DECODE_GENERATED_ACTIVE_DEN,
    F_TOTAL_REMAINING_DECODE_DEN,
    F_TOTAL_REMAINING_PREFILL_DEN,
    F_VIOLATED_COUNT_DEN,
    MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW,
    MAX_REQUESTS_PER_LAUNCH_WINDOW,
    N_DECODE_SLOTS,
    N_PREFILL_SLOTS,
    PREFILL_SLACK_BUCKET_EDGES,
    _is_decode,
    _is_prefill,
    _safe_float,
    _safe_int,
    _stats_dict,
    _stats_set,
    feature_names,
)


def _load_manifest_offsets(manifest_path: Path, root_player_filter: str) -> list[tuple[int, int, str]]:
    """Return list of (start_psid, end_psid_exclusive, shard_path)."""
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for line in manifest_path.read_text().splitlines():
        m = json.loads(line)
        n_kept = int((m.get("player_counts") or {}).get(root_player_filter, m.get("num_records", 0)))
        out.append((cursor, cursor + n_kept, m["shard_path"]))
        cursor += n_kept
    return out


def _locate(combined_psid: int, n_d1: int, offsets_d1, offsets_d2) -> tuple[str, str, int]:
    """Map a combined psid to (dataset_tag, shard_relpath, in_shard_idx)."""
    if combined_psid < n_d1:
        local_psid = combined_psid
        offsets = offsets_d1
        tag = "d1"
    else:
        local_psid = combined_psid - n_d1
        offsets = offsets_d2
        tag = "d2"
    for start, end, shard in offsets:
        if start <= local_psid < end:
            return tag, shard, local_psid - start
    raise IndexError(f"psid {combined_psid} (local {local_psid}) not in any shard")


def _slack_bucket_idx(slack: float) -> int:
    if slack <= 0.0:
        return 0
    for i, edge in enumerate(PREFILL_SLACK_BUCKET_EDGES):
        if slack <= edge:
            return i + 1
    return len(PREFILL_SLACK_BUCKET_EDGES) + 1


def _lateness_bucket_idx(late: float) -> int:
    if late < 0.5:
        return 0
    if late < 1.0:
        return 1
    if late < 1.5:
        return 2
    return 3


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


def _per_request_raw_row(req: dict, sim_time: float, stats: Any, violated_set: set, type_tag: str) -> dict[str, Any]:
    rid = _safe_int(req.get("id"), -1)
    arrived_at = _safe_float(req.get("arrived_at"))
    queued_at = _safe_float(req.get("queued_at"), arrived_at)

    num_prefill_tokens = _safe_int(req.get("num_prefill_tokens"))
    remaining_prefill_tokens = _safe_int(req.get("remaining_prefill_tokens"))
    done_prefill_tokens = max(0, num_prefill_tokens - remaining_prefill_tokens)

    num_decode_tokens = _safe_int(req.get("num_decode_tokens"))
    num_processed_tokens = _safe_int(req.get("num_processed_tokens"))
    done_decode_tokens = max(0, num_processed_tokens - num_prefill_tokens)
    remaining_decode_tokens = max(0, num_decode_tokens - done_decode_tokens)

    prefill_slo_time = _safe_float(req.get("prefill_slo_time"))
    decode_slo_time = _safe_float(req.get("decode_slo_time"))
    completion_slo_time = _safe_float(req.get("completion_slo_time"))

    prefill_deadline = arrived_at + prefill_slo_time
    prefill_slack_sec = prefill_deadline - sim_time
    prefill_slack_clamped = max(0.0, prefill_slack_sec)

    per_req_pref_late = _stats_dict(stats, "per_request_prefill_lateness")
    per_req_dec_late = _stats_dict(stats, "per_request_decode_lateness")
    decode_deadlines = _stats_dict(stats, "decode_next_deadline_by_id")

    stored_prefill_late = _safe_float(per_req_pref_late.get(rid, 0.0))
    current_prefill_late = max(0.0, sim_time - prefill_deadline) if not bool(req.get("is_prefill_complete")) else 0.0
    prefill_lateness_sec = max(stored_prefill_late, current_prefill_late)

    stored_decode_late = _safe_float(per_req_dec_late.get(rid, 0.0))
    decode_deadline = _safe_float(decode_deadlines.get(rid, arrived_at + decode_slo_time))
    current_decode_late = max(0.0, sim_time - decode_deadline) if bool(req.get("is_prefill_complete")) else 0.0
    decode_lateness_sec = max(stored_decode_late, current_decode_late)

    return {
        "rid": rid,
        "type": type_tag,
        "arrived_at": arrived_at,
        "queued_at": queued_at,
        "sim_time": sim_time,
        "num_prefill_tokens": num_prefill_tokens,
        "remaining_prefill_tokens": remaining_prefill_tokens,
        "done_prefill_tokens": done_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
        "num_processed_tokens": num_processed_tokens,
        "done_decode_tokens": done_decode_tokens,
        "remaining_decode_tokens": remaining_decode_tokens,
        "prefill_slo_time": prefill_slo_time,
        "decode_slo_time": decode_slo_time,
        "completion_slo_time": completion_slo_time,
        "prefill_deadline": prefill_deadline,
        "prefill_slack_sec": prefill_slack_sec,
        "prefill_slack_clamped": prefill_slack_clamped,
        "prefill_lateness_sec": prefill_lateness_sec,
        "decode_deadline": decode_deadline,
        "decode_lateness_sec": decode_lateness_sec,
        "scheduled": int(bool(req.get("scheduled"))),
        "preempted": int(bool(req.get("preempted"))),
        "completed": int(bool(req.get("completed"))),
        "is_prefill_complete": int(bool(req.get("is_prefill_complete"))),
        "num_restarts": _safe_int(req.get("num_restarts"), 0),
        "violated": int(rid in violated_set),
        "prefill_late_bucket": _lateness_bucket_idx(prefill_lateness_sec),
        "prefill_slack_bucket": _slack_bucket_idx(prefill_slack_clamped),
    }


def _raw_globals(snapshot: dict, stats: Any) -> dict[str, Any]:
    sim_time = _safe_float(snapshot.get("time"))
    request_states = list((snapshot.get("request_states") or {}).values())
    active_id_set: set[int] = _stats_set(stats, "active_request_ids")
    if active_id_set:
        active_requests = [r for r in request_states if isinstance(r, dict) and _safe_int(r.get("id"), -1) in active_id_set]
    else:
        active_requests = [r for r in request_states if isinstance(r, dict)]

    violated_set = _stats_set(stats, "violated_request_ids")
    per_req_pref_late = _stats_dict(stats, "per_request_prefill_lateness")
    per_req_dec_late = _stats_dict(stats, "per_request_decode_lateness")

    prefill_reqs = [r for r in active_requests if _is_prefill(r)]
    decode_reqs = [r for r in active_requests if _is_decode(r)]

    num_prefill = len(prefill_reqs)
    num_decode = len(decode_reqs)
    num_active = len(active_requests)

    total_remaining_prefill = sum(max(0.0, _safe_float(r.get("remaining_prefill_tokens"))) for r in prefill_reqs)
    total_remaining_decode = sum(_decode_remaining(r) for r in decode_reqs)
    total_decode_generated_active = sum(_decode_done(r) for r in decode_reqs)

    active_ids = {_safe_int(r.get("id"), -1) for r in active_requests}
    num_violated_active = len(active_ids & violated_set)
    num_prefill_violated = sum(1 for r in prefill_reqs if _safe_int(r.get("id"), -1) in violated_set)
    num_decode_violated = sum(1 for r in decode_reqs if _safe_int(r.get("id"), -1) in violated_set)

    p_late_05_15 = 0
    p_late_15 = 0
    for r in prefill_reqs:
        rid = _safe_int(r.get("id"), -1)
        late = _safe_float(per_req_pref_late.get(rid, 0.0))
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
        late = max(_safe_float(per_req_pref_late.get(rid, 0.0)), _safe_float(per_req_dec_late.get(rid, 0.0)))
        if late > F_NEAR_DROP_LATENESS_LOW_SEC and late < F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_05_15 += 1
        elif late >= F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_15 += 1

    launch_count = 0.0
    launch_prefill = 0.0
    ewma = 0.0
    n_recent_arrival_events = 0
    for item in (getattr(stats, "recent_arrivals", None) or []):
        n_recent_arrival_events += 1
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

    decode_tokens_counted = _stats_dict(stats, "decode_tokens_counted")
    META_DECODE_CREDIT_BAL = -9_100_005
    decode_credit = max(0.0, _safe_float(decode_tokens_counted.get(META_DECODE_CREDIT_BAL, 0.0)))

    return {
        "sim_time": sim_time,
        "num_prefill": num_prefill,
        "num_decode": num_decode,
        "num_active": num_active,
        "total_remaining_prefill": total_remaining_prefill,
        "total_remaining_decode": total_remaining_decode,
        "total_decode_generated_active": total_decode_generated_active,
        "num_violated_active": num_violated_active,
        "num_prefill_violated_active": num_prefill_violated,
        "num_decode_violated_active": num_decode_violated,
        "p_late_05_15": p_late_05_15,
        "p_late_15": p_late_15,
        "d_late_05_15": d_late_05_15,
        "d_late_15": d_late_15,
        "launch_count": launch_count,
        "launch_prefill": launch_prefill,
        "remaining_launch_request_headroom": remaining_launch_request_headroom,
        "remaining_launch_prefill_headroom": remaining_launch_prefill_headroom,
        "ewma_launch": ewma,
        "decode_credit": decode_credit,
        "n_recent_arrival_events": n_recent_arrival_events,
        # constants used to interpret the normalised values
        "den_active_prefill_count": F_ACTIVE_PREFILL_COUNT_DEN,
        "den_active_decode_count": F_ACTIVE_DECODE_COUNT_DEN,
        "den_active_total_count": F_ACTIVE_TOTAL_COUNT_DEN,
        "den_total_remaining_prefill": F_TOTAL_REMAINING_PREFILL_DEN,
        "den_total_remaining_decode": F_TOTAL_REMAINING_DECODE_DEN,
        "den_total_decode_generated_active": F_TOTAL_DECODE_GENERATED_ACTIVE_DEN,
        "den_violated_count": F_VIOLATED_COUNT_DEN,
        "den_prefill_near_drop": F_PREFILL_NEAR_DROP_DEN,
        "den_decode_near_drop": F_DECODE_NEAR_DROP_DEN,
        "den_recent_launch_count": F_RECENT_LAUNCH_COUNT_DEN,
        "den_recent_launch_prefill": F_RECENT_LAUNCH_PREFILL_DEN,
        "den_decode_credit": F_DECODE_CREDIT_DEN,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="path to model dir containing eval_results.csv")
    parser.add_argument("--features-d1", required=True)
    parser.add_argument("--features-d2", required=True)
    parser.add_argument("--manifest-d1", required=True)
    parser.add_argument("--shard-dir-d1", required=True)
    parser.add_argument("--manifest-d2", required=True)
    parser.add_argument("--shard-dir-d2", required=True)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--root-player-filter", default="controller")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    eval_csv = model_dir / "eval_results.csv"
    print(f"[inspect] reading {eval_csv}")
    rows: list[dict[str, Any]] = []
    with open(eval_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"psid": int(row["psid"]), "y_true": float(row["y_true"]),
                         "y_pred": float(row["y_pred"]), "abs_err": float(row["abs_err"])})
    rows.sort(key=lambda d: -d["abs_err"])
    top = rows[: args.top_n]
    print(f"[inspect] picked top {len(top)} outliers; max abs_err={top[0]['abs_err']:.4f}")

    print("[inspect] loading combined feature arrays…")
    feats_d1 = np.load(args.features_d1, mmap_mode="r")
    feats_d2 = np.load(args.features_d2, mmap_mode="r")
    n_d1 = int(feats_d1.shape[0])

    print("[inspect] indexing manifests…")
    offsets_d1 = _load_manifest_offsets(Path(args.manifest_d1), args.root_player_filter)
    offsets_d2 = _load_manifest_offsets(Path(args.manifest_d2), args.root_player_filter)

    # group outliers by shard so we load each shard at most once
    by_shard: dict[tuple[str, str], list[tuple[int, int, dict]]] = {}
    located: list[tuple[str, str, int]] = []
    for o in top:
        tag, shard, idx = _locate(o["psid"], n_d1, offsets_d1, offsets_d2)
        located.append((tag, shard, idx))
        by_shard.setdefault((tag, shard), []).append((idx, o["psid"], o))

    fnames = feature_names()
    g_names = fnames[:D_GLOBAL]
    pref_per_slot_names = fnames[D_GLOBAL : D_GLOBAL + D_PREFILL_PER_SLOT]
    dec_per_slot_names = fnames[
        D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT :
        D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT + D_DECODE_PER_SLOT
    ]
    pref_per_slot_names = [n.replace("pref0_", "") for n in pref_per_slot_names]
    dec_per_slot_names = [n.replace("dec0_", "") for n in dec_per_slot_names]

    inspect_rows: list[dict[str, Any]] = []
    raw_global_rows: list[dict[str, Any]] = []
    norm_global_rows: list[dict[str, Any]] = []
    raw_request_rows: list[dict[str, Any]] = []
    norm_request_rows: list[dict[str, Any]] = []

    for (tag, shard), items in by_shard.items():
        shard_dir = Path(args.shard_dir_d1 if tag == "d1" else args.shard_dir_d2)
        shard_path = shard_dir / shard
        print(f"[inspect] loading shard {tag}/{shard} for {len(items)} outliers")
        records = torch.load(shard_path, weights_only=False)

        for in_shard_idx, psid, o in items:
            rec = records[in_shard_idx]
            snapshot = rec.get("simulator_snapshot") or {}
            stats = rec.get("stats")

            raw_g = _raw_globals(snapshot, stats)
            inspect_rows.append({
                "psid": psid, "y_true": o["y_true"], "y_pred": o["y_pred"], "abs_err": o["abs_err"],
                "dataset": tag, "shard_path": shard, "in_shard_idx": in_shard_idx,
                "root_id": rec.get("root_id"), "root_player": rec.get("root_player"),
                "root_depth": rec.get("root_depth", 0), "history_hops": rec.get("history_hops", 0),
                "best_action_index": rec.get("best_action_index"),
                "best_action_repr": rec.get("best_action_repr"),
                "best_reward": rec.get("best_reward"),
                "best_discount": rec.get("best_discount"),
                "best_bootstrap": rec.get("best_bootstrap"),
                "best_child_cost": rec.get("best_child_cost"),
                "best_child_time": rec.get("best_child_time"),
                "target_value": rec.get("target_value"),
                "is_nonzero_target": rec.get("is_nonzero_target"),
                "sim_time": raw_g["sim_time"],
                "num_active": raw_g["num_active"],
                "num_prefill": raw_g["num_prefill"],
                "num_decode": raw_g["num_decode"],
                "num_violated_active": raw_g["num_violated_active"],
            })
            raw_global_rows.append({"psid": psid, "y_true": o["y_true"], "y_pred": o["y_pred"], "abs_err": o["abs_err"], **raw_g})

            # normalised globals from the cached feature array
            arr = feats_d1 if tag == "d1" else feats_d2
            local = psid if tag == "d1" else psid - n_d1
            vec = np.asarray(arr[local])
            norm_g_row = {"psid": psid, "y_true": o["y_true"], "y_pred": o["y_pred"], "abs_err": o["abs_err"]}
            for i, name in enumerate(g_names):
                norm_g_row[name] = float(vec[i])
            norm_global_rows.append(norm_g_row)

            # raw per-request rows
            request_states = list((snapshot.get("request_states") or {}).values())
            active_id_set: set[int] = _stats_set(stats, "active_request_ids")
            if active_id_set:
                active_requests = [r for r in request_states if isinstance(r, dict) and _safe_int(r.get("id"), -1) in active_id_set]
            else:
                active_requests = [r for r in request_states if isinstance(r, dict)]
            violated_set = _stats_set(stats, "violated_request_ids")

            # which prefill requests were selected into slots? recompute the same sort.
            prefill_reqs = [r for r in active_requests if _is_prefill(r)]
            enriched = []
            for r in prefill_reqs:
                rid = _safe_int(r.get("id"), -1)
                arrived = _safe_float(r.get("arrived_at"))
                slo = _safe_float(r.get("prefill_slo_time"))
                slack = max(0.0, (arrived + slo) - raw_g["sim_time"])
                viol = rid in violated_set
                enriched.append((slack, rid, viol, r))
            all_violated = enriched and all(e[2] for e in enriched)
            if all_violated:
                enriched.sort(key=lambda e: e[1])
            else:
                enriched.sort(key=lambda e: (e[0], e[1]))
            selected_prefill_rids = [e[1] for e in enriched[:N_PREFILL_SLOTS]]
            prefill_rid_to_slot = {rid: i for i, rid in enumerate(selected_prefill_rids)}

            # decode selection (deterministic random)
            decode_reqs = [r for r in active_requests if _is_decode(r)]
            non_violated_decode = [r for r in decode_reqs if _safe_int(r.get("id"), -1) not in violated_set]
            non_violated_decode.sort(key=lambda r: _safe_int(r.get("id"), -1))
            decode_rid_to_slot: dict[int, int] = {}
            n_nv = len(non_violated_decode)
            if n_nv > N_DECODE_SLOTS:
                from experiments.new_features_v1.build_state_local_features import _det_random_subset_indices
                seed_key = "|".join([
                    str(rec.get("root_id", -1)),
                    f"{raw_g['sim_time']:.9f}",
                    ",".join(str(_safe_int(r.get("id"), -1)) for r in non_violated_decode),
                ])
                idxs = _det_random_subset_indices(n_nv, N_DECODE_SLOTS, seed_key=seed_key)
                for slot_i, src_i in enumerate(idxs):
                    decode_rid_to_slot[_safe_int(non_violated_decode[src_i].get("id"), -1)] = slot_i
            else:
                for slot_i, r in enumerate(non_violated_decode[:N_DECODE_SLOTS]):
                    decode_rid_to_slot[_safe_int(r.get("id"), -1)] = slot_i

            for r in active_requests:
                rid = _safe_int(r.get("id"), -1)
                if _is_prefill(r):
                    type_tag = "prefill"
                elif _is_decode(r):
                    type_tag = "decode"
                else:
                    type_tag = "other"
                row = _per_request_raw_row(r, raw_g["sim_time"], stats, violated_set, type_tag)
                slot_idx = -1
                if type_tag == "prefill":
                    slot_idx = prefill_rid_to_slot.get(rid, -1)
                elif type_tag == "decode":
                    slot_idx = decode_rid_to_slot.get(rid, -1)
                raw_request_rows.append({
                    "psid": psid, "y_true": o["y_true"], "y_pred": o["y_pred"], "abs_err": o["abs_err"],
                    "selected_slot_idx": slot_idx, **row,
                })

            # normalised per-slot rows
            for slot_i in range(N_PREFILL_SLOTS):
                base = D_GLOBAL + slot_i * D_PREFILL_PER_SLOT
                slot_vec = vec[base : base + D_PREFILL_PER_SLOT]
                row = {
                    "psid": psid, "y_true": o["y_true"], "y_pred": o["y_pred"], "abs_err": o["abs_err"],
                    "slot_type": "prefill", "slot_idx": slot_i,
                }
                for i, name in enumerate(pref_per_slot_names):
                    row[name] = float(slot_vec[i])
                norm_request_rows.append(row)
            for slot_i in range(N_DECODE_SLOTS):
                base = D_GLOBAL + N_PREFILL_SLOTS * D_PREFILL_PER_SLOT + slot_i * D_DECODE_PER_SLOT
                slot_vec = vec[base : base + D_DECODE_PER_SLOT]
                row = {
                    "psid": psid, "y_true": o["y_true"], "y_pred": o["y_pred"], "abs_err": o["abs_err"],
                    "slot_type": "decode", "slot_idx": slot_i,
                }
                for i, name in enumerate(dec_per_slot_names):
                    row[name] = float(slot_vec[i])
                # pad zeros for unused 16 dims so it lines up cell-wise with prefill rows? No — keep narrow.
                norm_request_rows.append(row)

    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            print(f"[inspect] no rows for {path}")
            return
        # union of keys (preserve insertion order from first row)
        keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                # fill missing keys with empty
                for k in keys:
                    if k not in r:
                        r[k] = ""
                w.writerow(r)
        print(f"[inspect] wrote {path} ({len(rows)} rows)")

    _write_csv(model_dir / "bellman_outliers_inspection.csv", inspect_rows)
    _write_csv(model_dir / "raw_state_global.csv", raw_global_rows)
    _write_csv(model_dir / "normalised_state_global.csv", norm_global_rows)
    _write_csv(model_dir / "raw_state_request.csv", raw_request_rows)
    _write_csv(model_dir / "normalised_state_request.csv", norm_request_rows)


if __name__ == "__main__":
    main()
