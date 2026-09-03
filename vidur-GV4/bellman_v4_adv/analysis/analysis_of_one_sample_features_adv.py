#!/usr/bin/env python3
"""
Inspect the v4-Adv state-local feature vector for one stored controller root sample.

Default command:

PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur-classical-search \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  -m vidur.bellman_v4_adv.analysis.analysis_of_one_sample_features_adv \
  --seed 12345

Default output directory:

/home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/GV3_Agent/feature_analysis/bellman_v4_adv_new_features
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from vidur.bellman_v4_adv.build_state_local_features_adv import (  # noqa: E402
    D_GLOBAL,
    D_TOTAL,
    DECODE_REMAINING_DEN,
    F_ACTIVE_DECODE_COUNT_DEN,
    F_ACTIVE_PREFILL_COUNT_DEN,
    F_ACTIVE_TOTAL_COUNT_DEN,
    F_ADV_TICK_SEC_DEN,
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
    _decode_time_for_tokens,
    _decode_time_using_simulator,
    _det_random_subset_indices,
    _is_decode,
    _is_prefill,
    _lateness_bucket_idx,
    _safe_float,
    _safe_int,
    _slack_bucket_idx,
    _stats_attr,
    _stats_dict,
    _stats_set,
    extract_features_one_record,
    feature_names,
)


CLASSICAL_DATASET_DIR = (
    REPO_ROOT
    / "simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40_repaired_stats_v1_mp20"
)
ORIGINAL_DATASET_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur/"
    "simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40_repaired_stats_v1_mp20"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "simulator_output/GV3_Agent/feature_analysis/bellman_v4_adv_new_features"
)


def _default_dataset_dir() -> Path:
    if (CLASSICAL_DATASET_DIR / "manifest.jsonl").exists():
        return CLASSICAL_DATASET_DIR
    return ORIGINAL_DATASET_DIR


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    return rows


def _controller_count(row: dict[str, Any], root_player_filter: str) -> int:
    player_counts = row.get("player_counts") or {}
    return int(player_counts.get(root_player_filter, row.get("num_records", 0)))


def load_sample(
    *,
    manifest_path: Path,
    shard_dir: Path,
    root_player_filter: str,
    seed: int,
    sample_index: int | None,
) -> tuple[int, str, int, dict[str, Any]]:
    """Load one filtered sample by global filtered index."""

    manifest = _read_manifest(manifest_path)
    total = sum(_controller_count(row, root_player_filter) for row in manifest)
    if total <= 0:
        raise ValueError(f"no {root_player_filter!r} rows in {manifest_path}")

    if sample_index is None:
        rng = random.Random(int(seed))
        global_index = rng.randrange(total)
    else:
        global_index = int(sample_index)
        if global_index < 0 or global_index >= total:
            raise ValueError(f"sample_index={global_index} outside [0, {total})")

    cursor = 0
    for row in manifest:
        n = _controller_count(row, root_player_filter)
        if global_index >= cursor + n:
            cursor += n
            continue

        shard_path = shard_dir / str(row["shard_path"])
        records = torch.load(shard_path, weights_only=False)
        filtered = [
            rec
            for rec in records
            if not root_player_filter or rec.get("root_player") == root_player_filter
        ]
        local_filtered_index = global_index - cursor
        if local_filtered_index >= len(filtered):
            raise ValueError(
                f"manifest count mismatch for {shard_path}: "
                f"wanted local index {local_filtered_index}, found {len(filtered)}"
            )
        return global_index, str(row["shard_path"]), int(local_filtered_index), filtered[local_filtered_index]

    raise RuntimeError("sample selection fell through manifest")


def _decode_remaining(req: dict[str, Any]) -> float:
    total_dec = max(0.0, _safe_float(req.get("num_decode_tokens")))
    proc_total = max(0.0, _safe_float(req.get("num_processed_tokens")))
    total_pref = max(0.0, _safe_float(req.get("num_prefill_tokens")))
    done_decode = max(0.0, proc_total - total_pref)
    return max(0.0, total_dec - done_decode)


def _decode_done(req: dict[str, Any]) -> float:
    proc_total = max(0.0, _safe_float(req.get("num_processed_tokens")))
    total_pref = max(0.0, _safe_float(req.get("num_prefill_tokens")))
    return max(0.0, proc_total - total_pref)


def _raw_feature_context(record: dict[str, Any]) -> dict[str, Any]:
    """Compute human-readable raw values matching the v4 feature layout."""

    snapshot = record.get("simulator_snapshot") or {}
    stats = record.get("stats")
    sim_time = _safe_float(snapshot.get("time"))
    request_states = list((snapshot.get("request_states") or {}).values())
    active_id_set = _stats_set(stats, "active_request_ids")
    active_requests = [
        r
        for r in request_states
        if isinstance(r, dict) and _safe_int(r.get("id"), -1) in active_id_set
    ]

    violated_set = _stats_set(stats, "violated_request_ids")
    per_req_prefill_late = _stats_dict(stats, "per_request_prefill_lateness")
    per_req_decode_late = _stats_dict(stats, "per_request_decode_lateness")
    decode_deadlines = _stats_dict(stats, "decode_next_deadline_by_id")

    prefill_reqs = [r for r in active_requests if _is_prefill(r)]
    decode_reqs = [r for r in active_requests if _is_decode(r)]
    active_ids = {_safe_int(r.get("id"), -1) for r in active_requests}

    total_remaining_prefill = sum(
        max(0.0, _safe_float(r.get("remaining_prefill_tokens"))) for r in prefill_reqs
    )
    total_remaining_decode = sum(_decode_remaining(r) for r in decode_reqs)
    total_decode_generated_active = sum(
        max(0.0, _safe_float(r.get("num_processed_decode_tokens")))
        if "num_processed_decode_tokens" in r
        else _decode_done(r)
        for r in decode_reqs
    )

    p_late_05_15 = 0
    p_late_15 = 0
    for r in prefill_reqs:
        rid = _safe_int(r.get("id"), -1)
        late = _safe_float(per_req_prefill_late.get(rid, 0.0))
        if late <= 0.0:
            late = max(
                0.0,
                sim_time
                - (_safe_float(r.get("arrived_at")) + _safe_float(r.get("prefill_slo_time"))),
            )
        if F_NEAR_DROP_LATENESS_LOW_SEC < late < F_NEAR_DROP_LATENESS_HIGH_SEC:
            p_late_05_15 += 1
        elif late >= F_NEAR_DROP_LATENESS_HIGH_SEC:
            p_late_15 += 1

    d_late_05_15 = 0
    d_late_15 = 0
    for r in decode_reqs:
        rid = _safe_int(r.get("id"), -1)
        late = max(
            _safe_float(per_req_prefill_late.get(rid, 0.0)),
            _safe_float(per_req_decode_late.get(rid, 0.0)),
        )
        if F_NEAR_DROP_LATENESS_LOW_SEC < late < F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_05_15 += 1
        elif late >= F_NEAR_DROP_LATENESS_HIGH_SEC:
            d_late_15 += 1

    META_NEXT_ADV_TICK = -9_100_001
    next_adv_tick = _safe_float(decode_deadlines.get(META_NEXT_ADV_TICK, sim_time), sim_time)
    delta_next_adv_tick = max(0.0, next_adv_tick - sim_time)

    launch_count = 0.0
    launch_prefill = 0.0
    ewma = 0.0
    latest_adv_launch_time: float | None = None
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
        latest_adv_launch_time = ts if latest_adv_launch_time is None else max(latest_adv_launch_time, ts)
        launch_count += max(0.0, float(cnt))
        launch_prefill += max(0.0, float(prefill))
        ewma += max(0.0, float(cnt)) * math.exp(-F_LAUNCH_EWMA_ALPHA * dt)

    delta_since_last_adv_launch = (
        F_LAUNCH_EWMA_WINDOW_SEC
        if latest_adv_launch_time is None
        else max(0.0, sim_time - latest_adv_launch_time)
    )

    decode_tokens_counted = _stats_dict(stats, "decode_tokens_counted")
    decode_credit = max(0.0, _safe_float(decode_tokens_counted.get(-9_100_005, 0.0)))

    if prefill_reqs:
        min_prefill_slack = min(
            (_safe_float(r.get("arrived_at")) + _safe_float(r.get("prefill_slo_time"))) - sim_time
            for r in prefill_reqs
        )
    else:
        min_prefill_slack = float("inf")

    if decode_reqs:
        max_decode_context_tokens = max(
            max(0, _safe_int(r.get("num_prefill_tokens")))
            + max(0, _safe_int(r.get("num_processed_tokens")) - _safe_int(r.get("num_prefill_tokens")))
            for r in decode_reqs
        )
    else:
        max_decode_context_tokens = 0
    decode_time_at_max = _decode_time_using_simulator(record, decode_reqs)

    n_active_edf_margin_gt_batch = 0
    for r in prefill_reqs:
        deadline = _safe_float(r.get("arrived_at")) + _safe_float(r.get("prefill_slo_time"))
        if (deadline - sim_time) - decode_time_at_max > 0.0:
            n_active_edf_margin_gt_batch += 1
    for r in decode_reqs:
        rid = _safe_int(r.get("id"), -1)
        deadline = _safe_float(
            decode_deadlines.get(
                rid,
                _safe_float(r.get("arrived_at")) + _safe_float(r.get("decode_slo_time")),
            )
        )
        if (deadline - sim_time) - decode_time_at_max > 0.0:
            n_active_edf_margin_gt_batch += 1

    raw_globals = {
        "g05_num_prefill": f"{len(prefill_reqs)} / {F_ACTIVE_PREFILL_COUNT_DEN}",
        "g06_num_decode": f"{len(decode_reqs)} / {F_ACTIVE_DECODE_COUNT_DEN}",
        "g07_num_active": f"{len(active_requests)} / {F_ACTIVE_TOTAL_COUNT_DEN}",
        "g08_total_remaining_prefill": f"{total_remaining_prefill} / {F_TOTAL_REMAINING_PREFILL_DEN}",
        "g09_total_remaining_decode": f"{total_remaining_decode} / {F_TOTAL_REMAINING_DECODE_DEN}",
        "g10_total_decode_generated_active": f"{total_decode_generated_active} / {F_TOTAL_DECODE_GENERATED_ACTIVE_DEN}",
        "g11_num_violated_active": f"{len(active_ids & violated_set)} / {F_VIOLATED_COUNT_DEN}",
        "g12_p_late_05_15": f"{p_late_05_15} / {F_PREFILL_NEAR_DROP_DEN}",
        "g13_p_late_15": f"{p_late_15} / {F_PREFILL_NEAR_DROP_DEN}",
        "g14_d_late_05_15": f"{d_late_05_15} / {F_DECODE_NEAR_DROP_DEN}",
        "g15_d_late_15": f"{d_late_15} / {F_DECODE_NEAR_DROP_DEN}",
        "g16_launch_count": f"{launch_count} / {F_RECENT_LAUNCH_COUNT_DEN}",
        "g17_launch_prefill": f"{launch_prefill} / {F_RECENT_LAUNCH_PREFILL_DEN}",
        "g18_remaining_launch_request_headroom": f"{max(0.0, MAX_REQUESTS_PER_LAUNCH_WINDOW - launch_count)} / {MAX_REQUESTS_PER_LAUNCH_WINDOW}",
        "g19_remaining_launch_prefill_headroom": f"{max(0.0, MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW - launch_prefill)} / {MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW}",
        "g20_ewma": f"{ewma} / {F_RECENT_LAUNCH_COUNT_DEN}",
        "g21_decode_credit": f"{decode_credit} / {F_DECODE_CREDIT_DEN}",
        "g_extra_prefill_violated_active": f"{sum(_safe_int(r.get('id'), -1) in violated_set for r in prefill_reqs)} / {F_ACTIVE_PREFILL_COUNT_DEN}",
        "g_extra_decode_violated_active": f"{sum(_safe_int(r.get('id'), -1) in violated_set for r in decode_reqs)} / {F_ACTIVE_DECODE_COUNT_DEN}",
        "g_edf_minus_batch_norm": (
            f"min_prefill_slack={min_prefill_slack}, "
            f"max_decode_context_tokens={max_decode_context_tokens}, "
            f"simulator_decode_only_batch_time={decode_time_at_max}"
        ),
        "g_n_active_edf_margin_gt_batch_norm": f"{n_active_edf_margin_gt_batch} / {F_ACTIVE_TOTAL_COUNT_DEN}",
        "g_delta_next_adv_tick_norm": (
            f"next_adv_tick={next_adv_tick}; sim_time={sim_time}; "
            f"delta={delta_next_adv_tick} / {F_ADV_TICK_SEC_DEN}"
        ),
        "g_delta_since_last_adv_launch_norm": (
            f"latest_recent_adv_launch_time={latest_adv_launch_time}; sim_time={sim_time}; "
            f"delta={delta_since_last_adv_launch} / {F_LAUNCH_EWMA_WINDOW_SEC}"
        ),
    }

    enriched_prefill = []
    for r in prefill_reqs:
        rid = _safe_int(r.get("id"), -1)
        deadline = _safe_float(r.get("arrived_at")) + _safe_float(r.get("prefill_slo_time"))
        slack = deadline - sim_time
        slack_clamped = max(0.0, slack)
        lateness = max(
            _safe_float(per_req_prefill_late.get(rid, 0.0)),
            max(0.0, sim_time - deadline),
        )
        enriched_prefill.append(
            {
                "rid": rid,
                "remaining": max(0.0, _safe_float(r.get("remaining_prefill_tokens"))),
                "total": max(0.0, _safe_float(r.get("num_prefill_tokens"))),
                "violated": rid in violated_set,
                "slack_sec": slack,
                "slack_clamped": slack_clamped,
                "slack_bucket": _slack_bucket_idx(slack_clamped),
                "lateness_sec": lateness,
                "lateness_bucket": _lateness_bucket_idx(lateness),
            }
        )
    if enriched_prefill and all(p["violated"] for p in enriched_prefill):
        enriched_prefill.sort(key=lambda p: p["rid"])
    else:
        enriched_prefill.sort(key=lambda p: (p["slack_clamped"], p["rid"]))

    non_violated_decode = [
        r for r in decode_reqs if _safe_int(r.get("id"), -1) not in violated_set
    ]
    non_violated_decode.sort(key=lambda r: _safe_int(r.get("id"), -1))
    if len(non_violated_decode) > N_DECODE_SLOTS:
        seed_key = "|".join(
            [
                str(record.get("root_id", -1)),
                f"{sim_time:.9f}",
                ",".join(str(_safe_int(r.get("id"), -1)) for r in non_violated_decode),
            ]
        )
        idxs = _det_random_subset_indices(len(non_violated_decode), N_DECODE_SLOTS, seed_key=seed_key)
        selected_decode = [non_violated_decode[i] for i in idxs]
    else:
        selected_decode = non_violated_decode

    decode_slots = []
    for r in selected_decode[:N_DECODE_SLOTS]:
        decode_slots.append(
            {
                "rid": _safe_int(r.get("id"), -1),
                "remaining": _decode_remaining(r),
                "done": _decode_done(r),
                "total_decode": _safe_float(r.get("num_decode_tokens")),
            }
        )

    return {
        "raw_globals": raw_globals,
        "prefill_slots": enriched_prefill[:N_PREFILL_SLOTS],
        "decode_slots": decode_slots,
        "sim_time": sim_time,
        "num_active": len(active_requests),
        "num_prefill": len(prefill_reqs),
        "num_decode": len(decode_reqs),
    }


def describe_feature(name: str, context: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return (group, slot, kind, raw_value_note)."""

    raw_globals = context["raw_globals"]
    if name in raw_globals:
        return "global", "", name, raw_globals[name]

    pref_match = re.fullmatch(r"pref(\d+)_(.+)", name)
    if pref_match:
        slot = int(pref_match.group(1))
        kind = pref_match.group(2)
        slots = context["prefill_slots"]
        if slot >= len(slots):
            return "prefill_slot", str(slot), kind, "padding/no selected prefill request"
        p = slots[slot]
        base = f"request_id={p['rid']}"
        if kind == "present":
            return "prefill_slot", str(slot), kind, base
        if kind == "remaining_norm":
            return "prefill_slot", str(slot), kind, f"{base}; remaining_tokens={p['remaining']} / 4096"
        if kind == "total_norm":
            return "prefill_slot", str(slot), kind, f"{base}; total_prefill_tokens={p['total']} / 4096"
        if kind == "violated":
            return "prefill_slot", str(slot), kind, f"{base}; violated={p['violated']}"
        if kind.startswith("late_b"):
            return (
                "prefill_slot",
                str(slot),
                kind,
                f"{base}; lateness_sec={p['lateness_sec']}; selected_bucket={p['lateness_bucket']}",
            )
        if kind.startswith("slack_b"):
            return (
                "prefill_slot",
                str(slot),
                kind,
                f"{base}; slack_sec={p['slack_sec']}; slack_clamped={p['slack_clamped']}; "
                f"selected_bucket={p['slack_bucket']}",
            )
        return "prefill_slot", str(slot), kind, base

    dec_match = re.fullmatch(r"dec(\d+)_(.+)", name)
    if dec_match:
        slot = int(dec_match.group(1))
        kind = dec_match.group(2)
        slots = context["decode_slots"]
        if slot >= len(slots):
            return "decode_slot", str(slot), kind, "padding/no selected decode request"
        d = slots[slot]
        base = f"request_id={d['rid']}"
        if kind == "present":
            return "decode_slot", str(slot), kind, base
        if kind == "remaining_norm":
            return "decode_slot", str(slot), kind, f"{base}; remaining_decode_tokens={d['remaining']} / {DECODE_REMAINING_DEN}"
        if kind == "done_gt_216":
            return "decode_slot", str(slot), kind, f"{base}; done_decode_tokens={d['done']} > 216"
        if kind == "done_gt_512":
            return "decode_slot", str(slot), kind, f"{base}; done_decode_tokens={d['done']} > 512"
        return "decode_slot", str(slot), kind, base

    return "unknown", "", name, ""


def write_feature_csv(path: Path, record: dict[str, Any], global_index: int, features) -> None:
    names = feature_names()
    context = _raw_feature_context(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "root_id",
                "target_value",
                "feature_index",
                "feature_name",
                "feature_group",
                "slot",
                "feature_kind",
                "raw_feature_value_in_simulator",
                "feature_created_for_model",
            ],
        )
        writer.writeheader()
        for i, name in enumerate(names):
            group, slot, kind, raw_note = describe_feature(name, context)
            writer.writerow(
                {
                    "sample_index": int(global_index),
                    "root_id": record.get("root_id", ""),
                    "target_value": record.get("target_value", ""),
                    "feature_index": i,
                    "feature_name": name,
                    "feature_group": group,
                    "slot": slot,
                    "feature_kind": kind,
                    "raw_feature_value_in_simulator": raw_note,
                    "feature_created_for_model": float(features[i]),
                }
            )


def write_metadata(path: Path, *, record: dict[str, Any], global_index: int, shard_path: str, local_index: int) -> None:
    snapshot = record.get("simulator_snapshot") or {}
    stats = record.get("stats")
    metadata = {
        "sample_index": int(global_index),
        "local_filtered_index_in_shard": int(local_index),
        "shard_path": shard_path,
        "root_id": record.get("root_id"),
        "root_player": record.get("root_player"),
        "root_depth": record.get("root_depth"),
        "target_value": record.get("target_value"),
        "best_action_index": record.get("best_action_index"),
        "best_action_repr": record.get("best_action_repr"),
        "best_reward": record.get("best_reward"),
        "best_discount": record.get("best_discount"),
        "best_child_cost": record.get("best_child_cost"),
        "sim_time": snapshot.get("time"),
        "num_request_states": len(snapshot.get("request_states") or {}),
        "active_request_ids": sorted(_stats_set(stats, "active_request_ids")),
        "feature_dim": D_TOTAL,
        "n_global": D_GLOBAL,
    }
    path.write_text(json.dumps(metadata, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    default_dataset_dir = _default_dataset_dir()
    parser = argparse.ArgumentParser(description="Analyze v4-Adv features for one stored root sample")
    parser.add_argument("--manifest", type=Path, default=default_dataset_dir / "manifest.jsonl")
    parser.add_argument("--shard-dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--root-player-filter", default="controller")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Filtered global sample index. If omitted, one sample is selected from --seed.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_index, shard_path, local_index, record = load_sample(
        manifest_path=args.manifest,
        shard_dir=args.shard_dir,
        root_player_filter=args.root_player_filter,
        seed=int(args.seed),
        sample_index=args.sample_index,
    )
    features = extract_features_one_record(record)
    if len(features) != D_TOTAL:
        raise RuntimeError(f"feature length mismatch: {len(features)} != {D_TOTAL}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"sample_{global_index:06d}_features.csv"
    meta_path = args.output_dir / f"sample_{global_index:06d}_metadata.json"
    write_feature_csv(csv_path, record, global_index, features)
    write_metadata(
        meta_path,
        record=record,
        global_index=global_index,
        shard_path=shard_path,
        local_index=local_index,
    )

    print(f"sample_index={global_index}")
    print(f"root_id={record.get('root_id')} target_value={record.get('target_value')}")
    print(f"feature_csv={csv_path}")
    print(f"metadata_json={meta_path}")


if __name__ == "__main__":
    main()
