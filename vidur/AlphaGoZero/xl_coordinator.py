
"""Classical XL replay ingest/status coordinator for AlphaGoZero GV3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from vidur.AlphaGoZero.config import (
    ADVERSARY_POLICY_SAMPLE_CAP,
    MAX_PROMOTIONS,
    MIN_ADVERSARY_STATES_FOR_EVAL,
    PROMOTION_WIN_RATE_THRESHOLD,
    TRAIN_SAMPLE_MIN_LARGE_REPLAY,
    TRAIN_SAMPLE_MIN_MID_REPLAY,
    TRAIN_TRIGGER_NEW_STATES,
    XL_KEEP_ACCEPTED_SHARDS,
    XL_MAX_REPLAY_STATES,
    XL_WRITE_AUDIT_REPLAY,
)
from vidur.AlphaGoZero.durable_transfer import (
    append_csv_file,
    append_csv_row,
    atomic_write_json,
    local_time_24h,
    replay_counts,
    utc_now,
    verify_sha256sums,
)

XL_REPLAY_FIELDS = [
    "states_produced",
    "controller_states",
    "adversary_states",
    "remaining_buffer_size",
    "time_24h",
    "time_since_last_10k_states_24h",
]
DATA_RECEIVED_FIELDS = [
    "server_number",
    "states_produced",
    "controller_states",
    "adversary_states",
    "time",
]
TRAIN_MODEL_FIELDS = [
    "model_config", "model_version", "time_of_training_24h", "sampled_controller_states", "sampled_adversary_states",
    "value_mse", "value_rmse", "value_max_abs_error", "value_p95_abs_error",
    "controller_policy_mse", "controller_policy_cross_entropy", "controller_policy_top1", "controller_policy_top3",
    "adversary_policy_mse", "adversary_policy_cross_entropy", "adversary_policy_top1", "adversary_policy_top3",
]
EVAL_FIELDS = [
    "promoted_model_version", "new_promoted_model_version", "candidate_win_ratio_in_100_games", "candidate_promoted", "time_of_eval",
    "game_hop_number", "total_cost_when_promoted_is_adv_and_new_is_controller", "total_cost_when_new_is_adv_and_promoted_is_controller",
]


def _state_path(root: Path) -> Path:
    return root / "xl_state.json"


def _load_state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    if not path.exists():
        return {"states": 0, "controller": 0, "adversary": 0, "accepted_shards": [], "last_10k_time": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def _reconcile_promotion_state(root: Path, state: dict[str, Any]) -> None:
    """Recover promotion counters from manifests after concurrent state writes."""
    models_dir = Path(root) / "models"
    if not models_dir.exists():
        return

    promoted_versions: list[int] = []
    latest_promotion_time = ""
    for manifest_path in sorted(models_dir.glob("Model_Version*/candidate_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("eval_status", "") or "") != "complete":
            continue
        if not bool(manifest.get("candidate_promoted", False)):
            continue
        try:
            version = int(manifest.get("new_promoted_model_version") or manifest.get("model_version") or 0)
        except Exception:
            version = 0
        if version <= 0:
            continue
        promoted_versions.append(version)
        promoted_at = str(manifest.get("promoted_at_utc", "") or manifest.get("updated_at_utc", "") or "")
        if promoted_at > latest_promotion_time:
            latest_promotion_time = promoted_at

    if not promoted_versions:
        return
    manifest_count = len(set(promoted_versions))
    state["promotions_completed"] = max(int(state.get("promotions_completed", 0) or 0), manifest_count)
    state["promoted_model_version"] = max(int(state.get("promoted_model_version", 0) or 0), max(promoted_versions))
    if latest_promotion_time and not str(state.get("last_promotion_at_utc", "") or ""):
        state["last_promotion_at_utc"] = latest_promotion_time


def _save_state(root: Path, state: dict[str, Any]) -> None:
    _reconcile_promotion_state(root, state)
    state["updated_at_utc"] = utc_now()
    atomic_write_json(_state_path(root), state)


def _ensure_headers(root: Path) -> None:
    for path, fields in [
        (root / "replay_buffer.csv", XL_REPLAY_FIELDS),
        (root / "data_recieved.csv", DATA_RECEIVED_FIELDS),
        (root / "train_model.csv", TRAIN_MODEL_FIELDS),
    ]:
        if not path.exists():
            append_csv_row(path, fields, {k: "" for k in fields})
            # remove the blank data row but keep header
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(lines[0] + "\n", encoding="utf-8")


def _append_xl_status(root: Path, state: dict[str, Any]) -> None:
    append_csv_row(root / "replay_buffer.csv", XL_REPLAY_FIELDS, {
        "states_produced": int(state.get("states", 0)),
        "controller_states": int(state.get("controller", 0)),
        "adversary_states": int(state.get("adversary", 0)),
        "remaining_buffer_size": max(0, int(state.get("max_replay_states", XL_MAX_REPLAY_STATES)) - int(state.get("states", 0))),
        "time_24h": local_time_24h(),
        "time_since_last_10k_states_24h": str(state.get("last_10k_time", "")),
    })



def _feature_complete_counts(path: Path) -> dict[str, int]:
    counts = {"states": 0, "controller": 0, "adversary": 0}
    path = Path(path)
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True"}:
                continue
            if not str(row.get("state_features_json", "")).strip():
                continue
            counts["states"] += 1
            player = str(row.get("player", "")).lower()
            if player == "controller":
                counts["controller"] += 1
            elif player == "adversary":
                counts["adversary"] += 1
    return counts


def _write_filtered_csv_file(
    dst: Path,
    src: Path,
    *,
    extra: dict[str, Any] | None = None,
    feature_complete_only: bool = False,
) -> int:
    dst = Path(dst)
    src = Path(src)
    if not src.exists():
        return 0
    extra = dict(extra or {})
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        src_fields = list(reader.fieldnames or [])
        fields = list(extra.keys()) + [x for x in src_fields if x not in extra]
        dst.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with dst.open("w", encoding="utf-8", newline="") as out:
            w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in reader:
                if feature_complete_only:
                    if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True"}:
                        continue
                    if not str(row.get("state_features_json", "")).strip():
                        continue
                row2 = dict(extra)
                row2.update(row)
                w.writerow(row2)
                count += 1
    return count


def _partition_root(root: Path) -> Path:
    return Path(root) / "global_replay" / "partitions"


def _partition_dir(root: Path, *, worker_id: str, shard_id: str) -> Path:
    return _partition_root(root) / str(worker_id) / str(shard_id)


def _partition_manifest_paths(root: Path) -> list[Path]:
    return sorted(_partition_root(root).glob("*/*/partition_manifest.json"))


def _write_partition(
    root: Path,
    *,
    shard: Path,
    manifest: dict[str, Any],
    raw_counts: dict[str, int],
    feature_counts: dict[str, int],
    extra: dict[str, Any],
) -> Path | None:
    if int(feature_counts.get("states", 0)) <= 0:
        return None
    worker_id = str(manifest.get("worker_id", shard.parent.name))
    shard_id = str(manifest["shard_id"])
    part = _partition_dir(root, worker_id=worker_id, shard_id=shard_id)
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True, exist_ok=True)
    replay = shard / "replay_target_runtime.csv"
    policy_rows = shard / "replay_policy_rows.csv"
    replay_rows = _write_filtered_csv_file(
        part / "replay_target_runtime_feature_complete.csv",
        replay,
        extra=extra,
        feature_complete_only=True,
    )
    policy_count = 0
    if policy_rows.exists():
        policy_count = _write_filtered_csv_file(part / "replay_policy_rows.csv", policy_rows, extra=extra)
    atomic_write_json(part / "partition_manifest.json", {
        "schema_version": 2,
        "shard_id": shard_id,
        "worker_id": worker_id,
        "model_version": int(manifest.get("model_version", 0) or 0),
        "raw_counts": raw_counts,
        "feature_counts": feature_counts,
        "replay_rows_written": int(replay_rows),
        "policy_rows_written": int(policy_count),
        "created_at_utc": utc_now(),
    })
    return part


def _record_accepted_manifest(
    root: Path,
    *,
    shard: Path,
    manifest: dict[str, Any],
    raw_counts: dict[str, int],
    feature_counts: dict[str, int],
    rows_added: int,
) -> None:
    worker_id = str(manifest.get("worker_id", shard.parent.name))
    shard_id = str(manifest["shard_id"])
    accepted_dir = root / "accepted" / worker_id / shard_id
    accepted_dir.parent.mkdir(parents=True, exist_ok=True)
    if accepted_dir.exists():
        shutil.rmtree(accepted_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(accepted_dir / "accepted_manifest.json", {
        "accepted_at_utc": utc_now(),
        "rows_added": int(rows_added),
        "raw_counts": raw_counts,
        "feature_counts": feature_counts,
        "source_manifest": manifest,
        "kept_full_shard": bool(XL_KEEP_ACCEPTED_SHARDS),
    })
    for name in ("shard_manifest.json", "SHA256SUMS"):
        src = shard / name
        if src.exists():
            shutil.copy2(src, accepted_dir / name)


def _prune_replay_partitions(root: Path, state: dict[str, Any]) -> None:
    max_states = int(state.get("max_replay_states", XL_MAX_REPLAY_STATES))
    if max_states <= 0:
        return
    current = int(state.get("feature_states", 0))
    if current <= max_states:
        return
    manifests: list[tuple[str, Path, dict[str, Any]]] = []
    for path in _partition_manifest_paths(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        manifests.append((str(data.get("created_at_utc", "")), path, data))
    for _created, path, data in sorted(manifests, key=lambda x: (x[0], str(x[1]))):
        if current <= max_states:
            break
        raw = dict(data.get("raw_counts", {}) or {})
        feat = dict(data.get("feature_counts", {}) or {})
        state["states"] = max(0, int(state.get("states", 0)) - int(raw.get("states", 0) or 0))
        state["controller"] = max(0, int(state.get("controller", 0)) - int(raw.get("controller", 0) or 0))
        state["adversary"] = max(0, int(state.get("adversary", 0)) - int(raw.get("adversary", 0) or 0))
        state["feature_states"] = max(0, int(state.get("feature_states", 0)) - int(feat.get("states", 0) or 0))
        state["feature_controller"] = max(0, int(state.get("feature_controller", 0)) - int(feat.get("controller", 0) or 0))
        state["feature_adversary"] = max(0, int(state.get("feature_adversary", 0)) - int(feat.get("adversary", 0) or 0))
        current = int(state.get("feature_states", 0))
        shutil.rmtree(path.parent, ignore_errors=True)

def _manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "shard_manifest.json").read_text(encoding="utf-8"))


def _ingest_shard(root: Path, shard: Path, state: dict[str, Any]) -> bool:
    ok, errors = verify_sha256sums(shard)
    if not ok:
        rejected = root / "rejected" / shard.parent.name / shard.name
        rejected.parent.mkdir(parents=True, exist_ok=True)
        if rejected.exists():
            shutil.rmtree(rejected)
        shutil.move(str(shard), str(rejected))
        atomic_write_json(rejected / "reject_reason.json", {"errors": errors, "time": utc_now()})
        return False
    manifest = _manifest(shard)
    shard_id = str(manifest["shard_id"])
    accepted = set(state.get("accepted_shards", []))
    if shard_id in accepted:
        ack_dir = root / "acks" / str(manifest.get("worker_id", shard.parent.name))
        ack_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(ack_dir / f"{shard_id}.accepted", {"accepted_at_utc": utc_now(), "duplicate": True, "manifest": manifest})
        shutil.rmtree(shard, ignore_errors=True)
        return False
    replay = shard / "replay_target_runtime.csv"
    policy_rows = shard / "replay_policy_rows.csv"
    audit_replay = root / "global_replay" / "replay_target_runtime_audit.csv"
    feature_replay = root / "global_replay" / "replay_target_runtime_feature_complete.csv"
    feature_policy = root / "global_replay" / "replay_policy_rows.csv"
    extra = {"accepted_shard_id": shard_id, "source_worker_id": manifest.get("worker_id", "")}
    added = append_csv_file(audit_replay, replay, extra=extra) if bool(XL_WRITE_AUDIT_REPLAY) else 0
    feature_counts = _feature_complete_counts(replay)
    counts = replay_counts(replay)
    _write_partition(
        root,
        shard=shard,
        manifest=manifest,
        raw_counts=counts,
        feature_counts=feature_counts,
        extra=extra,
    )
    # Legacy monolithic writes are intentionally disabled for new ingests. The
    # trainer still supports old monolithic files as a fallback, but new runs use
    # prunable per-shard partitions to bound disk and memory.
    if False and int(feature_counts["states"]) > 0:
        append_csv_file(feature_replay, replay, extra=extra)
        if policy_rows.exists():
            append_csv_file(feature_policy, policy_rows, extra=extra)
    state["states"] = int(state.get("states", 0)) + int(counts["states"])
    state["controller"] = int(state.get("controller", 0)) + int(counts["controller"])
    state["adversary"] = int(state.get("adversary", 0)) + int(counts["adversary"])
    state["feature_states"] = int(state.get("feature_states", 0)) + int(feature_counts["states"])
    state["feature_controller"] = int(state.get("feature_controller", 0)) + int(feature_counts["controller"])
    state["feature_adversary"] = int(state.get("feature_adversary", 0)) + int(feature_counts["adversary"])
    state["new_states_since_last_training"] = int(state.get("new_states_since_last_training", 0)) + int(feature_counts["states"])
    if int(counts["states"]) >= 10_000 or int(state.get("new_states_since_last_10k", 0)) + int(counts["states"]) >= 10_000:
        state["last_10k_time"] = local_time_24h()
        state["new_states_since_last_10k"] = 0
    else:
        state["new_states_since_last_10k"] = int(state.get("new_states_since_last_10k", 0)) + int(counts["states"])
    state.setdefault("accepted_shards", []).append(shard_id)
    append_csv_row(root / "data_recieved.csv", DATA_RECEIVED_FIELDS, {
        "server_number": manifest.get("worker_id", shard.parent.name),
        "states_produced": int(counts["states"]),
        "controller_states": int(counts["controller"]),
        "adversary_states": int(counts["adversary"]),
        "time": local_time_24h(),
    })
    rows_added = int(feature_counts["states"] if int(feature_counts["states"]) > 0 else added)
    _record_accepted_manifest(
        root,
        shard=shard,
        manifest=manifest,
        raw_counts=counts,
        feature_counts=feature_counts,
        rows_added=rows_added,
    )
    ack_dir = root / "acks" / str(manifest.get("worker_id", shard.parent.name))
    ack_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ack_dir / f"{shard_id}.accepted", {"accepted_at_utc": utc_now(), "rows_added": rows_added, "manifest": manifest})
    if bool(XL_KEEP_ACCEPTED_SHARDS):
        accepted_full = root / "accepted_full_shards" / shard.parent.name / shard.name
        accepted_full.parent.mkdir(parents=True, exist_ok=True)
        if accepted_full.exists():
            shutil.rmtree(accepted_full)
        shutil.move(str(shard), str(accepted_full))
    else:
        shutil.rmtree(shard, ignore_errors=True)
    _prune_replay_partitions(root, state)
    return True


def ingest_once(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    _ensure_headers(root)
    state = _load_state(root)
    state.setdefault("max_replay_states", XL_MAX_REPLAY_STATES)
    ingested = 0
    incoming = root / "incoming"
    for worker_dir in sorted(incoming.glob("*")) if incoming.exists() else []:
        if not worker_dir.is_dir():
            continue
        for shard in sorted(worker_dir.glob("*")):
            if not shard.is_dir() or not (shard / "shard_manifest.json").exists():
                continue
            if _ingest_shard(root, shard, state):
                ingested += 1
    _append_xl_status(root, state)
    _save_state(root, state)
    return ingested


def _training_sample_floor(total_states: int) -> int:
    if int(total_states) > 100_000:
        return int(TRAIN_SAMPLE_MIN_LARGE_REPLAY)
    if int(total_states) >= 50_000:
        return int(TRAIN_SAMPLE_MIN_MID_REPLAY)
    return 0


def _adversary_policy_sample_count(adversary_states: int) -> int:
    if int(adversary_states) < int(MIN_ADVERSARY_STATES_FOR_EVAL):
        return 0
    return min(int(adversary_states), int(ADVERSARY_POLICY_SAMPLE_CAP))


def write_training_gate_status(root: Path) -> dict[str, Any]:
    state = _load_state(root)
    raw_total_states = int(state.get("states", 0))
    raw_controller_states = int(state.get("controller", 0))
    raw_adversary_states = int(state.get("adversary", 0))
    total_states = int(state.get("feature_states", 0))
    controller_states = int(state.get("feature_controller", 0))
    adversary_states = int(state.get("feature_adversary", 0))
    new_states = int(state.get("new_states_since_last_training", 0))
    train_sample_floor = _training_sample_floor(total_states)
    adversary_policy_samples = _adversary_policy_sample_count(adversary_states)
    promotions_completed = int(state.get("promotions_completed", 0))
    promotion_cap_reached = promotions_completed >= int(MAX_PROMOTIONS)
    has_enough_new_replay = new_states >= int(TRAIN_TRIGGER_NEW_STATES)
    has_enough_adversary = adversary_states >= int(MIN_ADVERSARY_STATES_FOR_EVAL)
    has_train_sample_floor = train_sample_floor > 0 and total_states >= train_sample_floor
    can_train = bool((not promotion_cap_reached) and has_enough_new_replay and has_enough_adversary and has_train_sample_floor)
    data = {
        "can_train": can_train,
        "can_eval": bool(has_enough_adversary),
        "promotion_win_rate_threshold": float(PROMOTION_WIN_RATE_THRESHOLD),
        "max_promotions": int(MAX_PROMOTIONS),
        "promotions_completed": int(promotions_completed),
        "promotion_cap_reached": bool(promotion_cap_reached),
        "new_states_since_last_training": new_states,
        "train_trigger_new_states": int(TRAIN_TRIGGER_NEW_STATES),
        "total_states": total_states,
        "controller_states": controller_states,
        "adversary_states": adversary_states,
        "raw_total_states": raw_total_states,
        "raw_controller_states": raw_controller_states,
        "raw_adversary_states": raw_adversary_states,
        "min_adversary_states_for_eval": int(MIN_ADVERSARY_STATES_FOR_EVAL),
        "adversary_policy_sample_count": int(adversary_policy_samples),
        "adversary_policy_sample_cap": int(ADVERSARY_POLICY_SAMPLE_CAP),
        "train_sample_floor": int(train_sample_floor),
        "has_enough_new_replay": bool(has_enough_new_replay),
        "has_enough_adversary": bool(has_enough_adversary),
        "has_train_sample_floor": bool(has_train_sample_floor),
        "checked_at_utc": utc_now(),
    }
    atomic_write_json(root / "training_gate_status.json", data)
    return data



def _pid_alive(pid: int) -> bool:
    pid = int(pid)
    if pid <= 0:
        return False
    # A killed child can remain as a zombie until reaped. os.kill(pid, 0)
    # still succeeds for zombies, but they are not an active trainer.
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        if stat_path.exists():
            fields = stat_path.read_text(encoding="utf-8", errors="ignore").split()
            if len(fields) >= 3 and fields[2] == "Z":
                return False
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _finalize_completed_training_cycle_if_needed(root: Path, train_dir: Path) -> bool:
    current = train_dir / "current_training.json"
    if not current.exists():
        return False
    try:
        data = json.loads(current.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if str(data.get("cycle_finalized_at_utc", "") or ""):
        return False

    state = _load_state(root)
    version = int(state.get("last_candidate_model_version", 0) or 0)
    if version <= 0:
        return False
    manifest_path = root / "models" / f"Model_Version{version}" / "candidate_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(manifest.get("eval_status", "") or "") != "complete":
        return False

    state["new_states_since_last_training"] = 0
    state["last_training_completed_at_utc"] = utc_now()
    state["last_training_cycle_completed_model_version"] = int(version)
    _save_state(root, state)

    data["cycle_finalized_at_utc"] = utc_now()
    data["cycle_finalized_model_version"] = int(version)
    atomic_write_json(current, data)
    return True


def maybe_launch_training(root: Path, gate: dict[str, Any]) -> dict[str, Any]:
    train_dir = Path(root) / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    pid_path = train_dir / "current.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            pid = 0
        if _pid_alive(pid):
            return {"training_launched": False, "training_running_pid": int(pid)}
        if _finalize_completed_training_cycle_if_needed(Path(root), train_dir):
            return {"training_launched": False, "training_running_pid": 0, "training_cycle_finalized": True}
    if not bool(gate.get("can_train", False)):
        return {"training_launched": False, "training_running_pid": 0}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = train_dir / f"train_{stamp}.log"
    cmd = [
        sys.executable,
        "-m",
        "vidur.AlphaGoZero.agz_train_eval_promote",
        "--output-root",
        str(Path(root)),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(int(proc.pid)) + "\n", encoding="utf-8")
    atomic_write_json(train_dir / "current_training.json", {
        "pid": int(proc.pid),
        "cmd": cmd,
        "log_path": str(log_path),
        "started_at_utc": utc_now(),
        "gate": gate,
    })
    return {"training_launched": True, "training_running_pid": int(proc.pid), "training_log": str(log_path)}

def init_eval_dir(root: Path, eval_number: int) -> Path:
    d = root / "eval_of_models" / f"eval_{int(eval_number):06d}"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "eval_game_details.csv").exists():
        with (d / "eval_game_details.csv").open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=EVAL_FIELDS).writeheader()
    (d / "SJF_256_Game").mkdir(parents=True, exist_ok=True)
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaGoZero XL coordinator ingest/status loop.")
    p.add_argument("--output-root", type=Path, default=Path("/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero"))
    p.add_argument("--loop", action="store_true")
    p.add_argument("--poll-sec", type=float, default=30.0)
    p.add_argument("--init-eval-dir", type=int, default=None)
    p.add_argument("--enable-training", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).expanduser()
    if args.init_eval_dir is not None:
        print(init_eval_dir(root, int(args.init_eval_dir)))
        return
    while True:
        n = ingest_once(root)
        gate = write_training_gate_status(root)
        train_status = maybe_launch_training(root, gate) if bool(args.enable_training) else {"training_launched": False, "training_disabled": True}
        print(json.dumps({"ingested_shards": n, **gate, **train_status}, sort_keys=True), flush=True)
        if not args.loop:
            break
        time.sleep(float(args.poll_sec))


if __name__ == "__main__":
    main()
