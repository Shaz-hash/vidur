
"""Classical XL replay ingest/status coordinator for AlphaGoZero GV3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from vidur.AlphaGoZero.adaptive_rollout import (
    activate_manifest_horizon,
    active_rollout_horizon_sec,
    ensure_runtime_search_config,
)
from vidur.AlphaGoZero.config import (
    ADVERSARY_POLICY_SAMPLE_CAP,
    AGZ_EVAL_MCTS_ITERATIONS,
    AGZ_REPLAY_INDEX_BUILD_WORKERS,
    AGZ_REPLAY_SAMPLER,
    AGZ_VALUE_FEATURE_SCHEMA,
    MAX_PROMOTIONS,
    MIN_CONTROLLER_STATES_FOR_EVAL,
    MIN_ADVERSARY_STATES_FOR_EVAL,
    Phase1SmokeConfig,
    PROMOTION_WIN_RATE_THRESHOLD,
    ROLE_PROMOTION_WIN_THRESHOLD,
    TRAIN_SAMPLE_MIN_LARGE_REPLAY,
    TRAIN_SAMPLE_MIN_MID_REPLAY,
    TRAIN_TRIGGER_NEW_STATES,
    XL_KEEP_ACCEPTED_SHARDS,
    XL_MAX_INGEST_SHARDS_PER_LOOP,
    XL_PAUSE_INGEST_WHILE_PRUNE_DEFERRED,
    XL_ADVERSARY_MAX_REPLAY_STATES,
    XL_CONTROLLER_MAX_REPLAY_STATES,
    XL_CONTROLLER_POLICY_SAMPLE_CAP,
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
from vidur.AlphaGoZero.spot_work_protocol import (
    configure_scheduler,
    default_selfplay_config,
    selfplay_config_sha256,
)

XL_REPLAY_FIELDS = [
    "states_produced",
    "controller_states",
    "adversary_states",
    "remaining_buffer_size",
    "time_24h",
    "time_since_last_10k_states_24h",
]
REPLAY_DISTRIBUTION_FIELDS = [
    "states_produced",
    "controller_states",
    "adversary_states",
    "remaining_buffer_size",
    "controller_remaining_buffer_size",
    "adversary_remaining_buffer_size",
    "current_promoted_controller_model_version",
    "current_promoted_controller_model_states",
    "other_controller_model_versions",
    "other_controller_model_states",
    "current_promoted_adversary_model_version",
    "current_promoted_adversary_model_states",
    "other_adversary_model_versions",
    "other_adversary_model_states",
]
DATA_RECEIVED_FIELDS = [
    "server_number",
    "states_produced",
    "controller_states",
    "adversary_states",
    "time",
]

_REPLAY_INDEX_EXECUTOR: ProcessPoolExecutor | None = None
_REPLAY_INDEX_FUTURES: dict[Any, Path] = {}


def _schedule_replay_index_build(state_path: Path) -> None:
    global _REPLAY_INDEX_EXECUTOR
    if str(AGZ_REPLAY_SAMPLER) != "indexed_v1":
        return
    if multiprocessing.current_process().name != "MainProcess":
        return
    from vidur.AlphaGoZero.indexed_replay import ensure_partition_index_worker

    if _REPLAY_INDEX_EXECUTOR is None:
        _REPLAY_INDEX_EXECUTOR = ProcessPoolExecutor(
            max_workers=max(1, int(AGZ_REPLAY_INDEX_BUILD_WORKERS))
        )
    future = _REPLAY_INDEX_EXECUTOR.submit(
        ensure_partition_index_worker,
        str(Path(state_path)),
        str(AGZ_VALUE_FEATURE_SCHEMA),
    )
    _REPLAY_INDEX_FUTURES[future] = Path(state_path)


def _drain_replay_index_builds() -> None:
    completed = [future for future in _REPLAY_INDEX_FUTURES if future.done()]
    for future in completed:
        state_path = _REPLAY_INDEX_FUTURES.pop(future)
        try:
            descriptor, rebuilt, elapsed = future.result()
            print(json.dumps({
                "event": "replay_index_ready",
                "source": str(state_path),
                "role": str(descriptor.role),
                "rows": int(descriptor.rows),
                "policy_roots": int(descriptor.policy_roots),
                "rebuilt": bool(rebuilt),
                "elapsed_s": float(elapsed),
            }, sort_keys=True), flush=True)
        except FileNotFoundError:
            # FIFO pruning may remove a partition while its background build is queued.
            continue
        except Exception as exc:
            print(json.dumps({
                "event": "replay_index_failed",
                "source": str(state_path),
                "error": repr(exc),
            }, sort_keys=True), flush=True)

INGESTION_AUDIT_FIELDS = [
    "time_utc",
    "worker_id",
    "shard_id",
    "controller_model_version",
    "adversary_model_version",
    "raw_controller_states",
    "raw_adversary_states",
    "admitted_controller_states",
    "admitted_adversary_states",
    "dropped_controller_states",
    "dropped_adversary_states",
    "controller_admission_reason",
    "adversary_admission_reason",
]
TRAIN_MODEL_FIELDS = [
    "model_config", "model_version", "time_of_training_24h", "sampled_controller_states", "sampled_adversary_states",
    "value_mse", "value_rmse", "value_max_abs_error", "value_p95_abs_error",
    "controller_value_mse", "controller_value_rmse", "controller_value_max_abs_error", "controller_value_p95_abs_error",
    "adversary_value_mse", "adversary_value_rmse", "adversary_value_max_abs_error", "adversary_value_p95_abs_error",
    "controller_policy_mse", "controller_policy_cross_entropy", "controller_policy_top1", "controller_policy_top3",
    "adversary_policy_mse", "adversary_policy_cross_entropy", "adversary_policy_top1", "adversary_policy_top3",
    "rollout_horizon_used_sec", "rollout_horizon_source_controller_p95_abs_error",
    "rollout_value_error_threshold", "rollout_discount_factor", "rollout_reference_step_sec",
    "rollout_max_horizon_sec", "rollout_horizon_tick_sec", "next_rollout_horizon_raw_sec",
    "next_rollout_horizon_calculated_sec", "next_rollout_horizon_rounded_sec",
    "next_rollout_discounted_error",
]
EVAL_FIELDS = [
    "promoted_controller_model_version", "promoted_adversary_model_version", "candidate_model_version",
    "new_controller_model_version", "new_adversary_model_version", "candidate_win_ratio_in_100_games",
    "candidate_controller_promoted", "candidate_adversary_promoted", "candidate_promoted", "time_of_eval",
    "game_mode", "game_id", "game_hop_number",
    "total_cost_when_promoted_adv_and_promoted_controller",
    "total_cost_when_candidate_adv_and_promoted_controller",
    "total_cost_when_promoted_adv_and_candidate_controller",
    "total_cost_when_candidate_adv_and_candidate_controller",
]


def _train_failure_backoff_sec() -> int:
    try:
        return max(0, int(os.environ.get("AGZ_TRAIN_FAILURE_BACKOFF_SEC", "600")))
    except Exception:
        return 600


def _training_subprocess_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AGZ_ROLLOUT_HORIZON_SEC"] = str(active_rollout_horizon_sec(Path(root)))
    return env


def _epoch_to_utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _active_spot_selfplay_config(root: Path) -> dict[str, Any]:
    config = default_selfplay_config()
    config["rollout_horizon_sec"] = float(active_rollout_horizon_sec(Path(root)))
    return config


def _refresh_spot_scheduler(
    root: Path,
    *,
    selfplay_limit: int,
    eval_limit: int,
    selfplay_lease_sec: int,
    worker_heartbeat_timeout_sec: int,
) -> dict[str, Any]:
    selfplay_config = _active_spot_selfplay_config(root)
    expected_hash = selfplay_config_sha256(selfplay_config)
    existing = _read_json_dict(
        Path(root) / "spot_work" / "control" / "scheduler_config.json"
    )
    if (
        int(existing.get("selfplay_total_parallel_games", -1)) == int(selfplay_limit)
        and int(existing.get("eval_total_parallel_games", -1)) == int(eval_limit)
        and int(existing.get("selfplay_lease_sec", -1))
        == min(int(selfplay_lease_sec), int(worker_heartbeat_timeout_sec))
        and int(existing.get("worker_heartbeat_timeout_sec", -1))
        == int(worker_heartbeat_timeout_sec)
        and str(existing.get("selfplay_config_sha256", "")) == expected_hash
    ):
        return existing
    return configure_scheduler(
        root,
        selfplay_total_parallel_games=int(selfplay_limit),
        eval_total_parallel_games=int(eval_limit),
        selfplay_lease_sec=int(selfplay_lease_sec),
        worker_heartbeat_timeout_sec=int(worker_heartbeat_timeout_sec),
        selfplay_config=selfplay_config,
    )


def _tail_text(path: Path, *, max_chars: int = 6000) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return data[-int(max_chars):]


def _state_path(root: Path) -> Path:
    return root / "xl_state.json"


def _initial_promoted_model_version() -> int:
    return int(Phase1SmokeConfig().model_version)


def _seed_initial_promoted_model_state(state: dict[str, Any]) -> dict[str, Any]:
    initial_version = _initial_promoted_model_version()
    legacy_current = int(state.get("promoted_model_version", 0) or 0)
    if legacy_current <= 0:
        legacy_current = int(initial_version)

    controller_current = int(state.get("controller_promoted_model_version", legacy_current) or legacy_current)
    adversary_current = int(state.get("adversary_promoted_model_version", legacy_current) or legacy_current)
    state["controller_promoted_model_version"] = int(controller_current)
    state["adversary_promoted_model_version"] = int(adversary_current)
    state["promoted_model_version"] = int(max(controller_current, adversary_current))

    controller_history = _to_int_list(state.get("controller_promoted_model_history", []))
    adversary_history = _to_int_list(state.get("adversary_promoted_model_history", []))
    legacy_history = _to_int_list(state.get("promoted_model_history", []))
    if not controller_history:
        controller_history = list(legacy_history) if legacy_history else [int(controller_current)]
    if not adversary_history:
        adversary_history = list(legacy_history) if legacy_history else [int(adversary_current)]
    if controller_current not in controller_history:
        controller_history.append(int(controller_current))
    if adversary_current not in adversary_history:
        adversary_history.append(int(adversary_current))
    state["controller_promoted_model_history"] = controller_history
    state["adversary_promoted_model_history"] = adversary_history

    merged_history: list[int] = []
    for version in [*controller_history, *adversary_history]:
        if int(version) > 0 and int(version) not in merged_history:
            merged_history.append(int(version))
    state["promoted_model_history"] = merged_history
    state.setdefault("controller_promotions_completed", 0)
    state.setdefault("adversary_promotions_completed", 0)
    state["promotions_completed"] = int(
        max(
            int(state.get("promotions_completed", 0) or 0),
            int(state.get("controller_promotions_completed", 0) or 0),
            int(state.get("adversary_promotions_completed", 0) or 0),
        )
    )
    return state


def _load_state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    if not path.exists():
        state = _seed_initial_promoted_model_state({
            "states": 0,
            "controller": 0,
            "adversary": 0,
            "accepted_shards": [],
            "last_10k_time": "",
        })
    else:
        state = _seed_initial_promoted_model_state(json.loads(path.read_text(encoding="utf-8")))
    # This counter never decreases when FIFO replay partitions are evicted. It
    # makes the training gate independent of concurrent replay-size rewrites.
    state.setdefault(
        "lifetime_admitted_states",
        max(
            int(state.get("feature_states", 0) or 0),
            int(state.get("states", 0) or 0),
            int(state.get("new_states_since_last_training", 0) or 0),
        ),
    )
    return state


def _to_int_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else str(raw).replace("[", "").replace("]", "").split(",")
    out: list[int] = []
    for item in items:
        try:
            value = int(item)
        except Exception:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return out


def _reconcile_promotion_state(root: Path, state: dict[str, Any]) -> None:
    """Recover role promotion counters from manifests after concurrent state writes."""
    models_dir = Path(root) / "models"
    if not models_dir.exists():
        return

    controller_versions: list[int] = []
    adversary_versions: list[int] = []
    latest_promotion_time = ""
    for manifest_path in sorted(models_dir.glob("Model_Version*/candidate_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("eval_status", "") or "") != "complete":
            continue
        try:
            candidate_version = int(manifest.get("model_version") or manifest.get("candidate_model_version") or 0)
        except Exception:
            candidate_version = 0
        if bool(manifest.get("candidate_controller_promoted", False)):
            version = int(manifest.get("new_controller_model_version") or candidate_version or 0)
            if version > 0:
                controller_versions.append(version)
        if bool(manifest.get("candidate_adversary_promoted", False)):
            version = int(manifest.get("new_adversary_model_version") or candidate_version or 0)
            if version > 0:
                adversary_versions.append(version)
        # Legacy all-or-nothing manifests promoted both roles.
        if bool(manifest.get("candidate_promoted", False)) and not (
            "candidate_controller_promoted" in manifest or "candidate_adversary_promoted" in manifest
        ):
            version = int(manifest.get("new_promoted_model_version") or manifest.get("model_version") or 0)
            if version > 0:
                controller_versions.append(version)
                adversary_versions.append(version)
        promoted_at = str(manifest.get("promoted_at_utc", "") or manifest.get("updated_at_utc", "") or "")
        if promoted_at > latest_promotion_time:
            latest_promotion_time = promoted_at

    controller_history = _to_int_list(state.get("controller_promoted_model_history", []))
    adversary_history = _to_int_list(state.get("adversary_promoted_model_history", []))
    controller_current = int(state.get("controller_promoted_model_version", state.get("promoted_model_version", 0)) or 0)
    adversary_current = int(state.get("adversary_promoted_model_version", state.get("promoted_model_version", 0)) or 0)
    if controller_current > 0 and controller_current not in controller_history:
        controller_history.append(controller_current)
    if adversary_current > 0 and adversary_current not in adversary_history:
        adversary_history.append(adversary_current)
    for version in sorted(set(controller_versions)):
        if version not in controller_history:
            controller_history.append(version)
    for version in sorted(set(adversary_versions)):
        if version not in adversary_history:
            adversary_history.append(version)

    if controller_history:
        state["controller_promoted_model_history"] = controller_history
        state["controller_promoted_model_version"] = int(max(controller_history))
    if adversary_history:
        state["adversary_promoted_model_history"] = adversary_history
        state["adversary_promoted_model_version"] = int(max(adversary_history))

    merged_history: list[int] = []
    for version in [*controller_history, *adversary_history]:
        if int(version) > 0 and int(version) not in merged_history:
            merged_history.append(int(version))
    if merged_history:
        state["promoted_model_history"] = merged_history
        state["promoted_model_version"] = int(
            max(
                int(state.get("controller_promoted_model_version", 0) or 0),
                int(state.get("adversary_promoted_model_version", 0) or 0),
            )
        )

    state["controller_promotions_completed"] = max(
        int(state.get("controller_promotions_completed", 0) or 0),
        len(set(controller_versions)),
    )
    state["adversary_promotions_completed"] = max(
        int(state.get("adversary_promotions_completed", 0) or 0),
        len(set(adversary_versions)),
    )
    state["promotions_completed"] = max(
        int(state.get("promotions_completed", 0) or 0),
        int(state.get("controller_promotions_completed", 0) or 0),
        int(state.get("adversary_promotions_completed", 0) or 0),
    )
    if latest_promotion_time and not str(state.get("last_promotion_at_utc", "") or ""):
        state["last_promotion_at_utc"] = latest_promotion_time

def _save_state(root: Path, state: dict[str, Any]) -> None:
    _reconcile_promotion_state(root, state)
    state["updated_at_utc"] = utc_now()
    atomic_write_json(_state_path(root), state)


def _ensure_headers(root: Path) -> None:
    for path, fields in [
        (root / "replay_buffer.csv", XL_REPLAY_FIELDS),
        (root / "replay_distribution.csv", REPLAY_DISTRIBUTION_FIELDS),
        (root / "data_recieved.csv", DATA_RECEIVED_FIELDS),
        (root / "ingestion_audit.csv", INGESTION_AUDIT_FIELDS),
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
    row_filter: Any | None = None,
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
                if row_filter is not None and not bool(row_filter(row)):
                    continue
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


def _source_root_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(float(row.get("game_id", 0) or 0)),
        int(float(row.get("turn_number", 0) or 0)),
        int(float(row.get("depth_number", 0) or 0)),
        str(row.get("player", "")).strip().lower(),
    )


def _select_role_root_keys(
    replay: Path,
    *,
    role: str,
    limit: int,
    seed_material: str,
) -> set[tuple[int, int, int, str]]:
    candidates: list[tuple[int, int, int, str]] = []
    with Path(replay).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("player", "")).strip().lower() != str(role):
                continue
            if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True"}:
                continue
            if not str(row.get("state_features_json", "")).strip():
                continue
            candidates.append(_source_root_key(row))
    limit = max(0, min(int(limit), len(candidates)))
    if limit >= len(candidates):
        return set(candidates)
    digest = hashlib.sha256(str(seed_material).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big", signed=False))
    return set(rng.sample(candidates, limit))


def _write_role_partition(
    root: Path,
    *,
    worker_id: str,
    source_shard_id: str,
    replay: Path,
    policy_rows: Path,
    manifest: dict[str, Any],
    role: str,
    selected_keys: set[tuple[int, int, int, str]],
    extra: dict[str, Any],
) -> Path | None:
    if not selected_keys:
        return None
    partition_id = f"{source_shard_id}__{role}"
    part = _partition_dir(root, worker_id=worker_id, shard_id=partition_id)
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True, exist_ok=True)

    def keep(row: dict[str, Any]) -> bool:
        return _source_root_key(row) in selected_keys

    replay_rows = _write_filtered_csv_file(
        part / "replay_target_runtime_feature_complete.csv",
        replay,
        extra=extra,
        feature_complete_only=True,
        row_filter=keep,
    )
    policy_count = 0
    if Path(policy_rows).exists():
        policy_count = _write_filtered_csv_file(
            part / "replay_policy_rows.csv",
            policy_rows,
            extra=extra,
            row_filter=keep,
        )
    counts = {
        "states": int(replay_rows),
        "controller": int(replay_rows if role == "controller" else 0),
        "adversary": int(replay_rows if role == "adversary" else 0),
    }
    legacy_version = int(manifest.get("model_version", 0) or 0)
    atomic_write_json(part / "partition_manifest.json", {
        "schema_version": 4,
        "partition_role": str(role),
        "shard_id": partition_id,
        "source_shard_id": str(source_shard_id),
        "worker_id": str(worker_id),
        "model_version": int(legacy_version),
        "controller_model_version": int(manifest.get("controller_model_version", legacy_version) or legacy_version),
        "adversary_model_version": int(manifest.get("adversary_model_version", legacy_version) or legacy_version),
        "raw_counts": counts,
        "feature_counts": counts,
        "replay_rows_written": int(replay_rows),
        "policy_rows_written": int(policy_count),
        "created_at_utc": utc_now(),
    })
    _schedule_replay_index_build(part / "replay_target_runtime_feature_complete.csv")
    return part


def _write_partition(
    root: Path,
    *,
    shard: Path,
    manifest: dict[str, Any],
    raw_counts: dict[str, int],
    feature_counts: dict[str, int],
    extra: dict[str, Any],
    role_limits: dict[str, int] | None = None,
) -> dict[str, int]:
    worker_id = str(manifest.get("worker_id", shard.parent.name))
    shard_id = str(manifest["shard_id"])
    replay = shard / "replay_target_runtime.csv"
    policy_rows = shard / "replay_policy_rows.csv"
    limits = dict(role_limits or feature_counts)
    admitted = {"states": 0, "controller": 0, "adversary": 0}
    for role in ("controller", "adversary"):
        selected = _select_role_root_keys(
            replay,
            role=role,
            limit=int(limits.get(role, 0) or 0),
            seed_material=f"{worker_id}:{shard_id}:{role}",
        )
        part = _write_role_partition(
            root,
            worker_id=worker_id,
            source_shard_id=shard_id,
            replay=replay,
            policy_rows=policy_rows,
            manifest=manifest,
            role=role,
            selected_keys=selected,
            extra=extra,
        )
        if part is not None:
            admitted[role] = len(selected)
            admitted["states"] += len(selected)
    return admitted


def _record_accepted_manifest(
    root: Path,
    *,
    shard: Path,
    manifest: dict[str, Any],
    raw_counts: dict[str, int],
    feature_counts: dict[str, int],
    rows_added: int,
    source_counts: dict[str, int] | None = None,
    admission: dict[str, Any] | None = None,
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
        "source_counts": dict(source_counts or raw_counts),
        "admission": dict(admission or {}),
        "source_manifest": manifest,
        "kept_full_shard": bool(XL_KEEP_ACCEPTED_SHARDS),
    })
    for name in ("shard_manifest.json", "SHA256SUMS"):
        src = shard / name
        if src.exists():
            shutil.copy2(src, accepted_dir / name)


def _partition_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _partition_manifest_paths(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            legacy_version = int(data.get("model_version", 0) or 0)
        except Exception:
            legacy_version = 0
        controller_version = int(data.get("controller_model_version", legacy_version) or legacy_version)
        adversary_version = int(data.get("adversary_model_version", legacy_version) or legacy_version)
        entries.append({
            "created": str(data.get("created_at_utc", "")),
            "path": path,
            "data": data,
            "version": int(legacy_version),
            "controller_version": int(controller_version),
            "adversary_version": int(adversary_version),
        })
    entries.sort(key=lambda x: (str(x["created"]), str(x["path"])))
    return entries


def _partition_version_counts(entries: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, int]]]:
    counts: dict[str, dict[int, dict[str, int]]] = {"controller": {}, "adversary": {}}
    for entry in entries:
        feat = dict(entry.get("data", {}).get("feature_counts", {}) or {})
        controller_version = int(entry.get("controller_version", entry.get("version", 0)) or 0)
        adversary_version = int(entry.get("adversary_version", entry.get("version", 0)) or 0)
        controller_states = int(feat.get("controller", 0) or 0)
        adversary_states = int(feat.get("adversary", 0) or 0)
        ctrl_bucket = counts["controller"].setdefault(controller_version, {"states": 0})
        adv_bucket = counts["adversary"].setdefault(adversary_version, {"states": 0})
        ctrl_bucket["states"] += controller_states
        adv_bucket["states"] += adversary_states
    return counts


def _rebuild_replay_state_counts(root: Path, state: dict[str, Any]) -> None:
    raw = {"states": 0, "controller": 0, "adversary": 0}
    feature = {"states": 0, "controller": 0, "adversary": 0}
    for entry in _partition_entries(root):
        data = dict(entry.get("data", {}) or {})
        for key in raw:
            raw[key] += int(dict(data.get("raw_counts", {}) or {}).get(key, 0) or 0)
            feature[key] += int(dict(data.get("feature_counts", {}) or {}).get(key, 0) or 0)
    state["states"] = int(raw["states"])
    state["controller"] = int(raw["controller"])
    state["adversary"] = int(raw["adversary"])
    state["feature_states"] = int(feature["states"])
    state["feature_controller"] = int(feature["controller"])
    state["feature_adversary"] = int(feature["adversary"])


def _migrate_mixed_partition(payload: tuple[str, str]) -> int:
    root_text, manifest_text = payload
    root = Path(root_text)
    manifest_path = Path(manifest_text)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0
    if str(data.get("partition_role", "")) in {"controller", "adversary"}:
        return 0
    source_dir = manifest_path.parent
    replay = source_dir / "replay_target_runtime_feature_complete.csv"
    policy_rows = source_dir / "replay_policy_rows.csv"
    if not replay.exists():
        return 0
    worker_id = str(data.get("worker_id", source_dir.parent.name))
    source_shard_id = str(data.get("source_shard_id") or data.get("shard_id") or source_dir.name)
    for role in ("controller", "adversary"):
        selected = _select_role_root_keys(
            replay,
            role=role,
            limit=2**63 - 1,
            seed_material=f"migration:{worker_id}:{source_shard_id}:{role}",
        )
        _write_role_partition(
            root,
            worker_id=worker_id,
            source_shard_id=source_shard_id,
            replay=replay,
            policy_rows=policy_rows,
            manifest=data,
            role=role,
            selected_keys=selected,
            extra={},
        )
    shutil.rmtree(source_dir, ignore_errors=True)
    return 1


def _migrate_mixed_replay_partitions(
    root: Path,
    state: dict[str, Any],
    *,
    workers: int = 1,
) -> int:
    manifests = [
        str(entry["path"])
        for entry in _partition_entries(root)
        if str(dict(entry.get("data", {}) or {}).get("partition_role", "")) not in {"controller", "adversary"}
    ]
    payloads = [(str(Path(root)), manifest) for manifest in manifests]
    worker_count = max(1, min(int(workers), len(payloads) or 1))
    if worker_count == 1:
        migrated = sum(_migrate_mixed_partition(payload) for payload in payloads)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            migrated = sum(pool.map(_migrate_mixed_partition, payloads, chunksize=1))
    _rebuild_replay_state_counts(root, state)
    state["role_partition_migration_complete"] = True
    state["role_partition_migration_count"] = int(state.get("role_partition_migration_count", 0) or 0) + int(migrated)
    state["last_role_partition_migration_at_utc"] = utc_now()
    return int(migrated)


def _role_admission_plan(feature_counts: dict[str, int]) -> tuple[dict[str, int], dict[str, str]]:
    limits = {
        role: max(0, int(feature_counts.get(role, 0) or 0))
        for role in ("controller", "adversary")
    }
    reasons = {
        role: "admit_fifo" if count > 0 else "no_source_rows"
        for role, count in limits.items()
    }
    return limits, reasons


def _subtract_partition_counts(state: dict[str, Any], data: dict[str, Any]) -> None:
    raw = dict(data.get("raw_counts", {}) or {})
    feat = dict(data.get("feature_counts", {}) or {})
    state["states"] = max(0, int(state.get("states", 0)) - int(raw.get("states", 0) or 0))
    state["controller"] = max(0, int(state.get("controller", 0)) - int(raw.get("controller", 0) or 0))
    state["adversary"] = max(0, int(state.get("adversary", 0)) - int(raw.get("adversary", 0) or 0))
    state["feature_states"] = max(0, int(state.get("feature_states", 0)) - int(feat.get("states", 0) or 0))
    state["feature_controller"] = max(0, int(state.get("feature_controller", 0)) - int(feat.get("controller", 0) or 0))
    state["feature_adversary"] = max(0, int(state.get("feature_adversary", 0)) - int(feat.get("adversary", 0) or 0))


def _entry_role_states(entry: dict[str, Any], role: str) -> int:
    feat = dict(entry.get("data", {}).get("feature_counts", {}) or {})
    return int(feat.get(role, 0) or 0)


def _role_replay_capacity(state: dict[str, Any], role: str) -> int:
    if role == "controller":
        return int(state.get("controller_max_replay_states", XL_CONTROLLER_MAX_REPLAY_STATES))
    if role == "adversary":
        return int(state.get("adversary_max_replay_states", XL_ADVERSARY_MAX_REPLAY_STATES))
    raise ValueError(f"unknown replay role: {role}")


def _select_replay_prune_victim(
    entries: list[dict[str, Any]],
    state: dict[str, Any],
) -> int | None:
    for role in ("controller", "adversary"):
        capacity = _role_replay_capacity(state, role)
        total = sum(_entry_role_states(entry, role) for entry in entries)
        if capacity > 0 and total > capacity:
            return next(
                (index for index, entry in enumerate(entries) if _entry_role_states(entry, role) > 0),
                None,
            )

    max_states = int(state.get("max_replay_states", XL_MAX_REPLAY_STATES))
    total = sum(
        int(dict(entry.get("data", {}).get("feature_counts", {}) or {}).get("states", 0) or 0)
        for entry in entries
    )
    if max_states > 0 and total > max_states and entries:
        return 0
    return None


def _prune_replay_partitions(root: Path, state: dict[str, Any]) -> None:
    entries = _partition_entries(root)
    while entries:
        victim_idx = _select_replay_prune_victim(entries, state)
        if victim_idx is None:
            break
        victim = entries.pop(victim_idx)
        _subtract_partition_counts(state, dict(victim.get("data", {}) or {}))
        shutil.rmtree(Path(victim["path"]).parent, ignore_errors=True)


def _training_backoff_status(train_dir: Path) -> dict[str, Any]:
    failure = _read_json_dict(Path(train_dir) / "last_failure.json")
    if not failure:
        return {}
    try:
        until_epoch = float(failure.get("backoff_until_epoch", 0) or 0)
    except Exception:
        return {}
    remaining = int(max(0.0, until_epoch - time.time()))
    if remaining <= 0:
        return {}
    out = dict(failure)
    out["backoff_remaining_s"] = int(remaining)
    return out


def _record_training_failure(train_dir: Path, pid: int, *, reason: str) -> dict[str, Any]:
    train_dir = Path(train_dir)
    current_path = train_dir / "current_training.json"
    current = _read_json_dict(current_path)
    if str(current.get("failure_recorded_at_utc", "") or ""):
        return current
    backoff_sec = int(_train_failure_backoff_sec())
    backoff_until_epoch = time.time() + float(backoff_sec)
    log_path = Path(str(current.get("log_path", "") or ""))
    failure = {
        "pid": int(pid),
        "reason": str(reason),
        "failed_at_utc": utc_now(),
        "backoff_sec": int(backoff_sec),
        "backoff_until_epoch": float(backoff_until_epoch),
        "backoff_until_utc": _epoch_to_utc(backoff_until_epoch),
        "log_path": str(log_path) if str(log_path) else "",
        "log_tail": _tail_text(log_path),
        "last_candidate_model_version_at_launch": int(current.get("last_candidate_model_version_at_launch", 0) or 0),
        "expected_candidate_model_version_min": int(current.get("expected_candidate_model_version_min", 0) or 0),
    }
    current.update({
        "failure_recorded_at_utc": failure["failed_at_utc"],
        "failure_reason": failure["reason"],
        "failure_backoff_sec": int(backoff_sec),
        "failure_backoff_until_epoch": float(backoff_until_epoch),
        "failure_backoff_until_utc": failure["backoff_until_utc"],
        "failure_log_tail": failure["log_tail"],
    })
    atomic_write_json(current_path, current)
    atomic_write_json(train_dir / "last_failure.json", failure)
    with (train_dir / "training_failures.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(failure, sort_keys=True) + "\n")
    return failure


def _prune_defer_reason(root: Path) -> str:
    train_dir = Path(root) / "training"
    pid_path = train_dir / "current.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            pid = 0
        if _pid_alive(pid):
            return f"training_pid_{int(pid)}_running"
    current = _read_json_dict(train_dir / "current_training.json")
    if current and not str(current.get("cycle_finalized_at_utc", "") or ""):
        if str(current.get("failure_recorded_at_utc", "") or ""):
            try:
                until_epoch = float(current.get("failure_backoff_until_epoch", 0) or 0)
            except Exception:
                until_epoch = 0.0
            if until_epoch > time.time():
                return "training_failure_backoff"
        else:
            return "training_cycle_unresolved"
    if _training_cycle_waiting_for_completion(Path(root), train_dir):
        return "training_cycle_waiting_for_completion"
    if _training_backoff_status(train_dir):
        return "training_failure_backoff"
    return ""


def _prune_or_defer_replay_partitions(root: Path, state: dict[str, Any]) -> bool:
    reason = _prune_defer_reason(Path(root))
    if reason:
        state["prune_pending_after_training"] = True
        state["last_prune_deferred_at_utc"] = utc_now()
        state["last_prune_deferred_reason"] = str(reason)
        state["prune_deferred_count"] = int(state.get("prune_deferred_count", 0) or 0) + 1
        return False
    pending = bool(state.pop("prune_pending_after_training", False))
    if pending:
        state["last_deferred_prune_started_at_utc"] = utc_now()
    _prune_replay_partitions(Path(root), state)
    now = utc_now()
    state["last_prune_completed_at_utc"] = now
    if pending:
        state["last_deferred_prune_completed_at_utc"] = now
    return True


def _run_pending_prune_after_training(root: Path, state: dict[str, Any]) -> bool:
    if not bool(state.pop("prune_pending_after_training", False)):
        return False
    state["last_deferred_prune_started_at_utc"] = utc_now()
    _prune_replay_partitions(Path(root), state)
    now = utc_now()
    state["last_prune_completed_at_utc"] = now
    state["last_deferred_prune_completed_at_utc"] = now
    state["last_prune_deferred_reason"] = "completed_training_cycle"
    return True


def _role_distribution_items(
    counts: dict[int, dict[str, int]],
    current_version: int,
) -> tuple[int, list[tuple[int, int]]]:
    current_states = int(counts.get(int(current_version), {}).get("states", 0) if current_version > 0 else 0)
    other_items = [
        (version, int(data.get("states", 0)))
        for version, data in sorted(counts.items())
        if int(version) != int(current_version) and int(data.get("states", 0)) > 0
    ]
    return current_states, other_items


def _append_replay_distribution(root: Path, state: dict[str, Any]) -> None:
    counts_by_role = _partition_version_counts(_partition_entries(root))
    controller_counts = counts_by_role.get("controller", {})
    adversary_counts = counts_by_role.get("adversary", {})
    initial_version = _initial_promoted_model_version()
    legacy_version = int(state.get("promoted_model_version", initial_version) or initial_version)
    controller_version = int(state.get("controller_promoted_model_version", legacy_version) or legacy_version)
    adversary_version = int(state.get("adversary_promoted_model_version", legacy_version) or legacy_version)
    controller_current_states, controller_other = _role_distribution_items(controller_counts, controller_version)
    adversary_current_states, adversary_other = _role_distribution_items(adversary_counts, adversary_version)
    total_feature_states = int(state.get("feature_states", state.get("states", 0)) or 0)
    total_controller_states = int(state.get("feature_controller", state.get("controller", 0)) or 0)
    total_adversary_states = int(state.get("feature_adversary", state.get("adversary", 0)) or 0)
    append_csv_row(root / "replay_distribution.csv", REPLAY_DISTRIBUTION_FIELDS, {
        "states_produced": total_feature_states,
        "controller_states": total_controller_states,
        "adversary_states": total_adversary_states,
        "remaining_buffer_size": max(0, int(state.get("max_replay_states", XL_MAX_REPLAY_STATES)) - total_feature_states),
        "controller_remaining_buffer_size": max(
            0,
            _role_replay_capacity(state, "controller") - total_controller_states,
        ),
        "adversary_remaining_buffer_size": max(
            0,
            _role_replay_capacity(state, "adversary") - total_adversary_states,
        ),
        "current_promoted_controller_model_version": int(controller_version),
        "current_promoted_controller_model_states": int(controller_current_states),
        "other_controller_model_versions": ",".join(str(version) for version, _states in controller_other),
        "other_controller_model_states": ",".join(str(states) for _version, states in controller_other),
        "current_promoted_adversary_model_version": int(adversary_version),
        "current_promoted_adversary_model_states": int(adversary_current_states),
        "other_adversary_model_versions": ",".join(str(version) for version, _states in adversary_other),
        "other_adversary_model_states": ",".join(str(states) for _version, states in adversary_other),
    })

def _manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "shard_manifest.json").read_text(encoding="utf-8"))


def _ingest_shard_impl(root: Path, shard: Path, state: dict[str, Any]) -> bool:
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
    role_limits, admission_reasons = _role_admission_plan(feature_counts)
    admitted_counts = _write_partition(
        root,
        shard=shard,
        manifest=manifest,
        raw_counts=counts,
        feature_counts=feature_counts,
        extra=extra,
        role_limits=role_limits,
    )
    # Legacy monolithic writes are intentionally disabled for new ingests. The
    # trainer still supports old monolithic files as a fallback, but new runs use
    # prunable per-shard partitions to bound disk and memory.
    if False and int(feature_counts["states"]) > 0:
        append_csv_file(feature_replay, replay, extra=extra)
        if policy_rows.exists():
            append_csv_file(feature_policy, policy_rows, extra=extra)
    state["states"] = int(state.get("states", 0)) + int(admitted_counts["states"])
    state["controller"] = int(state.get("controller", 0)) + int(admitted_counts["controller"])
    state["adversary"] = int(state.get("adversary", 0)) + int(admitted_counts["adversary"])
    state["feature_states"] = int(state.get("feature_states", 0)) + int(admitted_counts["states"])
    state["feature_controller"] = int(state.get("feature_controller", 0)) + int(admitted_counts["controller"])
    state["feature_adversary"] = int(state.get("feature_adversary", 0)) + int(admitted_counts["adversary"])
    state["lifetime_admitted_states"] = int(state.get("lifetime_admitted_states", 0)) + int(
        admitted_counts["states"]
    )
    state["new_states_since_last_training"] = int(state.get("new_states_since_last_training", 0)) + int(admitted_counts["states"])
    if int(admitted_counts["states"]) >= 10_000 or int(state.get("new_states_since_last_10k", 0)) + int(admitted_counts["states"]) >= 10_000:
        state["last_10k_time"] = local_time_24h()
        state["new_states_since_last_10k"] = 0
    else:
        state["new_states_since_last_10k"] = int(state.get("new_states_since_last_10k", 0)) + int(admitted_counts["states"])
    state.setdefault("accepted_shards", []).append(shard_id)
    append_csv_row(root / "data_recieved.csv", DATA_RECEIVED_FIELDS, {
        "server_number": manifest.get("worker_id", shard.parent.name),
        "states_produced": int(admitted_counts["states"]),
        "controller_states": int(admitted_counts["controller"]),
        "adversary_states": int(admitted_counts["adversary"]),
        "time": local_time_24h(),
    })
    dropped_controller = max(0, int(feature_counts["controller"]) - int(admitted_counts["controller"]))
    dropped_adversary = max(0, int(feature_counts["adversary"]) - int(admitted_counts["adversary"]))
    admission = {
        "controller_reason": str(admission_reasons["controller"]),
        "adversary_reason": str(admission_reasons["adversary"]),
        "dropped_controller_states": int(dropped_controller),
        "dropped_adversary_states": int(dropped_adversary),
    }
    append_csv_row(root / "ingestion_audit.csv", INGESTION_AUDIT_FIELDS, {
        "time_utc": utc_now(),
        "worker_id": manifest.get("worker_id", shard.parent.name),
        "shard_id": shard_id,
        "controller_model_version": int(manifest.get("controller_model_version", manifest.get("model_version", 0)) or 0),
        "adversary_model_version": int(manifest.get("adversary_model_version", manifest.get("model_version", 0)) or 0),
        "raw_controller_states": int(feature_counts["controller"]),
        "raw_adversary_states": int(feature_counts["adversary"]),
        "admitted_controller_states": int(admitted_counts["controller"]),
        "admitted_adversary_states": int(admitted_counts["adversary"]),
        "dropped_controller_states": int(dropped_controller),
        "dropped_adversary_states": int(dropped_adversary),
        "controller_admission_reason": str(admission_reasons["controller"]),
        "adversary_admission_reason": str(admission_reasons["adversary"]),
    })
    rows_added = int(admitted_counts["states"] if int(admitted_counts["states"]) > 0 else added)
    _record_accepted_manifest(
        root,
        shard=shard,
        manifest=manifest,
        raw_counts=admitted_counts,
        feature_counts=admitted_counts,
        rows_added=rows_added,
        source_counts=counts,
        admission=admission,
    )
    ack_dir = root / "acks" / str(manifest.get("worker_id", shard.parent.name))
    ack_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ack_dir / f"{shard_id}.accepted", {
        "accepted_at_utc": utc_now(),
        "rows_added": rows_added,
        "admitted_counts": admitted_counts,
        "admission": admission,
        "manifest": manifest,
    })
    if bool(XL_KEEP_ACCEPTED_SHARDS):
        accepted_full = root / "accepted_full_shards" / shard.parent.name / shard.name
        accepted_full.parent.mkdir(parents=True, exist_ok=True)
        if accepted_full.exists():
            shutil.rmtree(accepted_full)
        shutil.move(str(shard), str(accepted_full))
    else:
        shutil.rmtree(shard, ignore_errors=True)
    _prune_or_defer_replay_partitions(root, state)
    _append_replay_distribution(root, state)
    return True


def _ingest_shard(root: Path, shard: Path, state: dict[str, Any]) -> bool:
    try:
        return _ingest_shard_impl(root, shard, state)
    except FileNotFoundError as exc:
        # Upload directories can be renamed/cleaned concurrently while the XL
        # scanner is deciding whether to reject or accept a shard. Treat that as
        # a benign race; crashing here stalls all subsequent ingestion.
        state["missing_shard_race_count"] = int(state.get("missing_shard_race_count", 0) or 0) + 1
        state["last_missing_shard_race"] = {
            "shard": str(shard),
            "error": str(exc),
            "time": utc_now(),
        }
        shutil.rmtree(shard, ignore_errors=True)
        return False


def ingest_once(root: Path, *, max_shards: int | None = None) -> int:
    root.mkdir(parents=True, exist_ok=True)
    _ensure_headers(root)
    state = _load_state(root)
    state.setdefault("max_replay_states", XL_MAX_REPLAY_STATES)
    state.setdefault("controller_max_replay_states", XL_CONTROLLER_MAX_REPLAY_STATES)
    state.setdefault("adversary_max_replay_states", XL_ADVERSARY_MAX_REPLAY_STATES)

    if not bool(state.get("role_partition_migration_complete", False)):
        _migrate_mixed_replay_partitions(root, state)
        _prune_replay_partitions(root, state)
        _rebuild_replay_state_counts(root, state)
        _append_replay_distribution(root, state)
        _save_state(root, state)

    if bool(XL_PAUSE_INGEST_WHILE_PRUNE_DEFERRED):
        pause_reason = _prune_defer_reason(root)
        # Training backoff should prevent immediate relaunch/pruning, but it
        # must not block ingestion. Otherwise old worker uploads can keep
        # occupying disk even though XL is able to safely accept them.
        if pause_reason == "training_failure_backoff":
            pause_reason = ""
        if pause_reason:
            state["ingest_paused"] = True
            state["ingest_paused_reason"] = str(pause_reason)
            state["last_ingest_paused_at_utc"] = utc_now()
            _append_xl_status(root, state)
            _save_state(root, state)
            return 0

    state["ingest_paused"] = False
    state["ingest_paused_reason"] = ""
    limit = int(max_shards if max_shards is not None else XL_MAX_INGEST_SHARDS_PER_LOOP)
    ingested = 0
    limit_reached = False
    incoming = root / "incoming"
    for worker_dir in sorted(incoming.glob("*")) if incoming.exists() else []:
        if not worker_dir.is_dir():
            continue
        for shard in sorted(worker_dir.glob("*")):
            if limit > 0 and ingested >= limit:
                limit_reached = True
                break
            if not shard.is_dir() or not (shard / "shard_manifest.json").exists():
                continue
            if _ingest_shard(root, shard, state):
                ingested += 1
                _save_state(root, state)
        if limit_reached:
            break
    state["last_ingest_limit"] = int(limit)
    state["last_ingest_limit_reached"] = bool(limit_reached)
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


def _controller_policy_sample_count(controller_states: int) -> int:
    return min(int(controller_states), int(XL_CONTROLLER_POLICY_SAMPLE_CAP))


def _policy_sample_requirement(value_state_minimum: int, policy_sample_cap: int) -> int:
    """Treat the policy cap as a maximum, not an additional replay gate."""

    return min(max(0, int(value_state_minimum)), max(0, int(policy_sample_cap)))


def write_training_gate_status(root: Path) -> dict[str, Any]:
    state = _load_state(root)
    raw_total_states = int(state.get("states", 0))
    raw_controller_states = int(state.get("controller", 0))
    raw_adversary_states = int(state.get("adversary", 0))
    total_states = int(state.get("feature_states", 0))
    controller_states = int(state.get("feature_controller", 0))
    adversary_states = int(state.get("feature_adversary", 0))
    lifetime_admitted = int(state.get("lifetime_admitted_states", total_states) or 0)
    legacy_new_states = int(state.get("new_states_since_last_training", 0) or 0)
    baseline = int(
        state.get(
            "last_training_gate_lifetime_admitted_states",
            max(0, lifetime_admitted - legacy_new_states),
        )
        or 0
    )
    new_states = max(0, lifetime_admitted - baseline)
    train_sample_floor = _training_sample_floor(total_states)
    controller_policy_samples = _controller_policy_sample_count(controller_states)
    adversary_policy_samples = _adversary_policy_sample_count(adversary_states)
    controller_policy_requirement = _policy_sample_requirement(
        MIN_CONTROLLER_STATES_FOR_EVAL,
        XL_CONTROLLER_POLICY_SAMPLE_CAP,
    )
    adversary_policy_requirement = _policy_sample_requirement(
        MIN_ADVERSARY_STATES_FOR_EVAL,
        ADVERSARY_POLICY_SAMPLE_CAP,
    )
    controller_promotions_completed = int(state.get("controller_promotions_completed", 0) or 0)
    adversary_promotions_completed = int(state.get("adversary_promotions_completed", 0) or 0)
    promotions_completed = max(
        int(state.get("promotions_completed", 0) or 0),
        int(controller_promotions_completed),
        int(adversary_promotions_completed),
    )
    promotion_cap_reached = promotions_completed >= int(MAX_PROMOTIONS)
    has_enough_new_replay = new_states >= int(TRAIN_TRIGGER_NEW_STATES)
    has_enough_controller = controller_states >= int(MIN_CONTROLLER_STATES_FOR_EVAL)
    has_enough_adversary = adversary_states >= int(MIN_ADVERSARY_STATES_FOR_EVAL)
    has_enough_controller_policy = controller_policy_samples >= int(controller_policy_requirement)
    has_enough_adversary_policy = adversary_policy_samples >= int(adversary_policy_requirement)
    has_train_sample_floor = train_sample_floor > 0 and total_states >= train_sample_floor
    can_train = bool(
        (not promotion_cap_reached)
        and has_enough_new_replay
        and has_enough_controller
        and has_enough_adversary
        and has_enough_controller_policy
        and has_enough_adversary_policy
        and has_train_sample_floor
    )
    data = {
        "can_train": can_train,
        "can_eval": bool(has_enough_adversary),
        "promotion_win_rate_threshold": float(PROMOTION_WIN_RATE_THRESHOLD),
        "role_promotion_win_threshold": int(ROLE_PROMOTION_WIN_THRESHOLD),
        "promotion_required_wins_per_role": int(ROLE_PROMOTION_WIN_THRESHOLD) + 1,
        "max_promotions": int(MAX_PROMOTIONS),
        "promotions_completed": int(promotions_completed),
        "controller_promotions_completed": int(controller_promotions_completed),
        "adversary_promotions_completed": int(adversary_promotions_completed),
        "controller_promoted_model_version": int(state.get("controller_promoted_model_version", state.get("promoted_model_version", 100)) or 100),
        "adversary_promoted_model_version": int(state.get("adversary_promoted_model_version", state.get("promoted_model_version", 100)) or 100),
        "promotion_cap_reached": bool(promotion_cap_reached),
        "new_states_since_last_training": new_states,
        "lifetime_admitted_states": int(lifetime_admitted),
        "last_training_gate_lifetime_admitted_states": int(baseline),
        "train_trigger_new_states": int(TRAIN_TRIGGER_NEW_STATES),
        "total_states": total_states,
        "controller_states": controller_states,
        "adversary_states": adversary_states,
        "raw_total_states": raw_total_states,
        "raw_controller_states": raw_controller_states,
        "raw_adversary_states": raw_adversary_states,
        "min_controller_states_for_eval": int(MIN_CONTROLLER_STATES_FOR_EVAL),
        "min_adversary_states_for_eval": int(MIN_ADVERSARY_STATES_FOR_EVAL),
        "controller_policy_sample_count": int(controller_policy_samples),
        "controller_policy_sample_cap": int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
        "controller_policy_sample_requirement": int(controller_policy_requirement),
        "adversary_policy_sample_count": int(adversary_policy_samples),
        "adversary_policy_sample_cap": int(ADVERSARY_POLICY_SAMPLE_CAP),
        "adversary_policy_sample_requirement": int(adversary_policy_requirement),
        "train_sample_floor": int(train_sample_floor),
        "has_enough_new_replay": bool(has_enough_new_replay),
        "has_enough_controller": bool(has_enough_controller),
        "has_enough_adversary": bool(has_enough_adversary),
        "has_enough_controller_policy": bool(has_enough_controller_policy),
        "has_enough_adversary_policy": bool(has_enough_adversary_policy),
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


def _highest_model_dir_version(root: Path) -> int:
    versions: list[int] = []
    models_dir = Path(root) / "models"
    for manifest_path in models_dir.glob("Model_Version*/candidate_manifest.json") if models_dir.exists() else []:
        suffix = manifest_path.parent.name.removeprefix("Model_Version")
        if suffix.isdigit():
            versions.append(int(suffix))
    return max(versions) if versions else 0


def _latest_failed_candidate_version(root: Path) -> int:
    state = _load_state(Path(root))
    completed = int(state.get("last_training_cycle_completed_model_version", 0) or 0)
    failed: list[int] = []
    for manifest_path in (Path(root) / "models").glob("Model_Version*/candidate_manifest.json"):
        suffix = manifest_path.parent.name.removeprefix("Model_Version")
        if not suffix.isdigit() or int(suffix) <= completed:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("eval_status", "") or "") == "failed":
            failed.append(int(suffix))
    return max(failed) if failed else 0


def _launch_existing_candidate_eval(
    root: Path,
    train_dir: Path,
    gate: dict[str, Any],
    version: int,
) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(train_dir) / f"eval_retry_v{int(version)}_{stamp}.log"
    cmd = [
        sys.executable,
        "-m",
        "vidur.AlphaGoZero.agz_train_eval_promote",
        "--output-root",
        str(Path(root)),
        "--eval-existing-version",
        str(int(version)),
        "--mcts-iterations",
        str(int(AGZ_EVAL_MCTS_ITERATIONS)),
    ]
    state_at_launch = _load_state(Path(root))
    previous_current = _read_json_dict(Path(train_dir) / "current_training.json")
    previous_version = int(
        previous_current.get("retry_candidate_model_version", 0)
        or previous_current.get("expected_candidate_model_version_min", 0)
        or 0
    )
    # Re-evaluation consumes the replay gate captured when this candidate was
    # trained. States received after that launch remain available to the next
    # training cycle.
    retry_gate = dict(previous_current.get("gate", {}) or {}) if previous_version == int(version) else dict(gate)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            env=_training_subprocess_env(root),
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (Path(train_dir) / "current.pid").write_text(str(int(proc.pid)) + "\n", encoding="utf-8")
    atomic_write_json(Path(train_dir) / "current_training.json", {
        "pid": int(proc.pid),
        "cmd": cmd,
        "operation": "eval_existing_candidate_retry",
        "retry_candidate_model_version": int(version),
        "log_path": str(log_path),
        "mcts_iterations": int(AGZ_EVAL_MCTS_ITERATIONS),
        "started_at_utc": utc_now(),
        "gate": retry_gate,
        "lifetime_admitted_states_at_launch": int(
            retry_gate.get("lifetime_admitted_states", state_at_launch.get("lifetime_admitted_states", 0)) or 0
        ),
        "last_candidate_model_version_at_launch": int(state_at_launch.get("last_candidate_model_version", 0) or 0),
        "last_training_cycle_completed_model_version_at_launch": int(
            state_at_launch.get("last_training_cycle_completed_model_version", 0) or 0
        ),
        "expected_candidate_model_version_min": int(version),
    })
    return {
        "training_launched": False,
        "evaluation_retry_launched": True,
        "evaluation_retry_version": int(version),
        "training_running_pid": int(proc.pid),
        "training_log": str(log_path),
    }


def _finalize_completed_training_cycle_if_needed(root: Path, train_dir: Path) -> bool:
    current = train_dir / "current_training.json"
    pid_path = train_dir / "current.pid"
    if not current.exists():
        return False
    try:
        data = json.loads(current.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if (
        str(data.get("coordinator_cycle_finalized_at_utc", "") or "")
        or str(data.get("cycle_finalized_at_utc", "") or "")
    ):
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
        return True

    min_version = int(data.get("expected_candidate_model_version_min", 0) or 0)
    if min_version <= 0:
        min_version = int(data.get("last_candidate_model_version_at_launch", 0) or 0) + 1

    state = _load_state(root)
    version = max(
        int(state.get("last_candidate_model_version", 0) or 0),
        int(_highest_model_dir_version(Path(root))),
    )
    if version < min_version:
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

    if "next_rollout_horizon_rounded_sec" in manifest:
        runtime_config = activate_manifest_horizon(
            root,
            candidate_version=int(version),
            manifest=manifest,
        )
        from vidur.AlphaGoZero.agz_train_eval_promote import (
            _broadcast_runtime_search_config_to_workers,
        )

        _broadcast_runtime_search_config_to_workers(root)
        data["activated_runtime_search_config"] = runtime_config

    gate = dict(data.get("gate", {}) or {})
    consumed = int(gate.get("new_states_since_last_training", 0) or 0)
    lifetime_admitted = int(
        state.get("lifetime_admitted_states", state.get("feature_states", 0)) or 0
    )
    launch_lifetime = int(
        data.get("lifetime_admitted_states_at_launch", 0)
        or gate.get("lifetime_admitted_states", 0)
        or gate.get("total_states", 0)
        or 0
    )
    if launch_lifetime > 0 and lifetime_admitted >= launch_lifetime:
        state["last_training_gate_lifetime_admitted_states"] = int(launch_lifetime)
        state["new_states_since_last_training"] = int(lifetime_admitted - launch_lifetime)
    else:
        available = int(state.get("new_states_since_last_training", 0) or 0)
        state["new_states_since_last_training"] = max(0, available - consumed)
    state["states_consumed_by_last_training_cycle"] = int(consumed)
    state["last_training_completed_at_utc"] = utc_now()
    state["last_training_cycle_completed_model_version"] = int(version)
    prior_candidate = int(state.get("last_candidate_model_version", 0) or 0)
    if int(version) > prior_candidate:
        state["candidate_counter"] = int(state.get("candidate_counter", 0) or 0) + 1
    state["last_candidate_model_version"] = int(version)
    pruned_after_training = _run_pending_prune_after_training(root, state)
    _save_state(root, state)
    if pruned_after_training:
        _append_replay_distribution(root, state)

    finalized_at = utc_now()
    data["cycle_finalized_at_utc"] = finalized_at
    data["cycle_finalized_model_version"] = int(version)
    data["coordinator_cycle_finalized_at_utc"] = finalized_at
    atomic_write_json(current, data)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass
    return True


def _training_cycle_waiting_for_completion(root: Path, train_dir: Path) -> bool:
    current = train_dir / "current_training.json"
    if not current.exists():
        return False
    try:
        data = json.loads(current.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(data.get("coordinator_cycle_finalized_at_utc", "") or ""):
        return False
    min_version = int(data.get("expected_candidate_model_version_min", 0) or 0)
    if min_version <= 0:
        min_version = int(data.get("last_candidate_model_version_at_launch", 0) or 0) + 1
    state = _load_state(root)
    version = max(
        int(state.get("last_candidate_model_version", 0) or 0),
        int(_highest_model_dir_version(Path(root))),
    )
    if version < min_version:
        return False
    manifest_path = Path(root) / "models" / f"Model_Version{version}" / "candidate_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    status = str(manifest.get("eval_status", "") or "")
    return status in {"", "running"}


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
        if _training_cycle_waiting_for_completion(Path(root), train_dir):
            return {"training_launched": False, "training_running_pid": 0, "training_waiting_for_cycle_completion": True}
        failure = _record_training_failure(
            train_dir,
            int(pid),
            reason="training_process_exited_without_completed_cycle",
        )
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
        return {
            "training_launched": False,
            "training_running_pid": 0,
            "training_failed_pid": int(pid),
            "training_failure_recorded": True,
            "training_backoff_until_utc": str(failure.get("backoff_until_utc", "")),
            "training_backoff_remaining_s": int(max(0.0, float(failure.get("backoff_until_epoch", 0) or 0) - time.time())),
        }
    current_path = train_dir / "current_training.json"
    current_data = _read_json_dict(current_path)
    current_unfinalized = bool(current_data) and not (
        str(current_data.get("coordinator_cycle_finalized_at_utc", "") or "")
        or str(current_data.get("cycle_finalized_at_utc", "") or "")
    )
    if current_unfinalized and _finalize_completed_training_cycle_if_needed(Path(root), train_dir):
        return {
            "training_launched": False,
            "training_running_pid": 0,
            "training_cycle_finalized": True,
            "training_cycle_recovered_without_pid": True,
        }
    backoff = _training_backoff_status(train_dir)
    if backoff:
        return {
            "training_launched": False,
            "training_running_pid": 0,
            "training_backoff_active": True,
            "training_backoff_until_utc": str(backoff.get("backoff_until_utc", "")),
            "training_backoff_remaining_s": int(backoff.get("backoff_remaining_s", 0) or 0),
            "training_last_failure_reason": str(backoff.get("reason", "")),
        }
    failed_version = _latest_failed_candidate_version(Path(root))
    if failed_version > 0:
        return _launch_existing_candidate_eval(Path(root), train_dir, gate, failed_version)
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
        "--mcts-iterations",
        str(int(AGZ_EVAL_MCTS_ITERATIONS)),
    ]
    state_at_launch = _load_state(Path(root))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            env=_training_subprocess_env(root),
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(int(proc.pid)) + "\n", encoding="utf-8")
    atomic_write_json(train_dir / "current_training.json", {
        "pid": int(proc.pid),
        "cmd": cmd,
        "log_path": str(log_path),
        "mcts_iterations": int(AGZ_EVAL_MCTS_ITERATIONS),
        "started_at_utc": utc_now(),
        "gate": gate,
        "lifetime_admitted_states_at_launch": int(
            gate.get("lifetime_admitted_states", state_at_launch.get("lifetime_admitted_states", 0)) or 0
        ),
        "last_candidate_model_version_at_launch": int(state_at_launch.get("last_candidate_model_version", 0) or 0),
        "last_training_cycle_completed_model_version_at_launch": int(state_at_launch.get("last_training_cycle_completed_model_version", 0) or 0),
        "expected_candidate_model_version_min": int(_highest_model_dir_version(Path(root))) + 1,
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
    p.add_argument("--max-ingest-shards", type=int, default=int(XL_MAX_INGEST_SHARDS_PER_LOOP))
    p.add_argument("--migrate-role-partitions-only", action="store_true")
    p.add_argument("--migration-workers", type=int, default=1)
    p.add_argument(
        "--spot-selfplay-total-parallel-games",
        type=int,
        default=(
            int(os.environ["AGZ_SPOT_SELFPLAY_TOTAL_PARALLEL_GAMES"])
            if os.environ.get("AGZ_SPOT_SELFPLAY_TOTAL_PARALLEL_GAMES")
            else None
        ),
    )
    p.add_argument(
        "--spot-eval-total-parallel-games",
        type=int,
        default=(
            int(os.environ["AGZ_SPOT_EVAL_TOTAL_PARALLEL_GAMES"])
            if os.environ.get("AGZ_SPOT_EVAL_TOTAL_PARALLEL_GAMES")
            else None
        ),
    )
    p.add_argument(
        "--spot-selfplay-lease-sec",
        type=int,
        default=int(os.environ.get("AGZ_SPOT_SELFPLAY_LEASE_SEC", "300")),
    )
    p.add_argument(
        "--spot-worker-heartbeat-timeout-sec",
        type=int,
        default=int(
            os.environ.get("AGZ_SPOT_WORKER_HEARTBEAT_TIMEOUT_SEC", "300")
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).expanduser()
    ensure_runtime_search_config(root)
    existing_scheduler = _read_json_dict(
        root / "spot_work" / "control" / "scheduler_config.json"
    )
    selfplay_limit = (
        int(args.spot_selfplay_total_parallel_games)
        if args.spot_selfplay_total_parallel_games is not None
        else int(existing_scheduler.get("selfplay_total_parallel_games", 0) or 0)
    )
    eval_limit = (
        int(args.spot_eval_total_parallel_games)
        if args.spot_eval_total_parallel_games is not None
        else int(existing_scheduler.get("eval_total_parallel_games", 0) or 0)
    )
    _refresh_spot_scheduler(
        root,
        selfplay_limit=selfplay_limit,
        eval_limit=eval_limit,
        selfplay_lease_sec=int(args.spot_selfplay_lease_sec),
        worker_heartbeat_timeout_sec=int(
            args.spot_worker_heartbeat_timeout_sec
        ),
    )
    if args.init_eval_dir is not None:
        print(init_eval_dir(root, int(args.init_eval_dir)))
        return
    if bool(args.migrate_role_partitions_only):
        _ensure_headers(root)
        state = _load_state(root)
        migrated = _migrate_mixed_replay_partitions(root, state, workers=int(args.migration_workers))
        _prune_replay_partitions(root, state)
        _rebuild_replay_state_counts(root, state)
        _append_replay_distribution(root, state)
        _save_state(root, state)
        print(json.dumps({"migrated_partitions": int(migrated), "total_states": int(state.get("feature_states", 0))}, sort_keys=True))
        return
    while True:
        _refresh_spot_scheduler(
            root,
            selfplay_limit=selfplay_limit,
            eval_limit=eval_limit,
            selfplay_lease_sec=int(args.spot_selfplay_lease_sec),
            worker_heartbeat_timeout_sec=int(
                args.spot_worker_heartbeat_timeout_sec
            ),
        )
        _drain_replay_index_builds()
        n = ingest_once(root, max_shards=int(args.max_ingest_shards))
        _drain_replay_index_builds()
        gate = write_training_gate_status(root)
        train_status = maybe_launch_training(root, gate) if bool(args.enable_training) else {"training_launched": False, "training_disabled": True}
        print(json.dumps({"ingested_shards": n, **gate, **train_status}, sort_keys=True), flush=True)
        if not args.loop:
            break
        time.sleep(float(args.poll_sec))


if __name__ == "__main__":
    main()
