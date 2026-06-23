#!/usr/bin/env python3
"""Build controller action features for GV3 controller prior/Q-head targets.

This script emits action-only 43D features. The 226D state features are already
produced by the parent-state feature pipeline, so policy/Q-head training should
join state rows with these action rows by (worker, state_id, canon_action_index).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_MODEL_LABEL = "worker1_200k_alpha2__hgb_sq_63leaf_1050iter_a2__v47__"
DEFAULT_REMOTE_MODEL_PATH = (
    "{remote_repo}/simulator_output/GV3_Agent/bellman_multiserver_HGB/"
    "policy_target_models/{model_label}/model.joblib"
)
DEFAULT_LOCAL_MODEL_PATH = (
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/bellman_multiserver_HGB/latest_version/"
    "worker1_200k_alpha2/hgb_sq_63leaf_1050iter_a2/Model_Version47/model.joblib"
)
DEFAULT_LOCAL_SMOKE_DIR = (
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/bellman_multiserver_HGB/bellman_prior_canon_test"
)
DEFAULT_SMOKE_STATE_IDS_CSV = (
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/bellman_multiserver_HGB/bellman_prior_canon_test/"
    "target_smoke_full.csv"
)

MAX_PREFILL_ACTION_ALLOC = 4096.0
MAX_DECODE_REQUESTS = 100.0
MAX_REQUESTS_PER_LAUNCH_WINDOW = 7
ACTION_FEATURE_VERSION = "controller_action_features_effectcanon_v1"


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    server_index: int
    hops_min: int
    hops_max: int

    @property
    def root_dir_name(self) -> str:
        return f"server_{self.server_index:02d}_{self.host}_hops_{self.hops_min}_{self.hops_max}"


WORKERS: tuple[WorkerSpec, ...] = (
    WorkerSpec("worker1", "bellman-classical-worker-1", 1, 151, 300),
    WorkerSpec("worker2", "bellman-classical-worker-2", 2, 301, 450),
    WorkerSpec("worker3", "bellman-classical-worker-3", 3, 451, 600),
    WorkerSpec("worker4", "bellman-classical-worker-4", 4, 601, 750),
)

ACTION_FEATURE_NAMES: list[str] = [
    "a_total_prefill_alloc_norm",
    "a_total_decode_alloc_norm",
    "a_n_prefill_alloc_reqs_norm",
    "a_n_decode_alloc_reqs_norm",
    "a_n_evicted_prefill_norm",
    "a_n_evicted_decode_norm",
    "a_has_prefill_alloc",
    "a_has_decode_alloc",
    "a_has_eviction",
    "a_strict_noop",
    "a_n_evicted_decode_late_over_0p5_norm",
    "a_n_evicted_prefill_late_over_0p5_norm",
    "a_n_evicted_prefill_missed_deadline_norm",
    "a_evict_includes_highest_lateness_prefill",
    "a_evict_includes_highest_lateness_decode",
]
for _slot in range(MAX_REQUESTS_PER_LAUNCH_WINDOW):
    ACTION_FEATURE_NAMES.extend(
        [
            f"a_prefill_slot_{_slot}_selected",
            f"a_prefill_slot_{_slot}_alloc_norm",
            f"a_prefill_slot_{_slot}_alloc_frac_of_remaining",
            f"a_prefill_slot_{_slot}_evicted",
        ]
    )
assert len(ACTION_FEATURE_NAMES) == 43

FEATURE_ROW_FIELDS = [
    "worker",
    "host",
    "state_id",
    "root_id",
    "player",
    "root_player",
    "root_depth",
    "history_hops",
    "canon_action_index",
    "action_repr",
    "action_feature_repr",
]
SUMMARY_ROW_FIELDS = [
    "worker",
    "host",
    "state_id",
    "root_id",
    "player",
    "root_player",
    "valid_action_count",
    "canonical_action_count",
    "feature_action_count",
]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vidur" / "Game_Version3").is_dir():
            return parent
    return p.parents[3]


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _ssh(host: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, cmd], check=check)


def _rsync(src: str, dst: str) -> subprocess.CompletedProcess[str]:
    return _run(["rsync", "-az", src, dst])


def _base_path(remote_repo: str, remote_output_base: str) -> str:
    return f"{remote_repo.rstrip('/')}/{remote_output_base.strip('/')}"


def _worker_root_dir(worker: WorkerSpec, *, remote_repo: str, remote_output_base: str, experiment_name: str) -> str:
    return f"{_base_path(remote_repo, remote_output_base)}/{experiment_name}/{worker.root_dir_name}"


def _remote_model_path(args: argparse.Namespace) -> str:
    return str(args.remote_model_path).format(
        remote_repo=str(args.remote_repo).rstrip("/"),
        model_label=str(args.model_label),
    )


def _target_base(args: argparse.Namespace) -> str:
    return (
        f"{_base_path(args.remote_repo, args.remote_output_base)}/"
        f"{args.experiment_name}/controller_prior_action_features/{ACTION_FEATURE_VERSION}"
    )


def _parse_workers(raw: str | None) -> list[WorkerSpec]:
    if not raw:
        return list(WORKERS)
    requested = {x.strip() for x in str(raw).split(",") if x.strip()}
    out = [w for w in WORKERS if w.worker_id in requested or w.host in requested]
    if not out:
        raise ValueError(f"no workers selected from {raw!r}")
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_smoke_state_ids_by_worker(path: str | None) -> dict[str, list[int]]:
    if not path:
        return {}
    out: dict[str, set[int]] = {}
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            worker = str(row.get("worker", "")).strip()
            if worker:
                out.setdefault(worker, set()).add(int(row["state_id"]))
    return {worker: sorted(ids) for worker, ids in out.items()}


def _parse_state_ids(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return sorted({int(x.strip()) for x in str(raw).replace("\n", ",").split(",") if x.strip()})


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _norm01(value: float, denom: float) -> float:
    d = float(denom)
    if d <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, float(value) / d)))


def _request_lookup(env: Any, state: Any) -> dict[int, Any]:
    if hasattr(env, "_build_request_lookup"):
        try:
            return {int(k): v for k, v in env._build_request_lookup(state.simulator, state=state).items()}
        except TypeError:
            return {int(k): v for k, v in env._build_request_lookup(state.simulator).items()}
    return {int(k): v for k, v in env._req_map(state.simulator).items()}


def _remaining_prefill(env: Any, req: Any) -> int:
    fn = getattr(env, "_remaining_prefill", None)
    if callable(fn):
        return max(0, int(fn(req)))
    return max(0, _safe_int(_safe_getattr(req, "num_prefill_tokens", 0)) - _safe_int(_safe_getattr(req, "num_processed_prefill_tokens", 0)))


def _remaining_decode(env: Any, req: Any) -> int:
    fn = getattr(env, "_remaining_decode", None)
    if callable(fn):
        return max(0, int(fn(req)))
    return max(0, _safe_int(_safe_getattr(req, "num_decode_tokens", 0)) - _safe_int(_safe_getattr(req, "num_processed_decode_tokens", 0)))


def _is_prefill_done(req: Any) -> bool:
    return bool(_safe_getattr(req, "_is_prefill_complete", _safe_getattr(req, "is_prefill_complete", False)))


def _arrived_at(req: Any) -> float:
    return _safe_float(_safe_getattr(req, "arrived_at", _safe_getattr(req, "_arrived_at", 0.0)))


def _prefill_slo_time(req: Any) -> float:
    return _safe_float(_safe_getattr(req, "prefill_slo_time", _safe_getattr(req, "_prefill_slo_time", 0.0)))


def _prefill_lateness(state: Any, rid: int, req: Any) -> float:
    per = dict(_safe_getattr(state.stats, "per_request_prefill_lateness", {}) or {})
    if int(rid) in per:
        return max(0.0, _safe_float(per[int(rid)]))
    sim_time = _safe_float(_safe_getattr(state.simulator, "_time", 0.0))
    return max(0.0, sim_time - (_arrived_at(req) + _prefill_slo_time(req)))


def _decode_lateness(state: Any, rid: int) -> float:
    per = dict(_safe_getattr(state.stats, "per_request_decode_lateness", {}) or {})
    return max(0.0, _safe_float(per.get(int(rid), 0.0)))


def _active_phase_maps(env: Any, state: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    req_map = _request_lookup(env, state)
    active_ids = {int(x) for x in (_safe_getattr(state.stats, "active_request_ids", set()) or set())} or set(req_map.keys())
    prefill: dict[int, Any] = {}
    decode: dict[int, Any] = {}
    for rid in sorted(active_ids):
        req = req_map.get(int(rid))
        if req is None or bool(_safe_getattr(req, "completed", False)):
            continue
        if (not _is_prefill_done(req)) and _remaining_prefill(env, req) > 0:
            prefill[int(rid)] = req
        elif _is_prefill_done(req) and _remaining_decode(env, req) > 0:
            decode[int(rid)] = req
    return prefill, decode


def _eviction_rule(action: Any) -> str:
    strategy = str(_safe_getattr(action, "strategy", "") or "")
    return strategy.split("|", 1)[1] if strategy.startswith("GV2|") else (strategy or "evict_none")


def _derive_evicted_ids(bundle: Any, state: Any, action: Any) -> list[int]:
    existing = getattr(action, "_evicted_request_ids", None)
    if existing is not None:
        return sorted(int(x) for x in (existing or []))
    rule = _eviction_rule(action)
    if rule == "evict_none":
        evicted: list[int] = []
    else:
        from vidur.Game_Version3.player_sample_actions import GV2PlayerActionSampler
        sampler = GV2PlayerActionSampler(bundle.pipeline_cfg.game_v2)
        req_views = sampler._build_req_views(
            sim_time=_safe_float(_safe_getattr(state.simulator, "_time", 0.0)),
            request_lookup=_request_lookup(bundle.env, state),
            per_request_prefill_lateness=dict(_safe_getattr(state.stats, "per_request_prefill_lateness", {}) or {}),
            per_request_decode_lateness=dict(_safe_getattr(state.stats, "per_request_decode_lateness", {}) or {}),
        )
        evicted = sampler._eviction_targets(
            rule=rule,
            req_views=req_views,
            violated_request_ids={int(x) for x in (_safe_getattr(state.stats, "violated_request_ids", set()) or set())},
        )
    setattr(action, "_evicted_request_ids", tuple(sorted(int(x) for x in evicted)))
    return sorted(int(x) for x in evicted)


def _controller_action_effect_key(action: Any) -> tuple[Any, ...]:
    token_alloc = tuple(sorted((int(k), int(v)) for k, v in dict(_safe_getattr(action, "token_allocations", {}) or {}).items()))
    prefill_alloc = tuple(sorted((int(k), int(v)) for k, v in dict(_safe_getattr(action, "prefill_allocations", {}) or {}).items()))
    decode_alloc = tuple(sorted((int(k), int(v)) for k, v in dict(_safe_getattr(action, "decode_allocations", {}) or {}).items()))
    evicted_ids = tuple(sorted(int(x) for x in (getattr(action, "_evicted_request_ids", ()) or ())))
    return (token_alloc, prefill_alloc, decode_alloc, evicted_ids)


def _effect_canonical_indices(actions_by_index: list[Any | None], valid_indices: list[int]) -> list[int]:
    seen: dict[tuple[Any, ...], int] = {}
    canonical: list[int] = []
    for idx in valid_indices:
        action = actions_by_index[int(idx)]
        if action is None:
            continue
        key = _controller_action_effect_key(action)
        if key in seen:
            continue
        seen[key] = int(idx)
        canonical.append(int(idx))
    return canonical


def _prefill_slot_map(env: Any, state: Any) -> dict[int, tuple[int, int]]:
    prefill, _decode = _active_phase_maps(env, state)
    sim_time = _safe_float(_safe_getattr(state.simulator, "_time", 0.0))
    violated = {int(x) for x in (_safe_getattr(state.stats, "violated_request_ids", set()) or set())}
    enriched: list[dict[str, Any]] = []
    for rid, req in prefill.items():
        deadline = _arrived_at(req) + _prefill_slo_time(req)
        enriched.append({"rid": int(rid), "remaining": _remaining_prefill(env, req), "slack": max(0.0, deadline - sim_time), "violated": int(rid) in violated})
    if enriched and all(bool(x["violated"]) for x in enriched):
        enriched.sort(key=lambda p: int(p["rid"]))
    else:
        enriched.sort(key=lambda p: (float(p["slack"]), int(p["rid"])))
    return {int(item["rid"]): (slot, int(item["remaining"])) for slot, item in enumerate(enriched[:MAX_REQUESTS_PER_LAUNCH_WINDOW])}


def build_action_feature_dict(bundle: Any, state: Any, action: Any) -> dict[str, float]:
    prefill_reqs, decode_reqs = _active_phase_maps(bundle.env, state)
    evicted_ids = set(_derive_evicted_ids(bundle, state, action))
    evicted_prefill = sorted(int(x) for x in evicted_ids if int(x) in prefill_reqs)
    evicted_decode = sorted(int(x) for x in evicted_ids if int(x) in decode_reqs)
    prefill_alloc = {int(k): int(v) for k, v in dict(_safe_getattr(action, "prefill_allocations", {}) or {}).items()}
    decode_alloc = {int(k): int(v) for k, v in dict(_safe_getattr(action, "decode_allocations", {}) or {}).items()}

    row: dict[str, float] = {
        "a_total_prefill_alloc_norm": _norm01(sum(prefill_alloc.values()), MAX_PREFILL_ACTION_ALLOC),
        "a_total_decode_alloc_norm": _norm01(sum(decode_alloc.values()), MAX_DECODE_REQUESTS),
        "a_n_prefill_alloc_reqs_norm": _norm01(len(prefill_alloc), MAX_REQUESTS_PER_LAUNCH_WINDOW),
        "a_n_decode_alloc_reqs_norm": _norm01(len(decode_alloc), MAX_DECODE_REQUESTS),
        "a_n_evicted_prefill_norm": _norm01(len(evicted_prefill), max(1, len(prefill_reqs))),
        "a_n_evicted_decode_norm": _norm01(len(evicted_decode), MAX_DECODE_REQUESTS),
        "a_has_prefill_alloc": 1.0 if sum(prefill_alloc.values()) > 0 else 0.0,
        "a_has_decode_alloc": 1.0 if sum(decode_alloc.values()) > 0 else 0.0,
        "a_has_eviction": 1.0 if evicted_ids else 0.0,
        "a_strict_noop": 1.0 if not prefill_alloc and not decode_alloc and not evicted_ids else 0.0,
    }
    row["a_n_evicted_decode_late_over_0p5_norm"] = _norm01(sum(1 for rid in evicted_decode if _decode_lateness(state, rid) > 0.5), MAX_DECODE_REQUESTS)
    row["a_n_evicted_prefill_late_over_0p5_norm"] = _norm01(sum(1 for rid in evicted_prefill if _prefill_lateness(state, rid, prefill_reqs[rid]) > 0.5), MAX_REQUESTS_PER_LAUNCH_WINDOW)
    sim_time = _safe_float(_safe_getattr(state.simulator, "_time", 0.0))
    row["a_n_evicted_prefill_missed_deadline_norm"] = _norm01(sum(1 for rid in evicted_prefill if sim_time > (_arrived_at(prefill_reqs[rid]) + _prefill_slo_time(prefill_reqs[rid]))), MAX_REQUESTS_PER_LAUNCH_WINDOW)
    hp = max(prefill_reqs, key=lambda rid: (_prefill_lateness(state, rid, prefill_reqs[rid]), -int(rid))) if prefill_reqs else None
    hd = max(decode_reqs, key=lambda rid: (_decode_lateness(state, rid), -int(rid))) if decode_reqs else None
    row["a_evict_includes_highest_lateness_prefill"] = 1.0 if hp is not None and int(hp) in evicted_ids else 0.0
    row["a_evict_includes_highest_lateness_decode"] = 1.0 if hd is not None and int(hd) in evicted_ids else 0.0

    slot_map = _prefill_slot_map(bundle.env, state)
    by_slot = {slot: (rid, remaining) for rid, (slot, remaining) in slot_map.items()}
    for slot in range(MAX_REQUESTS_PER_LAUNCH_WINDOW):
        rid, remaining = by_slot.get(int(slot), (-1, 0))
        alloc = int(prefill_alloc.get(int(rid), 0)) if rid >= 0 else 0
        row[f"a_prefill_slot_{slot}_selected"] = 1.0 if alloc > 0 else 0.0
        row[f"a_prefill_slot_{slot}_alloc_norm"] = _norm01(alloc, MAX_PREFILL_ACTION_ALLOC)
        row[f"a_prefill_slot_{slot}_alloc_frac_of_remaining"] = _norm01(alloc, max(1, int(remaining)))
        row[f"a_prefill_slot_{slot}_evicted"] = 1.0 if rid >= 0 and int(rid) in evicted_ids else 0.0
    return {name: float(row.get(name, 0.0)) for name in ACTION_FEATURE_NAMES}


def _feature_repr(feature_dict: dict[str, float]) -> str:
    return json.dumps({k: round(float(feature_dict[k]), 10) for k in ACTION_FEATURE_NAMES}, sort_keys=True, separators=(",", ":"))


def _build_bundle(model_path: str, output_dir: str) -> Any:
    from dataclasses import replace
    import joblib
    from vidur.Game_Version3.Model_Tester.config import DEFAULT_MODEL_TESTER_CONFIG
    try:
        from vidur.bellman_v4_adv_2000k_multiprocess import runner as tester_runner
    except Exception:
        from vidur.Game_Version3.Model_Tester import runner as tester_runner

    model_path_obj = Path(model_path).expanduser()
    harness_model_path = model_path_obj
    model = joblib.load(model_path_obj)
    if not callable(getattr(model, "infer_from_inputs", None)):
        from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper
        wrapped = V4AdvHGBWrapper(model, feature_dim=226, model_tag="controller_prior_feature_hgb")
        harness_dir = Path(output_dir).expanduser() / "_feature_harness_model"
        harness_dir.mkdir(parents=True, exist_ok=True)
        harness_model_path = harness_dir / "v4_adv_hgb_wrapper.joblib"
        joblib.dump(wrapped, harness_model_path, compress=3)

    cfg = replace(DEFAULT_MODEL_TESTER_CONFIG, model_kind="classical_joblib", model_checkpoint_path=str(harness_model_path), output_dir=str(Path(output_dir) / "_feature_harness"), write_arena_game_logs=False, write_model_action_detail_logs=False, environment_lang="native", use_virtual_env=True)
    return tester_runner._build_bundle(cfg)


def _state_from_record(env: Any, record: dict[str, Any]) -> Any:
    clone_fn = getattr(env, "clone_state_from_snapshot", None)
    if callable(clone_fn):
        return clone_fn(record["simulator_snapshot"], record["stats"])
    state = env.initial_state()
    state.simulator.restore_state(record["simulator_snapshot"])
    state.stats = record["stats"].clone()
    return state


def _load_records(dataset_dir: str, *, state_ids: list[int] | None, max_states: int | None) -> list[tuple[int, dict[str, Any]]]:
    from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import load_samples
    out: list[tuple[int, dict[str, Any]]] = []
    if state_ids is not None:
        wanted = {int(x) for x in state_ids}
        max_id = max(wanted) if wanted else -1
        for sid, record in load_samples(dataset_dir, root_player_filter="controller", parent_state_id_start=0, parent_state_id_end=max_id + 1):
            if int(sid) in wanted:
                out.append((int(sid), record))
                if len(out) == len(wanted):
                    break
        missing = wanted.difference(int(sid) for sid, _ in out)
        if missing:
            raise RuntimeError(f"missing requested state ids: {sorted(missing)[:20]}")
        return sorted(out, key=lambda x: int(x[0]))
    for sid, record in load_samples(dataset_dir, root_player_filter="controller"):
        out.append((int(sid), record))
        if max_states is not None and len(out) >= int(max_states):
            break
    return out



def _iter_records(dataset_dir: str, *, state_ids: list[int] | None, max_states: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield records without materializing the full dataset for large runs."""
    from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import load_samples

    if state_ids is not None:
        yield from _load_records(dataset_dir, state_ids=state_ids, max_states=max_states)
        return

    count = 0
    for sid, record in load_samples(dataset_dir, root_player_filter="controller"):
        yield int(sid), record
        count += 1
        if max_states is not None and count >= int(max_states):
            break


def build_features_for_records(*, worker_id: str, host: str, dataset_dir: str, output_dir: str, model_path: str, state_ids: list[int] | None = None, max_states: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from vidur.bellman_v4_adv_2000k_multiprocess import runner as tester_runner
    except Exception:
        from vidur.Game_Version3.Model_Tester import runner as tester_runner
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle(model_path, str(output))
    feature_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    records = _load_records(dataset_dir, state_ids=state_ids, max_states=max_states)
    for state_id, record in records:
        before = len(feature_rows)
        root_player = str(record.get("root_player", "controller"))
        root_depth = int(record.get("root_depth", 0) or 0)
        state = _state_from_record(bundle.env, record)
        player, expanded = tester_runner._align_player_to_valid_actions(bundle=bundle, state=state, player=root_player, pending_adv_pre_ctrl_snapshot=record.get("pre_controller_snapshot"), pending_adv_pre_ctrl_stats=record.get("pre_controller_stats"))
        valid_indices: list[int] = []
        canonical_indices: list[int] = []
        if str(player) == "controller":
            valid_indices = [int(x) for x in list(expanded.valid_indices)]
            for vidx in valid_indices:
                action = expanded.actions_by_index[int(vidx)]
                if action is not None:
                    _derive_evicted_ids(bundle, expanded.search_state, action)
            canonical_indices = _effect_canonical_indices(list(expanded.actions_by_index), valid_indices)
            for canon_idx in canonical_indices:
                action = expanded.actions_by_index[int(canon_idx)]
                if action is None:
                    continue
                fd = build_action_feature_dict(bundle, expanded.search_state, action)
                feature_rows.append({
                    "worker": worker_id,
                    "host": host,
                    "state_id": int(state_id),
                    "root_id": int(record.get("root_id", state_id) or state_id),
                    "player": str(player),
                    "root_player": root_player,
                    "root_depth": root_depth,
                    "history_hops": int(record.get("history_hops", -1) or -1),
                    "canon_action_index": int(canon_idx),
                    "action_repr": repr(action),
                    "action_feature_repr": _feature_repr(fd),
                })
        summary_rows.append({
            "worker": worker_id,
            "host": host,
            "state_id": int(state_id),
            "root_id": int(record.get("root_id", state_id) or state_id),
            "player": str(player),
            "root_player": root_player,
            "valid_action_count": int(len(valid_indices)),
            "canonical_action_count": int(len(canonical_indices)),
            "feature_action_count": int(len(feature_rows) - before),
        })
        try:
            bundle.mcts.clear_search_state(drop_scratch=True)
        except Exception:
            pass
        gc.collect()
    return feature_rows, summary_rows


def write_features_for_records(*, worker_id: str, host: str, dataset_dir: str, output_dir: str, model_path: str, state_ids: list[int] | None = None, max_states: int | None = None, progress_every: int = 10_000) -> dict[str, Any]:
    """Build and stream action features to disk for full-dataset runs."""
    try:
        from vidur.bellman_v4_adv_2000k_multiprocess import runner as tester_runner
    except Exception:
        from vidur.Game_Version3.Model_Tester import runner as tester_runner

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle(model_path, str(out))
    feature_path = out / "target_smoke_feature.csv"
    summary_path = out / "target_canon_feature_full.csv"
    num_states = 0
    num_feature_rows = 0
    mismatch_count = 0
    first_mismatch: dict[str, Any] | None = None
    t0 = time.time()

    with feature_path.open("w", encoding="utf-8", newline="") as ff, summary_path.open("w", encoding="utf-8", newline="") as sf:
        feature_writer = csv.DictWriter(ff, fieldnames=FEATURE_ROW_FIELDS, extrasaction="ignore")
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_ROW_FIELDS, extrasaction="ignore")
        feature_writer.writeheader()
        summary_writer.writeheader()

        for state_id, record in _iter_records(dataset_dir, state_ids=state_ids, max_states=max_states):
            before = num_feature_rows
            root_player = str(record.get("root_player", "controller"))
            root_depth = int(record.get("root_depth", 0) or 0)
            state = _state_from_record(bundle.env, record)
            player, expanded = tester_runner._align_player_to_valid_actions(
                bundle=bundle,
                state=state,
                player=root_player,
                pending_adv_pre_ctrl_snapshot=record.get("pre_controller_snapshot"),
                pending_adv_pre_ctrl_stats=record.get("pre_controller_stats"),
            )
            valid_indices: list[int] = []
            canonical_indices: list[int] = []
            if str(player) == "controller":
                valid_indices = [int(x) for x in list(expanded.valid_indices)]
                for vidx in valid_indices:
                    action = expanded.actions_by_index[int(vidx)]
                    if action is not None:
                        _derive_evicted_ids(bundle, expanded.search_state, action)
                canonical_indices = _effect_canonical_indices(list(expanded.actions_by_index), valid_indices)
                for canon_idx in canonical_indices:
                    action = expanded.actions_by_index[int(canon_idx)]
                    if action is None:
                        continue
                    fd = build_action_feature_dict(bundle, expanded.search_state, action)
                    feature_writer.writerow({
                        "worker": worker_id,
                        "host": host,
                        "state_id": int(state_id),
                        "root_id": int(record.get("root_id", state_id) or state_id),
                        "player": str(player),
                        "root_player": root_player,
                        "root_depth": root_depth,
                        "history_hops": int(record.get("history_hops", -1) or -1),
                        "canon_action_index": int(canon_idx),
                        "action_repr": repr(action),
                        "action_feature_repr": _feature_repr(fd),
                    })
                    num_feature_rows += 1

            summary_row = {
                "worker": worker_id,
                "host": host,
                "state_id": int(state_id),
                "root_id": int(record.get("root_id", state_id) or state_id),
                "player": str(player),
                "root_player": root_player,
                "valid_action_count": int(len(valid_indices)),
                "canonical_action_count": int(len(canonical_indices)),
                "feature_action_count": int(num_feature_rows - before),
            }
            summary_writer.writerow(summary_row)
            num_states += 1
            if int(summary_row["canonical_action_count"]) != int(summary_row["feature_action_count"]):
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = dict(summary_row)

            try:
                bundle.mcts.clear_search_state(drop_scratch=True)
            except Exception:
                pass
            if progress_every > 0 and num_states % int(progress_every) == 0:
                ff.flush()
                sf.flush()
                print(json.dumps({"states": num_states, "feature_rows": num_feature_rows, "elapsed_s": round(time.time() - t0, 3)}, sort_keys=True), flush=True)
                gc.collect()

    if mismatch_count:
        raise RuntimeError(f"feature_action_count mismatch in {mismatch_count} states; first={first_mismatch}")

    return {
        "worker": str(worker_id),
        "host": str(host),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out),
        "num_states": int(num_states),
        "num_feature_rows": int(num_feature_rows),
        "feature_dim": len(ACTION_FEATURE_NAMES),
        "feature_names": ACTION_FEATURE_NAMES,
        "feature_version": ACTION_FEATURE_VERSION,
        "elapsed_s": float(time.time() - t0),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def cmd_run_local(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).expanduser()
    manifest = write_features_for_records(worker_id=str(args.worker_id), host=str(args.host), dataset_dir=str(args.dataset_dir), output_dir=str(out), model_path=str(args.model_path), state_ids=_parse_state_ids(args.state_ids), max_states=args.max_states)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": manifest["num_feature_rows"], "states": manifest["num_states"], "output_dir": str(out)}, sort_keys=True), flush=True)


def _combine_worker_outputs(local_dir: Path, worker_dirs: list[tuple[WorkerSpec, Path]]) -> dict[str, int]:
    feature_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _worker, path in worker_dirs:
        fpath = path / "target_smoke_feature.csv"
        spath = path / "target_canon_feature_full.csv"
        if fpath.exists():
            with fpath.open("r", encoding="utf-8", newline="") as f:
                feature_rows.extend(dict(row) for row in csv.DictReader(f))
        if spath.exists():
            with spath.open("r", encoding="utf-8", newline="") as f:
                summary_rows.extend(dict(row) for row in csv.DictReader(f))
    feature_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"]), int(r["canon_action_index"])))
    summary_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"])))
    _write_csv(local_dir / "target_smoke_feature.csv", FEATURE_ROW_FIELDS, feature_rows)
    _write_csv(local_dir / "target_canon_feature_full.csv", SUMMARY_ROW_FIELDS, summary_rows)
    summary = {"feature_rows": len(feature_rows), "summary_rows": len(summary_rows)}
    (local_dir / "target_feature_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def cmd_smoke(args: argparse.Namespace) -> None:
    local_dir = Path(args.local_smoke_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    remote_model_path = _remote_model_path(args)
    local_model = Path(args.local_model_path).expanduser()
    if not local_model.exists():
        raise FileNotFoundError(f"local model not found: {local_model}")
    state_ids_by_worker = _load_smoke_state_ids_by_worker(args.smoke_state_ids_csv)
    workers = _parse_workers(args.workers)
    for worker in workers:
        remote_script = f"{args.remote_repo.rstrip('/')}/{script_path.relative_to(_repo_root())}"
        _ssh(worker.host, f"mkdir -p {shlex.quote(str(Path(remote_script).parent))} {shlex.quote(str(Path(remote_model_path).parent))}")
        _rsync(str(script_path), f"{worker.host}:{remote_script}")
        _rsync(str(local_model), f"{worker.host}:{remote_model_path}")
    procs: list[tuple[WorkerSpec, subprocess.Popen[str], str]] = []
    target_base = _target_base(args)
    for worker in workers:
        ids = state_ids_by_worker.get(worker.worker_id)
        if not ids:
            raise RuntimeError(f"no smoke state ids for {worker.worker_id} in {args.smoke_state_ids_csv}")
        dataset_dir = _worker_root_dir(worker, remote_repo=args.remote_repo, remote_output_base=args.remote_output_base, experiment_name=args.experiment_name)
        out_dir = f"{target_base}/smoke_{worker.worker_id}"
        cmd = (f"cd {shlex.quote(args.remote_repo)} && {shlex.quote(args.remote_repo.rstrip('/') + '/.venv/bin/python3')} -m vidur.bellman_v4_adv_2000k_multiprocess.multi_server_hgb_controller_prior_feature_bulding run-local --worker-id {shlex.quote(worker.worker_id)} --host {shlex.quote(worker.host)} --dataset-dir {shlex.quote(dataset_dir)} --output-dir {shlex.quote(out_dir)} --model-path {shlex.quote(remote_model_path)} --state-ids {shlex.quote(','.join(str(x) for x in ids))}")
        print(f"[launch] {worker.host}: {cmd}", flush=True)
        procs.append((worker, subprocess.Popen(["ssh", worker.host, cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT), out_dir))
    failed: list[str] = []
    for worker, proc, _out_dir in procs:
        stdout, _ = proc.communicate()
        (local_dir / f"{worker.worker_id}_feature_remote_stdout.log").write_text(stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            failed.append(f"{worker.worker_id}:{proc.returncode}")
            print(stdout or "", flush=True)
    if failed:
        raise RuntimeError(f"remote smoke feature generation failed: {failed}")
    pulled: list[tuple[WorkerSpec, Path]] = []
    for worker, _proc, out_dir in procs:
        worker_local = local_dir / f"feature_{worker.worker_id}"
        worker_local.mkdir(parents=True, exist_ok=True)
        _rsync(f"{worker.host}:{out_dir.rstrip('/')}/", f"{worker_local}/")
        pulled.append((worker, worker_local))
    print(json.dumps(_combine_worker_outputs(local_dir, pulled), indent=2, sort_keys=True), flush=True)


def cmd_status(args: argparse.Namespace) -> None:
    for worker in _parse_workers(args.workers):
        out_dir = f"{_target_base(args)}/smoke_{worker.worker_id}"
        res = _ssh(worker.host, f"cat {shlex.quote(out_dir + '/manifest.json')} 2>/dev/null || echo missing", check=False)
        print(f"--- {worker.worker_id} {worker.host} ---")
        print(res.stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-local")
    run.add_argument("--worker-id", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--dataset-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--state-ids", default=None)
    run.add_argument("--max-states", type=int, default=None)
    run.set_defaults(func=cmd_run_local)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    smoke.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    smoke.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    smoke.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    smoke.add_argument("--local-model-path", default=DEFAULT_LOCAL_MODEL_PATH)
    smoke.add_argument("--remote-model-path", default=DEFAULT_REMOTE_MODEL_PATH)
    smoke.add_argument("--local-smoke-dir", default=DEFAULT_LOCAL_SMOKE_DIR)
    smoke.add_argument("--smoke-state-ids-csv", default=DEFAULT_SMOKE_STATE_IDS_CSV)
    smoke.add_argument("--workers", default="worker1,worker2,worker3,worker4")
    smoke.set_defaults(func=cmd_smoke)
    status = sub.add_parser("status")
    status.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    status.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    status.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    status.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    status.add_argument("--remote-model-path", default=DEFAULT_REMOTE_MODEL_PATH)
    status.add_argument("--workers", default="worker1,worker2,worker3,worker4")
    status.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
