#!/usr/bin/env python3
"""Train a GV3 controller policy/prior HGB head from native MCTS visit targets.

Expected pipeline after MCTS targets and action features are generated:

1. postprocess-targets-local on each worker:
   raw canonical-action visits -> visit probabilities and centered-logit targets.
2. sync-inputs on the coordinator:
   copy each worker's policy targets and action features onto the training host.
3. assemble-local on the training host, default worker 2:
   join 226D state features + 43D action features + centered-logit target.
4. prepare-split-local on the training host:
   select 100k random roots for eval, rest for train.
5. train-local on the training host:
   fit HGB action scorer and report row-level plus root-group policy metrics.

The model is trained as a scorer f(state, action) with target:

    centered_logit_i = log(visits_i + alpha) - mean_j log(visits_j + alpha)

where the mean is over canonical actions from the same controller root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import shlex
import tempfile
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_TRAIN_HOST = "bellman-classical-worker-2"
DEFAULT_TARGET_MODEL_LABEL = "worker1_200k_alpha2__hgb_sq_63leaf_1050iter_a2__v47__"
DEFAULT_MCTS_ITERATIONS = 10_000
DEFAULT_TARGET_DIR_SUFFIX = "effectcanon_full"
DEFAULT_ACTION_FEATURE_VERSION = "controller_action_features_effectcanon_v1"
DEFAULT_POLICY_ALPHA = 1.0
DEFAULT_SEED = 20260617
DEFAULT_EVAL_ROOTS = 100_000
DEFAULT_THREADS = 48
DEFAULT_PREDICT_CHUNK = 524_288
DEFAULT_FIT_SUBSAMPLE_ROWS = 0

DEFAULT_STATE_FEATURE_ASSEMBLY_DIR = "{base}/{experiment_name}/assembled_parent_child_features_adv"
DEFAULT_POLICY_WORK_DIR = "{base}/{experiment_name}/controller_prior_training_effectcanon"
DEFAULT_TARGET_BASE = (
    "{base}/{experiment_name}/mcts_value_function_visit_targets/"
    "{model_label}_iter{mcts_iterations}_{target_dir_suffix}"
)
DEFAULT_ACTION_FEATURE_BASE = (
    "{base}/{experiment_name}/controller_prior_action_features/{action_feature_version}"
)
DEFAULT_INPUT_DIR = "{policy_work_dir}/worker_inputs"
DEFAULT_ASSEMBLED_DATASET_DIR = "{policy_work_dir}/assembled_state_action_dataset"
DEFAULT_OUTPUT_DIR = "{policy_work_dir}/HGB_policy_head"

STATE_FEATURE_DIM = 226
ACTION_FEATURE_DIM = 43
COMBINED_FEATURE_DIM = STATE_FEATURE_DIM + ACTION_FEATURE_DIM


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    server_index: int
    hops_min: int
    hops_max: int

    @property
    def job_id(self) -> str:
        return f"server_{self.server_index:02d}"

    @property
    def root_dir_name(self) -> str:
        return f"{self.job_id}_{self.host}_hops_{self.hops_min}_{self.hops_max}"


WORKERS: tuple[WorkerSpec, ...] = (
    WorkerSpec("worker1", "bellman-classical-worker-1", 1, 151, 300),
    WorkerSpec("worker2", "bellman-classical-worker-2", 2, 301, 450),
    WorkerSpec("worker3", "bellman-classical-worker-3", 3, 451, 600),
    WorkerSpec("worker4", "bellman-classical-worker-4", 4, 601, 750),
)


@dataclass(frozen=True)
class HGBPolicyConfig:
    name: str
    max_leaf_nodes: int
    max_iter: int
    learning_rate: float = 0.05
    l2_regularization: float = 1.0
    loss: str = "squared_error"

    @property
    def approx_params(self) -> int:
        return int(3 * int(self.max_leaf_nodes) * int(self.max_iter))


POLICY_CONFIGS: tuple[HGBPolicyConfig, ...] = (
    HGBPolicyConfig("hgb_policy_47leaf_1410iter", 47, 1410),
    HGBPolicyConfig("hgb_policy_63leaf_1050iter", 63, 1050),
    HGBPolicyConfig("hgb_policy_95leaf_0700iter", 95, 700),
)


POLICY_TARGET_FIELDS = [
    "worker",
    "host",
    "state_id",
    "root_id",
    "player",
    "root_player",
    "root_depth",
    "history_hops",
    "canon_action_index",
    "visit_count",
    "mcts_visit_prob",
    "mcts_visit_prob_smoothed",
    "mcts_centered_logit",
    "is_best_visit_action",
    "total_child_visits",
    "mcts_iterations",
    "valid_action_count",
    "canonical_action_count",
    "num_actions_in_root",
    "target_smoothing_alpha",
    "action_repr",
]

POLICY_TARGET_SUMMARY_FIELDS = [
    "worker",
    "host",
    "state_id",
    "root_id",
    "player",
    "root_player",
    "canonical_action_count",
    "total_child_visits",
    "best_canon_action_index",
    "best_visit_count",
    "target_entropy",
]

SUMMARY_FIELDS = [
    "split",
    "config_name",
    "metric_source",
    "n_rows",
    "n_roots",
    "mse",
    "rmse",
    "mae",
    "p50_abs",
    "p90_abs",
    "p95_abs",
    "p99_abs",
    "max_abs",
    "top1_match",
    "top3_contains_best",
    "cross_entropy",
    "kl_divergence",
    "fit_elapsed_s",
    "predict_elapsed_s",
    "n_params_estimate",
]


FALLBACK_ACTION_FEATURE_NAMES = [
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
for _slot in range(7):
    FALLBACK_ACTION_FEATURE_NAMES.extend(
        [
            f"a_prefill_slot_{_slot}_selected",
            f"a_prefill_slot_{_slot}_alloc_norm",
            f"a_prefill_slot_{_slot}_alloc_frac_of_remaining",
            f"a_prefill_slot_{_slot}_evicted",
        ]
    )
assert len(FALLBACK_ACTION_FEATURE_NAMES) == ACTION_FEATURE_DIM


def _action_feature_names() -> list[str]:
    try:
        from vidur.bellman_v4_adv_2000k_multiprocess.multi_server_hgb_controller_prior_feature_bulding import ACTION_FEATURE_NAMES

        names = list(ACTION_FEATURE_NAMES)
    except Exception:
        names = list(FALLBACK_ACTION_FEATURE_NAMES)
    if len(names) != ACTION_FEATURE_DIM:
        raise RuntimeError(f"action feature dim={len(names)} expected={ACTION_FEATURE_DIM}")
    return names


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vidur" / "Game_Version3").is_dir():
            return parent
    return p.parents[3]


def _run(cmd: list[str], *, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _ssh(host: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, cmd], check=check)


def _rsync(src: str, dst: str) -> subprocess.CompletedProcess[str]:
    return _run(["rsync", "-az", src, dst])


def _base_path(remote_repo: str, remote_output_base: str) -> str:
    return f"{remote_repo.rstrip('/')}/{remote_output_base.strip('/')}"


def _format_common(template: str, args: argparse.Namespace) -> str:
    base = _base_path(args.remote_repo, args.remote_output_base)
    policy_work_dir = str(args.policy_work_dir).format(
        base=base,
        experiment_name=args.experiment_name,
        remote_repo=args.remote_repo.rstrip("/"),
    )
    return str(template).format(
        base=base,
        experiment_name=args.experiment_name,
        remote_repo=args.remote_repo.rstrip("/"),
        model_label=args.model_label,
        mcts_iterations=int(args.mcts_iterations),
        target_dir_suffix=str(args.target_dir_suffix).strip(),
        action_feature_version=args.action_feature_version,
        policy_work_dir=policy_work_dir,
    )


def _target_base(args: argparse.Namespace) -> str:
    return _format_common(args.target_base, args)


def _action_feature_base(args: argparse.Namespace) -> str:
    return _format_common(args.action_feature_base, args)


def _policy_work_dir(args: argparse.Namespace) -> str:
    return _format_common(args.policy_work_dir, args)


def _input_dir(args: argparse.Namespace) -> str:
    return _format_common(args.input_dir, args)


def _state_feature_assembly_dir(args: argparse.Namespace) -> str:
    return _format_common(args.state_feature_assembly_dir, args)


def _assembled_dataset_dir(args: argparse.Namespace) -> str:
    return _format_common(args.assembled_dataset_dir, args)


def _output_dir(args: argparse.Namespace) -> str:
    return _format_common(args.output_dir, args)


def _policy_alpha_label(alpha: float) -> str:
    s = (f"{float(alpha):.6g}").replace(".", "p").replace("-", "m")
    return f"alpha{s}"


def _worker_target_dir(args: argparse.Namespace, worker: WorkerSpec) -> str:
    return f"{_target_base(args).rstrip('/')}/{worker.worker_id}"


def _worker_policy_target_dir(args: argparse.Namespace, worker: WorkerSpec) -> str:
    return f"{_worker_target_dir(args, worker).rstrip('/')}/policy_targets_{_policy_alpha_label(args.policy_alpha)}"


def _worker_action_feature_dir(args: argparse.Namespace, worker: WorkerSpec) -> str:
    return f"{_action_feature_base(args).rstrip('/')}/{worker.worker_id}"


def _parse_workers(raw: str | None) -> list[WorkerSpec]:
    if not raw:
        return list(WORKERS)
    wanted = {x.strip() for x in str(raw).split(",") if x.strip()}
    out = [w for w in WORKERS if w.worker_id in wanted or w.host in wanted or w.job_id in wanted]
    if not out:
        raise SystemExit(f"no workers selected from {raw!r}")
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _find_first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {names} exist under {root}")


def _target_csv_files(input_dir: Path) -> list[Path]:
    direct = input_dir / "targets.csv"
    if direct.exists():
        return [direct]
    partials = sorted((input_dir / "partials").glob("targets_batch_*.csv"))
    if partials:
        return partials
    raise FileNotFoundError(f"no targets.csv or partials/targets_batch_*.csv in {input_dir}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _root_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("worker", "")),
        _safe_int(row.get("state_id")),
        _safe_int(row.get("root_id", row.get("state_id"))),
        str(row.get("player", "controller")),
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row.get("worker", "")),
        _safe_int(row.get("state_id")),
        _safe_int(row.get("root_id", row.get("state_id"))),
        _safe_int(row.get("canon_action_index")),
    )


def _load_raw_targets(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _target_csv_files(input_dir):
        for row in _read_csv_rows(path):
            if str(row.get("player", "controller")) != "controller":
                continue
            rows.append(dict(row))
    rows.sort(key=_row_sort_key)
    return rows


def build_policy_targets(
    raw_rows: list[dict[str, Any]],
    *,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for finite centered logits")

    out_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    i = 0
    while i < len(raw_rows):
        key = _root_key(raw_rows[i])
        group: list[dict[str, Any]] = []
        while i < len(raw_rows) and _root_key(raw_rows[i]) == key:
            group.append(raw_rows[i])
            i += 1

        by_action: dict[int, dict[str, Any]] = {}
        for row in group:
            cidx = _safe_int(row.get("canon_action_index"))
            if cidx not in by_action:
                merged = dict(row)
                merged["visit_count"] = _safe_int(row.get("visit_count"))
                by_action[cidx] = merged
            else:
                by_action[cidx]["visit_count"] = _safe_int(by_action[cidx].get("visit_count")) + _safe_int(row.get("visit_count"))

        actions = [by_action[k] for k in sorted(by_action)]
        visits = [_safe_int(r.get("visit_count")) for r in actions]
        total_visits = int(sum(visits))
        n_actions = int(len(actions))
        if n_actions == 0:
            continue
        if total_visits > 0:
            probs = [float(v) / float(total_visits) for v in visits]
        else:
            probs = [1.0 / float(n_actions)] * n_actions
        smooth_den = float(sum(v + alpha for v in visits))
        smooth_probs = [float(v + alpha) / smooth_den for v in visits]
        logs = [math.log(float(v) + alpha) for v in visits]
        mean_log = float(sum(logs) / len(logs))
        centered = [float(x - mean_log) for x in logs]
        max_visit = max(visits) if visits else 0
        best_indices = [idx for idx, v in enumerate(visits) if v == max_visit]
        best_canon = _safe_int(actions[best_indices[0]].get("canon_action_index")) if best_indices else -1
        entropy = -sum(p * math.log(max(p, 1e-30)) for p in probs)

        for row, prob, smooth_prob, logit, visit in zip(actions, probs, smooth_probs, centered, visits):
            next_row = dict(row)
            next_row["visit_count"] = int(visit)
            next_row["mcts_visit_prob"] = float(prob)
            next_row["mcts_visit_prob_smoothed"] = float(smooth_prob)
            next_row["mcts_centered_logit"] = float(logit)
            next_row["is_best_visit_action"] = int(visit == max_visit)
            next_row["total_child_visits"] = int(total_visits)
            next_row["canonical_action_count"] = int(n_actions)
            next_row["num_actions_in_root"] = int(n_actions)
            next_row["target_smoothing_alpha"] = float(alpha)
            out_rows.append(next_row)

        first = actions[0]
        summary_rows.append(
            {
                "worker": first.get("worker", ""),
                "host": first.get("host", ""),
                "state_id": _safe_int(first.get("state_id")),
                "root_id": _safe_int(first.get("root_id", first.get("state_id"))),
                "player": first.get("player", "controller"),
                "root_player": first.get("root_player", "controller"),
                "canonical_action_count": int(n_actions),
                "total_child_visits": int(total_visits),
                "best_canon_action_index": int(best_canon),
                "best_visit_count": int(max_visit),
                "target_entropy": float(entropy),
            }
        )
    return out_rows, summary_rows


def command_postprocess_targets_local(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_target_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    raw_rows = _load_raw_targets(input_dir)
    target_rows, summary_rows = build_policy_targets(raw_rows, alpha=float(args.policy_alpha))
    _write_csv(output_dir / "policy_targets.csv", POLICY_TARGET_FIELDS, target_rows)
    _write_csv(output_dir / "policy_target_summary.csv", POLICY_TARGET_SUMMARY_FIELDS, summary_rows)
    manifest = {
        "input_target_dir": str(input_dir),
        "output_dir": str(output_dir),
        "policy_alpha": float(args.policy_alpha),
        "num_raw_rows": int(len(raw_rows)),
        "num_policy_rows": int(len(target_rows)),
        "num_roots": int(len(summary_rows)),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy_target_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


@dataclass(frozen=True)
class ParentShard:
    worker_id: str
    job_id: str
    host: str
    local_dir: Path
    rows: int
    dim: int


def _read_state_feature_shards(assembly_dir: Path, workers: list[WorkerSpec]) -> dict[str, ParentShard]:
    manifest_path = assembly_dir / "assembly_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"state feature assembly manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [e for e in manifest.get("parent", []) if e.get("valid", True)]
    by_job = {str(e.get("job_id", "")): e for e in entries}
    out: dict[str, ParentShard] = {}
    for worker in workers:
        entry = by_job.get(worker.job_id)
        if entry is None:
            raise RuntimeError(f"missing parent feature shard for {worker.worker_id} job_id={worker.job_id}")
        local_dir = Path(str(entry["local_dir"]))
        meta = json.loads((local_dir / "parent_features.meta.json").read_text(encoding="utf-8"))
        rows = int(meta["num_records"])
        dim = int(meta["feature_dim"])
        if dim != STATE_FEATURE_DIM:
            raise RuntimeError(f"{local_dir} state feature dim={dim}, expected={STATE_FEATURE_DIM}")
        if not (local_dir / "parent_features.npy").exists():
            raise FileNotFoundError(f"missing parent_features.npy in {local_dir}")
        out[worker.worker_id] = ParentShard(
            worker_id=worker.worker_id,
            job_id=worker.job_id,
            host=str(entry.get("source_host", worker.host)),
            local_dir=local_dir,
            rows=rows,
            dim=dim,
        )
    return out


def _input_worker_dir(input_dir: Path, worker: WorkerSpec) -> Path:
    return input_dir / worker.worker_id


def _policy_target_file(input_dir: Path, worker: WorkerSpec) -> Path:
    return _find_first_existing(
        _input_worker_dir(input_dir, worker) / "policy_targets",
        ("policy_targets.csv", "targets_policy.csv"),
    )


def _action_feature_file(input_dir: Path, worker: WorkerSpec) -> Path:
    return _find_first_existing(
        _input_worker_dir(input_dir, worker) / "action_features",
        ("target_action_features.csv", "action_features.csv", "target_smoke_feature.csv"),
    )


def _csv_key(row: dict[str, Any]) -> tuple[int, int]:
    return (_safe_int(row.get("state_id")), _safe_int(row.get("canon_action_index")))


def _iter_sorted_csv(path: Path) -> Iterator[dict[str, str]]:
    yield from _read_csv_rows(path)


def _joined_worker_rows(*, action_path: Path, target_path: Path) -> Iterator[tuple[dict[str, str], dict[str, str]]]:
    target_iter = iter(_iter_sorted_csv(target_path))
    try:
        target = next(target_iter)
    except StopIteration:
        return
    target_key = _csv_key(target)
    for feat in _iter_sorted_csv(action_path):
        fkey = _csv_key(feat)
        while target_key < fkey:
            try:
                target = next(target_iter)
                target_key = _csv_key(target)
            except StopIteration:
                return
        if target_key == fkey:
            yield feat, target


def _parse_action_features(raw: str, names: list[str]) -> list[float]:
    obj = json.loads(raw)
    if isinstance(obj, list):
        values = [float(x) for x in obj]
    elif isinstance(obj, dict):
        values = [float(obj.get(name, 0.0)) for name in names]
    else:
        raise TypeError(f"unsupported action_feature_repr type={type(obj).__name__}")
    if len(values) != ACTION_FEATURE_DIM:
        raise RuntimeError(f"action feature vector len={len(values)}, expected={ACTION_FEATURE_DIM}")
    return values


def _count_joined_rows(
    *,
    workers: list[WorkerSpec],
    input_dir: Path,
    parent_shards: dict[str, ParentShard],
    max_rows: int | None,
) -> tuple[int, int, dict[str, int]]:
    total_rows = 0
    total_roots = 0
    per_worker: dict[str, int] = {}
    for worker in workers:
        action_path = _action_feature_file(input_dir, worker)
        target_path = _policy_target_file(input_dir, worker)
        last_root: tuple[str, int, int] | None = None
        count = 0
        for feat, target in _joined_worker_rows(action_path=action_path, target_path=target_path):
            state_id = _safe_int(feat.get("state_id"))
            if state_id < 0 or state_id >= parent_shards[worker.worker_id].rows:
                continue
            root_key = (worker.worker_id, state_id, _safe_int(feat.get("root_id", state_id)))
            if root_key != last_root:
                total_roots += 1
                last_root = root_key
            count += 1
            total_rows += 1
            if max_rows is not None and total_rows >= int(max_rows):
                per_worker[worker.worker_id] = count
                return total_rows, total_roots, per_worker
        per_worker[worker.worker_id] = count
    return total_rows, total_roots, per_worker


def command_assemble_local(args: argparse.Namespace) -> None:
    import numpy as np

    workers = _parse_workers(args.workers)
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = _action_feature_names()
    parent_shards = _read_state_feature_shards(Path(args.state_feature_assembly_dir).expanduser(), workers)

    max_rows = None if int(args.max_rows) <= 0 else int(args.max_rows)
    row_count, root_count, per_worker_counts = _count_joined_rows(
        workers=workers,
        input_dir=input_dir,
        parent_shards=parent_shards,
        max_rows=max_rows,
    )
    if row_count <= 0:
        raise RuntimeError(f"no joined policy rows found under {input_dir}")

    print(f"[assemble] rows={row_count} roots={root_count} per_worker={per_worker_counts}", flush=True)
    X = np.lib.format.open_memmap(output_dir / "X_state_action.npy", mode="w+", dtype=np.float32, shape=(row_count, COMBINED_FEATURE_DIM))
    y = np.lib.format.open_memmap(output_dir / "y_centered_logit.npy", mode="w+", dtype=np.float32, shape=(row_count,))
    visit_prob = np.lib.format.open_memmap(output_dir / "mcts_visit_prob.npy", mode="w+", dtype=np.float32, shape=(row_count,))
    visit_count = np.lib.format.open_memmap(output_dir / "visit_count.npy", mode="w+", dtype=np.int32, shape=(row_count,))
    is_best = np.lib.format.open_memmap(output_dir / "is_best_visit_action.npy", mode="w+", dtype=np.int8, shape=(row_count,))
    root_group_id = np.lib.format.open_memmap(output_dir / "root_group_id.npy", mode="w+", dtype=np.int64, shape=(row_count,))
    state_id_arr = np.lib.format.open_memmap(output_dir / "state_id.npy", mode="w+", dtype=np.int64, shape=(row_count,))
    canon_action_index = np.lib.format.open_memmap(output_dir / "canon_action_index.npy", mode="w+", dtype=np.int32, shape=(row_count,))
    worker_code = np.lib.format.open_memmap(output_dir / "worker_code.npy", mode="w+", dtype=np.int16, shape=(row_count,))

    parent_features = {wid: np.load(shard.local_dir / "parent_features.npy", mmap_mode="r") for wid, shard in parent_shards.items()}
    worker_to_code = {worker.worker_id: i for i, worker in enumerate(workers)}
    root_offsets: list[int] = [0]
    root_rows: list[dict[str, Any]] = []
    idx = 0
    current_root: tuple[str, int, int] | None = None
    current_gid = -1

    for worker in workers:
        action_path = _action_feature_file(input_dir, worker)
        target_path = _policy_target_file(input_dir, worker)
        for feat, target in _joined_worker_rows(action_path=action_path, target_path=target_path):
            if idx >= row_count:
                break
            sid = _safe_int(feat.get("state_id"))
            if sid < 0 or sid >= parent_shards[worker.worker_id].rows:
                continue
            rid = _safe_int(feat.get("root_id", sid))
            root_key = (worker.worker_id, sid, rid)
            if root_key != current_root:
                if current_root is not None:
                    root_offsets.append(idx)
                current_gid += 1
                current_root = root_key
                root_rows.append(
                    {
                        "root_group_id": int(current_gid),
                        "worker": worker.worker_id,
                        "host": worker.host,
                        "worker_code": int(worker_to_code[worker.worker_id]),
                        "state_id": int(sid),
                        "root_id": int(rid),
                        "canonical_action_count": _safe_int(target.get("num_actions_in_root", target.get("canonical_action_count"))),
                    }
                )
            X[idx, :STATE_FEATURE_DIM] = parent_features[worker.worker_id][sid]
            X[idx, STATE_FEATURE_DIM:] = np.asarray(_parse_action_features(feat["action_feature_repr"], names), dtype=np.float32)
            y[idx] = _safe_float(target.get("mcts_centered_logit"))
            visit_prob[idx] = _safe_float(target.get("mcts_visit_prob"))
            visit_count[idx] = _safe_int(target.get("visit_count"))
            is_best[idx] = 1 if _safe_int(target.get("is_best_visit_action")) else 0
            root_group_id[idx] = int(current_gid)
            state_id_arr[idx] = int(sid)
            canon_action_index[idx] = _safe_int(feat.get("canon_action_index"))
            worker_code[idx] = int(worker_to_code[worker.worker_id])
            idx += 1
        if max_rows is not None and idx >= row_count:
            break

    if current_root is not None:
        root_offsets.append(idx)
    if idx != row_count:
        raise RuntimeError(f"assembled row count mismatch: wrote {idx}, expected {row_count}")

    np.save(output_dir / "root_offsets.npy", np.asarray(root_offsets, dtype=np.int64))
    _write_csv(
        output_dir / "root_meta.csv",
        ["root_group_id", "worker", "host", "worker_code", "state_id", "root_id", "canonical_action_count"],
        root_rows,
    )
    manifest = {
        "output_dir": str(output_dir),
        "input_dir": str(input_dir),
        "state_feature_assembly_dir": str(args.state_feature_assembly_dir),
        "num_rows": int(row_count),
        "num_roots": int(len(root_rows)),
        "state_feature_dim": STATE_FEATURE_DIM,
        "action_feature_dim": ACTION_FEATURE_DIM,
        "combined_feature_dim": COMBINED_FEATURE_DIM,
        "action_feature_names": names,
        "workers": [asdict(w) for w in workers],
        "worker_to_code": worker_to_code,
        "per_worker_rows": per_worker_counts,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "assembly_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def _root_rows_from_groups(root_offsets: Any, groups: Any) -> Any:
    import numpy as np

    pieces = [np.arange(int(root_offsets[g]), int(root_offsets[g + 1]), dtype=np.int64) for g in groups]
    if not pieces:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(pieces).astype(np.int64)


def command_prepare_split_local(args: argparse.Namespace) -> None:
    import numpy as np

    dataset_dir = Path(args.dataset_dir).expanduser()
    output_path = Path(args.output_path).expanduser() if args.output_path else dataset_dir / f"split_seed{int(args.seed)}_evalroots{int(args.eval_roots)}.npz"
    root_offsets = np.load(dataset_dir / "root_offsets.npy", mmap_mode="r")
    num_roots = int(root_offsets.shape[0] - 1)
    eval_roots = min(int(args.eval_roots), max(1, num_roots - 1))
    if output_path.exists() and not args.force:
        print(json.dumps({"status": "exists", "output_path": str(output_path)}, indent=2), flush=True)
        return
    rng = np.random.default_rng(int(args.seed))
    eval_groups = np.sort(rng.choice(num_roots, size=eval_roots, replace=False)).astype(np.int64)
    is_eval = np.zeros(num_roots, dtype=bool)
    is_eval[eval_groups] = True
    train_groups = np.nonzero(~is_eval)[0].astype(np.int64)
    eval_rows = _root_rows_from_groups(root_offsets, eval_groups)
    train_rows = _root_rows_from_groups(root_offsets, train_groups)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        seed=np.asarray([int(args.seed)], dtype=np.int64),
        num_roots=np.asarray([num_roots], dtype=np.int64),
        eval_root_groups=eval_groups,
        train_root_groups=train_groups,
        eval_rows=eval_rows,
        train_rows=train_rows,
    )
    summary = {
        "status": "created",
        "output_path": str(output_path),
        "seed": int(args.seed),
        "num_roots": int(num_roots),
        "eval_roots": int(eval_groups.size),
        "train_roots": int(train_groups.size),
        "eval_rows": int(eval_rows.size),
        "train_rows": int(train_rows.size),
    }
    (output_path.with_suffix(".json")).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _set_threads(n: int) -> None:
    n = max(1, int(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n)
    except Exception:
        pass


def _hgb_n_params(model: Any) -> int:
    import numpy as np

    total = 0
    for est in getattr(model, "_predictors", []) or []:
        for tree in est:
            try:
                total += int(np.sum(tree.nodes["is_leaf"])) * 3
            except Exception:
                total += int(tree.nodes.size) * 3
    return int(total)


def _row_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "n_rows": int(y_true.size),
        "mse": float(np.mean(err * err)) if y_true.size else 0.0,
        "rmse": float(math.sqrt(np.mean(err * err))) if y_true.size else 0.0,
        "mae": float(np.mean(abs_err)) if y_true.size else 0.0,
        "p50_abs": float(np.quantile(abs_err, 0.50)) if y_true.size else 0.0,
        "p90_abs": float(np.quantile(abs_err, 0.90)) if y_true.size else 0.0,
        "p95_abs": float(np.quantile(abs_err, 0.95)) if y_true.size else 0.0,
        "p99_abs": float(np.quantile(abs_err, 0.99)) if y_true.size else 0.0,
        "max_abs": float(abs_err.max()) if y_true.size else 0.0,
    }


def _softmax(scores: Any) -> Any:
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    z = scores - float(np.max(scores))
    ez = np.exp(z)
    denom = float(np.sum(ez))
    if denom <= 0.0 or not math.isfinite(denom):
        return np.full(scores.shape, 1.0 / float(scores.size), dtype=np.float64)
    return ez / denom


def _group_metrics(*, root_groups: Any, root_offsets: Any, y_pred: Any, visit_prob: Any, is_best: Any) -> dict[str, float]:
    import numpy as np

    top1 = 0
    top3 = 0
    ce_sum = 0.0
    kl_sum = 0.0
    roots = 0
    eps = 1e-30
    for g in np.asarray(root_groups, dtype=np.int64):
        s = int(root_offsets[g])
        e = int(root_offsets[g + 1])
        if e <= s:
            continue
        scores = np.asarray(y_pred[s:e], dtype=np.float64)
        p = np.asarray(visit_prob[s:e], dtype=np.float64)
        if p.sum() <= 0.0 or not np.isfinite(p).all():
            p = np.full(e - s, 1.0 / float(e - s), dtype=np.float64)
        else:
            p = p / p.sum()
        q = _softmax(scores)
        best_mask = np.asarray(is_best[s:e], dtype=np.int8) > 0
        if not best_mask.any():
            best_mask[np.argmax(p)] = True
        pred_order = np.argsort(-scores, kind="stable")
        if bool(best_mask[pred_order[0]]):
            top1 += 1
        if bool(best_mask[pred_order[: min(3, pred_order.size)]].any()):
            top3 += 1
        ce = -float(np.sum(p * np.log(np.maximum(q, eps))))
        kl = float(np.sum(p * (np.log(np.maximum(p, eps)) - np.log(np.maximum(q, eps)))))
        ce_sum += ce
        kl_sum += kl
        roots += 1
    denom = float(max(1, roots))
    return {
        "n_roots": int(roots),
        "top1_match": float(top1 / denom),
        "top3_contains_best": float(top3 / denom),
        "cross_entropy": float(ce_sum / denom),
        "kl_divergence": float(kl_sum / denom),
    }


def _predict_chunked(model: Any, X: Any, *, chunk: int) -> Any:
    import numpy as np

    n = int(X.shape[0])
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, int(chunk)):
        e = min(n, s + int(chunk))
        out[s:e] = model.predict(X[s:e]).astype(np.float32)
    return out


def _parse_policy_configs(raw: str) -> list[HGBPolicyConfig]:
    if raw.strip().lower() == "all":
        return list(POLICY_CONFIGS)
    wanted = {x.strip() for x in raw.split(",") if x.strip()}
    by_name = {c.name: c for c in POLICY_CONFIGS}
    missing = wanted - set(by_name)
    if missing:
        raise SystemExit(f"unknown policy config(s) {sorted(missing)}; valid={sorted(by_name)}")
    return [by_name[name] for name in sorted(wanted)]


def command_train_local(args: argparse.Namespace) -> None:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    _set_threads(int(args.threads))
    dataset_dir = Path(args.dataset_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    split_path = Path(args.split_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(dataset_dir / "X_state_action.npy", mmap_mode="r")
    y = np.load(dataset_dir / "y_centered_logit.npy", mmap_mode="r")
    visit_prob = np.load(dataset_dir / "mcts_visit_prob.npy", mmap_mode="r")
    is_best = np.load(dataset_dir / "is_best_visit_action.npy", mmap_mode="r")
    root_offsets = np.load(dataset_dir / "root_offsets.npy", mmap_mode="r")
    split = np.load(split_path)
    train_rows = split["train_rows"].astype(np.int64)
    eval_rows = split["eval_rows"].astype(np.int64)
    train_roots = split["train_root_groups"].astype(np.int64)
    eval_roots = split["eval_root_groups"].astype(np.int64)

    configs = _parse_policy_configs(args.configs)
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        cfg_dir = output_dir / cfg.name / "Model_Version1"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        config_payload = asdict(cfg)
        (cfg_dir / "model_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        fit_rows = train_rows
        if int(args.fit_subsample_rows) > 0 and int(args.fit_subsample_rows) < train_rows.size:
            rng = np.random.default_rng(int(args.seed))
            fit_rows = np.sort(rng.choice(train_rows, size=int(args.fit_subsample_rows), replace=False)).astype(np.int64)

        est = HistGradientBoostingRegressor(
            loss=cfg.loss,
            max_leaf_nodes=int(cfg.max_leaf_nodes),
            max_iter=int(cfg.max_iter),
            learning_rate=float(cfg.learning_rate),
            l2_regularization=float(cfg.l2_regularization),
            random_state=int(args.seed),
            early_stopping=False,
        )
        print(f"[train:{cfg.name}] fitting rows={fit_rows.size} leaves={cfg.max_leaf_nodes} max_iter={cfg.max_iter}", flush=True)
        t0 = time.time()
        est.fit(X[fit_rows], y[fit_rows])
        fit_elapsed = time.time() - t0
        n_params = _hgb_n_params(est)
        joblib.dump(est, cfg_dir / "model.joblib", compress=3)

        pred_t0 = time.time()
        y_pred = _predict_chunked(est, X, chunk=int(args.predict_chunk))
        predict_elapsed = time.time() - pred_t0
        np.save(cfg_dir / "predictions_all_rows.npy", y_pred.astype(np.float32))

        for split_name, row_idx, root_idx in (("train", train_rows, train_roots), ("eval", eval_rows, eval_roots)):
            rm = _row_metrics(y[row_idx], y_pred[row_idx])
            gm = _group_metrics(root_groups=root_idx, root_offsets=root_offsets, y_pred=y_pred, visit_prob=visit_prob, is_best=is_best)
            metric = {
                "split": split_name,
                "config_name": cfg.name,
                "metric_source": "forward",
                **rm,
                **gm,
                "fit_elapsed_s": float(fit_elapsed),
                "predict_elapsed_s": float(predict_elapsed),
                "n_params_estimate": int(n_params),
            }
            rows.append(metric)
            (cfg_dir / f"metrics_{split_name}.json").write_text(json.dumps(metric, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[train:{cfg.name}] {split_name} metrics={metric}", flush=True)

        meta = {
            "config": config_payload,
            "dataset_dir": str(dataset_dir),
            "split_path": str(split_path),
            "fit_rows": int(fit_rows.size),
            "train_rows": int(train_rows.size),
            "eval_rows": int(eval_rows.size),
            "train_roots": int(train_roots.size),
            "eval_roots": int(eval_roots.size),
            "fit_elapsed_s": float(fit_elapsed),
            "predict_elapsed_s": float(predict_elapsed),
            "n_params_estimate": int(n_params),
        }
        (cfg_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    _write_csv(output_dir / "all_policy_train_eval_metric_summary.csv", SUMMARY_FIELDS, rows)
    print(json.dumps({"output_dir": str(output_dir), "num_metric_rows": len(rows)}, indent=2), flush=True)


def _script_path(remote_repo: str) -> str:
    return f"{remote_repo.rstrip('/')}/vidur/bellman_v4_adv_2000k_multiprocess/multi_server_hgb_training_controller_prior.py"


def command_sync_script(args: argparse.Namespace) -> None:
    local_script = Path(__file__).resolve()
    hosts = {args.train_host} | {w.host for w in _parse_workers(args.workers)}
    for host in sorted(hosts):
        remote_script = _script_path(args.remote_repo)
        _rsync(str(local_script), f"{host}:{remote_script}")
        print(f"[sync] {host}:{remote_script}")


def command_show_plan(args: argparse.Namespace) -> None:
    workers = _parse_workers(args.workers)
    print(f"train_host={args.train_host}")
    print(f"policy_work_dir={_policy_work_dir(args)}")
    print(f"state_feature_assembly_dir={_state_feature_assembly_dir(args)}")
    print(f"input_dir={_input_dir(args)}")
    print(f"assembled_dataset_dir={_assembled_dataset_dir(args)}")
    print(f"output_dir={_output_dir(args)}")
    print(f"policy_alpha={args.policy_alpha} seed={args.seed} eval_roots={args.eval_roots}")
    for worker in workers:
        print(f"\n{worker.worker_id} host={worker.host} job_id={worker.job_id}")
        print(f"  raw_targets={_worker_target_dir(args, worker)}")
        print(f"  policy_targets={_worker_policy_target_dir(args, worker)}")
        print(f"  action_features={_worker_action_feature_dir(args, worker)}")


def command_launch_postprocess_targets(args: argparse.Namespace) -> None:
    if not args.no_sync_script:
        command_sync_script(args)
    for worker in _parse_workers(args.workers):
        out_dir = _worker_policy_target_dir(args, worker)
        raw_dir = _worker_target_dir(args, worker)
        log_path = f"{out_dir.rstrip('/')}/postprocess_targets.log"
        pid_path = f"{out_dir.rstrip('/')}/postprocess_targets.pid"
        cmd = [
            f"{args.remote_repo.rstrip('/')}/.venv/bin/python3",
            _script_path(args.remote_repo),
            "postprocess-targets-local",
            "--input-target-dir",
            raw_dir,
            "--output-dir",
            out_dir,
            "--policy-alpha",
            str(args.policy_alpha),
        ]
        cmd_text = " ".join(shlex.quote(x) for x in cmd)
        remote = (
            "set -e; "
            f"mkdir -p {shlex.quote(out_dir)}; "
            f"nohup {cmd_text} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(pid_path)}; "
            f"echo launched worker={worker.worker_id} pid=$(cat {shlex.quote(pid_path)}) log={shlex.quote(log_path)}"
        )
        res = _ssh(worker.host, remote)
        print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")


def command_status_postprocess_targets(args: argparse.Namespace) -> None:
    for worker in _parse_workers(args.workers):
        out_dir = _worker_policy_target_dir(args, worker)
        remote = (
            f"PID=$(cat {shlex.quote(out_dir + '/postprocess_targets.pid')} 2>/dev/null || true); "
            "RUNNING=0; if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then RUNNING=1; fi; "
            f"echo worker={shlex.quote(worker.worker_id)} host={shlex.quote(worker.host)} pid=$PID running=$RUNNING; "
            f"cat {shlex.quote(out_dir + '/policy_target_manifest.json')} 2>/dev/null || true; "
            f"tail -n {int(args.tail_lines)} {shlex.quote(out_dir + '/postprocess_targets.log')} 2>/dev/null || true"
        )
        res = _ssh(worker.host, remote, check=False)
        print("=" * 100)
        print(res.stdout)


def command_sync_inputs(args: argparse.Namespace) -> None:
    input_dir = _input_dir(args)
    for worker in _parse_workers(args.workers):
        dest = f"{input_dir.rstrip('/')}/{worker.worker_id}"
        policy_src = _worker_policy_target_dir(args, worker).rstrip("/") + "/"
        feature_src = _worker_action_feature_dir(args, worker).rstrip("/") + "/"
        mkdir_remote = (
            f"mkdir -p {shlex.quote(dest + '/policy_targets')} {shlex.quote(dest + '/action_features')}"
        )
        _ssh(args.train_host, mkdir_remote)

        if worker.host == args.train_host:
            remote = (
                "set -e; "
                f"rsync -az {shlex.quote(policy_src)} {shlex.quote(dest + '/policy_targets/')}; "
                f"rsync -az {shlex.quote(feature_src)} {shlex.quote(dest + '/action_features/')}; "
                f"echo synced worker={shlex.quote(worker.worker_id)} dest={shlex.quote(dest)} mode=local_on_train_host"
            )
            res = _ssh(args.train_host, remote)
            print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")
            continue

        # The worker SSH aliases are local to this coordinator, not necessarily
        # resolvable from the training host. Relay through a temporary local dir.
        tmp_root = Path(tempfile.mkdtemp(prefix=f"controller_prior_sync_{worker.worker_id}_"))
        try:
            tmp_policy = tmp_root / "policy_targets"
            tmp_features = tmp_root / "action_features"
            tmp_policy.mkdir(parents=True, exist_ok=True)
            tmp_features.mkdir(parents=True, exist_ok=True)
            _rsync(f"{worker.host}:{policy_src}", str(tmp_policy) + "/")
            _rsync(f"{worker.host}:{feature_src}", str(tmp_features) + "/")
            _rsync(str(tmp_policy) + "/", f"{args.train_host}:{dest}/policy_targets/")
            _rsync(str(tmp_features) + "/", f"{args.train_host}:{dest}/action_features/")
            print(f"synced worker={worker.worker_id} dest={dest} mode=local_relay", flush=True)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

def command_launch_assemble(args: argparse.Namespace) -> None:
    if not args.no_sync_script:
        command_sync_script(args)
    output_dir = _assembled_dataset_dir(args)
    log_path = f"{output_dir.rstrip('/')}/assemble.log"
    pid_path = f"{output_dir.rstrip('/')}/assemble.pid"
    cmd = [
        f"{args.remote_repo.rstrip('/')}/.venv/bin/python3",
        _script_path(args.remote_repo),
        "assemble-local",
        "--workers",
        args.workers,
        "--input-dir",
        _input_dir(args),
        "--state-feature-assembly-dir",
        _state_feature_assembly_dir(args),
        "--output-dir",
        output_dir,
        "--max-rows",
        str(args.max_rows),
    ]
    cmd_text = " ".join(shlex.quote(x) for x in cmd)
    remote = (
        "set -e; "
        f"mkdir -p {shlex.quote(output_dir)}; "
        f"nohup {cmd_text} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(pid_path)}; "
        f"echo launched assemble pid=$(cat {shlex.quote(pid_path)}) log={shlex.quote(log_path)}"
    )
    res = _ssh(args.train_host, remote)
    print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")


def command_launch_train(args: argparse.Namespace) -> None:
    if not args.no_sync_script:
        command_sync_script(args)
    dataset_dir = _assembled_dataset_dir(args)
    split_path = f"{dataset_dir.rstrip('/')}/split_seed{int(args.seed)}_evalroots{int(args.eval_roots)}.npz"
    output_dir = _output_dir(args)
    log_path = f"{output_dir.rstrip('/')}/train.log"
    pid_path = f"{output_dir.rstrip('/')}/train.pid"
    cmd = [
        f"{args.remote_repo.rstrip('/')}/.venv/bin/python3",
        _script_path(args.remote_repo),
        "train-local",
        "--dataset-dir",
        dataset_dir,
        "--split-path",
        split_path,
        "--output-dir",
        output_dir,
        "--configs",
        args.configs,
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--predict-chunk",
        str(args.predict_chunk),
        "--fit-subsample-rows",
        str(args.fit_subsample_rows),
    ]
    cmd_text = " ".join(shlex.quote(x) for x in cmd)
    remote = (
        "set -e; "
        f"mkdir -p {shlex.quote(output_dir)}; "
        f"{shlex.quote(args.remote_repo.rstrip('/') + '/.venv/bin/python3')} {shlex.quote(_script_path(args.remote_repo))} prepare-split-local "
        f"--dataset-dir {shlex.quote(dataset_dir)} --output-path {shlex.quote(split_path)} --seed {int(args.seed)} --eval-roots {int(args.eval_roots)}; "
        f"nohup {cmd_text} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(pid_path)}; "
        f"echo launched train pid=$(cat {shlex.quote(pid_path)}) log={shlex.quote(log_path)}"
    )
    res = _ssh(args.train_host, remote)
    print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")


def command_status_train(args: argparse.Namespace) -> None:
    for name, out_dir in (("assemble", _assembled_dataset_dir(args)), ("train", _output_dir(args))):
        remote = (
            f"PID=$(cat {shlex.quote(out_dir.rstrip('/') + '/' + name + '.pid')} 2>/dev/null || true); "
            "RUNNING=0; if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then RUNNING=1; fi; "
            f"echo name={name} out={shlex.quote(out_dir)} pid=$PID running=$RUNNING; "
            f"find {shlex.quote(out_dir)} -maxdepth 3 \( -name '*summary*.csv' -o -name 'assembly_manifest.json' \) 2>/dev/null | sort | tail -20; "
            f"tail -n {int(args.tail_lines)} {shlex.quote(out_dir.rstrip('/') + '/' + name + '.log')} 2>/dev/null || true"
        )
        res = _ssh(args.train_host, remote, check=False)
        print("=" * 100)
        print(res.stdout)


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    p.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--train-host", default=DEFAULT_TRAIN_HOST)
    p.add_argument("--workers", default="worker1,worker2,worker3,worker4")
    p.add_argument("--model-label", default=DEFAULT_TARGET_MODEL_LABEL)
    p.add_argument("--mcts-iterations", type=int, default=DEFAULT_MCTS_ITERATIONS)
    p.add_argument("--target-dir-suffix", default=DEFAULT_TARGET_DIR_SUFFIX)
    p.add_argument("--action-feature-version", default=DEFAULT_ACTION_FEATURE_VERSION)
    p.add_argument("--policy-alpha", type=float, default=DEFAULT_POLICY_ALPHA)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--eval-roots", type=int, default=DEFAULT_EVAL_ROOTS)
    p.add_argument("--policy-work-dir", default=DEFAULT_POLICY_WORK_DIR)
    p.add_argument("--target-base", default=DEFAULT_TARGET_BASE)
    p.add_argument("--action-feature-base", default=DEFAULT_ACTION_FEATURE_BASE)
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--state-feature-assembly-dir", default=DEFAULT_STATE_FEATURE_ASSEMBLY_DIR)
    p.add_argument("--assembled-dataset-dir", default=DEFAULT_ASSEMBLED_DATASET_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("show-plan")
    add_common(p)
    p.set_defaults(func=command_show_plan)

    p = sub.add_parser("sync-script")
    add_common(p)
    p.set_defaults(func=command_sync_script)

    p = sub.add_parser("postprocess-targets-local")
    p.add_argument("--input-target-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--policy-alpha", type=float, default=DEFAULT_POLICY_ALPHA)
    p.set_defaults(func=command_postprocess_targets_local)

    p = sub.add_parser("launch-postprocess-targets")
    add_common(p)
    p.add_argument("--no-sync-script", action="store_true")
    p.set_defaults(func=command_launch_postprocess_targets)

    p = sub.add_parser("status-postprocess-targets")
    add_common(p)
    p.add_argument("--tail-lines", type=int, default=20)
    p.set_defaults(func=command_status_postprocess_targets)

    p = sub.add_parser("sync-inputs")
    add_common(p)
    p.set_defaults(func=command_sync_inputs)

    p = sub.add_parser("assemble-local")
    p.add_argument("--workers", default="worker1,worker2,worker3,worker4")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--state-feature-assembly-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-rows", type=int, default=0, help="Debug limit; 0 means all rows.")
    p.set_defaults(func=command_assemble_local)

    p = sub.add_parser("launch-assemble")
    add_common(p)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--no-sync-script", action="store_true")
    p.set_defaults(func=command_launch_assemble)

    p = sub.add_parser("prepare-split-local")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-path", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--eval-roots", type=int, default=DEFAULT_EVAL_ROOTS)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=command_prepare_split_local)

    p = sub.add_parser("train-local")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--configs", default="all")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--predict-chunk", type=int, default=DEFAULT_PREDICT_CHUNK)
    p.add_argument("--fit-subsample-rows", type=int, default=DEFAULT_FIT_SUBSAMPLE_ROWS)
    p.set_defaults(func=command_train_local)

    p = sub.add_parser("launch-train")
    add_common(p)
    p.add_argument("--configs", default="all")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--predict-chunk", type=int, default=DEFAULT_PREDICT_CHUNK)
    p.add_argument("--fit-subsample-rows", type=int, default=DEFAULT_FIT_SUBSAMPLE_ROWS)
    p.add_argument("--no-sync-script", action="store_true")
    p.set_defaults(func=command_launch_train)

    p = sub.add_parser("status-train")
    add_common(p)
    p.add_argument("--tail-lines", type=int, default=40)
    p.set_defaults(func=command_status_train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
