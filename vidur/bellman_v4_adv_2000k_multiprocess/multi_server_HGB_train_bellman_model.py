#!/usr/bin/env python3
"""Coordinate distributed HGB Bellman training on assembled GV3 datasets.

The heavy lifting happens on each worker through ``train-local``.  The
coordinator subcommands only prepare the reproducible train/eval split, launch
assigned model configs on their designated hosts, and report status.

The input expected by ``train-local`` is the assembled layout produced by
``assemble_parent_child_dataset.py``:

  assembled_parent_child_features_adv/
    assembly_manifest.json
    parent_shards/<source_parent_feature_dir>/
    child_shards/<source_child_feature_dir>/

The trainer concatenates only parent features in RAM.  Child features stay as
per-shard memmaps and are streamed every Bellman iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_ASSEMBLY_DIR = "{base}/{experiment_name}/assembled_parent_child_features_adv"
DEFAULT_OUTPUT_DIR = "{base}/{experiment_name}/Bellman_HGB_100iter_assembled_226d"
DEFAULT_SEED = 20260613
DEFAULT_EVAL_SIZE = 100_000
DEFAULT_END_ITER = 100
DEFAULT_THREADS_PER = 48
DEFAULT_PREDICT_CHUNK = 524_288
DEFAULT_LAUNCH_ROLES = (
    "worker1_200k_alpha2,"
    "worker2_200k_alpha4,"
    "worker3_400k_alpha2,"
    "worker4_400k_alpha4"
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_leaf_nodes: int
    max_iter: int
    tail_alpha: float
    loss: str = "squared_error"
    learning_rate: float = 0.05
    l2_regularization: float = 1.0

    @property
    def approx_params(self) -> int:
        return int(3 * int(self.max_leaf_nodes) * int(self.max_iter))


@dataclass(frozen=True)
class TrainRole:
    role_id: str
    host: str
    configs: tuple[ModelConfig, ...]


TRAIN_ROLES: dict[str, TrainRole] = {
    "worker1_200k_alpha2": TrainRole(
        "worker1_200k_alpha2",
        "bellman-classical-worker-1",
        (
            ModelConfig("hgb_sq_47leaf_1410iter_a2", 47, 1410, 2.0),
            ModelConfig("hgb_sq_63leaf_1050iter_a2", 63, 1050, 2.0),
            ModelConfig("hgb_sq_95leaf_0700iter_a2", 95, 700, 2.0),
        ),
    ),
    "worker2_200k_alpha4": TrainRole(
        "worker2_200k_alpha4",
        "bellman-classical-worker-2",
        (
            ModelConfig("hgb_sq_47leaf_1410iter_a4", 47, 1410, 4.0),
            ModelConfig("hgb_sq_63leaf_1050iter_a4", 63, 1050, 4.0),
            ModelConfig("hgb_sq_95leaf_0700iter_a4", 95, 700, 4.0),
        ),
    ),
    "worker3_400k_alpha2": TrainRole(
        "worker3_400k_alpha2",
        "bellman-classical-worker-3",
        (
            ModelConfig("hgb_sq_47leaf_2837iter_a2", 47, 2837, 2.0),
            ModelConfig("hgb_sq_63leaf_2116iter_a2", 63, 2116, 2.0),
            ModelConfig("hgb_sq_95leaf_1404iter_a2", 95, 1404, 2.0),
        ),
    ),
    "worker4_400k_alpha4": TrainRole(
        "worker4_400k_alpha4",
        "bellman-classical-worker-4",
        (
            ModelConfig("hgb_sq_47leaf_2837iter_a4", 47, 2837, 4.0),
            ModelConfig("hgb_sq_63leaf_2116iter_a4", 63, 2116, 4.0),
            ModelConfig("hgb_sq_95leaf_1404iter_a4", 95, 1404, 4.0),
        ),
    ),
}


def _set_threads(n: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n)
    except Exception:
        pass


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
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


def _ssh_python(
    host: str,
    remote_repo: str,
    script: str,
    payload: dict[str, Any],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    py = f"{remote_repo.rstrip('/')}/.venv/bin/python3"
    remote = f"{shlex.quote(py)} - {shlex.quote(json.dumps(payload))}"
    return _run(["ssh", host, remote], check=check, input_text=script)


def _base_path(remote_repo: str, remote_output_base: str) -> str:
    return f"{remote_repo.rstrip('/')}/{remote_output_base.strip('/')}"


def _format_assembly_dir(args: argparse.Namespace) -> str:
    base = _base_path(args.remote_repo, args.remote_output_base)
    return str(args.assembly_dir).format(
        base=base,
        experiment_name=args.experiment_name,
        remote_repo=args.remote_repo.rstrip("/"),
    )


def _format_output_dir(args: argparse.Namespace) -> str:
    base = _base_path(args.remote_repo, args.remote_output_base)
    return str(args.output_dir).format(
        base=base,
        experiment_name=args.experiment_name,
        remote_repo=args.remote_repo.rstrip("/"),
    )


def _split_json_path(assembly_dir: str, seed: int, eval_size: int) -> str:
    return (
        f"{assembly_dir.rstrip('/')}/"
        f"train_eval_split_seed{int(seed)}_eval{int(eval_size)}.json"
    )


def parse_role_host_overrides(raw: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                "--role-host-overrides entries must be role=host, "
                f"got {item!r}"
            )
        role_id, host = [x.strip() for x in item.split("=", 1)]
        if role_id not in TRAIN_ROLES:
            raise SystemExit(
                f"unknown role override {role_id!r}; valid={sorted(TRAIN_ROLES)}"
            )
        if not host:
            raise SystemExit(f"empty host override for role {role_id!r}")
        overrides[role_id] = host
    return overrides


def parse_roles(
    raw: str,
    host_overrides: dict[str, str] | None = None,
) -> list[TrainRole]:
    role_ids = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in role_ids if x not in TRAIN_ROLES]
    if unknown:
        raise SystemExit(f"unknown role(s): {unknown}; valid={sorted(TRAIN_ROLES)}")
    host_overrides = host_overrides or {}
    roles: list[TrainRole] = []
    for role_id in role_ids:
        role = TRAIN_ROLES[role_id]
        override = host_overrides.get(role_id)
        if override:
            role = TrainRole(role.role_id, override, role.configs)
        roles.append(role)
    return roles


def _parsed_roles(args: argparse.Namespace) -> list[TrainRole]:
    return parse_roles(
        args.roles,
        parse_role_host_overrides(getattr(args, "role_host_overrides", "")),
    )


def _hgb_n_params(model: HistGradientBoostingRegressor) -> int:
    total = 0
    for est in model._predictors:
        for tree in est:
            try:
                total += int(np.sum(tree.nodes["is_leaf"])) * 3
            except Exception:
                total += int(tree.nodes.size) * 3
    return total


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "n": int(y_true.size),
        "mse": float(np.mean(err * err)) if y_true.size else 0.0,
        "rmse": float(math.sqrt(np.mean(err * err))) if y_true.size else 0.0,
        "mae": float(np.mean(abs_err)) if y_true.size else 0.0,
        "p50_abs": float(np.quantile(abs_err, 0.50)) if y_true.size else 0.0,
        "p90_abs": float(np.quantile(abs_err, 0.90)) if y_true.size else 0.0,
        "p95_abs": float(np.quantile(abs_err, 0.95)) if y_true.size else 0.0,
        "p99_abs": float(np.quantile(abs_err, 0.99)) if y_true.size else 0.0,
        "max_abs": float(abs_err.max()) if y_true.size else 0.0,
        "n_over_05": int(np.sum(abs_err > 0.5)),
        "n_over_1": int(np.sum(abs_err > 1.0)),
        "n_over_2": int(np.sum(abs_err > 2.0)),
        "frac_over_05": float(np.mean(abs_err > 0.5)) if y_true.size else 0.0,
    }


def _write_predictions_csv(
    path: Path,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    abs_err = np.abs(y_pred - y_true)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["psid", "y_true", "y_pred", "abs_err"])
        for i in range(len(idx)):
            w.writerow(
                [
                    int(idx[i]),
                    float(y_true[i]),
                    float(y_pred[i]),
                    float(abs_err[i]),
                ]
            )


def _predict_chunked(
    model: Any,
    X: np.ndarray,
    *,
    chunk: int = DEFAULT_PREDICT_CHUNK,
) -> np.ndarray:
    n = int(X.shape[0])
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        block = np.asarray(X[s:e])
        pred = model.predict(block).astype(np.float32)
        np.minimum(pred, 0.0, out=pred)
        out[s:e] = pred
    return out


def _load_child_cache(cache_dir: Path) -> dict[str, Any]:
    feats = np.load(cache_dir / "child_features.npy", mmap_mode="r")
    pi = np.load(cache_dir / "parent_index.npz")
    required = (
        "order",
        "parent_ids",
        "offsets",
        "rewards",
        "discounts",
        "is_valid",
        "controller_action_indices",
        "adversary_action_indices",
    )
    missing = [name for name in required if name not in pi.files]
    if missing:
        raise RuntimeError(f"{cache_dir}/parent_index.npz missing keys={missing}")
    return {
        "dir": str(cache_dir),
        "features": feats,
        "order": pi["order"],
        "parent_ids": pi["parent_ids"].astype(np.int64),
        "offsets": pi["offsets"].astype(np.int64),
        "rewards": pi["rewards"].astype(np.float32),
        "discounts": pi["discounts"].astype(np.float32),
        "is_valid": pi["is_valid"].astype(np.bool_),
        "controller_action_indices": pi["controller_action_indices"].astype(np.int64),
        "adversary_action_indices": pi["adversary_action_indices"].astype(np.int64),
    }


def _update_targets_from_child_cache(
    *,
    cache: dict[str, Any],
    child_v: np.ndarray,
    global_offset: int,
    local_parent_rows: int,
    total_parent_rows: int,
    parent_target: np.ndarray,
) -> None:
    """Max-controller/min-adversary Bellman update for one child shard.

    Child caches normally use parent_state_id local to the source parent shard,
    so the shard's global offset is added.  If ids already look global, they are
    accepted as-is for forward compatibility.
    """
    parent_ids = cache["parent_ids"]
    if parent_ids.size == 0:
        return

    max_pid = int(parent_ids.max())
    min_pid = int(parent_ids.min())
    if min_pid >= 0 and max_pid < int(local_parent_rows):
        def to_global(pid: int) -> int:
            return int(global_offset) + int(pid)
    elif min_pid >= 0 and max_pid < int(total_parent_rows):
        def to_global(pid: int) -> int:
            return int(pid)
    else:
        raise RuntimeError(
            f"cannot map parent ids for {cache['dir']}: "
            f"range=[{min_pid}, {max_pid}] local_rows={local_parent_rows} "
            f"total_rows={total_parent_rows}"
        )

    order = cache["order"]
    rewards_sorted = cache["rewards"][order]
    disc_sorted = cache["discounts"][order]
    valid_sorted = cache["is_valid"][order]
    ctrl_sorted = cache["controller_action_indices"][order]
    cv_sorted = child_v[order]

    candidates = rewards_sorted + disc_sorted * cv_sorted
    np.minimum(candidates, 0.0, out=candidates)

    offsets = cache["offsets"]
    for i, pid in enumerate(parent_ids):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        if s == e:
            continue
        slot = to_global(int(pid))
        if slot < 0 or slot >= int(total_parent_rows):
            continue

        valid = valid_sorted[s:e]
        if not np.any(valid):
            continue

        seg_candidates = candidates[s:e][valid]
        seg_ctrl = ctrl_sorted[s:e][valid]
        controller_values: list[float] = []
        for ctrl_action in np.unique(seg_ctrl):
            mask = seg_ctrl == ctrl_action
            if np.any(mask):
                controller_values.append(float(np.min(seg_candidates[mask])))
        if not controller_values:
            continue

        value = float(np.max(np.asarray(controller_values, dtype=np.float32)))
        if value > parent_target[slot]:
            parent_target[slot] = value


@dataclass(frozen=True)
class ParentShard:
    job_id: str
    host: str
    source_dir: str
    local_dir: Path
    rows: int
    dim: int
    global_offset: int


@dataclass(frozen=True)
class ChildShard:
    job_id: str
    host: str
    source_dir: str
    local_dir: Path
    parent_global_offset: int
    parent_rows: int


def _read_assembly_manifest(assembly_dir: Path) -> dict[str, Any]:
    path = assembly_dir / "assembly_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"assembly manifest not found: {path}")
    return json.loads(path.read_text())


def _manifest_entries(manifest: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    entries = manifest.get(dataset) or []
    entries = [e for e in entries if e.get("valid", True)]
    entries.sort(key=lambda e: str(e.get("job_id", "")))
    return entries


def _discover_parent_shards(assembly_dir: Path) -> list[ParentShard]:
    manifest = _read_assembly_manifest(assembly_dir)
    entries = _manifest_entries(manifest, "parent")
    if not entries:
        raise RuntimeError(f"no valid parent entries in {assembly_dir}/assembly_manifest.json")

    out: list[ParentShard] = []
    cursor = 0
    for e in entries:
        local_dir = Path(str(e["local_dir"]))
        meta_path = local_dir / "parent_features.meta.json"
        meta = json.loads(meta_path.read_text())
        rows = int(meta["num_records"])
        dim = int(meta["feature_dim"])
        if dim != 226:
            raise RuntimeError(f"{local_dir} feature_dim={dim}, expected 226")
        out.append(
            ParentShard(
                job_id=str(e["job_id"]),
                host=str(e.get("source_host", "")),
                source_dir=str(e.get("source_dir", "")),
                local_dir=local_dir,
                rows=rows,
                dim=dim,
                global_offset=cursor,
            )
        )
        cursor += rows
    return out


def _discover_child_shards(
    assembly_dir: Path,
    parents: list[ParentShard],
) -> list[ChildShard]:
    manifest = _read_assembly_manifest(assembly_dir)
    entries = _manifest_entries(manifest, "child")
    if not entries:
        raise RuntimeError(f"no valid child entries in {assembly_dir}/assembly_manifest.json")

    parent_by_job = {p.job_id: p for p in parents}
    out: list[ChildShard] = []
    for e in entries:
        job_id = str(e["job_id"])
        parent = parent_by_job.get(job_id)
        if parent is None:
            raise RuntimeError(f"child shard {job_id} has no matching parent shard")
        local_dir = Path(str(e["local_dir"]))
        for required in ("child_features.npy", "parent_index.npz"):
            if not (local_dir / required).exists():
                raise FileNotFoundError(f"missing {required} in {local_dir}")
        out.append(
            ChildShard(
                job_id=job_id,
                host=str(e.get("source_host", "")),
                source_dir=str(e.get("source_dir", "")),
                local_dir=local_dir,
                parent_global_offset=parent.global_offset,
                parent_rows=parent.rows,
            )
        )
    out.sort(key=lambda c: c.job_id)
    return out


def _load_parent_matrix(parents: list[ParentShard]) -> tuple[np.ndarray, np.ndarray]:
    total_rows = sum(p.rows for p in parents)
    X = np.empty((total_rows, 226), dtype=np.float32)
    y = np.empty((total_rows,), dtype=np.float32)
    for p in parents:
        s = p.global_offset
        e = s + p.rows
        feats = np.load(p.local_dir / "parent_features.npy", mmap_mode="r")
        targets = np.load(p.local_dir / "parent_targets.npy", mmap_mode="r")
        if feats.shape != (p.rows, 226):
            raise RuntimeError(f"{p.local_dir} feature shape={feats.shape}, expected {(p.rows, 226)}")
        if targets.shape != (p.rows,):
            raise RuntimeError(f"{p.local_dir} target shape={targets.shape}, expected {(p.rows,)}")
        X[s:e] = np.asarray(feats)
        y[s:e] = np.asarray(targets)
    return X, y


def _ensure_split(
    split_json: Path,
    *,
    total_rows: int,
    eval_size: int,
    seed: int,
    parent_shards: list[ParentShard],
) -> tuple[np.ndarray, np.ndarray]:
    if split_json.exists():
        split = json.loads(split_json.read_text())
        train_idx = np.asarray(split["train_indices"], dtype=np.int64)
        eval_idx = np.asarray(split["eval_indices"], dtype=np.int64)
        if int(split.get("total_rows", -1)) != int(total_rows):
            raise RuntimeError(
                f"split total_rows mismatch: {split.get('total_rows')} != {total_rows}"
            )
        return train_idx, eval_idx

    if eval_size <= 0 or eval_size >= total_rows:
        raise ValueError(f"eval_size must be in [1, total_rows); got {eval_size}")
    rng = np.random.default_rng(int(seed))
    eval_idx = np.sort(rng.choice(total_rows, size=eval_size, replace=False)).astype(np.int64)
    is_eval = np.zeros(total_rows, dtype=bool)
    is_eval[eval_idx] = True
    train_idx = np.nonzero(~is_eval)[0].astype(np.int64)
    payload = {
        "seed": int(seed),
        "eval_size": int(eval_size),
        "total_rows": int(total_rows),
        "train_size": int(train_idx.size),
        "eval_indices": eval_idx.tolist(),
        "train_indices": train_idx.tolist(),
        "parent_shards": [
            {
                "job_id": p.job_id,
                "rows": p.rows,
                "global_offset": p.global_offset,
                "local_dir": str(p.local_dir),
            }
            for p in parent_shards
        ],
    }
    split_json.parent.mkdir(parents=True, exist_ok=True)
    split_json.write_text(json.dumps(payload, indent=2))
    return train_idx, eval_idx


def _compute_targets_bellman(
    *,
    child_shards: list[ChildShard],
    total_parent_rows: int,
    v_child_fn: Any,
) -> np.ndarray:
    target = np.full(total_parent_rows, -np.inf, dtype=np.float32)
    for shard in child_shards:
        cache = _load_child_cache(shard.local_dir)
        t0 = time.time()
        child_v = v_child_fn(cache["features"])
        print(
            f"[train] child_v {shard.job_id}: rows={child_v.shape[0]} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        _update_targets_from_child_cache(
            cache=cache,
            child_v=child_v,
            global_offset=shard.parent_global_offset,
            local_parent_rows=shard.parent_rows,
            total_parent_rows=total_parent_rows,
            parent_target=target,
        )
        del child_v
        del cache

    n_no_child = int(np.sum(~np.isfinite(target)))
    if n_no_child:
        print(f"[train] {n_no_child} parents had no valid child; target=0", flush=True)
        target[~np.isfinite(target)] = 0.0
    np.minimum(target, 0.0, out=target)
    return target


def _write_iter_summary_tsv(
    path: Path,
    *,
    n: int,
    n_params: int,
    fit_elapsed: float,
    m_train: dict[str, float],
    m_eval: dict[str, float],
    same_train_prev: dict[str, float] | None,
    same_eval_prev: dict[str, float] | None,
    start_iter: int,
) -> None:
    write_header = not path.exists()
    with path.open("a") as f:
        if write_header:
            f.write(
                "iter\tparams\tfit_s\t"
                "fwd_train_max\tfwd_eval_max\tfwd_train_p95\tfwd_eval_p95\t"
                "same_train_max_for_iter_n-1\tsame_eval_max_for_iter_n-1\t"
                "same_train_p95_for_iter_n-1\tsame_eval_p95_for_iter_n-1\t"
                "fwd_eval_n>0.5\tsame_eval_n>0.5_for_iter_n-1\n"
            )
        if same_train_prev and same_eval_prev:
            same_train_max = same_train_prev.get("max_abs", float("nan"))
            same_eval_max = same_eval_prev.get("max_abs", float("nan"))
            same_train_p95 = same_train_prev.get("p95_abs", float("nan"))
            same_eval_p95 = same_eval_prev.get("p95_abs", float("nan"))
            same_eval_n05 = same_eval_prev.get("n_over_05", -1)
        else:
            same_train_max = same_eval_max = same_train_p95 = same_eval_p95 = float("nan")
            same_eval_n05 = -1
        f.write(
            f"{n}\t{n_params}\t{fit_elapsed:.1f}\t"
            f"{m_train['max_abs']:.6f}\t{m_eval['max_abs']:.6f}\t"
            f"{m_train['p95_abs']:.6f}\t{m_eval['p95_abs']:.6f}\t"
            f"{same_train_max:.6f}\t{same_eval_max:.6f}\t"
            f"{same_train_p95:.6f}\t{same_eval_p95:.6f}\t"
            f"{m_eval['n_over_05']}\t{same_eval_n05}\n"
        )


def _train_one_config(
    *,
    config: ModelConfig,
    X: np.ndarray,
    parent_targets: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    child_shards: list[ChildShard],
    out_dir: Path,
    start_iter: int,
    end_iter: int,
    predict_chunk: int,
    skip_train_csv: bool,
    skip_v1_target_consistency_check: bool,
    v1_target_consistency_atol: float,
    shared_iter1_target: np.ndarray | None,
) -> None:
    cfg_dir = out_dir / config.name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "model_config.json").write_text(json.dumps(asdict(config), indent=2))

    total_rows = int(X.shape[0])
    iter_summary_path = cfg_dir / "iter_summary.json"
    iter_summary: list[dict[str, Any]] = []
    if start_iter > 1 and iter_summary_path.exists():
        try:
            loaded_summary = json.loads(iter_summary_path.read_text())
            if isinstance(loaded_summary, list):
                # Keep completed earlier iterations and append the resumed run.
                iter_summary = [
                    entry for entry in loaded_summary
                    if int(entry.get("iter", -1)) < int(start_iter)
                ]
        except Exception as exc:
            print(
                f"[train:{config.name}] warning: could not load existing "
                f"{iter_summary_path}: {exc}; starting a fresh summary",
                flush=True,
            )
    prev_yp_full: np.ndarray | None = None
    precomputed_iter1_target = shared_iter1_target
    if precomputed_iter1_target is not None:
        (cfg_dir / "v1_target_consistency_reused.json").write_text(
            json.dumps({"source": "role-level preflight"}, indent=2)
        )

    for n in range(start_iter, end_iter + 1):
        sub = cfg_dir / f"Model_Version{n}"
        sub.mkdir(parents=True, exist_ok=True)
        print(f"\n[train:{config.name}] ===== Bellman iter {n} =====", flush=True)
        it_t0 = time.time()

        same_pred_full = prev_yp_full
        if n == 1:
            if precomputed_iter1_target is not None:
                target = precomputed_iter1_target.copy()
            else:
                target = _compute_targets_bellman(
                    child_shards=child_shards,
                    total_parent_rows=total_rows,
                    v_child_fn=lambda feats: np.zeros(feats.shape[0], dtype=np.float32),
                )
        else:
            prev_path = cfg_dir / f"Model_Version{n - 1}" / "model.joblib"
            print(f"[train:{config.name}] loading {prev_path}", flush=True)
            v_prev = joblib.load(prev_path)

            def v_pred(feats: np.ndarray, model: Any = v_prev) -> np.ndarray:
                return _predict_chunked(model, feats, chunk=predict_chunk)

            target = _compute_targets_bellman(
                child_shards=child_shards,
                total_parent_rows=total_rows,
                v_child_fn=v_pred,
            )
            if same_pred_full is None:
                print(
                    f"[train:{config.name}] predicting previous model on "
                    "parent states for resumed same-version metrics",
                    flush=True,
                )
                same_pred_full = _predict_chunked(v_prev, X, chunk=predict_chunk)
            del v_prev

        same_train_prev: dict[str, float] | None = None
        same_eval_prev: dict[str, float] | None = None
        if n > 1 and same_pred_full is not None:
            prev_sub = cfg_dir / f"Model_Version{n - 1}"
            same_train_prev = _metrics(target[train_idx], same_pred_full[train_idx])
            same_eval_prev = _metrics(target[eval_idx], same_pred_full[eval_idx])
            if not skip_train_csv:
                _write_predictions_csv(
                    prev_sub / "train_results_same_version.csv",
                    train_idx,
                    target[train_idx],
                    same_pred_full[train_idx],
                )
            _write_predictions_csv(
                prev_sub / "eval_results_same_version.csv",
                eval_idx,
                target[eval_idx],
                same_pred_full[eval_idx],
            )
            meta_path = prev_sub / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["metrics_train_same_version"] = same_train_prev
                meta["metrics_eval_same_version"] = same_eval_prev
                meta_path.write_text(json.dumps(meta, indent=2))
            for entry in iter_summary:
                if entry.get("iter") == n - 1:
                    entry["metrics_train_same_version"] = same_train_prev
                    entry["metrics_eval_same_version"] = same_eval_prev
                    break

        y_train = target[train_idx]
        y_eval = target[eval_idx]
        sample_weight = (1.0 + config.tail_alpha * np.abs(y_train)).astype(np.float32)
        est = HistGradientBoostingRegressor(
            loss=config.loss,
            max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            learning_rate=config.learning_rate,
            l2_regularization=config.l2_regularization,
            random_state=0,
            early_stopping=False,
        )

        fit_t0 = time.time()
        print(
            f"[train:{config.name}] fitting HGB leaves={config.max_leaf_nodes} "
            f"max_iter={config.max_iter} alpha={config.tail_alpha}",
            flush=True,
        )
        est.fit(X[train_idx], y_train, sample_weight=sample_weight)
        fit_elapsed = time.time() - fit_t0
        n_params = _hgb_n_params(est)
        print(
            f"[train:{config.name}] fit_elapsed={fit_elapsed:.1f}s "
            f"n_params={n_params:,}",
            flush=True,
        )

        yp_full = _predict_chunked(est, X, chunk=predict_chunk)
        yp_train = yp_full[train_idx]
        yp_eval = yp_full[eval_idx]
        m_train = _metrics(y_train, yp_train)
        m_eval = _metrics(y_eval, yp_eval)
        print(f"[train:{config.name}] forward train={m_train}", flush=True)
        print(f"[train:{config.name}] forward eval={m_eval}", flush=True)

        if not skip_train_csv:
            _write_predictions_csv(sub / "train_results.csv", train_idx, y_train, yp_train)
        _write_predictions_csv(sub / "eval_results.csv", eval_idx, y_eval, yp_eval)
        joblib.dump(est, sub / "model.joblib", compress=3)

        meta = {
            "iter": int(n),
            "config": asdict(config),
            "approx_params_requested": config.approx_params,
            "n_params_estimate": int(n_params),
            "fit_elapsed_s": float(fit_elapsed),
            "iter_total_elapsed_s": float(time.time() - it_t0),
            "metrics_train_forward": m_train,
            "metrics_eval_forward": m_eval,
            "metrics_train_same_version": None,
            "metrics_eval_same_version": None,
        }
        (sub / "metadata.json").write_text(json.dumps(meta, indent=2))
        iter_summary.append(meta)
        iter_summary_path.write_text(json.dumps(iter_summary, indent=2))
        _write_iter_summary_tsv(
            cfg_dir / "iter_summary.tsv",
            n=n,
            n_params=n_params,
            fit_elapsed=fit_elapsed,
            m_train=m_train,
            m_eval=m_eval,
            same_train_prev=same_train_prev,
            same_eval_prev=same_eval_prev,
            start_iter=start_iter,
        )

        prev_yp_full = yp_full
        del est
        del target

    print(
        f"[train:{config.name}] done. V_{end_iter} same-version CSV requires "
        f"iteration {end_iter + 1}, so it is intentionally absent.",
        flush=True,
    )


def command_train_local(args: argparse.Namespace) -> None:
    _set_threads(args.threads_per)
    role = TRAIN_ROLES.get(args.role)
    if role is None:
        raise SystemExit(f"unknown --role={args.role}; valid={sorted(TRAIN_ROLES)}")

    assembly_dir = Path(args.assembly_dir)
    out_dir = Path(args.output_dir) / role.role_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "role_config.json").write_text(json.dumps(asdict(role), indent=2))

    print(f"[train] role={role.role_id}", flush=True)
    print(f"[train] assembly_dir={assembly_dir}", flush=True)
    print(f"[train] output_dir={out_dir}", flush=True)

    parents = _discover_parent_shards(assembly_dir)
    child_shards = _discover_child_shards(assembly_dir, parents)
    total_rows = sum(p.rows for p in parents)
    print(
        f"[train] parent_shards={len(parents)} child_shards={len(child_shards)} "
        f"total_parent_rows={total_rows}",
        flush=True,
    )

    split_json = Path(args.split_json)
    train_idx, eval_idx = _ensure_split(
        split_json,
        total_rows=total_rows,
        eval_size=args.eval_size,
        seed=args.seed,
        parent_shards=parents,
    )
    print(
        f"[train] split={split_json} train={train_idx.size} eval={eval_idx.size}",
        flush=True,
    )

    print("[train] loading parent matrix into RAM", flush=True)
    X, parent_targets = _load_parent_matrix(parents)
    print(f"[train] X={X.shape} targets={parent_targets.shape}", flush=True)

    selected_configs = role.configs
    if args.only_configs:
        wanted = {x.strip() for x in args.only_configs.split(",") if x.strip()}
        selected_configs = tuple(c for c in selected_configs if c.name in wanted)
        missing = wanted - {c.name for c in selected_configs}
        if missing:
            raise SystemExit(f"unknown config(s) for role {role.role_id}: {sorted(missing)}")


    shared_iter1_target: np.ndarray | None = None
    if args.start_iter == 1 and not args.skip_v1_target_consistency_check:
        print("[train] preflight V1 target consistency once for this role", flush=True)
        shared_iter1_target = _compute_targets_bellman(
            child_shards=child_shards,
            total_parent_rows=total_rows,
            v_child_fn=lambda feats: np.zeros(feats.shape[0], dtype=np.float32),
        )
        abs_diff = np.abs(shared_iter1_target - parent_targets)
        report = {
            "atol": float(args.v1_target_consistency_atol),
            "n": int(abs_diff.size),
            "n_bad": int(np.sum(abs_diff > args.v1_target_consistency_atol)),
            "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
            "p95_abs_diff": float(np.quantile(abs_diff, 0.95)) if abs_diff.size else 0.0,
        }
        (out_dir / "v1_target_consistency.json").write_text(json.dumps(report, indent=2))
        print(f"[train] V1 target consistency report={report}", flush=True)
        if report["n_bad"]:
            raise RuntimeError(
                f"V1 target mismatch for role {role.role_id}: {report}; "
                f"see {out_dir / 'v1_target_consistency.json'}"
            )

    for config in selected_configs:
        _train_one_config(
            config=config,
            X=X,
            parent_targets=parent_targets,
            train_idx=train_idx,
            eval_idx=eval_idx,
            child_shards=child_shards,
            out_dir=out_dir,
            start_iter=args.start_iter,
            end_iter=args.end_iter,
            predict_chunk=args.predict_chunk,
            skip_train_csv=args.skip_train_csv,
            skip_v1_target_consistency_check=args.skip_v1_target_consistency_check,
            v1_target_consistency_atol=args.v1_target_consistency_atol,
            shared_iter1_target=shared_iter1_target,
        )


PREPARE_SPLIT_SCRIPT = r"""
import json
import numpy as np
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
assembly_dir = Path(payload["assembly_dir"])
split_json = Path(payload["split_json"])
seed = int(payload["seed"])
eval_size = int(payload["eval_size"])

manifest = json.loads((assembly_dir / "assembly_manifest.json").read_text())
parents = [e for e in manifest.get("parent", []) if e.get("valid")]
parents.sort(key=lambda e: str(e.get("job_id", "")))
rows = []
cursor = 0
for entry in parents:
    meta = json.loads((Path(entry["local_dir"]) / "parent_features.meta.json").read_text())
    n = int(meta["num_records"])
    rows.append({
        "job_id": entry["job_id"],
        "host": entry.get("source_host", ""),
        "local_dir": entry["local_dir"],
        "rows": n,
        "global_offset": cursor,
    })
    cursor += n

if cursor <= eval_size:
    raise SystemExit(f"eval_size={eval_size} must be smaller than total_rows={cursor}")

if split_json.exists():
    existing = json.loads(split_json.read_text())
    if int(existing.get("total_rows", -1)) != cursor:
        raise SystemExit(
            f"existing split total_rows={existing.get('total_rows')} != {cursor}: {split_json}"
        )
    print(json.dumps({
        "status": "exists",
        "split_json": str(split_json),
        "total_rows": cursor,
        "eval_size": int(existing.get("eval_size", eval_size)),
    }))
    raise SystemExit(0)

rng = np.random.default_rng(seed)
eval_idx = np.sort(rng.choice(cursor, size=eval_size, replace=False)).astype(np.int64)
mask = np.zeros(cursor, dtype=bool)
mask[eval_idx] = True
train_idx = np.nonzero(~mask)[0].astype(np.int64)
payload_out = {
    "seed": seed,
    "eval_size": eval_size,
    "total_rows": cursor,
    "train_size": int(train_idx.size),
    "eval_indices": eval_idx.tolist(),
    "train_indices": train_idx.tolist(),
    "parent_shards": rows,
}
split_json.parent.mkdir(parents=True, exist_ok=True)
split_json.write_text(json.dumps(payload_out, indent=2))
print(json.dumps({
    "status": "created",
    "split_json": str(split_json),
    "total_rows": cursor,
    "train_size": int(train_idx.size),
    "eval_size": eval_size,
}))
"""


def command_prepare_split(args: argparse.Namespace) -> None:
    assembly_dir = _format_assembly_dir(args)
    split_json = args.split_json or _split_json_path(
        assembly_dir, args.seed, args.eval_size
    )
    hosts = sorted({role.host for role in _parsed_roles(args)})
    for host in hosts:
        res = _ssh_python(
            host,
            args.remote_repo,
            PREPARE_SPLIT_SCRIPT,
            {
                "assembly_dir": assembly_dir,
                "split_json": split_json,
                "seed": args.seed,
                "eval_size": args.eval_size,
            },
        )
        print(f"[{host}] {res.stdout}", end="" if res.stdout.endswith("\n") else "\n")


def command_show_plan(args: argparse.Namespace) -> None:
    assembly_dir = _format_assembly_dir(args)
    output_dir = _format_output_dir(args)
    split_json = args.split_json or _split_json_path(
        assembly_dir, args.seed, args.eval_size
    )
    print(f"assembly_dir={assembly_dir}")
    print(f"output_dir={output_dir}")
    print(f"split_json={split_json}")
    print(f"seed={args.seed} eval_size={args.eval_size} end_iter={args.end_iter}")
    for role in _parsed_roles(args):
        print(f"\nrole={role.role_id} host={role.host}")
        for cfg in role.configs:
            print(
                f"  {cfg.name}: leaves={cfg.max_leaf_nodes} "
                f"max_iter={cfg.max_iter} approx_params={cfg.approx_params} "
                f"alpha={cfg.tail_alpha} loss={cfg.loss}"
            )


def _script_path(remote_repo: str) -> str:
    return (
        f"{remote_repo.rstrip('/')}/"
        "vidur/bellman_v4_adv_2000k_multiprocess/"
        "multi_server_HGB_train_bellman_model.py"
    )


def command_sync_script(args: argparse.Namespace) -> None:
    local_script = Path(__file__).resolve()
    roles = _parsed_roles(args)
    for host in sorted({r.host for r in roles}):
        remote_script = _script_path(args.remote_repo)
        _run(["rsync", "-az", str(local_script), f"{host}:{remote_script}"])
        print(f"[sync] {host}:{remote_script}")


def command_launch(args: argparse.Namespace) -> None:
    if not args.no_sync_script:
        command_sync_script(args)
    if args.prepare_split:
        command_prepare_split(args)

    assembly_dir = _format_assembly_dir(args)
    output_dir = _format_output_dir(args)
    split_json = args.split_json or _split_json_path(
        assembly_dir, args.seed, args.eval_size
    )

    for role in _parsed_roles(args):
        role_out = f"{output_dir.rstrip('/')}/{role.role_id}"
        log_dir = f"{output_dir.rstrip('/')}/launcher_logs"
        remote_script = _script_path(args.remote_repo)

        def base_cmd() -> list[str]:
            cmd = [
                f"{args.remote_repo.rstrip('/')}/.venv/bin/python3",
                remote_script,
                "train-local",
                "--role",
                role.role_id,
                "--assembly-dir",
                assembly_dir,
                "--output-dir",
                output_dir,
                "--split-json",
                split_json,
                "--seed",
                str(args.seed),
                "--eval-size",
                str(args.eval_size),
                "--start-iter",
                str(args.start_iter),
                "--end-iter",
                str(args.end_iter),
                "--threads-per",
                str(args.threads_per),
                "--predict-chunk",
                str(args.predict_chunk),
            ]
            if args.skip_train_csv:
                cmd.append("--skip-train-csv")
            if args.skip_v1_target_consistency_check:
                cmd.append("--skip-v1-target-consistency-check")
            return cmd

        if args.parallel_configs:
            for cfg in role.configs:
                cmd = base_cmd() + ["--only-configs", cfg.name]
                cmd_text = " ".join(shlex.quote(x) for x in cmd)
                pid_path = f"{role_out}/pids/{cfg.name}.pid"
                command_path = f"{role_out}/commands/{cfg.name}.txt"
                log_path = f"{log_dir}/{role.role_id}__{cfg.name}.log"
                remote = (
                    "set -e; "
                    f"mkdir -p {shlex.quote(role_out + '/pids')} "
                    f"{shlex.quote(role_out + '/commands')} "
                    f"{shlex.quote(log_dir)}; "
                    f"rm -f {shlex.quote(role_out + '/train.pid')}; "
                    f"printf '%s\n' {shlex.quote(cmd_text)} > {shlex.quote(command_path)}; "
                    f"nohup {cmd_text} > {shlex.quote(log_path)} "
                    "2>&1 < /dev/null & "
                    f"echo $! > {shlex.quote(pid_path)}; "
                    f"echo launched role={shlex.quote(role.role_id)} "
                    f"config={shlex.quote(cfg.name)} "
                    f"pid=$(cat {shlex.quote(pid_path)}) "
                    f"log={shlex.quote(log_path)}"
                )
                res = _ssh(role.host, remote)
                print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")
        else:
            cmd = base_cmd()
            cmd_text = " ".join(shlex.quote(x) for x in cmd)
            remote = (
                "set -e; "
                f"mkdir -p {shlex.quote(role_out)} {shlex.quote(log_dir)}; "
                f"printf '%s\n' {shlex.quote(cmd_text)} > {shlex.quote(role_out + '/command.txt')}; "
                f"nohup {cmd_text} > {shlex.quote(log_dir + '/' + role.role_id + '.log')} "
                "2>&1 < /dev/null & "
                f"echo $! > {shlex.quote(role_out + '/train.pid')}; "
                f"echo launched role={shlex.quote(role.role_id)} "
                f"pid=$(cat {shlex.quote(role_out + '/train.pid')}) "
                f"log={shlex.quote(log_dir + '/' + role.role_id + '.log')}"
            )
            res = _ssh(role.host, remote)
            print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")


def command_status(args: argparse.Namespace) -> None:
    output_dir = _format_output_dir(args)
    rows: list[dict[str, Any]] = []
    for role in _parsed_roles(args):
        role_out = f"{output_dir.rstrip('/')}/{role.role_id}"
        log_path = f"{output_dir.rstrip('/')}/launcher_logs/{role.role_id}.log"
        remote = (
            f"PID=$(cat {shlex.quote(role_out + '/train.pid')} 2>/dev/null || true); "
            "RUNNING=0; if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then RUNNING=1; fi; "
            f"echo STATUS_ROLE={shlex.quote(role.role_id)}; "
            f"echo STATUS_PID=$PID; echo STATUS_RUNNING=$RUNNING; "
            f"echo STATUS_LOG={shlex.quote(log_path)}; "
            "echo STATUS_CONFIGS_BEGIN; "
            f"for pf in {shlex.quote(role_out + '/pids')}/*.pid; do "
            "[ -e \"$pf\" ] || continue; "
            "cfg=$(basename \"$pf\" .pid); pid=$(cat \"$pf\" 2>/dev/null || true); "
            "running=0; if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then running=1; fi; "
            "echo CONFIG_STATUS \"$cfg\" \"$pid\" \"$running\"; "
            "done; "
            "echo STATUS_CONFIGS_END; "
            "echo STATUS_TAIL_BEGIN; "
            f"tail -n {int(args.tail_lines)} {shlex.quote(log_path)} 2>/dev/null || true; "
            f"for lf in {shlex.quote(output_dir.rstrip('/') + '/launcher_logs/' + role.role_id + '__')}*.log; do "
            "[ -e \"$lf\" ] || continue; echo \"--- $(basename \"$lf\") ---\"; "
            f"tail -n {int(args.tail_lines)} \"$lf\" 2>/dev/null || true; "
            "done; "
            "echo STATUS_TAIL_END"
        )
        res = _ssh(role.host, remote, check=False)
        row: dict[str, Any] = {"role": role.role_id, "host": role.host, "raw": res.stdout}
        for line in res.stdout.splitlines():
            if line.startswith("STATUS_PID="):
                row["pid"] = line.split("=", 1)[1]
            elif line.startswith("STATUS_RUNNING="):
                row["running"] = line.split("=", 1)[1] == "1"
            elif line.startswith("STATUS_LOG="):
                row["log"] = line.split("=", 1)[1]
            elif line.startswith("CONFIG_STATUS "):
                row.setdefault("configs", []).append(line)
                parts = line.split()
                if len(parts) >= 4 and parts[3] == "1":
                    row["running"] = True
                if len(parts) >= 3:
                    row.setdefault("config_pids", []).append(f"{parts[1]}:{parts[2]}")
        if row.get("config_pids"):
            row["pid"] = ",".join(row["config_pids"])
        rows.append(row)

    for row in rows:
        print("=" * 100)
        print(
            f"{row['host']} {row['role']} running={row.get('running')} "
            f"pid={row.get('pid', '')} log={row.get('log', '')}"
        )
        if row.get("configs"):
            print("\n".join(row["configs"]))
        tail = str(row.get("raw", "")).split("STATUS_TAIL_BEGIN", 1)[-1]
        tail = tail.split("STATUS_TAIL_END", 1)[0]
        print(tail)
    print("=" * 100)
    w = csv.DictWriter(sys.stdout, fieldnames=["role", "host", "running", "pid", "log"])
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in w.fieldnames})


def command_stop(args: argparse.Namespace) -> None:
    output_dir = _format_output_dir(args)
    for role in _parsed_roles(args):
        role_out = f"{output_dir.rstrip('/')}/{role.role_id}"
        remote = (
            f"PID=$(cat {shlex.quote(role_out + '/train.pid')} 2>/dev/null || true); "
            "if [ -n \"$PID\" ]; then kill -TERM \"$PID\" 2>/dev/null || true; fi; "
            f"for pf in {shlex.quote(role_out + '/pids')}/*.pid; do "
            "[ -e \"$pf\" ] || continue; "
            "pid=$(cat \"$pf\" 2>/dev/null || true); "
            "if [ -n \"$pid\" ]; then kill -TERM \"$pid\" 2>/dev/null || true; fi; "
            "done; "
            f"echo stopped role={shlex.quote(role.role_id)} pid=$PID"
        )
        res = _ssh(role.host, remote, check=False)
        print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")


def add_remote_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    p.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--assembly-dir", default=DEFAULT_ASSEMBLY_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--roles", default=DEFAULT_LAUNCH_ROLES)
    p.add_argument(
        "--role-host-overrides",
        default="",
        help="Comma-separated role=host overrides, e.g. "
        "worker4_400k_alpha4=bellman-classical-worker-1",
    )
    p.add_argument("--split-json", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("show-plan")
    add_remote_common(p)
    p.add_argument("--end-iter", type=int, default=DEFAULT_END_ITER)

    p = sub.add_parser("prepare-split")
    add_remote_common(p)

    p = sub.add_parser("sync-script")
    add_remote_common(p)

    p = sub.add_parser("launch")
    add_remote_common(p)
    p.add_argument("--start-iter", type=int, default=1)
    p.add_argument("--end-iter", type=int, default=DEFAULT_END_ITER)
    p.add_argument("--threads-per", type=int, default=DEFAULT_THREADS_PER)
    p.add_argument("--predict-chunk", type=int, default=DEFAULT_PREDICT_CHUNK)
    p.add_argument("--prepare-split", action="store_true")
    p.add_argument("--no-sync-script", action="store_true")
    p.add_argument(
        "--parallel-configs",
        action="store_true",
        help="Launch one train-local process per model config, each with its own PID/log.",
    )
    p.add_argument("--skip-train-csv", action="store_true")
    p.add_argument("--skip-v1-target-consistency-check", action="store_true")

    p = sub.add_parser("status")
    add_remote_common(p)
    p.add_argument("--tail-lines", type=int, default=30)

    p = sub.add_parser("stop")
    add_remote_common(p)

    p = sub.add_parser("train-local")
    p.add_argument("--role", required=True, choices=sorted(TRAIN_ROLES))
    p.add_argument("--assembly-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-json", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    p.add_argument("--start-iter", type=int, default=1)
    p.add_argument("--end-iter", type=int, default=DEFAULT_END_ITER)
    p.add_argument("--threads-per", type=int, default=DEFAULT_THREADS_PER)
    p.add_argument("--predict-chunk", type=int, default=DEFAULT_PREDICT_CHUNK)
    p.add_argument("--only-configs", default="")
    p.add_argument("--skip-train-csv", action="store_true")
    p.add_argument("--skip-v1-target-consistency-check", action="store_true")
    p.add_argument("--v1-target-consistency-atol", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "show-plan":
        command_show_plan(args)
    elif args.cmd == "prepare-split":
        command_prepare_split(args)
    elif args.cmd == "sync-script":
        command_sync_script(args)
    elif args.cmd == "launch":
        command_launch(args)
    elif args.cmd == "status":
        command_status(args)
    elif args.cmd == "stop":
        command_stop(args)
    elif args.cmd == "train-local":
        command_train_local(args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
