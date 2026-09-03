#!/usr/bin/env python3
"""Generate MCTS visit targets from GV3 roots using native C++ HGB bootstrap.

The heavy path is ``run-local`` and is intended to execute on one bellman
worker. It reads that worker's root dataset, runs native C++ MCTS from each
stored state using a supplied HGB value function, and writes one row per
canonical action with raw visit counts and visit probabilities.

The coordinator ``smoke`` subcommand synchronizes this script and the default
model to the four bellman workers, runs 10 random roots per worker, pulls the
results back locally, and writes combined smoke CSVs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import torch


DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_LOCAL_SMOKE_DIR = (
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/bellman_multiserver_HGB/bellman_prior_smoke_test"
)
DEFAULT_MODEL_LABEL = "worker1_200k_alpha2__hgb_sq_63leaf_1050iter_a2__v47__"
DEFAULT_LOCAL_MODEL_PATH = (
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/bellman_multiserver_HGB/latest_version/"
    "worker1_200k_alpha2/hgb_sq_63leaf_1050iter_a2/Model_Version47/model.joblib"
)
DEFAULT_REMOTE_MODEL_PATH = (
    "{remote_repo}/simulator_output/GV3_Agent/bellman_multiserver_HGB/"
    "policy_target_models/{model_label}/model.joblib"
)
DEFAULT_TARGET_DIR = (
    "{base}/{experiment_name}/mcts_value_function_visit_targets/"
    "{model_label}_iter{mcts_iterations}"
)
DEFAULT_MCTS_ITERATIONS = 10_000
DEFAULT_BATCH_SIZE = 10
DEFAULT_NUM_PROCESSES = 60
DEFAULT_WORKER_THREADS = 1
DEFAULT_UCT_C = 1.4
DEFAULT_SEED = 20260616


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


def _parse_worker_specs(raw: str | None) -> list[WorkerSpec] | None:
    """Parse worker specs as worker=host:server_index:hops_min:hops_max."""

    if raw is None or not str(raw).strip():
        return None
    out: list[WorkerSpec] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid worker spec {item!r}; expected worker=host:server_index:hops_min:hops_max")
        worker_id, rest = [x.strip() for x in item.split("=", 1)]
        parts = [x.strip() for x in rest.split(":")]
        if len(parts) != 4:
            raise ValueError(f"invalid worker spec {item!r}; expected worker=host:server_index:hops_min:hops_max")
        host, server_index, hops_min, hops_max = parts
        out.append(WorkerSpec(worker_id, host, int(server_index), int(hops_min), int(hops_max)))
    if not out:
        raise ValueError("--worker-specs produced no workers")
    return out


@dataclass(frozen=True)
class BatchTask:
    worker_id: str
    host: str
    dataset_dir: str
    output_dir: str
    model_path: str
    model_version: int
    model_label: str
    feature_dim: int
    mcts_iterations: int
    uct_c: float
    seed: int
    worker_threads: int
    root_player_filter: str
    batch_index: int
    state_ids: tuple[int, ...] | None = None
    range_start: int | None = None
    range_end: int | None = None


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vidur" / "Game_Version3_Cpp").is_dir():
            return parent
    return p.parents[3]


def _cpp_dir() -> Path:
    return _repo_root() / "vidur" / "Game_Version3_Cpp"


def _native_module_path() -> Path | None:
    candidates = sorted(_cpp_dir().glob("mcts_native_gv2*.so"))
    return candidates[0] if candidates else None


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


def _worker_root_dir(worker: WorkerSpec, *, remote_repo: str, remote_output_base: str, experiment_name: str) -> str:
    return f"{_base_path(remote_repo, remote_output_base)}/{experiment_name}/{worker.root_dir_name}"


def _target_base_dir(args: argparse.Namespace) -> str:
    base = _base_path(args.remote_repo, args.remote_output_base)
    return str(args.target_dir).format(
        base=base,
        experiment_name=args.experiment_name,
        remote_repo=args.remote_repo.rstrip("/"),
        model_label=args.model_label,
        mcts_iterations=int(args.mcts_iterations),
    )


def _remote_model_path(args: argparse.Namespace) -> str:
    return str(args.remote_model_path).format(
        remote_repo=args.remote_repo.rstrip("/"),
        model_label=args.model_label,
    )


def _limit_native_threads(num_threads: int) -> None:
    threads = max(1, int(num_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=threads)
    except Exception:
        pass
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass


def _import_native_cpp(*, force_build: bool = False) -> Any:
    cpp_dir = _cpp_dir()
    build_dir = cpp_dir / "build"
    existing = _native_module_path()
    if force_build or existing is None:
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "cmake",
                "-S",
                str(cpp_dir),
                "-B",
                str(build_dir),
                f"-DPython_EXECUTABLE={sys.executable}",
            ],
            cwd=str(_repo_root()),
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "-j", str(max(1, min(16, os.cpu_count() or 1)))],
            cwd=str(_repo_root()),
            check=True,
        )
        existing = _native_module_path()
        if existing is None:
            raise FileNotFoundError(f"native module was not produced in {cpp_dir}")

    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))
    return importlib.import_module("mcts_native_gv2")


def _export_hgb_to_native_text(model: Any, out_path: Path) -> Path:
    """Export sklearn HGB internals for the C++ runtime."""

    import numpy as np

    hgb = getattr(model, "hgb", model)
    predictors = list(getattr(hgb, "_predictors", []) or [])
    if not predictors:
        raise TypeError("expected sklearn HistGradientBoostingRegressor with _predictors")
    baseline = float(np.ravel(getattr(hgb, "_baseline_prediction"))[0])
    feature_dim = int(getattr(model, "feature_dim", 226))
    model_tag = str(getattr(model, "model_tag", "v4_adv_hgb"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write("hgb226_v1\n")
        f.write(f"model_tag\t{model_tag}\n")
        f.write(f"feature_dim\t{feature_dim}\n")
        f.write(f"baseline\t{baseline:.17g}\n")
        for tree_idx, pred in enumerate(predictors):
            tree = pred[0]
            nodes = tree.nodes
            if "is_categorical" in nodes.dtype.names and bool(np.any(nodes["is_categorical"])):
                raise RuntimeError("native HGB export does not support categorical splits")
            f.write(f"tree\t{tree_idx}\t{len(nodes)}\n")
            for node in nodes:
                f.write(
                    "node"
                    f"\t{float(node['value']):.17g}"
                    f"\t{int(node['feature_idx'])}"
                    f"\t{float(node['num_threshold']):.17g}"
                    f"\t{1 if bool(node['missing_go_to_left']) else 0}"
                    f"\t{int(node['left'])}"
                    f"\t{int(node['right'])}"
                    f"\t{1 if bool(node['is_leaf']) else 0}\n"
                )
            f.write("end_tree\n")
    tmp.replace(out_path)
    return out_path


def _load_hgb_wrapper_or_wrap(model_path: Path, *, output_dir: Path, feature_dim: int) -> tuple[Any, Path]:
    model = joblib.load(model_path)
    if callable(getattr(model, "infer_from_inputs", None)):
        return model, model_path

    from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper

    wrapped = V4AdvHGBWrapper(model, feature_dim=int(feature_dim), model_tag=f"wrapped:{model_path.name}")
    wrapper_path = output_dir / "_harness_model" / f"v4_adv_hgb_wrapper_pid{os.getpid()}.joblib"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapped, wrapper_path, compress=3)
    return wrapped, wrapper_path


def _import_tester_runner() -> Any:
    try:
        from vidur.bellman_v4_adv_2000k_multiprocess import runner as tester_runner

        return tester_runner
    except Exception:
        from vidur.Game_Version3.Model_Tester import runner as tester_runner

        return tester_runner

def _prepare_bundle_and_runtime(task: BatchTask) -> tuple[Any, Any, Any, dict[str, Any]]:
    from dataclasses import replace

    from vidur.Game_Version3.DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
    from vidur.Game_Version3.Model_Tester.config import DEFAULT_MODEL_TESTER_CONFIG
    tester_runner = _import_tester_runner()

    output_dir = Path(task.output_dir).expanduser()
    model_path = Path(task.model_path).expanduser()
    model, harness_model_path = _load_hgb_wrapper_or_wrap(
        model_path,
        output_dir=output_dir,
        feature_dim=int(task.feature_dim),
    )

    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind="classical_joblib",
        model_checkpoint_path=str(harness_model_path),
        output_dir=str(output_dir / "_harness"),
        bootstrap_model_version=int(task.model_version),
    )
    bundle = tester_runner._build_bundle(cfg)

    native = _import_native_cpp(force_build=False)
    export_path = output_dir / "_native_model" / f"{task.model_label}_batch{int(task.batch_index):06d}_pid{os.getpid()}_native_export.tsv"
    export = _export_hgb_to_native_text(model, export_path)
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(export))

    pipeline_cfg = cfg.to_pipeline_cfg()
    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, bundle.simulator)
    payload["use_model_bootstrap"] = bool(int(task.model_version) > 0)
    payload["native_search_mode"] = "full_tree"
    payload["root_dirichlet_noise_enabled"] = False
    payload["pb_c_base"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(task.uct_c)
    payload["max_forced_hops"] = int(getattr(pipeline_cfg, "max_forced_hops_per_root", 0) or 0)
    return cfg, bundle, runtime, payload


def _load_samples_for_task(task: BatchTask) -> list[tuple[int, dict[str, Any]]]:
    from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import load_samples

    if task.state_ids is not None:
        wanted = {int(x) for x in task.state_ids}
        out: list[tuple[int, dict[str, Any]]] = []
        max_id = max(wanted) if wanted else -1
        for sid, record in load_samples(
            task.dataset_dir,
            root_player_filter=str(task.root_player_filter),
            parent_state_id_start=0,
            parent_state_id_end=max_id + 1,
        ):
            if int(sid) in wanted:
                out.append((int(sid), record))
                if len(out) == len(wanted):
                    break
        missing = wanted.difference(int(sid) for sid, _ in out)
        if missing:
            raise RuntimeError(f"missing requested state ids: {sorted(missing)[:20]}")
        return sorted(out, key=lambda x: int(x[0]))

    if task.range_start is None or task.range_end is None:
        raise ValueError("BatchTask needs either state_ids or range_start/range_end")
    return [
        (int(sid), record)
        for sid, record in load_samples(
            task.dataset_dir,
            root_player_filter=str(task.root_player_filter),
            parent_state_id_start=int(task.range_start),
            parent_state_id_end=int(task.range_end),
        )
    ]


def _state_from_record(env: Any, record: dict[str, Any]) -> Any:
    clone_fn = getattr(env, "clone_state_from_snapshot", None)
    if callable(clone_fn):
        return clone_fn(record["simulator_snapshot"], record["stats"])
    state = env.initial_state()
    state.simulator.restore_state(record["simulator_snapshot"])
    state.stats = record["stats"].clone()
    return state


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _run_mcts_for_state(
    *,
    task: BatchTask,
    native: Any,
    bundle: Any,
    runtime: Any,
    payload: dict[str, Any],
    state_id: int,
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    tester_runner = _import_tester_runner()

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

    valid_indices = [int(x) for x in list(expanded.valid_indices)]
    if valid_indices:
        alias_to_canon, _canon_to_aliases, canonical_indices = bundle.mcts._canonicalize_action_indices(
            player=str(player),
            actions_by_index=list(expanded.actions_by_index),
            valid_indices=valid_indices,
        )
        canonical_indices = [int(x) for x in sorted(set(int(x) for x in canonical_indices))]
    else:
        alias_to_canon = {}
        canonical_indices = []

    root_id = int(record.get("root_id", state_id) or state_id)
    native_root_id = int(root_id % 200_000_000)
    native_seed = int(task.seed) + int(state_id) + (0 if str(player) == "controller" else 1_000_000)
    t0 = time.perf_counter()
    native_out = native.search_mcts_hgb226(
        runtime,
        int(task.model_version),
        nlt._native_state_payload(bundle.env, expanded.search_state),
        dict(payload),
        int(task.mcts_iterations),
        str(player),
        int(native_root_id),
        int(root_depth),
        int(native_root_id),
        int(native_root_id),
        int(native_seed),
        False,
        False,
        "",
        "",
    )
    elapsed = time.perf_counter() - t0

    native_alias_to_canon = {
        int(k): int(v)
        for k, v in dict(native_out.get("action_alias_to_canonical", {}) or {}).items()
    }
    if native_alias_to_canon:
        canonical_indices = sorted(
            {int(native_alias_to_canon.get(i, alias_to_canon.get(i, i))) for i in valid_indices}
        )

    visits_by_idx = {
        int(child.get("index", -1)): int(child.get("visits", 0) or 0)
        for child in list(native_out.get("children", []) or [])
        if int(child.get("index", -1)) >= 0
    }
    if len(canonical_indices) == 1 and visits_by_idx.get(int(canonical_indices[0]), 0) <= 0:
        visits_by_idx[int(canonical_indices[0])] = int(task.mcts_iterations)
    total_visits = int(sum(int(visits_by_idx.get(int(idx), 0)) for idx in canonical_indices))
    denom = float(total_visits) if total_visits > 0 else 1.0

    rows: list[dict[str, Any]] = []
    for canon_idx in canonical_indices:
        visits = int(visits_by_idx.get(int(canon_idx), 0))
        action = expanded.actions_by_index[int(canon_idx)] if 0 <= int(canon_idx) < len(expanded.actions_by_index) else None
        rows.append(
            {
                "worker": task.worker_id,
                "host": task.host,
                "state_id": int(state_id),
                "root_id": int(root_id),
                "player": str(player),
                "root_player": str(root_player),
                "root_depth": int(root_depth),
                "history_hops": int(record.get("history_hops", -1) or -1),
                "canon_action_index": int(canon_idx),
                "visit_count": int(visits),
                "visit_prob": float(visits / denom),
                "total_child_visits": int(total_visits),
                "mcts_iterations": int(task.mcts_iterations),
                "valid_action_count": int(len(valid_indices)),
                "canonical_action_count": int(len(canonical_indices)),
                "action_repr": repr(action),
            }
        )

    root_visits = int(native_out.get("root_visits", 0) or 0)
    root_value_sum = _finite_float(native_out.get("root_value_sum", 0.0), 0.0)
    root_value = float(root_value_sum / float(root_visits)) if root_visits > 0 else 0.0

    time_row = {
        "worker": task.worker_id,
        "host": task.host,
        "state_id": int(state_id),
        "root_id": int(root_id),
        "player": str(player),
        "root_player": str(root_player),
        "valid_action_count": int(len(valid_indices)),
        "canonical_action_count": int(len(canonical_indices)),
        "mcts_iterations": int(task.mcts_iterations),
        "total_child_visits": int(total_visits),
        "time_for_generation_s": float(elapsed),
        "root_value": float(root_value),
        "root_visits": int(root_visits),
    }
    return rows, time_row


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


TARGET_FIELDS = [
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
    "visit_prob",
    "total_child_visits",
    "mcts_iterations",
    "valid_action_count",
    "canonical_action_count",
    "action_repr",
]

TIME_FIELDS = [
    "worker",
    "host",
    "state_id",
    "root_id",
    "player",
    "root_player",
    "valid_action_count",
    "canonical_action_count",
    "mcts_iterations",
    "total_child_visits",
    "time_for_generation_s",
    "root_value",
    "root_visits",
]


def _run_batch(task: BatchTask) -> dict[str, Any]:
    _limit_native_threads(int(task.worker_threads))
    out_dir = Path(task.output_dir).expanduser()
    partial_dir = out_dir / "partials"
    target_path = partial_dir / f"targets_batch_{int(task.batch_index):06d}.csv"
    time_path = partial_dir / f"times_batch_{int(task.batch_index):06d}.csv"
    err_path = partial_dir / f"errors_batch_{int(task.batch_index):06d}.jsonl"
    done_path = partial_dir / f"done_batch_{int(task.batch_index):06d}.json"
    if done_path.exists() and target_path.exists() and time_path.exists():
        return json.loads(done_path.read_text(encoding="utf-8"))

    native = _import_native_cpp(force_build=False)
    _cfg, bundle, runtime, payload = _prepare_bundle_and_runtime(task)
    samples = _load_samples_for_task(task)

    target_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for state_id, record in samples:
        try:
            rows, time_row = _run_mcts_for_state(
                task=task,
                native=native,
                bundle=bundle,
                runtime=runtime,
                payload=payload,
                state_id=int(state_id),
                record=record,
            )
            target_rows.extend(rows)
            time_rows.append(time_row)
        except Exception as exc:
            errors.append(
                {
                    "worker": task.worker_id,
                    "host": task.host,
                    "batch_index": int(task.batch_index),
                    "state_id": int(state_id),
                    "error": repr(exc),
                }
            )

    _write_csv(target_path, TARGET_FIELDS, target_rows)
    _write_csv(time_path, TIME_FIELDS, time_rows)
    if errors:
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with err_path.open("w", encoding="utf-8") as f:
            for row in errors:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "worker": task.worker_id,
        "host": task.host,
        "batch_index": int(task.batch_index),
        "num_states": int(len(samples)),
        "num_success": int(len(time_rows)),
        "num_errors": int(len(errors)),
        "num_target_rows": int(len(target_rows)),
        "target_path": str(target_path),
        "time_path": str(time_path),
        "error_path": str(err_path) if errors else "",
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    try:
        bundle.mcts.clear_search_state(drop_scratch=True)
    except Exception:
        pass
    gc.collect()
    return summary


def _count_samples(dataset_dir: str, *, root_player_filter: str, max_roots: int | None = None) -> int:
    from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import count_samples

    return int(count_samples(dataset_dir, root_player_filter=root_player_filter, max_roots=max_roots))


def _split_batches(ids: list[int], batch_size: int) -> list[tuple[int, ...]]:
    batch_size = max(1, int(batch_size))
    return [tuple(int(x) for x in ids[i : i + batch_size]) for i in range(0, len(ids), batch_size)]


def _parse_state_ids(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out: list[int] = []
    for part in str(raw).replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _load_smoke_state_ids_by_worker(path: str | None) -> dict[str, list[int]]:
    if path is None or str(path).strip() == "":
        return {}
    csv_path = Path(path).expanduser()
    out: dict[str, set[int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            worker = str(row.get("worker", "")).strip()
            if not worker:
                continue
            state_id = int(row["state_id"])
            out.setdefault(worker, set()).add(state_id)
    return {worker: sorted(ids) for worker, ids in out.items()}


def _merge_local_outputs(output_dir: Path) -> dict[str, Any]:
    partial_dir = output_dir / "partials"
    target_files = sorted(partial_dir.glob("targets_batch_*.csv"))
    time_files = sorted(partial_dir.glob("times_batch_*.csv"))
    error_files = sorted(partial_dir.glob("errors_batch_*.jsonl"))

    target_rows: list[dict[str, Any]] = []
    for path in target_files:
        with path.open("r", encoding="utf-8", newline="") as f:
            target_rows.extend(dict(row) for row in csv.DictReader(f))
    target_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"]), int(r["canon_action_index"])))

    time_rows: list[dict[str, Any]] = []
    for path in time_files:
        with path.open("r", encoding="utf-8", newline="") as f:
            time_rows.extend(dict(row) for row in csv.DictReader(f))
    time_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"])))

    _write_csv(output_dir / "targets.csv", TARGET_FIELDS, target_rows)
    _write_csv(output_dir / "target_time.csv", TIME_FIELDS, time_rows)

    errors = 0
    for path in error_files:
        with path.open("r", encoding="utf-8") as f:
            errors += sum(1 for _ in f)
    manifest = {
        "output_dir": str(output_dir),
        "num_target_rows": int(len(target_rows)),
        "num_time_rows": int(len(time_rows)),
        "num_error_rows": int(errors),
        "partial_target_files": int(len(target_files)),
        "partial_time_files": int(len(time_files)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def cmd_run_local(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    total = _count_samples(args.dataset_dir, root_player_filter=args.root_player_filter)
    rng = random.Random(int(args.seed) + int(args.worker_index))
    explicit_state_ids = _parse_state_ids(getattr(args, "state_ids", None))

    if explicit_state_ids is not None:
        batches = [
            BatchTask(
                worker_id=args.worker_id,
                host=args.host,
                dataset_dir=args.dataset_dir,
                output_dir=str(output_dir),
                model_path=args.model_path,
                model_version=int(args.model_version),
                model_label=args.model_label,
                feature_dim=int(args.feature_dim),
                mcts_iterations=int(args.mcts_iterations),
                uct_c=float(args.uct_c),
                seed=int(args.seed),
                worker_threads=int(args.worker_threads),
                root_player_filter=args.root_player_filter,
                batch_index=i,
                state_ids=batch,
            )
            for i, batch in enumerate(_split_batches(explicit_state_ids, int(args.batch_size)))
        ]
    elif args.max_states is not None:
        n = min(int(args.max_states), total)
        selected = sorted(rng.sample(range(total), n)) if n < total else list(range(total))
        batches = [
            BatchTask(
                worker_id=args.worker_id,
                host=args.host,
                dataset_dir=args.dataset_dir,
                output_dir=str(output_dir),
                model_path=args.model_path,
                model_version=int(args.model_version),
                model_label=args.model_label,
                feature_dim=int(args.feature_dim),
                mcts_iterations=int(args.mcts_iterations),
                uct_c=float(args.uct_c),
                seed=int(args.seed),
                worker_threads=int(args.worker_threads),
                root_player_filter=args.root_player_filter,
                batch_index=i,
                state_ids=batch,
            )
            for i, batch in enumerate(_split_batches(selected, int(args.batch_size)))
        ]
    else:
        batches = [
            BatchTask(
                worker_id=args.worker_id,
                host=args.host,
                dataset_dir=args.dataset_dir,
                output_dir=str(output_dir),
                model_path=args.model_path,
                model_version=int(args.model_version),
                model_label=args.model_label,
                feature_dim=int(args.feature_dim),
                mcts_iterations=int(args.mcts_iterations),
                uct_c=float(args.uct_c),
                seed=int(args.seed),
                worker_threads=int(args.worker_threads),
                root_player_filter=args.root_player_filter,
                batch_index=i,
                range_start=start,
                range_end=min(total, start + int(args.batch_size)),
            )
            for i, start in enumerate(range(0, total, int(args.batch_size)))
        ]

    run_manifest = {
        "worker_id": args.worker_id,
        "host": args.host,
        "dataset_dir": args.dataset_dir,
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "model_label": args.model_label,
        "model_version": int(args.model_version),
        "total_available_states": int(total),
        "explicit_state_ids": explicit_state_ids,
        "max_states": None if args.max_states is None else int(args.max_states),
        "batch_size": int(args.batch_size),
        "num_batches": int(len(batches)),
        "num_processes": int(args.num_processes),
        "mcts_iterations": int(args.mcts_iterations),
        "uct_c": float(args.uct_c),
        "seed": int(args.seed),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not batches:
        _merge_local_outputs(output_dir)
        return

    ctx = mp.get_context("spawn")
    nproc = max(1, min(int(args.num_processes), len(batches)))
    summaries: list[dict[str, Any]] = []
    with ctx.Pool(processes=nproc, maxtasksperchild=1) as pool:
        for summary in pool.imap_unordered(_run_batch, batches):
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)

    manifest = _merge_local_outputs(output_dir)
    manifest["batch_summaries"] = summaries
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_combined_smoke(local_dir: Path, worker_dirs: list[tuple[WorkerSpec, Path]]) -> dict[str, int]:
    target_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    for worker, path in worker_dirs:
        target_path = path / "targets.csv"
        time_path = path / "target_time.csv"
        if target_path.exists():
            with target_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row["worker"] = row.get("worker") or worker.worker_id
                    row["host"] = row.get("host") or worker.host
                    target_rows.append(row)
        if time_path.exists():
            with time_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row["worker"] = row.get("worker") or worker.worker_id
                    row["host"] = row.get("host") or worker.host
                    time_rows.append(row)

    target_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"]), int(r["canon_action_index"])))
    time_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"])))

    smoke_fields = ["worker", "host", "state_id", "player", "canon_action_index", "visit_count", "visit_prob"]
    time_fields = ["worker", "host", "state_id", "player", "time_for_generation_s"]
    _write_csv(local_dir / "target_smoke.csv", smoke_fields, target_rows)
    _write_csv(local_dir / "target_time.csv", time_fields, time_rows)
    _write_csv(local_dir / "target_smoke_full.csv", TARGET_FIELDS, target_rows)
    _write_csv(local_dir / "target_time_full.csv", TIME_FIELDS, time_rows)
    summary = {"target_rows": int(len(target_rows)), "time_rows": int(len(time_rows))}
    (local_dir / "combined_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def cmd_smoke(args: argparse.Namespace) -> None:
    local_dir = Path(args.local_smoke_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_model_path = _remote_model_path(args)
    model_src = Path(args.local_model_path).expanduser()
    if not model_src.exists():
        raise FileNotFoundError(f"default model artifact not found: {model_src}")

    script_path = Path(__file__).resolve()
    target_base = _target_base_dir(args)
    custom_workers = _parse_worker_specs(getattr(args, "worker_specs", None))
    if custom_workers is not None:
        workers = custom_workers
    else:
        selected_workers = {x.strip() for x in args.workers.split(",") if x.strip()}
        workers = [w for w in WORKERS if w.worker_id in selected_workers]
    if not workers:
        raise SystemExit("no workers selected")
    smoke_state_ids_by_worker = _load_smoke_state_ids_by_worker(getattr(args, "smoke_state_ids_csv", None))

    for worker in workers:
        remote_script = f"{args.remote_repo.rstrip('/')}/{script_path.relative_to(_repo_root())}"
        _ssh(worker.host, f"mkdir -p {shlex.quote(str(Path(remote_script).parent))} {shlex.quote(str(Path(remote_model_path).parent))}")
        _rsync(str(script_path), f"{worker.host}:{remote_script}")
        _rsync(str(model_src), f"{worker.host}:{remote_model_path}")

    procs: list[tuple[WorkerSpec, subprocess.Popen[str], str]] = []
    for worker in workers:
        dataset_dir = _worker_root_dir(
            worker,
            remote_repo=args.remote_repo,
            remote_output_base=args.remote_output_base,
            experiment_name=args.experiment_name,
        )
        out_dir = f"{target_base}/smoke_{worker.worker_id}_seed{int(args.seed)}_n{int(args.smoke_states)}"
        explicit_ids = smoke_state_ids_by_worker.get(worker.worker_id)
        state_selection_arg = (
            f"--state-ids {shlex.quote(','.join(str(x) for x in explicit_ids))} "
            if explicit_ids is not None
            else f"--max-states {int(args.smoke_states)} "
        )
        cmd = (
            f"cd {shlex.quote(args.remote_repo)} && "
            f"{shlex.quote(args.remote_repo.rstrip('/') + '/.venv/bin/python3')} "
            f"-m vidur.bellman_v4_adv_2000k_multiprocess.multi_server_hgb_Mcts_value_function_targets "
            f"run-local "
            f"--worker-id {shlex.quote(worker.worker_id)} "
            f"--host {shlex.quote(worker.host)} "
            f"--worker-index {int(worker.server_index)} "
            f"--dataset-dir {shlex.quote(dataset_dir)} "
            f"--output-dir {shlex.quote(out_dir)} "
            f"--model-path {shlex.quote(remote_model_path)} "
            f"--model-label {shlex.quote(args.model_label)} "
            f"--model-version {int(args.model_version)} "
            f"--feature-dim {int(args.feature_dim)} "
            f"--mcts-iterations {int(args.mcts_iterations)} "
            f"--uct-c {float(args.uct_c)} "
            f"--seed {int(args.seed)} "
            f"{state_selection_arg}"
            f"--batch-size {int(args.batch_size)} "
            f"--num-processes {int(args.smoke_processes)} "
            f"--worker-threads {int(args.worker_threads)} "
            f"--root-player-filter {shlex.quote(args.root_player_filter)}"
        )
        print(f"[launch] {worker.host}: {cmd}", flush=True)
        p = subprocess.Popen(
            ["ssh", worker.host, cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append((worker, p, out_dir))

    failed: list[str] = []
    for worker, proc, _out_dir in procs:
        stdout, _ = proc.communicate()
        log_path = local_dir / f"{worker.worker_id}_remote_stdout.log"
        log_path.write_text(stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            failed.append(f"{worker.worker_id}:{proc.returncode}")
            print(stdout or "", flush=True)
    if failed:
        raise RuntimeError(f"smoke generation failed on {failed}")

    pulled: list[tuple[WorkerSpec, Path]] = []
    for worker, _proc, out_dir in procs:
        worker_local = local_dir / worker.worker_id
        worker_local.mkdir(parents=True, exist_ok=True)
        _rsync(f"{worker.host}:{out_dir.rstrip('/')}/", f"{worker_local}/")
        pulled.append((worker, worker_local))

    summary = _write_combined_smoke(local_dir, pulled)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def cmd_status(args: argparse.Namespace) -> None:
    target_base = _target_base_dir(args)
    for worker in WORKERS:
        out_dir = f"{target_base}/{worker.worker_id}"
        cmd = (
            f"if [ -f {shlex.quote(out_dir + '/manifest.json')} ]; then "
            f"cat {shlex.quote(out_dir + '/manifest.json')}; "
            f"else echo '{{\"worker\":\"{worker.worker_id}\",\"status\":\"missing\"}}'; fi"
        )
        result = _ssh(worker.host, cmd, check=False)
        print(f"--- {worker.worker_id} {worker.host} ---")
        print(result.stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    local = sub.add_parser("run-local", help="Run target generation on one worker host.")
    local.add_argument("--worker-id", required=True)
    local.add_argument("--host", required=True)
    local.add_argument("--worker-index", type=int, default=0)
    local.add_argument("--dataset-dir", required=True)
    local.add_argument("--output-dir", required=True)
    local.add_argument("--model-path", required=True)
    local.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    local.add_argument("--model-version", type=int, default=47)
    local.add_argument("--feature-dim", type=int, default=226)
    local.add_argument("--mcts-iterations", type=int, default=DEFAULT_MCTS_ITERATIONS)
    local.add_argument("--uct-c", type=float, default=DEFAULT_UCT_C)
    local.add_argument("--seed", type=int, default=DEFAULT_SEED)
    local.add_argument("--max-states", type=int, default=None)
    local.add_argument("--state-ids", default=None, help="Comma-separated explicit parent state IDs to process.")
    local.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    local.add_argument("--num-processes", type=int, default=DEFAULT_NUM_PROCESSES)
    local.add_argument("--worker-threads", type=int, default=DEFAULT_WORKER_THREADS)
    local.add_argument("--root-player-filter", choices=("controller", "adversary", "any"), default="controller")
    local.set_defaults(func=cmd_run_local)

    smoke = sub.add_parser("smoke", help="Run 10-state target-generation smoke on all bellman workers and pull outputs.")
    smoke.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    smoke.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    smoke.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    smoke.add_argument("--target-dir", default=DEFAULT_TARGET_DIR)
    smoke.add_argument("--local-smoke-dir", default=DEFAULT_LOCAL_SMOKE_DIR)
    smoke.add_argument("--local-model-path", default=DEFAULT_LOCAL_MODEL_PATH)
    smoke.add_argument("--remote-model-path", default=DEFAULT_REMOTE_MODEL_PATH)
    smoke.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    smoke.add_argument("--model-version", type=int, default=47)
    smoke.add_argument("--feature-dim", type=int, default=226)
    smoke.add_argument("--mcts-iterations", type=int, default=DEFAULT_MCTS_ITERATIONS)
    smoke.add_argument("--uct-c", type=float, default=DEFAULT_UCT_C)
    smoke.add_argument("--seed", type=int, default=DEFAULT_SEED)
    smoke.add_argument("--smoke-states", type=int, default=10)
    smoke.add_argument(
        "--smoke-state-ids-csv",
        default=None,
        help="Existing target_smoke_full.csv to reuse exact per-worker state IDs.",
    )
    smoke.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    smoke.add_argument("--smoke-processes", type=int, default=1)
    smoke.add_argument("--worker-threads", type=int, default=DEFAULT_WORKER_THREADS)
    smoke.add_argument("--root-player-filter", choices=("controller", "adversary", "any"), default="controller")
    smoke.add_argument("--workers", default="worker1,worker2,worker3,worker4")
    smoke.add_argument(
        "--worker-specs",
        default=None,
        help="Comma-separated custom workers as worker=host:server_index:hops_min:hops_max.",
    )
    smoke.set_defaults(func=cmd_smoke)

    status = sub.add_parser("status", help="Print remote manifest status for full target dirs.")
    status.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    status.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    status.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    status.add_argument("--target-dir", default=DEFAULT_TARGET_DIR)
    status.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    status.add_argument("--mcts-iterations", type=int, default=DEFAULT_MCTS_ITERATIONS)
    status.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
