"""Bellman convergence harness for GV3 ModelSearchBed experiments.

This script trains value models across Bellman depths:

    V1 learns T[V0], where V0 is zero bootstrap and T[V0] is immediate reward.
    V2 learns T[V1], where targets are reward + discount * V1(child).
    ...

It also optionally measures same-model Bellman residuals:

    Vi(s) vs T[Vi](s)

The root dataset is treated as read-only. This file never generates roots.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from ..DNN import infer as infer_module
from . import logger as bellman_logger
from .self_model_test import (
    RootStateLoader,
    SelfModelTestConfig,
    build_self_model_test_config,
    count_trainable_parameters,
    load_root_records,
    predict_candidate_values,
    train_candidate_model,
    write_training_metrics,
)


def _current_rss_mib() -> float | None:
    """Return current parent-process RSS in MiB when running on Linux."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _log_process_memory(label: str) -> None:
    """Emit compact memory diagnostics at version boundaries."""

    rss_mib = _current_rss_mib()
    cuda_allocated = 0.0
    cuda_reserved = 0.0
    if torch.cuda.is_available():
        try:
            cuda_allocated = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            cuda_reserved = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
        except Exception:
            cuda_allocated = 0.0
            cuda_reserved = 0.0
    rss_text = "unknown" if rss_mib is None else f"{rss_mib:.1f}"
    print(
        "[bellman_convergence] memory "
        f"{label}: parent_rss_mib={rss_text} "
        f"cuda_allocated_mib={cuda_allocated:.1f} "
        f"cuda_reserved_mib={cuda_reserved:.1f}",
        flush=True,
    )


def _release_iteration_memory(*, model: Any | None = None, label: str) -> None:
    """Drop large per-version caches that are not needed for bootstrap."""

    if model is not None and hasattr(model, "_model_search_cached_features"):
        try:
            model._model_search_cached_features = {}
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _log_process_memory(label)


@dataclass(frozen=True)
class BellmanConvergenceConfig:
    """Configuration for iterative Bellman target/training runs."""

    dataset_dir: Path
    output_dir: Path
    num_versions: int = 5
    num_roots: int = 10_000
    eval_ratio: float = 0.20
    split_seed: int = 12345
    seed: int = 2027
    batch_size: int = 256
    root_player_filter: str = "controller"
    model_name_prefix: str = "bellman_model"
    abs_error_threshold: float = 1.0
    reuse_record_targets_for_v0: bool = True
    cache_targets: bool = True
    same_model_analysis_start_version: int = 2
    extra_config: dict[str, Any] | None = None


class BootstrapModelAdapter:
    """Adapter used by mctsDNN bootstrap calls.

    mctsDNN expects `infer_from_inputs(inputs, player, device=...)`. The
    existing GV3 model already implements that method. Custom ModelSearchBed
    experiments can either implement the same method on their model object or
    expose `DNN/infer.py::predict_model_search_value_from_inputs(...)`.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: torch.device | None = None,
    ) -> tuple[float, list[float]]:
        infer_fn = getattr(self.model, "infer_from_inputs", None)
        if callable(infer_fn):
            return infer_fn(inputs, player, device=device)

        hook = getattr(infer_module, "predict_model_search_value_from_inputs", None)
        if callable(hook):
            value = hook(
                model=self.model,
                inputs=inputs,
                player=player,
                device=device,
            )
            return float(value), []

        raise TypeError(
            "Bootstrap model must implement infer_from_inputs(...) or "
            "DNN/infer.py must define predict_model_search_value_from_inputs(...)."
        )


@dataclass(frozen=True)
class TargetWorkerTask:
    """One independent target-computation task for a worker process."""

    worker_id: int
    cfg: BellmanConvergenceConfig
    records_with_positions: list[tuple[int, dict[str, Any]]]
    bootstrap_model: Any
    split_name: str


@dataclass(frozen=True)
class TargetWorkerResult:
    """Target values produced by one worker, keyed by parent split position."""

    worker_id: int
    split_name: str
    shard_path: str
    target_count: int


@dataclass(frozen=True)
class TransitionCacheWorkerTask:
    """One independent fixed-transition cache build task."""

    worker_id: int
    cfg: BellmanConvergenceConfig
    records_with_positions: list[tuple[int, dict[str, Any]]]
    split_name: str


@dataclass(frozen=True)
class TransitionCacheWorkerResult:
    """Transition-cache shards produced by one worker."""

    worker_id: int
    split_name: str
    shard_paths: list[str]
    empty_root_positions: list[int]
    root_count: int
    child_count: int


def build_bellman_config(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    num_versions: int = 5,
    num_roots: int = 10_000,
    eval_ratio: float = 0.20,
    split_seed: int = 12345,
    seed: int = 2027,
    batch_size: int = 256,
    root_player_filter: str = "controller",
    model_name_prefix: str = "bellman_model",
    abs_error_threshold: float = 1.0,
    reuse_record_targets_for_v0: bool = True,
    cache_targets: bool = True,
    same_model_analysis_start_version: int = 2,
    extra_config: dict[str, Any] | None = None,
) -> BellmanConvergenceConfig:
    """Validate and build a Bellman convergence config."""

    if int(num_versions) <= 0:
        raise ValueError("num_versions must be > 0")
    if int(num_roots) <= 1:
        raise ValueError("num_roots must be > 1")
    if not (0.0 < float(eval_ratio) < 1.0):
        raise ValueError("eval_ratio must be in (0, 1)")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if str(root_player_filter) not in {"controller", "adversary", "any"}:
        raise ValueError("root_player_filter must be 'controller', 'adversary', or 'any'")
    if float(abs_error_threshold) < 0.0:
        raise ValueError("abs_error_threshold must be >= 0")
    if int(same_model_analysis_start_version) <= 0:
        raise ValueError("same_model_analysis_start_version must be > 0")
    if extra_config is not None and not isinstance(extra_config, dict):
        raise TypeError("extra_config must be a dict when provided")

    return BellmanConvergenceConfig(
        dataset_dir=Path(dataset_dir).expanduser(),
        output_dir=Path(output_dir).expanduser(),
        num_versions=int(num_versions),
        num_roots=int(num_roots),
        eval_ratio=float(eval_ratio),
        split_seed=int(split_seed),
        seed=int(seed),
        batch_size=int(batch_size),
        root_player_filter=str(root_player_filter),
        model_name_prefix=str(model_name_prefix),
        abs_error_threshold=float(abs_error_threshold),
        reuse_record_targets_for_v0=bool(reuse_record_targets_for_v0),
        cache_targets=bool(cache_targets),
        same_model_analysis_start_version=int(same_model_analysis_start_version),
        extra_config=dict(extra_config or {}),
    )


def make_self_model_config(
    cfg: BellmanConvergenceConfig,
    *,
    output_dir: Path,
    model_name: str,
) -> SelfModelTestConfig:
    """Build the ModelSearchBed trainer/infer config for one Bellman version."""

    return build_self_model_test_config(
        dataset_dir=cfg.dataset_dir,
        output_dir=output_dir,
        num_roots=int(cfg.num_roots),
        max_candidate_roots=int(cfg.num_roots),
        seed=int(cfg.seed),
        root_player_filter=str(cfg.root_player_filter),
        allow_dataset_generation=False,
        eval_ratio=float(cfg.eval_ratio),
        split_seed=int(cfg.split_seed),
        model_name=str(model_name),
        batch_size=int(cfg.batch_size),
        extra_config=dict(cfg.extra_config or {}),
    )


def split_records_with_indices(
    records: Sequence[dict[str, Any]],
    *,
    eval_ratio: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    """Split records deterministically and keep original sample indices."""

    if not records:
        raise ValueError("cannot split an empty record list")
    indices = list(range(len(records)))
    random.Random(int(split_seed)).shuffle(indices)

    eval_count = int(round(float(len(indices)) * float(eval_ratio)))
    eval_count = max(1, min(len(indices) - 1, eval_count)) if len(indices) > 1 else 0
    eval_indices = sorted(indices[:eval_count])
    eval_index_set = set(eval_indices)
    train_indices = [idx for idx in range(len(records)) if idx not in eval_index_set]

    train_records = [dict(records[idx]) for idx in train_indices]
    eval_records = [dict(records[idx]) for idx in eval_indices]
    if not train_records or not eval_records:
        raise ValueError("train/eval split produced an empty split")
    return train_records, eval_records, train_indices, eval_indices


def records_with_targets(
    records: Sequence[dict[str, Any]],
    targets: Sequence[float],
) -> list[dict[str, Any]]:
    """Return shallow record copies with version-specific target values."""

    if len(records) != len(targets):
        raise ValueError(f"records/targets length mismatch: {len(records)} != {len(targets)}")
    out: list[dict[str, Any]] = []
    for record, target in zip(records, targets):
        row = dict(record)
        row["target_value"] = float(target)
        out.append(row)
    return out


def target_cache_path(
    output_dir: Path,
    *,
    model_version: int,
    bootstrap_version: int,
    split_name: str,
) -> Path:
    """Path for cached MCTS target values."""

    return (
        output_dir
        / f"Model_Version{int(model_version)}"
        / f"{split_name}_targets_from_model_{int(bootstrap_version)}.pt"
    )


def load_cached_targets(path: Path, *, expected_count: int) -> list[float] | None:
    """Load cached targets if present and shape-compatible."""

    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    targets = payload.get("targets") if isinstance(payload, dict) else payload
    targets = [float(x) for x in targets]
    if len(targets) != int(expected_count):
        return None
    return targets


def save_cached_targets(
    path: Path,
    *,
    targets: Sequence[float],
    metadata: dict[str, Any],
) -> Path:
    """Write MCTS target cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "targets": [float(x) for x in targets],
            "metadata": dict(metadata),
        },
        path,
    )
    return path


def _snapshot_for_model_search_record(state: Any) -> dict[str, Any]:
    sim = state.simulator
    if hasattr(sim, "snapshot_state"):
        simulator_snapshot = sim.snapshot_state()
    else:
        simulator_snapshot = sim.snapshot_state_fast()
    return {
        "simulator_snapshot": simulator_snapshot,
        "stats": state.stats.clone(),
        "root_player": "controller",
        "root_depth": 0,
        "history_hops": 0,
        "target_value": 0.0,
    }


def _transition_cache_enabled(cfg: BellmanConvergenceConfig) -> bool:
    return bool((cfg.extra_config or {}).get("use_transition_cache", False))


def _base_transition_split_name(split_name: str) -> str:
    split = str(split_name)
    if split.startswith("train"):
        return "train"
    if split.startswith("eval"):
        return "eval"
    return split


def _transition_cache_dir(cfg: BellmanConvergenceConfig) -> Path:
    extra = dict(cfg.extra_config or {})
    raw = extra.get("transition_cache_dir", None)
    if raw:
        return Path(str(raw)).expanduser()
    return cfg.output_dir / "_transition_cache"


def _transition_cache_metadata_path(cfg: BellmanConvergenceConfig, split_name: str) -> Path:
    return _transition_cache_dir(cfg) / _base_transition_split_name(split_name) / "metadata.json"


def _transition_cache_feature_dtype(cfg: BellmanConvergenceConfig) -> torch.dtype:
    raw = str((cfg.extra_config or {}).get("transition_cache_feature_dtype", "float32")).lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32", "single"}:
        return torch.float32
    raise ValueError(f"unsupported transition_cache_feature_dtype: {raw!r}")


def _transition_cache_num_processes(
    cfg: BellmanConvergenceConfig,
    *,
    record_count: int,
) -> int:
    extra = dict(cfg.extra_config or {})
    if "transition_cache_num_processes" in extra:
        raw = int(extra["transition_cache_num_processes"])
        cap = int(extra.get("transition_cache_max_processes", raw))
        return max(1, min(int(raw), int(cap), int(record_count)))
    return _target_num_processes(cfg, record_count=record_count)


def _transition_cache_child_batch_size(cfg: BellmanConvergenceConfig) -> int:
    return max(1, int((cfg.extra_config or {}).get("transition_cache_child_batch_size", 65536)))


def _transition_cache_inference_device(cfg: BellmanConvergenceConfig) -> str:
    extra = dict(cfg.extra_config or {})
    return str(
        extra.get(
            "transition_cache_inference_device",
            extra.get("eval_device", extra.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        )
    )


def _transition_cache_inference_batch_size(cfg: BellmanConvergenceConfig) -> int:
    return max(1, int((cfg.extra_config or {}).get("transition_cache_inference_batch_size", 65536)))


def _predict_bootstrap_features(
    *,
    model: Any,
    features: torch.Tensor,
    cfg: BellmanConvergenceConfig,
) -> list[float]:
    hook = getattr(infer_module, "predict_model_search_values_from_features", None)
    if not callable(hook):
        raise TypeError("DNN/infer.py does not expose predict_model_search_values_from_features")
    return [
        float(x)
        for x in hook(
            model=model,
            features=features,
            device=_transition_cache_inference_device(cfg),
            batch_size=_transition_cache_inference_batch_size(cfg),
        )
    ]


def _write_transition_cache_shard(
    *,
    shard_path: Path,
    child_records: list[dict[str, Any]],
    root_positions: list[int],
    action_indices: list[int],
    rewards: list[float],
    discounts: list[float],
    cfg: BellmanConvergenceConfig,
    split_name: str,
    worker_id: int,
    part_id: int,
) -> int:
    """Tensorize and write one fixed-transition cache shard."""

    if not child_records:
        return 0
    from ..DNN.model_search_nn import records_to_feature_tensor

    features = records_to_feature_tensor(child_records).to(dtype=_transition_cache_feature_dtype(cfg))
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "split_name": str(split_name),
            "worker_id": int(worker_id),
            "part_id": int(part_id),
            "root_positions": torch.tensor(root_positions, dtype=torch.long),
            "action_indices": torch.tensor(action_indices, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "discounts": torch.tensor(discounts, dtype=torch.float32),
            "child_features": features,
        },
        shard_path,
    )
    return int(features.shape[0])


def _target_inference_device(cfg: BellmanConvergenceConfig) -> str:
    """Device used inside target-generation bootstrap inference."""

    extra = dict(cfg.extra_config or {})
    if "target_device" in extra:
        return str(extra["target_device"])
    if "bootstrap_device" in extra:
        return str(extra["bootstrap_device"])
    if int(extra.get("target_num_processes", extra.get("num_target_processes", 1))) > 1:
        return "cpu"
    return str(extra.get("device", "cuda" if torch.cuda.is_available() else "cpu"))


def _target_num_processes(
    cfg: BellmanConvergenceConfig,
    *,
    record_count: int,
) -> int:
    """Return target worker count, capped because each worker owns a simulator."""

    extra = dict(cfg.extra_config or {})
    raw = int(extra.get("target_num_processes", extra.get("num_target_processes", 1)))
    cap = int(extra.get("target_max_processes", extra.get("max_target_processes", 64)))
    if int(record_count) <= 0:
        return 0
    return max(1, min(max(1, int(cap)), int(raw), int(record_count)))


def _target_mp_start_method(cfg: BellmanConvergenceConfig) -> str:
    """Multiprocessing start method for target workers."""

    return str((cfg.extra_config or {}).get("target_mp_start_method", "fork"))


def _predict_bootstrap_records(
    *,
    model: Any,
    records: list[dict[str, Any]],
    cfg: BellmanConvergenceConfig,
) -> list[float]:
    hook = getattr(infer_module, "predict_model_search_values_from_records", None)
    if not callable(hook):
        raise TypeError("DNN/infer.py does not expose predict_model_search_values_from_records")
    device = _target_inference_device(cfg)
    batch_size = int((cfg.extra_config or {}).get("bootstrap_batch_size", max(4096, int(cfg.batch_size))))
    return [
        float(x)
        for x in hook(
            model=model,
            records=records,
            device=device,
            batch_size=batch_size,
        )
    ]


def compute_controller_mcts_targets_batched(
    *,
    records: Sequence[dict[str, Any]],
    state_loader: RootStateLoader,
    bootstrap_model: Any,
    cfg: BellmanConvergenceConfig,
    split_name: str,
    return_actions: bool = False,
) -> list[float] | tuple[list[float], list[int]]:
    """Compute controller-root Bellman targets with batched child bootstrap.

    This mirrors the depth-1 controller branch in `mctsDNN`, but defers model
    inference until many child states have been collected. Simulator action
    application is still per action; the speedup is from removing per-action
    model calls and batching feature/model work.
    """

    if any(str(record.get("root_player", "")) != "controller" for record in records):
        raise ValueError("batched controller target path requires controller-only records")

    mcts = state_loader.mcts
    child_batch_limit = int((cfg.extra_config or {}).get("bootstrap_child_batch_size", 8192))
    child_batch_limit = max(1, int(child_batch_limit))

    candidates: list[list[tuple[int, float]]] = [[] for _ in records]
    child_records: list[dict[str, Any]] = []
    child_meta: list[tuple[int, int, float, float]] = []

    def flush_child_batch() -> None:
        if not child_records:
            return
        bootstraps = _predict_bootstrap_records(
            model=bootstrap_model,
            records=child_records,
            cfg=cfg,
        )
        if len(bootstraps) != len(child_meta):
            raise RuntimeError(
                f"bootstrap prediction count mismatch: {len(bootstraps)} != {len(child_meta)}"
            )
        for (root_pos, action_idx, reward, discount), bootstrap in zip(child_meta, bootstraps):
            q = float(reward) + float(discount) * float(bootstrap)
            candidates[int(root_pos)].append((int(action_idx), float(q)))
        child_records.clear()
        child_meta.clear()

    for sample_number, record in enumerate(records):
        state = state_loader(record)
        mcts.clear_search_state(drop_scratch=False)

        decision_state = state
        root_cost = float(mcts._state_cost(decision_state))
        root_time = float(decision_state.simulator._time)

        actions_by_index, mask_t = mcts._actions_and_mask(
            decision_state,
            "controller",
            forbidden_stop_ids=None,
        )
        valid_mask = [bool(x) for x in mask_t.tolist()]
        valid_indices = [
            i for i, ok in enumerate(valid_mask) if ok and actions_by_index[i] is not None
        ]
        if not valid_indices:
            candidates[int(sample_number)].append((-1, 0.0))
            continue

        _alias_to_canon, _canon_to_aliases, canonical_indices = mcts._canonicalize_action_indices(
            player="controller",
            actions_by_index=actions_by_index,
            valid_indices=valid_indices,
        )
        decision_snapshot, decision_stats = mcts._snapshot_state_and_stats(decision_state)

        for action_idx in canonical_indices:
            action = actions_by_index[action_idx]
            if action is None:
                continue

            child_state = mcts._scratch_restore(decision_snapshot, decision_stats)
            child_state = mcts._env.apply_controller_action_only(
                child_state,
                action,
                inplace=True,
                fast_forward=False,
            )

            child_cost = float(mcts._state_cost(child_state))
            reward = float(mcts._transition_reward(root_cost, child_cost))
            leaf_time = float(child_state.simulator._time)
            discount_time = getattr(child_state.stats, "transition_discount_time", None)
            if discount_time is None:
                discount_time = leaf_time
            discount = float(mcts._time_discount(float(discount_time), root_time))

            child_records.append(_snapshot_for_model_search_record(child_state))
            child_meta.append((int(sample_number), int(action_idx), float(reward), float(discount)))

            if len(child_records) >= child_batch_limit:
                flush_child_batch()

        if (sample_number + 1) % 1000 == 0:
            print(
                f"[bellman_convergence] {split_name}: prepared {sample_number + 1}/{len(records)} roots "
                f"pending_child_batch={len(child_records)}",
                flush=True,
            )

    flush_child_batch()

    targets: list[float] = []
    best_actions: list[int] = []
    for sample_number, rows in enumerate(candidates):
        if not rows:
            raise RuntimeError(f"no candidate q-values produced for sample {sample_number}")
        best_idx, q = max(rows, key=lambda item: (float(item[1]), -int(item[0])))
        targets.append(float(q))
        best_actions.append(int(best_idx))
    if bool(return_actions):
        return targets, best_actions
    return targets


def _chunk_records_for_target_workers(
    records: Sequence[dict[str, Any]],
    *,
    num_workers: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Split records into contiguous chunks while preserving original positions."""

    worker_count = max(1, int(num_workers))
    chunk_size = (len(records) + worker_count - 1) // worker_count
    chunks: list[list[tuple[int, dict[str, Any]]]] = []
    for start in range(0, len(records), chunk_size):
        chunk = [
            (int(pos), dict(record))
            for pos, record in enumerate(records[start : start + chunk_size], start=start)
        ]
        if chunk:
            chunks.append(chunk)
    return chunks


def _transition_cache_metadata_is_ready(
    metadata_path: Path,
    *,
    expected_count: int,
    split_name: str,
    expected_feature_dtype: str,
) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return False
    if int(metadata.get("schema_version", -1)) != 1:
        return False
    if str(metadata.get("split_name", "")) != _base_transition_split_name(split_name):
        return False
    if int(metadata.get("record_count", -1)) != int(expected_count):
        return False
    if str(metadata.get("feature_dtype", "")) != str(expected_feature_dtype):
        return False
    for shard_path in metadata.get("shard_paths", []):
        if not Path(str(shard_path)).exists():
            return False
    return True


def _run_transition_cache_worker(task: TransitionCacheWorkerTask) -> TransitionCacheWorkerResult:
    """Worker entrypoint for fixed controller transition-cache construction."""

    extra = dict(task.cfg.extra_config or {})
    torch_threads = max(1, int(extra.get("target_torch_threads_per_worker", 1)))
    torch.set_num_threads(torch_threads)

    worker_output_dir = (
        task.cfg.output_dir
        / "_transition_cache_workers"
        / f"{_base_transition_split_name(task.split_name)}_worker_{int(task.worker_id):02d}"
    )
    worker_cfg = replace(
        task.cfg,
        output_dir=worker_output_dir,
        num_roots=max(2, len(task.records_with_positions)),
        extra_config=extra,
    )
    worker_model_cfg = make_self_model_config(
        worker_cfg,
        output_dir=worker_output_dir,
        model_name=(
            f"{task.cfg.model_name_prefix}_{_base_transition_split_name(task.split_name)}"
            f"_transition_cache_worker_{int(task.worker_id)}"
        ),
    )

    state_loader = RootStateLoader(worker_model_cfg)
    mcts = state_loader.mcts
    child_batch_limit = _transition_cache_child_batch_size(task.cfg)
    split_base = _base_transition_split_name(task.split_name)
    shard_dir = _transition_cache_dir(task.cfg) / split_base / f"worker_{int(task.worker_id):03d}"

    child_records: list[dict[str, Any]] = []
    root_positions: list[int] = []
    action_indices: list[int] = []
    rewards: list[float] = []
    discounts: list[float] = []
    shard_paths: list[str] = []
    empty_root_positions: list[int] = []
    child_count = 0
    part_id = 0

    def flush_child_cache() -> None:
        nonlocal child_count, part_id
        if not child_records:
            return
        shard_path = shard_dir / f"part_{int(part_id):05d}.pt"
        written = _write_transition_cache_shard(
            shard_path=shard_path,
            child_records=child_records,
            root_positions=root_positions,
            action_indices=action_indices,
            rewards=rewards,
            discounts=discounts,
            cfg=task.cfg,
            split_name=split_base,
            worker_id=int(task.worker_id),
            part_id=int(part_id),
        )
        if written > 0:
            shard_paths.append(str(shard_path))
            child_count += int(written)
            part_id += 1
        child_records.clear()
        root_positions.clear()
        action_indices.clear()
        rewards.clear()
        discounts.clear()

    try:
        for local_idx, (root_pos, record) in enumerate(task.records_with_positions):
            state = state_loader(record)
            mcts.clear_search_state(drop_scratch=False)

            root_cost = float(mcts._state_cost(state))
            root_time = float(state.simulator._time)
            actions_by_index, mask_t = mcts._actions_and_mask(
                state,
                "controller",
                forbidden_stop_ids=None,
            )
            valid_mask = [bool(x) for x in mask_t.tolist()]
            valid_indices = [
                i for i, ok in enumerate(valid_mask) if ok and actions_by_index[i] is not None
            ]
            if not valid_indices:
                empty_root_positions.append(int(root_pos))
                continue

            _alias_to_canon, _canon_to_aliases, canonical_indices = mcts._canonicalize_action_indices(
                player="controller",
                actions_by_index=actions_by_index,
                valid_indices=valid_indices,
            )
            decision_snapshot, decision_stats = mcts._snapshot_state_and_stats(state)

            for action_idx in canonical_indices:
                action = actions_by_index[action_idx]
                if action is None:
                    continue

                child_state = mcts._scratch_restore(decision_snapshot, decision_stats)
                child_state = mcts._env.apply_controller_action_only(
                    child_state,
                    action,
                    inplace=True,
                    fast_forward=False,
                )

                child_cost = float(mcts._state_cost(child_state))
                reward = float(mcts._transition_reward(root_cost, child_cost))
                leaf_time = float(child_state.simulator._time)
                discount_time = getattr(child_state.stats, "transition_discount_time", None)
                if discount_time is None:
                    discount_time = leaf_time
                discount = float(mcts._time_discount(float(discount_time), root_time))

                child_records.append(_snapshot_for_model_search_record(child_state))
                root_positions.append(int(root_pos))
                action_indices.append(int(action_idx))
                rewards.append(float(reward))
                discounts.append(float(discount))

            if len(child_records) >= child_batch_limit:
                flush_child_cache()

            if (local_idx + 1) % 1000 == 0:
                print(
                    "[bellman_convergence] "
                    f"{split_base}/cache_worker_{int(task.worker_id)}: prepared "
                    f"{local_idx + 1}/{len(task.records_with_positions)} roots "
                    f"pending_children={len(child_records)}",
                    flush=True,
                )

        flush_child_cache()
    finally:
        state_loader.close()

    return TransitionCacheWorkerResult(
        worker_id=int(task.worker_id),
        split_name=split_base,
        shard_paths=shard_paths,
        empty_root_positions=empty_root_positions,
        root_count=len(task.records_with_positions),
        child_count=int(child_count),
    )


def ensure_controller_transition_cache(
    *,
    records: Sequence[dict[str, Any]],
    cfg: BellmanConvergenceConfig,
    split_name: str,
) -> dict[str, Any]:
    """Build or load fixed controller child-transition feature cache."""

    split_base = _base_transition_split_name(split_name)
    metadata_path = _transition_cache_metadata_path(cfg, split_base)
    if _transition_cache_metadata_is_ready(
        metadata_path,
        expected_count=len(records),
        split_name=split_base,
        expected_feature_dtype=str(_transition_cache_feature_dtype(cfg)).replace("torch.", ""),
    ):
        return json.loads(metadata_path.read_text())

    if any(str(record.get("root_player", "")) != "controller" for record in records):
        raise ValueError("transition cache requires controller-only records")

    chunks = _chunk_records_for_target_workers(
        records,
        num_workers=_transition_cache_num_processes(cfg, record_count=len(records)),
    )
    tasks = [
        TransitionCacheWorkerTask(
            worker_id=int(worker_id),
            cfg=cfg,
            records_with_positions=chunk,
            split_name=split_base,
        )
        for worker_id, chunk in enumerate(chunks)
    ]
    print(
        "[bellman_convergence] "
        f"{split_base}: building transition cache with {len(tasks)} workers "
        f"feature_dtype={str(_transition_cache_feature_dtype(cfg)).replace('torch.', '')}",
        flush=True,
    )

    if len(tasks) <= 1:
        results = [_run_transition_cache_worker(tasks[0])]
    else:
        ctx = mp.get_context(_target_mp_start_method(cfg))
        with ctx.Pool(processes=len(tasks)) as pool:
            results = list(pool.imap_unordered(_run_transition_cache_worker, tasks))

    shard_paths: list[str] = []
    empty_root_positions: list[int] = []
    child_count = 0
    root_count = 0
    for result in sorted(results, key=lambda item: int(item.worker_id)):
        shard_paths.extend(str(path) for path in result.shard_paths)
        empty_root_positions.extend(int(pos) for pos in result.empty_root_positions)
        child_count += int(result.child_count)
        root_count += int(result.root_count)
        print(
            "[bellman_convergence] "
            f"{split_base}: cache worker {int(result.worker_id)} finished "
            f"roots={int(result.root_count)} children={int(result.child_count)}",
            flush=True,
        )

    metadata = {
        "schema_version": 1,
        "split_name": split_base,
        "record_count": int(len(records)),
        "root_count": int(root_count),
        "child_count": int(child_count),
        "feature_dtype": str(_transition_cache_feature_dtype(cfg)).replace("torch.", ""),
        "shard_paths": shard_paths,
        "empty_root_positions": sorted(set(int(pos) for pos in empty_root_positions)),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def compute_controller_mcts_targets_from_transition_cache(
    *,
    records: Sequence[dict[str, Any]],
    bootstrap_model: Any,
    cfg: BellmanConvergenceConfig,
    split_name: str,
    root_limit: int | None = None,
    return_actions: bool = False,
) -> list[float] | tuple[list[float], list[int]]:
    """Compute Bellman targets from cached child features/reward/discount rows."""

    metadata = ensure_controller_transition_cache(
        records=records,
        cfg=cfg,
        split_name=split_name,
    )
    record_count = int(len(records))
    if int(metadata.get("record_count", -1)) != record_count:
        raise RuntimeError(
            f"transition cache record count mismatch: {metadata.get('record_count')} != {record_count}"
        )
    active_count = record_count if root_limit is None else min(int(root_limit), record_count)
    best_values = [float("-inf")] * active_count
    best_actions = [2**31 - 1] * active_count

    for shard_path_raw in metadata.get("shard_paths", []):
        payload = torch.load(str(shard_path_raw), map_location="cpu", weights_only=False)
        root_positions_t = payload["root_positions"].to(dtype=torch.long)
        if root_limit is not None:
            mask = root_positions_t < int(active_count)
            if not bool(mask.any()):
                continue
            root_positions_t = root_positions_t[mask]
            action_indices_t = payload["action_indices"].to(dtype=torch.long)[mask]
            rewards_t = payload["rewards"].to(dtype=torch.float32)[mask]
            discounts_t = payload["discounts"].to(dtype=torch.float32)[mask]
            features_t = payload["child_features"][mask]
        else:
            action_indices_t = payload["action_indices"].to(dtype=torch.long)
            rewards_t = payload["rewards"].to(dtype=torch.float32)
            discounts_t = payload["discounts"].to(dtype=torch.float32)
            features_t = payload["child_features"]

        predictions = torch.tensor(
            _predict_bootstrap_features(
                model=bootstrap_model,
                features=features_t,
                cfg=cfg,
            ),
            dtype=torch.float32,
        )
        q_values = rewards_t + discounts_t * predictions
        for root_pos, action_idx, q_value in zip(
            root_positions_t.tolist(),
            action_indices_t.tolist(),
            q_values.tolist(),
        ):
            pos = int(root_pos)
            action = int(action_idx)
            q = float(q_value)
            if pos >= active_count:
                continue
            if q > best_values[pos] or (q == best_values[pos] and action < best_actions[pos]):
                best_values[pos] = q
                best_actions[pos] = action

    for pos in metadata.get("empty_root_positions", []):
        pos = int(pos)
        if pos < active_count:
            best_values[pos] = 0.0
            best_actions[pos] = -1

    missing = [idx for idx, value in enumerate(best_values) if value == float("-inf")]
    if missing:
        raise RuntimeError(f"transition cache missing roots: {missing[:10]}")

    actions = [(-1 if action == 2**31 - 1 else int(action)) for action in best_actions]
    if bool(return_actions):
        return [float(value) for value in best_values], actions
    return [float(value) for value in best_values]


def validate_controller_transition_cache(
    *,
    records: Sequence[dict[str, Any]],
    state_loader: RootStateLoader,
    bootstrap_model: Any,
    cfg: BellmanConvergenceConfig,
    split_name: str,
) -> None:
    """Compare cached Bellman targets/actions against the current direct path."""

    validation_roots = int((cfg.extra_config or {}).get("transition_cache_validation_roots", 0))
    if validation_roots <= 0:
        return
    sample_count = min(int(validation_roots), len(records))
    if sample_count <= 0:
        return

    direct_targets, direct_actions = compute_controller_mcts_targets_batched(
        records=records[:sample_count],
        state_loader=state_loader,
        bootstrap_model=bootstrap_model,
        cfg=cfg,
        split_name=f"{split_name}/validation_direct",
        return_actions=True,
    )
    cached_targets, cached_actions = compute_controller_mcts_targets_from_transition_cache(
        records=records,
        bootstrap_model=bootstrap_model,
        cfg=cfg,
        split_name=split_name,
        root_limit=sample_count,
        return_actions=True,
    )
    atol = float((cfg.extra_config or {}).get("transition_cache_validation_atol", 1e-4))
    max_diff = 0.0
    bad_examples: list[str] = []
    for idx, (direct_target, cached_target, direct_action, cached_action) in enumerate(
        zip(direct_targets, cached_targets, direct_actions, cached_actions)
    ):
        diff = abs(float(direct_target) - float(cached_target))
        max_diff = max(max_diff, float(diff))
        if diff > atol or int(direct_action) != int(cached_action):
            bad_examples.append(
                f"idx={idx} direct=({direct_target:.8f},a={direct_action}) "
                f"cached=({cached_target:.8f},a={cached_action}) diff={diff:.8f}"
            )
        if len(bad_examples) >= 5:
            break
    if bad_examples:
        raise RuntimeError(
            "transition cache validation failed: "
            + "; ".join(bad_examples)
            + f"; max_diff={max_diff:.8f}; atol={atol}"
        )
    print(
        "[bellman_convergence] "
        f"{_base_transition_split_name(split_name)}: transition cache validation passed "
        f"roots={sample_count} max_abs_diff={max_diff:.8f} atol={atol}",
        flush=True,
    )


def _run_target_worker(task: TargetWorkerTask) -> TargetWorkerResult:
    """Worker process entrypoint for Bellman target computation."""

    extra = dict(task.cfg.extra_config or {})
    torch_threads = max(1, int(extra.get("target_torch_threads_per_worker", 1)))
    torch.set_num_threads(torch_threads)

    worker_output_dir = (
        task.cfg.output_dir
        / "_target_workers"
        / f"{task.split_name}_worker_{int(task.worker_id):02d}"
    )
    worker_cfg = replace(
        task.cfg,
        output_dir=worker_output_dir,
        num_roots=max(2, len(task.records_with_positions)),
        extra_config=extra,
    )
    worker_model_cfg = make_self_model_config(
        worker_cfg,
        output_dir=worker_output_dir,
        model_name=f"{task.cfg.model_name_prefix}_{task.split_name}_target_worker_{int(task.worker_id)}",
    )
    positions = [int(pos) for pos, _record in task.records_with_positions]
    worker_records = [dict(record) for _pos, record in task.records_with_positions]
    state_loader = RootStateLoader(worker_model_cfg)
    try:
        targets = compute_controller_mcts_targets_batched(
            records=worker_records,
            state_loader=state_loader,
            bootstrap_model=task.bootstrap_model,
            cfg=worker_cfg,
            split_name=f"{task.split_name}/worker_{int(task.worker_id)}",
        )
    finally:
        state_loader.close()

    if len(targets) != len(positions):
        raise RuntimeError(
            f"worker {task.worker_id} target count mismatch: {len(targets)} != {len(positions)}"
        )
    shard_dir = task.cfg.output_dir / "_target_worker_shards" / str(task.split_name)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"worker_{int(task.worker_id):03d}.pt"
    torch.save(
        {
            "worker_id": int(task.worker_id),
            "split_name": str(task.split_name),
            "positions": [int(pos) for pos in positions],
            "targets": [float(target) for target in targets],
        },
        shard_path,
    )
    return TargetWorkerResult(
        worker_id=int(task.worker_id),
        split_name=str(task.split_name),
        shard_path=str(shard_path),
        target_count=len(targets),
    )


def compute_controller_mcts_targets_multiprocess(
    *,
    records: Sequence[dict[str, Any]],
    bootstrap_model: Any,
    cfg: BellmanConvergenceConfig,
    split_name: str,
) -> list[float]:
    """Compute controller Bellman targets across multiple local simulator workers."""

    if any(str(record.get("root_player", "")) != "controller" for record in records):
        raise ValueError("multiprocess target path requires controller-only records")

    num_workers = _target_num_processes(cfg, record_count=len(records))
    if num_workers <= 1:
        worker_model_cfg = make_self_model_config(
            cfg,
            output_dir=cfg.output_dir / "_target_workers" / f"{split_name}_worker_00",
            model_name=f"{cfg.model_name_prefix}_{split_name}_target_worker_0",
        )
        state_loader = RootStateLoader(worker_model_cfg)
        try:
            return compute_controller_mcts_targets_batched(
                records=records,
                state_loader=state_loader,
                bootstrap_model=bootstrap_model,
                cfg=cfg,
                split_name=str(split_name),
            )
        finally:
            state_loader.close()

    chunks = _chunk_records_for_target_workers(records, num_workers=num_workers)
    tasks = [
        TargetWorkerTask(
            worker_id=int(worker_id),
            cfg=cfg,
            records_with_positions=chunk,
            bootstrap_model=bootstrap_model,
            split_name=str(split_name),
        )
        for worker_id, chunk in enumerate(chunks)
    ]
    print(
        f"[bellman_convergence] {split_name}: computing targets with {len(tasks)} workers "
        f"device={_target_inference_device(cfg)} start_method={_target_mp_start_method(cfg)}",
        flush=True,
    )

    targets: list[float | None] = [None] * len(records)
    ctx = mp.get_context(_target_mp_start_method(cfg))
    with ctx.Pool(processes=len(tasks)) as pool:
        for result in pool.imap_unordered(_run_target_worker, tasks):
            payload = torch.load(result.shard_path, map_location="cpu")
            positions = [int(pos) for pos in payload["positions"]]
            shard_targets = [float(target) for target in payload["targets"]]
            if len(positions) != len(shard_targets):
                raise RuntimeError(
                    f"worker {result.worker_id} shard count mismatch: "
                    f"{len(positions)} != {len(shard_targets)}"
                )
            for pos, target in zip(positions, shard_targets):
                targets[int(pos)] = float(target)
            completed = sum(1 for item in targets if item is not None)
            print(
                f"[bellman_convergence] {split_name}: worker {result.worker_id} finished "
                f"shard_targets={int(result.target_count)} stored_targets={completed}/{len(records)}",
                flush=True,
            )

    missing = [idx for idx, target in enumerate(targets) if target is None]
    if missing:
        raise RuntimeError(f"missing targets after multiprocess run: {missing[:10]}")
    return [float(target) for target in targets]


def compute_mcts_targets(
    *,
    records: Sequence[dict[str, Any]],
    state_loader: RootStateLoader,
    bootstrap_model: Any | None,
    cfg: BellmanConvergenceConfig,
    bootstrap_version: int,
    model_version: int,
    output_dir: Path,
    split_name: str,
    cache_targets: bool,
    reuse_record_targets_for_v0: bool,
) -> list[float]:
    """Compute T[V_bootstrap](s) with mctsDNN for one split."""

    cache_path = target_cache_path(
        output_dir,
        model_version=int(model_version),
        bootstrap_version=int(bootstrap_version),
        split_name=str(split_name),
    )
    if bool(cache_targets):
        cached = load_cached_targets(cache_path, expected_count=len(records))
        if cached is not None:
            return cached

    if int(bootstrap_version) == 0 and bool(reuse_record_targets_for_v0):
        targets = [float(record["target_value"]) for record in records]
        if bool(cache_targets):
            save_cached_targets(
                cache_path,
                targets=targets,
                metadata={
                    "split_name": str(split_name),
                    "model_version": int(model_version),
                    "bootstrap_version": int(bootstrap_version),
                    "source": "stored_no_bootstrap_root_targets",
                },
            )
        return targets

    wrapped_model = None if bootstrap_model is None else BootstrapModelAdapter(bootstrap_model)
    use_bootstrap = wrapped_model is not None and int(bootstrap_version) > 0

    if use_bootstrap:
        if all(str(record.get("root_player", "")) == "controller" for record in records) and _transition_cache_enabled(cfg):
            targets = compute_controller_mcts_targets_from_transition_cache(
                records=records,
                bootstrap_model=bootstrap_model,
                cfg=cfg,
                split_name=str(split_name),
            )
            validate_controller_transition_cache(
                records=records,
                state_loader=state_loader,
                bootstrap_model=bootstrap_model,
                cfg=cfg,
                split_name=str(split_name),
            )
            if bool(cache_targets):
                save_cached_targets(
                    cache_path,
                    targets=targets,
                    metadata={
                        "split_name": str(split_name),
                        "model_version": int(model_version),
                        "bootstrap_version": int(bootstrap_version),
                        "source": "transition_cache_controller_mcts_dnn_search",
                        "use_model_bootstrap": bool(use_bootstrap),
                        "target_device": _transition_cache_inference_device(cfg),
                        "transition_cache_dir": str(_transition_cache_dir(cfg)),
                    },
                )
            return targets

        try:
            if all(str(record.get("root_player", "")) == "controller" for record in records):
                target_workers = _target_num_processes(cfg, record_count=len(records))
                if target_workers > 1:
                    targets = compute_controller_mcts_targets_multiprocess(
                        records=records,
                        bootstrap_model=bootstrap_model,
                        cfg=cfg,
                        split_name=str(split_name),
                    )
                    target_source = "multiprocess_batched_controller_mcts_dnn_search"
                else:
                    targets = compute_controller_mcts_targets_batched(
                        records=records,
                        state_loader=state_loader,
                        bootstrap_model=bootstrap_model,
                        cfg=cfg,
                        split_name=str(split_name),
                    )
                    target_source = "batched_controller_mcts_dnn_search"
                if bool(cache_targets):
                    save_cached_targets(
                        cache_path,
                        targets=targets,
                        metadata={
                            "split_name": str(split_name),
                            "model_version": int(model_version),
                            "bootstrap_version": int(bootstrap_version),
                            "source": target_source,
                            "use_model_bootstrap": bool(use_bootstrap),
                            "target_num_processes": int(target_workers),
                            "target_device": _target_inference_device(cfg),
                        },
                    )
                return targets
        except Exception as exc:
            print(
                "[bellman_convergence] batched controller target path failed; "
                f"falling back to per-sample search_dnn: {exc!r}",
                flush=True,
            )

    targets: list[float] = []
    mcts = state_loader.mcts
    for sample_number, record in enumerate(records):
        state = state_loader(record)
        out = mcts.search_dnn(
            dnn_model=wrapped_model,
            rootState=state,
            root_player=str(record["root_player"]),
            game_id=0,
            root_id=int(record.get("root_id", sample_number)),
            root_node_id_override=record.get("root_node_id_override", None),
            root_depth=int(record.get("root_depth", 0)),
            model_version=int(bootstrap_version),
            use_model_bootstrap=bool(use_bootstrap),
            one_step_value_mode=True,
        )
        targets.append(float(out.best_action_value))

    if bool(cache_targets):
        save_cached_targets(
            cache_path,
            targets=targets,
            metadata={
                "split_name": str(split_name),
                "model_version": int(model_version),
                "bootstrap_version": int(bootstrap_version),
                "source": "mcts_dnn_search",
                "use_model_bootstrap": bool(use_bootstrap),
            },
        )
    return targets


def train_model(
    *,
    model_version: int,
    train_records: Sequence[dict[str, Any]],
    eval_records: Sequence[dict[str, Any]],
    train_targets: Sequence[float],
    eval_targets: Sequence[float],
    cfg: BellmanConvergenceConfig,
    state_loader: RootStateLoader,
) -> Any:
    """Train model version Vi on T[V{i-1}] targets."""

    model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
    model_cfg = make_self_model_config(
        cfg,
        output_dir=model_dir,
        model_name=f"{cfg.model_name_prefix}_v{int(model_version)}",
    )
    train_labeled = records_with_targets(train_records, train_targets)
    eval_labeled = records_with_targets(eval_records, eval_targets)
    artifacts = train_candidate_model(
        train_labeled,
        eval_labeled,
        model_cfg,
        state_loader,
    )
    write_training_metrics(artifacts.train_metrics, model_cfg)
    return artifacts.model


def evaluate_model_on_dataset(
    *,
    model: Any,
    records: Sequence[dict[str, Any]],
    targets: Sequence[float],
    sample_numbers: Sequence[int],
    cfg: BellmanConvergenceConfig,
    state_loader: RootStateLoader,
    model_version: int,
    split_name: str,
    output_csv: Path,
) -> dict[str, Any]:
    """Predict model values, write per-sample CSV, and return aggregate stats."""

    model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
    model_cfg = make_self_model_config(
        cfg,
        output_dir=model_dir,
        model_name=f"{cfg.model_name_prefix}_v{int(model_version)}",
    )
    labeled_records = records_with_targets(records, targets)
    predictions = predict_candidate_values(
        model,
        labeled_records,
        model_cfg,
        state_loader,
        split_name=str(split_name),
    )
    bellman_logger.write_prediction_results_csv(
        output_csv,
        predictions=predictions,
        targets=targets,
        sample_numbers=sample_numbers,
    )
    metrics = bellman_logger.compute_error_metrics(
        predictions,
        targets,
        abs_error_threshold=float(cfg.abs_error_threshold),
    )
    row = {
        "model_version": int(model_version),
        "split": str(split_name),
        "trainable_params": int(count_trainable_parameters(model)),
    }
    row.update(metrics)
    return row


def write_transition_summary(
    cfg: BellmanConvergenceConfig,
    *,
    source_model_version: int,
    target_model_version: int,
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Write aggregate stats for a source->target Bellman transition."""

    out = cfg.output_dir / f"version_{int(source_model_version)}_to_{int(target_model_version)}.csv"
    enriched = []
    for row in rows:
        item = {
            "source_model_version": int(source_model_version),
            "target_model_version": int(target_model_version),
        }
        item.update(dict(row))
        enriched.append(item)
    return bellman_logger.write_summary_csv(out, enriched)


def bellman_convergence(cfg: BellmanConvergenceConfig) -> None:
    """Run iterative Bellman training and same-model residual analysis."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model0_cfg = make_self_model_config(
        cfg,
        output_dir=cfg.output_dir / "Model_Version0",
        model_name=f"{cfg.model_name_prefix}_v0",
    )
    records = load_root_records(model0_cfg)
    train_records, eval_records, train_indices, eval_indices = split_records_with_indices(
        records,
        eval_ratio=float(cfg.eval_ratio),
        split_seed=int(cfg.split_seed),
    )
    bellman_logger.write_split_indices(
        cfg.output_dir / "split_indices.json",
        train_indices=train_indices,
        eval_indices=eval_indices,
    )
    bellman_logger.write_json(
        cfg.output_dir / "config.json",
        {
            "dataset_dir": str(cfg.dataset_dir),
            "output_dir": str(cfg.output_dir),
            "num_versions": int(cfg.num_versions),
            "num_roots": int(cfg.num_roots),
            "eval_ratio": float(cfg.eval_ratio),
            "split_seed": int(cfg.split_seed),
            "root_player_filter": str(cfg.root_player_filter),
            "model_name_prefix": str(cfg.model_name_prefix),
            "extra_config": dict(cfg.extra_config or {}),
        },
    )

    state_loader = RootStateLoader(model0_cfg)
    previous_model: Any | None = None
    try:
        for model_version in range(1, int(cfg.num_versions) + 1):
            bootstrap_version = int(model_version) - 1
            model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
            model_dir.mkdir(parents=True, exist_ok=True)
            _log_process_memory(f"version_{int(model_version)}_start")

            train_targets = compute_mcts_targets(
                records=train_records,
                state_loader=state_loader,
                bootstrap_model=previous_model,
                cfg=cfg,
                bootstrap_version=bootstrap_version,
                model_version=int(model_version),
                output_dir=cfg.output_dir,
                split_name="train",
                cache_targets=bool(cfg.cache_targets),
                reuse_record_targets_for_v0=bool(cfg.reuse_record_targets_for_v0),
            )
            eval_targets = compute_mcts_targets(
                records=eval_records,
                state_loader=state_loader,
                bootstrap_model=previous_model,
                cfg=cfg,
                bootstrap_version=bootstrap_version,
                model_version=int(model_version),
                output_dir=cfg.output_dir,
                split_name="eval",
                cache_targets=bool(cfg.cache_targets),
                reuse_record_targets_for_v0=bool(cfg.reuse_record_targets_for_v0),
            )
            _log_process_memory(f"version_{int(model_version)}_after_targets")

            # The previous model is only needed to generate this version's
            # Bellman targets. Release it before training the next model so
            # cached feature tensors and CUDA allocations do not accumulate.
            old_model = previous_model
            previous_model = None
            if old_model is not None:
                _release_iteration_memory(
                    model=old_model,
                    label=f"version_{int(model_version)}_after_previous_model_release",
                )
                del old_model

            model = train_model(
                model_version=int(model_version),
                train_records=train_records,
                eval_records=eval_records,
                train_targets=train_targets,
                eval_targets=eval_targets,
                cfg=cfg,
                state_loader=state_loader,
            )

            train_row = evaluate_model_on_dataset(
                model=model,
                records=train_records,
                targets=train_targets,
                sample_numbers=train_indices,
                cfg=cfg,
                state_loader=state_loader,
                model_version=int(model_version),
                split_name="train",
                output_csv=model_dir / "train_results.csv",
            )
            eval_row = evaluate_model_on_dataset(
                model=model,
                records=eval_records,
                targets=eval_targets,
                sample_numbers=eval_indices,
                cfg=cfg,
                state_loader=state_loader,
                model_version=int(model_version),
                split_name="eval",
                output_csv=model_dir / "eval_results.csv",
            )
            write_transition_summary(
                cfg,
                source_model_version=bootstrap_version,
                target_model_version=int(model_version),
                rows=[train_row, eval_row],
            )

            if int(model_version) >= int(cfg.same_model_analysis_start_version):
                same_train_targets = compute_mcts_targets(
                    records=train_records,
                    state_loader=state_loader,
                    bootstrap_model=model,
                    cfg=cfg,
                    bootstrap_version=int(model_version),
                    model_version=int(model_version),
                    output_dir=cfg.output_dir,
                    split_name="train_same_model",
                    cache_targets=bool(cfg.cache_targets),
                    reuse_record_targets_for_v0=False,
                )
                same_eval_targets = compute_mcts_targets(
                    records=eval_records,
                    state_loader=state_loader,
                    bootstrap_model=model,
                    cfg=cfg,
                    bootstrap_version=int(model_version),
                    model_version=int(model_version),
                    output_dir=cfg.output_dir,
                    split_name="eval_same_model",
                    cache_targets=bool(cfg.cache_targets),
                    reuse_record_targets_for_v0=False,
                )
                same_train_row = evaluate_model_on_dataset(
                    model=model,
                    records=train_records,
                    targets=same_train_targets,
                    sample_numbers=train_indices,
                    cfg=cfg,
                    state_loader=state_loader,
                    model_version=int(model_version),
                    split_name="train_same_model_bootstrap",
                    output_csv=model_dir / "train_result_same_model_boostrap.csv",
                )
                same_eval_row = evaluate_model_on_dataset(
                    model=model,
                    records=eval_records,
                    targets=same_eval_targets,
                    sample_numbers=eval_indices,
                    cfg=cfg,
                    state_loader=state_loader,
                    model_version=int(model_version),
                    split_name="eval_same_model_bootstrap",
                    output_csv=model_dir / "eval_result_same_model_boostrap.csv",
                )
                write_transition_summary(
                    cfg,
                    source_model_version=int(model_version),
                    target_model_version=int(model_version),
                    rows=[same_train_row, same_eval_row],
                )
                del same_train_targets
                del same_eval_targets
                del same_train_row
                del same_eval_row

            _release_iteration_memory(
                model=model,
                label=f"version_{int(model_version)}_after_prediction_cache_release",
            )
            del train_targets
            del eval_targets
            del train_row
            del eval_row
            previous_model = model
            _log_process_memory(f"version_{int(model_version)}_end")
    finally:
        state_loader.close()
        _release_iteration_memory(
            model=previous_model,
            label="final_cleanup",
        )


def parse_args() -> BellmanConvergenceConfig:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run GV3 ModelSearchBed Bellman convergence.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-versions", type=int, default=5)
    parser.add_argument("--num-roots", type=int, default=10_000)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--root-player-filter", choices=("controller", "adversary", "any"), default="controller")
    parser.add_argument("--model-name-prefix", default="bellman_model")
    parser.add_argument("--abs-error-threshold", type=float, default=1.0)
    parser.add_argument(
        "--recompute-v0-targets",
        action="store_true",
        help="Recompute no-bootstrap V0 targets with mctsDNN instead of reusing stored root targets.",
    )
    parser.add_argument(
        "--no-cache-targets",
        action="store_true",
        help="Do not read/write cached MCTS targets.",
    )
    parser.add_argument("--same-model-analysis-start-version", type=int, default=2)
    parser.add_argument(
        "--extra-config-json",
        default="{}",
        help="Free-form JSON object passed through to trainer/infer hooks.",
    )
    args = parser.parse_args()
    extra_config = json.loads(args.extra_config_json)
    if not isinstance(extra_config, dict):
        raise ValueError("--extra-config-json must decode to a JSON object")
    return build_bellman_config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_versions=args.num_versions,
        num_roots=args.num_roots,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        seed=args.seed,
        batch_size=args.batch_size,
        root_player_filter=args.root_player_filter,
        model_name_prefix=args.model_name_prefix,
        abs_error_threshold=args.abs_error_threshold,
        reuse_record_targets_for_v0=not bool(args.recompute_v0_targets),
        cache_targets=not bool(args.no_cache_targets),
        same_model_analysis_start_version=args.same_model_analysis_start_version,
        extra_config=extra_config,
    )


def main() -> None:
    """CLI entrypoint."""

    bellman_convergence(parse_args())


if __name__ == "__main__":
    main()
