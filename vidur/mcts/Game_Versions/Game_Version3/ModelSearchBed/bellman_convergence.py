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
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from ..DNN import infer as infer_module
from ..DNN import infer as dnn_infer
from ..mctsDNN import MCTSNode
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
    target_num_processes: int = 1
    extra_config: dict[str, Any] | None = None


_TARGET_WORKER_STATE_LOADER: RootStateLoader | None = None
_TARGET_WORKER_BOOTSTRAP_MODEL: Any | None = None
_TARGET_WORKER_BOOTSTRAP_VERSION = 0
_TARGET_WORKER_CHILD_BATCH_SIZE = 4096


def _limit_native_threads(num_threads: int) -> None:
    """Limit BLAS/OpenMP pools inside spawned target workers."""

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
    target_num_processes: int = 1,
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
    if int(target_num_processes) <= 0:
        raise ValueError("target_num_processes must be > 0")
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
        target_num_processes=int(target_num_processes),
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


def _predict_bootstrap_inputs_batch(
    *,
    model: Any,
    inputs_list: Sequence[Any],
    players: Sequence[str],
    child_batch_size: int,
    action_feature_rows: Sequence[Sequence[float]] | None = None,
) -> list[float]:
    """Predict bootstrap values for child states in vectorized chunks.

    When `action_feature_rows` is provided (one row per inputs entry), the
    NN backend uses them as action-conditional augmentation. Other backends
    ignore them.
    """

    hook = getattr(infer_module, "predict_model_search_values_from_inputs_batch", None)
    adapter = BootstrapModelAdapter(model)
    out: list[float] = []
    batch_size = max(1, int(child_batch_size))
    for start in range(0, len(inputs_list), batch_size):
        end = min(len(inputs_list), start + batch_size)
        chunk_inputs = list(inputs_list[start:end])
        chunk_players = list(players[start:end])
        chunk_action_rows = (
            list(action_feature_rows[start:end])
            if action_feature_rows is not None
            else None
        )
        if callable(hook):
            try:
                values = hook(
                    model=model,
                    inputs_list=chunk_inputs,
                    players=chunk_players,
                    device=torch.device("cpu"),
                    action_feature_rows=chunk_action_rows,
                )
            except TypeError:
                # Older hook signature without action_feature_rows.
                values = hook(
                    model=model,
                    inputs_list=chunk_inputs,
                    players=chunk_players,
                    device=torch.device("cpu"),
                )
            out.extend(float(x) for x in values)
        else:
            for inputs, player in zip(chunk_inputs, chunk_players):
                value, _priors = adapter.infer_from_inputs(
                    inputs,
                    str(player),
                    device=torch.device("cpu"),
                )
                out.append(float(value))
    return out


def _compute_controller_targets_for_chunk(
    *,
    records: Sequence[dict[str, Any]],
    state_loader: RootStateLoader,
    bootstrap_model: Any,
    bootstrap_version: int,
    child_batch_size: int,
) -> list[float]:
    """Compute controller-root Bellman targets with batched child bootstrap."""

    mcts = state_loader.mcts
    targets: list[float | None] = [None] * len(records)
    root_infos: list[dict[str, Any] | None] = [None] * len(records)
    child_inputs: list[Any] = []
    child_players: list[str] = []
    child_jobs: list[tuple[int, int, float, float, float, float]] = []
    child_action_rows: list[list[float]] = []
    # Action features for child states keep the bootstrap distribution
    # consistent with the training distribution. The NN backend trains on
    # cliff+action features and would receive zero-padded action features
    # at bootstrap if we don't compute them here. Lazily import to avoid a
    # hard dep when other backends are in use.
    bc_extra = getattr(state_loader, "_bellman_extra_config", {}) or {}
    # Two independent flags:
    #   nn_use_action_features    -> training uses action features (always cheap, cached)
    #   nn_bootstrap_action_features -> bootstrap also computes action features per
    #     child (expensive: O(N^2) sim steps per record). Default OFF since the
    #     mask-augmentation training already handles the "action features missing"
    #     case at bootstrap.
    use_action_features_at_bootstrap = bool(
        bc_extra.get("nn_bootstrap_action_features", False)
    ) and str(bc_extra.get("classical_backend", "")) == "neural"
    if use_action_features_at_bootstrap:
        from .action_features import (
            DEFAULT_TOP_K_ACTIONS,
            extract_action_features_from_state,
        )
        from ....game_types import AdversaryAction

        action_top_k = int(bc_extra.get("nn_action_top_k", DEFAULT_TOP_K_ACTIONS))
        # If True, advance the post-controller leaf state by a noop adversary
        # action so the resulting state is controller-to-act and matches the
        # training distribution of action-feature rows.
        bootstrap_advance_noop_adversary = bool(
            bc_extra.get("nn_bootstrap_advance_noop_adversary", True)
        )
    else:
        extract_action_features_from_state = None  # type: ignore
        AdversaryAction = None  # type: ignore
        action_top_k = 0
        bootstrap_advance_noop_adversary = False

    for local_index, record in enumerate(records):
        state = state_loader(record)
        root_player = str(record["root_player"])
        if root_player != "controller":
            raise ValueError("batched target path currently supports controller roots only")

        mcts.clear_search_state(drop_scratch=False)
        root_node_raw = record.get("root_node_id_override", None)
        if root_node_raw is None:
            root_node_raw = record.get("root_id", local_index)
        root_node_id = int(root_node_raw)
        root = MCTSNode(
            player="controller",
            node_id=int(root_node_id),
            depth=int(record.get("root_depth", 0)),
            parent=None,
        )
        mcts._root = root

        decision_state, _ = mcts._decision_state(root, state, "controller")
        root_cost = float(mcts._state_cost(decision_state))
        root_time = float(decision_state.simulator._time)
        root.state_cost = root_cost
        root.sim_time = root_time

        decision_snapshot, decision_stats = mcts._snapshot_state_and_stats(decision_state)
        actions_by_index, mask_t = mcts._actions_and_mask(
            decision_state,
            "controller",
            forbidden_stop_ids=None,
        )
        valid_mask = [bool(x) for x in mask_t.tolist()]
        valid_indices = [
            i
            for i, ok in enumerate(valid_mask)
            if ok and actions_by_index[i] is not None
        ]
        if not valid_indices:
            targets[local_index] = 0.0
            continue

        alias_to_canon, _canon_to_aliases, canonical_indices = mcts._canonicalize_action_indices(
            player="controller",
            actions_by_index=actions_by_index,
            valid_indices=valid_indices,
        )
        root_infos[local_index] = {
            "valid_indices": list(valid_indices),
            "alias_to_canon": dict(alias_to_canon),
            "action_values": [float("-inf")] * len(actions_by_index),
            "canonical_q": {},
        }

        for cidx in canonical_indices:
            action = actions_by_index[cidx]
            if action is None:
                continue

            leaf_state = mcts._scratch_restore(decision_snapshot, decision_stats)
            leaf_state = mcts._env.apply_controller_action_only(
                leaf_state,
                action,
                inplace=True,
                fast_forward=False,
            )
            leaf_cost = float(mcts._state_cost(leaf_state))
            reward = float(mcts._transition_reward(root_cost, leaf_cost))
            leaf_time = float(leaf_state.simulator._time)
            discount_time = getattr(leaf_state.stats, "transition_discount_time", None)
            if discount_time is None:
                discount_time = leaf_time
            discount = float(mcts._time_discount(float(discount_time), root_time))
            next_player = "adversary"
            inputs = dnn_infer.build_model_inputs(
                leaf_state,
                next_player,
                torch.device("cpu"),
                build_action_mask_flag=False,
            )
            if use_action_features_at_bootstrap and extract_action_features_from_state is not None:
                # Snapshot leaf BEFORE action features modify scratch state.
                leaf_snap = leaf_stats = None
                try:
                    leaf_snap, leaf_stats = mcts._snapshot_state_and_stats(leaf_state)
                    # Optionally advance one noop adversary action so the
                    # state is controller-to-act and matches the action-
                    # feature training distribution. AdversaryAction is
                    # imported above when this branch is taken.
                    feature_state = leaf_state
                    if bootstrap_advance_noop_adversary and AdversaryAction is not None:
                        try:
                            noop = AdversaryAction(requests=[], stop_decode_ids=[])
                            feature_state = mcts._env.apply_adversary_action_only(
                                leaf_state, noop, inplace=True
                            )
                        except Exception:
                            feature_state = leaf_state
                    act_row, _act_names = extract_action_features_from_state(
                        feature_state,
                        mcts=mcts,
                        env=mcts._env,
                        top_k=int(action_top_k),
                    )
                except Exception as exc:
                    act_row = []
                    print(f"[bellman_convergence] action feature extraction failed: {exc!r}", flush=True)
                finally:
                    if leaf_snap is not None and leaf_stats is not None:
                        try:
                            _ = mcts._scratch_restore(leaf_snap, leaf_stats)
                        except Exception:
                            pass
            else:
                act_row = []
            child_jobs.append(
                (
                    int(local_index),
                    int(cidx),
                    float(reward),
                    float(discount),
                    float(leaf_cost),
                    float(leaf_time),
                )
            )
            child_inputs.append(inputs)
            child_players.append(next_player)
            child_action_rows.append(act_row)

    bootstrap_values = _predict_bootstrap_inputs_batch(
        model=bootstrap_model,
        inputs_list=child_inputs,
        players=child_players,
        child_batch_size=int(child_batch_size),
        action_feature_rows=(child_action_rows if use_action_features_at_bootstrap else None),
    )
    if len(bootstrap_values) != len(child_jobs):
        raise RuntimeError(
            f"bootstrap batch returned {len(bootstrap_values)} values for {len(child_jobs)} jobs"
        )

    for job, bootstrap_value in zip(child_jobs, bootstrap_values):
        local_index, cidx, reward, discount, _leaf_cost, _leaf_time = job
        info = root_infos[int(local_index)]
        if info is None:
            continue
        q = float(reward + discount * float(bootstrap_value))
        info["canonical_q"][int(cidx)] = q

    for local_index, info in enumerate(root_infos):
        if targets[local_index] is not None:
            continue
        if info is None:
            targets[local_index] = 0.0
            continue
        action_values = info["action_values"]
        for alias_idx, canon_idx in info["alias_to_canon"].items():
            action_values[int(alias_idx)] = float(
                info["canonical_q"].get(int(canon_idx), float("-inf"))
            )
        best_idx = mcts._select_depth1_best_action_index(
            root_player="controller",
            valid_indices=info["valid_indices"],
            action_values=action_values,
        )
        targets[local_index] = float(action_values[int(best_idx)])

    mcts.clear_search_state(drop_scratch=True)
    return [float(x) for x in targets]


def _init_target_worker(
    model_cfg: SelfModelTestConfig,
    extra_config: dict[str, Any],
    bootstrap_model: Any,
    bootstrap_version: int,
    child_batch_size: int,
) -> None:
    """Initialize a target worker with its own simulator/MCTS state."""

    global _TARGET_WORKER_STATE_LOADER
    global _TARGET_WORKER_BOOTSTRAP_MODEL
    global _TARGET_WORKER_BOOTSTRAP_VERSION
    global _TARGET_WORKER_CHILD_BATCH_SIZE
    _limit_native_threads(int((extra_config or {}).get("target_worker_threads", 1)))
    _TARGET_WORKER_STATE_LOADER = RootStateLoader(model_cfg)
    _TARGET_WORKER_STATE_LOADER._bellman_extra_config = dict(extra_config or {})
    _TARGET_WORKER_BOOTSTRAP_MODEL = bootstrap_model
    _TARGET_WORKER_BOOTSTRAP_VERSION = int(bootstrap_version)
    _TARGET_WORKER_CHILD_BATCH_SIZE = int(child_batch_size)


def _compute_controller_targets_worker(
    item: tuple[int, list[dict[str, Any]]],
) -> tuple[int, list[float]]:
    """Compute one record chunk inside a forked target worker."""

    chunk_index, records = item
    if _TARGET_WORKER_STATE_LOADER is None:
        raise RuntimeError("target worker state loader was not initialized")
    if _TARGET_WORKER_BOOTSTRAP_MODEL is None:
        raise RuntimeError("target worker bootstrap model was not initialized")
    targets = _compute_controller_targets_for_chunk(
        records=records,
        state_loader=_TARGET_WORKER_STATE_LOADER,
        bootstrap_model=_TARGET_WORKER_BOOTSTRAP_MODEL,
        bootstrap_version=int(_TARGET_WORKER_BOOTSTRAP_VERSION),
        child_batch_size=int(_TARGET_WORKER_CHILD_BATCH_SIZE),
    )
    gc.collect()
    return int(chunk_index), targets


def _split_records_for_target_workers(
    records: Sequence[dict[str, Any]],
    *,
    chunk_roots: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Split records into ordered chunks for process-parallel target builds."""

    chunks: list[tuple[int, list[dict[str, Any]]]] = []
    size = max(1, int(chunk_roots))
    for chunk_index, start in enumerate(range(0, len(records), size)):
        end = min(len(records), start + size)
        chunks.append((int(chunk_index), list(records[start:end])))
    return chunks


def _prepare_model_for_target_workers(model: Any) -> Any:
    """Drop runtime-only caches before sending a model to worker processes."""

    cache = getattr(model, "runtime_prediction_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    return model


def compute_mcts_targets_batched_controller(
    *,
    records: Sequence[dict[str, Any]],
    state_loader: RootStateLoader,
    bootstrap_model: Any,
    bootstrap_version: int,
    model_version: int,
    output_dir: Path,
    split_name: str,
    cache_targets: bool,
    chunk_roots: int,
    child_batch_size: int,
    target_num_processes: int,
    worker_model_cfg: SelfModelTestConfig | None = None,
    worker_extra_config: dict[str, Any] | None = None,
    target_maxtasks_per_child: int | None = None,
    target_mp_start_method: str = "spawn",
) -> list[float]:
    """Compute T[V] for controller roots using batched child-state V calls."""

    targets: list[float]
    chunks = _split_records_for_target_workers(records, chunk_roots=int(chunk_roots))
    process_count = min(max(1, int(target_num_processes)), max(1, len(chunks)))
    maxtasks = (
        int(target_maxtasks_per_child)
        if target_maxtasks_per_child is not None and int(target_maxtasks_per_child) > 0
        else None
    )
    if process_count <= 1:
        targets = []
        for _chunk_index, chunk_records in chunks:
            targets.extend(
                _compute_controller_targets_for_chunk(
                    records=chunk_records,
                    state_loader=state_loader,
                    bootstrap_model=bootstrap_model,
                    bootstrap_version=int(bootstrap_version),
                    child_batch_size=int(child_batch_size),
                )
            )
            gc.collect()
    else:
        start_method = str(target_mp_start_method or "spawn")
        if start_method not in mp.get_all_start_methods():
            raise RuntimeError(
                f"multiprocessing start method {start_method!r} is not available"
            )
        if worker_model_cfg is None:
            raise RuntimeError(
                "parallel target computation requires a worker SelfModelTestConfig"
            )
        print(
            "[bellman_convergence] "
            f"computing {split_name} targets with {process_count} processes, "
            f"{len(chunks)} chunks, chunk_roots={int(chunk_roots)}, "
            f"start_method={start_method}",
            flush=True,
        )
        ctx = mp.get_context(start_method)
        try:
            with ctx.Pool(
                processes=int(process_count),
                initializer=_init_target_worker,
                initargs=(
                    worker_model_cfg,
                    dict(worker_extra_config or {}),
                    _prepare_model_for_target_workers(bootstrap_model),
                    int(bootstrap_version),
                    int(child_batch_size),
                ),
                maxtasksperchild=maxtasks,
            ) as pool:
                results = pool.map(_compute_controller_targets_worker, chunks)
        finally:
            gc.collect()
        results.sort(key=lambda item: int(item[0]))
        targets = []
        for _chunk_index, chunk_targets in results:
            targets.extend(float(x) for x in chunk_targets)

    if bool(cache_targets):
        save_cached_targets(
            target_cache_path(
                output_dir,
                model_version=int(model_version),
                bootstrap_version=int(bootstrap_version),
                split_name=str(split_name),
            ),
            targets=targets,
            metadata={
                "split_name": str(split_name),
                "model_version": int(model_version),
                "bootstrap_version": int(bootstrap_version),
                "source": "batched_controller_mcts_dnn_search",
                "use_model_bootstrap": True,
                "chunk_roots": int(chunk_roots),
                "child_batch_size": int(child_batch_size),
                "target_num_processes": int(process_count),
                "target_maxtasks_per_child": maxtasks,
                "target_mp_start_method": str(target_mp_start_method),
            },
        )
    return targets


def compute_mcts_targets(
    *,
    records: Sequence[dict[str, Any]],
    state_loader: RootStateLoader,
    bootstrap_model: Any | None,
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

    extra_cfg = getattr(state_loader, "cfg", None)
    # RootStateLoader does not expose cfg in older harnesses; use environment
    # extra config from records path via caller fallback below.
    del extra_cfg
    all_controller = all(str(record.get("root_player", "")) == "controller" for record in records)
    if bool(use_bootstrap) and all_controller:
        # Pull free-form settings from the output config if available through a
        # private attribute attached by bellman_convergence below.
        bc_extra = getattr(state_loader, "_bellman_extra_config", {}) or {}
        if bool(bc_extra.get("use_batched_bootstrap_targets", True)):
            return compute_mcts_targets_batched_controller(
                records=records,
                state_loader=state_loader,
                bootstrap_model=bootstrap_model,
                bootstrap_version=int(bootstrap_version),
                model_version=int(model_version),
                output_dir=output_dir,
                split_name=str(split_name),
                cache_targets=bool(cache_targets),
                chunk_roots=int(bc_extra.get("bootstrap_batch_roots", 32)),
                child_batch_size=int(bc_extra.get("bootstrap_child_batch_size", 4096)),
                target_num_processes=int(bc_extra.get("target_num_processes", 1)),
                worker_model_cfg=getattr(state_loader, "_bellman_worker_model_cfg", None),
                worker_extra_config=dict(bc_extra),
                target_maxtasks_per_child=bc_extra.get("target_maxtasks_per_child", None),
                target_mp_start_method=str(bc_extra.get("target_mp_start_method", "spawn")),
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
    # If the previous iteration produced an NN state dict, warm-start the
    # next iteration's network from it. This dramatically reduces
    # per-iteration variance in the trained model and accelerates
    # convergence of the same-version Bellman residual.
    if int(model_version) > 1:
        prev_dir = cfg.output_dir / f"Model_Version{int(model_version) - 1}"
        prev_state = prev_dir / "neural_value_state_dict.pt"
        if prev_state.exists():
            extra = dict(model_cfg.extra_config or {})
            extra["nn_warm_start_state_dict_path"] = str(prev_state)
            model_cfg = replace(model_cfg, extra_config=extra)
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
            "target_num_processes": int(cfg.target_num_processes),
            "extra_config": dict(cfg.extra_config or {}),
        },
    )

    state_loader = RootStateLoader(model0_cfg)
    bellman_extra_config = dict(cfg.extra_config or {})
    bellman_extra_config["target_num_processes"] = int(cfg.target_num_processes)
    state_loader._bellman_extra_config = bellman_extra_config
    state_loader._bellman_worker_model_cfg = model0_cfg
    previous_model: Any | None = None
    try:
        for model_version in range(1, int(cfg.num_versions) + 1):
            bootstrap_version = int(model_version) - 1
            model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
            model_dir.mkdir(parents=True, exist_ok=True)

            train_targets = compute_mcts_targets(
                records=train_records,
                state_loader=state_loader,
                bootstrap_model=previous_model,
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
                bootstrap_version=bootstrap_version,
                model_version=int(model_version),
                output_dir=cfg.output_dir,
                split_name="eval",
                cache_targets=bool(cfg.cache_targets),
                reuse_record_targets_for_v0=bool(cfg.reuse_record_targets_for_v0),
            )

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

            previous_model = model
    finally:
        state_loader.close()


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
        "--target-num-processes",
        type=int,
        default=1,
        help="Number of forked worker processes for bootstrapped target computation.",
    )
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
        target_num_processes=args.target_num_processes,
        extra_config=extra_config,
    )


def main() -> None:
    """CLI entrypoint."""

    bellman_convergence(parse_args())


if __name__ == "__main__":
    main()
