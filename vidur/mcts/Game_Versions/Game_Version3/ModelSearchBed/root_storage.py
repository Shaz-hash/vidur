"""Local root-state storage for GV3 model architecture experiments.

This module is the first stage of the ModelSearchBed workflow.

Goal:
    Generate a fixed local dataset of GV3 frontier/root states once, store the
    simulator state plus debugging trace, and attach a first-layer/no-bootstrap
    target value. Later model-search scripts can reload this fixed dataset and
    test different feature builders or model architectures without
    regenerating roots.

Important scope:
    This file is local-machine only. It should not use the distributed network
    worker path, SSH, or remote EC2 orchestration.

Target semantics:
    For this search bed, targets are first-layer Bellman/tree-search values.
    That means model bootstrap must be disabled when computing labels:

        target = immediate_reward + discount * 0.0

    For adversary roots, the target is still controller-perspective value after
    the adversary action and the controller response logic used by GV3 depth-1
    search. The stored value remains in the same controller-perspective scale as
    the rest of GV3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from ....game_types import AdversaryAction, ControllerAction
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG
from ..DNN.history_root import HistoryRootGenerator
from ..logger.mctsDNN_logger import (
    DNNMCTSIterationLogger,
    _j,
    _safe_float,
    _safe_int,
)
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import _build_env_and_simulator, _set_global_seeds


SCHEMA_VERSION = 1
DEFAULT_NONZERO_EPS = 1e-9
NUMPY_SEED_MODULUS = 2**32


def _normalize_numpy_seed(seed: int) -> int:
    """Return a deterministic seed accepted by NumPy's legacy RNG."""

    return int(seed) % int(NUMPY_SEED_MODULUS)


@dataclass(frozen=True)
class RootStorageConfig:
    """Configuration for generating and storing local root states.

    `num_roots` is the desired final number of stored roots. The generator may
    inspect more candidate roots than this so it can satisfy
    `min_nonzero_target_ratio`.
    """

    output_dir: Path
    num_roots: int = 10_000
    max_candidate_roots: int = 100_000
    candidate_batch_size: int = 1_024
    generation_batch_size: int = 32
    min_nonzero_target_ratio: float = 0.50
    nonzero_eps: float = DEFAULT_NONZERO_EPS
    target_abs_threshold: float = DEFAULT_NONZERO_EPS
    history_hops_min: int = 0
    history_hops_max: int = 200
    history_max_total_steps: int = 20_000
    max_children_per_expand: int | None = None
    shard_size: int = 512
    seed: int = 2027
    game_id: int = 0
    start_root_id: int = 0
    start_depth: int = 0
    start_player: str = "adversary"
    root_player_filter: str = "controller"
    environment_lang: str = "python"
    include_history_trace_logs: bool = True
    allow_duplicate_history_fallback: bool = False
    deduplicate_history_signatures: bool = False
    history_signature_cache_size: int = 10_000
    num_processes: int = 1
    worker_roots_per_task: int = 1024
    max_processes_per_interval: int = 4
    allow_partial_results: bool = False
    shared_seen_signatures: Any | None = None
    shared_seen_lock: Any | None = None
    exclude_signature_dataset_dirs: tuple[Path, ...] = ()
    progress_every: int = 1000


@dataclass
class RootStorageStats:
    """Running counters used while building the fixed root dataset."""

    candidates_seen: int = 0
    roots_stored: int = 0
    nonzero_roots_stored: int = 0
    zero_roots_stored: int = 0
    shards_written: int = 0


@dataclass
class _RootStorageInterval:
    """Parent-side interval state for uniqueness/quota scheduling."""

    interval_id: int
    target_roots: int
    target_selected: int
    stored_roots: int = 0
    selected_roots: int = 0
    other_roots: int = 0
    shared_seen_signatures: Any | None = None
    shared_seen_lock: Any | None = None

    @property
    def remaining_roots(self) -> int:
        return max(0, int(self.target_roots) - int(self.stored_roots))

    @property
    def remaining_selected(self) -> int:
        return max(0, int(self.target_selected) - int(self.selected_roots))

    @property
    def complete(self) -> bool:
        return int(self.remaining_roots) <= 0 and int(self.remaining_selected) <= 0


@dataclass(frozen=True)
class PreparedHistoryRoot:
    """A candidate root generated from GV3 history-root machinery."""

    root_id: int
    root_player: str
    root_depth: int
    history_hops: int
    state: Any
    history_signature: Any | None = None
    history_trace_logs: list[dict[str, Any]] | None = None
    root_node_id_override: int | None = None
    pre_controller_snapshot: Any | None = None
    pre_controller_stats: Any | None = None


@dataclass(frozen=True)
class FirstLayerTarget:
    """Depth-1/no-bootstrap label and selected action metadata."""

    target_value: float
    best_action_index: int
    best_action_repr: str
    best_reward: float
    best_discount: float
    best_bootstrap: float
    best_child_cost: float
    best_child_time: float = 0.0
    model_state_snapshot: Any | None = None
    model_state_stats: Any | None = None


class _InMemoryIterationLogger:
    """A minimal in-memory version of `DNNMCTSIterationLogger.log_expand`."""

    FIELDS = DNNMCTSIterationLogger.FIELDS

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def close(self) -> None:
        return None

    def log_expand(
        self,
        *,
        game_id: int,
        root_id: int,
        sim_iteration: int,
        root_depth: int,
        root_node_id: int,
        root_player: str,
        node_depth: int,
        parent_node_id: int | None,
        node_id: int,
        player_to_act: str,
        player_acted_to_create_this_node: str,
        action_index: int | None,
        action_repr: str,
        prior: float,
        model_prior_json: str = "[]",
        normalized_prior_json: str = "[]",
        reward: float = 0.0,
        nn_called: bool = False,
        num_valid_actions: int = 0,
        unique_actions: int = 0,
        nn_value_controller: float | None = None,
        objective_cost: float = 0.0,
        state_snapshot: dict[str, Any] | None = None,
        adversary_action_json: str = "[]",
        adversary_prefill_slos_json: str = "[]",
        adversary_prefill_deadlines_by_id_json: str = "{}",
        adversary_decode_slos_json: str = "[]",
        controller_token_budget: int | str = "",
        controller_selected_ids_json: str = "[]",
        controller_allocations_json: str = "{}",
        controller_prefill_allocations_json: str = "{}",
        controller_decode_allocations_json: str = "{}",
        controller_prefill_total: int = 0,
        controller_decode_total: int = 0,
        controller_heuristic: str = "",
        controller_strategy: str = "",
        phase: str = "expand",
        decision_state_time: float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        stage_total_time: float | None = None,
    ) -> None:
        snap = state_snapshot or {}
        sim_time = _safe_float(snap.get("sim_time", 0.0))
        st = sim_time if start_time is None else _safe_float(start_time, sim_time)
        et = sim_time if end_time is None else _safe_float(end_time, sim_time)
        dt = (
            _safe_float(stage_total_time, max(0.0, et - st))
            if stage_total_time is not None
            else max(0.0, et - st)
        )

        row = {field: "" for field in self.FIELDS}
        row.update(
            {
                "game_id": int(game_id),
                "root_id": int(root_id),
                "sim_iteration": int(sim_iteration),
                "root_depth": int(root_depth),
                "root_node_id": int(root_node_id),
                "root_player": str(root_player),
                "phase": str(phase),
                "node_depth": int(node_depth),
                "parent_node_id": "" if parent_node_id is None else str(int(parent_node_id)),
                "node_id": int(node_id),
                "player_acted_to_create_this_node": str(player_acted_to_create_this_node),
                "player_to_act_in_this_node": str(player_to_act),
                "action_index": "" if action_index is None else int(action_index),
                "action_repr": str(action_repr),
                "prior": _safe_float(prior),
                "model_prior_json": str(model_prior_json),
                "normalized_prior_json": str(normalized_prior_json),
                "reward": _safe_float(reward),
                "nn_called": bool(nn_called),
                "num_valid_actions": int(num_valid_actions),
                "unique_actions": int(unique_actions),
                "nn_value_controller": "" if nn_value_controller is None else _safe_float(nn_value_controller),
                "objective_cost": _safe_float(objective_cost),
                "model_top5_actions_json": "[]",
                "mcts_top5_actions_json": "[]",
                "sim_time": sim_time,
                "decision_state_time": (
                    sim_time if decision_state_time is None else _safe_float(decision_state_time, sim_time)
                ),
                "start_time": st,
                "end_time": et,
                "stage_total_time": dt,
                "requests_in_system": snap.get("requests_in_system", 0),
                "requests_generated": snap.get("requests_generated", 0),
                "requests_completed": snap.get("requests_completed", 0),
                "slo_violations": snap.get("slo_violations", 0),
                "total_lateness": snap.get("total_lateness", snap.get("avg_lateness", 0.0)),
                "avg_lateness": snap.get("avg_lateness", 0.0),
                "state_active_ids": _j(snap.get("active_request_ids", [])),
                "state_waiting_ids": _j(snap.get("waiting_request_ids", [])),
                "state_completed_request_ids": _j(snap.get("completed_request_ids", [])),
                "state_dropped_request_ids": _j(snap.get("dropped_request_ids", [])),
                "state_stopped_decode_request_ids": _j(snap.get("stopped_decode_request_ids", [])),
                "state_pending_adv_tick": bool(snap.get("pending_adv_tick", False)),
                "state_last_adv_tick": snap.get("last_adv_tick", ""),
                "state_decode_credit_balance": _safe_int(
                    snap.get("decode_credit_balance", snap.get("decode_credit_available", 0))
                ),
                "state_decode_tokens_counted_by_id": _j(snap.get("decode_tokens_counted_by_id", {})),
                "state_violated_request_ids": _j(snap.get("violated_request_ids", [])),
                "state_per_request_prefill_lateness_by_id": _j(
                    snap.get("per_request_prefill_lateness_by_id", {})
                ),
                "state_per_request_decode_lateness_by_id": _j(
                    snap.get("per_request_decode_lateness_by_id", {})
                ),
                "adversary_requests": str(adversary_action_json),
                "adversary_prefill_slos": str(adversary_prefill_slos_json),
                "adversary_prefill_deadlines_by_id": str(adversary_prefill_deadlines_by_id_json),
                "adversary_decode_slos": str(adversary_decode_slos_json),
                "controller_token_budget": controller_token_budget,
                "controller_selected_ids": str(controller_selected_ids_json),
                "controller_allocations": str(controller_allocations_json),
                "controller_prefill_allocations": str(controller_prefill_allocations_json),
                "controller_decode_allocations": str(controller_decode_allocations_json),
                "controller_prefill_total": _safe_int(controller_prefill_total),
                "controller_decode_total": _safe_int(controller_decode_total),
                "controller_heuristic": str(controller_heuristic),
                "controller_strategy": str(controller_strategy),
            }
        )
        self.rows.append(row)


def build_storage_config(
    *,
    output_dir: str | Path,
    num_roots: int = 10_000,
    max_candidate_roots: int = 100_000,
    candidate_batch_size: int = 1_024,
    generation_batch_size: int = 32,
    min_nonzero_target_ratio: float = 0.50,
    nonzero_eps: float = DEFAULT_NONZERO_EPS,
    target_abs_threshold: float | None = None,
    history_hops_min: int = 0,
    history_hops_max: int = 200,
    history_max_total_steps: int = 20_000,
    max_children_per_expand: int | None = None,
    shard_size: int = 512,
    seed: int = 2027,
    game_id: int = 0,
    start_root_id: int = 0,
    start_depth: int = 0,
    start_player: str = "adversary",
    root_player_filter: str = "controller",
    environment_lang: str = "python",
    include_history_trace_logs: bool = True,
    allow_duplicate_history_fallback: bool = False,
    deduplicate_history_signatures: bool = False,
    history_signature_cache_size: int = 10_000,
    num_processes: int = 1,
    worker_roots_per_task: int = 1024,
    max_processes_per_interval: int = 4,
    allow_partial_results: bool = False,
    shared_seen_signatures: Any | None = None,
    shared_seen_lock: Any | None = None,
    exclude_signature_dataset_dirs: Iterable[str | Path] = (),
    progress_every: int = 1000,
) -> RootStorageConfig:
    """Create a validated config for local root-state storage.

    This should be the only place where CLI/default values are normalized.
    Validation should reject impossible ratios, negative hop ranges, unsupported
    environments, and invalid shard sizes.
    """

    if int(num_roots) <= 0:
        raise ValueError("num_roots must be > 0")
    if int(max_candidate_roots) < int(num_roots):
        raise ValueError("max_candidate_roots must be >= num_roots")
    if int(candidate_batch_size) <= 0:
        raise ValueError("candidate_batch_size must be > 0")
    if int(generation_batch_size) <= 0:
        raise ValueError("generation_batch_size must be > 0")
    if not (0.0 <= float(min_nonzero_target_ratio) <= 1.0):
        raise ValueError("min_nonzero_target_ratio must be in [0, 1]")
    if float(nonzero_eps) < 0.0:
        raise ValueError("nonzero_eps must be >= 0")
    threshold = float(nonzero_eps if target_abs_threshold is None else target_abs_threshold)
    if threshold < 0.0:
        raise ValueError("target_abs_threshold must be >= 0")
    if int(history_hops_min) < 0:
        raise ValueError("history_hops_min must be >= 0")
    if int(history_hops_max) < int(history_hops_min):
        raise ValueError("history_hops_max must be >= history_hops_min")
    if int(history_max_total_steps) <= 0:
        raise ValueError("history_max_total_steps must be > 0")
    if int(shard_size) <= 0:
        raise ValueError("shard_size must be > 0")
    if str(start_player) not in {"adversary", "controller"}:
        raise ValueError("start_player must be 'adversary' or 'controller'")
    if str(environment_lang) != "python":
        raise ValueError("root_storage currently supports only environment_lang='python'")
    if max_children_per_expand is not None and int(max_children_per_expand) <= 0:
        raise ValueError("max_children_per_expand must be > 0 when provided")
    if int(num_processes) <= 0:
        raise ValueError("num_processes must be > 0")
    if int(history_signature_cache_size) < 0:
        raise ValueError("history_signature_cache_size must be >= 0")
    if int(worker_roots_per_task) <= 0:
        raise ValueError("worker_roots_per_task must be > 0")
    if int(max_processes_per_interval) <= 0:
        raise ValueError("max_processes_per_interval must be > 0")
    if int(progress_every) < 0:
        raise ValueError("progress_every must be >= 0")
    if str(root_player_filter) not in {"controller", "adversary", "any"}:
        raise ValueError("root_player_filter must be 'controller', 'adversary', or 'any'")

    return RootStorageConfig(
        output_dir=Path(output_dir).expanduser(),
        num_roots=int(num_roots),
        max_candidate_roots=int(max_candidate_roots),
        candidate_batch_size=int(candidate_batch_size),
        generation_batch_size=int(generation_batch_size),
        min_nonzero_target_ratio=float(min_nonzero_target_ratio),
        nonzero_eps=float(nonzero_eps),
        target_abs_threshold=float(threshold),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        history_max_total_steps=int(history_max_total_steps),
        max_children_per_expand=(
            None if max_children_per_expand is None else int(max_children_per_expand)
        ),
        shard_size=int(shard_size),
        seed=int(seed),
        game_id=int(game_id),
        start_root_id=int(start_root_id),
        start_depth=int(start_depth),
        start_player=str(start_player),
        root_player_filter=str(root_player_filter),
        environment_lang=str(environment_lang),
        include_history_trace_logs=bool(include_history_trace_logs),
        allow_duplicate_history_fallback=bool(allow_duplicate_history_fallback),
        deduplicate_history_signatures=bool(deduplicate_history_signatures),
        history_signature_cache_size=int(history_signature_cache_size),
        num_processes=int(num_processes),
        worker_roots_per_task=int(worker_roots_per_task),
        max_processes_per_interval=int(max_processes_per_interval),
        allow_partial_results=bool(allow_partial_results),
        shared_seen_signatures=shared_seen_signatures,
        shared_seen_lock=shared_seen_lock,
        exclude_signature_dataset_dirs=tuple(
            Path(path).expanduser() for path in exclude_signature_dataset_dirs
        ),
        progress_every=int(progress_every),
    )


def make_local_env_and_mcts(cfg: RootStorageConfig) -> tuple[Any, Any]:
    """Build the local GV3 virtual environment and MCTS object.

    This function should reuse the same codepath as GV3 local sample generation,
    but keep everything on the local machine. It should force Python environment
    mode for the first implementation.
    """

    seed = _normalize_numpy_seed(int(cfg.seed))
    _set_global_seeds(seed, torch_deterministic=False)
    mp_cfg = replace(
        DEFAULT_MULTIPROCESS_TRAINING_CONFIG,
        environment_lang="python",
        use_virtual_env=True,
        num_processes=1,
        max_concurrent_selfplay_workers=1,
        max_workers_per_interval=1,
        roots_per_generation=max(int(cfg.num_roots), int(cfg.candidate_batch_size)),
        history_seed=seed,
        history_hops_min=int(cfg.history_hops_min),
        history_hops_max=int(cfg.history_hops_max),
        history_max_total_steps=int(cfg.history_max_total_steps),
        history_root_batch_size=int(cfg.candidate_batch_size),
        history_allow_duplicate_root_fallback=bool(cfg.allow_duplicate_history_fallback),
        log_history_rows=bool(cfg.include_history_trace_logs),
    )
    _simulator, env, _constraints, explore_cfg = _build_env_and_simulator(
        mp_cfg,
        use_virtual_env=True,
    )
    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        rng=random.Random(seed),
    )
    return env, mcts


def _history_signature_key(sig: Any) -> Any:
    if isinstance(sig, str):
        return sig
    if isinstance(sig, (list, tuple)):
        return tuple(_history_signature_key(x) for x in sig)
    return sig


def load_history_signature_keys_from_dataset(dataset_dir: str | Path) -> set[Any]:
    """Load normalized history signatures from a stored root dataset.

    This streams shard-by-shard so pre-seeding uniqueness from a large dataset
    does not materialize all simulator snapshots at once.
    """

    root_dir = Path(dataset_dir).expanduser()
    manifest = root_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    out: set[Any] = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            shard_path = root_dir / str(entry["shard_path"])
            try:
                records = torch.load(shard_path, map_location="cpu", weights_only=False)
            except TypeError:
                records = torch.load(shard_path, map_location="cpu")
            for record in records:
                sig = record.get("history_signature", None)
                if sig is not None:
                    out.add(_history_signature_key(sig))
            del records
    return out


def _row_parent_id(row: dict[str, Any]) -> int | None:
    raw = row.get("parent_node_id", "")
    if raw in ("", None):
        return None
    return int(raw)


def _trace_rows_for_leaf(
    *,
    rows_by_node: dict[int, dict[str, Any]],
    leaf_node_id: int | None,
    root_id: int,
) -> list[dict[str, Any]]:
    if leaf_node_id is None:
        return []

    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    node_id: int | None = int(leaf_node_id)
    while node_id is not None:
        if node_id in seen:
            raise RuntimeError(f"cycle detected while reconstructing history trace at node_id={node_id}")
        seen.add(int(node_id))
        row = rows_by_node.get(int(node_id))
        if row is None:
            break
        copied = dict(row)
        copied["root_id"] = int(root_id)
        out.append(copied)
        node_id = _row_parent_id(row)

    out.reverse()
    return out


def _coerce_raw_history_root(raw: dict[str, Any]) -> PreparedHistoryRoot:
    return PreparedHistoryRoot(
        root_id=int(raw.get("root_id", 0)),
        root_player=str(raw.get("root_player", "adversary")),
        root_depth=int(raw.get("root_depth", 0)),
        history_hops=int(raw.get("history_hops", 0)),
        state=raw["root_state"],
        history_signature=raw.get("history_signature", None),
        history_trace_logs=None,
        root_node_id_override=raw.get("root_node_id_override", None),
        pre_controller_snapshot=raw.get("pre_controller_snapshot", None),
        pre_controller_stats=raw.get("pre_controller_stats", None),
    )


def generate_candidate_root_batches(
    cfg: RootStorageConfig,
    env: Any,
) -> Iterator[list[PreparedHistoryRoot]]:
    """Yield candidate roots in local batches with attached full trace rows."""

    history = HistoryRootGenerator(env=env)
    remaining = int(cfg.max_candidate_roots)
    next_root_id = int(cfg.start_root_id)
    batch_index = 0
    seen_signatures: set[Any] = set()
    if (
        bool(cfg.deduplicate_history_signatures)
        and cfg.shared_seen_signatures is None
        and cfg.exclude_signature_dataset_dirs
    ):
        for dataset_dir in cfg.exclude_signature_dataset_dirs:
            seen_signatures.update(load_history_signature_keys_from_dataset(dataset_dir))
    seen_signature_order: deque[Any] = deque()
    use_shared_dedup = (
        bool(cfg.deduplicate_history_signatures)
        and cfg.shared_seen_signatures is not None
    )
    use_bounded_dedup = (
        bool(cfg.deduplicate_history_signatures)
        and int(cfg.history_signature_cache_size) > 0
        and not bool(use_shared_dedup)
    )

    def remember_signature(sig: Any) -> None:
        key = _history_signature_key(sig)
        if key in seen_signatures:
            return
        seen_signatures.add(key)
        seen_signature_order.append(key)
        while len(seen_signature_order) > int(cfg.history_signature_cache_size):
            old = seen_signature_order.popleft()
            seen_signatures.discard(old)

    while remaining > 0:
        # Keep each history-root session small. A session retains frontier
        # snapshots in `active_path`, so using the full candidate batch here can
        # make every worker climb into multi-GB RSS before any shard flushes.
        request_count = min(
            int(cfg.candidate_batch_size),
            int(cfg.generation_batch_size),
            int(remaining),
        )
        trace_logger = _InMemoryIterationLogger() if cfg.include_history_trace_logs else None
        raw_batches = history.generate_roots_batch_iter(
            initial_state=None,
            start_player=str(cfg.start_player),
            start_depth=int(cfg.start_depth),
            num_roots=int(request_count),
            game_id=int(cfg.game_id),
            start_root_id=int(next_root_id),
            nontrivial_hops=0,
            seed=_normalize_numpy_seed(int(cfg.seed) + int(batch_index)),
            max_total_steps=int(cfg.history_max_total_steps),
            log_history=bool(cfg.include_history_trace_logs),
            min_history_hops=int(cfg.history_hops_min),
            max_history_hops=int(cfg.history_hops_max),
            max_children_per_expand=cfg.max_children_per_expand,
            batch_size=max(1, min(int(request_count), int(cfg.generation_batch_size))),
            initial_seen_signatures=(
                tuple(seen_signatures) if bool(use_bounded_dedup) else ()
            ),
            shared_seen_signatures=(
                cfg.shared_seen_signatures if bool(use_shared_dedup) else None
            ),
            shared_seen_lock=(
                cfg.shared_seen_lock if bool(use_shared_dedup) else None
            ),
            allow_duplicate_fallback=bool(cfg.allow_duplicate_history_fallback),
            history_trace_logger=trace_logger,
        )

        emitted_this_batch = 0
        for raw_batch in raw_batches:
            rows_by_node: dict[int, dict[str, Any]] = {}
            if trace_logger is not None:
                rows_by_node = {
                    int(row["node_id"]): dict(row)
                    for row in trace_logger.rows
                    if row.get("node_id", "") != ""
                }

            roots: list[PreparedHistoryRoot] = []
            for raw in raw_batch:
                root = _coerce_raw_history_root(raw)
                trace_logs = _trace_rows_for_leaf(
                    rows_by_node=rows_by_node,
                    leaf_node_id=raw.get("history_log_node_id", None),
                    root_id=int(root.root_id),
                )
                roots.append(
                    replace(
                        root,
                        history_trace_logs=trace_logs,
                    )
                )
                if bool(use_bounded_dedup) and root.history_signature is not None:
                    remember_signature(root.history_signature)

            emitted_this_batch += len(roots)
            if roots:
                yield roots

        if emitted_this_batch <= 0:
            break
        next_root_id += int(emitted_this_batch)
        remaining -= int(emitted_this_batch)
        batch_index += 1



def collect_history_trace_for_root(root: PreparedHistoryRoot) -> list[dict[str, Any]]:
    """Return full MCTS-iteration logs for the path to this root.

    The stored rows should be the same kind of rows written by
    `DNNMCTSIterationLogger` into `history_mcts_iter.csv`. The goal is that a
    stored root can later be debugged by exporting `history_trace_logs` to CSV
    and running the existing GV3 history trace validators on it.

    Each row should preserve the full logger schema, including identifiers,
    tree structure, player/action metadata, simulator timing, active/completed
    request state, SLO/cost counters, queue/request snapshots, transition timing,
    and any GV3-specific columns required by the trace tests.

    This function should not invent a reduced schema. If the logger adds a new
    column later, the stored trace rows should keep that column too.
    """

    return [dict(row) for row in (root.history_trace_logs or [])]


def _build_root_decision_state_for_adversary(
    *,
    env: Any,
    current_state: Any,
    pre_controller_snapshot: Any | None,
    pre_controller_stats: Any | None,
) -> Any:
    if pre_controller_snapshot is None or pre_controller_stats is None:
        return current_state

    get_src = getattr(env, "_v2_missed_adv_source", None)
    if not callable(get_src):
        print("The missed source function is not working!!!")
        return current_state

    try:
        miss_src = int(get_src(current_state))
    except Exception:
        print("Exception occured while getting MISSED source for the current state !!!")
        return current_state

    if miss_src != 1:
        return current_state

    clone_fn = getattr(env, "clone_state_from_snapshot", None)
    if not callable(clone_fn):
        print("The clone function is not working!!!")
        return current_state

    try:
        decision_state = clone_fn(pre_controller_snapshot, pre_controller_stats)
    except Exception:
        return current_state

    try:
        tick = float(env._v2_current_adv_tick(current_state))
        t_dec = float(decision_state.simulator._time)
        if t_dec + 1e-9 < tick:
            decision_state.simulator._set_time(tick)
        else : 
            print("This should not occur , decision state cannot have the time after the tick !")
    except Exception:
        pass

    try:
        replay_ids = set(
            int(k)
            for k in env._build_request_lookup(decision_state.simulator, state=decision_state).keys()
        )
        live_ids = set(
            int(k)
            for k in env._build_request_lookup(current_state.simulator, state=current_state).keys()
        )
    except Exception:
        return decision_state

    forbidden = replay_ids - live_ids
    if not forbidden:
        return decision_state

    s = decision_state.stats
    s.active_request_ids = {int(rid) for rid in s.active_request_ids if int(rid) not in forbidden}
    for rid in forbidden:
        rid = int(rid)
        s.decode_tokens_counted.pop(rid, None)
        s.decode_next_deadline_by_id.pop(rid, None)
        s.per_request_prefill_lateness.pop(rid, None)
        s.per_request_decode_lateness.pop(rid, None)
        s.violated_request_ids.discard(rid)
        s.dropped_request_ids.discard(rid)
        s.stopped_decode_request_ids.discard(rid)
        s.prefill_lateness_finalized.discard(rid)
    return decision_state


def evaluate_first_layer_target(
    cfg: RootStorageConfig,
    mcts: Any,
    root: PreparedHistoryRoot,
) -> FirstLayerTarget:
    """Compute the first-layer/no-bootstrap target for one root.

    This must call GV3 MCTS/Bellman scoring with model bootstrap disabled.
    The result should identify the selected best action and expose the pieces
    used to form the target:

        target_value = best_reward + best_discount * 0.0

    `best_bootstrap` should therefore always be 0.0 for this storage bed.
    """

    search_state = root.state.fork(flag=False)
    root_player = str(root.root_player)
    if root_player == "adversary":
        search_state = _build_root_decision_state_for_adversary(
            env=mcts._env,
            current_state=search_state,
            pre_controller_snapshot=root.pre_controller_snapshot,
            pre_controller_stats=root.pre_controller_stats,
        )

    if hasattr(search_state.simulator, "snapshot_state"):
        model_state_snapshot = search_state.simulator.snapshot_state()
    else:
        model_state_snapshot = search_state.simulator.snapshot_state_fast()
    model_state_stats = search_state.stats.clone()

    out = mcts.search_dnn(
        dnn_model=None,
        rootState=search_state,
        root_player=root_player,
        game_id=int(cfg.game_id),
        root_id=int(root.root_id),
        root_node_id_override=root.root_node_id_override,
        root_depth=int(root.root_depth),
        model_version=0,
        use_model_bootstrap=False,
        one_step_value_mode=True,
    )

    best_idx_raw = getattr(out, "best_action_index", None)
    best_action = getattr(out, "best_action", None)

    if best_idx_raw is None or best_action is None:
        print("Best index raw should not be none !!!!")
        return FirstLayerTarget(
            target_value=0.0,
            best_action_index=-1,
            best_action_repr="",
            best_reward=0.0,
            best_discount=1.0,
            best_bootstrap=0.0,
            best_child_cost=float(mcts._state_cost(search_state)),
            best_child_time=float(search_state.simulator._time),
            model_state_snapshot=model_state_snapshot,
            model_state_stats=model_state_stats,
        )

    return FirstLayerTarget(
        target_value=float(out.best_action_value),
        best_action_index=int(best_idx_raw),
        best_action_repr=repr(best_action),
        best_reward=float(out.best_reward),
        best_discount=float(out.best_discount),
        best_bootstrap=float(out.best_bootstrap),
        best_child_cost=float(out.best_child_cost),
        best_child_time=float(out.best_child_time),
        model_state_snapshot=model_state_snapshot,
        model_state_stats=model_state_stats,
    )


def pack_root_record(
    cfg: RootStorageConfig,
    root: PreparedHistoryRoot,
    target: FirstLayerTarget,
) -> dict[str, Any]:
    """Convert a root state and target into the on-disk record schema."""

    frontier_sim = root.state.simulator
    if hasattr(frontier_sim, "snapshot_state"):
        frontier_simulator_snapshot = frontier_sim.snapshot_state()
    else:
        frontier_simulator_snapshot = frontier_sim.snapshot_state_fast()

    simulator_snapshot = (
        target.model_state_snapshot
        if target.model_state_snapshot is not None
        else frontier_simulator_snapshot
    )
    stats = (
        target.model_state_stats.clone()
        if target.model_state_stats is not None
        else root.state.stats.clone()
    )

    record = {
        "schema_version": int(SCHEMA_VERSION),
        "environment_lang": str(cfg.environment_lang),
        "root_id": int(root.root_id),
        "root_player": str(root.root_player),
        "root_depth": int(root.root_depth),
        "history_hops": int(root.history_hops),
        "history_signature": root.history_signature,
        "history_trace_logs": collect_history_trace_for_root(root),
        "simulator_snapshot": simulator_snapshot,
        "stats": stats,
        "frontier_simulator_snapshot": frontier_simulator_snapshot,
        "frontier_stats": root.state.stats.clone(),
        "root_node_id_override": root.root_node_id_override,
        "pre_controller_snapshot": root.pre_controller_snapshot,
        "pre_controller_stats": (
            None if root.pre_controller_stats is None else root.pre_controller_stats.clone()
        ),
        "target_value": float(target.target_value),
        "best_action_index": int(target.best_action_index),
        "best_action_repr": str(target.best_action_repr),
        "best_reward": float(target.best_reward),
        "best_discount": float(target.best_discount),
        "best_bootstrap": float(target.best_bootstrap),
        "best_child_cost": float(target.best_child_cost),
        "best_child_time": float(target.best_child_time),
        "target_abs_threshold": float(cfg.target_abs_threshold),
    }
    record["is_nonzero_target"] = is_nonzero_target(record, cfg.nonzero_eps)
    record["is_selected_target"] = is_selected_target(record, cfg.target_abs_threshold)
    return record


def is_nonzero_target(record: dict[str, Any], eps: float) -> bool:
    """Return whether a stored target should count as nonzero.

    GV3 values are controller-perspective and are usually non-positive, so this
    should use absolute magnitude rather than checking for positive values.
    """

    return abs(float(record.get("target_value", 0.0))) > float(eps)


def is_selected_target(record: dict[str, Any], threshold: float) -> bool:
    """Return whether the target belongs to the high-signal quota bucket.

    With `target_abs_threshold=1.0`, this means `abs(target_value) >= 1.0`.
    Since controller values are non-positive in this setup, this is equivalent
    to `target_value <= -1.0` for the controller-only datasets.
    """

    return abs(float(record.get("target_value", 0.0))) >= float(threshold)


def should_keep_record(
    record: dict[str, Any],
    stats: RootStorageStats,
    cfg: RootStorageConfig,
) -> bool:
    """Decide whether to keep a candidate while enforcing dataset balance.

    The intended policy is:
        - keep enough selected-target roots to reach
          `cfg.min_nonzero_target_ratio`
        - keep enough other roots to fill the remainder
        - continue scanning candidates until both quotas are satisfied

    The selected bucket is controlled by `cfg.target_abs_threshold`. The old
    nonzero behavior is preserved when the threshold is left at `nonzero_eps`.
    """

    target_nonzero = int(math.ceil(float(cfg.num_roots) * float(cfg.min_nonzero_target_ratio)))
    target_zero = int(cfg.num_roots) - int(target_nonzero)
    if bool(record.get("is_selected_target", is_selected_target(record, cfg.target_abs_threshold))):
        return int(stats.nonzero_roots_stored) < int(target_nonzero)
    return int(stats.zero_roots_stored) < int(target_zero)


def root_player_matches_filter(cfg: RootStorageConfig, root_player: str) -> bool:
    """Return whether this root should be part of the stored dataset."""

    root_filter = str(cfg.root_player_filter)
    if root_filter == "any":
        return True
    return str(root_player) == root_filter


def _maybe_print_progress(cfg: RootStorageConfig, stats: RootStorageStats, *, scope: str) -> None:
    every = int(cfg.progress_every)
    if every <= 0:
        return
    stored = int(stats.roots_stored)
    if stored <= 0 or stored % every != 0:
        return
    print(
        f"[{scope}] stored={stored}/{int(cfg.num_roots)} "
        f"selected={int(stats.nonzero_roots_stored)} "
        f"other={int(stats.zero_roots_stored)} "
        f"candidates_seen={int(stats.candidates_seen)}",
        flush=True,
    )


def write_root_shard(
    records: list[dict[str, Any]],
    output_dir: Path,
    shard_index: int,
) -> Path:
    """Persist one shard of root records to disk.

    The first implementation should use an atomic temp-file write followed by
    rename. A `.pt` shard via `torch.save` is acceptable because simulator
    snapshots and stats clones may contain nested Python objects.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"roots_{int(shard_index):06d}.pt"
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    torch.save(list(records), tmp_path)
    tmp_path.replace(shard_path)
    return shard_path


def append_manifest_entry(
    manifest_path: Path,
    shard_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Append one JSONL manifest row for a written shard."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    nonzero = sum(1 for r in records if bool(r.get("is_selected_target", r.get("is_nonzero_target", False))))
    root_ids = [int(r.get("root_id", -1)) for r in records]
    player_counts: dict[str, int] = {}
    for r in records:
        player = str(r.get("root_player", ""))
        player_counts[player] = int(player_counts.get(player, 0)) + 1

    row = {
        "schema_version": int(SCHEMA_VERSION),
        "shard_path": str(shard_path.name),
        "num_records": int(len(records)),
        "nonzero_records": int(nonzero),
        "zero_records": int(len(records) - nonzero),
        "root_id_min": min(root_ids) if root_ids else None,
        "root_id_max": max(root_ids) if root_ids else None,
        "player_counts": player_counts,
    }
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush_shard_if_needed(
    buffer: list[dict[str, Any]],
    cfg: RootStorageConfig,
    stats: RootStorageStats,
) -> list[dict[str, Any]]:
    """Write and clear the current shard buffer when it reaches shard size."""

    if len(buffer) < int(cfg.shard_size):
        return buffer
    shard_path = write_root_shard(
        records=buffer,
        output_dir=cfg.output_dir,
        shard_index=int(stats.shards_written),
    )
    append_manifest_entry(cfg.output_dir / "manifest.jsonl", shard_path, buffer)
    stats.shards_written += 1
    return []


def write_storage_summary(
    output_dir: Path,
    cfg: RootStorageConfig,
    stats: RootStorageStats,
) -> None:
    """Write final summary metadata for the generated root dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_payload = {field.name: getattr(cfg, field.name) for field in fields(cfg)}
    cfg_payload["output_dir"] = str(cfg.output_dir)
    cfg_payload["shared_seen_signatures"] = None
    cfg_payload["shared_seen_lock"] = None
    cfg_payload["exclude_signature_dataset_dirs"] = [
        str(path) for path in cfg.exclude_signature_dataset_dirs
    ]
    summary = {
        "schema_version": int(SCHEMA_VERSION),
        "config": cfg_payload,
        "stats": asdict(stats),
        "nonzero_ratio": (
            float(stats.nonzero_roots_stored) / float(stats.roots_stored)
            if int(stats.roots_stored) > 0
            else 0.0
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidates_seen",
                "roots_stored",
                "nonzero_roots_stored",
                "zero_roots_stored",
                "shards_written",
                "nonzero_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidates_seen": int(stats.candidates_seen),
                "roots_stored": int(stats.roots_stored),
                "nonzero_roots_stored": int(stats.nonzero_roots_stored),
                "zero_roots_stored": int(stats.zero_roots_stored),
                "shards_written": int(stats.shards_written),
                "nonzero_ratio": summary["nonzero_ratio"],
            }
        )


def _worker_root_count(total_roots: int, num_processes: int, worker_id: int) -> int:
    base = int(total_roots) // int(num_processes)
    remainder = int(total_roots) % int(num_processes)
    return int(base + (1 if int(worker_id) < int(remainder) else 0))


def _worker_candidate_count(total_candidates: int, num_processes: int, worker_id: int) -> int:
    base = int(total_candidates) // int(num_processes)
    remainder = int(total_candidates) % int(num_processes)
    return int(base + (1 if int(worker_id) < int(remainder) else 0))


def _build_worker_storage_configs(cfg: RootStorageConfig) -> list[RootStorageConfig]:
    """Split work into small process-recycled root chunks.

    Long-lived Python workers retain simulator/MCTS allocations over time. For
    large local root stores, each task should generate only a bounded number of
    accepted roots and then let the process exit.
    """

    worker_parent_dir = cfg.output_dir / "_worker_parts"
    candidate_ratio = float(cfg.max_candidate_roots) / float(max(1, int(cfg.num_roots)))
    roots_per_task = max(1, int(cfg.worker_roots_per_task))

    worker_cfgs: list[RootStorageConfig] = []
    roots_remaining = int(cfg.num_roots)
    next_root_id = int(cfg.start_root_id)
    task_id = 0
    while roots_remaining > 0:
        worker_num_roots = min(int(roots_per_task), int(roots_remaining))
        worker_max_candidates = int(math.ceil(candidate_ratio * float(worker_num_roots)))
        worker_max_candidates = max(int(worker_num_roots), int(worker_max_candidates))
        worker_cfgs.append(
            replace(
                cfg,
                output_dir=worker_parent_dir / f"task_{task_id:06d}",
                num_roots=int(worker_num_roots),
                max_candidate_roots=int(worker_max_candidates),
                seed=_normalize_numpy_seed(int(cfg.seed) + int(task_id) * 1_000_003),
                start_root_id=int(next_root_id),
                num_processes=1,
            )
        )
        next_root_id += int(worker_max_candidates) + 1
        roots_remaining -= int(worker_num_roots)
        task_id += 1
    return worker_cfgs


def _build_worker_task_config(
    cfg: RootStorageConfig,
    *,
    interval: _RootStorageInterval,
    task_id: int,
    num_roots: int,
    selected_roots: int,
    max_candidate_roots: int,
    start_root_id: int,
) -> RootStorageConfig:
    """Build one recycled worker task config."""

    selected_ratio = float(selected_roots) / float(max(1, int(num_roots)))
    return replace(
        cfg,
        output_dir=(
            cfg.output_dir
            / "_worker_parts"
            / f"interval_{int(interval.interval_id):06d}"
            / f"task_{int(task_id):06d}"
        ),
        num_roots=int(num_roots),
        max_candidate_roots=int(max_candidate_roots),
        min_nonzero_target_ratio=float(selected_ratio),
        seed=_normalize_numpy_seed(int(cfg.seed) + int(task_id) * 1_000_003),
        start_root_id=int(start_root_id),
        num_processes=1,
        allow_partial_results=True,
        shared_seen_signatures=interval.shared_seen_signatures,
        shared_seen_lock=interval.shared_seen_lock,
    )


def _run_root_storage_worker(worker_cfg: RootStorageConfig) -> dict[str, Any]:
    """Generate one worker-local root shard set in an isolated process."""

    stats = generate_and_store_roots(
        replace(worker_cfg, num_processes=1, allow_partial_results=True)
    )
    return {
        "output_dir": str(worker_cfg.output_dir),
        "stats": asdict(stats),
    }


def _merge_worker_outputs(
    *,
    cfg: RootStorageConfig,
    worker_cfgs: list[RootStorageConfig],
    stats: RootStorageStats,
) -> None:
    """Move completed worker shards into the final root store."""

    import shutil

    final_manifest = cfg.output_dir / "manifest.jsonl"
    for worker_cfg in worker_cfgs:
        worker_manifest = worker_cfg.output_dir / "manifest.jsonl"
        if not worker_manifest.exists():
            continue

        with worker_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                src_shard = worker_cfg.output_dir / str(entry["shard_path"])
                if not src_shard.exists():
                    raise FileNotFoundError(f"worker shard not found: {src_shard}")

                dst_shard = cfg.output_dir / f"roots_{int(stats.shards_written):06d}.pt"
                shutil.move(str(src_shard), str(dst_shard))

                entry["shard_path"] = str(dst_shard.name)
                with final_manifest.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")

                num_records = int(entry.get("num_records", 0))
                selected_records = int(entry.get("nonzero_records", 0))
                other_records = int(entry.get("zero_records", max(0, num_records - selected_records)))
                stats.roots_stored += int(num_records)
                stats.nonzero_roots_stored += int(selected_records)
                stats.zero_roots_stored += int(other_records)
                stats.shards_written += 1
                _maybe_print_progress(cfg, stats, scope="root_storage parent")


def generate_and_store_roots_multiprocess(cfg: RootStorageConfig) -> RootStorageStats:
    """Generate roots with multiple independent local processes.

    Each process owns a disjoint root-id interval and its own random seed. The
    parent process merges worker outputs into the final manifest and shards.
    When exclude_signature_dataset_dirs is set, all intervals share one
    signature table pre-seeded from those datasets so the new roots are unique
    relative to the existing store.
    """

    import multiprocessing as mp
    import shutil

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = cfg.output_dir / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()

    worker_parent_dir = cfg.output_dir / "_worker_parts"
    if worker_parent_dir.exists():
        shutil.rmtree(worker_parent_dir)
    worker_parent_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    manager = mp.Manager()
    global_shared_seen = None
    global_shared_lock = None
    if cfg.exclude_signature_dataset_dirs:
        if not bool(cfg.deduplicate_history_signatures):
            raise ValueError(
                "exclude_signature_dataset_dirs requires deduplicate_history_signatures=True"
            )
        global_shared_seen = manager.dict()
        global_shared_lock = manager.Lock()
        excluded_count = 0
        for dataset_dir in cfg.exclude_signature_dataset_dirs:
            keys = load_history_signature_keys_from_dataset(dataset_dir)
            excluded_count += len(keys)
            for key in keys:
                global_shared_seen[key] = 1
            del keys
        print(
            "[root_storage parent] "
            f"preseeded excluded history signatures={int(excluded_count)} "
            f"from {len(cfg.exclude_signature_dataset_dirs)} dataset(s)",
            flush=True,
        )
    stats = RootStorageStats()
    target_selected = int(math.ceil(float(cfg.num_roots) * float(cfg.min_nonzero_target_ratio)))
    candidate_ratio = float(cfg.max_candidate_roots) / float(max(1, int(cfg.num_roots)))
    worker_roots_per_task = max(1, int(cfg.worker_roots_per_task))
    max_processes_per_interval = max(1, int(cfg.max_processes_per_interval))
    task_id = 0
    next_root_id = int(cfg.start_root_id)
    candidates_assigned = 0

    intervals: list[_RootStorageInterval] = []
    roots_remaining = int(cfg.num_roots)
    while roots_remaining > 0:
        target_roots = min(int(worker_roots_per_task), int(roots_remaining))
        intervals.append(
            _RootStorageInterval(
                interval_id=len(intervals),
                target_roots=int(target_roots),
                target_selected=int(
                    math.ceil(float(target_roots) * float(cfg.min_nonzero_target_ratio))
                ),
                shared_seen_signatures=(
                    global_shared_seen if global_shared_seen is not None else manager.dict()
                ),
                shared_seen_lock=(
                    global_shared_lock if global_shared_lock is not None else manager.Lock()
                ),
            )
        )
        roots_remaining -= int(target_roots)

    # Memory note:
    # Workers are intentionally short-lived and may return partial results. If a
    # sparse interval cannot produce its requested 512 roots inside its local
    # candidate budget, the parent stores whatever it produced and launches more
    # recycled tasks against the same interval. Interval uniqueness is enforced
    # through a shared signature dictionary + lock.
    try:
        while (
            any(not interval.complete for interval in intervals)
            and int(candidates_assigned) < int(cfg.max_candidate_roots)
        ):
            task_cfgs: list[RootStorageConfig] = []
            task_intervals: list[_RootStorageInterval] = []
            wave_processes_by_interval: dict[int, int] = {}
            wave_planned_roots_by_interval: dict[int, int] = {}
            wave_planned_selected_by_interval: dict[int, int] = {}

            while len(task_cfgs) < int(cfg.num_processes):
                if int(candidates_assigned) >= int(cfg.max_candidate_roots):
                    break

                available: list[_RootStorageInterval] = []
                for interval in intervals:
                    if interval.complete:
                        continue
                    interval_id = int(interval.interval_id)
                    if int(wave_processes_by_interval.get(interval_id, 0)) >= int(max_processes_per_interval):
                        continue
                    planned_roots = int(wave_planned_roots_by_interval.get(interval_id, 0))
                    planned_selected = int(wave_planned_selected_by_interval.get(interval_id, 0))
                    if int(interval.remaining_roots) - int(planned_roots) <= 0:
                        continue
                    if int(interval.remaining_selected) - int(planned_selected) < 0:
                        continue
                    available.append(interval)

                if not available:
                    break

                interval = max(
                    available,
                    key=lambda x: (
                        int(x.remaining_selected)
                        - int(wave_planned_selected_by_interval.get(int(x.interval_id), 0)),
                        int(x.remaining_roots)
                        - int(wave_planned_roots_by_interval.get(int(x.interval_id), 0)),
                    ),
                )
                interval_id = int(interval.interval_id)
                running_for_interval = int(wave_processes_by_interval.get(interval_id, 0))
                slots_left = max(1, int(max_processes_per_interval) - int(running_for_interval))
                remaining_roots_for_interval = (
                    int(interval.remaining_roots)
                    - int(wave_planned_roots_by_interval.get(interval_id, 0))
                )
                remaining_selected_for_interval = max(
                    0,
                    int(interval.remaining_selected)
                    - int(wave_planned_selected_by_interval.get(interval_id, 0)),
                )
                if int(remaining_roots_for_interval) <= 0:
                    break

                task_roots = int(math.ceil(float(remaining_roots_for_interval) / float(slots_left)))
                task_roots = max(1, min(int(task_roots), int(remaining_roots_for_interval)))
                selected_fraction = (
                    float(remaining_selected_for_interval)
                    / float(max(1, int(remaining_roots_for_interval)))
                )
                task_selected = int(math.ceil(float(task_roots) * float(selected_fraction)))
                task_selected = min(
                    int(task_selected),
                    int(task_roots),
                    int(remaining_selected_for_interval),
                )
                if int(remaining_selected_for_interval) > 0 and int(task_selected) <= 0:
                    task_selected = 1
                task_candidates = int(math.ceil(float(task_roots) * float(candidate_ratio)))
                task_candidates = max(int(task_roots), int(task_candidates))
                task_candidates = min(
                    int(task_candidates),
                    int(cfg.max_candidate_roots) - int(candidates_assigned),
                )
                if int(task_candidates) <= 0:
                    break

                task_cfgs.append(
                    _build_worker_task_config(
                        cfg,
                        interval=interval,
                        task_id=int(task_id),
                        num_roots=int(task_roots),
                        selected_roots=int(task_selected),
                        max_candidate_roots=int(task_candidates),
                        start_root_id=int(next_root_id),
                    )
                )
                task_intervals.append(interval)
                wave_processes_by_interval[interval_id] = int(running_for_interval) + 1
                wave_planned_roots_by_interval[interval_id] = (
                    int(wave_planned_roots_by_interval.get(interval_id, 0)) + int(task_roots)
                )
                wave_planned_selected_by_interval[interval_id] = (
                    int(wave_planned_selected_by_interval.get(interval_id, 0)) + int(task_selected)
                )
                candidates_assigned += int(task_candidates)
                next_root_id += int(task_candidates) + 1
                task_id += 1

            if not task_cfgs:
                break

            with ctx.Pool(
                processes=min(int(cfg.num_processes), len(task_cfgs)),
                maxtasksperchild=1,
            ) as pool:
                worker_summaries = pool.map(_run_root_storage_worker, task_cfgs)

            stats.candidates_seen += sum(
                int(row["stats"]["candidates_seen"]) for row in worker_summaries
            )
            _merge_worker_outputs(cfg=cfg, worker_cfgs=task_cfgs, stats=stats)
            for interval, summary in zip(task_intervals, worker_summaries):
                worker_stats = summary["stats"]
                interval.stored_roots += int(worker_stats.get("roots_stored", 0))
                interval.selected_roots += int(worker_stats.get("nonzero_roots_stored", 0))
                interval.other_roots += int(worker_stats.get("zero_roots_stored", 0))
                if interval.complete:
                    interval.shared_seen_signatures = None
                    interval.shared_seen_lock = None
    finally:
        manager.shutdown()

    write_storage_summary(cfg.output_dir, cfg, stats)

    if int(stats.roots_stored) < int(cfg.num_roots):
        raise RuntimeError(
            "multiprocess root storage did not reach requested size before candidate budget ended: "
            f"stored={stats.roots_stored} requested={cfg.num_roots} "
            f"candidates_seen={stats.candidates_seen} candidates_assigned={candidates_assigned}"
        )
    if int(stats.nonzero_roots_stored) < int(target_selected):
        raise RuntimeError(
            "multiprocess root storage did not find enough selected-target roots: "
            f"selected={stats.nonzero_roots_stored} required={target_selected} "
            f"candidates_seen={stats.candidates_seen}"
        )
    if worker_parent_dir.exists():
        shutil.rmtree(worker_parent_dir)
    return stats


def generate_and_store_roots(cfg: RootStorageConfig) -> RootStorageStats:
    """Main local orchestration function for root-state storage.

    High-level flow:
        1. Build local GV3 environment and MCTS.
        2. Generate candidate history roots.
        3. Evaluate each root with no-bootstrap first-layer Bellman scoring.
        4. Keep roots according to nonzero/zero balance quotas.
        5. Write shards and a manifest under `cfg.output_dir`.
        6. Write a final summary.

    Or more optimised way :
    for batch_id in batches:
    1. generate N candidate roots locally
    2. keep their full history iter logs
    3. run no-bootstrap mcts_dnn scoring over those N roots
    4. filter/select roots needed for the final balance
    5. write selected roots to shard(s)
    
    """

    if int(cfg.num_processes) > 1:
        return generate_and_store_roots_multiprocess(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = cfg.output_dir / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()

    env, mcts = make_local_env_and_mcts(cfg)
    stats = RootStorageStats()
    buffer: list[dict[str, Any]] = []

    try:
        for batch in generate_candidate_root_batches(cfg, env):
            for root in batch:
                if int(stats.roots_stored) >= int(cfg.num_roots):
                    break
                stats.candidates_seen += 1
                if not root_player_matches_filter(cfg, root.root_player):
                    continue
                target = evaluate_first_layer_target(cfg, mcts, root)
                record = pack_root_record(cfg, root, target)
                if not should_keep_record(record, stats, cfg):
                    continue

                buffer.append(record)
                stats.roots_stored += 1
                if bool(record.get("is_selected_target", record.get("is_nonzero_target", False))):
                    stats.nonzero_roots_stored += 1
                else:
                    stats.zero_roots_stored += 1
                _maybe_print_progress(
                    cfg,
                    stats,
                    scope=f"root_storage worker_start={int(cfg.start_root_id)}",
                )
                buffer = flush_shard_if_needed(buffer, cfg, stats)

            if int(stats.roots_stored) >= int(cfg.num_roots):
                break
            if int(stats.candidates_seen) >= int(cfg.max_candidate_roots):
                break

        if buffer:
            shard_path = write_root_shard(
                records=buffer,
                output_dir=cfg.output_dir,
                shard_index=int(stats.shards_written),
            )
            append_manifest_entry(cfg.output_dir / "manifest.jsonl", shard_path, buffer)
            stats.shards_written += 1
            buffer = []

        write_storage_summary(cfg.output_dir, cfg, stats)
    finally:
        if hasattr(mcts, "close"):
            mcts.close()

    if int(stats.roots_stored) < int(cfg.num_roots) and not bool(cfg.allow_partial_results):
        raise RuntimeError(
            "root storage did not reach requested size: "
            f"stored={stats.roots_stored} requested={cfg.num_roots} "
            f"candidates_seen={stats.candidates_seen}"
        )
    return stats


def load_stored_roots(
    dataset_dir: str | Path,
    *,
    max_roots: int | None = None,
) -> list[dict[str, Any]]:
    """Load stored root records from a previously generated dataset."""

    root_dir = Path(dataset_dir).expanduser()
    manifest = root_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    out: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            shard_path = root_dir / str(entry["shard_path"])
            try:
                records = torch.load(shard_path, map_location="cpu", weights_only=False)
            except TypeError:
                records = torch.load(shard_path, map_location="cpu")
            for record in records:
                out.append(record)
                if max_roots is not None and len(out) >= int(max_roots):
                    return out
    return out


def parse_args() -> RootStorageConfig:
    """Parse CLI args and return `RootStorageConfig`."""

    parser = argparse.ArgumentParser(description="Generate local GV3 ModelSearchBed root storage.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-roots", type=int, default=10_000)
    parser.add_argument("--max-candidate-roots", type=int, default=100_000)
    parser.add_argument("--candidate-batch-size", type=int, default=1_024)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=32,
        help=(
            "Number of live generated root states yielded/evaluated at once. "
            "Lower values reduce worker RSS; candidate-batch-size remains the "
            "per-session candidate budget."
        ),
    )
    parser.add_argument("--min-nonzero-target-ratio", type=float, default=0.50)
    parser.add_argument("--nonzero-eps", type=float, default=DEFAULT_NONZERO_EPS)
    parser.add_argument(
        "--target-abs-threshold",
        type=float,
        default=None,
        help=(
            "Threshold for the quota bucket. Use 1.0 to require "
            "abs(target_value) >= 1 for the configured ratio."
        ),
    )
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=200)
    parser.add_argument("--history-max-total-steps", type=int, default=20_000)
    parser.add_argument("--max-children-per-expand", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--game-id", type=int, default=0)
    parser.add_argument("--start-root-id", type=int, default=0)
    parser.add_argument("--start-depth", type=int, default=0)
    parser.add_argument("--start-player", choices=("adversary", "controller"), default="adversary")
    parser.add_argument("--root-player-filter", choices=("controller", "adversary", "any"), default="controller")
    parser.add_argument("--no-history-trace-logs", action="store_true")
    parser.add_argument("--allow-duplicate-history-fallback", action="store_true")
    parser.add_argument(
        "--deduplicate-history-signatures",
        action="store_true",
        help=(
            "Track generated history signatures across sessions. Disabled by "
            "default for large storage runs to avoid unbounded memory growth."
        ),
    )
    parser.add_argument(
        "--history-signature-cache-size",
        type=int,
        default=10_000,
        help=(
            "Maximum recent history signatures retained per worker task when "
            "--deduplicate-history-signatures is enabled."
        ),
    )
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument(
        "--worker-roots-per-task",
        type=int,
        default=1024,
        help=(
            "Accepted roots generated by one worker process before that "
            "process exits and releases memory."
        ),
    )
    parser.add_argument("--max-processes-per-interval", type=int, default=4)
    parser.add_argument(
        "--exclude-signature-dataset-dir",
        action="append",
        default=[],
        help=(
            "Existing root dataset whose history signatures should be rejected "
            "during this generation run. May be supplied multiple times."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    return build_storage_config(
        output_dir=args.output_dir,
        num_roots=args.num_roots,
        max_candidate_roots=args.max_candidate_roots,
        candidate_batch_size=args.candidate_batch_size,
        generation_batch_size=args.generation_batch_size,
        min_nonzero_target_ratio=args.min_nonzero_target_ratio,
        nonzero_eps=args.nonzero_eps,
        target_abs_threshold=args.target_abs_threshold,
        history_hops_min=args.history_hops_min,
        history_hops_max=args.history_hops_max,
        history_max_total_steps=args.history_max_total_steps,
        max_children_per_expand=args.max_children_per_expand,
        shard_size=args.shard_size,
        seed=args.seed,
        game_id=args.game_id,
        start_root_id=args.start_root_id,
        start_depth=args.start_depth,
        start_player=args.start_player,
        root_player_filter=args.root_player_filter,
        include_history_trace_logs=not bool(args.no_history_trace_logs),
        allow_duplicate_history_fallback=bool(args.allow_duplicate_history_fallback),
        deduplicate_history_signatures=bool(args.deduplicate_history_signatures),
        history_signature_cache_size=args.history_signature_cache_size,
        num_processes=args.num_processes,
        worker_roots_per_task=args.worker_roots_per_task,
        max_processes_per_interval=args.max_processes_per_interval,
        exclude_signature_dataset_dirs=args.exclude_signature_dataset_dir,
        progress_every=args.progress_every,
    )


def main() -> None:
    """CLI entrypoint for local root-state generation."""

    cfg = parse_args()
    generate_and_store_roots(cfg)


if __name__ == "__main__":
    main()
