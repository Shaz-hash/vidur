## # (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)


"""
    This script will create all of the child states along with the immediate reward, 
    simulation time of the parent, simulation time of the child, discount factor based on the time difference 
    of the given root parent state in the dataset. And it will carry a log trace row that we can use to debug the child generation process


    one possibility :
    {
        "parent_state_id": int,
        "action_index": int,
        "canonical_action_index": int,
        "action_repr": str,
        "reward": float,
        "discount": float,
        "child_cost": float,
        "child_time": float,
        "child_simulator_snapshot": ...,
        "row" : ....,
        "child_stats": ...,
        "is_valid": bool,
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import torch

from ..self_model_test import RootStateLoader, build_self_model_test_config

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RootChildGenerationConfig:
    dataset_dir: Path
    output_dir: Path
    max_roots: int | None = None
    root_player_filter: str = "controller"
    transition_shard_size: int = 4096
    include_alias_rows: bool = False
    validate_first_n: int = 0
    overwrite: bool = False
    seed: int = 12345
    num_processes: int = 1
    parents_per_task: int = 1000
    parent_state_id_start: int = 0
    parent_state_id_end: int | None = None
    worker_index: int | None = None


def _safe_float(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name, default))
    except Exception:
        return float(default)


def _safe_int(obj: Any, name: str, default: int = 0) -> int:
    try:
        return int(getattr(obj, name, default))
    except Exception:
        return int(default)


def _sorted_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return sorted(int(x) for x in value)
    except Exception:
        return []


def _snapshot_hash(snapshot: Any) -> str:
    return hashlib.sha256(
        pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _stats_hash(stats: Any) -> str:
    return hashlib.sha256(
        pickle.dumps(stats, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _state_summary(prefix: str, state: Any, *, cost: float) -> dict[str, Any]:
    stats = state.stats
    sim = state.simulator

    return {
        f"{prefix}_sim_time": float(sim._time),
        f"{prefix}_cost": float(cost),
        f"{prefix}_active_request_ids": _sorted_int_list(
            getattr(stats, "active_request_ids", [])
        ),
        f"{prefix}_completed_request_ids": _sorted_int_list(
            getattr(stats, "completed_request_ids", [])
        ),
        f"{prefix}_violated_request_ids": _sorted_int_list(
            getattr(stats, "violated_request_ids", [])
        ),
        f"{prefix}_slo_violations": len(
            _sorted_int_list(getattr(stats, "violated_request_ids", []))
        ),
        f"{prefix}_total_lateness": _safe_float(stats, "total_lateness", 0.0),
        f"{prefix}_total_cost": _safe_float(stats, "total_cost", float(cost)),
        f"{prefix}_transition_discount_time": _safe_float(
            stats,
            "transition_discount_time",
            float(sim._time),
        ),
    }





def count_samples(
    dataset_dir: str | Path,
    *,
    root_player_filter: str = "controller",
    max_roots: int | None = None,
) -> int:
    """Count filtered root samples using manifest metadata when available."""

    root_dir = Path(dataset_dir).expanduser()
    manifest = root_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    total = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if root_player_filter == "any":
                count = int(entry.get("num_records", 0))
            else:
                player_counts = entry.get("player_counts") or {}
                count = int(player_counts.get(root_player_filter, 0))
                if count <= 0 and "player_counts" not in entry:
                    shard_path = root_dir / str(entry["shard_path"])
                    records = torch.load(shard_path, map_location="cpu", weights_only=False)
                    count = sum(
                        1
                        for record in records
                        if str(record.get("root_player", "")) == root_player_filter
                    )
            total += int(count)
            if max_roots is not None and total >= int(max_roots):
                return int(max_roots)

    return int(total)


def load_samples(
    dataset_dir: str | Path,
    *,
    root_player_filter: str = "controller",
    max_roots: int | None = None,
    parent_state_id_start: int = 0,
    parent_state_id_end: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield `(parent_state_id, root_record)` from existing root shards.

    `parent_state_id` is the sample id in the filtered dataset order. This is
    the id stored in each child transition row. The optional start/end range is
    an absolute half-open interval over those filtered sample ids, which lets
    multiprocessing workers own disjoint parent ranges.
    """

    root_dir = Path(dataset_dir).expanduser()
    manifest = root_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    start_id = max(0, int(parent_state_id_start))
    end_id = None if parent_state_id_end is None else max(start_id, int(parent_state_id_end))
    if max_roots is not None:
        max_end = int(max_roots)
        end_id = max_end if end_id is None else min(end_id, max_end)

    sample_id = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            shard_path = root_dir / str(entry["shard_path"])
            records = torch.load(shard_path, map_location="cpu", weights_only=False)

            for record in records:
                if (
                    root_player_filter != "any"
                    and str(record.get("root_player", "")) != root_player_filter
                ):
                    continue

                if end_id is not None and sample_id >= int(end_id):
                    return

                if sample_id >= int(start_id):
                    yield sample_id, record

                sample_id += 1

def _make_state_loader(cfg: RootChildGenerationConfig) -> RootStateLoader:
    """Build the same local GV3 env/MCTS wrapper used by ModelSearchBed."""

    if cfg.parent_state_id_end is not None:
        n = max(1, int(cfg.parent_state_id_end) - int(cfg.parent_state_id_start))
    else:
        n = int(cfg.max_roots or 1)
    self_cfg = build_self_model_test_config(
        dataset_dir=cfg.dataset_dir,
        output_dir=cfg.output_dir / "_loader_tmp",
        num_roots=max(1, n),
        max_candidate_roots=max(1, n),
        root_player_filter=cfg.root_player_filter,
        allow_dataset_generation=False,
        seed=int(cfg.seed),
    )
    return RootStateLoader(self_cfg)


def _pickle_hash(obj: Any) -> str:
    """Stable-enough hash for comparing snapshots generated in the same codebase."""

    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(blob).hexdigest()



def build_transition_trace_row(
    *,
    parent_state_id: int,
    root_record: dict[str, Any],
    parent_state: Any,
    child_state: Any,
    parent_snapshot: Any,
    parent_stats: Any,
    child_snapshot: Any,
    child_stats: Any,
    action_index: int,
    canonical_action_index: int,
    alias_action_indices: list[int],
    action_repr: str,
    parent_cost: float,
    child_cost: float,
    reward: float,
    discount: float,
    q_no_bootstrap: float,
    next_player: str,
    num_valid_actions: int,
    num_canonical_actions: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_type": "root_child_transition",

        "parent_state_id": int(parent_state_id),
        "parent_root_id": int(root_record.get("root_id", -1)),
        "root_player": str(root_record.get("root_player", "")),
        "root_depth": int(root_record.get("root_depth", 0)),
        "history_hops": int(root_record.get("history_hops", -1)),

        "action_index": int(action_index),
        "canonical_action_index": int(canonical_action_index),
        "alias_action_indices": [int(x) for x in alias_action_indices],
        "action_repr": str(action_repr),

        "player_acted": str(root_record.get("root_player", "controller")),
        "player_to_act_next": str(next_player),

        "reward": float(reward),
        "discount": float(discount),
        "q_no_bootstrap": float(q_no_bootstrap),

        "num_valid_actions": int(num_valid_actions),
        "num_canonical_actions": int(num_canonical_actions),

        "parent_snapshot_hash": _snapshot_hash(parent_snapshot),
        "parent_stats_hash": _stats_hash(parent_stats),
        "child_snapshot_hash": _snapshot_hash(child_snapshot),
        "child_stats_hash": _stats_hash(child_stats),
    }

    row.update(_state_summary("parent", parent_state, cost=parent_cost))
    row.update(_state_summary("child", child_state, cost=child_cost))
    return row


def _snapshot_state_and_stats(mcts: Any, state: Any) -> tuple[Any, Any]:
    """Use the exact MCTS snapshot helper when available."""

    return mcts._snapshot_state_and_stats(state)


def generate_child_states(
    *,
    parent_state_id: int,
    root_record: dict[str, Any],
    root_parent_state: Any,
    mcts: Any,
    include_alias_rows: bool = False,
) -> list[dict[str, Any]]:
    """Generate cached one-step child transitions for one controller root.

    Stored transition target pieces are model-independent:

        q_i = reward_i + discount_i * V(child_i)

    Later Bellman iterations can load these rows, run the current bootstrap
    model on each `child_*` state, and take max over valid controller actions.
    """

    parent_player = str(root_record.get("root_player", "controller"))
    if parent_player != "controller":
        raise ValueError(
            "rootChildGeneration currently supports controller roots only; "
            f"got root_player={parent_player!r}"
        )

    actions_by_index, mask_t = mcts._actions_and_mask(
        root_parent_state,
        parent_player,
        forbidden_stop_ids=None,
    )
    valid_mask = [bool(x) for x in mask_t.tolist()]
    valid_indices = [
        int(i)
        for i, ok in enumerate(valid_mask)
        if ok and actions_by_index[i] is not None
    ]

    if not valid_indices:
        return []

    alias_to_canon, canon_to_aliases, canonical_indices = mcts._canonicalize_action_indices(
        player=parent_player,
        actions_by_index=actions_by_index,
        valid_indices=valid_indices,
    )

    decision_snapshot, decision_stats = _snapshot_state_and_stats(mcts, root_parent_state)
    parent_cost = float(mcts._state_cost(root_parent_state))
    parent_time = float(root_parent_state.simulator._time)
    next_player = mcts._next_player(parent_player)

    canonical_rows: dict[int, dict[str, Any]] = {}

    for canonical_action_index in canonical_indices:
        action = actions_by_index[canonical_action_index]
        if action is None:
            continue

        child_state = mcts._scratch_restore(decision_snapshot, decision_stats)
        child_state = mcts._env.apply_controller_action_only(
            child_state,
            action,
            inplace=True,
            fast_forward=False,
        )

        q, reward, discount, bootstrap, child_cost, child_time = mcts._compose_q_from_state(
            leaf_state=child_state,
            parent_cost=parent_cost,
            parent_time=parent_time,
            next_player=next_player,
            dnn_model=None,
            model_version=0,
            use_model_bootstrap=False,
        )

        child_snapshot, child_stats = _snapshot_state_and_stats(mcts, child_state)
        alias_action_indices = [
            int(x) for x in canon_to_aliases.get(int(canonical_action_index), [canonical_action_index])
        ]

        transition_row = build_transition_trace_row(
            parent_state_id=parent_state_id,
            root_record=root_record,
            parent_state=root_parent_state,
            child_state=child_state,
            parent_snapshot=decision_snapshot,
            parent_stats=decision_stats,
            child_snapshot=child_snapshot,
            child_stats=child_stats,
            action_index=int(canonical_action_index),
            canonical_action_index=int(canonical_action_index),
            alias_action_indices=alias_action_indices,
            action_repr=repr(action),
            parent_cost=parent_cost,
            child_cost=child_cost,
            reward=reward,
            discount=discount,
            q_no_bootstrap=q,
            next_player=next_player,
            num_valid_actions=len(valid_indices),
            num_canonical_actions=len(canonical_indices),
        )

        canonical_rows[int(canonical_action_index)] = {
            "schema_version": SCHEMA_VERSION,
            "parent_state_id": int(parent_state_id),
            "parent_root_id": int(root_record.get("root_id", -1)),
            "action_index": int(canonical_action_index),
            "canonical_action_index": int(canonical_action_index),
            "alias_action_indices": alias_action_indices,
            "action_repr": repr(action),
            "reward": float(reward),
            "discount": float(discount),
            "child_cost": float(child_cost),
            "child_time": float(child_time),
            "child_simulator_snapshot": child_snapshot,
            "child_stats": child_stats,
            "row": transition_row,
            "is_valid": True,
        }

    if not include_alias_rows:
        return [canonical_rows[i] for i in sorted(canonical_rows)]

    rows: list[dict[str, Any]] = []
    for action_index in valid_indices:
        canonical_action_index = int(alias_to_canon.get(int(action_index), int(action_index)))
        base = canonical_rows.get(canonical_action_index)
        if base is None:
            continue

        action = actions_by_index[action_index]
        alias_row = dict(base)
        alias_row["action_index"] = int(action_index)
        alias_row["canonical_action_index"] = int(canonical_action_index)
        alias_row["action_repr"] = repr(action)
        alias_row["is_canonical"] = int(action_index) == int(canonical_action_index)
        rows.append(alias_row)

    return rows


def store_child_states(
    child_states: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    shard_index: int,
) -> dict[str, Any]:
    """Write one child-transition shard and return its manifest row."""

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_name = f"child_transitions_{int(shard_index):06d}.pt"
    shard_path = out_dir / shard_name
    torch.save(child_states, shard_path)

    parent_ids = [int(row["parent_state_id"]) for row in child_states]
    root_ids = [int(row.get("parent_root_id", -1)) for row in child_states]
    canonical_count = sum(
        1
        for row in child_states
        if int(row["action_index"]) == int(row["canonical_action_index"])
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "shard_path": shard_name,
        "num_transitions": int(len(child_states)),
        "num_canonical_transitions": int(canonical_count),
        "parent_state_id_min": min(parent_ids) if parent_ids else None,
        "parent_state_id_max": max(parent_ids) if parent_ids else None,
        "parent_root_id_min": min(root_ids) if root_ids else None,
        "parent_root_id_max": max(root_ids) if root_ids else None,
    }



def assert_transition_rows_match(
    cached: dict[str, Any],
    regenerated: dict[str, Any],
    *,
    atol: float = 1e-9,
) -> None:
    exact_keys = [
        "parent_state_id",
        "parent_root_id",
        "root_player",
        "root_depth",
        "action_index",
        "canonical_action_index",
        "action_repr",
        "player_acted",
        "player_to_act_next",
        "parent_snapshot_hash",
        "child_snapshot_hash",
    ]

    float_keys = [
        "parent_sim_time",
        "child_sim_time",
        "parent_cost",
        "child_cost",
        "reward",
        "discount",
        "q_no_bootstrap",
        "parent_total_lateness",
        "child_total_lateness",
        "parent_total_cost",
        "child_total_cost",
    ]

    list_keys = [
        "parent_active_request_ids",
        "child_active_request_ids",
        "parent_completed_request_ids",
        "child_completed_request_ids",
        "parent_violated_request_ids",
        "child_violated_request_ids",
        "alias_action_indices",
    ]

    for key in exact_keys:
        if cached.get(key) != regenerated.get(key):
            raise AssertionError(
                f"{key} mismatch: cached={cached.get(key)!r}, "
                f"regenerated={regenerated.get(key)!r}"
            )

    for key in float_keys:
        a = float(cached.get(key, 0.0))
        b = float(regenerated.get(key, 0.0))
        if abs(a - b) > float(atol):
            raise AssertionError(f"{key} mismatch: cached={a}, regenerated={b}")

    for key in list_keys:
        if list(cached.get(key, [])) != list(regenerated.get(key, [])):
            raise AssertionError(
                f"{key} mismatch: cached={cached.get(key)!r}, "
                f"regenerated={regenerated.get(key)!r}"
            )


def test_child_states(
    child_states: list[dict[str, Any]],
    *,
    parent_records_by_id: dict[int, dict[str, Any]],
    state_loader: RootStateLoader,
    max_checks: int = 100,
    atol: float = 1e-9,
    strict_snapshot_hash: bool = False,
) -> None:
    """Validate cached transitions by regenerating them from parent records.

    This checks reward, discount, child cost, child time, and optionally child
    snapshot hash. It regenerates only the requested `max_checks` rows.
    """

    mcts = state_loader.mcts
    checked = 0

    for cached in child_states:
        if checked >= int(max_checks):
            break

        parent_state_id = int(cached["parent_state_id"])
        record = parent_records_by_id[parent_state_id]
        parent_state = state_loader(record)

        regenerated = generate_child_states(
            parent_state_id=parent_state_id,
            root_record=record,
            root_parent_state=parent_state,
            mcts=mcts,
            include_alias_rows=True,
        )

        match = None
        for row in regenerated:
            if int(row["action_index"]) == int(cached["action_index"]):
                match = row
                break

        if match is None:
            raise AssertionError(
                f"could not regenerate action_index={cached['action_index']} "
                f"for parent_state_id={parent_state_id}"
            )

        for key in ("reward", "discount", "child_cost", "child_time"):
            a = float(cached[key])
            b = float(match[key])
            if abs(a - b) > float(atol):
                raise AssertionError(
                    f"{key} mismatch for parent_state_id={parent_state_id}, "
                    f"action_index={cached['action_index']}: cached={a}, regenerated={b}"
                )

        assert_transition_rows_match(
            cached["row"],
            match["row"],
            atol=atol,
        )

        if strict_snapshot_hash:
            cached_hash = _pickle_hash(cached["child_simulator_snapshot"])
            regen_hash = _pickle_hash(match["child_simulator_snapshot"])
            if cached_hash != regen_hash:
                raise AssertionError(
                    f"child snapshot hash mismatch for parent_state_id={parent_state_id}, "
                    f"action_index={cached['action_index']}"
                )

        checked += 1



def _split_ranges(total: int, num_parts: int) -> list[tuple[int, int]]:
    """Split `[0, total)` into near-even half-open ranges."""

    total = max(0, int(total))
    num_parts = max(1, int(num_parts))
    ranges: list[tuple[int, int]] = []
    for i in range(num_parts):
        start = (total * i) // num_parts
        end = (total * (i + 1)) // num_parts
        if end > start:
            ranges.append((int(start), int(end)))
    return ranges


def _split_ranges_by_size(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split `[0, total)` into fixed-size task ranges."""

    total = max(0, int(total))
    chunk_size = max(1, int(chunk_size))
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def _run_child_generation_worker(cfg: RootChildGenerationConfig) -> dict[str, Any]:
    """Multiprocessing entrypoint for one disjoint parent-state range."""

    worker_cfg = replace(cfg, num_processes=1)
    return generate_child_cache(worker_cfg)


def generate_child_cache_multiprocess(cfg: RootChildGenerationConfig) -> dict[str, Any]:
    """Generate child-transition shards with recycled short-lived worker tasks.

    Each task owns a disjoint `parent_state_id` interval of at most
    `cfg.parents_per_task` parents and writes shards to
    `cfg.output_dir/task_XXXXXX`. The multiprocessing pool uses
    `maxtasksperchild=1`, so worker processes exit after one task and the OS
    releases memory accumulated by simulator/MCTS/pickle allocations.
    """

    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()) and not cfg.overwrite:
        raise FileExistsError(
            f"output_dir is not empty: {cfg.output_dir}. Use --overwrite to replace/append knowingly."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), default=str, indent=2) + "\n",
        encoding="utf-8",
    )

    total_parents = count_samples(
        cfg.dataset_dir,
        root_player_filter=cfg.root_player_filter,
        max_roots=cfg.max_roots,
    )
    ranges = _split_ranges_by_size(total_parents, int(cfg.parents_per_task))
    if not ranges:
        raise RuntimeError("no parent samples available for child generation")

    task_cfgs: list[RootChildGenerationConfig] = []
    for task_index, (start, end) in enumerate(ranges):
        task_cfgs.append(
            replace(
                cfg,
                output_dir=cfg.output_dir / f"task_{task_index:06d}",
                num_processes=1,
                parent_state_id_start=int(start),
                parent_state_id_end=int(end),
                worker_index=int(task_index),
                validate_first_n=(int(cfg.validate_first_n) if task_index == 0 else 0),
                seed=(int(cfg.seed) + int(task_index)) % (2**32 - 1),
            )
        )

    ctx = mp.get_context("spawn")
    pool_size = min(int(cfg.num_processes), len(task_cfgs))
    with ctx.Pool(processes=pool_size, maxtasksperchild=1) as pool:
        task_summaries = pool.map(_run_child_generation_worker, task_cfgs)

    merged_manifest = cfg.output_dir / "manifest.jsonl"
    total_transitions = 0
    total_canonical = 0
    total_shards = 0
    total_parent_states = 0

    with merged_manifest.open("w", encoding="utf-8") as out_f:
        for task_cfg, summary in zip(task_cfgs, task_summaries):
            task_manifest = task_cfg.output_dir / "manifest.jsonl"
            if not task_manifest.exists():
                continue
            with task_manifest.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    entry["task_index"] = int(task_cfg.worker_index or 0)
                    entry["parent_state_id_start"] = int(task_cfg.parent_state_id_start)
                    entry["parent_state_id_end"] = task_cfg.parent_state_id_end
                    entry["shard_path"] = str(Path(task_cfg.output_dir.name) / Path(str(entry["shard_path"])))
                    out_f.write(json.dumps(entry) + "\n")
                    total_shards += 1

            total_parent_states += int(summary.get("num_parent_states", 0))
            total_transitions += int(summary.get("num_transitions", 0))
            total_canonical += int(summary.get("num_canonical_transitions", 0))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(cfg.dataset_dir),
        "output_dir": str(cfg.output_dir),
        "num_processes": int(cfg.num_processes),
        "parents_per_task": int(cfg.parents_per_task),
        "num_tasks": int(len(task_cfgs)),
        "num_parent_states": int(total_parent_states),
        "num_transitions": int(total_transitions),
        "num_canonical_transitions": int(total_canonical),
        "num_shards": int(total_shards),
        "include_alias_rows": bool(cfg.include_alias_rows),
        "task_summaries": task_summaries,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def generate_child_cache(cfg: RootChildGenerationConfig) -> dict[str, Any]:
    """Main orchestration: stream roots, generate child transitions, write shards."""

    if int(cfg.num_processes) > 1 and cfg.parent_state_id_end is None:
        return generate_child_cache_multiprocess(cfg)

    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()) and not cfg.overwrite:
        raise FileExistsError(
            f"output_dir is not empty: {cfg.output_dir}. Use --overwrite to replace/append knowingly."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.output_dir / "manifest.jsonl"
    summary_path = cfg.output_dir / "summary.json"
    config_path = cfg.output_dir / "config.json"

    config_path.write_text(json.dumps(asdict(cfg), default=str, indent=2) + "\n")

    state_loader = _make_state_loader(cfg)

    shard_buffer: list[dict[str, Any]] = []
    shard_index = 0
    num_parents = 0
    num_transitions = 0
    num_canonical_transitions = 0
    validation_remaining = int(cfg.validate_first_n)
    parent_records_for_validation: dict[int, dict[str, Any]] = {}

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest_f:
            for parent_state_id, record in load_samples(
                cfg.dataset_dir,
                root_player_filter=cfg.root_player_filter,
                max_roots=cfg.max_roots,
                parent_state_id_start=int(cfg.parent_state_id_start),
                parent_state_id_end=cfg.parent_state_id_end,
            ):
                # print (f"Processing parent_state_id={parent_state_id} (root_id={record.get('root_id', 'N/A')})...")
                # print(json.dumps(record["simulator_snapshot"], indent=2, default=str))
        
                # print("=== best action details ===")
                # print(json.dumps({
                #     "parent_state_id": parent_state_id,
                #     "root_id": record.get("root_id"),
                #     "target_value": record.get("target_value"),
                #     "best_action_index": record.get("best_action_index"),
                #     "best_action_repr": record.get("best_action_repr"),
                #     "best_reward": record.get("best_reward"),
                #     "best_discount": record.get("best_discount"),
                #     "best_bootstrap": record.get("best_bootstrap"),
                #     "best_child_cost": record.get("best_child_cost"),
                #     "best_child_time": record.get("best_child_time"),
                #     "target_abs_threshold": record.get("target_abs_threshold"),
                #     "is_nonzero_target": record.get("is_nonzero_target"),
                #     "is_selected_target": record.get("is_selected_target"),
                # }, indent=2, default=str))

                parent_state = state_loader(record)

                children = generate_child_states(
                    parent_state_id=parent_state_id,
                    root_record=record,
                    root_parent_state=parent_state,
                    mcts=state_loader.mcts,
                    include_alias_rows=bool(cfg.include_alias_rows),
                )

                # print("Generated Children :")
                # for child in children:
                    # if int(child["action_index"]) == int(record.get("best_action_index", -1)):
                    #     print("Best Child:")
                    #     print(json.dumps(child, indent=2, default=str))
                    # else :
                    #     print("Other child !!!")



                num_parents += 1
                num_transitions += len(children)
                num_canonical_transitions += sum(
                    1
                    for row in children
                    if int(row["action_index"]) == int(row["canonical_action_index"])
                )

                if validation_remaining > 0 and children:
                    parent_records_for_validation[int(parent_state_id)] = record
                    checks = min(validation_remaining, len(children))
                    test_child_states(
                        children[:checks],
                        parent_records_by_id=parent_records_for_validation,
                        state_loader=state_loader,
                        max_checks=checks,
                    )
                    validation_remaining -= checks

                shard_buffer.extend(children)

                while len(shard_buffer) >= int(cfg.transition_shard_size):
                    to_write = shard_buffer[: int(cfg.transition_shard_size)]
                    shard_buffer = shard_buffer[int(cfg.transition_shard_size) :]

                    entry = store_child_states(
                        to_write,
                        output_dir=cfg.output_dir,
                        shard_index=shard_index,
                    )
                    manifest_f.write(json.dumps(entry) + "\n")
                    manifest_f.flush()
                    shard_index += 1

            if shard_buffer:
                entry = store_child_states(
                    shard_buffer,
                    output_dir=cfg.output_dir,
                    shard_index=shard_index,
                )
                manifest_f.write(json.dumps(entry) + "\n")
                shard_index += 1

    finally:
        state_loader.close()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(cfg.dataset_dir),
        "output_dir": str(cfg.output_dir),
        "num_parent_states": int(num_parents),
        "parent_state_id_start": int(cfg.parent_state_id_start),
        "parent_state_id_end": cfg.parent_state_id_end,
        "worker_index": cfg.worker_index,
        "num_transitions": int(num_transitions),
        "num_canonical_transitions": int(num_canonical_transitions),
        "num_shards": int(shard_index),
        "include_alias_rows": bool(cfg.include_alias_rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary




def build_config_from_args(args: argparse.Namespace) -> RootChildGenerationConfig:
    dataset_dir = Path(args.dataset_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        dataset_dir.parent / f"{dataset_dir.name}_child_transitions"
    )

    return RootChildGenerationConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        max_roots=args.max_roots,
        root_player_filter=str(args.root_player_filter),
        transition_shard_size=int(args.transition_shard_size),
        include_alias_rows=bool(args.include_alias_rows),
        validate_first_n=int(args.validate_first_n),
        overwrite=bool(args.overwrite),
        seed=int(args.seed),
        num_processes=int(args.num_processes),
        parents_per_task=int(args.parents_per_task),
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GV3 root child-transition cache.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Existing ModelSearchBed root dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output child-transition cache dir. Defaults next to dataset dir.",
    )
    parser.add_argument("--max-roots", type=int, default=None)
    parser.add_argument("--root-player-filter", default="controller")
    parser.add_argument("--transition-shard-size", type=int, default=4096)
    parser.add_argument("--include-alias-rows", action="store_true")
    parser.add_argument("--validate-first-n", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--parents-per-task", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    cfg = build_config_from_args(parse_args())


    # print("Config passed is : ", cfg)
    summary = generate_child_cache(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()












