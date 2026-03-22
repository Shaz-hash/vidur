from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - runtime fallback
    pa = None
    pq = None

from .state_schema import (
    ActionRow,
    AnchorSampleRow,
    NodeRow,
    RequestRow,
    StateRow,
    TransitionRequestDeltaRow,
    TransitionRow,
)


@dataclass(frozen=True)
class WorkerShardPaths:
    worker_id: int
    round_idx: int
    state_path: str
    request_path: str
    action_path: str
    transition_path: str
    node_path: str
    transition_delta_path: str
    anchor_path: str


def _records_from_rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = row.as_record() if hasattr(row, "as_record") else dict(row)
        out.append(rec)
    return out


def write_parquet_records(path: Path, records: Sequence[Dict[str, Any]], *, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pa is None or pq is None:
        # deterministic JSON fallback when pyarrow is unavailable
        with path.with_suffix(path.suffix + ".jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True))
                f.write("\n")
        return

    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, path, compression=compression)


def read_parquet_records(path: Path) -> List[Dict[str, Any]]:
    if pa is None or pq is None:
        p = path.with_suffix(path.suffix + ".jsonl")
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                out.append(json.loads(s))
        return out

    if not path.exists():
        return []
    table = pq.read_table(path)
    return table.to_pylist()


def write_worker_shard(
    *,
    out_dir: Path,
    worker_id: int,
    round_idx: int,
    compression: str,
    states: Sequence[StateRow],
    requests: Sequence[RequestRow],
    actions: Sequence[ActionRow],
    transitions: Sequence[TransitionRow],
    nodes: Sequence[NodeRow],
    transition_deltas: Sequence[TransitionRequestDeltaRow],
    anchors: Sequence[AnchorSampleRow],
) -> WorkerShardPaths:
    base = out_dir / f"worker_{int(worker_id):02d}"
    base.mkdir(parents=True, exist_ok=True)

    state_path = base / "states.parquet"
    request_path = base / "requests.parquet"
    action_path = base / "actions.parquet"
    transition_path = base / "transitions.parquet"
    node_path = base / "nodes.parquet"
    transition_delta_path = base / "transition_request_deltas.parquet"
    anchor_path = base / "anchors.parquet"

    write_parquet_records(state_path, _records_from_rows(states), compression=compression)
    write_parquet_records(request_path, _records_from_rows(requests), compression=compression)
    write_parquet_records(action_path, _records_from_rows(actions), compression=compression)
    write_parquet_records(transition_path, _records_from_rows(transitions), compression=compression)
    write_parquet_records(node_path, _records_from_rows(nodes), compression=compression)
    write_parquet_records(
        transition_delta_path,
        _records_from_rows(transition_deltas),
        compression=compression,
    )
    write_parquet_records(anchor_path, _records_from_rows(anchors), compression=compression)

    return WorkerShardPaths(
        worker_id=int(worker_id),
        round_idx=int(round_idx),
        state_path=str(state_path),
        request_path=str(request_path),
        action_path=str(action_path),
        transition_path=str(transition_path),
        node_path=str(node_path),
        transition_delta_path=str(transition_delta_path),
        anchor_path=str(anchor_path),
    )


def _dedup_records(
    records: Iterable[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], Tuple[Any, ...]],
) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for rec in records:
        k = key_fn(rec)
        if k not in seen:
            seen[k] = rec
    keys = sorted(seen.keys())
    return [seen[k] for k in keys]


@dataclass(frozen=True)
class MergeOutput:
    merged_state_path: str
    merged_request_path: str
    merged_action_path: str
    merged_transition_path: str
    merged_node_path: str
    merged_transition_delta_path: str
    merged_anchor_path: str
    manifest_path: str
    unique_states: int
    unique_anchor_samples: int
    controller_lp_samples: int
    adversary_lp_samples: int


def merge_worker_shards(
    *,
    out_dir: Path,
    round_idx: int,
    shards: Sequence[WorkerShardPaths],
    compression: str,
) -> MergeOutput:
    merged_dir = out_dir / f"round_{int(round_idx):03d}" / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    all_states: List[Dict[str, Any]] = []
    all_requests: List[Dict[str, Any]] = []
    all_actions: List[Dict[str, Any]] = []
    all_transitions: List[Dict[str, Any]] = []
    all_nodes: List[Dict[str, Any]] = []
    all_deltas: List[Dict[str, Any]] = []
    all_anchors: List[Dict[str, Any]] = []

    for shard in sorted(shards, key=lambda s: (s.round_idx, s.worker_id)):
        all_states.extend(read_parquet_records(Path(shard.state_path)))
        all_requests.extend(read_parquet_records(Path(shard.request_path)))
        all_actions.extend(read_parquet_records(Path(shard.action_path)))
        all_transitions.extend(read_parquet_records(Path(shard.transition_path)))
        all_nodes.extend(read_parquet_records(Path(shard.node_path)))
        all_deltas.extend(read_parquet_records(Path(shard.transition_delta_path)))
        all_anchors.extend(read_parquet_records(Path(shard.anchor_path)))

    states = _dedup_records(all_states, key_fn=lambda r: (str(r["state_id"]),))
    requests = _dedup_records(all_requests, key_fn=lambda r: (str(r["state_id"]), int(r["request_id"])))
    actions = _dedup_records(all_actions, key_fn=lambda r: (str(r["action_id"]),))
    transitions = _dedup_records(all_transitions, key_fn=lambda r: (str(r["transition_id"]),))
    nodes = _dedup_records(all_nodes, key_fn=lambda r: (str(r["node_id"]),))
    deltas = _dedup_records(
        all_deltas,
        key_fn=lambda r: (str(r["transition_id"]), int(r["request_id"])),
    )
    anchors = _dedup_records(all_anchors, key_fn=lambda r: (str(r["anchor_state_id"]),))

    state_path = merged_dir / "states.parquet"
    request_path = merged_dir / "requests.parquet"
    action_path = merged_dir / "actions.parquet"
    transition_path = merged_dir / "transitions.parquet"
    node_path = merged_dir / "nodes.parquet"
    delta_path = merged_dir / "transition_request_deltas.parquet"
    anchor_path = merged_dir / "anchors.parquet"

    write_parquet_records(state_path, states, compression=compression)
    write_parquet_records(request_path, requests, compression=compression)
    write_parquet_records(action_path, actions, compression=compression)
    write_parquet_records(transition_path, transitions, compression=compression)
    write_parquet_records(node_path, nodes, compression=compression)
    write_parquet_records(delta_path, deltas, compression=compression)
    write_parquet_records(anchor_path, anchors, compression=compression)

    controller_lp_samples = int(sum(1 for r in transitions if str(r.get("actor", "")) == "controller"))
    adversary_lp_samples = int(sum(1 for r in transitions if str(r.get("actor", "")) == "adversary"))

    manifest = {
        "round_idx": int(round_idx),
        "num_worker_shards": int(len(shards)),
        "counts": {
            "states": int(len(states)),
            "requests": int(len(requests)),
            "actions": int(len(actions)),
            "transitions": int(len(transitions)),
            "nodes": int(len(nodes)),
            "transition_request_deltas": int(len(deltas)),
            "anchor_samples": int(len(anchors)),
            "controller_lp_samples": int(controller_lp_samples),
            "adversary_lp_samples": int(adversary_lp_samples),
        },
        "paths": {
            "states": str(state_path),
            "requests": str(request_path),
            "actions": str(action_path),
            "transitions": str(transition_path),
            "nodes": str(node_path),
            "transition_request_deltas": str(delta_path),
            "anchors": str(anchor_path),
        },
    }
    manifest_path = merged_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return MergeOutput(
        merged_state_path=str(state_path),
        merged_request_path=str(request_path),
        merged_action_path=str(action_path),
        merged_transition_path=str(transition_path),
        merged_node_path=str(node_path),
        merged_transition_delta_path=str(delta_path),
        merged_anchor_path=str(anchor_path),
        manifest_path=str(manifest_path),
        unique_states=int(len(states)),
        unique_anchor_samples=int(len(anchors)),
        controller_lp_samples=int(controller_lp_samples),
        adversary_lp_samples=int(adversary_lp_samples),
    )
