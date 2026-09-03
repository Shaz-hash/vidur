"""Write one consistent four-file diagnostic snapshot per GV4 MCTS node.

The logger reads authoritative ``GV4State`` snapshots already stored on MCTS
nodes. It never advances the simulator, resolves actions, or mutates the tree.
JSON cells are complete, deterministic, and intentionally not based on ``repr``
strings so test tools can parse them without knowing Python object formatting.
"""

from __future__ import annotations

import csv
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..action_resolver import (
    CanonicalAdversaryAction,
    CanonicalControllerAction,
)
from ..config import GV4EngineConfig
from ..state import (
    GV4State,
    NO_ID,
    RequestLifecycle,
    RequestState,
    TerminalReason,
    UNSET_TIME,
)


__all__ = ["GV4LoggerError", "GV4NodeCSVLogger"]


LOGGER_SCHEMA_VERSION = "gv4_node_logs_v1"

_COMMON_FIELDS = [
    "logger_schema_version",
    "run_id",
    "game_id",
    "root_id",
    "node_id",
    "state_schema_version",
    "config_manifest_sha256",
    "state_hash",
    "state_time",
]

_MCTS_FIELDS = _COMMON_FIELDS + [
    "root_node_id",
    "parent_node_id",
    "depth",
    "root_player",
    "player_acted",
    "next_player",
    "incoming_action_json",
    "raw_action_index",
    "canonical_action_index",
    "resolver_canonical_action_index",
    "alias_indices_json",
    "parent_time",
    "child_time",
    "next_adversary_tick",
    "requests_generated",
    "requests_completed",
    "requests_stopped",
    "requests_dropped",
    "slo_violations",
    "prefill_lateness_sec",
    "decode_lateness_sec",
    "total_lateness_sec",
    "terminal_cost",
    "objective_total_cost",
    "reward",
    "discount",
    "next_request_id",
    "next_microbatch_id",
    "adversary_launch_history_json",
    "total_completed_requests",
    "total_evicted_requests",
    "total_requests_in_system",
    "total_stopped_requests",
    "total_stop_pending_requests",
    "total_drop_pending_requests",
    "decode_credits_available",
    "decode_credits_reserved",
    "visits",
    "value_sum",
    "mean_value",
    "canonical_action_indices_json",
    "valid_mask_json",
]

_REQUEST_FIELDS = _COMMON_FIELDS + [
    "completed_count",
    "in_progress_count",
    "evicted_count",
    "stopped_count",
    "completed_requests_json",
    "in_progress_requests_json",
    "evicted_requests_json",
    "stopped_requests_json",
]

_PIPELINE_BASE_FIELDS = _COMMON_FIELDS + [
    "replica_count",
    "pipeline_parallel_size",
    "total_inflight_microbatches",
    "inflight_microbatch_ids_json",
]

_KV_BASE_FIELDS = _COMMON_FIELDS + [
    "block_size_tokens",
    "total_capacity_blocks",
    "total_available_blocks",
    "total_consumed_blocks",
    "total_reserved_blocks",
    "total_occupied_blocks",
    "total_tokens_in_memory",
    "total_available_tokens",
    "total_free_block_token_slots",
]


class GV4LoggerError(RuntimeError):
    """Raised when a node cannot be represented as a consistent GV4 snapshot."""


class _CsvSink:
    """Small CSV writer with a fixed schema and predictable flush behavior."""

    def __init__(
        self,
        path: Path,
        fieldnames: list[str],
        *,
        flush_every: int,
        overwrite: bool,
    ) -> None:
        mode = "w" if overwrite else "x"
        self._file = path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        self._writer.writeheader()
        self._flush_every = flush_every
        self._rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        self._file.close()


def _to_plain_data(value: Any) -> Any:
    """Convert state/action records into deterministic JSON-compatible data."""

    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return {
            item.name: _to_plain_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _to_plain_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json(value: Any) -> str:
    return json.dumps(
        _to_plain_data(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _state_hash(state: GV4State) -> str:
    return hashlib.sha256(_json(state).encode("utf-8")).hexdigest()


def _optional_time(value: float) -> float | None:
    return None if value == UNSET_TIME else value


def _player_name(value: Any) -> str:
    if isinstance(value, Enum):
        return value.name.lower()
    return str(value).lower()


def _action_payload(action: Any) -> dict[str, Any] | None:
    if action is None:
        return None

    if isinstance(action, CanonicalControllerAction):
        resolved = action.action
        return {
            "actor": "controller",
            "raw_action_index": resolved.raw_action_index,
            "canonical_action_index": action.canonical_action_index,
            "alias_indices": list(action.equivalent_raw_indices),
            "replica_id": resolved.replica_id,
            "eviction_rule": resolved.eviction_rule,
            "prefill_budget": resolved.prefill_budget,
            "ordering_heuristic": resolved.ordering_heuristic,
            "transition_kind": resolved.transition_kind.name,
            "evicted_request_ids": list(resolved.evicted_request_ids),
            "allocations": [_to_plain_data(item) for item in resolved.allocations],
            "released_kv_blocks": resolved.released_kv_blocks,
            "reserved_kv_blocks": resolved.reserved_kv_blocks,
            "rank_kv_delta": [list(item) for item in resolved.rank_kv_delta],
        }

    if isinstance(action, CanonicalAdversaryAction):
        resolved = action.action
        return {
            "actor": "adversary",
            "raw_action_index": resolved.raw_action_index,
            "canonical_action_index": action.canonical_action_index,
            "alias_indices": list(action.equivalent_raw_indices),
            "launch_count": resolved.launch_count,
            "prefill_tokens": resolved.prefill_tokens,
            "stop_rule": resolved.stop_rule,
            "stop_request_ids": list(resolved.stop_request_ids),
        }

    raise GV4LoggerError(f"unsupported incoming action type: {type(action).__name__}")


def _request_phase(request: RequestState) -> str:
    if request.remaining_prefill_tokens or request.reserved_prefill_tokens:
        return "prefill"
    return "decode"


def _request_payload(request: RequestState) -> dict[str, Any]:
    """Serialize all request progress needed for lifecycle/SLO assertions."""

    return {
        "request_id": request.request_id,
        "owner_replica_id": request.owner_replica_id,
        "lifecycle": request.lifecycle.name,
        "current_phase": _request_phase(request),
        "arrival_time": request.arrival_time,
        "completion_time": _optional_time(request.terminal_time),
        "terminal_requested_at": _optional_time(request.terminal_requested_at),
        "terminal_reason": request.terminal_reason.name,
        "original_prefill_tokens": request.original_prefill_tokens,
        "committed_prefill_tokens": request.committed_prefill_tokens,
        "reserved_prefill_tokens": request.reserved_prefill_tokens,
        "remaining_prefill_tokens": request.remaining_prefill_tokens,
        "original_decode_tokens": request.original_decode_tokens,
        "committed_decode_tokens": request.committed_decode_tokens,
        "reserved_decode_tokens": request.reserved_decode_tokens,
        "remaining_decode_tokens": request.remaining_decode_tokens,
        "total_committed_tokens": (
            request.committed_prefill_tokens + request.committed_decode_tokens
        ),
        "resident_tokens": request.resident_tokens,
        "committed_kv_blocks": request.committed_kv_blocks,
        "reserved_kv_blocks": request.reserved_kv_blocks,
        "inflight_microbatch_id": (
            None
            if request.inflight_microbatch_id == NO_ID
            else request.inflight_microbatch_id
        ),
        "prefill_deadline": request.prefill_deadline,
        "decode_token_slo_sec": request.decode_token_slo_sec,
        "next_decode_deadline": _optional_time(request.next_decode_deadline),
        "prefill_lateness_sec": request.prefill_lateness_sec,
        "decode_lateness_sec": request.decode_lateness_sec,
        "violation_recorded": request.violation_recorded,
        "decode_credit_minted": request.decode_credit_minted,
    }


def _request_groups(state: GV4State) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "completed": [],
        "in_progress": [],
        "evicted": [],
        "stopped": [],
    }
    for request in state.requests:
        payload = _request_payload(request)
        if request.lifecycle == RequestLifecycle.COMPLETED:
            groups["completed"].append(payload)
        elif request.lifecycle == RequestLifecycle.DROPPED:
            # terminal_reason distinguishes controller eviction from automatic drop
            groups["evicted"].append(payload)
        elif request.lifecycle == RequestLifecycle.STOPPED:
            groups["stopped"].append(payload)
        else:
            # Pending stop/drop records remain physically in the system until drain.
            groups["in_progress"].append(payload)
    return groups


def _canonical_indices(node: Any) -> list[int]:
    aliases = getattr(node, "canonical_to_action_aliases", {}) or {}
    if aliases:
        return sorted(int(index) for index in aliases)

    indices = {
        int(action.canonical_action_index)
        for action in (getattr(node, "actions_by_index", []) or [])
        if action is not None and hasattr(action, "canonical_action_index")
    }
    return sorted(indices)


def _node_snapshot(node: Any) -> GV4State:
    state = getattr(node, "cached_sim_snapshot", None)
    if not isinstance(state, GV4State):
        raise GV4LoggerError(
            f"node {getattr(node, 'node_id', '?')} lacks a GV4State snapshot"
        )
    return state


class GV4NodeCSVLogger:
    """Write MCTS, request, pipeline, and KV rows joined by one node key."""

    FILE_NAMES = {
        "mcts": "mcts_nodes.csv",
        "requests": "requests.csv",
        "pipeline": "pipeline_nodes.csv",
        "kv": "kv_cache_nodes.csv",
    }

    def __init__(
        self,
        output_dir: str | Path,
        config: GV4EngineConfig,
        *,
        flush_every: int = 1,
        overwrite: bool = False,
        validate_states: bool = True,
    ) -> None:
        if (
            isinstance(flush_every, bool)
            or not isinstance(flush_every, int)
            or flush_every <= 0
        ):
            raise ValueError("flush_every must be a positive integer")

        self.config = config
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validate_states = bool(validate_states)
        self._seen_keys: set[tuple[str, str, str, str]] = set()

        self._stage_columns = [
            f"replica_{replica_id}_stage_{stage_index}_json"
            for replica_id in range(config.topology.num_replicas)
            for stage_index in range(config.topology.pipeline_parallel_size)
        ]
        self._rank_columns = [
            f"rank_{rank_id}_json" for rank_id in range(config.topology.total_ranks)
        ]

        paths = {
            name: self.output_dir / filename
            for name, filename in self.FILE_NAMES.items()
        }
        if not overwrite:
            existing = [str(path) for path in paths.values() if path.exists()]
            if existing:
                raise FileExistsError(
                    "GV4 logger output already exists: " + ", ".join(existing)
                )

        schemas = {
            "mcts": _MCTS_FIELDS,
            "requests": _REQUEST_FIELDS,
            "pipeline": _PIPELINE_BASE_FIELDS + self._stage_columns,
            "kv": _KV_BASE_FIELDS + self._rank_columns,
        }
        self._sinks: dict[str, _CsvSink] = {}
        try:
            for name in ("mcts", "requests", "pipeline", "kv"):
                self._sinks[name] = _CsvSink(
                    paths[name],
                    schemas[name],
                    flush_every=flush_every,
                    overwrite=overwrite,
                )
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "GV4NodeCSVLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for sink in getattr(self, "_sinks", {}).values():
            sink.close()

    def log_tree(
        self,
        *,
        run_id: int | str,
        game_id: int | str,
        root_id: int | str,
        root: Any,
    ) -> int:
        """Log every currently materialized node in deterministic DFS order."""

        root_player = _player_name(getattr(root, "player", ""))
        root_node_id = int(getattr(root, "node_id"))
        stack = [root]
        seen_node_ids: set[int] = set()
        rows = 0

        while stack:
            node = stack.pop()
            node_id = int(getattr(node, "node_id"))
            if node_id in seen_node_ids:
                raise GV4LoggerError(f"tree contains duplicate node ID {node_id}")
            seen_node_ids.add(node_id)

            self.log_node(
                run_id=run_id,
                game_id=game_id,
                root_id=root_id,
                root_node_id=root_node_id,
                root_player=root_player,
                node=node,
            )
            rows += 1

            children = getattr(node, "children", {}) or {}
            ordered_children = sorted(
                children.items(), key=lambda item: int(item[0]), reverse=True
            )
            for edge_index, child in ordered_children:
                if getattr(child, "parent", None) is not node:
                    raise GV4LoggerError("child points to a different parent")
                if int(getattr(child, "parent_action_index")) != int(edge_index):
                    raise GV4LoggerError("child edge index and parent link disagree")
                stack.append(child)

        return rows

    def log_path(
        self,
        *,
        run_id: int | str,
        game_id: int | str,
        root_id: int | str,
        path: Sequence[Any],
    ) -> int:
        """Log one exact MCTS simulation path after its backpropagation.

        A node may appear in multiple simulation paths. ``run_id`` distinguishes
        those observations, so its visits and value statistics show their value
        after that specific MCTS iteration rather than only at the final tree.
        """

        if not path:
            raise GV4LoggerError("an MCTS iteration path cannot be empty")

        root = path[0]
        root_node_id = int(getattr(root, "node_id"))
        root_player = _player_name(getattr(root, "player", ""))
        seen_node_ids: set[int] = set()

        for index, node in enumerate(path):
            node_id = int(getattr(node, "node_id"))
            if node_id in seen_node_ids:
                raise GV4LoggerError(
                    f"MCTS iteration path contains node {node_id} more than once"
                )
            seen_node_ids.add(node_id)

            if index == 0:
                if getattr(node, "parent", None) is not None:
                    raise GV4LoggerError("iteration path does not begin at a root")
            else:
                parent = path[index - 1]
                if getattr(node, "parent", None) is not parent:
                    raise GV4LoggerError("iteration path contains a broken parent link")
                edge_index = int(getattr(node, "parent_action_index"))
                if (getattr(parent, "children", {}) or {}).get(edge_index) is not node:
                    raise GV4LoggerError("iteration path edge is absent from the tree")

            self.log_node(
                run_id=run_id,
                game_id=game_id,
                root_id=root_id,
                root_node_id=root_node_id,
                root_player=root_player,
                node=node,
            )

        return len(path)

    def log_node(
        self,
        *,
        run_id: int | str,
        game_id: int | str,
        root_id: int | str,
        root_node_id: int,
        root_player: str,
        node: Any,
    ) -> None:
        """Build all four records first, then append one joined node snapshot."""

        state = _node_snapshot(node)
        if state.config_manifest_sha256 != self.config.manifest_sha256():
            raise GV4LoggerError("node state belongs to a different GV4 config")
        if self.validate_states:
            state.assert_valid(self.config)

        node_id = int(getattr(node, "node_id"))
        key = tuple(str(value) for value in (run_id, game_id, root_id, node_id))
        if key in self._seen_keys:
            raise GV4LoggerError(f"duplicate logger key {key}")

        common = {
            "logger_schema_version": LOGGER_SCHEMA_VERSION,
            "run_id": run_id,
            "game_id": game_id,
            "root_id": root_id,
            "node_id": node_id,
            "state_schema_version": state.state_schema_version,
            "config_manifest_sha256": state.config_manifest_sha256,
            "state_hash": _state_hash(state),
            "state_time": state.now,
        }
        groups = _request_groups(state)

        rows = {
            "mcts": self._mcts_row(
                common, node, state, groups, root_node_id, root_player
            ),
            "requests": self._requests_row(common, groups),
            "pipeline": self._pipeline_row(common, state),
            "kv": self._kv_row(common, state),
        }

        for name in ("mcts", "requests", "pipeline", "kv"):
            self._sinks[name].write(rows[name])
        self._seen_keys.add(key)

    def _mcts_row(
        self,
        common: dict[str, Any],
        node: Any,
        state: GV4State,
        groups: dict[str, list[dict[str, Any]]],
        root_node_id: int,
        root_player: str,
    ) -> dict[str, Any]:
        parent = getattr(node, "parent", None)
        action = getattr(node, "parent_action", None)
        action_payload = _action_payload(action)
        objective = state.objective

        parent_time: float | str = ""
        parent_node_id: int | str = ""
        player_acted = ""
        raw_action_index: int | str = ""
        canonical_action_index: int | str = ""
        resolver_canonical_action_index: int | str = ""
        aliases: list[int] = []
        if parent is not None:
            parent_state = _node_snapshot(parent)
            parent_time = parent_state.now
            parent_node_id = int(getattr(parent, "node_id"))
            player_acted = _player_name(getattr(parent, "player", ""))
        if action_payload is not None:
            raw_action_index = int(action_payload["raw_action_index"])
            resolver_canonical_action_index = int(
                action_payload["canonical_action_index"]
            )
            canonical_action_index = int(getattr(node, "parent_action_index"))
            aliases = [int(index) for index in action_payload["alias_indices"]]
            action_payload = {
                **action_payload,
                "mcts_canonical_action_index": canonical_action_index,
                "resolver_canonical_action_index": resolver_canonical_action_index,
            }

        total_evicted = sum(
            request.terminal_reason == TerminalReason.CONTROLLER_EVICTION
            and request.lifecycle
            in (RequestLifecycle.DROP_PENDING, RequestLifecycle.DROPPED)
            for request in state.requests
        )
        requests_in_system = sum(
            not request.lifecycle.is_terminal for request in state.requests
        )

        visits = int(getattr(node, "visits", 0))
        value_sum = float(getattr(node, "value_sum", 0.0))
        mean_value = value_sum / visits if visits else 0.0
        valid_mask = [bool(value) for value in (getattr(node, "valid_mask", []) or [])]

        return {
            **common,
            "root_node_id": root_node_id,
            "parent_node_id": parent_node_id,
            "depth": int(getattr(node, "depth", 0)),
            "root_player": root_player,
            "player_acted": player_acted,
            "next_player": _player_name(state.next_player),
            "incoming_action_json": _json(action_payload),
            "raw_action_index": raw_action_index,
            "canonical_action_index": canonical_action_index,
            "resolver_canonical_action_index": resolver_canonical_action_index,
            "alias_indices_json": _json(aliases),
            "parent_time": parent_time,
            "child_time": state.now,
            "next_adversary_tick": state.next_adversary_tick,
            "requests_generated": objective.requests_generated,
            "requests_completed": objective.requests_completed,
            "requests_stopped": objective.requests_stopped,
            "requests_dropped": objective.requests_dropped,
            "slo_violations": objective.slo_violations,
            "prefill_lateness_sec": objective.prefill_lateness_sec,
            "decode_lateness_sec": objective.decode_lateness_sec,
            "total_lateness_sec": (
                objective.prefill_lateness_sec + objective.decode_lateness_sec
            ),
            "terminal_cost": objective.terminal_cost,
            "objective_total_cost": objective.total_cost,
            "reward": float(getattr(node, "reward", 0.0)),
            "discount": float(getattr(node, "edge_discount", 1.0)),
            "next_request_id": state.next_request_id,
            "next_microbatch_id": state.next_microbatch_id,
            "adversary_launch_history_json": _json(state.launch_history),
            "total_completed_requests": len(groups["completed"]),
            "total_evicted_requests": total_evicted,
            "total_requests_in_system": requests_in_system,
            "total_stopped_requests": len(groups["stopped"]),
            "total_stop_pending_requests": sum(
                request.lifecycle == RequestLifecycle.STOP_PENDING
                for request in state.requests
            ),
            "total_drop_pending_requests": sum(
                request.lifecycle == RequestLifecycle.DROP_PENDING
                for request in state.requests
            ),
            "decode_credits_available": state.decode_credits_available,
            "decode_credits_reserved": state.decode_credits_reserved,
            "visits": visits,
            "value_sum": value_sum,
            "mean_value": mean_value,
            "canonical_action_indices_json": _json(_canonical_indices(node)),
            "valid_mask_json": _json(valid_mask),
        }

    def _requests_row(
        self,
        common: dict[str, Any],
        groups: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            **common,
            "completed_count": len(groups["completed"]),
            "in_progress_count": len(groups["in_progress"]),
            "evicted_count": len(groups["evicted"]),
            "stopped_count": len(groups["stopped"]),
            "completed_requests_json": _json(groups["completed"]),
            "in_progress_requests_json": _json(groups["in_progress"]),
            "evicted_requests_json": _json(groups["evicted"]),
            "stopped_requests_json": _json(groups["stopped"]),
        }

    def _pipeline_row(
        self,
        common: dict[str, Any],
        state: GV4State,
    ) -> dict[str, Any]:
        batch_ids = {
            str(replica.replica_id): [
                batch.microbatch_id for batch in replica.inflight_microbatches
            ]
            for replica in state.replicas
        }
        row: dict[str, Any] = {
            **common,
            "replica_count": len(state.replicas),
            "pipeline_parallel_size": self.config.topology.pipeline_parallel_size,
            "total_inflight_microbatches": sum(
                replica.inflight_count for replica in state.replicas
            ),
            "inflight_microbatch_ids_json": _json(batch_ids),
        }

        for replica in state.replicas:
            for stage_index in range(replica.pipeline_parallel_size):
                column = f"replica_{replica.replica_id}_stage_{stage_index}_json"
                row[column] = _json(
                    self._stage_payload(state, replica.replica_id, stage_index)
                )
        return row

    def _stage_payload(
        self,
        state: GV4State,
        replica_id: int,
        stage_index: int,
    ) -> dict[str, Any]:
        replica = state.replica(replica_id)
        calendars: list[dict[str, Any]] = []
        active: list[dict[str, Any]] = []

        for batch in replica.inflight_microbatches:
            ready = batch.stage_ready_times[stage_index]
            start = batch.stage_start_times[stage_index]
            finish = batch.stage_finish_times[stage_index]
            if state.now + self.config.timing.epsilon < start:
                status = "queued"
            elif state.now < finish - self.config.timing.epsilon:
                status = "active"
            else:
                status = "stage_complete"

            request_tokens = {
                str(item.request_id): {
                    "prefill_tokens": item.prefill_tokens,
                    "decode_tokens": item.decode_tokens,
                    "total_tokens": item.total_tokens,
                }
                for item in batch.allocations
            }
            payload = {
                "microbatch_id": batch.microbatch_id,
                "raw_action_index": batch.raw_action_index,
                "canonical_action_index": batch.canonical_action_index,
                "request_tokens": request_tokens,
                "ready_time": ready,
                "start_time": start,
                "stage_completion_time": finish,
                "batch_completion_time": batch.final_completion_time,
                "status": status,
            }
            calendars.append(payload)
            if status == "active":
                active.append(payload)

        if len(active) > 1:
            raise GV4LoggerError(
                f"replica {replica_id} stage {stage_index} has overlapping active batches"
            )

        last_batch_id = replica.stage_last_microbatch_ids[stage_index]
        return {
            "stage_index": stage_index,
            "tail_finish_time": replica.stage_tail_finish_times[stage_index],
            "last_microbatch_id": None if last_batch_id == NO_ID else last_batch_id,
            "active_batch": active[0] if active else None,
            "inflight_batch_calendars": calendars,
        }

    def _kv_row(
        self,
        common: dict[str, Any],
        state: GV4State,
    ) -> dict[str, Any]:
        block_size = self.config.kv_cache.block_size_tokens
        resident_by_replica = [0] * len(state.replicas)
        for request in state.requests:
            if request.lifecycle.is_terminal or request.owner_replica_id < 0:
                continue
            resident_by_replica[request.owner_replica_id] += request.resident_tokens

        total_capacity = 0
        total_available = 0
        total_committed = 0
        total_reserved = 0
        rank_payloads: dict[int, dict[str, Any]] = {}

        for replica in state.replicas:
            committed_values = set(replica.rank_kv_committed_blocks)
            reserved_values = set(replica.rank_kv_reserved_blocks)
            if len(committed_values) != 1 or len(reserved_values) != 1:
                raise GV4LoggerError(
                    f"replica {replica.replica_id} rank KV mirrors disagree"
                )

            logical_capacity = min(replica.rank_kv_capacity_blocks)
            logical_committed = replica.rank_kv_committed_blocks[0]
            logical_reserved = replica.rank_kv_reserved_blocks[0]
            logical_available = min(replica.free_kv_blocks_by_rank())
            total_capacity += logical_capacity
            total_available += logical_available
            total_committed += logical_committed
            total_reserved += logical_reserved

            resident_tokens = resident_by_replica[replica.replica_id]
            for local_index, rank_id in enumerate(replica.rank_ids):
                capacity = replica.rank_kv_capacity_blocks[local_index]
                committed = replica.rank_kv_committed_blocks[local_index]
                reserved = replica.rank_kv_reserved_blocks[local_index]
                available = capacity - committed - reserved
                rank_payloads[rank_id] = {
                    "rank_id": rank_id,
                    "replica_id": replica.replica_id,
                    "pipeline_stage": self.config.topology.stage_index_for_rank(
                        rank_id
                    ),
                    "capacity_blocks": capacity,
                    "available_blocks": available,
                    "consumed_blocks": committed,
                    "reserved_blocks": reserved,
                    "occupied_blocks": committed + reserved,
                    "resident_tokens": resident_tokens,
                    "capacity_token_slots": capacity * block_size,
                    "available_token_slots": capacity * block_size - resident_tokens,
                    "free_block_token_slots": available * block_size,
                }

        total_resident_tokens = sum(resident_by_replica)
        row: dict[str, Any] = {
            **common,
            "block_size_tokens": block_size,
            "total_capacity_blocks": total_capacity,
            "total_available_blocks": total_available,
            "total_consumed_blocks": total_committed,
            "total_reserved_blocks": total_reserved,
            "total_occupied_blocks": total_committed + total_reserved,
            "total_tokens_in_memory": total_resident_tokens,
            "total_available_tokens": total_capacity * block_size
            - total_resident_tokens,
            "total_free_block_token_slots": total_available * block_size,
        }
        for rank_id in range(self.config.topology.total_ranks):
            payload = rank_payloads.get(rank_id)
            if payload is None:
                raise GV4LoggerError(f"state is missing configured rank {rank_id}")
            row[f"rank_{rank_id}_json"] = _json(payload)
        return row
