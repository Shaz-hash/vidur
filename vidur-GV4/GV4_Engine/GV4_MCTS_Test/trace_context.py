"""Strict four-file joins and reconstruction of logged GV4 node states."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..action_resolver import (
    ControllerTransitionKind,
    ResolvedAdversaryAction,
    ResolvedControllerAction,
)
from ..config import GV4EngineConfig
from ..state import (
    BatchAllocation,
    GV4State,
    InflightMicrobatchState,
    LaunchRecord,
    NO_ID,
    ObjectiveState,
    Player,
    ReplicaState,
    RequestLifecycle,
    RequestState,
    TerminalReason,
    UNSET_TIME,
)
from .trace_capture import KEY_FIELDS, LOG_FILE_NAMES


class TraceContextError(RuntimeError):
    """Raised when the four files cannot describe one unambiguous path."""


@dataclass(frozen=True, slots=True, order=True)
class NodeKey:
    run_id: int
    game_id: int
    root_id: int
    node_id: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "NodeKey":
        try:
            return cls(*(int(row[field]) for field in KEY_FIELDS))
        except (KeyError, ValueError) as error:
            raise TraceContextError("invalid composite node key") from error


def _json_cell(row: dict[str, str], field: str) -> Any:
    try:
        return json.loads(row[field])
    except (KeyError, json.JSONDecodeError) as error:
        raise TraceContextError(f"invalid JSON column {field}") from error


def _optional_time(value: Any) -> float:
    return UNSET_TIME if value is None else float(value)


@dataclass(frozen=True, slots=True)
class TraceNode:
    key: NodeKey
    mcts: dict[str, str]
    requests: dict[str, str]
    pipeline: dict[str, str]
    kv_cache: dict[str, str]

    @property
    def depth(self) -> int:
        return int(self.mcts["depth"])

    @property
    def now(self) -> float:
        return float(self.mcts["state_time"])

    @property
    def incoming_action(self) -> dict[str, Any] | None:
        value = _json_cell(self.mcts, "incoming_action_json")
        if value is not None and not isinstance(value, dict):
            raise TraceContextError("incoming action must be an object or null")
        return value

    @property
    def launch_history(self) -> list[dict[str, Any]]:
        value = _json_cell(self.mcts, "adversary_launch_history_json")
        if not isinstance(value, list):
            raise TraceContextError("launch history must be a JSON list")
        return value

    def request_groups(self) -> dict[str, list[dict[str, Any]]]:
        groups = {
            "completed": _json_cell(self.requests, "completed_requests_json"),
            "in_progress": _json_cell(self.requests, "in_progress_requests_json"),
            "evicted": _json_cell(self.requests, "evicted_requests_json"),
            "stopped": _json_cell(self.requests, "stopped_requests_json"),
        }
        if any(not isinstance(value, list) for value in groups.values()):
            raise TraceContextError("request group columns must be JSON lists")
        return groups

    def requests_by_id(self) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for requests in self.request_groups().values():
            for payload in requests:
                request_id = int(payload["request_id"])
                if request_id in result:
                    raise TraceContextError(f"request {request_id} appears twice")
                result[request_id] = payload
        return result

    def stage(self, replica_id: int, stage_index: int) -> dict[str, Any]:
        value = _json_cell(
            self.pipeline,
            f"replica_{replica_id}_stage_{stage_index}_json",
        )
        if not isinstance(value, dict):
            raise TraceContextError("pipeline stage column must be a JSON object")
        return value

    def rank(self, rank_id: int) -> dict[str, Any]:
        value = _json_cell(self.kv_cache, f"rank_{rank_id}_json")
        if not isinstance(value, dict):
            raise TraceContextError("rank column must be a JSON object")
        return value

    def resolved_controller_action(self) -> ResolvedControllerAction | None:
        payload = self.incoming_action
        if payload is None or payload.get("actor") != "controller":
            return None
        allocations = tuple(
            BatchAllocation(
                request_id=int(item["request_id"]),
                prefill_tokens=int(item["prefill_tokens"]),
                decode_tokens=int(item["decode_tokens"]),
                new_kv_blocks=int(item["new_kv_blocks"]),
            )
            for item in payload["allocations"]
        )
        return ResolvedControllerAction(
            raw_action_index=int(payload["raw_action_index"]),
            replica_id=int(payload["replica_id"]),
            eviction_rule=str(payload["eviction_rule"]),
            prefill_budget=int(payload["prefill_budget"]),
            ordering_heuristic=str(payload["ordering_heuristic"]),
            transition_kind=ControllerTransitionKind[payload["transition_kind"]],
            evicted_request_ids=tuple(int(value) for value in payload["evicted_request_ids"]),
            allocations=allocations,
            released_kv_blocks=int(payload["released_kv_blocks"]),
            reserved_kv_blocks=int(payload["reserved_kv_blocks"]),
            rank_kv_delta=tuple(
                (int(rank_id), int(delta)) for rank_id, delta in payload["rank_kv_delta"]
            ),
        )

    def resolved_adversary_action(self) -> ResolvedAdversaryAction | None:
        payload = self.incoming_action
        if payload is None or payload.get("actor") != "adversary":
            return None
        prefill = payload["prefill_tokens"]
        return ResolvedAdversaryAction(
            raw_action_index=int(payload["raw_action_index"]),
            launch_count=int(payload["launch_count"]),
            prefill_tokens=None if prefill is None else int(prefill),
            stop_rule=str(payload["stop_rule"]),
            stop_request_ids=tuple(int(value) for value in payload["stop_request_ids"]),
        )

    def reconstruct_state(self, config: GV4EngineConfig) -> GV4State:
        """Rebuild a real state object from the four joined CSV records."""

        request_payloads = self.requests_by_id()
        requests = [
            _request_state(request_payloads[request_id])
            for request_id in sorted(request_payloads)
        ]
        request_by_id = {request.request_id: request for request in requests}
        replicas = [
            self._reconstruct_replica(config, replica_id, request_by_id)
            for replica_id in range(config.topology.num_replicas)
        ]
        objective = ObjectiveState(
            requests_generated=int(self.mcts["requests_generated"]),
            requests_completed=int(self.mcts["requests_completed"]),
            requests_stopped=int(self.mcts["requests_stopped"]),
            requests_dropped=int(self.mcts["requests_dropped"]),
            slo_violations=int(self.mcts["slo_violations"]),
            prefill_lateness_sec=float(self.mcts["prefill_lateness_sec"]),
            decode_lateness_sec=float(self.mcts["decode_lateness_sec"]),
            terminal_cost=float(self.mcts["terminal_cost"]),
            total_cost=float(self.mcts["objective_total_cost"]),
        )
        minted_requests = sum(request.decode_credit_minted for request in requests)
        committed_decode = sum(request.committed_decode_tokens for request in requests)
        state = GV4State(
            state_schema_version=self.mcts["state_schema_version"],
            config_manifest_sha256=self.mcts["config_manifest_sha256"],
            now=self.now,
            next_player=Player[self.mcts["next_player"].upper()],
            next_adversary_tick=float(self.mcts["next_adversary_tick"]),
            next_request_id=int(self.mcts["next_request_id"]),
            next_microbatch_id=int(self.mcts["next_microbatch_id"]),
            tie_break_counter=0,
            rng_seed=config.global_seed,
            rng_counter=0,
            launch_history=[
                LaunchRecord(
                    launch_time=float(item["launch_time"]),
                    request_count=int(item["request_count"]),
                    prefill_tokens=int(item["prefill_tokens"]),
                )
                for item in self.launch_history
            ],
            decode_credits_available=int(self.mcts["decode_credits_available"]),
            decode_credits_reserved=int(self.mcts["decode_credits_reserved"]),
            decode_credits_minted_total=(
                minted_requests
                * config.credits.decode_credit_mint_per_prefill_completion
            ),
            decode_tokens_committed_total=committed_decode,
            requests=requests,
            replicas=replicas,
            objective=objective,
        )
        return state

    def _reconstruct_replica(
        self,
        config: GV4EngineConfig,
        replica_id: int,
        request_by_id: dict[int, RequestState],
    ) -> ReplicaState:
        placement = config.topology.replica_placements[replica_id]
        rank_payloads = [self.rank(rank_id) for rank_id in placement.rank_ids]
        stage_payloads = [
            self.stage(replica_id, stage)
            for stage in range(config.topology.pipeline_parallel_size)
        ]
        calendars_by_stage = [
            {int(item["microbatch_id"]): item for item in stage["inflight_batch_calendars"]}
            for stage in stage_payloads
        ]
        batch_ids = sorted(calendars_by_stage[0])
        if any(set(calendars) != set(batch_ids) for calendars in calendars_by_stage):
            raise TraceContextError("PP stages disagree on in-flight batch IDs")

        batches: list[InflightMicrobatchState] = []
        for microbatch_id in batch_ids:
            first = calendars_by_stage[0][microbatch_id]
            request_tokens = first["request_tokens"]
            allocations = []
            for request_id_text, work in sorted(
                request_tokens.items(), key=lambda item: int(item[0])
            ):
                request_id = int(request_id_text)
                request = request_by_id[request_id]
                new_blocks = (
                    request.reserved_kv_blocks
                    if request.inflight_microbatch_id == microbatch_id
                    else 0
                )
                allocations.append(
                    BatchAllocation(
                        request_id=request_id,
                        prefill_tokens=int(work["prefill_tokens"]),
                        decode_tokens=int(work["decode_tokens"]),
                        new_kv_blocks=new_blocks,
                    )
                )
            batches.append(
                InflightMicrobatchState(
                    microbatch_id=microbatch_id,
                    replica_id=replica_id,
                    raw_action_index=int(first["raw_action_index"]),
                    canonical_action_index=int(first["canonical_action_index"]),
                    allocations=tuple(allocations),
                    stage_ready_times=tuple(
                        float(calendars[microbatch_id]["ready_time"])
                        for calendars in calendars_by_stage
                    ),
                    stage_start_times=tuple(
                        float(calendars[microbatch_id]["start_time"])
                        for calendars in calendars_by_stage
                    ),
                    stage_finish_times=tuple(
                        float(calendars[microbatch_id]["stage_completion_time"])
                        for calendars in calendars_by_stage
                    ),
                )
            )

        return ReplicaState(
            replica_id=replica_id,
            rank_ids=placement.rank_ids,
            rank_kv_capacity_blocks=tuple(
                int(payload["capacity_blocks"]) for payload in rank_payloads
            ),
            rank_kv_committed_blocks=[
                int(payload["consumed_blocks"]) for payload in rank_payloads
            ],
            rank_kv_reserved_blocks=[
                int(payload["reserved_blocks"]) for payload in rank_payloads
            ],
            stage_tail_finish_times=[
                float(payload["tail_finish_time"]) for payload in stage_payloads
            ],
            stage_last_microbatch_ids=[
                NO_ID
                if payload["last_microbatch_id"] is None
                else int(payload["last_microbatch_id"])
                for payload in stage_payloads
            ],
            inflight_microbatches=batches,
        )


def _request_state(payload: dict[str, Any]) -> RequestState:
    return RequestState(
        request_id=int(payload["request_id"]),
        owner_replica_id=int(payload["owner_replica_id"]),
        lifecycle=RequestLifecycle[payload["lifecycle"]],
        arrival_time=float(payload["arrival_time"]),
        prefill_deadline=float(payload["prefill_deadline"]),
        decode_token_slo_sec=float(payload["decode_token_slo_sec"]),
        original_prefill_tokens=int(payload["original_prefill_tokens"]),
        original_decode_tokens=int(payload["original_decode_tokens"]),
        decode_credit_minted=bool(payload["decode_credit_minted"]),
        committed_prefill_tokens=int(payload["committed_prefill_tokens"]),
        reserved_prefill_tokens=int(payload["reserved_prefill_tokens"]),
        committed_decode_tokens=int(payload["committed_decode_tokens"]),
        reserved_decode_tokens=int(payload["reserved_decode_tokens"]),
        committed_kv_blocks=int(payload["committed_kv_blocks"]),
        reserved_kv_blocks=int(payload["reserved_kv_blocks"]),
        inflight_microbatch_id=(
            NO_ID
            if payload["inflight_microbatch_id"] is None
            else int(payload["inflight_microbatch_id"])
        ),
        next_decode_deadline=_optional_time(payload["next_decode_deadline"]),
        prefill_lateness_sec=float(payload["prefill_lateness_sec"]),
        decode_lateness_sec=float(payload["decode_lateness_sec"]),
        violation_recorded=bool(payload["violation_recorded"]),
        terminal_reason=TerminalReason[payload["terminal_reason"]],
        terminal_requested_at=_optional_time(payload["terminal_requested_at"]),
        terminal_time=_optional_time(payload["completion_time"]),
    )


@dataclass(frozen=True, slots=True)
class IterationTrace:
    run_id: int
    nodes: tuple[TraceNode, ...]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TraceContextError(f"missing CSV header: {path}")
        return list(reader)


def load_iteration_trace(iteration_dir: str | Path) -> IterationTrace:
    """Join one iteration's four mini-files and return its ordered path."""

    root = Path(iteration_dir).expanduser().resolve()
    tables: dict[str, dict[NodeKey, dict[str, str]]] = {}
    for file_name in LOG_FILE_NAMES:
        path = root / file_name
        if not path.is_file():
            raise TraceContextError(f"missing iteration log {path}")
        rows = _read_rows(path)
        keyed = {NodeKey.from_row(row): row for row in rows}
        if len(keyed) != len(rows):
            raise TraceContextError(f"duplicate node key in {path}")
        tables[file_name] = keyed

    keys = set(tables[LOG_FILE_NAMES[0]])
    if not keys:
        raise TraceContextError(f"empty iteration trace: {root}")
    for file_name in LOG_FILE_NAMES[1:]:
        if set(tables[file_name]) != keys:
            raise TraceContextError(f"four-file join mismatch in {root}")
    run_ids = {key.run_id for key in keys}
    if len(run_ids) != 1:
        raise TraceContextError(f"iteration directory contains multiple run IDs: {root}")

    nodes = []
    for key in keys:
        rows = [tables[file_name][key] for file_name in LOG_FILE_NAMES]
        for field in (
            "logger_schema_version",
            "state_schema_version",
            "config_manifest_sha256",
            "state_hash",
            "state_time",
        ):
            if len({row[field] for row in rows}) != 1:
                raise TraceContextError(f"common field {field} disagrees for {key}")
        nodes.append(TraceNode(key, *rows))

    nodes.sort(key=lambda node: node.depth)
    return IterationTrace(run_id=next(iter(run_ids)), nodes=tuple(nodes))


def load_iteration_traces(
    trace_root: str | Path,
    *,
    expected_iterations: int,
) -> tuple[IterationTrace, ...]:
    root = Path(trace_root).expanduser().resolve()
    traces = tuple(
        load_iteration_trace(root / f"mcts_iter_{index:06d}")
        for index in range(1, expected_iterations + 1)
    )
    if tuple(trace.run_id for trace in traces) != tuple(
        range(1, expected_iterations + 1)
    ):
        raise TraceContextError("iteration folder and run_id ordering disagree")
    return traces
