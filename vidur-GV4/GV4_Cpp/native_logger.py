"""Serialize selected native MCTS paths with the existing GV4 CSV contract.

The engine and search stay in C++. Only immutable path snapshots cross the
binding after backup, which keeps diagnostic serialization out of hot native
transition code and avoids maintaining a second CSV/JSON implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from GV4_Engine.action_resolver import (
    CanonicalAdversaryAction,
    CanonicalControllerAction,
    ControllerTransitionKind,
    ResolvedAdversaryAction,
    ResolvedControllerAction,
)
from GV4_Engine.logger import GV4NodeCSVLogger
from GV4_Engine.state import (
    BatchAllocation,
    GV4State,
    InflightMicrobatchState,
    LaunchRecord,
    ObjectiveState,
    Player,
    ReplicaState,
    RequestLifecycle,
    RequestState,
    TerminalReason,
)

from . import gv4_native as native


def _allocation(item: Any) -> BatchAllocation:
    return BatchAllocation(
        request_id=int(item.request_id),
        prefill_tokens=int(item.prefill_tokens),
        decode_tokens=int(item.decode_tokens),
        new_kv_blocks=int(item.new_kv_blocks),
    )


def _request(item: Any) -> RequestState:
    return RequestState(
        request_id=int(item.request_id),
        owner_replica_id=int(item.owner_replica_id),
        lifecycle=RequestLifecycle[item.lifecycle.name],
        arrival_time=float(item.arrival_time),
        prefill_deadline=float(item.prefill_deadline),
        decode_token_slo_sec=float(item.decode_token_slo_sec),
        original_prefill_tokens=int(item.original_prefill_tokens),
        original_decode_tokens=int(item.original_decode_tokens),
        decode_credit_minted=bool(item.decode_credit_minted),
        committed_prefill_tokens=int(item.committed_prefill_tokens),
        reserved_prefill_tokens=int(item.reserved_prefill_tokens),
        committed_decode_tokens=int(item.committed_decode_tokens),
        reserved_decode_tokens=int(item.reserved_decode_tokens),
        committed_kv_blocks=int(item.committed_kv_blocks),
        reserved_kv_blocks=int(item.reserved_kv_blocks),
        inflight_microbatch_id=int(item.inflight_microbatch_id),
        next_decode_deadline=float(item.next_decode_deadline),
        prefill_lateness_sec=float(item.prefill_lateness_sec),
        decode_lateness_sec=float(item.decode_lateness_sec),
        violation_recorded=bool(item.violation_recorded),
        terminal_reason=TerminalReason[item.terminal_reason.name],
        terminal_requested_at=float(item.terminal_requested_at),
        terminal_time=float(item.terminal_time),
    )


def _microbatch(item: Any) -> InflightMicrobatchState:
    return InflightMicrobatchState(
        microbatch_id=int(item.microbatch_id),
        replica_id=int(item.replica_id),
        raw_action_index=int(item.raw_action_index),
        canonical_action_index=int(item.canonical_action_index),
        allocations=tuple(_allocation(value) for value in item.allocations),
        stage_ready_times=tuple(float(value) for value in item.stage_ready_times),
        stage_start_times=tuple(float(value) for value in item.stage_start_times),
        stage_finish_times=tuple(float(value) for value in item.stage_finish_times),
        completion_applied=bool(item.completion_applied),
    )


def state_from_native(state: native.State) -> GV4State:
    """Copy one native snapshot into the authoritative Python log schema."""

    replica = state.replica
    replica_state = ReplicaState(
        replica_id=int(replica.replica_id),
        rank_ids=tuple(int(value) for value in replica.rank_ids),
        rank_kv_capacity_blocks=tuple(
            int(value) for value in replica.rank_kv_capacity_blocks
        ),
        rank_kv_committed_blocks=[
            int(value) for value in replica.rank_kv_committed_blocks
        ],
        rank_kv_reserved_blocks=[
            int(value) for value in replica.rank_kv_reserved_blocks
        ],
        stage_tail_finish_times=[
            float(value) for value in replica.stage_tail_finish_times
        ],
        stage_last_microbatch_ids=[
            int(value) for value in replica.stage_last_microbatch_ids
        ],
        inflight_microbatches=[
            _microbatch(value) for value in replica.inflight_microbatches
        ],
    )
    objective = state.objective
    return GV4State(
        state_schema_version=str(state.state_schema_version),
        config_manifest_sha256=str(state.config_manifest_sha256),
        now=float(state.now),
        next_player=Player[state.next_player.name],
        next_adversary_tick=float(state.next_adversary_tick),
        next_request_id=int(state.next_request_id),
        next_microbatch_id=int(state.next_microbatch_id),
        tie_break_counter=int(state.tie_break_counter),
        rng_seed=int(state.rng_seed),
        rng_counter=int(state.rng_counter),
        launch_history=[
            LaunchRecord(
                launch_time=float(item.launch_time),
                request_count=int(item.request_count),
                prefill_tokens=int(item.prefill_tokens),
            )
            for item in state.launch_history
        ],
        decode_credits_available=int(state.decode_credits_available),
        decode_credits_reserved=int(state.decode_credits_reserved),
        decode_credits_minted_total=int(state.decode_credits_minted_total),
        decode_tokens_committed_total=int(state.decode_tokens_committed_total),
        requests=[_request(item) for item in state.requests],
        replicas=[replica_state],
        objective=ObjectiveState(
            requests_generated=int(objective.requests_generated),
            requests_completed=int(objective.requests_completed),
            requests_stopped=int(objective.requests_stopped),
            requests_dropped=int(objective.requests_dropped),
            slo_violations=int(objective.slo_violations),
            prefill_lateness_sec=float(objective.prefill_lateness_sec),
            decode_lateness_sec=float(objective.decode_lateness_sec),
            terminal_cost=float(objective.terminal_cost),
            total_cost=float(objective.total_cost),
        ),
    )


def _canonical_action(action: Any) -> CanonicalControllerAction | CanonicalAdversaryAction:
    if isinstance(action, native.CanonicalControllerAction):
        resolved = action.action
        return CanonicalControllerAction(
            canonical_action_index=int(action.canonical_action_index),
            action=ResolvedControllerAction(
                raw_action_index=int(resolved.raw_action_index),
                replica_id=int(resolved.replica_id),
                eviction_rule=str(resolved.eviction_rule),
                prefill_budget=int(resolved.prefill_budget),
                ordering_heuristic=str(resolved.ordering_heuristic),
                transition_kind=ControllerTransitionKind[
                    resolved.transition_kind.name
                ],
                evicted_request_ids=tuple(
                    int(value) for value in resolved.evicted_request_ids
                ),
                allocations=tuple(_allocation(value) for value in resolved.allocations),
                released_kv_blocks=int(resolved.released_kv_blocks),
                reserved_kv_blocks=int(resolved.reserved_kv_blocks),
                rank_kv_delta=tuple(
                    (int(rank_id), int(delta))
                    for rank_id, delta in resolved.rank_kv_delta
                ),
            ),
            equivalent_raw_indices=tuple(
                int(value) for value in action.equivalent_raw_indices
            ),
        )

    resolved = action.action
    return CanonicalAdversaryAction(
        canonical_action_index=int(action.canonical_action_index),
        action=ResolvedAdversaryAction(
            raw_action_index=int(resolved.raw_action_index),
            launch_count=int(resolved.launch_count),
            prefill_tokens=(
                int(resolved.prefill_tokens) if resolved.launch_count else None
            ),
            stop_rule=str(resolved.stop_rule),
            stop_request_ids=tuple(int(value) for value in resolved.stop_request_ids),
        ),
        equivalent_raw_indices=tuple(
            int(value) for value in action.equivalent_raw_indices
        ),
    )


@dataclass(slots=True)
class _NodeView:
    """Minimum Python node protocol consumed by ``GV4NodeCSVLogger``."""

    player: str
    node_id: int
    depth: int
    cached_sim_snapshot: GV4State
    parent_action: CanonicalControllerAction | CanonicalAdversaryAction | None
    parent_action_index: int
    reward: float
    edge_discount: float
    visits: int
    value_sum: float
    valid_mask: list[bool]
    canonical_to_action_aliases: dict[int, list[int]]
    parent: _NodeView | None = None
    children: dict[int, _NodeView] = field(default_factory=dict)


def _node_view(node: native.MCTSNode) -> _NodeView:
    aliases = {
        int(entry.representative_raw_index): [
            int(value) for value in entry.equivalent_raw_indices
        ]
        for entry in node.actions
    }
    parent_action = (
        None if node.parent_action is None else _canonical_action(node.parent_action)
    )
    return _NodeView(
        player=node.player.name.lower(),
        node_id=int(node.node_id),
        depth=int(node.depth),
        cached_sim_snapshot=state_from_native(node.state),
        parent_action=parent_action,
        parent_action_index=int(node.parent_action_index),
        reward=float(node.reward),
        edge_discount=float(node.edge_discount),
        visits=int(node.visits),
        value_sum=float(node.value_sum),
        valid_mask=[bool(value) for value in node.valid_mask],
        canonical_to_action_aliases=aliases,
    )


class NativeIterationPathLogger:
    """Adapt the post-backup native observer callback to four CSV rows per node."""

    __slots__ = ("logger", "game_id", "root_id", "_expected_iteration")

    def __init__(
        self,
        logger: GV4NodeCSVLogger,
        *,
        game_id: int = 0,
        root_id: int = 0,
    ) -> None:
        self.logger = logger
        self.game_id = int(game_id)
        self.root_id = int(root_id)
        self._expected_iteration = 1

    def __call__(
        self,
        *,
        iteration_index: int,
        path: Sequence[native.MCTSNode],
        root: native.MCTSNode,
    ) -> None:
        if iteration_index != self._expected_iteration:
            raise RuntimeError(
                f"expected native iteration {self._expected_iteration}, "
                f"got {iteration_index}"
            )
        if not path or int(path[0].node_id) != int(root.node_id):
            raise RuntimeError("native MCTS observer returned a broken root path")

        views = [_node_view(node) for node in path]
        for parent, child in zip(views, views[1:]):
            child.parent = parent
            parent.children[child.parent_action_index] = child
        self.logger.log_path(
            run_id=iteration_index,
            game_id=self.game_id,
            root_id=self.root_id,
            path=views,
        )
        self._expected_iteration += 1

    @property
    def iterations_logged(self) -> int:
        return self._expected_iteration - 1


__all__ = ["NativeIterationPathLogger", "state_from_native"]
