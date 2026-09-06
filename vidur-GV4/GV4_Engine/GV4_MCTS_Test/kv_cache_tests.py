"""Independent logical/per-rank KV accounting checks for every logged node."""

from __future__ import annotations

import math

from ..action_resolver import ControllerTransitionKind
from .trace_context import IterationTrace, TraceNode
from .validation import ValidationContext, ValidationReport


GROUP = "kv_cache"


def _batch_is_present(node: TraceNode, replica_id: int, microbatch_id: int) -> bool:
    stage = node.stage(replica_id, 0)
    return any(
        int(batch["microbatch_id"]) == microbatch_id
        for batch in stage["inflight_batch_calendars"]
    )


def _check_node_kv(
    node: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    config = context.config
    block_size = config.kv_cache.block_size_tokens
    requests = node.requests_by_id().values()
    logical_capacity = logical_committed = logical_reserved = resident_total = 0

    for replica_id, placement in enumerate(config.topology.replica_placements):
        replica_requests = [
            request
            for request in requests
            if int(request["owner_replica_id"]) == replica_id
            and request["lifecycle"] not in {"COMPLETED", "STOPPED", "DROPPED"}
        ]
        expected_committed = sum(
            int(request["committed_kv_blocks"]) for request in replica_requests
        )
        expected_reserved = sum(
            int(request["reserved_kv_blocks"]) for request in replica_requests
        )
        expected_resident = sum(int(request["resident_tokens"]) for request in replica_requests)
        resident_total += expected_resident

        rank_payloads = [node.rank(rank_id) for rank_id in placement.rank_ids]
        capacities = [int(payload["capacity_blocks"]) for payload in rank_payloads]
        logical_capacity += min(capacities)
        logical_committed += expected_committed
        logical_reserved += expected_reserved

        for payload, rank_id in zip(rank_payloads, placement.rank_ids):
            capacity = int(payload["capacity_blocks"])
            available = capacity - expected_committed - expected_reserved
            report.check(
                GROUP,
                "rank capacity matches engine manifest",
                capacity == config.rank_kv_block_capacities()[rank_id],
                node,
            )
            report.check(
                GROUP,
                "rank committed mirror matches requests",
                int(payload["consumed_blocks"]) == expected_committed,
                node,
            )
            report.check(
                GROUP,
                "rank reserved mirror matches requests",
                int(payload["reserved_blocks"]) == expected_reserved,
                node,
            )
            report.check(
                GROUP,
                "rank free blocks are exact",
                int(payload["available_blocks"]) == available,
                node,
            )
            report.check(
                GROUP,
                "rank resident-token view matches replica",
                int(payload["resident_tokens"]) == expected_resident,
                node,
            )
            report.check(
                GROUP,
                "rank available token slots include partial blocks",
                int(payload["available_token_slots"])
                == capacity * block_size - expected_resident,
                node,
            )

        report.check(
            GROUP,
            "all TP/PP ranks mirror logical ownership",
            len({int(payload["consumed_blocks"]) for payload in rank_payloads}) == 1
            and len({int(payload["reserved_blocks"]) for payload in rank_payloads}) == 1,
            node,
        )

    total_available = logical_capacity - logical_committed - logical_reserved
    expected_totals = {
        "total_capacity_blocks": logical_capacity,
        "total_available_blocks": total_available,
        "total_consumed_blocks": logical_committed,
        "total_reserved_blocks": logical_reserved,
        "total_occupied_blocks": logical_committed + logical_reserved,
        "total_tokens_in_memory": resident_total,
        "total_available_tokens": logical_capacity * block_size - resident_total,
        "total_free_block_token_slots": total_available * block_size,
    }
    for field, expected in expected_totals.items():
        report.check(
            GROUP,
            f"logical aggregate {field}",
            int(node.kv_cache[field]) == expected,
            node,
            f"logged={node.kv_cache[field]}, expected={expected}",
        )

    for request in requests:
        if request["lifecycle"] in {"COMPLETED", "STOPPED", "DROPPED"}:
            continue
        required = math.ceil(int(request["resident_tokens"]) / block_size)
        owned = int(request["committed_kv_blocks"]) + int(request["reserved_kv_blocks"])
        report.check(
            GROUP,
            "request owns exactly its rounded KV demand",
            owned == required,
            node,
        )


def _check_controller_kv_delta(
    parent: TraceNode,
    child: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    action = child.resolved_controller_action()
    if action is None:
        return
    net_delta = action.reserved_kv_blocks - action.released_kv_blocks
    expected_rank_delta = tuple(
        (rank_id, net_delta)
        for rank_id in context.config.topology.replica_placements[
            action.replica_id
        ].rank_ids
    )
    report.check(
        GROUP,
        "resolved rank KV delta is TP/PP mirrored",
        action.rank_kv_delta == expected_rank_delta,
        child,
    )

    should_compare_immediate = (
        action.transition_kind
        in {
            ControllerTransitionKind.EVICT_ONLY,
            ControllerTransitionKind.PREEMPT_ONLY,
            ControllerTransitionKind.EVICT_AND_PREEMPT,
        }
        and abs(child.now - parent.now) <= context.epsilon
    )
    if action.transition_kind == ControllerTransitionKind.BATCH:
        microbatch_id = int(parent.mcts["next_microbatch_id"])
        should_compare_immediate = _batch_is_present(
            child, action.replica_id, microbatch_id
        )
    if not should_compare_immediate:
        report.cover("kv_deltas_obscured_by_time_advance")
        return

    placement = context.config.topology.replica_placements[action.replica_id]
    for rank_id in placement.rank_ids:
        before = parent.rank(rank_id)
        after = child.rank(rank_id)
        before_occupied = int(before["occupied_blocks"])
        after_occupied = int(after["occupied_blocks"])
        report.check(
            GROUP,
            "immediate rank occupancy follows action delta",
            after_occupied - before_occupied == net_delta,
            child,
        )


def run_kv_cache_tests(
    traces: tuple[IterationTrace, ...],
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    for trace in traces:
        for node in trace.nodes:
            _check_node_kv(node, context, report)
        for parent, child in zip(trace.nodes, trace.nodes[1:]):
            _check_controller_kv_delta(parent, child, context, report)
