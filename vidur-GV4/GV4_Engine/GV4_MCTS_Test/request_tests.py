"""Adversary launch/stop, controller batch/eviction, and lifecycle checks."""

from __future__ import annotations

import math

from ..action_resolver import ControllerTransitionKind
from .trace_context import IterationTrace, TraceNode
from .validation import ValidationContext, ValidationReport


GROUP = "requests"
TERMINAL = {"COMPLETED", "STOPPED", "DROPPED"}


def _expected_evictions(parent: TraceNode, rule: str, epsilon: float) -> tuple[int, ...]:
    requests = parent.requests_by_id().values()
    prefills = [
        request
        for request in requests
        if request["lifecycle"] == "WAITING_PREFILL"
        and int(request["committed_kv_blocks"]) > 0
    ]
    decodes = [
        request
        for request in requests
        if request["lifecycle"] == "WAITING_DECODE"
        and int(request["committed_kv_blocks"]) > 0
    ]
    if rule == "evict_none":
        return ()
    if rule == "evict_largest_prefill" and prefills:
        chosen = max(
            prefills,
            key=lambda item: (int(item["remaining_prefill_tokens"]), -int(item["request_id"])),
        )
        return (int(chosen["request_id"]),)
    if rule == "evict_earliest_prefill_deadline" and prefills:
        chosen = min(
            prefills,
            key=lambda item: (float(item["prefill_deadline"]), int(item["request_id"])),
        )
        return (int(chosen["request_id"]),)

    def prefill_lateness(request: dict) -> float:
        return max(
            float(request["prefill_lateness_sec"]),
            parent.now - float(request["prefill_deadline"]),
            0.0,
        )

    def decode_lateness(request: dict) -> float:
        return max(
            0.0,
            float(request["prefill_lateness_sec"])
            + float(request["decode_lateness_sec"]),
        )

    if rule == "evict_prefill_missed_deadline":
        return tuple(
            int(item["request_id"])
            for item in prefills
            if prefill_lateness(item) > epsilon
        )
    if rule == "evict_prefill_lateness_over_0p5":
        return tuple(
            int(item["request_id"])
            for item in prefills
            if prefill_lateness(item) > 0.5
        )
    if rule == "evict_longest_decode" and decodes:
        chosen = max(
            decodes,
            key=lambda item: (int(item["committed_decode_tokens"]), -int(item["request_id"])),
        )
        return (int(chosen["request_id"]),)
    if rule == "evict_decode_lateness_over_0p5":
        return tuple(
            int(item["request_id"])
            for item in decodes
            if decode_lateness(item) > 0.5
        )
    if rule == "evict_prefill_highest_lateness" and prefills:
        chosen = max(
            prefills,
            key=lambda item: (prefill_lateness(item), -int(item["request_id"])),
        )
        return (
            (int(chosen["request_id"]),)
            if prefill_lateness(chosen) > epsilon
            else ()
        )
    if rule == "evict_decode_highest_lateness" and decodes:
        chosen = max(
            decodes,
            key=lambda item: (decode_lateness(item), -int(item["request_id"])),
        )
        return (
            (int(chosen["request_id"]),)
            if decode_lateness(chosen) > epsilon
            else ()
        )
    return ()


def _expected_stops(parent: TraceNode, rule: str) -> tuple[int, ...]:
    decodes = [
        request
        for request in parent.requests_by_id().values()
        if request["current_phase"] == "decode"
        and request["lifecycle"]
        in {
            "WAITING_DECODE",
            "INFLIGHT_DECODE",
            "INFLIGHT_RECOMPUTE",
            "PREEMPT_PENDING",
        }
    ]
    if rule == "stop_none" or not decodes:
        return ()
    if rule == "stop_longest_decode":
        chosen = max(
            decodes,
            key=lambda item: (int(item["committed_decode_tokens"]), -int(item["request_id"])),
        )
        return (int(chosen["request_id"]),)
    if rule == "stop_shortest_decode":
        chosen = min(
            decodes,
            key=lambda item: (int(item["committed_decode_tokens"]), int(item["request_id"])),
        )
        return (int(chosen["request_id"]),)
    threshold = 512 if rule == "stop_all_decodes_over_512" else 216
    return tuple(
        int(item["request_id"])
        for item in decodes
        if int(item["committed_decode_tokens"]) > threshold
    )


def _check_request_snapshot(
    node: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    groups = node.request_groups()
    expected_groups = {
        "completed": {"COMPLETED"},
        "evicted": {"DROPPED"},
        "stopped": {"STOPPED"},
    }
    for group, lifecycles in expected_groups.items():
        report.check(
            GROUP,
            f"{group} CSV grouping",
            all(request["lifecycle"] in lifecycles for request in groups[group]),
            node,
        )
    report.check(
        GROUP,
        "in-progress CSV grouping",
        all(request["lifecycle"] not in TERMINAL for request in groups["in_progress"]),
        node,
    )

    requests = node.requests_by_id()
    report.check(
        GROUP,
        "request IDs are append-only and contiguous",
        sorted(requests) == list(range(int(node.mcts["next_request_id"]))),
        node,
    )
    for request in requests.values():
        committed_prefill = int(request["committed_prefill_tokens"])
        reserved_prefill = int(request["reserved_prefill_tokens"])
        committed_decode = int(request["committed_decode_tokens"])
        reserved_decode = int(request["reserved_decode_tokens"])
        reserved_recompute = int(request["reserved_recompute_tokens"])
        original_prefill = int(request["original_prefill_tokens"])
        original_decode = int(request["original_decode_tokens"])
        report.check(
            GROUP,
            "prefill work does not exceed request length",
            committed_prefill + reserved_prefill <= original_prefill,
            node,
        )
        report.check(
            GROUP,
            "decode work does not exceed request length",
            committed_decode + reserved_decode <= original_decode,
            node,
        )
        report.check(
            GROUP,
            "request uses configured decode cap",
            original_decode <= context.config.request.max_decode_tokens_per_request,
            node,
        )
        report.check(
            GROUP,
            "decode starts only after prefill completion",
            not (committed_decode or reserved_decode)
            or committed_prefill == original_prefill,
            node,
        )
        report.check(
            GROUP,
            "one decode token at most per request batch",
            reserved_decode in {0, 1},
            node,
        )
        report.check(
            GROUP,
            "new work waits for KV reconstruction",
            not (reserved_prefill or reserved_decode)
            or (
                int(request["remaining_recompute_tokens"]) == 0
                and reserved_recompute == 0
            ),
            node,
        )
        expected_phase = (
            "prefill" if int(request["remaining_prefill_tokens"]) or reserved_prefill else "decode"
        )
        report.check(
            GROUP,
            "logged request phase follows progress",
            request["current_phase"] == expected_phase,
            node,
        )

    minted = sum(bool(request["decode_credit_minted"]) for request in requests.values())
    committed = sum(int(request["committed_decode_tokens"]) for request in requests.values())
    available = int(node.mcts["decode_credits_available"])
    reserved = int(node.mcts["decode_credits_reserved"])
    report.check(
        GROUP,
        "pooled decode-credit conservation",
        available + reserved + committed
        == minted * context.config.credits.decode_credit_mint_per_prefill_completion,
        node,
    )
    if available == 0:
        report.check(
            GROUP,
            "credit exhaustion stops every decode",
            all(
                request["current_phase"] != "decode"
                or request["lifecycle"]
                not in {
                    "WAITING_DECODE",
                    "INFLIGHT_DECODE",
                    "INFLIGHT_RECOMPUTE",
                    "PREEMPT_PENDING",
                }
                for request in requests.values()
            ),
            node,
        )


def _check_adversary_edge(
    parent: TraceNode,
    child: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    action = child.resolved_adversary_action()
    if action is None:
        return
    components = context.config.adversary_actions.raw_action_components(
        action.raw_action_index
    )
    report.check(
        GROUP,
        "adversary raw index maps to payload",
        components == (action.launch_count, action.prefill_tokens, action.stop_rule),
        child,
    )
    expected_stops = _expected_stops(parent, action.stop_rule)
    report.check(
        GROUP,
        "adversary stop rule selects expected decodes",
        action.stop_request_ids == expected_stops,
        child,
    )
    if action.stop_request_ids:
        report.cover("adversary_stops", len(action.stop_request_ids))

    before = parent.requests_by_id()
    after = child.requests_by_id()
    new_ids = sorted(set(after) - set(before))
    expected_ids = list(
        range(int(parent.mcts["next_request_id"]), int(parent.mcts["next_request_id"]) + action.launch_count)
    )
    report.check(GROUP, "adversary assigns sequential request IDs", new_ids == expected_ids, child)
    if action.launch_count:
        report.cover("launched_requests", action.launch_count)
        assert action.prefill_tokens is not None
        estimate = float(context.timing_provider.estimate_prefill_time(action.prefill_tokens))
        deadline = round(
            parent.now + context.config.slo.prefill_slowdown_factor * estimate,
            context.config.timing.time_round_digits,
        )
        for request_id in new_ids:
            request = after[request_id]
            report.check(
                GROUP,
                "launch uses allowed prefill template",
                int(request["original_prefill_tokens"])
                in context.config.adversary_actions.prefill_token_templates,
                child,
            )
            report.check(
                GROUP,
                "launch payload and request size agree",
                int(request["original_prefill_tokens"]) == action.prefill_tokens,
                child,
            )
            report.close(GROUP, "request arrival time", float(request["arrival_time"]), parent.now, child)
            report.close(GROUP, "prefill deadline", float(request["prefill_deadline"]), deadline, child)
            report.check(
                GROUP,
                "new request starts in waiting prefill",
                request["lifecycle"] == "WAITING_PREFILL",
                child,
            )

    for request_id in action.stop_request_ids:
        before_request = before[request_id]
        after_request = after[request_id]
        expected_lifecycle = (
            "STOP_PENDING"
            if int(before_request["inflight_microbatch_id"]) >= 0
            else "STOPPED"
        )
        report.check(
            GROUP,
            "adversary stop updates lifecycle",
            after_request["lifecycle"] == expected_lifecycle,
            child,
        )


def _check_controller_edge(
    parent: TraceNode,
    child: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    action = child.resolved_controller_action()
    if action is None:
        return
    report.cover("controller_actions")
    components = context.config.controller_actions.raw_action_components(
        action.raw_action_index
    )
    report.check(
        GROUP,
        "controller raw index maps to payload",
        components
        == (
            action.preemption_rule,
            action.eviction_rule,
            action.prefill_budget,
            action.ordering_heuristic,
        ),
        child,
    )
    expected_evictions = _expected_evictions(parent, action.eviction_rule, context.epsilon)
    report.check(
        GROUP,
        "eviction policy selects expected requests",
        action.evicted_request_ids == expected_evictions,
        child,
    )
    if action.evicted_request_ids:
        report.cover("controller_evictions", len(action.evicted_request_ids))
    if action.preempted_request_ids:
        report.cover("controller_preemptions", len(action.preempted_request_ids))
    if action.pending_preemption_request_ids:
        report.cover(
            "controller_inflight_preemptions",
            len(action.pending_preemption_request_ids),
        )
    report.check(
        GROUP,
        "a preemption rule selects at most one request",
        len(action.preempted_request_ids) <= 1,
        child,
    )
    report.check(
        GROUP,
        "eviction and preemption targets are disjoint",
        not set(action.evicted_request_ids) & set(action.preempted_request_ids),
        child,
    )

    allocations = action.allocations
    recompute_allocations = sum(bool(item.recompute_tokens) for item in allocations)
    if recompute_allocations:
        report.cover("controller_recomputations", recompute_allocations)
    request_ids = [allocation.request_id for allocation in allocations]
    report.check(
        GROUP,
        "batch request IDs sorted and unique",
        request_ids == sorted(set(request_ids)),
        child,
    )
    report.check(
        GROUP,
        "batch token cap",
        sum(item.total_tokens for item in allocations)
        <= context.config.scheduler.max_batch_tokens,
        child,
    )
    report.check(
        GROUP,
        "batch sequence cap",
        len(allocations) <= context.config.scheduler.max_sequences,
        child,
    )
    report.check(
        GROUP,
        "prefill budget is an upper bound",
        sum(item.prefill_tokens + item.recompute_tokens for item in allocations)
        <= action.prefill_budget,
        child,
    )
    if action.prefill_budget == 0:
        report.check(
            GROUP,
            "zero prefill budget schedules no prefill-class work",
            all(
                item.prefill_tokens == 0 and item.recompute_tokens == 0
                for item in allocations
            ),
            child,
        )

    parent_requests = parent.requests_by_id()
    expected_pending = tuple(
        request_id
        for request_id in action.preempted_request_ids
        if int(parent_requests[request_id]["inflight_microbatch_id"]) >= 0
    )
    report.check(
        GROUP,
        "pending preemption IDs are exactly the in-flight victims",
        action.pending_preemption_request_ids == expected_pending,
        child,
    )
    for allocation in allocations:
        request = parent_requests.get(allocation.request_id)
        report.check(GROUP, "allocation targets an existing request", request is not None, child)
        if request is None:
            continue
        if allocation.recompute_tokens:
            expected_lifecycles = {"WAITING_PREFILL", "WAITING_DECODE"}
            phase_is_legal = int(request["remaining_recompute_tokens"]) > 0
        elif allocation.prefill_tokens:
            expected_lifecycles = {"WAITING_PREFILL"}
            phase_is_legal = int(request["remaining_recompute_tokens"]) == 0
        else:
            expected_lifecycles = {"WAITING_DECODE"}
            phase_is_legal = int(request["remaining_recompute_tokens"]) == 0
        report.check(
            GROUP,
            "allocation targets the matching waiting phase",
            request["lifecycle"] in expected_lifecycles and phase_is_legal,
            child,
        )
        resident_after = int(request["resident_tokens"]) + allocation.total_tokens
        needed_after = math.ceil(
            resident_after / context.config.kv_cache.block_size_tokens
        )
        owned = int(request["committed_kv_blocks"]) + int(request["reserved_kv_blocks"])
        report.check(
            GROUP,
            "allocation records exact new KV blocks",
            allocation.new_kv_blocks == max(0, needed_after - owned),
            child,
        )

    expected_kind = (
        ControllerTransitionKind.BATCH
        if allocations
        else ControllerTransitionKind.EVICT_AND_PREEMPT
        if action.evicted_request_ids and action.preempted_request_ids
        else ControllerTransitionKind.EVICT_ONLY
        if action.evicted_request_ids
        else ControllerTransitionKind.PREEMPT_ONLY
        if action.preempted_request_ids
        else ControllerTransitionKind.WAIT
    )
    report.check(GROUP, "controller transition kind matches effects", action.transition_kind == expected_kind, child)
    report.check(
        GROUP,
        "reserved block total matches allocations",
        action.reserved_kv_blocks == sum(item.new_kv_blocks for item in allocations),
        child,
    )
    report.check(
        GROUP,
        "released block total matches immediate memory actions",
        action.released_kv_blocks
        == sum(
            int(parent_requests[request_id]["committed_kv_blocks"])
            for request_id in action.evicted_request_ids
        )
        + action.preempted_kv_blocks,
        child,
    )
    immediate_preemptions = tuple(
        request_id
        for request_id in action.preempted_request_ids
        if request_id not in action.pending_preemption_request_ids
    )
    report.check(
        GROUP,
        "preempted block total excludes in-flight victims",
        action.preempted_kv_blocks
        == sum(
            int(parent_requests[request_id]["committed_kv_blocks"])
            for request_id in immediate_preemptions
        ),
        child,
    )

    child_requests = child.requests_by_id()
    for request_id in action.evicted_request_ids:
        report.check(
            GROUP,
            "evicted request becomes dropped",
            child_requests[request_id]["lifecycle"] == "DROPPED",
            child,
        )
    if abs(child.now - parent.now) <= context.epsilon:
        for request_id in immediate_preemptions:
            request = child_requests[request_id]
            report.check(
                GROUP,
                "immediate preemption releases physical KV",
                int(request["committed_kv_blocks"]) == 0
                and int(request["kv_computed_tokens"]) == 0
                and int(request["remaining_recompute_tokens"])
                == int(request["logical_context_tokens"]),
                child,
            )
        for request_id in action.pending_preemption_request_ids:
            report.check(
                GROUP,
                "in-flight preemption is marked pending",
                child_requests[request_id]["lifecycle"] == "PREEMPT_PENDING",
                child,
            )
    if action.transition_kind == ControllerTransitionKind.BATCH:
        report.cover("controller_batches")
        microbatch_id = int(parent.mcts["next_microbatch_id"])
        for allocation in allocations:
            request = child_requests[allocation.request_id]
            if request["inflight_microbatch_id"] == microbatch_id:
                report.check(
                    GROUP,
                    "in-flight request reservations match batch",
                    int(request["reserved_prefill_tokens"]) == allocation.prefill_tokens
                    and int(request["reserved_decode_tokens"]) == allocation.decode_tokens
                    and int(request["reserved_recompute_tokens"])
                    == allocation.recompute_tokens
                    and int(request["reserved_kv_blocks"]) == allocation.new_kv_blocks,
                    child,
                )


def run_request_tests(
    traces: tuple[IterationTrace, ...],
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    for trace in traces:
        for node in trace.nodes:
            _check_request_snapshot(node, context, report)
        for parent, child in zip(trace.nodes, trace.nodes[1:]):
            _check_adversary_edge(parent, child, context, report)
            _check_controller_edge(parent, child, context, report)
