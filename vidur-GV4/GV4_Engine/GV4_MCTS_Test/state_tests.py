"""Tree, turn, adversary-clock, objective, and reconstructed-state checks."""

from __future__ import annotations

import json

from ..state import RequestLifecycle
from .trace_context import IterationTrace, TraceNode
from .validation import ValidationContext, ValidationReport


GROUP = "state"


def _all_requests(node: TraceNode) -> dict[int, dict]:
    return node.requests_by_id()


def _terminal_ids(node: TraceNode) -> set[int]:
    terminal = {"COMPLETED", "STOPPED", "DROPPED"}
    return {
        request_id
        for request_id, request in _all_requests(node).items()
        if request["lifecycle"] in terminal
    }


def _check_objective(
    node: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    config = context.config
    requests = _all_requests(node).values()
    completed = stopped = dropped = violations = 0
    prefill_lateness = decode_lateness = terminal_cost = total_cost = 0.0
    for request in requests:
        lifecycle = request["lifecycle"]
        completed += lifecycle == "COMPLETED"
        stopped += lifecycle == "STOPPED"
        dropped += lifecycle == "DROPPED"
        if lifecycle in {"DROP_PENDING", "DROPPED"}:
            terminal_cost += config.cost.terminal_drop_cost
            total_cost += config.cost.terminal_drop_cost
            continue
        prefill_lateness += float(request["prefill_lateness_sec"])
        decode_lateness += float(request["decode_lateness_sec"])
        if request["violation_recorded"]:
            violations += 1
            lateness = float(request["prefill_lateness_sec"]) + float(
                request["decode_lateness_sec"]
            )
            total_cost += config.cost.violation_base_cost + min(
                lateness, config.cost.lateness_cap_sec
            )

    expected = {
        "requests_generated": len(_all_requests(node)),
        "requests_completed": completed,
        "requests_stopped": stopped,
        "requests_dropped": dropped,
        "slo_violations": violations,
    }
    for field, value in expected.items():
        report.check(
            GROUP,
            f"objective {field}",
            int(node.mcts[field]) == value,
            node,
            f"logged={node.mcts[field]}, expected={value}",
        )
    report.close(
        GROUP,
        "prefill lateness aggregate",
        float(node.mcts["prefill_lateness_sec"]),
        prefill_lateness,
        node,
    )
    report.close(
        GROUP,
        "decode lateness aggregate",
        float(node.mcts["decode_lateness_sec"]),
        decode_lateness,
        node,
    )
    report.close(
        GROUP,
        "terminal cost aggregate",
        float(node.mcts["terminal_cost"]),
        terminal_cost,
        node,
    )
    report.close(
        GROUP,
        "complete objective cost",
        float(node.mcts["objective_total_cost"]),
        total_cost,
        node,
    )


def _check_action_mask(
    node: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    mask = json.loads(node.mcts["valid_mask_json"])
    canonical = json.loads(node.mcts["canonical_action_indices_json"])
    expected_width = (
        context.config.adversary_actions.raw_action_count
        if node.mcts["next_player"] == "adversary"
        else context.config.controller_actions.raw_action_count
    )
    # The newly expanded leaf is observed before its own action space is built.
    report.check(
        GROUP,
        "fixed action-mask width",
        len(mask) in {0, expected_width},
        node,
    )
    report.check(GROUP, "unexpanded leaf has no canonical actions", bool(mask) or not canonical, node)
    report.check(
        GROUP,
        "canonical indices sorted and unique",
        canonical == sorted(set(canonical)),
        node,
    )
    report.check(
        GROUP,
        "canonical indices are valid",
        all(0 <= index < len(mask) and mask[index] for index in canonical),
        node,
    )


def _check_adversary_clock(
    parent: TraceNode,
    child: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    action = child.resolved_adversary_action()
    if action is None:
        return
    report.cover("adversary_actions")
    epsilon = context.epsilon
    tick = context.config.timing.adversary_tick_sec
    parent_tick = float(parent.mcts["next_adversary_tick"])
    child_tick = float(child.mcts["next_adversary_tick"])
    if parent.now + epsilon < parent_tick:
        report.cover("forced_adversary_noops")
        report.check(
            GROUP,
            "pre-tick adversary action is forced no-op",
            action.raw_action_index == 0
            and action.launch_count == 0
            and not action.stop_request_ids,
            child,
        )
        report.close(GROUP, "forced no-op preserves tick", child_tick, parent_tick, child)
    else:
        report.cover("processed_adversary_ticks")
        report.close(GROUP, "adversary acts at exposed tick", parent.now, parent_tick, child)
        report.close(
            GROUP,
            "one adversary tick per action",
            child_tick,
            parent.now + tick,
            child,
        )
        nearest_grid = round(parent.now / tick) * tick
        report.close(GROUP, "adversary tick grid", parent.now, nearest_grid, child)


def run_state_tests(
    traces: tuple[IterationTrace, ...],
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    """Validate one exact root-to-leaf chain for every MCTS simulation."""

    for trace in traces:
        nodes = trace.nodes
        root = nodes[0]
        report.check(GROUP, "trace run_id matches folder", root.key.run_id == trace.run_id, root)
        report.check(GROUP, "path starts at depth zero", root.depth == 0, root)
        report.check(GROUP, "root has no parent", root.mcts["parent_node_id"] == "", root)
        report.check(GROUP, "root has no incoming action", root.incoming_action is None, root)
        report.check(
            GROUP,
            "root visit count equals completed iteration",
            int(root.mcts["visits"]) == trace.run_id,
            root,
        )

        for node in nodes:
            report.cover("logged_nodes")
            try:
                reconstructed = node.reconstruct_state(context.config)
                reconstructed.assert_valid(context.config)
            except Exception as error:
                report.check(
                    GROUP,
                    "reconstructed GV4State is valid",
                    False,
                    node,
                    repr(error),
                )
            else:
                report.check(GROUP, "reconstructed GV4State is valid", True, node)

            report.check(
                GROUP,
                "next adversary tick is not behind state time",
                float(node.mcts["next_adversary_tick"]) + context.epsilon >= node.now,
                node,
            )
            report.check(
                GROUP,
                "request count matches append-only next ID",
                len(_all_requests(node)) == int(node.mcts["next_request_id"]),
                node,
            )
            report.check(
                GROUP,
                "in-system count matches nonterminal requests",
                int(node.mcts["total_requests_in_system"])
                == sum(
                    request["lifecycle"] not in {"COMPLETED", "STOPPED", "DROPPED"}
                    for request in _all_requests(node).values()
                ),
                node,
            )
            _check_objective(node, context, report)
            _check_action_mask(node, context, report)

        for parent, child in zip(nodes, nodes[1:]):
            report.check(
                GROUP,
                "parent-child node chain",
                int(child.mcts["parent_node_id"]) == parent.key.node_id,
                child,
            )
            report.check(GROUP, "depth increments once", child.depth == parent.depth + 1, child)
            report.close(
                GROUP,
                "logged parent time",
                float(child.mcts["parent_time"]),
                parent.now,
                child,
            )
            report.check(GROUP, "time never moves backward", child.now >= parent.now, child)
            report.check(
                GROUP,
                "acting player equals parent turn",
                child.mcts["player_acted"] == parent.mcts["next_player"],
                child,
            )
            action = child.incoming_action
            report.check(
                GROUP,
                "action actor equals acting player",
                action is not None and action["actor"] == child.mcts["player_acted"],
                child,
            )
            expected_next = (
                "controller" if child.mcts["player_acted"] == "adversary" else "adversary"
            )
            report.check(
                GROUP,
                "strict alternating next player",
                child.mcts["next_player"] == expected_next,
                child,
            )
            report.check(
                GROUP,
                "terminal request never becomes active",
                _terminal_ids(parent).issubset(_terminal_ids(child)),
                child,
            )
            report.close(
                GROUP,
                "edge reward is objective delta",
                float(child.mcts["reward"]),
                float(parent.mcts["objective_total_cost"])
                - float(child.mcts["objective_total_cost"]),
                child,
            )
            report.close(
                GROUP,
                "elapsed-time edge discount",
                float(child.mcts["discount"]),
                context.config.reward.discount_for_elapsed(child.now - parent.now),
                child,
            )
            aliases = json.loads(child.mcts["alias_indices_json"])
            raw_index = int(child.mcts["raw_action_index"])
            mcts_index = int(child.mcts["canonical_action_index"])
            report.check(GROUP, "raw action belongs to alias set", raw_index in aliases, child)
            report.check(GROUP, "MCTS edge belongs to alias set", mcts_index in aliases, child)
            report.check(GROUP, "MCTS representative is minimum alias", mcts_index == min(aliases), child)
            _check_adversary_clock(parent, child, context, report)

        for node in nodes:
            used_count = sum(int(item["request_count"]) for item in node.launch_history)
            used_prefill = sum(int(item["prefill_tokens"]) for item in node.launch_history)
            report.check(
                GROUP,
                "launch-window request cap",
                used_count <= context.config.timing.max_requests_per_launch_window,
                node,
            )
            report.check(
                GROUP,
                "launch-window prefill cap",
                used_prefill
                <= context.config.request.target_prefill_tokens_per_request_window_average
                * context.config.timing.max_requests_per_launch_window,
                node,
            )
