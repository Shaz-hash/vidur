"""Independent Vidur timing and compact FIFO PP-calendar checks."""

from __future__ import annotations

from ..action_resolver import ControllerTransitionKind
from .trace_context import IterationTrace, TraceNode
from .validation import ValidationContext, ValidationReport


GROUP = "pipeline"


def _calendar_by_id(stage: dict) -> dict[int, dict]:
    return {
        int(batch["microbatch_id"]): batch
        for batch in stage["inflight_batch_calendars"]
    }


def _check_pipeline_snapshot(
    node: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    config = context.config
    expected_inflight = 0
    for replica_id in range(config.topology.num_replicas):
        stages = [
            node.stage(replica_id, stage_index)
            for stage_index in range(config.topology.pipeline_parallel_size)
        ]
        calendars = [_calendar_by_id(stage) for stage in stages]
        batch_ids = set(calendars[0])
        expected_inflight += len(batch_ids)
        report.check(
            GROUP,
            "all PP stages describe the same in-flight batches",
            all(set(stage_calendars) == batch_ids for stage_calendars in calendars),
            node,
        )
        report.check(
            GROUP,
            "in-flight pipeline bound",
            len(batch_ids) <= config.scheduler.max_inflight_microbatches,
            node,
        )

        for stage_index, (stage, stage_calendars) in enumerate(zip(stages, calendars)):
            active = [
                batch for batch in stage_calendars.values() if batch["status"] == "active"
            ]
            report.check(GROUP, "at most one active batch per stage", len(active) <= 1, node)
            ordered = [stage_calendars[batch_id] for batch_id in sorted(stage_calendars)]
            previous_finish = None
            for batch in ordered:
                ready = float(batch["ready_time"])
                start = float(batch["start_time"])
                finish = float(batch["stage_completion_time"])
                report.check(GROUP, "stage calendar satisfies ready <= start < finish", ready <= start < finish, node)
                if previous_finish is not None:
                    report.check(
                        GROUP,
                        "FIFO stage batches do not overlap",
                        start + context.epsilon >= previous_finish,
                        node,
                    )
                previous_finish = finish
                expected_status = (
                    "queued"
                    if node.now + context.epsilon < start
                    else "active"
                    if node.now < finish - context.epsilon
                    else "stage_complete"
                )
                report.check(
                    GROUP,
                    "stage status matches node time",
                    batch["status"] == expected_status,
                    node,
                )
                report.check(
                    GROUP,
                    "stage tail covers scheduled work",
                    float(stage["tail_finish_time"]) + context.epsilon >= finish,
                    node,
                )

            if stage_index > 0:
                prior = calendars[stage_index - 1]
                for batch_id in batch_ids:
                    report.check(
                        GROUP,
                        "next PP stage waits for prior-stage completion",
                        float(stage_calendars[batch_id]["ready_time"]) + context.epsilon
                        >= float(prior[batch_id]["stage_completion_time"]),
                        node,
                    )

        for batch_id in batch_ids:
            reference_tokens = calendars[0][batch_id]["request_tokens"]
            report.check(
                GROUP,
                "all PP stages carry the same microbatch",
                all(
                    stage_calendars[batch_id]["request_tokens"] == reference_tokens
                    for stage_calendars in calendars
                ),
                node,
            )
            final_completion = float(calendars[-1][batch_id]["stage_completion_time"])
            report.close(
                GROUP,
                "batch completion equals final-stage finish",
                float(calendars[0][batch_id]["batch_completion_time"]),
                final_completion,
                node,
            )

    report.check(
        GROUP,
        "pipeline aggregate in-flight count",
        int(node.pipeline["total_inflight_microbatches"]) == expected_inflight,
        node,
    )


def _expected_calendar(
    parent: TraceNode,
    replica_id: int,
    service: tuple[float, ...],
    communication: tuple[float, ...],
    context: ValidationContext,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    ready_times = []
    start_times = []
    finish_times = []
    ready = parent.now
    for stage_index, duration in enumerate(service):
        stage = parent.stage(replica_id, stage_index)
        start = max(ready, float(stage["tail_finish_time"]))
        finish = round(
            start + duration,
            context.config.timing.time_round_digits,
        )
        ready_times.append(ready)
        start_times.append(start)
        finish_times.append(finish)
        if stage_index < len(communication):
            ready = round(
                finish + communication[stage_index],
                context.config.timing.time_round_digits,
            )
    return tuple(ready_times), tuple(start_times), tuple(finish_times)


def _check_batch_edge(
    parent: TraceNode,
    child: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    action = child.resolved_controller_action()
    if action is None or action.transition_kind != ControllerTransitionKind.BATCH:
        return
    report.cover("vidur_timed_batches")
    parent_state = parent.reconstruct_state(context.config)
    service, communication = context.timing_provider(parent_state, action)
    report.check(
        GROUP,
        "one Vidur service time per PP stage",
        len(service) == context.config.topology.pipeline_parallel_size,
        child,
    )
    report.check(
        GROUP,
        "one Vidur communication time per PP boundary",
        len(communication) == context.config.topology.pipeline_parallel_size - 1,
        child,
    )
    ready, start, finish = _expected_calendar(
        parent, action.replica_id, tuple(service), tuple(communication), context
    )
    microbatch_id = int(parent.mcts["next_microbatch_id"])
    child_stages = [
        child.stage(action.replica_id, stage_index)
        for stage_index in range(context.config.topology.pipeline_parallel_size)
    ]
    child_calendars = [_calendar_by_id(stage) for stage in child_stages]
    present = microbatch_id in child_calendars[0]

    if present:
        report.cover("batches_visible_after_action")
        for stage_index, calendars in enumerate(child_calendars):
            batch = calendars[microbatch_id]
            report.close(GROUP, "Vidur-derived ready time", float(batch["ready_time"]), ready[stage_index], child)
            report.close(GROUP, "Vidur-derived stage start", float(batch["start_time"]), start[stage_index], child)
            report.close(
                GROUP,
                "Vidur-derived stage completion",
                float(batch["stage_completion_time"]),
                finish[stage_index],
                child,
            )
            expected_tokens = {
                str(item.request_id): {
                    "prefill_tokens": item.prefill_tokens,
                    "decode_tokens": item.decode_tokens,
                    "total_tokens": item.total_tokens,
                }
                for item in action.allocations
            }
            report.check(
                GROUP,
                "first-stage microbatch equals controller action",
                batch["request_tokens"] == expected_tokens,
                child,
            )
        report.close(
            GROUP,
            "stage-zero next admission time",
            float(child_stages[0]["tail_finish_time"]),
            finish[0],
            child,
        )
    else:
        report.cover("batches_completed_before_child_snapshot")
        report.check(
            GROUP,
            "removed batch reached final completion",
            child.now + context.epsilon >= finish[-1],
            child,
        )
        parent_requests = parent.requests_by_id()
        child_requests = child.requests_by_id()
        for allocation in action.allocations:
            before = parent_requests[allocation.request_id]
            after = child_requests[allocation.request_id]
            report.check(
                GROUP,
                "completed prefill allocation commits progress",
                int(after["committed_prefill_tokens"])
                >= int(before["committed_prefill_tokens"]) + allocation.prefill_tokens,
                child,
            )
            report.check(
                GROUP,
                "completed decode allocation commits progress",
                int(after["committed_decode_tokens"])
                >= int(before["committed_decode_tokens"]) + allocation.decode_tokens,
                child,
            )


def _check_disappearing_batches(
    parent: TraceNode,
    child: TraceNode,
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    for replica_id in range(context.config.topology.num_replicas):
        parent_batches = _calendar_by_id(parent.stage(replica_id, 0))
        child_batches = _calendar_by_id(child.stage(replica_id, 0))
        for microbatch_id in set(parent_batches) - set(child_batches):
            report.cover("pipeline_completions")
            completion = float(parent_batches[microbatch_id]["batch_completion_time"])
            report.check(
                GROUP,
                "batch leaves ring only after final-stage completion",
                child.now + context.epsilon >= completion,
                child,
            )


def run_pipeline_tests(
    traces: tuple[IterationTrace, ...],
    context: ValidationContext,
    report: ValidationReport,
) -> None:
    for trace in traces:
        for node in trace.nodes:
            _check_pipeline_snapshot(node, context, report)
        for parent, child in zip(trace.nodes, trace.nodes[1:]):
            _check_batch_edge(parent, child, context, report)
            _check_disappearing_batches(parent, child, context, report)
