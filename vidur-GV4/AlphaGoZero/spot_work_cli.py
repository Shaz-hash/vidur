"""Coordinator-side CLI invoked by Spot workers over SSH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidur.AlphaGoZero.spot_work_protocol import (
    batch_status,
    complete_task,
    configure_scheduler,
    fail_task,
    heartbeat_selfplay_assignment,
    heartbeat_tasks,
    release_selfplay_assignment,
    request_work,
    scheduler_status,
)


def _json_object(raw: str) -> dict[str, str]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return {str(key): str(item) for key, item in value.items()}


def _typed_json_object(raw: str) -> dict[str, object]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return {str(key): item for key, item in value.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve AlphaGoZero Spot worker requests.")
    parser.add_argument("--root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    request = subparsers.add_parser("request")
    request.add_argument("--worker-id", required=True)
    request.add_argument("--capacity", type=int, required=True)
    request.add_argument("--cpu-count", type=int, default=None)
    request.add_argument("--reserved-cpu-count", type=int, default=0)
    request.add_argument("--available-memory-bytes", type=int, default=None)
    request.add_argument("--cpu-threads-per-game", type=int, default=1)
    request.add_argument("--memory-bytes-per-game", type=int, default=1024**3)

    configure = subparsers.add_parser("configure")
    configure.add_argument("--selfplay-total-parallel-games", type=int, required=True)
    configure.add_argument("--eval-total-parallel-games", type=int, required=True)
    configure.add_argument("--selfplay-lease-sec", type=int, default=300)
    configure.add_argument("--worker-heartbeat-timeout-sec", type=int, default=300)
    configure.add_argument("--selfplay-config-json", type=_typed_json_object, default=None)

    heartbeat = subparsers.add_parser("heartbeat")
    heartbeat.add_argument("--worker-id", required=True)
    heartbeat.add_argument("--leases-json", type=_json_object, required=True)
    heartbeat.add_argument("--extend-sec", type=int, default=300)

    selfplay_heartbeat = subparsers.add_parser("selfplay-heartbeat")
    selfplay_heartbeat.add_argument("--worker-id", required=True)
    selfplay_heartbeat.add_argument("--assignment-id", required=True)
    selfplay_heartbeat.add_argument("--lease-token", required=True)
    selfplay_heartbeat.add_argument("--extend-sec", type=int, default=300)

    selfplay_release = subparsers.add_parser("selfplay-release")
    selfplay_release.add_argument("--worker-id", required=True)
    selfplay_release.add_argument("--assignment-id", required=True)
    selfplay_release.add_argument("--lease-token", required=True)
    selfplay_release.add_argument("--status", required=True)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--worker-id", required=True)
    complete.add_argument("--task-id", required=True)
    complete.add_argument("--lease-token", required=True)

    fail = subparsers.add_parser("fail")
    fail.add_argument("--worker-id", required=True)
    fail.add_argument("--task-id", required=True)
    fail.add_argument("--lease-token", required=True)
    fail.add_argument("--error", required=True)
    fail.add_argument("--max-attempts", type=int, default=5)

    status = subparsers.add_parser("status")
    status.add_argument("--batch-id", required=True)

    scheduler = subparsers.add_parser("scheduler-status")
    scheduler.add_argument("--worker-freshness-sec", type=int, default=300)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "request":
        return request_work(
            args.root,
            worker_id=str(args.worker_id),
            capacity=int(args.capacity),
            cpu_count=args.cpu_count,
            reserved_cpu_count=int(args.reserved_cpu_count),
            available_memory_bytes=args.available_memory_bytes,
            cpu_threads_per_game=int(args.cpu_threads_per_game),
            memory_bytes_per_game=int(args.memory_bytes_per_game),
        )
    if args.command == "configure":
        return configure_scheduler(
            args.root,
            selfplay_total_parallel_games=int(args.selfplay_total_parallel_games),
            eval_total_parallel_games=int(args.eval_total_parallel_games),
            selfplay_lease_sec=int(args.selfplay_lease_sec),
            worker_heartbeat_timeout_sec=int(args.worker_heartbeat_timeout_sec),
            selfplay_config=args.selfplay_config_json,
        )
    if args.command == "heartbeat":
        return heartbeat_tasks(
            args.root,
            worker_id=str(args.worker_id),
            leases=dict(args.leases_json),
            extend_sec=int(args.extend_sec),
        )
    if args.command == "selfplay-heartbeat":
        return heartbeat_selfplay_assignment(
            args.root,
            worker_id=str(args.worker_id),
            assignment_id=str(args.assignment_id),
            lease_token=str(args.lease_token),
            extend_sec=int(args.extend_sec),
        )
    if args.command == "selfplay-release":
        return release_selfplay_assignment(
            args.root,
            worker_id=str(args.worker_id),
            assignment_id=str(args.assignment_id),
            lease_token=str(args.lease_token),
            status=str(args.status),
        )
    if args.command == "complete":
        return complete_task(
            args.root,
            worker_id=str(args.worker_id),
            task_id=str(args.task_id),
            lease_token=str(args.lease_token),
        )
    if args.command == "fail":
        return fail_task(
            args.root,
            worker_id=str(args.worker_id),
            task_id=str(args.task_id),
            lease_token=str(args.lease_token),
            error=str(args.error),
            max_attempts=int(args.max_attempts),
        )
    if args.command == "status":
        return batch_status(args.root, str(args.batch_id))
    if args.command == "scheduler-status":
        return scheduler_status(
            args.root,
            worker_freshness_sec=int(args.worker_freshness_sec),
        )
    raise ValueError(f"unsupported command {args.command!r}")


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
