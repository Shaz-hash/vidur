from __future__ import annotations

import argparse
from pathlib import Path

from ..common.csv_logs import CLIENT_COLUMNS, append_csv_row
from ..common.executor import execute_selfplay_task
from ..common.files import read_json
from ..common.types import NetworkSelfplayTask


def _client_log_row(task: NetworkSelfplayTask, result) -> dict:
    stats = dict(result.run_stats or {})
    return {
        "session_id": task.session_id,
        "task_id": task.task_id,
        "received_at_utc": result.received_at_utc,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "sent_back_at_utc": result.sent_back_at_utc,
        "model_version": int(task.model_version),
        "num_roots": int(task.num_roots),
        "history_hops_min": int(task.history_hops_min),
        "history_hops_max": int(task.history_hops_max),
        "history_seed": int(task.history_seed),
        "roots_generated": int(stats.get("num_roots_generated", 0)),
        "train_samples": int(stats.get("train_samples_total", 0)),
        "eval_samples": int(stats.get("eval_samples_total", 0)),
        "status": "ok" if bool(result.ok) else "failed",
        "error": str(result.error or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GV3 network self-play task.")
    parser.add_argument("--task", required=True, help="Path to task.json")
    args = parser.parse_args()

    task = NetworkSelfplayTask.from_dict(read_json(Path(args.task)))
    result = execute_selfplay_task(task)
    append_csv_row(
        Path(task.result_dir) / "client_tasks.csv",
        CLIENT_COLUMNS,
        _client_log_row(task, result),
    )
    if not result.ok:
        raise RuntimeError(f"Network task failed: {result.error}\n{result.traceback}")


if __name__ == "__main__":
    main()

