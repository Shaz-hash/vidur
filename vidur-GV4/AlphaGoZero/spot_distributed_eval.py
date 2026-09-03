"""Bridge the arena evaluator to the pull-based Spot work queue."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

from vidur.AlphaGoZero.spot_work_protocol import (
    begin_eval_batch,
    cancel_eval_batch,
    finish_eval_batch,
    spot_root,
    wait_for_batch,
)


MergeBlock = Callable[[Path, list[Path], int], None]


def _env_int(name: str, default: int) -> int:
    value = str(os.environ.get(name, "")).strip()
    return int(value) if value else int(default)


def _env_float(name: str, default: float) -> float:
    value = str(os.environ.get(name, "")).strip()
    return float(value) if value else float(default)


def run_spot_pull_commands(
    *,
    root: Path,
    block_commands: dict[str, list[str]],
    final_dirs: dict[str, Path],
    expected_games_by_block: dict[str, int],
    merge_block: MergeBlock,
) -> dict[str, Path]:
    """Queue one-game assignments and merge their standard arena outputs."""

    started_at = time.time()
    batch = begin_eval_batch(
        root,
        block_commands=block_commands,
        expected_games_by_block=expected_games_by_block,
        lease_sec=_env_int("AGZ_SPOT_ASSIGNMENT_LEASE_SEC", 3600),
    )
    batch_id = str(batch["batch_id"])
    try:
        status = wait_for_batch(
            root,
            batch_id=batch_id,
            timeout_sec=_env_int("AGZ_SPOT_PULL_EVAL_TIMEOUT_SEC", 7200),
            poll_sec=_env_float("AGZ_SPOT_PULL_EVAL_POLL_SEC", 2.0),
        )
        games_finished_at = time.time()
        parts_by_block: dict[str, list[Path]] = {}
        for block_name, task_ids in status["tasks_by_block"].items():
            final_dir = Path(final_dirs[block_name])
            parts: list[Path] = []
            for task_id in task_ids:
                source = spot_root(root) / "results" / str(task_id)
                if not source.is_dir():
                    raise FileNotFoundError(source)
                target = final_dir / "distributed_parts" / "spot" / str(task_id)
                shutil.rmtree(target, ignore_errors=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
                parts.append(target)
            parts_by_block[block_name] = parts
        for block_name, final_dir in final_dirs.items():
            merge_block(
                Path(final_dir),
                parts_by_block[block_name],
                int(expected_games_by_block[block_name]),
            )
        merged_at = time.time()
        finish_eval_batch(root, batch_id=batch_id)
        for task_id in status["task_ids"]:
            shutil.rmtree(spot_root(root) / "results" / str(task_id), ignore_errors=True)
        finished_at = time.time()
        plan = {
            "mode": "spot_pull_arena_v1",
            "batch_id": batch_id,
            "started_at_epoch": started_at,
            "finished_at_epoch": finished_at,
            "elapsed_sec": finished_at - started_at,
            "game_elapsed_sec": games_finished_at - started_at,
            "collect_merge_elapsed_sec": merged_at - games_finished_at,
            "cleanup_elapsed_sec": finished_at - merged_at,
            "tasks_by_block": status["tasks_by_block"],
        }
        for final_dir in final_dirs.values():
            (Path(final_dir) / "launch_command.json").write_text(
                json.dumps(plan, indent=2) + "\n",
                encoding="utf-8",
            )
        return {
            name: Path(path) / "arena_results.csv"
            for name, path in final_dirs.items()
        }
    except Exception as exc:
        cancel_eval_batch(root, batch_id=batch_id, reason=repr(exc))
        raise
