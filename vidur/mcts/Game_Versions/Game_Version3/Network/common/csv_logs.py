from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


ALLOCATION_COLUMNS = (
    "session_id",
    "task_id",
    "machine_name",
    "machine_ip",
    "sent_at_utc",
    "generation",
    "model_version",
    "num_roots",
    "history_hops_min",
    "history_hops_max",
    "history_seed",
    "start_root_id",
    "game_id",
    "status",
)

RECEIVED_COLUMNS = (
    "session_id",
    "task_id",
    "machine_name",
    "machine_ip",
    "received_at_utc",
    "generation",
    "model_version",
    "num_roots",
    "history_hops_min",
    "history_hops_max",
    "history_seed",
    "roots_generated",
    "train_samples",
    "eval_samples",
    "result_dir",
    "status",
    "error",
)

CLIENT_COLUMNS = (
    "session_id",
    "task_id",
    "received_at_utc",
    "started_at_utc",
    "finished_at_utc",
    "sent_back_at_utc",
    "model_version",
    "num_roots",
    "history_hops_min",
    "history_hops_max",
    "history_seed",
    "roots_generated",
    "train_samples",
    "eval_samples",
    "status",
    "error",
)


def append_csv_row(path: Path, columns: Iterable[str], row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = tuple(columns)
    write_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in cols})

