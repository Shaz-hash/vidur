"""Capture exact MCTS simulation paths and split them into iteration folders."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from ..logger import GV4NodeCSVLogger


LOG_FILE_NAMES = (
    "mcts_nodes.csv",
    "requests.csv",
    "pipeline_nodes.csv",
    "kv_cache_nodes.csv",
)
KEY_FIELDS = ("run_id", "game_id", "root_id", "node_id")


class TraceCaptureError(RuntimeError):
    """Raised when raw iteration logs are missing, duplicated, or misaligned."""


class IterationPathObserver:
    """MCTS callback that records the selected path immediately after backup."""

    __slots__ = ("_expected_iteration", "_logger")

    def __init__(self, logger: GV4NodeCSVLogger) -> None:
        self._logger = logger
        self._expected_iteration = 1

    def __call__(
        self,
        *,
        iteration_index: int,
        path: Sequence[Any],
        root: Any,
        game_id: int,
        root_id: int,
    ) -> None:
        if iteration_index != self._expected_iteration:
            raise TraceCaptureError(
                f"expected MCTS iteration {self._expected_iteration}, got {iteration_index}"
            )
        if not path or path[0] is not root:
            raise TraceCaptureError("MCTS observer received a path from another root")
        self._logger.log_path(
            run_id=iteration_index,
            game_id=game_id,
            root_id=root_id,
            path=path,
        )
        self._expected_iteration += 1

    @property
    def iterations_logged(self) -> int:
        return self._expected_iteration - 1


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise TraceCaptureError(f"missing raw log {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TraceCaptureError(f"log has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    try:
        return tuple(row[field] for field in KEY_FIELDS)  # type: ignore[return-value]
    except KeyError as error:
        raise TraceCaptureError(f"missing join column {error.args[0]}") from error


def split_iteration_logs(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_iterations: int,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write exactly four path-local CSVs under one folder per MCTS iteration."""

    if expected_iterations <= 0:
        raise ValueError("expected_iterations must be positive")
    raw_root = Path(raw_dir).expanduser().resolve()
    trace_root = Path(output_dir).expanduser().resolve()
    trace_root.mkdir(parents=True, exist_ok=True)

    tables: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    keys_by_file: dict[str, set[tuple[str, str, str, str]]] = {}
    for file_name in LOG_FILE_NAMES:
        fields, rows = _read_csv(raw_root / file_name)
        keys = [_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise TraceCaptureError(f"duplicate composite key in {file_name}")
        tables[file_name] = fields, rows
        keys_by_file[file_name] = set(keys)

    reference_keys = keys_by_file[LOG_FILE_NAMES[0]]
    for file_name in LOG_FILE_NAMES[1:]:
        if keys_by_file[file_name] != reference_keys:
            raise TraceCaptureError(f"{file_name} does not join one-to-one with MCTS rows")

    expected_run_ids = {str(index) for index in range(1, expected_iterations + 1)}
    actual_run_ids = {key[0] for key in reference_keys}
    if actual_run_ids != expected_run_ids:
        raise TraceCaptureError(
            f"iteration IDs differ: expected {sorted(expected_run_ids)}, "
            f"got {sorted(actual_run_ids)}"
        )

    expected_directories = {
        f"mcts_iter_{index:06d}" for index in range(1, expected_iterations + 1)
    }
    unexpected = {path.name for path in trace_root.iterdir()} - expected_directories
    if unexpected:
        raise TraceCaptureError(
            "trace root contains unexpected entries: "
            + ", ".join(sorted(unexpected))
        )

    output_paths: list[Path] = []
    for iteration in range(1, expected_iterations + 1):
        run_id = str(iteration)
        iteration_dir = trace_root / f"mcts_iter_{iteration:06d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        unknown_entries = {
            path.name for path in iteration_dir.iterdir()
        } - set(LOG_FILE_NAMES)
        if unknown_entries:
            raise TraceCaptureError(
                f"{iteration_dir} contains unexpected entries: {sorted(unknown_entries)}"
            )

        _, mcts_rows = tables[LOG_FILE_NAMES[0]]
        selected_mcts = [row for row in mcts_rows if row["run_id"] == run_id]
        selected_mcts.sort(key=lambda row: int(row["depth"]))
        ordered_keys = [_key(row) for row in selected_mcts]
        if not ordered_keys:
            raise TraceCaptureError(f"iteration {iteration} has no MCTS path")

        for file_name in LOG_FILE_NAMES:
            fields, rows = tables[file_name]
            by_key = {_key(row): row for row in rows if row["run_id"] == run_id}
            selected = [by_key[key] for key in ordered_keys]
            path = iteration_dir / file_name
            if path.exists() and not overwrite:
                raise FileExistsError(f"iteration output already exists: {path}")
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(selected)
        output_paths.append(iteration_dir)

    return tuple(output_paths)
