from __future__ import annotations

import csv
from pathlib import Path


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)  # type: ignore[arg-type]
    except Exception:
        return default


class ReplayBufferLogger:
    FIELDS = [
        "candidate_generation",
        "best_model_generation",
        "replay_samples_size",
        "effective_training_steps",
        "replay_generation_ids",
    ]

    def __init__(self, path: Path, *, flush_every: int = 1) -> None:
        self._path = Path(path)
        self._flush_every = max(1, int(flush_every))
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._rows = 0

    def _ensure(self) -> None:
        if self._writer is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._path.exists()
        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()

    def log_row(
        self,
        *,
        candidate_generation: int,
        best_model_generation: int,
        replay_samples_size: int,
        effective_training_steps: int,
        replay_generation_ids: str = "",
    ) -> None:
        self._ensure()
        assert self._writer is not None

        self._writer.writerow(
            {
                "candidate_generation": _safe_int(candidate_generation),
                "best_model_generation": _safe_int(best_model_generation),
                "replay_samples_size": _safe_int(replay_samples_size),
                "effective_training_steps": _safe_int(effective_training_steps),
                "replay_generation_ids": str(replay_generation_ids),
            }
        )
        self._rows += 1
        if self._rows % self._flush_every == 0 and self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
