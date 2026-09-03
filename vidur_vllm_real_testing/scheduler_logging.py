"""Append-only scheduler decision logging and blocking-time accounting."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
import threading
from typing import Any, Mapping


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class SchedulerDecisionLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path).expanduser().resolve()
        self._planning_path = (
            None if self.path is None else Path(f"{self.path}.planning")
        )
        self._lock = threading.Lock()
        self._blocking_s = 0.0

    @property
    def total_blocking_s(self) -> float:
        with self._lock:
            return self._blocking_s

    def begin_planning(self, started_monotonic_s: float) -> None:
        with self._lock:
            if self._planning_path is None:
                return
            self._planning_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "started_monotonic_s": float(started_monotonic_s),
                "blocking_total_before_s": self._blocking_s,
            }
            temporary = Path(f"{self._planning_path}.tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(self._planning_path)

    def cancel_planning(self) -> None:
        with self._lock:
            if self._planning_path is not None:
                self._planning_path.unlink(missing_ok=True)

    def write(self, row: Mapping[str, Any], *, blocking_s: float) -> None:
        with self._lock:
            self._blocking_s += max(0.0, float(blocking_s))
            if self.path is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(row)
            payload["controller_blocking_s"] = float(blocking_s)
            payload["controller_blocking_total_s"] = self._blocking_s
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=_json_default,
                    )
                    + "\n"
                )
            if self._planning_path is not None:
                self._planning_path.unlink(missing_ok=True)
