# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


class EvalArenaStepLogger:
    """
    Per-step debug logger for one (game_id, cycle_label) trajectory.
    """

    FIELDS = [
        "game_id",
        "history_length",
        "best_model_player",
        "candidate_model_player",
        "root_id",
        "root_depth",
        "root_node_id",
        "root_player",
        "model_root_value_controller",
        "model_root_prior_json",
        "best_action_index",
        "best_action_repr",
        "requests_in_system",
        "state_waiting_ids",
        "state_completed_request_ids",
        "adversary_requests_generated",
        "adversary_moves_total_so_far",
        "adversary_request_moves_so_far",
        "adversary_prefill_deadlines_by_id",
        "sim_time",
        "slo_violations",
        "total_lateness",
        "total_cost",
        "cycle_label",
        "phase",
        "step_index",
        "acting_player",
        "end_reason",
    ]

    def __init__(self, path: Path, *, flush_every: int = 1) -> None:
        self._path = Path(path)
        self._flush_every = max(1, int(flush_every))
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows = 0

    def _ensure(self) -> None:
        if self._writer is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def log_step(self, row: Dict[str, Any]) -> None:
        self._ensure()
        assert self._writer is not None

        full = {
            "game_id": _safe_int(row.get("game_id", 0)),
            "history_length": _safe_int(row.get("history_length", 0)),
            "best_model_player": str(row.get("best_model_player", "")),
            "candidate_model_player": str(row.get("candidate_model_player", "")),
            "root_id": _safe_int(row.get("root_id", 0)),
            "root_depth": _safe_int(row.get("root_depth", 0)),
            "root_node_id": "" if row.get("root_node_id", None) is None else _safe_int(row.get("root_node_id")),
            "root_player": str(row.get("root_player", "")),
            "model_root_value_controller": "" if row.get("model_root_value_controller", None) is None else _safe_float(row.get("model_root_value_controller")),
            "model_root_prior_json": str(row.get("model_root_prior_json", "[]")),
            "best_action_index": "" if row.get("best_action_index", None) is None else _safe_int(row.get("best_action_index")),
            "best_action_repr": str(row.get("best_action_repr", "")),
            "requests_in_system": _safe_int(row.get("requests_in_system", 0)),
            "state_waiting_ids": str(row.get("state_waiting_ids", "[]")),
            "state_completed_request_ids": str(row.get("state_completed_request_ids", "[]")),
            "adversary_requests_generated": _safe_int(row.get("adversary_requests_generated", 0)),
            "adversary_moves_total_so_far": _safe_int(row.get("adversary_moves_total_so_far", 0)),
            "adversary_request_moves_so_far": _safe_int(row.get("adversary_request_moves_so_far", 0)),
            "adversary_prefill_deadlines_by_id": str(row.get("adversary_prefill_deadlines_by_id", "{}")),
            "sim_time": _safe_float(row.get("sim_time", 0.0)),
            "slo_violations": _safe_int(row.get("slo_violations", 0)),
            "total_lateness": _safe_float(row.get("total_lateness", 0.0)),
            "total_cost": _safe_float(row.get("total_cost", 0.0)),
            "cycle_label": str(row.get("cycle_label", "")),
            "phase": str(row.get("phase", "")),
            "step_index": _safe_int(row.get("step_index", 0)),
            "acting_player": str(row.get("acting_player", "")),
            "end_reason": str(row.get("end_reason", "")),
        }

        self._writer.writerow(full)
        self._rows += 1
        if self._rows % self._flush_every == 0 and self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


class EvalArenaSummaryLogger:
    """
    Per-generation summary logger.
    """

    FIELDS = [
        "game_id",
        "history_length",
        "best_model_player",
        "candidate_model_player",
        "slo_cost",
        "winner",
        "cycle_label",
    ]

    def __init__(self, path: Path, *, flush_every: int = 1) -> None:
        self._path = Path(path)
        self._flush_every = max(1, int(flush_every))
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows = 0

    def _ensure(self) -> None:
        if self._writer is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def log_row(self, row: Dict[str, Any]) -> None:
        self._ensure()
        assert self._writer is not None
        full = {
            "game_id": _safe_int(row.get("game_id", 0)),
            "history_length": _safe_int(row.get("history_length", 0)),
            "best_model_player": str(row.get("best_model_player", "")),
            "candidate_model_player": str(row.get("candidate_model_player", "")),
            "slo_cost": _safe_float(row.get("slo_cost", 0.0)),
            "winner": str(row.get("winner", "")),
            "cycle_label": str(row.get("cycle_label", "")),
        }
        self._writer.writerow(full)
        self._rows += 1
        if self._rows % self._flush_every == 0 and self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


class EvalArenaGenerationLogger:
    """
    Owns per-game step CSVs and one generation summary CSV.

    File naming:
      - candidate_as_adversary/game_<game_id>.csv
      - best_as_adversary/game_<game_id>.csv
      - arena_results.csv
    """

    def __init__(
        self,
        *,
        gen_dir: Path,
        sampled_game_ids: Sequence[int],
        flush_every: int = 1,
    ) -> None:
        self.gen_dir = Path(gen_dir)
        self.sampled_game_ids = {int(x) for x in sampled_game_ids}
        self.flush_every = max(1, int(flush_every))
        self._step_loggers: Dict[Tuple[int, str], EvalArenaStepLogger] = {}
        self.summary = EvalArenaSummaryLogger(self.gen_dir / "arena_results.csv", flush_every=self.flush_every)

    def should_log_game(self, game_id: int) -> bool:
        return int(game_id) in self.sampled_game_ids

    def step_logger(self, game_id: int, cycle_label: str) -> Optional[EvalArenaStepLogger]:
        gid = int(game_id)
        if gid not in self.sampled_game_ids:
            return None
        key = (gid, str(cycle_label))
        lg = self._step_loggers.get(key)
        if lg is not None:
            return lg

        path = self.gen_dir / cycle_label / f"game_{gid}.csv"
        lg = EvalArenaStepLogger(path, flush_every=self.flush_every)
        self._step_loggers[key] = lg
        return lg

    def close(self) -> None:
        for lg in self._step_loggers.values():
            lg.close()
        self._step_loggers.clear()
        self.summary.close()
