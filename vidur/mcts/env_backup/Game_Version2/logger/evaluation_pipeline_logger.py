from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any
import json


def generation_log_dir_from_iter_log(iter_log_path: str, generation: int) -> Path:
    return Path(iter_log_path).parent / f"gen_{int(generation):06d}"

def arena_state_snapshot_for_log(env: Any, state: Any) -> dict:
    req_lookup = env._build_request_lookup(state.simulator, state=state)

    active_ids = sorted(int(x) for x in state.stats.active_request_ids)
    completed_ids = sorted(int(x) for x in state.stats.completed_request_ids)
    decode_credit_balance = int(env._v2_decode_credit_balance_raw(state))

    decode_processed_tokens_by_id: dict[int, int] = {}
    prefill_remaining_by_id: dict[int, int] = {}

    for rid in active_ids:
        req = req_lookup.get(int(rid))
        if req is None or bool(getattr(req, "completed", False)):
            continue

        prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
        rem_pref = int(env._remaining_prefill(req))
        rem_dec = int(env._remaining_decode(req))

        if prefill_done and rem_dec > 0:
            decode_processed_tokens_by_id[int(rid)] = int(getattr(req, "num_processed_decode_tokens", 0))

        if (not prefill_done) and rem_pref > 0:
            prefill_remaining_by_id[int(rid)] = int(rem_pref)

    return {
        "active_request_ids": active_ids,
        "completed_request_ids": completed_ids,
        "decode_credit_balance": decode_credit_balance,
        "decode_processed_tokens_by_id": decode_processed_tokens_by_id,
        "prefill_remaining_by_id": prefill_remaining_by_id,
    }

class ArenaGameCycleFileLogger:
    _FIELDS = [
        "game_id",
        "cycle_label",
        "phase",
        "turn",
        "depth",
        "player_acted",
        "player_to_act_next",
        "action_repr",
        "sim_time_before",
        "sim_time_after",
        "total_cost",
        "slo_violations",
        "total_lateness",
        "active_request_ids",
        "completed_request_ids",
        "decode_credit_balance",
        "decode_processed_tokens_by_id",
        "prefill_remaining_by_id",
        "end_reason",
    ]

    def __init__(self, games_dir: Path) -> None:
        self.games_dir = Path(games_dir)
        self.games_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_cycle(self, game_id: int, cycle_label: str) -> Path:
        if cycle_label == "candidate_as_adversary":
            name = f"game_{int(game_id)}_adv_candidate_ctrl_best.csv"
        elif cycle_label == "best_as_adversary":
            name = f"game_{int(game_id)}_adv_best_ctrl_candidate.csv"
        else:
            name = f"game_{int(game_id)}_{str(cycle_label)}.csv"
        return self.games_dir / name

    def _append_row(self, path: Path, row: dict) -> None:
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def write_step(
        self,
        *,
        game_id: int,
        cycle_label: str,
        phase: str,
        turn: int,
        depth: int,
        player_acted: str,
        player_to_act_next: str,
        action_repr: str,
        sim_time_before: float,
        sim_time_after: float,
        total_cost: float | None = None,
        slo_violations: int | None = None,
        total_lateness: float | None = None,
        active_request_ids: list[int] | None = None,
        completed_request_ids: list[int] | None = None,
        decode_credit_balance: int | None = None,
        decode_processed_tokens_by_id: dict[int, int] | None = None,
        prefill_remaining_by_id: dict[int, int] | None = None,
        end_reason: str = "",
    ) -> None:
        path = self._path_for_cycle(game_id=game_id, cycle_label=cycle_label)
        self._append_row(
            path,
            {
                "game_id": int(game_id),
                "cycle_label": str(cycle_label),
                "phase": str(phase),
                "turn": int(turn),
                "depth": int(depth),
                "player_acted": str(player_acted),
                "player_to_act_next": str(player_to_act_next),
                "action_repr": str(action_repr),
                "sim_time_before": float(sim_time_before),
                "sim_time_after": float(sim_time_after),
                "total_cost": "" if total_cost is None else float(total_cost),
                "slo_violations": "" if slo_violations is None else int(slo_violations),
                "total_lateness": "" if total_lateness is None else float(total_lateness),
                "active_request_ids": "" if active_request_ids is None else json.dumps([int(x) for x in active_request_ids]),
                "completed_request_ids": "" if completed_request_ids is None else json.dumps([int(x) for x in completed_request_ids]),
                "decode_credit_balance": "" if decode_credit_balance is None else int(decode_credit_balance),
                "decode_processed_tokens_by_id": "" if decode_processed_tokens_by_id is None else json.dumps({int(k): int(v) for k, v in decode_processed_tokens_by_id.items()}, sort_keys=True),
                "prefill_remaining_by_id": "" if prefill_remaining_by_id is None else json.dumps({int(k): int(v) for k, v in prefill_remaining_by_id.items()}, sort_keys=True),
                "end_reason": str(end_reason),
            },
        )


    def write_cycle_end(
        self,
        *,
        game_id: int,
        cycle_label: str,
        total_cost: float,
        slo_violations: int,
        total_lateness: float,
        end_reason: str = "",
    ) -> None:
        path = self._path_for_cycle(game_id=game_id, cycle_label=cycle_label)
        self._append_row(
            path,
            {
                "game_id": int(game_id),
                "cycle_label": str(cycle_label),
                "phase": "arena_end",
                "turn": "",
                "depth": "",
                "player_acted": "",
                "player_to_act_next": "",
                "action_repr": "",
                "sim_time_before": "",
                "sim_time_after": "",
                "total_cost": float(total_cost),
                "slo_violations": int(slo_violations),
                "total_lateness": float(total_lateness),
                "active_request_ids": "",
                "completed_request_ids": "",
                "decode_credit_balance": "",
                "decode_processed_tokens_by_id": "",
                "prefill_remaining_by_id": "",
                "end_reason": str(end_reason),
            },
        )



class EvaluationMetricsLogger:
    """
    Per-generation evaluation summary.
    """

    FIELDS = [
        "time",
        "generation",
        "candidate_checkpoint",
        "best_checkpoint_before",
        "best_checkpoint_after",
        "num_games",
        "candidate_points",
        "best_points",
        "total_points",
        "candidate_win_rate",
        "win_threshold",
        "passed",
        "promoted_to_best",
        "arena_results_csv",
        "arena_games_dir",
    ]

    def __init__(self, path: Path, *, flush_every: int = 1) -> None:
        self._path = Path(path)
        self._flush_every = max(1, int(flush_every))
        self._file = None
        self._writer = None
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

    def log_generation(
        self,
        *,
        generation: int,
        candidate_checkpoint: str,
        best_checkpoint_before: str,
        best_checkpoint_after: str,
        num_games: int,
        candidate_points: float,
        best_points: float,
        total_points: float,
        candidate_win_rate: float,
        win_threshold: float,
        passed: bool,
        promoted_to_best: bool,
        arena_results_csv: str,
        arena_games_dir: str,
    ) -> None:
        self._ensure()
        assert self._writer is not None

        row = {
            "time": float(time.time()),
            "generation": int(generation),
            "candidate_checkpoint": str(candidate_checkpoint),
            "best_checkpoint_before": str(best_checkpoint_before),
            "best_checkpoint_after": str(best_checkpoint_after),
            "num_games": int(num_games),
            "candidate_points": float(candidate_points),
            "best_points": float(best_points),
            "total_points": float(total_points),
            "candidate_win_rate": float(candidate_win_rate),
            "win_threshold": float(win_threshold),
            "passed": bool(passed),
            "promoted_to_best": bool(promoted_to_best),
            "arena_results_csv": str(arena_results_csv),
            "arena_games_dir": str(arena_games_dir),
        }

        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0 and self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
