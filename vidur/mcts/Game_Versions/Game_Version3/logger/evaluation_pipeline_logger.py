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
        "selection_mode",          # NEW
        "valid_action_count",      # NEW
        "iterations_requested",    # NEW
        "iterations_used",         # NEW
        "chosen_q_value",
        "chosen_reward",
        "chosen_discount",
        "chosen_bootstrap",
        "chosen_child_cost",
        "candidate_ranking_mode",
        "candidate_top5_action_reprs",
        "candidate_top5_q_values",
        "candidate_top5_rewards",
        "candidate_top5_discounts",
        "candidate_top5_bootstraps",
        "candidate_top5_child_costs",
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
        selection_mode: str | None = None,          # NEW
        valid_action_count: int | None = None,      # NEW
        iterations_requested: int | None = None,    # NEW
        iterations_used: int | None = None,         # NEW
        chosen_q_value: float | None = None,
        chosen_reward: float | None = None,
        chosen_discount: float | None = None,
        chosen_bootstrap: float | None = None,
        chosen_child_cost: float | None = None,
        candidate_ranking_mode: str | None = None,
        candidate_top5_action_reprs: list[str] | None = None,
        candidate_top5_q_values: list[float] | None = None,
        candidate_top5_rewards: list[float] | None = None,
        candidate_top5_discounts: list[float] | None = None,
        candidate_top5_bootstraps: list[float] | None = None,
        candidate_top5_child_costs: list[float] | None = None,
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
                "selection_mode": "" if selection_mode is None else str(selection_mode),                        # NEW
                "valid_action_count": "" if valid_action_count is None else int(valid_action_count),            # NEW
                "iterations_requested": "" if iterations_requested is None else int(iterations_requested),      # NEW
                "iterations_used": "" if iterations_used is None else int(iterations_used),                     # NEW
                "chosen_q_value": "" if chosen_q_value is None else float(chosen_q_value),
                "chosen_reward": "" if chosen_reward is None else float(chosen_reward),
                "chosen_discount": "" if chosen_discount is None else float(chosen_discount),
                "chosen_bootstrap": "" if chosen_bootstrap is None else float(chosen_bootstrap),
                "chosen_child_cost": "" if chosen_child_cost is None else float(chosen_child_cost),
                "candidate_ranking_mode": "" if candidate_ranking_mode is None else str(candidate_ranking_mode),
                "candidate_top5_action_reprs": "" if candidate_top5_action_reprs is None else json.dumps([str(x) for x in candidate_top5_action_reprs]),
                "candidate_top5_q_values": "" if candidate_top5_q_values is None else json.dumps([float(x) for x in candidate_top5_q_values]),
                "candidate_top5_rewards": "" if candidate_top5_rewards is None else json.dumps([float(x) for x in candidate_top5_rewards]),
                "candidate_top5_discounts": "" if candidate_top5_discounts is None else json.dumps([float(x) for x in candidate_top5_discounts]),
                "candidate_top5_bootstraps": "" if candidate_top5_bootstraps is None else json.dumps([float(x) for x in candidate_top5_bootstraps]),
                "candidate_top5_child_costs": "" if candidate_top5_child_costs is None else json.dumps([float(x) for x in candidate_top5_child_costs]),
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
                "selection_mode": "",
                "valid_action_count": "",
                "iterations_requested": "",
                "iterations_used": "",
                "chosen_q_value": "",
                "chosen_reward": "",
                "chosen_discount": "",
                "chosen_bootstrap": "",
                "chosen_child_cost": "",
                "candidate_ranking_mode": "",
                "candidate_top5_action_reprs": "",
                "candidate_top5_q_values": "",
                "candidate_top5_rewards": "",
                "candidate_top5_discounts": "",
                "candidate_top5_bootstraps": "",
                "candidate_top5_child_costs": "",
                "end_reason": str(end_reason),
            },
        )


class ArenaModelActionDetailLogger:
    _FIELDS = [
        "game_id",
        "cycle_label",
        "phase",
        "turn",
        "depth",
        "player_acted",
        "rank",
        "action_repr",
        "q_value",
        "immediate_reward",
        "discount",
        "bootstrap_value",
        "child_cost",
    ]

    def __init__(self, games_dir: Path) -> None:
        self.games_dir = Path(games_dir)
        self.games_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_cycle(self, game_id: int, cycle_label: str) -> Path:
        return self.games_dir / f"game_{int(game_id)}_{str(cycle_label)}_model_action_details.csv"

    def write_ranked_actions(
        self,
        *,
        game_id: int,
        cycle_label: str,
        phase: str,
        turn: int,
        depth: int,
        player_acted: str,
        ranked_rows: list[dict],
    ) -> None:
        path = self._path_for_cycle(game_id=game_id, cycle_label=cycle_label)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._FIELDS)
            if write_header:
                w.writeheader()
            for row in ranked_rows:
                w.writerow(
                    {
                        "game_id": int(game_id),
                        "cycle_label": str(cycle_label),
                        "phase": str(phase),
                        "turn": int(turn),
                        "depth": int(depth),
                        "player_acted": str(player_acted),
                        "rank": int(row.get("rank", 0)),
                        "action_repr": str(row.get("action_repr", "")),
                        "q_value": float(row.get("q_value", 0.0)),
                        "immediate_reward": float(row.get("immediate_reward", 0.0)),
                        "discount": float(row.get("discount", 0.0)),
                        "bootstrap_value": float(row.get("bootstrap_value", 0.0)),
                        "child_cost": float(row.get("child_cost", 0.0)),
                    }
                )


class EvaluationMetricsLogger:
    """Generation-level training/eval value-metric logger."""

    FIELDS = [
        "time",
        "generation",
        "model_version",
        "num_roots_required",
        "num_unique_roots_created",
        "phase",
        "samples_needed",
        "controller_training_samples",
        "controller_eval_samples",
        "adversary_training_samples",
        "adversary_eval_samples",
        "loss_for_selection",
        "policy_loss",
        "value_loss",
        "value_logsumexp_loss",
        "controller_policy_loss",
        "controller_value_loss",
        "controller_value_logsumexp_loss",
        "adversary_policy_loss",
        "adversary_value_loss",
        "adversary_value_logsumexp_loss",
        "value_mse_error",
        "value_mae_error",
        "controller_value_mse_error",
        "controller_value_mae_error",
        "adversary_value_mse_error",
        "adversary_value_mae_error",
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
        expected_header = ",".join(self.FIELDS)
        write_header = True
        if self._path.exists():
            try:
                if self._path.stat().st_size > 0:
                    write_header = False
                    with self._path.open("r", encoding="utf-8") as f:
                        existing_header = f.readline().rstrip("\n")
                    if existing_header != expected_header:
                        backup = self._path.with_suffix(self._path.suffix + f".bak.{int(time.time())}")
                        self._path.replace(backup)
                        write_header = True
            except OSError:
                write_header = True

        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()

    def log_phase(
        self,
        *,
        generation: int,
        model_version: int,
        num_roots_required: int,
        num_unique_roots_created: int,
        phase: str,
        samples_needed: int,
        controller_training_samples: int,
        controller_eval_samples: int,
        adversary_training_samples: int,
        adversary_eval_samples: int,
        loss_for_selection: float,
        policy_loss: float,
        value_loss: float,
        controller_policy_loss: float,
        controller_value_loss: float,
        adversary_policy_loss: float,
        adversary_value_loss: float,
        value_mse_error: float,
        value_mae_error: float,
        controller_value_mse_error: float,
        controller_value_mae_error: float,
        adversary_value_mse_error: float,
        adversary_value_mae_error: float,
    ) -> None:
        self._ensure()
        assert self._writer is not None

        row = {
            "time": float(time.time()),
            "generation": int(generation),
            "model_version": int(model_version),
            "num_roots_required": int(num_roots_required),
            "num_unique_roots_created": int(num_unique_roots_created),
            "phase": str(phase),
            "samples_needed": int(samples_needed),
            "controller_training_samples": int(controller_training_samples),
            "controller_eval_samples": int(controller_eval_samples),
            "adversary_training_samples": int(adversary_training_samples),
            "adversary_eval_samples": int(adversary_eval_samples),
            "loss_for_selection": float(loss_for_selection),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "value_logsumexp_loss": float(value_loss),
            "controller_policy_loss": float(controller_policy_loss),
            "controller_value_loss": float(controller_value_loss),
            "controller_value_logsumexp_loss": float(controller_value_loss),
            "adversary_policy_loss": float(adversary_policy_loss),
            "adversary_value_loss": float(adversary_value_loss),
            "adversary_value_logsumexp_loss": float(adversary_value_loss),
            "value_mse_error": float(value_mse_error),
            "value_mae_error": float(value_mae_error),
            "controller_value_mse_error": float(controller_value_mse_error),
            "controller_value_mae_error": float(controller_value_mae_error),
            "adversary_value_mse_error": float(adversary_value_mse_error),
            "adversary_value_mae_error": float(adversary_value_mae_error),
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


class EvalRootPredictionLogger:
    """Per-generation eval-root prediction logger."""

    FIELDS = [
        "model_version",
        "root_number",
        "root_player_type",
        "model_predicted_value",
        "true_tree_search_value",
    ]

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "model_version": int(row["model_version"]),
                        "root_number": int(row["root_number"]),
                        "root_player_type": str(row["root_player_type"]),
                        "model_predicted_value": float(row["model_predicted_value"]),
                        "true_tree_search_value": float(row["true_tree_search_value"]),
                    }
                )
