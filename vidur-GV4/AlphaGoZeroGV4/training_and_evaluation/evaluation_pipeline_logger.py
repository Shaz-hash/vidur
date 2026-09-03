"""Append-only CSV logging for GV4 training, arena, and baseline evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping, Sequence

from ..engine_runtime import StateSummary
from ..runner import GameCycleResult, PlayedStep


__all__ = ["EvaluationPipelineLogger"]


_TERMINAL = {"COMPLETED", "STOPPED", "DROPPED"}


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "cycle"


def _state_fields(state: StateSummary) -> dict[str, Any]:
    active = [
        request for request in state.requests if request.lifecycle not in _TERMINAL
    ]
    completed = [
        request.request_id
        for request in state.requests
        if request.lifecycle == "COMPLETED"
    ]
    stopped = [
        request.request_id
        for request in state.requests
        if request.lifecycle == "STOPPED"
    ]
    dropped = [
        request.request_id
        for request in state.requests
        if request.lifecycle == "DROPPED"
    ]
    prefill_remaining = {
        request.request_id: max(
            0,
            request.original_prefill_tokens
            - request.committed_prefill_tokens
            - request.reserved_prefill_tokens,
        )
        for request in active
        if request.committed_prefill_tokens + request.reserved_prefill_tokens
        < request.original_prefill_tokens
    }
    decode_progress = {
        request.request_id: request.committed_decode_tokens
        for request in active
        if request.committed_prefill_tokens >= request.original_prefill_tokens
    }
    rank_capacity = [
        value for replica in state.replicas for value in replica.rank_kv_capacity_blocks
    ]
    rank_committed = [
        value
        for replica in state.replicas
        for value in replica.rank_kv_committed_blocks
    ]
    rank_reserved = [
        value for replica in state.replicas for value in replica.rank_kv_reserved_blocks
    ]
    microbatch_ids = [
        batch.microbatch_id
        for replica in state.replicas
        for batch in replica.inflight_microbatches
    ]
    objective = state.objective
    return {
        "total_cost": objective.total_cost,
        "slo_violations": objective.slo_violations,
        "total_lateness": objective.prefill_lateness_sec
        + objective.decode_lateness_sec,
        "prefill_lateness": objective.prefill_lateness_sec,
        "decode_lateness": objective.decode_lateness_sec,
        "active_request_ids": _json([request.request_id for request in active]),
        "completed_request_ids": _json(completed),
        "stopped_request_ids": _json(stopped),
        "dropped_request_ids": _json(dropped),
        "decode_credits_available": state.decode_credits_available,
        "decode_credit_balance": state.decode_credits_available,
        "decode_credits_reserved": state.decode_credits_reserved,
        "decode_credits_minted_total": state.decode_credits_minted_total,
        "decode_tokens_committed_total": state.decode_tokens_committed_total,
        "decode_processed_tokens_by_id": _json(decode_progress),
        "prefill_remaining_by_id": _json(prefill_remaining),
        "kv_capacity_blocks_by_rank": _json(rank_capacity),
        "kv_committed_blocks_by_rank": _json(rank_committed),
        "kv_reserved_blocks_by_rank": _json(rank_reserved),
        "inflight_microbatch_ids": _json(microbatch_ids),
        "next_adversary_tick": state.next_adversary_tick,
    }


class EvaluationPipelineLogger:
    """GV3-style per-game traces plus compact cycle-level result tables."""

    STEP_FIELDS = (
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
        "controller_version",
        "adversary_version",
        "selection_mode",
        "total_cost",
        "slo_violations",
        "total_lateness",
        "prefill_lateness",
        "decode_lateness",
        "active_request_ids",
        "completed_request_ids",
        "stopped_request_ids",
        "dropped_request_ids",
        "decode_credits_available",
        "decode_credit_balance",
        "decode_credits_reserved",
        "decode_credits_minted_total",
        "decode_tokens_committed_total",
        "decode_processed_tokens_by_id",
        "prefill_remaining_by_id",
        "kv_capacity_blocks_by_rank",
        "kv_committed_blocks_by_rank",
        "kv_reserved_blocks_by_rank",
        "inflight_microbatch_ids",
        "next_adversary_tick",
        "valid_action_count",
        "canonical_action_count",
        "iterations_requested",
        "iterations_used",
        "chosen_q_value",
        "chosen_reward",
        "chosen_discount",
        "chosen_bootstrap",
        "chosen_child_cost",
        "model_value_at_state",
        "mcts_root_value",
        "candidate_ranking_mode",
        "candidate_top5_action_reprs",
        "candidate_top5_q_values",
        "candidate_top5_rewards",
        "candidate_top5_discounts",
        "candidate_top5_child_costs",
        "candidate_top5_visits",
        "candidate_top5_priors",
        "candidate_top5_mcts_probs",
        "policy_prior_temperature",
        "mcts_action_temperature",
        "mcts_action_sample_count",
        "end_reason",
    )
    GAME_FIELDS = (
        "game_id",
        "cycle_label",
        "scenario",
        "paired_seed",
        "controller_version",
        "adversary_version",
        "final_cost",
        "slo_violations",
        "prefill_lateness",
        "decode_lateness",
        "requests_generated",
        "requests_completed",
        "requests_stopped",
        "requests_dropped",
        "actions_applied",
        "final_time",
        "end_reason",
    )
    SJF_FIELDS = (
        "pair_index",
        "paired_seed",
        "model_cost",
        "sjf_cost",
        "model_improvement",
        "outcome",
        "model_actions",
        "sjf_actions",
    )

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.games_dir = self.output_dir / "arena_games"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()

    def _append(
        self,
        path: Path,
        fields: Sequence[str],
        row: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        clean = {name: row.get(name, "") for name in fields}
        with self._thread_lock, path.open("a+", newline="", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0, 2)
            writer = csv.DictWriter(stream, fieldnames=fields)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(clean)
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _game_path(self, game_id: int, cycle_label: str) -> Path:
        return self.games_dir / f"game_{int(game_id)}_{_safe_name(cycle_label)}.csv"

    def observer(
        self,
        *,
        game_id: int,
        cycle_label: str,
        controller_version: int,
        adversary_version: int,
        selection_mode: str,
        iterations_requested: int | None = None,
    ) -> Callable[[PlayedStep], None]:
        """Return the callback consumed directly by the backend-neutral runner."""

        def write(step: PlayedStep) -> None:
            search = step.search
            selected = None if search is None else search.stats_for(step.edge.action)
            ranked = []
            if search is not None:
                sign = 1.0 if search.root_player == "controller" else -1.0
                ranked = sorted(
                    search.action_stats,
                    key=lambda item: (
                        sign * item.mean_value,
                        item.visits,
                        -item.action.representative_raw_index,
                    ),
                    reverse=True,
                )[:5]
            total_visits = (
                sum(item.visits for item in search.action_stats) if search else 0
            )
            action = step.edge.action
            row = {
                "game_id": game_id,
                "cycle_label": cycle_label,
                "phase": step.phase,
                "turn": step.sequence,
                "depth": step.sequence,
                "player_acted": step.edge.player,
                "player_to_act_next": step.edge.next_player,
                "action_repr": _json(
                    {
                        "canonical_index": action.canonical_action_index,
                        "representative_raw_index": action.representative_raw_index,
                        "alias_indices": action.equivalent_raw_indices,
                        "transition_kind": step.edge.transition_kind,
                    }
                ),
                "sim_time_before": step.edge.started_at,
                "sim_time_after": step.edge.finished_at,
                "controller_version": controller_version,
                "adversary_version": adversary_version,
                "selection_mode": (
                    selection_mode if step.phase == "searched" else "forced"
                ),
                **_state_fields(step.state_after),
                "valid_action_count": sum(search.raw_valid_mask) if search else "",
                "canonical_action_count": len(search.action_stats) if search else "",
                "iterations_requested": (
                    ""
                    if search is None or iterations_requested is None
                    else iterations_requested
                ),
                "iterations_used": total_visits if search else "",
                "chosen_q_value": "" if selected is None else selected.mean_value,
                "chosen_reward": step.edge.reward,
                "chosen_discount": step.edge.discount,
                "chosen_bootstrap": "",
                "chosen_child_cost": step.edge.objective_after,
                "model_value_at_state": "",
                "mcts_root_value": "" if search is None else search.root_value,
                "candidate_ranking_mode": (
                    ""
                    if search is None
                    else (
                        "controller_max_q"
                        if search.root_player == "controller"
                        else "adversary_min_q"
                    )
                ),
                "candidate_top5_action_reprs": _json(
                    [item.action.representative_raw_index for item in ranked]
                ),
                "candidate_top5_q_values": _json([item.mean_value for item in ranked]),
                "candidate_top5_rewards": "",
                "candidate_top5_discounts": "",
                "candidate_top5_child_costs": "",
                "candidate_top5_visits": _json([item.visits for item in ranked]),
                "candidate_top5_priors": _json([item.prior for item in ranked]),
                "candidate_top5_mcts_probs": _json(
                    [
                        item.visits / total_visits if total_visits else 0.0
                        for item in ranked
                    ]
                ),
                "policy_prior_temperature": "",
                "mcts_action_temperature": "",
                "mcts_action_sample_count": "",
                "end_reason": "",
            }
            self._append(self._game_path(game_id, cycle_label), self.STEP_FIELDS, row)

        return write

    def _log_cycle_end(
        self,
        *,
        game_id: int,
        cycle_label: str,
        summary: StateSummary,
        end_reason: str,
        controller_version: int,
        adversary_version: int,
    ) -> None:
        self._append(
            self._game_path(game_id, cycle_label),
            self.STEP_FIELDS,
            {
                "game_id": game_id,
                "cycle_label": cycle_label,
                "phase": "arena_end",
                "controller_version": controller_version,
                "adversary_version": adversary_version,
                **_state_fields(summary),
                "end_reason": end_reason,
            },
        )

    def _game_row(
        self,
        result: Any,
        *,
        scenario: str,
        paired_seed: int,
        controller_version: int,
        adversary_version: int,
    ) -> dict[str, Any]:
        summary = result.final_state_summary
        objective = summary.objective
        return {
            "game_id": result.game_id,
            "cycle_label": result.cycle_label,
            "scenario": scenario,
            "paired_seed": paired_seed,
            "controller_version": controller_version,
            "adversary_version": adversary_version,
            "final_cost": result.final_objective,
            "slo_violations": objective.slo_violations,
            "prefill_lateness": objective.prefill_lateness_sec,
            "decode_lateness": objective.decode_lateness_sec,
            "requests_generated": objective.requests_generated,
            "requests_completed": objective.requests_completed,
            "requests_stopped": objective.requests_stopped,
            "requests_dropped": objective.requests_dropped,
            "actions_applied": result.actions_applied,
            "final_time": result.final_time,
            "end_reason": result.end_reason,
        }

    def log_game(
        self,
        result: GameCycleResult,
        *,
        scenario: str,
        paired_seed: int,
        controller_version: int,
        adversary_version: int,
    ) -> None:
        row = self._game_row(
            result,
            scenario=scenario,
            paired_seed=paired_seed,
            controller_version=controller_version,
            adversary_version=adversary_version,
        )
        self._append(self.output_dir / "arena_results.csv", self.GAME_FIELDS, row)
        self._log_cycle_end(
            game_id=result.game_id,
            cycle_label=result.cycle_label,
            summary=result.final_state_summary,
            end_reason=result.end_reason,
            controller_version=controller_version,
            adversary_version=adversary_version,
        )

    def log_baseline_game(self, result: Any, **context: Any) -> None:
        row = self._game_row(result, **context)
        self._append(self.output_dir / "arena_results.csv", self.GAME_FIELDS, row)
        self._log_cycle_end(
            game_id=result.game_id,
            cycle_label=result.cycle_label,
            summary=result.final_state_summary,
            end_reason=result.end_reason,
            controller_version=int(context["controller_version"]),
            adversary_version=int(context["adversary_version"]),
        )

    def log_sjf_comparison(self, result: Any) -> None:
        self._append(
            self.output_dir / "sjf_results.csv",
            self.SJF_FIELDS,
            asdict(result),
        )

    def log_training(self, result: Any) -> None:
        self._append_mapping("training_cycles.csv", result.summary())

    def log_arena(self, result: Any) -> None:
        summary = result.summary()
        summary.pop("games", None)
        self._append_mapping("arena_summary.csv", summary)

    def log_promotion(self, result: Any) -> None:
        self._append_mapping("promotion_results.csv", result.summary())

    def log_sjf_summary(self, result: Any) -> None:
        summary = result.summary()
        summary.pop("games", None)
        self._append_mapping("sjf_summary.csv", summary)

    def _append_mapping(self, filename: str, value: Mapping[str, Any]) -> None:
        flat = {
            key: _json(item) if isinstance(item, (dict, list, tuple)) else item
            for key, item in value.items()
        }
        self._append(self.output_dir / filename, tuple(flat), flat)
