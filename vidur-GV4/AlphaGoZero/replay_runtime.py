"""Runtime replay target recording for GV3 AlphaGoZero self-play."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from vidur.AlphaGoZero.markov_value_features import features_from_replay_row

DISCOUNT_FACTOR = 0.995
DISCOUNT_TIME_DENOM_SEC = 0.015725797204323228

FIELDS = [
    "game_id",
    "phase",
    "turn_number",
    "depth_number",
    "player",
    "canonical_action_count",
    "immediate_cost",
    "immediate_reward",
    "discount",
    "temperature",
    "value_by_model_at_state",
    "mcts_root_value",
    "target_value",
    "model_prior_top5",
    "target_mcts_distribution_top5",
    "time_at_state",
    "time_after_state",
    "feature_complete",
    "state_features_json",
    "value_feature_complete",
    "value_feature_schema",
    "value_global_features_json",
    "value_request_features_json",
    "value_launch_features_json",
    "value_request_count",
    "value_launch_count",
    "policy_row_count",
    "trajectory_start_time",
    "trajectory_terminal_time",
    "replay_sample_window_sec",
    "terminal_bootstrap_value",
    "target_backup_backend",
]

POLICY_FIELDS = [
    "game_id",
    "phase",
    "turn_number",
    "depth_number",
    "player",
    "canon_action_index",
    "action_repr",
    "visit_count",
    "mcts_visit_prob",
    "model_prior",
    "q_value",
    "immediate_reward",
    "discount",
    "bootstrap_value",
    "child_cost",
    "action_features_json",
]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _float_list(values: Any) -> list[float]:
    if values is None or values == "":
        return []
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            return []
    if not isinstance(values, (list, tuple)):
        return []
    out: list[float] = []
    for value in values:
        try:
            x = float(value)
            if math.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return out


def _json_float_list(values: Any, *, expected: int | None = None) -> str:
    vals = _float_list(values)
    if expected is not None and len(vals) != int(expected):
        return ""
    return json.dumps([float(x) for x in vals], separators=(",", ":"))


def _time_discount(t_child: float, t_parent: float, *, discount_factor: float = DISCOUNT_FACTOR) -> float:
    dt = max(0.0, float(t_child) - float(t_parent))
    return float(float(discount_factor) ** (dt / max(DISCOUNT_TIME_DENOM_SEC, 1e-9)))


def discounted_trajectory_targets(
    rewards: Sequence[float],
    discounts: Sequence[float],
    terminal_bootstrap_value: float,
) -> list[float]:
    """Back up a complete time-discounted trajectory from its terminal value."""

    if len(rewards) != len(discounts):
        raise ValueError("rewards and discounts must have equal length")
    running_value = float(terminal_bootstrap_value)
    if not math.isfinite(running_value):
        raise ValueError("terminal_bootstrap_value must be finite")
    targets = [0.0] * len(rewards)
    for index in range(len(rewards) - 1, -1, -1):
        reward = float(rewards[index])
        discount = float(discounts[index])
        if not math.isfinite(reward):
            raise ValueError(f"reward at index {index} must be finite")
        if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError(f"discount at index {index} must be finite and in [0, 1]")
        running_value = reward + discount * running_value
        targets[index] = running_value
    return targets


class AlphaGoZeroReplayRecorder:
    """Collect full trajectory transitions, then write filtered trainable rows.

    The state CSV stores one row per replay state. The companion policy CSV stores
    one row per canonical action for those replay states. Targets are computed on
    the full trajectory first, then states with too few canonical actions are
    filtered out.
    """

    def __init__(
        self,
        output_csv: str | Path,
        *,
        target_cycle_label: str = "model_adv_depth1_vs_model_ctrl_depth1",
        min_canonical_actions: int = 2,
        policy_rows_csv: str | Path | None = None,
        discount_factor: float = DISCOUNT_FACTOR,
        sample_window_sec: float = 0.0,
        target_backup_fn: Callable[
            [Sequence[float], Sequence[float], float],
            Sequence[float],
        ] | None = None,
        target_backup_backend: str | None = None,
    ) -> None:
        self.output_csv = Path(output_csv)
        self.policy_rows_csv = Path(policy_rows_csv) if policy_rows_csv else self.output_csv.with_name("replay_policy_rows.csv")
        self.target_cycle_label = str(target_cycle_label)
        self.min_canonical_actions = int(min_canonical_actions)
        self.discount_factor = float(discount_factor)
        if not (0.0 < self.discount_factor <= 1.0):
            raise ValueError("discount_factor must be in (0, 1]")
        self.sample_window_sec = float(sample_window_sec)
        if not math.isfinite(self.sample_window_sec) or self.sample_window_sec < 0.0:
            raise ValueError("sample_window_sec must be finite and nonnegative")
        self._target_backup_fn = target_backup_fn or discounted_trajectory_targets
        self.target_backup_backend = str(
            target_backup_backend
            or ("python" if target_backup_fn is None else "native")
        )
        self._transitions: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._last_total_cost: dict[tuple[int, str], float] = {}
        self._cycle_start_time: dict[tuple[int, str], float] = {}

    def record_transition(
        self,
        *,
        game_id: int,
        cycle_label: str,
        phase: str,
        turn: int,
        depth: int,
        player: str,
        sim_time_before: float,
        sim_time_after: float,
        total_cost_after: float,
        selection_info: dict[str, Any],
    ) -> None:
        if str(cycle_label) != self.target_cycle_label:
            return
        key = (int(game_id), str(cycle_label))
        self._cycle_start_time.setdefault(key, float(sim_time_before))
        previous_cost = float(self._last_total_cost.get(key, 0.0))
        total_cost = float(total_cost_after)
        immediate_cost = float(total_cost - previous_cost)
        self._last_total_cost[key] = total_cost

        chosen_reward = _optional_float(selection_info.get("chosen_reward"))
        immediate_reward = float(chosen_reward) if chosen_reward is not None else -immediate_cost
        chosen_discount = _optional_float(selection_info.get("chosen_discount"))
        discount = float(chosen_discount) if chosen_discount is not None else _time_discount(
            sim_time_after,
            sim_time_before,
            discount_factor=self.discount_factor,
        )
        valid_n = _int(selection_info.get("valid_action_count"), 0)
        canonical_n = _int(selection_info.get("canonical_action_count"), valid_n)
        mcts_probs = _float_list(selection_info.get("candidate_top5_mcts_probs"))
        if not mcts_probs:
            visits = _float_list(selection_info.get("candidate_top5_visits"))
            denom = max(1.0, _finite_float(selection_info.get("iterations_used"), sum(visits) or 1.0))
            mcts_probs = [float(v) / denom for v in visits]

        state_features = _float_list(selection_info.get("state_features"))
        policy_rows_raw = selection_info.get("policy_rows") or []
        policy_rows = list(policy_rows_raw) if isinstance(policy_rows_raw, list) else []
        feature_complete = bool(len(state_features) == 226 and policy_rows)
        value_replay_fields = {
            "value_feature_schema": str(selection_info.get("value_feature_schema", "")),
            "value_global_features_json": str(selection_info.get("value_global_features_json", "")),
            "value_request_features_json": str(selection_info.get("value_request_features_json", "")),
            "value_launch_features_json": str(selection_info.get("value_launch_features_json", "")),
            "value_request_count": _int(selection_info.get("value_request_count"), 0),
            "value_launch_count": _int(selection_info.get("value_launch_count"), 0),
        }
        try:
            features_from_replay_row(value_replay_fields)
            value_feature_complete = True
        except ValueError:
            value_feature_complete = False

        self._transitions.setdefault(key, []).append(
            {
                "game_id": int(game_id),
                "phase": str(phase),
                "turn_number": int(turn),
                "depth_number": int(depth),
                "player": str(player),
                "canonical_action_count": int(canonical_n),
                "immediate_cost": float(immediate_cost),
                "immediate_reward": float(immediate_reward),
                "discount": float(discount),
                "temperature": _finite_float(
                    selection_info.get("mcts_action_temperature", selection_info.get("policy_prior_temperature")),
                    1.0,
                ),
                "value_by_model_at_state": _finite_float(selection_info.get("model_value_at_state"), 0.0),
                "mcts_root_value": _finite_float(selection_info.get("mcts_root_value"), 0.0),
                "model_prior_top5": _float_list(selection_info.get("candidate_top5_priors")),
                "target_mcts_distribution_top5": mcts_probs,
                "time_at_state": float(sim_time_before),
                "time_after_state": float(sim_time_after),
                "feature_complete": bool(feature_complete),
                "state_features": state_features,
                "value_feature_complete": bool(value_feature_complete),
                **value_replay_fields,
                "policy_rows": policy_rows,
            }
        )

    def finish_cycle(
        self,
        *,
        game_id: int,
        cycle_label: str,
        terminal_bootstrap_value: float,
    ) -> int:
        if str(cycle_label) != self.target_cycle_label:
            return 0
        key = (int(game_id), str(cycle_label))
        transitions = self._transitions.pop(key, [])
        self._last_total_cost.pop(key, None)
        cycle_start_time = self._cycle_start_time.pop(key, None)
        if not transitions:
            return 0

        targets = list(
            self._target_backup_fn(
                [float(item["immediate_reward"]) for item in transitions],
                [float(item["discount"]) for item in transitions],
                float(terminal_bootstrap_value),
            )
        )
        if len(targets) != len(transitions):
            raise RuntimeError(
                "target backup returned "
                f"{len(targets)} targets for {len(transitions)} transitions"
            )

        trajectory_start_time = float(
            transitions[0]["time_at_state"]
            if cycle_start_time is None
            else cycle_start_time
        )
        trajectory_terminal_time = float(transitions[-1]["time_after_state"])
        sample_window_end = (
            trajectory_start_time + self.sample_window_sec
            if self.sample_window_sec > 0.0
            else math.inf
        )
        for item, target in zip(transitions, targets):
            target_value = float(target)
            if not math.isfinite(target_value):
                raise RuntimeError("target backup returned a non-finite value")
            item["target_value"] = target_value
            item["trajectory_start_time"] = trajectory_start_time
            item["trajectory_terminal_time"] = trajectory_terminal_time
            item["replay_sample_window_sec"] = self.sample_window_sec
            item["terminal_bootstrap_value"] = float(terminal_bootstrap_value)
            item["target_backup_backend"] = self.target_backup_backend

        rows = [
            item for item in transitions
            if int(item["canonical_action_count"]) > self.min_canonical_actions
            and float(item["time_at_state"]) <= sample_window_end + 1e-12
        ]
        self._append_rows(rows)
        return len(rows)

    def _append_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_csv.exists()
        write_policy_header = not self.policy_rows_csv.exists()
        with self.output_csv.open("a", newline="", encoding="utf-8") as f, self.policy_rows_csv.open("a", newline="", encoding="utf-8") as pf:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            policy_writer = csv.DictWriter(pf, fieldnames=POLICY_FIELDS)
            if write_header:
                writer.writeheader()
            if write_policy_header:
                policy_writer.writeheader()
            for row in rows:
                policy_rows = list(row.get("policy_rows") or []) if bool(row.get("feature_complete")) else []
                writer.writerow(
                    {
                        "game_id": int(row["game_id"]),
                        "phase": str(row["phase"]),
                        "turn_number": int(row["turn_number"]),
                        "depth_number": int(row["depth_number"]),
                        "player": str(row["player"]),
                        "canonical_action_count": int(row["canonical_action_count"]),
                        "immediate_cost": float(row["immediate_cost"]),
                        "immediate_reward": float(row["immediate_reward"]),
                        "discount": float(row["discount"]),
                        "temperature": float(row["temperature"]),
                        "value_by_model_at_state": float(row["value_by_model_at_state"]),
                        "mcts_root_value": float(row["mcts_root_value"]),
                        "target_value": float(row["target_value"]),
                        "model_prior_top5": json.dumps([float(x) for x in row["model_prior_top5"]]),
                        "target_mcts_distribution_top5": json.dumps([float(x) for x in row["target_mcts_distribution_top5"]]),
                        "time_at_state": float(row["time_at_state"]),
                        "time_after_state": float(row["time_after_state"]),
                        "feature_complete": 1 if bool(row.get("feature_complete")) else 0,
                        "state_features_json": _json_float_list(row.get("state_features"), expected=226),
                        "value_feature_complete": 1 if bool(row.get("value_feature_complete")) else 0,
                        "value_feature_schema": str(row.get("value_feature_schema", "")),
                        "value_global_features_json": str(row.get("value_global_features_json", "")),
                        "value_request_features_json": str(row.get("value_request_features_json", "")),
                        "value_launch_features_json": str(row.get("value_launch_features_json", "")),
                        "value_request_count": int(row.get("value_request_count", 0) or 0),
                        "value_launch_count": int(row.get("value_launch_count", 0) or 0),
                        "policy_row_count": int(len(policy_rows)),
                        "trajectory_start_time": float(row["trajectory_start_time"]),
                        "trajectory_terminal_time": float(row["trajectory_terminal_time"]),
                        "replay_sample_window_sec": float(row["replay_sample_window_sec"]),
                        "terminal_bootstrap_value": float(row["terminal_bootstrap_value"]),
                        "target_backup_backend": str(row["target_backup_backend"]),
                    }
                )
                for prow in policy_rows:
                    policy_writer.writerow(
                        {
                            "game_id": int(row["game_id"]),
                            "phase": str(row["phase"]),
                            "turn_number": int(row["turn_number"]),
                            "depth_number": int(row["depth_number"]),
                            "player": str(row["player"]),
                            "canon_action_index": int(prow.get("canon_action_index", -1)),
                            "action_repr": str(prow.get("action_repr", "")),
                            "visit_count": int(prow.get("visit_count", 0) or 0),
                            "mcts_visit_prob": float(prow.get("mcts_visit_prob", 0.0) or 0.0),
                            "model_prior": float(prow.get("model_prior", 0.0) or 0.0),
                            "q_value": float(prow.get("q_value", 0.0) or 0.0),
                            "immediate_reward": float(prow.get("immediate_reward", 0.0) or 0.0),
                            "discount": float(prow.get("discount", 1.0) or 1.0),
                            "bootstrap_value": float(prow.get("bootstrap_value", 0.0) or 0.0),
                            "child_cost": float(prow.get("child_cost", 0.0) or 0.0),
                            "action_features_json": _json_float_list(prow.get("action_features")),
                        }
                    )
