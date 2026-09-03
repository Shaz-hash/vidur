"""Diagnose late-game controller Q preference in AlphaGoZero arenas and replay.

This is deliberately read-only.  It does not depend on the exported native
``bootstrap`` field because full-tree search currently logs backed-up Q there.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


def _float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_list(value: str) -> list[Any]:
    try:
        out = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return out if isinstance(out, list) else []


def _json_dict(value: str) -> dict[str, Any]:
    try:
        out = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return out if isinstance(out, dict) else {}


def _action_family(action_repr: str) -> str:
    text = str(action_repr)
    prefill = "prefill_allocations={}" not in text and "prefill_allocations={" in text
    decode = "decode_allocations={}" not in text and "decode_allocations={" in text
    if prefill and decode:
        return "mixed"
    if prefill:
        return "prefill"
    if decode:
        return "decode_only"
    return "other"


def _prefill_family(family: str) -> bool:
    return family in {"prefill", "mixed"}


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _load_game_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _candidate_rows(row: dict[str, str]) -> list[dict[str, Any]]:
    actions = _json_list(row.get("candidate_top5_action_reprs", ""))
    q_values = _json_list(row.get("candidate_top5_q_values", ""))
    rewards = _json_list(row.get("candidate_top5_rewards", ""))
    visits = _json_list(row.get("candidate_top5_visits", ""))
    priors = _json_list(row.get("candidate_top5_priors", ""))
    count = min(len(actions), len(q_values), len(rewards), len(visits), len(priors))
    out = []
    for index in range(count):
        q_value = _float(q_values[index])
        if not math.isfinite(q_value):
            continue
        out.append(
            {
                "family": _action_family(str(actions[index])),
                "q": q_value,
                "reward": _float(rewards[index], 0.0),
                "visits": max(0, _int(visits[index])),
                "prior": max(0.0, _float(priors[index], 0.0)),
            }
        )
    return out


def _summarize_arena(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"states": 0}
    q_gaps = [float(r["q_gap_decode_minus_prefill"]) for r in records]
    reward_gaps = [float(r["reward_gap_decode_minus_prefill"]) for r in records]
    visit_mass = [float(r["prefill_visit_mass"]) for r in records]
    prior_mass = [float(r["prefill_prior_mass"]) for r in records]
    q_decode = sum(float(r["q_gap_decode_minus_prefill"]) > 1e-9 for r in records)
    q_prefill = sum(float(r["q_gap_decode_minus_prefill"]) < -1e-9 for r in records)
    q_near_tie = sum(abs(float(r["q_gap_decode_minus_prefill"])) <= 0.02 for r in records)
    immediate_decode = sum(float(r["reward_gap_decode_minus_prefill"]) > 1e-9 for r in records)
    value_decode = sum(
        float(r["q_gap_decode_minus_prefill"]) > 1e-9
        and float(r["reward_gap_decode_minus_prefill"]) <= 1e-9
        for r in records
    )
    q_prefill_chosen_decode = sum(
        float(r["q_gap_decode_minus_prefill"]) < -1e-9 and r["chosen_family"] == "decode_only"
        for r in records
    )
    return {
        "states": len(records),
        "q_prefers_prefill_rate": _rate(q_prefill, len(records)),
        "q_prefers_decode_rate": _rate(q_decode, len(records)),
        "q_near_tie_abs_le_0.02_rate": _rate(q_near_tie, len(records)),
        "median_q_gap_decode_minus_prefill": statistics.median(q_gaps),
        "p25_q_gap": _quantile(q_gaps, 0.25),
        "p75_q_gap": _quantile(q_gaps, 0.75),
        "decode_has_better_immediate_reward_rate": _rate(immediate_decode, len(records)),
        "q_decode_without_immediate_advantage_rate": _rate(value_decode, len(records)),
        "median_reward_gap_decode_minus_prefill": statistics.median(reward_gaps),
        "chosen_decode_rate": _rate(sum(r["chosen_family"] == "decode_only" for r in records), len(records)),
        "top_visit_prefill_rate": _rate(sum(bool(r["top_visit_prefill"]) for r in records), len(records)),
        "top_prior_prefill_rate": _rate(sum(bool(r["top_prior_prefill"]) for r in records), len(records)),
        "mean_prefill_visit_mass_top5": statistics.fmean(visit_mass),
        "mean_prefill_prior_mass_top5": statistics.fmean(prior_mass),
        "q_prefill_but_chosen_decode_rate": _rate(q_prefill_chosen_decode, len(records)),
        "mean_top5_prefill_action_count": statistics.fmean(float(r["top5_prefill_actions"]) for r in records),
        "mean_top5_decode_action_count": statistics.fmean(float(r["top5_decode_actions"]) for r in records),
        "mean_canonical_action_count": statistics.fmean(float(r["canonical_action_count"]) for r in records),
    }


def analyze_arena(eval_dir: Path) -> dict[str, Any]:
    arena_dir = eval_dir / "promoted_adversary_vs_candidate_controller" / "arena_games"
    game_files = sorted(arena_dir.glob("*model_ctrl_depth1.csv"))
    records: list[dict[str, Any]] = []

    for game_file in game_files:
        rows = _load_game_rows(game_file)
        controller_rows = [
            row
            for row in rows
            if row.get("player_acted") == "controller"
            and _int(row.get("canonical_action_count")) > 1
        ]
        if not controller_rows:
            continue
        max_time = max(_float(row.get("sim_time_before"), 0.0) for row in controller_rows)
        max_time = max(max_time, 1e-12)
        for row in controller_rows:
            pending_prefill = any(
                _float(value, 0.0) > 0.0
                for value in _json_dict(row.get("prefill_remaining_by_id", "")).values()
            )
            if not pending_prefill:
                continue
            candidates = _candidate_rows(row)
            prefill = [candidate for candidate in candidates if _prefill_family(candidate["family"])]
            decode = [candidate for candidate in candidates if candidate["family"] == "decode_only"]
            if not prefill or not decode:
                continue

            best_prefill_q = max(float(candidate["q"]) for candidate in prefill)
            best_decode_q = max(float(candidate["q"]) for candidate in decode)
            best_prefill_reward = max(float(candidate["reward"]) for candidate in prefill)
            best_decode_reward = max(float(candidate["reward"]) for candidate in decode)
            visits_total = sum(int(candidate["visits"]) for candidate in candidates)
            priors_total = sum(float(candidate["prior"]) for candidate in candidates)
            top_visit = max(candidates, key=lambda candidate: int(candidate["visits"]))
            top_prior = max(candidates, key=lambda candidate: float(candidate["prior"]))
            sim_time = _float(row.get("sim_time_before"), 0.0)
            records.append(
                {
                    "game_id": row.get("game_id", ""),
                    "turn": _int(row.get("turn")),
                    "sim_time": sim_time,
                    "game_half": "second" if sim_time >= 0.5 * max_time else "first",
                    "sampling_phase": "after_20" if _int(row.get("mcts_action_sample_count")) >= 20 else "first_20",
                    "q_gap_decode_minus_prefill": best_decode_q - best_prefill_q,
                    "reward_gap_decode_minus_prefill": best_decode_reward - best_prefill_reward,
                    "chosen_family": _action_family(row.get("action_repr", "")),
                    "top_visit_prefill": _prefill_family(top_visit["family"]),
                    "top_prior_prefill": _prefill_family(top_prior["family"]),
                    "prefill_visit_mass": (
                        sum(int(candidate["visits"]) for candidate in prefill) / visits_total
                        if visits_total else 0.0
                    ),
                    "prefill_prior_mass": (
                        sum(float(candidate["prior"]) for candidate in prefill) / priors_total
                        if priors_total else 0.0
                    ),
                    "top5_prefill_actions": len(prefill),
                    "top5_decode_actions": len(decode),
                    "canonical_action_count": _int(row.get("canonical_action_count")),
                }
            )

    output = {"game_files": len(game_files), "eligible_states": len(records)}
    output["all"] = _summarize_arena(records)
    for field, values in (
        ("game_half", ("first", "second")),
        ("sampling_phase", ("first_20", "after_20")),
    ):
        output[field] = {
            value: _summarize_arena([record for record in records if record[field] == value])
            for value in values
        }
    return output


def _sample_cache_rows(
    replay_dir: Path,
    *,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_files = sorted(replay_dir.glob("partitions/*/*__controller/*.state_cache_v1.npz"))
    random.Random(seed).shuffle(cache_files)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    turn_parts: list[np.ndarray] = []
    remaining = int(max_rows)

    for cache_path in cache_files:
        if remaining <= 0:
            break
        with np.load(cache_path, allow_pickle=False) as cache:
            features = np.asarray(cache["features"], dtype=np.float32)
            targets = np.asarray(cache["targets"], dtype=np.float32)
            keys = np.asarray(cache["keys"])
            players = np.asarray(cache["players"])
        indices = np.flatnonzero(players == 0)
        if indices.size > remaining:
            local_rng = np.random.default_rng(seed + len(feature_parts))
            indices = np.sort(local_rng.choice(indices, size=remaining, replace=False))
        if indices.size == 0:
            continue
        selected_keys = keys[indices]
        turns = np.fromiter(
            (_int(str(key).split("\x1f")[3], -1) for key in selected_keys),
            dtype=np.int32,
            count=indices.size,
        )
        feature_parts.append(features[indices])
        target_parts.append(targets[indices])
        turn_parts.append(turns)
        remaining -= int(indices.size)

    if not feature_parts:
        return (
            np.empty((0, 226), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(target_parts, axis=0),
        np.concatenate(turn_parts, axis=0),
    )


def _prediction_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if targets.size == 0:
        return {"states": 0}
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    return {
        "states": int(targets.size),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "prediction_mean": float(np.mean(predictions)),
        "bias_prediction_minus_target": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "p95_abs_error": float(np.quantile(np.abs(errors), 0.95)),
        "correlation": (
            float(np.corrcoef(predictions, targets)[0, 1])
            if targets.size > 1 and np.std(predictions) > 0 and np.std(targets) > 0
            else math.nan
        ),
    }


def analyze_value_models(
    experiment_root: Path,
    *,
    promoted_version: int,
    candidate_version: int,
    max_rows: int,
    seed: int,
) -> dict[str, Any]:
    features, targets, turns = _sample_cache_rows(
        experiment_root / "global_replay",
        max_rows=max_rows,
        seed=seed,
    )
    pending_both = (features[:, 0] > 0.0) & (features[:, 1] > 0.0)
    phase_masks = {
        "all": np.ones(targets.shape, dtype=bool),
        "turn_le_20": turns <= 20,
        "turn_gt_20": turns > 20,
        "pending_prefill_and_decode": pending_both,
        "pending_both_turn_le_20": pending_both & (turns <= 20),
        "pending_both_turn_gt_20": pending_both & (turns > 20),
        "turn_21_40": (turns > 20) & (turns <= 40),
        "turn_gt_40": turns > 40,
    }
    output: dict[str, Any] = {
        "sampled_states": int(targets.size),
        "pending_prefill_and_decode_rate": float(np.mean(pending_both)) if targets.size else math.nan,
        "turn_gt_20_rate": float(np.mean(turns > 20)) if targets.size else math.nan,
    }
    for label, version in (("promoted", promoted_version), ("candidate", candidate_version)):
        model_path = (
            experiment_root
            / "models"
            / f"Model_Version{version}"
            / "controller_value"
            / "dnn_value_residual_192_v1"
            / "model.joblib"
        )
        model = joblib.load(model_path)
        predictions = np.asarray(model.predict(features), dtype=np.float32)
        output[label] = {
            "version": int(version),
            "model_path": str(model_path),
            "phases": {
                phase: _prediction_metrics(predictions[mask], targets[mask])
                for phase, mask in phase_masks.items()
            },
        }
    return output


def _print_section(title: str, payload: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--promoted-controller-version", type=int, required=True)
    parser.add_argument("--max-replay-rows", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    experiment_root = args.experiment_root.expanduser().resolve()
    eval_dir = experiment_root / "eval_of_models" / f"eval_{args.candidate_version:06d}"
    _print_section("arena", analyze_arena(eval_dir))
    _print_section(
        "value_models",
        analyze_value_models(
            experiment_root,
            promoted_version=args.promoted_controller_version,
            candidate_version=args.candidate_version,
            max_rows=args.max_replay_rows,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
