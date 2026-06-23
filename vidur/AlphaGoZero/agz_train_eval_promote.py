"""XL-side AlphaGoZero HGB train/eval/promotion driver.

This script is intentionally conservative: it trains only from replay rows marked
feature_complete=1 and writes candidate artifacts/metrics atomically. Evaluation
and promotion orchestration is isolated here so the ingest loop can launch it as
an asynchronous subprocess.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from vidur.AlphaGoZero.config import (
    ADVERSARY_POLICY_SAMPLE_CAP,
    XL_CONTROLLER_POLICY_SAMPLE_CAP,
    DEFAULT_ADVERSARY_PRIOR_MODEL_PATH,
    DEFAULT_CONTROLLER_PRIOR_MODEL_PATH,
    DEFAULT_VALUE_MODEL_PATH,
    MIN_ADVERSARY_STATES_FOR_EVAL,
    PROMOTION_WIN_RATE_THRESHOLD,
)
from vidur.AlphaGoZero.cluster import REMOTE_OUTPUT_ROOT, REMOTE_REPO, WORKERS
from vidur.AlphaGoZero.durable_transfer import append_csv_row, atomic_write_json, local_time_24h, utc_now
from vidur.bellman_v4_adv.arena_mcts_value_runnerCPP import _export_hgb_to_native_text

CONFIG_NAME = "hgb_sq_63leaf_1050iter_a2"
POLICY_CONFIG_NAME = "hgb_policy_63leaf_1050iter"
VALUE_FEATURE_DIM = 226
CTRL_ACTION_DIM = 43
ADV_ACTION_DIM = 7
POLICY_ALPHA = 1.0

TRAIN_MODEL_FIELDS = [
    "model_config", "model_version", "time_of_training_24h", "sampled_controller_states", "sampled_adversary_states",
    "value_mse", "value_rmse", "value_max_abs_error", "value_p95_abs_error",
    "controller_policy_mse", "controller_policy_cross_entropy", "controller_policy_top1", "controller_policy_top3",
    "adversary_policy_mse", "adversary_policy_cross_entropy", "adversary_policy_top1", "adversary_policy_top3",
]


EVAL_GAME_FIELDS = [
    "promoted_model_version",
    "new_promoted_model_version",
    "candidate_win_ratio_in_100_games",
    "candidate_promoted",
    "time_of_eval",
    "game_id",
    "game_hop_number",
    "total_cost_when_promoted_is_adv_and_new_is_controller",
    "total_cost_when_new_is_adv_and_promoted_is_controller",
]


@dataclass(frozen=True)
class ModelBundle:
    model_version: int
    value_model_path: Path
    controller_prior_model_path: Path
    adversary_prior_model_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "model_version": int(self.model_version),
            "value_model_path": str(self.value_model_path),
            "controller_prior_model_path": str(self.controller_prior_model_path),
            "adversary_prior_model_path": str(self.adversary_prior_model_path),
            "updated_at_utc": utc_now(),
        }


def _read_json_list(raw: str) -> list[float]:
    if not raw:
        return []
    try:
        vals = json.loads(raw)
    except Exception:
        return []
    if not isinstance(vals, list):
        return []
    out: list[float] = []
    for v in vals:
        try:
            x = float(v)
            if math.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return out


def _root_key(row: dict[str, str]) -> tuple[str, str, int, int, int, str]:
    return (
        str(row.get("accepted_shard_id", "")),
        str(row.get("source_worker_id", "")),
        int(float(row.get("game_id", 0) or 0)),
        int(float(row.get("turn_number", 0) or 0)),
        int(float(row.get("depth_number", 0) or 0)),
        str(row.get("player", "")),
    )


def _load_feature_state_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True"}:
                continue
            features = _read_json_list(row.get("state_features_json", ""))
            if len(features) != VALUE_FEATURE_DIM:
                continue
            item = dict(row)
            item["_key"] = _root_key(row)
            item["_state_features"] = features
            item["_target_value"] = float(row.get("target_value", 0.0) or 0.0)
            rows.append(item)
    return rows


def _load_policy_rows(path: Path) -> dict[tuple[str, str, int, int, int, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str, int, int, int, str], list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            player = str(row.get("player", ""))
            feats = _read_json_list(row.get("action_features_json", ""))
            if player == "controller" and len(feats) != CTRL_ACTION_DIM:
                continue
            if player == "adversary" and len(feats) != ADV_ACTION_DIM:
                continue
            item = dict(row)
            item["_key"] = _root_key(row)
            item["_action_features"] = feats
            item["_visit_count"] = int(float(row.get("visit_count", 0) or 0))
            item["_mcts_visit_prob"] = float(row.get("mcts_visit_prob", 0.0) or 0.0)
            out.setdefault(item["_key"], []).append(item)
    for rows in out.values():
        rows.sort(key=lambda r: int(float(r.get("canon_action_index", -1) or -1)))
    return out


def _feature_replay_paths(root: Path) -> list[Path]:
    root = Path(root)
    partitioned = sorted(
        (root / "global_replay" / "partitions").glob("*/*/replay_target_runtime_feature_complete.csv")
    )
    if partitioned:
        return partitioned
    legacy = root / "global_replay" / "replay_target_runtime_feature_complete.csv"
    return [legacy] if legacy.exists() else []


def _policy_replay_paths(root: Path) -> list[Path]:
    root = Path(root)
    partitioned = sorted((root / "global_replay" / "partitions").glob("*/*/replay_policy_rows.csv"))
    if partitioned:
        return partitioned
    legacy = root / "global_replay" / "replay_policy_rows.csv"
    return [legacy] if legacy.exists() else []


def _reservoir_add(
    sample: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    seen: int,
    max_rows: int,
    rng: random.Random,
) -> None:
    if max_rows <= 0:
        return
    if len(sample) < int(max_rows):
        sample.append(item)
        return
    j = rng.randrange(int(seen))
    if j < int(max_rows):
        sample[j] = item


def _minimal_state_row(row: dict[str, str]) -> dict[str, Any] | None:
    if str(row.get("feature_complete", "0")).strip() not in {"1", "true", "True", ""}:
        return None
    features = _read_json_list(row.get("state_features_json", ""))
    if len(features) != VALUE_FEATURE_DIM:
        return None
    try:
        target = float(row.get("target_value", 0.0) or 0.0)
    except Exception:
        target = 0.0
    player = str(row.get("player", row.get("root_player", ""))).lower()
    return {
        "_key": _root_key(row),
        "_state_features": np.asarray(features, dtype=np.float32),
        "_target_value": float(target),
        "player": player,
    }


def _stream_sample_state_rows(
    paths: list[Path],
    *,
    seed: int,
    max_value_rows: int,
    max_controller_policy_roots: int,
    max_adversary_policy_roots: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, int, int, str], np.ndarray], dict[tuple[str, str, int, int, int, str], np.ndarray], dict[str, int]]:
    rng_value = random.Random(int(seed) + 101)
    rng_ctrl = random.Random(int(seed) + 102)
    rng_adv = random.Random(int(seed) + 103)
    value_sample: list[dict[str, Any]] = []
    ctrl_sample: list[dict[str, Any]] = []
    adv_sample: list[dict[str, Any]] = []
    counts = {"states": 0, "controller": 0, "adversary": 0}
    seen_value = 0
    seen_ctrl = 0
    seen_adv = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item = _minimal_state_row(row)
                if item is None:
                    continue
                counts["states"] += 1
                player = str(item["player"])
                if player == "controller":
                    counts["controller"] += 1
                elif player == "adversary":
                    counts["adversary"] += 1
                seen_value += 1
                _reservoir_add(
                    value_sample,
                    item,
                    seen=seen_value,
                    max_rows=int(max_value_rows),
                    rng=rng_value,
                )
                if player == "controller":
                    seen_ctrl += 1
                    _reservoir_add(
                        ctrl_sample,
                        item,
                        seen=seen_ctrl,
                        max_rows=int(max_controller_policy_roots),
                        rng=rng_ctrl,
                    )
                elif player == "adversary":
                    seen_adv += 1
                    _reservoir_add(
                        adv_sample,
                        item,
                        seen=seen_adv,
                        max_rows=int(max_adversary_policy_roots),
                        rng=rng_adv,
                    )
    ctrl_roots = {r["_key"]: r["_state_features"] for r in ctrl_sample}
    adv_roots = {r["_key"]: r["_state_features"] for r in adv_sample}
    return value_sample, ctrl_roots, adv_roots, counts


def _collect_policy_training_arrays(
    paths: list[Path],
    roots: dict[tuple[str, str, int, int, int, str], np.ndarray],
    *,
    player: str,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if not roots:
        return (
            np.empty((0, VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    actions_by_key: dict[tuple[str, str, int, int, int, str], list[tuple[int, np.ndarray, int]]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("player", "")).lower() != str(player):
                    continue
                key = _root_key(row)
                if key not in roots:
                    continue
                feats = _read_json_list(row.get("action_features_json", ""))
                if len(feats) != int(action_dim):
                    continue
                try:
                    idx = int(float(row.get("canon_action_index", -1) or -1))
                except Exception:
                    idx = -1
                try:
                    visits = int(float(row.get("visit_count", 0) or 0))
                except Exception:
                    visits = 0
                actions_by_key.setdefault(key, []).append((idx, np.asarray(feats, dtype=np.float32), visits))

    total_actions = sum(len(v) for v in actions_by_key.values())
    if total_actions <= 0:
        return (
            np.empty((0, VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    X = np.empty((int(total_actions), VALUE_FEATURE_DIM + int(action_dim)), dtype=np.float32)
    y = np.empty(int(total_actions), dtype=np.float32)
    probs_out = np.empty(int(total_actions), dtype=np.float32)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for key in sorted(actions_by_key.keys()):
        actions = sorted(actions_by_key[key], key=lambda x: x[0])
        visits = np.asarray([max(0, int(a[2])) for a in actions], dtype=np.float64)
        total = float(np.sum(visits))
        probs = visits / total if total > 0.0 else np.full(visits.size, 1.0 / float(visits.size), dtype=np.float64)
        logits = np.log(visits + float(POLICY_ALPHA))
        logits = logits - float(np.mean(logits))
        start = pos
        sf = roots[key]
        for action, logit, prob in zip(actions, logits, probs):
            X[pos, :VALUE_FEATURE_DIM] = sf
            X[pos, VALUE_FEATURE_DIM:] = action[1]
            y[pos] = float(logit)
            probs_out[pos] = float(prob)
            pos += 1
        offsets.append((start, pos))
    return X[:pos], y[:pos], probs_out[:pos], offsets


def _value_arrays_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty((0, VALUE_FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.float32)
    X = np.empty((len(rows), VALUE_FEATURE_DIM), dtype=np.float32)
    y = np.empty(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        X[i] = row["_state_features"]
        y[i] = float(row["_target_value"])
    return X, y


def _sample_rows(rows: list[dict[str, Any]], *, seed: int, max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= int(max_rows):
        return list(rows)
    rng = random.Random(int(seed))
    return sorted(rng.sample(rows, int(max_rows)), key=lambda r: r["_key"])


def _value_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred.astype(np.float64) - y.astype(np.float64)
    abs_err = np.abs(err)
    mse = float(np.mean(err * err)) if err.size else 0.0
    return {
        "mse": mse,
        "rmse": float(math.sqrt(max(0.0, mse))),
        "max_abs_error": float(np.max(abs_err)) if abs_err.size else 0.0,
        "p95_abs_error": float(np.percentile(abs_err, 95)) if abs_err.size else 0.0,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    z = x.astype(np.float64) - float(np.max(x))
    e = np.exp(z)
    s = float(np.sum(e))
    if not math.isfinite(s) or s <= 0.0:
        return np.full(x.size, 1.0 / float(x.size), dtype=np.float64)
    return e / s


def _policy_arrays(
    state_rows: list[dict[str, Any]],
    policy_by_key: dict[tuple[str, str, int, int, int, str], list[dict[str, Any]]],
    *,
    player: str,
    sample_roots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    roots = [r for r in state_rows if str(r.get("player")) == player and r["_key"] in policy_by_key]
    roots = _sample_rows(roots, seed=seed, max_rows=int(sample_roots))
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    p_rows: list[float] = []
    offsets: list[tuple[int, int]] = []
    for root in roots:
        actions = policy_by_key.get(root["_key"], [])
        if not actions:
            continue
        visits = np.asarray([max(0, int(a["_visit_count"])) for a in actions], dtype=np.float64)
        if visits.size == 0:
            continue
        total = float(np.sum(visits))
        probs = visits / total if total > 0.0 else np.full(visits.size, 1.0 / float(visits.size), dtype=np.float64)
        logs = np.log(visits + float(POLICY_ALPHA))
        centered = logs - float(np.mean(logs))
        start = len(X_rows)
        sf = list(root["_state_features"])
        for action, logit, prob in zip(actions, centered, probs):
            X_rows.append(sf + list(action["_action_features"]))
            y_rows.append(float(logit))
            p_rows.append(float(prob))
        offsets.append((start, len(X_rows)))
    if not X_rows:
        return np.empty((0, VALUE_FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), []
    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(p_rows, dtype=np.float32),
        offsets,
    )


def _policy_metrics(model: Any, X: np.ndarray, y: np.ndarray, probs: np.ndarray, offsets: list[tuple[int, int]]) -> dict[str, float]:
    if X.shape[0] == 0:
        return {"mse": 0.0, "cross_entropy": 0.0, "top1": 0.0, "top3": 0.0}
    pred = model.predict(X).astype(np.float64)
    mse = float(np.mean((pred - y.astype(np.float64)) ** 2))
    ce_sum = 0.0
    top1 = 0
    top3 = 0
    roots = 0
    eps = 1e-12
    for s, e in offsets:
        if e <= s:
            continue
        p = probs[s:e].astype(np.float64)
        if p.sum() <= 0.0:
            p = np.full(e - s, 1.0 / float(e - s), dtype=np.float64)
        else:
            p = p / p.sum()
        q = _softmax(pred[s:e])
        best = int(np.argmax(p))
        order = np.argsort(-pred[s:e], kind="stable")
        top1 += int(order[0] == best)
        top3 += int(best in set(int(x) for x in order[: min(3, order.size)]))
        ce_sum += -float(np.sum(p * np.log(np.maximum(q, eps))))
        roots += 1
    denom = float(max(1, roots))
    return {"mse": mse, "cross_entropy": float(ce_sum / denom), "top1": float(top1 / denom), "top3": float(top3 / denom)}


def _fit_value(X: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    est = HistGradientBoostingRegressor(
        loss="squared_error",
        max_leaf_nodes=63,
        max_iter=1050,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=int(seed),
        early_stopping=False,
    )
    weights = (1.0 + 2.0 * np.abs(y)).astype(np.float32)
    est.fit(X, y, sample_weight=weights)
    return est


def _fit_policy(X: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    est = HistGradientBoostingRegressor(
        loss="squared_error",
        max_leaf_nodes=63,
        max_iter=1050,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=int(seed),
        early_stopping=False,
    )
    est.fit(X, y)
    return est


def _next_candidate_version(root: Path) -> int:
    state_path = root / "xl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    versions = [int(state.get("promoted_model_version", 100) or 100)]
    models_dir = Path(root) / "models"
    for path in models_dir.glob("Model_Version*") if models_dir.exists() else []:
        suffix = path.name.removeprefix("Model_Version")
        if suffix.isdigit():
            versions.append(int(suffix))
    return max(versions) + 1


def _mark_training_complete(root: Path, version: int) -> None:
    state_path = root / "xl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state["candidate_counter"] = int(state.get("candidate_counter", 0)) + 1
    state["last_candidate_model_version"] = int(version)
    state["last_model_fit_completed_at_utc"] = utc_now()
    atomic_write_json(state_path, state)


def _mark_training_cycle_complete(root: Path, version: int) -> None:
    state_path = root / "xl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state["new_states_since_last_training"] = 0
    state["last_training_completed_at_utc"] = utc_now()
    state["last_training_cycle_completed_model_version"] = int(version)
    atomic_write_json(state_path, state)


def _mark_current_training_finalized(root: Path, version: int) -> None:
    path = root / "training" / "current_training.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data["cycle_finalized_at_utc"] = utc_now()
    data["cycle_finalized_model_version"] = int(version)
    atomic_write_json(path, data)


def _load_state(root: Path) -> dict[str, Any]:
    path = Path(root) / "xl_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_write_json(Path(root) / "xl_state.json", state)


def _default_promoted_bundle() -> ModelBundle:
    return ModelBundle(
        model_version=100,
        value_model_path=Path(DEFAULT_VALUE_MODEL_PATH),
        controller_prior_model_path=Path(DEFAULT_CONTROLLER_PRIOR_MODEL_PATH),
        adversary_prior_model_path=Path(DEFAULT_ADVERSARY_PRIOR_MODEL_PATH),
    )


def _current_promoted_bundle(root: Path) -> ModelBundle:
    current = Path(root) / "models" / "current_model.json"
    if not current.exists():
        return _default_promoted_bundle()
    data = json.loads(current.read_text(encoding="utf-8"))
    return ModelBundle(
        model_version=int(data.get("model_version", 100)),
        value_model_path=Path(data["value_model_path"]),
        controller_prior_model_path=Path(data["controller_prior_model_path"]),
        adversary_prior_model_path=Path(data["adversary_prior_model_path"]),
    )


def _candidate_bundle_from_output(version: int, out: Path) -> ModelBundle:
    return ModelBundle(
        model_version=int(version),
        value_model_path=out / "value" / CONFIG_NAME / "model.joblib",
        controller_prior_model_path=out / "controller_prior" / POLICY_CONFIG_NAME / "model.joblib",
        adversary_prior_model_path=out / "adversary_prior" / POLICY_CONFIG_NAME / "model.joblib",
    )


def _run(cmd: list[str], *, cwd: Path | None = None, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=log, stderr=subprocess.STDOUT, check=True, text=True)


def _arena_cmd(
    *,
    output_dir: Path,
    model_path: Path,
    model_version: int,
    controller_prior_path: Path,
    adversary_prior_path: Path,
    num_games: int,
    parallel_games: int,
    game_id_start: int,
    iterations: int,
    history_seed: int,
    history_hops_min: int,
    history_hops_max: int,
    only_model_ctrl_cycle: bool,
    write_arena_game_logs: bool = True,
    role_controller: ModelBundle | None = None,
    role_adversary: ModelBundle | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP",
        "--model-path",
        str(model_path),
        "--model-version",
        str(int(model_version)),
        "--feature-dim",
        "226",
        "--output-dir",
        str(output_dir),
        "--game-id-start",
        str(int(game_id_start)),
        "--num-games",
        str(int(num_games)),
        "--num-parallel-games",
        str(int(parallel_games)),
        "--shared-root-mcts-iterations",
        str(int(iterations)),
        "--worker-threads",
        "1",
        "--trivial-budget-tokens",
        "256",
        "--arena-time-limit-sec",
        "5.0",
        "--history-hops-min",
        str(int(history_hops_min)),
        "--history-hops-max",
        str(int(history_hops_max)),
        "--history-seed",
        str(int(history_seed)),
        "--history-hops-unique",
        "--no-history-hops-force-zero",
        "--seed",
        str(int(history_seed) + int(game_id_start)),
        "--puct-c",
        "1.0",
        "--policy-prior-temperature",
        "1.0",
        "--no-root-dirichlet-noise-enabled",
        "--no-agz-sample-initial-moves",
        "--controller-prior-model-path",
        str(controller_prior_path),
        "--adversary-prior-model-path",
        str(adversary_prior_path),
    ]
    if only_model_ctrl_cycle:
        cmd.append("--only-model-ctrl-cycle")
    if not bool(write_arena_game_logs):
        cmd.append("--no-arena-game-logs")
    if role_controller is not None and role_adversary is not None:
        cmd.extend([
            "--role-controller-value-model-path",
            str(role_controller.value_model_path),
            "--role-controller-prior-model-path",
            str(role_controller.controller_prior_model_path),
            "--role-adversary-value-model-path",
            str(role_adversary.value_model_path),
            "--role-adversary-prior-model-path",
            str(role_adversary.adversary_prior_model_path),
        ])
    return cmd


def _run_arena(
    *,
    root: Path,
    output_dir: Path,
    model_path: Path,
    model_version: int,
    controller_prior_path: Path,
    adversary_prior_path: Path,
    num_games: int,
    parallel_games: int,
    game_id_start: int,
    iterations: int,
    history_seed: int,
    history_hops_min: int,
    history_hops_max: int,
    only_model_ctrl_cycle: bool,
    write_arena_game_logs: bool = True,
    role_controller: ModelBundle | None = None,
    role_adversary: ModelBundle | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _arena_cmd(
        output_dir=output_dir,
        model_path=model_path,
        model_version=int(model_version),
        controller_prior_path=controller_prior_path,
        adversary_prior_path=adversary_prior_path,
        num_games=int(num_games),
        parallel_games=int(parallel_games),
        game_id_start=int(game_id_start),
        iterations=int(iterations),
        history_seed=int(history_seed),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        only_model_ctrl_cycle=bool(only_model_ctrl_cycle),
        write_arena_game_logs=bool(write_arena_game_logs),
        role_controller=role_controller,
        role_adversary=role_adversary,
    )
    (output_dir / "launch_command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    _run(cmd, cwd=Path(__file__).resolve().parents[2], log_path=output_dir / "arena_launcher.log")
    return output_dir / "arena_results.csv"


def _read_planned_hops(output_dir: Path) -> dict[int, int]:
    path = Path(output_dir) / "planned_games.csv"
    out: dict[int, int] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(float(row.get("game_id", 0) or 0))] = int(float(row.get("history_hops", 0) or 0))
    return out


def _read_cycle2_costs(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not Path(path).exists():
        return out
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gid = int(float(row.get("game_id", 0) or 0))
            out[gid] = float(row.get("cycle2_total_cost", 0.0) or 0.0)
    return out


def _promote_candidate(root: Path, candidate: ModelBundle) -> None:
    models_dir = Path(root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(models_dir / "current_model.json", candidate.to_json())
    state = _load_state(root)
    state["promoted_model_version"] = int(candidate.model_version)
    state["promotions_completed"] = int(state.get("promotions_completed", 0)) + 1
    state["last_promotion_at_utc"] = utc_now()
    _save_state(root, state)


def _broadcast_candidate_to_workers(root: Path, candidate: ModelBundle) -> None:
    version = int(candidate.model_version)
    model_dir = Path(root) / "models" / f"Model_Version{version}"
    for worker in WORKERS:
        remote_model_dir = f"{REMOTE_OUTPUT_ROOT}/models/Model_Version{version}"
        worker_current_dir = f"{REMOTE_OUTPUT_ROOT}/worker_large/{worker.worker_id}/models"
        current_payload = {
            "model_version": version,
            "value_model_path": f"{remote_model_dir}/value/{CONFIG_NAME}/model.joblib",
            "controller_prior_model_path": f"{remote_model_dir}/controller_prior/{POLICY_CONFIG_NAME}/model.joblib",
            "adversary_prior_model_path": f"{remote_model_dir}/adversary_prior/{POLICY_CONFIG_NAME}/model.joblib",
            "updated_at_utc": utc_now(),
        }
        _run(["ssh", worker.host, f"mkdir -p {remote_model_dir!r} {worker_current_dir!r}"])
        _run(["rsync", "-az", "--partial", "--delay-updates", "--timeout=120", str(model_dir).rstrip("/") + "/", f"{worker.host}:{remote_model_dir.rstrip('/')}/"])
        payload = json.dumps(current_payload, sort_keys=True)
        _run(["ssh", worker.host, f"printf '%s\n' {payload!r} > {worker_current_dir!r}/current_model.json"])


def evaluate_and_maybe_promote(
    root: Path,
    *,
    candidate: ModelBundle,
    eval_games: int,
    eval_parallel: int,
    benchmark_games: int,
    iterations: int,
    history_seed: int,
) -> dict[str, Any]:
    promoted = _current_promoted_bundle(root)
    eval_number = int(candidate.model_version)
    eval_dir = Path(root) / "eval_of_models" / f"eval_{eval_number:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    start_gid = 70_000_000 + int(candidate.model_version) * 10_000
    max_hop = max(100, int(eval_games) + 5)

    cand_ctrl_dir = eval_dir / "candidate_controller_vs_promoted_adversary"
    prom_ctrl_dir = eval_dir / "promoted_controller_vs_candidate_adversary"
    cand_ctrl_results = _run_arena(
        root=root,
        output_dir=cand_ctrl_dir,
        model_path=candidate.value_model_path,
        model_version=int(candidate.model_version),
        controller_prior_path=candidate.controller_prior_model_path,
        adversary_prior_path=promoted.adversary_prior_model_path,
        num_games=int(eval_games),
        parallel_games=int(eval_parallel),
        game_id_start=int(start_gid),
        iterations=int(iterations),
        history_seed=int(history_seed),
        history_hops_min=0,
        history_hops_max=int(max_hop),
        only_model_ctrl_cycle=True,
        role_controller=candidate,
        role_adversary=promoted,
    )
    prom_ctrl_results = _run_arena(
        root=root,
        output_dir=prom_ctrl_dir,
        model_path=promoted.value_model_path,
        model_version=int(promoted.model_version),
        controller_prior_path=promoted.controller_prior_model_path,
        adversary_prior_path=candidate.adversary_prior_model_path,
        num_games=int(eval_games),
        parallel_games=int(eval_parallel),
        game_id_start=int(start_gid),
        iterations=int(iterations),
        history_seed=int(history_seed),
        history_hops_min=0,
        history_hops_max=int(max_hop),
        only_model_ctrl_cycle=True,
        role_controller=promoted,
        role_adversary=candidate,
    )

    cand_costs = _read_cycle2_costs(cand_ctrl_results)
    prom_costs = _read_cycle2_costs(prom_ctrl_results)
    hops = _read_planned_hops(cand_ctrl_dir)
    common = sorted(set(cand_costs).intersection(prom_costs))
    wins = sum(1 for gid in common if float(cand_costs[gid]) < float(prom_costs[gid]) - 1e-9)
    win_ratio = float(wins / max(1, len(common)))
    did_promote = bool(common and win_ratio >= float(PROMOTION_WIN_RATE_THRESHOLD))
    new_promoted_version = int(candidate.model_version) if did_promote else int(promoted.model_version)
    eval_time = local_time_24h()
    details = eval_dir / "eval_game_details.csv"
    if not details.exists():
        with details.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=EVAL_GAME_FIELDS).writeheader()
    for gid in common:
        append_csv_row(details, EVAL_GAME_FIELDS, {
            "promoted_model_version": int(promoted.model_version),
            "new_promoted_model_version": int(new_promoted_version),
            "candidate_win_ratio_in_100_games": float(win_ratio),
            "candidate_promoted": int(did_promote),
            "time_of_eval": eval_time,
            "game_id": int(gid),
            "game_hop_number": int(hops.get(gid, 0)),
            "total_cost_when_promoted_is_adv_and_new_is_controller": float(cand_costs[gid]),
            "total_cost_when_new_is_adv_and_promoted_is_controller": float(prom_costs[gid]),
        })

    result = {
        "eval_dir": str(eval_dir),
        "promoted_model_version": int(promoted.model_version),
        "candidate_model_version": int(candidate.model_version),
        "new_promoted_model_version": int(new_promoted_version),
        "candidate_win_ratio": float(win_ratio),
        "candidate_wins": int(wins),
        "games_compared": int(len(common)),
        "candidate_promoted": bool(did_promote),
        "promotion_win_rate_threshold": float(PROMOTION_WIN_RATE_THRESHOLD),
    }
    atomic_write_json(eval_dir / "eval_summary.json", result)

    if did_promote:
        _promote_candidate(root, candidate)
        _broadcast_candidate_to_workers(root, candidate)
        sjf_dir = eval_dir / "SJF_256_Game"
        _run_arena(
            root=root,
            output_dir=sjf_dir,
            model_path=candidate.value_model_path,
            model_version=int(candidate.model_version),
            controller_prior_path=candidate.controller_prior_model_path,
            adversary_prior_path=candidate.adversary_prior_model_path,
            num_games=int(benchmark_games),
            parallel_games=int(eval_parallel),
            game_id_start=80_000_000 + int(candidate.model_version) * 10_000,
            iterations=int(iterations),
            history_seed=int(history_seed) + 909,
            history_hops_min=0,
            history_hops_max=max(100, int(benchmark_games) + 5),
            only_model_ctrl_cycle=False,
            write_arena_game_logs=True,
        )
    return result


def train_candidate(root: Path, *, seed: int = 2026, max_value_states: int = 500_000, eval_games: int = 100, eval_parallel: int = 60, benchmark_games: int = 50, mcts_iterations: int = 1000) -> dict[str, Any]:
    root = Path(root)
    replay_paths = _feature_replay_paths(root)
    policy_paths = _policy_replay_paths(root)
    if not replay_paths:
        raise RuntimeError(f"no feature-complete replay partitions or legacy replay found under {root / 'global_replay'}")
    if not policy_paths:
        raise RuntimeError(f"no policy replay partitions or legacy replay found under {root / 'global_replay'}")
    sampled, controller_roots, adversary_roots, replay_counts_sampled = _stream_sample_state_rows(
        replay_paths,
        seed=int(seed),
        max_value_rows=int(max_value_states),
        max_controller_policy_roots=int(XL_CONTROLLER_POLICY_SAMPLE_CAP),
        max_adversary_policy_roots=int(ADVERSARY_POLICY_SAMPLE_CAP),
    )
    if not sampled:
        raise RuntimeError(f"no feature-complete replay rows found in {len(replay_paths)} replay file(s)")
    if int(replay_counts_sampled["adversary"]) < int(MIN_ADVERSARY_STATES_FOR_EVAL):
        raise RuntimeError(
            f"not enough adversary feature rows: {replay_counts_sampled['adversary']} < {MIN_ADVERSARY_STATES_FOR_EVAL}"
        )

    Xv, yv = _value_arrays_from_rows(sampled)
    version = _next_candidate_version(root)
    out = root / "models" / f"Model_Version{int(version)}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    value_model = _fit_value(Xv, yv, seed=seed + version)
    value_pred = value_model.predict(Xv).astype(np.float32)
    value_metrics = _value_metrics(yv, value_pred)
    value_dir = out / "value" / CONFIG_NAME
    value_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(value_model, value_dir / "model.joblib", compress=3)
    _export_hgb_to_native_text(value_model, value_dir / "native_model.tsv", feature_dim_override=226, model_tag_override=f"agz_value_v{version}")

    Xc, yc, pc, offc = _collect_policy_training_arrays(
        policy_paths,
        controller_roots,
        player="controller",
        action_dim=CTRL_ACTION_DIM,
    )
    Xa, ya, pa, offa = _collect_policy_training_arrays(
        policy_paths,
        adversary_roots,
        player="adversary",
        action_dim=ADV_ACTION_DIM,
    )
    if Xc.shape[0] == 0 or Xa.shape[0] == 0:
        raise RuntimeError(f"policy rows missing: controller_rows={Xc.shape[0]} adversary_rows={Xa.shape[0]}")

    ctrl_model = _fit_policy(Xc, yc, seed=seed + version + 101)
    adv_model = _fit_policy(Xa, ya, seed=seed + version + 201)
    ctrl_metrics = _policy_metrics(ctrl_model, Xc, yc, pc, offc)
    adv_metrics = _policy_metrics(adv_model, Xa, ya, pa, offa)

    ctrl_dir = out / "controller_prior" / POLICY_CONFIG_NAME
    adv_dir = out / "adversary_prior" / POLICY_CONFIG_NAME
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    adv_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ctrl_model, ctrl_dir / "model.joblib", compress=3)
    joblib.dump(adv_model, adv_dir / "model.joblib", compress=3)
    _export_hgb_to_native_text(ctrl_model, ctrl_dir / "native_model.tsv", feature_dim_override=269, model_tag_override=f"agz_controller_prior_v{version}")
    _export_hgb_to_native_text(adv_model, adv_dir / "native_model.tsv", feature_dim_override=233, model_tag_override=f"agz_adversary_prior_v{version}")

    metrics = {
        "model_config": CONFIG_NAME,
        "model_version": int(version),
        "time_of_training_24h": local_time_24h(),
        "sampled_controller_states": int(sum(1 for r in sampled if str(r.get("player")) == "controller")),
        "sampled_adversary_states": int(sum(1 for r in sampled if str(r.get("player")) == "adversary")),
        "value_mse": value_metrics["mse"],
        "value_rmse": value_metrics["rmse"],
        "value_max_abs_error": value_metrics["max_abs_error"],
        "value_p95_abs_error": value_metrics["p95_abs_error"],
        "controller_policy_mse": ctrl_metrics["mse"],
        "controller_policy_cross_entropy": ctrl_metrics["cross_entropy"],
        "controller_policy_top1": ctrl_metrics["top1"],
        "controller_policy_top3": ctrl_metrics["top3"],
        "adversary_policy_mse": adv_metrics["mse"],
        "adversary_policy_cross_entropy": adv_metrics["cross_entropy"],
        "adversary_policy_top1": adv_metrics["top1"],
        "adversary_policy_top3": adv_metrics["top3"],
    }
    append_csv_row(root / "train_model.csv", TRAIN_MODEL_FIELDS, metrics)
    candidate = _candidate_bundle_from_output(int(version), out)
    manifest = {
        **metrics,
        "value_model_path": str(value_dir / "model.joblib"),
        "controller_prior_model_path": str(ctrl_dir / "model.joblib"),
        "adversary_prior_model_path": str(adv_dir / "model.joblib"),
        "fit_elapsed_s": float(time.time() - t0),
        "created_at_utc": utc_now(),
        "promotion_win_rate_threshold": float(PROMOTION_WIN_RATE_THRESHOLD),
        "eval_status": "running",
    }
    atomic_write_json(out / "candidate_manifest.json", manifest)
    _mark_training_complete(root, int(version))
    del sampled
    del controller_roots
    del adversary_roots
    del Xv, yv, value_pred
    del Xc, yc, pc, offc
    del Xa, ya, pa, offa
    del value_model, ctrl_model, adv_model
    gc.collect()
    eval_result = evaluate_and_maybe_promote(
        root,
        candidate=candidate,
        eval_games=int(eval_games),
        eval_parallel=int(eval_parallel),
        benchmark_games=int(benchmark_games),
        iterations=int(mcts_iterations),
        history_seed=int(seed) + int(version),
    )
    manifest.update({"eval_status": "complete", **eval_result})
    atomic_write_json(out / "candidate_manifest.json", manifest)
    _mark_training_cycle_complete(root, int(version))
    _mark_current_training_finalized(root, int(version))
    return {**metrics, **eval_result}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one AGZ candidate HGB bundle from XL feature-complete replay.")
    p.add_argument("--output-root", type=Path, default=Path("/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-value-states", type=int, default=500_000)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--eval-parallel", type=int, default=60)
    p.add_argument("--benchmark-games", type=int, default=50)
    p.add_argument("--mcts-iterations", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_candidate(
        Path(args.output_root).expanduser(),
        seed=int(args.seed),
        max_value_states=int(args.max_value_states),
        eval_games=int(args.eval_games),
        eval_parallel=int(args.eval_parallel),
        benchmark_games=int(args.benchmark_games),
        mcts_iterations=int(args.mcts_iterations),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
