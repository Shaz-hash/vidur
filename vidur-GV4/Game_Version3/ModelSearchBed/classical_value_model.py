"""Classical value models for GV3 ModelSearchBed experiments.

The default model in this branch is a learned, non-neural controller value
surrogate. It uses only state-derived features from the stored simulator
snapshot and stats, then fits compact tree ensembles under the requested
150k scalar-parameter budget.

The symbolic Bellman evaluator remains available as an explicit diagnostic
backend, but it is not the default because it recomputes the label generator
instead of learning a reusable bootstrap value function.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


TARGET_EPS = 1e-12
DEFAULT_TOP_K_REQUESTS = 12
DEFAULT_MODEL_BUDGET = 150_000
MODEL_INPUT_FEATURE_CACHE_VERSION = "model_inputs_v3"


_FEATURE_WORKER_STATE_LOADER: Any | None = None
_FEATURE_WORKER_PLAYER = "controller"


def _limit_native_threads(num_threads: int) -> None:
    """Limit BLAS/OpenMP pools in spawned workers."""

    threads = max(1, int(num_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=threads)
    except Exception:
        pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _stat_float(stats: Any, name: str, default: float = 0.0) -> float:
    return _safe_float(getattr(stats, name, default), default)


def _as_dict(value: Any) -> dict[Any, Any]:
    return value if isinstance(value, dict) else {}


def _stats_dict_value(stats: Any, name: str, key: int, default: float = 0.0) -> float:
    mapping = getattr(stats, name, None)
    if not isinstance(mapping, dict):
        return float(default)
    return _safe_float(mapping.get(int(key), default), default)


def _aggregate(values: list[float]) -> list[float]:
    """Return compact count/sum/location/range stats for a numeric list."""

    if not values:
        return [0.0] * 8
    arr = np.asarray(values, dtype=np.float32)
    return [
        float(arr.size),
        float(np.sum(arr)),
        float(np.mean(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
    ]


def _request_feature_row(request: dict[str, Any], stats: Any, sim_time: float) -> list[float]:
    """Build raw per-request features from one stored request snapshot."""

    rid = _safe_int(request.get("id"), -1)
    arrived_at = _safe_float(request.get("arrived_at"))
    queued_at = _safe_float(request.get("queued_at"), arrived_at)
    age = max(0.0, float(sim_time) - arrived_at)
    queued_age = max(0.0, float(sim_time) - queued_at)

    total_prefill = max(0.0, _safe_float(request.get("num_prefill_tokens")))
    remaining_prefill = max(0.0, _safe_float(request.get("remaining_prefill_tokens")))
    done_prefill = max(0.0, total_prefill - remaining_prefill)
    total_decode = max(0.0, _safe_float(request.get("num_decode_tokens")))
    processed_total = max(0.0, _safe_float(request.get("num_processed_tokens")))
    done_decode = max(0.0, processed_total - total_prefill)
    remaining_decode = max(0.0, total_decode - done_decode)

    prefill_slo = max(0.0, _safe_float(request.get("prefill_slo_time")))
    decode_slo = max(0.0, _safe_float(request.get("decode_slo_time")))
    completion_slo = max(0.0, _safe_float(request.get("completion_slo_time")))
    prefill_deadline = arrived_at + prefill_slo

    decode_deadlines = getattr(stats, "decode_next_deadline_by_id", {}) or {}
    if isinstance(decode_deadlines, dict) and rid in decode_deadlines:
        decode_deadline = _safe_float(decode_deadlines.get(rid), arrived_at + decode_slo)
    else:
        decode_deadline = arrived_at + decode_slo

    prefill_slack = prefill_deadline - float(sim_time)
    decode_slack = decode_deadline - float(sim_time)
    prefill_complete = bool(request.get("is_prefill_complete")) or remaining_prefill <= 0.0

    stored_prefill_late = _stats_dict_value(stats, "per_request_prefill_lateness", rid, 0.0)
    stored_decode_late = _stats_dict_value(stats, "per_request_decode_lateness", rid, 0.0)
    current_prefill_late = max(0.0, float(sim_time) - prefill_deadline) if not prefill_complete else 0.0
    current_decode_late = max(0.0, float(sim_time) - decode_deadline) if prefill_complete else 0.0
    prefill_late = max(stored_prefill_late, current_prefill_late)
    decode_late = max(stored_decode_late, current_decode_late)

    processed_frac = processed_total / max(1.0, total_prefill + total_decode)
    prefill_frac = done_prefill / max(1.0, total_prefill)
    decode_frac = done_decode / max(1.0, total_decode)

    return [
        age,
        queued_age,
        total_prefill,
        remaining_prefill,
        done_prefill,
        total_decode,
        remaining_decode,
        done_decode,
        processed_total,
        processed_frac,
        prefill_frac,
        decode_frac,
        prefill_slo,
        decode_slo,
        completion_slo,
        prefill_slack,
        decode_slack,
        prefill_late,
        decode_late,
        1.0 if bool(request.get("scheduled")) else 0.0,
        1.0 if bool(request.get("preempted")) else 0.0,
        1.0 if bool(request.get("completed")) else 0.0,
        1.0 if prefill_complete else 0.0,
        float(_safe_int(request.get("num_restarts"), 0)),
    ]


@dataclass(frozen=True)
class LearnedClassicalControllerValueModel:
    """Compact non-neural controller value surrogate."""

    backend: str
    estimators: dict[str, Any]
    feature_names: list[str]
    feature_config: dict[str, Any]
    trainable_params: int
    blend_weight_hgb: float = 0.35
    zero_threshold: float = 0.70
    zero_after_residual_threshold: float = 1.01
    residual_scale: float = 1.0
    model_name: str = "learned_classical_controller_value"
    uses_neural_network: bool = False
    uses_target_leakage: bool = False
    cached_predictions: dict[str, list[float]] = field(default_factory=dict, repr=False, compare=False)
    runtime_prediction_cache: dict[bytes, float] = field(default_factory=dict, repr=False, compare=False)

    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: Any | None = None,
    ) -> tuple[float, list[float]]:
        """MCTS bootstrap adapter for non-neural models.

        GV3 MCTS only passes `ModelInputs` at bootstrap time. Models trained
        with `feature_source=model_inputs` can therefore be used directly in
        `reward + discount * V(child_state)`.
        """

        del player, device
        if str(self.feature_config.get("feature_source", "")) != "model_inputs":
            raise RuntimeError(
                "classical bootstrap requires feature_source='model_inputs'; "
                f"got {self.feature_config.get('feature_source')!r}"
            )
        row, _names = extract_model_input_feature_vector(inputs)
        x = np.asarray([row], dtype=np.float32)
        key = x.tobytes()
        cached = self.runtime_prediction_cache.get(key)
        if cached is not None:
            return float(cached), []
        # Single-row bootstrap calls happen inside MCTS action loops. Avoid
        # joblib process/thread dispatch overhead from forest predictors.
        extra = self.estimators.get("extra")
        if extra is not None and hasattr(extra, "n_jobs"):
            try:
                extra.n_jobs = 1
            except Exception:
                pass
        value = float(self.predict_matrix(x)[0])
        if len(self.runtime_prediction_cache) > 50_000:
            self.runtime_prediction_cache.clear()
        self.runtime_prediction_cache[key] = value
        return value, []

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "extra_trees":
            pred = np.asarray(self.estimators["extra"].predict(x), dtype=np.float64)
        elif self.backend == "hist_gradient_boosting":
            pred = np.asarray(self.estimators["hgb"].predict(x), dtype=np.float64)
        elif self.backend in {"hybrid_extra_hgb", "bellman_shaped"}:
            extra = np.asarray(self.estimators["extra"].predict(x), dtype=np.float64)
            hgb = np.asarray(self.estimators["hgb"].predict(x), dtype=np.float64)
            w = float(self.blend_weight_hgb)
            pred = (1.0 - w) * extra + w * hgb
        else:
            raise ValueError(f"unknown classical backend: {self.backend!r}")

        # The controller first-layer value is a non-positive penalty.
        pred = np.minimum(pred, 0.0)

        if self.backend == "bellman_shaped":
            zero_gate = self.estimators.get("zero_gate")
            zero_prob = None
            if zero_gate is not None:
                proba = np.asarray(zero_gate.predict_proba(x), dtype=np.float64)
                classes = list(getattr(zero_gate, "classes_", []))
                if True in classes:
                    zero_idx = classes.index(True)
                elif 1 in classes:
                    zero_idx = classes.index(1)
                else:
                    zero_idx = None
                if zero_idx is None:
                    zero_prob = np.zeros((x.shape[0],), dtype=np.float64)
                else:
                    zero_prob = proba[:, int(zero_idx)]
                pred = pred.copy()
                pred[zero_prob >= float(self.zero_threshold)] = 0.0

            residual = self.estimators.get("residual")
            if residual is not None and float(self.residual_scale) != 0.0:
                pred = np.minimum(pred + float(self.residual_scale) * np.asarray(residual.predict(x), dtype=np.float64), 0.0)
                if zero_prob is not None:
                    pred = pred.copy()
                    pred[zero_prob >= float(self.zero_after_residual_threshold)] = 0.0

        return pred


@dataclass(frozen=True)
class SymbolicBellmanControllerValueModel:
    """Zero-parameter diagnostic evaluator, not the default learned model."""

    model_name: str = "symbolic_bellman_controller_value"
    trainable_params: int = 0
    uses_neural_network: bool = False
    uses_target_leakage: bool = False


def extract_state_feature_vector(
    record: dict[str, Any],
    *,
    top_k_requests: int = DEFAULT_TOP_K_REQUESTS,
) -> tuple[list[float], list[str]]:
    """Extract deployable state-only features from one stored root record.

    This intentionally does not consume target-construction fields:
    `target_value`, `best_reward`, `best_child_cost`, `best_action_repr`,
    `best_action_index`, or `history_signature`.
    """

    snapshot = _as_dict(record.get("simulator_snapshot"))
    stats = record.get("stats")
    sim_time = _safe_float(snapshot.get("time"))
    request_states = list(_as_dict(snapshot.get("request_states")).values())

    # Request-level deployable features must describe only in-system requests.
    # Snapshot request_states can include finalized/historical records; if the
    # active set is empty, the active request feature groups should be zero.
    active_ids = getattr(stats, "active_request_ids", None)
    active_id_set = {int(x) for x in (active_ids or [])}
    active_requests = [
        req
        for req in request_states
        if isinstance(req, dict) and _safe_int(req.get("id"), -999999) in active_id_set
    ]

    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: Any) -> None:
        names.append(name)
        values.append(_safe_float(value))

    add("sim_time", sim_time)
    add("log1p_sim_time", math.log1p(max(0.0, sim_time)))
    add("root_depth", record.get("root_depth", 0))
    add("history_hops", record.get("history_hops", 0))
    add("num_snapshot_requests", len(request_states))
    add("num_active_requests", len(active_requests))
    add("num_stats_active_ids", len(active_id_set))
    add("num_waiting_ids", len(snapshot.get("waiting_ids") or []))
    add("num_running_ids", len(snapshot.get("running_ids") or []))

    for stat_name in (
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "slo_lateness_sum",
        "last_prefill_batch_time",
        "transition_discount_time",
        "transition_final_time",
        "transition_fast_forward_time",
    ):
        add(f"stats_{stat_name}", _stat_float(stats, stat_name))
    add("objective_cost", _stat_float(stats, "slo_violations") + _stat_float(stats, "slo_lateness_sum"))

    completed_ids = getattr(stats, "completed_request_ids", set()) or set()
    stopped_ids = getattr(stats, "stopped_decode_request_ids", set()) or set()
    dropped_ids = getattr(stats, "dropped_request_ids", set()) or set()
    violated_ids = getattr(stats, "violated_request_ids", set()) or set()
    add("num_completed_ids", len(completed_ids))
    add("num_stopped_decode_ids", len(stopped_ids))
    add("num_dropped_ids", len(dropped_ids))
    add("num_violated_ids", len(violated_ids))

    recent_arrivals = list(getattr(stats, "recent_arrivals", []) or [])
    add("recent_arrival_events", len(recent_arrivals))
    add("recent_arrival_req_sum", sum(_safe_float(x[1]) for x in recent_arrivals))
    add("recent_arrival_prefill_sum", sum(_safe_float(x[2]) for x in recent_arrivals))
    for window in (2, 4, 8):
        sub = recent_arrivals[-window:]
        add(f"recent_arrival_req_sum_last_{window}", sum(_safe_float(x[1]) for x in sub))
        add(f"recent_arrival_prefill_sum_last_{window}", sum(_safe_float(x[2]) for x in sub))

    row_names = [
        "age",
        "queued_age",
        "total_prefill",
        "remaining_prefill",
        "done_prefill",
        "total_decode",
        "remaining_decode",
        "done_decode",
        "processed_total",
        "processed_frac",
        "prefill_frac",
        "decode_frac",
        "prefill_slo",
        "decode_slo",
        "completion_slo",
        "prefill_slack",
        "decode_slack",
        "prefill_late",
        "decode_late",
        "scheduled_bit",
        "preempted_bit",
        "completed_bit",
        "prefill_complete_bit",
        "num_restarts",
    ]
    rows = [_request_feature_row(req, stats, sim_time) for req in active_requests]

    by_name = {name: [] for name in row_names}
    for row in rows:
        for name, value in zip(row_names, row):
            by_name[name].append(float(value))

    for name in row_names:
        for suffix, value in zip(
            ("count", "sum", "mean", "min", "max", "p10", "p50", "p90"),
            _aggregate(by_name[name]),
        ):
            add(f"req_{name}_{suffix}", value)

    threshold_specs = (
        ("prefill_slack", "le_0", lambda x: x <= 0.0),
        ("prefill_slack", "le_20ms", lambda x: x <= 0.02),
        ("decode_slack", "le_0", lambda x: x <= 0.0),
        ("decode_slack", "le_20ms", lambda x: x <= 0.02),
        ("prefill_late", "gt_0", lambda x: x > 0.0),
        ("prefill_late", "gt_500ms", lambda x: x > 0.5),
        ("decode_late", "gt_0", lambda x: x > 0.0),
        ("decode_late", "gt_500ms", lambda x: x > 0.5),
        ("remaining_prefill", "gt_0", lambda x: x > 0.0),
        ("remaining_decode", "gt_0", lambda x: x > 0.0),
        ("done_decode", "gt_0", lambda x: x > 0.0),
    )
    for feature_name, label, pred in threshold_specs:
        add(f"req_{feature_name}_{label}_count", sum(1.0 for x in by_name[feature_name] if pred(x)))

    # Preserve the most urgent request-level details. Sorting by slack makes the
    # fixed slots stable and useful for tree splits.
    urgent_rows = []
    for req, row in zip(active_requests, rows):
        rid = _safe_int(req.get("id"), -1)
        prefill_slack = row[row_names.index("prefill_slack")]
        decode_slack = row[row_names.index("decode_slack")]
        prefill_complete = row[row_names.index("prefill_complete_bit")] >= 0.5
        urgency = decode_slack if prefill_complete else prefill_slack
        late_score = row[row_names.index("prefill_late")] + row[row_names.index("decode_late")]
        urgent_rows.append((float(urgency), -float(late_score), rid, row))
    urgent_rows.sort(key=lambda item: (item[0], item[1], item[2]))

    for slot in range(int(top_k_requests)):
        if slot < len(urgent_rows):
            row = urgent_rows[slot][3]
        else:
            row = [0.0] * len(row_names)
        for name, value in zip(row_names, row):
            add(f"slot_{slot:02d}_{name}", value)

    return values, names


def build_feature_matrix(
    records: list[dict[str, Any]],
    *,
    top_k_requests: int = DEFAULT_TOP_K_REQUESTS,
) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    feature_names: list[str] | None = None
    for record in records:
        row, names = extract_state_feature_vector(record, top_k_requests=top_k_requests)
        if feature_names is None:
            feature_names = list(names)
        rows.append(row)
    if not rows:
        raise ValueError("cannot build features for empty records")
    return np.asarray(rows, dtype=np.float32), list(feature_names or [])


def _tensor_to_2d_array(value: Any) -> np.ndarray:
    arr = value.detach().to("cpu").float().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return np.asarray(arr.reshape(1, -1), dtype=np.float32)
    return np.asarray(arr.reshape(1, -1), dtype=np.float32)


def _tensor_to_1d_array(value: Any, *, fallback_len: int) -> np.ndarray:
    if value is None:
        return np.ones((int(fallback_len),), dtype=np.float32)
    arr = value.detach().to("cpu").float().numpy()
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr.reshape(-1), dtype=np.float32)


def _add_masked_request_features(
    *,
    values: list[float],
    names: list[str],
    prefix: str,
    features: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Append flat and aggregate request features from `ModelInputs` tensors."""

    n_req = int(features.shape[0])
    width = int(features.shape[1]) if features.ndim == 2 else 0
    mask = np.asarray(mask[:n_req] > 0.5, dtype=bool)
    masked = np.asarray(features, dtype=np.float32).copy()
    if masked.size:
        masked[~mask, :] = 0.0

    values.append(float(np.sum(mask)))
    names.append(f"{prefix}_active_count")

    if width <= 0:
        return

    active = masked[mask]
    if active.size == 0:
        active = np.zeros((0, width), dtype=np.float32)

    for dim in range(width):
        col = active[:, dim] if active.shape[0] else np.asarray([], dtype=np.float32)
        if col.size:
            stats = (
                float(np.sum(col)),
                float(np.mean(col)),
                float(np.min(col)),
                float(np.max(col)),
            )
        else:
            stats = (0.0, 0.0, 0.0, 0.0)
        for stat_name, stat_value in zip(("sum", "mean", "min", "max"), stats):
            names.append(f"{prefix}_dim_{dim:02d}_{stat_name}")
            values.append(stat_value)

    for idx, flag in enumerate(mask.astype(np.float32).tolist()):
        names.append(f"{prefix}_mask_{idx:02d}")
        values.append(float(flag))

    flat = masked.reshape(-1)
    for idx, value in enumerate(flat.tolist()):
        names.append(f"{prefix}_flat_{idx:04d}")
        values.append(float(value))


def extract_model_input_feature_vector(inputs: Any) -> tuple[list[float], list[str]]:
    """Extract classical features from GV3 `ModelInputs`.

    This is the bootstrap-compatible representation: the same function is used
    for training records converted through `build_model_inputs` and for MCTS
    child-state bootstrap calls.
    """

    values: list[float] = []
    names: list[str] = []

    global_arr = _tensor_to_2d_array(inputs.global_features).reshape(-1)
    for idx, value in enumerate(global_arr.tolist()):
        # The bootstrap value is controller-perspective V(s), not a separate
        # player-head prediction. Excluding the player indicator bits avoids a
        # train/bootstrap distribution shift: roots are controller-to-act, while
        # child states after controller actions are adversary-to-act.
        if idx in (0, 1):
            continue
        names.append(f"global_{idx:03d}")
        values.append(float(value))

    prefill = _tensor_to_2d_array(inputs.prefill_req_features)
    decode = _tensor_to_2d_array(inputs.decode_req_features)
    prefill_mask = _tensor_to_1d_array(inputs.prefill_req_mask, fallback_len=int(prefill.shape[0]))
    decode_mask = _tensor_to_1d_array(inputs.decode_req_mask, fallback_len=int(decode.shape[0]))

    _add_masked_request_features(
        values=values,
        names=names,
        prefix="prefill",
        features=prefill,
        mask=prefill_mask,
    )
    _add_masked_request_features(
        values=values,
        names=names,
        prefix="decode",
        features=decode,
        mask=decode_mask,
    )
    return values, names


def build_model_input_feature_matrix(
    records: list[dict[str, Any]],
    *,
    state_loader: Any,
    player: str = "controller",
) -> tuple[np.ndarray, list[str]]:
    """Build features by reconstructing states and using DNN/infer inputs."""

    if state_loader is None:
        raise ValueError("state_loader is required for feature_source='model_inputs'")

    from ..DNN import infer as dnn_infer

    rows: list[list[float]] = []
    feature_names: list[str] | None = None
    for index, record in enumerate(records, start=1):
        state = state_loader(record)
        inputs = dnn_infer.build_model_inputs(
            state,
            str(record.get("root_player", player)),
            device="cpu",
            build_action_mask_flag=False,
        )
        row, names = extract_model_input_feature_vector(inputs)
        if feature_names is None:
            feature_names = list(names)
        rows.append(row)
        if index % 2048 == 0:
            mcts = getattr(state_loader, "mcts", None)
            clear_fn = getattr(mcts, "clear_search_state", None)
            if callable(clear_fn):
                clear_fn(drop_scratch=True)
            gc.collect()
    if not rows:
        raise ValueError("cannot build features for empty records")
    return np.asarray(rows, dtype=np.float32), list(feature_names or [])


def _records_feature_fingerprint(
    records: list[dict[str, Any]],
    *,
    player: str,
) -> str:
    """Return a target-independent cache key for an ordered record split."""

    h = hashlib.blake2b(digest_size=16)
    h.update(MODEL_INPUT_FEATURE_CACHE_VERSION.encode("utf-8"))
    h.update(str(player).encode("utf-8"))
    h.update(str(len(records)).encode("utf-8"))
    for record in records:
        for key in ("root_id", "root_depth", "root_player", "root_node_id_override"):
            h.update(str(record.get(key, "")).encode("utf-8", errors="replace"))
            h.update(b"\0")
    return h.hexdigest()


def _feature_cache_paths(
    *,
    cache_dir: Path,
    split_name: str,
    fingerprint: str,
) -> tuple[Path, Path, Path]:
    stem = f"{MODEL_INPUT_FEATURE_CACHE_VERSION}_{split_name}_{fingerprint}"
    return (
        cache_dir / f"{stem}.npy",
        cache_dir / f"{stem}.names.json",
        cache_dir / f"{stem}.meta.json",
    )


def _load_feature_cache(
    *,
    cache_dir: Path,
    split_name: str,
    fingerprint: str,
    expected_count: int,
) -> tuple[np.ndarray, list[str]] | None:
    matrix_path, names_path, _meta_path = _feature_cache_paths(
        cache_dir=cache_dir,
        split_name=split_name,
        fingerprint=fingerprint,
    )
    if not matrix_path.exists() or not names_path.exists():
        return None
    matrix = np.load(matrix_path, allow_pickle=False)
    if int(matrix.shape[0]) != int(expected_count):
        return None
    names = json.loads(names_path.read_text(encoding="utf-8"))
    return np.asarray(matrix, dtype=np.float32), [str(x) for x in names]


def _write_feature_cache(
    *,
    cache_dir: Path,
    split_name: str,
    fingerprint: str,
    matrix: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, names_path, meta_path = _feature_cache_paths(
        cache_dir=cache_dir,
        split_name=split_name,
        fingerprint=fingerprint,
    )
    tmp_matrix = matrix_path.with_name(matrix_path.name + ".tmp")
    tmp_names = names_path.with_name(names_path.name + ".tmp")
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_matrix.open("wb") as f:
        np.save(f, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
    tmp_names.write_text(json.dumps(list(feature_names)), encoding="utf-8")
    tmp_meta.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True), encoding="utf-8")
    tmp_matrix.replace(matrix_path)
    tmp_names.replace(names_path)
    tmp_meta.replace(meta_path)


def _init_model_input_feature_worker(
    model_cfg: Any,
    extra_config: dict[str, Any],
    player: str,
) -> None:
    """Initialize one feature extraction worker with its own GV3 environment."""

    global _FEATURE_WORKER_STATE_LOADER
    global _FEATURE_WORKER_PLAYER
    from .self_model_test import RootStateLoader

    _limit_native_threads(int((extra_config or {}).get("classical_worker_threads", 1)))
    _FEATURE_WORKER_STATE_LOADER = RootStateLoader(model_cfg)
    _FEATURE_WORKER_STATE_LOADER._bellman_extra_config = dict(extra_config or {})
    _FEATURE_WORKER_PLAYER = str(player)


def _model_input_feature_worker(
    item: tuple[int, list[dict[str, Any]]],
) -> tuple[int, np.ndarray, list[str]]:
    """Build model-input features for one record chunk in a worker process."""

    chunk_index, records = item
    if _FEATURE_WORKER_STATE_LOADER is None:
        raise RuntimeError("feature worker state loader was not initialized")
    matrix, names = build_model_input_feature_matrix(
        records,
        state_loader=_FEATURE_WORKER_STATE_LOADER,
        player=str(_FEATURE_WORKER_PLAYER),
    )
    mcts = getattr(_FEATURE_WORKER_STATE_LOADER, "mcts", None)
    clear_fn = getattr(mcts, "clear_search_state", None)
    if callable(clear_fn):
        clear_fn(drop_scratch=True)
    gc.collect()
    return int(chunk_index), matrix, names


def _build_model_input_feature_matrix_parallel(
    records: list[dict[str, Any]],
    *,
    model_cfg: Any,
    extra_config: dict[str, Any],
    player: str,
    num_processes: int,
    chunk_records: int,
    maxtasks_per_child: int | None,
    start_method: str,
) -> tuple[np.ndarray, list[str]]:
    """Build model-input features with one local simulator per process."""

    if not records:
        raise ValueError("cannot build features for empty records")
    chunk_size = max(1, int(chunk_records))
    chunks: list[tuple[int, list[dict[str, Any]]]] = []
    for chunk_index, start in enumerate(range(0, len(records), chunk_size)):
        chunks.append((int(chunk_index), list(records[start : start + chunk_size])))
    process_count = min(max(1, int(num_processes)), len(chunks))
    if process_count <= 1:
        return build_model_input_feature_matrix(
            records,
            state_loader=_FEATURE_WORKER_STATE_LOADER,
            player=str(player),
        )
    method = str(start_method or "spawn")
    if method not in mp.get_all_start_methods():
        raise RuntimeError(f"multiprocessing start method {method!r} is not available")
    maxtasks = (
        int(maxtasks_per_child)
        if maxtasks_per_child is not None and int(maxtasks_per_child) > 0
        else None
    )
    print(
        "[classical_features] "
        f"building {len(records)} rows with {process_count} processes, "
        f"{len(chunks)} chunks, chunk_records={chunk_size}, start_method={method}",
        flush=True,
    )
    ctx = mp.get_context(method)
    results: dict[int, tuple[np.ndarray, list[str]]] = {}
    completed = 0
    with ctx.Pool(
        processes=int(process_count),
        initializer=_init_model_input_feature_worker,
        initargs=(model_cfg, dict(extra_config or {}), str(player)),
        maxtasksperchild=maxtasks,
    ) as pool:
        for chunk_index, matrix, names in pool.imap_unordered(
            _model_input_feature_worker,
            chunks,
        ):
            results[int(chunk_index)] = (matrix, names)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(chunks):
                print(
                    "[classical_features] "
                    f"completed_chunks={completed}/{len(chunks)}",
                    flush=True,
                )
    ordered = [results[index] for index in range(len(chunks))]
    feature_names = list(ordered[0][1])
    matrices = [item[0] for item in ordered]
    return np.concatenate(matrices, axis=0).astype(np.float32, copy=False), feature_names


def build_or_load_model_input_feature_matrix(
    records: list[dict[str, Any]],
    *,
    state_loader: Any,
    model_cfg: Any,
    extra_config: dict[str, Any],
    output_dir: Path,
    split_name: str,
    player: str = "controller",
) -> tuple[np.ndarray, list[str]]:
    """Build model-input features once and reuse them across Bellman versions."""

    if state_loader is None:
        raise ValueError("state_loader is required for feature_source='model_inputs'")
    cache_enabled = bool(extra_config.get("classical_feature_cache_enabled", True))
    cache_dir = Path(
        extra_config.get(
            "classical_feature_cache_dir",
            str(output_dir.parent / "_classical_feature_cache"),
        )
    )
    fingerprint = _records_feature_fingerprint(records, player=str(player))
    if cache_enabled:
        cached = _load_feature_cache(
            cache_dir=cache_dir,
            split_name=str(split_name),
            fingerprint=fingerprint,
            expected_count=len(records),
        )
        if cached is not None:
            matrix, names = cached
            print(
                "[classical_features] "
                f"cache hit split={split_name} rows={matrix.shape[0]} "
                f"features={matrix.shape[1]} path={cache_dir}",
                flush=True,
            )
            return matrix, names

    num_processes = int(extra_config.get("classical_feature_num_processes", 1))
    chunk_records = int(extra_config.get("classical_feature_chunk_records", 1024))
    start_method = str(extra_config.get("classical_feature_mp_start_method", "spawn"))
    maxtasks_raw = extra_config.get("classical_feature_maxtasks_per_child", None)
    t0 = time.time()
    if num_processes > 1 and len(records) > max(1, int(chunk_records)):
        matrix, names = _build_model_input_feature_matrix_parallel(
            records,
            model_cfg=model_cfg,
            extra_config=extra_config,
            player=str(player),
            num_processes=int(num_processes),
            chunk_records=int(chunk_records),
            maxtasks_per_child=maxtasks_raw,
            start_method=str(start_method),
        )
    else:
        matrix, names = build_model_input_feature_matrix(
            records,
            state_loader=state_loader,
            player=str(player),
        )
    seconds = time.time() - t0
    print(
        "[classical_features] "
        f"built split={split_name} rows={matrix.shape[0]} features={matrix.shape[1]} "
        f"seconds={seconds:.2f}",
        flush=True,
    )
    if cache_enabled:
        _write_feature_cache(
            cache_dir=cache_dir,
            split_name=str(split_name),
            fingerprint=fingerprint,
            matrix=matrix,
            feature_names=names,
            metadata={
                "cache_version": MODEL_INPUT_FEATURE_CACHE_VERSION,
                "split_name": str(split_name),
                "rows": int(matrix.shape[0]),
                "features": int(matrix.shape[1]),
                "seconds": float(seconds),
                "feature_num_processes": int(num_processes),
                "feature_chunk_records": int(chunk_records),
            },
        )
        print(
            "[classical_features] "
            f"cache written split={split_name} path={cache_dir}",
            flush=True,
        )
    return matrix, names


def _target_array(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(record["target_value"]) for record in records], dtype=np.float32)


def _tree_node_budget(estimator: Any) -> int:
    estimators = getattr(estimator, "estimators_", None)
    if estimators is not None:
        total = 0
        for item in np.ravel(estimators):
            tree = getattr(item, "tree_", None)
            if tree is not None:
                total += int(tree.node_count)
        if total > 0:
            return int(total)
    max_iter = int(getattr(estimator, "max_iter", 0) or 0)
    max_leaf_nodes = int(getattr(estimator, "max_leaf_nodes", 0) or 0)
    if max_iter > 0 and max_leaf_nodes > 0:
        return int(max_iter * max(1, 2 * max_leaf_nodes - 1))
    return 0


def estimate_trainable_params(model: Any) -> int:
    attr = getattr(model, "trainable_params", None)
    if attr is not None:
        return int(attr)
    if isinstance(model, LearnedClassicalControllerValueModel):
        return int(model.trainable_params)
    return 0


def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    abs_errors = np.abs(errors)
    mse = float(np.mean(errors * errors))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(abs_errors)),
        "p50_abs_error": float(np.percentile(abs_errors, 50)),
        "p95_abs_error": float(np.percentile(abs_errors, 95)),
        "max_abs_error": float(np.max(abs_errors)),
    }


def _error_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return outlier and target-bucket diagnostics for model-search reports."""

    y_true64 = np.asarray(y_true, dtype=np.float64)
    y_pred64 = np.asarray(y_pred, dtype=np.float64)
    abs_errors = np.abs(y_pred64 - y_true64)
    abs_target = np.abs(y_true64)

    out: dict[str, Any] = {
        "num_abs_error_gt_0p05": int(np.sum(abs_errors > 0.05)),
        "num_abs_error_gt_0p10": int(np.sum(abs_errors > 0.10)),
        "num_abs_error_gt_0p50": int(np.sum(abs_errors > 0.50)),
        "num_abs_error_gt_1p00": int(np.sum(abs_errors > 1.00)),
    }

    buckets = {
        "target_abs_eq_0": abs_target <= TARGET_EPS,
        "target_abs_le_0p05": abs_target <= 0.05,
        "target_abs_0p05_1": (abs_target > 0.05) & (abs_target < 1.0),
        "target_abs_1_2": (abs_target >= 1.0) & (abs_target < 2.0),
        "target_abs_2_3": (abs_target >= 2.0) & (abs_target < 3.0),
        "target_abs_ge_3": abs_target >= 3.0,
    }
    for name, mask in buckets.items():
        count = int(np.sum(mask))
        prefix = f"bucket_{name}"
        out[f"{prefix}_count"] = count
        if count <= 0:
            continue
        bucket_errors = abs_errors[mask]
        out[f"{prefix}_mae"] = float(np.mean(bucket_errors))
        out[f"{prefix}_p95_abs_error"] = float(np.percentile(bucket_errors, 95))
        out[f"{prefix}_max_abs_error"] = float(np.max(bucket_errors))
    return out


def train_learned_classical_model(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    cfg: Any,
    output_dir: str | Path,
    state_loader: Any | None = None,
) -> dict[str, Any]:
    """Fit the default learned classical controller value surrogate."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extra_cfg = getattr(cfg, "extra_config", {}) or {}
    backend = str(extra_cfg.get("classical_backend", "bellman_shaped"))
    feature_source = str(extra_cfg.get("classical_feature_source", "model_inputs"))
    top_k = int(extra_cfg.get("classical_top_k_requests", DEFAULT_TOP_K_REQUESTS))
    max_params = int(extra_cfg.get("max_trainable_params", DEFAULT_MODEL_BUDGET))

    t0 = time.time()
    if feature_source == "model_inputs":
        x_train, feature_names = build_or_load_model_input_feature_matrix(
            train_records,
            state_loader=state_loader,
            model_cfg=cfg,
            extra_config=dict(extra_cfg),
            output_dir=output,
            split_name="train",
            player="controller",
        )
    elif feature_source == "record_snapshot":
        x_train, feature_names = build_feature_matrix(train_records, top_k_requests=top_k)
    else:
        raise ValueError(
            "classical_feature_source must be 'model_inputs' or 'record_snapshot', "
            f"got {feature_source!r}"
        )
    y_train = _target_array(train_records)
    if feature_source == "model_inputs":
        x_eval, _ = build_or_load_model_input_feature_matrix(
            eval_records,
            state_loader=state_loader,
            model_cfg=cfg,
            extra_config=dict(extra_cfg),
            output_dir=output,
            split_name="eval",
            player="controller",
        )
    else:
        x_eval, _ = build_feature_matrix(eval_records, top_k_requests=top_k)
    y_eval = _target_array(eval_records)
    feature_seconds = time.time() - t0

    estimators: dict[str, Any] = {}
    if backend in {"extra_trees", "hybrid_extra_hgb", "bellman_shaped"}:
        estimators["extra"] = ExtraTreesRegressor(
            n_estimators=int(extra_cfg.get("extra_n_estimators", 64)),
            max_leaf_nodes=int(extra_cfg.get("extra_max_leaf_nodes", 672)),
            min_samples_leaf=int(extra_cfg.get("extra_min_samples_leaf", 1)),
            max_features=float(extra_cfg.get("extra_max_features", 0.85)),
            bootstrap=False,
            random_state=int(extra_cfg.get("random_state", 2027)),
            n_jobs=int(extra_cfg.get("n_jobs", -1)),
        )
    if backend in {"hist_gradient_boosting", "hybrid_extra_hgb", "bellman_shaped"}:
        estimators["hgb"] = HistGradientBoostingRegressor(
            max_iter=int(extra_cfg.get("hgb_max_iter", 600)),
            max_leaf_nodes=int(extra_cfg.get("hgb_max_leaf_nodes", 15)),
            learning_rate=float(extra_cfg.get("hgb_learning_rate", 0.04)),
            l2_regularization=float(extra_cfg.get("hgb_l2_regularization", 0.0)),
            loss=str(extra_cfg.get("hgb_loss", "squared_error")),
            max_bins=int(extra_cfg.get("hgb_max_bins", 255)),
            early_stopping=False,
            random_state=int(extra_cfg.get("random_state", 2027)) + 17,
        )
    if backend == "bellman_shaped":
        estimators["zero_gate"] = HistGradientBoostingClassifier(
            max_iter=int(extra_cfg.get("zero_gate_max_iter", 300)),
            max_leaf_nodes=int(extra_cfg.get("zero_gate_max_leaf_nodes", 15)),
            learning_rate=float(extra_cfg.get("zero_gate_learning_rate", 0.04)),
            l2_regularization=float(extra_cfg.get("zero_gate_l2_regularization", 0.0)),
            max_bins=int(extra_cfg.get("zero_gate_max_bins", 255)),
            early_stopping=False,
            random_state=int(extra_cfg.get("random_state", 2027)) + 29,
        )
        estimators["residual"] = HistGradientBoostingRegressor(
            max_iter=int(extra_cfg.get("residual_max_iter", 800)),
            max_leaf_nodes=int(extra_cfg.get("residual_max_leaf_nodes", 21)),
            learning_rate=float(extra_cfg.get("residual_learning_rate", 0.03)),
            l2_regularization=float(extra_cfg.get("residual_l2_regularization", 0.0)),
            loss=str(extra_cfg.get("residual_loss", "squared_error")),
            max_bins=int(extra_cfg.get("residual_max_bins", 255)),
            early_stopping=False,
            random_state=int(extra_cfg.get("random_state", 2027)) + 41,
        )
    if not estimators:
        raise ValueError(f"unknown classical backend: {backend!r}")

    fit_start = time.time()
    if backend == "bellman_shaped":
        estimators["extra"].fit(x_train, y_train)
        gc.collect()
        estimators["hgb"].fit(x_train, y_train)
        gc.collect()

        blend_weight_hgb = float(extra_cfg.get("blend_weight_hgb", 0.25))
        base_extra = np.asarray(estimators["extra"].predict(x_train), dtype=np.float64)
        base_hgb = np.asarray(estimators["hgb"].predict(x_train), dtype=np.float64)
        base_train = np.minimum(
            (1.0 - blend_weight_hgb) * base_extra + blend_weight_hgb * base_hgb,
            0.0,
        )

        zero_label_eps = float(extra_cfg.get("zero_label_eps", 0.05))
        zero_labels = np.asarray(np.abs(y_train) <= zero_label_eps, dtype=np.int8)
        estimators["zero_gate"].fit(x_train, zero_labels)
        gc.collect()

        zero_threshold = float(extra_cfg.get("zero_threshold", 0.70))
        zero_proba = np.asarray(estimators["zero_gate"].predict_proba(x_train), dtype=np.float64)
        zero_classes = list(getattr(estimators["zero_gate"], "classes_", []))
        zero_idx = zero_classes.index(1) if 1 in zero_classes else None
        if zero_idx is not None:
            zero_prob_train = zero_proba[:, int(zero_idx)]
            base_train = base_train.copy()
            base_train[zero_prob_train >= zero_threshold] = 0.0

        residual_target = np.asarray(y_train, dtype=np.float64) - base_train
        residual_abs = np.abs(residual_target)
        residual_weights = (
            1.0
            + float(extra_cfg.get("residual_error_weight", 6.0))
            * np.minimum(1.0, residual_abs / max(1e-9, float(extra_cfg.get("residual_error_scale", 0.20))))
            + float(extra_cfg.get("residual_large_target_weight", 2.0))
            * (np.abs(y_train) >= float(extra_cfg.get("residual_large_target_abs", 3.0)))
        )
        estimators["residual"].fit(
            x_train,
            residual_target.astype(np.float32),
            sample_weight=residual_weights.astype(np.float32),
        )
        gc.collect()
    else:
        for estimator in estimators.values():
            estimator.fit(x_train, y_train)
            gc.collect()
    fit_seconds = time.time() - fit_start

    trainable_params = int(sum(_tree_node_budget(estimator) for estimator in estimators.values()))
    if trainable_params > max_params:
        raise ValueError(
            f"classical model has estimated {trainable_params} scalar parameters, "
            f"above budget {max_params}"
        )

    model = LearnedClassicalControllerValueModel(
        backend=backend,
        estimators=estimators,
        feature_names=feature_names,
        feature_config={
            "feature_source": str(feature_source),
            "top_k_requests": int(top_k),
        },
        trainable_params=trainable_params,
        blend_weight_hgb=float(extra_cfg.get("blend_weight_hgb", 0.35 if backend == "bellman_shaped" else 0.10)),
        zero_threshold=float(extra_cfg.get("zero_threshold", 0.70)),
        zero_after_residual_threshold=float(extra_cfg.get("zero_after_residual_threshold", 1.01)),
        residual_scale=float(extra_cfg.get("residual_scale", 1.0)),
        model_name=str(getattr(cfg, "model_name", "learned_classical_controller_value")),
    )

    train_pred = model.predict_matrix(x_train)
    eval_pred = model.predict_matrix(x_eval)
    train_metrics = _error_metrics(y_train, train_pred)
    eval_metrics = _error_metrics(y_eval, eval_pred)
    train_diagnostics = _error_diagnostics(y_train, train_pred)
    eval_diagnostics = _error_diagnostics(y_eval, eval_pred)

    extra_estimator = estimators.get("extra")
    if extra_estimator is not None and hasattr(extra_estimator, "n_jobs"):
        try:
            extra_estimator.n_jobs = 1
        except Exception:
            pass

    model = LearnedClassicalControllerValueModel(
        backend=model.backend,
        estimators=model.estimators,
        feature_names=model.feature_names,
        feature_config=model.feature_config,
        trainable_params=model.trainable_params,
        blend_weight_hgb=model.blend_weight_hgb,
        zero_threshold=model.zero_threshold,
        zero_after_residual_threshold=model.zero_after_residual_threshold,
        residual_scale=model.residual_scale,
        model_name=model.model_name,
        cached_predictions={
            "train": [float(x) for x in train_pred.tolist()],
            "eval": [float(x) for x in eval_pred.tolist()],
        },
    )

    model_path = output / "learned_classical_controller_value.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model.model_name,
        "backend": model.backend,
        "trainable_params_estimate": int(model.trainable_params),
        "feature_count": int(len(feature_names)),
        "feature_config": dict(model.feature_config),
        "uses_neural_network": False,
        "uses_target_leakage": False,
        "num_train_records": int(len(train_records)),
        "num_eval_records": int(len(eval_records)),
        "feature_seconds": float(feature_seconds),
        "fit_seconds": float(fit_seconds),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "train_diagnostics": train_diagnostics,
        "eval_diagnostics": eval_diagnostics,
        "description": (
            "Learned non-neural, Bellman-shaped tree surrogate for controller value. "
            "The model predicts state value directly from simulator snapshot/stats "
            "using a base regressor, zero-value gate, and residual tail corrector."
        ),
    }
    metadata_path = output / "learned_classical_controller_value.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "model": model,
        "best_checkpoint_path": model_path,
        "train_metrics": [
            {
                "backend": str(model.backend),
                "epoch": 0,
                "model_name": str(model.model_name),
                "trainable_params": int(model.trainable_params),
                "num_train_records": int(len(train_records)),
                "num_eval_records": int(len(eval_records)),
                "feature_seconds": float(feature_seconds),
                "fit_seconds": float(fit_seconds),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"eval_{k}": float(v) for k, v in eval_metrics.items()},
                **{f"train_diag_{k}": float(v) for k, v in train_diagnostics.items() if isinstance(v, (int, float))},
                **{f"eval_diag_{k}": float(v) for k, v in eval_diagnostics.items() if isinstance(v, (int, float))},
            }
        ],
    }


def predict_learned_classical_values(
    *,
    model: LearnedClassicalControllerValueModel,
    records: list[dict[str, Any]],
    split_name: str | None = None,
    state_loader: Any | None = None,
) -> list[float]:
    if not isinstance(model, LearnedClassicalControllerValueModel):
        raise TypeError(f"expected LearnedClassicalControllerValueModel, got {type(model)!r}")
    if split_name is not None:
        cached = model.cached_predictions.get(str(split_name))
        if cached is not None and len(cached) == len(records):
            return [float(x) for x in cached]
        if str(split_name).startswith("train"):
            cached = model.cached_predictions.get("train")
            if cached is not None and len(cached) == len(records):
                return [float(x) for x in cached]
        if str(split_name).startswith("eval"):
            cached = model.cached_predictions.get("eval")
            if cached is not None and len(cached) == len(records):
                return [float(x) for x in cached]
    if str(model.feature_config.get("feature_source", "record_snapshot")) == "model_inputs":
        x, _ = build_model_input_feature_matrix(
            records,
            state_loader=state_loader,
            player="controller",
        )
        return [float(x) for x in model.predict_matrix(x).tolist()]
    x, _ = build_feature_matrix(
        records,
        top_k_requests=int(model.feature_config.get("top_k_requests", DEFAULT_TOP_K_REQUESTS)),
    )
    return [float(x) for x in model.predict_matrix(x).tolist()]


def predict_learned_classical_value_from_inputs(
    *,
    model: LearnedClassicalControllerValueModel,
    inputs: Any,
    player: str,
    device: Any | None = None,
) -> float:
    """Predict one controller-perspective value from MCTS `ModelInputs`."""

    value, _priors = model.infer_from_inputs(inputs, player, device=device)
    return float(value)


def predict_learned_classical_values_from_inputs_batch(
    *,
    model: LearnedClassicalControllerValueModel,
    inputs_list: Sequence[Any],
) -> list[float]:
    """Vectorized bootstrap prediction from many child-state `ModelInputs`."""

    if not isinstance(model, LearnedClassicalControllerValueModel):
        raise TypeError(f"expected LearnedClassicalControllerValueModel, got {type(model)!r}")
    if str(model.feature_config.get("feature_source", "")) != "model_inputs":
        raise RuntimeError(
            "batched bootstrap requires feature_source='model_inputs'; "
            f"got {model.feature_config.get('feature_source')!r}"
        )
    rows = [extract_model_input_feature_vector(inputs)[0] for inputs in inputs_list]
    if not rows:
        return []
    x = np.asarray(rows, dtype=np.float32)
    return [float(v) for v in model.predict_matrix(x).tolist()]


def train_symbolic_bellman_model(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Return the diagnostic zero-parameter symbolic evaluator."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = SymbolicBellmanControllerValueModel()
    metadata_path = output / "symbolic_bellman_model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model.model_name,
                "trainable_params": int(model.trainable_params),
                "uses_neural_network": False,
                "uses_target_leakage": False,
                "num_train_records": int(len(train_records)),
                "num_eval_records": int(len(eval_records)),
                "description": "Diagnostic only: recomputes one-step Bellman target from simulator state.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "best_checkpoint_path": metadata_path,
        "train_metrics": [
            {
                "backend": "symbolic_bellman_controller_value",
                "epoch": 0,
                "model_name": model.model_name,
                "trainable_params": 0,
                "num_train_records": int(len(train_records)),
                "num_eval_records": int(len(eval_records)),
            }
        ],
    }


def predict_symbolic_bellman_values(
    *,
    model: SymbolicBellmanControllerValueModel,
    records: list[dict[str, Any]],
    state_loader: Any,
) -> list[float]:
    """Diagnostic Bellman recomputation path."""

    if not isinstance(model, SymbolicBellmanControllerValueModel):
        raise TypeError(f"expected SymbolicBellmanControllerValueModel, got {type(model)!r}")
    mcts = getattr(state_loader, "mcts", None)
    if mcts is None:
        raise AttributeError("state_loader must expose `.mcts` for symbolic predictions")

    preds: list[float] = []
    for index, record in enumerate(records, start=1):
        state = state_loader(record)
        out = mcts.search_dnn(
            dnn_model=None,
            rootState=state,
            root_player="controller",
            game_id=0,
            root_id=int(record.get("root_id", 0)),
            root_node_id_override=record.get("root_node_id_override", None),
            root_depth=int(record.get("root_depth", 0)),
            model_version=0,
            use_model_bootstrap=False,
            one_step_value_mode=True,
        )
        preds.append(float(out.best_action_value))
        if index % 2048 == 0:
            clear_fn = getattr(mcts, "clear_search_state", None)
            if callable(clear_fn):
                clear_fn(drop_scratch=True)
            gc.collect()
    clear_fn = getattr(mcts, "clear_search_state", None)
    if callable(clear_fn):
        clear_fn(drop_scratch=True)
    return preds
