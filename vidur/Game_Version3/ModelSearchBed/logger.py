"""Logging helpers for GV3 ModelSearchBed Bellman convergence runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


def compute_error_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    abs_error_threshold: float = 1.0,
) -> dict[str, float | int]:
    """Return aggregate real-scale prediction-error statistics."""

    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions/targets length mismatch: {len(predictions)} != {len(targets)}"
        )
    if not predictions:
        raise ValueError("cannot compute metrics for an empty prediction set")

    errors = [float(pred) - float(target) for pred, target in zip(predictions, targets)]
    abs_errors = [abs(x) for x in errors]
    squared_errors = [x * x for x in errors]
    n = len(errors)

    def mean(xs: Sequence[float]) -> float:
        return float(sum(float(x) for x in xs) / float(len(xs)))

    def percentile(xs: Sequence[float], p: float) -> float:
        if not xs:
            return float("nan")
        ordered = sorted(float(x) for x in xs)
        idx = min(len(ordered) - 1, max(0, int(math.ceil(float(p) * len(ordered))) - 1))
        return float(ordered[idx])

    mse = mean(squared_errors)
    mae = mean(abs_errors)
    return {
        "num_samples": int(n),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mae),
        "p95_squared_error": percentile(squared_errors, 0.95),
        "p95_abs_error": percentile(abs_errors, 0.95),
        "max_abs_error": float(max(abs_errors)),
        "num_abs_error_ge_1": int(
            sum(1 for x in abs_errors if float(x) >= float(abs_error_threshold))
        ),
    }


def write_prediction_results_csv(
    path: str | Path,
    *,
    predictions: Sequence[float],
    targets: Sequence[float],
    sample_numbers: Sequence[int] | None = None,
) -> Path:
    """Write per-sample prediction-vs-target rows."""

    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions/targets length mismatch: {len(predictions)} != {len(targets)}"
        )
    if sample_numbers is not None and len(sample_numbers) != len(predictions):
        raise ValueError(
            f"sample_numbers/predictions length mismatch: {len(sample_numbers)} != {len(predictions)}"
        )

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_number",
                "model_prediction",
                "true_MCTS_DNN_value",
            ],
        )
        writer.writeheader()
        for idx, (pred, target) in enumerate(zip(predictions, targets)):
            sample_number = int(sample_numbers[idx]) if sample_numbers is not None else int(idx)
            writer.writerow(
                {
                    "sample_number": int(sample_number),
                    "model_prediction": float(pred),
                    "true_MCTS_DNN_value": float(target),
                }
            )
    return out


def write_summary_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    """Write aggregate metric rows to a CSV file."""

    materialized = [dict(row) for row in rows]
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        out.write_text("", encoding="utf-8")
        return out

    fieldnames: list[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    return out


def append_summary_csv(
    path: str | Path,
    row: dict[str, Any],
) -> Path:
    """Append one aggregate metric row to a CSV file."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return out


def write_split_indices(
    path: str | Path,
    *,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
) -> Path:
    """Persist deterministic train/eval split indices."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_indices": [int(x) for x in train_indices],
        "eval_indices": [int(x) for x in eval_indices],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with stable formatting."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out
