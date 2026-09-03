"""Adaptive rollout-horizon calculation and runtime state."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vidur.AlphaGoZero.config import (
    AGZ_ADAPTIVE_ROLLOUT_HORIZON,
    AGZ_DISCOUNT_FACTOR,
    AGZ_ROLLOUT_HORIZON_SEC,
    AGZ_ROLLOUT_HORIZON_TICK_SEC,
    AGZ_ROLLOUT_MAX_HORIZON_SEC,
    AGZ_ROLLOUT_REFERENCE_STEP_SEC,
    AGZ_ROLLOUT_VALUE_ERROR_THRESHOLD,
)
from vidur.AlphaGoZero.durable_transfer import atomic_write_json, utc_now


RUNTIME_SEARCH_CONFIG_FILENAME = "runtime_search_config.json"


@dataclass(frozen=True)
class RolloutHorizonCalculation:
    controller_p95_abs_error: float
    value_error_threshold: float
    discount_factor: float
    reference_step_sec: float
    max_horizon_sec: float
    tick_sec: float
    raw_horizon_sec: float
    calculated_horizon_sec: float
    rounded_horizon_sec: float
    discounted_error_at_rounded_horizon: float

    def to_json(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _finite_positive(name: str, value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return resolved


def round_to_nearest_tick(value: float, tick_sec: float) -> float:
    """Round a non-negative duration to the nearest tick, with ties rounded up."""

    duration = float(value)
    tick = _finite_positive("tick_sec", tick_sec)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"value must be finite and non-negative, got {value!r}")
    ticks = math.floor(duration / tick + 0.5)
    return float(round(ticks * tick, 12))


def calculate_rollout_horizon(
    controller_p95_abs_error: float,
    *,
    value_error_threshold: float,
    discount_factor: float,
    reference_step_sec: float,
    max_horizon_sec: float,
    tick_sec: float,
) -> RolloutHorizonCalculation:
    """Calculate the capped and adversary-tick-rounded next-cycle horizon."""

    p95 = float(controller_p95_abs_error)
    if not math.isfinite(p95) or p95 < 0.0:
        raise ValueError(
            "controller_p95_abs_error must be finite and non-negative, "
            f"got {controller_p95_abs_error!r}"
        )
    threshold = _finite_positive("value_error_threshold", value_error_threshold)
    gamma = float(discount_factor)
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError(f"discount_factor must be in (0, 1), got {discount_factor!r}")
    reference = _finite_positive("reference_step_sec", reference_step_sec)
    maximum = _finite_positive("max_horizon_sec", max_horizon_sec)
    tick = _finite_positive("tick_sec", tick_sec)

    if p95 <= threshold:
        raw = 0.0
    else:
        raw = reference * math.log(threshold / p95) / math.log(gamma)
    calculated = min(maximum, max(0.0, float(raw)))
    rounded = min(maximum, round_to_nearest_tick(calculated, tick))
    rounded = float(round(max(0.0, rounded), 12))
    residual = (
        0.0
        if p95 == 0.0
        else p95 * math.exp(math.log(gamma) * (rounded / reference))
    )
    return RolloutHorizonCalculation(
        controller_p95_abs_error=p95,
        value_error_threshold=threshold,
        discount_factor=gamma,
        reference_step_sec=reference,
        max_horizon_sec=maximum,
        tick_sec=tick,
        raw_horizon_sec=float(raw),
        calculated_horizon_sec=float(calculated),
        rounded_horizon_sec=rounded,
        discounted_error_at_rounded_horizon=float(residual),
    )


def configured_rollout_horizon(
    controller_p95_abs_error: float,
) -> RolloutHorizonCalculation:
    return calculate_rollout_horizon(
        controller_p95_abs_error,
        value_error_threshold=float(AGZ_ROLLOUT_VALUE_ERROR_THRESHOLD),
        discount_factor=float(AGZ_DISCOUNT_FACTOR),
        reference_step_sec=float(AGZ_ROLLOUT_REFERENCE_STEP_SEC),
        max_horizon_sec=float(AGZ_ROLLOUT_MAX_HORIZON_SEC),
        tick_sec=float(AGZ_ROLLOUT_HORIZON_TICK_SEC),
    )


def runtime_search_config_path(root: Path) -> Path:
    return Path(root) / RUNTIME_SEARCH_CONFIG_FILENAME


def initial_runtime_search_config() -> dict[str, Any]:
    initial = min(
        float(AGZ_ROLLOUT_MAX_HORIZON_SEC),
        max(0.0, float(AGZ_ROLLOUT_HORIZON_SEC)),
    )
    return {
        "adaptive_rollout_horizon": bool(AGZ_ADAPTIVE_ROLLOUT_HORIZON),
        "active_rollout_horizon_sec": float(initial),
        "source_candidate_version": 100,
        "target_candidate_version": 101,
        "controller_p95_abs_error": None,
        "value_error_threshold": float(AGZ_ROLLOUT_VALUE_ERROR_THRESHOLD),
        "discount_factor": float(AGZ_DISCOUNT_FACTOR),
        "reference_step_sec": float(AGZ_ROLLOUT_REFERENCE_STEP_SEC),
        "max_horizon_sec": float(AGZ_ROLLOUT_MAX_HORIZON_SEC),
        "tick_sec": float(AGZ_ROLLOUT_HORIZON_TICK_SEC),
        "raw_horizon_sec": float(initial),
        "calculated_horizon_sec": float(initial),
        "rounded_horizon_sec": float(initial),
        "updated_at_utc": utc_now(),
    }


def ensure_runtime_search_config(root: Path) -> dict[str, Any]:
    path = runtime_search_config_path(root)
    if path.exists():
        return read_runtime_search_config(root)
    payload = initial_runtime_search_config()
    atomic_write_json(path, payload)
    return payload


def read_runtime_search_config(root: Path) -> dict[str, Any]:
    path = runtime_search_config_path(root)
    if not path.exists():
        return initial_runtime_search_config()
    payload = json.loads(path.read_text(encoding="utf-8"))
    horizon = float(payload.get("active_rollout_horizon_sec", AGZ_ROLLOUT_HORIZON_SEC))
    maximum = float(payload.get("max_horizon_sec", AGZ_ROLLOUT_MAX_HORIZON_SEC))
    if not math.isfinite(horizon) or not 0.0 <= horizon <= maximum:
        raise ValueError(f"invalid active rollout horizon in {path}: {horizon!r}")
    return payload


def active_rollout_horizon_sec(root: Path) -> float:
    if not bool(AGZ_ADAPTIVE_ROLLOUT_HORIZON):
        return float(AGZ_ROLLOUT_HORIZON_SEC)
    payload = read_runtime_search_config(root)
    return float(payload["active_rollout_horizon_sec"])


def activate_manifest_horizon(
    root: Path,
    *,
    candidate_version: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    rounded = float(manifest["next_rollout_horizon_rounded_sec"])
    maximum = float(manifest["rollout_max_horizon_sec"])
    if not math.isfinite(rounded) or not 0.0 <= rounded <= maximum:
        raise ValueError(
            f"candidate v{candidate_version} has invalid next rollout horizon {rounded!r}"
        )
    payload = {
        "adaptive_rollout_horizon": bool(AGZ_ADAPTIVE_ROLLOUT_HORIZON),
        "active_rollout_horizon_sec": rounded,
        "source_candidate_version": int(candidate_version),
        "target_candidate_version": int(candidate_version) + 1,
        "controller_p95_abs_error": float(
            manifest["rollout_horizon_source_controller_p95_abs_error"]
        ),
        "value_error_threshold": float(manifest["rollout_value_error_threshold"]),
        "discount_factor": float(manifest["rollout_discount_factor"]),
        "reference_step_sec": float(manifest["rollout_reference_step_sec"]),
        "max_horizon_sec": maximum,
        "tick_sec": float(manifest["rollout_horizon_tick_sec"]),
        "raw_horizon_sec": float(manifest["next_rollout_horizon_raw_sec"]),
        "calculated_horizon_sec": float(
            manifest["next_rollout_horizon_calculated_sec"]
        ),
        "rounded_horizon_sec": rounded,
        "discounted_error_at_rounded_horizon": float(
            manifest["next_rollout_discounted_error"]
        ),
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(runtime_search_config_path(root), payload)
    return payload
