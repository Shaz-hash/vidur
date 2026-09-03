"""Fail-closed validation for the uncalibrated FlashInfer Vidur profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_TOKENS = list(range(128, 4097, 128))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_profile(path: Path) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    profile = {
        int(row["prefill_tokens"]): float(row["prefill_time_seconds"])
        for row in rows
    }
    if sorted(profile) != EXPECTED_TOKENS or len(rows) != len(EXPECTED_TOKENS):
        raise AssertionError("profile must contain exactly 128, 256, ..., 4096 once")
    if any(not math.isfinite(value) or value <= 0.0 for value in profile.values()):
        raise AssertionError("all profile times must be finite and positive")
    return profile


def _validate_predictions(path: Path, profile: dict[int, float]) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    one_request = {
        int(row["prefill_tokens_per_request"]): row
        for row in rows
        if int(row["request_count"]) == 1
    }
    if sorted(one_request) != EXPECTED_TOKENS:
        raise AssertionError("prediction CSV is missing one-request profile points")

    max_error_sec = 0.0
    for tokens in EXPECTED_TOKENS:
        row = one_request[tokens]
        if str(row["calibration_applied"]).strip().lower() not in {"false", "0"}:
            raise AssertionError(f"calibration is enabled at {tokens} tokens")
        if not math.isclose(float(row["calibration_factor"]), 1.0, abs_tol=0.0):
            raise AssertionError(f"calibration factor is not 1.0 at {tokens} tokens")
        predicted_sec = float(row["vidur_model_ms"]) / 1000.0
        error = abs(profile[tokens] - predicted_sec)
        max_error_sec = max(max_error_sec, error)
        if not math.isclose(profile[tokens], predicted_sec, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"profile mismatch at {tokens}: {profile[tokens]} != {predicted_sec}"
            )
    return max_error_sec


def _validate_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = dict(manifest.get("config") or {})
    measurement = dict(manifest.get("measurement") or {})
    if config.get("calibration") is not None:
        raise AssertionError("manifest config contains calibration")
    if measurement.get("calibration_applied") is not False:
        raise AssertionError("manifest does not explicitly disable calibration")
    if config.get("attention_backend") != "FLASHINFER":
        raise AssertionError("attention backend is not FLASHINFER")
    if str(config.get("vllm_version")) != "0.26.0":
        raise AssertionError("unexpected vLLM version")
    if str(config.get("flashinfer_version")) != "0.6.14":
        raise AssertionError("unexpected FlashInfer version")


def _validate_cache(path: Path) -> dict[str, int]:
    model_files = list(path.glob("*.pkl"))
    prediction_files = list(path.glob("*_pred.dat"))
    metadata_files = list(path.glob("*_pred.meta.json"))
    if not model_files or not prediction_files or len(prediction_files) != len(metadata_files):
        raise AssertionError("predictor cache is incomplete")
    return {
        "model_files": len(model_files),
        "prediction_files": len(prediction_files),
        "metadata_files": len(metadata_files),
    }


def _load_native(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("mcts_native_gv2_profile_check", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_native_lookup(path: Path, profile: dict[int, float]) -> float:
    native = _load_native(path)
    tokens = sorted(profile)
    times = [profile[token] for token in tokens]
    queries = sorted(set(tokens + [1, 127, 129, 255, 257, 4095, 4097]))
    native_values = native.debug_prefill_profile_lookup(tokens, times, queries)
    max_error = 0.0
    for query, native_value in zip(queries, native_values):
        nearest = min(tokens, key=lambda token: abs(token - query))
        error = abs(float(native_value) - profile[nearest])
        max_error = max(max_error, error)
        if error > 1e-12:
            raise AssertionError(f"native lookup mismatch for {query}: {error}")
    return max_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--native-module", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = _read_profile(args.profile)
    result: dict[str, Any] = {
        "profile": str(args.profile),
        "profile_sha256": _sha256(args.profile),
        "profile_rows": len(profile),
        "prefill_128_sec": profile[128],
        "prefill_4096_sec": profile[4096],
        "cache": _validate_cache(args.cache_dir),
        "calibration_applied": False,
    }
    if args.predictions:
        result["prediction_max_abs_error_sec"] = _validate_predictions(
            args.predictions, profile
        )
    if args.manifest:
        _validate_manifest(args.manifest)
        result["manifest_sha256"] = _sha256(args.manifest)
    if args.native_module:
        result["native_lookup_max_abs_error_sec"] = _validate_native_lookup(
            args.native_module, profile
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

