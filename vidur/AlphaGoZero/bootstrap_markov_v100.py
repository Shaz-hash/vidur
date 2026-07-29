"""Bootstrap a Markov-v2 v100 value bundle without inventing state labels.

Legacy replay stores only the 226-D representation, so it cannot be converted
back into the complete Markov state.  This utility preserves the distilled v100
policy models and initializes both Markov value heads to the legacy value
model's measured mean on v100 replay.  Subsequent candidates warm-start all
four networks and train the value heads on genuine Markov-v2 replay.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from vidur.AlphaGoZero.dnn_models import (
    MarkovValueDeepSet,
    PolicyRankMLP,
    export_dnn_to_native,
    save_dnn_model,
    write_artifact_metadata,
)
from vidur.AlphaGoZero.durable_transfer import atomic_write_json, utc_now


VALUE_CONFIG = "dnn_value_markov_deepset_192_v2"
POLICY_CONFIG = "dnn_policy_rank_192_v1"


def _v100_replay_paths(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    manifests = sorted(
        (Path(source_root) / "global_replay" / "partitions").glob(
            "*/*/partition_manifest.json"
        )
    )
    for manifest_path in manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        role = str(data.get("partition_role", "")).strip().lower()
        legacy_version = int(data.get("model_version", 0) or 0)
        if role == "controller":
            version = int(data.get("controller_model_version", legacy_version) or legacy_version)
        elif role == "adversary":
            version = int(data.get("adversary_model_version", legacy_version) or legacy_version)
        else:
            continue
        replay = manifest_path.parent / "replay_target_runtime_feature_complete.csv"
        if version == 100 and replay.is_file():
            paths.append(replay)
    return paths


def _reservoir_legacy_features(
    paths: list[Path],
    *,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = random.Random(int(seed))
    reservoir: list[np.ndarray] = []
    seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("feature_complete", "0")).strip().lower() not in {
                    "1",
                    "true",
                }:
                    continue
                raw = str(row.get("state_features_json", "")).strip()
                if not raw:
                    continue
                values = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
                if values.size != 226 or not np.all(np.isfinite(values)):
                    raise ValueError(f"invalid legacy value row in {path}: {values.shape}")
                seen += 1
                if len(reservoir) < int(limit):
                    reservoir.append(values)
                    continue
                selected = rng.randrange(seen)
                if selected < int(limit):
                    reservoir[selected] = values
    if not reservoir:
        raise RuntimeError("no feature-complete v100 legacy replay rows found")
    return np.stack(reservoir).astype(np.float32, copy=False), seen


def _initialize_constant_value(
    *,
    role: str,
    target: float,
    source_model: Path,
    source_root: Path,
) -> MarkovValueDeepSet:
    target = float(np.clip(target, -49.99, -0.01))
    normalized = target / 25.0 + 1.0
    if not -1.0 < normalized < 1.0:
        raise ValueError(f"invalid normalized constant target: {normalized}")
    model = MarkovValueDeepSet(role=role)
    with torch.no_grad():
        model.head2.weight.zero_()
        model.head2.bias.fill_(math.atanh(normalized))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.optimizer_state = optimizer.state_dict()
    model.training_metadata = {
        "bootstrap_version": 100,
        "bootstrap_strategy": "constant_legacy_v100_teacher_mean",
        "bootstrap_constant_cost": target,
        "legacy_value_model": str(source_model),
        "source_replay": str(source_root),
        "feature_schema": "markov_v2",
        "target_perspective": "controller",
        "incremental_update": False,
    }
    model.eval()
    return model


def _write_model(model: Any, directory: Path, *, tag: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_dnn_model(model, directory / "model.joblib")
    export_dnn_to_native(model, directory / "native_model.tsv", model_tag=tag)
    write_artifact_metadata(model, directory / "metadata.json")


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    source_model_root = Path(args.source_model_root)
    source_value_path = (
        source_model_root / "controller_value" / "dnn_value_residual_192_v1" / "model.joblib"
    )
    source_controller_policy = (
        source_model_root / "controller_prior" / POLICY_CONFIG / "model.joblib"
    )
    source_adversary_policy = (
        source_model_root / "adversary_prior" / POLICY_CONFIG / "model.joblib"
    )
    for required in (
        source_value_path,
        source_controller_policy,
        source_adversary_policy,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    replay_paths = _v100_replay_paths(Path(args.source_replay_root))
    legacy_features, available = _reservoir_legacy_features(
        replay_paths,
        limit=int(args.teacher_sample_rows),
        seed=int(args.seed),
    )
    teacher = joblib.load(source_value_path)
    teacher_values = np.asarray(teacher.predict(legacy_features), dtype=np.float32)
    if teacher_values.shape != (legacy_features.shape[0],):
        raise ValueError(f"unexpected teacher shape: {teacher_values.shape}")
    finite = teacher_values[np.isfinite(teacher_values)]
    if finite.size != teacher_values.size:
        raise ValueError("legacy v100 value model returned non-finite predictions")
    clipped = np.clip(finite, -50.0, 0.0)
    teacher_mean = float(np.mean(clipped))

    controller_value = _initialize_constant_value(
        role="controller",
        target=teacher_mean,
        source_model=source_value_path,
        source_root=Path(args.source_replay_root),
    )
    adversary_value = copy.deepcopy(controller_value)
    adversary_value.role = "adversary"
    adversary_value.training_metadata = dict(controller_value.training_metadata)
    adversary_value.training_metadata["copied_from_controller_value_v100"] = True

    controller_policy = joblib.load(source_controller_policy)
    adversary_policy = joblib.load(source_adversary_policy)
    if not isinstance(controller_policy, PolicyRankMLP) or controller_policy.role != "controller":
        raise TypeError("source controller policy is not a controller PolicyRankMLP")
    if not isinstance(adversary_policy, PolicyRankMLP) or adversary_policy.role != "adversary":
        raise TypeError("source adversary policy is not an adversary PolicyRankMLP")

    output_root = Path(args.output_root)
    model_root = output_root / "models" / "Model_Version100"
    if model_root.exists():
        if not bool(args.replace):
            raise FileExistsError(model_root)
        shutil.rmtree(model_root)
    controller_value_dir = model_root / "controller_value" / VALUE_CONFIG
    adversary_value_dir = model_root / "adversary_value" / VALUE_CONFIG
    controller_policy_dir = model_root / "controller_prior" / POLICY_CONFIG
    adversary_policy_dir = model_root / "adversary_prior" / POLICY_CONFIG
    _write_model(
        controller_value,
        controller_value_dir,
        tag="agz_markov_controller_value_v100",
    )
    _write_model(
        adversary_value,
        adversary_value_dir,
        tag="agz_markov_adversary_value_v100",
    )
    _write_model(
        controller_policy,
        controller_policy_dir,
        tag="agz_dnn_controller_prior_v100",
    )
    _write_model(
        adversary_policy,
        adversary_policy_dir,
        tag="agz_dnn_adversary_prior_v100",
    )

    result = {
        "model_family": "dnn",
        "value_feature_schema": "markov_v2",
        "model_version": 100,
        "controller_model_version": 100,
        "adversary_model_version": 100,
        "created_at_utc": utc_now(),
        "native_ready": True,
        "incremental_update": True,
        "target_perspective": "controller",
        "eval_status": "bootstrap",
        "bootstrap_strategy": "constant_legacy_v100_teacher_mean",
        "teacher_replay_rows_available": int(available),
        "teacher_replay_rows_sampled": int(legacy_features.shape[0]),
        "teacher_value_min": float(np.min(clipped)),
        "teacher_value_max": float(np.max(clipped)),
        "teacher_value_mean": teacher_mean,
        "teacher_value_std": float(np.std(clipped)),
        "controller_value_model_path": str(controller_value_dir / "model.joblib"),
        "adversary_value_model_path": str(adversary_value_dir / "model.joblib"),
        "controller_prior_model_path": str(controller_policy_dir / "model.joblib"),
        "adversary_prior_model_path": str(adversary_policy_dir / "model.joblib"),
        "value_model_path": str(controller_value_dir / "model.joblib"),
    }
    atomic_write_json(model_root / "candidate_manifest.json", result)
    atomic_write_json(output_root / "models" / "current_model.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-replay-root", type=Path, required=True)
    parser.add_argument("--source-model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-sample-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(bootstrap(parse_args()), indent=2, sort_keys=True))
