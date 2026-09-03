"""Evaluate candidate checkpoints on replay partitions newer than a cutoff.

This is intentionally read-only. It uses one fixed sample for every checkpoint
so training-set metrics cannot hide candidate drift.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from vidur.AlphaGoZero.dnn_models import (
    MarkovPolicyRankDeepSet,
    MarkovValueDeepSet,
    _collate_markov_features,
    load_dnn_model,
)
from vidur.AlphaGoZero.indexed_replay import (
    CONTROLLER_ACTION_DIM,
    ensure_partition_indexes,
    materialize_markov_policy_arrays,
    materialize_policy_roots,
    materialize_value_samples,
    sample_addresses,
)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _new_controller_state_paths(root: Path, cutoff: datetime) -> list[Path]:
    paths: list[Path] = []
    for manifest_path in root.glob("global_replay/partitions/*/*__controller/partition_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _parse_utc(str(manifest["created_at_utc"])) > cutoff:
            state_path = manifest_path.parent / "replay_target_runtime_feature_complete.csv"
            if state_path.is_file():
                paths.append(state_path)
    return sorted(paths)


def _policy_predictions(
    model: MarkovPolicyRankDeepSet,
    states: list,
    actions: np.ndarray,
    offsets: list[tuple[int, int]],
    batch_roots: int,
) -> np.ndarray:
    output = np.empty(actions.shape[0], dtype=np.float32)
    action_tensor = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(states), batch_roots):
            end = min(len(states), begin + batch_roots)
            chosen_states = states[begin:end]
            chosen_ranges = offsets[begin:end]
            embeddings = model.encode_state(*_collate_markov_features(chosen_states))
            counts = torch.tensor([b - a for a, b in chosen_ranges], dtype=torch.int64)
            expanded = torch.repeat_interleave(embeddings, counts, dim=0)
            row_indices = np.concatenate(
                [np.arange(a, b, dtype=np.int64) for a, b in chosen_ranges]
            )
            logits = model.forward_with_state_embedding(
                expanded,
                action_tensor.index_select(0, torch.from_numpy(row_indices)),
            )
            output[row_indices] = logits.cpu().numpy()
    return output


def _policy_metrics(
    predicted_logits: np.ndarray,
    targets: np.ndarray,
    offsets: list[tuple[int, int]],
) -> dict[str, float]:
    cross_entropy = 0.0
    kl = 0.0
    top1 = 0
    target_entropy = 0.0
    for begin, end in offsets:
        target = np.asarray(targets[begin:end], dtype=np.float64)
        target /= max(float(target.sum()), 1e-30)
        logits = np.asarray(predicted_logits[begin:end], dtype=np.float64)
        logits -= float(logits.max())
        predicted = np.exp(logits)
        predicted /= max(float(predicted.sum()), 1e-30)
        safe_target = np.clip(target, 1e-30, 1.0)
        safe_predicted = np.clip(predicted, 1e-30, 1.0)
        ce = -float(np.sum(target * np.log(safe_predicted)))
        entropy = -float(np.sum(target * np.log(safe_target)))
        cross_entropy += ce
        target_entropy += entropy
        kl += ce - entropy
        top1 += int(int(np.argmax(predicted)) == int(np.argmax(target)))
    count = max(1, len(offsets))
    return {
        "policy_cross_entropy": cross_entropy / count,
        "policy_kl": kl / count,
        "policy_top1": top1 / count,
        "target_entropy": target_entropy / count,
    }


def _value_predictions(model: MarkovValueDeepSet, states: list, batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for begin in range(0, len(states), batch_size):
        predictions.append(model.predict_structured(states[begin : begin + batch_size]))
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cutoff-version", type=int, required=True)
    parser.add_argument("--versions", type=int, nargs="+", required=True)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=917_503)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--batch-roots", type=int, default=512)
    args = parser.parse_args()

    cutoff_manifest = args.root / "models" / f"Model_Version{args.cutoff_version}" / "candidate_manifest.json"
    cutoff = _parse_utc(json.loads(cutoff_manifest.read_text(encoding="utf-8"))["created_at_utc"])
    paths = _new_controller_state_paths(args.root, cutoff)
    if not paths:
        raise SystemExit("no controller replay partitions were created after the cutoff")

    descriptors, _ = ensure_partition_indexes(
        paths, value_feature_schema="markov_v2", workers=args.cache_workers
    )
    policy_addresses, policy_available = sample_addresses(
        descriptors,
        role="controller",
        sample_size=args.sample_size,
        seed=args.seed,
        policy_only=True,
    )
    value_addresses, value_available = sample_addresses(
        descriptors,
        role="controller",
        sample_size=args.sample_size,
        seed=args.seed + 1,
    )
    policy_roots = materialize_policy_roots(
        policy_addresses, value_feature_schema="markov_v2", workers=args.cache_workers
    )
    (states, actions, _target_logits, target_probs, offsets), _ = materialize_markov_policy_arrays(
        policy_roots,
        action_dim=CONTROLLER_ACTION_DIM,
        root_cap=args.sample_size,
    )
    value_rows = materialize_value_samples(
        value_addresses, value_feature_schema="markov_v2", workers=args.cache_workers
    )
    value_states = [row["_value_features"] for row in value_rows]
    value_targets = np.asarray([row["_target_value"] for row in value_rows], dtype=np.float64)

    print(
        json.dumps(
            {
                "cutoff_utc": cutoff.isoformat(),
                "new_partitions": len(paths),
                "policy_roots_available": policy_available,
                "policy_roots_tested": len(states),
                "value_rows_available": value_available,
                "value_rows_tested": len(value_states),
            },
            sort_keys=True,
        )
    )
    for version in args.versions:
        model_root = args.root / "models" / f"Model_Version{version}"
        policy_path = model_root / "controller_prior" / "dnn_policy_markov_deepset_192_v3" / "model.joblib"
        value_path = model_root / "controller_value" / "dnn_value_markov_deepset_192_v2" / "model.joblib"
        policy_model = load_dnn_model(policy_path, MarkovPolicyRankDeepSet)
        value_model = load_dnn_model(value_path, MarkovValueDeepSet)
        assert isinstance(policy_model, MarkovPolicyRankDeepSet)
        assert isinstance(value_model, MarkovValueDeepSet)
        policy_logits = _policy_predictions(
            policy_model, states, actions, offsets, args.batch_roots
        )
        metrics = _policy_metrics(policy_logits, target_probs, offsets)
        value_predictions = _value_predictions(value_model, value_states, args.batch_roots)
        errors = value_predictions.astype(np.float64) - value_targets
        metrics.update(
            {
                "version": int(version),
                "value_rmse": math.sqrt(float(np.mean(errors * errors))),
                "value_mae": float(np.mean(np.abs(errors))),
                "value_bias": float(np.mean(errors)),
                "value_correlation": float(np.corrcoef(value_predictions, value_targets)[0, 1]),
            }
        )
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
