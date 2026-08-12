"""Create a genuinely untrained Markov-v2 DNN bundle for AGZ version 100."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from vidur.AlphaGoZero.dnn_models import (
    MarkovPolicyRankDeepSet,
    MarkovValueDeepSet,
    export_dnn_to_native,
    save_dnn_model,
    write_artifact_metadata,
)
from vidur.AlphaGoZero.durable_transfer import atomic_write_json, utc_now


VALUE_CONFIG = "dnn_value_markov_deepset_192_v2"
POLICY_CONFIG = "dnn_policy_markov_deepset_192_v3"


def _neutralize_output_head(model: torch.nn.Module) -> None:
    """Keep the hidden representation random while starting predictions neutral."""
    with torch.no_grad():
        model.head2.weight.zero_()
        model.head2.bias.zero_()


def _attach_fresh_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    model.optimizer_state = optimizer.state_dict()
    model.training_metadata = {
        "bootstrap_version": 100,
        "bootstrap_strategy": "random_hidden_zero_output_head_v100",
        "initialization_seed": int(seed),
        "feature_schema": getattr(model, "feature_schema", "legacy_226"),
        "target_perspective": "controller",
        "incremental_update": True,
        "teacher_rows": 0,
        "training_steps": 0,
    }
    model.eval()


def _write_model(model: Any, directory: Path, *, tag: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_dnn_model(model, directory / "model.joblib")
    export_dnn_to_native(model, directory / "native_model.tsv", model_tag=tag)
    write_artifact_metadata(model, directory / "metadata.json")


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    controller_value = MarkovValueDeepSet(role="controller")
    _neutralize_output_head(controller_value)
    _attach_fresh_optimizer(
        controller_value,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=seed,
    )

    # Both value heads start from identical knowledge, as in the original split-value setup.
    adversary_value = copy.deepcopy(controller_value)
    adversary_value.role = "adversary"
    adversary_value.training_metadata = dict(controller_value.training_metadata)
    adversary_value.training_metadata["copied_from_controller_value_v100"] = True

    torch.manual_seed(seed + 1)
    controller_policy = MarkovPolicyRankDeepSet(43, role="controller")
    _neutralize_output_head(controller_policy)
    _attach_fresh_optimizer(
        controller_policy,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=seed + 1,
    )

    torch.manual_seed(seed + 2)
    adversary_policy = MarkovPolicyRankDeepSet(7, role="adversary")
    _neutralize_output_head(adversary_policy)
    _attach_fresh_optimizer(
        adversary_policy,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=seed + 2,
    )

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
        tag="agz_untrained_markov_controller_value_v100",
    )
    _write_model(
        adversary_value,
        adversary_value_dir,
        tag="agz_untrained_markov_adversary_value_v100",
    )
    _write_model(
        controller_policy,
        controller_policy_dir,
        tag="agz_untrained_controller_prior_v100",
    )
    _write_model(
        adversary_policy,
        adversary_policy_dir,
        tag="agz_untrained_adversary_prior_v100",
    )

    result = {
        "model_family": "dnn",
        "value_feature_schema": "markov_v2",
        "policy_feature_schema": "markov_v2",
        "model_version": 100,
        "controller_model_version": 100,
        "adversary_model_version": 100,
        "created_at_utc": utc_now(),
        "native_ready": True,
        "incremental_update": True,
        "target_perspective": "controller",
        "eval_status": "bootstrap",
        "bootstrap_strategy": "random_hidden_zero_output_head_v100",
        "initialization_seed": seed,
        "neutral_value_cost": -25.0,
        "neutral_policy": "uniform",
        "teacher_replay_rows_available": 0,
        "teacher_replay_rows_sampled": 0,
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(bootstrap(parse_args()), indent=2, sort_keys=True))
