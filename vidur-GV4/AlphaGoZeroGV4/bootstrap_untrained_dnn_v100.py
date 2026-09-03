"""Create the neutral four-model bundle used to start GV4 self-play."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from GV4_Engine.config import GV4EngineConfig

from .dnn_models import GV4ModelSpec, GV4PolicyDeepSet, GV4ValueDeepSet
from .model_bundle import (
    LoadedModelBundle,
    publish_model_bundle,
    write_current_model_pointer,
)


def _fresh_optimizer_state(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    return optimizer.state_dict()


def _neutralize(model: GV4ValueDeepSet | GV4PolicyDeepSet) -> None:
    """Retain random hidden features while making the output initially neutral."""

    with torch.no_grad():
        model.head_output.weight.zero_()
        if isinstance(model, GV4ValueDeepSet):
            # The value head is constrained to nonpositive values. A large
            # negative pre-softplus bias is therefore the closest stable zero.
            model.head_output.bias.fill_(-12.0)
        else:
            model.head_output.bias.zero_()


def build_untrained_models(
    config: GV4EngineConfig,
    *,
    seed: int,
    version: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, GV4ValueDeepSet | GV4PolicyDeepSet]:
    """Build deterministic zero-policy, near-zero-value warm-start models."""

    spec = GV4ModelSpec.from_config(config)
    definitions = (
        ("controller_value", GV4ValueDeepSet, "controller"),
        ("adversary_value", GV4ValueDeepSet, "adversary"),
        ("controller_policy", GV4PolicyDeepSet, "controller"),
        ("adversary_policy", GV4PolicyDeepSet, "adversary"),
    )
    models: dict[str, GV4ValueDeepSet | GV4PolicyDeepSet] = {}
    for offset, (name, model_type, role) in enumerate(definitions):
        # Resetting per model makes bootstrap artifacts reproducible.
        torch.manual_seed(int(seed) + offset)
        model = model_type(spec, role=role)
        _neutralize(model)
        model.optimizer_state = _fresh_optimizer_state(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        model.training_metadata = {
            "bootstrap": True,
            "bootstrap_version": int(version),
            "initialization_seed": int(seed) + offset,
            "training_steps": 0,
            "target_perspective": "controller",
        }
        model.eval()
        models[name] = model
    return models


def bootstrap_bundle(
    output_root: str | Path,
    *,
    config: GV4EngineConfig,
    version: int = 100,
    seed: int = 2026,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    set_current: bool = True,
) -> LoadedModelBundle:
    models = build_untrained_models(
        config,
        seed=seed,
        version=version,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model_root = Path(output_root).expanduser().resolve() / "models"
    bundle = publish_model_bundle(
        model_root / f"Model_Version{version}",
        bundle_version=version,
        config=config,
        models=models,
        metadata={
            "eval_status": "bootstrap",
            "bootstrap_strategy": "random_hidden_neutral_output",
            "initialization_seed": int(seed),
        },
    )
    if set_current:
        write_current_model_pointer(model_root / "current_model.json", bundle)
    return bundle


def _load_config_factory(specification: str) -> GV4EngineConfig:
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("config factory must have the form module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    config = factory()
    if not isinstance(config, GV4EngineConfig):
        raise TypeError("config factory did not return GV4EngineConfig")
    config.validate()
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--engine-config-factory", required=True)
    parser.add_argument("--version", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--set-current",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    bundle = bootstrap_bundle(
        args.output_root,
        config=_load_config_factory(args.engine_config_factory),
        version=args.version,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        set_current=args.set_current,
    )
    print(
        json.dumps(
            {
                "bundle_version": bundle.bundle_version,
                "manifest_path": str(bundle.manifest_path),
                "manifest_sha256": bundle.manifest_sha256,
                "model_versions": bundle.model_versions,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
