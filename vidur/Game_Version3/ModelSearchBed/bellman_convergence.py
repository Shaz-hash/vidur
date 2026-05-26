"""Bellman convergence harness for GV3 ModelSearchBed experiments.

This script trains value models across Bellman depths:

    V1 learns T[V0], where V0 is zero bootstrap and T[V0] is immediate reward.
    V2 learns T[V1], where targets are reward + discount * V1(child).
    ...

It also optionally measures same-model Bellman residuals:

    Vi(s) vs T[Vi](s)

The root dataset is treated as read-only. This file never generates roots.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from ..DNN import infer as infer_module
from . import logger as bellman_logger
from .self_model_test import (
    RootStateLoader,
    SelfModelTestConfig,
    build_self_model_test_config,
    count_trainable_parameters,
    load_root_records,
    predict_candidate_values,
    train_candidate_model,
    write_training_metrics,
)


@dataclass(frozen=True)
class BellmanConvergenceConfig:
    """Configuration for iterative Bellman target/training runs."""

    dataset_dir: Path
    output_dir: Path
    num_versions: int = 5
    num_roots: int = 10_000
    eval_ratio: float = 0.20
    split_seed: int = 12345
    seed: int = 2027
    batch_size: int = 256
    root_player_filter: str = "controller"
    model_name_prefix: str = "bellman_model"
    abs_error_threshold: float = 1.0
    reuse_record_targets_for_v0: bool = True
    cache_targets: bool = True
    same_model_analysis_start_version: int = 2
    extra_config: dict[str, Any] | None = None


class BootstrapModelAdapter:
    """Adapter used by mctsDNN bootstrap calls.

    mctsDNN expects `infer_from_inputs(inputs, player, device=...)`. The
    existing GV3 model already implements that method. Custom ModelSearchBed
    experiments can either implement the same method on their model object or
    expose `DNN/infer.py::predict_model_search_value_from_inputs(...)`.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: torch.device | None = None,
    ) -> tuple[float, list[float]]:
        infer_fn = getattr(self.model, "infer_from_inputs", None)
        if callable(infer_fn):
            return infer_fn(inputs, player, device=device)

        hook = getattr(infer_module, "predict_model_search_value_from_inputs", None)
        if callable(hook):
            value = hook(
                model=self.model,
                inputs=inputs,
                player=player,
                device=device,
            )
            return float(value), []

        raise TypeError(
            "Bootstrap model must implement infer_from_inputs(...) or "
            "DNN/infer.py must define predict_model_search_value_from_inputs(...)."
        )


def build_bellman_config(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    num_versions: int = 5,
    num_roots: int = 10_000,
    eval_ratio: float = 0.20,
    split_seed: int = 12345,
    seed: int = 2027,
    batch_size: int = 256,
    root_player_filter: str = "controller",
    model_name_prefix: str = "bellman_model",
    abs_error_threshold: float = 1.0,
    reuse_record_targets_for_v0: bool = True,
    cache_targets: bool = True,
    same_model_analysis_start_version: int = 2,
    extra_config: dict[str, Any] | None = None,
) -> BellmanConvergenceConfig:
    """Validate and build a Bellman convergence config."""

    if int(num_versions) <= 0:
        raise ValueError("num_versions must be > 0")
    if int(num_roots) <= 1:
        raise ValueError("num_roots must be > 1")
    if not (0.0 < float(eval_ratio) < 1.0):
        raise ValueError("eval_ratio must be in (0, 1)")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if str(root_player_filter) not in {"controller", "adversary", "any"}:
        raise ValueError("root_player_filter must be 'controller', 'adversary', or 'any'")
    if float(abs_error_threshold) < 0.0:
        raise ValueError("abs_error_threshold must be >= 0")
    if int(same_model_analysis_start_version) <= 0:
        raise ValueError("same_model_analysis_start_version must be > 0")
    if extra_config is not None and not isinstance(extra_config, dict):
        raise TypeError("extra_config must be a dict when provided")

    return BellmanConvergenceConfig(
        dataset_dir=Path(dataset_dir).expanduser(),
        output_dir=Path(output_dir).expanduser(),
        num_versions=int(num_versions),
        num_roots=int(num_roots),
        eval_ratio=float(eval_ratio),
        split_seed=int(split_seed),
        seed=int(seed),
        batch_size=int(batch_size),
        root_player_filter=str(root_player_filter),
        model_name_prefix=str(model_name_prefix),
        abs_error_threshold=float(abs_error_threshold),
        reuse_record_targets_for_v0=bool(reuse_record_targets_for_v0),
        cache_targets=bool(cache_targets),
        same_model_analysis_start_version=int(same_model_analysis_start_version),
        extra_config=dict(extra_config or {}),
    )


def make_self_model_config(
    cfg: BellmanConvergenceConfig,
    *,
    output_dir: Path,
    model_name: str,
) -> SelfModelTestConfig:
    """Build the ModelSearchBed trainer/infer config for one Bellman version."""

    return build_self_model_test_config(
        dataset_dir=cfg.dataset_dir,
        output_dir=output_dir,
        num_roots=int(cfg.num_roots),
        max_candidate_roots=int(cfg.num_roots),
        seed=int(cfg.seed),
        root_player_filter=str(cfg.root_player_filter),
        allow_dataset_generation=False,
        eval_ratio=float(cfg.eval_ratio),
        split_seed=int(cfg.split_seed),
        model_name=str(model_name),
        batch_size=int(cfg.batch_size),
        extra_config=dict(cfg.extra_config or {}),
    )


def split_records_with_indices(
    records: Sequence[dict[str, Any]],
    *,
    eval_ratio: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    """Split records deterministically and keep original sample indices."""

    if not records:
        raise ValueError("cannot split an empty record list")
    indices = list(range(len(records)))
    random.Random(int(split_seed)).shuffle(indices)

    eval_count = int(round(float(len(indices)) * float(eval_ratio)))
    eval_count = max(1, min(len(indices) - 1, eval_count)) if len(indices) > 1 else 0
    eval_indices = sorted(indices[:eval_count])
    eval_index_set = set(eval_indices)
    train_indices = [idx for idx in range(len(records)) if idx not in eval_index_set]

    train_records = [dict(records[idx]) for idx in train_indices]
    eval_records = [dict(records[idx]) for idx in eval_indices]
    if not train_records or not eval_records:
        raise ValueError("train/eval split produced an empty split")
    return train_records, eval_records, train_indices, eval_indices


def records_with_targets(
    records: Sequence[dict[str, Any]],
    targets: Sequence[float],
) -> list[dict[str, Any]]:
    """Return shallow record copies with version-specific target values."""

    if len(records) != len(targets):
        raise ValueError(f"records/targets length mismatch: {len(records)} != {len(targets)}")
    out: list[dict[str, Any]] = []
    for record, target in zip(records, targets):
        row = dict(record)
        row["target_value"] = float(target)
        out.append(row)
    return out


def target_cache_path(
    output_dir: Path,
    *,
    model_version: int,
    bootstrap_version: int,
    split_name: str,
) -> Path:
    """Path for cached MCTS target values."""

    return (
        output_dir
        / f"Model_Version{int(model_version)}"
        / f"{split_name}_targets_from_model_{int(bootstrap_version)}.pt"
    )


def load_cached_targets(path: Path, *, expected_count: int) -> list[float] | None:
    """Load cached targets if present and shape-compatible."""

    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    targets = payload.get("targets") if isinstance(payload, dict) else payload
    targets = [float(x) for x in targets]
    if len(targets) != int(expected_count):
        return None
    return targets


def save_cached_targets(
    path: Path,
    *,
    targets: Sequence[float],
    metadata: dict[str, Any],
) -> Path:
    """Write MCTS target cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "targets": [float(x) for x in targets],
            "metadata": dict(metadata),
        },
        path,
    )
    return path


def compute_mcts_targets(
    *,
    records: Sequence[dict[str, Any]],
    state_loader: RootStateLoader,
    bootstrap_model: Any | None,
    bootstrap_version: int,
    model_version: int,
    output_dir: Path,
    split_name: str,
    cache_targets: bool,
    reuse_record_targets_for_v0: bool,
) -> list[float]:
    """Compute T[V_bootstrap](s) with mctsDNN for one split."""

    cache_path = target_cache_path(
        output_dir,
        model_version=int(model_version),
        bootstrap_version=int(bootstrap_version),
        split_name=str(split_name),
    )
    if bool(cache_targets):
        cached = load_cached_targets(cache_path, expected_count=len(records))
        if cached is not None:
            return cached

    if int(bootstrap_version) == 0 and bool(reuse_record_targets_for_v0):
        targets = [float(record["target_value"]) for record in records]
        if bool(cache_targets):
            save_cached_targets(
                cache_path,
                targets=targets,
                metadata={
                    "split_name": str(split_name),
                    "model_version": int(model_version),
                    "bootstrap_version": int(bootstrap_version),
                    "source": "stored_no_bootstrap_root_targets",
                },
            )
        return targets

    wrapped_model = None if bootstrap_model is None else BootstrapModelAdapter(bootstrap_model)
    use_bootstrap = wrapped_model is not None and int(bootstrap_version) > 0

    targets: list[float] = []
    mcts = state_loader.mcts
    for sample_number, record in enumerate(records):
        state = state_loader(record)
        out = mcts.search_dnn(
            dnn_model=wrapped_model,
            rootState=state,
            root_player=str(record["root_player"]),
            game_id=0,
            root_id=int(record.get("root_id", sample_number)),
            root_node_id_override=record.get("root_node_id_override", None),
            root_depth=int(record.get("root_depth", 0)),
            model_version=int(bootstrap_version),
            use_model_bootstrap=bool(use_bootstrap),
            one_step_value_mode=True,
        )
        targets.append(float(out.best_action_value))

    if bool(cache_targets):
        save_cached_targets(
            cache_path,
            targets=targets,
            metadata={
                "split_name": str(split_name),
                "model_version": int(model_version),
                "bootstrap_version": int(bootstrap_version),
                "source": "mcts_dnn_search",
                "use_model_bootstrap": bool(use_bootstrap),
            },
        )
    return targets


def train_model(
    *,
    model_version: int,
    train_records: Sequence[dict[str, Any]],
    eval_records: Sequence[dict[str, Any]],
    train_targets: Sequence[float],
    eval_targets: Sequence[float],
    cfg: BellmanConvergenceConfig,
    state_loader: RootStateLoader,
) -> Any:
    """Train model version Vi on T[V{i-1}] targets."""

    model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
    model_cfg = make_self_model_config(
        cfg,
        output_dir=model_dir,
        model_name=f"{cfg.model_name_prefix}_v{int(model_version)}",
    )
    train_labeled = records_with_targets(train_records, train_targets)
    eval_labeled = records_with_targets(eval_records, eval_targets)
    artifacts = train_candidate_model(
        train_labeled,
        eval_labeled,
        model_cfg,
        state_loader,
    )
    write_training_metrics(artifacts.train_metrics, model_cfg)
    return artifacts.model


def evaluate_model_on_dataset(
    *,
    model: Any,
    records: Sequence[dict[str, Any]],
    targets: Sequence[float],
    sample_numbers: Sequence[int],
    cfg: BellmanConvergenceConfig,
    state_loader: RootStateLoader,
    model_version: int,
    split_name: str,
    output_csv: Path,
) -> dict[str, Any]:
    """Predict model values, write per-sample CSV, and return aggregate stats."""

    model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
    model_cfg = make_self_model_config(
        cfg,
        output_dir=model_dir,
        model_name=f"{cfg.model_name_prefix}_v{int(model_version)}",
    )
    labeled_records = records_with_targets(records, targets)
    predictions = predict_candidate_values(
        model,
        labeled_records,
        model_cfg,
        state_loader,
        split_name=str(split_name),
    )
    bellman_logger.write_prediction_results_csv(
        output_csv,
        predictions=predictions,
        targets=targets,
        sample_numbers=sample_numbers,
    )
    metrics = bellman_logger.compute_error_metrics(
        predictions,
        targets,
        abs_error_threshold=float(cfg.abs_error_threshold),
    )
    row = {
        "model_version": int(model_version),
        "split": str(split_name),
        "trainable_params": int(count_trainable_parameters(model)),
    }
    row.update(metrics)
    return row


def write_transition_summary(
    cfg: BellmanConvergenceConfig,
    *,
    source_model_version: int,
    target_model_version: int,
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Write aggregate stats for a source->target Bellman transition."""

    out = cfg.output_dir / f"version_{int(source_model_version)}_to_{int(target_model_version)}.csv"
    enriched = []
    for row in rows:
        item = {
            "source_model_version": int(source_model_version),
            "target_model_version": int(target_model_version),
        }
        item.update(dict(row))
        enriched.append(item)
    return bellman_logger.write_summary_csv(out, enriched)


def bellman_convergence(cfg: BellmanConvergenceConfig) -> None:
    """Run iterative Bellman training and same-model residual analysis."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model0_cfg = make_self_model_config(
        cfg,
        output_dir=cfg.output_dir / "Model_Version0",
        model_name=f"{cfg.model_name_prefix}_v0",
    )
    records = load_root_records(model0_cfg)
    train_records, eval_records, train_indices, eval_indices = split_records_with_indices(
        records,
        eval_ratio=float(cfg.eval_ratio),
        split_seed=int(cfg.split_seed),
    )
    bellman_logger.write_split_indices(
        cfg.output_dir / "split_indices.json",
        train_indices=train_indices,
        eval_indices=eval_indices,
    )
    bellman_logger.write_json(
        cfg.output_dir / "config.json",
        {
            "dataset_dir": str(cfg.dataset_dir),
            "output_dir": str(cfg.output_dir),
            "num_versions": int(cfg.num_versions),
            "num_roots": int(cfg.num_roots),
            "eval_ratio": float(cfg.eval_ratio),
            "split_seed": int(cfg.split_seed),
            "root_player_filter": str(cfg.root_player_filter),
            "model_name_prefix": str(cfg.model_name_prefix),
            "extra_config": dict(cfg.extra_config or {}),
        },
    )

    state_loader = RootStateLoader(model0_cfg)
    previous_model: Any | None = None
    try:
        for model_version in range(1, int(cfg.num_versions) + 1):
            bootstrap_version = int(model_version) - 1
            model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
            model_dir.mkdir(parents=True, exist_ok=True)

            train_targets = compute_mcts_targets(
                records=train_records,
                state_loader=state_loader,
                bootstrap_model=previous_model,
                bootstrap_version=bootstrap_version,
                model_version=int(model_version),
                output_dir=cfg.output_dir,
                split_name="train",
                cache_targets=bool(cfg.cache_targets),
                reuse_record_targets_for_v0=bool(cfg.reuse_record_targets_for_v0),
            )
            eval_targets = compute_mcts_targets(
                records=eval_records,
                state_loader=state_loader,
                bootstrap_model=previous_model,
                bootstrap_version=bootstrap_version,
                model_version=int(model_version),
                output_dir=cfg.output_dir,
                split_name="eval",
                cache_targets=bool(cfg.cache_targets),
                reuse_record_targets_for_v0=bool(cfg.reuse_record_targets_for_v0),
            )

            model = train_model(
                model_version=int(model_version),
                train_records=train_records,
                eval_records=eval_records,
                train_targets=train_targets,
                eval_targets=eval_targets,
                cfg=cfg,
                state_loader=state_loader,
            )

            train_row = evaluate_model_on_dataset(
                model=model,
                records=train_records,
                targets=train_targets,
                sample_numbers=train_indices,
                cfg=cfg,
                state_loader=state_loader,
                model_version=int(model_version),
                split_name="train",
                output_csv=model_dir / "train_results.csv",
            )
            eval_row = evaluate_model_on_dataset(
                model=model,
                records=eval_records,
                targets=eval_targets,
                sample_numbers=eval_indices,
                cfg=cfg,
                state_loader=state_loader,
                model_version=int(model_version),
                split_name="eval",
                output_csv=model_dir / "eval_results.csv",
            )
            write_transition_summary(
                cfg,
                source_model_version=bootstrap_version,
                target_model_version=int(model_version),
                rows=[train_row, eval_row],
            )

            if int(model_version) >= int(cfg.same_model_analysis_start_version):
                same_train_targets = compute_mcts_targets(
                    records=train_records,
                    state_loader=state_loader,
                    bootstrap_model=model,
                    bootstrap_version=int(model_version),
                    model_version=int(model_version),
                    output_dir=cfg.output_dir,
                    split_name="train_same_model",
                    cache_targets=bool(cfg.cache_targets),
                    reuse_record_targets_for_v0=False,
                )
                same_eval_targets = compute_mcts_targets(
                    records=eval_records,
                    state_loader=state_loader,
                    bootstrap_model=model,
                    bootstrap_version=int(model_version),
                    model_version=int(model_version),
                    output_dir=cfg.output_dir,
                    split_name="eval_same_model",
                    cache_targets=bool(cfg.cache_targets),
                    reuse_record_targets_for_v0=False,
                )
                same_train_row = evaluate_model_on_dataset(
                    model=model,
                    records=train_records,
                    targets=same_train_targets,
                    sample_numbers=train_indices,
                    cfg=cfg,
                    state_loader=state_loader,
                    model_version=int(model_version),
                    split_name="train_same_model_bootstrap",
                    output_csv=model_dir / "train_result_same_model_boostrap.csv",
                )
                same_eval_row = evaluate_model_on_dataset(
                    model=model,
                    records=eval_records,
                    targets=same_eval_targets,
                    sample_numbers=eval_indices,
                    cfg=cfg,
                    state_loader=state_loader,
                    model_version=int(model_version),
                    split_name="eval_same_model_bootstrap",
                    output_csv=model_dir / "eval_result_same_model_boostrap.csv",
                )
                write_transition_summary(
                    cfg,
                    source_model_version=int(model_version),
                    target_model_version=int(model_version),
                    rows=[same_train_row, same_eval_row],
                )

            previous_model = model
    finally:
        state_loader.close()


def parse_args() -> BellmanConvergenceConfig:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run GV3 ModelSearchBed Bellman convergence.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-versions", type=int, default=5)
    parser.add_argument("--num-roots", type=int, default=10_000)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--root-player-filter", choices=("controller", "adversary", "any"), default="controller")
    parser.add_argument("--model-name-prefix", default="bellman_model")
    parser.add_argument("--abs-error-threshold", type=float, default=1.0)
    parser.add_argument(
        "--recompute-v0-targets",
        action="store_true",
        help="Recompute no-bootstrap V0 targets with mctsDNN instead of reusing stored root targets.",
    )
    parser.add_argument(
        "--no-cache-targets",
        action="store_true",
        help="Do not read/write cached MCTS targets.",
    )
    parser.add_argument("--same-model-analysis-start-version", type=int, default=2)
    parser.add_argument(
        "--extra-config-json",
        default="{}",
        help="Free-form JSON object passed through to trainer/infer hooks.",
    )
    args = parser.parse_args()
    extra_config = json.loads(args.extra_config_json)
    if not isinstance(extra_config, dict):
        raise ValueError("--extra-config-json must decode to a JSON object")
    return build_bellman_config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_versions=args.num_versions,
        num_roots=args.num_roots,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        seed=args.seed,
        batch_size=args.batch_size,
        root_player_filter=args.root_player_filter,
        model_name_prefix=args.model_name_prefix,
        abs_error_threshold=args.abs_error_threshold,
        reuse_record_targets_for_v0=not bool(args.recompute_v0_targets),
        cache_targets=not bool(args.no_cache_targets),
        same_model_analysis_start_version=args.same_model_analysis_start_version,
        extra_config=extra_config,
    )


def main() -> None:
    """CLI entrypoint."""

    bellman_convergence(parse_args())


if __name__ == "__main__":
    main()
