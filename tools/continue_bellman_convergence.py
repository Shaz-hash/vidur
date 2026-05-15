"""Continue a ModelSearchBed Bellman convergence run from an existing version.

This is intentionally a thin runner around the existing harness internals.  The
standard CLI starts at V1, which is not appropriate when a long remote run has
already produced checkpoints and cached target artifacts.  This script loads
the previous version checkpoint, then continues training/generation from the
next requested version without touching completed earlier versions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from vidur.mcts.Game_Versions.Game_Version3.DNN.model_search_nn import HorizonValueNet
from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.bellman_convergence import (
    RootStateLoader,
    build_bellman_config,
    compute_mcts_targets,
    evaluate_model_on_dataset,
    load_root_records,
    make_self_model_config,
    split_records_with_indices,
    train_model,
    write_transition_summary,
    _log_process_memory,
    _release_iteration_memory,
)


def _load_horizon_checkpoint(checkpoint_path: Path) -> HorizonValueNet:
    """Load one trained HorizonValueNet from a ModelSearchBed checkpoint."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    feature_mean = state["feature_mean"]
    in_proj_weight = state["in_proj.0.weight"]
    input_dim = int(feature_mean.numel())
    hidden_dim = int(in_proj_weight.shape[0])
    model = HorizonValueNet(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state)
    model.eval()
    return model


def _build_config_from_existing_run(
    *,
    output_dir: Path,
    end_version: int,
    seed: int,
) -> Any:
    """Rebuild Bellman config from the run's saved config.json."""

    saved = json.loads((output_dir / "config.json").read_text())
    return build_bellman_config(
        dataset_dir=saved["dataset_dir"],
        output_dir=output_dir,
        num_versions=int(end_version),
        num_roots=int(saved["num_roots"]),
        eval_ratio=float(saved["eval_ratio"]),
        split_seed=int(saved["split_seed"]),
        seed=int(seed),
        root_player_filter=str(saved["root_player_filter"]),
        model_name_prefix=str(saved["model_name_prefix"]),
        extra_config=dict(saved.get("extra_config", {}) or {}),
    )


def continue_bellman_run(
    *,
    output_dir: Path,
    start_version: int,
    end_version: int,
    seed: int,
) -> None:
    """Run Bellman versions [start_version, end_version] inclusive."""

    if int(start_version) <= 1:
        raise ValueError("start_version must be > 1 for continuation")
    if int(end_version) < int(start_version):
        raise ValueError("end_version must be >= start_version")

    cfg = _build_config_from_existing_run(
        output_dir=Path(output_dir),
        end_version=int(end_version),
        seed=int(seed),
    )
    previous_version = int(start_version) - 1
    previous_checkpoint = cfg.output_dir / f"Model_Version{previous_version}" / "best_model.pt"
    if not previous_checkpoint.exists():
        raise FileNotFoundError(f"missing previous checkpoint: {previous_checkpoint}")

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

    state_loader = RootStateLoader(model0_cfg)
    previous_model: Any | None = _load_horizon_checkpoint(previous_checkpoint)
    print(
        "[continue_bellman] "
        f"loaded bootstrap Model_Version{previous_version} from {previous_checkpoint}",
        flush=True,
    )
    try:
        for model_version in range(int(start_version), int(end_version) + 1):
            bootstrap_version = int(model_version) - 1
            model_dir = cfg.output_dir / f"Model_Version{int(model_version)}"
            model_dir.mkdir(parents=True, exist_ok=True)
            _log_process_memory(f"continuation_version_{int(model_version)}_start")

            train_targets = compute_mcts_targets(
                records=train_records,
                state_loader=state_loader,
                bootstrap_model=previous_model,
                cfg=cfg,
                bootstrap_version=bootstrap_version,
                model_version=int(model_version),
                output_dir=cfg.output_dir,
                split_name="train",
                cache_targets=bool(cfg.cache_targets),
                reuse_record_targets_for_v0=False,
            )
            eval_targets = compute_mcts_targets(
                records=eval_records,
                state_loader=state_loader,
                bootstrap_model=previous_model,
                cfg=cfg,
                bootstrap_version=bootstrap_version,
                model_version=int(model_version),
                output_dir=cfg.output_dir,
                split_name="eval",
                cache_targets=bool(cfg.cache_targets),
                reuse_record_targets_for_v0=False,
            )
            _log_process_memory(f"continuation_version_{int(model_version)}_after_targets")

            old_model = previous_model
            previous_model = None
            if old_model is not None:
                _release_iteration_memory(
                    model=old_model,
                    label=f"continuation_version_{int(model_version)}_after_previous_model_release",
                )
                del old_model

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
                    cfg=cfg,
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
                    cfg=cfg,
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
                del same_train_targets, same_eval_targets, same_train_row, same_eval_row

            _release_iteration_memory(
                model=model,
                label=f"continuation_version_{int(model_version)}_after_prediction_cache_release",
            )
            del train_targets, eval_targets, train_row, eval_row
            previous_model = model
            _log_process_memory(f"continuation_version_{int(model_version)}_end")
    finally:
        state_loader.close()
        _release_iteration_memory(model=previous_model, label="continuation_final_cleanup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-version", type=int, default=51)
    parser.add_argument("--end-version", type=int, default=250)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    continue_bellman_run(
        output_dir=args.output_dir,
        start_version=int(args.start_version),
        end_version=int(args.end_version),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
