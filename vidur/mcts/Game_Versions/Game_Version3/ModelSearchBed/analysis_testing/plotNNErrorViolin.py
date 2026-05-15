"""Plot ModelSearchBed NN prediction-error violin distributions.

This script is intentionally analysis-only. It loads a fixed root dataset and a
saved ModelSearchBed NN checkpoint, recomputes predictions, and plots:

    prediction error = model_prediction - target_value

For the current controller-only root dataset, groups are:

    all, train, eval
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ...DNN.model_search_nn import HorizonValueNet, records_to_feature_tensor, records_to_target_tensor
from ..self_model_test import build_self_model_test_config, load_root_records, split_records


DEFAULT_DATASET_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur/"
    "simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40"
)
DEFAULT_RUN_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur-nn-search/"
    "simulator_output/GV3_Agent/model_search_results/nn_agent/exact_basis_residual_full_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NN ModelSearchBed error violin.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--num-roots", type=int, default=292_713)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--root-player-filter", default="controller")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def predict_errors(
    *,
    records: list[dict],
    model: HorizonValueNet,
    batch_size: int,
) -> torch.Tensor:
    features = records_to_feature_tensor(records)
    targets = records_to_target_tensor(records)
    preds: list[torch.Tensor] = []
    model.eval()
    for start in range(0, int(features.shape[0]), int(batch_size)):
        preds.append(model(features[start : start + int(batch_size)]).detach().cpu())
    if not preds:
        return torch.empty((0,), dtype=torch.float32)
    pred = torch.cat(preds, dim=0).view(-1)
    return pred - targets.view(-1)


def summarize(errors: torch.Tensor) -> dict[str, float | int]:
    abs_err = errors.abs()
    return {
        "count": int(errors.numel()),
        "mean_error": float(errors.mean().item()),
        "mean_abs_error": float(abs_err.mean().item()),
        "mse": float((errors * errors).mean().item()),
        "rmse": float(torch.sqrt((errors * errors).mean()).item()),
        "p50_abs_error": float(torch.quantile(abs_err, 0.50).item()),
        "p95_abs_error": float(torch.quantile(abs_err, 0.95).item()),
        "max_abs_error": float(abs_err.max().item()),
        "count_abs_error_gt_0_1": int((abs_err > 0.1).sum().item()),
        "count_abs_error_gt_0_5": int((abs_err > 0.5).sum().item()),
        "count_abs_error_gt_1": int((abs_err > 1.0).sum().item()),
    }


def write_summary_csv(path: Path, summaries: dict[str, dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group"] + list(next(iter(summaries.values())).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, row in summaries.items():
            out = {"group": group}
            out.update(row)
            writer.writerow(out)


def plot_violin(path: Path, grouped_errors: dict[str, torch.Tensor], summaries: dict[str, dict[str, float | int]]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = []
    for group, errors in grouped_errors.items():
        for value in errors.tolist():
            rows.append({"group": group, "prediction_error": float(value)})
    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 7.5), dpi=160)
    order = ["all", "train", "eval"]
    sns.violinplot(
        data=df,
        x="group",
        y="prediction_error",
        order=order,
        inner="quartile",
        cut=0,
        linewidth=1.2,
        color="#9dbbd6",
        ax=ax,
    )

    # Overlay sparse outliers so the max-error cases are visible.
    outlier_rows = df[df["prediction_error"].abs() > 0.1]
    if not outlier_rows.empty:
        sns.stripplot(
            data=outlier_rows,
            x="group",
            y="prediction_error",
            order=order,
            color="#202020",
            size=3,
            jitter=0.12,
            alpha=0.65,
            ax=ax,
        )

    ax.axhline(0.0, color="#b00030", linestyle="--", linewidth=1.6, label="zero error")
    ax.set_title(
        "GV3 controller value error distribution (NN exact_basis_residual_full_v3)\n"
        "Model prediction vs stored first-layer tree-search/Bellman target\n"
        f"all={summaries['all']['count']}, train={summaries['train']['count']}, eval={summaries['eval']['count']}",
        fontsize=15,
    )
    ax.set_xlabel("Dataset split")
    ax.set_ylabel("Prediction error = model predicted value - target_value")
    ax.legend(loc="upper right")

    # Keep the true tails visible but emphasize the dense near-zero distribution.
    all_err = grouped_errors["all"]
    lower = min(-0.25, float(all_err.min().item()) - 0.25)
    upper = max(0.25, float(all_err.max().item()) + 0.25)
    ax.set_ylim(lower, upper)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = run_dir / str(args.checkpoint_name)
    output_png = args.output_png or (run_dir / "exact_basis_residual_full_v3_error_violin.png")
    output_csv = args.output_csv or (run_dir / "exact_basis_residual_full_v3_error_violin_summary.csv")

    cfg = build_self_model_test_config(
        dataset_dir=args.dataset_dir,
        output_dir=run_dir,
        num_roots=int(args.num_roots),
        max_candidate_roots=int(args.num_roots),
        root_player_filter=str(args.root_player_filter),
        eval_ratio=float(args.eval_ratio),
        split_seed=int(args.split_seed),
    )
    split = split_records(load_root_records(cfg), cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HorizonValueNet(input_dim=int(checkpoint["feature_dim"]))
    # Some later experiments added auxiliary heads. strict=False keeps this
    # script compatible with older best checkpoints.
    model.load_state_dict(checkpoint["model_state"], strict=False)

    train_errors = predict_errors(records=split.train_records, model=model, batch_size=int(args.batch_size))
    eval_errors = predict_errors(records=split.eval_records, model=model, batch_size=int(args.batch_size))
    all_errors = torch.cat([train_errors, eval_errors], dim=0)

    grouped = {
        "all": all_errors,
        "train": train_errors,
        "eval": eval_errors,
    }
    summaries = {group: summarize(errors) for group, errors in grouped.items()}
    write_summary_csv(output_csv, summaries)
    plot_violin(output_png, grouped, summaries)

    print(f"wrote plot: {output_png}")
    print(f"wrote summary: {output_csv}")
    for group, row in summaries.items():
        print(
            f"{group:5s} count={row['count']} mean_abs={row['mean_abs_error']:.6f} "
            f"p95={row['p95_abs_error']:.6f} max_abs={row['max_abs_error']:.6f} "
            f"abs_err>1={row['count_abs_error_gt_1']}"
        )


if __name__ == "__main__":
    main()
