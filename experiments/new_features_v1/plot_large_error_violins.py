"""Plot value distributions for large-error train/eval samples.

This script reads ``train_results.csv`` and ``eval_results.csv`` from one model
directory, filters rows with ``abs_err > threshold``, and writes one violin plot
that compares ``y_true`` and ``y_pred`` distributions for train vs eval.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_large_error_rows(path: Path, *, split: str, threshold: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for raw in reader:
            row = {
                str(k).strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in raw.items()
            }
            abs_err = float(row["abs_err"])
            if abs_err <= float(threshold):
                continue
            rows.append(
                {
                    "split": split,
                    "psid": int(row["psid"]),
                    "y_true": float(row["y_true"]),
                    "y_pred": float(row["y_pred"]),
                    "abs_err": abs_err,
                }
            )
    return rows


def _flatten(rows: list[dict[str, float | str]], value_key: str, split: str) -> list[float]:
    return [
        float(row[value_key])
        for row in rows
        if str(row["split"]) == split
    ]


def plot_large_error_violins(
    *,
    model_dir: Path,
    threshold: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_path = model_dir / "train_results.csv"
    eval_path = model_dir / "eval_results.csv"
    train_rows = _read_large_error_rows(train_path, split="train", threshold=threshold)
    eval_rows = _read_large_error_rows(eval_path, split="eval", threshold=threshold)
    rows = train_rows + eval_rows
    if not rows:
        raise ValueError(f"No rows found with abs_err > {threshold}")

    data = [
        _flatten(rows, "y_true", "train"),
        _flatten(rows, "y_true", "eval"),
        _flatten(rows, "y_pred", "train"),
        _flatten(rows, "y_pred", "eval"),
    ]
    labels = [
        f"Y_true\ntrain\nn={len(data[0])}",
        f"Y_true\neval\nn={len(data[1])}",
        f"Y_pred\ntrain\nn={len(data[2])}",
        f"Y_pred\neval\nn={len(data[3])}",
    ]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    parts = ax.violinplot(
        data,
        positions=[1, 2, 4, 5],
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.82,
    )
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#1f2933")
        body.set_alpha(0.72)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#1f2933")
            parts[key].set_linewidth(1.0)

    ax.axhline(0.0, color="#6b7280", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_xticks([1, 2, 4, 5])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title(f"Large-error samples (abs_err > {threshold:g})")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)

    summary = (
        f"train: {len(train_rows):,} / "
        f"eval: {len(eval_rows):,} large-error rows"
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"wrote {output_path}")
    print(f"train_large_error_rows={len(train_rows)}")
    print(f"eval_large_error_rows={len(eval_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
            "simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v1/"
            "V1_models/hgb_sq_31leaf_700iter"
        ),
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.model_dir / "large_abs_err_gt1_ytrue_ypred_violin.png"

    plot_large_error_violins(
        model_dir=args.model_dir.expanduser(),
        threshold=float(args.threshold),
        output_path=output.expanduser(),
    )


if __name__ == "__main__":
    main()
