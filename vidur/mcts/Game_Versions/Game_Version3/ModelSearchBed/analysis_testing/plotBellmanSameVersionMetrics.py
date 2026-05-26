"""Plot Bellman same-version convergence metrics across model versions.

This script reads same-model Bellman convergence summary files:

    version_2_to_2.csv
    version_3_to_3.csv
    ...
    version_50_to_50.csv

These files measure how close model version i is to the Bellman target when
that same model version is used as the bootstrap value source.

Outputs are written under:

    simulator_output/GV3_Agent/analysis_same_version/{nn,classical}/{train,eval}/

Each split directory gets one compact CSV plus individual metric plots and a
four-panel summary plot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRICS: tuple[tuple[str, str], ...] = (
    ("rmse", "RMSE"),
    ("p95_abs_error", "p95 Absolute Error"),
    ("max_abs_error", "Max Absolute Error"),
    ("num_abs_error_ge_1", "# Samples with abs(error) >= 1"),
)

SPLIT_ALIASES = {
    "train": "train",
    "eval": "eval",
    "train_same_model_bootstrap": "train",
    "eval_same_model_bootstrap": "eval",
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    input_dir: Path
    output_dir: Path


def _repo_root_from_script() -> Path:
    """Return the main repo root containing this script."""

    return Path(__file__).resolve().parents[6]


def _workspace_root_from_repo(repo_root: Path) -> Path:
    """Return the parent directory that contains sibling worktrees."""

    return repo_root.parent


def read_same_version_rows(
    input_dir: Path,
    *,
    min_version: int,
    max_version: int,
) -> list[dict[str, str]]:
    """Read `version_i_to_i.csv` rows for i in [min_version, max_version]."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {input_dir}")

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for version in range(int(min_version), int(max_version) + 1):
        path = input_dir / f"version_{version}_to_{version}.csv"
        if not path.exists():
            missing.append(path.name)
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                raw_split = str(row.get("split", ""))
                normalized_split = SPLIT_ALIASES.get(raw_split)
                if normalized_split is None:
                    continue
                row["raw_split"] = raw_split
                row["split"] = normalized_split
                row["source_file"] = path.name
                row["iteration"] = str(version)
                rows.append(row)

    if missing:
        missing_preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(
            f"missing same-version summaries in {input_dir}: "
            f"{missing_preview}{suffix}"
        )
    if not rows:
        raise RuntimeError(f"no same-version rows found in {input_dir}")
    return rows


def rows_for_split(rows: Iterable[dict[str, str]], split: str) -> list[dict[str, str]]:
    """Filter rows for `train` or `eval` and sort by iteration."""

    filtered = [row for row in rows if str(row.get("split", "")) == split]
    filtered.sort(key=lambda row: int(row["iteration"]))
    return filtered


def write_split_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write the compact metric table used by the plots."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "source_model_version",
        "target_model_version",
        "model_version",
        "split",
        "raw_split",
        "trainable_params",
        "num_samples",
        "rmse",
        "p95_abs_error",
        "max_abs_error",
        "num_abs_error_ge_1",
        "mse",
        "mae",
        "p95_squared_error",
        "source_file",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _metric_values(rows: list[dict[str, str]], metric: str) -> tuple[list[int], list[float]]:
    x = [int(row["iteration"]) for row in rows]
    y = [float(row[metric]) for row in rows]
    return x, y


def plot_metric(rows: list[dict[str, str]], metric: str, title: str, output_path: Path) -> None:
    """Write one line graph for one metric."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = _metric_values(rows, metric)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Model Version")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(x), max(x))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_grid(rows: list[dict[str, str]], title: str, output_path: Path) -> None:
    """Write a four-panel plot for quick slide/review inspection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (metric, metric_title) in zip(axes.flat, METRICS):
        x, y = _metric_values(rows, metric)
        ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.6)
        ax.set_title(metric_title)
        ax.set_xlabel("Model Version")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(x), max(x))
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_run(run: RunSpec, *, min_version: int, max_version: int) -> dict[str, int]:
    """Create available train/eval plots for one NN or classical run."""

    rows = read_same_version_rows(
        run.input_dir,
        min_version=int(min_version),
        max_version=int(max_version),
    )
    counts: dict[str, int] = {}
    run.output_dir.mkdir(parents=True, exist_ok=True)
    (run.output_dir / "source_run.txt").write_text(str(run.input_dir) + "\n")

    expected_rows = int(max_version) - int(min_version) + 1
    for split in ("train", "eval"):
        split_rows = rows_for_split(rows, split)
        counts[split] = len(split_rows)
        if not split_rows:
            continue
        if len(split_rows) != expected_rows:
            print(
                f"[warn] {run.name} {split} has {len(split_rows)} rows; "
                f"expected {expected_rows}"
            )
        split_dir = run.output_dir / split
        write_split_csv(split_rows, split_dir / "same_version_metrics.csv")
        for metric, metric_title in METRICS:
            plot_metric(
                split_rows,
                metric=metric,
                title=f"{run.name} {split}: same-version {metric_title}",
                output_path=split_dir / f"{metric}.png",
            )
        plot_summary_grid(
            split_rows,
            title=f"{run.name} {split}: Same-Version Bellman Metrics",
            output_path=split_dir / "all_same_version_metrics.png",
        )
    return counts


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root_from_script()
    default_output = repo_root / "simulator_output" / "GV3_Agent" / "analysis_same_version"
    workspace = _workspace_root_from_repo(repo_root)
    default_nn = (
        workspace
        / "vidur-nn-search"
        / "simulator_output"
        / "GV3_Agent"
        / "BellmanConvergence"
        / "nn_agent_50v_transition_cache_float32_mp38_feature_mp38"
    )
    default_classical = (
        workspace
        / "vidur-classical-search"
        / "simulator_output"
        / "GV3_Agent"
        / "BellmanConvergence"
        / "classical_mp64_train32_featurecache_full_50v"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nn-run-dir", type=Path, default=default_nn)
    parser.add_argument("--classical-run-dir", type=Path, default=default_classical)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--min-version", type=int, default=2)
    parser.add_argument("--max-version", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    runs = [
        RunSpec("nn", args.nn_run_dir.expanduser().resolve(), output_root / "nn"),
        RunSpec(
            "classical",
            args.classical_run_dir.expanduser().resolve(),
            output_root / "classical",
        ),
    ]

    for run in runs:
        counts = plot_run(
            run,
            min_version=int(args.min_version),
            max_version=int(args.max_version),
        )
        print(
            f"[{run.name}] wrote plots to {run.output_dir} "
            f"(train_rows={counts.get('train', 0)}, eval_rows={counts.get('eval', 0)})"
        )


if __name__ == "__main__":
    main()
