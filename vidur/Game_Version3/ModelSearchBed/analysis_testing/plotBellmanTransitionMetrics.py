"""Plot Bellman transition metrics across model versions.

This script reads Bellman convergence transition summary files:

    version_0_to_1.csv
    version_1_to_2.csv
    ...
    version_49_to_50.csv

It intentionally ignores same-model convergence files such as
`version_10_to_10.csv`. The output is written under:

    simulator_output/GV3_Agent/analysis/{nn,classical}/{train,eval}/

Each split directory gets one CSV plus individual metric plots and a compact
four-panel summary plot.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRANSITION_RE = re.compile(r"^version_(\d+)_to_(\d+)\.csv$")

METRICS: tuple[tuple[str, str], ...] = (
    ("rmse", "RMSE"),
    ("p95_abs_error", "p95 Absolute Error"),
    ("max_abs_error", "Max Absolute Error"),
    ("num_abs_error_ge_1", "# Samples with abs(error) >= 1"),
)


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


def default_run_specs(output_root: Path) -> list[RunSpec]:
    """Build default NN/classical inputs from the current local worktree layout."""

    repo_root = _repo_root_from_script()
    workspace = _workspace_root_from_repo(repo_root)
    return [
        RunSpec(
            name="nn",
            input_dir=workspace
            / "vidur-nn-search"
            / "simulator_output"
            / "GV3_Agent"
            / "BellmanConvergence"
            / "nn_agent_50v_transition_cache_float32_mp38_feature_mp38",
            output_dir=output_root / "nn",
        ),
        RunSpec(
            name="classical",
            input_dir=workspace
            / "vidur-classical-search"
            / "simulator_output"
            / "GV3_Agent"
            / "BellmanConvergence"
            / "classical_mp64_train32_featurecache_full_50v",
            output_dir=output_root / "classical",
        ),
    ]


def read_transition_rows(input_dir: Path, max_version: int) -> list[dict[str, str]]:
    """Read only sequential transition CSVs up to `max_version`.

    `version_0_to_1.csv` contributes model version 1, and
    `version_49_to_50.csv` contributes model version 50.
    """

    if not input_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {input_dir}")

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for source_version in range(0, int(max_version)):
        target_version = source_version + 1
        path = input_dir / f"version_{source_version}_to_{target_version}.csv"
        if not path.exists():
            missing.append(path.name)
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["source_file"] = path.name
                row["iteration"] = str(target_version)
                rows.append(row)

    if missing:
        missing_preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(
            f"missing transition summaries in {input_dir}: {missing_preview}{suffix}"
        )
    if not rows:
        raise RuntimeError(f"no transition rows found in {input_dir}")
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
    ax.set_xlabel("Iteration / Model Version")
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
        ax.set_xlabel("Iteration / Model Version")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(x), max(x))
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_run(run: RunSpec, max_version: int) -> dict[str, int]:
    """Create train/eval plots for one NN or classical run."""

    rows = read_transition_rows(run.input_dir, max_version=max_version)
    counts: dict[str, int] = {}
    run.output_dir.mkdir(parents=True, exist_ok=True)
    (run.output_dir / "source_run.txt").write_text(str(run.input_dir) + "\n")

    for split in ("train", "eval"):
        split_rows = rows_for_split(rows, split)
        if len(split_rows) != int(max_version):
            raise RuntimeError(
                f"{run.name} {split} has {len(split_rows)} rows; expected {max_version}"
            )
        split_dir = run.output_dir / split
        write_split_csv(split_rows, split_dir / "transition_metrics.csv")
        for metric, metric_title in METRICS:
            plot_metric(
                split_rows,
                metric=metric,
                title=f"{run.name} {split}: {metric_title}",
                output_path=split_dir / f"{metric}.png",
            )
        plot_summary_grid(
            split_rows,
            title=f"{run.name} {split}: Bellman Transition Metrics",
            output_path=split_dir / "all_transition_metrics.png",
        )
        counts[split] = len(split_rows)
    return counts


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root_from_script()
    default_output = repo_root / "simulator_output" / "GV3_Agent" / "analysis"
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
        counts = plot_run(run, max_version=int(args.max_version))
        print(
            f"[{run.name}] wrote plots to {run.output_dir} "
            f"(train_rows={counts['train']}, eval_rows={counts['eval']})"
        )


if __name__ == "__main__":
    main()
