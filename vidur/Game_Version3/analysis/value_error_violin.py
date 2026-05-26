from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


DEFAULT_INPUT_CSV = (
    "simulator_output/Game_Version3_Fresh8_Hops200/"
    "mcts_dnn_logs/gen_000000/eval_root_value_predictions.csv"
)


def _repo_root() -> Path:
    # value_error_violin.py -> analysis -> Game_Version3 -> Game_Versions
    # -> mcts -> vidur(package) -> repo root.
    return Path(__file__).resolve().parents[5]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _read_errors(input_csv: Path) -> tuple[list[dict[str, float | str]], str]:
    rows: list[dict[str, float | str]] = []
    model_versions: set[str] = set()
    with input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"model_predicted_value", "true_tree_search_value", "root_player_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_csv} missing required columns: {sorted(missing)}")

        for row in reader:
            pred = float(row["model_predicted_value"])
            target = float(row["true_tree_search_value"])
            player = str(row.get("root_player_type", "") or "unknown")
            model_version = str(row.get("model_version", "") or "")
            if model_version:
                model_versions.add(model_version)
            rows.append(
                {
                    "model_version": model_version,
                    "root_number": str(row.get("root_number", "")),
                    "root_player_type": player,
                    "model_predicted_value": pred,
                    "true_tree_search_value": target,
                    "signed_error": pred - target,
                    "abs_error": abs(pred - target),
                }
            )

    if not rows:
        raise RuntimeError(f"no rows found in {input_csv}")

    label = ",".join(sorted(model_versions)) if model_versions else input_csv.parent.name
    return rows, label


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _summarize(values: list[float]) -> dict[str, float | int]:
    abs_values = [abs(x) for x in values]
    return {
        "count": len(values),
        "mean_signed_error": mean(values) if values else 0.0,
        "median_signed_error": median(values) if values else 0.0,
        "mean_abs_error": mean(abs_values) if abs_values else 0.0,
        "median_abs_error": median(abs_values) if abs_values else 0.0,
        "p95_abs_error": _percentile(abs_values, 0.95),
        "max_abs_error": max(abs_values) if abs_values else 0.0,
    }


def _write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    groups["all"] = []
    for row in rows:
        err = float(row["signed_error"])
        groups["all"].append(err)
        groups[str(row["root_player_type"])].append(err)

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group",
        "count",
        "mean_signed_error",
        "median_signed_error",
        "mean_abs_error",
        "median_abs_error",
        "p95_abs_error",
        "max_abs_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group in ["all", "controller", "adversary"]:
            if group not in groups:
                continue
            stats = _summarize(groups[group])
            writer.writerow({"group": group, **stats})


def _write_error_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_version",
        "root_number",
        "root_player_type",
        "model_predicted_value",
        "true_tree_search_value",
        "signed_error",
        "abs_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_violin(
    path: Path,
    rows: list[dict[str, float | str]],
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    grouped["all"] = []
    for row in rows:
        err = float(row["signed_error"])
        grouped["all"].append(err)
        grouped[str(row["root_player_type"])].append(err)

    labels = [label for label in ["all", "controller", "adversary"] if grouped.get(label)]
    data = [grouped[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_edgecolor("#26384f")
        body.set_alpha(0.55)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#222222")
            parts[key].set_linewidth(1.2)

    ax.axhline(0.0, color="#b00020", linestyle="--", linewidth=1.4, label="zero error")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prediction error = model predicted value - tree-search target")
    ax.set_xlabel("Eval root group")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    input_csv = _resolve_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")

    rows, model_label = _read_errors(input_csv)
    filter_suffix = ""
    filter_title = ""
    if bool(args.nonzero_target_only):
        eps = max(0.0, float(args.target_zero_eps))
        before = len(rows)
        rows = [
            row
            for row in rows
            if abs(float(row["true_tree_search_value"])) > float(eps)
        ]
        if not rows:
            raise RuntimeError(
                f"no rows remain after true_tree_search_value nonzero filter: "
                f"input={input_csv}, eps={eps}"
            )
        filter_suffix = "_nonzero_target"
        filter_title = f"\nFiltered to true_tree_search_value != 0, eps={eps:g} ({len(rows)}/{before} rows)"

    output_leaf = f"{input_csv.parent.name}{filter_suffix}"
    output_dir = _resolve_path(args.output_dir) if args.output_dir else input_csv.parents[2] / "analysis" / "value_error_violin" / output_leaf
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"{output_leaf}_value_error_violin.png"
    summary_path = output_dir / f"{output_leaf}_value_error_summary.csv"
    error_rows_path = output_dir / f"{output_leaf}_value_errors.csv"

    title = (
        f"GV3 eval value error distribution ({input_csv.parent.name}, model_version={model_label})\n"
        "Model prediction vs generated tree-search/Bellman target"
        f"{filter_title}"
    )
    _plot_violin(plot_path, rows, title=title)
    _write_summary(summary_path, rows)
    _write_error_rows(error_rows_path, rows)

    print(f"plot={plot_path}")
    print(f"summary_csv={summary_path}")
    print(f"errors_csv={error_rows_path}")
    print(f"rows={len(rows)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create violin plots for GV3 model prediction error against tree-search target values."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--nonzero-target-only",
        action="store_true",
        help="Only plot rows where true_tree_search_value is nonzero.",
    )
    parser.add_argument(
        "--target-zero-eps",
        type=float,
        default=1e-12,
        help="Absolute tolerance used by --nonzero-target-only.",
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
