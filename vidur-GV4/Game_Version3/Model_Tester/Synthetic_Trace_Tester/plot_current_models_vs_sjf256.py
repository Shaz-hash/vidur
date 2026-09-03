from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_RUNS = {
    "xl3_controller_v197_adv_v212": "XL3 controller v197",
    "xl4_controller_v126_adv_v203": "XL4 controller v126",
}


def _read_one_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one result row in {path}, found {len(rows)}")
    return rows[0]


def _prefill_budgets(steps_csv: Path) -> set[int]:
    budgets: set[int] = set()
    with steps_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            action = row.get("action_repr", "")
            match = re.search(r"prefill_allocations=(\{.*?\}), decode_allocations=", action)
            if not match:
                continue
            allocations = ast.literal_eval(match.group(1))
            if allocations:
                budgets.add(sum(int(value) for value in allocations.values()))
    return budgets


def _sjf_summary(root: Path, trace_index: int) -> tuple[dict[str, str], Path]:
    trace_root = root.parent
    if trace_index == 0:
        summary = trace_root / "eval_gv3_legal_20s" / "sjf256" / "summary.csv"
    else:
        summary = (
            trace_root
            / "eval_gv3_legal_variants_20s"
            / f"trace_{trace_index:02d}"
            / "sjf256"
            / "summary.csv"
        )
    row = _read_one_row(summary)
    steps = summary.parent / "trace_steps.csv"
    budgets = _prefill_budgets(steps)
    if not budgets or max(budgets) != 256:
        raise ValueError(
            f"{steps} is not the expected SJF-256 baseline; observed prefill budgets={sorted(budgets)}"
        )
    return row, summary


def _collect(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for trace_index in range(5):
        trace_dir = root / "results" / f"trace_{trace_index:02d}"
        sjf, sjf_source = _sjf_summary(root, trace_index)
        for model_dir, model_label in MODEL_RUNS.items():
            model_source = trace_dir / model_dir / "summary.csv"
            model = _read_one_row(model_source)
            records.append(
                {
                    "trace": f"Trace {trace_index}",
                    "trace_index": trace_index,
                    "model": model_label,
                    "model_version": int(model["model_version"]),
                    "model_total_cost": float(model["total_cost"]),
                    "sjf256_total_cost": float(sjf["total_cost"]),
                    "model_slo_violations": int(model["slo_violations"]),
                    "sjf256_slo_violations": int(sjf["slo_violations"]),
                    "model_total_lateness": float(model["total_lateness"]),
                    "sjf256_total_lateness": float(sjf["total_lateness"]),
                    "model_wins": float(model["total_cost"]) < float(sjf["total_cost"]),
                    "model_source": str(model_source.resolve()),
                    "sjf256_source": str(sjf_source.resolve()),
                }
            )
    return records


def _write_csv(records: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot(records: list[dict[str, object]], model_label: str, output: Path) -> None:
    selected = sorted(
        (row for row in records if row["model"] == model_label),
        key=lambda row: int(row["trace_index"]),
    )
    labels = [str(row["trace"]) for row in selected]
    model_costs = np.asarray([float(row["model_total_cost"]) for row in selected])
    sjf_costs = np.asarray([float(row["sjf256_total_cost"]) for row in selected])
    wins = int(np.sum(model_costs < sjf_costs))
    reduction = 100.0 * (1.0 - float(np.mean(model_costs)) / float(np.mean(sjf_costs)))

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)
    model_bars = ax.bar(
        x - width / 2,
        model_costs,
        width,
        label=model_label,
        color="#176B87",
    )
    sjf_bars = ax.bar(
        x + width / 2,
        sjf_costs,
        width,
        label="SJF-256",
        color="#E97937",
    )
    ax.bar_label(model_bars, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(sjf_bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Final total cost (SLO violations + total lateness; lower is better)")
    ax.set_title(
        f"{model_label} vs SJF-256 on five 20-second synthetic traces\n"
        f"model mean={np.mean(model_costs):.2f}, SJF mean={np.mean(sjf_costs):.2f}, "
        f"model wins={wins}/5, mean cost reduction={reduction:.1f}%"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(float(np.max(model_costs)), float(np.max(sjf_costs))) * 1.18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the current XL3 and XL4 controllers against matched SJF-256 baselines."
    )
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    root = args.run_root.expanduser().resolve()
    records = _collect(root)
    output_dir = root / "plots"
    _write_csv(records, output_dir / "model_vs_sjf256_outcomes.csv")
    for model_dir, model_label in MODEL_RUNS.items():
        _plot(records, model_label, output_dir / f"{model_dir}_vs_sjf256_total_cost.png")


if __name__ == "__main__":
    main()
