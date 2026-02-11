#!/usr/bin/env python3

## (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)


"""
python3 vidur/vidur/mcts/tests/scripts/plot_adv_best5_ratio.py 24 102 

python3 -m vidur.mcts.tests.scripts.adversary_action 24 102 --out simulator_output/mcts_dnn_logs/adv_best5_ratio_gen24_vs_gen102.png


"""


from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class GenStats:
    gen_name: str
    adv_total: int
    adv_best5: int

    @property
    def ratio(self) -> float:
        return (self.adv_best5 / self.adv_total) if self.adv_total > 0 else 0.0


def _normalize_gen_arg(s: str) -> str:
    s = str(s).strip()
    if s.startswith("gen_"):
        return s
    # allow passing "24" or "000024"
    if s.isdigit():
        return f"gen_{int(s):06d}"
    raise ValueError(f"Invalid generation arg {s!r}. Use e.g. 24 or gen_000024.")


def _iter_root_csvs(gen_dir: Path) -> List[Path]:
    # Prefer per-process files if present; also include mcts_root.csv if present
    files = sorted(gen_dir.glob("mcts_root_p*.csv"))
    root_single = gen_dir / "mcts_root.csv"
    if root_single.exists():
        files.append(root_single)
    return files


def _count_adv_best5_in_file(path: Path) -> Tuple[int, int]:
    adv_total = 0
    adv_best5 = 0

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return (0, 0)

        for row in reader:
            rp = (row.get("root_player") or "").strip().lower()
            if rp != "adversary":
                continue

            adv_total += 1
            bai = (row.get("best_action_index") or "").strip()
            try:
                if int(float(bai)) == 5:
                    adv_best5 += 1
            except Exception:
                # if missing/invalid, just ignore (still counts toward adv_total)
                pass

    return adv_total, adv_best5


def compute_generation_stats(logs_root: Path, gen_name: str) -> GenStats:
    gen_dir = logs_root / gen_name
    if not gen_dir.exists():
        raise FileNotFoundError(f"Missing generation dir: {gen_dir}")

    files = _iter_root_csvs(gen_dir)
    if not files:
        raise FileNotFoundError(f"No mcts_root*.csv files found in: {gen_dir}")

    adv_total = 0
    adv_best5 = 0

    for p in files:
        a, b = _count_adv_best5_in_file(p)
        adv_total += a
        adv_best5 += b

    return GenStats(gen_name=gen_name, adv_total=adv_total, adv_best5=adv_best5)


def plot_bar(stats: List[GenStats], *, out_path: Path, title: Optional[str] = None, show: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it (pip install matplotlib) or run with --no-plot."
        ) from e

    labels = [s.gen_name for s in stats]
    values = [s.ratio for s in stats]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("adv best_action_index==5 ratio")
    ax.set_xlabel("generation")
    ax.set_title(title or "Adversary best_action_index==5 ratio by generation")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # annotate bars
    for bar, s in zip(bars, stats):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{s.ratio:.3f}\n({s.adv_best5}/{s.adv_total})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen1", help="e.g. 24 or gen_000024")
    ap.add_argument("gen2", help="e.g. 102 or gen_000102")
    ap.add_argument(
        "--logs-root",
        default="simulator_output/mcts_dnn_logs",
        help="Root directory containing gen_XXXXXX folders.",
    )
    ap.add_argument(
        "--out",
        default="simulator_output/mcts_dnn_logs/adv_best5_ratio.png",
        help="Output plot path (.png).",
    )
    ap.add_argument("--show", action="store_true", help="Also show the plot window.")
    ap.add_argument("--no-plot", action="store_true", help="Only print stats, do not generate plot.")
    args = ap.parse_args()

    logs_root = Path(args.logs_root)
    gen1 = _normalize_gen_arg(args.gen1)
    gen2 = _normalize_gen_arg(args.gen2)

    s1 = compute_generation_stats(logs_root, gen1)
    s2 = compute_generation_stats(logs_root, gen2)

    print(f"{s1.gen_name}: adv_best5={s1.adv_best5} adv_total={s1.adv_total} ratio={s1.ratio:.6f}")
    print(f"{s2.gen_name}: adv_best5={s2.adv_best5} adv_total={s2.adv_total} ratio={s2.ratio:.6f}")

    if not args.no_plot:
        out_path = Path(args.out)
        title = f"Adversary best_action_index==5 ratio: {s1.gen_name} vs {s2.gen_name}"
        plot_bar([s1, s2], out_path=out_path, title=title, show=bool(args.show))
        print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
