"""Quick summary of arena results from a Model_Tester run.

In `--skip-model-ctrl-cycle` mode the harness runs only cycle1
(model_adv_depth1 vs trivial_ctrl). In that case the trivial controller drives
the game; the columns reflect the cost it accrues. To compute an apples-to-
apples model_ctrl win count we need the model_ctrl_cycle (cycle2) columns,
which aren't populated when --skip-model-ctrl-cycle is set.

For now this script prints both cycle1 (trivial-ctrl) and cycle2 (model-ctrl)
metrics. If model is winning, cycle2_total_cost should be << cycle1_total_cost
and `better_cycle` should be 'model_ctrl_depth1_vs_trivial_adv'.
"""
import csv
import sys
from pathlib import Path
import statistics


def summarize(p: Path) -> None:
    rows = []
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    n = len(rows)
    if n == 0:
        print(f"{p}: no rows")
        return
    label_c1 = rows[0].get("cycle1_label", "?")
    label_c2 = rows[0].get("cycle2_label", "")
    print(f"\n=== {p.parent.name} ===")
    print(f"  rows={n} cycle1_label={label_c1} cycle2_label={label_c2 or '<empty>'}")
    c1_cost = [float(r.get("cycle1_total_cost", 0) or 0) for r in rows]
    c1_viol = [int(r.get("cycle1_slo_violations", 0) or 0) for r in rows]
    c2_cost = [float(r.get("cycle2_total_cost", 0) or 0) for r in rows if r.get("cycle2_total_cost")]
    c2_viol = [int(r.get("cycle2_slo_violations", 0) or 0) for r in rows if r.get("cycle2_slo_violations")]
    better = [(r.get("better_cycle") or "").strip() for r in rows]
    delta = [float(r.get("cost_delta_cycle2_minus_cycle1") or 0.0) for r in rows]

    print(f"  cycle1 (trivial-ctrl) cost mean={statistics.mean(c1_cost):.3f} max={max(c1_cost):.3f} "
          f"viols mean={statistics.mean(c1_viol):.2f} max={max(c1_viol)}")
    if c2_cost:
        print(f"  cycle2 (model-ctrl) cost mean={statistics.mean(c2_cost):.3f} max={max(c2_cost):.3f} "
              f"viols mean={statistics.mean(c2_viol):.2f} max={max(c2_viol)}")
        print(f"  delta (c2-c1) mean={statistics.mean(delta):.3f} min={min(delta):.3f} max={max(delta):.3f}")
        n_model = sum(1 for d in delta if d < 0)
        n_trivial = sum(1 for d in delta if d > 0)
        n_tie = sum(1 for d in delta if d == 0)
        print(f"  WINS: model={n_model} trivial={n_trivial} ties={n_tie} (out of {n})")
    else:
        # cycle1-only mode: 'better_cycle' is just the cycle1 label echo.
        # The proper interpretation is via the CSV's per-game CSV files where
        # `model_cost` is the model_ctrl side. Use the log-file column instead.
        print(f"  (cycle2 empty — cycle1-only run; check log file for per-game model_cost)")
        # Tally per-game CSVs
        n_model_clean = 0
        n_trivial_clean = 0
        n_tie = 0
        for r in rows:
            pcsv = r.get("cycle1_log_file", "").strip()
            if not pcsv:
                continue
            pcsv_path = Path(pcsv)
            if not pcsv_path.exists():
                # may be on remote; try relative path inside same dir
                rel = Path(p.parent / "arena_games" / pcsv_path.name)
                if rel.exists():
                    pcsv_path = rel
                else:
                    continue
            # not used for now
        # Fall back to log file
        log = p.parent.with_suffix(".log")
        if log.exists():
            n_model = 0
            n_trivial = 0
            n_tie = 0
            with open(log) as f:
                for line in f:
                    if "model_cost=" in line and "trivial_cost=" in line:
                        try:
                            tc = float(line.split("trivial_cost=")[1].split()[0])
                            mc = float(line.split("model_cost=")[1].split()[0])
                        except Exception:
                            continue
                        if mc < tc:
                            n_model += 1
                        elif mc > tc:
                            n_trivial += 1
                        else:
                            n_tie += 1
            print(f"  WINS (from log): model={n_model} trivial={n_trivial} ties={n_tie}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(Path(arg))
