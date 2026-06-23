"""
Per-psid earliest-deadline-first (EDF) summary for V1 model outliers.

Reads <model_dir>/raw_state_request.csv produced by inspect_outliers.py.

For each unique psid (one row per psid in the output), computes:
  - time_to_edf = min over active requests of (deadline - sim_time)
      where deadline = decode_deadline if the request is in decode phase
      (is_prefill_complete=1 / type=decode), else prefill_deadline
  - n_requests_at_edf = number of active requests whose own (deadline - sim_time)
      equals time_to_edf (within a small epsilon)

Writes:
  <model_dir>/request_edf_analysis.csv with columns:
      psid, y_true, y_pred, time_to_edf, n_requests_at_edf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_MODEL_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/"
    "GV3_Agent/BellmanConvergence/cached_state_local_v1/V1_models/hgb_sq_31leaf_700iter"
)

EPS = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="dir containing raw_state_request.csv")
    args = parser.parse_args()
    SRC = args.model_dir / "raw_state_request.csv"
    OUT = args.model_dir / "request_edf_analysis.csv"
    by_psid: dict[int, dict] = {}

    with open(SRC, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx = {name: i for i, name in enumerate(header)}
        for raw in reader:
            row = [c.strip() for c in raw]
            psid = int(row[idx["psid"]])
            sim_time = float(row[idx["sim_time"]])
            type_tag = row[idx["type"]]
            if type_tag == "decode":
                deadline = float(row[idx["decode_deadline"]])
            else:
                deadline = float(row[idx["prefill_deadline"]])
            slack = deadline - sim_time

            entry = by_psid.setdefault(psid, {
                "y_true": float(row[idx["y_true"]]),
                "y_pred": float(row[idx["y_pred"]]),
                "slacks": [],
            })
            entry["slacks"].append(slack)

    rows_out = []
    for psid in sorted(by_psid):
        entry = by_psid[psid]
        slacks = entry["slacks"]
        edf_slack = min(slacks)
        n_at_edf = sum(1 for s in slacks if abs(s - edf_slack) <= EPS)
        rows_out.append({
            "psid": psid,
            "y_true": entry["y_true"],
            "y_pred": entry["y_pred"],
            "time_to_edf": edf_slack,
            "n_requests_at_edf": n_at_edf,
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["psid", "y_true", "y_pred", "time_to_edf", "n_requests_at_edf"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"[edf] wrote {OUT} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
