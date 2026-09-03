from __future__ import annotations

import argparse
import csv
import io
import subprocess
from typing import Iterable

from .config import load_config


def _gpu_rows() -> list[dict[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, int]] = []
    for values in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
        index, used, free, utilization = (int(value.strip()) for value in values)
        rows.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "utilization_percent": utilization,
            }
        )
    return rows


def select_gpu(*, max_used_mib: int, max_utilization: int) -> int:
    allowed = set(load_config().gpu_candidates)
    candidates = [
        row
        for row in _gpu_rows()
        if row["index"] in allowed
        and row["memory_used_mib"] <= max_used_mib
        and row["utilization_percent"] <= max_utilization
    ]
    if not candidates:
        raise RuntimeError(
            "neither permitted GPU 2 nor GPU 3 is idle; no process was modified"
        )
    return int(max(candidates, key=lambda row: row["memory_free_mib"])["index"])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select an idle permitted mew1 GPU")
    parser.add_argument("--max-used-mib", type=int, default=4096)
    parser.add_argument("--max-utilization", type=int, default=5)
    args = parser.parse_args(argv)
    print(
        select_gpu(
            max_used_mib=args.max_used_mib,
            max_utilization=args.max_utilization,
        )
    )


if __name__ == "__main__":
    main()
