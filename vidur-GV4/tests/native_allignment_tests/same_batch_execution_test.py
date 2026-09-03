from __future__ import annotations

import csv
import math
from pathlib import Path

from common import REPO_ROOT, import_native_cpp, output_dir
from vidur.mcts.prefill_calibrator import PrefillProfile


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_prefill_profile_lookup_matches_native_cpp() -> None:
    native = import_native_cpp()
    profile_path = REPO_ROOT / "simulator_output" / "prefill_profile.csv"
    if not profile_path.exists():
        raise FileNotFoundError(f"missing prefill profile: {profile_path}")

    profile = PrefillProfile.load(profile_path)
    tokens = sorted(int(k) for k in profile.entries.keys())
    times = [float(profile.entries[t]) for t in tokens]
    queries = sorted(set(tokens + [1, 127, 128, 129, 255, 256, 257, 511, 512, 768, 1024, 1536, 2048, 3072, 4096]))

    native_values = [float(x) for x in native.debug_prefill_profile_lookup(tokens, times, queries)]
    rows = []
    failures = []
    for q, nv in zip(queries, native_values):
        pyv = float(profile.lookup(int(q)))
        diff = abs(float(nv) - float(pyv))
        ok = math.isclose(float(nv), float(pyv), rel_tol=0.0, abs_tol=1e-12)
        if not ok:
            failures.append((q, pyv, nv, diff))
        rows.append({"query_tokens": q, "python_lookup": pyv, "native_lookup": nv, "abs_diff": diff, "passed": ok})

    _write_rows(output_dir("same_batch_execution") / "prefill_lookup_alignment.csv", rows)
    assert not failures, f"native prefill lookup mismatch: {failures[:10]}"


if __name__ == "__main__":
    test_prefill_profile_lookup_matches_native_cpp()
    print("same_batch_execution_test passed")
