from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from vidur.AlphaGoZero.agz_train_eval_promote import (
    _ensure_policy_cache,
    _ensure_state_cache,
    _feature_replay_paths,
    _policy_replay_paths,
)


def _build_state_cache(path: str) -> tuple[str, bool, float]:
    cache_path, rebuilt, elapsed = _ensure_state_cache(Path(path))
    return str(cache_path), bool(rebuilt), float(elapsed)


def _build_policy_cache(path: str) -> tuple[str, bool, float]:
    cache_path, rebuilt, elapsed = _ensure_policy_cache(Path(path))
    return str(cache_path), bool(rebuilt), float(elapsed)


def _run_parallel(
    *,
    phase: str,
    paths: list[Path],
    worker_fn: Callable[[str], tuple[str, bool, float]],
    workers: int,
    progress_interval: int,
) -> dict[str, float | int]:
    t0 = time.time()
    total = len(paths)
    rebuilt = 0
    completed = 0
    if total == 0:
        return {"files": 0, "rebuilt": 0, "elapsed_s": 0.0}

    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(worker_fn, str(path)) for path in paths]
        for future in as_completed(futures):
            _, did_rebuild, last_elapsed = future.result()
            completed += 1
            rebuilt += int(did_rebuild)
            if completed % int(progress_interval) == 0 or completed == total or did_rebuild:
                print(
                    {
                        "phase": phase,
                        "files": completed,
                        "total": total,
                        "rebuilt": rebuilt,
                        "last_elapsed_s": round(float(last_elapsed), 3),
                        "elapsed_s": round(time.time() - t0, 1),
                    },
                    flush=True,
                )
    return {"files": total, "rebuilt": rebuilt, "elapsed_s": round(time.time() - t0, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild AlphaGoZero replay cache files in parallel.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="AlphaGoZero output root containing global_replay.",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--skip-state", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    t0 = time.time()
    summary: dict[str, dict[str, float | int]] = {}
    if not args.skip_state:
        state_paths = _feature_replay_paths(root)
        print({"phase": "state", "files": len(state_paths), "workers": int(args.workers)}, flush=True)
        summary["state"] = _run_parallel(
            phase="state",
            paths=state_paths,
            worker_fn=_build_state_cache,
            workers=int(args.workers),
            progress_interval=int(args.progress_interval),
        )
    if not args.skip_policy:
        policy_paths = _policy_replay_paths(root)
        print({"phase": "policy", "files": len(policy_paths), "workers": int(args.workers)}, flush=True)
        summary["policy"] = _run_parallel(
            phase="policy",
            paths=policy_paths,
            worker_fn=_build_policy_cache,
            workers=int(args.workers),
            progress_interval=int(args.progress_interval),
        )
    print({"summary": summary, "elapsed_s": round(time.time() - t0, 3)}, flush=True)


if __name__ == "__main__":
    main()
