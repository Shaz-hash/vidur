## (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
cd /home/shazer/Desktop/Research/Vidur/vidur && \
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.linear.sampler.run \
  --target-unique-states 1000 \
  --workers 1 \
  --max-trace-length 10 \
  --max-branching 10 \
  --max-forced-hops 20000 \
  --history-hops 10 \
  --parquet-compression snappy \
  --out-dir simulator_output/linear_sampler_fast \
  --export-compat-root \
  --export-compat-iter



cd /home/shazer/Desktop/Research/Vidur/vidur && \
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.linear.sampler.run \
  --target-unique-states 125000 \
  --workers 8 \
  --max-trace-length 8 \
  --max-branching 10 \
  --max-forced-hops 5000 \
  --history-hops 0,15,30,45,60,75,90,105 \
  --parquet-compression snappy \
  --out-dir simulator_output/linear_sampler_fast_LP






"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import (
    SamplerCollectionSettings,
    SamplerOutputSettings,
    SamplerRunConfig,
    parse_history_hops,
)
from .sampler_parallel import run_parallel_sampler


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Depth-aware deduplicated sampler for LP-ready linear states")
    ap.add_argument(
        "--target-unique-states",
        type=int,
        default=400000,
        help="Target count of unique accepted anchor samples (branching states).",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-depth", type=int, default=0)
    ap.add_argument(
        "--max-trace-length",
        type=int,
        default=10,
        help="Per-sample random trace length cap; L is drawn uniformly from [1, max_trace_length].",
    )
    ap.add_argument("--max-branching", type=int, default=10)
    ap.add_argument("--enum-max-samples", type=int, default=10000)
    ap.add_argument("--max-forced-hops", type=int, default=20000)
    ap.add_argument("--max-total-steps-per-state", type=int, default=512)
    ap.add_argument("--max-snapshot-cache", type=int, default=40960)
    ap.add_argument("--shard-unique-states-per-worker", type=int, default=10000)
    ap.add_argument("--max-expansions-per-worker", type=int, default=800000)
    ap.add_argument("--max-rounds", type=int, default=100)
    ap.add_argument("--history-hops", type=str, default="0,5,10,15,20,25,30,35")
    ap.add_argument("--history-csv", type=str, default="")
    ap.add_argument("--history-hop-mode", type=str, default="fixed", choices=["fixed", "cyclic"])
    ap.add_argument("--root-player", type=str, default="adversary", choices=["controller", "adversary"])
    ap.add_argument("--align-branching-roots", dest="align_branching_roots", action="store_true", default=True)
    ap.add_argument("--no-align-branching-roots", dest="align_branching_roots", action="store_false")
    ap.add_argument("--use-virtual-env", dest="use_virtual_env", action="store_true", default=True)
    ap.add_argument("--use-real-env", dest="use_virtual_env", action="store_false")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", type=str, default="simulator_output/linear_sampler")
    ap.add_argument("--parquet-compression", type=str, default="zstd")
    ap.add_argument("--export-compat-root", action="store_true", default=False)
    ap.add_argument("--export-compat-iter", action="store_true", default=False)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.max_trace_length) < 1:
        raise ValueError("--max-trace-length must be >= 1")
    if int(args.target_unique_states) < 1:
        raise ValueError("--target-unique-states must be >= 1")
    hops = parse_history_hops(str(args.history_hops))

    cfg = SamplerRunConfig(
        collection=SamplerCollectionSettings(
            target_unique_states=int(args.target_unique_states),
            workers=int(args.workers),
            start_depth=int(args.start_depth),
            max_trace_length=int(args.max_trace_length),
            max_branching=int(args.max_branching),
            enum_max_samples=int(args.enum_max_samples),
            max_forced_hops=int(args.max_forced_hops),
            max_total_steps_per_state=int(args.max_total_steps_per_state),
            max_snapshot_cache=int(args.max_snapshot_cache),
            shard_unique_states_per_worker=int(args.shard_unique_states_per_worker),
            max_expansions_per_worker=int(args.max_expansions_per_worker),
            max_rounds=int(args.max_rounds),
            history_hops=hops,
            history_csv=str(args.history_csv),
            root_player=str(args.root_player),
            align_branching_roots=bool(args.align_branching_roots),
            use_virtual_env=bool(args.use_virtual_env),
        ),
        output=SamplerOutputSettings(
            out_dir=str(args.out_dir),
            parquet_compression=str(args.parquet_compression),
        ),
        seed=int(args.seed),
    )

    Path(cfg.output.out_dir).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(cfg.output.out_dir) / "sampler_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": cfg.seed,
                "collection": {
                    "target_unique_states": cfg.collection.target_unique_states,
                    "workers": cfg.collection.workers,
                    "start_depth": cfg.collection.start_depth,
                    "max_trace_length": cfg.collection.max_trace_length,
                    "max_branching": cfg.collection.max_branching,
                    "enum_max_samples": cfg.collection.enum_max_samples,
                    "max_forced_hops": cfg.collection.max_forced_hops,
                    "max_total_steps_per_state": cfg.collection.max_total_steps_per_state,
                    "max_snapshot_cache": cfg.collection.max_snapshot_cache,
                    "shard_unique_states_per_worker": cfg.collection.shard_unique_states_per_worker,
                    "max_expansions_per_worker": cfg.collection.max_expansions_per_worker,
                    "max_rounds": cfg.collection.max_rounds,
                    "history_hops": list(cfg.collection.history_hops),
                    "history_csv": cfg.collection.history_csv,
                    "root_player": cfg.collection.root_player,
                    "align_branching_roots": cfg.collection.align_branching_roots,
                    "use_virtual_env": cfg.collection.use_virtual_env,
                },
                "output": {
                    "out_dir": cfg.output.out_dir,
                    "parquet_compression": cfg.output.parquet_compression,
                    "export_compat_root": bool(args.export_compat_root),
                    "export_compat_iter": bool(args.export_compat_iter),
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )

    results = run_parallel_sampler(
        cfg,
        export_compat_root=bool(args.export_compat_root),
        export_compat_iter=bool(args.export_compat_iter),
    )
    if not results:
        raise RuntimeError("sampler produced no rounds")
    last = results[-1]
    print(
        f"[linear.sampler] completed rounds={len(results)} "
        f"last_round={last.round_idx} unique_states={last.merge_output.unique_states} "
        f"unique_anchor_samples={last.merge_output.unique_anchor_samples} "
        f"controller_lp_samples={last.merge_output.controller_lp_samples} "
        f"adversary_lp_samples={last.merge_output.adversary_lp_samples} "
        f"compat_iter={last.compat_output.mcts_iter_compat_csv} "
        f"compat_root={last.compat_output.mcts_root_compat_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
