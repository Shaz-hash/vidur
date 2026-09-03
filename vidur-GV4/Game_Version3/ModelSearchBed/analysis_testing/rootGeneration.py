"""Generate the fixed ModelSearchBed root store for architecture experiments.

Default dataset:
    - 350,000 controller roots
    - 1,500,000 candidate-root budget
    - 16 local worker processes
    - at least 40% roots with abs(target_value) >= 1.0

The purpose is to create the root dataset once so different coding agents can
experiment with model architectures/features without regenerating simulator
states and Bellman labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


try:
    from ..root_storage import (
        DEFAULT_NONZERO_EPS,
        RootStorageStats,
        build_storage_config,
        generate_and_store_roots,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[6]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vidur.Game_Version3.ModelSearchBed.root_storage import (
        DEFAULT_NONZERO_EPS,
        RootStorageStats,
        build_storage_config,
        generate_and_store_roots,
    )


@dataclass(frozen=True)
class RootGenerationConfig:
    output_dir: Path
    num_roots: int = 350_000
    max_candidate_roots: int = 1_500_000
    num_processes: int = 16
    candidate_batch_size: int = 1024
    generation_batch_size: int = 32
    shard_size: int = 512
    worker_roots_per_task: int = 1024
    max_processes_per_interval: int = 4
    history_signature_cache_size: int = 10_000
    min_large_abs_target_ratio: float = 0.40
    target_abs_threshold: float = 1.0
    history_hops_min: int = 1
    history_hops_max: int = 200
    history_max_total_steps: int = 20_000
    seed: int = 2027
    nonzero_eps: float = DEFAULT_NONZERO_EPS
    root_player_filter: str = "controller"
    min_canonical_actions: int = 0
    include_history_trace_logs: bool = False
    overwrite: bool = False
    progress_every: int = 1000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_output_dir() -> Path:
    return (
        _repo_root()
        / "simulator_output"
        / "GV3_Agent"
        / "model_search_roots_controller_350k_abs1_ratio40"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _validate_config(cfg: RootGenerationConfig) -> None:
    if int(cfg.num_roots) <= 0:
        raise ValueError("num_roots must be > 0")
    if int(cfg.max_candidate_roots) < int(cfg.num_roots):
        raise ValueError("max_candidate_roots must be >= num_roots")
    if int(cfg.num_processes) <= 0:
        raise ValueError("num_processes must be > 0")
    if int(cfg.candidate_batch_size) <= 0:
        raise ValueError("candidate_batch_size must be > 0")
    if int(cfg.generation_batch_size) <= 0:
        raise ValueError("generation_batch_size must be > 0")
    if int(cfg.shard_size) <= 0:
        raise ValueError("shard_size must be > 0")
    if int(cfg.worker_roots_per_task) <= 0:
        raise ValueError("worker_roots_per_task must be > 0")
    if int(cfg.max_processes_per_interval) <= 0:
        raise ValueError("max_processes_per_interval must be > 0")
    if int(cfg.history_signature_cache_size) < 0:
        raise ValueError("history_signature_cache_size must be >= 0")
    if not (0.0 <= float(cfg.min_large_abs_target_ratio) <= 1.0):
        raise ValueError("min_large_abs_target_ratio must be in [0, 1]")
    if float(cfg.target_abs_threshold) < 0.0:
        raise ValueError("target_abs_threshold must be >= 0")
    if int(cfg.history_hops_min) < 0:
        raise ValueError("history_hops_min must be >= 0")
    if int(cfg.history_hops_max) < int(cfg.history_hops_min):
        raise ValueError("history_hops_max must be >= history_hops_min")
    if str(cfg.root_player_filter) not in {"controller", "adversary", "any"}:
        raise ValueError("root_player_filter must be 'controller', 'adversary', or 'any'")
    if int(cfg.min_canonical_actions) < 0:
        raise ValueError("min_canonical_actions must be >= 0")
    if int(cfg.progress_every) < 0:
        raise ValueError("progress_every must be >= 0")


def generate_roots(cfg: RootGenerationConfig) -> RootStorageStats:
    """Generate the large fixed controller-root store."""

    _validate_config(cfg)
    output_dir = cfg.output_dir.expanduser()
    manifest = output_dir / "manifest.jsonl"
    if manifest.exists() and not bool(cfg.overwrite):
        raise FileExistsError(
            f"dataset already exists at {output_dir}; use --overwrite to regenerate"
        )
    if output_dir.exists() and bool(cfg.overwrite):
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "root_generation_config.json", asdict(cfg))

    print(
        "starting root generation: "
        f"num_roots={int(cfg.num_roots)}, "
        f"max_candidate_roots={int(cfg.max_candidate_roots)}, "
        f"num_processes={int(cfg.num_processes)}, "
        f"candidate_batch_size={int(cfg.candidate_batch_size)}, "
        f"generation_batch_size={int(cfg.generation_batch_size)}, "
        f"worker_roots_per_task={int(cfg.worker_roots_per_task)}, "
        f"max_processes_per_interval={int(cfg.max_processes_per_interval)}, "
        f"history_signature_cache_size={int(cfg.history_signature_cache_size)}, "
        f"target_abs_threshold={float(cfg.target_abs_threshold)}, "
        f"min_large_abs_target_ratio={float(cfg.min_large_abs_target_ratio)}, "
        f"root_player_filter={str(cfg.root_player_filter)}, "
        f"min_canonical_actions={int(cfg.min_canonical_actions)}, "
        f"output_dir={output_dir}",
        flush=True,
    )

    storage_cfg = build_storage_config(
        output_dir=output_dir,
        num_roots=int(cfg.num_roots),
        max_candidate_roots=int(cfg.max_candidate_roots),
        candidate_batch_size=int(cfg.candidate_batch_size),
        generation_batch_size=int(cfg.generation_batch_size),
        min_nonzero_target_ratio=float(cfg.min_large_abs_target_ratio),
        nonzero_eps=float(cfg.nonzero_eps),
        target_abs_threshold=float(cfg.target_abs_threshold),
        history_hops_min=int(cfg.history_hops_min),
        history_hops_max=int(cfg.history_hops_max),
        history_max_total_steps=int(cfg.history_max_total_steps),
        shard_size=int(cfg.shard_size),
        seed=int(cfg.seed),
        start_player="adversary",
        root_player_filter=str(cfg.root_player_filter),
        min_canonical_actions=int(cfg.min_canonical_actions),
        include_history_trace_logs=bool(cfg.include_history_trace_logs),
        allow_duplicate_history_fallback=False,
        deduplicate_history_signatures=True,
        history_signature_cache_size=int(cfg.history_signature_cache_size),
        num_processes=int(cfg.num_processes),
        worker_roots_per_task=int(cfg.worker_roots_per_task),
        max_processes_per_interval=int(cfg.max_processes_per_interval),
        progress_every=int(cfg.progress_every),
    )
    stats = generate_and_store_roots(storage_cfg)

    print(
        "root generation complete: "
        f"roots_stored={int(stats.roots_stored)}, "
        f"large_abs_roots={int(stats.nonzero_roots_stored)}, "
        f"other_roots={int(stats.zero_roots_stored)}, "
        f"candidates_seen={int(stats.candidates_seen)}, "
        f"shards_written={int(stats.shards_written)}, "
        f"output_dir={output_dir}",
        flush=True,
    )
    return stats


def parse_args() -> RootGenerationConfig:
    parser = argparse.ArgumentParser(description="Generate GV3 ModelSearchBed root storage dataset.")
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--num-roots", type=int, default=350_000)
    parser.add_argument("--max-candidate-roots", type=int, default=1_500_000)
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=1024)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=32,
        help="Live root-state batch size per worker. Keep small to cap memory.",
    )
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument(
        "--worker-roots-per-task",
        type=int,
        default=1024,
        help=(
            "Accepted roots per recycled worker task. Lower values reduce "
            "memory growth from long-lived worker processes."
        ),
    )
    parser.add_argument("--max-processes-per-interval", type=int, default=4)
    parser.add_argument(
        "--history-signature-cache-size",
        type=int,
        default=10_000,
        help="Recent history signatures retained per worker task for bounded dedup.",
    )
    parser.add_argument("--min-large-abs-target-ratio", type=float, default=0.40)
    parser.add_argument("--target-abs-threshold", type=float, default=1.0)
    parser.add_argument("--history-hops-min", type=int, default=1)
    parser.add_argument("--history-hops-max", type=int, default=200)
    parser.add_argument("--history-max-total-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--nonzero-eps", type=float, default=DEFAULT_NONZERO_EPS)
    parser.add_argument("--root-player-filter", choices=("controller", "adversary", "any"), default="controller")
    parser.add_argument(
        "--min-canonical-actions",
        type=int,
        default=0,
        help="Keep only roots with at least this many canonical actions.",
    )
    parser.add_argument("--include-history-trace-logs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    return RootGenerationConfig(
        output_dir=Path(args.output_dir),
        num_roots=int(args.num_roots),
        max_candidate_roots=int(args.max_candidate_roots),
        num_processes=int(args.num_processes),
        candidate_batch_size=int(args.candidate_batch_size),
        generation_batch_size=int(args.generation_batch_size),
        shard_size=int(args.shard_size),
        worker_roots_per_task=int(args.worker_roots_per_task),
        max_processes_per_interval=int(args.max_processes_per_interval),
        history_signature_cache_size=int(args.history_signature_cache_size),
        min_large_abs_target_ratio=float(args.min_large_abs_target_ratio),
        target_abs_threshold=float(args.target_abs_threshold),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_max_total_steps=int(args.history_max_total_steps),
        seed=int(args.seed),
        nonzero_eps=float(args.nonzero_eps),
        root_player_filter=str(args.root_player_filter),
        min_canonical_actions=int(args.min_canonical_actions),
        include_history_trace_logs=bool(args.include_history_trace_logs),
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
    )


def main() -> None:
    generate_roots(parse_args())


if __name__ == "__main__":
    main()
