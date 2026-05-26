"""Generate an additional controller-root dataset with external signature exclusion.

This script is for extending the ModelSearchBed root store without duplicating
history signatures already present in an existing dataset. It uses the same
root_storage multiprocessing machinery as the original large generation script,
but pre-seeds the shared history-signature table from one or more existing
datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


try:
    from ..root_storage import (
        DEFAULT_NONZERO_EPS,
        RootStorageStats,
        _history_signature_key,
        build_storage_config,
        generate_and_store_roots,
        load_history_signature_keys_from_dataset,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[6]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vidur.Game_Version3.ModelSearchBed.root_storage import (
        DEFAULT_NONZERO_EPS,
        RootStorageStats,
        _history_signature_key,
        build_storage_config,
        generate_and_store_roots,
        load_history_signature_keys_from_dataset,
    )


@dataclass(frozen=True)
class UniqueRootGenerationConfig:
    output_dir: Path
    exclude_signature_dataset_dirs: tuple[Path, ...]
    num_roots: int = 300_000
    max_candidate_roots: int = 8_000_000
    num_processes: int = 16
    candidate_batch_size: int = 512
    generation_batch_size: int = 4
    shard_size: int = 256
    worker_roots_per_task: int = 512
    max_processes_per_interval: int = 4
    min_large_abs_target_ratio: float = 0.40
    target_abs_threshold: float = 1.0
    history_hops_min: int = 1
    history_hops_max: int = 200
    history_max_total_steps: int = 20_000
    seed: int = 3031
    nonzero_eps: float = DEFAULT_NONZERO_EPS
    include_history_trace_logs: bool = False
    overwrite: bool = False
    progress_every: int = 1000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_existing_dataset_dir() -> Path:
    return (
        _repo_root()
        / "simulator_output"
        / "GV3_Agent"
        / "model_search_roots_controller_350k_abs1_ratio40"
    )


def _default_output_dir() -> Path:
    return (
        _repo_root()
        / "simulator_output"
        / "GV3_Agent"
        / "model_search_roots_controller_extra_300k_abs1_ratio40_unique_from_292k"
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


def _validate_config(cfg: UniqueRootGenerationConfig) -> None:
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
    if not cfg.exclude_signature_dataset_dirs:
        raise ValueError("at least one exclude signature dataset dir is required")
    for dataset_dir in cfg.exclude_signature_dataset_dirs:
        manifest = Path(dataset_dir).expanduser() / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"exclude dataset manifest not found: {manifest}")
    if not (0.0 <= float(cfg.min_large_abs_target_ratio) <= 1.0):
        raise ValueError("min_large_abs_target_ratio must be in [0, 1]")
    if float(cfg.target_abs_threshold) < 0.0:
        raise ValueError("target_abs_threshold must be >= 0")
    if int(cfg.history_hops_min) < 0:
        raise ValueError("history_hops_min must be >= 0")
    if int(cfg.history_hops_max) < int(cfg.history_hops_min):
        raise ValueError("history_hops_max must be >= history_hops_min")
    if int(cfg.history_max_total_steps) <= 0:
        raise ValueError("history_max_total_steps must be > 0")
    if int(cfg.progress_every) < 0:
        raise ValueError("progress_every must be >= 0")


def _iter_manifest_records(dataset_dir: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            shard_rel = str(entry["shard_path"])
            shard_path = dataset_dir / shard_rel
            try:
                records = torch.load(shard_path, map_location="cpu", weights_only=False)
            except TypeError:
                records = torch.load(shard_path, map_location="cpu")
            if not isinstance(records, list):
                raise TypeError(f"shard did not contain a list: {shard_rel}")
            yield shard_rel, records


def validate_unique_dataset(cfg: UniqueRootGenerationConfig) -> dict[str, Any]:
    """Validate loadability, target ratio, and signature uniqueness."""

    output_dir = cfg.output_dir.expanduser()
    excluded_keys: set[Any] = set()
    for dataset_dir in cfg.exclude_signature_dataset_dirs:
        excluded_keys.update(load_history_signature_keys_from_dataset(dataset_dir))

    seen_new: set[Any] = set()
    duplicate_with_excluded: list[dict[str, Any]] = []
    duplicate_inside_new: list[dict[str, Any]] = []
    missing_signature_count = 0
    controller_count = 0
    large_abs_count = 0
    large_negative_best_reward_count = 0
    target_reward_max_abs_diff = 0.0
    root_count = 0
    shard_count = 0

    for shard_rel, records in _iter_manifest_records(output_dir):
        shard_count += 1
        for local_index, record in enumerate(records):
            root_count += 1
            if str(record.get("root_player", "")) == "controller":
                controller_count += 1
            target = float(record.get("target_value", 0.0))
            if abs(target) >= float(cfg.target_abs_threshold):
                large_abs_count += 1
            best_reward = float(record.get("best_reward", 0.0))
            if best_reward <= -float(cfg.target_abs_threshold):
                large_negative_best_reward_count += 1
            target_reward_max_abs_diff = max(
                float(target_reward_max_abs_diff),
                abs(float(target) - float(best_reward)),
            )
            sig = record.get("history_signature", None)
            if sig is None:
                missing_signature_count += 1
                continue
            key = _history_signature_key(sig)
            if key in excluded_keys and len(duplicate_with_excluded) < 20:
                duplicate_with_excluded.append(
                    {
                        "shard_path": shard_rel,
                        "local_index": int(local_index),
                        "root_id": int(record.get("root_id", -1)),
                    }
                )
            if key in seen_new and len(duplicate_inside_new) < 20:
                duplicate_inside_new.append(
                    {
                        "shard_path": shard_rel,
                        "local_index": int(local_index),
                        "root_id": int(record.get("root_id", -1)),
                    }
                )
            seen_new.add(key)
        del records

    large_abs_ratio = float(large_abs_count) / float(root_count) if root_count > 0 else 0.0
    large_negative_best_reward_ratio = (
        float(large_negative_best_reward_count) / float(root_count) if root_count > 0 else 0.0
    )
    summary = {
        "output_dir": str(output_dir),
        "exclude_signature_dataset_dirs": [str(path) for path in cfg.exclude_signature_dataset_dirs],
        "excluded_signature_count": int(len(excluded_keys)),
        "root_count": int(root_count),
        "expected_num_roots": int(cfg.num_roots),
        "shard_count": int(shard_count),
        "controller_count": int(controller_count),
        "large_abs_count": int(large_abs_count),
        "large_abs_ratio": float(large_abs_ratio),
        "large_negative_best_reward_count": int(large_negative_best_reward_count),
        "large_negative_best_reward_ratio": float(large_negative_best_reward_ratio),
        "required_large_abs_ratio": float(cfg.min_large_abs_target_ratio),
        "target_reward_max_abs_diff": float(target_reward_max_abs_diff),
        "missing_signature_count": int(missing_signature_count),
        "unique_new_signature_count": int(len(seen_new)),
        "duplicate_with_excluded_examples": duplicate_with_excluded,
        "duplicate_inside_new_examples": duplicate_inside_new,
        "passed": (
            int(root_count) == int(cfg.num_roots)
            and int(controller_count) == int(root_count)
            and int(missing_signature_count) == 0
            and not duplicate_with_excluded
            and not duplicate_inside_new
            and large_abs_ratio + 1e-12 >= float(cfg.min_large_abs_target_ratio)
            and large_negative_best_reward_ratio + 1e-12
            >= float(cfg.min_large_abs_target_ratio)
        ),
    }
    _write_json(output_dir / "unique_generation_validation.json", summary)
    with (output_dir / "unique_generation_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "root_count",
                "expected_num_roots",
                "shard_count",
                "controller_count",
                "large_abs_count",
                "large_abs_ratio",
                "large_negative_best_reward_count",
                "large_negative_best_reward_ratio",
                "required_large_abs_ratio",
                "target_reward_max_abs_diff",
                "missing_signature_count",
                "unique_new_signature_count",
                "excluded_signature_count",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerow({key: summary[key] for key in writer.fieldnames})

    if not bool(summary["passed"]):
        raise RuntimeError(f"unique dataset validation failed: {summary}")
    return summary


def generate_unique_roots(cfg: UniqueRootGenerationConfig) -> RootStorageStats:
    """Generate and validate a new root dataset unique from excluded datasets."""

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
    _write_json(output_dir / "root_generation_unique_config.json", asdict(cfg))

    print(
        "starting unique root generation: "
        f"num_roots={int(cfg.num_roots)}, "
        f"max_candidate_roots={int(cfg.max_candidate_roots)}, "
        f"num_processes={int(cfg.num_processes)}, "
        f"candidate_batch_size={int(cfg.candidate_batch_size)}, "
        f"generation_batch_size={int(cfg.generation_batch_size)}, "
        f"worker_roots_per_task={int(cfg.worker_roots_per_task)}, "
        f"history_hops=[{int(cfg.history_hops_min)}, {int(cfg.history_hops_max)}], "
        f"target_abs_threshold={float(cfg.target_abs_threshold)}, "
        f"min_large_abs_target_ratio={float(cfg.min_large_abs_target_ratio)}, "
        f"exclude_dirs={[str(path) for path in cfg.exclude_signature_dataset_dirs]}, "
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
        root_player_filter="controller",
        include_history_trace_logs=bool(cfg.include_history_trace_logs),
        allow_duplicate_history_fallback=False,
        deduplicate_history_signatures=True,
        history_signature_cache_size=0,
        num_processes=int(cfg.num_processes),
        worker_roots_per_task=int(cfg.worker_roots_per_task),
        max_processes_per_interval=int(cfg.max_processes_per_interval),
        exclude_signature_dataset_dirs=cfg.exclude_signature_dataset_dirs,
        progress_every=int(cfg.progress_every),
    )
    stats = generate_and_store_roots(storage_cfg)
    validation = validate_unique_dataset(cfg)

    print(
        "unique root generation complete: "
        f"roots_stored={int(stats.roots_stored)}, "
        f"large_abs_roots={int(stats.nonzero_roots_stored)}, "
        f"other_roots={int(stats.zero_roots_stored)}, "
        f"candidates_seen={int(stats.candidates_seen)}, "
        f"shards_written={int(stats.shards_written)}, "
        f"validation_passed={bool(validation['passed'])}, "
        f"output_dir={output_dir}",
        flush=True,
    )
    return stats


def parse_args() -> UniqueRootGenerationConfig:
    parser = argparse.ArgumentParser(
        description="Generate an additional GV3 ModelSearchBed root dataset with unique history signatures."
    )
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument(
        "--exclude-signature-dataset-dir",
        action="append",
        default=[],
        help="Existing root dataset to exclude by history_signature. May be supplied multiple times.",
    )
    parser.add_argument("--num-roots", type=int, default=300_000)
    parser.add_argument("--max-candidate-roots", type=int, default=8_000_000)
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=512)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--worker-roots-per-task", type=int, default=512)
    parser.add_argument("--max-processes-per-interval", type=int, default=4)
    parser.add_argument("--min-large-abs-target-ratio", type=float, default=0.40)
    parser.add_argument("--target-abs-threshold", type=float, default=1.0)
    parser.add_argument("--history-hops-min", type=int, default=1)
    parser.add_argument("--history-hops-max", type=int, default=200)
    parser.add_argument("--history-max-total-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=3031)
    parser.add_argument("--nonzero-eps", type=float, default=DEFAULT_NONZERO_EPS)
    parser.add_argument("--include-history-trace-logs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    exclude_args = list(args.exclude_signature_dataset_dir)
    if not exclude_args:
        exclude_args = [str(_default_existing_dataset_dir())]
    exclude_dirs = tuple(Path(path).expanduser() for path in exclude_args)
    return UniqueRootGenerationConfig(
        output_dir=Path(args.output_dir).expanduser(),
        exclude_signature_dataset_dirs=exclude_dirs,
        num_roots=int(args.num_roots),
        max_candidate_roots=int(args.max_candidate_roots),
        num_processes=int(args.num_processes),
        candidate_batch_size=int(args.candidate_batch_size),
        generation_batch_size=int(args.generation_batch_size),
        shard_size=int(args.shard_size),
        worker_roots_per_task=int(args.worker_roots_per_task),
        max_processes_per_interval=int(args.max_processes_per_interval),
        min_large_abs_target_ratio=float(args.min_large_abs_target_ratio),
        target_abs_threshold=float(args.target_abs_threshold),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_max_total_steps=int(args.history_max_total_steps),
        seed=int(args.seed),
        nonzero_eps=float(args.nonzero_eps),
        include_history_trace_logs=bool(args.include_history_trace_logs),
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
    )


def main() -> None:
    generate_unique_roots(parse_args())


if __name__ == "__main__":
    main()
