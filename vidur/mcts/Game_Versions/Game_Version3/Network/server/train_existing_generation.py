from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from ..network_config import DEFAULT_NETWORK_CONFIG, namespace_path_defaults, repo_root, resolve_output_name
from .dataset_integrity import validate_or_repair_generation_from_received
from .training import train_network_generation


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return repo_root() / p


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message)
        if not message.endswith("\n"):
            f.write("\n")


def _parse_args() -> argparse.Namespace:
    defaults = DEFAULT_NETWORK_CONFIG.task
    paths = DEFAULT_NETWORK_CONFIG.paths
    parser = argparse.ArgumentParser(
        description="Train/evaluate one GV3 network generation from already-collected dataset shards."
    )
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--model-version", type=int, default=None)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Simulator output namespace. Defaults to Game_Version3; use Game_Version3_Native for isolated native runs.",
    )
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--total-roots", type=int, default=defaults.total_roots_per_generation)
    parser.add_argument("--roots-per-cycle", type=int, default=defaults.roots_per_cycle)
    parser.add_argument("--sample-cycles-per-generation", type=int, default=defaults.sample_cycles_per_generation)
    parser.add_argument("--local-training-device", default=defaults.local_training_device)
    parser.add_argument("--train-batch-size", type=int, default=defaults.local_train_batch_size)
    parser.add_argument("--train-target-epochs", type=float, default=defaults.local_train_target_epochs_per_generation)
    parser.add_argument("--train-progress-every-steps", type=int, default=defaults.local_train_progress_print_every_steps)
    parser.add_argument("--train-num-threads", type=int, default=defaults.local_train_num_threads)
    parser.add_argument("--replay-capacity-samples", type=int, default=defaults.local_replay_capacity_samples)
    parser.add_argument("--replay-max-cached-shards", type=int, default=defaults.local_replay_max_cached_shards)
    parser.add_argument("--replay-seed", type=int, default=defaults.local_replay_seed)
    parser.add_argument("--trainer-lr", type=float, default=0.0)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--eval-metrics-csv", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--process-log-path", default=None)
    parser.add_argument("--received-root", default=None)
    parser.add_argument("--skip-dataset-repair", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generation = int(args.generation)
    model_version = int(generation + 1 if args.model_version is None else args.model_version)
    output_name = resolve_output_name(args.output_name)
    ns_paths = namespace_path_defaults(output_name)
    cycles = max(1, int(args.sample_cycles_per_generation))
    if int(args.roots_per_cycle) > 0:
        roots_per_cycle = int(args.roots_per_cycle)
    else:
        roots_per_cycle = int((int(args.total_roots) + int(cycles) - 1) // int(cycles))
    total_roots_required = int(roots_per_cycle) * int(cycles)

    process_log_path = _resolve_path(args.process_log_path or ns_paths["process_log_path"])
    weights_path = _resolve_path(args.weights_path or ns_paths["default_weights_path"])
    dataset_dir = _resolve_path(args.dataset_dir or ns_paths["dataset_dir"])
    logs_dir = _resolve_path(args.logs_dir or ns_paths["logs_dir"])
    eval_metrics_csv = _resolve_path(args.eval_metrics_csv or ns_paths["eval_metrics_csv"])
    checkpoints_dir = _resolve_path(args.checkpoints_dir or ns_paths["checkpoints_dir"])
    received_root = _resolve_path(args.received_root or (ns_paths["output_dir"] / "received"))

    _append_log(
        process_log_path,
        (
            f"[GV3 network server] train-existing starting: generation={generation}, "
            f"model_version={model_version}, dataset={dataset_dir}, weights={weights_path}"
        ),
    )
    if not bool(args.skip_dataset_repair):
        repair_stats = validate_or_repair_generation_from_received(
            generation=generation,
            dataset_dir=dataset_dir,
            received_root=received_root,
            log_line=lambda message: _append_log(process_log_path, message),
        )
        _append_log(
            process_log_path,
            (
                f"[GV3 gen={generation:06d}] dataset integrity check complete: "
                f"validated_shards={int(repair_stats['validated_shards'])}, "
                f"repaired_shards={int(repair_stats['repaired_shards'])}, "
                f"samples={int(repair_stats['samples'])}"
            ),
        )
    summary = train_network_generation(
        generation=generation,
        model_version=model_version,
        total_roots_required=total_roots_required,
        roots_per_cycle=roots_per_cycle,
        sample_cycles_per_generation=cycles,
        dataset_dir=dataset_dir,
        logs_dir=logs_dir,
        eval_metrics_csv=eval_metrics_csv,
        checkpoints_dir=checkpoints_dir,
        initial_checkpoint_path=weights_path,
        collection_stats={},
        local_training_device=str(args.local_training_device),
        train_batch_size=int(args.train_batch_size),
        train_target_epochs=float(args.train_target_epochs),
        train_progress_every_steps=int(args.train_progress_every_steps),
        train_num_threads=int(args.train_num_threads),
        replay_capacity_samples=int(args.replay_capacity_samples),
        replay_max_cached_shards=int(args.replay_max_cached_shards),
        replay_seed=int(args.replay_seed),
        trainer_lr=float(args.trainer_lr),
        log_line=lambda message: _append_log(process_log_path, message),
    )
    _append_log(
        process_log_path,
        (
            f"[GV3 network server] train-existing complete: generation={generation}, "
            f"model_version={model_version}, checkpoint={summary.checkpoint_path}, "
            f"train_samples={summary.train_samples}, eval_samples={summary.eval_samples}"
        ),
    )
    print(asdict(summary), flush=True)


if __name__ == "__main__":
    main()
