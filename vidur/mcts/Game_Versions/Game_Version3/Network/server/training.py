from __future__ import annotations

import gc
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from ...DNN.dnn_spec import make_dnn_spec
from ...DNN.replay_buffer import BestModelReplayBuffer
from ...DNN.replay_dataset import collate_mixed_samples
from ...DNN.trainer import Trainer, TrainerConfig
from ...DNN.value_models import AlphaZeroModel
from ...config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig
from ...logger.evaluation_pipeline_logger import EvaluationMetricsLogger, EvalRootPredictionLogger
from ...multiProcessUtils import (
    _aggregate_metric_rows,
    _build_eval_prediction_rows,
    _compute_train_steps_this_generation,
    _evaluate_samples,
    _load_generation_samples,
    _load_trainer_from_checkpoint,
    _safe_float,
    _scan_generation_dataset_stats,
    _set_global_seeds,
    _set_runtime_cpu_thread_env,
    _set_torch_intraop_threads,
)
from ..network_config import resolve_device


LogLine = Callable[[str], None]


@dataclass(frozen=True)
class NetworkTrainingSummary:
    generation: int
    model_version: int
    train_samples: int
    eval_samples: int
    train_steps: int
    checkpoint_path: str
    best_path: str
    latest_path: str
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float]


def _default_log_line(message: str) -> None:
    print(message, flush=True)


def _raise_if_nonfinite_metrics(*, metrics: dict[str, Any], context: str) -> None:
    required_keys = {
        "loss",
        "policy_loss",
        "value_loss",
        "value_mse_error",
        "value_mae_error",
        "loss_for_selection",
        "controller_value_mse_error",
        "controller_value_mae_error",
        "adversary_value_mse_error",
        "adversary_value_mae_error",
    }
    bad: list[str] = []
    for key, value in dict(metrics).items():
        if str(key) not in required_keys:
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                bad.append(str(key))
    if bad:
        raise RuntimeError(f"Non-finite training metric(s) in {context}: {', '.join(sorted(bad))}")


def _network_training_cfg(
    *,
    dataset_dir: Path,
    logs_dir: Path,
    eval_metrics_csv: Path,
    checkpoints_dir: Path,
    roots_per_cycle: int,
    sample_cycles_per_generation: int,
    local_training_device: str,
    train_batch_size: int,
    train_target_epochs: float,
    train_progress_every_steps: int,
    train_num_threads: int,
    replay_capacity_samples: int,
    replay_max_cached_shards: int,
    replay_seed: int,
    trainer_lr: float = 0.0,
) -> MultipleProcessTrainingConfig:
    base = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
    cfg = replace(
        base,
        model=replace(base.model, device=resolve_device(local_training_device)),
        dataset=replace(base.dataset, out_dir=str(Path(dataset_dir))),
        logging=replace(
            base.logging,
            mcts_iter_log=str(Path(logs_dir) / "mcts_iter.csv"),
            mcts_root_log=str(Path(logs_dir) / "mcts_root.csv"),
        ),
        evaluation_logging=replace(base.evaluation_logging, metrics_csv=str(Path(eval_metrics_csv))),
        checkpoints_dir=str(Path(checkpoints_dir)),
        roots_per_generation=max(1, int(roots_per_cycle)),
        sample_cycles_per_generation=max(1, int(sample_cycles_per_generation)),
        num_generations=1,
        replay_capacity_samples=max(1, int(replay_capacity_samples)),
        replay_max_cached_shards=max(1, int(replay_max_cached_shards)),
        replay_seed=int(replay_seed),
    )
    if int(train_batch_size) > 0:
        cfg = replace(cfg, train_batch_size=int(train_batch_size))
    if float(train_target_epochs) > 0.0:
        cfg = replace(cfg, train_target_epochs_per_generation=float(train_target_epochs))
    if int(train_progress_every_steps) > 0:
        cfg = replace(cfg, train_progress_print_every_steps=int(train_progress_every_steps))
    if int(train_num_threads) > 0:
        cfg = replace(cfg, train_num_threads=int(train_num_threads))
    if float(trainer_lr) > 0.0:
        cfg = replace(
            cfg,
            game_v2=replace(
                cfg.game_v2,
                trainer=replace(cfg.game_v2.trainer, lr=float(trainer_lr)),
            ),
        )
    cfg.validate()
    return cfg


def _make_trainer(cfg: MultipleProcessTrainingConfig) -> Trainer:
    spec = make_dnn_spec(cfg=cfg.game_v2)
    model = AlphaZeroModel(spec=spec)
    trainer_h = cfg.game_v2.trainer
    return Trainer(
        model=model,
        cfg=TrainerConfig(
            lr=float(trainer_h.lr),
            weight_decay=float(trainer_h.weight_decay),
            policy_weight=float(trainer_h.policy_weight),
            value_weight=float(trainer_h.value_weight),
            value_loss_alpha=float(trainer_h.value_loss_alpha),
            value_only=bool(trainer_h.value_only),
            grad_clip_norm=float(trainer_h.grad_clip_norm),
            checkpoint_every=int(trainer_h.checkpoint_every),
            eval_every=int(trainer_h.eval_every),
            invalid_logit=float(trainer_h.invalid_logit),
        ),
        device=torch.device(cfg.model.device),
    )


def train_network_generation(
    *,
    generation: int,
    model_version: int,
    total_roots_required: int,
    roots_per_cycle: int,
    sample_cycles_per_generation: int,
    dataset_dir: Path,
    logs_dir: Path,
    eval_metrics_csv: Path,
    checkpoints_dir: Path,
    initial_checkpoint_path: Path,
    collection_stats: dict[str, int],
    local_training_device: str = "auto",
    train_batch_size: int = 0,
    train_target_epochs: float = 0.0,
    train_progress_every_steps: int = 0,
    train_num_threads: int = 0,
    replay_capacity_samples: int = 400_000,
    replay_max_cached_shards: int = 5_000,
    replay_seed: int = 2026,
    trainer_lr: float = 0.0,
    log_line: LogLine | None = None,
) -> NetworkTrainingSummary:
    log = log_line or _default_log_line
    gen = int(generation)
    trained_model_version = int(gen if model_version is None else model_version)

    cfg = _network_training_cfg(
        dataset_dir=Path(dataset_dir),
        logs_dir=Path(logs_dir),
        eval_metrics_csv=Path(eval_metrics_csv),
        checkpoints_dir=Path(checkpoints_dir),
        roots_per_cycle=int(roots_per_cycle),
        sample_cycles_per_generation=int(sample_cycles_per_generation),
        local_training_device=str(local_training_device),
        train_batch_size=int(train_batch_size),
        train_target_epochs=float(train_target_epochs),
        train_progress_every_steps=int(train_progress_every_steps),
        train_num_threads=int(train_num_threads),
        replay_capacity_samples=int(replay_capacity_samples),
        replay_max_cached_shards=int(replay_max_cached_shards),
        replay_seed=int(replay_seed),
        trainer_lr=float(trainer_lr),
    )

    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed) + int(gen),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )
    _set_runtime_cpu_thread_env(int(getattr(cfg, "train_num_threads", 1)))

    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"
    initial_checkpoint_path = Path(initial_checkpoint_path)
    if not best_ckpt_path.exists() and initial_checkpoint_path.exists():
        shutil.copyfile(initial_checkpoint_path, best_ckpt_path)

    trainer = _make_trainer(cfg)
    load_path = initial_checkpoint_path if initial_checkpoint_path.exists() else best_ckpt_path
    if load_path.exists():
        _load_trainer_from_checkpoint(trainer, load_path)
    elif not best_ckpt_path.exists():
        trainer.save_checkpoint(best_ckpt_path)

    replay_buffer = BestModelReplayBuffer(
        capacity_samples=int(cfg.replay_capacity_samples),
        max_cached_shards=int(cfg.replay_max_cached_shards),
        seed=int(cfg.replay_seed) + int(gen),
    )

    gen_dataset_dir = Path(dataset_dir) / f"gen_{int(gen):06d}"
    train_source_dir = gen_dataset_dir / "train"
    eval_source_dir = gen_dataset_dir / "eval"
    gen_logs_dir = Path(logs_dir) / f"gen_{int(gen):06d}"
    gen_logs_dir.mkdir(parents=True, exist_ok=True)

    if not train_source_dir.exists():
        raise FileNotFoundError(f"network train dataset dir not found: {train_source_dir}")
    if not eval_source_dir.exists():
        eval_source_dir.mkdir(parents=True, exist_ok=True)

    added_from_gen = replay_buffer.add_generation_dir(train_source_dir)
    if replay_buffer.total_samples <= 0:
        raise RuntimeError(
            f"Replay buffer empty after network generation={gen}, "
            f"added_from_generation={added_from_gen}, dir={train_source_dir}"
        )

    train_steps_this_gen = _compute_train_steps_this_generation(cfg, replay_buffer.total_samples)
    actual_train_threads = _set_torch_intraop_threads(int(getattr(cfg, "train_num_threads", 1)))
    if bool(getattr(cfg, "replay_preload_all_shards_each_generation", False)):
        preload_limit = int(getattr(cfg, "replay_preload_max_shards", 512))
        log(
            f"[GV3 gen={int(gen):06d}] replay preload starting: "
            f"num_shards={int(replay_buffer.num_shards)}, "
            f"max_preload_shards={int(preload_limit)}"
        )
        preload_stats = replay_buffer.preload_all_shards(max_shards=preload_limit)
        log(
            f"[GV3 gen={int(gen):06d}] replay preload complete: "
            f"loaded_shards={int(preload_stats['loaded_shards'])}, "
            f"cached_shards={int(preload_stats['cached_shards'])}, "
            f"num_shards={int(preload_stats['num_shards'])}"
        )

    effective_epochs = (
        float(train_steps_this_gen) * float(int(cfg.train_batch_size))
    ) / float(max(1, int(replay_buffer.total_samples)))
    steps_per_epoch = max(
        1,
        int(
            math.ceil(
                float(max(1, int(replay_buffer.total_samples)))
                / float(max(1, int(cfg.train_batch_size)))
            )
        ),
    )
    epoch_count = max(1, int(math.ceil(float(train_steps_this_gen) / float(steps_per_epoch))))
    log(
        f"[GV3 gen={int(gen):06d}] training starting: "
        f"replay_total_samples={int(replay_buffer.total_samples)}, "
        f"added_from_generation={int(added_from_gen)}, "
        f"train_steps={int(train_steps_this_gen)}, batch_size={int(cfg.train_batch_size)}, "
        f"target_epochs={float(cfg.train_target_epochs_per_generation):.3f}, "
        f"effective_epochs={float(effective_epochs):.3f}, "
        f"steps_per_epoch={int(steps_per_epoch)}, epochs={int(epoch_count)}, "
        f"progress_every_steps={int(getattr(cfg, 'train_progress_print_every_steps', 100))}, "
        f"train_threads={int(actual_train_threads)}"
    )

    train_rows: list[dict] = []
    global_step = 0
    progress_print_every = max(1, int(getattr(cfg, "train_progress_print_every_steps", 100)))
    progress_window_start = time.perf_counter()
    for epoch_idx in range(int(epoch_count)):
        epoch_rows: list[dict] = []
        progress_rows: list[dict] = []
        epoch_steps = min(int(steps_per_epoch), int(train_steps_this_gen) - int(global_step))
        if epoch_steps <= 0:
            break

        for _ in range(int(epoch_steps)):
            samples = replay_buffer.sample_batch(int(cfg.train_batch_size))
            batch_by_player = collate_mixed_samples(
                samples,
                device=trainer.device,
                include_policy_tensors=not bool(trainer.cfg.value_only),
                include_legacy_fallback=False,
                include_ids=False,
            )
            row = trainer.train_step(batch_by_player)
            _raise_if_nonfinite_metrics(
                metrics=row,
                context=(
                    f"gen={int(gen):06d}, global_step={int(global_step) + 1}/"
                    f"{int(train_steps_this_gen)}"
                ),
            )
            train_rows.append(row)
            epoch_rows.append(row)
            progress_rows.append(row)
            global_step += 1

            if (
                (int(global_step) % int(progress_print_every) == 0)
                or (int(global_step) == int(train_steps_this_gen))
            ):
                progress_metrics = _aggregate_metric_rows(progress_rows)
                epoch_step_idx = len(epoch_rows)
                elapsed_sec = max(0.0, float(time.perf_counter() - progress_window_start))
                steps_in_window = max(1, len(progress_rows))
                log(
                    f"[GV3 gen={int(gen):06d}] training progress: "
                    f"epoch={int(epoch_idx) + 1}/{int(epoch_count)}, "
                    f"epoch_step={int(epoch_step_idx)}/{int(epoch_steps)}, "
                    f"global_step={int(global_step)}/{int(train_steps_this_gen)}, "
                    f"window_steps={int(steps_in_window)}, "
                    f"window_sec={float(elapsed_sec):.3f}, "
                    f"sec_per_step={float(elapsed_sec) / float(steps_in_window):.3f}, "
                    f"loss={float(progress_metrics['loss_for_selection']):.6f}, "
                    f"value_mse={float(progress_metrics['value_mse_error']):.6f}, "
                    f"value_mae={float(progress_metrics['value_mae_error']):.6f}"
                )
                progress_rows.clear()
                progress_window_start = time.perf_counter()

        epoch_metrics = _aggregate_metric_rows(epoch_rows)
        log(
            f"[GV3 gen={int(gen):06d}] epoch {int(epoch_idx) + 1}/{int(epoch_count)} complete: "
            f"steps={int(global_step)}/{int(train_steps_this_gen)}, "
            f"loss={float(epoch_metrics['loss_for_selection']):.6f}, "
            f"value_mse={float(epoch_metrics['value_mse_error']):.6f}, "
            f"value_mae={float(epoch_metrics['value_mae_error']):.6f}"
        )

    train_metrics = _aggregate_metric_rows(train_rows)
    _raise_if_nonfinite_metrics(metrics=train_metrics, context=f"gen={int(gen):06d} train aggregate")

    eval_samples = _load_generation_samples(eval_source_dir)
    log(
        f"[GV3 gen={int(gen):06d}] evaluation starting: "
        f"eval_samples={int(len(eval_samples))}, batch_size={int(cfg.train_batch_size)}"
    )
    eval_metrics = _evaluate_samples(
        trainer,
        samples=eval_samples,
        batch_size=int(cfg.train_batch_size),
    )
    _raise_if_nonfinite_metrics(metrics=eval_metrics, context=f"gen={int(gen):06d} eval aggregate")
    eval_prediction_rows = _build_eval_prediction_rows(
        trainer,
        samples=eval_samples,
        batch_size=int(cfg.train_batch_size),
        model_version=int(trained_model_version),
    )
    EvalRootPredictionLogger(gen_logs_dir / "eval_root_value_predictions.csv").write_rows(
        eval_prediction_rows
    )

    gen_ckpt_path = ckpt_dir / f"gen_{int(gen):06d}.pt"
    trainer.save_checkpoint(gen_ckpt_path)
    shutil.copyfile(gen_ckpt_path, latest_path)

    select_loss = _safe_float(eval_metrics.get("loss_for_selection", float("nan")))
    if math.isnan(select_loss):
        select_loss = _safe_float(train_metrics.get("loss_for_selection", float("nan")))
    if not math.isnan(select_loss):
        trainer.best_eval_loss = float(select_loss)
    trainer.save_checkpoint(best_ckpt_path)

    scanned_stats = _scan_generation_dataset_stats(
        train_dir=train_source_dir,
        eval_dir=eval_source_dir,
    )
    stats: dict[str, Any] = {**dict(collection_stats or {}), **scanned_stats}
    common_row = {
        "generation": int(gen),
        "model_version": int(trained_model_version),
        "num_roots_required": int(total_roots_required),
        "num_unique_roots_created": int(stats.get("num_unique_roots", 0)),
        "controller_training_samples": int(stats.get("controller_train_samples", 0)),
        "controller_eval_samples": int(stats.get("controller_eval_samples", 0)),
        "adversary_training_samples": int(stats.get("adversary_train_samples", 0)),
        "adversary_eval_samples": int(stats.get("adversary_eval_samples", 0)),
    }

    eval_metrics_logger = EvaluationMetricsLogger(Path(eval_metrics_csv), flush_every=1)
    try:
        eval_metrics_logger.log_phase(
            **common_row,
            phase="training",
            samples_needed=int(train_metrics["samples_needed"]),
            value_mse_error=float(train_metrics["value_mse_error"]),
            value_mae_error=float(train_metrics["value_mae_error"]),
            controller_value_mse_error=float(train_metrics["controller_value_mse_error"]),
            controller_value_mae_error=float(train_metrics["controller_value_mae_error"]),
            adversary_value_mse_error=float(train_metrics["adversary_value_mse_error"]),
            adversary_value_mae_error=float(train_metrics["adversary_value_mae_error"]),
        )
        eval_metrics_logger.log_phase(
            **common_row,
            phase="eval",
            samples_needed=int(eval_metrics["samples_needed"]),
            value_mse_error=float(eval_metrics["value_mse_error"]),
            value_mae_error=float(eval_metrics["value_mae_error"]),
            controller_value_mse_error=float(eval_metrics["controller_value_mse_error"]),
            controller_value_mae_error=float(eval_metrics["controller_value_mae_error"]),
            adversary_value_mse_error=float(eval_metrics["adversary_value_mse_error"]),
            adversary_value_mae_error=float(eval_metrics["adversary_value_mae_error"]),
        )
    finally:
        eval_metrics_logger.close()

    train_samples_total = int(replay_buffer.total_samples)
    summary = NetworkTrainingSummary(
        generation=int(gen),
        model_version=int(trained_model_version),
        train_samples=int(train_samples_total),
        eval_samples=int(len(eval_samples)),
        train_steps=int(train_steps_this_gen),
        checkpoint_path=str(gen_ckpt_path),
        best_path=str(best_ckpt_path),
        latest_path=str(latest_path),
        train_metrics=dict(train_metrics),
        eval_metrics=dict(eval_metrics),
    )

    replay_buffer.reset_for_new_best()
    del train_rows
    del eval_samples
    del eval_prediction_rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
