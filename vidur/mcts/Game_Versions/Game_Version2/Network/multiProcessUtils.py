# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import json
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import torch

from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.eval_utils import grade_arena_from_game_logs
from ..DNN.export_torchscript_gv2 import export_torchscript_artifacts_gv2
from ..DNN.models import AlphaZeroModel
from ..DNN.replay_buffer import BestModelReplayBuffer
from ..DNN.replay_dataset import collate_mixed_samples
from ..DNN.trainer import Trainer, TrainerConfig
from ..config import MultipleProcessTrainingConfig
from ..logger.evaluation_pipeline_logger import (
    EvaluationMetricsLogger,
    generation_log_dir_from_iter_log,
)
from ..multiProcessUtils import (
    _append_train_log_row,
    _build_arena_game_entries,
    _compute_train_steps_this_generation,
    _load_trainer_from_checkpoint,
    _next_generation_index,
    _set_global_seeds,
)
from .aws_backend import AwsNetworkBackend
from .types import ArenaTask, RemoteModelRef, RemoteResult, SelfplayTask


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _rewrite_manifest_paths(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    updated: List[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            basename = Path(str(entry.get("path", ""))).name
            if basename:
                entry["path"] = str((manifest_path.parent / basename).resolve())
            updated.append(json.dumps(entry, ensure_ascii=False))
    with manifest_path.open("w", encoding="utf-8") as f:
        for line in updated:
            f.write(line + "\n")


def _remote_selfplay_hop(cfg: MultipleProcessTrainingConfig, generation: int, task_index: int) -> int:
    base = list(cfg.selfplay_hops_for_generation(generation))
    if task_index < len(base):
        return int(base[task_index])
    rng = random.Random(int(cfg.history_seed) + int(generation) * 100_000 + int(task_index))
    return int(rng.randint(int(cfg.history_hops_min), int(cfg.history_hops_max)))


def _chunk_entries(entries: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    chunk_size = max(1, int(chunk_size))
    return [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]


def _upload_model_ref(
    backend: AwsNetworkBackend,
    *,
    session_id: str,
    generation: int,
    label: str,
    checkpoint_path: Path,
    model_version: int,
    spec: Any,
    native_on: bool,
    torchscript_root: Path,
) -> RemoteModelRef:
    weights_key = backend.model_key(session_id, generation, f"{label}.pt")
    backend.upload_file(checkpoint_path, weights_key)

    controller_key = ""
    adversary_key = ""
    if native_on:
        ts = export_torchscript_artifacts_gv2(
            checkpoint_path=checkpoint_path,
            out_dir=torchscript_root / label,
            model_version=int(model_version),
            spec=spec,
            device="cpu",
        )
        controller_key = backend.model_key(session_id, generation, f"{label}_controller.pt")
        adversary_key = backend.model_key(session_id, generation, f"{label}_adversary.pt")
        backend.upload_file(Path(ts.controller_path), controller_key)
        backend.upload_file(Path(ts.adversary_path), adversary_key)

    return RemoteModelRef(
        weights_s3_key=str(weights_key),
        model_version=int(model_version),
        controller_ts_s3_key=str(controller_key),
        adversary_ts_s3_key=str(adversary_key),
    )


def _enqueue_selfplay_task(
    *,
    backend: AwsNetworkBackend,
    session_id: str,
    generation: int,
    task_index: int,
    count: int,
    start_root_id: int,
    game_id: int,
    model_ref: RemoteModelRef,
    cfg: MultipleProcessTrainingConfig,
) -> SelfplayTask:
    proc_rel_dir = Path(cfg.dataset.out_dir).resolve().relative_to(_repo_root()) / f"gen_{generation:06d}" / f"proc_{task_index:05d}"
    logs_dir = Path(cfg.logging.mcts_iter_log).resolve().parent.relative_to(_repo_root()) / f"gen_{generation:06d}"
    task = SelfplayTask(
        session_id=session_id,
        task_id=f"{session_id}-selfplay-{generation:06d}-{task_index:05d}",
        generation=int(generation),
        task_type="selfplay",
        model_ref=model_ref,
        output_dataset_rel_dir=str(proc_rel_dir),
        iter_log_relpath=str(logs_dir / f"mcts_iter_remote_{task_index:05d}.csv"),
        root_log_relpath=str(logs_dir / f"mcts_root_remote_{task_index:05d}.csv"),
        game_id=int(game_id),
        num_roots=int(count),
        start_root_id=int(start_root_id),
        start_root_depth=int(cfg.run.root_depth),
        start_player=str(cfg.run.root_player),
        feature_version=int(cfg.run.feature_version),
        adv_iterations_per_root=int(cfg.adv_iterations_per_root),
        cont_iterations_per_root=int(cfg.cont_iterations_per_root),
        max_batch_size=int(cfg.max_batch_size),
        history_nontrivial_hops=int(_remote_selfplay_hop(cfg, generation, task_index)),
        history_seed=int(cfg.history_seed) + generation * 100_000 + task_index,
        sample_from_mcts_policy=bool(cfg.sample_from_mcts_policy),
        selfplay_policy_temperature=float(cfg.selfplay_policy_temperature),
        action_seed_base=int(cfg.action_seed_base) + generation * 1_000_000 + task_index * 1000,
        max_forced_hops_per_root=int(cfg.max_forced_hops_per_root),
        history_max_total_steps=int(cfg.history_max_total_steps),
        log_history_rows=bool(cfg.log_history_rows),
        task_seed=int(cfg.game_v2.reproducibility.global_seed) + generation * 100_000 + task_index,
    )
    backend.put_job_record(
        task_id=task.task_id,
        session_id=session_id,
        generation=int(generation),
        task_type="selfplay",
        status="queued",
        extra={"num_roots": int(count)},
    )
    backend.send_job(task.to_dict())
    return task


def _enqueue_arena_task(
    *,
    backend: AwsNetworkBackend,
    session_id: str,
    generation: int,
    task_index: int,
    entries: List[Dict[str, Any]],
    candidate_model_ref: RemoteModelRef,
    best_model_ref: RemoteModelRef,
    cfg: MultipleProcessTrainingConfig,
) -> ArenaTask:
    logs_dir = Path(cfg.logging.mcts_iter_log).resolve().parent.relative_to(_repo_root()) / f"gen_{generation:06d}" / "arena_games"
    task = ArenaTask(
        session_id=session_id,
        task_id=f"{session_id}-arena-{generation:06d}-{task_index:05d}",
        generation=int(generation),
        task_type="arena",
        candidate_model_ref=candidate_model_ref,
        best_model_ref=best_model_ref,
        arena_entries=[dict(x) for x in entries],
        arena_games_rel_dir=str(logs_dir),
        adv_iterations_per_root=int(cfg.evaluation.arena_iters_adversary),
        cont_iterations_per_root=int(cfg.evaluation.arena_iters_controller),
        arena_time_limit_sec=float(cfg.evaluation.arena_time_limit_sec),
        arena_max_controller_cleanup_steps=int(cfg.evaluation.arena_max_controller_cleanup_steps_safety),
        arena_max_total_turns=int(cfg.evaluation.arena_max_total_turns_safety),
        feature_version=int(cfg.evaluation.feature_version),
        tie_points=float(cfg.evaluation.tie_points),
        task_seed=int(cfg.game_v2.reproducibility.global_seed) + generation * 200_000 + task_index,
        action_seed_base=int(cfg.action_seed_base) + generation * 2_000_000 + task_index,
    )
    backend.put_job_record(
        task_id=task.task_id,
        session_id=session_id,
        generation=int(generation),
        task_type="arena",
        status="queued",
        extra={"num_games": len(entries)},
    )
    backend.send_job(task.to_dict())
    return task


def _receive_matching_results(
    backend: AwsNetworkBackend,
    *,
    session_id: str,
    pending: Dict[str, Dict[str, Any]],
    timeout_sec: int,
) -> RemoteResult:
    deadline = time.time() + float(timeout_sec)
    while True:
        remaining = max(1, int(deadline - time.time()))
        messages = backend.receive_results(max_number=10, wait_time_sec=min(20, remaining))
        if not messages:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for remote result; pending={sorted(pending.keys())[:8]}"
                )
            continue

        for message in messages:
            payload = json.loads(str(message["Body"]))
            result = RemoteResult.from_dict(payload)
            backend.delete_result_message(str(message["ReceiptHandle"]))
            if result.session_id != session_id:
                continue
            if result.task_id not in pending:
                continue
            return result


def _download_result_files(
    backend: AwsNetworkBackend,
    *,
    session_id: str,
    task_id: str,
    produced_files: List[str],
) -> None:
    backend.download_relative_files(
        _repo_root(),
        produced_files,
        backend.result_prefix(session_id, task_id),
    )


def _ingest_generation_dir(
    replay_buffer: BestModelReplayBuffer,
    generation: int,
    cfg: MultipleProcessTrainingConfig,
) -> int:
    gen_dataset_dir = Path(cfg.dataset.out_dir) / f"gen_{generation:06d}"
    return replay_buffer.add_generation_dir(gen_dataset_dir)


def run_parallel_self_improvement(cfg: MultipleProcessTrainingConfig) -> None:
    if not bool(cfg.network.enabled):
        raise RuntimeError("network.enabled must be True for the distributed Network pipeline")
    if str(cfg.network.node_role) != "head":
        raise RuntimeError("network.node_role must be 'head' for the distributed Network head pipeline")

    cfg.validate()
    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    dataset_base = Path(cfg.dataset.out_dir)
    dataset_base.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log_csv = Path(cfg.train_metrics_csv)

    eval_metrics_logger = EvaluationMetricsLogger(
        Path(cfg.evaluation_logging.metrics_csv),
        flush_every=int(cfg.logging.flush_every),
    )
    backend = AwsNetworkBackend(cfg)
    backend.wait_for_workers(
        min_count=1,
        timeout_sec=int(cfg.network.worker_registration_timeout_sec),
    )

    spec = make_dnn_spec(cfg=cfg.game_v2)
    model = AlphaZeroModel(spec=spec)
    trainer_h = cfg.game_v2.trainer
    trainer = Trainer(
        model=model,
        cfg=TrainerConfig(
            lr=float(trainer_h.lr),
            weight_decay=float(trainer_h.weight_decay),
            policy_weight=float(trainer_h.policy_weight),
            value_weight=float(trainer_h.value_weight),
            grad_clip_norm=float(trainer_h.grad_clip_norm),
            checkpoint_every=int(trainer_h.checkpoint_every),
            eval_every=int(trainer_h.eval_every),
            invalid_logit=float(trainer_h.invalid_logit),
        ),
        device=torch.device(cfg.model.device),
    )
    replay_buffer = BestModelReplayBuffer(
        capacity_samples=int(cfg.replay_capacity_samples),
        max_cached_shards=int(cfg.replay_max_cached_shards),
        seed=int(cfg.replay_seed),
    )

    best_ckpt_path = ckpt_dir / "best.pt"
    if not best_ckpt_path.exists():
        trainer.save_checkpoint(best_ckpt_path)
    _load_trainer_from_checkpoint(trainer, best_ckpt_path)

    gen_start = _next_generation_index(dataset_base)
    native_on = bool(str(cfg.environment_lang).strip().lower() == "native")
    session_id = f"gv2-{uuid.uuid4().hex[:10]}"

    try:
        for local_gen in range(int(cfg.num_generations)):
            gen = int(gen_start + local_gen)
            gen_dataset_dir = dataset_base / f"gen_{gen:06d}"
            gen_dataset_dir.mkdir(parents=True, exist_ok=True)

            _load_trainer_from_checkpoint(trainer, best_ckpt_path)
            selfplay_ckpt_path = ckpt_dir / f"selfplay_weights_gen_{gen:06d}.pt"
            shutil.copyfile(best_ckpt_path, selfplay_ckpt_path)

            ts_root = Path(cfg.native_torchscript_dir) / f"gen_{gen:06d}" / "network"
            best_model_ref = _upload_model_ref(
                backend,
                session_id=session_id,
                generation=gen,
                label="best_selfplay",
                checkpoint_path=selfplay_ckpt_path,
                model_version=int(gen * 100 + 1),
                spec=spec,
                native_on=native_on,
                torchscript_root=ts_root,
            )

            pending: Dict[str, Dict[str, Any]] = {}
            outstanding_selfplay_roots = 0
            next_selfplay_index = 0
            roots_submitted = 0
            roots_completed = 0
            added_from_generation = 0

            while roots_completed < int(cfg.roots_per_generation):
                while (
                    roots_submitted < int(cfg.roots_per_generation)
                    and len(pending) < int(cfg.network.max_total_workers)
                ):
                    count = min(
                        int(cfg.network.selfplay_roots_per_task),
                        int(cfg.roots_per_generation) - roots_submitted,
                    )
                    task = _enqueue_selfplay_task(
                        backend=backend,
                        session_id=session_id,
                        generation=gen,
                        task_index=next_selfplay_index,
                        count=count,
                        start_root_id=int(cfg.run.root_id) + roots_submitted,
                        game_id=int(cfg.run.game_id) + gen * 1_000_000 + next_selfplay_index,
                        model_ref=best_model_ref,
                        cfg=cfg,
                    )
                    pending[task.task_id] = {"kind": "selfplay", "num_roots": int(count)}
                    outstanding_selfplay_roots += int(count)
                    roots_submitted += int(count)
                    next_selfplay_index += 1

                result = _receive_matching_results(
                    backend,
                    session_id=session_id,
                    pending=pending,
                    timeout_sec=int(cfg.worker_result_timeout_sec),
                )
                meta = pending.pop(result.task_id)
                if not result.ok:
                    raise RuntimeError(f"Remote {result.task_type} task failed: {result.error}\n{result.traceback}")
                _download_result_files(
                    backend,
                    session_id=result.session_id,
                    task_id=result.task_id,
                    produced_files=result.produced_files,
                )
                for rel in result.produced_files:
                    if rel.endswith("manifest.jsonl"):
                        _rewrite_manifest_paths(_repo_root() / rel)
                added_from_generation += _ingest_generation_dir(replay_buffer, gen, cfg)
                roots_completed += int(meta["num_roots"])
                outstanding_selfplay_roots -= int(meta["num_roots"])

            if replay_buffer.total_samples <= 0:
                raise RuntimeError(
                    f"Replay buffer empty after generation={gen}, dir={gen_dataset_dir}"
                )

            train_steps_this_gen = _compute_train_steps_this_generation(cfg, replay_buffer.total_samples)
            for step_in_gen in range(int(train_steps_this_gen)):
                samples = replay_buffer.sample_batch(int(cfg.train_batch_size))
                batch_by_player = collate_mixed_samples(samples, device=trainer.device)
                metrics = trainer.train_step(batch_by_player)
                _append_train_log_row(
                    train_log_csv,
                    {
                        "time": time.time(),
                        "event": "train_step",
                        "generation": int(gen),
                        "train_step": int(metrics["step"]),
                        "train_step_in_generation": int(step_in_gen),
                        "loss": float(metrics["loss"]),
                        "policy_loss": float(metrics["policy_loss"]),
                        "value_loss": float(metrics["value_loss"]),
                        "controller_policy_loss": float(metrics["controller_policy_loss"]),
                        "controller_value_loss": float(metrics["controller_value_loss"]),
                        "controller_count": float(metrics["controller_count"]),
                        "adversary_policy_loss": float(metrics["adversary_policy_loss"]),
                        "adversary_value_loss": float(metrics["adversary_value_loss"]),
                        "adversary_count": float(metrics["adversary_count"]),
                        "replay_total_samples": int(replay_buffer.total_samples),
                        "added_from_generation": int(added_from_generation),
                        "dataset_dir": str(gen_dataset_dir),
                        "checkpoint_path": "",
                        "latest_path": "",
                    },
                )

            gen_ckpt_path = ckpt_dir / f"gen_{gen:06d}.pt"
            trainer.save_checkpoint(gen_ckpt_path)
            latest_path = ckpt_dir / "latest.pt"
            shutil.copyfile(gen_ckpt_path, latest_path)
            _append_train_log_row(
                train_log_csv,
                {
                    "time": time.time(),
                    "event": "checkpoint",
                    "generation": int(gen),
                    "train_step": int(trainer.step),
                    "train_step_in_generation": "",
                    "loss": "",
                    "policy_loss": "",
                    "value_loss": "",
                    "controller_policy_loss": "",
                    "controller_value_loss": "",
                    "controller_count": "",
                    "adversary_policy_loss": "",
                    "adversary_value_loss": "",
                    "adversary_count": "",
                    "replay_total_samples": int(replay_buffer.total_samples),
                    "added_from_generation": int(added_from_generation),
                    "dataset_dir": str(gen_dataset_dir),
                    "checkpoint_path": str(gen_ckpt_path),
                    "latest_path": str(latest_path),
                },
            )

            if bool(cfg.evaluation.enabled):
                gen_logs_dir = generation_log_dir_from_iter_log(cfg.logging.mcts_iter_log, gen)
                arena_games_dir = gen_logs_dir / "arena_games"
                arena_games_dir.mkdir(parents=True, exist_ok=True)

                candidate_model_ref = _upload_model_ref(
                    backend,
                    session_id=session_id,
                    generation=gen,
                    label="candidate_arena",
                    checkpoint_path=gen_ckpt_path,
                    model_version=int(gen * 100 + 2),
                    spec=spec,
                    native_on=native_on,
                    torchscript_root=ts_root,
                )
                best_arena_model_ref = _upload_model_ref(
                    backend,
                    session_id=session_id,
                    generation=gen,
                    label="best_arena",
                    checkpoint_path=best_ckpt_path,
                    model_version=int(gen * 100 + 3),
                    spec=spec,
                    native_on=native_on,
                    torchscript_root=ts_root,
                )

                arena_chunks = _chunk_entries(
                    _build_arena_game_entries(cfg=cfg, generation=gen),
                    int(cfg.network.arena_games_per_task),
                )
                arena_next_index = 0

                while arena_next_index < len(arena_chunks) or pending:
                    arena_pending = sum(1 for meta in pending.values() if meta["kind"] == "arena")
                    while (
                        arena_next_index < len(arena_chunks)
                        and arena_pending < int(cfg.network.max_arena_workers)
                        and len(pending) < int(cfg.network.max_total_workers)
                    ):
                        task = _enqueue_arena_task(
                            backend=backend,
                            session_id=session_id,
                            generation=gen,
                            task_index=arena_next_index,
                            entries=arena_chunks[arena_next_index],
                            candidate_model_ref=candidate_model_ref,
                            best_model_ref=best_arena_model_ref,
                            cfg=cfg,
                        )
                        pending[task.task_id] = {
                            "kind": "arena",
                            "num_games": len(task.arena_entries),
                        }
                        arena_pending += 1
                        arena_next_index += 1

                    if bool(cfg.network.background_selfplay_during_arena):
                        while (
                            len(pending) < int(cfg.network.max_total_workers)
                            and replay_buffer.total_samples + outstanding_selfplay_roots < int(cfg.replay_capacity_samples)
                        ):
                            count = int(cfg.network.selfplay_roots_per_task)
                            task = _enqueue_selfplay_task(
                                backend=backend,
                                session_id=session_id,
                                generation=gen,
                                task_index=next_selfplay_index,
                                count=count,
                                start_root_id=int(cfg.run.root_id) + roots_submitted,
                                game_id=int(cfg.run.game_id) + gen * 1_000_000 + next_selfplay_index,
                                model_ref=best_model_ref,
                                cfg=cfg,
                            )
                            pending[task.task_id] = {"kind": "selfplay", "num_roots": int(count)}
                            outstanding_selfplay_roots += int(count)
                            roots_submitted += int(count)
                            next_selfplay_index += 1

                    if not pending:
                        break

                    result = _receive_matching_results(
                        backend,
                        session_id=session_id,
                        pending=pending,
                        timeout_sec=int(cfg.worker_result_timeout_sec),
                    )
                    meta = pending.pop(result.task_id)
                    if not result.ok:
                        raise RuntimeError(f"Remote {result.task_type} task failed: {result.error}\n{result.traceback}")

                    _download_result_files(
                        backend,
                        session_id=result.session_id,
                        task_id=result.task_id,
                        produced_files=result.produced_files,
                    )
                    if meta["kind"] == "selfplay":
                        for rel in result.produced_files:
                            if rel.endswith("manifest.jsonl"):
                                _rewrite_manifest_paths(_repo_root() / rel)
                        _ingest_generation_dir(replay_buffer, gen, cfg)
                        outstanding_selfplay_roots -= int(meta["num_roots"])

                arena_results_csv = gen_logs_dir / "arena_results.csv"
                arena_metrics = grade_arena_from_game_logs(
                    game_log_dir=arena_games_dir,
                    out_csv=arena_results_csv,
                    tie_points=float(cfg.evaluation.tie_points),
                    win_threshold=float(cfg.evaluation.arena_win_threshold),
                )

                best_before = str(best_ckpt_path)
                promoted = bool(arena_metrics["passed"])
                if promoted:
                    shutil.copyfile(gen_ckpt_path, best_ckpt_path)

                eval_metrics_logger.log_generation(
                    generation=int(gen),
                    candidate_checkpoint=str(gen_ckpt_path),
                    best_checkpoint_before=str(best_before),
                    best_checkpoint_after=str(best_ckpt_path),
                    candidate_train_samples_used=int(train_steps_this_gen) * int(cfg.train_batch_size),
                    num_games=int(arena_metrics["num_eval_roots"]),
                    candidate_points=float(arena_metrics["candidate_points"]),
                    best_points=float(arena_metrics["best_points"]),
                    total_points=float(arena_metrics["total_points"]),
                    candidate_win_rate=float(arena_metrics["candidate_win_rate"]),
                    win_threshold=float(arena_metrics["arena_win_threshold"]),
                    passed=bool(arena_metrics["passed"]),
                    promoted_to_best=bool(promoted),
                    arena_results_csv=str(arena_results_csv),
                    arena_games_dir=str(arena_games_dir),
                )
    finally:
        try:
            eval_metrics_logger.close()
        except Exception:
            pass
