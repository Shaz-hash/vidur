from __future__ import annotations

import random
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...DNN.dnn_spec import make_dnn_spec
from ...DNN.replay_write import ReplayWriter, ReplayWriterConfig
from ...DNN.selfPlay import SelfPlayRunner
from ...DNN.value_models import AlphaZeroModel
from ...config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig
from ...mctsDNN import VidurMCTS
from ...multiProcessUtils import (
    _build_env_and_simulator,
    _load_weights_into_model,
    _set_global_seeds,
    _set_runtime_cpu_thread_env,
)
from ..network_config import resolve_device
from .files import list_relative_files, utc_now_iso, write_json
from .types import NetworkSelfplayTask, NetworkTaskResult


def _task_cfg(task: NetworkSelfplayTask) -> MultipleProcessTrainingConfig:
    base = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
    return replace(
        base,
        model=replace(base.model, device=resolve_device(task.model_device)),
        dataset=replace(base.dataset, shard_size=int(task.shard_size)),
        use_virtual_env=bool(task.use_virtual_env),
    )


def execute_selfplay_task(task: NetworkSelfplayTask) -> NetworkTaskResult:
    received_at = utc_now_iso()
    started_at = received_at
    writer_train = None
    writer_eval = None
    mcts = None

    try:
        cfg = _task_cfg(task)
        _set_runtime_cpu_thread_env(1)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        _set_global_seeds(
            int(task.task_seed),
            torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
        )

        _, env, _, explore_cfg = _build_env_and_simulator(
            cfg,
            use_virtual_env=bool(task.use_virtual_env),
        )

        spec = make_dnn_spec(cfg=cfg.game_v2)
        device = torch.device(resolve_device(task.model_device))
        model = AlphaZeroModel(spec=spec).to(device)
        model.eval()
        _load_weights_into_model(model, Path(task.weights_path))

        out_dir_train = Path(task.out_dir_train)
        out_dir_eval = Path(task.out_dir_eval)
        logs_dir = Path(task.logs_dir)
        out_dir_train.mkdir(parents=True, exist_ok=True)
        out_dir_eval.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        writer_train = ReplayWriter(
            ReplayWriterConfig(out_dir=out_dir_train, shard_size=int(task.shard_size))
        )
        writer_eval = ReplayWriter(
            ReplayWriterConfig(out_dir=out_dir_eval, shard_size=int(task.shard_size))
        )

        iter_log = logs_dir / f"mcts_iter_{task.task_id}.csv"
        root_log = logs_dir / f"mcts_root_{task.task_id}.csv"

        mcts = VidurMCTS(
            env=env,
            explore_cfg=explore_cfg,
            rng=random.Random(int(task.action_seed_base)),
            log_path=iter_log,
            tree_log_path=root_log,
            logger_flush_every=int(cfg.logging.flush_every),
            verbose=False,
            complete_log=False,
        )

        runner = SelfPlayRunner(
            env=env,
            mcts=mcts,
            model=model,
            writer=writer_train,
            eval_writer=writer_eval,
            device_for_features=device,
            game_v2_cfg=cfg.game_v2,
        )

        runner.run_n_roots(
            game_id=int(task.game_id),
            num_roots=int(task.num_roots),
            adv_iterations_per_root=int(task.adv_iterations_per_root),
            cont_iterations_per_root=int(task.cont_iterations_per_root),
            max_batch_size=int(task.max_batch_size),
            start_root_id=int(task.start_root_id),
            start_root_depth=int(task.start_root_depth),
            start_player=str(task.start_player),
            feature_version=int(task.feature_version),
            sample_from_mcts_policy=bool(task.sample_from_mcts_policy),
            selfplay_policy_temperature=float(task.selfplay_policy_temperature),
            action_seed_base=int(task.action_seed_base),
            history_nontrivial_hops=int(task.history_nontrivial_hops),
            history_hops_min=int(task.history_hops_min),
            history_hops_max=int(task.history_hops_max),
            history_seed=int(task.history_seed),
            max_forced_hops_per_root=int(task.max_forced_hops_per_root),
            history_max_total_steps=int(task.history_max_total_steps),
            history_root_batch_size=int(task.history_root_batch_size),
            log_history_rows=bool(task.log_history_rows),
            progress_prefix=f"[GV3 network task={task.task_id}]",
            model_version=int(task.model_version),
            eval_split_ratio=float(task.eval_split_ratio),
            eval_split_seed=int(task.eval_split_seed),
            history_seen_signatures=None,
            shared_history_signatures=None,
            shared_history_lock=None,
            allow_duplicate_history_fallback=bool(task.allow_duplicate_history_fallback),
        )

        if writer_train is not None:
            writer_train.close()
            writer_train = None
        if writer_eval is not None:
            writer_eval.close()
            writer_eval = None
        if mcts is not None:
            mcts.close()
            mcts = None

        finished_at = utc_now_iso()
        result_dir = Path(task.result_dir)
        produced_files = list_relative_files(result_dir)
        result = NetworkTaskResult(
            ok=True,
            session_id=task.session_id,
            task_id=task.task_id,
            machine_name=task.machine_name,
            machine_ip=task.machine_ip,
            generation=int(task.generation),
            model_version=int(task.model_version),
            received_at_utc=received_at,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            sent_back_at_utc=finished_at,
            result_dir=str(result_dir),
            train_dir=str(out_dir_train),
            eval_dir=str(out_dir_eval),
            logs_dir=str(logs_dir),
            run_stats=dict(getattr(runner, "last_run_stats", {}) or {}),
            produced_files=produced_files,
        )
        write_json(result_dir / "result.json", result.to_dict())
        return result

    except Exception as exc:
        finished_at = utc_now_iso()
        result = NetworkTaskResult(
            ok=False,
            session_id=task.session_id,
            task_id=task.task_id,
            machine_name=task.machine_name,
            machine_ip=task.machine_ip,
            generation=int(task.generation),
            model_version=int(task.model_version),
            received_at_utc=received_at,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            sent_back_at_utc=finished_at,
            result_dir=str(task.result_dir),
            train_dir=str(task.out_dir_train),
            eval_dir=str(task.out_dir_eval),
            logs_dir=str(task.logs_dir),
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_json(Path(task.result_dir) / "result.json", result.to_dict())
        return result

    finally:
        if writer_train is not None:
            try:
                writer_train.close()
            except Exception:
                pass
        if writer_eval is not None:
            try:
                writer_eval.close()
            except Exception:
                pass
        if mcts is not None:
            try:
                mcts.close()
            except Exception:
                pass

