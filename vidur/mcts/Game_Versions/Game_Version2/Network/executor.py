# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحْمَٰنِ)

from __future__ import annotations

import copy
import random
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.eval_utils import NoopReplayWriter
from ..DNN.export_torchscript_gv2 import export_torchscript_artifacts_gv2
from ..DNN.models import AlphaZeroModel
from ..DNN.replay_write import ReplayWriter, ReplayWriterConfig
from ..DNN.selfPlay import SelfPlayRunner
from ..config import ModelGroup, MultipleProcessTrainingConfig
from ..logger.evaluation_pipeline_logger import ArenaGameCycleFileLogger
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import (
    _attach_native_runtime,
    _build_env_and_simulator,
    _load_weights_into_model,
    _set_global_seeds,
)
from .types import ArenaTask, RemoteResult, SelfplayTask


def _list_relative_files(root: Path, rel_paths: Iterable[Path]) -> List[str]:
    out: List[str] = []
    for rel_path in rel_paths:
        full = root / rel_path
        if full.is_file():
            out.append(str(rel_path).replace("\\", "/"))
        elif full.is_dir():
            for p in sorted(full.rglob("*")):
                if p.is_file():
                    out.append(str(p.relative_to(root)).replace("\\", "/"))
    return sorted(set(out))


@dataclass
class WorkerRuntime:
    cfg: MultipleProcessTrainingConfig
    env: Any
    explore_cfg: Any
    device: torch.device
    model: AlphaZeroModel
    arena_candidate_model: AlphaZeroModel
    arena_best_model: AlphaZeroModel
    native_runtime: Any | None


def build_worker_runtime(cfg: MultipleProcessTrainingConfig) -> WorkerRuntime:
    worker_cfg = replace(
        cfg,
        model=replace(cfg.model, device=str(cfg.network.worker_model_device)),
    )
    _set_global_seeds(
        int(worker_cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(worker_cfg.game_v2.reproducibility.torch_deterministic),
    )
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    _, env, _, explore_cfg = _build_env_and_simulator(
        worker_cfg,
        use_virtual_env=bool(worker_cfg.use_virtual_env),
    )
    spec = make_dnn_spec(cfg=worker_cfg.game_v2)
    device = torch.device(worker_cfg.model.device)

    model = AlphaZeroModel(spec=spec).to(device)
    model.eval()
    arena_candidate_model = AlphaZeroModel(spec=spec).to(device)
    arena_candidate_model.eval()
    arena_best_model = AlphaZeroModel(spec=spec).to(device)
    arena_best_model.eval()

    native_runtime = None
    if bool(getattr(explore_cfg, "native_mcts_enabled", False)):
        from ... import mcts_native_gv2 as _mcts_native_gv2

        native_runtime = _mcts_native_gv2.NativeTorchScriptInferRuntimeGV2(
            str(worker_cfg.model.device),
            -50.0,
            100.0,
        )

    return WorkerRuntime(
        cfg=worker_cfg,
        env=env,
        explore_cfg=explore_cfg,
        device=device,
        model=model,
        arena_candidate_model=arena_candidate_model,
        arena_best_model=arena_best_model,
        native_runtime=native_runtime,
    )


def execute_selfplay_task(
    runtime: WorkerRuntime,
    task: SelfplayTask,
    workspace_root: Path,
    artifacts: Dict[str, Path],
) -> RemoteResult:
    writer = None
    mcts = None
    try:
        _load_weights_into_model(runtime.model, Path(artifacts["weights_path"]))
        if runtime.native_runtime is not None and artifacts.get("controller_ts_path") and artifacts.get("adversary_ts_path"):
            spec = f"{artifacts['controller_ts_path']}||{artifacts['adversary_ts_path']}"
            runtime.native_runtime.load_models({int(task.model_ref.model_version): spec})
            _attach_native_runtime(runtime.model, runtime.native_runtime, int(task.model_ref.model_version))

        _set_global_seeds(
            int(task.task_seed),
            torch_deterministic=bool(runtime.cfg.game_v2.reproducibility.torch_deterministic),
        )

        out_dir = Path(workspace_root) / task.output_dataset_rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        iter_log = Path(workspace_root) / task.iter_log_relpath
        root_log = Path(workspace_root) / task.root_log_relpath
        iter_log.parent.mkdir(parents=True, exist_ok=True)
        root_log.parent.mkdir(parents=True, exist_ok=True)

        writer = ReplayWriter(
            ReplayWriterConfig(
                out_dir=out_dir,
                shard_size=int(runtime.cfg.dataset.shard_size),
            )
        )

        mcts = VidurMCTS(
            env=runtime.env,
            explore_cfg=runtime.explore_cfg,
            rng=random.Random(int(task.action_seed_base)),
            log_path=iter_log,
            tree_log_path=root_log,
            logger_flush_every=int(runtime.cfg.logging.flush_every),
            verbose=False,
            complete_log=False,
        )

        runner = SelfPlayRunner(
            env=runtime.env,
            mcts=mcts,
            model=runtime.model,
            writer=writer,
            device_for_features=runtime.device,
            game_v2_cfg=runtime.cfg.game_v2,
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
            history_seed=int(task.history_seed),
            max_forced_hops_per_root=int(task.max_forced_hops_per_root),
            history_max_total_steps=int(task.history_max_total_steps),
            log_history_rows=bool(task.log_history_rows),
        )

        writer.close()
        writer = None
        mcts.close()
        mcts = None

        produced_files = _list_relative_files(
            Path(workspace_root),
            [
                Path(task.output_dataset_rel_dir),
                Path(task.iter_log_relpath),
                Path(task.root_log_relpath),
            ],
        )
        return RemoteResult(
            session_id=task.session_id,
            task_id=task.task_id,
            generation=int(task.generation),
            task_type="selfplay",
            ok=True,
            produced_files=produced_files,
            num_roots=int(task.num_roots),
        )
    except Exception as exc:
        return RemoteResult(
            session_id=task.session_id,
            task_id=task.task_id,
            generation=int(task.generation),
            task_type="selfplay",
            ok=False,
            produced_files=[],
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if mcts is not None:
            try:
                mcts.close()
            except Exception:
                pass


def execute_arena_task(
    runtime: WorkerRuntime,
    task: ArenaTask,
    workspace_root: Path,
    artifacts: Dict[str, Path],
) -> RemoteResult:
    arena_mcts = None
    try:
        _load_weights_into_model(runtime.arena_candidate_model, Path(artifacts["candidate_weights_path"]))
        _load_weights_into_model(runtime.arena_best_model, Path(artifacts["best_weights_path"]))

        if runtime.native_runtime is not None:
            candidate_ctrl = artifacts.get("candidate_controller_ts_path")
            candidate_adv = artifacts.get("candidate_adversary_ts_path")
            best_ctrl = artifacts.get("best_controller_ts_path")
            best_adv = artifacts.get("best_adversary_ts_path")
            model_specs: Dict[int, str] = {}
            if candidate_ctrl and candidate_adv:
                model_specs[int(task.candidate_model_ref.model_version)] = f"{candidate_ctrl}||{candidate_adv}"
            if best_ctrl and best_adv:
                model_specs[int(task.best_model_ref.model_version)] = f"{best_ctrl}||{best_adv}"
            if model_specs:
                runtime.native_runtime.load_models(model_specs)
                _attach_native_runtime(
                    runtime.arena_candidate_model,
                    runtime.native_runtime,
                    int(task.candidate_model_ref.model_version),
                )
                _attach_native_runtime(
                    runtime.arena_best_model,
                    runtime.native_runtime,
                    int(task.best_model_ref.model_version),
                )

        _set_global_seeds(
            int(task.task_seed),
            torch_deterministic=bool(runtime.cfg.game_v2.reproducibility.torch_deterministic),
        )
        arena_explore_cfg = copy.deepcopy(runtime.explore_cfg)
        setattr(arena_explore_cfg, "root_dirichlet_noise_enabled", False)
        setattr(arena_explore_cfg, "root_dirichlet_alpha", 0.0)
        setattr(arena_explore_cfg, "root_dirichlet_epsilon", 0.0)

        arena_mcts = VidurMCTS(
            env=runtime.env,
            explore_cfg=arena_explore_cfg,
            rng=random.Random(int(task.action_seed_base)),
            log_path=None,
            tree_log_path=None,
            logger_flush_every=int(runtime.cfg.logging.flush_every),
            verbose=False,
            complete_log=False,
        )

        runner = SelfPlayRunner(
            env=runtime.env,
            mcts=arena_mcts,
            model=runtime.arena_candidate_model,
            writer=NoopReplayWriter(),
            device_for_features=runtime.device,
            game_v2_cfg=runtime.cfg.game_v2,
        )

        arena_games_dir = Path(workspace_root) / task.arena_games_rel_dir
        arena_games_dir.mkdir(parents=True, exist_ok=True)
        cycle_logger = ArenaGameCycleFileLogger(arena_games_dir)

        for entry in task.arena_entries:
            out = runner.run_arena_game(
                game_id=int(entry["game_id"]),
                candidate_model=runtime.arena_candidate_model,
                best_model=runtime.arena_best_model,
                history_nontrivial_hops=int(entry["history_hops"]),
                adv_iterations_per_root=int(task.adv_iterations_per_root),
                cont_iterations_per_root=int(task.cont_iterations_per_root),
                arena_time_limit_sec=float(task.arena_time_limit_sec),
                arena_max_controller_cleanup_steps=int(task.arena_max_controller_cleanup_steps),
                arena_max_total_turns=int(task.arena_max_total_turns),
                feature_version=int(task.feature_version),
                tie_points=float(task.tie_points),
                start_player=str(entry.get("player_to_act", "adversary")),
                start_root_depth=int(entry.get("depth", 0)),
                history_root_id_for_logs=int(entry.get("root_id", 0)),
                cycle_file_logger=cycle_logger,
            )
            cycle_a = out["cycle_a"]
            cycle_b = out["cycle_b"]
            cycle_logger.write_cycle_end(
                game_id=int(entry["game_id"]),
                cycle_label="candidate_as_adversary",
                total_cost=float(cycle_a["total_cost"]),
                slo_violations=int(cycle_a["slo_violations"]),
                total_lateness=float(cycle_a["total_lateness"]),
                end_reason=str(cycle_a.get("end_reason", "")),
            )
            cycle_logger.write_cycle_end(
                game_id=int(entry["game_id"]),
                cycle_label="best_as_adversary",
                total_cost=float(cycle_b["total_cost"]),
                slo_violations=int(cycle_b["slo_violations"]),
                total_lateness=float(cycle_b["total_lateness"]),
                end_reason=str(cycle_b.get("end_reason", "")),
            )

        produced_files = _list_relative_files(Path(workspace_root), [Path(task.arena_games_rel_dir)])
        return RemoteResult(
            session_id=task.session_id,
            task_id=task.task_id,
            generation=int(task.generation),
            task_type="arena",
            ok=True,
            produced_files=produced_files,
            num_games=len(task.arena_entries),
        )
    except Exception as exc:
        return RemoteResult(
            session_id=task.session_id,
            task_id=task.task_id,
            generation=int(task.generation),
            task_type="arena",
            ok=False,
            produced_files=[],
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        if arena_mcts is not None:
            try:
                arena_mcts.close()
            except Exception:
                pass
