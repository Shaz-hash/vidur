# (بِسْمِ ٱللَّهِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Dict

from ..config import MultipleProcessTrainingConfig
from .aws_backend import AwsNetworkBackend
from .executor import build_worker_runtime, execute_arena_task, execute_selfplay_task
from .types import ArenaTask, RemoteResult, SelfplayTask, task_from_dict


def _stable_worker_id(cfg: MultipleProcessTrainingConfig) -> str:
    configured = str(cfg.network.worker_name).strip()
    if configured:
        return configured
    return f"{socket.gethostname()}-{os.getpid()}"


def _artifact_cache_path(cfg: MultipleProcessTrainingConfig, s3_key: str) -> Path:
    digest = hashlib.sha1(str(s3_key).encode("utf-8")).hexdigest()
    basename = Path(str(s3_key)).name or "artifact.bin"
    return Path(cfg.network.local_staging_dir) / "artifacts" / digest[:12] / basename


class _TaskHeartbeat:
    def __init__(
        self,
        *,
        backend: AwsNetworkBackend,
        worker_id: str,
        task_id: str,
        receipt_handle: str,
        heartbeat_sec: int,
        visibility_timeout_sec: int,
    ) -> None:
        self.backend = backend
        self.worker_id = worker_id
        self.task_id = task_id
        self.receipt_handle = receipt_handle
        self.heartbeat_sec = int(heartbeat_sec)
        self.visibility_timeout_sec = int(visibility_timeout_sec)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self.heartbeat_sec):
            try:
                self.backend.heartbeat_worker(
                    worker_id=self.worker_id,
                    status="busy",
                    current_task_id=self.task_id,
                )
                self.backend.change_job_visibility(
                    self.receipt_handle,
                    self.visibility_timeout_sec,
                )
            except Exception:
                # Best-effort heartbeat only; job execution still owns the final outcome.
                pass


def _download_model_artifacts(
    backend: AwsNetworkBackend,
    cfg: MultipleProcessTrainingConfig,
    task: SelfplayTask | ArenaTask,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if isinstance(task, SelfplayTask):
        out["weights_path"] = backend.download_file(
            task.model_ref.weights_s3_key,
            _artifact_cache_path(cfg, task.model_ref.weights_s3_key),
        )
        if task.model_ref.controller_ts_s3_key and task.model_ref.adversary_ts_s3_key:
            out["controller_ts_path"] = backend.download_file(
                task.model_ref.controller_ts_s3_key,
                _artifact_cache_path(cfg, task.model_ref.controller_ts_s3_key),
            )
            out["adversary_ts_path"] = backend.download_file(
                task.model_ref.adversary_ts_s3_key,
                _artifact_cache_path(cfg, task.model_ref.adversary_ts_s3_key),
            )
        return out

    out["candidate_weights_path"] = backend.download_file(
        task.candidate_model_ref.weights_s3_key,
        _artifact_cache_path(cfg, task.candidate_model_ref.weights_s3_key),
    )
    out["best_weights_path"] = backend.download_file(
        task.best_model_ref.weights_s3_key,
        _artifact_cache_path(cfg, task.best_model_ref.weights_s3_key),
    )
    if task.candidate_model_ref.controller_ts_s3_key and task.candidate_model_ref.adversary_ts_s3_key:
        out["candidate_controller_ts_path"] = backend.download_file(
            task.candidate_model_ref.controller_ts_s3_key,
            _artifact_cache_path(cfg, task.candidate_model_ref.controller_ts_s3_key),
        )
        out["candidate_adversary_ts_path"] = backend.download_file(
            task.candidate_model_ref.adversary_ts_s3_key,
            _artifact_cache_path(cfg, task.candidate_model_ref.adversary_ts_s3_key),
        )
    if task.best_model_ref.controller_ts_s3_key and task.best_model_ref.adversary_ts_s3_key:
        out["best_controller_ts_path"] = backend.download_file(
            task.best_model_ref.controller_ts_s3_key,
            _artifact_cache_path(cfg, task.best_model_ref.controller_ts_s3_key),
        )
        out["best_adversary_ts_path"] = backend.download_file(
            task.best_model_ref.adversary_ts_s3_key,
            _artifact_cache_path(cfg, task.best_model_ref.adversary_ts_s3_key),
        )
    return out


def run_worker_agent(cfg: MultipleProcessTrainingConfig) -> None:
    if not bool(cfg.network.enabled):
        raise RuntimeError("network.enabled must be True when running a network worker")
    if str(cfg.network.node_role) != "worker":
        raise RuntimeError("network.node_role must be 'worker' when running a network worker")

    backend = AwsNetworkBackend(cfg)
    worker_id = _stable_worker_id(cfg)
    runtime = build_worker_runtime(cfg)
    staging_root = Path(cfg.network.local_staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)

    while True:
        backend.heartbeat_worker(worker_id=worker_id, status="idle", current_task_id="")
        messages = backend.receive_jobs(
            max_number=1,
            visibility_timeout=int(cfg.network.job_visibility_timeout_sec),
        )
        if not messages:
            continue

        message = messages[0]
        payload = json.loads(str(message["Body"]))
        task = task_from_dict(payload)

        backend.put_job_record(
            task_id=task.task_id,
            session_id=task.session_id,
            generation=int(task.generation),
            task_type=str(task.task_type),
            status="running",
            worker_id=worker_id,
        )
        backend.heartbeat_worker(worker_id=worker_id, status="busy", current_task_id=task.task_id)

        task_workspace = staging_root / "tasks" / task.task_id
        if task_workspace.exists():
            shutil.rmtree(task_workspace)
        task_workspace.mkdir(parents=True, exist_ok=True)

        heartbeat = _TaskHeartbeat(
            backend=backend,
            worker_id=worker_id,
            task_id=task.task_id,
            receipt_handle=str(message["ReceiptHandle"]),
            heartbeat_sec=int(cfg.network.worker_heartbeat_sec),
            visibility_timeout_sec=int(cfg.network.job_visibility_timeout_sec),
        )
        heartbeat.start()
        try:
            artifacts = _download_model_artifacts(backend, cfg, task)
            if isinstance(task, SelfplayTask):
                result = execute_selfplay_task(runtime, task, task_workspace, artifacts)
            elif isinstance(task, ArenaTask):
                result = execute_arena_task(runtime, task, task_workspace, artifacts)
            else:  # pragma: no cover
                raise TypeError(f"Unsupported task type: {type(task)}")
        finally:
            heartbeat.stop()

        if result.ok:
            uploaded = backend.upload_relative_files(
                task_workspace,
                result.produced_files,
                backend.result_prefix(task.session_id, task.task_id),
            )
            result = RemoteResult(
                session_id=result.session_id,
                task_id=result.task_id,
                generation=int(result.generation),
                task_type=result.task_type,
                ok=True,
                produced_files=uploaded,
                num_roots=int(result.num_roots),
                num_games=int(result.num_games),
            )
            backend.put_job_record(
                task_id=task.task_id,
                session_id=task.session_id,
                generation=int(task.generation),
                task_type=str(task.task_type),
                status="succeeded",
                worker_id=worker_id,
                extra={
                    "produced_files": uploaded,
                    "num_roots": int(result.num_roots),
                    "num_games": int(result.num_games),
                },
            )
        else:
            backend.put_job_record(
                task_id=task.task_id,
                session_id=task.session_id,
                generation=int(task.generation),
                task_type=str(task.task_type),
                status="failed",
                worker_id=worker_id,
                extra={
                    "error": str(result.error),
                    "traceback": str(result.traceback),
                },
            )

        backend.send_result(result.to_dict())
        backend.delete_job_message(str(message["ReceiptHandle"]))
        backend.heartbeat_worker(worker_id=worker_id, status="idle", current_task_id="")

        try:
            shutil.rmtree(task_workspace)
        except Exception:
            pass
