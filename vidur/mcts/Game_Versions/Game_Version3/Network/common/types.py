from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class NetworkSelfplayTask:
    session_id: str
    task_id: str
    machine_name: str
    machine_ip: str
    generation: int
    cycle_index: int
    sample_cycles_per_generation: int
    model_version: int
    weights_path: str
    result_dir: str
    out_dir_train: str
    out_dir_eval: str
    logs_dir: str
    game_id: int
    num_roots: int
    start_root_id: int
    start_root_depth: int
    start_player: str
    feature_version: int
    adv_iterations_per_root: int
    cont_iterations_per_root: int
    max_batch_size: int
    history_nontrivial_hops: int
    history_hops_min: int
    history_hops_max: int
    history_hop_interval_width: int
    history_seed: int
    sample_from_mcts_policy: bool
    selfplay_policy_temperature: float
    action_seed_base: int
    max_forced_hops_per_root: int
    history_max_total_steps: int
    history_root_batch_size: int
    log_history_rows: bool
    eval_split_ratio: float
    eval_split_seed: int
    task_seed: int
    shard_size: int
    worker_cpu_fraction: float = 0.70
    worker_processes: int = 0
    max_concurrent_workers: int = 0
    max_workers_per_interval: int = 1
    selfplay_dynamic_chunk_roots: int = 128
    selfplay_zero_progress_interval_patience: int = 4
    selfplay_launch_rss_limit_gb: float = 120.0
    selfplay_launch_poll_sec: float = 2.0
    worker_result_timeout_sec: int = 7200
    model_device: str = "auto"
    use_virtual_env: bool = True
    environment_lang: str = "python"
    allow_duplicate_history_fallback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Keep Python-path tasks backward-compatible with workers that have not
        # pulled the native-switch field yet. Native tasks still require updated workers.
        if str(data.get("environment_lang", "python")) == "python":
            data.pop("environment_lang", None)
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NetworkSelfplayTask":
        data = dict(payload)
        data.setdefault("environment_lang", "python")
        return cls(**data)


@dataclass(frozen=True)
class NetworkTaskResult:
    ok: bool
    session_id: str
    task_id: str
    machine_name: str
    machine_ip: str
    generation: int
    cycle_index: int
    model_version: int
    received_at_utc: str
    started_at_utc: str
    finished_at_utc: str
    sent_back_at_utc: str
    result_dir: str
    train_dir: str
    eval_dir: str
    logs_dir: str
    run_stats: Dict[str, Any] = field(default_factory=dict)
    produced_files: list[str] = field(default_factory=list)
    error: str = ""
    traceback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NetworkTaskResult":
        return cls(**payload)

    @property
    def result_path(self) -> Path:
        return Path(self.result_dir) / "result.json"
