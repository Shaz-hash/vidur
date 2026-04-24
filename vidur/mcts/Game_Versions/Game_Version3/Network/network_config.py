from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def gv3_output_dir() -> Path:
    return repo_root() / "simulator_output" / "Game_Version3"


def _auto_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_device(value: str) -> str:
    requested = str(value).strip().lower()
    if requested == "auto":
        return _auto_torch_device()
    return str(value)


@dataclass(frozen=True)
class NetworkMachineConfig:
    name: str
    ssh_host: str
    public_ip: str = ""
    repo_dir: str = "/home/ubuntu/vidur"
    python: str = "/home/ubuntu/vidur/.venv/bin/python"


@dataclass(frozen=True)
class NetworkPathConfig:
    output_dir: Path = field(default_factory=lambda: gv3_output_dir() / "network")
    machines_json: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "server" / "machines.json"
    )
    default_weights_path: Path = field(
        default_factory=lambda: gv3_output_dir() / "mcts_dnn_checkpoints" / "best.pt"
    )

    @property
    def server_tasks_dir(self) -> Path:
        return self.output_dir / "server_tasks"

    @property
    def received_dir(self) -> Path:
        return self.output_dir / "received"

    @property
    def allocation_log_csv(self) -> Path:
        return self.output_dir / "server_allocations.csv"

    @property
    def received_log_csv(self) -> Path:
        return self.output_dir / "server_received.csv"

    @property
    def cleanup_log_csv(self) -> Path:
        return self.output_dir / "cleanup_log.csv"

    @property
    def process_log_path(self) -> Path:
        return gv3_output_dir() / "mcts_dnn_logs" / "alphaZeroParrallel.out"

    @property
    def dataset_dir(self) -> Path:
        return gv3_output_dir() / "mcts_dnn_dataset"

    @property
    def logs_dir(self) -> Path:
        return gv3_output_dir() / "mcts_dnn_logs"

    @property
    def eval_metrics_csv(self) -> Path:
        return self.logs_dir / "eval_metrics.csv"

    @property
    def checkpoints_dir(self) -> Path:
        return gv3_output_dir() / "mcts_dnn_checkpoints"


@dataclass(frozen=True)
class NetworkTaskDefaults:
    generation: int = 0
    model_version: int = 0
    total_roots_per_generation: int = 0
    roots_per_cycle: int = 0
    sample_cycles_per_generation: int = 1
    num_roots_per_machine: int = 500
    history_hops_min: int = 0
    history_hops_max: int = 25
    history_seed: int = 2026
    game_id_base: int = 50_000_000
    start_root_id: int = 0
    start_root_depth: int = 0
    start_player: str = "adversary"
    feature_version: int = 1
    adv_iterations_per_root: int = 4000
    cont_iterations_per_root: int = 4000
    max_batch_size: int = 256
    history_root_batch_size: int = 64
    eval_split_ratio: float = 0.1
    eval_split_seed: int = 0
    action_seed_base: int = 4
    task_seed_base: int = 0
    sample_from_mcts_policy: bool = True
    selfplay_policy_temperature: float = 2.0
    max_forced_hops_per_root: int = 1024
    history_max_total_steps: int = 20000
    log_history_rows: bool = True
    allow_duplicate_history_fallback: bool = True
    shard_size: int = 512
    worker_model_device: str = "cpu"
    local_training_device: str = "auto"
    local_evaluation_device: str = "cpu"
    use_virtual_env: bool = True
    worker_cpu_fraction: float = 0.70
    worker_processes: int = 0
    max_concurrent_workers: int = 0
    max_workers_per_interval: int = 1
    selfplay_dynamic_chunk_roots: int = 128
    selfplay_zero_progress_interval_patience: int = 4
    selfplay_launch_rss_limit_gb: float = 120.0
    selfplay_launch_poll_sec: float = 2.0
    worker_result_timeout_sec: int = 7200


@dataclass(frozen=True)
class NetworkCleanupConfig:
    cleanup_after_generation_done: bool = True
    remove_local_received: bool = True
    remove_remote_results: bool = True
    remove_remote_tasks: bool = True
    fail_on_remote_cleanup_error: bool = False


@dataclass(frozen=True)
class NetworkConfig:
    paths: NetworkPathConfig = field(default_factory=NetworkPathConfig)
    task: NetworkTaskDefaults = field(default_factory=NetworkTaskDefaults)
    cleanup: NetworkCleanupConfig = field(default_factory=NetworkCleanupConfig)


DEFAULT_NETWORK_CONFIG = NetworkConfig()


def with_local_training_device(
    cfg: MultipleProcessTrainingConfig = DEFAULT_MULTIPROCESS_TRAINING_CONFIG,
    *,
    device: str = DEFAULT_NETWORK_CONFIG.task.local_training_device,
) -> MultipleProcessTrainingConfig:
    """Return a GV3 training config with the model/trainer moved to the requested device."""
    return replace(cfg, model=replace(cfg.model, device=resolve_device(device)))


def selected_machines(
    machines: Sequence[NetworkMachineConfig],
    names: Sequence[str] | None,
) -> list[NetworkMachineConfig]:
    if not names:
        return list(machines)
    wanted = {str(x) for x in names}
    out = [m for m in machines if m.name in wanted or m.ssh_host in wanted or m.public_ip in wanted]
    missing = wanted - {m.name for m in out} - {m.ssh_host for m in out} - {m.public_ip for m in out}
    if missing:
        raise ValueError(f"Unknown network machine(s): {sorted(missing)}")
    return out
