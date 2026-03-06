from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple


def _default_sim_cli_args() -> Tuple[str, ...]:
    return (
        "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device", "h100",
        "--replica_config_network_device", "h100_dgx",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--global_scheduler_config_type", "round_robin",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap", "512",
        "--no-snapshot_rng_state",
    )


@dataclass(frozen=True)
class SimulationSettings:
    cli_args: Sequence[str] = field(default_factory=_default_sim_cli_args)


@dataclass(frozen=True)
class ConstraintSettings:
    maximum_qps: int = 5
    interval_request_size: int = 512
    min_request_tokens: int = 512
    max_request_tokens: int = 3072
    prefill_profile_path: str = "simulator_output/prefill_profile.csv"
    prefill_slowdown: float = 3.0
    prefill_slos: Sequence[float] = (3.0,)
    decode_slos: Sequence[float] = (50.0,)


@dataclass(frozen=True)
class CollectionSettings:
    workers: int = 8
    train_samples_per_worker: int = 10000
    eval_samples_per_worker: int = 1000
    max_branching: int = 10
    enum_max_samples: int = 10000
    max_forced_hops: int = 20000
    max_total_steps_per_state: int = 512
    history_hops: Sequence[int] = (0, 5, 10, 15, 20, 25, 30, 35)
    history_csv: str = ""
    root_player: str = "adversary"
    align_branching_roots: bool = True
    use_virtual_env: bool = True


@dataclass(frozen=True)
class BellmanSettings:
    discount_factor: float = 0.98
    # The same token-step concept used in current MCTS discount calibration.
    base_step_tokens: int = 512


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # adam | sgd
    weight_decay: float = 0.0


@dataclass(frozen=True)
class RolloutSettings:
    num_traces: int = 8
    max_steps: int = 256


@dataclass(frozen=True)
class OutputSettings:
    out_dir: str = "simulator_output/linear_value"


@dataclass(frozen=True)
class LinearPipelineConfig:
    sim: SimulationSettings = field(default_factory=SimulationSettings)
    constraints: ConstraintSettings = field(default_factory=ConstraintSettings)
    collection: CollectionSettings = field(default_factory=CollectionSettings)
    bellman: BellmanSettings = field(default_factory=BellmanSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    rollout: RolloutSettings = field(default_factory=RolloutSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    rounds: int = 1
    seed: int = 12345
    device: str = "cpu"


@dataclass(frozen=True)
class WorkerTask:
    worker_id: int
    round_idx: int
    seed: int
    history_hop: int
    out_dir: str
    model_ckpt_path: str
    cfg: LinearPipelineConfig


@dataclass(frozen=True)
class WorkerResult:
    worker_id: int
    history_hop: int
    train_npz: str
    eval_npz: str
    train_meta_csv: str
    eval_meta_csv: str
    train_count: int
    eval_count: int
    ok: bool
    error: str = ""


def round_dir(cfg: LinearPipelineConfig, round_idx: int) -> Path:
    return Path(cfg.output.out_dir) / f"round_{int(round_idx):03d}"
