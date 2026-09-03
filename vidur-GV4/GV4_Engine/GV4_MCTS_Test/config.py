"""Resolved TP2/PP2 configuration used by the GV4 MCTS trace tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import (
    GV4EngineConfig,
    KVCacheConfig,
    ModelConfig,
    SchedulerConfig,
    TopologyConfig,
    VidurPredictorConfig,
)


CLASSICAL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    CLASSICAL_ROOT / "simulator_output" / "GV4_MCTS_Test" / "h100_tp2_pp2_uniform"
)
DEFAULT_CACHE_DIR = CLASSICAL_ROOT / "cache" / "gv4_h100_tp2_pp2"


@dataclass(frozen=True, slots=True)
class GV4MCTSTestConfig:
    """All runtime choices for one reproducible trace-validation run."""

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    predictor_cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    predictor_cache_mode: str = "use_cache"
    timing_mode: str = "vidur"
    mcts_iterations: int = 100
    game_id: int = 0
    root_id: int = 0
    root_node_id: int = 0
    seed: int = 7
    num_replicas: int = 1
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 2
    max_inflight_microbatches: int = 2
    kv_budget_bytes_per_rank: int = 512 * 1024 * 1024
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        object.__setattr__(
            self,
            "predictor_cache_dir",
            Path(self.predictor_cache_dir).expanduser().resolve(),
        )
        if self.predictor_cache_mode not in {
            "ignore_cache",
            "use_cache",
            "require_cache",
        }:
            raise ValueError("invalid predictor_cache_mode")
        if self.timing_mode not in {"vidur", "deterministic"}:
            raise ValueError("timing_mode must be vidur or deterministic")
        for name in (
            "mcts_iterations",
            "num_replicas",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "max_inflight_microbatches",
            "kv_budget_bytes_per_rank",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_replicas != 1:
            raise ValueError(
                "GV4 Python MCTS requires one replica until routing exists"
            )
        if self.tensor_parallel_size != 2 or self.pipeline_parallel_size != 2:
            raise ValueError("this integration contract intentionally tests TP2/PP2")

    @property
    def raw_log_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def iteration_trace_dir(self) -> Path:
        return self.output_dir / "mcts_iterations"


def build_engine_config(test: GV4MCTSTestConfig) -> GV4EngineConfig:
    """Build the single authoritative engine manifest for capture and checks."""

    model = ModelConfig(
        model_id="meta-llama/Meta-Llama-3-8B",
        model_revision="vidur-h100-profile-2026-08-30",
    )
    topology = TopologyConfig.contiguous(
        num_replicas=test.num_replicas,
        tensor_parallel_size=test.tensor_parallel_size,
        pipeline_parallel_size=test.pipeline_parallel_size,
        num_layers=model.num_layers,
    )
    predictor = VidurPredictorConfig(
        device="h100",
        network_device="h100_dgx",
        cache_dir=str(test.predictor_cache_dir),
        cache_mode=test.predictor_cache_mode,
        prediction_max_tokens_per_request=8192,
        prediction_max_batch_size=256,
        prediction_max_prefill_chunk_size=4096,
        kv_cache_prediction_granularity=64,
        prefill_chunk_size_prediction_granularity=32,
        num_training_job_threads=1,
    )
    engine = GV4EngineConfig(
        model=model,
        topology=topology,
        vidur_predictor=predictor,
        kv_cache=KVCacheConfig(
            kv_budget_bytes_per_rank=(test.kv_budget_bytes_per_rank,)
            * topology.total_ranks,
            block_size_tokens=16,
            memory_safety_margin_fraction=0.10,
        ),
        scheduler=SchedulerConfig(
            max_batch_tokens=4608,
            max_sequences=256,
            max_prefill_chunk_tokens=4096,
            max_inflight_microbatches=test.max_inflight_microbatches,
            inter_stage_queue_capacity=test.max_inflight_microbatches,
        ),
        global_seed=test.seed,
        enable_debug_asserts=True,
        fail_fast_on_invalid_state=True,
    )
    engine.validate()
    return engine


def build_default_engine_config() -> GV4EngineConfig:
    """Zero-argument TP2/PP2 factory suitable for portable config manifests."""

    return build_engine_config(GV4MCTSTestConfig())
