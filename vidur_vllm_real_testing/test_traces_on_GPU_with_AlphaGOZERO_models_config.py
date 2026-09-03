"""Single source of truth for Task 1.1.2 real-GPU trace evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Iterable


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_REMOTE_ROOT = "/home/shaz/vidur_vllm_task_1_1_2"
DEFAULT_MODEL_BUNDLE = (
    f"{DEFAULT_REMOTE_ROOT}/artifacts/promoted_models/"
    "AlphaGOZERO_new_vidur_flash_infer_models"
)
DEFAULT_PREFILL_PROFILE = (
    REPO_ROOT
    / "vidur/AlphaGoZero/new_vidur_cache_experiment/flash-infer_prefill_profile.csv"
)
GV3_IN_DISTRIBUTION_PREFILL_TOKENS = (
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
)


class TraceType(str, Enum):
    IN_DISTRIBUTION = "in_distribution"
    OUT_DISTRIBUTION = "out_distribution"

    @classmethod
    def parse(cls, value: str | "TraceType") -> "TraceType":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"trace_type must be one of {choices}") from exc


@dataclass(frozen=True)
class AlphaGoZeroGPUTraceConfig:
    """Reproducible hardware, trace, model, and search contract."""

    task_name: str = "task_1_1_2"

    # Served model and exact Task 1.1.1 kernel/runtime contract.
    model_type: str = "meta-llama/Meta-Llama-3-8B"
    vllm_model_name: str = "NousResearch/Meta-Llama-3-8B"
    controller_model_family: str = "dnn"
    vllm_version: str = "0.26.0"
    flashinfer_version: str = "0.6.14"
    torch_version: str = "2.11.0"
    cuda_compiler_version: str = "13.0.88"
    cuda_home: str = "/usr/local/cuda-13.0"
    attention_backend: str = "FLASHINFER"
    dtype: str = "float16"

    # This task intentionally supports only one physical GPU.
    gpu_index: int = 2
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.50
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 512
    block_size: int = 16
    async_scheduling: bool = False
    frontend_multiprocessing: bool = False

    remote_root: str = DEFAULT_REMOTE_ROOT
    remote_environment: str = "/home/shaz/vidur_vllm_profile_accuracy/.venv"
    remote_hf_home: str = "/home/shaz/vidur_vllm_profile_accuracy/hf_cache"
    remote_source_root: str = f"{DEFAULT_REMOTE_ROOT}/source/vidur-classical-search"
    vllm_model_path: str = "NousResearch/Meta-Llama-3-8B"
    model_bundle_path: str = DEFAULT_MODEL_BUNDLE
    controller_value_model_path: str = f"{DEFAULT_MODEL_BUNDLE}/controller_value"
    controller_policy_model_path: str = f"{DEFAULT_MODEL_BUNDLE}/controller_prior"
    adversary_policy_model_path: str = f"{DEFAULT_MODEL_BUNDLE}/adversary_prior"
    native_mcts_config_path: str = f"{DEFAULT_MODEL_BUNDLE}/native_mcts_cfg.json"
    prefill_profile_path: str = str(DEFAULT_PREFILL_PROFILE)

    static_trace_length_s: float = 20.0
    trace_type: TraceType = TraceType.IN_DISTRIBUTION
    trace_seed: int = 2026
    arrival_group_interval_s: float = 1.2
    decode_tokens_per_request: int = 864
    prefill_grid_tokens: int = 128
    min_prefill_tokens: int = 128
    max_prefill_tokens: int = 4096
    adversary_tick_s: float = 0.2
    launch_window_s: float = 1.0
    launch_window_request_cap: int = 7
    launch_window_prefill_cap: int = 7168

    # Task 1.1.1 AlphaGoZero/MCTS values.
    mcts_iterations: int = 2000
    search_mode: str = "full_tree_rollout"
    rollout_count: int = 1
    rollout_horizon_s: float = 3.0
    rollout_threads: int = 1
    rollout_policy_threads: int = 1
    native_threads: int = 1
    discount_factor: float = 0.98
    puct_c: float = 0.5
    uct_c: float = 1.0
    rollout_policy_temperature: float = 1.0
    policy_prior_temperature: float = 1.0
    game_horizon_s: float = 5.0

    # Retained for provenance. Evaluation itself is deterministic and has no
    # root noise; these are the matching Task 1.1.1 self-play settings.
    dirichlet_epsilon: float = 0.25
    dirichlet_total_concentration: float = 12.0
    sampled_controller_actions: int = 15
    root_noise_enabled_for_gpu_evaluation: bool = False

    timing_scope: str = "transformer_blocks"
    batch_duration_source: str = "gpu_forward"
    implicit_prefill_output_tokens: int = 1
    scheduler_mode: str = "controller"
    output_root: str = f"{DEFAULT_REMOTE_ROOT}/runs"

    def validate(self) -> None:
        trace_type = TraceType.parse(self.trace_type)
        if self.controller_model_family != "dnn":
            raise ValueError("Task 1.1.2 accepts only promoted DNN model bundles")
        if self.attention_backend != "FLASHINFER":
            raise ValueError("Task 1.1.2 must use the Task 1.1.1 FlashInfer backend")
        if self.gpu_index not in {2, 3}:
            raise ValueError("mew1 GPU index must be 2 or 3")
        if self.tensor_parallel_size != 1 or self.pipeline_parallel_size != 1:
            raise ValueError("Task 1.1.2 currently requires TP=1 and PP=1")
        if self.async_scheduling or self.frontend_multiprocessing:
            raise ValueError("async scheduling and frontend multiprocessing must stay disabled")
        if not 0.0 < self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if self.static_trace_length_s <= 0.0:
            raise ValueError("static_trace_length_s must be positive")
        if self.arrival_group_interval_s <= self.launch_window_s:
            raise ValueError("arrival groups must be separated by more than the GV3 launch window")
        if self.prefill_grid_tokens != 128:
            raise ValueError("the promoted model contract requires a 128-token grid")
        if (self.min_prefill_tokens, self.max_prefill_tokens) != (128, 4096):
            raise ValueError("the promoted model contract requires prefill bounds [128, 4096]")
        if not 1 <= self.decode_tokens_per_request <= 864:
            raise ValueError("decode_tokens_per_request must be in [1, 864]")
        if self.mcts_iterations <= 0 or self.rollout_count <= 0:
            raise ValueError("MCTS iterations and rollout count must be positive")
        if self.search_mode != "full_tree_rollout":
            raise ValueError("Task 1.1.2 requires full_tree_rollout")
        if self.rollout_horizon_s <= 0.0 or self.rollout_threads <= 0:
            raise ValueError("rollout horizon and thread counts must be positive")
        if not 0.0 < self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        if self.puct_c < 0.0 or self.uct_c < 0.0:
            raise ValueError("PUCT/UCT constants must be nonnegative")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.dirichlet_total_concentration <= 0.0:
            raise ValueError("dirichlet concentration must be positive")
        if self.root_noise_enabled_for_gpu_evaluation:
            raise ValueError("real-GPU evaluation must not inject self-play root noise")
        if self.timing_scope != "transformer_blocks":
            raise ValueError("canonical time must use Task 1.1.1 transformer-block timing")
        if self.batch_duration_source != "gpu_forward":
            raise ValueError("canonical time must use measured GPU batch duration")
        if trace_type is TraceType.IN_DISTRIBUTION and not GV3_IN_DISTRIBUTION_PREFILL_TOKENS:
            raise ValueError("in-distribution prefill support is empty")

    def validate_model_artifacts(self) -> None:
        bundle = Path(self.model_bundle_path).expanduser()
        expected = {
            "bundle manifest": bundle / "current_model.json",
            "controller value": Path(self.controller_value_model_path).expanduser() / "native_model.tsv",
            "controller policy": Path(self.controller_policy_model_path).expanduser() / "native_model.tsv",
            "adversary policy": Path(self.adversary_policy_model_path).expanduser() / "native_model.tsv",
            "native MCTS config": Path(self.native_mcts_config_path).expanduser(),
        }
        missing = [f"{label}: {path}" for label, path in expected.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing promoted DNN artifacts:\n" + "\n".join(missing))

    def to_environment(self, *, canonical_trace: Path, scheduler_log: Path) -> dict[str, str]:
        "converts these params into vidur suited enviornment params"
        
        self.validate()
        return {
            "VIDUR_PHYSICAL_GPU": str(self.gpu_index),
            "VIDUR_REAL_MODEL": self.vllm_model_path,
            "VIDUR_REAL_HF_HOME": self.remote_hf_home,
            "VIDUR_REAL_CUDA_HOME": self.cuda_home,
            "VIDUR_REAL_TRACE": str(canonical_trace),
            "VIDUR_REAL_MODEL_BUNDLE": self.model_bundle_path,
            "VIDUR_REAL_NATIVE_CFG": self.native_mcts_config_path,
            "VIDUR_VLLM_GV3_MODEL_BUNDLE": self.model_bundle_path,
            "VIDUR_VLLM_GV3_NATIVE_CFG": self.native_mcts_config_path,
            "VIDUR_VLLM_CANONICAL_TRACE": str(canonical_trace),
            "VIDUR_VLLM_SCHEDULER_LOG": str(scheduler_log),
            "VIDUR_VLLM_GV3_MCTS_ITERATIONS": str(self.mcts_iterations),
            "VIDUR_VLLM_GV3_DISCOUNT_FACTOR": str(self.discount_factor),
            "VIDUR_VLLM_GV3_PUCT_C": str(self.puct_c),
            "VIDUR_VLLM_GV3_UCT_C": str(self.uct_c),
            "VIDUR_VLLM_GV3_NATIVE_SEARCH_MODE": self.search_mode,
            "VIDUR_VLLM_GV3_ROLLOUT_COUNT": str(self.rollout_count),
            "VIDUR_VLLM_GV3_ROLLOUT_HORIZON_S": str(self.rollout_horizon_s),
            "VIDUR_VLLM_GV3_ROLLOUT_THREADS": str(self.rollout_threads),
            "VIDUR_VLLM_GV3_ROLLOUT_POLICY_THREADS": str(self.rollout_policy_threads),
            "VIDUR_VLLM_GV3_NATIVE_THREADS": str(self.native_threads),
            "VIDUR_VLLM_GV3_ADV_TICK_S": str(self.adversary_tick_s),
            "VIDUR_VLLM_GV3_BATCH_DURATION_SOURCE": self.batch_duration_source,
            "VIDUR_VLLM_GPU_TIMING_SCOPE": self.timing_scope,
            "VIDUR_VLLM_GV3_IMPLICIT_PREFILL_OUTPUT_TOKENS": str(
                self.implicit_prefill_output_tokens
            ),
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["trace_type"] = TraceType.parse(self.trace_type).value
        payload["in_distribution_prefill_tokens"] = list(
            GV3_IN_DISTRIBUTION_PREFILL_TOKENS
        )
        payload["out_distribution_canonicalization"] = "ceiling to next 128-token grid point"
        payload["controller_model_paths"] = {
            "value": self.controller_value_model_path,
            "policy": self.controller_policy_model_path,
        }
        payload["adversary_model_paths"] = {"policy": self.adversary_policy_model_path}
        payload["calibration"] = None
        return payload


def _env(name: str, default: object) -> str:
    return os.environ.get(f"VIDUR_TASK112_{name}", str(default))


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, int(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"VIDUR_TASK112_{name} must be boolean")


def load_config() -> AlphaGoZeroGPUTraceConfig:
    bundle = _env("MODEL_BUNDLE", DEFAULT_MODEL_BUNDLE)
    config = AlphaGoZeroGPUTraceConfig(
        gpu_index=int(_env("GPU", 2)),
        cuda_home=_env("CUDA_HOME", "/usr/local/cuda-13.0"),
        static_trace_length_s=float(_env("TRACE_LENGTH_S", 20.0)),
        trace_type=TraceType.parse(_env("TRACE_TYPE", TraceType.IN_DISTRIBUTION.value)),
        trace_seed=int(_env("TRACE_SEED", 2026)),
        vllm_model_path=_env("VLLM_MODEL", "NousResearch/Meta-Llama-3-8B"),
        model_bundle_path=bundle,
        controller_value_model_path=_env(
            "CONTROLLER_VALUE_MODEL", str(Path(bundle) / "controller_value")
        ),
        controller_policy_model_path=_env(
            "CONTROLLER_POLICY_MODEL", str(Path(bundle) / "controller_prior")
        ),
        adversary_policy_model_path=_env(
            "ADVERSARY_POLICY_MODEL", str(Path(bundle) / "adversary_prior")
        ),
        native_mcts_config_path=_env(
            "NATIVE_MCTS_CONFIG", str(Path(bundle) / "native_mcts_cfg.json")
        ),
        prefill_profile_path=_env("PREFILL_PROFILE", DEFAULT_PREFILL_PROFILE),
        remote_root=_env("REMOTE_ROOT", DEFAULT_REMOTE_ROOT),
        remote_environment=_env(
            "REMOTE_ENVIRONMENT", "/home/shaz/vidur_vllm_profile_accuracy/.venv"
        ),
        remote_hf_home=_env(
            "REMOTE_HF_HOME", "/home/shaz/vidur_vllm_profile_accuracy/hf_cache"
        ),
        remote_source_root=_env(
            "REMOTE_SOURCE_ROOT", f"{DEFAULT_REMOTE_ROOT}/source/vidur-classical-search"
        ),
        output_root=_env("OUTPUT_ROOT", f"{DEFAULT_REMOTE_ROOT}/runs"),
        decode_tokens_per_request=int(_env("DECODE_TOKENS", 864)),
        mcts_iterations=int(_env("MCTS_ITERATIONS", 2000)),
        rollout_count=int(_env("ROLLOUT_COUNT", 1)),
        rollout_horizon_s=float(_env("ROLLOUT_HORIZON_S", 3.0)),
        rollout_threads=int(_env("ROLLOUT_THREADS", 1)),
        rollout_policy_threads=int(_env("ROLLOUT_POLICY_THREADS", 1)),
        native_threads=int(_env("NATIVE_THREADS", 1)),
        discount_factor=float(_env("DISCOUNT_FACTOR", 0.98)),
        puct_c=float(_env("PUCT_C", 0.5)),
        uct_c=float(_env("UCT_C", 1.0)),
        root_noise_enabled_for_gpu_evaluation=_env_bool("ROOT_NOISE", False),
    )
    config.validate()
    return config


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print Task 1.1.2 configuration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = json.dumps(load_config().to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
