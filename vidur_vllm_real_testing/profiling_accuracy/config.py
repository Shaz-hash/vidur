from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProfilingConfig:
    """Single source of truth for the reproducible profiling experiment."""

    vllm_version: str = "0.26.0"
    flashinfer_version: str = "0.6.14"
    torch_version: str = "2.11.0"
    cuda_compiler_version: str = "13.0.88"
    sarathi_revision: str = "c94dbf438711a4670dbf7477175062e87748999d"
    vidur_model_name: str = "meta-llama/Meta-Llama-3-8B"
    # This ungated mirror has the same Llama-3-8B tensor geometry. Kernel timing
    # does not depend on the learned weight values.
    vllm_model_name: str = "NousResearch/Meta-Llama-3-8B"
    device_name: str = "a100_mew1_gpu2_vllm026_flashinfer0614"
    simulator_device: str = "a100"
    network_device: str = "a100_dgx"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "float16"
    attention_backend: str = "FLASHINFER"
    max_tokens_per_request: int = 8192
    max_batch_size: int = 256
    max_prefill_chunk_size: int = 4096
    vllm_max_num_batched_tokens: int = 8192
    vllm_max_num_seqs: int = 512
    vllm_gpu_memory_utilization: float = 0.50
    vllm_async_scheduling: bool = False
    vllm_enable_chunked_prefill: bool = True
    block_size: int = 16
    gpu_candidates: tuple[int, ...] = (2, 3)
    request_counts: tuple[int, ...] = (1, 2)
    operation_warmups: int = 2
    operation_repetitions: int = 20
    vllm_warmups: int = 3
    vllm_repetitions: int = 10
    random_seed: int = 42
    predictor_training_threads: int = 1
    remote_root: str = "/home/shaz/vidur_vllm_profile_accuracy"
    prefill_profile_path: str = str(REPO_ROOT / "simulator_output/prefill_profile.csv")
    output_root: str = str(
        REPO_ROOT / "simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING"
    )

    @property
    def raw_profile_dir(self) -> Path:
        return (
            REPO_ROOT
            / "data/profiling/compute"
            / self.device_name
            / self.vidur_model_name
        )

    @property
    def predictor_cache_dir(self) -> Path:
        return Path(self.output_root) / "vidur_predictor_cache"

    def prefill_sizes(self) -> tuple[int, ...]:
        path = Path(self.prefill_profile_path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            values = tuple(int(row["prefill_tokens"]) for row in rows)
        if not values:
            raise ValueError(f"no prefill sizes found in {path}")
        if values != tuple(sorted(set(values))):
            raise ValueError(f"prefill sizes must be sorted and unique in {path}")
        if values[-1] > self.max_prefill_chunk_size:
            raise ValueError(
                f"prefill size {values[-1]} exceeds configured maximum "
                f"{self.max_prefill_chunk_size}"
            )
        return values

    def cases(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (request_count, tokens)
            for request_count in self.request_counts
            for tokens in self.prefill_sizes()
        )

    def validate(self) -> None:
        if self.vllm_max_num_batched_tokens < (
            max(self.request_counts) * self.max_prefill_chunk_size
        ):
            raise ValueError("vLLM token budget cannot contain the largest test batch")
        if self.vllm_max_num_seqs < max(self.request_counts):
            raise ValueError("vLLM sequence cap cannot contain the largest test batch")
        if not 0 < self.vllm_gpu_memory_utilization < 1:
            raise ValueError("vLLM GPU memory utilization must be between 0 and 1")
        if self.predictor_training_threads != 1:
            raise ValueError("task requires one predictor training thread")
        if not self.gpu_candidates or any(gpu < 0 for gpu in self.gpu_candidates):
            raise ValueError("at least one non-negative GPU candidate is required")
        if self.attention_backend != "FLASHINFER":
            raise ValueError("this experiment must use the FlashInfer attention backend")
        if self.simulator_device != "a100":
            raise ValueError("the mew1 experiment requires Vidur's A100 device model")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["prefill_sizes"] = list(self.prefill_sizes())
        result["cases"] = [
            {"request_count": count, "prefill_tokens_per_request": tokens}
            for count, tokens in self.cases()
        ]
        result["raw_profile_dir"] = str(self.raw_profile_dir)
        result["predictor_cache_dir"] = str(self.predictor_cache_dir)
        result["calibration"] = None
        return result


def _integer_tuple_from_environment(
    name: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return values


def load_config() -> ProfilingConfig:
    config = ProfilingConfig(
        vllm_model_name=os.environ.get(
            "VIDUR_PROFILE_VLLM_MODEL", ProfilingConfig.vllm_model_name
        ),
        request_counts=_integer_tuple_from_environment(
            "VIDUR_PROFILE_REQUEST_COUNTS", ProfilingConfig.request_counts
        ),
        vllm_max_num_batched_tokens=int(
            os.environ.get(
                "VIDUR_PROFILE_MAX_BATCHED_TOKENS",
                ProfilingConfig.vllm_max_num_batched_tokens,
            )
        ),
        remote_root=os.environ.get(
            "VIDUR_PROFILE_REMOTE_ROOT", ProfilingConfig.remote_root
        ),
    )
    config.validate()
    return config


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the profiling configuration")
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
