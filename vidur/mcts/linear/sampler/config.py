from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

from ..config import ConstraintSettings, SimulationSettings


@dataclass(frozen=True)
class SamplerCollectionSettings:
    target_unique_states: int = 400_000
    workers: int = 8
    start_depth: int = 0
    max_trace_length: int = 10
    max_branching: int = 10
    enum_max_samples: int = 10_000
    max_forced_hops: int = 20_000
    max_total_steps_per_state: int = 512
    max_snapshot_cache: int = 4_096
    shard_unique_states_per_worker: int = 10_000
    max_expansions_per_worker: int = 80_000
    max_rounds: int = 100
    history_hops: Sequence[int] = (0, 5, 10, 15, 20, 25, 30, 35)
    history_csv: str = ""
    root_player: str = "adversary"
    align_branching_roots: bool = True
    use_virtual_env: bool = True


@dataclass(frozen=True)
class SamplerOutputSettings:
    out_dir: str = "simulator_output/linear_sampler"
    parquet_compression: str = "zstd"


@dataclass(frozen=True)
class SamplerRunConfig:
    sim: SimulationSettings = field(default_factory=SimulationSettings)
    constraints: ConstraintSettings = field(default_factory=ConstraintSettings)
    collection: SamplerCollectionSettings = field(default_factory=SamplerCollectionSettings)
    output: SamplerOutputSettings = field(default_factory=SamplerOutputSettings)
    seed: int = 12345


def output_dir(cfg: SamplerRunConfig) -> Path:
    return Path(cfg.output.out_dir)


def round_dir(cfg: SamplerRunConfig, round_idx: int) -> Path:
    return output_dir(cfg) / f"round_{int(round_idx):03d}"


def parse_history_hops(raw: str) -> Tuple[int, ...]:
    out = []
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        return (0, 5, 10, 15, 20, 25, 30, 35)
    return tuple(out)
