from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.events.request_arrival_event import RequestArrivalEvent
from vidur.logger import init_logger
from vidur.simulator import Simulator
from vidur.utils.memory_planner import MemoryPlanner


logger = init_logger(__name__)


@dataclass
class PrefillProfile:
    step: int
    max_tokens: int
    entries: Dict[int, float]

    @classmethod
    def load_or_generate(
        cls,
        sim_config: SimulationConfig,
        step: int,
        slowdown: float,
        path: Optional[str],
        max_tokens: Optional[int] = None,
    ) -> "PrefillProfile":
        output_path = Path(path) if path else None
        if output_path and output_path.exists():
            profile = cls.load(output_path)
        else:
            profile = cls.generate(
                sim_config,
                step,
                output_path,
                max_tokens_override=max_tokens,
            )

        if slowdown != 1.0:
            scaled = {k: v * slowdown for k, v in profile.entries.items()}
            return PrefillProfile(step=profile.step, max_tokens=profile.max_tokens, entries=scaled)
        return profile

    @classmethod
    def load(cls, path: Path) -> "PrefillProfile":
        entries: Dict[int, float] = {}
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries[int(row["prefill_tokens"])] = float(row["prefill_time_seconds"])
        keys = sorted(entries.keys())
        step = keys[0] if keys else 1
        max_tokens = keys[-1] if keys else step
        return PrefillProfile(step=step, max_tokens=max_tokens, entries=entries)

    @classmethod
    def generate(
        cls,
        sim_config: SimulationConfig,
        step: int,
        output_path: Optional[Path] = None,
        max_tokens_override: Optional[int] = None,
    ) -> "PrefillProfile":
        if max_tokens_override is not None:
            max_tokens = max(max_tokens_override, step)
        else:
            max_tokens = _derive_max_prefill_tokens(sim_config)
        max_tokens = max(step, (max_tokens // step) * step)

        # Base grid: [step, 2*step, ..., max_tokens]
        tokens = list(range(step, max_tokens + 1, step))
        entries: Dict[int, float] = {}
        for size in tokens:
            entries[size] = _measure_prefill_time(sim_config, size)

        # Extra large sizes we always want to profile, regardless of step
        extra_sizes = [20000, 30000, 40000, 50000,
                       60000, 70000, 80000, 90000 ,100000]
        for size in extra_sizes:
            if size not in entries:
                entries[size] = _measure_prefill_time(sim_config, size)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["prefill_tokens", "prefill_time_seconds"])
                # Write base grid first
                for size in tokens:
                    writer.writerow([size, entries[size]])
                # Then append the extra big sizes at the end (sorted)
                for size in sorted(extra_sizes):
                    if size not in tokens and size in entries:
                        writer.writerow([size, entries[size]])

        return PrefillProfile(step=step, max_tokens=max_tokens, entries=entries)

    def lookup(self, tokens: int) -> float:
        if not self.entries:
            return 0.0
        closest = min(self.entries.keys(), key=lambda k: abs(k - tokens))
        return self.entries[closest]


def _measure_prefill_time(sim_config: SimulationConfig, prefill_tokens: int) -> float:
    cfg = copy.deepcopy(sim_config)
    scheduler_cfg = getattr(cfg, "replica_scheduler_config", None)
    if scheduler_cfg is not None and hasattr(scheduler_cfg, "chunk_size"):
        scheduler_cfg.chunk_size = max(prefill_tokens, getattr(scheduler_cfg, "chunk_size", 0) or prefill_tokens)
    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]
    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False

    simulator = Simulator(cfg, register_atexit=False)
    simulator._event_queue.clear()

    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=0,
        block_hash_ids=None,
        block_size=None,
    )
    simulator._add_event(RequestArrivalEvent(0.0, request))
    simulator.run()

    logger.debug(
        "Prefill probe tokens=%s scheduled_at=%s prefill_completed_at=%s completed_at=%s processed=%s total=%s",
        prefill_tokens,
        getattr(request, "scheduled_at", None),
        getattr(request, "prefill_completed_at", None),
        getattr(request, "completed_at", None),
        request.num_processed_tokens,
        request.total_tokens,
    )

    if request.prefill_completed_at is not None and request.scheduled_at is not None:
        if request.prefill_completed_at < request.scheduled_at:
            logger.warning(
                "Request %s prefill completion earlier than scheduled. scheduled=%s completed=%s",
                request.id,
                request.scheduled_at,
                request.prefill_completed_at,
            )
        return request.prefill_completed_at - request.scheduled_at
    if request.completed_at is not None and request.scheduled_at is not None:
        return request.completed_at - request.scheduled_at
    return 0.0


def _derive_max_prefill_tokens(sim_config: SimulationConfig) -> int:
    cache_cfg = sim_config.cluster_config.cache_config
    block_size = max(cache_cfg.block_size, 1)
    num_blocks = cache_cfg.num_blocks
    if not num_blocks:
        planner = MemoryPlanner(sim_config.cluster_config.replica_config, cache_cfg)
        capacity_tokens = planner.get_max_kv_cache_size_in_tokens()
        num_blocks = max(int(capacity_tokens // block_size), 1)
    max_tokens = block_size * max(num_blocks or 1, 1)
    return max(max_tokens, block_size)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate prefill timing profile")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional maximum prefill tokens to profile (overrides auto-derived value).",
    )
    args, unknown = parser.parse_known_args(argv)

    original_argv = sys.argv
    try:
        cli_argv = [original_argv[0]] + unknown if argv is None else ["prefill_calibrator"] + list(unknown)
        sys.argv = cli_argv
        cfg = SimulationConfig.create_from_cli_args()

    finally:
        sys.argv = original_argv

    et_cfg = cfg.execution_time_predictor_config
    et_cfg.prediction_max_tokens_per_request = 100000
    et_cfg.prediction_max_batch_size = 64

    step = args.step or cfg.cluster_config.cache_config.block_size
    step = max(1, step)
    max_tokens_override = args.max_tokens
    if max_tokens_override is not None and max_tokens_override < step:
        max_tokens_override = step
    PrefillProfile.generate(
        cfg,
        step=step,
        output_path=Path(args.output),
        max_tokens_override=max_tokens_override,
    )
    print(f"Generated prefill profile -> {args.output}")


if __name__ == "__main__":
    main()
