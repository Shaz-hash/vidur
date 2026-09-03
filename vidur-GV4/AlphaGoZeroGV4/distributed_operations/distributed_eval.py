"""Distribute paired GV4 arena games and merge exact role-isolated results."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shlex
import threading
import time
from typing import Any, Mapping, Sequence

from ..config import ExperimentConfig, load_experiment_config
from ..model_bundle import LoadedModelBundle, load_model_bundle
from ..training_and_evaluation.arena import (
    ArenaGameResult,
    ArenaResult,
    evaluate_candidate,
    summarize_role_results,
)
from ..training_and_evaluation.evaluation_pipeline_logger import (
    EvaluationPipelineLogger,
)
from .cluster import (
    ClusterSpec,
    HostSpec,
    copy_from_host,
    copy_to_host,
    load_cluster,
    run_command,
    run_shell,
)
from .durable_transfer import sync_model_bundle


DISTRIBUTED_ARENA_SCHEMA_VERSION = "gv4_distributed_arena_v1"

__all__ = [
    "DISTRIBUTED_ARENA_SCHEMA_VERSION",
    "EvaluationChunk",
    "evaluate_candidate_distributed",
    "merge_evaluation_chunks",
    "plan_evaluation_chunks",
    "run_evaluation_chunk",
]


@dataclass(frozen=True, slots=True)
class EvaluationChunk:
    chunk_id: int
    pair_start: int
    pair_count: int
    host: HostSpec


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation result {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("evaluation result must be a JSON object")
    return value


def plan_evaluation_chunks(
    games: int,
    hosts: Sequence[HostSpec],
    *,
    games_per_chunk: int,
) -> tuple[EvaluationChunk, ...]:
    """Assign contiguous, non-overlapping pair ranges in stable round-robin order."""

    if games <= 0 or games_per_chunk <= 0:
        raise ValueError("games and games_per_chunk must be positive")
    if not hosts:
        raise ValueError("distributed evaluation requires at least one host")
    chunks: list[EvaluationChunk] = []
    pair_start = 0
    while pair_start < games:
        pair_count = min(games_per_chunk, games - pair_start)
        chunks.append(
            EvaluationChunk(
                chunk_id=len(chunks),
                pair_start=pair_start,
                pair_count=pair_count,
                host=hosts[len(chunks) % len(hosts)],
            )
        )
        pair_start += pair_count
    return tuple(chunks)


def run_evaluation_chunk(
    experiment: ExperimentConfig,
    incumbent: LoadedModelBundle,
    candidate: LoadedModelBundle,
    *,
    pair_start: int,
    pair_count: int,
    output_dir: str | Path,
) -> Path:
    """Run one arena slice and rewrite local pair IDs into global pair IDs."""

    if pair_start < 0 or pair_count <= 0:
        raise ValueError("invalid distributed arena pair range")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    arena = replace(
        experiment.arena,
        games=pair_count,
        seed=experiment.arena.seed + pair_start,
    )
    local = evaluate_candidate(
        experiment.resolve_engine_config(),
        incumbent,
        candidate,
        arena=arena,
        logger=EvaluationPipelineLogger(output / "evaluation"),
    )
    games = tuple(
        replace(game, pair_index=game.pair_index + pair_start) for game in local.games
    )
    result = ArenaResult(
        incumbent_bundle_version=local.incumbent_bundle_version,
        candidate_bundle_version=local.candidate_bundle_version,
        games=games,
        controller=local.controller,
        adversary=local.adversary,
    )
    destination = output / "chunk_result.json"
    _atomic_json(
        destination,
        {
            "schema_version": DISTRIBUTED_ARENA_SCHEMA_VERSION,
            "pair_start": pair_start,
            "pair_count": pair_count,
            "arena": result.summary(),
        },
    )
    return destination


def merge_evaluation_chunks(
    result_paths: Sequence[str | Path],
    *,
    expected_games: int,
    tie_tolerance: float = 1e-9,
) -> ArenaResult:
    """Require exact pair/scenario coverage and recompute aggregate role scores."""

    if expected_games <= 0:
        raise ValueError("expected_games must be positive")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be nonnegative and finite")
    games: list[ArenaGameResult] = []
    incumbent_versions: set[int] = set()
    candidate_versions: set[int] = set()
    declared_ranges: set[int] = set()
    for path in result_paths:
        value = _read_object(Path(path).expanduser().resolve())
        if value.get("schema_version") != DISTRIBUTED_ARENA_SCHEMA_VERSION:
            raise ValueError("distributed arena chunk uses another schema")
        start = int(value.get("pair_start", -1))
        count = int(value.get("pair_count", -1))
        expected_range = set(range(start, start + count))
        if start < 0 or count <= 0 or declared_ranges & expected_range:
            raise ValueError("distributed arena chunks overlap or have invalid ranges")
        declared_ranges.update(expected_range)
        arena = value.get("arena")
        if not isinstance(arena, Mapping):
            raise ValueError("distributed arena chunk has no arena result")
        incumbent_versions.add(int(arena["incumbent_bundle_version"]))
        candidate_versions.add(int(arena["candidate_bundle_version"]))
        games.extend(ArenaGameResult(**row) for row in arena.get("games", ()))

    if declared_ranges != set(range(expected_games)):
        raise ValueError("distributed arena chunks do not cover every requested pair")
    if len(incumbent_versions) != 1 or len(candidate_versions) != 1:
        raise ValueError("distributed arena chunks mixed model bundles")

    by_pair: dict[int, dict[str, ArenaGameResult]] = {}
    for game in games:
        scenarios = by_pair.setdefault(game.pair_index, {})
        if game.scenario in scenarios:
            raise ValueError("distributed arena repeats one pair scenario")
        scenarios[game.scenario] = game
    required = {"incumbent", "candidate_controller", "candidate_adversary"}
    if set(by_pair) != set(range(expected_games)) or any(
        set(scenarios) != required for scenarios in by_pair.values()
    ):
        raise ValueError("distributed arena has incomplete scenario coverage")
    for scenarios in by_pair.values():
        if len({game.paired_seed for game in scenarios.values()}) != 1:
            raise ValueError("paired arena scenarios used different seeds")

    baseline = [
        by_pair[index]["incumbent"].final_cost for index in range(expected_games)
    ]
    controller = [
        by_pair[index]["candidate_controller"].final_cost
        for index in range(expected_games)
    ]
    adversary = [
        by_pair[index]["candidate_adversary"].final_cost
        for index in range(expected_games)
    ]
    return ArenaResult(
        incumbent_bundle_version=next(iter(incumbent_versions)),
        candidate_bundle_version=next(iter(candidate_versions)),
        games=tuple(
            game
            for index in range(expected_games)
            for game in (
                by_pair[index]["incumbent"],
                by_pair[index]["candidate_controller"],
                by_pair[index]["candidate_adversary"],
            )
        ),
        controller=summarize_role_results(
            "controller", baseline, controller, tie_tolerance
        ),
        adversary=summarize_role_results(
            "adversary", baseline, adversary, tie_tolerance
        ),
    )


def evaluate_candidate_distributed(
    experiment: ExperimentConfig,
    cluster: ClusterSpec,
    incumbent: LoadedModelBundle,
    candidate: LoadedModelBundle,
    *,
    experiment_config_path: str | Path,
    output_dir: str | Path,
) -> ArenaResult:
    """Stage inputs, run host chunks concurrently, collect, and merge results."""

    settings = experiment.distributed_evaluation
    if not settings.enabled:
        raise ValueError("distributed evaluation is disabled")
    engine = experiment.resolve_engine_config()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    chunks = plan_evaluation_chunks(
        experiment.arena.games,
        cluster.arena_hosts,
        games_per_chunk=settings.games_per_chunk,
    )
    run_id = f"arena_{candidate.bundle_version}_{time.time_ns()}"
    staged: dict[str, tuple[str, str, str]] = {}
    for host in {chunk.host.host_id: chunk.host for chunk in chunks}.values():
        remote_config = f"{host.experiment_root.rstrip('/')}/experiment_config.json"
        copy_to_host(experiment_config_path, host, remote_config)
        staged[host.host_id] = (
            remote_config,
            sync_model_bundle(incumbent.manifest_path, host, config=engine),
            sync_model_bundle(candidate.manifest_path, host, config=engine),
        )

    semaphores = {
        host.host_id: threading.Semaphore(settings.max_parallel_chunks_per_host)
        for host in cluster.arena_hosts
    }

    def execute(chunk: EvaluationChunk) -> Path:
        host = chunk.host
        remote_config, remote_incumbent, remote_candidate = staged[host.host_id]
        remote_output = (
            f"{host.experiment_root.rstrip('/')}/distributed_eval/{run_id}/"
            f"chunk_{chunk.chunk_id:06d}"
        )
        command = [
            host.python_executable,
            f"{host.package_root}/AlphaGoZeroGV4/process_entrypoint.py",
            "AlphaGoZeroGV4.distributed_operations.distributed_eval",
            "--experiment-config",
            remote_config,
            "--incumbent",
            remote_incumbent,
            "--candidate",
            remote_candidate,
            "--pair-start",
            str(chunk.pair_start),
            "--pair-count",
            str(chunk.pair_count),
            "--output-dir",
            remote_output,
        ]
        environment = dict(experiment.deployment.environment)
        environment["PYTHONPATH"] = host.repo_root
        with semaphores[host.host_id]:
            completed = run_command(
                host,
                command,
                cwd=host.repo_root,
                environment=environment,
            )
        local_chunk = output / "chunks" / f"chunk_{chunk.chunk_id:06d}"
        copy_from_host(host, remote_output, local_chunk)
        (local_chunk / "process_output.log").write_text(
            completed.stdout or "", encoding="utf-8"
        )
        if not settings.keep_remote_outputs and not host.is_local:
            run_shell(host, f"rm -rf {shlex.quote(remote_output)}", check=False)
        return local_chunk / "chunk_result.json"

    with ThreadPoolExecutor(
        max_workers=len(cluster.arena_hosts) * settings.max_parallel_chunks_per_host
    ) as executor:
        result_paths = tuple(executor.map(execute, chunks))
    merged = merge_evaluation_chunks(
        result_paths,
        expected_games=experiment.arena.games,
        tie_tolerance=experiment.arena.tie_tolerance,
    )
    _atomic_json(
        output / "distributed_arena_result.json",
        {
            "schema_version": DISTRIBUTED_ARENA_SCHEMA_VERSION,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "pair_start": chunk.pair_start,
                    "pair_count": chunk.pair_count,
                    "host_id": chunk.host.host_id,
                }
                for chunk in chunks
            ],
            "arena": merged.summary(),
        },
    )
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--cluster", type=Path)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pair-start", type=int)
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    experiment = load_experiment_config(args.experiment_config)
    engine = experiment.resolve_engine_config()
    incumbent = load_model_bundle(args.incumbent, config=engine, device="cpu")
    candidate = load_model_bundle(args.candidate, config=engine, device="cpu")
    if args.pair_start is not None or args.pair_count is not None:
        if args.pair_start is None or args.pair_count is None:
            raise ValueError("pair-start and pair-count must be supplied together")
        path = run_evaluation_chunk(
            experiment,
            incumbent,
            candidate,
            pair_start=args.pair_start,
            pair_count=args.pair_count,
            output_dir=args.output_dir,
        )
        print(path)
        return
    if args.cluster is None:
        raise ValueError("--cluster is required for distributed orchestration")
    result = evaluate_candidate_distributed(
        experiment,
        load_cluster(args.cluster),
        incumbent,
        candidate,
        experiment_config_path=args.experiment_config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
