"""Distribute independent arena games across XL and self-play workers.

The existing arena launcher remains the source of truth for individual games.
This module only partitions each deterministic game plan, launches the existing
launcher on isolated CPU sets, and merges its standard CSV outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from vidur.AlphaGoZero.cluster import REMOTE_REPO, WORKERS


MODEL_PATH_FLAGS = {
    "--model-path",
    "--controller-prior-model-path",
    "--adversary-prior-model-path",
    "--role-controller-value-model-path",
    "--role-controller-prior-model-path",
    "--role-adversary-value-model-path",
    "--role-adversary-prior-model-path",
}


@dataclass(frozen=True)
class EvalHost:
    label: str
    ssh_host: str | None
    game_quota: int
    parallel_games: int
    cpu_set: str
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    memory_reserve_bytes: int = 0
    memory_per_game_bytes: int = 0
    rollout_parallel_threads: int = 1
    cpu_set_count: int = 0
    cpu_idle_fraction: float = 0.0
    cpu_reserve_cores: int = 0
    cpu_usable_cores: int = 0

    @property
    def is_local(self) -> bool:
        return self.ssh_host is None


@dataclass(frozen=True)
class ArenaChunk:
    block_name: str
    host: EvalHost
    game_offset: int
    num_games: int
    parallel_games: int
    output_dir: Path
    command: tuple[str, ...]


def _env_int(name: str, default: int) -> int:
    value = str(os.environ.get(name, "")).strip()
    return int(value) if value else int(default)


def _env_float(name: str, default: float) -> float:
    value = str(os.environ.get(name, "")).strip()
    return float(value) if value else float(default)


def _parse_cpu_set(cpu_set: str) -> set[int]:
    cpus: set[int] = set()
    for raw_part in str(cpu_set).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start, end = int(raw_start), int(raw_end)
            if end < start:
                raise ValueError(f"invalid CPU range {part!r}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"empty CPU set {cpu_set!r}")
    return cpus


def _parse_proc_stat_pair(raw: str) -> tuple[int, float]:
    snapshots = str(raw).split("__AGZ_CPU_SAMPLE__")
    if len(snapshots) != 2:
        raise RuntimeError("CPU sample did not contain exactly two /proc/stat snapshots")

    def parse(snapshot: str) -> tuple[int, int, int]:
        aggregate: list[int] | None = None
        logical_cpus = 0
        for line in snapshot.splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "cpu":
                aggregate = [int(value) for value in fields[1:]]
            elif fields[0].startswith("cpu") and fields[0][3:].isdigit():
                logical_cpus += 1
        if aggregate is None or len(aggregate) < 5 or logical_cpus <= 0:
            raise RuntimeError("invalid /proc/stat CPU sample")
        total = sum(aggregate)
        idle = aggregate[3] + aggregate[4]
        return logical_cpus, total, idle

    logical_before, total_before, idle_before = parse(snapshots[0])
    logical_after, total_after, idle_after = parse(snapshots[1])
    if logical_before != logical_after:
        raise RuntimeError("logical CPU count changed while sampling")
    total_delta = total_after - total_before
    idle_delta = idle_after - idle_before
    if total_delta <= 0:
        raise RuntimeError("non-positive /proc/stat sampling interval")
    idle_fraction = min(1.0, max(0.0, float(idle_delta) / float(total_delta)))
    return logical_after, idle_fraction


def _host_cpu_snapshot(host: EvalHost, sample_sec: float) -> tuple[int, float]:
    marker = "\n__AGZ_CPU_SAMPLE__\n"
    if host.is_local:
        before = Path("/proc/stat").read_text(encoding="utf-8")
        time.sleep(sample_sec)
        after = Path("/proc/stat").read_text(encoding="utf-8")
        return _parse_proc_stat_pair(before + marker + after)
    command = (
        "cat /proc/stat; printf '\\n__AGZ_CPU_SAMPLE__\\n'; "
        f"sleep {max(0.05, float(sample_sec)):.3f}; cat /proc/stat"
    )
    result = subprocess.run(
        ["ssh", str(host.ssh_host), command],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(15, int(math.ceil(sample_sec)) + 10),
    )
    return _parse_proc_stat_pair(result.stdout)


def cpu_safe_hosts(
    hosts: list[EvalHost],
    *,
    phase: str,
    total_games: int,
    cpu_by_label: dict[str, tuple[int, float]] | None = None,
) -> list[EvalHost]:
    """Resolve one-wave game slots and MCTS threads from live free CPU capacity."""

    if phase not in {"role", "sjf"}:
        raise ValueError(f"unsupported evaluation phase {phase!r}")
    thread_env = (
        "AGZ_DISTRIBUTED_EVAL_ROLE_THREADS_PER_GAME"
        if phase == "role"
        else "AGZ_DISTRIBUTED_EVAL_SJF_THREADS_PER_GAME"
    )
    threads_per_game = max(1, _env_int(thread_env, 2 if phase == "role" else 8))
    reserve_floor = max(0, _env_int("AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_CORES", 2))
    reserve_fraction = min(
        0.95,
        max(0.0, _env_float("AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_FRACTION", 0.0)),
    )
    sample_sec = max(0.05, _env_float("AGZ_DISTRIBUTED_EVAL_CPU_SAMPLE_SEC", 0.25))
    if cpu_by_label is None:
        with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as executor:
            snapshots = list(executor.map(lambda host: _host_cpu_snapshot(host, sample_sec), hosts))
        cpu_by_label = dict(zip((host.label for host in hosts), snapshots, strict=True))

    resolved_hosts: list[EvalHost] = []
    for host in hosts:
        logical_cpus, idle_fraction = cpu_by_label[host.label]
        selected_cpus = {cpu for cpu in _parse_cpu_set(host.cpu_set) if cpu < logical_cpus}
        selected_count = len(selected_cpus)
        reserve_cores = max(reserve_floor, int(math.ceil(selected_count * reserve_fraction)))
        live_free_cores = int(math.floor(selected_count * idle_fraction))
        usable_cores = max(0, min(selected_count - reserve_cores, live_free_cores - reserve_cores))
        cpu_slots = usable_cores // threads_per_game
        effective_slots = min(int(host.parallel_games), int(cpu_slots))
        resolved = replace(
            host,
            parallel_games=max(0, int(effective_slots)),
            rollout_parallel_threads=int(threads_per_game),
            cpu_set_count=int(selected_count),
            cpu_idle_fraction=float(idle_fraction),
            cpu_reserve_cores=int(reserve_cores),
            cpu_usable_cores=int(usable_cores),
        )
        if resolved.parallel_games > 0:
            resolved_hosts.append(resolved)
        print(
            "[distributed-eval-cpu] phase={} host={} logical={} selected={} idle={:.1%} "
            "reserve={} usable={} threads_per_game={} cpu_slots={} memory_limited_slots={} effective={}".format(
                phase,
                host.label,
                logical_cpus,
                selected_count,
                idle_fraction,
                reserve_cores,
                usable_cores,
                threads_per_game,
                cpu_slots,
                host.parallel_games,
                effective_slots,
            ),
            flush=True,
        )

    if phase == "role":
        one_wave_capacity = 2 * sum(host.parallel_games // 2 for host in resolved_hosts)
    else:
        one_wave_capacity = sum(host.parallel_games for host in resolved_hosts)
    if one_wave_capacity < int(total_games):
        raise RuntimeError(
            f"insufficient one-wave CPU capacity for {phase}: "
            f"capacity={one_wave_capacity} required={int(total_games)}"
        )
    return resolved_hosts


def _parse_meminfo(raw: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in {"MemTotal", "MemAvailable"}:
            continue
        fields = value.strip().split()
        if not fields:
            continue
        multiplier = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
        values[key] = int(fields[0]) * multiplier
    if values.get("MemTotal", 0) <= 0 or values.get("MemAvailable", 0) <= 0:
        raise RuntimeError("could not read MemTotal/MemAvailable from /proc/meminfo")
    return values["MemTotal"], values["MemAvailable"]


def _host_meminfo(host: EvalHost) -> tuple[int, int]:
    if host.is_local:
        return _parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    result = subprocess.run(
        ["ssh", str(host.ssh_host), "cat /proc/meminfo"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    return _parse_meminfo(result.stdout)


def memory_safe_hosts(
    hosts: list[EvalHost],
    *,
    active_blocks: int,
    meminfo_by_label: dict[str, tuple[int, int]] | None = None,
) -> list[EvalHost]:
    """Cap aggregate arena concurrency using live available RAM per machine."""

    gib = 1024**3
    per_game_bytes = max(1, int(_env_float("AGZ_DISTRIBUTED_EVAL_GAME_MEMORY_GIB", 4.0) * gib))
    reserve_bytes_floor = max(0, int(_env_float("AGZ_DISTRIBUTED_EVAL_MEMORY_RESERVE_GIB", 16.0) * gib))
    reserve_fraction = min(
        0.95,
        max(0.0, _env_float("AGZ_DISTRIBUTED_EVAL_MEMORY_RESERVE_FRACTION", 0.25)),
    )
    required_slots = max(1, int(active_blocks))
    limited: list[EvalHost] = []
    for host in hosts:
        total_bytes, available_bytes = (
            meminfo_by_label[host.label]
            if meminfo_by_label is not None and host.label in meminfo_by_label
            else _host_meminfo(host)
        )
        reserve_bytes = max(reserve_bytes_floor, int(total_bytes * reserve_fraction))
        memory_budget_bytes = max(0, int(available_bytes) - int(reserve_bytes))
        ram_slots = int(memory_budget_bytes // per_game_bytes)
        effective_slots = min(int(host.parallel_games), ram_slots)
        if effective_slots < required_slots:
            raise RuntimeError(
                f"insufficient RAM on {host.label}: available={available_bytes / gib:.1f} GiB "
                f"reserve={reserve_bytes / gib:.1f} GiB per_game={per_game_bytes / gib:.1f} GiB "
                f"requires at least {required_slots} aggregate slots"
            )
        resolved = replace(
            host,
            parallel_games=int(effective_slots),
            memory_total_bytes=int(total_bytes),
            memory_available_bytes=int(available_bytes),
            memory_reserve_bytes=int(reserve_bytes),
            memory_per_game_bytes=int(per_game_bytes),
        )
        limited.append(resolved)
        print(
            "[distributed-eval-memory] host={} total={:.1f}GiB available={:.1f}GiB "
            "reserve={:.1f}GiB per_game={:.1f}GiB cpu_cap={} ram_cap={} effective={}".format(
                host.label,
                total_bytes / gib,
                available_bytes / gib,
                reserve_bytes / gib,
                per_game_bytes / gib,
                host.parallel_games,
                ram_slots,
                effective_slots,
            ),
            flush=True,
        )
    return limited


def distributed_eval_enabled() -> bool:
    return bool(_env_int("AGZ_DISTRIBUTED_EVAL_ENABLED", 1))


def distributed_eval_pauses_selfplay() -> bool:
    return bool(_env_int("AGZ_DISTRIBUTED_EVAL_PAUSE_SELFPLAY", 0))


def _worker_eval_control_paths(root: Path, worker_id: str) -> tuple[Path, Path]:
    control = Path(root) / "worker_large" / str(worker_id) / "control"
    return control / "eval_pause.request", control / "eval_pause.ack.json"


def _request_worker_eval_pause(root: Path, worker: Any, timeout_sec: int) -> str:
    request, ack = _worker_eval_control_paths(root, worker.worker_id)
    setup = (
        f"mkdir -p {shlex.quote(str(request.parent))} && "
        f"rm -f {shlex.quote(str(ack))} && "
        f": > {shlex.quote(str(request))}"
    )
    subprocess.run(["ssh", str(worker.host), setup], check=True, timeout=20)
    wait = (
        f"i=0; while [ ! -f {shlex.quote(str(ack))} ]; do "
        f"i=$((i+1)); [ $i -ge {max(1, int(timeout_sec) * 2)} ] && exit 1; "
        "sleep 0.5; done; "
        f"cat {shlex.quote(str(ack))}"
    )
    result = subprocess.run(
        ["ssh", str(worker.host), wait],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(5, int(timeout_sec) + 10),
    )
    return result.stdout.strip()


def _resume_worker_selfplay(root: Path, worker: Any) -> None:
    request, ack = _worker_eval_control_paths(root, worker.worker_id)
    subprocess.run(
        [
            "ssh",
            str(worker.host),
            f"rm -f {shlex.quote(str(request))} {shlex.quote(str(ack))}",
        ],
        check=True,
        timeout=20,
    )


@contextmanager
def pause_selfplay_for_eval(root: Path) -> Iterable[None]:
    """Quiesce worker games so evaluation can use the full cluster CPU budget."""

    if not distributed_eval_enabled() or not distributed_eval_pauses_selfplay():
        yield
        return
    timeout_sec = _env_int("AGZ_DISTRIBUTED_EVAL_PAUSE_TIMEOUT_SEC", 30)
    try:
        with ThreadPoolExecutor(max_workers=len(WORKERS)) as executor:
            acknowledgements = list(
                executor.map(
                    lambda worker: _request_worker_eval_pause(root, worker, timeout_sec),
                    WORKERS,
                )
            )
        print(
            f"[distributed-eval-pause] paused_workers={len(acknowledgements)}",
            flush=True,
        )
        yield
    finally:
        with ThreadPoolExecutor(max_workers=len(WORKERS)) as executor:
            list(executor.map(lambda worker: _resume_worker_selfplay(root, worker), WORKERS))
        print(f"[distributed-eval-pause] resumed_workers={len(WORKERS)}", flush=True)


def default_role_hosts() -> list[EvalHost]:
    xl_quota = _env_int("AGZ_DISTRIBUTED_EVAL_XL_GAMES", 32)
    xl_parallel = _env_int("AGZ_DISTRIBUTED_EVAL_XL_PARALLEL", 46)
    worker_parallel = _env_int("AGZ_DISTRIBUTED_EVAL_WORKER_PARALLEL", 46)
    worker_cpu_set = os.environ.get("AGZ_DISTRIBUTED_EVAL_WORKER_CPUSET", "2-95")
    xl_cpu_set = os.environ.get("AGZ_DISTRIBUTED_EVAL_XL_CPUSET", "4-95")
    hosts = [EvalHost("xl", None, xl_quota, xl_parallel, xl_cpu_set)]
    remaining = 400 - int(xl_quota)
    base, extra = divmod(remaining, len(WORKERS))
    if extra % 2:
        raise ValueError("role-evaluation worker remainder must be pair-distributable")
    worker_quotas = [base + 2 * int(index < extra // 2) for index in range(len(WORKERS))]
    for worker, quota in zip(WORKERS, worker_quotas, strict=True):
        hosts.append(EvalHost(worker.worker_id, worker.host, quota, worker_parallel, worker_cpu_set))
    return hosts


def default_sjf_hosts(num_games: int) -> list[EvalHost]:
    # Quotas express the preferred split; parallel capacity permits live redistribution.
    worker_games = min(
        _env_int("AGZ_DISTRIBUTED_EVAL_SJF_WORKER_GAMES", 5),
        max(0, int(num_games) // max(1, len(WORKERS) + 2)),
    )
    xl_games = int(num_games) - worker_games * len(WORKERS)
    sjf_parallel = max(1, _env_int("AGZ_DISTRIBUTED_EVAL_SJF_HOST_PARALLEL", 12))
    hosts = [
        EvalHost(
            "xl",
            None,
            xl_games,
            min(int(num_games), sjf_parallel),
            os.environ.get("AGZ_DISTRIBUTED_EVAL_XL_CPUSET", "4-95"),
        )
    ]
    for worker in WORKERS:
        hosts.append(
            EvalHost(
                worker.worker_id,
                worker.host,
                worker_games,
                min(int(num_games), sjf_parallel),
                os.environ.get("AGZ_DISTRIBUTED_EVAL_WORKER_CPUSET", "2-95"),
            )
        )
    return [host for host in hosts if host.game_quota > 0]


def _command_value(command: list[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing command argument {flag}") from exc


def _set_command_value(command: list[str], flag: str, value: str | int) -> None:
    try:
        command[command.index(flag) + 1] = str(value)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing command argument {flag}") from exc


def _weighted_capacity_allocation(capacities: list[int], total: int) -> list[int]:
    if sum(capacities) < int(total):
        raise RuntimeError(f"one-wave capacity {sum(capacities)} is below required {int(total)}")
    capacity_sum = sum(capacities)
    targets = [float(total) * capacity / capacity_sum for capacity in capacities]
    allocations = [min(capacity, int(math.floor(target))) for capacity, target in zip(capacities, targets, strict=True)]
    while sum(allocations) < int(total):
        candidates = [index for index, capacity in enumerate(capacities) if allocations[index] < capacity]
        if not candidates:
            raise RuntimeError("could not finish one-wave capacity allocation")
        index = max(
            candidates,
            key=lambda item: (
                targets[item] - allocations[item],
                capacities[item] - allocations[item],
                -item,
            ),
        )
        allocations[index] += 1
    return allocations


def plan_role_chunks(
    block_commands: dict[str, list[str]],
    *,
    scratch_root: Path,
    hosts: list[EvalHost] | None = None,
) -> list[ArenaChunk]:
    hosts = list(hosts or default_role_hosts())
    if not block_commands:
        return []
    block_names = list(block_commands)
    games_per_block = {_command_value(cmd, "--num-games") for cmd in block_commands.values()}
    if len(games_per_block) != 1:
        raise ValueError("distributed role blocks must have equal game counts")
    num_games = int(next(iter(games_per_block)))

    pair_keys: list[tuple[str, str]] = []
    pair_index_by_block: dict[str, int] = {}
    for name in block_names:
        command = block_commands[name]
        key = (_command_value(command, "--game-id-start"), _command_value(command, "--history-seed"))
        if key not in pair_keys:
            pair_keys.append(key)
        pair_index_by_block[name] = pair_keys.index(key)
    pair_sizes = [
        sum(index == pair_index_by_block[name] for name in block_names)
        for index in range(len(pair_keys))
    ]
    if sorted(pair_sizes) != [2, 2]:
        raise ValueError("role evaluation requires exactly two paired baseline/candidate block groups")

    # A pair unit runs matching baseline/candidate games concurrently on one host.
    pair_capacities = [host.parallel_games // 2 for host in hosts]
    host_pair_units = _weighted_capacity_allocation(pair_capacities, 2 * num_games)
    pair_zero_units = [units // 2 for units in host_pair_units]
    pair_zero_gap = num_games - sum(pair_zero_units)
    odd_hosts = [index for index, units in enumerate(host_pair_units) if units % 2]
    if pair_zero_gap < 0 or pair_zero_gap > len(odd_hosts):
        raise RuntimeError("could not split role capacity evenly across paired blocks")
    for index in odd_hosts[:pair_zero_gap]:
        pair_zero_units[index] += 1
    pair_one_units = [
        total_units - first_pair
        for total_units, first_pair in zip(host_pair_units, pair_zero_units, strict=True)
    ]
    if sum(pair_zero_units) != num_games or sum(pair_one_units) != num_games:
        raise RuntimeError("role pair allocation did not cover each 100-game block")

    pair_units_by_host = {
        host.label: (pair_zero_units[index], pair_one_units[index])
        for index, host in enumerate(hosts)
    }
    counts_by_host = {
        host.label: [
            pair_units_by_host[host.label][pair_index_by_block[block_name]]
            for block_name in block_names
        ]
        for host in hosts
    }

    chunks: list[ArenaChunk] = []
    offsets = {name: 0 for name in block_names}
    for host in hosts:
        host_counts = counts_by_host[host.label]
        if sum(host_counts) > host.parallel_games:
            raise RuntimeError(f"role plan oversubscribes {host.label}")
        for block_index, block_name in enumerate(block_names):
            count = host_counts[block_index]
            if count <= 0:
                continue
            offset = offsets[block_name]
            offsets[block_name] += count
            part_command = list(block_commands[block_name])
            base_game_id = int(_command_value(part_command, "--game-id-start"))
            part_dir = scratch_root / block_name / host.label
            _set_command_value(part_command, "--output-dir", str(part_dir))
            _set_command_value(part_command, "--game-id-start", base_game_id + offset)
            _set_command_value(part_command, "--num-games", count)
            _set_command_value(part_command, "--num-parallel-games", count)
            _set_command_value(
                part_command,
                "--rollout-parallel-threads",
                host.rollout_parallel_threads,
            )
            part_command.extend(["--history-hops-offset", str(offset)])
            part_command.append("--history-hops-prefix-stable")
            chunks.append(
                ArenaChunk(
                    block_name=block_name,
                    host=host,
                    game_offset=offset,
                    num_games=count,
                    parallel_games=count,
                    output_dir=part_dir,
                    command=tuple(part_command),
                )
            )
    for block_name, offset in offsets.items():
        if offset != num_games:
            raise ValueError(f"incomplete block plan for {block_name}: {offset} != {num_games}")
    return chunks


def plan_single_block_chunks(
    block_name: str,
    command: list[str],
    *,
    scratch_root: Path,
    hosts: list[EvalHost],
) -> list[ArenaChunk]:
    num_games = int(_command_value(command, "--num-games"))
    counts = [0] * len(hosts)
    remaining = num_games
    for index, host in enumerate(hosts):
        preferred = min(host.game_quota, host.parallel_games, remaining)
        counts[index] = preferred
        remaining -= preferred
    while remaining > 0:
        candidates = [
            index for index, host in enumerate(hosts)
            if counts[index] < host.parallel_games
        ]
        if not candidates:
            raise RuntimeError(
                f"insufficient one-wave SJF capacity: planned={num_games - remaining} required={num_games}"
            )
        index = max(
            candidates,
            key=lambda item: (hosts[item].parallel_games - counts[item], -item),
        )
        counts[index] += 1
        remaining -= 1

    chunks: list[ArenaChunk] = []
    offset = 0
    base_game_id = int(_command_value(command, "--game-id-start"))
    for host, count in zip(hosts, counts, strict=True):
        if count <= 0:
            continue
        part_dir = scratch_root / block_name / host.label
        part_command = list(command)
        _set_command_value(part_command, "--output-dir", str(part_dir))
        _set_command_value(part_command, "--game-id-start", base_game_id + offset)
        _set_command_value(part_command, "--num-games", count)
        _set_command_value(part_command, "--num-parallel-games", count)
        _set_command_value(
            part_command,
            "--rollout-parallel-threads",
            host.rollout_parallel_threads,
        )
        part_command.extend(["--history-hops-offset", str(offset)])
        part_command.append("--history-hops-prefix-stable")
        chunks.append(
            ArenaChunk(
                block_name=block_name,
                host=host,
                game_offset=offset,
                num_games=count,
                parallel_games=count,
                output_dir=part_dir,
                command=tuple(part_command),
            )
        )
        offset += count
    if offset != num_games:
        raise RuntimeError(f"incomplete SJF plan: {offset} != {num_games}")
    return chunks


def plan_independent_block_chunks(
    block_commands: dict[str, list[str]],
    *,
    scratch_root: Path,
    hosts: list[EvalHost],
) -> list[ArenaChunk]:
    """Plan independent blocks together without exceeding per-host capacity.

    This is used for the two SJF cycles. Each cycle keeps the same global game
    offsets, so prefix-stable history sampling gives both cycles identical roots,
    while the cycle jobs can execute concurrently across the cluster.
    """

    if not block_commands:
        return []
    capacities = [int(host.parallel_games) for host in hosts]
    total_games = sum(
        int(_command_value(command, "--num-games"))
        for command in block_commands.values()
    )
    if sum(capacities) < total_games:
        raise RuntimeError(
            f"insufficient one-wave independent-block capacity: "
            f"capacity={sum(capacities)} required={total_games}"
        )

    remaining_capacities = list(capacities)
    chunks: list[ArenaChunk] = []
    for block_name, command in block_commands.items():
        num_games = int(_command_value(command, "--num-games"))
        counts = _weighted_capacity_allocation(remaining_capacities, num_games)
        offset = 0
        base_game_id = int(_command_value(command, "--game-id-start"))
        for host_index, (host, count) in enumerate(zip(hosts, counts, strict=True)):
            if count <= 0:
                continue
            remaining_capacities[host_index] -= int(count)
            part_dir = scratch_root / block_name / host.label
            part_command = list(command)
            _set_command_value(part_command, "--output-dir", str(part_dir))
            _set_command_value(part_command, "--game-id-start", base_game_id + offset)
            _set_command_value(part_command, "--num-games", count)
            _set_command_value(part_command, "--num-parallel-games", count)
            _set_command_value(
                part_command,
                "--rollout-parallel-threads",
                host.rollout_parallel_threads,
            )
            part_command.extend(["--history-hops-offset", str(offset)])
            part_command.append("--history-hops-prefix-stable")
            chunks.append(
                ArenaChunk(
                    block_name=block_name,
                    host=host,
                    game_offset=offset,
                    num_games=count,
                    parallel_games=count,
                    output_dir=part_dir,
                    command=tuple(part_command),
                )
            )
            offset += int(count)
        if offset != num_games:
            raise RuntimeError(
                f"incomplete independent block plan for {block_name}: {offset} != {num_games}"
            )

    for host in hosts:
        assigned = sum(
            chunk.num_games for chunk in chunks if chunk.host.label == host.label
        )
        if assigned > int(host.parallel_games):
            raise RuntimeError(f"independent block plan oversubscribes {host.label}")
    return chunks


def _required_model_paths(commands: Iterable[Iterable[str]]) -> list[Path]:
    paths: set[Path] = set()
    for raw_command in commands:
        command = list(raw_command)
        for index, item in enumerate(command[:-1]):
            if item in MODEL_PATH_FLAGS:
                paths.add(Path(command[index + 1]))
    return sorted(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_models(chunks: list[ArenaChunk]) -> dict[str, str]:
    model_paths = _required_model_paths(chunk.command for chunk in chunks)
    local_hashes: dict[str, str] = {}
    for path in model_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        local_hashes[str(path)] = _sha256(path)

    remote_hosts = sorted(
        {chunk.host for chunk in chunks if not chunk.host.is_local},
        key=lambda host: host.label,
    )

    def stage_host(host: EvalHost) -> None:
        parents = sorted({str(path.parent) for path in model_paths})
        subprocess.run(
            ["ssh", str(host.ssh_host), "mkdir -p " + " ".join(shlex.quote(p) for p in parents)],
            check=True,
        )
        for path in model_paths:
            subprocess.run(
                ["rsync", "-az", "--partial", "--delay-updates", str(path), f"{host.ssh_host}:{path}"],
                check=True,
            )
        remote_hashes = subprocess.run(
            ["ssh", str(host.ssh_host), "sha256sum " + " ".join(shlex.quote(str(p)) for p in model_paths)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
        parsed = {
            line.split(maxsplit=1)[1].lstrip("*"): line.split(maxsplit=1)[0]
            for line in remote_hashes
        }
        for path in model_paths:
            if parsed.get(str(path)) != local_hashes[str(path)]:
                raise RuntimeError(f"model checksum mismatch on {host.label}: {path}")

    with ThreadPoolExecutor(max_workers=max(1, len(remote_hosts))) as executor:
        list(executor.map(stage_host, remote_hosts))
    return local_hashes
def validate_arena_runner_compatibility(chunks: list[ArenaChunk]) -> None:
    """Fail before staging/launching when a host has a stale arena CLI."""

    module = "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP"
    repo = Path(__file__).resolve().parents[2]
    seen: set[tuple[str, str | None]] = set()
    for chunk in chunks:
        host = chunk.host
        key = (host.label, host.ssh_host)
        if key in seen:
            continue
        seen.add(key)
        if host.is_local:
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=str(repo),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        else:
            remote_python = f"{REMOTE_REPO}/.venv/bin/python3"
            remote = (
                f"cd {shlex.quote(REMOTE_REPO)} && "
                f"{shlex.quote(remote_python)} -m {shlex.quote(module)} --help"
            )
            result = subprocess.run(
                ["ssh", str(host.ssh_host), remote],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        required_flags = ("--discount-factor", "--native-search-mode", "--rollout-horizon-sec")
        missing_flags = [flag for flag in required_flags if flag not in result.stdout]
        if result.returncode != 0 or missing_flags:
            tail = result.stdout[-1200:].strip()
            raise RuntimeError(
                f"incompatible arena runner on {host.label}: missing flags={missing_flags}; "
                f"rc={result.returncode}; output={tail!r}"
            )


def _safe_scratch(path: Path) -> None:
    if ".distributed_eval_scratch" not in path.parts:
        raise ValueError(f"refusing to manage non-scratch path: {path}")


def _launch_chunk(chunk: ArenaChunk, log_path: Path) -> subprocess.Popen[str]:
    _safe_scratch(chunk.output_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_prefix = [
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    ]
    wrapped = [
        *env_prefix,
        "taskset",
        "-c",
        chunk.host.cpu_set,
        "nice",
        "-n",
        "10",
        "ionice",
        "-c2",
        "-n7",
        *chunk.command,
    ]
    if chunk.host.is_local:
        shutil.rmtree(chunk.output_dir, ignore_errors=True)
        chunk.output_dir.mkdir(parents=True, exist_ok=True)
        command = wrapped
    else:
        remote = (
            f"rm -rf {shlex.quote(str(chunk.output_dir))} && "
            f"mkdir -p {shlex.quote(str(chunk.output_dir))} && "
            f"cd {shlex.quote(REMOTE_REPO)} && exec {shlex.join(wrapped)}"
        )
        command = ["ssh", str(chunk.host.ssh_host), remote]
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process._agz_log_handle = handle  # type: ignore[attr-defined]
    return process


def _merge_csv(parts: list[Path], output: Path, *, expected_games: int) -> None:
    header: list[str] | None = None
    rows: dict[int, dict[str, str]] = {}
    for path in parts:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            current_header = list(reader.fieldnames or [])
            if header is None:
                header = current_header
            elif header != current_header:
                raise RuntimeError(f"CSV schema mismatch while merging {path}")
            for row in reader:
                game_id = int(float(row.get("game_id", 0) or 0))
                if game_id in rows:
                    raise RuntimeError(f"duplicate distributed result for game {game_id}")
                rows[game_id] = row
    if len(rows) != int(expected_games):
        raise RuntimeError(f"distributed result count {len(rows)} != {expected_games}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or [])
        writer.writeheader()
        for game_id in sorted(rows):
            writer.writerow(rows[game_id])


def _collect_remote_chunk(chunk: ArenaChunk, local_part: Path) -> Path:
    if chunk.host.is_local:
        return chunk.output_dir
    shutil.rmtree(local_part, ignore_errors=True)
    local_part.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", f"{chunk.host.ssh_host}:{str(chunk.output_dir).rstrip('/')}/", str(local_part).rstrip("/") + "/"],
        check=True,
    )
    return local_part


def _cleanup_successful_chunks(chunks: list[ArenaChunk], final_dirs: Iterable[Path]) -> None:
    remote_paths: dict[str, set[Path]] = {}
    for chunk in chunks:
        _safe_scratch(chunk.output_dir)
        if chunk.host.is_local:
            shutil.rmtree(chunk.output_dir, ignore_errors=True)
        else:
            remote_paths.setdefault(str(chunk.host.ssh_host), set()).add(chunk.output_dir)
    for host, paths in remote_paths.items():
        command = "rm -rf " + " ".join(shlex.quote(str(path)) for path in sorted(paths))
        subprocess.run(["ssh", host, command], check=True)
    for final_dir in final_dirs:
        shutil.rmtree(Path(final_dir) / "distributed_parts", ignore_errors=True)


def _merge_block(block_dir: Path, part_dirs: list[Path], expected_games: int) -> None:
    block_dir.mkdir(parents=True, exist_ok=True)
    _merge_csv([path / "arena_results.csv" for path in part_dirs], block_dir / "arena_results.csv", expected_games=expected_games)

    planned_rows: dict[int, dict[str, str]] = {}
    status_rows: dict[int, dict[str, str]] = {}
    planned_header: list[str] = []
    status_header: list[str] = []
    for part in part_dirs:
        with (part / "planned_games.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            planned_header = planned_header or list(reader.fieldnames or [])
            for row in reader:
                planned_rows[int(float(row["game_id"]))] = row
        with (part / "job_status.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            status_header = status_header or list(reader.fieldnames or [])
            for row in reader:
                if row.get("status") != "ok":
                    raise RuntimeError(f"distributed arena job did not finish successfully: {row}")
                status_rows[int(float(row["game_id"]))] = row
    if len(planned_rows) != expected_games or len(status_rows) != expected_games:
        raise RuntimeError("distributed planned/status game count mismatch")
    for filename, header, rows in (
        ("planned_games.csv", planned_header, planned_rows),
        ("job_status.csv", status_header, status_rows),
    ):
        with (block_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for game_id in sorted(rows):
                writer.writerow(rows[game_id])

    arena_games = block_dir / "arena_games"
    arena_games.mkdir(parents=True, exist_ok=True)
    for part in part_dirs:
        for source in (part / "arena_games").glob("*.csv"):
            target = arena_games / source.name
            if target.exists():
                raise RuntimeError(f"duplicate arena game log {target.name}")
            shutil.copy2(source, target)


def merge_split_sjf_cycles(
    *,
    trivial_dir: Path,
    model_dir: Path,
    output_dir: Path,
    expected_games: int,
) -> Path:
    """Merge independently evaluated SJF cycles into the standard arena schema."""

    def read_rows(path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = {int(float(row["game_id"])): row for row in reader}
        if len(rows) != int(expected_games):
            raise RuntimeError(
                f"split SJF result count {len(rows)} != {expected_games}: {path}"
            )
        return header, rows

    header, trivial_rows = read_rows(Path(trivial_dir) / "arena_results.csv")
    model_header, model_rows = read_rows(Path(model_dir) / "arena_results.csv")
    if header != model_header:
        raise RuntimeError("split SJF cycle result schemas differ")
    if set(trivial_rows) != set(model_rows):
        raise RuntimeError("split SJF cycles do not cover identical game IDs")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arena_games = output_dir / "arena_games"
    shutil.rmtree(arena_games, ignore_errors=True)
    arena_games.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, str]] = []
    eps = 1e-9
    for game_id in sorted(trivial_rows):
        trivial = trivial_rows[game_id]
        model = model_rows[game_id]
        if int(float(trivial.get("history_hops", 0) or 0)) != int(
            float(model.get("history_hops", 0) or 0)
        ):
            raise RuntimeError(f"split SJF history mismatch for game {game_id}")
        cycle1_cost = float(trivial.get("cycle1_total_cost", 0.0) or 0.0)
        cycle2_cost = float(model.get("cycle2_total_cost", 0.0) or 0.0)
        delta = cycle2_cost - cycle1_cost
        better = (
            str(model.get("cycle2_label", ""))
            if delta < -eps
            else (str(trivial.get("cycle1_label", "")) if delta > eps else "tie")
        )
        merged = dict(model)
        for field in (
            "cycle1_label",
            "cycle1_slo_violations",
            "cycle1_total_lateness",
            "cycle1_total_cost",
            "cycle1_end_reason",
        ):
            merged[field] = trivial.get(field, "")
        merged["cost_delta_cycle2_minus_cycle1"] = str(delta)
        merged["better_cycle"] = better

        for field, component_dir in (
            ("cycle1_log_file", Path(trivial_dir)),
            ("cycle2_log_file", Path(model_dir)),
        ):
            raw_path = (
                trivial.get(field, "")
                if field == "cycle1_log_file"
                else model.get(field, "")
            )
            if not raw_path:
                merged[field] = ""
                continue
            filename = Path(raw_path).name
            source = component_dir / "arena_games" / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            target = arena_games / filename
            if target.exists():
                raise RuntimeError(f"duplicate split SJF arena log {filename}")
            shutil.copy2(source, target)
            merged[field] = str(target)
        merged_rows.append(merged)

    results_path = output_dir / "arena_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(merged_rows)

    for filename in ("planned_games.csv", "job_status.csv"):
        trivial_path = Path(trivial_dir) / filename
        model_path = Path(model_dir) / filename
        _, trivial_meta = read_rows(trivial_path)
        _, model_meta = read_rows(model_path)
        if set(trivial_meta) != set(model_meta):
            raise RuntimeError(f"split SJF {filename} game IDs differ")
        for game_id in trivial_meta:
            trivial_hops = int(float(trivial_meta[game_id].get("history_hops", 0) or 0))
            model_hops = int(float(model_meta[game_id].get("history_hops", 0) or 0))
            if trivial_hops != model_hops:
                raise RuntimeError(
                    f"split SJF {filename} history mismatch for game {game_id}"
                )
        shutil.copy2(model_path, output_dir / filename)

    manifest = {
        "mode": "distributed_split_sjf_cycles_v1",
        "expected_games": int(expected_games),
        "trivial_cycle_dir": str(trivial_dir),
        "model_cycle_dir": str(model_dir),
        "arena_results_csv": str(results_path),
    }
    (output_dir / "split_cycle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return results_path


def run_distributed_chunks(
    chunks: list[ArenaChunk],
    *,
    final_dirs: dict[str, Path],
    expected_games_by_block: dict[str, int],
) -> dict[str, Path]:
    if not chunks:
        raise ValueError("no distributed chunks were planned")
    started_at = time.time()
    validate_arena_runner_compatibility(chunks)
    model_hashes = stage_models(chunks)
    staged_at = time.time()
    log_root = next(iter(final_dirs.values())).parent / "distributed_launcher_logs"
    processes: list[tuple[ArenaChunk, subprocess.Popen[str]]] = []
    for chunk in chunks:
        log_path = log_root / f"{chunk.block_name}_{chunk.host.label}.log"
        processes.append((chunk, _launch_chunk(chunk, log_path)))

    failures: list[str] = []
    for chunk, process in processes:
        returncode = process.wait()
        handle = getattr(process, "_agz_log_handle", None)
        if handle is not None:
            handle.close()
        if returncode != 0:
            failures.append(f"{chunk.block_name}/{chunk.host.label}: rc={returncode}")
    if failures:
        raise RuntimeError("distributed arena launch failed: " + ", ".join(failures))
    games_finished_at = time.time()

    parts_by_block: dict[str, list[Path]] = {name: [] for name in final_dirs}
    for chunk in chunks:
        local_part = final_dirs[chunk.block_name] / "distributed_parts" / chunk.host.label
        parts_by_block[chunk.block_name].append(_collect_remote_chunk(chunk, local_part))
    for block_name, final_dir in final_dirs.items():
        _merge_block(final_dir, parts_by_block[block_name], expected_games_by_block[block_name])
    merged_at = time.time()
    _cleanup_successful_chunks(chunks, final_dirs.values())
    cleaned_at = time.time()

    plan = {
        "mode": "distributed_arena_v1",
        "started_at_epoch": started_at,
        "finished_at_epoch": cleaned_at,
        "elapsed_sec": cleaned_at - started_at,
        "model_stage_elapsed_sec": staged_at - started_at,
        "game_elapsed_sec": games_finished_at - staged_at,
        "collect_merge_elapsed_sec": merged_at - games_finished_at,
        "cleanup_elapsed_sec": cleaned_at - merged_at,
        "model_sha256": model_hashes,
        "chunks": [
            {
                "block_name": chunk.block_name,
                "host": chunk.host.label,
                "ssh_host": chunk.host.ssh_host,
                "game_offset": chunk.game_offset,
                "num_games": chunk.num_games,
                "parallel_games": chunk.parallel_games,
                "cpu_set": chunk.host.cpu_set,
                "rollout_parallel_threads": chunk.host.rollout_parallel_threads,
                "cpu_set_count": chunk.host.cpu_set_count,
                "cpu_idle_fraction_at_plan": chunk.host.cpu_idle_fraction,
                "cpu_reserve_cores": chunk.host.cpu_reserve_cores,
                "cpu_usable_cores": chunk.host.cpu_usable_cores,
                "memory_total_bytes": chunk.host.memory_total_bytes,
                "memory_available_bytes_at_plan": chunk.host.memory_available_bytes,
                "memory_reserve_bytes": chunk.host.memory_reserve_bytes,
                "memory_per_game_bytes": chunk.host.memory_per_game_bytes,
                "output_dir": str(chunk.output_dir),
                "command": list(chunk.command),
            }
            for chunk in chunks
        ],
    }
    for final_dir in final_dirs.values():
        (final_dir / "launch_command.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {name: path / "arena_results.csv" for name, path in final_dirs.items()}
