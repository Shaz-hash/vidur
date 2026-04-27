from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable

import torch


LogLine = Callable[[str], None]

_CANONICAL_PROC_RE = re.compile(
    r"^proc_net_(?P<task_id>.+)_(?P<proc_suffix>\d+_cycle_\d+_task_\d+)$"
)
_SESSION_FROM_TASK_RE = re.compile(r"^(?P<session_id>.+)_c\d{3}_.+_\d{3}$")


def _load_replay_shard(path: Path) -> int:
    shard = torch.load(path, map_location="cpu")
    if not isinstance(shard, list):
        raise TypeError(f"Expected replay shard list at {path}, got {type(shard)}")
    return int(len(shard))


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def validate_or_repair_shard(*, src: Path | None, dst: Path) -> dict[str, int]:
    """
    Validate a canonical replay shard. If it is corrupt and a valid source shard is
    available, recopy source -> destination with an atomic rename and validate again.
    """
    dst = Path(dst)
    src = Path(src) if src is not None else None
    try:
        samples = _load_replay_shard(dst)
        return {"validated_shards": 1, "repaired_shards": 0, "samples": int(samples)}
    except Exception as dst_exc:
        if src is None or not src.exists():
            raise RuntimeError(f"Corrupt replay shard and no repair source: {dst}: {dst_exc}") from dst_exc

        try:
            _load_replay_shard(src)
        except Exception as src_exc:
            raise RuntimeError(
                f"Corrupt canonical replay shard and corrupt repair source: "
                f"dst={dst}: {dst_exc}; src={src}: {src_exc}"
            ) from src_exc

        _atomic_copy_file(src, dst)
        try:
            samples = _load_replay_shard(dst)
        except Exception as repaired_exc:
            raise RuntimeError(
                f"Replay shard still corrupt after repair copy: dst={dst}, src={src}: {repaired_exc}"
            ) from repaired_exc
        return {"validated_shards": 1, "repaired_shards": 1, "samples": int(samples)}


def validate_or_repair_proc_dir(*, src_proc_dir: Path | None, dst_proc_dir: Path) -> dict[str, int]:
    src_proc_dir = Path(src_proc_dir) if src_proc_dir is not None else None
    dst_proc_dir = Path(dst_proc_dir)
    stats = {"validated_shards": 0, "repaired_shards": 0, "samples": 0}
    for dst_shard in sorted(dst_proc_dir.glob("*.pt")):
        src_shard = src_proc_dir / dst_shard.name if src_proc_dir is not None else None
        shard_stats = validate_or_repair_shard(src=src_shard, dst=dst_shard)
        for key in stats:
            stats[key] += int(shard_stats.get(key, 0))
    return stats


def _source_proc_from_received(
    *,
    canonical_proc_dir: Path,
    received_root: Path,
    partition: str,
) -> Path | None:
    match = _CANONICAL_PROC_RE.match(canonical_proc_dir.name)
    if not match:
        return None

    task_id = match.group("task_id")
    proc_suffix = match.group("proc_suffix")
    session_match = _SESSION_FROM_TASK_RE.match(task_id)
    candidates: list[Path] = []
    if session_match:
        session_id = session_match.group("session_id")
        candidates.append(received_root / session_id / task_id / partition / f"proc_{proc_suffix}")

    candidates.extend(sorted(received_root.glob(f"*/{task_id}/{partition}/proc_{proc_suffix}")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def validate_or_repair_generation_from_received(
    *,
    generation: int,
    dataset_dir: Path,
    received_root: Path,
    log_line: LogLine | None = None,
) -> dict[str, int]:
    """
    Validate canonical generation shards and repair any corrupt shard from
    network/received when that source copy is available.
    """
    gen_dir = Path(dataset_dir) / f"gen_{int(generation):06d}"
    received_root = Path(received_root)
    stats = {"validated_shards": 0, "repaired_shards": 0, "samples": 0}
    for partition in ("train", "eval"):
        partition_dir = gen_dir / partition
        if not partition_dir.exists():
            continue
        for canonical_proc in sorted(partition_dir.glob("proc_net_*")):
            if not canonical_proc.is_dir():
                continue
            source_proc = _source_proc_from_received(
                canonical_proc_dir=canonical_proc,
                received_root=received_root,
                partition=partition,
            )
            proc_stats = validate_or_repair_proc_dir(src_proc_dir=source_proc, dst_proc_dir=canonical_proc)
            for key in stats:
                stats[key] += int(proc_stats.get(key, 0))
            if log_line is not None and int(proc_stats.get("repaired_shards", 0)) > 0:
                log_line(
                    f"[GV3 gen={int(generation):06d}] repaired replay shards: "
                    f"partition={partition}, proc={canonical_proc.name}, "
                    f"count={int(proc_stats['repaired_shards'])}"
                )
    return stats
