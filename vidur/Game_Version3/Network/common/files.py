from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_relative_files(root: Path) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    out: list[str] = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            out.append(str(p.relative_to(base)))
    return out


def rewrite_manifest_paths_absolute(dataset_partition_dir: Path) -> int:
    """
    Rewrite ReplayWriter manifests under a train/eval partition so shard paths point
    at the local pulled files. This keeps existing ReplayBuffer code unchanged.
    """
    root = Path(dataset_partition_dir)
    rewritten = 0
    for manifest in sorted(root.glob("proc_*/manifest.jsonl")):
        lines: list[str] = []
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            entry = json.loads(raw)
            shard_name = Path(str(entry["path"])).name
            entry["path"] = str((manifest.parent / shard_name).resolve())
            lines.append(json.dumps(entry, ensure_ascii=False))
        manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rewritten += 1
    return rewritten


def count_partition_samples(dataset_partition_dir: Path) -> dict[str, int]:
    stats = {
        "samples_total": 0,
        "controller_samples": 0,
        "adversary_samples": 0,
        "num_shards": 0,
        "num_manifests": 0,
    }
    root = Path(dataset_partition_dir)
    for manifest in sorted(root.glob("proc_*/manifest.jsonl")):
        stats["num_manifests"] += 1
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            entry = json.loads(raw)
            shard_path = Path(str(entry["path"]))
            if not shard_path.is_absolute():
                shard_path = manifest.parent / shard_path
            shard = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard, list):
                raise TypeError(f"Expected list shard at {shard_path}, got {type(shard)}")
            stats["num_shards"] += 1
            stats["samples_total"] += len(shard)
            for sample in shard:
                player = str(sample.get("player", ""))
                if player == "controller":
                    stats["controller_samples"] += 1
                elif player == "adversary":
                    stats["adversary_samples"] += 1
    return stats

