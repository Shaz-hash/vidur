"""Count duplicate history signatures in a ModelSearchBed root store.

This script is safe to run while generation is still in progress. It streams
one shard at a time from the final manifest and/or worker manifests, then counts
duplicate `history_signature` values without loading all root records at once.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_dataset_dir() -> Path:
    return (
        _repo_root()
        / "simulator_output"
        / "GV3_Agent"
        / "model_search_roots_controller_350k_abs1_ratio40"
    )


def _normalize_signature(value: Any) -> Any:
    """Convert a signature into a deterministic hashable key."""

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return tuple(
            (str(k), _normalize_signature(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_signature(x) for x in value)
    if isinstance(value, set):
        return tuple(sorted(_normalize_signature(x) for x in value))
    return repr(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, set)):
        return [_jsonable(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def find_manifest_paths(dataset_dir: Path, *, include_worker_parts: bool) -> list[Path]:
    """Return manifests to scan.

    During generation, only `_worker_parts/worker_*/manifest.jsonl` exists.
    After generation, the final top-level `manifest.jsonl` exists.
    """

    manifests: list[Path] = []
    final_manifest = dataset_dir / "manifest.jsonl"
    if final_manifest.exists():
        manifests.append(final_manifest)

    if include_worker_parts:
        worker_parent = dataset_dir / "_worker_parts"
        if worker_parent.exists():
            manifests.extend(sorted(worker_parent.glob("worker_*/manifest.jsonl")))

    return manifests


def iter_shard_paths(manifest_paths: list[Path]) -> Iterator[Path]:
    """Yield shard paths referenced by manifest rows."""

    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                shard_path = manifest_path.parent / str(entry["shard_path"])
                if shard_path.exists():
                    yield shard_path


def load_shard(path: Path) -> list[dict[str, Any]]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def count_duplicate_signatures(
    dataset_dir: Path,
    *,
    include_worker_parts: bool = True,
    top_k: int = 10,
) -> dict[str, Any]:
    """Count duplicate `history_signature` values across stored roots."""

    manifest_paths = find_manifest_paths(dataset_dir, include_worker_parts=include_worker_parts)
    if not manifest_paths:
        raise FileNotFoundError(f"no manifest.jsonl files found under {dataset_dir}")

    counts: Counter[Any] = Counter()
    sample_root_ids: dict[Any, list[int]] = {}
    total_records = 0
    missing_signature_records = 0
    shards_scanned = 0

    for shard_path in iter_shard_paths(manifest_paths):
        records = load_shard(shard_path)
        shards_scanned += 1
        for record in records:
            total_records += 1
            signature = record.get("history_signature", None)
            if signature is None:
                missing_signature_records += 1
                continue
            key = _normalize_signature(signature)
            counts[key] += 1
            if len(sample_root_ids.get(key, [])) < 5:
                sample_root_ids.setdefault(key, []).append(int(record.get("root_id", -1)))

    duplicate_items = [(key, count) for key, count in counts.items() if int(count) > 1]
    duplicate_records = sum(int(count) - 1 for _key, count in duplicate_items)
    duplicate_groups = len(duplicate_items)
    unique_signatures = len(counts)

    top_duplicates = []
    for key, count in sorted(duplicate_items, key=lambda item: int(item[1]), reverse=True)[: int(top_k)]:
        top_duplicates.append(
            {
                "count": int(count),
                "sample_root_ids": sample_root_ids.get(key, []),
                "history_signature": _jsonable(key),
            }
        )

    return {
        "dataset_dir": str(dataset_dir),
        "manifests_scanned": [str(path) for path in manifest_paths],
        "shards_scanned": int(shards_scanned),
        "total_records": int(total_records),
        "records_with_signature": int(total_records - missing_signature_records),
        "missing_signature_records": int(missing_signature_records),
        "unique_signatures": int(unique_signatures),
        "duplicate_groups": int(duplicate_groups),
        "duplicate_records": int(duplicate_records),
        "duplicate_record_ratio": (
            float(duplicate_records) / float(total_records - missing_signature_records)
            if int(total_records - missing_signature_records) > 0
            else 0.0
        ),
        "top_duplicates": top_duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count duplicate ModelSearchBed root signatures.")
    parser.add_argument("--dataset-dir", default=str(_default_dataset_dir()))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-worker-parts",
        action="store_true",
        help="Only scan the final top-level manifest, not in-progress worker manifests.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write duplicate_signature_report.json under the dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    report = count_duplicate_signatures(
        dataset_dir,
        include_worker_parts=not bool(args.no_worker_parts),
        top_k=int(args.top_k),
    )

    if bool(args.write_report):
        path = dataset_dir / "duplicate_signature_report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(path)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
