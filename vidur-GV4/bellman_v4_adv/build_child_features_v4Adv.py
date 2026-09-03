#!/usr/bin/env python3
"""
Build a child-features cache for Adv controller->adversary child transitions.

The feature vector is still the final child controller state. The transition
metadata now carries both decision edges:

  meta columns:
    [parent_state_id,
     controller_action_index,
     controller_canonical_action_index,
     adversary_action_index,
     is_valid,
     reward,
     discount,
     child_time]

`reward` and `discount` are the controller edge parent -> intermediate. The
adversary action changes the final child state used for bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vidur.bellman_v4_adv.build_state_local_features_adv import (  # noqa: E402
    D_TOTAL,
    extract_features_one_record,
    feature_names,
)

SCHEMA_VERSION = 2
TRANSITION_TYPE = "controller_then_adversary"
META_COLUMNS = [
    "parent_state_id",
    "controller_action_index",
    "controller_canonical_action_index",
    "adversary_action_index",
    "is_valid",
    "reward",
    "discount",
    "child_time",
]


def _warm_worker() -> int:
    """Start worker processes before the parent opens large output memmaps."""
    sys.path.insert(0, str(REPO_ROOT))
    return 1

def _process_shard(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load one child transition shard and return feature/meta arrays."""

    import torch  # late import for worker isolation

    sys.path.insert(0, str(REPO_ROOT))
    shard_path, expected_n = args
    transitions = torch.load(shard_path, map_location="cpu", weights_only=False)
    n = len(transitions)
    if int(expected_n) >= 0 and n != int(expected_n):
        print(
            f"[build-child-adv] WARN manifest expected {expected_n} rows but "
            f"{shard_path} loaded {n}",
            flush=True,
        )

    feats = np.zeros((n, D_TOTAL), dtype=np.float32)
    meta = np.zeros((n, len(META_COLUMNS)), dtype=np.float64)
    for i, t in enumerate(transitions):
        rec = {
            "simulator_snapshot": t.get("child_simulator_snapshot") or {},
            "stats": t.get("child_stats"),
            "root_id": int(t.get("parent_state_id", -1)),
        }
        feats[i] = extract_features_one_record(rec)

        controller_action_index = int(t.get("controller_action_index", t.get("action_index", -1)))
        controller_canonical_action_index = int(
            t.get(
                "controller_canonical_action_index",
                t.get("canonical_action_index", controller_action_index),
            )
        )
        meta[i, 0] = float(t.get("parent_state_id", -1))
        meta[i, 1] = float(controller_action_index)
        meta[i, 2] = float(controller_canonical_action_index)
        meta[i, 3] = float(t.get("adversary_action_index", -1))
        meta[i, 4] = 1.0 if t.get("is_valid", True) else 0.0
        meta[i, 5] = float(t.get("controller_reward", t.get("reward", 0.0)) or 0.0)
        meta[i, 6] = float(t.get("controller_discount", t.get("discount", 0.0)) or 0.0)
        meta[i, 7] = float(t.get("child_time", 0.0) or 0.0)
    return feats, meta


def _read_manifest(cache_dir: Path) -> list[tuple[str, int]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"child cache manifest not found: {manifest_path}")

    shard_paths: list[tuple[str, int]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        shard_paths.append((str(cache_dir / row["shard_path"]), int(row.get("num_transitions", -1))))
    if not shard_paths:
        raise RuntimeError(f"empty child cache manifest: {manifest_path}")
    return shard_paths


def _create_npy_for_pwrite(path: Path, *, dtype: np.dtype, shape: tuple[int, ...]) -> tuple[int, int]:
    """Create a valid sparse .npy file and return (fd, data_offset)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    header = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": tuple(int(x) for x in shape),
    }
    with path.open("wb") as f:
        np.lib.format.write_array_header_2_0(f, header)
        data_offset = int(f.tell())
        total_items = 1
        for dim in shape:
            total_items *= int(dim)
        f.truncate(data_offset + total_items * dtype.itemsize)
    return os.open(path, os.O_RDWR), data_offset


def _pwrite_all(fd: int, data: memoryview, offset: int) -> None:
    total = 0
    n = len(data)
    while total < n:
        written = os.pwrite(fd, data[total:], offset + total)
        if written <= 0:
            raise OSError(f"pwrite wrote {written} bytes at offset {offset + total}")
        total += written


def _pwrite_rows(
    *,
    fd: int,
    data_offset: int,
    row_start: int,
    arr: np.ndarray,
    dtype: np.dtype,
    n_cols: int,
) -> tuple[int, int]:
    arr = np.ascontiguousarray(arr, dtype=dtype)
    byte_offset = int(data_offset + int(row_start) * int(n_cols) * np.dtype(dtype).itemsize)
    view = memoryview(arr).cast("B")
    _pwrite_all(fd, view, byte_offset)
    return byte_offset, len(view)


def _sync_and_drop(fd: int, start: int | None, end: int | None) -> None:
    if start is None or end is None or end <= start:
        return
    os.fdatasync(fd)
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, int(start), int(end - start), os.POSIX_FADV_DONTNEED)
        except OSError:
            # Some filesystems may reject fadvise for sparse or network-backed files.
            pass


def _range_add(
    cur_start: int | None,
    cur_end: int | None,
    start: int,
    nbytes: int,
) -> tuple[int, int]:
    end = int(start + nbytes)
    if cur_start is None or cur_end is None:
        return int(start), end
    return min(cur_start, int(start)), max(cur_end, end)


def _write_parent_index(out_dir: Path, meta: np.ndarray) -> int:
    parent_ids_arr = meta[:, 0].astype(np.int64)
    controller_action_indices = meta[:, 1].astype(np.int64)
    controller_canonical_action_indices = meta[:, 2].astype(np.int64)
    adversary_action_indices = meta[:, 3].astype(np.int64)
    is_valid = meta[:, 4].astype(np.bool_)
    rewards = meta[:, 5].astype(np.float32)
    discounts = meta[:, 6].astype(np.float32)
    child_times = meta[:, 7].astype(np.float32)

    order = np.argsort(parent_ids_arr, kind="stable")
    sorted_pids = parent_ids_arr[order]
    unique_pids, idx_first = np.unique(sorted_pids, return_index=True)
    offsets = np.empty(unique_pids.size + 1, dtype=np.int64)
    offsets[:-1] = idx_first
    offsets[-1] = sorted_pids.size

    np.savez(
        out_dir / "parent_index.npz",
        order=order,
        parent_ids=unique_pids,
        offsets=offsets,
        rewards=rewards,
        discounts=discounts,
        is_valid=is_valid,
        controller_action_indices=controller_action_indices,
        controller_canonical_action_indices=controller_canonical_action_indices,
        adversary_action_indices=adversary_action_indices,
        child_times=child_times,
    )
    return int(unique_pids.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Adv child-feature cache.")
    parser.add_argument("--child-cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=48)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Maximum submitted shards held in-flight. 0 = 2 * num_processes.",
    )
    parser.add_argument(
        "--flush-every-shards",
        type=int,
        default=256,
        help="Flush memmap outputs after this many completed shards. 0 disables periodic flush.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.child_cache_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = _read_manifest(cache_dir)
    total_transitions = sum(max(0, int(n)) for _path, n in shard_paths)
    if total_transitions <= 0:
        raise RuntimeError(f"manifest has no transitions: {cache_dir / 'manifest.jsonl'}")

    print(
        f"[build-child-adv] manifest_shards={len(shard_paths)} "
        f"total_transitions={total_transitions} D_TOTAL={D_TOTAL}",
        flush=True,
    )

    features_path = out_dir / "child_features.npy"
    meta_path = out_dir / "child_meta.npy"
    for stale_path in (features_path, meta_path, out_dir / "parent_index.npz"):
        if stale_path.exists():
            stale_path.unlink()

    feature_shape = (total_transitions, D_TOTAL)
    meta_shape = (total_transitions, len(META_COLUMNS))

    cursor = 0
    shard_offsets: list[tuple[str, int, int]] = []
    for path, n in shard_paths:
        shard_offsets.append((path, cursor, int(n)))
        cursor += int(n)
    if cursor != total_transitions:
        raise AssertionError("internal cursor mismatch")

    started = time.time()
    completed = 0
    last_log = started
    max_workers = max(1, int(args.num_processes))
    batch_size = int(args.max_in_flight) if int(args.max_in_flight) > 0 else max(max_workers * 2, 16)
    flush_every = max(0, int(args.flush_every_shards))
    print(
        f"[build-child-adv] output_mode=pwrite_npy batch_size={batch_size} "
        f"flush_every_shards={flush_every}",
        flush=True,
    )

    feature_fd: int | None = None
    meta_fd: int | None = None
    pending_feature_start: int | None = None
    pending_feature_end: int | None = None
    pending_meta_start: int | None = None
    pending_meta_end: int | None = None

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Force the forked worker pool to exist before opening huge output files.
        # Otherwise workers inherit those file descriptors and mappings/accounting
        # gets harder to reason about under high parallelism.
        for fut in [ex.submit(_warm_worker) for _ in range(max_workers)]:
            fut.result()

        feature_fd, feature_data_offset = _create_npy_for_pwrite(
            features_path, dtype=np.float32, shape=feature_shape
        )
        meta_fd, meta_data_offset = _create_npy_for_pwrite(
            meta_path, dtype=np.float64, shape=meta_shape
        )

        i = 0
        in_flight = {}
        while i < len(shard_offsets) or in_flight:
            while len(in_flight) < batch_size and i < len(shard_offsets):
                path, off, n = shard_offsets[i]
                fut = ex.submit(_process_shard, (path, n))
                in_flight[fut] = (off, n, path)
                i += 1

            progressed = False
            for fut in list(in_flight.keys()):
                if not fut.done():
                    continue
                off, n, path = in_flight.pop(fut)
                feats, m = fut.result()
                if feats.shape[0] != n:
                    print(
                        f"[build-child-adv] WARN shard {path} returned "
                        f"{feats.shape[0]} != {n}",
                        flush=True,
                    )
                feature_start, feature_nbytes = _pwrite_rows(
                    fd=feature_fd,
                    data_offset=feature_data_offset,
                    row_start=off,
                    arr=feats,
                    dtype=np.float32,
                    n_cols=D_TOTAL,
                )
                meta_start, meta_nbytes = _pwrite_rows(
                    fd=meta_fd,
                    data_offset=meta_data_offset,
                    row_start=off,
                    arr=m,
                    dtype=np.float64,
                    n_cols=len(META_COLUMNS),
                )
                pending_feature_start, pending_feature_end = _range_add(
                    pending_feature_start, pending_feature_end, feature_start, feature_nbytes
                )
                pending_meta_start, pending_meta_end = _range_add(
                    pending_meta_start, pending_meta_end, meta_start, meta_nbytes
                )
                completed += 1
                if flush_every and completed % flush_every == 0:
                    _sync_and_drop(feature_fd, pending_feature_start, pending_feature_end)
                    _sync_and_drop(meta_fd, pending_meta_start, pending_meta_end)
                    pending_feature_start = pending_feature_end = None
                    pending_meta_start = pending_meta_end = None
                progressed = True

                now = time.time()
                if now - last_log > 5.0 or completed == len(shard_offsets):
                    rate = completed / max(1e-9, now - started)
                    eta = (len(shard_offsets) - completed) / max(1e-9, rate)
                    print(
                        f"[build-child-adv] shards_done={completed}/{len(shard_offsets)} "
                        f"rate_shards_s={rate:.2f} eta_s={eta:.1f}",
                        flush=True,
                    )
                    last_log = now
            if not progressed:
                time.sleep(0.05)

        _sync_and_drop(feature_fd, pending_feature_start, pending_feature_end)
        _sync_and_drop(meta_fd, pending_meta_start, pending_meta_end)

    if feature_fd is not None:
        os.close(feature_fd)
    if meta_fd is not None:
        os.close(meta_fd)

    meta_for_index = np.load(meta_path, mmap_mode="r")
    num_parents = _write_parent_index(out_dir, meta_for_index)
    del meta_for_index
    (out_dir / "child_features.names.json").write_text(
        json.dumps(feature_names(), indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "transition_type": TRANSITION_TYPE,
        "num_rows": int(total_transitions),
        "feature_dim": int(D_TOTAL),
        "meta_columns": META_COLUMNS,
        "num_parents": int(num_parents),
        "child_cache_dir": str(cache_dir),
        "elapsed_s": float(time.time() - started),
    }
    (out_dir / "child_features.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[build-child-adv] DONE in {time.time() - started:.1f}s "
        f"features.shape={feature_shape} meta.shape={meta_shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
