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
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.new_features_v1.build_state_local_features import (  # noqa: E402
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


def _write_parent_index(out_dir: Path, meta: np.ndarray) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Adv child-feature cache.")
    parser.add_argument("--child-cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=48)
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

    features = np.zeros((total_transitions, D_TOTAL), dtype=np.float32)
    meta = np.zeros((total_transitions, len(META_COLUMNS)), dtype=np.float64)

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
    batch_size = max(max_workers * 2, 16)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
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
                features[off : off + feats.shape[0]] = feats
                meta[off : off + m.shape[0]] = m
                completed += 1
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

    np.save(out_dir / "child_features.npy", features)
    np.save(out_dir / "child_meta.npy", meta)
    _write_parent_index(out_dir, meta)
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
        "num_parents": int(np.unique(meta[:, 0].astype(np.int64)).size),
        "child_cache_dir": str(cache_dir),
        "elapsed_s": float(time.time() - started),
    }
    (out_dir / "child_features.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[build-child-adv] DONE in {time.time() - started:.1f}s "
        f"features.shape={features.shape} meta.shape={meta.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
