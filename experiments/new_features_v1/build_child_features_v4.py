"""
Build a child-features cache (state-local v4 schema, 224-d) for one child-transitions
dataset directory.

Output layout (matches the older cached_v22_extra_400k template):
  <out_dir>/child_features.npy       (N x 224, float32)
  <out_dir>/child_meta.npy           (N x 5, float64)
                                     [parent_state_id, action_index, is_valid, reward, discount]
  <out_dir>/parent_index.npz         keys: order(N,), parent_ids(P,), offsets(P+1,),
                                           rewards(N,), discounts(N,), is_valid(N,)
  <out_dir>/child_features.summary.json

Where `order` is the index into child_features for each (parent_id-sorted) child row,
`offsets` carves the per-parent ranges, and `parent_ids` is the sorted unique parent id list.

Concurrency: shard-level multiprocessing. Each worker reads one .pt shard, calls
extract_features_one_record() on the child snapshot+stats wrapped in a record dict, and
returns (features, meta_rows). The main process concatenates in the original (task,
shard) order — i.e. parent-state-id ascending overall — so we can build offsets in one
pass.

Usage:
  python build_child_features_v4.py \
      --child-cache-dir simulator_output/GV3_Agent/.../child_transitions_recycled \
      --out-dir         simulator_output/GV3_Agent/BellmanConvergence/v4_child_d1 \
      --num-processes   48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.new_features_v1.build_state_local_features import (  # noqa: E402
    extract_features_one_record,
    D_TOTAL,
    feature_names,
)


def _process_shard(args: tuple) -> tuple:
    """
    args = (shard_path, expected_n_transitions)
    returns (features (n,224), meta (n,5))
      meta cols: [parent_state_id, action_index, is_valid, reward, discount]
    """
    import torch  # late import for worker
    sys.path.insert(0, str(REPO_ROOT))
    shard_path, expected_n = args
    transitions = torch.load(shard_path, weights_only=False)
    n = len(transitions)
    feats = np.zeros((n, D_TOTAL), dtype=np.float32)
    meta = np.zeros((n, 5), dtype=np.float64)
    for i, t in enumerate(transitions):
        rec = {
            "simulator_snapshot": t.get("child_simulator_snapshot") or {},
            "stats": t.get("child_stats"),
            "root_id": int(t.get("parent_state_id", -1)),
        }
        feats[i] = extract_features_one_record(rec)
        meta[i, 0] = float(t.get("parent_state_id", -1))
        meta[i, 1] = float(t.get("action_index", -1))
        meta[i, 2] = 1.0 if t.get("is_valid", True) else 0.0
        meta[i, 3] = float(t.get("reward", 0.0) or 0.0)
        meta[i, 4] = float(t.get("discount", 0.0) or 0.0)
    return feats, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-cache-dir", required=True,
                        help="dir containing manifest.jsonl + task_*/child_transitions_*.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=48)
    args = parser.parse_args()

    cache_dir = Path(args.child_cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = (cache_dir / "manifest.jsonl").read_text().splitlines()
    shard_paths: list[tuple[str, int]] = []
    total_transitions = 0
    for line in manifest:
        m = json.loads(line)
        shard_paths.append((str(cache_dir / m["shard_path"]), int(m["num_transitions"])))
        total_transitions += int(m["num_transitions"])

    print(f"[build-child] manifest_shards={len(shard_paths)} total_transitions={total_transitions}",
          flush=True)
    print(f"[build-child] D_TOTAL={D_TOTAL}", flush=True)

    # Allocate output arrays in advance and write in shard-order positions.
    features = np.zeros((total_transitions, D_TOTAL), dtype=np.float32)
    meta = np.zeros((total_transitions, 5), dtype=np.float64)

    cursor = 0
    shard_offsets = []
    for path, n in shard_paths:
        shard_offsets.append((path, cursor, n))
        cursor += n
    assert cursor == total_transitions

    started = time.time()
    completed = 0
    last_log = started
    with ProcessPoolExecutor(max_workers=args.num_processes) as ex:
        # submit in batches to keep memory bounded
        BATCH = max(args.num_processes * 2, 16)
        i = 0
        in_flight = {}
        while i < len(shard_offsets) or in_flight:
            while len(in_flight) < BATCH and i < len(shard_offsets):
                path, off, n = shard_offsets[i]
                fut = ex.submit(_process_shard, (path, n))
                in_flight[fut] = (off, n, path)
                i += 1
            done, _ = (set(), set())
            for fut in list(in_flight.keys()):
                if fut.done():
                    off, n, path = in_flight.pop(fut)
                    feats, m = fut.result()
                    if feats.shape[0] != n:
                        print(f"[build-child] WARN shard {path} returned {feats.shape[0]} != {n}",
                              flush=True)
                    features[off:off + feats.shape[0]] = feats
                    meta[off:off + m.shape[0]] = m
                    completed += 1
                    now = time.time()
                    if now - last_log > 5.0 or completed == len(shard_offsets):
                        rate = completed / max(1e-9, now - started)
                        eta = (len(shard_offsets) - completed) / max(1e-9, rate)
                        print(
                            f"[build-child] shards_done={completed}/{len(shard_offsets)} "
                            f"rate_shards_s={rate:.2f} eta_s={eta:.1f}",
                            flush=True,
                        )
                        last_log = now
            # very light idle; ProcessPool dispatches will continue
            if not any(f.done() for f in in_flight):
                time.sleep(0.05)

    # Build parent_index.npz
    parent_ids_arr = meta[:, 0].astype(np.int64)
    rewards = meta[:, 3].astype(np.float32)
    discounts = meta[:, 4].astype(np.float32)
    is_valid = meta[:, 2].astype(np.bool_)
    # Parents are not guaranteed sorted across shards (different tasks can re-emit), so do
    # a stable sort by parent_id.
    order = np.argsort(parent_ids_arr, kind="stable")
    sorted_pids = parent_ids_arr[order]
    unique_pids, idx_first = np.unique(sorted_pids, return_index=True)
    # offsets length = P+1
    offsets = np.empty(unique_pids.size + 1, dtype=np.int64)
    offsets[:-1] = idx_first
    offsets[-1] = sorted_pids.size

    np.save(out_dir / "child_features.npy", features)
    np.save(out_dir / "child_meta.npy", meta)
    np.savez(
        out_dir / "parent_index.npz",
        order=order,
        parent_ids=unique_pids,
        offsets=offsets,
        rewards=rewards,
        discounts=discounts,
        is_valid=is_valid,
    )
    (out_dir / "child_features.names.json").write_text(json.dumps(feature_names(), indent=2))
    summary = {
        "num_rows": int(total_transitions),
        "feature_dim": int(D_TOTAL),
        "num_parents": int(unique_pids.size),
        "child_cache_dir": str(cache_dir),
        "elapsed_s": time.time() - started,
    }
    (out_dir / "child_features.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[build-child] DONE in {time.time() - started:.1f}s", flush=True)
    print(f"[build-child] features.shape={features.shape} meta.shape={meta.shape} parents={unique_pids.size}",
          flush=True)


if __name__ == "__main__":
    main()
