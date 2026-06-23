"""Action-conditional Q-learning driver using the cached child cache.

Architecture:
- Per-row features: cliff(parent) + action_aggregate(parent) + forecast(parent) + per_action(s, a).
  Parent features are looked up via parent_state_id → parent feature row.
  Per-action features are parsed from action_repr per row (cheap).
- Per-row target: r + γ * V_{N-1}(s'_child).
  V_{N-1}(s'_child) comes from the existing child feature memmap + a state-only
  V model trained alongside Q.
- Q model: HistGradientBoosting (handles 14M rows well).
- V model (state-only, used for next-iter bootstrap and arena inference):
  trained on parent rows with target = max_a Q_N(s, a).

Outputs match the cached driver (per-version Model_Version{N}/ folder with
joblib + train/eval CSVs).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


from . import logger as bellman_logger
from .self_model_test import (
    RootStateLoader,
    SelfModelTestConfig,
    build_self_model_test_config,
    load_root_records,
)
from .per_action_features import (
    extract_per_action_features,
    per_action_feature_names,
)


def _limit_native_threads(num_threads: int) -> None:
    threads = max(1, int(num_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


@dataclass(frozen=True)
class QLearningConfig:
    dataset_dir: Path
    child_cache_dir: Path
    parent_features_path: Path  # 498-dim parent features memmap
    child_features_path: Path   # 498-dim child features memmap (used as s' for bootstrap)
    output_dir: Path
    num_versions: int = 25
    eval_ratio: float = 0.20
    split_seed: int = 12345
    seed: int = 2027
    abs_error_threshold: float = 1.0
    extra_config: dict[str, Any] | None = None
    # Q model hparams
    q_hgb_max_iter: int = 400
    q_hgb_max_leaf_nodes: int = 31
    q_hgb_learning_rate: float = 0.05
    # V distillation hparams
    v_hgb_max_iter: int = 250
    v_hgb_max_leaf_nodes: int = 21
    v_hgb_learning_rate: float = 0.05
    # Subsampling
    q_subsample_rows: int | None = None  # None = use all rows
    target_chunk_size: int = 1_000_000


def _read_meta_and_actions(child_cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Walk child cache shards and gather (parent_state_id, canonical_action_index) per row.

    We also parse action_repr per row to build a per-row action-features array.
    Returns (per_action_features array shape (N, D_pa), action_idx array shape (N,)).
    """
    raise NotImplementedError("use precompute_per_row_actions to build the on-disk caches")


def precompute_per_row_action_features(
    *,
    child_cache_dir: Path,
    output_dir: Path,
    num_workers: int = 16,
) -> Path:
    """Walk every shard, parse action_repr, write per_row_action.npy memmap.

    Output shape: (num_canonical_transitions, len(per_action_feature_names())).
    """

    out_path = output_dir / "per_row_action.npy"
    summary_path = output_dir / "per_row_action.summary.json"
    if out_path.exists() and summary_path.exists():
        print(f"[qlearn] per-row action features already present: {out_path}", flush=True)
        return out_path

    import torch

    manifest = child_cache_dir / "manifest.jsonl"
    shards: list[str] = []
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            shards.append(str(entry["shard_path"]))

    summary = json.loads((child_cache_dir / "summary.json").read_text())
    n_total = int(summary.get("num_canonical_transitions", summary.get("num_transitions", 0)))
    if n_total <= 0:
        raise RuntimeError("could not determine row count from summary.json")
    D = len(per_action_feature_names())
    print(f"[qlearn] allocating per-row action memmap shape=({n_total}, {D})", flush=True)
    capacity = int(n_total * 1.02) + 1024
    mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(capacity, D)
    )

    cursor = 0
    bad: list[tuple[str, str]] = []
    t0 = time.time()
    for i, shard in enumerate(shards):
        full = child_cache_dir / shard
        try:
            recs = torch.load(full, map_location="cpu", weights_only=False)
        except Exception as exc:
            bad.append((shard, repr(exc)))
            continue
        block: list[list[float]] = []
        for r in recs:
            if not bool(r.get("is_valid", False)):
                continue
            try:
                feats = extract_per_action_features(str(r.get("action_repr", "")))
            except Exception:
                feats = [0.0] * D
            block.append(feats)
        if not block:
            continue
        arr = np.asarray(block, dtype=np.float32)
        end = cursor + arr.shape[0]
        if end > capacity:
            raise RuntimeError(f"per_row_action capacity overflow {end}>{capacity}")
        mm[cursor:end] = arr
        cursor = end
        if (i + 1) % 200 == 0 or (i + 1) == len(shards):
            rate = cursor / max(1.0, time.time() - t0)
            print(
                f"[qlearn] per-row actions: shard {i+1}/{len(shards)}, "
                f"rows={cursor}/{n_total}, rate={rate:,.0f} rows/s, bad={len(bad)}",
                flush=True,
            )

    # Trim if needed
    if cursor < capacity:
        del mm
        full_arr = np.load(out_path, mmap_mode="r")
        trim = np.lib.format.open_memmap(
            out_path.with_suffix(".tmp.npy"),
            mode="w+", dtype=np.float32, shape=(cursor, D),
        )
        trim[:] = full_arr[:cursor]
        del trim
        del full_arr
        out_path.with_suffix(".tmp.npy").replace(out_path)

    summary_path.write_text(json.dumps({
        "num_rows": int(cursor),
        "feature_dim": int(D),
        "feature_names": per_action_feature_names(),
        "bad_shards": bad[:64],
    }, indent=2))
    print(f"[qlearn] per-row action features done: rows={cursor}, D={D}", flush=True)
    return out_path


# (driver main loop will go here once the per-row action memmap exists; this
# module only provides the precompute helper for now to keep the change
# focused.)
