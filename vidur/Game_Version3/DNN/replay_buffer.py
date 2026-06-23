# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)



from __future__ import annotations

import random
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List
import re
import torch

from .replay_dataset import load_manifest
from .replay_write import RootSample


ShardRepairFn = Callable[[Path, Exception], bool]


@dataclass
class _ShardWindow:
    path: Path
    num_samples: int
    drop_prefix: int = 0

    @property
    def effective_size(self) -> int:
        return max(0, int(self.num_samples) - int(self.drop_prefix))


_GEN_RE = re.compile(r"gen_(\d+)")

class BestModelReplayBuffer:
    """
    Sliding-window replay over generated samples.
    Keeps only the newest `capacity_samples` entries in creation order.
    """

    def __init__(
        self,
        *,
        capacity_samples: int = 12_000,
        max_cached_shards: int = 8,
        seed: int = 0,
        repair_shard_after_load_error: ShardRepairFn | None = None,
    ) -> None:
        self.capacity_samples = int(capacity_samples)
        self.max_cached_shards = int(max_cached_shards)
        self._rng = random.Random(int(seed))
        self._repair_shard_after_load_error = repair_shard_after_load_error

        self._windows: Deque[_ShardWindow] = deque()
        self._total_samples: int = 0
        self._seen_paths: set[Path] = set()

        # LRU cache: shard_path -> list[RootSample]
        self._cache: "OrderedDict[Path, List[RootSample]]" = OrderedDict()
        self._weights: List[int] = []

    @property
    def total_samples(self) -> int:
        return int(self._total_samples)

    @property
    def num_shards(self) -> int:
        return int(len(self._windows))

    def active_generation_ids(self) -> list[int]:
        gens: set[int] = set()
        for window in self._windows:
            m = _GEN_RE.search(str(window.path))
            if m:
                gens.add(int(m.group(1)))
        return sorted(gens)

    def active_generation_ids_csv(self, *, max_items: int = 256) -> str:
        gens = self.active_generation_ids()
        if len(gens) <= max_items:
            return ",".join(f"{g:06d}" for g in gens)
        kept = gens[:max_items]
        return ",".join(f"{g:06d}" for g in kept) + f",...(+{len(gens)-max_items})"


    def reseed(self, seed: int) -> None:
        self._rng.seed(int(seed))

    def reset_for_new_best(self) -> None:
        # Optional hard reset hook; default training flow now keeps replay across best promotions.
        self._windows.clear()
        self._total_samples = 0
        self._seen_paths.clear()
        self._cache.clear()
        self._weights = []

    def add_generation_dir(self, gen_dataset_dir: Path) -> int:
        """
        Adds all replay shards from gen_dataset_dir/proc_*/manifest.jsonl.
        Returns raw samples added before eviction.
        """
        gen_dataset_dir = Path(gen_dataset_dir)
        raw_added = 0

        for proc_dir in sorted(gen_dataset_dir.glob("proc_*")):
            manifest = proc_dir / "manifest.jsonl"
            if not manifest.exists():
                continue

            for entry in load_manifest(manifest, allow_empty=True):
                raw_added += self._append_shard(
                    path=self._canonical_path(entry.path),
                    num_samples=int(entry.num_samples),
                )

        if raw_added > 0:
            self._evict_to_capacity()
            self._refresh_weights()

        return int(raw_added)

    def sample_batch(self, batch_size: int) -> List[RootSample]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []
        if self._total_samples <= 0 or not self._windows:
            raise RuntimeError("Replay buffer is empty; add generation shards before sampling")
        if not self._weights or max(self._weights) <= 0:
            raise RuntimeError("Replay buffer has no effective shard weights to sample from")

        windows_list = list(self._windows)
        chosen_indices = self._rng.choices(
            range(len(windows_list)),
            weights=self._weights,
            k=batch_size,
        )

        index_counts: dict[int, int] = {}
        for idx in chosen_indices:
            index_counts[idx] = int(index_counts.get(idx, 0)) + 1

        sampled_by_index: dict[int, list[RootSample]] = {}
        for idx, count in index_counts.items():
            window = windows_list[int(idx)]
            shard = self._load_shard(window.path)
            upper = min(len(shard), int(window.num_samples))
            low = int(window.drop_prefix)
            if upper <= low:
                raise RuntimeError(
                    f"Replay sampling encountered empty effective shard window: "
                    f"path={window.path}, low={low}, upper={upper}"
                )

            picks: list[RootSample] = []
            for _ in range(int(count)):
                sample_idx = self._rng.randrange(low, upper)
                item = shard[sample_idx]
                if not isinstance(item, dict):
                    raise TypeError(f"Unexpected sample type {type(item)} in shard {window.path}")
                picks.append(item)
            sampled_by_index[int(idx)] = picks

        offsets: dict[int, int] = {int(idx): 0 for idx in index_counts.keys()}
        out: List[RootSample] = []
        for idx in chosen_indices:
            idx = int(idx)
            pos = int(offsets[idx])
            out.append(sampled_by_index[idx][pos])
            offsets[idx] = pos + 1

        return out

    def preload_all_shards(self, *, max_shards: int) -> dict[str, int]:
        limit = max(1, int(max_shards))
        windows_list = list(self._windows)
        if len(windows_list) > limit:
            return {
                "loaded_shards": 0,
                "cached_shards": int(len(self._cache)),
                "num_shards": int(len(windows_list)),
            }

        if len(windows_list) > self.max_cached_shards:
            self.max_cached_shards = int(len(windows_list))

        loaded = 0
        for window in windows_list:
            if window.path not in self._cache:
                self._load_shard(window.path)
                loaded += 1

        return {
            "loaded_shards": int(loaded),
            "cached_shards": int(len(self._cache)),
            "num_shards": int(len(windows_list)),
        }

    def _append_shard(self, *, path: Path, num_samples: int) -> int:
        if num_samples <= 0:
            return 0
        if path in self._seen_paths:
            return 0

        self._windows.append(_ShardWindow(path=path, num_samples=int(num_samples), drop_prefix=0))
        self._seen_paths.add(path)
        self._total_samples += int(num_samples)
        return int(num_samples)

    def _evict_to_capacity(self) -> None:
        overflow = self._total_samples - self.capacity_samples
        while overflow > 0 and self._windows:
            oldest = self._windows[0]
            can_drop = oldest.effective_size
            if can_drop <= 0:
                self._drop_oldest_window()
                continue

            drop_now = min(overflow, can_drop)
            oldest.drop_prefix += int(drop_now)
            self._total_samples -= int(drop_now)
            overflow -= int(drop_now)

            if oldest.effective_size <= 0:
                self._drop_oldest_window()

    def _drop_oldest_window(self) -> None:
        oldest = self._windows.popleft()
        self._seen_paths.discard(oldest.path)
        self._cache.pop(oldest.path, None)

    def _refresh_weights(self) -> None:
        self._weights = [max(0, w.effective_size) for w in self._windows]

        if any(w <= 0 for w in self._weights):
            kept: Deque[_ShardWindow] = deque()
            for w in self._windows:
                if w.effective_size > 0:
                    kept.append(w)
                else:
                    self._seen_paths.discard(w.path)
                    self._cache.pop(w.path, None)
            self._windows = kept
            self._weights = [w.effective_size for w in self._windows]

    def _sample_from_window(self, window: _ShardWindow) -> RootSample | None:
        shard = self._load_shard(window.path)
        upper = min(len(shard), int(window.num_samples))
        low = int(window.drop_prefix)

        if upper <= low:
            return None

        idx = self._rng.randrange(low, upper)
        item = shard[idx]
        if not isinstance(item, dict):
            raise TypeError(f"Unexpected sample type {type(item)} in shard {window.path}")
        return item

    def _load_shard(self, path: Path) -> List[RootSample]:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path, last=True)
            return cached

        try:
            shard = self._load_shard_uncached(path)
        except Exception as first_exc:
            repair_fn = self._repair_shard_after_load_error
            if repair_fn is None:
                raise first_exc

            try:
                repaired_or_validated = bool(repair_fn(path, first_exc))
            except Exception as repair_exc:
                raise RuntimeError(
                    f"Replay shard load failed and repair failed: path={path}; "
                    f"load_error={type(first_exc).__name__}: {first_exc}; "
                    f"repair_error={type(repair_exc).__name__}: {repair_exc}"
                ) from repair_exc

            if not repaired_or_validated:
                raise first_exc

            try:
                shard = self._load_shard_uncached(path)
            except Exception as retry_exc:
                raise RuntimeError(
                    f"Replay shard still failed after repair retry: path={path}; "
                    f"first_error={type(first_exc).__name__}: {first_exc}; "
                    f"retry_error={type(retry_exc).__name__}: {retry_exc}"
                ) from retry_exc

        self._cache[path] = shard
        self._cache.move_to_end(path, last=True)

        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)

        return shard

    def _load_shard_uncached(self, path: Path) -> List[RootSample]:
        try:
            shard = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load replay shard: path={path}; "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(shard, list):
            raise TypeError(f"Shard must contain list[RootSample], got {type(shard)} at {path}")
        return shard

    @staticmethod
    def _canonical_path(path: Path) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
