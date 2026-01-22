# File: vidur/vidur/kv_cache/infinite_kv_cache_manager.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vidur.entities.request import Request
from vidur.kv_cache.base_kv_cache_manager import PrefixCacheStats
from vidur.utils import cdiv

_SNAP_VERSION_INFINITE_KV = 1


@dataclass(frozen=True)
class InfiniteKVCacheManagerSnapshot:
    """
    Lightweight snapshot for InfiniteKVCacheManager.

    NOTE: We only snapshot per-request *counts* (not per-block state), so this is
    O(#active_requests) and NOT O(#blocks).
    """
    __v__: int
    req_to_num_blocks: Dict[int, int]
    prefix_cache_stats: Dict[str, int]


class InfiniteKVCacheManager:
    """
    Drop-in KV cache manager that assumes "always sufficient" KV capacity.

    Key semantics:
    - allocate_slots(...) NEVER returns None -> scheduler will never preempt due to KV.
    - get_computed_blocks(...) returns ([], 0) -> no prefix caching behavior.
    - snapshot_state()/restore_state() are cheap and do not serialize per-block state.

    This is intended for fast snapshot/fork during MCTS (option-2).
    """

    def __init__(
        self,
        *,
        block_size: int,
        num_gpu_blocks: Optional[int] = None,
        enable_caching: bool = False,
        caching_hash_algo: str = "builtin",
        num_preallocate_tokens: int = 0,
    ) -> None:
        self.block_size = int(block_size)
        self.num_gpu_blocks = int(num_gpu_blocks) if num_gpu_blocks is not None else 0

        # We keep these attrs so schedulers/configs don't break, but we do not
        # implement prefix caching in this manager.
        self.enable_caching = bool(enable_caching)
        self.caching_hash_algo = str(caching_hash_algo)

        self.num_preallocate_tokens = int(num_preallocate_tokens)
        self.num_preallocate_blocks = (
            int(cdiv(self.num_preallocate_tokens, self.block_size)) if self.block_size > 0 else 0
        )

        # Cheap accounting only (optional): request_id -> allocated blocks count
        self._req_to_num_blocks: Dict[int, int] = {}
        self._num_blocks_used: int = 0

        self.prefix_cache_stats = PrefixCacheStats()

    # ---------------- Metrics helpers ----------------

    @property
    def usage(self) -> float:
        if self.num_gpu_blocks <= 0:
            return 0.0
        return min(1.0, float(self._num_blocks_used) / float(self.num_gpu_blocks))

    def free_blocks(self) -> int:
        return int(self.get_num_free_blocks())

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    # ---------------- Required KVManager-ish API ----------------

    def get_computed_blocks(self, request: Request) -> Tuple[List[Any], int]:
        # No prefix caching here (but keep stats shape compatible).
        if self.enable_caching:
            self.prefix_cache_stats.requests += 1
            # queries/hits remain 0
        return [], 0

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[List[Any]] = None,
    ) -> Optional[List[Any]]:
        """
        Always succeeds. Returns [] (non-None) so schedulers won't preempt.

        We keep *approximate* accounting for usage/free-block metrics only.
        """
        if int(num_tokens) <= 0:
            raise ValueError("num_tokens must be greater than 0")

        rid = int(request.id)

        # Approximate "required blocks after this step" (like vLLM reserve-before-run).
        processed = int(getattr(request, "num_processed_tokens", 0))
        total_tokens_after = processed + int(num_tokens)

        required_blocks = int(cdiv(total_tokens_after, self.block_size)) if self.block_size > 0 else 0

        # Mimic preallocation behavior (optional; matches base manager a bit better).
        prev_blocks = int(self._req_to_num_blocks.get(rid, 0))
        if required_blocks > prev_blocks:
            required_blocks = required_blocks + int(self.num_preallocate_blocks)

        if required_blocks > prev_blocks:
            self._req_to_num_blocks[rid] = required_blocks
            self._num_blocks_used += (required_blocks - prev_blocks)

        # Scheduler only checks None vs not-None; it does not consume blocks list.
        return []

    def allocate_slots_with_disk_cache(
        self,
        *,
        request: Request,
        num_tokens: int,
        num_disk_computed_blocks: int,
        new_computed_blocks: Optional[List[Any]] = None,
    ) -> Optional[List[Any]]:
        # For disk scheduler compatibility: also always succeed.
        return self.allocate_slots(request=request, num_tokens=num_tokens, new_computed_blocks=new_computed_blocks)

    def free(self, request: Request) -> None:
        rid = int(request.id)
        prev = int(self._req_to_num_blocks.pop(rid, 0))
        self._num_blocks_used -= prev
        if self._num_blocks_used < 0:
            self._num_blocks_used = 0

    def reset_prefix_cache(self) -> bool:
        # No real cache; mark as reset for stats parity.
        self.prefix_cache_stats.reset = True
        return True

    def free_block_hashes(self, request: Request) -> None:
        # No-op (we don't store block hashes).
        return None

    def get_num_free_blocks(self) -> int:
        # If num_gpu_blocks is unknown, act "infinite".
        if self.num_gpu_blocks <= 0:
            return 2**31 - 1
        free = self.num_gpu_blocks - self._num_blocks_used
        return int(max(0, free))

    def get_allotted_blocks(self, request: Request) -> int:
        return int(self._req_to_num_blocks.get(int(request.id), 0))

    # ---------------- Snapshot / Restore (fast) ----------------

    def snapshot_state(self) -> InfiniteKVCacheManagerSnapshot:
        return InfiniteKVCacheManagerSnapshot(
            __v__=_SNAP_VERSION_INFINITE_KV,
            req_to_num_blocks=dict(self._req_to_num_blocks),
            prefix_cache_stats={
                "reset": int(bool(self.prefix_cache_stats.reset)),
                "requests": int(self.prefix_cache_stats.requests),
                "queries": int(self.prefix_cache_stats.queries),
                "hits": int(self.prefix_cache_stats.hits),
            },
        )

    def restore_state(self, snapshot: InfiniteKVCacheManagerSnapshot) -> None:
        if int(snapshot.__v__) != _SNAP_VERSION_INFINITE_KV:
            raise ValueError("InfiniteKVCacheManager snapshot version mismatch")

        self._req_to_num_blocks = {int(k): int(v) for k, v in snapshot.req_to_num_blocks.items()}
        self._num_blocks_used = int(sum(self._req_to_num_blocks.values()))

        ps = snapshot.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats(
            reset=bool(int(ps.get("reset", 0))),
            requests=int(ps.get("requests", 0)),
            queries=int(ps.get("queries", 0)),
            hits=int(ps.get("hits", 0)),
        )
