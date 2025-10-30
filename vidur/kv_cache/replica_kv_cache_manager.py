# from __future__ import annotations

# from collections import defaultdict
# from dataclasses import dataclass
# from typing import Dict, List, Optional

# from vidur.entities import Request
# from vidur.kv_cache.base_kv_cache_manager import KVCacheManager, PrefixCacheStats
# from vidur.kv_cache.kv_cache_block import KVCacheBlock
# from vidur.kv_cache.utils import BlockHashType
# from vidur.logger import init_logger
from vidur.utils import cdiv

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

from vidur.logger import init_logger
from vidur.entities.request import Request
from vidur.kv_cache.base_kv_cache_manager import KVCacheManager, PrefixCacheStats
from vidur.kv_cache.kv_cache_block import KVCacheBlock
from vidur.kv_cache.utils import BlockHashType
from vidur.utils.snapshot_utils import to_primitive_tree

_SNAP_VERSION_KV = 1



logger = init_logger(__name__)


@dataclass(frozen=True)
class KVCacheBlockSnapshot:
    block_id: int
    ref_cnt: int
    block_hash: Optional[dict]  # encoded BlockHashType

@dataclass(frozen=True)
class BlockPoolSnapshot:
    __v__: int
    blocks: List[KVCacheBlockSnapshot]                 # one per block_id, index/ID stable
    free_block_ids: List[int]                          # ordered free list
    # list of {"block_hash": <encoded bht>, "block_ids": [int, ...]}
    cached_entries: List[dict]

@dataclass(frozen=True)
class ReplicaKVCacheManagerSnapshot:
    __v__: int
    block_pool: Dict[str, Any]                 # BlockPoolSnapshot as dict
    req_to_blocks: Dict[int, List[int]]
    req_to_block_hashes: Dict[int, List[dict]] # <-- encoded BHTs
    num_cached_block: Dict[int, int]
    prefix_cache_stats: Dict[str, int]






class ReplicaKVCacheManager(KVCacheManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _set_block_hash(self, block: KVCacheBlock, h: Optional[BlockHashType]) -> None:
        # TODO: use a public setter if/when available
        block._block_hash = h


    @staticmethod
    def _encode_bht(bht: Optional[BlockHashType]) -> Optional[dict]:
        if bht is None:
            return None
        return {"hash_value": int(bht.hash_value), "token_ids": list(bht.token_ids)}

    @staticmethod
    def _decode_bht(d: Optional[dict]) -> Optional[BlockHashType]:
        if d is None:
            return None
        return BlockHashType(int(d["hash_value"]), tuple(int(x) for x in d["token_ids"]))

    def _encode_bht_list(self, items: List[BlockHashType]) -> List[dict]:
        return [self._encode_bht(x) for x in items]

    def _decode_bht_list(self, items: List[dict]) -> List[BlockHashType]:
        return [self._decode_bht(x) for x in items]


    def _validate_invariants(self) -> None:
        blocks_by_id = {b.block_id: b for b in self.block_pool.blocks}

        # collect free ids
        free_ids = set()
        cur = self.block_pool.free_block_queue.free_list_head
        while cur is not None:
            free_ids.add(cur.block_id)
            cur = cur.next_free_block

        # allocated ids
        alloc_ids = {blk.block_id for blks in self.req_to_blocks.values() for blk in blks}

        # disjointness
        assert free_ids.isdisjoint(alloc_ids), "Block in both free and allocated sets"

        # ref_cnt consistency (adjust if sharing semantics differ)
        expected_ref = {bid: 0 for bid in blocks_by_id}
        for blks in self.req_to_blocks.values():
            for blk in blks:
                expected_ref[blk.block_id] += 1
        for bid, blk in blocks_by_id.items():
            assert blk.ref_cnt == expected_ref[bid], f"ref_cnt mismatch for block {bid}"

        # cached map validity
        for _, id_map in self.block_pool.cached_block_hash_to_block.items():
            for bid in id_map.keys():
                assert bid in blocks_by_id, f"cached map references unknown block {bid}"

        # optional: free blocks must have zero ref_cnt
        for bid in free_ids:
            assert blocks_by_id[bid].ref_cnt == 0, f"free block {bid} has nonzero ref_cnt"

    

    # def snapshot_state(self) -> ReplicaKVCacheManagerSnapshot:
    #     """Capture a minimal representation of the cache allocator state."""
    #     blocks_state = [
    #         KVCacheBlockSnapshot(
    #             block_id=block.block_id,
    #             ref_cnt=block.ref_cnt,
    #             block_hash=block.block_hash,
    #             prev_free_block_id=block.prev_free_block.block_id
    #             if block.prev_free_block
    #             else None,
    #             next_free_block_id=block.next_free_block.block_id
    #             if block.next_free_block
    #             else None,
    #         )
    #         for block in self.block_pool.blocks
    #     ]

    #     free_block_ids: List[int] = []
    #     cursor = self.block_pool.free_block_queue.free_list_head
    #     while cursor is not None:
    #         free_block_ids.append(cursor.block_id)
    #         cursor = cursor.next_free_block

    #     cached_map: Dict[BlockHashType, List[int]] = {
    #         block_hash: list(blocks.keys())
    #         for block_hash, blocks in self.block_pool.cached_block_hash_to_block.items()
    #     }

    #     req_to_blocks = {
    #         int(req_id): [block.block_id for block in blocks]
    #         for req_id, blocks in self.req_to_blocks.items()
    #     }
    #     req_to_block_hashes = {
    #         int(req_id): list(block_hashes)
    #         for req_id, block_hashes in self.req_to_block_hashes.items()
    #     }

    #     prefix_stats = PrefixCacheStats(
    #         reset=self.prefix_cache_stats.reset,
    #         requests=self.prefix_cache_stats.requests,
    #         queries=self.prefix_cache_stats.queries,
    #         hits=self.prefix_cache_stats.hits,
    #     )

    #     return ReplicaKVCacheManagerSnapshot(
    #         block_pool=BlockPoolSnapshot(
    #             blocks=blocks_state,
    #             free_block_ids=free_block_ids,
    #             cached_block_hash_to_ids=cached_map,
    #         ),
    #         req_to_blocks=req_to_blocks,
    #         req_to_block_hashes=req_to_block_hashes,
    #         num_cached_block=dict(self.num_cached_block),
    #         prefix_cache_stats=prefix_stats,
    #     )

    # def restore_state(self, snapshot: ReplicaKVCacheManagerSnapshot) -> None:
    #     """Restore allocator state captured via ``snapshot_state``."""
    #     blocks_by_id = {block.block_id: block for block in self.block_pool.blocks}

    #     for block_snapshot in snapshot.block_pool.blocks:
    #         block = blocks_by_id[block_snapshot.block_id]
    #         block.ref_cnt = block_snapshot.ref_cnt
    #         block._block_hash = block_snapshot.block_hash
    #         block.prev_free_block = None
    #         block.next_free_block = None

    #     free_ids = snapshot.block_pool.free_block_ids
    #     queue = self.block_pool.free_block_queue
    #     queue.num_free_blocks = len(free_ids)
    #     queue.free_list_head = None
    #     queue.free_list_tail = None

    #     prev_block = None
    #     free_set = set(free_ids)
    #     for block_id in free_ids:
    #         block = blocks_by_id[block_id]
    #         block.prev_free_block = prev_block
    #         block.next_free_block = None
    #         if prev_block is None:
    #             queue.free_list_head = block
    #         else:
    #             prev_block.next_free_block = block
    #         prev_block = block
    #     queue.free_list_tail = prev_block

    #     # Ensure allocated blocks are detached from the free list.
    #     for block in blocks_by_id.values():
    #         if block.block_id not in free_set:
    #             block.prev_free_block = None
    #             block.next_free_block = None

    #     self.block_pool.cached_block_hash_to_block = defaultdict(dict)
    #     for block_hash, block_ids in snapshot.block_pool.cached_block_hash_to_ids.items():
    #         self.block_pool.cached_block_hash_to_block[block_hash] = {
    #             block_id: blocks_by_id[block_id] for block_id in block_ids
    #         }

    #     self.req_to_blocks = defaultdict(list)
    #     for req_id, block_ids in snapshot.req_to_blocks.items():
    #         self.req_to_blocks[req_id] = [blocks_by_id[block_id] for block_id in block_ids]

    #     self.req_to_block_hashes = defaultdict(list)
    #     for req_id, block_hashes in snapshot.req_to_block_hashes.items():
    #         self.req_to_block_hashes[req_id] = list(block_hashes)

    #     self.num_cached_block = dict(snapshot.num_cached_block)
    #     self.prefix_cache_stats = PrefixCacheStats(
    #         reset=snapshot.prefix_cache_stats.reset,
    #         requests=snapshot.prefix_cache_stats.requests,
    #         queries=snapshot.prefix_cache_stats.queries,
    #         hits=snapshot.prefix_cache_stats.hits,
    #     )


    # --- snapshot/restore ------------------------------------------------
    def snapshot_state(self) -> ReplicaKVCacheManagerSnapshot:
        blocks_state = [
            KVCacheBlockSnapshot(
                block_id=blk.block_id,
                ref_cnt=blk.ref_cnt,
                block_hash=self._encode_bht(blk.block_hash),
            )
            for blk in self.block_pool.blocks
        ]

        # free list as ordered ids
        free_block_ids: List[int] = []
        cursor = self.block_pool.free_block_queue.free_list_head
        while cursor is not None:
            free_block_ids.append(cursor.block_id)
            cursor = cursor.next_free_block

        # encode cached map as a list of entries
        cached_entries = [
            {"block_hash": self._encode_bht(h), "block_ids": list(block_dict.keys())}
            for h, block_dict in self.block_pool.cached_block_hash_to_block.items()
        ]

        req_to_blocks = {
            int(req_id): [blk.block_id for blk in blks]
            for req_id, blks in self.req_to_blocks.items()
        }
        # Encode namedtuples -> dicts
        req_to_block_hashes = {
            int(req_id): self._encode_bht_list(list(hashes))
            for req_id, hashes in self.req_to_block_hashes.items()
        }

        prefix_stats = {
            "reset": self.prefix_cache_stats.reset,
            "requests": self.prefix_cache_stats.requests,
            "queries": self.prefix_cache_stats.queries,
            "hits": self.prefix_cache_stats.hits,
        }

        bp = BlockPoolSnapshot(
            __v__=_SNAP_VERSION_KV,
            blocks=blocks_state,
            free_block_ids=free_block_ids,
            cached_entries=cached_entries,
        )

        snap = ReplicaKVCacheManagerSnapshot(
            __v__=_SNAP_VERSION_KV,
            block_pool=to_primitive_tree(asdict(bp)),
            req_to_blocks=req_to_blocks,
            req_to_block_hashes=req_to_block_hashes,  # encoded
            num_cached_block=dict(self.num_cached_block),
            prefix_cache_stats=prefix_stats,
        )
        # normalize outer layer too
        return ReplicaKVCacheManagerSnapshot(**to_primitive_tree(asdict(snap)))

    def restore_state(self, snapshot: ReplicaKVCacheManagerSnapshot) -> None:
        assert int(snapshot.__v__) == _SNAP_VERSION_KV, "KV snapshot version mismatch"
        bp = snapshot.block_pool
        assert int(bp["__v__"]) == _SNAP_VERSION_KV, "BlockPool snapshot version mismatch"

        blocks_by_id = {blk.block_id: blk for blk in self.block_pool.blocks}

        # restore per-block attrs
        for bs in bp["blocks"]:
            blk = blocks_by_id[bs["block_id"]]
            blk.ref_cnt = int(bs["ref_cnt"])
            self._set_block_hash(blk, self._decode_bht(bs.get("block_hash")))
            blk.prev_free_block = None
            blk.next_free_block = None

        # rebuild free list
        free_ids = list(bp["free_block_ids"])
        q = self.block_pool.free_block_queue
        q.num_free_blocks = len(free_ids)
        q.free_list_head = None
        q.free_list_tail = None
        prev = None
        free_set = set(free_ids)
        for bid in free_ids:
            b = blocks_by_id[bid]
            b.prev_free_block = prev
            b.next_free_block = None
            if prev is None:
                q.free_list_head = b
            else:
                prev.next_free_block = b
            prev = b
        q.free_list_tail = prev

        # ensure allocated blocks have no free links
        for b in blocks_by_id.values():
            if b.block_id not in free_set:
                b.prev_free_block = None
                b.next_free_block = None

        # rebuild cached hash map (decode keys)
        # cached map
        self.block_pool.cached_block_hash_to_block = defaultdict(dict)
        for entry in bp["cached_entries"]:
            h = self._decode_bht(entry["block_hash"])
            self.block_pool.cached_block_hash_to_block[h] = {
                bid: blocks_by_id[bid] for bid in entry["block_ids"]
            }
        # ownership maps
        from collections import defaultdict as _dd
        self.req_to_blocks = _dd(list)
        for rid, ids in snapshot.req_to_blocks.items():
            self.req_to_blocks[int(rid)] = [blocks_by_id[i] for i in ids]

        self.req_to_block_hashes = _dd(list)
        for rid, enc_hashes in snapshot.req_to_block_hashes.items():
            self.req_to_block_hashes[int(rid)] = self._decode_bht_list(enc_hashes)
            
        self.num_cached_block = {int(k): int(v) for k, v in snapshot.num_cached_block.items()}
        self.prefix_cache_stats = PrefixCacheStats(
            reset=snapshot.prefix_cache_stats["reset"],
            requests=snapshot.prefix_cache_stats["requests"],
            queries=snapshot.prefix_cache_stats["queries"],
            hits=snapshot.prefix_cache_stats["hits"],
        )

        # final safety checks
        self._validate_invariants()




    def allocate_slots_with_disk_cache(
        self,
        request: Request,
        num_tokens: int,
        num_disk_computed_blocks: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            num_disk_computed_blocks: The number of blocks that are cached in the
                disk for this request
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < disk computed >    | < new > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []

        assert num_disk_computed_blocks >= len(
            new_computed_blocks
        ), f"Cached blocks in disk ({num_disk_computed_blocks}) can not be less than local ({len(new_computed_blocks)})"

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (
            request.num_processed_tokens + num_disk_computed_blocks * self.block_size
        )

        num_required_blocks = cdiv(num_computed_tokens + num_tokens, self.block_size)

        req_blocks = self.req_to_blocks[request.id]
        num_new_blocks = (
            num_required_blocks - len(req_blocks) - len(new_computed_blocks)
        )

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(
            1 for blk in new_computed_blocks if blk.ref_cnt == 0
        )
        if (
            num_new_blocks
            > self.block_pool.get_num_free_blocks() - num_evictable_computed_blocks
        ):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not new_computed_blocks, (
                "Computed blocks should be empty when " "prefix caching is disabled"
            )

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(),
            )
            assert num_new_blocks > 0

            # Concatenate the computed block IDs and the new block IDs.
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(
            request.id, len(new_computed_blocks)
        )
        # We only cache blocks with generated (accepted) tokens.
        num_full_blocks_after_append = (
            num_computed_tokens + num_tokens
        ) // self.block_size

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
            hash_fn=self.caching_hash_fn,
        )

        self.num_cached_block[request.id] = num_full_blocks_after_append
        return new_blocks




