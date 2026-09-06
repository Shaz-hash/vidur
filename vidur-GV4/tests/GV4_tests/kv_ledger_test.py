"""Tests for GV4 logical KV-block accounting."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from GV4_Engine.kv_ledger import (  # noqa: E402
    KVCapacityError,
    KVLedgerError,
    additional_blocks_for_work,
    blocks_for_tokens,
    can_reserve_blocks,
    commit_batch_blocks,
    free_logical_blocks,
    preempt_request_blocks,
    release_request_blocks,
    reserve_batch_blocks,
)
from GV4_Engine.pipeline_calendar import admit_microbatch  # noqa: E402
from GV4_Engine.state import (  # noqa: E402
    BatchAllocation,
    GV4State,
    InflightMicrobatchState,
    ReplicaState,
    RequestLifecycle,
    RequestState,
)
from GV4_Engine.transition_engine import advance_to  # noqa: E402
from state_test import make_config  # noqa: E402


def waiting_prefill(
    request_id: int,
    *,
    replica_id: int = 0,
    tokens: int = 256,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        owner_replica_id=replica_id,
        lifecycle=RequestLifecycle.WAITING_PREFILL,
        arrival_time=0.0,
        prefill_deadline=1.0,
        decode_token_slo_sec=0.05,
        original_prefill_tokens=tokens,
        original_decode_tokens=16,
    )


def waiting_decode(
    request_id: int,
    *,
    replica_id: int = 0,
    prefill_tokens: int = 128,
    committed_decode_tokens: int = 0,
    committed_blocks: int = 8,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        owner_replica_id=replica_id,
        lifecycle=RequestLifecycle.WAITING_DECODE,
        arrival_time=0.0,
        prefill_deadline=1.0,
        decode_token_slo_sec=0.05,
        original_prefill_tokens=prefill_tokens,
        original_decode_tokens=32,
        decode_credit_minted=True,
        committed_prefill_tokens=prefill_tokens,
        committed_decode_tokens=committed_decode_tokens,
        committed_kv_blocks=committed_blocks,
        next_decode_deadline=0.05,
    )


def small_replica(*, capacity: tuple[int, ...]) -> ReplicaState:
    rank_count = len(capacity)
    return ReplicaState(
        replica_id=0,
        rank_ids=tuple(range(rank_count)),
        rank_kv_capacity_blocks=capacity,
        rank_kv_committed_blocks=[0] * rank_count,
        rank_kv_reserved_blocks=[0] * rank_count,
        stage_tail_finish_times=[0.0],
        stage_last_microbatch_ids=[-1],
        inflight_microbatches=[],
    )


class KVLedgerTest(unittest.TestCase):
    def test_block_count_uses_ceiling_division(self) -> None:
        self.assertEqual(blocks_for_tokens(0, 16), 0)
        self.assertEqual(blocks_for_tokens(1, 16), 1)
        self.assertEqual(blocks_for_tokens(16, 16), 1)
        self.assertEqual(blocks_for_tokens(17, 16), 2)

        with self.assertRaises(KVLedgerError):
            blocks_for_tokens(-1, 16)
        with self.assertRaises(KVLedgerError):
            blocks_for_tokens(1, 0)

    def test_incremental_blocks_distinguish_partial_and_boundary_decode(self) -> None:
        partial = waiting_decode(
            0,
            committed_decode_tokens=2,
            committed_blocks=9,
        )
        boundary = waiting_decode(1)

        self.assertEqual(
            additional_blocks_for_work(
                partial,
                decode_tokens=1,
                block_size_tokens=16,
            ),
            0,
        )
        self.assertEqual(
            additional_blocks_for_work(
                boundary,
                decode_tokens=1,
                block_size_tokens=16,
            ),
            1,
        )

    def test_existing_preallocated_blocks_are_reused(self) -> None:
        request = waiting_decode(0, committed_blocks=12)
        self.assertEqual(
            additional_blocks_for_work(
                request,
                decode_tokens=1,
                block_size_tokens=16,
            ),
            0,
        )

    def test_free_capacity_uses_bottleneck_rank(self) -> None:
        replica = small_replica(capacity=(12, 9, 15))
        replica.rank_kv_committed_blocks[:] = [6, 6, 6]
        replica.rank_kv_reserved_blocks[:] = [1, 1, 1]

        self.assertEqual(free_logical_blocks(replica), 2)
        self.assertTrue(can_reserve_blocks(replica, 2))
        self.assertFalse(can_reserve_blocks(replica, 3))
        self.assertTrue(can_reserve_blocks(replica, 3, releasable_blocks=1))

    def test_multi_rank_batch_reservation_is_atomic(self) -> None:
        config = make_config(tensor_parallel_size=2, pipeline_parallel_size=2)
        state = GV4State.initial(config)
        state.requests.extend((waiting_prefill(0), waiting_prefill(1, tokens=64)))
        allocations = (
            BatchAllocation(0, prefill_tokens=128, new_kv_blocks=8),
            BatchAllocation(1, prefill_tokens=32, new_kv_blocks=2),
        )

        reserved = reserve_batch_blocks(
            state.replica(0),
            state.requests,
            allocations,
            block_size_tokens=16,
        )

        self.assertEqual(reserved, 10)
        self.assertEqual([item.reserved_kv_blocks for item in state.requests], [8, 2])
        self.assertEqual(state.replica(0).rank_kv_reserved_blocks, [10, 10, 10, 10])

    def test_capacity_failure_changes_nothing(self) -> None:
        replica = small_replica(capacity=(10, 9))
        replica.rank_kv_committed_blocks[:] = [8, 8]
        request = waiting_prefill(0, tokens=64)
        allocation = (BatchAllocation(0, prefill_tokens=32, new_kv_blocks=2),)

        before_committed = replica.rank_kv_committed_blocks.copy()
        before_reserved = replica.rank_kv_reserved_blocks.copy()
        with self.assertRaises(KVCapacityError):
            reserve_batch_blocks(
                replica,
                [request],
                allocation,
                block_size_tokens=16,
            )

        self.assertEqual(request.reserved_kv_blocks, 0)
        self.assertEqual(replica.rank_kv_committed_blocks, before_committed)
        self.assertEqual(replica.rank_kv_reserved_blocks, before_reserved)

    def test_wrong_block_delta_is_rejected_before_mutation(self) -> None:
        replica = small_replica(capacity=(16, 16))
        request = waiting_prefill(0)
        wrong = (BatchAllocation(0, prefill_tokens=128, new_kv_blocks=7),)

        with self.assertRaisesRegex(KVLedgerError, "exact KV requirement"):
            reserve_batch_blocks(
                replica,
                [request],
                wrong,
                block_size_tokens=16,
            )

        self.assertEqual(request.reserved_kv_blocks, 0)
        self.assertEqual(replica.rank_kv_reserved_blocks, [0, 0])

    def test_zero_increment_decode_is_legal_at_full_capacity(self) -> None:
        replica = small_replica(capacity=(9, 9))
        replica.rank_kv_committed_blocks[:] = [9, 9]
        request = waiting_decode(
            0,
            committed_decode_tokens=2,
            committed_blocks=9,
        )
        allocation = (BatchAllocation(0, decode_tokens=1, new_kv_blocks=0),)

        self.assertEqual(
            reserve_batch_blocks(
                replica,
                [request],
                allocation,
                block_size_tokens=16,
            ),
            0,
        )
        self.assertEqual(free_logical_blocks(replica), 0)

    def test_commit_moves_reservation_without_changing_free_capacity(self) -> None:
        replica = small_replica(capacity=(16, 16))
        request = waiting_prefill(0)
        allocations = (BatchAllocation(0, prefill_tokens=128, new_kv_blocks=8),)
        reserve_batch_blocks(
            replica,
            [request],
            allocations,
            block_size_tokens=16,
        )
        free_before = free_logical_blocks(replica)

        committed = commit_batch_blocks(replica, [request], allocations)

        self.assertEqual(committed, 8)
        self.assertEqual(request.reserved_kv_blocks, 0)
        self.assertEqual(request.committed_kv_blocks, 8)
        self.assertEqual(replica.rank_kv_reserved_blocks, [0, 0])
        self.assertEqual(replica.rank_kv_committed_blocks, [8, 8])
        self.assertEqual(free_logical_blocks(replica), free_before)

    def test_reserve_and_commit_reconcile_with_full_state_validation(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        request = waiting_prefill(0)
        state.requests.append(request)
        state.next_request_id = 1
        state.objective.requests_generated = 1
        allocation = BatchAllocation(0, prefill_tokens=128, new_kv_blocks=8)
        allocations = (allocation,)

        reserve_batch_blocks(
            state.replica(0),
            state.requests,
            allocations,
            block_size_tokens=16,
        )
        request.lifecycle = RequestLifecycle.INFLIGHT_PREFILL
        request.reserved_prefill_tokens = 128
        request.inflight_microbatch_id = 0
        batch = InflightMicrobatchState(
            microbatch_id=0,
            replica_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=allocations,
            stage_ready_times=(0.0,),
            stage_start_times=(0.0,),
            stage_finish_times=(0.1,),
        )
        replica = state.replica(0)
        replica.stage_tail_finish_times[0] = 0.1
        replica.stage_last_microbatch_ids[0] = 0
        replica.inflight_microbatches.append(batch)
        state.next_microbatch_id = 1
        state.assert_valid(config)

        commit_batch_blocks(replica, state.requests, allocations)
        request.reserved_prefill_tokens = 0
        request.committed_prefill_tokens = 128
        request.kv_computed_tokens = 128
        request.lifecycle = RequestLifecycle.WAITING_PREFILL
        request.inflight_microbatch_id = -1
        replica.inflight_microbatches.clear()
        state.now = 0.1
        state.next_adversary_tick = 0.2
        state.assert_valid(config)

    def test_release_returns_all_request_blocks_to_every_rank(self) -> None:
        replica = small_replica(capacity=(16, 16))
        replica.rank_kv_committed_blocks[:] = [11, 11]
        request = waiting_decode(0, committed_blocks=8)

        released = release_request_blocks(request, replica)

        self.assertEqual(released, 8)
        self.assertEqual(request.committed_kv_blocks, 0)
        self.assertEqual(replica.rank_kv_committed_blocks, [3, 3])

    def test_release_rejects_inflight_request_without_mutation(self) -> None:
        replica = small_replica(capacity=(16,))
        replica.rank_kv_committed_blocks[:] = [8]
        request = waiting_decode(0, committed_blocks=8)
        request.lifecycle = RequestLifecycle.INFLIGHT_DECODE
        request.reserved_decode_tokens = 1
        request.inflight_microbatch_id = 3

        with self.assertRaisesRegex(KVLedgerError, "in-flight"):
            release_request_blocks(request, replica)

        self.assertEqual(request.committed_kv_blocks, 8)
        self.assertEqual(replica.rank_kv_committed_blocks, [8])

    def test_clone_release_does_not_modify_parent_branch(self) -> None:
        config = make_config()
        parent = GV4State.initial(config)
        parent.requests.append(waiting_decode(0, committed_blocks=8))
        parent.replica(0).rank_kv_committed_blocks[:] = [8]
        parent.next_request_id = 1
        parent.decode_credits_available = 216
        parent.decode_credits_minted_total = 216
        parent.objective.requests_generated = 1
        parent.assert_valid(config)
        child = parent.clone()

        release_request_blocks(child.request(0), child.replica(0))

        self.assertEqual(child.request(0).committed_kv_blocks, 0)
        self.assertEqual(child.replica(0).rank_kv_committed_blocks, [0])
        self.assertEqual(parent.request(0).committed_kv_blocks, 8)
        self.assertEqual(parent.replica(0).rank_kv_committed_blocks, [8])

    def test_preemption_releases_physical_kv_but_preserves_progress(self) -> None:
        replica = small_replica(capacity=(16, 16))
        replica.rank_kv_committed_blocks[:] = [10, 10]
        request = waiting_decode(
            0,
            committed_decode_tokens=20,
            committed_blocks=10,
        )
        request.prefill_lateness_sec = 0.2
        request.decode_lateness_sec = 0.1

        released = preempt_request_blocks(request, replica)

        self.assertEqual(released, 10)
        self.assertEqual(replica.rank_kv_committed_blocks, [0, 0])
        self.assertEqual(request.committed_prefill_tokens, 128)
        self.assertEqual(request.committed_decode_tokens, 20)
        self.assertEqual(request.kv_computed_tokens, 0)
        self.assertEqual(request.remaining_recompute_tokens, 148)
        self.assertEqual(request.next_decode_deadline, 0.05)
        self.assertEqual(request.prefill_lateness_sec, 0.2)
        self.assertEqual(request.decode_lateness_sec, 0.1)

    def test_partial_recomputation_restores_kv_without_new_progress(self) -> None:
        config = make_config(tensor_parallel_size=2)
        state = GV4State.initial(config)
        request = waiting_decode(
            0,
            committed_decode_tokens=20,
            committed_blocks=10,
        )
        state.requests.append(request)
        state.next_request_id = 1
        state.next_adversary_tick = 1.0
        state.decode_credits_available = 196
        state.decode_credits_minted_total = 216
        state.decode_tokens_committed_total = 20
        state.objective.requests_generated = 1
        state.replica(0).rank_kv_committed_blocks[:] = [10, 10]
        state.assert_valid(config)

        preempt_request_blocks(request, state.replica(0))
        with self.assertRaisesRegex(KVLedgerError, "fully reconstructed"):
            additional_blocks_for_work(
                request,
                decode_tokens=1,
                block_size_tokens=16,
            )

        for recompute_tokens, new_blocks, finish in (
            (64, 4, 0.1),
            (84, 6, 0.2),
        ):
            allocation = BatchAllocation(
                request_id=0,
                recompute_tokens=recompute_tokens,
                new_kv_blocks=new_blocks,
            )
            allocations = (allocation,)
            reserve_batch_blocks(
                state.replica(0),
                state.requests,
                allocations,
                block_size_tokens=16,
            )
            request.reserved_recompute_tokens = recompute_tokens
            request.inflight_microbatch_id = state.next_microbatch_id
            request.lifecycle = RequestLifecycle.INFLIGHT_RECOMPUTE
            admit_microbatch(
                state.replica(0),
                microbatch_id=state.next_microbatch_id,
                raw_action_index=0,
                canonical_action_index=0,
                allocations=allocations,
                admitted_at=state.now,
                stage_service_times=(finish - state.now,),
                pp_communication_times=(),
                scheduler=config.scheduler,
                timing=config.timing,
            )
            state.next_microbatch_id += 1
            advance_to(state, config, finish, inplace=True)

        self.assertEqual(request.lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(request.logical_context_tokens, 148)
        self.assertEqual(request.kv_computed_tokens, 148)
        self.assertEqual(request.remaining_recompute_tokens, 0)
        self.assertEqual(request.committed_prefill_tokens, 128)
        self.assertEqual(request.committed_decode_tokens, 20)
        self.assertEqual(state.decode_credits_available, 196)
        self.assertEqual(state.decode_tokens_committed_total, 20)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [10, 10])
        self.assertEqual(
            additional_blocks_for_work(
                request,
                decode_tokens=1,
                block_size_tokens=16,
            ),
            0,
        )
        state.assert_valid(config)

    def test_recompute_allocation_is_exclusive_and_bounded(self) -> None:
        replica = small_replica(capacity=(16,))
        replica.rank_kv_committed_blocks[:] = [8]
        request = waiting_decode(0, committed_blocks=8)
        preempt_request_blocks(request, replica)

        with self.assertRaisesRegex(KVLedgerError, "exactly one"):
            additional_blocks_for_work(
                request,
                decode_tokens=1,
                recompute_tokens=1,
                block_size_tokens=16,
            )
        with self.assertRaisesRegex(KVLedgerError, "missing KV"):
            additional_blocks_for_work(
                request,
                recompute_tokens=129,
                block_size_tokens=16,
            )


if __name__ == "__main__":
    unittest.main()
