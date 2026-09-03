"""Tests for the compact GV4 FIFO pipeline calendar."""

from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from GV4_Engine.pipeline_calendar import (  # noqa: E402
    PipelineCalendarError,
    PipelineUnavailableError,
    admit_microbatch,
    build_microbatch_calendar,
    can_admit_microbatch,
    next_pipeline_admission_time,
)
from GV4_Engine.kv_ledger import reserve_batch_blocks  # noqa: E402
from GV4_Engine.state import (  # noqa: E402
    BatchAllocation,
    GV4State,
    RequestLifecycle,
    RequestState,
)
from state_test import make_config  # noqa: E402


def one_prefill(request_id: int = 0) -> tuple[BatchAllocation, ...]:
    return (BatchAllocation(request_id, prefill_tokens=128, new_kv_blocks=8),)


class PipelineCalendarTest(unittest.TestCase):
    def test_builder_is_pure_and_pp1_timing_is_exact(self) -> None:
        config = make_config()
        replica = GV4State.initial(config, now=1.0).replica(0)

        batch = build_microbatch_calendar(
            replica,
            microbatch_id=0,
            raw_action_index=12,
            canonical_action_index=4,
            allocations=one_prefill(),
            admitted_at=1.0,
            stage_service_times=(0.125,),
            pp_communication_times=(),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertEqual(batch.stage_ready_times, (1.0,))
        self.assertEqual(batch.stage_start_times, (1.0,))
        self.assertEqual(batch.stage_finish_times, (1.125,))
        self.assertEqual(batch.final_completion_time, 1.125)
        self.assertEqual(replica.stage_tail_finish_times, [1.0])
        self.assertEqual(replica.stage_last_microbatch_ids, [-1])
        self.assertEqual(replica.inflight_microbatches, [])

    def test_admission_updates_only_the_compact_calendar(self) -> None:
        config = make_config()
        replica = GV4State.initial(config).replica(0)

        batch = admit_microbatch(
            replica,
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(),
            admitted_at=0.0,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertIs(replica.inflight_microbatches[0], batch)
        self.assertEqual(replica.stage_tail_finish_times, [0.1])
        self.assertEqual(replica.stage_last_microbatch_ids, [0])
        replica.assert_valid()

    def test_calendar_reconciles_with_request_and_kv_reservations(self) -> None:
        config = make_config(pipeline_parallel_size=2)
        state = GV4State.initial(config)
        request = RequestState(
            request_id=0,
            owner_replica_id=0,
            lifecycle=RequestLifecycle.WAITING_PREFILL,
            arrival_time=0.0,
            prefill_deadline=1.0,
            decode_token_slo_sec=0.05,
            original_prefill_tokens=256,
            original_decode_tokens=16,
        )
        state.requests.append(request)
        state.next_request_id = 1
        state.objective.requests_generated = 1
        allocations = one_prefill()

        reserve_batch_blocks(
            state.replica(0),
            state.requests,
            allocations,
            block_size_tokens=config.kv_cache.block_size_tokens,
        )
        request.lifecycle = RequestLifecycle.INFLIGHT_PREFILL
        request.reserved_prefill_tokens = 128
        request.inflight_microbatch_id = 0
        admit_microbatch(
            state.replica(0),
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=allocations,
            admitted_at=0.0,
            stage_service_times=(0.1, 0.1),
            pp_communication_times=(0.01,),
            scheduler=config.scheduler,
            timing=config.timing,
        )
        state.next_microbatch_id = 1

        state.assert_valid(config)

    def test_pp_stall_and_communication_are_scheduled_separately(self) -> None:
        config = make_config(
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        replica = GV4State.initial(config).replica(0)

        first = admit_microbatch(
            replica,
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(0),
            admitted_at=0.0,
            stage_service_times=(0.1, 0.3),
            pp_communication_times=(0.02,),
            scheduler=config.scheduler,
            timing=config.timing,
        )
        second = admit_microbatch(
            replica,
            microbatch_id=1,
            raw_action_index=1,
            canonical_action_index=1,
            allocations=one_prefill(1),
            admitted_at=0.1,
            stage_service_times=(0.1, 0.05),
            pp_communication_times=(0.02,),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertEqual(first.stage_ready_times, (0.0, 0.12))
        self.assertEqual(first.stage_start_times, (0.0, 0.12))
        self.assertEqual(first.stage_finish_times, (0.1, 0.42))
        self.assertEqual(second.stage_ready_times, (0.1, 0.22))
        self.assertEqual(second.stage_start_times, (0.1, 0.42))
        self.assertEqual(second.stage_finish_times, (0.2, 0.47))
        self.assertEqual(replica.stage_tail_finish_times, [0.2, 0.47])
        replica.assert_valid()

    def test_next_admission_respects_stage_zero_and_inflight_limit(self) -> None:
        config = make_config(
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        replica = GV4State.initial(config).replica(0)
        admit_microbatch(
            replica,
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(0),
            admitted_at=0.0,
            stage_service_times=(0.1, 0.3),
            pp_communication_times=(0.0,),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertEqual(
            next_pipeline_admission_time(
                replica,
                now=0.0,
                scheduler=config.scheduler,
                timing=config.timing,
            ),
            0.1,
        )

        admit_microbatch(
            replica,
            microbatch_id=1,
            raw_action_index=1,
            canonical_action_index=1,
            allocations=one_prefill(1),
            admitted_at=0.1,
            stage_service_times=(0.1, 0.05),
            pp_communication_times=(0.02,),
            scheduler=config.scheduler,
            timing=config.timing,
        )
        self.assertEqual(
            next_pipeline_admission_time(
                replica,
                now=0.1,
                scheduler=config.scheduler,
                timing=config.timing,
            ),
            0.4,
        )

    def test_busy_stage_zero_rejects_without_mutation(self) -> None:
        config = make_config(max_inflight_microbatches=2)
        replica = GV4State.initial(config).replica(0)
        admit_microbatch(
            replica,
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(0),
            admitted_at=0.0,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            scheduler=config.scheduler,
            timing=config.timing,
        )
        tails_before = replica.stage_tail_finish_times.copy()
        ids_before = replica.stage_last_microbatch_ids.copy()
        batches_before = replica.inflight_microbatches.copy()

        self.assertFalse(
            can_admit_microbatch(
                replica,
                admitted_at=0.05,
                scheduler=config.scheduler,
                timing=config.timing,
            )
        )
        with self.assertRaisesRegex(PipelineUnavailableError, "stage 0"):
            admit_microbatch(
                replica,
                microbatch_id=1,
                raw_action_index=1,
                canonical_action_index=1,
                allocations=one_prefill(1),
                admitted_at=0.05,
                stage_service_times=(0.1,),
                pp_communication_times=(),
                scheduler=config.scheduler,
                timing=config.timing,
            )

        self.assertEqual(replica.stage_tail_finish_times, tails_before)
        self.assertEqual(replica.stage_last_microbatch_ids, ids_before)
        self.assertEqual(replica.inflight_microbatches, batches_before)

    def test_full_inflight_ring_blocks_reuse_after_stage_zero_finishes(self) -> None:
        config = make_config(pipeline_parallel_size=2)
        replica = GV4State.initial(config).replica(0)
        admit_microbatch(
            replica,
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(),
            admitted_at=0.0,
            stage_service_times=(0.1, 0.4),
            pp_communication_times=(0.0,),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertEqual(replica.stage_tail_finish_times[0], 0.1)
        self.assertFalse(
            can_admit_microbatch(
                replica,
                admitted_at=0.1,
                scheduler=config.scheduler,
                timing=config.timing,
            )
        )
        with self.assertRaisesRegex(PipelineUnavailableError, "in-flight"):
            build_microbatch_calendar(
                replica,
                microbatch_id=1,
                raw_action_index=1,
                canonical_action_index=1,
                allocations=one_prefill(1),
                admitted_at=0.1,
                stage_service_times=(0.1, 0.1),
                pp_communication_times=(0.0,),
                scheduler=config.scheduler,
                timing=config.timing,
            )

    def test_invalid_prediction_changes_nothing(self) -> None:
        config = make_config(pipeline_parallel_size=2)
        replica = GV4State.initial(config).replica(0)

        for services, communication in (
            ((0.1,), (0.01,)),
            ((0.1, 0.1), (0.01, 0.01)),
            ((0.1, 0.0), (0.01,)),
            ((0.1, math.nan), (0.01,)),
            ((0.1, 0.1), (-0.01,)),
        ):
            with self.subTest(services=services, communication=communication):
                with self.assertRaises(PipelineCalendarError):
                    admit_microbatch(
                        replica,
                        microbatch_id=0,
                        raw_action_index=0,
                        canonical_action_index=0,
                        allocations=one_prefill(),
                        admitted_at=0.0,
                        stage_service_times=services,
                        pp_communication_times=communication,
                        scheduler=config.scheduler,
                        timing=config.timing,
                    )
                self.assertEqual(replica.stage_tail_finish_times, [0.0, 0.0])
                self.assertEqual(replica.stage_last_microbatch_ids, [-1, -1])
                self.assertEqual(replica.inflight_microbatches, [])

    def test_ids_and_allocation_order_are_enforced(self) -> None:
        config = make_config(max_inflight_microbatches=2)
        replica = GV4State.initial(config).replica(0)
        admit_microbatch(
            replica,
            microbatch_id=4,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(),
            admitted_at=0.0,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        with self.assertRaisesRegex(PipelineCalendarError, "greater"):
            build_microbatch_calendar(
                replica,
                microbatch_id=4,
                raw_action_index=0,
                canonical_action_index=0,
                allocations=one_prefill(1),
                admitted_at=0.1,
                stage_service_times=(0.1,),
                pp_communication_times=(),
                scheduler=config.scheduler,
                timing=config.timing,
            )

        unsorted = (
            BatchAllocation(2, prefill_tokens=128, new_kv_blocks=8),
            BatchAllocation(1, prefill_tokens=128, new_kv_blocks=8),
        )
        with self.assertRaisesRegex(PipelineCalendarError, "sorted and unique"):
            build_microbatch_calendar(
                replica,
                microbatch_id=5,
                raw_action_index=0,
                canonical_action_index=0,
                allocations=unsorted,
                admitted_at=0.1,
                stage_service_times=(0.1,),
                pp_communication_times=(),
                scheduler=config.scheduler,
                timing=config.timing,
            )

    def test_tp_inclusive_stage_time_is_not_scaled_again(self) -> None:
        config = make_config(tensor_parallel_size=4)
        replica = GV4State.initial(config).replica(0)

        batch = admit_microbatch(
            replica,
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(),
            admitted_at=0.0,
            stage_service_times=(0.075,),
            pp_communication_times=(),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertEqual(batch.stage_finish_times, (0.075,))

    def test_time_rounding_is_deterministic(self) -> None:
        config = make_config()
        replica = GV4State.initial(config, now=0.1).replica(0)

        batch = admit_microbatch(
            replica,
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(),
            admitted_at=0.1,
            stage_service_times=(0.2,),
            pp_communication_times=(),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertEqual(batch.final_completion_time, 0.3)

    def test_clone_has_independent_calendar_arrays_and_ring(self) -> None:
        config = make_config(max_inflight_microbatches=2)
        original = GV4State.initial(config)
        clone = original.clone()

        admit_microbatch(
            clone.replica(0),
            microbatch_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=one_prefill(),
            admitted_at=0.0,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            scheduler=config.scheduler,
            timing=config.timing,
        )

        self.assertEqual(original.replica(0).stage_tail_finish_times, [0.0])
        self.assertEqual(original.replica(0).inflight_microbatches, [])
        self.assertEqual(clone.replica(0).stage_tail_finish_times, [0.1])
        self.assertEqual(len(clone.replica(0).inflight_microbatches), 1)


if __name__ == "__main__":
    unittest.main()
