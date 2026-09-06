"""Tests for the compact GV4 runtime state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT))

from GV4_Engine.config import (  # noqa: E402
    DecodeCreditConfig,
    GV4EngineConfig,
    KVCacheConfig,
    ModelConfig,
    RoutingConfig,
    SchedulerConfig,
    TopologyConfig,
    VidurPredictorConfig,
)
from GV4_Engine.state import (  # noqa: E402
    BatchAllocation,
    GV4State,
    GV4StateError,
    InflightMicrobatchState,
    LaunchRecord,
    ObjectiveState,
    Player,
    RequestLifecycle,
    RequestState,
    TerminalReason,
)


def make_config(
    *,
    num_replicas: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_inflight_microbatches: int = 1,
) -> GV4EngineConfig:
    """Build a small resolved config without reading profile artifacts."""

    model = ModelConfig(
        model_id="gv4-state-test",
        model_revision="test-revision-001",
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
        num_kv_heads=2,
        vocab_size=256,
    )
    topology = TopologyConfig.contiguous(
        num_replicas=num_replicas,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        num_layers=model.num_layers,
    )
    predictor = VidurPredictorConfig(
        device="test-gpu",
        network_device="test-topology",
        cache_dir="/unused",
        prediction_max_batch_size=256,
        prediction_max_prefill_chunk_size=4096,
    )
    return GV4EngineConfig(
        model=model,
        topology=topology,
        vidur_predictor=predictor,
        kv_cache=KVCacheConfig(
            kv_budget_bytes_per_rank=(1 << 20,) * topology.total_ranks,
            block_size_tokens=16,
            memory_safety_margin_fraction=0.0,
        ),
        scheduler=SchedulerConfig(
            max_inflight_microbatches=max_inflight_microbatches,
            inter_stage_queue_capacity=max_inflight_microbatches,
        ),
        routing=RoutingConfig(enabled=num_replicas > 1),
    )


def add_inflight_prefill(state: GV4State) -> InflightMicrobatchState:
    """Install one internally consistent TP2/PP2 prefill reservation."""

    request = RequestState(
        request_id=0,
        owner_replica_id=0,
        lifecycle=RequestLifecycle.INFLIGHT_PREFILL,
        arrival_time=0.0,
        prefill_deadline=1.0,
        decode_token_slo_sec=0.05,
        original_prefill_tokens=256,
        original_decode_tokens=16,
        reserved_prefill_tokens=128,
        reserved_kv_blocks=8,
        inflight_microbatch_id=0,
    )
    batch = InflightMicrobatchState(
        microbatch_id=0,
        replica_id=0,
        raw_action_index=12,
        canonical_action_index=4,
        allocations=(BatchAllocation(0, prefill_tokens=128, new_kv_blocks=8),),
        stage_ready_times=(0.0, 0.11),
        stage_start_times=(0.0, 0.11),
        stage_finish_times=(0.10, 0.21),
    )
    replica = state.replica(0)
    replica.rank_kv_reserved_blocks[:] = [8] * len(replica.rank_ids)
    replica.stage_tail_finish_times[:] = [0.10, 0.21]
    replica.stage_last_microbatch_ids[:] = [0, 0]
    replica.inflight_microbatches.append(batch)

    state.requests.append(request)
    state.next_request_id = 1
    state.next_microbatch_id = 1
    state.objective.requests_generated = 1
    return batch


class GV4StateTest(unittest.TestCase):
    def test_initial_state_matches_multi_replica_topology(self) -> None:
        config = make_config(
            num_replicas=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        state = GV4State.initial(config, now=1.25, next_player=Player.CONTROLLER)

        self.assertEqual(state.now, 1.25)
        self.assertEqual(state.next_adversary_tick, 1.25)
        self.assertIs(state.next_player, Player.CONTROLLER)
        self.assertEqual(state.config_manifest_sha256, config.manifest_sha256())
        self.assertEqual(state.state_schema_version, "gv4_state_v5")
        self.assertEqual(config.layout.feature_schema_version, "gv4_markov_v5")
        self.assertEqual(
            [r.rank_ids for r in state.replicas],
            [
                (0, 1, 2, 3),
                (4, 5, 6, 7),
            ],
        )
        self.assertTrue(
            all(r.stage_tail_finish_times == [1.25, 1.25] for r in state.replicas)
        )
        capacities = config.rank_kv_block_capacities()
        self.assertTrue(
            all(
                replica.rank_kv_capacity_blocks
                == tuple(capacities[rank] for rank in replica.rank_ids)
                for replica in state.replicas
            )
        )
        self.assertFalse(hasattr(state, "__dict__"))
        self.assertFalse(hasattr(state.replicas[0], "__dict__"))
        state.assert_valid(config)

    def test_request_accounting_is_derived_not_duplicated(self) -> None:
        request = RequestState(
            request_id=0,
            owner_replica_id=0,
            lifecycle=RequestLifecycle.WAITING_DECODE,
            arrival_time=0.25,
            prefill_deadline=0.75,
            decode_token_slo_sec=0.05,
            original_prefill_tokens=128,
            original_decode_tokens=10,
            decode_credit_minted=True,
            committed_prefill_tokens=128,
            committed_decode_tokens=3,
            committed_kv_blocks=9,
            next_decode_deadline=0.85,
        )

        self.assertEqual(request.remaining_prefill_tokens, 0)
        self.assertEqual(request.remaining_decode_tokens, 7)
        self.assertEqual(request.resident_tokens, 131)
        self.assertAlmostEqual(request.age_at(1.0), 0.75)
        self.assertAlmostEqual(request.prefill_slack_at(1.0), -0.25)
        self.assertEqual(request.tokens_used_in_final_kv_block(16), 3)
        self.assertEqual(request.tokens_until_next_kv_block(16), 13)

    def test_valid_inflight_pp_state_reconciles_all_ledgers(self) -> None:
        config = make_config(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        state = GV4State.initial(config)
        batch = add_inflight_prefill(state)

        state.assert_valid(config)

        self.assertAlmostEqual(batch.final_completion_time, 0.21)
        self.assertEqual(batch.total_prefill_tokens, 128)
        self.assertEqual(batch.total_decode_tokens, 0)
        replica = state.replica(0)
        self.assertEqual(
            replica.free_kv_blocks_by_rank(),
            tuple(capacity - 8 for capacity in replica.rank_kv_capacity_blocks),
        )
        self.assertIs(replica.find_microbatch(0), batch)

    def test_decode_at_block_boundary_reserves_credit_and_new_kv(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.INFLIGHT_DECODE,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=16,
                decode_credit_minted=True,
                committed_prefill_tokens=128,
                reserved_decode_tokens=1,
                committed_kv_blocks=8,
                reserved_kv_blocks=1,
                inflight_microbatch_id=0,
                next_decode_deadline=0.05,
            )
        )
        batch = InflightMicrobatchState(
            microbatch_id=0,
            replica_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=(BatchAllocation(0, decode_tokens=1, new_kv_blocks=1),),
            stage_ready_times=(0.0,),
            stage_start_times=(0.0,),
            stage_finish_times=(0.01,),
        )
        replica = state.replica(0)
        replica.rank_kv_committed_blocks[:] = [8]
        replica.rank_kv_reserved_blocks[:] = [1]
        replica.stage_tail_finish_times[:] = [0.01]
        replica.stage_last_microbatch_ids[:] = [0]
        replica.inflight_microbatches.append(batch)
        state.next_request_id = 1
        state.next_microbatch_id = 1
        state.decode_credits_available = 215
        state.decode_credits_reserved = 1
        state.decode_credits_minted_total = 216
        state.objective.requests_generated = 1

        state.assert_valid(config)

        self.assertEqual(state.request(0).committed_decode_tokens, 0)
        self.assertEqual(state.request(0).reserved_decode_tokens, 1)
        self.assertEqual(
            replica.free_kv_blocks_by_rank()[0],
            replica.rank_kv_capacity_blocks[0] - 9,
        )

    def test_clone_isolates_every_mutable_branch_record(self) -> None:
        config = make_config(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        parent = GV4State.initial(config)
        parent.launch_history.append(LaunchRecord(0.0, 1, 256))
        parent_batch = add_inflight_prefill(parent)
        parent.assert_valid(config)

        child = parent.clone()
        child_batch = child.replica(0).inflight_microbatches[0]
        self.assertIsNot(child.requests[0], parent.requests[0])
        self.assertIsNot(child.replicas[0], parent.replicas[0])
        self.assertIsNot(child.launch_history, parent.launch_history)
        self.assertIsNot(child.objective, parent.objective)
        self.assertIsNot(child_batch, parent_batch)
        self.assertIs(child_batch.allocations, parent_batch.allocations)
        self.assertIs(child_batch.stage_finish_times, parent_batch.stage_finish_times)

        child.now = 0.05
        child.requests[0].prefill_lateness_sec = 0.5
        child.launch_history.clear()
        child.replicas[0].rank_kv_reserved_blocks[0] = 7
        child_batch.completion_applied = True
        child.objective.total_cost = 2.0

        self.assertEqual(parent.now, 0.0)
        self.assertEqual(parent.requests[0].prefill_lateness_sec, 0.0)
        self.assertEqual(parent.launch_history, [LaunchRecord(0.0, 1, 256)])
        self.assertEqual(parent.replicas[0].rank_kv_reserved_blocks[0], 8)
        self.assertFalse(parent_batch.completion_applied)
        self.assertEqual(parent.objective.total_cost, 0.0)
        parent.assert_valid(config)

    def test_rejects_zero_token_launch_record(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        state.launch_history.append(LaunchRecord(0.0, 1, 0))

        with self.assertRaisesRegex(GV4StateError, "prefill_tokens must be positive"):
            state.assert_valid(config)

    def test_rejects_decode_state_without_next_deadline(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=16,
                decode_credit_minted=True,
                committed_prefill_tokens=128,
                committed_kv_blocks=8,
            )
        )
        state.replica(0).rank_kv_committed_blocks[:] = [8]
        state.next_request_id = 1
        state.objective.requests_generated = 1

        with self.assertRaisesRegex(GV4StateError, "next_decode_deadline"):
            state.assert_valid(config)

    def test_rejects_request_batch_reservation_mismatch(self) -> None:
        config = make_config(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        state = GV4State.initial(config)
        add_inflight_prefill(state)
        state.requests[0].reserved_prefill_tokens = 64

        with self.assertRaisesRegex(GV4StateError, "reservations differ"):
            state.assert_valid(config)

    def test_rejects_rank_kv_ledger_mismatch(self) -> None:
        config = make_config(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        state = GV4State.initial(config)
        add_inflight_prefill(state)
        state.replica(0).rank_kv_reserved_blocks[0] = 7

        with self.assertRaisesRegex(GV4StateError, "reserved KV ledger"):
            state.assert_valid(config)

    def test_rejects_invalid_lifecycle_progress(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=256,
                original_decode_tokens=16,
            )
        )
        state.next_request_id = 1
        state.objective.requests_generated = 1

        with self.assertRaisesRegex(GV4StateError, "WAITING_DECODE"):
            state.assert_valid(config)

    def test_rejects_unsorted_batch_request_ids(self) -> None:
        batch = InflightMicrobatchState(
            microbatch_id=0,
            replica_id=0,
            raw_action_index=0,
            canonical_action_index=0,
            allocations=(
                BatchAllocation(1, prefill_tokens=128, new_kv_blocks=8),
                BatchAllocation(0, prefill_tokens=128, new_kv_blocks=8),
            ),
            stage_ready_times=(0.0,),
            stage_start_times=(0.0,),
            stage_finish_times=(0.1,),
        )

        with self.assertRaisesRegex(GV4StateError, "sorted and unique"):
            batch.assert_valid(expected_stage_count=1)

    def test_lookup_methods_fail_closed(self) -> None:
        state = GV4State.initial(make_config())

        with self.assertRaisesRegex(GV4StateError, "unknown request"):
            state.request(0)
        with self.assertRaisesRegex(GV4StateError, "unknown replica"):
            state.replica(1)

    def test_terminal_request_releases_kv_and_matches_objective(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.COMPLETED,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=2,
                decode_credit_minted=True,
                committed_prefill_tokens=128,
                committed_decode_tokens=2,
                terminal_reason=TerminalReason.NATURAL_COMPLETION,
                terminal_time=0.5,
            )
        )
        state.decode_credits_available = 214
        state.decode_credits_minted_total = 216
        state.decode_tokens_committed_total = 2
        state.next_request_id = 1
        state.objective = ObjectiveState(requests_generated=1, requests_completed=1)

        state.assert_valid(config)

    def test_decode_credit_mint_must_match_target_average(self) -> None:
        config = make_config()

        with self.assertRaisesRegex(
            ValueError, "must equal the target decode-token average"
        ):
            replace(
                config,
                credits=DecodeCreditConfig(
                    decode_credit_mint_per_prefill_completion=215
                ),
            )

    def test_rejects_decode_credit_conservation_mismatch(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        state.decode_credits_available = 1

        with self.assertRaisesRegex(GV4StateError, "conservation invariant"):
            state.assert_valid(config)

    def test_pooled_credit_can_fund_one_request_beyond_one_mint(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        for request_id in range(3):
            state.requests.append(
                RequestState(
                    request_id=request_id,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.STOPPED,
                    arrival_time=0.0,
                    prefill_deadline=0.1,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=128,
                    original_decode_tokens=864,
                    decode_credit_minted=True,
                    committed_prefill_tokens=128,
                    terminal_reason=TerminalReason.ADVERSARY_STOP,
                    terminal_time=0.2,
                )
            )
        active = RequestState(
            request_id=3,
            owner_replica_id=0,
            lifecycle=RequestLifecycle.WAITING_DECODE,
            arrival_time=0.0,
            prefill_deadline=0.1,
            decode_token_slo_sec=0.05,
            original_prefill_tokens=128,
            original_decode_tokens=864,
            decode_credit_minted=True,
            committed_prefill_tokens=128,
            committed_decode_tokens=648,
            committed_kv_blocks=49,
            next_decode_deadline=0.25,
        )
        state.requests.append(active)
        state.next_request_id = 4
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 864
        state.decode_tokens_committed_total = 648
        state.replica(0).rank_kv_committed_blocks[:] = [49]
        state.objective = ObjectiveState(requests_generated=4, requests_stopped=3)

        state.assert_valid(config)
        self.assertGreater(
            active.committed_decode_tokens,
            config.credits.decode_credit_mint_per_prefill_completion,
        )

    def test_request_decode_length_cannot_exceed_864(self) -> None:
        config = make_config()
        state = GV4State.initial(config)
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_PREFILL,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=865,
            )
        )
        state.next_request_id = 1
        state.objective.requests_generated = 1

        with self.assertRaisesRegex(GV4StateError, "decode token count exceeds"):
            state.assert_valid(config)


if __name__ == "__main__":
    unittest.main()
