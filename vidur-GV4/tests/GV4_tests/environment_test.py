"""Focused tests for GV4 fast-forward and its thin MCTS facade."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from GV4_Engine.action_resolver import ControllerTransitionKind  # noqa: E402
from GV4_Engine.fast_forward import (  # noqa: E402
    fast_forward_decode_only_to_next_tick,
)
from GV4_Engine.state import (  # noqa: E402
    GV4State,
    Player,
    RequestLifecycle,
    RequestState,
)
from GV4_Engine.virtual_environment import (  # noqa: E402
    GV4VirtualVidurMCTSEnvironment,
)
from state_test import make_config  # noqa: E402


def timing_provider(stage_time: float):
    def predict(state, _action):
        stages = state.replicas[0].pipeline_parallel_size
        return (stage_time,) * stages, (0.0,) * (stages - 1)

    return predict


def seed_decodes(
    state: GV4State,
    config,
    *,
    count: int = 1,
    committed_decode_tokens: int = 0,
    original_decode_tokens: int = 864,
) -> None:
    total_blocks = 0
    for request_id in range(count):
        resident_tokens = 128 + committed_decode_tokens
        blocks = (resident_tokens + config.kv_cache.block_size_tokens - 1) // (
            config.kv_cache.block_size_tokens
        )
        state.requests.append(
            RequestState(
                request_id=request_id,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=0.1,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=original_decode_tokens,
                decode_credit_minted=True,
                committed_prefill_tokens=128,
                committed_decode_tokens=committed_decode_tokens,
                committed_kv_blocks=blocks,
                next_decode_deadline=0.05,
            )
        )
        total_blocks += blocks

    minted = count * config.credits.decode_credit_mint_per_prefill_completion
    committed = count * committed_decode_tokens
    state.next_request_id = count
    state.decode_credits_available = minted - committed
    state.decode_credits_minted_total = minted
    state.decode_tokens_committed_total = committed
    state.objective.requests_generated = count
    state.replica(0).rank_kv_committed_blocks[:] = [total_blocks] * len(
        state.replica(0).rank_ids
    )


def controller_edge(
    environment: GV4VirtualVidurMCTSEnvironment,
    state: GV4State,
    *,
    eviction: str = "evict_none",
    preemption: str = "preempt_none",
):
    """Resolve a zero-prefill controller edge by its readable policy names."""

    config = environment.config
    action_config = config.controller_actions
    raw_index = action_config.encode_raw_index(
        action_config.eviction_rule_names.index(eviction),
        0,
        0,
        preemption_rule_index=action_config.preemption_rule_names.index(preemption),
    )
    actions, mask = environment.sample_controller_actions(state)
    if not mask[raw_index] or actions[raw_index] is None:
        raise AssertionError(f"controller action {raw_index} is unexpectedly masked")
    return actions[raw_index]


class FastForwardTest(unittest.TestCase):
    def test_idle_state_jumps_to_tick_without_calling_predictor(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2

        child = fast_forward_decode_only_to_next_tick(
            state,
            config,
            timing_provider=lambda *_args: self.fail("predictor was called"),
        )

        self.assertEqual(child.now, 0.2)
        self.assertIs(child.next_player, Player.ADVERSARY)
        self.assertEqual(state.now, 0.0)

    def test_repeated_decode_batches_stop_after_last_credit(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        seed_decodes(state, config, committed_decode_tokens=211)
        state.assert_valid(config)

        child = fast_forward_decode_only_to_next_tick(
            state,
            config,
            timing_provider=timing_provider(0.03),
        )

        self.assertEqual(child.now, 0.2)
        self.assertEqual(child.next_microbatch_id, 5)
        self.assertEqual(child.decode_tokens_committed_total, 216)
        self.assertEqual(child.decode_credits_available, 0)
        self.assertIs(child.request(0).lifecycle, RequestLifecycle.STOPPED)
        child.assert_valid(config)

    def test_batch_crossing_tick_remains_reserved_and_inflight(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        seed_decodes(state, config)

        child = fast_forward_decode_only_to_next_tick(
            state,
            config,
            timing_provider=timing_provider(0.3),
        )

        request = child.request(0)
        self.assertEqual(child.now, 0.2)
        self.assertIs(child.next_player, Player.ADVERSARY)
        self.assertIs(request.lifecycle, RequestLifecycle.INFLIGHT_DECODE)
        self.assertEqual(request.committed_decode_tokens, 0)
        self.assertEqual(request.reserved_decode_tokens, 1)
        self.assertEqual(child.replica(0).inflight_count, 1)
        self.assertEqual(
            child.replica(0).inflight_microbatches[0].final_completion_time,
            0.3,
        )
        child.assert_valid(config)

    def test_active_prefill_disables_decode_fast_forward(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_PREFILL,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=864,
            )
        )
        state.next_request_id = 1
        state.objective.requests_generated = 1

        child = fast_forward_decode_only_to_next_tick(
            state,
            config,
            timing_provider=lambda *_args: self.fail("predictor was called"),
        )

        self.assertEqual(child.now, 0.0)
        self.assertIs(child.next_player, Player.CONTROLLER)
        self.assertIsNot(child, state)

    def test_kv_blocked_decode_returns_controller_decision(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        capacity = state.replica(0).rank_kv_capacity_blocks[0]
        resident_tokens = capacity * config.kv_cache.block_size_tokens
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=0.1,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=resident_tokens,
                original_decode_tokens=864,
                decode_credit_minted=True,
                committed_prefill_tokens=resident_tokens,
                committed_kv_blocks=capacity,
                next_decode_deadline=0.05,
            )
        )
        state.next_request_id = 1
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 216
        state.objective.requests_generated = 1
        state.replica(0).rank_kv_committed_blocks[:] = [capacity]

        child = fast_forward_decode_only_to_next_tick(
            state,
            config,
            timing_provider=lambda *_args: self.fail("predictor was called"),
        )

        self.assertEqual(child.now, 0.0)
        self.assertIs(child.next_player, Player.CONTROLLER)
        self.assertEqual(child.next_microbatch_id, 0)

    def test_pp_stage_zero_release_admits_a_second_batch(self) -> None:
        config = make_config(pipeline_parallel_size=2, max_inflight_microbatches=2)
        config = replace(
            config,
            scheduler=replace(config.scheduler, max_sequences=1),
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        seed_decodes(state, config, count=2, original_decode_tokens=1)

        child = fast_forward_decode_only_to_next_tick(
            state,
            config,
            timing_provider=timing_provider(0.03),
        )

        self.assertEqual(child.next_microbatch_id, 2)
        self.assertEqual(child.now, 0.2)
        self.assertTrue(
            all(request.lifecycle.is_terminal for request in child.requests)
        )
        self.assertEqual(child.decode_tokens_committed_total, 2)
        child.assert_valid(config)


class VirtualEnvironmentTest(unittest.TestCase):
    def test_preemption_without_decodes_stays_at_the_same_time(self) -> None:
        config = make_config()
        environment = GV4VirtualVidurMCTSEnvironment(
            config,
            batch_timing_provider=timing_provider(0.05),
            prefill_time_estimator=lambda _tokens: 0.1,
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_PREFILL,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=256,
                original_decode_tokens=864,
                committed_prefill_tokens=128,
                committed_kv_blocks=8,
            )
        )
        state.next_request_id = 1
        state.objective.requests_generated = 1
        state.replica(0).rank_kv_committed_blocks[:] = [8]

        action = controller_edge(
            environment,
            state,
            preemption="preempt_largest_kv",
        )
        child = environment.apply_controller_action_only(state, action)

        self.assertIs(
            action.action.transition_kind,
            ControllerTransitionKind.PREEMPT_ONLY,
        )
        self.assertEqual(child.now, 0.0)
        self.assertIs(child.next_player, Player.ADVERSARY)
        self.assertEqual(child.request(0).remaining_recompute_tokens, 128)
        self.assertEqual(child.request(0).committed_kv_blocks, 0)
        child.assert_valid(config)

    def test_eviction_that_empties_the_system_still_jumps_to_the_tick(self) -> None:
        config = make_config()
        environment = GV4VirtualVidurMCTSEnvironment(
            config,
            batch_timing_provider=timing_provider(0.05),
            prefill_time_estimator=lambda _tokens: 0.1,
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_PREFILL,
                arrival_time=0.0,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=256,
                original_decode_tokens=864,
                committed_prefill_tokens=128,
                committed_kv_blocks=8,
            )
        )
        state.next_request_id = 1
        state.objective.requests_generated = 1
        state.replica(0).rank_kv_committed_blocks[:] = [8]

        action = controller_edge(
            environment,
            state,
            eviction="evict_largest_prefill",
        )
        child = environment.apply_controller_action_only(state, action)

        self.assertIs(
            action.action.transition_kind,
            ControllerTransitionKind.EVICT_ONLY,
        )
        self.assertEqual(child.now, 0.2)
        self.assertIs(child.next_player, Player.ADVERSARY)
        self.assertIs(child.request(0).lifecycle, RequestLifecycle.DROPPED)
        child.assert_valid(config)

    def test_eviction_with_a_decode_batch_stays_at_the_same_time(self) -> None:
        config = make_config()
        environment = GV4VirtualVidurMCTSEnvironment(
            config,
            batch_timing_provider=timing_provider(0.05),
            prefill_time_estimator=lambda _tokens: 0.1,
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        seed_decodes(state, config, count=2)

        action = controller_edge(
            environment,
            state,
            eviction="evict_longest_decode",
        )
        child = environment.apply_controller_action_only(state, action)

        self.assertIs(action.action.transition_kind, ControllerTransitionKind.BATCH)
        self.assertEqual(action.action.evicted_request_ids, (0,))
        self.assertEqual(action.action.total_decode_tokens, 1)
        self.assertEqual(child.now, 0.0)
        self.assertIs(child.next_player, Player.ADVERSARY)
        self.assertIs(child.request(0).lifecycle, RequestLifecycle.DROPPED)
        self.assertIs(child.request(1).lifecycle, RequestLifecycle.INFLIGHT_DECODE)
        child.assert_valid(config)

    def test_ordinary_decode_only_action_still_fast_forwards(self) -> None:
        config = make_config()
        environment = GV4VirtualVidurMCTSEnvironment(
            config,
            batch_timing_provider=timing_provider(0.03),
            prefill_time_estimator=lambda _tokens: 0.1,
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        seed_decodes(state, config, committed_decode_tokens=211)

        action = controller_edge(environment, state)
        child = environment.apply_controller_action_only(state, action)

        self.assertIs(action.action.transition_kind, ControllerTransitionKind.BATCH)
        self.assertFalse(action.action.evicted_request_ids)
        self.assertFalse(action.action.preempted_request_ids)
        self.assertEqual(child.now, 0.2)
        self.assertEqual(child.decode_tokens_committed_total, 216)
        self.assertIs(child.request(0).lifecycle, RequestLifecycle.STOPPED)
        child.assert_valid(config)

    def test_facade_samples_and_applies_without_mutating_parent(self) -> None:
        config = make_config()
        env = GV4VirtualVidurMCTSEnvironment(
            config,
            batch_timing_provider=timing_provider(0.05),
            prefill_time_estimator=lambda _tokens: 0.1,
        )
        root = env.initial_state()

        adversary_actions, adversary_mask = env.sample_adversary_actions(root)
        adversary_raw = config.adversary_actions.encode_raw_index(1, 0, 0)
        self.assertTrue(adversary_mask[adversary_raw])
        launched = env.apply_adversary_action_only(
            root,
            adversary_actions[adversary_raw],
        )

        controller_actions, controller_mask = env.sample_controller_actions(launched)
        controller_raw = config.controller_actions.encode_raw_index(0, 1, 0)
        self.assertTrue(controller_mask[controller_raw])
        admitted = env.apply_controller_action_only(
            launched,
            controller_actions[controller_raw],
        )

        self.assertEqual(root.requests, [])
        self.assertEqual(launched.request(0).reserved_prefill_tokens, 0)
        self.assertIs(
            admitted.request(0).lifecycle,
            RequestLifecycle.INFLIGHT_PREFILL,
        )
        description = env.describe_state(admitted)
        self.assertEqual(description["requests"], {"INFLIGHT_PREFILL": 1})
        self.assertEqual(env.evaluate_objective(admitted), (0, 0.0))
        admitted.assert_valid(config)

        forced_actions, forced_mask = env.sample_adversary_actions(admitted)
        after_noop = env.apply_adversary_action_only(admitted, forced_actions[0])
        waiting_actions, waiting_mask = env.sample_controller_actions(after_noop)
        self.assertTrue(forced_mask[0])
        self.assertTrue(waiting_mask[0])
        advanced = env.apply_controller_action_only(
            after_noop,
            waiting_actions[0],
            fast_forward=False,
        )
        self.assertEqual(advanced.now, 0.05)
        self.assertIs(
            advanced.request(0).lifecycle,
            RequestLifecycle.WAITING_DECODE,
        )


if __name__ == "__main__":
    unittest.main()
