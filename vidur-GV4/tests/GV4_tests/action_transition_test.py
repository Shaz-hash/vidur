"""Integration tests for GV4 action resolution and atomic transitions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from GV4_Engine.action_resolver import (  # noqa: E402
    ControllerTransitionKind,
    resolve_adversary_action,
    resolve_adversary_actions,
    resolve_controller_action,
    resolve_controller_actions,
)
from GV4_Engine.state import (  # noqa: E402
    GV4State,
    GV4StateError,
    Player,
    RequestLifecycle,
    RequestState,
    TerminalReason,
)
from GV4_Engine.transition_engine import (  # noqa: E402
    TransitionError,
    advance_to,
    apply_adversary_action,
    apply_controller_action,
)
from state_test import make_config  # noqa: E402


def controller_raw(
    config,
    rule: str,
    budget: int,
    heuristic: str = "SJF",
    *,
    preemption: str = "preempt_none",
) -> int:
    actions = config.controller_actions
    return actions.encode_raw_index(
        actions.eviction_rule_names.index(rule),
        actions.prefill_budget_options.index(budget),
        actions.ordering_heuristics.index(heuristic),
        preemption_rule_index=actions.preemption_rule_names.index(preemption),
    )


def adversary_raw(
    config, launch_count: int, prefill_tokens: int | None, stop_rule: str
) -> int:
    actions = config.adversary_actions
    template_index = (
        None
        if prefill_tokens is None
        else actions.prefill_token_templates.index(prefill_tokens)
    )
    return actions.encode_raw_index(
        launch_count,
        template_index,
        actions.stop_rule_names.index(stop_rule),
    )


def canonical_for_raw(state, config, raw_index: int):
    raw, canonical = resolve_controller_actions(
        state,
        config,
        replica_id=0,
        prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
    )
    resolved = raw[raw_index]
    if resolved is None:
        raise AssertionError(f"raw action {raw_index} is masked")
    return next(
        action
        for action in canonical
        if raw_index in action.equivalent_raw_indices
    )


def canonical_adversary_for_raw(state, config, raw_index: int):
    raw, canonical = resolve_adversary_actions(state, config)
    resolved = raw[raw_index]
    if resolved is None:
        raise AssertionError(f"raw adversary action {raw_index} is masked")
    return next(
        action
        for action in canonical
        if raw_index in action.equivalent_raw_indices
    )


def launch_requests(state, config, *, count: int = 1, tokens: int = 128):
    raw_index = adversary_raw(config, count, tokens, "stop_none")
    action = canonical_adversary_for_raw(state, config, raw_index)
    return apply_adversary_action(
        state,
        config,
        action,
        prefill_time_estimator=lambda _tokens: 0.1,
    ).state


class ActionResolverTest(unittest.TestCase):
    def test_preemption_feature_gate_masks_memory_effects(self) -> None:
        base = make_config()
        config = replace(
            base,
            scheduler=replace(
                base.scheduler,
                request_preemption_enabled=False,
            ),
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
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
        state.replica(0).rank_kv_committed_blocks[:] = [8]
        state.objective.requests_generated = 1
        state.assert_valid(config)

        action = resolve_controller_action(
            state,
            config,
            replica_id=0,
            raw_action_index=controller_raw(
                config,
                "evict_none",
                0,
                preemption="preempt_largest_kv",
            ),
            prefill_time_estimator=lambda _request, _tokens: 0.1,
        )
        self.assertIsNone(action)

    def test_all_four_preemption_policies_choose_one_deterministic_victim(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        contexts = (64, 160, 96)
        deadlines = (2.0, 0.05, 1.0)
        for request_id, (context_tokens, deadline) in enumerate(
            zip(contexts, deadlines)
        ):
            state.requests.append(
                RequestState(
                    request_id=request_id,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_PREFILL,
                    arrival_time=0.0,
                    prefill_deadline=deadline,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=256,
                    original_decode_tokens=864,
                    committed_prefill_tokens=context_tokens,
                    committed_kv_blocks=context_tokens // 16,
                )
            )
        state.next_request_id = 3
        state.replica(0).rank_kv_committed_blocks[:] = [sum(contexts) // 16]
        state.objective.requests_generated = 3
        state.assert_valid(config)

        expected = {
            "preempt_min_recompute": 0,
            "preempt_largest_kv": 1,
            "preempt_max_recovery_slack": 0,
            "preempt_best_relief_cost": 2,
        }
        for rule, request_id in expected.items():
            with self.subTest(rule=rule):
                action = resolve_controller_action(
                    state,
                    config,
                    replica_id=0,
                    raw_action_index=controller_raw(
                        config,
                        "evict_none",
                        0,
                        preemption=rule,
                    ),
                    prefill_time_estimator=lambda _request, _tokens: 0.1,
                )
                self.assertIsNotNone(action)
                assert action is not None
                self.assertEqual(action.preempted_request_ids, (request_id,))
                self.assertEqual(
                    action.transition_kind,
                    ControllerTransitionKind.PREEMPT_ONLY,
                )

    def test_recompute_and_prefill_share_one_budget_and_sjf_queue(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.extend(
            (
                RequestState(
                    request_id=0,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_DECODE,
                    arrival_time=0.0,
                    prefill_deadline=0.1,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=32,
                    original_decode_tokens=864,
                    decode_credit_minted=True,
                    committed_prefill_tokens=32,
                    committed_decode_tokens=32,
                    kv_computed_tokens=0,
                    next_decode_deadline=0.5,
                ),
                RequestState(
                    request_id=1,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_PREFILL,
                    arrival_time=0.0,
                    prefill_deadline=1.0,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=128,
                    original_decode_tokens=864,
                ),
            )
        )
        state.next_request_id = 2
        state.decode_credits_minted_total = 216
        state.decode_tokens_committed_total = 32
        state.decode_credits_available = 184
        state.objective.requests_generated = 2
        state.assert_valid(config)

        action = resolve_controller_action(
            state,
            config,
            replica_id=0,
            raw_action_index=controller_raw(config, "evict_none", 128),
            prefill_time_estimator=lambda _request, tokens: tokens / 1000.0,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(
            [
                (item.request_id, item.prefill_tokens, item.recompute_tokens)
                for item in action.allocations
            ],
            [(0, 0, 64), (1, 64, 0)],
        )
        self.assertEqual(action.total_prefill_class_tokens, 128)

    def test_prefill_budget_is_split_in_heuristic_order_without_mutation(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        for request_id, tokens in enumerate((300, 700)):
            state.requests.append(
                RequestState(
                    request_id=request_id,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_PREFILL,
                    arrival_time=0.0,
                    prefill_deadline=1.0,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=tokens,
                    original_decode_tokens=16,
                )
            )
        state.next_request_id = 2
        state.objective.requests_generated = 2
        before = state.clone()

        action = resolve_controller_action(
            state,
            config,
            replica_id=0,
            raw_action_index=controller_raw(config, "evict_none", 512),
            prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(
            [(item.request_id, item.prefill_tokens) for item in action.allocations],
            [(0, 300), (1, 212)],
        )
        self.assertEqual(action.transition_kind, ControllerTransitionKind.BATCH)
        self.assertEqual(state.requests[0].committed_prefill_tokens, 0)
        self.assertFalse(hasattr(state, "prefill_credit_lots"))
        self.assertEqual(state.requests[0].clone(), before.requests[0])

    def test_equivalent_heuristics_share_one_canonical_edge(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        raw_sjf = controller_raw(config, "evict_none", 128, "SJF")
        raw_edf = controller_raw(config, "evict_none", 128, "EDF")

        raw, canonical = resolve_controller_actions(
            state,
            config,
            replica_id=0,
            prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
        )

        self.assertEqual(raw[raw_sjf].canonical_key, raw[raw_edf].canonical_key)
        edge = next(item for item in canonical if raw_sjf in item.equivalent_raw_indices)
        self.assertIn(raw_edf, edge.equivalent_raw_indices)
        self.assertEqual(edge.representative_raw_index, raw_sjf)

    def test_equivalent_adversary_stop_rules_share_one_canonical_edge(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.ADVERSARY)
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
                next_decode_deadline=0.05,
            )
        )
        state.next_request_id = 1
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 216
        state.objective.requests_generated = 1
        state.replica(0).rank_kv_committed_blocks[:] = [8]
        raw_longest = adversary_raw(config, 0, None, "stop_longest_decode")
        raw_shortest = adversary_raw(config, 0, None, "stop_shortest_decode")

        raw, canonical = resolve_adversary_actions(state, config)

        assert raw[raw_longest] is not None
        assert raw[raw_shortest] is not None
        self.assertEqual(
            raw[raw_longest].canonical_key,
            raw[raw_shortest].canonical_key,
        )
        edge = next(
            item for item in canonical if raw_longest in item.equivalent_raw_indices
        )
        self.assertIn(raw_shortest, edge.equivalent_raw_indices)
        self.assertEqual(edge.representative_raw_index, raw_longest)

    def test_decode_inside_partial_block_is_legal_at_full_capacity(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        capacity = state.replica(0).rank_kv_capacity_blocks[0]
        resident_tokens = capacity * config.kv_cache.block_size_tokens - 1
        request = RequestState(
            request_id=0,
            owner_replica_id=0,
            lifecycle=RequestLifecycle.WAITING_DECODE,
            arrival_time=0.0,
            prefill_deadline=1.0,
            decode_token_slo_sec=0.05,
            original_prefill_tokens=resident_tokens,
            original_decode_tokens=2,
            decode_credit_minted=True,
            committed_prefill_tokens=resident_tokens,
            committed_kv_blocks=capacity,
            next_decode_deadline=0.05,
        )
        state.requests.append(request)
        state.next_request_id = 1
        state.objective.requests_generated = 1
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 216
        state.replica(0).rank_kv_committed_blocks[:] = [capacity]
        state.assert_valid(config)

        action = resolve_controller_action(
            state,
            config,
            replica_id=0,
            raw_action_index=0,
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.allocations[0].decode_tokens, 1)
        self.assertEqual(action.allocations[0].new_kv_blocks, 0)

        request.committed_prefill_tokens += 1
        request.original_prefill_tokens += 1
        blocked = resolve_controller_action(
            state,
            config,
            replica_id=0,
            raw_action_index=0,
        )
        self.assertIsNotNone(blocked)
        assert blocked is not None
        self.assertEqual(blocked.transition_kind, ControllerTransitionKind.WAIT)
        self.assertEqual(blocked.allocations, ())


    def test_zero_credit_cannot_leave_a_waiting_decode(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=0.1,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=864,
                decode_credit_minted=True,
                committed_prefill_tokens=128,
                committed_decode_tokens=216,
                committed_kv_blocks=22,
                next_decode_deadline=0.25,
            )
        )
        state.next_request_id = 1
        state.decode_credits_minted_total = 216
        state.decode_tokens_committed_total = 216
        state.replica(0).rank_kv_committed_blocks[:] = [22]
        state.objective.requests_generated = 1
        with self.assertRaisesRegex(
            GV4StateError, "zero available decode credit"
        ):
            state.assert_valid(config)


class TransitionEngineTest(unittest.TestCase):
    def test_inflight_recompute_preemption_drains_then_restarts_full_recovery(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=0.1,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=256,
                original_decode_tokens=864,
                decode_credit_minted=True,
                committed_prefill_tokens=256,
                kv_computed_tokens=0,
                next_decode_deadline=0.15,
            )
        )
        state.next_request_id = 1
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 216
        state.objective.requests_generated = 1
        state.assert_valid(config)

        recovery = canonical_for_raw(
            state,
            config,
            controller_raw(config, "evict_none", 128),
        )
        state = apply_controller_action(
            state,
            config,
            recovery,
            stage_service_times=(0.1,),
            pp_communication_times=(),
        ).state
        self.assertEqual(state.request(0).reserved_recompute_tokens, 128)

        state.next_player = Player.CONTROLLER
        preempt = canonical_for_raw(
            state,
            config,
            controller_raw(
                config,
                "evict_none",
                0,
                preemption="preempt_largest_kv",
            ),
        )
        state = apply_controller_action(state, config, preempt).state
        self.assertEqual(state.request(0).lifecycle, RequestLifecycle.PREEMPT_PENDING)

        state = advance_to(state, config, 0.1).state
        request = state.request(0)
        self.assertEqual(request.lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(request.committed_prefill_tokens, 256)
        self.assertEqual(request.committed_decode_tokens, 0)
        self.assertEqual(request.kv_computed_tokens, 0)
        self.assertEqual(request.remaining_recompute_tokens, 256)
        self.assertEqual(request.committed_kv_blocks, 0)
        self.assertEqual(state.decode_credits_available, 216)
        self.assertEqual(state.decode_tokens_committed_total, 0)
        state.assert_valid(config)

    def test_waiting_decode_preemption_releases_then_reconstructs_kv(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=0.1,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=864,
                decode_credit_minted=True,
                committed_prefill_tokens=128,
                committed_kv_blocks=8,
                next_decode_deadline=0.15,
            )
        )
        state.next_request_id = 1
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 216
        state.replica(0).rank_kv_committed_blocks[:] = [8]
        state.objective.requests_generated = 1
        state.assert_valid(config)

        raw = controller_raw(
            config,
            "evict_none",
            0,
            preemption="preempt_largest_kv",
        )
        preempt = canonical_for_raw(state, config, raw)
        state = apply_controller_action(state, config, preempt).state

        request = state.request(0)
        self.assertEqual(request.lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(request.committed_prefill_tokens, 128)
        self.assertEqual(request.kv_computed_tokens, 0)
        self.assertEqual(request.remaining_recompute_tokens, 128)
        self.assertEqual(request.committed_kv_blocks, 0)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [0])

        state.next_player = Player.CONTROLLER
        recovery_raw = controller_raw(config, "evict_none", 128)
        recovery = canonical_for_raw(state, config, recovery_raw)
        self.assertEqual(recovery.action.total_prefill_tokens, 0)
        self.assertEqual(recovery.action.total_recompute_tokens, 128)
        state = apply_controller_action(
            state,
            config,
            recovery,
            stage_service_times=(0.05,),
            pp_communication_times=(),
            prefill_time_estimator=lambda _request, tokens: tokens / 1000.0,
        ).state
        state = advance_to(state, config, 0.05).state

        request = state.request(0)
        self.assertEqual(request.lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(request.kv_computed_tokens, 128)
        self.assertEqual(request.remaining_recompute_tokens, 0)
        self.assertEqual(request.committed_prefill_tokens, 128)
        self.assertEqual(request.committed_decode_tokens, 0)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [8])
        state.assert_valid(config)

    def test_inflight_prefill_preemption_commits_mints_then_releases(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        prefill = canonical_for_raw(
            state, config, controller_raw(config, "evict_none", 128)
        )
        state = apply_controller_action(
            state,
            config,
            prefill,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            prefill_time_estimator=lambda _request, tokens: tokens / 1000.0,
        ).state
        state.next_player = Player.CONTROLLER

        raw = controller_raw(
            config,
            "evict_none",
            0,
            preemption="preempt_largest_kv",
        )
        preempt = canonical_for_raw(state, config, raw)
        self.assertEqual(preempt.action.pending_preemption_request_ids, (0,))
        state = apply_controller_action(state, config, preempt).state

        self.assertEqual(state.request(0).lifecycle, RequestLifecycle.PREEMPT_PENDING)
        self.assertEqual(state.replica(0).rank_kv_reserved_blocks, [8])
        state = advance_to(state, config, 0.1).state

        request = state.request(0)
        self.assertEqual(request.lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(request.committed_prefill_tokens, 128)
        self.assertTrue(request.decode_credit_minted)
        self.assertEqual(state.decode_credits_available, 216)
        self.assertEqual(state.decode_credits_minted_total, 216)
        self.assertEqual(request.kv_computed_tokens, 0)
        self.assertEqual(request.remaining_recompute_tokens, 128)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [0])
        self.assertEqual(state.replica(0).rank_kv_reserved_blocks, [0])
        state.assert_valid(config)

    def test_inflight_decode_preemption_commits_lateness_and_credit_first(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=0.1,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=10,
                decode_credit_minted=True,
                committed_prefill_tokens=128,
                committed_kv_blocks=8,
                next_decode_deadline=0.05,
            )
        )
        state.next_request_id = 1
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 216
        state.replica(0).rank_kv_committed_blocks[:] = [8]
        state.objective.requests_generated = 1
        decode = canonical_for_raw(state, config, 0)
        state = apply_controller_action(
            state,
            config,
            decode,
            stage_service_times=(0.1,),
            pp_communication_times=(),
        ).state
        state.next_player = Player.CONTROLLER

        raw = controller_raw(
            config,
            "evict_none",
            0,
            preemption="preempt_largest_kv",
        )
        preempt = canonical_for_raw(state, config, raw)
        state = apply_controller_action(state, config, preempt).state
        state = advance_to(state, config, 0.1).state

        request = state.request(0)
        self.assertEqual(request.lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(request.committed_decode_tokens, 1)
        self.assertAlmostEqual(request.decode_lateness_sec, 0.05)
        self.assertTrue(request.violation_recorded)
        self.assertEqual(request.next_decode_deadline, 0.15)
        self.assertEqual(state.decode_credits_available, 215)
        self.assertEqual(state.decode_credits_reserved, 0)
        self.assertEqual(state.decode_tokens_committed_total, 1)
        self.assertEqual(request.kv_computed_tokens, 0)
        self.assertEqual(request.remaining_recompute_tokens, 129)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [0])
        state.assert_valid(config)

    def test_launch_admission_and_prefill_completion_reconcile_all_ledgers(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        request = state.request(0)
        self.assertEqual(request.prefill_deadline, 0.3)
        self.assertEqual(request.original_decode_tokens, 864)
        self.assertFalse(hasattr(state, "prefill_credit_lots"))

        raw_index = controller_raw(config, "evict_none", 128)
        action = canonical_for_raw(state, config, raw_index)
        admitted = apply_controller_action(
            state,
            config,
            action,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
        ).state

        self.assertEqual(admitted.request(0).lifecycle, RequestLifecycle.INFLIGHT_PREFILL)
        self.assertEqual(admitted.decode_credits_reserved, 0)
        self.assertEqual(admitted.replica(0).rank_kv_reserved_blocks, [8])
        admitted.assert_valid(config)

        completed = advance_to(admitted, config, 0.1).state
        request = completed.request(0)
        self.assertEqual(request.lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(request.committed_prefill_tokens, 128)
        self.assertEqual(request.next_decode_deadline, 0.15)
        self.assertEqual(completed.decode_credits_available, 216)
        self.assertTrue(request.decode_credit_minted)
        self.assertEqual(completed.decode_tokens_committed_total, 0)
        self.assertEqual(completed.decode_credits_minted_total, 216)
        self.assertEqual(completed.replica(0).rank_kv_committed_blocks, [8])
        completed.assert_valid(config)

    def test_pp2_commits_only_after_the_final_stage(self) -> None:
        config = make_config(pipeline_parallel_size=2)
        state = launch_requests(GV4State.initial(config), config)
        action = canonical_for_raw(
            state, config, controller_raw(config, "evict_none", 128)
        )
        state = apply_controller_action(
            state,
            config,
            action,
            stage_service_times=(0.05, 0.09),
            pp_communication_times=(0.01,),
            prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
        ).state

        stage_zero_done = advance_to(state, config, 0.05).state
        self.assertEqual(
            stage_zero_done.request(0).lifecycle,
            RequestLifecycle.INFLIGHT_PREFILL,
        )
        self.assertEqual(stage_zero_done.request(0).committed_prefill_tokens, 0)
        self.assertFalse(stage_zero_done.request(0).decode_credit_minted)
        self.assertEqual(stage_zero_done.decode_credits_minted_total, 0)

        completed = advance_to(stage_zero_done, config, 0.15).state
        self.assertEqual(completed.request(0).lifecycle, RequestLifecycle.WAITING_DECODE)
        self.assertEqual(completed.request(0).committed_prefill_tokens, 128)
        self.assertTrue(completed.request(0).decode_credit_minted)
        self.assertEqual(completed.decode_credits_minted_total, 216)
        self.assertEqual(completed.replica(0).inflight_microbatches, [])
        completed.assert_valid(config)

    def test_invalid_inplace_timing_and_skipped_tick_leave_state_unchanged(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        action = canonical_for_raw(
            state, config, controller_raw(config, "evict_none", 128)
        )
        before = state.clone()

        with self.assertRaises(ValueError):
            apply_controller_action(
                state,
                config,
                action,
                stage_service_times=(),
                pp_communication_times=(),
                prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
                inplace=True,
            )
        self.assertEqual(state.requests[0].lifecycle, before.requests[0].lifecycle)
        self.assertEqual(
            state.decode_credits_minted_total, before.decode_credits_minted_total
        )
        self.assertEqual(state.replica(0).inflight_microbatches, [])

        with self.assertRaisesRegex(TransitionError, "adversary tick"):
            advance_to(state, config, 0.3, inplace=True)
        self.assertEqual(state.now, before.now)
        self.assertEqual(state.requests[0].lifecycle, before.requests[0].lifecycle)

    def test_late_decode_records_one_violation_and_exact_cost(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        prefill = canonical_for_raw(
            state, config, controller_raw(config, "evict_none", 128)
        )
        state = apply_controller_action(
            state,
            config,
            prefill,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
        ).state
        state = advance_to(state, config, 0.1).state
        state.next_player = Player.CONTROLLER
        decode = canonical_for_raw(state, config, 0)
        state = apply_controller_action(
            state,
            config,
            decode,
            stage_service_times=(0.06,),
            pp_communication_times=(),
        ).state
        state = advance_to(state, config, 0.16).state

        self.assertAlmostEqual(state.request(0).decode_lateness_sec, 0.01)
        self.assertEqual(state.objective.slo_violations, 1)
        self.assertAlmostEqual(state.objective.total_cost, 1.01)

    def test_decode_only_batch_uses_only_decode_credit(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 0.2
        state.requests.append(
            RequestState(
                request_id=0,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_DECODE,
                arrival_time=0.0,
                prefill_deadline=0.05,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=16,
                original_decode_tokens=1,
                decode_credit_minted=True,
                committed_prefill_tokens=16,
                committed_kv_blocks=1,
                next_decode_deadline=0.1,
            )
        )
        state.next_request_id = 1
        state.objective.requests_generated = 1
        state.decode_credits_available = 216
        state.decode_credits_minted_total = 216
        state.replica(0).rank_kv_committed_blocks[:] = [1]
        state.assert_valid(config)

        action = canonical_for_raw(state, config, 0)
        state = apply_controller_action(
            state,
            config,
            action,
            stage_service_times=(0.01,),
            pp_communication_times=(),
        ).state
        self.assertEqual(state.decode_credits_available, 215)
        self.assertEqual(state.decode_credits_reserved, 1)
        self.assertFalse(hasattr(state, "prefill_credit_lots"))

        state = advance_to(state, config, 0.01).state
        self.assertEqual(state.request(0).lifecycle, RequestLifecycle.COMPLETED)
        self.assertEqual(state.decode_credits_reserved, 0)
        self.assertEqual(state.decode_credits_available, 215)
        self.assertEqual(state.decode_tokens_committed_total, 1)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [0])
        state.assert_valid(config)

    def test_controller_eviction_is_zero_time_terminal_drop(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        prefill = canonical_for_raw(
            state, config, controller_raw(config, "evict_none", 128)
        )
        state = apply_controller_action(
            state,
            config,
            prefill,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            prefill_time_estimator=lambda _request, tokens: 0.01,
        ).state
        state = advance_to(state, config, 0.1).state
        state.next_player = Player.CONTROLLER
        raw_index = controller_raw(config, "evict_longest_decode", 0)
        eviction = canonical_for_raw(state, config, raw_index)
        outcome = apply_controller_action(state, config, eviction)

        self.assertEqual(outcome.transition_kind, "EVICT_ONLY")
        self.assertEqual(outcome.elapsed_sec, 0.0)
        self.assertEqual(outcome.state.request(0).lifecycle, RequestLifecycle.DROPPED)
        self.assertEqual(
            outcome.state.request(0).terminal_reason,
            TerminalReason.CONTROLLER_EVICTION,
        )
        self.assertEqual(outcome.state.objective.total_cost, 3.0)
        self.assertEqual(outcome.state.replica(0).rank_kv_committed_blocks, [0])

    def test_inflight_adversary_stop_drains_before_releasing_kv(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        prefill = canonical_for_raw(
            state, config, controller_raw(config, "evict_none", 128)
        )
        state = apply_controller_action(
            state,
            config,
            prefill,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            prefill_time_estimator=lambda _request, tokens: 0.01,
        ).state
        state = advance_to(state, config, 0.1).state
        state.next_player = Player.CONTROLLER
        decode = canonical_for_raw(state, config, 0)
        state = apply_controller_action(
            state,
            config,
            decode,
            stage_service_times=(0.3,),
            pp_communication_times=(),
        ).state
        self.assertEqual(state.decode_credits_available, 215)
        self.assertEqual(state.decode_credits_reserved, 1)
        state = advance_to(state, config, 0.2).state

        stop_raw = adversary_raw(config, 0, None, "stop_longest_decode")
        stop = canonical_adversary_for_raw(state, config, stop_raw)
        state = apply_adversary_action(
            state,
            config,
            stop,
            prefill_time_estimator=lambda _tokens: 0.1,
        ).state
        self.assertEqual(state.request(0).lifecycle, RequestLifecycle.STOP_PENDING)
        self.assertEqual(state.decode_credits_available, 215)
        self.assertEqual(state.decode_credits_reserved, 1)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [8])

        state = advance_to(state, config, 0.4).state
        self.assertEqual(state.request(0).lifecycle, RequestLifecycle.STOPPED)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [0])
        self.assertEqual(state.replica(0).rank_kv_reserved_blocks, [0])
        self.assertEqual(state.decode_credits_available, 215)
        self.assertEqual(state.decode_credits_reserved, 0)
        self.assertEqual(state.decode_tokens_committed_total, 1)
        state.assert_valid(config)

    def test_wait_advances_and_automatic_drop_charges_only_terminal_cost(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        wait = canonical_for_raw(state, config, 0)
        outcome = apply_controller_action(state, config, wait)
        self.assertEqual(outcome.transition_kind, "WAIT")
        self.assertEqual(outcome.state.now, 0.2)
        self.assertEqual(outcome.state.next_player, Player.ADVERSARY)

        # Isolate automatic dropping without skipping a pending adversary turn.
        outcome.state.next_adversary_tick = 3.0

        dropped = advance_to(outcome.state, config, 2.31).state
        self.assertEqual(dropped.request(0).lifecycle, RequestLifecycle.DROPPED)
        self.assertEqual(
            dropped.request(0).terminal_reason,
            TerminalReason.AUTOMATIC_SLO_DROP,
        )
        self.assertEqual(dropped.objective.slo_violations, 0)
        self.assertEqual(dropped.objective.total_cost, 3.0)

    def test_last_decode_credit_stops_waiting_and_inflight_requests(self) -> None:
        config = make_config(
            pipeline_parallel_size=2, max_inflight_microbatches=2
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.next_adversary_tick = 1.0
        for request_id, committed_decode in enumerate((216, 215)):
            state.requests.append(
                RequestState(
                    request_id=request_id,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_DECODE,
                    arrival_time=0.0,
                    prefill_deadline=0.1,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=128,
                    original_decode_tokens=864,
                    decode_credit_minted=True,
                    committed_prefill_tokens=128,
                    committed_decode_tokens=committed_decode,
                    committed_kv_blocks=22,
                    next_decode_deadline=0.05,
                )
            )
        state.next_request_id = 2
        state.decode_credits_available = 1
        state.decode_credits_minted_total = 432
        state.decode_tokens_committed_total = 431
        state.replica(0).rank_kv_committed_blocks[:] = [44, 44]
        state.objective.requests_generated = 2
        state.assert_valid(config)

        decode = canonical_for_raw(state, config, 0)
        state = apply_controller_action(
            state,
            config,
            decode,
            stage_service_times=(0.05, 0.3),
            pp_communication_times=(0.0,),
        ).state
        self.assertEqual(state.decode_credits_available, 0)
        self.assertEqual(state.decode_credits_reserved, 1)
        self.assertEqual(
            state.request(0).lifecycle, RequestLifecycle.STOP_PENDING
        )
        self.assertEqual(state.request(1).lifecycle, RequestLifecycle.STOPPED)
        self.assertEqual(
            state.request(0).terminal_reason,
            TerminalReason.DECODE_CREDIT_EXHAUSTED,
        )
        self.assertEqual(
            state.request(1).terminal_reason,
            TerminalReason.DECODE_CREDIT_EXHAUSTED,
        )
        self.assertEqual(state.objective.requests_stopped, 1)

        state = advance_to(state, config, 0.35).state

        self.assertEqual(state.request(0).lifecycle, RequestLifecycle.STOPPED)
        self.assertEqual(state.objective.requests_stopped, 2)
        self.assertEqual(state.decode_credits_available, 0)
        self.assertEqual(state.decode_credits_reserved, 0)
        self.assertEqual(state.decode_tokens_committed_total, 432)
        self.assertEqual(state.replica(0).rank_kv_committed_blocks, [0, 0])
        state.assert_valid(config)

    def test_early_stop_preserves_unused_pooled_decode_credit(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config)
        prefill = canonical_for_raw(
            state, config, controller_raw(config, "evict_none", 128)
        )
        state = apply_controller_action(
            state,
            config,
            prefill,
            stage_service_times=(0.1,),
            pp_communication_times=(),
            prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
        ).state
        state = advance_to(state, config, 0.1).state
        state = advance_to(state, config, 0.2).state

        stop_raw = adversary_raw(config, 0, None, "stop_longest_decode")
        stop = canonical_adversary_for_raw(state, config, stop_raw)
        state = apply_adversary_action(
            state,
            config,
            stop,
            prefill_time_estimator=lambda _tokens: 0.1,
        ).state

        self.assertEqual(state.request(0).lifecycle, RequestLifecycle.STOPPED)
        self.assertEqual(state.decode_credits_available, 216)
        self.assertEqual(state.decode_credits_reserved, 0)
        self.assertEqual(state.decode_tokens_committed_total, 0)
        self.assertEqual(state.decode_credits_minted_total, 216)
        state.assert_valid(config)

    def test_launch_window_masks_aggregate_prefill_over_7168(self) -> None:
        config = make_config()
        state = launch_requests(
            GV4State.initial(config), config, count=1, tokens=4096
        )
        wait = canonical_for_raw(state, config, 0)
        state = apply_controller_action(state, config, wait).state
        launch_another_4096 = adversary_raw(config, 1, 4096, "stop_none")

        self.assertIsNone(
            resolve_adversary_action(
                state, config, raw_action_index=launch_another_4096
            )
        )

    def test_launch_window_strictly_masks_an_eighth_request(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config, count=7)
        wait = canonical_for_raw(state, config, 0)
        state = apply_controller_action(state, config, wait).state
        launch_one = adversary_raw(config, 1, 128, "stop_none")
        self.assertIsNone(
            resolve_adversary_action(state, config, raw_action_index=launch_one)
        )


if __name__ == "__main__":
    unittest.main()
