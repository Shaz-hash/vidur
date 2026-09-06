"""Exact deterministic transition parity between Python GV4 and native GV4."""

from __future__ import annotations

import math
from typing import Any

from GV4_Cpp import gv4_native as native
from GV4_Cpp.runtime import config_from_python, environment_from_python
from GV4_Engine.GV4_MCTS_Test.config import (
    GV4MCTSTestConfig,
    build_engine_config,
)
from GV4_Engine.GV4_MCTS_Test.timing import DeterministicTimingProvider
from GV4_Engine.kv_ledger import preempt_request_blocks
from GV4_Engine.state import Player
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment


def _enum_name(value: Any) -> str:
    return str(value.name)


def _allocation(item: Any) -> tuple[int, int, int, int, int]:
    return (
        int(item.request_id),
        int(item.prefill_tokens),
        int(item.decode_tokens),
        int(item.new_kv_blocks),
        int(item.recompute_tokens),
    )


def _request(item: Any) -> tuple[Any, ...]:
    return (
        int(item.request_id),
        int(item.owner_replica_id),
        _enum_name(item.lifecycle),
        float(item.arrival_time),
        float(item.prefill_deadline),
        float(item.decode_token_slo_sec),
        int(item.original_prefill_tokens),
        int(item.original_decode_tokens),
        bool(item.decode_credit_minted),
        int(item.committed_prefill_tokens),
        int(item.reserved_prefill_tokens),
        int(item.committed_decode_tokens),
        int(item.reserved_decode_tokens),
        int(item.kv_computed_tokens),
        int(item.reserved_recompute_tokens),
        int(item.committed_kv_blocks),
        int(item.reserved_kv_blocks),
        int(item.inflight_microbatch_id),
        float(item.next_decode_deadline),
        float(item.prefill_lateness_sec),
        float(item.decode_lateness_sec),
        bool(item.violation_recorded),
        _enum_name(item.terminal_reason),
        float(item.terminal_requested_at),
        float(item.terminal_time),
    )


def _batch(item: Any) -> tuple[Any, ...]:
    return (
        int(item.microbatch_id),
        int(item.replica_id),
        int(item.raw_action_index),
        int(item.canonical_action_index),
        tuple(_allocation(value) for value in item.allocations),
        tuple(float(value) for value in item.stage_ready_times),
        tuple(float(value) for value in item.stage_start_times),
        tuple(float(value) for value in item.stage_finish_times),
        bool(item.completion_applied),
    )


def _objective(item: Any) -> tuple[Any, ...]:
    return (
        int(item.requests_generated),
        int(item.requests_completed),
        int(item.requests_stopped),
        int(item.requests_dropped),
        int(item.slo_violations),
        float(item.prefill_lateness_sec),
        float(item.decode_lateness_sec),
        float(item.terminal_cost),
        float(item.total_cost),
    )


def _state_fingerprint(state: Any, *, python_state: bool) -> tuple[Any, ...]:
    replica = state.replicas[0] if python_state else state.replica
    return (
        str(state.state_schema_version),
        str(state.config_manifest_sha256),
        float(state.now),
        _enum_name(state.next_player),
        float(state.next_adversary_tick),
        int(state.next_request_id),
        int(state.next_microbatch_id),
        int(state.tie_break_counter),
        int(state.rng_seed),
        int(state.rng_counter),
        tuple(
            (float(item.launch_time), int(item.request_count), int(item.prefill_tokens))
            for item in state.launch_history
        ),
        int(state.decode_credits_available),
        int(state.decode_credits_reserved),
        int(state.decode_credits_minted_total),
        int(state.decode_tokens_committed_total),
        tuple(_request(item) for item in state.requests),
        (
            int(replica.replica_id),
            tuple(int(value) for value in replica.rank_ids),
            tuple(int(value) for value in replica.rank_kv_capacity_blocks),
            tuple(int(value) for value in replica.rank_kv_committed_blocks),
            tuple(int(value) for value in replica.rank_kv_reserved_blocks),
            tuple(float(value) for value in replica.stage_tail_finish_times),
            tuple(int(value) for value in replica.stage_last_microbatch_ids),
            tuple(_batch(item) for item in replica.inflight_microbatches),
        ),
        _objective(state.objective),
    )


def _assert_nested_close(left: Any, right: Any, path: str = "state") -> None:
    if isinstance(left, tuple):
        assert isinstance(right, tuple), f"{path}: type mismatch"
        assert len(left) == len(right), f"{path}: length mismatch"
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            _assert_nested_close(lhs, rhs, f"{path}[{index}]")
        return
    if isinstance(left, float):
        assert isinstance(right, float), f"{path}: type mismatch"
        assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10), (
            f"{path}: Python={left!r}, native={right!r}"
        )
        return
    assert left == right, f"{path}: Python={left!r}, native={right!r}"


def _controller_signature(edge: Any) -> tuple[Any, ...]:
    action = edge.action
    return (
        int(edge.canonical_action_index),
        int(edge.representative_raw_index),
        tuple(int(value) for value in edge.equivalent_raw_indices),
        int(action.replica_id),
        str(action.preemption_rule),
        str(action.eviction_rule),
        int(action.prefill_budget),
        str(action.ordering_heuristic),
        _enum_name(action.transition_kind),
        tuple(int(value) for value in action.evicted_request_ids),
        tuple(int(value) for value in action.preempted_request_ids),
        tuple(int(value) for value in action.pending_preemption_request_ids),
        tuple(_allocation(value) for value in action.allocations),
        int(action.preempted_kv_blocks),
        int(action.released_kv_blocks),
        int(action.reserved_kv_blocks),
        tuple(tuple(int(part) for part in value) for value in action.rank_kv_delta),
    )


def _adversary_signature(edge: Any) -> tuple[Any, ...]:
    action = edge.action
    return (
        int(edge.canonical_action_index),
        int(edge.representative_raw_index),
        tuple(int(value) for value in edge.equivalent_raw_indices),
        int(action.launch_count),
        int(action.prefill_tokens or 0),
        str(action.stop_rule),
        tuple(int(value) for value in action.stop_request_ids),
    )


def _native_edge(space: Any, raw_index: int) -> Any:
    canonical = int(space.raw_to_canonical[raw_index])
    assert canonical >= 0
    return space.canonical_actions[canonical]


def _select_controller_raw(actions: list[Any | None]) -> int:
    representatives = [
        edge
        for index, edge in enumerate(actions)
        if edge is not None and edge.representative_raw_index == index
    ]
    batches = [
        edge for edge in representatives if _enum_name(edge.action.transition_kind) == "BATCH"
    ]
    if batches:
        return max(
            batches,
            key=lambda edge: (
                edge.action.total_prefill_tokens,
                edge.action.total_decode_tokens,
                -edge.representative_raw_index,
            ),
        ).representative_raw_index
    return representatives[0].representative_raw_index


def _select_adversary_raw(actions: list[Any | None], step: int) -> int:
    representatives = [
        edge
        for index, edge in enumerate(actions)
        if edge is not None and edge.representative_raw_index == index
    ]
    launches = [edge for edge in representatives if edge.action.launch_count]
    if launches and step % 5 == 0:
        return max(
            launches,
            key=lambda edge: (
                edge.action.launch_count,
                edge.action.prefill_tokens or 0,
                -edge.representative_raw_index,
            ),
        ).representative_raw_index
    return representatives[0].representative_raw_index


def test_scripted_engine_and_action_parity() -> None:
    test = GV4MCTSTestConfig(timing_mode="deterministic", seed=7)
    config = build_engine_config(test)
    timing = DeterministicTimingProvider(config.topology.pipeline_parallel_size)
    python_env = GV4VirtualVidurMCTSEnvironment(
        config,
        batch_timing_provider=timing,
        prefill_time_estimator=timing.estimate_prefill_time,
    )
    native_env = environment_from_python(config, timing)

    python_state = python_env.initial_state(now=0.0, next_player=Player.ADVERSARY)
    native_state = native_env.initial_state(0.0, native.Player.ADVERSARY)

    for step in range(80):
        _assert_nested_close(
            _state_fingerprint(python_state, python_state=True),
            _state_fingerprint(native_state, python_state=False),
        )

        if python_state.next_player == Player.ADVERSARY:
            python_actions, python_mask = python_env.sample_adversary_actions(python_state)
            native_space = native_env.sample_adversary_actions(native_state)
            assert list(python_mask) == [value >= 0 for value in native_space.raw_to_canonical]
            for raw_index, edge in enumerate(python_actions):
                if edge is not None:
                    assert _adversary_signature(edge) == _adversary_signature(
                        _native_edge(native_space, raw_index)
                    )
            raw_index = _select_adversary_raw(list(python_actions), step)
            python_state = python_env.apply_adversary_action_only(
                python_state, python_actions[raw_index]
            )
            native_state = native_env.apply_adversary_action_only(
                native_state, _native_edge(native_space, raw_index)
            )
        else:
            python_actions, python_mask = python_env.sample_controller_actions(
                python_state, replica_id=0
            )
            native_space = native_env.sample_controller_actions(native_state)
            assert list(python_mask) == [value >= 0 for value in native_space.raw_to_canonical]
            for raw_index, edge in enumerate(python_actions):
                if edge is not None:
                    assert _controller_signature(edge) == _controller_signature(
                        _native_edge(native_space, raw_index)
                    )
            raw_index = _select_controller_raw(list(python_actions))
            python_state = python_env.apply_controller_action_only(
                python_state, python_actions[raw_index], fast_forward=True
            )
            native_state = native_env.apply_controller_action_only(
                native_state, _native_edge(native_space, raw_index), True
            )

    _assert_nested_close(
        _state_fingerprint(python_state, python_state=True),
        _state_fingerprint(native_state, python_state=False),
    )


def test_native_memory_action_with_decode_batch_is_zero_time() -> None:
    """A native eviction boundary must not trigger hidden decode progression."""

    test = GV4MCTSTestConfig(timing_mode="deterministic", seed=7)
    config = build_engine_config(test)
    timing = DeterministicTimingProvider(config.topology.pipeline_parallel_size)
    native_config = config_from_python(config)
    environment = environment_from_python(config, timing)
    state = environment.initial_state(0.0, native.Player.CONTROLLER)

    requests = []
    for request_id in range(2):
        request = native.RequestState()
        request.request_id = request_id
        request.owner_replica_id = 0
        request.lifecycle = native.RequestLifecycle.WAITING_DECODE
        request.arrival_time = 0.0
        request.prefill_deadline = 0.1
        request.decode_token_slo_sec = 0.05
        request.original_prefill_tokens = 128
        request.original_decode_tokens = 864
        request.decode_credit_minted = True
        request.committed_prefill_tokens = 128
        request.kv_computed_tokens = 128
        request.committed_kv_blocks = 8
        request.next_decode_deadline = 0.05
        requests.append(request)

    state.requests = requests
    state.next_request_id = 2
    state.decode_credits_available = 432
    state.decode_credits_minted_total = 432
    replica = state.replica
    replica.rank_kv_committed_blocks = [16] * len(replica.rank_ids)
    state.replica = replica
    objective = state.objective
    objective.requests_generated = 2
    state.objective = objective
    state.validate(native_config)

    action_config = config.controller_actions
    raw_index = action_config.encode_raw_index(
        action_config.eviction_rule_names.index("evict_longest_decode"),
        0,
        0,
    )
    action_space = environment.sample_controller_actions(state)
    action = _native_edge(action_space, raw_index)
    assert action.action.transition_kind == native.ControllerTransitionKind.BATCH
    assert list(action.action.evicted_request_ids) == [0]
    assert len(action.action.allocations) == 1

    child = environment.apply_controller_action_only(state, action, True)

    assert child.now == state.now
    assert child.next_player == native.Player.ADVERSARY
    assert child.requests[0].lifecycle == native.RequestLifecycle.DROPPED
    assert child.requests[1].lifecycle == native.RequestLifecycle.INFLIGHT_DECODE
    child.validate(native_config)


def test_preemption_primitive_has_exact_python_native_parity() -> None:
    test = GV4MCTSTestConfig(timing_mode="deterministic", seed=11)
    config = build_engine_config(test)
    timing = DeterministicTimingProvider(config.topology.pipeline_parallel_size)
    python_env = GV4VirtualVidurMCTSEnvironment(
        config,
        batch_timing_provider=timing,
        prefill_time_estimator=timing.estimate_prefill_time,
    )
    native_env = environment_from_python(config, timing)
    python_state = python_env.initial_state(now=0.0, next_player=Player.ADVERSARY)
    native_state = native_env.initial_state(0.0, native.Player.ADVERSARY)

    request_id = None
    for step in range(80):
        candidates = [
            request.request_id
            for request in python_state.requests
            if not request.has_inflight_work
            and not request.lifecycle.is_terminal
            and request.committed_kv_blocks > 0
        ]
        if candidates:
            request_id = candidates[0]
            break

        if python_state.next_player == Player.ADVERSARY:
            python_actions, _ = python_env.sample_adversary_actions(python_state)
            native_space = native_env.sample_adversary_actions(native_state)
            raw_index = _select_adversary_raw(list(python_actions), step)
            python_state = python_env.apply_adversary_action_only(
                python_state, python_actions[raw_index]
            )
            native_state = native_env.apply_adversary_action_only(
                native_state, _native_edge(native_space, raw_index)
            )
        else:
            python_actions, _ = python_env.sample_controller_actions(
                python_state, replica_id=0
            )
            native_space = native_env.sample_controller_actions(native_state)
            raw_index = _select_controller_raw(list(python_actions))
            python_state = python_env.apply_controller_action_only(
                python_state, python_actions[raw_index], fast_forward=True
            )
            native_state = native_env.apply_controller_action_only(
                native_state, _native_edge(native_space, raw_index), True
            )

    assert request_id is not None, "script never produced preemptible resident KV"
    python_request = python_state.request(request_id)
    logical_progress = python_request.logical_context_tokens
    python_released = preempt_request_blocks(
        python_request, python_state.replica(python_request.owner_replica_id)
    )
    native_released = native.preempt_request_blocks(native_state, request_id)

    assert python_released == native_released
    assert python_request.logical_context_tokens == logical_progress
    assert python_request.kv_computed_tokens == 0
    assert python_request.remaining_recompute_tokens == logical_progress
    python_state.assert_valid(config)
    native_state.validate(config_from_python(config))
    _assert_nested_close(
        _state_fingerprint(python_state, python_state=True),
        _state_fingerprint(native_state, python_state=False),
    )


def test_preemption_actions_and_recovery_have_exact_python_native_parity() -> None:
    """Exercise deferred drain, immediate release, and mixed recovery batching."""

    test = GV4MCTSTestConfig(timing_mode="deterministic", seed=7)
    config = build_engine_config(test)
    timing = DeterministicTimingProvider(config.topology.pipeline_parallel_size)
    python_env = GV4VirtualVidurMCTSEnvironment(
        config,
        batch_timing_provider=timing,
        prefill_time_estimator=timing.estimate_prefill_time,
    )
    native_env = environment_from_python(config, timing)
    python_state = python_env.initial_state(now=0.0, next_player=Player.ADVERSARY)
    native_state = native_env.initial_state(0.0, native.Player.ADVERSARY)

    def assert_state_parity() -> None:
        _assert_nested_close(
            _state_fingerprint(python_state, python_state=True),
            _state_fingerprint(native_state, python_state=False),
        )

    def apply_adversary(raw_index: int) -> None:
        nonlocal python_state, native_state
        python_actions, _ = python_env.sample_adversary_actions(python_state)
        native_space = native_env.sample_adversary_actions(native_state)
        python_edge = python_actions[raw_index]
        assert python_edge is not None
        native_edge = _native_edge(native_space, raw_index)
        assert _adversary_signature(python_edge) == _adversary_signature(native_edge)
        python_state = python_env.apply_adversary_action_only(
            python_state, python_edge
        )
        native_state = native_env.apply_adversary_action_only(
            native_state, native_edge
        )
        assert_state_parity()

    def apply_controller(raw_index: int) -> Any:
        nonlocal python_state, native_state
        python_actions, _ = python_env.sample_controller_actions(
            python_state, replica_id=0
        )
        native_space = native_env.sample_controller_actions(native_state)
        python_edge = python_actions[raw_index]
        assert python_edge is not None
        native_edge = _native_edge(native_space, raw_index)
        assert _controller_signature(python_edge) == _controller_signature(native_edge)
        python_state = python_env.apply_controller_action_only(
            python_state, python_edge, fast_forward=False
        )
        native_state = native_env.apply_controller_action_only(
            native_state, native_edge, False
        )
        assert_state_parity()
        return python_edge

    launch_actions, _ = python_env.sample_adversary_actions(python_state)
    apply_adversary(_select_adversary_raw(list(launch_actions), 0))

    controller_actions, _ = python_env.sample_controller_actions(
        python_state, replica_id=0
    )
    apply_controller(_select_controller_raw(list(controller_actions)))

    # Expose the controller again while the first microbatch is still in flight.
    apply_adversary(0)
    preempt_raw = config.controller_actions.encode_raw_index(
        0, 0, 0, preemption_rule_index=1
    )
    pending_edge = apply_controller(preempt_raw)
    pending_id = pending_edge.action.preempted_request_ids[0]
    assert pending_edge.action.pending_preemption_request_ids == (pending_id,)
    assert python_state.request(pending_id).lifecycle.name == "PREEMPT_PENDING"

    # WAIT may expose stage-zero release before final-stage completion.
    while python_state.request(pending_id).has_inflight_work:
        apply_adversary(0)
        wait_edge = apply_controller(0)
        assert wait_edge.action.transition_kind.name == "WAIT"

    # At final-stage completion, work commits first and KV is then released.
    drained = python_state.request(pending_id)
    assert drained.committed_prefill_tokens == drained.original_prefill_tokens
    assert drained.kv_computed_tokens == 0
    assert drained.remaining_recompute_tokens == drained.logical_context_tokens
    assert drained.decode_credit_minted is True

    # A resident waiting request is released immediately by the same raw policy.
    apply_adversary(0)
    immediate_edge = apply_controller(preempt_raw)
    immediate_id = immediate_edge.action.preempted_request_ids[0]
    assert immediate_edge.action.pending_preemption_request_ids == ()
    immediate = python_state.request(immediate_id)
    assert immediate.kv_computed_tokens == 0
    assert immediate.remaining_recompute_tokens == immediate.logical_context_tokens

    # Recovery and ordinary prefill share one budget and one physical microbatch.
    apply_adversary(0)
    budget_index = config.controller_actions.prefill_budget_options.index(4096)
    recovery_raw = config.controller_actions.encode_raw_index(0, budget_index, 0)
    while True:
        recovery_actions, _ = python_env.sample_controller_actions(
            python_state, replica_id=0
        )
        if recovery_actions[recovery_raw] is not None:
            break
        wait_edge = apply_controller(0)
        assert wait_edge.action.transition_kind.name == "WAIT"
        apply_adversary(0)
    recovery_edge = apply_controller(recovery_raw)
    allocations = recovery_edge.action.allocations
    assert any(item.recompute_tokens for item in allocations)
    assert any(item.prefill_tokens for item in allocations)
    assert all(
        not (item.recompute_tokens and (item.prefill_tokens or item.decode_tokens))
        for item in allocations
    )
