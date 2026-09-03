"""Exact deterministic transition parity between Python GV4 and native GV4."""

from __future__ import annotations

import math
from typing import Any

from GV4_Engine.GV4_Cpp import gv4_native as native
from GV4_Engine.GV4_Cpp.runtime import environment_from_python
from GV4_Engine.GV4_MCTS_Test.config import (
    GV4MCTSTestConfig,
    build_engine_config,
)
from GV4_Engine.GV4_MCTS_Test.timing import DeterministicTimingProvider
from GV4_Engine.state import Player
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment


def _enum_name(value: Any) -> str:
    return str(value.name)


def _allocation(item: Any) -> tuple[int, int, int, int]:
    return (
        int(item.request_id),
        int(item.prefill_tokens),
        int(item.decode_tokens),
        int(item.new_kv_blocks),
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
        str(action.eviction_rule),
        int(action.prefill_budget),
        str(action.ordering_heuristic),
        _enum_name(action.transition_kind),
        tuple(int(value) for value in action.evicted_request_ids),
        tuple(_allocation(value) for value in action.allocations),
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
