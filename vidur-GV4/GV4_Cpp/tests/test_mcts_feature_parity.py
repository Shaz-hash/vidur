"""Permanent feature, inference, and uniform-MCTS parity checks."""

from __future__ import annotations

from dataclasses import asdict
import importlib
import random
from typing import Any

import numpy as np

from GV4_Engine.GV4_Cpp import gv4_native as native
from GV4_Engine.GV4_Cpp.runtime import config_from_python, environment_from_python
from GV4_Engine.GV4_Cpp.uniform_parity import compare_uniform_search
from GV4_Engine.GV4_MCTS_Test.config import GV4MCTSTestConfig, build_engine_config
from GV4_Engine.GV4_MCTS_Test.timing import DeterministicTimingProvider
from GV4_Engine.dnn_inference.dnn_features import GV4FeatureBuilder
from GV4_Engine.state import Player
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment

from .test_engine_parity import _select_adversary_raw, _select_controller_raw


def _assert_vector(python_values: Any, native_values: Any) -> None:
    python_array = np.asarray(python_values, dtype=np.float32).reshape(-1)
    native_array = np.asarray(native_values, dtype=np.float32).reshape(-1)
    assert python_array.tobytes() == native_array.tobytes()


def _assert_matrix(python_values: Any, native_values: native.FeatureMatrix) -> None:
    python_array = np.asarray(python_values, dtype=np.float32)
    assert python_array.shape == (native_values.rows, native_values.columns)
    _assert_vector(python_array, native_values.values)


def _assert_state_features(python_features: Any, native_features: Any) -> None:
    assert python_features.schema_version == native_features.schema_version
    assert (
        python_features.config_manifest_sha256
        == native_features.config_manifest_sha256
    )
    _assert_vector(python_features.global_features, native_features.global_features)
    _assert_matrix(python_features.request_rows, native_features.request_rows)
    _assert_matrix(python_features.launch_rows, native_features.launch_rows)
    _assert_matrix(python_features.replica_rows, native_features.replica_rows)
    _assert_matrix(python_features.microbatch_rows, native_features.microbatch_rows)
    assert python_features.request_replica_offsets.tolist() == list(
        native_features.request_replica_offsets
    )
    assert python_features.microbatch_replica_offsets.tolist() == list(
        native_features.microbatch_replica_offsets
    )


def _assert_action_features(python_features: Any, native_features: Any) -> None:
    _assert_vector(python_features.header, native_features.header)
    _assert_matrix(
        python_features.affected_request_rows,
        native_features.affected_request_rows,
    )


def _environments() -> tuple[Any, Any, Any, Any]:
    test = GV4MCTSTestConfig(timing_mode="deterministic", seed=7)
    config = build_engine_config(test)
    timing = DeterministicTimingProvider(config.topology.pipeline_parallel_size)
    python_environment = GV4VirtualVidurMCTSEnvironment(
        config,
        batch_timing_provider=timing,
        prefill_time_estimator=timing.estimate_prefill_time,
    )
    return config, timing, python_environment, environment_from_python(config, timing)


def test_native_features_match_python_bit_for_bit() -> None:
    config, _, python_environment, native_environment = _environments()
    python_builder = GV4FeatureBuilder(config)
    native_builder = native.FeatureBuilder(config_from_python(config))
    python_state = python_environment.initial_state(
        now=0.0,
        next_player=Player.ADVERSARY,
    )
    native_state = native_environment.initial_state(0.0, native.Player.ADVERSARY)

    assert tuple(python_builder.layout.global_names) == tuple(
        native_builder.layout.global_names
    )
    assert tuple(python_builder.layout.request_names) == tuple(
        native_builder.layout.request_names
    )

    for step in range(40):
        _assert_state_features(
            python_builder.build_state(python_state),
            native_builder.build_state(native_state),
        )
        if python_state.next_player == Player.ADVERSARY:
            python_actions, _ = python_environment.sample_adversary_actions(
                python_state
            )
            native_space = native_environment.sample_adversary_actions(native_state)
            for native_edge in native_space.canonical_actions:
                python_edge = python_actions[native_edge.representative_raw_index]
                _assert_action_features(
                    python_builder.build_adversary_action(python_state, python_edge),
                    native_builder.build_adversary_action(native_state, native_edge),
                )
            raw_index = _select_adversary_raw(list(python_actions), step)
            canonical_index = native_space.raw_to_canonical[raw_index]
            python_state = python_environment.apply_adversary_action_only(
                python_state,
                python_actions[raw_index],
            )
            native_state = native_environment.apply_adversary_action_only(
                native_state,
                native_space.canonical_actions[canonical_index],
            )
        else:
            python_actions, _ = python_environment.sample_controller_actions(
                python_state,
                replica_id=0,
            )
            native_space = native_environment.sample_controller_actions(native_state)
            for native_edge in native_space.canonical_actions:
                python_edge = python_actions[native_edge.representative_raw_index]
                _assert_action_features(
                    python_builder.build_controller_action(python_state, python_edge),
                    native_builder.build_controller_action(native_state, native_edge),
                )
            raw_index = _select_controller_raw(list(python_actions))
            canonical_index = native_space.raw_to_canonical[raw_index]
            python_state = python_environment.apply_controller_action_only(
                python_state,
                python_actions[raw_index],
                fast_forward=True,
            )
            native_state = native_environment.apply_controller_action_only(
                native_state,
                native_space.canonical_actions[canonical_index],
                True,
            )


def test_uniform_root_visits_match_at_100_and_1000_iterations() -> None:
    _, _, python_environment, native_environment = _environments()
    for iterations in (100, 1000):
        report = compare_uniform_search(
            python_environment,
            native_environment,
            iterations=iterations,
            seed=7,
        )
        report.assert_exact()


class _ValueModel:
    def __init__(self, config: Any, role: str) -> None:
        self.role = role
        self.feature_schema_version = config.feature_schema_version
        self.config_manifest_sha256 = config.manifest_sha256

    def predict_structured(self, states: Any) -> np.ndarray:
        return -np.arange(1, len(states) + 1, dtype=np.float32)


class _PolicyModel:
    def __init__(self, config: Any, role: str) -> None:
        self.role = role
        self.feature_schema_version = config.feature_schema_version
        self.config_manifest_sha256 = config.manifest_sha256

    def predict_root_structured(self, _state: Any, actions: Any) -> np.ndarray:
        return np.arange(len(actions), dtype=np.float32)


def test_native_inference_checks_schema_and_batches_structured_features() -> None:
    config, _, _, native_environment = _environments()
    native_config = config_from_python(config)
    value = _ValueModel(native_config, "adversary")
    policy = _PolicyModel(native_config, "adversary")
    inference = native.InferenceRuntime(
        native_config,
        None,
        value,
        None,
        policy,
    )
    state = native_environment.initial_state(0.0, native.Player.ADVERSARY)
    actions = native_environment.sample_adversary_actions(state).canonical_actions

    assert inference.predict_values(
        [state, state], native.Player.ADVERSARY
    ) == [-1.0, -2.0]
    assert inference.predict_adversary_logits(state, actions) == list(
        np.arange(len(actions), dtype=np.float32)
    )

    value.feature_schema_version = "wrong-schema"
    try:
        native.InferenceRuntime(native_config, None, value, None, policy)
    except ValueError as error:
        assert "schema mismatch" in str(error)
    else:
        raise AssertionError("native inference accepted a mismatched feature schema")


def _root_rows(result: Any) -> list[tuple[int, int, float]]:
    return [
        (
            int(row["representative_raw_index"]),
            int(row["visits"]),
            float(row["value_sum"]),
        )
        for row in result["root_action_stats"]
    ]


def test_zero_rollout_count_preserves_uniform_mcts_exactly() -> None:
    _, _, _, native_environment = _environments()
    state = native_environment.initial_state(0.0, native.Player.ADVERSARY)
    uniform = native.run_uniform_mcts(
        native_environment,
        state,
        native.Player.ADVERSARY,
        iterations=100,
    )
    disabled = native.run_policy_rollout_mcts(
        native_environment,
        state,
        native.Player.ADVERSARY,
        iterations=100,
        rollout_count=0,
    )

    assert disabled["used_rollout"] is False
    assert disabled["rollout_stats"]["leaf_evaluations"] == 0
    assert disabled["best_action_index"] == uniform["best_action_index"]
    assert disabled["action_values"] == uniform["action_values"]
    assert _root_rows(disabled) == _root_rows(uniform)


def test_seeded_native_rollout_matches_python_exactly() -> None:
    _, _, python_environment, native_environment = _environments()
    base = importlib.import_module("vidur-GV4.mcts_value_prior")
    rollout = importlib.import_module("vidur-GV4.mcts_value_prior_rollout")

    python_config = base.MCTSConfig()
    python_config.rng = random.Random(7)
    python_config.use_policy_prior = False
    python_config.rollout_horizon_sec = 0.05
    python_config.rollout_count = 2
    python_config.rollout_max_actions = 64
    python_config.rollout_seed = 11
    python_mcts = rollout.VidurMCTSPolicyRollout(
        python_environment,
        python_config,
    )
    python_result = python_mcts.search_dnn(
        None,
        python_environment.initial_state(now=0.0, next_player=Player.ADVERSARY),
        "adversary",
        game_id=1,
        root_id=2,
        root_node_id_override=0,
        root_depth=0,
        mcts_iter=12,
        model_version=0,
        use_model_bootstrap=False,
    )

    native_result = native.run_policy_rollout_mcts(
        native_environment,
        native_environment.initial_state(0.0, native.Player.ADVERSARY),
        native.Player.ADVERSARY,
        iterations=12,
        rollout_count=2,
        rollout_horizon_sec=0.05,
        rollout_seed=11,
        rollout_max_actions=64,
    )
    python_rows = sorted(
        (index, child.visits, child.value_sum)
        for index, child in python_mcts._root.children.items()
    )

    assert native_result["used_rollout"] is True
    assert native_result["best_action_index"] == python_result.best_action_index
    assert _root_rows(native_result) == python_rows
    assert native_result["rollout_stats"] == asdict(python_mcts.rollout_stats)
    assert native_result["rollout_stats"]["actions"] > 0
    assert (
        native_result["rollout_stats"]["min_final_time"]
        >= native_result["rollout_stats"]["min_deadline"]
    )


def test_rollout_can_use_native_policy_and_value_inference() -> None:
    config, _, _, native_environment = _environments()
    native_config = config_from_python(config)
    inference = native.InferenceRuntime(
        native_config,
        _ValueModel(native_config, "controller"),
        _ValueModel(native_config, "adversary"),
        _PolicyModel(native_config, "controller"),
        _PolicyModel(native_config, "adversary"),
    )
    state = native_environment.initial_state(0.0, native.Player.ADVERSARY)
    result = native.run_policy_rollout_mcts(
        native_environment,
        state,
        native.Player.ADVERSARY,
        inference=inference,
        iterations=3,
        rollout_count=2,
        rollout_horizon_sec=0.05,
        rollout_seed=17,
        rollout_max_actions=64,
        use_policy_prior=True,
        use_model_bootstrap=True,
    )

    assert result["used_rollout"] is True
    assert result["used_bootstrap"] is True
    assert result["rollout_stats"]["leaf_evaluations"] == 3
    assert result["rollout_stats"]["trajectories"] == 6
    assert result["rollout_stats"]["bootstrap_calls"] == 6
    assert all(row["value_sum"] <= 0.0 for row in result["root_action_stats"])
