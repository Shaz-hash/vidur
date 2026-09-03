"""Focused tests for the GV4 feature and inference boundary."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from GV4_Engine.action_resolver import (  # noqa: E402
    CanonicalControllerAction,
    resolve_controller_actions,
)
from GV4_Engine.dnn_inference.dnn_features import (  # noqa: E402
    DNNFeatureError,
    GV4FeatureBuilder,
)
from GV4_Engine.dnn_inference.inference import (  # noqa: E402
    GV4DNNInference,
    GV4InferenceError,
)
from GV4_Engine.state import (  # noqa: E402
    GV4State,
    LaunchRecord,
    Player,
    RequestLifecycle,
    RequestState,
)
from action_transition_test import (  # noqa: E402
    adversary_raw,
    canonical_adversary_for_raw,
    canonical_for_raw,
    controller_raw,
    launch_requests,
)
from state_test import add_inflight_prefill, make_config  # noqa: E402


class _FakeValueModel:
    def __init__(self, config, role: str) -> None:
        self.role = role
        self.feature_schema_version = config.layout.feature_schema_version
        self.config_manifest_sha256 = config.manifest_sha256()
        self.calls = 0

    def predict_structured(self, states):
        self.calls += 1
        return np.asarray([-float(index + 1) for index in range(len(states))])


class _FakePolicyModel:
    def __init__(self, config, role: str) -> None:
        self.role = role
        self.feature_schema_version = config.layout.feature_schema_version
        self.config_manifest_sha256 = config.manifest_sha256()
        self.calls = 0
        self.last_state = None
        self.last_actions = ()

    def predict_root_structured(self, state, actions):
        self.calls += 1
        self.last_state = state
        self.last_actions = tuple(actions)
        return np.arange(len(actions), dtype=np.float32)


class GV4DNNFeatureTest(unittest.TestCase):
    def test_variable_rows_and_current_pipeline_occupancy(self) -> None:
        config = make_config(
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        add_inflight_prefill(state)
        state.now = 0.05
        state.next_adversary_tick = 0.2
        state.launch_history.append(LaunchRecord(0.0, 1, 256))
        state.requests.append(
            RequestState(
                request_id=1,
                owner_replica_id=0,
                lifecycle=RequestLifecycle.WAITING_PREFILL,
                arrival_time=0.04,
                prefill_deadline=1.0,
                decode_token_slo_sec=0.05,
                original_prefill_tokens=128,
                original_decode_tokens=16,
            )
        )
        state.next_request_id = 2
        state.objective.requests_generated = 2

        builder = GV4FeatureBuilder(config)
        features = builder.build_state(state)
        request_names = builder.layout.request_names
        microbatch_names = builder.layout.microbatch_names

        self.assertEqual(features.global_features.shape, (27,))
        self.assertEqual(features.request_rows.shape, (2, 32))
        self.assertEqual(features.launch_rows.shape, (1, 3))
        self.assertEqual(features.replica_rows.shape, (1, 9))
        self.assertEqual(features.microbatch_rows.shape, (1, 11))
        self.assertEqual(features.request_replica_offsets.tolist(), [0, 2])
        self.assertEqual(features.microbatch_replica_offsets.tolist(), [0, 1])
        self.assertEqual(
            features.request_rows[0, request_names.index("active_pipeline_stage_0")],
            1.0,
        )
        self.assertEqual(
            features.microbatch_rows[0, microbatch_names.index("active_stage_0")],
            1.0,
        )
        self.assertFalse(features.request_rows.flags.writeable)
        self.assertFalse(
            any(
                "finish" in name or "start_time" in name or "ready_time" in name
                for name in (
                    *builder.layout.request_names,
                    *builder.layout.replica_names,
                    *builder.layout.microbatch_names,
                )
            )
        )

    def test_replica_offsets_preserve_multi_replica_ownership(self) -> None:
        config = make_config(num_replicas=2)
        state = GV4State.initial(config)
        for request_id, replica_id in enumerate((0, 1, 1)):
            state.requests.append(
                RequestState(
                    request_id=request_id,
                    owner_replica_id=replica_id,
                    lifecycle=RequestLifecycle.WAITING_PREFILL,
                    arrival_time=0.0,
                    prefill_deadline=1.0,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=128,
                    original_decode_tokens=16,
                )
            )
        state.next_request_id = 3
        state.objective.requests_generated = 3

        features = GV4FeatureBuilder(config).build_state(state)

        self.assertEqual(features.request_rows.shape[0], 3)
        self.assertEqual(features.request_replica_offsets.tolist(), [0, 1, 3])
        self.assertEqual(features.microbatch_replica_offsets.tolist(), [0, 0, 0])
        self.assertEqual(features.replica_rows.shape[0], 2)

    def test_launch_rows_use_the_exact_open_window(self) -> None:
        config = make_config()
        state = GV4State.initial(config, now=1.2)
        state.launch_history[:] = [
            LaunchRecord(0.1, 1, 128),
            LaunchRecord(0.3, 2, 256),
        ]
        builder = GV4FeatureBuilder(config)

        features = builder.build_state(state)

        self.assertEqual(features.launch_rows.shape, (1, 3))
        self.assertAlmostEqual(
            float(features.launch_rows[0, 0]),
            np.arcsinh(0.9),
            places=6,
        )

        state.launch_history.append(LaunchRecord(1.3, 1, 128))
        with self.assertRaisesRegex(DNNFeatureError, "future-dated"):
            builder.build_state(state)

    def test_controller_alias_labels_do_not_change_physical_features(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config, tokens=128)
        raw_index = controller_raw(config, "evict_none", 128, "SJF")
        edge = canonical_for_raw(state, config, raw_index)
        relabeled = CanonicalControllerAction(
            canonical_action_index=999,
            action=edge.action,
            equivalent_raw_indices=(raw_index, raw_index + 1),
        )
        builder = GV4FeatureBuilder(config)

        first = builder.build_controller_action(state, edge)
        second = builder.build_controller_action(state, relabeled)

        np.testing.assert_array_equal(first.header, second.header)
        np.testing.assert_array_equal(
            first.affected_request_rows, second.affected_request_rows
        )
        self.assertEqual(first.affected_request_rows.shape, (1, 10))
        self.assertEqual(first.header[2], 1.0)  # BATCH one-hot position.

    def test_adversary_stop_action_identifies_the_concrete_decode(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.ADVERSARY)
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
                committed_decode_tokens=10,
                committed_kv_blocks=9,
                next_decode_deadline=0.05,
            )
        )
        state.next_request_id = 1
        state.objective.requests_generated = 1
        state.decode_credits_available = 206
        state.decode_credits_minted_total = 216
        state.decode_tokens_committed_total = 10
        state.replica(0).rank_kv_committed_blocks[:] = [9]
        raw_index = adversary_raw(config, 0, None, "stop_longest_decode")
        edge = canonical_adversary_for_raw(state, config, raw_index)

        features = GV4FeatureBuilder(config).build_adversary_action(state, edge)

        self.assertEqual(features.header.shape, (5,))
        self.assertEqual(features.affected_request_rows.shape, (1, 9))
        self.assertAlmostEqual(float(features.header[3]), 1.0 / 7.0)
        self.assertAlmostEqual(
            float(features.affected_request_rows[0, 1]), 10.0 / 864.0
        )


class GV4DNNInferenceTest(unittest.TestCase):
    def test_value_batch_and_all_policy_actions_use_one_call_each(self) -> None:
        config = make_config()
        state = launch_requests(GV4State.initial(config), config, tokens=128)
        _, canonical = resolve_controller_actions(
            state,
            config,
            replica_id=0,
            prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
        )
        value_model = _FakeValueModel(config, "controller")
        policy_model = _FakePolicyModel(config, "controller")
        inference = GV4DNNInference(
            config,
            controller_value_model=value_model,
            controller_policy_model=policy_model,
        )
        state_features = inference.build_state_features(state)

        value = inference.predict_value(
            state,
            player="controller",
            state_features=state_features,
        )
        logits = inference.predict_controller_logits(
            state,
            canonical,
            state_features=state_features,
        )

        self.assertEqual(value, -1.0)
        self.assertEqual(value_model.calls, 1)
        self.assertEqual(policy_model.calls, 1)
        self.assertIs(policy_model.last_state, state_features)
        self.assertEqual(len(policy_model.last_actions), len(canonical))
        np.testing.assert_array_equal(
            logits, np.arange(len(canonical), dtype=np.float32)
        )

    def test_model_schema_mismatch_fails_before_prediction(self) -> None:
        config = make_config()
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        model = _FakeValueModel(config, "controller")
        model.feature_schema_version = "wrong-schema"
        inference = GV4DNNInference(config, controller_value_model=model)

        with self.assertRaisesRegex(GV4InferenceError, "model schema"):
            inference.predict_value(state)
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
