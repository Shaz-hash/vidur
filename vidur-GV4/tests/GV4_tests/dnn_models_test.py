"""Focused tests for GV4 DeepSets inference, training, and artifacts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from AlphaGoZeroGV4.dnn_models import (  # noqa: E402
    GV4ModelError,
    GV4ModelSpec,
    GV4PolicyDeepSet,
    GV4ValueDeepSet,
    fit_policy_dnn,
    fit_value_dnn,
    load_dnn_model,
    save_dnn_model,
)
from GV4_Engine.action_resolver import (  # noqa: E402
    resolve_adversary_actions,
    resolve_controller_actions,
)
from GV4_Engine.dnn_inference.dnn_features import GV4FeatureBuilder  # noqa: E402
from GV4_Engine.dnn_inference.inference import GV4DNNInference  # noqa: E402
from GV4_Engine.state import GV4State, Player  # noqa: E402
from action_transition_test import launch_requests  # noqa: E402
from state_test import make_config  # noqa: E402


def _controller_fixture():
    config = make_config(pipeline_parallel_size=2)
    initial = GV4State.initial(config, next_player=Player.ADVERSARY)
    state = launch_requests(initial, config, count=2, tokens=128)
    _, actions = resolve_controller_actions(
        state,
        config,
        replica_id=0,
        prefill_time_estimator=lambda _request, tokens: tokens / 10_000.0,
    )
    builder = GV4FeatureBuilder(config)
    return (
        config,
        state,
        builder.build_state(state),
        tuple(actions),
        tuple(builder.build_controller_action(state, action) for action in actions),
    )


class _NativeMatrix:
    """Minimum shape exposed by the C++ FeatureMatrix pybind class."""

    def __init__(self, values: np.ndarray) -> None:
        matrix = np.asarray(values, dtype=np.float32)
        self.rows, self.columns = matrix.shape
        self.values = matrix.reshape(-1).tolist()


def _native_like_state(features):
    return SimpleNamespace(
        schema_version=features.schema_version,
        config_manifest_sha256=features.config_manifest_sha256,
        global_features=features.global_features.tolist(),
        request_rows=_NativeMatrix(features.request_rows),
        request_replica_offsets=features.request_replica_offsets.tolist(),
        launch_rows=_NativeMatrix(features.launch_rows),
        replica_rows=_NativeMatrix(features.replica_rows),
        microbatch_rows=_NativeMatrix(features.microbatch_rows),
        microbatch_replica_offsets=features.microbatch_replica_offsets.tolist(),
    )


class GV4DNNArchitectureTest(unittest.TestCase):
    def test_value_is_nonpositive_and_request_order_invariant(self) -> None:
        config, _, features, _, _ = _controller_fixture()
        torch.manual_seed(7)
        model = GV4ValueDeepSet(GV4ModelSpec.from_config(config), role="controller")
        with torch.no_grad():
            model.head_output.weight.normal_(mean=0.0, std=0.1)

        reordered = replace(
            features,
            request_rows=np.ascontiguousarray(features.request_rows[::-1]),
        )
        values = model.predict_structured((features, reordered))

        self.assertTrue(np.all(values <= 0.0))
        self.assertAlmostEqual(float(values[0]), float(values[1]), places=6)

    def test_policy_scores_actions_and_reuses_the_inference_facade(self) -> None:
        config, state, features, actions, _ = _controller_fixture()
        spec = GV4ModelSpec.from_config(config)
        value = GV4ValueDeepSet(spec, role="controller")
        policy = GV4PolicyDeepSet(spec, role="controller")
        inference = GV4DNNInference(
            config,
            controller_value_model=value,
            controller_policy_model=policy,
        )

        logits = inference.predict_controller_logits(
            state, actions, state_features=features
        )
        estimate = inference.predict_value(
            state, player="controller", state_features=features
        )

        self.assertEqual(logits.shape, (len(actions),))
        self.assertTrue(np.isfinite(logits).all())
        self.assertLessEqual(estimate, 0.0)

    def test_policy_logits_follow_action_permutation(self) -> None:
        config, _, state_features, _, action_features = _controller_fixture()
        model = GV4PolicyDeepSet(GV4ModelSpec.from_config(config), role="controller")
        forward = model.predict_root_structured(state_features, action_features)
        backward = model.predict_root_structured(
            state_features, tuple(reversed(action_features))
        )
        np.testing.assert_allclose(forward, backward[::-1], rtol=0.0, atol=1e-6)

    def test_native_feature_matrix_shape_is_accepted(self) -> None:
        config, _, features, _, _ = _controller_fixture()
        model = GV4ValueDeepSet(GV4ModelSpec.from_config(config), role="controller")
        python_value = model.predict_structured(features)
        native_value = model.predict_structured(_native_like_state(features))
        np.testing.assert_array_equal(python_value, native_value)

    def test_v1_rejects_multi_replica_config(self) -> None:
        with self.assertRaisesRegex(GV4ModelError, "exactly one replica"):
            GV4ModelSpec.from_config(make_config(num_replicas=2))


class GV4DNNTrainingTest(unittest.TestCase):
    def test_huber_value_fit_and_checkpoint_round_trip(self) -> None:
        config, _, features, _, _ = _controller_fixture()
        samples = (features, features)
        model, metrics = fit_value_dnn(
            samples,
            np.asarray([-1.0, -3.0], dtype=np.float32),
            config=config,
            role="controller",
            epochs=1,
            batch_size=2,
            torch_threads=1,
            device="cpu",
        )
        before = model.predict_structured(samples)

        self.assertTrue(np.isfinite(metrics["normalized_huber"]))
        self.assertIsNotNone(model.optimizer_state)
        self.assertEqual(model.training_metadata["rows"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller_value.pt"
            save_dnn_model(model, path)
            loaded = load_dnn_model(
                path,
                GV4ValueDeepSet,
                expected_config=config,
                expected_role="controller",
            )
            self.assertIsInstance(loaded, GV4ValueDeepSet)
            np.testing.assert_array_equal(before, loaded.predict_structured(samples))
            with self.assertRaisesRegex(GV4ModelError, "wrong player role"):
                load_dnn_model(path, expected_role="adversary")

            updated, warm_metrics = fit_value_dnn(
                samples,
                np.asarray([-1.0, -3.0], dtype=np.float32),
                config=config,
                role="controller",
                initial_model_path=path,
                epochs=1,
                batch_size=2,
                torch_threads=1,
                device="cpu",
            )
            self.assertEqual(warm_metrics["warm_start"], 1)
            self.assertTrue(updated.training_metadata["warm_start"])

    def test_visit_distribution_policy_fit(self) -> None:
        config, _, state, _, actions = _controller_fixture()
        selected = actions[:4]
        targets = np.zeros(len(selected), dtype=np.float32)
        targets[0] = 0.7
        targets[1:] = 0.3 / (len(selected) - 1)

        model, metrics = fit_policy_dnn(
            (state,),
            selected,
            targets,
            ((0, len(selected)),),
            config=config,
            role="controller",
            epochs=1,
            root_batch_size=1,
            torch_threads=1,
            device="cpu",
        )
        logits = model.predict_root_structured(state, selected)

        self.assertEqual(logits.shape, targets.shape)
        self.assertTrue(np.isfinite(metrics["cross_entropy"]))
        self.assertTrue(model.training_metadata["state_encoded_once_per_root"])

    def test_adversary_policy_uses_its_own_action_dimensions(self) -> None:
        config = make_config(pipeline_parallel_size=2)
        state = GV4State.initial(config, next_player=Player.ADVERSARY)
        _, actions = resolve_adversary_actions(state, config)
        builder = GV4FeatureBuilder(config)
        state_features = builder.build_state(state)
        action_features = tuple(
            builder.build_adversary_action(state, action) for action in actions[:3]
        )
        model = GV4PolicyDeepSet(GV4ModelSpec.from_config(config), role="adversary")

        logits = model.predict_root_structured(state_features, action_features)

        self.assertEqual(model.action_header_dim, 5)
        self.assertEqual(model.action_request_dim, 9)
        self.assertEqual(logits.shape, (len(action_features),))


if __name__ == "__main__":
    unittest.main()
