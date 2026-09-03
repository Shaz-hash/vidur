"""Tests continuous candidate-checkpoint training for all four DNN artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vidur.AlphaGoZero import agz_train_eval_promote as trainer
from vidur.AlphaGoZero.dnn_models import (
    PolicyRankMLP,
    fit_policy_dnn,
    fit_value_dnn,
    save_dnn_model,
)


class IncrementalDNNTest(unittest.TestCase):
    def test_replay_sampling_seed_changes_with_candidate_version(self) -> None:
        self.assertEqual(trainer._candidate_sampling_seed(2026, 160), 2186)
        self.assertEqual(trainer._candidate_sampling_seed(2026, 161), 2187)
        self.assertNotEqual(
            trainer._candidate_sampling_seed(2026, 160),
            trainer._candidate_sampling_seed(2026, 161),
        )

    def test_all_four_models_continue_from_selected_parent_optimizer_state(self) -> None:
        rng = np.random.default_rng(7)
        value_x = rng.random((128, 226), dtype=np.float32)
        value_y = -40.0 * rng.random(128, dtype=np.float32)

        def policy_arrays(action_dim: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
            roots = 12
            actions = 4
            rows = np.empty((roots * actions, 226 + action_dim), dtype=np.float32)
            target = np.empty(roots * actions, dtype=np.float32)
            offsets: list[tuple[int, int]] = []
            for root in range(roots):
                begin = root * actions
                end = begin + actions
                offsets.append((begin, end))
                rows[begin:end, :226] = rng.random(226, dtype=np.float32)
                rows[begin:end, 226:] = rng.random((actions, action_dim), dtype=np.float32)
                p = rng.random(actions, dtype=np.float32)
                target[begin:end] = p / p.sum()
            return rows, target, offsets

        controller_x, controller_p, controller_offsets = policy_arrays(43)
        adversary_x, adversary_p, adversary_offsets = policy_arrays(7)

        with tempfile.TemporaryDirectory(prefix="agz_incremental_test_") as tmp:
            root = Path(tmp)
            controller_value, _ = fit_value_dnn(
                value_x, value_y, role="controller", epochs=1, batch_size=64, torch_threads=1
            )
            adversary_value, _ = fit_value_dnn(
                value_x, value_y, role="adversary", epochs=1, batch_size=64, torch_threads=1
            )
            controller_policy, _ = fit_policy_dnn(
                controller_x,
                controller_p,
                controller_offsets,
                role="controller",
                action_dim=43,
                epochs=1,
                root_batch_size=4,
                torch_threads=1,
            )
            adversary_policy, _ = fit_policy_dnn(
                adversary_x,
                adversary_p,
                adversary_offsets,
                role="adversary",
                action_dim=7,
                epochs=1,
                root_batch_size=4,
                torch_threads=1,
            )
            paths = {
                "controller_value": root / "controller_value.joblib",
                "adversary_value": root / "adversary_value.joblib",
                "controller_policy": root / "controller_policy.joblib",
                "adversary_policy": root / "adversary_policy.joblib",
            }
            save_dnn_model(controller_value, paths["controller_value"])
            save_dnn_model(adversary_value, paths["adversary_value"])
            save_dnn_model(controller_policy, paths["controller_policy"])
            save_dnn_model(adversary_policy, paths["adversary_policy"])

            promoted = trainer.ModelBundle(
                model_version=100,
                controller_model_version=100,
                adversary_model_version=100,
                controller_value_model_path=paths["controller_value"],
                adversary_value_model_path=paths["adversary_value"],
                controller_prior_model_path=paths["controller_policy"],
                adversary_prior_model_path=paths["adversary_policy"],
            )
            old = (
                trainer.AGZ_DNN_EPOCHS,
                trainer.AGZ_DNN_VALUE_BATCH_SIZE,
                trainer.AGZ_DNN_POLICY_ROOT_BATCH_SIZE,
                trainer.AGZ_DNN_TORCH_THREADS_PER_MODEL,
            )
            try:
                trainer.AGZ_DNN_EPOCHS = 1
                trainer.AGZ_DNN_VALUE_BATCH_SIZE = 64
                trainer.AGZ_DNN_POLICY_ROOT_BATCH_SIZE = 4
                trainer.AGZ_DNN_TORCH_THREADS_PER_MODEL = 1
                new_controller_value, new_adversary_value, _ = trainer._fit_dnn_value_models_parallel(
                    value_x,
                    value_y,
                    seed=9,
                    version=101,
                    training_parent=promoted,
                )
                new_controller_policy, new_adversary_policy, _ = trainer._fit_dnn_policy_models_parallel(
                    controller_x,
                    controller_p,
                    controller_offsets,
                    adversary_x,
                    adversary_p,
                    adversary_offsets,
                    seed=9,
                    version=101,
                    training_parent=promoted,
                )
            finally:
                (
                    trainer.AGZ_DNN_EPOCHS,
                    trainer.AGZ_DNN_VALUE_BATCH_SIZE,
                    trainer.AGZ_DNN_POLICY_ROOT_BATCH_SIZE,
                    trainer.AGZ_DNN_TORCH_THREADS_PER_MODEL,
                ) = old

            for model in (
                new_controller_value,
                new_adversary_value,
                new_controller_policy,
                new_adversary_policy,
            ):
                self.assertTrue(model.optimizer_state)
                self.assertTrue(model.training_metadata["incremental_update"])
                self.assertEqual(model.training_metadata["parent_model_version"], 100)
                self.assertEqual(len(model.training_metadata["parent_sha256"]), 64)
            self.assertEqual(new_controller_value.training_metadata["target_perspective"], "controller")
            self.assertEqual(new_adversary_value.training_metadata["target_perspective"], "controller")

    def test_latest_valid_candidate_precedes_promoted_parent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agz_candidate_lineage_test_") as tmp:
            root = Path(tmp)
            promoted = trainer.ModelBundle(
                model_version=130,
                controller_model_version=118,
                adversary_model_version=130,
                controller_value_model_path=root / "promoted_controller_value.joblib",
                adversary_value_model_path=root / "promoted_adversary_value.joblib",
                controller_prior_model_path=root / "promoted_controller_prior.joblib",
                adversary_prior_model_path=root / "promoted_adversary_prior.joblib",
            )
            fallback = trainer._latest_dnn_training_parent_bundle(
                root,
                next_version=131,
                promoted=promoted,
            )
            self.assertEqual(fallback.controller_model_version, 118)
            self.assertEqual(fallback.adversary_model_version, 130)

            for version, native_ready in ((131, True), (132, False)):
                model_dir = root / "models" / f"Model_Version{version}"
                bundle = trainer._candidate_bundle_from_output(version, model_dir)
                for path in (
                    bundle.controller_value_model_path,
                    bundle.adversary_value_model_path,
                    bundle.controller_prior_model_path,
                    bundle.adversary_prior_model_path,
                ):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                (model_dir / "candidate_manifest.json").write_text(
                    json.dumps({
                        "model_version": version,
                        "model_family": "dnn",
                        "native_ready": native_ready,
                    }),
                    encoding="utf-8",
                )

            selected = trainer._latest_dnn_training_parent_bundle(
                root,
                next_version=133,
                promoted=promoted,
            )
            self.assertEqual(selected.controller_model_version, 131)
            self.assertEqual(selected.adversary_model_version, 131)
            self.assertIn("Model_Version131", str(selected.controller_value_model_path))
    def test_missing_parent_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing DNN training parent"):
            trainer._require_incremental_dnn_parent(
                Path("/definitely/missing/model.joblib"),
                expected_type=PolicyRankMLP,
                label="controller_prior",
            )


if __name__ == "__main__":
    unittest.main()
