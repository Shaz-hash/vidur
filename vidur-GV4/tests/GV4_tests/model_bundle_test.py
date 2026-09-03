"""Tests for immutable GV4 model bundles and model-backed game cycles."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


GV4_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_ROOT = GV4_ROOT.parent
sys.path.insert(0, str(CLASSICAL_ROOT))
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from AlphaGoZeroGV4.bootstrap_untrained_dnn_v100 import (  # noqa: E402
    bootstrap_bundle,
)
from AlphaGoZeroGV4.engine_runtime import (  # noqa: E402
    SearchConfig,
    SearchContext,
    create_engine_runtime,
)
from AlphaGoZeroGV4.model_bundle import (  # noqa: E402
    MODEL_ARTIFACT_NAMES,
    ModelBundleError,
    load_model_bundle,
)
from AlphaGoZeroGV4.replay_runtime import GV4ReplayRecorder  # noqa: E402
from AlphaGoZeroGV4.runner import GameCycleConfig, run_game_cycle  # noqa: E402
from GV4_Engine.action_resolver import resolve_adversary_actions  # noqa: E402
from GV4_Engine.GV4_MCTS_Test.timing import (  # noqa: E402
    DeterministicTimingProvider,
)
from GV4_Engine.state import GV4State, Player  # noqa: E402
from state_test import make_config  # noqa: E402


class ModelBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_root = Path(cls.temporary.name)
        cls.config = make_config(
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        cls.bundle = bootstrap_bundle(
            cls.output_root,
            config=cls.config,
            version=100,
            seed=11,
        )
        cls.pointer = cls.output_root / "models" / "current_model.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_bootstrap_bundle_is_complete_and_neutral(self) -> None:
        loaded = load_model_bundle(self.pointer, config=self.config)

        self.assertEqual(set(loaded.artifacts), set(MODEL_ARTIFACT_NAMES))
        self.assertEqual(
            loaded.role_versions,
            {"controller": 100, "adversary": 100},
        )
        self.assertEqual(loaded.metadata["eval_status"], "bootstrap")

        state = GV4State.initial(self.config, next_player=Player.ADVERSARY)
        _, actions = resolve_adversary_actions(state, self.config)
        inference = loaded.create_inference("python", self.config)
        logits = inference.predict_adversary_logits(state, actions)
        value = inference.predict_value(state, player=Player.ADVERSARY)

        # Zero policy logits mean exactly uniform priors after softmax. The
        # nonpositive value head approaches zero without violating its range.
        np.testing.assert_array_equal(logits, np.zeros_like(logits))
        self.assertLessEqual(value, 0.0)
        self.assertGreater(value, -0.001)

    def test_load_fails_closed_on_config_or_checksum_mismatch(self) -> None:
        with self.assertRaisesRegex(ModelBundleError, "different GV4 config"):
            load_model_bundle(
                self.pointer,
                config=make_config(pipeline_parallel_size=1),
            )

        artifact = self.bundle.artifacts["controller_value"]
        checkpoint = self.bundle.manifest_path.parent / artifact.checkpoint_path
        original = checkpoint.read_bytes()
        try:
            with checkpoint.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ModelBundleError, "checksum mismatch"):
                load_model_bundle(self.pointer, config=self.config)
        finally:
            checkpoint.write_bytes(original)

    def test_model_backed_runner_records_versions_and_bootstrap(self) -> None:
        inference = self.bundle.create_inference("python", self.config)
        runtime = create_engine_runtime(
            "python",
            self.config,
            search_config=SearchConfig(
                iterations=4,
                use_policy_prior=True,
                use_model_bootstrap=True,
            ),
            seed=23,
            timing_provider=DeterministicTimingProvider(2),
            python_inference=inference,
        )
        events = []
        replay_dir = self.output_root / "model_backed_replay"
        recorder = GV4ReplayRecorder(
            replay_dir,
            game_id=5,
            cycle_label="model_self_play",
            engine_metadata=runtime.metadata,
            model_versions=self.bundle.model_versions,
            game_seed=23,
            history_hops=1,
        )
        try:
            result = run_game_cycle(
                runtime,
                GameCycleConfig(
                    game_id=5,
                    cycle_label="model_self_play",
                    seed=23,
                    history_hops=1,
                    horizon_sec=0.003,
                    max_actions=30,
                    selection_temperature=0.0,
                    bootstrap_mode="model",
                ),
                replay=recorder,
                observer=events.append,
                model_versions=self.bundle.role_versions,
            )
        finally:
            runtime.close()

        searched = [event for event in events if event.search is not None]
        self.assertTrue(searched)
        self.assertTrue(any(event.search.used_bootstrap for event in searched))
        self.assertEqual(result.model_versions, self.bundle.role_versions)
        self.assertEqual(result.bootstrap_kind, "model")
        self.assertLessEqual(result.bootstrap_value, 0.0)
        self.assertIsNotNone(result.replay)

        manifest = json.loads((replay_dir / "replay_manifest.json").read_text())
        self.assertEqual(manifest["model_versions"], self.bundle.model_versions)
        self.assertEqual(
            manifest["engine_metadata"]["feature_schema_version"],
            self.config.layout.feature_schema_version,
        )

    def test_same_bundle_drives_native_policy_and_value_callbacks(self) -> None:
        try:
            inference = self.bundle.create_inference("native", self.config)
            runtime = create_engine_runtime(
                "native",
                self.config,
                search_config=SearchConfig(
                    iterations=4,
                    use_policy_prior=True,
                    use_model_bootstrap=True,
                ),
                seed=29,
                timing_provider=DeterministicTimingProvider(2),
                native_inference=inference,
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"native GV4 module is unavailable: {error}")

        try:
            state = runtime.initial_state()
            result = runtime.search(
                state,
                SearchContext(
                    game_id=6,
                    root_id=0,
                    root_depth=0,
                    model_version=100,
                ),
            )
        finally:
            runtime.close()

        self.assertTrue(result.used_bootstrap)
        self.assertGreater(len(result.action_stats), 1)
        self.assertTrue(all(item.visits >= 0 for item in result.action_stats))


if __name__ == "__main__":
    unittest.main()
