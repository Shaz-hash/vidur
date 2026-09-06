"""End-to-end checks for GV4 replay training, evaluation, and promotion."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_ROOT = GV4_ROOT.parent
sys.path.insert(0, str(CLASSICAL_ROOT))
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from AlphaGoZeroGV4.bootstrap_untrained_dnn_v100 import bootstrap_bundle  # noqa: E402
from AlphaGoZeroGV4.engine_runtime import (  # noqa: E402
    SearchConfig,
    create_engine_runtime,
)
from AlphaGoZeroGV4.model_bundle import (  # noqa: E402
    load_model_bundle,
    write_current_model_pointer,
)
from AlphaGoZeroGV4.replay_runtime import GV4ReplayRecorder  # noqa: E402
from AlphaGoZeroGV4.runner import GameCycleConfig, run_game_cycle  # noqa: E402
from AlphaGoZeroGV4.training_and_evaluation.agz_train_eval_promote import (  # noqa: E402
    TrainEvalPromoteConfig,
    run_train_eval_promote,
)
from AlphaGoZeroGV4.training_and_evaluation.arena import (  # noqa: E402
    ArenaConfig,
    ArenaResult,
    RoleArenaStats,
    evaluate_candidate,
)
from AlphaGoZeroGV4.training_and_evaluation.baselines import (  # noqa: E402
    select_sjf_controller_action,
)
from AlphaGoZeroGV4.training_and_evaluation.evaluation_pipeline_logger import (  # noqa: E402
    EvaluationPipelineLogger,
)
from AlphaGoZeroGV4.training_and_evaluation.indexed_replay import (  # noqa: E402
    open_replay_index,
)
from AlphaGoZeroGV4.training_and_evaluation.promotion import (  # noqa: E402
    PromotionPolicy,
    decide_promotion,
    publish_promotion,
)
from AlphaGoZeroGV4.training_and_evaluation.sjf_runner import (  # noqa: E402
    SJFBenchmarkConfig,
    evaluate_against_sjf,
)
from AlphaGoZeroGV4.training_and_evaluation.trainer import (  # noqa: E402
    TrainerConfig,
    train_candidate,
)
from GV4_Engine.GV4_MCTS_Test.timing import (  # noqa: E402
    DeterministicTimingProvider,
)
from GV4_Engine.history_root import HistoryRootGenerator  # noqa: E402
from state_test import make_config  # noqa: E402


class TrainingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = make_config(
            pipeline_parallel_size=2,
            max_inflight_microbatches=2,
        )
        cls.timing_factory = staticmethod(lambda: DeterministicTimingProvider(2))
        cls.incumbent = bootstrap_bundle(
            cls.root,
            config=cls.config,
            version=100,
            seed=11,
        )
        cls._write_replay()
        cls.index = open_replay_index(cls.root / "replay", cls.config)
        cls.training = train_candidate(
            cls.index,
            cls.config,
            cls.incumbent,
            candidate_version=101,
            destination=cls.root / "models" / "Model_Version101",
            training=TrainerConfig(
                max_roots_per_role=8,
                epochs=1,
                value_batch_size=8,
                policy_root_batch_size=8,
                torch_threads=1,
                seed=17,
            ),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def _write_replay(cls) -> None:
        inference = cls.incumbent.create_inference("python", cls.config)
        runtime = create_engine_runtime(
            "python",
            cls.config,
            search_config=SearchConfig(
                iterations=2,
                use_policy_prior=True,
                use_model_bootstrap=True,
            ),
            seed=2,
            timing_provider=cls.timing_factory(),
            python_inference=inference,
        )
        output = cls.root / "replay" / "game_2"
        recorder = GV4ReplayRecorder(
            output,
            game_id=2,
            cycle_label="self_play",
            engine_metadata=runtime.metadata,
            model_versions=cls.incumbent.model_versions,
            game_seed=2,
            history_hops=1,
        )
        try:
            run_game_cycle(
                runtime,
                GameCycleConfig(
                    game_id=2,
                    cycle_label="self_play",
                    seed=2,
                    history_hops=1,
                    horizon_sec=0.25,
                    max_actions=50,
                    selection_temperature=0.0,
                    bootstrap_mode="model",
                ),
                replay=recorder,
                model_versions=cls.incumbent.role_versions,
            )
        finally:
            runtime.close()

    def test_index_samples_structured_roots_without_replacement(self) -> None:
        counts = self.index.role_counts
        self.assertGreater(counts["controller"], 0)
        self.assertGreater(counts["adversary"], 0)

        first = self.index.sample_addresses("controller", 100, seed=9)
        second = self.index.sample_addresses("controller", 100, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), counts["controller"])
        self.assertEqual(
            len({(item.partition, item.state_offset) for item in first}), len(first)
        )

        roots = self.index.sample("controller", 2, seed=3)
        self.assertTrue(all(root.player == "controller" for root in roots))
        for root in roots:
            self.assertAlmostEqual(
                sum(action.visit_probability for action in root.actions),
                1.0,
            )

    def test_trainer_warm_starts_and_publishes_all_four_models(self) -> None:
        bundle = self.training.bundle
        self.assertEqual(bundle.bundle_version, 101)
        self.assertEqual(bundle.role_versions, {"controller": 101, "adversary": 101})
        self.assertGreater(self.training.sampled_roots["controller"], 0)
        self.assertGreater(self.training.sampled_roots["adversary"], 0)
        self.assertTrue(
            all(
                int(metrics["warm_start"]) == 1
                for metrics in self.training.metrics.values()
            )
        )

    def test_arena_and_sjf_write_backend_neutral_evaluation_logs(self) -> None:
        output = self.root / "evaluation_test"
        logger = EvaluationPipelineLogger(output)
        search = SearchConfig(
            iterations=2,
            use_policy_prior=True,
            use_model_bootstrap=True,
        )
        arena = evaluate_candidate(
            self.config,
            self.incumbent,
            self.training.bundle,
            arena=ArenaConfig(
                games=1,
                backend="python",
                search=search,
                horizon_sec=0.01,
                max_actions=10,
            ),
            timing_provider_factory=self.timing_factory,
            logger=logger,
        )
        self.assertEqual(len(arena.games), 3)
        self.assertEqual(arena.controller.games, 1)
        self.assertEqual(arena.adversary.games, 1)

        sjf = evaluate_against_sjf(
            self.config,
            self.training.bundle,
            benchmark=SJFBenchmarkConfig(
                games=1,
                backend="python",
                search=search,
                horizon_sec=0.01,
                max_actions=10,
            ),
            timing_provider_factory=self.timing_factory,
            logger=logger,
        )
        self.assertEqual(len(sjf.games), 1)
        with (output / "arena_results.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 5)
        self.assertTrue((output / "sjf_results.csv").is_file())
        self.assertEqual(len(list((output / "arena_games").glob("*.csv"))), 5)

    def test_sjf_selection_matches_raw_action_aliases(self) -> None:
        inference = self.incumbent.create_inference("python", self.config)
        runtime = create_engine_runtime(
            "python",
            self.config,
            search_config=SearchConfig(iterations=2),
            seed=2,
            timing_provider=self.timing_factory(),
            python_inference=inference,
        )
        try:
            root = HistoryRootGenerator(runtime).generate(hops=1, seed=2)
            self.assertEqual(runtime.player_to_move(root.state), "controller")
            selection = select_sjf_controller_action(
                runtime.canonical_actions(root.state), self.config
            )
        finally:
            runtime.close()
        self.assertIn(
            selection.matched_raw_index,
            selection.action.equivalent_raw_indices,
        )
        preemption, rule, budget, ordering = (
            self.config.controller_actions.raw_action_components(
                selection.matched_raw_index
            )
        )
        self.assertEqual(
            (preemption, rule, budget, ordering),
            ("preempt_none", "evict_none", 256, "SJF"),
        )

    def test_partial_promotion_keeps_value_and_policy_versions_together(self) -> None:
        arena = ArenaResult(
            incumbent_bundle_version=100,
            candidate_bundle_version=101,
            games=(),
            controller=RoleArenaStats("controller", 4, 4, 0, 0, 1.0, 1.0),
            adversary=RoleArenaStats("adversary", 4, 0, 0, 4, 0.0, -1.0),
        )
        decision = decide_promotion(
            arena,
            PromotionPolicy(
                min_games=4,
                controller_min_score_rate=0.75,
                adversary_min_score_rate=0.75,
            ),
        )
        pointer = self.root / "partial_current.json"
        result = publish_promotion(
            self.root / "models",
            self.config,
            self.incumbent,
            self.training.bundle,
            decision,
            current_model_path=pointer,
            record_path=self.root / "partial_promotion.json",
        )
        self.assertTrue(result.created_composite_bundle)
        self.assertEqual(
            result.promoted_bundle.role_versions,
            {"controller": 101, "adversary": 100},
        )
        self.assertEqual(
            load_model_bundle(pointer, config=self.config).role_versions,
            {"controller": 101, "adversary": 100},
        )

    def test_thin_orchestrator_runs_the_complete_cycle(self) -> None:
        pointer = self.root / "pipeline_current.json"
        write_current_model_pointer(pointer, self.incumbent)
        search = SearchConfig(
            iterations=2,
            use_policy_prior=True,
            use_model_bootstrap=True,
        )
        result = run_train_eval_promote(
            self.config,
            TrainEvalPromoteConfig(
                replay_root=self.root / "replay",
                replay_index_dir=self.root / "pipeline_index",
                models_root=self.root / "pipeline_models",
                current_model_path=pointer,
                output_dir=self.root / "pipeline_output",
                candidate_version=102,
                trainer=TrainerConfig(
                    max_roots_per_role=8,
                    epochs=1,
                    value_batch_size=8,
                    policy_root_batch_size=8,
                    torch_threads=1,
                    seed=29,
                ),
                arena=ArenaConfig(
                    games=1,
                    backend="python",
                    search=search,
                    horizon_sec=0.01,
                    max_actions=10,
                ),
                promotion=PromotionPolicy(min_games=2),
            ),
            timing_provider_factory=self.timing_factory,
        )
        self.assertTrue(result.summary_path.is_file())
        self.assertEqual(result.training.candidate_bundle_version, 102)
        self.assertFalse(result.promotion.decision.promote_controller)
        self.assertFalse(result.promotion.decision.promote_adversary)


if __name__ == "__main__":
    unittest.main()
