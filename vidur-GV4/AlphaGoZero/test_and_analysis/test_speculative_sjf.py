"""Tests for one-wave Spot role evaluation with retained speculative SJF."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vidur.AlphaGoZero import agz_train_eval_promote as trainer


def _bundle(version: int, *, controller: int | None = None, adversary: int | None = None) -> trainer.ModelBundle:
    controller_version = int(version if controller is None else controller)
    adversary_version = int(version if adversary is None else adversary)
    prefix = Path(f"/models/v{version}")
    return trainer.ModelBundle(
        model_version=int(version),
        controller_model_version=controller_version,
        adversary_model_version=adversary_version,
        controller_value_model_path=prefix / "controller_value.joblib",
        adversary_value_model_path=prefix / "adversary_value.joblib",
        controller_prior_model_path=prefix / "controller_policy.joblib",
        adversary_prior_model_path=prefix / "adversary_policy.joblib",
    )


class SpeculativeSjfTest(unittest.TestCase):
    def test_role_composition_is_pure_and_role_specific(self) -> None:
        current = _bundle(102, controller=101, adversary=102)
        candidate = _bundle(103)

        controller_only = trainer._compose_role_bundle(
            current=current,
            candidate=candidate,
            use_candidate_controller=True,
            use_candidate_adversary=False,
        )
        adversary_only = trainer._compose_role_bundle(
            current=current,
            candidate=candidate,
            use_candidate_controller=False,
            use_candidate_adversary=True,
        )
        both = trainer._compose_role_bundle(
            current=current,
            candidate=candidate,
            use_candidate_controller=True,
            use_candidate_adversary=True,
        )

        self.assertEqual((controller_only.controller_model_version, controller_only.adversary_model_version), (103, 102))
        self.assertEqual((adversary_only.controller_model_version, adversary_only.adversary_model_version), (101, 103))
        self.assertEqual((both.controller_model_version, both.adversary_model_version), (103, 103))
        self.assertEqual(controller_only.adversary_value_model_path, current.adversary_value_model_path)
        self.assertEqual(adversary_only.controller_value_model_path, current.controller_value_model_path)

    def test_selected_scenario_matches_only_arena_promotion_flags(self) -> None:
        self.assertEqual(
            trainer._selected_speculative_sjf_scenario(
                promote_controller=False, promote_adversary=False
            ),
            "",
        )
        self.assertEqual(
            trainer._selected_speculative_sjf_scenario(
                promote_controller=True, promote_adversary=False
            ),
            "controller_candidate",
        )
        self.assertEqual(
            trainer._selected_speculative_sjf_scenario(
                promote_controller=False, promote_adversary=True
            ),
            "adversary_candidate",
        )
        self.assertEqual(
            trainer._selected_speculative_sjf_scenario(
                promote_controller=True, promote_adversary=True
            ),
            "both_candidates",
        )

    def test_combined_batch_has_ten_blocks_and_does_not_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval_of_models" / "eval_000103"
            current_path = root / "models" / "current_model.json"
            current_path.parent.mkdir(parents=True)
            current_path.write_text('{"sentinel":"unchanged"}\n', encoding="utf-8")
            before = current_path.read_bytes()
            captured: dict[str, object] = {}

            def fake_run_spot_pull_commands(**kwargs):
                captured.update(kwargs)
                results = {}
                for name, directory in kwargs["final_dirs"].items():
                    directory = Path(directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    result = directory / "arena_results.csv"
                    result.write_text("game_id\n1\n", encoding="utf-8")
                    results[name] = result
                return results

            def fake_merge_split_sjf_cycles(**kwargs):
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                result = output_dir / "arena_results.csv"
                result.write_text("game_id\n1\n", encoding="utf-8")
                return result

            with mock.patch.object(
                trainer,
                "run_spot_pull_commands",
                side_effect=fake_run_spot_pull_commands,
            ), mock.patch.object(
                trainer,
                "merge_split_sjf_cycles",
                side_effect=fake_merge_split_sjf_cycles,
            ):
                outputs = trainer._run_spot_role_eval_with_speculative_sjf(
                    root=root,
                    eval_dir=eval_dir,
                    promoted=_bundle(102, controller=101, adversary=102),
                    candidate=_bundle(103),
                    eval_games=2,
                    eval_parallel=2,
                    benchmark_games=1,
                    iterations=5,
                    adversary_start_gid=1_000,
                    controller_start_gid=2_000,
                    adversary_seed=11,
                    controller_seed=12,
                    history_seed=13,
                    max_hop=100,
                )

            commands = captured["block_commands"]
            expected = captured["expected_games_by_block"]
            self.assertEqual(len(commands), 10)
            self.assertEqual(sum(name.startswith("sjf__") for name in commands), 6)
            self.assertEqual(sum(not name.startswith("sjf__") for name in commands), 4)
            self.assertEqual(sum(expected.values()), 14)
            self.assertEqual(set(outputs[4]), {"controller_candidate", "adversary_candidate", "both_candidates"})
            self.assertEqual(current_path.read_bytes(), before)
            manifest = json.loads((eval_dir / "SJF_256_Speculative" / "manifest.json").read_text())
            self.assertEqual(manifest["promotion_inputs"], "role_arena_results_only")

    def test_selected_result_is_retained_and_materialized_as_standard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "eval_000103"
            source_dir = eval_dir / "SJF_256_Speculative" / "controller_candidate"
            source_dir.mkdir(parents=True)
            source_results = source_dir / "arena_results.csv"
            source_results.write_text("game_id,cost\n1,2.5\n", encoding="utf-8")
            (source_dir / "planned_games.csv").write_text(
                "game_id,history_hops\n1,4\n", encoding="utf-8"
            )

            standard = trainer._publish_speculative_sjf_as_standard(
                eval_dir=eval_dir,
                scenario="controller_candidate",
                source_results=source_results,
            )

            self.assertTrue(source_results.is_file())
            self.assertEqual(standard.read_bytes(), source_results.read_bytes())
            selection = json.loads(
                (eval_dir / "SJF_256_Game" / "speculative_selection.json").read_text()
            )
            self.assertEqual(selection["scenario"], "controller_candidate")

    def test_live_evaluator_runs_role_blocks_before_conditional_sjf(self) -> None:
        source = inspect.getsource(trainer._evaluate_and_maybe_promote_unpaused)
        self.assertIn("_run_role_eval_blocks(", source)
        self.assertIn("if did_promote:", source)
        self.assertIn("sjf_results = _run_sjf_benchmark(", source)
        self.assertNotIn("_run_spot_role_eval_with_speculative_sjf(", source)
        self.assertNotIn("SJF_256_Speculative", source)

        promotion_index = source.index("if did_promote:")
        sjf_index = source.index("sjf_results = _run_sjf_benchmark(")
        self.assertGreater(sjf_index, promotion_index)


if __name__ == "__main__":
    unittest.main()
