"""Focused tests for the first GV4 AlphaGoZero orchestration boundary."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_ROOT = GV4_ROOT.parent
sys.path.insert(0, str(CLASSICAL_ROOT))
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from AlphaGoZeroGV4.engine_runtime import (  # noqa: E402
    ActionFeatureSnapshot,
    AppliedEdge,
    CanonicalActionRef,
    EngineMetadata,
    RootActionStats,
    SearchConfig,
    SearchResult,
    StateFeatureSnapshot,
    create_engine_runtime,
)
from AlphaGoZeroGV4.replay_runtime import GV4ReplayRecorder  # noqa: E402
from AlphaGoZeroGV4.runner import (  # noqa: E402
    GameCycleConfig,
    run_game_cycle,
)
from GV4_Engine.GV4_MCTS_Test.timing import (  # noqa: E402
    DeterministicTimingProvider,
)
from GV4_Engine.history_root import HistoryRootGenerator  # noqa: E402
from state_test import make_config  # noqa: E402


def _runtime(backend: str):
    config = make_config(
        pipeline_parallel_size=2,
        max_inflight_microbatches=2,
    )
    runtime = create_engine_runtime(
        backend,
        config,
        search_config=SearchConfig(iterations=8),
        seed=7,
        timing_provider=DeterministicTimingProvider(2),
    )
    return config, runtime


def _synthetic_metadata() -> EngineMetadata:
    return EngineMetadata(
        backend="python",
        config_manifest_sha256="manifest",
        manifest_schema_version="manifest-v1",
        state_schema_version="state-v1",
        action_schema_version="actions-v1",
        feature_schema_version="features-v1",
        native_layout_version="native-v1",
        time_epsilon=1e-9,
    )


class HistoryRootTest(unittest.TestCase):
    def test_seed_repeats_and_one_applied_action_is_one_hop(self) -> None:
        _, runtime = _runtime("python")
        try:
            initial = runtime.initial_state()
            first = HistoryRootGenerator(runtime).generate(
                hops=6,
                seed=19,
                initial_state=initial,
            )
            second = HistoryRootGenerator(runtime).generate(
                hops=6,
                seed=19,
                initial_state=initial,
            )

            first_actions = [
                (step.player, step.representative_raw_index) for step in first.steps
            ]
            second_actions = [
                (step.player, step.representative_raw_index) for step in second.steps
            ]
            self.assertTrue(first.complete)
            self.assertEqual(first.achieved_hops, 6)
            self.assertEqual([step.hop for step in first.steps], [1, 2, 3, 4, 5, 6])
            self.assertEqual(first_actions, second_actions)
            self.assertEqual(first.final_time, second.final_time)

            # The source is reusable for another cycle.
            self.assertEqual(initial.now, 0.0)
            self.assertEqual(initial.next_request_id, 0)
            self.assertEqual(initial.requests, [])
        finally:
            runtime.close()


class ReplayRuntimeTest(unittest.TestCase):
    def test_forced_edges_are_composed_before_target_backup(self) -> None:
        state_features = StateFeatureSnapshot(
            schema_version="features-v1",
            config_manifest_sha256="manifest",
            global_features=(1.0,),
            request_rows=(),
            request_replica_offsets=(0, 0),
            launch_rows=(),
            replica_rows=((1.0,),),
            microbatch_rows=(),
            microbatch_replica_offsets=(0, 0),
        )
        action_features = ActionFeatureSnapshot(
            header=(1.0,),
            affected_request_rows=(),
        )
        first_action = CanonicalActionRef(
            "python", "controller", 0, 4, (4, 8), object()
        )
        second_action = CanonicalActionRef(
            "python", "controller", 1, 12, (12,), object()
        )
        forced_action = CanonicalActionRef("python", "adversary", 0, 0, (0,), object())
        search = SearchResult(
            root_node_id=0,
            root_player="controller",
            next_player="adversary",
            best_action=first_action,
            best_action_value=-1.0,
            root_value=-1.25,
            action_stats=(
                RootActionStats(first_action, 3, -3.0, -1.0, 0.5, action_features),
                RootActionStats(second_action, 1, -2.0, -2.0, 0.5, action_features),
            ),
            raw_valid_mask=(True, False, False, False, True),
            state_features=state_features,
            used_bootstrap=False,
            used_rollout=False,
            diagnostics={},
        )
        selected_edge = AppliedEdge(
            object(),
            first_action,
            "controller",
            "adversary",
            0.0,
            1.0,
            0.0,
            1.0,
            -1.0,
            0.5,
            "BATCH",
        )
        forced_edge = AppliedEdge(
            object(),
            forced_action,
            "adversary",
            "controller",
            1.0,
            2.0,
            1.0,
            3.0,
            -2.0,
            0.25,
            "ADVERSARY",
        )

        with tempfile.TemporaryDirectory() as directory:
            recorder = GV4ReplayRecorder(
                directory,
                game_id=9,
                cycle_label="test",
                engine_metadata=_synthetic_metadata(),
                game_seed=17,
                history_hops=3,
            )
            recorder.record_decision(
                decision_index=0,
                search=search,
                selected_action=first_action,
                selected_edge=selected_edge,
                forced_edges=(forced_edge,),
            )
            result = recorder.finish_cycle(
                bootstrap_kind="model",
                bootstrap_value=-4.0,
                end_reason="horizon_reached",
                final_time=2.0,
            )

            state_row = json.loads(result.state_path.read_text().strip())
            action_rows = [
                json.loads(line) for line in result.action_path.read_text().splitlines()
            ]
            manifest = json.loads(result.manifest_path.read_text())

        # Composed edge: -1 + 0.5*(-2) = -2, discount = 0.5*0.25.
        # Backed-up target: -2 + 0.125*(-4) = -2.5.
        self.assertEqual(state_row["reward"], -2.0)
        self.assertEqual(state_row["discount"], 0.125)
        self.assertEqual(state_row["target_value"], -2.5)
        self.assertEqual(len(state_row["forced_chain"]), 1)
        self.assertEqual(
            [row["visit_probability"] for row in action_rows], [0.75, 0.25]
        )
        self.assertEqual([row["selected"] for row in action_rows], [True, False])
        self.assertEqual(manifest["state_rows"], 1)
        self.assertEqual(manifest["action_rows"], 2)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["engine_metadata"], asdict(_synthetic_metadata()))
        self.assertEqual(manifest["game_seed"], 17)
        self.assertEqual(manifest["history_hops"], 3)
        self.assertEqual(manifest["state_sha256"], result.state_sha256)
        self.assertEqual(manifest["action_sha256"], result.action_sha256)


class RuntimeBoundaryTest(unittest.TestCase):
    def test_python_and_native_expose_the_same_initial_state_summary(self) -> None:
        _, python_runtime = _runtime("python")
        try:
            _, native_runtime = _runtime("native")
        except (ImportError, ModuleNotFoundError) as error:
            python_runtime.close()
            self.skipTest(f"native GV4 module is unavailable: {error}")

        try:
            python_summary = python_runtime.summarize_state(
                python_runtime.initial_state()
            )
            native_summary = native_runtime.summarize_state(
                native_runtime.initial_state()
            )

            self.assertEqual(python_summary, native_summary)
            self.assertEqual(python_runtime.metadata.backend, "python")
            self.assertEqual(native_runtime.metadata.backend, "native")
            python_metadata = asdict(python_runtime.metadata)
            native_metadata = asdict(native_runtime.metadata)
            python_metadata.pop("backend")
            native_metadata.pop("backend")
            self.assertEqual(python_metadata, native_metadata)
        finally:
            python_runtime.close()
            native_runtime.close()

    def test_observer_receives_immutable_state_summaries(self) -> None:
        _, runtime = _runtime("python")
        events = []
        try:
            result = run_game_cycle(
                runtime,
                GameCycleConfig(
                    game_id=4,
                    seed=3,
                    horizon_sec=0.002,
                    max_actions=20,
                    selection_temperature=0.0,
                ),
                observer=events.append,
            )
        finally:
            runtime.close()

        self.assertTrue(events)
        self.assertEqual(
            [event.sequence for event in events],
            list(range(len(events))),
        )
        self.assertEqual(result.actions_applied, len(events))
        for event in events:
            self.assertEqual(
                event.state_before.config_manifest_sha256,
                event.state_after.config_manifest_sha256,
            )
            self.assertEqual(event.state_before.next_player, event.edge.player)
            self.assertEqual(event.state_after.next_player, event.edge.next_player)
            if event.phase == "searched":
                self.assertIsNotNone(event.search)
            else:
                self.assertIsNone(event.search)


class RunnerParityTest(unittest.TestCase):
    def test_python_and_native_run_the_same_seeded_uniform_cycle(self) -> None:
        cycle_config = GameCycleConfig(
            game_id=1,
            seed=7,
            history_hops=2,
            horizon_sec=0.005,
            max_actions=100,
            selection_temperature=0.0,
        )
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("python", "native"):
                try:
                    engine_config, runtime = _runtime(backend)
                except (ImportError, ModuleNotFoundError) as error:
                    if backend == "native":
                        self.skipTest(f"native GV4 module is unavailable: {error}")
                    raise
                recorder = GV4ReplayRecorder(
                    Path(directory) / backend,
                    game_id=cycle_config.game_id,
                    cycle_label=cycle_config.cycle_label,
                    engine_metadata=runtime.metadata,
                    game_seed=cycle_config.seed,
                    history_hops=cycle_config.history_hops,
                )
                try:
                    results.append(
                        run_game_cycle(
                            runtime,
                            cycle_config,
                            replay=recorder,
                        )
                    )
                finally:
                    runtime.close()

            for result in results:
                self.assertIsNotNone(result.replay)
                assert result.replay is not None
                self.assertGreater(result.replay.state_rows, 0)
                state_rows = [
                    json.loads(line)
                    for line in result.replay.state_path.read_text().splitlines()
                ]
                root_ids = [row["root_node_id"] for row in state_rows]
                self.assertEqual(len(root_ids), len(set(root_ids)))
                self.assertEqual(
                    root_ids,
                    [index * 10 for index in range(len(root_ids))],
                )

        python, native = results
        python_history = [
            (step.player, step.representative_raw_index)
            for step in python.history.steps
        ]
        native_history = [
            (step.player, step.representative_raw_index)
            for step in native.history.steps
        ]
        self.assertEqual(python_history, native_history)
        self.assertEqual(python.end_reason, native.end_reason)
        self.assertEqual(python.actions_applied, native.actions_applied)
        self.assertEqual(python.searched_decisions, native.searched_decisions)
        self.assertEqual(python.forced_actions, native.forced_actions)
        self.assertAlmostEqual(python.final_time, native.final_time, places=12)
        self.assertAlmostEqual(
            python.final_objective, native.final_objective, places=12
        )


if __name__ == "__main__":
    unittest.main()
