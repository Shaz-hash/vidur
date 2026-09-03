"""Integration tests for the GV4-native Python MCTS entry points."""

from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_ROOT = GV4_ROOT.parent
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(CLASSICAL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from GV4_Engine.action_resolver import (  # noqa: E402
    CanonicalAdversaryAction,
    CanonicalControllerAction,
)
from GV4_Engine.mcts_value_prior import (  # noqa: E402
    MCTSConfig,
    VidurMCTS,
)
from GV4_Engine.mcts_value_prior_rollout import (  # noqa: E402
    VidurMCTSPolicyRollout,
)
from GV4_Engine.state import GV4State, Player  # noqa: E402
from GV4_Engine.virtual_environment import (  # noqa: E402
    GV4VirtualVidurMCTSEnvironment,
)
from state_test import make_config  # noqa: E402


def timing_provider(state, action):
    stage_count = state.replica(action.replica_id).pipeline_parallel_size
    return (0.01,) * stage_count, (0.001,) * (stage_count - 1)


def make_environment(*, num_replicas: int = 1):
    config = make_config(
        num_replicas=num_replicas,
        pipeline_parallel_size=2,
        max_inflight_microbatches=2,
    )
    environment = GV4VirtualVidurMCTSEnvironment(
        config,
        batch_timing_provider=timing_provider,
        prefill_time_estimator=lambda tokens: float(tokens) * 1e-5,
    )
    return config, environment


def make_mcts_config() -> MCTSConfig:
    config = MCTSConfig()
    config.rng = random.Random(7)
    config.use_policy_prior = False
    return config


def run_search(mcts, state: GV4State, player: str, *, iterations: int = 4):
    return mcts.search_dnn(
        None,
        state,
        player,
        game_id=1,
        root_id=2,
        root_node_id_override=None,
        root_depth=0,
        mcts_iter=iterations,
        model_version=0,
        use_model_bootstrap=False,
    )


class GV4MCTSIntegrationTest(unittest.TestCase):
    def test_adversary_search_uses_gv4_snapshots_and_canonical_aliases(self) -> None:
        _, environment = make_environment()
        root_state = environment.initial_state()
        mcts = VidurMCTS(environment, make_mcts_config())

        result = run_search(mcts, root_state, "adversary")

        self.assertIsInstance(result.best_action, CanonicalAdversaryAction)
        self.assertEqual(result.next_player, "controller")
        self.assertIsInstance(mcts._root.cached_sim_snapshot, GV4State)
        self.assertIsNot(mcts._root.cached_sim_snapshot, root_state)
        self.assertEqual(root_state.requests, [])
        self.assertEqual(root_state.now, 0.0)

        action = result.best_action
        representative = min(action.equivalent_raw_indices)
        self.assertEqual(result.best_action_index, representative)
        self.assertEqual(
            mcts._root.canonical_to_action_aliases[representative],
            list(action.equivalent_raw_indices),
        )
        for raw_index in action.equivalent_raw_indices:
            self.assertEqual(
                mcts._root.action_alias_to_canonical[raw_index], representative
            )

    def test_controller_search_runs_on_launched_gv4_state(self) -> None:
        _, environment = make_environment()
        initial = environment.initial_state()
        actions, _ = environment.sample_adversary_actions(initial)
        launch = next(
            action
            for action in actions
            if action is not None and action.action.launch_count > 0
        )
        controller_state = environment.apply_adversary_action_only(initial, launch)
        before = controller_state.clone()

        mcts = VidurMCTS(environment, make_mcts_config())
        result = run_search(mcts, controller_state, "controller", iterations=6)

        self.assertIsInstance(result.best_action, CanonicalControllerAction)
        child = mcts._root.children[result.best_action_index]
        self.assertEqual(result.next_player, child.player)
        self.assertEqual(
            child.player,
            (
                "adversary"
                if child.cached_sim_snapshot.next_player == Player.ADVERSARY
                else "controller"
            ),
        )
        self.assertEqual(controller_state.now, before.now)
        self.assertEqual(len(controller_state.requests), len(before.requests))
        self.assertTrue(
            all(not request.has_inflight_work for request in controller_state.requests)
        )

    def test_policy_rollout_advances_gv4_clones_to_fixed_horizon(self) -> None:
        _, environment = make_environment()
        state = environment.initial_state()
        config = make_mcts_config()
        config.rollout_horizon_sec = 0.05
        config.rollout_count = 2
        config.rollout_max_actions = 64
        config.rollout_seed = 11
        mcts = VidurMCTSPolicyRollout(environment, config)

        run_search(mcts, state, "adversary", iterations=2)

        self.assertEqual(mcts.rollout_stats.leaf_evaluations, 2)
        self.assertEqual(mcts.rollout_stats.trajectories, 4)
        self.assertGreater(mcts.rollout_stats.actions, 0)
        self.assertGreaterEqual(mcts.rollout_stats.min_final_time, 0.05)
        self.assertEqual(state.now, 0.0)
        self.assertEqual(state.requests, [])

    def test_root_player_must_match_gv4_turn(self) -> None:
        _, environment = make_environment()
        mcts = VidurMCTS(environment, make_mcts_config())

        with self.assertRaisesRegex(ValueError, "does not match state turn"):
            run_search(mcts, environment.initial_state(), "controller")

    def test_multiple_replicas_require_router_mcts(self) -> None:
        _, environment = make_environment(num_replicas=2)

        with self.assertRaisesRegex(NotImplementedError, "router actions"):
            VidurMCTS(environment, make_mcts_config())


if __name__ == "__main__":
    unittest.main()
