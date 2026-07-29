from __future__ import annotations

import csv
import json
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vidur.AlphaGoZero import agz_train_eval_promote as trainer
from vidur.AlphaGoZero import deploy
from vidur.AlphaGoZero import worker_daemon
from vidur.AlphaGoZero.cluster import get_cluster
from vidur.AlphaGoZero.config import (
    ADVERSARY_POLICY_SAMPLE_CAP,
    AGZ_DISCOUNT_FACTOR,
    AGZ_EVAL_MCTS_ITERATIONS,
    AGZ_INITIAL_SAMPLE_MOVE_COUNT,
    AGZ_MAX_ADVERSARY_VALUE_STATES,
    AGZ_MCTS_ITERATIONS,
    AGZ_ROOT_DIRICHLET_ALPHA,
    AGZ_ROOT_DIRICHLET_EPSILON,
    AGZ_SELFPLAY_ARENA_TIME_LIMIT_SEC,
    Phase1SmokeConfig,
    XL_ADVERSARY_MAX_REPLAY_STATES,
    XL_CONTROLLER_MAX_REPLAY_STATES,
)
from vidur.AlphaGoZero.replay_runtime import AlphaGoZeroReplayRecorder
from vidur.bellman_v4_adv import arena_mcts_value_runnerCPP as arena_cpp


class ExperimentConfigurationTests(unittest.TestCase):
    def test_deploy_forwards_dynamic_eval_layout(self) -> None:
        expected = {
            "AGZ_DISTRIBUTED_EVAL_ROLE_THREADS_PER_GAME": "2",
            "AGZ_DISTRIBUTED_EVAL_SJF_THREADS_PER_GAME": "8",
            "AGZ_EVAL_ROLLOUT_COUNT": "8",
            "AGZ_SJF_ROLLOUT_COUNT": "3",
            "AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_CORES": "0",
            "AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_FRACTION": "0",
            "AGZ_DISTRIBUTED_EVAL_CPU_SAMPLE_SEC": "0.25",
            "AGZ_DISTRIBUTED_EVAL_SJF_HOST_PARALLEL": "12",
        }
        with mock.patch.dict(deploy.os.environ, expected, clear=False):
            prefix = deploy._agz_env_prefix()
        for key, value in expected.items():
            self.assertIn(f"{key}={value}", prefix)

    def test_arena_commands_use_phase_specific_rollout_counts(self) -> None:
        common = dict(
            output_dir=Path("/tmp/eval"),
            model_path=Path("/tmp/value.dnn"),
            model_version=103,
            controller_prior_path=Path("/tmp/controller.dnn"),
            adversary_prior_path=Path("/tmp/adversary.dnn"),
            num_games=1,
            parallel_games=1,
            game_id_start=1,
            iterations=500,
            history_seed=7,
            history_hops_min=0,
            history_hops_max=1,
        )
        with mock.patch.object(trainer, "AGZ_EVAL_ROLLOUT_COUNT", 3), mock.patch.object(
            trainer, "AGZ_SJF_ROLLOUT_COUNT", 2
        ):
            role = trainer._arena_cmd(only_model_ctrl_cycle=True, **common)
            trivial = trainer._arena_cmd(
                only_model_ctrl_cycle=False, skip_model_ctrl_cycle=True, **common
            )
            explicit = trainer._arena_cmd(
                only_model_ctrl_cycle=True, rollout_count=7, **common
            )

        def option(command: list[str], name: str) -> str:
            return command[command.index(name) + 1]

        self.assertEqual(option(role, "--rollout-count"), "3")
        self.assertEqual(option(trivial, "--rollout-count"), "2")
        self.assertEqual(option(explicit, "--rollout-count"), "7")

    def test_simple_replay_and_search_defaults(self) -> None:
        self.assertEqual(XL_CONTROLLER_MAX_REPLAY_STATES, 20_000_000)
        self.assertEqual(XL_ADVERSARY_MAX_REPLAY_STATES, 15_000_000)
        self.assertEqual(ADVERSARY_POLICY_SAMPLE_CAP, 250_000)
        self.assertEqual(AGZ_MCTS_ITERATIONS, 1_000)
        self.assertEqual(AGZ_EVAL_MCTS_ITERATIONS, 1_000)
        self.assertEqual(AGZ_SELFPLAY_ARENA_TIME_LIMIT_SEC, 20.0)
        self.assertEqual(AGZ_MAX_ADVERSARY_VALUE_STATES, 300_000)
        self.assertEqual(AGZ_INITIAL_SAMPLE_MOVE_COUNT, 20)
        self.assertEqual(AGZ_ROOT_DIRICHLET_ALPHA, 0.05)
        self.assertEqual(AGZ_ROOT_DIRICHLET_EPSILON, 0.25)
        cfg = Phase1SmokeConfig()
        self.assertEqual(cfg.mcts_iterations, 1_000)
        self.assertEqual(cfg.arena_time_limit_sec, 20.0)
        self.assertEqual(cfg.root_dirichlet_alpha, 0.05)
        self.assertEqual(cfg.root_dirichlet_epsilon, 0.25)
        self.assertEqual(cfg.agz_sample_initial_move_count, 20)
        trainer_default = inspect.signature(trainer.train_candidate).parameters["mcts_iterations"].default
        self.assertEqual(trainer_default, 1_000)
        self.assertEqual(AGZ_DISCOUNT_FACTOR, 0.995)

    def test_explicit_adversary_root_target_does_not_change_training_cap(self) -> None:
        self.assertEqual(
            trainer._policy_root_sample_target(350_000, explicit_target=400_000),
            400_000,
        )
        self.assertEqual(
            trainer._policy_root_sample_target(350_000, explicit_target=300_000),
            350_000,
        )

    def test_value_sampling_applies_independent_role_caps(self) -> None:
        self.assertEqual(trainer._value_role_sample_targets(900_000, 300_000), (600_000, 300_000))
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.csv"
            fields = ["feature_complete", "state_features_json", "target_value", "player"]
            with replay_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(20):
                    writer.writerow(
                        {
                            "feature_complete": "1",
                            "state_features_json": json.dumps([float(index)] * 226),
                            "target_value": str(float(-index)),
                            "player": "controller" if index < 10 else "adversary",
                        }
                    )
            sampled, controller_roots, adversary_roots, counts = trainer._stream_sample_state_rows(
                [replay_path],
                seed=2026,
                max_value_rows=12,
                max_controller_policy_roots=0,
                max_adversary_policy_roots=0,
                max_controller_value_rows=7,
                max_adversary_value_rows=5,
            )

        self.assertEqual(len(sampled), 12)
        self.assertEqual(sum(row["player"] == "controller" for row in sampled), 7)
        self.assertEqual(sum(row["player"] == "adversary" for row in sampled), 5)
        self.assertEqual(controller_roots, {})
        self.assertEqual(adversary_roots, {})
        self.assertEqual(counts, {"states": 20, "controller": 10, "adversary": 10})

    def test_worker_game_ids_use_a_persisted_int32_safe_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "worker_state.json"
            args = SimpleNamespace(game_id_start=700_000_000)
            state = worker_daemon.WorkerState(shard_index=1_500)
            self.assertEqual(worker_daemon._allocate_game_id(args, state, state_path), 700_000_000)
            self.assertEqual(worker_daemon._allocate_game_id(args, state, state_path), 700_000_001)
            restored = worker_daemon._load_state(state_path, 100)
            self.assertEqual(restored.next_game_sequence, 2)

            restored.next_game_sequence = (
                worker_daemon.NATIVE_GAME_ID_MAX - int(args.game_id_start) + 1
            )
            with self.assertRaisesRegex(RuntimeError, "int32 range"):
                worker_daemon._allocate_game_id(args, restored, state_path)

    def test_exp2_cluster_has_one_xl_and_eight_dedicated_workers(self) -> None:
        cluster = get_cluster("exp2")
        self.assertEqual(cluster.xl.host, "bellman-classical-exp2-xl")
        self.assertEqual(len(cluster.workers), 8)
        self.assertEqual(
            [worker.host for worker in cluster.workers],
            [f"bellman-classical-exp2-worker-{index}" for index in range(1, 9)],
        )

    def test_exp3_cluster_has_one_xl_and_eight_dedicated_workers(self) -> None:
        cluster = get_cluster("exp3")
        self.assertEqual(cluster.xl.host, "bellman-classical-exp3-xl")
        self.assertEqual(len(cluster.workers), 8)
        self.assertEqual(
            [worker.host for worker in cluster.workers],
            [f"bellman-classical-exp3-worker-{index}" for index in range(1, 9)],
        )

    def test_native_payload_uses_explicit_discount_override(self) -> None:
        simulator = object()
        args = SimpleNamespace(
            discount_factor=0.98,
            model_version=100,
            disable_model_bootstrap=False,
            root_dirichlet_noise_enabled=True,
            root_dirichlet_alpha=0.1,
            root_dirichlet_epsilon=0.35,
            uct_c=1.4,
            puct_c=2.5,
            policy_prior_temperature=1.0,
            prior_min_prob=1e-8,
            native_search_mode="full_tree_rollout",
            rollout_count=10,
            rollout_parallel_threads=1,
            rollout_horizon_sec=1.0,
            rollout_policy_temperature=1.0,
            rollout_probability_quantum=1e-6,
            rollout_max_actions=4096,
        )
        pipeline_cfg = SimpleNamespace(
            game_v2=SimpleNamespace(
                mcts_search=SimpleNamespace(pb_c_base=19_652.0, pb_c_init=1.25),
            ),
            max_forced_hops_per_root=0,
        )
        cfg = SimpleNamespace(to_pipeline_cfg=lambda: pipeline_cfg)
        bundle = SimpleNamespace(simulator=simulator)

        arena_cpp._CFG_PAYLOAD_CACHE.clear()
        try:
            with mock.patch.object(arena_cpp, "_cfg_payload", return_value={"discount_factor": 0.995}), mock.patch.object(
                arena_cpp,
                "attach_execution_predictor_payload",
            ):
                payload = arena_cpp._get_cfg_payload(args, cfg, bundle)
        finally:
            arena_cpp._CFG_PAYLOAD_CACHE.clear()

        self.assertEqual(payload["discount_factor"], 0.98)
        self.assertEqual(payload["native_search_mode"], "full_tree_rollout")
        self.assertEqual(payload["rollout_count"], 10)
        self.assertEqual(payload["rollout_horizon_sec"], 1.0)

    def test_replay_fallback_uses_recorder_discount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.csv"
            recorder = AlphaGoZeroReplayRecorder(replay_path, discount_factor=0.98)
            recorder.record_transition(
                game_id=1,
                cycle_label="model_adv_depth1_vs_model_ctrl_depth1",
                phase="arena_step",
                turn=1,
                depth=1,
                player="controller",
                sim_time_before=0.0,
                sim_time_after=0.015725797204323228,
                total_cost_after=0.0,
                selection_info={"canonical_action_count": 3, "valid_action_count": 3},
            )
            self.assertEqual(
                recorder.finish_cycle(
                    game_id=1,
                    cycle_label="model_adv_depth1_vs_model_ctrl_depth1",
                    terminal_bootstrap_value=0.0,
                ),
                1,
            )
            with replay_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["discount"]), 0.98)

    def test_worker_recorder_interface_accepts_explicit_discount(self) -> None:
        self.assertIn(
            "discount_factor",
            inspect.signature(AlphaGoZeroReplayRecorder.__init__).parameters,
        )


if __name__ == "__main__":
    unittest.main()
