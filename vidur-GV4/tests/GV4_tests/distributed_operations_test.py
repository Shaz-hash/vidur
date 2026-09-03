"""Integration checks for GV4 experiment config and distributed operations."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


GV4_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_ROOT = GV4_ROOT.parent
sys.path.insert(0, str(CLASSICAL_ROOT))
sys.path.insert(0, str(GV4_ROOT))

from AlphaGoZeroGV4.bootstrap_untrained_dnn_v100 import bootstrap_bundle  # noqa: E402
from AlphaGoZeroGV4.config import (  # noqa: E402
    CoordinatorConfig,
    DeploymentConfig,
    DistributedEvaluationConfig,
    ExperimentConfig,
    SelfPlayConfig,
    WorkerConfig,
    load_experiment_config,
    write_experiment_config,
)
from AlphaGoZeroGV4.distributed_operations.cluster import (  # noqa: E402
    ClusterSpec,
    HostSpec,
    load_cluster,
    sync_tree_to_host,
    write_cluster,
)
from AlphaGoZeroGV4.distributed_operations.deploy import (  # noqa: E402
    coordinator_command,
    deployment_environment,
    stage_current_model,
    stage_manifests,
    validate_hosts,
    worker_command,
    write_deployment_record,
)
from AlphaGoZeroGV4.distributed_operations.distributed_eval import (  # noqa: E402
    DISTRIBUTED_ARENA_SCHEMA_VERSION,
    merge_evaluation_chunks,
    plan_evaluation_chunks,
)
from AlphaGoZeroGV4.distributed_operations.durable_transfer import (  # noqa: E402
    freeze_replay_shard,
    publish_replay_shard,
    read_ack,
    retire_acknowledged_shard,
    validate_replay_shard,
)
from AlphaGoZeroGV4.distributed_operations.worker_daemon import (  # noqa: E402
    BundleSnapshot,
    build_game_command,
)
from AlphaGoZeroGV4.distributed_operations.xl_coordinator import (  # noqa: E402
    Coordinator,
)
from AlphaGoZeroGV4.engine_runtime import (  # noqa: E402
    SearchConfig,
    create_engine_runtime,
)
from AlphaGoZeroGV4.model_bundle import load_model_bundle  # noqa: E402
from AlphaGoZeroGV4.replay_runtime import GV4ReplayRecorder  # noqa: E402
from AlphaGoZeroGV4.runner import GameCycleConfig, run_game_cycle  # noqa: E402
from AlphaGoZeroGV4.training_and_evaluation.arena import (  # noqa: E402
    ArenaConfig,
    ArenaGameResult,
    ArenaResult,
    RoleArenaStats,
)
from AlphaGoZeroGV4.training_and_evaluation.promotion import (  # noqa: E402
    PromotionPolicy,
)
from AlphaGoZeroGV4.training_and_evaluation.trainer import (  # noqa: E402
    TrainerConfig,
)
from GV4_Engine.GV4_MCTS_Test.config import (  # noqa: E402
    build_default_engine_config,
)
from GV4_Engine.GV4_MCTS_Test.timing import (  # noqa: E402
    DeterministicTimingProvider,
)


ENGINE_FACTORY = "GV4_Engine.GV4_MCTS_Test.config:build_default_engine_config"


def _experiment(*, distributed_evaluation: bool = False) -> ExperimentConfig:
    search = SearchConfig(
        iterations=2,
        use_policy_prior=True,
        use_model_bootstrap=True,
    )
    return ExperimentConfig(
        experiment_id="gv4-distributed-test",
        engine_config_factory=ENGINE_FACTORY,
        self_play=SelfPlayConfig(
            backend="python",
            search=search,
            history_hops_max=2,
            horizon_sec=0.25,
            max_actions=50,
            selection_temperature=0.0,
        ),
        worker=WorkerConfig(
            shard_max_games=1,
            shard_max_roots=100,
            shard_max_bytes=1 << 24,
            poll_sec=0.01,
            retry_sec=0.01,
            ack_poll_sec=0.01,
        ),
        coordinator=CoordinatorConfig(
            max_replay_roots=10_000,
            train_trigger_new_roots=1,
            min_controller_roots=1,
            min_adversary_roots=1,
            poll_sec=0.01,
            training_enabled=False,
        ),
        trainer=TrainerConfig(
            max_roots_per_role=8,
            epochs=1,
            value_batch_size=8,
            policy_root_batch_size=8,
            torch_threads=1,
        ),
        arena=ArenaConfig(
            games=2,
            backend="python",
            search=search,
            horizon_sec=0.01,
            max_actions=10,
        ),
        promotion=PromotionPolicy(min_games=2),
        distributed_evaluation=DistributedEvaluationConfig(
            enabled=distributed_evaluation,
            games_per_chunk=1,
        ),
        deployment=DeploymentConfig(
            native_build_jobs=2,
            environment=(("OMP_NUM_THREADS", "1"),),
        ),
    )


def _host(
    host_id: str,
    role: str,
    ordinal: int,
    experiment_root: Path,
) -> HostSpec:
    return HostSpec(
        host_id=host_id,
        address="local",
        ordinal=ordinal,
        role=role,
        repo_root=str(CLASSICAL_ROOT),
        experiment_root=str(experiment_root),
        python_executable=sys.executable,
    )


def _arena_game(pair: int, scenario: str, cost: float) -> ArenaGameResult:
    return ArenaGameResult(
        pair_index=pair,
        paired_seed=2026 + pair,
        scenario=scenario,
        controller_version=101 if scenario == "candidate_controller" else 100,
        adversary_version=101 if scenario == "candidate_adversary" else 100,
        final_cost=cost,
        slo_violations=0,
        prefill_lateness_sec=0.0,
        decode_lateness_sec=0.0,
        requests_completed=0,
        requests_stopped=0,
        requests_dropped=0,
        actions_applied=1,
        end_reason="horizon",
    )


class DistributedOperationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = build_default_engine_config()
        cls.experiment = _experiment()
        cls.config_path = write_experiment_config(
            cls.root / "manifests" / "experiment.json",
            cls.experiment,
            engine_config=cls.config,
        )
        cls.cluster = ClusterSpec(
            name="gv4-local-test",
            coordinator=_host(
                "coordinator",
                "coordinator",
                0,
                cls.root / "coordinator_host",
            ),
            workers=(_host("worker-1", "worker", 1, cls.root / "worker_host"),),
        )
        cls.cluster_path = write_cluster(
            cls.root / "manifests" / "cluster.json",
            cls.cluster,
        )
        cls.bundle = bootstrap_bundle(
            cls.root / "bootstrap",
            config=cls.config,
            version=100,
            seed=19,
        )
        cls.replay_template = cls.root / "replay_template"
        cls._write_replay_template()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def _write_replay_template(cls) -> None:
        runtime = create_engine_runtime(
            "python",
            cls.config,
            search_config=cls.experiment.self_play.search,
            seed=7,
            timing_provider=DeterministicTimingProvider(2),
            python_inference=cls.bundle.create_inference("python", cls.config),
        )
        recorder = GV4ReplayRecorder(
            cls.replay_template,
            game_id=7,
            cycle_label="self_play",
            engine_metadata=runtime.metadata,
            model_versions=cls.bundle.model_versions,
            game_seed=7,
            history_hops=1,
        )
        try:
            run_game_cycle(
                runtime,
                GameCycleConfig(
                    game_id=7,
                    cycle_label="self_play",
                    seed=7,
                    history_hops=1,
                    horizon_sec=0.25,
                    max_actions=50,
                    selection_temperature=0.0,
                    bootstrap_mode="model",
                ),
                replay=recorder,
                model_versions=cls.bundle.role_versions,
            )
        finally:
            runtime.close()

    def _freeze(self, name: str):
        root = self.root / name
        active = root / "active"
        destination = active / "games" / "game_7"
        destination.parent.mkdir(parents=True)
        shutil.copytree(self.replay_template, destination)
        return freeze_replay_shard(
            active,
            root / "ready",
            worker_id="worker-1",
            shard_id=f"shard-{name}",
            config=self.config,
        )

    def test_experiment_and_cluster_manifests_round_trip(self) -> None:
        loaded = load_experiment_config(self.config_path)
        self.assertEqual(loaded, self.experiment)
        self.assertEqual(load_cluster(self.cluster_path), self.cluster)
        self.assertEqual(
            loaded.resolve_engine_config().manifest_sha256(),
            self.config.manifest_sha256(),
        )

        corrupted = self.root / "manifests" / "wrong_engine.json"
        value = json.loads(self.config_path.read_text(encoding="utf-8"))
        value["engine_config_sha256"] = "0" * 64
        corrupted.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "factory no longer matches"):
            load_experiment_config(corrupted)

    def test_worker_command_is_a_complete_frozen_runner_invocation(self) -> None:
        snapshot = BundleSnapshot(
            manifest_path=self.bundle.manifest_path,
            manifest_sha256=self.bundle.manifest_sha256,
            bundle_version=self.bundle.bundle_version,
            artifact_versions=self.bundle.model_versions,
            role_versions=self.bundle.role_versions,
        )
        command = build_game_command(
            self.experiment,
            python_executable=sys.executable,
            output_dir=self.root / "game_command",
            game_id=1_000_000_003,
            seed=23,
            history_hops=2,
            bundle=snapshot,
        )
        self.assertEqual(command[0], sys.executable)
        self.assertIn("AlphaGoZeroGV4.runner", command)
        self.assertIn("--use-policy-prior", command)
        self.assertIn("--use-model-bootstrap", command)
        self.assertEqual(command[command.index("--mcts-iterations") + 1], "2")
        self.assertEqual(command[command.index("--rollout-seed") + 1], "23")
        self.assertEqual(
            command[command.index("--model-bundle") + 1],
            str(self.bundle.manifest_path),
        )

    def test_source_sync_excludes_generated_directories(self) -> None:
        source = self.root / "source_sync"
        target = self.root / "target_sync"
        (source / "cache").mkdir(parents=True)
        (source / "package").mkdir()
        (source / "cache" / "large.bin").write_bytes(b"ignored")
        (source / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        host = _host("copy-target", "worker", 3, self.root / "copy_exp")
        host = replace(host, repo_root=str(target))
        sync_tree_to_host(source, host, str(target), exclude_names=("cache",))
        self.assertTrue((target / "package" / "module.py").is_file())
        self.assertFalse((target / "cache").exists())

    def test_model_staging_and_subprocess_preflight(self) -> None:
        record = write_deployment_record(
            self.root / "manifests" / "deployment.json",
            experiment_config_path=self.config_path,
            cluster_path=self.cluster_path,
            source_root=self.root,
            current_model_path=self.bundle.manifest_path,
        )
        stage_manifests(
            self.config_path,
            self.cluster_path,
            self.cluster,
            deployment_record=record,
        )
        destinations = stage_current_model(
            self.bundle.manifest_path,
            self.experiment,
            self.cluster,
        )
        for host in self.cluster.all_hosts:
            pointer = Path(host.experiment_root) / "models" / "current_model.json"
            staged = load_model_bundle(pointer, config=self.config, device="cpu")
            self.assertEqual(staged.manifest_sha256, self.bundle.manifest_sha256)
            self.assertEqual(destinations[host.host_id], str(pointer))
            pointer_value = json.loads(pointer.read_text(encoding="utf-8"))
            self.assertIn(
                self.bundle.manifest_sha256[:16],
                pointer_value["bundle_manifest_path"],
            )

        validation = validate_hosts(self.experiment, self.cluster)
        self.assertEqual(set(validation), {"coordinator", "worker-1"})
        self.assertTrue(
            all(
                row["config_manifest_sha256"] == self.config.manifest_sha256()
                for row in validation.values()
            )
        )
        self.assertEqual(
            deployment_environment(self.experiment, self.cluster.coordinator)[
                "PYTHONUNBUFFERED"
            ],
            "1",
        )
        self.assertIn(
            "xl_coordinator", " ".join(coordinator_command(self.cluster.coordinator))
        )
        self.assertIn(
            "worker_daemon", " ".join(worker_command(self.cluster.workers[0]))
        )

    def test_replay_delivery_is_checksum_verified_idempotent_and_acknowledged(
        self,
    ) -> None:
        shard = self._freeze("delivery")
        coordinator_root = self.root / "delivery_coordinator"
        cluster = ClusterSpec(
            name="delivery-cluster",
            coordinator=_host("delivery-xl", "coordinator", 0, coordinator_root),
            workers=(_host("worker-1", "worker", 1, self.root / "delivery_worker"),),
        )
        coordinator = Coordinator(
            self.experiment,
            cluster,
            root=coordinator_root,
        )

        first = publish_replay_shard(
            shard.path,
            cluster.coordinator,
            config=self.config,
        )
        second = publish_replay_shard(
            shard.path,
            cluster.coordinator,
            config=self.config,
        )
        self.assertEqual(first, second)
        self.assertEqual(coordinator.ingest_once(), 1)
        self.assertEqual(len(coordinator.state.accepted), 1)
        self.assertEqual(coordinator.training_gate().reason, "training is disabled")

        publish_replay_shard(shard.path, cluster.coordinator, config=self.config)
        self.assertEqual(coordinator.ingest_once(), 0)
        self.assertFalse(Path(first).exists())
        acknowledgement = read_ack(
            cluster.coordinator,
            worker_id=shard.worker_id,
            shard_id=shard.shard_id,
            expected_digest=shard.digest,
        )
        self.assertIsNotNone(acknowledgement)
        self.assertTrue(
            retire_acknowledged_shard(
                shard.path,
                cluster.coordinator,
                config=self.config,
            )
        )
        self.assertFalse(shard.path.exists())

        restarted = Coordinator(self.experiment, cluster, root=coordinator_root)
        self.assertEqual(len(restarted.state.accepted), 1)
        self.assertEqual(restarted.state.next_accept_sequence, 1)

    def test_replay_transport_rejects_a_modified_payload(self) -> None:
        shard = self._freeze("corrupt")
        actions = next(shard.path.rglob("replay_actions.jsonl"))
        with actions.open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_replay_shard(shard.path, self.config)

    def test_distributed_chunk_plan_and_exact_result_merge(self) -> None:
        second_worker = _host("worker-2", "worker", 2, self.root / "worker_2")
        chunks = plan_evaluation_chunks(
            5,
            (self.cluster.workers[0], second_worker),
            games_per_chunk=2,
        )
        self.assertEqual(
            [
                (chunk.pair_start, chunk.pair_count, chunk.host.host_id)
                for chunk in chunks
            ],
            [(0, 2, "worker-1"), (2, 2, "worker-2"), (4, 1, "worker-1")],
        )

        costs = {
            0: (10.0, 8.0, 12.0),
            1: (10.0, 10.0, 9.0),
            2: (10.0, 12.0, 10.0),
        }
        paths: list[Path] = []
        for chunk_id, pair_indices in enumerate(((0, 1), (2,))):
            games = tuple(
                game
                for pair in pair_indices
                for game in (
                    _arena_game(pair, "incumbent", costs[pair][0]),
                    _arena_game(pair, "candidate_controller", costs[pair][1]),
                    _arena_game(pair, "candidate_adversary", costs[pair][2]),
                )
            )
            placeholder = RoleArenaStats("controller", 1, 0, 1, 0, 0.5, 0.0)
            result = ArenaResult(
                100, 101, games, placeholder, replace(placeholder, role="adversary")
            )
            path = self.root / "chunk_results" / f"chunk_{chunk_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": DISTRIBUTED_ARENA_SCHEMA_VERSION,
                        "pair_start": pair_indices[0],
                        "pair_count": len(pair_indices),
                        "arena": result.summary(),
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)

        merged = merge_evaluation_chunks(paths, expected_games=3)
        self.assertEqual(len(merged.games), 9)
        self.assertEqual(
            (merged.controller.wins, merged.controller.ties, merged.controller.losses),
            (1, 1, 1),
        )
        self.assertEqual(
            (merged.adversary.wins, merged.adversary.ties, merged.adversary.losses),
            (1, 1, 1),
        )
        self.assertAlmostEqual(merged.adversary.mean_improvement, 1.0 / 3.0)

        with self.assertRaisesRegex(ValueError, "do not cover"):
            merge_evaluation_chunks(paths[:1], expected_games=3)


if __name__ == "__main__":
    unittest.main()
