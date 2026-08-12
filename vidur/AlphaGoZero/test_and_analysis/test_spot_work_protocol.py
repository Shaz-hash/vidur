"""Tests for pull-based Spot work assignment and lease safety."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vidur.AlphaGoZero.durable_transfer import write_sha256sums
from vidur.AlphaGoZero.spot_distributed_eval import run_spot_pull_commands
from vidur.AlphaGoZero.spot_work_protocol import (
    batch_status,
    begin_eval_batch,
    complete_task,
    configure_scheduler,
    default_selfplay_config,
    fail_task,
    finish_eval_batch,
    heartbeat_selfplay_assignment,
    heartbeat_tasks,
    release_selfplay_assignment,
    request_work,
    selfplay_config_sha256,
    spot_root,
)


def _write_model_bundle(root: Path) -> dict[str, object]:
    paths = {}
    for role in (
        "controller_value",
        "adversary_value",
        "controller_prior",
        "adversary_prior",
    ):
        version = 105 if role.startswith("adversary") else 107
        model_root = root / "models" / f"Model_Version{version}"
        directory = model_root / role / "dnn_test"
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / "model.joblib"
        model.write_bytes(f"{role}-model".encode())
        (directory / "native_model.tsv").write_text(
            f"{role}-native\n",
            encoding="utf-8",
        )
        paths[role] = model
    bundle: dict[str, object] = {
        "model_version": 107,
        "controller_model_version": 107,
        "adversary_model_version": 105,
        "model_family": "dnn",
        "native_ready": True,
        "controller_value_model_path": str(paths["controller_value"]),
        "adversary_value_model_path": str(paths["adversary_value"]),
        "controller_prior_model_path": str(paths["controller_prior"]),
        "adversary_prior_model_path": str(paths["adversary_prior"]),
    }
    current = root / "models" / "current_model.json"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(json.dumps(bundle), encoding="utf-8")
    return bundle


def _arena_command(root: Path, *, games: int) -> list[str]:
    bundle = json.loads((root / "models" / "current_model.json").read_text())
    return [
        "/usr/bin/python3",
        "-m",
        "fake.arena",
        "--output-dir",
        str(root / "unused"),
        "--game-id-start",
        "9000",
        "--num-games",
        str(games),
        "--num-parallel-games",
        str(games),
        "--history-seed",
        "33",
        "--model-version",
        str(bundle["controller_model_version"]),
        "--model-path",
        str(bundle["controller_value_model_path"]),
        "--controller-prior-model-path",
        str(bundle["controller_prior_model_path"]),
        "--adversary-prior-model-path",
        str(bundle["adversary_prior_model_path"]),
        "--role-controller-value-model-path",
        str(bundle["controller_value_model_path"]),
        "--role-controller-prior-model-path",
        str(bundle["controller_prior_model_path"]),
        "--role-adversary-value-model-path",
        str(bundle["adversary_value_model_path"]),
        "--role-adversary-prior-model-path",
        str(bundle["adversary_prior_model_path"]),
    ]


def _write_result(root: Path, task: dict[str, object]) -> None:
    result = spot_root(root) / "results" / str(task["task_id"])
    result.mkdir(parents=True, exist_ok=True)
    for filename in ("arena_results.csv", "planned_games.csv", "job_status.csv"):
        (result / filename).write_text("game_id,status\n9000,ok\n", encoding="utf-8")
    (result / "spot_result_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": str(task["task_id"]),
                "batch_id": str(task["batch_id"]),
                "block_name": str(task["block_name"]),
                "game_id": int(task["game_id"]),
                "worker_id": str(task["worker_id"]),
                "lease_token": str(task["lease_token"]),
            }
        ),
        encoding="utf-8",
    )
    write_sha256sums(result, include_manifest=True)


class SpotWorkProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="agz_spot_protocol_")
        self.root = Path(self.temp.name)
        self.bundle = _write_model_bundle(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_selfplay_assignment_has_exact_role_versions_and_artifacts(self) -> None:
        assignment = request_work(
            self.root,
            worker_id="spot-a",
            capacity=12,
            now_epoch=100.0,
        )
        self.assertEqual(assignment["mode"], "selfplay")
        self.assertEqual(assignment["games"], 12)
        self.assertEqual(assignment["controller_model_version"], 107)
        self.assertEqual(assignment["adversary_model_version"], 105)
        self.assertEqual(assignment["model_bundle"], self.bundle)
        self.assertEqual(
            assignment["selfplay_config_sha256"],
            selfplay_config_sha256(assignment["selfplay_config"]),
        )
        artifacts = assignment["model_artifacts"]
        self.assertEqual(len(artifacts), 4)
        self.assertTrue(all(len(item["files"]) == 2 for item in artifacts))

    def test_coordinator_config_controls_assignment_and_cpu_slots(self) -> None:
        config = default_selfplay_config()
        config.update(
            {
                "iterations": 37,
                "discount_factor": 0.913,
                "native_search_mode": "full_tree_rollout",
                "rollout_count": 3,
                "rollout_parallel_threads": 4,
                "rollout_horizon_sec": 0.17,
                "buffer_threshold": 2,
            }
        )
        configured = configure_scheduler(
            self.root,
            selfplay_total_parallel_games=100,
            eval_total_parallel_games=100,
            selfplay_config=config,
        )
        self.assertEqual(configured["worker_heartbeat_timeout_sec"], 300)
        self.assertEqual(configured["selfplay_lease_sec"], 300)
        assignment = request_work(
            self.root,
            worker_id="spot-configured",
            capacity=30,
            cpu_count=32,
            reserved_cpu_count=2,
            available_memory_bytes=64 * 1024**3,
            cpu_threads_per_game=1,
            now_epoch=100.0,
        )
        self.assertEqual(assignment["selfplay_config"], configured["selfplay_config"])
        self.assertEqual(assignment["selfplay_config"]["iterations"], 37)
        self.assertEqual(assignment["selfplay_config"]["rollout_count"], 3)
        self.assertEqual(assignment["resources"]["cpu_threads_per_game"], 4)
        self.assertEqual(assignment["resources"]["cpu_slots"], 7)
        self.assertEqual(assignment["parallel_games"], 7)
        self.assertEqual(
            assignment["selfplay_config_sha256"],
            configured["selfplay_config_sha256"],
        )

    def test_cpu_and_ram_resource_caps_are_enforced(self) -> None:
        configure_scheduler(
            self.root,
            selfplay_total_parallel_games=100,
            eval_total_parallel_games=100,
        )
        cpu_limited = request_work(
            self.root,
            worker_id="spot-16-core",
            capacity=100,
            cpu_count=16,
            reserved_cpu_count=2,
            available_memory_bytes=32 * 1024**3,
            memory_bytes_per_game=1024**3,
            now_epoch=100.0,
        )
        self.assertEqual(cpu_limited["mode"], "selfplay")
        self.assertEqual(cpu_limited["parallel_games"], 14)
        self.assertEqual(cpu_limited["resources"]["cpu_slots"], 14)
        self.assertEqual(cpu_limited["resources"]["memory_slots"], 32)

        ram_limited = request_work(
            self.root,
            worker_id="spot-32-core",
            capacity=100,
            cpu_count=32,
            reserved_cpu_count=2,
            available_memory_bytes=11 * 1024**3,
            memory_bytes_per_game=1024**3,
            now_epoch=100.0,
        )
        self.assertEqual(ram_limited["mode"], "selfplay")
        self.assertEqual(ram_limited["parallel_games"], 11)
        self.assertEqual(ram_limited["resources"]["cpu_slots"], 30)
        self.assertEqual(ram_limited["resources"]["memory_slots"], 11)

    def test_global_selfplay_cap_is_shared_and_released(self) -> None:
        configure_scheduler(
            self.root,
            selfplay_total_parallel_games=20,
            eval_total_parallel_games=20,
            selfplay_lease_sec=600,
        )
        first = request_work(
            self.root,
            worker_id="spot-a",
            capacity=100,
            cpu_count=16,
            reserved_cpu_count=2,
            available_memory_bytes=64 * 1024**3,
            now_epoch=100.0,
        )
        second = request_work(
            self.root,
            worker_id="spot-b",
            capacity=100,
            cpu_count=32,
            reserved_cpu_count=2,
            available_memory_bytes=64 * 1024**3,
            now_epoch=100.0,
        )
        waiting = request_work(
            self.root,
            worker_id="spot-c",
            capacity=100,
            cpu_count=32,
            reserved_cpu_count=2,
            available_memory_bytes=64 * 1024**3,
            now_epoch=100.0,
        )
        self.assertEqual(first["parallel_games"], 14)
        self.assertEqual(second["parallel_games"], 6)
        self.assertEqual(waiting["mode"], "idle")
        self.assertEqual(waiting["reason"], "selfplay_global_capacity_full")

        heartbeat = heartbeat_selfplay_assignment(
            self.root,
            worker_id="spot-a",
            assignment_id=str(first["assignment_id"]),
            lease_token=str(first["lease_token"]),
            extend_sec=1200,
            now_epoch=110.0,
        )
        self.assertTrue(heartbeat["extended"])
        released = release_selfplay_assignment(
            self.root,
            worker_id="spot-a",
            assignment_id=str(first["assignment_id"]),
            lease_token=str(first["lease_token"]),
            status="completed",
        )
        self.assertEqual(released, {"released": True, "slots": 14})

        replacement = request_work(
            self.root,
            worker_id="spot-c",
            capacity=100,
            cpu_count=32,
            reserved_cpu_count=2,
            available_memory_bytes=64 * 1024**3,
            now_epoch=120.0,
        )
        self.assertEqual(replacement["mode"], "selfplay")
        self.assertEqual(replacement["parallel_games"], 14)

    def test_dead_selfplay_worker_slots_recover_within_five_minutes(self) -> None:
        configured = configure_scheduler(
            self.root,
            selfplay_total_parallel_games=14,
            eval_total_parallel_games=14,
            selfplay_lease_sec=7200,
            worker_heartbeat_timeout_sec=300,
        )
        self.assertEqual(configured["selfplay_lease_sec"], 300)
        original = request_work(
            self.root,
            worker_id="spot-old",
            capacity=14,
            now_epoch=100.0,
        )
        self.assertEqual(original["lease_sec"], 300)
        heartbeat = heartbeat_selfplay_assignment(
            self.root,
            worker_id="spot-old",
            assignment_id=str(original["assignment_id"]),
            lease_token=str(original["lease_token"]),
            extend_sec=7200,
            now_epoch=110.0,
        )
        self.assertEqual(heartbeat["lease_sec"], 300)
        lease = json.loads(
            (
                spot_root(self.root)
                / "selfplay_leased"
                / "spot-old.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(lease["lease_expires_epoch"], 410.0)

        waiting = request_work(
            self.root,
            worker_id="spot-new",
            capacity=14,
            now_epoch=409.0,
        )
        self.assertEqual(waiting["mode"], "idle")
        replacement = request_work(
            self.root,
            worker_id="spot-new",
            capacity=14,
            now_epoch=411.0,
        )
        self.assertEqual(replacement["mode"], "selfplay")
        self.assertEqual(replacement["parallel_games"], 14)

    def test_reconfigure_immediately_caps_existing_long_lease(self) -> None:
        configure_scheduler(
            self.root,
            selfplay_total_parallel_games=1,
            eval_total_parallel_games=1,
            selfplay_lease_sec=300,
            worker_heartbeat_timeout_sec=300,
        )
        lease_start = time.time()
        assignment = request_work(
            self.root,
            worker_id="spot-old-config",
            capacity=1,
            now_epoch=lease_start,
        )
        lease_path = (
            spot_root(self.root)
            / "selfplay_leased"
            / "spot-old-config.json"
        )
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        self.assertEqual(assignment["lease_sec"], 300)
        lease["lease_expires_epoch"] = lease_start + 7200.0
        lease["worker_heartbeat_timeout_sec"] = 7200
        lease_path.write_text(json.dumps(lease), encoding="utf-8")

        before = time.time()
        configure_scheduler(
            self.root,
            selfplay_total_parallel_games=1,
            eval_total_parallel_games=1,
            selfplay_lease_sec=300,
            worker_heartbeat_timeout_sec=300,
        )
        capped = json.loads(lease_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(capped["lease_expires_epoch"], before + 299.0)
        self.assertLessEqual(capped["lease_expires_epoch"], time.time() + 300.0)
        self.assertEqual(capped["worker_heartbeat_timeout_sec"], 300)

    def test_heartbeat_timeout_cannot_be_configured_above_five_minutes(self) -> None:
        configured = configure_scheduler(
            self.root,
            selfplay_total_parallel_games=1,
            eval_total_parallel_games=1,
            selfplay_lease_sec=7200,
            worker_heartbeat_timeout_sec=7200,
        )
        self.assertEqual(configured["worker_heartbeat_timeout_sec"], 300)
        self.assertEqual(configured["selfplay_lease_sec"], 300)

    def test_global_eval_cap_limits_leases_across_workers(self) -> None:
        configure_scheduler(
            self.root,
            selfplay_total_parallel_games=100,
            eval_total_parallel_games=3,
        )
        begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=8)},
            expected_games_by_block={"candidate": 8},
            batch_id="eval-global-cap",
        )
        first = request_work(
            self.root,
            worker_id="spot-a",
            capacity=2,
            cpu_count=2,
            available_memory_bytes=8 * 1024**3,
            now_epoch=100.0,
        )
        second = request_work(
            self.root,
            worker_id="spot-b",
            capacity=2,
            cpu_count=2,
            available_memory_bytes=8 * 1024**3,
            now_epoch=100.0,
        )
        waiting = request_work(
            self.root,
            worker_id="spot-c",
            capacity=2,
            cpu_count=2,
            available_memory_bytes=8 * 1024**3,
            now_epoch=100.0,
        )
        self.assertEqual(len(first["assignments"]), 2)
        self.assertEqual(len(second["assignments"]), 1)
        self.assertEqual(waiting["mode"], "idle")
        self.assertEqual(waiting["reason"], "eval_global_capacity_full")

    def test_active_eval_requests_selfplay_preemption_and_worker_can_take_eval(self) -> None:
        configure_scheduler(
            self.root,
            selfplay_total_parallel_games=10,
            eval_total_parallel_games=10,
        )
        selfplay = request_work(
            self.root,
            worker_id="spot-preempt",
            capacity=4,
            now_epoch=100.0,
        )
        batch = begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=1)},
            expected_games_by_block={"candidate": 1},
            batch_id="eval-preempts-selfplay",
        )

        heartbeat = heartbeat_selfplay_assignment(
            self.root,
            worker_id="spot-preempt",
            assignment_id=str(selfplay["assignment_id"]),
            lease_token=str(selfplay["lease_token"]),
            extend_sec=300,
            now_epoch=110.0,
        )
        self.assertTrue(heartbeat["extended"])
        self.assertTrue(heartbeat["preempt_requested"])
        self.assertEqual(heartbeat["preempt_reason"], "evaluation_active")
        self.assertEqual(heartbeat["eval_batch_id"], batch["batch_id"])

        released = release_selfplay_assignment(
            self.root,
            worker_id="spot-preempt",
            assignment_id=str(selfplay["assignment_id"]),
            lease_token=str(selfplay["lease_token"]),
            status="preempted",
        )
        self.assertTrue(released["released"])
        evaluation = request_work(
            self.root,
            worker_id="spot-preempt",
            capacity=4,
            now_epoch=111.0,
        )
        self.assertEqual(evaluation["mode"], "eval")
        self.assertEqual(len(evaluation["assignments"]), 1)

    def test_eval_tasks_materialize_distinct_history_hops_coordinator_side(self) -> None:
        command = _arena_command(self.root, games=140)
        command.extend(
            [
                "--history-hops-min",
                "0",
                "--history-hops-max",
                "145",
                "--history-hops-unique",
                "--no-history-hops-force-zero",
            ]
        )
        batch = begin_eval_batch(
            self.root,
            block_commands={"candidate": command},
            expected_games_by_block={"candidate": 140},
            batch_id="exact-history-hops",
        )
        queued = spot_root(self.root) / "queued"
        commands = [
            json.loads((queued / f"{task_id}.json").read_text(encoding="utf-8"))[
                "command"
            ]
            for task_id in batch["tasks_by_block"]["candidate"]
        ]
        minimums = [
            int(task_command[task_command.index("--history-hops-min") + 1])
            for task_command in commands
        ]
        maximums = [
            int(task_command[task_command.index("--history-hops-max") + 1])
            for task_command in commands
        ]
        offsets = [
            int(task_command[task_command.index("--history-hops-offset") + 1])
            for task_command in commands
        ]
        self.assertEqual(minimums, maximums)
        self.assertEqual(len(set(minimums)), 140)
        self.assertTrue(all(0 <= hop <= 145 for hop in minimums))
        self.assertEqual(offsets, [0] * 140)

    def test_paired_eval_blocks_pin_identical_hops_for_matching_seeds(self) -> None:
        commands = {}
        for name in ("promoted", "candidate"):
            command = _arena_command(self.root, games=100)
            command.extend(
                [
                    "--history-hops-min",
                    "0",
                    "--history-hops-max",
                    "105",
                    "--history-hops-unique",
                    "--no-history-hops-force-zero",
                ]
            )
            commands[name] = command
        batch = begin_eval_batch(
            self.root,
            block_commands=commands,
            expected_games_by_block={name: 100 for name in commands},
            batch_id="paired-history-hops",
        )
        queued = spot_root(self.root) / "queued"

        def pinned_hops(block: str) -> list[int]:
            result = []
            for task_id in batch["tasks_by_block"][block]:
                task = json.loads(
                    (queued / f"{task_id}.json").read_text(encoding="utf-8")
                )
                command = task["command"]
                result.append(
                    int(command[command.index("--history-hops-min") + 1])
                )
            return result

        self.assertEqual(pinned_hops("promoted"), pinned_hops("candidate"))

    def test_eval_priority_retry_completion_and_resume_selfplay(self) -> None:
        batch = begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=2)},
            expected_games_by_block={"candidate": 2},
            lease_sec=120,
            batch_id="eval-test",
        )
        first = request_work(
            self.root,
            worker_id="spot-a",
            capacity=1,
            now_epoch=100.0,
        )
        second = request_work(
            self.root,
            worker_id="spot-b",
            capacity=1,
            now_epoch=100.0,
        )
        waiting = request_work(
            self.root,
            worker_id="spot-c",
            capacity=1,
            now_epoch=100.0,
        )
        task_a = first["assignments"][0]
        task_b = second["assignments"][0]
        self.assertEqual(first["mode"], "eval")
        self.assertEqual(second["mode"], "eval")
        self.assertEqual(waiting["mode"], "idle")
        self.assertEqual(task_a["controller_model_version"], 107)
        self.assertEqual(task_a["adversary_model_version"], 105)
        heartbeat = heartbeat_tasks(
            self.root,
            worker_id="spot-a",
            leases={str(task_a["task_id"]): str(task_a["lease_token"])},
            extend_sec=500,
            now_epoch=110.0,
        )
        self.assertEqual(heartbeat["extended"], [task_a["task_id"]])

        _write_result(self.root, task_a)
        accepted = complete_task(
            self.root,
            worker_id="spot-a",
            task_id=str(task_a["task_id"]),
            lease_token=str(task_a["lease_token"]),
        )
        self.assertTrue(accepted["accepted"])
        stale = complete_task(
            self.root,
            worker_id="spot-x",
            task_id=str(task_a["task_id"]),
            lease_token="wrong",
        )
        self.assertTrue(stale["accepted"])
        self.assertTrue(stale["idempotent"])

        failed = fail_task(
            self.root,
            worker_id="spot-b",
            task_id=str(task_b["task_id"]),
            lease_token=str(task_b["lease_token"]),
            error="synthetic interruption",
        )
        self.assertTrue(failed["requeued"])
        retry = request_work(
            self.root,
            worker_id="spot-c",
            capacity=1,
            now_epoch=120.0,
        )["assignments"][0]
        self.assertEqual(retry["task_id"], task_b["task_id"])
        self.assertEqual(retry["attempt"], 2)
        _write_result(self.root, retry)
        self.assertTrue(
            complete_task(
                self.root,
                worker_id="spot-c",
                task_id=str(retry["task_id"]),
                lease_token=str(retry["lease_token"]),
            )["accepted"]
        )

        status = batch_status(self.root, str(batch["batch_id"]))
        self.assertEqual(status["counts"]["completed"], 2)
        finish_eval_batch(self.root, batch_id=str(batch["batch_id"]))
        resumed = request_work(
            self.root,
            worker_id="spot-a",
            capacity=3,
            now_epoch=130.0,
        )
        self.assertEqual(resumed["mode"], "selfplay")
        self.assertEqual(resumed["games"], 3)

    def test_expired_lease_is_reassigned_and_old_token_is_rejected(self) -> None:
        begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=1)},
            expected_games_by_block={"candidate": 1},
            lease_sec=60,
            batch_id="expiry-test",
        )
        original = request_work(
            self.root,
            worker_id="spot-old",
            capacity=1,
            now_epoch=100.0,
        )["assignments"][0]
        reassigned = request_work(
            self.root,
            worker_id="spot-new",
            capacity=1,
            now_epoch=161.0,
        )["assignments"][0]
        self.assertEqual(original["task_id"], reassigned["task_id"])
        self.assertNotEqual(original["lease_token"], reassigned["lease_token"])
        _write_result(self.root, reassigned)
        stale = complete_task(
            self.root,
            worker_id="spot-old",
            task_id=str(original["task_id"]),
            lease_token=str(original["lease_token"]),
        )
        self.assertEqual(stale["reason"], "stale_lease")
        current = complete_task(
            self.root,
            worker_id="spot-new",
            task_id=str(reassigned["task_id"]),
            lease_token=str(reassigned["lease_token"]),
        )
        self.assertTrue(current["accepted"])

    def test_published_result_is_reconciled_before_expired_lease_requeue(self) -> None:
        begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=1)},
            expected_games_by_block={"candidate": 1},
            lease_sec=60,
            batch_id="publish-before-ack",
        )
        original = request_work(
            self.root,
            worker_id="spot-old",
            capacity=1,
            now_epoch=100.0,
        )["assignments"][0]
        _write_result(self.root, original)

        response = request_work(
            self.root,
            worker_id="spot-new",
            capacity=1,
            now_epoch=161.0,
        )

        self.assertEqual(response["mode"], "idle")
        self.assertEqual(response["reason"], "evaluation_tasks_leased")
        status = batch_status(self.root, "publish-before-ack")
        self.assertEqual(status["counts"]["completed"], 1)
        self.assertEqual(status["counts"]["queued"], 0)
        completed = json.loads(
            next((spot_root(self.root) / "completed").glob("*.json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(completed["reconciled_from_published_result"])

    def test_reconciliation_rejects_result_from_wrong_active_lease(self) -> None:
        begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=1)},
            expected_games_by_block={"candidate": 1},
            lease_sec=60,
            batch_id="wrong-lease-result",
        )
        assignment = request_work(
            self.root, worker_id="spot-current", capacity=1, now_epoch=100.0
        )["assignments"][0]
        stale = dict(assignment)
        stale["worker_id"] = "spot-stale"
        stale["lease_token"] = "stale-token"
        _write_result(self.root, stale)

        status = batch_status(self.root, "wrong-lease-result")
        self.assertEqual(status["counts"]["leased"], 1)
        self.assertEqual(status["counts"]["completed"], 0)

    def test_dead_eval_worker_task_recovers_within_five_minutes(self) -> None:
        configure_scheduler(
            self.root,
            selfplay_total_parallel_games=1,
            eval_total_parallel_games=1,
            worker_heartbeat_timeout_sec=300,
        )
        begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=1)},
            expected_games_by_block={"candidate": 1},
            lease_sec=7200,
            batch_id="five-minute-eval-expiry",
        )
        original = request_work(
            self.root,
            worker_id="spot-old",
            capacity=1,
            now_epoch=100.0,
        )["assignments"][0]
        self.assertEqual(original["effective_lease_sec"], 300)
        heartbeat = heartbeat_tasks(
            self.root,
            worker_id="spot-old",
            leases={str(original["task_id"]): str(original["lease_token"])},
            extend_sec=7200,
            now_epoch=110.0,
        )
        self.assertEqual(heartbeat["lease_sec"], 300)
        waiting = request_work(
            self.root,
            worker_id="spot-new",
            capacity=1,
            now_epoch=409.0,
        )
        self.assertEqual(waiting["mode"], "idle")
        reassigned = request_work(
            self.root,
            worker_id="spot-new",
            capacity=1,
            now_epoch=411.0,
        )["assignments"][0]
        self.assertEqual(reassigned["task_id"], original["task_id"])
        self.assertEqual(reassigned["attempt"], 2)
        self.assertNotEqual(reassigned["lease_token"], original["lease_token"])

    def test_concurrent_requests_never_duplicate_tasks(self) -> None:
        begin_eval_batch(
            self.root,
            block_commands={"candidate": _arena_command(self.root, games=8)},
            expected_games_by_block={"candidate": 8},
            batch_id="concurrent-test",
        )

        def request(index: int) -> str:
            response = request_work(
                self.root,
                worker_id=f"spot-{index}",
                capacity=1,
                now_epoch=100.0,
            )
            return str(response["assignments"][0]["task_id"])

        with ThreadPoolExecutor(max_workers=8) as executor:
            task_ids = list(executor.map(request, range(8)))
        self.assertEqual(len(task_ids), len(set(task_ids)))

    def test_eval_bridge_waits_for_workers_merges_and_finishes_batch(self) -> None:
        final_dir = self.root / "eval_output" / "candidate"
        worker_error: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    response = request_work(
                        self.root,
                        worker_id="spot-bridge",
                        capacity=2,
                    )
                    if response["mode"] != "eval":
                        time.sleep(0.01)
                        continue
                    for task in response["assignments"]:
                        _write_result(self.root, task)
                        result = complete_task(
                            self.root,
                            worker_id="spot-bridge",
                            task_id=str(task["task_id"]),
                            lease_token=str(task["lease_token"]),
                        )
                        self.assertTrue(result["accepted"])
                    return
                raise TimeoutError("test worker did not receive evaluation work")
            except BaseException as exc:  # pragma: no cover - surfaced by main thread.
                worker_error.append(exc)

        merged: list[tuple[Path, int, int]] = []

        def merge(block_dir: Path, parts: list[Path], expected: int) -> None:
            merged.append((block_dir, len(parts), expected))
            block_dir.mkdir(parents=True, exist_ok=True)
            (block_dir / "arena_results.csv").write_text(
                "game_id,status\n9000,ok\n",
                encoding="utf-8",
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        results = run_spot_pull_commands(
            root=self.root,
            block_commands={"candidate": _arena_command(self.root, games=2)},
            final_dirs={"candidate": final_dir},
            expected_games_by_block={"candidate": 2},
            merge_block=merge,
        )
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertFalse(worker_error)
        self.assertEqual(merged, [(final_dir, 2, 2)])
        self.assertEqual(results["candidate"], final_dir / "arena_results.csv")
        self.assertTrue((final_dir / "launch_command.json").is_file())
        resumed = request_work(self.root, worker_id="spot-bridge", capacity=1)
        self.assertEqual(resumed["mode"], "selfplay")


if __name__ == "__main__":
    unittest.main()
