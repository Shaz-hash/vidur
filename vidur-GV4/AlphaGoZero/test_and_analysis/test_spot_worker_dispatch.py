"""Tests that Spot controller assignments select the correct worker path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from vidur.AlphaGoZero import worker_daemon as worker_module
from vidur.AlphaGoZero.durable_transfer import (
    remote_verify_and_publish,
    write_sha256sums,
)
from vidur.AlphaGoZero.spot_work_protocol import (
    default_selfplay_config,
    selfplay_config_sha256,
)
from vidur.AlphaGoZero.spot_worker_daemon import (
    _apply_controller_selfplay_config,
    _automatic_reserved_cpus,
    _controller_call,
    _recover_pending_eval_results,
    _run_eval_wave,
    _run_selfplay_wave,
    _selfplay_heartbeat_loop,
    _verify_eval_command,
    _write_single_game_metadata,
    detect_worker_resources,
    run_spot_worker,
)
from vidur.AlphaGoZero.worker_daemon import (
    CurrentModelPaths,
    RunningGame,
    WorkerState,
    _rollout_horizon_sec_for_game,
    _upload_ready_shards,
    run_worker,
)


class SpotWorkerDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="agz_spot_dispatch_")
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(
            worker_id="spot-test",
            parallel_games=4,
            max_waves=1,
            output_root=self.root,
            poll_sec=0.01,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_automatic_cpu_reserve_scales_with_machine_size(self) -> None:
        self.assertEqual(_automatic_reserved_cpus(1), 0)
        self.assertEqual(_automatic_reserved_cpus(16), 2)
        self.assertEqual(_automatic_reserved_cpus(32), 2)
        self.assertEqual(_automatic_reserved_cpus(64), 4)
        self.assertEqual(_automatic_reserved_cpus(96), 6)

    @patch("vidur.AlphaGoZero.spot_worker_daemon._cgroup_memory_headroom_bytes")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._cgroup_cpu_quota_count")
    @patch("vidur.AlphaGoZero.spot_worker_daemon.psutil.virtual_memory")
    @patch.object(os, "sched_getaffinity")
    def test_resource_detection_uses_affinity_and_available_ram(
        self,
        affinity,
        virtual_memory,
        cpu_quota,
        cgroup_memory,
    ) -> None:
        affinity.return_value = set(range(32))
        virtual_memory.return_value.available = 12 * 1024**3
        cpu_quota.return_value = None
        cgroup_memory.return_value = None
        args = argparse.Namespace(
            parallel_games=0,
            resource_reserve_cpus=-1,
            resource_memory_gib_per_game=1.0,
            worker_threads=1,
            rollout_parallel_threads=1,
        )
        resources = detect_worker_resources(args)
        self.assertEqual(resources["cpu_count"], 32)
        self.assertEqual(resources["reserved_cpu_count"], 2)
        self.assertEqual(resources["cpu_slots"], 30)
        self.assertEqual(resources["memory_slots"], 12)
        self.assertEqual(resources["advertised_capacity"], 12)
        self.assertEqual(resources["effective_capacity"], 12)

    def test_selfplay_config_overrides_worker_defaults_and_rejects_tampering(self) -> None:
        config = default_selfplay_config()
        config.update(
            {
                "iterations": 73,
                "discount_factor": 0.917,
                "native_search_mode": "full_tree_rollout",
                "rollout_count": 5,
                "rollout_parallel_threads": 3,
                "rollout_horizon_sec": 0.23,
                "root_dirichlet_alpha": 0.071,
                "root_dirichlet_epsilon": 0.19,
                "buffer_threshold": 2,
            }
        )
        assignment = {
            "assignment_id": "selfplay-test",
            "selfplay_config": config,
            "selfplay_config_sha256": selfplay_config_sha256(config),
        }
        args = argparse.Namespace(
            iterations=999,
            discount_factor=0.5,
            rollout_count=99,
            rollout_horizon_sec=9.0,
        )
        applied = _apply_controller_selfplay_config(args, assignment)
        self.assertEqual(applied, config)
        self.assertEqual(args.iterations, 73)
        self.assertEqual(args.discount_factor, 0.917)
        self.assertEqual(args.rollout_count, 5)
        self.assertEqual(args.rollout_horizon_sec, 0.23)
        self.assertEqual(args.assignment_config_sha256, assignment["selfplay_config_sha256"])

        tampered = dict(assignment)
        tampered["selfplay_config"] = {**config, "iterations": 74}
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            _apply_controller_selfplay_config(argparse.Namespace(), tampered)
        with self.assertRaisesRegex(ValueError, "missing coordinator configuration"):
            _apply_controller_selfplay_config(argparse.Namespace(), {})
        missing_field = dict(config)
        missing_field.pop("iterations")
        with self.assertRaisesRegex(ValueError, "missing=.*iterations"):
            _apply_controller_selfplay_config(
                argparse.Namespace(),
                {"selfplay_config": missing_field, "selfplay_config_sha256": "invalid"},
            )
        unknown_field = {**config, "worker_local_fallback": 1}
        with self.assertRaisesRegex(ValueError, "unknown=.*worker_local_fallback"):
            _apply_controller_selfplay_config(
                argparse.Namespace(),
                {"selfplay_config": unknown_field, "selfplay_config_sha256": "invalid"},
            )

    def test_spot_game_uses_fingerprint_verified_assignment_horizon(self) -> None:
        args = argparse.Namespace(
            assignment_config_sha256="verified-config",
            rollout_horizon_sec=3.0,
            output_root=self.root,
        )
        with patch(
            "vidur.AlphaGoZero.worker_daemon.active_rollout_horizon_sec",
            return_value=0.4,
        ) as runtime_horizon:
            self.assertEqual(_rollout_horizon_sec_for_game(args), 3.0)
        runtime_horizon.assert_not_called()

        args.assignment_config_sha256 = ""
        with patch(
            "vidur.AlphaGoZero.worker_daemon.active_rollout_horizon_sec",
            return_value=2.4,
        ) as runtime_horizon:
            self.assertEqual(_rollout_horizon_sec_for_game(args), 2.4)
        runtime_horizon.assert_called_once_with(self.root)

    def test_eval_command_fingerprint_rejects_tampering(self) -> None:
        command = ["python3", "-m", "fake.arena", "--iterations", "17"]
        fingerprint = hashlib.sha256(
            json.dumps(command, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _verify_eval_command({"command": command, "command_sha256": fingerprint})
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            _verify_eval_command(
                {"command": [*command[:-1], "18"], "command_sha256": fingerprint}
            )

    @patch("vidur.AlphaGoZero.spot_worker_daemon._run_eval_wave")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._controller_call")
    def test_eval_assignment_runs_eval_not_selfplay(
        self,
        controller_call,
        run_eval_wave,
    ) -> None:
        assignments = [
            {
                "task_id": "eval-1",
                "controller_model_version": 107,
                "adversary_model_version": 105,
            }
        ]
        controller_call.return_value = {"mode": "eval", "assignments": assignments}

        with patch("vidur.AlphaGoZero.spot_worker_daemon._run_selfplay_wave") as selfplay:
            run_spot_worker(self.args)

        run_eval_wave.assert_called_once_with(self.args, assignments)
        selfplay.assert_not_called()

    @patch("vidur.AlphaGoZero.spot_worker_daemon.time.sleep")
    @patch("vidur.AlphaGoZero.spot_worker_daemon.subprocess.run")
    def test_controller_call_retries_transient_ssh_failure(
        self, run_command, sleep
    ) -> None:
        run_command.side_effect = [
            subprocess.CalledProcessError(255, ["ssh"]),
            subprocess.CompletedProcess(
                ["ssh"],
                0,
                stdout='{"accepted": true}\n',
                stderr="",
            ),
        ]
        args = argparse.Namespace(
            controller_python="python3",
            xl_output_root=self.root,
            controller_repo=self.root,
            xl_host="coordinator",
            controller_timeout_sec=30,
            controller_retry_attempts=2,
            controller_retry_initial_sec=0.01,
            controller_retry_max_sec=0.01,
            output_root=self.root,
        )

        response = _controller_call(args, ["complete"])

        self.assertTrue(response["accepted"])
        self.assertEqual(run_command.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_remote_publish_retry_accepts_already_moved_result(self) -> None:
        uploading = self.root / "uploading"
        incoming = self.root / "incoming"
        rejected = self.root / "rejected"
        uploading.mkdir()
        (uploading / "arena_results.csv").write_text(
            "game_id\n17\n", encoding="utf-8"
        )
        write_sha256sums(uploading, include_manifest=True)

        def local_ssh(command, *, check=True, capture=True):
            del capture
            self.assertEqual(command[:2], ["ssh", "fake-host"])
            return subprocess.run(
                ["bash", "-c", command[2]],
                check=check,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        with patch(
            "vidur.AlphaGoZero.durable_transfer.run_cmd",
            side_effect=local_ssh,
        ):
            remote_verify_and_publish(
                host="fake-host",
                uploading_dir=str(uploading),
                incoming_dir=str(incoming),
                rejected_dir=str(rejected),
            )
            self.assertTrue((incoming / "arena_results.csv").is_file())
            self.assertFalse(uploading.exists())
            # This is the exact retry after a lost SSH acknowledgement.
            remote_verify_and_publish(
                host="fake-host",
                uploading_dir=str(uploading),
                incoming_dir=str(incoming),
                rejected_dir=str(rejected),
            )

        self.assertTrue((incoming / "arena_results.csv").is_file())
        self.assertFalse(rejected.exists())

    @patch("vidur.AlphaGoZero.spot_worker_daemon._publish_eval_result")
    def test_pending_completed_result_is_republished_after_restart(
        self, publish
    ) -> None:
        work_dir = self.root / "spot_eval_runs" / "eval-task-1"
        work_dir.mkdir(parents=True)
        assignment = {
            "task_id": "eval-task-1",
            "batch_id": "eval-batch",
            "block_name": "candidate",
            "game_id": 17,
            "worker_id": "spot-test",
            "lease_token": "lease-test",
        }
        (work_dir / "spot_assignment.json").write_text(
            json.dumps({"assignment": assignment}), encoding="utf-8"
        )
        (work_dir / "spot_result_manifest.json").write_text(
            json.dumps(assignment), encoding="utf-8"
        )
        (work_dir / "arena_results.csv").write_text(
            "game_id\n17\n", encoding="utf-8"
        )
        write_sha256sums(work_dir, include_manifest=True)
        args = argparse.Namespace(
            output_root=self.root, keep_game_runs=False, worker_id="spot-test"
        )

        recovered, pending = _recover_pending_eval_results(args)

        self.assertEqual((recovered, pending), (1, 0))
        publish.assert_called_once_with(args, assignment, work_dir)
        self.assertFalse(work_dir.exists())

    @patch("vidur.AlphaGoZero.spot_worker_daemon._heartbeat_loop")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._report_eval_failure")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._publish_eval_result")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._run_eval_assignment")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._stage_artifacts")
    def test_publish_failure_preserves_success_without_failing_task(
        self,
        _stage,
        run_assignment,
        publish,
        report_failure,
        _heartbeat,
    ) -> None:
        work_dir = self.root / "spot_eval_runs" / "eval-task-2"
        work_dir.mkdir(parents=True)
        assignment = {
            "task_id": "eval-task-2",
            "model_artifacts": [],
            "lease_token": "lease-test",
        }
        run_assignment.return_value = work_dir
        publish.side_effect = RuntimeError("lost SSH acknowledgement")
        args = argparse.Namespace(
            output_root=self.root,
            keep_game_runs=False,
            heartbeat_sec=10,
            assignment_lease_sec=300,
        )

        _run_eval_wave(args, [assignment])

        self.assertTrue(work_dir.is_dir())
        report_failure.assert_not_called()

    @patch("vidur.AlphaGoZero.spot_worker_daemon._run_selfplay_wave")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._controller_call")
    def test_selfplay_assignment_runs_selfplay_not_eval(
        self,
        controller_call,
        run_selfplay_wave,
    ) -> None:
        assignment = {
            "mode": "selfplay",
            "controller_model_version": 107,
            "adversary_model_version": 105,
            "games": 4,
        }
        controller_call.return_value = assignment

        with patch("vidur.AlphaGoZero.spot_worker_daemon._run_eval_wave") as run_eval:
            run_spot_worker(self.args)

        run_selfplay_wave.assert_called_once_with(self.args, assignment)
        run_eval.assert_not_called()

    @patch("vidur.AlphaGoZero.spot_worker_daemon._controller_call")
    def test_selfplay_heartbeat_sets_preemption_only_on_explicit_signal(
        self,
        controller_call,
    ) -> None:
        controller_call.return_value = {
            "extended": True,
            "preempt_requested": True,
            "preempt_reason": "evaluation_active",
            "eval_batch_id": "eval-123",
        }
        args = argparse.Namespace(
            heartbeat_sec=0.01,
            assignment_lease_sec=300,
            worker_id="spot-test",
            output_root=self.root,
        )
        assignment = {
            "assignment_id": "selfplay-123",
            "lease_token": "lease-123",
            "lease_sec": 300,
        }
        stop = MagicMock()
        stop.wait.return_value = False
        preempt_requested = threading.Event()

        _selfplay_heartbeat_loop(args, assignment, stop, preempt_requested)

        self.assertTrue(preempt_requested.is_set())
        record = json.loads(
            (self.root / "last_selfplay_preemption.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(record["eval_batch_id"], "eval-123")

    @patch("vidur.AlphaGoZero.spot_worker_daemon._controller_call")
    @patch("vidur.AlphaGoZero.spot_worker_daemon.run_worker")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._apply_controller_selfplay_config")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._install_selfplay_bundle")
    @patch("vidur.AlphaGoZero.spot_worker_daemon._selfplay_heartbeat_loop")
    def test_preempted_wave_releases_lease_as_preempted(
        self,
        heartbeat_loop,
        install_bundle,
        apply_config,
        run_worker,
        controller_call,
    ) -> None:
        heartbeat_loop.side_effect = (
            lambda _args, _assignment, _stop, preempt: preempt.set()
        )
        apply_config.return_value = {"iterations": 500}
        run_worker.return_value = True
        controller_call.return_value = {"released": True, "slots": 4}
        args = argparse.Namespace(**vars(self.args))
        args.heartbeat_sec = 0.01
        args.assignment_lease_sec = 300
        assignment = {
            "assignment_id": "selfplay-preempt",
            "lease_token": "lease-preempt",
            "lease_sec": 300,
            "parallel_games": 4,
            "games": 4,
            "selfplay_config_sha256": "config-sha",
        }

        _run_selfplay_wave(args, assignment)

        install_bundle.assert_called_once_with(args, assignment)
        run_args = run_worker.call_args.args[0]
        preempt_event = run_worker.call_args.kwargs["preempt_event"]
        self.assertEqual(run_args.parallel_games, 4)
        self.assertTrue(preempt_event.is_set())
        release_command = controller_call.call_args.args[1]
        self.assertEqual(release_command[-1], "preempted")

    def test_worker_preemption_keeps_completed_game_and_terminates_inflight_game(
        self,
    ) -> None:
        class FakeProcess:
            def __init__(self, return_code):
                self.return_code = return_code
                self.terminated = False

            def poll(self):
                return self.return_code

            def terminate(self):
                self.terminated = True
                self.return_code = -15

            def wait(self, timeout=None):
                del timeout
                return self.return_code

            def kill(self):
                self.return_code = -9

        preempt = threading.Event()
        completed_process = FakeProcess(0)
        inflight_process = FakeProcess(None)
        launch_count = 0

        def launch_game(_args, *, game_id, out_dir, **_kwargs):
            nonlocal launch_count
            launch_count += 1
            process = completed_process if launch_count == 1 else inflight_process
            if launch_count == 2:
                preempt.set()
            return RunningGame(
                game_id=int(game_id),
                out_dir=Path(out_dir),
                replay_csv=Path(out_dir) / "replay_target_runtime.csv",
                log_path=Path(out_dir) / "game.log",
                proc=process,
                model_version=100,
                controller_model_version=100,
                adversary_model_version=100,
            )

        args = argparse.Namespace(
            output_root=self.root,
            model_version=100,
            worker_id="spot-test",
            run_id="test",
            upload_after_game=False,
            seed=1,
            max_games=4,
            parallel_games=4,
            poll_sec=0.001,
            flush_at_end=False,
            continue_on_game_error=False,
            max_consecutive_game_errors=1,
            game_error_backoff_sec=0.001,
        )
        model_paths = CurrentModelPaths(
            controller_value_model_path=self.root / "controller_value",
            adversary_value_model_path=self.root / "adversary_value",
            controller_prior_model_path=self.root / "controller_prior",
            adversary_prior_model_path=self.root / "adversary_prior",
            model_version=100,
            controller_model_version=100,
            adversary_model_version=100,
        )
        with (
            patch.object(worker_module, "_validate_runtime_compatibility"),
            patch.object(
                worker_module,
                "_load_state",
                return_value=WorkerState(),
            ),
            patch.object(worker_module, "_append_model_comm"),
            patch.object(
                worker_module,
                "_current_model_paths",
                return_value=model_paths,
            ),
            patch.object(
                worker_module,
                "_allocate_game_id",
                side_effect=[101, 102],
            ),
            patch.object(
                worker_module,
                "_launch_one_game",
                side_effect=launch_game,
            ),
            patch.object(worker_module, "_merge_game_into_active") as merge,
            patch.object(worker_module, "_cleanup_game_output"),
            patch.object(worker_module, "_save_state"),
            patch.object(worker_module, "_append_worker_status"),
            patch.object(worker_module, "_freeze_if_needed") as freeze,
        ):
            was_preempted = run_worker(args, preempt_event=preempt)

        self.assertTrue(was_preempted)
        self.assertEqual(merge.call_count, 1)
        self.assertEqual(merge.call_args.kwargs["game_id"], 101)
        self.assertTrue(inflight_process.terminated)
        self.assertTrue(
            any(call.kwargs.get("force") is True for call in freeze.call_args_list)
        )

    def test_single_game_eval_metadata_matches_distributed_schema(self) -> None:
        work_dir = self.root / "eval"
        work_dir.mkdir()
        with (work_dir / "arena_results.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["game_id", "history_hops"])
            writer.writeheader()
            writer.writerow({"game_id": 77, "history_hops": 13})

        _write_single_game_metadata(
            work_dir,
            {"game_id": 77},
            elapsed_sec=1.25,
        )

        with (work_dir / "planned_games.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            planned = list(csv.DictReader(handle))
        with (work_dir / "job_status.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            status = list(csv.DictReader(handle))

        self.assertEqual(planned[0]["game_id"], "77")
        self.assertEqual(planned[0]["history_hops"], "13")
        self.assertEqual(status[0]["game_id"], "77")
        self.assertEqual(status[0]["status"], "ok")
        self.assertEqual(status[0]["elapsed_sec"], "1.250000")

    @patch("vidur.AlphaGoZero.worker_daemon._upload_ready_shards_serial")
    def test_ready_shard_uploads_are_serialized(self, upload_serial) -> None:
        state_lock = threading.Lock()
        active = 0
        max_active = 0

        def run_serial(_args) -> None:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

        upload_serial.side_effect = run_serial
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_upload_ready_shards, self.args)
                for _ in range(2)
            ]
            for future in futures:
                future.result()

        self.assertEqual(upload_serial.call_count, 2)
        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
