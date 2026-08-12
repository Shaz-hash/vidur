from __future__ import annotations

import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from vidur.AlphaGoZero.distributed_eval import (
    EvalHost,
    _merge_csv,
    default_role_hosts,
    _parse_cpu_set,
    _parse_proc_stat_pair,
    cpu_safe_hosts,
    default_sjf_hosts,
    memory_safe_hosts,
    merge_split_sjf_cycles,
    plan_independent_block_chunks,
    plan_role_chunks,
    plan_single_block_chunks,
)
from vidur.bellman_v4_adv.arena_mcts_value_runnerCPP import (
    _pin_single_game_history_hop,
    _planned_history_hops,
)
from vidur.AlphaGoZero.xl_coordinator import _finalize_completed_training_cycle_if_needed


def _arena_command(name: str, game_id: int, games: int = 100) -> list[str]:
    return [
        "python3",
        "-m",
        "arena",
        "--model-path",
        "/tmp/value.joblib",
        "--controller-prior-model-path",
        "/tmp/controller.joblib",
        "--adversary-prior-model-path",
        "/tmp/adversary.joblib",
        "--output-dir",
        f"/tmp/{name}",
        "--game-id-start",
        str(game_id),
        "--num-games",
        str(games),
        "--num-parallel-games",
        "60",
        "--rollout-parallel-threads",
        "1",
        "--shared-root-mcts-iterations",
        "4000",
        "--history-hops-min",
        "0",
        "--history-hops-max",
        "105",
        "--history-seed",
        "2129",
    ]


class DistributedEvalPlanTests(unittest.TestCase):
    def test_cpu_set_and_proc_stat_parsing(self) -> None:
        self.assertEqual(_parse_cpu_set("0-2,4,6-7"), {0, 1, 2, 4, 6, 7})
        raw = (
            "cpu 100 0 50 800 10 0 0 0\n"
            "cpu0 0 0 0 0 0 0 0 0\n"
            "cpu1 0 0 0 0 0 0 0 0\n"
            "__AGZ_CPU_SAMPLE__"
            "cpu 120 0 60 870 10 0 0 0\n"
            "cpu0 0 0 0 0 0 0 0 0\n"
            "cpu1 0 0 0 0 0 0 0 0\n"
        )
        logical, idle = _parse_proc_stat_pair(raw)
        self.assertEqual(logical, 2)
        self.assertAlmostEqual(idle, 0.7)

    def test_cpu_planner_assigns_phase_specific_threads(self) -> None:
        env = {
            "AGZ_DISTRIBUTED_EVAL_ROLE_THREADS_PER_GAME": "2",
            "AGZ_DISTRIBUTED_EVAL_SJF_THREADS_PER_GAME": "8",
            "AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_CORES": "2",
            "AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_FRACTION": "0",
            "AGZ_DISTRIBUTED_EVAL_XL_CPUSET": "4-95",
            "AGZ_DISTRIBUTED_EVAL_WORKER_CPUSET": "2-95",
        }
        with mock.patch.dict("os.environ", env):
            role_hosts = default_role_hosts()
            role_hosts = cpu_safe_hosts(
                role_hosts,
                phase="role",
                total_games=400,
                cpu_by_label={host.label: (96, 1.0) for host in role_hosts},
            )
            sjf_hosts = default_sjf_hosts(50)
            sjf_hosts = cpu_safe_hosts(
                sjf_hosts,
                phase="sjf",
                total_games=50,
                cpu_by_label={host.label: (96, 1.0) for host in sjf_hosts},
            )
        self.assertTrue(all(host.rollout_parallel_threads == 2 for host in role_hosts))
        self.assertEqual(role_hosts[0].parallel_games, 45)
        self.assertTrue(all(host.parallel_games == 46 for host in role_hosts[1:]))
        self.assertTrue(all(host.rollout_parallel_threads == 8 for host in sjf_hosts))
        self.assertTrue(all(host.parallel_games == 11 for host in sjf_hosts))

    def test_cpu_planner_allows_bounded_multiwave_capacity(self) -> None:
        hosts = [EvalHost("xl", None, 400, 8, "0-15")]
        with mock.patch.dict(
            "os.environ",
            {
                "AGZ_DISTRIBUTED_EVAL_ROLE_THREADS_PER_GAME": "2",
                "AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_CORES": "2",
            },
        ):
            resolved = cpu_safe_hosts(
                hosts,
                phase="role",
                total_games=400,
                cpu_by_label={"xl": (16, 1.0)},
            )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].parallel_games, 7)

    def test_finalized_cycle_removes_stale_pid_without_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "training"
            train_dir.mkdir(parents=True)
            (train_dir / "current_training.json").write_text(
                json.dumps({"cycle_finalized_at_utc": "2026-07-10T09:00:00Z"}),
                encoding="utf-8",
            )
            (train_dir / "current.pid").write_text("12345\n", encoding="utf-8")
            self.assertTrue(_finalize_completed_training_cycle_if_needed(root, train_dir))
            self.assertFalse((train_dir / "current.pid").exists())

    def test_memory_limit_caps_aggregate_host_parallelism(self) -> None:
        gib = 1024**3
        hosts = [EvalHost("xl", None, 100, 90, "6-95")]
        with mock.patch.dict(
            "os.environ",
            {
                "AGZ_DISTRIBUTED_EVAL_GAME_MEMORY_GIB": "4",
                "AGZ_DISTRIBUTED_EVAL_MEMORY_RESERVE_GIB": "16",
                "AGZ_DISTRIBUTED_EVAL_MEMORY_RESERVE_FRACTION": "0.25",
            },
        ):
            limited = memory_safe_hosts(
                hosts,
                active_blocks=4,
                meminfo_by_label={"xl": (200 * gib, 100 * gib)},
            )
        self.assertEqual(limited[0].parallel_games, 12)
        self.assertEqual(limited[0].memory_available_bytes, 100 * gib)
        self.assertEqual(limited[0].memory_reserve_bytes, 50 * gib)

    def test_memory_limit_fails_closed_when_blocks_cannot_fit(self) -> None:
        gib = 1024**3
        hosts = [EvalHost("worker1", "worker", 40, 34, "61-94")]
        with mock.patch.dict(
            "os.environ",
            {
                "AGZ_DISTRIBUTED_EVAL_GAME_MEMORY_GIB": "4",
                "AGZ_DISTRIBUTED_EVAL_MEMORY_RESERVE_GIB": "16",
                "AGZ_DISTRIBUTED_EVAL_MEMORY_RESERVE_FRACTION": "0.25",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "insufficient RAM"):
                memory_safe_hosts(
                    hosts,
                    active_blocks=4,
                    meminfo_by_label={"worker1": (64 * gib, 20 * gib)},
                )

    def test_role_plan_covers_all_games_in_one_wave(self) -> None:
        blocks = {
            f"block{i}": _arena_command(f"block{i}", 71_030_000 + (i // 2) * 5_000)
            for i in range(4)
        }
        hosts = default_role_hosts()
        chunks = plan_role_chunks(
            blocks,
            scratch_root=Path("/tmp/.distributed_eval_scratch/test"),
            hosts=hosts,
        )
        self.assertEqual(sum(chunk.num_games for chunk in chunks), 400)
        for host in hosts:
            host_chunks = [chunk for chunk in chunks if chunk.host == host]
            self.assertLessEqual(
                sum(chunk.parallel_games for chunk in host_chunks),
                host.parallel_games,
            )
        self.assertEqual({chunk.wave_index for chunk in chunks}, {0})

    def test_exp3_pair_safe_plan_starts_all_400_games_in_one_wave(self) -> None:
        blocks = {
            f"block{i}": _arena_command(f"block{i}", 71_030_000 + (i // 2) * 5_000)
            for i in range(4)
        }
        env = {
            "AGZ_DISTRIBUTED_EVAL_XL_GAMES": "32",
            "AGZ_DISTRIBUTED_EVAL_XL_PARALLEL": "46",
            "AGZ_DISTRIBUTED_EVAL_XL_CPUSET": "4-95",
            "AGZ_DISTRIBUTED_EVAL_WORKER_PARALLEL": "46",
            "AGZ_DISTRIBUTED_EVAL_WORKER_CPUSET": "2-95",
            "AGZ_DISTRIBUTED_EVAL_ROLE_THREADS_PER_GAME": "2",
            "AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_CORES": "2",
        }
        with mock.patch.dict("os.environ", env):
            hosts = default_role_hosts()
            hosts = cpu_safe_hosts(
                hosts,
                phase="role",
                total_games=400,
                cpu_by_label={host.label: (96, 1.0) for host in hosts},
            )
            chunks = plan_role_chunks(
                blocks,
                scratch_root=Path("/tmp/.distributed_eval_scratch/exp3"),
                hosts=hosts,
            )

        self.assertEqual(sum(chunk.num_games for chunk in chunks), 400)
        for host in hosts:
            host_chunks = [chunk for chunk in chunks if chunk.host == host]
            self.assertLessEqual(
                sum(chunk.parallel_games for chunk in host_chunks),
                host.parallel_games,
            )
        for chunk in chunks:
            index = chunk.command.index("--rollout-parallel-threads")
            self.assertEqual(chunk.command[index + 1], "2")
            index = chunk.command.index("--num-parallel-games")
            self.assertEqual(int(chunk.command[index + 1]), chunk.parallel_games)

    def test_exp3_140_game_role_eval_queues_all_hosts_in_one_wave(self) -> None:
        blocks = {
            f"block{i}": _arena_command(
                f"block{i}",
                71_030_000 + (i // 2) * 5_000,
                games=140,
            )
            for i in range(4)
        }
        hosts = [
            EvalHost(
                f"host{i}",
                None,
                140,
                45,
                "0-89",
                rollout_parallel_threads=2,
            )
            for i in range(9)
        ]
        chunks = plan_role_chunks(
            blocks,
            scratch_root=Path("/tmp/.distributed_eval_scratch/exp3_140"),
            hosts=hosts,
        )

        self.assertEqual(sum(chunk.num_games for chunk in chunks), 560)
        self.assertEqual({chunk.wave_index for chunk in chunks}, {0})
        self.assertEqual({chunk.host.label for chunk in chunks}, {host.label for host in hosts})
        self.assertTrue(any(chunk.num_games > chunk.parallel_games for chunk in chunks))
        for host in hosts:
            live_games = sum(
                chunk.parallel_games for chunk in chunks if chunk.host == host
            )
            self.assertLessEqual(live_games, host.parallel_games)
        for left, right in (("block0", "block1"), ("block2", "block3")):
            left_plan = sorted(
                (c.wave_index, c.host.label, c.game_offset, c.num_games)
                for c in chunks
                if c.block_name == left
            )
            right_plan = sorted(
                (c.wave_index, c.host.label, c.game_offset, c.num_games)
                for c in chunks
                if c.block_name == right
            )
            self.assertEqual(left_plan, right_plan)


    def test_paired_blocks_preserve_identical_offsets_and_4k(self) -> None:
        blocks = {
            "baseline": _arena_command("baseline", 71_030_000),
            "candidate": _arena_command("candidate", 71_030_000),
            "ctrl_baseline": _arena_command("ctrl_baseline", 71_035_000),
            "ctrl_candidate": _arena_command("ctrl_candidate", 71_035_000),
        }
        chunks = plan_role_chunks(
            blocks,
            scratch_root=Path("/tmp/.distributed_eval_scratch/test"),
            hosts=default_role_hosts(),
        )
        for left, right in (("baseline", "candidate"), ("ctrl_baseline", "ctrl_candidate")):
            left_chunks = sorted((c for c in chunks if c.block_name == left), key=lambda c: c.game_offset)
            right_chunks = sorted((c for c in chunks if c.block_name == right), key=lambda c: c.game_offset)
            self.assertEqual(
                [(c.host.label, c.game_offset, c.num_games) for c in left_chunks],
                [(c.host.label, c.game_offset, c.num_games) for c in right_chunks],
            )
            left_offsets = [offset for chunk in left_chunks for offset in range(chunk.game_offset, chunk.game_offset + chunk.num_games)]
            right_offsets = [offset for chunk in right_chunks for offset in range(chunk.game_offset, chunk.game_offset + chunk.num_games)]
            self.assertEqual(left_offsets, list(range(100)))
            self.assertEqual(right_offsets, left_offsets)
        for chunk in chunks:
            self.assertEqual(chunk.command[chunk.command.index("--shared-root-mcts-iterations") + 1], "4000")
            self.assertEqual(chunk.command[chunk.command.index("--history-hops-offset") + 1], str(chunk.game_offset))
            self.assertIn("--history-hops-prefix-stable", chunk.command)

    def test_prefix_stable_hops_match_single_launcher_plan(self) -> None:
        def planned(count: int, offset: int) -> list[int]:
            return _planned_history_hops(SimpleNamespace(
                num_games=count,
                history_hops_offset=offset,
                history_hops_min=0,
                history_hops_max=105,
                history_seed=2129,
                history_hops_unique=True,
                history_hops_force_zero=False,
                history_hops_prefix_stable=True,
            ))

        chunk_sizes = [25, 10, 10, 10, 9, 9, 9, 9, 9]
        offset = 0
        combined: list[int] = []
        for count in chunk_sizes:
            combined.extend(planned(count, offset))
            offset += count
        self.assertEqual(offset, 100)
        self.assertEqual(combined, planned(100, 0))
        self.assertEqual(len(set(combined)), 100)

    def test_single_game_tasks_apply_distinct_history_offsets(self) -> None:
        assigned: list[int] = []
        for offset in range(140):
            args = SimpleNamespace(
                num_games=1,
                history_hops_offset=offset,
                history_hops_min=0,
                history_hops_max=145,
                history_seed=2134,
                history_hops_unique=True,
                history_hops_force_zero=False,
                history_hops_prefix_stable=True,
            )
            assigned.append(_pin_single_game_history_hop(args))
            self.assertEqual(args.history_hops_min, assigned[-1])
            self.assertEqual(args.history_hops_max, assigned[-1])
            self.assertEqual(args.history_hops_offset, 0)

        self.assertEqual(len(set(assigned)), 140)

    def test_sjf_plan_is_one_wave_and_xl_has_most_games(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "AGZ_DISTRIBUTED_EVAL_SJF_THREADS_PER_GAME": "8",
                "AGZ_DISTRIBUTED_EVAL_CPU_RESERVE_CORES": "2",
            },
        ):
            hosts = default_sjf_hosts(50)
            hosts = cpu_safe_hosts(
                hosts,
                phase="sjf",
                total_games=50,
                cpu_by_label={host.label: (96, 1.0) for host in hosts},
            )
            chunks = plan_single_block_chunks(
                "SJF_256_Game",
                _arena_command("SJF_256_Game", 81_030_000, games=50),
                scratch_root=Path("/tmp/.distributed_eval_scratch/sjf"),
                hosts=hosts,
            )
        self.assertEqual(sum(chunk.num_games for chunk in chunks), 50)
        self.assertTrue(all(chunk.parallel_games == chunk.num_games for chunk in chunks))
        self.assertGreater(chunks[0].num_games, max(chunk.num_games for chunk in chunks[1:]))
        for chunk in chunks:
            index = chunk.command.index("--rollout-parallel-threads")
            self.assertEqual(chunk.command[index + 1], "8")

    def test_split_sjf_plan_runs_both_cycles_in_one_wave(self) -> None:
        commands = {
            "cycle1_trivial": _arena_command("cycle1", 81_010_000, games=3),
            "cycle2_model": _arena_command("cycle2", 81_010_000, games=3),
        }
        hosts = [
            EvalHost("xl", None, 3, 3, "0-2", rollout_parallel_threads=1),
            EvalHost("worker1", "worker", 3, 3, "3-5", rollout_parallel_threads=1),
        ]
        chunks = plan_independent_block_chunks(
            commands,
            scratch_root=Path("/tmp/.distributed_eval_scratch/split_sjf"),
            hosts=hosts,
        )
        self.assertEqual(sum(chunk.num_games for chunk in chunks), 6)
        for block_name in commands:
            block_chunks = sorted(
                (chunk for chunk in chunks if chunk.block_name == block_name),
                key=lambda chunk: chunk.game_offset,
            )
            offsets = [
                offset
                for chunk in block_chunks
                for offset in range(chunk.game_offset, chunk.game_offset + chunk.num_games)
            ]
            self.assertEqual(offsets, [0, 1, 2])
            self.assertTrue(
                all("--history-hops-prefix-stable" in chunk.command for chunk in block_chunks)
            )
        for host in hosts:
            assigned = sum(
                chunk.num_games for chunk in chunks if chunk.host.label == host.label
            )
            self.assertLessEqual(assigned, host.parallel_games)

    def test_split_sjf_merge_restores_standard_paired_result(self) -> None:
        fields = [
            "game_id",
            "history_hops",
            "model_kind",
            "model_checkpoint",
            "cycle1_label",
            "cycle1_slo_violations",
            "cycle1_total_lateness",
            "cycle1_total_cost",
            "cycle1_end_reason",
            "cycle2_label",
            "cycle2_slo_violations",
            "cycle2_total_lateness",
            "cycle2_total_cost",
            "cycle2_end_reason",
            "cost_delta_cycle2_minus_cycle1",
            "better_cycle",
            "cycle1_log_file",
            "cycle2_log_file",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trivial_dir = root / "cycle1_trivial"
            model_dir = root / "cycle2_model"
            output_dir = root / "SJF_256_Game"
            for directory in (trivial_dir, model_dir):
                (directory / "arena_games").mkdir(parents=True)
            for game_id, hops, trivial_cost, model_cost in (
                (10, 4, 7.5, 6.0),
                (11, 9, 3.0, 4.0),
            ):
                cycle1_log = f"game_{game_id}_trivial.csv"
                cycle2_log = f"game_{game_id}_model.csv"
                (trivial_dir / "arena_games" / cycle1_log).write_text(
                    "step,cost\n1,1\n", encoding="utf-8"
                )
                (model_dir / "arena_games" / cycle2_log).write_text(
                    "step,cost\n1,2\n", encoding="utf-8"
                )
                trivial_row = {
                    "game_id": game_id,
                    "history_hops": hops,
                    "model_kind": "dnn",
                    "model_checkpoint": "/tmp/model",
                    "cycle1_label": "model_adv_vs_trivial_ctrl",
                    "cycle1_slo_violations": 1,
                    "cycle1_total_lateness": trivial_cost,
                    "cycle1_total_cost": trivial_cost,
                    "cycle1_end_reason": "time_limit",
                    "cycle2_label": "",
                    "cycle2_slo_violations": 0,
                    "cycle2_total_lateness": 0,
                    "cycle2_total_cost": 0,
                    "cycle2_end_reason": "",
                    "cost_delta_cycle2_minus_cycle1": 0,
                    "better_cycle": "model_adv_vs_trivial_ctrl",
                    "cycle1_log_file": f"/scratch/{cycle1_log}",
                    "cycle2_log_file": "",
                }
                model_row = {
                    **trivial_row,
                    "cycle1_slo_violations": 0,
                    "cycle1_total_lateness": 0,
                    "cycle1_total_cost": 0,
                    "cycle1_end_reason": "",
                    "cycle2_label": "model_adv_vs_model_ctrl",
                    "cycle2_slo_violations": 2,
                    "cycle2_total_lateness": model_cost,
                    "cycle2_total_cost": model_cost,
                    "cycle2_end_reason": "time_limit",
                    "better_cycle": "model_adv_vs_model_ctrl",
                    "cycle1_log_file": "",
                    "cycle2_log_file": f"/scratch/{cycle2_log}",
                }
                for directory, row in (
                    (trivial_dir, trivial_row),
                    (model_dir, model_row),
                ):
                    csv_path = directory / "arena_results.csv"
                    write_header = not csv_path.exists()
                    with csv_path.open("a", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)
                    for filename in ("planned_games.csv", "job_status.csv"):
                        meta_path = directory / filename
                        meta_header = ["game_id", "history_hops", "status"]
                        write_meta_header = not meta_path.exists()
                        with meta_path.open("a", encoding="utf-8", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=meta_header)
                            if write_meta_header:
                                writer.writeheader()
                            writer.writerow(
                                {"game_id": game_id, "history_hops": hops, "status": "ok"}
                            )

            result = merge_split_sjf_cycles(
                trivial_dir=trivial_dir,
                model_dir=model_dir,
                output_dir=output_dir,
                expected_games=2,
            )
            with result.open("r", encoding="utf-8", newline="") as handle:
                rows = {int(row["game_id"]): row for row in csv.DictReader(handle)}
            self.assertAlmostEqual(float(rows[10]["cost_delta_cycle2_minus_cycle1"]), -1.5)
            self.assertEqual(rows[10]["better_cycle"], "model_adv_vs_model_ctrl")
            self.assertAlmostEqual(float(rows[11]["cost_delta_cycle2_minus_cycle1"]), 1.0)
            self.assertEqual(rows[11]["better_cycle"], "model_adv_vs_trivial_ctrl")
            self.assertEqual(len(list((output_dir / "arena_games").glob("*.csv"))), 4)
            self.assertTrue((output_dir / "split_cycle_manifest.json").is_file())

    def test_merge_rejects_duplicate_game_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = []
            for index in range(2):
                path = root / f"part{index}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["game_id", "cycle2_total_cost"])
                    writer.writeheader()
                    writer.writerow({"game_id": 10, "cycle2_total_cost": index})
                parts.append(path)
            with self.assertRaisesRegex(RuntimeError, "duplicate distributed result"):
                _merge_csv(parts, root / "merged.csv", expected_games=2)


if __name__ == "__main__":
    unittest.main()
