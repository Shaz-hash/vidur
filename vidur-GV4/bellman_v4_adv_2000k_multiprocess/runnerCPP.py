"""Run GV3 depth-1 Bellman arena games with native C++ child evaluation.

This is the non-MCTS counterpart of ``arena_mcts_value_runnerCPP.py``: Python
keeps the arena harness, history generation, and CSV logging, while C++ evaluates
all canonical root actions using reward + discounted HGB bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from vidur.Game_Version3.DNN.native_selfplay import (
    _cfg_payload,
    attach_execution_predictor_payload,
)
from vidur.Game_Version3.Model_Tester.config import DEFAULT_MODEL_TESTER_CONFIG
from vidur.Game_Version3.tests import native_logger_tests as nlt
from vidur.bellman_v4_adv.arena_mcts_value_runnerCPP import (
    _export_hgb_to_native_text,
    _import_native_cpp,
    _limit_native_threads,
    _repo_root,
)
from vidur.bellman_v4_adv_2000k_multiprocess import runner as tester_runner


_RUNTIME_ARGS: argparse.Namespace | None = None
_NATIVE_RUNTIME: Any | None = None
_NATIVE_MODEL_EXPORT_PATH: Path | None = None
_CFG_PAYLOAD_CACHE: dict[int, dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_label(value: Any) -> str:
    text = str(value or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _prepare_native_runtime(args: argparse.Namespace) -> None:
    global _NATIVE_RUNTIME, _NATIVE_MODEL_EXPORT_PATH
    native = _import_native_cpp(force_build=bool(args.force_build_native))
    model_path = Path(args.model_path).expanduser()
    model = joblib.load(model_path)
    setattr(args, "harness_model_path", str(model_path))
    if not callable(getattr(model, "infer_from_inputs", None)):
        from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper

        model = V4AdvHGBWrapper(model, feature_dim=int(args.feature_dim), model_tag=f"wrapped:{model_path.name}")
        wrapper_path = Path(args.output_dir).expanduser() / "arena_model" / "v4_adv_hgb_wrapper.joblib"
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, wrapper_path, compress=3)
        setattr(args, "harness_model_path", str(wrapper_path))

    export_path = (
        Path(args.native_model_export_path).expanduser()
        if str(args.native_model_export_path or "")
        else Path(args.output_dir).expanduser() / "native_model" / "v4_adv_hgb_native_export.tsv"
    )
    _NATIVE_MODEL_EXPORT_PATH = _export_hgb_to_native_text(model, export_path)
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(_NATIVE_MODEL_EXPORT_PATH))
    _NATIVE_RUNTIME = runtime


def _get_cfg_payload(args: argparse.Namespace, cfg: Any, bundle: Any) -> dict[str, Any]:
    key = id(bundle.simulator)
    cached = _CFG_PAYLOAD_CACHE.get(key)
    if cached is not None:
        return cached
    pipeline_cfg = cfg.to_pipeline_cfg() if callable(getattr(cfg, "to_pipeline_cfg", None)) else cfg
    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, bundle.simulator)
    payload["use_model_bootstrap"] = bool(int(args.model_version) > 0) and not bool(args.disable_model_bootstrap)
    payload["native_search_mode"] = "depth_one"
    payload["root_dirichlet_noise_enabled"] = False
    payload["max_forced_hops"] = int(getattr(pipeline_cfg, "max_forced_hops_per_root", 0) or 0)
    _CFG_PAYLOAD_CACHE[key] = payload
    return payload


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _select_model_depth1_cpp_action(
    *,
    bundle: Any,
    cfg: Any,
    expanded: Any,
    model: Any,
) -> tuple[Any | None, dict[str, Any]]:
    del model
    args = _RUNTIME_ARGS
    if args is None:
        raise RuntimeError("CPP depth-1 runner args were not initialised")
    if _NATIVE_RUNTIME is None:
        raise RuntimeError("native HGB runtime was not initialised")

    player = str(expanded.player)
    valid_indices = [int(x) for x in list(expanded.valid_indices)]
    actions_by_index = list(expanded.actions_by_index)
    if not valid_indices:
        return None, {
            "selection_mode": f"model_depth1_cpp_{player}_no_valid_action",
            "valid_action_count": 0,
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    native = _import_native_cpp(force_build=False)
    payload = dict(_get_cfg_payload(args, cfg, bundle))
    native_out = native.search_mcts_hgb226(
        _NATIVE_RUNTIME,
        int(args.model_version),
        nlt._native_state_payload(bundle.env, expanded.search_state),
        payload,
        1,
        str(player),
        0,
        0,
        int(getattr(args, "current_game_id", 0) or 0),
        -1,
        int(args.seed),
        False,
        False,
        "",
        "",
    )

    q_values = [float(x) for x in list(native_out.get("root_action_values", []) or [])]
    rewards = [float(x) for x in list(native_out.get("root_action_rewards", []) or [])]
    discounts = [float(x) for x in list(native_out.get("root_action_discounts", []) or [])]
    bootstraps = [float(x) for x in list(native_out.get("root_action_bootstraps", []) or [])]
    child_cost_by_idx = {
        int(c.get("index", -1)): float(c.get("state_cost", 0.0) or 0.0)
        for c in list(native_out.get("children", []) or [])
        if int(c.get("index", -1)) >= 0
    }
    alias_to_canon = {int(k): int(v) for k, v in dict(native_out.get("action_alias_to_canonical", {}) or {}).items()}
    canonical_indices = sorted({int(alias_to_canon.get(i, i)) for i in valid_indices})

    rows: list[dict[str, Any]] = []
    for canon_idx in canonical_indices:
        action = actions_by_index[canon_idx] if 0 <= canon_idx < len(actions_by_index) else None
        q = q_values[canon_idx] if 0 <= canon_idx < len(q_values) else float("nan")
        if not _finite(q):
            q = float("-inf") if player == "controller" else float("inf")
        rows.append(
            {
                "canon_idx": int(canon_idx),
                "action_repr": repr(action),
                "q_value": float(q),
                "reward": rewards[canon_idx] if 0 <= canon_idx < len(rewards) else 0.0,
                "discount": discounts[canon_idx] if 0 <= canon_idx < len(discounts) else 1.0,
                "bootstrap_value": bootstraps[canon_idx] if 0 <= canon_idx < len(bootstraps) else 0.0,
                "child_cost": child_cost_by_idx.get(int(canon_idx), 0.0),
            }
        )

    if not rows:
        return None, {
            "selection_mode": f"model_depth1_cpp_{player}_empty_canonical_set",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    reverse = bool(player != "adversary")
    rows_sorted = sorted(rows, key=lambda r: ((-float(r["q_value"])) if reverse else float(r["q_value"]), int(r["canon_idx"])))
    best = rows_sorted[0]
    best_idx = int(best["canon_idx"])
    top5 = rows_sorted[:5]
    selection_mode = "model_depth1_cpp_ctrl_argmax_q" if player == "controller" else "model_depth1_cpp_adv_argmin_q"
    return actions_by_index[best_idx], {
        "selection_mode": selection_mode,
        "valid_action_count": int(len(valid_indices)),
        "iterations_requested": 0,
        "iterations_used": int(len(canonical_indices)),
        "chosen_q_value": float(best["q_value"]),
        "chosen_reward": float(best["reward"]),
        "chosen_discount": float(best["discount"]),
        "chosen_bootstrap": float(best["bootstrap_value"]),
        "chosen_child_cost": float(best["child_cost"]),
        "candidate_ranking_mode": "q_desc" if reverse else "q_asc",
        "candidate_top5_action_reprs": [str(r["action_repr"]) for r in top5],
        "candidate_top5_q_values": [float(r["q_value"]) for r in top5],
        "candidate_top5_rewards": [float(r["reward"]) for r in top5],
        "candidate_top5_discounts": [float(r["discount"]) for r in top5],
        "candidate_top5_bootstraps": [float(r["bootstrap_value"]) for r in top5],
        "candidate_top5_child_costs": [float(r["child_cost"]) for r in top5],
        "candidate_ranked_rows": [
            {
                "rank": int(rank),
                "canon_action_index": int(r["canon_idx"]),
                "action_repr": str(r["action_repr"]),
                "q_value": float(r["q_value"]),
                "immediate_reward": float(r["reward"]),
                "discount": float(r["discount"]),
                "bootstrap_value": float(r["bootstrap_value"]),
                "child_cost": float(r["child_cost"]),
            }
            for rank, r in enumerate(rows_sorted, start=1)
        ],
    }


def _planned_history_hops(args: argparse.Namespace) -> list[int]:
    n = int(args.num_games)
    offset = max(0, int(getattr(args, "history_hops_offset", 0) or 0))
    planned_n = n + offset
    lo = int(args.history_hops_min)
    hi = int(args.history_hops_max)
    rng = random.Random(int(args.history_seed))
    if bool(args.history_hops_unique):
        population = list(range(lo, hi + 1))
        if planned_n > len(population):
            raise ValueError("num_games exceeds unique history-hop capacity")
        hops = rng.sample(population, k=planned_n)
    else:
        hops = [int(rng.randint(lo, hi)) for _ in range(planned_n)]
    if bool(args.history_hops_force_zero):
        if not (lo <= 0 <= hi):
            raise ValueError("history hop range must include 0 when history_hops_force_zero=True")
        if 0 in hops:
            zi = hops.index(0)
            hops[0], hops[zi] = hops[zi], hops[0]
        else:
            hops[0] = 0
    return [int(x) for x in hops[offset:]]


def _worker_command(args: argparse.Namespace, *, game_id: int, hop: int, job_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vidur.bellman_v4_adv_2000k_multiprocess.runnerCPP",
        "--launcher-worker",
        "--model-path",
        str(args.model_path),
        "--model-version",
        str(int(args.model_version)),
        "--feature-dim",
        str(int(args.feature_dim)),
        "--output-dir",
        str(job_dir),
        "--game-id-start",
        str(int(game_id)),
        "--num-games",
        "1",
        "--num-parallel-games",
        "1",
        "--worker-threads",
        str(int(args.worker_threads)),
        "--trivial-budget-tokens",
        str(int(args.trivial_budget_tokens)),
        "--arena-time-limit-sec",
        str(float(args.arena_time_limit_sec)),
        "--arena-max-total-turns",
        str(int(args.arena_max_total_turns)),
        "--arena-max-controller-cleanup-steps",
        str(int(args.arena_max_controller_cleanup_steps)),
        "--history-hops-min",
        str(int(hop)),
        "--history-hops-max",
        str(int(hop)),
        "--no-history-hops-force-zero",
        "--seed",
        str(int(args.seed)),
    ]
    if bool(args.skip_model_ctrl_cycle):
        cmd.append("--skip-model-ctrl-cycle")
    if bool(args.disable_model_bootstrap):
        cmd.append("--disable-model-bootstrap")
    if str(args.native_model_export_path or ""):
        export_name = Path(str(args.native_model_export_path)).name
        cmd.extend(["--native-model-export-path", str(job_dir / "native_model" / export_name)])
    if bool(args.no_arena_game_logs):
        cmd.append("--no-arena-game-logs")
    if bool(args.write_model_action_detail_logs):
        cmd.append("--write-model-action-detail-logs")
    return cmd


def _write_planned_games(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    with (output_dir / "planned_games.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_index", "game_id", "history_hops", "job_output_dir"])
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "game_index": int(job["game_index"]),
                    "game_id": int(job["game_id"]),
                    "history_hops": int(job["history_hops"]),
                    "job_output_dir": str(job["job_output_dir"]),
                }
            )


def _write_job_status(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp",
        "game_index",
        "game_id",
        "history_hops",
        "status",
        "pid",
        "returncode",
        "elapsed_sec",
        "job_output_dir",
        "log_file",
        "arena_results_csv",
    ]
    with (output_dir / "job_status.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({k: job.get(k, "") for k in fields})


def _copy_job_arena_logs(job: dict[str, Any], output_dir: Path) -> None:
    dst_dir = output_dir / "arena_games"
    dst_dir.mkdir(parents=True, exist_ok=True)
    job_games = Path(job["job_output_dir"]) / "arena_games"
    if not job_games.exists():
        return
    for src in sorted(job_games.glob("*.csv")):
        if src.name.endswith("_model_action_details.csv"):
            continue
        shutil.copy2(src, dst_dir / src.name)


def _merge_job_results(output_dir: Path) -> int:
    result_files = sorted((output_dir / "jobs").glob("game_*_hop_*/arena_results.csv"))
    out_path = output_dir / "arena_results.csv"
    if not result_files:
        return 0
    merged = 0
    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer: csv.DictWriter[str] | None = None
        for path in result_files:
            with path.open(newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=list(reader.fieldnames or []))
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    merged += 1
    return merged


def _cleanup_process_group(proc: subprocess.Popen[Any] | None, pid: int) -> None:
    if proc is None or pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        time.sleep(0.5)
    except ProcessLookupError:
        pass
    except Exception:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass
    try:
        proc.wait(timeout=1)
    except Exception:
        pass


def _run_parallel_launcher(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser()
    jobs_dir = output_dir / "jobs"
    logs_dir = output_dir / "launcher_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "arena_games").mkdir(parents=True, exist_ok=True)

    if bool(args.force_build_native):
        _import_native_cpp(force_build=True)

    hops = _planned_history_hops(args)
    jobs: list[dict[str, Any]] = []
    for game_index, hop in enumerate(hops):
        game_id = int(args.game_id_start) + int(game_index)
        job_dir = jobs_dir / f"game_{game_id}_hop_{int(hop)}"
        log_file = logs_dir / f"game_{game_id}_hop_{int(hop)}.log"
        jobs.append(
            {
                "timestamp": "",
                "game_index": int(game_index),
                "game_id": int(game_id),
                "history_hops": int(hop),
                "status": "pending",
                "pid": "",
                "returncode": "",
                "elapsed_sec": "",
                "job_output_dir": str(job_dir),
                "log_file": str(log_file),
                "arena_results_csv": str(job_dir / "arena_results.csv"),
                "process": None,
                "log_handle": None,
                "start_time": None,
            }
        )

    _write_planned_games(output_dir, jobs)
    _write_job_status(output_dir, jobs)

    max_parallel = max(1, int(args.num_parallel_games))
    poll_sec = max(0.25, float(args.launcher_poll_sec))
    print(
        f"[depth1-cpp-launcher] out={output_dir} games={len(jobs)} parallel={max_parallel} "
        f"time_limit={float(args.arena_time_limit_sec)}s",
        flush=True,
    )

    next_idx = 0
    running: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    start_all = time.time()
    try:
        while completed + failed < len(jobs):
            while next_idx < len(jobs) and len(running) < max_parallel:
                job = jobs[next_idx]
                job_dir = Path(job["job_output_dir"])
                job_dir.mkdir(parents=True, exist_ok=True)
                log_handle = Path(job["log_file"]).open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    _worker_command(args, game_id=int(job["game_id"]), hop=int(job["history_hops"]), job_dir=job_dir),
                    cwd=str(_repo_root()),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=True,
                )
                job["process"] = proc
                job["log_handle"] = log_handle
                job["pid"] = int(proc.pid)
                job["status"] = "running"
                job["timestamp"] = _utc_now()
                job["start_time"] = time.time()
                running.append(job)
                print(f"[depth1-cpp-launcher] started game={job['game_id']} hop={job['history_hops']} pid={proc.pid}", flush=True)
                next_idx += 1

            time.sleep(poll_sec)
            still_running: list[dict[str, Any]] = []
            for job in running:
                proc = job.get("process")
                if proc is None:
                    continue
                rc = proc.poll()
                if rc is None:
                    job["elapsed_sec"] = f"{time.time() - float(job['start_time']):.3f}"
                    still_running.append(job)
                    continue

                job["returncode"] = int(rc)
                job["elapsed_sec"] = f"{time.time() - float(job['start_time']):.3f}"
                job["timestamp"] = _utc_now()
                try:
                    job["log_handle"].close()
                except Exception:
                    pass
                _cleanup_process_group(proc, int(job.get("pid") or 0))
                job["process"] = None
                job["log_handle"] = None
                gc.collect()
                if int(rc) == 0:
                    job["status"] = "ok"
                    completed += 1
                    _copy_job_arena_logs(job, output_dir)
                    print(f"[depth1-cpp-launcher] done game={job['game_id']} hop={job['history_hops']}", flush=True)
                else:
                    job["status"] = "failed"
                    failed += 1
                    print(f"[depth1-cpp-launcher] FAILED game={job['game_id']} hop={job['history_hops']} rc={rc}", flush=True)

            running = still_running
            merged = _merge_job_results(output_dir)
            _write_job_status(output_dir, jobs)
            print(
                "[depth1-cpp-launcher] progress done={} failed={} running={} pending={} merged={} elapsed={}s".format(
                    completed,
                    failed,
                    len(running),
                    len(jobs) - next_idx,
                    merged,
                    int(time.time() - start_all),
                ),
                flush=True,
            )
    finally:
        for job in running:
            try:
                handle = job.get("log_handle")
                if handle is not None:
                    handle.close()
            except Exception:
                pass
            _cleanup_process_group(job.get("process"), int(job.get("pid") or 0))
        _merge_job_results(output_dir)
        _write_job_status(output_dir, jobs)
        gc.collect()

    if failed:
        raise RuntimeError(f"{failed} depth-1 CPP arena jobs failed; see {logs_dir}")
    print(f"[depth1-cpp-launcher] complete failures=0 out={output_dir}", flush=True)


def _run_single_game(args: argparse.Namespace) -> None:
    global _RUNTIME_ARGS
    _RUNTIME_ARGS = args
    _limit_native_threads(int(args.worker_threads))
    _prepare_native_runtime(args)

    trivial_policy = replace(
        DEFAULT_MODEL_TESTER_CONFIG.trivial_policy,
        budget_tokens=int(args.trivial_budget_tokens),
    )
    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind="classical_joblib",
        model_checkpoint_path=str(getattr(args, "harness_model_path", args.model_path)),
        output_dir=str(args.output_dir),
        num_games=int(args.num_games),
        game_id_start=int(args.game_id_start),
        environment_lang="python",
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        history_hops_unique=bool(args.history_hops_unique),
        history_hops_force_zero=bool(args.history_hops_force_zero),
        arena_time_limit_sec=float(args.arena_time_limit_sec),
        arena_max_total_turns=int(args.arena_max_total_turns),
        arena_max_controller_cleanup_steps=int(args.arena_max_controller_cleanup_steps),
        bootstrap_model_version=int(args.model_version),
        native_torchscript_dir=str(Path(args.output_dir) / "native_torchscript"),
        write_arena_game_logs=not bool(args.no_arena_game_logs),
        write_model_action_detail_logs=bool(args.write_model_action_detail_logs),
        arena_num_processes=1,
        arena_worker_threads=1,
        arena_mp_start_method="spawn",
        trivial_policy=trivial_policy,
        skip_model_ctrl_cycle=bool(args.skip_model_ctrl_cycle),
    )

    old_selector = tester_runner._select_model_depth1_action
    try:
        tester_runner._select_model_depth1_action = _select_model_depth1_cpp_action
        out_csv = tester_runner.run_model_vs_trivial_tester(cfg)
        print(f"[depth1-cpp-game] completed {out_csv}", flush=True)
        if _NATIVE_MODEL_EXPORT_PATH is not None:
            print(f"[depth1-cpp-game] native_model_export={_NATIVE_MODEL_EXPORT_PATH}", flush=True)
    finally:
        tester_runner._select_model_depth1_action = old_selector
        _CFG_PAYLOAD_CACHE.clear()
        gc.collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model_Tester arena games with native C++ depth-1 HGB226 action selection.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-version", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=226)
    parser.add_argument("--disable-model-bootstrap", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--game-id-start", type=int, default=12_000_000)
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--num-parallel-games", type=int, default=5)
    parser.add_argument("--launcher-poll-sec", type=float, default=10.0)
    parser.add_argument("--launcher-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--native-model-export-path", default="")
    parser.add_argument("--force-build-native", action="store_true")
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--trivial-budget-tokens", type=int, default=512)
    parser.add_argument("--skip-model-ctrl-cycle", action="store_true")
    parser.add_argument("--arena-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--arena-max-total-turns", type=int, default=4096)
    parser.add_argument("--arena-max-controller-cleanup-steps", type=int, default=1024)
    parser.add_argument("--history-hops-min", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_min)
    parser.add_argument("--history-hops-max", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_max)
    parser.add_argument("--history-seed", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_seed)
    parser.add_argument("--history-hops-offset", type=int, default=0)
    parser.add_argument("--history-hops-unique", action=argparse.BooleanOptionalAction, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_unique)
    parser.add_argument("--history-hops-force-zero", action=argparse.BooleanOptionalAction, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_force_zero)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-arena-game-logs", action="store_true")
    parser.add_argument("--write-model-action-detail-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.num_games) > 1 and not bool(args.launcher_worker):
        _run_parallel_launcher(args)
        return
    _run_single_game(args)


if __name__ == "__main__":
    main()
