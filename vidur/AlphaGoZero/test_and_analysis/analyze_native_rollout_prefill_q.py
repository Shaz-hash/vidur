"""Measure native rollout-MCTS Q preference by controller prefill size."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
from vidur.Game_Version3.DNN.native_selfplay import (
    _cfg_payload,
    attach_execution_predictor_payload,
)
from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner
from vidur.Game_Version3.tests import native_logger_tests as nlt
from vidur.tests.native_allignment_tests.common import (
    import_native_cpp,
    make_args,
    prepare_python_roots,
)


REQUESTED_CATEGORIES = (
    "decode_only",
    "prefill_128",
    "prefill_256",
    "prefill_512",
    "prefill_1024",
)
ALL_CATEGORIES = REQUESTED_CATEGORIES + ("prefill_other", "other")
_WORKER: dict[str, Any] = {}


def _action_category(action: Any) -> str:
    prefill_tokens = sum(
        int(value) for value in (getattr(action, "prefill_allocations", {}) or {}).values()
    )
    decode_tokens = sum(
        int(value) for value in (getattr(action, "decode_allocations", {}) or {}).values()
    )
    if prefill_tokens == 0 and decode_tokens > 0:
        return "decode_only"
    if prefill_tokens in {128, 256, 512, 1024}:
        return f"prefill_{prefill_tokens}"
    if prefill_tokens > 0:
        return "prefill_other"
    return "other"


def _pending_prefill(payload: dict[str, Any]) -> tuple[int, int]:
    requests = list(payload.get("requests") or [])
    remaining = [
        max(
            0,
            int(request.get("num_prefill_tokens", 0))
            - int(request.get("num_processed_prefill_tokens", 0)),
        )
        for request in requests
        if not bool(request.get("completed", False))
        and not bool(request.get("dropped", False))
        and not bool(request.get("feature_only", False))
    ]
    positive = [value for value in remaining if value > 0]
    return sum(positive), len(positive)


def _worker_init(
    value_model: str,
    controller_policy_model: str,
    adversary_policy_model: str,
    cfg_payload: dict[str, Any],
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    native = import_native_cpp(build_if_missing=False)
    value_runtime = native.NewFeatures226HGBRuntime()
    value_runtime.load_model_export(value_model)
    controller_runtime = native.NativeHGBModelRuntime()
    controller_runtime.load_model_export(controller_policy_model)
    adversary_runtime = native.NativeHGBModelRuntime()
    adversary_runtime.load_model_export(adversary_policy_model)
    _WORKER.update(
        native=native,
        value_runtime=value_runtime,
        controller_runtime=controller_runtime,
        adversary_runtime=adversary_runtime,
        cfg_payload=cfg_payload,
    )


def _run_root(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    native_result = _WORKER["native"].search_mcts_hgb226_value_prior_hgb(
        _WORKER["value_runtime"],
        _WORKER["controller_runtime"],
        _WORKER["adversary_runtime"],
        int(task["controller_version"]),
        task["state_payload"],
        _WORKER["cfg_payload"],
        int(task["iterations"]),
        "controller",
        int(task["root_node_id"]),
        int(task["root_depth"]),
        0,
        int(task["sample_id"]),
        int(task["seed"]),
        False,
        False,
        "",
        "",
    )
    action_rows: list[dict[str, Any]] = []
    category_q: dict[str, list[float]] = {}
    rollout_reward_returns = list(
        native_result.get("root_action_rollout_reward_returns", [])
    )
    rollout_bootstrap_returns = list(
        native_result.get("root_action_rollout_bootstrap_returns", [])
    )
    for child in native_result.get("children", []):
        index = int(child["index"])
        visits = int(child.get("visits", 0))
        if visits <= 0:
            continue
        value_sum = float(child.get("value_sum", 0.0))
        q_value = value_sum / visits
        rollout_reward_return = (
            float(rollout_reward_returns[index])
            if index < len(rollout_reward_returns)
            else math.nan
        )
        rollout_bootstrap_return = (
            float(rollout_bootstrap_returns[index])
            if index < len(rollout_bootstrap_returns)
            else math.nan
        )
        category = str(task["action_categories"].get(str(index), "other"))
        category_q.setdefault(category, []).append(q_value)
        action_rows.append(
            {
                "sample_id": int(task["sample_id"]),
                "action_index": index,
                "category": category,
                "q_value": q_value,
                "visits": visits,
                "prior": float(child.get("prior", 0.0)),
                "reward": float(child.get("reward", 0.0)),
                "mean_discounted_rollout_reward_return": rollout_reward_return,
                "mean_discounted_rollout_cost": -rollout_reward_return,
                "mean_discounted_bootstrap_return": rollout_bootstrap_return,
                "q_decomposition_error": (
                    q_value - rollout_reward_return - rollout_bootstrap_return
                ),
                "action_repr": str(task["action_reprs"].get(str(index), "")),
            }
        )
    best_q = {category: max(values) for category, values in category_q.items()}
    preferred = max(best_q, key=lambda category: (best_q[category], -ALL_CATEGORIES.index(category)))
    best_action_alias = int(native_result.get("best_action_index", -1))
    aliases = {
        int(key): int(value)
        for key, value in dict(native_result.get("action_alias_to_canonical", {})).items()
    }
    best_action_index = aliases.get(best_action_alias, best_action_alias)
    rollout_trace_rows: list[dict[str, Any]] = []
    for step in native_result.get("rollout_trace_steps", []):
        root_action_index = int(step["root_action_index"])
        rollout_trace_rows.append(
            {
                "sample_id": int(task["sample_id"]),
                "root_id": int(task["source_root_id"]),
                "root_sim_time": float(step["root_sim_time"]),
                "pending_prefill_tokens": int(task["pending_prefill_tokens"]),
                "pending_prefill_requests": int(task["pending_prefill_requests"]),
                "rollout_category": str(step["root_action_category"]),
                "root_action_index": root_action_index,
                "search_iteration": int(step["sim_iteration"]),
                "leaf_evaluation_id": int(step["leaf_evaluation_id"]),
                "rollout_id": int(step["rollout_id"]),
                "rollout_step_number": int(step["step_number"]),
                "rollout_simulator_time": float(step["step_sim_time_after"]),
                "rollout_simulator_time_before": float(
                    step["step_sim_time_before"]
                ),
                "rollout_deadline": float(step["rollout_deadline"]),
                "rollout_step_cost": float(step["step_cost"]),
                "rollout_discounted_step_cost": float(
                    step["discounted_step_cost"]
                ),
                "rollout_cumulative_discounted_cost": float(
                    step["cumulative_discounted_cost"]
                ),
                "trajectory_reward_return_from_root": float(
                    step["trajectory_reward_return_from_root"]
                ),
                "trajectory_bootstrap_return_from_root": float(
                    step["trajectory_bootstrap_return_from_root"]
                ),
                "trajectory_total_return_from_root": float(
                    step["trajectory_total_return_from_root"]
                ),
                "step_phase": str(step["step_phase"]),
                "step_player": str(step["step_player"]),
                "step_action_index": int(step["step_action_index"]),
                "step_action_category": str(step["step_action_category"]),
                "root_action_repr": str(
                    task["action_reprs"].get(str(root_action_index), "")
                ),
            }
        )
    return {
        "sample_id": int(task["sample_id"]),
        "source_root_id": int(task["source_root_id"]),
        "history_hops": int(task["history_hops"]),
        "sim_time": float(task["sim_time"]),
        "pending_prefill_tokens": int(task["pending_prefill_tokens"]),
        "pending_prefill_requests": int(task["pending_prefill_requests"]),
        "canonical_action_count": len(action_rows),
        "root_visits": int(native_result.get("root_visits", 0)),
        "root_q": (
            float(native_result.get("root_value_sum", 0.0))
            / max(1, int(native_result.get("root_visits", 0)))
        ),
        "preferred_q_category": preferred,
        "chosen_visit_category": str(
            task["action_categories"].get(str(best_action_index), "other")
        ),
        "best_q": best_q,
        "action_rows": action_rows,
        "rollout_trace_rows": rollout_trace_rows,
        "elapsed_sec": time.perf_counter() - started,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    for category in ALL_CATEGORIES:
        best_values = [
            float(result["best_q"][category])
            for result in results
            if category in result["best_q"]
        ]
        all_action_values = [
            float(row["q_value"])
            for result in results
            for row in result["action_rows"]
            if row["category"] == category
        ]
        preferred = sum(result["preferred_q_category"] == category for result in results)
        categories[category] = {
            "available_states": len(best_values),
            "preferred_states": preferred,
            "preferred_fraction_all_states": preferred / len(results) if results else None,
            "mean_best_q": statistics.fmean(best_values) if best_values else None,
            "median_best_q": statistics.median(best_values) if best_values else None,
            "p25_best_q": _quantile(best_values, 0.25),
            "p75_best_q": _quantile(best_values, 0.75),
            "action_observations": len(all_action_values),
            "mean_all_action_q": (
                statistics.fmean(all_action_values) if all_action_values else None
            ),
        }
    pairwise: dict[str, Any] = {}
    for category in REQUESTED_CATEGORIES[1:]:
        pairs = [
            (
                float(result["best_q"][category]),
                float(result["best_q"]["decode_only"]),
            )
            for result in results
            if category in result["best_q"] and "decode_only" in result["best_q"]
        ]
        gaps = [prefill_q - decode_q for prefill_q, decode_q in pairs]
        pairwise[f"{category}_vs_decode_only"] = {
            "comparable_states": len(pairs),
            "prefill_better": sum(gap > 1e-9 for gap in gaps),
            "decode_better": sum(gap < -1e-9 for gap in gaps),
            "ties": sum(abs(gap) <= 1e-9 for gap in gaps),
            "mean_q_gap_prefill_minus_decode": statistics.fmean(gaps) if gaps else None,
            "median_q_gap_prefill_minus_decode": statistics.median(gaps) if gaps else None,
        }
    return {
        "samples": len(results),
        "controller_version": int(args.controller_version),
        "adversary_version": int(args.adversary_version),
        "iterations": int(args.iterations),
        "rollout_count": int(args.rollout_count),
        "rollout_horizon_sec": float(args.rollout_horizon_sec),
        "discount_factor": float(args.discount_factor),
        "seed": int(args.seed),
        "preference_definition": "maximum backed-up child Q among expanded canonical actions in each category",
        "q_direction": "higher (less negative) is better for the controller",
        "mean_search_sec": statistics.fmean(float(result["elapsed_sec"]) for result in results),
        "categories": categories,
        "pairwise": pairwise,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    output_dir = Path(args.output_dir)
    generation_args = make_args(
        "native_rollout_prefill_q_roots",
        num_roots=1 if args.fixed_tasks else int(args.source_roots),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.seed),
        frontier_parity_roots=0,
    )
    pipeline_cfg, simulator, env, explore_cfg, roots = prepare_python_roots(generation_args)
    root_runner = SelfPlayRunner(
        env=env,
        mcts=nlt.VidurMCTS(env=env, explore_cfg=explore_cfg),
        model=None,
        writer=NoopReplayWriter(),
        device_for_features=torch.device("cpu"),
        game_v2_cfg=pipeline_cfg.game_v2,
    )
    tasks: list[dict[str, Any]] = []
    signatures: set[str] = set()
    if args.fixed_tasks:
        tasks = json.loads(Path(args.fixed_tasks).read_text(encoding="utf-8"))
        selected_ids = (
            {int(value) for value in args.sample_ids.split(",") if value.strip()}
            if args.sample_ids
            else None
        )
        if selected_ids is not None:
            tasks = [
                task for task in tasks if int(task["sample_id"]) in selected_ids
            ]
        for task in tasks:
            task["iterations"] = int(args.iterations)
            task["controller_version"] = int(args.controller_version)
    for prepared in [] if args.fixed_tasks else roots:
        state, player, depth = root_runner._advance_to_branching_root(
            prepared.root_state,
            str(prepared.root_player),
            int(prepared.root_depth),
            max_hops=int(pipeline_cfg.max_forced_hops_per_root),
        )
        if player != "controller":
            continue
        state_payload = nlt._native_state_payload(env, state)
        pending_tokens, pending_requests = _pending_prefill(state_payload)
        if pending_tokens <= 0:
            continue
        actions, mask = env.sample_controller_actions(state)
        valid_indices = [
            index
            for index, action in enumerate(actions)
            if index < len(mask) and bool(mask[index]) and action is not None
        ]
        if len(valid_indices) <= 1:
            continue
        signature = json.dumps(state_payload, sort_keys=True, separators=(",", ":"))
        if signature in signatures:
            continue
        signatures.add(signature)
        sample_id = len(tasks)
        tasks.append(
            {
                "sample_id": sample_id,
                "source_root_id": int(prepared.root_id),
                "history_hops": int(prepared.history_hops),
                "root_node_id": int(prepared.root_node_id_override or prepared.root_id),
                "root_depth": int(depth),
                "sim_time": float(state.simulator._time),
                "pending_prefill_tokens": pending_tokens,
                "pending_prefill_requests": pending_requests,
                "state_payload": state_payload,
                "action_categories": {
                    str(index): _action_category(actions[index]) for index in valid_indices
                },
                "action_reprs": {
                    str(index): repr(actions[index]) for index in valid_indices
                },
                "iterations": int(args.iterations),
                "controller_version": int(args.controller_version),
                "seed": int(args.seed) + sample_id,
            }
        )
        if len(tasks) >= int(args.samples):
            break
    if not args.fixed_tasks and len(tasks) < int(args.samples):
        raise RuntimeError(
            f"found only {len(tasks)} unique pending-prefill controller roots "
            f"from {len(roots)} generated roots"
        )

    cfg_payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(cfg_payload, simulator)
    cfg_payload.update(
        {
            "discount_factor": float(args.discount_factor),
            "use_model_bootstrap": True,
            "use_policy_prior": True,
            "native_search_mode": "full_tree_rollout",
            "rollout_count": int(args.rollout_count),
            "rollout_parallel_threads": 1,
            "rollout_horizon_sec": float(args.rollout_horizon_sec),
            "rollout_policy_temperature": 1.0,
            "rollout_probability_quantum": float(args.rollout_probability_quantum),
            "rollout_max_actions": int(args.rollout_max_actions),
            "capture_rollout_trace": bool(args.capture_rollout_steps),
            "root_dirichlet_noise_enabled": False,
            "root_dirichlet_alpha": 0.0,
            "root_dirichlet_epsilon": 0.0,
            "puct_c": float(args.puct_c),
            "policy_prior_temperature": 1.0,
            "prior_min_prob": 1e-8,
            "max_forced_hops": int(pipeline_cfg.max_forced_hops_per_root),
        }
    )
    results: list[dict[str, Any]] = []
    rollout_trace_handle = None
    rollout_trace_writer = None
    if args.capture_rollout_steps:
        rollout_trace_path = output_dir / "rollout_steps.csv"
        rollout_trace_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_trace_handle = rollout_trace_path.open(
            "w", newline="", encoding="utf-8"
        )
    with ProcessPoolExecutor(
        max_workers=int(args.processes),
        initializer=_worker_init,
        initargs=(
            str(Path(args.value_model).resolve()),
            str(Path(args.controller_policy_model).resolve()),
            str(Path(args.adversary_policy_model).resolve()),
            cfg_payload,
        ),
    ) as executor:
        futures = [executor.submit(_run_root, task) for task in tasks]
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                rollout_trace_rows = result.pop("rollout_trace_rows")
                if rollout_trace_handle is not None and rollout_trace_rows:
                    if rollout_trace_writer is None:
                        rollout_trace_writer = csv.DictWriter(
                            rollout_trace_handle,
                            fieldnames=list(rollout_trace_rows[0]),
                        )
                        rollout_trace_writer.writeheader()
                    rollout_trace_writer.writerows(rollout_trace_rows)
                    rollout_trace_handle.flush()
                results.append(result)
                if completed % 20 == 0 or completed == len(futures):
                    print(f"completed {completed}/{len(futures)} roots", flush=True)
        finally:
            if rollout_trace_handle is not None:
                rollout_trace_handle.close()
    results.sort(key=lambda result: int(result["sample_id"]))

    state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for result in results:
        state_row = {
            key: value
            for key, value in result.items()
            if key not in {"best_q", "action_rows", "rollout_trace_rows"}
        }
        for category in ALL_CATEGORIES:
            state_row[f"best_q_{category}"] = result["best_q"].get(category)
        state_rows.append(state_row)
        action_rows.extend(result["action_rows"])
    _write_csv(output_dir / "state_results.csv", list(state_rows[0]), state_rows)
    _write_csv(output_dir / "action_results.csv", list(action_rows[0]), action_rows)
    comparison_rows: list[dict[str, Any]] = []
    for result in results:
        decode_rows = [
            row for row in result["action_rows"] if row["category"] == "decode_only"
        ]
        prefill_rows = [
            row
            for row in result["action_rows"]
            if row["category"].startswith("prefill_")
        ]
        if not decode_rows or not prefill_rows:
            continue
        decode = max(decode_rows, key=lambda row: float(row["q_value"]))
        prefill = max(prefill_rows, key=lambda row: float(row["q_value"]))
        comparison_rows.append(
            {
                "sample_id": result["sample_id"],
                "source_root_id": result["source_root_id"],
                "sim_time": result["sim_time"],
                "pending_prefill_tokens": result["pending_prefill_tokens"],
                "pending_prefill_requests": result["pending_prefill_requests"],
                "decode_action_index": decode["action_index"],
                "best_prefill_action_index": prefill["action_index"],
                "best_prefill_category": prefill["category"],
                "decode_mean_discounted_rollout_cost": decode[
                    "mean_discounted_rollout_cost"
                ],
                "prefill_mean_discounted_rollout_cost": prefill[
                    "mean_discounted_rollout_cost"
                ],
                "rollout_cost_delta_decode_minus_prefill": (
                    decode["mean_discounted_rollout_cost"]
                    - prefill["mean_discounted_rollout_cost"]
                ),
                "decode_mean_discounted_bootstrap_return": decode[
                    "mean_discounted_bootstrap_return"
                ],
                "prefill_mean_discounted_bootstrap_return": prefill[
                    "mean_discounted_bootstrap_return"
                ],
                "bootstrap_delta_decode_minus_prefill": (
                    decode["mean_discounted_bootstrap_return"]
                    - prefill["mean_discounted_bootstrap_return"]
                ),
                "decode_q": decode["q_value"],
                "best_prefill_q": prefill["q_value"],
                "q_delta_decode_minus_prefill": (
                    decode["q_value"] - prefill["q_value"]
                ),
                "decode_root_edge_reward": decode["reward"],
                "prefill_root_edge_reward": prefill["reward"],
                "decode_prior": decode["prior"],
                "prefill_prior": prefill["prior"],
                "decode_visits": decode["visits"],
                "prefill_visits": prefill["visits"],
                "decode_q_decomposition_error": decode["q_decomposition_error"],
                "prefill_q_decomposition_error": prefill[
                    "q_decomposition_error"
                ],
                "decode_action_repr": decode["action_repr"],
                "prefill_action_repr": prefill["action_repr"],
            }
        )
    if comparison_rows:
        _write_csv(
            output_dir / "decode_vs_best_prefill_rollout_costs.csv",
            list(comparison_rows[0]),
            comparison_rows,
        )
    summary = _summary(results, args)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-model", type=Path, required=True)
    parser.add_argument("--controller-policy-model", type=Path, required=True)
    parser.add_argument("--adversary-policy-model", type=Path, required=True)
    parser.add_argument("--controller-version", type=int, required=True)
    parser.add_argument("--adversary-version", type=int, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--source-roots", type=int, default=1200)
    parser.add_argument("--fixed-tasks", type=Path)
    parser.add_argument("--sample-ids", type=str, default="")
    parser.add_argument("--history-hops-min", type=int, default=1)
    parser.add_argument("--history-hops-max", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rollout-count", type=int, default=10)
    parser.add_argument("--rollout-horizon-sec", type=float, default=1.0)
    parser.add_argument("--rollout-probability-quantum", type=float, default=1e-6)
    parser.add_argument("--rollout-max-actions", type=int, default=4096)
    parser.add_argument("--capture-rollout-steps", action="store_true")
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--puct-c", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--processes", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
