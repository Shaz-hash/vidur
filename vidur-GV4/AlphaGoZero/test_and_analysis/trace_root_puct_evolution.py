"""Trace root PUCT terms for the final two most-visited controller actions.

The native trace is opt-in and records the root action table before search
(iteration 0) and after every simulation backup. It does not change selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _pending_prefill(payload: dict[str, Any]) -> tuple[int, int]:
    remaining = []
    for request in payload.get("requests") or []:
        if (
            bool(request.get("completed", False))
            or bool(request.get("dropped", False))
            or bool(request.get("feature_only", False))
        ):
            continue
        tokens = max(
            0,
            int(request.get("num_prefill_tokens", 0))
            - int(request.get("num_processed_prefill_tokens", 0)),
        )
        if tokens > 0:
            remaining.append(tokens)
    return sum(remaining), len(remaining)


def _python_action_label(action: Any) -> str:
    prefill = sum(
        int(value)
        for value in (getattr(action, "prefill_allocations", {}) or {}).values()
    )
    decode = sum(
        int(value)
        for value in (getattr(action, "decode_allocations", {}) or {}).values()
    )
    if prefill == 0 and decode > 0:
        return "Decode only"
    if prefill > 0:
        return f"Prefill {prefill}"
    return "Other"


def _native_action_label(action_json: str) -> str:
    try:
        action = json.loads(action_json)
    except (TypeError, json.JSONDecodeError):
        return "Unknown"
    prefill = sum(int(value) for value in action.get("prefill_allocations", {}).values())
    decode = sum(int(value) for value in action.get("decode_allocations", {}).values())
    if prefill == 0 and decode > 0:
        return "Decode only"
    if prefill > 0:
        return f"Prefill {prefill}"
    return "Other"


def _state_description(
    task: dict[str, Any],
    canonical_action_count: int,
) -> dict[str, Any]:
    payload = task["state_payload"]
    stats = payload.get("stats") or {}
    active_requests = []
    decode_ready = 0
    for request in payload.get("requests") or []:
        if bool(request.get("completed", False)) or bool(request.get("dropped", False)):
            continue
        prefill_total = int(request.get("num_prefill_tokens", 0))
        prefill_done = int(request.get("num_processed_prefill_tokens", 0))
        remaining_prefill = max(0, prefill_total - prefill_done)
        if remaining_prefill == 0 and not bool(request.get("feature_only", False)):
            decode_ready += 1
        active_requests.append(
            {
                "request_id": int(request.get("request_id", -1)),
                "remaining_prefill_tokens": remaining_prefill,
                "processed_decode_tokens": int(
                    request.get("num_processed_decode_tokens", 0)
                ),
                "prefill_slo": request.get("prefill_slo"),
                "decode_slo": request.get("decode_slo"),
            }
        )
    return {
        "sample_id": int(task["sample_id"]),
        "source_root_id": int(task["source_root_id"]),
        "history_hops": int(task["history_hops"]),
        "sim_time": float(task["sim_time"]),
        "active_request_count": len(active_requests),
        "pending_prefill_requests": int(task["pending_prefill_requests"]),
        "pending_prefill_tokens": int(task["pending_prefill_tokens"]),
        "decode_ready_requests": decode_ready,
        "decode_credit_balance": int(stats.get("decode_credit_balance", 0)),
        "requests_generated": int(stats.get("requests_generated", 0)),
        "requests_completed": int(stats.get("requests_completed", 0)),
        "slo_violations": int(stats.get("slo_violations", 0)),
        "total_lateness": float(stats.get("slo_lateness_sum", 0.0)),
        "canonical_action_count": canonical_action_count,
        "active_requests_json": json.dumps(active_requests, sort_keys=True),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_cfg_payload(
    pipeline_cfg: Any,
    simulator: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload.update(
        {
            "discount_factor": float(args.discount_factor),
            "use_model_bootstrap": True,
            "use_policy_prior": True,
            "native_search_mode": "full_tree_rollout",
            "rollout_count": int(args.rollout_count),
            "rollout_parallel_threads": int(args.rollout_threads),
            "rollout_policy_parallel_threads": 1,
            "rollout_horizon_sec": float(args.rollout_horizon_sec),
            "rollout_policy_temperature": 1.0,
            "rollout_probability_quantum": 1e-6,
            "rollout_max_actions": 4096,
            "rollout_optimized_execution": True,
            "capture_rollout_trace": False,
            "capture_root_puct_trace": True,
            "root_dirichlet_noise_enabled": bool(args.root_noise),
            "root_dirichlet_alpha": float(args.dirichlet_alpha),
            "root_dirichlet_total_concentration": 0.0,
            "root_dirichlet_epsilon": float(args.dirichlet_epsilon),
            "puct_c": float(args.puct_c),
            "policy_prior_temperature": 1.0,
            "prior_min_prob": 1e-8,
            "max_forced_hops": int(pipeline_cfg.max_forced_hops_per_root),
            "value_feature_schema": "markov_v2",
            "policy_feature_schema": "markov_v2",
        }
    )
    return payload


def _run_search(
    native: Any,
    value_runtime: Any,
    controller_runtime: Any,
    adversary_runtime: Any,
    task: dict[str, Any],
    cfg_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return native.search_mcts_hgb226_value_prior_hgb(
        value_runtime,
        controller_runtime,
        adversary_runtime,
        int(args.controller_version),
        task["state_payload"],
        cfg_payload,
        int(args.iterations),
        "controller",
        int(task["root_node_id"]),
        int(task["root_depth"]),
        0,
        int(task["sample_id"]),
        int(args.seed) + int(task["sample_id"]),
        False,
        False,
        "",
        "",
    )


def _verify_trace(
    trace_rows: list[dict[str, Any]],
    canonical_action_count: int,
    iterations: int,
) -> None:
    expected = canonical_action_count * (iterations + 1)
    if len(trace_rows) != expected:
        raise AssertionError(
            f"expected {expected} root trace rows, observed {len(trace_rows)}"
        )
    observed_iterations = sorted({int(row["sim_iteration"]) for row in trace_rows})
    if observed_iterations != list(range(iterations + 1)):
        raise AssertionError("root PUCT trace does not cover every iteration")
    for row in trace_rows:
        expected_puct = float(row["normalized_q"]) + float(
            row["exploration_weighted"]
        )
        if not math.isclose(
            float(row["puct_score"]), expected_puct, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError("PUCT decomposition mismatch")


def _plot_metric(
    output_path: Path,
    trace_rows: list[dict[str, Any]],
    selected_actions: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    colors = ("#176B87", "#D97706")
    for rank, action in enumerate(selected_actions, start=1):
        action_index = int(action["action_index"])
        rows = [
            row
            for row in trace_rows
            if int(row["action_index"]) == action_index
        ]
        rows.sort(key=lambda row: int(row["sim_iteration"]))
        label = (
            f"{'Best' if rank == 1 else 'Second'}: {action['action_label']} "
            f"(a={action_index}, final N={action['visits']}, "
            f"prior={float(action['prior']):.4f})"
        )
        ax.plot(
            [int(row["sim_iteration"]) for row in rows],
            [float(row[metric]) for row in rows],
            color=colors[rank - 1],
            linewidth=1.8,
            label=label,
        )
    if metric == "normalized_q":
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(f"{title_prefix}\n{ylabel} across root MCTS simulations")
    ax.set_xlabel("Completed MCTS simulations")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.24)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    generation_args = make_args(
        "root_puct_evolution",
        num_roots=int(args.source_roots),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.seed),
        frontier_parity_roots=0,
    )
    pipeline_cfg, simulator, env, explore_cfg, roots = prepare_python_roots(
        generation_args
    )
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
    for prepared in roots:
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
        labels = {_python_action_label(actions[index]) for index in valid_indices}
        if "Decode only" not in labels or not any(
            label.startswith("Prefill ") for label in labels
        ):
            continue
        signature = json.dumps(state_payload, sort_keys=True, separators=(",", ":"))
        if signature in signatures:
            continue
        signatures.add(signature)
        tasks.append(
            {
                "sample_id": len(tasks),
                "source_root_id": int(prepared.root_id),
                "history_hops": int(prepared.history_hops),
                "root_node_id": int(
                    prepared.root_node_id_override or prepared.root_id
                ),
                "root_depth": int(depth),
                "sim_time": float(state.simulator._time),
                "pending_prefill_tokens": pending_tokens,
                "pending_prefill_requests": pending_requests,
                "state_payload": state_payload,
                "action_labels": {
                    str(index): _python_action_label(actions[index])
                    for index in valid_indices
                },
                "action_reprs": {
                    str(index): repr(actions[index]) for index in valid_indices
                },
            }
        )
        if len(tasks) >= int(args.samples):
            break
    if len(tasks) < int(args.samples):
        raise RuntimeError(
            f"found only {len(tasks)} suitable pending-prefill roots "
            f"from {len(roots)} candidates"
        )

    native = import_native_cpp(build_if_missing=False)
    value_runtime = native.NewFeatures226HGBRuntime()
    value_runtime.load_model_export(str(Path(args.value_model).resolve()))
    controller_runtime = native.NativeHGBModelRuntime()
    controller_runtime.load_model_export(
        str(Path(args.controller_policy_model).resolve())
    )
    adversary_runtime = native.NativeHGBModelRuntime()
    adversary_runtime.load_model_export(
        str(Path(args.adversary_policy_model).resolve())
    )
    cfg_payload = _make_cfg_payload(pipeline_cfg, simulator, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_root_tasks.json").write_text(
        json.dumps(tasks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    all_trace_rows: list[dict[str, Any]] = []
    all_final_actions: list[dict[str, Any]] = []
    all_selected_actions: list[dict[str, Any]] = []
    state_descriptions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for task in tasks:
        result = _run_search(
            native,
            value_runtime,
            controller_runtime,
            adversary_runtime,
            task,
            cfg_payload,
            args,
        )
        children = []
        for child in result.get("children", []):
            visits = int(child["visits"])
            if visits <= 0:
                continue
            action_index = int(child["index"])
            action_json = str(child.get("parent_action_json", ""))
            children.append(
                {
                    "sample_id": int(task["sample_id"]),
                    "action_index": action_index,
                    "action_label": _native_action_label(action_json),
                    "visits": visits,
                    "q_value": float(child["value_sum"]) / visits,
                    "prior": float(child["prior"]),
                    "action_json": action_json,
                }
            )
        children.sort(key=lambda row: (-int(row["visits"]), int(row["action_index"])))
        if len(children) < 2:
            raise RuntimeError("search expanded fewer than two root actions")
        selected = children[:2]
        labels_by_action: dict[int, str] = {}
        repr_by_action: dict[int, str] = {}
        for alias_raw, canonical_raw in dict(
            result.get("action_alias_to_canonical", {})
        ).items():
            alias = int(alias_raw)
            canonical = int(canonical_raw)
            labels_by_action.setdefault(
                canonical,
                str(task["action_labels"].get(str(alias), f"Action {canonical}")),
            )
            repr_by_action.setdefault(
                canonical, str(task["action_reprs"].get(str(alias), ""))
            )
        labels_by_action.update(
            {
                int(row["action_index"]): str(row["action_label"])
                for row in children
            }
        )
        repr_by_action.update(
            {
                int(row["action_index"]): str(row["action_json"])
                for row in children
            }
        )

        trace_rows = []
        for native_row in result.get("root_puct_trace_steps", []):
            action_index = int(native_row["action_index"])
            trace_rows.append(
                {
                    "sample_id": int(task["sample_id"]),
                    "sim_iteration": int(native_row["sim_iteration"]),
                    "action_index": action_index,
                    "action_label": labels_by_action.get(
                        action_index, f"Action {action_index}"
                    ),
                    "visited": bool(native_row["visited"]),
                    "visits": int(native_row["visits"]),
                    "q_value": float(native_row["q_value"]),
                    "normalized_q": float(native_row["normalized_q"]),
                    "prior": float(native_row["prior"]),
                    "exploration_raw": float(native_row["exploration_raw"]),
                    "exploration_weighted": float(
                        native_row["exploration_weighted"]
                    ),
                    "puct_score": float(native_row["puct_score"]),
                    "parent_min_value": float(native_row["parent_min_value"]),
                    "parent_max_value": float(native_row["parent_max_value"]),
                    "action_json": repr_by_action.get(action_index, ""),
                }
            )
        canonical_action_count = len(trace_rows) // (int(args.iterations) + 1)
        _verify_trace(trace_rows, canonical_action_count, int(args.iterations))

        description = _state_description(task, canonical_action_count)
        state_descriptions.append(description)
        all_trace_rows.extend(trace_rows)
        all_final_actions.extend(children)
        all_selected_actions.extend(selected)

        state_dir = output_dir / f"state_{int(task['sample_id']):03d}"
        state_dir.mkdir(parents=True, exist_ok=True)
        title_prefix = (
            f"State {task['sample_id']}: t={task['sim_time']:.3f}s, "
            f"{task['pending_prefill_requests']} pending-prefill requests, "
            f"{task['pending_prefill_tokens']} pending tokens"
        )
        plots = (
            ("q_value", "Backed-up controller Q", "q_value.png"),
            ("normalized_q", "Normalized Q / exploit", "normalized_q.png"),
            (
                "exploration_weighted",
                f"Weighted exploration (C={args.puct_c:g})",
                "exploration_component.png",
            ),
            ("puct_score", "Total PUCT score", "puct_score.png"),
        )
        for metric, ylabel, filename in plots:
            _plot_metric(
                state_dir / filename,
                trace_rows,
                selected,
                metric,
                ylabel,
                title_prefix,
            )
        summaries.append(
            {
                **{key: value for key, value in description.items() if key != "active_requests_json"},
                "best_action": selected[0],
                "second_action": selected[1],
                "plot_directory": str(state_dir.resolve()),
            }
        )

    _write_csv(output_dir / "root_action_trace_all.csv", all_trace_rows)
    _write_csv(output_dir / "final_actions.csv", all_final_actions)
    _write_csv(output_dir / "selected_actions.csv", all_selected_actions)
    _write_csv(output_dir / "state_descriptions.csv", state_descriptions)
    summary = {
        "configuration": {
            "controller_version": int(args.controller_version),
            "adversary_version": int(args.adversary_version),
            "iterations": int(args.iterations),
            "rollout_count": int(args.rollout_count),
            "rollout_horizon_sec": float(args.rollout_horizon_sec),
            "rollout_threads": int(args.rollout_threads),
            "discount_factor": float(args.discount_factor),
            "puct_c": float(args.puct_c),
            "root_noise_enabled": bool(args.root_noise),
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "dirichlet_epsilon": float(args.dirichlet_epsilon),
            "trace_timing": "iteration 0 before search; iterations 1..N after backup",
            "exploration_graph_term": "puct_c * prior * sqrt(parent_visits) / (1 + child_visits)",
        },
        "states": summaries,
    }
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--source-roots", type=int, default=400)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rollout-count", type=int, default=4)
    parser.add_argument("--rollout-horizon-sec", type=float, default=3.0)
    parser.add_argument("--rollout-threads", type=int, default=2)
    parser.add_argument("--discount-factor", type=float, default=0.98)
    parser.add_argument("--puct-c", type=float, default=0.05)
    parser.add_argument(
        "--root-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.05)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
