#!/usr/bin/env python3
"""Evaluate the value-model bootstrap on immediate controller child states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from vidur.Game_Version3.DNN.native_selfplay import (
    _cfg_payload,
    attach_execution_predictor_payload,
)
from vidur.tests.native_allignment_tests.common import (
    import_native_cpp,
    make_args,
    prepare_python_roots,
)


ACTION_CATEGORIES = (
    "decode_only",
    "prefill_128",
    "prefill_256",
    "prefill_512",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-tasks", type=Path, required=True)
    parser.add_argument("--sample-ids", type=str, required=True)
    parser.add_argument("--value-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def _pending_prefill(payload: dict[str, Any]) -> tuple[int, int]:
    remaining = [
        max(
            0,
            int(request.get("num_prefill_tokens", 0))
            - int(request.get("num_processed_prefill_tokens", 0)),
        )
        for request in list(payload.get("requests") or [])
        if not bool(request.get("completed", False))
        and not bool(request.get("dropped", False))
        and not bool(request.get("feature_only", False))
    ]
    positive = [value for value in remaining if value > 0]
    return sum(positive), len(positive)


def _select_standard_action(task: dict[str, Any], category: str) -> int | None:
    categories = {
        int(index): str(value)
        for index, value in dict(task["action_categories"]).items()
    }
    representations = {
        int(index): str(value)
        for index, value in dict(task["action_reprs"]).items()
    }
    candidates = [
        index for index, value in categories.items() if value == category
    ]
    if not candidates:
        return None

    standard = [
        index
        for index in candidates
        if "strategy='GV2|evict_none'" in representations.get(index, "")
        and (
            "heuristic=None" in representations.get(index, "")
            if category == "decode_only"
            else "heuristic='SJF'" in representations.get(index, "")
        )
    ]
    if standard:
        return min(standard)
    no_eviction = [
        index
        for index in candidates
        if "strategy='GV2|evict_none'" in representations.get(index, "")
    ]
    return min(no_eviction or candidates)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    selected_ids = {
        int(value) for value in args.sample_ids.split(",") if value.strip()
    }
    tasks = [
        task
        for task in json.loads(args.fixed_tasks.read_text(encoding="utf-8"))
        if int(task["sample_id"]) in selected_ids
    ]
    tasks.sort(key=lambda task: int(task["sample_id"]))
    found_ids = {int(task["sample_id"]) for task in tasks}
    if found_ids != selected_ids:
        raise AssertionError(
            f"missing fixed tasks: {sorted(selected_ids - found_ids)}"
        )

    generation_args = make_args(
        "native_child_value_bootstrap_analysis",
        num_roots=1,
        history_hops_min=0,
        history_hops_max=0,
        history_seed=int(args.seed),
        frontier_parity_roots=0,
    )
    pipeline_cfg, simulator, _env, _explore_cfg, _roots = prepare_python_roots(
        generation_args
    )
    cfg_payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(cfg_payload, simulator)

    native = import_native_cpp(build_if_missing=False)
    value_runtime = native.NewFeatures226HGBRuntime()
    value_runtime.load_model_export(str(args.value_model))

    compact_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    values_by_category: dict[str, list[float]] = {
        category: [] for category in ACTION_CATEGORIES
    }
    deltas_by_category: dict[str, list[float]] = {
        category: [] for category in ACTION_CATEGORIES[1:]
    }

    for task in tasks:
        sample_id = int(task["sample_id"])
        root_id = int(task["source_root_id"])
        state_payload = dict(task["state_payload"])
        pending_tokens, pending_requests = _pending_prefill(state_payload)
        if pending_tokens != int(task["pending_prefill_tokens"]):
            raise AssertionError(f"pending-token mismatch for sample {sample_id}")
        compact: dict[str, Any] = {
            "sample_id": sample_id,
            "root_id": root_id,
            "root_sim_time": float(task["sim_time"]),
            "pending_prefill_tokens": pending_tokens,
            "pending_prefill_requests": pending_requests,
        }
        per_state_values: dict[str, float] = {}

        for category in ACTION_CATEGORIES:
            action_index = _select_standard_action(task, category)
            compact[category] = ""
            if action_index is None:
                detail_rows.append(
                    {
                        "sample_id": sample_id,
                        "root_id": root_id,
                        "root_sim_time": float(task["sim_time"]),
                        "pending_prefill_tokens_before": pending_tokens,
                        "pending_prefill_requests_before": pending_requests,
                        "action_category": category,
                        "action_available": False,
                        "action_index": "",
                        "action_repr": "",
                        "child_sim_time": "",
                        "pending_prefill_tokens_after": "",
                        "pending_prefill_requests_after": "",
                        "model_raw_value": "",
                        "model_bootstrap_value": "",
                    }
                )
                continue

            result = native.debug_apply_root_action_hgb226(
                value_runtime,
                state_payload,
                cfg_payload,
                "controller",
                action_index,
                root_id,
                True,
            )
            child_payload = dict(result["state"])
            child_pending_tokens, child_pending_requests = _pending_prefill(
                child_payload
            )
            raw_value = float(result["raw_value"])
            bootstrap = float(result["value"])
            if not math.isfinite(raw_value) or not math.isfinite(bootstrap):
                raise AssertionError(
                    f"non-finite value for sample {sample_id}, {category}"
                )
            compact[category] = bootstrap
            per_state_values[category] = bootstrap
            values_by_category[category].append(bootstrap)
            detail_rows.append(
                {
                    "sample_id": sample_id,
                    "root_id": root_id,
                    "root_sim_time": float(task["sim_time"]),
                    "pending_prefill_tokens_before": pending_tokens,
                    "pending_prefill_requests_before": pending_requests,
                    "action_category": category,
                    "action_available": True,
                    "action_index": action_index,
                    "action_repr": str(
                        dict(task["action_reprs"]).get(str(action_index), "")
                    ),
                    "child_sim_time": float(child_payload["sim_time"]),
                    "pending_prefill_tokens_after": child_pending_tokens,
                    "pending_prefill_requests_after": child_pending_requests,
                    "model_raw_value": raw_value,
                    "model_bootstrap_value": bootstrap,
                }
            )

        decode_value = per_state_values.get("decode_only")
        if decode_value is not None:
            for category in ACTION_CATEGORIES[1:]:
                if category in per_state_values:
                    deltas_by_category[category].append(
                        per_state_values[category] - decode_value
                    )
        compact_rows.append(compact)

    summary: dict[str, Any] = {
        "states": len(tasks),
        "model": str(args.value_model),
        "value_direction": "higher (less negative) is better for the controller",
        "transition_semantics": (
            "native controller child with fast_forward_controller=true, matching "
            "full-tree rollout MCTS"
        ),
        "action_selection": (
            "GV2|evict_none; SJF for prefill and heuristic=None for decode"
        ),
        "categories": {},
    }
    for category in ACTION_CATEGORIES:
        values = values_by_category[category]
        category_summary: dict[str, Any] = {
            "available_states": len(values),
            "mean_bootstrap": statistics.fmean(values) if values else None,
            "median_bootstrap": statistics.median(values) if values else None,
        }
        if category != "decode_only":
            deltas = deltas_by_category[category]
            category_summary.update(
                comparable_states=len(deltas),
                better_than_decode=sum(delta > 1e-12 for delta in deltas),
                worse_than_decode=sum(delta < -1e-12 for delta in deltas),
                ties_with_decode=sum(abs(delta) <= 1e-12 for delta in deltas),
                mean_bootstrap_delta_vs_decode=(
                    statistics.fmean(deltas) if deltas else None
                ),
                median_bootstrap_delta_vs_decode=(
                    statistics.median(deltas) if deltas else None
                ),
            )
        summary["categories"][category] = category_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = args.output_dir / "child_value_bootstraps_32_states.csv"
    detail_path = args.output_dir / "child_value_bootstraps_32_states_detailed.csv"
    summary_path = args.output_dir / "child_value_bootstraps_32_states_summary.json"
    _write_csv(compact_path, compact_rows)
    _write_csv(detail_path, detail_rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
