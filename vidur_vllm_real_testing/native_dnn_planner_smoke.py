"""Run one promoted-DNN/native-MCTS decision through the persistent adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .gv3_adapter import GV3PersistentAdapter
from .native_dnn_mcts_planner import (
    NativeDNNMCTSPlannerConfig,
    PromotedNativeDNNMCTSPlanner,
)
from .scheduler_contract import LiveRequestSnapshot, LiveStateSnapshot, RequestPhase


def _request(request_id: str, tokens: int, arrival: float) -> LiveRequestSnapshot:
    return LiveRequestSnapshot(
        request_id=request_id,
        phase=RequestPhase.PREFILL,
        arrival_time_s=arrival,
        actual_prefill_tokens=tokens,
        actual_prefill_remaining=tokens,
        canonical_prefill_tokens=tokens,
        canonical_prefill_remaining=tokens,
        actual_decode_tokens=216,
        actual_decode_remaining=216,
        canonical_decode_tokens=216,
        canonical_decode_remaining=216,
        actual_prefill_slo_s=1.0,
        canonical_prefill_slo_s=1.0,
        actual_decode_slo_s=0.05,
        canonical_decode_slo_s=0.05,
        num_computed_tokens=0,
        num_output_tokens=0,
        queue_name="waiting",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--rollout-count", type=int, default=1)
    parser.add_argument("--rollout-horizon-s", type=float, default=0.2)
    parser.add_argument("--puct-c", type=float, default=0.5)
    args = parser.parse_args()

    config = NativeDNNMCTSPlannerConfig(
        model_bundle=args.model_bundle,
        iterations=args.iterations,
        discount_factor=0.98,
        puct_c=args.puct_c,
        search_mode=("full_tree_rollout" if args.rollout_count else "full_tree"),
        rollout_count=args.rollout_count,
        rollout_horizon_s=args.rollout_horizon_s,
        native_threads=1,
    )
    started = time.perf_counter()
    planner = PromotedNativeDNNMCTSPlanner(config)
    load_seconds = time.perf_counter() - started
    adapter = GV3PersistentAdapter(controller=planner.plan_state)
    snapshot = LiveStateSnapshot.build(
        (
            _request("req-0", 128, 0.0),
            _request("req-1", 256, 0.0),
            _request("req-2", 512, 0.0),
        ),
        max_num_scheduled_tokens=4608,
        captured_monotonic_s=1.0,
    )
    started = time.perf_counter()
    plan = adapter.plan(snapshot)
    search_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "controller_model_version": planner.controller_model_version,
                "adversary_model_version": planner.adversary_model_version,
                "value_model_tag": str(planner.value_runtime.model_tag),
                "controller_policy_model_tag": str(
                    planner.controller_prior_runtime.model_tag
                ),
                "adversary_policy_model_tag": str(
                    planner.adversary_prior_runtime.model_tag
                ),
                "load_seconds": load_seconds,
                "search_seconds": search_seconds,
                "policy": plan.policy,
                "allocations": [
                    {
                        "request_id": item.request_id,
                        "phase": item.phase.value,
                        "num_tokens": item.num_tokens,
                    }
                    for item in plan.allocations
                ],
                "details": dict(plan.details or {}),
                "native_selection": planner.last_search,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

