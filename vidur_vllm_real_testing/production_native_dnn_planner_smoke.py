"""Smoke the lean environment-loaded production DNN planner."""

from __future__ import annotations

import json
import time

from .gv3_adapter import GV3PersistentAdapter
from .native_dnn_mcts_state_planner import ProductionNativeDNNMCTSPlanner
from .native_dnn_planner_smoke import _request
from .scheduler_contract import LiveStateSnapshot


def main() -> None:
    started = time.perf_counter()
    planner = ProductionNativeDNNMCTSPlanner.from_environment()
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
                "status": "passed",
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
                "allocations": [
                    {
                        "request_id": item.request_id,
                        "phase": item.phase.value,
                        "num_tokens": item.num_tokens,
                    }
                    for item in plan.allocations
                ],
                "native_selection": planner.last_search,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

