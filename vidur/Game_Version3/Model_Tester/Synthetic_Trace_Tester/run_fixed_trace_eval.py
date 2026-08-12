from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .fixed_trace_eval_runner import FixedTraceEvalConfig, run_fixed_trace_eval


def _default_model_root() -> Path:
    repo = Path(__file__).resolve()
    for parent in repo.parents:
        candidate = parent / "simulator_output" / "GV3_Agent" / "AlphaGoZero" / "models" / "Model_Version139"
        if candidate.exists():
            return candidate
    return Path(
        "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
        "simulator_output/GV3_Agent/AlphaGoZero/models/Model_Version139"
    )


def parse_args() -> argparse.Namespace:
    model_root = _default_model_root()
    parser = argparse.ArgumentParser(
        description="Run a fixed external trace through GV3 using either SJF-512 or native C++ value+prior MCTS."
    )
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=("sjf512", "model_cpp"), required=True)
    parser.add_argument("--time-limit-sec", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--max-trace-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--token-policy", choices=("clip", "bucket_gv3", "raw"), default="clip")
    parser.add_argument("--rebase-to-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--record-launch-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record external bursts in the GV3 launch history consumed by markov_v2.",
    )
    parser.add_argument(
        "--strict-gv3-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require every external burst to be a legal native GV3 adversary launch.",
    )

    parser.add_argument("--sjf-budget-tokens", type=int, default=512)
    parser.add_argument("--sjf-heuristic", default="SJF")
    parser.add_argument("--sjf-eviction-rule", default="evict_none")

    parser.add_argument(
        "--value-model-path",
        default=str(model_root / "value" / "hgb_sq_63leaf_1050iter_a2" / "model.joblib"),
    )
    parser.add_argument(
        "--controller-prior-model-path",
        default=str(model_root / "controller_prior" / "hgb_policy_63leaf_1050iter" / "model.joblib"),
    )
    parser.add_argument(
        "--adversary-prior-model-path",
        default=str(model_root / "adversary_prior" / "hgb_policy_63leaf_1050iter" / "model.joblib"),
    )
    parser.add_argument("--model-version", type=int, default=139)
    parser.add_argument("--mcts-iterations", type=int, default=1000)
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--uct-c", type=float, default=1.0)
    parser.add_argument("--policy-prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-min-prob", type=float, default=1e-8)
    parser.add_argument(
        "--native-search-mode",
        choices=("full_tree", "full_tree_rollout"),
        default="full_tree",
    )
    parser.add_argument("--rollout-count", type=int, default=10)
    parser.add_argument("--rollout-parallel-threads", type=int, default=1)
    parser.add_argument("--rollout-horizon-sec", type=float, default=1.0)
    parser.add_argument("--rollout-policy-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-probability-quantum", type=float, default=1e-6)
    parser.add_argument("--rollout-max-actions", type=int, default=4096)
    parser.add_argument("--disable-model-bootstrap", action="store_true")
    parser.add_argument("--force-build-native", action="store_true")
    parser.add_argument("--worker-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    cfg = FixedTraceEvalConfig(
        trace_csv=Path(args.trace_csv).expanduser(),
        output_dir=output_dir,
        policy=str(args.policy),
        time_limit_sec=float(args.time_limit_sec),
        max_steps=int(args.max_steps),
        max_trace_rows=int(args.max_trace_rows),
        seed=int(args.seed),
        token_policy=str(args.token_policy),
        rebase_to_zero=bool(args.rebase_to_zero),
        record_launch_history=bool(args.record_launch_history),
        strict_gv3_trace=bool(args.strict_gv3_trace),
        sjf_budget_tokens=int(args.sjf_budget_tokens),
        sjf_heuristic=str(args.sjf_heuristic),
        sjf_eviction_rule=str(args.sjf_eviction_rule),
        value_model_path=Path(args.value_model_path).expanduser(),
        controller_prior_model_path=Path(args.controller_prior_model_path).expanduser(),
        adversary_prior_model_path=Path(args.adversary_prior_model_path).expanduser(),
        model_version=int(args.model_version),
        mcts_iterations=int(args.mcts_iterations),
        discount_factor=float(args.discount_factor),
        puct_c=float(args.puct_c),
        uct_c=float(args.uct_c),
        policy_prior_temperature=float(args.policy_prior_temperature),
        prior_min_prob=float(args.prior_min_prob),
        native_search_mode=str(args.native_search_mode),
        rollout_count=int(args.rollout_count),
        rollout_parallel_threads=int(args.rollout_parallel_threads),
        rollout_horizon_sec=float(args.rollout_horizon_sec),
        rollout_policy_temperature=float(args.rollout_policy_temperature),
        rollout_probability_quantum=float(args.rollout_probability_quantum),
        rollout_max_actions=int(args.rollout_max_actions),
        disable_model_bootstrap=bool(args.disable_model_bootstrap),
        force_build_native=bool(args.force_build_native),
        worker_threads=int(args.worker_threads),
    )
    result = run_fixed_trace_eval(cfg)
    print(f"fixed trace {result.policy} steps: {result.steps_csv}")
    print(f"fixed trace {result.policy} summary: {result.summary_csv}")
    print(f"fixed trace {result.policy} validation: {result.validation_csv}")
    print(
        "summary: "
        f"cost={result.total_cost:.6f} violations={result.slo_violations} "
        f"lateness={result.total_lateness:.6f} generated={result.requests_generated} "
        f"completed={result.requests_completed} final_t={result.final_sim_time:.6f} "
        f"prefill_steps={result.prefill_policy_steps} decode_steps={result.decode_drain_steps} "
        f"end={result.end_reason}"
    )

    compare_path = output_dir.parent / "fixed_trace_comparison_partial.csv"
    summaries = sorted(output_dir.parent.glob("*/summary.csv"))
    if summaries:
        rows: list[dict[str, str]] = []
        for path in summaries:
            with path.open("r", newline="", encoding="utf-8") as f:
                data = list(csv.DictReader(f))
            if data:
                rows.append(data[0])
        fields = sorted({key for row in rows for key in row.keys()})
        with compare_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"comparison summary: {compare_path}")


if __name__ == "__main__":
    main()
