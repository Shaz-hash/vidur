from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import time 

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

from .launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions
from .environment import VidurMCTSEnvironment
from .mcts import VidurMCTS

import csv
import math


def _parse_sequence(values: Optional[Sequence[float]], fallback: Sequence[float]) -> Sequence[float]:
    if values is None or len(values) == 0:
        return fallback
    return tuple(values)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Vidur MCTS loop on top of a simulator configuration.",
        add_help=True,
    )
    parser.add_argument("--mcts_iterations", type=int, default=50)
    parser.add_argument(
        "--mcts_log_csv",
        type=str,
        default="simulator_output/mcts_trace.csv",
        help="Destination CSV capturing the explored actions/states.",
    )
    parser.add_argument("--mcts_maximum_qps", type=int, default=12)
    parser.add_argument("--mcts_interval_request_size", type=int, default=512)
    parser.add_argument("--mcts_min_request_tokens", type=int, default=512)
    parser.add_argument("--mcts_max_request_tokens", type=int, default=3072)
    parser.add_argument("--mcts_prefill_profile", type=str, default=None)
    parser.add_argument("--mcts_prefill_slowdown", type=float, default=1.0)
    parser.add_argument("--mcts_simulation_depth", type=int, default=4)
    parser.add_argument("--mcts_simulation_random_tries", type=int, default=1)
    parser.add_argument("--mcts_exploration_constant", type=float, default=1.4)
    parser.add_argument("--mcts_max_branching", type=int, default=30)
    parser.add_argument("--mcts_controller_budget_combs", type=int, default=15)
    parser.add_argument(
        "--mcts_prefill_slos",
        type=float,
        nargs="+",
        default=None,
        help="Override adversary SLO catalogue for prefill stage.",
    )
    parser.add_argument(
        "--mcts_decode_slos",
        type=float,
        nargs="+",
        default=None,
        help="Override adversary SLO catalogue for decode stage.",
    )
    parser.add_argument(
        "--mcts_tree_csv",
        type=str,
        default=None,
        help="Optional CSV capturing tree nodes with controller actions.",
    )
    parser.add_argument(
        "--mcts_tree_dump_interval",
        type=int,
        default=100,
        help="Dump full MCTS tree to tree CSV every N iterations (0 = never).",
    )
    parser.add_argument(
        "--mcts_run_id",
        type=str,
        default="",
        help="Optional run ID suffix (e.g. P1, P2) for per-process CSVs.",
    )
    parser.add_argument(
        "--mcts_history_depth",
        type=int,
        default=0,
        help="Number of random tree steps (adversary/controller plies) before MCTS search.",
    )



    return parser


def configure_simulation(sim_args: Iterable[str]) -> SimulationConfig:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(sim_args)
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv
    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False
    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args, remaining = parser.parse_known_args(argv) ## Divides all the arguments into MCTS related config and Simulation related Config

    sim_cfg = configure_simulation(remaining)
    et_cfg = sim_cfg.execution_time_predictor_config
    et_cfg.prediction_max_tokens_per_request = 8192     
    et_cfg.prediction_max_batch_size = 64               
    simulator = Simulator(sim_cfg, register_atexit=False)

    # t0 = time.perf_counter()
    # sim1 = Simulator(sim_cfg, register_atexit=False,
    #              execution_time_predictor=simulator._execution_time_predictor)
    # t1 = time.perf_counter()
    
    # print(
    #         f"[PROFILE] SIMULATOR COPY RESULTS RESULTS : "
    #         f"={t1 - t0:.4f}s"
    #     )

    # Suffix for this run (empty if not provided)
    suffix = f"_{args.mcts_run_id}" if args.mcts_run_id else ""

    base_log = Path(args.mcts_log_csv)
    log_path = base_log.with_name(base_log.stem + suffix + base_log.suffix)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mcts_tree_csv:
        base_tree = Path(args.mcts_tree_csv)
    else:
        base_tree = base_log.with_name(base_log.stem + "_tree" + base_log.suffix)
    tree_path = base_tree.with_name(base_tree.stem + suffix + base_tree.suffix)



    default_slos = RequestSLOOptions()
    slo_options = RequestSLOOptions(
        prefill_slos=_parse_sequence(args.mcts_prefill_slos, default_slos.prefill_slos),
        decode_slos=_parse_sequence(args.mcts_decode_slos, default_slos.decode_slos),
    )

    constraints = MCTSConstraintConfig(
        maximum_qps=args.mcts_maximum_qps,
        min_request_tokens=args.mcts_min_request_tokens,
        max_request_tokens=args.mcts_max_request_tokens,
        interval_request_size=args.mcts_interval_request_size,
        request_slo_options=slo_options,
        prefill_slowdown=args.mcts_prefill_slowdown,
        prefill_profile_path=args.mcts_prefill_profile,
    )

    explore_cfg = MCTSExploreConfig(
        simulation_depth=args.mcts_simulation_depth,
        simulation_random_tries=args.mcts_simulation_random_tries,
        exploration_constant=args.mcts_exploration_constant,
        max_branching=args.mcts_max_branching,
        controller_budget_combs=args.mcts_controller_budget_combs,
    )

    # log_path = Path(args.mcts_log_csv)
    # log_path.parent.mkdir(parents=True, exist_ok=True)

    # if args.mcts_tree_csv:
    #     tree_path = Path(args.mcts_tree_csv)
    # else:
    #     tree_path = log_path.with_name(log_path.stem + "_tree.csv")

    env = VidurMCTSEnvironment(
        base_simulator=simulator,
        constraints=constraints,
        explore_cfg=explore_cfg,
    )
    mcts = VidurMCTS(env, explore_cfg, log_path=log_path, tree_log_path=tree_path , tree_dump_interval=args.mcts_tree_dump_interval, history_depth=args.mcts_history_depth,)
    best_action = mcts.search(args.mcts_iterations)

    # root_visits = mcts._root.visits  # or via a getter if you add one
    # state_csv_path = log_path.with_name(log_path.stem + "_controller_states.csv")
    # write_controller_state_summary(env, explore_cfg, root_visits, state_csv_path)

    print("Best controller action discovered:")
    print(f"  token_budget={best_action.token_budget}")
    print(f"  selected_request_ids={best_action.selected_request_ids}")
    print(f"  token_allocations={best_action.token_allocations}")
    print(f"Log written to: {log_path.resolve()}")


def write_controller_state_summary(env, explore_cfg, root_visits: int, path: Path) -> None:
    data = env.all_possible_Controller_States
    if not data:
        return

    fields = [
        "mapping",
        "heuristic",
        "strategy",
        "sample_visits",
        "mcts_visits",
        "mean_cost",
        "cumulative_cost",
        "last_slo_violations",
        "last_avg_lateness",
        "ucb_score",
    ]

    c = explore_cfg.exploration_constant

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (mapping, heuristic, strategy), stats in data.items():
            n = max(1, stats.get("mcts_visits", 0))
            mean_cost = stats.get("mean_cost", 0.0)
            exploit = -mean_cost
            explore_term = c * math.sqrt(math.log(max(1, root_visits)) / n)
            ucb = exploit + explore_term

            w.writerow({
                "mapping": list(mapping),
                "heuristic": heuristic,
                "strategy": strategy,
                "sample_visits": stats.get("sample_visits", 0),
                "mcts_visits": stats.get("mcts_visits", 0),
                "mean_cost": mean_cost,
                "cumulative_cost": stats.get("cumulative_cost", 0.0),
                "last_slo_violations": stats.get("last_slo_violations", 0),
                "last_avg_lateness": stats.get("last_avg_lateness", 0.0),
                "ucb_score": ucb,
            })



if __name__ == "__main__":
    main()
