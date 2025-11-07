from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

from .config import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions
from .environment import VidurMCTSEnvironment
from .mcts import VidurMCTS


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
    parser.add_argument("--mcts_max_request_tokens", type=int, default=None)
    parser.add_argument("--mcts_prefill_profile", type=str, default=None)
    parser.add_argument("--mcts_prefill_slowdown", type=float, default=1.0)
    parser.add_argument("--mcts_simulation_depth", type=int, default=4)
    parser.add_argument("--mcts_simulation_random_tries", type=int, default=1)
    parser.add_argument("--mcts_exploration_constant", type=float, default=4.0)
    parser.add_argument("--mcts_max_branching", type=int, default=25)
    parser.add_argument("--mcts_controller_budget_combs", type=int, default=10)
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
    args, remaining = parser.parse_known_args(argv)

    sim_cfg = configure_simulation(remaining)
    simulator = Simulator(sim_cfg, register_atexit=False)

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

    log_path = Path(args.mcts_log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = VidurMCTSEnvironment(
        base_simulator=simulator,
        constraints=constraints,
        explore_cfg=explore_cfg,
    )
    mcts = VidurMCTS(env, explore_cfg, log_path=log_path)
    best_action = mcts.search(args.mcts_iterations)

    print("Best controller action discovered:")
    print(f"  token_budget={best_action.token_budget}")
    print(f"  selected_request_ids={best_action.selected_request_ids}")
    print(f"  token_allocations={best_action.token_allocations}")
    print(f"Log written to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
