"""Run uniform GV4 MCTS, split exact iteration paths, and validate all logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Sequence

from GV4_Engine.mcts_value_prior import MCTSConfig, VidurMCTS
from ..logger import GV4NodeCSVLogger
from ..state import Player
from ..virtual_environment import GV4VirtualVidurMCTSEnvironment
from .config import GV4MCTSTestConfig, build_engine_config
from .kv_cache_tests import run_kv_cache_tests
from .pipeline_tests import run_pipeline_tests
from .request_tests import run_request_tests
from .state_tests import run_state_tests
from .timing import build_timing_provider
from .trace_capture import IterationPathObserver, split_iteration_logs
from .trace_context import load_iteration_traces
from .validation import ValidationContext, ValidationReport


def _mcts_config(
    test: GV4MCTSTestConfig,
    observer: IterationPathObserver,
) -> MCTSConfig:
    config = MCTSConfig()
    config.mcts_iterations = test.mcts_iterations
    config.rng = random.Random(test.seed)
    config.use_policy_prior = False
    config.controller_prior_model = None
    config.adversary_prior_model = None
    config.root_dirichlet_alpha = 0.0
    config.root_dirichlet_epsilon = 0.0
    config.root_dirichlet_total_concentration = 0.0
    config.log_flag = False
    config.iteration_observer = observer
    return config


def _capture(
    test: GV4MCTSTestConfig,
    environment: GV4VirtualVidurMCTSEnvironment,
) -> None:
    with GV4NodeCSVLogger(
        test.raw_log_dir,
        environment.config,
        flush_every=1,
        overwrite=test.overwrite,
        validate_states=True,
    ) as logger:
        observer = IterationPathObserver(logger)
        mcts = VidurMCTS(environment, _mcts_config(test, observer))
        try:
            state = environment.initial_state(
                now=0.0,
                next_player=Player.ADVERSARY,
            )
            mcts.search_dnn(
                None,
                state,
                "adversary",
                game_id=test.game_id,
                root_id=test.root_id,
                root_node_id_override=test.root_node_id,
                root_depth=0,
                mcts_iter=test.mcts_iterations,
                model_version=0,
                use_model_bootstrap=False,
            )
            if observer.iterations_logged != test.mcts_iterations:
                raise RuntimeError(
                    f"logged {observer.iterations_logged} paths, expected "
                    f"{test.mcts_iterations}"
                )
        finally:
            mcts.close()


def _validate(
    test: GV4MCTSTestConfig,
    context: ValidationContext,
) -> ValidationReport:
    traces = load_iteration_traces(
        test.iteration_trace_dir,
        expected_iterations=test.mcts_iterations,
    )
    report = ValidationReport()
    run_state_tests(traces, context, report)
    run_request_tests(traces, context, report)
    run_kv_cache_tests(traces, context, report)
    run_pipeline_tests(traces, context, report)

    for event in (
        "adversary_actions",
        "launched_requests",
        "controller_actions",
        "controller_batches",
        "vidur_timed_batches",
    ):
        report.check(
            "coverage",
            f"uniform search exercised {event}",
            report.coverage[event] > 0,
            detail=f"observed={report.coverage[event]}",
        )
    return report


def run(test: GV4MCTSTestConfig, *, validate_only: bool = False) -> ValidationReport:
    """Execute the complete capture/split/check pipeline or recheck existing logs."""

    engine = build_engine_config(test)
    timing = build_timing_provider(test, engine)
    environment = GV4VirtualVidurMCTSEnvironment(
        engine,
        batch_timing_provider=timing,
        prefill_time_estimator=timing.estimate_prefill_time,
    )

    test.output_dir.mkdir(parents=True, exist_ok=True)
    if not validate_only:
        _capture(test, environment)
        split_iteration_logs(
            test.raw_log_dir,
            test.iteration_trace_dir,
            expected_iterations=test.mcts_iterations,
            overwrite=test.overwrite,
        )

    report = _validate(test, ValidationContext(engine, timing))
    summary_path = test.output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    report.assert_clean()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--cache-mode",
        choices=("ignore_cache", "use_cache", "require_cache"),
        default="use_cache",
    )
    parser.add_argument(
        "--timing-mode",
        choices=("vidur", "deterministic"),
        default="vidur",
        help="deterministic is only for a fast harness smoke test",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    defaults = GV4MCTSTestConfig()
    test = GV4MCTSTestConfig(
        output_dir=defaults.output_dir if args.output_dir is None else args.output_dir,
        predictor_cache_dir=(
            defaults.predictor_cache_dir if args.cache_dir is None else args.cache_dir
        ),
        predictor_cache_mode=args.cache_mode,
        timing_mode=args.timing_mode,
        mcts_iterations=args.iterations,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    report = run(test, validate_only=args.validate_only)
    print(
        f"GV4 MCTS trace validation passed: {report.total_checks} checks, "
        f"coverage={dict(sorted(report.coverage.items()))}"
    )
    print(f"outputs: {test.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
