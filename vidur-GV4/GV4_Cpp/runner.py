"""Run native uniform GV4 MCTS, validate its traces, and check Python parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from GV4_Engine.GV4_MCTS_Test.config import (
    GV4MCTSTestConfig,
    build_engine_config,
)
from GV4_Engine.GV4_MCTS_Test.runner import _validate
from GV4_Engine.GV4_MCTS_Test.timing import build_timing_provider
from GV4_Engine.GV4_MCTS_Test.trace_capture import split_iteration_logs
from GV4_Engine.GV4_MCTS_Test.validation import ValidationContext, ValidationReport
from GV4_Engine.config import GV4EngineConfig
from GV4_Engine.logger import GV4NodeCSVLogger
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment

from . import gv4_native as native
from .native_logger import NativeIterationPathLogger
from .runtime import environment_from_python
from .uniform_parity import UniformParityReport, compare_uniform_search


def _capture(
    test: GV4MCTSTestConfig,
    config: GV4EngineConfig,
    environment: native.Environment,
) -> dict:
    with GV4NodeCSVLogger(
        test.raw_log_dir,
        config,
        flush_every=1,
        overwrite=test.overwrite,
        validate_states=True,
    ) as logger:
        observer = NativeIterationPathLogger(
            logger,
            game_id=test.game_id,
            root_id=test.root_id,
        )
        result = native.run_uniform_mcts(
            environment,
            environment.initial_state(0.0, native.Player.ADVERSARY),
            native.Player.ADVERSARY,
            test.mcts_iterations,
            1.0,
            test.root_node_id,
            0,
            observer,
        )
        if observer.iterations_logged != test.mcts_iterations:
            raise RuntimeError(
                f"logged {observer.iterations_logged} native paths, expected "
                f"{test.mcts_iterations}"
            )
        return dict(result)


def run(
    test: GV4MCTSTestConfig,
    *,
    validate_only: bool = False,
    check_python_parity: bool = True,
) -> tuple[ValidationReport, UniformParityReport | None]:
    """Execute native capture/split/validation and optional exact visit parity."""

    config = build_engine_config(test)
    timing = build_timing_provider(test, config)
    python_environment = GV4VirtualVidurMCTSEnvironment(
        config,
        batch_timing_provider=timing,
        prefill_time_estimator=timing.estimate_prefill_time,
    )
    native_environment = environment_from_python(config, timing)

    test.output_dir.mkdir(parents=True, exist_ok=True)
    native_result = None
    if not validate_only:
        native_result = _capture(test, config, native_environment)
        split_iteration_logs(
            test.raw_log_dir,
            test.iteration_trace_dir,
            expected_iterations=test.mcts_iterations,
            overwrite=test.overwrite,
        )

    report = _validate(test, ValidationContext(config, timing))
    with (test.output_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    report.assert_clean()

    parity = None
    if check_python_parity:
        parity = compare_uniform_search(
            python_environment,
            native_environment,
            iterations=test.mcts_iterations,
            seed=test.seed,
            native_result=native_result,
        )
        with (test.output_dir / "uniform_parity.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(parity.as_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        parity.assert_exact()

    return report, parity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-python-parity", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    defaults = GV4MCTSTestConfig()
    test = GV4MCTSTestConfig(
        output_dir=args.output_dir,
        predictor_cache_dir=(
            defaults.predictor_cache_dir
            if args.cache_dir is None
            else args.cache_dir
        ),
        predictor_cache_mode=args.cache_mode,
        timing_mode=args.timing_mode,
        mcts_iterations=args.iterations,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    report, parity = run(
        test,
        validate_only=args.validate_only,
        check_python_parity=not args.skip_python_parity,
    )
    print(
        f"Native GV4 trace validation passed: {report.total_checks} checks, "
        f"coverage={dict(sorted(report.coverage.items()))}"
    )
    if parity is not None:
        print(
            f"Uniform root parity exact: {parity.compared_actions} actions, "
            f"best_action={parity.native_best_action}"
        )
    print(f"outputs: {test.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
