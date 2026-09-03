from __future__ import annotations

from pathlib import Path

from common import import_native_cpp, make_args, prepare_python_roots, run_native_logger


def test_random_native_traces_pass_game_engine_validation() -> None:
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    native = import_native_cpp()
    args = make_args(
        "random_native_trace",
        num_roots=20,
        history_hops_min=0,
        history_hops_max=200,
        history_seed=202611,
        frontier_parity_roots=1,
    )
    _cfg_python, simulator, _env, _explore_cfg, _roots = prepare_python_roots(args)
    stats, frontier_rows, search_rows, detail_rows, feature_rows = run_native_logger(native, simulator, args)

    assert int(stats.get("trace_chains", 0)) > 0
    assert int(stats.get("adv_actions_checked", 0)) >= 0
    assert len(frontier_rows) > 0
    assert len(search_rows) > 0
    assert len(detail_rows) > 0
    assert len(feature_rows) > 0

    trace_csv = Path(args.output_dir) / "history_mcts_iter.csv"
    nlt._validate_trace_csv(trace_csv)


if __name__ == "__main__":
    test_random_native_traces_pass_game_engine_validation()
    print("random_native_trace_test passed")
