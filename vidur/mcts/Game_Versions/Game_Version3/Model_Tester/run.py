from __future__ import annotations

import argparse
from dataclasses import replace

from .config import DEFAULT_MODEL_TESTER_CONFIG
from .runner import run_model_vs_trivial_tester


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GV3 model-vs-trivial arena tester.")
    parser.add_argument(
        "--model-kind",
        choices=("torch_checkpoint", "classical_joblib"),
        default=DEFAULT_MODEL_TESTER_CONFIG.model_kind,
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_TESTER_CONFIG.model_checkpoint_path,
        help="Path to torch checkpoint or classical .joblib model.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_MODEL_TESTER_CONFIG.output_dir,
    )
    parser.add_argument("--num-games", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.num_games)
    parser.add_argument(
        "--game-id-start",
        type=int,
        default=DEFAULT_MODEL_TESTER_CONFIG.game_id_start,
    )
    parser.add_argument(
        "--environment-lang",
        choices=("python", "native"),
        default=DEFAULT_MODEL_TESTER_CONFIG.environment_lang,
    )
    parser.add_argument(
        "--history-hops-min",
        type=int,
        default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_min,
    )
    parser.add_argument(
        "--history-hops-max",
        type=int,
        default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_max,
    )
    parser.add_argument(
        "--history-seed",
        type=int,
        default=DEFAULT_MODEL_TESTER_CONFIG.history_seed,
    )
    parser.add_argument(
        "--arena-time-limit-sec",
        type=float,
        default=DEFAULT_MODEL_TESTER_CONFIG.arena_time_limit_sec,
    )
    parser.add_argument(
        "--arena-max-total-turns",
        type=int,
        default=DEFAULT_MODEL_TESTER_CONFIG.arena_max_total_turns,
    )
    parser.add_argument(
        "--arena-max-controller-cleanup-steps",
        type=int,
        default=DEFAULT_MODEL_TESTER_CONFIG.arena_max_controller_cleanup_steps,
    )
    parser.add_argument(
        "--bootstrap-model-version",
        type=int,
        default=DEFAULT_MODEL_TESTER_CONFIG.bootstrap_model_version,
    )
    parser.add_argument(
        "--write-model-action-detail-logs",
        action="store_true",
        default=DEFAULT_MODEL_TESTER_CONFIG.write_model_action_detail_logs,
    )
    parser.add_argument(
        "--no-arena-game-logs",
        action="store_true",
        help="Disable per-game arena logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind=str(args.model_kind),
        model_checkpoint_path=str(args.model_path),
        output_dir=str(args.output_dir),
        num_games=int(args.num_games),
        game_id_start=int(args.game_id_start),
        environment_lang=str(args.environment_lang),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        arena_time_limit_sec=float(args.arena_time_limit_sec),
        arena_max_total_turns=int(args.arena_max_total_turns),
        arena_max_controller_cleanup_steps=int(args.arena_max_controller_cleanup_steps),
        bootstrap_model_version=int(args.bootstrap_model_version),
        write_arena_game_logs=not bool(args.no_arena_game_logs),
        write_model_action_detail_logs=bool(args.write_model_action_detail_logs),
    )
    out_csv = run_model_vs_trivial_tester(cfg)
    print(f"[Model_Tester] Completed. Results: {out_csv}")


if __name__ == "__main__":
    main()
