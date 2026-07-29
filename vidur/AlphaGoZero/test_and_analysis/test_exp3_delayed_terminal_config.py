"""Contract checks for the EXP3 delayed-terminal experiment."""

from __future__ import annotations

from pathlib import Path

from vidur.AlphaGoZero import agz_train_eval_promote as trainer
from vidur.AlphaGoZero import deploy


def main() -> None:
    env_path = Path(__file__).resolve().parents[3] / "settingUpServers" / "exp3_delayed_terminal_env.sh"
    text = env_path.read_text(encoding="utf-8")
    expected_lines = {
        "export AGZ_MCTS_ITERATIONS=2000",
        "export AGZ_EVAL_MCTS_ITERATIONS=2000",
        "export AGZ_SELFPLAY_ARENA_TIME_LIMIT_SEC=12",
        "export AGZ_REPLAY_SAMPLE_WINDOW_SEC=5",
        "export AGZ_DISCOUNT_FACTOR=0.99",
        "export AGZ_NATIVE_SEARCH_MODE=full_tree",
        "export AGZ_ROLLOUT_COUNT=0",
        "export AGZ_REPLAY_SAMPLER=indexed_v1",
        "export AGZ_EVAL_GAMES=140",
        "export AGZ_BENCHMARK_GAMES=50",
        "export AGZ_ROLE_PROMOTION_WIN_THRESHOLD=81",
    }
    for line in expected_lines:
        assert line in text, line

    assert not trainer._role_candidate_promoted(
        wins=81,
        games_compared=140,
        expected_games=140,
        strict_win_threshold=81,
    )
    assert trainer._role_candidate_promoted(
        wins=82,
        games_compared=140,
        expected_games=140,
        strict_win_threshold=81,
    )
    assert not trainer._role_candidate_promoted(
        wins=82,
        games_compared=139,
        expected_games=140,
        strict_win_threshold=81,
    )

    forwarded = deploy._agz_env_prefix.__code__.co_consts
    flattened = repr(forwarded)
    for key in (
        "AGZ_REPLAY_SAMPLER",
        "AGZ_REPLAY_INDEX_BUILD_WORKERS",
        "AGZ_REPLAY_EXTRACTION_WORKERS",
        "AGZ_EVAL_GAMES",
        "AGZ_BENCHMARK_GAMES",
    ):
        assert key in flattened

    print("EXP3 delayed-terminal configuration contract: OK")


if __name__ == "__main__":
    main()
