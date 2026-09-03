from __future__ import annotations

import math
from pathlib import Path

from vidur.AlphaGoZero import replay_runtime, replay_target_smoke
from vidur.AlphaGoZero.test_and_analysis import calculate_discounted_reward
from vidur.Game_Version3.config import GameVersion2Config

EXPECTED_DISCOUNT_FACTOR = 0.995
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_runtime_discount_defaults_are_consistent() -> None:
    cfg = GameVersion2Config()
    values = [
        cfg.mcts_search.discount_factor,
        replay_runtime.DISCOUNT_FACTOR,
        replay_target_smoke.DISCOUNT_FACTOR,
        calculate_discounted_reward.DISCOUNT_FACTOR,
    ]
    for value in values:
        assert math.isclose(float(value), EXPECTED_DISCOUNT_FACTOR, rel_tol=0.0, abs_tol=1e-12)


def test_native_search_input_default_matches_python_default() -> None:
    headers = [
        REPO_ROOT / "vidur/Game_Version3/native/include/gv2_mcts_dnn.hpp",
        REPO_ROOT / "vidur/Game_Version3_Cpp/include/gv2_mcts_dnn.hpp",
    ]
    for header in headers:
        text = header.read_text(encoding="utf-8")
        assert "double discount_factor = 0.995;" in text
        assert "double discount_factor = 0.98;" not in text


def test_discount_cli_defaults_match_python_default() -> None:
    cli_files = [
        REPO_ROOT / "vidur/bellman_v4_adv/arena_mcts_model_tester.py",
        REPO_ROOT / "vidur/bellman_v4_adv/arena_mcts_value_runner.py",
        REPO_ROOT / "vidur/bellman_v4_adv/mcts_two_step_visit_logger.py",
    ]
    for path in cli_files:
        text = path.read_text(encoding="utf-8")
        assert "--discount-factor" in text
        assert "default=0.995" in text
        assert "default=0.98" not in text


def test_no_discount_default_left_at_098() -> None:
    forbidden = (
        "DISCOUNT_FACTOR = 0.98",
        "discount_factor: float = 0.98",
        "discount_factor = 0.98",
        "default=0.98",
        "\"discount_factor\", 0.98",
        "double discount_factor = 0.98;",
        "# GAMMA: float = 0.98",
        "DEFAULT_GAMMA = 0.98",
        "--discount-factor 0.98",
        "discount_factor=0.98",
        "discount = 0.98",
        "effective_discount(0.98",
        "0.98 *",
    )
    suffixes = {".py", ".cpp", ".hpp", ".h", ".md"}
    this_file = Path(__file__).resolve()
    for path in (REPO_ROOT / "vidur").rglob("*"):
        if path.resolve() == this_file or path.name == "test_experiment_configuration.py" or path.suffix not in suffixes:
            continue
        parts = set(path.parts)
        if "__pycache__" in parts or any(part.startswith("build") for part in parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden:
            assert needle not in text, f"{needle!r} remains in {path}"
