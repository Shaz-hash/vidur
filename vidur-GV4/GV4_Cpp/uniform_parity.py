"""Exact Python/native checks for uniform-prior GV4 MCTS."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from GV4_Engine.mcts_value_prior import MCTSConfig, VidurMCTS
from GV4_Engine.state import Player
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment

from . import gv4_native as native


@dataclass(frozen=True, slots=True)
class UniformParityReport:
    iterations: int
    python_best_action: int | None
    native_best_action: int | None
    compared_actions: int
    visit_mismatches: tuple[tuple[int, int, int], ...]
    value_mismatches: tuple[tuple[int, float, float], ...]
    mask_matches: bool

    @property
    def exact(self) -> bool:
        return (
            self.mask_matches
            and self.python_best_action == self.native_best_action
            and not self.visit_mismatches
            and not self.value_mismatches
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "python_best_action": self.python_best_action,
            "native_best_action": self.native_best_action,
            "compared_actions": self.compared_actions,
            "visit_mismatches": [list(item) for item in self.visit_mismatches],
            "value_mismatches": [list(item) for item in self.value_mismatches],
            "mask_matches": self.mask_matches,
            "exact": self.exact,
        }

    def assert_exact(self) -> None:
        if self.exact:
            return
        raise AssertionError(
            "uniform Python/native MCTS diverged: "
            f"best=({self.python_best_action},{self.native_best_action}), "
            f"visit_mismatches={self.visit_mismatches[:5]}, "
            f"value_mismatches={self.value_mismatches[:5]}, "
            f"mask_matches={self.mask_matches}"
        )


def _python_mcts_config(iterations: int, seed: int) -> MCTSConfig:
    config = MCTSConfig()
    config.mcts_iterations = iterations
    config.rng = random.Random(seed)
    config.use_policy_prior = False
    config.controller_prior_model = None
    config.adversary_prior_model = None
    config.root_dirichlet_alpha = 0.0
    config.root_dirichlet_epsilon = 0.0
    config.root_dirichlet_total_concentration = 0.0
    config.log_flag = False
    config.iteration_observer = None
    return config


def compare_uniform_search(
    python_environment: GV4VirtualVidurMCTSEnvironment,
    native_environment: native.Environment,
    *,
    iterations: int,
    seed: int,
    native_result: dict[str, Any] | None = None,
) -> UniformParityReport:
    """Run the same zero-bootstrap search and compare root statistics exactly."""

    python_mcts = VidurMCTS(
        python_environment,
        _python_mcts_config(iterations, seed),
    )
    try:
        python_result = python_mcts.search_dnn(
            None,
            python_environment.initial_state(
                now=0.0,
                next_player=Player.ADVERSARY,
            ),
            "adversary",
            game_id=0,
            root_id=0,
            root_node_id_override=0,
            root_depth=0,
            mcts_iter=iterations,
            model_version=0,
            use_model_bootstrap=False,
        )
        root = python_mcts._root
        if root is None:
            raise RuntimeError("Python MCTS did not retain its root")
        python_stats = {
            int(index): (int(child.visits), float(child.value_sum))
            for index, child in root.children.items()
        }
    finally:
        python_mcts.close()

    if native_result is None:
        native_result = native.run_uniform_mcts(
            native_environment,
            native_environment.initial_state(0.0, native.Player.ADVERSARY),
            native.Player.ADVERSARY,
            iterations,
            1.0,
            0,
            0,
        )
    native_stats = {
        int(item["representative_raw_index"]): (
            int(item["visits"]),
            float(item["value_sum"]),
        )
        for item in native_result["root_action_stats"]
    }

    all_actions = sorted(set(python_stats) | set(native_stats))
    visit_mismatches = []
    value_mismatches = []
    for action in all_actions:
        python_visit, python_value = python_stats.get(action, (-1, math.nan))
        native_visit, native_value = native_stats.get(action, (-1, math.nan))
        if python_visit != native_visit:
            visit_mismatches.append((action, python_visit, native_visit))
        if not math.isclose(
            python_value,
            native_value,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            value_mismatches.append((action, python_value, native_value))

    return UniformParityReport(
        iterations=iterations,
        python_best_action=python_result.best_action_index,
        native_best_action=int(native_result["best_action_index"]),
        compared_actions=len(all_actions),
        visit_mismatches=tuple(visit_mismatches),
        value_mismatches=tuple(value_mismatches),
        mask_matches=(
            list(python_result.valid_mask) == list(native_result["valid_mask"])
        ),
    )


__all__ = ["UniformParityReport", "compare_uniform_search"]
