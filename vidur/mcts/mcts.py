from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import MCTSExploreConfig
from .environment import (
    AdversaryAction,
    ControllerAction,
    VidurMCTSEnvironment,
    VidurMCTSState, 
)


@dataclass
class MCTSNode:
    state: VidurMCTSState
    player: str  # "adversary" or "controller"
    parent: Optional["MCTSNode"] = None
    parent_action: Optional[Tuple[AdversaryAction, ControllerAction]] = None
    visits: int = 0
    cumulative_cost: float = 0.0  # smaller is better for controller
    children: List["MCTSChildEdge"] = field(default_factory=list)
    untried_actions: List[Tuple[AdversaryAction, ControllerAction]] = field(default_factory=list)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


@dataclass
class MCTSChildEdge:
    action: Tuple[AdversaryAction, ControllerAction]
    node: MCTSNode


class VidurMCTS:
    """UCB1-based Monte Carlo Tree Search coordinating adversary and controller."""

    def __init__(
        self,
        env: VidurMCTSEnvironment,
        explore_cfg: MCTSExploreConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._env = env
        self._cfg = explore_cfg
        self._rng = rng or random.Random(0)

    def search(self, iterations: int) -> ControllerAction:
        root = self._create_root() ## ?? In this case root controller should not be making any branches. System is essentially empty. 
        for _ in range(iterations):
            node = self._select(root)
            expanded = self._expand(node)
            leaf = expanded or node
            cost = self._simulate(leaf)
            self._backpropagate(leaf, cost)

        best_child = min(
            root.children,
            key=lambda edge: edge.node.cumulative_cost / max(edge.node.visits, 1),
            default=None,
        )
        if best_child is None:
            return ControllerAction(token_budget=self._env._constraints.interval_request_size)
        return best_child.action[1]

    # ------------------------------------------------------------------ #
    # Core phases
    # ------------------------------------------------------------------ #
    def _create_root(self) -> MCTSNode:
        state = self._env.initial_state()
        node = MCTSNode(state=state, player="adversary")
        node.untried_actions = self._enumerate_joint_actions(node)
        return node

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children and node.is_fully_expanded():
            node = self._best_child(node)
        return node

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        if not node.untried_actions:
            return None
        action = node.untried_actions.pop()
        child_state = self._env.apply_actions(
            node.state,
            adversary_action=action[0],
            controller_action=action[1],
        )
        child = MCTSNode(
            state=child_state,
            player="adversary" if node.player == "controller" else "controller",
            parent=node,
            parent_action=action,
        )
        child.untried_actions = self._enumerate_joint_actions(child)
        node.children.append(MCTSChildEdge(action=action, node=child))
        return child

    def _simulate(self, node: MCTSNode) -> float:
        rollout_state = node.state.fork()
        current_player = node.player

        for _ in range(self._cfg.simulation_random_tries):
            adversary_action = self._rng.choice(
                self._env.sample_adversary_actions(rollout_state, max(1, self._cfg.simulation_random_tries))
            )
            controller_action = self._rng.choice(
                self._env.sample_controller_actions(rollout_state, max(1, self._cfg.simulation_random_tries))
            )
            rollout_state = self._env.apply_actions(
                rollout_state, adversary_action, controller_action
            )
            current_player = "controller" if current_player == "adversary" else "adversary"

        violations, avg_lateness = self._env.evaluate_objective(rollout_state)
        return violations + avg_lateness

    def _backpropagate(self, node: MCTSNode, cost: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.cumulative_cost += cost
            current = current.parent

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _enumerate_joint_actions(
        self, node: MCTSNode
    ) -> List[Tuple[AdversaryAction, ControllerAction]]:
        adversary_actions = self._env.sample_adversary_actions(
            node.state, self._cfg.max_branching
        )
        controller_actions = self._env.sample_controller_actions(
            node.state, self._cfg.max_branching
        )
        joint: List[Tuple[AdversaryAction, ControllerAction]] = []
        for adv in adversary_actions:
            for ctrl in controller_actions:
                joint.append((adv, ctrl))
        self._rng.shuffle(joint)
        return joint

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -float("inf")
        best = None
        for edge in node.children:
            child = edge.node
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = -(child.cumulative_cost / child.visits)
                explore = self._cfg.exploration_constant * math.sqrt(
                    math.log(node.visits) / child.visits
                )
                score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best or node
