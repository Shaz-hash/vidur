from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import MCTSExploreConfig
from .environment import (
    AdversaryAction,
    ControllerAction,
    VidurMCTSEnvironment,
    VidurMCTSState,
)


class _MCTSLogger:
    """CSV logger capturing joint actions and resulting state metrics."""

    _FIELDS = [
        "iteration",
        "phase",
        "depth",
        "parent_node_id",
        "node_id",
        "player_to_act",
        "next_player",
        "sim_time",
        "requests_in_system",
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "avg_lateness",
        "objective_cost",
        "state_waiting_ids",
        "state_completed_request_ids",
        "adversary_requests",
        "adversary_prefill_slos",
        "adversary_decode_slos",
        "controller_token_budget",
        "controller_selected_ids",
        "controller_allocations",
        "controller_prefill_allocations",
        "controller_decode_allocations",
        "controller_prefill_total",
        "controller_decode_total",
    ]

    def __init__(self, path: Optional[Union[str, Path]]) -> None:
        self._path = Path(path) if path else None
        self._writer: Optional[csv.DictWriter] = None
        self._file = None

    def log(
        self,
        iteration: int,
        phase: str,
        depth: float,
        parent_node_id: Union[int, str, None],
        node_id: Union[int, str],
        player_to_act: str,
        next_player: str,
        state_snapshot: Dict[str, Any],
        adversary_action: Optional[AdversaryAction],
        controller_action: Optional[ControllerAction],
        objective_cost: float,
    ) -> None:
        if not self._path:
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
            self._writer.writeheader()

        row = {
            "iteration": iteration,
            "phase": phase,
            "depth": depth,
            "parent_node_id": "" if parent_node_id is None else str(parent_node_id),
            "node_id": str(node_id),
            "player_to_act": player_to_act,
            "next_player": next_player,
            "sim_time": state_snapshot["sim_time"],
            "requests_in_system": state_snapshot["requests_in_system"],
            "requests_generated": state_snapshot["requests_generated"],
            "requests_completed": state_snapshot["requests_completed"],
            "slo_violations": state_snapshot["slo_violations"],
            "avg_lateness": state_snapshot["avg_lateness"],
            "objective_cost": objective_cost,
            "state_waiting_ids": json.dumps(state_snapshot["waiting_request_ids"]),
            "state_completed_request_ids": json.dumps(
                state_snapshot["completed_request_ids"]
            ),
            "adversary_requests": json.dumps(
                [
                    {
                        "prefill_tokens": spec.prefill_tokens,
                        "decode_tokens": spec.decode_tokens,
                        "prefill_slo": spec.prefill_slo,
                        "decode_slo": spec.decode_slo,
                    }
                    for spec in adversary_action.requests
                ]
            )
            if adversary_action
            else json.dumps([]),
            "adversary_prefill_slos": json.dumps(
                [spec.prefill_slo for spec in adversary_action.requests]
            )
            if adversary_action
            else json.dumps([]),
            "adversary_decode_slos": json.dumps(
                [spec.decode_slo for spec in adversary_action.requests]
            )
            if adversary_action
            else json.dumps([]),
            "controller_token_budget": controller_action.token_budget
            if controller_action
            else "",
            "controller_selected_ids": json.dumps(
                controller_action.selected_request_ids or []
            )
            if controller_action
            else json.dumps([]),
            "controller_allocations": json.dumps(
                controller_action.token_allocations if controller_action else {}
            ),
            "controller_prefill_allocations": json.dumps(
                controller_action.prefill_allocations if controller_action else {}
            ),
            "controller_decode_allocations": json.dumps(
                controller_action.decode_allocations if controller_action else {}
            ),
            "controller_prefill_total": sum(
                controller_action.prefill_allocations.values()
            )
            if controller_action
            else 0,
            "controller_decode_total": sum(
                controller_action.decode_allocations.values()
            )
            if controller_action
            else 0,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _compute_objective_cost(violations: int, avg_lateness: float) -> float:
    return violations + avg_lateness


@dataclass
class MCTSNode:
    state: VidurMCTSState
    player: str  # "adversary" or "controller"
    node_id: int = 0
    depth: int = 0
    parent: Optional["MCTSNode"] = None
    parent_action: Optional[Union[AdversaryAction, ControllerAction]] = None
    visits: int = 0
    cumulative_cost: float = 0.0  # smaller is better for controller
    children: List["MCTSChildEdge"] = field(default_factory=list)
    untried_actions: List[Union[AdversaryAction, ControllerAction]] = field(default_factory=list)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


@dataclass
class MCTSChildEdge:
    action: Union[AdversaryAction, ControllerAction]
    node: MCTSNode


class VidurMCTS:
    """UCB1-based Monte Carlo Tree Search coordinating adversary and controller."""

    def __init__(
        self,
        env: VidurMCTSEnvironment,
        explore_cfg: MCTSExploreConfig,
        rng: Optional[random.Random] = None,
        log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._env = env
        self._cfg = explore_cfg
        self._rng = rng or random.Random(0)
        self._logger = _MCTSLogger(log_path)
        self._node_counter = 0
        self._current_iteration = 0

    def search(self, iterations: int) -> ControllerAction:
        root = self._create_root()
        try:
            root_cost = self._log_state(
                iteration=0,
                phase="root",
                node=root,
                parent_id=None,
                acting_player=None,
                next_player=root.player,
                action=None,
            )
            root.cumulative_cost = root_cost
            for itr in range(iterations):
                self._current_iteration = itr + 1
                node = self._select(root)
                expanded = self._expand(node)
                leaf = expanded or node
                cost = self._simulate(leaf)
                self._backpropagate(leaf, cost)
        finally:
            self._logger.close()

        # TODO: reinstate policy extraction (best-action selection) once we export
        # the decision tree. For now return a placeholder action so callers can
        # inspect the logged tree instead of a single suggestion.
        return ControllerAction(
            token_budget=self._env._constraints.interval_request_size
        )

    # ------------------------------------------------------------------ #
    # Core phases
    # ------------------------------------------------------------------ #
    def _create_root(self) -> MCTSNode:
        state = self._env.initial_state()
        node = MCTSNode(
            state=state,
            player="adversary",
            node_id=self._next_node_id(),
            depth=0,
        )
        node.untried_actions = self._enumerate_actions(node)
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
        if node.player == "adversary":
            child_state = self._env.apply_adversary_action_only(node.state, action)
            next_player = "controller"
        else:
            child_state = self._env.apply_controller_action_only(node.state, action)
            next_player = "adversary"
        child = MCTSNode(
            state=child_state,
            player=next_player,
            node_id=self._next_node_id(),
            depth=node.depth + 1,
            parent=node,
            parent_action=action,
        )
        child.untried_actions = self._enumerate_actions(child)
        node.children.append(MCTSChildEdge(action=action, node=child))
        self._log_state(
            iteration=self._current_iteration,
            phase="tree",
            node=child,
            parent_id=node.node_id,
            acting_player=node.player,
            next_player=child.player,
            action=action,
        )
        return child

    def _simulate(self, node: MCTSNode) -> float:
        total_cost = 0.0
        num_trials = max(1, self._cfg.simulation_random_tries)

        for trial in range(num_trials):
            rollout_state = node.state.fork()
            current_player = node.player
            parent_id: Union[int, str] = node.node_id

            for depth_idx in range(self._cfg.simulation_depth):
                for turn in range(2):
                    if current_player == "adversary":
                        candidates = self._env.sample_adversary_actions(
                            rollout_state, self._cfg.max_branching
                        )
                        if not candidates:
                            action = None
                        else:
                            action = self._rng.choice(candidates)
                            rollout_state = self._env.apply_adversary_action_only(
                                rollout_state, action
                            )
                    else:
                        print("CONTROLLER's PHASE SIMULATION : ")
                        print(f"rollout_{self._current_iteration}_{node.node_id}_{trial}_{depth_idx}_{turn}")
                        candidates = self._env.sample_controller_actions(
                            rollout_state, self._cfg.max_branching
                        )
                        if not candidates:
                            action = None
                        else:
                            action = self._rng.choice(candidates)
                            print("Action choose for this phase is : ", action)
                            rollout_state = self._env.apply_controller_action_only(
                                rollout_state, action
                            )

                    if action is not None:
                        node_id = (
                            f"rollout_{self._current_iteration}_{node.node_id}_{trial}_{depth_idx}_{turn}"
                        )
                        self._log_state(
                            iteration=self._current_iteration,
                            phase="rollout",
                            node_id=node_id,
                            depth=node.depth + depth_idx + (turn + 1) / 2,
                            state=rollout_state,
                            parent_id=parent_id,
                            acting_player=current_player,
                            next_player="controller"
                            if current_player == "adversary"
                            else "adversary",
                            action=action,
                        )
                        parent_id = node_id

                    current_player = (
                        "controller" if current_player == "adversary" else "adversary"
                    )

            violations, avg_lateness = self._env.evaluate_objective(rollout_state)
            total_cost += _compute_objective_cost(violations, avg_lateness)

        return total_cost / num_trials

    def _backpropagate(self, node: MCTSNode, cost: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.cumulative_cost += cost
            current = current.parent

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _enumerate_actions(
        self, node: MCTSNode
    ) -> List[Union[AdversaryAction, ControllerAction]]:
        if node.player == "adversary":
            actions = self._env.sample_adversary_actions(
                node.state, self._cfg.max_branching
            )
        else:
            actions = self._env.sample_controller_actions(
                node.state, self._cfg.max_branching
            )
        self._rng.shuffle(actions)
        return actions

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -float("inf")
        best = None
        for edge in node.children:
            child = edge.node
            if child.visits == 0:
                score = float("inf")
            else:
                mean_cost = child.cumulative_cost / child.visits
                exploit = -mean_cost if node.player == "controller" else mean_cost
                explore = self._cfg.exploration_constant * (math.log(node.visits) / child.visits)
                explore = self._cfg.exploration_constant * math.sqrt(
                    math.log(node.visits) / child.visits
                )
                score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best or node

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    def _log_state(
        self,
        iteration: int,
        phase: str,
        node: Optional[MCTSNode] = None,
        parent_id: Optional[Union[int, str]] = None,
        acting_player: Optional[str] = None,
        next_player: Optional[str] = None,
        action: Optional[Union[AdversaryAction, ControllerAction]] = None,
        node_id: Optional[Union[int, str]] = None,
        depth: Optional[float] = None,
        state: Optional[VidurMCTSState] = None,
    ) -> float:
        if node is not None:
            state = node.state
            node_id = node.node_id
            depth = node.depth
            acting_player = node.parent.player if node.parent else acting_player
            next_player = node.player
        assert state is not None
        snapshot = self._env.describe_state(state)
        violations = snapshot["slo_violations"]
        avg_lateness = snapshot["avg_lateness"]
        cost = _compute_objective_cost(violations, avg_lateness)
        adversary_action = (
            action if acting_player == "adversary" else None
        )
        controller_action = (
            action if acting_player == "controller" else None
        )
        player_label = acting_player or (node.player if node else "")
        next_label = next_player or player_label
        depth_value = depth if depth is not None else 0.0
        self._logger.log(
            iteration=iteration,
            phase=phase,
            depth=depth_value,
            parent_node_id=parent_id,
            node_id=node_id or "root",
            player_to_act=player_label,
            next_player=next_label,
            state_snapshot=snapshot,
            adversary_action=adversary_action,
            controller_action=controller_action,
            objective_cost=cost,
        )
        return cost

    def _next_node_id(self) -> int:
        node_id = self._node_counter
        self._node_counter += 1
        return node_id
