from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import time ## Just to test the performance 

from .launch_mcts_job import MCTSExploreConfig
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
        "controller_heuristic",         # NEW
        "controller_strategy",          # NEW
    ]

    def __init__(
        self,
        path: Optional[Union[str, Path]],
        *,
        log_rollouts: bool = True,
        flush_every: int = 1,
    ) -> None:
        self._path = Path(path) if path else None
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._log_rollouts = log_rollouts
        self._flush_every = max(0, int(flush_every))
        self._row_count = 0

    def enabled_for(self, phase: str) -> bool:
        if not self._path:
            return False
        if phase == "rollout" and not self._log_rollouts:
            return False
        return True

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
        if not self.enabled_for(phase):
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
            self._writer.writeheader()
        ctrl_heuristic = (
            controller_action.heuristic
            if controller_action is not None and getattr(controller_action, "heuristic", None) is not None
            else ""
        )
        ctrl_strategy = (
            controller_action.strategy
            if controller_action is not None and getattr(controller_action, "strategy", None) is not None
            else ""
        )

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
            "controller_heuristic": ctrl_heuristic,
            "controller_strategy": ctrl_strategy,
        }
        self._writer.writerow(row)
        self._row_count += 1
        if self._flush_every == 1:
            self._file.flush()
        elif self._flush_every > 1 and (self._row_count % self._flush_every == 0):
            self._file.flush()

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not self._path or not rows:
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
            self._writer.writeheader()
        for row in rows:
            if not self.enabled_for(row.get("phase", "")):
                continue
            self._writer.writerow(row)
            self._row_count += 1
        if self._flush_every == 1 or (self._flush_every > 1 and (self._row_count % self._flush_every == 0)):
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class _MCTSTreeLogger:
    """CSV logger for tree-only rows, with heuristic/strategy columns."""

    # _FIELDS = [
    #     "iteration",
    #     "phase",
    #     "depth",
    #     "parent_node_id",
    #     "node_id",
    #     "player_to_act",
    #     "next_player",
    #     "sim_time",
    #     "requests_in_system",
    #     "requests_generated",
    #     "requests_completed",
    #     "slo_violations",
    #     "avg_lateness",
    #     "objective_cost",
    #     "state_waiting_ids",
    #     "state_completed_request_ids",
    #     "adversary_requests",
    #     "adversary_prefill_slos",
    #     "adversary_decode_slos",
    #     "controller_token_budget",
    #     "controller_selected_ids",
    #     "controller_allocations",
    #     "controller_prefill_allocations",
    #     "controller_decode_allocations",
    #     "controller_prefill_total",
    #     "controller_decode_total",
    #     "controller_heuristic",    # extra
    #     "controller_strategy",     # extra
    # ]


    _FIELDS = [
        "iteration",
        "node_id",
        "parent_node_id",
        "depth",
        "player_to_act",
        "action_type",       # "adversary" or "controller"
        "visits",            # node.visits
        "parent_visits",     # parent.visits (0 if root)
        "node_cost",         # mean cost at this node
        "ucb_score",
        "action_repr",       # simple string/JSON for the action
    ]
    def __init__(self, path: Optional[Union[str, Path]], flush_every: int = 1) -> None:
        self._path = Path(path) if path else None
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._flush_every = max(0, int(flush_every))
        self._row_count = 0

    def write_row(self, row: Dict[str, Any]) -> None:
        if not self._path:
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._row_count += 1
        if self._flush_every == 1 or (
            self._flush_every > 1 and (self._row_count % self._flush_every == 0)
        ):
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None



def _compute_objective_cost(violations: int, avg_lateness: float) -> float:
    return violations + avg_lateness


@dataclass
class MCTSNode:
    # state: VidurMCTSState
    player: str  # "adversary" or "controller"
    node_id: int = 0
    depth: int = 0
    parent: Optional["MCTSNode"] = None
    parent_action: Optional[Union[AdversaryAction, ControllerAction]] = None
    visits: int = 0
    cumulative_cost: float = 0.0  # smaller is better for controller
    children: List["MCTSChildEdge"] = field(default_factory=list)
    untried_actions: List[Union[AdversaryAction, ControllerAction]] = field(default_factory=list)
    sim_time: float = 0.0   # NEW: simulator time at this node after performing action



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
        tree_log_path: Optional[Union[str, Path]] = None,   # NEW
        *,
        log_rollouts: bool = True,
        logger_flush_every: int = 1,
        verbose: bool = False,
        tree_dump_interval: int = 0,
        history_depth: int = 0,          # NEW
    ) -> None:
        self._env = env
        self._cfg = explore_cfg
        self._rng = rng or random.Random(0)
        self._logger = _MCTSLogger(
            log_path, log_rollouts=log_rollouts, flush_every=logger_flush_every
        )
        self._node_counter = 0
        self._current_iteration = 0
        self._verbose = verbose

        self._tree_dump_interval = max(0, int(tree_dump_interval))
        self._history_depth = max(0, int(history_depth))    # NEW

        self._root: Optional[MCTSNode] = None
        
        # Derive default tree path if not provided explicitly
        if tree_log_path is None and log_path:
            base = Path(log_path)
            tree_log_path = base.with_name(base.stem + "_tree" + base.suffix)

        self._tree_logger = _MCTSTreeLogger(tree_log_path, flush_every=logger_flush_every)

        self._history_root_state: Optional[VidurMCTSState] = None
        self._history_root_node: Optional[MCTSNode] = None
        # ## Pre Computing the necessary states here 
        # max_budget = self._env._max_request_tokens_allowed()
        # self._env.precompute_controller_state_space(token_budget=max_budget)

    def search(self, iterations: int) -> ControllerAction:
        root = self._create_root()
        try:
            self._root = root

            # --- Random history prefix (in plies: adversary/controller steps) ---
            current = root
            current_state = self._replay_to_node(root)  # or reuse state from _create_root if you return it
            for step in range(self._history_depth):
                if not current.untried_actions:
                    current.untried_actions = self._enumerate_actions(current, current_state)
                    if not current.untried_actions:
                        break

                action = self._rng.choice(current.untried_actions)
                current.untried_actions.remove(action)

                if current.player == "adversary":
                    new_state = self._env.apply_adversary_action_only(current_state, action)
                    next_player = "controller"
                else:
                    new_state = self._env.apply_controller_action_only(current_state, action)
                    next_player = "adversary"

                child = MCTSNode(
                    player=next_player,
                    node_id=self._next_node_id(),
                    depth=current.depth + 1,
                    parent=current,
                    parent_action=action,
                    sim_time=new_state.simulator._time,   # NEW
                )
                child.untried_actions = self._enumerate_actions(child, new_state)
                current.children.append(MCTSChildEdge(action=action, node=child))

                self._log_state(
                    iteration=0,
                    phase="INITIAL_HISTORY_GEN",
                    node=child,
                    parent_id=current.node_id,
                    acting_player=current.player,
                    next_player=child.player,
                    action=action,
                    state=new_state,
                )

                current = child
                current_state = new_state

        
            # after the history loop
            root_for_search = current
            self._root = root_for_search
            self._history_root_node = root_for_search
            self._history_root_state = current_state


            # --- Vanilla MCTS from history root ---
            for itr in range(iterations):
                if self._env.controller_states_fully_visited():
                    if self._verbose:
                        print("Stopping MCTS: all controller states visited >= 1")
                    break

                self._current_iteration = itr + 1
                node = self._select(root_for_search)

                # Reconstruct state at selected node once
                # Reconstruct state at selected node once
                if self._history_root_node is not None and node is self._history_root_node and self._history_root_state is not None:
                    parent_state = self._history_root_state
                else:
                    parent_state = self._replay_to_node(node)
                # Expand once and reuse the resulting state
                expanded, child_state = self._expand(node, parent_state)
                leaf = expanded or node
                leaf_state = child_state if expanded is not None else parent_state

                cost = self._simulate(leaf, leaf_state)
                self._backpropagate(leaf, cost)

                if (
                    self._tree_dump_interval > 0
                    and (itr + 1) % self._tree_dump_interval == 0
                ):
                    self._dump_tree_snapshot(iteration=itr + 1)
        finally:
            self._logger.close()
            self._tree_logger.close()

        return ControllerAction(
            token_budget=self._env._constraints.interval_request_size
        )

    # ------------------------------------------------------------------ #
    # Core phases
    # ------------------------------------------------------------------ #
    def _create_root(self) -> MCTSNode:
        state = self._env.initial_state()
        node = MCTSNode(
            player="adversary",
            node_id=self._next_node_id(),
            depth=0,
            sim_time=state.simulator._time,  # typically 0.0
        )
        node.untried_actions = self._enumerate_actions(node, state)
        cost = self._log_state(
            iteration=0,
            phase="root",
            node=node,
            parent_id=None,
            acting_player=None,
            next_player=node.player,
            action=None,
            state=state,
        )
        node.cumulative_cost = cost
        return node


    def _replay_to_node(self, node: MCTSNode) -> VidurMCTSState:
        """Reconstruct a fresh state at `node` by replaying actions.

        If a random history prefix was generated, we treat the history root
        state as the baseline and only replay actions *below* that node.
        Otherwise, we start from a fresh initial state and replay from the
        true root.
        """
        # Decide the baseline state and the cut node in the ancestry.
        if self._history_root_node is not None and self._history_root_state is not None:
            baseline_state = self._history_root_state
            cut_node = self._history_root_node
        else:
            baseline_state = self._env.initial_state()
            cut_node = None

        # Collect path from `cut_node` (exclusive) down to `node`.
        path: List[MCTSNode] = []
        cur = node
        while cur is not None and cur is not cut_node:
            path.append(cur)
            cur = cur.parent
        path.reverse()

        # Work on a fork so we never mutate the stored baseline state.
        state = baseline_state.fork()

        # Replay actions inplace along the path segment.
        for n in path:
            parent = n.parent
            action = n.parent_action
            if parent is None or action is None:
                continue
            if parent.player == "adversary":
                state = self._env.apply_adversary_action_only(state, action, inplace=True)
            else:
                state = self._env.apply_controller_action_only(state, action, inplace=True)

        return state
    

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children and node.is_fully_expanded():
            node = self._best_child(node)
        return node

    def _expand(
        self,
        node: MCTSNode,
        parent_state: VidurMCTSState,
    ) -> Tuple[Optional[MCTSNode], Optional[VidurMCTSState]]:
        if not node.untried_actions:
            return None, None

        action = node.untried_actions.pop()

        if node.player == "adversary":
            child_state = self._env.apply_adversary_action_only(parent_state, action)
            next_player = "controller"
        else:
            child_state = self._env.apply_controller_action_only(parent_state, action)
            next_player = "adversary"

        child = MCTSNode(
            player=next_player,
            node_id=self._next_node_id(),
            depth=node.depth + 1,
            parent=node,
            parent_action=action,
            sim_time=child_state.simulator._time,  # NEW
        )
        child.untried_actions = self._enumerate_actions(child, child_state)
        node.children.append(MCTSChildEdge(action=action, node=child))

        self._log_state(
            iteration=self._current_iteration,
            phase="tree",
            node=child,
            parent_id=node.node_id,
            acting_player=node.player,
            next_player=child.player,
            action=action,
            state=child_state,
        )

        return child, child_state
 
    #     #print(f" Expansion time : {t2 - t0:.4f}s")
    #     return child

    # def _simulate(self, node: MCTSNode, start_state: VidurMCTSState) -> float:
    #     total_cost = 0.0
    #     num_trials = max(1, self._cfg.simulation_random_tries)

    #     rollout_batch_rows: List[Dict[str, Any]] = []
    #     for trial in range(num_trials):
    #         # rollout_state = node.state.fork()
    #         # current_player = node.player
    #         # base_state = self._replay_to_node(node)
    #         # rollout_state = base_state.fork()
    #         if num_trials > 1 :
    #             rollout_state = start_state.fork() 
    #         else :
    #             rollout_state = start_state

    #         current_player = node.player
    #         parent_id: Union[int, str] = node.node_id

    #         for depth_idx in range(self._cfg.simulation_depth):
    #             for turn in range(2):
    #                 if current_player == "adversary":
    #                     candidates = self._env.sample_adversary_actions(
    #                         rollout_state, self._cfg.max_branching
    #                     )
    #                     if not candidates:
    #                         action = None
    #                     else:
    #                         action = self._rng.choice(candidates)
    #                         rollout_state = self._env.apply_adversary_action_only(
    #                             rollout_state, action, inplace=True
    #                         )
    #                 else:
    #                     candidates = self._env.sample_controller_actions(
    #                         rollout_state, self._cfg.max_branching, False
    #                     )
    #                     if not candidates:
    #                         action = None
    #                     else:
    #                         action = self._rng.choice(candidates)
    #                         rollout_state = self._env.apply_controller_action_only(
    #                             rollout_state, action, inplace=True
    #                         )

    #                 if action is not None:
    #                     if self._logger.enabled_for("rollout"):
    #                         node_id = (
    #                             f"rollout_{self._current_iteration}_{node.node_id}_{trial}_{depth_idx}_{turn}"
    #                         )
    #                         row = self._make_log_row(
    #                             iteration=self._current_iteration,
    #                             phase="rollout",
    #                             node_id=node_id,
    #                             depth=node.depth + depth_idx + (turn + 1) / 2,
    #                             state=rollout_state,
    #                             parent_id=parent_id,
    #                             acting_player=current_player,
    #                             next_player=(
    #                                 "controller" if current_player == "adversary" else "adversary"
    #                             ),
    #                             action=action,
    #                         )
    #                         rollout_batch_rows.append(row)
    #                         parent_id = node_id

    #                 current_player = (
    #                     "controller" if current_player == "adversary" else "adversary"
    #                 )

    #         violations, avg_lateness = self._env.evaluate_objective(rollout_state)
    #         total_cost += _compute_objective_cost(violations, avg_lateness)

    #     # Commit rollout rows in a single batch for this simulate() call.
    #     if rollout_batch_rows:
    #         self._logger.write_rows(rollout_batch_rows)
    #     return total_cost / num_trials

    def _simulate(self, node: MCTSNode, start_state: VidurMCTSState) -> float:
        total_cost = 0.0
        num_trials = max(1, self._cfg.simulation_random_tries)

        rollout_batch_rows: List[Dict[str, Any]] = []

        # NEW: compute horizon based on the parent node's simulator time, read from the node
        base_node = node.parent if node.parent is not None else node 
        parent_time = getattr(base_node, "sim_time", 0.0)
        horizon_time = parent_time + float(self._cfg.simulation_depth)
        total_dt = 0.0  # accumulate rollout duration across trials
        for trial in range(num_trials):
            if num_trials > 1:
                rollout_state = start_state.fork()
            else:
                rollout_state = start_state

            current_player = node.player
            parent_id: Union[int, str] = node.node_id

            max_steps = max(1, int(self._cfg.simulation_depth) * 80)
            steps = 0
            depth_idx = 0

            while rollout_state.simulator._time < horizon_time and steps < max_steps:
                for turn in range(2):
                    if rollout_state.simulator._time >= horizon_time or steps >= max_steps:
                        break

                    if current_player == "adversary":
                        candidates = self._env.sample_adversary_actions(
                            rollout_state, self._cfg.max_branching
                        )
                        if not candidates:
                            action = None
                        else:
                            action = self._rng.choice(candidates)
                            rollout_state = self._env.apply_adversary_action_only(
                                rollout_state, action, inplace=True
                            )
                    else:
                        candidates = self._env.sample_controller_actions(
                            rollout_state, self._cfg.max_branching, False
                        )
                        if not candidates:
                            action = None
                        else:
                            action = self._rng.choice(candidates)
                            rollout_state = self._env.apply_controller_action_only(
                                rollout_state, action, inplace=True
                            )

                    if action is not None and self._logger.enabled_for("rollout"):
                        node_id = (
                            f"rollout_{self._current_iteration}_{node.node_id}_{trial}_{depth_idx}_{turn}"
                        )
                        row = self._make_log_row(
                            iteration=self._current_iteration,
                            phase="rollout",
                            node_id=node_id,
                            depth=node.depth + depth_idx + (turn + 1) / 2,
                            state=rollout_state,
                            parent_id=parent_id,
                            acting_player=current_player,
                            next_player=(
                                "controller" if current_player == "adversary" else "adversary"
                            ),
                            action=action,
                        )
                        rollout_batch_rows.append(row)
                        parent_id = node_id

                    current_player = (
                        "controller" if current_player == "adversary" else "adversary"
                    )
                    steps += 1

                depth_idx += 1

            violations, avg_lateness = self._env.evaluate_objective(rollout_state)
            total_cost += _compute_objective_cost(violations, avg_lateness)

            dt = rollout_state.simulator._time - parent_time
            if dt <= 0.0:
                dt = 1e-9  # guard against divide-by-zero
            total_dt += dt

        if rollout_batch_rows:
            self._logger.write_rows(rollout_batch_rows)
        # Base average cost
        base_cost = total_cost / num_trials
        avg_dt = total_dt / max(num_trials, 1)

        # Scale by (simulation_depth / actual_avg_duration)
        sim_depth = float(self._cfg.simulation_depth)
        if avg_dt > 0.0:
            scaled_cost = base_cost * (sim_depth / avg_dt)
        else:
            scaled_cost = base_cost

        # Debug/dummy log row: same iteration, node, parent, next player;
        # all snapshot fields empty/zero except objective_cost.
        if self._logger.enabled_for("rollout"):
            dummy_snapshot = {
                "sim_time": 0.0,
                "requests_in_system": 0,
                "requests_generated": 0,
                "requests_completed": 0,
                "slo_violations": 0,
                "avg_lateness": 0.0,
                "waiting_request_ids": [],
                "completed_request_ids": [],
            }
            parent_id_field = node.parent.node_id if node.parent is not None else None
            next_player = "controller" if node.player == "adversary" else "adversary"
            self._logger.log(
                iteration=self._current_iteration,
                phase="rollout_scaled",   # distinguish from real rollout rows
                depth=node.depth,
                parent_node_id=parent_id_field,
                node_id=node.node_id,
                player_to_act=node.player,
                next_player=next_player,
                state_snapshot=dummy_snapshot,
                adversary_action=None,
                controller_action=None,
                objective_cost=scaled_cost,
            )

        return scaled_cost

        # if rollout_batch_rows:
        #     self._logger.write_rows(rollout_batch_rows)
        # return total_cost / num_trials


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
        self, node: MCTSNode, state: Optional[VidurMCTSState] = None
    ) -> List[Union[AdversaryAction, ControllerAction]]:
 
        # if node.player == "adversary":
        #     actions = self._env.sample_adversary_actions(
        #         node.state, self._cfg.max_branching
        #     )
        # else:
        #     actions = self._env.sample_controller_actions(
        #         node.state, self._cfg.max_branching
        #     )
        # self._rng.shuffle(actions)
        # return actions

        # If no state given, reconstruct it from the root
        if state is None:
            state = self._replay_to_node(node)
        if node.player == "adversary":
            actions = self._env.sample_adversary_actions(
                state, self._cfg.max_branching
            )
        else:
            actions = self._env.sample_controller_actions(
                state, self._cfg.max_branching
            )
        self._rng.shuffle(actions)
        return actions



    def _best_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -float("inf")
        best = None
        # Precompute ln(N) once; guard against zero.
        parent_visits = max(1, node.visits)
        ln_parent = math.log(parent_visits)
        for edge in node.children:
            child = edge.node
            if child.visits == 0:
                score = float("inf")
            else:
                mean_cost = child.cumulative_cost / child.visits
                exploit = -mean_cost if node.player == "controller" else mean_cost
                explore = self._cfg.exploration_constant * math.sqrt(
                    ln_parent / max(1, child.visits)
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
        # if node is not None:
        #     state = node.state
        #     node_id = node.node_id
        #     depth = node.depth
        #     acting_player = node.parent.player if node.parent else acting_player
        #     next_player = node.player
        # assert state is not None
      
        # If node is given and id/depth not provided, fill them
        if node is not None:
            node_id = node_id if node_id is not None else node.node_id
            depth = depth if depth is not None else node.depth
            acting_player = acting_player or (node.parent.player if node.parent else None)
            next_player = next_player or node.player
        if state is None:
            if node is None:
                raise ValueError("Either `state` or `node` must be provided to _log_state")
            state = self._replay_to_node(node)


        snapshot = self._env.describe_state(state)
   
        violations = snapshot["slo_violations"]
        avg_lateness = snapshot["avg_lateness"]
        cost = _compute_objective_cost(violations, avg_lateness)
        adversary_action = (action if acting_player == "adversary" else None)
        controller_action = (action if acting_player == "controller" else None)
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

    def _make_log_row(
        self,
        iteration: int,
        phase: str,
        *,
        parent_id: Optional[Union[int, str]] = None,
        node_id: Optional[Union[int, str]] = None,
        acting_player: Optional[str] = None,
        next_player: Optional[str] = None,
        depth: Optional[float] = None,
        state: Optional[VidurMCTSState] = None,
        action: Optional[Union[AdversaryAction, ControllerAction]] = None,
    ) -> Dict[str, Any]:
        assert state is not None
        snapshot = self._env.describe_state(state)
        violations = snapshot["slo_violations"]
        avg_lateness = snapshot["avg_lateness"]
        cost = _compute_objective_cost(violations, avg_lateness)
        adversary_action = (action if acting_player == "adversary" else None)
        controller_action = (action if acting_player == "controller" else None)
        player_label = acting_player or ""
        next_label = next_player or player_label
        depth_value = depth if depth is not None else 0.0
        if isinstance(controller_action, ControllerAction):
            ctrl_heuristic = controller_action.heuristic or ""
            ctrl_strategy = controller_action.strategy or ""
        else:
            ctrl_heuristic = ""
            ctrl_strategy = ""
        
        row = {
            "iteration": iteration,
            "phase": phase,
            "depth": depth_value,
            "parent_node_id": "" if parent_id is None else str(parent_id),
            "node_id": str(node_id or "root"),
            "player_to_act": player_label,
            "next_player": next_label,
            "sim_time": snapshot["sim_time"],
            "requests_in_system": snapshot["requests_in_system"],
            "requests_generated": snapshot["requests_generated"],
            "requests_completed": snapshot["requests_completed"],
            "slo_violations": snapshot["slo_violations"],
            "avg_lateness": snapshot["avg_lateness"],
            "objective_cost": cost,
            "state_waiting_ids": json.dumps(snapshot["waiting_request_ids"]),
            "state_completed_request_ids": json.dumps(snapshot["completed_request_ids"]),
            "adversary_requests": json.dumps([
                {
                    "prefill_tokens": spec.prefill_tokens,
                    "decode_tokens": spec.decode_tokens,
                    "prefill_slo": spec.prefill_slo,
                    "decode_slo": spec.decode_slo,
                }
                for spec in action.requests
            ]) if isinstance(action, AdversaryAction) else json.dumps([]),
            "adversary_prefill_slos": json.dumps([
                spec.prefill_slo for spec in action.requests
            ]) if isinstance(action, AdversaryAction) else json.dumps([]),
            "adversary_decode_slos": json.dumps([
                spec.decode_slo for spec in action.requests
            ]) if isinstance(action, AdversaryAction) else json.dumps([]),
            "controller_token_budget": action.token_budget if isinstance(action, ControllerAction) else "",
            "controller_selected_ids": json.dumps(action.selected_request_ids or []) if isinstance(action, ControllerAction) else json.dumps([]),
            "controller_allocations": json.dumps(action.token_allocations) if isinstance(action, ControllerAction) else json.dumps({}),
            "controller_prefill_allocations": json.dumps(action.prefill_allocations) if isinstance(action, ControllerAction) else json.dumps({}),
            "controller_decode_allocations": json.dumps(action.decode_allocations) if isinstance(action, ControllerAction) else json.dumps({}),
            "controller_prefill_total": sum(action.prefill_allocations.values()) if isinstance(action, ControllerAction) else 0,
            "controller_decode_total": sum(action.decode_allocations.values()) if isinstance(action, ControllerAction) else 0,
            "controller_heuristic": ctrl_heuristic, 
            "controller_strategy": ctrl_strategy,     
        }
        return row

    def _dump_tree_snapshot(self, iteration: int) -> None:
        if self._tree_logger is None or self._root is None:
            return

        # BFS over the current tree
        queue = [self._root]
        while queue:
            node = queue.pop(0)
            for edge in node.children:
                queue.append(edge.node)

            parent = node.parent
            parent_id = "" if parent is None else parent.node_id
            depth = node.depth
            player_to_act = node.player
            action = node.parent_action  # action that led to this node

            # Pure tree statistics (no simulator replay)
            visits = node.visits
            parent_visits = parent.visits if parent is not None else 0
            mean_cost = (node.cumulative_cost / visits) if visits > 0 else 0.0

            # UCB score relative to parent (same formula as _best_child)
            if parent is not None and visits > 0 and parent_visits > 0:
                ln_parent = math.log(max(1, parent_visits))
                exploit = -mean_cost if parent.player == "controller" else mean_cost
                explore = self._cfg.exploration_constant * math.sqrt(
                    ln_parent / max(1, visits)
                )
                ucb = exploit + explore
            else:
                ucb = 0.0

            # Simple action representation
            if action is None:
                action_type = ""
                action_repr = ""
            else:
                # from vidur.mcts.environment import AdversaryAction, ControllerAction  # adjust import if needed
                if isinstance(action, AdversaryAction):
                    action_type = "adversary"
                elif isinstance(action, ControllerAction):
                    action_type = "controller"
                else:
                    action_type = type(action).__name__
                action_repr = repr(action)

            self._tree_logger.write_row(
                {
                    "iteration": iteration,
                    "node_id": node.node_id,
                    "parent_node_id": parent_id,
                    "depth": depth,
                    "player_to_act": player_to_act,
                    "action_type": action_type,
                    "visits": visits,
                    "parent_visits": parent_visits,
                    "node_cost": mean_cost,
                    "ucb_score": ucb,
                    "action_repr": action_repr,
                }
            )




    def _next_node_id(self) -> int:
        node_id = self._node_counter
        self._node_counter += 1
        return node_id
