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
        "sim_time",
        "requests_in_system",
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
        # ## Pre Computing the necessary states here 
        # max_budget = self._env._max_request_tokens_allowed()
        # self._env.precompute_controller_state_space(token_budget=max_budget)

    def search(self, iterations: int) -> ControllerAction:
        root = self._create_root()
        try:
            self._root = root
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


            # --- Random history prefix (depth in plies) ---
            current = root
            for step in range(self._history_depth):
                # Ensure we have actions to choose from
                if not current.untried_actions:
                    current.untried_actions = self._enumerate_actions(current)
                    if not current.untried_actions:
                        break

                # Pick a random action from this node
                action = self._rng.choice(current.untried_actions)
                current.untried_actions.remove(action)

                if current.player == "adversary":
                    child_state = self._env.apply_adversary_action_only(current.state, action)
                    next_player = "controller"
                else:
                    child_state = self._env.apply_controller_action_only(current.state, action)
                    next_player = "adversary"

                child = MCTSNode(
                    state=child_state,
                    player=next_player,
                    node_id=self._next_node_id(),
                    depth=current.depth + 1,
                    parent=current,
                    parent_action=action,
                )
                child.untried_actions = self._enumerate_actions(child)
                current.children.append(MCTSChildEdge(action=action, node=child))

                # Log these as part of iteration 0 (history)
                self._log_state(
                    iteration=0,
                    phase="INITIAL_HISTORY_GEN",
                    node=child,
                    parent_id=current.node_id,
                    acting_player=current.player,
                    next_player=child.player,
                    action=action,
                )

                current = child

            # Use the last node of the history as the root for MCTS
            root_for_search = current
            self._root = root_for_search
            root = self._root
            ## --------- RANDOM HISTORY ENDS HERE -----------

            ## -------- VANILLA MCTS STARTS HERE -----------
            for itr in range(iterations):
                # EARLY EXIT: all controller states visited at least once
                if self._env.controller_states_fully_visited():
                    if self._verbose:
                        print("Stopping MCTS: all controller states visited >= 1")
                    break

                self._current_iteration = itr + 1
                node = self._select(root)
                expanded = self._expand(node)
                leaf = expanded or node
                cost = self._simulate(leaf)

                self._backpropagate(leaf, cost)

                # Periodically dump the entire tree to the tree CSV
                if (
                    self._tree_dump_interval > 0
                    and (itr + 1) % self._tree_dump_interval == 0
                ):
                    print("PRINTING THE STATE INTO THE FOR THE TREE CSV!")
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
     
            # print(f"[PROFILE] apply_adversary_actions: {t1 - t0:.4f}s")
        else:
            # Controller applies the action → compute cost delta and update metrics
            child_state = self._env.apply_controller_action_only(node.state, action)
            next_player = "adversary"

            # parent cost
            parent_snapshot = self._env.describe_state(node.state)
            parent_cost = _compute_objective_cost(
                parent_snapshot["slo_violations"],
                parent_snapshot["avg_lateness"],
            )

            # child cost (this is exactly what the tree log will see)
            child_snapshot = self._env.describe_state(child_state)
            violations = child_snapshot["slo_violations"]
            avg_lateness = child_snapshot["avg_lateness"]
            cost = _compute_objective_cost(violations, avg_lateness)
            delta_cost = cost - parent_cost

            # update per-controller-state statistics
            self._env.update_controller_state_metrics(
                action,
                cost,
                delta_cost,
                violations,
                avg_lateness,
            )

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
   
        #print(f" Expansion time : {t2 - t0:.4f}s")
        return child

    def _simulate(self, node: MCTSNode) -> float:
        total_cost = 0.0
        num_trials = max(1, self._cfg.simulation_random_tries)

        rollout_batch_rows: List[Dict[str, Any]] = []
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

                    if action is not None:
                        if self._logger.enabled_for("rollout"):
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

            violations, avg_lateness = self._env.evaluate_objective(rollout_state)
            total_cost += _compute_objective_cost(violations, avg_lateness)

        # Commit rollout rows in a single batch for this simulate() call.
        if rollout_batch_rows:
            self._logger.write_rows(rollout_batch_rows)
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
        t0 = time.perf_counter()
        if node.player == "adversary":
            t1 = time.perf_counter()
            actions = self._env.sample_adversary_actions(
                node.state, self._cfg.max_branching
            )
            t2 = time.perf_counter()
            #print(f"[PROFILE] sample_adversary_actions: {t2 - t1:.4f}s")
        else:
            t1 = time.perf_counter()
            actions = self._env.sample_controller_actions(
                node.state, self._cfg.max_branching
            )
            t2 = time.perf_counter()
            #print(f"[PROFILE] sample_controller_actions: {t2 - t1:.4f}s")
        self._rng.shuffle(actions)
        t3 = time.perf_counter()
        #print(f"[PROFILE] _enumerate_actions total ({node.player}): {t3 - t0:.4f}s")
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
       
        # New: tree-only CSV (no rollouts)
        # if phase in ("root", "tree") and self._tree_logger is not None:
        #     row = self._make_log_row(
        #         iteration=iteration,
        #         phase=phase,
        #         parent_id=parent_id,
        #         node_id=node_id or "root",
        #         acting_player=player_label,
        #         next_player=next_label,
        #         depth=depth_value,
        #         state=state,
        #         action=action,
        #     )
        #     self._tree_logger.write_row(row)
    
        # print(
        #     f"[PROFILE] log_state phase={phase}: "
        #     f"describe_state Snap_shot={t1 - t0:.4f}s, main_log={t2 - t1:.4f}s, "
        #     f"tree_log={t3 - t2:.4f}s"
        # )
        # New: per-controller-state metrics with delta cost
        # if isinstance(controller_action, ControllerAction) and phase in ("root", "tree"):
        #     # Compute parent cost if there is a parent node
        #     if node is not None and node.parent is not None:
        #         parent_snapshot = self._env.describe_state(node.parent.state)
        #         parent_cost = _compute_objective_cost(
        #             parent_snapshot["slo_violations"],
        #             parent_snapshot["avg_lateness"],
        #         )
        #     else:
        #         parent_cost = 0.0  # treat root as baseline
        #     delta_cost = cost - parent_cost

        #     self._env.update_controller_state_metrics(
        #         controller_action,
        #         cost,
        #         delta_cost,
        #         violations,
        #         avg_lateness,
        #     )

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

            # Describe current simulator state at this node
            snapshot = self._env.describe_state(node.state)
            sim_time = snapshot["sim_time"]
            num_requests = snapshot["requests_in_system"]

            # Cost estimate at this node (mean MCTS cost)
            mean_cost = (
                node.cumulative_cost / node.visits if node.visits > 0 else 0.0
            )

            # UCB score relative to parent (same formula as _best_child)
            if parent is not None and node.visits > 0:
                parent_visits = max(1, parent.visits)
                ln_parent = math.log(parent_visits)
                exploit = -mean_cost if parent.player == "controller" else mean_cost
                explore = self._cfg.exploration_constant * math.sqrt(
                    ln_parent / max(1, node.visits)
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
                    "sim_time": sim_time,
                    "requests_in_system": num_requests,
                    "node_cost": mean_cost,
                    "ucb_score": ucb,
                    "action_repr": action_repr,
                }
            )




    def _next_node_id(self) -> int:
        node_id = self._node_counter
        self._node_counter += 1
        return node_id

