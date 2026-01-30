# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import random
import json
from typing import Any, Optional, Tuple

import torch

from ..environment import VidurMCTSEnvironment, VidurMCTSState
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger

## TODO: Use this class later for the verification of a given trace
def _mask_to_list(mask) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


class HistoryRootGenerator:
    """
    Builds a "history root" state by advancing from a given state:
    - forced single-action chains are applied but do NOT count toward nontrivial_hops
    - branching steps (valid_actions > 1) pick a random valid action and DO count
    - returns a (state, player_to_act, depth) positioned at a branching (or terminal) node

    Optional: logs each applied step to the MCTS iteration logger with phase="history-...".
    """

    def __init__(
        self,
        *,
        env: VidurMCTSEnvironment,
        max_branching: int,
        iter_logger: Optional[DNNMCTSIterationLogger] = None,
    ) -> None:
        self.env = env
        self.max_branching = int(max_branching)
        self.iter_logger = iter_logger

    def _actions_valid(self, state: VidurMCTSState, player: str) -> tuple[list[Optional[object]], list[int], list[bool]]:
        if player == "controller":
            actions_by_index, mask = self.env.sample_controller_actions(state, self.max_branching)
        else:
            actions_by_index, mask = self.env.sample_adversary_actions(state, self.max_branching)

        mask_list = _mask_to_list(mask)
        valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
        return actions_by_index, valid, mask_list

    def _phase_label(self, player: str, action: object, n_valid: int) -> str:
        if int(n_valid) == 1:
            return "forced_step:single-child"
        if player == "adversary":
            reqs = getattr(action, "requests", None) or []
            stop_ids = getattr(action, "stop_decode_ids", None) or []
            if (not reqs) and (not stop_ids):
                return "trivial-multiple-child"
        return "multiple-child"

    def _apply(self, state: VidurMCTSState, player: str, action: object) -> tuple[VidurMCTSState, str]:
        if player == "adversary":
            state = self.env.apply_adversary_action_only(state, action, inplace=True)
            return state, "controller"
        state = self.env.apply_controller_action_only(state, action, inplace=True)
        return state, "adversary"

    def _log_step(
        self,
        *,
        game_id: int,
        root_id: int,
        root_depth: int,
        node_depth: int,
        node_id: int,
        parent_node_id: int | None,
        player_acted: str,
        next_player: str,
        action_index: int,
        action: object,
        n_valid: int,
        phase: str,
        state_after: VidurMCTSState,
    ) -> None:
        if self.iter_logger is None:
            print("returningg")
            return

        violations, lateness_sum = self.env.evaluate_objective(state_after)
        objective_cost = float(violations) + float(lateness_sum)
        snap = self.env.describe_state(state_after)


        adv_deadlines_json = "{}"
        if player_acted == "adversary":
            req_specs = getattr(action, "requests", None) or []
            if req_specs:
                lookup = self.env._build_request_lookup(state_after.simulator)
                if lookup:
                    k = len(req_specs)
                    new_ids = sorted(lookup.keys())[-k:]  # assumes newest have largest ids
                    out = {}
                    for rid in new_ids:
                        req = lookup.get(rid)
                        if req is None:
                            continue
                        queued_at = float(getattr(req, "queued_at", getattr(req, "arrived_at", 0.0)) or 0.0)
                        slo = float(getattr(req, "prefill_slo_time", 0.0) or 0.0)
                        out[int(rid)] = queued_at + slo
                    adv_deadlines_json = json.dumps(out)


        self.iter_logger.log_expand(
            game_id=int(game_id),
            root_id=int(root_id),
            sim_iteration=-1,  # history row
            root_depth=int(root_depth),
            root_node_id=0,
            root_player=str(player_acted),
            node_depth=int(node_depth),
            parent_node_id=None if parent_node_id is None else int(parent_node_id),
            node_id=int(node_id),
            player_to_act=str(next_player),
            player_acted_to_create_this_node=str(player_acted),
            action_index=int(action_index),
            action_repr=repr(action),
            prior=1.0,
            reward=0.0,
            nn_called=False,
            num_valid_actions=int(n_valid),
            unique_actions=int(n_valid),
            nn_value_controller=None,
            objective_cost=float(objective_cost),
            adversary_prefill_deadlines_by_id_json=adv_deadlines_json,
            state_snapshot=snap,
            phase=f"history-{phase}",
        )

    def advance_to_branching_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        game_id: int,
        root_id: int,
        log_node_id: int,
        log_parent_id: int | None,
        max_hops: int = 10000,
        log_steps: bool = False,
    ) -> tuple[VidurMCTSState, str, int, int, int | None]:
        """
        Apply forced single-action steps until valid_actions != 1.
        Returns (state, player_to_act, depth).
        """
        root_depth = int(depth)
        for _ in range(int(max_hops)):
            actions_by_index, valid, _mask = self._actions_valid(state, player)
            n_valid = len(valid)

            if n_valid != 1:
                return state, player, int(depth), int(log_node_id), log_parent_id

            idx = int(valid[0])
            action = actions_by_index[idx]
            assert action is not None

            acted = player
            state, next_player = self._apply(state, player, action)
            root_depth = int(depth)
            depth = int(depth) + 1

            if log_steps:
                self._log_step(
                    game_id=game_id,
                    root_id=root_id,
                    root_depth=root_depth,
                    node_depth=int(depth),
                    node_id=int(log_node_id),
                    parent_node_id=log_parent_id,
                    player_acted=acted,
                    next_player=next_player,
                    action_index=idx,
                    action=action,
                    n_valid=n_valid,
                    phase=self._phase_label(acted, action, n_valid),
                    state_after=state,
                )
                log_parent_id = int(log_node_id)
                log_node_id += 1

            player = next_player

        return state, player, int(depth), int(log_node_id), log_parent_id

    def generate_history_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        nontrivial_hops: int,
        game_id: int,
        root_id_for_logs: int,
        seed: Optional[int] = None,
        log_history: bool = True,
        max_total_steps: int = 200000,
        log_node_id_start: int,
        log_parent_id_start: int | None = None,

    ) -> tuple[VidurMCTSState, str, int, int, int | None]:
        """
        Take `nontrivial_hops` branching decisions (valid_actions > 1),
        skipping forced chains in between.
        where:
        - next_log_node_id = next free id (like a counter)
        - last_log_node_id = id of the last logged node (parent for the next step)
        """
        target = int(nontrivial_hops)
  
        log_node_id = int(log_node_id_start)
        log_parent_id: int | None = None if log_parent_id_start is None else int(log_parent_id_start)

        if target <= 0:
            # still make sure we're at a branching root if caller wants
            return state, player, int(depth)

        rng = random.Random(int(seed) if seed is not None else (1000003 * int(game_id) + int(root_id_for_logs)))

        
        steps = 0
        done = 0

        while done < target:
            # 1) skip forced chain (optionally log)
            state, player, depth, log_node_id, log_parent_id = self.advance_to_branching_root(
                state,
                player,
                depth,
                game_id=game_id,
                root_id=root_id_for_logs,
                max_hops=min(10000, int(max_total_steps)),
                log_node_id=log_node_id,
                log_parent_id=log_parent_id,
                log_steps=log_history,
            )
            steps += 1
            if steps >= int(max_total_steps):
                raise RuntimeError("history generator hit max_total_steps")

            # 2) branching / terminal check
            actions_by_index, valid, _mask = self._actions_valid(state, player)
            if not valid:
                return state, player, int(depth)  # terminal

            if len(valid) == 1:
                # should not happen because advance_to_branching_root would have consumed it
                print("Error in the creation of the root history, expected branching node but got single child")
                continue

            # 3) pick random branching action
            idx = int(rng.choice(valid))
            action = actions_by_index[idx]
            assert action is not None

            acted = player
            state, next_player = self._apply(state, player, action)
            root_depth = int(depth)
            depth = int(depth) + 1

            if log_history:
                self._log_step(
                    game_id=game_id,
                    root_id=root_id_for_logs,
                    root_depth=root_depth,
                    node_depth=int(depth),
                    node_id=int(log_node_id),
                    parent_node_id=log_parent_id,
                    player_acted=acted,
                    next_player=next_player,
                    action_index=idx,
                    action=action,
                    n_valid=len(valid),
                    phase=self._phase_label(acted, action, len(valid)),
                    state_after=state,
                )
                log_parent_id = int(log_node_id)
                log_node_id += 1
            player = next_player
            done += 1

        # finish by skipping any trailing forced chain so caller gets a branching root
        state, player, depth, log_node_id, log_parent_id = self.advance_to_branching_root(
            state,
            player,
            depth,
            game_id=game_id,
            root_id=root_id_for_logs,
            max_hops=min(10000, int(max_total_steps)),
            log_node_id=log_node_id,
            log_parent_id=log_parent_id,
            log_steps=log_history,
        )
        return state, player, int(depth), int(log_node_id), log_parent_id
