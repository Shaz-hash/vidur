# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
DNN self-play runner.

No configuration lives here. alphaZero.py builds:
- Simulator + Env
- DNN model
- MCTS (mctsDNN)
- ReplayWriter

Then calls SelfPlayRunner.run_single_root(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import time  # add at file top if missing

from ..environment import VidurMCTSEnvironment, VidurMCTSState
from ..mctsDNN import VidurMCTS
from .infer import build_model_inputs
from .types import ModelInputs
from .replay_write import ReplayWriter, make_root_sample


def _mask_to_list(mask) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


def _compute_mcts_prior_from_root(root, mask: Sequence[bool]) -> Tuple[list[float], int]:
    a = len(mask)

    # Canonical visit totals from the actual MCTS tree
    canon_visits: dict[int, int] = {}
    for idx, child in root.children.items():
        ii = int(idx)
        if 0 <= ii < a:
            canon_visits[ii] = int(child.visits)

    # Best action should be picked by canonical totals (NOT split totals)
    valid_canon = [i for i in canon_visits.keys() if mask[i]]
    if valid_canon:
        best_idx = max(valid_canon, key=lambda i: canon_visits[i])
    else:
        best_idx = next((i for i, ok in enumerate(mask) if ok), 0)

    # Build policy target counts over FULL action space
    counts = [0.0] * a

    canon_to_aliases = getattr(root, "canonical_to_action_aliases", None)

    if canon_to_aliases:
        # Distribute each canonical child’s visits across its alias indices (uniform split)
        for canon, aliases in canon_to_aliases.items():
            canon = int(canon)
            v = float(canon_visits.get(canon, 0))
            valid_aliases = [int(i) for i in aliases if 0 <= int(i) < a and mask[int(i)]]
            if not valid_aliases:
                continue
            share = v / float(len(valid_aliases))
            for i in valid_aliases:
                counts[i] += share
    else:
        # No dedupe: each index is its own action
        for i, v in canon_visits.items():
            if mask[i]:
                counts[i] = float(v)

    total = sum(counts[i] for i, ok in enumerate(mask) if ok)
    if total > 0:
        prior = [(counts[i] / total) if mask[i] else 0.0 for i in range(a)]
    else:
        valid = [i for i, ok in enumerate(mask) if ok]
        prior = [0.0] * a
        if valid:
            p = 1.0 / len(valid)
            for i in valid:
                prior[i] = p

    return prior, int(best_idx)



@dataclass(frozen=True)
class SingleRootRun:
    game_id: int = 0
    root_id: int = 0
    root_depth: int = 0
    root_player: str = "adversary"  # "adversary" or "controller"
    iterations: int = 5000
    feature_version: int = 1


class SelfPlayRunner:
    def __init__(
        self,
        *,
        env: VidurMCTSEnvironment,
        mcts: VidurMCTS,
        model,
        writer: ReplayWriter,
        device_for_features: torch.device = torch.device("cpu"),
    ) -> None:
        self.env = env
        self.mcts = mcts
        self.model = model
        self.writer = writer
        self.device = device_for_features



    def _advance_to_branching_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        max_hops: int = 10000,
    ) -> tuple[VidurMCTSState, str, int]:
        """
        Advance the real self-play state through forced moves until the current player
        has >1 *unique* actions (controller uniqueness uses the same key as MCTS dedupe).

        Returns: (state, player_to_act, updated_depth)
        """
        for _ in range(int(max_hops)):
           
            if player == "controller":
                actions_by_index, mask = self.env.sample_controller_actions(state, self.mcts._cfg.max_branching)
                mask_list = _mask_to_list(mask)
                valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]

                if len(valid) != 1:
                    return state, player, depth  # 0 (terminal) or >1 (branching)

                forced_idx = valid[0]
                action = actions_by_index[forced_idx]
                assert action is not None
                state = self.env.apply_controller_action_only(state, action, inplace=True)
                player = "adversary"
                depth += 1
                continue

            # player == "adversary"
            actions_by_index, mask = self.env.sample_adversary_actions(
                state, self.mcts._cfg.max_branching
            )
            mask_list = _mask_to_list(mask)
            valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
            if not valid:
                return state, player, depth

            if len(valid) > 1:
                return state, player, depth  # branching root

            # forced: apply the only valid adversary action and continue
            forced_idx = valid[0]
            action = actions_by_index[forced_idx]
            assert action is not None
            state = self.env.apply_adversary_action_only(state, action, inplace=True)
            player = "controller"
            depth += 1

        return state, player, depth




    def run_single_root(self, cfg: SingleRootRun, root_state: Optional[VidurMCTSState] = None) -> None:
        state = root_state or self.env.initial_state()

        t0 = time.perf_counter()
        snap0 = self.env.describe_state(state)
        # run MCTS on a fork so it cannot mutate the selfplay root state
        #TODO : Remove this bool variable later 
        use_fork_logging = True
        search_state = state.fork(flag =use_fork_logging)
        t1 = time.perf_counter()
        # Run MCTS search (will also write MCTS CSV logs if enabled inside mcts)
        self.mcts.search_dnn(
            dnn_model=self.model,
            rootState=search_state,
            root_player=cfg.root_player,
            iterations=cfg.iterations,
            game_id=cfg.game_id,
            root_id=cfg.root_id,
            root_depth=cfg.root_depth,
        )
        t2 = time.perf_counter()
        # confirm real root state was NOT mutated by search_dnn
        snap_after = self.env.describe_state(state)
        print(
            f"[run_single_root_search_DNN:after_search] dt={t2 - t1:.3f}s "
        )

        # Mask from env for this root/player (fixed action indexing)
        if cfg.root_player == "controller":
            _, mask = self.env.sample_controller_actions(state, self.mcts._cfg.max_branching)
        else:
            _, mask = self.env.sample_adversary_actions(state, self.mcts._cfg.max_branching)
        mask_list = _mask_to_list(mask)

        # MCTS targets from the built root
        root = self.mcts._root
        mcts_prior, best_idx = _compute_mcts_prior_from_root(root, mask_list)
        mcts_value = float(root.mean_value())

        # Build model inputs (CPU) and force action_mask to env mask
        base_inputs = build_model_inputs(state, cfg.root_player, self.device)
        inputs = ModelInputs(
            req_features=base_inputs.req_features,
            global_features=base_inputs.global_features,
            req_mask=base_inputs.req_mask,
            action_mask=torch.tensor(mask_list, dtype=torch.bool, device=self.device).unsqueeze(0),
        )

        # Write one root sample into dataset shards
        sample = make_root_sample(
            feature_version=cfg.feature_version,
            game_id=cfg.game_id,
            root_id=cfg.root_id,
            root_node_id=int(root.node_id),
            root_depth=int(cfg.root_depth),
            player=cfg.root_player,
            model_inputs=inputs,
            action_mask=mask_list,
            mcts_policy=mcts_prior,
            mcts_value_controller=mcts_value,
            meta={
                "best_action_index": int(best_idx),
                "num_simulations": int(cfg.iterations),
            },
        )
        self.writer.add(sample)
        t3 = time.perf_counter()
        print(
            f"[run_single_root_search_DNN:end] total_dt={t3 - t0:.6f}s "
            f"fork_dt={t1 - t0:.6f}s "
            f"search_dt={t2 - t1:.6f}s encode/write_dt={t3 - t2:.6f}s"
        )

    ## TODO: ENSURE THAT MINIMAX IS RESET AGAIN & THE NEXT NODE IS ALWAYS THE NODE WHERE NN CAN BE CALLED AGAIN & WHY ARE THE ROOT ITEREATIONS LOGS CREATED AGAIN...
    def run_n_roots(
        self,
        *,
        game_id: int,
        num_roots: int,
        # iterations_per_root: int = 5000,
        adv_iterations_per_root: int = 1000,
        cont_iterations_per_root: int = 500,
        max_batch_size: int = 72,
        start_root_id: int = 0,
        start_root_depth: int = 0,
        start_player: str = "adversary",
        feature_version: int = 1,
        initial_state: Optional[VidurMCTSState] = None,
    ) -> VidurMCTSState:
        # state = initial_state or self.env.initial_state()
        # player = start_player

        state = initial_state or self.env.initial_state()
        player = start_player
        depth = int(start_root_depth)

        for k in range(int(num_roots)):
            root_id = start_root_id + k
            # root_depth = start_root_depth + k

            # force-advance until branching before running MCTS
            state, player, depth = self._advance_to_branching_root(state, player, depth)

            ## TODO : This terminal condition needs to be later avoided. 
            # NEW: terminal cutoff for dataset collection
            requests_in_system = int(self.env.describe_state(state).get("requests_in_system", 0))
            if requests_in_system > int(max_batch_size):
                print(
                    f"[SelfPlayRunner] stop: requests_in_system={requests_in_system} "
                    f"> max_batch_size={max_batch_size}"
                )
                return state

            # Determine iterations per root based on player to act
            iters = int(adv_iterations_per_root if player == "adversary" else cont_iterations_per_root)
            
            # 1) Run one root search + write one dataset sample
            self.run_single_root(
                SingleRootRun(
                    game_id=game_id,
                    root_id=root_id,
                    root_depth=depth,
                    root_player=player,
                    iterations=iters,
                    feature_version=feature_version,
                ),
                root_state=state,
            )

            # 2) Choose best action index from visit counts (greedy argmax)
            if player == "controller":
                actions_by_index, mask = self.env.sample_controller_actions(state, self.mcts._cfg.max_branching)
            else:
                actions_by_index, mask = self.env.sample_adversary_actions(state, self.mcts._cfg.max_branching)

            mask_list = _mask_to_list(mask)

            root = self.mcts._root
            _, best_idx = _compute_mcts_prior_from_root(root, mask_list)

            if not (0 <= best_idx < len(actions_by_index)):
                raise RuntimeError(f"best_idx={best_idx} out of range for actions_by_index length={len(actions_by_index)}")
            if not mask_list[best_idx]:
                raise RuntimeError(f"best_idx={best_idx} is not valid per mask")
            action = actions_by_index[best_idx]
            if action is None:
                raise RuntimeError(f"actions_by_index[{best_idx}] is None even though mask is True")

            # 3) Advance simulator state in-place
            if player == "adversary":
                state = self.env.apply_adversary_action_only(state, action, inplace=True)
                player = "controller"
            else:
                state = self.env.apply_controller_action_only(state, action, inplace=True)
                player = "adversary"
            depth += 1

        return state
