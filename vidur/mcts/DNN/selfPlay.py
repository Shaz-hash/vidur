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
import time  

# from ..environment import VidurMCTSEnvironment, VidurMCTSState
from ..environment import VidurMCTSEnvironment, VidurMCTSState, AdversaryAction

from ..mctsDNN import VidurMCTS
from .history_root import HistoryRootGenerator
from .infer import build_model_inputs
from .types import ModelInputs
from .replay_write import ReplayWriter, make_root_sample
import math 
import random


def _mask_to_list(mask) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


def _sample_action_from_mcts_policy(
    *,
    prior: Sequence[float],
    mask: Sequence[bool],
    temperature: float,
    seed: int,
) -> int:
    valid = [i for i, ok in enumerate(mask) if ok]
    if not valid:
        return -1

    if temperature <= 1e-8:
        return max(valid, key=lambda i: float(prior[i]))

    inv_t = 1.0 / float(temperature)
    weights = [(max(float(prior[i]), 0.0) + 1e-12) ** inv_t for i in valid]
    if sum(weights) <= 0.0:
        weights = [1.0] * len(valid)

    rng = random.Random(int(seed) & 0xFFFFFFFF)
    pick = rng.choices(range(len(valid)), weights=weights, k=1)[0]
    return int(valid[pick])


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
    root_node_id_override: int | None = None


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
        #self.history = HistoryRootGenerator(env=self.env , max_branching = self.mcts._cfg.max_branching, iter_logger = getattr(self.mcts, '_iter_logger', None))
        self.history = HistoryRootGenerator(
            env=self.env,
            max_branching=self.mcts._cfg.max_branching,
            iter_logger=getattr(self.mcts, "_iter_logger", None),
            root_logger=getattr(self.mcts, "_root_logger", None),
        )


    def _advance_to_branching_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        max_hops: int = 1024,
    ) -> tuple[VidurMCTSState, str, int]:
        """
        Advance the real self-play state through forced moves until the current player
        has >1 *unique* actions (controller uniqueness uses the same key as MCTS dedupe).

        Returns: (state, player_to_act, updated_depth)
        """
        max_hops_i = max(1, int(max_hops))
        for _ in range(max_hops_i):
           
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

        print(
            f"[SelfPlayRunner] advance cap reached: max_hops={max_hops_i} "
            f"depth={depth} player={player}; proceeding without guaranteed branching"
        )
        return state, player, depth




    def run_single_root(self, cfg: SingleRootRun, root_state: Optional[VidurMCTSState] = None  ) -> None:
        state = root_state or self.env.initial_state()

        t0 = time.perf_counter()
        snap0 = self.env.describe_state(state)
        # run MCTS on a fork so it cannot mutate the selfplay root state
        #TODO : Remove this bool variable later 
        use_fork_logging = False
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
            root_node_id_override=cfg.root_node_id_override,
            root_depth=cfg.root_depth,
        )
        t2 = time.perf_counter()
        # confirm real root state was NOT mutated by search_dnn
        # TODO: Why are we doing this even ? should be removing it 
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

    # TODO : seems like start_root_depth is unncessary & we might not even need to use the best action if run_single_root is doing it 
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
        history_nontrivial_hops: int = 0,
        feature_version: int = 1,
        initial_state: Optional[VidurMCTSState] = None,
        history_seed: Optional[int] = None,
        sample_from_mcts_policy: bool = False,
        selfplay_policy_temperature: float = 0.0,
        action_seed_base: int = 0,
        max_forced_hops_per_root: int = 1024,
        history_max_total_steps: int = 20000,
    ) -> VidurMCTSState:
        # state = initial_state or self.env.initial_state()
        # player = start_player

        state = initial_state or self.env.initial_state()
        player = start_player
        depth = int(start_root_depth)
        next_log_node_id = int(getattr(self.mcts, "_node_counter", 0))
        last_log_node_id: int | None = None


        if int(history_nontrivial_hops) > 0:
            state, player, depth, next_log_node_id, last_log_node_id = self.history.generate_history_root(
                state,
                player,
                depth,
                nontrivial_hops=int(history_nontrivial_hops),
                game_id=int(game_id),
                root_id_for_logs=int(start_root_id),  # history rows attach to the first root
                log_history=True,
                log_node_id_start=next_log_node_id,
                log_parent_id_start=last_log_node_id,
                seed=history_seed,
                max_total_steps=int(history_max_total_steps),
            )
            # IMPORTANT: prevent MCTS from reusing history node ids
            self.mcts._node_counter = int(next_log_node_id)

        warned_overflow_soft = False

        for k in range(int(num_roots)):
            root_node_id_override = int(last_log_node_id) if (k == 0 and last_log_node_id is not None) else None
            root_id = start_root_id + k
            # root_depth = start_root_depth + k

            # force-advance until branching before running MCTS
            state, player, depth = self._advance_to_branching_root(
                state,
                player,
                depth,
                max_hops=int(max_forced_hops_per_root),
            )

            requests_in_system = int(self.env.describe_state(state).get("requests_in_system", 0))
            if requests_in_system > int(max_batch_size):
                if not warned_overflow_soft:
                    print(
                        f"[SelfPlayRunner] soft_overflow: requests_in_system={requests_in_system} "
                        f"> max_batch_size={max_batch_size}; continuing search "
                        f"(game_id={game_id}, root_id={root_id})",
                        flush=True,
                    )
                    warned_overflow_soft = True

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
                    root_node_id_override=root_node_id_override,
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
            mcts_prior, best_idx = _compute_mcts_prior_from_root(root, mask_list)


            ## Sampling done using MCTS generated Prior if set in the configuration 
            if sample_from_mcts_policy and player == "controller":
                node_seed = (
                    int(action_seed_base) * 1000003 + int(game_id) * 9176 + int(root_id) * 37 + int(depth) * 13
                )
                sampled_idx = _sample_action_from_mcts_policy(
                    prior=mcts_prior,
                    mask=mask_list,
                    temperature=float(selfplay_policy_temperature),
                    seed=int(node_seed),
                )
                if 0 <= sampled_idx < len(actions_by_index) and actions_by_index[sampled_idx] is not None:
                    best_idx = sampled_idx


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

            if hasattr(self.mcts, "clear_search_state"):
                self.mcts.clear_search_state(drop_scratch=True)
        return state



    ### ARENA GAME FUNCTIONS FOR EVAL
    def _has_prefill_pending(self, state: VidurMCTSState) -> bool:
        reqs = self.env._build_request_lookup(state.simulator).values()
        for req in reqs:
            if not bool(getattr(req, "is_prefill_complete", False)):
                return True
        return False


    def _pick_mcts_action_for_arena_step(
        self,
        *,
        state: VidurMCTSState,
        player: str,
        model,
        game_id: int,
        root_id: int,
        root_depth: int,
        iterations: int,
        feature_version: int,
        cycle_label: str,
        prefer_nonempty_adversary: bool,
    ):
        # search on fork so arena state is not mutated by search
        search_state = state.fork(flag=False)

        self.mcts.search_dnn(
            dnn_model=model,
            rootState=search_state,
            root_player=player,
            iterations=int(iterations),
            game_id=int(game_id),
            root_id=int(root_id),
            root_depth=int(root_depth),
            root_node_id_override=None,
            root_phase="arena_root",
            cycle_label=str(cycle_label),
        )

        try:
            if player == "controller":
                actions_by_index, mask = self.env.sample_controller_actions(state, self.mcts._cfg.max_branching)
            else:
                actions_by_index, mask = self.env.sample_adversary_actions(state, self.mcts._cfg.max_branching)

            mask_list = _mask_to_list(mask)
            valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
            if not valid:
                return None, -1

            candidate_valid = list(valid)
            if player == "adversary" and prefer_nonempty_adversary:
                nonempty = [
                    i for i in valid
                    if isinstance(actions_by_index[i], AdversaryAction)
                    and len((actions_by_index[i].requests or [])) > 0
                ]
                if nonempty:
                    candidate_valid = nonempty

            root = self.mcts._root
            mcts_prior, best_idx = _compute_mcts_prior_from_root(root, mask_list)

            if best_idx not in candidate_valid:
                best_idx = max(candidate_valid, key=lambda i: (float(mcts_prior[i]), -int(i)))

            action = actions_by_index[int(best_idx)]
            return action, int(best_idx)
        finally:
            if hasattr(self.mcts, "clear_search_state"):
                self.mcts.clear_search_state(drop_scratch=True)


    def _run_arena_cycle(
        self,
        *,
        base_snapshot,
        base_stats,
        base_player: str,
        base_depth: int,
        game_id: int,
        root_id_base: int,
        adversary_model,
        controller_model,
        cycle_label: str,
        adv_iterations_per_root: int,
        cont_iterations_per_root: int,
        arena_max_adversary_moves: int,
        arena_max_controller_cleanup_steps: int,
        arena_max_total_turns: int,
        arena_max_adversary_total_turns: int,
        feature_version: int,
    ) -> dict:
        state = self.env.clone_state_from_snapshot(base_snapshot, base_stats)
        player = str(base_player)
        depth = int(base_depth)

        turns = 0
        cleanup_steps = 0
        adv_moves_total = 0
        adv_request_moves = 0

        while (
            turns < int(arena_max_total_turns)
            and adv_request_moves < int(arena_max_adversary_moves)
            # and adv_moves_total < int(arena_max_adversary_total_turns)
        ):
            model = adversary_model if player == "adversary" else controller_model
            iters = int(adv_iterations_per_root if player == "adversary" else cont_iterations_per_root)

            action, _ = self._pick_mcts_action_for_arena_step(
                state=state,
                player=player,
                model=model,
                game_id=int(game_id),
                root_id=int(root_id_base + turns),
                root_depth=int(depth),
                iterations=iters,
                feature_version=int(feature_version),
                cycle_label=str(cycle_label),
                prefer_nonempty_adversary=True,
            )
            if action is None:
                break

            if player == "adversary":
                generated = len((action.requests or [])) if isinstance(action, AdversaryAction) else 0
                state = self.env.apply_adversary_action_only(state, action, inplace=True)
                player = "controller"
                
                if generated > 0:
                    adv_request_moves += 1
                    adv_moves_total += 1
            else:
                state = self.env.apply_controller_action_only(state, action, inplace=True)
                player = "adversary"

            depth += 1
            turns += 1

        while (
            turns < int(arena_max_total_turns)
            and cleanup_steps < int(arena_max_controller_cleanup_steps)
            and self._has_prefill_pending(state)
        ):
            if player == "adversary":
                state = self.env.apply_adversary_action_only(
                    state,
                    AdversaryAction(requests=[], stop_decode_ids=[]),
                    inplace=True,
                )
                player = "controller"
                depth += 1
                turns += 1
                continue

            action, _ = self._pick_mcts_action_for_arena_step(
                state=state,
                player="controller",
                model=controller_model,
                game_id=int(game_id),
                root_id=int(root_id_base + turns),
                root_depth=int(depth),
                iterations=int(cont_iterations_per_root),
                feature_version=int(feature_version),
                cycle_label=str(cycle_label),
                prefer_nonempty_adversary=False,
            )
            if action is None:
                break

            state = self.env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"
            depth += 1
            turns += 1
            cleanup_steps += 1

        viol, lateness = self.env.evaluate_objective(state)
        total_cost = float(viol) + float(lateness)

        # explicit end marker row in root log for easy parser grading
        if self.mcts._root_logger is not None:
            self.mcts._root_logger.log_root(
                game_id=int(game_id),
                root_id=int(root_id_base + turns + 1),
                root_depth=int(depth),
                root_node_id=-1,
                root_player="",
                num_simulations=0,
                model_root_value_controller=0.0,
                model_root_prior=[],
                normalized_root_prior=[],
                valid_action_mask=[],
                mcts_root_value_controller=0.0,
                mcts_root_prior=[],
                best_action_index=None,
                best_action_repr="",
                best_action_json="",
                phase="arena_end",
                cycle_label=str(cycle_label),
                sim_time=float(state.simulator._time),
                slo_violations=int(viol),
                total_lateness=float(lateness),
                total_cost=float(total_cost),
            )

        return {
            "total_cost": float(total_cost),
            "slo_violations": int(viol),
            "total_lateness": float(lateness),
            "turns": int(turns),
            "cleanup_steps": int(cleanup_steps),
            "adversary_moves_total": int(adv_moves_total),
            "adversary_request_moves": int(adv_request_moves),
        }


    def run_arena_game(
        self,
        *,
        game_id: int,
        candidate_model,
        best_model,
        history_nontrivial_hops: int,
        adv_iterations_per_root: int,
        cont_iterations_per_root: int,
        arena_max_adversary_moves: int,
        arena_max_controller_cleanup_steps: int,
        arena_max_total_turns: int,
        feature_version: int = 1,
        tie_points: float = 0.5,
        start_player: str = "adversary",
        start_root_depth: int = 0,
        history_root_id_for_logs: int = 0,
    ) -> dict:
        state = self.env.initial_state()
        player = str(start_player)
        depth = int(start_root_depth)
        next_log_node_id = int(getattr(self.mcts, "_node_counter", 0))
        last_log_node_id = None

        if int(history_nontrivial_hops) > 0:
            state, player, depth, next_log_node_id, last_log_node_id = self.history.generate_history_root(
                state,
                player,
                depth,
                nontrivial_hops=int(history_nontrivial_hops),
                game_id=int(game_id),
                root_id_for_logs=int(history_root_id_for_logs),
                log_history=True,
                log_node_id_start=next_log_node_id,
                log_parent_id_start=last_log_node_id,
            )
            self.mcts._node_counter = int(next_log_node_id)

        base_snapshot = state.simulator.snapshot_state()
        base_stats = state.stats.clone()

        max_adv_total = max(1, 4 * int(arena_max_adversary_moves))

        cycle_a = self._run_arena_cycle(
            base_snapshot=base_snapshot,
            base_stats=base_stats,
            base_player=player,
            base_depth=depth,
            game_id=int(game_id),
            root_id_base=0,
            adversary_model=candidate_model,
            controller_model=best_model,
            cycle_label="candidate_as_adversary",
            adv_iterations_per_root=int(adv_iterations_per_root),
            cont_iterations_per_root=int(cont_iterations_per_root),
            arena_max_adversary_moves=int(arena_max_adversary_moves),
            arena_max_controller_cleanup_steps=int(arena_max_controller_cleanup_steps),
            arena_max_total_turns=int(arena_max_total_turns),
            arena_max_adversary_total_turns=int(max_adv_total),
            feature_version=int(feature_version),
        )

        cycle_b = self._run_arena_cycle(
            base_snapshot=base_snapshot,
            base_stats=base_stats,
            base_player=player,
            base_depth=depth,
            game_id=int(game_id),
            root_id_base=1_000_000,
            adversary_model=best_model,
            controller_model=candidate_model,
            cycle_label="best_as_adversary",
            adv_iterations_per_root=int(adv_iterations_per_root),
            cont_iterations_per_root=int(cont_iterations_per_root),
            arena_max_adversary_moves=int(arena_max_adversary_moves),
            arena_max_controller_cleanup_steps=int(arena_max_controller_cleanup_steps),
            arena_max_total_turns=int(arena_max_total_turns),
            arena_max_adversary_total_turns=int(max_adv_total),
            feature_version=int(feature_version),
        )

        ca = float(cycle_a["total_cost"])
        cb = float(cycle_b["total_cost"])
        if ca > cb + 1e-9:
            cp, bp, winner = 1.0, 0.0, "candidate"
        elif cb > ca + 1e-9:
            cp, bp, winner = 0.0, 1.0, "best"
        else:
            cp, bp, winner = float(tie_points), float(tie_points), "tie"

        return {
            "game_id": int(game_id),
            "candidate_as_adv_cost": ca,
            "best_as_adv_cost": cb,
            "candidate_points": float(cp),
            "best_points": float(bp),
            "winner": str(winner),
            "cycle_a": cycle_a,
            "cycle_b": cycle_b,
        }
