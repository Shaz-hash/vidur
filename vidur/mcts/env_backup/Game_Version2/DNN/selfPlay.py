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

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

import torch
import time  

# from ..environment import VidurMCTSEnvironment, VidurMCTSState
from ..config import GameVersion2Config
from ....environment import VidurMCTSState
from ....game_types import AdversaryAction, ControllerAction
from ..mctsDNN import VidurMCTS
from .history_root import HistoryRootGenerator
from .infer import build_model_inputs
from .replay_write import ReplayWriter, make_root_sample
from ..logger.evaluation_pipeline_logger import ArenaGameCycleFileLogger, arena_state_snapshot_for_log
import math 
import random
import json


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


def _action_to_json(action) -> str:
    try:
        if isinstance(action, ControllerAction):
            return json.dumps(
                {
                    "type": "controller",
                    "token_budget": int(action.token_budget),
                    "selected_request_ids": [int(x) for x in (action.selected_request_ids or [])],
                    "token_allocations": {str(int(k)): int(v) for k, v in (action.token_allocations or {}).items()},
                    "prefill_allocations": {str(int(k)): int(v) for k, v in (action.prefill_allocations or {}).items()},
                    "decode_allocations": {str(int(k)): int(v) for k, v in (action.decode_allocations or {}).items()},
                    "heuristic": action.heuristic,
                    "strategy": action.strategy,
                },
                ensure_ascii=False,
            )
        if isinstance(action, AdversaryAction):
            return json.dumps(
                {
                    "type": "adversary",
                    "requests": [
                        {
                            "prefill_tokens": int(r.prefill_tokens),
                            "decode_tokens": int(r.decode_tokens),
                            "prefill_slo": float(r.prefill_slo),
                            "decode_slo": float(r.decode_slo),
                        }
                        for r in (action.requests or [])
                    ],
                    "stop_decode_ids": [int(x) for x in (action.stop_decode_ids or [])],
                },
                ensure_ascii=False,
            )
    except Exception:
        return ""
    return ""



@dataclass(frozen=True)
class SingleRootRun:
    game_id: int = 0
    root_id: int = 0
    root_depth: int = 0
    root_player: str = "adversary"  # "adversary" or "controller"
    iterations: int = 5000
    feature_version: int = 1
    root_node_id_override: int | None = None

@dataclass(frozen=True)
class RootSearchResult:
    mask_list: list[bool]
    mcts_prior: list[float]
    best_idx: int
    root_node_id: int


class SelfPlayRunner:
    def __init__(
        self,
        *,
        env: Any,
        mcts: VidurMCTS,
        model,
        writer: ReplayWriter,
        device_for_features: torch.device = torch.device("cpu"),
        game_v2_cfg: Optional[GameVersion2Config] = None,
    ) -> None:
        self.env = env
        self.mcts = mcts
        self.model = model
        self.writer = writer
        self.device = device_for_features
        self._gv2_cfg = game_v2_cfg if game_v2_cfg is not None else getattr(env, "_gv2_cfg", None)
        self.history = HistoryRootGenerator(
            env=self.env,
            iter_logger=getattr(self.mcts, "_iter_logger", None),
            root_logger=getattr(self.mcts, "_root_logger", None),
        )

    def _sample_actions_readonly(self, state: VidurMCTSState, player: str):
        probe = state.fork(flag=False)
        if player == "controller":
            return self.env.sample_controller_actions(probe)
        return self.env.sample_adversary_actions(probe)

    def _resolve_history_settings(
        self,
        *,
        history_nontrivial_hops: Optional[int] = None,
        history_seed: Optional[int] = None,
        max_forced_hops_per_root: Optional[int] = None,
        history_max_total_steps: Optional[int] = None,
        log_history_rows: Optional[bool] = None,
    ) -> tuple[int, int, int, int, bool]:
        hcfg = getattr(self._gv2_cfg, "history_root", None)

        hops = int(history_nontrivial_hops if history_nontrivial_hops is not None else getattr(hcfg, "nontrivial_hops", 0))
        seed = int(history_seed if history_seed is not None else getattr(hcfg, "seed", 0))
        max_forced = int(
            max_forced_hops_per_root
            if max_forced_hops_per_root is not None
            else getattr(hcfg, "max_forced_hops_per_root", 1024)
        )
        max_total = int(
            history_max_total_steps
            if history_max_total_steps is not None
            else getattr(hcfg, "max_total_steps", 20000)
        )
        log_rows = bool(log_history_rows if log_history_rows is not None else getattr(hcfg, "log_history_rows", True))
        return hops, seed, max_forced, max_total, log_rows


    def _advance_to_branching_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        max_hops: int = 1024,
    ) -> tuple[VidurMCTSState, str, int]:
        state, player, depth, _next_id, _last_parent, _forced_steps = self.history.advance_to_branching_root(
            state,
            player,
            depth,
            game_id=-1,          # unused when log_steps=False
            root_id=-1,          # unused when log_steps=False
            log_node_id=0,       # unused when log_steps=False
            log_parent_id=None,  # unused when log_steps=False
            max_hops=int(max_hops),
            log_steps=False,
        )
        return state, player, depth


    def run_single_root(self, cfg: SingleRootRun, root_state: Optional[VidurMCTSState] = None  ) -> RootSearchResult:
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
        _, mask = self._sample_actions_readonly(state, cfg.root_player)
        mask_list = _mask_to_list(mask)

        # MCTS targets from the built root
        root = self.mcts._root
        mcts_prior, best_idx = _compute_mcts_prior_from_root(root, mask_list)
        mcts_value = float(root.mean_value())

        # Build model inputs (CPU) and force action_mask to env mask
        base_inputs = build_model_inputs(state, cfg.root_player, self.device)
        inputs = replace(
            base_inputs,
            action_mask=torch.tensor(
                mask_list,
                dtype=torch.bool,
                device=base_inputs.global_features.device,
            ).unsqueeze(0),
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
        return RootSearchResult(
            mask_list=mask_list,
            mcts_prior=mcts_prior,
            best_idx=int(best_idx),
            root_node_id=int(root.node_id),
        )

    # TODO : seems like start_root_depth is unncessary & we might not even need to use the best action if run_single_root is doing it 
    def run_n_roots(
        self,
        *,
        game_id: int,
        num_roots: int,
        adv_iterations_per_root: int = 1000,
        cont_iterations_per_root: int = 500,
        max_batch_size: int = 72,
        start_root_id: int = 0,
        start_root_depth: int = 0,
        start_player: str = "adversary",
        feature_version: int = 1,
        initial_state: Optional[VidurMCTSState] = None,
        sample_from_mcts_policy: bool = False,
        selfplay_policy_temperature: float = 0.0,
        action_seed_base: int = 0,
        history_nontrivial_hops: Optional[int] = None,
        history_seed: Optional[int] = None,
        max_forced_hops_per_root: Optional[int] = None,
        history_max_total_steps: Optional[int] = None,
        log_history_rows: Optional[bool] = None,
    ) -> VidurMCTSState:
        # state = initial_state or self.env.initial_state()
        # player = start_player

        # Resolving History State 
        hist_hops, hist_seed, max_forced_hops, hist_max_total_steps, hist_log_rows = self._resolve_history_settings(
            history_nontrivial_hops=history_nontrivial_hops,
            history_seed=history_seed,
            max_forced_hops_per_root=max_forced_hops_per_root,
            history_max_total_steps=history_max_total_steps,
            log_history_rows=log_history_rows,
        )

        state = initial_state or self.env.initial_state()
        player = start_player
        depth = int(start_root_depth)
        next_log_node_id = int(getattr(self.mcts, "_node_counter", 0))
        last_log_node_id: int | None = None


        if int(hist_hops) > 0:
            state, player, depth, next_log_node_id, last_log_node_id = self.history.generate_history_root(
                state,
                player,
                depth,
                nontrivial_hops=int(hist_hops),
                game_id=int(game_id),
                root_id_for_logs=int(start_root_id),
                log_history=bool(hist_log_rows),
                log_node_id_start=next_log_node_id,
                log_parent_id_start=last_log_node_id,
                seed=int(hist_seed),
                max_total_steps=int(hist_max_total_steps),
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
                max_hops=int(max_forced_hops),
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
            root_res = self.run_single_root(
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
            root = self.mcts._root
            if root is None or not root.children:
                if hasattr(self.mcts, "clear_search_state"):
                    self.mcts.clear_search_state(drop_scratch=True)
                break

            mask_list = list(root_res.mask_list)
            mcts_prior = list(root_res.mcts_prior)
            best_idx = int(root_res.best_idx)

            chosen_idx = int(best_idx)
            # Sampling
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
                if 0 <= sampled_idx < len(mask_list) and mask_list[sampled_idx]:
                    chosen_idx = int(sampled_idx)

            alias_to_canon = getattr(root, "action_alias_to_canonical", {}) or {}
            canon_idx = int(alias_to_canon.get(int(chosen_idx), int(chosen_idx)))

            child = root.children.get(canon_idx)
            action = child.parent_action if child is not None else None
            best_idx = int(chosen_idx)  # keep logged best as selected policy index

            
            if action is None:
                actions_by_index, mask = self._sample_actions_readonly(state, player)
                mask_list_fb = _mask_to_list(mask)
                if not (0 <= chosen_idx < len(actions_by_index)) or not mask_list_fb[chosen_idx]:
                    valid = [i for i, ok in enumerate(mask_list_fb) if ok and actions_by_index[i] is not None]
                    if not valid:
                        if hasattr(self.mcts, "clear_search_state"):
                            self.mcts.clear_search_state(drop_scratch=True)
                        break
                    chosen_idx = int(max(valid, key=lambda i: (float(mcts_prior[i]), -int(i))))
                    best_idx = int(chosen_idx)  # update best_idx to reflect fallback choice
                action = actions_by_index[chosen_idx]
                if action is None:
                    if hasattr(self.mcts, "clear_search_state"):
                        self.mcts.clear_search_state(drop_scratch=True)
                    break




            # 2b) Log the actually executed action (after optional sampling).
            # This is separate from mcts.search_dnn() root log, which records
            # the best action from root visits before self-play sampling.
            root_logger = getattr(self.mcts, "_root_logger", None)
            if root_logger is not None and root is not None and hasattr(root_logger, "log_root"):
                model_prior = list(getattr(root, "nn_priors", []) or [])
                norm_prior = list(getattr(root, "nn_priors_after_threshold", []) or [])
                if len(model_prior) < len(mask_list):
                    model_prior = model_prior + [0.0] * (len(mask_list) - len(model_prior))
                if len(norm_prior) < len(mask_list):
                    norm_prior = norm_prior + [0.0] * (len(mask_list) - len(norm_prior))
                model_prior = model_prior[: len(mask_list)]
                norm_prior = norm_prior[: len(mask_list)]

                nn_value = getattr(root, "nn_value_controller", None)
                model_v = float(nn_value) if nn_value is not None else 0.0
                viol, lateness = self.env.evaluate_objective(state)
                sampled_flag = int(sample_from_mcts_policy and player == "controller")

                sim_time_now = float(state.simulator._time)
                if str(player).strip().lower() == "adversary":
                    try:
                        decision_state_time = float(self.env._v2_current_adv_tick(state))
                    except Exception:
                        decision_state_time = sim_time_now
                else:
                    decision_state_time = sim_time_now

                state_desc = self.env.describe_state(state)
                last_adv_raw = state_desc.get("last_adv_tick", "")
                try:
                    state_last_adv_tick = None if last_adv_raw in ("", None) else float(last_adv_raw)
                except Exception:
                    state_last_adv_tick = None

                decode_counted: dict[int, int] = {}
                for k, v in (state_desc.get("decode_tokens_counted_by_id") or {}).items():
                    try:
                        ik = int(k)
                        if ik >= 0:
                            decode_counted[ik] = int(v)
                    except Exception:
                        pass

                root_logger.log_root(
                    game_id=int(game_id),
                    root_id=int(root_id),
                    root_depth=int(depth),
                    root_node_id=int(getattr(root, "node_id", -1)),
                    root_player=str(player),
                    num_simulations=int(iters),
                    model_root_value_controller=float(model_v),
                    model_root_prior=model_prior,
                    normalized_root_prior=norm_prior,
                    valid_action_mask=mask_list,
                    mcts_root_value_controller=float(root.mean_value()),
                    mcts_root_prior=mcts_prior,
                    best_action_index=int(best_idx),
                    best_action_repr=repr(action),
                    best_action_json=_action_to_json(action),
                    phase="train_root_applied",
                    cycle_label=f"sampled={sampled_flag}",
                    sim_time=sim_time_now,
                    decision_state_time=float(decision_state_time),
                    state_pending_adv_tick=bool(state_desc.get("pending_adv_tick", False)),
                    state_last_adv_tick=state_last_adv_tick,
                    state_active_ids=[int(x) for x in (state_desc.get("active_request_ids") or [])],
                    state_completed_request_ids=[int(x) for x in (state_desc.get("completed_request_ids") or [])],
                    state_decode_credit_balance=int(state_desc.get("decode_credit_balance", 0)),
                    state_decode_tokens_counted_by_id=decode_counted,
                    slo_violations=int(viol),
                    total_lateness=float(lateness),
                    total_cost=float(float(viol) + float(lateness)),
                )



                

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
        # Search on a fork so arena state is not mutated by search.
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
            root = self.mcts._root
            if root is None or not root.children:
                return None, -1

            # Use canonical children directly from the searched tree.
            candidate_idxs = [
                int(i)
                for i, ch in root.children.items()
                if ch is not None and ch.parent_action is not None
            ]
            if not candidate_idxs:
                return None, -1

            # Optional arena bias: if adversary has launch actions, prefer those over pure stop/no-op.
            if player == "adversary" and bool(prefer_nonempty_adversary):
                nonempty = [
                    i
                    for i in candidate_idxs
                    if isinstance(root.children[i].parent_action, AdversaryAction)
                    and len((root.children[i].parent_action.requests or [])) > 0
                ]
                if nonempty:
                    candidate_idxs = nonempty

            # Choose by visits first (MCTS policy), then prior, then stable index tie-break.
            best_idx = max(
                candidate_idxs,
                key=lambda i: (
                    int(getattr(root.children[i], "visits", 0)),
                    float(getattr(root.children[i], "prior", 0.0)),
                    -int(i),
                ),
            )

            action = root.children[int(best_idx)].parent_action
            if action is None:
                return None, -1
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
        arena_time_limit_sec: float,
        arena_max_controller_cleanup_steps: int,
        arena_max_total_turns: int,
        feature_version: int,
        cycle_file_logger = None,
    ) -> dict:
        state = self.env.clone_state_from_snapshot(base_snapshot, base_stats)
        player = str(base_player)
        depth = int(base_depth)

        turns = 0
        cleanup_steps = 0
        adv_moves_total = 0
        adv_request_moves = 0

        deadline_t = float(state.simulator._time) + float(arena_time_limit_sec)
        end_reason = ""

        while (
            turns < int(arena_max_total_turns)
            and float(state.simulator._time) < deadline_t
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
                end_reason = "no_valid_action"
                break

            # States for Arena logging :
            sim_before = float(state.simulator._time)
            player_before = str(player)
            depth_before = int(depth)
            turn_before = int(turns)

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

            viol_step, lateness_step = self.env.evaluate_objective(state)
            step_cost = float(viol_step) + float(lateness_step)
            state_log = arena_state_snapshot_for_log(self.env, state)

            ## Logging Arena step move :
            if cycle_file_logger is not None:
                cycle_file_logger.write_step(
                    game_id=int(game_id),
                    cycle_label=str(cycle_label),
                    phase="arena_step",  # or "cleanup_step" in cleanup loop
                    turn=int(turn_before),
                    depth=int(depth_before),
                    player_acted=str(player_before),
                    player_to_act_next=str(player),
                    action_repr=repr(action),
                    sim_time_before=float(sim_before),
                    sim_time_after=float(state.simulator._time),
                    total_cost=step_cost,
                    slo_violations=int(viol_step),
                    total_lateness=float(lateness_step),
                    **state_log,
                )



        if not end_reason and float(state.simulator._time) >= deadline_t:
            end_reason = "time_limit"

        while (
            turns < int(arena_max_total_turns)
            and cleanup_steps < int(arena_max_controller_cleanup_steps)
            and float(state.simulator._time) < deadline_t
            and self._has_prefill_pending(state)
        ):

            sim_before = float(state.simulator._time)
            player_before = str(player)
            depth_before = int(depth)
            turn_before = int(turns)

            if player == "adversary":
                noop = AdversaryAction(requests=[], stop_decode_ids=[])
                state = self.env.apply_adversary_action_only(state, noop, inplace=True)
                player = "controller"
                depth += 1
                turns += 1
                # Eval logging for noop adversary action in cleanup :
                viol_step, lateness_step = self.env.evaluate_objective(state)
                step_cost = float(viol_step) + float(lateness_step)
                state_log = arena_state_snapshot_for_log(self.env, state)

                if cycle_file_logger is not None:
                    cycle_file_logger.write_step(
                        game_id=int(game_id),
                        cycle_label=str(cycle_label),
                        phase="cleanup_step",
                        turn=int(turn_before),
                        depth=int(depth_before),
                        player_acted=str(player_before),
                        player_to_act_next=str(player),
                        action_repr=repr(noop),
                        sim_time_before=float(sim_before),
                        sim_time_after=float(state.simulator._time),
                        total_cost=step_cost,
                        slo_violations=int(viol_step),
                        total_lateness=float(lateness_step),
                        **state_log,
                    )
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
                end_reason = end_reason or "cleanup_no_valid_action"
                break

            state = self.env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"
            depth += 1
            turns += 1
            cleanup_steps += 1

            viol_step, lateness_step = self.env.evaluate_objective(state)
            step_cost = float(viol_step) + float(lateness_step)
            state_log = arena_state_snapshot_for_log(self.env, state)   
            if cycle_file_logger is not None:
                cycle_file_logger.write_step(
                    game_id=int(game_id),
                    cycle_label=str(cycle_label),
                    phase="cleanup_step",  # or "cleanup_step" in cleanup loop
                    turn=int(turn_before),
                    depth=int(depth_before),
                    player_acted=str(player_before),
                    player_to_act_next=str(player),
                    action_repr=repr(action),
                    sim_time_before=float(sim_before),
                    sim_time_after=float(state.simulator._time),
                    total_cost=step_cost,
                    slo_violations=int(viol_step),
                    total_lateness=float(lateness_step),
                    **state_log,
                )


        if not end_reason:
            if float(state.simulator._time) >= deadline_t:
                end_reason = "time_limit"
            elif turns >= int(arena_max_total_turns):
                end_reason = "max_total_turns"
            elif cleanup_steps >= int(arena_max_controller_cleanup_steps):
                end_reason = "max_cleanup_steps"

        viol, lateness = self.env.evaluate_objective(state)
        total_cost = float(viol) + float(lateness)

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
            "end_reason": str(end_reason),
            "sim_time_end": float(state.simulator._time),
            "sim_time_deadline": float(deadline_t),
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
        arena_time_limit_sec: float,
        arena_max_controller_cleanup_steps: int,
        arena_max_total_turns: int,
        feature_version: int = 1,
        tie_points: float = 0.5,
        start_player: str = "adversary",
        start_root_depth: int = 0,
        history_root_id_for_logs: int = 0,
        cycle_file_logger = None,
    ) -> dict:

        # Util functions for logging :
        history_events: list[dict] = []

        def _history_cb(ev: dict) -> None:
            ev2 = dict(ev)
            ev2["turn"] = len(history_events)
            history_events.append(ev2)

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
                step_callback=_history_cb,
            )
            self.mcts._node_counter = int(next_log_node_id)

        base_snapshot = state.simulator.snapshot_state()
        base_stats = state.stats.clone()

        ## Logging the History Events for the Arena Game :

        if cycle_file_logger is not None and history_events:
            for cycle_label in ("candidate_as_adversary", "best_as_adversary"):
                for ev in history_events:
                    cycle_file_logger.write_step(
                        game_id=int(game_id),
                        cycle_label=cycle_label,
                        phase=f"history_step:{ev.get('phase', '')}",
                        turn=int(ev.get("turn", 0)),
                        depth=int(ev.get("depth_before", depth)),
                        player_acted=str(ev.get("player_acted", "")),
                        player_to_act_next=str(ev.get("player_to_act_next", "")),
                        action_repr=str(ev.get("action_repr", "")),
                        sim_time_before=float(ev.get("sim_time_before", state.simulator._time)),
                        sim_time_after=float(ev.get("sim_time_after", state.simulator._time)),
                        total_cost=float(ev.get("total_cost", 0.0)),
                        slo_violations=int(ev.get("slo_violations", 0)),
                        total_lateness=float(ev.get("total_lateness", 0.0)),
                    )


        ## Logging ends

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
            arena_time_limit_sec=float(arena_time_limit_sec),
            arena_max_controller_cleanup_steps=int(arena_max_controller_cleanup_steps),
            arena_max_total_turns=int(arena_max_total_turns),
            feature_version=int(feature_version),
            cycle_file_logger=cycle_file_logger,
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
            arena_time_limit_sec=float(arena_time_limit_sec),
            arena_max_controller_cleanup_steps=int(arena_max_controller_cleanup_steps),
            arena_max_total_turns=int(arena_max_total_turns),
            feature_version=int(feature_version),
            cycle_file_logger=cycle_file_logger,
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
