# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import json
import random
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import multiprocessing as mp
from dataclasses import dataclass, replace


import torch

from ..environment import AdversaryAction, VidurMCTSEnvironment, VidurMCTSState
from vidur.config import SimulationConfig
from vidur.simulator import Simulator

from .infer import build_model_inputs
from .types import ModelInputs
from ..launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions
# from ..logger.eval_logger import EvalArenaGenerationLogger
from ..logger.eval_logger import EvalArenaGenerationLogger, EvalArenaStepLogger

from .models import AlphaZeroModel
from .trainer import Trainer
from .selfPlay import _mask_to_list
from .history_root import HistoryRootGenerator
from ..mctsDNN import VidurMCTS



@dataclass(frozen=True)
class _ArenaWorkerArgs:
    worker_id: int
    env_kind: str  # "standard" | "virtual"
    # sim_cfg: Any
    sim_cli_args: List[str]
    assume_infinite_kv: bool
    constraints: MCTSConstraintConfig
    explore_cfg: MCTSExploreConfig
    eval_cfg: EvaluatorConfig
    root_entries: List[Dict[str, Any]]
    history_trace_by_game: Dict[int, List[Dict[str, Any]]]
    candidate_state_dict: Dict[str, torch.Tensor]
    best_state_dict: Dict[str, torch.Tensor]
    num_actions_controller: int
    num_actions_adversary: int
    sampled_game_ids: List[int]
    gen_dir: str
    debug_flush_every: int


def _split_even_ranges(total: int, parts: int) -> List[tuple[int, int]]:
    total = max(0, int(total))
    parts = max(1, min(int(parts), total if total > 0 else 1))
    base = total // parts
    rem = total % parts
    out: List[tuple[int, int]] = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            continue
        end = start + size
        out.append((start, end))
        start = end
    return out


class _NoopSummary:
    def log_row(self, row: Dict[str, Any]) -> None:
        return


class _StepOnlyArenaLogger:
    def __init__(self, *, gen_dir: Path, sampled_game_ids: Sequence[int], flush_every: int = 1) -> None:
        self.gen_dir = Path(gen_dir)
        self.sampled_game_ids = {int(x) for x in sampled_game_ids}
        self.flush_every = max(1, int(flush_every))
        self._step_loggers: Dict[tuple[int, str], EvalArenaStepLogger] = {}
        self.summary = _NoopSummary()  # evaluate_arena expects .summary.log_row(...)

    def step_logger(self, game_id: int, cycle_label: str) -> Optional[EvalArenaStepLogger]:
        gid = int(game_id)
        if gid not in self.sampled_game_ids:
            return None
        key = (gid, str(cycle_label))
        lg = self._step_loggers.get(key)
        if lg is not None:
            return lg
        path = self.gen_dir / str(cycle_label) / f"game_{gid}.csv"
        lg = EvalArenaStepLogger(path, flush_every=self.flush_every)
        self._step_loggers[key] = lg
        return lg

    def close(self) -> None:
        for lg in self._step_loggers.values():
            lg.close()
        self._step_loggers.clear()


def _arena_worker_main(result_q: "mp.Queue", args: _ArenaWorkerArgs) -> None:
    try:

        sim_cfg = _configure_simulation_from_cli_args(args.sim_cli_args)
        if bool(args.assume_infinite_kv):
            setattr(sim_cfg.cluster_config.cache_config, "assume_infinite_kv", True)

        if args.env_kind == "virtual":
            from ..virtual_simulator import VirtualSimulator
            from ..virtual_environment import VirtualVidurMCTSEnvironment

            sim = VirtualSimulator(sim_cfg, register_atexit=False)
            env = VirtualVidurMCTSEnvironment(
                base_simulator=sim,
                constraints=args.constraints,
                explore_cfg=args.explore_cfg,
            )
        else:
            sim = Simulator(sim_cfg, register_atexit=False)
            env = VidurMCTSEnvironment(
                base_simulator=sim,
                constraints=args.constraints,
                explore_cfg=args.explore_cfg,
            )

        mcts = VidurMCTS(
            env=env,
            explore_cfg=args.explore_cfg,
            log_path=None,
            tree_log_path=None,
            logger_flush_every=1,
        )

        worker_cfg = replace(args.eval_cfg, arena_num_processes=1)
        evaluator = FixedEvalStatesEvaluator(env=env, mcts=mcts, cfg=worker_cfg)

        roots: List[EvalRoot] = []
        for e in args.root_entries:
            state = env.clone_state_from_snapshot(e["snapshot"], e["stats"])
            roots.append(
                EvalRoot(
                    hops=int(e["history_hops"]),
                    seed=int(e["seed"]),
                    game_id=int(e["game_id"]),
                    root_id=int(e["root_id"]),
                    state=state,
                    player_to_act=str(e["player_to_act"]),
                    depth=int(e["depth"]),
                )
            )

        evaluator.eval_roots = roots
        evaluator.history_trace_by_game = {
            int(k): v for k, v in args.history_trace_by_game.items()
        }

        candidate = AlphaZeroModel(
            num_actions_controller=int(args.num_actions_controller),
            num_actions_adversary=int(args.num_actions_adversary),
        ).to(torch.device("cpu"))
        candidate.load_state_dict(args.candidate_state_dict, strict=True)
        candidate.eval()

        best = AlphaZeroModel(
            num_actions_controller=int(args.num_actions_controller),
            num_actions_adversary=int(args.num_actions_adversary),
        ).to(torch.device("cpu"))
        best.load_state_dict(args.best_state_dict, strict=True)
        best.eval()

        step_logger = _StepOnlyArenaLogger(
            gen_dir=Path(args.gen_dir),
            sampled_game_ids=args.sampled_game_ids,
            flush_every=int(args.debug_flush_every),
        )

        try:
            metrics = evaluator.evaluate_arena(
                candidate_model=candidate,
                best_model=best,
                generation_logger=step_logger,
            )
        finally:
            step_logger.close()
            mcts.close()

        # fix root_index to global index provided in payload
        gid_to_idx = {int(e["game_id"]): int(e["global_root_index"]) for e in args.root_entries}
        for row in metrics.get("per_root", []):
            gid = int(row.get("game_id", -1))
            if gid in gid_to_idx:
                row["root_index"] = int(gid_to_idx[gid])

        result_q.put({"ok": True, "worker_id": int(args.worker_id), "metrics": metrics})
    except Exception as exc:
        result_q.put({"ok": False, "worker_id": int(args.worker_id), "error": repr(exc)})




def _configure_simulation_from_cli_args(sim_args: Sequence[str]) -> SimulationConfig:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(sim_args)
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv

    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False

    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]

    return cfg


@dataclass(frozen=True)
class EvaluatorConfig:
    num_random_games: int = 8
    max_history_depth: int = 20
    random_seed_base: int = 12345

    arena_num_processes: int = 1

    eval_game_id_base: int = 900_000
    eval_root_id_base: int = 0
    history_max_total_steps: int = 200_000

    adv_iterations_per_root: int = 2000
    cont_iterations_per_root: int = 2000

    # TODO: Remove these later onwards 
    arena_iters_adversary: int = 2000
    arena_iters_controller: int = 2000

    arena_max_adversary_moves: int = 5
    arena_max_adversary_total_turns: int = 256  # safety cap including no-op adversary turns
    arena_require_request_generating_adversary: bool = True
    arena_log_model_prior_on_forced_adv_noop: bool = True
    arena_max_controller_cleanup_steps: int = 64
    arena_max_total_turns: int = 512

    arena_win_threshold: float = 0.55
    tie_points: float = 0.5

    debug_sample_games: int = 5
    debug_flush_every: int = 1
    arena_per_gen: int = 1


@dataclass(frozen=True)
class EvalGameSpec:
    game_id: int
    root_id: int
    history_hops: int
    seed: int


@dataclass
class EvalRoot:
    hops: int
    seed: int
    game_id: int
    root_id: int
    state: VidurMCTSState
    player_to_act: str
    depth: int


class FixedEvalStatesEvaluator:
    def __init__(
        self,
        *,
        env: VidurMCTSEnvironment,
        mcts: VidurMCTS,
        cfg: EvaluatorConfig,
        sim_cli_args: Optional[Sequence[str]] = None
    ) -> None:
        self.env = env
        self.mcts = mcts
        self.cfg = cfg

        self.eval_roots: List[EvalRoot] = []
        self.history_trace_by_game: Dict[int, List[Dict[str, Any]]] = {}
        self._sim_cli_args = list(sim_cli_args) if sim_cli_args is not None else None
        self.max_branching = int(getattr(getattr(self.env, "_cfg", None), "max_branching", 10))

        self.history = HistoryRootGenerator(
            env=self.env,
            max_branching=self.max_branching,
            iter_logger=None,  # evaluator keeps its own csv logging
        )


    def _advance_to_branching_root(
        self,
        state: VidurMCTSState,
        player: str,
        depth: int,
        *,
        game_id: int,
        root_id: int,
    ) -> tuple[VidurMCTSState, str, int]:
        state, player, depth, _next_node_id, _last_node_id = self.history.advance_to_branching_root(
            state,
            player,
            depth,
            game_id=int(game_id),
            root_id=int(root_id),
            log_node_id=0,
            log_parent_id=None,
            log_steps=False,  # do NOT log forced steps
            max_hops=min(10000, int(self.cfg.history_max_total_steps)),
        )
        return state, player, int(depth)



    @staticmethod
    def make_random_history_lengths(
        *,
        num_random_games: int,
        max_history_depth: int,
        rng: random.Random,
    ) -> List[int]:
        n = max(1, int(num_random_games))
        dmax = max(0, int(max_history_depth))

        out = [0]
        for _ in range(n - 1):
            out.append(0 if dmax <= 0 else int(rng.randint(1, dmax)))
        return out

    def make_generation_game_specs(self, *, gen: int) -> List[EvalGameSpec]:
        seed = int(self.cfg.random_seed_base) + int(gen) * 100_003
        rng = random.Random(seed)

        lengths = self.make_random_history_lengths(
            num_random_games=self.cfg.num_random_games,
            max_history_depth=self.cfg.max_history_depth,
            rng=rng,
        )

        specs: List[EvalGameSpec] = []
        for i, h in enumerate(lengths):
            specs.append(
                EvalGameSpec(
                    game_id=int(self.cfg.eval_game_id_base + gen * 10_000 + i),
                    root_id=int(self.cfg.eval_root_id_base + i),
                    history_hops=int(h),
                    seed=int(rng.randrange(2**31 - 1)),
                )
            )
        return specs



    def _iters_for_player(self, player: str) -> int:
        if player == "adversary":
            return int(getattr(self.cfg, "adv_iterations_per_root",
                            getattr(self.cfg, "arena_iters_adversary", 2000)))
        return int(getattr(self.cfg, "cont_iterations_per_root",
                        getattr(self.cfg, "arena_iters_controller", 2000)))



    def _has_prefill_pending(self, state: VidurMCTSState) -> bool:
        req_lookup = self.env._build_request_lookup(state.simulator)
        for req in req_lookup.values():
            remaining_prefill = max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))
            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            if remaining_prefill > 0 and not prefill_done:
                return True
        return False

   
    def _sample_actions_and_mask(self, state: VidurMCTSState, player: str) -> tuple[list[Optional[object]], list[bool]]:
        if player == "controller":
            actions_by_index, mask = self.env.sample_controller_actions(state, self.max_branching)
        else:
            actions_by_index, mask = self.env.sample_adversary_actions(state, self.max_branching)
        return actions_by_index, _mask_to_list(mask)

    def _infer_policy_with_env_mask(
        self,
        *,
        state: VidurMCTSState,
        player: str,
        model: Any,
        env_mask: list[bool],
    ) -> tuple[float, list[float]]:
        if not hasattr(model, "infer_from_inputs"):
            raise TypeError("Arena model must implement infer_from_inputs(inputs, player, device=...)")

        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

        base_inputs = build_model_inputs(state, player, device)
        inputs = ModelInputs(
            req_features=base_inputs.req_features,
            global_features=base_inputs.global_features,
            req_mask=base_inputs.req_mask,
            action_mask=torch.tensor(env_mask, dtype=torch.bool, device=device).unsqueeze(0),
        )

        model_value, priors = model.infer_from_inputs(inputs, player, device=device)

        a = len(env_mask)
        if len(priors) != a:
            priors = [0.0] * a

        masked = []
        for i in range(a):
            p = float(priors[i]) if env_mask[i] else 0.0
            if (not math.isfinite(p)) or p < 0.0:
                p = 0.0
            masked.append(p)

        valid = [i for i, ok in enumerate(env_mask) if ok]
        denom = sum(masked[i] for i in valid)
        if valid and denom <= 0.0:
            u = 1.0 / float(len(valid))
            masked = [u if env_mask[i] else 0.0 for i in range(a)]
        elif denom > 0.0:
            inv = 1.0 / denom
            masked = [masked[i] * inv if env_mask[i] else 0.0 for i in range(a)]

        return float(model_value), masked




    def _apply_action(self, state: VidurMCTSState, player: str, action: object) -> tuple[VidurMCTSState, str]:
        if player == "adversary":
            state = self.env.apply_adversary_action_only(state, action, inplace=True)
            return state, "controller"
        state = self.env.apply_controller_action_only(state, action, inplace=True)
        return state, "adversary"

    # evaluator.py (inside FixedEvalStatesEvaluator)

    def _adversary_new_prefill_deadlines_json(
        self,
        *,
        acting_player: str,
        state_after: VidurMCTSState,
        before_request_ids: set[int],
    ) -> str:
        if acting_player != "adversary":
            return "{}"

        lookup = self.env._build_request_lookup(state_after.simulator)
        new_ids = sorted(set(lookup.keys()) - set(before_request_ids))
        out: Dict[int, float] = {}
        for rid in new_ids:
            req = lookup.get(rid)
            if req is None:
                continue
            queued_at = float(getattr(req, "queued_at", getattr(req, "arrived_at", 0.0)) or 0.0)
            slo = float(getattr(req, "prefill_slo_time", 0.0) or 0.0)
            out[int(rid)] = queued_at + slo
        return json.dumps(out, ensure_ascii=False)

    def _state_row(
        self,
        state: VidurMCTSState,
        *,
        adversary_prefill_deadlines_by_id: str = "{}",
    ) -> Dict[str, Any]:
        snap = self.env.describe_state(state)
        violations, total_lateness = self.env.evaluate_objective(state)
        return {
            "requests_in_system": int(snap.get("requests_in_system", 0)),
            "state_waiting_ids": json.dumps(snap.get("waiting_request_ids", []), ensure_ascii=False),
            "state_completed_request_ids": json.dumps(snap.get("completed_request_ids", []), ensure_ascii=False),
            "adversary_prefill_deadlines_by_id": str(adversary_prefill_deadlines_by_id or "{}"),
            "sim_time": float(snap.get("sim_time", 0.0)),
            "slo_violations": int(violations),
            "total_lateness": float(total_lateness),
            "total_cost": float(float(violations) + float(total_lateness)),
        }


    
    # evaluator.py (replace _generate_history_root_with_trace)
    def _generate_history_root_with_trace(
        self,
        *,
        spec: EvalGameSpec,
        start_player: str,
    ) -> tuple[VidurMCTSState, str, int, List[Dict[str, Any]]]:
        state = self.env.initial_state()
        player = str(start_player)
        depth = 0
        trace: List[Dict[str, Any]] = []

        rng = random.Random(int(spec.seed))
        target = int(spec.history_hops)
        done = 0

        total_steps = 0
        max_steps = int(self.cfg.history_max_total_steps)

        def _record_branch_step(
            *,
            acting_player: str,
            action_repr: str,
            root_depth: int,
            step_index: int,
            best_action_index: int,
            adversary_prefill_deadlines_by_id: str,
        ) -> None:
            row = self._state_row(
                state,
                adversary_prefill_deadlines_by_id=adversary_prefill_deadlines_by_id,
            )
            row.update(
                {
                    "phase": "history",
                    "step_index": int(step_index),
                    "acting_player": str(acting_player),
                    "best_action_index": int(best_action_index),
                    "best_action_repr": str(action_repr),
                    "root_depth": int(root_depth),
                }
            )
            trace.append(row)

        while done < target:
            prev_depth = int(depth)
            state, player, depth = self._advance_to_branching_root(
                state,
                player,
                depth,
                game_id=int(spec.game_id),
                root_id=int(spec.root_id),
            )
            total_steps += max(0, int(depth) - prev_depth)
            if total_steps >= max_steps:
                raise RuntimeError("history generation exceeded history_max_total_steps")

            actions_by_index, mask_list = self._sample_actions_and_mask(state, player)
            valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
            if not valid:
                return state, player, int(depth), trace

            idx = int(valid[0] if len(valid) == 1 else rng.choice(valid))
            action = actions_by_index[idx]
            assert action is not None

            acting = player
            cur_depth = int(depth)

            before_ids: set[int] = set()
            if acting == "adversary":
                before_ids = set(self.env._build_request_lookup(state.simulator).keys())

            state, player = self._apply_action(state, player, action)

            if acting == "adversary":
                adv_deadlines_json = self._adversary_new_prefill_deadlines_json(
                    acting_player=acting,
                    state_after=state,
                    before_request_ids=before_ids,
                )
            else:
                adv_deadlines_json = "{}"

            depth += 1
            done += 1
            total_steps += 1

            _record_branch_step(
                acting_player=acting,
                action_repr=repr(action),
                root_depth=cur_depth,
                step_index=done,
                best_action_index=idx,
                adversary_prefill_deadlines_by_id=adv_deadlines_json,
            )

            if total_steps >= max_steps:
                raise RuntimeError("history generation exceeded history_max_total_steps")

        prev_depth = int(depth)
        state, player, depth = self._advance_to_branching_root(
            state,
            player,
            depth,
            game_id=int(spec.game_id),
            root_id=int(spec.root_id),
        )
        total_steps += max(0, int(depth) - prev_depth)
        if total_steps >= max_steps:
            raise RuntimeError("history generation exceeded history_max_total_steps")

        return state, player, int(depth), trace



    def build_eval_roots_from_game_specs(
        self,
        *,
        specs: Sequence[EvalGameSpec],
        start_player: str = "adversary",
    ) -> tuple[List[EvalRoot], Dict[int, List[Dict[str, Any]]]]:
        roots: List[EvalRoot] = []
        history_trace_by_game: Dict[int, List[Dict[str, Any]]] = {}

        for spec in specs:
            state, player, depth, trace = self._generate_history_root_with_trace(
                spec=spec,
                start_player=start_player,
            )
            roots.append(
                EvalRoot(
                    hops=int(spec.history_hops),
                    seed=int(spec.seed),
                    game_id=int(spec.game_id),
                    root_id=int(spec.root_id),
                    state=state,
                    player_to_act=str(player),
                    depth=int(depth),
                )
            )
            history_trace_by_game[int(spec.game_id)] = trace

        self.eval_roots = roots
        self.history_trace_by_game = history_trace_by_game
        return roots, history_trace_by_game

    def build_eval_roots_for_generation(
        self,
        *,
        gen: int,
        start_player: str = "adversary",
    ) -> Dict[str, Any]:
        specs = self.make_generation_game_specs(gen=gen)
        roots, history_trace_by_game = self.build_eval_roots_from_game_specs(
            specs=specs,
            start_player=start_player,
        )
        return self.export_eval_roots_payload(
            generation=int(gen),
            specs=specs,
            roots=roots,
            history_trace_by_game=history_trace_by_game,
        )

    def export_eval_roots_payload(
        self,
        *,
        generation: int,
        specs: Sequence[EvalGameSpec],
        roots: Sequence[EvalRoot],
        history_trace_by_game: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        spec_map = {int(s.game_id): s for s in specs}
        root_entries: List[Dict[str, Any]] = []

        for root in roots:
            spec = spec_map[int(root.game_id)]
            root_entries.append(
                {
                    "game_id": int(root.game_id),
                    "root_id": int(root.root_id),
                    "history_hops": int(spec.history_hops),
                    "seed": int(root.seed),
                    "player_to_act": str(root.player_to_act),
                    "depth": int(root.depth),
                    "snapshot": root.state.simulator.snapshot_state(),
                    "stats": root.state.stats.clone(),
                }
            )

        return {
            "generation": int(generation),
            "num_random_games": int(self.cfg.num_random_games),
            "max_history_depth": int(self.cfg.max_history_depth),
            "roots": root_entries,
            "history_trace_by_game": history_trace_by_game,
        }

    def load_eval_roots_payload(self, payload: Dict[str, Any]) -> List[EvalRoot]:
        entries = list(payload.get("roots", []))
        roots: List[EvalRoot] = []

        for e in entries:
            state = self.env.clone_state_from_snapshot(e["snapshot"], e["stats"])
            roots.append(
                EvalRoot(
                    hops=int(e["history_hops"]),
                    seed=int(e["seed"]),
                    game_id=int(e["game_id"]),
                    root_id=int(e["root_id"]),
                    state=state,
                    player_to_act=str(e["player_to_act"]),
                    depth=int(e["depth"]),
                )
            )

        self.eval_roots = roots
        self.history_trace_by_game = {
            int(k): v for k, v in dict(payload.get("history_trace_by_game", {})).items()
        }
        return roots

    
    def _pick_action_with_mcts(
        self,
        *,
        state: VidurMCTSState,
        player: str,
        model: Any,
        game_id: int,
        root_id: int,
        root_depth: int,
        prefer_nonempty_adversary: bool = False,
    ) -> tuple[Optional[object], int, Dict[str, Any]]:
        iters = max(1, int(self._iters_for_player(player)))
        self.mcts.search_dnn(
            dnn_model=model,
            rootState=state,
            root_player=player,
            iterations=iters,
            game_id=int(game_id),
            root_id=int(root_id),
            root_node_id_override=0,
            root_depth=int(root_depth),
        )

        actions_by_index, mask_list = self._sample_actions_and_mask(state, player)
        valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
        if not valid:
            return None, -1, {
                "root_node_id": None,
                "model_root_value_controller": None,
                "model_root_prior_json": "[]",
                "mcts_root_value_controller": None,
                "mcts_root_prior_json": "[]",
                "adversary_requests_generated": 0,
                "best_action_index": None,
                "best_action_repr": "",
            }

        root = getattr(self.mcts, "_root", None)
        action_dim = len(mask_list)

        model_v = None
        model_prior = [0.0] * action_dim
        mcts_prior = [0.0] * action_dim
        mcts_v = None
        best_idx: Optional[int] = None
        root_node_id = None

        if root is not None:
            root_node_id = int(getattr(root, "node_id", 0))
            mcts_v = float(root.mean_value())
            try:
                m_v, m_p, _norm_p, _mask, mc_p, b_idx = self.mcts._compute_root_log_payload_from_cache(root=root)
                model_v = float(m_v)
                if len(m_p) == action_dim:
                    model_prior = [float(x) for x in m_p]
                if len(mc_p) == action_dim:
                    mcts_prior = [float(x) for x in mc_p]
                best_idx = int(b_idx)
            except Exception:
                pass

        # forced/single-child fallback: still log model prior if enabled
        if (model_v is None or sum(model_prior) <= 0.0) and bool(
            getattr(self.cfg, "arena_log_model_prior_on_forced_adv_noop", True)
        ):
            mv, mp = self._infer_policy_with_env_mask(
                state=state,
                player=player,
                model=model,
                env_mask=mask_list,
            )
            model_v = float(mv)
            if len(mp) == action_dim:
                model_prior = [float(x) for x in mp]

        # mcts prior fallback if unavailable
        if sum(mcts_prior) <= 0.0:
            if best_idx is not None and best_idx in valid:
                mcts_prior = [1.0 if i == best_idx else 0.0 for i in range(action_dim)]
            else:
                u = 1.0 / float(len(valid))
                mcts_prior = [u if i in valid else 0.0 for i in range(action_dim)]

        candidate_valid = list(valid)
        if prefer_nonempty_adversary and player == "adversary":
            nonempty = [
                i for i in valid
                if isinstance(actions_by_index[i], AdversaryAction)
                and len((actions_by_index[i].requests or [])) > 0
            ]
            if nonempty:
                candidate_valid = nonempty

        if best_idx is None or best_idx not in candidate_valid:
            best_idx = max(candidate_valid, key=lambda i: (mcts_prior[i], -i))

        action = actions_by_index[int(best_idx)]
        if action is None:
            best_idx = int(candidate_valid[0])
            action = actions_by_index[best_idx]
            if action is None:
                return None, -1, {
                    "root_node_id": root_node_id,
                    "model_root_value_controller": model_v,
                    "model_root_prior_json": json.dumps(model_prior, ensure_ascii=False),
                    "mcts_root_value_controller": mcts_v,
                    "mcts_root_prior_json": json.dumps(mcts_prior, ensure_ascii=False),
                    "adversary_requests_generated": 0,
                    "best_action_index": None,
                    "best_action_repr": "",
                }

        adv_req_generated = len(action.requests) if isinstance(action, AdversaryAction) else 0
        return action, int(best_idx), {
            "root_node_id": root_node_id,
            "model_root_value_controller": model_v,
            "model_root_prior_json": json.dumps(model_prior, ensure_ascii=False),
            "mcts_root_value_controller": mcts_v,
            "mcts_root_prior_json": json.dumps(mcts_prior, ensure_ascii=False),
            "adversary_requests_generated": int(adv_req_generated),
            "best_action_index": int(best_idx),
            "best_action_repr": repr(action),
        }



    def _run_cycle(
        self,
        *,
        root: EvalRoot,
        adversary_model: Any,
        controller_model: Any,
        cycle_id: int,
        cycle_label: str,
        best_model_player: str,
        candidate_model_player: str,
        generation_logger: Optional[EvalArenaGenerationLogger],
    ) -> Dict[str, Any]:
        state = root.state.fork(flag=False)
        player = str(root.player_to_act)
        depth = int(root.depth)

        # adv_moves = 0
        adv_moves_total = 0
        adv_request_moves = 0
        turns = 0
        cleanup_steps = 0
        end_reason = "unknown"


        # max_adv = int(self.cfg.arena_max_adversary_moves)
        max_turns = int(self.cfg.arena_max_total_turns)
        max_cleanup = int(self.cfg.arena_max_controller_cleanup_steps)
        max_adv_req = int(self.cfg.arena_max_adversary_moves)
        max_adv_total = int(getattr(self.cfg, "arena_max_adversary_total_turns", max(1, 4 * max_adv_req)))


        step_logger = generation_logger.step_logger(int(root.game_id), cycle_label) if generation_logger else None
        # TODO: Make a function that can call this log step with 4 different phases: history, mcts, post-mcts, and cleanup, and use it for all logging in this function
        if step_logger is not None:
            for hist_row in self.history_trace_by_game.get(int(root.game_id), []):
                step_logger.log_step(
                    {
                        "game_id": int(root.game_id),
                        "history_length": int(root.hops),
                        "best_model_player": str(best_model_player),
                        "candidate_model_player": str(candidate_model_player),
                        "root_id": int(root.root_id),
                        "root_depth": int(hist_row.get("root_depth", 0)),
                        "root_node_id": None,
                        "root_player": str(hist_row.get("acting_player", "")),
                        "model_root_value_controller": None,
                        "model_root_prior_json": "[]",
                        "best_action_index": hist_row.get("best_action_index"),
                        "best_action_repr": str(hist_row.get("best_action_repr", "")),
                        "requests_in_system": int(hist_row.get("requests_in_system", 0)),
                        "state_waiting_ids": str(hist_row.get("state_waiting_ids", "[]")),
                        "state_completed_request_ids": str(hist_row.get("state_completed_request_ids", "[]")),
                        "adversary_requests_generated": 0,
                        "adversary_moves_total_so_far": 0,
                        "adversary_request_moves_so_far": 0,
                        "adversary_prefill_deadlines_by_id": str(hist_row.get("adversary_prefill_deadlines_by_id", "{}")),
                        "sim_time": float(hist_row.get("sim_time", 0.0)),
                        "slo_violations": int(hist_row.get("slo_violations", 0)),
                        "total_lateness": float(hist_row.get("total_lateness", 0.0)),
                        "total_cost": float(hist_row.get("total_cost", 0.0)),
                        "cycle_label": cycle_label,
                        "phase": "history",
                        "step_index": int(hist_row.get("step_index", 0)),
                        "acting_player": str(hist_row.get("acting_player", "")),
                        "end_reason": "",
                    }
                )

        step_index = len(self.history_trace_by_game.get(int(root.game_id), []))

        # prefer_nonempty_adv = bool(getattr(self.cfg, "arena_require_request_generating_adversary", True)) and player == "adversary"
        # while turns < max_turns and adv_moves < max_adv and adv_moves_total < max_adv_total:
        while turns < max_turns and adv_request_moves < max_adv_req and adv_moves_total < max_adv_total:
            model = adversary_model if player == "adversary" else controller_model
            cur_depth = int(depth)
            prefer_nonempty_adv = bool(getattr(self.cfg, "arena_require_request_generating_adversary", True)) and player == "adversary"
            action, _best_idx, meta = self._pick_action_with_mcts(
                state=state,
                player=player,
                model=model,
                game_id=int(root.game_id),
                root_id=int(root.root_id + 10000 * cycle_id),
                root_depth=int(cur_depth),
                prefer_nonempty_adversary=prefer_nonempty_adv,
            )
            if action is None:
                end_reason = "terminal_before_adv_budget"
                break

            acting_player = str(player)
            generated_now = 0

            before_ids: set[int] = set()
            if acting_player == "adversary":
                before_ids = set(self.env._build_request_lookup(state.simulator).keys())
                state = self.env.apply_adversary_action_only(state, action, inplace=True)
                player = "controller"
                # adv_moves += 1
                adv_moves_total += 1
                # if int(meta.get("adversary_requests_generated", 0)) > 0:
                #     adv_request_moves += 1
                generated_now = int(meta.get("adversary_requests_generated", 0))
                if generated_now > 0:
                    adv_request_moves += 1

                adv_deadlines_json = self._adversary_new_prefill_deadlines_json(
                    acting_player=acting_player,
                    state_after=state,
                    before_request_ids=before_ids,
                )
            else:
                state = self.env.apply_controller_action_only(state, action, inplace=True)
                player = "adversary"
                adv_deadlines_json = "{}"

            depth += 1
            turns += 1
            step_index += 1

            if step_logger is not None:
                row = self._state_row(state, adversary_prefill_deadlines_by_id=adv_deadlines_json)
                step_logger.log_step(
                    {
                        "game_id": int(root.game_id),
                        "history_length": int(root.hops),
                        "best_model_player": str(best_model_player),
                        "candidate_model_player": str(candidate_model_player),
                        "root_id": int(root.root_id),
                        "root_depth": int(cur_depth),
                        "root_node_id": meta.get("root_node_id"),
                        "root_player": acting_player,
                        "model_root_value_controller": meta.get("model_root_value_controller"),
                        "model_root_prior_json": meta.get("model_root_prior_json", "[]"),
                        "mcts_root_value_controller": meta.get("mcts_root_value_controller"),
                        "mcts_root_prior_json": meta.get("mcts_root_prior_json", "[]"),
                        "best_action_index": meta.get("best_action_index"),
                        "best_action_repr": meta.get("best_action_repr", ""),
                        "requests_in_system": row["requests_in_system"],
                        "state_waiting_ids": row["state_waiting_ids"],
                        "state_completed_request_ids": row["state_completed_request_ids"],
                        # "adversary_requests_generated": int(meta.get("adversary_requests_generated", 0)) if acting_player == "adversary" else 0,
                        "adversary_requests_generated": int(generated_now),
                        "adversary_prefill_deadlines_by_id": row["adversary_prefill_deadlines_by_id"],
                        "sim_time": row["sim_time"],
                        "slo_violations": row["slo_violations"],
                        "total_lateness": row["total_lateness"],
                        "total_cost": row["total_cost"],
                        "cycle_label": cycle_label,
                        "phase": "arena",
                        "step_index": int(step_index),
                        "acting_player": acting_player,
                        "adversary_moves_total_so_far": int(adv_moves_total),
                        "adversary_request_moves_so_far": int(adv_request_moves),
                        "end_reason": "",
                    }
                )

        # if turns >= max_turns and end_reason == "unknown":
        #     end_reason = "max_total_turns"
        if end_reason == "unknown":
            if adv_request_moves >= max_adv_req:
                end_reason = "adv_request_budget_reached"
            elif adv_moves_total >= max_adv_total:
                end_reason = "max_adversary_turns_reached"
            elif turns >= max_turns:
                end_reason = "max_total_turns"

        while turns < max_turns and cleanup_steps < max_cleanup and self._has_prefill_pending(state):
            cur_depth = int(depth)

            if player == "adversary":
                before_ids = set(self.env._build_request_lookup(state.simulator).keys())
                state = self.env.apply_adversary_action_only(
                    state,
                    AdversaryAction(requests=[], stop_decode_ids=[]),
                    inplace=True,
                )
                acting_player = "adversary"
                player = "controller"
                depth += 1
                turns += 1
                step_index += 1

                adv_deadlines_json = self._adversary_new_prefill_deadlines_json(
                    acting_player=acting_player,
                    state_after=state,
                    before_request_ids=before_ids,
                )

                if step_logger is not None:
                    row = self._state_row(state, adversary_prefill_deadlines_by_id=adv_deadlines_json)
                    step_logger.log_step(
                        {
                            "game_id": int(root.game_id),
                            "history_length": int(root.hops),
                            "best_model_player": str(best_model_player),
                            "candidate_model_player": str(candidate_model_player),
                            "root_id": int(root.root_id),
                            "root_depth": int(cur_depth),
                            "root_node_id": None,
                            "root_player": acting_player,
                            "model_root_value_controller": None,
                            "model_root_prior_json": "[]",
                            "mcts_root_value_controller": None,
                            "mcts_root_prior_json": "[]",
                            "best_action_index": None,
                            "best_action_repr": "AdversaryAction(requests=[], stop_decode_ids=[])",
                            "requests_in_system": row["requests_in_system"],
                            "state_waiting_ids": row["state_waiting_ids"],
                            "state_completed_request_ids": row["state_completed_request_ids"],
                            "adversary_requests_generated": 0,
                            "adversary_moves_total_so_far": int(adv_moves_total),
                            "adversary_request_moves_so_far": int(adv_request_moves),
                            "adversary_prefill_deadlines_by_id": row["adversary_prefill_deadlines_by_id"],
                            "sim_time": row["sim_time"],
                            "slo_violations": row["slo_violations"],
                            "total_lateness": row["total_lateness"],
                            "total_cost": row["total_cost"],
                            "cycle_label": cycle_label,
                            "phase": "cleanup",
                            "step_index": int(step_index),
                            "acting_player": acting_player,
                            "end_reason": "",
                        }
                    )
                continue

            action, _best_idx, meta = self._pick_action_with_mcts(
                state=state,
                player="controller",
                model=controller_model,
                game_id=int(root.game_id),
                root_id=int(root.root_id + 10000 * cycle_id + 1),
                root_depth=int(cur_depth),
                prefer_nonempty_adversary=False,  # no need to prefer nonempty adversary actions during cleanup
            )
            if action is None:
                end_reason = "terminal_during_cleanup"
                break

            state = self.env.apply_controller_action_only(state, action, inplace=True)
            acting_player = "controller"
            player = "adversary"
            depth += 1
            turns += 1
            cleanup_steps += 1
            step_index += 1

            if step_logger is not None:
                row = self._state_row(state, adversary_prefill_deadlines_by_id="{}")
                step_logger.log_step(
                    {
                        "game_id": int(root.game_id),
                        "history_length": int(root.hops),
                        "best_model_player": str(best_model_player),
                        "candidate_model_player": str(candidate_model_player),
                        "root_id": int(root.root_id),
                        "root_depth": int(cur_depth),
                        "root_node_id": meta.get("root_node_id"),
                        "root_player": acting_player,
                        "model_root_value_controller": meta.get("model_root_value_controller"),
                        "model_root_prior_json": meta.get("model_root_prior_json", "[]"),
                        "mcts_root_value_controller": meta.get("mcts_root_value_controller"),
                        "mcts_root_prior_json": meta.get("mcts_root_prior_json", "[]"),
                        "best_action_index": meta.get("best_action_index"),
                        "best_action_repr": meta.get("best_action_repr", ""),
                        "requests_in_system": row["requests_in_system"],
                        "state_waiting_ids": row["state_waiting_ids"],
                        "state_completed_request_ids": row["state_completed_request_ids"],
                        "adversary_requests_generated": 0,
                        "adversary_moves_total_so_far": int(adv_moves_total),
                        "adversary_request_moves_so_far": int(adv_request_moves),
                        "adversary_prefill_deadlines_by_id": row["adversary_prefill_deadlines_by_id"],
                        "sim_time": row["sim_time"],
                        "slo_violations": row["slo_violations"],
                        "total_lateness": row["total_lateness"],
                        "total_cost": row["total_cost"],
                        "cycle_label": cycle_label,
                        "phase": "cleanup",
                        "step_index": int(step_index),
                        "acting_player": acting_player,
                        "end_reason": "",
                    }
                )

        # if end_reason == "unknown":
        #     if not self._has_prefill_pending(state):
        #         end_reason = "prefill_drained"
        #     elif cleanup_steps >= max_cleanup:
        #         end_reason = "cleanup_budget_exhausted"
        #     elif turns >= max_turns:
        #         end_reason = "max_total_turns"
        #     else:
        #         end_reason = "finished"

        if end_reason == "unknown":
            if not self._has_prefill_pending(state):
                end_reason = "prefill_drained"
            elif cleanup_steps >= max_cleanup:
                end_reason = "cleanup_budget_exhausted"
            elif turns >= max_turns:
                end_reason = "max_total_turns"
            else:
                end_reason = "finished"

        final_row = self._state_row(state, adversary_prefill_deadlines_by_id="{}")
        if step_logger is not None:
            step_logger.log_step(
                {
                    "game_id": int(root.game_id),
                    "history_length": int(root.hops),
                    "best_model_player": str(best_model_player),
                    "candidate_model_player": str(candidate_model_player),
                    "root_id": int(root.root_id),
                    "root_depth": int(depth),
                    "root_node_id": None,
                    "root_player": "",
                    "model_root_value_controller": None,
                    "model_root_prior_json": "[]",
                    "best_action_index": None,
                    "best_action_repr": "",
                    "requests_in_system": final_row["requests_in_system"],
                    "state_waiting_ids": final_row["state_waiting_ids"],
                    "state_completed_request_ids": final_row["state_completed_request_ids"],
                    "adversary_requests_generated": 0,
                    "adversary_moves_total_so_far": int(adv_moves_total),
                    "adversary_request_moves_so_far": int(adv_request_moves),
                    "adversary_prefill_deadlines_by_id": final_row["adversary_prefill_deadlines_by_id"],
                    "sim_time": final_row["sim_time"],
                    "slo_violations": final_row["slo_violations"],
                    "total_lateness": final_row["total_lateness"],
                    "total_cost": final_row["total_cost"],
                    "cycle_label": cycle_label,
                    "phase": "end",
                    "step_index": int(step_index + 1),
                    "acting_player": "",
                    "end_reason": str(end_reason),
                }
            )

        return {
            "violations": int(final_row["slo_violations"]),
            "total_lateness": float(final_row["total_lateness"]),
            "total_cost": float(final_row["total_cost"]),
            "turns": int(turns),
            "cleanup_steps": int(cleanup_steps),
            "end_reason": str(end_reason),
            "best_model_player": str(best_model_player),
            "candidate_model_player": str(candidate_model_player),
            "cycle_label": str(cycle_label),
            "adversary_moves": int(adv_moves_total),  # keep for compatibility
            "adversary_moves_total": int(adv_moves_total),
            "adversary_request_moves": int(adv_request_moves),
        }



    def evaluate_arena(
        self,
        *,
        candidate_model: Any,
        best_model: Any,
        generation_logger: Optional[EvalArenaGenerationLogger] = None,
    ) -> Dict[str, Any]:
        if not self.eval_roots:
            raise RuntimeError("No eval_roots loaded. Build/load roots before arena evaluation.")

        candidate_points = 0.0
        best_points = 0.0
        cand_adv_cost_sum = 0.0
        best_adv_cost_sum = 0.0
        per_root: List[Dict[str, Any]] = []

        # TODO: Condense the code here so that we don't have to duplicate the logic for parallel vs. single-process evaluation. We can still have a worker function that runs the cycle and returns results, but we can call it directly in the single-process case instead of spawning processes.
        
        num_procs = int(getattr(self.cfg, "arena_num_processes", 1))
        if num_procs > 1 and len(self.eval_roots) > 1:

            if not self._sim_cli_args:
                raise RuntimeError("arena_num_processes>1 requires sim_cli_args on evaluator")

            sim_cli_args = list(self._sim_cli_args)
            assume_infinite_kv = bool(
                getattr(getattr(self.env._base._config.cluster_config, "cache_config", None), "assume_infinite_kv", False))

            ctx = mp.get_context("spawn")
            n = len(self.eval_roots)
            ranges = _split_even_ranges(n, num_procs)

            # serialize roots for workers
            serialized: List[Dict[str, Any]] = []
            for i, r in enumerate(self.eval_roots):
                serialized.append(
                    {
                        "global_root_index": int(i),
                        "game_id": int(r.game_id),
                        "root_id": int(r.root_id),
                        "history_hops": int(r.hops),
                        "seed": int(r.seed),
                        "player_to_act": str(r.player_to_act),
                        "depth": int(r.depth),
                        "snapshot": r.state.simulator.snapshot_state(),
                        "stats": r.state.stats.clone(),
                    }
                )

            env_kind = "virtual" if self.env.__class__.__name__ == "VirtualVidurMCTSEnvironment" else "standard"
            sim_cfg = self.env._base._config
            constraints = self.env._constraints
            explore_cfg = self.env._cfg

            candidate_sd = {k: v.detach().cpu() for k, v in candidate_model.state_dict().items()}
            best_sd = {k: v.detach().cpu() for k, v in best_model.state_dict().items()}

            num_actions_controller = int(getattr(candidate_model, "num_actions_controller"))
            num_actions_adversary = int(getattr(candidate_model, "num_actions_adversary"))

            sampled_ids = list(getattr(generation_logger, "sampled_game_ids", set())) if generation_logger else []
            gen_dir = str(getattr(generation_logger, "gen_dir", Path(".")))
            flush_every = int(getattr(self.cfg, "debug_flush_every", 1))

            result_q = ctx.Queue()
            procs = []

            for wid, (lo, hi) in enumerate(ranges):
                chunk = serialized[lo:hi]
                chunk_game_ids = {int(x["game_id"]) for x in chunk}
                chunk_sampled = [gid for gid in sampled_ids if gid in chunk_game_ids]
                chunk_hist = {gid: self.history_trace_by_game.get(gid, []) for gid in chunk_game_ids}

                args = _ArenaWorkerArgs(
                    worker_id=int(wid),
                    env_kind=env_kind,
                    sim_cli_args=sim_cli_args,
                    assume_infinite_kv=assume_infinite_kv,
                    constraints=constraints,
                    explore_cfg=explore_cfg,
                    eval_cfg=self.cfg,
                    root_entries=chunk,
                    history_trace_by_game=chunk_hist,
                    candidate_state_dict=candidate_sd,
                    best_state_dict=best_sd,
                    num_actions_controller=num_actions_controller,
                    num_actions_adversary=num_actions_adversary,
                    sampled_game_ids=chunk_sampled,
                    gen_dir=gen_dir,
                    debug_flush_every=flush_every,
                )

                p = ctx.Process(target=_arena_worker_main, args=(result_q, args))
                p.start()
                procs.append(p)

            worker_msgs = [result_q.get() for _ in procs]
            for p in procs:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"arena worker died: pid={p.pid} exitcode={p.exitcode}")

            for msg in worker_msgs:
                if not bool(msg.get("ok", False)):
                    raise RuntimeError(f"arena worker failed: {msg.get('error', 'unknown error')}")

            all_per_root: List[Dict[str, Any]] = []
            for msg in worker_msgs:
                all_per_root.extend(list(msg["metrics"].get("per_root", [])))
            all_per_root.sort(key=lambda x: int(x.get("root_index", 0)))

            candidate_points = float(sum(float(x.get("candidate_points", 0.0)) for x in all_per_root))
            best_points = float(sum(float(x.get("best_points", 0.0)) for x in all_per_root))
            total_points = candidate_points + best_points

            cand_adv_cost_sum = float(sum(float(x.get("candidate_as_adv_cost", 0.0)) for x in all_per_root))
            best_adv_cost_sum = float(sum(float(x.get("best_as_adv_cost", 0.0)) for x in all_per_root))

            if generation_logger is not None:
                for row in all_per_root:
                    winner = str(row.get("winner", "tie"))
                    ca = float(row.get("candidate_as_adv_cost", 0.0))
                    cb = float(row.get("best_as_adv_cost", 0.0))
                    cycle_a = row.get("cycle_a", {})
                    cycle_b = row.get("cycle_b", {})
                    generation_logger.summary.log_row(
                        {
                            "game_id": int(row.get("game_id", 0)),
                            "history_length": int(row.get("hops", 0)),
                            "best_model_player": cycle_a.get("best_model_player", "controller"),
                            "candidate_model_player": cycle_a.get("candidate_model_player", "adversary"),
                            "slo_cost": ca,
                            "winner": winner,
                            "cycle_label": cycle_a.get("cycle_label", "candidate_as_adversary"),
                        }
                    )
                    generation_logger.summary.log_row(
                        {
                            "game_id": int(row.get("game_id", 0)),
                            "history_length": int(row.get("hops", 0)),
                            "best_model_player": cycle_b.get("best_model_player", "adversary"),
                            "candidate_model_player": cycle_b.get("candidate_model_player", "controller"),
                            "slo_cost": cb,
                            "winner": winner,
                            "cycle_label": cycle_b.get("cycle_label", "best_as_adversary"),
                        }
                    )

            n_roots = float(len(self.eval_roots))
            candidate_win_rate = (candidate_points / total_points) if total_points > 0 else 0.0
            passed = candidate_win_rate > float(self.cfg.arena_win_threshold)

            return {
                "num_eval_roots": n_roots,
                "candidate_points": candidate_points,
                "best_points": best_points,
                "total_points": float(total_points),
                "candidate_win_rate": float(candidate_win_rate),
                "arena_win_threshold": float(self.cfg.arena_win_threshold),
                "passed": bool(passed),
                "candidate_as_adv_mean_cost": float(cand_adv_cost_sum / max(n_roots, 1.0)),
                "best_as_adv_mean_cost": float(best_adv_cost_sum / max(n_roots, 1.0)),
                "per_root": all_per_root,
            }




        for i, root in enumerate(self.eval_roots):
            cycle_a = self._run_cycle(
                root=root,
                adversary_model=candidate_model,
                controller_model=best_model,
                cycle_id=(2 * i),
                cycle_label="candidate_as_adversary",
                best_model_player="controller",
                candidate_model_player="adversary",
                generation_logger=generation_logger,
            )

            cycle_b = self._run_cycle(
                root=root,
                adversary_model=best_model,
                controller_model=candidate_model,
                cycle_id=(2 * i + 1),
                cycle_label="best_as_adversary",
                best_model_player="adversary",
                candidate_model_player="controller",
                generation_logger=generation_logger,
            )

            ca = float(cycle_a["total_cost"])
            cb = float(cycle_b["total_cost"])
            cand_adv_cost_sum += ca
            best_adv_cost_sum += cb

            if ca > cb + 1e-9:
                cp, bp, winner = 1.0, 0.0, "candidate"
            elif cb > ca + 1e-9:
                cp, bp, winner = 0.0, 1.0, "best"
            else:
                tie = float(self.cfg.tie_points)
                cp, bp, winner = tie, tie, "tie"

            candidate_points += cp
            best_points += bp

            if generation_logger is not None:
                generation_logger.summary.log_row(
                    {
                        "game_id": int(root.game_id),
                        "history_length": int(root.hops),
                        "best_model_player": cycle_a["best_model_player"],
                        "candidate_model_player": cycle_a["candidate_model_player"],
                        "slo_cost": ca,
                        "winner": winner,
                        "cycle_label": cycle_a["cycle_label"],
                    }
                )
                generation_logger.summary.log_row(
                    {
                        "game_id": int(root.game_id),
                        "history_length": int(root.hops),
                        "best_model_player": cycle_b["best_model_player"],
                        "candidate_model_player": cycle_b["candidate_model_player"],
                        "slo_cost": cb,
                        "winner": winner,
                        "cycle_label": cycle_b["cycle_label"],
                    }
                )

            per_root.append(
                {
                    "root_index": int(i),
                    "game_id": int(root.game_id),
                    "hops": int(root.hops),
                    "seed": int(root.seed),
                    "candidate_as_adv_cost": ca,
                    "best_as_adv_cost": cb,
                    "winner": winner,
                    "candidate_points": cp,
                    "best_points": bp,
                    "cycle_a": cycle_a,
                    "cycle_b": cycle_b,
                }
            )

        total_points = candidate_points + best_points
        candidate_win_rate = (candidate_points / total_points) if total_points > 0 else 0.0
        passed = candidate_win_rate > float(self.cfg.arena_win_threshold)

        n = float(len(self.eval_roots))
        return {
            "num_eval_roots": n,
            "candidate_points": float(candidate_points),
            "best_points": float(best_points),
            "total_points": float(total_points),
            "candidate_win_rate": float(candidate_win_rate),
            "arena_win_threshold": float(self.cfg.arena_win_threshold),
            "passed": bool(passed),
            "candidate_as_adv_mean_cost": float(cand_adv_cost_sum / max(n, 1.0)),
            "best_as_adv_mean_cost": float(best_adv_cost_sum / max(n, 1.0)),
            "per_root": per_root,
        }


@dataclass
class FixedEvalHarness:
    evaluator: FixedEvalStatesEvaluator
    cpu_candidate_for_mcts: AlphaZeroModel
    cpu_best_for_mcts: AlphaZeroModel

    _simulator: Simulator
    # _mcts: VidurMCTS

    @staticmethod
    def _to_cpu_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in sd.items()}

    def sync_candidate_from_trainer(self, trainer: Trainer) -> None:
        sd = self._to_cpu_state_dict(trainer.model.state_dict())
        self.cpu_candidate_for_mcts.load_state_dict(sd, strict=True)
        self.cpu_candidate_for_mcts.eval()

    def sync_best_from_state_dict(self, best_state_dict: Dict[str, torch.Tensor]) -> None:
        sd = self._to_cpu_state_dict(best_state_dict)
        self.cpu_best_for_mcts.load_state_dict(sd, strict=True)
        self.cpu_best_for_mcts.eval()

    def run_arena(
        self,
        *,
        trainer: Trainer,
        best_state_dict: Dict[str, torch.Tensor],
        generation_logger: Optional[EvalArenaGenerationLogger] = None,
    ) -> Dict[str, Any]:
        self.sync_candidate_from_trainer(trainer)
        self.sync_best_from_state_dict(best_state_dict)
        return self.evaluator.evaluate_arena(
            candidate_model=self.cpu_candidate_for_mcts,
            best_model=self.cpu_best_for_mcts,
            generation_logger=generation_logger,
        )

    @classmethod
    def from_alpha_zero_cfg(
        cls,
        *,
        az_cfg: Any,
        eval_cfg: EvaluatorConfig,
        start_player: str = "adversary",
    ) -> "FixedEvalHarness":
        sim_group = getattr(az_cfg, "sim")
        constraints_group = getattr(az_cfg, "constraints")
        explore_group = getattr(az_cfg, "explore")
        model_group = getattr(az_cfg, "model")

        sim_cli_args = list(getattr(sim_group, "cli_args"))

        sim_cfg_eval = _configure_simulation_from_cli_args(sim_cli_args)
        setattr(sim_cfg_eval.cluster_config.cache_config, "assume_infinite_kv", True)
        simulator_eval = Simulator(sim_cfg_eval, register_atexit=False)

        slo_options_eval = RequestSLOOptions(
            prefill_slos=tuple(getattr(constraints_group, "prefill_slos")),
            decode_slos=tuple(getattr(constraints_group, "decode_slos")),
        )
        constraints_eval = MCTSConstraintConfig(
            maximum_qps=int(getattr(constraints_group, "maximum_qps")),
            min_request_tokens=int(getattr(constraints_group, "min_request_tokens")),
            max_request_tokens=int(getattr(constraints_group, "max_request_tokens")),
            interval_request_size=int(getattr(constraints_group, "interval_request_size")),
            request_slo_options=slo_options_eval,
            prefill_slowdown=float(getattr(constraints_group, "prefill_slowdown")),
            prefill_profile_path=str(getattr(constraints_group, "prefill_profile_path")),
        )

        explore_cfg_eval = MCTSExploreConfig(
            simulation_depth=int(getattr(explore_group, "simulation_depth")),
            simulation_random_tries=int(getattr(explore_group, "simulation_random_tries")),
            exploration_constant=float(getattr(explore_group, "exploration_constant")),
            max_branching=int(getattr(explore_group, "max_branching")),
            controller_budget_combs=int(getattr(explore_group, "controller_budget_combs")),
        )
        setattr(explore_cfg_eval, "controller_min_prior_threshold", float(getattr(explore_group, "controller_min_prior_threshold", 0.0)))
        setattr(explore_cfg_eval, "adversary_min_prior_threshold", float(getattr(explore_group, "adversary_min_prior_threshold", 0.0)))

        env_eval = VidurMCTSEnvironment(
            base_simulator=simulator_eval,
            constraints=constraints_eval,
            explore_cfg=explore_cfg_eval,
        )
        mcts_eval = VidurMCTS(
            env=env_eval,
            explore_cfg=explore_cfg_eval,
            log_path=None,
            tree_log_path=None,
            logger_flush_every=1,
        )

        evaluator = FixedEvalStatesEvaluator(env=env_eval, cfg=eval_cfg, mcts=mcts_eval, sim_cli_args=sim_cli_args)

        cpu_candidate_for_mcts = AlphaZeroModel(
            num_actions_controller=int(getattr(model_group, "num_actions_controller")),
            num_actions_adversary=int(getattr(model_group, "num_actions_adversary")),
        ).to(torch.device("cpu"))
        cpu_candidate_for_mcts.eval()

        cpu_best_for_mcts = AlphaZeroModel(
            num_actions_controller=int(getattr(model_group, "num_actions_controller")),
            num_actions_adversary=int(getattr(model_group, "num_actions_adversary")),
        ).to(torch.device("cpu"))
        cpu_best_for_mcts.eval()

        return cls(
            evaluator=evaluator,
            cpu_candidate_for_mcts=cpu_candidate_for_mcts,
            cpu_best_for_mcts=cpu_best_for_mcts,
            _simulator=simulator_eval,
        )


def build_eval_roots_payload_from_cfg(
    *,
    az_cfg: Any,
    eval_cfg: EvaluatorConfig,
    gen: int,
    start_player: str = "adversary",
) -> Dict[str, Any]:
    harness = FixedEvalHarness.from_alpha_zero_cfg(
        az_cfg=az_cfg,
        eval_cfg=eval_cfg,
        start_player=start_player,
    )
    return harness.evaluator.build_eval_roots_for_generation(
        gen=int(gen),
        start_player=start_player,
    )


def save_eval_roots_payload_from_cfg(
    *,
    az_cfg: Any,
    eval_cfg: EvaluatorConfig,
    gen: int,
    out_path: Path,
    start_player: str = "adversary",
) -> Path:
    payload = build_eval_roots_payload_from_cfg(
        az_cfg=az_cfg,
        eval_cfg=eval_cfg,
        gen=int(gen),
        start_player=start_player,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out
