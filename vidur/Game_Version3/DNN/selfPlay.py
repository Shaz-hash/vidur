from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Set, List, Iterator

import random
import torch

from ..config import GameVersion2Config
from ....environment import VidurMCTSState
from ....game_types import AdversaryAction, ControllerAction
from ..mctsDNN import VidurMCTS
from .history_root import HistoryRootGenerator
from .infer import build_model_inputs
from .replay_write import ReplayWriter, make_root_sample
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger, DNNMCTSRootSummaryLogger


def _mask_to_list(mask) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


def _one_hot_prior(mask: Sequence[bool], best_idx: int) -> list[float]:
    out = [0.0] * len(mask)
    if 0 <= int(best_idx) < len(mask) and bool(mask[int(best_idx)]):
        out[int(best_idx)] = 1.0
        return out

    valid = [i for i, ok in enumerate(mask) if ok]
    if not valid:
        return out

    p = 1.0 / float(len(valid))
    for i in valid:
        out[int(i)] = p
    return out


@dataclass(frozen=True)
class SingleRootRun:
    game_id: int = 0
    root_id: int = 0
    root_depth: int = 0
    root_player: str = "adversary"
    feature_version: int = 1
    root_node_id_override: int | None = None
    model_version: int = 0
    use_model_bootstrap: bool | None = None


@dataclass(frozen=True)
class RootSearchResult:
    mask_list: list[bool]
    mcts_prior: list[float]
    best_idx: int
    root_node_id: int
    mcts_value: float
    used_bootstrap: bool


@dataclass(frozen=True)
class PreparedHistoryRoot:
    root_state: VidurMCTSState
    root_player: str
    root_depth: int
    root_id: int
    root_node_id_override: int | None = None
    pre_controller_snapshot: Any | None = None
    pre_controller_stats: Any | None = None
    history_hops: int = 0
    history_signature: Any | None = None


class SelfPlayRunner:
    def __init__(
        self,
        *,
        env: Any,
        mcts: VidurMCTS,
        model,
        writer: ReplayWriter,
        eval_writer: Optional[ReplayWriter] = None,
        device_for_features: torch.device = torch.device("cpu"),
        game_v2_cfg: Optional[GameVersion2Config] = None,
    ) -> None:
        self.env = env
        self.mcts = mcts
        self.model = model
        self.writer = writer
        self.eval_writer = eval_writer
        self.device = device_for_features
        self._gv2_cfg = game_v2_cfg if game_v2_cfg is not None else getattr(env, "_gv2_cfg", None)

        iter_logger_native = None
        root_logger_native = None
        try:
            cfg = getattr(self.mcts, "_cfg", None)
            native_on = bool(getattr(cfg, "native_mcts_enabled", False))
            native_log_events = bool(getattr(cfg, "native_log_events", True))
            if native_on and native_log_events:
                it = getattr(self.mcts, "_iter_logger", None)
                it_path = getattr(it, "_path", None)
                if it_path is not None:
                    iter_logger_native = DNNMCTSIterationLogger(
                        str(it_path).replace(".csv", "_native.csv"),
                        flush_every=int(getattr(it, "_flush_every", 1)),
                    )

                rt = getattr(self.mcts, "_root_logger", None)
                rt_path = getattr(rt, "_path", None)
                if rt_path is not None:
                    root_logger_native = DNNMCTSRootSummaryLogger(
                        str(rt_path).replace(".csv", "_native.csv"),
                        flush_every=int(getattr(rt, "_flush_every", 1)),
                    )
        except Exception:
            iter_logger_native = None
            root_logger_native = None

        self.history = HistoryRootGenerator(
            env=self.env,
            iter_logger=getattr(self.mcts, "_iter_logger", None),
            root_logger=getattr(self.mcts, "_root_logger", None),
            iter_logger_native=iter_logger_native,
            root_logger_native=root_logger_native,
        )
        self.last_run_stats: dict[str, Any] = {}

    def _sample_actions_readonly(self, state: VidurMCTSState, player: str):
        probe = state.fork(flag=False)
        if player == "controller":
            return self.env.sample_controller_actions(probe)
        return self.env.sample_adversary_actions(probe)

    def _build_root_decision_state_for_adversary(
        self,
        *,
        current_state: VidurMCTSState,
        pre_controller_snapshot: Any | None,
        pre_controller_stats: Any | None,
    ) -> tuple[VidurMCTSState, Set[int]]:
        if pre_controller_snapshot is None or pre_controller_stats is None:
            return current_state, set()

        get_src = getattr(self.env, "_v2_missed_adv_source", None)
        if not callable(get_src):
            return current_state, set()

        try:
            miss_src = int(get_src(current_state))
        except Exception:
            return current_state, set()

        if miss_src != 1:
            return current_state, set()

        clone_fn = getattr(self.env, "clone_state_from_snapshot", None)
        if not callable(clone_fn):
            return current_state, set()

        try:
            decision_state = clone_fn(pre_controller_snapshot, pre_controller_stats)
        except Exception:
            return current_state, set()

        try:
            tick = float(self.env._v2_current_adv_tick(current_state))
            t_dec = float(decision_state.simulator._time)
            if t_dec + 1e-9 < tick:
                decision_state.simulator._set_time(tick)
        except Exception:
            pass

        try:
            replay_ids = set(
                int(k) for k in self.env._build_request_lookup(
                    decision_state.simulator, state=decision_state
                ).keys()
            )
            live_ids = set(
                int(k) for k in self.env._build_request_lookup(
                    current_state.simulator, state=current_state
                ).keys()
            )
        except Exception:
            return decision_state, set()

        forbidden = replay_ids - live_ids
        if not forbidden:
            return decision_state, set()

        s = decision_state.stats
        s.active_request_ids = {int(rid) for rid in s.active_request_ids if int(rid) not in forbidden}

        for rid in forbidden:
            rid = int(rid)
            s.decode_tokens_counted.pop(rid, None)
            s.decode_next_deadline_by_id.pop(rid, None)
            s.per_request_prefill_lateness.pop(rid, None)
            s.per_request_decode_lateness.pop(rid, None)
            s.violated_request_ids.discard(rid)
            s.dropped_request_ids.discard(rid)
            s.stopped_decode_request_ids.discard(rid)
            s.prefill_lateness_finalized.discard(rid)

        return decision_state, forbidden

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
            game_id=-1,
            root_id=-1,
            log_node_id=0,
            log_parent_id=None,
            max_hops=int(max_hops),
            log_steps=False,
        )
        return state, player, depth

    def _coerce_prepared_root(self, obj: Any, default_root_id: int) -> PreparedHistoryRoot:
        if isinstance(obj, PreparedHistoryRoot):
            return obj

        if isinstance(obj, dict):
            return PreparedHistoryRoot(
                root_state=obj["root_state"],
                root_player=str(obj.get("root_player", "adversary")),
                root_depth=int(obj.get("root_depth", 0)),
                root_id=int(obj.get("root_id", default_root_id)),
                root_node_id_override=obj.get("root_node_id_override", None),
                pre_controller_snapshot=obj.get("pre_controller_snapshot", None),
                pre_controller_stats=obj.get("pre_controller_stats", None),
                history_hops=int(obj.get("history_hops", 0)),
                history_signature=obj.get("history_signature", None),
            )

        return PreparedHistoryRoot(
            root_state=getattr(obj, "root_state"),
            root_player=str(getattr(obj, "root_player", "adversary")),
            root_depth=int(getattr(obj, "root_depth", 0)),
            root_id=int(getattr(obj, "root_id", default_root_id)),
            root_node_id_override=getattr(obj, "root_node_id_override", None),
            pre_controller_snapshot=getattr(obj, "pre_controller_snapshot", None),
            pre_controller_stats=getattr(obj, "pre_controller_stats", None),
            history_hops=int(getattr(obj, "history_hops", 0)),
            history_signature=getattr(obj, "history_signature", None),
        )

    def _prepare_history_roots(
        self,
        *,
        game_id: int,
        num_roots: int,
        start_root_id: int,
        start_root_depth: int,
        start_player: str,
        initial_state: Optional[VidurMCTSState],
        history_nontrivial_hops: int,
        history_hops_min: int,
        history_hops_max: int,
        history_seed: int,
        history_max_total_steps: int,
        log_history_rows: bool,
        history_seen_signatures: Optional[Sequence[Any]] = None,
    ) -> List[PreparedHistoryRoot]:
        if hasattr(self.history, "generate_roots_batch"):
            try:
                raw = self.history.generate_roots_batch(
                    initial_state=initial_state,
                    start_player=str(start_player),
                    start_depth=int(start_root_depth),
                    num_roots=int(num_roots),
                    game_id=int(game_id),
                    start_root_id=int(start_root_id),
                    nontrivial_hops=int(history_nontrivial_hops),
                    min_history_hops=int(history_hops_min),
                    max_history_hops=int(history_hops_max),
                    seed=int(history_seed),
                    max_total_steps=int(history_max_total_steps),
                    log_history=bool(log_history_rows),
                    initial_seen_signatures=history_seen_signatures,
                )
                return [
                    self._coerce_prepared_root(x, int(start_root_id) + i)
                    for i, x in enumerate(raw)
                ]
            except Exception:
                pass

        roots: List[PreparedHistoryRoot] = []
        hops_rng = random.Random(int(history_seed))
        hop_lo = int(min(history_hops_min, history_hops_max))
        hop_hi = int(max(history_hops_min, history_hops_max))
        for k in range(int(num_roots)):
            root_id = int(start_root_id) + int(k)
            state = initial_state.fork(flag=False) if initial_state is not None else self.env.initial_state()
            player = str(start_player)
            depth = int(start_root_depth)
            next_log_node_id = int(getattr(self.mcts, "_node_counter", 0))
            last_log_node_id: int | None = None

            hops = int(hops_rng.randint(hop_lo, hop_hi))
            if hops > 0:
                state, player, depth, next_log_node_id, last_log_node_id = self.history.generate_history_root(
                    state,
                    player,
                    depth,
                    nontrivial_hops=hops,
                    game_id=int(game_id),
                    root_id_for_logs=int(root_id),
                    log_history=bool(log_history_rows),
                    log_node_id_start=next_log_node_id,
                    log_parent_id_start=last_log_node_id,
                    seed=int(history_seed) + int(k),
                    max_total_steps=int(history_max_total_steps),
                )
                self.mcts._node_counter = max(int(self.mcts._node_counter), int(next_log_node_id))

            roots.append(
                PreparedHistoryRoot(
                    root_state=state,
                    root_player=str(player),
                    root_depth=int(depth),
                    root_id=int(root_id),
                    root_node_id_override=(None if last_log_node_id is None else int(last_log_node_id)),
                    pre_controller_snapshot=None,
                    pre_controller_stats=None,
                    history_hops=int(hops),
                )
            )

        return roots

    def _iter_prepared_history_root_batches(
        self,
        *,
        game_id: int,
        num_roots: int,
        start_root_id: int,
        start_root_depth: int,
        start_player: str,
        initial_state: Optional[VidurMCTSState],
        history_nontrivial_hops: int,
        history_hops_min: int,
        history_hops_max: int,
        history_seed: int,
        history_max_total_steps: int,
        log_history_rows: bool,
        root_batch_size: int,
        history_seen_signatures: Optional[Sequence[Any]] = None,
        shared_history_signatures: Any | None = None,
        shared_history_lock: Any | None = None,
        allow_duplicate_history_fallback: bool = True,
    ) -> Iterator[list[PreparedHistoryRoot]]:
        batch_size = max(1, int(root_batch_size))
        if hasattr(self.history, "generate_roots_batch_iter"):
            try:
                raw_batches = self.history.generate_roots_batch_iter(
                    initial_state=initial_state,
                    start_player=str(start_player),
                    start_depth=int(start_root_depth),
                    num_roots=int(num_roots),
                    game_id=int(game_id),
                    start_root_id=int(start_root_id),
                    nontrivial_hops=int(history_nontrivial_hops),
                    min_history_hops=int(history_hops_min),
                    max_history_hops=int(history_hops_max),
                    seed=int(history_seed),
                    max_total_steps=int(history_max_total_steps),
                    log_history=bool(log_history_rows),
                    batch_size=int(batch_size),
                    initial_seen_signatures=history_seen_signatures,
                    shared_seen_signatures=shared_history_signatures,
                    shared_seen_lock=shared_history_lock,
                    allow_duplicate_fallback=bool(allow_duplicate_history_fallback),
                )
                for batch_idx, raw_batch in enumerate(raw_batches):
                    offset = int(batch_idx) * int(batch_size)
                    yield [
                        self._coerce_prepared_root(x, int(start_root_id) + offset + i)
                        for i, x in enumerate(raw_batch)
                    ]
                return
            except Exception:
                pass

        roots = self._prepare_history_roots(
            game_id=int(game_id),
            num_roots=int(num_roots),
            start_root_id=int(start_root_id),
            start_root_depth=int(start_root_depth),
            start_player=str(start_player),
            initial_state=initial_state,
            history_nontrivial_hops=int(history_nontrivial_hops),
            history_hops_min=int(history_hops_min),
            history_hops_max=int(history_hops_max),
            history_seed=int(history_seed),
            history_max_total_steps=int(history_max_total_steps),
            log_history_rows=bool(log_history_rows),
        )
        for start in range(0, len(roots), int(batch_size)):
            yield roots[start : start + int(batch_size)]

    def _root_signature(self, state: VidurMCTSState, player: str, depth: int) -> tuple[Any, ...]:
        desc = self.env.describe_state(state)
        active_ids = tuple(int(x) for x in (desc.get("active_request_ids") or []))
        completed_ids = tuple(int(x) for x in (desc.get("completed_request_ids") or []))
        return (
            str(player),
            int(depth),
            round(float(desc.get("sim_time", 0.0)), 9),
            bool(desc.get("pending_adv_tick", False)),
            int(desc.get("decode_credit_balance", 0)),
            active_ids,
            completed_ids,
        )

    def run_single_root(self, cfg: SingleRootRun, root_state: Optional[VidurMCTSState] = None) -> RootSearchResult:
        state = root_state or self.env.initial_state()
        search_state = state.fork(flag=False)
        use_model_bootstrap = (
            bool(cfg.use_model_bootstrap)
            if cfg.use_model_bootstrap is not None
            else bool(int(cfg.model_version) > 0)
        )

        out = self.mcts.search_dnn(
            dnn_model=self.model,
            rootState=search_state,
            root_player=cfg.root_player,
            game_id=cfg.game_id,
            root_id=cfg.root_id,
            root_node_id_override=cfg.root_node_id_override,
            root_depth=cfg.root_depth,
            model_version=int(cfg.model_version),
            use_model_bootstrap=bool(use_model_bootstrap),
            one_step_value_mode=True,
        )

        if hasattr(out, "valid_mask") and hasattr(out, "best_action_value"):
            mask_list = [bool(x) for x in list(getattr(out, "valid_mask", []))]
            best_idx_raw = getattr(out, "best_action_index", None)
            best_idx = int(best_idx_raw) if best_idx_raw is not None else -1
            mcts_prior = _one_hot_prior(mask_list, best_idx)
            mcts_value = float(getattr(out, "best_action_value", 0.0))
            root_node_id = int(getattr(out, "root_node_id", -1))
            used_bootstrap = bool(getattr(out, "used_bootstrap", False))
            return RootSearchResult(
                mask_list=mask_list,
                mcts_prior=mcts_prior,
                best_idx=int(best_idx),
                root_node_id=int(root_node_id),
                mcts_value=float(mcts_value),
                used_bootstrap=used_bootstrap,
            )

        # fallback for legacy output
        root = self.mcts._root
        if root is None:
            return RootSearchResult(
                mask_list=[],
                mcts_prior=[],
                best_idx=-1,
                root_node_id=-1,
                mcts_value=0.0,
                used_bootstrap=bool(use_model_bootstrap),
            )

        _, mask = self._sample_actions_readonly(state, cfg.root_player)
        mask_list = _mask_to_list(mask)
        best_idx = -1
        if root.children:
            best_idx = max(root.children.keys(), key=lambda i: int(getattr(root.children[i], "visits", 0)))
        mcts_prior = _one_hot_prior(mask_list, int(best_idx))
        mcts_value = float(root.mean_value())

        return RootSearchResult(
            mask_list=mask_list,
            mcts_prior=mcts_prior,
            best_idx=int(best_idx),
            root_node_id=int(getattr(root, "node_id", -1)),
            mcts_value=float(mcts_value),
            used_bootstrap=bool(use_model_bootstrap),
        )

    def _pick_writer(self, *, is_eval: bool) -> ReplayWriter:
        if is_eval and self.eval_writer is not None:
            return self.eval_writer
        return self.writer

    def run_n_roots(
        self,
        *,
        game_id: int,
        num_roots: int,
        max_batch_size: int = 72,
        start_root_id: int = 0,
        start_root_depth: int = 0,
        start_player: str = "adversary",
        feature_version: int = 1,
        initial_state: Optional[VidurMCTSState] = None,
        action_seed_base: int = 0,
        history_nontrivial_hops: Optional[int] = None,
        history_hops_min: Optional[int] = None,
        history_hops_max: Optional[int] = None,
        history_seed: Optional[int] = None,
        max_forced_hops_per_root: Optional[int] = None,
        history_max_total_steps: Optional[int] = None,
        log_history_rows: Optional[bool] = None,
        history_root_batch_size: int = 64,
        progress_prefix: str = "",
        generation: int | None = None,
        model_version: int = 0,
        eval_split_ratio: float = 0.0,
        eval_split_seed: int = 0,
        history_seen_signatures: Optional[Sequence[Any]] = None,
        shared_history_signatures: Any | None = None,
        shared_history_lock: Any | None = None,
        allow_duplicate_history_fallback: bool = True,
    ) -> VidurMCTSState:
        del action_seed_base
        del max_batch_size

        hist_hops, hist_seed, max_forced_hops, hist_max_total_steps, hist_log_rows = self._resolve_history_settings(
            history_nontrivial_hops=history_nontrivial_hops,
            history_seed=history_seed,
            max_forced_hops_per_root=max_forced_hops_per_root,
            history_max_total_steps=history_max_total_steps,
            log_history_rows=log_history_rows,
        )
        hist_hops_min = int(hist_hops if history_hops_min is None else history_hops_min)
        hist_hops_max = int(hist_hops if history_hops_max is None else history_hops_max)
        if hist_hops_max < hist_hops_min:
            hist_hops_max = int(hist_hops_min)

        split_ratio = max(0.0, min(1.0, float(eval_split_ratio)))
        split_rng = random.Random(int(eval_split_seed))
        unique_root_sigs: set[tuple[Any, ...]] = set()
        emitted_history_signatures: list[Any] = []
        roots_generated = 0
        processed_batches = 0

        controller_train_samples = 0
        controller_eval_samples = 0
        adversary_train_samples = 0
        adversary_eval_samples = 0
        bootstrap_generation = int(model_version) if generation is None else int(generation)
        use_model_bootstrap = bool(int(bootstrap_generation) > 0)

        last_state = initial_state if initial_state is not None else self.env.initial_state()
        progress_tag = str(progress_prefix).strip()
        if progress_tag:
            print(
                f"{progress_tag} self-play starting: roots={int(num_roots)}, "
                f"hop_range=[{int(hist_hops_min)}, {int(hist_hops_max)}], "
                f"root_batch_size={int(history_root_batch_size)}, eval_ratio={float(split_ratio):.3f}, "
                f"use_model_bootstrap={bool(use_model_bootstrap)}",
                flush=True,
            )

        for prepared_batch in self._iter_prepared_history_root_batches(
            game_id=int(game_id),
            num_roots=int(num_roots),
            start_root_id=int(start_root_id),
            start_root_depth=int(start_root_depth),
            start_player=str(start_player),
            initial_state=initial_state,
            history_nontrivial_hops=int(hist_hops),
            history_hops_min=int(hist_hops_min),
            history_hops_max=int(hist_hops_max),
            history_seed=int(hist_seed),
            history_max_total_steps=int(hist_max_total_steps),
            log_history_rows=bool(hist_log_rows),
            root_batch_size=int(history_root_batch_size),
            history_seen_signatures=history_seen_signatures,
            shared_history_signatures=shared_history_signatures,
            shared_history_lock=shared_history_lock,
            allow_duplicate_history_fallback=bool(allow_duplicate_history_fallback),
        ):
            processed_batches += 1
            for pr in prepared_batch:
                roots_generated += 1
                root_state = pr.root_state
                root_player = str(pr.root_player)
                root_depth = int(pr.root_depth)
                root_id = int(pr.root_id)

                root_state, root_player, root_depth = self._advance_to_branching_root(
                    root_state,
                    root_player,
                    root_depth,
                    max_hops=int(max_forced_hops),
                )
                unique_root_sigs.add(self._root_signature(root_state, root_player, root_depth))
                if pr.history_signature is not None:
                    emitted_history_signatures.append(pr.history_signature)

                search_root_state = root_state
                if root_player == "adversary":
                    search_root_state, _ = self._build_root_decision_state_for_adversary(
                        current_state=root_state,
                        pre_controller_snapshot=pr.pre_controller_snapshot,
                        pre_controller_stats=pr.pre_controller_stats,
                    )

                res = self.run_single_root(
                    SingleRootRun(
                        game_id=int(game_id),
                        root_id=int(root_id),
                        root_depth=int(root_depth),
                        root_player=str(root_player),
                        feature_version=int(feature_version),
                        root_node_id_override=pr.root_node_id_override,
                        model_version=int(model_version),
                        use_model_bootstrap=bool(use_model_bootstrap),
                    ),
                    root_state=search_root_state,
                )

                base_inputs = build_model_inputs(search_root_state, root_player, self.device)
                inputs = replace(
                    base_inputs,
                    action_mask=torch.tensor(
                        list(res.mask_list),
                        dtype=torch.bool,
                        device=base_inputs.global_features.device,
                    ).unsqueeze(0),
                )

                sample = make_root_sample(
                    feature_version=int(feature_version),
                    game_id=int(game_id),
                    root_id=int(root_id),
                    root_node_id=int(res.root_node_id),
                    root_depth=int(root_depth),
                    player=str(root_player),
                    model_inputs=inputs,
                    action_mask=list(res.mask_list),
                    mcts_policy=list(res.mcts_prior),
                    mcts_value_controller=float(res.mcts_value),
                    meta={
                        "best_action_index": int(res.best_idx),
                        "search_mode": "depth1_value_backup",
                        "used_bootstrap": bool(res.used_bootstrap),
                        "bootstrap_generation": int(bootstrap_generation),
                        "use_model_bootstrap": bool(use_model_bootstrap),
                        "model_version": int(model_version),
                        "history_hops": int(pr.history_hops),
                    },
                )

                is_eval = bool(split_rng.random() < split_ratio)
                self._pick_writer(is_eval=is_eval).add(sample)
                if str(root_player) == "controller":
                    if is_eval:
                        controller_eval_samples += 1
                    else:
                        controller_train_samples += 1
                else:
                    if is_eval:
                        adversary_eval_samples += 1
                    else:
                        adversary_train_samples += 1

                last_state = root_state
                # if hasattr(self.mcts, "clear_search_state"):
                #     self.mcts.clear_search_state(drop_scratch=True)

                if hasattr(self.mcts, "clear_search_state"):
                    # Keep the reusable scratch simulator alive across roots.
                    # Tree state is still cleared every root.
                    self.mcts.clear_search_state(drop_scratch=False)

                if progress_tag and (int(roots_generated) % 400 == 0):
                    train_total = int(controller_train_samples + adversary_train_samples)
                    eval_total = int(controller_eval_samples + adversary_eval_samples)
                    print(
                        f"{progress_tag} samples written: roots={int(roots_generated)}/{int(num_roots)}, "
                        f"train={train_total}, eval={eval_total}, unique_roots={int(len(unique_root_sigs))}",
                        flush=True,
                    )

            if progress_tag and (int(processed_batches) % 4 == 0):
                batch_end_root = int(roots_generated)
                batch_start_root = max(1, batch_end_root - len(prepared_batch) + 1) if prepared_batch else batch_end_root
                print(
                    f"{progress_tag} root batch {int(processed_batches)} completed: "
                    f"batch_size={int(len(prepared_batch))}, roots={batch_start_root}-{batch_end_root}/"
                    f"{int(num_roots)}, unique_roots={int(len(unique_root_sigs))}",
                    flush=True,
                )

        self.last_run_stats = {
            "num_roots_requested": int(num_roots),
            "num_roots_generated": int(roots_generated),
            "num_unique_roots": int(len(unique_root_sigs)),
            "controller_train_samples": int(controller_train_samples),
            "controller_eval_samples": int(controller_eval_samples),
            "adversary_train_samples": int(adversary_train_samples),
            "adversary_eval_samples": int(adversary_eval_samples),
            "train_samples_total": int(controller_train_samples + adversary_train_samples),
            "eval_samples_total": int(controller_eval_samples + adversary_eval_samples),
            "history_hops_min": int(hist_hops_min),
            "history_hops_max": int(hist_hops_max),
            "eval_split_ratio": float(split_ratio),
            "generation": int(bootstrap_generation),
            "use_model_bootstrap": bool(use_model_bootstrap),
            "history_signatures": emitted_history_signatures,
            "num_history_roots_emitted": int(len(emitted_history_signatures)),
        }

        if progress_tag:
            print(
                f"{progress_tag} self-play finished: roots_generated={int(roots_generated)}, "
                f"unique_roots={int(len(unique_root_sigs))}, "
                f"train_samples={int(controller_train_samples + adversary_train_samples)}, "
                f"eval_samples={int(controller_eval_samples + adversary_eval_samples)}",
                flush=True,
            )


        if hasattr(self.mcts, "clear_search_state"):
            # Now release the scratch simulator once the whole worker batch is done.
            self.mcts.clear_search_state(drop_scratch=True)
        if hasattr(self.history, "clear_scratch_state"):
            self.history.clear_scratch_state()
        return last_state

    def run_arena_game(self, *args, **kwargs):
        raise NotImplementedError(
            "GV3 selfPlay arena flow has been removed. Use value-root generation via run_n_roots()."
        )
