# # (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import time
from vidur.entities import Request
from vidur.entities.batch import Batch
from vidur.entities.batch_stage import BatchStage
from vidur.types.replica_id import ReplicaId
from vidur.utils.slo_manager import SLOManager

from .environment import (
    AdversaryAction,
    AdversaryRequestSpec,
    ControllerAction,
    VidurGameStats,
    VidurMCTSState,
)
from .launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig
from .prefill_calibrator import PrefillProfile
from .virtual_simulator import VirtualSimulator
from vidur.entities.execution_time_predictor_request import ExecutionTimePredictorRequest

try:
    from . import mcts_native as _mcts_native
except Exception:  # pragma: no cover - optional native runtime
    _mcts_native = None


class VirtualVidurMCTSEnvironment:
    """
    Drop-in, speed-focused environment that preserves the same external API
    used by mctsDNN/selfPlay/evaluator, while using VirtualSimulator internally.
    """

    def __init__(
        self,
        *,
        base_simulator: VirtualSimulator,
        constraints: MCTSConstraintConfig,
        explore_cfg: MCTSExploreConfig,
        native_enabled: bool = False,
        native_strict_compat: bool = True,
    ) -> None:
        self._base = base_simulator
        self._constraints = constraints
        self._cfg = explore_cfg

        n_replicas = int(getattr(base_simulator._config.cluster_config, "num_replicas", 1))
        n_stages = int(getattr(base_simulator._config.cluster_config.replica_config, "num_pipeline_stages", 1))
        if n_replicas != 1:
            raise ValueError(
                f"VirtualVidurMCTSEnvironment supports single-replica only, got num_replicas={n_replicas}"
            )
        if n_stages != 1:
            raise ValueError(
                f"VirtualVidurMCTSEnvironment supports single-stage only, got num_pipeline_stages={n_stages}"
            )
        if getattr(base_simulator, "_execution_time_predictor", None) is None:
            raise ValueError("VirtualVidurMCTSEnvironment requires execution_time_predictor")

        self._slo_manager = SLOManager(base_simulator._config.slo_config)
        self._rng = random.Random(base_simulator._config.seed)
        self._prefill_profile = PrefillProfile.load_or_generate(
            base_simulator._config,
            step=constraints.interval_request_size,
            slowdown=constraints.prefill_slowdown,
            path=constraints.prefill_profile_path,
            max_tokens=constraints.max_request_tokens,
        )
        self._base_snapshot = base_simulator.snapshot_state()
        self._history_root_snapshot = None

        if self._constraints.max_request_tokens is None:
            self._constraints.max_request_tokens = self._prefill_profile.max_tokens

        self._controller_budgets_cache: Dict[int, Tuple[int, ...]] = {}
        self._native_enabled = bool(native_enabled)
        self._native_strict_compat = bool(native_strict_compat)
        if self._native_enabled and _mcts_native is None and self._native_strict_compat:
            raise RuntimeError(
                "native_enabled=True but vidur.mcts.mcts_native is not importable. "
                "Build the native module or set native_strict_compat=False."
            )
        self._perf = {
            "apply_ctrl_calls": 0.0,
            "apply_ctrl_total": 0.0,
            "lookup": 0.0,
            "alloc_norm": 0.0,
            "predictor": 0.0,
            "stats": 0.0,
            "rebuild": 0.0,
            "ff_decode": 0.0,

             # stats profiler
            "stats_calls": 0.0,
            "stats_total_internal": 0.0,
            "stats_batch_unpack": 0.0,
            "stats_ids_union": 0.0,
            "stats_req_get": 0.0,
            "stats_hooks": 0.0,
            "stats_prefill": 0.0,
            "stats_decode": 0.0,
            "stats_violation": 0.0,
            "stats_complete": 0.0,
        }


    def initial_state(self) -> VidurMCTSState:
        sim = VirtualSimulator(
            self._base._config,
            register_atexit=False,
            execution_time_predictor=getattr(self._base, "_execution_time_predictor", None),
        )
        sim.restore_state(self._base_snapshot)
        state = VidurMCTSState(sim, VidurGameStats())
        self._sync_active_ids_once(state)
        return state

    def snapshot_history_root(self, state: VidurMCTSState) -> None:
        self._history_root_snapshot = state.simulator.snapshot_state()

    def clone_history_root_state(self, stats_template: VidurGameStats) -> VidurMCTSState:
        if self._history_root_snapshot is None:
            return self.initial_state()

        sim = VirtualSimulator(
            self._base._config,
            register_atexit=False,
            execution_time_predictor=getattr(self._base, "_execution_time_predictor", None),
        )
        sim.restore_state(self._history_root_snapshot)
        state = VidurMCTSState(sim, stats_template.clone())
        self._sync_active_ids_once(state)
        return state

    def clone_state_from_snapshot(self, snapshot: Any, stats_template: VidurGameStats) -> VidurMCTSState:
        sim = VirtualSimulator(
            self._base._config,
            register_atexit=False,
            execution_time_predictor=getattr(self._base, "_execution_time_predictor", None),
        )
        sim.restore_state(snapshot)
        state = VidurMCTSState(sim, stats_template.clone())
        self._sync_active_ids_once(state)
        return state


    ## Util functions :

    def _req_map(self, simulator: VirtualSimulator) -> Dict[int, Request]:
        rs = simulator._scheduler.get_replica_scheduler(simulator.replica_id)
        return getattr(rs, "_requests", {})

    def _remaining_prefill(self, req: Request) -> int:
        return int(getattr(req, "remaining_prefill_tokens",
                        max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))))

    def _remaining_decode(self, req: Request) -> int:
        return max(0, int(req.num_decode_tokens) - int(req.num_processed_decode_tokens))

    def _is_pending(self, req: Request) -> bool:
        prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
        rem_pref = self._remaining_prefill(req)
        rem_dec = self._remaining_decode(req)
        return (rem_pref > 0 and not prefill_done) or (prefill_done and rem_dec > 0)

    def _sync_active_ids_once(self, state: VidurMCTSState) -> None:
        req_map = self._req_map(state.simulator)
        state.stats.active_request_ids = {
            int(rid) for rid, req in req_map.items()
            if (not bool(getattr(req, "completed", False))) and self._is_pending(req)
        }

    def _update_active_ids_for(self, state: VidurMCTSState, touched_ids: Iterable[int]) -> None:
        req_map = self._req_map(state.simulator)
        for rid in touched_ids:
            rid = int(rid)
            req = req_map.get(rid)
            if req is None or bool(getattr(req, "completed", False)) or (not self._is_pending(req)):
                state.stats.active_request_ids.discard(rid)
            else:
                state.stats.active_request_ids.add(rid)


    ## Util function ends here 



    def sample_adversary_actions(
        self, state: VidurMCTSState, max_samples: int
    ) -> Tuple[List[Optional[AdversaryAction]], List[bool]]:
        del max_samples
        self._drain_arrivals(state.simulator)

        num_actions = 6
        actions_by_index: List[Optional[AdversaryAction]] = [None] * num_actions
        mask: List[bool] = [False] * num_actions

        sim_time = float(state.simulator._time)
        last = getattr(state.stats, "last_prefill_batch_time", None)
        can_send = (last is None) or (sim_time >= float(last) + 1.0 - 1e-9)

        if not can_send:
            actions_by_index[0] = AdversaryAction(requests=[], stop_decode_ids=[])
            mask[0] = True
            return actions_by_index, mask

        prefill_size = int(getattr(self._constraints, "max_request_tokens", 3072) or 3072)
        decode_tokens_fixed = 5000
        slo_opts = self._constraints.request_slo_options

        base_prefill = float(self._prefill_profile.lookup(prefill_size))
        prefill_slo_time = base_prefill
        decode_slo_time = (
            float(slo_opts.decode_slos[0]) / 1000.0
            if getattr(slo_opts, "decode_slos", None)
            else 0.0
        )

        for i in range(num_actions):
            count = i + 1
            specs: List[AdversaryRequestSpec] = []
            for _ in range(count):
                specs.append(
                    AdversaryRequestSpec(
                        prefill_tokens=prefill_size,
                        decode_tokens=decode_tokens_fixed,
                        prefill_slo=float(prefill_slo_time),
                        decode_slo=float(decode_slo_time),
                    )
                )
            actions_by_index[i] = AdversaryAction(requests=specs, stop_decode_ids=[])
            mask[i] = True

        return actions_by_index, mask


    def sample_controller_actions(
        self,
        state: VidurMCTSState,
        max_samples: int,
        use_state_cache: bool = True,
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        if self._native_enabled and _mcts_native is not None:
            try:
                return self._sample_controller_actions_native(
                    state,
                    max_samples=max_samples,
                    use_state_cache=use_state_cache,
                )
            except Exception:
                if self._native_strict_compat:
                    raise
        return self._sample_controller_actions_python(
            state,
            max_samples=max_samples,
            use_state_cache=use_state_cache,
        )

    def _sample_controller_actions_native(
        self,
        state: VidurMCTSState,
        max_samples: int,
        use_state_cache: bool = True,
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        del max_samples, use_state_cache

        self._drain_arrivals(state.simulator)
        sim_time = float(state.simulator._time)
        step = int(self._constraints.interval_request_size or 512)

        budget_cache = self._controller_budgets_cache
        budgets = budget_cache.get(step)
        if budgets is None:
            budgets = tuple(step * i for i in range(1, 7))
            budget_cache[step] = budgets

        num_heur = 4
        num_actions = num_heur * len(budgets)
        actions_by_index: List[Optional[ControllerAction]] = [None] * num_actions
        mask: List[bool] = [False] * num_actions

        req_map = self._req_map(state.simulator)
        active_ids = state.stats.active_request_ids
        if not active_ids:
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True
            return actions_by_index, mask

        if not hasattr(self, "_cpp_profile_tokens") or not hasattr(self, "_cpp_profile_times"):
            entries = dict(getattr(self._prefill_profile, "entries", {}) or {})
            self._cpp_profile_tokens = sorted(int(k) for k in entries.keys())
            self._cpp_profile_times = [float(entries[k]) for k in self._cpp_profile_tokens]

        self._perf.setdefault("ctrl_sample_total", 0.0)
        self._perf.setdefault("ctrl_sample_pack", 0.0)
        self._perf.setdefault("ctrl_sample_cpp", 0.0)
        self._perf.setdefault("ctrl_sample_unpack", 0.0)
        self._perf.setdefault("ctrl_sample_calls", 0.0)
        self._perf["ctrl_sample_calls"] += 1.0

        t_all = time.perf_counter()

        t_pack = time.perf_counter()
        req_states = []
        for rid0 in active_ids:
            rid = int(rid0)
            req = req_map.get(rid)
            if req is None:
                continue
            rs = _mcts_native.ControllerRequestStateNative()
            rs.request_id = rid
            rs.prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            rs.remaining_prefill = int(self._remaining_prefill(req))
            rs.remaining_decode = int(self._remaining_decode(req))
            rs.arrived_at = float(getattr(req, "arrived_at", 0.0))
            rs.prefill_slo = float(getattr(req, "prefill_slo_time", 0.0))
            req_states.append(rs)
        self._perf["ctrl_sample_pack"] += time.perf_counter() - t_pack

        t_cpp = time.perf_counter()
        out_actions, out_mask = _mcts_native.sample_controller_actions_pyready(
            req_states,
            sim_time,
            [int(b) for b in budgets],
            self._cpp_profile_tokens,
            self._cpp_profile_times,
            int(num_actions),
        )
        self._perf["ctrl_sample_cpp"] += time.perf_counter() - t_cpp

        t_unpack = time.perf_counter()
        actions = list(out_actions)
        if len(actions) < num_actions:
            actions.extend([None] * (num_actions - len(actions)))
        elif len(actions) > num_actions:
            actions = actions[:num_actions]

        mask = [bool(x) for x in out_mask]
        if len(mask) < num_actions:
            mask.extend([False] * (num_actions - len(mask)))
        elif len(mask) > num_actions:
            mask = mask[:num_actions]

        actions_by_index = [a for a in actions]
        if not any(mask):
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True

        self._perf["ctrl_sample_unpack"] += time.perf_counter() - t_unpack
        self._perf["ctrl_sample_total"] += time.perf_counter() - t_all
        return actions_by_index, mask

    def _sample_controller_actions_python(
        self,
        state: VidurMCTSState,
        max_samples: int,
        use_state_cache: bool = True,
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        del max_samples, use_state_cache

        self._drain_arrivals(state.simulator)
        sim_time = float(state.simulator._time)
        step = int(self._constraints.interval_request_size or 512)

        budget_cache = self._controller_budgets_cache
        budgets = budget_cache.get(step)
        if budgets is None:
            budgets = tuple(step * i for i in range(1, 7))
            budget_cache[step] = budgets

        req_map = self._req_map(state.simulator)
        active_ids = state.stats.active_request_ids

        num_heur = 4
        num_budgets = len(budgets)
        num_actions = num_heur * num_budgets
        actions_by_index: List[Optional[ControllerAction]] = [None] * num_actions
        mask: List[bool] = [False] * num_actions

        if not active_ids:
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True
            return actions_by_index, mask

        prefill_ids: List[int] = []
        rem_pref_by_id: Dict[int, int] = {}
        edf_key_by_id: Dict[int, float] = {}
        lst_key_by_id: Dict[int, float] = {}
        total_remaining_prefill = 0
        decode_candidates: List[int] = []
        decode_base_template: Dict[int, int] = {}

        for rid0 in active_ids:
            rid = int(rid0)
            req = req_map.get(rid)
            if req is None:
                continue

            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            if not prefill_done:
                rem_pref = self._remaining_prefill(req)
                if rem_pref > 0:
                    prefill_ids.append(rid)
                    rem_pref_by_id[rid] = rem_pref
                    total_remaining_prefill += rem_pref

                    arrived_at = float(getattr(req, "arrived_at", 0.0))
                    prefill_slo = float(getattr(req, "prefill_slo_time", 0.0))
                    edf_key_by_id[rid] = arrived_at + prefill_slo

                    remaining_slo = prefill_slo - max(0.0, sim_time - arrived_at)
                    est = float(self._prefill_profile.lookup(rem_pref))
                    lst_key_by_id[rid] = remaining_slo - est
            else:
                if self._remaining_decode(req) > 0:
                    decode_candidates.append(rid)
                    decode_base_template[rid] = 1

        if total_remaining_prefill == 0:
            decode_base = dict(decode_base_template)
            token_alloc = dict(decode_base)
            a = ControllerAction(
                token_budget=len(decode_base),
                selected_request_ids=list(token_alloc.keys()) if token_alloc else None,
                token_allocations=token_alloc,
                prefill_allocations={},
                decode_allocations=decode_base,
                heuristic="SJF",
                strategy="Fixed",
            )
            actions_by_index[0] = a
            mask[0] = True
            return actions_by_index, mask

        ordered_sjf = sorted(prefill_ids, key=rem_pref_by_id.__getitem__)
        ordered_edf = sorted(prefill_ids, key=edf_key_by_id.__getitem__)
        ordered_lst = sorted(prefill_ids, key=lst_key_by_id.__getitem__)
        ordered_ljf = list(reversed(ordered_sjf))
        heuristics = [
            ("SJF", ordered_sjf),
            ("EDF", ordered_edf),
            ("LST", ordered_lst),
            ("LJF", ordered_ljf),
        ]
        decode_budget = len(decode_candidates)

        def build_action(ordered_prefill: List[int], prefill_budget: int, heur_name: str) -> ControllerAction:
            remaining_budget = max(0, int(prefill_budget))
            pre: Dict[int, int] = {}
            used_prefill = 0

            for rid in ordered_prefill:
                if remaining_budget <= 0:
                    break
                cap = rem_pref_by_id[rid]
                if cap <= 0:
                    continue
                alloc = cap if cap < remaining_budget else remaining_budget
                pre[rid] = alloc
                remaining_budget -= alloc
                used_prefill += alloc

            decode_alloc = dict(decode_base_template)
            token_alloc = dict(decode_alloc)
            token_alloc.update(pre)
            selected = list(token_alloc.keys()) if token_alloc else None
            return ControllerAction(
                token_budget=decode_budget + used_prefill,
                selected_request_ids=selected,
                token_allocations=token_alloc,
                prefill_allocations=pre,
                decode_allocations=decode_alloc,
                heuristic=heur_name,
                strategy="Fixed",
            )

        for b_idx, budget in enumerate(budgets):
            budget_valid = (budget <= total_remaining_prefill) if total_remaining_prefill > 0 else (b_idx == 0)
            for h_idx, (h_name, ordered_prefill) in enumerate(heuristics):
                idx = b_idx * num_heur + h_idx
                if not budget_valid:
                    mask[idx] = False
                    actions_by_index[idx] = None
                    continue
                mask[idx] = True
                actions_by_index[idx] = build_action(ordered_prefill, budget, h_name)

        if not any(mask):
            actions_by_index[0] = ControllerAction(token_budget=0, selected_request_ids=None)
            mask[0] = True
        return actions_by_index, mask






    def apply_adversary_action_only(
        self, state: VidurMCTSState, action: AdversaryAction, *, inplace: bool = False
    ) -> VidurMCTSState:
        target_state = state if inplace else state.fork()
        self._apply_adversary_action(target_state, action)
        self._drain_arrivals(target_state.simulator)
        return target_state

    def apply_controller_action_only(
        self, state: VidurMCTSState, action: ControllerAction, *, inplace: bool = False
    ) -> VidurMCTSState:

        # t_all = time.perf_counter()
        # self._perf["apply_ctrl_calls"] += 1

        new_state = state if inplace else state.fork()
        self._drain_arrivals(new_state.simulator)

        # t = time.perf_counter()
        req_map = self._req_map(new_state.simulator)
        active_ids = new_state.stats.active_request_ids

        if __debug__:
            missing = [rid for rid in active_ids if rid not in req_map]
            assert not missing, f"active_request_ids not in req_map: {missing[:8]}"
        # self._perf["lookup"] += time.perf_counter() - t

        if not active_ids:
            # t = time.perf_counter()
            self._update_requests_and_stats(new_state, batch_exec=None)
            # self._perf["stats"] += time.perf_counter() - t
            # self._perf["apply_ctrl_total"] += time.perf_counter() - t_all
            return new_state

        # t = time.perf_counter()
        (
            prefill_alloc,
            decode_alloc,
            token_alloc,
            req_ids,
            _requests,      # returned by normalize; not passed onward
            num_tokens,
            pred_reqs,
        ) = self._normalize_and_collect_batch(
            action,
            req_map=req_map,
            active_ids=active_ids,
        )
        # self._perf["alloc_norm"] += time.perf_counter() - t

        batch_exec = None

        if pred_reqs:
            start_time = float(new_state.simulator._time)

            # t = time.perf_counter()
            execution_time = new_state.simulator._execution_time_predictor.get_execution_time(
                pred_reqs, 0
            )
            # self._perf["predictor"] += time.perf_counter() - t

            end_time = start_time + float(execution_time.total_time)
            new_state.simulator._set_time(end_time)

            # only ids/tokens/times are passed (no request objects)
            batch_exec = {
                "request_ids": list(req_ids),
                "num_tokens": list(num_tokens),
                "start_time": start_time,
                "end_time": end_time,
                "stage_total_time": float(execution_time.total_time),
                "stage_model_time": float(execution_time.model_time),
            }

        if not prefill_alloc:
            # t = time.perf_counter()
            self._maybe_fast_forward_decode_only_to_next_adv_second(new_state)
            # self._perf["ff_decode"] += time.perf_counter() - t

        # t = time.perf_counter()
        self._update_requests_and_stats(new_state, batch_exec=batch_exec)
        # self._perf["stats"] += time.perf_counter() - t

        # self._perf["apply_ctrl_total"] += time.perf_counter() - t_all
        return new_state



    def apply_actions(
        self,
        state: VidurMCTSState,
        adversary_action: AdversaryAction,
        controller_action: ControllerAction,
    ) -> VidurMCTSState:
        intermediate = self.apply_adversary_action_only(state, adversary_action)
        return self.apply_controller_action_only(intermediate, controller_action)

    def evaluate_objective(self, state: VidurMCTSState) -> Tuple[int, float]:
        st = state.stats
        return int(st.slo_violations), float(st.slo_lateness_sum)

    def describe_state(self, state: VidurMCTSState) -> Dict[str, Any]:
        violations, total_lateness = self.evaluate_objective(state)
        simulator = state.simulator
        # request_lookup = self._build_request_lookup(simulator)
        req_map = self._req_map(simulator)
        # waiting_ids = self._collect_waiting_request_ids(simulator)
        waiting_ids = sorted(state.stats.active_request_ids)
        return {
            "sim_time": float(simulator._time),
            "requests_in_system": len(state.stats.active_request_ids),
            "requests_generated": int(state.stats.requests_generated),
            "requests_completed": int(state.stats.requests_completed),
            "slo_violations": int(violations),
            "total_lateness": float(total_lateness),
            "waiting_request_ids": waiting_ids,
            "completed_request_ids": list(state.stats.completed_request_ids),
        }

    def _max_request_tokens_allowed(self) -> int:
        if self._constraints.max_request_tokens is not None:
            return int(self._constraints.max_request_tokens)
        return int(self._prefill_profile.max_tokens)

    def _apply_adversary_action(self, state: VidurMCTSState, action: AdversaryAction) -> None:
        if (not action.requests) and (not action.stop_decode_ids):
            return

        sim = state.simulator
        time_now = float(sim._time)

        if action.requests:
            if state.stats.last_prefill_batch_time is None:
                arrival_time = math.floor(time_now)
            else:
                arrival_time = float(state.stats.last_prefill_batch_time) + 1.0
            state.stats.last_prefill_batch_time = float(arrival_time)
        else:
            arrival_time = math.floor(time_now)

        # lookup = self._build_request_lookup(sim)
        # Request._id = max(lookup.keys()) if lookup else -1
        req_map = self._req_map(sim)
        Request._id = max((int(rid) for rid in req_map.keys()), default=-1)

        for spec in action.requests:
            req = Request(
                arrived_at=float(arrival_time),
                num_prefill_tokens=int(spec.prefill_tokens),
                num_decode_tokens=int(spec.decode_tokens),
                block_hash_ids=None,
                block_size=None,
            )
            self._slo_manager.set_slos(req)
            req.prefill_slo_time = float(spec.prefill_slo)
            req.decode_slo_time = float(spec.decode_slo)
            req.completion_slo_time = -1
            setattr(req, "_desired_prefill_slo_time", float(spec.prefill_slo))
            setattr(req, "_desired_decode_slo_time", float(spec.decode_slo))

            req.assign_replica(sim.replica_id)
            rs = sim._scheduler.get_replica_scheduler(sim.replica_id)
            rs.add_request(req)

            state.stats.active_request_ids.add(int(req.id))
            state.stats.requests_generated += 1
            state.stats.recent_arrivals.append(float(arrival_time))

        if action.stop_decode_ids:
            # request_lookup = self._build_request_lookup(sim)

            for rid in action.stop_decode_ids:
                # req = request_lookup.get(int(rid))
                req = req_map.get(int(rid))
                if req is None:
                    continue
                req._num_decode_tokens = max(int(req.num_processed_decode_tokens), 0)

        self._update_active_ids_for(state, action.stop_decode_ids)
        window_start = float(time_now) - 1.0
        state.stats.recent_arrivals = [t for t in state.stats.recent_arrivals if float(t) >= window_start]
        # self._rebuild_scheduler_views(sim)



    def _normalize_and_collect_batch(
        self,
        action: ControllerAction,
        *,
        req_map: Dict[int, Request],
        active_ids: set[int],
    ) -> tuple[
        Dict[int, int],  # prefill_alloc
        Dict[int, int],  # decode_alloc
        Dict[int, int],  # token_alloc
        List[int],       # req_ids (batch order)
        List[Request],   # requests (batch order)
        List[int],       # num_tokens (batch order)
        List[ExecutionTimePredictorRequest],  # predictor requests (batch order)
    ]:
        prefill_alloc_in = dict(action.prefill_allocations or {})
        decode_alloc_in = dict(action.decode_allocations or {})
        token_alloc_in = dict(action.token_allocations or {})

        # Fast path: trust selected ids when present.
        # Fallback keeps correctness for externally-constructed actions.
        if action.selected_request_ids is not None:
            selected_ids = [int(x) for x in dict.fromkeys(action.selected_request_ids)]
        else:
            selected_ids = [
                int(x)
                for x in dict.fromkeys(
                    list(token_alloc_in.keys())
                    + list(prefill_alloc_in.keys())
                    + list(decode_alloc_in.keys())
                )
            ]

        prefill_alloc: Dict[int, int] = {}
        decode_alloc: Dict[int, int] = {}
        token_alloc: Dict[int, int] = {}

        req_ids: List[int] = []
        requests: List[Request] = []
        num_tokens: List[int] = []
        pred_reqs: List[ExecutionTimePredictorRequest] = []

        for rid in selected_ids:
            if rid not in active_ids:
                continue

            req = req_map.get(rid)
            if req is None:
                continue

            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            rem_pref = self._remaining_prefill(req)
            rem_dec = self._remaining_decode(req)

            pre_tok = max(0, int(prefill_alloc_in.get(rid, 0)))
            dec_tok = max(0, int(decode_alloc_in.get(rid, 0)))

            # Single fallback from generic token_alloc_in if specific split not provided.
            if pre_tok == 0 and dec_tok == 0:
                base_tok = max(0, int(token_alloc_in.get(rid, 0)))
                if base_tok > 0:
                    if prefill_done:
                        dec_tok = base_tok
                    else:
                        pre_tok = base_tok

            pre_tok = min(pre_tok, rem_pref)
            if not prefill_done:
                dec_tok = 0
            dec_tok = min(dec_tok, rem_dec)

            total = pre_tok + dec_tok
            if total <= 0:
                continue

            if pre_tok > 0:
                prefill_alloc[rid] = pre_tok
            if dec_tok > 0:
                decode_alloc[rid] = dec_tok
            token_alloc[rid] = total

            req_ids.append(rid)
            requests.append(req)
            num_tokens.append(total)
            pred_reqs.append(
                ExecutionTimePredictorRequest(
                    num_processed_tokens=int(req.num_processed_tokens),
                    num_tokens_to_process=int(total),
                    is_prefill_complete=prefill_done,
                )
            )

        action.prefill_allocations = dict(prefill_alloc)
        action.decode_allocations = dict(decode_alloc)
        action.token_allocations = dict(token_alloc)
        action.selected_request_ids = list(req_ids) if req_ids else None
        action.token_budget = int(sum(num_tokens))

        return (
            prefill_alloc,
            decode_alloc,
            token_alloc,
            req_ids,
            requests,
            num_tokens,
            pred_reqs,
        )


    def _rebuild_scheduler_views(self, simulator: VirtualSimulator) -> None:
        rs = simulator._scheduler.get_replica_scheduler(simulator.replica_id)

        active_reqs = [req for req in rs._requests.values() if not bool(getattr(req, "completed", False))]

        waiting: List[Request] = []
        running: List[Request] = []
        for req in active_reqs:
            # rem_prefill = max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))
            rem_prefill = self._remaining_prefill(req)
            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            rem_decode = max(0, int(req.num_decode_tokens) - int(req.num_processed_decode_tokens))
            if rem_prefill > 0 and not prefill_done:
                waiting.append(req)
            elif rem_decode > 0:
                running.append(req)

        waiting.sort(key=lambda r: (float(getattr(r, "arrived_at", 0.0)), int(r.id)))
        running.sort(key=lambda r: int(r.id))

        rs._waiting_queue.clear()
        for req in waiting:
            rs._waiting_queue.push(req)
        rs._running = running

    def _maybe_fast_forward_decode_only_to_next_adv_second(self, state: VidurMCTSState) -> None:
        sim = state.simulator
        stats = state.stats

        last = getattr(stats, "last_prefill_batch_time", None)
        if last is None:
            return

        sim_t = float(sim._time)
        target_t = float(last) + 1.0
        if sim_t >= target_t - 1e-9:
            return

        # reqs = self._build_request_lookup(sim)
        # if not reqs:
        #     return
        req_map = self._req_map(sim)
        if not req_map:
            return

        decode_active: list[Request] = []
        # for req in reqs.values():

        for rid in state.stats.active_request_ids:
            req = req_map.get(int(rid))
            if req is None:
                continue
            remaining_prefill = self._remaining_prefill(req)
            prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
            if remaining_prefill > 0 and not prefill_done:
                return

            remaining_decode = max(0, int(req.num_decode_tokens) - int(req.num_processed_decode_tokens))
            if prefill_done and remaining_decode > 0:
                decode_active.append(req)

        if not decode_active:
            return

        sim._set_time(target_t)
        for req in decode_active:
            rid = int(req.id)
            decode_slo = getattr(req, "_decode_slo_time", None)
            if decode_slo is None:
                continue
            stats.decode_next_deadline_by_id[rid] = float(target_t) + float(decode_slo)

    def _update_stats(self, state: VidurMCTSState) -> None:
        self._update_requests_and_stats(state, batch_exec=None)


    # def _update_requests_and_stats(
    #     self,
    #     state: VidurMCTSState,
    #     *,
    #     batch_exec: Optional[Dict[str, Any]] = None,
    # ) -> None:

    #     sim = state.simulator
    #     stats = state.stats
    #     sim_time = float(sim._time)
    #     req_map = self._req_map(sim)

    #     # O(batch) map for inline hook application 
    #     if batch_exec is not None:
    #         batch_tokens_by_id : Dict[int , int] = { int(rid) : int(rid_tokens) for rid , rid_tokens in zip(batch_exec["request_ids"], batch_exec["num_tokens"])}
    #         st = float(batch_exec["start_time"])
    #         et = float(batch_exec["end_time"])
    #         stage_total = float(batch_exec["stage_total_time"])
    #         stage_model = float(batch_exec["stage_model_time"])

    #     else :
    #         batch_tokens_by_id = {}
    #         st = et = stage_total = stage_model = 0.0


    #     # snapshot set (safe to mutate stats.active_request_ids in-loop)
    #     ids = set(state.stats.active_request_ids)
    #     ids.update(batch_tokens_by_id.keys())

    #     for rid in ids:
    #         rid = int(rid)
    #         request = req_map.get(rid)

    #         if request is None:
    #             stats.active_request_ids.discard(rid)
    #             continue
            
    #         # Inline lifecycle hooks for requests executed in this controller batch
    #         tok = batch_tokens_by_id.get(rid)
    #         if tok is not None:
    #             request.on_batch_schedule(st)
    #             request.on_batch_stage_schedule(st)
    #             request.on_batch_stage_end(et, stage_total, stage_model)
    #             request.on_batch_end(et, int(tok))            

    #          # ---------- per-request lateness / violation accounting ----------
    #         if rid not in stats.prefill_lateness_finalized:
    #             prefill_slo = getattr(request, "_prefill_slo_time", None)
    #             if prefill_slo is not None:
    #                 arrived_at = float(getattr(request, "_arrived_at", request.arrived_at))
    #                 deadline = arrived_at + float(prefill_slo)

    #                 is_prefill_complete = bool(
    #                     getattr(request, "_is_prefill_complete", request.is_prefill_complete)
    #                 )
    #                 prefill_completed_at = getattr(request, "_prefill_completed_at", None)

    #                 if is_prefill_complete and prefill_completed_at not in (None, 0):
    #                     actual = float(prefill_completed_at)
    #                 else:
    #                     actual = sim_time

    #                 prefill_late = max(0.0, actual - deadline)
    #                 prev_prefill = float(stats.per_request_prefill_lateness.get(rid, 0.0))
    #                 if prefill_late > prev_prefill:
    #                     stats.slo_lateness_sum += (prefill_late - prev_prefill)
    #                     stats.per_request_prefill_lateness[rid] = prefill_late

    #                 if is_prefill_complete:
    #                     stats.prefill_lateness_finalized.add(rid)

    #         decode_slo = getattr(request, "_decode_slo_time", None)
    #         has_decode_tokens = int(getattr(request, "_num_decode_tokens", request.num_decode_tokens)) > 0
    #         is_prefill_complete = bool(getattr(request, "_is_prefill_complete", request.is_prefill_complete))
    #         prefill_completed_at = getattr(request, "_prefill_completed_at", None)

    #         if (
    #             decode_slo is not None
    #             and float(decode_slo) >= 0.0
    #             and has_decode_tokens
    #             and is_prefill_complete
    #             and prefill_completed_at not in (None, 0)
    #         ):
    #             if rid not in stats.decode_next_deadline_by_id:
    #                 stats.decode_next_deadline_by_id[rid] = float(prefill_completed_at) + float(decode_slo)

    #             done = int(request.num_processed_decode_tokens)
    #             counted = int(stats.decode_tokens_counted.get(rid, 0))
    #             new_tokens = done - counted

    #             if new_tokens:
    #                 assert new_tokens == 1, f"Expected 1 new decode token for req {rid}, got {new_tokens}"
    #                 deadline = float(stats.decode_next_deadline_by_id[rid])
    #                 token_late = max(0.0, sim_time - deadline)
    #                 stats.per_request_decode_lateness[rid] = float(
    #                     stats.per_request_decode_lateness.get(rid, 0.0)
    #                 ) + float(token_late)
    #                 stats.slo_lateness_sum += float(token_late)
    #                 stats.decode_tokens_counted[rid] = done
    #                 stats.decode_next_deadline_by_id[rid] = sim_time + float(decode_slo)

    #         total_lateness = float(stats.per_request_prefill_lateness.get(rid, 0.0)) + float(
    #             stats.per_request_decode_lateness.get(rid, 0.0)
    #         )
    #         if total_lateness > 0.0 and rid not in stats.violated_request_ids:
    #             stats.violated_request_ids.add(rid)
    #             stats.slo_violations += 1

    #         # ---------- completion / active-set maintenance ----------
    #         if request.completed:
    #             if rid not in stats.completed_request_ids:
    #                 stats.requests_completed += 1
    #                 stats.completed_request_ids.add(rid)
    #             stats.active_request_ids.discard(rid)
    #         elif self._is_pending(request):
    #             stats.active_request_ids.add(rid)
    #         else:
    #             stats.active_request_ids.discard(rid)

    def _update_requests_and_stats(
        self,
        state: VidurMCTSState,
        *,
        batch_exec: Optional[Dict[str, Any]] = None,
    ) -> None:
        sim = state.simulator
        stats = state.stats
        sim_time = float(sim._time)
        req_map = self._req_map(sim)

        # t_all = time.perf_counter()
        # self._perf["stats_calls"] += 1

        # t = time.perf_counter()
        if batch_exec is not None:
            batch_tokens_by_id: Dict[int, int] = {
                int(rid): int(tok)
                for rid, tok in zip(batch_exec["request_ids"], batch_exec["num_tokens"])
            }
            st = float(batch_exec["start_time"])
            et = float(batch_exec["end_time"])
            stage_total = float(batch_exec["stage_total_time"])
            stage_model = float(batch_exec["stage_model_time"])
        else:
            batch_tokens_by_id = {}
            st = et = stage_total = stage_model = 0.0
        # self._perf["stats_batch_unpack"] += time.perf_counter() - t

        # t = time.perf_counter()
        ids = set(stats.active_request_ids)
        ids.update(batch_tokens_by_id.keys())
        # self._perf["stats_ids_union"] += time.perf_counter() - t

        for rid in ids:
            rid = int(rid)

            # t = time.perf_counter()
            request = req_map.get(rid)
            # self._perf["stats_req_get"] += time.perf_counter() - t

            if request is None:
                # t = time.perf_counter()
                stats.active_request_ids.discard(rid)
                # self._perf["stats_complete"] += time.perf_counter() - t
                continue

            tok = batch_tokens_by_id.get(rid)
            if tok is not None:
                # t = time.perf_counter()
                request.on_batch_schedule(st)
                request.on_batch_stage_schedule(st)
                request.on_batch_stage_end(et, stage_total, stage_model)
                request.on_batch_end(et, int(tok))
                # self._perf["stats_hooks"] += time.perf_counter() - t

            # t = time.perf_counter()
            if rid not in stats.prefill_lateness_finalized:
                prefill_slo = getattr(request, "_prefill_slo_time", None)
                if prefill_slo is not None:
                    arrived_at = float(getattr(request, "_arrived_at", request.arrived_at))
                    deadline = arrived_at + float(prefill_slo)

                    is_prefill_complete = bool(
                        getattr(request, "_is_prefill_complete", request.is_prefill_complete)
                    )
                    prefill_completed_at = getattr(request, "_prefill_completed_at", None)

                    actual = float(prefill_completed_at) if (is_prefill_complete and prefill_completed_at not in (None, 0)) else sim_time
                    prefill_late = max(0.0, actual - deadline)

                    prev_prefill = float(stats.per_request_prefill_lateness.get(rid, 0.0))
                    if prefill_late > prev_prefill:
                        stats.slo_lateness_sum += (prefill_late - prev_prefill)
                        stats.per_request_prefill_lateness[rid] = prefill_late

                    if is_prefill_complete:
                        stats.prefill_lateness_finalized.add(rid)
            # self._perf["stats_prefill"] += time.perf_counter() - t

            # t = time.perf_counter()
            decode_slo = getattr(request, "_decode_slo_time", None)
            has_decode_tokens = int(getattr(request, "_num_decode_tokens", request.num_decode_tokens)) > 0
            is_prefill_complete = bool(getattr(request, "_is_prefill_complete", request.is_prefill_complete))
            prefill_completed_at = getattr(request, "_prefill_completed_at", None)

            if (
                decode_slo is not None
                and float(decode_slo) >= 0.0
                and has_decode_tokens
                and is_prefill_complete
                and prefill_completed_at not in (None, 0)
            ):
                if rid not in stats.decode_next_deadline_by_id:
                    stats.decode_next_deadline_by_id[rid] = float(prefill_completed_at) + float(decode_slo)

                done = int(request.num_processed_decode_tokens)
                counted = int(stats.decode_tokens_counted.get(rid, 0))
                new_tokens = done - counted

                if new_tokens:
                    assert new_tokens == 1, f"Expected 1 new decode token for req {rid}, got {new_tokens}"
                    deadline = float(stats.decode_next_deadline_by_id[rid])
                    token_late = max(0.0, sim_time - deadline)
                    stats.per_request_decode_lateness[rid] = float(
                        stats.per_request_decode_lateness.get(rid, 0.0)
                    ) + float(token_late)
                    stats.slo_lateness_sum += float(token_late)
                    stats.decode_tokens_counted[rid] = done
                    stats.decode_next_deadline_by_id[rid] = sim_time + float(decode_slo)
            # self._perf["stats_decode"] += time.perf_counter() - t

            # t = time.perf_counter()
            total_lateness = float(stats.per_request_prefill_lateness.get(rid, 0.0)) + float(
                stats.per_request_decode_lateness.get(rid, 0.0)
            )
            if total_lateness > 0.0 and rid not in stats.violated_request_ids:
                stats.violated_request_ids.add(rid)
                stats.slo_violations += 1
            # self._perf["stats_violation"] += time.perf_counter() - t

            # t = time.perf_counter()
            if request.completed:
                if rid not in stats.completed_request_ids:
                    stats.requests_completed += 1
                    stats.completed_request_ids.add(rid)
                stats.active_request_ids.discard(rid)
            elif self._is_pending(request):
                stats.active_request_ids.add(rid)
            else:
                stats.active_request_ids.discard(rid)
            # self._perf["stats_complete"] += time.perf_counter() - t

        # self._perf["stats_total_internal"] += time.perf_counter() - t_all



    # def _build_request_lookup(self, simulator: VirtualSimulator) -> Dict[int, Request]:
    #     lookup: Dict[int, Request] = {}
    #     for replica_scheduler in simulator._scheduler._replica_schedulers.values():
    #         waiting = getattr(replica_scheduler, "_waiting_queue", None)
    #         if waiting and hasattr(waiting, "to_list"):
    #             for req in waiting.to_list():
    #                 lookup[int(req.id)] = req
    #         for req in getattr(replica_scheduler, "_running", []):
    #             lookup[int(req.id)] = req
    #         for req in getattr(replica_scheduler, "_requests", {}).values():
    #             if not bool(getattr(req, "completed", False)):
    #                 lookup[int(req.id)] = req
    #     return lookup

    def _build_request_lookup(self, simulator: VirtualSimulator, state: Optional[VidurMCTSState] = None) -> Dict[int, Request]:
        req_map = self._req_map(simulator)
        if state is None:
            return {int(rid): req for rid, req in req_map.items()
                    if (not bool(getattr(req, "completed", False))) and self._is_pending(req)}
        return {rid: req_map[rid] for rid in state.stats.active_request_ids if rid in req_map}


    def _collect_waiting_request_ids(self, simulator: VirtualSimulator) -> List[int]:
        ids: set[int] = set()
        for replica_scheduler in simulator._scheduler._replica_schedulers.values():
            waiting = getattr(replica_scheduler, "_waiting_queue", None)
            if waiting and hasattr(waiting, "to_list"):
                for req in waiting.to_list():
                    ids.add(int(req.id))
            for req in getattr(replica_scheduler, "_running", []):
                if not bool(getattr(req, "completed", False)):
                    ids.add(int(req.id))
        return sorted(ids)

    def _drain_arrivals(self, simulator: VirtualSimulator) -> None:
        del simulator
        return
