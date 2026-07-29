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

from .game_types import AdversaryAction, AdversaryRequestSpec, ControllerAction
from .environment import VidurGameStats, VidurMCTSState
from .launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig
from .prefill_calibrator import PrefillProfile
from .virtual_simulator import VirtualSimulator
from vidur.entities.execution_time_predictor_request import ExecutionTimePredictorRequest
from .config import DEFAULT_GAME_V2_CONFIG, GameVersion2Config
from .player_sample_actions import (
    GV2PlayerActionSampler,
    adversary_action_space_size,
    controller_action_space_size,
)
from .utils import GV2RuntimeOps

class VirtualVidurMCTSEnvironment:
    """
    used by mctsDNN/selfPlay/evaluator, while using VirtualSimulator internally.
    """
    _EPS = 1e-9
    _MINI_TICK_SEC = 0.2
    _META_NEXT_ADV_TICK = -9_100_001
    _META_LAST_ADV_TICK = -9_100_002
    _META_SENT_WINDOW_START = -9_100_003
    _META_CTRL_IDLE_UNTIL = -9_100_004
    _META_DECODE_CREDIT_BAL = -9_100_005
    _META_MISSED_ADV_SOURCE = -9_100_006  # 0=none, 1=controller-cross, 2=ff/jump-cross


    def __init__(
        self,
        *,
        base_simulator: VirtualSimulator,
        constraints: MCTSConstraintConfig,
        explore_cfg: MCTSExploreConfig,
        game_v2_cfg: Optional[GameVersion2Config] = None,
        native_enabled: bool = False,
        native_strict_compat: bool = True,
    ) -> None:
        self._base = base_simulator
        self._constraints = constraints
        self._cfg = explore_cfg

        self._gv2_cfg: GameVersion2Config = game_v2_cfg or DEFAULT_GAME_V2_CONFIG
        self._gv2_cfg.validate()

        self._action_sampler = GV2PlayerActionSampler(self._gv2_cfg)
        self._adv_action_space = adversary_action_space_size(self._gv2_cfg)
        self._ctrl_action_space = controller_action_space_size(self._gv2_cfg)

        self._prefill_window_cap_tokens = int(
            self._gv2_cfg.request.target_prefill_tokens_per_request_avg_window
            * self._gv2_cfg.timing.max_requests_per_launch_window
        )

        self._enable_internal_step_logging: bool = False
        self._ops = GV2RuntimeOps(env=self, cfg=self._gv2_cfg)

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

        # Internal (non-branching) debug timeline events, drained by MCTS logger.
        self._internal_step_events: List[Dict[str, Any]] = []
        self._internal_step_event_counter: int = 0


        if self._constraints.max_request_tokens is None:
            self._constraints.max_request_tokens = self._prefill_profile.max_tokens

        self._controller_budgets_cache: Dict[int, Tuple[int, ...]] = {}
        self._native_enabled = False
        self._native_strict_compat = bool(native_strict_compat)
        if native_enabled and self._native_strict_compat:
            raise RuntimeError(
                "Game Version 2 virtual environment currently supports python-only controller sampling."
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
        self._v2_init_clock(state, float(sim._time))
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
        self._v2_init_clock(state, float(sim._time))
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
        self._v2_init_clock(state, float(sim._time))
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

    def _is_controller_strict_noop(self, action: ControllerAction) -> bool:
        return (
            action.selected_request_ids is None
            and not (action.token_allocations or {})
            and not (action.prefill_allocations or {})
            and not (action.decode_allocations or {})
            and int(getattr(action, "token_budget", 0) or 0) == 0
        )


    # For debugging fast forward of decodes
    def _record_internal_event(
        self,
        state: VidurMCTSState,
        *,
        phase: str,
        start_time: float,
        end_time: float,
        reason: str = "",
        request_ids: Optional[Sequence[int]] = None,
        num_tokens: Optional[Sequence[int]] = None,
        stage_total_time: float = 0.0,
        decode_credit_before: Optional[int] = None,
        decode_credit_after: Optional[int] = None,
    ) -> None:
        # if not bool(getattr(self, "_enable_internal_step_logging", False)):
        #     return
        sim_snapshot = None
        stats_snapshot = None



        self._internal_step_event_counter += 1
        self._internal_step_events.append(
            {
                "event_id": int(self._internal_step_event_counter),
                "phase": str(phase),
                "start_time": float(start_time),
                "end_time": float(end_time),
                "reason": str(reason),
                "request_ids": [int(x) for x in (request_ids or [])],
                "num_tokens": [int(x) for x in (num_tokens or [])],
                "stage_total_time": float(stage_total_time),
                "decode_credit_before": (None if decode_credit_before is None else int(decode_credit_before)),
                "decode_credit_after": (None if decode_credit_after is None else int(decode_credit_after)),
                "state_snapshot": self.describe_state(state),
            }
        )

    def drain_internal_events(self) -> List[Dict[str, Any]]:
        if not self._internal_step_events:
            return []
        out = self._internal_step_events
        self._internal_step_events = []
        return out

    def clear_internal_events(self) -> None:
        self._internal_step_events = []


    # ----------------------------
    # GV2 utility delegation
    # ----------------------------
    def _v2_round(self, t: float) -> float:
        return self._ops._v2_round(t)

    def _v2_quantize_up(self, t: float) -> float:
        return self._ops._v2_quantize_up(t)

    def _v2_quantize_down(self, t: float) -> float:
        return self._ops._v2_quantize_down(t)

    def _v2_meta_get(self, stats: VidurGameStats, key: int, default: float) -> float:
        return self._ops._v2_meta_get(stats, key, default)

    def _v2_meta_set(self, stats: VidurGameStats, key: int, value: float) -> None:
        self._ops._v2_meta_set(stats, key, value)

    def _v2_init_clock(self, state: VidurMCTSState, sim_time: float) -> None:
        self._ops._v2_init_clock(state, sim_time)

    def _v2_current_adv_tick(self, state: VidurMCTSState) -> float:
        return self._ops._v2_current_adv_tick(state)

    def _v2_next_adv_tick(self, state: VidurMCTSState) -> float:
        return self._ops._v2_next_adv_tick(state)

    def _v2_has_pending_adv_tick(self, state: VidurMCTSState) -> bool:
        return self._ops._v2_has_pending_adv_tick(state)

    def _v2_sent_window_start(self, state: VidurMCTSState) -> Optional[int]:
        return self._ops._v2_sent_window_start(state)

    def _v2_set_sent_window_start(self, state: VidurMCTSState, window_start: Optional[int]) -> None:
        self._ops._v2_set_sent_window_start(state, window_start)

    def _v2_controller_idle_until(self, state: VidurMCTSState) -> Optional[float]:
        return self._ops._v2_controller_idle_until(state)

    def _v2_set_controller_idle_until(self, state: VidurMCTSState, idle_until: Optional[float]) -> None:
        self._ops._v2_set_controller_idle_until(state, idle_until)

    def _v2_has_active_prefill(self, state: VidurMCTSState) -> bool:
        return self._ops._v2_has_active_prefill(state)

    def _v2_controller_noop_space(self) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        return self._ops._v2_controller_noop_space()

    def _v2_decode_credit_balance(self, state: VidurMCTSState) -> int:
        return self._ops._v2_decode_credit_balance(state)

    def _v2_set_decode_credit_balance(self, state: VidurMCTSState, value: int) -> None:
        self._ops._v2_set_decode_credit_balance(state, value)

    def _v2_add_decode_credit(self, state: VidurMCTSState, delta: int) -> None:
        self._ops._v2_add_decode_credit(state, delta)

    def _v2_consume_decode_credit(self, state: VidurMCTSState, need: int) -> int:
        return self._ops._v2_consume_decode_credit(state, need)

    def _v2_get_recent_launches(self, state: VidurMCTSState) -> List[Tuple[float, int, int]]:
        return self._ops._v2_get_recent_launches(state)

    def _v2_set_recent_launches(self, state: VidurMCTSState, launches: List[Tuple[float, int, int]]) -> None:
        self._ops._v2_set_recent_launches(state, launches)

    def _v2_prune_recent_launches(self, state: VidurMCTSState, anchor_time: float) -> List[Tuple[float, int, int]]:
        return self._ops._v2_prune_recent_launches(state, anchor_time)

    def _v2_window_usage(self, anchor_time: float, launches: List[Tuple[float, int, int]]) -> Tuple[int, int]:
        return self._ops._v2_window_usage(anchor_time, launches)

    def _v2_has_active_decode(self, state: VidurMCTSState) -> bool:
        return self._ops._v2_has_active_decode(state)

    def _v2_collect_decode_ids(self, state: VidurMCTSState) -> List[int]:
        return self._ops._v2_collect_decode_ids(state)

    def _v2_enforce_decode_caps(self, state: VidurMCTSState) -> None:
        self._ops._v2_enforce_decode_caps(state)

    def _v2_finalize_decodes_to_credit_budget(self, state: VidurMCTSState) -> None:
        self._ops._v2_finalize_decodes_to_credit_budget(state)

    def _v2_decode_credit_balance_raw(self, state: VidurMCTSState) -> int:
        return self._ops._v2_decode_credit_balance_raw(state)

    def _v2_eviction_rule_from_action(self, action: ControllerAction) -> str:
        return self._ops._v2_eviction_rule_from_action(action)

    def _v2_drop_request(self, state: VidurMCTSState, rid: int, *, by_controller: bool) -> None:
        self._ops._v2_drop_request(state, rid, by_controller=by_controller)

    def _v2_apply_controller_eviction(self, state: VidurMCTSState, action: ControllerAction) -> None:
        self._ops._v2_apply_controller_eviction(state, action)

    def _v2_missed_adv_source(self, state: VidurMCTSState) -> int:
        return self._ops._v2_missed_adv_source(state)

    def _v2_set_missed_adv_source(self, state: VidurMCTSState, source: int) -> None:
        self._ops._v2_set_missed_adv_source(state, source)

    
    ### Functions for the trace Testing :

    @staticmethod
    def _trace_spec_get(spec: Any, name: str, default: Any = None) -> Any:
        if isinstance(spec, dict):
            return spec.get(name, default)
        return getattr(spec, name, default)

    def _trace_effective_tokens(
        self,
        *,
        prefill_tokens: int,
        decode_tokens: int,
        token_policy: str,
    ) -> tuple[int, int]:
        policy = str(token_policy or "clip").strip().lower()

        min_decode = int(self._gv2_cfg.request.min_decode_tokens_per_request)
        max_decode = int(self._gv2_cfg.request.max_decode_tokens_per_request)
        max_prefill = int(self._gv2_cfg.request.max_prefill_tokens_per_request)

        raw_prefill = int(prefill_tokens)
        raw_decode = int(decode_tokens)

        if policy == "clip":
            eff_prefill = max(1, min(raw_prefill, max_prefill))
            eff_decode = max(min_decode, min(raw_decode, max_decode))
            return int(eff_prefill), int(eff_decode)

        if policy == "bucket_gv3":
            allowed = sorted(int(x) for x in self._gv2_cfg.request.allowed_prefill_tokens)
            clipped_prefill = max(1, min(raw_prefill, max_prefill))
            eff_prefill = min(allowed, key=lambda x: (abs(int(x) - int(clipped_prefill)), int(x)))
            eff_decode = max(min_decode, min(raw_decode, max_decode))
            return int(eff_prefill), int(eff_decode)

        if policy in {"raw", "none"}:
            eff_prefill = max(1, raw_prefill)
            eff_decode = max(min_decode, raw_decode)
            return int(eff_prefill), int(eff_decode)

        raise ValueError(
            f"Unsupported trace token_policy={token_policy!r}; "
            "expected one of: clip, bucket_gv3, raw"
        )

    def _trace_default_decode_slo_time(self) -> float:
        legacy = getattr(self._gv2_cfg, "legacy_mcts", None)
        decode_slos = tuple(float(x) for x in (getattr(legacy, "decode_slos", None) or (50.0,)))
        return float(decode_slos[0]) / 1000.0 if decode_slos else 0.05

    def _trace_launch_groups(
        self,
        state: VidurMCTSState,
        trace_requests: Sequence[Any],
        *,
        now: float,
        token_policy: str,
        allow_future_arrivals: bool,
        enforce_gv3_launch_constraints: bool,
    ) -> list[tuple[float, int, int]]:
        by_arrival: dict[float, dict[str, Any]] = {}
        for spec in trace_requests:
            arrival_time = float(self._trace_spec_get(spec, "arrived_at", now))
            if (not bool(allow_future_arrivals)) and arrival_time > now + self._EPS:
                raise ValueError(
                    f"Trace request arrived_at={arrival_time} is in the future "
                    f"relative to simulator time={now}. Fast-forward before injecting."
                )

            eff_prefill, eff_decode = self._trace_effective_tokens(
                prefill_tokens=int(self._trace_spec_get(spec, "num_prefill_tokens", 0)),
                decode_tokens=int(self._trace_spec_get(spec, "num_decode_tokens", 0)),
                token_policy=str(token_policy),
            )
            group = by_arrival.setdefault(
                float(arrival_time),
                {"count": 0, "prefill_total": 0, "prefill_values": set(), "decode_values": set()},
            )
            group["count"] = int(group["count"]) + 1
            group["prefill_total"] = int(group["prefill_total"]) + int(eff_prefill)
            group["prefill_values"].add(int(eff_prefill))
            group["decode_values"].add(int(eff_decode))

        grouped_launches: list[tuple[float, int, int]] = []
        launch_history = list(self._v2_get_recent_launches(state))
        allowed_prefill = {int(x) for x in self._gv2_cfg.request.allowed_prefill_tokens}
        expected_decode = int(self._gv2_cfg.request.max_decode_tokens_per_request)
        max_launch_count = int(self._gv2_cfg.adversary_action.max_launch_count_per_tick)
        request_window_cap = int(self._gv2_cfg.timing.max_requests_per_launch_window)
        prefill_window_cap = int(self._prefill_window_cap_tokens)
        tick_sec = float(self._gv2_cfg.timing.adversary_tick_sec)
        tick_tolerance = max(1e-8, 10.0 * float(self._EPS))

        for arrival_time in sorted(by_arrival):
            group = by_arrival[arrival_time]
            count = int(group["count"])
            prefill_values = set(group["prefill_values"])
            decode_values = set(group["decode_values"])

            if bool(enforce_gv3_launch_constraints):
                if tick_sec > 0.0:
                    nearest_tick = round(float(arrival_time) / tick_sec) * tick_sec
                    if abs(float(arrival_time) - float(nearest_tick)) > tick_tolerance:
                        raise ValueError(
                            "GV3 trace launch is off the adversary decision grid: "
                            f"arrival={arrival_time}, tick_sec={tick_sec}."
                        )
                if len(prefill_values) != 1:
                    raise ValueError(
                        "GV3 trace launch must use one homogeneous prefill template per tick: "
                        f"arrival={arrival_time}, prefill_values={sorted(prefill_values)}."
                    )
                prefill_tokens = int(next(iter(prefill_values)))
                if prefill_tokens not in allowed_prefill:
                    raise ValueError(
                        "GV3 trace launch uses an unsupported prefill template: "
                        f"arrival={arrival_time}, prefill={prefill_tokens}, "
                        f"allowed={sorted(allowed_prefill)}."
                    )
                if decode_values != {expected_decode}:
                    raise ValueError(
                        "GV3 trace launch must use the native adversary decode size: "
                        f"arrival={arrival_time}, decode_values={sorted(decode_values)}, "
                        f"expected={expected_decode}."
                    )
                if count > max_launch_count:
                    raise ValueError(
                        "GV3 trace launch exceeds the per-tick request cap: "
                        f"arrival={arrival_time}, count={count}, cap={max_launch_count}."
                    )
            launch_prefill = int(group["prefill_total"])

            if bool(enforce_gv3_launch_constraints):
                used_count, used_prefill = self._v2_window_usage(
                    float(arrival_time), launch_history
                )
                if used_count + count > request_window_cap:
                    raise ValueError(
                        "GV3 trace launch exceeds the sliding request-window cap: "
                        f"arrival={arrival_time}, used={used_count}, launch={count}, "
                        f"cap={request_window_cap}."
                    )
                if used_prefill + launch_prefill > prefill_window_cap:
                    raise ValueError(
                        "GV3 trace launch exceeds the sliding prefill-window cap: "
                        f"arrival={arrival_time}, used={used_prefill}, launch={launch_prefill}, "
                        f"cap={prefill_window_cap}."
                    )

            launch = (float(arrival_time), int(count), int(launch_prefill))
            grouped_launches.append(launch)
            launch_history.append(launch)

        return grouped_launches

    def inject_trace_requests(
        self,
        state: VidurMCTSState,
        trace_requests: Sequence[Any],
        *,
        inplace: bool = True,
        token_policy: str = "clip",
        slo_policy: str = "gv3_default",
        allow_future_arrivals: bool = False,
        record_launch_history: bool = False,
        enforce_gv3_launch_constraints: bool = False,
    ) -> tuple[VidurMCTSState, dict[str, Any]]:
        """
        Inject external trace arrivals directly into the live environment.

        This intentionally does NOT use AdversaryAction. AdversaryAction applies
        GV3 synthetic adversary tick/window rules, while trace arrivals are an
        external workload source.

        Expected fields per trace request, as dict keys or object attrs:
          - arrived_at
          - num_prefill_tokens
          - num_decode_tokens
          - trace_row_id, optional
          - prefill_slo, optional
          - decode_slo, optional
        """
        target_state = state if inplace else state.fork()
        sim = target_state.simulator
        now = float(sim._time)

        if not trace_requests:
            return target_state, {
                "created_count": 0,
                "created_request_ids": [],
                "token_policy": str(token_policy),
                "slo_policy": str(slo_policy),
                "launch_history_recorded": bool(record_launch_history),
                "gv3_launch_constraints_enforced": bool(enforce_gv3_launch_constraints),
            }

        grouped_launches = self._trace_launch_groups(
            target_state,
            trace_requests,
            now=float(now),
            token_policy=str(token_policy),
            allow_future_arrivals=bool(allow_future_arrivals),
            enforce_gv3_launch_constraints=bool(enforce_gv3_launch_constraints),
        )

        req_map = self._req_map(sim)
        request_counter = getattr(sim, "_request_id_counter", None)
        if request_counter is None:
            request_counter = max((int(rid) for rid in req_map.keys()), default=-1)
        Request._id = int(request_counter)

        created_ids: list[int] = []
        created_rows: list[dict[str, Any]] = []
        latest_arrival = None

        for spec in trace_requests:
            arrival_time = float(self._trace_spec_get(spec, "arrived_at", now))
            if (not bool(allow_future_arrivals)) and arrival_time > now + self._EPS:
                raise ValueError(
                    f"Trace request arrived_at={arrival_time} is in the future "
                    f"relative to simulator time={now}. Fast-forward before injecting."
                )

            raw_prefill = int(self._trace_spec_get(spec, "num_prefill_tokens", 0))
            raw_decode = int(self._trace_spec_get(spec, "num_decode_tokens", 0))
            eff_prefill, eff_decode = self._trace_effective_tokens(
                prefill_tokens=raw_prefill,
                decode_tokens=raw_decode,
                token_policy=str(token_policy),
            )

            req = Request(
                arrived_at=float(arrival_time),
                num_prefill_tokens=int(eff_prefill),
                num_decode_tokens=int(eff_decode),
                block_hash_ids=None,
                block_size=None,
            )

            self._slo_manager.set_slos(req)

            prefill_slo = self._trace_spec_get(spec, "prefill_slo", None)
            decode_slo = self._trace_spec_get(spec, "decode_slo", None)
            slo_mode = str(slo_policy or "gv3_default").strip().lower()
            if prefill_slo is not None:
                req.prefill_slo_time = float(prefill_slo)
            elif slo_mode in {"gv3_default", "profile", "trace_default"}:
                req.prefill_slo_time = float(self._prefill_profile.lookup(int(eff_prefill)))
            if decode_slo is not None:
                req.decode_slo_time = float(decode_slo)
            elif slo_mode in {"gv3_default", "profile", "trace_default"}:
                req.decode_slo_time = float(self._trace_default_decode_slo_time())

            req.completion_slo_time = -1
            setattr(req, "_desired_prefill_slo_time", float(req.prefill_slo_time))
            setattr(req, "_desired_decode_slo_time", float(req.decode_slo_time))

            trace_row_id = self._trace_spec_get(spec, "trace_row_id", None)
            if trace_row_id is not None:
                setattr(req, "_trace_row_id", int(trace_row_id))
            setattr(req, "_trace_original_prefill_tokens", int(raw_prefill))
            setattr(req, "_trace_original_decode_tokens", int(raw_decode))
            setattr(req, "_trace_effective_prefill_tokens", int(eff_prefill))
            setattr(req, "_trace_effective_decode_tokens", int(eff_decode))
            setattr(req, "_trace_token_policy", str(token_policy))
            setattr(req, "_trace_slo_policy", str(slo_policy))

            req.assign_replica(sim.replica_id)
            rs = sim._scheduler.get_replica_scheduler(sim.replica_id)
            rs.add_request(req)
            sim._request_id_counter = int(Request._id)

            rid = int(req.id)
            target_state.stats.active_request_ids.add(rid)
            target_state.stats.requests_generated += 1

            created_ids.append(rid)
            created_rows.append(
                {
                    "request_id": rid,
                    "trace_row_id": "" if trace_row_id is None else int(trace_row_id),
                    "arrived_at": float(arrival_time),
                    "original_prefill_tokens": int(raw_prefill),
                    "original_decode_tokens": int(raw_decode),
                    "effective_prefill_tokens": int(eff_prefill),
                    "effective_decode_tokens": int(eff_decode),
                    "prefill_slo": float(req.prefill_slo_time),
                    "decode_slo": float(req.decode_slo_time),
                }
            )
            latest_arrival = (
                float(arrival_time)
                if latest_arrival is None
                else max(float(latest_arrival), float(arrival_time))
            )

        if latest_arrival is not None:
            target_state.stats.last_prefill_batch_time = float(latest_arrival)

        if bool(record_launch_history):
            launch_history = list(self._v2_get_recent_launches(target_state))
            launch_history.extend(grouped_launches)
            oldest_in_window = float(now) - float(self._gv2_cfg.timing.launch_window_sec)
            launch_history = [
                (float(ts), int(count), int(prefill))
                for ts, count, prefill in launch_history
                if float(ts) + self._EPS >= oldest_in_window
            ]
            launch_history.sort(key=lambda item: float(item[0]))
            self._v2_set_recent_launches(target_state, launch_history)

        self._update_requests_and_stats(target_state, batch_exec=None)

        return target_state, {
            "created_count": int(len(created_ids)),
            "created_request_ids": [int(x) for x in created_ids],
            "created_rows": created_rows,
            "token_policy": str(token_policy),
            "slo_policy": str(slo_policy),
            "launch_history_recorded": bool(record_launch_history),
            "gv3_launch_constraints_enforced": bool(enforce_gv3_launch_constraints),
            "recorded_launches": grouped_launches if bool(record_launch_history) else [],
        }

    def prepare_trace_controller_turn(
        self,
        state: VidurMCTSState,
        *,
        inplace: bool = True,
        adversary_tick_delay_sec: float | None = None,
    ) -> VidurMCTSState:
        """
        Make the current external-trace state controller-actionable.

        GV3 normally blocks controller actions when the synthetic adversary tick is
        pending. In trace evaluation, arrivals come from the trace, so the outer
        controller should not be blocked by a stale synthetic adversary tick.

        Keep the next synthetic adversary tick close, not far in the future. That
        lets AlphaGoZero MCTS still imagine its learned adversary shortly after
        the controller action when this state is used for model selection.
        """
        target_state = state if inplace else state.fork()
        now = float(target_state.simulator._time)
        delay = (
            float(self._MINI_TICK_SEC)
            if adversary_tick_delay_sec is None
            else max(float(adversary_tick_delay_sec), self._EPS)
        )

        next_tick = self._v2_round(now + delay)
        if next_tick <= now + self._EPS:
            next_tick = self._v2_round(now + float(self._MINI_TICK_SEC))

        self._v2_meta_set(target_state.stats, self._META_NEXT_ADV_TICK, float(next_tick))
        self._v2_set_controller_idle_until(target_state, None)
        self._v2_set_missed_adv_source(target_state, 0)
        return target_state

    ### Functions for trace testing end here

    ## Util function ends here 

    def sample_adversary_actions(
        self, state: VidurMCTSState, *, forbidden_stop_ids: Optional[Set[int]] = None
    ) -> Tuple[List[Optional[AdversaryAction]], List[bool]]:
        self._drain_arrivals(state.simulator)

        sim_time = float(state.simulator._time)
        tick = self._v2_current_adv_tick(state)

        launches = self._v2_prune_recent_launches(state, tick)

        # prefill_ref = int(max(self._gv2_cfg.request.allowed_prefill_tokens))
        # prefill_slo_time = float(self._prefill_profile.lookup(prefill_ref))
        slo_opts = self._constraints.request_slo_options
        decode_slo_time = (
            float(slo_opts.decode_slos[0]) / 1000.0
            if getattr(slo_opts, "decode_slos", None)
            else 0.0
        )

        forbidden_set = set(int(x) for x in (forbidden_stop_ids or []))
        actions, mask = self._action_sampler.sample_adversary_actions(
            sim_time=sim_time,
            decision_tick=float(tick),
            request_lookup=self._build_request_lookup(state.simulator, state=state),
            recent_launches=launches,
            prefill_slo_lookup_fn=lambda x: float(self._prefill_profile.lookup(int(x))),
            decode_slo_time=float(decode_slo_time),
            per_request_prefill_lateness=dict(state.stats.per_request_prefill_lateness),
            per_request_decode_lateness=dict(state.stats.per_request_decode_lateness),
            forbidden_stop_ids=forbidden_set,
        )
        return actions, mask

    

    def sample_controller_actions(
        self,
        state: VidurMCTSState,
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        self._drain_arrivals(state.simulator)

        if self._v2_has_pending_adv_tick(state):
            return self._v2_controller_noop_space()

        request_lookup = self._build_request_lookup(state.simulator, state=state)
        if not request_lookup:
            return self._v2_controller_noop_space()

        actions, mask = self._action_sampler.sample_controller_actions(
            sim_time=float(state.simulator._time),
            request_lookup=request_lookup,
            prefill_eta_lookup_fn=lambda t: float(self._prefill_profile.lookup(int(t))),
            decode_next_deadline_by_id=dict(state.stats.decode_next_deadline_by_id),
            per_request_prefill_lateness=dict(state.stats.per_request_prefill_lateness),
            per_request_decode_lateness=dict(state.stats.per_request_decode_lateness),
            violated_request_ids=set(state.stats.violated_request_ids),
            decode_credit_balance=(
                self._v2_decode_credit_balance(state)
                if self._gv2_cfg.credits.enforce_nonnegative_decode_credits
                else None
            ),
        )
        return actions, mask



    def _sample_controller_actions_native(
        self,
        state: VidurMCTSState
    ) -> Tuple[List[Optional[ControllerAction]], List[bool]]:
        raise RuntimeError(
            "Game Version 2 virtual environment uses python-only controller action sampling."
        )

    
    def _clear_transition_timing(self, state: VidurMCTSState) -> None:
        state.stats.transition_discount_time = None
        state.stats.transition_final_time = None
        state.stats.transition_fast_forward_time = 0.0

    def _set_transition_timing(
        self,
        state: VidurMCTSState,
        *,
        discount_time: float,
        final_time: float,
    ) -> None:
        state.stats.transition_discount_time = float(discount_time)
        state.stats.transition_final_time = float(final_time)
        state.stats.transition_fast_forward_time = max(
            0.0,
            float(final_time) - float(discount_time),
        )

    def apply_adversary_action_only(
        self, state: VidurMCTSState, action: AdversaryAction, *, inplace: bool = False
    ) -> VidurMCTSState:
        target_state = state if inplace else state.fork()
        self._clear_transition_timing(target_state)
        self._apply_adversary_action(target_state, action)
        self._drain_arrivals(target_state.simulator)
        action_time = float(target_state.simulator._time)
        self._set_transition_timing(
            target_state,
            discount_time=action_time,
            final_time=action_time,
        )
        return target_state

    def apply_controller_action_only(
        self, state: VidurMCTSState, action: ControllerAction, *, inplace: bool = False, fast_forward: bool = True,
    ) -> VidurMCTSState:
        new_state = state if inplace else state.fork()
        self._clear_transition_timing(new_state)

        tick_before = float(self._v2_next_adv_tick(new_state))
        self._drain_arrivals(new_state.simulator)

        # If adversary tick is pending (missed or exact), controller must no-op.
        if self._v2_has_pending_adv_tick(new_state):
            self._update_requests_and_stats(new_state, batch_exec=None)
            action_time = float(new_state.simulator._time)
            self._set_transition_timing(
                new_state,
                discount_time=action_time,
                final_time=action_time,
            )
            return new_state

        # Apply eviction branch first (Head-1 controller decision).
        self._v2_apply_controller_eviction(new_state, action)

        req_map = self._req_map(new_state.simulator)
        active_ids = new_state.stats.active_request_ids
        if not active_ids:
            self._update_requests_and_stats(new_state, batch_exec=None)
            action_end_time = float(new_state.simulator._time)

            pending_after_controller = self._v2_has_pending_adv_tick(new_state)
            if fast_forward and not pending_after_controller:
                self._maybe_fast_forward_decode_only_to_next_adv_second(new_state)

            time_after_final = float(new_state.simulator._time)
            self._set_transition_timing(
                new_state,
                discount_time=action_end_time,
                final_time=time_after_final,
            )

            miss_src = 2 if (time_after_final > tick_before + self._EPS) else 0
            self._v2_set_missed_adv_source(new_state, miss_src)
            return new_state



        decode_credit_limit = (
            self._v2_decode_credit_balance(new_state)
            if self._gv2_cfg.credits.enforce_nonnegative_decode_credits
            else None
        )

        (
            prefill_alloc,
            decode_alloc,
            token_alloc,
            req_ids,
            _requests,
            num_tokens,
            pred_reqs,
        ) = self._normalize_and_collect_batch(
            action,
            req_map=req_map,
            active_ids=active_ids,
            decode_credit_limit=decode_credit_limit,
        )

        batch_exec = None
        controller_credit_before: Optional[int] = None

        if pred_reqs:
            controller_credit_before = int(self._v2_decode_credit_balance_raw(new_state))
            start_time = float(new_state.simulator._time)
            execution_time = new_state.simulator._execution_time_predictor.get_execution_time(pred_reqs, 0)
            end_time = start_time + float(execution_time.total_time)
            new_state.simulator._set_time(end_time)

            batch_exec = {
                "request_ids": list(req_ids),
                "num_tokens": list(num_tokens),
                "start_time": start_time,
                "end_time": end_time,
                "stage_total_time": float(execution_time.total_time),
                "stage_model_time": float(execution_time.model_time),
            }

        self._update_requests_and_stats(new_state, batch_exec=batch_exec)
        action_end_time = float(new_state.simulator._time)

        time_after_controller = float(new_state.simulator._time)
        controller_crossed_adv_tick = time_after_controller > tick_before + self._EPS
        miss_src = 1 if controller_crossed_adv_tick else 0


        if batch_exec is not None:
            self._record_internal_event(
                new_state,
                phase="internal:controller_batch",
                start_time=float(batch_exec["start_time"]),
                end_time=float(batch_exec["end_time"]),
                reason="controller_apply_batch",
                request_ids=[int(x) for x in batch_exec["request_ids"]],
                num_tokens=[int(x) for x in batch_exec["num_tokens"]],
                stage_total_time=float(batch_exec["stage_total_time"]),
                decode_credit_before=(
                    None if controller_credit_before is None else int(controller_credit_before)
                ),
                decode_credit_after=int(self._v2_decode_credit_balance_raw(new_state)),
            )


        # only transition-time fast-forward
        # transition-time advancement, also considering the case if Controller is no-op while there are prefills active, advance to next tick
        is_strict_noop = self._is_controller_strict_noop(action)

        pending_after_controller = controller_crossed_adv_tick or self._v2_has_pending_adv_tick(new_state)
        if fast_forward and not pending_after_controller:
            if not self._v2_has_active_prefill(new_state):
                self._maybe_fast_forward_decode_only_to_next_adv_second(new_state)
            elif (
                bool(getattr(self._gv2_cfg.timing, "controller_noop_prefill_only_jump_to_next_adv_tick", True))
                and is_strict_noop
                and not self._v2_has_active_decode(new_state)
            ):
                next_tick = float(self._v2_next_adv_tick(new_state))
                now = float(new_state.simulator._time)
                if now + self._EPS < next_tick:
                    old_t = now
                    new_t = next_tick
                    new_state.simulator._set_time(new_t)
                    self._record_internal_event(
                        new_state,
                        phase="internal:jump_to_adv_tick",
                        start_time=old_t,
                        end_time=new_t,
                        reason="controller_noop_prefill_only_jump_to_tick",
                        request_ids=[],
                        num_tokens=[],
                        stage_total_time=max(0.0, new_t - old_t),
                        decode_credit_before=int(self._v2_decode_credit_balance_raw(new_state)),
                        decode_credit_after=int(self._v2_decode_credit_balance_raw(new_state)),
                    )
                    self._update_requests_and_stats(new_state, batch_exec=None)


        time_after_final = float(new_state.simulator._time)

        self._set_transition_timing(
            new_state,
            discount_time=action_end_time,
            final_time=time_after_final,
        )

        if miss_src == 0 and (time_after_final > tick_before + self._EPS):
            miss_src = 2

        self._v2_set_missed_adv_source(new_state, miss_src)

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
        
        stats = state.stats
        last_adv = self._v2_meta_get(stats, self._META_LAST_ADV_TICK, -1.0)

        return {
            "sim_time": float(simulator._time),
            "requests_in_system": len(stats.active_request_ids),
            "requests_generated": int(stats.requests_generated),
            "requests_completed": int(stats.requests_completed),
            "slo_violations": int(violations),
            "total_lateness": float(total_lateness),
            "avg_lateness": float(total_lateness),

            "active_request_ids": sorted(int(x) for x in stats.active_request_ids),
            "waiting_request_ids": sorted(int(x) for x in stats.active_request_ids),
            "completed_request_ids": sorted(int(x) for x in stats.completed_request_ids),
            "dropped_request_ids": sorted(int(x) for x in stats.dropped_request_ids),
            "stopped_decode_request_ids": sorted(int(x) for x in stats.stopped_decode_request_ids),

            "pending_adv_tick": bool(self._v2_has_pending_adv_tick(state)),
            "last_adv_tick": ("" if last_adv < 0 else float(last_adv)),

            "decode_credit_balance": int(self._v2_decode_credit_balance_raw(state)),
            "decode_credit_available": int(self._v2_decode_credit_balance(state)),
            "decode_tokens_counted_by_id": {int(k): int(v) for k, v in stats.decode_tokens_counted.items()},

            "violated_request_ids": sorted(int(x) for x in stats.violated_request_ids),
            "per_request_prefill_lateness_by_id": {int(k): float(v) for k, v in stats.per_request_prefill_lateness.items()},
            "per_request_decode_lateness_by_id": {int(k): float(v) for k, v in stats.per_request_decode_lateness.items()},
        }

    def _max_request_tokens_allowed(self) -> int:
        if self._constraints.max_request_tokens is not None:
            return int(self._constraints.max_request_tokens)
        return int(self._prefill_profile.max_tokens)

    def _apply_adversary_action(self, state: VidurMCTSState, action: AdversaryAction) -> None:
        sim = state.simulator
        time_now = float(sim._time)
        tick = self._v2_current_adv_tick(state)

        is_pre_tick = (time_now + self._EPS) < float(tick)
        is_strict_noop = (len(action.requests or []) == 0) and (len(action.stop_decode_ids or []) == 0)

        # For the case if adv action is no-op , and there's active prefill requests then we shouldnt advance to next tick
        if is_strict_noop and is_pre_tick:
            return

        if time_now + self._EPS < tick:
            sim._set_time(float(tick))
            time_now = float(sim._time)

        
        launches = self._v2_prune_recent_launches(state, tick)
        used_count, used_prefill = self._v2_window_usage(tick, launches)

        req_cap = int(self._gv2_cfg.timing.max_requests_per_launch_window)
        prefill_cap = int(self._prefill_window_cap_tokens)

        requested_count = len(action.requests or [])
        requested_prefill = sum(int(s.prefill_tokens) for s in (action.requests or []))

        # sliding-window only
        can_send = (
            requested_count > 0
            and (used_count + requested_count <= req_cap)
            and (used_prefill + requested_prefill <= prefill_cap)
        )


        req_map = self._req_map(sim)
        request_counter = getattr(sim, "_request_id_counter", None)
        if request_counter is None:
            request_counter = max((int(rid) for rid in req_map.keys()), default=-1)
        Request._id = int(request_counter)

        created_count = 0
        created_prefill_total = 0
        arrival_time = float(tick)

        if can_send:
            for spec in action.requests:
                prefill_toks = int(spec.prefill_tokens)
                decode_toks = int(spec.decode_tokens)

                prefill_toks = max(1, min(prefill_toks, int(self._gv2_cfg.request.max_prefill_tokens_per_request)))
                decode_toks = max(
                    int(self._gv2_cfg.request.min_decode_tokens_per_request),
                    min(decode_toks, int(self._gv2_cfg.request.max_decode_tokens_per_request)),
                )

                req = Request(
                    arrived_at=float(arrival_time),
                    num_prefill_tokens=prefill_toks,
                    num_decode_tokens=decode_toks,
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
                sim._request_id_counter = int(Request._id)

                state.stats.active_request_ids.add(int(req.id))
                state.stats.requests_generated += 1

                created_count += 1
                created_prefill_total += prefill_toks

            launches.append((float(arrival_time), int(created_count), int(created_prefill_total)))
            self._v2_set_recent_launches(state, launches)
            state.stats.last_prefill_batch_time = float(arrival_time)

            next_tick = float(tick) + float(self._MINI_TICK_SEC)

            if self._v2_has_active_prefill(state):
                self._v2_set_controller_idle_until(state, None)
            else:
                self._v2_set_controller_idle_until(state, float(next_tick))

        else:
            # No-op adversary: next micro-tick.
            next_tick = float(tick) + float(self._MINI_TICK_SEC)

            # If no active prefill, controller can be idled until next adversary tick.
            if self._v2_has_active_prefill(state):
                self._v2_set_controller_idle_until(state, None)
            else:
                self._v2_set_controller_idle_until(state, float(next_tick))

        self._v2_meta_set(state.stats, self._META_LAST_ADV_TICK, float(tick))
        self._v2_meta_set(state.stats, self._META_NEXT_ADV_TICK, float(self._v2_round(next_tick)))
        self._v2_set_missed_adv_source(state, 0)

        if action.stop_decode_ids:
            for rid in action.stop_decode_ids:
                rid = int(rid)
                req = req_map.get(rid)
                if req is None:
                    continue

                # Stop decode at current processed token.
                req._num_decode_tokens = max(int(req.num_processed_decode_tokens), 0)

                prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
                done_decode = int(req.num_processed_decode_tokens) >= int(req._num_decode_tokens)

                if prefill_done and done_decode:
                    req._completed = True
                    req._completed_at = float(state.simulator._time)

                    if rid not in state.stats.completed_request_ids:
                        state.stats.completed_request_ids.add(rid)
                        state.stats.requests_completed += 1

                    state.stats.active_request_ids.discard(rid)
                    state.stats.stopped_decode_request_ids.add(rid)

                    # Optional cleanup
                    state.stats.decode_next_deadline_by_id.pop(rid, None)
                    state.stats.decode_tokens_counted.pop(rid, None)

        self._update_active_ids_for(state, action.stop_decode_ids)
        self._v2_enforce_decode_caps(state)




    def _normalize_and_collect_batch(
        self,
        action: ControllerAction,
        *,
        req_map: Dict[int, Request],
        active_ids: set[int],
        decode_credit_limit: Optional[int] = None,
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


        # Controller no-op semantics:
        # no-op means "no prefill admission", not "no decode execution".
        # If action is strict no-op, auto-schedule 1 decode token per decode-ready active request.
        strict_noop = (
            action.selected_request_ids is None
            and not prefill_alloc_in
            and not decode_alloc_in
            and not token_alloc_in
        )
        if strict_noop:
            for rid in sorted(int(x) for x in active_ids):
                req = req_map.get(int(rid))
                if req is None:
                    continue
                prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
                if not prefill_done:
                    continue
                if self._remaining_decode(req) <= 0:
                    continue
                decode_alloc_in[int(rid)] = 1

            if decode_alloc_in:
                token_alloc_in.update(decode_alloc_in)


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

        decode_credit_left = None if decode_credit_limit is None else max(0, int(decode_credit_limit))

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

            # Re-apply decode-credit clamp AFTER fallback so token_alloc fallback
            # cannot bypass the decode credit limit.
            if dec_tok > 0 and decode_credit_left is not None:
                if decode_credit_left <= 0:
                    dec_tok = 0
                else:
                    dec_tok = min(dec_tok, decode_credit_left)
                    decode_credit_left -= dec_tok


            # Single fallback from generic token_alloc_in if specific split not provided.

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
        self._ops._maybe_fast_forward_decode_only_to_next_adv_second(state)

    def _update_stats(self, state: VidurMCTSState) -> None:
        self._update_requests_and_stats(state, batch_exec=None)

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

            # Capture prefill status before this batch update.
            prefill_complete_before = bool(
                getattr(request, "_is_prefill_complete", request.is_prefill_complete)
            )

            if tok is not None:
                # t = time.perf_counter()
                request.on_batch_schedule(st)
                request.on_batch_stage_schedule(st)
                request.on_batch_stage_end(et, stage_total, stage_model)
                request.on_batch_end(et, int(tok))
                # self._perf["stats_hooks"] += time.perf_counter() - t

            # Current status after applying batch hooks.
            prefill_complete_after = bool(
                getattr(request, "_is_prefill_complete", request.is_prefill_complete)
            )


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

            # Mint exactly once when request first enters decode-eligible state.
            # Mint exactly at prefill-complete transition for this request.
            became_prefill_complete = (not prefill_complete_before) and prefill_complete_after

            if (
                became_prefill_complete
                and has_decode_tokens
                and is_prefill_complete
                and not bool(getattr(request, "completed", False))
                and rid not in stats.decode_tokens_counted
                and self._remaining_decode(request) > 0
            ):
                self._v2_add_decode_credit(
                    state,
                    int(self._gv2_cfg.credits.decode_credit_mint_per_prefill_complete),
                )
                stats.decode_tokens_counted[rid] = 0

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

                if new_tokens > 0:
                    allowed = int(new_tokens)
                    if self._gv2_cfg.credits.enforce_nonnegative_decode_credits:
                        allowed = self._v2_consume_decode_credit(state, int(new_tokens))

                    if allowed > 0:
                        deadline = float(stats.decode_next_deadline_by_id[rid])
                        token_late = max(0.0, sim_time - deadline)

                        stats.per_request_decode_lateness[rid] = float(
                            stats.per_request_decode_lateness.get(rid, 0.0)
                        ) + float(token_late) * float(allowed)
                        stats.slo_lateness_sum += float(token_late) * float(allowed)

                        stats.decode_tokens_counted[rid] = counted + int(allowed)
                        stats.decode_next_deadline_by_id[rid] = sim_time + float(decode_slo)



            # self._perf["stats_decode"] += time.perf_counter() - t

            # t = time.perf_counter()
            total_lateness = float(stats.per_request_prefill_lateness.get(rid, 0.0)) + float(
                stats.per_request_decode_lateness.get(rid, 0.0)
            )

            # Dropping Requests which exceeds the max lateness :
            if total_lateness >= float(self._gv2_cfg.cost.auto_drop_lateness_sec):
                self._v2_drop_request(state, rid, by_controller=False)
                continue

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

        # Post-update decode-credit invariant:
        # if remaining credit cannot support current decode-active requests,
        # complete overflow requests by priority rule.
        self._v2_finalize_decodes_to_credit_budget(state)
        # self._perf["stats_total_internal"] += time.perf_counter() - t_all

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
