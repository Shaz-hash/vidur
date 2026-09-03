from __future__ import annotations

import csv
import gc
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib

from ...DNN.native_selfplay import _cfg_payload, attach_execution_predictor_payload
from ...game_types import ControllerAction
from ...logger.evaluation_pipeline_logger import arena_state_snapshot_for_log
from ...tests import native_logger_tests as nlt
from .. import runner as tester_runner
from ..config import DEFAULT_MODEL_TESTER_CONFIG
from ..trivial_controller import (
    make_strict_noop_controller_action,
    select_trivial_controller_action,
)
from .trace_loader import load_trace_csv, rebase_trace
from .trace_types import TraceRequest


@dataclass(frozen=True)
class FixedTraceEvalConfig:
    trace_csv: Path
    output_dir: Path
    policy: str
    time_limit_sec: float = 20.0
    max_steps: int = 8192
    max_trace_rows: int = 100_000
    seed: int = 2026
    token_policy: str = "clip"
    rebase_to_zero: bool = True
    record_launch_history: bool = True
    strict_gv3_trace: bool = False

    sjf_budget_tokens: int = 512
    sjf_heuristic: str = "SJF"
    sjf_eviction_rule: str = "evict_none"

    value_model_path: Path | None = None
    controller_prior_model_path: Path | None = None
    adversary_prior_model_path: Path | None = None
    model_version: int = 0
    mcts_iterations: int = 1000
    discount_factor: float = 0.995
    puct_c: float = 1.0
    uct_c: float = 1.0
    policy_prior_temperature: float = 1.0
    prior_min_prob: float = 1e-8
    native_search_mode: str = "full_tree"
    rollout_count: int = 10
    rollout_parallel_threads: int = 1
    rollout_horizon_sec: float = 1.0
    rollout_policy_temperature: float = 1.0
    rollout_probability_quantum: float = 1e-6
    rollout_max_actions: int = 4096
    disable_model_bootstrap: bool = False
    force_build_native: bool = False
    worker_threads: int = 1


@dataclass
class _NativeModelRuntime:
    native: Any
    value_runtime: Any
    controller_prior_runtime: Any
    adversary_prior_runtime: Any
    harness_model_path: Path


@dataclass(frozen=True)
class FixedTraceEvalResult:
    policy: str
    output_dir: Path
    steps_csv: Path
    summary_csv: Path
    validation_csv: Path
    rows_read: int
    rows_released: int
    requests_generated: int
    requests_completed: int
    final_sim_time: float
    slo_violations: int
    total_lateness: float
    total_cost: float
    prefill_policy_steps: int
    decode_drain_steps: int
    end_reason: str


def _arena_cpp_module() -> Any:
    from vidur.bellman_v4_adv import arena_mcts_value_runnerCPP as arena_cpp

    return arena_cpp


def _prepare_native_model_runtime(cfg: FixedTraceEvalConfig) -> _NativeModelRuntime:
    if cfg.value_model_path is None:
        raise ValueError("value_model_path is required for model_cpp policy")
    if cfg.controller_prior_model_path is None:
        raise ValueError("controller_prior_model_path is required for model_cpp policy")
    if cfg.adversary_prior_model_path is None:
        raise ValueError("adversary_prior_model_path is required for model_cpp policy")

    arena_cpp = _arena_cpp_module()
    arena_cpp._limit_native_threads(int(cfg.worker_threads))
    native = arena_cpp._import_native_cpp(force_build=bool(cfg.force_build_native))

    native_dir = Path(cfg.output_dir) / "native_model"
    value_runtime, _value_export, model, wrapped = arena_cpp._load_value_runtime_from_joblib(
        native=native,
        model_path=Path(cfg.value_model_path).expanduser(),
        export_path=native_dir / "value_hgb_native_export.tsv",
        feature_dim=226,
        model_tag=f"fixed_trace_value:{Path(cfg.value_model_path).name}",
    )

    harness_model_path = Path(cfg.value_model_path).expanduser()
    if wrapped:
        harness_model_path = Path(cfg.output_dir) / "arena_model" / "v4_adv_hgb_wrapper.joblib"
        harness_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, harness_model_path, compress=3)

    controller_prior_runtime, _controller_export = arena_cpp._load_prior_runtime_from_joblib(
        native=native,
        model_path=Path(cfg.controller_prior_model_path).expanduser(),
        export_path=native_dir / "controller_prior_hgb_native_export.tsv",
        feature_dim=269,
        model_tag=f"fixed_trace_controller_prior:{Path(cfg.controller_prior_model_path).name}",
    )
    adversary_prior_runtime, _adversary_export = arena_cpp._load_prior_runtime_from_joblib(
        native=native,
        model_path=Path(cfg.adversary_prior_model_path).expanduser(),
        export_path=native_dir / "adversary_prior_hgb_native_export.tsv",
        feature_dim=233,
        model_tag=f"fixed_trace_adversary_prior:{Path(cfg.adversary_prior_model_path).name}",
    )

    return _NativeModelRuntime(
        native=native,
        value_runtime=value_runtime,
        controller_prior_runtime=controller_prior_runtime,
        adversary_prior_runtime=adversary_prior_runtime,
        harness_model_path=Path(harness_model_path),
    )


def _build_tester_config(cfg: FixedTraceEvalConfig, runtime: _NativeModelRuntime | None) -> Any:
    trivial_policy = DEFAULT_MODEL_TESTER_CONFIG.trivial_policy.__class__(
        heuristic=str(cfg.sjf_heuristic),
        budget_tokens=int(cfg.sjf_budget_tokens),
        eviction_rule=str(cfg.sjf_eviction_rule),
    )
    model_path = (
        Path(runtime.harness_model_path)
        if runtime is not None
        else _ensure_callable_harness_model(
            _default_local_value_model_path(),
            Path(cfg.output_dir) / "arena_model" / "v4_adv_hgb_wrapper.joblib",
        )
    )
    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint path does not exist: {model_path}")

    from dataclasses import replace

    return replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind="classical_joblib",
        model_checkpoint_path=str(model_path),
        output_dir=str(cfg.output_dir),
        num_games=1,
        game_id_start=91_000_000,
        start_player="controller",
        environment_lang="python",
        arena_time_limit_sec=float(cfg.time_limit_sec),
        arena_max_total_turns=int(cfg.max_steps),
        bootstrap_model_version=max(1, int(cfg.model_version or 1)),
        trivial_policy=trivial_policy,
        write_arena_game_logs=False,
        write_model_action_detail_logs=False,
        arena_num_processes=1,
        arena_worker_threads=1,
    )


def _default_local_value_model_path() -> Path:
    repo = Path(__file__).resolve()
    for parent in repo.parents:
        candidate = (
            parent
            / "simulator_output"
            / "GV3_Agent"
            / "AlphaGoZero"
            / "models"
            / "Model_Version139"
            / "value"
            / "hgb_sq_63leaf_1050iter_a2"
            / "model.joblib"
        )
        if candidate.exists():
            return candidate
    return Path(
        "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
        "simulator_output/GV3_Agent/AlphaGoZero/models/Model_Version139/"
        "value/hgb_sq_63leaf_1050iter_a2/model.joblib"
    )


def _ensure_callable_harness_model(model_path: Path, wrapper_path: Path) -> Path:
    model_path = Path(model_path).expanduser()
    model = joblib.load(model_path)
    if callable(getattr(model, "infer_from_inputs", None)):
        return model_path
    from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper

    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = V4AdvHGBWrapper(model, feature_dim=226, model_tag=f"fixed_trace_harness:{model_path.name}")
    joblib.dump(wrapped, wrapper_path, compress=3)
    return Path(wrapper_path)


def _model_cpp_payload(
    *,
    cfg: FixedTraceEvalConfig,
    tester_cfg: Any,
    bundle: Any,
) -> dict[str, Any]:
    pipeline_cfg = tester_cfg.to_pipeline_cfg()
    pipeline_cfg = replace(
        pipeline_cfg,
        game_v2=replace(
            pipeline_cfg.game_v2,
            mcts_search=replace(
                pipeline_cfg.game_v2.mcts_search,
                discount_factor=float(cfg.discount_factor),
            ),
        ),
    )
    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, bundle.simulator)
    payload["use_model_bootstrap"] = bool(int(cfg.model_version) > 0) and not bool(cfg.disable_model_bootstrap)
    payload["native_search_mode"] = str(cfg.native_search_mode)
    payload["rollout_count"] = int(cfg.rollout_count)
    payload["rollout_parallel_threads"] = int(cfg.rollout_parallel_threads)
    payload["rollout_horizon_sec"] = float(cfg.rollout_horizon_sec)
    payload["rollout_policy_temperature"] = float(cfg.rollout_policy_temperature)
    payload["rollout_probability_quantum"] = float(cfg.rollout_probability_quantum)
    payload["rollout_max_actions"] = int(cfg.rollout_max_actions)
    payload["use_policy_prior"] = True
    payload["root_dirichlet_noise_enabled"] = False
    payload["root_dirichlet_alpha"] = 0.3
    payload["root_dirichlet_epsilon"] = 0.0
    payload["pb_c_base"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(cfg.uct_c)
    payload["puct_c"] = float(cfg.puct_c)
    payload["policy_prior_temperature"] = float(cfg.policy_prior_temperature)
    payload["prior_min_prob"] = float(cfg.prior_min_prob)
    payload["max_forced_hops"] = int(getattr(pipeline_cfg, "max_forced_hops_per_root", 0) or 0)
    return payload


def _has_prefill_work(bundle: Any, state: Any) -> bool:
    env = bundle.env
    req_map = env._req_map(state.simulator)
    for rid in sorted(int(x) for x in (getattr(state.stats, "active_request_ids", set()) or set())):
        req = req_map.get(int(rid))
        if req is None or bool(getattr(req, "completed", False)):
            continue
        if int(env._remaining_prefill(req)) > 0:
            return True
    return False


def _has_decode_work(bundle: Any, state: Any) -> bool:
    env = bundle.env
    req_map = env._req_map(state.simulator)
    for rid in sorted(int(x) for x in (getattr(state.stats, "active_request_ids", set()) or set())):
        req = req_map.get(int(rid))
        if req is None or bool(getattr(req, "completed", False)):
            continue
        prefill_done = bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)))
        if prefill_done and int(env._remaining_decode(req)) > 0:
            return True
    return False


def _expand_controller_actions(bundle: Any, state: Any) -> Any:
    return tester_runner._expand_action_space(
        bundle=bundle,
        state=state,
        player="controller",
        pending_adv_pre_ctrl_snapshot=None,
        pending_adv_pre_ctrl_stats=None,
    )


def _select_decode_drain_action(bundle: Any, state: Any) -> tuple[ControllerAction | None, dict[str, Any]]:
    expanded = _expand_controller_actions(bundle, state)
    candidates: list[tuple[int, int, ControllerAction]] = []
    for idx in expanded.valid_indices:
        action = expanded.actions_by_index[int(idx)]
        if not isinstance(action, ControllerAction):
            continue
        prefill_alloc = sum(int(v) for v in (getattr(action, "prefill_allocations", {}) or {}).values())
        decode_alloc = sum(int(v) for v in (getattr(action, "decode_allocations", {}) or {}).values())
        if int(prefill_alloc) == 0 and int(decode_alloc) > 0:
            candidates.append((int(decode_alloc), -int(idx), action))

    if candidates:
        decode_alloc, neg_idx, action = max(candidates, key=lambda t: (int(t[0]), int(t[1])))
        return action, {
            "selection_mode": "trace_decode_drain_max_decode",
            "valid_action_count": int(len(expanded.valid_indices)),
            "canonical_action_count": "",
            "iterations_requested": 0,
            "iterations_used": 0,
            "chosen_decode_alloc": int(decode_alloc),
            "chosen_action_index": int(-neg_idx),
        }

    return make_strict_noop_controller_action(eviction_rule="evict_none"), {
        "selection_mode": "trace_decode_drain_no_decode_action_noop",
        "valid_action_count": int(len(expanded.valid_indices)),
        "canonical_action_count": "",
        "iterations_requested": 0,
        "iterations_used": 0,
    }


def _select_sjf_action(
    *,
    cfg: FixedTraceEvalConfig,
    bundle: Any,
    state: Any,
) -> tuple[ControllerAction | None, dict[str, Any]]:
    expanded = _expand_controller_actions(bundle, state)
    action, info = select_trivial_controller_action(
        runner=bundle.runner,
        state=state,
        heuristic=str(cfg.sjf_heuristic),
        budget_tokens=int(cfg.sjf_budget_tokens),
        eviction_rule=str(cfg.sjf_eviction_rule),
        actions_by_index=expanded.actions_by_index,
        valid_indices=expanded.valid_indices,
    )
    return action, dict(info or {})


def _select_model_cpp_action(
    *,
    cfg: FixedTraceEvalConfig,
    tester_cfg: Any,
    runtime: _NativeModelRuntime,
    payload: dict[str, Any],
    bundle: Any,
    state: Any,
    step: int,
    depth: int,
) -> tuple[ControllerAction | None, dict[str, Any]]:
    expanded = _expand_controller_actions(bundle, state)
    valid_indices = [int(x) for x in list(expanded.valid_indices)]
    actions_by_index = list(expanded.actions_by_index)
    if not valid_indices:
        return None, {
            "selection_mode": "model_mcts_cpp_controller_no_valid_action",
            "valid_action_count": 0,
            "canonical_action_count": 0,
            "iterations_requested": 0,
            "iterations_used": 0,
        }
    if len(valid_indices) == 1:
        idx = int(valid_indices[0])
        return actions_by_index[idx], {
            "selection_mode": "model_mcts_cpp_controller_single_valid_action",
            "valid_action_count": 1,
            "canonical_action_count": 1,
            "iterations_requested": 0,
            "iterations_used": 0,
            "candidate_top5_visits": [0],
            "candidate_top5_priors": [],
            "candidate_top5_action_reprs": [repr(actions_by_index[idx])],
        }

    root_id = int(91_000_000 + int(step))
    iterations = max(int(cfg.mcts_iterations), int(len(valid_indices)))
    native_out = runtime.native.search_mcts_hgb226_value_prior_hgb(
        runtime.value_runtime,
        runtime.controller_prior_runtime,
        runtime.adversary_prior_runtime,
        int(cfg.model_version),
        nlt._native_state_payload(bundle.env, expanded.search_state),
        dict(payload),
        int(iterations),
        "controller",
        int(root_id),
        int(depth),
        91_000_000,
        int(root_id),
        int(cfg.seed) + int(root_id) + int(step),
        False,
        False,
        "",
        "",
    )

    q_values = [float(x) for x in list(native_out.get("root_action_values", []) or [])]
    rewards = [float(x) for x in list(native_out.get("root_action_rewards", []) or [])]
    discounts = [float(x) for x in list(native_out.get("root_action_discounts", []) or [])]
    bootstraps = [float(x) for x in list(native_out.get("root_action_bootstraps", []) or [])]
    mcts_probs = [float(x) for x in list(native_out.get("mcts_root_prior", []) or [])]
    children = list(native_out.get("children", []) or [])
    alias_to_canon = {int(k): int(v) for k, v in dict(native_out.get("action_alias_to_canonical", {}) or {}).items()}
    canonical_indices = sorted({int(alias_to_canon.get(i, i)) for i in valid_indices})

    rows: list[dict[str, Any]] = []
    for child in children:
        idx = int(child.get("index", -1))
        if idx < 0 or idx >= len(actions_by_index) or actions_by_index[idx] is None:
            continue
        visits = int(child.get("visits", 0) or 0)
        q = q_values[idx] if 0 <= idx < len(q_values) else float("-inf")
        rows.append(
            {
                "idx": int(idx),
                "visits": int(visits),
                "q_value": float(q),
                "prior": float(child.get("prior", 0.0) or 0.0),
                "mcts_prob": mcts_probs[idx] if 0 <= idx < len(mcts_probs) else 0.0,
                "reward": rewards[idx] if 0 <= idx < len(rewards) else 0.0,
                "discount": discounts[idx] if 0 <= idx < len(discounts) else 1.0,
                "bootstrap": bootstraps[idx] if 0 <= idx < len(bootstraps) else 0.0,
                "child_cost": float(child.get("state_cost", 0.0) or 0.0),
                "action_repr": repr(actions_by_index[idx]),
            }
        )

    visited_rows = [r for r in rows if int(r["visits"]) > 0]
    if not visited_rows:
        idx = int(valid_indices[0])
        return actions_by_index[idx], {
            "selection_mode": "model_mcts_cpp_controller_no_visited_fallback_first_valid",
            "valid_action_count": int(len(valid_indices)),
            "canonical_action_count": int(len(canonical_indices)),
            "iterations_requested": int(iterations),
            "iterations_used": int(native_out.get("root_visits", 0) or 0),
        }

    ranked = sorted(visited_rows, key=lambda r: (-int(r["visits"]), -float(r["q_value"]), int(r["idx"])))
    best = ranked[0]
    best_idx = int(best["idx"])
    top5 = ranked[:5]
    root_visits = int(native_out.get("root_visits", 0) or 0)
    root_value_sum = float(native_out.get("root_value_sum", 0.0) or 0.0)
    root_value = float(root_value_sum / root_visits) if root_visits > 0 else 0.0
    return actions_by_index[best_idx], {
        "selection_mode": "model_mcts_cpp_shared_root_ctrl_most_visits",
        "valid_action_count": int(len(valid_indices)),
        "canonical_action_count": int(len(canonical_indices)),
        "iterations_requested": int(iterations),
        "iterations_used": int(root_visits),
        "chosen_q_value": float(best["q_value"]),
        "chosen_reward": float(best["reward"]),
        "chosen_discount": float(best["discount"]),
        "chosen_bootstrap": float(best["bootstrap"]),
        "chosen_child_cost": float(best["child_cost"]),
        "model_value_at_state": float(native_out.get("root_nn_value_controller", 0.0) or 0.0),
        "mcts_root_value": float(root_value),
        "candidate_ranking_mode": "visits_desc_q_desc",
        "candidate_top5_action_reprs": [str(r["action_repr"]) for r in top5],
        "candidate_top5_q_values": [float(r["q_value"]) for r in top5],
        "candidate_top5_rewards": [float(r["reward"]) for r in top5],
        "candidate_top5_discounts": [float(r["discount"]) for r in top5],
        "candidate_top5_bootstraps": [float(r["bootstrap"]) for r in top5],
        "candidate_top5_child_costs": [float(r["child_cost"]) for r in top5],
        "candidate_top5_visits": [int(r["visits"]) for r in top5],
        "candidate_top5_priors": [float(r["prior"]) for r in top5],
        "candidate_top5_mcts_probs": [float(r["mcts_prob"]) for r in top5],
        "policy_prior_temperature": float(cfg.policy_prior_temperature),
        "mcts_action_temperature": 0.0,
        "mcts_action_sample_count": 0,
    }


class FixedTraceEvalRunner:
    _STEP_FIELDS = [
        "step",
        "policy",
        "phase",
        "sim_time_before",
        "sim_time_after",
        "rows_injected",
        "created_request_ids",
        "action_repr",
        "selection_mode",
        "valid_action_count",
        "canonical_action_count",
        "iterations_requested",
        "iterations_used",
        "chosen_q_value",
        "chosen_reward",
        "chosen_discount",
        "chosen_bootstrap",
        "chosen_child_cost",
        "model_value_at_state",
        "mcts_root_value",
        "candidate_top5_visits",
        "candidate_top5_priors",
        "candidate_top5_mcts_probs",
        "active_request_ids",
        "completed_request_ids",
        "decode_credit_balance",
        "decode_processed_tokens_by_id",
        "prefill_remaining_by_id",
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "total_lateness",
        "total_cost",
    ]

    def __init__(self, cfg: FixedTraceEvalConfig) -> None:
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps_csv = self.cfg.output_dir / "trace_steps.csv"
        self.summary_csv = self.cfg.output_dir / "summary.csv"
        self.validation_csv = self.cfg.output_dir / "validation.csv"

        self.runtime: _NativeModelRuntime | None = None
        if str(cfg.policy) == "model_cpp":
            self.runtime = _prepare_native_model_runtime(cfg)

        self.tester_cfg = _build_tester_config(cfg, self.runtime)
        self.bundle = tester_runner._build_bundle(self.tester_cfg)
        self.payload = (
            _model_cpp_payload(cfg=cfg, tester_cfg=self.tester_cfg, bundle=self.bundle)
            if self.runtime is not None
            else {}
        )

        rows = load_trace_csv(cfg.trace_csv, limit=int(cfg.max_trace_rows))
        self.trace_rows = rebase_trace(rows) if bool(cfg.rebase_to_zero) else rows
        self.trace_ptr = 0
        self.rows_released = 0
        self.expected_rows_by_limit = sum(
            1
            for row in self.trace_rows
            if float(row.arrived_at) <= float(self.cfg.time_limit_sec) + float(self.bundle.env._EPS)
        )
        self.injected_trace_row_ids: list[int] = []
        self.prefill_policy_steps = 0
        self.decode_drain_steps = 0

    def close(self) -> None:
        try:
            self.bundle.mcts.clear_search_state(drop_scratch=True)
        except Exception:
            pass
        try:
            tester_runner._safe_close_simulator(self.bundle.simulator)
        except Exception:
            pass
        gc.collect()

    def _trace_dict(self, row: TraceRequest) -> dict[str, Any]:
        return {
            "trace_row_id": int(row.trace_row_id),
            "arrived_at": float(row.arrived_at),
            "num_prefill_tokens": int(row.num_prefill_tokens),
            "num_decode_tokens": int(row.num_decode_tokens),
        }

    def _inject_due_rows(self, state: Any, *, due_time: float) -> tuple[Any, dict[str, Any]]:
        due: list[TraceRequest] = []
        eps = float(self.bundle.env._EPS)
        limit_t = float(self.cfg.time_limit_sec)
        due_cap = min(float(due_time), limit_t)
        while self.trace_ptr < len(self.trace_rows):
            row = self.trace_rows[self.trace_ptr]
            if float(row.arrived_at) > due_cap + eps:
                break
            due.append(row)
            self.trace_ptr += 1

        if not due:
            return state, {
                "created_count": 0,
                "created_request_ids": [],
                "created_rows": [],
            }

        state, info = self.bundle.env.inject_trace_requests(
            state,
            [self._trace_dict(row) for row in due],
            inplace=True,
            token_policy=str(self.cfg.token_policy),
            slo_policy="gv3_default",
            allow_future_arrivals=False,
            record_launch_history=bool(self.cfg.record_launch_history),
            enforce_gv3_launch_constraints=bool(self.cfg.strict_gv3_trace),
        )
        self.rows_released += int(len(due))
        self.injected_trace_row_ids.extend(int(row.trace_row_id) for row in due)
        return state, dict(info or {})

    def _jump_to_next_arrival_or_limit(self, state: Any) -> tuple[Any, bool]:
        limit_t = float(self.cfg.time_limit_sec)
        now = float(state.simulator._time)
        next_arrival = (
            float(self.trace_rows[self.trace_ptr].arrived_at)
            if self.trace_ptr < len(self.trace_rows)
            else None
        )
        target = limit_t if next_arrival is None else min(float(next_arrival), limit_t)
        if target > now + float(self.bundle.env._EPS):
            state.simulator._set_time(float(target))
        return state, bool(next_arrival is not None and float(next_arrival) <= limit_t + float(self.bundle.env._EPS))

    def _select_prefill_action(self, state: Any, *, step: int, depth: int) -> tuple[ControllerAction | None, dict[str, Any]]:
        if str(self.cfg.policy) == "sjf512":
            return _select_sjf_action(cfg=self.cfg, bundle=self.bundle, state=state)
        if str(self.cfg.policy) == "model_cpp":
            if self.runtime is None:
                raise RuntimeError("model_cpp runtime was not prepared")
            return _select_model_cpp_action(
                cfg=self.cfg,
                tester_cfg=self.tester_cfg,
                runtime=self.runtime,
                payload=self.payload,
                bundle=self.bundle,
                state=state,
                step=int(step),
                depth=int(depth),
            )
        raise ValueError(f"unsupported fixed trace policy: {self.cfg.policy!r}")

    def _write_step(self, row: dict[str, Any]) -> None:
        write_header = not self.steps_csv.exists()
        with self.steps_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._STEP_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._STEP_FIELDS})

    def _log_step(
        self,
        *,
        state: Any,
        step: int,
        phase: str,
        sim_before: float,
        action: ControllerAction | None,
        selection_info: dict[str, Any],
        injection_info: dict[str, Any],
    ) -> None:
        violations, lateness = self.bundle.env.evaluate_objective(state)
        total_cost = float(violations) + float(lateness)
        state_log = arena_state_snapshot_for_log(self.bundle.env, state)
        self._write_step(
            {
                "step": int(step),
                "policy": str(self.cfg.policy),
                "phase": str(phase),
                "sim_time_before": float(sim_before),
                "sim_time_after": float(state.simulator._time),
                "rows_injected": int(injection_info.get("created_count", 0) or 0),
                "created_request_ids": json.dumps([int(x) for x in list(injection_info.get("created_request_ids", []) or [])]),
                "action_repr": "" if action is None else repr(action),
                "selection_mode": str(selection_info.get("selection_mode", "")),
                "valid_action_count": selection_info.get("valid_action_count", ""),
                "canonical_action_count": selection_info.get("canonical_action_count", ""),
                "iterations_requested": selection_info.get("iterations_requested", ""),
                "iterations_used": selection_info.get("iterations_used", ""),
                "chosen_q_value": selection_info.get("chosen_q_value", ""),
                "chosen_reward": selection_info.get("chosen_reward", ""),
                "chosen_discount": selection_info.get("chosen_discount", ""),
                "chosen_bootstrap": selection_info.get("chosen_bootstrap", ""),
                "chosen_child_cost": selection_info.get("chosen_child_cost", ""),
                "model_value_at_state": selection_info.get("model_value_at_state", ""),
                "mcts_root_value": selection_info.get("mcts_root_value", ""),
                "candidate_top5_visits": json.dumps(list(selection_info.get("candidate_top5_visits", []) or [])),
                "candidate_top5_priors": json.dumps(list(selection_info.get("candidate_top5_priors", []) or [])),
                "candidate_top5_mcts_probs": json.dumps(list(selection_info.get("candidate_top5_mcts_probs", []) or [])),
                "active_request_ids": json.dumps([int(x) for x in state_log["active_request_ids"]]),
                "completed_request_ids": json.dumps([int(x) for x in state_log["completed_request_ids"]]),
                "decode_credit_balance": int(state_log["decode_credit_balance"]),
                "decode_processed_tokens_by_id": json.dumps(state_log["decode_processed_tokens_by_id"], sort_keys=True),
                "prefill_remaining_by_id": json.dumps(state_log["prefill_remaining_by_id"], sort_keys=True),
                "requests_generated": int(state.stats.requests_generated),
                "requests_completed": int(state.stats.requests_completed),
                "slo_violations": int(violations),
                "total_lateness": float(lateness),
                "total_cost": float(total_cost),
            }
        )

    def run(self) -> FixedTraceEvalResult:
        state = self.bundle.env.initial_state()
        depth = 0
        step = 0
        end_reason = "max_steps"
        injection_info: dict[str, Any] = {
            "created_count": 0,
            "created_request_ids": [],
            "created_rows": [],
        }

        while step < int(self.cfg.max_steps):
            state = self.bundle.env.prepare_trace_controller_turn(state, inplace=True)
            state, injection_info = self._inject_due_rows(state, due_time=float(state.simulator._time))

            has_prefill = _has_prefill_work(self.bundle, state)
            has_decode = _has_decode_work(self.bundle, state)
            has_future_rows = self.trace_ptr < len(self.trace_rows)
            now = float(state.simulator._time)

            if now >= float(self.cfg.time_limit_sec) - float(self.bundle.env._EPS):
                end_reason = "time_limit"
                break

            if not has_prefill and not has_decode:
                if not has_future_rows:
                    state.simulator._set_time(float(self.cfg.time_limit_sec))
                    end_reason = "trace_exhausted"
                    break
                state, has_arrival = self._jump_to_next_arrival_or_limit(state)
                if not has_arrival and float(state.simulator._time) >= float(self.cfg.time_limit_sec) - float(self.bundle.env._EPS):
                    end_reason = "time_limit"
                    break
                continue

            sim_before = float(state.simulator._time)
            if has_prefill:
                action, selection_info = self._select_prefill_action(state, step=int(step), depth=int(depth))
                phase = "prefill_policy"
                self.prefill_policy_steps += 1
            else:
                action, selection_info = _select_decode_drain_action(self.bundle, state)
                phase = "decode_drain"
                self.decode_drain_steps += 1

            if action is None:
                end_reason = "no_valid_action"
                break

            state = self.bundle.env.apply_controller_action_only(
                state,
                action,
                inplace=True,
                fast_forward=False,
            )
            depth += 1
            step += 1

            # If an action crosses a trace arrival boundary, inject those rows
            # before logging the resulting state/cost. Cap at the trace horizon.
            state, post_injection_info = self._inject_due_rows(
                state,
                due_time=min(float(state.simulator._time), float(self.cfg.time_limit_sec)),
            )
            merged_injection_info = {
                "created_count": int(injection_info.get("created_count", 0) or 0)
                + int(post_injection_info.get("created_count", 0) or 0),
                "created_request_ids": list(injection_info.get("created_request_ids", []) or [])
                + list(post_injection_info.get("created_request_ids", []) or []),
                "created_rows": list(injection_info.get("created_rows", []) or [])
                + list(post_injection_info.get("created_rows", []) or []),
            }
            self._log_step(
                state=state,
                step=int(step),
                phase=str(phase),
                sim_before=float(sim_before),
                action=action,
                selection_info=selection_info,
                injection_info=merged_injection_info,
            )
        else:
            end_reason = "max_steps"

        violations, lateness = self.bundle.env.evaluate_objective(state)
        total_cost = float(violations) + float(lateness)
        result = FixedTraceEvalResult(
            policy=str(self.cfg.policy),
            output_dir=Path(self.cfg.output_dir),
            steps_csv=self.steps_csv,
            summary_csv=self.summary_csv,
            validation_csv=self.validation_csv,
            rows_read=int(len(self.trace_rows)),
            rows_released=int(self.rows_released),
            requests_generated=int(state.stats.requests_generated),
            requests_completed=int(state.stats.requests_completed),
            final_sim_time=float(state.simulator._time),
            slo_violations=int(violations),
            total_lateness=float(lateness),
            total_cost=float(total_cost),
            prefill_policy_steps=int(self.prefill_policy_steps),
            decode_drain_steps=int(self.decode_drain_steps),
            end_reason=str(end_reason),
        )
        self._write_summary(result)
        self._write_validation(result)
        return result

    def _write_summary(self, result: FixedTraceEvalResult) -> None:
        fields = [
            "policy",
            "trace_csv",
            "steps_csv",
            "rows_read",
            "expected_rows_by_time_limit",
            "rows_released",
            "requests_generated",
            "requests_completed",
            "final_sim_time",
            "slo_violations",
            "total_lateness",
            "total_cost",
            "prefill_policy_steps",
            "decode_drain_steps",
            "end_reason",
            "model_version",
            "mcts_iterations",
            "discount_factor",
            "native_search_mode",
            "rollout_count",
            "rollout_parallel_threads",
            "rollout_horizon_sec",
            "rollout_policy_temperature",
            "rollout_probability_quantum",
            "rollout_max_actions",
        ]
        with self.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "policy": str(result.policy),
                    "trace_csv": str(self.cfg.trace_csv),
                    "steps_csv": str(result.steps_csv),
                    "rows_read": int(result.rows_read),
                    "expected_rows_by_time_limit": int(self.expected_rows_by_limit),
                    "rows_released": int(result.rows_released),
                    "requests_generated": int(result.requests_generated),
                    "requests_completed": int(result.requests_completed),
                    "final_sim_time": float(result.final_sim_time),
                    "slo_violations": int(result.slo_violations),
                    "total_lateness": float(result.total_lateness),
                    "total_cost": float(result.total_cost),
                    "prefill_policy_steps": int(result.prefill_policy_steps),
                    "decode_drain_steps": int(result.decode_drain_steps),
                    "end_reason": str(result.end_reason),
                    "model_version": int(self.cfg.model_version),
                    "mcts_iterations": int(self.cfg.mcts_iterations),
                    "discount_factor": float(self.cfg.discount_factor),
                    "native_search_mode": str(self.cfg.native_search_mode),
                    "rollout_count": int(self.cfg.rollout_count),
                    "rollout_parallel_threads": int(self.cfg.rollout_parallel_threads),
                    "rollout_horizon_sec": float(self.cfg.rollout_horizon_sec),
                    "rollout_policy_temperature": float(self.cfg.rollout_policy_temperature),
                    "rollout_probability_quantum": float(self.cfg.rollout_probability_quantum),
                    "rollout_max_actions": int(self.cfg.rollout_max_actions),
                }
            )

    def _write_validation(self, result: FixedTraceEvalResult) -> None:
        checks = [
            (
                "all_due_trace_rows_released",
                int(result.rows_released) == int(self.expected_rows_by_limit),
                f"released={result.rows_released} expected={self.expected_rows_by_limit}",
            ),
            (
                "requests_generated_matches_released",
                int(result.requests_generated) == int(result.rows_released),
                f"generated={result.requests_generated} released={result.rows_released}",
            ),
            (
                "no_duplicate_trace_row_injection",
                len(set(self.injected_trace_row_ids)) == len(self.injected_trace_row_ids),
                f"injected={len(self.injected_trace_row_ids)} unique={len(set(self.injected_trace_row_ids))}",
            ),
            (
                "final_time_reached_limit",
                float(result.final_sim_time) >= float(self.cfg.time_limit_sec) - float(self.bundle.env._EPS),
                f"final={result.final_sim_time} limit={self.cfg.time_limit_sec}",
            ),
            (
                "trace_launch_history_recorded",
                bool(self.cfg.record_launch_history),
                f"record_launch_history={int(bool(self.cfg.record_launch_history))}",
            ),
            (
                "strict_gv3_trace_validation_enabled",
                bool(self.cfg.strict_gv3_trace),
                f"strict_gv3_trace={int(bool(self.cfg.strict_gv3_trace))}",
            ),
        ]
        with self.validation_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["check", "passed", "details"])
            writer.writeheader()
            for name, passed, details in checks:
                writer.writerow({"check": str(name), "passed": int(bool(passed)), "details": str(details)})


def run_fixed_trace_eval(cfg: FixedTraceEvalConfig) -> FixedTraceEvalResult:
    runner = FixedTraceEvalRunner(cfg)
    try:
        return runner.run()
    finally:
        runner.close()
