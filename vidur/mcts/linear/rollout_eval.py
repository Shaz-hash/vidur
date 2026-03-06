from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..DNN.history_root import HistoryRootGenerator
from .bellman import compute_base_step_time, effective_discount, state_cost, transition_reward
from .collector_worker import (
    _actions_and_valid,
    _advance_forced_until_branching,
    _advance_to_controller_branching,
    _apply_action_inplace,
    _build_env,
    _is_explicit_history_row,
    _load_history_rows,
    _parse_action_json,
)
from .config import LinearPipelineConfig, WorkerTask, round_dir
from .features import extract_features
from .model import LinearValueModel


class _HistoryCaptureIterLogger:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def log_expand(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))


def _state_row(env: Any, state: Any, *, adversary_prefill_deadlines_by_id: str = "{}") -> Dict[str, Any]:
    d = env.describe_state(state)
    return {
        "sim_time": float(d.get("sim_time", 0.0)),
        "requests_in_system": int(d.get("requests_in_system", 0)),
        "state_waiting_ids": json.dumps(d.get("waiting_request_ids", []), ensure_ascii=False),
        "state_completed_request_ids": json.dumps(d.get("completed_request_ids", []), ensure_ascii=False),
        "slo_violations": int(d.get("slo_violations", 0)),
        "total_lateness": float(d.get("total_lateness", 0.0)),
        "total_cost": float(state_cost(env, state)),
        "adversary_prefill_deadlines_by_id": str(adversary_prefill_deadlines_by_id),
    }


def _adversary_new_prefill_deadlines_json(
    env: Any,
    *,
    state_after: Any,
    before_request_ids: set[int],
) -> str:
    try:
        lookup = env._build_request_lookup(state_after.simulator)
    except Exception:
        return "{}"

    new_ids = sorted(int(rid) for rid in set(lookup.keys()) - set(before_request_ids))
    if not new_ids:
        return "{}"

    out: Dict[int, float] = {}
    for rid in new_ids:
        req = lookup.get(rid)
        if req is None:
            continue
        queued_at = float(getattr(req, "queued_at", getattr(req, "arrived_at", 0.0)) or 0.0)
        slo = float(getattr(req, "prefill_slo_time", getattr(req, "_prefill_slo_time", 0.0)) or 0.0)
        out[int(rid)] = queued_at + slo

    return json.dumps(out, ensure_ascii=False)


def _append_row(
    rows: List[Dict[str, Any]],
    *,
    trace_id: int,
    history_hop: int,
    step_index: int,
    phase: str,
    acting_player: str,
    state_info: Dict[str, Any],
    best_action_index: Optional[int] = None,
    best_action_repr: str = "",
    num_valid_actions: Optional[int] = None,
    q_values_by_action_json: str = "",
    v_hat_s: Optional[float] = None,
    cost_s: Optional[float] = None,
    cost_next: Optional[float] = None,
    reward: Optional[float] = None,
    delta_time: Optional[float] = None,
    gamma_eff: Optional[float] = None,
    adversary_requests_generated: int = 0,
    next_player: str = "",
) -> None:
    rows.append(
        {
            "trace_id": int(trace_id),
            "history_hop": int(history_hop),
            "step_index": int(step_index),
            "phase": str(phase),
            "acting_player": str(acting_player),
            "best_action_index": "" if best_action_index is None else int(best_action_index),
            "best_action_repr": str(best_action_repr),
            "num_valid_actions": "" if num_valid_actions is None else int(num_valid_actions),
            "q_values_by_action_json": str(q_values_by_action_json),
            "v_hat_s": "" if v_hat_s is None else float(v_hat_s),
            "cost_s": "" if cost_s is None else float(cost_s),
            "cost_next": "" if cost_next is None else float(cost_next),
            "reward": "" if reward is None else float(reward),
            "delta_time": "" if delta_time is None else float(delta_time),
            "gamma_eff": "" if gamma_eff is None else float(gamma_eff),
            "sim_time": float(state_info.get("sim_time", 0.0)),
            "slo_violations": int(state_info.get("slo_violations", 0)),
            "total_lateness": float(state_info.get("total_lateness", 0.0)),
            "total_cost": float(state_info.get("total_cost", 0.0)),
            "requests_in_system": int(state_info.get("requests_in_system", 0)),
            "state_waiting_ids": str(state_info.get("state_waiting_ids", "[]")),
            "state_completed_request_ids": str(state_info.get("state_completed_request_ids", "[]")),
            "adversary_requests_generated": int(adversary_requests_generated),
            "adversary_prefill_deadlines_by_id": str(state_info.get("adversary_prefill_deadlines_by_id", "{}")),
            "next_player": str(next_player),
        }
    )


def _generate_root_with_history_logs(
    *,
    cfg: LinearPipelineConfig,
    env: Any,
    task: WorkerTask,
    trace_id: int,
    rows: List[Dict[str, Any]],
    step_index: int,
) -> tuple[Any, str, int]:
    root_player = str(task.cfg.collection.root_player)
    state = env.initial_state()
    player = root_player

    history_csv = str(task.cfg.collection.history_csv or "").strip()
    if history_csv:
        hist_rows = _load_history_rows(Path(history_csv))
        action_rows = [r for r in hist_rows if (r.best_action_json or "").strip()]

        for i, row in enumerate(action_rows):
            is_hist = _is_explicit_history_row(row)
            if bool(task.cfg.collection.align_branching_roots) and (not is_hist):
                state, player = _advance_forced_until_branching(
                    env,
                    state,
                    player,
                    max_hops=cfg.collection.max_forced_hops,
                    max_samples=cfg.collection.enum_max_samples,
                )

            if row.root_player and row.root_player != player:
                raise RuntimeError(
                    f"History replay mismatch at row={i}: row.root_player={row.root_player}, player={player}"
                )

            actions_by_index, valid = _actions_and_valid(env, state, player, cfg.collection.enum_max_samples)
            action = _parse_action_json(row.best_action_json)

            before_ids: set[int] = set()
            if player == "adversary":
                try:
                    before_ids = set(env._build_request_lookup(state.simulator).keys())
                except Exception:
                    before_ids = set()

            acting_player = str(player)
            state, player = _apply_action_inplace(env, state, player, action)

            if acting_player == "adversary":
                adv_deadlines = _adversary_new_prefill_deadlines_json(
                    env,
                    state_after=state,
                    before_request_ids=before_ids,
                )
                try:
                    adv_generated = len(json.loads(adv_deadlines))
                except Exception:
                    adv_generated = 0
            else:
                adv_deadlines = "{}"
                adv_generated = 0

            srow = _state_row(env, state, adversary_prefill_deadlines_by_id=adv_deadlines)
            _append_row(
                rows,
                trace_id=trace_id,
                history_hop=int(task.history_hop),
                step_index=step_index,
                phase="history_csv",
                acting_player=acting_player,
                state_info=srow,
                best_action_index=None,
                best_action_repr=repr(action),
                num_valid_actions=int(len(valid)),
                adversary_requests_generated=int(adv_generated),
                next_player=player,
            )
            step_index += 1

            if i < len(action_rows) - 1 and bool(task.cfg.collection.align_branching_roots) and (not is_hist):
                state, player = _advance_forced_until_branching(
                    env,
                    state,
                    player,
                    max_hops=cfg.collection.max_forced_hops,
                    max_samples=cfg.collection.enum_max_samples,
                )

    hist_capture = _HistoryCaptureIterLogger()
    hgen = HistoryRootGenerator(
        env=env,
        max_branching=int(task.cfg.collection.max_branching),
        iter_logger=hist_capture,
        root_logger=None,
    )

    state, player, _depth, _next_node_id, _last_node_id = hgen.generate_history_root(
        state=state,
        player=player,
        depth=0,
        nontrivial_hops=int(task.history_hop),
        game_id=int(task.round_idx) * 1000 + int(task.worker_id),
        root_id_for_logs=int(task.worker_id),
        seed=int(task.seed),
        log_history=True,
        max_total_steps=int(task.cfg.collection.max_forced_hops),
        log_node_id_start=0,
        log_parent_id_start=None,
    )

    for hr in hist_capture.rows:
        snap = hr.get("state_snapshot", {}) or {}
        adv_deadlines = str(hr.get("adversary_prefill_deadlines_by_id_json", "{}"))
        try:
            adv_generated = len(json.loads(adv_deadlines))
        except Exception:
            adv_generated = 0
        srow = {
            "sim_time": float(snap.get("sim_time", 0.0)),
            "slo_violations": int(snap.get("slo_violations", 0)),
            "total_lateness": float(snap.get("total_lateness", 0.0)),
            "total_cost": float(hr.get("objective_cost", 0.0)),
            "requests_in_system": int(snap.get("requests_in_system", 0)),
            "state_waiting_ids": json.dumps(snap.get("waiting_request_ids", []), ensure_ascii=False),
            "state_completed_request_ids": json.dumps(snap.get("completed_request_ids", []), ensure_ascii=False),
            "adversary_prefill_deadlines_by_id": adv_deadlines,
        }
        _append_row(
            rows,
            trace_id=trace_id,
            history_hop=int(task.history_hop),
            step_index=step_index,
            phase=str(hr.get("phase", "history_generated")),
            acting_player=str(hr.get("player_acted_to_create_this_node", "")),
            state_info=srow,
            best_action_index=int(hr.get("action_index", 0)),
            best_action_repr=str(hr.get("action_repr", "")),
            num_valid_actions=int(hr.get("num_valid_actions", 0)),
            adversary_requests_generated=int(adv_generated),
            next_player=str(hr.get("player_to_act", "")),
        )
        step_index += 1

    return state, player, step_index


def run_greedy_rollouts(
    *,
    cfg: LinearPipelineConfig,
    round_idx: int,
    model: LinearValueModel,
) -> List[Dict[str, Any]]:
    rd = round_dir(cfg, round_idx)
    rd.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    base_step_time = compute_base_step_time(
        _build_env(cfg, use_virtual_env=bool(cfg.collection.use_virtual_env))[1],
        cfg.bellman.base_step_tokens,
    )

    summaries: List[Dict[str, Any]] = []
    hops = list(cfg.collection.history_hops) or [0]

    for trace_id in range(int(cfg.rollout.num_traces)):
        _, env = _build_env(cfg, use_virtual_env=bool(cfg.collection.use_virtual_env))
        task = WorkerTask(
            worker_id=int(trace_id),
            round_idx=int(round_idx),
            seed=int(cfg.seed + 500000 + round_idx * 1000 + trace_id),
            history_hop=int(hops[trace_id % len(hops)]),
            out_dir=str(rd),
            model_ckpt_path="",
            cfg=cfg,
        )

        rows: List[Dict[str, Any]] = []
        step_index = 0

        state, player, step_index = _generate_root_with_history_logs(
            cfg=cfg,
            env=env,
            task=task,
            trace_id=trace_id,
            rows=rows,
            step_index=step_index,
        )

        state, player, terminal = _advance_to_controller_branching(
            env,
            state,
            player,
            max_hops=cfg.collection.max_forced_hops,
            max_samples=cfg.collection.enum_max_samples,
        )

        out_csv = rd / f"greedy_trace_worker_{trace_id:02d}.csv"
        fieldnames = [
            "trace_id",
            "history_hop",
            "step_index",
            "phase",
            "acting_player",
            "best_action_index",
            "best_action_repr",
            "num_valid_actions",
            "q_values_by_action_json",
            "v_hat_s",
            "cost_s",
            "cost_next",
            "reward",
            "delta_time",
            "gamma_eff",
            "sim_time",
            "slo_violations",
            "total_lateness",
            "total_cost",
            "requests_in_system",
            "state_waiting_ids",
            "state_completed_request_ids",
            "adversary_requests_generated",
            "adversary_prefill_deadlines_by_id",
            "next_player",
        ]

        decision_steps = 0

        while (not terminal) and decision_steps < int(cfg.rollout.max_steps):
            if player != "controller":
                def _pre_controller_cb(ev: Dict[str, Any]) -> None:
                    nonlocal step_index
                    st_after = ev["state_after"]
                    acting = str(ev.get("acting_player", ""))
                    before_ids = set(ev.get("before_request_ids", set()) or set())
                    if acting == "adversary":
                        adv_deadlines = _adversary_new_prefill_deadlines_json(
                            env,
                            state_after=st_after,
                            before_request_ids=before_ids,
                        )
                        try:
                            adv_generated = len(json.loads(adv_deadlines))
                        except Exception:
                            adv_generated = 0
                    else:
                        adv_deadlines = "{}"
                        adv_generated = 0
                    srow = _state_row(env, st_after, adversary_prefill_deadlines_by_id=adv_deadlines)
                    _append_row(
                        rows,
                        trace_id=trace_id,
                        history_hop=int(task.history_hop),
                        step_index=step_index,
                        phase=str(ev.get("phase", "forced_transition")),
                        acting_player=acting,
                        state_info=srow,
                        best_action_index=int(ev.get("action_index", 0)),
                        best_action_repr=str(ev.get("action_repr", "")),
                        num_valid_actions=int(ev.get("num_valid_actions", 0)),
                        adversary_requests_generated=int(adv_generated),
                        next_player=str(ev.get("next_player", "")),
                    )
                    step_index += 1

                state, player, terminal = _advance_to_controller_branching(
                    env,
                    state,
                    player,
                    max_hops=cfg.collection.max_total_steps_per_state,
                    max_samples=cfg.collection.enum_max_samples,
                    step_callback=_pre_controller_cb,
                )
                if terminal:
                    break

            actions_by_index, valid = _actions_and_valid(
                env,
                state,
                "controller",
                cfg.collection.enum_max_samples,
            )
            if not valid:
                break

            feat_s = extract_features(env, state)
            with torch.no_grad():
                v_hat_s = float(model.predict(torch.from_numpy(feat_s).to(device=device, dtype=torch.float32)).item())

            sim_time_s = float(getattr(state.simulator, "_time", 0.0))
            cost_s = state_cost(env, state)

            candidate_logs: List[Dict[str, Any]] = []
            q_values: List[float] = []

            for aidx in valid:
                action = actions_by_index[aidx]
                assert action is not None

                child = env.apply_controller_action_only(state, action, inplace=False)
                child_player = "adversary"
                child, child_player, child_terminal = _advance_to_controller_branching(
                    env,
                    child,
                    child_player,
                    max_hops=cfg.collection.max_total_steps_per_state,
                    max_samples=cfg.collection.enum_max_samples,
                )

                cost_next = state_cost(env, child)
                sim_time_next = float(getattr(child.simulator, "_time", 0.0))
                dt = max(0.0, sim_time_next - sim_time_s)

                feat_next = extract_features(env, child)
                with torch.no_grad():
                    v_next = float(model.predict(torch.from_numpy(feat_next).to(device=device, dtype=torch.float32)).item())

                gamma_eff = effective_discount(cfg.bellman.discount_factor, dt, base_step_time)
                reward = transition_reward(cost_s, cost_next)
                q = reward + gamma_eff * v_next

                q_values.append(float(q))
                candidate_logs.append(
                    {
                        "action_index": int(aidx),
                        "action_repr": repr(action),
                        "q": float(q),
                        "reward": float(reward),
                        "gamma_eff": float(gamma_eff),
                        "delta_time": float(dt),
                        "cost_next": float(cost_next),
                        "v_next": float(v_next),
                        "terminal_after": bool(child_terminal),
                        "next_player": str(child_player),
                    }
                )

            if not q_values:
                break

            best_j = int(np.argmax(np.asarray(q_values, dtype=np.float32)))
            best_info = candidate_logs[best_j]
            best_action_idx = int(best_info["action_index"])
            best_action = actions_by_index[best_action_idx]
            assert best_action is not None

            decision_state_row = _state_row(env, state, adversary_prefill_deadlines_by_id="{}")
            _append_row(
                rows,
                trace_id=trace_id,
                history_hop=int(task.history_hop),
                step_index=step_index,
                phase="greedy_controller_decision",
                acting_player="controller",
                state_info=decision_state_row,
                best_action_index=int(best_action_idx),
                best_action_repr=repr(best_action),
                num_valid_actions=int(len(valid)),
                q_values_by_action_json=json.dumps(candidate_logs, ensure_ascii=False),
                v_hat_s=float(v_hat_s),
                cost_s=float(cost_s),
                cost_next=float(best_info["cost_next"]),
                reward=float(best_info["reward"]),
                delta_time=float(best_info["delta_time"]),
                gamma_eff=float(best_info["gamma_eff"]),
                adversary_requests_generated=0,
                next_player="adversary",
            )
            step_index += 1

            chosen_after_controller = env.apply_controller_action_only(state, best_action, inplace=False)

            def _chosen_path_cb(ev: Dict[str, Any]) -> None:
                nonlocal step_index
                st_after = ev["state_after"]
                acting = str(ev.get("acting_player", ""))
                before_ids = set(ev.get("before_request_ids", set()) or set())
                if acting == "adversary":
                    adv_deadlines = _adversary_new_prefill_deadlines_json(
                        env,
                        state_after=st_after,
                        before_request_ids=before_ids,
                    )
                    try:
                        adv_generated = len(json.loads(adv_deadlines))
                    except Exception:
                        adv_generated = 0
                else:
                    adv_deadlines = "{}"
                    adv_generated = 0
                srow = _state_row(env, st_after, adversary_prefill_deadlines_by_id=adv_deadlines)
                _append_row(
                    rows,
                    trace_id=trace_id,
                    history_hop=int(task.history_hop),
                    step_index=step_index,
                    phase=str(ev.get("phase", "forced_transition")),
                    acting_player=acting,
                    state_info=srow,
                    best_action_index=int(ev.get("action_index", 0)),
                    best_action_repr=str(ev.get("action_repr", "")),
                    num_valid_actions=int(ev.get("num_valid_actions", 0)),
                    adversary_requests_generated=int(adv_generated),
                    next_player=str(ev.get("next_player", "")),
                )
                step_index += 1

            state, player, terminal = _advance_to_controller_branching(
                env,
                chosen_after_controller,
                "adversary",
                max_hops=cfg.collection.max_total_steps_per_state,
                max_samples=cfg.collection.enum_max_samples,
                step_callback=_chosen_path_cb,
            )
            decision_steps += 1

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        summaries.append(
            {
                "trace_id": int(trace_id),
                "history_hop": int(task.history_hop),
                "steps": int(decision_steps),
                "terminal": bool(terminal),
                "trace_csv": str(out_csv),
            }
        )

    return summaries
