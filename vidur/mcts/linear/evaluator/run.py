from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...DNN.history_root import HistoryRootGenerator
from ...environment import AdversaryAction, ControllerAction, VidurMCTSState
from ..bellman import state_cost
from ..collector_worker import (
    _advance_forced_until_branching,
    _apply_action_inplace,
    _build_env,
    _is_explicit_history_row,
    _load_history_rows,
    _parse_action_json,
)
from ..config import (
    BellmanSettings,
    CollectionSettings,
    ConstraintSettings,
    LinearPipelineConfig,
)
from ..lp.features import feature_names_for_set
from ..features import (
    _build_request_lookup,
    _prefill_complete,
    _remaining_prefill,
    _req_completed,
)
from .feature_adapter import extract_lp_baseline_v1_vector
from .weights import load_weights


@dataclass(frozen=True)
class CanonicalAction:
    canonical_index: int
    alias_indices: List[int]
    action: object
    canonical_key: str


@dataclass(frozen=True)
class ScoredAction:
    canonical_index: int
    action: object
    action_json: str
    action_repr: str
    child_value: float
    edge_value: float


class _HistoryCaptureLogger:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def log_expand(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))

    def log_root(self, **kwargs: Any) -> None:
        if not self.rows:
            self.rows.append({})
        self.rows[-1].update(dict(kwargs))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate one game using LP linear value policy (no MCTS)")
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--history-hop", type=int, default=0)
    ap.add_argument("--history-csv", type=str, default="")
    ap.add_argument("--discount-factor", type=float, default=0.98)
    ap.add_argument("--max-steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--game-id", type=int, default=-1)
    ap.add_argument("--root-player", type=str, default="adversary", choices=["controller", "adversary"])

    ap.add_argument("--lp-solution-dir", type=str, default="")
    ap.add_argument("--weights-json", type=str, default="")
    ap.add_argument("--feature-names-json", type=str, default="")

    ap.add_argument("--prefill-profile", type=str, default="simulator_output/prefill_profile.csv")
    ap.add_argument("--use-virtual-env", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--align-branching-roots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-branching", type=int, default=10)
    ap.add_argument("--enum-max-samples", type=int, default=10000)
    ap.add_argument("--max-forced-hops", type=int, default=20000)
    ap.add_argument("--max-total-steps-per-state", type=int, default=512)
    return ap.parse_args(argv)


def _mask_to_list(mask: Any) -> List[bool]:
    if hasattr(mask, "to"):
        try:
            return [bool(x) for x in mask.to(dtype=bool).cpu().tolist()]
        except Exception:
            pass
    return [bool(x) for x in mask]


def _sample_actions_with_mask(
    env: Any,
    state: VidurMCTSState,
    player: str,
    *,
    max_samples: int,
) -> Tuple[List[Optional[object]], List[int], List[bool]]:
    if player == "controller":
        actions_by_index, mask = env.sample_controller_actions(state, max_samples)
    else:
        actions_by_index, mask = env.sample_adversary_actions(state, max_samples)
    mask_list = _mask_to_list(mask)
    valid = [
        i
        for i, ok in enumerate(mask_list)
        if ok and i < len(actions_by_index) and actions_by_index[i] is not None
    ]
    return actions_by_index, valid, mask_list


def _controller_action_key(action: ControllerAction) -> Tuple[Tuple[int, int], ...]:
    alloc = action.token_allocations or {}
    return tuple(sorted((int(rid), int(tok)) for rid, tok in alloc.items()))


def _canonical_actions(
    env: Any,
    state: VidurMCTSState,
    player: str,
    *,
    max_samples: int,
) -> Tuple[List[CanonicalAction], List[bool]]:
    actions_by_index, valid, mask = _sample_actions_with_mask(
        env, state, player, max_samples=max_samples
    )
    if not valid:
        return [], mask

    if player != "controller" or len(valid) <= 1:
        out = []
        for idx in sorted(valid):
            action = actions_by_index[idx]
            assert action is not None
            out.append(
                CanonicalAction(
                    canonical_index=int(idx),
                    alias_indices=[int(idx)],
                    action=action,
                    canonical_key=f"idx:{int(idx)}",
                )
            )
        return out, mask

    groups: Dict[Tuple[Tuple[int, int], ...], List[int]] = {}
    action_for_idx: Dict[int, object] = {}
    for idx in sorted(valid):
        action = actions_by_index[idx]
        assert action is not None
        key = _controller_action_key(action)  # type: ignore[arg-type]
        groups.setdefault(key, []).append(int(idx))
        action_for_idx[int(idx)] = action

    out: List[CanonicalAction] = []
    for key in sorted(groups.keys()):
        aliases = sorted(groups[key])
        canon = int(aliases[0])
        out.append(
            CanonicalAction(
                canonical_index=canon,
                alias_indices=[int(x) for x in aliases],
                action=action_for_idx[canon],
                canonical_key=json.dumps(key, ensure_ascii=False, separators=(",", ":")),
            )
        )
    return out, mask


def _action_to_json(action: object) -> str:
    if isinstance(action, ControllerAction):
        payload = {
            "type": "controller",
            "token_budget": int(action.token_budget),
            "selected_request_ids": [int(x) for x in (action.selected_request_ids or [])],
            "token_allocations": {str(int(k)): int(v) for k, v in (action.token_allocations or {}).items()},
            "prefill_allocations": {str(int(k)): int(v) for k, v in (action.prefill_allocations or {}).items()},
            "decode_allocations": {str(int(k)): int(v) for k, v in (action.decode_allocations or {}).items()},
            "heuristic": action.heuristic,
            "strategy": action.strategy,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    if isinstance(action, AdversaryAction):
        payload = {
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
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    return json.dumps({"type": "unknown", "repr": repr(action)}, ensure_ascii=False, sort_keys=True)


def _state_metrics(env: Any, state: VidurMCTSState) -> Dict[str, float]:
    desc = {}
    try:
        desc = dict(env.describe_state(state))
    except Exception:
        desc = {}
    sim_time = float(desc.get("sim_time", getattr(state.simulator, "_time", 0.0)))
    slo_violations = int(desc.get("slo_violations", 0))
    total_lateness = float(desc.get("total_lateness", 0.0))
    total_cost = float(state_cost(env, state))
    return {
        "sim_time": sim_time,
        "slo_violations": slo_violations,
        "total_lateness": total_lateness,
        "total_cost": total_cost,
    }


def _has_active_prefill(env: Any, state: VidurMCTSState) -> bool:
    lookup = _build_request_lookup(env, state)
    for req in lookup.values():
        if _req_completed(req):
            continue
        if (not _prefill_complete(req)) and _remaining_prefill(req) > 0:
            return True
    return False


def _append_row(
    rows: List[Dict[str, Any]],
    *,
    row_id: int,
    game_id: int,
    root_player: str,
    player_acted_to_create_this_state: str,
    player_to_act_next: str,
    model_root_value_json: str,
    action_value_json: str,
    valid_action_mask_json: str,
    best_action_index: Any,
    best_action_value: Any,
    best_action_repr: str,
    best_action_json: str,
    metrics: Dict[str, float],
) -> None:
    rows.append(
        {
            "game_id": int(game_id),
            "root_id": int(row_id),
            "root_depth": int(row_id),
            "root_node_id": int(row_id),
            "root_player": str(root_player),
            "player_acted_to_create_this_state": str(player_acted_to_create_this_state),
            "player_to_act_next": str(player_to_act_next),
            "model_root_value_json": str(model_root_value_json),
            "action_value_json": str(action_value_json),
            "valid_action_mask_json": str(valid_action_mask_json),
            "best_action_index": best_action_index,
            "best_action_value": best_action_value,
            "best_action_repr": str(best_action_repr),
            "best_action_json": str(best_action_json),
            "sim_time": float(metrics["sim_time"]),
            "slo_violations": int(metrics["slo_violations"]),
            "total_lateness": float(metrics["total_lateness"]),
            "total_cost": float(metrics["total_cost"]),
        }
    )


def _find_action_index_by_repr(actions_by_index: List[Optional[object]], valid: List[int], action: object) -> int:
    target = repr(action)
    for idx in valid:
        cand = actions_by_index[idx]
        if cand is not None and repr(cand) == target:
            return int(idx)
    return -1


def _replay_history_csv_with_rows(
    *,
    env: Any,
    state: VidurMCTSState,
    player: str,
    history_csv: str,
    align_branching_roots: bool,
    max_forced_hops: int,
    enum_max_samples: int,
    rows: List[Dict[str, Any]],
    game_id: int,
    row_counter: int,
) -> Tuple[VidurMCTSState, str, int, str]:
    raw_rows = _load_history_rows(Path(history_csv))
    action_rows = [r for r in raw_rows if (r.best_action_json or "").strip()]
    last_actor = ""
    if not action_rows:
        return state, player, row_counter, last_actor

    for i, row in enumerate(action_rows):
        is_hist = _is_explicit_history_row(row)
        if align_branching_roots and (not is_hist):
            state, player = _advance_forced_until_branching(
                env,
                state,
                player,
                max_hops=max_forced_hops,
                max_samples=enum_max_samples,
            )

        if row.root_player and row.root_player != player:
            raise RuntimeError(
                f"History replay mismatch row={i}: row.root_player={row.root_player}, player={player}"
            )

        actions_by_index, valid, mask_list = _sample_actions_with_mask(
            env,
            state,
            player,
            max_samples=enum_max_samples,
        )
        action = _parse_action_json(row.best_action_json)
        action_idx = _find_action_index_by_repr(actions_by_index, valid, action)

        acted = str(player)
        state, next_player = _apply_action_inplace(env, state, player, action)
        player = next_player
        last_actor = acted

        _append_row(
            rows,
            row_id=row_counter,
            game_id=game_id,
            root_player=player,
            player_acted_to_create_this_state=acted,
            player_to_act_next=player,
            model_root_value_json="{}",
            action_value_json="{}",
            valid_action_mask_json=json.dumps(mask_list, ensure_ascii=False),
            best_action_index="" if action_idx < 0 else int(action_idx),
            best_action_value="",
            best_action_repr=repr(action),
            best_action_json=row.best_action_json,
            metrics=_state_metrics(env, state),
        )
        row_counter += 1

        if i < len(action_rows) - 1 and align_branching_roots and (not is_hist):
            state, player = _advance_forced_until_branching(
                env,
                state,
                player,
                max_hops=max_forced_hops,
                max_samples=enum_max_samples,
            )

    return state, player, row_counter, last_actor


def _apply_history_hops_with_rows(
    *,
    env: Any,
    state: VidurMCTSState,
    player: str,
    history_hop: int,
    max_branching: int,
    max_total_steps: int,
    seed: int,
    game_id: int,
    row_counter: int,
    rows: List[Dict[str, Any]],
) -> Tuple[VidurMCTSState, str, int, str]:
    if int(history_hop) <= 0:
        return state, player, row_counter, ""

    logger = _HistoryCaptureLogger()
    hgen = HistoryRootGenerator(
        env=env,
        max_branching=int(max_branching),
        iter_logger=logger,
        root_logger=logger,
    )
    state, player, _depth, _next_node_id, _last_node_id = hgen.generate_history_root(
        state=state,
        player=player,
        depth=0,
        nontrivial_hops=int(history_hop),
        game_id=int(game_id),
        root_id_for_logs=0,
        seed=int(seed),
        log_history=True,
        max_total_steps=int(max_total_steps),
        log_node_id_start=0,
        log_parent_id_start=None,
    )

    last_actor = ""
    for hr in logger.rows:
        acted = str(hr.get("player_acted_to_create_this_node", "") or "")
        next_player = str(hr.get("player_to_act", "") or "")
        action_idx = hr.get("action_index", "")
        action_repr = str(hr.get("action_repr", "") or "")
        action_json = str(hr.get("best_action_json", "") or "")
        snap = hr.get("state_snapshot", {}) or {}
        total_cost = float(hr.get("objective_cost", 0.0))
        metrics = {
            "sim_time": float(snap.get("sim_time", 0.0)),
            "slo_violations": int(snap.get("slo_violations", 0)),
            "total_lateness": float(snap.get("total_lateness", 0.0)),
            "total_cost": total_cost,
        }

        _append_row(
            rows,
            row_id=row_counter,
            game_id=game_id,
            root_player=next_player,
            player_acted_to_create_this_state=acted,
            player_to_act_next=next_player,
            model_root_value_json="{}",
            action_value_json="{}",
            valid_action_mask_json="[]",
            best_action_index=action_idx,
            best_action_value="",
            best_action_repr=action_repr,
            best_action_json=action_json,
            metrics=metrics,
        )
        row_counter += 1
        last_actor = acted

    return state, player, row_counter, last_actor


def _score_canonical_actions(
    *,
    env: Any,
    state: VidurMCTSState,
    player: str,
    feature_names: List[str],
    weights: np.ndarray,
    discount: float,
    enum_max_samples: int,
) -> Tuple[List[ScoredAction], List[bool]]:
    canonical, mask_list = _canonical_actions(
        env, state, player, max_samples=enum_max_samples
    )
    if not canonical:
        return [], mask_list

    cost_s = float(state_cost(env, state))
    out: List[ScoredAction] = []
    for ca in canonical:
        if player == "controller":
            child = env.apply_controller_action_only(state, ca.action, inplace=False)
        else:
            child = env.apply_adversary_action_only(state, ca.action, inplace=False)

        cost_next = float(state_cost(env, child))
        delta = float(cost_next - cost_s)
        child_vec, _fmap = extract_lp_baseline_v1_vector(
            env,
            child,
            feature_names=feature_names,
        )
        child_value = float(np.dot(weights, child_vec))
        edge_value = float(delta + float(discount) * child_value)
        out.append(
            ScoredAction(
                canonical_index=int(ca.canonical_index),
                action=ca.action,
                action_json=_action_to_json(ca.action),
                action_repr=repr(ca.action),
                child_value=child_value,
                edge_value=edge_value,
            )
        )
    return out, mask_list


def _choose_best(scored: List[ScoredAction], player: str) -> ScoredAction:
    if player == "adversary":
        return max(scored, key=lambda s: (float(s.edge_value), -int(s.canonical_index)))
    return min(scored, key=lambda s: (float(s.edge_value), int(s.canonical_index)))


def _build_linear_cfg(args: argparse.Namespace) -> LinearPipelineConfig:
    return LinearPipelineConfig(
        constraints=ConstraintSettings(prefill_profile_path=str(args.prefill_profile)),
        collection=CollectionSettings(
            workers=1,
            train_samples_per_worker=0,
            eval_samples_per_worker=0,
            max_branching=int(args.max_branching),
            enum_max_samples=int(args.enum_max_samples),
            max_forced_hops=int(args.max_forced_hops),
            max_total_steps_per_state=int(args.max_total_steps_per_state),
            history_hops=(int(args.history_hop),),
            history_csv=str(args.history_csv),
            root_player=str(args.root_player),
            align_branching_roots=bool(args.align_branching_roots),
            use_virtual_env=bool(args.use_virtual_env),
        ),
        bellman=BellmanSettings(discount_factor=float(args.discount_factor)),
        seed=int(args.seed),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    game_id = int(args.game_id) if int(args.game_id) >= 0 else int(args.seed) * 1000 + int(args.history_hop)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    loaded = load_weights(
        lp_solution_dir=str(args.lp_solution_dir).strip() or None,
        weights_json=str(args.weights_json).strip() or None,
        feature_names_json=str(args.feature_names_json).strip() or None,
    )
    expected_names = feature_names_for_set("baseline_v1")
    if loaded.feature_names != expected_names:
        raise ValueError(
            "loaded weight feature names do not match current baseline_v1 schema; "
            "re-solve LP with current features and use updated artifacts"
        )

    cfg = _build_linear_cfg(args)
    _sim, env = _build_env(cfg, use_virtual_env=bool(cfg.collection.use_virtual_env))

    state = env.initial_state()
    player = str(cfg.collection.root_player)
    rows: List[Dict[str, Any]] = []
    row_counter = 0
    last_actor = ""
    adversary_actions_made = 0

    history_csv = str(cfg.collection.history_csv or "").strip()
    if history_csv:
        state, player, row_counter, last_actor = _replay_history_csv_with_rows(
            env=env,
            state=state,
            player=player,
            history_csv=history_csv,
            align_branching_roots=bool(cfg.collection.align_branching_roots),
            max_forced_hops=int(cfg.collection.max_forced_hops),
            enum_max_samples=int(cfg.collection.enum_max_samples),
            rows=rows,
            game_id=game_id,
            row_counter=row_counter,
        )
        adversary_actions_made += sum(
            1 for r in rows if str(r.get("player_acted_to_create_this_state", "")) == "adversary"
        )

    state, player, row_counter, hgen_last_actor = _apply_history_hops_with_rows(
        env=env,
        state=state,
        player=player,
        history_hop=int(args.history_hop),
        max_branching=int(cfg.collection.max_branching),
        max_total_steps=int(cfg.collection.max_forced_hops),
        seed=int(args.seed),
        game_id=game_id,
        row_counter=row_counter,
        rows=rows,
    )
    if hgen_last_actor:
        last_actor = hgen_last_actor
    adversary_actions_made += sum(
        1
        for r in rows
        if str(r.get("player_acted_to_create_this_state", "")) == "adversary"
    ) - adversary_actions_made

    stop_reason = "max_steps"
    for _ in range(int(args.max_steps)):
        scored, mask_list = _score_canonical_actions(
            env=env,
            state=state,
            player=player,
            feature_names=loaded.feature_names,
            weights=loaded.weights,
            discount=float(args.discount_factor),
            enum_max_samples=int(cfg.collection.enum_max_samples),
        )
        if not scored:
            stop_reason = "no_valid_actions"
            break

        action_value_json = json.dumps(
            {str(int(s.canonical_index)): float(s.child_value) for s in scored},
            ensure_ascii=False,
            sort_keys=True,
        )
        model_root_value_json = json.dumps(
            {str(int(s.canonical_index)): float(s.edge_value) for s in scored},
            ensure_ascii=False,
            sort_keys=True,
        )
        best = _choose_best(scored, player)

        if player == "adversary":
            adversary_actions_made += 1

        next_state, next_player = _apply_action_inplace(env, state, player, best.action)
        _append_row(
            rows,
            row_id=row_counter,
            game_id=game_id,
            root_player=player,
            player_acted_to_create_this_state=last_actor,
            player_to_act_next=next_player,
            model_root_value_json=model_root_value_json,
            action_value_json=action_value_json,
            valid_action_mask_json=json.dumps(mask_list, ensure_ascii=False),
            best_action_index=int(best.canonical_index),
            best_action_value=float(best.edge_value),
            best_action_repr=best.action_repr,
            best_action_json=best.action_json,
            metrics=_state_metrics(env, state),
        )
        row_counter += 1

        state = next_state
        last_actor = player
        player = next_player

        if (adversary_actions_made >= 1) and (not _has_active_prefill(env, state)):
            stop_reason = "all_prefill_complete_after_adversary"
            break

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_id",
        "root_id",
        "root_depth",
        "root_node_id",
        "root_player",
        "player_acted_to_create_this_state",
        "player_to_act_next",
        "model_root_value_json",
        "action_value_json",
        "valid_action_mask_json",
        "best_action_index",
        "best_action_value",
        "best_action_repr",
        "best_action_json",
        "sim_time",
        "slo_violations",
        "total_lateness",
        "total_cost",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(
        f"[linear.evaluator] wrote rows={len(rows)} stop_reason={stop_reason} out_csv={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
