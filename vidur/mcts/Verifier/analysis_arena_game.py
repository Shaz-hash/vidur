# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
Command to run :

cd /home/shazer/Desktop/Research/Vidur/vidur && \
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.Verifier.analysis_arena_game \
  --gen-start 7 \
  --gen-end 383

"""


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vidur.mcts.Verifier.verifier import (
    RootLogRow,
    _parse_action_json,
    _pick_fixed_controller_action,
    build_env,
    compute_metrics,
)
from vidur.mcts.environment import AdversaryAction, ControllerAction, VidurMCTSEnvironment, VidurMCTSState


MCTS_LOGS_ROOT = Path("simulator_output/mcts_dnn_logs")

# Per-scenario cycle labels.
# candidate file => evaluate candidate policy trace (best plays adversary)
CANDIDATE_TRACE_CYCLE_LABEL = "best_as_adversary"
# best file => evaluate best policy trace (candidate plays adversary)
BEST_TRACE_CYCLE_LABEL = "candidate_as_adversary"

SJF_1024_BUDGET = 1024
SJF_512_BUDGET = 512
CONTROLLER_HEURISTIC = "SJF"
MAX_CONTROLLER_STEPS = 2000
TIE_EPS = 1e-9


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_root_row(row: Dict[str, str]) -> RootLogRow:
    return RootLogRow(
        game_id=_safe_int(row.get("game_id", 0)),
        root_id=_safe_int(row.get("root_id", 0)),
        root_depth=_safe_int(row.get("root_depth", 0)),
        root_player=str(row.get("root_player", "") or "").strip(),
        best_action_index=_safe_int(row.get("best_action_index", 0)),
        best_action_repr=str(row.get("best_action_repr", "") or ""),
        best_action_json=str(row.get("best_action_json", "") or ""),
    )


def _is_adv_important(row: Dict[str, str]) -> bool:
    action_json = str(row.get("best_action_json", "") or "").strip()
    if action_json:
        try:
            d = json.loads(action_json)
            if str(d.get("type", "")).strip().lower() != "adversary":
                return False
            reqs = d.get("requests", [])
            return isinstance(reqs, list) and len(reqs) > 0
        except Exception:
            pass

    action_repr = str(row.get("best_action_repr", "") or "")
    return ("AdversaryAction(" in action_repr) and ("requests=[]" not in action_repr)


def _is_controller_trivial_for_debug(row: Dict[str, str]) -> bool:
    action_json = str(row.get("best_action_json", "") or "").strip()
    if action_json:
        try:
            d = json.loads(action_json)
            if str(d.get("type", "")).strip().lower() != "controller":
                return False
            pa = d.get("prefill_allocations", {}) or {}
            if isinstance(pa, dict):
                return sum(int(v) for v in pa.values()) <= 0
        except Exception:
            pass

    action_repr = str(row.get("best_action_repr", "") or "")
    return ("ControllerAction(" in action_repr) and ("prefill_allocations={}" in action_repr)


def _select_timeline_rows(rows: Sequence[Dict[str, str]], *, cycle_label: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        phase = str(row.get("phase", "") or "").strip()
        cyc = str(row.get("cycle_label", "") or "").strip()
        if phase == "train_root":
            out.append(row)
            continue
        if phase == "arena_root" and cyc == cycle_label:
            out.append(row)
            continue
    return out


def _find_last_important_adv_idx(timeline_rows: Sequence[Dict[str, str]]) -> int:
    last = -1
    for i, row in enumerate(timeline_rows):
        root_player = str(row.get("root_player", "") or "").strip().lower()
        if root_player == "adversary" and _is_adv_important(row):
            last = i
    return last


def _replay_to_target_root(
    env: VidurMCTSEnvironment,
    *,
    base_sim_snapshot: Any,
    base_stats_snapshot: Any,
    root_rows: Sequence[RootLogRow],
) -> Tuple[VidurMCTSState, str]:
    if not root_rows:
        raise RuntimeError("replay requested with empty root_rows")

    state = env.clone_state_from_snapshot(base_sim_snapshot, base_stats_snapshot)
    history = list(root_rows[:-1])
    target = root_rows[-1]

    player = (history[0].root_player if history else target.root_player) or "adversary"

    for i, row in enumerate(history):
        action = _parse_action_json(row.best_action_json)
        if isinstance(action, ControllerAction):
            state = env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"
        elif isinstance(action, AdversaryAction):
            state = env.apply_adversary_action_only(state, action, inplace=True)
            player = "controller"
        else:
            raise RuntimeError(f"unsupported action at i={i}: {type(action).__name__}")

    return state, player


def _lookup_ids(env: VidurMCTSEnvironment, state: VidurMCTSState) -> set[int]:
    d = env._build_request_lookup(state.simulator)  # type: ignore[attr-defined]
    return {int(x) for x in d.keys()}


def _is_prefill_complete(req: Any) -> bool:
    return bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)))


def _remaining_prefill(req: Any) -> int:
    return max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))


def _target_prefill_done(
    env: VidurMCTSEnvironment,
    state: VidurMCTSState,
    target_ids: Iterable[int],
) -> bool:
    lookup = env._build_request_lookup(state.simulator)  # type: ignore[attr-defined]
    for rid in target_ids:
        req = lookup.get(int(rid))
        if req is None:
            # Request no longer present: treat as done for prefill completion tracking.
            continue
        if (not _is_prefill_complete(req)) and (_remaining_prefill(req) > 0):
            return False
    return True


def _run_fixed_cleanup_cost(
    env: VidurMCTSEnvironment,
    *,
    scenario_sim_snapshot: Any,
    scenario_stats_snapshot: Any,
    target_ids: Sequence[int],
    controller_budget: int,
    controller_heuristic: str,
    max_steps: int,
) -> Tuple[float, int, bool]:
    state = env.clone_state_from_snapshot(scenario_sim_snapshot, scenario_stats_snapshot)
    steps = 0

    while steps < int(max_steps):
        if _target_prefill_done(env, state, target_ids):
            break
        idx, action = _pick_fixed_controller_action(
            env,
            state,
            prefill_budget=int(controller_budget),
            heuristic=str(controller_heuristic),
        )
        if idx < 0 or action is None:
            break
        state = env.apply_controller_action_only(state, action, inplace=True)
        steps += 1

    done = _target_prefill_done(env, state, target_ids)
    m = compute_metrics(env, state)
    return float(m.objective_cost), int(steps), bool(done)


def _winner_model_vs_sjf(
    *,
    model_name: str,
    model_cost: float,
    sjf_1024_cost: float,
    sjf_512_cost: float,
    eps: float = TIE_EPS,
) -> str:
    best_sjf = min(float(sjf_1024_cost), float(sjf_512_cost))
    if float(model_cost) <= (best_sjf + float(eps)):
        return str(model_name)
    if abs(float(sjf_1024_cost) - float(sjf_512_cost)) <= float(eps):
        return "tie"
    return "sjf-1024" if float(sjf_1024_cost) < float(sjf_512_cost) else "sjf-512"


def _evaluate_cycle_costs(
    env: VidurMCTSEnvironment,
    *,
    base_sim_snapshot: Any,
    base_stats_snapshot: Any,
    game_id: int,
    game_rows: Sequence[Dict[str, str]],
    cycle_label: str,
) -> Tuple[float, float, int, int, int, bool, int, bool]:
    timeline = _select_timeline_rows(game_rows, cycle_label=cycle_label)
    if not timeline:
        raise RuntimeError(f"no timeline rows for cycle={cycle_label}")

    last_idx = _find_last_important_adv_idx(timeline)
    if last_idx < 0:
        raise RuntimeError("no important adversary action")

    replay_rows_dict = timeline[: last_idx + 1]
    replay_rows = [_to_root_row(r) for r in replay_rows_dict]
    target_row = replay_rows[-1]

    state, _ = _replay_to_target_root(
        env,
        base_sim_snapshot=base_sim_snapshot,
        base_stats_snapshot=base_stats_snapshot,
        root_rows=replay_rows,
    )

    adv_action = _parse_action_json(target_row.best_action_json)
    if not isinstance(adv_action, AdversaryAction):
        raise RuntimeError("target row best_action_json is not adversary action")

    before_ids = _lookup_ids(env, state)
    state = env.apply_adversary_action_only(state, adv_action, inplace=True)
    after_ids = _lookup_ids(env, state)
    injected_ids = sorted(int(x) for x in (after_ids - before_ids))
    if not injected_ids:
        raise RuntimeError("target adversary action injected no new IDs")

    scenario_sim_snapshot = state.simulator.snapshot_state()
    scenario_stats_snapshot = state.stats.clone()

    cost_1024, steps_1024, done_1024 = _run_fixed_cleanup_cost(
        env,
        scenario_sim_snapshot=scenario_sim_snapshot,
        scenario_stats_snapshot=scenario_stats_snapshot,
        target_ids=injected_ids,
        controller_budget=SJF_1024_BUDGET,
        controller_heuristic=CONTROLLER_HEURISTIC,
        max_steps=MAX_CONTROLLER_STEPS,
    )
    cost_512, steps_512, done_512 = _run_fixed_cleanup_cost(
        env,
        scenario_sim_snapshot=scenario_sim_snapshot,
        scenario_stats_snapshot=scenario_stats_snapshot,
        target_ids=injected_ids,
        controller_budget=SJF_512_BUDGET,
        controller_heuristic=CONTROLLER_HEURISTIC,
        max_steps=MAX_CONTROLLER_STEPS,
    )

    print(
        f"[analysis_arena_game] game_id={game_id} cycle={cycle_label} "
        f"rows={len(replay_rows_dict)} target_injected={len(injected_ids)} "
        f"SJF1024={cost_1024:.6f} (steps={steps_1024},done={done_1024}) "
        f"SJF512={cost_512:.6f} (steps={steps_512},done={done_512})"
    )
    return (
        float(cost_1024),
        float(cost_512),
        int(len(replay_rows_dict)),
        int(len(injected_ids)),
        int(steps_1024),
        bool(done_1024),
        int(steps_512),
        bool(done_512),
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _analyze_generation(
    *,
    env: VidurMCTSEnvironment,
    base_sim_snapshot: Any,
    base_stats_snapshot: Any,
    gen_dir: Path,
) -> None:
    arena_results_csv = gen_dir / "arena_results.csv"
    arena_games_dir = gen_dir / "arena_games"
    candidate_out_csv = gen_dir / "candidate_arena_performance.csv"
    best_out_csv = gen_dir / "best_arena_performance.csv"

    if not arena_results_csv.exists():
        print(f"[analysis_arena_game] skip gen={gen_dir.name}: missing {arena_results_csv}")
        return
    if not arena_games_dir.exists():
        print(f"[analysis_arena_game] skip gen={gen_dir.name}: missing {arena_games_dir}")
        return

    arena_rows = _read_csv_rows(arena_results_csv)
    if not arena_rows:
        print(f"[analysis_arena_game] skip gen={gen_dir.name}: no rows in {arena_results_csv}")
        return

    candidate_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    skipped_candidate = 0
    skipped_best = 0

    for arow in arena_rows:
        game_id = _safe_int(arow.get("game_id", 0), default=-1)
        if game_id < 0:
            skipped_candidate += 1
            skipped_best += 1
            continue

        game_csv = arena_games_dir / f"game_{game_id}.csv"
        if not game_csv.exists():
            skipped_candidate += 1
            skipped_best += 1
            print(f"[analysis_arena_game] skip gen={gen_dir.name} game_id={game_id}: missing {game_csv}")
            continue

        game_rows = _read_csv_rows(game_csv)
        cand_ok = False
        best_ok = False

        try:
            cost_1024, cost_512, _, _, _, _, _, _ = _evaluate_cycle_costs(
                env,
                base_sim_snapshot=base_sim_snapshot,
                base_stats_snapshot=base_stats_snapshot,
                game_id=int(game_id),
                game_rows=game_rows,
                cycle_label=CANDIDATE_TRACE_CYCLE_LABEL,
            )
            best_as_adv_cost = _safe_float(arow.get("best_as_adv_cost", 0.0))
            winner = _winner_model_vs_sjf(
                model_name="candidate",
                model_cost=float(best_as_adv_cost),
                sjf_1024_cost=float(cost_1024),
                sjf_512_cost=float(cost_512),
            )
            candidate_rows.append(
                {
                    "game_id": int(game_id),
                    "best_as_adv_cost": float(best_as_adv_cost),
                    "cost_SJF_1024": float(cost_1024),
                    "cost_SJF": float(cost_512),
                    "winner": str(winner),
                }
            )
            cand_ok = True
        except Exception as e:
            skipped_candidate += 1
            print(f"[analysis_arena_game] skip candidate file gen={gen_dir.name} game_id={game_id}: {e}")

        try:
            cost_1024, cost_512, _, _, _, _, _, _ = _evaluate_cycle_costs(
                env,
                base_sim_snapshot=base_sim_snapshot,
                base_stats_snapshot=base_stats_snapshot,
                game_id=int(game_id),
                game_rows=game_rows,
                cycle_label=BEST_TRACE_CYCLE_LABEL,
            )
            candidate_as_adv_cost = _safe_float(arow.get("candidate_as_adv_cost", 0.0))
            winner = _winner_model_vs_sjf(
                model_name="best",
                model_cost=float(candidate_as_adv_cost),
                sjf_1024_cost=float(cost_1024),
                sjf_512_cost=float(cost_512),
            )
            best_rows.append(
                {
                    "game_id": int(game_id),
                    "candidate_as_adv_cost": float(candidate_as_adv_cost),
                    "cost_SJF_1024": float(cost_1024),
                    "cost_SJF": float(cost_512),
                    "winner": str(winner),
                }
            )
            best_ok = True
        except Exception as e:
            skipped_best += 1
            print(f"[analysis_arena_game] skip best file gen={gen_dir.name} game_id={game_id}: {e}")

        # Explicitly reset to fresh empty-state baseline before moving to next game.
        _ = env.clone_state_from_snapshot(base_sim_snapshot, base_stats_snapshot)
        print(
            f"[analysis_arena_game] done gen={gen_dir.name} game_id={game_id} "
            f"candidate={'ok' if cand_ok else 'skip'} best={'ok' if best_ok else 'skip'}"
        )

    _write_csv(
        candidate_out_csv,
        fieldnames=["game_id", "best_as_adv_cost", "cost_SJF_1024", "cost_SJF", "winner"],
        rows=candidate_rows,
    )
    _write_csv(
        best_out_csv,
        fieldnames=["game_id", "candidate_as_adv_cost", "cost_SJF_1024", "cost_SJF", "winner"],
        rows=best_rows,
    )

    print(
        f"[analysis_arena_game] wrote {candidate_out_csv} "
        f"(rows={len(candidate_rows)}, skipped={skipped_candidate}, total={len(arena_rows)})"
    )
    print(
        f"[analysis_arena_game] wrote {best_out_csv} "
        f"(rows={len(best_rows)}, skipped={skipped_best}, total={len(arena_rows)})"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arena game baseline comparison for MCTS-DNN logs")
    p.add_argument(
        "--logs-root",
        type=Path,
        default=MCTS_LOGS_ROOT,
        help="Root directory containing gen_xxxxxx directories",
    )
    p.add_argument("--gen-start", type=int, default=7, help="Start generation index (inclusive)")
    p.add_argument("--gen-end", type=int, default=7, help="End generation index (inclusive)")
    p.add_argument(
        "--gen-list",
        type=str,
        default="",
        help="Optional comma-separated generation indices; overrides start/end when provided",
    )
    return p.parse_args()


def _iter_generation_indices(args: argparse.Namespace) -> List[int]:
    gen_list = str(args.gen_list or "").strip()
    if gen_list:
        out: List[int] = []
        for t in gen_list.split(","):
            t = t.strip()
            if t:
                out.append(int(t))
        return sorted(set(out))
    s = int(args.gen_start)
    e = int(args.gen_end)
    if s > e:
        raise ValueError(f"invalid generation range: gen-start ({s}) > gen-end ({e})")
    return list(range(s, e + 1))


def main() -> None:
    args = _parse_args()
    gen_indices = _iter_generation_indices(args)

    env = build_env()
    base_state = env.initial_state()
    base_sim_snapshot = base_state.simulator.snapshot_state()
    base_stats_snapshot = base_state.stats.clone()

    for g in gen_indices:
        gen_dir = Path(args.logs_root) / f"gen_{int(g):06d}"
        _analyze_generation(
            env=env,
            base_sim_snapshot=base_sim_snapshot,
            base_stats_snapshot=base_stats_snapshot,
            gen_dir=gen_dir,
        )


if __name__ == "__main__":
    main()
