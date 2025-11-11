from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _as_float(v: str, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _as_int(v: str, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def _load_json(v: str, fallback):
    if v is None or v == "":
        return fallback
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return fallback


@dataclass
class Row:
    raw: Dict[str, str]
    idx: int
    phase: str
    node_id: str
    parent_id: Optional[str]
    sim_time: float
    violations: int
    avg_lateness: float
    objective_cost: float
    waiting_ids: List[int]
    completed_ids: List[int]
    requests_generated: int
    requests_completed: int
    player_to_act: str
    controller_token_budget: Optional[int] = None
    controller_prefill_total: Optional[int] = None
    controller_decode_total: Optional[int] = None
    controller_selected_ids: List[int] = None  # type: ignore[assignment]
    controller_allocations: Dict[str, int] = None  # type: ignore[assignment]
    controller_prefill_allocations: Dict[str, int] = None  # type: ignore[assignment]
    controller_decode_allocations: Dict[str, int] = None  # type: ignore[assignment]


def parse_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            phase = row.get("phase", "")
            node_id = (row.get("node_id") or "").strip() or "root"
            parent_id = (row.get("parent_node_id") or "").strip() or None
            waiting_ids = _load_json(row.get("state_waiting_ids"), [])
            completed_ids = _load_json(row.get("state_completed_request_ids"), [])
            rows.append(
                Row(
                    raw=row,
                    idx=i + 2,  # +2 for header + 1-based
                    phase=phase,
                    node_id=node_id,
                    parent_id=parent_id,
                    sim_time=_as_float(row.get("sim_time")),
                    violations=_as_int(row.get("slo_violations")),
                    avg_lateness=_as_float(row.get("avg_lateness")),
                    objective_cost=_as_float(row.get("objective_cost")),
                    waiting_ids=[int(x) for x in waiting_ids],
                    completed_ids=[int(x) for x in completed_ids],
                    requests_generated=_as_int(row.get("requests_generated")),
                    requests_completed=_as_int(row.get("requests_completed")),
                    player_to_act=row.get("player_to_act", ""),
                    controller_token_budget=(
                        _as_int(row.get("controller_token_budget"))
                        if row.get("controller_token_budget", "") != ""
                        else None
                    ),
                    controller_prefill_total=(
                        _as_int(row.get("controller_prefill_total"))
                        if row.get("controller_prefill_total", "") != ""
                        else None
                    ),
                    controller_decode_total=(
                        _as_int(row.get("controller_decode_total"))
                        if row.get("controller_decode_total", "") != ""
                        else None
                    ),
                    controller_selected_ids=[
                        int(x) for x in _load_json(row.get("controller_selected_ids"), [])
                    ],
                    controller_allocations=_load_json(row.get("controller_allocations"), {}),
                    controller_prefill_allocations=_load_json(row.get("controller_prefill_allocations"), {}),
                    controller_decode_allocations=_load_json(row.get("controller_decode_allocations"), {}),
                )
            )
    return rows


def build_index(rows: List[Row]) -> Dict[str, Row]:
    idx: Dict[str, Row] = {}
    for r in rows:
        idx[r.node_id] = r
    return idx


def verify_objective(rows: List[Row], errors: List[str], eps: float = 1e-9) -> None:
    for r in rows:
        expected = r.violations + r.avg_lateness
        if abs(r.objective_cost - expected) > eps:
            errors.append(
                f"[L{r.idx}] objective_cost mismatch: {r.objective_cost} != violations+avg_lateness ({expected})"
            )


def verify_time_progression(rows: List[Row], index: Dict[str, Row], errors: List[str]) -> None:
    for r in rows:
        if not r.parent_id:
            continue
        parent = index.get(r.parent_id)
        if not parent:
            # It may refer to an earlier rollout row; if missing, skip.
            continue
        if r.sim_time < parent.sim_time - 1e-12:
            errors.append(
                f"[L{r.idx}] time regression: {r.sim_time} < parent({r.parent_id}) time {parent.sim_time}"
            )


def verify_request_sets(rows: List[Row], index: Dict[str, Row], errors: List[str]) -> None:
    for r in rows:
        if not r.parent_id:
            continue
        parent = index.get(r.parent_id)
        if not parent:
            continue
        # completed IDs should be non-decreasing (superset) on tree and rollout rows
        parent_completed = set(parent.completed_ids)
        child_completed = set(r.completed_ids)
        if not parent_completed.issubset(child_completed):
            missing = parent_completed.difference(child_completed)
            errors.append(
                f"[L{r.idx}] completed ids lost relative to parent {r.parent_id}: {sorted(missing)}"
            )
        # Waiting requests of parent should remain either waiting or become completed at child.
        parent_waiting = set(parent.waiting_ids)
        child_waiting = set(r.waiting_ids)
        vanished = [rid for rid in parent_waiting if rid not in child_waiting and rid not in child_completed]
        # We allow disappearance if an adversary added/removes? Normally disappearance must be due to completion.
        if vanished:
            errors.append(
                f"[L{r.idx}] parent waiting ids missing at child without completion: {sorted(vanished)}"
            )
        # Cross-check counts
        if len(r.completed_ids) != r.requests_completed:
            errors.append(
                f"[L{r.idx}] requests_completed ({r.requests_completed}) != len(completed_ids) ({len(r.completed_ids)})"
            )


def verify_adversary_generation(rows: List[Row], index: Dict[str, Row], errors: List[str]) -> None:
    for r in rows:
        if r.phase != "tree":
            continue
        if not r.parent_id:
            continue
        parent = index.get(r.parent_id)
        if not parent:
            continue
        # Child row was logged with acting_player = parent's player (the one who acted to produce child)
        actor = r.player_to_act
        if actor == "adversary":
            specs = _load_json(r.raw.get("adversary_requests"), [])
            expected_gen = parent.requests_generated + len(specs)
            if r.requests_generated < expected_gen:
                errors.append(
                    f"[L{r.idx}] requests_generated did not increase by adversary specs count: got {r.requests_generated}, expected >= {expected_gen}"
                )


def verify_controller_allocations(rows: List[Row], errors: List[str]) -> None:
    for r in rows:
        if r.controller_token_budget is None:
            continue
        # Sum consistency
        pre = r.controller_prefill_total or 0
        dec = r.controller_decode_total or 0
        if pre + dec != r.controller_token_budget:
            errors.append(
                f"[L{r.idx}] token budget mismatch: pre({pre}) + dec({dec}) != budget({r.controller_token_budget})"
            )
        # Keys subset of selected ids
        sel = set(r.controller_selected_ids or [])
        alloc_keys = set(int(k) for k in (r.controller_allocations or {}).keys())
        if not alloc_keys.issubset(sel):
            bad = sorted(list(alloc_keys.difference(sel)))
            errors.append(
                f"[L{r.idx}] allocation keys not subset of selected ids: {bad} not in {sorted(sel)}"
            )


def verify_decode_after_prefill(rows: List[Row], errors: List[str]) -> None:
    """Ensure no request gets decode allocation before it has ever received prefill allocation.

    This is a conservative check based on what the CSV exposes. We can't know when prefill is *complete*, but we can ensure
    that a request never receives decode allocation without at least some prefill allocation since it first appeared.
    """
    seen_waiting: set[int] = set()
    has_prefill_alloc: Dict[int, bool] = {}
    for r in rows:
        # Update seen waiting ids (first appearance means generation or carry-over)
        for rid in r.waiting_ids:
            if rid not in seen_waiting:
                seen_waiting.add(rid)
                has_prefill_alloc.setdefault(rid, False)

        # Mark prefill allocations
        if r.controller_prefill_allocations:
            for k, v in r.controller_prefill_allocations.items():
                rid = int(k)
                if v and v > 0:
                    has_prefill_alloc[rid] = True

        # Check decode allocations
        if r.controller_decode_allocations:
            for k, v in r.controller_decode_allocations.items():
                rid = int(k)
                if v and v > 0 and not has_prefill_alloc.get(rid, False):
                    errors.append(
                        f"[L{r.idx}] decode allocation assigned to request {rid} before any prefill allocation was observed."
                    )


def verify_rollout_start_times(rows: List[Row], index: Dict[str, Row], errors: List[str]) -> None:
    for r in rows:
        if r.phase != "rollout":
            continue
        if not r.parent_id:
            continue
        parent = index.get(r.parent_id)
        if not parent:
            continue
        # First rollout row under a tree node should not go back in time.
        if r.parent_id.isdigit():
            if r.sim_time < parent.sim_time - 1e-12:
                errors.append(
                    f"[L{r.idx}] rollout sim_time {r.sim_time} < tree node {r.parent_id} sim_time {parent.sim_time}"
                )


def run_verification(path: Path) -> Tuple[bool, List[str]]:
    rows = parse_rows(path)
    index = build_index(rows)
    errors: List[str] = []
    # Basic numeric/time checks
    verify_objective(rows, errors)
    verify_time_progression(rows, index, errors)
    verify_rollout_start_times(rows, index, errors)
    # Structural request set checks
    verify_request_sets(rows, index, errors)
    verify_adversary_generation(rows, index, errors)
    # Controller action strict checks
    verify_controller_allocations(rows, errors)
    verify_decode_after_prefill(rows, errors)
    ok = len(errors) == 0
    return ok, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify structural and numerical consistency of Vidur MCTS trace CSV")
    ap.add_argument("--csv", type=Path, required=True, help="Path to test_mcts_trace.csv")
    args = ap.parse_args()

    ok, errors = run_verification(args.csv)
    if ok:
        print("Verification passed.")
    else:
        print(f"Verification failed: {len(errors)} issues found.")
        for e in errors:
            print("- ", e)


if __name__ == "__main__":
    main()
