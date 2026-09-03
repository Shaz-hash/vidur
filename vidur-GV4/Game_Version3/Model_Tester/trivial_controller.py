from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..game_types import ControllerAction


def make_strict_noop_controller_action(*, eviction_rule: str) -> ControllerAction:
    return ControllerAction(
        token_budget=0,
        selected_request_ids=None,
        token_allocations={},
        prefill_allocations={},
        decode_allocations={},
        heuristic=None,
        strategy=f"GV2|{eviction_rule}",
        mapping=None,
    )


def _mask_to_list(mask: Any) -> List[bool]:
    if mask is None:
        return []
    if hasattr(mask, "tolist"):
        try:
            return [bool(x) for x in mask.tolist()]
        except Exception:
            pass
    return [bool(x) for x in list(mask)]


def _total_remaining_prefill(env: Any, state: Any) -> int:
    req_map = env._req_map(state.simulator)
    total = 0
    for rid in list(getattr(state.stats, "active_request_ids", set()) or set()):
        req = req_map.get(int(rid))
        if req is None or bool(getattr(req, "completed", False)):
            continue
        prefill_done = bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)))
        if prefill_done:
            continue
        total += max(0, int(env._remaining_prefill(req)))
    return int(total)


def select_trivial_controller_action(
    *,
    runner: Any,
    state: Any,
    heuristic: str,
    budget_tokens: int,
    eviction_rule: str,
    actions_by_index: Optional[Sequence[Any | None]] = None,
    valid_indices: Optional[Sequence[int]] = None,
) -> Tuple[Optional[ControllerAction], Dict[str, Any]]:
    """
    Select a fixed controller policy action from existing sampled valid actions:
      - fixed ordering heuristic (e.g., SJF/LST)
      - fixed budget cap with min(cap, total_remaining_prefill)
      - fixed eviction rule

    If no active prefill exists, returns strict no-op to encourage immediate jump-to-tick.
    """
    env = runner.env
    total_prefill = _total_remaining_prefill(env, state)
    target_prefill = int(min(max(0, int(budget_tokens)), max(0, int(total_prefill))))

    if actions_by_index is None or valid_indices is None:
        actions_by_index, mask = runner._sample_actions_readonly(state, "controller")
        mask_list = _mask_to_list(mask)
        valid_indices = [
            int(i)
            for i, ok in enumerate(mask_list)
            if bool(ok) and i < len(actions_by_index) and actions_by_index[i] is not None
        ]
    else:
        actions_by_index = list(actions_by_index)
        valid_indices = [
            int(i)
            for i in list(valid_indices)
            if int(i) < len(actions_by_index) and actions_by_index[int(i)] is not None
        ]

    if total_prefill <= 0:
        return make_strict_noop_controller_action(eviction_rule=eviction_rule), {
            "selection_mode": "trivial_no_prefill_noop",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
            "target_prefill_budget": 0,
            "chosen_prefill_budget": 0,
        }

    candidates: List[Tuple[int, ControllerAction, int]] = []
    for idx in valid_indices:
        act = actions_by_index[idx]
        if not isinstance(act, ControllerAction):
            continue

        strategy = str(getattr(act, "strategy", "") or "")
        if eviction_rule and not (strategy == eviction_rule or strategy.endswith(f"|{eviction_rule}")):
            continue

        heur = str(getattr(act, "heuristic", "") or "")
        if target_prefill > 0 and heur != str(heuristic):
            continue

        prefill_budget = int(sum((getattr(act, "prefill_allocations", {}) or {}).values()))
        candidates.append((int(idx), act, int(prefill_budget)))

    if not candidates:
        # Fallback to first valid sampled controller action.
        if valid_indices:
            idx = int(valid_indices[0])
            act = actions_by_index[idx]
            if isinstance(act, ControllerAction):
                chosen_prefill = int(sum((act.prefill_allocations or {}).values()))
                return act, {
                    "selection_mode": "trivial_fallback_first_valid",
                    "valid_action_count": int(len(valid_indices)),
                    "iterations_requested": 0,
                    "iterations_used": 0,
                    "target_prefill_budget": int(target_prefill),
                    "chosen_prefill_budget": int(chosen_prefill),
                }
        return None, {
            "selection_mode": "trivial_no_valid_action",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
            "target_prefill_budget": int(target_prefill),
            "chosen_prefill_budget": 0,
        }

    exact = [x for x in candidates if int(x[2]) == int(target_prefill)]
    if exact:
        idx, act, chosen_pref = min(exact, key=lambda t: t[0])
        return act, {
            "selection_mode": "trivial_exact_budget",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
            "target_prefill_budget": int(target_prefill),
            "chosen_prefill_budget": int(chosen_pref),
        }

    below = [x for x in candidates if int(x[2]) <= int(target_prefill)]
    if below:
        idx, act, chosen_pref = max(below, key=lambda t: (int(t[2]), -int(t[0])))
        return act, {
            "selection_mode": "trivial_nearest_below_budget",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
            "target_prefill_budget": int(target_prefill),
            "chosen_prefill_budget": int(chosen_pref),
        }

    # Otherwise pick smallest above target.
    idx, act, chosen_pref = min(candidates, key=lambda t: (int(t[2]), int(t[0])))
    return act, {
        "selection_mode": "trivial_nearest_above_budget",
        "valid_action_count": int(len(valid_indices)),
        "iterations_requested": 0,
        "iterations_used": 0,
        "target_prefill_budget": int(target_prefill),
        "chosen_prefill_budget": int(chosen_pref),
    }
