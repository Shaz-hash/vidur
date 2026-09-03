from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..game_types import AdversaryAction


def make_noop_adversary_action() -> AdversaryAction:
    return AdversaryAction(requests=[], stop_decode_ids=[])


def _request_count(action: AdversaryAction) -> int:
    return int(len(list(action.requests or [])))


def _prefill_tokens(action: AdversaryAction) -> int:
    total = 0
    for req in list(action.requests or []):
        total += int(getattr(req, "prefill_tokens", 0) or 0)
    return int(total)


def select_trivial_adversary_action(
    *,
    heuristic: str,
    actions_by_index: Sequence[Any | None],
    valid_indices: Sequence[int],
) -> Tuple[Optional[AdversaryAction], Dict[str, Any]]:
    valid_indices = [
        int(i)
        for i in list(valid_indices)
        if int(i) < len(actions_by_index) and isinstance(actions_by_index[int(i)], AdversaryAction)
    ]

    if not valid_indices:
        return None, {
            "selection_mode": "trivial_adv_no_valid_action",
            "valid_action_count": 0,
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    valid_actions = [(int(idx), actions_by_index[int(idx)]) for idx in valid_indices]
    nonempty = [(idx, act) for idx, act in valid_actions if _request_count(act) > 0]

    if heuristic == "noop":
        return make_noop_adversary_action(), {
            "selection_mode": "trivial_adv_noop",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    if heuristic == "first_valid":
        idx, act = valid_actions[0]
        return act, {
            "selection_mode": "trivial_adv_first_valid",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    if heuristic == "first_valid_nonempty":
        if nonempty:
            idx, act = nonempty[0]
            return act, {
                "selection_mode": "trivial_adv_first_valid_nonempty",
                "valid_action_count": int(len(valid_indices)),
                "iterations_requested": 0,
                "iterations_used": 0,
            }
        idx, act = valid_actions[0]
        return act, {
            "selection_mode": "trivial_adv_first_valid_fallback",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    pool = nonempty if nonempty else valid_actions
    if heuristic == "max_request_count":
        idx, act = max(pool, key=lambda item: (_request_count(item[1]), _prefill_tokens(item[1]), -int(item[0])))
        return act, {
            "selection_mode": "trivial_adv_max_request_count",
            "valid_action_count": int(len(valid_indices)),
            "iterations_requested": 0,
            "iterations_used": 0,
        }

    # default + explicit "max_prefill_tokens"
    idx, act = max(pool, key=lambda item: (_prefill_tokens(item[1]), _request_count(item[1]), -int(item[0])))
    return act, {
        "selection_mode": "trivial_adv_max_prefill_tokens",
        "valid_action_count": int(len(valid_indices)),
        "iterations_requested": 0,
        "iterations_used": 0,
    }
