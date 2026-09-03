"""Regression tests for eviction-aware controller action identity."""

from __future__ import annotations

from vidur.Game_Version3.game_types import ControllerAction
from vidur.Game_Version3.mcts_value_prior import VidurMCTS


def _decode_action(evicted_request_ids: list[int]) -> ControllerAction:
    return ControllerAction(
        token_budget=2,
        selected_request_ids=[0, 1],
        token_allocations={0: 1, 1: 1},
        prefill_allocations={},
        decode_allocations={0: 1, 1: 1},
        heuristic=None,
        strategy="GV2|evict_none",
        mapping=(0, 0, 0),
        _evicted_request_ids=evicted_request_ids,
    )


def test_eviction_changes_controller_action_identity() -> None:
    keep_all = _decode_action([])
    evict_request_2 = _decode_action([2])
    mcts = object.__new__(VidurMCTS)

    keep_key = mcts.get_controller_action_key(keep_all)
    evict_key = mcts.get_controller_action_key(evict_request_2)

    assert keep_key != evict_key
    assert keep_key[-1] == ()
    assert evict_key[-1] == (2,)


def test_canonicalization_keeps_distinct_eviction_actions() -> None:
    keep_all = _decode_action([])
    duplicate_keep_all = _decode_action([])
    evict_request_2 = _decode_action([2])
    mcts = object.__new__(VidurMCTS)
    actions = [keep_all, duplicate_keep_all, evict_request_2]

    alias_to_canon, canon_to_aliases, canonical = (
        mcts.canonicalize_action_indices(
            player="controller",
            actions_by_index=actions,
            valid_indices=[0, 1, 2],
        )
    )

    assert canonical == [0, 2]
    assert alias_to_canon == {0: 0, 1: 0, 2: 2}
    assert canon_to_aliases == {0: [0, 1], 2: [2]}


def test_default_eviction_metadata_is_not_shared() -> None:
    first = _decode_action([])
    second = _decode_action([])

    first._evicted_request_ids.append(3)

    assert second._evicted_request_ids == []
