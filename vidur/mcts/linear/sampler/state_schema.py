from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence


def encode_int_list(values: Sequence[int]) -> str:
    return json.dumps([int(v) for v in values], ensure_ascii=False, separators=(",", ":"))


def encode_int_float_map(values: Dict[int, float]) -> str:
    return json.dumps(
        {str(int(k)): float(v) for k, v in values.items()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def encode_int_int_map(values: Dict[int, int]) -> str:
    return json.dumps(
        {str(int(k)): int(v) for k, v in values.items()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class StateRow:
    state_id: str
    worker_id: int
    round_idx: int
    root_seed: int
    history_hop: int
    player_to_act: str
    branching_depth: int
    sim_time: float
    requests_in_system: int
    requests_generated: int
    requests_completed: int
    slo_violations: int
    total_lateness: float
    total_cost: float
    completed_request_ids_json: str
    waiting_request_ids_json: str
    terminal: bool

    def as_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestRow:
    state_id: str
    request_id: int
    prefill_tokens_total: int
    decode_tokens_total: int
    prefill_tokens_remaining: int
    decode_tokens_remaining: int
    arrived_at: float
    queued_at: float
    prefill_complete: bool
    completed: bool
    prefill_slo: float
    decode_slo: float
    prefill_deadline: float
    decode_deadline: float
    prefill_lateness_now: float
    decode_lateness_now: float
    prefill_violated_now: bool
    decode_violated_now: bool
    total_lateness_now: float
    violated_now: bool

    def as_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRow:
    action_id: str
    state_id: str
    actor: str
    canonical_index: int
    canonical_key: str
    alias_indices_json: str
    action_json: str
    action_repr: str

    def as_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionRow:
    transition_id: str
    state_id: str
    action_id: str
    next_state_id: str
    actor: str
    sim_time_before: float
    sim_time_after_action: float
    sim_time_after_advance: float
    delta_time_action: float
    delta_time_advance: float
    delta_time: float
    cost_s: float
    cost_next: float
    reward: float
    branching_depth: int
    next_branching_depth: int
    terminal: bool
    adversary_prefill_deadlines_by_id_json: str

    def as_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeRow:
    node_id: str
    worker_id: int
    round_idx: int
    game_id: int
    trace_id: str
    root_id: int
    step_idx: int
    parent_node_id: str
    state_id: str
    incoming_action_id: str
    branching_depth: int

    def as_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionRequestDeltaRow:
    transition_id: str
    request_id: int
    prefill_lateness_delta: float
    decode_lateness_delta: float
    total_lateness_delta: float
    violation_delta: int

    def as_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorSampleRow:
    anchor_state_id: str
    worker_id: int
    round_idx: int
    history_hop: int
    player_to_act: str
    branching_depth: int
    trace_len: int
    attempt_index: int

    def as_record(self) -> Dict[str, Any]:
        return asdict(self)
