from __future__ import annotations

import hashlib
import json
import random
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...DNN.history_root import HistoryRootGenerator
from ..bellman import state_cost
from ..collector_worker import (
    _build_env,
    _load_history_rows,
    _replay_history_to_state_after_last_action,
)
from ...environment import AdversaryAction, ControllerAction, VidurMCTSState
from .config import SamplerRunConfig
from .state_schema import (
    ActionRow,
    AnchorSampleRow,
    NodeRow,
    RequestRow,
    StateRow,
    TransitionRequestDeltaRow,
    TransitionRow,
    encode_int_float_map,
    encode_int_list,
)
from .storage import WorkerShardPaths, write_worker_shard


@dataclass(frozen=True)
class SamplerWorkerTask:
    worker_id: int
    round_idx: int
    seed: int
    history_hop: int
    out_dir: str
    cfg: SamplerRunConfig
    target_anchor_samples: int = 0


@dataclass(frozen=True)
class SamplerWorkerResult:
    worker_id: int
    round_idx: int
    history_hop: int
    ok: bool
    unique_branching_states: int
    accepted_anchor_samples: int
    accepted_anchor_state_ids: Tuple[str, ...]
    shard: Optional[WorkerShardPaths]
    error: str = ""


@dataclass
class RuntimeAction:
    action_id: str
    actor: str
    canonical_index: int
    canonical_key: str
    alias_indices: List[int]
    action_obj: object
    action_json: str
    action_repr: str


@dataclass
class RuntimeNode:
    node_id: str
    state_id: str
    player_to_act: str
    branching_depth: int
    parent_node_id: str
    incoming_action_id: str
    trace_id: str
    step_idx: int
    game_id: int
    root_id: int
    snapshot: Any
    stats_snapshot: Any
    terminal: bool


class _WorkerSampler:
    def __init__(self, task: SamplerWorkerTask) -> None:
        self.task = task
        self.cfg = task.cfg
        self.rng = random.Random(int(task.seed))

        _, self.env = _build_env(self.cfg, use_virtual_env=bool(self.cfg.collection.use_virtual_env))

        self.runtime_nodes: Dict[str, RuntimeNode] = {}
        self.runtime_actions: Dict[str, RuntimeAction] = {}

        self.state_rows_by_id: Dict[str, StateRow] = {}
        self.request_rows_by_key: Dict[Tuple[str, int], RequestRow] = {}
        self.action_rows_by_id: Dict[str, ActionRow] = {}
        self.transition_rows_by_id: Dict[str, TransitionRow] = {}
        self.node_rows_by_id: Dict[str, NodeRow] = {}
        self.transition_deltas_by_key: Dict[Tuple[str, int], TransitionRequestDeltaRow] = {}
        self.anchor_rows_by_state_id: Dict[str, AnchorSampleRow] = {}

        self.state_to_first_node: Dict[str, str] = {}
        self.expanded_state_ids: set[str] = set()
        self.unique_branching_state_ids: set[str] = set()
        self.accepted_anchor_state_ids: set[str] = set()
        self.edge_to_child: Dict[Tuple[str, str], str] = {}

        self._next_node_serial = 0
        self._next_action_serial = 0
        self._next_transition_serial = 0

        self._snapshot_lru: OrderedDict[str, None] = OrderedDict()

    def _new_node_id(self) -> str:
        nid = f"w{int(self.task.worker_id):02d}_r{int(self.task.round_idx):03d}_n{self._next_node_serial:09d}"
        self._next_node_serial += 1
        return nid

    def _new_action_id(self, state_id: str, actor: str, canonical_index: int, canonical_key: str) -> str:
        payload = f"{state_id}|{actor}|{int(canonical_index)}|{canonical_key}"
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
        aid = f"w{int(self.task.worker_id):02d}_a{self._next_action_serial:09d}_{digest}"
        self._next_action_serial += 1
        return aid

    def _new_transition_id(self, state_id: str, action_id: str, next_state_id: str) -> str:
        payload = f"{state_id}|{action_id}|{next_state_id}"
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
        tid = f"w{int(self.task.worker_id):02d}_t{self._next_transition_serial:09d}_{digest}"
        self._next_transition_serial += 1
        return tid

    def _touch_snapshot(self, node_id: str) -> None:
        if node_id in self._snapshot_lru:
            self._snapshot_lru.move_to_end(node_id)
        else:
            self._snapshot_lru[node_id] = None
        self._evict_snapshots_if_needed()

    def _store_snapshot(self, node_id: str, snapshot: Any, stats_snapshot: Any) -> None:
        node = self.runtime_nodes[node_id]
        node.snapshot = snapshot
        node.stats_snapshot = stats_snapshot
        self._touch_snapshot(node_id)

    def _evict_snapshots_if_needed(self) -> None:
        cap = int(self.cfg.collection.max_snapshot_cache)
        if cap <= 0:
            cap = 1
        while len(self._snapshot_lru) > cap:
            old_node_id, _ = self._snapshot_lru.popitem(last=False)
            n = self.runtime_nodes.get(old_node_id)
            if n is None:
                continue
            n.snapshot = None
            n.stats_snapshot = None

    def _clone_from_snapshot(self, snapshot: Any, stats_template: Any) -> VidurMCTSState:
        if hasattr(self.env, "clone_state_from_snapshot"):
            return self.env.clone_state_from_snapshot(snapshot, stats_template)
        s = self.env.initial_state()
        s.simulator.restore_state(snapshot)
        s.stats = stats_template.clone()
        return s

    def _materialize_node_state(self, node_id: str) -> VidurMCTSState:
        node = self.runtime_nodes[node_id]
        if node.snapshot is not None and node.stats_snapshot is not None:
            self._touch_snapshot(node_id)
            return self._clone_from_snapshot(node.snapshot, node.stats_snapshot)

        path: List[str] = []
        cur = node
        while cur.snapshot is None or cur.stats_snapshot is None:
            if not cur.parent_node_id:
                raise RuntimeError(f"node {node_id} has no recoverable snapshot path")
            path.append(cur.node_id)
            parent = self.runtime_nodes.get(cur.parent_node_id)
            if parent is None:
                raise RuntimeError(f"missing parent node {cur.parent_node_id} while restoring {node_id}")
            cur = parent

        path.reverse()
        state = self._clone_from_snapshot(cur.snapshot, cur.stats_snapshot)
        player = str(cur.player_to_act)
        self._touch_snapshot(cur.node_id)

        for nid in path:
            child = self.runtime_nodes[nid]
            action_id = child.incoming_action_id
            if not action_id:
                continue
            ra = self.runtime_actions[action_id]
            state, player = self._apply_action(state, player, ra.action_obj)
            state, player, _terminal = self._advance_to_branching_or_terminal(state, player)

        self._store_snapshot(node_id, state.simulator.snapshot_state(), state.stats.clone())
        return self._clone_from_snapshot(self.runtime_nodes[node_id].snapshot, self.runtime_nodes[node_id].stats_snapshot)

    @staticmethod
    def _mask_to_list(mask: Any) -> List[bool]:
        try:
            import torch
        except Exception:  # pragma: no cover
            torch = None
        if torch is not None and isinstance(mask, torch.Tensor):
            return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
        return [bool(x) for x in mask]

    def _actions_and_valid(self, state: VidurMCTSState, player: str) -> Tuple[List[Optional[object]], List[int]]:
        max_samples = int(self.cfg.collection.enum_max_samples)
        if player == "controller":
            actions_by_index, mask = self.env.sample_controller_actions(state, max_samples)
        else:
            actions_by_index, mask = self.env.sample_adversary_actions(state, max_samples)
        mask_list = self._mask_to_list(mask)
        valid = [
            i
            for i, ok in enumerate(mask_list)
            if ok and i < len(actions_by_index) and actions_by_index[i] is not None
        ]
        return actions_by_index, valid

    def _apply_action(self, state: VidurMCTSState, player: str, action: object) -> Tuple[VidurMCTSState, str]:
        if player == "controller":
            st = self.env.apply_controller_action_only(state, action, inplace=True)
            return st, "adversary"
        st = self.env.apply_adversary_action_only(state, action, inplace=True)
        return st, "controller"

    def _advance_to_branching_or_terminal(
        self,
        state: VidurMCTSState,
        player: str,
    ) -> Tuple[VidurMCTSState, str, bool]:
        max_hops = int(self.cfg.collection.max_forced_hops)
        for _ in range(max_hops):
            actions_by_index, valid = self._actions_and_valid(state, player)
            if not valid:
                return state, player, True
            if len(valid) >= 2:
                return state, player, False
            forced = actions_by_index[int(valid[0])]
            assert forced is not None
            state, player = self._apply_action(state, player, forced)
        raise RuntimeError(f"exceeded max_forced_hops={max_hops}")

    @staticmethod
    def _controller_action_key(action: ControllerAction) -> Tuple[Tuple[int, int], ...]:
        alloc = action.token_allocations or {}
        return tuple(sorted((int(rid), int(tok)) for rid, tok in alloc.items()))

    def _canonical_actions(
        self,
        state: VidurMCTSState,
        player: str,
    ) -> List[Tuple[int, List[int], object, str]]:
        actions_by_index, valid = self._actions_and_valid(state, player)
        if not valid:
            return []

        if player != "controller" or len(valid) <= 1:
            out: List[Tuple[int, List[int], object, str]] = []
            for idx in sorted(valid):
                action = actions_by_index[idx]
                assert action is not None
                out.append((int(idx), [int(idx)], action, f"idx:{int(idx)}"))
            return out

        groups: Dict[Tuple[Tuple[int, int], ...], List[int]] = defaultdict(list)
        action_for_idx: Dict[int, object] = {}
        for idx in sorted(valid):
            action = actions_by_index[idx]
            assert action is not None
            key = self._controller_action_key(action)
            groups[key].append(int(idx))
            action_for_idx[int(idx)] = action

        canonical: List[Tuple[int, List[int], object, str]] = []
        for key in sorted(groups.keys()):
            aliases = sorted(groups[key])
            canon = int(aliases[0])
            canonical_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            canonical.append((canon, aliases, action_for_idx[canon], canonical_key))
        return canonical

    def _action_to_json(self, action: object) -> str:
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

    def _build_request_lookup(self, state: VidurMCTSState) -> Dict[int, Any]:
        fn = getattr(self.env, "_build_request_lookup")
        try:
            raw = fn(state.simulator, state)
        except TypeError:
            raw = fn(state.simulator)
        return {int(k): v for k, v in dict(raw).items()}

    @staticmethod
    def _state_stats_map(state: VidurMCTSState, req_ids: Iterable[int]) -> Dict[int, Tuple[float, float, float, int]]:
        out: Dict[int, Tuple[float, float, float, int]] = {}
        stats = state.stats
        for rid in req_ids:
            prefill = float(stats.per_request_prefill_lateness.get(int(rid), 0.0))
            decode = float(stats.per_request_decode_lateness.get(int(rid), 0.0))
            total = prefill + decode
            violated = 1 if (int(rid) in stats.violated_request_ids or total > 0.0) else 0
            out[int(rid)] = (prefill, decode, total, violated)
        return out

    def _extract_state_and_requests(
        self,
        state: VidurMCTSState,
        *,
        player_to_act: str,
        branching_depth: int,
        terminal: bool,
    ) -> Tuple[StateRow, List[RequestRow], Dict[int, Tuple[float, float, float, int]], str]:
        desc = self.env.describe_state(state)
        request_lookup = self._build_request_lookup(state)
        stats = state.stats

        request_rows: List[RequestRow] = []
        req_digest_parts: List[Tuple[Any, ...]] = []

        for rid in sorted(request_lookup.keys()):
            req = request_lookup[rid]
            prefill_total = int(getattr(req, "num_prefill_tokens", 0))
            decode_total = int(getattr(req, "num_decode_tokens", 0))
            prefill_done = int(getattr(req, "num_processed_prefill_tokens", 0))
            decode_done = int(getattr(req, "num_processed_decode_tokens", 0))
            prefill_remaining = max(0, prefill_total - prefill_done)
            decode_remaining = max(0, decode_total - decode_done)

            arrived_at = float(getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0)) or 0.0)
            queued_at = float(getattr(req, "queued_at", arrived_at) or arrived_at)
            prefill_complete = bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)))
            completed = bool(getattr(req, "completed", False))

            prefill_slo = float(getattr(req, "_prefill_slo_time", getattr(req, "prefill_slo_time", 0.0)) or 0.0)
            decode_slo = float(getattr(req, "_decode_slo_time", getattr(req, "decode_slo_time", 0.0)) or 0.0)
            prefill_deadline = float(arrived_at + prefill_slo)

            decode_deadline = float(stats.decode_next_deadline_by_id.get(rid, 0.0))
            if decode_deadline == 0.0 and prefill_complete:
                prefill_completed_at = float(getattr(req, "_prefill_completed_at", 0.0) or 0.0)
                if prefill_completed_at > 0.0 and decode_slo > 0.0:
                    decode_deadline = prefill_completed_at + decode_slo

            prefill_lateness = float(stats.per_request_prefill_lateness.get(rid, 0.0))
            decode_lateness = float(stats.per_request_decode_lateness.get(rid, 0.0))
            total_lateness = prefill_lateness + decode_lateness
            prefill_violated = prefill_lateness > 0.0
            decode_violated = decode_lateness > 0.0
            violated = bool((rid in stats.violated_request_ids) or (total_lateness > 0.0))

            rr = RequestRow(
                state_id="",
                request_id=int(rid),
                prefill_tokens_total=prefill_total,
                decode_tokens_total=decode_total,
                prefill_tokens_remaining=prefill_remaining,
                decode_tokens_remaining=decode_remaining,
                arrived_at=arrived_at,
                queued_at=queued_at,
                prefill_complete=prefill_complete,
                completed=completed,
                prefill_slo=prefill_slo,
                decode_slo=decode_slo,
                prefill_deadline=prefill_deadline,
                decode_deadline=decode_deadline,
                prefill_lateness_now=prefill_lateness,
                decode_lateness_now=decode_lateness,
                prefill_violated_now=prefill_violated,
                decode_violated_now=decode_violated,
                total_lateness_now=total_lateness,
                violated_now=violated,
            )
            request_rows.append(rr)
            req_digest_parts.append(
                (
                    int(rid),
                    prefill_total,
                    decode_total,
                    prefill_done,
                    decode_done,
                    round(arrived_at, 9),
                    round(queued_at, 9),
                    int(prefill_complete),
                    int(completed),
                    round(prefill_slo, 9),
                    round(decode_slo, 9),
                )
            )

        waiting_ids = [int(r.request_id) for r in request_rows if not r.completed]
        completed_ids = sorted(int(x) for x in set(stats.completed_request_ids))

        sim_time = float(desc.get("sim_time", getattr(state.simulator, "_time", 0.0)))
        slo_viol = int(desc.get("slo_violations", 0))
        total_late = float(desc.get("total_lateness", desc.get("avg_lateness", 0.0)))
        total_cost = float(slo_viol) + float(total_late)

        state_payload = {
            "player_to_act": str(player_to_act),
            "sim_time": round(sim_time, 9),
            "slo_violations": int(slo_viol),
            "total_lateness": round(total_late, 9),
            "requests": req_digest_parts,
            "completed_ids": completed_ids,
            "last_prefill_batch_time": float(getattr(stats, "last_prefill_batch_time", 0.0) or 0.0),
        }
        digest = hashlib.sha1(
            json.dumps(state_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        state_id = f"s_{digest}"

        sr = StateRow(
            state_id=state_id,
            worker_id=int(self.task.worker_id),
            round_idx=int(self.task.round_idx),
            root_seed=int(self.task.seed),
            history_hop=int(self.task.history_hop),
            player_to_act=str(player_to_act),
            branching_depth=int(branching_depth),
            sim_time=sim_time,
            requests_in_system=int(desc.get("requests_in_system", len(request_rows))),
            requests_generated=int(desc.get("requests_generated", 0)),
            requests_completed=int(desc.get("requests_completed", 0)),
            slo_violations=int(slo_viol),
            total_lateness=float(total_late),
            total_cost=float(total_cost),
            completed_request_ids_json=encode_int_list(completed_ids),
            waiting_request_ids_json=encode_int_list(waiting_ids),
            terminal=bool(terminal),
        )

        metrics = self._state_stats_map(state, req_ids=[r.request_id for r in request_rows])
        return sr, request_rows, metrics, state_id

    def _record_state(
        self,
        sr: StateRow,
        request_rows: List[RequestRow],
        *,
        is_branching: bool,
    ) -> bool:
        is_new = sr.state_id not in self.state_rows_by_id
        if is_new:
            self.state_rows_by_id[sr.state_id] = sr
            for rr in request_rows:
                rr2 = RequestRow(state_id=sr.state_id, **{k: v for k, v in rr.as_record().items() if k != "state_id"})
                self.request_rows_by_key[(sr.state_id, rr2.request_id)] = rr2
        if is_branching:
            self.unique_branching_state_ids.add(sr.state_id)
        return is_new

    def _generate_root_state(self) -> Tuple[VidurMCTSState, str]:
        root_player = str(self.cfg.collection.root_player)
        state = self.env.initial_state()
        player = root_player

        history_csv = str(self.cfg.collection.history_csv or "").strip()
        if history_csv:
            rows = _load_history_rows(Path(history_csv))
            state, player = _replay_history_to_state_after_last_action(
                self.env,
                rows,
                root_player=root_player,
                align_branching_roots=bool(self.cfg.collection.align_branching_roots),
                max_forced_hops=int(self.cfg.collection.max_forced_hops),
                max_samples=int(self.cfg.collection.enum_max_samples),
            )

        hgen = HistoryRootGenerator(
            env=self.env,
            max_branching=int(self.cfg.collection.max_branching),
            iter_logger=None,
            root_logger=None,
        )
        state, player, _depth, _next_node_id, _last_node_id = hgen.generate_history_root(
            state=state,
            player=player,
            depth=0,
            nontrivial_hops=int(self.task.history_hop),
            game_id=int(self.task.round_idx) * 1000 + int(self.task.worker_id),
            root_id_for_logs=int(self.task.worker_id),
            seed=int(self.task.seed),
            log_history=False,
            max_total_steps=int(self.cfg.collection.max_forced_hops),
            log_node_id_start=0,
            log_parent_id_start=None,
        )
        return state, player

    def _walk_random_trace_to_anchor(
        self,
        root_state: VidurMCTSState,
        root_player: str,
        *,
        trace_len: int,
    ) -> Tuple[VidurMCTSState, str, int, bool]:
        state = self._clone_from_snapshot(root_state.simulator.snapshot_state(), root_state.stats.clone())
        player = str(root_player)
        terminal = False
        branching_depth = 0

        for _ in range(int(trace_len)):
            state, player, terminal = self._advance_to_branching_or_terminal(state, player)
            if terminal:
                break

            canonical = self._canonical_actions(state, player)
            if len(canonical) < 2:
                terminal = True
                break

            pick = self.rng.randrange(len(canonical))
            _canon_idx, _aliases, action_obj, _key = canonical[pick]
            state, player = self._apply_action(state, player, action_obj)
            branching_depth += 1

        state, player, terminal = self._advance_to_branching_or_terminal(state, player)
        return state, player, branching_depth, terminal

    def _get_or_create_child_for_action(
        self,
        *,
        parent_node_id: str,
        parent_state: VidurMCTSState,
        parent_player: str,
        canonical_index: int,
        aliases: List[int],
        action_obj: object,
        canonical_key: str,
    ) -> str:
        edge_key = (str(parent_node_id), str(canonical_key))
        existing = self.edge_to_child.get(edge_key, "")
        if existing:
            return existing

        node = self.runtime_nodes[parent_node_id]
        actor = str(parent_player)
        parent_cost = float(state_cost(self.env, parent_state))
        parent_sim_time = float(getattr(parent_state.simulator, "_time", 0.0))

        _parent_sr, _parent_rrs, parent_metrics, _ = self._extract_state_and_requests(
            parent_state,
            player_to_act=parent_player,
            branching_depth=int(node.branching_depth),
            terminal=bool(node.terminal),
        )
        before_ids = set(parent_metrics.keys())

        action_json = self._action_to_json(action_obj)
        action_repr = repr(action_obj)
        action_id = self._new_action_id(node.state_id, actor, int(canonical_index), str(canonical_key))
        runtime_action = RuntimeAction(
            action_id=action_id,
            actor=actor,
            canonical_index=int(canonical_index),
            canonical_key=str(canonical_key),
            alias_indices=[int(x) for x in aliases],
            action_obj=action_obj,
            action_json=action_json,
            action_repr=action_repr,
        )
        self.runtime_actions[action_id] = runtime_action
        self.action_rows_by_id[action_id] = ActionRow(
            action_id=action_id,
            state_id=node.state_id,
            actor=actor,
            canonical_index=int(canonical_index),
            canonical_key=str(canonical_key),
            alias_indices_json=encode_int_list([int(x) for x in aliases]),
            action_json=action_json,
            action_repr=action_repr,
        )

        child_state = self._clone_from_snapshot(parent_state.simulator.snapshot_state(), parent_state.stats.clone())
        child_state, next_player = self._apply_action(child_state, parent_player, action_obj)
        child_sim_time_after_action = float(getattr(child_state.simulator, "_time", 0.0))
        child_state, next_player, child_terminal = self._advance_to_branching_or_terminal(child_state, next_player)

        child_sr, child_rrs, child_metrics, child_state_id = self._extract_state_and_requests(
            child_state,
            player_to_act=next_player,
            branching_depth=int(node.branching_depth + 1),
            terminal=bool(child_terminal),
        )
        child_actions = []
        if not child_terminal:
            child_actions = self._canonical_actions(child_state, next_player)
        is_child_branching = (not child_terminal) and (len(child_actions) >= 2)
        self._record_state(child_sr, child_rrs, is_branching=is_child_branching)

        child_cost = float(state_cost(self.env, child_state))
        child_sim_time = float(getattr(child_state.simulator, "_time", 0.0))
        reward = float(parent_cost - child_cost)

        after_ids = set(child_metrics.keys())
        new_req_ids = sorted(after_ids - before_ids)
        adv_deadlines: Dict[int, float] = {}
        if actor == "adversary" and isinstance(action_obj, AdversaryAction):
            for rid in new_req_ids:
                rr = self.request_rows_by_key.get((child_state_id, int(rid)))
                if rr is not None:
                    adv_deadlines[int(rid)] = float(rr.prefill_deadline)

        transition_id = self._new_transition_id(node.state_id, action_id, child_state_id)
        self.transition_rows_by_id[transition_id] = TransitionRow(
            transition_id=transition_id,
            state_id=node.state_id,
            action_id=action_id,
            next_state_id=child_state_id,
            actor=actor,
            sim_time_before=float(parent_sim_time),
            sim_time_after_action=float(child_sim_time_after_action),
            sim_time_after_advance=float(child_sim_time),
            delta_time_action=max(0.0, child_sim_time_after_action - parent_sim_time),
            delta_time_advance=max(0.0, child_sim_time - child_sim_time_after_action),
            delta_time=max(0.0, child_sim_time - parent_sim_time),
            cost_s=float(parent_cost),
            cost_next=float(child_cost),
            reward=float(reward),
            branching_depth=int(node.branching_depth),
            next_branching_depth=int(node.branching_depth + 1),
            terminal=bool(child_terminal),
            adversary_prefill_deadlines_by_id_json=encode_int_float_map(adv_deadlines),
        )

        delta_ids = sorted(before_ids | after_ids)
        for rid in delta_ids:
            b_pref, b_dec, b_tot, b_v = parent_metrics.get(int(rid), (0.0, 0.0, 0.0, 0))
            a_pref, a_dec, a_tot, a_v = child_metrics.get(int(rid), (0.0, 0.0, 0.0, 0))
            delta = TransitionRequestDeltaRow(
                transition_id=transition_id,
                request_id=int(rid),
                prefill_lateness_delta=float(a_pref - b_pref),
                decode_lateness_delta=float(a_dec - b_dec),
                total_lateness_delta=float(a_tot - b_tot),
                violation_delta=int(a_v - b_v),
            )
            self.transition_deltas_by_key[(transition_id, int(rid))] = delta

        child_node = self._create_node(
            state_id=child_state_id,
            player_to_act=next_player,
            branching_depth=int(node.branching_depth + 1),
            parent_node_id=parent_node_id,
            incoming_action_id=action_id,
            trace_id=node.trace_id,
            game_id=node.game_id,
            root_id=node.root_id,
            terminal=bool(child_terminal),
            snapshot=child_state.simulator.snapshot_state(),
            stats_snapshot=child_state.stats.clone(),
        )
        if child_state_id not in self.state_to_first_node:
            self.state_to_first_node[child_state_id] = child_node.node_id

        self.edge_to_child[edge_key] = child_node.node_id
        return child_node.node_id

    def _walk_random_trace_to_anchor_node(
        self,
        *,
        root_node_id: str,
        trace_len: int,
    ) -> str:
        current_node_id = str(root_node_id)
        for _ in range(int(trace_len)):
            cur = self.runtime_nodes[current_node_id]
            if cur.terminal:
                break
            cur_state = self._materialize_node_state(current_node_id)
            cur_player = str(cur.player_to_act)
            canonical = self._canonical_actions(cur_state, cur_player)
            if len(canonical) < 2:
                break
            pick = self.rng.randrange(len(canonical))
            canonical_index, aliases, action_obj, canonical_key = canonical[pick]
            current_node_id = self._get_or_create_child_for_action(
                parent_node_id=current_node_id,
                parent_state=cur_state,
                parent_player=cur_player,
                canonical_index=int(canonical_index),
                aliases=[int(x) for x in aliases],
                action_obj=action_obj,
                canonical_key=str(canonical_key),
            )
        return current_node_id

    def _create_node(
        self,
        *,
        state_id: str,
        player_to_act: str,
        branching_depth: int,
        parent_node_id: str,
        incoming_action_id: str,
        trace_id: str,
        game_id: int,
        root_id: int,
        terminal: bool,
        snapshot: Any,
        stats_snapshot: Any,
    ) -> RuntimeNode:
        node_id = self._new_node_id()
        node = RuntimeNode(
            node_id=node_id,
            state_id=state_id,
            player_to_act=str(player_to_act),
            branching_depth=int(branching_depth),
            parent_node_id=str(parent_node_id),
            incoming_action_id=str(incoming_action_id),
            trace_id=str(trace_id),
            step_idx=int(branching_depth),
            game_id=int(game_id),
            root_id=int(root_id),
            snapshot=snapshot,
            stats_snapshot=stats_snapshot,
            terminal=bool(terminal),
        )
        self.runtime_nodes[node_id] = node
        self.node_rows_by_id[node_id] = NodeRow(
            node_id=node_id,
            worker_id=int(self.task.worker_id),
            round_idx=int(self.task.round_idx),
            game_id=int(game_id),
            trace_id=str(trace_id),
            root_id=int(root_id),
            step_idx=int(branching_depth),
            parent_node_id=str(parent_node_id),
            state_id=str(state_id),
            incoming_action_id=str(incoming_action_id),
            branching_depth=int(branching_depth),
        )
        self._store_snapshot(node_id, snapshot, stats_snapshot)
        return node

    def _expand_node(self, node_id: str) -> None:
        node = self.runtime_nodes[node_id]
        if node.state_id in self.expanded_state_ids:
            return

        state = self._materialize_node_state(node_id)
        player = str(node.player_to_act)

        canonical_actions = self._canonical_actions(state, player)
        if len(canonical_actions) == 0:
            self.expanded_state_ids.add(node.state_id)
            return

        if len(canonical_actions) < 2:
            # Non-branching nodes are not expanded for LP constraints.
            self.expanded_state_ids.add(node.state_id)
            return

        for canonical_index, aliases, action_obj, canonical_key in canonical_actions:
            self._get_or_create_child_for_action(
                parent_node_id=node_id,
                parent_state=state,
                parent_player=player,
                canonical_index=int(canonical_index),
                aliases=[int(x) for x in aliases],
                action_obj=action_obj,
                canonical_key=str(canonical_key),
            )

        self.expanded_state_ids.add(node.state_id)

    def run(self) -> SamplerWorkerResult:
        try:
            root_state, root_player = self._generate_root_state()
            root_state, root_player, root_terminal = self._advance_to_branching_or_terminal(root_state, root_player)

            root_sr, root_rrs, _root_metrics, root_state_id = self._extract_state_and_requests(
                root_state,
                player_to_act=root_player,
                branching_depth=0,
                terminal=bool(root_terminal),
            )
            root_actions = [] if root_terminal else self._canonical_actions(root_state, root_player)
            root_is_branching = (not root_terminal) and (len(root_actions) >= 2)
            self._record_state(root_sr, root_rrs, is_branching=root_is_branching)

            game_id = int(self.task.round_idx) * 1_000_000 + int(self.task.worker_id)
            trace_id = f"worker_{int(self.task.worker_id):02d}_game_{game_id}"
            root_node = self._create_node(
                state_id=root_state_id,
                player_to_act=root_player,
                branching_depth=0,
                parent_node_id="",
                incoming_action_id="",
                trace_id=trace_id,
                game_id=game_id,
                root_id=0,
                terminal=bool(root_terminal),
                snapshot=root_state.simulator.snapshot_state(),
                stats_snapshot=root_state.stats.clone(),
            )
            self.state_to_first_node[root_state_id] = root_node.node_id

            target_samples = int(self.cfg.collection.shard_unique_states_per_worker)
            if int(self.task.target_anchor_samples) > 0:
                target_samples = int(self.task.target_anchor_samples)
            if target_samples <= 0:
                target_samples = int(self.cfg.collection.target_unique_states // max(1, int(self.cfg.collection.workers)))
            max_attempts = int(self.cfg.collection.max_expansions_per_worker)
            max_trace_length = max(1, int(self.cfg.collection.max_trace_length))

            unique_anchor_sample_count = 0
            attempts = 0

            while unique_anchor_sample_count < target_samples and attempts < max_attempts:
                attempts += 1
                trace_len = self.rng.randint(1, max_trace_length)
                anchor_node_id = self._walk_random_trace_to_anchor_node(
                    root_node_id=root_node.node_id,
                    trace_len=trace_len,
                )
                anchor_node = self.runtime_nodes[anchor_node_id]
                anchor_terminal = bool(anchor_node.terminal)
                anchor_player = str(anchor_node.player_to_act)
                anchor_depth = int(anchor_node.branching_depth)
                anchor_state = self._materialize_node_state(anchor_node_id)

                anchor_canonical = []
                if not anchor_terminal:
                    anchor_canonical = self._canonical_actions(anchor_state, anchor_player)
                is_anchor_branching = (not anchor_terminal) and (len(anchor_canonical) >= 2)
                if not is_anchor_branching:
                    continue

                anchor_sr, anchor_rrs, _anchor_metrics, anchor_state_id = self._extract_state_and_requests(
                    anchor_state,
                    player_to_act=anchor_player,
                    branching_depth=int(anchor_depth),
                    terminal=bool(anchor_terminal),
                )

                is_new_anchor = (anchor_state_id not in self.accepted_anchor_state_ids)
                if not is_new_anchor:
                    continue

                self._record_state(anchor_sr, anchor_rrs, is_branching=True)
                self.accepted_anchor_state_ids.add(anchor_state_id)
                self.anchor_rows_by_state_id[anchor_state_id] = AnchorSampleRow(
                    anchor_state_id=str(anchor_state_id),
                    worker_id=int(self.task.worker_id),
                    round_idx=int(self.task.round_idx),
                    history_hop=int(self.task.history_hop),
                    player_to_act=str(anchor_player),
                    branching_depth=int(anchor_depth),
                    trace_len=int(trace_len),
                    attempt_index=int(attempts),
                )

                if anchor_state_id not in self.expanded_state_ids:
                    self._expand_node(anchor_node_id)

                unique_anchor_sample_count += 1
                if unique_anchor_sample_count % 100 == 0:
                    print(
                        f"[sampler worker {int(self.task.worker_id)} hop={int(self.task.history_hop)}] "
                        f"attempts={attempts} accepted_samples={unique_anchor_sample_count}/{target_samples} "
                        f"max_attempts={max_attempts} max_trace_length={max_trace_length}",
                        flush=True,
                    )

            shard = write_worker_shard(
                out_dir=Path(self.task.out_dir) / f"round_{int(self.task.round_idx):03d}" / "shards",
                worker_id=int(self.task.worker_id),
                round_idx=int(self.task.round_idx),
                compression=str(self.cfg.output.parquet_compression),
                states=list(self.state_rows_by_id.values()),
                requests=list(self.request_rows_by_key.values()),
                actions=list(self.action_rows_by_id.values()),
                transitions=list(self.transition_rows_by_id.values()),
                nodes=list(self.node_rows_by_id.values()),
                transition_deltas=list(self.transition_deltas_by_key.values()),
                anchors=list(self.anchor_rows_by_state_id.values()),
            )

            return SamplerWorkerResult(
                worker_id=int(self.task.worker_id),
                round_idx=int(self.task.round_idx),
                history_hop=int(self.task.history_hop),
                ok=True,
                unique_branching_states=int(len(self.unique_branching_state_ids)),
                accepted_anchor_samples=int(len(self.accepted_anchor_state_ids)),
                accepted_anchor_state_ids=tuple(sorted(self.accepted_anchor_state_ids)),
                shard=shard,
            )
        except Exception:
            return SamplerWorkerResult(
                worker_id=int(self.task.worker_id),
                round_idx=int(self.task.round_idx),
                history_hop=int(self.task.history_hop),
                ok=False,
                unique_branching_states=0,
                accepted_anchor_samples=0,
                accepted_anchor_state_ids=tuple(),
                shard=None,
                error=traceback.format_exc(),
            )


def run_sampler_worker(task: SamplerWorkerTask, result_q: Any) -> None:
    sampler = _WorkerSampler(task)
    result = sampler.run()
    result_q.put(result)
