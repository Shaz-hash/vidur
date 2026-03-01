# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
"""
PUCT trace validator for mcts_iter_puct.csv using mcts_iter.csv as fallback.

Run:
  python3 -m vidur.mcts.tests.DNN_TESTS.PUCT_tests
or
  python3 -m vidur.mcts.tests.DNN_TESTS.PUCT_tests \
    --puct-log vidur/simulator_output/mcts_dnn_logs/mcts_iter_puct.csv \
    --iter-log vidur/simulator_output/mcts_dnn_logs/mcts_iter.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Defaults (match mctsDNN.py)
# ---------------------------
DEFAULT_GAMMA = 0.98
DEFAULT_STEP_TIME = 0.02114359945138969  # fallback used in mctsDNN.py
DEFAULT_PB_C_BASE = 5000.0
DEFAULT_PB_C_INIT = 0.75
ABS_TOL = 1e-6
REL_TOL = 1e-6


def _isclose(a: float, b: float, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    return math.isclose(float(a), float(b), abs_tol=abs_tol, rel_tol=rel_tol)


def _to_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        s = str(x).strip()
        if s == "":
            return default
        return int(s)
    except Exception:
        return default


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _json_load(s: Any, default: Any) -> Any:
    if s is None:
        return default
    t = str(s).strip()
    if t == "":
        return default
    try:
        return json.loads(t)
    except Exception:
        try:
            return ast.literal_eval(t)
        except Exception:
            return default


def _resolve_default_path(name: str) -> Path:
    candidates = [
        Path(f"vidur/simulator_output/mcts_dnn_logs/{name}"),
        Path(f"simulator_output/mcts_dnn_logs/{name}"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@dataclass
class NodeState:
    node_id: int
    parent_id: Optional[int] = None
    visits: int = 0
    value_sum: float = 0.0
    reward_incoming: Optional[float] = None  # reward on edge parent -> this node
    sim_time: Optional[float] = None

    # table fields requested
    prior_by_action: Dict[int, float] = field(default_factory=dict)      # action_idx -> prior
    child_by_action: Dict[int, int] = field(default_factory=dict)        # action_idx -> child_id
    dedup_by_canonical: Dict[int, List[int]] = field(default_factory=dict)  # canonical_child_id -> alias action idx list

    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass
class RootContext:
    nodes: Dict[int, NodeState] = field(default_factory=dict)
    minmax_min: float = float("inf")
    minmax_max: float = float("-inf")


class PUCTValidator:
    def __init__(
        self,
        puct_rows: List[Dict[str, str]],
        iter_rows: List[Dict[str, str]],
        *,
        gamma: float,
        step_time: float,
        pb_c_base: float,
        pb_c_init: float,
        tol: float,
    ) -> None:
        self.rows = sorted(
            puct_rows,
            key=lambda r: (
                _to_int(r.get("game_id"), -1) or -1,
                _to_int(r.get("root_id"), -1) or -1,
                _to_int(r.get("sim_iteration"), -1) or -1,
            ),
        )
        self.gamma = float(gamma)
        self.step_time = max(float(step_time), 1e-12)
        self.pb_c_base = float(pb_c_base)
        self.pb_c_init = float(pb_c_init)
        self.tol = float(tol)

        self.roots: Dict[Tuple[int, int], RootContext] = {}
        self.iter_index: Dict[Tuple[int, int, int], List[Tuple[int, Dict[str, str]]]] = {}
        self._build_iter_index(iter_rows)

        self.total_rows = 0
        self.test1_pass = 0
        self.test2_pass = 0
        self.test3_pass = 0

    # -------------------
    # Index / table utils
    # -------------------
    def _build_iter_index(self, iter_rows: List[Dict[str, str]]) -> None:
        tmp: Dict[Tuple[int, int, int], List[Tuple[int, Dict[str, str]]]] = {}
        for r in iter_rows:
            g = _to_int(r.get("game_id"))
            root = _to_int(r.get("root_id"))
            n = _to_int(r.get("node_id"))
            s = _to_int(r.get("sim_iteration"), -1)
            if g is None or root is None or n is None:
                continue
            tmp.setdefault((g, root, n), []).append((s if s is not None else -1, r))
        for k, lst in tmp.items():
            lst.sort(key=lambda x: x[0])
            self.iter_index[k] = lst

    def _iter_lookup_row(self, game_id: int, root_id: int, node_id: int, sim_iter: int) -> Optional[Dict[str, str]]:
        lst = self.iter_index.get((game_id, root_id, node_id))
        if not lst:
            return None
        chosen = None
        for s, r in lst:
            if s <= sim_iter:
                chosen = r
            else:
                break
        if chosen is None:
            chosen = lst[-1][1]
        return chosen

    def _ctx(self, game_id: int, root_id: int) -> RootContext:
        return self.roots.setdefault((game_id, root_id), RootContext())

    def _node(self, ctx: RootContext, node_id: int) -> NodeState:
        st = ctx.nodes.get(node_id)
        if st is None:
            st = NodeState(node_id=node_id)
            ctx.nodes[node_id] = st
        return st

    def _discount(self, child_t: float, parent_t: float) -> float:
        dt = max(0.0, float(child_t) - float(parent_t))
        return self.gamma ** (dt / self.step_time)

    @staticmethod
    def _normalize(v: float, mn: float, mx: float) -> float:
        if mx > mn:
            return (v - mn) / (mx - mn)
        return v

    def _err(self, row: Dict[str, str], msg: str) -> None:
        g = _to_int(row.get("game_id"), -1)
        root = _to_int(row.get("root_id"), -1)
        sim = _to_int(row.get("sim_iteration"), -1)
        nid = _to_int(row.get("node_id"), -1)
        raise AssertionError(f"[g={g} root={root} sim={sim} node={nid}] {msg}")

    def _fill_node_fallbacks(
        self,
        ctx: RootContext,
        game_id: int,
        root_id: int,
        node_id: int,
        sim_iter: int,
    ) -> None:
        st = self._node(ctx, node_id)
        ir = self._iter_lookup_row(game_id, root_id, node_id, sim_iter)
        if ir is None:
            return
        if st.sim_time is None:
            st.sim_time = _to_float(ir.get("sim_time"), 0.0)
        if st.reward_incoming is None:
            st.reward_incoming = _to_float(ir.get("reward"), 0.0)
        if st.parent_id is None:
            st.parent_id = _to_int(ir.get("parent_node_id"))

    def _is_branching(
        self,
        ctx: RootContext,
        game_id: int,
        root_id: int,
        parent_id: int,
        sim_iter: int,
    ) -> bool:
        # First, use runtime-truth from iter log if available
        ir = self._iter_lookup_row(game_id, root_id, parent_id, sim_iter)
        if ir is not None:
            nva = _to_int(ir.get("num_valid_actions"))
            if nva is not None:
                return nva > 1

        # Fallback only when iter info is missing
        p = self._node(ctx, parent_id)
        return len(p.child_by_action) > 1

    # -------------------
    # Test 3: selection
    # -------------------
    def _verify_selection_trace(
        self,
        row: Dict[str, str],
        ctx: RootContext,
        game_id: int,
        root_id: int,
        sim_iter: int,
        pre_min: float,
        pre_max: float,
    ) -> None:
        trace = _json_load(row.get("selection_trace_json"), [])
        if not isinstance(trace, list):
            self._err(row, "selection_trace_json is not a list")

        for hop in trace:
            if not isinstance(hop, dict):
                self._err(row, "selection_trace hop is not dict")

            parent_id = _to_int(hop.get("parent_node_id"))
            parent_player = str(hop.get("parent_player") or "").strip()
            parent_visits_log = _to_int(hop.get("parent_visits"), 0)
            chosen_action = _to_int(hop.get("chosen_action_index"))
            chosen_child = _to_int(hop.get("chosen_child_node_id"))
            candidates = hop.get("candidates", [])

            if parent_id is None or chosen_action is None or chosen_child is None:
                self._err(row, "selection_trace missing required ids")
            if not isinstance(candidates, list) or len(candidates) == 0:
                self._err(row, "selection_trace candidates empty")

            p = self._node(ctx, parent_id)
            self._fill_node_fallbacks(ctx, game_id, root_id, parent_id, sim_iter)

            if parent_visits_log is not None and p.visits != int(parent_visits_log):
                self._err(
                    row,
                    f"parent visits mismatch in trace for parent={parent_id}: table={p.visits}, log={parent_visits_log}",
                )

            scored_actions: List[Tuple[int, float]] = []

            for c in candidates:
                aidx = _to_int(c.get("action_index"))
                cid = _to_int(c.get("child_node_id"))
                if aidx is None or cid is None:
                    self._err(row, "candidate missing action_index/child_node_id")

                child = self._node(ctx, cid)
                self._fill_node_fallbacks(ctx, game_id, root_id, cid, sim_iter)

                p.child_by_action.setdefault(aidx, cid)
                if aidx not in p.prior_by_action:
                    pb_c_log = _to_float(c.get("pb_c"), 0.0) or 0.0
                    prior_score_log = _to_float(c.get("prior_score"), 0.0) or 0.0
                    prior_est = prior_score_log / pb_c_log if abs(pb_c_log) > 1e-15 else 0.0
                    p.prior_by_action[aidx] = prior_est

                prior = float(p.prior_by_action.get(aidx, 0.0))

                pb_c = math.log((p.visits + self.pb_c_base + 1.0) / self.pb_c_base) + self.pb_c_init
                pb_c *= math.sqrt(p.visits + 1.0) / (child.visits + 1.0)

                prior_score = pb_c * prior

                if child.visits > 0:
                    parent_t = float(p.sim_time or 0.0)
                    child_t = float(child.sim_time or 0.0)
                    disc = self._discount(child_t, parent_t)

                    # parent_branching = self._is_branching(ctx, game_id, root_id, parent_id, sim_iter)
                    # reward_used = float(child.reward_incoming or 0.0) if parent_branching else 0.0

                    # q_controller = reward_used + disc * child.mean_value()

                    # For selection trace, branching is known from candidates at this hop
                    # parent_branching = len(candidates) > 1 --> this is not right as dedup can occur and collaspse multiple candidates into single child

                    ir_parent = self._iter_lookup_row(game_id, root_id, parent_id, sim_iter)
                    parent_num_valid_actions = _to_int(ir_parent.get("num_valid_actions")) if ir_parent is not None else None
                    if parent_num_valid_actions is None:
                        parent_num_valid_actions = 0
                    parent_branching = (parent_num_valid_actions > 1) or (len(candidates) > 1)


                    parent_t = float(p.sim_time or 0.0)
                    child_t = float(child.sim_time or 0.0)
                    disc = self._discount(child_t, parent_t) if parent_branching else 1.0

                    # mctsDNN ucb_score uses child.reward always in q_controller
                    reward_used = float(child.reward_incoming or 0.0)
                    q_controller = reward_used + disc * child.mean_value()
                    q_norm = self._normalize(q_controller, pre_min, pre_max)
                    value_score = q_norm if parent_player == "controller" else -q_norm



                    q_norm = self._normalize(q_controller, pre_min, pre_max)
                    value_score = q_norm if parent_player == "controller" else -q_norm
                else:
                    q_controller = 0.0
                    q_norm = 0.0
                    value_score = 0.0

                ucb = prior_score + value_score
                scored_actions.append((aidx, ucb))

                # Compare against logged per-candidate values
                for k, v_calc in [
                    ("pb_c", pb_c),
                    ("prior_score", prior_score),
                    ("value_score", value_score),
                    ("q_controller", q_controller),
                    ("q_norm", q_norm),
                    ("ucb", ucb),
                ]:
                    v_log = _to_float(c.get(k))
                    if v_log is None:
                        self._err(row, f"candidate missing {k} for action {aidx}")
                    if not _isclose(v_calc, v_log, abs_tol=self.tol, rel_tol=self.tol):
                        self._err(
                            row,
                            f"selection {k} mismatch parent={parent_id} action={aidx}: calc={v_calc}, log={v_log}",
                        )

            max_ucb = max(u for _, u in scored_actions)
            best_actions = {a for a, u in scored_actions if _isclose(u, max_ucb, abs_tol=self.tol, rel_tol=self.tol)}

            if chosen_action not in best_actions:
                self._err(
                    row,
                    f"chosen action not argmax parent={parent_id}: chosen={chosen_action}, best={sorted(best_actions)}",
                )

            mapped_child = p.child_by_action.get(chosen_action)
            if mapped_child is not None and int(mapped_child) != int(chosen_child):
                self._err(
                    row,
                    f"chosen child mismatch for parent={parent_id} action={chosen_action}: table={mapped_child}, log={chosen_child}",
                )

    # -------------------
    # Test 1 + Test 2
    # -------------------
    def _verify_backprop_and_minmax(
        self,
        row: Dict[str, str],
        ctx: RootContext,
        game_id: int,
        root_id: int,
        sim_iter: int,
    ) -> None:
        leaf_id = _to_int(row.get("node_id"))
        leaf_parent_id = _to_int(row.get("parent_node_id"))
        leaf_reward = _to_float(row.get("reward"), 0.0) or 0.0
        leaf_dnn_value = _to_float(row.get("node_dnn_value"))
        if leaf_id is None or leaf_dnn_value is None:
            self._err(row, "missing leaf_id or node_dnn_value")

        leaf = self._node(ctx, leaf_id)
        if leaf.parent_id is None:
            leaf.parent_id = leaf_parent_id
        leaf.reward_incoming = leaf_reward

        anc = _json_load(row.get("ancestor_chain_json"), [])
        if not isinstance(anc, list) or len(anc) == 0:
            self._err(row, "ancestor_chain_json empty or invalid")

        chain_ids: List[int] = []
        chain_obs: Dict[int, Dict[str, float]] = {}
        for i, e in enumerate(anc):
            if not isinstance(e, dict):
                self._err(row, "ancestor chain entry is not dict")
            nid = _to_int(e.get("node_id"))
            pid = _to_int(e.get("parent_node_id"))
            vs = _to_float(e.get("value_sum"))
            vis = _to_int(e.get("visits"))
            mv = _to_float(e.get("mean_value"))
            if nid is None or vs is None or vis is None or mv is None:
                self._err(row, f"bad ancestor entry at idx={i}: {e}")
            chain_ids.append(nid)
            chain_obs[nid] = {"value_sum": vs, "visits": vis, "mean_value": mv}
            nst = self._node(ctx, nid)
            if nst.parent_id is None:
                nst.parent_id = pid
            self._fill_node_fallbacks(ctx, game_id, root_id, nid, sim_iter)

        if chain_ids[0] != leaf_id:
            self._err(row, f"ancestor chain leaf mismatch: chain leaf={chain_ids[0]}, row leaf={leaf_id}")

        # Snapshot pre-state (before this row's backup)
        pre: Dict[int, Tuple[int, float]] = {}
        for nid in chain_ids:
            n = self._node(ctx, nid)
            pre[nid] = (int(n.visits), float(n.value_sum))

        pre_min = ctx.minmax_min
        pre_max = ctx.minmax_max

        # Compute expected updates exactly like backpropagate()
        expected: Dict[int, Tuple[int, float]] = {}
        minmax_candidates: List[float] = []
        v = float(leaf_dnn_value)

        for i, nid in enumerate(chain_ids):
            n = self._node(ctx, nid)
            pre_vis, pre_vs = pre[nid]

            new_vs = pre_vs + v
            new_vis = pre_vis + 1
            expected[nid] = (new_vis, new_vs)

            # Compare with logged ancestor values (test 1)
            obs = chain_obs[nid]
            if int(obs["visits"]) != int(new_vis):
                self._err(
                    row,
                    f"visits mismatch node={nid}: calc={new_vis}, log={int(obs['visits'])}",
                )
            if not _isclose(float(obs["value_sum"]), float(new_vs), abs_tol=self.tol, rel_tol=self.tol):
                self._err(
                    row,
                    f"value_sum mismatch node={nid}: calc={new_vs}, log={obs['value_sum']}",
                )

            calc_mean = new_vs / new_vis
            if not _isclose(float(obs["mean_value"]), float(calc_mean), abs_tol=self.tol, rel_tol=self.tol):
                self._err(
                    row,
                    f"mean_value mismatch node={nid}: calc={calc_mean}, log={obs['mean_value']}",
                )

            if i == len(chain_ids) - 1:
                break  # reached root

            parent_id = chain_ids[i + 1]
            parent = self._node(ctx, parent_id)

            child_t = float(n.sim_time or 0.0)
            parent_t = float(parent.sim_time or 0.0)
            disc = self._discount(child_t, parent_t)

            parent_branching = self._is_branching(ctx, game_id, root_id, parent_id, sim_iter)
            reward_used = float(n.reward_incoming or 0.0) if parent_branching else 0.0

            if parent_branching:
                child_mean_after = new_vs / new_vis
                mm_val = reward_used + disc * child_mean_after
                minmax_candidates.append(mm_val)

            v = reward_used + disc * v

        # Leaf increment-by-dnn sanity (requested check)
        leaf_pre_vis, leaf_pre_vs = pre[leaf_id]
        leaf_new_vis, leaf_new_vs = expected[leaf_id]
        if leaf_new_vis - leaf_pre_vis != 1:
            self._err(row, "leaf visits did not increment by 1")
        if not _isclose(leaf_new_vs - leaf_pre_vs, leaf_dnn_value, abs_tol=self.tol, rel_tol=self.tol):
            self._err(
                row,
                f"leaf increment mismatch: delta={leaf_new_vs - leaf_pre_vs}, dnn={leaf_dnn_value}",
            )

        self.test1_pass += 1

        # Test 2: MinMax update
        exp_min = pre_min
        exp_max = pre_max
        for x in minmax_candidates:
            if math.isinf(exp_min):
                exp_min = x
            else:
                exp_min = min(exp_min, x)
            if math.isinf(exp_max):
                exp_max = x
            else:
                exp_max = max(exp_max, x)

        row_min = _to_float(row.get("minmax_min"))
        row_max = _to_float(row.get("minmax_max"))
        if row_min is None or row_max is None:
            self._err(row, "row minmax_min/max missing")

        if not _isclose(exp_min, row_min, abs_tol=self.tol, rel_tol=self.tol):
            self._err(row, f"minmax_min mismatch: calc={exp_min}, log={row_min}")
        if not _isclose(exp_max, row_max, abs_tol=self.tol, rel_tol=self.tol):
            self._err(row, f"minmax_max mismatch: calc={exp_max}, log={row_max}")

        self.test2_pass += 1

        # Commit post-row state from log (authoritative)
        for nid in chain_ids:
            n = self._node(ctx, nid)
            n.visits = int(chain_obs[nid]["visits"])
            n.value_sum = float(chain_obs[nid]["value_sum"])

        ctx.minmax_min = row_min
        ctx.minmax_max = row_max

    # -------------------
    # Row processing
    # -------------------
    def _bootstrap_row_table(
        self,
        row: Dict[str, str],
        ctx: RootContext,
        game_id: int,
        root_id: int,
        sim_iter: int,
    ) -> None:
        leaf_id = _to_int(row.get("node_id"))
        parent_id = _to_int(row.get("parent_node_id"))
        reward = _to_float(row.get("reward"), 0.0) or 0.0
        children_created = _json_load(row.get("children_created_json"), [])
        dedup_children = _json_load(row.get("dedup_children_json"), [])
        selection_trace = _json_load(row.get("selection_trace_json"), [])

        if leaf_id is None:
            self._err(row, "missing node_id")

        leaf = self._node(ctx, leaf_id)
        if leaf.parent_id is None:
            leaf.parent_id = parent_id
        leaf.reward_incoming = reward
        self._fill_node_fallbacks(ctx, game_id, root_id, leaf_id, sim_iter)

        if isinstance(children_created, list):
            for item in children_created:
                if not isinstance(item, list) or len(item) < 3:
                    continue
                cid = _to_int(item[0])
                prior = _to_float(item[1], 0.0) or 0.0
                aidx = _to_int(item[2])
                if cid is None or aidx is None:
                    continue
                leaf.child_by_action[aidx] = cid
                leaf.prior_by_action[aidx] = prior
                child = self._node(ctx, cid)
                if child.parent_id is None:
                    child.parent_id = leaf_id

        if isinstance(dedup_children, list):
            for item in dedup_children:
                if not isinstance(item, list) or len(item) < 3:
                    continue
                canonical_id = _to_int(item[0])
                aliases = item[2]
                if canonical_id is None or not isinstance(aliases, list):
                    continue
                leaf.dedup_by_canonical[canonical_id] = [int(x) for x in aliases if _to_int(x) is not None]

        if isinstance(selection_trace, list):
            for hop in selection_trace:
                if not isinstance(hop, dict):
                    continue
                pid = _to_int(hop.get("parent_node_id"))
                if pid is None:
                    continue
                p = self._node(ctx, pid)
                self._fill_node_fallbacks(ctx, game_id, root_id, pid, sim_iter)
                cands = hop.get("candidates", [])
                if not isinstance(cands, list):
                    continue
                for c in cands:
                    if not isinstance(c, dict):
                        continue
                    aidx = _to_int(c.get("action_index"))
                    cid = _to_int(c.get("child_node_id"))
                    if aidx is None or cid is None:
                        continue
                    p.child_by_action[aidx] = cid
                    if aidx not in p.prior_by_action:
                        pb_c = _to_float(c.get("pb_c"), 0.0) or 0.0
                        prior_score = _to_float(c.get("prior_score"), 0.0) or 0.0
                        p.prior_by_action[aidx] = (prior_score / pb_c) if abs(pb_c) > 1e-15 else 0.0
                    child = self._node(ctx, cid)
                    if child.parent_id is None:
                        child.parent_id = pid

    def process(self) -> None:
        for row in self.rows:
            self.total_rows += 1

            game_id = _to_int(row.get("game_id"))
            root_id = _to_int(row.get("root_id"))
            sim_iter = _to_int(row.get("sim_iteration"))
            if game_id is None or root_id is None or sim_iter is None:
                self._err(row, "missing game_id/root_id/sim_iteration")

            ctx = self._ctx(game_id, root_id)

            # Build/expand table for this row first
            self._bootstrap_row_table(row, ctx, game_id, root_id, sim_iter)

            # Pre-row minmax for selection checks
            pre_min = ctx.minmax_min
            pre_max = ctx.minmax_max

            # Test 3: PUCT selection trace
            self._verify_selection_trace(row, ctx, game_id, root_id, sim_iter, pre_min, pre_max)
            self.test3_pass += 1

            # Test 1 + Test 2: backprop + minmax
            self._verify_backprop_and_minmax(row, ctx, game_id, root_id, sim_iter)

    def summary(self) -> str:
        return (
            f"[OK] validated rows={self.total_rows} "
            f"test1(backprop)={self.test1_pass} "
            f"test2(minmax)={self.test2_pass} "
            f"test3(selection)={self.test3_pass}"
        )


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puct-log", type=str, default=None, help="Path to mcts_iter_puct.csv")
    ap.add_argument("--iter-log", type=str, default=None, help="Path to mcts_iter.csv")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--step-time", type=float, default=DEFAULT_STEP_TIME)
    ap.add_argument("--pb-c-base", type=float, default=DEFAULT_PB_C_BASE)
    ap.add_argument("--pb-c-init", type=float, default=DEFAULT_PB_C_INIT)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    puct_path = Path(args.puct_log) if args.puct_log else _resolve_default_path("mcts_iter_puct.csv")
    iter_path = Path(args.iter_log) if args.iter_log else _resolve_default_path("mcts_iter.csv")

    if not puct_path.exists():
        raise SystemExit(f"PUCT log not found: {puct_path}")
    if not iter_path.exists():
        raise SystemExit(f"mcts_iter log not found: {iter_path}")

    puct_rows = _load_csv(puct_path)
    iter_rows = _load_csv(iter_path)

    if len(puct_rows) == 0:
        raise SystemExit(f"PUCT log has no rows: {puct_path}")

    v = PUCTValidator(
        puct_rows,
        iter_rows,
        gamma=args.gamma,
        step_time=args.step_time,
        pb_c_base=args.pb_c_base,
        pb_c_init=args.pb_c_init,
        tol=args.tol,
    )
    v.process()
    print(v.summary())


if __name__ == "__main__":
    main()
