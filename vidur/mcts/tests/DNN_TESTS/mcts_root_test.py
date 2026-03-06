"""
Validate mcts_root logs across generations.

Run:
    python3 -m vidur.mcts.tests.DNN_TESTS.mcts_root_test

Optional:
    python3 -m vidur.mcts.tests.DNN_TESTS.mcts_root_test \
      simulator_output/mcts_dnn_logs/gen_000010/mcts_root_p*.csv
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


class RootLogFailure(AssertionError):
    pass


RELAX_TAIL_PENDING = True
TAIL_PREFILL_GRACE_ROWS = 2


@dataclass
class RootRow:
    rownum: int
    file_path: str
    game_id: int
    root_id: int
    root_depth: int
    root_player: str
    best_action_json: str
    best_action_repr: str
    phase: str
    sim_time: float


@dataclass
class RequestState:
    request_id: int
    prefill_initial: int
    decode_initial: int
    prefill_remaining: int
    decode_remaining: int
    decode_seen: bool = False
    stopped: bool = False


@dataclass
class BatchState:
    request_ids: List[int]
    created_rownum: int
    seen_in_first_controller: bool = False
    prefill_seen: Set[int] = field(default_factory=set)


def _safe_int(x: object, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _parse_json_dict(s: str) -> Dict[str, object]:
    try:
        val = json.loads((s or "").strip())
    except Exception:
        return {}
    return val if isinstance(val, dict) else {}


def _parse_int_map(d: Dict[str, object], key: str) -> Dict[int, int]:
    raw = d.get(key)
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in raw.items():
        out[_safe_int(k, -1)] = _safe_int(v, 0)
    out.pop(-1, None)
    return out


def _parse_int_list(d: Dict[str, object], key: str) -> List[int]:
    raw = d.get(key)
    if not isinstance(raw, list):
        return []
    return [_safe_int(x, -1) for x in raw if _safe_int(x, -1) >= 0]


def _fail(row: RootRow, trace_id: str, msg: str) -> None:
    raise RootLogFailure(
        f"[mcts_root_test] trace={trace_id} file={row.file_path} rownum={row.rownum} "
        f"game_id={row.game_id} root_id={row.root_id} depth={row.root_depth} "
        f"player={row.root_player} sim_time={row.sim_time:.6f}\n"
        f"  action_repr={row.best_action_repr}\n"
        f"  ERROR: {msg}"
    )


def _resolve_log_paths(args: Sequence[str]) -> List[str]:
    paths: List[str] = []

    if args:
        for a in args:
            m = glob.glob(a)
            if m:
                paths.extend(m)
            elif os.path.exists(a):
                paths.append(a)
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No files matched args: {list(args)}")
        return paths

    patterns = [
        "simulator_output/mcts_dnn_logs/gen_*/mcts_root_p*.csv",
        "simulator_output/mcts_dnn_logs/gen_*/mcts_root.csv",
        "vidur/simulator_output/mcts_dnn_logs/gen_*/mcts_root_p*.csv",
        "vidur/simulator_output/mcts_dnn_logs/gen_*/mcts_root.csv",
    ]
    for pat in patterns:
        paths.extend(glob.glob(pat))

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No mcts_root logs found under simulator_output/mcts_dnn_logs/gen_*")
    return paths


def _load_rows(path: str) -> List[RootRow]:
    out: List[RootRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, raw in enumerate(r, start=2):
            action_json = str(raw.get("best_action_json", "") or "").strip()
            if not action_json:
                continue
            out.append(
                RootRow(
                    rownum=i,
                    file_path=path,
                    game_id=_safe_int(raw.get("game_id"), 0),
                    root_id=_safe_int(raw.get("root_id"), 0),
                    root_depth=_safe_int(raw.get("root_depth"), 0),
                    root_player=str(raw.get("root_player", "") or "").strip().lower(),
                    best_action_json=action_json,
                    best_action_repr=str(raw.get("best_action_repr", "") or "").strip(),
                    phase=str(raw.get("phase", "") or "").strip(),
                    sim_time=_safe_float(raw.get("sim_time"), 0.0),
                )
            )
    return out


def _effective_rows_one_per_root(rows: List[RootRow]) -> List[RootRow]:
    """
    Use one action row per root key:
      (game_id, root_id, root_depth, root_player)

    For each key:
      - prefer phase == train_root_applied (actual executed action),
      - else fallback to last row for that key.
    """
    by_key: Dict[tuple[int, int, int, str], List[RootRow]] = {}
    for r in rows:
        by_key.setdefault((r.game_id, r.root_id, r.root_depth, r.root_player), []).append(r)

    effective: List[RootRow] = []
    for key, group in by_key.items():
        group_sorted = sorted(group, key=lambda x: x.rownum)
        applied = [r for r in group_sorted if str(r.phase).strip().lower() == "train_root_applied"]
        pick = applied[-1] if applied else group_sorted[-1]
        effective.append(pick)

    effective.sort(key=lambda x: x.rownum)
    return effective


def _group_traces(rows: List[RootRow]) -> List[List[RootRow]]:
    # Keep file order, and split traces when we see obvious resets.
    traces: List[List[RootRow]] = []
    cur: List[RootRow] = []
    prev: Optional[RootRow] = None

    for row in rows:
        start_new = False
        if prev is None:
            start_new = True
        elif row.game_id != prev.game_id:
            start_new = True
        elif row.root_depth == 0 and prev.root_depth > 0:
            start_new = True
        elif row.root_id < prev.root_id:
            start_new = True

        if start_new:
            if cur:
                traces.append(cur)
            cur = [row]
        else:
            cur.append(row)
        prev = row

    if cur:
        traces.append(cur)

    return traces


def _pending_prefill_ids(active: Dict[int, RequestState]) -> List[int]:
    return sorted(rid for rid, req in active.items() if req.prefill_remaining > 0)


def _apply_trace_checks(trace_rows: List[RootRow], *, trace_id: str) -> Dict[str, int]:
    next_request_id = 0
    active: Dict[int, RequestState] = {}
    all_requests: Dict[int, RequestState] = {}
    open_batch: Optional[BatchState] = None
    total_adversary_rows = 0
    total_controller_rows = 0
    last_prefill_rownum: Dict[int, int] = {}

    for row in trace_rows:
        action = _parse_json_dict(row.best_action_json)
        action_type = str(action.get("type", "")).strip().lower()

        if row.root_player == "adversary":
            total_adversary_rows += 1
            if action_type != "adversary":
                _fail(row, trace_id, f"root_player=adversary but action.type={action_type!r}")

            requests = action.get("requests")
            if not isinstance(requests, list):
                _fail(row, trace_id, "adversary action missing requests list")

            if requests:
                pending = _pending_prefill_ids(active)
                if pending:
                    _fail(
                        row,
                        trace_id,
                        f"adversary generated new requests while prefill still pending for request_ids={pending}",
                    )

                if open_batch is not None and _pending_prefill_ids(
                    {rid: all_requests[rid] for rid in open_batch.request_ids if rid in all_requests}
                ):
                    _fail(row, trace_id, "new adversary batch started before previous batch prefill finished")

                new_ids: List[int] = []
                for req in requests:
                    if not isinstance(req, dict):
                        _fail(row, trace_id, "adversary request entry is not a JSON object")
                    prefill = _safe_int(req.get("prefill_tokens"), 0)
                    decode = _safe_int(req.get("decode_tokens"), 0)
                    rid = next_request_id
                    next_request_id += 1
                    new_ids.append(rid)
                    st = RequestState(
                        request_id=rid,
                        prefill_initial=prefill,
                        decode_initial=decode,
                        prefill_remaining=prefill,
                        decode_remaining=decode,
                    )
                    active[rid] = st
                    all_requests[rid] = st

                # rule 2: no missing ids in this trace
                expected = list(range(new_ids[0], new_ids[0] + len(new_ids))) if new_ids else []
                if new_ids != expected:
                    _fail(row, trace_id, f"missing request ids in generated batch: got={new_ids}, expected={expected}")

                open_batch = BatchState(request_ids=list(new_ids), created_rownum=row.rownum)

            stop_decode_ids = action.get("stop_decode_ids")
            if isinstance(stop_decode_ids, list):
                for rid_raw in stop_decode_ids:
                    rid = _safe_int(rid_raw, -1)
                    if rid < 0:
                        continue
                    req = all_requests.get(rid)
                    if req is not None:
                        req.stopped = True
                    active.pop(rid, None)

            continue

        if row.root_player == "controller":
            total_controller_rows += 1
            if action_type != "controller":
                _fail(row, trace_id, f"root_player=controller but action.type={action_type!r}")

            prefill_alloc = _parse_int_map(action, "prefill_allocations")
            decode_alloc = _parse_int_map(action, "decode_allocations")
            selected_ids = set(_parse_int_list(action, "selected_request_ids"))

            touched_ids = set(prefill_alloc.keys()) | set(decode_alloc.keys())
            for rid in touched_ids:
                if rid not in all_requests:
                    _fail(row, trace_id, f"controller action references unknown request_id={rid}")

            if open_batch is not None and not open_batch.seen_in_first_controller:
                if selected_ids or prefill_alloc:
                    appears = any((rid in selected_ids) or (rid in prefill_alloc) for rid in open_batch.request_ids)
                    if not appears:
                        _fail(
                            row,
                            trace_id,
                            "new adversary requests did not appear in the next controller action",
                        )
                open_batch.seen_in_first_controller = True

            for rid, tok in prefill_alloc.items():
                req = all_requests[rid]
                if tok <= 0:
                    continue
                last_prefill_rownum[rid] = row.rownum
                if req.prefill_remaining <= 0:
                    _fail(row, trace_id, f"prefill allocated to request_id={rid} but prefill already complete")
                req.prefill_remaining -= tok
                if req.prefill_remaining < 0:
                    _fail(
                        row,
                        trace_id,
                        f"prefill over-allocation for request_id={rid}: remaining became {req.prefill_remaining}",
                    )
                if open_batch is not None and rid in open_batch.request_ids:
                    open_batch.prefill_seen.add(rid)

            for rid, tok in decode_alloc.items():
                req = all_requests[rid]
                if tok <= 0:
                    continue
                if req.prefill_remaining > 0:
                    _fail(
                        row,
                        trace_id,
                        f"decode allocated before prefill complete for request_id={rid} prefill_remaining={req.prefill_remaining}",
                    )
                req.decode_seen = True
                req.decode_remaining -= tok
                if req.decode_remaining <= 0:
                    active.pop(rid, None)

            # if active batch prefill finished, ensure each generated request was served in prefill
            if open_batch is not None:
                still_pending = [rid for rid in open_batch.request_ids if all_requests[rid].prefill_remaining > 0]
                if not still_pending:
                    missing_prefill = [rid for rid in open_batch.request_ids if rid not in open_batch.prefill_seen]
                    if missing_prefill:
                        _fail(
                            row,
                            trace_id,
                            "adversary-generated requests finished without ever appearing in prefill allocations: "
                            f"{missing_prefill}",
                        )
                    open_batch = None

            continue

        _fail(row, trace_id, f"unknown root_player={row.root_player!r}")

    # Final checks after replaying one trace
    final_pending = _pending_prefill_ids(active)
    tail_pending_set = set(final_pending)
    last_rownum = trace_rows[-1].rownum
    if RELAX_TAIL_PENDING:
        tail_prefill_ids = {
            rid
            for rid, rownum in last_prefill_rownum.items()
            if rownum >= (last_rownum - max(0, int(TAIL_PREFILL_GRACE_ROWS)))
        }
        tail_pending_set.update(tail_prefill_ids)
    tail_open_batch = 1 if open_batch is not None else 0
    if final_pending and (not RELAX_TAIL_PENDING):
        last = trace_rows[-1]
        _fail(last, trace_id, f"trace ended with unfinished prefill requests: {final_pending}")

    if open_batch is not None and (not RELAX_TAIL_PENDING):
        last = trace_rows[-1]
        _fail(last, trace_id, "trace ended before current adversary batch fully completed prefill")

    # rule 1: every generated request should reach decode stage eventually (unless stopped)
    for rid, req in sorted(all_requests.items()):
        if req.prefill_remaining != 0:
            if RELAX_TAIL_PENDING and rid in tail_pending_set:
                continue
            last = trace_rows[-1]
            _fail(last, trace_id, f"request_id={rid} never completed prefill stage")
        if (not req.stopped) and req.decode_initial > 0 and (not req.decode_seen):
            if RELAX_TAIL_PENDING and rid in tail_pending_set:
                continue
            last = trace_rows[-1]
            _fail(last, trace_id, f"request_id={rid} never appeared in decode allocations")

    return {
        "rows": len(trace_rows),
        "requests": len(all_requests),
        "adversary_rows": total_adversary_rows,
        "controller_rows": total_controller_rows,
        "tail_pending": len(final_pending),
        "tail_open_batch": int(tail_open_batch),
    }


def main() -> None:
    log_paths = _resolve_log_paths(sys.argv[1:])
    total_files = 0
    total_traces = 0
    total_rows = 0
    total_requests = 0
    total_tail_pending = 0
    total_tail_open_batch = 0

    try:
        for path in log_paths:
            rows = _load_rows(path)
            if not rows:
                continue
            rows_eff = _effective_rows_one_per_root(rows)
            traces = _group_traces(rows_eff)
            print(f"\n=== Testing {path} ===")
            print(f"Loaded rows={len(rows)} effective_rows={len(rows_eff)} traces={len(traces)}")

            file_requests = 0
            file_rows = 0
            file_tail_pending = 0
            file_tail_open_batch = 0
            for tr in traces:
                trace_id = f"game={tr[0].game_id}"
                stats = _apply_trace_checks(tr, trace_id=trace_id)
                file_requests += stats["requests"]
                file_rows += stats["rows"]
                file_tail_pending += int(stats.get("tail_pending", 0))
                file_tail_open_batch += int(stats.get("tail_open_batch", 0))

            print(
                f"✅ Passed {path}: trace_rows={file_rows} requests={file_requests} "
                f"tail_pending={file_tail_pending} tail_open_batch={file_tail_open_batch}"
            )
            total_files += 1
            total_traces += len(traces)
            total_rows += file_rows
            total_requests += file_requests
            total_tail_pending += file_tail_pending
            total_tail_open_batch += file_tail_open_batch

    except RootLogFailure as e:
        print("\n❌ mcts_root log test failed:\n")
        print(str(e))
        sys.exit(1)

    if total_files == 0:
        raise SystemExit("No non-empty mcts_root files found to test.")

    print(
        f"\n✅ All root logs passed. files={total_files} traces={total_traces} "
        f"rows={total_rows} requests={total_requests} "
        f"tail_pending={total_tail_pending} tail_open_batch={total_tail_open_batch}"
    )


if __name__ == "__main__":
    main()
