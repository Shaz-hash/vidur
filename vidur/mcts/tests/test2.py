# vidur/vidur/mcts/tests/test2.py

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import json
import math

from vidur.mcts.run_mcts import main as run_mcts_main
from vidur.mcts.prefill_calibrator import PrefillProfile

TEST_OUTPUT_ROOT = Path("simulator_output/Test_Parrallel_Launch")
ITERATIONS = 5000
HISTORY_DEPTH = 0

# MCTS constraints used for this test (TODO : We will extract them directly from the config)
MCTS_MAX_QPS = 10
MCTS_STEP = 512
MCTS_MIN_TOKENS = 512
MCTS_MAX_TOKENS = 3072

# SLO Related metrics/constraints for the test (TODO : We will extract them directly from the config)
PREFILL_PROFILE_PATH = Path("simulator_output/prefill_profile.csv")
PREFILL_SLO_MULT = 3.0          # from --mcts_prefill_slos 3.0
DECODE_SLO_MS = 50              # from --mcts_decode_slos 50
DECODE_SLO_SEC = DECODE_SLO_MS / 1000.0
SLO_EPS = 1e-6
PREF_PROFILE = PrefillProfile.load(PREFILL_PROFILE_PATH)



RUN_ID = "T1"


def run_single_mcts() -> Path:
    """
    Run a single MCTS job with fixed config and return the path
    to the main MCTS CSV (test_mcts_trace_*.csv) that was produced.
    """
    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    base_csv = TEST_OUTPUT_ROOT / "test_mcts_trace.csv"

    args = [
        "--mcts_run_id",
        RUN_ID,
        "--mcts_history_depth",
        str(HISTORY_DEPTH),
        "--mcts_iterations",
        str(ITERATIONS),
        "--mcts_simulation_random_tries",
        "1",
        "--mcts_simulation_depth",
        "2",
        "--mcts_interval_request_size",
        "512",
        "--mcts_maximum_qps",
        "10",
        "--mcts_min_request_tokens",
        "512",
        "--mcts_exploration_constant",
        "1.7",
        "--mcts_max_branching",
        "10",
        "--mcts_max_request_tokens",
        "3072",
        "--mcts_prefill_profile",
        "simulator_output/prefill_profile.csv",
        "--mcts_prefill_slos",
        "3.0",
        "--mcts_decode_slos",
        "50",
        "--mcts_log_csv",
        str(base_csv),
        "--replica_config_model_name",
        "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device",
        "a100",
        "--replica_config_network_device",
        "a100_dgx",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_tensor_parallel_size",
        "1",
        "--replica_config_num_pipeline_stages",
        "1",
        "--global_scheduler_config_type",
        "round_robin",
        "--replica_scheduler_config_type",
        "vllm_v1",
    ]

    t0 = time.perf_counter()
    run_mcts_main(args)
    t1 = time.perf_counter()

    total = t1 - t0
    avg = total / ITERATIONS
    print(f"[MCTS] Total time for {ITERATIONS} iterations: {total:.3f} s")
    print(f"[MCTS] Average time per iteration: {avg:.6f} s")

    # run_mcts suffixes the log with _<run_id>
    log_path = base_csv.with_name(base_csv.stem + f"_{RUN_ID}" + base_csv.suffix)
    print(f"[MCTS] Log CSV: {log_path}")
    return log_path


def load_log_rows(log_path: Path) -> List[Dict[str, Any]]:
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_traces_from_log(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Build all root→leaf traces using parent_node_id / node_id from the main
    MCTS CSV. For each trace, slice from the first INITIAL_HISTORY_GEN node
    if present, otherwise from the root node.

    Handles the special case where the root is logged with node_id="root"
    but its children use parent_node_id="0".
    """
    keep_phases = {"root", "INITIAL_HISTORY_GEN", "tree"}

    def norm_id(x: Any) -> str:
        """Normalize node ids so 'root' and 0 both map to '0'."""
        if x is None:
            return ""
        s = str(x)
        if not s:
            return ""
        if s == "root":
            return "0"
        return s

    nodes: Dict[str, Dict[str, Any]] = {}
    children: Dict[str, List[str]] = {}
    roots: List[str] = []

    for r in rows:
        phase = r.get("phase", "")
        if phase not in keep_phases:
            continue
        raw_nid = r.get("node_id", "")
        nid = norm_id(raw_nid)
        if not nid:
            continue
        nodes[nid] = r

        raw_parent = r.get("parent_node_id", "")
        parent = norm_id(raw_parent)
        if parent:
            children.setdefault(parent, []).append(nid)
        else:
            roots.append(nid)

    # Choose the root node: prefer phase == "root"
    root_id = None
    for nid in roots:
        if nodes[nid].get("phase") == "root":
            root_id = nid
            break
    if root_id is None:
        if not roots:
            return []
        root_id = roots[0]

    traces_ids: List[List[str]] = []

    def dfs(nid: str, path: List[str]) -> None:
        path.append(nid)
        if nid not in children or not children[nid]:
            traces_ids.append(list(path))
        else:
            for child in children[nid]:
                dfs(child, path)
        path.pop()

    dfs(root_id, [])

    # Slice each trace from the first INITIAL_HISTORY_GEN node if present
    traces: List[List[Dict[str, Any]]] = []
    for path in traces_ids:
        start_idx = 0
        for i, nid in enumerate(path):
            if nodes[nid].get("phase") == "INITIAL_HISTORY_GEN":
                start_idx = i
                break
        sliced_ids = path[start_idx:]
        traces.append([nodes[nid] for nid in sliced_ids])

    return traces

def print_random_trace(traces: List[List[Dict[str, Any]]]) -> None:
    if not traces:
        print("[TRACE] No traces found.")
        return
    trace = random.choice(traces)
    print(f"[TRACE] Selected trace with {len(trace)} nodes:")
    for r in trace:
        print(
            f"  depth={r.get('depth')} "
            f"node_id={r.get('node_id')} "
            f"parent={r.get('parent_node_id')} "
            f"phase={r.get('phase')} "
            f"player={r.get('player_to_act')} "
            f"next={r.get('next_player')}"
        )

"""
    ________TEST CASES________ 
 1- Requests Related Test Cases in the Trace 
"""

def _parse_int_map(raw: str) -> Dict[int, int]:
    """Parse a JSON object mapping request_id -> int, normalizing keys to int."""
    if not raw or raw == "{}":
        return {}
    data = json.loads(raw)
    out: Dict[int, int] = {}
    for k, v in data.items():
        out[int(k)] = int(v)
    return out

def _parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(x) for x in json.loads(raw)]

def validate_trace(trace: List[Dict[str, Any]], trace_index: int) -> None:
    """
    Walk a single root→leaf trace and enforce adversary/controller invariants.

    - Maintain:
        prefill_remaining: request_id -> remaining prefill tokens
        decode_ready: set of request_ids whose prefill is complete
    - For each adversary row:
        * Adv Req Ids Uniqueness
        * Adv QPS Check
        * Adv Request Size Check
    - For each controller row:
        * Cont Total Size Check
        * Cont Request Size Check
        * Cont Prefill Request Check
        * Cont Decode Request Check
    """

    prefill_remaining: Dict[int, int] = {}
    decode_ready: set[int] = set()

    def known_ids() -> set[int]:
        return set(prefill_remaining.keys()) | decode_ready

    for idx, row in enumerate(trace):
        phase = row.get("phase", "")
        player = row.get("player_to_act", "")

        # We only care about structural phases
        if phase not in {"INITIAL_HISTORY_GEN", "tree"}:
            continue

        # Current waiting and completed IDs in the simulator snapshot at this node
        waiting_ids = set(_parse_int_list(row.get("state_waiting_ids", "[]")))
        # completed_ids = set(_parse_int_list(row.get("state_completed_request_ids", "[]")))

        # --- Adversary actions ------------------------------------------------
        if player == "adversary":
            raw_adv = row.get("adversary_requests", "[]")
            specs = json.loads(raw_adv) if raw_adv and raw_adv != "[]" else []
            if not specs:
                continue  # no new requests from this adversary step

            # New request IDs created by this adversary step
            new_ids = sorted(waiting_ids - known_ids())
            if len(new_ids) != len(specs):
                raise AssertionError(
                    f"[Trace {trace_index}, row {idx}] "
                    f"Adv Req Ids Uniquness: expected {len(specs)} new ids, "
                    f"found {len(new_ids)}. "
                    f"new_ids={new_ids}"
                )

            # Adv QPS Check
            if len(specs) > MCTS_MAX_QPS:
                raise AssertionError(
                    f"[Trace {trace_index}, row {idx}] Adv QPS Check failed: "
                    f"{len(specs)} > max_qps={MCTS_MAX_QPS}"
                )

            # Adv Request Size Check + initialize prefill_remaining
            for rid, spec in zip(new_ids, specs):
                prefill = int(spec.get("prefill_tokens", 0))
                if prefill < MCTS_MIN_TOKENS or prefill > MCTS_MAX_TOKENS:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Adv Request Size Check failed "
                        f"for request {rid}: prefill_tokens={prefill} "
                        f"not in [{MCTS_MIN_TOKENS}, {MCTS_MAX_TOKENS}]"
                    )
                if prefill % MCTS_STEP != 0:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Adv Request Size Check failed "
                        f"for request {rid}: prefill_tokens={prefill} "
                        f"not multiple of step={MCTS_STEP}"
                    )
                if rid in prefill_remaining or rid in decode_ready:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Adv Req Ids Uniquness failed: "
                        f"request {rid} already present"
                    )
                prefill_remaining[rid] = prefill

            continue  # done with adversary row

        # --- Controller actions ----------------------------------------------
        if player == "controller":
            # Parse controller fields
            token_budget_raw = row.get("controller_token_budget", "") or "0"
            try:
                token_budget = int(token_budget_raw)
            except ValueError:
                token_budget = 0

            pre_alloc = _parse_int_map(row.get("controller_prefill_allocations", "{}"))
            dec_alloc = _parse_int_map(row.get("controller_decode_allocations", "{}"))

            pre_total = sum(pre_alloc.values())
            dec_total = sum(dec_alloc.values())
            total_alloc = pre_total + dec_total

            # Skip pure-noop controller steps
            if total_alloc == 0 and token_budget == 0:
                continue

            # Cont Total Size Check: prefill combined within [min,max] and multiple of step
            if pre_total > 0:
                if pre_total < MCTS_MIN_TOKENS or pre_total > MCTS_MAX_TOKENS:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Total Size Check failed: "
                        f"prefill_total={pre_total} not in "
                        f"[{MCTS_MIN_TOKENS}, {MCTS_MAX_TOKENS}]"
                    )
                if pre_total % MCTS_STEP != 0:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Total Size Check failed: "
                        f"prefill_total={pre_total} not multiple of step={MCTS_STEP}"
                    )

            # Optionally: check token_budget consistency
            if token_budget > 0 and total_alloc > token_budget:
                raise AssertionError(
                    f"[Trace {trace_index}, row {idx}] Cont Total Size Check failed: "
                    f"total_alloc={total_alloc} > token_budget={token_budget}"
                )

            # Cont Request Size Check: each prefill allocation > 0 must be in [step,max] & multiple of step
            for rid, alloc in pre_alloc.items():
                if alloc <= 0:
                    continue
                if alloc < MCTS_STEP or alloc > MCTS_MAX_TOKENS:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Request Size Check failed "
                        f"for request {rid}: alloc={alloc}"
                    )
                if alloc % MCTS_STEP != 0:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Request Size Check failed "
                        f"for request {rid}: alloc={alloc} not multiple of step={MCTS_STEP}"
                    )

            # Cont Prefill Request Check:
            # - must allocate only to requests in prefill_remaining
            # - allocation cannot exceed remaining prefill tokens
            for rid, alloc in pre_alloc.items():
                if alloc <= 0:
                    continue
                if rid not in prefill_remaining:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Prefill Request Check failed: "
                        f"allocating prefill to unknown request {rid}"
                    )
                remaining = prefill_remaining[rid]
                if alloc > remaining:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Prefill Request Check failed: "
                        f"alloc={alloc} exceeds remaining={remaining} for request {rid}"
                    )

            # If all good, apply prefill allocations to state
            for rid, alloc in pre_alloc.items():
                if alloc <= 0:
                    continue
                prefill_remaining[rid] -= alloc
                if prefill_remaining[rid] == 0:
                    # move to decode_ready
                    decode_ready.add(rid)
                    del prefill_remaining[rid]

            # Cont Decode Request Check:
            # - every decode allocation must be for a request in decode_ready
            # - each decode allocation must be exactly 1
            for rid, alloc in dec_alloc.items():
                if rid not in decode_ready:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Decode Request Check failed: "
                        f"decode alloc for request {rid} which is not decode-ready"
                    )
                if alloc != 1:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Decode Request Check failed: "
                        f"decode alloc for request {rid} is {alloc}, expected 1"
                    )

            continue

    # If we reach here, all checks in this trace passed
    return

"""
 2- SLO correctness in the Trace 
"""

def validate_trace_slos(trace: List[Dict[str, Any]], trace_index: int) -> None:
    """
    Validate SLO semantics along a single root→leaf trace.

    - SLO Value Correctness:
        * Prefill SLO == prefill_profile.lookup(prefill_tokens) * PREFILL_SLO_MULT
        * Decode SLO == DECODE_SLO_SEC
    - SLO Cost Correctness:
        * Maintain per-request prefill/decode deadlines and a violation flag.
        * After each controller action, recompute how many requests have (ever) violated;
          this must equal `slo_violations` in the log.
    - Cont Simulator Time Correctness:
        * Controller sim_time increases vs previous log.
        * If there is any prefill allocation, Δt >= prefill_profile.lookup(max_prefill_alloc).
        * If no prefill allocation, Δt > 0.
    - Adv Simulator Time Correctness:
        * Adversary sim_time is unchanged vs previous log.
    """

    # Per-request state
    prefill_deadline: Dict[int, float] = {}   # rid -> abs time for prefill SLO
    prefill_tokens_left: Dict[int, int] = {}  # rid -> remaining prefill tokens
    decode_slo: Dict[int, float] = {}         # rid -> decode_slo (seconds)
    decode_deadline: Dict[int, float] = {}    # rid -> abs time for decode SLO (once decode-ready)
    decode_ready: set[int] = set()            # rid whose prefill is complete
    slo_violated: Dict[int, bool] = {}        # rid -> has ever violated prefill or decode SLO?
    # NEW: decode tracking
    decode_tokens_left: Dict[int, int] = {}   # rid -> remaining decode tokens
    decode_deadline: Dict[int, float] = {}    # rid -> abs time for decode SLO

    def _parse_int_list(raw: str) -> List[int]:
        if not raw:
            return []
        return [int(x) for x in json.loads(raw)]

    def _parse_int_map(raw: str) -> Dict[int, int]:
        if not raw or raw == "{}":
            return {}
        data = json.loads(raw)
        out: Dict[int, int] = {}
        for k, v in data.items():
            out[int(k)] = int(v)
        return out

    def known_ids() -> set[int]:
        return set(prefill_tokens_left.keys()) | decode_ready | set(slo_violated.keys())

    prev_sim_time: float | None = None

    for idx, row in enumerate(trace):
        phase = row.get("phase", "")
        player = row.get("player_to_act", "")
        sim_time = float(row.get("sim_time", "0.0") or 0.0)

        # Time correctness vs previous log
        if prev_sim_time is not None:
            if player == "adversary":
                # Adv Simulator Time Correctness: no time advance
                if abs(sim_time - prev_sim_time) > SLO_EPS:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Adv Simulator Time Correctness failed: "
                        f"sim_time changed {prev_sim_time:.6f} -> {sim_time:.6f}"
                    )
            elif player == "controller":
                # Cont Simulator Time Correctness: must advance
                if sim_time <= prev_sim_time + SLO_EPS:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] Cont Simulator Time Correctness failed: "
                        f"sim_time did not increase ({prev_sim_time:.6f} -> {sim_time:.6f})"
                    )
        prev_sim_time = sim_time

        if phase not in {"INITIAL_HISTORY_GEN", "tree"}:
            continue

        waiting_ids = set(_parse_int_list(row.get("state_waiting_ids", "[]")))

        # ---------------- Adversary actions: SLO Value Correctness ----------------
        if player == "adversary":
            raw_adv = row.get("adversary_requests", "[]")
            specs = json.loads(raw_adv) if raw_adv and raw_adv != "[]" else []
            if not specs:
                continue

            # Discover new ids from waiting set
            new_ids = sorted(waiting_ids - known_ids())
            if len(new_ids) != len(specs):
                raise AssertionError(
                    f"[Trace {trace_index}, row {idx}] SLO Value Correctness: "
                    f"expected {len(specs)} new request_ids from adversary, got {len(new_ids)}"
                )

            arrival_time = math.floor(sim_time)

            for rid, spec in zip(new_ids, specs):
                prefill_tokens = int(spec.get("prefill_tokens", 0))
                prefill_slo_actual = float(spec.get("prefill_slo", 0.0))
                decode_slo_actual = float(spec.get("decode_slo", 0.0))
                decode_tokens = int(spec.get("decode_tokens", 0))

                # Prefill SLO Value Correctness
                base_pref_t = PREF_PROFILE.lookup(prefill_tokens)
                prefill_slo_expected = base_pref_t * PREFILL_SLO_MULT
                if abs(prefill_slo_actual - prefill_slo_expected) > 1e-5 * max(
                    1.0, abs(prefill_slo_expected)
                ):
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] SLO Value Correctness failed for request {rid}: "
                        f"prefill_slo={prefill_slo_actual:.6f}, expected≈{prefill_slo_expected:.6f}"
                    )

                # Decode SLO Value Correctness
                if abs(decode_slo_actual - DECODE_SLO_SEC) > 1e-6:
                    raise AssertionError(
                        f"[Trace {trace_index}, row {idx}] SLO Value Correctness failed for request {rid}: "
                        f"decode_slo={decode_slo_actual:.6f}, expected={DECODE_SLO_SEC:.6f}"
                    )

                # Initialize SLO tracking
                prefill_deadline[rid] = arrival_time + prefill_slo_actual
                prefill_tokens_left[rid] = prefill_tokens
                decode_slo[rid] = decode_slo_actual
                decode_tokens_left[rid] = decode_tokens
                slo_violated[rid] = False

            continue  # done with adversary row

        # ---------------- Controller actions: SLO Cost / Time Correctness --------
        if player == "controller":
            # Parse allocations
            pre_alloc = _parse_int_map(row.get("controller_prefill_allocations", "{}"))
            dec_alloc = _parse_int_map(row.get("controller_decode_allocations", "{}"))

            # Cont Simulator Time Correctness vs prefill allocations
            # (we know prev_sim_time is not None for controller rows)
            # note: prev_sim_time currently holds *current* sim_time, so recompute delta manually
            # using the previous row's sim_time from trace.
            prev_row_sim_time = float(
                trace[idx - 1].get("sim_time", "0.0") if idx > 0 else sim_time
            )
            delta_t = sim_time - prev_row_sim_time

            if pre_alloc:
                max_pref_alloc = max(pre_alloc.values())
                if max_pref_alloc > 0:
                    min_dt = PREF_PROFILE.lookup(max_pref_alloc)
                    if delta_t + 1e-9 < min_dt:  # small epsilon
                        raise AssertionError(
                            f"[Trace {trace_index}, row {idx}] Cont Simulator Time Correctness failed: "
                            f"Δt={delta_t:.6f} < profile(min_prefill)={min_dt:.6f} "
                            f"for max_prefill_alloc={max_pref_alloc}"
                        )
            else:
                # No prefill allocation: we already checked Δt>0 via prev_sim_time logic
                pass

            # 1) Prefill SLO violations: check all requests still in prefill
            for rid, deadline in list(prefill_deadline.items()):
                if rid in prefill_tokens_left:
                    if not slo_violated.get(rid, False) and sim_time > deadline + SLO_EPS:
                        slo_violated[rid] = True
                        # once prefill SLO is violated, we no longer need this deadline
                        prefill_deadline.pop(rid, None)

            # 2) Apply prefill allocations; when prefill completes, start decode SLO timer
            for rid, alloc in pre_alloc.items():
                if alloc <= 0:
                    continue
                if rid in prefill_tokens_left:
                    prefill_tokens_left[rid] -= alloc
                    if prefill_tokens_left[rid] <= 0:
                        # Prefill done -> move to decode phase and set decode deadline
                        prefill_tokens_left.pop(rid, None)
                        prefill_deadline.pop(rid, None)
                        if decode_tokens_left.get(rid, 0) > 0:
                            # decode SLO starts at prefill completion
                            decode_deadline[rid] = sim_time + decode_slo.get(rid, 0.0)

            # 3) Apply decode allocations; when decode completes, drop decode deadline
            for rid, alloc in dec_alloc.items():
                if alloc <= 0:
                    continue
                if rid in decode_tokens_left:
                    decode_tokens_left[rid] -= alloc
                    if decode_tokens_left[rid] <= 0:
                        decode_tokens_left.pop(rid, None)
                        decode_deadline.pop(rid, None)

            # 4) Decode SLO violations: any request still in decode phase past its deadline
            for rid, deadline in list(decode_deadline.items()):
                if rid in decode_tokens_left:  # only while decode still incomplete
                    if not slo_violated.get(rid, False) and sim_time > deadline + SLO_EPS:
                        slo_violated[rid] = True
                        # we can keep the deadline if you want lateness to keep growing,
                        # but for a per-request violation count it's safe to drop it
                        decode_deadline.pop(rid, None)

            # 5) Compare computed vs logged violations (1 per request, total)
            expected_violations = sum(1 for v in slo_violated.values() if v)
            # make sure you use the correct column name, accounting for the CSV typo
            actual_violations = int(
                row.get("slo_violations")
                or row.get("slo_violati ons")  # fallback to current header
                or 0
            )
            if expected_violations != actual_violations:
                raise AssertionError(
                    f"[Trace {trace_index}, row {idx}] SLO Cost Correctness failed: "
                    f"computed_violations={expected_violations}, "
                    f"log_slo_violations={actual_violations}"
                )
            continue

    # All SLO checks passed for this trace
    return




def main() -> None:

    ## ** Comment the Line below & Hardcode the log path if you just need to apply test cases on a trace rather than generating a fresh new sample trace:
    log_path = run_single_mcts()
    # log_path: Path = Path("simulator_output/Test_Parrallel_Launch/test_mcts_trace_T1.csv")
    rows = load_log_rows(log_path)
    traces = build_traces_from_log(rows)
    print(f"[TRACE] Total distinct root→leaf traces found: {len(traces)}")
    print("Random Trace Below : ")
    print_random_trace(traces)
    
    print("\n\n\n")
    
    # ____ Request Related Test Cases ____
    all_ok = True
    for ti, trace in enumerate(traces):
        try:
            ## General Request Related Tests
            validate_trace(trace, ti)

            ## SLO Related Tests 
            validate_trace_slos(trace, ti)



        except AssertionError as e:
            all_ok = False
            print(f"\n[ERROR] Trace {ti} failed:\n{e}")
            print("[ERROR] Full trace:")
            for r in trace:
                print(
                    f"  phase={r.get('phase')} "
                    f"depth={r.get('depth')} "
                    f"node_id={r.get('node_id')} "
                    f"parent={r.get('parent_node_id')} "
                    f"player={r.get('player_to_act')} "
                    f"next={r.get('next_player')}"
                )
                
            # If you want to stop on first failure, uncomment:
            # raise
        
      
    if all_ok:
        print("✅ All traces for Requests + SLOs passed adversary/controller consistency checks.")




if __name__ == "__main__":
    main()
