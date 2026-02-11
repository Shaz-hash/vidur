# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
# vidur/vidur/mcts/verifier/verifier.py

# RUN IT WITH : python3 -m vidur.mcts.Verifier.verifier
from __future__ import annotations

import csv
import json
import math
import sys
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from vidur.mcts.DNN.infer import build_model_inputs

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

from vidur.mcts.environment import (
    AdversaryAction,
    AdversaryRequestSpec,
    ControllerAction,
    VidurGameStats,
    VidurMCTSEnvironment,
    VidurMCTSState,
)
from vidur.mcts.launch_mcts_job import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions


_TOKEN_ALLOC_RE = re.compile(r"token_allocations\s*=\s*(\{.*\})")

# -----------------------------
# User config (edit these)
# -----------------------------

JUST_VERIFY_COST = False  # If True, only verify cost at root; skip best-path search

# Put the copied mcts_root_pXX.csv here as root.csv
ROOT_CSV_PATH = Path(__file__).with_name("root.csv")

# Depth measured in *branching* nodes only (forced single-action nodes do not count).
BRANCH_DEPTH = 3

# If True: ignore BRANCH_DEPTH and search until no prefill remains (or MAX_COVER_DEPTH reached)
COVER_TO_PREFILL: bool = True
MAX_COVER_DEPTH: int = 5

# If we want the best_path.csv results to extend the history state
BEST_AS_HISTORY_EXTEND: bool = True


# Root selection:
# - True: replay history using all rows except the last, and verify the last row's root state
# - False: replay ALL rows and verify the resulting state (less "verifier-ish", but sometimes useful)
VERIFY_LAST_ROOT_STATE = False

# Hard cap to avoid infinite loops when auto-advancing forced single-action chains
MAX_FORCED_HOPS = 20_000

# Ensure we enumerate all actions at branching nodes (don’t use your MCTS max_branching here)
ENUM_MAX_SAMPLES = 10_000

# Output
BEST_PATH_OUT = Path("simulator_output/best_path.csv")

# TO CHECK AGAINST TRIVIAL STRATEGIES :
# -----------------------------
# Fixed controller policy runner
# -----------------------------
CHECK_TRIVIAL = True # If True, run fixed policy verifier after MCTS verifier, You need to have the last row to be adversary in the adversary.csv for it to work, and the fixed policy will start from the state right after applying that adversary action.
ADVERSARY_CSV_PATH = Path(__file__).with_name("adversary.csv")

FIXED_POLICY_PREFILL_BUDGET = 1024  # e.g. 512, 1024, ..., 3072
FIXED_POLICY_HEURISTIC = "SJF"     # one of: SJF, EDF, LST, LJF
FIXED_POLICY_MAX_STEPS = 200       # safety cap
FIXED_POLICY_PRINT_EACH_STEP = False


_CONTROLLER_HEUR_ORDER = ["SJF", "EDF", "LST", "LJF"]
_HEUR_TO_IDX = {h: i for i, h in enumerate(_CONTROLLER_HEUR_ORDER)}



## Verifying the features :
DUMP_INFER_FEATURES: bool = True
INFER_FEATURES_OUT: Optional[Path] = Path("simulator_output/verifier_infer_features.jsonl")
INFER_FEATURES_PRINT: bool = False  # set True if you want stdout too


# -----------------------------
# Simulation / Env config
# (keep consistent with alphaZeroParrallel.py)
# -----------------------------

SIM_CLI_ARGS: Sequence[str] = [
    "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
    "--replica_config_device", "h100",
    "--replica_config_network_device", "h100_dgx",
    "--cluster_config_num_replicas", "1",
    "--replica_config_tensor_parallel_size", "1",
    "--replica_config_num_pipeline_stages", "1",
    "--global_scheduler_config_type", "round_robin",
    "--replica_scheduler_config_type", "vllm_v1",
    "--vllm_v1_scheduler_config_batch_size_cap", "512",
]

# Constraints (match your AlphaZero config)
MAXIMUM_QPS = 5
INTERVAL_REQUEST_SIZE = 512
MIN_REQUEST_TOKENS = 512
MAX_REQUEST_TOKENS = 3072
PREFILL_PROFILE_PATH = "simulator_output/prefill_profile.csv"
PREFILL_SLOWDOWN = 3.0
PREFILL_SLOS = (3.0,)   # seconds multiplier is already baked into environment logic in your current code
DECODE_SLOS = (50.0,)   # ms


# Explore cfg (only matters for env._advance_simulation_fast depth cap; keep same as training run)
SIMULATION_DEPTH = 2
SIMULATION_RANDOM_TRIES = 1
EXPLORATION_CONSTANT = 1.7
MAX_BRANCHING_UNUSED_HERE = 10
CONTROLLER_BUDGET_COMBS_UNUSED_HERE = 10


# -----------------------------
# Helpers
# -----------------------------

def _mask_to_list(mask: Any) -> List[bool]:
    # environment returns plain list[bool] already; keep generic
    if isinstance(mask, list):
        return [bool(x) for x in mask]
    try:
        # torch tensor etc
        return [bool(x) for x in mask]
    except Exception:
        return list(mask)


def configure_simulation(sim_args: Iterable[str]) -> SimulationConfig:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(sim_args)
        cfg = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = original_argv

    # Keep lightweight
    cfg.metrics_config.write_metrics = False
    cfg.metrics_config.enable_chrome_trace = False
    cfg.metrics_config.write_json_trace = False

    # Keep generator quiet (self-play adversary injects requests)
    if hasattr(cfg.request_generator_config, "num_requests"):
        cfg.request_generator_config.num_requests = 0  # type: ignore[attr-defined]

    # If you rely on infinite KV shortcut:
    try:
        setattr(cfg.cluster_config.cache_config, "assume_infinite_kv", True)
    except Exception:
        pass

    return cfg


def _controller_action_key(action: ControllerAction) -> tuple:
    alloc = action.token_allocations or {}
    return tuple(sorted((int(rid), int(tok)) for rid, tok in alloc.items()))


def _parse_action_json(action_json: str) -> Union[AdversaryAction, ControllerAction]:
    d = json.loads(action_json)
    typ = str(d.get("type", "")).strip().lower()

    if typ == "adversary":
        reqs: List[AdversaryRequestSpec] = []
        for r in (d.get("requests") or []):
            reqs.append(
                AdversaryRequestSpec(
                    prefill_tokens=int(r.get("prefill_tokens", 0)),
                    decode_tokens=int(r.get("decode_tokens", 0)),
                    prefill_slo=float(r.get("prefill_slo", 0.0)),
                    decode_slo=float(r.get("decode_slo", 0.0)),
                )
            )
        stop = [int(x) for x in (d.get("stop_decode_ids") or [])]
        return AdversaryAction(requests=reqs, stop_decode_ids=stop)

    if typ == "controller":
        tok_alloc_raw = d.get("token_allocations") or {}
        prefill_alloc_raw = d.get("prefill_allocations") or {}
        decode_alloc_raw = d.get("decode_allocations") or {}
        print("Decode Allocations : ", decode_alloc_raw)
        tok_alloc = {int(k): int(v) for k, v in tok_alloc_raw.items()}
        prefill_alloc = {int(k): int(v) for k, v in prefill_alloc_raw.items()}
        decode_alloc = {int(k): int(v) for k, v in decode_alloc_raw.items()}

        sel = d.get("selected_request_ids")
        if sel is None:
            selected_ids = None
        else:
            selected_ids = [int(x) for x in sel]

        mapping = d.get("mapping")
        mapping_t = tuple(int(x) for x in mapping) if mapping is not None else None

        return ControllerAction(
            token_budget=int(d.get("token_budget", 0)),
            selected_request_ids=selected_ids,
            token_allocations=tok_alloc,
            prefill_allocations=prefill_alloc,
            decode_allocations=decode_alloc,
            heuristic=d.get("heuristic"),
            strategy=d.get("strategy"),
            mapping=mapping_t,
        )

    raise ValueError(f"Unknown action type in best_action_json: type={typ!r}")


def _parse_token_allocations_from_action_repr(action_repr: str) -> Dict[int, int]:
    m = _TOKEN_ALLOC_RE.search(action_repr)
    if not m:
        raise ValueError(f"cannot parse token_allocations from action_repr={action_repr!r}")
    d = ast.literal_eval(m.group(1))
    if not isinstance(d, dict):
        raise ValueError(f"token_allocations parsed to non-dict: {type(d)}")
    return {int(k): int(v) for k, v in d.items()}





def _action_repr(action: Union[AdversaryAction, ControllerAction]) -> str:
    if isinstance(action, ControllerAction):
        return (
            f"ControllerAction(token_budget={int(action.token_budget)}, "
            f"heuristic={action.heuristic!r}, strategy={action.strategy!r}, "
            f"token_allocations={dict(action.token_allocations)})"
        )
    # adversary
    return (
        f"AdversaryAction(num_requests={len(action.requests)}, "
        f"prefill_tokens={action.requests[0].prefill_tokens if action.requests else 0}, "
        f"stop_decode_ids={list(action.stop_decode_ids)})"
    )

def has_prefill_work(env: VidurMCTSEnvironment, state: VidurMCTSState) -> bool:
    # ok to use the env helper (private) for verifier
    lookup = env._build_request_lookup(state.simulator)  # type: ignore[attr-defined]
    for req in lookup.values():
        remaining = max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))
        prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
        if remaining > 0 and (not prefill_done):
            return True
    return False


def _dump_infer_features(state: VidurMCTSState, *, tag: str = "") -> None:
    # CPU-only so this never touches CUDA
    inputs = build_model_inputs(state, player="controller", device=torch.device("cpu"))

    req_mask = inputs.req_mask.squeeze(0).to("cpu").tolist()          # [20] bools
    req_feat = inputs.req_features.squeeze(0).to("cpu").tolist()      # [20][3]
    req_valid = [req_feat[i] for i, ok in enumerate(req_mask) if ok]  # only real req slots

    glob = inputs.global_features.squeeze(0).to("cpu").tolist()       # [9]

    rec = {
        "tag": tag,
        "sim_time": float(getattr(state.simulator, "_time", 0.0)),
        "global_features": glob,
        "req_features": req_valid,
    }

    if INFER_FEATURES_OUT is not None:
        INFER_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
        with INFER_FEATURES_OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    if INFER_FEATURES_PRINT:
        print(f"[INFER] {tag} t={rec['sim_time']:.6f} global={glob}")
        for j, (rem_norm, cached_norm, slack) in enumerate(req_valid):
            print(f"  req[{j}] rem={rem_norm:.6f} cached={cached_norm:.6f} slack={slack:.6f}")


@dataclass(frozen=True)
class RootLogRow:
    game_id: int
    root_id: int
    root_depth: int
    root_player: str
    best_action_index: int
    best_action_repr: str
    best_action_json: str


def load_root_csv(path: Path) -> List[RootLogRow]:
    rows: List[RootLogRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            # robust to missing fields
            rows.append(
                RootLogRow(
                    game_id=int(d.get("game_id", 0) or 0),
                    root_id=int(d.get("root_id", 0) or 0),
                    root_depth=int(d.get("root_depth", 0) or 0),
                    root_player=str(d.get("root_player", "") or "").strip(),
                    best_action_index=int(d.get("best_action_index", 0) or 0),
                    best_action_repr=str(d.get("best_action_repr", "") or ""),
                    best_action_json=str(d.get("best_action_json", "") or ""),
                )
            )
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    return rows


def build_env() -> VidurMCTSEnvironment:
    sim_cfg = configure_simulation(SIM_CLI_ARGS)
    simulator = Simulator(sim_cfg, register_atexit=False)

    slo_options = RequestSLOOptions(prefill_slos=tuple(PREFILL_SLOS), decode_slos=tuple(DECODE_SLOS))
    constraints = MCTSConstraintConfig(
        maximum_qps=MAXIMUM_QPS,
        min_request_tokens=MIN_REQUEST_TOKENS,
        max_request_tokens=MAX_REQUEST_TOKENS,
        interval_request_size=INTERVAL_REQUEST_SIZE,
        request_slo_options=slo_options,
        prefill_slowdown=PREFILL_SLOWDOWN,
        prefill_profile_path=PREFILL_PROFILE_PATH,
    )

    explore_cfg = MCTSExploreConfig(
        simulation_depth=SIMULATION_DEPTH,
        simulation_random_tries=SIMULATION_RANDOM_TRIES,
        exploration_constant=EXPLORATION_CONSTANT,
        max_branching=MAX_BRANCHING_UNUSED_HERE,
        controller_budget_combs=CONTROLLER_BUDGET_COMBS_UNUSED_HERE,
    )

    return VidurMCTSEnvironment(base_simulator=simulator, constraints=constraints, explore_cfg=explore_cfg)


def actions_and_valid(
    env: VidurMCTSEnvironment, state: VidurMCTSState, player: str
) -> Tuple[List[Optional[Union[AdversaryAction, ControllerAction]]], List[int]]:
    if player == "controller":
        actions_by_index, mask = env.sample_controller_actions(state, ENUM_MAX_SAMPLES)
    else:
        actions_by_index, mask = env.sample_adversary_actions(state, ENUM_MAX_SAMPLES)

    mask_list = _mask_to_list(mask)
    valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
    return actions_by_index, valid


def advance_forced_until_branching(
    env: VidurMCTSEnvironment,
    state: VidurMCTSState,
    player: str,
    *,
    max_hops: int,
) -> Tuple[VidurMCTSState, str]:
    """
    Apply forced moves (len(valid)==1) until we reach a branching node (len(valid)!=1).
    """
    for _ in range(int(max_hops)):
        actions_by_index, valid = actions_and_valid(env, state, player)
        if len(valid) != 1:
            return state, player

        idx = valid[0]
        action = actions_by_index[idx]
        assert action is not None

        if player == "controller":
            assert isinstance(action, ControllerAction)
            state = env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"
        else:
            assert isinstance(action, AdversaryAction)
            state = env.apply_adversary_action_only(state, action, inplace=True)
            player = "controller"

    raise RuntimeError(f"advance_forced_until_branching exceeded max_hops={max_hops}")


def dedupe_branch_actions(
    player: str,
    actions_by_index: List[Optional[Union[AdversaryAction, ControllerAction]]],
    valid: List[int],
) -> List[int]:
    """
    For controller: dedupe by token_allocations (same as VidurMCTS._controller_action_key).
    For adversary: no dedupe (keep all).
    """
    if player != "controller":
        return list(valid)

    sig_to_idx: Dict[tuple, int] = {}
    for idx in valid:
        act = actions_by_index[idx]
        if act is None:
            continue
        if not isinstance(act, ControllerAction):
            continue
        sig = _controller_action_key(act)
        if sig not in sig_to_idx:
            sig_to_idx[sig] = int(idx)

    return sorted(sig_to_idx.values())


@dataclass(frozen=True)
class StepMetrics:
    sim_time: float
    violations: int
    lateness: float
    violated_requests: int
    objective_cost: float


def compute_metrics(env: VidurMCTSEnvironment, state: VidurMCTSState) -> StepMetrics:
    v, late = env.evaluate_objective(state)
    violated = len(getattr(state.stats, "violated_request_ids", set()) or set())
    t = float(getattr(state.simulator, "_time", 0.0))
    cost = float(v) + float(late)
   
    return StepMetrics(sim_time=t, violations=int(v), lateness=float(late), violated_requests=int(violated), objective_cost=cost)



# TRIVIAL POLICY RUNNER
def _remaining_prefill_total(env: VidurMCTSEnvironment, state: VidurMCTSState) -> int:
    lookup = env._build_request_lookup(state.simulator)  # type: ignore[attr-defined]
    total = 0
    for req in lookup.values():
        prefill_done = bool(getattr(req, "_is_prefill_complete", req.is_prefill_complete))
        if prefill_done:
            continue
        remaining = max(0, int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens))
        total += remaining
    return int(total)


def _pick_fixed_controller_action(
    env: VidurMCTSEnvironment,
    state: VidurMCTSState,
    *,
    prefill_budget: int,
    heuristic: str,
) -> tuple[int, ControllerAction]:
    actions_by_index, mask = env.sample_controller_actions(state, ENUM_MAX_SAMPLES)
    mask_list = _mask_to_list(mask)

    valid: list[int] = []
    for i, ok in enumerate(mask_list):
        if not ok:
            continue
        act = actions_by_index[i]
        if isinstance(act, ControllerAction):
            valid.append(i)

    if not valid:
        raise RuntimeError("No valid controller actions in this state (unexpected).")

    step = int(INTERVAL_REQUEST_SIZE)  # keep consistent with env constraints
    heur_key = str(heuristic).strip().upper()
    if heur_key not in _HEUR_TO_IDX:
        raise ValueError(f"Unknown heuristic={heuristic!r}; expected one of {_CONTROLLER_HEUR_ORDER}")
    heur_idx = _HEUR_TO_IDX[heur_key]

    # budget groups are: step*(1..6) -> budget_idx 0..5
    b = int(prefill_budget)
    if b <= 0:
        budget_idx = 0
    else:
        budget_idx = (b // step) - 1
    budget_idx = max(0, min(5, int(budget_idx)))

    desired_idx = int(budget_idx * 4 + heur_idx)
    if desired_idx in valid:
        act = actions_by_index[desired_idx]
        assert isinstance(act, ControllerAction)
        return desired_idx, act

    # Fallback (e.g. tail end when remaining_prefill < requested budget):
    # choose the largest valid budget group for this heuristic that is <= desired budget group,
    # else choose the smallest available.
    cand = [i for i in valid if (i % 4) == heur_idx]
    if not cand:
        raise RuntimeError(f"No valid actions for heuristic={heur_key} in this state (unexpected).")

    leq = [i for i in cand if (i // 4) <= budget_idx]
    chosen = max(leq, key=lambda i: i // 4) if leq else min(cand, key=lambda i: i // 4)

    act = actions_by_index[chosen]
    assert isinstance(act, ControllerAction)
    return int(chosen), act


def run_fixed_controller_policy_from_adversary_csv(
    adversary_csv: Path,
    *,
    controller_prefill_budget: int,
    controller_heuristic: str,
    max_steps: int = 200,
    print_each_step: bool = True,
) -> None:
    if not adversary_csv.exists():
        raise FileNotFoundError(f"Missing adversary csv at {adversary_csv}")

    env = build_env()
    # adv_row = load_root_csv(adversary_csv)[0]

    # state = env.initial_state()
    # player = "adversary"

    # # Ensure we start on a branching node before applying the logged adversary action
    # state, player = advance_forced_until_branching(env, state, player, max_hops=MAX_FORCED_HOPS)
    # if adv_row.root_player and adv_row.root_player != player:
    #     raise RuntimeError(
    #         f"adversary.csv root_player mismatch: row.root_player={adv_row.root_player!r} but current player={player!r}"
    #     )

    # adv_action = _parse_action_json(adv_row.best_action_json)
    # if not isinstance(adv_action, AdversaryAction):
    #     raise RuntimeError("adversary.csv best_action_json must be an adversary action.")
    # state = env.apply_adversary_action_only(state, adv_action, inplace=True)

    adv_rows = load_root_csv(adversary_csv)

    if len(adv_rows) == 1:
        state = env.initial_state()
        player = "adversary"
        state, player = advance_forced_until_branching(env, state, player, max_hops=MAX_FORCED_HOPS)
        adv_row = adv_rows[0]
    else:
        # Replay history up to last row's root; last row should be the adversary action root.
        state, player, target_row = replay_history_to_root_state(
            env,
            adv_rows,
            verify_last_root_state=True,
        )
        if target_row is None:
            raise RuntimeError("Failed to recover target row from adversary history.")
        adv_row = target_row

    if adv_row.root_player and adv_row.root_player != player:
        raise RuntimeError(
            f"adversary.csv root_player mismatch: row.root_player={adv_row.root_player!r} but current player={player!r}"
        )

    adv_action = _parse_action_json(adv_row.best_action_json)
    if not isinstance(adv_action, AdversaryAction):
        raise RuntimeError("adversary.csv best_action_json must be an adversary action.")
    state = env.apply_adversary_action_only(state, adv_action, inplace=True)


    m0 = compute_metrics(env, state)
    print("=" * 100)
    print("[FIXED_POLICY] After applying adversary action:")
    print(f"  sim_time={m0.sim_time:.6f}")
    print(f"  violations={m0.violations}")
    print(f"  lateness_sum={m0.lateness:.9f}")
    print(f"  violated_requests={m0.violated_requests}")
    print(f"  objective_cost={m0.objective_cost:.9f}")
    print(f"  remaining_prefill_total={_remaining_prefill_total(env, state)}")

    if not has_prefill_work(env, state):
        print("[FIXED_POLICY] no prefill work in system; nothing to do.")
        return

    for step_idx in range(int(max_steps)):
        if not has_prefill_work(env, state):
            break

        idx, act = _pick_fixed_controller_action(
            env,
            state,
            prefill_budget=int(controller_prefill_budget),
            heuristic=str(controller_heuristic),
        )
        prefill_total = int(sum((act.prefill_allocations or {}).values()))
        decode_total = int(sum((act.decode_allocations or {}).values()))

        state = env.apply_controller_action_only(state, act, inplace=True)

        m = compute_metrics(env, state)
        rem = _remaining_prefill_total(env, state)

        if print_each_step:
            print(
                f"[FIXED_POLICY] step={step_idx:04d} idx={idx:02d} "
                f"heur={str(controller_heuristic).upper()} "
                f"prefill_budget_req={int(controller_prefill_budget)} "
                f"prefill_total={prefill_total} decode_total={decode_total} "
                f"rem_prefill={rem} "
                f"t={m.sim_time:.6f} viol={m.violations} late={m.lateness:.9f} req_viol={m.violated_requests}"
            )

    m_end = compute_metrics(env, state)
    print("=" * 100)
    print("[FIXED_POLICY] Final state (stopped when no prefill remained OR max_steps reached):")
    print(f"  sim_time={m_end.sim_time:.6f}")
    print(f"  violations={m_end.violations}")
    print(f"  lateness_sum={m_end.lateness:.9f}")
    print(f"  violated_requests={m_end.violated_requests}")
    print(f"  objective_cost={m_end.objective_cost:.9f}")
    print(f"  remaining_prefill_total={_remaining_prefill_total(env, state)}")



@dataclass(frozen=True)
class PathStep:
    step_idx: int
    player_acted: str
    action_index: int
    action_repr: str
    sim_time: float
    violations: int
    lateness: float
    violated_requests: int
    objective_cost: float


def replay_history_to_root_state(
    env: VidurMCTSEnvironment,
    root_rows: List[RootLogRow],
    *,
    verify_last_root_state: bool,
) -> Tuple[VidurMCTSState, str, Optional[RootLogRow]]:
    """
    Reconstruct the root state by replaying best_action_json from root_rows.

    If verify_last_root_state=True:
      - Treat last row as the "target root" to verify.
      - Replay all rows except last; stop at the branching root state that should match last.root_player.
      - Return (state, player_to_act, target_row)

    Else:
      - Replay all rows; return final state and player after forced-advance to branching.
    """
    state = env.initial_state()
    target: Optional[RootLogRow] = None

    if verify_last_root_state:
        target = root_rows[-1]
        history = root_rows[:-1]
    else:
        history = root_rows

    # Start from the first row's root_player if possible, else default
    player = (history[0].root_player if history else (target.root_player if target else "adversary")) or "adversary"

    # Make sure we’re actually at a branching node before applying the first recorded action
    state, player = advance_forced_until_branching(env, state, player, max_hops=MAX_FORCED_HOPS)

    for i, row in enumerate(history):
        if row.root_player and row.root_player != player:
            raise RuntimeError(
                f"History replay mismatch at row {i}: row.root_player={row.root_player} but current player={player}"
            )

        if DUMP_INFER_FEATURES:
            _dump_infer_features(state, tag="compute_metrics")

        action = _parse_action_json(row.best_action_json)

        if player == "controller":
            assert isinstance(action, ControllerAction)
            state = env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"
        else:
            assert isinstance(action, AdversaryAction)
            state = env.apply_adversary_action_only(state, action, inplace=True)
            player = "controller"

        m = compute_metrics(env, state)
        print(f"For Row {i}")
        print(f"  sim_time={m.sim_time:.6f}")
        print(f"  violations={m.violations}")
        print(f"  lateness_sum={m.lateness:.9f}")
        print(f"  violated_requests={m.violated_requests}")
        print(f"  objective_cost={m.objective_cost:.9f}")


        # Skip forced moves to next branching root (this mimics self-play advance)
        state, player = advance_forced_until_branching(env, state, player, max_hops=MAX_FORCED_HOPS)
        
    # If we’re verifying last root row, enforce player alignment
    if verify_last_root_state and target is not None:
        if target.root_player and target.root_player != player:
            raise RuntimeError(
                f"Target root_player mismatch after history replay: target={target.root_player} current={player}"
            )

    return state, player, target


def replay_best_path_csv_as_history(
    env: VidurMCTSEnvironment,
    state: VidurMCTSState,
    player: str,
    best_path_csv: Path,
    *,
    cover_to_prefill: bool,
) -> Tuple[VidurMCTSState, str]:
    if not best_path_csv.exists() or best_path_csv.stat().st_size == 0:
        return state, player

    with best_path_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    for j, row in enumerate(rows):
        # always move to the next branching/terminal node first
        state, player = advance_forced_until_branching(env, state, player, max_hops=MAX_FORCED_HOPS)

        # cover_to_prefill: never allow adversary actions at branching; force a no-op turn
        if cover_to_prefill and player == "adversary":
            noop = AdversaryAction(requests=[], stop_decode_ids=[])
            state = env.apply_adversary_action_only(state, noop, inplace=True)
            player = "controller"
            # continue loop (next CSV row should still be controller)
            continue

        acted = str(row.get("player_acted", "") or "").strip()
        action_repr = str(row.get("action_repr", "") or "")

        if acted != player:
            raise RuntimeError(f"best_path replay mismatch at row {j}: csv player={acted} but current player={player}")

        if acted != "controller":
            # If you ever need adversary replay too, you must store action_json or action_index in best_path.csv.
            raise RuntimeError(f"best_path replay only supports controller right now; saw player_acted={acted!r}")

        wanted_alloc = _parse_token_allocations_from_action_repr(action_repr)
        wanted_sig = tuple(sorted((int(k), int(v)) for k, v in wanted_alloc.items()))

        actions_by_index, valid = actions_and_valid(env, state, player)
        found_idx: Optional[int] = None
        found_act: Optional[ControllerAction] = None
        for idx in valid:
            act = actions_by_index[idx]
            if not isinstance(act, ControllerAction):
                continue
            sig = tuple(sorted((int(k), int(v)) for k, v in (act.token_allocations or {}).items()))
            if sig == wanted_sig:
                found_idx = int(idx)
                found_act = act
                break

        if found_act is None:
            raise RuntimeError(f"Could not find controller action for token_allocations={wanted_alloc} at row={j}")

        state = env.apply_controller_action_only(state, found_act, inplace=True)
        player = "adversary"

    state, player = advance_forced_until_branching(env, state, player, max_hops=MAX_FORCED_HOPS)
    return state, player

def count_trajectories(
    env: VidurMCTSEnvironment,
    state: VidurMCTSState,
    player: str,
    *,
    depth: int,
) -> int:
    """
    Count unique trajectories of length `depth` (branching decisions only),
    using snapshot/restore backtracking.
    """
    # Ensure current is branching/terminal
    state, player = advance_forced_until_branching(env, state, player, max_hops=MAX_FORCED_HOPS)
    actions_by_index, valid = actions_and_valid(env, state, player)

    if depth <= 0 or not valid:
        return 1

    if len(valid) == 1:
        print("[WARNING] count_trajectories reached forced single-action node unexpectedly")
        # should not happen after advance_forced_until_branching, but handle defensively
        return 1

    uniq = dedupe_branch_actions(player, actions_by_index, valid)

    total = 0
    for idx in uniq:
        sim_snap = state.simulator.snapshot_state()
        stats_snap = state.stats.clone()

        act = actions_by_index[idx]
        assert act is not None

        next_player = "controller" if player == "adversary" else "adversary"
        if player == "controller":
            assert isinstance(act, ControllerAction)
            env.apply_controller_action_only(state, act, inplace=True)
        else:
            assert isinstance(act, AdversaryAction)
            env.apply_adversary_action_only(state, act, inplace=True)

        # Advance forced nodes before next branching decision
        state2, player2 = advance_forced_until_branching(env, state, next_player, max_hops=MAX_FORCED_HOPS)
        total += count_trajectories(env, state2, player2, depth=depth - 1)

        # Restore
        state.simulator.restore_state(sim_snap)
        state.stats = stats_snap.clone()

    return total


# def find_best_path(
#     env: VidurMCTSEnvironment,
#     state: VidurMCTSState,
#     player: str,
#     *,
#     depth: int,
# ) -> Tuple[float, List[PathStep], int]:
#     """
#     Brute force best path minimizing (violations + lateness) at the leaf.
#     Returns: (best_cost, best_steps, total_trajectories)
#     """
#     best_cost = math.inf
#     best_steps: List[PathStep] = []
#     total_traj = 0

#     def dfs(
#         st: VidurMCTSState,
#         pl: str,
#         depth_left: int,
#         steps: List[PathStep],
#     ) -> None:
#         nonlocal best_cost, best_steps, total_traj

#         st, pl = advance_forced_until_branching(env, st, pl, max_hops=MAX_FORCED_HOPS)
#         actions_by_index, valid = actions_and_valid(env, st, pl)

#         if depth_left <= 0 or not valid:
#             total_traj += 1
#             leaf_m = compute_metrics(env, st)
#             if leaf_m.objective_cost < best_cost:
#                 best_cost = leaf_m.objective_cost
#                 best_steps = list(steps)
#             return

#         uniq = dedupe_branch_actions(pl, actions_by_index, valid)
#         if not uniq:
#             total_traj += 1
#             leaf_m = compute_metrics(env, st)
#             if leaf_m.objective_cost < best_cost:
#                 best_cost = leaf_m.objective_cost
#                 best_steps = list(steps)
#             return

#         for idx in uniq:
#             sim_snap = st.simulator.snapshot_state()
#             stats_snap = st.stats.clone()

#             act = actions_by_index[idx]
#             assert act is not None

#             next_player = "controller" if pl == "adversary" else "adversary"
#             if pl == "controller":
#                 assert isinstance(act, ControllerAction)
#                 env.apply_controller_action_only(st, act, inplace=True)
#             else:
#                 assert isinstance(act, AdversaryAction)
#                 env.apply_adversary_action_only(st, act, inplace=True)

#             # advance forced chain to the next branching root (this is what we log per "step")
#             st2, pl2 = advance_forced_until_branching(env, st, next_player, max_hops=MAX_FORCED_HOPS)
#             m = compute_metrics(env, st2)

#             steps.append(
#                 PathStep(
#                     step_idx=len(steps),
#                     player_acted=pl,
#                     action_index=int(idx),
#                     action_repr=_action_repr(act),
#                     sim_time=m.sim_time,
#                     violations=m.violations,
#                     lateness=m.lateness,
#                     violated_requests=m.violated_requests,
#                     objective_cost=m.objective_cost,
#                 )
#             )

#             dfs(st2, pl2, depth_left - 1, steps)
#             steps.pop()

#             # restore
#             st.simulator.restore_state(sim_snap)
#             st.stats = stats_snap.clone()

#     dfs(state, player, depth, [])
#     return float(best_cost), best_steps, int(total_traj)


def find_best_path(
    env: VidurMCTSEnvironment,
    state: VidurMCTSState,
    player: str,
    *,
    depth: int,
    cover_to_prefill: bool = False,   # NEW
) -> Tuple[float, List[PathStep], int]:
    """
    If cover_to_prefill=False: brute force best path of length `depth` (branching decisions only).
    If cover_to_prefill=True: brute force until there is NO remaining prefill work,
    with a safety cap of `depth` branching decisions.
    """
    best_cost = math.inf
    best_steps: List[PathStep] = []
    total_traj = 0

    def dfs(
        st: VidurMCTSState,
        pl: str,
        depth_left: int,
        steps: List[PathStep],
    ) -> None:
        nonlocal best_cost, best_steps, total_traj

        st, pl = advance_forced_until_branching(env, st, pl, max_hops=MAX_FORCED_HOPS)

        # NEW termination: stop as soon as prefill is fully cleared
        if cover_to_prefill and (not has_prefill_work(env, st)):
            total_traj += 1
            print("[VERIFIER] reached no-prefill-work state; terminating this path.")
            leaf_m = compute_metrics(env, st)
            if leaf_m.objective_cost < best_cost:
                best_cost = leaf_m.objective_cost
                best_steps = list(steps)
            return
        
        # --- NEW: cover_to_prefill => forbid adversary generating requests ---
        if cover_to_prefill and pl == "adversary":
            # Force adversary to be a no-op (never send new requests) while we try to clear prefill.
            noop = AdversaryAction(requests=[], stop_decode_ids=[])
            env.apply_adversary_action_only(st, noop, inplace=True)

            # Do NOT consume depth_left (this isn’t a branching decision we care about).
            dfs(st, "controller", depth_left, steps)
            return
        # -------------------------------------------------------------------

        actions_by_index, valid = actions_and_valid(env, st, pl)

        # existing termination: depth cap or terminal
        if depth_left <= 0 or not valid:
            total_traj += 1
            leaf_m = compute_metrics(env, st)
            if leaf_m.objective_cost < best_cost:
                best_cost = leaf_m.objective_cost
                best_steps = list(steps)
            return

        uniq = dedupe_branch_actions(pl, actions_by_index, valid)
        if not uniq:
            total_traj += 1
            leaf_m = compute_metrics(env, st)
            if leaf_m.objective_cost < best_cost:
                best_cost = leaf_m.objective_cost
                best_steps = list(steps)
            return

        for idx in uniq:
            sim_snap = st.simulator.snapshot_state()
            stats_snap = st.stats.clone()

            act = actions_by_index[idx]
            assert act is not None

            next_player = "controller" if pl == "adversary" else "adversary"
            if pl == "controller":
                assert isinstance(act, ControllerAction)
                env.apply_controller_action_only(st, act, inplace=True)
            else:
                assert isinstance(act, AdversaryAction)
                env.apply_adversary_action_only(st, act, inplace=True)

            st2, pl2 = advance_forced_until_branching(env, st, next_player, max_hops=MAX_FORCED_HOPS)
            m = compute_metrics(env, st2)

            steps.append(
                PathStep(
                    step_idx=len(steps),
                    player_acted=pl,
                    action_index=int(idx),
                    action_repr=_action_repr(act),
                    sim_time=m.sim_time,
                    violations=m.violations,
                    lateness=m.lateness,
                    violated_requests=m.violated_requests,
                    objective_cost=m.objective_cost,
                )
            )

            dfs(st2, pl2, depth_left - 1, steps)
            steps.pop()

            st.simulator.restore_state(sim_snap)
            st.stats = stats_snap.clone()

    dfs(state, player, depth, [])
    return float(best_cost), best_steps, int(total_traj)



def write_best_path_csv(path: Path, steps: List[PathStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["player_acted", "action_repr", "violations", "lateness", "requests_violated", "sim_time", "objective_cost"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in steps:
            w.writerow(
                {
                    "player_acted": s.player_acted,
                    "action_repr": s.action_repr,
                    "violations": s.violations,
                    "lateness": f"{s.lateness:.9f}",
                    "requests_violated": s.violated_requests,
                    "sim_time": f"{s.sim_time:.9f}",
                    "objective_cost": f"{s.objective_cost:.9f}",
                }
            )


def append_best_path_csv(path: Path, steps: List[PathStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["player_acted", "action_repr", "violations", "lateness", "requests_violated", "sim_time", "objective_cost"]

    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for s in steps:
            w.writerow(
                {
                    "player_acted": s.player_acted,
                    "action_repr": s.action_repr,
                    "violations": s.violations,
                    "lateness": f"{s.lateness:.9f}",
                    "requests_violated": s.violated_requests,
                    "sim_time": f"{s.sim_time:.9f}",
                    "objective_cost": f"{s.objective_cost:.9f}",
                }
            )



def main() -> None:
    if not ROOT_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing root.csv at {ROOT_CSV_PATH}")

    root_rows = load_root_csv(ROOT_CSV_PATH)
    env = build_env()

    # Reconstruct root state from history
    state, player, target_row = replay_history_to_root_state(
        env, root_rows, verify_last_root_state=VERIFY_LAST_ROOT_STATE
    )

    if JUST_VERIFY_COST == True:
        m = compute_metrics(env, state)
        print("=" * 100)
        print("[VERIFIER] Root trace summary (after replaying ALL root.csv rows):")
        print(f"  sim_time={m.sim_time:.6f}")
        print(f"  violations={m.violations}")
        print(f"  lateness_sum={m.lateness:.9f}")
        print(f"  violated_requests={m.violated_requests}")
        print(f"  objective_cost={m.objective_cost:.9f}")
        return

    if CHECK_TRIVIAL == True :
        print("Check trivial fixed-policy run:")
        # --- quick fixed-policy run (optional) ---
        if ADVERSARY_CSV_PATH.exists():
            run_fixed_controller_policy_from_adversary_csv(
                ADVERSARY_CSV_PATH,
                controller_prefill_budget=FIXED_POLICY_PREFILL_BUDGET,
                controller_heuristic=FIXED_POLICY_HEURISTIC,
                max_steps=FIXED_POLICY_MAX_STEPS,
                print_each_step=FIXED_POLICY_PRINT_EACH_STEP,
            )
            return
        else :
            print(f"[VERIFIER] Skipping fixed-policy run; no adversary csv at {ADVERSARY_CSV_PATH}")



    if BEST_AS_HISTORY_EXTEND:
        state, player = replay_best_path_csv_as_history(
            env,
            state,
            player,
            BEST_PATH_OUT,
            cover_to_prefill=COVER_TO_PREFILL,
        )

    if COVER_TO_PREFILL and (not has_prefill_work(env, state)):
        print("[VERIFIER] no prefill work in system; nothing to verify.")
        return


    # This is the state we’ll verify from
    base_metrics = compute_metrics(env, state)
    print("=" * 100)
    print(f"[VERIFIER] root.csv={ROOT_CSV_PATH}")
    # print(f"[VERIFIER] branching_depth={BRANCH_DEPTH}")
    print(f"[VERIFIER] verify_last_root_state={VERIFY_LAST_ROOT_STATE}")
    print(f"[VERIFIER] start_player={player}")
    print(
        f"[VERIFIER] start sim_time={base_metrics.sim_time:.6f} "
        f"violations={base_metrics.violations} lateness={base_metrics.lateness:.6f} "
        f"objective_cost={base_metrics.objective_cost:.6f}"
    )

    if target_row is not None:
        print("-" * 100)
        print(f"[VERIFIER] Target root row: game_id={target_row.game_id} root_id={target_row.root_id} root_depth={target_row.root_depth} root_player={target_row.root_player}")
        print(f"[VERIFIER] Logged best_action_index={target_row.best_action_index} best_action_repr={target_row.best_action_repr}")

    # Snapshot so count/eval start from identical root
    root_sim_snap = state.simulator.snapshot_state()
    root_stats_snap = state.stats.clone()

    # 1) Count trajectories (same rules as search)
    # count = count_trajectories(env, state, player, depth=BRANCH_DEPTH)
    # print(f"[VERIFIER] total_unique_trajectories(depth={BRANCH_DEPTH}) = {count}")

    # Restore root
    state.simulator.restore_state(root_sim_snap)
    state.stats = root_stats_snap.clone()

    # 2) Evaluate and pick best
    # best_cost, best_steps, total_traj = find_best_path(env, state, player, depth=BRANCH_DEPTH)
    search_depth = MAX_COVER_DEPTH if COVER_TO_PREFILL else BRANCH_DEPTH

    best_cost, best_steps, total_traj = find_best_path(
        env,
        state,
        player,
        depth=search_depth,
        cover_to_prefill=COVER_TO_PREFILL,
    )
    print(f"[VERIFIER] branching_depth={search_depth}")
    print(f"[VERIFIER] evaluated_trajectories = {total_traj}")
    print(f"[VERIFIER] best_cost = {best_cost:.9f}")
    print("[VERIFIER] best path:")
    for s in best_steps:
        print(
            f"  step={s.step_idx} player={s.player_acted} idx={s.action_index} "
            f"viol={s.violations} late={s.lateness:.6f} req_viol={s.violated_requests} "
            f"t={s.sim_time:.6f} :: {s.action_repr}"
        )

    # write_best_path_csv(BEST_PATH_OUT, best_steps)
    append_best_path_csv(BEST_PATH_OUT, best_steps)
    print(f"[VERIFIER] wrote {BEST_PATH_OUT}")

    # Optional: compare verifier best first action to logged best action (only meaningful when verifying last root state)
    if target_row is not None and best_steps:
        try:
            logged_act = _parse_action_json(target_row.best_action_json)
            logged_key = _controller_action_key(logged_act) if isinstance(logged_act, ControllerAction) else None
        except Exception:
            logged_key = None

        # Recompute key for our best first action
        # (Note: best_steps[0].action_index is an index from env sampling at this root.)
        # We can recover the action object by sampling once more at root.
        state.simulator.restore_state(root_sim_snap)
        state.stats = root_stats_snap.clone()
        actions_by_index, valid = actions_and_valid(env, state, player)
        act0 = actions_by_index[best_steps[0].action_index]
        our_key = _controller_action_key(act0) if isinstance(act0, ControllerAction) else None

        print("-" * 100)
        print(f"[VERIFIER] compare-first-action:")
        print(f"  logged best_action_index={target_row.best_action_index} key={logged_key}")
        print(f"  brute  best_action_index={best_steps[0].action_index} key={our_key}")


if __name__ == "__main__":
    main()
