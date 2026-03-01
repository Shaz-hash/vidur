# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
    SAMPLE COMMAND TO RUN THIS :

python3 -m vidur.mcts.Verifier.check_mcts_dnn_distribution \
  --history-csv simulator_output/mcts_dnn_logs/gen_000154/mcts_root_p00_gen00.csv \
  --prior-mode uniform \
  --discount-factor 0.98 \
  --simulations 10000 \
  --out-csv simulator_output/mcts_dnn_logs/mcts_distribution_uniform.csv


cd /home/shazer/Desktop/Research/Vidur/vidur
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.Verifier.check_mcts_distribution \
  --checkpoint simulator_output/mcts_dnn_checkpoints/best.pt \
  --history-csv simulator_output/mcts_dnn_logs/sample_history.csv \
  --prior-mode model \
  --discount-factor 0.98 \
  --simulations 100000 \
  --game-id 0 \
  --root-id 0 \
  --root-depth 0 \
  --out-csv simulator_output/mcts_dnn_logs/mcts_distribution_uniform_fresh.csv


"""



from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from vidur.mcts.alphaZeroParrallel import (
    AlphaZeroConfig,
    DatasetGroup,
    LoggingGroup,
    MCTSConstraintsGroup,
    MCTSExploreGroup,
    ModelGroup,
    RunGroup,
    SimulationCLIGroup,
    _build_env_and_simulator,
)
from vidur.mcts.environment import (
    AdversaryAction,
    AdversaryRequestSpec,
    ControllerAction,
    VidurMCTSState,
)
from vidur.mcts.mctsDNN import VidurMCTS
from vidur.mcts.DNN.models import AlphaZeroModel


@dataclass(frozen=True)
class RootCSVRow:
    game_id: int
    root_id: int
    root_depth: int
    root_node_id: int
    root_player: str
    phase: str
    best_action_json: str


def _mask_to_list(mask: Any) -> List[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in mask]


def _is_explicit_history_row(row: RootCSVRow) -> bool:
    s = (row.best_action_json or "").strip()
    if not s:
        return False
    try:
        d = json.loads(s)
    except Exception:
        return False
    return isinstance(d, dict) and ("history_phase" in d)


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
        tok_alloc = {int(k): int(v) for k, v in (d.get("token_allocations") or {}).items()}
        prefill_alloc = {int(k): int(v) for k, v in (d.get("prefill_allocations") or {}).items()}
        decode_alloc = {int(k): int(v) for k, v in (d.get("decode_allocations") or {}).items()}
        sel = d.get("selected_request_ids")
        selected_ids = None if sel is None else [int(x) for x in sel]
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

    raise ValueError(f"Unknown action type in best_action_json: {typ!r}")


def load_history_rows(path: Path) -> List[RootCSVRow]:
    rows: List[RootCSVRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                RootCSVRow(
                    game_id=int(d.get("game_id", 0) or 0),
                    root_id=int(d.get("root_id", 0) or 0),
                    root_depth=int(d.get("root_depth", 0) or 0),
                    root_node_id=int(d.get("root_node_id", 0) or 0),
                    root_player=str(d.get("root_player", "") or "").strip(),
                    phase=str(d.get("phase", "") or "").strip(),
                    best_action_json=str(d.get("best_action_json", "") or "").strip(),
                )
            )
    if not rows:
        raise RuntimeError(f"No rows found in history CSV: {path}")
    return rows


def actions_and_valid(env: Any, state: VidurMCTSState, player: str, max_samples: int) -> Tuple[List[Optional[object]], List[int]]:
    if player == "controller":
        actions_by_index, mask = env.sample_controller_actions(state, max_samples)
    else:
        actions_by_index, mask = env.sample_adversary_actions(state, max_samples)
    mask_list = _mask_to_list(mask)
    valid = [i for i, ok in enumerate(mask_list) if ok and actions_by_index[i] is not None]
    return actions_by_index, valid


def advance_forced_until_branching(
    env: Any,
    state: VidurMCTSState,
    player: str,
    *,
    max_hops: int,
    max_samples: int,
) -> Tuple[VidurMCTSState, str]:
    for _ in range(int(max_hops)):
        actions_by_index, valid = actions_and_valid(env, state, player, max_samples)
        if len(valid) != 1:
            return state, player

        idx = int(valid[0])
        action = actions_by_index[idx]
        assert action is not None

        if player == "controller":
            state = env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"
        else:
            state = env.apply_adversary_action_only(state, action, inplace=True)
            player = "controller"

    raise RuntimeError(f"Exceeded max_hops={max_hops} while advancing forced chain")


def replay_history_to_state_after_last_action(
    env: Any,
    rows: List[RootCSVRow],
    *,
    align_branching_roots: bool,
    max_forced_hops: int,
    max_samples: int,
) -> Tuple[VidurMCTSState, str, RootCSVRow]:
    state = env.initial_state()
    player = rows[0].root_player or "adversary"

    # Keep only rows with actions
    action_rows = [r for r in rows if (r.best_action_json or "").strip()]
    if not action_rows:
        raise RuntimeError("No actionable rows (best_action_json empty)")

    for i, row in enumerate(action_rows):
        is_hist = _is_explicit_history_row(row)

        if align_branching_roots and (not is_hist):
            state, player = advance_forced_until_branching(
                env,
                state,
                player,
                max_hops=max_forced_hops,
                max_samples=max_samples,
            )

        if row.root_player and row.root_player != player:
            raise RuntimeError(
                f"Replay mismatch row={i}: row.root_player={row.root_player} current_player={player}"
            )

        action = _parse_action_json(row.best_action_json)

        if player == "controller":
            if not isinstance(action, ControllerAction):
                raise RuntimeError(f"Expected controller action at row={i}, got {type(action).__name__}")
            state = env.apply_controller_action_only(state, action, inplace=True)
            player = "adversary"
        else:
            if not isinstance(action, AdversaryAction):
                raise RuntimeError(f"Expected adversary action at row={i}, got {type(action).__name__}")
            state = env.apply_adversary_action_only(state, action, inplace=True)
            player = "controller"

        # Move to next branching root between rows, but NOT after last row (user expectation)
        if i < len(action_rows) - 1 and align_branching_roots and (not is_hist):
            state, player = advance_forced_until_branching(
                env,
                state,
                player,
                max_hops=max_forced_hops,
                max_samples=max_samples,
            )

    return state, player, action_rows[-1]


class _DummyModel:
    """Used only when prior_mode=uniform (model should never be called)."""

    def __init__(self) -> None:
        self._p = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def parameters(self):
        yield self._p

    def infer_from_inputs(self, *args, **kwargs):
        raise RuntimeError("infer_from_inputs called in uniform mode; mctsDNN uniform switch not applied correctly.")


def _build_default_cfg(simulations: int, prefill_profile_path: str) -> AlphaZeroConfig:
    return AlphaZeroConfig(
        sim=SimulationCLIGroup(
            cli_args=[
                "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
                "--replica_config_device", "h100",
                "--replica_config_network_device", "h100_dgx",
                "--cluster_config_num_replicas", "1",
                "--replica_config_tensor_parallel_size", "1",
                "--replica_config_num_pipeline_stages", "1",
                "--global_scheduler_config_type", "round_robin",
                "--replica_scheduler_config_type", "vllm_v1",
                "--vllm_v1_scheduler_config_batch_size_cap", "512",
                "--no-snapshot_rng_state",
            ]
        ),
        constraints=MCTSConstraintsGroup(
            maximum_qps=5,
            interval_request_size=512,
            min_request_tokens=512,
            max_request_tokens=3072,
            prefill_profile_path=prefill_profile_path,
            prefill_slowdown=3.0,
            prefill_slos=(3.0,),
            decode_slos=(50.0,),
        ),
        explore=MCTSExploreGroup(
            simulation_depth=2,
            simulation_random_tries=1,
            exploration_constant=1.7,
            max_branching=10,
            controller_budget_combs=10,
            controller_min_prior_threshold=0.01,
            adversary_min_prior_threshold=0.1,
            root_dirichlet_noise_enabled=False,
            root_dirichlet_alpha=0.6,
            root_dirichlet_epsilon=0.25,
        ),
        model=ModelGroup(
            num_actions_controller=24,
            num_actions_adversary=6,
            device="cpu",
        ),
        logging=LoggingGroup(
            mcts_iter_log="/tmp/mcts_iter_unused.csv",
            mcts_root_log="/tmp/mcts_root_unused.csv",
            flush_every=1,
        ),
        dataset=DatasetGroup(out_dir="/tmp/mcts_dataset_unused", shard_size=16),
        run=RunGroup(iterations=int(simulations)),
    )


def _load_model(checkpoint: Optional[Path], device: torch.device) -> AlphaZeroModel:
    model = AlphaZeroModel(num_actions_controller=24, num_actions_adversary=6).to(device).eval()
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
    return model


def collect_tree_distribution_rows(
    root: Any,
    *,
    prior_mode: str,
    history_last_actor: str,
    only_visited: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    q = deque([(root, 0)])

    while q:
        node, level = q.popleft()

        if only_visited and level > 0 and int(getattr(node, "visits", 0)) <= 0:
            continue

        parent = getattr(node, "parent", None)
        node_id = int(getattr(node, "node_id", -1))
        parent_id = None if parent is None else int(getattr(parent, "node_id", -1))
        acted = history_last_actor if parent is None else str(getattr(parent, "player", ""))
        action = getattr(node, "parent_action", None)

        if prior_mode == "uniform":
            initial_dnn_value = 0.0
        else:
            nnv = getattr(node, "nn_value_controller", None)
            initial_dnn_value = "" if nnv is None else float(nnv)

        row = {
            "level": int(level),
            "node_id": node_id,
            "parent_node_id": "" if parent_id is None else int(parent_id),
            "node_player_to_act": str(getattr(node, "player", "")),
            "player_acted_to_create_this_node": str(acted),
            "action_index": "" if getattr(node, "parent_action_index", None) is None else int(node.parent_action_index),
            "action_repr": "" if action is None else repr(action),
            "avg_value": float(node.mean_value()),
            "value_sum": float(getattr(node, "value_sum", 0.0)),
            "visits": int(getattr(node, "visits", 0)),
            "initial_dnn_value": initial_dnn_value,
            "model_prior": "" if parent is None else float(getattr(node, "prior", 0.0)),
            "children_count": int(len(getattr(node, "children", {}))),
        }
        rows.append(row)

        children = getattr(node, "children", {})
        for aidx in sorted(children.keys()):
            q.append((children[aidx], level + 1))

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-csv", required=True, type=str, help="mcts_root_p*.csv / mcts_root.csv history file")
    ap.add_argument("--prior-mode", choices=["model", "uniform"], default="model")
    ap.add_argument("--discount-factor", type=float, default=0.98)
    ap.add_argument("--simulations", type=int, default=4000)
    ap.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint (.pt) for model mode")
    ap.add_argument("--prefill-profile", type=str, default="simulator_output/prefill_profile.csv")
    ap.add_argument("--use-virtual-env", action="store_true", default=True)
    ap.add_argument("--align-branching-roots", action="store_true", default=True)
    ap.add_argument("--max-forced-hops", type=int, default=20000)
    ap.add_argument("--enum-max-samples", type=int, default=10000)
    ap.add_argument("--root-node-id-override", type=int, default=None)
    ap.add_argument("--root-depth", type=int, default=None)
    ap.add_argument("--game-id", type=int, default=None)
    ap.add_argument("--root-id", type=int, default=None)
    ap.add_argument("--only-visited", action="store_true", default=True)
    ap.add_argument("--out-csv", type=str, default="simulator_output/mcts_dnn_logs/mcts_distribution.csv")
    ap.add_argument("--print-stdout", action="store_true", default=False)
    args = ap.parse_args()

    if not (0.0 <= float(args.discount_factor) <= 1.0):
        raise SystemExit("--discount-factor must be in [0, 1]")

    history_path = Path(args.history_csv)
    if not history_path.exists():
        raise SystemExit(f"History CSV not found: {history_path}")

    rows = load_history_rows(history_path)
    cfg = _build_default_cfg(simulations=int(args.simulations), prefill_profile_path=str(args.prefill_profile))

    _, env, _, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=bool(args.use_virtual_env))

    # runtime knobs for this checker
    setattr(explore_cfg, "prior_value_mode", str(args.prior_mode))
    setattr(explore_cfg, "discount_factor", float(args.discount_factor))

    state, player, last_row = replay_history_to_state_after_last_action(
        env,
        rows,
        align_branching_roots=bool(args.align_branching_roots),
        max_forced_hops=int(args.max_forced_hops),
        max_samples=int(args.enum_max_samples),
    )

    device = torch.device("cpu")
    if args.prior_mode == "uniform":
        model: Any = _DummyModel()
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        model = _load_model(ckpt, device=device)

    mcts = VidurMCTS(env=env, explore_cfg=explore_cfg, log_path=None, tree_log_path=None)

    root_node_id_override = int(args.root_node_id_override) if args.root_node_id_override is not None else int(last_row.root_node_id)
    root_depth = int(args.root_depth) if args.root_depth is not None else int(last_row.root_depth + 1)
    game_id = int(args.game_id) if args.game_id is not None else int(last_row.game_id)
    root_id = int(args.root_id) if args.root_id is not None else int(last_row.root_id + 1)

    mcts.search_dnn(
        dnn_model=model,
        rootState=state,
        root_player=player,
        iterations=int(args.simulations),
        game_id=game_id,
        root_id=root_id,
        root_node_id_override=root_node_id_override,
        root_depth=root_depth,
        root_phase="history_probe",
        cycle_label=f"prior={args.prior_mode}",
    )

    root = mcts._root
    if root is None:
        raise RuntimeError("MCTS root is None after search")

    dist_rows = collect_tree_distribution_rows(
        root,
        prior_mode=str(args.prior_mode),
        history_last_actor=str(last_row.root_player),
        only_visited=bool(args.only_visited),
    )

    out_csv = Path(args.out_csv)
    write_csv(out_csv, dist_rows)

    print(
        f"[OK] history_rows={len(rows)} replay_final_player={player} "
        f"prior_mode={args.prior_mode} discount={args.discount_factor} "
        f"simulations={args.simulations} nodes_dumped={len(dist_rows)} out={out_csv}"
    )

    if args.print_stdout and dist_rows:
        w = csv.DictWriter(sys.stdout, fieldnames=list(dist_rows[0].keys()))
        w.writeheader()
        w.writerows(dist_rows)

    mcts.close()


if __name__ == "__main__":
    main()
