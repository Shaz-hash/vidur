from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.selfPlay import SelfPlayRunner, SingleRootRun
from ..DNN.value_models import AlphaZeroModel
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import (
    _build_env_and_simulator,
    _load_weights_into_model,
    _set_global_seeds,
)


class _NoopWriter:
    def add(self, sample: Any) -> None:
        del sample

    def close(self) -> None:
        return None


SEARCH_FIELDS = [
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "best_action_index",
    "best_valid",
    "best_q_value",
    "best_discount",
    "best_bootstrap_value",
    "best_reward_cost",
    "action_repr",
    "expected_action_index",
    "expected_q_value",
    "selection_passed",
]


DETAIL_FIELDS = [
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "action_index",
    "valid",
    "action_q_value",
    "action_discount",
    "action_bootstrap_value",
    "action_reward_cost",
    "action_repr",
    "rank",
    "selected_by_mcts",
    "expected_best",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_checkpoint_path() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "mcts_dnn_checkpoints" / "best.pt"


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "tests"


def _float_or_blank(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _rank_valid_details(details: list[dict[str, Any]], root_player: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    valid = [row for row in details if bool(row["valid"])]
    if not valid:
        return details, None

    reverse = str(root_player) == "controller"
    ranked = sorted(
        valid,
        key=lambda row: ((-float(row["action_q_value"])) if reverse else float(row["action_q_value"]), int(row["action_index"])),
    )
    expected = ranked[0]
    ranks = {int(row["action_index"]): int(rank) for rank, row in enumerate(ranked, start=1)}
    expected_idx = int(expected["action_index"])

    for row in details:
        idx = int(row["action_index"])
        row["rank"] = ranks.get(idx, "")
        row["expected_best"] = bool(idx == expected_idx)
    return details, expected


def _details_for_root(
    *,
    mcts: VidurMCTS,
    model: AlphaZeroModel,
    root_player: str,
    root_id: int,
    root_depth: int,
    history_hops: int,
    model_version: int,
    selected_index: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    root = mcts._root
    if root is None:
        raise RuntimeError("mcts root missing after search_dnn")
    if root.cached_sim_snapshot is None or root.cached_stats is None:
        raise RuntimeError("mcts root snapshot missing after search_dnn")

    decision_state = mcts._scratch_restore(root.cached_sim_snapshot, root.cached_stats)
    actions_by_index, mask_t = mcts._actions_and_mask(decision_state, str(root_player), forbidden_stop_ids=None)
    mask_list = [bool(x) for x in mask_t.tolist()]
    valid_indices = [
        int(i)
        for i, ok in enumerate(mask_list)
        if bool(ok) and actions_by_index[int(i)] is not None
    ]
    alias_to_canon, _canon_to_aliases, canonical_indices = mcts._canonicalize_action_indices(
        player=str(root_player),
        actions_by_index=actions_by_index,
        valid_indices=valid_indices,
    )

    canonical_values: dict[int, tuple[float, float, float, float, float, float]] = {}
    for cidx in canonical_indices:
        action = actions_by_index[int(cidx)]
        if action is None:
            continue
        if str(root_player) == "adversary":
            q_tuple = mcts._evaluate_adversary_action_q_two_step(
                decision_snapshot=root.cached_sim_snapshot,
                decision_stats=root.cached_stats,
                parent_cost=float(root.state_cost),
                parent_time=float(root.sim_time),
                adv_action=action,
                dnn_model=model,
                model_version=int(model_version),
                use_model_bootstrap=True,
            )
        else:
            q_tuple = mcts._evaluate_depth1_action_q(
                decision_snapshot=root.cached_sim_snapshot,
                decision_stats=root.cached_stats,
                parent_player=str(root_player),
                parent_cost=float(root.state_cost),
                parent_time=float(root.sim_time),
                action=action,
                dnn_model=model,
                model_version=int(model_version),
                use_model_bootstrap=True,
            )
        canonical_values[int(cidx)] = q_tuple

    details: list[dict[str, Any]] = []
    selected_idx = None if selected_index is None else int(selected_index)
    for idx, action in enumerate(actions_by_index):
        valid = bool(idx in valid_indices)
        q_tuple = None
        if valid:
            canon_idx = int(alias_to_canon[int(idx)])
            q_tuple = canonical_values.get(canon_idx)
        q, reward, discount, bootstrap = (None, None, None, None)
        if q_tuple is not None:
            q, reward, discount, bootstrap = (
                float(q_tuple[0]),
                float(q_tuple[1]),
                float(q_tuple[2]),
                float(q_tuple[3]),
            )

        details.append(
            {
                "root_id": int(root_id),
                "root_player": str(root_player),
                "root_depth": int(root_depth),
                "history_hops": int(history_hops),
                "action_index": int(idx),
                "valid": valid,
                "action_q_value": q,
                "action_discount": discount,
                "action_bootstrap_value": bootstrap,
                "action_reward_cost": reward,
                "action_repr": "" if action is None else repr(action),
                "rank": "",
                "selected_by_mcts": bool(selected_idx is not None and int(idx) == int(selected_idx)),
                "expected_best": False,
            }
        )

    return _rank_valid_details(details, str(root_player))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = _bool_text(value)
                elif isinstance(value, float):
                    value = _float_or_blank(value)
                elif value is None:
                    value = ""
                out[key] = value
            writer.writerow(out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GV3 Python depth-one search min/max action selection."
    )
    parser.add_argument("--num-roots", type=int, default=64)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=100)
    parser.add_argument("--history-seed", type=int, default=2026)
    parser.add_argument("--model-version", type=int, default=1)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--checkpoint-path", default=str(_default_checkpoint_path()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--require-both-players", action="store_true", default=True)
    parser.add_argument("--allow-single-player", action="store_false", dest="require_both_players")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    search_csv = output_dir / "test_checking_depth1_search.csv"
    details_csv = output_dir / "test_checking_depth1_details.csv"

    cfg = replace(
        DEFAULT_MULTIPROCESS_TRAINING_CONFIG,
        environment_lang="python",
        use_virtual_env=True,
        model=replace(DEFAULT_MULTIPROCESS_TRAINING_CONFIG.model, device=str(args.model_device)),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
    )
    cfg.validate()
    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )

    _simulator, env, _constraints, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=True)
    spec = make_dnn_spec(cfg=cfg.game_v2)
    model = AlphaZeroModel(spec=spec).to(torch.device(cfg.model.device))
    _load_weights_into_model(model, checkpoint_path)
    model.eval()

    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        rng=random.Random(int(args.history_seed)),
        log_path=str(output_dir / "test_checking_depth1_mcts_iter.csv"),
        tree_log_path=str(output_dir / "test_checking_depth1_mcts_root.csv"),
        logger_flush_every=1,
        verbose=False,
        complete_log=False,
    )
    runner = SelfPlayRunner(
        env=env,
        mcts=mcts,
        model=model,
        writer=_NoopWriter(),
        eval_writer=None,
        device_for_features=torch.device(cfg.model.device),
        game_v2_cfg=cfg.game_v2,
    )

    search_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    players_seen: set[str] = set()
    failures: list[int] = []

    for prepared_batch in runner._iter_prepared_history_root_batches(
        game_id=0,
        num_roots=int(args.num_roots),
        start_root_id=0,
        start_root_depth=0,
        start_player="adversary",
        initial_state=None,
        history_nontrivial_hops=int(args.history_hops_min),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        history_max_total_steps=int(cfg.history_max_total_steps),
        log_history_rows=False,
        root_batch_size=int(cfg.history_root_batch_size),
        allow_duplicate_history_fallback=True,
    ):
        for pr in prepared_batch:
            root_state = pr.root_state
            root_player = str(pr.root_player)
            root_depth = int(pr.root_depth)
            root_id = int(pr.root_id)

            root_state, root_player, root_depth = runner._advance_to_branching_root(
                root_state,
                root_player,
                root_depth,
                max_hops=int(cfg.max_forced_hops_per_root),
            )
            if root_player == "adversary":
                root_state, _ = runner._build_root_decision_state_for_adversary(
                    current_state=root_state,
                    pre_controller_snapshot=pr.pre_controller_snapshot,
                    pre_controller_stats=pr.pre_controller_stats,
                )

            result = runner.run_single_root(
                SingleRootRun(
                    game_id=0,
                    root_id=int(root_id),
                    root_depth=int(root_depth),
                    root_player=str(root_player),
                    feature_version=int(cfg.run.feature_version),
                    root_node_id_override=pr.root_node_id_override,
                    model_version=int(args.model_version),
                ),
                root_state=root_state,
            )

            root_details, expected = _details_for_root(
                mcts=mcts,
                model=model,
                root_player=str(root_player),
                root_id=int(root_id),
                root_depth=int(root_depth),
                history_hops=int(pr.history_hops),
                model_version=int(args.model_version),
                selected_index=int(result.best_idx) if int(result.best_idx) >= 0 else None,
            )
            detail_rows.extend(root_details)

            expected_idx = None if expected is None else int(expected["action_index"])
            expected_q = None if expected is None else float(expected["action_q_value"])
            selected = next((row for row in root_details if bool(row["selected_by_mcts"])), None)
            selected_idx = None if selected is None else int(selected["action_index"])
            passed = bool(expected_idx is not None and selected_idx == expected_idx)
            if not passed:
                failures.append(int(root_id))
            players_seen.add(str(root_player))

            search_rows.append(
                {
                    "root_id": int(root_id),
                    "root_player": str(root_player),
                    "root_depth": int(root_depth),
                    "history_hops": int(pr.history_hops),
                    "best_action_index": "" if selected_idx is None else int(selected_idx),
                    "best_valid": bool(selected is not None and bool(selected["valid"])),
                    "best_q_value": None if selected is None else selected["action_q_value"],
                    "best_discount": None if selected is None else selected["action_discount"],
                    "best_bootstrap_value": None if selected is None else selected["action_bootstrap_value"],
                    "best_reward_cost": None if selected is None else selected["action_reward_cost"],
                    "action_repr": "" if selected is None else str(selected["action_repr"]),
                    "expected_action_index": "" if expected_idx is None else int(expected_idx),
                    "expected_q_value": expected_q,
                    "selection_passed": bool(passed),
                }
            )

            if hasattr(mcts, "clear_search_state"):
                mcts.clear_search_state(drop_scratch=False)

    _write_csv(search_csv, SEARCH_FIELDS, search_rows)
    _write_csv(details_csv, DETAIL_FIELDS, detail_rows)

    if bool(args.require_both_players) and players_seen != {"controller", "adversary"}:
        raise RuntimeError(
            "audit did not cover both players; "
            f"players_seen={sorted(players_seen)}, increase --num-roots or use --allow-single-player"
        )
    if failures:
        raise RuntimeError(
            f"depth1 selection audit failed for root_id(s): {failures[:20]} "
            f"(wrote {search_csv} and {details_csv})"
        )

    print(
        "depth1 selection audit passed: "
        f"roots={len(search_rows)}, players={sorted(players_seen)}, "
        f"search_csv={search_csv}, details_csv={details_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
