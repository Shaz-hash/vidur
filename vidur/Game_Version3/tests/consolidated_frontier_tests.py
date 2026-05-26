from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.history_root import HistoryRootGenerator
from ..DNN.selfPlay import SelfPlayRunner, SingleRootRun
from ..DNN.value_models import AlphaZeroModel
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import _build_env_and_simulator, _load_weights_into_model, _set_global_seeds
from .feature_conversion_tests import _make_action_mask_fn
from .frontier_feature_trace_logger import (
    build_frontier_state_row,
    build_production_feature_row,
    compare_feature_rows,
    read_csv,
    write_csv,
)
from .frontier_feature_trace_tests import (
    COMPARE_FIELDS,
    FEATURE_FIELDS,
    FRONTIER_FIELDS,
    _expand_final_root_iter_csv,
)
from .history_node_tests import (
    _json_dumps,
    _strict_signature,
    _validate_trace_csv,
    _valid_count_for_frontier,
    _write_frontier_csv,
)
from .test_checking_depth1 import (
    DETAIL_FIELDS,
    SEARCH_FIELDS,
    _NoopWriter,
    _details_for_root,
    _write_csv,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "tests"


def _default_checkpoint_path() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "mcts_dnn_checkpoints" / "best.pt"


def _generate_frontiers(
    *,
    env: Any,
    output_dir: Path,
    num_roots: int,
    history_hops_min: int,
    history_hops_max: int,
    history_seed: int,
    batch_size: int,
    max_total_steps: int,
    allow_duplicate_fallback: bool,
) -> tuple[list[dict[str, Any]], Path, Path, dict[str, Any]]:
    trace_csv = output_dir / "consolidated_history_mcts_iter.csv"
    frontier_csv = output_dir / "consolidated_frontiers.csv"

    trace_logger = DNNMCTSIterationLogger(trace_csv, flush_every=1)
    history = HistoryRootGenerator(env=env)
    try:
        records: list[dict[str, Any]] = []
        for batch in history.generate_roots_batch_iter(
            initial_state=None,
            start_player="adversary",
            start_depth=0,
            num_roots=int(num_roots),
            game_id=0,
            start_root_id=0,
            nontrivial_hops=int(history_hops_min),
            min_history_hops=int(history_hops_min),
            max_history_hops=int(history_hops_max),
            seed=int(history_seed),
            max_total_steps=int(max_total_steps),
            batch_size=int(batch_size),
            allow_duplicate_fallback=bool(allow_duplicate_fallback),
            history_trace_logger=trace_logger,
        ):
            records.extend(batch)
    finally:
        trace_logger.close()

    if len(records) != int(num_roots):
        raise RuntimeError(f"expected {int(num_roots)} frontier roots, generated {len(records)}")

    seen_history: set[str] = set()
    seen_strict: set[str] = set()
    frontier_rows: list[dict[str, Any]] = []
    duplicate_history: list[int] = []
    duplicate_strict: list[int] = []
    forced_frontiers: list[int] = []

    for record in records:
        root_id = int(record["root_id"])
        history_sig = record.get("history_signature")
        strict_sig = _strict_signature(env, record)
        history_key = _json_dumps(history_sig)
        strict_key = _json_dumps(strict_sig)

        hist_dup = history_key in seen_history
        strict_dup = strict_key in seen_strict
        if hist_dup:
            duplicate_history.append(root_id)
        if strict_dup:
            duplicate_strict.append(root_id)
        seen_history.add(history_key)
        seen_strict.add(strict_key)

        valid_count = _valid_count_for_frontier(env, record)
        frontier_kind = "terminal" if valid_count == 0 else ("forced" if valid_count == 1 else "branching")
        if frontier_kind == "forced":
            forced_frontiers.append(root_id)

        frontier_rows.append(
            {
                "root_id": root_id,
                "root_player": str(record["root_player"]),
                "root_depth": int(record["root_depth"]),
                "history_hops": int(record.get("history_hops", 0)),
                "history_log_node_id": "" if record.get("history_log_node_id") is None else int(record["history_log_node_id"]),
                "history_signature_json": history_key,
                "strict_signature_json": strict_key,
                "duplicate_history_signature": str(bool(hist_dup)).lower(),
                "duplicate_strict_signature": str(bool(strict_dup)).lower(),
                "valid_action_count": int(valid_count),
                "frontier_kind": frontier_kind,
            }
        )

    _write_frontier_csv(frontier_csv, frontier_rows)

    if duplicate_history or duplicate_strict:
        raise RuntimeError(
            "frontier uniqueness failed: "
            f"history_duplicates={duplicate_history[:20]}, strict_duplicates={duplicate_strict[:20]} "
            f"(wrote {frontier_csv})"
        )
    if forced_frontiers:
        raise RuntimeError(f"frontier correctness failed: forced single-action roots={forced_frontiers[:20]}")

    traces_checked, adv_actions_checked = _validate_trace_csv(trace_csv)
    return (
        records,
        trace_csv,
        frontier_csv,
        {
            "frontiers": len(records),
            "trace_chains": int(traces_checked),
            "adv_actions_checked": int(adv_actions_checked),
        },
    )


def _run_depth1_selection_check(
    *,
    records: list[dict[str, Any]],
    env: Any,
    explore_cfg: Any,
    cfg: Any,
    checkpoint_path: Path,
    output_dir: Path,
    history_seed: int,
    model_version: int,
) -> dict[str, Any]:
    search_csv = output_dir / "consolidated_depth1_search.csv"
    details_csv = output_dir / "consolidated_depth1_details.csv"

    spec = make_dnn_spec(cfg=cfg.game_v2)
    model = AlphaZeroModel(spec=spec).to(torch.device(cfg.model.device))
    _load_weights_into_model(model, checkpoint_path)
    model.eval()

    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        rng=random.Random(int(history_seed)),
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
    failures: list[int] = []
    players_seen: set[str] = set()
    root_states_for_feature_checks: list[dict[str, Any]] = []

    for record in records:
        root_state = record["root_state"]
        root_player = str(record["root_player"])
        root_depth = int(record["root_depth"])
        root_id = int(record["root_id"])
        history_hops = int(record.get("history_hops", 0))

        root_state, root_player, root_depth = runner._advance_to_branching_root(
            root_state,
            root_player,
            root_depth,
            max_hops=int(cfg.max_forced_hops_per_root),
        )
        if root_player == "adversary":
            root_state, _ = runner._build_root_decision_state_for_adversary(
                current_state=root_state,
                pre_controller_snapshot=record.get("pre_controller_snapshot", None),
                pre_controller_stats=record.get("pre_controller_stats", None),
            )

        root_states_for_feature_checks.append(
            {
                "root_id": root_id,
                "root_player": root_player,
                "root_depth": root_depth,
                "history_hops": history_hops,
                "history_log_node_id": record.get("history_log_node_id"),
                "root_state": root_state.fork(flag=False),
            }
        )

        result = runner.run_single_root(
            SingleRootRun(
                game_id=0,
                root_id=root_id,
                root_depth=root_depth,
                root_player=root_player,
                feature_version=int(cfg.run.feature_version),
                root_node_id_override=record.get("root_node_id_override", None),
                model_version=int(model_version),
            ),
            root_state=root_state,
        )

        root_details, expected = _details_for_root(
            mcts=mcts,
            model=model,
            root_player=root_player,
            root_id=root_id,
            root_depth=root_depth,
            history_hops=history_hops,
            model_version=int(model_version),
            selected_index=int(result.best_idx) if int(result.best_idx) >= 0 else None,
        )
        detail_rows.extend(root_details)

        expected_idx = None if expected is None else int(expected["action_index"])
        expected_q = None if expected is None else float(expected["action_q_value"])
        selected = next((row for row in root_details if bool(row["selected_by_mcts"])), None)
        selected_idx = None if selected is None else int(selected["action_index"])
        passed = bool(expected_idx is not None and selected_idx == expected_idx)
        if not passed:
            failures.append(root_id)
        players_seen.add(root_player)

        search_rows.append(
            {
                "root_id": root_id,
                "root_player": root_player,
                "root_depth": root_depth,
                "history_hops": history_hops,
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

    if failures:
        raise RuntimeError(f"depth1 Bellman selection failed for root_ids={failures[:20]} (wrote {search_csv})")

    return {
        "depth1_roots": len(search_rows),
        "depth1_detail_rows": len(detail_rows),
        "depth1_players": sorted(players_seen),
        "search_csv": search_csv,
        "details_csv": details_csv,
        "root_states_for_feature_checks": root_states_for_feature_checks,
    }


def _run_frontier_feature_trace_check(
    *,
    root_records: list[dict[str, Any]],
    env: Any,
    cfg: Any,
    output_dir: Path,
    tolerance: float,
    source_trace_csv: Path,
) -> dict[str, Any]:
    frontier_csv = output_dir / "consolidated_feature_frontiers.csv"
    feature_csv = output_dir / "consolidated_frontier_feature.csv"
    compare_csv = output_dir / "consolidated_frontier_feature_compare.csv"
    final_trace_csv = output_dir / "consolidated_feature_final_root_iter.csv"

    frontier_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    action_mask_fn = _make_action_mask_fn(env)

    for record in root_records:
        root_id = int(record["root_id"])
        root_player = str(record["root_player"])
        root_depth = int(record["root_depth"])
        history_hops = int(record["history_hops"])
        root_state = record["root_state"]
        history_log_node_id = record.get("history_log_node_id")

        frontier_rows.append(
            build_frontier_state_row(
                env=env,
                state=root_state,
                root_id=root_id,
                root_player=root_player,
                root_depth=root_depth,
                history_hops=history_hops,
                history_log_node_id=history_log_node_id,
            )
        )
        feature_rows.append(
            build_production_feature_row(
                env=env,
                state=root_state,
                root_id=root_id,
                root_player=root_player,
                root_depth=root_depth,
                history_hops=history_hops,
                action_mask_fn=action_mask_fn,
            )
        )

    write_csv(frontier_csv, FRONTIER_FIELDS, frontier_rows)
    write_csv(feature_csv, FEATURE_FIELDS, feature_rows)
    _expand_final_root_iter_csv(
        shared_trace_csv=source_trace_csv,
        final_trace_csv=final_trace_csv,
        frontier_rows=frontier_rows,
    )
    final_trace_chains, final_trace_adv_actions = _validate_trace_csv(final_trace_csv)

    compare_rows, failures = compare_feature_rows(
        frontier_rows=read_csv(frontier_csv),
        feature_rows=read_csv(feature_csv),
        cfg=cfg.game_v2,
        tolerance=float(tolerance),
    )
    write_csv(compare_csv, COMPARE_FIELDS, compare_rows)

    if failures:
        raise RuntimeError(
            f"frontier trace feature invariant failed for root_ids={failures[:20]} "
            f"(wrote {compare_csv})"
        )

    max_feature_diff = max((float(row["max_feature_diff"]) for row in compare_rows), default=0.0)

    return {
        "feature_roots": len(compare_rows),
        "feature_max_diff": max_feature_diff,
        "feature_trace_chains": int(final_trace_chains),
        "feature_trace_adv_actions": int(final_trace_adv_actions),
        "feature_frontier_csv": frontier_csv,
        "feature_csv": feature_csv,
        "feature_compare_csv": compare_csv,
        "feature_final_trace_csv": final_trace_csv,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run consolidated GV3 frontier, Bellman selection, and trace-derived feature tests."
    )
    parser.add_argument("--num-roots", type=int, default=32)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=32)
    parser.add_argument("--history-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-total-steps", type=int, default=20000)
    parser.add_argument("--allow-duplicate-fallback", action="store_true", default=False)
    parser.add_argument("--checkpoint-path", default=str(_default_checkpoint_path()))
    parser.add_argument("--model-version", type=int, default=1)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--feature-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

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

    records, trace_csv, frontier_csv, history_report = _generate_frontiers(
        env=env,
        output_dir=output_dir,
        num_roots=int(args.num_roots),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        batch_size=int(args.batch_size),
        max_total_steps=int(args.max_total_steps),
        allow_duplicate_fallback=bool(args.allow_duplicate_fallback),
    )

    depth_report = _run_depth1_selection_check(
        records=records,
        env=env,
        explore_cfg=explore_cfg,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        history_seed=int(args.history_seed),
        model_version=int(args.model_version),
    )

    feature_report = _run_frontier_feature_trace_check(
        root_records=depth_report["root_states_for_feature_checks"],
        env=env,
        cfg=cfg,
        output_dir=output_dir,
        tolerance=float(args.feature_tolerance),
        source_trace_csv=trace_csv,
    )

    print(
        "consolidated GV3 tests passed: "
        f"frontiers={history_report['frontiers']}, "
        f"trace_chains={history_report['trace_chains']}, "
        f"depth1_roots={depth_report['depth1_roots']}, "
        f"feature_roots={feature_report['feature_roots']}, "
        f"feature_max_diff={feature_report['feature_max_diff']:.3g}"
    )
    print(f"history_trace_csv={trace_csv}")
    print(f"frontier_csv={frontier_csv}")
    print(f"depth1_search_csv={depth_report['search_csv']}")
    print(f"depth1_details_csv={depth_report['details_csv']}")
    print(f"feature_frontier_csv={feature_report['feature_frontier_csv']}")
    print(f"feature_csv={feature_report['feature_csv']}")
    print(f"feature_compare_csv={feature_report['feature_compare_csv']}")
    print(f"feature_final_trace_csv={feature_report['feature_final_trace_csv']}")


if __name__ == "__main__":
    main()
