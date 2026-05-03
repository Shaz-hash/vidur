from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..DNN.dnn_spec import make_dnn_spec
from ..DNN.history_root import HistoryRootGenerator
from ..DNN.infer import build_model_inputs
from ..DNN.selfPlay import SelfPlayRunner, SingleRootRun
from ..DNN.value_models import AlphaZeroModel
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import _build_env_and_simulator, _load_weights_into_model, _set_global_seeds
from .feature_conversion_tests import (
    DECODE_FEATURE_LABELS,
    GLOBAL_FEATURE_LABELS,
    PREFILL_FEATURE_LABELS,
    _expected_features,
    _json,
    _make_action_mask_fn,
    _max_abs_diff,
    _tensor_row,
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


def _write_plain_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def _run_feature_conversion_check(
    *,
    root_records: list[dict[str, Any]],
    env: Any,
    cfg: Any,
    output_dir: Path,
    tolerance: float,
) -> dict[str, Any]:
    summary_csv = output_dir / "consolidated_feature_summary.csv"
    prefill_csv = output_dir / "consolidated_feature_prefill.csv"
    decode_csv = output_dir / "consolidated_feature_decode.csv"
    global_csv = output_dir / "consolidated_feature_global.csv"

    summary_rows: list[dict[str, Any]] = []
    prefill_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    failures: list[int] = []
    action_mask_fn = _make_action_mask_fn(env)

    for record in root_records:
        root_id = int(record["root_id"])
        root_player = str(record["root_player"])
        root_state = record["root_state"]

        inputs = build_model_inputs(
            root_state,
            root_player,
            torch.device("cpu"),
            build_action_mask_flag=True,
            action_mask_fn=action_mask_fn,
        )
        expected = _expected_features(env, root_state, root_player, cfg.game_v2)

        p_diff = _max_abs_diff(inputs.prefill_req_features, expected["prefill_req_features"])
        d_diff = _max_abs_diff(inputs.decode_req_features, expected["decode_req_features"])
        g_diff = _max_abs_diff(inputs.global_features, expected["global_features"])
        p_mask_match = bool(torch.equal(inputs.prefill_req_mask.cpu(), expected["prefill_req_mask"].cpu()))
        d_mask_match = bool(torch.equal(inputs.decode_req_mask.cpu(), expected["decode_req_mask"].cpu()))
        action_mask_expected = action_mask_fn(
            root_state,
            root_player,
            int(inputs.action_mask.shape[-1]),
            torch.device("cpu"),
        )
        action_mask_match = bool(torch.equal(inputs.action_mask.cpu(), action_mask_expected.cpu()))
        legacy_feat_expected = torch.cat(
            [
                torch.nn.functional.pad(
                    expected["prefill_req_features"],
                    (0, int(inputs.req_features.shape[-1]) - int(expected["prefill_req_features"].shape[-1])),
                ),
                torch.nn.functional.pad(
                    expected["decode_req_features"],
                    (0, int(inputs.req_features.shape[-1]) - int(expected["decode_req_features"].shape[-1])),
                ),
            ],
            dim=1,
        )
        legacy_mask_expected = torch.cat([expected["prefill_req_mask"], expected["decode_req_mask"]], dim=1)
        legacy_feat_diff = _max_abs_diff(inputs.req_features, legacy_feat_expected)
        legacy_mask_match = bool(torch.equal(inputs.req_mask.cpu(), legacy_mask_expected.cpu()))
        passed = (
            p_diff <= float(tolerance)
            and d_diff <= float(tolerance)
            and g_diff <= float(tolerance)
            and legacy_feat_diff <= float(tolerance)
            and p_mask_match
            and d_mask_match
            and action_mask_match
            and legacy_mask_match
        )
        if not passed:
            failures.append(root_id)

        counts = expected["counts"]
        summary_rows.append(
            {
                "root_id": root_id,
                "root_player": root_player,
                "root_depth": int(record["root_depth"]),
                "history_hops": int(record["history_hops"]),
                "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                "num_prefill": int(counts["num_prefill"]),
                "num_decode": int(counts["num_decode"]),
                "num_active": int(counts["num_active"]),
                "slo_violations": int(counts["slo_violations"]),
                "objective_cost": float(counts["objective_cost"]),
                "prefill_max_abs_diff": p_diff,
                "decode_max_abs_diff": d_diff,
                "global_max_abs_diff": g_diff,
                "legacy_max_abs_diff": legacy_feat_diff,
                "prefill_mask_match": str(p_mask_match).lower(),
                "decode_mask_match": str(d_mask_match).lower(),
                "action_mask_match": str(action_mask_match).lower(),
                "legacy_mask_match": str(legacy_mask_match).lower(),
                "passed": str(passed).lower(),
            }
        )

        for item in expected["prefill_debug"]:
            slot = int(item["slot"])
            actual_vec = _tensor_row(inputs.prefill_req_features[0, slot])
            expected_vec = _tensor_row(expected["prefill_req_features"][0, slot])
            prefill_rows.append(
                {
                    "root_id": root_id,
                    "slot": slot,
                    "request_id": int(item["request_id"]),
                    "root_player": root_player,
                    "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                    "labels_json": _json(PREFILL_FEATURE_LABELS),
                    "actual_features_json": _json(actual_vec),
                    "expected_features_json": _json(expected_vec),
                    "max_abs_diff": max(abs(a - b) for a, b in zip(actual_vec, expected_vec)),
                }
            )

        for item in expected["decode_debug"]:
            slot = int(item["slot"])
            actual_vec = _tensor_row(inputs.decode_req_features[0, slot])
            expected_vec = _tensor_row(expected["decode_req_features"][0, slot])
            decode_rows.append(
                {
                    "root_id": root_id,
                    "slot": slot,
                    "request_id": int(item["request_id"]),
                    "root_player": root_player,
                    "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                    "labels_json": _json(DECODE_FEATURE_LABELS),
                    "actual_features_json": _json(actual_vec),
                    "expected_features_json": _json(expected_vec),
                    "max_abs_diff": max(abs(a - b) for a, b in zip(actual_vec, expected_vec)),
                }
            )

        actual_global = _tensor_row(inputs.global_features[0])
        expected_global = _tensor_row(expected["global_features"][0])
        for idx, (actual_value, expected_value) in enumerate(zip(actual_global, expected_global)):
            global_rows.append(
                {
                    "root_id": root_id,
                    "root_player": root_player,
                    "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                    "feature_index": int(idx),
                    "feature_label": GLOBAL_FEATURE_LABELS[int(idx)],
                    "actual_value": float(actual_value),
                    "expected_value": float(expected_value),
                    "abs_diff": abs(float(actual_value) - float(expected_value)),
                }
            )

    feature_summary_fields = [
        "root_id",
        "root_player",
        "root_depth",
        "history_hops",
        "sim_time",
        "num_prefill",
        "num_decode",
        "num_active",
        "slo_violations",
        "objective_cost",
        "prefill_max_abs_diff",
        "decode_max_abs_diff",
        "global_max_abs_diff",
        "legacy_max_abs_diff",
        "prefill_mask_match",
        "decode_mask_match",
        "action_mask_match",
        "legacy_mask_match",
        "passed",
    ]
    _write_plain_csv(summary_csv, feature_summary_fields, summary_rows)
    _write_plain_csv(
        prefill_csv,
        [
            "root_id",
            "slot",
            "request_id",
            "root_player",
            "sim_time",
            "labels_json",
            "actual_features_json",
            "expected_features_json",
            "max_abs_diff",
        ],
        prefill_rows,
    )
    _write_plain_csv(
        decode_csv,
        [
            "root_id",
            "slot",
            "request_id",
            "root_player",
            "sim_time",
            "labels_json",
            "actual_features_json",
            "expected_features_json",
            "max_abs_diff",
        ],
        decode_rows,
    )
    _write_plain_csv(
        global_csv,
        [
            "root_id",
            "root_player",
            "sim_time",
            "feature_index",
            "feature_label",
            "actual_value",
            "expected_value",
            "abs_diff",
        ],
        global_rows,
    )

    if failures:
        raise RuntimeError(f"feature conversion failed for root_ids={failures[:20]} (wrote {summary_csv})")

    return {
        "feature_roots": len(summary_rows),
        "feature_prefill_rows": len(prefill_rows),
        "feature_decode_rows": len(decode_rows),
        "feature_summary_csv": summary_csv,
        "feature_prefill_csv": prefill_csv,
        "feature_decode_csv": decode_csv,
        "feature_global_csv": global_csv,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run consolidated GV3 frontier, Bellman selection, and feature-conversion tests."
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

    feature_report = _run_feature_conversion_check(
        root_records=depth_report["root_states_for_feature_checks"],
        env=env,
        cfg=cfg,
        output_dir=output_dir,
        tolerance=float(args.feature_tolerance),
    )

    print(
        "consolidated GV3 tests passed: "
        f"frontiers={history_report['frontiers']}, "
        f"trace_chains={history_report['trace_chains']}, "
        f"depth1_roots={depth_report['depth1_roots']}, "
        f"feature_roots={feature_report['feature_roots']}"
    )
    print(f"history_trace_csv={trace_csv}")
    print(f"frontier_csv={frontier_csv}")
    print(f"depth1_search_csv={depth_report['search_csv']}")
    print(f"depth1_details_csv={depth_report['details_csv']}")
    print(f"feature_summary_csv={feature_report['feature_summary_csv']}")


if __name__ == "__main__":
    main()
