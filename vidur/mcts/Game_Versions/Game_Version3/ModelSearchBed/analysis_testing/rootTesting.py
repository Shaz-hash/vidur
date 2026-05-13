"""Generate and validate a small controller-root dataset for ModelSearchBed.

Tests covered:
    1. Create one debug trace CSV per generated root and validate every trace
       with the existing GV3 history trace validator.
    2. Confirm that at least the requested fraction of stored roots have
       abs(selected-action reward/target) >= the configured threshold.

The selected MCTS/Bellman action is written separately from the history trace
CSV so the trace CSV remains compatible with `DNNMCTSIterationLogger` tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


try:
    from ..root_storage import (
        DEFAULT_NONZERO_EPS,
        build_storage_config,
        generate_and_store_roots,
        load_stored_roots,
    )
    from ...logger.mctsDNN_logger import DNNMCTSIterationLogger
    from ...tests.history_node_tests import _validate_trace_csv
except ImportError:
    repo_root = Path(__file__).resolve().parents[6]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.root_storage import (
        DEFAULT_NONZERO_EPS,
        build_storage_config,
        generate_and_store_roots,
        load_stored_roots,
    )
    from vidur.mcts.Game_Versions.Game_Version3.logger.mctsDNN_logger import DNNMCTSIterationLogger
    from vidur.mcts.Game_Versions.Game_Version3.tests.history_node_tests import _validate_trace_csv


ACTION_FIELDS = [
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "target_value",
    "best_action_index",
    "best_action_repr",
    "best_reward",
    "best_discount",
    "best_bootstrap",
    "best_child_cost",
    "best_child_time",
    "is_nonzero_target",
    "is_nonzero_reward",
    "is_large_abs_target",
    "is_large_abs_reward",
    "target_reward_abs_diff",
]


@dataclass(frozen=True)
class RootTestingConfig:
    output_dir: Path
    num_roots: int = 128
    num_processes: int = 16
    max_candidate_roots: int = 8192
    candidate_batch_size: int = 8
    min_nonzero_reward_ratio: float = 0.50
    target_abs_threshold: float = 1.0
    history_hops_min: int = 1
    history_hops_max: int = 200
    history_max_total_steps: int = 20_000
    seed: int = 2027
    nonzero_eps: float = DEFAULT_NONZERO_EPS
    overwrite: bool = True

    @property
    def storage_dir(self) -> Path:
        return self.output_dir / "stored_roots"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "GV3_Agent" / "rootTests"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _clean_generated_outputs(cfg: RootTestingConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(cfg.overwrite):
        return

    if cfg.storage_dir.exists():
        shutil.rmtree(cfg.storage_dir)

    for pattern in (
        "root_*_trace.csv",
        "root_*_selected_action.csv",
        "all_selected_actions.csv",
        "trace_test_summary.csv",
        "trace_test_summary.json",
        "ratio_test_summary.csv",
        "ratio_test_summary.json",
        "root_testing_config.json",
    ):
        for path in cfg.output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _is_nonzero(value: Any, eps: float) -> bool:
    return abs(float(value or 0.0)) > float(eps)


def create_roots(cfg: RootTestingConfig) -> list[dict[str, Any]]:
    """Generate or load the controller roots used by these tests.

    Default behavior creates 128 controller roots with 16 local processes. The
    root storage layer gives each worker an independent seed and root-id range.
    """

    _clean_generated_outputs(cfg)
    _write_json(cfg.output_dir / "root_testing_config.json", asdict(cfg))

    manifest = cfg.storage_dir / "manifest.jsonl"
    if manifest.exists() and not bool(cfg.overwrite):
        records = load_stored_roots(cfg.storage_dir)
    else:
        storage_cfg = build_storage_config(
            output_dir=cfg.storage_dir,
            num_roots=int(cfg.num_roots),
            max_candidate_roots=int(cfg.max_candidate_roots),
            candidate_batch_size=int(cfg.candidate_batch_size),
            min_nonzero_target_ratio=float(cfg.min_nonzero_reward_ratio),
            nonzero_eps=float(cfg.nonzero_eps),
            target_abs_threshold=float(cfg.target_abs_threshold),
            history_hops_min=int(cfg.history_hops_min),
            history_hops_max=int(cfg.history_hops_max),
            history_max_total_steps=int(cfg.history_max_total_steps),
            shard_size=max(1, min(512, int(cfg.num_roots))),
            seed=int(cfg.seed),
            start_player="adversary",
            root_player_filter="controller",
            include_history_trace_logs=True,
            allow_duplicate_history_fallback=False,
            num_processes=int(cfg.num_processes),
        )
        generate_and_store_roots(storage_cfg)
        records = load_stored_roots(cfg.storage_dir)

    controller_records = [
        record
        for record in records
        if str(record.get("root_player", "")) == "controller"
    ]
    return controller_records[: int(cfg.num_roots)]


def _trace_csv_path(cfg: RootTestingConfig, record: dict[str, Any]) -> Path:
    return cfg.output_dir / f"root_{int(record['root_id']):06d}_trace.csv"


def _selected_action_csv_path(cfg: RootTestingConfig, record: dict[str, Any]) -> Path:
    return cfg.output_dir / f"root_{int(record['root_id']):06d}_selected_action.csv"


def _action_row(record: dict[str, Any], eps: float, target_abs_threshold: float) -> dict[str, Any]:
    target = float(record.get("target_value", 0.0))
    reward = float(record.get("best_reward", 0.0))
    threshold = float(record.get("target_abs_threshold", target_abs_threshold))
    return {
        "root_id": int(record.get("root_id", -1)),
        "root_player": str(record.get("root_player", "")),
        "root_depth": int(record.get("root_depth", 0)),
        "history_hops": int(record.get("history_hops", 0)),
        "target_value": target,
        "best_action_index": int(record.get("best_action_index", -1)),
        "best_action_repr": str(record.get("best_action_repr", "")),
        "best_reward": reward,
        "best_discount": float(record.get("best_discount", 1.0)),
        "best_bootstrap": float(record.get("best_bootstrap", 0.0)),
        "best_child_cost": float(record.get("best_child_cost", 0.0)),
        "best_child_time": float(record.get("best_child_time", 0.0)),
        "is_nonzero_target": _is_nonzero(target, eps),
        "is_nonzero_reward": _is_nonzero(reward, eps),
        "is_large_abs_target": abs(target) >= threshold,
        "is_large_abs_reward": abs(reward) >= threshold,
        "target_reward_abs_diff": abs(target - reward),
    }


def create_logs(
    records: list[dict[str, Any]],
    cfg: RootTestingConfig,
) -> dict[int, dict[str, Path]]:
    """Write per-root debug traces and selected Bellman action logs."""

    if not records:
        raise RuntimeError("no root records available for log creation")

    paths: dict[int, dict[str, Path]] = {}
    action_rows: list[dict[str, Any]] = []

    for record in records:
        root_id = int(record["root_id"])
        trace_rows = [dict(row) for row in (record.get("history_trace_logs") or [])]
        if not trace_rows:
            raise RuntimeError(
                f"root_id={root_id} has no history_trace_logs; "
                "use history_hops_min >= 1 and include_history_trace_logs=True"
            )

        trace_path = _trace_csv_path(cfg, record)
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DNNMCTSIterationLogger.FIELDS)
            writer.writeheader()
            for row in trace_rows:
                writer.writerow({field: row.get(field, "") for field in DNNMCTSIterationLogger.FIELDS})

        action_path = _selected_action_csv_path(cfg, record)
        action_row = _action_row(record, cfg.nonzero_eps, cfg.target_abs_threshold)
        with action_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ACTION_FIELDS)
            writer.writeheader()
            writer.writerow({field: action_row.get(field, "") for field in ACTION_FIELDS})

        action_rows.append(action_row)
        paths[root_id] = {
            "trace_csv": trace_path,
            "selected_action_csv": action_path,
        }

    all_actions = cfg.output_dir / "all_selected_actions.csv"
    with all_actions.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ACTION_FIELDS)
        writer.writeheader()
        for row in action_rows:
            writer.writerow({field: row.get(field, "") for field in ACTION_FIELDS})

    return paths


def run_test_trace(
    trace_paths: dict[int, dict[str, Path]],
    cfg: RootTestingConfig,
) -> dict[str, Any]:
    """Run existing GV3 semantic trace validation on every root trace CSV."""

    failures: list[dict[str, Any]] = []
    traces_checked = 0
    adv_actions_checked = 0

    for root_id, paths in sorted(trace_paths.items()):
        trace_csv = paths["trace_csv"]
        try:
            root_traces, root_adv_actions = _validate_trace_csv(trace_csv)
            traces_checked += int(root_traces)
            adv_actions_checked += int(root_adv_actions)
        except Exception as exc:
            failures.append(
                {
                    "root_id": int(root_id),
                    "trace_csv": str(trace_csv),
                    "error": repr(exc),
                }
            )

    summary = {
        "roots_checked": int(len(trace_paths)),
        "trace_chains_checked": int(traces_checked),
        "adv_actions_checked": int(adv_actions_checked),
        "failures": failures,
        "passed": not failures,
    }
    _write_json(cfg.output_dir / "trace_test_summary.json", summary)
    _write_single_row_csv(
        cfg.output_dir / "trace_test_summary.csv",
        {
            "roots_checked": summary["roots_checked"],
            "trace_chains_checked": summary["trace_chains_checked"],
            "adv_actions_checked": summary["adv_actions_checked"],
            "failures": len(failures),
            "passed": bool(summary["passed"]),
        },
    )

    if failures:
        raise RuntimeError(f"{len(failures)} root trace validation(s) failed")
    return summary


def run_test_ratio(
    records: list[dict[str, Any]],
    cfg: RootTestingConfig,
) -> dict[str, Any]:
    """Confirm controller-only records and large-absolute reward/target ratio."""

    if not records:
        raise RuntimeError("no records available for ratio test")

    non_controller_ids = [
        int(record.get("root_id", -1))
        for record in records
        if str(record.get("root_player", "")) != "controller"
    ]
    action_rows = [
        _action_row(record, cfg.nonzero_eps, cfg.target_abs_threshold)
        for record in records
    ]
    large_reward_count = sum(1 for row in action_rows if bool(row["is_large_abs_reward"]))
    large_target_count = sum(1 for row in action_rows if bool(row["is_large_abs_target"]))
    reward_ratio = float(large_reward_count) / float(len(records))
    target_ratio = float(large_target_count) / float(len(records))
    inconsistent_reward_target_ids = [
        int(row["root_id"])
        for row in action_rows
        if float(row["target_reward_abs_diff"]) > 1e-6
    ]

    passed = (
        not non_controller_ids
        and not inconsistent_reward_target_ids
        and reward_ratio + 1e-12 >= float(cfg.min_nonzero_reward_ratio)
    )
    summary = {
        "num_records": int(len(records)),
        "controller_records": int(len(records) - len(non_controller_ids)),
        "large_abs_reward_count": int(large_reward_count),
        "large_abs_target_count": int(large_target_count),
        "large_abs_reward_ratio": float(reward_ratio),
        "large_abs_target_ratio": float(target_ratio),
        "required_large_abs_reward_ratio": float(cfg.min_nonzero_reward_ratio),
        "target_abs_threshold": float(cfg.target_abs_threshold),
        "non_controller_root_ids": non_controller_ids,
        "inconsistent_reward_target_ids": inconsistent_reward_target_ids,
        "passed": bool(passed),
    }
    _write_json(cfg.output_dir / "ratio_test_summary.json", summary)
    _write_single_row_csv(
        cfg.output_dir / "ratio_test_summary.csv",
        {
            "num_records": summary["num_records"],
            "controller_records": summary["controller_records"],
            "large_abs_reward_count": summary["large_abs_reward_count"],
            "large_abs_target_count": summary["large_abs_target_count"],
            "large_abs_reward_ratio": summary["large_abs_reward_ratio"],
            "large_abs_target_ratio": summary["large_abs_target_ratio"],
            "required_large_abs_reward_ratio": summary["required_large_abs_reward_ratio"],
            "target_abs_threshold": summary["target_abs_threshold"],
            "non_controller_roots": len(non_controller_ids),
            "inconsistent_reward_target_roots": len(inconsistent_reward_target_ids),
            "passed": bool(passed),
        },
    )

    if non_controller_ids:
        raise RuntimeError(f"found non-controller roots: {non_controller_ids[:20]}")
    if inconsistent_reward_target_ids:
        raise RuntimeError(
            "target_value and best_reward differ for no-bootstrap records: "
            f"{inconsistent_reward_target_ids[:20]}"
        )
    if reward_ratio + 1e-12 < float(cfg.min_nonzero_reward_ratio):
        raise RuntimeError(
            "large-absolute reward ratio below requirement: "
            f"{reward_ratio:.4f} < {cfg.min_nonzero_reward_ratio:.4f}"
        )
    return summary


def parse_args() -> RootTestingConfig:
    parser = argparse.ArgumentParser(description="Generate and validate GV3 ModelSearchBed controller roots.")
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--num-roots", type=int, default=128)
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--max-candidate-roots", type=int, default=8192)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--min-nonzero-reward-ratio", type=float, default=0.50)
    parser.add_argument(
        "--target-abs-threshold",
        type=float,
        default=1.0,
        help="Require this fraction of roots to have abs(target_value) >= this threshold.",
    )
    parser.add_argument("--history-hops-min", type=int, default=1)
    parser.add_argument("--history-hops-max", type=int, default=200)
    parser.add_argument("--history-max-total-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--nonzero-eps", type=float, default=DEFAULT_NONZERO_EPS)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if int(args.num_roots) <= 0:
        raise ValueError("--num-roots must be > 0")
    if int(args.num_processes) <= 0:
        raise ValueError("--num-processes must be > 0")
    if int(args.max_candidate_roots) < int(args.num_roots):
        raise ValueError("--max-candidate-roots must be >= --num-roots")
    if not (0.0 <= float(args.min_nonzero_reward_ratio) <= 1.0):
        raise ValueError("--min-nonzero-reward-ratio must be in [0, 1]")
    if float(args.target_abs_threshold) < 0.0:
        raise ValueError("--target-abs-threshold must be >= 0")
    if int(args.history_hops_min) < 1:
        raise ValueError("--history-hops-min must be >= 1 so trace CSVs are non-empty")
    if int(args.history_hops_max) < int(args.history_hops_min):
        raise ValueError("--history-hops-max must be >= --history-hops-min")

    return RootTestingConfig(
        output_dir=Path(args.output_dir).expanduser(),
        num_roots=int(args.num_roots),
        num_processes=int(args.num_processes),
        max_candidate_roots=int(args.max_candidate_roots),
        candidate_batch_size=int(args.candidate_batch_size),
        min_nonzero_reward_ratio=float(args.min_nonzero_reward_ratio),
        target_abs_threshold=float(args.target_abs_threshold),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_max_total_steps=int(args.history_max_total_steps),
        seed=int(args.seed),
        nonzero_eps=float(args.nonzero_eps),
        overwrite=not bool(args.reuse_existing),
    )


def main() -> None:
    cfg = parse_args()
    records = create_roots(cfg)
    trace_paths = create_logs(records, cfg)
    trace_summary = run_test_trace(trace_paths, cfg)
    ratio_summary = run_test_ratio(records, cfg)
    print(
        "root tests passed: "
        f"roots={len(records)}, "
        f"trace_chains={trace_summary['trace_chains_checked']}, "
        f"large_abs_reward_ratio={ratio_summary['large_abs_reward_ratio']:.4f}, "
        f"target_abs_threshold={ratio_summary['target_abs_threshold']:.4f}, "
        f"output_dir={cfg.output_dir}"
    )


if __name__ == "__main__":
    main()
