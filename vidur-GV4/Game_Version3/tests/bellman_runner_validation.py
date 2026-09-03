from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4_adv_226"
    / "V_iter_xl_50/hgb_sq_47leaf_850iter/Model_Version47/v4_adv_hgb_wrapper.joblib"
)
DEFAULT_OUT_DIR = REPO_ROOT / "simulator_output/GV3_Agent/native_alignment_tests/bellman_runner_validation"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "state_number",
        "player",
        "canon_valid_action_number",
        "python_reward",
        "python_discounted_boostrap",
        "cpp_reward",
        "cpp__discounted_boostrap",
        "best_action",
        "python_best_action",
        "cpp_best_action",
        "python_q",
        "cpp_q",
        "reward_abs_diff",
        "discounted_bootstrap_abs_diff",
        "q_abs_diff",
        "action_repr",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, float):
                    value = f"{value:.17g}" if math.isfinite(value) else str(value)
                out[key] = value
            writer.writerow(out)


def _prepare_native_hgb(model_path: Path, out_dir: Path, *, force_build: bool) -> tuple[Any, Path]:
    from vidur.bellman_v4_adv_2000k_multiprocess.runnerCPP import (
        _export_hgb_to_native_text,
        _import_native_cpp,
    )

    native = _import_native_cpp(force_build=bool(force_build))
    model = joblib.load(model_path)
    export_path = out_dir / "v4_adv_hgb_native_export.tsv"
    _export_hgb_to_native_text(model, export_path)
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(export_path))
    return runtime, export_path


def _cfg_payload_for_cpp(cfg: Any, simulator: Any) -> dict[str, Any]:
    from vidur.Game_Version3.DNN.native_selfplay import (
        _cfg_payload,
        attach_execution_predictor_payload,
    )

    payload = _cfg_payload(cfg.to_pipeline_cfg(), torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["use_model_bootstrap"] = True
    payload["native_search_mode"] = "depth_one"
    payload["root_dirichlet_noise_enabled"] = False
    return payload


def _native_depth1_out(
    *,
    native_runtime: Any,
    model_version: int,
    bundle: Any,
    expanded: Any,
    cfg_payload: dict[str, Any],
    state_number: int,
    seed: int,
) -> dict[str, Any]:
    from vidur.Game_Version3.tests import native_logger_tests as nlt
    from vidur.bellman_v4_adv_2000k_multiprocess.runnerCPP import _import_native_cpp

    native = _import_native_cpp(force_build=False)
    return native.search_mcts_hgb226(
        native_runtime,
        int(model_version),
        nlt._native_state_payload(bundle.env, expanded.search_state),
        cfg_payload,
        1,
        str(expanded.player),
        0,
        0,
        int(state_number),
        -1,
        int(seed) + int(state_number),
        False,
        False,
        "",
        "",
    )


def run_validation(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    num_states: int = 10,
    history_hops_min: int = 0,
    history_hops_max: int = 100,
    history_seed: int = 202633,
    model_version: int = 47,
    tolerance: float = 1e-6,
    force_build_native: bool = False,
) -> Path:
    from vidur.Game_Version3.Model_Tester.config import DEFAULT_MODEL_TESTER_CONFIG
    from vidur.bellman_v4_adv_2000k_multiprocess import runner as py_runner

    model_path = Path(model_path).expanduser()
    out_dir = Path(out_dir).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")

    harness_model_path = model_path
    loaded_model = joblib.load(model_path)
    if not callable(getattr(loaded_model, "infer_from_inputs", None)):
        from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper

        wrapped = V4AdvHGBWrapper(
            loaded_model,
            feature_dim=226,
            model_tag=f"validation_wrapped:{model_path.name}",
        )
        harness_model_path = out_dir / "arena_model" / "v4_adv_hgb_wrapper.joblib"
        harness_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(wrapped, harness_model_path, compress=3)

    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind="classical_joblib",
        model_checkpoint_path=str(harness_model_path),
        output_dir=str(out_dir / "python_runner_tmp"),
        num_games=int(num_states),
        game_id_start=12_500_000,
        environment_lang="python",
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        history_seed=int(history_seed),
        history_hops_unique=True,
        history_hops_force_zero=True,
        bootstrap_model_version=int(model_version),
        write_arena_game_logs=False,
        write_model_action_detail_logs=False,
    )

    native_runtime, export_path = _prepare_native_hgb(model_path, out_dir, force_build=bool(force_build_native))
    bundle = py_runner._build_bundle(cfg)
    cfg_payload = _cfg_payload_for_cpp(cfg, bundle.simulator)
    history_hops = cfg.sample_history_hops()

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for state_number in range(int(num_states)):
            base_snapshot, base_stats, base_player, _base_depth = py_runner._prepare_base_state_for_game(
                cfg=cfg,
                bundle=bundle,
                game_id=int(cfg.game_id_start) + int(state_number),
                history_nontrivial_hops=int(history_hops[state_number]),
                cycle_file_logger=None,
            )
            state = bundle.runner.env.clone_state_from_snapshot(base_snapshot, base_stats)
            player, expanded = py_runner._align_player_to_valid_actions(
                bundle=bundle,
                state=state,
                player=str(base_player),
                pending_adv_pre_ctrl_snapshot=None,
                pending_adv_pre_ctrl_stats=None,
            )
            if not expanded.valid_indices:
                continue

            _py_action, py_info = py_runner._select_model_depth1_action(
                bundle=bundle,
                cfg=cfg,
                expanded=expanded,
                model=bundle.model,
            )
            py_ranked = list(py_info.get("candidate_ranked_rows") or [])
            if not py_ranked:
                continue
            py_by_idx = {int(r["canon_action_index"]): r for r in py_ranked}
            py_best_idx = int(py_ranked[0]["canon_action_index"])

            native_out = _native_depth1_out(
                native_runtime=native_runtime,
                model_version=int(model_version),
                bundle=bundle,
                expanded=expanded,
                cfg_payload=cfg_payload,
                state_number=int(state_number),
                seed=int(history_seed),
            )
            q_values = [float(x) for x in list(native_out.get("root_action_values", []) or [])]
            rewards = [float(x) for x in list(native_out.get("root_action_rewards", []) or [])]
            discounts = [float(x) for x in list(native_out.get("root_action_discounts", []) or [])]
            bootstraps = [float(x) for x in list(native_out.get("root_action_bootstraps", []) or [])]
            cpp_best_idx = int(native_out.get("best_action_index", -1))

            for canon_idx in sorted(py_by_idx):
                py_row = py_by_idx[int(canon_idx)]
                py_reward = float(py_row["immediate_reward"])
                py_discounted_bootstrap = float(py_row["discount"]) * float(py_row["bootstrap_value"])
                py_q = float(py_row["q_value"])

                cpp_reward = rewards[canon_idx] if 0 <= canon_idx < len(rewards) else float("nan")
                cpp_discount = discounts[canon_idx] if 0 <= canon_idx < len(discounts) else float("nan")
                cpp_bootstrap = bootstraps[canon_idx] if 0 <= canon_idx < len(bootstraps) else float("nan")
                cpp_discounted_bootstrap = float(cpp_discount) * float(cpp_bootstrap)
                cpp_q = q_values[canon_idx] if 0 <= canon_idx < len(q_values) else float("nan")

                reward_abs_diff = abs(py_reward - cpp_reward)
                discounted_abs_diff = abs(py_discounted_bootstrap - cpp_discounted_bootstrap)
                q_abs_diff = abs(py_q - cpp_q)
                py_best = int(canon_idx) == int(py_best_idx)
                cpp_best = int(canon_idx) == int(cpp_best_idx)
                if (
                    reward_abs_diff > float(tolerance)
                    or discounted_abs_diff > float(tolerance)
                    or q_abs_diff > float(tolerance)
                    or py_best != cpp_best
                ):
                    failures.append(
                        f"state={state_number} player={player} action={canon_idx} "
                        f"reward_diff={reward_abs_diff:.3g} disc_boot_diff={discounted_abs_diff:.3g} "
                        f"q_diff={q_abs_diff:.3g} py_best={py_best} cpp_best={cpp_best}"
                    )

                rows.append(
                    {
                        "state_number": int(state_number),
                        "player": str(player),
                        "canon_valid_action_number": int(canon_idx),
                        "python_reward": float(py_reward),
                        "python_discounted_boostrap": float(py_discounted_bootstrap),
                        "cpp_reward": float(cpp_reward),
                        "cpp__discounted_boostrap": float(cpp_discounted_bootstrap),
                        "best_action": bool(py_best and cpp_best),
                        "python_best_action": bool(py_best),
                        "cpp_best_action": bool(cpp_best),
                        "python_q": float(py_q),
                        "cpp_q": float(cpp_q),
                        "reward_abs_diff": float(reward_abs_diff),
                        "discounted_bootstrap_abs_diff": float(discounted_abs_diff),
                        "q_abs_diff": float(q_abs_diff),
                        "action_repr": str(py_row.get("action_repr", "")),
                    }
                )
    finally:
        try:
            py_runner._safe_close_simulator(bundle.simulator)
        except Exception:
            pass

    csv_path = out_dir / "bellman_runner_validation.csv"
    _write_csv(csv_path, rows)
    (out_dir / "native_model_export_path.txt").write_text(str(export_path) + "\n", encoding="utf-8")
    if failures:
        preview = "\n".join(failures[:20])
        raise AssertionError(f"Bellman runner C++ depth-1 validation failed ({len(failures)} mismatches):\n{preview}\nCSV: {csv_path}")
    return csv_path


def test_cpp_depth1_bellman_runner_matches_python_depth1() -> None:
    run_validation()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Python depth-1 Bellman runner against native C++ depth-1 evaluation.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num-states", type=int, default=10)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=100)
    parser.add_argument("--history-seed", type=int, default=202633)
    parser.add_argument("--model-version", type=int, default=47)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--force-build-native", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    csv_path = run_validation(
        model_path=Path(args.model_path),
        out_dir=Path(args.output_dir),
        num_states=int(args.num_states),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        model_version=int(args.model_version),
        tolerance=float(args.tolerance),
        force_build_native=bool(args.force_build_native),
    )
    print(f"[bellman-runner-validation] wrote {csv_path}")


if __name__ == "__main__":
    main()
