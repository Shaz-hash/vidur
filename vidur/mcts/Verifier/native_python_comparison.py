# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
    Command to run :

    cd /home/shazer/Desktop/Research/Vidur/vidur
    PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.Verifier.native_python_comparison \
    --history-csv simulator_output/mcts_dnn_logs/gen_000154/mcts_root_p00_gen00.csv \
    --prior-mode model \
    --simulations 1000 \
    --native-device cuda:0

    OR 
    cd /home/shazer/Desktop/Research/Vidur/vidur
    PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.Verifier.native_python_comparison \
    --prior-mode model \
    --simulations 1000 \
    --native-device cpu

    OR FOR specific checkpoint :
    cd /home/shazer/Desktop/Research/Vidur/vidur
    PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.Verifier.native_python_comparison \
    --checkpoint simulator_output/mcts_dnn_checkpoints/best.pt \
    --prior-mode model \
    --simulations 1000 \
    --native-device cuda:0


"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from vidur.mcts.DNN.export_torchscript import export_torchscript_artifacts
from vidur.mcts.Verifier.check_mcts_dnn_distribution import (
    _DummyModel,
    _build_default_cfg,
    _load_model,
    load_history_rows,
    replay_history_to_state_after_last_action,
)
from vidur.mcts.alphaZeroParrallel import _build_env_and_simulator
from vidur.mcts.mctsDNN import VidurMCTS
from vidur.mcts.native.python.infer_client import (
    NativeTorchScriptModelAdapter,
    TorchScriptInferClient,
    build_native_infer_service_runtime,
    start_torchscript_infer_service,
)


COMPARE_FIELDS = [
    "game_id",
    "root_id",
    "root_depth",
    "root_player",
    "num_simulations",
    "prior_mode",
    "history_csv",
    "native_infer_mode",
    "native_device",
    "python_model_root_value_controller",
    "native_model_root_value_controller",
    "abs_model_root_value_diff",
    "python_mcts_root_value_controller",
    "native_mcts_root_value_controller",
    "abs_mcts_root_value_diff",
    "python_best_action_index",
    "native_best_action_index",
    "best_action_match",
    "python_model_root_prior_json",
    "native_model_root_prior_json",
    "model_root_prior_l1_diff",
    "model_root_prior_max_abs_diff",
    "python_normalized_root_prior_json",
    "native_normalized_root_prior_json",
    "normalized_root_prior_l1_diff",
    "normalized_root_prior_max_abs_diff",
    "python_mcts_root_prior_json",
    "native_mcts_root_prior_json",
    "mcts_root_prior_l1_diff",
    "mcts_root_prior_max_abs_diff",
    "python_valid_action_mask_json",
    "native_valid_action_mask_json",
]


def _clone_state(state: Any) -> Any:
    if hasattr(state, "fork"):
        try:
            return state.fork(flag=False)
        except TypeError:
            return state.fork()
    raise RuntimeError("State object does not provide fork() for comparison cloning")


def _payload_for_root(mcts: VidurMCTS, root_state: Any) -> Tuple[float, List[float], List[float], List[bool], List[float], Optional[int]]:
    root = mcts._root
    if root is None:
        raise RuntimeError("MCTS root is missing after search")

    if (
        root.nn_value_controller is None
        or root.nn_priors is None
        or root.nn_valid_mask is None
    ):
        return mcts._compute_root_log_payload_without_nn(root=root, root_state=root_state)

    return mcts._compute_root_log_payload_from_cache(root=root)


def _vector_diffs(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    n = max(len(a), len(b))
    aa = [float(a[i]) if i < len(a) else 0.0 for i in range(n)]
    bb = [float(b[i]) if i < len(b) else 0.0 for i in range(n)]
    l1 = sum(abs(x - y) for x, y in zip(aa, bb))
    max_abs = max((abs(x - y) for x, y in zip(aa, bb)), default=0.0)
    return float(l1), float(max_abs)


def _write_compare_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    full = {k: row.get(k, "") for k in COMPARE_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COMPARE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(full)


def _wait_for_service(addr: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + float(timeout_s)
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            c = TorchScriptInferClient(addr=str(addr))
            if c.ping():
                c.close()
                return
            c.close()
        except Exception as exc:
            last_err = exc
        time.sleep(0.2)
    raise RuntimeError(f"inference service did not become ready at {addr}: {last_err!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-csv", type=str, default=None, help="Optional mcts_root*.csv history file")
    ap.add_argument("--prior-mode", choices=["model", "uniform"], default="model")
    ap.add_argument("--discount-factor", type=float, default=0.98)
    ap.add_argument("--simulations", type=int, default=1000)
    ap.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint for model mode")
    ap.add_argument("--prefill-profile", type=str, default="simulator_output/prefill_profile.csv")
    ap.add_argument("--use-virtual-env", action="store_true", default=True)
    ap.add_argument("--align-branching-roots", action="store_true", default=True)
    ap.add_argument("--max-forced-hops", type=int, default=20)
    ap.add_argument("--enum-max-samples", type=int, default=10000)
    ap.add_argument("--game-id", type=int, default=0)
    ap.add_argument("--root-id", type=int, default=0)
    ap.add_argument("--root-depth", type=int, default=0)
    ap.add_argument("--root-player", type=str, default="adversary")
    ap.add_argument("--root-node-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native-device", type=str, default="cuda:0", help="cpu or cuda:0 for torchscript_service")
    ap.add_argument("--infer-service-addr", type=str, default="127.0.0.1:50231")
    ap.add_argument("--infer-service-impl", type=str, default="cpp")
    ap.add_argument("--infer-max-batch", type=int, default=256)
    ap.add_argument("--infer-max-wait-us", type=int, default=2000)
    ap.add_argument(
        "--out-csv",
        type=str,
        default="simulator_output/mcts_dnn_logs/native_python_comparison.csv",
    )
    args = ap.parse_args()

    if not (0.0 <= float(args.discount_factor) <= 1.0):
        raise SystemExit("--discount-factor must be in [0, 1]")

    cfg = _build_default_cfg(
        simulations=int(args.simulations),
        prefill_profile_path=str(args.prefill_profile),
    )
    _, env, _, explore_cfg = _build_env_and_simulator(
        cfg,
        use_virtual_env=bool(args.use_virtual_env),
    )

    setattr(explore_cfg, "prior_value_mode", str(args.prior_mode))
    setattr(explore_cfg, "discount_factor", float(args.discount_factor))
    setattr(explore_cfg, "root_dirichlet_noise_enabled", False)
    setattr(explore_cfg, "native_mcts_enabled", False)
    setattr(explore_cfg, "torchscript_full_native_search", False)

    history_csv = ""
    if args.history_csv:
        history_path = Path(args.history_csv)
        if not history_path.exists():
            raise SystemExit(f"History CSV not found: {history_path}")
        rows = load_history_rows(history_path)
        root_state, root_player, last_row = replay_history_to_state_after_last_action(
            env,
            rows,
            align_branching_roots=bool(args.align_branching_roots),
            max_forced_hops=int(args.max_forced_hops),
            max_samples=int(args.enum_max_samples),
        )
        game_id = int(last_row.game_id)
        root_id = int(last_row.root_id) + 1
        root_depth = int(last_row.root_depth) + 1
        root_node_id = int(last_row.root_node_id)
        history_csv = str(history_path)
    else:
        root_state = env.initial_state()
        root_player = str(args.root_player)
        game_id = int(args.game_id)
        root_id = int(args.root_id)
        root_depth = int(args.root_depth)
        root_node_id = int(args.root_node_id)

    device = torch.device("cpu")
    if args.prior_mode == "uniform":
        model: Any = _DummyModel()
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        model = _load_model(ckpt, device=device)

    root_state_py = _clone_state(root_state)
    root_state_native = _clone_state(root_state)

    py_cfg = copy.copy(explore_cfg)
    native_cfg = copy.copy(explore_cfg)

    setattr(py_cfg, "native_mcts_enabled", False)
    setattr(py_cfg, "torchscript_full_native_search", False)
    setattr(native_cfg, "native_mcts_enabled", True)
    setattr(native_cfg, "torchscript_full_native_search", True)

    mcts_py = VidurMCTS(
        env=env,
        explore_cfg=py_cfg,
        log_path=None,
        tree_log_path=None,
        verbose=False,
        complete_log=False,
        rng=random.Random(int(args.seed)),
    )
    mcts_native: Optional[VidurMCTS] = None
    service_proc = None
    service_client: Optional[TorchScriptInferClient] = None
    native_service_runtime = None
    temp_artifacts_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        service_proc = start_torchscript_infer_service(
            addr=str(args.infer_service_addr),
            device=str(args.native_device),
            max_batch=int(args.infer_max_batch),
            max_wait_us=int(args.infer_max_wait_us),
            impl=str(args.infer_service_impl),
        )
        _wait_for_service(str(args.infer_service_addr), timeout_s=45.0)
        service_client = TorchScriptInferClient(addr=str(args.infer_service_addr))
        native_service_runtime = build_native_infer_service_runtime(
            addr=str(args.infer_service_addr),
        )

        native_model: Any
        if args.prior_mode == "model":
            model_version = int(time.time()) % 2_000_000_000
            temp_artifacts_dir = tempfile.TemporaryDirectory(prefix="vidur_native_compare_")
            tmp_dir = Path(temp_artifacts_dir.name)
            if args.checkpoint:
                ckpt_path = Path(args.checkpoint)
            else:
                ckpt_path = tmp_dir / f"compare_model_{model_version}.pt"
                torch.save(
                    {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                    ckpt_path,
                )

            ts_out = export_torchscript_artifacts(
                checkpoint_path=ckpt_path,
                out_dir=tmp_dir,
                model_version=int(model_version),
                device="cpu",
                num_actions_controller=int(cfg.model.num_actions_controller),
                num_actions_adversary=int(cfg.model.num_actions_adversary),
            )
            native_service_runtime.load_models(
                int(model_version),
                str(ts_out.controller_path),
                str(ts_out.adversary_path),
            )
            native_model = NativeTorchScriptModelAdapter(
                runtime=native_service_runtime,
                model_version=int(model_version),
                fallback_model=model,
                fallback_to_python=False,
            )
        else:
            native_model = _DummyModel()
            setattr(native_model, "_native_ts_runtime", native_service_runtime)
            setattr(native_model, "_native_ts_model_version", 0)

        mcts_native = VidurMCTS(
            env=env,
            explore_cfg=native_cfg,
            log_path=None,
            tree_log_path=None,
            verbose=False,
            complete_log=False,
            rng=random.Random(int(args.seed)),
        )

        mcts_py.search_dnn(
            dnn_model=model,
            rootState=root_state_py,
            root_player=str(root_player),
            iterations=int(args.simulations),
            game_id=int(game_id),
            root_id=int(root_id),
            root_node_id_override=int(root_node_id),
            root_depth=int(root_depth),
            root_phase="compare_root",
            cycle_label="python",
        )

        mcts_native.search_dnn(
            dnn_model=native_model,
            rootState=root_state_native,
            root_player=str(root_player),
            iterations=int(args.simulations),
            game_id=int(game_id),
            root_id=int(root_id),
            root_node_id_override=int(root_node_id),
            root_depth=int(root_depth),
            root_phase="compare_root",
            cycle_label="native",
        )

        py_model_v, py_model_prior, py_norm_prior, py_mask, py_mcts_prior, py_best_idx = _payload_for_root(
            mcts_py, root_state_py
        )
        na_model_v, na_model_prior, na_norm_prior, na_mask, na_mcts_prior, na_best_idx = _payload_for_root(
            mcts_native, root_state_native
        )

        model_l1, model_max_abs = _vector_diffs(py_model_prior, na_model_prior)
        norm_l1, norm_max_abs = _vector_diffs(py_norm_prior, na_norm_prior)
        mcts_l1, mcts_max_abs = _vector_diffs(py_mcts_prior, na_mcts_prior)

        row = {
            "game_id": int(game_id),
            "root_id": int(root_id),
            "root_depth": int(root_depth),
            "root_player": str(root_player),
            "num_simulations": int(args.simulations),
            "prior_mode": str(args.prior_mode),
            "history_csv": history_csv,
            "native_infer_mode": "torchscript_service",
            "native_device": str(args.native_device),
            "python_model_root_value_controller": float(py_model_v),
            "native_model_root_value_controller": float(na_model_v),
            "abs_model_root_value_diff": abs(float(py_model_v) - float(na_model_v)),
            "python_mcts_root_value_controller": float(mcts_py._root.mean_value()) if mcts_py._root else math.nan,
            "native_mcts_root_value_controller": float(mcts_native._root.mean_value()) if mcts_native._root else math.nan,
            "abs_mcts_root_value_diff": abs(
                (float(mcts_py._root.mean_value()) if mcts_py._root else 0.0)
                - (float(mcts_native._root.mean_value()) if mcts_native._root else 0.0)
            ),
            "python_best_action_index": "" if py_best_idx is None else int(py_best_idx),
            "native_best_action_index": "" if na_best_idx is None else int(na_best_idx),
            "best_action_match": bool(py_best_idx == na_best_idx),
            "python_model_root_prior_json": json.dumps(list(py_model_prior)),
            "native_model_root_prior_json": json.dumps(list(na_model_prior)),
            "model_root_prior_l1_diff": float(model_l1),
            "model_root_prior_max_abs_diff": float(model_max_abs),
            "python_normalized_root_prior_json": json.dumps(list(py_norm_prior)),
            "native_normalized_root_prior_json": json.dumps(list(na_norm_prior)),
            "normalized_root_prior_l1_diff": float(norm_l1),
            "normalized_root_prior_max_abs_diff": float(norm_max_abs),
            "python_mcts_root_prior_json": json.dumps(list(py_mcts_prior)),
            "native_mcts_root_prior_json": json.dumps(list(na_mcts_prior)),
            "mcts_root_prior_l1_diff": float(mcts_l1),
            "mcts_root_prior_max_abs_diff": float(mcts_max_abs),
            "python_valid_action_mask_json": json.dumps(list(py_mask)),
            "native_valid_action_mask_json": json.dumps(list(na_mask)),
        }

        out_csv = Path(args.out_csv)
        _write_compare_row(out_csv, row)

        print(
            "[OK] "
            f"simulations={args.simulations} "
            f"prior_mode={args.prior_mode} "
            f"best_match={row['best_action_match']} "
            f"mcts_l1={row['mcts_root_prior_l1_diff']:.10f} "
            f"mcts_max_abs={row['mcts_root_prior_max_abs_diff']:.10f} "
            f"root_value_abs={row['abs_mcts_root_value_diff']:.10f} "
            f"native_device={args.native_device} "
            f"out={out_csv}"
        )
    finally:
        mcts_py.close()
        if mcts_native is not None:
            try:
                mcts_native.close()
            except Exception:
                pass
        if service_client is not None:
            try:
                service_client.shutdown()
            except Exception:
                pass
            try:
                service_client.close()
            except Exception:
                pass
        if service_proc is not None:
            try:
                service_proc.terminate()
            except Exception:
                pass
            try:
                service_proc.join(timeout=5.0)
            except Exception:
                pass
        if temp_artifacts_dir is not None:
            temp_artifacts_dir.cleanup()


if __name__ == "__main__":
    main()
