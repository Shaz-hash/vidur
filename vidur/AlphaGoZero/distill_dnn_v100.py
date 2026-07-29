"""One-time DNN v100 bootstrap by distilling classical v100 on v100 replay samples."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from vidur.AlphaGoZero import agz_train_eval_promote as trainer
from vidur.AlphaGoZero.dnn_models import (
    ADVERSARY_ACTION_DIM,
    CONTROLLER_ACTION_DIM,
    PolicyRankMLP,
    ValueResidualMLP,
    export_dnn_to_native,
    fit_policy_dnn,
    fit_value_dnn,
    save_dnn_model,
    write_artifact_metadata,
)
from vidur.AlphaGoZero.durable_transfer import atomic_write_json, utc_now


VALUE_CONFIG = "dnn_value_residual_192_v1"
POLICY_CONFIG = "dnn_policy_rank_192_v1"


def _v100_partition_paths(source_root: Path) -> tuple[list[Path], list[Path], dict[str, int]]:
    feature_paths: list[Path] = []
    policy_paths: list[Path] = []
    counts = {"controller": 0, "adversary": 0}
    for manifest_path in sorted(
        (Path(source_root) / "global_replay" / "partitions").glob("*/*/partition_manifest.json")
    ):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        role = str(data.get("partition_role", ""))
        if role not in counts:
            continue
        role_version = (
            int(data.get("controller_model_version", -1))
            if role == "controller"
            else int(data.get("adversary_model_version", -1))
        )
        if role_version != 100:
            continue
        partition = manifest_path.parent
        feature = partition / "replay_target_runtime_feature_complete.csv"
        policy = partition / "replay_policy_rows.csv"
        if feature.is_file() and policy.is_file():
            feature_paths.append(feature)
            policy_paths.append(policy)
            counts[role] += int(data.get("replay_rows_written", 0))
    return feature_paths, policy_paths, counts


def _teacher_policy_probabilities(
    model: Any,
    features: np.ndarray,
    offsets: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    logits = model.predict(features).astype(np.float32)
    probabilities = np.empty(logits.shape[0], dtype=np.float32)
    for begin, end in offsets:
        root_logits = logits[begin:end].astype(np.float64)
        root_logits -= float(np.max(root_logits))
        exp = np.exp(root_logits)
        total = float(np.sum(exp))
        if not math.isfinite(total) or total <= 0.0:
            probabilities[begin:end] = 1.0 / float(end - begin)
        else:
            probabilities[begin:end] = (exp / total).astype(np.float32)
    return logits, probabilities


def _policy_distillation_metrics(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    target_probabilities: np.ndarray,
    offsets: list[tuple[int, int]],
) -> dict[str, float]:
    top1 = 0
    cross_entropy = 0.0
    for begin, end in offsets:
        teacher_root = teacher_logits[begin:end]
        student_root = student_logits[begin:end].astype(np.float64)
        target = target_probabilities[begin:end].astype(np.float64)
        student_root -= float(np.max(student_root))
        log_norm = math.log(float(np.sum(np.exp(student_root))))
        log_prob = student_root - log_norm
        cross_entropy += -float(np.sum(target * log_prob))
        top1 += int(np.argmax(teacher_root) == np.argmax(student_root))
    roots = max(1, len(offsets))
    return {
        "teacher_top1": float(top1 / roots),
        "cross_entropy": float(cross_entropy / roots),
    }


def _write_model(
    model: Any,
    directory: Path,
    *,
    tag: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_dnn_model(model, directory / "model.joblib")
    export_dnn_to_native(model, directory / "native_model.tsv", model_tag=tag)
    write_artifact_metadata(model, directory / "metadata.json")


def distill(args: argparse.Namespace) -> dict[str, Any]:
    feature_paths, policy_paths, available = _v100_partition_paths(args.source_root)
    if available["controller"] < int(args.controller_roots):
        raise RuntimeError(f"insufficient v100 controller states: {available}")
    if available["adversary"] < int(args.adversary_roots):
        raise RuntimeError(f"insufficient v100 adversary states: {available}")

    sampled, controller_roots, adversary_roots, replay_counts, cache_timings = (
        trainer._stream_sample_state_rows_cached(
            feature_paths,
            seed=int(args.seed),
            max_value_rows=int(args.value_states),
            max_controller_policy_roots=int(args.controller_roots),
            max_adversary_policy_roots=int(args.adversary_roots),
        )
    )
    (Xc, _, _, offc), (Xa, _, _, offa), policy_timings = (
        trainer._collect_policy_training_arrays_for_players_cached(
            policy_paths,
            controller_roots,
            adversary_roots,
            controller_root_cap=int(args.controller_roots),
            adversary_root_cap=int(args.adversary_roots),
        )
    )
    if len(offc) < int(args.controller_roots) or len(offa) < int(args.adversary_roots):
        raise RuntimeError(
            f"insufficient feature-complete policy roots: controller={len(offc)} adversary={len(offa)}"
        )

    Xv, _ = trainer._value_arrays_from_rows(sampled)
    value_teacher = joblib.load(args.classical_value_model)
    controller_teacher = joblib.load(args.classical_controller_policy)
    adversary_teacher = joblib.load(args.classical_adversary_policy)

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="agz_v100_teacher") as executor:
        value_future = executor.submit(value_teacher.predict, Xv)
        controller_future = executor.submit(_teacher_policy_probabilities, controller_teacher, Xc, offc)
        adversary_future = executor.submit(_teacher_policy_probabilities, adversary_teacher, Xa, offa)
        value_targets = np.asarray(value_future.result(), dtype=np.float32)
        controller_teacher_logits, controller_targets = controller_future.result()
        adversary_teacher_logits, adversary_targets = adversary_future.result()

    teacher_value_min_raw = float(np.min(value_targets))
    teacher_value_max_raw = float(np.max(value_targets))
    teacher_value_clipped_count = int(
        np.count_nonzero((value_targets < -50.0) | (value_targets > 0.0))
    )
    # Distillation enforces the declared true-value domain on HGB approximation
    # overshoot. Normal replay-target training remains strict and never clips.
    value_targets = np.clip(value_targets, -50.0, 0.0).astype(np.float32, copy=False)
    if float(np.min(value_targets)) < -50.0 or float(np.max(value_targets)) > 0.0:
        raise AssertionError("distillation value-target clipping failed")

    value_model, value_timing = fit_value_dnn(
        Xv,
        value_targets,
        role="controller",
        seed=int(args.seed) + 1,
        epochs=int(args.epochs),
        batch_size=int(args.value_batch_size),
        lr=float(args.learning_rate),
        torch_threads=int(args.torch_threads),
    )
    value_model.training_metadata.update({
        "bootstrap_version": 100,
        "distilled_from": str(args.classical_value_model),
        "source_replay": str(args.source_root),
        "target_perspective": "controller",
        "incremental_update": False,
    })
    adversary_value = copy.deepcopy(value_model)
    adversary_value.role = "adversary"
    adversary_value.training_metadata = dict(value_model.training_metadata)
    adversary_value.training_metadata["copied_from_controller_value_v100"] = True

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agz_v100_policy") as executor:
        controller_future = executor.submit(
            fit_policy_dnn,
            Xc,
            controller_targets,
            offc,
            role="controller",
            action_dim=CONTROLLER_ACTION_DIM,
            seed=int(args.seed) + 11,
            epochs=int(args.epochs),
            root_batch_size=int(args.policy_root_batch_size),
            lr=float(args.learning_rate),
            torch_threads=int(args.torch_threads),
        )
        adversary_future = executor.submit(
            fit_policy_dnn,
            Xa,
            adversary_targets,
            offa,
            role="adversary",
            action_dim=ADVERSARY_ACTION_DIM,
            seed=int(args.seed) + 17,
            epochs=int(args.epochs),
            root_batch_size=int(args.policy_root_batch_size),
            lr=float(args.learning_rate),
            torch_threads=int(args.torch_threads),
        )
        controller_policy, controller_timing = controller_future.result()
        adversary_policy, adversary_timing = adversary_future.result()

    for model, source in (
        (controller_policy, args.classical_controller_policy),
        (adversary_policy, args.classical_adversary_policy),
    ):
        model.training_metadata.update({
            "bootstrap_version": 100,
            "distilled_from": str(source),
            "source_replay": str(args.source_root),
            "incremental_update": False,
        })

    output = Path(args.output_root)
    model_root = output / "models" / "Model_Version100"
    if model_root.exists():
        if not args.replace:
            raise FileExistsError(model_root)
        shutil.rmtree(model_root)
    controller_value_dir = model_root / "controller_value" / VALUE_CONFIG
    adversary_value_dir = model_root / "adversary_value" / VALUE_CONFIG
    controller_policy_dir = model_root / "controller_prior" / POLICY_CONFIG
    adversary_policy_dir = model_root / "adversary_prior" / POLICY_CONFIG
    _write_model(value_model, controller_value_dir, tag="agz_dnn_controller_value_v100")
    _write_model(adversary_value, adversary_value_dir, tag="agz_dnn_adversary_value_v100")
    _write_model(controller_policy, controller_policy_dir, tag="agz_dnn_controller_prior_v100")
    _write_model(adversary_policy, adversary_policy_dir, tag="agz_dnn_adversary_prior_v100")
    parity_script = Path(__file__).parent / "test_and_analysis" / "test_dnn_native_parity.py"
    parity: dict[str, Any] = {}
    for role, value_path in (
        ("controller", controller_value_dir / "model.joblib"),
        ("adversary", adversary_value_dir / "model.joblib"),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(parity_script),
                "--value-model",
                str(value_path),
                "--controller-policy-model",
                str(controller_policy_dir / "model.joblib"),
                "--adversary-policy-model",
                str(adversary_policy_dir / "model.joblib"),
                "--rows",
                "256",
                "--tolerance",
                "1e-4",
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        parity[role] = json.loads(completed.stdout)

    value_prediction = value_model.predict(Xv)
    controller_prediction = controller_policy.predict(Xc)
    adversary_prediction = adversary_policy.predict(Xa)
    result = {
        "model_family": "dnn",
        "model_version": 100,
        "created_at_utc": utc_now(),
        "source_v100_states_available": available,
        "sampled_value_states": int(Xv.shape[0]),
        "sampled_controller_policy_roots": int(len(offc)),
        "sampled_adversary_policy_roots": int(len(offa)),
        "value_teacher_min_raw": teacher_value_min_raw,
        "value_teacher_max_raw": teacher_value_max_raw,
        "value_teacher_clipped_count": teacher_value_clipped_count,
        "value_teacher_rmse": float(np.sqrt(np.mean((value_prediction - value_targets) ** 2))),
        "controller_policy": _policy_distillation_metrics(
            controller_teacher_logits,
            controller_prediction,
            controller_targets,
            offc,
        ),
        "adversary_policy": _policy_distillation_metrics(
            adversary_teacher_logits,
            adversary_prediction,
            adversary_targets,
            offa,
        ),
        "value_training": value_timing,
        "controller_policy_training": controller_timing,
        "adversary_policy_training": adversary_timing,
        "cache_timings": cache_timings,
        "policy_timings": policy_timings,
        "native_ready": True,
        "dnn_parity": parity,
        "eval_status": "bootstrap",
        "controller_value_model_path": str(controller_value_dir / "model.joblib"),
        "adversary_value_model_path": str(adversary_value_dir / "model.joblib"),
        "controller_prior_model_path": str(controller_policy_dir / "model.joblib"),
        "adversary_prior_model_path": str(adversary_policy_dir / "model.joblib"),
        "value_model_path": str(controller_value_dir / "model.joblib"),
        "controller_model_version": 100,
        "adversary_model_version": 100,
    }
    atomic_write_json(model_root / "candidate_manifest.json", result)
    atomic_write_json(output / "models" / "current_model.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classical-value-model", type=Path, required=True)
    parser.add_argument("--classical-controller-policy", type=Path, required=True)
    parser.add_argument("--classical-adversary-policy", type=Path, required=True)
    parser.add_argument("--value-states", type=int, default=500_000)
    parser.add_argument("--controller-roots", type=int, default=250_000)
    parser.add_argument("--adversary-roots", type=int, default=250_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--value-batch-size", type=int, default=4096)
    parser.add_argument("--policy-root-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--torch-threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(distill(parse_args()), indent=2, sort_keys=True))
