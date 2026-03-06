from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import torch

from .collector_parallel import collect_samples_parallel
from .config import LinearPipelineConfig, round_dir
from .model import LinearValueModel
from .rollout_eval import run_greedy_rollouts
from .trainer import train_value_model


def _append_summary(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = [
        "round",
        "train_samples",
        "eval_samples",
        "train_mse",
        "train_mae",
        "train_r2",
        "eval_mse",
        "eval_mae",
        "eval_r2",
        "train_npz",
        "eval_npz",
        "train_meta_csv",
        "eval_meta_csv",
        "model_ckpt",
        "rollout_summary_csv",
        "device",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _save_epoch_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_mse", "train_mae", "train_r2", "eval_mse", "eval_mae", "eval_r2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, 0.0) for k in fieldnames})


def _save_rollout_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trace_id", "history_hop", "steps", "terminal", "trace_csv"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def run_self_improvement(cfg: LinearPipelineConfig) -> None:
    out_dir = Path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    if device.type.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError(f"requested device={cfg.device}, but CUDA is not available")

    model = LinearValueModel(num_features=30, zero_init=True).to(device)
    model.eval()

    cfg_json = out_dir / "config.json"
    if not cfg_json.exists():
        cfg_json.write_text(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o)), indent=2), encoding="utf-8")

    summary_csv = out_dir / "summary.csv"

    for round_idx in range(int(cfg.rounds)):
        rd = round_dir(cfg, round_idx)
        rd.mkdir(parents=True, exist_ok=True)

        model_for_collection = rd / f"model_for_collection_round_{round_idx:03d}.pt"
        torch.save(
            {
                "num_features": 30,
                "model_state": model.state_dict(),
            },
            model_for_collection,
        )

        collected = collect_samples_parallel(
            cfg=cfg,
            round_idx=round_idx,
            model_ckpt_path=model_for_collection,
        )

        if collected.train_features.shape[0] == 0:
            raise RuntimeError(f"round={round_idx}: no train samples collected")
        if collected.eval_features.shape[0] == 0:
            raise RuntimeError(f"round={round_idx}: no eval samples collected")

        train_result = train_value_model(
            cfg=cfg,
            model=model,
            train_x=collected.train_features,
            train_y=collected.train_targets,
            eval_x=collected.eval_features,
            eval_y=collected.eval_targets,
            device=device,
        )

        metrics_csv = rd / f"metrics_round_{round_idx:03d}.csv"
        _save_epoch_metrics(metrics_csv, train_result.epoch_rows)

        rollout_rows = run_greedy_rollouts(
            cfg=cfg,
            round_idx=round_idx,
            model=model,
        )
        rollout_summary_csv = rd / f"rollout_summary_round_{round_idx:03d}.csv"
        _save_rollout_summary(rollout_summary_csv, rollout_rows)

        model_ckpt = rd / f"model_round_{round_idx:03d}.pt"
        torch.save(
            {
                "num_features": 30,
                "model_state": model.state_dict(),
                "round": int(round_idx),
                "final_metrics": train_result.final_metrics,
            },
            model_ckpt,
        )

        final = train_result.final_metrics
        _append_summary(
            summary_csv,
            {
                "round": int(round_idx),
                "train_samples": int(collected.train_features.shape[0]),
                "eval_samples": int(collected.eval_features.shape[0]),
                "train_mse": float(final["train_mse"]),
                "train_mae": float(final["train_mae"]),
                "train_r2": float(final["train_r2"]),
                "eval_mse": float(final["eval_mse"]),
                "eval_mae": float(final["eval_mae"]),
                "eval_r2": float(final["eval_r2"]),
                "train_npz": str(collected.train_npz),
                "eval_npz": str(collected.eval_npz),
                "train_meta_csv": str(collected.train_meta_csv),
                "eval_meta_csv": str(collected.eval_meta_csv),
                "model_ckpt": str(model_ckpt),
                "rollout_summary_csv": str(rollout_summary_csv),
                "device": str(cfg.device),
            },
        )
