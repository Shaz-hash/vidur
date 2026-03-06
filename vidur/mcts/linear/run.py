# # (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import argparse
from typing import Sequence

import torch

from .config import (
    BellmanSettings,
    CollectionSettings,
    ConstraintSettings,
    LinearPipelineConfig,
    OutputSettings,
    RolloutSettings,
    SimulationSettings,
    TrainingSettings,
)
from .pipeline import run_self_improvement

"""
    RUN COMMAND :


    PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
    /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.linear.run \
    --rounds 100 \
    --workers 8 \
    --train-samples-per-worker 1000 \
    --eval-samples-per-worker 100 \
    --epochs 8 \
    --batch-size 32 \
    --discount-factor 0.98 \
    --history-hops 0,10,20,30,40,50,60,70 \
    --out-dir simulator_output/linear_value \
    --device cuda:0 \


"""


def _parse_hops(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    return tuple(vals) if vals else (0, 5, 10, 15, 20, 25, 30, 35)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Linear Bellman value-only self-improvement pipeline")

    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str, default=_default_device())

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-samples-per-worker", type=int, default=10000)
    ap.add_argument("--eval-samples-per-worker", type=int, default=1000)
    ap.add_argument("--history-hops", type=str, default="0,5,10,15,20,25,30,35")
    ap.add_argument("--history-hop-mode", type=str, default="fixed", choices=["fixed", "cyclic"], help="currently alias of fixed")
    ap.add_argument("--history-csv", type=str, default="")
    ap.add_argument("--root-player", type=str, default="adversary", choices=["controller", "adversary"])
    ap.add_argument("--max-branching", type=int, default=10)
    ap.add_argument("--enum-max-samples", type=int, default=10000)
    ap.add_argument("--max-forced-hops", type=int, default=20000)
    ap.add_argument("--max-total-steps-per-state", type=int, default=512)

    ap.add_argument("--discount-factor", type=float, default=0.98)
    ap.add_argument("--base-step-tokens", type=int, default=512)

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--num-traces", type=int, default=8)
    ap.add_argument("--max-rollout-steps", type=int, default=256)

    ap.add_argument("--out-dir", type=str, default="simulator_output/linear_value")
    ap.add_argument("--prefill-profile", type=str, default="simulator_output/prefill_profile.csv")

    ap.add_argument("--use-virtual-env", dest="use_virtual_env", action="store_true", default=True)
    ap.add_argument("--use-real-env", dest="use_virtual_env", action="store_false")
    ap.add_argument("--align-branching-roots", dest="align_branching_roots", action="store_true", default=True)
    ap.add_argument("--no-align-branching-roots", dest="align_branching_roots", action="store_false")

    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg = LinearPipelineConfig(
        sim=SimulationSettings(),
        constraints=ConstraintSettings(prefill_profile_path=str(args.prefill_profile)),
        collection=CollectionSettings(
            workers=int(args.workers),
            train_samples_per_worker=int(args.train_samples_per_worker),
            eval_samples_per_worker=int(args.eval_samples_per_worker),
            max_branching=int(args.max_branching),
            enum_max_samples=int(args.enum_max_samples),
            max_forced_hops=int(args.max_forced_hops),
            max_total_steps_per_state=int(args.max_total_steps_per_state),
            history_hops=_parse_hops(args.history_hops),
            history_csv=str(args.history_csv),
            root_player=str(args.root_player),
            align_branching_roots=bool(args.align_branching_roots),
            use_virtual_env=bool(args.use_virtual_env),
        ),
        bellman=BellmanSettings(
            discount_factor=float(args.discount_factor),
            base_step_tokens=int(args.base_step_tokens),
        ),
        training=TrainingSettings(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            optimizer=str(args.optimizer),
            weight_decay=float(args.weight_decay),
        ),
        rollout=RolloutSettings(
            num_traces=int(args.num_traces),
            max_steps=int(args.max_rollout_steps),
        ),
        output=OutputSettings(out_dir=str(args.out_dir)),
        rounds=int(args.rounds),
        seed=int(args.seed),
        device=str(args.device),
    )

    run_self_improvement(cfg)


if __name__ == "__main__":
    main()
