# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import os

from .config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig


def _set_main_process_thread_env(num_threads: int) -> None:
    n = str(max(1, int(num_threads)))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n


def main() -> None:
    cfg: MultipleProcessTrainingConfig = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
    cfg.validate()
    _set_main_process_thread_env(int(getattr(cfg, "train_num_threads", 1)))
    from .multiProcessUtils import run_parallel_self_improvement
    run_parallel_self_improvement(cfg)


if __name__ == "__main__":
    main()
