# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

from .config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig
from .multiProcessUtils import run_parallel_self_improvement


def main() -> None:
    cfg: MultipleProcessTrainingConfig = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
    cfg.validate()
    run_parallel_self_improvement(cfg)


if __name__ == "__main__":
    main()
