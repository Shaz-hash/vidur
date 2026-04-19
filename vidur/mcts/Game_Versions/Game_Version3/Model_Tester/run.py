from __future__ import annotations

from .config import DEFAULT_MODEL_TESTER_CONFIG
from .runner import run_model_vs_trivial_tester


def main() -> None:
    cfg = DEFAULT_MODEL_TESTER_CONFIG
    out_csv = run_model_vs_trivial_tester(cfg)
    print(f"[Model_Tester] Completed. Results: {out_csv}")


if __name__ == "__main__":
    main()

