#!/usr/bin/env python3
"""Remote status for adversary MCTS target generation on workers 5-8.

This is a thin wrapper around ``count_mcts_targets_remote.py`` with the defaults
set to the current adversary target run:

- experiment: bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon
- workers: bellman-classical-worker-5 through bellman-classical-worker-8
- target suffix: adversary_effectcanon_full
"""

from __future__ import annotations

import sys

from count_mcts_targets_remote import main as count_main


DEFAULT_ARGS = [
    "--experiment-name",
    "bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon",
    "--workers",
    ",".join(
        [
            "worker5=bellman-classical-worker-5",
            "worker6=bellman-classical-worker-6",
            "worker7=bellman-classical-worker-7",
            "worker8=bellman-classical-worker-8",
        ]
    ),
    "--model-label",
    "worker1_200k_alpha2__hgb_sq_63leaf_1050iter_a2__v47__",
    "--mcts-iterations",
    "10000",
    "--target-dir-suffix",
    "adversary_effectcanon_full",
    "--max-parallel",
    "4",
]


if __name__ == "__main__":
    # User-provided duplicate flags intentionally win because argparse keeps the
    # last value for repeated scalar options.
    raise SystemExit(count_main(DEFAULT_ARGS + sys.argv[1:]))
