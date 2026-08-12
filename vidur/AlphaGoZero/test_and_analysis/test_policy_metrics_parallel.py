"""Parity checks for multiprocess post-fit policy metrics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from vidur.AlphaGoZero.agz_train_eval_promote import (
    _policy_metrics,
    _policy_metrics_parallel_pair,
)
from vidur.AlphaGoZero.dnn_models import MarkovPolicyRankDeepSet
from vidur.AlphaGoZero.markov_value_features import build_markov_value_features
from vidur.AlphaGoZero.test_and_analysis.test_markov_value_features import (
    representative_state,
    successor_states,
)


def _policy_data(
    *,
    roots: int,
    actions_per_root: int,
    action_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    rows = roots * actions_per_root
    actions = rng.normal(0.0, 0.25, size=(rows, action_dim)).astype(np.float32)
    targets = rng.normal(0.0, 0.5, size=rows).astype(np.float32)
    probabilities = rng.uniform(0.01, 1.0, size=rows).astype(np.float32)
    offsets = [
        (root * actions_per_root, (root + 1) * actions_per_root)
        for root in range(roots)
    ]
    return actions, targets, probabilities, offsets


class PolicyMetricsParallelTest(unittest.TestCase):
    def test_markov_dnn_parallel_metrics_match_serial_metrics(self) -> None:
        base, decode_only, with_prefill = (
            representative_state(),
            *successor_states(),
        )
        feature_cycle = [
            build_markov_value_features(payload)
            for payload in (base, decode_only, with_prefill)
        ]
        controller_states = [feature_cycle[index % 3] for index in range(12)]
        adversary_states = [feature_cycle[index % 3] for index in range(8)]
        Xc, yc, pc, offc = _policy_data(
            roots=12,
            actions_per_root=4,
            action_dim=43,
            seed=41,
        )
        Xa, ya, pa, offa = _policy_data(
            roots=8,
            actions_per_root=3,
            action_dim=7,
            seed=43,
        )
        torch.manual_seed(47)
        controller_model = MarkovPolicyRankDeepSet(43, role="controller")
        adversary_model = MarkovPolicyRankDeepSet(7, role="adversary")

        parallel_controller, parallel_adversary, timings = (
            _policy_metrics_parallel_pair(
                controller_model,
                Xc,
                yc,
                pc,
                offc,
                adversary_model,
                Xa,
                ya,
                pa,
                offa,
                controller_states,
                adversary_states,
                max_workers=4,
            )
        )
        serial_controller = _policy_metrics(
            controller_model,
            Xc,
            yc,
            pc,
            offc,
            controller_states,
        )
        serial_adversary = _policy_metrics(
            adversary_model,
            Xa,
            ya,
            pa,
            offa,
            adversary_states,
        )

        for metric in ("mse", "cross_entropy", "top1", "top3"):
            self.assertAlmostEqual(
                parallel_controller[metric],
                serial_controller[metric],
                places=10,
            )
            self.assertAlmostEqual(
                parallel_adversary[metric],
                serial_adversary[metric],
                places=10,
            )
        self.assertEqual(timings["policy_metrics_mode"], "multiprocess_fork_shared")
        self.assertEqual(timings["policy_metrics_workers"], 4)


if __name__ == "__main__":
    unittest.main()
