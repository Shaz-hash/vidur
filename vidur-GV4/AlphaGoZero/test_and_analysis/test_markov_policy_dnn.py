"""Regression checks for structured Markov policy fitting and warm starts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from vidur.AlphaGoZero.dnn_models import (
    CONTROLLER_ACTION_DIM,
    MarkovPolicyRankDeepSet,
    fit_markov_policy_dnn,
    save_dnn_model,
)
from vidur.AlphaGoZero.markov_value_features import build_markov_value_features
from vidur.AlphaGoZero.test_and_analysis.test_markov_value_features import (
    representative_state,
    successor_states,
)


def main() -> None:
    decode_only, with_prefill = successor_states()
    states = [
        build_markov_value_features(payload)
        for payload in (representative_state(), decode_only, with_prefill)
    ]
    rng = np.random.default_rng(2127)
    actions = rng.normal(0.0, 0.25, size=(9, CONTROLLER_ACTION_DIM)).astype(np.float32)
    targets = np.asarray(
        [0.05, 0.15, 0.80, 0.10, 0.20, 0.70, 0.15, 0.25, 0.60],
        dtype=np.float32,
    )
    offsets = [(0, 3), (3, 6), (6, 9)]

    model, first = fit_markov_policy_dnn(
        states,
        actions,
        targets,
        offsets,
        role="controller",
        action_dim=CONTROLLER_ACTION_DIM,
        seed=2127,
        epochs=3,
        root_batch_size=2,
        torch_threads=1,
    )
    assert isinstance(model, MarkovPolicyRankDeepSet)
    assert int(first["warm_start"]) == 0
    assert model.training_metadata["state_encoded_once_per_root"] is True
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with tempfile.TemporaryDirectory(prefix="agz_markov_policy_fit_") as raw:
        parent = Path(raw) / "model.joblib"
        save_dnn_model(model, parent)
        updated, second = fit_markov_policy_dnn(
            states,
            actions,
            targets,
            offsets,
            role="controller",
            action_dim=CONTROLLER_ACTION_DIM,
            initial_model_path=parent,
            seed=2128,
            epochs=1,
            root_batch_size=3,
            torch_threads=1,
        )

    assert int(second["warm_start"]) == 1
    assert updated.optimizer_state is not None
    changed = any(
        not torch.equal(before[name], value.detach())
        for name, value in updated.state_dict().items()
    )
    assert changed
    logits = updated.predict_structured(states, actions, offsets)
    assert logits.shape == (9,)
    assert np.all(np.isfinite(logits))
    print(
        json.dumps(
            {
                "ok": True,
                "incremental_warm_start": True,
                "state_encoded_once_per_root": True,
                "rows": int(logits.size),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
