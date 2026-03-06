from __future__ import annotations

import numpy as np
import torch

from vidur.mcts.linear.config import LinearPipelineConfig
from vidur.mcts.linear.config import TrainingSettings
from vidur.mcts.linear.model import LinearValueModel
from vidur.mcts.linear.trainer import train_value_model


def test_trainer_reduces_loss_on_linear_data() -> None:
    rng = np.random.default_rng(7)
    n = 512
    d = 30
    x = rng.normal(size=(n, d)).astype(np.float32)

    w = np.zeros((d,), dtype=np.float32)
    w[0] = 2.0
    w[7] = -1.0
    w[15] = 0.5
    y = (x @ w + 0.1).astype(np.float32)

    split = 384
    train_x, train_y = x[:split], y[:split]
    eval_x, eval_y = x[split:], y[split:]

    cfg = LinearPipelineConfig(
        rounds=1,
        training=TrainingSettings(
            epochs=12,
            batch_size=32,
            learning_rate=5e-2,
            optimizer="adam",
            weight_decay=0.0,
        ),
    )

    model = LinearValueModel(num_features=d, zero_init=True)
    res = train_value_model(
        cfg=cfg,
        model=model,
        train_x=train_x,
        train_y=train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        device=torch.device("cpu"),
    )

    assert res.final_metrics["eval_mse"] < 0.2
