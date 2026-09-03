"""V15 wrapper class — provides arena-compatible interface around a raw HGB regressor.

The HGB was trained on (parent_features 498 dims) + (lookahead 10 dims).
Lookahead computed from forecast columns: realizability sigmoid + chunk-aware doomed count.

Loaded by the arena harness as a `classical_joblib` model.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np


# Column indices in the cliff feature names; resolved at construction from the names list
class V15Wrapper:
    """Wraps a sklearn HGB into an arena-compatible value model."""

    def __init__(self, hgb, feature_names: list[str], drop_grace: float = 1.0):
        self.hgb = hgb
        self.feature_names = list(feature_names)
        self.drop_grace = float(drop_grace)
        self.runtime_prediction_cache: dict = {}
        # Resolve indices once
        self._tick = self.feature_names.index("fc_tick_advance_est")
        self._doomed = self.feature_names.index("fc_doomed_floor_estimate")
        self._eta_max = self.feature_names.index("fc_prefill_eta_max")
        self._eta_total = self.feature_names.index("fc_prefill_eta_total")
        self._min_slack = self.feature_names.index("fc_prefill_min_slack_minus_eta")
        self._slot_present = [self.feature_names.index(f"fc_slot{k:02d}_present") for k in range(8)]
        self._slot_is_prefill = [self.feature_names.index(f"fc_slot{k:02d}_is_prefill") for k in range(8)]
        self._slot_time_to_target = [self.feature_names.index(f"fc_slot{k:02d}_time_to_target") for k in range(8)]
        self._slot_eta = [self.feature_names.index(f"fc_slot{k:02d}_eta") for k in range(8)]

    # ----- arena-compatible properties -----

    @property
    def model_name(self) -> str:
        return "v15_chunk_aware_supervised_v1"

    @property
    def feature_config(self) -> dict:
        return {"name": "v15_chunk_aware", "feature_names": self.feature_names}

    @property
    def trainable_params(self) -> int:
        return 0  # not exposed for HGB

    @property
    def uses_neural_network(self) -> bool:
        return False

    @property
    def uses_target_leakage(self) -> bool:
        return False

    @property
    def cached_predictions(self) -> dict:
        return self.runtime_prediction_cache

    # ----- prediction core -----

    def _la_for_block(self, blk_pf: np.ndarray) -> np.ndarray:
        out = np.zeros((blk_pf.shape[0], 10), dtype=np.float32)
        eta_tot = blk_pf[:, self._eta_total]
        tick = blk_pf[:, self._tick]
        doomed = blk_pf[:, self._doomed]
        eta_max = blk_pf[:, self._eta_max]
        min_slack = blk_pf[:, self._min_slack]
        ratio = eta_tot / np.maximum(tick, 0.01)
        real = 1.0 / (1.0 + np.exp(-(ratio - 1.0) * 4.0))
        out[:, 0] = ratio
        out[:, 1] = real
        out[:, 2] = doomed * real
        out[:, 3] = min_slack / np.maximum(tick, 0.01)
        out[:, 4] = eta_max / np.maximum(tick, 0.01)
        out[:, 5] = np.maximum(0.0, eta_max - tick)
        n = blk_pf.shape[0]
        cum_eta = np.zeros(n, dtype=np.float32)
        chunk_doomed = np.zeros(n, dtype=np.float32)
        chunk_savable = np.zeros(n, dtype=np.float32)
        chunk_double = np.zeros(n, dtype=np.float32)
        for k in range(8):
            present = blk_pf[:, self._slot_present[k]]
            is_pre = blk_pf[:, self._slot_is_prefill[k]]
            t_tt = blk_pf[:, self._slot_time_to_target[k]]
            eta = blk_pf[:, self._slot_eta[k]]
            mask = (present > 0.5) & (is_pre > 0.5)
            cum_eta = cum_eta + np.where(mask, eta, 0.0)
            chunk_doomed += (mask & (cum_eta > t_tt)).astype(np.float32)
            chunk_savable += (mask & (cum_eta <= t_tt)).astype(np.float32)
            chunk_double += (mask & (cum_eta > t_tt + self.drop_grace)).astype(np.float32)
        chunk_floor = -(chunk_doomed + chunk_double)
        out[:, 6] = chunk_doomed
        out[:, 7] = chunk_savable
        out[:, 8] = chunk_floor
        out[:, 9] = chunk_floor * real
        return out

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        x32 = x.astype(np.float32, copy=False)
        la = self._la_for_block(x32)
        full = np.concatenate([x32, la], axis=1)
        v = np.asarray(self.hgb.predict(full), dtype=np.float64)
        return np.minimum(v, 0.0)

    def infer_from_inputs(self, inputs: Any, player: str, *, device: Any | None = None) -> tuple[float, list[float]]:
        del player, device
        from .cliff_aware_value_model import extract_cliff_features_from_inputs
        row, _ = extract_cliff_features_from_inputs(inputs)
        target_len = len(self.feature_names)
        if len(row) != target_len:
            if len(row) < target_len:
                row = list(row) + [0.0] * (target_len - len(row))
            else:
                row = list(row)[:target_len]
        x = np.asarray([row], dtype=np.float32)
        key = x.tobytes()
        cached = self.runtime_prediction_cache.get(key)
        if cached is not None:
            return float(cached), []
        v = float(self.predict_matrix(x)[0])
        if len(self.runtime_prediction_cache) > 50_000:
            self.runtime_prediction_cache.clear()
        self.runtime_prediction_cache[key] = v
        return v, []
