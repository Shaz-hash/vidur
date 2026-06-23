"""Arena-side wrapper for the v4 state-local HGB controller value model.

The HGB was trained on the 224-dim v4 schema produced by
`experiments.new_features_v1.build_state_local_features.extract_features_one_record`,
which consumes a `record` dict with simulator_snapshot + stats. The arena
harness only hands the model `ModelInputs`, so this wrapper relies on the
`ModelInputs.extras` side channel populated by `DNN/infer.build_model_inputs`
when `enable_inputs_extras()` has been called for the current process.

The wrapper deliberately mirrors the cliff-aware/V15 wrappers in this folder:
- It is loaded by the arena harness via `joblib.load(...)` and must therefore
  be picklable on its own (the bare HGB is the only sklearn object inside).
- It implements the same arena-facing surface
  (`infer_from_inputs`, `model_name`, `feature_config`, `trainable_params`,
  `uses_neural_network`, `uses_target_leakage`, `cached_predictions`).

Loading semantics: `__setstate__` flips on the inputs-extras attachment in
`DNN.infer` so that as soon as a worker `joblib.load`s a wrapped model, the
next `build_model_inputs` call from MCTS bootstrap will populate
`inputs.extras`. The torch DNN ignores the field, so this is safe even if a
mixed-model run is happening (which the harness does not currently do).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class V4HGBWrapper:
    """Arena-compatible adapter around a bare HGB regressor trained on v4 features."""

    def __init__(self, hgb: Any, feature_dim: int = 224, model_tag: str = "v4_hgb"):
        self.hgb = hgb
        self.feature_dim = int(feature_dim)
        self.model_tag = str(model_tag)
        self._init_runtime_state()

    # ----- pickle support -----
    def _init_runtime_state(self) -> None:
        self.runtime_prediction_cache: dict = {}

    def __getstate__(self) -> dict:
        return {
            "hgb": self.hgb,
            "feature_dim": self.feature_dim,
            "model_tag": self.model_tag,
        }

    def __setstate__(self, state: dict) -> None:
        self.hgb = state["hgb"]
        self.feature_dim = int(state.get("feature_dim", 224))
        self.model_tag = str(state.get("model_tag", "v4_hgb"))
        self._init_runtime_state()
        # As soon as this object is unpickled (e.g. inside an arena worker),
        # ensure the inputs-extras side channel is on for this process.
        try:
            from ..DNN.infer import enable_inputs_extras

            enable_inputs_extras()
        except Exception:
            # Best effort; an explicit caller should still toggle it on the
            # main process before the workers spawn.
            pass

    # ----- arena-required surface -----
    @property
    def model_name(self) -> str:
        return self.model_tag

    @property
    def feature_config(self) -> dict:
        return {"name": "v4_state_local", "feature_dim": self.feature_dim}

    @property
    def trainable_params(self) -> int:
        return 0

    @property
    def uses_neural_network(self) -> bool:
        return False

    @property
    def uses_target_leakage(self) -> bool:
        return False

    @property
    def cached_predictions(self) -> dict:
        return self.runtime_prediction_cache

    # ----- prediction -----
    def _features_from_extras(self, extras: dict) -> np.ndarray:
        # Late import keeps `experiments/` off the import path of generic
        # GV3 consumers; only this wrapper depends on it.
        from experiments.new_features_v1.build_state_local_features import (
            extract_features_one_record,
        )

        record = {
            "simulator_snapshot": extras.get("simulator_snapshot") or {},
            "stats": extras.get("stats"),
            "root_id": -1,
        }
        feat = extract_features_one_record(record)
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        if feat.size != self.feature_dim:
            raise RuntimeError(
                f"V4HGBWrapper: extracted feature dim {feat.size} != expected {self.feature_dim}"
            )
        return feat

    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: Optional[Any] = None,
    ) -> tuple[float, list[float]]:
        del player, device
        extras = getattr(inputs, "extras", None)
        if not extras:
            raise RuntimeError(
                "V4HGBWrapper.infer_from_inputs called without inputs.extras; "
                "did you forget to call DNN.infer.enable_inputs_extras() before "
                "building inputs? (this normally happens automatically at unpickle "
                "time)"
            )
        feat = self._features_from_extras(extras)
        x = feat.reshape(1, -1)
        key = x.tobytes()
        cached = self.runtime_prediction_cache.get(key)
        if cached is not None:
            return float(cached), []
        # Bootstrap is a single-row call inside MCTS; force serial predict.
        try:
            self.hgb.n_jobs = 1
        except Exception:
            pass
        v_raw = float(np.asarray(self.hgb.predict(x), dtype=np.float64)[0])
        # Controller value is a non-positive penalty in this codebase; clamp to
        # match the training-time post-process (np.minimum(yp, 0.0)).
        value = float(min(v_raw, 0.0))
        if len(self.runtime_prediction_cache) > 50_000:
            self.runtime_prediction_cache.clear()
        self.runtime_prediction_cache[key] = value
        return value, []


def pack_v4_hgb(in_joblib_path: str, out_joblib_path: str, *, model_tag: str = "v4_hgb") -> None:
    """Load a bare HGB from `in_joblib_path` and rewrite as a V4HGBWrapper joblib."""
    import joblib  # local import avoids hard dep at module import time.

    hgb = joblib.load(in_joblib_path)
    wrapped = V4HGBWrapper(hgb, feature_dim=224, model_tag=model_tag)
    joblib.dump(wrapped, out_joblib_path, compress=3)
