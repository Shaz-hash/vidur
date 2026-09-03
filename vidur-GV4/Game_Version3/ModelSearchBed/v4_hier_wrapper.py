"""Arena wrapper for the hierarchical V1 (cls + reg) trained on v4 features.

Inference: y_hat = 0 if cls.predict_proba(x)[1] > tau else min(reg.predict(x), 0).
This matches `experiments/new_features_v1/train_v1_hierarchical_200k.py`.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class V4HierWrapper:
    """Arena adapter around an hierarchical (HGBClassifier + HGBRegressor) V1."""

    def __init__(self, cls: Any, reg: Any, tau: float, thr: float = 0.05,
                 feature_dim: int = 224, model_tag: str = "v4_hier"):
        self.cls = cls
        self.reg = reg
        self.tau = float(tau)
        self.thr = float(thr)
        self.feature_dim = int(feature_dim)
        self.model_tag = str(model_tag)
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        self.runtime_prediction_cache: dict = {}

    def __getstate__(self) -> dict:
        return {
            "cls": self.cls, "reg": self.reg, "tau": self.tau, "thr": self.thr,
            "feature_dim": self.feature_dim, "model_tag": self.model_tag,
        }

    def __setstate__(self, state: dict) -> None:
        self.cls = state["cls"]
        self.reg = state["reg"]
        self.tau = float(state["tau"])
        self.thr = float(state.get("thr", 0.05))
        self.feature_dim = int(state.get("feature_dim", 224))
        self.model_tag = str(state.get("model_tag", "v4_hier"))
        self._init_runtime_state()
        try:
            from ..DNN.infer import enable_inputs_extras
            enable_inputs_extras()
        except Exception:
            pass

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

    def _features_from_extras(self, extras: dict) -> np.ndarray:
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
                f"V4HierWrapper: feature dim {feat.size} != {self.feature_dim}"
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
                "V4HierWrapper.infer_from_inputs called without inputs.extras"
            )
        feat = self._features_from_extras(extras)
        x = feat.reshape(1, -1)
        key = x.tobytes()
        cached = self.runtime_prediction_cache.get(key)
        if cached is not None:
            return float(cached), []
        try:
            self.cls.n_jobs = 1
            self.reg.n_jobs = 1
        except Exception:
            pass
        p_pos = float(self.cls.predict_proba(x)[0, 1])
        if p_pos > self.tau:
            value = 0.0
        else:
            r = float(np.asarray(self.reg.predict(x), dtype=np.float64)[0])
            value = float(min(r, 0.0))
        if len(self.runtime_prediction_cache) > 50_000:
            self.runtime_prediction_cache.clear()
        self.runtime_prediction_cache[key] = value
        return value, []


def pack_v4_hier(in_joblib_path: str, out_joblib_path: str, *,
                 model_tag: str = "v4_hier") -> None:
    """Load a {cls, reg, tau, thr} dict joblib and rewrap as V4HierWrapper."""
    import joblib
    pack = joblib.load(in_joblib_path)
    wrapped = V4HierWrapper(
        cls=pack["cls"], reg=pack["reg"],
        tau=float(pack.get("tau", 0.5)),
        thr=float(pack.get("thr", 0.05)),
        feature_dim=224, model_tag=model_tag,
    )
    joblib.dump(wrapped, out_joblib_path, compress=3)
