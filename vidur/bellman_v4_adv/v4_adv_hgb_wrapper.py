"""Arena wrapper for v4-Adv HGB value models.

The v4-Adv models are trained on the 226-dimensional feature schema produced by
``vidur.bellman_v4_adv.build_state_local_features_adv``. The arena harness only
passes ``ModelInputs`` to the model, so this wrapper uses the ``inputs.extras``
side channel populated by ``DNN.infer.enable_inputs_extras``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class V4AdvHGBWrapper:
    """Arena-compatible adapter around a bare HGB regressor trained on v4-Adv features."""

    def __init__(self, hgb: Any, feature_dim: int = 226, model_tag: str = "v4_adv_hgb"):
        self.hgb = hgb
        self.feature_dim = int(feature_dim)
        self.model_tag = str(model_tag)
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        self.runtime_prediction_cache: dict[bytes, float] = {}

    def __getstate__(self) -> dict[str, Any]:
        return {
            "hgb": self.hgb,
            "feature_dim": self.feature_dim,
            "model_tag": self.model_tag,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.hgb = state["hgb"]
        self.feature_dim = int(state.get("feature_dim", 226))
        self.model_tag = str(state.get("model_tag", "v4_adv_hgb"))
        self._init_runtime_state()
        try:
            from vidur.mcts.Game_Versions.Game_Version3.DNN.infer import enable_inputs_extras

            enable_inputs_extras()
        except Exception:
            pass

    @property
    def model_name(self) -> str:
        return self.model_tag

    @property
    def feature_config(self) -> dict[str, Any]:
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
    def cached_predictions(self) -> dict[bytes, float]:
        return self.runtime_prediction_cache

    def _features_from_extras(self, extras: dict[str, Any]) -> np.ndarray:
        from vidur.bellman_v4_adv.build_state_local_features_adv import (
            extract_features_one_record,
        )

        record = {
            "simulator_snapshot": extras.get("simulator_snapshot") or {},
            "stats": extras.get("stats"),
            "root_id": -1,
        }
        feat = np.asarray(extract_features_one_record(record), dtype=np.float32).reshape(-1)
        if feat.size != self.feature_dim:
            raise RuntimeError(
                f"V4AdvHGBWrapper: extracted feature dim {feat.size} != expected {self.feature_dim}"
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
                "V4AdvHGBWrapper.infer_from_inputs called without inputs.extras; "
                "the arena must enable DNN.infer inputs extras before feature extraction."
            )
        x = self._features_from_extras(extras).reshape(1, -1)
        key = x.tobytes()
        cached = self.runtime_prediction_cache.get(key)
        if cached is not None:
            return float(cached), []
        try:
            self.hgb.n_jobs = 1
        except Exception:
            pass
        v_raw = float(np.asarray(self.hgb.predict(x), dtype=np.float64)[0])
        value = float(min(v_raw, 0.0))
        if len(self.runtime_prediction_cache) > 50_000:
            self.runtime_prediction_cache.clear()
        self.runtime_prediction_cache[key] = value
        return value, []


def pack_v4_adv_hgb(
    in_joblib_path: str,
    out_joblib_path: str,
    *,
    model_tag: str = "v4_adv_hgb",
) -> None:
    """Load a bare HGB joblib and rewrite it as an arena-compatible wrapper."""
    import joblib

    hgb = joblib.load(in_joblib_path)
    wrapped = V4AdvHGBWrapper(hgb, feature_dim=226, model_tag=model_tag)
    joblib.dump(wrapped, out_joblib_path, compress=3)

def main() -> None:
    import argparse
    from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import (
        pack_v4_adv_hgb as canonical_pack_v4_adv_hgb,
    )

    parser = argparse.ArgumentParser(description="Wrap or check a v4-Adv HGB joblib for arena testing.")
    parser.add_argument("--input", help="Input bare HGB model.joblib path.")
    parser.add_argument("--output", help="Output wrapped joblib path.")
    parser.add_argument("--model-tag", default="v4_adv_hgb")
    parser.add_argument("--check", help="Wrapped joblib path to sanity-load and report.")
    args = parser.parse_args()
    if args.check:
        import joblib

        model = joblib.load(args.check)
        print(type(model).__module__, type(model).__name__, model.feature_config)
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --check is used")
    canonical_pack_v4_adv_hgb(args.input, args.output, model_tag=args.model_tag)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
