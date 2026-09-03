"""Thin, batched inference facade for future GV4 AlphaGoZero models.

The facade owns feature construction and model-contract validation. Models own
their neural architecture, tensor/device handling, normalization of value
targets, and row padding/masking. Policy methods return raw logits; MCTS remains
responsible for softmax temperature, legal masks, and root Dirichlet noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..action_resolver import CanonicalAdversaryAction, CanonicalControllerAction
from ..config import GV4EngineConfig
from ..state import GV4State, Player
from .dnn_features import (
    GV4AdversaryActionFeatures,
    GV4ControllerActionFeatures,
    GV4FeatureBuilder,
    GV4StateFeatures,
)


class GV4InferenceError(RuntimeError):
    """Raised when a model or prediction violates the GV4 inference contract."""


class GV4ValueModel(Protocol):
    """Method a GV4 value model in ``dnn_models.py`` must expose."""

    role: str
    feature_schema_version: str
    config_manifest_sha256: str

    def predict_structured(self, features: Sequence[GV4StateFeatures]) -> object:
        pass


class GV4PolicyModel(Protocol):
    """Method a GV4 policy ranker in ``dnn_models.py`` must expose."""

    role: str
    feature_schema_version: str
    config_manifest_sha256: str

    def predict_root_structured(
        self,
        state_features: GV4StateFeatures,
        action_features: Sequence[
            GV4ControllerActionFeatures | GV4AdversaryActionFeatures
        ],
    ) -> object:
        pass


def _player_name(player: Player | str) -> str:
    if player == Player.CONTROLLER or player == "controller":
        return "controller"
    if player == Player.ADVERSARY or player == "adversary":
        return "adversary"
    raise GV4InferenceError("GV4 DNN inference supports controller/adversary only")


def _numpy_vector(result: object, expected: int, label: str) -> NDArray[np.float32]:
    """Normalize NumPy, Torch, and list outputs without importing Torch."""

    value = result
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != expected:
        raise GV4InferenceError(
            f"{label} returned {array.size} values; expected {expected}"
        )
    if not np.isfinite(array).all():
        raise GV4InferenceError(f"{label} returned a non-finite value")
    return array


class GV4DNNInference:
    """Prepare GV4 features once and invoke role-specific models in batches."""

    __slots__ = (
        "builder",
        "_controller_value_model",
        "_adversary_value_model",
        "_controller_policy_model",
        "_adversary_policy_model",
    )

    def __init__(
        self,
        config: GV4EngineConfig,
        *,
        controller_value_model: GV4ValueModel | None = None,
        adversary_value_model: GV4ValueModel | None = None,
        controller_policy_model: GV4PolicyModel | None = None,
        adversary_policy_model: GV4PolicyModel | None = None,
    ) -> None:
        self.builder = GV4FeatureBuilder(config)
        self._controller_value_model = controller_value_model
        self._adversary_value_model = adversary_value_model
        self._controller_policy_model = controller_policy_model
        self._adversary_policy_model = adversary_policy_model

    @property
    def config(self) -> GV4EngineConfig:
        return self.builder.config

    def build_state_features(self, state: GV4State) -> GV4StateFeatures:
        """Public seam for replay generation and state reuse within one MCTS node."""

        return self.builder.build_state(state)

    def _check_model(self, model: Any, role: str, label: str) -> None:
        if model is None:
            raise GV4InferenceError(f"no {role} {label} model is configured")
        model_role = getattr(model, "role", role)
        if str(model_role) != role:
            raise GV4InferenceError(
                f"{label} model role is {model_role!r}, expected {role!r}"
            )
        model_schema = getattr(model, "feature_schema_version", None)
        expected_schema = self.builder.layout.schema_version
        if model_schema is None:
            raise GV4InferenceError(f"{label} model has no feature schema metadata")
        if str(model_schema) != expected_schema:
            raise GV4InferenceError(
                f"{label} model schema is {model_schema!r}, "
                f"expected {expected_schema!r}"
            )
        model_manifest = getattr(model, "config_manifest_sha256", None)
        if model_manifest is None:
            raise GV4InferenceError(f"{label} model has no config manifest metadata")
        if str(model_manifest) != self.builder.manifest_sha256:
            raise GV4InferenceError(f"{label} model uses a different GV4 manifest")

    def _check_state_features(self, features: GV4StateFeatures) -> None:
        if features.schema_version != self.builder.layout.schema_version:
            raise GV4InferenceError("state features use a different schema")
        if features.config_manifest_sha256 != self.builder.manifest_sha256:
            raise GV4InferenceError("state features use a different GV4 manifest")

    def predict_values_from_features(
        self,
        features: Sequence[GV4StateFeatures],
        *,
        player: Player | str,
    ) -> NDArray[np.float32]:
        """Evaluate several states in one model call."""

        samples = tuple(features)
        if not samples:
            return np.empty(0, dtype=np.float32)
        for sample in samples:
            self._check_state_features(sample)

        role = _player_name(player)
        model = (
            self._controller_value_model
            if role == "controller"
            else self._adversary_value_model
        )
        self._check_model(model, role, "value")
        predictor = getattr(model, "predict_structured", None)
        if not callable(predictor):
            raise GV4InferenceError(
                "GV4 value model must implement predict_structured(states)"
            )
        return _numpy_vector(predictor(samples), len(samples), f"{role} value model")

    def predict_values(
        self,
        states: Sequence[GV4State],
        *,
        player: Player | str,
    ) -> NDArray[np.float32]:
        return self.predict_values_from_features(
            tuple(self.builder.build_state(state) for state in states),
            player=player,
        )

    def predict_value(
        self,
        state: GV4State,
        *,
        player: Player | str | None = None,
        state_features: GV4StateFeatures | None = None,
    ) -> float:
        """Evaluate one state, optionally reusing features built for policy."""

        role = state.next_player if player is None else player
        features = state_features or self.builder.build_state(state)
        return float(self.predict_values_from_features((features,), player=role)[0])

    def _policy_logits(
        self,
        state_features: GV4StateFeatures,
        action_features: Sequence[
            GV4ControllerActionFeatures | GV4AdversaryActionFeatures
        ],
        *,
        role: str,
    ) -> NDArray[np.float32]:
        self._check_state_features(state_features)
        features = tuple(action_features)
        if not features:
            return np.empty(0, dtype=np.float32)
        model = (
            self._controller_policy_model
            if role == "controller"
            else self._adversary_policy_model
        )
        self._check_model(model, role, "policy")
        predictor = getattr(model, "predict_root_structured", None)
        if not callable(predictor):
            raise GV4InferenceError(
                "GV4 policy model must implement "
                "predict_root_structured(state, actions)"
            )
        return _numpy_vector(
            predictor(state_features, features),
            len(features),
            f"{role} policy model",
        )

    def predict_controller_logits(
        self,
        state: GV4State,
        actions: Sequence[CanonicalControllerAction],
        *,
        state_features: GV4StateFeatures | None = None,
    ) -> NDArray[np.float32]:
        """Score all canonical controller edges in one model call."""

        if state.next_player != Player.CONTROLLER:
            raise GV4InferenceError(
                "controller policy called on a non-controller state"
            )
        edges = tuple(actions)
        features = state_features or self.builder.build_state(state)
        action_features = tuple(
            self.builder.build_controller_action(state, edge) for edge in edges
        )
        return self._policy_logits(features, action_features, role="controller")

    def predict_adversary_logits(
        self,
        state: GV4State,
        actions: Sequence[CanonicalAdversaryAction],
        *,
        state_features: GV4StateFeatures | None = None,
    ) -> NDArray[np.float32]:
        """Score all canonical adversary edges in one model call."""

        if state.next_player != Player.ADVERSARY:
            raise GV4InferenceError("adversary policy called on a non-adversary state")
        edges = tuple(actions)
        features = state_features or self.builder.build_state(state)
        action_features = tuple(
            self.builder.build_adversary_action(state, edge) for edge in edges
        )
        return self._policy_logits(features, action_features, role="adversary")


__all__ = [
    "GV4DNNInference",
    "GV4InferenceError",
    "GV4PolicyModel",
    "GV4ValueModel",
]
