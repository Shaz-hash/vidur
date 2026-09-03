"""GV4 DeepSets value and policy models.

The engine owns feature meaning. This module only validates that contract,
pads variable row sets for a minibatch, and learns from replay targets. No
request or action is truncated, and no simulator state is interpreted here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Literal, Mapping, Sequence, TypeVar

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from GV4_Engine.config import GV4EngineConfig
from GV4_Engine.dnn_inference.dnn_features import GV4FeatureLayout


Role = Literal["controller", "adversary"]
ARCHITECTURE_VERSION = "gv4_deepset_v1"
CHECKPOINT_VERSION = "gv4_dnn_checkpoint_v1"
NATIVE_EXPORT_VERSION = "gv4_dnn_text_v1"
DEFAULT_VALUE_SCALE = 25.0

__all__ = [
    "ARCHITECTURE_VERSION",
    "GV4ModelError",
    "GV4ModelSpec",
    "GV4PolicyDeepSet",
    "GV4ValueDeepSet",
    "artifact_metadata",
    "export_dnn_to_native",
    "fit_markov_policy_dnn",
    "fit_markov_value_dnn",
    "fit_policy_dnn",
    "fit_value_dnn",
    "is_dnn_model",
    "load_dnn_model",
    "model_parameter_count",
    "save_dnn_model",
    "write_artifact_metadata",
]


class GV4ModelError(ValueError):
    """Raised when model inputs do not match their immutable GV4 contract."""


def _role(value: str) -> Role:
    if value not in {"controller", "adversary"}:
        raise GV4ModelError("role must be 'controller' or 'adversary'")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GV4ModelSpec:
    """Dimensions and identities that make one model artifact safe to load."""

    feature_schema_version: str
    config_manifest_sha256: str
    pipeline_stage_count: int
    num_replicas: int
    global_dim: int
    request_dim: int
    launch_dim: int
    replica_dim: int
    microbatch_dim: int
    controller_header_dim: int
    controller_request_dim: int
    adversary_header_dim: int
    adversary_request_dim: int

    @classmethod
    def from_config(cls, config: GV4EngineConfig) -> "GV4ModelSpec":
        """Bind a model to the exact engine manifest and feature layout."""

        if config.topology.num_replicas != 1:
            raise GV4ModelError(
                "GV4 AlphaGoZero v1 supports exactly one replica; "
                "multi-replica pooling needs an explicit ownership encoder"
            )
        layout = GV4FeatureLayout.from_config(config)
        return cls(
            feature_schema_version=layout.schema_version,
            config_manifest_sha256=config.manifest_sha256(),
            pipeline_stage_count=layout.pipeline_stage_count,
            num_replicas=config.topology.num_replicas,
            global_dim=len(layout.global_names),
            request_dim=len(layout.request_names),
            launch_dim=len(layout.launch_names),
            replica_dim=len(layout.replica_names),
            microbatch_dim=len(layout.microbatch_names),
            controller_header_dim=len(layout.controller_header_names),
            controller_request_dim=len(layout.controller_request_names),
            adversary_header_dim=len(layout.adversary_header_names),
            adversary_request_dim=len(layout.adversary_request_names),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GV4ModelSpec":
        names = tuple(cls.__dataclass_fields__)
        missing = [name for name in names if name not in values]
        if missing:
            raise GV4ModelError(f"checkpoint model spec is missing {missing}")
        return cls(**{name: values[name] for name in names})

    def action_dimensions(self, role: str) -> tuple[int, int]:
        if _role(role) == "controller":
            return self.controller_header_dim, self.controller_request_dim
        return self.adversary_header_dim, self.adversary_request_dim


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise GV4ModelError(f"feature payload has no {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise GV4ModelError(f"feature object has no {name!r}")
    return getattr(value, name)


def _vector(value: object, width: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (width,):
        raise GV4ModelError(f"{label} has shape {result.shape}, expected ({width},)")
    if not np.isfinite(result).all():
        raise GV4ModelError(f"{label} contains a non-finite value")
    return np.ascontiguousarray(result)


def _matrix(value: object, width: int, label: str) -> np.ndarray:
    """Read NumPy/replay matrices and the native pybind FeatureMatrix."""

    if all(hasattr(value, name) for name in ("values", "rows", "columns")):
        rows = int(getattr(value, "rows"))
        columns = int(getattr(value, "columns"))
        raw = getattr(value, "values")
        if columns != width:
            raise GV4ModelError(f"{label} has {columns} columns, expected {width}")
        result = np.asarray(raw, dtype=np.float32).reshape(rows, columns)
    else:
        result = np.asarray(value, dtype=np.float32)
        result = (
            np.empty((0, width), dtype=np.float32)
            if result.size == 0
            else result.reshape(-1, width)
        )
    if not np.isfinite(result).all():
        raise GV4ModelError(f"{label} contains a non-finite value")
    return np.ascontiguousarray(result)


def _offsets(value: object, row_count: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if (
        result.size == 0
        or result[0] != 0
        or result[-1] != row_count
        or np.any(result[1:] < result[:-1])
    ):
        raise GV4ModelError(f"{label} does not form valid row boundaries")
    return result


StateArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
ActionArrays = tuple[np.ndarray, np.ndarray]


def _state_arrays(feature: object, spec: GV4ModelSpec) -> StateArrays:
    if str(_field(feature, "schema_version")) != spec.feature_schema_version:
        raise GV4ModelError("state feature schema differs from the model")
    if str(_field(feature, "config_manifest_sha256")) != spec.config_manifest_sha256:
        raise GV4ModelError("state feature manifest differs from the model")

    global_values = _vector(
        _field(feature, "global_features"), spec.global_dim, "global features"
    )
    requests = _matrix(
        _field(feature, "request_rows"), spec.request_dim, "request rows"
    )
    launches = _matrix(_field(feature, "launch_rows"), spec.launch_dim, "launch rows")
    replicas = _matrix(
        _field(feature, "replica_rows"), spec.replica_dim, "replica rows"
    )
    microbatches = _matrix(
        _field(feature, "microbatch_rows"), spec.microbatch_dim, "microbatch rows"
    )
    request_offsets = _offsets(
        _field(feature, "request_replica_offsets"), len(requests), "request offsets"
    )
    microbatch_offsets = _offsets(
        _field(feature, "microbatch_replica_offsets"),
        len(microbatches),
        "microbatch offsets",
    )
    if len(replicas) != spec.num_replicas:
        raise GV4ModelError("replica row count differs from the model")
    if len(request_offsets) != spec.num_replicas + 1:
        raise GV4ModelError("request offsets differ from the replica count")
    if len(microbatch_offsets) != spec.num_replicas + 1:
        raise GV4ModelError("microbatch offsets differ from the replica count")
    return global_values, requests, launches, replicas, microbatches


def _action_arrays(feature: object, spec: GV4ModelSpec, role: str) -> ActionArrays:
    header_dim, request_dim = spec.action_dimensions(role)
    return (
        _vector(_field(feature, "header"), header_dim, f"{role} action header"),
        _matrix(
            _field(feature, "affected_request_rows"),
            request_dim,
            f"{role} affected-request rows",
        ),
    )


def _pad_rows(
    rows: Sequence[np.ndarray], width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_rows = max(1, *(len(value) for value in rows))
    values = np.zeros((len(rows), max_rows, width), dtype=np.float32)
    mask = np.zeros((len(rows), max_rows), dtype=np.bool_)
    for index, value in enumerate(rows):
        count = len(value)
        if count:
            values[index, :count] = value
            mask[index, :count] = True
    return torch.from_numpy(values), torch.from_numpy(mask)


def _collate_states(
    features: Sequence[object], spec: GV4ModelSpec
) -> tuple[torch.Tensor, ...]:
    samples = [_state_arrays(feature, spec) for feature in features]
    if not samples:
        raise GV4ModelError("at least one state feature is required")
    global_values = torch.from_numpy(np.stack([item[0] for item in samples]))
    result: list[torch.Tensor] = [global_values]
    for position, width in (
        (1, spec.request_dim),
        (2, spec.launch_dim),
        (3, spec.replica_dim),
        (4, spec.microbatch_dim),
    ):
        values, mask = _pad_rows([item[position] for item in samples], width)
        result.extend((values, mask))
    return tuple(result)


def _collate_actions(
    features: Sequence[object], spec: GV4ModelSpec, role: str
) -> tuple[torch.Tensor, ...]:
    samples = [_action_arrays(feature, spec, role) for feature in features]
    if not samples:
        raise GV4ModelError("at least one action feature is required")
    header_dim, request_dim = spec.action_dimensions(role)
    headers = torch.from_numpy(
        np.stack([_vector(item[0], header_dim, "action header") for item in samples])
    )
    rows, mask = _pad_rows([item[1] for item in samples], request_dim)
    return headers, rows, mask


def _to_device(
    tensors: Sequence[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in tensors)


class BottleneckResidual(nn.Module):
    def __init__(self, width: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.silu(self.down(self.norm(values))))
        return values + residual


def _masked_sum_max(
    encoded: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask3 = mask.to(torch.bool).unsqueeze(-1)
    summed = (encoded * mask3.to(encoded.dtype)).sum(dim=1)
    lowest = torch.finfo(encoded.dtype).min
    maximum = encoded.masked_fill(~mask3, lowest).max(dim=1).values
    maximum = torch.where(
        mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
    )
    return summed, maximum


class _RowSetEncoder(nn.Module):
    """Encode every row independently, then pool without depending on order."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.first = nn.Linear(input_dim, output_dim)
        self.second = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, rows: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = F.silu(self.first(rows))
        encoded = F.silu(self.norm(self.second(encoded)))
        return _masked_sum_max(encoded, mask)


class GV4StateEncoder(nn.Module):
    """Shared, permutation-invariant encoder for the complete GV4 state."""

    output_dim = 192

    def __init__(self, spec: GV4ModelSpec) -> None:
        super().__init__()
        self.global_layer = nn.Linear(spec.global_dim, 64)
        self.global_norm = nn.LayerNorm(64)
        self.requests = _RowSetEncoder(spec.request_dim, 64)
        self.launches = _RowSetEncoder(spec.launch_dim, 32)
        self.replicas = _RowSetEncoder(spec.replica_dim, 32)
        self.microbatches = _RowSetEncoder(spec.microbatch_dim, 32)
        self.fusion = nn.Linear(384, self.output_dim)
        self.block1 = BottleneckResidual(self.output_dim, 64)
        self.block2 = BottleneckResidual(self.output_dim, 64)

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        if len(values) != 9:
            raise GV4ModelError("state encoder expects global plus four row-set pairs")
        global_embedding = F.silu(self.global_norm(self.global_layer(values[0])))
        pooled = [global_embedding]
        encoders = (self.requests, self.launches, self.replicas, self.microbatches)
        for encoder, index in zip(encoders, range(1, 9, 2)):
            pooled.extend(encoder(values[index], values[index + 1]))
        result = F.silu(self.fusion(torch.cat(pooled, dim=-1)))
        return self.block2(self.block1(result))


class _ActionEncoder(nn.Module):
    output_dim = 64

    def __init__(self, header_dim: int, request_dim: int) -> None:
        super().__init__()
        self.header = nn.Linear(header_dim, 48)
        self.header_norm = nn.LayerNorm(48)
        self.requests = _RowSetEncoder(request_dim, 32)
        self.fusion = nn.Linear(112, self.output_dim)
        self.fusion_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self, header: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        header_embedding = F.silu(self.header_norm(self.header(header)))
        request_sum, request_max = self.requests(rows, mask)
        values = torch.cat((header_embedding, request_sum, request_max), dim=-1)
        return F.silu(self.fusion_norm(self.fusion(values)))


class _GV4ModelBase(nn.Module):
    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, spec: GV4ModelSpec, *, role: str) -> None:
        super().__init__()
        self.spec = spec
        self.role: Role = _role(role)
        self.feature_schema_version = spec.feature_schema_version
        self.config_manifest_sha256 = spec.config_manifest_sha256
        self.optimizer_state: dict[str, Any] | None = None
        self.training_metadata: dict[str, Any] = {}

    @property
    def trainable_params(self) -> int:
        return model_parameter_count(self)

    @property
    def uses_neural_network(self) -> bool:
        return True

    @property
    def uses_target_leakage(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class GV4ValueDeepSet(_GV4ModelBase):
    """Estimate controller-valued future return from a complete GV4 state."""

    model_kind = "value_dnn"

    def __init__(
        self,
        spec: GV4ModelSpec,
        *,
        role: str,
        value_scale: float = DEFAULT_VALUE_SCALE,
    ) -> None:
        super().__init__(spec, role=role)
        if not math.isfinite(value_scale) or value_scale <= 0.0:
            raise GV4ModelError("value_scale must be positive and finite")
        self.value_scale = float(value_scale)
        self.state_encoder = GV4StateEncoder(spec)
        self.head_norm = nn.LayerNorm(GV4StateEncoder.output_dim)
        self.head_hidden = nn.Linear(GV4StateEncoder.output_dim, 32)
        self.head_output = nn.Linear(32, 1)

        # Initial predictions stay close to the neutral zero bootstrap.
        nn.init.normal_(self.head_output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head_output.bias, -6.0)

    @property
    def model_name(self) -> str:
        return f"agz_gv4_value_deepset_{self.role}"

    def forward_normalized(self, *state_tensors: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(*state_tensors)
        hidden = F.silu(self.head_hidden(self.head_norm(state)))
        # Reward is objective_before - objective_after, so valid returns are <= 0.
        return -F.softplus(self.head_output(hidden)).squeeze(-1)

    def forward(self, *state_tensors: torch.Tensor) -> torch.Tensor:
        return self.value_scale * self.forward_normalized(*state_tensors)

    def predict_structured(self, features: object | Sequence[object]) -> np.ndarray:
        samples = list(features) if isinstance(features, (tuple, list)) else [features]
        if not samples:
            return np.empty(0, dtype=np.float32)
        tensors = _to_device(_collate_states(samples, self.spec), self.device)
        self.eval()
        with torch.inference_mode():
            result = self(*tensors).cpu().numpy()
        return result.astype(np.float32, copy=False)


class GV4PolicyDeepSet(_GV4ModelBase):
    """Rank canonical actions while encoding their shared root state once."""

    model_kind = "policy_dnn"

    def __init__(self, spec: GV4ModelSpec, *, role: str) -> None:
        super().__init__(spec, role=role)
        header_dim, request_dim = spec.action_dimensions(self.role)
        self.action_header_dim = header_dim
        self.action_request_dim = request_dim
        self.state_encoder = GV4StateEncoder(spec)
        self.action_encoder = _ActionEncoder(header_dim, request_dim)
        fusion_dim = GV4StateEncoder.output_dim + _ActionEncoder.output_dim
        self.fusion = nn.Linear(fusion_dim, 128)
        self.fusion_norm = nn.LayerNorm(128)
        self.fusion_block = BottleneckResidual(128, 32)
        self.head_norm = nn.LayerNorm(128)
        self.head_hidden = nn.Linear(128, 64)
        self.head_output = nn.Linear(64, 1)

    @property
    def model_name(self) -> str:
        return f"agz_gv4_policy_deepset_{self.role}"

    def encode_state(self, *state_tensors: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(*state_tensors)

    def forward_with_state_embedding(
        self,
        state_embedding: torch.Tensor,
        action_header: torch.Tensor,
        action_rows: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_embedding = self.action_encoder(action_header, action_rows, action_mask)
        if state_embedding.shape[0] != action_embedding.shape[0]:
            if state_embedding.shape[0] != 1:
                raise GV4ModelError("state/action batch sizes differ")
            state_embedding = state_embedding.expand(action_embedding.shape[0], -1)
        fused = torch.cat((state_embedding, action_embedding), dim=-1)
        hidden = F.silu(self.fusion_norm(self.fusion(fused)))
        hidden = self.fusion_block(hidden)
        hidden = F.silu(self.head_hidden(self.head_norm(hidden)))
        return self.head_output(hidden).squeeze(-1)

    def predict_root_structured(
        self, state_features: object, action_features: Sequence[object]
    ) -> np.ndarray:
        actions = tuple(action_features)
        if not actions:
            return np.empty(0, dtype=np.float32)
        state_tensors = _to_device(
            _collate_states((state_features,), self.spec), self.device
        )
        action_tensors = _to_device(
            _collate_actions(actions, self.spec, self.role), self.device
        )
        self.eval()
        with torch.inference_mode():
            state_embedding = self.encode_state(*state_tensors)
            result = (
                self.forward_with_state_embedding(state_embedding, *action_tensors)
                .cpu()
                .numpy()
            )
        return result.astype(np.float32, copy=False)

    def predict_structured(
        self,
        states: Sequence[object],
        action_features: Sequence[object],
        offsets: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        """Score several replay roots without rebuilding a state per action."""

        roots = tuple(states)
        actions = tuple(action_features)
        ranges = _policy_ranges(offsets, len(roots), len(actions))
        if not roots:
            return np.empty(0, dtype=np.float32)
        state_tensors = _to_device(_collate_states(roots, self.spec), self.device)
        action_tensors = _to_device(
            _collate_actions(actions, self.spec, self.role), self.device
        )
        counts = torch.tensor(
            [end - begin for begin, end in ranges],
            dtype=torch.int64,
            device=self.device,
        )
        self.eval()
        with torch.inference_mode():
            state_embedding = self.encode_state(*state_tensors)
            repeated = torch.repeat_interleave(state_embedding, counts, dim=0)
            result = (
                self.forward_with_state_embedding(repeated, *action_tensors)
                .cpu()
                .numpy()
            )
        return result.astype(np.float32, copy=False)


def is_dnn_model(model: Any) -> bool:
    return isinstance(model, (GV4ValueDeepSet, GV4PolicyDeepSet))


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


ModelT = TypeVar("ModelT", bound=_GV4ModelBase)


def _checkpoint(model: _GV4ModelBase) -> dict[str, Any]:
    values: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture_version": model.architecture_version,
        "model_kind": model.model_kind,
        "role": model.role,
        "spec": asdict(model.spec),
        "state_dict": model.state_dict(),
        "optimizer_state": model.optimizer_state,
        "training_metadata": model.training_metadata,
    }
    if isinstance(model, GV4ValueDeepSet):
        values["value_scale"] = model.value_scale
    return values


def save_dnn_model(model: _GV4ModelBase, path: str | Path) -> None:
    """Atomically save a data-only Torch checkpoint, not a pickled model object."""

    if not is_dnn_model(model):
        raise TypeError(f"unsupported GV4 DNN type: {type(model)!r}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{time.time_ns()}")
    with temporary.open("wb") as stream:
        torch.save(_checkpoint(model.cpu().eval()), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def load_dnn_model(
    path: str | Path,
    expected_type: type[ModelT] | None = None,
    *,
    expected_config: GV4EngineConfig | None = None,
    expected_role: str | None = None,
) -> ModelT | _GV4ModelBase:
    """Load a checkpoint and fail closed on schema, manifest, role, or type drift."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise GV4ModelError("GV4 model checkpoint is not a mapping")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise GV4ModelError("unsupported GV4 model checkpoint version")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise GV4ModelError("unsupported GV4 model architecture")

    spec = GV4ModelSpec.from_dict(payload.get("spec", {}))
    role = _role(str(payload.get("role", "")))
    if expected_config is not None and spec != GV4ModelSpec.from_config(
        expected_config
    ):
        raise GV4ModelError("checkpoint was trained for a different GV4 config")
    if expected_role is not None and role != _role(expected_role):
        raise GV4ModelError("checkpoint has the wrong player role")

    kind = payload.get("model_kind")
    if kind == GV4ValueDeepSet.model_kind:
        model: _GV4ModelBase = GV4ValueDeepSet(
            spec, role=role, value_scale=float(payload.get("value_scale", 0.0))
        )
    elif kind == GV4PolicyDeepSet.model_kind:
        model = GV4PolicyDeepSet(spec, role=role)
    else:
        raise GV4ModelError(f"unsupported GV4 model kind {kind!r}")
    if expected_type is not None and not isinstance(model, expected_type):
        raise TypeError(
            f"expected {expected_type.__name__}, got {type(model).__name__}"
        )

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise GV4ModelError("checkpoint has no state dictionary")
    model.load_state_dict(state_dict, strict=True)
    optimizer_state = payload.get("optimizer_state")
    model.optimizer_state = (
        dict(optimizer_state) if isinstance(optimizer_state, Mapping) else None
    )
    metadata = payload.get("training_metadata")
    model.training_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    return model.eval()


def artifact_metadata(model: _GV4ModelBase) -> dict[str, Any]:
    if not is_dnn_model(model):
        raise TypeError(f"unsupported GV4 DNN type: {type(model)!r}")
    metadata: dict[str, Any] = {
        "architecture_version": model.architecture_version,
        "model_kind": model.model_kind,
        "role": model.role,
        "model_spec": asdict(model.spec),
        "parameter_count": model_parameter_count(model),
        "training_metadata": dict(model.training_metadata),
    }
    if isinstance(model, GV4ValueDeepSet):
        metadata["value_scale"] = model.value_scale
    return metadata


def _atomic_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def write_artifact_metadata(model: _GV4ModelBase, path: str | Path) -> Path:
    text = json.dumps(artifact_metadata(model), indent=2, sort_keys=True) + "\n"
    return _atomic_text(path, text)


def export_dnn_to_native(
    model: _GV4ModelBase, path: str | Path, *, model_tag: str = ""
) -> Path:
    """Export named float32 tensors for the forthcoming standalone C++ loader."""

    metadata = artifact_metadata(model)
    metadata["native_export_version"] = NATIVE_EXPORT_VERSION
    metadata["model_tag"] = str(model_tag or model.role)
    lines = [NATIVE_EXPORT_VERSION, json.dumps(metadata, separators=(",", ":"))]
    for name, tensor in model.cpu().eval().state_dict().items():
        value = tensor.detach().to(torch.float32).contiguous()
        shape = ",".join(str(dimension) for dimension in value.shape)
        flattened = value.reshape(-1).numpy()
        numbers = ",".join(format(float(item), ".9g") for item in flattened)
        lines.append(f"tensor\t{name}\t{shape}\t{numbers}")
    return _atomic_text(path, "\n".join(lines) + "\n")


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GV4ModelError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GV4ModelError(f"{name} must be positive and finite")
    return result


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))


def _training_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise GV4ModelError("CUDA training was requested but CUDA is unavailable")
    return resolved


def _prepare_optimizer(
    model: _GV4ModelBase,
    *,
    lr: float,
    weight_decay: float,
    preserve_state: bool,
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    if preserve_state and model.optimizer_state:
        optimizer.load_state_dict(model.optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = float(lr)
            group["weight_decay"] = float(weight_decay)
    return optimizer


def _new_or_warm_value_model(
    config: GV4EngineConfig,
    role: str,
    initial_model_path: str | Path | None,
    value_scale: float,
) -> tuple[GV4ValueDeepSet, bool]:
    warm_start = bool(initial_model_path and Path(initial_model_path).is_file())
    if warm_start:
        loaded = load_dnn_model(
            Path(initial_model_path),
            GV4ValueDeepSet,
            expected_config=config,
            expected_role=role,
        )
        assert isinstance(loaded, GV4ValueDeepSet)
        if not math.isclose(loaded.value_scale, value_scale):
            raise GV4ModelError("warm-start value scale differs from requested scale")
        return loaded, True
    return (
        GV4ValueDeepSet(
            GV4ModelSpec.from_config(config), role=role, value_scale=value_scale
        ),
        False,
    )


def fit_value_dnn(
    features: Sequence[object],
    targets: Sequence[float] | np.ndarray,
    *,
    config: GV4EngineConfig,
    role: str,
    initial_model_path: str | Path | None = None,
    seed: int = 2026,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    torch_threads: int = 24,
    huber_delta: float = 0.1,
    value_scale: float = DEFAULT_VALUE_SCALE,
    device: str | torch.device | None = None,
) -> tuple[GV4ValueDeepSet, dict[str, float | int]]:
    """Incrementally fit one role's value model with normalized Huber loss."""

    samples = tuple(features)
    values = np.asarray(targets, dtype=np.float32).reshape(-1)
    if not samples or len(samples) != len(values):
        raise GV4ModelError(
            f"invalid value data: states={len(samples)} targets={values.shape}"
        )
    if not np.isfinite(values).all():
        raise GV4ModelError("value targets contain a non-finite number")
    if np.any(values > 1e-6):
        raise GV4ModelError("GV4 controller-valued return targets must be <= 0")

    _seed_all(seed)
    torch.set_num_threads(_positive_int("torch_threads", torch_threads))
    epochs = _positive_int("epochs", epochs)
    batch_size = _positive_int("batch_size", batch_size)
    huber_delta = _positive_float("huber_delta", huber_delta)
    value_scale = _positive_float("value_scale", value_scale)
    training_device = _training_device(device)
    model, warm_start = _new_or_warm_value_model(
        config, _role(role), initial_model_path, value_scale
    )
    model.to(training_device).train()
    optimizer = _prepare_optimizer(
        model, lr=lr, weight_decay=weight_decay, preserve_state=warm_start
    )

    target_tensor = torch.from_numpy(values / value_scale).to(training_device)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    final_huber = final_mse = math.nan
    for _ in range(epochs):
        order = rng.permutation(len(samples))
        total_huber = total_mse = 0.0
        seen = 0
        for begin in range(0, len(order), batch_size):
            indices = order[begin : begin + batch_size]
            batch = tuple(samples[int(index)] for index in indices)
            tensors = _to_device(_collate_states(batch, model.spec), training_device)
            index_tensor = torch.from_numpy(indices.astype(np.int64, copy=False)).to(
                training_device
            )
            target = target_tensor.index_select(0, index_tensor)
            prediction = model.forward_normalized(*tensors)
            loss = F.huber_loss(prediction, target, delta=huber_delta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count = len(indices)
            total_huber += float(loss.detach()) * count
            total_mse += float(F.mse_loss(prediction.detach(), target)) * count
            seen += count
        final_huber = total_huber / seen
        final_mse = total_mse / seen

    elapsed = time.perf_counter() - started
    model.optimizer_state = optimizer.state_dict()
    model.training_metadata = {
        "warm_start": warm_start,
        "epochs": epochs,
        "rows": len(samples),
        "normalized_huber": final_huber,
        "normalized_mse": final_mse,
        "huber_delta": huber_delta,
    }
    model.cpu().eval()
    metrics: dict[str, float | int] = {
        "elapsed_s": elapsed,
        "normalized_huber": final_huber,
        "normalized_mse": final_mse,
        "rows": len(samples),
        "warm_start": int(warm_start),
        "parameters": model_parameter_count(model),
    }
    return model, metrics


def _policy_ranges(
    offsets: Sequence[tuple[int, int]], root_count: int, action_count: int
) -> tuple[tuple[int, int], ...]:
    ranges = tuple((int(begin), int(end)) for begin, end in offsets)
    if len(ranges) != root_count:
        raise GV4ModelError("policy root and offset counts differ")
    cursor = 0
    for begin, end in ranges:
        if begin != cursor or end <= begin:
            raise GV4ModelError("policy offsets must be contiguous and nonempty")
        cursor = end
    if cursor != action_count:
        raise GV4ModelError("policy offsets do not cover every action")
    return ranges


def _new_or_warm_policy_model(
    config: GV4EngineConfig,
    role: str,
    initial_model_path: str | Path | None,
) -> tuple[GV4PolicyDeepSet, bool]:
    warm_start = bool(initial_model_path and Path(initial_model_path).is_file())
    if warm_start:
        loaded = load_dnn_model(
            Path(initial_model_path),
            GV4PolicyDeepSet,
            expected_config=config,
            expected_role=role,
        )
        assert isinstance(loaded, GV4PolicyDeepSet)
        return loaded, True
    return GV4PolicyDeepSet(GV4ModelSpec.from_config(config), role=role), False


def fit_policy_dnn(
    states: Sequence[object],
    action_features: Sequence[object],
    target_probabilities: Sequence[float] | np.ndarray,
    offsets: Sequence[tuple[int, int]],
    *,
    config: GV4EngineConfig,
    role: str,
    initial_model_path: str | Path | None = None,
    seed: int = 2026,
    epochs: int = 5,
    root_batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    torch_threads: int = 24,
    device: str | torch.device | None = None,
) -> tuple[GV4PolicyDeepSet, dict[str, float | int]]:
    """Fit visit distributions with one state encoding per replay root."""

    roots = tuple(states)
    actions = tuple(action_features)
    targets = np.asarray(target_probabilities, dtype=np.float32).reshape(-1)
    ranges = _policy_ranges(offsets, len(roots), len(actions))
    if not roots or len(targets) != len(actions):
        raise GV4ModelError(
            f"invalid policy data: roots={len(roots)} actions={len(actions)} "
            f"targets={targets.shape}"
        )
    if not np.isfinite(targets).all() or np.any(targets < 0.0):
        raise GV4ModelError("policy targets must be finite and nonnegative")

    _seed_all(seed)
    torch.set_num_threads(_positive_int("torch_threads", torch_threads))
    epochs = _positive_int("epochs", epochs)
    root_batch_size = _positive_int("root_batch_size", root_batch_size)
    training_device = _training_device(device)
    model, warm_start = _new_or_warm_policy_model(
        config, _role(role), initial_model_path
    )
    model.to(training_device).train()
    optimizer = _prepare_optimizer(
        model, lr=lr, weight_decay=weight_decay, preserve_state=warm_start
    )
    target_tensor = torch.from_numpy(targets).to(training_device)

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    final_loss = math.nan
    for _ in range(epochs):
        order = rng.permutation(len(roots))
        total_loss = 0.0
        seen_roots = 0
        for begin in range(0, len(order), root_batch_size):
            selected = order[begin : begin + root_batch_size]
            selected_roots = tuple(roots[int(index)] for index in selected)
            selected_ranges = tuple(ranges[int(index)] for index in selected)
            selected_actions = tuple(
                actions[index]
                for start, end in selected_ranges
                for index in range(start, end)
            )
            state_tensors = _to_device(
                _collate_states(selected_roots, model.spec), training_device
            )
            action_tensors = _to_device(
                _collate_actions(selected_actions, model.spec, model.role),
                training_device,
            )
            counts = [end - start for start, end in selected_ranges]
            state_embedding = model.encode_state(*state_tensors)
            repeated = torch.repeat_interleave(
                state_embedding,
                torch.tensor(counts, dtype=torch.int64, device=training_device),
                dim=0,
            )
            logits = model.forward_with_state_embedding(repeated, *action_tensors)

            losses: list[torch.Tensor] = []
            cursor = 0
            for (start, end), count in zip(selected_ranges, counts):
                target = target_tensor[start:end]
                total = target.sum()
                target = (
                    torch.full_like(target, 1.0 / count)
                    if float(total) <= 0.0
                    else target / total
                )
                root_logits = logits[cursor : cursor + count]
                losses.append(-(target * F.log_softmax(root_logits, dim=0)).sum())
                cursor += count
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.detach()) * len(selected_ranges)
            seen_roots += len(selected_ranges)
        final_loss = total_loss / seen_roots

    elapsed = time.perf_counter() - started
    model.optimizer_state = optimizer.state_dict()
    model.training_metadata = {
        "warm_start": warm_start,
        "epochs": epochs,
        "roots": len(roots),
        "actions": len(actions),
        "cross_entropy": final_loss,
        "state_encoded_once_per_root": True,
    }
    model.cpu().eval()
    metrics: dict[str, float | int] = {
        "elapsed_s": elapsed,
        "cross_entropy": final_loss,
        "roots": len(roots),
        "actions": len(actions),
        "warm_start": int(warm_start),
        "parameters": model_parameter_count(model),
    }
    return model, metrics


# These aliases keep the terminology used by the GV3 training code while the
# signatures remain explicitly GV4-structured.
fit_markov_value_dnn = fit_value_dnn
fit_markov_policy_dnn = fit_policy_dnn
