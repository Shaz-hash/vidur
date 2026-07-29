"""Small incremental residual DNNs for GV3 AlphaGoZero.

The public prediction methods retain the existing value/prior contracts:
values are returned in cost units and policy predictions are unbounded logits.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .markov_value_features import (
    GLOBAL_DIM as MARKOV_GLOBAL_DIM,
    LAUNCH_DIM as MARKOV_LAUNCH_DIM,
    MARKOV_VALUE_SCHEMA,
    REQUEST_DIM as MARKOV_REQUEST_DIM,
    MarkovValueFeatures,
    build_markov_value_features,
    feature_schema_metadata,
)

VALUE_FEATURE_DIM = 226
CONTROLLER_ACTION_DIM = 43
ADVERSARY_ACTION_DIM = 7
VALUE_MIN = -50.0
VALUE_MAX = 0.0
ARCHITECTURE_VERSION = "agz_residual_mlp_v1"
MARKOV_ARCHITECTURE_VERSION = "agz_markov_value_deepset_v2"
NATIVE_EXPORT_VERSION = "agz_dnn_v1"
MARKOV_NATIVE_EXPORT_VERSION = "agz_dnn_v2"
def normalize_value(value: torch.Tensor) -> torch.Tensor:
    return value / 25.0 + 1.0


def denormalize_value(value: torch.Tensor) -> torch.Tensor:
    return 25.0 * (value - 1.0)


class BottleneckResidual(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class ValueResidualMLP(nn.Module):
    model_kind = "value_dnn"
    architecture_version = ARCHITECTURE_VERSION
    feature_dim = VALUE_FEATURE_DIM

    def __init__(self, *, role: str = "controller") -> None:
        super().__init__()
        self.role = str(role)
        self.stem = nn.Linear(VALUE_FEATURE_DIM, 192)
        self.block1 = BottleneckResidual(192, 64)
        self.block2 = BottleneckResidual(192, 64)
        self.head_norm = nn.LayerNorm(192)
        self.head1 = nn.Linear(192, 32)
        self.head2 = nn.Linear(32, 1)
        self.optimizer_state: dict[str, Any] | None = None
        self.training_metadata: dict[str, Any] = {}
        self.runtime_prediction_cache: dict[bytes, float] = {}

    @property
    def model_name(self) -> str:
        return f"agz_value_residual_mlp_{self.role}"

    @property
    def feature_config(self) -> dict[str, Any]:
        return {"name": "v4_state_local", "feature_dim": VALUE_FEATURE_DIM}

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
    def cached_predictions(self) -> dict[bytes, float]:
        cache = getattr(self, "runtime_prediction_cache", None)
        if cache is None:
            cache = {}
            self.runtime_prediction_cache = cache
        return cache

    def forward_normalized(self, state_features: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.stem(state_features))
        x = self.block1(x)
        x = self.block2(x)
        x = F.silu(self.head1(self.head_norm(x)))
        return torch.tanh(self.head2(x)).squeeze(-1)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return denormalize_value(self.forward_normalized(state_features))

    def predict(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != VALUE_FEATURE_DIM:
            raise ValueError(f"value DNN expected [N,{VALUE_FEATURE_DIM}], got {x.shape}")
        self.eval()
        with torch.inference_mode():
            out = self(torch.from_numpy(np.ascontiguousarray(x))).cpu().numpy()
        return out.astype(np.float32, copy=False)

    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: Any | None = None,
    ) -> tuple[float, list[float]]:
        """Implement the unchanged GV3 Python MCTS value-model contract."""
        del player, device
        extras = getattr(inputs, "extras", None)
        if not extras:
            raise RuntimeError(
                "ValueResidualMLP.infer_from_inputs requires inputs.extras; "
                "enable the existing GV3 input-extras feature path"
            )
        from vidur.bellman_v4_adv.build_state_local_features_adv import (
            extract_features_one_record,
        )

        features = np.asarray(
            extract_features_one_record(
                {
                    "simulator_snapshot": extras.get("simulator_snapshot") or {},
                    "stats": extras.get("stats"),
                    "root_id": -1,
                }
            ),
            dtype=np.float32,
        ).reshape(-1)
        if features.size != VALUE_FEATURE_DIM:
            raise RuntimeError(
                f"ValueResidualMLP extracted {features.size} features; "
                f"expected {VALUE_FEATURE_DIM}"
            )
        cache = getattr(self, "runtime_prediction_cache", None)
        if cache is None:
            cache = {}
            self.runtime_prediction_cache = cache
        key = features.tobytes()
        cached = cache.get(key)
        if cached is None:
            cached = float(self.predict(features)[0])
            if len(cache) >= 50_000:
                cache.clear()
            cache[key] = cached
        return float(cached), []


def _collate_markov_features(
    samples: Sequence[MarkovValueFeatures],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty Markov value batch")
    batch_size = len(samples)
    max_requests = max(1, max(int(sample.request_features.shape[0]) for sample in samples))
    max_launches = max(1, max(int(sample.launch_features.shape[0]) for sample in samples))
    global_values = np.empty((batch_size, MARKOV_GLOBAL_DIM), dtype=np.float32)
    request_values = np.zeros((batch_size, max_requests, MARKOV_REQUEST_DIM), dtype=np.float32)
    request_mask = np.zeros((batch_size, max_requests), dtype=np.bool_)
    launch_values = np.zeros((batch_size, max_launches, MARKOV_LAUNCH_DIM), dtype=np.float32)
    launch_mask = np.zeros((batch_size, max_launches), dtype=np.bool_)
    for index, sample in enumerate(samples):
        if not isinstance(sample, MarkovValueFeatures):
            raise TypeError(f"expected MarkovValueFeatures, got {type(sample)!r}")
        global_values[index] = sample.global_features
        request_count = int(sample.request_features.shape[0])
        launch_count = int(sample.launch_features.shape[0])
        if request_count:
            request_values[index, :request_count] = sample.request_features
            request_mask[index, :request_count] = True
        if launch_count:
            launch_values[index, :launch_count] = sample.launch_features
            launch_mask[index, :launch_count] = True
    return (
        torch.from_numpy(global_values),
        torch.from_numpy(request_values),
        torch.from_numpy(request_mask),
        torch.from_numpy(launch_values),
        torch.from_numpy(launch_mask),
    )


class MarkovValueDeepSet(nn.Module):
    """Permutation-invariant value network over complete live GV3 state sets."""

    model_kind = "value_dnn"
    architecture_version = MARKOV_ARCHITECTURE_VERSION
    feature_schema = MARKOV_VALUE_SCHEMA
    feature_dim = 0
    state_dim = 0
    global_dim = MARKOV_GLOBAL_DIM
    request_dim = MARKOV_REQUEST_DIM
    launch_dim = MARKOV_LAUNCH_DIM

    def __init__(self, *, role: str = "controller") -> None:
        super().__init__()
        self.role = str(role)
        self.global_fc = nn.Linear(MARKOV_GLOBAL_DIM, 64)
        self.global_norm = nn.LayerNorm(64)
        self.request_fc1 = nn.Linear(MARKOV_REQUEST_DIM, 64)
        self.request_fc2 = nn.Linear(64, 64)
        self.request_norm = nn.LayerNorm(64)
        self.launch_fc1 = nn.Linear(MARKOV_LAUNCH_DIM, 32)
        self.launch_fc2 = nn.Linear(32, 32)
        self.launch_norm = nn.LayerNorm(32)
        self.fusion_fc = nn.Linear(256, 192)
        self.block1 = BottleneckResidual(192, 64)
        self.block2 = BottleneckResidual(192, 64)
        self.head_norm = nn.LayerNorm(192)
        self.head1 = nn.Linear(192, 32)
        self.head2 = nn.Linear(32, 1)
        self.optimizer_state: dict[str, Any] | None = None
        self.training_metadata: dict[str, Any] = {}
        self.runtime_prediction_cache: dict[bytes, float] = {}

    @property
    def model_name(self) -> str:
        return f"agz_markov_value_deepset_{self.role}"

    @property
    def feature_config(self) -> dict[str, Any]:
        return feature_schema_metadata()

    @property
    def trainable_params(self) -> int:
        return model_parameter_count(self)

    @property
    def uses_neural_network(self) -> bool:
        return True

    @property
    def uses_target_leakage(self) -> bool:
        return False

    @staticmethod
    def _masked_sum_max(
        encoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded.ndim != 3 or mask.ndim != 2 or encoded.shape[:2] != mask.shape:
            raise ValueError(
                f"invalid encoded/mask shapes: {tuple(encoded.shape)} {tuple(mask.shape)}"
            )
        mask_bool = mask.to(torch.bool)
        mask_values = mask_bool.unsqueeze(-1)
        summed = torch.sum(encoded * mask_values.to(encoded.dtype), dim=1)
        lowest = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask_values, lowest).max(dim=1).values
        has_any = mask_bool.any(dim=1, keepdim=True)
        maximum = torch.where(has_any, maximum, torch.zeros_like(maximum))
        return summed, maximum

    def forward_normalized(
        self,
        global_features: torch.Tensor,
        request_features: torch.Tensor,
        request_mask: torch.Tensor,
        launch_features: torch.Tensor,
        launch_mask: torch.Tensor,
    ) -> torch.Tensor:
        global_embedding = F.silu(self.global_norm(self.global_fc(global_features)))
        request_embedding = F.silu(self.request_fc1(request_features))
        request_embedding = F.silu(self.request_norm(self.request_fc2(request_embedding)))
        request_sum, request_max = self._masked_sum_max(request_embedding, request_mask)
        launch_embedding = F.silu(self.launch_fc1(launch_features))
        launch_embedding = F.silu(self.launch_norm(self.launch_fc2(launch_embedding)))
        launch_sum, launch_max = self._masked_sum_max(launch_embedding, launch_mask)
        fused = torch.cat(
            (global_embedding, request_sum, request_max, launch_sum, launch_max),
            dim=-1,
        )
        x = F.silu(self.fusion_fc(fused))
        x = self.block1(x)
        x = self.block2(x)
        x = F.silu(self.head1(self.head_norm(x)))
        return torch.tanh(self.head2(x)).squeeze(-1)

    def forward(
        self,
        global_features: torch.Tensor,
        request_features: torch.Tensor,
        request_mask: torch.Tensor,
        launch_features: torch.Tensor,
        launch_mask: torch.Tensor,
    ) -> torch.Tensor:
        return denormalize_value(
            self.forward_normalized(
                global_features,
                request_features,
                request_mask,
                launch_features,
                launch_mask,
            )
        )

    def predict_structured(
        self,
        features: MarkovValueFeatures | Sequence[MarkovValueFeatures],
    ) -> np.ndarray:
        samples = [features] if isinstance(features, MarkovValueFeatures) else list(features)
        tensors = _collate_markov_features(samples)
        self.eval()
        with torch.inference_mode():
            result = self(*tensors).cpu().numpy()
        return result.astype(np.float32, copy=False)

    def infer_from_inputs(
        self,
        inputs: Any,
        player: str,
        *,
        device: Any | None = None,
    ) -> tuple[float, list[float]]:
        del player, device
        extras = getattr(inputs, "extras", None) or {}
        value_features = extras.get("markov_value_features")
        if isinstance(value_features, MarkovValueFeatures):
            sample = value_features
        else:
            payload = extras.get("markov_state_payload")
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "MarkovValueDeepSet requires markov_value_features or "
                    "markov_state_payload in ModelInputs.extras"
                )
            sample = build_markov_value_features(payload)
        key = (
            sample.global_features.tobytes()
            + sample.request_features.shape[0].to_bytes(4, "little")
            + sample.request_features.tobytes()
            + sample.launch_features.shape[0].to_bytes(4, "little")
            + sample.launch_features.tobytes()
        )
        cached = self.runtime_prediction_cache.get(key)
        if cached is None:
            cached = float(self.predict_structured(sample)[0])
            if len(self.runtime_prediction_cache) >= 50_000:
                self.runtime_prediction_cache.clear()
            self.runtime_prediction_cache[key] = cached
        return cached, []


class PolicyRankMLP(nn.Module):
    model_kind = "policy_dnn"
    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, action_dim: int, *, role: str) -> None:
        super().__init__()
        if int(action_dim) not in {CONTROLLER_ACTION_DIM, ADVERSARY_ACTION_DIM}:
            raise ValueError(f"unsupported action dimension: {action_dim}")
        self.role = str(role)
        self.action_dim = int(action_dim)
        self.feature_dim = VALUE_FEATURE_DIM + self.action_dim
        self.state_fc = nn.Linear(VALUE_FEATURE_DIM, 192)
        self.state_norm = nn.LayerNorm(192)
        self.action_fc = nn.Linear(self.action_dim, 64)
        self.action_norm = nn.LayerNorm(64)
        self.fusion_fc = nn.Linear(256, 128)
        self.fusion_norm = nn.LayerNorm(128)
        self.fusion_block = BottleneckResidual(128, 32)
        self.head_norm = nn.LayerNorm(128)
        self.head1 = nn.Linear(128, 64)
        self.head2 = nn.Linear(64, 1)
        self.optimizer_state: dict[str, Any] | None = None
        self.training_metadata: dict[str, Any] = {}

    def encode_state(self, state_features: torch.Tensor) -> torch.Tensor:
        return F.silu(self.state_norm(self.state_fc(state_features)))

    def forward_with_state_embedding(
        self,
        state_embedding: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        action_embedding = F.silu(self.action_norm(self.action_fc(action_features)))
        if state_embedding.shape[0] == 1 and action_embedding.shape[0] != 1:
            state_embedding = state_embedding.expand(action_embedding.shape[0], -1)
        x = torch.cat((state_embedding, action_embedding), dim=-1)
        x = F.silu(self.fusion_norm(self.fusion_fc(x)))
        x = self.fusion_block(x)
        x = F.silu(self.head1(self.head_norm(x)))
        return self.head2(x).squeeze(-1)

    def forward(self, state_features: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_state_embedding(self.encode_state(state_features), action_features)

    def predict(self, features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"policy DNN expected [N,{self.feature_dim}], got {x.shape}")
        self.eval()
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x))
            out = self(t[:, :VALUE_FEATURE_DIM], t[:, VALUE_FEATURE_DIM:]).cpu().numpy()
        return out.astype(np.float32, copy=False)

    def predict_root(
        self,
        state_features: np.ndarray | Sequence[float],
        action_features: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        state = np.asarray(state_features, dtype=np.float32).reshape(1, VALUE_FEATURE_DIM)
        actions = np.asarray(action_features, dtype=np.float32).reshape(-1, self.action_dim)
        self.eval()
        with torch.inference_mode():
            state_t = torch.from_numpy(np.ascontiguousarray(state))
            action_t = torch.from_numpy(np.ascontiguousarray(actions))
            out = self.forward_with_state_embedding(self.encode_state(state_t), action_t).cpu().numpy()
        return out.astype(np.float32, copy=False)


def is_dnn_model(model: Any) -> bool:
    return isinstance(model, (ValueResidualMLP, MarkovValueDeepSet, PolicyRankMLP))


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def _atomic_joblib_dump(model: nn.Module, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    joblib.dump(model.cpu().eval(), tmp, compress=3)
    tmp.replace(path)


def save_dnn_model(model: nn.Module, path: Path) -> None:
    _atomic_joblib_dump(model, Path(path))


def load_dnn_model(path: Path, expected_type: type[nn.Module] | None = None) -> nn.Module:
    model = joblib.load(Path(path))
    if not is_dnn_model(model):
        raise TypeError(f"not an AlphaGoZero DNN artifact: {path}")
    if expected_type is not None and not isinstance(model, expected_type):
        raise TypeError(f"expected {expected_type.__name__}, got {type(model).__name__}")
    return model.cpu().eval()


def _write_native_tensor(f: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    shape = list(value.shape)
    flat = value.reshape(-1).numpy()
    shape_text = ",".join(str(int(x)) for x in shape)
    values_text = ",".join(format(float(x), ".9g") for x in flat)
    f.write(f"tensor\t{name}\t{shape_text}\t{values_text}\n")


def export_dnn_to_native(model: nn.Module, path: Path, *, model_tag: str = "") -> Path:
    if not is_dnn_model(model):
        raise TypeError(f"unsupported DNN export type: {type(model)!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    role = str(getattr(model, "role", ""))
    action_dim = int(getattr(model, "action_dim", 0))
    feature_dim = int(getattr(model, "feature_dim", VALUE_FEATURE_DIM))
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(f"{MARKOV_NATIVE_EXPORT_VERSION if isinstance(model, MarkovValueDeepSet) else NATIVE_EXPORT_VERSION}\n")
        f.write(f"model_kind\t{model.model_kind}\n")
        f.write(f"architecture\t{getattr(model, 'architecture_version', ARCHITECTURE_VERSION)}\n")
        f.write(f"model_tag\t{str(model_tag or role)}\n")
        f.write(f"role\t{role}\n")
        f.write(f"feature_dim\t{feature_dim}\n")
        f.write(f"state_dim\t{int(getattr(model, 'state_dim', VALUE_FEATURE_DIM))}\n")
        f.write(f"action_dim\t{action_dim}\n")
        f.write(f"value_min\t{VALUE_MIN:.17g}\n")
        f.write(f"value_max\t{VALUE_MAX:.17g}\n")
        f.write("layer_norm_eps\t1e-5\n")
        f.write(f"feature_schema\t{getattr(model, 'feature_schema', 'legacy_226')}\n")
        f.write(f"global_dim\t{int(getattr(model, 'global_dim', 0))}\n")
        f.write(f"request_dim\t{int(getattr(model, 'request_dim', 0))}\n")
        f.write(f"launch_dim\t{int(getattr(model, 'launch_dim', 0))}\n")
        for name, tensor in model.state_dict().items():
            _write_native_tensor(f, name, tensor)
    tmp.replace(path)
    return path


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))


def _prepare_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    preserve_optimizer: bool,
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    previous = getattr(model, "optimizer_state", None)
    if preserve_optimizer and previous:
        optimizer.load_state_dict(previous)
        for group in optimizer.param_groups:
            group["lr"] = float(lr)
            group["weight_decay"] = float(weight_decay)
    return optimizer


def fit_value_dnn(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    role: str,
    initial_model_path: Path | None = None,
    seed: int = 2026,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    torch_threads: int = 24,
) -> tuple[ValueResidualMLP, dict[str, float | int]]:
    _seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    x = np.ascontiguousarray(features, dtype=np.float32)
    y = np.ascontiguousarray(targets, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or x.shape[1] != VALUE_FEATURE_DIM or x.shape[0] != y.shape[0]:
        raise ValueError(f"invalid value arrays: X={x.shape} y={y.shape}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("value training arrays contain non-finite values")
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_min < VALUE_MIN - 1e-6 or y_max > VALUE_MAX + 1e-6:
        raise ValueError(f"value target outside [{VALUE_MIN},{VALUE_MAX}]: min={y_min} max={y_max}")

    warm_start = bool(initial_model_path and Path(initial_model_path).is_file())
    if warm_start:
        model = load_dnn_model(Path(initial_model_path), ValueResidualMLP)
        assert isinstance(model, ValueResidualMLP)
        model.role = str(role)
    else:
        model = ValueResidualMLP(role=str(role))
    model.train()
    optimizer = _prepare_optimizer(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        preserve_optimizer=warm_start,
    )
    x_t = torch.from_numpy(x)
    y_t = normalize_value(torch.from_numpy(y))
    rng = np.random.default_rng(int(seed))
    start = time.perf_counter()
    final_loss = math.nan
    for _ in range(max(1, int(epochs))):
        order = rng.permutation(x.shape[0])
        running = 0.0
        seen = 0
        for begin in range(0, order.size, max(1, int(batch_size))):
            idx = torch.from_numpy(order[begin : begin + int(batch_size)].astype(np.int64, copy=False))
            pred = model.forward_normalized(x_t.index_select(0, idx))
            target = y_t.index_select(0, idx)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int(idx.numel())
            running += float(loss.detach()) * n
            seen += n
        final_loss = running / max(1, seen)
    elapsed = time.perf_counter() - start
    model.optimizer_state = optimizer.state_dict()
    model.training_metadata = {
        "warm_start": warm_start,
        "epochs": int(epochs),
        "rows": int(x.shape[0]),
        "normalized_mse": float(final_loss),
    }
    model.eval()
    return model, {
        "elapsed_s": float(elapsed),
        "normalized_mse": float(final_loss),
        "rows": int(x.shape[0]),
        "warm_start": int(warm_start),
        "parameters": model_parameter_count(model),
    }


def fit_markov_value_dnn(
    features: Sequence[MarkovValueFeatures],
    targets: np.ndarray,
    *,
    role: str,
    initial_model_path: Path | None = None,
    seed: int = 2026,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    torch_threads: int = 24,
    huber_delta: float = 0.1,
) -> tuple[MarkovValueDeepSet, dict[str, float | int]]:
    _seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    samples = list(features)
    y = np.ascontiguousarray(targets, dtype=np.float32).reshape(-1)
    if not samples or len(samples) != int(y.shape[0]):
        raise ValueError(f"invalid Markov value arrays: X={len(samples)} y={y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("Markov value targets contain non-finite values")
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_min < VALUE_MIN - 1e-6 or y_max > VALUE_MAX + 1e-6:
        raise ValueError(f"value target outside [{VALUE_MIN},{VALUE_MAX}]: min={y_min} max={y_max}")
    for sample in samples:
        if not isinstance(sample, MarkovValueFeatures):
            raise TypeError(f"expected MarkovValueFeatures, got {type(sample)!r}")

    warm_start = bool(initial_model_path and Path(initial_model_path).is_file())
    if warm_start:
        model = load_dnn_model(Path(initial_model_path), MarkovValueDeepSet)
        assert isinstance(model, MarkovValueDeepSet)
        model.role = str(role)
    else:
        model = MarkovValueDeepSet(role=str(role))
    model.train()
    optimizer = _prepare_optimizer(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        preserve_optimizer=warm_start,
    )
    y_t = normalize_value(torch.from_numpy(y))
    rng = np.random.default_rng(int(seed))
    start = time.perf_counter()
    final_huber = math.nan
    final_mse = math.nan
    for _ in range(max(1, int(epochs))):
        order = rng.permutation(len(samples))
        running_huber = 0.0
        running_mse = 0.0
        seen = 0
        for begin in range(0, order.size, max(1, int(batch_size))):
            chosen = order[begin : begin + int(batch_size)]
            batch = [samples[int(index)] for index in chosen]
            tensors = _collate_markov_features(batch)
            target = y_t.index_select(
                0,
                torch.from_numpy(chosen.astype(np.int64, copy=False)),
            )
            pred = model.forward_normalized(*tensors)
            loss = F.huber_loss(pred, target, delta=float(huber_delta))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int(chosen.size)
            running_huber += float(loss.detach()) * n
            running_mse += float(F.mse_loss(pred.detach(), target)) * n
            seen += n
        final_huber = running_huber / max(1, seen)
        final_mse = running_mse / max(1, seen)
    elapsed = time.perf_counter() - start
    model.optimizer_state = optimizer.state_dict()
    model.training_metadata = {
        "warm_start": warm_start,
        "epochs": int(epochs),
        "rows": int(len(samples)),
        "normalized_huber": float(final_huber),
        "normalized_mse": float(final_mse),
        "huber_delta": float(huber_delta),
        "feature_schema": MARKOV_VALUE_SCHEMA,
    }
    model.eval()
    return model, {
        "elapsed_s": float(elapsed),
        "normalized_huber": float(final_huber),
        "normalized_mse": float(final_mse),
        "rows": int(len(samples)),
        "warm_start": int(warm_start),
        "parameters": model_parameter_count(model),
    }


def fit_policy_dnn(
    features: np.ndarray,
    target_probabilities: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    *,
    role: str,
    action_dim: int,
    initial_model_path: Path | None = None,
    seed: int = 2026,
    epochs: int = 5,
    root_batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    torch_threads: int = 24,
) -> tuple[PolicyRankMLP, dict[str, float | int]]:
    _seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    x = np.ascontiguousarray(features, dtype=np.float32)
    target = np.ascontiguousarray(target_probabilities, dtype=np.float32).reshape(-1)
    expected_dim = VALUE_FEATURE_DIM + int(action_dim)
    if x.ndim != 2 or x.shape[1] != expected_dim or x.shape[0] != target.shape[0]:
        raise ValueError(f"invalid policy arrays: X={x.shape} target={target.shape}")
    roots = [(int(a), int(b)) for a, b in offsets if int(b) > int(a)]
    if not roots:
        raise ValueError("policy training has no non-empty roots")

    warm_start = bool(initial_model_path and Path(initial_model_path).is_file())
    if warm_start:
        model = load_dnn_model(Path(initial_model_path), PolicyRankMLP)
        assert isinstance(model, PolicyRankMLP)
        if model.action_dim != int(action_dim):
            raise ValueError(f"warm-start action dim {model.action_dim} != {action_dim}")
        model.role = str(role)
    else:
        model = PolicyRankMLP(int(action_dim), role=str(role))
    model.train()
    optimizer = _prepare_optimizer(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        preserve_optimizer=warm_start,
    )
    x_t = torch.from_numpy(x)
    target_t = torch.from_numpy(target)
    rng = np.random.default_rng(int(seed))
    start_time = time.perf_counter()
    final_loss = math.nan
    for _ in range(max(1, int(epochs))):
        root_order = rng.permutation(len(roots))
        running = 0.0
        seen_roots = 0
        for begin in range(0, root_order.size, max(1, int(root_batch_size))):
            chosen = [roots[int(i)] for i in root_order[begin : begin + int(root_batch_size)]]
            row_indices = np.concatenate(
                [np.arange(a, b, dtype=np.int64) for a, b in chosen],
                dtype=np.int64,
            )
            idx_t = torch.from_numpy(row_indices)
            batch_x = x_t.index_select(0, idx_t)
            logits = model(
                batch_x[:, :VALUE_FEATURE_DIM],
                batch_x[:, VALUE_FEATURE_DIM:],
            )
            losses: list[torch.Tensor] = []
            cursor = 0
            for a, b in chosen:
                count = b - a
                root_target = target_t[a:b]
                root_sum = torch.sum(root_target)
                if float(root_sum) <= 0.0:
                    root_target = torch.full_like(root_target, 1.0 / float(count))
                else:
                    root_target = root_target / root_sum
                root_logits = logits[cursor : cursor + count]
                losses.append(-(root_target * F.log_softmax(root_logits, dim=0)).sum())
                cursor += count
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(chosen)
            seen_roots += len(chosen)
        final_loss = running / max(1, seen_roots)
    elapsed = time.perf_counter() - start_time
    model.optimizer_state = optimizer.state_dict()
    model.training_metadata = {
        "warm_start": warm_start,
        "epochs": int(epochs),
        "roots": int(len(roots)),
        "rows": int(x.shape[0]),
        "cross_entropy": float(final_loss),
    }
    model.eval()
    return model, {
        "elapsed_s": float(elapsed),
        "cross_entropy": float(final_loss),
        "roots": int(len(roots)),
        "rows": int(x.shape[0]),
        "warm_start": int(warm_start),
        "parameters": model_parameter_count(model),
    }


def artifact_metadata(model: nn.Module) -> dict[str, Any]:
    return {
        "architecture_version": str(getattr(model, "architecture_version", ARCHITECTURE_VERSION)),
        "model_kind": str(getattr(model, "model_kind", "")),
        "role": str(getattr(model, "role", "")),
        "feature_dim": int(getattr(model, "feature_dim", 0)),
        "state_dim": int(getattr(model, "state_dim", VALUE_FEATURE_DIM)),
        "action_dim": int(getattr(model, "action_dim", 0)),
        "feature_schema": str(getattr(model, "feature_schema", "legacy_226")),
        "global_dim": int(getattr(model, "global_dim", 0)),
        "request_dim": int(getattr(model, "request_dim", 0)),
        "launch_dim": int(getattr(model, "launch_dim", 0)),
        "value_min": VALUE_MIN,
        "value_max": VALUE_MAX,
        "parameter_count": model_parameter_count(model),
        "training_metadata": dict(getattr(model, "training_metadata", {})),
    }


def write_artifact_metadata(model: nn.Module, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    tmp.write_text(json.dumps(artifact_metadata(model), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
