"""Versioned four-model bundles shared by self-play and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import torch

from GV4_Engine.config import GV4EngineConfig
from GV4_Engine.dnn_inference.inference import GV4DNNInference

from .dnn_models import (
    GV4ModelSpec,
    GV4PolicyDeepSet,
    GV4ValueDeepSet,
    export_dnn_to_native,
    load_dnn_model,
    save_dnn_model,
    write_artifact_metadata,
)
from .engine_runtime import BackendName


MODEL_BUNDLE_SCHEMA_VERSION = "gv4_model_bundle_v1"
CURRENT_MODEL_SCHEMA_VERSION = "gv4_current_model_v1"
MODEL_ARTIFACT_NAMES = (
    "controller_value",
    "adversary_value",
    "controller_policy",
    "adversary_policy",
)

_ARTIFACT_CONTRACT = {
    "controller_value": ("controller", "value_dnn", GV4ValueDeepSet),
    "adversary_value": ("adversary", "value_dnn", GV4ValueDeepSet),
    "controller_policy": ("controller", "policy_dnn", GV4PolicyDeepSet),
    "adversary_policy": ("adversary", "policy_dnn", GV4PolicyDeepSet),
}

__all__ = [
    "CURRENT_MODEL_SCHEMA_VERSION",
    "LoadedModelBundle",
    "MODEL_ARTIFACT_NAMES",
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "ModelArtifact",
    "ModelBundleError",
    "load_model_bundle",
    "publish_model_bundle",
    "sha256_file",
    "write_current_model_pointer",
]


class ModelBundleError(RuntimeError):
    """Raised when a model bundle is incomplete or incompatible with GV4."""


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One checkpoint and its human-readable/native companion files."""

    name: str
    role: str
    model_kind: str
    model_version: int
    checkpoint_path: str
    checkpoint_sha256: str
    metadata_path: str
    metadata_sha256: str
    native_export_path: str
    native_export_sha256: str


@dataclass(slots=True)
class LoadedModelBundle:
    """Validated models ready to be handed to either engine backend."""

    manifest_path: Path
    manifest_sha256: str
    bundle_version: int
    config_manifest_sha256: str
    feature_schema_version: str
    artifacts: dict[str, ModelArtifact]
    models: dict[str, GV4ValueDeepSet | GV4PolicyDeepSet]
    metadata: dict[str, Any]

    @property
    def model_versions(self) -> dict[str, int]:
        return {
            name: artifact.model_version for name, artifact in self.artifacts.items()
        }

    @property
    def role_versions(self) -> dict[str, int]:
        """Return one version per player; value and policy must move together."""

        controller = {
            self.artifacts["controller_value"].model_version,
            self.artifacts["controller_policy"].model_version,
        }
        adversary = {
            self.artifacts["adversary_value"].model_version,
            self.artifacts["adversary_policy"].model_version,
        }
        if len(controller) != 1 or len(adversary) != 1:
            raise ModelBundleError("value and policy versions differ within one role")
        return {
            "controller": next(iter(controller)),
            "adversary": next(iter(adversary)),
        }

    def create_inference(
        self,
        backend: BackendName,
        config: GV4EngineConfig,
    ) -> Any:
        """Build the backend-specific inference facade from the same four models."""

        _check_config_identity(
            config,
            config_manifest_sha256=self.config_manifest_sha256,
            feature_schema_version=self.feature_schema_version,
        )
        values = {
            "controller_value_model": self.models["controller_value"],
            "adversary_value_model": self.models["adversary_value"],
            "controller_policy_model": self.models["controller_policy"],
            "adversary_policy_model": self.models["adversary_policy"],
        }
        if backend == "python":
            return GV4DNNInference(config, **values)
        if backend == "native":
            from GV4_Cpp import gv4_native
            from GV4_Cpp.runtime import config_from_python

            return gv4_native.InferenceRuntime(config_from_python(config), **values)
        raise ValueError(f"unsupported GV4 backend {backend!r}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelBundleError(f"cannot read model manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelBundleError(f"model manifest is not an object: {path}")
    return value


def _check_config_identity(
    config: GV4EngineConfig,
    *,
    config_manifest_sha256: str,
    feature_schema_version: str,
) -> None:
    if config.manifest_sha256() != config_manifest_sha256:
        raise ModelBundleError("model bundle was built for a different GV4 config")
    if config.layout.feature_schema_version != feature_schema_version:
        raise ModelBundleError("model bundle uses a different feature schema")


def _safe_artifact_path(bundle_root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ModelBundleError("model artifact paths must be nonempty and relative")
    root = bundle_root.resolve()
    result = (root / relative_path).resolve()
    if result != root and root not in result.parents:
        raise ModelBundleError("model artifact path leaves its bundle directory")
    return result


def _verify_file(bundle_root: Path, relative_path: str, expected_sha256: str) -> Path:
    path = _safe_artifact_path(bundle_root, relative_path)
    if not path.is_file():
        raise ModelBundleError(f"model bundle artifact is missing: {relative_path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ModelBundleError(f"checksum mismatch for {relative_path}")
    return path


def _resolve_manifest(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "model_bundle.json"
    value = _read_json(source)
    if value.get("schema_version") != CURRENT_MODEL_SCHEMA_VERSION:
        return source

    relative = str(value.get("bundle_manifest_path", ""))
    manifest = _safe_artifact_path(source.parent, relative)
    expected = str(value.get("bundle_manifest_sha256", ""))
    if not manifest.is_file() or sha256_file(manifest) != expected:
        raise ModelBundleError("current-model pointer references an invalid manifest")
    return manifest


def load_model_bundle(
    path: str | Path,
    *,
    config: GV4EngineConfig,
    device: str | torch.device = "cpu",
) -> LoadedModelBundle:
    """Load all four models only after every declared artifact is verified."""

    manifest_path = _resolve_manifest(path)
    value = _read_json(manifest_path)
    if value.get("schema_version") != MODEL_BUNDLE_SCHEMA_VERSION:
        raise ModelBundleError("unsupported model bundle schema")

    config_hash = str(value.get("config_manifest_sha256", ""))
    feature_schema = str(value.get("feature_schema_version", ""))
    _check_config_identity(
        config,
        config_manifest_sha256=config_hash,
        feature_schema_version=feature_schema,
    )
    try:
        bundle_version = int(value["bundle_version"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelBundleError("model bundle has no valid version") from error
    if bundle_version < 0:
        raise ModelBundleError("model bundle version cannot be negative")

    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != set(
        MODEL_ARTIFACT_NAMES
    ):
        raise ModelBundleError("model bundle must contain exactly four artifacts")

    artifacts: dict[str, ModelArtifact] = {}
    models: dict[str, GV4ValueDeepSet | GV4PolicyDeepSet] = {}
    root = manifest_path.parent
    for name in MODEL_ARTIFACT_NAMES:
        raw = raw_artifacts[name]
        if not isinstance(raw, dict):
            raise ModelBundleError(f"{name} artifact is not an object")
        try:
            artifact = ModelArtifact(**raw)
        except TypeError as error:
            raise ModelBundleError(f"{name} artifact fields are invalid") from error
        expected_role, expected_kind, expected_type = _ARTIFACT_CONTRACT[name]
        if (
            artifact.name != name
            or artifact.role != expected_role
            or artifact.model_kind != expected_kind
            or artifact.model_version < 0
        ):
            raise ModelBundleError(f"{name} does not satisfy its role contract")

        checkpoint = _verify_file(
            root, artifact.checkpoint_path, artifact.checkpoint_sha256
        )
        _verify_file(root, artifact.metadata_path, artifact.metadata_sha256)
        _verify_file(root, artifact.native_export_path, artifact.native_export_sha256)
        model = load_dnn_model(
            checkpoint,
            expected_type,
            expected_config=config,
            expected_role=expected_role,
        )
        model.to(device).eval()
        artifacts[name] = artifact
        models[name] = model

    raw_metadata = value.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise ModelBundleError("model bundle metadata must be an object")

    result = LoadedModelBundle(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        bundle_version=bundle_version,
        config_manifest_sha256=config_hash,
        feature_schema_version=feature_schema,
        artifacts=artifacts,
        models=models,
        metadata=dict(raw_metadata),
    )
    result.role_versions
    return result


def _model_versions(
    bundle_version: int,
    values: Mapping[str, int] | None,
) -> dict[str, int]:
    if values is None:
        return {name: bundle_version for name in MODEL_ARTIFACT_NAMES}
    if set(values) != set(MODEL_ARTIFACT_NAMES):
        raise ModelBundleError("model_versions must name all four model artifacts")
    result = {name: int(values[name]) for name in MODEL_ARTIFACT_NAMES}
    if any(version < 0 for version in result.values()):
        raise ModelBundleError("model versions cannot be negative")
    return result


def publish_model_bundle(
    destination: str | Path,
    *,
    bundle_version: int,
    config: GV4EngineConfig,
    models: Mapping[str, GV4ValueDeepSet | GV4PolicyDeepSet],
    model_versions: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LoadedModelBundle:
    """Atomically publish four validated checkpoints and their manifest."""

    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if bundle_version < 0:
        raise ValueError("bundle_version cannot be negative")
    if set(models) != set(MODEL_ARTIFACT_NAMES):
        raise ModelBundleError("exactly four models are required")

    expected_spec = GV4ModelSpec.from_config(config)
    versions = _model_versions(bundle_version, model_versions)
    temporary = output.with_name(output.name + f".tmp.{time.time_ns()}")
    temporary.mkdir(parents=True)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        for name in MODEL_ARTIFACT_NAMES:
            expected_role, expected_kind, expected_type = _ARTIFACT_CONTRACT[name]
            model = models[name]
            if (
                not isinstance(model, expected_type)
                or model.role != expected_role
                or model.model_kind != expected_kind
                or model.spec != expected_spec
            ):
                raise ModelBundleError(f"{name} model does not match its contract")

            artifact_dir = temporary / name
            checkpoint = artifact_dir / "model.pt"
            model_metadata = artifact_dir / "metadata.json"
            native_export = artifact_dir / "native_model.tsv"
            save_dnn_model(model, checkpoint)
            write_artifact_metadata(model, model_metadata)
            export_dnn_to_native(model, native_export, model_tag=name)
            artifact = ModelArtifact(
                name=name,
                role=expected_role,
                model_kind=expected_kind,
                model_version=versions[name],
                checkpoint_path=str(checkpoint.relative_to(temporary)),
                checkpoint_sha256=sha256_file(checkpoint),
                metadata_path=str(model_metadata.relative_to(temporary)),
                metadata_sha256=sha256_file(model_metadata),
                native_export_path=str(native_export.relative_to(temporary)),
                native_export_sha256=sha256_file(native_export),
            )
            artifacts[name] = asdict(artifact)

        _atomic_json(
            temporary / "model_bundle.json",
            {
                "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
                "bundle_version": int(bundle_version),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_manifest_sha256": config.manifest_sha256(),
                "feature_schema_version": config.layout.feature_schema_version,
                "target_perspective": "controller",
                "artifacts": artifacts,
                "metadata": dict(metadata or {}),
            },
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_model_bundle(output, config=config)


def write_current_model_pointer(
    path: str | Path,
    bundle: LoadedModelBundle,
) -> Path:
    """Atomically point workers at one already-published immutable bundle."""

    destination = Path(path).expanduser().resolve()
    relative_manifest = os.path.relpath(bundle.manifest_path, destination.parent)
    _atomic_json(
        destination,
        {
            "schema_version": CURRENT_MODEL_SCHEMA_VERSION,
            "bundle_version": bundle.bundle_version,
            "bundle_manifest_path": relative_manifest,
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "config_manifest_sha256": bundle.config_manifest_sha256,
            "feature_schema_version": bundle.feature_schema_version,
            "model_versions": bundle.model_versions,
        },
    )
    return destination
