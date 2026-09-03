"""Checksum-verified replay shards, acknowledgements, and model broadcast."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import time
from typing import Any, Mapping

from GV4_Engine.config import GV4EngineConfig

from ..model_bundle import LoadedModelBundle, load_model_bundle
from ..training_and_evaluation.replay_dataset import (
    REPLAY_SCHEMA_VERSION,
    discover_replay_manifests,
    load_replay_partition,
)
from .cluster import HostSpec, copy_to_host, run_shell


REPLAY_SHARD_SCHEMA_VERSION = "gv4_replay_shard_v1"
SHARD_ACK_SCHEMA_VERSION = "gv4_replay_ack_v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

__all__ = [
    "REPLAY_SHARD_SCHEMA_VERSION",
    "SHARD_ACK_SCHEMA_VERSION",
    "ValidatedReplayShard",
    "ack_path",
    "freeze_replay_shard",
    "publish_replay_shard",
    "read_ack",
    "retire_acknowledged_shard",
    "sha256_file",
    "sync_model_bundle",
    "sync_current_model",
    "validate_replay_shard",
    "write_ack",
]


@dataclass(frozen=True, slots=True)
class ValidatedReplayShard:
    path: Path
    worker_id: str
    shard_id: str
    digest: str
    game_ids: tuple[int, ...]
    games: int
    state_rows: int
    action_rows: int
    controller_roots: int
    adversary_roots: int
    created_at_utc: str

    @property
    def identity(self) -> str:
        return f"{self.worker_id}/{self.shard_id}"


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


def _payload_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        path.relative_to(root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"shard_manifest.json", "SHA256SUMS"}
        and ".tmp." not in path.name
        and not path.name.endswith(".tmp")
    )


def _write_sums(root: Path) -> None:
    files = tuple(
        path.relative_to(root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    )
    content = "".join(
        f"{sha256_file(root / relative)}  {relative.as_posix()}\n" for relative in files
    )
    temporary = root / "SHA256SUMS.tmp"
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, root / "SHA256SUMS")


def _verify_sums(root: Path) -> None:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError("replay shard has no SHA256SUMS")
    declared: set[Path] = set()
    for line_number, line in enumerate(
        sums.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            expected, relative_text = line.split(None, 1)
        except ValueError as error:
            raise ValueError(f"malformed SHA256SUMS line {line_number}") from error
        relative = Path(relative_text.strip())
        if relative.is_absolute() or ".." in relative.parts or relative in declared:
            raise ValueError("unsafe or duplicate path in SHA256SUMS")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"replay shard checksum mismatch: {relative}")
        declared.add(relative)
    actual = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if declared != actual:
        raise ValueError("SHA256SUMS does not declare every shard file exactly once")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return value


def _safe_id(label: str, value: Any) -> str:
    result = str(value)
    if not _SAFE_ID.fullmatch(result):
        raise ValueError(f"{label} is invalid")
    return result


def validate_replay_shard(
    path: str | Path,
    config: GV4EngineConfig,
) -> ValidatedReplayShard:
    """Verify transport checksums and every contained structured replay root."""

    root = Path(path).expanduser().resolve()
    _verify_sums(root)
    value = _read_object(root / "shard_manifest.json")
    if value.get("schema_version") != REPLAY_SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported replay shard schema")
    if value.get("status") != "complete":
        raise ValueError("replay shard is not complete")
    if value.get("replay_schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("replay shard uses another replay schema")
    if value.get("config_manifest_sha256") != config.manifest_sha256():
        raise ValueError("replay shard uses another GV4 engine config")
    if value.get("feature_schema_version") != config.layout.feature_schema_version:
        raise ValueError("replay shard uses another feature schema")

    worker_id = _safe_id("worker_id", value.get("worker_id"))
    shard_id = _safe_id("shard_id", value.get("shard_id"))
    declared_files = value.get("files")
    if not isinstance(declared_files, Mapping):
        raise ValueError("replay shard has no file manifest")
    actual_payload = _payload_files(root)
    if set(declared_files) != {path.as_posix() for path in actual_payload}:
        raise ValueError("replay shard payload differs from its manifest")
    for relative in actual_payload:
        record = declared_files[relative.as_posix()]
        if not isinstance(record, Mapping):
            raise ValueError("replay shard file record is invalid")
        payload = root / relative
        if int(record.get("bytes", -1)) != payload.stat().st_size:
            raise ValueError(f"replay shard byte count differs: {relative}")
        if str(record.get("sha256", "")) != sha256_file(payload):
            raise ValueError(f"replay shard file digest differs: {relative}")

    manifests = discover_replay_manifests(root / "games")
    if not manifests:
        raise ValueError("replay shard contains no games")
    game_ids: list[int] = []
    state_rows = action_rows = controller = adversary = 0
    for manifest_path in manifests:
        partition = load_replay_partition(manifest_path, config)
        game_ids.append(partition.manifest.game_id)
        state_rows += partition.manifest.state_rows
        action_rows += partition.manifest.action_rows
        controller += sum(root.player == "controller" for root in partition.roots)
        adversary += sum(root.player == "adversary" for root in partition.roots)
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("replay shard repeats a game ID")
    expected = {
        "games": len(game_ids),
        "state_rows": state_rows,
        "action_rows": action_rows,
        "controller_roots": controller,
        "adversary_roots": adversary,
    }
    for name, actual in expected.items():
        if int(value.get(name, -1)) != actual:
            raise ValueError(f"replay shard {name} differs from its contents")
    if sorted(game_ids) != value.get("game_ids"):
        raise ValueError("replay shard game IDs differ from its contents")

    return ValidatedReplayShard(
        path=root,
        worker_id=worker_id,
        shard_id=shard_id,
        digest=sha256_file(root / "SHA256SUMS"),
        game_ids=tuple(sorted(game_ids)),
        games=len(game_ids),
        state_rows=state_rows,
        action_rows=action_rows,
        controller_roots=controller,
        adversary_roots=adversary,
        created_at_utc=str(value.get("created_at_utc", "")),
    )


def freeze_replay_shard(
    active_dir: str | Path,
    ready_root: str | Path,
    *,
    worker_id: str,
    shard_id: str,
    config: GV4EngineConfig,
) -> ValidatedReplayShard:
    """Rename mutable active data, add its manifest, then publish it atomically."""

    active = Path(active_dir).expanduser().resolve()
    ready = Path(ready_root).expanduser().resolve()
    worker_id = _safe_id("worker_id", worker_id)
    shard_id = _safe_id("shard_id", shard_id)
    if not active.is_dir():
        raise FileNotFoundError(active)
    source_manifests = discover_replay_manifests(active / "games")
    partitions = [load_replay_partition(path, config) for path in source_manifests]
    if not partitions:
        raise ValueError("cannot freeze an empty replay shard")

    game_ids = sorted(partition.manifest.game_id for partition in partitions)
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("active replay repeats a game ID")
    controller = sum(
        root.player == "controller" for part in partitions for root in part.roots
    )
    adversary = sum(
        root.player == "adversary" for part in partitions for root in part.roots
    )
    lineages: set[tuple[tuple[str, int], ...]] = set()
    for partition in partitions:
        raw = _read_object(partition.manifest.manifest_path).get("model_versions", {})
        if not isinstance(raw, Mapping):
            raise ValueError("replay model_versions must be an object")
        lineages.add(tuple(sorted((str(key), int(item)) for key, item in raw.items())))

    ready.mkdir(parents=True, exist_ok=True)
    destination = ready / shard_id
    if destination.exists():
        raise FileExistsError(destination)
    staging = ready / f".{shard_id}.tmp.{time.time_ns()}"
    os.replace(active, staging)
    try:
        files = {
            relative.as_posix(): {
                "bytes": (staging / relative).stat().st_size,
                "sha256": sha256_file(staging / relative),
            }
            for relative in _payload_files(staging)
        }
        _atomic_json(
            staging / "shard_manifest.json",
            {
                "schema_version": REPLAY_SHARD_SCHEMA_VERSION,
                "status": "complete",
                "worker_id": worker_id,
                "shard_id": shard_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_manifest_sha256": config.manifest_sha256(),
                "feature_schema_version": config.layout.feature_schema_version,
                "replay_schema_version": REPLAY_SCHEMA_VERSION,
                "games": len(partitions),
                "game_ids": game_ids,
                "state_rows": sum(part.manifest.state_rows for part in partitions),
                "action_rows": sum(part.manifest.action_rows for part in partitions),
                "controller_roots": controller,
                "adversary_roots": adversary,
                "model_lineages": [dict(items) for items in sorted(lineages)],
                "files": files,
            },
        )
        _write_sums(staging)
        validated = validate_replay_shard(staging, config)
        os.replace(staging, destination)
        return ValidatedReplayShard(
            path=destination,
            worker_id=validated.worker_id,
            shard_id=validated.shard_id,
            digest=validated.digest,
            game_ids=validated.game_ids,
            games=validated.games,
            state_rows=validated.state_rows,
            action_rows=validated.action_rows,
            controller_roots=validated.controller_roots,
            adversary_roots=validated.adversary_roots,
            created_at_utc=validated.created_at_utc,
        )
    except BaseException:
        if staging.exists() and not active.exists():
            os.replace(staging, active)
        raise


def _publish_local(
    shard: ValidatedReplayShard,
    coordinator_root: Path,
    config: GV4EngineConfig,
) -> Path:
    uploading_root = coordinator_root / "coordinator" / "incoming_uploading"
    incoming_root = coordinator_root / "coordinator" / "incoming"
    incoming = incoming_root / f"{shard.worker_id}__{shard.shard_id}"
    if incoming.exists():
        existing = validate_replay_shard(incoming, config)
        if existing.digest != shard.digest:
            raise ValueError("incoming shard identity already has another digest")
        return incoming
    uploading_root.mkdir(parents=True, exist_ok=True)
    incoming_root.mkdir(parents=True, exist_ok=True)
    temporary = uploading_root / (
        f".{shard.worker_id}__{shard.shard_id}.tmp.{time.time_ns()}"
    )
    shutil.copytree(shard.path, temporary)
    copied = validate_replay_shard(temporary, config)
    if copied.digest != shard.digest:
        raise ValueError("copied replay shard digest changed")
    os.replace(temporary, incoming)
    return incoming


def publish_replay_shard(
    shard_path: str | Path,
    coordinator: HostSpec,
    *,
    config: GV4EngineConfig,
) -> str:
    """Idempotently publish a ready shard into coordinator/incoming."""

    shard = validate_replay_shard(shard_path, config)
    coordinator_root = Path(coordinator.experiment_root)
    if coordinator.is_local:
        return str(_publish_local(shard, coordinator_root, config))

    name = f"{shard.worker_id}__{shard.shard_id}"
    uploading = (
        f"{coordinator.experiment_root.rstrip('/')}/coordinator/"
        f"incoming_uploading/{name}.{shard.digest[:12]}"
    )
    incoming = f"{coordinator.experiment_root.rstrip('/')}/coordinator/incoming/{name}"
    copy_to_host(shard.path, coordinator, uploading)
    sums_digest = shlex.quote(shard.digest)
    up = shlex.quote(uploading)
    final = shlex.quote(incoming)
    script = "\n".join(
        (
            "set -e",
            f"mkdir -p {shlex.quote(str(Path(incoming).parent))}",
            f"test $(sha256sum {up}/SHA256SUMS | awk '{{print $1}}') = {sums_digest}",
            f"(cd {up} && sha256sum -c SHA256SUMS >/dev/null)",
            f"if [ -d {final} ]; then",
            f"  test $(sha256sum {final}/SHA256SUMS | awk '{{print $1}}') = {sums_digest}",
            f"  rm -rf {up}",
            "else",
            f"  mv {up} {final}",
            "fi",
        )
    )
    run_shell(coordinator, script)
    return incoming


def ack_path(coordinator_root: str | Path, worker_id: str, shard_id: str) -> Path:
    return (
        Path(coordinator_root).expanduser().resolve()
        / "coordinator"
        / "acks"
        / _safe_id("worker_id", worker_id)
        / f"{_safe_id('shard_id', shard_id)}.json"
    )


def write_ack(
    coordinator_root: str | Path,
    shard: ValidatedReplayShard,
    *,
    accepted_path: str,
) -> Path:
    destination = ack_path(coordinator_root, shard.worker_id, shard.shard_id)
    _atomic_json(
        destination,
        {
            "schema_version": SHARD_ACK_SCHEMA_VERSION,
            "accepted": True,
            "worker_id": shard.worker_id,
            "shard_id": shard.shard_id,
            "shard_digest": shard.digest,
            "accepted_path": accepted_path,
            "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return destination


def read_ack(
    coordinator: HostSpec,
    *,
    worker_id: str,
    shard_id: str,
    expected_digest: str,
) -> dict[str, Any] | None:
    relative = f"coordinator/acks/{_safe_id('worker_id', worker_id)}/{_safe_id('shard_id', shard_id)}.json"
    path = f"{coordinator.experiment_root.rstrip('/')}/{relative}"
    if coordinator.is_local:
        source = Path(path)
        if not source.is_file():
            return None
        value = _read_object(source)
    else:
        completed = run_shell(
            coordinator,
            f"test -f {shlex.quote(path)} && cat {shlex.quote(path)}",
            check=False,
        )
        if completed.returncode != 0 or not (completed.stdout or "").strip():
            return None
        value = json.loads(completed.stdout or "{}")
    if (
        value.get("schema_version") != SHARD_ACK_SCHEMA_VERSION
        or value.get("accepted") is not True
        or value.get("worker_id") != worker_id
        or value.get("shard_id") != shard_id
        or value.get("shard_digest") != expected_digest
    ):
        raise ValueError("coordinator acknowledgement does not match the shard")
    return value


def retire_acknowledged_shard(
    shard_path: str | Path,
    coordinator: HostSpec,
    *,
    config: GV4EngineConfig,
) -> bool:
    shard = validate_replay_shard(shard_path, config)
    acknowledgement = read_ack(
        coordinator,
        worker_id=shard.worker_id,
        shard_id=shard.shard_id,
        expected_digest=shard.digest,
    )
    if acknowledgement is None:
        return False
    shutil.rmtree(shard.path)
    return True


def _pointer_for_remote(bundle: LoadedModelBundle) -> dict[str, Any]:
    return {
        "schema_version": "gv4_current_model_v1",
        "bundle_version": bundle.bundle_version,
        "bundle_manifest_path": (
            f"{bundle.manifest_path.parent.name}/model_bundle.json"
        ),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "config_manifest_sha256": bundle.config_manifest_sha256,
        "feature_schema_version": bundle.feature_schema_version,
        "model_versions": bundle.model_versions,
    }


def sync_current_model(
    current_model_path: str | Path,
    destination: HostSpec,
    *,
    config: GV4EngineConfig,
) -> str:
    """Copy immutable model files first, then atomically expose their pointer."""

    bundle = load_model_bundle(current_model_path, config=config, device="cpu")
    remote_manifest = sync_model_bundle(
        bundle.manifest_path, destination, config=config
    )
    remote_models = f"{destination.experiment_root.rstrip('/')}/models"

    pointer_value = _pointer_for_remote(bundle)
    pointer_value["bundle_manifest_path"] = str(
        Path(remote_manifest).relative_to(Path(remote_models))
    )
    if destination.is_local:
        pointer = Path(remote_models) / "current_model.json"
        _atomic_json(pointer, pointer_value)
        load_model_bundle(pointer, config=config, device="cpu")
        return str(pointer)

    local_temporary = bundle.manifest_path.parent.parent / (
        f".current_model.{destination.host_id}.{time.time_ns()}.json"
    )
    try:
        _atomic_json(local_temporary, pointer_value)
        remote_temporary = f"{remote_models}/.current_model.{time.time_ns()}.tmp"
        copy_to_host(local_temporary, destination, remote_temporary)
        run_shell(
            destination,
            f"mv {shlex.quote(remote_temporary)} {shlex.quote(remote_models + '/current_model.json')}",
        )
    finally:
        local_temporary.unlink(missing_ok=True)
    return f"{remote_models}/current_model.json"


def sync_model_bundle(
    bundle_path: str | Path,
    destination: HostSpec,
    *,
    config: GV4EngineConfig,
) -> str:
    """Stage one immutable validated bundle without changing the active pointer."""

    bundle = load_model_bundle(bundle_path, config=config, device="cpu")
    remote_models = f"{destination.experiment_root.rstrip('/')}/models"
    directory_name = f"{bundle.manifest_path.parent.name}_{bundle.manifest_sha256[:16]}"
    remote_bundle = f"{remote_models}/{directory_name}"
    copy_to_host(bundle.manifest_path.parent, destination, remote_bundle)
    remote_manifest = f"{remote_bundle}/model_bundle.json"
    if destination.is_local:
        load_model_bundle(remote_manifest, config=config, device="cpu")
    else:
        files = [(Path("model_bundle.json"), bundle.manifest_sha256)]
        for artifact in bundle.artifacts.values():
            files.extend(
                (
                    (Path(artifact.checkpoint_path), artifact.checkpoint_sha256),
                    (Path(artifact.metadata_path), artifact.metadata_sha256),
                    (Path(artifact.native_export_path), artifact.native_export_sha256),
                )
            )
        checks = ["set -e"]
        for relative, expected in files:
            remote_path = shlex.quote(f"{remote_bundle}/{relative.as_posix()}")
            checks.append(
                f"test \"$(sha256sum {remote_path} | awk '{{print $1}}')\" "
                f"= {shlex.quote(expected)}"
            )
        run_shell(
            destination,
            "\n".join(checks),
        )
    return remote_manifest
