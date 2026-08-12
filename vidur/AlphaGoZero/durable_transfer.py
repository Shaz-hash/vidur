
"""Durable shard/model transfer helpers for AlphaGoZero GV3."""

from __future__ import annotations

import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def local_time_24h() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def csv_row_count(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def replay_counts(path: Path) -> dict[str, int]:
    counts = {"states": 0, "controller": 0, "adversary": 0}
    path = Path(path)
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts["states"] += 1
            player = str(row.get("player", row.get("root_player", ""))).lower()
            if player == "controller":
                counts["controller"] += 1
            elif player == "adversary":
                counts["adversary"] += 1
    return counts


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def append_csv_file(dst: Path, src: Path, *, extra: dict[str, Any] | None = None) -> int:
    dst = Path(dst)
    src = Path(src)
    if not src.exists():
        return 0
    extra = dict(extra or {})
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        src_fields = list(reader.fieldnames or [])
        fields = list(extra.keys()) + [x for x in src_fields if x not in extra]
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_header = not dst.exists()
        count = 0
        with dst.open("a", encoding="utf-8", newline="") as out:
            w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for row in reader:
                row2 = dict(extra)
                row2.update(row)
                w.writerow(row2)
                count += 1
    return count


def _relative_files(root: Path, *, include_manifest: bool = False) -> list[Path]:
    skip = {"SHA256SUMS"}
    if not include_manifest:
        skip.add("shard_manifest.json")
        skip.add("model_manifest.json")
    files: list[Path] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        if path.name in skip or path.name.endswith(".tmp"):
            continue
        files.append(path.relative_to(root))
    return files


def write_sha256sums(root: Path, *, include_manifest: bool = False) -> Path:
    root = Path(root)
    rows = []
    for rel in _relative_files(root, include_manifest=include_manifest):
        digest = sha256_file(root / rel)
        rows.append(f"{digest}  {rel.as_posix()}\n")
    tmp = root / "SHA256SUMS.tmp"
    out = root / "SHA256SUMS"
    tmp.write_text("".join(rows), encoding="utf-8")
    tmp.replace(out)
    return out


def verify_sha256sums(root: Path) -> tuple[bool, list[str]]:
    root = Path(root)
    path = root / "SHA256SUMS"
    errors: list[str] = []
    if not path.exists():
        return False, ["missing SHA256SUMS"]
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, rel = line.split(None, 1)
        except ValueError:
            errors.append(f"line {lineno}: malformed")
            continue
        rel = rel.strip()
        f = root / rel
        if not f.exists():
            errors.append(f"missing {rel}")
            continue
        actual = sha256_file(f)
        if actual != digest:
            errors.append(f"sha256 mismatch {rel}: expected={digest} actual={actual}")
    return not errors, errors


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_shard_manifest(
    *,
    shard_dir: Path,
    shard_id: str,
    worker_id: str,
    model_version: int,
    games_executed: int,
    controller_model_version: int | None = None,
    adversary_model_version: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shard_dir = Path(shard_dir)
    replay_path = shard_dir / "replay_target_runtime.csv"
    counts = replay_counts(replay_path)
    files: dict[str, dict[str, Any]] = {}
    for rel in _relative_files(shard_dir, include_manifest=False):
        f = shard_dir / rel
        files[rel.as_posix()] = {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
    controller_version = int(controller_model_version if controller_model_version is not None else model_version)
    adversary_version = int(adversary_model_version if adversary_model_version is not None else model_version)
    manifest = {
        "schema_version": 2,
        "shard_id": str(shard_id),
        "worker_id": str(worker_id),
        "model_version": int(model_version),
        "controller_model_version": int(controller_version),
        "adversary_model_version": int(adversary_version),
        "games_executed": int(games_executed),
        "states_generated": int(counts["states"]),
        "controller_states": int(counts["controller"]),
        "adversary_states": int(counts["adversary"]),
        "created_at_utc": utc_now(),
        "created_at_local_24h": local_time_24h(),
        "files": files,
    }
    if metadata:
        manifest.update(dict(metadata))
    return manifest


def finalize_shard(
    *,
    active_dir: Path,
    ready_root: Path,
    shard_id: str,
    worker_id: str,
    model_version: int,
    games_executed: int,
    controller_model_version: int | None = None,
    adversary_model_version: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    active_dir = Path(active_dir)
    ready_root = Path(ready_root)
    ready_root.mkdir(parents=True, exist_ok=True)
    shard_dir = ready_root / shard_id
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    active_dir.rename(shard_dir)
    manifest = build_shard_manifest(
        shard_dir=shard_dir,
        shard_id=shard_id,
        worker_id=worker_id,
        model_version=int(model_version),
        controller_model_version=controller_model_version,
        adversary_model_version=adversary_model_version,
        games_executed=int(games_executed),
        metadata=metadata,
    )
    atomic_write_json(shard_dir / "shard_manifest.json", manifest)
    write_sha256sums(shard_dir, include_manifest=True)
    ok, errors = verify_sha256sums(shard_dir)
    if not ok:
        raise RuntimeError(f"local shard checksum failed: {errors}")
    return shard_dir


def run_cmd(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)


def rsync_dir_to(local_dir: Path, host: str, remote_dir: str) -> None:
    local = str(Path(local_dir).resolve()).rstrip("/") + "/"
    remote_dir_clean = remote_dir.rstrip("/")
    run_cmd(["ssh", host, f"mkdir -p {shlex.quote(remote_dir_clean)}"], check=True)
    remote = f"{host}:{remote_dir_clean}/"
    run_cmd(["rsync", "-az", "--partial", "--delay-updates", "--timeout=60", local, remote], check=True)


def rsync_dir_from(host: str, remote_dir: str, local_dir: Path) -> None:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    remote = f"{host}:{remote_dir.rstrip('/')}/"
    local = str(Path(local_dir).resolve()).rstrip("/") + "/"
    run_cmd(["rsync", "-az", "--partial", "--delay-updates", "--timeout=60", remote, local], check=True)


def remote_verify_and_publish(*, host: str, uploading_dir: str, incoming_dir: str, rejected_dir: str) -> None:
    """Verify and atomically publish a transfer, including acknowledgement retries.

    A transport can disappear after the move succeeds but before SSH returns. A
    retry therefore treats an already-valid incoming directory as success and
    never overwrites it with a second result.
    """

    script = "\n".join([
        "set -euo pipefail",
        f"UP={shlex.quote(uploading_dir)}",
        f"IN={shlex.quote(incoming_dir)}",
        f"RJ={shlex.quote(rejected_dir)}",
        'mkdir -p "$(dirname "$IN")" "$(dirname "$RJ")"',
        'verify_dir() { (cd "$1" && sha256sum -c SHA256SUMS); }',
        'if [ -d "$IN" ]; then',
        '  if verify_dir "$IN" >/tmp/agz_sha_verify.$$ 2>&1; then',
        '    rm -rf "$UP"',
        '    rm -f /tmp/agz_sha_verify.$$ || true',
        '    echo already_published="$IN"',
        '    exit 0',
        '  fi',
        '  rm -rf "$RJ"',
        '  mv "$IN" "$RJ"',
        'fi',
        'if [ ! -d "$UP" ]; then echo "missing_uploading=$UP"; exit 2; fi',
        'if ! verify_dir "$UP" >/tmp/agz_sha_verify.$$ 2>&1; then',
        '  mkdir -p "$(dirname "$RJ")"',
        '  rm -rf "$RJ"',
        '  mv "$UP" "$RJ"',
        '  cat /tmp/agz_sha_verify.$$ || true',
        '  rm -f /tmp/agz_sha_verify.$$ || true',
        '  exit 3',
        'fi',
        'rm -f /tmp/agz_sha_verify.$$ || true',
        'mv "$UP" "$IN"',
        'echo published="$IN"',
    ])
    run_cmd(["ssh", host, script], check=True)


@dataclass(frozen=True)
class ShardAck:
    accepted: bool
    ack_path: str


def wait_for_ack(*, host: str, ack_path: str, timeout_sec: int = 0, poll_sec: float = 5.0) -> ShardAck:
    start = time.time()
    # timeout_sec <= 0 means wait indefinitely. This is the intended durable
    # transfer backpressure: workers must not publish unlimited shards to XL
    # when the coordinator is down or lagging.
    timeout = int(timeout_sec)
    while True:
        cp = run_cmd(["ssh", host, f"test -f {ack_path!r} && echo yes || echo no"], check=False)
        accepted = "yes" in (cp.stdout or "")
        if accepted:
            return ShardAck(True, ack_path)
        if timeout > 0 and time.time() - start >= timeout:
            return ShardAck(False, ack_path)
        time.sleep(float(poll_sec))
