"""Verify prepared trace/profile checksums before a benchmark or container run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .canonicalization import TRACE_SCHEMA_VERSION
from .trace_contract import CANONICAL_COLUMNS, file_sha256


EXPECTED_DECODE_PROFILE_SHA256 = (
    "b14044faaa5f9f5fea1b159f5bd031ce8538b30b59eb8acd32cd9625093c4d2a"
)


def _require_hash(path: str | Path, expected: str, *, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def verify(
    *,
    manifest_path: str | Path,
    source_path: str | Path,
    canonical_path: str | Path,
    prefill_profile_path: str | Path,
    decode_profile_path: str | Path,
) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema {manifest.get('schema_version')!r}")
    _require_hash(source_path, manifest["source"]["sha256"], label="raw trace")
    _require_hash(canonical_path, manifest["canonical"]["sha256"], label="canonical trace")
    _require_hash(
        prefill_profile_path,
        manifest["profiles"]["prefill_sha256"],
        label="prefill profile",
    )
    _require_hash(
        decode_profile_path,
        EXPECTED_DECODE_PROFILE_SHA256,
        label="decode profile",
    )

    with Path(canonical_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANONICAL_COLUMNS:
            raise ValueError("canonical trace columns do not match gv3_vllm_trace_v1")
        rows = list(reader)
    if len(rows) != int(manifest["canonical"]["row_count"]):
        raise ValueError("canonical trace row count differs from manifest")
    if any(row["schema_version"] != TRACE_SCHEMA_VERSION for row in rows):
        raise ValueError("canonical trace contains a mismatched schema version")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify vLLM trace/profile artifacts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--prefill-profile", required=True)
    parser.add_argument("--decode-profile", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify(
        manifest_path=args.manifest,
        source_path=args.source,
        canonical_path=args.canonical,
        prefill_profile_path=args.prefill_profile,
        decode_profile_path=args.decode_profile,
    )
    print("trace and profile artifacts verified")


if __name__ == "__main__":
    main()
