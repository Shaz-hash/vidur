"""Attach tokenizer-verified English prompt artifacts to a prepared trace manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .prompt_materialization import (
    LLAMA3_BOS_TOKEN_ID,
    TOKENIZER_JSON_SHA256,
    TOKENIZER_REPOSITORY,
    TOKENIZER_REVISION,
    file_sha256,
    verify_prompt_artifacts,
)


def finalize(
    *,
    manifest_path: str | Path,
    raw_trace_path: str | Path,
    prompt_catalog_path: str | Path,
    tokenizer_dir: str | Path,
) -> int:
    manifest_file = Path(manifest_path).expanduser().resolve()
    raw_trace = Path(raw_trace_path).expanduser().resolve()
    catalog_path = Path(prompt_catalog_path).expanduser().resolve()
    verified = verify_prompt_artifacts(
        raw_trace_path=raw_trace,
        tokenizer_dir=tokenizer_dir,
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if file_sha256(raw_trace) != manifest["source"]["sha256"]:
        raise ValueError("raw English trace differs from the prepared trace manifest")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if int(catalog.get("request_count", -1)) != verified:
        raise ValueError("prompt catalog count differs from verified trace")

    aggregate = hashlib.sha256()
    for row in catalog["prompts"]:
        aggregate.update(str(row["request_id"]).encode("utf-8"))
        aggregate.update(str(row["text_sha256"]).encode("ascii"))
        aggregate.update(str(row["token_ids_sha256"]).encode("ascii"))
    manifest["prompt_artifacts"] = {
        "mode": "token_ids_file_with_english_text",
        "request_count": verified,
        "catalog_path": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "aggregate_prompt_sha256": aggregate.hexdigest(),
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "add_special_tokens": True,
        "bos_token_id": LLAMA3_BOS_TOKEN_ID,
        "token_count_semantics": "BOS plus English body tokens",
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize an English trace manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-trace", required=True)
    parser.add_argument("--prompt-catalog", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = finalize(
        manifest_path=args.manifest,
        raw_trace_path=args.raw_trace,
        prompt_catalog_path=args.prompt_catalog,
        tokenizer_dir=args.tokenizer_dir,
    )
    print(f"verified and pinned {count} English prompt artifacts")


if __name__ == "__main__":
    main()
