"""Single pre-container gate for a tokenizer-materialized real-vLLM trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prompt_materialization import file_sha256, verify_prompt_artifacts
from .verify_artifacts import verify


def verify_ready_trace(
    *,
    manifest_path: str | Path,
    source_path: str | Path,
    canonical_path: str | Path,
    prompt_catalog_path: str | Path,
    tokenizer_dir: str | Path,
    prefill_profile_path: str | Path,
    decode_profile_path: str | Path,
) -> int:
    verify(
        manifest_path=manifest_path,
        source_path=source_path,
        canonical_path=canonical_path,
        prefill_profile_path=prefill_profile_path,
        decode_profile_path=decode_profile_path,
    )
    verified = verify_prompt_artifacts(
        raw_trace_path=source_path,
        tokenizer_dir=tokenizer_dir,
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    prompt_manifest = manifest.get("prompt_artifacts")
    if not isinstance(prompt_manifest, dict):
        raise ValueError("trace manifest does not pin prompt artifacts")
    if int(prompt_manifest.get("request_count", -1)) != verified:
        raise ValueError("verified prompt count differs from trace manifest")
    if file_sha256(prompt_catalog_path) != prompt_manifest.get("catalog_sha256"):
        raise ValueError("prompt catalog SHA256 differs from trace manifest")
    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a ready English vLLM trace.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--prompt-catalog", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prefill-profile", required=True)
    parser.add_argument("--decode-profile", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = verify_ready_trace(
        manifest_path=args.manifest,
        source_path=args.source,
        canonical_path=args.canonical,
        prompt_catalog_path=args.prompt_catalog,
        tokenizer_dir=args.tokenizer_dir,
        prefill_profile_path=args.prefill_profile,
        decode_profile_path=args.decode_profile,
    )
    print(f"ready trace verified: {count} tokenizer-exact English prompts")


if __name__ == "__main__":
    main()
