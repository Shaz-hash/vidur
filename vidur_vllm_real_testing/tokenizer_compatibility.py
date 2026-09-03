"""Fail-closed compatibility gate for the prepared prompts and model tokenizer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

from .prompt_materialization import file_sha256, load_pinned_tokenizer


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _resolve_tokenizer(
    model_tokenizer: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    local_path = Path(model_tokenizer).expanduser()
    if local_path.exists():
        return local_path.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_tokenizer,
            revision=revision,
            allow_patterns=list(TOKENIZER_FILES),
            local_files_only=local_files_only,
        )
    ).resolve()


def _load_vllm_tokenizer(tokenizer_dir: Path) -> Any:
    try:
        from vllm.tokenizers import get_tokenizer
    except ImportError:
        from vllm.transformers_utils.tokenizer import get_tokenizer

    return get_tokenizer(
        str(tokenizer_dir),
        tokenizer_mode="auto",
        trust_remote_code=False,
    )


def _iter_prompt_payloads(trace_paths: Iterable[str | Path]) -> Iterable[tuple[str, str, list[int]]]:
    for trace_value in trace_paths:
        trace_path = Path(trace_value).expanduser().resolve()
        with trace_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                token_path = (trace_path.parent / row["prompt_ref"]).resolve()
                payload = json.loads(token_path.read_text(encoding="utf-8"))
                text_path = (trace_path.parent / payload["text_ref"]).resolve()
                yield (
                    f"{trace_path.name}:{row_number}:{row['request_id']}",
                    text_path.read_text(encoding="utf-8"),
                    [int(value) for value in payload["token_ids"]],
                )


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return [int(value) for value in encoded]


def verify_model_tokenizer(
    *,
    model_tokenizer: str,
    pinned_tokenizer_dir: str | Path,
    trace_paths: Iterable[str | Path],
    revision: str | None = None,
    local_files_only: bool = False,
    require_vllm: bool = False,
) -> dict[str, Any]:
    """Verify file identity and prompt-level parity with the model tokenizer."""

    pinned_root = Path(pinned_tokenizer_dir).expanduser().resolve()
    pinned_tokenizer, manifest = load_pinned_tokenizer(pinned_root)
    actual_root = _resolve_tokenizer(
        model_tokenizer,
        revision=revision,
        local_files_only=local_files_only,
    )

    actual_hashes: dict[str, str] = {}
    for file_name in TOKENIZER_FILES:
        path = actual_root / file_name
        if not path.is_file():
            raise ValueError(f"model tokenizer is missing {file_name}: {actual_root}")
        actual_hashes[file_name] = file_sha256(path)

    expected_tokenizer_hash = manifest["files"]["tokenizer.json"]
    if actual_hashes["tokenizer.json"] != expected_tokenizer_hash:
        raise ValueError(
            "model tokenizer.json SHA256 mismatch: "
            f"expected {expected_tokenizer_hash}, got {actual_hashes['tokenizer.json']}"
        )

    model_backend = Tokenizer.from_file(str(actual_root / "tokenizer.json"))
    if model_backend.get_vocab_size(with_added_tokens=False) != int(
        manifest["vocab_size_without_added_tokens"]
    ):
        raise ValueError("model tokenizer base vocabulary size differs from the pinned tokenizer")
    if model_backend.get_vocab_size(with_added_tokens=True) != int(
        manifest["vocab_size_with_added_tokens"]
    ):
        raise ValueError("model tokenizer added vocabulary size differs from the pinned tokenizer")

    vllm_tokenizer: Any | None = None
    vllm_version: str | None = None
    try:
        import vllm

        vllm_version = str(vllm.__version__)
        vllm_tokenizer = _load_vllm_tokenizer(actual_root)
    except (ImportError, ModuleNotFoundError):
        if require_vllm:
            raise RuntimeError("vLLM is required but unavailable in this environment") from None

    checked = 0
    for label, text, stored_ids in _iter_prompt_payloads(trace_paths):
        pinned_ids = _encode(pinned_tokenizer, text)
        model_ids = _encode(model_backend, text)
        if pinned_ids != stored_ids:
            raise ValueError(f"{label}: bundled tokenizer differs from stored token IDs")
        if model_ids != stored_ids:
            raise ValueError(f"{label}: model tokenizer differs from stored token IDs")
        if vllm_tokenizer is not None and _encode(vllm_tokenizer, text) != stored_ids:
            raise ValueError(f"{label}: vLLM tokenizer differs from stored token IDs")
        checked += 1

    if checked == 0:
        raise ValueError("no prompts were checked")

    return {
        "status": "compatible",
        "model_tokenizer": model_tokenizer,
        "resolved_tokenizer_dir": str(actual_root),
        "revision": revision,
        "tokenizer_hashes": actual_hashes,
        "prompt_count": checked,
        "vllm_checked": vllm_tokenizer is not None,
        "vllm_version": vllm_version,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify model/vLLM tokenizer compatibility.")
    parser.add_argument("--model-tokenizer", required=True)
    parser.add_argument("--pinned-tokenizer-dir", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_model_tokenizer(
        model_tokenizer=args.model_tokenizer,
        pinned_tokenizer_dir=args.pinned_tokenizer_dir,
        trace_paths=args.trace,
        revision=args.revision,
        local_files_only=args.local_files_only,
        require_vllm=args.require_vllm,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
