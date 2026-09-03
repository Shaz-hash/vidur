"""Generate and verify exact-length English prompts with the pinned Llama-3 tokenizer."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

from .canonicalization import CanonicalizationError
from .trace_contract import RAW_COLUMNS


PROMPT_SCHEMA_VERSION = "gv3_llama3_english_prompt_v1"
TOKENIZER_REPOSITORY = "NousResearch/Meta-Llama-3-8B"
TOKENIZER_REVISION = "315b20096dc791d381d514deb5f8bd9c8d6d3061"
TOKENIZER_JSON_SHA256 = "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"
LLAMA3_BOS_TOKEN_ID = 128000

TOPICS = (
    "distributed inference scheduling",
    "urban water conservation",
    "reliable software deployment",
    "renewable energy planning",
    "public transport coordination",
    "scientific experiment design",
    "digital library preservation",
    "emergency logistics management",
)

PARAGRAPHS = (
    "A careful analysis begins by separating observations from assumptions. The report should identify the available resources, the requests waiting for service, and the deadlines that make one task more urgent than another. It should explain each conclusion in plain language and preserve enough detail for another engineer to reproduce the result.",
    "The system changes over time, so a useful plan must account for both immediate work and delayed consequences. Serving a short task may look efficient at first, yet repeatedly postponing larger tasks can create a backlog. A balanced decision considers current latency, future queue pressure, capacity limits, and uncertainty in new arrivals.",
    "Reliable evaluation requires consistent measurements. Record the starting state, selected action, execution duration, completed work, remaining work, and any service objective violation. Compare alternatives over the same interval, use the same inputs, and keep raw measurements separate from normalized features used by a predictive model.",
    "Operational recommendations should remain general rather than depending on one trace. Prefer rules that respond to workload structure, such as urgency, remaining work, and available capacity. Avoid conclusions based only on a request identifier or its position in a file, because those details do not describe the underlying scheduling problem.",
    "When a model guides search, its prediction is only one part of the decision. Immediate simulator rewards, discounted future outcomes, policy priors, and exploration all influence the final ranking. Diagnostics should therefore report each component independently and verify that Python and native implementations receive identical state features.",
    "A production test must distinguish simulated time from wall clock time. The service executes real batches on an accelerator, while the planning layer may run on a CPU. Report the observed result and any adjusted result separately, subtracting only planning time that actually blocks useful execution rather than time hidden behind concurrent work.",
    "Data quality matters as much as model capacity. Each request needs a stable arrival time, an exact prompt length, a bounded output length, and explicit service objectives. Tokenization must be deterministic, and every generated artifact should carry a checksum so a later run cannot silently use different prompts or model inputs.",
    "The final explanation should state limitations. A simulator can approximate hardware behavior but cannot guarantee identical timing, especially when memory pressure, kernel selection, or preemption changes. Calibration quantifies this gap; it should not silently rewrite the frozen assumptions used by a previously trained controller.",
)

FILLER_WORDS = (
    "system",
    "service",
    "request",
    "queue",
    "model",
    "policy",
    "value",
    "search",
    "timing",
    "cost",
    "state",
    "action",
    "result",
    "capacity",
    "deadline",
    "process",
    "analysis",
    "evidence",
    "workload",
    "measurement",
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative_ref(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path, anchor)).as_posix()


def load_pinned_tokenizer(tokenizer_dir: str | Path) -> tuple[Tokenizer, dict[str, Any]]:
    root = Path(tokenizer_dir).expanduser().resolve()
    manifest_path = root / "tokenizer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != TOKENIZER_REPOSITORY:
        raise CanonicalizationError("unexpected tokenizer repository")
    if manifest.get("revision") != TOKENIZER_REVISION:
        raise CanonicalizationError("unexpected tokenizer revision")
    for name, expected in manifest.get("files", {}).items():
        actual = file_sha256(root / name)
        if actual != expected:
            raise CanonicalizationError(
                f"tokenizer file {name} SHA256 mismatch: expected {expected}, got {actual}"
            )
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=False) != 128000:
        raise CanonicalizationError("unexpected Llama-3 base vocabulary size")
    probe = tokenizer.encode("English prompt.", add_special_tokens=True)
    if not probe.ids or probe.ids[0] != LLAMA3_BOS_TOKEN_ID:
        raise CanonicalizationError("tokenizer does not prepend the expected Llama-3 BOS")
    return tokenizer, manifest


def _token_count(tokenizer: Tokenizer, text: str, *, add_special_tokens: bool) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)


def generate_english_prompt(
    tokenizer: Tokenizer,
    *,
    target_tokens: int,
    request_index: int,
) -> tuple[str, list[int]]:
    """Construct readable text whose BOS-inclusive tokenization is exact."""

    if target_tokens < 16:
        raise CanonicalizationError("English prompt generation requires at least 16 tokens")
    body_target = target_tokens - 1
    topic = TOPICS[request_index % len(TOPICS)]
    text = (
        f"Prepare a precise technical response about {topic}. "
        "Use the following background to reason carefully and state a practical conclusion."
    )
    if _token_count(tokenizer, text, add_special_tokens=False) >= body_target:
        raise CanonicalizationError(f"target {target_tokens} is too short for prompt header")

    paragraph_index = request_index % len(PARAGRAPHS)
    while True:
        paragraph = PARAGRAPHS[paragraph_index % len(PARAGRAPHS)]
        candidate = f"{text}\n\n{paragraph}"
        # Keep room for a short natural closing and exact one-token fillers.
        if _token_count(tokenizer, candidate, add_special_tokens=False) > body_target - 8:
            break
        text = candidate
        paragraph_index += 1

    closing_candidates = (
        "\n\nKey considerations include",
        "\n\nThe review considers",
        "\n\nImportant factors include",
        "\n\nThe final check covers",
    )
    for closing in closing_candidates:
        candidate = text + closing
        if _token_count(tokenizer, candidate, add_special_tokens=False) <= body_target - 2:
            text = candidate
            break

    filler_index = request_index % len(FILLER_WORDS)
    while _token_count(tokenizer, text, add_special_tokens=False) < body_target - 1:
        before = _token_count(tokenizer, text, add_special_tokens=False)
        selected: str | None = None
        for offset in range(len(FILLER_WORDS)):
            word = FILLER_WORDS[(filler_index + offset) % len(FILLER_WORDS)]
            candidate = f"{text} {word}"
            delta = _token_count(tokenizer, candidate, add_special_tokens=False) - before
            if delta == 1:
                selected = candidate
                filler_index = (filler_index + offset + 1) % len(FILLER_WORDS)
                break
        if selected is None:
            raise CanonicalizationError("could not find a stable one-token English filler")
        text = selected

    if _token_count(tokenizer, text + ".", add_special_tokens=False) == body_target:
        text += "."
    elif _token_count(tokenizer, text, add_special_tokens=False) != body_target:
        raise CanonicalizationError(
            f"failed to construct exact {target_tokens}-token prompt for request {request_index}"
        )

    encoding = tokenizer.encode(text, add_special_tokens=True)
    if len(encoding.ids) != target_tokens or encoding.ids[0] != LLAMA3_BOS_TOKEN_ID:
        raise CanonicalizationError(
            f"tokenizer produced {len(encoding.ids)} tokens, expected {target_tokens}"
        )
    if tokenizer.encode(text, add_special_tokens=True).ids != encoding.ids:
        raise CanonicalizationError("prompt tokenization is not deterministic")
    return text, list(encoding.ids)


def write_prompt_artifacts(
    *,
    raw_trace_path: str | Path,
    output_trace_path: str | Path,
    prompt_root: str | Path,
    tokenizer_dir: str | Path,
) -> tuple[int, Path]:
    source = Path(raw_trace_path).expanduser().resolve()
    output = Path(output_trace_path).expanduser().resolve()
    root = Path(prompt_root).expanduser().resolve()
    text_dir = root / "text"
    token_dir = root / "token_ids"
    text_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_manifest = load_pinned_tokenizer(tokenizer_dir)

    output_rows: list[dict[str, str]] = []
    catalog_rows: list[dict[str, Any]] = []
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise CanonicalizationError("input trace does not match the strict raw contract")
        for index, row in enumerate(reader):
            request_id = row["request_id"]
            declared_tokens = int(row["num_prefill_tokens"])
            text, token_ids = generate_english_prompt(
                tokenizer,
                target_tokens=declared_tokens,
                request_index=index,
            )
            text_path = text_dir / f"{request_id}.txt"
            token_path = token_dir / f"{request_id}.json"
            text_path.write_text(text, encoding="utf-8")
            token_payload = {
                "schema_version": PROMPT_SCHEMA_VERSION,
                "request_id": request_id,
                "text_ref": _relative_ref(text_path, output.parent),
                "text_sha256": file_sha256(text_path),
                "tokenizer_repository": TOKENIZER_REPOSITORY,
                "tokenizer_revision": TOKENIZER_REVISION,
                "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
                "add_special_tokens": True,
                "bos_token_id": LLAMA3_BOS_TOKEN_ID,
                "token_count": len(token_ids),
                "token_ids": token_ids,
            }
            token_path.write_text(
                json.dumps(token_payload, separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            updated = dict(row)
            updated["prompt_mode"] = "token_ids_file"
            updated["prompt_ref"] = _relative_ref(token_path, output.parent)
            output_rows.append(updated)
            catalog_rows.append(
                {
                    "request_id": request_id,
                    "declared_prefill_tokens": declared_tokens,
                    "verified_token_count": len(token_ids),
                    "text_ref": _relative_ref(text_path, output.parent),
                    "text_sha256": token_payload["text_sha256"],
                    "token_ids_ref": _relative_ref(token_path, output.parent),
                    "token_ids_sha256": file_sha256(token_path),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    catalog_path = root / "prompt_catalog.json"
    catalog = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "trace_path": str(output),
        "tokenizer": tokenizer_manifest,
        "request_count": len(catalog_rows),
        "prompts": catalog_rows,
    }
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(output_rows), catalog_path


def verify_prompt_artifacts(
    *,
    raw_trace_path: str | Path,
    tokenizer_dir: str | Path,
) -> int:
    trace = Path(raw_trace_path).expanduser().resolve()
    tokenizer, _ = load_pinned_tokenizer(tokenizer_dir)
    verified = 0
    with trace.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if row.get("prompt_mode") != "token_ids_file":
                raise CanonicalizationError(f"row {row_number}: prompt is not materialized")
            token_path = (trace.parent / row["prompt_ref"]).resolve()
            payload = json.loads(token_path.read_text(encoding="utf-8"))
            text_path = (trace.parent / payload["text_ref"]).resolve()
            text = text_path.read_text(encoding="utf-8")
            ids = tokenizer.encode(text, add_special_tokens=True).ids
            declared = int(row["num_prefill_tokens"])
            if ids != payload["token_ids"]:
                raise CanonicalizationError(f"row {row_number}: stored token IDs differ from text")
            if len(ids) != declared or payload["token_count"] != declared:
                raise CanonicalizationError(
                    f"row {row_number}: prompt has {len(ids)} tokens, expected {declared}"
                )
            if file_sha256(text_path) != payload["text_sha256"]:
                raise CanonicalizationError(f"row {row_number}: prompt text checksum mismatch")
            verified += 1
    return verified
