"""Generate exact-token, GV3-legal traces for Task 1.1.2."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

from .canonicalization import (
    CanonicalizationConfig,
    PREFILL_ROUNDING_CEILING,
    PREFILL_ROUNDING_NEAREST,
    PrefillProfile,
    canonicalize_prefill_tokens,
)
from .prepare_trace import prepare
from .prompt_materialization import verify_prompt_artifacts, write_prompt_artifacts
from .test_traces_on_GPU_with_AlphaGOZERO_models_config import (
    AlphaGoZeroGPUTraceConfig,
    GV3_IN_DISTRIBUTION_PREFILL_TOKENS,
    TraceType,
    load_config,
)
from .trace_contract import RAW_COLUMNS, file_sha256


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_DIR = PACKAGE_DIR / "tokenizer/llama3_8b"


@dataclass(frozen=True)
class PreparedTask112Trace:
    root: Path
    template_trace: Path
    raw_trace: Path
    canonical_trace: Path
    manifest: Path
    prompt_catalog: Path
    config_snapshot: Path
    request_count: int


def _canonicalization_config(config: AlphaGoZeroGPUTraceConfig) -> CanonicalizationConfig:
    rounding = (
        PREFILL_ROUNDING_CEILING
        if TraceType.parse(config.trace_type) is TraceType.OUT_DISTRIBUTION
        else PREFILL_ROUNDING_NEAREST
    )
    return CanonicalizationConfig(
        prefill_grid_tokens=config.prefill_grid_tokens,
        min_prefill_tokens=config.min_prefill_tokens,
        max_prefill_tokens=config.max_prefill_tokens,
        launch_window_s=config.launch_window_s,
        launch_window_request_cap=config.launch_window_request_cap,
        launch_window_prefill_cap=config.launch_window_prefill_cap,
        prefill_rounding=rounding,
    )


def _group_prefill_tokens(
    config: AlphaGoZeroGPUTraceConfig,
    *,
    group_index: int,
    rng: random.Random,
) -> tuple[int, ...]:
    trace_type = TraceType.parse(config.trace_type)
    if trace_type is TraceType.IN_DISTRIBUTION:
        support = GV3_IN_DISTRIBUTION_PREFILL_TOKENS
        canonical = support[(group_index + 3) % len(support)]
        count = min(
            config.launch_window_request_cap,
            max(1, config.launch_window_prefill_cap // canonical),
        )
        return (canonical,) * count

    grid = tuple(
        range(
            config.min_prefill_tokens,
            config.max_prefill_tokens + 1,
            config.prefill_grid_tokens,
        )
    )
    canonical = grid[(group_index * 7 + 7) % len(grid)]
    count = min(
        config.launch_window_request_cap,
        max(1, config.launch_window_prefill_cap // canonical),
    )
    lower = max(config.min_prefill_tokens, canonical - config.prefill_grid_tokens + 1)
    actual: list[int] = []
    for _ in range(count):
        value = rng.randint(lower, canonical)
        if value == canonical and lower < canonical:
            value -= 1
        actual.append(value)
    return tuple(actual)


def build_template_rows(
    config: AlphaGoZeroGPUTraceConfig,
    *,
    profile: PrefillProfile,
) -> list[dict[str, object]]:
    """Build deterministic launch groups without violating GV3's one-second caps."""

    config.validate()
    canonical_config = _canonicalization_config(config)
    rng = random.Random(config.trace_seed)
    rows: list[dict[str, object]] = []
    group_index = 0
    request_index = 0
    while True:
        arrival_s = round(group_index * config.arrival_group_interval_s, 10)
        if arrival_s >= config.static_trace_length_s:
            break
        actual_prefills = _group_prefill_tokens(
            config,
            group_index=group_index,
            rng=rng,
        )
        for actual_prefill in actual_prefills:
            canonical_prefill = canonicalize_prefill_tokens(
                actual_prefill,
                canonical_config,
            )
            prefill_slo_s = (
                profile.execution_time_s(canonical_prefill)
                * canonical_config.prefill_slowdown
            )
            rows.append(
                {
                    "request_id": f"task112-{TraceType.parse(config.trace_type).value}-{request_index:06d}",
                    "arrived_at_s": arrival_s,
                    "num_prefill_tokens": actual_prefill,
                    "num_decode_tokens": config.decode_tokens_per_request,
                    "prefill_slo_s": prefill_slo_s,
                    "decode_slo_s": canonical_config.canonical_decode_slo_s,
                    "prompt_mode": "synthetic_token_ids",
                    "prompt_ref": "",
                    "seed": config.trace_seed + request_index,
                    "ignore_eos": "true",
                }
            )
            request_index += 1
        group_index += 1
    if not rows:
        raise ValueError("trace duration produced no requests")
    return rows


def _write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_prompt_hash(catalog_path: Path) -> str:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for row in catalog.get("prompts", ()):
        digest.update(str(row["request_id"]).encode("utf-8"))
        digest.update(str(row["text_sha256"]).encode("ascii"))
        digest.update(str(row["token_ids_sha256"]).encode("ascii"))
    return digest.hexdigest()


def prepare_task_1_1_2_trace(
    config: AlphaGoZeroGPUTraceConfig,
    *,
    output_root: str | Path,
    tokenizer_dir: str | Path = DEFAULT_TOKENIZER_DIR,
) -> PreparedTask112Trace:
    """
        reads the config + prefill_profile.csv (for SLOS for created requests) and creates :
        template_raw_trace.csv
        raw_trace.csv
        canonical_trace.csv
        prompts/
        trace_manifest.json
        experiment_config.json
    """

    config.validate()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    template = root / "template_raw_trace.csv"
    raw = root / "raw_trace.csv"
    canonical = root / "canonical_trace.csv"
    manifest_path = root / "trace_manifest.json"
    prompt_root = root / "prompts"
    config_snapshot = root / "experiment_config.json"

    profile = PrefillProfile.load(config.prefill_profile_path)
    rows = build_template_rows(config, profile=profile)
    _write_raw(template, rows)
    prompt_count, prompt_catalog = write_prompt_artifacts(
        raw_trace_path=template,
        output_trace_path=raw,
        prompt_root=prompt_root,
        tokenizer_dir=tokenizer_dir,
    )
    verified = verify_prompt_artifacts(
        raw_trace_path=raw,
        tokenizer_dir=tokenizer_dir,
    )
    if prompt_count != verified or verified != len(rows):
        raise RuntimeError(
            f"prompt count mismatch: generated={prompt_count}, verified={verified}, rows={len(rows)}"
        )

    result = prepare(
        input_path=raw,
        output_path=canonical,
        manifest_path=manifest_path,
        prefill_profile_path=config.prefill_profile_path,
        config=_canonicalization_config(config),
        derive_missing_slos=False,
        enforce_gv3_window=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrival_counts: dict[float, int] = {}
    for row in result.rows:
        arrival_counts[row.arrived_at_s] = arrival_counts.get(row.arrived_at_s, 0) + 1
    manifest["task_1_1_2"] = {
        "trace_type": TraceType.parse(config.trace_type).value,
        "static_trace_length_s": config.static_trace_length_s,
        "request_count": len(result.rows),
        "simultaneous_arrival_groups": sum(count > 1 for count in arrival_counts.values()),
        "largest_arrival_group": max(arrival_counts.values()),
        "physical_prefill_tokens_drive": "vLLM execution and exact prompt token IDs",
        "canonical_prefill_tokens_drive": "persistent GV3 state, DNN features, and native MCTS",
        "out_distribution_rounding": "ceiling_128",
        "canonical_clock": "sum of completed transformer-block CUDA durations only",
        "controller_wall_time_in_clock": False,
        "calibration": None,
    }
    manifest["prompt_artifacts"] = {
        "catalog_path": str(prompt_catalog),
        "catalog_sha256": file_sha256(prompt_catalog),
        "aggregate_prompt_sha256": _aggregate_prompt_hash(prompt_catalog),
        "verified_prompt_count": verified,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_snapshot.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparedTask112Trace(
        root=root,
        template_trace=template,
        raw_trace=raw,
        canonical_trace=canonical,
        manifest=manifest_path,
        prompt_catalog=prompt_catalog,
        config_snapshot=config_snapshot,
        request_count=len(result.rows),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a Task 1.1.2 exact-token trace")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    args = parser.parse_args(argv)
    artifact = prepare_task_1_1_2_trace(
        load_config(),
        output_root=args.output_root,
        tokenizer_dir=args.tokenizer_dir,
    )
    print(
        json.dumps(
            {
                "canonical_trace": str(artifact.canonical_trace),
                "manifest": str(artifact.manifest),
                "request_count": artifact.request_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
