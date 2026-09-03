"""CSV contracts and validation for reproducible real-vLLM request traces."""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .canonicalization import (
    CanonicalizationConfig,
    CanonicalizationError,
    PrefillProfile,
    TRACE_SCHEMA_VERSION,
    canonicalize_decode_slo,
    canonicalize_decode_tokens,
    canonicalize_prefill_slo,
    canonicalize_prefill_tokens,
    validate_actual_prefill_slo,
)


RAW_COLUMNS = (
    "request_id",
    "arrived_at_s",
    "num_prefill_tokens",
    "num_decode_tokens",
    "prefill_slo_s",
    "decode_slo_s",
    "prompt_mode",
    "prompt_ref",
    "seed",
    "ignore_eos",
)

CANONICAL_COLUMNS = (
    "schema_version",
    "request_id",
    "source_row",
    "arrived_at_s",
    "actual_prefill_tokens",
    "canonical_prefill_tokens",
    "prefill_rounding_delta_tokens",
    "actual_decode_tokens",
    "canonical_decode_tokens",
    "actual_prefill_slo_s",
    "canonical_prefill_slo_s",
    "prefill_profile_time_s",
    "actual_decode_slo_s",
    "canonical_decode_slo_s",
    "prompt_mode",
    "prompt_ref",
    "seed",
    "ignore_eos",
)

PROMPT_MODES = frozenset({"synthetic_token_ids", "text_file", "token_ids_file"})


@dataclass(frozen=True)
class RawTraceRequest:
    request_id: str
    arrived_at_s: float
    num_prefill_tokens: int
    num_decode_tokens: int
    prefill_slo_s: float
    decode_slo_s: float
    prompt_mode: str
    prompt_ref: str
    seed: int
    ignore_eos: bool
    source_row: int


@dataclass(frozen=True)
class CanonicalTraceRequest:
    schema_version: str
    request_id: str
    source_row: int
    arrived_at_s: float
    actual_prefill_tokens: int
    canonical_prefill_tokens: int
    prefill_rounding_delta_tokens: int
    actual_decode_tokens: int
    canonical_decode_tokens: int
    actual_prefill_slo_s: float
    canonical_prefill_slo_s: float
    prefill_profile_time_s: float
    actual_decode_slo_s: float
    canonical_decode_slo_s: float
    prompt_mode: str
    prompt_ref: str
    seed: int
    ignore_eos: bool

    def csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["ignore_eos"] = "true" if self.ignore_eos else "false"
        return row


@dataclass(frozen=True)
class TracePreparationResult:
    rows: tuple[CanonicalTraceRequest, ...]
    derived_prefill_slos: int
    derived_decode_slos: int


def _finite_float(raw: object, *, field: str, row: int, minimum: float | None = None) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(f"row {row}: {field} is not numeric: {raw!r}") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise CanonicalizationError(f"row {row}: invalid {field}: {raw!r}")
    return value


def _integer(raw: object, *, field: str, row: int, minimum: int | None = None) -> int:
    try:
        decimal = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError) as exc:
        raise CanonicalizationError(f"row {row}: {field} is not integral: {raw!r}") from exc
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        raise CanonicalizationError(f"row {row}: {field} is not integral: {raw!r}")
    value = int(decimal)
    if minimum is not None and value < minimum:
        raise CanonicalizationError(f"row {row}: invalid {field}: {value}")
    return value


def _boolean(raw: object, *, field: str, row: int) -> bool:
    normalized = str(raw).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise CanonicalizationError(f"row {row}: {field} must be true/false, got {raw!r}")


def _optional_positive_float(
    raw: object,
    *,
    field: str,
    row: int,
) -> float | None:
    if raw is None or not str(raw).strip():
        return None
    value = _finite_float(raw, field=field, row=row)
    if value <= 0.0:
        raise CanonicalizationError(f"row {row}: {field} must be > 0")
    return value


def load_raw_trace(
    path: str | Path,
    *,
    profile: PrefillProfile,
    config: CanonicalizationConfig = CanonicalizationConfig(),
    derive_missing_slos: bool = False,
) -> tuple[list[RawTraceRequest], int, int]:
    trace_path = Path(path).expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)

    rows: list[RawTraceRequest] = []
    ids: set[str] = set()
    derived_prefill = 0
    derived_decode = 0
    with trace_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"arrived_at_s", "num_prefill_tokens", "num_decode_tokens"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise CanonicalizationError(f"trace is missing required columns: {missing}")

        for source_row, raw in enumerate(reader, start=2):
            request_id = str(raw.get("request_id", "")).strip() or f"req-{source_row - 2:06d}"
            if request_id in ids:
                raise CanonicalizationError(f"row {source_row}: duplicate request_id {request_id!r}")
            ids.add(request_id)

            arrived_at = _finite_float(
                raw.get("arrived_at_s"), field="arrived_at_s", row=source_row, minimum=0.0
            )
            prefill = _integer(
                raw.get("num_prefill_tokens"), field="num_prefill_tokens", row=source_row, minimum=1
            )
            decode = _integer(
                raw.get("num_decode_tokens"), field="num_decode_tokens", row=source_row, minimum=1
            )
            canonical_prefill = canonicalize_prefill_tokens(prefill, config)
            canonicalize_decode_tokens(decode, config)
            profile_time, canonical_prefill_slo = canonicalize_prefill_slo(
                canonical_prefill, profile, config
            )

            prefill_slo = _optional_positive_float(
                raw.get("prefill_slo_s"), field="prefill_slo_s", row=source_row
            )
            if prefill_slo is None:
                if not derive_missing_slos:
                    raise CanonicalizationError(
                        f"row {source_row}: prefill_slo_s is required; "
                        "use --derive-missing-slos only for synthetic traces"
                    )
                prefill_slo = canonical_prefill_slo
                derived_prefill += 1
            decode_slo = _optional_positive_float(
                raw.get("decode_slo_s"), field="decode_slo_s", row=source_row
            )
            if decode_slo is None:
                if not derive_missing_slos:
                    raise CanonicalizationError(
                        f"row {source_row}: decode_slo_s is required; "
                        "use --derive-missing-slos only for synthetic traces"
                    )
                decode_slo = config.canonical_decode_slo_s
                derived_decode += 1

            prompt_mode = str(raw.get("prompt_mode", "synthetic_token_ids")).strip()
            if prompt_mode not in PROMPT_MODES:
                raise CanonicalizationError(
                    f"row {source_row}: unsupported prompt_mode {prompt_mode!r}"
                )
            prompt_ref = str(raw.get("prompt_ref", "")).strip()
            if prompt_mode != "synthetic_token_ids" and not prompt_ref:
                raise CanonicalizationError(
                    f"row {source_row}: prompt_ref is required for {prompt_mode}"
                )
            seed = _integer(raw.get("seed", source_row - 2), field="seed", row=source_row, minimum=0)
            ignore_eos = _boolean(
                raw.get("ignore_eos", "true"), field="ignore_eos", row=source_row
            )
            rows.append(
                RawTraceRequest(
                    request_id=request_id,
                    arrived_at_s=arrived_at,
                    num_prefill_tokens=prefill,
                    num_decode_tokens=decode,
                    prefill_slo_s=validate_actual_prefill_slo(prefill_slo),
                    decode_slo_s=decode_slo,
                    prompt_mode=prompt_mode,
                    prompt_ref=prompt_ref,
                    seed=seed,
                    ignore_eos=ignore_eos,
                    source_row=source_row,
                )
            )

    rows.sort(key=lambda item: (item.arrived_at_s, item.source_row, item.request_id))
    return rows, derived_prefill, derived_decode


def canonicalize_trace(
    raw_rows: Iterable[RawTraceRequest],
    *,
    profile: PrefillProfile,
    config: CanonicalizationConfig = CanonicalizationConfig(),
    enforce_gv3_window: bool = True,
) -> tuple[CanonicalTraceRequest, ...]:
    output: list[CanonicalTraceRequest] = []
    for raw in raw_rows:
        canonical_prefill = canonicalize_prefill_tokens(raw.num_prefill_tokens, config)
        canonical_decode = canonicalize_decode_tokens(raw.num_decode_tokens, config)
        profile_time, canonical_prefill_slo = canonicalize_prefill_slo(
            canonical_prefill, profile, config
        )
        canonical_decode_slo = canonicalize_decode_slo(raw.decode_slo_s, config)
        output.append(
            CanonicalTraceRequest(
                schema_version=TRACE_SCHEMA_VERSION,
                request_id=raw.request_id,
                source_row=raw.source_row,
                arrived_at_s=raw.arrived_at_s,
                actual_prefill_tokens=raw.num_prefill_tokens,
                canonical_prefill_tokens=canonical_prefill,
                prefill_rounding_delta_tokens=canonical_prefill - raw.num_prefill_tokens,
                actual_decode_tokens=raw.num_decode_tokens,
                canonical_decode_tokens=canonical_decode,
                actual_prefill_slo_s=raw.prefill_slo_s,
                canonical_prefill_slo_s=canonical_prefill_slo,
                prefill_profile_time_s=profile_time,
                actual_decode_slo_s=raw.decode_slo_s,
                canonical_decode_slo_s=canonical_decode_slo,
                prompt_mode=raw.prompt_mode,
                prompt_ref=raw.prompt_ref,
                seed=raw.seed,
                ignore_eos=raw.ignore_eos,
            )
        )
    if enforce_gv3_window:
        validate_gv3_launch_windows(output, config=config)
    return tuple(output)


def validate_gv3_launch_windows(
    rows: Iterable[CanonicalTraceRequest],
    *,
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> None:
    active: deque[CanonicalTraceRequest] = deque()
    active_prefill = 0
    for row in rows:
        lower = row.arrived_at_s - config.launch_window_s
        while active and active[0].arrived_at_s < lower - 1e-9:
            expired = active.popleft()
            active_prefill -= expired.canonical_prefill_tokens
        active.append(row)
        active_prefill += row.canonical_prefill_tokens
        if len(active) > config.launch_window_request_cap:
            raise CanonicalizationError(
                f"request {row.request_id}: {len(active)} arrivals in inclusive "
                f"{config.launch_window_s}s window exceed cap {config.launch_window_request_cap}"
            )
        if active_prefill > config.launch_window_prefill_cap:
            raise CanonicalizationError(
                f"request {row.request_id}: {active_prefill} canonical prefill tokens in "
                f"inclusive {config.launch_window_s}s window exceed cap "
                f"{config.launch_window_prefill_cap}"
            )


def write_canonical_trace(path: str | Path, rows: Iterable[CanonicalTraceRequest]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.csv_row())
    return output_path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(
    *,
    source_path: str | Path,
    canonical_path: str | Path,
    profile: PrefillProfile,
    result: TracePreparationResult,
    config: CanonicalizationConfig,
    enforce_gv3_window: bool,
) -> dict[str, object]:
    rows = result.rows
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "source": {
            "path": str(Path(source_path).expanduser().resolve()),
            "sha256": file_sha256(source_path),
        },
        "canonical": {
            "path": str(Path(canonical_path).expanduser().resolve()),
            "sha256": file_sha256(canonical_path),
            "row_count": len(rows),
            "first_arrival_s": rows[0].arrived_at_s if rows else None,
            "last_arrival_s": rows[-1].arrived_at_s if rows else None,
        },
        "profiles": {
            "prefill_path": str(profile.path),
            "prefill_sha256": profile.sha256,
        },
        "canonicalization": {
            "prefill_rounding": config.prefill_rounding,
            "prefill_grid_tokens": config.prefill_grid_tokens,
            "prefill_bounds": [config.min_prefill_tokens, config.max_prefill_tokens],
            "decode_policy": "identity_with_strict_bounds",
            "decode_bounds": [config.min_decode_tokens, config.max_decode_tokens],
            "prefill_slo": "prefill_profile_time_s * 3.0",
            "prefill_slowdown": config.prefill_slowdown,
            "decode_slo_s": config.canonical_decode_slo_s,
            "derived_prefill_slo_rows": result.derived_prefill_slos,
            "derived_decode_slo_rows": result.derived_decode_slos,
            "enforce_gv3_window": enforce_gv3_window,
            "launch_window_s": config.launch_window_s,
            "launch_window_request_cap": config.launch_window_request_cap,
            "launch_window_prefill_cap": config.launch_window_prefill_cap,
        },
        "model_contract": {
            "value_feature_schema": "markov_v2",
            "policy_feature_schema": "markov_v2",
            "actual_fields_drive": ["vllm_execution", "raw_slo_scoring"],
            "canonical_fields_drive": ["vidur_digital_twin", "dnn_features", "mcts"],
        },
    }


def write_manifest(path: str | Path, manifest: Mapping[str, object]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def assert_canonical_column_contract() -> None:
    actual = tuple(field.name for field in fields(CanonicalTraceRequest))
    if actual != CANONICAL_COLUMNS:
        raise AssertionError(f"canonical dataclass/CSV columns differ: {actual} != {CANONICAL_COLUMNS}")
