"""Deterministic mapping from real request fields to the GV3 model domain."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import math
from pathlib import Path


TRACE_SCHEMA_VERSION = "gv3_vllm_trace_v1"
PREFILL_GRID_TOKENS = 128
MIN_CANONICAL_PREFILL_TOKENS = 128
MAX_PREFILL_TOKENS = 4096
MIN_DECODE_TOKENS = 1
MAX_DECODE_TOKENS = 864
PREFILL_SLOWDOWN = 3.0
CANONICAL_DECODE_SLO_S = 0.05
ADVERSARY_TICK_S = 0.2
LAUNCH_WINDOW_S = 1.0
LAUNCH_WINDOW_REQUEST_CAP = 7
LAUNCH_WINDOW_PREFILL_CAP = 7 * 1024
PREFILL_ROUNDING_NEAREST = "nearest_128_half_up"
PREFILL_ROUNDING_CEILING = "ceiling_128"


class CanonicalizationError(ValueError):
    """A raw request cannot be represented by the configured GV3 contract."""


@dataclass(frozen=True)
class CanonicalizationConfig:
    prefill_grid_tokens: int = PREFILL_GRID_TOKENS
    min_prefill_tokens: int = MIN_CANONICAL_PREFILL_TOKENS
    max_prefill_tokens: int = MAX_PREFILL_TOKENS
    min_decode_tokens: int = MIN_DECODE_TOKENS
    max_decode_tokens: int = MAX_DECODE_TOKENS
    prefill_slowdown: float = PREFILL_SLOWDOWN
    canonical_decode_slo_s: float = CANONICAL_DECODE_SLO_S
    launch_window_s: float = LAUNCH_WINDOW_S
    launch_window_request_cap: int = LAUNCH_WINDOW_REQUEST_CAP
    launch_window_prefill_cap: int = LAUNCH_WINDOW_PREFILL_CAP
    prefill_rounding: str = PREFILL_ROUNDING_NEAREST

    def __post_init__(self) -> None:
        if self.prefill_grid_tokens <= 0:
            raise ValueError("prefill_grid_tokens must be positive")
        if self.min_prefill_tokens <= 0 or self.max_prefill_tokens < self.min_prefill_tokens:
            raise ValueError("invalid prefill bounds")
        if self.min_decode_tokens <= 0 or self.max_decode_tokens < self.min_decode_tokens:
            raise ValueError("invalid decode bounds")
        if self.prefill_slowdown <= 0.0 or self.canonical_decode_slo_s <= 0.0:
            raise ValueError("SLO parameters must be positive")
        if self.prefill_rounding not in {
            PREFILL_ROUNDING_NEAREST,
            PREFILL_ROUNDING_CEILING,
        }:
            raise ValueError(
                "prefill_rounding must be nearest_128_half_up or ceiling_128"
            )


@dataclass(frozen=True)
class PrefillProfile:
    path: Path
    seconds_by_tokens: dict[int, float]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "PrefillProfile":
        profile_path = Path(path).expanduser().resolve()
        if not profile_path.is_file():
            raise FileNotFoundError(profile_path)

        values: dict[int, float] = {}
        with profile_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["prefill_tokens", "prefill_time_seconds"]:
                raise CanonicalizationError(
                    f"unexpected prefill profile columns: {reader.fieldnames}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    tokens = int(row["prefill_tokens"])
                    seconds = float(row["prefill_time_seconds"])
                except (TypeError, ValueError) as exc:
                    raise CanonicalizationError(
                        f"invalid prefill profile row {line_number}: {row}"
                    ) from exc
                if tokens <= 0 or not math.isfinite(seconds) or seconds <= 0.0:
                    raise CanonicalizationError(
                        f"invalid prefill profile row {line_number}: {row}"
                    )
                if tokens in values:
                    raise CanonicalizationError(f"duplicate prefill profile size {tokens}")
                values[tokens] = seconds

        expected = list(range(PREFILL_GRID_TOKENS, MAX_PREFILL_TOKENS + 1, PREFILL_GRID_TOKENS))
        if sorted(values) != expected:
            raise CanonicalizationError(
                "prefill profile must contain every 128-token point from 128 through 4096"
            )
        digest = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        return cls(path=profile_path, seconds_by_tokens=values, sha256=digest)

    def execution_time_s(self, canonical_prefill_tokens: int) -> float:
        try:
            return self.seconds_by_tokens[int(canonical_prefill_tokens)]
        except KeyError as exc:
            raise CanonicalizationError(
                f"prefill size {canonical_prefill_tokens} has no exact profile entry"
            ) from exc


def _positive_finite(value: float, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CanonicalizationError(f"{field} must be finite and > 0, got {value!r}")
    return result


def canonicalize_prefill_tokens(
    actual_tokens: int,
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> int:
    """Map a physical prompt to the configured 128-token model grid."""

    actual = int(actual_tokens)
    if actual != actual_tokens or actual <= 0:
        raise CanonicalizationError(f"num_prefill_tokens must be a positive integer, got {actual_tokens!r}")
    if actual > config.max_prefill_tokens:
        raise CanonicalizationError(
            f"num_prefill_tokens {actual} exceeds GV3 maximum {config.max_prefill_tokens}"
        )
    step = config.prefill_grid_tokens
    if config.prefill_rounding == PREFILL_ROUNDING_CEILING:
        rounded = ((actual + step - 1) // step) * step
    else:
        rounded = ((actual + step // 2) // step) * step
    return min(config.max_prefill_tokens, max(config.min_prefill_tokens, rounded))


def canonicalize_decode_tokens(
    actual_tokens: int,
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> int:
    """Decode length is continuous inside the trained GV3 range; do not bucket it."""

    actual = int(actual_tokens)
    if actual != actual_tokens:
        raise CanonicalizationError(f"num_decode_tokens must be integral, got {actual_tokens!r}")
    if not config.min_decode_tokens <= actual <= config.max_decode_tokens:
        raise CanonicalizationError(
            f"num_decode_tokens {actual} is outside "
            f"[{config.min_decode_tokens}, {config.max_decode_tokens}]"
        )
    return actual


def canonicalize_prefill_slo(
    canonical_prefill_tokens: int,
    profile: PrefillProfile,
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> tuple[float, float]:
    profile_time = profile.execution_time_s(canonical_prefill_tokens)
    return profile_time, profile_time * config.prefill_slowdown


def canonicalize_decode_slo(
    actual_decode_slo_s: float,
    config: CanonicalizationConfig = CanonicalizationConfig(),
) -> float:
    _positive_finite(actual_decode_slo_s, field="decode_slo_s")
    return config.canonical_decode_slo_s


def validate_actual_prefill_slo(actual_prefill_slo_s: float) -> float:
    return _positive_finite(actual_prefill_slo_s, field="prefill_slo_s")
