"""Prepare and validate a real-request trace before building the GPU image."""

from __future__ import annotations

import argparse
from pathlib import Path

from .canonicalization import (
    CanonicalizationConfig,
    PREFILL_ROUNDING_CEILING,
    PREFILL_ROUNDING_NEAREST,
    PrefillProfile,
)
from .trace_contract import (
    TracePreparationResult,
    assert_canonical_column_contract,
    build_manifest,
    canonicalize_trace,
    load_raw_trace,
    write_canonical_trace,
    write_manifest,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_PREFILL_PROFILE = REPO_ROOT / "simulator_output" / "prefill_profile.csv"


def prepare(
    *,
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    prefill_profile_path: str | Path = DEFAULT_PREFILL_PROFILE,
    config: CanonicalizationConfig | None = None,
    derive_missing_slos: bool = False,
    enforce_gv3_window: bool = True,
) -> TracePreparationResult:
    assert_canonical_column_contract()
    canonical_config = config or CanonicalizationConfig()
    profile = PrefillProfile.load(prefill_profile_path)
    raw_rows, derived_prefill, derived_decode = load_raw_trace(
        input_path,
        profile=profile,
        config=canonical_config,
        derive_missing_slos=derive_missing_slos,
    )
    canonical_rows = canonicalize_trace(
        raw_rows,
        profile=profile,
        config=canonical_config,
        enforce_gv3_window=enforce_gv3_window,
    )
    result = TracePreparationResult(
        rows=canonical_rows,
        derived_prefill_slos=derived_prefill,
        derived_decode_slos=derived_decode,
    )
    canonical_path = write_canonical_trace(output_path, canonical_rows)
    manifest = build_manifest(
        source_path=input_path,
        canonical_path=canonical_path,
        profile=profile,
        result=result,
        config=canonical_config,
        enforce_gv3_window=enforce_gv3_window,
    )
    write_manifest(manifest_path, manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and canonicalize a real vLLM request trace for GV3."
    )
    parser.add_argument("--input", required=True, help="Raw trace CSV")
    parser.add_argument("--output", required=True, help="Canonical trace CSV")
    parser.add_argument("--manifest", required=True, help="Output JSON manifest")
    parser.add_argument(
        "--prefill-profile",
        default=str(DEFAULT_PREFILL_PROFILE),
        help="Frozen Vidur prefill profile CSV",
    )
    parser.add_argument(
        "--prefill-rounding",
        choices=(PREFILL_ROUNDING_NEAREST, PREFILL_ROUNDING_CEILING),
        default=PREFILL_ROUNDING_NEAREST,
        help="Map physical prompt lengths to the model grid",
    )
    parser.add_argument(
        "--derive-missing-slos",
        action="store_true",
        help="Derive absent SLOs for synthetic fixtures; real benchmark traces should supply them",
    )
    parser.add_argument(
        "--allow-out-of-window-trace",
        action="store_true",
        help="Do not enforce GV3's seven-request/7168-token inclusive one-second launch window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        prefill_profile_path=args.prefill_profile,
        config=CanonicalizationConfig(prefill_rounding=args.prefill_rounding),
        derive_missing_slos=bool(args.derive_missing_slos),
        enforce_gv3_window=not bool(args.allow_out_of_window_trace),
    )
    print(
        f"prepared {len(result.rows)} rows; "
        f"derived prefill SLOs={result.derived_prefill_slos}, "
        f"decode SLOs={result.derived_decode_slos}"
    )


if __name__ == "__main__":
    main()
