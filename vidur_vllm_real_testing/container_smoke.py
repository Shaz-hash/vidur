"""Container acceptance gate for frozen profiles, traces, and tokenizer parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tokenizer_compatibility import verify_model_tokenizer
from .verify_ready_trace import verify_ready_trace


TRACE_NAMES = ("splitwise_conv_20s", "gv3_legal_20s")


def run_smoke(
    *,
    package_root: str | Path,
    model_tokenizer: str,
    revision: str | None,
    require_vllm: bool,
    local_files_only: bool,
    report_path: str | Path | None,
) -> dict[str, object]:
    root = Path(package_root).expanduser().resolve()
    tokenizer_dir = root / "tokenizer" / "llama3_8b"
    prefill_profile = root / "artifacts" / "prefill_profile.csv"
    decode_profile = root / "artifacts" / "decode_profile.csv"

    trace_paths: list[Path] = []
    trace_counts: dict[str, int] = {}
    for name in TRACE_NAMES:
        raw_trace = root / "traces" / f"{name}_english_raw.csv"
        count = verify_ready_trace(
            manifest_path=root / "traces" / f"{name}_english_manifest.json",
            source_path=raw_trace,
            canonical_path=root / "traces" / f"{name}_english_canonical.csv",
            prompt_catalog_path=root / "prompts" / name / "prompt_catalog.json",
            tokenizer_dir=tokenizer_dir,
            prefill_profile_path=prefill_profile,
            decode_profile_path=decode_profile,
        )
        trace_paths.append(raw_trace)
        trace_counts[name] = count

    tokenizer_report = verify_model_tokenizer(
        model_tokenizer=model_tokenizer,
        pinned_tokenizer_dir=tokenizer_dir,
        trace_paths=trace_paths,
        revision=revision,
        require_vllm=require_vllm,
        local_files_only=local_files_only,
    )
    scheduler_report: dict[str, object] = {"checked": False}
    if require_vllm:
        from .patch_vllm_scheduler import verify_patch

        scheduler_path = verify_patch()
        from .vllm_scheduler import GV3Scheduler

        scheduler_report = {
            "checked": True,
            "class": f"{GV3Scheduler.__module__}.{GV3Scheduler.__name__}",
            "patched_source": str(scheduler_path),
        }
    report: dict[str, object] = {
        "status": "ready",
        "trace_counts": trace_counts,
        "tokenizer": tokenizer_report,
        "scheduler": scheduler_report,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if report_path is not None:
        output = Path(report_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM image acceptance gates.")
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--model-tokenizer", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_smoke(
        package_root=args.package_root,
        model_tokenizer=args.model_tokenizer,
        revision=args.revision,
        require_vllm=args.require_vllm,
        local_files_only=args.local_files_only,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
