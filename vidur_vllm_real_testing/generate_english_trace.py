"""CLI for materializing exact-length English prompts for a strict raw trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from .prompt_materialization import write_prompt_artifacts


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_DIR = PACKAGE_DIR / "tokenizer" / "llama3_8b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact Llama-3 English prompts.")
    parser.add_argument("--input", required=True, help="Strict raw trace template")
    parser.add_argument("--output", required=True, help="Materialized raw trace")
    parser.add_argument("--prompt-root", required=True, help="Text/token-ID output directory")
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, catalog = write_prompt_artifacts(
        raw_trace_path=args.input,
        output_trace_path=args.output,
        prompt_root=args.prompt_root,
        tokenizer_dir=args.tokenizer_dir,
    )
    print(f"materialized {count} English prompts; catalog={catalog}")


if __name__ == "__main__":
    main()
