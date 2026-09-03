#!/usr/bin/env python3
"""Warm vLLM kernels without entering the measured GV3 scheduler state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-tokens",
        default="128,256,512,1024,2048,3072,4096,128",
        help="comma-separated prefill shapes; repeat a shape to verify it is warm",
    )
    args = parser.parse_args()

    token_counts = [int(value) for value in args.prompt_tokens.split(",")]
    if not token_counts or any(value <= 0 for value in token_counts):
        raise ValueError("all warm-up prompt-token counts must be positive")
    durations: list[float] = []
    for index, prompt_tokens in enumerate(token_counts):
        payload = json.dumps(
            {
                "model": args.model,
                "prompt": [1000] * prompt_tokens,
                "max_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
                "logprobs": 1,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": f"vidur-warmup-{index:06d}",
            },
            method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.status != 200:
                raise RuntimeError(f"warm-up HTTP status {response.status}")
            result = json.loads(response.read().decode("utf-8"))
        durations.append(time.monotonic() - started)
        if not result.get("choices"):
            raise RuntimeError("warm-up returned no completion choice")

    report = {
        "requests": len(token_counts),
        "prompt_tokens": token_counts,
        "durations_s": durations,
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
