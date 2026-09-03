"""Install the guarded per-request token-cap hook into tested vLLM releases."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path


SUPPORTED_VLLM_VERSIONS = ("0.13.0", "0.26.0")
PATCH_MARKER = "VIDUR_GV3_PER_REQUEST_TOKEN_CAP_V1"

_RUNNING_NEEDLE = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n            num_new_tokens = min(num_new_tokens, token_budget)\n"""

_RUNNING_REPLACEMENT = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n            # VIDUR_GV3_PER_REQUEST_TOKEN_CAP_V1: policy only constrains work;\n            # stock vLLM retains KV allocation, preemption, and output ownership.\n            vidur_caps = getattr(self, \"_vidur_request_token_caps\", None)\n            if vidur_caps is not None and request.request_id in vidur_caps:\n                num_new_tokens = min(num_new_tokens, vidur_caps[request.request_id])\n            num_new_tokens = min(num_new_tokens, token_budget)\n"""

_WAITING_NEEDLE = """                    threshold = self.scheduler_config.long_prefill_token_threshold\n                    if 0 < threshold < num_new_tokens:\n                        num_new_tokens = threshold\n\n                    # chunked prefill has to be enabled explicitly to allow\n"""

_WAITING_REPLACEMENT = """                    threshold = self.scheduler_config.long_prefill_token_threshold\n                    if 0 < threshold < num_new_tokens:\n                        num_new_tokens = threshold\n                    # VIDUR_GV3_PER_REQUEST_TOKEN_CAP_V1\n                    vidur_caps = getattr(self, \"_vidur_request_token_caps\", None)\n                    if vidur_caps is not None and request.request_id in vidur_caps:\n                        num_new_tokens = min(num_new_tokens, vidur_caps[request.request_id])\n\n                    # chunked prefill has to be enabled explicitly to allow\n"""


def scheduler_source_path() -> Path:
    from vllm.v1.core.sched.scheduler import Scheduler

    path = inspect.getsourcefile(Scheduler)
    if path is None:
        raise RuntimeError("could not locate vLLM Scheduler source")
    return Path(path).resolve()


def verify_vllm_version() -> None:
    import vllm

    observed = str(vllm.__version__)
    if observed not in SUPPORTED_VLLM_VERSIONS:
        supported = ", ".join(SUPPORTED_VLLM_VERSIONS)
        raise RuntimeError(
            f"scheduler patch requires a tested vLLM version ({supported}), got {observed}"
        )


def install_patch() -> Path:
    verify_vllm_version()
    path = scheduler_source_path()
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return path
    if source.count(_RUNNING_NEEDLE) != 1 or source.count(_WAITING_NEEDLE) != 1:
        raise RuntimeError(
            "pinned vLLM scheduler source no longer matches guarded patch needles"
        )
    source = source.replace(_RUNNING_NEEDLE, _RUNNING_REPLACEMENT)
    source = source.replace(_WAITING_NEEDLE, _WAITING_REPLACEMENT)
    path.write_text(source, encoding="utf-8")
    return path


def verify_patch() -> Path:
    verify_vllm_version()
    path = scheduler_source_path()
    source = path.read_text(encoding="utf-8")
    if source.count(PATCH_MARKER) != 2:
        raise RuntimeError("vLLM per-request token-cap hook is absent or malformed")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("install", "verify"))
    args = parser.parse_args()
    path = install_patch() if args.command == "install" else verify_patch()
    print(path)


if __name__ == "__main__":
    main()
