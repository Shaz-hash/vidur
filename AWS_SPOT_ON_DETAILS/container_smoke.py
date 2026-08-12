#!/usr/bin/env python3
"""Fail-fast validation for the portable AlphaGoZero worker image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


APP_HOME = Path(os.environ.get("APP_HOME", "/home/ubuntu/vidur-classical-search"))
EXPECTED_PROFILE_SHA256 = {
    "prefill_profile.csv": "47f2ed85aca4a5ec47eacae8425d38284e76759b75f5c069852d393ecc62329b",
    "decode_profile.csv": "b14044faaa5f9f5fea1b159f5bd031ce8538b30b59eb8acd32cd9625093c4d2a",
}
REQUIRED_NATIVE_EXPORTS = (
    "NewFeatures226HGBRuntime",
    "NativeHGBModelRuntime",
    "discounted_trajectory_targets",
    "search_mcts_hgb226_value_prior_hgb",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def _command_check(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=APP_HOME,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"{module} --help failed rc={result.returncode}\n{result.stdout[-4000:]}")


def run(*, build_check: bool) -> dict[str, object]:
    del build_check
    if not APP_HOME.is_dir():
        raise FileNotFoundError(f"application directory is missing: {APP_HOME}")

    versions = {
        name: _module_version(name)
        for name in ("joblib", "numpy", "psutil", "ray", "scipy", "sklearn", "torch", "wandb")
    }
    expected_versions = {
        "joblib": "1.5.0",
        "numpy": "1.26.4",
        "psutil": "7.0.0",
        "ray": "2.31.0",
        "scipy": "1.15.3",
        "sklearn": "1.5.0",
        "torch": "2.9.1",
        "wandb": "0.16.6",
    }
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in expected_versions.items()
        if str(versions.get(name, "")).split("+", 1)[0] != expected
    }
    if mismatches:
        raise RuntimeError(f"runtime package mismatch: {mismatches}")

    profile_hashes: dict[str, str] = {}
    for filename, expected in EXPECTED_PROFILE_SHA256.items():
        path = APP_HOME / "simulator_output" / filename
        if not path.is_file():
            raise FileNotFoundError(f"runtime profile is missing: {path}")
        actual = _sha256(path)
        profile_hashes[filename] = actual
        if actual != expected:
            raise RuntimeError(f"{filename} sha256={actual}, expected={expected}")

    cache_root = APP_HOME / "cache"
    cache_files = sum(1 for path in cache_root.rglob("*") if path.is_file())
    if cache_files < 100:
        raise RuntimeError(f"execution cache is incomplete: files={cache_files}")

    cpp_dir = APP_HOME / "vidur" / "Game_Version3_Cpp"
    sys.path.insert(0, str(cpp_dir))
    native = importlib.import_module("mcts_native_gv2")
    missing_exports = [name for name in REQUIRED_NATIVE_EXPORTS if not hasattr(native, name)]
    if missing_exports:
        raise RuntimeError(f"native module is missing exports: {missing_exports}")
    discounted = list(native.discounted_trajectory_targets([-1.0, -2.0], [0.9, 0.8], -3.0))
    if len(discounted) != 2:
        raise RuntimeError(f"native discounted return smoke failed: {discounted}")

    _command_check("vidur.AlphaGoZero.worker_daemon")
    _command_check("vidur.AlphaGoZero.spot_worker_daemon")
    _command_check("vidur.AlphaGoZero.spot_work_cli")
    _command_check("vidur.bellman_v4_adv.arena_mcts_value_runnerCPP")

    from vidur.AlphaGoZero.spot_worker_daemon import detect_worker_resources

    resource_probe = detect_worker_resources(
        argparse.Namespace(
            parallel_games=0,
            resource_reserve_cpus=-1,
            resource_memory_gib_per_game=1.0,
            worker_threads=1,
            rollout_parallel_threads=1,
        )
    )
    required_resource_fields = {
        "cpu_count",
        "reserved_cpu_count",
        "cpu_slots",
        "available_memory_bytes",
        "memory_slots",
        "effective_capacity",
    }
    missing_resource_fields = sorted(required_resource_fields - resource_probe.keys())
    if missing_resource_fields:
        raise RuntimeError(
            f"resource detection missing fields: {missing_resource_fields}"
        )

    result: dict[str, object] = {
        "ok": True,
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "git_sha": os.environ.get("AGZ_IMAGE_GIT_SHA", "unknown"),
        "packages": versions,
        "profiles": profile_hashes,
        "execution_cache_files": cache_files,
        "native_module": str(Path(native.__file__).resolve()),
        "native_exports_checked": list(REQUIRED_NATIVE_EXPORTS),
        "discounted_return_smoke": discounted,
        "resource_probe": resource_probe,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-check", action="store_true")
    args = parser.parse_args()
    run(build_check=bool(args.build_check))


if __name__ == "__main__":
    main()
