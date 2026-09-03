from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CPP_DIR = REPO_ROOT / "vidur" / "Game_Version3_Cpp"
CPP_BUILD_DIR = CPP_DIR / "build"
OUT_ROOT = REPO_ROOT / "simulator_output" / "GV3_Agent" / "native_alignment_tests"


def native_module_path() -> Path | None:
    candidates = sorted(CPP_DIR.glob("mcts_native_gv2*.so"))
    return candidates[0] if candidates else None


def build_native_cpp(*, force: bool = False, jobs: int | None = None) -> Path:
    existing = native_module_path()
    if existing is not None and not force:
        return existing

    CPP_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmake = [
        "cmake",
        "-S",
        str(CPP_DIR),
        "-B",
        str(CPP_BUILD_DIR),
        f"-DPython_EXECUTABLE={sys.executable}",
    ]
    subprocess.run(cmake, check=True, cwd=str(REPO_ROOT))
    build = ["cmake", "--build", str(CPP_BUILD_DIR)]
    if jobs is None:
        jobs = max(1, min(16, os.cpu_count() or 1))
    build += ["-j", str(int(jobs))]
    subprocess.run(build, check=True, cwd=str(REPO_ROOT))

    built = native_module_path()
    if built is None:
        raise FileNotFoundError(f"native module was not produced in {CPP_DIR}")
    return built


def import_native_cpp(*, build_if_missing: bool = True, force_build: bool = False) -> Any:
    if build_if_missing:
        build_native_cpp(force=force_build)
    if str(CPP_DIR) not in sys.path:
        sys.path.insert(0, str(CPP_DIR))
    # Make sure we do not accidentally reuse the old top-level module from a
    # previous import path.  The package import vidur.mcts.mcts_native_gv2 is a
    # different key and is intentionally not used by these tests.
    sys.modules.pop("mcts_native_gv2", None)
    return importlib.import_module("mcts_native_gv2")


def output_dir(name: str) -> Path:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_args(
    name: str,
    *,
    num_roots: int = 4,
    history_hops_min: int = 0,
    history_hops_max: int = 8,
    history_seed: int = 2026,
    frontier_parity_roots: int = 2,
    feature_tolerance: float = 1e-6,
    bellman_q_tolerance: float = 1e-2,
    strict_adversary_q_parity: bool = False,
    skip_bootstrap_bellman: bool = True,
    model_version: int = 0,
    model_device: str = "cpu",
    checkpoint_path: str | None = None,
) -> argparse.Namespace:
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    if checkpoint_path is None:
        checkpoint_path = str(nlt._default_checkpoint_path())
    return argparse.Namespace(
        num_roots=int(num_roots),
        history_hops_min=int(history_hops_min),
        history_hops_max=int(history_hops_max),
        history_seed=int(history_seed),
        frontier_parity_roots=int(frontier_parity_roots),
        feature_tolerance=float(feature_tolerance),
        bellman_q_tolerance=float(bellman_q_tolerance),
        strict_adversary_q_parity=bool(strict_adversary_q_parity),
        skip_bootstrap_bellman=bool(skip_bootstrap_bellman),
        model_version=int(model_version),
        model_device=str(model_device),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir(name)),
    )


def prepare_python_roots(args: argparse.Namespace):
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    out = Path(args.output_dir)
    cfg_python = nlt._make_cfg(args, environment_lang="python")
    simulator, env, explore_cfg, roots = nlt._generate_python_roots(args, cfg_python, out)
    return cfg_python, simulator, env, explore_cfg, roots


def run_native_logger(native: Any, simulator: Any, args: argparse.Namespace):
    from vidur.Game_Version3.tests import native_logger_tests as nlt

    out = Path(args.output_dir)
    cfg_native = nlt._make_cfg(args, environment_lang="native")
    return nlt._run_native_logger(
        native=native,
        simulator=simulator,
        args=args,
        cfg=cfg_native,
        out_dir=out,
    )
