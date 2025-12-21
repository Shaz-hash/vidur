"""
MCTS public API.

This package historically re-exported many symbols at import time. Some of those
pull in optional heavyweight dependencies (e.g. numpy). To keep lightweight
submodules (like `vidur.mcts.analysis.*`) runnable without requiring the full
stack, we lazily import the heavier modules when their symbols are accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import MCTSConstraintConfig, MCTSExploreConfig, RequestSLOOptions

if TYPE_CHECKING:
    from .environment import (  # noqa: F401
        AdversaryAction,
        ControllerAction,
        VidurGameStats,
        VidurMCTSEnvironment,
        VidurMCTSState,
    )
    from .mcts import VidurMCTS  # noqa: F401
    from .prefill_calibrator import PrefillProfile  # noqa: F401

__all__ = [
    "AdversaryAction",
    "ControllerAction",
    "MCTSConstraintConfig",
    "MCTSExploreConfig",
    "RequestSLOOptions",
    "VidurMCTS",
    "VidurMCTSEnvironment",
    "VidurMCTSState",
    "VidurGameStats",
    "PrefillProfile",
]


def __getattr__(name: str):
    if name in {
        "AdversaryAction",
        "ControllerAction",
        "VidurGameStats",
        "VidurMCTSEnvironment",
        "VidurMCTSState",
    }:
        from . import environment as _env

        return getattr(_env, name)
    if name == "PrefillProfile":
        from . import prefill_calibrator as _pref

        return getattr(_pref, name)
    if name == "VidurMCTS":
        from . import mcts as _mcts

        return getattr(_mcts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
