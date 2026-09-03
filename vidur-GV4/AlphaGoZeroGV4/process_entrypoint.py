"""Start a GV4 module without exposing the local types package during startup."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: process_entrypoint.py MODULE [ARG ...]")
    module = sys.argv[1]
    package_root = Path(__file__).resolve().parent.parent
    repository_root = package_root.parent

    # Insert GV4 only after startup, when the standard-library types module is loaded.
    sys.path.insert(0, str(repository_root))
    sys.path.insert(0, str(package_root))
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
