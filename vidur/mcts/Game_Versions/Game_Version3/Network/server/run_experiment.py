from __future__ import annotations

import argparse
import subprocess
import sys

from ..network_config import DEFAULT_NETWORK_CONFIG


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    defaults = DEFAULT_NETWORK_CONFIG.task
    parser = argparse.ArgumentParser(
        description="Run multiple GV3 network generations as separate orchestrator processes."
    )
    parser.add_argument("--start-generation", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=defaults.num_generations)
    parser.add_argument("--session-prefix", default="gv3_network_experiment")
    parser.add_argument("--machine", action="append", default=None)
    parser.add_argument("--weights-path", default=str(DEFAULT_NETWORK_CONFIG.paths.default_weights_path))
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def main() -> None:
    args, extra = _parse_args()
    start = int(args.start_generation)
    count = max(1, int(args.num_generations))
    machines = list(args.machine or [])

    for gen in range(start, start + count):
        session_id = f"{args.session_prefix}_gen_{int(gen):06d}"
        cmd = [
            sys.executable,
            "-m",
            "vidur.mcts.Game_Versions.Game_Version3.Network.server.orchestrator",
            "--session-id",
            session_id,
            "--generation",
            str(int(gen)),
            "--model-version",
            str(int(gen)),
            "--weights-path",
            str(args.weights_path),
        ]
        for machine in machines:
            cmd.extend(["--machine", str(machine)])
        cmd.extend(extra)

        print(
            f"[GV3 network experiment] generation {int(gen) - int(start) + 1}/{int(count)} "
            f"starting: gen={int(gen):06d}, session_id={session_id}",
            flush=True,
        )
        subprocess.run(cmd, check=True)
        print(
            f"[GV3 network experiment] generation {int(gen) - int(start) + 1}/{int(count)} "
            f"complete: gen={int(gen):06d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
