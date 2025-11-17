from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Sequence, Iterable


# Local MCTS configuration dataclasses (moved from config.py)
@dataclass
class RequestSLOOptions:
    """Discrete SLO options available to the adversary."""

    prefill_slos: Sequence[float] = field(default_factory=lambda: [3.0])
    decode_slos: Sequence[float] = field(default_factory=lambda: [5])


@dataclass
class MCTSConstraintConfig:
    """Bounds that govern admissible controller/adversary actions."""

    maximum_qps: int = 5
    min_request_tokens: int = 512
    max_request_tokens: Optional[int] = None
    interval_request_size: int = 512
    request_slo_options: RequestSLOOptions = field(default_factory=RequestSLOOptions)
    prefill_slowdown: float = 1.0 ## Unnnecesary variables that will be removed soon IA
    prefill_profile_path: Optional[str] = None


@dataclass
class MCTSExploreConfig:
    """Parameters that shape the Monte-Carlo Tree Search behaviour."""

    simulation_depth: int = 4
    simulation_random_tries: int = 1
    exploration_constant: float = 1.4
    max_branching: int = 10 ## Max branching factor only for Adversary
    controller_budget_combs: int = 10  # for potential future use

def main() -> None:
    # Resolve repo root (folder that contains simulator_output/ and top-level vidur/)
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]  # .../vidur

    out_dir = repo_root / "simulator_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefill_profile = out_dir / "prefill_profile.csv"
    trace_csv = out_dir / "test_mcts_trace.csv"
    log_file = out_dir / "MCTS_JOB_logs.txt"

    # Single-source configuration: define once and generate CLI args
    constraint = MCTSConstraintConfig(
        maximum_qps=5,
        min_request_tokens=512,
        max_request_tokens=3072,
        interval_request_size=512,
        prefill_slowdown=1.0,
        prefill_profile_path=str(prefill_profile),
    )
    explore = MCTSExploreConfig(
        simulation_depth=10,
        simulation_random_tries=1,
        exploration_constant=1.4,
        max_branching=10,
    )
    iterations = 50

    def cfg_to_args() -> Iterable[str]:
        args: list[str] = [
            "--mcts_iterations", str(iterations),
            "--mcts_simulation_random_tries", str(explore.simulation_random_tries),
            "--mcts_simulation_depth", str(explore.simulation_depth),
            "--mcts_interval_request_size", str(constraint.interval_request_size),
            "--mcts_maximum_qps", str(constraint.maximum_qps),
            "--mcts_min_request_tokens", str(constraint.min_request_tokens),
            "--mcts_exploration_constant", str(explore.exploration_constant),
            "--mcts_max_branching", str(explore.max_branching),
        ]
        if constraint.max_request_tokens is not None:
            args += ["--mcts_max_request_tokens", str(constraint.max_request_tokens)]
        if constraint.prefill_profile_path:
            args += ["--mcts_prefill_profile", str(constraint.prefill_profile_path)]
        # If SLO overrides are singletons, pass them explicitly.
        if len(constraint.request_slo_options.prefill_slos) == 1:
            args += ["--mcts_prefill_slos", str(constraint.request_slo_options.prefill_slos[0])]
        if len(constraint.request_slo_options.decode_slos) == 1:
            args += ["--mcts_decode_slos", str(constraint.request_slo_options.decode_slos[0])]
        return args

    cmd = [
        sys.executable,
        "-m",
        "vidur.mcts.run_mcts",
        *cfg_to_args(),
        "--mcts_log_csv", str(trace_csv),
        "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device", "h100",
        "--replica_config_network_device", "h100_dgx",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--global_scheduler_config_type", "round_robin",
        "--replica_scheduler_config_type", "vllm_v1",
    ]

    # Open the log file for both stdout and stderr, append mode
    log_fh = log_file.open("a", buffering=1)

    # Write a header so we can see restarts clearly
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log_fh.write("\n" + "=" * 80 + "\n")
    log_fh.write(f"Launch time: {ts}\n")
    log_fh.write("Command: " + " ".join(cmd) + "\n")
    log_fh.flush()

    # Detach from the controlling terminal so the job survives SSH/VSCode disconnects.
    # start_new_session=True creates a new session and process group (like setsid).
    # Also redirect stdin to devnull so the child has no terminal input.
    with open(os.devnull, "rb") as devnull:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdin=devnull,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )

    log_fh.write(f"Launched PID: {proc.pid}\n")
    log_fh.flush()
    # Intentionally do not wait; exit to return control to the caller.

    # Print a short message to the console as well (if any)
    print(f"Launched background MCTS job (PID {proc.pid}). Logs: {log_file}")


if __name__ == "__main__":
    main()
