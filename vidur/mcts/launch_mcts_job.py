from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path


def main() -> None:
    # Resolve repo root (folder that contains simulator_output/ and top-level vidur/)
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]  # .../vidur

    out_dir = repo_root / "simulator_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefill_profile = out_dir / "prefill_profile.csv"
    trace_csv = out_dir / "test_mcts_trace.csv"
    log_file = out_dir / "MCTS_JOB_logs.txt"

    # Construct the run_mcts command with the requested hard-coded configuration
    cmd = [
        sys.executable,
        "-m",
        "vidur.mcts.run_mcts",
        "--mcts_iterations",
        "100000",
        "--mcts_simulation_random_tries",
        "1",
        "--mcts_simulation_depth",
        "4",
        "--mcts_interval_request_size",
        "512",
        "--mcts_min_request_tokens",
        "512",
        "--mcts_max_request_tokens",
        "10240",
        "--mcts_exploration_constant",
        "1.4",
        "--mcts_max_branching",
        "30",
        "--mcts_controller_budget_combs",
        "20",
        "--mcts_prefill_profile",
        str(prefill_profile),
        "--mcts_log_csv",
        str(trace_csv),
        "--replica_config_model_name",
        "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device",
        "h100",
        "--replica_config_network_device",
        "h100_dgx",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_tensor_parallel_size",
        "1",
        "--replica_config_num_pipeline_stages",
        "1",
        "--global_scheduler_config_type",
        "round_robin",
        "--replica_scheduler_config_type",
        "vllm_v1",
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

