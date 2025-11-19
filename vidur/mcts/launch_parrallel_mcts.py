#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

NUM_RUNS = 12


LOG_PATH = Path("simulator_output/Parrallel_Launch/MCTS_JOB_logs.txt")
BASE_CSV = "simulator_output/Parrallel_Launch/test_mcts_trace.csv"


depth_per_process = [0, 22, 94, 38, 71, 19, 83, 46, 29, 100, 22, 14, 67, 31, 89]
# depth_per_process = [0]

def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    procs = []
    # Open the log file once and share the handle across processes
    with LOG_PATH.open("a") as log_file:
        for i in range(1, NUM_RUNS + 1):
            run_id = f"P{i}"
            depth = depth_per_process[i - 1]
            cmd = [
                sys.executable, "-m", "vidur.mcts.run_mcts",
                "--mcts_run_id", run_id,
                "--mcts_history_depth", str(depth),
                "--mcts_iterations", "50000",
                "--mcts_simulation_random_tries", "1",
                "--mcts_simulation_depth", "8",
                "--mcts_interval_request_size", "512",
                "--mcts_maximum_qps", "10",
                "--mcts_min_request_tokens", "512",
                "--mcts_exploration_constant", "1.7",
                "--mcts_max_branching", "10",
                "--mcts_max_request_tokens", "3072",
                "--mcts_prefill_profile", "simulator_output/prefill_profile.csv",
                "--mcts_prefill_slos", "3.0",
                "--mcts_decode_slos", "50",
                "--mcts_log_csv", BASE_CSV,
                "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
                "--replica_config_device", "a100",
                "--replica_config_network_device", "a100_dgx",
                "--cluster_config_num_replicas", "1",
                "--replica_config_tensor_parallel_size", "1",
                "--replica_config_num_pipeline_stages", "1",
                "--global_scheduler_config_type", "round_robin",
                "--replica_scheduler_config_type", "vllm_v1",
            ]
            # p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            # procs.append(p)

            p = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,    # no terminal input
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,      # detach from SSH/controlling terminal
            )
            procs.append(p)

        # Wait for all runs to finish
        for p in procs:
            p.wait()

if __name__ == "__main__":
    main()
