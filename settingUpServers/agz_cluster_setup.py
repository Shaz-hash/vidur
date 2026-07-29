#!/usr/bin/env python3
"""Inventory-driven provisioning and validation for AlphaGoZero ARM clusters."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = Path(__file__).with_name("agz_clusters.json")
REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_KEY = Path("~/.ssh/shaz-pr1.pem").expanduser()
DEFAULT_SSH_CONFIG = Path("~/.ssh/config").expanduser()

PARENT_DATASETS = {
    "worker1": "bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_01_bellman-classical-worker-1_hops_151_300",
    "worker2": "bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_02_bellman-classical-worker-2_hops_301_450",
    "worker3": "bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_03_bellman-classical-worker-3_hops_451_600",
    "worker4": "bellman_v4_adv_2250k_roots_hops0_750_ratio40/server_04_bellman-classical-worker-4_hops_601_750",
    "worker5": "bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_00_bellman-classical-worker-5_hops_0_100",
    "worker6": "bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_01_bellman-classical-worker-6_hops_100_200",
    "worker7": "bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_02_bellman-classical-worker-7_hops_200_300",
    "worker8": "bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon/server_03_bellman-classical-worker-8_hops_300_400",
}


def _run(command: list[str], *, timeout: int = 7200) -> str:
    print("[setup] " + shlex.join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {shlex.join(command)}\n{result.stdout}")
    return result.stdout


def _ssh(host: str, remote: str, *, timeout: int = 7200) -> str:
    return _run(["ssh", "-o", "BatchMode=yes", host, remote], timeout=timeout)


def _rsync(source: str, destination: str, *, delete: bool = False, timeout: int = 7200) -> str:
    command = ["rsync", "-az", "--partial", "--delay-updates"]
    if delete:
        command.append("--delete")
    command.extend([source, destination])
    return _run(command, timeout=timeout)


def _inventory(cluster_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    try:
        return dict(payload["reference"]), list(payload["clusters"][cluster_name])
    except KeyError as exc:
        raise ValueError(f"unknown cluster {cluster_name!r}") from exc


def _select(hosts: list[dict[str, Any]], raw: str) -> list[dict[str, Any]]:
    if raw.strip().lower() == "all":
        return hosts
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    selected = [
        host for host in hosts
        if str(host["role"]) in wanted or str(host["alias"]) in wanted
    ]
    missing = wanted - {
        value
        for host in selected
        for value in (str(host["role"]), str(host["alias"]))
    }
    if missing:
        raise ValueError(f"unknown hosts: {sorted(missing)}")
    return selected


def _parallel(hosts: list[dict[str, Any]], action: Callable[[dict[str, Any]], str]) -> None:
    with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
        futures = {pool.submit(action, host): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            output = future.result()
            print(f"[setup] completed {host['alias']}\n{output}".rstrip(), flush=True)


def check_ssh(hosts: list[dict[str, Any]], _args: argparse.Namespace) -> None:
    _parallel(hosts, lambda host: _ssh(str(host["alias"]), "uname -m; nproc; free -g | head -2; df -BG /home/ubuntu | tail -1"))


def install_access(hosts: list[dict[str, Any]], args: argparse.Namespace) -> None:
    key = Path(args.key).expanduser().resolve()
    ssh_config = Path(args.ssh_config).expanduser().resolve()
    if not key.is_file() or not ssh_config.is_file():
        raise FileNotFoundError(f"missing key/config: {key}, {ssh_config}")

    def one(host: dict[str, Any]) -> str:
        alias = str(host["alias"])
        _ssh(alias, "mkdir -p ~/.ssh && chmod 700 ~/.ssh")
        _rsync(str(key), f"{alias}:~/.ssh/{key.name}")
        _rsync(str(ssh_config), f"{alias}:~/.ssh/config")
        return _ssh(alias, f"chmod 600 ~/.ssh/{shlex.quote(key.name)} ~/.ssh/config")

    _parallel(hosts, one)


def prepare(hosts: list[dict[str, Any]], _args: argparse.Namespace) -> None:
    command = (
        f"mkdir -p {REMOTE_REPO} /home/ubuntu/.local/share/uv "
        f"/home/ubuntu/.local/bin {REMOTE_REPO}/simulator_output "
        f"{REMOTE_REPO}/cache {REMOTE_REPO}/vidur/kv_cache"
    )
    _parallel(hosts, lambda host: _ssh(str(host["alias"]), command))


def install_system(hosts: list[dict[str, Any]], _args: argparse.Namespace) -> None:
    command = (
        "sudo apt-get update && "
        "sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "build-essential ca-certificates cmake git pkg-config "
        "python3 python3-pip python3-venv rsync"
    )
    _parallel(hosts, lambda host: _ssh(str(host["alias"]), command, timeout=7200))


def sync_code(hosts: list[dict[str, Any]], _args: argparse.Namespace) -> None:
    excludes = [
        "--exclude=.git/",
        "--exclude=.venv",
        "--exclude=.venv/",
        "--exclude=cache",
        "--exclude=cache/",
        "--exclude=simulator_output",
        "--exclude=simulator_output/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=vidur/Game_Version3_Cpp/build/",
        "--exclude=vidur/Game_Version3_Cpp/mcts_native_gv2*.so",
    ]

    def one(host: dict[str, Any]) -> str:
        alias = str(host["alias"])
        _ssh(alias, f"mkdir -p {REMOTE_REPO}")
        return _run([
            "rsync", "-az", "--partial", "--delay-updates",
            *excludes,
            str(REPO_ROOT) + "/",
            f"{alias}:{REMOTE_REPO}/",
        ])

    _parallel(hosts, one)


def _pull_command(source: str, source_path: str, target_path: str) -> str:
    return (
        f"mkdir -p {shlex.quote(target_path)} && "
        "rsync -az --delete --partial --delay-updates "
        "-e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' "
        f"{shlex.quote(source + ':' + source_path.rstrip('/') + '/')} "
        f"{shlex.quote(target_path.rstrip('/') + '/')}"
    )


def clone_runtime(hosts: list[dict[str, Any]], _args: argparse.Namespace) -> None:
    def one(host: dict[str, Any]) -> str:
        target = str(host["alias"])
        source = str(host["reference_alias"])
        _ssh(
            target,
            "for path in "
            f"{REMOTE_REPO}/.venv {REMOTE_REPO}/cache {REMOTE_REPO}/vidur/kv_cache; "
            "do if [ -L \"$path\" ]; then unlink \"$path\"; fi; done; "
            f"mkdir -p {REMOTE_REPO}/.venv {REMOTE_REPO}/cache {REMOTE_REPO}/vidur/kv_cache",
        )
        commands = [
            _pull_command(source, "/home/ubuntu/.local/share/uv", "/home/ubuntu/.local/share/uv"),
            _pull_command(source, "/home/ubuntu/.local/bin", "/home/ubuntu/.local/bin"),
            _pull_command(source, f"{REMOTE_REPO}/.venv", f"{REMOTE_REPO}/.venv"),
            _pull_command(source, f"{REMOTE_REPO}/cache", f"{REMOTE_REPO}/cache"),
            _pull_command(source, f"{REMOTE_REPO}/vidur/kv_cache", f"{REMOTE_REPO}/vidur/kv_cache"),
            (
                f"mkdir -p {REMOTE_REPO}/simulator_output && "
                "rsync -az --partial --delay-updates "
                "-e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' "
                f"{source}:{REMOTE_REPO}/simulator_output/prefill_profile.csv "
                f"{REMOTE_REPO}/simulator_output/prefill_profile.csv"
            ),
        ]
        return _ssh(target, " && ".join(commands), timeout=7200)

    _parallel(hosts, one)


def clone_datasets(hosts: list[dict[str, Any]], _args: argparse.Namespace) -> None:
    workers = [host for host in hosts if str(host["role"]).startswith("worker")]

    def one(host: dict[str, Any]) -> str:
        role = str(host["role"])
        relative = PARENT_DATASETS[role]
        source_path = f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/{relative}"
        target_path = source_path
        return _ssh(
            str(host["alias"]),
            _pull_command(str(host["reference_alias"]), source_path, target_path),
            timeout=7200,
        )

    _parallel(workers, one)


def build_native(hosts: list[dict[str, Any]], _args: argparse.Namespace) -> None:
    command = (
        f"cd {REMOTE_REPO} && "
        "rm -rf vidur/Game_Version3_Cpp/build && "
        "cmake -S vidur/Game_Version3_Cpp -B vidur/Game_Version3_Cpp/build "
        f"-DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE={REMOTE_REPO}/.venv/bin/python3 "
        f"-Dpybind11_DIR=$({REMOTE_REPO}/.venv/bin/python3 -m pybind11 --cmakedir) && "
        "cmake --build vidur/Game_Version3_Cpp/build -j$(nproc)"
    )
    _parallel(hosts, lambda host: _ssh(str(host["alias"]), command, timeout=7200))


def _deep_profile_command(expected_sha: str) -> str:
    output = "/tmp/exp3_prefill_profile_check.csv"
    log = "/tmp/exp3_prefill_profile_check.log"
    return (
        f"cd {REMOTE_REPO} && .venv/bin/python3 -m vidur.Game_Version3.prefill_calibrator "
        f"--output {output} --step 128 --max_tokens 4096 "
        "--replica_config_model_name meta-llama/Meta-Llama-3-8B "
        "--replica_config_device a100 --replica_config_network_device a100_dgx "
        "--cluster_config_num_replicas 1 --replica_config_tensor_parallel_size 1 "
        "--replica_config_num_pipeline_stages 1 "
        "--global_scheduler_config_type round_robin --replica_scheduler_config_type vllm_v1 "
        "--vllm_v1_scheduler_config_batch_size_cap 512 "
        "--execution_time_predictor_config_type random_forest "
        "--random_forest_execution_time_predictor_config_prediction_max_tokens_per_request 8192 "
        "--random_forest_execution_time_predictor_config_prediction_max_batch_size 256 "
        "--random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096 "
        f"--random_forest_execution_time_predictor_config_cache_dir {REMOTE_REPO}/cache "
        "--random_forest_execution_time_predictor_config_cache_mode require_cache "
        "--random_forest_execution_time_predictor_config_num_training_job_threads 1 "
        f"--no-snapshot_rng_state > {log} 2>&1 && "
        f"test $(sha256sum {output} | cut -d' ' -f1) = {expected_sha}"
    )


def validate(hosts: list[dict[str, Any]], args: argparse.Namespace) -> None:
    reference, _ = _inventory(args.cluster)
    expected_sha = str(reference["prefill_profile_sha256"])
    expected_python = str(reference["python_version"])
    expected_arch = str(reference["architecture"])
    expected_vcpus = int(reference["vcpu_count"])
    minimum_memory = int(reference["minimum_memory_gib"])
    minimum_disk = int(reference["minimum_root_disk_gib"])

    def one(host: dict[str, Any]) -> str:
        role = str(host["role"])
        checks = [
            f"test $(uname -m) = {shlex.quote(expected_arch)}",
            f"test $(nproc) -eq {expected_vcpus}",
            f"test $(free -g | awk '/^Mem:/ {{print $2}}') -ge {minimum_memory}",
            f"test $(df -BG /home/ubuntu | awk 'NR==2 {{gsub(/G/,\"\",$2); print $2}}') -ge {minimum_disk}",
            f"cd {REMOTE_REPO}",
            f"test $(.venv/bin/python3 -c 'import platform; print(platform.python_version())') = {expected_python}",
            ".venv/bin/python3 -c 'import joblib,numpy,sklearn,torch; print(torch.__version__, numpy.__version__, sklearn.__version__)'",
            f"test $(sha256sum simulator_output/prefill_profile.csv | cut -d' ' -f1) = {expected_sha}",
            "test -d cache -a -d vidur/kv_cache",
            "test -f vidur/Game_Version3_Cpp/mcts_native_gv2*.so",
            ".venv/bin/python3 -m vidur.bellman_v4_adv.arena_mcts_value_runnerCPP --help | grep -q -- --rollout-horizon-sec",
        ]
        if role.startswith("worker"):
            dataset = f"{REMOTE_REPO}/simulator_output/GV3_Agent/ModelSearchBed/{PARENT_DATASETS[role]}"
            checks.append(f"test -d {shlex.quote(dataset)}")
        if args.deep:
            checks.append(_deep_profile_command(expected_sha))
        checks.append("printf 'validated '; hostname; sha256sum simulator_output/prefill_profile.csv")
        return _ssh(str(host["alias"]), " && ".join(checks), timeout=7200)

    _parallel(hosts, one)


ACTIONS: dict[str, Callable[[list[dict[str, Any]], argparse.Namespace], None]] = {
    "check-ssh": check_ssh,
    "install-access": install_access,
    "prepare": prepare,
    "install-system": install_system,
    "sync-code": sync_code,
    "clone-runtime": clone_runtime,
    "clone-datasets": clone_datasets,
    "build-native": build_native,
    "validate": validate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(ACTIONS) + ("all",))
    parser.add_argument("--cluster", default="exp3")
    parser.add_argument("--hosts", default="all", help="all or comma-separated role/alias values")
    parser.add_argument("--key", default=str(DEFAULT_KEY))
    parser.add_argument("--ssh-config", default=str(DEFAULT_SSH_CONFIG))
    parser.add_argument("--deep", action="store_true", help="regenerate the prefill profile from cache")
    args = parser.parse_args()

    _reference, inventory = _inventory(args.cluster)
    hosts = _select(inventory, args.hosts)
    if args.command == "all":
        for name in (
            "check-ssh",
            "install-access",
            "prepare",
            "install-system",
            "sync-code",
            "clone-runtime",
            "clone-datasets",
            "build-native",
            "validate",
        ):
            print(f"[setup] stage={name}", flush=True)
            ACTIONS[name](hosts, args)
    else:
        ACTIONS[args.command](hosts, args)


if __name__ == "__main__":
    main()
