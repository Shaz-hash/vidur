"""Assemble distributed parent/child feature datasets on every worker.

The parent and child 226D features are produced per worker shard.  This
coordinator makes each worker see the complete dataset by placing all feature
shards under a canonical assembly directory on every host.

The script intentionally does not concatenate the large child arrays.  It
copies each source shard as a directory and writes an assembly manifest.  That
keeps memory flat, avoids rewriting a 100GB+ monolith, and still gives training
code a complete local view of the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any


DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_2250k_roots_hops0_750_ratio40"
DEFAULT_HOSTS = [
    "bellman-classical",
    "bellman-classical-worker-1",
    "bellman-classical-worker-2",
    "bellman-classical-worker-3",
    "bellman-classical-worker-4",
]

PARENT_REQUIRED_FILES = (
    "parent_features.npy",
    "parent_targets.npy",
    "parent_features.meta.json",
)
CHILD_REQUIRED_FILES = (
    "child_features.npy",
    "child_meta.npy",
    "parent_index.npz",
    "child_features.names.json",
    "child_features.summary.json",
)


@dataclass(frozen=True)
class ShardSpec:
    job_id: str
    host: str
    suffix: str
    root_dataset_dir: str
    parent_feature_dir: str
    child_feature_dir: str


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
    stdout: int | None = subprocess.PIPE,
) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        input=input_text,
        stdout=stdout,
        stderr=subprocess.STDOUT,
    )


def _ssh(host: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, cmd], check=check)


def _ssh_python(
    host: str,
    remote_repo: str,
    script: str,
    payload: dict[str, Any],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    py = f"{remote_repo.rstrip('/')}/.venv/bin/python3"
    remote = f"{shlex.quote(py)} - {shlex.quote(json.dumps(payload))}"
    return _run(["ssh", host, remote], check=check, input_text=script)


def parse_hosts(raw: str | None) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else list(DEFAULT_HOSTS)


def split_hop_intervals(hop_min: int, hop_max: int, parts: int) -> list[tuple[int, int]]:
    total = hop_max - hop_min + 1
    out: list[tuple[int, int]] = []
    cur = hop_min
    for i in range(parts):
        width = total // parts + (1 if i < total % parts else 0)
        out.append((cur, cur + width - 1))
        cur += width
    return out


def build_shards(args: argparse.Namespace) -> list[ShardSpec]:
    hosts = parse_hosts(args.hosts)
    base = f"{str(args.remote_repo).rstrip('/')}/{str(args.remote_output_base).strip('/')}"
    shards: list[ShardSpec] = []
    intervals = split_hop_intervals(args.hop_min, args.hop_max, len(hosts))
    for idx, (host, (hmin, hmax)) in enumerate(zip(hosts, intervals)):
        job_id = f"server_{idx:02d}"
        suffix = f"{job_id}_{host}_hops_{hmin}_{hmax}"
        root_dir = f"{base}/{args.experiment_name}/{suffix}"
        shards.append(
            ShardSpec(
                job_id=job_id,
                host=host,
                suffix=suffix,
                root_dataset_dir=root_dir,
                parent_feature_dir=f"{root_dir}_parent_features_adv",
                child_feature_dir=f"{root_dir}_child_transitions_adv_features_adv",
            )
        )
    return shards


def assembly_base(args: argparse.Namespace) -> str:
    base = f"{str(args.remote_repo).rstrip('/')}/{str(args.remote_output_base).strip('/')}"
    return str(args.assembly_dir).format(
        base=base,
        experiment_name=args.experiment_name,
        remote_repo=str(args.remote_repo).rstrip("/"),
    )


def assembly_shard_dir(args: argparse.Namespace, source: ShardSpec, dataset: str) -> str:
    root = assembly_base(args)
    if dataset == "parent":
        return f"{root}/parent_shards/{PurePosixPath(source.parent_feature_dir).name}"
    if dataset == "child":
        return f"{root}/child_shards/{PurePosixPath(source.child_feature_dir).name}"
    raise ValueError(f"unknown dataset: {dataset}")


def datasets_from_arg(raw: str) -> tuple[str, ...]:
    if raw == "both":
        return ("parent", "child")
    return (raw,)


INSPECT_SCRIPT = r"""
import json
import os
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])

def file_info(root: Path, names):
    out = {}
    for name in names:
        p = root / name
        out[name] = {
            "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else None,
            "is_symlink": p.is_symlink(),
        }
    return out

def npy_shape(path: Path):
    if not path.exists():
        return None
    try:
        import numpy as np
        return list(np.load(path, mmap_mode="r").shape)
    except Exception as exc:
        return {"error": str(exc)}

def inspect_parent(path: str):
    root = Path(path)
    required = ("parent_features.npy", "parent_targets.npy", "parent_features.meta.json")
    info = {
        "path": str(root),
        "exists": root.exists(),
        "is_symlink": root.is_symlink(),
        "files": file_info(root, required),
        "valid": False,
    }
    if not root.exists():
        return info
    try:
        meta_path = root / "parent_features.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        info["rows"] = int(meta.get("num_records", -1))
        info["dim"] = int(meta.get("feature_dim", -1))
        info["feature_shape"] = npy_shape(root / "parent_features.npy")
        info["target_shape"] = npy_shape(root / "parent_targets.npy")
        info["valid"] = (
            all(v["exists"] and int(v["bytes"] or 0) > 0 for v in info["files"].values())
            and info["dim"] == 226
            and isinstance(info["feature_shape"], list)
            and isinstance(info["target_shape"], list)
            and info["feature_shape"] == [info["rows"], info["dim"]]
            and info["target_shape"] == [info["rows"]]
        )
    except Exception as exc:
        info["error"] = str(exc)
    return info

def inspect_child(path: str):
    root = Path(path)
    required = (
        "child_features.npy",
        "child_meta.npy",
        "parent_index.npz",
        "child_features.names.json",
        "child_features.summary.json",
    )
    info = {
        "path": str(root),
        "exists": root.exists(),
        "is_symlink": root.is_symlink(),
        "files": file_info(root, required),
        "valid": False,
    }
    if not root.exists():
        return info
    try:
        summary_path = root / "child_features.summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        info["rows"] = int(summary.get("num_rows", -1))
        info["parents"] = int(summary.get("num_parents", -1))
        info["dim"] = int(summary.get("feature_dim", -1))
        meta_cols = summary.get("meta_columns") or []
        info["meta_dim"] = len(meta_cols)
        info["feature_shape"] = npy_shape(root / "child_features.npy")
        info["meta_shape"] = npy_shape(root / "child_meta.npy")
        info["valid"] = (
            all(v["exists"] and int(v["bytes"] or 0) > 0 for v in info["files"].values())
            and info["dim"] == 226
            and isinstance(info["feature_shape"], list)
            and isinstance(info["meta_shape"], list)
            and info["feature_shape"] == [info["rows"], info["dim"]]
            and info["meta_shape"] == [info["rows"], info["meta_dim"]]
        )
    except Exception as exc:
        info["error"] = str(exc)
    return info

out = {}
for item in payload.get("items", []):
    dataset = item["dataset"]
    path = item["path"]
    key = item.get("key") or path
    if dataset == "parent":
        out[key] = inspect_parent(path)
    elif dataset == "child":
        out[key] = inspect_child(path)
    else:
        out[key] = {"path": path, "valid": False, "error": "unknown dataset"}
print(json.dumps(out, sort_keys=True))
"""


def inspect_paths(
    host: str,
    remote_repo: str,
    items: list[dict[str, str]],
    *,
    check: bool = True,
) -> dict[str, Any]:
    res = _ssh_python(host, remote_repo, INSPECT_SCRIPT, {"items": items}, check=check)
    if not res.stdout.strip():
        return {}
    return json.loads(res.stdout)


def gather_source_manifest(args: argparse.Namespace, shards: list[ShardSpec]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "experiment_name": args.experiment_name,
        "remote_repo": args.remote_repo,
        "assembly_base": assembly_base(args),
        "parent": [],
        "child": [],
    }
    for shard in shards:
        items = [
            {"key": "parent", "dataset": "parent", "path": shard.parent_feature_dir},
            {"key": "child", "dataset": "child", "path": shard.child_feature_dir},
        ]
        inspected = inspect_paths(shard.host, args.remote_repo, items)
        parent = inspected.get("parent", {})
        child = inspected.get("child", {})
        for dataset, info, src_dir in (
            ("parent", parent, shard.parent_feature_dir),
            ("child", child, shard.child_feature_dir),
        ):
            if not info.get("valid"):
                raise RuntimeError(
                    f"{shard.host}:{src_dir} is not a valid {dataset} feature shard: {info}"
                )
            manifest[dataset].append(
                {
                    "job_id": shard.job_id,
                    "source_host": shard.host,
                    "suffix": shard.suffix,
                    "source_dir": src_dir,
                    "rows": info.get("rows"),
                    "parents": info.get("parents"),
                    "dim": info.get("dim"),
                    "files": info.get("files"),
                    "feature_shape": info.get("feature_shape"),
                    "target_shape": info.get("target_shape"),
                    "meta_shape": info.get("meta_shape"),
                }
            )
    return manifest


def expected_items_for_target(
    args: argparse.Namespace,
    shards: list[ShardSpec],
    datasets: tuple[str, ...],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for shard in shards:
        for dataset in datasets:
            items.append(
                {
                    "key": f"{dataset}:{shard.job_id}",
                    "dataset": dataset,
                    "path": assembly_shard_dir(args, shard, dataset),
                }
            )
    return items


def _expected_source_entry(manifest: dict[str, Any], dataset: str, job_id: str) -> dict[str, Any]:
    for entry in manifest[dataset]:
        if entry["job_id"] == job_id:
            return entry
    raise KeyError((dataset, job_id))


def _same_file_sizes(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual_files = actual.get("files") or {}
    expected_files = expected.get("files") or {}
    for name, exp in expected_files.items():
        got = actual_files.get(name) or {}
        if not got.get("exists"):
            return False
        if int(got.get("bytes") or -1) != int(exp.get("bytes") or -2):
            return False
    return True


def status_for_target(
    args: argparse.Namespace,
    target_host: str,
    shards: list[ShardSpec],
    manifest: dict[str, Any],
    datasets: tuple[str, ...],
) -> list[dict[str, Any]]:
    inspected = inspect_paths(
        target_host,
        args.remote_repo,
        expected_items_for_target(args, shards, datasets),
        check=False,
    )
    rows: list[dict[str, Any]] = []
    for shard in shards:
        for dataset in datasets:
            key = f"{dataset}:{shard.job_id}"
            actual = inspected.get(key, {})
            expected = _expected_source_entry(manifest, dataset, shard.job_id)
            complete = bool(actual.get("valid")) and _same_file_sizes(actual, expected)
            rows.append(
                {
                    "target_host": target_host,
                    "dataset": dataset,
                    "job_id": shard.job_id,
                    "source_host": shard.host,
                    "dest_dir": assembly_shard_dir(args, shard, dataset),
                    "complete": complete,
                    "valid_shape": bool(actual.get("valid")),
                    "rows": actual.get("rows", ""),
                    "expected_rows": expected.get("rows", ""),
                    "dim": actual.get("dim", ""),
                    "is_symlink": actual.get("is_symlink", False),
                    "error": actual.get("error", ""),
                }
            )
    return rows


def print_status(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target_host",
        "dataset",
        "job_id",
        "source_host",
        "complete",
        "valid_shape",
        "rows",
        "expected_rows",
        "dim",
        "is_symlink",
        "dest_dir",
        "error",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    print("", flush=True)
    by_host: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_host.setdefault(str(row["target_host"]), []).append(row)
    for host, host_rows in by_host.items():
        done = sum(1 for r in host_rows if r["complete"])
        print(f"{host}: {done}/{len(host_rows)} complete")


def remote_rm_and_mkdir_tmp(host: str, remote_repo: str, tmp_dir: str) -> None:
    parent = str(PurePosixPath(tmp_dir).parent)
    _ssh(
        host,
        "set -e; "
        f"mkdir -p {shlex.quote(parent)}; "
        f"rm -rf {shlex.quote(tmp_dir)}; "
        f"mkdir -p {shlex.quote(tmp_dir)}",
    )


def remote_promote_tmp(host: str, dest_dir: str, tmp_dir: str) -> None:
    _ssh(
        host,
        "set -e; "
        f"rm -rf {shlex.quote(dest_dir)}; "
        f"mv {shlex.quote(tmp_dir)} {shlex.quote(dest_dir)}",
    )


def make_local_symlink(host: str, source_dir: str, dest_dir: str) -> None:
    parent = str(PurePosixPath(dest_dir).parent)
    _ssh(
        host,
        "set -e; "
        f"mkdir -p {shlex.quote(parent)}; "
        f"rm -rf {shlex.quote(dest_dir)}; "
        f"ln -s {shlex.quote(source_dir)} {shlex.quote(dest_dir)}",
    )


def stream_copy_dir(source_host: str, source_dir: str, target_host: str, tmp_dir: str) -> None:
    remote_rm_and_mkdir_tmp(target_host, DEFAULT_REMOTE_REPO, tmp_dir)
    src_cmd = ["ssh", source_host, f"tar -C {shlex.quote(source_dir)} -cf - ."]
    dst_cmd = ["ssh", target_host, f"tar -C {shlex.quote(tmp_dir)} -xf -"]
    print(
        "[copy] "
        + " ".join(shlex.quote(x) for x in src_cmd)
        + " | "
        + " ".join(shlex.quote(x) for x in dst_cmd),
        flush=True,
    )
    src = subprocess.Popen(src_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert src.stdout is not None
    dst = subprocess.Popen(dst_cmd, stdin=src.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    src.stdout.close()
    dst_out, dst_err = dst.communicate()
    src_err = src.stderr.read() if src.stderr is not None else b""
    src_rc = src.wait()
    if src_rc != 0 or dst.returncode != 0:
        raise RuntimeError(
            "stream copy failed "
            f"source_rc={src_rc} target_rc={dst.returncode}\n"
            f"source_err={src_err.decode(errors='replace')}\n"
            f"target_out={dst_out.decode(errors='replace')}\n"
            f"target_err={dst_err.decode(errors='replace')}"
        )


def write_remote_manifest(
    args: argparse.Namespace,
    target_host: str,
    shards: list[ShardSpec],
    manifest: dict[str, Any],
    datasets: tuple[str, ...],
) -> None:
    local_manifest = json.loads(json.dumps(manifest))
    for dataset in ("parent", "child"):
        for entry in local_manifest[dataset]:
            if dataset not in datasets:
                continue
            source = next(s for s in shards if s.job_id == entry["job_id"])
            entry["local_dir"] = assembly_shard_dir(args, source, dataset)
    local_manifest["target_host"] = target_host
    local_manifest["datasets_present"] = list(datasets)
    payload = shlex.quote(json.dumps(local_manifest, indent=2, sort_keys=True))
    out = assembly_base(args)
    _ssh(
        target_host,
        "set -e; "
        f"mkdir -p {shlex.quote(out)}; "
        f"printf '%s\\n' {payload} > {shlex.quote(out + '/assembly_manifest.json')}",
    )


def sync_target(
    args: argparse.Namespace,
    target_host: str,
    shards: list[ShardSpec],
    manifest: dict[str, Any],
    datasets: tuple[str, ...],
) -> None:
    before_rows = status_for_target(args, target_host, shards, manifest, datasets)
    by_key = {(r["dataset"], r["job_id"]): r for r in before_rows}
    for shard in shards:
        for dataset in datasets:
            row = by_key[(dataset, shard.job_id)]
            if row["complete"] and not args.force:
                print(
                    f"[skip] {target_host} already has {dataset} {shard.job_id} "
                    f"at {row['dest_dir']}",
                    flush=True,
                )
                continue
            dest_dir = assembly_shard_dir(args, shard, dataset)
            source_dir = shard.parent_feature_dir if dataset == "parent" else shard.child_feature_dir
            if args.dry_run:
                action = "link" if shard.host == target_host else "stream-copy"
                print(
                    f"[dry-run] would {action} {dataset} "
                    f"{shard.host}:{source_dir} -> {target_host}:{dest_dir}"
                )
                continue
            if shard.host == target_host:
                print(f"[link] {target_host}:{dest_dir} -> {source_dir}", flush=True)
                make_local_symlink(target_host, source_dir, dest_dir)
            else:
                tmp_dir = f"{dest_dir}.tmp.{os.getpid()}"
                print(f"[sync] {dataset} {shard.host}:{source_dir} -> {target_host}:{dest_dir}", flush=True)
                stream_copy_dir(shard.host, source_dir, target_host, tmp_dir)
                expected = _expected_source_entry(manifest, dataset, shard.job_id)
                inspected = inspect_paths(
                    target_host,
                    args.remote_repo,
                    [{"key": "tmp", "dataset": dataset, "path": tmp_dir}],
                ).get("tmp", {})
                if not inspected.get("valid") or not _same_file_sizes(inspected, expected):
                    raise RuntimeError(
                        f"copied shard failed verification on {target_host}:{tmp_dir}: {inspected}"
                    )
                remote_promote_tmp(target_host, dest_dir, tmp_dir)
    if args.dry_run:
        return
    write_remote_manifest(args, target_host, shards, manifest, datasets)
    after_rows = status_for_target(args, target_host, shards, manifest, datasets)
    incomplete = [r for r in after_rows if not r["complete"]]
    if incomplete:
        raise RuntimeError(f"{target_host} still has incomplete assembled shards: {incomplete[:3]}")


def command_manifest(args: argparse.Namespace) -> None:
    shards = build_shards(args)
    manifest = gather_source_manifest(args, shards)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")


def command_status(args: argparse.Namespace) -> None:
    shards = build_shards(args)
    manifest = gather_source_manifest(args, shards)
    target_hosts = parse_hosts(args.target_hosts) if args.target_hosts else parse_hosts(args.hosts)
    datasets = datasets_from_arg(args.dataset)
    rows: list[dict[str, Any]] = []
    for host in target_hosts:
        rows.extend(status_for_target(args, host, shards, manifest, datasets))
    print_status(rows)


def command_sync(args: argparse.Namespace) -> None:
    shards = build_shards(args)
    manifest = gather_source_manifest(args, shards)
    target_hosts = parse_hosts(args.target_hosts) if args.target_hosts else parse_hosts(args.hosts)
    datasets = datasets_from_arg(args.dataset)
    print(
        f"[sync] targets={target_hosts} datasets={datasets} assembly_base={assembly_base(args)}",
        flush=True,
    )
    for host in target_hosts:
        sync_target(args, host, shards, manifest, datasets)
        print(f"[sync] completed target={host}", flush=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hosts", default=None, help="Comma-separated source worker hosts.")
    parser.add_argument("--target-hosts", default=None, help="Comma-separated target hosts. Defaults to --hosts.")
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--hop-min", type=int, default=0)
    parser.add_argument("--hop-max", type=int, default=750)
    parser.add_argument(
        "--assembly-dir",
        default="{base}/{experiment_name}/assembled_parent_child_features_adv",
        help="Remote assembly directory template.",
    )
    parser.add_argument("--dataset", choices=("parent", "child", "both"), default="both")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest", help="Inspect source shards and emit the global manifest.")
    add_common_args(p)
    p.add_argument("--out", default=None)

    p = sub.add_parser("status", help="Check whether every target host has every assembled shard.")
    add_common_args(p)

    p = sub.add_parser("sync", help="Copy/link missing or incomplete shards to every target host.")
    add_common_args(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Rebuild assembled shard dirs even if complete.")

    args = parser.parse_args()
    if args.cmd == "manifest":
        command_manifest(args)
    elif args.cmd == "status":
        command_status(args)
    elif args.cmd == "sync":
        command_sync(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
