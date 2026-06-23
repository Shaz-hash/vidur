#!/usr/bin/env python3
"""Build adversary action features for GV3 adversary policy/prior targets.

The feature rows are action-only. State features are already produced by the
226D parent-state feature pipeline. Training should join by
(worker, state_id, canon_action_index).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REMOTE_REPO = "/home/ubuntu/vidur-classical-search"
DEFAULT_REMOTE_OUTPUT_BASE = "simulator_output/GV3_Agent/ModelSearchBed"
DEFAULT_EXPERIMENT_NAME = "bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon"
DEFAULT_MODEL_LABEL = "worker1_200k_alpha2__hgb_sq_63leaf_1050iter_a2__v47__"
DEFAULT_REMOTE_MODEL_PATH = (
    "{remote_repo}/simulator_output/GV3_Agent/bellman_multiserver_HGB/"
    "policy_target_models/{model_label}/model.joblib"
)
DEFAULT_LOCAL_SMOKE_DIR = (
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/bellman_multiserver_HGB/"
    "bellman_adv_prior_smoke_test_effectcanon"
)
DEFAULT_SMOKE_STATE_IDS_CSV = (
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/bellman_multiserver_HGB/"
    "bellman_adv_prior_smoke_test/target_smoke_full.csv"
)

MAX_REQUESTS_PER_LAUNCH_WINDOW = 7.0
# GV3 config currently allows/templates up to 4096 prefill tokens.
MAX_PREFILL_TOKENS_PER_REQUEST = 4096.0
ACTION_FEATURE_VERSION = "adversary_action_features_effectcanon_v1"

STOP_RULE_NAMES = [
    "stop_none",
    "stop_longest_decode",
    "stop_shortest_decode",
    "stop_all_decodes_over_512",
    "stop_all_decodes_over_216",
]

ACTION_FEATURE_NAMES = [
    "a_adv_launch_count_norm",
    "a_adv_prefill_size_norm",
]
ACTION_FEATURE_NAMES.extend(f"a_adv_stop_rule__{rule}" for rule in STOP_RULE_NAMES)

FEATURE_ROW_FIELDS = [
    "worker",
    "host",
    "state_id",
    "player",
    "canon_action_index",
    "action_repr",
    "action_feature_repr",
]

SUMMARY_ROW_FIELDS = [
    "worker",
    "host",
    "state_id",
    "root_id",
    "player",
    "root_player",
    "valid_action_count",
    "canonical_action_count",
    "feature_action_count",
]


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    server_index: int
    hops_min: int
    hops_max: int

    @property
    def root_dir_name(self) -> str:
        return f"server_{self.server_index:02d}_{self.host}_hops_{self.hops_min}_{self.hops_max}"


WORKERS: tuple[WorkerSpec, ...] = (
    WorkerSpec("worker5", "bellman-classical-worker-5", 0, 0, 100),
    WorkerSpec("worker6", "bellman-classical-worker-6", 1, 100, 200),
    WorkerSpec("worker7", "bellman-classical-worker-7", 2, 200, 300),
    WorkerSpec("worker8", "bellman-classical-worker-8", 3, 300, 400),
)


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vidur" / "Game_Version3").is_dir():
            return parent
    return p.parents[3]


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _ssh(host: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, cmd], check=check)


def _rsync(src: str, dst: str) -> subprocess.CompletedProcess[str]:
    return _run(["rsync", "-az", src, dst])


def _base_path(remote_repo: str, remote_output_base: str) -> str:
    return f"{remote_repo.rstrip('/')}/{remote_output_base.strip('/')}"


def _worker_root_dir(worker: WorkerSpec, *, remote_repo: str, remote_output_base: str, experiment_name: str) -> str:
    return f"{_base_path(remote_repo, remote_output_base)}/{experiment_name}/{worker.root_dir_name}"


def _remote_model_path(args: argparse.Namespace) -> str:
    return str(args.remote_model_path).format(
        remote_repo=str(args.remote_repo).rstrip("/"),
        model_label=str(args.model_label),
    )


def _target_base(args: argparse.Namespace) -> str:
    return (
        f"{_base_path(args.remote_repo, args.remote_output_base)}/"
        f"{args.experiment_name}/adversary_prior_action_features/{ACTION_FEATURE_VERSION}"
    )


def _parse_workers(raw: str | None) -> list[WorkerSpec]:
    if not raw:
        return list(WORKERS)
    requested = {x.strip() for x in str(raw).split(",") if x.strip()}
    out = [w for w in WORKERS if w.worker_id in requested or w.host in requested]
    if not out:
        raise ValueError(f"no workers selected from {raw!r}")
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_smoke_state_ids_by_worker(path: str | None) -> dict[str, list[int]]:
    if not path:
        return {}
    out: dict[str, set[int]] = {}
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            worker = str(row.get("worker", "")).strip()
            if worker:
                out.setdefault(worker, set()).add(int(row["state_id"]))
    return {worker: sorted(ids) for worker, ids in out.items()}


def _parse_state_ids(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return sorted({int(x.strip()) for x in str(raw).replace("\n", ",").split(",") if x.strip()})


def _norm01(value: float, denom: float) -> float:
    d = float(denom)
    if d <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, float(value) / d)))


def _stop_rule_from_action_index(action_index: int, stop_rules: list[str]) -> str:
    if not stop_rules:
        return "stop_none"
    return str(stop_rules[int(action_index) % len(stop_rules)])


def _adversary_action_key(action: Any) -> tuple[Any, ...]:
    request_specs = tuple(
        sorted(
            (
                int(getattr(req, "prefill_tokens", 0)),
                int(getattr(req, "decode_tokens", 0)),
                float(getattr(req, "prefill_slo", 0.0)),
                float(getattr(req, "decode_slo", 0.0)),
            )
            for req in (getattr(action, "requests", None) or [])
        )
    )
    stop_decode_ids = tuple(sorted(set(int(x) for x in (getattr(action, "stop_decode_ids", None) or []))))
    return (request_specs, stop_decode_ids)


def _effect_canonical_indices(actions_by_index: list[Any | None], valid_indices: list[int]) -> list[int]:
    seen: dict[tuple[Any, ...], int] = {}
    canonical: list[int] = []
    for idx in valid_indices:
        action = actions_by_index[int(idx)]
        if action is None:
            continue
        key = _adversary_action_key(action)
        if key in seen:
            continue
        seen[key] = int(idx)
        canonical.append(int(idx))
    return canonical


def build_action_feature_dict(action: Any, *, canon_action_index: int, stop_rules: list[str] | None = None) -> dict[str, float]:
    stop_rules = list(stop_rules or STOP_RULE_NAMES)
    requests = list(getattr(action, "requests", None) or [])
    launch_count = len(requests)
    if requests:
        prefill_size = sum(float(getattr(req, "prefill_tokens", 0.0)) for req in requests) / float(len(requests))
    else:
        prefill_size = 0.0
    stop_rule = _stop_rule_from_action_index(int(canon_action_index), stop_rules)
    row: dict[str, float] = {
        "a_adv_launch_count_norm": _norm01(float(launch_count), MAX_REQUESTS_PER_LAUNCH_WINDOW),
        "a_adv_prefill_size_norm": _norm01(float(prefill_size), MAX_PREFILL_TOKENS_PER_REQUEST),
    }
    for rule in STOP_RULE_NAMES:
        row[f"a_adv_stop_rule__{rule}"] = 1.0 if str(rule) == str(stop_rule) else 0.0
    return {name: float(row.get(name, 0.0)) for name in ACTION_FEATURE_NAMES}


def _feature_repr(feature_dict: dict[str, float]) -> str:
    return json.dumps({k: round(float(feature_dict[k]), 10) for k in ACTION_FEATURE_NAMES}, sort_keys=True, separators=(",", ":"))


def _build_bundle(model_path: str, output_dir: str) -> Any:
    from dataclasses import replace
    import joblib
    from vidur.Game_Version3.Model_Tester.config import DEFAULT_MODEL_TESTER_CONFIG
    try:
        from vidur.bellman_v4_adv_2000k_multiprocess import runner as tester_runner
    except Exception:
        from vidur.Game_Version3.Model_Tester import runner as tester_runner

    model_path_obj = Path(model_path).expanduser()
    harness_model_path = model_path_obj
    model = joblib.load(model_path_obj)
    if not callable(getattr(model, "infer_from_inputs", None)):
        from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper
        wrapped = V4AdvHGBWrapper(model, feature_dim=226, model_tag="adversary_prior_feature_hgb")
        harness_dir = Path(output_dir).expanduser() / "_feature_harness_model"
        harness_dir.mkdir(parents=True, exist_ok=True)
        harness_model_path = harness_dir / "v4_adv_hgb_wrapper.joblib"
        joblib.dump(wrapped, harness_model_path, compress=3)

    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind="classical_joblib",
        model_checkpoint_path=str(harness_model_path),
        output_dir=str(Path(output_dir) / "_feature_harness"),
        write_arena_game_logs=False,
        write_model_action_detail_logs=False,
        environment_lang="native",
        use_virtual_env=True,
    )
    return tester_runner._build_bundle(cfg)


def _state_from_record(env: Any, record: dict[str, Any]) -> Any:
    clone_fn = getattr(env, "clone_state_from_snapshot", None)
    if callable(clone_fn):
        return clone_fn(record["simulator_snapshot"], record["stats"])
    state = env.initial_state()
    state.simulator.restore_state(record["simulator_snapshot"])
    state.stats = record["stats"].clone()
    return state


def _load_records(dataset_dir: str, *, state_ids: list[int] | None, max_states: int | None) -> list[tuple[int, dict[str, Any]]]:
    from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import load_samples
    out: list[tuple[int, dict[str, Any]]] = []
    if state_ids is not None:
        wanted = {int(x) for x in state_ids}
        max_id = max(wanted) if wanted else -1
        for sid, record in load_samples(dataset_dir, root_player_filter="adversary", parent_state_id_start=0, parent_state_id_end=max_id + 1):
            if int(sid) in wanted:
                out.append((int(sid), record))
                if len(out) == len(wanted):
                    break
        missing = wanted.difference(int(sid) for sid, _ in out)
        if missing:
            raise RuntimeError(f"missing requested state ids: {sorted(missing)[:20]}")
        return sorted(out, key=lambda x: int(x[0]))
    for sid, record in load_samples(dataset_dir, root_player_filter="adversary"):
        out.append((int(sid), record))
        if max_states is not None and len(out) >= int(max_states):
            break
    return out


def _iter_records(dataset_dir: str, *, state_ids: list[int] | None, max_states: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import load_samples
    if state_ids is not None:
        yield from _load_records(dataset_dir, state_ids=state_ids, max_states=max_states)
        return
    count = 0
    for sid, record in load_samples(dataset_dir, root_player_filter="adversary"):
        yield int(sid), record
        count += 1
        if max_states is not None and count >= int(max_states):
            break


def write_features_for_records(*, worker_id: str, host: str, dataset_dir: str, output_dir: str, model_path: str, state_ids: list[int] | None = None, max_states: int | None = None, progress_every: int = 10_000) -> dict[str, Any]:
    try:
        from vidur.bellman_v4_adv_2000k_multiprocess import runner as tester_runner
    except Exception:
        from vidur.Game_Version3.Model_Tester import runner as tester_runner

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle(model_path, str(out))
    stop_rules = list(getattr(bundle.pipeline_cfg.game_v2.adversary_action, "stop_rule_names", STOP_RULE_NAMES))
    feature_path = out / "target_smoke_feature.csv"
    summary_path = out / "target_canon_feature_full.csv"
    num_states = 0
    num_feature_rows = 0
    mismatch_count = 0
    first_mismatch: dict[str, Any] | None = None
    t0 = time.time()

    with feature_path.open("w", encoding="utf-8", newline="") as ff, summary_path.open("w", encoding="utf-8", newline="") as sf:
        feature_writer = csv.DictWriter(ff, fieldnames=FEATURE_ROW_FIELDS, extrasaction="ignore")
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_ROW_FIELDS, extrasaction="ignore")
        feature_writer.writeheader()
        summary_writer.writeheader()

        for state_id, record in _iter_records(dataset_dir, state_ids=state_ids, max_states=max_states):
            before = num_feature_rows
            root_player = str(record.get("root_player", "adversary"))
            state = _state_from_record(bundle.env, record)
            player, expanded = tester_runner._align_player_to_valid_actions(
                bundle=bundle,
                state=state,
                player=root_player,
                pending_adv_pre_ctrl_snapshot=record.get("pre_controller_snapshot"),
                pending_adv_pre_ctrl_stats=record.get("pre_controller_stats"),
            )
            valid_indices: list[int] = []
            canonical_indices: list[int] = []
            if str(player) == "adversary":
                valid_indices = [int(x) for x in list(expanded.valid_indices)]
                canonical_indices = _effect_canonical_indices(list(expanded.actions_by_index), valid_indices)
                for canon_idx in canonical_indices:
                    action = expanded.actions_by_index[int(canon_idx)]
                    if action is None:
                        continue
                    fd = build_action_feature_dict(action, canon_action_index=int(canon_idx), stop_rules=stop_rules)
                    feature_writer.writerow({
                        "worker": worker_id,
                        "host": host,
                        "state_id": int(state_id),
                        "player": str(player),
                        "canon_action_index": int(canon_idx),
                        "action_repr": repr(action),
                        "action_feature_repr": _feature_repr(fd),
                    })
                    num_feature_rows += 1

            summary_row = {
                "worker": worker_id,
                "host": host,
                "state_id": int(state_id),
                "root_id": int(record.get("root_id", state_id) or state_id),
                "player": str(player),
                "root_player": root_player,
                "valid_action_count": int(len(valid_indices)),
                "canonical_action_count": int(len(canonical_indices)),
                "feature_action_count": int(num_feature_rows - before),
            }
            summary_writer.writerow(summary_row)
            num_states += 1
            if int(summary_row["canonical_action_count"]) != int(summary_row["feature_action_count"]):
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = dict(summary_row)

            try:
                bundle.mcts.clear_search_state(drop_scratch=True)
            except Exception:
                pass
            if progress_every > 0 and num_states % int(progress_every) == 0:
                ff.flush()
                sf.flush()
                print(json.dumps({"states": num_states, "feature_rows": num_feature_rows, "elapsed_s": round(time.time() - t0, 3)}, sort_keys=True), flush=True)
                gc.collect()

    if mismatch_count:
        raise RuntimeError(f"feature_action_count mismatch in {mismatch_count} states; first={first_mismatch}")

    return {
        "worker": str(worker_id),
        "host": str(host),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out),
        "num_states": int(num_states),
        "num_feature_rows": int(num_feature_rows),
        "feature_dim": len(ACTION_FEATURE_NAMES),
        "feature_names": ACTION_FEATURE_NAMES,
        "feature_version": ACTION_FEATURE_VERSION,
        "stop_rule_names": stop_rules,
        "max_requests_per_launch_window": MAX_REQUESTS_PER_LAUNCH_WINDOW,
        "max_prefill_tokens_per_request": MAX_PREFILL_TOKENS_PER_REQUEST,
        "elapsed_s": float(time.time() - t0),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def cmd_run_local(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).expanduser()
    manifest = write_features_for_records(
        worker_id=str(args.worker_id),
        host=str(args.host),
        dataset_dir=str(args.dataset_dir),
        output_dir=str(out),
        model_path=str(args.model_path),
        state_ids=_parse_state_ids(args.state_ids),
        max_states=args.max_states,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": manifest["num_feature_rows"], "states": manifest["num_states"], "output_dir": str(out)}, sort_keys=True), flush=True)


def _combine_worker_outputs(local_dir: Path, worker_dirs: list[tuple[WorkerSpec, Path]]) -> dict[str, int]:
    feature_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _worker, path in worker_dirs:
        fpath = path / "target_smoke_feature.csv"
        spath = path / "target_canon_feature_full.csv"
        if fpath.exists():
            with fpath.open("r", encoding="utf-8", newline="") as f:
                feature_rows.extend(dict(row) for row in csv.DictReader(f))
        if spath.exists():
            with spath.open("r", encoding="utf-8", newline="") as f:
                summary_rows.extend(dict(row) for row in csv.DictReader(f))
    feature_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"]), int(r["canon_action_index"])))
    summary_rows.sort(key=lambda r: (str(r.get("worker", "")), int(r["state_id"])))
    _write_csv(local_dir / "target_smoke_feature.csv", FEATURE_ROW_FIELDS, feature_rows)
    _write_csv(local_dir / "target_canon_feature_full.csv", SUMMARY_ROW_FIELDS, summary_rows)
    summary = {"feature_rows": len(feature_rows), "summary_rows": len(summary_rows)}
    (local_dir / "target_feature_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def cmd_smoke(args: argparse.Namespace) -> None:
    local_dir = Path(args.local_smoke_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    remote_model_path = _remote_model_path(args)
    state_ids_by_worker = _load_smoke_state_ids_by_worker(args.smoke_state_ids_csv)
    workers = _parse_workers(args.workers)
    for worker in workers:
        remote_script = f"{args.remote_repo.rstrip('/')}/{script_path.relative_to(_repo_root())}"
        _ssh(worker.host, f"mkdir -p {shlex.quote(str(Path(remote_script).parent))} {shlex.quote(str(Path(remote_model_path).parent))}")
        _rsync(str(script_path), f"{worker.host}:{remote_script}")
    procs: list[tuple[WorkerSpec, subprocess.Popen[str], str]] = []
    target_base = _target_base(args)
    for worker in workers:
        ids = state_ids_by_worker.get(worker.worker_id)
        if not ids:
            raise RuntimeError(f"no smoke state ids for {worker.worker_id} in {args.smoke_state_ids_csv}")
        dataset_dir = _worker_root_dir(worker, remote_repo=args.remote_repo, remote_output_base=args.remote_output_base, experiment_name=args.experiment_name)
        out_dir = f"{target_base}/smoke_{worker.worker_id}"
        cmd = (
            f"cd {shlex.quote(args.remote_repo)} && "
            f"{shlex.quote(args.remote_repo.rstrip('/') + '/.venv/bin/python3')} "
            f"-m vidur.bellman_v4_adv_2000k_multiprocess.multi_server_hgb_adversary_prior_feature_building run-local "
            f"--worker-id {shlex.quote(worker.worker_id)} --host {shlex.quote(worker.host)} "
            f"--dataset-dir {shlex.quote(dataset_dir)} --output-dir {shlex.quote(out_dir)} "
            f"--model-path {shlex.quote(remote_model_path)} --state-ids {shlex.quote(','.join(str(x) for x in ids))}"
        )
        print(f"[launch] {worker.host}: {cmd}", flush=True)
        procs.append((worker, subprocess.Popen(["ssh", worker.host, cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT), out_dir))
    failed: list[str] = []
    for worker, proc, _out_dir in procs:
        stdout, _ = proc.communicate()
        (local_dir / f"{worker.worker_id}_feature_remote_stdout.log").write_text(stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            failed.append(f"{worker.worker_id}:{proc.returncode}")
            print(stdout or "", flush=True)
    if failed:
        raise RuntimeError(f"remote smoke feature generation failed: {failed}")
    pulled: list[tuple[WorkerSpec, Path]] = []
    for worker, _proc, out_dir in procs:
        worker_local = local_dir / f"adv_feature_{worker.worker_id}"
        worker_local.mkdir(parents=True, exist_ok=True)
        _rsync(f"{worker.host}:{out_dir.rstrip('/')}/", f"{worker_local}/")
        pulled.append((worker, worker_local))
    print(json.dumps(_combine_worker_outputs(local_dir, pulled), indent=2, sort_keys=True), flush=True)


def cmd_status(args: argparse.Namespace) -> None:
    for worker in _parse_workers(args.workers):
        out_dir = f"{_target_base(args)}/smoke_{worker.worker_id}"
        res = _ssh(worker.host, f"cat {shlex.quote(out_dir + '/manifest.json')} 2>/dev/null || echo missing", check=False)
        print(f"--- {worker.worker_id} {worker.host} ---")
        print(res.stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-local")
    run.add_argument("--worker-id", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--dataset-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--state-ids", default=None)
    run.add_argument("--max-states", type=int, default=None)
    run.set_defaults(func=cmd_run_local)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    smoke.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    smoke.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    smoke.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    smoke.add_argument("--remote-model-path", default=DEFAULT_REMOTE_MODEL_PATH)
    smoke.add_argument("--local-smoke-dir", default=DEFAULT_LOCAL_SMOKE_DIR)
    smoke.add_argument("--smoke-state-ids-csv", default=DEFAULT_SMOKE_STATE_IDS_CSV)
    smoke.add_argument("--workers", default="worker5,worker6,worker7,worker8")
    smoke.set_defaults(func=cmd_smoke)
    status = sub.add_parser("status")
    status.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    status.add_argument("--remote-output-base", default=DEFAULT_REMOTE_OUTPUT_BASE)
    status.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    status.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    status.add_argument("--remote-model-path", default=DEFAULT_REMOTE_MODEL_PATH)
    status.add_argument("--workers", default="worker5,worker6,worker7,worker8")
    status.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
