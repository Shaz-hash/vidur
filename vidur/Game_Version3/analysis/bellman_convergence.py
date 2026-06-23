from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import torch

from ..DNN.replay_dataset import collate_player_samples
from ..DNN.replay_write import RootSample
from ..DNN.value_models import AlphaZeroModel


GEN_RE = re.compile(r"gen_(\d+)")


@dataclass(frozen=True)
class CheckpointInfo:
    generation: int
    path: Path
    label: str = ""


@dataclass(frozen=True)
class InferenceTable:
    generation: int
    checkpoint_path: Path
    values: torch.Tensor
    label: str = ""
    state_ids: Optional[list[str]] = None


def _repo_root() -> Path:
    # bellman_convergence.py -> analysis -> Game_Version3 -> Game_Versions
    # -> mcts -> vidur(package) -> repo root.
    return Path(__file__).resolve().parents[5]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _parse_generation(path: Path) -> Optional[int]:
    match = GEN_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _discover_checkpoints(
    checkpoint_dir: Path,
    *,
    checkpoint_glob: str,
    fallback_checkpoint_glob: str,
    start_gen: Optional[int],
    end_gen: Optional[int],
) -> list[CheckpointInfo]:
    paths = sorted(checkpoint_dir.glob(checkpoint_glob))
    if not paths and fallback_checkpoint_glob:
        paths = sorted(checkpoint_dir.glob(fallback_checkpoint_glob))

    checkpoints: list[CheckpointInfo] = []
    seen_generations: set[int] = set()
    for path in paths:
        generation = _parse_generation(path)
        if generation is None:
            continue
        if start_gen is not None and generation < start_gen:
            continue
        if end_gen is not None and generation > end_gen:
            continue
        if generation in seen_generations:
            raise ValueError(
                f"Multiple checkpoints for generation {generation}; use a narrower --checkpoint-glob"
            )
        seen_generations.add(generation)
        checkpoints.append(CheckpointInfo(generation=generation, path=path, label=f"{generation:06d}"))

    checkpoints.sort(key=lambda item: item.generation)
    if len(checkpoints) < 2:
        raise ValueError(
            f"Need at least two checkpoints in {checkpoint_dir} matching {checkpoint_glob!r}"
        )
    return checkpoints


def _discover_old_current_bridge_checkpoints(
    *,
    old_checkpoints_dir: Path,
    current_checkpoints_dir: Path,
    old_start_gen: int,
    old_end_gen: int,
    current_gen: int,
    descriptive_labels: bool,
) -> list[CheckpointInfo]:
    if old_end_gen < old_start_gen:
        raise ValueError("--old-end-gen must be >= --old-start-gen")

    checkpoints: list[CheckpointInfo] = []
    sequence_generation = 0
    for generation in range(int(old_start_gen), int(old_end_gen) + 1):
        path = old_checkpoints_dir / f"selfplay_weights_gen_{generation:06d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Old checkpoint not found: {path}")
        checkpoints.append(
            CheckpointInfo(
                generation=sequence_generation,
                path=path,
                label=f"old_{generation:06d}" if descriptive_labels else f"{sequence_generation:06d}",
            )
        )
        sequence_generation += 1

    current_path = current_checkpoints_dir / f"selfplay_weights_gen_{int(current_gen):06d}.pt"
    if not current_path.exists():
        raise FileNotFoundError(f"Current checkpoint not found: {current_path}")
    checkpoints.append(
        CheckpointInfo(
            generation=sequence_generation,
            path=current_path,
            label=(
                f"current_{int(current_gen):06d}"
                if descriptive_labels
                else f"{sequence_generation:06d}"
            ),
        )
    )
    return checkpoints


def _iter_replay_shards(eval_dir: Path, *, max_shards: Optional[int]) -> list[Path]:
    shards = sorted(eval_dir.rglob("replay_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No replay_*.pt shards found under {eval_dir}")
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    return shards


def _chunked(samples: Sequence[RootSample], batch_size: int) -> Iterator[list[RootSample]]:
    for start in range(0, len(samples), batch_size):
        yield list(samples[start : start + batch_size])


def _state_id(sample: RootSample) -> str:
    return (
        f"game={sample.get('game_id')}:"
        f"root={sample.get('root_id')}:"
        f"node={sample.get('root_node_id')}:"
        f"depth={sample.get('root_depth')}:"
        f"player={sample.get('player')}"
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    if isinstance(checkpoint, dict) and checkpoint:
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise TypeError("Checkpoint does not look like a model state dict")


def _load_model(checkpoint_path: Path, *, device: torch.device) -> AlphaZeroModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)

    model = AlphaZeroModel()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if not all(key.startswith("module.") for key in state_dict.keys()):
            raise
        stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model.load_state_dict(stripped, strict=True)

    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def _infer_chunk_values(
    model: AlphaZeroModel,
    samples: Sequence[RootSample],
    *,
    device: torch.device,
) -> torch.Tensor:
    values = torch.empty((len(samples),), dtype=torch.float32)
    by_player: dict[str, list[tuple[int, RootSample]]] = {"controller": [], "adversary": []}
    for index, sample in enumerate(samples):
        player = sample.get("player")
        if player not in by_player:
            raise ValueError(f"Unknown player in sample: {player!r}")
        by_player[str(player)].append((index, sample))

    for player, indexed_samples in by_player.items():
        if not indexed_samples:
            continue
        original_indices = [item[0] for item in indexed_samples]
        player_samples = [item[1] for item in indexed_samples]
        batch = collate_player_samples(
            player_samples,
            device=device,
            include_policy_tensors=False,
            include_legacy_fallback=False,
            include_ids=False,
        )
        _, value_raw = model.forward(
            player=player,
            prefill_req_features=batch["prefill_req_features"],
            decode_req_features=batch["decode_req_features"],
            global_features=batch["global_features"],
            prefill_req_mask=batch["prefill_req_mask"],
            decode_req_mask=batch["decode_req_mask"],
            action_mask=None,
        )
        player_values = model.value_scalar_from_logits(value_raw).view(-1).detach().to("cpu", dtype=torch.float32)
        for dst_index, value in zip(original_indices, player_values):
            values[dst_index] = value

    return values


def _iter_samples_from_shards(
    shard_paths: Sequence[Path],
    *,
    skip_corrupt_shards: bool,
    max_samples: Optional[int],
) -> Iterator[RootSample]:
    emitted = 0
    for shard_path in shard_paths:
        try:
            shard = torch.load(shard_path, map_location="cpu")
        except Exception as exc:
            if skip_corrupt_shards:
                print(f"[bellman] skipping unreadable shard: {shard_path} ({exc})", file=sys.stderr)
                continue
            raise

        if not isinstance(shard, list):
            raise TypeError(f"Shard must contain a list, got {type(shard)} at {shard_path}")
        for sample in shard:
            if max_samples is not None and emitted >= int(max_samples):
                return
            emitted += 1
            yield sample


def infer_checkpoint_values(
    checkpoint: CheckpointInfo,
    shard_paths: Sequence[Path],
    *,
    device: torch.device,
    batch_size: int,
    collect_state_ids: bool,
    skip_corrupt_shards: bool,
    max_samples: Optional[int],
) -> InferenceTable:
    print(
        f"[bellman] inference starting: gen={checkpoint.label or f'{checkpoint.generation:06d}'}, "
        f"checkpoint={checkpoint.path.name}",
        flush=True,
    )
    model = _load_model(checkpoint.path, device=device)

    values: list[torch.Tensor] = []
    state_ids: Optional[list[str]] = [] if collect_state_ids else None
    pending: list[RootSample] = []
    total_samples = 0

    for sample in _iter_samples_from_shards(
        shard_paths,
        skip_corrupt_shards=skip_corrupt_shards,
        max_samples=max_samples,
    ):
        pending.append(sample)
        if len(pending) < batch_size:
            continue
        values.append(_infer_chunk_values(model, pending, device=device))
        if state_ids is not None:
            state_ids.extend(_state_id(item) for item in pending)
        total_samples += len(pending)
        pending.clear()

    if pending:
        values.append(_infer_chunk_values(model, pending, device=device))
        if state_ids is not None:
            state_ids.extend(_state_id(item) for item in pending)
        total_samples += len(pending)

    if not values:
        raise ValueError("No samples were inferred")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = torch.cat(values, dim=0)
    if out.numel() != total_samples:
        raise RuntimeError(f"Internal sample count mismatch: values={out.numel()} total={total_samples}")
    finite_mask = torch.isfinite(out)
    if not bool(finite_mask.all()):
        bad_count = int((~finite_mask).sum().item())
        first_bad = int((~finite_mask).nonzero(as_tuple=False)[0].item())
        raise RuntimeError(
            f"Non-finite value prediction(s) for gen={checkpoint.label or f'{checkpoint.generation:06d}'}: "
            f"bad_count={bad_count}, first_bad_state_index={first_bad}"
        )
    print(
        f"[bellman] inference complete: gen={checkpoint.label or f'{checkpoint.generation:06d}'}, samples={out.numel()}",
        flush=True,
    )
    return InferenceTable(
        generation=checkpoint.generation,
        checkpoint_path=checkpoint.path,
        values=out,
        label=checkpoint.label or f"{checkpoint.generation:06d}",
        state_ids=state_ids,
    )


def _write_generation_values(
    path: Path,
    *,
    generation: int,
    checkpoint_path: Path,
    state_ids: Sequence[str],
    values: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["generation", "checkpoint", "state_index", "state_id", "value"],
        )
        writer.writeheader()
        for index, (state_id, value) in enumerate(zip(state_ids, values.tolist())):
            writer.writerow(
                {
            "generation": f"{generation:06d}",
                    "checkpoint": checkpoint_path.name,
                    "state_index": index,
                    "state_id": state_id,
                    "value": f"{float(value):.10g}",
                }
            )


def _compare_tables(
    prev: InferenceTable,
    curr: InferenceTable,
    *,
    state_ids: Sequence[str],
) -> dict[str, Any]:
    if prev.values.numel() != curr.values.numel():
        raise ValueError(
            f"Sample count mismatch: gen {prev.generation} has {prev.values.numel()}, "
            f"gen {curr.generation} has {curr.values.numel()}"
        )
    diff = torch.abs(curr.values - prev.values)
    max_diff, max_index_t = torch.max(diff, dim=0)
    max_index = int(max_index_t.item())
    p95 = torch.quantile(diff, 0.95).item() if diff.numel() > 1 else float(max_diff.item())
    prev_label = prev.label or f"{prev.generation:06d}"
    curr_label = curr.label or f"{curr.generation:06d}"
    return {
        "prev_generation": prev_label,
        "curr_generation": curr_label,
        "prev_checkpoint": prev.checkpoint_path.name,
        "curr_checkpoint": curr.checkpoint_path.name,
        "num_states": int(diff.numel()),
        "max_abs_diff": f"{float(max_diff.item()):.10g}",
        "mean_abs_diff": f"{float(diff.mean().item()):.10g}",
        "p95_abs_diff": f"{float(p95):.10g}",
        "argmax_state_index": max_index,
        "argmax_state_id": state_ids[max_index],
        "prev_value": f"{float(prev.values[max_index].item()):.10g}",
        "curr_value": f"{float(curr.values[max_index].item()):.10g}",
    }


def _write_pair_metrics(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prev_generation",
        "curr_generation",
        "prev_checkpoint",
        "curr_checkpoint",
        "num_states",
        "max_abs_diff",
        "mean_abs_diff",
        "p95_abs_diff",
        "argmax_state_index",
        "argmax_state_id",
        "prev_value",
        "curr_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_pair_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Pair CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_generation_text(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Could not parse generation from {value!r}")
    return int(match.group(1))


def _offset_pair_rows(
    rows: Sequence[dict[str, Any]],
    *,
    offset: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["prev_generation"] = f"{_parse_generation_text(row['prev_generation']) + int(offset):06d}"
        new_row["curr_generation"] = f"{_parse_generation_text(row['curr_generation']) + int(offset):06d}"
        out.append(new_row)
    return out


def _plot_metric(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    ylabel: str,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[bellman] matplotlib unavailable; skipping plot ({exc})", file=sys.stderr)
        return

    x_labels = [f"{row['prev_generation']}->{row['curr_generation']}" for row in rows]
    y = [float(row[metric_key]) for row in rows]
    x = list(range(len(rows)))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10.0, len(rows) * 0.35), 5.5))
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Consecutive model pair")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=60, ha="right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pair_metrics(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    _plot_metric(
        output_dir / "bellman_convergence_max_abs_diff.png",
        rows,
        metric_key="max_abs_diff",
        ylabel="Max absolute value difference on fixed eval states",
        title="GV3 Bellman Convergence: Max Consecutive Checkpoint Value Drift",
    )


def _compute_pair_rows(
    checkpoints: Sequence[CheckpointInfo],
    shard_paths: Sequence[Path],
    *,
    device: torch.device,
    batch_size: int,
    skip_corrupt_shards: bool,
    max_samples: Optional[int],
    save_generation_values: bool,
    output_dir: Path,
    write_incremental_outputs: bool,
    no_plot: bool,
) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    prev_table: Optional[InferenceTable] = None
    state_ids: Optional[list[str]] = None

    for checkpoint in checkpoints:
        table = infer_checkpoint_values(
            checkpoint,
            shard_paths,
            device=device,
            batch_size=int(batch_size),
            collect_state_ids=state_ids is None,
            skip_corrupt_shards=bool(skip_corrupt_shards),
            max_samples=max_samples,
        )
        if state_ids is None:
            if table.state_ids is None:
                raise RuntimeError("First inference did not collect state ids")
            state_ids = table.state_ids
        if len(state_ids) != int(table.values.numel()):
            raise ValueError(
                f"State id/value count mismatch for gen {checkpoint.generation}: "
                f"ids={len(state_ids)} values={table.values.numel()}"
            )

        if save_generation_values:
            values_label = checkpoint.label or f"{checkpoint.generation:06d}"
            values_path = output_dir / "generation_values" / f"values_gen_{values_label}.csv"
            _write_generation_values(
                values_path,
                generation=checkpoint.generation,
                checkpoint_path=checkpoint.path,
                state_ids=state_ids,
                values=table.values,
            )

        if prev_table is not None:
            row = _compare_tables(prev_table, table, state_ids=state_ids)
            pair_rows.append(row)
            print(
                f"[bellman] pair {row['prev_generation']}->{row['curr_generation']}: "
                f"max_abs_diff={row['max_abs_diff']}, mean_abs_diff={row['mean_abs_diff']}",
                flush=True,
            )
            if write_incremental_outputs:
                _write_pair_metrics(output_dir / "bellman_convergence_pairs.csv", pair_rows)
                if not no_plot:
                    _plot_pair_metrics(output_dir, pair_rows)

        prev_table = table

    return pair_rows
    _plot_metric(
        output_dir / "bellman_convergence_mean_abs_diff.png",
        rows,
        metric_key="mean_abs_diff",
        ylabel="Mean absolute value difference on fixed eval states",
        title="GV3 Bellman Convergence: Mean Consecutive Checkpoint Value Drift",
    )


def run(args: argparse.Namespace) -> None:
    eval_dir = _resolve_path(args.eval_dir)
    checkpoint_dir = _resolve_path(args.checkpoints_dir)
    output_dir = _resolve_path(args.output_dir)

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval backup dir not found: {eval_dir}")
    if not args.old_current_bridge_only and not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    device = torch.device(args.device)

    if args.old_current_bridge_only:
        checkpoints = _discover_old_current_bridge_checkpoints(
            old_checkpoints_dir=_resolve_path(args.old_checkpoints_dir),
            current_checkpoints_dir=_resolve_path(args.current_checkpoints_dir),
            old_start_gen=int(args.old_start_gen),
            old_end_gen=int(args.old_end_gen),
            current_gen=int(args.current_bridge_gen),
            descriptive_labels=not bool(args.merge_existing_current_csv),
        )
    else:
        checkpoints = _discover_checkpoints(
            checkpoint_dir,
            checkpoint_glob=args.checkpoint_glob,
            fallback_checkpoint_glob=args.fallback_checkpoint_glob,
            start_gen=args.start_gen,
            end_gen=args.end_gen,
        )
    shards = _iter_replay_shards(eval_dir, max_shards=args.max_shards)

    print(
        f"[bellman] setup: checkpoints={len(checkpoints)}, shards={len(shards)}, "
        f"device={device}, batch_size={args.batch_size}, output_dir={output_dir}",
        flush=True,
    )

    pair_rows = _compute_pair_rows(
        checkpoints,
        shards,
        device=device,
        batch_size=int(args.batch_size),
        skip_corrupt_shards=bool(args.skip_corrupt_shards),
        max_samples=args.max_samples,
        save_generation_values=bool(args.save_generation_values),
        output_dir=output_dir,
        write_incremental_outputs=not bool(args.merge_existing_current_csv),
        no_plot=bool(args.no_plot),
    )

    if args.merge_existing_current_csv:
        if not args.old_current_bridge_only:
            raise ValueError("--merge-existing-current-csv requires --old-current-bridge-only")
        current_rows = _read_pair_metrics(_resolve_path(args.current_pairs_csv))
        shifted_current_rows = _offset_pair_rows(
            current_rows,
            offset=int(args.current_generation_offset),
        )
        pair_rows = pair_rows + shifted_current_rows
        _write_pair_metrics(output_dir / "bellman_convergence_pairs.csv", pair_rows)
        if not args.no_plot:
            _plot_pair_metrics(output_dir, pair_rows)
        print(
            f"[bellman] merged existing current CSV: bridge_rows={len(pair_rows) - len(shifted_current_rows)}, "
            f"shifted_current_rows={len(shifted_current_rows)}, offset={args.current_generation_offset}",
            flush=True,
        )

    if not pair_rows:
        raise RuntimeError("No pair comparisons were generated")

    print(
        f"[bellman] complete: pairs={len(pair_rows)}, "
        f"csv={output_dir / 'bellman_convergence_pairs.csv'}",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Plot Bellman-style convergence by comparing consecutive GV3 model "
            "checkpoint value predictions on a fixed eval-state backup set."
        )
    )
    parser.add_argument(
        "--eval-dir",
        default=str(repo_root / "simulator_output/Game_Version3/backups/gen_000036_eval"),
        help="Directory containing the fixed eval replay shards.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=str(repo_root / "simulator_output/Game_Version3/mcts_dnn_checkpoints"),
        help="Directory containing per-generation checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="selfplay_weights_gen_*.pt",
        help="Checkpoint glob to use inside --checkpoints-dir.",
    )
    parser.add_argument(
        "--fallback-checkpoint-glob",
        default="gen_*.pt",
        help="Fallback checkpoint glob used only if --checkpoint-glob matches nothing.",
    )
    parser.add_argument(
        "--old-current-bridge-only",
        action="store_true",
        help=(
            "Quick mode: compare old AWS selfplay checkpoints old_start..old_end, "
            "then one bridge pair old_end -> current checkpoint."
        ),
    )
    parser.add_argument(
        "--old-checkpoints-dir",
        default=str(repo_root / "simulator_output/OldGame_Version3/mcts_dnn_checkpoints"),
        help="Old AWS checkpoint directory used by --old-current-bridge-only.",
    )
    parser.add_argument(
        "--current-checkpoints-dir",
        default=str(repo_root / "simulator_output/Game_Version3/mcts_dnn_checkpoints"),
        help="Current local checkpoint directory used by --old-current-bridge-only.",
    )
    parser.add_argument(
        "--old-start-gen",
        type=int,
        default=0,
        help="First old AWS generation used by --old-current-bridge-only.",
    )
    parser.add_argument(
        "--old-end-gen",
        type=int,
        default=13,
        help="Last old AWS generation used by --old-current-bridge-only.",
    )
    parser.add_argument(
        "--current-bridge-gen",
        type=int,
        default=0,
        help="Current local generation to compare against old_end in --old-current-bridge-only.",
    )
    parser.add_argument(
        "--merge-existing-current-csv",
        action="store_true",
        help=(
            "With --old-current-bridge-only, compute old+bridge rows, then append an existing "
            "current-lane pair CSV after shifting its generation labels."
        ),
    )
    parser.add_argument(
        "--current-pairs-csv",
        default=str(repo_root / "simulator_output/Game_Version3/analysis/bellman_convergence/bellman_convergence_pairs.csv"),
        help="Existing current-lane Bellman pair CSV used by --merge-existing-current-csv.",
    )
    parser.add_argument(
        "--current-generation-offset",
        type=int,
        default=14,
        help="Generation offset for current-lane rows when using --merge-existing-current-csv.",
    )
    parser.add_argument("--start-gen", type=int, default=None, help="Optional first generation to include.")
    parser.add_argument("--end-gen", type=int, default=None, help="Optional final generation to include.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Inference batch size. CPU default is conservative for running alongside training.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Model inference device for the analysis run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "simulator_output/Game_Version3/analysis/bellman_convergence"),
        help="Directory for convergence CSV, plot, and optional per-generation value tables.",
    )
    parser.add_argument("--max-shards", type=int, default=None, help="Debug limit on number of eval shards.")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug limit on number of eval samples.")
    parser.add_argument(
        "--skip-corrupt-shards",
        action="store_true",
        help="Skip unreadable eval shards instead of failing.",
    )
    parser.add_argument(
        "--no-save-generation-values",
        dest="save_generation_values",
        action="store_false",
        help="Do not write values_gen_XXXX.csv files.",
    )
    parser.set_defaults(save_generation_values=True)
    parser.add_argument("--no-plot", action="store_true", help="Write CSV only; skip PNG plot creation.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
