"""Continue ALM-64MB-4C through three TinyStories epochs and probe each epoch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.causal_attribution import SelfCausalAttribution
from manifoldflow.dendritic_basin.native_cortex import (
    NativeGrowableCortex,
    native_available,
)
from manifoldflow.dendritic_basin.native_cortex_import import native_cortex_from_python
from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.dendritic_basin.temporal_difference_credit import (
    LocalValueEstimate,
)
from manifoldflow.experiments.dendritic_basin_phase122_full_corpus_4c import (
    DEFAULT_TINY_TRAIN,
    DEFAULT_TINY_VALID,
    _atomic_json,
    _atomic_pickle,
    _evaluate,
    _generation_panel,
    _partition,
)


def _load_any_cortex(path: str | Path) -> object:
    """Load either backend's checkpoint: whichever type was pickled comes
    back out, since NativeGrowableCortex is pickle-transparent via
    __reduce__. This is the native-aware sibling of phase122's own
    _load_cortex, which intentionally stays strict (Python-only) for that
    script's own use.
    """
    with Path(path).open("rb") as handle:
        cortex = pickle.load(handle)["learner"]
    if isinstance(cortex, SequenceCortexOpponentLearner):
        cortex.reset_episode()
        return cortex
    if native_available() and isinstance(cortex, NativeGrowableCortex):
        cortex.reset_episode()
        return cortex
    raise TypeError("column checkpoint does not contain a supported cortex")


def _apply_competition_settings(
    learner: object,
    *,
    competition_exponent: float,
    adaptive_competition: bool,
    trust_gamma: float,
    trust_learning_rate: float,
    attribution_gate_threshold: float,
    connectivity_pruning: bool = False,
) -> None:
    """Patch a possibly pre-Phase-153 checkpoint with the new equation.

    Older checkpoints were pickled before ``competition_exponent`` and
    ``adaptive_competition`` existed on ``ContextualOpponentAttractorLearner``,
    so the deserialized object simply lacks those attributes (pickle restores
    exact prior state, it does not re-run ``__init__``). This backfills them
    once, right after loading, so a continued run can turn the mechanism on
    for a lineage that started before it existed.
    """
    learner.competition_exponent = competition_exponent
    learner.adaptive_competition = adaptive_competition
    learner.trust_gamma = trust_gamma
    learner.attribution_gate_threshold = attribution_gate_threshold
    learner.connectivity_pruning = connectivity_pruning
    if not hasattr(learner, "context_trust"):
        learner.context_trust = {}
        learner.context_value = LocalValueEstimate(learning_rate=trust_learning_rate)
        learner.context_attribution = SelfCausalAttribution()
        learner._pending_context_credit = None


def _develop(
    arguments: tuple[int, int, int, str, str, int, int, int, str],
) -> dict[str, object]:
    (
        epoch,
        generation,
        column,
        ancestor_path,
        data_path,
        offset,
        length,
        chunk_bytes,
        output_path,
    ) = arguments
    output = Path(output_path)
    progress = output.with_suffix(".progress.json")
    learner = _load_any_cortex(ancestor_path)
    learner.begin_lineage_tracking()
    births_before = learner.sequence_births
    evictions_before = learner.sequence_nursery_evictions
    processed = 0
    scored = 0
    bit_sum = 0.0
    started = time.perf_counter()
    with Path(data_path).open("rb") as handle:
        handle.seek(offset)
        while processed < length:
            raw = handle.read(min(chunk_bytes, length - processed))
            if not raw:
                raise ValueError("TinyStories ended inside an assigned territory")
            probability = learner.observe_chunk_probabilities(
                np.frombuffer(raw, dtype=np.uint8), learn=True
            )
            finite = probability[np.isfinite(probability)]
            bit_sum += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
            scored += len(finite)
            processed += len(raw)
            elapsed = time.perf_counter() - started
            _atomic_json(
                progress,
                {
                    "status": "training",
                    "pid": os.getpid(),
                    "epoch": epoch,
                    "generation": generation,
                    "column": column,
                    "processed_bytes": processed,
                    "bytes": length,
                    "bytes_per_second": processed / max(elapsed, 1e-12),
                    "train_bpb": bit_sum / max(scored, 1),
                },
            )
    elapsed = time.perf_counter() - started
    learner.reset_episode()
    metadata: dict[str, object] = {
        "status": "complete",
        "pid": os.getpid(),
        "epoch": epoch,
        "generation": generation,
        "column": column,
        "offset": offset,
        "bytes": length,
        "scored_bytes": scored,
        "train_bits": bit_sum,
        "train_bpb": bit_sum / max(scored, 1),
        "seconds": elapsed,
        "bytes_per_second": length / max(elapsed, 1e-12),
        "new_sequence_births": learner.sequence_births - births_before,
        "nursery_evictions": learner.sequence_nursery_evictions - evictions_before,
        "body_bytes": learner.embodied_bytes,
        "checkpoint": str(output),
    }
    _atomic_pickle(output, {"learner": learner, "column": metadata})
    _atomic_json(progress, metadata)
    return metadata


def _epoch_one(report_path: Path) -> dict[str, object]:
    source = json.loads(report_path.read_text(encoding="utf-8"))
    # Accept either phase122's own report shape (consolidated_body_bytes) or
    # a phase126 per-epoch report (body_bytes) -- the latter matters when
    # the effective "epoch 1" for a run is actually a later, already
    # completed epoch (e.g. resuming a lineage part-way and switching
    # backends there), not literally the first phase122 checkpoint.
    body_bytes = source.get("consolidated_body_bytes", source.get("body_bytes"))
    return {
        "status": "complete",
        "epoch": 1,
        "source_report": str(report_path),
        "training_bpb": source["training_bpb"],
        "body_bytes": body_bytes,
        "mature_sequence_paths": source["mature_sequence_paths"],
        "evaluations": source["evaluations"],
        "generations": source["generations"],
        "checkpoint": source["checkpoint"],
    }


def _load_finished_epoch(work_dir: Path, epoch: int) -> dict[str, object] | None:
    report_path = work_dir / f"epoch_{epoch}" / "report.json"
    checkpoint_path = work_dir / f"epoch_{epoch}" / "checkpoint.pkl"
    if not report_path.exists() or not checkpoint_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report if report.get("status") == "complete" else None


def _ensure_native_epoch1_checkpoint(
    epoch1_checkpoint: Path,
    work_dir: Path,
    *,
    connectivity_pruning: bool = False,
) -> Path:
    """One-time Python -> native conversion of the epoch-1 ancestor.

    Every epoch after this loads whatever `current_checkpoint` last pointed
    at via `_load_any_cortex`, which is native-aware -- so converting once,
    up front, is enough to carry the whole run natively from here on
    (native checkpoints stay native through consolidate_delta, sleep, and
    ordinary save/load). Native config is baked in at this one conversion
    point and never revisited per epoch (unlike the Python path's
    _apply_competition_settings, called every epoch to backfill a
    possibly-older checkpoint) -- so connectivity_pruning has to be set on
    the source Python object here, before conversion, exactly like
    competition_exponent/adaptive_competition already need to be true on
    the source epoch1 pickle for the native path to pick them up.
    """
    if not native_available():
        raise RuntimeError(
            "--native requires the alm_native shared library; see cpp/README.md"
        )
    converted_path = work_dir / "epoch1_native_checkpoint.pkl"
    if converted_path.exists():
        cortex = _load_any_cortex(converted_path)
        if isinstance(cortex, NativeGrowableCortex):
            return converted_path
    with epoch1_checkpoint.open("rb") as handle:
        source = pickle.load(handle)["learner"]
    if isinstance(source, NativeGrowableCortex):
        native_cortex = source
    else:
        source.connectivity_pruning = connectivity_pruning
        native_cortex = native_cortex_from_python(source)
    _atomic_pickle(converted_path, {"learner": native_cortex})
    return converted_path


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.target_epochs < 1:
        raise ValueError("target epochs must be positive")
    if args.sleep_bytes < args.columns:
        raise ValueError("sleep interval must provide work to every column")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.tinystories.resolve()
    total_bytes = train_path.stat().st_size
    use_native = getattr(args, "native", False)
    if use_native:
        current_checkpoint = _ensure_native_epoch1_checkpoint(
            args.epoch1_checkpoint.resolve(),
            work_dir,
            connectivity_pruning=getattr(args, "connectivity_pruning", False),
        )
    else:
        current_checkpoint = args.epoch1_checkpoint.resolve()
    epoch_rows = [_epoch_one(args.epoch1_report.resolve())]
    run_started = time.perf_counter()

    for epoch in range(2, args.target_epochs + 1):
        finished = _load_finished_epoch(work_dir, epoch)
        if finished is not None:
            epoch_rows.append(finished)
            current_checkpoint = Path(finished["checkpoint"])
            continue

        epoch_dir = work_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        ancestor = _load_any_cortex(current_checkpoint)
        if isinstance(ancestor, NativeGrowableCortex):
            # Native config is fixed at construction (baked in once, by the
            # Python -> native conversion above) and survives save/load on
            # its own -- there are no loose Python attributes to backfill.
            pass
        else:
            # Callers that build a Namespace by hand (tests, other
            # harnesses) predate these settings, so every one of them falls
            # back to the value that reproduces the original behaviour.
            _apply_competition_settings(
                ancestor,
                competition_exponent=getattr(args, "competition_exponent", 1.0),
                adaptive_competition=getattr(args, "adaptive_competition", False),
                trust_gamma=getattr(args, "trust_gamma", 0.5),
                trust_learning_rate=getattr(args, "trust_learning_rate", 0.2),
                attribution_gate_threshold=getattr(
                    args, "attribution_gate_threshold", 0.5
                ),
                connectivity_pruning=getattr(args, "connectivity_pruning", False),
            )
        wake_rows: list[dict[str, object]] = []
        epoch_started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.columns) as pool:
            for generation, round_offset in enumerate(
                range(0, total_bytes, args.sleep_bytes)
            ):
                round_bytes = min(args.sleep_bytes, total_bytes - round_offset)
                # No longer shedding the nursery here: consolidate_delta
                # already merges nursery observations correctly across
                # columns (a context two columns each see below
                # sequence_confirmations individually can jointly promote
                # once merged), but wiping it every round threw that away
                # before candidates had a real chance to accumulate enough
                # repeated observations -- see the docstring on
                # SequenceCortexOpponentLearner.consolidate_to_byte_budget.
                ancestor.reset_episode()
                ancestor_path = epoch_dir / "awake_ancestor.pkl"
                _atomic_pickle(ancestor_path, {"learner": ancestor})
                arguments = []
                for column, (local_offset, length) in enumerate(
                    _partition(round_bytes, args.columns)
                ):
                    arguments.append(
                        (
                            epoch,
                            generation,
                            column,
                            str(ancestor_path),
                            str(train_path),
                            round_offset + local_offset,
                            length,
                            args.chunk_bytes,
                            str(epoch_dir / f"column_{column}.pkl"),
                        )
                    )
                wake_started = time.perf_counter()
                columns = list(pool.map(_develop, arguments))
                wake_seconds = time.perf_counter() - wake_started
                master = copy.deepcopy(ancestor)
                merge_started = time.perf_counter()
                for column in columns:
                    descendant = _load_any_cortex(column["checkpoint"])
                    master.consolidate_delta(descendant, ancestor)
                    del descendant
                merge_seconds = time.perf_counter() - merge_started
                before_sleep = master.embodied_bytes
                sleep_started = time.perf_counter()
                sleep = master.consolidate_to_byte_budget(args.max_body_bytes)
                sleep_seconds = time.perf_counter() - sleep_started
                wake_rows.append(
                    {
                        "generation": generation,
                        "offset": round_offset,
                        "bytes": round_bytes,
                        "wake_seconds": wake_seconds,
                        "merge_seconds": merge_seconds,
                        "sleep_seconds": sleep_seconds,
                        "body_bytes_before_sleep": before_sleep,
                        "body_bytes_after_sleep": master.embodied_bytes,
                        "sleep": sleep,
                        "columns": columns,
                    }
                )
                ancestor = master
                current_checkpoint = epoch_dir / "latest_checkpoint.pkl"
                _atomic_pickle(current_checkpoint, {"learner": ancestor})
                _atomic_json(
                    epoch_dir / "progress.json",
                    {
                        "status": "training",
                        "epoch": epoch,
                        "completed_generations": generation + 1,
                        "processed_bytes": round_offset + round_bytes,
                        "total_bytes": total_bytes,
                        "body_bytes": ancestor.embodied_bytes,
                        "elapsed_seconds": time.perf_counter() - epoch_started,
                    },
                )

        scored_bytes = sum(
            int(column["scored_bytes"])
            for wake in wake_rows
            for column in wake["columns"]
        )
        train_bits = sum(
            float(column["train_bits"])
            for wake in wake_rows
            for column in wake["columns"]
        )
        evaluation = {
            "tinystories": _evaluate(
                ancestor, args.tinystories_valid.resolve(), args.eval_bytes
            )
        }
        generations = _generation_panel(
            ancestor, max_new_bytes=args.generation_bytes
        )
        checkpoint = epoch_dir / "checkpoint.pkl"
        report: dict[str, object] = {
            "status": "complete",
            "epoch": epoch,
            "training_stream": str(train_path),
            "training_bytes": total_bytes,
            "sleep_bytes": args.sleep_bytes,
            "sleep_generations": len(wake_rows),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "training_bpb": train_bits / max(scored_bytes, 1),
            "body_bytes": ancestor.embodied_bytes,
            "mature_sequence_paths": ancestor.live_sequence_count,
            "evaluations": evaluation,
            "generations": generations,
            "wake_generations": wake_rows,
            "checkpoint": str(checkpoint.resolve()),
        }
        _atomic_pickle(checkpoint, {"learner": ancestor, "report": report})
        _atomic_json(epoch_dir / "report.json", report)
        epoch_rows.append(report)
        current_checkpoint = checkpoint

    result = {
        "task": "phase126_three_epoch_generation",
        "status": "complete",
        "target_epochs": args.target_epochs,
        "columns": args.columns,
        "max_body_bytes": args.max_body_bytes,
        "total_wall_seconds": time.perf_counter() - run_started,
        "epochs": epoch_rows,
        "final_checkpoint": str(current_checkpoint),
    }
    _atomic_json(work_dir / "report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epoch1-checkpoint",
        type=Path,
        default=Path("runs/alm_phase122_tiny_full_64mb_4c/checkpoint.pkl"),
    )
    parser.add_argument(
        "--epoch1-report",
        type=Path,
        default=Path("runs/alm_phase122_tiny_full_64mb_4c/report.json"),
    )
    parser.add_argument("--tinystories", type=Path, default=DEFAULT_TINY_TRAIN)
    parser.add_argument("--tinystories-valid", type=Path, default=DEFAULT_TINY_VALID)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--target-epochs", type=int, default=3)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--sleep-bytes", type=int, default=256_000_000)
    parser.add_argument("--chunk-bytes", type=int, default=4_000_000)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--generation-bytes", type=int, default=512)
    parser.add_argument("--max-body-bytes", type=int, default=64_000_000)
    # Defaults below match the actual Phase 153 experiment configuration
    # (gamma=0.5, trust learning rate=0.2, gate threshold=0.5, exponent=1.0)
    # so --adaptive-competition alone reproduces "whatever matches phase 153".
    parser.add_argument(
        "--native",
        action="store_true",
        help=(
            "Run on the C++ cortex (cpp/) instead of the pure-Python one. "
            "The epoch-1 checkpoint is converted once (including any "
            "learned adaptive-competition trust) and every epoch after "
            "that stays native -- consolidate_delta, sleep, and checkpoint "
            "save/load are all bit-exact-verified native implementations."
        ),
    )
    parser.add_argument("--adaptive-competition", action="store_true")
    parser.add_argument(
        "--connectivity-pruning",
        action="store_true",
        help=(
            "sleep-time-only: rank mature sequence rows for pruning by how "
            "many longer, deeper mature contexts use them as their trailing "
            "backoff suffix (a hub measure, ported from the vision "
            "organism's relation-degree utility) instead of raw count. "
            "Validated to beat count-based pruning at matched row count on "
            "both TinyStories and Wikipedia, with the margin growing (not "
            "shrinking) at 5x more training data -- see "
            "tests/test_dendritic_basin_connectivity_pruning.py and "
            "docs/ALM_V1_Architecture.md. Does not change growth, "
            "promotion, or the probability math, and costs nothing extra "
            "in memory (degree is computed fresh at sleep time)."
        ),
    )
    parser.add_argument("--competition-exponent", type=float, default=1.0)
    parser.add_argument("--trust-gamma", type=float, default=0.5)
    parser.add_argument("--trust-learning-rate", type=float, default=0.2)
    parser.add_argument("--attribution-gate-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
