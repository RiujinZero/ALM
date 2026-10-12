"""Develop, consolidate, and evaluate a bounded four-column language ALM."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.dendritic_basin.tasks.interleaved_language_bytes import (
    build_interleaved_stream,
)
from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    generate_bytes,
)

DEFAULT_TINY_TRAIN = Path(r"C:\path\to\gdrive\training_data\TinyStories-train.txt")
DEFAULT_TINY_VALID = Path(r"C:\path\to\gdrive\training_data\TinyStories-valid.txt")
DEFAULT_SMOL = Path(r"C:\path\to\manifoldflow_chat_sources\smoltalk_everyday.jsonl")


def _replace_with_retry(
    temporary: Path, path: Path, *, attempts: int = 6, delay_seconds: float = 0.25
) -> None:
    """Retry an atomic rename past transient Windows ERROR_ACCESS_DENIED.

    ``Path.replace`` on Windows can fail with ``PermissionError``
    (``WinError 5``) when the destination is briefly held open by another
    process (antivirus scanning, an indexer, a concurrent reader) without
    ``FILE_SHARE_DELETE`` -- not a real conflict, just a race that clears
    itself within milliseconds. Observed in practice under many concurrent
    columns writing small progress files. Retrying is safe: the rename
    itself is atomic, so a retry never risks a torn write.
    """
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)


def _atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _replace_with_retry(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _new_cortex(max_nursery_paths: int) -> SequenceCortexOpponentLearner:
    return SequenceCortexOpponentLearner(
        max_context_bytes=6,
        max_sequence_nursery_paths=max_nursery_paths,
        packed_paths=True,
    )


def _partition(total: int, columns: int) -> list[tuple[int, int]]:
    base, remainder = divmod(total, columns)
    result = []
    offset = 0
    for column in range(columns):
        length = base + int(column < remainder)
        result.append((offset, length))
        offset += length
    return result


def _train_column(
    arguments: tuple[int, str, int, int, int, int, str]
) -> dict[str, object]:
    (
        column,
        data_path,
        offset,
        length,
        chunk_bytes,
        max_nursery_paths,
        work_dir,
    ) = arguments
    output = Path(work_dir) / f"column_{column}.pkl"
    metadata_path = Path(work_dir) / f"column_{column}.json"
    expected = {
        "column": column,
        "data_path": str(Path(data_path).resolve()),
        "offset": offset,
        "train_bytes": length,
        "max_nursery_paths": max_nursery_paths,
    }
    if output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            print(f"[column {column}] resuming completed tissue", flush=True)
            return metadata

    cortex = _new_cortex(max_nursery_paths)
    bit_sum = 0.0
    scored = 0
    processed = 0
    next_progress = 64_000_000
    started = time.perf_counter()
    progress_path = Path(work_dir) / f"column_{column}_progress.json"
    with Path(data_path).open("rb") as handle:
        handle.seek(offset)
        while processed < length:
            raw = handle.read(min(chunk_bytes, length - processed))
            if not raw:
                raise ValueError("training stream ended inside a cortical territory")
            probability = cortex.observe_chunk_probabilities(
                np.frombuffer(raw, dtype=np.uint8), learn=True
            )
            finite = probability[np.isfinite(probability)]
            bit_sum += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
            scored += len(finite)
            processed += len(raw)
            if processed >= next_progress or processed == length:
                elapsed = time.perf_counter() - started
                progress = {
                    **expected,
                    "processed_bytes": processed,
                    "elapsed_seconds": elapsed,
                    "bytes_per_second": processed / max(elapsed, 1e-12),
                    "body_bytes": cortex.embodied_bytes,
                    "status": "training" if processed < length else "checkpointing",
                }
                _atomic_json(progress_path, progress)
                print(
                    f"[column {column}] {processed:,}/{length:,} B | "
                    f"{processed / max(elapsed, 1e-12):,.0f} B/s | "
                    f"body {progress['body_bytes'] / 1_000_000:.2f} MB",
                    flush=True,
                )
                next_progress += 64_000_000
    elapsed = time.perf_counter() - started
    cortex.reset_episode()
    metadata = {
        **expected,
        "scored_bytes": scored,
        "train_bits": bit_sum,
        "train_bpb": bit_sum / max(scored, 1),
        "train_seconds": elapsed,
        "train_bytes_per_second": length / max(elapsed, 1e-12),
        "body_bytes": cortex.embodied_bytes,
        "mature_sequence_paths": cortex.live_sequence_count,
        "nursery_sequence_paths": cortex.sequence_nursery_count,
        "checkpoint": str(output),
    }
    _atomic_pickle(output, {"learner": cortex, "column": metadata})
    _atomic_json(metadata_path, metadata)
    _atomic_json(progress_path, {**metadata, "status": "complete"})
    return metadata


def _load_cortex(path: str | Path) -> SequenceCortexOpponentLearner:
    with Path(path).open("rb") as handle:
        cortex = pickle.load(handle)["learner"]
    if not isinstance(cortex, SequenceCortexOpponentLearner):
        raise TypeError("column checkpoint does not contain a sequence cortex")
    cortex.reset_episode()
    return cortex


def _evaluate(
    cortex: SequenceCortexOpponentLearner,
    path: Path,
    length: int,
) -> dict[str, object]:
    cortex.reset_episode()
    bit_sum = 0.0
    scored = 0
    processed = 0
    started = time.perf_counter()
    with path.open("rb") as handle:
        while processed < length:
            raw = handle.read(min(4_000_000, length - processed))
            if not raw:
                break
            probability = cortex.observe_chunk_probabilities(
                np.frombuffer(raw, dtype=np.uint8), learn=False
            )
            finite = probability[np.isfinite(probability)]
            bit_sum += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
            scored += len(finite)
            processed += len(raw)
    elapsed = time.perf_counter() - started
    cortex.reset_episode()
    return {
        "path": str(path),
        "bytes": processed,
        "scored_bytes": scored,
        "bits_per_byte": bit_sum / max(scored, 1),
        "bytes_per_second": processed / max(elapsed, 1e-12),
        "seconds": elapsed,
    }


def _generation_panel(
    cortex: SequenceCortexOpponentLearner,
    *,
    max_new_bytes: int,
) -> list[dict[str, object]]:
    prompts = (
        ("story", b"Once upon a time"),
        ("story_identity", b"Lily had a red ball. One day, Lily"),
        ("chat_greeting", b"USER: Hello! How are you?\nASSISTANT:"),
        (
            "chat_advice",
            b"USER: What should I do if I feel tired?\nASSISTANT:",
        ),
        (
            "chat_capital",
            b"USER: What is the capital of France?\nASSISTANT:",
        ),
        (
            # Referent ("Paris") sits ~42 bytes before generation resumes --
            # past even the deepest long-range band (32 bytes, see
            # cortex.hpp's long_range_band_depths). No lookup mechanism can
            # see that far back regardless of training, so this variant
            # measures local plausibility only, not memory.
            "chat_multiturn_far",
            b"USER: What is the capital of France?\nASSISTANT: The capital of "
            b"France is Paris.\nUSER: What is its population?\nASSISTANT:",
        ),
        (
            # Referent sits ~28 bytes back -- inside the deepest band's
            # reach, so a correct answer here is at least structurally
            # possible. Still depends on that specific tail context having
            # a stored association; a coherent "no" (falls back gracefully,
            # doesn't hallucinate a wrong city) is informative too.
            "chat_multiturn_near",
            b"USER: What is the capital of France?\nASSISTANT: Paris.\nUSER: "
            b"Its population?\nASSISTANT:",
        ),
    )
    panel = []
    boundary = b"<|endoftext|>\n"
    for label, prompt in prompts:
        for mode, temperature, top_k in (
            ("greedy", 0.0, 1),
            ("sample", 0.8, 32),
        ):
            started = time.perf_counter()
            generated = generate_bytes(
                cortex,
                prompt,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=0.95,
                seed=73,
                stop_at_story_boundary=False,
            )
            elapsed = time.perf_counter() - started
            panel.append(
                {
                    "label": label,
                    "mode": mode,
                    "prompt": prompt.decode("utf-8"),
                    "generated_bytes": len(generated),
                    "bytes_per_second": len(generated) / max(elapsed, 1e-12),
                    "text": (prompt + generated).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "generated_hex": generated.hex(),
                    "diagnostics": _generation_diagnostics(generated, boundary),
                }
            )
    cortex.reset_episode()
    return panel


def run(args: argparse.Namespace) -> dict[str, object]:
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.tinystories.resolve()
    mixture = None
    if args.smoltalk is not None:
        mixed_path = work_dir / "tinystories_smoltalk_full.bin"
        manifest_path = work_dir / "mixture_manifest.json"
        if mixed_path.exists() and manifest_path.exists():
            mixture = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            print("[mixture] building deterministic whole-episode stream", flush=True)
            mixture_report = build_interleaved_stream(
                train_path,
                args.smoltalk.resolve(),
                mixed_path,
                total_bytes=train_path.stat().st_size,
                seed=args.seed,
                validation_fraction=0.1,
                smoltalk_valid_output=args.smoltalk_valid,
            )
            mixture = asdict(mixture_report)
            _atomic_json(manifest_path, mixture)
        train_path = mixed_path

    total = train_path.stat().st_size
    partitions = _partition(total, args.columns)
    column_arguments = [
        (
            column,
            str(train_path),
            offset,
            length,
            args.chunk_bytes,
            args.max_nursery_paths,
            str(work_dir),
        )
        for column, (offset, length) in enumerate(partitions)
    ]
    training_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.columns) as pool:
        columns = list(pool.map(_train_column, column_arguments))
    training_wall = time.perf_counter() - training_started

    print("[sleep] consolidating four cortical lineages", flush=True)
    merge_started = time.perf_counter()
    master = _load_cortex(columns[0]["checkpoint"])
    for metadata in columns[1:]:
        incoming = _load_cortex(metadata["checkpoint"])
        master.consolidate(incoming)
        del incoming
    merge_seconds = time.perf_counter() - merge_started
    body_before_sleep = master.embodied_bytes
    print(
        f"[sleep] merged body {body_before_sleep / 1_000_000:.2f} MB; "
        f"competing to {args.max_body_bytes / 1_000_000:.2f} MB",
        flush=True,
    )
    sleep_started = time.perf_counter()
    sleep = master.consolidate_to_byte_budget(args.max_body_bytes)
    sleep_seconds = time.perf_counter() - sleep_started
    if master.embodied_bytes > args.max_body_bytes:
        raise AssertionError("sleep failed to enforce ALM byte class")

    evaluations = {
        "tinystories": _evaluate(master, args.tinystories_valid, args.eval_bytes),
    }
    if args.smoltalk_valid.exists():
        evaluations["smoltalk"] = _evaluate(
            master,
            args.smoltalk_valid,
            min(args.eval_bytes, args.smoltalk_valid.stat().st_size),
        )
    generations = _generation_panel(master, max_new_bytes=args.generation_bytes)
    report = {
        "task": "phase122_full_corpus_4c",
        "alm_name": f"ALM-{args.max_body_bytes // 1_000_000}MB-{args.columns}C",
        "pid": os.getpid(),
        "training_stream": str(train_path),
        "training_stream_bytes": total,
        "mixture": mixture,
        "columns": columns,
        "column_count": args.columns,
        "awake_aggregate_body_bytes": sum(int(row["body_bytes"]) for row in columns),
        "training_wall_seconds": training_wall,
        "training_wall_bytes_per_second": total / max(training_wall, 1e-12),
        "training_bpb": sum(float(row["train_bits"]) for row in columns)
        / max(sum(int(row["scored_bytes"]) for row in columns), 1),
        "merge_seconds": merge_seconds,
        "body_bytes_before_sleep": body_before_sleep,
        "sleep": sleep,
        "sleep_seconds": sleep_seconds,
        "body_class_bytes": args.max_body_bytes,
        "consolidated_body_bytes": master.embodied_bytes,
        "mature_sequence_paths": master.live_sequence_count,
        "evaluations": evaluations,
        "generations": generations,
    }
    checkpoint = work_dir / "checkpoint.pkl"
    report_path = work_dir / "report.json"
    _atomic_pickle(checkpoint, {"learner": master, "report": report})
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_file_bytes"] = checkpoint.stat().st_size
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinystories", type=Path, default=DEFAULT_TINY_TRAIN)
    parser.add_argument("--tinystories-valid", type=Path, default=DEFAULT_TINY_VALID)
    parser.add_argument(
        "--smoltalk",
        type=Path,
        default=None,
        help="build and train the matched TinyStories/SmolTalk arm",
    )
    parser.add_argument(
        "--smoltalk-valid",
        type=Path,
        default=Path("runs/smoltalk_everyday_valid_seed73.txt"),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--max-body-bytes", type=int, default=64_000_000)
    parser.add_argument("--max-nursery-paths", type=int, default=50_000)
    parser.add_argument("--chunk-bytes", type=int, default=4_000_000)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--generation-bytes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if min(
        args.columns,
        args.max_body_bytes,
        args.max_nursery_paths,
        args.chunk_bytes,
        args.eval_bytes,
    ) < 1:
        parser.error("counts and byte budgets must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
