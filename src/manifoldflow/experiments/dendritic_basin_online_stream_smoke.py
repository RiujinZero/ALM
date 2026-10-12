"""Phase 13: byte-class cortical ALM on a bounded one-pass raw-byte stream.

Unlike the Phase 8 diary experiment, this run never stores trace snapshots
or revisits training observations.  Prediction is prequential: the
probability assigned to byte t is recorded before byte t is allowed to alter
the learner.  Validation uses a cloned, frozen learner (dynamic traces evolve;
anatomy and local statistics do not).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from io import TextIOBase
from itertools import islice
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.cpu_column_stream import (
    CpuColumnDevelopmentalRunner,
)
from manifoldflow.dendritic_basin.gpu_multicolumn_stream import (
    CudaDevelopmentalRunner,
)
from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)
from manifoldflow.dendritic_basin.tasks.tinystories_bytes import (
    BYTE_VOCAB_SIZE,
    iter_byte_stream,
)

DEFAULT_TRAIN_PATH = r"C:\path\to\gdrive\training_data\TinyStories-train.txt"
DEFAULT_VALID_PATH = r"C:\path\to\gdrive\training_data\TinyStories-valid.txt"
BYTE_CLASSES = {
    "2MB": 2_000_000,
    "16MB": 16_000_000,
    "64MB": 64_000_000,
    "256MB": 256_000_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-bytes", type=int, default=100_000)
    parser.add_argument("--eval-bytes", type=int, default=20_000)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--stream-chunk-bytes", type=int, default=1 << 20)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--cpu-column-workers",
        type=int,
        default=0,
        help="use resident developmental CPU columns with this many workers",
    )
    execution.add_argument(
        "--cuda-developmental",
        action="store_true",
        help="use resident CUDA metabolism with host structural wake windows",
    )
    parser.add_argument(
        "--structural-wake-bytes",
        type=int,
        default=256,
        help="causal host bytes after a CUDA structural interrupt",
    )
    parser.add_argument("--max-relations", type=int, default=128)
    parser.add_argument("--max-nursery-candidates", type=int, default=256)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-alm-bytes", type=int, default=None)
    budget.add_argument("--alm-class", choices=tuple(BYTE_CLASSES), default=None)
    parser.add_argument("--max-active-relations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--column-size", type=int, default=256)
    parser.add_argument("--max-active-columns", type=int, default=4)
    parser.add_argument("--max-active-per-column", type=int, default=2)
    parser.add_argument("--inter-column-birth-probability", type=float, default=0.05)
    parser.add_argument(
        "--metabolic-backend",
        choices=("dense", "event"),
        default="dense",
        help="dense production path or experimental sparse event routing",
    )
    parser.add_argument("--surprise-birth-bits", type=float, default=6.0)
    parser.add_argument("--birth-cooldown", type=int, default=8)
    parser.add_argument("--nursery-confirmations", type=int, default=3)
    parser.add_argument("--nursery-target-consistency", type=float, default=0.60)
    parser.add_argument("--persistence-decay", type=float, default=0.97)
    parser.add_argument(
        "--utility-gated-persistence",
        action="store_true",
        help="stop propagating mature persistent states with negative utility",
    )
    parser.add_argument("--recursive-birth-probability", type=float, default=0.25)
    parser.add_argument("--perturb-step", type=float, default=0.08)
    parser.add_argument(
        "--response-backend",
        choices=("scalar", "batched", "compiled"),
        default="compiled",
    )
    parser.add_argument(
        "--adaptation-backend",
        choices=("scalar", "compiled"),
        default="compiled",
    )
    parser.add_argument(
        "--adaptation-rng",
        choices=("global", "counter"),
        default="counter",
    )
    parser.add_argument(
        "--outcome-backend",
        choices=("scalar", "compiled"),
        default="compiled",
    )
    parser.add_argument(
        "--competition-backend",
        choices=("scalar", "batched", "compiled"),
        default="batched",
    )
    parser.add_argument(
        "--contextual-residual",
        action="store_true",
        help=(
            "use a causal byte-transition basal column and bounded "
            "dendritic log-probability residuals"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="emit live telemetry every N input bytes; 0 disables it",
    )
    return parser.parse_args()


@dataclass
class StreamMetrics:
    """Constant-memory prequential score, including a bounded tail window."""

    tail_size: int
    count: int = 0
    model_bit_sum: float = 0.0
    base_bit_sum: float = 0.0

    def __post_init__(self) -> None:
        self.tail_model_bits: deque[float] = deque(maxlen=self.tail_size)
        self.tail_base_bits: deque[float] = deque(maxlen=self.tail_size)

    def add(self, model_probability: float, base_probability: float) -> None:
        model_bits = -math.log2(max(model_probability, 1e-12))
        base_bits = -math.log2(max(base_probability, 1e-12))
        self.count += 1
        self.model_bit_sum += model_bits
        self.base_bit_sum += base_bits
        self.tail_model_bits.append(model_bits)
        self.tail_base_bits.append(base_bits)

    def add_many(
        self, model_probabilities: np.ndarray, base_probabilities: np.ndarray
    ) -> None:
        """Accumulate a compiled byte chunk while retaining only its tail."""
        valid = np.logical_and(
            np.isfinite(model_probabilities), np.isfinite(base_probabilities)
        )
        if not np.any(valid):
            return
        model_bits = -np.log2(np.maximum(model_probabilities[valid], 1e-12))
        base_bits = -np.log2(np.maximum(base_probabilities[valid], 1e-12))
        self.count += len(model_bits)
        self.model_bit_sum += float(model_bits.sum())
        self.base_bit_sum += float(base_bits.sum())
        self.tail_model_bits.extend(model_bits[-self.tail_size :].tolist())
        self.tail_base_bits.extend(base_bits[-self.tail_size :].tolist())

    @property
    def model_bits_per_byte(self) -> float:
        return self.model_bit_sum / max(self.count, 1)

    @property
    def base_bits_per_byte(self) -> float:
        return self.base_bit_sum / max(self.count, 1)

    @property
    def tail_model_bits_per_byte(self) -> float:
        return sum(self.tail_model_bits) / max(len(self.tail_model_bits), 1)

    @property
    def tail_base_bits_per_byte(self) -> float:
        return sum(self.tail_base_bits) / max(len(self.tail_base_bits), 1)

    def state_dict(self) -> dict:
        return {
            "tail_size": self.tail_size,
            "count": self.count,
            "model_bit_sum": self.model_bit_sum,
            "base_bit_sum": self.base_bit_sum,
            "tail_model_bits": list(self.tail_model_bits),
            "tail_base_bits": list(self.tail_base_bits),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> StreamMetrics:
        metrics = cls(tail_size=int(state["tail_size"]))
        metrics.count = int(state["count"])
        metrics.model_bit_sum = float(state["model_bit_sum"])
        metrics.base_bit_sum = float(state["base_bit_sum"])
        metrics.tail_model_bits.extend(state["tail_model_bits"])
        metrics.tail_base_bits.extend(state["tail_base_bits"])
        return metrics


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )


def _emit_progress(
    *,
    label: str,
    processed: int,
    rate_processed: int,
    total: int,
    elapsed: float,
    metrics: StreamMetrics,
    learner: OnlineDendriticStreamLearner,
    output: TextIOBase,
) -> None:
    rate = rate_processed / max(elapsed, 1e-12)
    eta = (total - processed) / rate if rate > 0.0 else math.inf
    percent = 100.0 * processed / max(total, 1)
    print(
        f"[{label}] {processed:,}/{total:,} bytes ({percent:5.1f}%) | "
        f"{rate:,.0f} bytes/s | ETA {_format_duration(eta)} | "
        f"rolling {metrics.tail_model_bits_per_byte:.4f} bpb | "
        f"relations {learner.live_relation_count}/{learner.config.max_relations} | "
        f"nursery {learner.live_nursery_count}/"
        f"{learner.config.max_nursery_candidates} | "
        f"births/prunes {learner.stats.births}/{learner.stats.prunes}",
        file=output,
        flush=True,
    )


def _run_stream(
    learner: OnlineDendriticStreamLearner,
    stream: Iterable[int],
    *,
    learn: bool,
    tail_size: int,
    total: int | None = None,
    metrics: StreamMetrics | None = None,
    initial_processed: int = 0,
    progress_every: int = 0,
    label: str = "stream",
    progress_output: TextIOBase = sys.stderr,
    progress_callback: Callable[[int, StreamMetrics], None] | None = None,
    callback_every: int = 0,
    cpu_column_workers: int = 0,
    cuda_developmental: bool = False,
    structural_wake_bytes: int = 256,
) -> StreamMetrics:
    if metrics is None:
        metrics = StreamMetrics(tail_size=tail_size)
    start = time.perf_counter()
    if total is None:
        try:
            total = len(stream)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("total is required for a streaming iterable") from exc
    resident = None
    if learn and cpu_column_workers > 0:
        resident = CpuColumnDevelopmentalRunner(learner, workers=cpu_column_workers)
    elif learn and cuda_developmental:
        resident = CudaDevelopmentalRunner(
            learner, structural_wake_bytes=structural_wake_bytes
        )
    if learner.supports_compiled_chunks or resident is not None:
        iterator = iter(stream)
        session_processed = 0
        while session_processed < total - initial_processed:
            processed = initial_processed + session_processed
            chunk_limit = min(8192, total - processed)
            for interval in (progress_every, callback_every):
                if interval > 0:
                    chunk_limit = min(chunk_limit, interval - processed % interval)
            chunk = np.fromiter(islice(iterator, chunk_limit), dtype=np.int64, count=-1)
            if len(chunk) == 0:
                break
            if resident is None:
                model_probabilities, base_probabilities = (
                    learner.observe_chunk_probabilities(chunk, learn=learn)
                )
            else:
                model_probabilities, base_probabilities = (
                    resident.observe_chunk_probabilities(chunk)
                )
            metrics.add_many(model_probabilities, base_probabilities)
            session_processed += len(chunk)
            processed = initial_processed + session_processed
            if progress_every > 0 and (
                processed % progress_every == 0 or processed == total
            ):
                if resident is not None:
                    resident.synchronize()
                _emit_progress(
                    label=label,
                    processed=processed,
                    rate_processed=session_processed,
                    total=total,
                    elapsed=time.perf_counter() - start,
                    metrics=metrics,
                    learner=learner,
                    output=progress_output,
                )
            if (
                progress_callback is not None
                and callback_every > 0
                and (processed % callback_every == 0 or processed == total)
            ):
                if resident is not None:
                    resident.synchronize()
                progress_callback(processed, metrics)
        if resident is not None:
            resident.synchronize()
        return metrics
    for session_processed, token_id in enumerate(stream, start=1):
        processed = initial_processed + session_processed
        step = learner.observe(token_id, learn=learn)
        if step.observed_probability is not None:
            assert step.base_observed_probability is not None
            metrics.add(step.observed_probability, step.base_observed_probability)
        if progress_every > 0 and (
            processed % progress_every == 0 or processed == total
        ):
            _emit_progress(
                label=label,
                processed=processed,
                rate_processed=session_processed,
                total=total,
                elapsed=time.perf_counter() - start,
                metrics=metrics,
                learner=learner,
                output=progress_output,
            )
        if (
            progress_callback is not None
            and callback_every > 0
            and (processed % callback_every == 0 or processed == total)
        ):
            progress_callback(processed, metrics)
    return metrics


class StreamingNGramControls:
    """One-pass fixed-order controls that never retain corpus bytes."""

    def __init__(self, max_order: int = 3, alpha: float = 0.1):
        self.max_order = max_order
        self.alpha = alpha
        self.context_totals: list[dict[tuple[int, ...], int]] = [
            defaultdict(int) for _ in range(max_order)
        ]
        self.transition_counts: list[dict[tuple[int, ...], int]] = [
            defaultdict(int) for _ in range(max_order)
        ]
        self.history: deque[int] = deque(maxlen=max_order - 1)
        self.eval_bit_sums = np.zeros(max_order)
        self.eval_count = 0

    def observe_train(self, token_id: int) -> None:
        history = tuple(self.history)
        for order in range(1, self.max_order + 1):
            context_size = order - 1
            if len(history) < context_size:
                continue
            context = history[-context_size:] if context_size else ()
            self.context_totals[order - 1][context] += 1
            self.transition_counts[order - 1][context + (token_id,)] += 1
        self.history.append(token_id)

    def begin_evaluation(self) -> None:
        self.history.clear()
        self.eval_bit_sums.fill(0.0)
        self.eval_count = 0

    def observe_eval(self, token_id: int) -> None:
        history = tuple(self.history)
        probability = 1.0 / BYTE_VOCAB_SIZE
        for order in range(1, self.max_order + 1):
            context_size = order - 1
            if len(history) >= context_size:
                context = history[-context_size:] if context_size else ()
                total = self.context_totals[order - 1].get(context, 0)
                if total:
                    count = self.transition_counts[order - 1].get(
                        context + (token_id,), 0
                    )
                    probability = (count + self.alpha) / (
                        total + self.alpha * BYTE_VOCAB_SIZE
                    )
            self.eval_bit_sums[order - 1] += -math.log2(max(probability, 1e-12))
        self.eval_count += 1
        self.history.append(token_id)

    def bits_per_byte(self, order: int) -> float:
        return float(self.eval_bit_sums[order - 1] / max(self.eval_count, 1))


def _streaming_ngram_controls(
    *,
    train_path: str,
    train_bytes: int,
    train_offset: int,
    eval_path: str,
    eval_bytes: int,
    chunk_size: int,
) -> tuple[float, float, float]:
    controls = StreamingNGramControls(max_order=3)
    for token_id in iter_byte_stream(
        train_path, train_bytes, train_offset, chunk_size=chunk_size
    ):
        controls.observe_train(token_id)
    controls.begin_evaluation()
    for token_id in iter_byte_stream(eval_path, eval_bytes, chunk_size=chunk_size):
        controls.observe_eval(token_id)
    return tuple(controls.bits_per_byte(order) for order in (1, 2, 3))


def _relation_report(learner: OnlineDendriticStreamLearner) -> list[dict]:
    result = []
    for slot in learner.slots:
        if slot is None:
            continue
        result.append(
            {
                "slot": slot.slot_id,
                "current_channel": slot.current_channel,
                "trace_channel": slot.trace_channel,
                "timescale": learner.config.timescales[slot.timescale_index],
                "depth": slot.depth,
                "column": slot.column_id,
                "lifecycle": ("juvenile", "consolidated", "dormant")[
                    int(learner._lifecycle[slot.slot_id])
                ],
                "age_bytes": int(
                    learner.stats.steps - learner._born_steps[slot.slot_id]
                ),
                "utility_bits": float(learner._utility_bits[slot.slot_id]),
                "state_utility_bits": float(learner._state_utility_bits[slot.slot_id]),
                "state_observations": int(learner._state_observations[slot.slot_id]),
                "observations": int(learner._observations[slot.slot_id]),
                "fires": int(learner._fires[slot.slot_id]),
                "predicted_byte": int(np.argmax(slot.predictor_counts)),
                "dendrite_objective": slot.dendrite.local_objective(),
            }
        )
    result.sort(key=lambda item: item["utility_bits"], reverse=True)
    return result


def _anatomy_report(learner: OnlineDendriticStreamLearner) -> dict:
    live_ids = np.flatnonzero(learner._live)
    ages = learner.stats.steps - learner._born_steps[live_ids]
    lifecycle_names = ("juvenile", "consolidated", "dormant")
    columns = []
    cross_column_relations = 0
    for column_id in range(learner.column_count):
        column_ids = live_ids[live_ids // learner.config.column_size == column_id]
        columns.append(
            {
                "column": column_id,
                "live_relations": len(column_ids),
                "mean_utility_bits": (
                    float(learner._utility_bits[column_ids].mean())
                    if len(column_ids)
                    else 0.0
                ),
            }
        )
    for slot in learner.slots:
        if slot is None:
            continue
        parent_columns = {
            (channel - learner.config.vocab_size) // learner.config.column_size
            for channel in (slot.current_channel, slot.trace_channel)
            if channel >= learner.config.vocab_size
        }
        if any(column != slot.column_id for column in parent_columns):
            cross_column_relations += 1
    return {
        "column_count": learner.column_count,
        "columns": columns,
        "cross_column_relations": cross_column_relations,
        "depth_histogram": {
            str(depth): sum(
                slot is not None and slot.depth == depth for slot in learner.slots
            )
            for depth in range(1, learner.config.max_depth + 1)
        },
        "lifecycle_histogram": {
            name: int(np.sum(learner._lifecycle[live_ids] == index))
            for index, name in enumerate(lifecycle_names)
        },
        "mean_branch_age_bytes": float(ages.mean()) if len(ages) else 0.0,
        "median_branch_age_bytes": float(np.median(ages)) if len(ages) else 0.0,
        "oldest_branch_age_bytes": int(ages.max()) if len(ages) else 0,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    max_alm_bytes = (
        BYTE_CLASSES[args.alm_class]
        if args.alm_class is not None
        else args.max_alm_bytes
    )
    config = OnlineDendriticConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        max_relations=args.max_relations,
        max_nursery_candidates=args.max_nursery_candidates,
        max_alm_bytes=max_alm_bytes,
        max_active_relations=args.max_active_relations,
        max_depth=args.max_depth,
        column_size=args.column_size,
        max_active_columns=args.max_active_columns,
        max_active_per_column=args.max_active_per_column,
        inter_column_birth_probability=args.inter_column_birth_probability,
        metabolic_backend=args.metabolic_backend,
        surprise_birth_bits=args.surprise_birth_bits,
        birth_cooldown=args.birth_cooldown,
        nursery_confirmations=args.nursery_confirmations,
        nursery_target_consistency=args.nursery_target_consistency,
        persistence_decay=args.persistence_decay,
        utility_gated_persistence=args.utility_gated_persistence,
        recursive_birth_probability=args.recursive_birth_probability,
        perturb_step=args.perturb_step,
        response_backend=args.response_backend,
        adaptation_backend=args.adaptation_backend,
        adaptation_rng=args.adaptation_rng,
        outcome_backend=args.outcome_backend,
        competition_backend=args.competition_backend,
        contextual_residual=args.contextual_residual,
        seed=args.seed,
    )
    if args.resume_from is not None:
        learner = OnlineDendriticStreamLearner.load_checkpoint(args.resume_from)
        config = learner.config
    else:
        learner = OnlineDendriticStreamLearner(config)
    if args.cpu_column_workers < 0:
        raise ValueError("cpu column workers cannot be negative")
    if args.structural_wake_bytes < 1:
        raise ValueError("structural wake bytes must be positive")
    if (args.cpu_column_workers > 0 or args.cuda_developmental) and (
        not learner.config.contextual_residual
        or learner.config.adaptation_rng != "counter"
    ):
        raise ValueError(
            "resident columns require contextual residuals and counter RNG"
        )
    learner.warm_execution_backend()
    population_at_resume = asdict(learner.stats)

    checkpoint_path = args.checkpoint or args.resume_from
    progress_path = (
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".progress.json")
        if checkpoint_path is not None
        else None
    )
    completed_train_bytes = 0
    train_metrics: StreamMetrics | None = None
    if args.resume_from is not None:
        completed_train_bytes = learner.stats.steps
        if progress_path is not None and progress_path.exists():
            progress_state = json.loads(progress_path.read_text(encoding="utf-8"))
            if int(progress_state["learner_steps"]) == learner.stats.steps:
                completed_train_bytes = int(progress_state["train_bytes_processed"])
                train_metrics = StreamMetrics.from_state_dict(
                    progress_state["train_metrics"]
                )
    remaining_train_bytes = max(0, args.train_bytes - completed_train_bytes)
    train_stream = iter_byte_stream(
        args.train_path,
        remaining_train_bytes,
        args.train_offset + completed_train_bytes,
        chunk_size=args.stream_chunk_bytes,
    )

    def checkpoint_progress(processed: int, metrics: StreamMetrics) -> None:
        if checkpoint_path is None or progress_path is None:
            return
        learner.save_checkpoint(checkpoint_path)
        _atomic_write_json(
            progress_path,
            {
                "train_path": args.train_path,
                "train_offset": args.train_offset,
                "train_bytes_target": args.train_bytes,
                "train_bytes_processed": processed,
                "learner_steps": learner.stats.steps,
                "train_metrics": metrics.state_dict(),
            },
        )

    start = time.perf_counter()
    tail = min(20_000, max(args.train_bytes - 1, 1))
    train_metrics = _run_stream(
        learner,
        train_stream,
        learn=True,
        tail_size=tail,
        total=args.train_bytes,
        metrics=train_metrics,
        initial_processed=completed_train_bytes,
        progress_every=args.progress_every,
        label="train",
        progress_callback=checkpoint_progress,
        callback_every=args.checkpoint_every,
        cpu_column_workers=args.cpu_column_workers,
        cuda_developmental=args.cuda_developmental,
        structural_wake_bytes=args.structural_wake_bytes,
    )
    train_seconds = time.perf_counter() - start

    evaluator = OnlineDendriticStreamLearner(config)
    evaluator.load_state_dict(learner.state_dict())
    evaluator.reset_episode()
    eval_stream = iter_byte_stream(
        args.valid_path,
        args.eval_bytes,
        chunk_size=args.stream_chunk_bytes,
    )
    start = time.perf_counter()
    eval_metrics = _run_stream(
        evaluator,
        eval_stream,
        learn=False,
        tail_size=max(args.eval_bytes - 1, 1),
        total=args.eval_bytes,
        progress_every=args.progress_every,
        label="eval",
    )
    eval_seconds = time.perf_counter() - start

    unigram_bpb, bigram_bpb, trigram_bpb = _streaming_ngram_controls(
        train_path=args.train_path,
        train_bytes=args.train_bytes,
        train_offset=args.train_offset,
        eval_path=args.valid_path,
        eval_bytes=args.eval_bytes,
        chunk_size=args.stream_chunk_bytes,
    )

    serialized_alm_bytes = learner.serialized_size_bytes()
    population = asdict(learner.stats)
    population_delta = {
        key: value - population_at_resume[key] for key, value in population.items()
    }
    report = {
        "phase": 14 if config.contextual_residual else 13,
        "mode": (
            "bounded_contextual_residual_raw_bytes"
            if config.contextual_residual
            else "bounded_strict_online_raw_bytes"
        ),
        "config": asdict(config),
        "train_bytes": args.train_bytes,
        "eval_bytes": args.eval_bytes,
        "resumed_from_bytes": completed_train_bytes,
        "train_tail_bytes": tail,
        "train_tail_model_bits_per_byte": train_metrics.tail_model_bits_per_byte,
        "train_tail_base_bits_per_byte": train_metrics.tail_base_bits_per_byte,
        "eval_model_bits_per_byte": eval_metrics.model_bits_per_byte,
        "eval_base_bits_per_byte": eval_metrics.base_bits_per_byte,
        "eval_static_unigram_bits_per_byte": unigram_bpb,
        "eval_bigram_bits_per_byte": bigram_bpb,
        "eval_trigram_bits_per_byte": trigram_bpb,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "train_bytes_per_second": remaining_train_bytes / max(train_seconds, 1e-12),
        "eval_bytes_per_second": args.eval_bytes / max(eval_seconds, 1e-12),
        "serialized_alm_bytes": serialized_alm_bytes,
        "sensory_context_bytes": learner.context_counts.nbytes,
        "max_alm_bytes": config.max_alm_bytes,
        "alm_class": (
            f"ALM-{config.max_alm_bytes / 1_000_000:g}MB"
            if config.max_alm_bytes is not None
            else None
        ),
        "estimated_max_serialized_bytes": (config.estimated_max_serialized_bytes),
        "effective_adaptation_backend": learner.effective_adaptation_backend,
        "effective_response_backend": learner.effective_response_backend,
        "effective_outcome_backend": learner.effective_outcome_backend,
        "effective_competition_backend": learner.effective_competition_backend,
        "effective_byte_loop": (
            "resident_cpu_columns"
            if args.cpu_column_workers > 0
            else (
                "resident_cuda_developmental"
                if args.cuda_developmental
                else (
                    "compiled_chunk"
                    if learner.supports_compiled_chunks
                    else "python_step"
                )
            )
        ),
        "cpu_column_workers": args.cpu_column_workers,
        "cuda_developmental": args.cuda_developmental,
        "structural_wake_bytes": args.structural_wake_bytes,
        "population": population,
        "population_at_resume": population_at_resume,
        "population_delta": population_delta,
        "branch_evaluations_per_byte": population_delta["branch_evaluations"]
        / max(remaining_train_bytes, 1),
        "births_per_kilobyte": learner.stats.births
        / max(args.train_bytes / 1000.0, 1e-12),
        "prunes_per_kilobyte": learner.stats.prunes
        / max(args.train_bytes / 1000.0, 1e-12),
        "corpus_bytes_per_alm_byte": args.train_bytes / max(serialized_alm_bytes, 1),
        "live_relations": learner.live_relation_count,
        "live_nursery_candidates": learner.live_nursery_count,
        "anatomy": _anatomy_report(learner),
        "relations": _relation_report(learner),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if checkpoint_path is not None:
        learner.save_checkpoint(checkpoint_path)
        if progress_path is not None:
            _atomic_write_json(
                progress_path,
                {
                    "train_path": args.train_path,
                    "train_offset": args.train_offset,
                    "train_bytes_target": args.train_bytes,
                    "train_bytes_processed": args.train_bytes,
                    "learner_steps": learner.stats.steps,
                    "train_metrics": train_metrics.state_dict(),
                    "complete": True,
                },
            )
    if args.report is not None:
        _atomic_write_json(args.report, report)


if __name__ == "__main__":
    main()
