"""Run ALM-64MB-4C on chronological CLEAR or Stream-51 image streams."""

from __future__ import annotations

import argparse
import io
import json
import platform
import time
import zipfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from manifoldflow.dendritic_basin.continual_vision import (
    ALM64MB4CContinualVision,
    ALM64MB4CGrowingVision,
    ContinualVisionConfig,
)
from manifoldflow.dendritic_basin.native_vision import (
    NativeGrowingVision,
    native_vision_available,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("clear10", "stream51"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--sleep-interval", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-per-experience", type=int, default=0)
    parser.add_argument("--max-test-per-experience", type=int, default=0)
    parser.add_argument("--max-eval-per-experience", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--visual-anatomy", choices=("growing", "fixed"), default="growing"
    )
    parser.add_argument("--maximum-visual-assemblies", type=int, default=50_000)
    parser.add_argument(
        "--maximum-synapses",
        type=int,
        default=128,
        help=(
            "Per-assembly receptive field size (top-K evidence elements by "
            "magnitude). Does not need to scale with image_size in "
            "principle, but in practice a fixed synapse count covers a "
            "shrinking fraction of the evidence vector as image_size "
            "grows -- at image_size=24 (5,760 dims) 128 synapses is ~2.2%% "
            "coverage; at image_size=480 (2,304,000 dims) the same 128 is "
            "~0.006%%, an extremely sparse and noisy sample rather than a "
            "holistic one. See docs/ALM_V1_Architecture.md."
        ),
    )
    parser.add_argument("--visual-routing-width", type=int, default=16)
    parser.add_argument("--visual-birth-activation", type=float, default=0.72)
    parser.add_argument("--relation-turnover-fraction", type=float, default=0.0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument(
        "--eval-frequency", choices=("experience", "pass"), default="experience"
    )
    parser.add_argument(
        "--sleep-frequency", choices=("experience", "pass"), default="experience"
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help=(
            "run the C++ alm_vision port instead of the PyTorch reference "
            "(--visual-anatomy growing only; --checkpoint/--resume are "
            "supported -- see cpp/README.md for the build and the "
            "correctness contract)"
        ),
    )
    parser.add_argument(
        "--full-eval-every-n-passes",
        type=int,
        default=10,
        help=(
            "CLEAR-10 with --eval-frequency experience only: pass 1 always "
            "evaluates after every experience (needed to build a valid "
            "first-pass accuracy_matrix / clear_metrics_first_pass, which "
            "requires one evaluation snapshot per experience boundary "
            "within that first sweep). A full CLEAR-style evaluation "
            "rescans every held-out bucket, so it is expensive -- for pass "
            "2 onward, only evaluate once every N passes (plus always the "
            "final requested pass), rather than after every experience."
        ),
    )
    parser.add_argument(
        "--episodic-enabled",
        action="store_true",
        help=(
            "--native only: enable the sparse episodic recall layer -- a "
            "locality-sensitive-hash (SimHash) analogue of the byte "
            "cortex's episodic memory, validated standalone "
            "(vision_lsh_recall_probe.py) since continuous evidence "
            "vectors are never bit-identical across two occurrences the "
            "way byte sequences are. Off by default. See VisionConfig in "
            "vision.hpp and docs/ALM_V1_Architecture.md."
        ),
    )
    parser.add_argument("--episodic-lsh-bits", type=int, default=16)
    parser.add_argument("--episodic-probe-radius", type=int, default=2)
    parser.add_argument("--episodic-chain-slots", type=int, default=4)
    parser.add_argument("--episodic-tier3-refresh-threshold", type=int, default=3)
    parser.add_argument("--episodic-strength", type=float, default=0.1)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=20260911,
        help=(
            "CLEAR-10 only: load_clear10 concatenates each experience's "
            "samples class-by-class (all of class 0, then all of class 1, "
            "...), which is not a natural photo order and makes the "
            "training-time prequential accuracy meaningless (it just rides "
            "long same-class runs). Training order within each experience "
            "is shuffled with this seed before training; the held-out "
            "accuracy_matrix is unaffected either way since it sums over a "
            "full bucket regardless of order."
        ),
    )
    return parser


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    bbox: tuple[float, float, float, float] | None = None
    archive: Path | None = None


def _evenly_capped(
    samples: list[ImageRecord], limit: int
) -> list[ImageRecord]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    indices = np.linspace(0, len(samples) - 1, num=limit, dtype=np.int64)
    return [samples[int(index)] for index in indices]


def load_clear10(
    root: Path, split: str, *, limit_per_experience: int = 0
) -> tuple[list[str], list[tuple[str, list[ImageRecord]]]]:
    """Read official CLEAR metadata without importing a framework."""
    split_root = root / split
    if not split_root.exists() and (root / "clear10" / split).exists():
        split_root = root / "clear10" / split
    if not split_root.exists():
        archive = root / (
            "clear10-train-image-only.zip" if split == "train" else "clear10-test.zip"
        )
        if archive.exists():
            return _load_clear10_archive(
                archive, split, limit_per_experience=limit_per_experience
            )
    metadata_path = split_root / "labeled_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing CLEAR metadata: {metadata_path}")
    class_names = [
        value for value in (split_root / "class_names.txt").read_text().splitlines()
        if value
    ]
    labeled = json.loads(metadata_path.read_text(encoding="utf-8"))
    experiences: list[tuple[str, list[ImageRecord]]] = []
    for bucket, class_metadata in labeled.items():
        samples: list[ImageRecord] = []
        for class_id, class_name in enumerate(class_names):
            item_path = split_root / class_metadata[class_name]
            items = json.loads(item_path.read_text(encoding="utf-8"))
            samples.extend(
                ImageRecord(split_root / item["IMG_PATH"], class_id)
                for item in items.values()
            )
        experiences.append((str(bucket), _evenly_capped(samples, limit_per_experience)))
    return class_names, experiences


def _load_clear10_archive(
    archive: Path, split: str, *, limit_per_experience: int
) -> tuple[list[str], list[tuple[str, list[ImageRecord]]]]:
    prefix = "train_image_only" if split == "train" else "test"
    with zipfile.ZipFile(archive) as handle:
        class_names = [
            value
            for value in handle.read(f"{prefix}/class_names.txt")
            .decode("utf-8")
            .splitlines()
            if value
        ]
        labeled = json.loads(
            handle.read(f"{prefix}/labeled_metadata.json").decode("utf-8")
        )
        experiences = []
        for bucket, class_metadata in labeled.items():
            samples = []
            for class_id, class_name in enumerate(class_names):
                metadata_member = str(
                    PurePosixPath(prefix) / class_metadata[class_name]
                )
                items = json.loads(handle.read(metadata_member).decode("utf-8"))
                samples.extend(
                    ImageRecord(
                        Path(item["IMG_PATH"]),
                        class_id,
                        archive=archive,
                    )
                    for item in items.values()
                )
            experiences.append(
                (str(bucket), _evenly_capped(samples, limit_per_experience))
            )
    return class_names, experiences


def load_stream51(
    root: Path, split: str, *, limit_per_experience: int = 0
) -> tuple[list[str], list[tuple[str, list[ImageRecord]]]]:
    """Read an official Stream-51 JSON ordering and preserve its exact order."""
    ordering_root = root / "orderings" / "instance_seed10"
    ordering_files = (
        [ordering_root / "train.txt"]
        if split == "train"
        else sorted(ordering_root.glob("test_[0-9]*.txt"))
    )
    if not ordering_files or not all(path.exists() for path in ordering_files):
        raise FileNotFoundError(f"no Stream-51 {split!r} ordering under {root}")
    metadata_name = (
        "Stream-51_meta_train.json" if split == "train" else "Stream-51_meta_test.json"
    )
    metadata_path = root / metadata_name
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    row_by_path = {str(row[-1]).replace("\\", "/"): row for row in rows}
    experiences = []
    for ordering_file in ordering_files:
        samples = []
        for line in ordering_file.read_text(encoding="utf-8").splitlines():
            relative, label_text = line.rsplit(maxsplit=1)
            row = row_by_path[relative.replace("\\", "/")]
            samples.append(
                ImageRecord(
                    root / relative,
                    int(label_text),
                    tuple(float(value) for value in row[-2]),
                )
            )
        if split == "train":
            checkpoints = [30_000, 60_000, 90_000, 120_000, 150_000, len(samples)]
            start = 0
            for stop in checkpoints:
                chunk = _evenly_capped(samples[start:stop], limit_per_experience)
                experiences.append((f"train_{stop:06d}", chunk))
                start = stop
        else:
            experiences.append(
                (
                    ordering_file.stem,
                    _evenly_capped(samples, limit_per_experience),
                )
            )
    class_count = max(
        sample.label for _, samples in experiences for sample in samples
    ) + 1
    return [str(index) for index in range(class_count)], experiences


@lru_cache(maxsize=4)
def _zip_handle(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def _open_image(record: ImageRecord) -> Image.Image:
    if record.archive is None:
        source: Path | io.BytesIO = record.path
    else:
        prefix = "train_image_only" if "train" in record.archive.name else "test"
        member = str(PurePosixPath(prefix) / record.path.as_posix())
        source = io.BytesIO(_zip_handle(record.archive).read(member))
    with Image.open(source) as image:
        image = image.convert("RGB")
        if record.bbox is None:
            return image
        x_max, x_min, y_max, y_min = record.bbox
        center_x = x_min + (x_max - x_min) / 2.0
        center_y = y_min + (y_max - y_min) / 2.0
        half_width = (x_max - x_min) * 1.10 / 2.0
        half_height = (y_max - y_min) * 1.10 / 2.0
        return image.crop(
            (
                max(int(center_x - half_width), 0),
                max(int(center_y - half_height), 0),
                min(int(center_x + half_width), image.width),
                min(int(center_y + half_height), image.height),
            )
        )


class NativeGrowingVisionAdapter:
    """Adapts ``NativeGrowingVision`` (ctypes over ``cpp/vision.hpp``) to the
    same duck-typed surface this harness already trains against
    (``ALM64MB4CGrowingVision``), so the training/eval loop below does not
    need to know which backend it is driving.

    See ``cpp/README.md`` and ``tests/test_native_vision_matches_python.py``
    for the correctness contract: evidence matches the PyTorch reference
    tightly and deterministically; per-step decisions can still diverge
    later in a stream (inherent top-k sensitivity, not a directional bug);
    aggregate quality tracks the reference within a wide band.

    Checkpoint/resume works the same way the byte cortex's does:
    ``NativeGrowingVision.__reduce__`` (native_vision.py) makes it directly
    picklable via its own binary save/load format
    (cpp/include/alm/vision_serialize.hpp), so ``torch.save``/``torch.load``
    round-trip this adapter transparently -- ``_native`` and ``_config``
    are its only state, and both are picklable on their own.
    """

    def __init__(self, native: NativeGrowingVision, config: ContinualVisionConfig) -> None:
        self._native = native
        self._config = config

    def _resized_rgb(self, image: Image.Image) -> np.ndarray:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return (
            np.asarray(
                image.convert("RGB").resize(
                    (self._config.image_size, self._config.image_size), resampling
                ),
                dtype=np.float32,
            )
            / 255.0
        )

    def predict(self, image: Image.Image) -> tuple[int, np.ndarray, np.ndarray]:
        evidence = self._native.evidence(self._resized_rgb(image))
        # Single-column votes, not the 4-column ensemble: measured identical
        # argmax predictions on every held-out eval across a 7-pass CLEAR-10
        # A/B (runs/alm_clear10_ensemble_vs_single/) and 0/200 flips on
        # synthetic evidence, at 2.67x the calls/s -- because between sleep()
        # merges, 3 of the 4 columns are near-identical stale copies of the
        # same base rows, so ensembling mostly just rescales votes uniformly
        # rather than changing which class wins. See
        # predict_evidence_single_column in vision.hpp.
        prediction, votes = self._native.predict_single_column(evidence)
        return prediction, votes, evidence

    def observe_external(self, evidence: np.ndarray, label: int) -> None:
        self._native.observe_external(evidence, label)

    def reset_episode(self) -> None:
        self._native.reset_episode()

    def sleep(self) -> None:
        self._native.sleep()

    @property
    def device(self) -> str:
        return "cpu (alm_vision native)"

    @property
    def columns(self) -> list[None]:
        return [None] * self._config.columns

    @property
    def embodied_bytes(self) -> int:
        # Native's own serialized-row byte count, not a pickle byte count
        # (no native serialize path exists yet) -- a different unit from
        # ALM64MB4CGrowingVision.embodied_bytes, same disclosed departure
        # as the byte cortex's native/pickle byte-count split.
        return self._native.memory_bytes

    @property
    def sleep_cycles(self) -> int:
        return self._native.sleep_cycles

    @property
    def external_updates(self) -> int:
        return self._native.steps

    @property
    def active_synapses(self) -> int:
        return self._native.live_assemblies * self._config.maximum_synapses

    @property
    def live_assemblies(self) -> int:
        return self._native.live_assemblies

    @property
    def visual_births(self) -> int:
        return self._native.visual_births

    @property
    def relation_turnover_releases(self) -> int:
        return self._native.relation_turnover_releases


def _evaluate(
    learner: ALM64MB4CContinualVision | ALM64MB4CGrowingVision | NativeGrowingVisionAdapter,
    experiences: list[tuple[str, list[ImageRecord]]],
) -> dict[str, object]:
    correct = 0
    count = 0
    by_experience = []
    for name, samples in experiences:
        learner.reset_episode()
        local_correct = 0
        for sample in samples:
            prediction, _, _ = learner.predict(_open_image(sample))
            local_correct += int(prediction == sample.label)
        correct += local_correct
        count += len(samples)
        by_experience.append(
            {
                "experience": name,
                "samples": len(samples),
                "accuracy": local_correct / max(len(samples), 1),
            }
        )
    return {
        "samples": count,
        "accuracy": correct / max(count, 1),
        "by_experience": by_experience,
    }


def _clear_metrics(matrix: list[list[float]]) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    rows, columns = values.shape
    if rows != columns:
        raise ValueError("CLEAR accuracy matrix must be square")
    diagonal = [values[index, index] for index in range(rows)]
    lower = [values[i, j] for i in range(rows) for j in range(i + 1)]
    backward = [values[i, j] for i in range(rows) for j in range(i)]
    forward = [values[i, j] for i in range(rows) for j in range(i + 1, rows)]
    next_domain = [values[i, i + 1] for i in range(rows - 1)]
    return {
        "in_domain_accuracy": float(np.mean(diagonal)),
        "next_domain_accuracy": float(np.mean(next_domain)),
        "seen_domain_accuracy": float(np.mean(lower)),
        "backward_transfer_accuracy": float(np.mean(backward)),
        "forward_transfer_accuracy": float(np.mean(forward)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    loader = load_clear10 if args.benchmark == "clear10" else load_stream51
    class_names, train_only = loader(
        args.data_root, "train", limit_per_experience=args.max_train_per_experience
    )
    test_names, test_only = loader(
        args.data_root, "test", limit_per_experience=args.max_test_per_experience
    )
    if class_names != test_names:
        raise ValueError("training and test class vocabularies differ")
    if args.benchmark == "clear10":
        # A genuinely held-out evaluation set: CLEAR-10 ships separate
        # train/test splits per bucket, but the official "streaming"
        # protocol (tmp/avalanche-official/examples/clear.py:
        # `EVALUATION_PROTOCOL = "streaming"  # trainset = testset per
        # timestamp`) merges them, so the paper's own in-domain accuracy is
        # evaluated on images the model just trained on. That is the
        # official convention, not a bug -- but it is not what we want
        # here: train only on train_only, evaluate only on the disjoint
        # test_only images the organism has never seen. Bucket "0" exists
        # only in test_only (no matching train bucket in CLEAR-10) and is
        # dropped so every remaining bucket has a matched, disjoint
        # train/test pair and the accuracy matrix stays square.
        test_by_name = dict(test_only)
        train = [
            (name, samples) for name, samples in train_only if name in test_by_name
        ]
        # load_clear10 concatenates each experience's samples class-by-class
        # (all of class 0, then all of class 1, ...) -- not a natural photo
        # order, and it makes the training-time prequential accuracy
        # meaningless (a model just rides long same-class runs rather than
        # discriminating). Shuffle per experience with a fixed, reproducible
        # seed. The held-out evaluation below is unaffected either way,
        # since it sums correct/total over a full bucket regardless of order.
        shuffle_rng = np.random.default_rng(args.shuffle_seed)
        train = [
            (name, [samples[index] for index in shuffle_rng.permutation(len(samples))])
            for name, samples in train
        ]
        test = [
            (name, _evenly_capped(test_by_name[name], args.max_eval_per_experience))
            for name, _ in train
        ]
    else:
        train = train_only
        test = [
            (name, _evenly_capped(samples, args.max_eval_per_experience))
            for name, samples in test_only
        ]
    config = ContinualVisionConfig(
        classes=len(class_names),
        image_size=args.image_size,
        sleep_interval=args.sleep_interval,
        maximum_visual_assemblies=args.maximum_visual_assemblies,
        maximum_synapses=args.maximum_synapses,
        visual_routing_width=args.visual_routing_width,
        visual_birth_activation=args.visual_birth_activation,
        relation_turnover_fraction=args.relation_turnover_fraction,
    )
    if args.native:
        if args.visual_anatomy != "growing":
            raise ValueError("--native only supports --visual-anatomy growing")
        if not native_vision_available():
            raise RuntimeError(
                "--native requires the alm_vision shared library; see cpp/README.md"
            )
        native = NativeGrowingVision(
            classes=config.classes,
            image_size=config.image_size,
            columns=config.columns,
            context_decay=config.context_decay,
            sleep_interval=config.sleep_interval,
            max_body_bytes=config.max_body_bytes,
            learning_rate=config.learning_rate,
            maximum_synapses=config.maximum_synapses,
            maximum_visual_assemblies=config.maximum_visual_assemblies,
            visual_birth_activation=config.visual_birth_activation,
            visual_routing_width=config.visual_routing_width,
            relation_sketch_words=config.relation_sketch_words,
            relation_turnover_fraction=config.relation_turnover_fraction,
            episodic_enabled=args.episodic_enabled,
            episodic_lsh_bits=args.episodic_lsh_bits,
            episodic_probe_radius=args.episodic_probe_radius,
            episodic_chain_slots=args.episodic_chain_slots,
            episodic_tier3_refresh_threshold=args.episodic_tier3_refresh_threshold,
            episodic_strength=args.episodic_strength,
        )
        learner: (
            ALM64MB4CContinualVision | ALM64MB4CGrowingVision | NativeGrowingVisionAdapter
        ) = NativeGrowingVisionAdapter(native, config)
    else:
        learner_type = (
            ALM64MB4CGrowingVision
            if args.visual_anatomy == "growing"
            else ALM64MB4CContinualVision
        )
        learner = learner_type(config, device=args.device)
    start_pass = 0
    prior_elapsed = 0.0
    train_count = 0
    train_correct = 0
    train_results: list[dict[str, object]] = []
    retention_matrix: list[dict[str, object]] = []
    if args.resume:
        if args.checkpoint is None:
            raise ValueError("--resume requires --checkpoint")
        payload = torch.load(
            args.checkpoint, map_location=args.device, weights_only=False
        )
        if payload.get("format") != "alm-continual-vision-v1":
            raise ValueError("unsupported continual-vision checkpoint format")
        if payload.get("benchmark") != args.benchmark:
            raise ValueError("checkpoint benchmark does not match requested benchmark")
        if payload.get("config") != asdict(config):
            raise ValueError("checkpoint organism configuration does not match")
        learner = payload["learner"]
        start_pass = int(payload["completed_passes"])
        prior_elapsed = float(payload.get("elapsed_seconds", 0.0))
        train_count = int(payload["train_count"])
        train_correct = int(payload["train_correct"])
        train_results = list(payload["train_results"])
        retention_matrix = list(payload["retention_matrix"])
        if start_pass >= args.passes:
            raise ValueError("checkpoint has already reached requested passes")
        print(
            f"[{args.benchmark}] resumed after pass {start_pass} at "
            f"{train_count:,} samples",
            flush=True,
        )
    started = time.perf_counter()
    for pass_index in range(start_pass, args.passes):
        for experience_index, (name, samples) in enumerate(train):
            learner.reset_episode()
            local_correct = 0
            local_started = time.perf_counter()
            for sample in samples:
                prediction, _, evidence = learner.predict(_open_image(sample))
                local_correct += int(prediction == sample.label)
                learner.observe_external(evidence, sample.label)
                train_count += 1
                train_correct += int(prediction == sample.label)
                if args.progress_every and train_count % args.progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{args.benchmark}] {train_count:,} samples | "
                        f"{train_count / max(elapsed, 1e-9):.1f} samples/s | "
                        f"prequential "
                        f"{train_correct / max(train_count, 1):.4f}",
                        flush=True,
                    )
            if (
                args.sleep_frequency == "experience"
                or experience_index == len(train) - 1
            ):
                learner.sleep()
            train_results.append(
                {
                    "pass": pass_index + 1,
                    "experience": name,
                    "samples": len(samples),
                    "prequential_accuracy": local_correct / max(len(samples), 1),
                    "samples_per_second": len(samples)
                    / max(time.perf_counter() - local_started, 1e-9),
                    "embodied_bytes": learner.embodied_bytes,
                    "live_visual_assemblies": getattr(
                        learner, "live_assemblies", None
                    ),
                }
            )
            is_last_experience = experience_index == len(train) - 1
            should_eval = False
            if args.eval_frequency == "experience":
                if pass_index == 0:
                    # Every experience boundary in the first sweep, so the
                    # first-pass accuracy_matrix has one row per experience
                    # -- required for a valid clear_metrics_first_pass.
                    should_eval = True
                elif is_last_experience:
                    pass_number = pass_index + 1
                    should_eval = (
                        pass_number % max(args.full_eval_every_n_passes, 1) == 0
                        or pass_index == args.passes - 1
                    )
            elif args.eval_frequency == "pass":
                should_eval = is_last_experience
            if should_eval:
                evaluation = _evaluate(learner, test)
                evaluation["after_pass"] = pass_index + 1
                evaluation["after_experience"] = name
                retention_matrix.append(evaluation)
                progress_path = args.report.with_suffix(".progress.json")
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(
                    json.dumps(
                        {
                            "benchmark": args.benchmark,
                            "completed_passes": pass_index + 1,
                            "requested_passes": args.passes,
                            "training_samples": train_count,
                            "prequential_accuracy": train_correct
                            / max(train_count, 1),
                            "latest_evaluation": evaluation,
                            "evaluation_curve": retention_matrix,
                            "embodied_bytes": learner.embodied_bytes,
                            "live_visual_assemblies": getattr(
                                learner, "live_assemblies", None
                            ),
                            "sleep_cycles": learner.sleep_cycles,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "format": "alm-continual-vision-v1",
                    "benchmark": args.benchmark,
                    "config": asdict(config),
                    "completed_passes": pass_index + 1,
                    "learner": learner,
                    "train_count": train_count,
                    "train_correct": train_correct,
                    "train_results": train_results,
                    "retention_matrix": retention_matrix,
                    "elapsed_seconds": prior_elapsed
                    + time.perf_counter()
                    - started,
                },
                temporary,
            )
            temporary.replace(args.checkpoint)

    final_test = retention_matrix[-1]
    elapsed = prior_elapsed + time.perf_counter() - started
    result: dict[str, object] = {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "protocol": {
            "stream": "chronological official ordering",
            "passes": args.passes,
            "evaluation_frequency": args.eval_frequency,
            "sleep_frequency": args.sleep_frequency,
            "prediction_timing": "before label reveal",
            "task_ids_exposed": False,
            "replay_examples": 0,
            "pretrained_encoder": False,
            "input": "raw resized RGB pixels plus immediate local contrasts",
            "selection_on_test": False,
            "sample_cap_is_evenly_spaced": True,
            "clear_test_stream_duplicates_train_stream": args.benchmark
            == "clear10",
        },
        "architecture": f"ALM-64MB-4C {args.visual_anatomy}-vision adapter",
        "architecture_phase": (
            "accepted four-column wake/sleep lifecycle with shared "
            "basal/apical receiving assemblies"
        ),
        "config": asdict(config),
        "class_names": class_names,
        "training": {
            "samples": train_count,
            "prequential_accuracy": train_correct / max(train_count, 1),
            "by_experience": train_results,
        },
        "retention_after_each_experience": retention_matrix,
        "accuracy_matrix": [
            [
                float(item["accuracy"])
                for item in stage["by_experience"]  # type: ignore[index]
            ]
            for stage in retention_matrix
        ],
        "final_test": final_test,
        "body": {
            "class_ceiling_bytes": config.max_body_bytes,
            "embodied_bytes": learner.embodied_bytes,
            "utilization": learner.embodied_bytes / config.max_body_bytes,
            "columns": len(learner.columns),
            "sleep_cycles": learner.sleep_cycles,
            "external_updates": learner.external_updates,
            "active_synapses": learner.active_synapses,
            "live_visual_assemblies": getattr(learner, "live_assemblies", None),
            "visual_births": getattr(learner, "visual_births", None),
            "relation_turnover_releases": getattr(
                learner, "relation_turnover_releases", None
            ),
        },
        "systems": {
            "cortical_device": str(learner.device),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "elapsed_seconds": elapsed,
            "end_to_end_samples_per_second": (train_count + int(final_test["samples"]))
            / max(elapsed, 1e-9),
        },
    }
    if args.benchmark == "clear10" and args.eval_frequency == "experience":
        official_matrix = result["accuracy_matrix"][: len(test)]  # type: ignore[index]
        result["clear_metrics_first_pass"] = _clear_metrics(official_matrix)
        if args.passes == 1:
            result["clear_metrics"] = result["clear_metrics_first_pass"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
