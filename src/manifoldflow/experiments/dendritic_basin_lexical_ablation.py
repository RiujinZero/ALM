"""Matched byte-versus-word/punctuation representation ablation on TinyStories."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.dendritic_basin.token_sequence_cortex import (
    LosslessLexicalCodec,
    TokenSequenceCortex,
)

DEFAULT_TRAIN = Path(r"C:\path\to\gdrive\training_data\TinyStories-train.txt")
DEFAULT_VALID = Path(r"C:\path\to\gdrive\training_data\TinyStories-valid.txt")
DEFAULT_PROMPTS = (
    b"Once upon a time, there was a little girl named Lily. ",
    b"Tom was very hungry, so he went to the kitchen. ",
)


def _read_prefix(path: Path, byte_count: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(byte_count)


def _bpb(probabilities: np.ndarray, source_bytes: int) -> tuple[float, float, int]:
    finite = probabilities[np.isfinite(probabilities)]
    bits = float(-np.log2(np.maximum(finite, 1e-300)).sum())
    return bits / source_bytes, bits, int(len(finite))


def _sample(
    distribution: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
) -> int:
    probability = np.asarray(distribution, dtype=np.float64).copy()
    if temperature <= 0.0:
        return int(np.argmax(probability))
    logits = np.log(np.maximum(probability, 1e-300)) / temperature
    if 0 < top_k < len(logits):
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits -= np.max(logits)
    probability = np.exp(logits)
    probability /= probability.sum()
    return int(rng.choice(len(probability), p=probability))


def _generate_bytes(
    source: SequenceCortexOpponentLearner,
    prompt: bytes,
    *,
    output_bytes: int,
    seed: int,
    temperature: float,
    top_k: int,
) -> str:
    learner = pickle.loads(pickle.dumps(source, protocol=5))
    learner.reset_episode()
    for value in prompt:
        learner.advance_observation(value, learn=False)
    rng = np.random.default_rng(seed)
    result = bytearray(prompt)
    for _ in range(output_bytes):
        token = _sample(learner.predict_next_distribution(), rng, temperature, top_k)
        result.append(token)
        learner.advance_observation(token, learn=False)
    return result.decode("utf-8", errors="replace")


def _decode_partial(codec: LosslessLexicalCodec, tokens: list[int]) -> bytes:
    output = bytearray()
    raw = False
    for token in tokens:
        if raw:
            if token == codec.RAW_END:
                raw = False
            elif token < 256:
                output.append(token)
        elif token == codec.RAW_BEGIN:
            raw = True
        elif token >= codec.FIRST_LEXICAL:
            unit = codec.id_to_unit[token]
            assert unit is not None
            output.extend(unit)
    return bytes(output)


def _generate_lexical(
    source: TokenSequenceCortex,
    codec: LosslessLexicalCodec,
    prompt: bytes,
    *,
    output_bytes: int,
    seed: int,
    temperature: float,
    top_k: int,
) -> str:
    learner = pickle.loads(pickle.dumps(source, protocol=5))
    learner.reset_episode()
    prompt_tokens = codec.encode(prompt)
    for token in prompt_tokens:
        learner.observe(int(token), learn=False)
    rng = np.random.default_rng(seed)
    generated: list[int] = []
    raw = False
    for _ in range(max(64, output_bytes * 3)):
        distribution = learner.distribution()
        if raw:
            distribution[codec.RAW_BEGIN] = 0.0
            distribution[codec.FIRST_LEXICAL :] = 0.0
        else:
            raw_begin_probability = distribution[codec.RAW_BEGIN]
            distribution[: codec.FIRST_LEXICAL] = 0.0
            distribution[codec.RAW_BEGIN] = raw_begin_probability
        total = distribution.sum()
        if total <= 0.0:
            break
        distribution /= total
        token = _sample(distribution, rng, temperature, top_k)
        generated.append(token)
        learner.observe(token, learn=False)
        if token == codec.RAW_BEGIN and not raw:
            raw = True
        elif token == codec.RAW_END and raw:
            raw = False
        if len(_decode_partial(codec, generated)) >= output_bytes:
            break
    text = prompt + _decode_partial(codec, generated)[:output_bytes]
    return text.decode("utf-8", errors="replace")


def _codec_metrics(codec: LosslessLexicalCodec, source: bytes) -> dict[str, Any]:
    tokens = codec.encode(source)
    raw_begins = int(np.sum(tokens == codec.RAW_BEGIN))
    return {
        "source_bytes": len(source),
        "encoded_tokens": int(len(tokens)),
        "tokens_per_source_byte": float(len(tokens) / max(1, len(source))),
        "raw_fallback_units": raw_begins,
        "raw_fallback_token_fraction": float(raw_begins / max(1, len(tokens))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    train = _read_prefix(args.train, args.train_bytes)
    valid = _read_prefix(args.valid, args.valid_bytes)
    codec = LosslessLexicalCodec.build(train, vocab_size=args.vocab_size)
    train_tokens = codec.encode(train)
    valid_tokens = codec.encode(valid)

    shared = dict(
        prior=args.prior,
        confirmations=args.confirmations,
        context_confirmations=args.context_confirmations,
        sequence_confirmations=args.sequence_confirmations,
        surprise_bits=args.surprise_bits,
        basal_strength=args.basal_strength,
        parent_strength=args.parent_strength,
    )
    byte_model = SequenceCortexOpponentLearner(
        **shared,
        max_context_bytes=args.max_context,
        packed_paths=True,
        max_sequence_nursery_paths=args.max_nursery_paths,
    )
    lexical_model = TokenSequenceCortex(
        codec.vocab_size,
        **shared,
        max_context_tokens=args.max_context,
        max_nursery_paths=args.max_nursery_paths,
    )

    started = time.perf_counter()
    byte_model.observe_chunk_probabilities(
        np.frombuffer(train, dtype=np.uint8), learn=True
    )
    byte_train_seconds = time.perf_counter() - started
    byte_model.reset_episode()
    started = time.perf_counter()
    byte_probability = byte_model.observe_chunk_probabilities(
        np.frombuffer(valid, dtype=np.uint8), learn=False
    )
    byte_eval_seconds = time.perf_counter() - started

    started = time.perf_counter()
    lexical_model.observe_chunk_probabilities(train_tokens, learn=True)
    lexical_train_seconds = time.perf_counter() - started
    sleep: dict[str, Any] | None = None
    if args.body_cap_bytes:
        byte_sleep = byte_model.consolidate_to_byte_budget(args.body_cap_bytes)
        lexical_budget = args.body_cap_bytes - codec.embodied_bytes
        if lexical_budget < 1:
            raise ValueError("body cap is smaller than the lexical codec")
        lexical_sleep = lexical_model.consolidate_to_byte_budget(lexical_budget)
        sleep = {"byte": byte_sleep, "lexical": lexical_sleep}
    lexical_model.reset_episode()
    started = time.perf_counter()
    lexical_probability = lexical_model.observe_chunk_probabilities(
        valid_tokens, learn=False
    )
    lexical_eval_seconds = time.perf_counter() - started

    byte_bpb, byte_bits, byte_scored = _bpb(byte_probability, len(valid))
    lexical_bpb, lexical_bits, lexical_scored = _bpb(
        lexical_probability, len(valid)
    )
    prompts = list(args.prompt) if args.prompt else list(DEFAULT_PROMPTS)
    generations: dict[str, Any] = {}
    for index, prompt in enumerate(prompts):
        generations[prompt.decode()] = {
            "byte_greedy": _generate_bytes(
                byte_model, prompt, output_bytes=args.generation_bytes,
                seed=args.seed + index, temperature=0.0, top_k=args.top_k,
            ),
            "lexical_greedy": _generate_lexical(
                lexical_model, codec, prompt, output_bytes=args.generation_bytes,
                seed=args.seed + index, temperature=0.0, top_k=args.top_k,
            ),
            "byte_sample": _generate_bytes(
                byte_model, prompt, output_bytes=args.generation_bytes,
                seed=args.seed + index, temperature=args.temperature, top_k=args.top_k,
            ),
            "lexical_sample": _generate_lexical(
                lexical_model, codec, prompt, output_bytes=args.generation_bytes,
                seed=args.seed + index, temperature=args.temperature, top_k=args.top_k,
            ),
        }

    return {
        "experiment": "lossless_word_punctuation_vs_raw_byte",
        "train_path": str(args.train),
        "valid_path": str(args.valid),
        "train_bytes": len(train),
        "valid_bytes": len(valid),
        "configuration": dict(vars(args))
        | {
            "train": str(args.train),
            "valid": str(args.valid),
            "output": str(args.output),
            "prompt": [value.decode("utf-8", "replace") for value in prompts],
        },
        "lexical_codec": {
            "vocab_size": codec.vocab_size,
            "embodied_bytes": codec.embodied_bytes,
            "train": _codec_metrics(codec, train),
            "validation": _codec_metrics(codec, valid),
        },
        "byte": {
            "bpb": byte_bpb,
            "total_bits": byte_bits,
            "scored_symbols": byte_scored,
            "body_bytes": byte_model.embodied_bytes,
            "train_seconds": byte_train_seconds,
            "train_source_bytes_per_second": len(train) / byte_train_seconds,
            "eval_seconds": byte_eval_seconds,
            "live_sequences": byte_model.live_sequence_count,
        },
        "lexical": {
            "bpb": lexical_bpb,
            "bits_per_token": lexical_bits / max(1, lexical_scored),
            "total_bits": lexical_bits,
            "scored_symbols": lexical_scored,
            "body_bytes_without_codec": lexical_model.embodied_bytes,
            "body_bytes_with_codec": (
                lexical_model.embodied_bytes + codec.embodied_bytes
            ),
            "train_seconds": lexical_train_seconds,
            "train_source_bytes_per_second": len(train) / lexical_train_seconds,
            "train_tokens_per_second": len(train_tokens) / lexical_train_seconds,
            "eval_seconds": lexical_eval_seconds,
            "live_sequences": lexical_model.live_sequence_count,
        },
        "delta_lexical_minus_byte_bpb": lexical_bpb - byte_bpb,
        "post_training_sleep": sleep,
        "generations": generations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-bytes", type=int, default=4_000_000)
    parser.add_argument("--valid-bytes", type=int, default=250_000)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--body-cap-bytes", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=6)
    parser.add_argument("--max-nursery-paths", type=int, default=50_000)
    parser.add_argument("--confirmations", type=int, default=3)
    parser.add_argument("--context-confirmations", type=int, default=3)
    parser.add_argument("--sequence-confirmations", type=int, default=3)
    parser.add_argument("--surprise-bits", type=float, default=2.0)
    parser.add_argument("--basal-strength", type=float, default=4.0)
    parser.add_argument("--parent-strength", type=float, default=16.0)
    parser.add_argument("--prior", type=float, default=0.1)
    parser.add_argument("--generation-bytes", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument(
        "--prompt",
        action="append",
        type=bytes,
        default=None,
        help="Generation prompt (repeatable); defaults to the built-in prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
