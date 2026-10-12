"""Section 23's "recommended first experiment" for MF-ALM-TS, run on the
real TinyStories corpus, CPU only.

    BUILD THIS FIRST: Train one 768-state plastic column plus its local
    token head on the 4K-token TinyStories stream. Use tied
    recognition/reconstruction weights, online variance control, and no MF
    bank. Compare its next-token loss and representation rank with the
    first block of a parameter-matched backprop model. Only add hierarchy
    and persistent memory after this primitive demonstrably learns.

This script does steps 1-3 of that section: build the corpus/tokenizer
manifests, train one column with local head + reconstruction updates only,
and report whether it's learning at all (see stream_trainer's
`sanity_gate_report` -- there's no fixed pass/fail percentage; see
docs/ManifoldFlow_ALM_Journal.md for why that was dropped). The backprop
comparison and hierarchy/MF steps are follow-up work.

Defaults now match the design's own D0 budget and reference config
(Appendix B: 10M tokens, state_dim 768, token_dim 384, vocab 4096) rather
than a CPU-feasibility-shrunk stand-in. At this scale a full run takes
hours on a CPU-only laptop, so the run is checkpointed: `--output-dir` is
also the resume point -- rerunning the same command after an interruption
(Ctrl-C, a reboot, a background-task timeout) picks back up from the last
checkpoint instead of starting over. The corpus/tokenizer/token shards are
cached in `--output-dir` too, so only the first run pays that cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from manifoldflow.alm.data.tinystories import (
    TokenShard,
    encode_shard,
    load_deduped_corpus,
)
from manifoldflow.alm.data.tokenizer import (
    StoryBpeTokenizer,
    tokenizer_manifest,
    write_manifest,
)
from manifoldflow.alm.eval.metrics import unigram_cross_entropy
from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig
from manifoldflow.alm.train.stream_trainer import (
    resume_or_init,
    sanity_gate_report,
    save_checkpoint,
    train_single_column,
)

DEFAULT_TRAIN_FILE = r"C:\path\to\gdrive\training_data\TinyStories-train.txt"
DEFAULT_VALID_FILE = r"C:\path\to\gdrive\training_data\TinyStories-valid.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE)
    parser.add_argument(
        "--story-limit",
        type=int,
        default=80_000,
        help="stories read from each source file (enough headroom for --tokens)",
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--state-dim", type=int, default=768)
    parser.add_argument("--token-dim", type=int, default=384)
    parser.add_argument(
        "--tokens",
        type=int,
        default=10_000_000,
        help="training tokens (design's D0 budget; see module docstring)",
    )
    parser.add_argument("--val-tokens", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20_000)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100_000,
        help="write a resumable checkpoint every N training tokens",
    )
    parser.add_argument("--output-dir", default="runs/alm_first_experiment")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="ignore cached corpus/tokenizer/shards in --output-dir and rebuild them",
    )
    return parser.parse_args()


def _load_or_build_corpus_and_tokenizer(
    args: argparse.Namespace, out_dir: Path
) -> tuple[StoryBpeTokenizer, TokenShard, TokenShard]:
    tokenizer_path = out_dir / "tokenizer.json"
    train_shard_path = out_dir / "train_shard.npz"
    valid_shard_path = out_dir / "valid_shard.npz"
    cached = (
        not args.force_rebuild
        and tokenizer_path.exists()
        and train_shard_path.exists()
        and valid_shard_path.exists()
    )
    if cached:
        print(f"[1-3/5] loading cached tokenizer + shards from {out_dir}...")
        tokenizer = StoryBpeTokenizer.load(tokenizer_path)
        train_shard = TokenShard.load(train_shard_path)
        valid_shard = TokenShard.load(valid_shard_path)
        assert train_shard.verify(), "cached train shard failed CRC check"
        assert valid_shard.verify(), "cached valid shard failed CRC check"
        print(
            f"  train tokens={len(train_shard.tokens)} "
            f"validation tokens={len(valid_shard.tokens)}"
        )
        return tokenizer, train_shard, valid_shard

    print(f"[1/5] loading + deduping corpus (story_limit={args.story_limit})...")
    corpus = load_deduped_corpus(
        {
            "tinystories_original_train": args.train_file,
            "tinystories_original_valid": args.valid_file,
        },
        val_fraction=0.02,
        limit_per_source=args.story_limit,
    )
    print(
        f"  train={len(corpus.train)} validation={len(corpus.validation)} "
        f"seen={corpus.n_seen} duplicates={corpus.n_duplicates}"
    )
    write_manifest(
        out_dir / "corpus_manifest.json",
        {
            "source_checksums": corpus.source_checksums,
            "n_train_stories": len(corpus.train),
            "n_validation_stories": len(corpus.validation),
            "n_seen": corpus.n_seen,
            "n_duplicates": corpus.n_duplicates,
        },
    )

    print(f"[2/5] training {args.vocab_size}-entry BPE tokenizer (train split only)...")
    tokenizer = StoryBpeTokenizer.train(corpus.train, vocab_size=args.vocab_size)
    tokenizer.save(tokenizer_path)
    manifest = tokenizer_manifest(tokenizer.vocab_size, "bpe")
    write_manifest(out_dir / "tokenizer_manifest.json", manifest)

    print("[3/5] encoding + caching shards...")
    train_shard = encode_shard(corpus.train, tokenizer)
    valid_shard = encode_shard(corpus.validation, tokenizer)
    train_shard.save(train_shard_path)
    valid_shard.save(valid_shard_path)
    print(
        f"  train tokens={len(train_shard.tokens)} "
        f"validation tokens={len(valid_shard.tokens)}"
    )
    return tokenizer, train_shard, valid_shard


def _evaluate(column: PlasticColumn, valid_stream: np.ndarray, eos_id: int) -> dict:
    val_ce_sum, val_n = 0.0, 0
    h_prev = torch.zeros(column.config.state_dim)
    with torch.no_grad():
        for t in range(len(valid_stream) - 1):
            token_t, token_next = int(valid_stream[t]), int(valid_stream[t + 1])
            e_t = column.E_token[token_t]
            h = column.forward(e_t, h_prev)
            p = column.token_probs(h)
            val_ce_sum += float(-torch.log(torch.clamp_min(p[token_next], 1e-12)))
            val_n += 1
            h_prev = torch.zeros(column.config.state_dim) if token_t == eos_id else h
    model_ce = val_ce_sum / max(val_n, 1)
    unigram_ce = unigram_cross_entropy(valid_stream, column.config.vocab_size)
    reduction = (unigram_ce - model_ce) / unigram_ce if unigram_ce > 0 else 0.0
    return {
        "model_token_ce": model_ce,
        "unigram_token_ce": unigram_ce,
        "cross_entropy_reduction": reduction,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    tokenizer, train_shard, valid_shard = _load_or_build_corpus_and_tokenizer(
        args, out_dir
    )
    if len(train_shard.tokens) < args.tokens:
        raise SystemExit(
            f"train shard has only {len(train_shard.tokens)} tokens, less than "
            f"--tokens={args.tokens}; increase --story-limit and rerun with "
            f"--force-rebuild (silently training on fewer tokens than requested "
            f"would misreport progress)"
        )
    train_stream = train_shard.tokens.astype(np.int64)[: args.tokens]
    valid_stream = valid_shard.tokens.astype(np.int64)[: args.val_tokens]

    print("[4/5] training one plastic column (local head + reconstruction only)...")
    config = PlasticColumnConfig(
        input_dim=args.token_dim,
        state_dim=args.state_dim,
        token_dim=args.token_dim,
        vocab_size=tokenizer.vocab_size,
        seed=args.seed,
    )
    checkpoint_path = out_dir / "checkpoint.pt"
    column, stats, start_index, h_prev = resume_or_init(
        checkpoint_path, config, generator=torch.Generator().manual_seed(args.seed)
    )
    if start_index > 0:
        print(f"  resuming from checkpoint at token {start_index}/{len(train_stream)}")

    stats, h_prev = train_single_column(
        column,
        train_stream,
        eos_id=tokenizer.ids.eos,
        log_every=args.log_every,
        start_index=start_index,
        stats=stats,
        h_prev=h_prev,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    save_checkpoint(
        checkpoint_path, column, stats, position=len(train_stream) - 1, h_prev=h_prev
    )
    train_report = sanity_gate_report(stats, train_stream, tokenizer.vocab_size)

    print("[5/5] evaluating on held-out validation stream (no weight updates)...")
    validation_report = _evaluate(column, valid_stream, tokenizer.ids.eos)

    report = {
        "elapsed_seconds": time.time() - t_start,
        "config": {
            **config.__dict__,
            "tokens": len(train_stream),
            "val_tokens": len(valid_stream),
        },
        "train": train_report,
        "training_log": stats.log,
        "validation": validation_report,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "train": train_report,
        "validation": validation_report,
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
