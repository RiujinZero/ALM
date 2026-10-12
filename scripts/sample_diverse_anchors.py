#!/usr/bin/env python
"""Samples a diverse pool of short anchor statements for
`generate_contrastive_pairs.py`, across every real text source available
on the G drive plus the existing Qwen-labeled corpus - not just the
existing corpus's own (still fairly narrow) labeled spans. Motivated
directly by the round-trip test's finding: contrastive training on 1,540
pairs drawn only from the existing corpus improved in-batch accuracy
(0.19 -> 0.68) but *regressed* on out-of-distribution test cases (0.500
-> 0.300 retrieval accuracy) - a real, measured sign that this dataset's
source diversity, not the training method, was the bottleneck.

Each source is a plain `{"text": "..."}` JSONL (wikipedia, LSAT, formal
logic, math CoT, several chat styles) or plain text (GSM8K, TinyStories,
Python source) - extracted into short (roughly one-sentence) candidate
anchors via a simple regex splitter, not a full NLP sentence tokenizer
(this doesn't need to be linguistically perfect, just short and usable -
Qwen's own paraphrase/hard-negative prompt tolerates an imperfectly cut
sentence fine, and a bad split just becomes a skipped/failed generation
downstream, at negligible cost given how much source material exists).

Output: one anchor per line (`--output`), each also carrying its source
name as a JSONL record instead of plain text, so a later audit can see
which source library any given anchor/pair came from.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_MIN_LEN = 15
_MAX_LEN = 220
# Markdown/LaTeX/formula-fragment artifacts from the sentence splitter
# hitting non-prose structure (headers, boxed answers, table cells) -
# not real statements, and Qwen's paraphrase/hard-negative prompt (built
# for natural-language facts) handles these badly. A cheap heuristic
# filter, not a real parser - good enough to cut the obvious cases
# without trying to perfectly clean every source.
_ARTIFACT_PATTERNS = [
    re.compile(r"^#{1,6}\s"),  # markdown headers
    re.compile(r":\s*$"),  # trailing-colon labels/headers
    re.compile(r"\*\*\s*$"),  # trailing bold-markdown fragments
    re.compile(r"\\[a-zA-Z]{2,}"),  # LaTeX commands (\boxed, \frac, ...)
    re.compile(r"^[\d\s.\-+*/=()]+$"),  # pure arithmetic, no words at all
]


@dataclass
class SourceSpec:
    name: str
    path: Path
    kind: str  # "jsonl_text" | "plain_text" | "labeled_spans"
    target_count: int


def _is_artifact(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _ARTIFACT_PATTERNS)


def _candidate_sentences(blob: str) -> Iterator[str]:
    for chunk in blob.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sentence in _SENTENCE_SPLIT.split(chunk):
            sentence = sentence.strip().strip('"')
            if _MIN_LEN <= len(sentence) <= _MAX_LEN and not _is_artifact(sentence):
                yield sentence


def _sample_from_jsonl_text(path: Path, target_count: int, generator: random.Random) -> list[str]:
    """Reservoir-samples candidate sentences while streaming - these files
    can be large (wikipedia_en.jsonl especially), so this never holds the
    whole file's sentence pool in memory at once."""
    reservoir: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if not text:
                continue
            for sentence in _candidate_sentences(text):
                seen += 1
                if len(reservoir) < target_count:
                    reservoir.append(sentence)
                else:
                    j = generator.randint(0, seen - 1)
                    if j < target_count:
                        reservoir[j] = sentence
            if seen >= target_count * 200:
                # Enough of the file has been scanned for a representative
                # sample without reading multi-GB files (wikipedia) to
                # completion - diminishing returns past this point.
                break
    return reservoir


def _sample_from_plain_text(path: Path, target_count: int, generator: random.Random) -> list[str]:
    reservoir: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for sentence in _candidate_sentences(line):
                seen += 1
                if len(reservoir) < target_count:
                    reservoir.append(sentence)
                else:
                    j = generator.randint(0, seen - 1)
                    if j < target_count:
                        reservoir[j] = sentence
            if seen >= target_count * 200:
                break
    return reservoir


def _sample_from_labeled_spans(path: Path, target_count: int, generator: random.Random) -> list[str]:
    seen_texts: set[str] = set()
    anchors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for span in record.get("spans") or []:
                text = (span.get("span_text") or "").strip()
                if not text or len(text) < 8 or text in seen_texts:
                    continue
                seen_texts.add(text)
                anchors.append(text)
    generator.shuffle(anchors)
    return anchors[:target_count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/gdrive/training_data"),
        help="Root under which every source path below is resolved.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/salience_labels/qwen_pass1.jsonl"),
        help="The existing Qwen-labeled corpus - sampled as one source "
        "among many now, not the sole anchor pool.",
    )
    args = parser.parse_args()

    root = args.data_root
    sources = [
        SourceSpec("existing_corpus", args.labels_path, "labeled_spans", 3000),
        SourceSpec("wikipedia", root / "knowledge" / "wikipedia_en.jsonl", "jsonl_text", 4000),
        SourceSpec("openscience", root / "science" / "openscience.jsonl", "jsonl_text", 3000),
        SourceSpec("lsat_lr", root / "reasoning" / "lsat_lr.jsonl", "jsonl_text", 1200),
        SourceSpec("lsat_ar", root / "reasoning" / "lsat_ar.jsonl", "jsonl_text", 900),
        SourceSpec("lsat_rc", root / "reasoning" / "lsat_rc.jsonl", "jsonl_text", 900),
        SourceSpec("synlogic_easy", root / "reasoning" / "synlogic_easy.jsonl", "jsonl_text", 1200),
        SourceSpec("synlogic_hard", root / "reasoning" / "synlogic_hard.jsonl", "jsonl_text", 1200),
        SourceSpec("proofwriter", root / "reasoning" / "proofwriter.jsonl", "jsonl_text", 3000),
        SourceSpec("numina_math_cot", root / "math" / "numina_math_cot.jsonl", "jsonl_text", 2500),
        SourceSpec("tulu3_sft", root / "chat" / "tulu3_sft.jsonl", "jsonl_text", 3000),
        SourceSpec("smoltalk_everyday", root / "chat" / "smoltalk_everyday.jsonl", "jsonl_text", 2500),
        SourceSpec("smoltalk_openthoughts", root / "chat" / "smoltalk_openthoughts_sft.jsonl", "jsonl_text", 2000),
        SourceSpec("lmsys_chat_1m", root / "chat" / "lmsys_chat_1m.jsonl", "jsonl_text", 2000),
        SourceSpec("gsm8k", root / "math" / "gsm8k.txt", "plain_text", 2000),
        SourceSpec("math_dataset", root / "math" / "math_dataset.txt", "plain_text", 1500),
        SourceSpec("tinystories", root / "TinyStories-train.txt", "plain_text", 1500),
        SourceSpec("python_code", root / "code" / "python_code.txt", "plain_text", 1000),
    ]

    generator = random.Random(args.seed)
    all_anchors: list[tuple[str, str]] = []  # (source_name, anchor_text)
    for spec in sources:
        if not spec.path.exists():
            print(f"skip {spec.name}: {spec.path} not found", flush=True)
            continue
        if spec.kind == "labeled_spans":
            sampled = _sample_from_labeled_spans(spec.path, spec.target_count, generator)
        elif spec.kind == "jsonl_text":
            sampled = _sample_from_jsonl_text(spec.path, spec.target_count, generator)
        else:
            sampled = _sample_from_plain_text(spec.path, spec.target_count, generator)
        print(f"{spec.name}: sampled {len(sampled)}/{spec.target_count}", flush=True)
        all_anchors.extend((spec.name, text) for text in sampled)

    # Dedupe across sources (some overlap is possible, e.g. lmsys_chat_1m
    # vs the existing corpus, which was itself partly sourced from it).
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for source_name, text in all_anchors:
        if text in seen:
            continue
        seen.add(text)
        deduped.append((source_name, text))
    generator.shuffle(deduped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for source_name, text in deduped:
            handle.write(json.dumps({"source": source_name, "anchor": text}))
            handle.write("\n")

    by_source: dict[str, int] = {}
    for source_name, _ in deduped:
        by_source[source_name] = by_source.get(source_name, 0) + 1
    print(f"\n{len(deduped)} total anchors (after cross-source dedup) -> {args.output}", flush=True)
    for name, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count}", flush=True)


if __name__ == "__main__":
    main()
