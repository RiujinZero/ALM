#!/usr/bin/env python
"""Prunes assistant-self-identification and generic-disclaimer boilerplate
out of a Qwen-labeled salience file's positive spans - the curation gap
recorded (but never implemented) in docs/Salient_Gate_Dataset_Runbook.md's
"Curation decisions for the merge step" section, and confirmed as a real
problem via scripts/inspect_gate_writes.py's spot-check: several of the
gate's real write-time selections were near-verbatim assistant self-ID
("I am NAME_1, an AI assistant created by Anthropic...") rather than
genuine per-conversation salience. Qwen's own rubric explicitly calls
self-identification "durable and worth writing," so every duplicate of
this boilerplate across `lmsys_chat_1m`-sourced sessions became another
positive training example, teaching the gate that the *pattern* itself
(not the specific fact) is what's salient.

Does NOT touch distractor-statement spans (the calibration anchor) or
correct/negative labeling elsewhere - only removes matched *positive*
spans from *this* label file's own text. A record left with zero spans
after pruning is kept in the output (not deleted) with an empty spans
list - load_salience_label_rows already skips zero-span rows, so this
naturally drops out of gate training without needing a second filtering
step, and the record's provenance (shard_path/session_id/window range)
stays inspectable for anyone auditing what got pruned and why.

Rewrites in place by default (original saved alongside as *.pre_prune.bak
unless --no-backup) since this operates on the single shared label file
every training run reads from - not a new parallel output a run would
need to be repointed at.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# Matched case-insensitively against each span's own span_text. Deliberately
# narrow and high-confidence (verbatim self-identification / stock AI
# disclaimers actually observed in real spot-checked output) rather than
# a broad heuristic that risks pruning genuine content - see this file's
# own module docstring for why breadth was not the goal here.
_BOILERPLATE_PATTERNS = [
    # "I am NAME_1, an AI assistant/language model..." / "My name is Claude."
    re.compile(
        r"\b(i am|i'm|my name is)\b[^.]{0,60}\b(an? )?(ai|artificial intelligence|"
        r"language model|machine learning model|ai assistant)\b",
        re.IGNORECASE,
    ),
    # "created by <company>" / "trained by researchers from <org>"
    re.compile(
        r"\b(created|trained|developed)\s+by\b[^.]{0,80}"
        r"(anthropic|openai|google|meta|lmsys|large model systems)",
        re.IGNORECASE,
    ),
    # Stock capability disclaimers - "I'm sorry, I don't have access to..."
    re.compile(
        r"\bi('m| am) sorry\b[^.]{0,40}\b(i (do not|don't)|i am not able|"
        r"as an ai\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi (do not|don't) (actually )?(experience|have access to|have the ability to)\b",
        re.IGNORECASE,
    ),
]


def _is_boilerplate(span_text: str) -> bool:
    return any(pattern.search(span_text) for pattern in _BOILERPLATE_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=Path("data/salience_labels/qwen_pass1.jsonl"))
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be pruned without writing anything.",
    )
    args = parser.parse_args()

    total_records = 0
    total_spans_before = 0
    total_spans_after = 0
    records_emptied = 0
    pruned_examples: list[str] = []

    out_lines: list[str] = []
    with args.labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_records += 1
            spans = record.get("spans") or []
            total_spans_before += len(spans)
            kept_spans = []
            for span in spans:
                text = span.get("span_text", "")
                if _is_boilerplate(text):
                    if len(pruned_examples) < 25:
                        pruned_examples.append(text.strip()[:150])
                else:
                    kept_spans.append(span)
            total_spans_after += len(kept_spans)
            if spans and not kept_spans:
                records_emptied += 1
            record["spans"] = kept_spans
            out_lines.append(json.dumps(record))

    pruned_count = total_spans_before - total_spans_after
    print(
        f"{total_records} records, {total_spans_before} spans before, "
        f"{total_spans_after} spans after ({pruned_count} pruned, "
        f"{records_emptied} records now empty)",
        flush=True,
    )
    print("\nsample pruned spans:", flush=True)
    for example in pruned_examples:
        print(f"  - {example}", flush=True)

    if args.dry_run:
        print("\n--dry-run: nothing written", flush=True)
        return

    if not args.no_backup:
        backup_path = args.labels_path.with_suffix(args.labels_path.suffix + ".pre_prune.bak")
        shutil.copy2(args.labels_path, backup_path)
        print(f"\nbackup written -> {backup_path}", flush=True)

    with args.labels_path.open("w", encoding="utf-8") as handle:
        for line in out_lines:
            handle.write(line)
            handle.write("\n")
    print(f"rewrote {args.labels_path}", flush=True)


if __name__ == "__main__":
    main()
