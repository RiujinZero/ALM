#!/usr/bin/env python
"""Generates contrastive training pairs for ManifoldFlow's retrieval
mechanism (`SonicKeyProjection`), using the same local Qwen server the
salience labeler already uses.

Why this exists: `scripts/round_trip_gate_retrieval_test.py` found that
`key_projection` - trained only on the main training loop's own
`info_nce_retrieval_loss`, itself only ever exercised on templated
distractor statements ("X has N Y") against their own oracle queries -
does not generalize to natural language. Retrieval accuracy on ten real
paraphrase-style (write, query) pairs came out at 50%, and no
`--gate-novelty-threshold` in a 0.05-1.2 sweep gave a clean separation
between true merges (a paraphrase of the same fact) and false merges (a
differently-worded but genuinely different fact) - see
docs/Salient_Gate_Dataset_Runbook.md's "Gate-driven writes for chat/filler
content, and why it isn't ready" section.

The main training loop's retrieval loss only ever sees one template
family and its own oracle query - it has no reason to learn paraphrase-
invariance (same meaning, different wording -> close) or hard-negative
discrimination (similar wording, different meaning -> far), because
nothing in its training data varies wording independently of meaning.
This script generates exactly that missing signal: for each real anchor
statement (drawn from the existing Qwen-labeled positive spans - real
salient content, not synthetic templates), asks Qwen for one paraphrase
(same fact, reworded) and one hard negative (similar wording/topic,
different fact) - real supervision for a contrastive/triplet loss, not
mechanistic extraction of "how Qwen does it" (this is the same
data-distillation pattern that already worked for salience labeling,
pointed at the retrieval problem instead of the classification one).

Requires an ik_llama.cpp server already running (same one
label_salience_qwen.py uses) - see that script's module docstring for the
launch command; this script does not start it.

Usage:
    python scripts/generate_contrastive_pairs.py ^
        --labels-path data/salience_labels/qwen_pass1.jsonl ^
        --max-anchors 2000 ^
        --output data/salience_labels/contrastive_pairs.jsonl

Safe to stop and re-run: anchors already present in --output are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests

_SYSTEM_PROMPT = """You are generating training data for a semantic \
retrieval system. You will be given one short ANCHOR statement. Produce \
exactly two rewrites of it:

1. "paraphrase": restate the SAME underlying fact using noticeably \
different wording, sentence structure, or phrasing - as if someone were \
asking about or restating this same fact later, in their own words. Do \
not just swap one or two words; genuinely reword it. The fact itself \
(who/what/how much/which one) must stay identical.

2. "hard_negative": write a DIFFERENT statement that is superficially \
similar to the anchor in topic, structure, or wording, but states a \
different fact - a different name, quantity, choice, date, or value. It \
should be the kind of statement someone could easily confuse with the \
anchor at a glance, but which is actually about something else or has a \
different value.

Respond with exactly one JSON object and nothing else: \
{"paraphrase": "...", "hard_negative": "..."}"""

_THINK_CLOSE_BUDGET_CHARS = 200


def _call_llama_server(
    base_url: str,
    anchor_text: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> tuple[str, str | None]:
    """Same streaming/think-abort convention as label_salience_qwen.py's
    _call_llama_server (see that function's docstring for why) - kept as
    a separate copy here rather than a shared import since the two
    scripts' system prompts differ and this one doesn't otherwise share
    state with the salience labeler."""
    payload = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"ANCHOR: {anchor_text}"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response = None
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions", json=payload, timeout=timeout, stream=True
            )
            response.raise_for_status()
            accumulated = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content_piece = delta.get("content")
                if not content_piece:
                    continue
                accumulated += content_piece
                think_start = accumulated.find("<think>")
                if think_start != -1 and "</think>" not in accumulated:
                    chars_since_think = len(accumulated) - (think_start + len("<think>"))
                    if chars_since_think > _THINK_CLOSE_BUDGET_CHARS:
                        response.close()
                        return "skipped_thinking", None
            return "ok", accumulated
        except Exception as error:  # noqa: BLE001 - a batch job must survive one bad call
            last_error = error
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
        finally:
            if response is not None:
                response.close()
    print(f"  request failed after {retries + 1} attempts: {last_error}", file=sys.stderr)
    return "request_error", None


def _parse_pair(raw_text: str) -> dict[str, str] | None:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return None
    paraphrase = parsed.get("paraphrase")
    hard_negative = parsed.get("hard_negative")
    if not isinstance(paraphrase, str) or not isinstance(hard_negative, str):
        return None
    if not paraphrase.strip() or not hard_negative.strip():
        return None
    return {"paraphrase": paraphrase.strip(), "hard_negative": hard_negative.strip()}


def _load_anchors(labels_path: Path, *, max_anchors: int, seed: int) -> list[str]:
    """Distinct span_texts from the existing Qwen-labeled positive spans -
    real salient content already validated as write-worthy, not a new
    sampling policy invented for this script. Deduplicated (many
    distractor statements repeat the same template with different
    slot-fills, e.g. "X owns N Y" - keeping every instance would just
    burn budget on near-identical anchors) and shuffled once for a
    representative, order-independent sample."""
    seen: set[str] = set()
    anchors: list[str] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for span in record.get("spans") or []:
                text = (span.get("span_text") or "").strip()
                if not text or len(text) < 8 or text in seen:
                    continue
                seen.add(text)
                anchors.append(text)
    random.Random(seed).shuffle(anchors)
    return anchors[:max_anchors]


def _load_anchors_file(anchors_path: Path, *, max_anchors: int, seed: int) -> list[str]:
    """scripts/sample_diverse_anchors.py's own output - already deduped
    and shuffled once by that script, but reshuffled here too with this
    script's own seed so --max-anchors takes a fresh, independent slice
    rather than always the same prefix of that file."""
    anchors: list[str] = []
    with anchors_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = (record.get("anchor") or "").strip()
            if text:
                anchors.append(text)
    random.Random(seed).shuffle(anchors)
    return anchors[:max_anchors]


def _load_done_anchors(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done.add(record["anchor"])
    return done


def _generate_pair(anchor: str, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    status, raw_response = _call_llama_server(
        args.llama_server_url,
        anchor,
        temperature=args.temperature,
        max_tokens=args.max_response_tokens,
        timeout=args.request_timeout,
        retries=args.retries,
    )
    if status == "skipped_thinking":
        return None, "skipped_thinking"
    if status == "request_error":
        return {"anchor": anchor, "parse_error": True, "request_error": True}, "request_error"

    pair = _parse_pair(raw_response)
    if pair is None:
        return (
            {"anchor": anchor, "parse_error": True, "raw_response": raw_response},
            "parse_error",
        )
    return (
        {
            "anchor": anchor,
            "paraphrase": pair["paraphrase"],
            "hard_negative": pair["hard_negative"],
            "parse_error": False,
        },
        "ok",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-path", type=Path, default=Path("data/salience_labels/qwen_pass1.jsonl"))
    parser.add_argument(
        "--anchors-file",
        type=Path,
        default=None,
        help="scripts/sample_diverse_anchors.py's own output ({\"source\": "
        "..., \"anchor\": ...} JSONL) - takes priority over --labels-path "
        "when given, since it already draws from many real sources "
        "(including --labels-path's own corpus as one of them), not just "
        "the existing labeled corpus alone.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-anchors", type=int, default=2000)
    parser.add_argument("--anchor-seed", type=int, default=13)
    parser.add_argument("--llama-server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=512,
        help="Much smaller than the salience labeler's 4096 - this is one "
        "short JSON object with two short sentences, not a list of spans.",
    )
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="See Salient_Gate_Dataset_Runbook.md's validated-configuration "
        "table for how this trades off against the server's own -ncmoe/-np "
        "settings - the same server, same tradeoffs, as the salience "
        "labeler.",
    )
    args = parser.parse_args()

    if args.anchors_file is not None:
        anchors = _load_anchors_file(args.anchors_file, max_anchors=args.max_anchors, seed=args.anchor_seed)
        source_note = str(args.anchors_file)
    else:
        anchors = _load_anchors(args.labels_path, max_anchors=args.max_anchors, seed=args.anchor_seed)
        source_note = str(args.labels_path)
    if not anchors:
        raise SystemExit(f"no usable anchors found in {source_note}")
    print(f"{len(anchors)} distinct anchors sampled from {source_note}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_anchors = _load_done_anchors(args.output)
    if done_anchors:
        print(f"resuming: {len(done_anchors)} anchors already done in {args.output}", flush=True)

    work_items = [a for a in anchors if a not in done_anchors]
    print(f"{len(work_items)} anchors queued (concurrency={args.concurrency})", flush=True)

    generated_this_run = 0
    parse_failures = 0
    request_failures = 0
    skipped_thinking = 0
    start_time = time.monotonic()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = [pool.submit(_generate_pair, anchor, args) for anchor in work_items]
        completed = 0
        for future in as_completed(futures):
            record, status = future.result()
            completed += 1
            if status == "skipped_thinking":
                skipped_thinking += 1
            elif status == "request_error":
                request_failures += 1
            else:
                if status == "parse_error":
                    parse_failures += 1
                    print(
                        f"  parse failure on anchor={record['anchor'][:60]!r} - "
                        f"raw response saved for review",
                        file=sys.stderr,
                    )
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                generated_this_run += 1

            if completed % args.progress_every == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"[{completed}/{len(work_items)} attempted, {generated_this_run} generated] "
                    f"{elapsed:.1f}s elapsed, {parse_failures} parse failures, "
                    f"{request_failures} request failures, {skipped_thinking} skipped (thinking)",
                    flush=True,
                )

    print(
        f"done: generated {generated_this_run} pairs this run ({parse_failures} parse "
        f"failures, {request_failures} request failures, {skipped_thinking} skipped for "
        f"thinking) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
