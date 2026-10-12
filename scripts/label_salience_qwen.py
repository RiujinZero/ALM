#!/usr/bin/env python
"""Labels ManifoldFlow write-eligibility ("salience") using a local
Qwen3.6-35B (REAP-pruned) model served via ik_llama.cpp.

Today's write trigger is entirely scripted: `session_builder.py` only
attaches `statement_spans` metadata to "distractor"-type episodes, and
`run_step`'s write phase only ever looks at that metadata - a chat episode
has structurally zero chance of being written, no matter what it says,
because nothing scores it. This script generates the graded-salience
training signal a future learned write-gate would need: it feeds the real
packed 8192-token windows Sonic already trains on (the natural mix of
chat/distractor/filler content, not a filtered subset) through a much
larger model and asks it to identify which spans are worth writing, with a
graded score - including distractor statements, which double as a free
calibration check (a labeler that doesn't reliably call known-salient
distractor statements salient needs its rubric or model fixed before the
run is trusted).

This does NOT merge results back into the training shards - that happens
later, once labels exist, so this only needs to preserve enough provenance
(shard path, session id, window index, window token range) for that future
step to find its way back to the right window.

Requires an ik_llama.cpp server already running with the model loaded -
this script deliberately does not launch it; starting a 10GB local model
process should be a decision the operator makes directly, not something
silently spawned by a labeling script. Example launch (adjust
--n-gpu-layers to how much VRAM is actually free - 0 for fully CPU,
i.e. "offload the experts to CPU"):

    C:\\ik_llama.cpp\\build\\bin\\Release\\llama-server.exe ^
        -m "C:\\Users\\user\\.cache\\huggingface\\hub\\models--lennyhans--Qwen3.6-35B-REAP-Pruned-ratio-0.5-Q4_K_M-GGUF\\snapshots\\<hash>\\qwen3.6-35b-reap-pruned-ratio-0.5-q4_k_m.gguf" ^
        --ctx-size 16384 -ngl 999 -ncmoe 20 --reasoning off --port 8080

--reasoning off matters beyond speed: this is a "thinking" model, and with
reasoning left on its default ("auto"), long <think> blocks on
statement-dense windows can consume the entire response budget before the
model ever reaches the actual JSON answer - a real, confirmed failure mode
(see _parse_spans' resulting parse_error records), not a hypothetical one.
-ncmoe N keeps the first N of the model's 40 layers' MoE expert weights
in CPU memory (ik_llama.cpp's own MoE-aware offload control) while -ngl
999 puts everything else (attention, norms, embeddings) on GPU - N=20
measured at ~7.7GB VRAM on this machine's RTX 5080, comfortably under the
production training run's own ~15.7GB footprint so the two need never
overlap in practice.

Usage:
    python scripts/label_salience_qwen.py ^
        --shard-dir C:/path/to/manifoldflow_sessions --max-windows 500 ^
        --output data/salience_labels/qwen_pass1.jsonl

Safe to stop and re-run: already-labeled (shard, session, window) keys in
--output are skipped, so a labeling pass can proceed across several
sittings (e.g. several training pauses) without redoing work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))

_DEFAULT_TOKENIZER_DIR = _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"

# Anchored against known-salient distractor statements (see module
# docstring) rather than invented from scratch, so the rubric's low/high
# ends have a concrete, verifiable referent instead of relying on the
# model's own untested notion of "important."
_SYSTEM_PROMPT = """You are labeling training text for a persistent-memory \
system. The system can only remember a small number of things it \
explicitly writes down; everything else is forgotten once it scrolls out \
of view. Your job is to find spans of text that state something worth \
writing down, and grade how worth-writing each one is.

Write-worthy spans are durable facts, stated states, preferences, \
identities, or corrections that someone might need to recall much later - \
things like a name, a stated preference, a quantity or status tied to a \
specific person or object, a decision, or a correction to something said \
earlier. NOT write-worthy: greetings, acknowledgments, filler, questions \
with no stated answer, and generic narration with no durable content.

The text you receive is packed training data and may contain several \
unrelated pieces: ordinary conversation, short templated statements about \
named people/objects (e.g. "Kevin has 4 cats." or "The vault code is \
8152."), and unrelated filler prose. Templated statements like these ARE \
write-worthy and should usually score high (0.8+) if they state a \
concrete, checkable fact - use them as your calibration anchor for what \
"high salience" means.

For every write-worthy span you find, output one entry with:
- "span_text": the exact text of the span, copied verbatim from the input
- "salience": a float from 0.0 (not worth writing) to 1.0 (clearly worth writing)
- "category": one of "fact", "preference", "identity", "correction", "decision"
- "reason": one short sentence explaining the score

Respond with ONLY a JSON object of the form {"spans": [...]}. If nothing \
in the text is write-worthy, respond with {"spans": []}. No other text, \
no markdown fences."""


def _iter_shard_paths(
    shard_dir: Path, shard_glob: str, *, order_file: Path | None = None
) -> list[Path]:
    """Default order is a plain sorted glob - fine for an initial pass
    covering the corpus broadly. `order_file` overrides that with an
    explicit path list (one per line), used to make a labeling pass
    follow a specific production training lane's own shard order (see
    docs/Salient_Gate_Dataset_Runbook.md's "Aligning labels with live
    training" section) - labeling forward from wherever that lane
    currently is, in the exact order it will actually read those shards,
    rather than an arbitrary disconnected slice of the corpus."""
    if order_file is not None:
        return [Path(line) for line in order_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(shard_dir.glob(shard_glob))


def _iter_session_windows(
    shard_paths: list[Path], tokenizer: Any, *, window_tokens: int
) -> Iterator[dict[str, Any]]:
    """Yields one dict per physical window, decoded back to text - the
    same ceil-div slicing `_precompute_windows` uses in the training
    script, but this only needs the raw token_ids slice (no write/query
    episode classification), so it's reimplemented directly rather than
    importing the training script's Lane/Window machinery."""
    from manifoldflow.dataset.session_builder import read_shard

    for shard_path in shard_paths:
        for session in read_shard(shard_path):
            token_ids = session["token_ids"]
            num_windows = max(1, -(-len(token_ids) // window_tokens))
            for window_index in range(num_windows):
                start = window_index * window_tokens
                end = min(start + window_tokens, len(token_ids))
                window_ids = token_ids[start:end]
                if not window_ids:
                    continue
                yield {
                    "shard_path": str(shard_path),
                    "session_id": session["session_id"],
                    "window_index": window_index,
                    "window_start": start,
                    "window_end": end,
                    "window_text": tokenizer.decode(window_ids, skip_special_tokens=True),
                }


def _load_done_keys(output_path: Path) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((record["shard_path"], record["session_id"], record["window_index"]))
    return done


# A harmless "<think>\n\n</think>" formality is ~15 chars between the
# tags; genuine reasoning traces observed in testing ran to thousands of
# characters. 200 gives real thinking nowhere to hide while leaving
# comfortable headroom above the empty case.
_THINK_CLOSE_BUDGET_CHARS = 200


def _call_llama_server(
    base_url: str,
    window_text: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> tuple[str, str | None]:
    """Streams the response and aborts if the model is genuinely reasoning
    at length, instead of either hoping --reasoning off holds (it doesn't
    always - confirmed: some windows reason at length regardless) or
    paying a --reasoning-budget 0 cap's cost on every request even when
    the model wasn't going to think anyway. A window that starts real
    reasoning gets skipped outright rather than forced to answer without
    it - cheap (aborted within a small char budget) given the corpus has
    far more windows than the target example count needs, so losing an
    occasional hard window costs nothing a plainer window can't replace.

    Every response - even ones that never really "think" - opens with a
    "<think>" tag; a harmless, immediately-closed "<think>\\n\\n</think>"
    is the model's formality, not real reasoning (confirmed: this is what
    every successful response in testing looked like). So the abort
    trigger isn't "<think>" appearing at all - it's "<think>" appearing
    without a "</think>" closing it within `_THINK_CLOSE_BUDGET_CHARS`
    characters, which only a genuinely long reasoning trace would exceed.

    Returns (status, content): status is "ok" (content is the full
    response text), "skipped_thinking" (content is None - move on to the
    next window), or "request_error" (content is None - a genuine
    network/HTTP failure, retried up to `retries` times first)."""
    payload = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": window_text},
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


def _parse_spans(raw_text: str) -> list[dict[str, Any]] | None:
    """Lenient JSON extraction: the model was asked for exactly one JSON
    object, but local models occasionally wrap it in prose or markdown
    fences anyway - take the first '{' through the matching last '}'
    rather than requiring the whole response to be pure JSON."""
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return None
    spans = parsed.get("spans")
    if not isinstance(spans, list):
        return None
    return spans


def _locate_span(window_text: str, span_text: str) -> tuple[int, int] | None:
    index = window_text.find(span_text)
    if index == -1:
        return None
    return index, index + len(span_text)


def _label_window(
    window: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any] | None, str]:
    """Does the actual server call + parse + offset-resolution for one
    window - pure with respect to shared state (only reads `window`/
    `args`), so it's safe to run from a thread pool worker. Returns
    (record, status): status is "ok", "parse_error", "request_error", or
    "skipped_thinking" (record is None only for that last one - see
    _call_llama_server's docstring for why a thinking window is skipped
    outright rather than logged as any kind of failure). The caller
    handles writing/logging, since only the main thread touches the
    output file."""
    status, raw_response = _call_llama_server(
        args.llama_server_url,
        window["window_text"],
        temperature=args.temperature,
        max_tokens=args.max_response_tokens,
        timeout=args.request_timeout,
        retries=args.retries,
    )
    if status == "skipped_thinking":
        return None, "skipped_thinking"
    if status == "request_error":
        return (
            {
                "shard_path": window["shard_path"],
                "session_id": window["session_id"],
                "window_index": window["window_index"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "spans": None,
                "parse_error": True,
                "raw_response": None,
                "request_error": True,
            },
            "request_error",
        )

    spans = _parse_spans(raw_response)
    if spans is None:
        return (
            {
                "shard_path": window["shard_path"],
                "session_id": window["session_id"],
                "window_index": window["window_index"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "spans": None,
                "parse_error": True,
                "raw_response": raw_response,
            },
            "parse_error",
        )

    resolved_spans = []
    for span in spans:
        span_text = span.get("span_text", "")
        offsets = _locate_span(window["window_text"], span_text)
        resolved_spans.append(
            {
                "span_text": span_text,
                "salience": span.get("salience"),
                "category": span.get("category"),
                "reason": span.get("reason"),
                "char_start": offsets[0] if offsets else None,
                "char_end": offsets[1] if offsets else None,
            }
        )

    record = {
        "shard_path": window["shard_path"],
        "session_id": window["session_id"],
        "window_index": window["window_index"],
        "window_start": window["window_start"],
        "window_end": window["window_end"],
        "spans": resolved_spans,
        "parse_error": False,
    }
    return record, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-dir", type=Path, default=Path("C:/path/to/manifoldflow_sessions"))
    parser.add_argument("--shard-glob", default="batch*/shard_*.jsonl")
    parser.add_argument(
        "--shard-order-file",
        type=Path,
        default=None,
        help="One shard path per line - overrides --shard-glob's sorted "
        "order with this explicit sequence. Used to align a labeling pass "
        "with a live production training lane's own shard order (see "
        "docs/Salient_Gate_Dataset_Runbook.md).",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--window-tokens", type=int, default=8192)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent in-flight requests to the llama-server. Default "
        "1 deliberately: measured slower (not faster) than sequential at "
        "concurrency=4 in testing, since the bottleneck is CPU-side MoE "
        "expert compute (a shared, already-saturated resource), not "
        "network wait - concurrency just divides the same fixed CPU pool "
        "and adds scheduling overhead. It also introduced generation "
        "failures never seen sequentially. Raise only if a different "
        "hardware profile (e.g. -ncmoe 0, fully GPU-resident) changes "
        "that tradeoff - verify with a real timed test first, as done "
        "here (see Salient_Gate_Dataset_Runbook.md).",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))

    shard_paths = _iter_shard_paths(args.shard_dir, args.shard_glob, order_file=args.shard_order_file)
    if not shard_paths:
        raise SystemExit(f"no shards matched {args.shard_dir}/{args.shard_glob}")
    print(f"{len(shard_paths)} shard files found under {args.shard_dir}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_keys = _load_done_keys(args.output)
    if done_keys:
        print(f"resuming: {len(done_keys)} windows already labeled in {args.output}", flush=True)

    # Build the work list single-threaded first (cheap - just iterating and
    # decoding token ids, no network calls yet), instead of interleaving
    # iteration with concurrent dispatch - keeps the resume-skip logic
    # simple and gives an exact, known-size batch to hand to the thread
    # pool.
    work_items: list[dict[str, Any]] = []
    for window in _iter_session_windows(shard_paths, tokenizer, window_tokens=args.window_tokens):
        if len(work_items) >= args.max_windows:
            break
        key = (window["shard_path"], window["session_id"], window["window_index"])
        if key in done_keys:
            continue
        work_items.append(window)
    print(f"{len(work_items)} windows queued for labeling (concurrency={args.concurrency})", flush=True)

    labeled_this_run = 0
    parse_failures = 0
    request_failures = 0
    skipped_thinking = 0
    start_time = time.monotonic()

    # Concurrency happens at the HTTP-request level (each _label_window
    # call blocks on a network round-trip, so a thread pool - not
    # multiprocessing - is the right tool: threads release the GIL while
    # waiting on requests.post). Only the main thread ever touches
    # `handle`/the output file, so no lock is needed around the writes
    # below - each worker just returns its record and the main thread
    # writes it as futures complete.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = [pool.submit(_label_window, window, args) for window in work_items]
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
                        f"  parse failure on session={record['session_id']} "
                        f"window={record['window_index']} - raw response saved for review",
                        file=sys.stderr,
                    )
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                labeled_this_run += 1

            if completed % args.progress_every == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"[{completed}/{len(work_items)} attempted, {labeled_this_run} labeled] "
                    f"{elapsed:.1f}s elapsed, {parse_failures} parse failures, "
                    f"{request_failures} request failures, {skipped_thinking} skipped (thinking)",
                    flush=True,
                )

    print(
        f"done: labeled {labeled_this_run} windows this run "
        f"({parse_failures} parse failures, {request_failures} request failures, "
        f"{skipped_thinking} skipped for thinking) -> {args.output}"
    )


if __name__ == "__main__":
    main()
