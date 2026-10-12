#!/usr/bin/env python
"""Converts a sample of smoltalk's reasoning/COT subset (a plain JSONL of
{"text": "USER: ...\\nASSISTANT: ..."} records, not pregenerated session
data) into the same shard format session_builder.write_shard/read_shard
already produce for the real production corpus - so the *existing*,
unmodified label_salience_qwen.py and build_salience_batch can consume it
with zero changes, rather than building a parallel loading path.

Each smoltalk record becomes one minimal "session": a single "chat"
episode spanning the whole (truncated-to-window_tokens) record. This is
deliberately not a full session_builder.build_session replay (no
interleaved distractor/filler mix) - the only thing this needs to support
is Qwen labeling + the salience gate's own build_salience_batch, both of
which only care about a session's token_ids and each episode's
start/end/episode_type, not how the session was assembled.

Streams the source file line by line - it is not safe to load into
memory (tens of GB) - and stops after --max-records, not at end of file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/path/to/gdrive/training_data/chat/smoltalk_reasoning.jsonl"),
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k")
    parser.add_argument("--output-dir", type=Path, default=Path("/path/to/manifoldflow_sessions_smoltalk_cot"))
    parser.add_argument("--max-records", type=int, default=4000)
    parser.add_argument("--sessions-per-shard", type=int, default=200)
    parser.add_argument("--window-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=64, help="Skip records shorter than this - too short to be a meaningful candidate span.")
    args = parser.parse_args()

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[dict] = []
    shard_index = 0
    session_id = 0
    kept = 0
    skipped_short = 0
    skipped_bad = 0

    def flush_shard() -> None:
        nonlocal shard_index, sessions
        if not sessions:
            return
        path = args.output_dir / f"shard_smoltalk_cot_{shard_index:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for session in sessions:
                handle.write(json.dumps(session))
                handle.write("\n")
        print(f"wrote {len(sessions)} sessions -> {path}", flush=True)
        shard_index += 1
        sessions = []

    with args.source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if kept >= args.max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad += 1
                continue
            text = record.get("text")
            if not text:
                skipped_bad += 1
                continue
            token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
            if len(token_ids) < args.min_tokens:
                skipped_short += 1
                continue
            token_ids = token_ids[: args.window_tokens]
            sessions.append(
                {
                    "session_id": session_id,
                    "token_ids": token_ids,
                    "episodes": [
                        {"episode_type": "chat", "start": 0, "end": len(token_ids)}
                    ],
                    "total_tokens": len(token_ids),
                }
            )
            session_id += 1
            kept += 1
            if len(sessions) >= args.sessions_per_shard:
                flush_shard()
            if kept % 500 == 0:
                print(f"...{kept} sessions converted so far", flush=True)

    flush_shard()
    print(
        f"done: {kept} sessions written across {shard_index} shards "
        f"({skipped_short} too-short, {skipped_bad} unparseable) -> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
