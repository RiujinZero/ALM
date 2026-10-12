"""Build a matched TinyStories/SmolTalk-everyday raw-byte diagnostic stream."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from manifoldflow.dendritic_basin.tasks.interleaved_language_bytes import (
    build_interleaved_stream,
)

DEFAULT_TINY = Path(r"C:\path\to\gdrive\training_data\TinyStories-train.txt")
DEFAULT_SMOL = Path(r"C:\path\to\manifoldflow_chat_sources\smoltalk_everyday.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinystories", type=Path, default=DEFAULT_TINY)
    parser.add_argument("--smoltalk", type=Path, default=DEFAULT_SMOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--smoltalk-valid-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = build_interleaved_stream(
        args.tinystories,
        args.smoltalk,
        args.output,
        total_bytes=args.total_bytes,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        smoltalk_valid_output=args.smoltalk_valid_output,
    )
    encoded = json.dumps(asdict(report), indent=2)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
