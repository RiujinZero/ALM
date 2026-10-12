"""Recruit learned context, then let its existing outgoing branch speak."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifoldflow.experiments import (
    dendritic_basin_phase86_context_recruitment as phase86,
)

CONDITIONS = (
    "recall_disabled",
    "adapted_branch_recruitment",
    "adapted_branch_content_successor",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=300)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = phase86.run_ablation(
        args.state,
        samples_per_prompt=args.samples_per_prompt,
        max_new_bytes=args.max_new_bytes,
        prompt_limit=args.prompt_limit,
        conditions=CONDITIONS,
        seed=8900,
    )
    report["task"] = "phase89_context_item_successor"
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
