"""Trace exact contextual-successor returns along controlled suffixes."""

from __future__ import annotations

import argparse
import json
import pickle
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import CASES
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _prime,
)


def _form(learner: object, item: tuple[int, int]) -> str:
    assembly, receptor = item
    if not 0 <= assembly < len(learner.word_forms):
        return f"?@{receptor}"
    return (
        learner.word_forms[assembly].decode("utf-8", errors="backslashreplace")
        + f"@{receptor}"
    )


def _winner(distribution: np.ndarray) -> str:
    return bytes((int(np.argmax(distribution)),)).decode(
        "utf-8", errors="backslashreplace"
    )


def _margin(distribution: np.ndarray, target: int) -> float:
    competitors = distribution.copy()
    competitors[target] = -np.inf
    return log2(max(float(distribution[target]), 1e-300)) - log2(
        max(float(np.max(competitors)), 1e-300)
    )


def _trace_case(
    developed: object,
    prompt: bytes,
    suffix: bytes,
    *,
    return_mode: str,
) -> dict[str, object]:
    prepared = _prime(developed, prompt, return_mode=return_mode)
    off = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    on = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    off.recall_enabled = False
    on.recall_enabled = True
    for arm in (off, on):
        arm.access_learning_enabled = False
        arm.expression_learning_enabled = False
    positions = []
    total_saved = 0.0
    returned_forms: set[str] = set()
    for index, byte_id in enumerate(suffix):
        off_distribution = _predict(off, live=False)
        on_distribution = _predict(on, live=True)
        off_probability = max(float(off_distribution[byte_id]), 1e-300)
        on_probability = max(float(on_distribution[byte_id]), 1e-300)
        saved = log2(on_probability / off_probability)
        total_saved += saved
        active = tuple(on.recalled_expression.active)
        forms = tuple(_form(on.learner, item) for item in active)
        returned_forms.update(forms)
        positions.append(
            {
                "index": index,
                "byte": bytes((byte_id,)).decode(
                    "utf-8", errors="backslashreplace"
                ),
                "saved_bits": saved,
                "off_probability": off_probability,
                "on_probability": on_probability,
                "off_margin_bits": _margin(off_distribution, byte_id),
                "on_margin_bits": _margin(on_distribution, byte_id),
                "off_winner": _winner(off_distribution),
                "on_winner": _winner(on_distribution),
                "returned_forms": forms,
                "return_current_norm": (
                    on.recalled_expression.last_return_current_norm
                ),
                "field": dict(on.field.last_diagnostics),
            }
        )
        _observe(off, byte_id, off_probability, external=False)
        _observe(on, byte_id, on_probability, external=False)
    return {
        "saved_bits": total_saved,
        "returned_forms": sorted(returned_forms),
        "positions": positions,
    }


def run(
    developed_state: Path,
    *,
    competition_gain: float = 0.0,
    return_cancellation: str = "none",
    return_current_limit: float = 0.0,
    case_limit: int | None = None,
    return_mode: str = "contextual_successor",
) -> dict[str, object]:
    with developed_state.open("rb") as handle:
        developed = pickle.load(handle)
    developed.field.contextual_successor_competition_gain = competition_gain
    developed.expression_cancellation_mode = return_cancellation
    developed.expression_current_limit = return_current_limit
    rows = []
    cases = CASES if case_limit is None else CASES[:case_limit]
    for case in cases:
        rows.append(
            {
                "case": case.name,
                "expected": case.expected,
                "trace": _trace_case(
                    developed,
                    case.prompt.encode(),
                    case.expected.encode(),
                    return_mode=return_mode,
                ),
            }
        )
    return {
        "task": "phase92_contextual_successor_return_trace",
        "developed_state": str(developed_state),
        "competition_gain": competition_gain,
        "return_cancellation": return_cancellation,
        "return_current_limit": return_current_limit,
        "return_mode": return_mode,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("developed_state", type=Path)
    parser.add_argument("--competition-gain", type=float, default=0.0)
    parser.add_argument(
        "--return-cancellation", choices=("none", "pre", "post"), default="none"
    )
    parser.add_argument("--return-current-limit", type=float, default=0.0)
    parser.add_argument(
        "--return-mode",
        choices=("direct", "contextual_successor"),
        default="contextual_successor",
    )
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.developed_state,
        competition_gain=args.competition_gain,
        return_cancellation=args.return_cancellation,
        return_current_limit=args.return_current_limit,
        case_limit=args.case_limit,
        return_mode=args.return_mode,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
