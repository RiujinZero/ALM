"""Co-develop existing cortical event dendrites under recalled apical context."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_apical_context_return as apical,
)
from manifoldflow.experiments import (
    dendritic_basin_phase145_residual_route_codevelopment as phase145,
)
from manifoldflow.experiments import (
    dendritic_basin_phase150_online_residual_permission as phase150,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _development_stream,
)
from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
)


def _is_word_byte(byte_id: int) -> bool:
    return 65 <= byte_id <= 90 or 97 <= byte_id <= 122


def _recalled_history(
    learner: object, support: dict[tuple[int, int], float]
) -> list[int] | None:
    history = learner.event_history.copy()
    if not history:
        return None
    winners: dict[int, tuple[float, int]] = {}
    for (assembly, receptor), amount in support.items():
        offset = apical._history_offset(learner, receptor)
        if offset is None or offset >= len(history) or amount <= 0.0:
            continue
        current = winners.get(offset)
        if current is None or (amount, -assembly) > (current[0], -current[1]):
            winners[offset] = amount, assembly
    if not winners:
        return None
    for offset, (_amount, assembly) in winners.items():
        history[-1 - offset] = assembly
    return history


def _teach_apical_context(
    learner: object, support: dict[tuple[int, int], float]
) -> bool:
    """Use real current word as target while recall supplies only context."""
    recalled_history = _recalled_history(learner, support)
    if recalled_history is None or not learner.current_word:
        return False
    original_history = learner.event_history
    original_word = learner.current_word
    try:
        learner.event_history = recalled_history
        learner.current_word = bytearray(original_word)
        learner._complete_word(learn=True)
    finally:
        learner.event_history = original_history
        learner.current_word = original_word
    return True


def _develop(
    payload: dict[str, object], stream: bytes
) -> tuple[object, dict[str, int]]:
    arm = phase150._configure(payload)
    arm.relational_residual_codevelopment = False
    pending_support: dict[tuple[int, int], float] = {}
    taught = 0
    proposed = 0
    for byte_id in stream:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        support = apical._population_support(arm)
        if support:
            proposed += 1
            for item, amount in support.items():
                pending_support[item] = max(pending_support.get(item, 0.0), amount)
        if not _is_word_byte(byte_id):
            taught += int(_teach_apical_context(arm.learner, pending_support))
            pending_support.clear()
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)
    return arm, {"proposed": proposed, "taught_words": taught}


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    development_epochs: int,
    scan_bytes: int,
    replay_scan: bool,
) -> dict[str, object]:
    if development_epochs < 1:
        raise ValueError("development epochs must be positive")
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    tiny = tiny_path.read_bytes()
    secondary = None if secondary_path is None else secondary_path.read_bytes()
    required = development_bytes + scan_bytes
    stream = (
        tiny[:required]
        if secondary is None
        else _development_stream(
            tiny, secondary, mode="mixed", byte_count=required, chunk_bytes=256
        )
    )
    arm, development = _develop(
        payload, stream[:development_bytes] * development_epochs
    )
    if replay_scan:
        phase145._reset_dynamic_state_for_replay(arm)
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    rows = []
    proposed = 0
    scan_stream = (
        stream[:scan_bytes]
        if replay_scan
        else stream[development_bytes:required]
    )
    for offset, byte_id in enumerate(scan_stream):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        support = apical._population_support(arm)
        proposed += int(bool(support))
        items = tuple(sorted(support))
        scrambled_items = apical._scramble_items(arm.learner, items) if items else None
        if scrambled_items is not None:
            natural = apical._context_overlay_distribution(
                arm.learner, items, support=support
            )
            scrambled_support = {
                replacement: support[item]
                for item, replacement in zip(items, scrambled_items, strict=True)
            }
            scrambled = apical._context_overlay_distribution(
                arm.learner, scrambled_items, support=scrambled_support
            )
            variation = 0.5 * float(np.sum(np.abs(natural - baseline)))
            if variation > 0.0 or not np.array_equal(scrambled, baseline):
                rows.append(
                    {
                        "position": development_bytes + offset,
                        "utility": float(
                            np.log2(max(natural[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300))
                        ),
                        "scrambled_utility": float(
                            np.log2(max(scrambled[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300))
                        ),
                        "variation": variation,
                    }
                )
        _advance_generated_state(arm, byte_id, float(baseline[byte_id]))
    utilities = np.asarray([row["utility"] for row in rows], dtype=np.float64)
    scrambled = np.asarray(
        [row["scrambled_utility"] for row in rows], dtype=np.float64
    )
    return {
        "task": "apical_context_codevelopment",
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "development_epochs": development_epochs,
        "replay_scan": replay_scan,
        "scan_bytes": scan_bytes,
        "development": development,
        "proposed_returns": proposed,
        "scored_returns": len(rows),
        "utility_sum": float(np.sum(utilities)) if len(utilities) else 0.0,
        "utility_positive_fraction": (
            float(np.mean(utilities > 0.0)) if len(utilities) else 0.0
        ),
        "scrambled_utility_sum": (
            float(np.sum(scrambled)) if len(scrambled) else 0.0
        ),
        "identity_advantage_mean": (
            float(np.mean(utilities - scrambled)) if len(utilities) else 0.0
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=1024)
    parser.add_argument("--development-epochs", type=int, default=1)
    parser.add_argument("--scan-bytes", type=int, default=1024)
    parser.add_argument("--replay-scan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        development_epochs=args.development_epochs,
        scan_bytes=args.scan_bytes,
        replay_scan=args.replay_scan,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
