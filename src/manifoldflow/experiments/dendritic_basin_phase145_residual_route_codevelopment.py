"""Teach cross-organ cortical routes only the cortex's missed word component."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments.dendritic_basin_phase76_generation import _predict
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _development_stream,
)
from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
    _prepare_development,
    _temporal_eligibility_statistics,
)
from manifoldflow.experiments.dendritic_basin_phase143_learned_route_permission import (
    _learned_support,
)


def _reset_dynamic_state_for_replay(arm: object) -> None:
    """Restart activity while preserving every learned tissue and strength."""
    arm.learner.reset_episode()
    arm.field.reset_episode()
    arm.event_memory.reset_episode()
    arm.relational_access.reset_episode()
    arm.recalled_expression.reset_episode()
    arm.multiscale.reset_episode()
    arm.temporal_relational_eligibility.reset_episode()
    arm.route_history.clear()
    arm.raw_window.clear()
    arm.previous_routes.clear()
    arm.previous_surprise = 0.0
    arm.pending_access_context = ()
    arm.pending_access_sequence_cue = ()


def _residual_overlay_distribution(
    learner: object,
    residual_routes: Mapping[tuple[int, int], Mapping[int, float]],
    items: tuple[tuple[int, int], ...],
    *,
    support: Mapping[tuple[int, int], float],
    gain: float,
    baseline: np.ndarray | None = None,
    cross_receptor_fallback: bool = False,
) -> np.ndarray:
    """Apply signed word-assembly residuals to the current cortical readout."""
    baseline = (
        learner.predict_next_distribution().copy()
        if baseline is None
        else baseline.copy()
    )
    current = np.zeros(256, dtype=np.float64)
    prefix = bytes(learner.current_word)
    for item in items:
        assembly, receptor = item
        row = residual_routes.get((receptor, assembly))
        if row is None and cross_receptor_fallback:
            candidates = tuple(
                (candidate_receptor, candidate_row)
                for (candidate_receptor, source), candidate_row in (
                    residual_routes.items()
                )
                if source == assembly
            )
            if candidates:
                _selected_receptor, row = min(
                    candidates,
                    key=lambda candidate: (
                        abs(candidate[0] - receptor),
                        candidate[0],
                    ),
                )
        if not row:
            continue
        amount = float(support.get(item, 0.0))
        if amount <= 0.0:
            continue
        for target, weight in row.items():
            if not 0 <= target < len(learner.word_forms):
                continue
            form = learner.word_forms[target]
            if len(form) <= len(prefix) or not form.startswith(prefix):
                continue
            current[form[len(prefix)]] += amount * float(weight)
    if not np.any(current):
        return baseline
    logits = np.log(np.maximum(baseline, 1e-300)) + gain * current
    logits -= float(np.max(logits))
    distribution = np.exp(logits)
    distribution /= float(np.sum(distribution))
    return distribution


def _scramble_residual_items(
    residual_routes: Mapping[tuple[int, int], Mapping[int, float]],
    items: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    excluded = {assembly for assembly, _receptor in items}
    replacements = []
    for assembly, receptor in items:
        candidates = sorted(
            source
            for candidate_receptor, source in residual_routes
            if candidate_receptor == receptor
            and source != assembly
            and source not in excluded
        )
        if not candidates:
            return None
        replacements.append((candidates[assembly % len(candidates)], receptor))
    return tuple(replacements)


def _scramble_cross_receptor_items(
    residual_routes: Mapping[tuple[int, int], Mapping[int, float]],
    items: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    excluded = {assembly for assembly, _receptor in items}
    identities = tuple(sorted({assembly for _receptor, assembly in residual_routes}))
    replacements = []
    for assembly, receptor in items:
        candidates = tuple(
            source
            for source in identities
            if source != assembly and source not in excluded
        )
        if not candidates:
            return None
        replacements.append((candidates[assembly % len(candidates)], receptor))
    return tuple(replacements)


def _population_support(arm: object) -> dict[tuple[int, int], float]:
    support: dict[tuple[int, int], float] = {}
    for (_engram, assembly, receptor), activity in (
        arm.pending_receptor_relational_population or {}
    ).items():
        item = assembly, receptor
        support[item] = support.get(item, 0.0) + float(activity)
    return support


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    replay_scan: bool,
    gain: float,
    permission_mode: str = "learned",
    cross_receptor_fallback: bool = False,
) -> dict[str, object]:
    if permission_mode not in {"learned", "activity"}:
        raise ValueError("unknown permission mode")
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    tiny = tiny_path.read_bytes()
    secondary = None if secondary_path is None else secondary_path.read_bytes()
    required = max(development_bytes, scan_bytes) if replay_scan else (
        development_bytes + scan_bytes
    )
    stream = (
        tiny[:required]
        if secondary is None
        else _development_stream(
            tiny,
            secondary,
            mode="mixed",
            byte_count=required,
            chunk_bytes=256,
        )
    )
    started = perf_counter()
    arm = _prepare_development(
        payload,
        stream[:development_bytes],
        mode="coupled",
        access_learning_mode="branches",
        frames_per_capsule=12,
        access_proposal_threshold=0.0,
        temporal_credit_mode="receptor_relation",
        relational_residual_codevelopment=True,
    )
    learned_before_scan = _temporal_eligibility_statistics(arm)
    residual_route_count = len(arm.relational_residual_routes)
    residual_synapses = sum(
        len(row) for row in arm.relational_residual_routes.values()
    )
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.relational_residual_codevelopment = False
    if replay_scan:
        _reset_dynamic_state_for_replay(arm)
    rows = []
    proposed = 0
    supported = 0
    paired_controls = 0
    scan_stream = (
        stream[:scan_bytes] if replay_scan else stream[development_bytes:required]
    )
    for offset, byte_id in enumerate(scan_stream):
        position = offset if replay_scan else development_bytes + offset
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        population = arm.pending_receptor_relational_population or {}
        proposed += int(bool(population))
        learned_support = (
            _learned_support(arm, 1.0)
            if permission_mode == "learned"
            else _population_support(arm)
        )
        support = {
            item: amount
            for item, amount in learned_support.items()
            if amount > 0.0
            and (
                (item[1], item[0]) in arm.relational_residual_routes
                or (
                    cross_receptor_fallback
                    and any(
                        source == item[0]
                        for _receptor, source in arm.relational_residual_routes
                    )
                )
            )
        }
        items = tuple(sorted(support))
        supported += int(bool(items))
        scrambled_items = (
            (
                _scramble_cross_receptor_items(
                    arm.relational_residual_routes, items
                )
                if cross_receptor_fallback
                else _scramble_residual_items(arm.relational_residual_routes, items)
            )
            if items
            else None
        )
        if items:
            learned = _residual_overlay_distribution(
                arm.learner,
                arm.relational_residual_routes,
                items,
                support=support,
                gain=gain,
                cross_receptor_fallback=cross_receptor_fallback,
            )
            scrambled = None
            if scrambled_items is not None:
                paired_controls += 1
                scrambled_support = {
                    replacement: support[item]
                    for item, replacement in zip(items, scrambled_items, strict=True)
                }
                scrambled = _residual_overlay_distribution(
                    arm.learner,
                    arm.relational_residual_routes,
                    scrambled_items,
                    support=scrambled_support,
                    gain=gain,
                    cross_receptor_fallback=cross_receptor_fallback,
                )
            learned_utility = float(
                np.log2(max(learned[byte_id], 1e-300))
                - np.log2(max(baseline[byte_id], 1e-300))
            )
            scrambled_utility = (
                None
                if scrambled is None
                else float(
                    np.log2(max(scrambled[byte_id], 1e-300))
                    - np.log2(max(baseline[byte_id], 1e-300))
                )
            )
            variation = 0.5 * float(np.sum(np.abs(learned - baseline)))
            if variation > 0.0 or (
                scrambled_utility is not None and scrambled_utility != 0.0
            ):
                rows.append(
                    {
                        "position": position,
                        "items": len(items),
                        "maximum_support": max(support.values()),
                        "learned_utility": learned_utility,
                        "scrambled_utility": scrambled_utility,
                        "variation": variation,
                    }
                )
        _advance_generated_state(arm, byte_id, float(baseline[byte_id]))
        arm.pending_temporal_population = {}
        arm.pending_temporal_predictions = {}
        arm.pending_temporal_global_prediction = None
        arm.pending_relational_receptors = {}
        arm.pending_receptor_relational_population = {}
    aggregate = {}
    for key in ("learned_utility", "scrambled_utility", "variation"):
        values = np.asarray(
            [row[key] for row in rows if row[key] is not None], dtype=np.float64
        )
        aggregate[key] = {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "sum": float(np.sum(values)) if len(values) else 0.0,
            "positive_fraction": float(np.mean(values > 0.0)) if len(values) else 0.0,
        }
    difference = np.asarray(
        [
            row["learned_utility"] - row["scrambled_utility"]
            for row in rows
            if row["scrambled_utility"] is not None
        ],
        dtype=np.float64,
    )
    aggregate["identity_advantage"] = {
        "mean": float(np.mean(difference)) if len(difference) else 0.0,
        "positive_fraction": (
            float(np.mean(difference > 0.0)) if len(difference) else 0.0
        ),
    }
    return {
        "task": "phase145_residual_route_codevelopment",
        "state": str(state),
        "tiny_path": str(tiny_path),
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "replay_scan": replay_scan,
        "gain": gain,
        "permission_mode": permission_mode,
        "cross_receptor_fallback": cross_receptor_fallback,
        "relational_residual_learning_events": (
            arm.relational_residual_learning_events
        ),
        "residual_routes": residual_route_count,
        "residual_synapses": residual_synapses,
        "proposed_returns": proposed,
        "supported_returns": supported,
        "paired_identity_controls": paired_controls,
        "scored_returns": len(rows),
        "learned_before_scan": learned_before_scan,
        "aggregate": aggregate,
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=2048)
    parser.add_argument("--scan-bytes", type=int, default=1024)
    parser.add_argument("--replay-scan", action="store_true")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument(
        "--permission-mode", choices=("learned", "activity"), default="learned"
    )
    parser.add_argument("--cross-receptor-fallback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        replay_scan=args.replay_scan,
        gain=args.gain,
        permission_mode=args.permission_mode,
        cross_receptor_fallback=args.cross_receptor_fallback,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
