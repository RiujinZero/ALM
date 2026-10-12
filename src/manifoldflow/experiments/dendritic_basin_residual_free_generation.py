"""Generate bytes with the best signed residual-return mechanism enabled."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

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
from manifoldflow.experiments.dendritic_basin_phase148_geometric_permission import (
    GeometricPermission,
)


def _clear_pending(arm: object) -> None:
    arm.pending_temporal_population = {}
    arm.pending_temporal_predictions = {}
    arm.pending_temporal_global_prediction = None
    arm.pending_relational_receptors = {}
    arm.pending_receptor_relational_population = {}


def _experience(
    arm: object,
    permission: GeometricPermission,
    stream: bytes,
    *,
    gain: float,
) -> int:
    shadows = 0
    for byte_id in stream:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        query = arm.relational_access.current_query.copy()
        candidates = phase150._candidate_predictions(arm, baseline, gain=gain)
        for relation, candidate in candidates.items():
            permission.observe(
                query,
                relation[2],
                phase150._utility(candidate, baseline, byte_id),
            )
        shadows += len(candidates)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)
    return shadows


def _sample(distribution: np.ndarray, uniform: float, temperature: float) -> int:
    if temperature <= 0.0:
        return int(np.argmax(distribution))
    logits = np.log(np.maximum(distribution, 1e-300)) / temperature
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return min(255, int(np.searchsorted(np.cumsum(probabilities), uniform)))


def _generate(
    prepared: object,
    permission: GeometricPermission,
    uniforms: np.ndarray,
    *,
    recall: bool,
    gain: float,
    temperature: float,
) -> dict[str, object]:
    arm = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.relational_residual_codevelopment = False
    emitted = bytearray()
    releases = 0
    total_variation = 0.0
    previous_relations: set[tuple[int, int, int]] = set()
    trace = []
    started = perf_counter()
    for position, uniform in enumerate(uniforms):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        population = arm.pending_receptor_relational_population or {}
        distribution = baseline
        chosen = None
        if recall:
            query = arm.relational_access.current_query.copy()
            support: dict[tuple[int, int], float] = {}
            for relation, activity in population.items():
                if relation in previous_relations:
                    continue
                _engram, assembly, receptor = relation
                if (receptor, assembly) not in arm.relational_residual_routes:
                    continue
                amount = float(activity) * permission.permission(query, receptor)
                if amount > 0.0:
                    support[assembly, receptor] = amount
            if support:
                chosen = max(
                    support, key=lambda item: (support[item], -item[0], -item[1])
                )
                distribution = phase145._residual_overlay_distribution(
                    arm.learner,
                    arm.relational_residual_routes,
                    (chosen,),
                    support={chosen: support[chosen]},
                    gain=gain,
                    baseline=baseline,
                )
                variation = 0.5 * float(np.sum(np.abs(distribution - baseline)))
                releases += int(variation > 0.0)
                total_variation += variation
        byte_id = _sample(distribution, float(uniform), temperature)
        if chosen is not None and len(trace) < 32:
            trace.append(
                {
                    "position": position,
                    "form": arm.learner.word_forms[chosen[0]].decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "receptor": chosen[1],
                    "baseline_byte": int(np.argmax(baseline)),
                    "return_byte": int(np.argmax(distribution)),
                    "emitted_byte": byte_id,
                }
            )
        emitted.append(byte_id)
        _advance_generated_state(
            arm, byte_id, float(max(distribution[byte_id], 1e-300))
        )
        previous_relations = set(population)
        _clear_pending(arm)
    return {
        "bytes": emitted.decode("utf-8", errors="backslashreplace"),
        "hex": emitted.hex(),
        "releases": releases,
        "total_variation": total_variation,
        "trace": trace,
        "bytes_per_second": len(emitted) / max(perf_counter() - started, 1e-12),
    }


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    prompt: bytes,
    generation_bytes: int,
    gain: float,
    risk_weight: float,
    temperature: float,
    seed: int,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    tiny = tiny_path.read_bytes()
    secondary = None if secondary_path is None else secondary_path.read_bytes()
    stream = (
        tiny[:development_bytes]
        if secondary is None
        else _development_stream(
            tiny,
            secondary,
            mode="mixed",
            byte_count=development_bytes,
            chunk_bytes=256,
        )
    )
    arm = phase150._configure(payload)
    permission = GeometricPermission(
        arm.field.cortical_dimensions, risk_weight=risk_weight
    )
    shadows = _experience(arm, permission, stream, gain=gain)
    phase145._reset_dynamic_state_for_replay(arm)
    shadows += _experience(arm, permission, prompt, gain=gain)
    uniforms = np.random.default_rng(seed).random(generation_bytes)
    off = _generate(
        arm,
        permission,
        uniforms,
        recall=False,
        gain=gain,
        temperature=temperature,
    )
    on = _generate(
        arm,
        permission,
        uniforms,
        recall=True,
        gain=gain,
        temperature=temperature,
    )
    return {
        "task": "residual_free_generation",
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "prompt": prompt.decode("utf-8", errors="backslashreplace"),
        "generation_bytes": generation_bytes,
        "temperature": temperature,
        "seed": seed,
        "shadow_trials": shadows,
        "permission_trials": permission.trials,
        "recall_disabled": off,
        "recall_enabled": on,
        "outputs_differ": off["hex"] != on["hex"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=2048)
    parser.add_argument(
        "--prompt", default="Once upon a time, there was a little girl named Lily. "
    )
    parser.add_argument("--generation-bytes", type=int, default=128)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        prompt=args.prompt.encode(),
        generation_bytes=args.generation_bytes,
        gain=args.gain,
        risk_weight=args.risk_weight,
        temperature=args.temperature,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
