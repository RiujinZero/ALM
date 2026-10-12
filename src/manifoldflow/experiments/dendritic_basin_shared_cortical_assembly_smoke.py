"""Exercise a shared basal/apical cortical assembly substrate on live streams."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.shared_cortical_assemblies import (
    SharedCorticalAssemblies,
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


def _codes(arm: object) -> tuple[np.ndarray, np.ndarray]:
    basal = arm.field.cortical_code(arm.pending_access_context)
    apical = np.zeros_like(basal, dtype=np.float64)
    for (engram, _assembly, _receptor), amount in (
        arm.pending_receptor_relational_population or {}
    ).items():
        if not 0 <= engram < len(arm.field.engram_codes):
            continue
        frames = arm.field.engram_codes[engram]
        if not frames:
            continue
        for dimension in frames[-1]:
            apical[dimension] += float(amount)
    maximum = float(np.max(apical, initial=0.0))
    if maximum > 0.0:
        apical /= maximum
    return basal, apical


def _assembly_distribution(
    learner: object,
    tissue: SharedCorticalAssemblies,
    features: np.ndarray,
) -> np.ndarray:
    baseline = learner.predict_next_distribution().copy()
    prefix = bytes(learner.current_word)
    votes = np.zeros(256, dtype=np.float64)
    for assembly, activity in tissue.activations(features).items():
        if not 0 <= assembly < len(learner.word_forms):
            continue
        form = learner.word_forms[assembly]
        if len(form) <= len(prefix) or not form.startswith(prefix):
            continue
        votes[form[len(prefix)]] += activity
    total = float(np.sum(votes))
    if total <= 0.0:
        return baseline
    votes /= total
    weight = min(0.25, total / (total + 4.0))
    return (1.0 - weight) * baseline + weight * votes


def _experience(
    arm: object, tissue: SharedCorticalAssemblies, stream: bytes
) -> None:
    trace = np.zeros(tissue.dimensions, dtype=np.float64)
    for byte_id in stream:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        basal, apical = _codes(arm)
        features = tissue.features(basal, apical)
        np.maximum(trace, features, out=trace)
        if not (65 <= byte_id <= 90 or 97 <= byte_id <= 122):
            if arm.learner.current_word and np.any(trace):
                word = bytes(arm.learner.current_word).lower()
                assembly = arm.learner._word_id(word, create=True)
                if assembly is not None:
                    tissue.observe_external(trace, assembly)
            trace.fill(0.0)
        _observe(arm, byte_id, float(baseline[byte_id]), external=True)


def _develop(
    payload: dict[str, object], stream: bytes
) -> tuple[object, SharedCorticalAssemblies]:
    arm = phase150._configure(payload)
    arm.relational_residual_codevelopment = False
    tissue = SharedCorticalAssemblies(arm.field.cortical_dimensions)
    _experience(arm, tissue, stream)
    return arm, tissue


def _generate(
    prepared: object,
    tissue: SharedCorticalAssemblies,
    uniforms: np.ndarray,
    *,
    recall: bool,
    temperature: float,
    internal_evidence_mode: str = "hold",
) -> dict[str, object]:
    arm = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    emitted = bytearray()
    affected = 0
    recall_variation = 0.0
    internal_evidence = np.zeros(tissue.dimensions, dtype=np.float64)
    internal_relation_support = 0.0
    relation_supports: list[float] = []
    for uniform in uniforms:
        _predict(arm, live=True)
        basal, apical = _codes(arm)
        native_features = tissue.features(basal, apical if recall else None)
        if recall:
            # Internally committed content persists only in proportion to its
            # independently reconstructed relation with the current cortex.
            # This is mutual constraint, not a next-token selector.
            features = np.maximum(
                native_features,
                (
                    internal_relation_support
                    if internal_evidence_mode == "relation_conditioned"
                    else 1.0
                )
                * internal_evidence,
            )
        else:
            features = native_features
        distribution = _assembly_distribution(arm.learner, tissue, features)
        baseline = arm.learner.predict_next_distribution()
        if recall:
            basal_distribution = _assembly_distribution(
                arm.learner, tissue, tissue.features(basal)
            )
            recall_variation += 0.5 * float(
                np.sum(np.abs(distribution - basal_distribution))
            )
        affected += int(not np.array_equal(distribution, baseline))
        if temperature <= 0.0:
            byte_id = int(np.argmax(distribution))
        else:
            logits = np.log(np.maximum(distribution, 1e-300)) / temperature
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(np.sum(probabilities))
            byte_id = min(
                255,
                int(np.searchsorted(np.cumsum(probabilities), float(uniform))),
            )
        if recall:
            if (
                not (65 <= byte_id <= 90 or 97 <= byte_id <= 122)
                and arm.learner.current_word
            ):
                word = bytes(arm.learner.current_word).lower()
                assembly = arm.learner._word_id(word, create=False)
                if assembly is not None and assembly in tissue.weights:
                    internal_evidence = tissue.reconstruct({assembly: 1.0})
                    internal_relation_support = tissue.activations(
                        native_features
                    ).get(assembly, 0.0)
                    relation_supports.append(internal_relation_support)
                else:
                    internal_evidence.fill(0.0)
                    internal_relation_support = 0.0
        emitted.append(byte_id)
        _advance_generated_state(
            arm, byte_id, float(max(distribution[byte_id], 1e-300))
        )
    return {
        "text": emitted.decode("utf-8", errors="backslashreplace"),
        "hex": emitted.hex(),
        "affected_bytes": affected,
        "recall_total_variation": recall_variation,
        "relation_supported_commits": len(relation_supports),
        "mean_relation_support": (
            float(np.mean(relation_supports)) if relation_supports else 0.0
        ),
        "maximum_relation_support": max(relation_supports, default=0.0),
        "internal_evidence_mode": internal_evidence_mode,
    }


def _generation_diagnostics(result: dict[str, object]) -> dict[str, object]:
    raw = bytes.fromhex(str(result["hex"]))
    words = re.findall(rb"[A-Za-z]+", raw.lower())
    windows = [raw[index : index + 4] for index in range(max(0, len(raw) - 3))]
    return {
        "valid_utf8": _is_valid_utf8(raw),
        "word_count": len(words),
        "unique_word_fraction": (
            len(set(words)) / len(words) if words else 0.0
        ),
        "repeated_4gram_fraction": (
            1.0 - len(set(windows)) / len(windows) if windows else 0.0
        ),
    }


def _is_valid_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    scan_bytes: int,
    prompt: bytes | None = None,
    generation_bytes: int = 0,
    temperature: float = 0.8,
    seed: int = 20260910,
    generation_samples: int = 1,
    internal_evidence_mode: str = "hold",
) -> dict[str, object]:
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
    arm, tissue = _develop(payload, stream[:development_bytes])
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    rows = []
    for offset, byte_id in enumerate(stream[development_bytes:required]):
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        basal, apical = _codes(arm)
        basal_only = _assembly_distribution(
            arm.learner, tissue, tissue.features(basal)
        )
        combined = _assembly_distribution(
            arm.learner, tissue, tissue.features(basal, apical)
        )
        variation = 0.5 * float(np.sum(np.abs(combined - basal_only)))
        if variation > 0.0:
            rows.append(
                {
                    "position": development_bytes + offset,
                    "recall_utility": float(
                        np.log2(max(combined[byte_id], 1e-300))
                        - np.log2(max(basal_only[byte_id], 1e-300))
                    ),
                    "combined_utility": float(
                        np.log2(max(combined[byte_id], 1e-300))
                        - np.log2(max(baseline[byte_id], 1e-300))
                    ),
                    "variation": variation,
                }
            )
        _advance_generated_state(arm, byte_id, float(baseline[byte_id]))
    recall = np.asarray([row["recall_utility"] for row in rows])
    combined = np.asarray([row["combined_utility"] for row in rows])
    generation = None
    if prompt is not None and generation_bytes > 0:
        phase145._reset_dynamic_state_for_replay(arm)
        _experience(arm, tissue, prompt)
        pairs = []
        for sample_index in range(generation_samples):
            sample_seed = seed + sample_index
            uniforms = np.random.default_rng(sample_seed).random(
                generation_bytes
            )
            off = _generate(
                arm,
                tissue,
                uniforms,
                recall=False,
                temperature=temperature,
                internal_evidence_mode=internal_evidence_mode,
            )
            on = _generate(
                arm,
                tissue,
                uniforms,
                recall=True,
                temperature=temperature,
                internal_evidence_mode=internal_evidence_mode,
            )
            off["diagnostics"] = _generation_diagnostics(off)
            on["diagnostics"] = _generation_diagnostics(on)
            pairs.append(
                {
                    "seed": sample_seed,
                    "basal_only": off,
                    "basal_apical": on,
                    "outputs_differ": off["hex"] != on["hex"],
                }
            )
        generation = {
            "prompt": prompt.decode("utf-8", errors="backslashreplace"),
            "internal_evidence_mode": internal_evidence_mode,
            "samples": pairs,
            "outputs_differ_fraction": float(
                np.mean([pair["outputs_differ"] for pair in pairs])
            ),
            "mean_internal_total_variation": float(
                np.mean(
                    [
                        pair["basal_apical"]["recall_total_variation"]
                        for pair in pairs
                    ]
                )
            ),
        }
        if generation_samples == 1:
            generation.update(pairs[0])
    return {
        "task": "shared_cortical_assembly_smoke",
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "scan_bytes": scan_bytes,
        "assemblies": len(tissue.weights),
        "synapses": sum(len(row) for row in tissue.weights.values()),
        "embodied_bytes": tissue.embodied_bytes,
        "affected_bytes": len(rows),
        "recall_utility_sum": float(np.sum(recall)) if len(recall) else 0.0,
        "recall_positive_fraction": (
            float(np.mean(recall > 0.0)) if len(recall) else 0.0
        ),
        "combined_utility_sum": (
            float(np.sum(combined)) if len(combined) else 0.0
        ),
        "generation": generation,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=1024)
    parser.add_argument("--scan-bytes", type=int, default=1024)
    parser.add_argument("--prompt")
    parser.add_argument("--generation-bytes", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--generation-samples", type=int, default=1)
    parser.add_argument(
        "--internal-evidence-mode",
        choices=("hold", "relation_conditioned"),
        default="hold",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        scan_bytes=args.scan_bytes,
        prompt=None if args.prompt is None else args.prompt.encode(),
        generation_bytes=args.generation_bytes,
        temperature=args.temperature,
        seed=args.seed,
        generation_samples=args.generation_samples,
        internal_evidence_mode=args.internal_evidence_mode,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
