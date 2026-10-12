"""Phase 153: shared-pool competition, TD-bootstrapped credit, felt agency.

The preceding shared-cortical-assembly line (phases through the
"relation-conditioned internal recruitment ablation") closed on a specific
open problem: gating whether an internally emitted word assembly should
keep constraining generation cannot rely on raw relation support, because
the current cortical representation does not express that support strongly
enough to regulate continuity by itself (rejecting the signal just recreates
silence). The stated missing primitive was a genuine causal episode-state
transition, not a corpus-frequency preference.

This experiment adds three general-purpose mechanisms and tests whether,
together, they give a better-founded version of that gate:

1. Shared-pool normalized competition (``contextual_competition.py``)
   replaces the ad hoc ``min(0.25, total / (total + 4))`` linear blend used
   to combine competing word-assembly votes with the boxed
   ``d_i = (A_i E_i)^n``, ``r_i = d_i / (kappa + sum_j d_j)`` equation.
2. TD-bootstrapped consequence credit (``temporal_difference_credit.py``)
   replaces the plain one-step utility signal with
   ``delta_t = u_{t+1} + gamma V(x_{t+1}) - V(x_t)`` over a per-word-prefix
   local value table, giving "was this genuinely useful, accounting for
   what followed" rather than only "was the very next byte better."
3. Bayesian felt-agency attribution (``causal_attribution.py``) replaces the
   rejected raw-support gate with ``P(C=self | d)``, calibrated only from
   real external consequences observed during teacher-forced development
   (never from generated or recalled activity confirming itself), then used
   online during free generation to decide whether the currently completed
   word's assembly should be held as persistent apical context.

All three are evaluated on real TinyStories-solo and TinyStories/SmolTalk
mixed streams, using the same paired-generation, matched-randomness
methodology as every prior phase in this line, so results are directly
comparable to the existing "hold" and "relation_conditioned" arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.causal_attribution import SelfCausalAttribution
from manifoldflow.dendritic_basin.contextual_competition import normalized_competition
from manifoldflow.dendritic_basin.temporal_difference_credit import LocalValueEstimate
from manifoldflow.experiments import (
    dendritic_basin_phase91_developmental_successor as phase91,
)
from manifoldflow.experiments import (
    dendritic_basin_shared_cortical_assembly_smoke as smoke,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import _predict
from manifoldflow.experiments.dendritic_basin_phase102_phase_aligned_chain import (
    _advance_generated_state,
)


def _competitive_assembly_distribution(
    learner: object,
    tissue: smoke.SharedCorticalAssemblies,
    features: np.ndarray,
    *,
    kappa: float,
) -> np.ndarray:
    """Blend competing word-assembly votes through normalized competition.

    Same role as ``smoke._assembly_distribution``, but the fraction of the
    baseline distribution replaced by assembly votes -- and how the votes
    themselves are weighted against each other -- both come from the boxed
    shared-pool equation instead of a fixed linear cap.
    """
    baseline = learner.predict_next_distribution().copy()
    prefix = bytes(learner.current_word)
    activity = tissue.activations(features)
    drive: list[float] = []
    votes: list[np.ndarray] = []
    for assembly, value in activity.items():
        if not 0 <= assembly < len(learner.word_forms):
            continue
        form = learner.word_forms[assembly]
        if len(form) <= len(prefix) or not form.startswith(prefix):
            continue
        vote = np.zeros(256, dtype=np.float64)
        vote[form[len(prefix)]] = 1.0
        drive.append(float(value))
        votes.append(vote)
    if not votes:
        return baseline
    drive_array = np.asarray(drive, dtype=np.float64)
    gain_array = np.ones_like(drive_array)
    influence = normalized_competition(drive_array, gain_array, kappa=kappa)
    total_influence = float(np.sum(influence))
    if total_influence <= 0.0:
        return baseline
    combined_votes = np.zeros(256, dtype=np.float64)
    for weight, vote in zip(influence, votes, strict=True):
        combined_votes += weight * vote
    return (1.0 - total_influence) * baseline + combined_votes


def _calibrate(
    arm: object,
    tissue: smoke.SharedCorticalAssemblies,
    stream: bytes,
    *,
    gamma: float,
    kappa: float,
) -> tuple[LocalValueEstimate, SelfCausalAttribution]:
    """Teacher-forced pass building the value table and causal scales.

    Every update here is driven by a real subsequently-observed external
    byte; nothing generated or recalled ever supplies its own confirmation.
    """
    value = LocalValueEstimate(learning_rate=0.2)
    attribution = SelfCausalAttribution(self_prior_scale=1.0, other_prior_scale=0.1)
    pending: tuple[bytes, float, float] | None = None
    for byte_id in stream:
        baseline = arm.learner.predict_next_distribution()
        _predict(arm, live=True)
        basal, apical = smoke._codes(arm)
        basal_only = _competitive_assembly_distribution(
            arm.learner, tissue, tissue.features(basal), kappa=kappa
        )
        combined = _competitive_assembly_distribution(
            arm.learner, tissue, tissue.features(basal, apical), kappa=kappa
        )
        variation = 0.5 * float(np.sum(np.abs(combined - basal_only)))
        context = bytes(arm.learner.current_word)
        if pending is not None:
            previous_context, previous_utility, previous_variation = pending
            delta = value.update(
                previous_context,
                outcome=previous_utility,
                next_key=context,
                gamma=gamma,
            )
            attribution.confirm(
                previous_variation, cause="self" if delta > 0.0 else "other"
            )
            pending = None
        if variation > 0.0:
            utility = float(
                np.log2(max(combined[byte_id], 1e-300))
                - np.log2(max(baseline[byte_id], 1e-300))
            )
            pending = (context, utility, variation)
        _advance_generated_state(arm, byte_id, float(baseline[byte_id]))
    if pending is not None:
        previous_context, previous_utility, previous_variation = pending
        delta = value.update(
            previous_context, outcome=previous_utility, next_key=None, gamma=gamma
        )
        attribution.confirm(
            previous_variation, cause="self" if delta > 0.0 else "other"
        )
    return value, attribution


def _generate_gated(
    prepared: object,
    tissue: smoke.SharedCorticalAssemblies,
    uniforms: np.ndarray,
    *,
    recall: bool,
    temperature: float,
    kappa: float,
    attribution: SelfCausalAttribution | None,
    gate_threshold: float,
) -> dict[str, object]:
    import pickle

    arm = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    emitted = bytearray()
    affected = 0
    recall_variation = 0.0
    internal_evidence = np.zeros(tissue.dimensions, dtype=np.float64)
    held = False
    posteriors: list[float] = []
    for uniform in uniforms:
        _predict(arm, live=True)
        basal, apical = smoke._codes(arm)
        native_features = tissue.features(basal, apical if recall else None)
        if recall and held:
            features = np.maximum(native_features, internal_evidence)
        else:
            features = native_features
        distribution = _competitive_assembly_distribution(
            arm.learner, tissue, features, kappa=kappa
        )
        baseline = arm.learner.predict_next_distribution()
        if recall:
            basal_distribution = _competitive_assembly_distribution(
                arm.learner, tissue, tissue.features(basal), kappa=kappa
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
                255, int(np.searchsorted(np.cumsum(probabilities), float(uniform)))
            )
        if recall:
            word_boundary = not (65 <= byte_id <= 90 or 97 <= byte_id <= 122)
            if word_boundary and arm.learner.current_word:
                word = bytes(arm.learner.current_word).lower()
                assembly = arm.learner._word_id(word, create=False)
                if assembly is not None and assembly in tissue.weights:
                    candidate_support = tissue.activations(native_features).get(
                        assembly, 0.0
                    )
                    posterior = (
                        1.0
                        if attribution is None
                        else attribution.posterior_self(candidate_support)
                    )
                    posteriors.append(posterior)
                    held = posterior > gate_threshold
                    internal_evidence = (
                        tissue.reconstruct({assembly: 1.0})
                        if held
                        else np.zeros_like(internal_evidence)
                    )
                else:
                    held = False
                    internal_evidence = np.zeros_like(internal_evidence)
        emitted.append(byte_id)
        _advance_generated_state(
            arm, byte_id, float(max(distribution[byte_id], 1e-300))
        )
    return {
        "text": emitted.decode("utf-8", errors="backslashreplace"),
        "hex": emitted.hex(),
        "affected_bytes": affected,
        "recall_total_variation": recall_variation,
        "mean_posterior_self": (
            float(np.mean(posteriors)) if posteriors else 0.0
        ),
        "held_fraction": (
            float(np.mean([p > gate_threshold for p in posteriors]))
            if posteriors
            else 0.0
        ),
    }


def run(
    state: Path,
    tiny_path: Path,
    *,
    secondary_path: Path | None,
    development_bytes: int,
    prompt: bytes,
    generation_bytes: int,
    generation_samples: int,
    temperature: float = 0.8,
    seed: int = 20260910,
    gamma: float = 0.5,
    kappa: float = 0.05,
    gate_threshold: float = 0.5,
) -> dict[str, object]:
    import pickle

    with state.open("rb") as handle:
        payload = pickle.load(handle)
    tiny = tiny_path.read_bytes()
    secondary = None if secondary_path is None else secondary_path.read_bytes()
    mode = "tiny" if secondary is None else "mixed"
    stream = phase91._development_stream(
        tiny,
        secondary if secondary is not None else b"",
        mode=mode,
        byte_count=development_bytes,
        chunk_bytes=256,
    )
    arm, tissue = smoke._develop(payload, stream)
    value, attribution = _calibrate(arm, tissue, stream, gamma=gamma, kappa=kappa)

    from manifoldflow.experiments import (
        dendritic_basin_phase145_residual_route_codevelopment as phase145,
    )

    pairs: list[dict[str, object]] = []
    for sample_index in range(generation_samples):
        sample_seed = seed + sample_index
        phase145._reset_dynamic_state_for_replay(arm)
        smoke._experience(arm, tissue, prompt)
        uniforms = np.random.default_rng(sample_seed).random(generation_bytes)
        off = _generate_gated(
            arm,
            tissue,
            uniforms,
            recall=False,
            temperature=temperature,
            kappa=kappa,
            attribution=None,
            gate_threshold=gate_threshold,
        )
        held_unconditional = _generate_gated(
            arm,
            tissue,
            uniforms,
            recall=True,
            temperature=temperature,
            kappa=kappa,
            attribution=None,
            gate_threshold=-1.0,
        )
        causal_gated = _generate_gated(
            arm,
            tissue,
            uniforms,
            recall=True,
            temperature=temperature,
            kappa=kappa,
            attribution=attribution,
            gate_threshold=gate_threshold,
        )
        for result in (off, held_unconditional, causal_gated):
            result["diagnostics"] = smoke._generation_diagnostics(result)
        pairs.append(
            {
                "seed": sample_seed,
                "recall_off": off,
                "competitive_hold": held_unconditional,
                "causal_gated": causal_gated,
            }
        )

    return {
        "task": "phase153_context_competition_consequence_agency",
        "secondary_path": None if secondary_path is None else str(secondary_path),
        "development_bytes": development_bytes,
        "generation_bytes": generation_bytes,
        "generation_samples": generation_samples,
        "gamma": gamma,
        "kappa": kappa,
        "gate_threshold": gate_threshold,
        "calibration": {
            "observations": attribution.observations,
            "confirmed_self": attribution.confirmed_self,
            "confirmed_other": attribution.confirmed_other,
            "self_scale": attribution.self_scale.scale,
            "other_scale": attribution.other_scale.scale,
            "value_table_size": len(value.values),
        },
        "samples": pairs,
        "mean_competitive_hold_variation": float(
            np.mean(
                [pair["competitive_hold"]["recall_total_variation"] for pair in pairs]
            )
        ),
        "mean_causal_gated_variation": float(
            np.mean([pair["causal_gated"]["recall_total_variation"] for pair in pairs])
        ),
        "mean_held_fraction": float(
            np.mean([pair["causal_gated"]["held_fraction"] for pair in pairs])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("tiny", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--development-bytes", type=int, default=1024)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--generation-bytes", type=int, default=256)
    parser.add_argument("--generation-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.tiny,
        secondary_path=args.secondary,
        development_bytes=args.development_bytes,
        prompt=args.prompt.encode(),
        generation_bytes=args.generation_bytes,
        generation_samples=args.generation_samples,
        temperature=args.temperature,
        seed=args.seed,
        gamma=args.gamma,
        kappa=args.kappa,
        gate_threshold=args.gate_threshold,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
