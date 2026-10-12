"""Automatic discovery of persistent-assembly-based relations (design doc
sections 21-22 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`;
the binding fix in
`Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`
section 6).

Mirrors `relation_discovery.py`'s trace-based discovery, but pairs the
anchor with a `PersistentBigramAssemblies` flag ("did token j immediately
follow token i, sticky since the last episode reset") instead of a
decaying trace -- this is what actually solves the first-entity binding
problem that decaying traces cannot (see the implementation journal).

One real difference from `relation_discovery.py`'s approach that matters:
these candidates are trained on *anchor-conditioned* samples (only
positions where the anchor is active), not samples drawn from the whole
diary. The useful relationship here is a *conditional* one ("given we're
at the anchor, does the assembly's binary state predict the outcome"),
not a symmetric two-way interaction the way XOR/parity's parents are --
training on the unconditional mix (as `relation_discovery.py` does for
continuous traces) systematically fails here because the assembly
parent is degenerate-binary and the anchor indicator becomes a second,
independent binary parent; the natural SAME/DIFFERENT solution the
generic dendrite finds for *that* framing is not what predicts the
downstream target. Conditioning on the anchor removes that spurious
degree of freedom, at the cost of J itself becoming uninformative
(anchor variance collapses to zero) -- so acceptance here is governed
purely by predictive information gain, not a J quality gate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from manifoldflow.dendritic_basin.persistent_assembly import PersistentBigramAssemblies
from manifoldflow.dendritic_basin.relation_unit import RelationUnit


def build_assembly_diary(
    token_stream: list[int], vocab_size: int, reset_channel: int
) -> list[tuple[int, np.ndarray, int]]:
    """One pass over `token_stream`, recording (token_id_t,
    assembly_matrix_t, token_id_{t+1}). `assembly_matrix_t` is the full
    (vocab_size, vocab_size) sticky bigram-assembly state *after*
    incorporating token_t (see `PersistentBigramAssemblies`).
    """
    assemblies = PersistentBigramAssemblies(vocab_size, reset_channel)
    diary: list[tuple[int, np.ndarray, int]] = []
    for t in range(len(token_stream) - 1):
        token_id = token_stream[t]
        assemblies.update(token_id)
        snapshot = assemblies.active.copy().astype(float)
        diary.append((token_id, snapshot, token_stream[t + 1]))
    return diary


def _make_anchor_conditioned_draw_batch(
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
    first_channel: int,
    second_channel: int,
) -> Callable[[int, np.random.Generator], list[tuple[float, float]]]:
    anchor_positions = np.array(
        [i for i, (tok, _m, _n) in enumerate(diary) if tok == anchor]
    )

    def draw_batch(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        idx = rng.choice(anchor_positions, size=n, replace=True)
        samples = []
        for i in idx:
            _tok, matrix, _next_token = diary[int(i)]
            samples.append((1.0, float(matrix[first_channel, second_channel])))
        return samples

    return draw_batch


def discover_assembly_relations(
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
    first_channels: list[int],
    second_channels: list[int],
    seed: int,
    n_candidates: int = 3,
    rounds: int = 60,
    perturbations_per_round: int = 12,
    step_size: float = 0.3,
    min_gain: float = 0.1,
    episodes_per_candidate: int = 5,
) -> list[tuple[tuple[int, int], RelationUnit, float]]:
    """Train one anchor-conditioned RelationUnit per (first_channel,
    second_channel) assembly candidate (restricted to the given candidate
    lists, since the full vocab^2 space is usually not worth searching
    blindly -- see `relation_discovery.py`'s module docstring on candidate
    selection needing to stay sparse at scale). Returns (key, unit, gain)
    sorted by descending information gain, keeping only those >= `min_gain`.

    `episodes_per_candidate` > 1 retries each candidate with fresh
    dendrites, keeping whichever episode scores the highest gain -- the
    same "not every single episode converges as cleanly" finding as every
    other discovery mechanism in this package (Phase 0/1's XOR and parity
    validation, Phase 2's trace discovery).
    """
    rng = np.random.default_rng(seed)
    scored: list[tuple[tuple[int, int], RelationUnit, float]] = []
    uid = 0
    for first_channel in first_channels:
        for second_channel in second_channels:
            if first_channel == second_channel:
                continue
            draw_batch = _make_anchor_conditioned_draw_batch(
                diary, anchor, first_channel, second_channel
            )
            key = (first_channel, second_channel)
            best_unit: RelationUnit | None = None
            best_gain = -np.inf
            for _episode in range(episodes_per_candidate):
                unit = RelationUnit(
                    n_candidates=n_candidates, seed=seed + uid, acceptance_threshold=0.0
                )
                unit.adapt(draw_batch, rounds, perturbations_per_round, step_size, rng)
                uid += 1
                gain = assembly_information_gain(unit, key, diary, anchor)
                if gain > best_gain:
                    best_gain = gain
                    best_unit = unit
            if best_unit is not None and best_gain >= min_gain:
                scored.append((key, best_unit, best_gain))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored


def assembly_information_gain(
    unit: RelationUnit,
    key: tuple[int, int],
    diary: list[tuple[int, np.ndarray, int]],
    anchor: int,
) -> float:
    """Same construction as `relation_discovery.relation_information_gain`,
    adapted for an assembly-matrix diary: entropy reduction in the real
    next-token distribution when the (anchor-conditioned) relation fires.
    """
    first_channel, second_channel = key
    vocab_size = diary[0][1].shape[0]
    with_relation_counts = np.zeros(vocab_size)
    without_relation_counts = np.zeros(vocab_size)
    for token_id, matrix, next_token in diary:
        if token_id != anchor:
            continue
        activation = unit.respond(1.0, float(matrix[first_channel, second_channel]))
        if activation > 0.5:
            with_relation_counts[next_token] += 1
        else:
            without_relation_counts[next_token] += 1

    def entropy(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    if with_relation_counts.sum() == 0:
        return 0.0
    base_entropy = entropy(with_relation_counts + without_relation_counts)
    return base_entropy - entropy(with_relation_counts)
