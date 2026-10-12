"""Competing interaction dendrites -> a relation unit (design doc section
10.2 of `Dendritic_Basin_Learning_Design_v0.2_Interaction_Dendrites.md`).

Several `InteractionDendrite` candidates run local perturbation/selection
in parallel; the one with the highest local objective `J` wins, but only
if `J` clears a quality gate (section 5: perfect SAME/DIFFERENT scores
J=0.25 on balanced binary parents, the next-best family only 0.0625 -- a
large, locally measurable gap). A relation unit that never clears the
gate stays "undecided" and passes through a neutral response rather than
forcing a low-confidence relation onto whatever consumes it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from manifoldflow.dendritic_basin.interaction_dendrite import InteractionDendrite

NEUTRAL_OUTPUT = 0.5
DEFAULT_ACCEPTANCE_THRESHOLD = 0.20


class RelationUnit:
    """N competing InteractionDendrites, adapted for a fixed budget, with
    an acceptance gate on the winner's local objective.
    """

    def __init__(
        self,
        n_candidates: int,
        seed: int,
        acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    ):
        self.candidates = [InteractionDendrite(seed + i) for i in range(n_candidates)]
        self.acceptance_threshold = acceptance_threshold
        self.winner: InteractionDendrite | None = None
        self.winner_objective: float = 0.0

    def adapt(
        self,
        draw_batch: Callable[[int, np.random.Generator], list[tuple[float, float]]],
        rounds: int,
        perturbations_per_round: int,
        step_size: float,
        rng: np.random.Generator,
        eval_batch_size: int = 200,
    ) -> InteractionDendrite | None:
        """Run local perturbation/selection for `rounds` rounds against a
        *growing* replay buffer: each round draws `perturbations_per_round`
        fresh (a, b) samples and appends them, then every candidate's
        accept/reject decision is scored on the *entire* buffer so far,
        not just that round's small batch. This is what the design's own
        finding that "coverage" (total local examples seen) is what
        governs reliability actually implies -- deciding each round from
        only a handful of fresh samples is far noisier than deciding from
        everything observed so far, and the design's reported reliability
        (e.g. 300/300 at this same rounds/perturbations budget) assumed
        accumulating evidence, not discarding it every round.

        After the search, re-scores every candidate on one larger, fresh
        `eval_batch_size` batch (independent of the buffer, to avoid
        rewarding overfitting to it) and accepts the best if it clears the
        gate.
        """
        buffer: list[tuple[float, float]] = []
        for _ in range(rounds):
            buffer.extend(draw_batch(perturbations_per_round, rng))
            for candidate in self.candidates:
                candidate.perturb_and_select(buffer, step_size, rng)

        eval_samples = draw_batch(eval_batch_size, rng)
        scored = [(c, c.evaluate_on(eval_samples)) for c in self.candidates]
        best, best_j = max(scored, key=lambda pair: pair[1])

        if best_j > self.acceptance_threshold:
            self.winner = best
            self.winner_objective = best_j
        else:
            self.winner = None
            self.winner_objective = best_j
        return self.winner

    def respond(self, a: float, b: float) -> float:
        if self.winner is None:
            return NEUTRAL_OUTPUT
        return self.winner.respond(a, b)

    def is_accepted(self) -> bool:
        return self.winner is not None
