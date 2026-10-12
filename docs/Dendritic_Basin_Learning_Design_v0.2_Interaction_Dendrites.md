# Dendritic Basin Learning v0.2: Interaction Dendrites

**Addendum to `Dendritic_Basin_Learning_Design_v0.1.md` -- replaces the
v0.1 prototype-distance dendrite (sections 5-9) with a generic
nonmonotonic "interaction dendrite" driven by a purely statistical local
objective. Sandbox findings + pseudocode for future implementation; not
yet wired into `src/manifoldflow/dendritic_basin/`.**

Status: Sandbox findings, Research Design v0.2 (pseudocode only)

## 0. Why this replaces v0.1's dendrite

The v0.1 implementation's `DendriticPopulation` (prototype-distance
matching against a literal encoding, grown by a hand-designed
surprise/poor-match trigger) passed the Phase 0 XOR ablations, but its
growth rule was overproposing prototypes -- consuming its entire branch
budget rather than converging on the small number of conjunctions the
task actually needs (see the implementation journal). While investigating
that, a sandbox experiment replacing the whole mechanism turned up
something more interesting than a fix to the growth rule: a *generic*
dendrite with no task-specific primitive in it, paired with a purely
statistical local objective, spontaneously discovers SAME/DIFFERENT
(XOR/XNOR) as the dominant two-input relation -- and appears to do so for
a principled, not coincidental, reason (section 6).

This document is the findings + forward-looking pseudocode from that
sandbox exploration. Nothing here has been ported into the actual
`dendritic_basin` package yet; see section 8 for the intended integration
points.

## 1. The core change: a generic nonmonotonic dendrite

Replace the explicit `DIFF(A, B)` (or any other hand-named primitive: AND,
MATCH, SUPPRESS, ...) with a single generic bell-shaped response over a
weighted sum of inputs:

```
d(A, B) = exp[ -(w_A*A + w_B*B - mu)^2 / (2*sigma^2) ]
```

A dendrite has exactly four local parameters: `w_A`, `w_B`, `mu`, `sigma`.
There is no XOR operator, no DIFF operator, no menu of primitives
anywhere in the definition. The dendrite fires (d near 1) when the
weighted drive `w_A*A + w_B*B` lands near `mu`, and fires less (d near 0)
the further the drive is from `mu`, at a rate set by `sigma`. This is a
strictly local, purely numerical computation -- no combinatorial logic.

## 2. The local learning objective

No label, no parity target, no output error, no gradient, no backward
signal. The objective is computed entirely from running local activity
statistics of the dendrite's own output and its two inputs:

```
J = Var(d) - Cov(d, A)^2 / Var(A) - Cov(d, B)^2 / Var(B)
```

In English: **be informative, but don't just copy parent A or parent B.**
`Var(d)` rewards the dendrite for varying at all (not collapsing to a
constant); the two subtracted terms penalize any variation that a linear
readout of `A` alone or `B` alone could already explain. What's left over
is the part of the dendrite's behavior that depends on `A` and `B`
*jointly* -- exactly the definition of a statistical interaction term
after removing first-order (main) effects.

## 3. Adaptation: local perturbation/selection, not gradient descent

The four parameters (`w_A, w_B, mu, sigma`) are adapted by local
perturbation and selection, not by differentiating `J`:

1. Propose a perturbed parameter set (small random step from the current
   one).
2. Evaluate `J` for the perturbation over recent (or freshly sampled)
   local activity.
3. Keep the perturbation if `J` improved; otherwise keep the current
   parameters.

This is a zeroth-order local search -- it never needs `dJ/dw` computed
analytically, only `J` evaluated at two parameter settings. Multiple
candidate dendrites can run this search in parallel and compete; the one
with the highest `J` (above a quality gate, section 5) wins and becomes
the relation unit's output.

## 4. It spontaneously invents SAME/DIFFERENT

Inspecting learned parameters after adaptation converged, two parameter
families recur:

**Family 1 (XOR-like):** `w_A ~= w_B ~= mu ~= 2.21`. The dendrite fires
near total drive = one active input:

```
00 -> 0
01 -> 1
10 -> 1
11 -> 0
```

This is `A XOR B`.

**Family 2 (XNOR-like):** `w_A ~= 4.21`, `w_B ~= -4.21`, `mu ~= 0`.
Cancellation occurs when the two inputs are equal:

```
00 -> 1
01 -> 0
10 -> 0
11 -> 1
```

This is `A XNOR B` (`A <-> B`).

SAME and DIFFERENT are both natural attractors of the same generic
dendritic physics -- neither was coded in anywhere.

## 5. Reliability and the local quality gate

- One dendrite, modest local perturbation search: ~95% reliable.
- Two competing dendrites, 60 local adaptation rounds, 12 perturbations
  per round: **300/300** successful SAME/DIFFERENT relations.
- Enumerating all 16 possible two-input Boolean functions: a perfect
  SAME/DIFFERENT dendrite scores `J = 0.25`; the next-best family only
  reaches `J = 0.0625` -- a 4x gap. This means the dendrite can locally
  self-assess whether it found a genuinely useful joint feature, with no
  external signal.
- Local acceptance threshold set at `J > 0.20`. Across 1,000 independent
  relation-learning trials: **1000/1000** found XOR/XNOR (95.5% on the
  first local search episode; every remaining case succeeded on the
  second episode). No task target was used to make the accept/reject
  decision -- only the dendrite's own `J`.

## 6. Hierarchical parity from composed relation units

Each hierarchical node independently learns its own relation from its two
inputs, e.g.:

```
r_1 = f(x_1, x_2)
r_2 = f(x_3, x_4)
r_3 = f(r_1, r_2)
```

where every `f` is an independently, locally learned interaction
dendrite. Whether a given node lands on XOR or XNOR doesn't matter for
overall parity, since either preserves the parity information up to a
global complement.

Scaling results (three independent complete trees per size, evaluated on
held-out random inputs):

| Bits | Learned relation units | Result |
|---|---|---|
| 128 | 127 | 100% |
| 256 | 255 | 100% |
| 512 | 511 | 100% |
| 1024 | 1023 | 100% |

All twelve trials: **100%**. The structural cost is `N-1` relation units
(a binary tree over `N` inputs), not `2^N` memorized patterns -- the
scaling transition the project has been after.

## 7. Robustness of the SAME/DIFFERENT result

**Marginal bias.** Varying `P(A=1) = P(B=1)` from 0.01 to 0.99 and
enumerating all 16 Boolean functions at each bias: SAME/DIFFERENT
remained the highest-scoring function at every bias tested.

**Arbitrary joint distributions.** Dropping the independence assumption
entirely: 10,000 joint distributions `P(00), P(01), P(10), P(11)` drawn
from a Dirichlet distribution, enumerating all 16 functions for each.
Result: **10,000/10,000** had XOR/XNOR as the maximum-`J` function.

### Why: J isolates the irreducible second-order interaction

Rewriting the objective:

```
J = Var(d) - Cov(d,A)^2/Var(A) - Cov(d,B)^2/Var(B)
```

this is asking for output variation that cannot be attributed to either
input individually -- in statistical language, the interaction term left
over after removing both main effects (an ANOVA-style decomposition). For
two binary variables under a `{-1, +1}` encoding (`a = 2A-1`, `b = 2B-1`),
the irreducible second-order interaction basis is essentially `a*b`:

```
a*b = +1 if A == B
a*b = -1 if A != B
```

which is exactly parity. So SAME/DIFFERENT isn't an arbitrary Boolean
trick the search happened to land on -- it is *the* pure interaction
direction remaining once first-order information is removed, for any
two-binary-variable joint distribution. That is the candidate for "the
mathematical principle" this line of work has been hunting for.

## 8. Where it breaks: coverage, not parity

The remaining weakness is **finite local experience**. If a dendrite
hasn't observed a given `(A, B)` combination, it has no local signal
telling it what that combination should produce -- and this compounds
badly through a hierarchy.

With uniformly sampled local inputs and no prior:

| Local examples | Relation success |
|---|---|
| 4 | ~24.5% |
| 8 | ~75% |
| 12 | ~87.5% |
| 16 | ~95.5% |
| 24 | ~99.5% |
| 32 | ~100% |

### Mitigation: a tiny homeostatic (pseudocount) prior

Adding a small pseudocount to the four possible local joint states before
computing statistics -- interpretable as spontaneous neural activity,
local exploratory noise, replay, or a weak prior -- nearly eliminates the
coverage problem. With a 0.1 pseudocount:

| Samples | Success |
|---|---|
| 4 | 96% |
| 8 | 96.5% |
| 12 | 98% |
| 16 | 100% |
| 24 | 100% |

Stronger pseudocounts saturate essentially immediately. This is not
claimed as a proof of anything biological -- the pseudocount here is an
imposed computational device -- but it does suggest a concrete functional
role for spontaneous activity/replay/noise in a local-plasticity
architecture: not biological mess, but a mechanism for local state-space
coverage that nonlinear interaction learning specifically depends on.

```
spontaneous activity -> local state-space coverage -> learn nonlinear interactions
```

## 9. Revised architecture

Drop the idea of a hand-designed menu of dendritic primitives (AND, DIFF,
MATCH, SUPPRESS, ...). Give every dendrite the same generic nonmonotonic
response and the same interaction-maximizing local pressure, let multiple
dendrites compete, and relation detectors emerge on their own:

```
x -> nonmonotonic dendrites (discover interactions)
  -> recursive assemblies (compose interactions)
  -> MF / recurrent basin (persistent state)
  -> y
```

The local learning principle compresses to one sentence:

> **Represent what your inputs only explain together.**

## 10. Pseudocode for future implementation

None of this is implemented yet in `src/manifoldflow/dendritic_basin/`.
The sketches below are for the next implementation pass; see section 11
for how they're expected to map onto (and replace parts of) the existing
v0.1 modules.

### 10.1 `InteractionDendrite`

```python
class InteractionDendrite:
    """A single generic nonmonotonic two-input dendrite (section 1) with
    the interaction-maximizing local objective (section 2), adapted by
    local perturbation/selection (section 3) -- no gradient anywhere.
    """

    def __init__(self, seed):
        rng = local_rng(seed)
        self.w_a = rng.normal(0, 1)
        self.w_b = rng.normal(0, 1)
        self.mu = rng.normal(0, 1)
        self.sigma = rng.uniform(0.5, 2.0)
        # Running local statistics (mean/var/cov), each with a small
        # pseudocount prior (section 8) mixed in at init so early
        # estimates aren't degenerate before any real data arrives.
        self.stats = RunningJointStats(pseudocount=0.1)

    def respond(self, a, b):
        drive = self.w_a * a + self.w_b * b
        return exp(-((drive - self.mu) ** 2) / (2 * self.sigma ** 2))

    def observe(self, a, b):
        """Update running statistics with one more (a, b, d) sample.
        Call this on every local exposure, not just during adaptation
        trials -- this is what section 8's coverage fix depends on.
        """
        d = self.respond(a, b)
        self.stats.update(a, b, d)

    def local_objective(self):
        """J = Var(d) - Cov(d,A)^2/Var(A) - Cov(d,B)^2/Var(B) (section 2),
        computed entirely from self.stats -- no label, no target.
        """
        var_d = self.stats.var_d()
        cov_da = self.stats.cov_d_a()
        cov_db = self.stats.cov_d_b()
        var_a = self.stats.var_a()
        var_b = self.stats.var_b()
        return (
            var_d
            - (cov_da ** 2) / (var_a + EPS)
            - (cov_db ** 2) / (var_b + EPS)
        )

    def perturb_and_select(self, samples, step_size, rng):
        """One local search step (section 3): propose a perturbed
        parameter set, evaluate J on `samples` (fresh (a, b) pairs), keep
        the perturbation only if J improves.
        """
        candidate = self.clone()
        candidate.w_a += rng.normal(0, step_size)
        candidate.w_b += rng.normal(0, step_size)
        candidate.mu += rng.normal(0, step_size)
        candidate.sigma = max(EPS, candidate.sigma + rng.normal(0, step_size))

        current_j = self._evaluate_on(samples)
        candidate_j = candidate._evaluate_on(samples)
        if candidate_j > current_j:
            self._copy_params_from(candidate)
            return candidate_j
        return current_j

    def _evaluate_on(self, samples):
        stats = RunningJointStats(pseudocount=0.1)
        for a, b in samples:
            stats.update(a, b, self.respond(a, b))
        return stats.objective()  # same formula as local_objective()
```

### 10.2 Competing dendrites -> a relation unit

```python
class RelationUnit:
    """N competing InteractionDendrites (section 3-5): runs local
    perturbation/selection search for a fixed budget, keeps whichever
    dendrite has the highest J, and only accepts it if J clears the
    local quality gate (section 5: J > ~0.20 for two binary parents).
    Falls back to "undecided" (e.g. pass-through or a neutral constant)
    if no candidate clears the gate -- never forces a low-confidence
    relation onto the hierarchy above it.
    """

    def __init__(self, n_candidates, seed, acceptance_threshold=0.20):
        self.candidates = [InteractionDendrite(seed + i) for i in range(n_candidates)]
        self.acceptance_threshold = acceptance_threshold
        self.winner = None

    def adapt(self, sample_source, rounds, perturbations_per_round, step_size, rng):
        for _ in range(rounds):
            samples = sample_source.draw_batch(perturbations_per_round)
            for candidate in self.candidates:
                candidate.perturb_and_select(samples, step_size, rng)
        best = max(self.candidates, key=lambda c: c.local_objective())
        if best.local_objective() > self.acceptance_threshold:
            self.winner = best
        return self.winner

    def respond(self, a, b):
        if self.winner is None:
            return NEUTRAL_OUTPUT  # e.g. 0.5, or raise/flag "undecided"
        return self.winner.respond(a, b)
```

### 10.3 Hierarchical composition (parity ladder, section 6)

```python
def build_relation_tree(leaf_count, seed):
    """N-1 RelationUnits over N leaves (a binary tree), each learning its
    own two-input relation independently and locally -- no shared
    parameters, no backward signal between levels.
    """
    nodes = [LeafNode(i) for i in range(leaf_count)]
    unit_id = 0
    while len(nodes) > 1:
        next_level = []
        for left, right in pairwise(nodes):
            unit = RelationUnit(n_candidates=2, seed=seed + unit_id)
            next_level.append(ComposedNode(unit, left, right))
            unit_id += 1
        nodes = next_level
    return nodes[0]  # root: the fully composed parity relation


def evaluate_tree(root, leaf_values):
    """Recursive forward pass; each ComposedNode's cached winner dendrite
    produces its output from its two children's (already-computed)
    outputs.
    """
    if isinstance(root, LeafNode):
        return leaf_values[root.index]
    left_value = evaluate_tree(root.left, leaf_values)
    right_value = evaluate_tree(root.right, leaf_values)
    return root.unit.respond(left_value, right_value)
```

### 10.4 Homeostatic pseudocount prior (section 8)

```python
class RunningJointStats:
    """Running mean/var/cov for (a, b, d), seeded with a small pseudocount
    over the (assumed-binary) joint states before any real data arrives
    -- section 8's coverage fix. Interpretable as spontaneous activity /
    replay / a weak prior, not just a numerical hack, per the design
    discussion in section 8.
    """

    def __init__(self, pseudocount=0.1):
        # Seed all four (a, b) in {0,1}x{0,1} cells with `pseudocount`
        # virtual observations at a neutral d (e.g. 0.5) before any real
        # exposure -- this is what turns "4 real samples -> ~24.5%
        # success" into "4 real samples -> ~96% success" (section 8).
        self._cells = {
            (a, b): VirtualCell(count=pseudocount, d_estimate=0.5)
            for a in (0, 1)
            for b in (0, 1)
        }

    def update(self, a, b, d):
        self._cells[(a, b)].add_observation(d)

    # var_a, var_b, var_d, cov_d_a, cov_d_b, objective(): computed from
    # self._cells' accumulated (count, sum, sum_sq, cross-sums) the usual
    # way; omitted here since they're standard moment arithmetic once the
    # pseudocount-seeded cells exist.
```

## 11. Integration points for the next implementation pass

These are notes for whoever picks this up next (human or otherwise), not
a commitment about how the port will actually go:

- `dendrites.py`'s `DendriticPopulation` (v0.1: prototype-distance
  matching, sections 5-9) is the direct analogue of `InteractionDendrite`
  above. The mask/prototype/eligibility machinery is v0.1-specific and
  would likely be replaced wholesale rather than extended -- the
  interaction dendrite has only 4 scalar parameters and no sparse mask.
- The v0.1 growth trigger (`adapt_structure`/`grow_branch`, section 8 of
  the v0.1 doc) is exactly the "overproposing prototypes" problem that
  prompted this exploration. `RelationUnit`'s quality gate (`J >
  threshold`, section 5 above) is a candidate replacement: only
  materialize/keep a relation unit once some candidate dendrite clears
  the gate, rather than growing on every surprise event.
- `basin.py`'s `RecurrentBasin` is unaffected by this change and is still
  expected to sit above the relation-unit hierarchy for persistent state
  (temporal XOR and beyond, per the v0.1 design's phase ladder) -- see
  the revised pipeline in section 9 above.
- The v0.1 XOR ablations (`experiments/dendritic_basin_xor_ablations.py`)
  and the parity ladder (`Parity_3/4/8/16...`, v0.1 section 26) are the
  natural first things to rerun once `InteractionDendrite`/`RelationUnit`
  are implemented, since section 6 above already has informal numbers
  (128-1024 bits, 100% across three trials each) to reproduce/verify
  formally with the same rigor as the v0.1 Phase 0 ablations (proper
  seeds, held-out evaluation, regression tests).
