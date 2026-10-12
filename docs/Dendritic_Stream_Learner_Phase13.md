# Phase 13: Byte-Class Cortical ALM

## Purpose

Phase 13 turns the Phase 12 organism into a corpus-scale system without
changing its ontology into a conventional trained network. Capacity is an
enforced durable byte ceiling. Larger organisms grow modular cortical columns,
consolidate useful branches, and process raw corpus bytes through resumable
one-pass streams.

## Execution backend

Predictor evidence is held in contiguous branch matrices. Response,
persistence, trace publication, delayed predictor updates, and competition
are batched. Local sample deposition uses a fixed anatomical ring tensor.
The repeated perturb/select Welford objective is optionally compiled with
Numba, with scalar implementations retained as exact semantic oracles.

On the unchanged seed-0 100KB/20KB 256-slot trajectory:

| Backend checkpoint | Train bytes/s | Eval bytes/s | Validation bpb |
|---|---:|---:|---:|
| Phase 12 causal-depth batch | 704 | 4,131 | 4.0661 |
| Phase 13 compiled/batched | 3,534 | 5,818 | 4.0661 |

Births/prunes remain exactly 397/141. Three 2,000-byte oracle tests cover
response batching, predictor competition, and compiled local adaptation.

An experimental `event` metabolic backend now retains an inverted afferent
index and evaluates only branches reached by the current sensory/assembly
frontier or by still-active persistence. Its frozen causal predictions are
checked exactly against the dense backend. It is intentionally not the
default yet: on current checkpoints, Python routing overhead still loses to
contiguous NumPy (2,584 versus 3,181 bytes/s at 214 branches, and 467 versus
595 bytes/s around 1,700 branches under the same contention). The prototype
is preserved behind `--metabolic-backend event` for the larger-body crossover;
production remains `dense`. New reports count actual branch evaluations per
training byte so that crossover can be selected from metabolism rather than
nominal ALM class.

The dense path no longer materializes and checkpoints a complete 256-way
distribution for every locally firing branch. Predictor counts cannot change
between forming a prediction and receiving its next causal outcome, so pending
state retains only branch identity plus event/state kind. The next observation
reads its one required probability directly from those unchanged counts.
Competition normalizes full distributions for the at-most-eight winners only.
This cuts the mature ~1,700-branch competition profile from 0.34 to 0.09
seconds per 1,000 bytes. A compiled local-outcome kernel then applies utility,
predictor decay, and the observed count in place. Under matched contention,
the combined path raises ALM-2MB from 3,181 to 3,666 bytes/s and the mature
ALM-16MB checkpoint from 595 to 1,048 bytes/s. Scalar and compiled outcome
trajectories agree through a 2,000-byte structural oracle.

The causal-depth response arithmetic was next compiled by stratum. Each
stratum gathers its afferents, evaluates nonmonotonic branches, advances
retained state, and publishes assembly activity and trace injections inside
one in-place numeric kernel. With compiled response plus compiled outcome,
matched mature checkpoints reach 4,549 bytes/s at 214 branches and 1,510
bytes/s around 1,700 branches under two-run contention, versus 3,378 and 896
for the previous batched/scalar combination. A separate 2,000-byte oracle
preserves predictions, winners, births, prunes, and population statistics.

The remaining competition chain was also fused experimentally into one
compiled kernel covering eligibility, event/state evidence, column winners,
and mixture construction. It is causally identical, but the boundary and
return-buffer cost made it slower at 1,788 live branches (551 versus 645
training bytes/second under the same two-run contention), so `batched` remains
the default and the compiled route is retained only for crossover profiling.
The profiler instead exposed 128,945 branch-local Welford updates crossing
through Python objects per 1,000 bytes. Those durable statistics now live in
contiguous branch arrays and update in one compiled local pass. On the same
1,788-branch checkpoint and 5,000-byte learning segment this raises throughput
from 645 to 920 bytes/second under contention, with identical 3.674372 bpb.
`dendritic_basin_execution_benchmark` makes backend comparisons reproducible
from any exact checkpoint.

The causal response frontier is now one compiled cortical pass rather than
one call per depth stratum. Structural growth already enforces that relation
parents occupy lower slot IDs than their children, so anatomical slot order
is a valid topological order: each branch sees the present sensory event and
only already-published lower-depth assemblies. The pass now performs trace
decay, response, retained-state publication, trace injection, normalization,
and trace-homeostasis without returning to Python between cortical depths.
On the 214-live-branch ALM-2MB regime it reaches 4,671 training bytes/second
versus 3,235 for the prior batched path, with identical 3.818603 bpb. At 1,788
live branches it reaches 1,179 versus 890 bytes/second under concurrent ladder
load, with identical 3.695325 bpb.

Local perturb/select events are batched too: proposal draws retain anatomical
order and both objective comparisons execute in one call. New organisms use
counter-addressed branch-local proposal noise keyed by structural seed, slot,
and local observation count, so the same local event is independent of chunk
boundaries. Old checkpoints lacking that config field automatically retain
the legacy global RNG stream and exact continuation behavior.

That independence enables the sequential byte clock itself to stay compiled
between structural events. The chunk kernel resolves the previous prediction,
updates local evidence, advances the full topological cortical pass, performs
lifecycle and local-search updates, and forms the next prediction before
moving to the next byte. It returns to Python *before* any surprise-triggered
birth, so nursery recruitment/pruning changes the very next byte exactly as in
the one-at-a-time oracle. Byte-budget weigh-ins are boundaries as well. A
1,000-byte oracle with six columns and forced lifecycle transitions preserves
probabilities, traces, predictor evidence, anatomy, and population counters.
On the 214-branch checkpoint this raises end-to-end learning from 7,642 to
16,628 bytes/second under concurrent ladder load, with identical 3.853010 bpb.

## Enforced byte classes

`max_alm_bytes` is now the primary capacity constraint. It derives mature and
nursery carrying capacities from a planning estimate covering
predictor evidence, local sample rings, pending causal distributions, traces,
and structural metadata. The exact serialized body is the authority. While an
organism is near its mature carrying capacity, it periodically weighs itself;
if it crosses its byte ceiling, the weakest provisional nursery tissue is
evicted until it fits. Every checkpoint repeats this homeostatic check before
its atomic write. Consolidated anatomy is never silently removed to make a
checkpoint fit.

Named initial ceilings use decimal bytes:

| Class | Mature capacity | Nursery capacity | Planning allocation |
|---|---:|---:|---:|
| ALM-2MB | 214 | 214 | 1,993,856 bytes |
| ALM-16MB | 1,788 | 1,788 | 15,996,160 bytes |
| ALM-64MB | 7,184 | 7,184 | 63,998,976 bytes |
| ALM-256MB | 28,766 | 28,766 | 255,992,448 bytes |

The class is a ceiling, not ballast: reports include both the ceiling and the
organism's current exact serialized body. Nursery and mature capacity are
upper bounds rather than a promise that both populations can simultaneously
occupy every slot; they share the one embodied byte budget.

## Cortical columns

Slots are divided into fixed local columns (256 branches by default). Sensory
afferent pairs map deterministically into niches. Birth and replacement occur
inside the selected column. Each column selects a small number of local
winners; only the strongest columns contribute to the final soma-level
mixture. Recursive branches inherit a parent column, while relations spanning
columns require an explicit low-probability bridge event.

This produces modular specialization without assigning linguistic roles or
hand-designed features. A three-column synthetic probe fills all columns,
enforces two winners per column, and creates no bridge when bridge probability
is zero.

## Consolidation and dormancy

Branches now have three developmental states:

1. **Juvenile:** fast predictor decay and frequent perturb/select adaptation.
2. **Consolidated:** earned through age, firing, and positive predictive
   utility; protected from direct recycling and plastic on a slower clock.
3. **Dormant:** a consolidated branch that becomes stale or loses utility;
   removed from competition but still probes its afferents and can reactivate
   before becoming eligible for local replacement.

Replacement is column-local, prefers dormant anatomy after a minimum residence
period, and gives older repeatedly firing branches an age-adjusted survival
credit. Reports expose consolidation, dormancy, reactivation, lifecycle
histograms, branch ages, depth use, column occupancy, and sparse bridges.

A one-megabyte audit exposed a wake/sleep oscillation: a low-utility dormant
branch could fire once, wake, and immediately fail the utility condition that
put it back to sleep. Reactivation now requires both renewed afferent firing
and consolidated-level predictive utility. On a matched 10KB continuation,
the three classes now produce only 1/1, 7/1, and 2/2 dormancy/reactivation
events instead of millions of counter transitions. Reports preserve cumulative
population counts and separately expose the delta for each resumed rung.

## Truly streaming corpus runs

The experiment no longer loads selected train or validation regions into
lists. It yields bytes from bounded file chunks. Unigram, bigram, and trigram
controls are trained and evaluated in streaming passes as well. Periodic
learner checkpoints and progress sidecars are atomic and retain cumulative and
tail log-loss. Resume seeks directly to the exact corpus byte and preserves
the pending causal prediction. A 5KB-to-10KB resume probe exactly matches an
uninterrupted 10KB run in validation bpb, births, prunes, and serialized size.

Scale reports now include bpb, bytes/second, births and prunes per kilobyte,
branch survival/age, depth and lifecycle histograms, column anatomy, and corpus
bytes per embodied ALM byte.

`dendritic_basin_scaling_ladder` owns the multi-seed experiment. It runs the
three byte classes concurrently within each seed, resumes any existing atomic
checkpoint, skips only a report whose seed/train/eval identity validates, and
writes an aggregate summary after every completed seed. It can wait on earlier
report paths, allowing the remaining ladder to continue independently of an
interactive session.

## First byte-class evidence (100KB, seed 0)

| Ceiling | Embodied bytes | Columns used | Validation bpb | Births | Prunes | Train bytes/s |
|---|---:|---:|---:|---:|---:|---:|
| ALM-16MB | 2,853,898 | 7 / 7 | 3.9175 | 293 | 0 | 2,515 |
| ALM-64MB | 3,608,820 | 28 / 29 | 3.8023 | 286 | 2 | 1,564 |

At this short exposure, larger modular niche space improves prediction while
reducing destructive competition. ALM-64MB has 92 juvenile, 167 consolidated,
and 25 dormant branches. These are one-seed developmental observations, not
yet a scaling claim.

Sparse-scale profiling found that empty byte-class territory was still being
charged metabolically. Birth novelty previously sorted every possible trace
channel, and column competition performed many tiny Python sorts. Birth now
searches sensory plus live eligible relation channels only, while a compiled
stable tournament performs branch-to-column-to-soma selection. On the mature
ALM-64MB continuation this raises training from 2,293 to 2,610 bytes/second
without changing birth/prune counters.

## One-megabyte developmental gate (seed 0)

All three byte classes now complete a one-megabyte exposure and a fixed 100KB
held-out probe. This is a machinery and developmental pilot, not the promised
three-seed scaling result.

| Ceiling | Embodied bytes | Validation bpb | Births / prunes | Lifecycle J / C / D | Train bytes/s |
|---|---:|---:|---:|---:|---:|
| ALM-2MB | 1,995,725 | 3.8310 | 1,769 / 1,555 | 14 / 126 / 74 | 3,231 |
| ALM-16MB | 8,507,186 | 3.6998 | 1,098 / 30 | 152 / 395 / 521 | 1,484 |
| ALM-64MB | 8,449,575 | 3.7078 | 969 / 23 | 90 / 396 / 460 | 1,497 |

The 2MB organism reached its hard ceiling and performed 78 explicit nursery
evictions under byte pressure; its final checkpoint remains 4,275 bytes under
budget. The two larger classes have not yet differentiated by body size at
this exposure: both grew to roughly 8.5MB. Their near-tied one-seed scores are
therefore evidence that the 10MB developmental run is needed, not evidence of
a 64MB scaling advantage. Larger tissue already reduces destructive mature
turnover by roughly two orders of magnitude relative to ALM-2MB.

## Remaining gates

1. Move the event-routing frontier into contiguous compiled arrays, then
   compile the outer causal byte loop around the fixed column state; the local
   plasticity kernel is compiled, but Python still orchestrates each byte.
2. Run the 10MB x {ALM-2MB, ALM-16MB, ALM-64MB} x three-seed ladder with a
   fixed 1MB validation segment and periodic checkpoints.
3. Advance the winning stable body to exactly 100,000,000 consecutive raw
   training bytes (5.20% of the local 1,924,281,556-byte TinyStories file),
   still followed by the fixed untouched 1MB validation segment, only if
   held-out bpb continues improving, mature turnover
   per kilobyte falls with experience, and useful branch survival increases.
4. Run full TinyStories unchanged, then test the same organisms on unrelated
   byte domains.
