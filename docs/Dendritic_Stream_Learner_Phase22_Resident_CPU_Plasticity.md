# Phase 22: Resident CPU-Column Plasticity

## Local learning now stays inside the cortical territory

The CPU-column backend now performs the complete fixed-anatomy local learning
loop without returning to Python between bytes. Each contiguous territory owns
its predictor rows, event/state credit, utility, observations and firings,
local sample rings, Welford interaction evidence, dendritic perturb/select
adaptation, persistence, and lazy temporal traces.

The implementation remains causal across arbitrary input chunks. A pending
prediction formed at the end of one chunk receives its outcome at the start of
the next. Tests compare one- and two-worker resident execution with the event
oracle across a split 512-byte stream. Predictions agree within `3e-7`; counts
are exact; float32 trace, sample, and evidence state remains within its stated
numeric tolerance. A materialization boundary analytically advances silent
traces and their homeostatic means before checkpoint transfer.

## Nature-shaped compaction

The first plastic resident implementation scanned all 112,341 predictor rows
to resolve the sparse set of local outcomes. It also rescanned the body for
column competition. That ignored the anatomical result of Phase 21.

Each territory now publishes its locally active predictor boutons into a
private compact list. The same worker resolves those rows on the following
byte and inserts candidates directly into its column-local winner set. Only
the small column scores and winners reach the somatic reduction. There is no
globally contended queue and no full-body outcome scan.

Literal zero relation activity is also represented as silence rather than six
trace writes. Its elapsed decay and mean update are folded into the next axonal
event or checkpoint boundary.

## ALM-1GB measurement on the Ryzen 9 9950X

The learned 155-branch Phase 14 organism was tiled into 112,341 independent
branch slots, the conservative ALM-1GB geometry, and fed real TinyStories
bytes. Developmental birth/pruning was held fixed so the same filled anatomy
could be compared across worker counts.

An 8,192-byte, three-repeat run before perturb/select adaptation measured:

| Workers | Median bytes/s | Best bytes/s |
|---:|---:|---:|
| 16 | 1,028 | 1,028 |
| 30 | 1,314 | 1,331 |

Unlike frozen inference, plastic predictor-row traffic benefits from SMT. The
30-thread local-bouton path is 21.5% faster than the preceding resident
evidence implementation's 1,110 B/s short measurement.

With branch-local counter-based perturb/select enabled, two 4,096-byte runs at
30 threads measured 1,262 and 1,215 B/s, with a midpoint of 1,238 B/s. At that
rate, one pass over the local 1,924,281,556-byte TinyStories training file is
about 18.0 days. The RTX 5080 fixed-anatomy learner remains faster at roughly
3,023 B/s, or 7.4 days.

These are filled-body stress tests, not a claim that a naturally developing
organism pays ALM-1GB cost from its first byte. A growing body starts sparse;
the proposed 10-100MB pilot must record occupancy-weighted throughput over
development.

## Developmental handoff

Resident state can now be synchronized back into the ordinary developmental
organism. The handoff includes basal/context transition tissue, predictor
counts, event/state utility, local evidence, dendritic parameters, lazy traces,
persistence, previous-byte state, and pending local outcome credit. A test
continues inference through this boundary and matches the uninterrupted event
oracle.

The structural membrane is now implemented. Ordinary bytes remain inside
resident territories. Before a surprise-eligible byte is consumed, the loop
materializes and hands off its causal state, the developmental organism
performs nursery/birth/pruning, and the resident columns resume against the
rebuilt axonal index. A 1,500-byte developmental test matches the uninterrupted
event oracle's probabilities, births, prunes, and final live population.

The bounded TinyStories runner exposes this path with
`--cpu-column-workers`. A from-scratch ALM-64MB 20KB/5KB smoke used all 28
anatomical territories, developed 127 branches without pruning, and serialized
to 2.68MB. Held-out loss was 3.4618 bpb versus its own 3.4676-bpb basal circuit
and a 3.4654-bpb streaming bigram. Training averaged 289 B/s because 1,741
early surprise proposals repeatedly woke Python development; frozen evaluation
ran at 15.6KB/s. This validates the membrane but also confirms CPU is not the
pilot fast path during the nursery-rich early phase.

## Gate to a full TinyStories run

The implementation gate is closed. One empirical gate remains:

1. Run a naturally growing 10-100MB TinyStories pilot, measuring held-out
   bits/byte, embodied ALM bytes, occupancy, turnover, and effective bytes/s.
2. Start the full corpus only if contextual gain survives and the measured ETA
   is operationally acceptable. GPU is the default pilot and ALM-1GB path;
   resident CPU columns remain the oracle/fallback and a many-core topology
   experiment.
