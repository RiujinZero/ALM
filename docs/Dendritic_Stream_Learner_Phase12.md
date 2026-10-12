# Phase 12: Bounded Online Dendritic Stream Learner

## Status

Implemented and validated on 2026-09-05. The new learner is the first single
causal organism in this research line that combines sensory events,
multi-timescale eligibility traces, structural birth/death, recursive relation
assemblies, local prediction, competition, and separately earned state
retention. It does not retain or replay the input stream.

Implementation:
`src/manifoldflow/dendritic_basin/online_stream.py`

Raw-byte experiment:
`src/manifoldflow/experiments/dendritic_basin_online_stream_smoke.py`

## Why Phase 12 was necessary

Phases 0-11 validated individual mechanisms, but the Phase 8 TinyStories
experiment retained a full diary of trace snapshots and trained candidates by
sampling that diary repeatedly. Candidate count was bounded; memory and
learning were not genuinely online. Scaling that experiment would have scaled
the wrong organism.

Phase 12 imposes these invariants:

1. The probability assigned to observation `t` is fixed before observation
   `t` is visible.
2. No corpus replay or whole-stream diary exists.
3. Every dendritic slot has a fixed-size local evidence cache.
4. The number of relation slots and global winners is fixed at construction.
5. Surprise may recruit structure but cannot transport a gradient.
6. Every firing branch learns from its own local outcome, including branches
   that lose global competition.
7. Only locally useful branches influence collective prediction.
8. A relation may read only sensory channels or older relation assemblies, so
   the recursive graph is causal and acyclic.
9. Episode reset clears traces and retained state without erasing anatomy.
10. Checkpoint/resume includes anatomy, local evidence, transient state, and RNG
    state and is bit-exact.

The package remains pure NumPy. The AST-based no-autograd test still rejects
any import of Torch, JAX, TensorFlow, or Autograd under
`manifoldflow.dendritic_basin`.

## Nature as the ontology

The biological literature is used here as an architectural constraint, not as
a claim that this simplified system is a faithful cortical simulation.

- A slot is a branch-local compartment, not a layer-wide weight matrix.
  Nearby synapses can alter one another's plasticity threshold and bind related
  activity locally ([Harvey and Svoboda, 2007](https://doi.org/10.1038/nature06416)).
- Structural birth is coactivity-driven and structural survival is earned.
  Repeated learning produces clustered new dendritic spines, and clustered
  spines persist more reliably than isolated ones
  ([Fu et al., 2012](https://doi.org/10.1038/nature10844)).
- The nonlinear branch is computational, not just a passive sum. Human
  cortical dendrites exhibit nonlinear operations capable of separating
  inputs that defeat a simple monotonic point-neuron response
  ([Gidon et al., 2020](https://doi.org/10.1126/science.aax6239)).
- Global competition is attentional rather than omniscient. Jumping spiders
  combine wide-field motion-sensitive eyes with narrow high-acuity principal
  eyes, and whether the high-acuity system redirects depends on both the
  attended target and distractor
  ([Bruce et al., 2021](https://doi.org/10.1242/jeb.231035)). The analogous rule
  here is: all local branches may gather evidence, but only a bounded useful
  subset controls the collective prediction.

These observations motivated the implemented separation between local branch
learning and global winner selection. They also motivated structural turnover
instead of allocating one permanent parameter block per observed context.

## Online anatomy

For a categorical stream, sensory input is a one-hot event vector. The fixed
channel space also reserves one output channel for every possible relation
slot. At each observation:

1. Score the preceding prediction before learning from the new observation.
2. Deliver the outcome locally to every branch that fired.
3. If surprise is high and the birth refractory period has elapsed, recruit or
   recycle one branch onto an active current afferent and a homeostatically
   novel trace afferent.
4. Update decaying sensory and assembly traces.
5. Evaluate branches in birth order; older relations may become afferents for
   newer relations.
6. Let only the most useful active branches vote alongside the base predictor.
7. Save the resulting distribution as the immutable prediction for the next
   observation.

The trace accumulator follows the existing design, `q <- lambda*q + x`.
Branch inputs multiply `q` by `1-lambda`, converting different timescales to a
common bounded activity range without changing their causal information.

## A real failure: prediction is not persistence

The first integrated version made any mature, immediately predictive relation
eligible to persist. That was wrong. On the same 100KB train / 20KB frozen
validation slices:

| Architecture | Validation bits/byte |
|---|---:|
| Dynamic unigram base | 4.5702 |
| Flat relations, no persistence | 4.3776 |
| Recursive relations, no persistence | **4.3218** |
| Flat relations, naive persistence | 4.5077 |
| Recursive relations, naive persistence | 4.5296 |

Recursion helped; persistence contaminated it. The implementation had collapsed
the design's event/state distinction: "predictive at this instant" had silently
become "worth holding after the event ends."

The fix gives retained state an independent local predictor, evidence count,
and utility estimate. A branch can probe delayed outcomes locally while its raw
event is inactive, but retained activity cannot enter the shared feature space
or influence prediction until that delayed predictor independently clears its
utility and evidence gates. Immediate event utility no longer grants state.

## Current result

Configuration: the first 100,000 raw bytes of TinyStories train, first 20,000
raw bytes of TinyStories validation, all 256 byte channels, six trace
timescales, 64 live relation slots, at most eight global winners, recursion to
depth four, and frozen validation. No input byte or trace snapshot is replayed.

| Structural seed | Model bits/byte | Dynamic base | Improvement |
|---:|---:|---:|---:|
| 0 | 4.3077 | 4.5702 | 0.2626 |
| 1 | 4.3236 | 4.5702 | 0.2466 |
| 2 | 4.3455 | 4.5702 | 0.2247 |
| **Mean ± sample SD** | **4.3256 ± 0.0190** | **4.5702** | **0.2446 (5.35%)** |

The stronger static full-training unigram control is 4.4650 bits/byte, so the
three-seed mean still improves by 0.1394 bits/byte (3.12%). Conventional frozen
byte n-gram controls remain much stronger: bigram 3.3335 and trigram 2.5788
bits/byte. This is a positive online-learning result, not a competitiveness
claim.

At seed 0, only 2 of the 64 live branches earned state eligibility. The learner
created 5,976 branches and recycled 5,912 over the 100KB stream. Memory is
strictly bounded, but turnover is still high and is now the clearest biological
and computational inefficiency.

Measured CPU throughput was approximately 1,800 training bytes/second and
3,100 frozen-evaluation bytes/second at 64 slots. All 147 dendritic-basin tests
pass, including the new causality, boundedness, hierarchy, frozen-evaluation,
reset, checkpoint, and no-autograd tests.

## Developmental nursery follow-up

The first next-scale gate is now implemented. A fixed-size nursery holds
provisional coactivity patterns separately from mature relation anatomy. A
single surprising observation creates or updates a provisional candidate; it
does not create a mature branch. Promotion requires the same local afferent
pair to recur at least three times and its observed targets to clear a
consistency gate. Stale or weak provisional candidates are evicted locally.
The nursery, its target evidence, and its lifecycle are checkpointed, and both
nursery size and every candidate's statistics are bounded.

Matched 100KB/20KB results:

| Structural seed | Nursery model bits/byte | Mature births | Mature prunes |
|---:|---:|---:|---:|
| 0 | 4.2148 | 711 | 647 |
| 1 | 4.2554 | 605 | 541 |
| 2 | 4.2641 | 636 | 572 |
| **Mean ± sample SD** | **4.2448 ± 0.0263** | **650.7 ± 54.5** | **586.7 ± 54.5** |

The nursery improves the three-seed mean from 4.3256 to 4.2448 bits/byte. It
beats the static unigram by 0.2202 bits/byte (4.93%) and the dynamic unigram by
0.3254 bits/byte (7.12%). Mature births fall by approximately 89% versus the
pre-nursery three-seed mean. The outcome supports a concrete developmental
principle: repeated local evidence before consolidation is better than
creating mature anatomy from single surprising events.

At seed 0 the nursery received 10,774 proposals, promoted 711 branches, evicted
3,870 provisional candidates, and ended with 128 of its 256 slots occupied.
The nursery therefore reduces mature anatomical churn rather than hiding an
unbounded candidate store elsewhere.

## Byte clock, embodied size, and batched cortical strata

ALM size is reported as the serialized byte footprint of the complete durable
organism, not as an analogy to gradient-trained parameter count. Reports now
include `serialized_alm_bytes`. Experience is measured independently in raw
bytes observed: one call to `observe` is one byte of the stream.

The experiment harness now maintains cumulative log-loss and a fixed 20KB
tail window instead of retaining one model and baseline probability for every
byte. N-gram control scoring is also accumulated online. Live telemetry emits
processed/target bytes, percentage, bytes/second, ETA, rolling bits/byte,
mature/nursery occupancy, and births/prunes. Telemetry goes to stderr so the
final JSON report remains machine-readable on stdout.

Dense branch integration is batched by causal depth. Each branch can read only
sensory channels or a strictly lower-depth relation assembly, so all branches
at one depth form an independent cortical stratum. Their afferent gathering,
nonmonotonic response, persistence, state eligibility, feature publication,
and trace injection use dense NumPy arrays. Structural birth/prune decisions,
local running statistics, local sample caches, and perturb/select plasticity
remain explicit and gradient-free. A scalar response backend remains as a
trajectory oracle.

On the matched seed-0 100KB/20KB, 256-slot run, the accelerated path preserved
the exact 4.0661 validation bits/byte, 397 births, and 141 prunes. Training
throughput increased from 611 to 704 bytes/second (+15.1%), while frozen
evaluation increased from 1,907 to 4,131 bytes/second (+116.6%). A mature
10KB checkpoint profile reduced `_advance_features` from 4.57 to 1.05 seconds
(4.35x) and total frozen execution from 6.77 to 3.24 seconds (2.09x).

The first 1MB/100KB run at 256 slots completed before this batching change:

| Measure | Result |
|---|---:|
| Serialized organism | 1,855,106 bytes (ALM-1.86MB) |
| Validation model | 4.0361 bits/byte |
| Dynamic unigram | 4.5470 bits/byte |
| Static unigram | 4.4426 bits/byte |
| Bigram | 3.3188 bits/byte |
| Trigram | 2.4066 bits/byte |
| Mature births / prunes | 3,386 / 3,130 |
| Live mature / nursery | 256 / 99 |
| Training / frozen evaluation | 761 / 1,910 bytes/second |

Thus an **ALM-1.86MB developed on 1MB of experience** reaches 4.0361
bits/byte. It improves over the static unigram by 9.15%, but remains well
behind fixed bigram and trigram controls. Mature births per kilobyte decline
slightly relative to 100KB, while pruning remains active enough that branch
stabilization is still an open biological question.

## Next scale gate

Do not jump directly to 100M bytes. The next sequence is:

1. **Complete:** add an immature-branch nursery so repeated coactivity, rather
   than a single surprise, is required before a branch occupies consolidated
   anatomy. Mature births fell by approximately 89% while validation improved.
2. **Complete for CPU:** batch response, persistent-state integration, and
   trace updates by causal depth while keeping structural decisions and local
   learning explicit and gradient-free. Predictor competition remains the
   next dense kernel for a future accelerator backend.
3. **First curve complete:** run population-size curves (64/128/256/512 slots)
   against the fixed unigram,
   bigram, and trigram controls. Determine whether additional anatomy closes a
   meaningful portion of the n-gram gap. Seed 0 saturates between 256 (4.0661
   bpb) and 512 (4.0657 bpb); repeat across structural seeds before treating
   that carrying capacity as established.
4. **1MB complete:** run 10MB raw TinyStories only after predictor competition
   is batched and mature turnover is reduced further.
5. Run the unchanged organism on Wikipedia, source code, and conversational
   bytes. Architecture changes made only for one dataset fail the stated
   cross-domain criterion.

The current result answers the first two Phase 12 questions: a single causal,
bounded, replay-free dendritic organism can discover predictive raw-byte
structure, and repeated-evidence development can make its structural turnover
far sparser without sacrificing prediction. The next question is whether that
organism can be vectorized and scaled without turning into either a lookup
table or a conventional trained network.

## Phase 13 execution checkpoint

The first Phase 13 backend checkpoint converts predictor evidence to dense
branch matrices, batches distribution/evidence/winner competition, stores
local branch samples in one fixed anatomical ring tensor, and compiles the
repeated Welford interaction-objective calculation with an optional Numba
kernel. Scalar adaptation and competition paths remain executable semantic
oracles. Three 2,000-byte oracle trajectories confirm exactly identical
predictions, structural decisions, and population statistics across scalar
and accelerated response, competition, and adaptation paths.

On the unchanged seed-0 100KB/20KB, 256-slot experiment, training throughput
increased from 704 to 3,534 bytes/second (5.02x) and frozen evaluation reached
5,818 bytes/second. Validation remains exactly 4.0661 bits/byte with exactly
397 mature births and 141 prunes. The optional accelerator is isolated in
`compiled_kernels.py`, imports no autograd framework, and falls back to the
scalar local rule when Numba is unavailable.
