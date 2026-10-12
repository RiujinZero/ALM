# Phase 40: Receptor-Routed Apical Prediction

## Result

The mature 300MB-exposure motif cortex now evaluates the same frozen 1MB
TinyStories window at **58,259 bytes/s** and **1.155120872 bits per byte**.
The Phase 39 run originally evaluated at 36,330 bytes/s. Receptor routing is
therefore about **60% faster**, with no probability or embodied-anatomy change.

| Apical candidate path | Validation bytes/s | Validation bpb |
|---|---:|---:|
| Full-prefix scan | 36,330 | 1.155120872 |
| Incremental newest-receptor filter | 45,850 | 1.155120872 |
| 4K first-receptor routes | 50,682 | 1.155120872 |
| 16K routed and aggregated signals | **58,259** | **1.155120872** |

## Measured bottleneck

Profiling the original mature cortex over 100,000 validation bytes attributed
3.79 of 5.63 seconds to apical candidate selection. It performed 8.75 million
whole-prefix comparisons. The event cortex was repeatedly asking every
candidate motif whether it matched all receptors already observed.

The event anatomy is heavy-tailed. Its 571,872 branches contain 15,895,869
synapses, but only 44,203 branches hold 65.8% of those synapses. Permanently
nesting every branch by receptor would make the Python object graph much
larger merely to accelerate a minority of active routes.

## Incremental dendritic exclusion

Once a motif hypothesis matches an observed prefix, the next receptor only
needs to test the newly arrived position. Earlier positions are already
established. The active candidate population now contracts one receptor at a
time rather than repeating a whole-prefix comparison.

A regression oracle retains the old full-prefix scan and verifies that every
emitted probability is bit-identical on the same stream.

## Transient receptor routing

When an event branch first becomes active, candidates are gated by the first
receptor. Recurrent branch/receptor pairs reuse both:

- the surviving motif population; and
- its already-aggregated next-receptor counts and total evidence.

The cache is bounded at 16,384 routes. It is working state, not anatomy:

- it is excluded from `embodied_bytes`;
- it is cleared at episode reset, developmental motif completion, structural
  consolidation, and mature weak-branch pruning;
- `__getstate__` strips it from checkpoints, so evaluation history cannot
  inflate the saved ALM class.

At the end of the 1MB benchmark the cache held 874,960 candidate references.
Its serialized equivalent was 5.9MB, but none of those bytes enter the
151.39MB checkpoint.

## Rejected extension

Caching every deeper partial prefix reduced throughput to 43,437 bytes/s.
Most long prefixes are too specific to recur before eviction, so hashing and
storing them costs more than incremental filtering. Only the broad,
frequently reused first-receptor gate is retained.

## Validation

The full dendritic-basin test suite passes. Focused motif tests include exact
comparison with the legacy scanner, frozen checkpoint continuation, cache
exclusion from serialization, awake nursery bounds, weak-branch competition,
receptor relabeling, and the four non-language generality gates.

The next scaling run should retain thirty independent developmental
territories while scheduling fewer simultaneously, for example ten-megabyte
territories with 8--16 workers. Territory count preserves column diversity;
worker count only controls cache and memory-bandwidth contention.
