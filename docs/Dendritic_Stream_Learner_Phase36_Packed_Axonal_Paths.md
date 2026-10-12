# Phase 36: Packed Axonal Paths and the 100MB Scale Gate

## Lossless anatomical compaction

Phase 35 represented every depth-three-through-six dendritic context as a
Python tuple of boxed byte identifiers. The representation was faithful but
anatomically wasteful: the tuple container and each repeated integer were part
of the durable body even though a context contains at most six eight-bit
receptor identities.

The sequence cortex now packs each complete byte path into one integer axonal
address. Depth remains implicit in its laminar bank, so the mapping is
lossless. For example, the most recent byte occupies the low eight bits and
each older receptor identity occupies the next eight-bit field. Tuple paths
remain available as an oracle mode rather than being deleted.

An exact developmental-trajectory test feeds the packed and tuple organisms
the same ambiguous nonlinear stream. Every emitted observed probability is
bit-identical, and both organisms develop the same number of mature assemblies
and target synapses. Column consolidation has a separate structural test.

## Compaction result

| Exposure and evaluation | Tuple paths | Packed paths | Reduction |
|---|---:|---:|---:|
| 1MB train / 100KB valid body | 6,564,017 B | 5,265,189 B | 19.79% |
| 30MB train / 1MB valid body | 26,589,591 B | 22,184,376 B | 16.57% |
| 30MB held-out loss | 1.278238 bpb | 1.278238 bpb | exact |
| 30MB consolidation | 6.909 s | 5.944 s | 13.97% faster |

Training and evaluation throughput varied by roughly five percent between
runs under machine contention, so no execution-speed claim is made beyond the
repeatable reduction in merge work. The primary result is smaller durable
anatomy with identical behavior.

Embedding totals into every local branch was also evaluated and rejected.
Although it would remove a top-level dictionary, the additional entry in each
small mature and nursery row increased serialized anatomy. Compaction must be
measured at the whole-organism boundary, not assumed from object count.

## Bounded 100MB TinyStories probe

The packed phenotype developed on 100,000,000 raw TinyStories bytes in fifty
2MB territories through 16 persistent CPU workers. Each completed territory
was consolidated immediately and released, retaining the bounded developmental
memory behavior established in Phase 34.

| Measurement | Result |
|---|---:|
| Training bytes | 100,000,000 |
| Frozen validation bytes | 1,000,000 |
| Training loss | 1.546061 bpb |
| **Held-out loss** | **1.251180 bpb** |
| Development throughput | 898,101 B/s |
| Consolidation time | 21.946 s |
| Frozen evaluation throughput | 210,118 B/s |
| **Embodied body** | **34,256,705 B** |
| Sequence assemblies | 749,854 |
| Sequence target synapses | 2,697,274 |
| Nursery paths | 500,392 |

Relative to the packed 30MB cortex, 3.33 times more experience grows the body
only 1.54 times, from 22.18MB to 34.26MB. Experience density rises from 1.35
to 2.92 training bytes per embodied byte. Recurrent local language paths are
being reused rather than causing linear memorization.

Held-out loss improves from 1.278238 to 1.251180 bpb. This is a real gain over
ten times the Phase 35 validation window, but the diminishing improvement also
locates the next scientific boundary: depth-six local sequence anatomy is
approaching saturation. Full-corpus development is justified to measure the
asymptote and mature body size; the next large accuracy gain will probably
require selective phrase/event timescales rather than indiscriminately deeper
paths.

## Next execution

Launch the full TinyStories training file with the same 16-worker, 2MB bounded
territories, a 1MB frozen validation window, and an atomic final checkpoint.
Keep the episodic organ disabled: Phase 35 showed that its current content
index adds size and latency without measurable predictive gain above the
sequence cortex.
