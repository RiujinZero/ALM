# Phase 32: Episodic Content Index

## From distance to reinstatement

The first content-addressable memory organ sits above the raw-byte opponent
cortex. Recurrent byte paths separated by non-content receptor events earn a
content assembly after three confirmations. There is no predefined vocabulary.
Assemblies are lowercase-stabilized sensory paths discovered in the stream.

Within a story, completed content assemblies compete for one persistent
working-memory position. Activation reflects path length and habituates with
global recurrence, favoring distinctive content over ubiquitous short forms.
The active assembly can grow sparse top-down synapses into the current local
byte assembly when the local predictor is repeatedly surprised. A branch is
allowed to influence prediction only after its measured residual utility
exceeds 0.05 bits. The explicit TinyStories end marker clears working state.

This is a minimal hippocampal-index analogue: content remains represented in
the local cortical organ, while a small persistent address can reinstate a
context-specific output tendency from partial current input.

## Controlled behavior

In alternating synthetic stories, the same local byte context leads to one of
two continuations depending on the remembered content assembly. The local
organ assigns roughly 50% to the correct continuation. The episodic organ
raises it to approximately 98.3%, establishing causal use of persistent
content rather than a larger local n-gram.

## TinyStories measurements

| Exposure | Local bpb | Episodic bpb | Gain | Episodic body | Train rate |
|---|---:|---:|---:|---:|---:|
| 100KB / 20KB eval | 2.459222 | 2.448832 | 0.010390 | 464,545 B | 76,479 B/s |
| 1MB / 100KB eval | 2.390727 | 2.354969 | 0.035758 | 2,826,148 B | 69,100 B/s |

After 1MB, the organ contains 3,077 content assemblies, 36,964 episodic
branches, and 270,639 episodic output synapses. Only 391 branches pass the
utility gate during held-out evaluation. Capacity screens favor one dominant
persistent assembly: capacities of two and four grow more tissue and perform
worse at 100KB, indicating interference rather than useful parallel recall.

## Interpretation and next constraint

Content-addressed reinstatement succeeds where fixed delays failed, but its
nursery currently records too many branches. The next step is metabolic:
consolidate and prune on utility before scaling exposure, then make content IDs
mergeable across cortical training territories. The correct identity for
cross-column consolidation is the discovered byte path itself, not each
column's local integer ID.

## First consolidation result

After 1MB, pruning branches with at least eight observations whose measured
utility cannot cross the 0.05-bit recall gate removes 23,064 of 36,964
episodic branches. The body falls from approximately 2.824MB to 1.207MB while
held-out loss remains exactly 2.354969 bpb. Frozen evaluation improves slightly
from about 151KB/s to 155KB/s, though that small timing difference needs a
larger benchmark.

This is genuine metabolic consolidation: 62.4% of mature episodic branches and
57.3% of total serialized tissue disappear with no behavioral change. Nursery
candidates and young branches remain able to mature with further experience.
