# Phase 35: Recursive Sequence Cortex

## Why the episodic organ was not enough

The compact episodic cortex improved the earlier two-byte local organ from
2.356158 to 2.313817 held-out bits per byte (bpb) at 30MB, but remained far
from the mid-1 bpb range. Fixed delayed fibers had already shown that temporal
distance alone was not coherence. The missing structure was a nested temporal
cortex: short paths should remain the prior for longer paths instead of every
long path becoming an isolated lookup.

`SequenceCortexOpponentLearner` grows sparse assemblies for recurring raw-byte
histories of depth three through six. A prospective path first lives in a
developmental nursery and becomes anatomy after three observations. Each
living child predicts through its immediate parent distribution:

```text
P(next | child) = (child evidence + 16 * P(next | parent))
                  / (child observations + 16)
```

The inherited basal and contextual organs provide depths one and two. Thus a
rare six-byte path does not make a brittle independent claim; it expresses a
residual over the five-byte path, which is itself a residual over the next
shallower context. This is a laminar ontology rather than a flat n-gram table:
specific distal context modulates progressively older, better-supported local
tissue.

## Depth selection on 1MB TinyStories

All rows used the same first 1,000,000 training bytes and 100,000 frozen
validation bytes. The local control has only the inherited one- and two-byte
organs.

| Maximum context | Held-out bpb | Embodied bytes |
|---:|---:|---:|
| local control | 2.390727 | 69,099 |
| 3 | 1.830927 | 281,935 |
| 4 | 1.594904 | 1,028,572 |
| 5 | 1.506711 | 2,921,895 |
| **6** | **1.482517** | **6,564,017** |
| 7 | 1.497796 | 12,460,000 |
| 8 | 1.529367 | 21,130,000 |

Depth six is the natural stopping point at this exposure. Deeper tissue grows
faster than evidence can support it and validation worsens. This is useful
structural feedback rather than a fixed architectural belief.

## Parent-strength selection

At depth six, increasing parent support prevented young distal assemblies from
overriding mature proximal tissue. A strength of 16 was best on the same
frozen split.

| Parent strength | Held-out bpb |
|---:|---:|
| 1 | 1.604417 |
| 2 | 1.538067 |
| 4 | 1.482517 |
| 8 | 1.442389 |
| **16** | **1.421763** |
| 32 | 1.423532 |
| 64 | 1.449440 |
| 128 | 1.500357 |

For reference, flat controls on this split scored 1.9242 bpb at order four,
1.8847 at order five, and degraded to 2.1008 at order six. Recursive backoff,
not merely longer memorization, accounts for the large gain.

## Thirty-column scale check

Thirty independent 1MB cortical territories were developed concurrently and
then consolidated by raw receptor path. This tests whether the assemblies
represent shareable anatomy rather than territory-local identifiers.

| Measurement | Sequence cortex | Sequence + episodic overlay |
|---|---:|---:|
| Training bytes | 30,000,000 | 30,000,000 |
| First held-out bytes | 100,000 | 100,000 |
| First-window held-out bpb | **1.180471** | 1.180507 |
| Extended held-out bytes | 1,000,000 | not run |
| Extended held-out bpb | **1.278238** | not run |
| Training throughput | 1,370,531 B/s | 884,942 B/s |
| Evaluation throughput | 210,802 B/s | 163,806 B/s |
| Consolidation | 6.902 s | 9.295 s |
| Embodied body | 26,589,591 B | 29,502,406 B |
| Living sequence assemblies | 485,739 | 485,739 |
| Living sequence synapses | 1,643,160 | 1,643,160 |
| Sequence nursery paths | 357,804 | 357,804 |

The first 100KB window was unusually easy, so it is not the headline result.
The 1MB frozen evaluation confirms **1.278238 bpb** across ten times as much
held-out text. The 30MB cortex therefore passes below the requested mid-1
range without relying on a small favorable validation slice. It develops
about 1.13 bytes of experience per embodied byte. The body is about 25.36MiB,
so the current experiment is still an ALM-27MB by serialized decimal-byte
class, not an ALM-1GB.

The episodic overlay retained only five utility-positive branches after global
metabolic pruning. Its 0.000036 bpb regression is within practical neutrality,
but it costs 34.5% training throughput and 2.9MB of body. Episodic anatomy
therefore remains available as an experimental organ but is not part of the
current best phenotype.

## Engineering and correctness

The scalar stimulated-basin path and full 256-byte broadcast remain equivalent.
Sequence history survives serialization and resets causally. Consolidation
checks configuration compatibility, merges both mature and nursery paths,
promotes nursery evidence when cross-territory confirmation is sufficient,
recomputes totals, and clears transient activity. Seventeen focused opponent-
stream tests pass and the changed files are ruff-clean.

## Next gate

The immediate question is no longer whether raw opponent receptors can reach
useful TinyStories loss. They can. The next experiment is a bounded full-corpus
sequence-cortex development. Body compaction should follow without weakening
hierarchical evidence: intern repeated context prefixes and pack byte
targets/counts into contiguous arrays rather than pruning predictive tissue.
Phrase- and event-scale organs should only be added after full-corpus residual
analysis shows which errors remain genuinely nonlocal.
