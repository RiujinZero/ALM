# Phase 14: Contextual Residual Columns

## Purpose

Phase 14 separates immediate sensory regularity from longer contextual
structure. A local basal circuit learns the next-byte distribution conditioned
on the current byte. Winning dendritic assemblies no longer average complete
predictions into a unigram distribution; they modulate the basal prediction
with bounded residual evidence in log-probability space.

This remains an embodied ALM. The sensory transition tissue, dendritic
predictor evidence, developmental nursery, traces, and lifecycle state all
count toward the exact serialized byte ceiling.

## Biological organization

The prediction path is divided into two local roles:

1. **Basal sensory drive:** a 256 by 256 transition circuit records immediate
   byte-to-byte experience. It is updated only after the transition is
   observed and therefore remains strictly causal.
2. **Apical contextual modulation:** active dendritic branches predict how the
   next-byte distribution differs under longer temporal and relational
   context. Branch utility is still the local number of bits saved relative
   to the basal prediction.

For basal distribution `B`, winning branch distributions `D_i`, and positive
competition weights `w_i`, fusion is a weighted geometric pool:

```text
log P = log B + sum_i alpha_i (log D_i - log B)
alpha_i = w_i / (1 + sum_j w_j)
```

The residual weights sum to less than one. Context can raise or suppress an
outcome relative to basal drive without allowing immature branches to erase
the sensory prediction. With no winner, the model prediction is exactly the
basal prediction.

## Causality, execution, and durability

The previous observed byte is transient causal state. At each step the learner
first scores the already-pending prediction, then updates the observed
transition, then forms the next prediction from the newly current byte.
Episode reset clears the previous byte so no transition crosses a document
boundary.

The scalar, batched, and compiled competition paths implement the same
residual equation. The compiled sequential byte loop updates sensory counts,
returns the previous-byte state across structural barriers, and preserves the
same birth and byte-budget boundaries as Phase 13. Checkpoints include the
transition tissue and previous-byte state. Older checkpoints default to the
Phase 13 unigram/arithmetic path and retain their original continuation
semantics.

The transition circuit occupies 524,288 bytes at a 256-byte vocabulary. Under
the ALM-2MB ceiling this reduces planned mature/nursery capacity from 214 to
155 branches. The 100KB residual organism serialized to 1,893,890 bytes.

## Evidence

A deterministic higher-order probe uses the repeating stream `1,0,2,0`. The
successor of byte `0` is ambiguous to an immediate transition circuit but is
resolved by earlier context. Over the final 1,000 observations:

| Predictor | Bits per byte |
|---|---:|
| Sensory transition baseline | 0.5025 |
| Contextual residual ALM | 0.0077 |

The real-data ablation uses the same first 1,000,000 TinyStories training bytes
and fixed first 1,000,000 validation bytes, seed 0, under ALM-2MB:

| System | Validation bpb | Train bytes/s | Embodied bytes |
|---|---:|---:|---:|
| Phase 13 unigram + arithmetic mixture | 3.859628 | 19,047 | 1,804,381 |
| Phase 14 sensory transition drive | 3.319963 | - | included below |
| Phase 14 contextual residual ALM | 3.308457 | 25,635 | 1,827,290 |
| Independent streaming bigram control | 3.319972 | - | external control |
| Independent streaming trigram control | 2.459936 | - | external control |

The residual organism saves 0.011507 bits per byte beyond its own sensory
column and 0.551171 bits per byte (14.28%) relative to the Phase 13 mixture.

The ALM-2MB residual pilot was then repeated for seeds 1 and 2 on the same
1MB/1MB split:

| Seed | Validation bpb | Gain beyond basal bpb | Embodied bytes |
|---:|---:|---:|---:|
| 0 | 3.308457 | 0.011507 | 1,827,290 |
| 1 | 3.304658 | 0.015306 | 1,840,290 |
| 2 | 3.282244 | 0.037719 | 1,827,271 |

Mean validation loss is 3.298453 bpb with sample standard deviation 0.014165.
Every seed improves on the identical 3.319963-bpb basal circuit; mean
contextual gain is 0.021510 bits per byte. This validates replicated residual
learning at one body class and exposure, not yet a body-size scaling law.

## Interface

`dendritic_basin_online_stream_smoke --contextual-residual` enables the Phase
14 organism. The scaling ladder accepts the same flag; use a distinct
`--run-prefix` so Phase 13 and Phase 14 checkpoints can never be mixed.

## Next gates

1. Repeat the 1MB ablation over three seeds and ALM-2MB/16MB/64MB.
2. Inspect which contexts earn positive residual utility and whether larger
   bodies create actual sparse inter-column bridges.
3. Run a 10MB Phase 14 ladder if the multi-seed pilot remains above its basal
   column.
4. Advance the winning stable body to the fixed 100MB gate, then consider full
   TinyStories exposure.
