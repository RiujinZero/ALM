# Phase 25: Opponent-Receptor Language Probe

## Raw bytes as simultaneous sensory vectors

The categorical byte channel is no longer the only tested language interface.
The opponent probe transduces each physical byte bit into an ON/OFF receptor
pair. Every byte is therefore a lossless 16-dimensional vector with exactly
eight simultaneous active receptors. There is no tokenizer, embedding table,
ordinal byte distance, or future context in the adapter.

This is a deliberately biological decomposition: receptors expose features,
coincidence dendrites bind features, recursive assemblies form identities, and
an output attractor preserves correlations when predicting the next sensory
pattern.

## What each cortical stage contributes

Four increasingly structured local circuits were measured:

1. Independent active receptors pool their next-bit evidence.
2. All 112 cross-bit receptor-pair dendrites pool their evidence.
3. A recursive coincidence tree binds two-bit, four-bit, and eight-bit input
   assemblies, while output receptors remain independent.
4. A sparse output basin predicts whole emerged assemblies and projects them
   back onto the receptor population.

The first three deliberately lack a categorical escape hatch. Their failures
localize which nonlinear operation is necessary rather than hiding it in a
learned embedding.

## TinyStories measurements

| Circuit | 20KB train / 5KB eval | 100KB train / 20KB eval |
|---|---:|---:|
| Independent opponent receptors | 5.5463 bpb | 5.5470 bpb |
| Pair coincidence dendrites | 5.4738 bpb | 5.4793 bpb |
| Recursive input assembly, independent output | 4.7377 bpb | 4.7348 bpb |
| Recursive input + output attractor | **3.4676 bpb** | **3.3352 bpb** |
| Categorical unigram control | 4.4805 bpb | 4.4650 bpb |
| Categorical bigram control | 3.4654 bpb | 3.3335 bpb |
| Categorical trigram control | 3.0246 bpb | 2.5788 bpb |

Independent receptors cannot bind the whole byte pattern; even pair dendrites
recover only a small fraction of the missing information. Recursive input
binding helps substantially, but independent output bits still discard
correlations among predicted features. Closing the recurrent loop with an
assembly-level output attractor recovers the categorical bigram result to
within 0.002 bpb at both scales.

At 100KB, 70 byte assemblies have appeared. The sparse attractor prototype's
serialized body is 24,848 bytes, including the fixed sensory codebook, versus
524,288 bytes for the current ALM's dense 256x256 float64 contextual tissue.
The comparison is not yet a complete ALM memory comparison, but it demonstrates
that the natural representation need not pay for nonexistent transitions.

## Honest limitation and next step

The current recursive tree uses a fixed balanced grouping of physical bit
pairs. It does not receive byte labels, but its complete hierarchy guarantees
that the top assembly is a lossless identity; development has not yet earned
that anatomy. The output attractor is therefore a mechanistic proof, not the
final nature-shaped learner.

The next implementation should replace the fixed tree with nursery-born
assemblies selected by repeated receptor coactivity and predictive survival.
The online ALM should expose both receptor and emerged-assembly channels to
contextual residual dendrites, while the sparse attractor replaces the dense
basal transition matrix. The existing one-hot GPU run remains the control for
loss, sample efficiency, throughput, and embodied bytes.

Phase 26 implements the nursery-grown hierarchy. At 100KB it reaches the
categorical bigram within 0.000156 bpb while training at 97,255 bytes/s in a
25,625-byte serialized body.
