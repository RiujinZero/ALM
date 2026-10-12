# Phase 38: Prediction-Error-Segmented Motif Cortex

## Why the word organ was rejected

Phase 37 established that a slower event layer can improve TinyStories at
small scale, but it defined its assemblies through ASCII letter boundaries.
That is a language sensory adapter, not a general ALM primitive. Worse, its
uncapped apical counts overwhelmed the lower cortex at full scale: first-window
validation regressed from the sequence cortex's 1.230665 to 1.335034 bits per
byte (bpb), while its body grew to 257.78MB.

The useful requirements were retained—slow assemblies, cross-column alignment,
transient hypotheses, and top-down residual prediction—while both privileged
word boundaries and unbounded apical control were removed.

## Biological ontology

`PredictiveMotifOpponentLearner` closes a motif when the lower sequence cortex
encounters a sufficiently large prediction error after at least three receptor
events. A maximum of sixteen events prevents an indefinitely stable state from
occupying unbounded working memory. No byte value, category, delimiter, reset
channel, or modality label has special meaning.

The structure follows four experimentally motivated constraints:

1. Cortical dendrites can discriminate the temporal order of synaptic input
   through nonlinear local mechanisms (Branco, Clark, and Hausser, 2010,
   <https://doi.org/10.1126/science.1189664>).
2. Plateau-associated plasticity can bind inputs over behavioral rather than
   millisecond timescales (Bittner et al., 2017,
   <https://doi.org/10.1126/science.aan3846>).
3. Hippocampal activity marks event boundaries in continuous experience, and
   prediction-error spikes are a candidate boundary signal (Ben-Yakov and
   Henson, 2018, <https://pmc.ncbi.nlm.nih.gov/articles/PMC6246887/>).
4. Hippocampal event codes can transfer across changes in spatial geometry
   (Sun et al., 2020, <https://doi.org/10.1038/s41593-020-0614-x>).

The resulting causal loop is:

```text
raw receptor events
  -> depth-six sequence dendrites
  -> local prediction error
  -> close recurrent motif / begin another
  -> sparse motif-event transition
  -> bounded apical residual into the lower cortex
```

The same motif and event plasticity receives every tested stream. Only the
sensory adapter maps observations into receptor identities.

## Receptor-identity invariance

A non-language stream contains a binary cue, eight identical filler events, a
common onset, and a cue-dependent target. The cue is outside the lower cortex's
six-event reach. Predictive motifs reduce target loss from approximately one
bit to 0.619 bit.

Every receptor identity was then replaced by an arbitrary unrelated byte ID.
Target loss remained bit-identical at **0.6189077229** in both encodings. The
organ responds to recurrence and causal ordering, not symbol value or textual
shape.

## Cross-domain frozen-evaluation gates

Each task trains and evaluates the same learner configuration. The target
metric is log-loss reduction over the depth-six sequence cortex at the delayed
ambiguous event.

| Domain | Sequence target bits | Motif target bits | Gain |
|---|---:|---:|---:|
| Arbitrary delayed symbols | 1.000000 | 0.486166 | **0.513834** |
| Occluded ring trajectory | 3.000754 | 2.647758 | **0.352996** |
| Time-warped audio motif | 1.001429 | 0.825603 | **0.175826** |
| Spider-style cross-modal cue/action | 1.000589 | 0.793873 | **0.206715** |

The ring task varies both starting position and occlusion duration. The audio
task varies motif duration. The cross-modal task presents a visual cue followed
by a long distractor interval, common vibration/action onset, and cue-dependent
action outcome. No task supplies a boundary marker to the learner.

These remain synthetic isolation tests, not evidence that the current byte
adapter fully represents real pixels, waveforms, or motor populations. They do
establish that the learned structure itself is not language-specific.

## Inhibitory apical bound

The Phase 37 failure identified a general scaling defect. Its event formula
treated lower-cortex prediction as a fixed pseudocount; as event observations
grew, the lower prior asymptotically vanished. The corrected soma caps slow
apical influence at 50%, regardless of accumulated count. Event evidence may
modulate but never replace current sensory cortex.

Post-hoc frozen evaluation of the completed Phase 37 checkpoint located a
broad bounded optimum:

| Maximum apical contribution | First-window bpb |
|---:|---:|
| 5% | 1.212297 |
| 10% | 1.202016 |
| 20% | 1.189381 |
| 30% | 1.182547 |
| **50%** | **1.180702** |
| 75% | 1.201124 |
| Unbounded | 1.335034 |

The 50% bound was then applied unchanged to all non-language gates. Every gain
survived, so this is treated as a general inhibitory constraint rather than a
language correction.

## TinyStories transfer without text boundaries

| Exposure | Sequence-only bpb | Generic motif bpb | Reported motif body |
|---:|---:|---:|---:|
| 1MB / 100KB valid | 1.421763 | **1.401609** | 9,174,905 B before sleep |
| 30MB / 1MB valid | 1.278238 | **1.226226** | 58,983,428 B |
| 100MB / 1MB valid | 1.251180 | **1.178840** | 146,735,444 B |

At 1MB the generic organ slightly exceeds the selected ASCII-word result
(1.402701 bpb). At 30MB it exceeds the word organ's 1.252502 bpb by 0.026275.
At 100MB it exceeds the word organ's 1.239107 by 0.060267 and continues
improving with exposure instead of collapsing.

The 100MB run develops at 517,664 bytes/s through sixteen two-megabyte
territories; consolidation takes 63.33 seconds and frozen evaluation runs at
65,058 bytes/s. Developmental sleep sheds 500,392 sequence candidates and
5,027,811 unconfirmed motif-event candidates without changing prediction.
Mature anatomy contains 697,616 motifs, 1,706,440 event branches, and
12,255,082 event synapses.

## Honest next constraint

The generic cortex is accurate but anatomically exuberant: its 146.7MB mature
body is larger than its 100MB exposure. A full-corpus extrapolation may approach
the intended ALM-1GB class, but the Python object graph could exceed available
developmental memory well before the serialized body does.

The next gate is therefore structural consolidation, not a full launch:

- share recurrent motif prefixes instead of storing whole forms repeatedly;
- align motifs that preserve transition behavior under duration variation;
- retain the 50% apical cap and measure all four non-language gates after every
  compaction;
- advance through a 300MB memory-envelope run before committing to the full
  corpus.

Twenty-five focused tests pass, including exact streamed/full prediction,
checkpoint continuation, cross-column receptor-form remapping, developmental
sleep, non-language delayed ambiguity, receptor relabeling, and the four-domain
generalization suite.
