# Phase 48: Event-Specific Temporal Basis

Date: 2026-09-07

Scope: Dendritic ALM only

## Result

Exact event lags were replaced by a bounded working set that preserves motif
identity and projects each active motif through a logarithmic age-band
receptor. A route is addressed by `(age band, source motif)` and predicts a
future motif through the existing apical event pathway.

This is a local long-range mechanism rather than a language adapter. Motifs
still emerge from raw-byte prediction error. The temporal cortex sees motif
identity, elapsed motif events, its local route counts, and the observed next
motif. It receives no word, sentence, delimiter, entity, or story annotation.

## Why This Substrate

Phase 45 showed that one leaky scalar trace can preserve correlation across a
delay but loses the identity of the event that earned credit. An exact delay
line preserves identity only when the causal distance is stable. Biological
working memory suggests a third option: several event-specific traces coexist,
while dendritic temporal receptors encode coarse elapsed time.

The implementation keeps at most 64 distinct active motif identities. Their
ages are represented by upper-bound receptors at 1, 2, 4, 8, 16, 32, and 64
motif events. Repeated activation refreshes a trace; the oldest trace leaves
when capacity is full. Durable route anatomy and transient trace state are both
included in embodied bytes.

## Variable-Delay Generic Gate

A binary cue was followed by 3–48 randomly selected filler motifs. The cue
determined one byte in a later target motif. Filler count varied every episode,
so local context and exact-lag routes could not identify the cause reliably.

Across five seeds after 2,000 training and 500 frozen evaluation episodes:

| cortex | target bits/outcome |
|---|---:|
| local motif/event | 1.019833 |
| exact lags 1/2/4/8/16/32/64 | 0.981858 |
| event-specific temporal basis | **0.641946** |

The temporal learner used about 88.5KB, 70 route branches, and approximately
678 route synapses. Every seed beat both controls. This closes Phase 45's
event-identity failure on the variable-delay assay, although it does not yet
demonstrate narrative coherence.

## TinyStories Transfer Gate

At equal 100,000-byte training exposure and 100,000 frozen validation bytes:

| cortex | validation bpb |
|---|---:|
| local motif/event control | 1.852537 |
| exact-lag routed cortex | 1.855389 |
| event-specific temporal basis | **1.829263** |

The new generic mechanism improves TinyStories by 0.02327 bpb rather than
paying a language-modeling penalty. The temporal body was 2,235,246 exact
bytes under a loose 64MB ceiling and trained at 27,598 bytes/second. This is a
transfer gate, not yet a sample-quality result.

## Coherence Pilot

The detached 30MB run completed 30,000,000 training bytes at 10,900 B/s and
scored 1.228898 validation bpb on 1,000,000 held-out bytes. Final embodied
anatomy was 37,782,737 bytes. It reached roughly 284MB immediately before the
last budget competition, then compacted heavily; 248,550 of 248,552 mature
budget evictions came from route tissue. The result is highly sample-efficient
compression but also exposes excessive candidate-route formation.

Generation evaluation now also records distinct 8- and 16-grams plus the
longest exactly repeated block. This directly exposes phrase attractors that
can look superficially grammatical while cycling indefinitely.

Frozen generation did not close the coherence gap. A ten-seed temperature
0.8/top-k 32/top-p 0.9 sweep remained valid UTF-8 and printable, completed a
learned story boundary in 70% of samples, and averaged 0.6491 distinct
four-grams. Samples had recognizable syntax but lost entity and event
consistency. Greedy generation entered a repeated Lily/crayons/park cycle,
with a 0.143 zlib ratio and no story boundary over 1,000 bytes.

Phase 48 therefore preserves distant event identity and materially improves
small-exposure compression, but memory of the past is not sufficient to select
a coherent future. This result opens the Phase 49 goal-attractor gate.

## Acceptance Gate

Phase 48 advances only if the 30MB checkpoint satisfies all of:

1. validation bpb improves over the matched local cortex;
2. greedy and stochastic samples do not worsen exact phrase-loop behavior;
3. boundary completion, compression ratio, and long n-gram diversity improve
   on a fixed prompt/seed sweep;
4. qualitative samples preserve actors, objects, or causal state across more
   than the local six-byte sequence window;
5. a non-language variable-delay ablation retains the measured advantage.

If bpb improves without generation stability, event identity is useful for
compression but still insufficient for a coherent recurrent state.
