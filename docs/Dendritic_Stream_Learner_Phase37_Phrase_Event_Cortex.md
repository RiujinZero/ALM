# Phase 37: Phrase and Event Cortex

## The residual exposed by generation

The full depth-six sequence cortex reached 1.249927 bits per byte across the
entire 19.45MB TinyStories validation corpus, but free generation exposed a
specific remaining failure. It produced well-formed words, punctuation,
dialogue, and local TinyStories phrasing while losing actors, causal relations,
and scene continuity. More local byte depth was already known to fragment at
small exposure. The next organ therefore operates on a slower discovered
timescale rather than extending another fixed delay.

`PhraseEventOpponentLearner` discovers word assemblies directly between ASCII
letter boundaries; it does not receive an external tokenizer or vocabulary.
The previous two normalized word assemblies form an event context. Each event
dendrite predicts a population of possible raw word forms. Once the next word's
first letter arrives, matching hypotheses return an apical next-byte signal to
the depth-six cortex. A divergent prefix immediately silences an incorrect
hypothesis, and the lower byte hierarchy remains the prior through a finite
apical strength.

A synthetic causal gate holds the most recent six bytes identical while an
earlier actor word determines whether `ap...` continues as `apple` or
`apricot`. The sequence-only cortex remains ambiguous; the event cortex solves
the selected bytes below 0.1 bit. This proves the upper pathway carries
information unavailable to the local organ rather than merely duplicating it.

## Developmental selection at 1MB

The initial two-word organ improved the 100KB held-out window from 1.421763 to
1.409899 bpb. Structural screens then selected the following phenotype:

- two prior word assemblies, because one lacked actor context and three
  fragmented evidence;
- two repeated observations before event-branch birth;
- apical strength 8, a broad optimum between overcommitment and abstention;
- activation after one matching letter, which provides more useful completion
  time than waiting for two or more letters.

| Configuration | Held-out bpb | Body |
|---|---:|---:|
| Sequence-only | 1.421763 | 5,265,189 B |
| Initial phrase/event | 1.409899 | 7,969,947 B |
| Selected phrase/event | **1.402701** | 8,028,339 B |
| Selected + cortical lexicon | **1.402701** | **6,991,796 B** |

The cortical lexicon assigns each discovered raw or normalized word form one
assembly ID. Event paths and targets store packed IDs, and column consolidation
remaps local IDs through the underlying receptor form. This removes repeated
word byte strings and cuts the upper organ's incremental anatomy by about
37.5% without changing a probability.

## Columnar evidence

Thirty independent 1MB territories align word assemblies and event dendrites
during consolidation. A final developmental-sleep pass sheds nursery paths
that remain globally unconfirmed. Nursery tissue never participates in
prediction, so this changes no validation probability.

| 30MB / 1MB frozen validation | Sequence | Phrase/event |
|---|---:|---:|
| Held-out bpb | 1.278238 | **1.252502** |
| Mature body after sleep | 16,768,271 B | 30,746,874 B |
| Training throughput | 1,299,658 B/s | 978,338 B/s |
| Evaluation throughput | 206,686 B/s | 103,351 B/s |
| Mature event branches | - | 211,384 |
| Event target synapses | - | 2,027,730 |
| Discovered word forms | - | 15,632 |

The event gain grows to 0.025737 bpb at 30MB. Sleep sheds 357,804 sequence
candidates and 234,209 event candidates, reducing the phrase/event body from
39.81MB to 30.75MB exactly.

At the bounded 100MB gate, the selected organ reaches **1.239107 bpb** on 1MB
frozen validation, versus 1.251180 for sequence-only. Its mature body is
58,237,132 bytes after shedding 928,701 combined nursery paths. It contains
23,076 word forms, 432,577 event branches, and 4,740,669 event synapses. The
gain remains real but narrows to 0.012073 bpb, indicating that this simple
two-word transition organ also begins to saturate.

## Rejected metabolic pruning

Global recurrence is not a safe utility proxy for mature event tissue. On the
30MB checkpoint:

| Minimum branch observations | Branches removed | Body | Held-out bpb |
|---:|---:|---:|---:|
| 2 (selected) | 0 | 30,826,157 B | **1.252502** |
| 3 | 63,912 | 28,962,640 B | 1.254525 |
| 4 | 94,896 | 27,962,018 B | 1.256261 |
| 8 | 142,908 | 26,133,366 B | 1.262931 |

Rare mature event branches carry useful residual information. They are kept;
only silent nursery candidates are shed.

## Transient hypothesis competition

The first implementation rescanned every target word in the active event row
for every character. The accepted implementation creates a tiny transient
candidate population when the first letter arrives and narrows it as the word
unfolds. This state is checkpointed for exact mid-word resume but adds no
durable synapses.

At 1MB it improves training from 65.5 to 76.9KB/s and evaluation from 105 to
152KB/s with identical loss. At 30MB it improves training from 844 to 978KB/s
and evaluation from 60.1 to 103.4KB/s. Twenty-two focused opponent-stream
tests pass, including exact streamed/full-broadcast behavior, actor-conditioned
ambiguity, packed-path parity, word-ID remapping across columns, lossless
developmental sleep, and exact mid-word checkpoint/resume.

## Next gate

Run the full TinyStories corpus with bounded 2MB territories and 16 workers,
then evaluate the complete validation corpus and compare generation against the
sequence-only ALM-88MB checkpoint. The organ is accepted as a useful residual
layer, but its full-corpus body and qualitative identity coherence determine
whether the next step is utility-aware event consolidation or a richer
persistent actor/object workspace.
