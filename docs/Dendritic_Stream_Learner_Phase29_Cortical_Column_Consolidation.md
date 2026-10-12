# Phase 29: Cortical Column Consolidation

## Territory before consensus

Sparse contextual organs can now learn disjoint byte territories in separate
processes and consolidate afterward. Each column develops its own opponent
assemblies, basal transitions, surprise-gated contextual branches, and nursery.
Consolidation adds observed synaptic counts, unions living anatomy, combines
nursery evidence, and promotes candidates whose pooled confirmations cross the
birth threshold. No dense Cartesian table is introduced.

The split discards at most two causal transitions per territory boundary.
Online training loss is the average of locally developing columns; held-out
loss is measured after anatomy has consolidated into one organ.

## Ryzen 9 9950X measurements

| Layout | Exposure | Throughput | Held-out bpb | Body | Consolidation |
|---|---:|---:|---:|---:|---:|
| 1 column | 10MB | 111,097 B/s | 2.357226 | 95,203 B | n/a |
| 4 columns | 4 x 1MB | 348,974 B/s | 2.366264 | 84,400 B | 8.6 ms |
| 16 columns | 16 x 2MB | 1,175,058 B/s | 2.355959 | 110,795 B | 36.5 ms |
| 30 columns | 30 x 1MB | 1,206,760 B/s | 2.356158 | 109,511 B | 70.3 ms |

Thirty columns are 10.9x faster than the checkpointed single-column 10MB run.
Sixteen columns are within 3% of that rate, showing that the 16 physical cores
are nearly saturated and the extra SMT territories provide a small gain. The
merged 30-column organ contains 1,266 contextual branches and 12,157 contextual
output synapses.

At the measured 30-column rate, 1,924,281,556 TinyStories bytes correspond to
26.6 minutes of pure training. This is a projection, not yet a completed full
pass; longer territories, Google Drive reads, per-column checkpoints, final
consolidation, and validation will determine the actual wall time.

## Next engineering step

The full-corpus runner should assign one persistent territory to each column,
checkpoint each territory independently, and resume only incomplete columns.
Completed columns can consolidate into a generation organ without waiting for
failed or interrupted peers. This preserves locality during learning and makes
the full run fault tolerant without synchronizing every byte.
