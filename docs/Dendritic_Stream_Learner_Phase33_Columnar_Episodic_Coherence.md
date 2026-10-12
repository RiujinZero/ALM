# Phase 33: Columnar Episodic Coherence

## Cross-column identity

Episodic organs can now consolidate across independently trained cortical
territories. Local integer IDs are not treated as identity. Content assemblies
are aligned by their discovered raw receptor path, then every episodic branch
is remapped before counts and utility evidence combine. Mature branches unable
to cross the recall gate are pruned after consolidation.

## Measurements

| Layout | Exposure | Held-out bpb | Training rate | Body after pruning |
|---|---:|---:|---:|---:|
| Local, 30 columns | 30MB | 2.356158 | 1,206,760 B/s | 109,511 B |
| Episodic, 4 columns | 1MB | 2.359417 | 152,449 B/s | 1,140,910 B |
| Episodic, 30 columns | 30MB | 2.313817 | 880,403 B/s | 3,030,625 B |

The matched 30MB episodic organ gains 0.042340 bpb over local context. This is
larger than the 0.035758-bpb single-column gain at 1MB, so territorial learning
and receptor-path consolidation preserve the coherence effect.

The 30-column run learned 9,175 content assemblies. Consolidation pruned
110,687 mature failed episodic branches and retained 39,423 branches with
120,053 output synapses; 636 currently cross the utility gate. The model is
still only 3.03MB.

## Runtime shift

Parallel training consumed 34.1 seconds under contention from the older
30-worker oracle. Consolidation consumed another 15.0 seconds. Sparse learning
is no longer the dominant scaling problem: Python object remapping and pruning
at the column boundary are. Compact sorted arrays, path interning before merge,
and per-column metabolic pruning should reduce that cost.

At the observed training rate, a full TinyStories pass projects to roughly
36 minutes before consolidation and validation. This remains a projection;
episodic anatomy growth and merge cost must be measured on a larger run before
launching the full corpus.
