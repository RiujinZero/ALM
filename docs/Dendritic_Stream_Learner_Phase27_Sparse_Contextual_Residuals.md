# Phase 27: Sparse Contextual Residual Dendrites

## Anatomy

The developmental opponent organ now grows a second contextual arbor above
its basal assembly transitions. A basal byte assembly predicts the next byte.
When that prediction remains surprising, the immediately older assembly may
join the current assembly on a candidate dendritic branch. Three surprising
coactivities recruit the branch from nursery into living tissue.

A live branch stores synapses only to outputs actually observed in that
context. Its prediction uses the basal basin as a finite-strength prior:

`P(output | older, current) = (context count + 4 * basal P) / (context total + 4)`

This is a contextual residual rather than an isolated trigram table. An
immature or absent branch is exactly the basal predictor; a young branch can
specialize without discarding the older circuit's evidence. Streaming loss
touches one active basal synapse and, when present, one active contextual
synapse. Full basin broadcast remains available for generation.

## 100KB TinyStories result

| Measurement | Sparse contextual organ |
|---|---:|
| Held-out loss | 2.459222 bpb |
| Developmental basal | 3.333684 bpb |
| Categorical trigram control | 2.578776 bpb |
| Training throughput | 118,525 bytes/s |
| Evaluation throughput | 242,085 bytes/s |
| Serialized body | 52,254 bytes |
| Living contextual branches | 563 |
| Living contextual output synapses | 3,749 |
| Context branches still in nursery | 140 |

The contextual arbor improves held-out loss by 0.874462 bpb over its own basal
organ and by 0.119554 bpb over the categorical trigram. The latter advantage
comes from hierarchical backoff: sparse contextual evidence is continuously
regularized by the already learned basal basin instead of receiving an
independent flat prior.

The complete TinyStories training file is 1,924,281,556 bytes. At the current
single-threaded Python rate, one causal pass projects to roughly 4.5 hours.
That estimate excludes checkpointing and final evaluation, and it is not yet a
measured full-corpus runtime. Compiled column workers can improve it further;
the important result is that runtime now scales with stimulated living tissue
rather than the maximum possible contextual graph.

## Next experiment

Run the same organ over a larger prefix to observe whether branch growth,
serialized bytes, and throughput stabilize. Then move its compact arrays into
the columnar CPU backend: partition by current assembly, let each column own
its contextual branches, and reduce only the stimulated output evidence.
GPU execution becomes useful again when enough simultaneously active columns
exist to fill warps without densifying dormant anatomy.

## 1MB scaling check

Increasing the training prefix tenfold did not trigger combinatorial growth.

| Measurement | 100KB | 1MB |
|---|---:|---:|
| Held-out loss | 2.459222 bpb | 2.390727 bpb |
| Trigram control | 2.578776 bpb | 2.406610 bpb |
| Training throughput | 118,525 B/s | 122,249 B/s |
| Serialized body | 52,254 B | 69,099 B |
| Context branches | 563 | 790 |
| Context synapses | 3,749 | 6,349 |

Data exposure grew 10x while the body grew only 1.32x and throughput remained
stable. At the measured 1MB rate, the full 1,924,281,556-byte file projects to
4.37 hours before checkpoint and filesystem overhead.

The sparse-only production runner
`dendritic_basin_sparse_contextual_train` streams directly from disk, writes
atomic checkpoints and compact progress JSON, resumes at the exact byte
offset with temporal state intact, and evaluates without learning. A 50KB
checkpointed smoke sustained 151,191 B/s and reached 2.474576 held-out bpb.

## 10MB checkpointed run

The isolated production path completed 10,000,000 bytes in 90 seconds while
writing an atomic checkpoint every 2MB.

| Measurement | Result |
|---|---:|
| Sustained training throughput | 111,097 B/s |
| Online training loss | 2.441189 bpb |
| Held-out loss over 100KB | 2.357226 bpb |
| Evaluation throughput | 213,974 B/s |
| Serialized body | 95,203 B |
| Living contextual branches | 1,096 |
| Living contextual synapses | 10,177 |
| Context candidates in nursery | 243 |

The 10MB body is only 1.38x the 1MB body after another 10x increase in
experience. At this checkpointed rate, a full pass projects to 4.81 hours on
one Python CPU core. Parallel cortical columns are now the main available
hardware multiplier: independent story territories can develop locally and
periodically consolidate count-based anatomy without densifying it.
