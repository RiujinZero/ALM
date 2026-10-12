# Phase 24: Resident GPU Development

## The complete local organism now lives on CUDA

The multi-column CUDA loop now includes the two remaining fixed-anatomy
developmental processes:

- branch-local counter-seeded perturb/select over each dendrite's sample ring;
- juvenile consolidation, consolidated dormancy, and firing-driven
  reactivation with exact organism-step timestamps.

These operations follow the column ontology rather than a matrix ontology.
The thread evaluating a branch owns its local evidence, proposes a deterministic
four-coordinate dendritic mutation, and retains it only when the local
interaction objective improves. A parallel lifecycle sweep changes branch
state before column competition. Device counters preserve every transition
even if a branch changes state more than once between host synchronizations.

The CPU is now needed only for the mutable anatomical graph: nursery
recruitment, accepted birth, column-local prune/replacement, checkpointing, and
rebuilding axonal fan-out. Routine prediction, outcome credit, traces,
persistence, dendritic adaptation, and lifecycle all remain resident on the
GPU.

## Parity gates

The structural oracle now runs with perturb/select enabled and crosses repeated
birth/prune barriers without changing the CPU trajectory. A second forced
lifecycle oracle exercises rapid consolidation, dormancy, and reactivation.
It matches lifecycle state, transition timestamps, transition counts, firing
timestamps, and the resident CPU-column dendritic parameters. Model
probabilities remain within `2e-4` through the deliberately aggressive
adaptation probe.

The GPU, CPU-column, and online-stream regression selection passes 33 tests.
The bounded TinyStories CLI exposes the complete path through
`--cuda-developmental`; `--structural-wake-bytes` controls how long the host
stays awake after a structural interrupt.

## Complete ALM-64MB TinyStories smoke

The original Phase 22 CPU developmental smoke and the new CUDA run both used
20,000 training bytes, 5,000 held-out bytes, seed 0, ALM-64MB carrying
capacity, contextual residuals, event metabolism, and utility-gated
persistence. CUDA used a 256-byte structural wake.

| Measurement | CPU columns | Resident CUDA |
|---|---:|---:|
| Training bytes/s | 289 | 3,710 |
| Held-out model bpb | 3.461789 | 3.461789 |
| Held-out basal bpb | 3.467624 | 3.467624 |
| Births / prunes | 127 / 0 | 127 / 0 |
| Live mature / nursery | 127 / 59 | 127 / 59 |
| Embodied bytes | 2,677,336 | 2,677,336 |

This is a 12.8x end-to-end training speedup with the same scientific outcome.
It includes real structural interruptions, adaptation, lifecycle, evaluation,
and serialization rather than a filled-body frozen benchmark.

## Next empirical gate

Restart the 10MB ALM-64MB pilot on resident CUDA. Record throughput as anatomy
grows, rolling and held-out bpb, birth/prune density, lifecycle occupancy, and
embodied bytes. The existing CPU run remains an independent oracle. If the
10MB result retains contextual gain and approaches the trigram control, extend
the same organism toward the full TinyStories corpus before increasing its
byte class.

The clean CUDA pilot is now running detached in
`runs/alm_phase24_gpu_developmental_10mb`. Its first 100,000-byte checkpoint
measured 3,116 bytes/s, 3.3217 rolling bpb, 458 live relations, and no prunes.
That implies an initial training ETA of approximately 53 minutes. The CPU
10MB organism continues independently for an end-to-end scientific comparison.
