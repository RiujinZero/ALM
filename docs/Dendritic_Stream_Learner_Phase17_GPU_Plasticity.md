# Phase 17: GPU Local Plasticity

## First learning primitive

The first CUDA plasticity rule updates causal predictor tissue, not anatomy.
Every active relation owns one 256-way outcome row. One CUDA block advances
that row through a stream of outcomes, using the same relation prior,
lifecycle-dependent decay, local information gain, utility decay, and state
observation rule as the CPU learner. No optimizer or global gradient is
introduced.

On 128 live relations from the Phase 14 contextual checkpoint, 4,096 causal
outcomes per relation produced 524,288 branch-local learning events:

- CPU reference: 1.060 seconds;
- cached GPU kernel: 0.0256 seconds;
- GPU rate: 20.45 million branch outcomes/second;
- speedup over the scalar oracle: about 41x.

After all 4,096 recurrent updates, maximum event-count drift was `0.00178`,
maximum state-count drift was `0.00342`, and state observation counts remained
exact. The quantities that can affect developmental thresholds—event and state
utility—differed by only `1.25e-6` and `1.86e-6` respectively.

This supports FP32 predictor masters on device. It does not yet authorize GPU
developmental decisions. The integrated learner must escalate utility values
near consolidation, dormancy, and pruning thresholds to the CPU oracle and
must reproduce the same structural event sequence before those barriers move
onto the device.

## Next integration

The local rule is now joined to the cooperative multi-column causal kernel.
Context and base counts, event/state predictor rows, utility, observations,
firings, traces, trace means, persistence, competition, and residual fusion
advance together without leaving the device. A synchronization bridge restores
that plastic and transient state to the CPU organism at a barrier.

On 4,096 learned-checkpoint TinyStories bytes with development held fixed,
integrated GPU and CPU learning differed by at most `1.75e-7` in causal
probability. Predictor-row drift stayed below `0.00025`, utility below
`8.18e-7`, and observation/firing counts were exact. After transferring the
GPU state back, a 512-byte CPU continuation matched the uninterrupted CPU
organism within `1.63e-8` probability.

## Dense scaling boundary

Filled-body throughput probes replicate learned branch motifs to measure
execution cost; they are not language-quality results:

| Body | Relations | GPU learned bytes/s | CPU learned bytes/s | Speedup |
|---|---:|---:|---:|---:|
| ALM-64MB | 7,125 | 11,919 | 4,070 | 2.93x |
| ALM-1GB | 112,341 | 2,008 | 121 | 16.66x |

GPU advantage grows with cortical body size, but dense ALM-1GB remains too
slow for routine full-corpus development. At the local TinyStories training
size of 1,924,281,556 bytes, these measured rates imply about 45 hours for a
filled ALM-64MB pass and 11 days for a filled ALM-1GB pass. Natural growth is
cheaper early in a run because most slots are empty, but this is still the
point where the deferred event-sparse backend becomes necessary.

The next barrier therefore has two parts: maintain dendrite sample/statistical
plasticity on GPU so CPU developmental decisions see complete evidence, then
activate event-sparse execution once the dense body is large enough for the
saved branch work to dominate scheduling overhead. Fast local synaptic
adaptation remains continuous; expensive anatomical rewiring remains rare.
