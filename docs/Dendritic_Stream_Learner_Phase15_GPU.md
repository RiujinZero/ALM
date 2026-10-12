# Phase 15: GPU Cortical Execution

## Purpose

Phase 15 maps the columnar ALM onto the available RTX 5080 without changing
the causal or biological organization. Bytes remain sequential. Parallelism
comes from cortical tissue within a byte and from independent reset episodes.
A CUDA block corresponds to a small cortical column and retains its transient
activity across the entire episode.

The target is a persistent kernel that keeps the complete organism resident
in VRAM and advances many consecutive bytes without host launches. It returns
to the CPU only at explicit structural barriers and checkpoint boundaries.

## First end-to-end frozen backend

`CudaFrozenDendriticEvaluator` is the first real backend slice. It loads a
learned dense one-column organism of at most 256 relation slots, converts its
inference tissue to FP32, and keeps eligibility traces, persistence, pending
probabilities, and causal byte position on the GPU. Predictor distributions
are consolidated into log-probability tissue once at load time instead of
recomputing identical logarithms on every byte.

Birth order is also the relation graph's topological order. The first kernel
therefore advances recursive branches in exact anatomical order. Its other
lanes decay traces, evaluate the 256-way predictor surface, and normalize the
residual fusion. This prioritizes a clean CPU oracle before introducing
multi-column synchronization.

On the learned 100KB ALM-2MB Phase 14 checkpoint, a 4,096-byte random stream
matched the CPU across 4,095 scored transitions with maximum absolute
probability error `3.97e-8`. Splitting the GPU stream at byte 1,000 produced
the same continuation, confirming that pending prediction and transient state
remain causal across host chunk boundaries.

A single 155-branch stream is too small to occupy this GPU: the frozen kernel
runs about 10KB/s versus roughly 234KB/s for the compiled CPU path. This is an
important lower-scale boundary, not a useful acceleration result. Independent
episodes expose the available parallelism. With 512 reset episodes of 4,096
TinyStories bytes sharing one frozen organism, the kernel scores 2,097,152
bytes at 2.28MB/s, 9.34x the measured one-stream CPU rate of 245KB/s. Across
three oracle episodes, maximum absolute probability error is `1.12e-7`, mean
error is `8.80e-9`, and GPU loss is 3.35317 bits/byte.

The same end-to-end kernel can store transient traces and persistence in
native BF16 while keeping predictor tissue, fusion, and output probabilities
in FP32. It scores 2.28MB/s (9.30x CPU), so BF16 does not improve speed at
this small body size. Its validation loss is 3.35320 bits/byte, maximum oracle
probability drift is `2.46e-4`, and mean drift is `1.02e-7`. BF16 is therefore
feasible as compact transient tissue, but it has not earned selection as the
default backend until a larger multi-column body shows a speed or capacity
benefit.

The batched measurement is frozen inference, not shared-state online learning.
It establishes the execution shape needed for story-batched validation and
shows why GPU crossover depends on either many independent episodes or a much
larger multi-column body.

## Runtime

The machine has an NVIDIA GeForce RTX 5080 with 16GB VRAM and compute
capability 12.0. CUDA Toolkit 13.3 and `nvcc` are installed. The existing
PyTorch package is CPU-only and Numba cannot compile CUDA 13.3 kernels because
the toolkit does not expose the dynamic NVVM library it expects.

CuPy 14.2 provides a working CUDA 13 runtime and compiles native BF16 and FP8
kernels on this GPU. The optional `alm-gpu` dependency installs
`cupy-cuda13x`; CPU ALM users do not inherit that dependency.

## Resident primitive benchmark

`dendritic_basin_gpu_precision_probe` keeps a causal token loop inside one
kernel and performs representative branch-local trace updates and
token-indexed predictor-table reads. It is an upper-bound primitive rather
than an end-to-end learner. Equivalent byte rates omit structural growth,
competition, and checkpoint work.

At ALM-64MB geometry (7,125 contextual-residual branches):

| Predictor storage | Branch evaluations/s | Equivalent causal bytes/s |
|---|---:|---:|
| FP64 | 26.9B | 3.78M |
| FP32 | 48.8B | 6.86M |
| BF16 | 71.7B | 10.07M |
| FP8 E4M3 | 74.6B | 10.47M |
| FP8 E5M2 | 74.7B | 10.48M |

At ALM-256MB geometry (28,707 branches), FP8 storage reaches approximately
150B branch evaluations/s versus 70B for FP32 and 54B for FP64. The full
learner will be far slower, but the result confirms that intra-byte cortical
work is large enough to occupy the GPU and that compact tissue improves the
large-body regime.

## Precision feasibility

Synthetic 256-way probability rows show:

| Storage | Mean distribution KL error |
|---|---:|
| BF16 | 0.000002 bits |
| Unscaled FP8 E4M3 | 0.7520 bits |
| Row-scaled FP8 E4M3 | 0.000490 bits |
| Unscaled FP8 E5M2 | 0.002016 bits |
| Row-scaled FP8 E5M2 | 0.001966 bits |

E4M3 cannot represent small categorical probabilities without a local scale.
A single scale per predictor row makes it more accurate than E5M2 because its
extra mantissa bit matters more than E5M2's range for normalized tissue.

The learned-checkpoint probe quantizes a real 100KB ALM-2MB contextual
residual organism, scores the fixed 100KB validation prefix, and continues
learning for 20KB:

| Policy | Validation bpb delta | Continuation bpb delta | Births/prunes |
|---|---:|---:|---:|
| FP64 authority | 0 | 0 | 124 / 124 |
| FP32 | −0.000000000017 | +0.000000000046 | 124 / 124 |
| BF16 | +0.000020978 | +0.000058112 | 124 / 124 |
| Scaled FP8 E4M3 | +0.000274728 | −0.000105080 | 123 / 123 |
| Scaled FP8 E5M2 | −0.000146212 | +0.000993985 | 126 / 126 |

FP32 and BF16 preserve structural counts on this probe. FP8 slightly changes
which surprising bytes cross developmental thresholds even though prediction
loss remains close.

## Precision policy

The initial GPU learner should use:

- FP32 master predictor counts, dendrite parameters, utility, and lifecycle
  thresholds;
- BF16 activity, traces, response values, and predictor inference mirrors;
- FP64 accumulated log-loss metrics;
- optional row-scaled FP8 E4M3 inference mirrors only after the BF16 backend
  is correct and faster end to end.

An FP8 fast path must escalate to FP32 when surprise or utility lies near a
birth, consolidation, dormancy, or prune threshold. Structural plasticity is
never decided from FP8 alone.

## Gates

1. Extend frozen execution from one column to many columns, with each column
   reducing local winners before a sparse soma fusion phase.
2. Re-test the implemented BF16 transient activity/traces on multi-column
   bodies. Retain FP32 as the oracle and require a speed or capacity win.
3. Add local outcome learning with FP32 masters and BF16 inference mirrors.
4. Return to the host before developmental events and require identical
   birth/prune decisions on an oracle stream.
5. Benchmark complete 2MB/16MB/64MB runs. Keep the backend only if ALM-16MB
   improves at least fivefold end to end.
6. Add threshold-aware scaled E4M3 and retain it only if structure and held-out
   BPB pass the same gates.
