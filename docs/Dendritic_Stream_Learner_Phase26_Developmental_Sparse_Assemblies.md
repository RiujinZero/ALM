# Phase 26: Developmental Sparse Assemblies

## Only living tissue computes

The opponent-receptor scaffold now has a developmental version. Two-bit
coincidences enter a nursery, repeated coactivity recruits four-bit assemblies,
and repeated coactivity between those assemblies recruits an eight-bit output
basin. No byte assembly exists merely because the byte is possible. Sparse
assembly-to-assembly synapses are allocated only for transitions that occur.

At three confirmations per anatomical level, a completely novel byte needs
repeated evidence through the hierarchy before it becomes a predictive
identity. Until then, prediction falls back to the shallower population. This
creates a measurable developmental cost rather than granting the final
assembly tree at initialization.

## 100KB TinyStories result

| Measurement | Developmental opponent attractor |
|---|---:|
| Held-out loss | 3.333684 bpb |
| Categorical bigram control | 3.333528 bpb |
| Difference | +0.000156 bpb |
| Training throughput | 97,255 bytes/s |
| Evaluation throughput | 123,247 bytes/s |
| Serialized body | 25,625 bytes |
| Born byte assemblies | 67 |
| Remaining nursery candidates | 3 |
| Living transition synapses | 757 |

The developmental hierarchy therefore recovers the categorical bigram within
0.005%, without preallocating the 65,536-transition Cartesian product. The
fixed sparse-attractor scaffold reached 3.335231 bpb; development is slightly
better because unconfirmed rare assemblies naturally back off rather than
receiving poorly estimated context rows.

The dense ALM contextual matrix alone occupies 524,288 bytes. The complete
developmental opponent body, including its fixed receptor codebook, nursery,
assemblies, sparse transitions, and counters, is 20.5x smaller. Its measured
CPU rate is far above the current full CUDA ALM because this probe contains
only the basal assembly circuit, not temporal traces and adaptive contextual
dendrites. The comparison isolates the value of sparse embodiment; it is not
yet an end-to-end backend comparison.

## Consequence for ALM scaling

ALM byte classes should allocate living assemblies and synapses, not reserve a
dense tensor for every possible relationship. The next nature-shaped learner
should:

1. keep the 16 opponent receptors and nursery-grown identity assemblies;
2. replace the dense basal transition matrix with sparse attractor synapses;
3. expose both receptor and assembly activity to temporal traces;
4. grow contextual residual dendrites only on repeated surprising coactivity;
5. compact the CUDA resident body around live branches and observed axonal
   edges, rebuilding only after accepted anatomy changes.

This changes the likely ALM-1GB regime from roughly 112,000 preallocated
branches to however many living assemblies, residual dendrites, and sparse
synapses actually fit in one billion serialized bytes.

## Sparse readout throughput

The first implementation broadcast a full 256-basin distribution at every
training step even though learning and loss measurement consume only the
observed basin. Generation needs that broadcast; streaming learning does not.
Reading only the stimulated basin preserves the body and probabilities exactly
while avoiding 255 unnecessary output reads and a temporary dense vector.

| 100KB measurement | Dense broadcast | Stimulated basin only | Gain |
|---|---:|---:|---:|
| Fixed sparse attractor train | 132,170 B/s | 392,687 B/s | 2.97x |
| Developmental attractor train | 97,255 B/s | 180,308 B/s | 1.85x |
| Developmental attractor eval | 123,247 B/s | 508,912 B/s | 4.13x |

The developmental path gains less during training because nursery growth is
now its dominant cost. This establishes the engineering rule for subsequent
layers: prediction, learning, and generation are distinct modes. Prediction
and learning touch only stimulated living tissue; whole-basin recruitment is
reserved for generation or explicit evaluation of the output field.
