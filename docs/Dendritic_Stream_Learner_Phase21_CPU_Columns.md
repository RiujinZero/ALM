# Phase 21: CPU Cortical Columns

## The hardware mismatch is real

The event-sparse ALM performs irregular causal fan-out, short predictor-row
updates, and depth barriers. Those are not the large dense matrix products for
which current GPUs are optimized. The Ryzen 9 9950X provides 16 physical cores
and 32 hardware threads, so a 30-worker experiment was run directly rather
than projecting from peak FLOPS.

The first experiment parallelized independent dendrite responses within each
causal depth using Numba `prange`. It preserved the causal depth barrier and
passed the online learner tests. It was then measured at 1, 16, and 30 workers.

| Filled body | Workers | Bytes/s |
|---|---:|---:|
| 155 learned branches | 1 | 13,527 |
| 155 learned branches | 16 | 11,231 |
| 155 learned branches | 30 | 8,414 |
| ALM-64MB / 7,125 branches | 1 | 1,493 |
| ALM-64MB / 7,125 branches | 16 | 991 |
| ALM-64MB / 7,125 branches | 30 | 956 |
| ALM-1GB / 112,341 branches | 1 | 107 |
| ALM-1GB / 112,341 branches | 16 | 75 |
| ALM-1GB / 112,341 branches | 30 | 71 |

The parallel patch was rejected and removed. Even at ALM-1GB, where about
9,000 branches are scheduled per byte, entering a threaded region once per
causal depth costs more than the work it distributes. More cores do not rescue
an execution topology with thousands of microscopic barriers.

This also means a Threadripper would not automatically beat the RTX 5080 with
the present CPU backend. The current ALM-1GB GPU learner sustains about 3,000
bytes/s. The Python/Numba depth-dispatch CPU path is roughly forty times slower
with 30 workers. That gap is primarily software topology, not a verdict on
sparse CPU execution.

## CPU-native anatomy

The next CPU backend should make a worker correspond to a cortical territory,
not to one transient batch of dendrites:

1. Start a persistent worker for each physical or selected logical core.
2. Give each worker contiguous columns, predictor rows, trace state, and local
   plasticity evidence so its working set remains cache- and NUMA-local.
3. Broadcast the sensory byte once to every territory.
4. Let each territory traverse its local axonal consumer lists without a task
   launch.
5. Exchange only compact cross-column spikes at causal-depth boundaries.
6. Reduce a few local column winners into the global somatic vote.
7. Keep an entire byte chunk inside one compiled resident loop; Python enters
   only at a structural birth, pruning, checkpoint, or progress barrier.

This is closer to cortex: a column does not recruit thirty new workers for
every gamma-cycle depth. It keeps local cells resident and exchanges spikes
with neighboring tissue. It is also the mapping under which a many-core,
high-memory-bandwidth Threadripper could plausibly outperform the GPU.

## The learned anatomy already clusters well

The new `dendritic_basin_cpu_column_probe` measures recursive-edge locality
and worker load before any execution backend is chosen. On the filled
ALM-1GB geometry, grouping contiguous anatomical columns gives:

| Workers | Recursive locality | Load imbalance | Relations/worker |
|---:|---:|---:|---:|
| 8 | 99.80% | 0.27% | 13,824–14,080 |
| 16 | 99.61% | 2.09% | 6,869–7,168 |
| 30 | 99.33% | 2.54% | 3,541–3,840 |

A generic greedy graph-affinity assignment reduced 30-worker locality to
97.79%. Development already places recursive children near their parent
columns, so rediscovering clusters after the fact damages the organism's
spatial structure. Contiguous cortical territories are therefore the default;
the affinity strategy remains in the probe as an explicit ablation.

This is the grouping result the resident backend needed: only 34 of 5,072
recursive motif edges cross a worker boundary at 30 workers. Most computation,
predictor memory, and plasticity evidence can remain owned by one persistent
territory.

## Next experiment

The frozen resident CPU-column evaluator is now implemented. An entire byte
chunk remains inside compiled code. A small parallel helper owns the
depth-local territory loop; the first attempt to parallelize the whole
organism exceeded Numba's parallel analysis, so somatic fusion and byte order
remain explicitly serial. Recursive axonal stamps cross a depth barrier before
their consumers run.

The evaluator matches the ordinary event oracle across split chunks within
`3e-7`; the measured maximum error is about `1.6e-8`. On 4,096-byte ALM-1GB
runs, median frozen throughput is:

| Workers | Bytes/s | Speedup vs. one worker |
|---:|---:|---:|
| 1 | 1,210 | 1.00x |
| 8 | 3,145 | 2.60x |
| 16 | 3,983 | 3.29x |
| 30 | 3,793 | 3.13x |

Removing Python depth dispatch makes the resident kernel roughly 37x faster
than the old ALM-1GB CPU event path. Sixteen workers—one per physical 9950X
core—beat 30 logical workers, so SMT is not useful for this memory-sensitive
tissue. The 16-core result is only about 1.42x behind the RTX 5080's 5,640 B/s
frozen rate. A Threadripper with more physical cores and enough memory
bandwidth can therefore plausibly overtake the GPU under this column mapping.

Phase 22 moves predictor outcome updates, utility, dendritic evidence, trace
means, and local perturb/select adaptation into the resident column loop. See
`Dendritic_Stream_Learner_Phase22_Resident_CPU_Plasticity.md` for the measured
learning rates and the remaining developmental barrier.
