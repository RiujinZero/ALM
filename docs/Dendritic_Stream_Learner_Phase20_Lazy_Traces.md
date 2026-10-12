# Phase 20: Lazy Trace Physiology

## Silent tissue keeps time, not state traffic

The event-sparse CUDA organism no longer decays every temporal trace on every
byte. Each sensory or relation channel stores its last causal update. When an
axon next arrives—or a dendrite reads that channel—the kernel analytically
applies the elapsed exponential decay. Active relation channels then inject
their new activation and advance all six timescales together.

The running trace mean is also advanced analytically. For a silent interval
the implementation evaluates the closed-form convolution between trace decay
and mean decay, including the equal-decay limit. This preserves the CPU rule
without replaying every silent byte. Before transferring state back to CPU,
one compact CUDA pass materializes all traces and means at the episode
frontier. Dense and event execution pass probability, causal-learning,
dendritic-evidence, state-transfer, CPU-continuation, and post-transfer
adaptation equivalence tests.

The lazy representation currently requires one learning mode within an event
episode. Resetting the episode permits switching between inference and
learning. This makes the temporal contract explicit instead of silently
mis-advancing trace means across a mode boundary.

## Reproducible filled-body measurement

`dendritic_basin_gpu_filled_scale_probe` now tiles all live branches from a
learned checkpoint into independent cortical columns. It reports inference
and causal-learning rates separately; previous one-off measurements could not
be rerun from a checked-in command.

On the local RTX 5080, using 8,192 real TinyStories bytes and the 155-branch
Phase 14 learned motif:

| Body | Relations | Active during learning | Inference B/s | Learning B/s |
|---|---:|---:|---:|---:|
| ALM-64MB | 7,125 | 8.15% | 25,822 | 14,840 |
| ALM-1GB | 112,341 | 8.15% | 5,640 | 3,062 |

The important result is negative as well as positive. Lazy traces make frozen
inference substantially cheaper than causal learning, but ALM-1GB learning
remains near the prior 2,997 B/s result. A full pass over the 1.924GB local
TinyStories file is still about 7.3 days at filled-body geometry. Removing the
remaining full-body evidence scan did not materially change that rate either.

## Next constraint: compact plasticity traffic

The next target is the set of locally active predictor synapses. The kernel
still scans the whole body to resolve the previous byte's local outcomes and
again to publish current local states, although only a sparse fraction can
learn. The biological analogue is short-range axonal boutons.

A first double-buffered global list preserved exact causal behavior but was
rejected by measurement: atomic publication made ALM-1GB learning fall from
about 3.02KB/s to 2.26KB/s. One globally contended queue is the wrong anatomy.
The next attempt should compact independently inside each cortical column,
then prefix-sum the small column counts. That follows local axon terminals and
avoids both whole-body polling and global append contention.

After that change, the same probe will tell us whether per-active-branch
Welford evidence or predictor-row plasticity is the true learning membrane.
Only then is it sensible to start the full TinyStories/ALM-1GB run.
