# Phase 19: Axonal GPU Routing

## From polling to fan-out

The event backend now has true axonal routing. A CSR consumer index maps each
sensory or relation channel to only the dendrites that receive it. Within the
persistent cooperative kernel:

1. the sensory spike seeds compact causal-depth queues;
2. active recursive assemblies enqueue their downstream consumers;
3. useful persistent branches carry a compact list into the next byte;
4. generation stamps deduplicate convergent sensory, recursive, and persistent
   routes;
5. each depth queue completes before the next laminar depth begins.

This queue path passes the same GPU/CPU causal learning, evidence, statistics,
state-transfer, continuation, and post-transfer adaptation tests as dense
execution.

At literal ALM-1GB geometry the consumer index occupies 1,799,512 bytes for
112,341 branches and 112,597 possible afferent channels. On the copied learned
anatomy, the worst-fanout sensory byte (space) reaches 19,565 branches, or
17.46% of the body. Weighting sensory fan-out by one million actual TinyStories
bytes gives an 8.02% scheduled upper bound before recursive propagation. The
index therefore avoids asking roughly 92% of the body whether it received the
typical sensory event.

## Persistence as metabolic selection

The first queue run revealed that persistence, rather than sensory fan-out,
was keeping 39% of the copied body active. This learned organism had mature
state evidence in 150 of 155 source branches, but no branch had positive state
utility. Continuing to propagate those states was metabolically wasteful.

The opt-in `utility_gated_persistence` rule gives immature states their full
observation window, then retains only mature states meeting the configured
persistence utility threshold. On a real 100KB learning continuation it raises
CPU event throughput from 8.28KB/s to 12.76KB/s while changing loss by only
`+0.000097` bits/byte.

## Filled-body throughput

The filled geometries below replicate learned motifs and measure execution,
not language quality:

| Body | Scheduled branches/byte | GPU bytes/s | CPU bytes/s | GPU speedup |
|---|---:|---:|---:|---:|
| ALM-64MB | 573 / 7,125 (8.04%) | 14,668 | 10,274 | 1.43x |
| ALM-1GB | 9,301 / 112,341 (8.28%) | 2,997 | 167 | 17.97x |

At ALM-1GB, axonal routing plus utility-gated persistence is about 1.49x faster
than the prior dense GPU result and would reduce a filled-body pass over the
local 1.924GB TinyStories file from roughly 11 days to 7.4 days. This is a real
improvement, but still not the desired training regime.

## Next constraint: the trace field

Silent dendrites no longer execute, but all six temporal traces still decay for
every sensory and relation channel on every byte. That dense field is now the
dominant large-body cost. The next optimization is lazy trace physiology:
store a channel's last-update time, analytically apply accumulated exponential
decay only when the channel is read or receives activity, and derive an exact
or bounded-lazy update for its trace mean. This should make cost follow axonal
events rather than total body size.
