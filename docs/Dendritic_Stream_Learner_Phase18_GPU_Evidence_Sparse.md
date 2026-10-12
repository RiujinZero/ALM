# Phase 18: GPU Dendritic Evidence and Sparse Gate

## Complete local evidence

The cooperative CUDA learner now advances the evidence required for dendrite
adaptation as well as predictor plasticity. Every metabolically evaluated
branch maintains:

- its circular `(current, trace)` sample membrane;
- sample count and write position;
- Welford means and second moments for current, trace, and response;
- response/current and response/trace covariance accumulators;
- normalized trace means used by developmental parent selection.

The CPU synchronization bridge restores all of this evidence together with
predictor rows, utility, observations, firings, traces, persistence, pending
local outcomes, column winners, and the pending successor distribution.

On the CUDA regression organism, ring positions/counts and Welford observation
counts are exact. Floating evidence remains within `1e-3` after 1,024 recurrent
learning bytes. Forcing CPU dendrite adaptation immediately after transfer
selects the same perturbations and produces matching dendrite parameters within
`2e-6`. A CPU continuation therefore sees a developmentally usable organism,
not merely a matching output distribution.

## First sparse gate

The GPU backend now honors the existing event metabolism. A branch is evaluated
only if its current sensory/relation afferent is active or its own persistent
state remains above support. Recursive activation still crosses an explicit
depth barrier. Dense and event modes both pass causal learning, evidence,
statistics, synchronization, and post-transfer adaptation tests.

This first implementation gates expensive dendrite and evidence arithmetic but
still scans branch slots and decays the full trace field. On a filled synthetic
ALM-64MB body copied from learned motifs, only 39.0% of branches are evaluated
per byte, but GPU throughput is 11.3KB/s versus 10.4KB/s on CPU: only 1.09x.
The gate is semantically correct but does not yet earn its scheduling overhead
on this geometry.

## Required sparse anatomy

The next implementation must stop scanning silent tissue. The natural data
structure is an axonal consumer index:

1. each sensory or relation channel owns a compact list of downstream
   dendrites;
2. an afferent spike appends only those consumers to the appropriate causal
   depth queue;
3. persistent branches append themselves to the next byte's queue;
4. generation stamps deduplicate branches reached by both routes;
5. compact depth queues execute across the cooperative grid;
6. trace state moves toward lazy per-channel decay so silent channels consume
   neither arithmetic nor bandwidth.

This is the GPU analogue of cortical fan-out: activity travels down axons to
actual synapses rather than asking every dendrite in the brain whether it heard
the spike. It is now justified by measurement, especially on the road to
ALM-1GB where a dense byte touches 112,341 relation slots.
