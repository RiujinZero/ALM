# Phase 23: GPU Structural Membrane

## Surprise now interrupts resident CUDA causally

The multi-column CUDA organism can now stop immediately before a byte that is
eligible to alter anatomy. The resident kernel tests the pending prediction's
surprise and the organism's birth refractory period before consuming the byte.
It returns the exact processed prefix, model and basal probabilities, and a
structural-interrupt flag.

At the membrane, lazy traces and local evidence are materialized into the host
organism. The canonical nursery, birth, column-local replacement, and prune
rules consume the interrupting byte. CUDA is then rebuilt from the revised
branch anatomy and axonal fan-out. No byte is replayed or learned twice.

The threshold check uses a `1e-5`-bit conservative margin. A numerically
borderline event may wake the host unnecessarily, but CUDA cannot silently
skip a birth that the synchronized organism would accept.

## Developmental wake windows

Early development produces surprise in clusters. Rebuilding a large device
body after every nursery proposal would turn structural rarity into allocator
traffic. `CudaDevelopmentalRunner` therefore accepts a
`structural_wake_bytes` interval. After an interrupt, the host organism remains
awake for that many causal bytes and handles every structural event in the
window before one anatomy rebuild.

This resembles punctuated cortical remodeling: routine signaling remains in
resident tissue, while clustered growth is consolidated during a short
developmental wake. The runner reports structural barriers, host wake bytes,
and anatomy rebuilds so the split is measurable rather than hidden.

## Oracle and TinyStories result

A 768-byte event-metabolism oracle crosses repeated wake barriers with a
64-byte window. CUDA matches the uninterrupted CPU learner's model and basal
probabilities within `2e-5`, with identical births, prunes, live relations, and
organism steps. The GPU, CPU-column, and online-stream regression set passes
32 tests.

A from-scratch 20,000-byte TinyStories probe used ALM-64MB carrying capacity,
event metabolism, a 256-byte structural wake, and intentionally disabled
perturb/select and lifecycle intervals to isolate this membrane:

| Measurement | Result |
|---|---:|
| Training throughput | 2,724 bytes/s |
| Mature births / prunes | 381 / 0 |
| Live nursery candidates | 150 |
| Embodied checkpoint bytes | 4,605,033 |
| Prequential model loss | 3.5308 bpb |

The earlier complete CPU developmental smoke measured 289 bytes/s. The new
structural path is 9.4x faster in this deliberately narrower comparison, but
it is not yet a replacement for the complete learner: CUDA still holds
dendritic geometry fixed between structural wakes.

## Remaining gate

Port branch-local perturb/select and lifecycle transitions to the same
resident CUDA loop. Those operations do not require graph reconstruction;
only an accepted birth or prune should rebuild the axonal index. Once that
parity test passes, expose the GPU developmental runner in the bounded
TinyStories CLI and restart the larger pilot from byte zero.

Phase 24 closes this gate: perturb/select and lifecycle are resident on CUDA,
the CLI integration is active, and the complete 20KB smoke matches the CPU
scientific result at 12.8x its training throughput.
