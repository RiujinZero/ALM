# alm_native — C++ backend for the ALM sequence cortex

Phase 123 measured the ceiling of the Python cortex and named the cause:
irreducibly sequential dictionary work inside a column, needing "a persistent
compiled sparse-map representation, not additional micro-optimizations or
weaker learning." This is that representation.

Two bodies, matching the wake/sleep lifecycle:

| Body | Role | Learns | Storage |
|---|---|---|---|
| `GrowableCortex` | wake development | yes | arena rows with spare capacity, nursery banks, free lists |
| `PackedCortex` | mature phenotype | no | one contiguous CSR block per level, exact-size rows |

## Correctness contract

The native backend is a **bit-exact** stand-in, not an approximation. Every
probability equals the Python reference's IEEE-754 bits exactly, and the
developed anatomy matches count for count — including which nursery paths
get evicted, since FIFO eviction order decides which contexts ever mature.

That is enforced by `tests/test_native_cortex_matches_python.py`, which
compares raw bits (never a tolerance) across development, frozen scoring,
packing, nursery eviction under a deliberately tight cap, dense-row
promotion, episode reset, and nursery shedding.

Consequently this code may **not** be built with fast-math: reassociation or
FMA contraction would change the low bits of the backoff chain. The build
pins `/fp:precise` on MSVC and `-ffp-contract=off` elsewhere.

## Build

```bash
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Release
```

On Linux/macOS the default generator works:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
```

Python finds the result automatically:

```python
from manifoldflow.dendritic_basin.native_cortex import NativeGrowableCortex

cortex = NativeGrowableCortex()
probabilities = cortex.observe_chunk_probabilities(stream, learn=True)
packed = cortex.pack()          # freeze into the mature inference body
```

## Measured on real TinyStories bytes

5 MB development, 1 MB frozen validation, single thread, Ryzen 9 9950X:

| Measure | Python | Native growable | Native packed |
|---|---:|---:|---:|
| Development | 125,417 B/s | 2,080,656 B/s (16.6x) | — |
| Frozen scoring | 206,954 B/s | 11,722,587 B/s | 14,431,472 B/s (69.7x) |
| Resident body | — | 35.03 MB | 18.39 MB (52.5%) |
| Probability mismatches | — | 0 | 0 |

Reproduce with:

```bash
python -m manifoldflow.experiments.native_cortex_benchmark
```

## Where the speed comes from

- **Flat open-addressing maps** (`hash_map.hpp`) instead of node-per-entry
  dictionaries: a context lookup touches one or two cache lines rather than
  chasing a pointer per probe. Deletion uses backward-shift rather than
  tombstones so nursery churn does not degrade probe chains.
- **One shared row arena** (`row_arena.hpp`) instead of millions of small
  dicts. Successor rows are `(byte, count)` runs scanned 16 keys at a time
  with SSE2, promoted to a direct-indexed dense table only once a row has
  enough distinct successors to pay for it. Evicted rows return their storage
  to size-classed free lists.
- **Prefetching the sequence banks** at the top of each byte's iteration:
  the packed contexts depend only on history, which is known before the
  basal levels are evaluated, so those bucket loads overlap the work that
  precedes them.
- **Packing** drops everything only development needs — spare capacity,
  nurseries, insertion-order queues, free lists — and lays each level out
  contiguously.

## Multi-column wake/sleep, now ported

Everything the `phase122`/`phase126` harness needs is now native:

- **`consolidate_delta`** (`alm/consolidate.hpp`) -- merges a wake column's
  anatomy past its shared ancestor into a growing master, using the same
  `begin_lineage_tracking()` touched-key sets as Python so the merge stays
  cheap on a body with millions of paths.
- **Native checkpoints** (`alm/serialize.hpp`, `to_bytes`/`from_bytes`) --
  same-machine binary format, not pickle-compatible by design (no per-object
  Python overhead to reproduce), but `NativeGrowableCortex.__reduce__`
  makes it pickle-transparent, so it drops into existing
  `_atomic_pickle`/`_load_cortex` call sites unchanged.
- **`consolidate_to_byte_budget`** (`alm/sleep.hpp`) -- sheds nursery
  anatomy, then prunes the weakest sequence paths (ranked strongest-target
  support, then total, then depth -- Python's exact tuple order) until the
  body fits. The byte budget itself is measured against the *native*
  serialized size, not a Python pickle byte count -- the two are different
  units, so a native body at a given cap holds more learned paths than a
  Python body at the same cap. That is a real, worth-knowing consequence of
  changing backend, not a bug.
- **Python -> native import** (`native_cortex_import.py`) -- converts an
  existing pickled `SequenceCortexOpponentLearner` (including the
  adaptive-competition trust/TD-credit/attribution state) into the native
  format once, so a lineage that started under the pure-Python cortex can
  continue natively without losing anything it had learned.

Verified bit-exact against the pure-Python reference, including on a real
production checkpoint with 1.27M learned sequences and 773 learned trust
values (`tests/test_native_cortex_matches_python.py`,
`tests/test_native_cortex_consolidate_and_trust.py`,
`tests/test_native_cortex_sleep.py`).

## Wired into production

`dendritic_basin_phase126_three_epoch_generation.py` takes a `--native` flag:
it converts an existing Python checkpoint once (`_ensure_native_epoch1_checkpoint`)
and runs every subsequent epoch through `NativeGrowableCortex` via
`_load_any_cortex`, using the same `consolidate_delta`/
`consolidate_to_byte_budget`/pickling call sites as the pure-Python path.
`dendritic_basin_phase122_full_corpus_4c.py` still constructs
`SequenceCortexOpponentLearner` directly -- not yet threaded there.

# alm_vision — C++ backend for the growing visual-assembly organism

A second, separate port (`vision.hpp`, `vision_c_api.h`, a **separate**
`alm_vision` DLL target -- see "Why a separate DLL" below) of
`continual_vision.py`'s `ALM64MB4CGrowingVision`: the sparse, growing
visual-assembly system behind the CLEAR-10 continual-vision benchmark.
Image decode/resize (JPEG/PNG, PIL bilinear resize) stays in Python; this
module starts from the already-resized `[image_size, image_size, 3]` float
array and owns the receptor math, per-assembly activation/voting, growth,
local weight updates, relation-sketch bookkeeping, and the sleep/merge
lifecycle across columns.

### Correctness contract -- NOT bit-exact, and not "functionally identical
### decisions" either, past a short horizon

Unlike the byte cortex, this is disclosed as numerically-close rather than
bit-exact: PyTorch's internal reduction order and its `topk` tie-breaking
under near-equal values are not a portable target to reproduce from a
from-scratch implementation. Building the correctness test for this port
surfaced something worth being explicit about, because it is easy to assume
"numerically close" implies "same decisions": it does not, beyond a short
horizon.

Growth selects the 128 strongest-`|evidence|` synapses out of ~2,880
dimensions by top-k; routing selects the top-16 assemblies by activation,
also top-k. Both are discontinuous functions of their input -- whenever two
candidates are close (common on smooth/low-frequency image regions, less
common on real photographic texture), a ~1e-7 float32 rounding difference
between PyTorch's reduction order and this port's can flip which candidate
wins the cut. That single flip changes what gets **stored** (a different
synapse selected at birth, a different row learned) rather than just that
step's output, so it compounds forward for the rest of the run. This was
confirmed directly (see `tests/test_native_vision_matches_python.py`'s
module docstring for the full trace): with `evidence()` computing correctly
(matching the Python reference to the float32 noise floor, ~1e-7/1e-8, on
every step), per-step grow-vs-update decisions could still diverge later in
a stream -- and the direction was not one-sided across 15 seeds at multiple
texture levels (native was not consistently better or worse), which rules
out a residual directional bug and points to genuine chaotic sensitivity
inherent to any non-bit-exact reimplementation of a top-k-routed growing
system.

The bar actually enforced, then: `evidence()` matches the reference
tightly and deterministically (this must hold, and does); aggregate quality
(mean classification accuracy over many independent streams, plus growth
staying in the same ballpark) tracks the reference within a wide-but-
meaningful band. Per-step lockstep tracking is not claimed and is not a
valid way to judge a bug report against this module -- see the test file
for what is and isn't asserted.

### A real, fixed bug along the way

`evidence()` concatenates the receptor's own unit-norm output (`basal`)
with the slow apical trace and L2-normalizes the *combined* vector a second
time -- but the trace update afterward has to use the original,
once-normalized `basal`, not the twice-normalized combined vector's basal
slice (`continual_vision.py`'s `evidence()` keeps `basal` in a separate
Python variable for exactly this reason). The first port of this method
reused one buffer for both, so the trace update silently read the
rescaled-by-the-second-norm values. Fixed by saving `basal` to a separate
buffer before the second normalization overwrites it in place.

### Performance: AVX2 gather for the O(live_assemblies) hot path

`activations()` -- computed for every live assembly on every `predict`/
`observe_external` call -- is O(live_assemblies * synapses), and
live_assemblies grows toward `maximum_visual_assemblies` (tens of
thousands) over a long run. Measured directly on a real CLEAR-10 run: with
the original scalar double-precision loop, end-to-end throughput fell from
~950 samples/s near-empty to ~110-150 samples/s as the population grew
during a single pass -- no faster than the pure-PyTorch reference at that
point, because the gather-multiply-accumulate cost had come to dominate
over JPEG decode (which is the same fixed cost for both backends, done in
Python either way).

Rewrote `activations()` to use AVX2 gather (`_mm256_i32gather_ps`) +
float32 accumulation instead of a scalar per-synapse loop with double-
precision accumulation. This module's correctness contract is already
numerically-close, not bit-exact (see above), so the reassociation this
introduces stays within the float32 noise floor already measured by
`tests/test_native_vision_matches_python.py` -- verified by rerunning that
suite after the change. Measured effect: at a realistic population
(~1,000 assemblies grown from real CLEAR-10 images), native is 2.8x faster
than the PyTorch reference end to end (807 vs 288 samples/s), and
degrades much more gracefully at scale (1,118 -> 513 samples/s from
live=1,003 to live=13,766, a ~2.2x drop across a ~13.7x population
increase, rather than collapsing to parity as the old scalar loop did).

This is AVX2-only (`alm_vision`'s CMake target adds `/arch:AVX2` /
`-mavx2`, unlike `alm_native`, which stays SSE2-baseline-portable for its
bit-exact contract) -- fine for the development machine, but not portable
to pre-Haswell/pre-Excavator x86_64 hardware without a scalar fallback
path being selected at build time (the scalar loop still exists in the
source, gated on `#if defined(__AVX2__)`, so building without `/arch:AVX2`
still works, just without the speedup).

### Why a separate DLL

`alm_vision` builds as its own CMake target/DLL, not part of `alm_native`:
the byte-cortex DLL stays loaded inside long-running processes (e.g. a
multi-hour production run), and Windows locks a DLL file while any process
has it loaded. Building both modules into one target would fail to link --
or force killing a live run -- every time either module changed.

### Not yet done

- Not wired into `dendritic_basin_continual_vision_benchmark.py` (the actual
  CLEAR-10 harness) -- still a standalone-tested module.
- No throughput benchmark yet against the PyTorch CPU/CUDA reference (the
  original motivating question -- "is it faster" -- is answered for the
  byte cortex but not yet for vision).
- Not validated against real photographic images (CLEAR-10 stream) end to
  end, only against synthetic class-correlated textured images -- the
  chaotic-divergence analysis above suggests real photos (richer,
  less-degenerate gradient statistics) should have fewer near-tie flips
  than the synthetic test data, but this has not been measured.
