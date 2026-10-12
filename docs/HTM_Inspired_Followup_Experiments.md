# HTM-Inspired Follow-up Experiments

**Status:** proposed, not started, 2026-09-11. Prompted by reading Hawkins &
Ahmad (2016), "Why Neurons Have Thousands of Synapses" (Numenta HTM sequence
memory theory) against ALM's own architecture (`docs/ALM_V1_Architecture.md`).

**Scope note:** none of these touch the currently-running production harness
(`dendritic_basin_phase126_three_epoch_generation.py`) or its live runs. Idea 3
needs no training at all. Ideas 1 and 2 are prototyped and validated at small,
matched-capacity scale — the same discipline `structural_support` and
`connectivity_pruning` were validated under — before any production wiring.

## Idea 3 first: capacity math instead of empirical floor-hunting

**Cheapest, safest, no training required. Do this first, in parallel with
everything else.**

Ahmad & Hawkins (2016, arXiv:1601.00720, the companion math paper to the
sequence-memory theory) derives closed-form error rates for a sparse
coincidence-detection population as a function of population size, sparsity,
and dendritic threshold. That math is generic to any sparse-coincidence
system — it doesn't assume neurons.

- **Build:** a small standalone module (pure math, no training) that maps
  ALM's own structural quantities onto that formula's terms — context-bank
  depth and fan-out as population size, `confirmations`/`sequence_confirmations`
  as the threshold analog, live vs. possible context count as the sparsity
  term — and estimates the unprunable "floor" (fixed anatomy size) for a given
  `max_context_bytes`/`confirmations`/`max_sequence_nursery_paths` combination.
- **Validate:** check its predictions against floors already measured this
  session (e.g., 88.3% and 41% of natural size under different configs, and
  the various `embodied_bytes` readings recorded in
  `tests/test_native_connectivity_pruning.py` and elsewhere). If the formula's
  estimates land within a reasonable margin of those known points, it's usable
  for planning; if not, the mismatch itself is informative about where ALM's
  actual anatomy diverges from the idealized model.
- **Ship as:** a helper (e.g., `estimate_floor_bytes(config)`), used to pick
  test/experiment budgets instead of the trial-and-error probing this session
  did repeatedly (deepcopy-and-clear probes, "prune to an absurd target and
  read the error back").
- **Decision criterion:** does it save real iteration time on the next few
  budget-calibration tasks? If it's within, say, 2x of true floors across a
  handful of configs, it's a net win even if not exact.

## Status update, 2026-09-11: all three implemented; Idea 1 redirected into Idea 2

All three ideas below were implemented and tested this session. Headline
results:

- **Idea 3**: done, validated to <5% relative error against real
  `embodied_bytes` measurements (`capacity_estimate.py`,
  `tests/test_capacity_estimate.py`).
- **Idea 1** (`burst_surprise_signal` on `SequenceCortexOpponentLearner`):
  implemented and unit-tested. As a *prediction-time damper* (its original
  proposed use), it showed no effect on steady-state BPB -- fully explained,
  not just a null result: `burst_fraction` is an EMA that decays toward
  numerical zero (measured: 5.25e-89 after 500K quiet bytes) over any long
  stretch without new structure forming, so by the time a full epoch
  finishes it has long since become a no-op. A follow-up windowed
  measurement right at a real TinyStories-to-Wikipedia corpus switch found
  a real, small, *consistent* effect instead: `burst_on` beat `burst_off`
  in all 50 of 50 measured 20K-byte windows immediately after the switch
  (-0.0036 to -0.0018 bits/byte in the first ten windows, decaying toward
  -0.0001 by the end of the measured range) -- a genuine ~0.1-0.15%
  relative BPB improvement during the transient right after a real
  distribution shift, consistent with the existing confirmation/backoff
  machinery already handling steady-state surprise well on its own (per
  the user's read), while leaving a small gap specifically in the first
  few thousand bytes after a shift.
- **Idea 1, redirected**: rather than keep it as a marginal prediction-time
  damper, the user proposed repurposing the same signal as Idea 2's
  event-capture trigger (see below) -- this turned out to be the more
  productive direction and is where `burst_surprise_signal` is actually
  used going forward.
- **Idea 2 Stage A** (`long_range_bank` on `SequenceCortexOpponentLearner`):
  implemented -- a second context bank keyed on far more trailing bytes
  than the ordinary sequence banks, blended in exactly like they are.
  Confirmed inert by default and confirmed to shift predictions when
  enabled. Gating its growth on `burst_fraction`
  (`long_range_gate_on_burst`) -- using Idea 1's signal as the event
  detector deciding *when* to commit a long-range association, instead of
  growing on every byte indiscriminately -- measurably produces a sparser,
  more selective bank: on a stream mixing repetitive and genuinely novel
  spans, unconditional growth produced 2,052 distinct entries (7,032 total
  growth events) versus 923 entries (1,057 events) gated -- a 55%
  reduction in entries and 85% reduction in growth events, concentrated in
  the actually-novel spans.

  The synthetic "identifying token beyond ordinary bank reach" recall probe
  that initially degenerated (two failed geometries: filler tokens that
  never reached "live" status, then a bridge phrase whose length exactly
  equaled `long_range_context_bytes`, silently dropping the identity byte
  from the trailing window entirely) was fixed on the third attempt with
  explicit geometry assertions
  (`tests/test_dendritic_basin_long_range_bank_recall.py`). **Clean
  positive result:** with the identifying name hidden behind a 24-byte
  bridge phrase identical in both training sentences (so the last 6 bytes
  before the target word are byte-for-byte identical either way), the
  ordinary short-range banks predict the correct word at only 53% (nearly
  chance -- just corpus-level byte-frequency bias, not real
  disambiguation), while the long-range bank alone resolves it at 95%.
  This is the core Idea 2 hypothesis validated on a clean, controlled
  case: long-range identity binding falls out of the *same* proven
  context-matching mechanism, no separate capture/commit/coincidence
  organ needed.

  **But burst-gating this growth has a real, measured cost, found on the
  same probe:** the correct ("Bob"->"blue") key gets reinforced only once
  out of 40 real training repetitions under gating, versus 36 times
  unconditionally -- confidence on the same clean case drops from 95% to
  63%. Root cause, confirmed directly: an EMA novelty signal cannot tell
  "boring because already well-established" apart from "boring because
  nothing is happening" -- both look identical once burst_fraction has
  decayed, so gating silently stops reinforcing a fact the moment it
  stops being novel, exactly when repeated confirmation should be
  building confidence in it.

  **Fixed, and confirmed to get both properties at once.** Refined
  `_grow_long_range` so the gate applies only to *creating* a new
  long-range key; an already-existing key is reinforced on every match
  regardless of current `burst_fraction`. Re-measured on the same two
  probes: recall confidence on the "Bob"->"blue" case recovered to 95.1%
  (from the flawed all-gated version's 63%, matching the unconditional
  95.3% -- the correct key is now reinforced 34 times instead of 1), while
  the sparsity result is unchanged (923 vs. 2,052 entries on the
  mixed-novelty stream, identical to before, since that number only ever
  depended on gating creation). This is the version to actually keep for
  production: selective about what gets remembered, without capping how
  confident the organism can become in something it keeps re-confirming.

**Naming note:** the long-range bank's parameters were initially named
`apical_bank`/`apical_context_bytes`/`apical_strength`, which collided with
an unrelated, pre-existing `apical_strength` parameter already used by
`PhraseEventOpponentLearner` for its own word-level event modulation.
Renamed to `long_range_bank`/`long_range_context_bytes`/
`long_range_strength` before this caused real confusion -- caught by
grepping for the name before it shipped anywhere, not by a failure.

## Idea 1: bursting fraction as a surprise signal

**Current state, confirmed by reading the code (not assumed):**
`adaptive_competition`'s existing signal
(`temporal_difference_credit.py::LocalValueEstimate`) is a per-context,
TD(0)-bootstrapped value estimate of *how much a specific pathway's activity
reduced surprise* — a one-step-then-bootstrapped utility. It is not a
population-level novelty signal. HTM's bursting fraction — what share of
currently active structure was unpredicted, at the population level, not
per-pathway — is a genuinely different, complementary signal, not a
duplicate of what's already there.

- **Define the ALM analog:** of the bytes/contexts processed in a window,
  what fraction landed on brand-new or still-immature (nursery, low
  confirmation count) structure versus already-mature, confirmed structure.
  The raw ingredients already exist as counters (`context_births`,
  `sequence_births`, `sequence_nursery_evictions` are already serialized —
  see `cpp/include/alm/serialize.hpp`), so this is mostly a rollup, not new
  instrumentation.
- **Wire it in:** as an alternate (or additional) input to the Regulatory
  Context State organ's existing basin/trust update, alongside — not
  necessarily replacing — the current TD(0) value signal.
- **Test at small, matched scale:** the same ablation style as the
  `structural_support`/`connectivity_pruning` validation — fixed byte budget,
  matched row/parameter count, both corpora (TinyStories and Wikipedia).
  Reuse the TinyStories→Wikipedia corpus switch already run this session as
  the natural stress test: does a burst-fraction-driven regulatory signal
  adapt faster or produce better BPB at the distribution-shift boundary than
  the current signal alone?
- **Decision criterion:** matched-capacity BPB or adaptation-speed
  improvement, measured honestly against the current mechanism — not "it's
  more principled" as its own justification. `structural_support` looked
  principled too and lost to the simpler baseline at matched capacity; this
  gets held to the same bar.

## Idea 2: apical-style long-range prediction instead of a separate memory organ

**Highest risk, highest potential value — "great if it works," per the ask.**

Important scoping fact: the Episodic Memory Field (`event_capsule.py`,
`recurrent_episodic_field.py`, `relational_access.py`, `recalled_expression.py`)
is **not** part of the currently-running production byte-cortex
(`opponent_byte_stream.py`/`native_cortex.py`) at all — it's a separate
experimental lineage that was never wired into the connectivity_pruning
production runs. So this idea doesn't touch anything live; it's about how to
evolve a parked research line that has a long history of partial/negative
results (see `ALM_V1_Architecture.md`'s phase 76-100+ log: recall arriving one
cortical phase too late, wrong occurrence binding, expression ceilings never
fully resolved after ~20 phases of dedicated engineering).

HTM does not solve long-range prediction with a separate memory subsystem. It
reuses the *same* dendritic-segment matching mechanism basal prediction
already uses, just with a second synapse population (apical) reading a
slower/broader context signal, primeable further ahead than one step. The
proposal: try building ALM's long-range recall the same way — as a second
context bank on the *existing, already-validated* cortex matching mechanism,
not as a new organ with its own capture/commit/coincidence apparatus.

**Staged, fail-fast plan:**

1. **Stage A — cheapest, most diagnostic, no production risk.** Reuse the
   *existing* controlled probe suite from the phase76-100 line (the
   "red ball"/identity-swap/distractor tests, e.g.
   `dendritic_basin_phase81_cortical_expression.py` and siblings) as the
   fixed evaluation harness. Add a minimal second context bank to the
   *production* cortex mechanism, keyed on a much longer trailing context
   (tens of bytes, versus the existing short local banks), blended into
   `predict_next_distribution()` the same way `basal_strength` already blends
   short-context evidence — no capture step, no commit step, no coincidence
   gating, just a second bank with its own blend weight. Test whether this
   alone reproduces any of the same controlled-identity results (e.g., the
   "red ball" continuation) that took ~20 phases of dedicated episodic-field
   engineering to partially achieve.
   - **Go/no-go:** if this recovers even a meaningfully positive fraction of
     the controlled cases the episodic field eventually got working, it's
     worth Stage B. If it reproduces the same failure modes (recall arrives
     too late, wrong content wins), that's evidence the problem is fundamental
     to sparse-byte-context matching at long range, not to the episodic
     field's specific machinery — a genuinely useful negative result either
     way.
2. **Stage B — capacity cost check.** If Stage A shows promise: measure
   held-out BPB impact on TinyStories and Wikipedia at matched capacity (a
   long-context bank costs real memory) — same matched-row-count discipline
   as every other pruning/mechanism validation this session. A win on
   controlled probes that costs too much capacity at matched budget is not
   a win.
3. **Stage C — scope decision.** If both A and B hold up: decide whether to
   deprioritize the separate episodic-field codeline in favor of this
   cortex-integrated approach, keep both as parallel threads, or fold the
   working parts of one into the other. Not a decision to make before A and B
   produce real numbers.

**Decision criterion overall:** Stage A is cheap enough (small controlled
probes, no corpus-scale training) that it should produce a clear go/no-go
signal quickly. Don't proceed to B without a real positive A result — this
mirrors the exact trap `structural_support` fell into (looked promising on
the mechanism's own terms, lost when actually measured at matched capacity).

## Sequencing

Idea 3 (no training) can start immediately and in parallel with anything
else. Idea 1 touches the production `adaptive_competition` mechanism, so its
ablation runs are small/local experiments, never the live 12-hour production
run. Idea 2's Stage A is entirely within the separate, already-parked
episodic-field experiment scripts — nothing here needs the live wiki or
CLEAR-10 runs to stop or change.
