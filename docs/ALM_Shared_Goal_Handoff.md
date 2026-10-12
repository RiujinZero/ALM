# ALM shared-goal research handoff

Date: 2026-09-15

Branch: `ManifoldFlow-ALM`

Latest preregistration commit: convergence runs on the co-adapt base (see the co-adapt campaign section below)

Latest completed-result commit: the commit containing this handoff update

## Objective

The missing mechanism is not another estimator of locally likely text. It is a
conversation-scale representation of what the interaction is trying to
accomplish, causally coupled to generation. The current leading architecture
factorizes this into:

1. a selective evidence/obligation ledger that retains facts, corrections,
   commitments, and unfinished work across turns;
2. a stochastic receding-horizon future-vector organ that preserves alternate
   trajectories rather than averaging them into one conditional mean; and
3. a directly coupled decoder that expresses the selected near-turn vector and
   tracks what remains to be expressed within the turn.

Free generation is the primary outcome. BPB, latent MSE, posterior coverage,
oracle reconstruction, and vector discrimination are diagnostics and cannot
promote a mechanism whose generated interaction does not improve. The intended
learning path uses local/surprise rules and no backpropagation or BPTT. Existing
PCA and ridge components are acknowledged batch estimators, so the present
assays do not establish a fully online-local system.

## What is established

- The v1 orchestrator improves plausibility but is not goal-directed. It better
  answers the same local-statistics question as its organs.
- The future-turn toy demonstrates a complete causal encode, predict, refine,
  and decode loop without backpropagation. Oracle vector decoding is exact and
  vector swaps causally control output. A full-history one-turn baseline still
  wins, and the farther horizon adds no benefit in that toy.
- The native receiver-curve work demonstrates that predicted turn vectors can
  causally control free generation. Longer decoder exposure did not explain the
  earlier null; the dominant issue was predicted-code versus oracle-code
  distribution shift.
- Surprise learning can grow a diverse population of useful future candidates.
  Hindsight best-mode coverage is strong, but unselected mixtures average
  incompatible futures and degrade the emitted point.
- A local categorical selector can reduce roughly 29 active futures to about
  7.5 while preserving useful posterior mass and recall. It is robust on the
  original prefix but becomes statistically competitive with global mode
  frequency after interruption.
- A decay-0.98 slow aggregate is a controlled null. Concatenated and nested
  additive-logit variants do not add distinct interruption-robust information
  beyond fast context and matched controls.
- A surprise-created, nondecaying event ledger is a partial positive. It
  strictly beats a current-turn-only control after interruption, and the
  comparison reverses on the original prefix. Persistence is therefore doing
  context-specific work rather than merely adding capacity. It does not yet
  beat fast context, global frequency, shuffled successors, or basal latent
  prediction decisively.
- Clustering the retained-ledger shortlist into shared-core families (Ward
  geometry, at most two families, up to eight edge vectors) finds a dominant
  family almost every time but does not shrink the edge set inside it (6.80
  vs. flat top-k's 6.79) or reduce pairwise competition (20.75 vs. 20.66);
  geometry-vs-random is directional but its CI crosses zero. The gate fails.
  This is a routing null, not a shared-core-generation null: it clustered
  completed vectors and never built `z_m = core + edge_m`. Full state:
  `runs/shared_goal/future_vector_family_v1_preregistration.json` /
  `future_vector_family_v1_dev.json`.
- The cheapest possible `z_m = core + edge_m` split -- core is a plain
  mass-weighted mean of the flat top-k shortlist, no learned parameters --
  **passes** its causal-specificity gate. Substituting a Sattolo-deranged
  foreign pair's core (holding this pair's own edge sum fixed) scores reliably
  worse against this pair's true target than its own core, on both MSE (95%
  CI 0.077 to 0.158) and proper score (95% CI -0.163 to -0.081), n=28,
  long-interruption evaluation. Within-shortlist reconstruction is exact by
  construction for a mean split and was never treated as evidence -- only
  cross-pair specificity was tested. Full state:
  `runs/shared_goal/future_vector_core_edge_v1_preregistration.json` /
  `future_vector_core_edge_v1_dev.json`.
- The first persistent-core estimator (context-affinity-gated EMA slots) fit
  only on selection-partition cores and evaluated on held-out evaluation
  pairs **fails** its noninferiority gate: proper-bits contrast +0.0488 (95%
  CI +0.0133 to +0.0838, past the +0.05 margin), with a diagnostic
  one-slot collapse on the selection partition (bandwidth 1.0, min affinity
  0.50 covered every context; all 16 slots unused). Full state:
  `runs/shared_goal/future_vector_persistent_core_v1_preregistration.json` /
  `future_vector_persistent_core_v1_dev.json`.
- The preregistered knob-sweep rescue of that failure also **fails**, and
  its cross-validation table separates the two possible explanations:
  contexts ARE separable (a mid-radius config holds six slots at 10% global
  fallback, another thirteen at 40%), but slots do NOT transfer -- every
  config with genuine slot usage is strictly worse than the global mean,
  and the mechanically selected "best" config only wins by falling back to
  the global core on essentially every row. The chosen configuration's
  final gate on the held-out evaluation partition reproduces the v1 failure
  almost exactly (proper bits +0.0485, 95% CI +0.0131 to +0.0835). The
  affinity-slot persistence family is concluded not to generalize; the
  persistence line stops per preregistration. Full state:
  `runs/shared_goal/future_vector_persistent_core_sweep_v1_preregistration.json` /
  `future_vector_persistent_core_sweep_v1_dev.json` /
  `future_vector_persistent_core_v2_dev.json`.
- The core/edge representation has now been read in **actual free
  generation** on the frozen native 500 MB cortex through its existing
  codec receiver, and the read settles the receiver-vs-stop decision. The
  core-conditioned arm matches references better than native on original
  prefixes (+0.0298 sampled margin, 95% CI +0.0070 to +0.0526) and is
  causally specific (foreign core minus core +0.0547, CI +0.0303 to
  +0.0784), but the effect is a register/language nudge ("Danke!",
  "Sure! I'd be happy to help"), not content. Three hard limits: the
  16-feature quantizer collapses core, mixture point, and swapped edge
  into byte-identical generations on all 116 sides, so the core/edge
  distinction does not exist at the actuator interface; every arm,
  including the oracle-code ceiling, produces semantically incoherent
  word salad on these long conversational contexts; and after the
  interruption even the nudge disappears (core minus foreign +0.0007,
  null). The oracle arm reproduces the codec native500 oracle outputs
  byte-for-byte on all 232 shared sides, so the null is not an
  implementation defect. Full state:
  `runs/shared_goal/future_core_edge_native_generation_probe_v1_preregistration.json` /
  `future_core_edge_native_generation_probe_v1_dev.json` /
  `future_core_edge_native_generation_probe_v1_dev.digest.txt`.

## Latest assay: core/edge free-generation probe (qualitative read)

The receiver-vs-stop decision was put to a generation-level read before any
receiver investment. The locked implementation is in:

- `src/manifoldflow/experiments/dendritic_basin_future_core_edge_native_generation.py`
- `runs/shared_goal/future_core_edge_native_generation_probe_v1_preregistration.json`
- `runs/shared_goal/future_core_edge_native_generation_probe_v1_dev.json`
- `runs/shared_goal/future_core_edge_native_generation_probe_v1_dev.digest.txt`

Both ends reuse locked machinery unchanged: the latent side is exactly the
core/edge chain (ledger shortlist, flat top-k, mass-weighted mean core,
probability-weighted edge sum); the actuator is exactly the codec native500
receiver (same ContextCoupledByteWriter, same one matched fit pass on oracle
turn codes, same seeds), which has already demonstrated causal code control.
Seven arms per side: native, local decoder (no code), oracle code (leakage
ceiling), mixture point (core plus this side's weighted edge sum), core
alone, foreign core (Sattolo), and edge-swapped (own core plus the paired
side's edge sum). Full generated text of every arm is dumped for every pair,
both context modes; no pair is omitted.

Three findings, in order of decisiveness:

1. **The actuator cannot see the edge.** Quantizing core, core plus this
   side's edge sum, and core plus the paired side's edge sum yields the
   same sparse code on every side, so those three arms produce
   byte-identical output on all 116 sides (contrast exactly 0.0000). The
   disputed content the whole core/edge split was built to express is
   discarded by the 16-feature magnitude quantization. Any future receiver
   that is supposed to express edges needs a wider or different interface
   first.
2. **The oracle ceiling is incoherent on this base.** Even the arm
   supplied with the held-out answer's own code generates word salad on
   these long LMSYS contexts. Since the base cannot articulate the answer
   even when handed its code, no receiver trained on cores can produce
   thoughtful output at this interface -- the same conclusion the
   reranker line reached from the other direction.
3. **The core's real effect is a register nudge, visible only on original
   prefixes.** Core-conditioned generation beats native on the paired
   4-gram reference margin (+0.0298 evaluation, CI above zero; +0.0382
   selection) and is causally specific (foreign core reliably worse,
   +0.0547 all pairs). Qualitatively the core opens in the right language
   or assistant register more often than native. After the interruption
   every contrast nulls (+0.0007 core minus foreign), matching the latent
   result that interrupted contexts retain little specific core signal.

Strict conclusion: do not train a receiver on this representation at this
interface on this frozen base. The line's remaining asset is the narrow
latent result (the per-episode mean core carries cross-pair specific
signal); making it matter for generation would require, at minimum, a base
that generates coherently on these contexts and an actuator that preserves
the content the core/edge split separates. That is a different project
than a receiver on the current 500 MB checkpoint.

## Latest assay: persistent-core knob sweep (the rescue ran and failed)

The one-slot collapse localized the v1 persistent-core failure to an affinity
radius too wide to separate episodes; the runbook prescribed a narrower
bandwidth / higher affinity floor as a separately preregistered sweep before
concluding persistence itself doesn't generalize. The locked implementation
is in:

- `src/manifoldflow/experiments/dendritic_basin_future_persistent_core_sweep_ablation.py`
- `runs/shared_goal/future_vector_persistent_core_sweep_v1_preregistration.json`
- `runs/shared_goal/future_vector_persistent_core_sweep_v1_dev.json`

The grid crosses bandwidths {1.0, 0.5, 0.25} with affinity floors
{0.50, 0.75, 0.90} -- effective eligibility radii from 1.177 down to 0.115
RMS -- with every other estimator knob frozen at v1 values and the v1 anchor
as config 0. The evaluation partition was never touched during selection:
configs were ranked by a deterministic pair-parity cross-validation inside
the 30-pair selection partition (15 pairs fit, 15 held out, whole pairs on
one side of the split), and only the chosen configuration was then scored
once on the held-out 28-pair evaluation partition by the unchanged locked
assay (`future_vector_persistent_core_v2_dev.json`, same margins, same seed
scheme as v1).

CV results on the long-interruption primary mode (paired persistent-minus-own
proper-bits contrast, 15 held-out pairs):

| Config | bandwidth / affinity | Slots | Global fallback | bits mean [95% CI] |
| --- | --- | ---: | ---: | --- |
| 0 (v1 anchor) | 1.0 / 0.50 | 1 | 0% | +0.0403 [+0.0098, +0.0710] |
| 1 | 1.0 / 0.75 | 6 | 10% | +0.0443 [+0.0144, +0.0744] |
| 3 | 0.5 / 0.50 | 13 | 40% | +0.0520 [+0.0192, +0.0873] |
| 5 (chosen) | 0.5 / 0.90 | 16 | 100% | +0.0387 [+0.0097, +0.0690] |
| 6-8 | 0.25 / any; 0.5 / 0.90 | 16 | 97-100% | +0.0387 (global fallback) |

Read this table as two separable verdicts. Contexts are separable: configs 1
and 3 keep six and thirteen slots alive with only 10-40% global fallback.
Slots do not transfer: both of those configs are strictly worse than the
global mean (+0.0443, +0.0520), and every interval in the table sits entirely
above zero -- even the global mean is reliably worse than per-episode own
cores on held-out pairs. The mechanical selection rule (minimum mean CV
proper-bits contrast) chose config 5, which "wins" only by predicting the
global core for 100% of eval rows -- the exact case the preregistration's
fallback caveat flagged in advance.

The final gate on the held-out evaluation partition failed with nearly the
same numbers as v1: MSE +0.0368 (95% CI +0.0033 to +0.0694; noninferiority
passes) but proper bits +0.0485 (95% CI +0.0131 to +0.0835; past the +0.05
margin), versus v1's +0.0488 (+0.0133 to +0.0838). The full 60-row fit used
16 slots with 42 recycles, so this is not a residual one-slot artifact.

Strict conclusion, per the locked interpretation: the affinity-slot
persistence family does not generalize after the prescribed rescue, and the
persistence line stops here. No further knob sweeps without a new mechanism
hypothesis. What remains established is narrower: the per-episode mean core
carries real context-specific signal (its cross-pair gate passed), but no
tested way of making that core persistent across episodes survives contact
with held-out episodes.

## Prior assay: core/edge context-specificity (cheap core passes)

Since the family result below localized the problem to "the choice just moved
inside the family" rather than indicting shared-core generation, and since the
family, categorical-selector, two-timescale-selector, nested-slow-selector,
and selective-ledger narrowing methods were all statistically indistinguishable
from plain flat top-k, narrowing was fixed to flat top-k (cheapest of the
tested methods) rather than re-litigated. This assay tests the cheapest
possible `z_m = core + edge_m` split before building anything more expensive.
The locked implementation is in:

- `src/manifoldflow/dendritic_basin/future_vector_core_edge.py`
- `src/manifoldflow/experiments/dendritic_basin_future_core_edge_ablation.py`
- `runs/shared_goal/future_vector_core_edge_v1_preregistration.json`
- `runs/shared_goal/future_vector_core_edge_v1_dev.json`

Core is the posterior-mass-weighted mean of the flat top-k shortlist; edge_m
is each retained candidate's residual from that mean. Within-shortlist
reconstruction (`core + edge_m == z_m`) is exact by construction for any mean
subtraction -- true of any vectors and any subtracted constant, so it is
**not** tested as a claim. The one genuinely falsifiable question at this
operationalization: is the resulting core context/episode-specific? For every
development pair/side, a Sattolo derangement (no fixed points) assigns a
different pair/side's core; that foreign core, combined with this pair's own
probability-weighted edge sum, is scored as a point prediction against this
pair's **true development target** -- never a candidate's own vector, which
would again be tautological. Own-core's point estimate is asserted equal
(1e-9 tolerance) to the flat-top-k point estimate the family assay already
computes, as an implementation cross-check; a mismatch raises and no result is
produced.

On the same 28-pair long-interruption evaluation partition used throughout
this chain:

| Metric | Own core | Foreign core | Contrast (own advantage) | 95% CI |
| --- | ---: | ---: | ---: | --- |
| MSE | -- | -- | own lower by 0.1166 | 0.0772 to 0.1575 |
| Proper bits/coordinate | 2.0266 | -- | own lower by 0.1222 | -0.1632 to -0.0810 |

Both components pass: the MSE contrast's CI is entirely above zero and the
proper-score contrast's CI is entirely below zero. Even the cheapest possible
core estimator -- one weighted average per episode, no learned parameters, no
training loop -- carries real, statistically significant, context-specific
predictive signal against the true target. It is not a generic offset
interchangeable across episodes. This clears the bar the family result never
reached.

Read this narrowly: it licenses building a learned/persistent core estimator
and training a receiver with causal controls (the recommendation below,
carried over from before this assay). It does **not** validate the mean
estimator for production use, and it runs no native checkpoint and decodes no
bytes.

## Prior assay: shared-core families and edge competition

The user proposed narrowing futures to vectors that share the most in common,
then competing on their edges. The locked implementation is in:

- `src/manifoldflow/dendritic_basin/future_vector_families.py`
- `src/manifoldflow/experiments/dendritic_basin_future_family_ablation.py`
- `runs/shared_goal/future_vector_family_v1_preregistration.json`
- `runs/shared_goal/future_vector_family_v1_dev.json`

The assay re-fits the unchanged retained-ledger distribution, standardizes
candidate vectors by training-only basal residual variance, and performs
deterministic Ward-centroid agglomeration to four families. It selects the
smallest family set reaching 0.80 probability mass, capped at two. It then
keeps the original candidate vectors within those families, ranked to 0.80
mass and capped at eight. It does not decode an averaged centroid. Flat top-k
and random size-matched families use exactly the same candidate probabilities.

On the 28-pair long-interruption evaluation:

| Arm | Selected families | Edge vectors | Posterior mass | Top-mode recall | Pairwise competitions | Proper bits/coordinate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat top-k | 6.79 singleton equivalents | 6.79 | 80.50% | 83.93% | 20.66 | 2.017760 |
| Geometric families | 1.02 | 6.80 | 80.83% | 83.93% | 20.75 | 2.017709 |
| Random size-matched families | 1.25 | 7.23 | 77.44% | 80.36% | 22.20 | 2.019228 |

Geometry preserves flat coverage: geometric-minus-flat posterior mass is
`+0.00325` with 95% CI `-0.00006 to +0.00980`, and recall is exactly tied.
However, it does not reduce the remaining edge-vector count or pairwise
competition. Geometric-minus-random mass is `+0.03387`, but its 95% CI
`-0.01624 to +0.08029` crosses zero; proper-score improvement is also
unresolved. The preregistered gate fails and this version does not advance to
an edge-competition sandbox.

The result localizes the issue. Identifying a dominant family is easy: nearly
all cases select one. But the dominant family still contains essentially the
same candidates as flat top-k, so the expensive choice has merely moved inside
the family. The shared-core fraction is only 0.101 after interruption, versus
0.525 on original prefixes. Raw latent Ward geometry does not isolate enough
interruption-robust common content.

## The co-adapt campaign: corpus v1, fresh base, and the convergence runs

The core/edge line's verdict (revive only above a coherent generation floor)
led to a three-part campaign: build a clean corpus, retrain the base on it,
and test the goal channel by co-adaptation instead of frozen-base bolting.

### Corpus v1 and the fresh base

- **Corpus** (`runs/coadapt_corpus_v1/`, build report with SHA-256s):
  de-duplicated (normalized-case 64-bit hashes, unique-only quotas, no
  cycling), de-boilerplated (global turn-prefix cap 16 on USER and
  ASSISTANT turns), fine-grained 256 KB interleaving of wiki + SmolTalk-CoT
  + lmsys + SmolTalk preference + everyday. 7.6 GB train + 20 MB valid,
  2.19M unique records. The everyday source was 99% template repeats
  (2,215 lines -> 27 unique conversations) -- the de-boilerplating working
  as intended and the old builder's cycling having masked it.
- **Infra findings**: the Google Drive streaming mount cannot sustain
  multi-GB sequential reads (three failed builds at 5.9/11.7 GB, then
  read-only), and its cache lives on a full C: drive -- the root cause of
  every ENOSPC. The corpus was built natively on Windows and written over
  UNC into WSL (71 MB/s). D: writes are unusable. The
  `scripts/stream_copy_slice.py` copier and the builder's self-healing
  reader exist for future sources.
- **Fresh base** (`runs/coadapt_base_500mb_v1`): the validated phase156
  configuration on the new corpus, --target-epochs 147.
- **Crash and fix**: training crashed at epoch 36, reproduced on resume --
  `alm_cortex_load` could not deserialize the epoch-35 native checkpoint.
  Root cause (user fix): `RowArena::increment` registered phantom
  zero-count keys once a narrow row's total saturated, so serialized rows
  declared more cells than `write_row` emitted, corrupting every following
  section. Fix: early-return on saturation plus a loader compatibility
  break for already-corrupt files; `cpp/tests/row_arena_test.cpp` locks the
  behavior. Epoch-35 loads post-fix; training resumed and is healthy.
- **Epoch-16 probe** (the same core/edge probe on the fresh base): core
  minus native on long-interruption +0.0414 (95% CI +0.0094 to +0.0751),
  significant -- the old corpus never produced this. The chat register
  shows SmolTalk-CoT influence and multi-turn persistence is emerging, but
  text is still not coherent multi-sentence content.

### Run 2: receiver-path convergence 2x2 -- gate fails

The preregistered convergence 2x2 (linear organ + sparse-code receiver),
initialized from the epoch-16 checkpoint after the serialize bug (prereg
amendment recorded before launch). The trainer completed all 8 passes with
all 7 marks captured; the report write died without a traceback (likely
OOM from the ~3.5 GB native-bases dict) -- snapshots intact, missing marks
evaluated directly.

Verdict at the 8-pass mark (long-interruption, evaluation n=28): D
arm-minus-native +0.0389 (not significant); D vs A +0.0275 (not
significant); ledger-by-horizon interaction negative (−0.008); lesion
D_reset minus D = 0.0000 exactly (the quantizer maps reset variants to
identical codes -- uninformative); shuffled specificity directional only;
oracle attenuated (+0.0093, the receiver's train/predict shift). D was the
most consistent riser (0.002 -> 0.019 -> 0.047 -> 0.039) but nothing
separates from noise. Strict conclusion per the preregistered rule: the
receiver-path mechanism family is recorded as a null on this base. The
binding constraints are the known interface defects -- quantizer collapse
and receiver distribution shift -- not organ capacity.

### Run 3a: veccond v1 -- the inverted interface, linear organ

The inverted interface (`vector_conditioned_base`): the organ's dense
vector primes the base directly as a canonical float32 frame; the observed
response bytes are learned by the base itself (surprise and plasticity in
the base); no sparse quantization, no residual receiver anywhere.

Result: **arm A (K1, no ledger) long-interruption curve rises
monotonically +0.0098 -> +0.0116 -> +0.0206 -> +0.0249 -> +0.0295 ->
+0.0208 -> +0.0408, significant at the 8-pass mark (95% CI +0.0001 to
+0.0837)**. The same base's native (unprimed) generation drifts down
(−0.0066 -> −0.0237) while its primed generation drifts up -- the base
organized around the primed frame. Arm B (K1 + ledger) hit +0.0536
(significant) at 1 pass, then noisier. Original-prefix is null for all
arms. This is the first statistically significant co-adaptation result for
the goal channel.

### Run 3b: veccond v2 -- the dendritic predictor (launched)

The predictor upgraded to a live `SurpriseGrownFutureTransition` grown by
its own surprise rules across the passes: per turn it forecasts a mixture
of candidate futures, the base learns the response bytes under the primed
mixture vector, and the organ resolves the observed code with learning
enabled. One growing organ is shared across arms (its growth is
arm-independent), so the 2x2 is a readout question: ledger arms add the
bounded core to the prime; K4 arms reweight the mixture by far-horizon
agreement with a bounded linear far head. A real finding from the first
launch: the unbounded delta rule diverged to NaN on arm B (~pass 6) --
organ weights are now clipped to ±50 (bounded-tissue rule, prereg
amendment recorded before any v2 evaluation), divergence reported as a
finding rather than tuned away. Comparison line: v1's A-curve (+0.0408 at
8 passes) is the number the dendritic organ must beat.

### Frozen-base byte-head generation ablation (null; interface closed)

A separate frozen-base bolt-on was tested directly on the co-adapt
epoch-147 checkpoint with `dendritic_basin_shared_goal_generation_ablation`:
a matched goal byte head applied likelihood-ratio correction to native
free generation, with native, goal-off, oracle, causal, swapped, prior,
and shuffled-action controls. The β=0 validity gate passed: native and
goal-off were byte-identical, and β=0 correction reproduced native
probabilities exactly.

The 16-pair/32-sample baseline was null under the original byte-ngram
metric: oracle `+0.0093` (p=0.55), causal `+0.0139` (p=0.24), and
shuffled-action `+0.0167` (p=0.26); causal-minus-shuffled was `-0.0028`
(p=0.88) and oracle-minus-shuffled was `-0.0074` (p=0.75). A 64-pair/128-
sample sweep at β={1,2,4,8} reproduced the null at every β. At β=8,
stronger correction degraded output health (oracle valid-UTF8 0.824,
entropy 4.701; shuffled valid-UTF8 0.699, entropy 4.818) without producing
a reference signal.

The result is not a metric artifact. Re-scoring the same bytes with
word-cosine and distinctive-word-capture also produced no significant
condition, and the oracle captured only 0.014 of the correct reference's
distinctive words versus 0.015 for unsteered native. References were
well-separated (shared/min ≈ 0.125), but every condition produced
incoherent word salad. The causal capture was misaligned (capA 0.006,
capB 0.016). This closes the frozen-base byte-head branch: it is consistent
with the earlier receiver-path and core/edge oracle-ceiling findings. A
frozen base that cannot articulate the answer even when handed its code
cannot be rescued by a denser likelihood-ratio head or by increasing β.

### veccond4 240-pass campaign status

The inverted-interface campaign continued as veccond4 (dendritic
predictor, k_inputs_per_branch=8) for 240 passes. Snapshots exist through
mark 3,256,800 (=240 × 13,570 updates), including all four arm bases and
the shared organ. The WSL restart interrupted the final report write, so
`convergence_veccond4_240pass.json` is absent, but the endpoint snapshots
are intact. Evaluations completed at marks 108,560 (8 passes) and 217,120
(16 passes); the remaining marks 434,240 (32), 868,480 (64), 1,736,960
(128), and 3,256,800 (240) are being evaluated.

At 8/16 passes, long-interruption arm-minus-native for A (K1, no ledger)
was `+0.0451 [-0.0042, +0.0960]` and `+0.0340 [-0.0145, +0.0823]`,
respectively (not significant); the 16-pass original-prefix arm was
`+0.0375 [0.0142, 0.0605]`, with arm-minus-shuffled `+0.0306 [0.0001,
0.0602]`. At 32/64 passes the signal regressed to null/negative, and the
240-pass endpoint completed with no surviving effect: A arm-minus-native was
`-0.0137 [-0.0480, +0.0204]` on long interruption and `-0.0149 [-0.0589,
+0.0308]` on original prefixes; oracle-minus-native was `-0.0030 [-0.0488,
+0.0429]` and `-0.0014 [-0.0429, +0.0434]`, respectively. K1 and K4 arms
were effectively identical (cross-arm D-minus-A `+0.0011 [-0.0433, +0.0462]`
long interruption; `-0.0002 [-0.0410, +0.0370]` original prefixes). The
endpoint does not beat v1's A-curve (+0.0408 at 8 passes) and does not reach
coherent multi-sentence content. The dendritic predictor's extra machinery
did not rescue the inverted interface; the remaining failure locus is the
stateful decoder/transducer path. A bounded endpoint run (4 dev pairs,
horizon 32, stateful mode) completed at mark 3,256,800; the stateful adapter
retained associative memory across turns, but no stateful control produced a
positive reference-discrimination margin (`arm_stateful_minus_arm_context`
≈ 0.0, `arm_stateful_clear_minus_arm_stateful` ≈ -0.006, `oracle_stateful_minus_oracle_context`
≈ -0.012). The multi-turn session API (`StatefulVectorConditionedTurn`,
`generate_stateful_multi_turn`) is implemented and tested; the evaluator now
includes causal context-matched controls (`arm_context`, `oracle_context`),
reset (`arm_stateful_clear`), no-context-adapt (`arm_stateful_no_context_adapt`),
no-generated-adapt (`arm_stateful_no_generated_adapt`), and no-updates
(`arm_stateful_no_updates`) conditions. The architecture preserves local
surprise-gated Hebbian updates with no backpropagation.

### Interface comparison so far

At the 8-pass mark, long-interruption: receiver path with the full D
mechanism +0.0389 (not significant) versus the inverted interface with the
simplest linear organ +0.0408 (significant). The inverted interface wins
with less machinery, confirming that the known receiver defects -- the
16-feature quantizer's collapse and the train/predict distribution shift
-- were the binding constraint all along.

## Current interpretation

The campaign changed the picture. The clean corpus plus co-adaptation
produced the first significant generation-level signal the goal channel has
ever had (veccond v1 arm A at 8 passes), the receiver path failed its gate
with its known interface defects now experimentally confirmed as binding
(quantizer collapse, distribution shift), and the dendritic-predictor
upgrade is the live test of whether predictor capacity converts that
signal into content.

The surviving architecture statement: the priming channel works when the
base gets to practice with it; the sparse-quantized receiver is the wrong
actuator; the open question is predictor capacity (linear vs dendritic)
and whether the readout can reach coherent multi-sentence content as the
base consolidates toward epoch 147.

Do not read the family result as evidence against shared-core generation. That
assay only clustered completed latent vectors and retained their original
members; it did not create a generative core/edge factorization. It tested
geometric routing, not the production proposal. The core/edge assay above
tested the production proposal's cheapest possible version and passed its
specificity gate -- but only that narrow claim.

The persistence question the core/edge result opened is now closed, and
negatively. Both the first affinity-slot estimator and the preregistered
narrowing sweep failed their gates; the sweep's cross-validation table shows
contexts can be separated into slots but slot cores do not transfer to held-out
episodes -- they are reliably worse than the global mean, and the only
"winning" configurations were the global mean in disguise.

The receiver question is now also closed, and negatively, by direct
generation evidence. Driving the frozen 500 MB base's existing codec receiver
with core vectors produces a real but tiny register nudge on original
prefixes and nothing after interruption; the quantizer discards the edge
entirely; and the oracle-code ceiling itself generates word salad on these
contexts, so no receiver on this base could produce thoughtful output. The
surviving positive claim is exactly the original narrow one: the per-episode
mass-weighted mean core carries context-specific latent signal. No tested
mechanism makes it persistent, and no tested interface expresses it in
free generation.

## Recommended next experiment

None within this line on the current base and interface. Every escalation
rung has now failed its own gate at the appropriate scope: family routing
(latent null), persistence (two preregistered failures), and generation
(qualitative probe with a byte-exact oracle cross-check). The three-part
ledger / future / decoder architecture remains the leading candidate in the
abstract, but each of its future-component claims is now bounded by results:

- If the line is revived, the prerequisites are upstream of the core: a
  frozen base that generates coherently on multi-turn conversational
  contexts (the word-salad ceiling binds everything downstream), and an
  actuator interface whose quantization preserves the content the
  core/edge split separates.
- The chain's reusable negatives are also inputs to the broader program:
  per-episode signal is real but does not survive being carried across
  episodes (selector, two-timescale, nested, ledger, family, persistence,
  generation attempts all converged on this), which is direct evidence
  about where conversation-scale goal representation must come from if it
  is to exist at all.

## Integrity and reproduction

- The protected test payload has not been opened and must remain sealed during
  development.
- Latest runs use 2,131 training conversations, 25,162 causal transitions, and
  58 development pairs split into fixed 30-pair selection and 28-pair
  evaluation partitions.
- Canonical WSL workspace: `/home/user/sonic-manifold-flow`
- Canonical Python: `/home/user/gtr_venv/bin/python`
- Family run command:

```bash
PYTHONPATH=src /home/user/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_family_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --report runs/shared_goal/future_vector_family_v1_dev.json
```

  The report and captured stdout are byte-identical with SHA-256
  `28e5555cd56fba3b699300a73b279a137562fd97b008c2f2ff757546c8165d23`.
- Core/edge run command:

```bash
PYTHONPATH=src /home/user/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_core_edge_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --report runs/shared_goal/future_vector_core_edge_v1_dev.json
```

  The report and captured stdout are byte-identical with SHA-256
  `b4d6033a30db1f025ca63ddd35504c82450c408a19d960a6b103b3950d6601fe`. Wall
  time 70.5s, well under the 1800s limit. All locked implementation hashes
  match.
- Persistent-core sweep run command:

```bash
PYTHONPATH=src /home/user/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_persistent_core_sweep_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --report runs/shared_goal/future_vector_persistent_core_sweep_v1_dev.json
```

  The report and captured stdout are byte-identical with SHA-256
  `f857ff8b92dbd2a24c3062285c886011da5a52e2036e855fc5af1162b4060ccf`. Wall
  time 70.4s. An earlier first launch crashed on an unbound name before any
  result was produced; the preregistration's implementation hash was updated
  before any result existed.
- Persistent-core final gate run command (chosen configuration, bandwidth 0.5,
  affinity 0.90, locked assay unchanged):

```bash
PYTHONPATH=src /home/user/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_persistent_core_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --report runs/shared_goal/future_vector_persistent_core_v2_dev.json \
  --persistent-context-bandwidth 0.5 --persistent-min-context-affinity 0.90
```

  The report and captured stdout are byte-identical with SHA-256
  `411732fd6325b117ea55556b0354090652cfac81dcf80fb325e47a9fe075f43a`. Wall
  time 70.8s. Reruns reproduce the report byte-for-byte except
  `wall_seconds`. All locked implementation hashes match.
- Core/edge free-generation probe run command:

```bash
PYTHONPATH=src /home/user/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_core_edge_native_generation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/future_core_edge_native_generation_probe_v1_dev.json \
  --digest runs/shared_goal/future_core_edge_native_generation_probe_v1_dev.digest.txt
```

  The report and captured stdout are byte-identical with SHA-256
  `e9180d5eb7eb89e435cac7fc25709eced8efeedf450569e89cf563ba76b24b3a`; the
  digest is `b5276945d15145c70602ab7ababb0a45f6fbb3053583466ea33bac46d0f4827c`.
  Wall time 603.0s. The probe's oracle arm reproduces the codec native500
  oracle outputs byte-for-byte on all 232 shared sides (16 absent sides are
  exactly the four pairs removed by the newer six-pair exclusion list). An
  earlier launch crashed at the digest write after the full computation; the
  implementation hash was corrected and the run rerun before any result was
  recorded.

All locked implementation hashes match. No result has been pushed.

Untracked user research files and archives in the Windows workspace are not
part of these commits and must be preserved.
