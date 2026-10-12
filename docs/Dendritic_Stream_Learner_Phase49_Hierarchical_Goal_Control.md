# Phase 49: Hierarchical Goal-Conditioned Local Control

Date: 2026-09-07

Status: Isolated goal gate passed; first byte-cortex wiring causally silent

Scope: Dendritic ALM only

## Central Hypothesis

Long-range coherence may require less transmitted detail, not more memory.
Specialist organs can remain local if they share one small persistent state
describing what the organism is currently trying to continue.

```text
global goal state
        ↓
goal-conditioned local prediction/action
        ↓
local outcome, achievement, surprise, novelty
        ↓
update, advance, or dissolve the goal state
        ↺
```

This family can be described as **hierarchical goal-conditioned local
control**. That phrase names the computational intersection; it is not an
established biological algorithm.

## Comparative Motivation

Portia jumping spiders visually inspect alternative detour routes and select a
complete route even when part of the alternative is impassable. The reported
scanning rules progressively channel attention onto the viable path. This is
compatible with maintaining a destination or intermediate constraint while
locally following route structure; it does not prove a symbolic internal plan.

Jumping-spider visual systems also divide labor. Wide-field lateral eyes can
direct the narrow, high-acuity principal eyes toward moving targets. This is a
clean active-sensing example: cheap broad surveillance selects where expensive
high-resolution sampling occurs.

Octopus arm extensions with normal-like kinematics can be evoked after an arm
is disconnected from the brain, demonstrating that substantial motor programs
reside in peripheral circuitry. Anatomical and physiological reviews support
integration of local arm sensation with broad central commands. The central
system need not transmit every actuator state or muscle command.

The octopus vertical lobe provides a separate learning analogy. Connectomics
shows a fan-out learning network in which compact input expands into many
small interneurons and converges again, with local plastic and modulatory
sites. Its resemblance to insect mushroom-body and vertebrate fan-out learning
networks is evidence for convergent circuit organization, not identical
molecular learning rules.

Salticid neuroanatomy identifies prominent higher visual centers and distinct
principal/secondary visual pathways. Current evidence does not establish that
salticid mushroom-body-like structures implement the octopus vertical-lobe
algorithm. That remains a comparative prediction rather than an ALM premise.

## Where the First Goal Comes From

Saying that local organs receive a goal leaves the most important origin
question unanswered. A low-level goal need not begin as a represented future.
It can begin as a condition the organism is organized to restore.

This gives two distinct senses of goal:

- an **implicit goal** is a regulated condition encoded by circuit dynamics;
- an **explicit goal** is a maintained state standing for a selected target.

Synthetic biological feedback circuits demonstrate regulation and disturbance
rejection without reasoning. Homeostatic reinforcement-learning models then
show how outcomes that reduce physiological deviation can supply primary
reward. Neither result implies conscious intention, and neither allows a
neutral learner to discover from nothing what ought to matter. The initial
asymmetry must come from design, development, sensing, physiology,
connectivity, or plasticity. Developmental-network experiments provide a
computational proof of principle that compact generative rules can place useful
structure in initial wiring.

For ALM, minimizing prediction error alone is not a sufficient homeostat: a
repetitive phrase loop can be highly predictable while being behaviorally
useless. The first implementation must declare its primitive regulated
variables explicitly. Modality-neutral candidates include bounded activation,
tissue occupancy and energy, habituation pressure against unchanging loops,
and calibration to the continuing external stream. None should be smuggled in
as a semantic story objective.

## From Restoration to a Provisional Goal

A predictor of restoration can acquire instructional value. In schematic
form:

```text
homeostatic deviation
        -> experienced restoration
        -> predictors acquire restoration value
        -> one reachable predictor wins goal competition
        -> the goal conditions local organs until reached or abandoned
```

Second-order conditioning in Drosophila supplies an empirical stepping stone:
a stable reward-associated memory can recruit dopamine circuitry and teach a
new association. This supports the principle that a useful predictor can
become a teacher. It does not collapse three separate capabilities into one:
valuing a cue, learning an action that reaches it, and actively maintaining it
as the present target.

Phase 41's receiver-specific consequence signals and Phase 48's event-specific
temporal traces make a local ALM version testable. A restored state can
reinforce the eligible event that preceded it; after learning, that event can
reinforce an earlier eligible event. This is local backward chaining through
conditioned restoration, not prospective reasoning.

## Selecting and Structuring the Current Target

A target can be selected by circuit dynamics rather than by a separate
executive. Candidate sparse goal assemblies receive four generic influences:

```text
regulatory need + sensory evidence + learned restoration value + accessibility
```

Mutual inhibition selects a winner and recurrent excitation sustains it.
Progress and arrival maintain or terminate the state. The score above is a
design decomposition, not a proposed universal biological equation.

Fly navigation shows why a single scalar such as hunger is insufficient.
Distinct heading and goal-direction signals converge to compute steering, and
causal stimulation of goal-direction neurons can impose a target direction.
Odor and wind-direction pathways likewise converge to produce persistent
upwind navigation. ALM therefore needs a low-dimensional but *structured*
goal: identity or direction plus urgency/relevance, not merely one global
reward bit.

## Goal-Seeking Is Not Yet Planning

Approaching a learned predictor is goal-seeking. Planning requires an
additional operation: evaluating possible transitions before acting. A clean
capability ladder is:

```text
regulate a condition
    -> learn what predicts restoration
    -> select and maintain an intermediate target
    -> evaluate alternative routes to that target
```

The first Phase 49 gate stops at target maintenance. A later planning gate must
block the familiar route and test whether the cortex immediately selects an
unexperienced viable alternative. Repeating the old action or relearning only
after failure does not pass that gate.

## What Phase 48 Is Missing

The three current organs answer:

1. what byte tends to follow locally;
2. what motif/event is unfolding;
3. which earlier motif identity remains relevant across time.

They do not explicitly answer:

> Which compact future constraint should remain active while local events
> unfold?

The temporal-basis working set preserves up to 64 past identities. A goal
state should instead be one or a few selected future constraints. Memory of
the past and commitment to a future are different computational objects.

## Proposed ALM Organ

Phase 49 adds a very small goal bottleneck above the existing temporal organ:

- `g(t)` is one sparse goal assembly, not a dense context vector;
- it is selected by fan-in competition over regulatory need, temporal routes,
  learned restoration value, local evidence, and accessibility;
- it weakly conditions byte, motif, and temporal-route competition through
  bounded apical gain;
- local organs continue resolving detailed next actions from their own state;
- completed motifs return three generic signals: prediction mismatch,
  novelty, and whether the goal's predicted attractor was reached;
- goal state persists on useful progress, dissolves after sustained failure,
  and advances when its attractor is reached.

A schematic local rule remains:

```text
Delta w_ij = eta * eligibility_ij * modulation(goal, local outcome)
```

The modulator does not encode a gradient. It reports only current goal
identity and low-dimensional relevance/outcome. Every durable update remains
computed at the receiving branch.

## Hierarchical Planning Interpretation

The organ does not construct a detailed action sequence. It learns a chain of
constraints:

```text
ultimate attractor → subgoal 1 → subgoal 2 → subgoal 3 → completion
```

Each subgoal is held while the lower organs repeatedly solve the immediate
prediction problem. This is the ALM analogue of maintaining a destination
while continuously re-solving local movement.

For narrative generation, a goal might eventually correspond to a sparse
learned event attractor such as returning an object, resolving danger, or
reaching a location. Those meanings must emerge from predictive structure;
the implementation may not receive semantic labels or story templates.

## Coherence Gate

Phase 49 should be built only if Phase 48 improves bpb but leaves free-running
drift or phrase loops. Its first integrated test must hold body bytes and
exposure fixed and require:

1. no validation-bpb regression;
2. fewer and shorter exact repeated blocks;
3. improved story-boundary completion;
4. persistent entity/event constraints across intervening local motifs;
5. goal turnover substantially slower than motif turnover;
6. goal changes causally preceded by achievement, sustained mismatch, or
   novelty rather than a fixed language-shaped timer;
7. success on a non-language hierarchical-control stream with variable local
   trajectories to the same distal goal.
8. no semantic goal label, correct future target, or ground-truth route is
   supplied by the harness;
9. every primitive regulated variable and every transient byte used by goal
   competition is reported.

The critical causal intervention is to clamp or swap `g` while leaving all
local sensory state unchanged. If local organs do not change coherently, the
goal signal is decorative. If clamping overrides sensory evidence completely,
its authority is too high.

## Isolated Gate Result

The first implementation is deliberately smaller than the byte cortex. A
16-assembly organ stores one signed restoration statistic per assembly, a
bounded eligibility vector, and one persistent winner. A decrease in a
non-negative regulated deviation reinforces recent eligible events. A learned
restorative event can then emit a weaker conditioned signal to earlier eligible
events. No desired target, correct action, future route, or modality label is
provided.

On a generic variable-trajectory chain with eight random local events between
successive useful predictors:

| Condition | First restorative predictor | Dead-loop value | Selected goal |
|---|---:|---:|---:|
| Direct restoration only | 0.0199 | -0.00337 | none |
| Conditioned restoration | 0.9943 | -0.00337 | restorative predictor |

The terminal restorative assembly reached 0.9983 in both conditions. Direct
eligibility learned the nearby intermediate (0.8012) but could not carry useful
value to the first predictor. Conditioned restoration propagated that value
without extending one global trace across the whole trajectory.

A measured predictability-only control chose the deterministic dead loop: its
conditional transition entropy was 0 bits versus 3.5233 bits for the variable
restorative route. The conditioned goal persisted across 16 neutral local
steps, dissolved when deviation reached zero, and swapping its identity changed
all four trained specialist actions. Durable goal anatomy was 451 bytes and
peak transient state was 270 bytes; the four-module local controller was 401
bytes.

This passes goal origination, persistence, anti-loop selection, local
conditioning, and accounting in isolation. It does not yet demonstrate
prospective planning or improve TinyStories. The next step is to derive an
honest modality-neutral viability deviation from the byte cortex and connect
the winner through bounded apical gain, first behind an ablation flag.

## First Cortical Integration Screen

The fourth organ is now available behind `--goal-conditioned-event`. Its
declared viability deviation is the sum of normalized externally scored motif
surprise, recent motif repetition, and one-quarter of route plus motif capacity
pressure. This is intentionally an engineering hypothesis, not a hidden claim
that these three quantities are biology's universal viability variables.

The sparse organ retains at most 4,096 valued goal assemblies and 64 live
eligibility traces. A winner contributes only a bounded fractional pseudo-count
to an already active temporal route row. Zero goal gain is bit-for-bit identical
to the Phase 48 temporal learner. Sustained worsening dissolves an unreached
goal, fixing the isolated prototype's possibility of indefinite persistence.

At 100KB TinyStories exposure, a gain sweep was monotonic but negligible:

| Goal apical gain | Validation bpb |
|---:|---:|
| 0.00 | 1.829463 |
| 0.10 | 1.829458 |
| 0.20 | 1.829454 |
| 0.35 | 1.829449 |
| 0.50 | **1.829444** |

The goal organ occupied 59,682 durable bytes, held 2,505 valued assemblies,
and produced 149 goal selections and 149 dissolutions. The 0.000019-bpb range
is too small to claim a predictive benefit. It does establish that bounded goal
conditioning does not tax the small transfer result and provides a safe gain
for a larger matched generation screen.

A subsequent structural screen reduced the default bank from 4,096 to 256
assemblies and replaced repeated full-bank weakest searches with a cached
competitor. At 100KB, capacities 64, 128, 256, and 512 occupied approximately
1.6KB, 3.1KB, 6.1KB, and 12.2KB respectively. Their validation bpb remained
within 0.000011 of the zero-gain control. The 256 default preserves substantially
more potential goal diversity than the smallest bank while making mature
rejection constant-time in the common case.

Independent gain conditions now run in separate processes. A four-condition
100KB screen completed in 11.05 seconds at 72,427 aggregate bytes/second while
other pilots were active. This parallelism is valid across gains, capacities,
and seeds; bytes within one continual cortex remain causally ordered and are
not partitioned by this harness.

A five-core 250KB screen extended goal pseudo-count gain through 4.0 while
retaining the downstream event/byte mixture cap. Validation bpb moved from
1.6628291 at zero gain to a shallow best of 1.6628259 at gain 2.0, then returned
to 1.6628289 at gain 4.0. The 0.0000032 range is noise-sized: stronger goal
authority alone is not the missing coherence mechanism. Gains 0 and 2 advance
to a checkpointed 1MB generation comparison; gain 4 does not.

The requested extension through gains 8, 12, and 16 changed that interpretation.
Gains 4, 8, 12, and 16 all produced exactly 1.662828882 validation bpb and the
same goal turnover. The pathway saturates structurally at gain 4; there is no
continuing high-gain regression curve.

At 1MB, gain 2 scored 1.450991 frozen validation bpb versus 1.450977 at zero
gain. More importantly, the matched ten-seed generation sweeps were byte-for-
byte identical. The first cortical wiring therefore fails its causal gate.
Inspection explains why: event apical counts are admitted only after at least
one byte of the current motif is present. If the locally selected first byte
does not match the goal motif, prefix filtering removes the goal before its
vote can act. Increasing a signal that arrives after target selection cannot
repair coherence.

The next integration must project the selected goal's learned initial-byte
distribution into the pre-prefix byte basin, still under a bounded mixture.
Only a clamp/swap test that changes generated bytes while preserving local
sensory state can reopen the TinyStories gate.

The preselection projection passed that causal test but failed the predictive
gate. At 100KB, validation worsened monotonically from 1.829463 at zero gain to
1.829788 at gain 0.02, 1.830984 at 0.05, 1.833365 at 0.1, and 1.841444 at 0.5.
Gain 0.5 changed generated bytes and raised boundary completion from 50% to 60%
in a ten-seed screen, but did not improve n-gram diversity or compression. The
sample is too small to treat the boundary change as coherence.

This falsifies a goal represented as one forced future motif. It is causal but
over-specific. The next design groups candidate motifs by learned restoration
signature and projects a distribution over all reachable members of the active
restorative basin. Local route evidence then chooses the concrete member. This
is the goal analogue of XOR/XNOR: different surface events may occupy one basin
because they have the same consequence for organization.

That restorative-basin projection removed nearly all of the forcing penalty,
but did not pass the gate. At 100KB, a narrow basin and gain 0.02 changed
validation from 1.829463 to 1.829468 bpb; gain 0.1 reached 1.829527 after goals
were made explicitly homeostatic. Broadening the basin from 0.0625 through 1.0
did not produce an improving point. At gain 0.5 only three of fifty generated
samples changed, and boundary completion remained 42%.

The lifecycle correction itself is retained. A goal no longer dissolves when
one motif—or another member of its basin—appears. It records the regulatory
deviation at goal birth and persists until deviation falls by a bounded
fraction, or repeated worsening kills it. On the 100KB validation slice these
states lasted a median 30 bytes and as many as 481 bytes. That is genuine
long-range state, but its direct byte projection was active on only 0.32% of
bytes and remained slightly anti-predictive.

The next cortical interface therefore implements the local policy suggested by
the biological synthesis directly: local route plasticity and recall are
conditioned on the low-dimensional goal basin, approximating
`action_i = policy_i(local_state_i, goal)`. The global organ must not specify a
byte or motif. It supplies context; local experience learns what response works
under that context.

Phase 51 implemented that interface as bounded goal-conditioned route anatomy,
with direct byte projection disabled. A 100KB screen produced the first
nonzero improving point at basin width 1 and gain 0.25, but the improvement was
only 0.0000012 bpb. The matched 1MB gate rejected it as noise: 1.50503917 versus
the 1.50503897 zero-gain control. The anatomy learned 3,884 route paths and
40,165 synapses, occupying about 510KB, but only 28 paths exceeded the existing
predictive-utility threshold.

This localizes the next bottleneck to the content of `goal`. A scalar signed
restoration value collapses qualitatively different regulatory needs. Phase 52
will retain scalar value for goal competition but learn a tiny restoration
signature over the generic viability components—predictive surprise,
repetition pressure, and tissue-capacity pressure. Goal basins will then encode
which regulated variable an event tends to restore, rather than bins of an
arbitrary scalar magnitude.

Phase 52 implements that vector without adding labels or modality-specific
features. Each event's existing eligibility trace receives three independent
local restoration signals. Scalar restoration still governs whether an event
is worth pursuing; the dominant positive component supplies the shared goal
basin used by local route plasticity. The scalar-bin rule remains only as a
checkpoint-compatible fallback when no component evidence exists.

The 100KB gate did not yet generalize. Gain 0.25 improved training from
2.163655 to 2.163651 bpb but worsened validation from 1.829463 to 1.829473.
The signature organ occupied 32.7KB, formed 1,402 goal-route paths and 12,291
synapses, and only 12 routes cleared predictive utility. A matched 1MB pair was
therefore launched detached to distinguish immature reusable structure from
context overfit before any full-corpus run.

The matched 1MB result confirmed the vector-event route was not reusable:
gain 0.25 scored 1.50504020 versus 1.50503897 at zero gain. Phase 53 then
inverted goal formation so the largest active regulatory deficit, rather than
a candidate event, selected a persistent winner. That state survived competing
needs until its own pressure fell by 10%, matching hierarchical competing
inhibition. Directly learning a separate route row for every mode saturated
the 10,000-path cap after only 100KB and regressed validation.

Phase 54 removed those route copies and multiplicatively gated each existing
route's event counts by the event's restoration signature. This eliminated all
duplicated paths, but every nonzero point still regressed; gain 0.25 changed
1.829463 to 1.829622 bpb and higher gains saturated near 1.829610. Together
with Phases 49 and 50, this is strong evidence that goal context must not alter
the categorical content distribution directly.

The next interface follows the selective-connectivity evidence more literally.
Existing temporal routes receive local eligibility and component-specific
restoration credit. The active deficit may then select among routes already
known to be predictive, while the selected route's event distribution remains
untouched. Goal context controls which message gets through, not what that
message says.

Phase 55 implements selective route communication. Existing temporal routes
carry a decaying local eligibility trace and learn three signed restoration
credits. During recall, the active deficit contributes a bounded bonus only to
competition among routes that already pass predictive utility. It never edits
the winning route's content counts. A frozen 100KB checkpoint improved from
1.829462965 to 1.829461501 bpb once authority became sufficient to change a
route winner, then produced the identical score at all stronger tested points.
This plateau distinguishes a discrete connectivity decision from continuous
output forcing.

A modality-neutral causal probe then trained 32 pairs of equally predictive
routes using restorative feedback delayed by 16 local steps. Deficit-gated
selection reached 100% accuracy, the zero-goal control remained at 50%, and
swapping the deficit mode changed the selected route in all 32 modules. The
probe used 64 credited paths, a 29KB body, and no language-derived feature.

The matched 1MB Phase 55 gate showed why restoration alone is insufficient:
gain 4 scored 1.50504026 versus 1.50503897. A route can be restorative yet be
the wrong prediction in a particular corpus context. Phase 56 therefore adds
a three-scalar metaplastic permission signal. For each regulatory mode, the
cortex evaluates the counterfactual goal-selected route against the ordinary
predictive winner using the observed event. Goal-based connectivity remains
silent until that switch has positive local information gain.

At 100KB this preserved exact control training bpb at every gain. Gain 1
learned negative surprise-mode selection utility and remained exactly equal to
control validation. Gains 4 and 16 unlocked one positive capacity-pressure
switch and both scored 1.829461501 versus 1.829462965. The identical plateau is
again consistent with a discrete communication decision, now protected by
learned evidence rather than a hand gate. The modality-neutral delayed probe
continues to pass at 100% because its route switches earn positive utility.

Multi-core consolidation was also repaired. Mature forms no longer re-enter a
birth nursery when territories merge; packed contexts are decoded using the
source vocabulary; unreachable legacy contexts containing the unknown sentinel
are discarded; and event restoration plus route credit are remapped by form.
A two-worker 200KB smoke merged in 0.12 seconds, retained 8,825 credited routes,
and produced a 3.09MB serialized body under the 64MB cap.

The matched 1MB Phase 56 gate was safe but silent. Gain 4 exactly matched the
1.505038972 control because the corpus-wide surprise and capacity permissions
both learned negative utility. That result rejects one permission scalar per
regulatory mode: it averages together local contexts in which the same
goal-conditioned route switch can help or hurt.

Phase 57 makes permission multisite and local. Its key is the conjunction of
regulatory mode, ordinary predictive route, and goal-preferred route. A switch
is therefore permitted only where that particular competition has produced
positive counterfactual information gain. This mirrors the biological
distinction between a compact persistent state and selective gating at several
sites; the broadcast goal remains low-dimensional, while its authority is
learned where it acts.

The modality-neutral delayed-feedback probe still reaches 100% conditioned
selection versus 50% without the goal. It learned 32 positive local permissions
across 32 independent modules, so the mechanism is not tied to text. On the
100KB TinyStories gate, gains 1, 4, and 16 all exactly matched the 1.829462965
control while learning only 2, 4, and 5 positive permissions respectively.
This establishes non-regression at small scale, but the evidence is sparse and
mostly one-shot. A detached 1MB sweep over gains 4, 8, 12, and 16 must decide
whether repeated local evidence turns the mechanism into useful communication
or merely preserves a silent control.

That sweep was exactly equal to control at every gain, but explicit telemetry
showed why: it learned 50 exact route-pair permissions from only 52 examples
and used none during frozen validation. Phase 58 pooled permission by deficit
and the two routes' temporal receptors. This reduced the generic probe from 32
permission records to two with 640 observations while preserving 100% goal-
conditioned accuracy, but it also made zero TinyStories switches. Its seven
learned compartment pairs did not include the validation transition.

A causal all-permitted clamp isolated the missing transition. Capacity pressure
needed to shift a route from lag 1 to lag 64 four times, improving frozen 100KB
validation from 1.829462965 to 1.829461501. Training had already observed that
capacity pressure benefited lag 1 to lag 2. The reusable relationship was not
the destination compartment but the direction of temporal integration.

Phase 59 consequently compresses metaplastic permission to nine possible
states: three regulatory deficits crossed with shorter, unchanged, or longer
temporal horizon. The restorative synapse still selects the concrete source;
the shared gate only learns whether that deficit should permit cortex to move
its integration horizon in a direction. On the generic delayed probe, two
permissions and 640 observations retained 100% conditioned selection against
50% control. On TinyStories, gains 4, 8, 12, and 16 all made 11 train and four
validation switches, improved training bpb by 0.00000350, and improved frozen
validation by 0.00000146. The identical high-gain plateau is the signature of
a bounded discrete route decision rather than probability forcing. A detached
1MB sweep is the next generalization gate.

The 1MB gate passed at gains 8, 12, and 16. Each made two frozen-validation
route switches and improved bpb from 1.505038972 to 1.505038278. Gain 4 made no
validation switch and exactly matched control. Gain 8 is selected because it
is the lowest effective authority and disturbed training least among the
winning settings. The improvement remains tiny, but it is causal, bounded,
modality-neutral in the synthetic probe, and non-regressing on held-out bytes.
That is sufficient to move the mechanism into a detached, 16-territory,
64MB-bounded full TinyStories run; generation will remain a separate coherence
gate when the checkpoint completes.

The 1MB checkpoint's paired free-running test exposes the remaining weakness.
Across twelve 1,000-byte samples and four prompts, gain 8 and gain 0 were
byte-identical. The goal state was active at 424 route-selection calls, but
restoration credit proposed an alternative only twice and the learned gate
denied both. Both proposals asked capacity pressure to lengthen the temporal
horizon; that permission had 23 training observations and utility -0.131.

A targeted in-memory intervention made only that capacity/longer permission
positive. It caused two free-running switches, although the matched sampler
still chose identical bytes, and caused ten switches over the first 100KB of
held-out validation. Surprisingly, held-out bpb improved from 1.408554512 to
1.408547459 even though training utility had assigned the transition a
negative sign. Immediate next-event utility is therefore an overly
conservative and imperfect teacher for a delayed-coherence action.

The full Phase 59 run should complete unchanged so scale can test whether
repeated experience matures this sparse pathway. If generation remains silent,
Phase 60 should treat a proposed horizon change like active sensing: permit
bounded developmental trials, leave a local eligibility trace, and judge the
gate from later restoration of the active deficit. The circuit must still pass
a matched bpb gate; delayed credit is not permission to hide a prediction loss.

Phase 60 implements that fallback without changing Phase 59's default. Each
previously unproven horizon direction receives at most three developmental
trials. An executed trial leaves a decaying local permission trace; subsequent
changes in the active regulatory component add signed restoration and absolute
evidence. Positive mature directions may act while negative directions exhaust
their nursery and become silent. A causal unit probe confirms that a harmful
direction receives exactly its bounded trials and is then denied.

On the 32-module nonlanguage probe, one shared directional permission learned
from delayed outcomes and retained 100% goal-conditioned selection versus 50%
control. The first 100KB TinyStories implementation kept separate delayed
permissions per deficit and failed transfer: its positive surprise/longer gate
did not authorize an almost-unobserved capacity/longer validation proposal.
Pooling the delayed gate by direction alone fixed that abstraction while
keeping route restoration deficit-specific. It executed all four held-out
proposals and improved validation from 1.829462965 to 1.829461501, matching
Phase 59's best point. Its 186 developmental switches worsened train bpb by
0.00000829, so a 1MB gate—not the 100KB win—must determine whether delayed
permission is worth retaining.

The 1MB Phase 60 gate rejected that broader permission. It made 325 training
and 58 frozen-validation switches. Training improved slightly from
1.646143063 to 1.646142128 bpb, but validation regressed from 1.505038972 to
1.505040477. Reducing frozen authority to gains 1 and 0.5 reduced switching
without removing the regression. Delayed directional permission is therefore
too broadly pooled; switch frequency alone is not the cause.

Phase 61 replaces the third regulated variable, anatomy-capacity pressure,
with disagreement among the active temporal receptors. Capacity pressure is
not behaviorally restorable while a frozen organism cannot grow or prune;
temporal disagreement is both dynamic and locally measurable. At 100KB the
conservative gate was bpb-neutral, while the active-sensing variant made 190
training and eight validation switches and again improved validation to
1.829461501. It also created 2,654 regulatory states rather than the 1,207 seen
with slowly changing capacity pressure, confirming that the signal responds
to ongoing experience.

That cleaner homeostat did not yet make free generation causal. Across twenty
paired 1,000-byte samples, 56 goal-context calls produced no alternative-route
proposal. Lowering the route-utility admission threshold to zero exposed 233
context calls, three proposals and two accepted switches, but every paired
sample remained byte-identical. The problem is therefore downstream of simple
route availability: a switched row has weak or redundant influence on the
sampled behavior. More importantly, route restoration is currently indexed by
the active deficit only. It transmits urgency without a structured destination.
The next interface must condition local route competition on the maintained
restorative basin while leaving byte choice local; direct goal-to-byte forcing
has already failed its bpb gate.

Phase 62 composes the maintained goal's learned restoration basin with each
active route's learned successor row. This adds no route-by-goal table: a route
receives context according to the fraction of its predicted event mass that
shares the current goal basin. At 100KB it preserved the small held-out win,
but ordinary-threshold generation remained silent. Globally lowering the
route threshold exposed real distribution changes—up to 0.132 total variation
on a switched decision—but also regressed validation by 0.00092 bpb before the
goal acted. Weak anatomy cannot simply be admitted everywhere.

Phase 63 therefore lets a goal selectively recruit a merely positive reserve
route only when basin alignment lifts its combined score above the ordinary
0.05-bit threshold. Delayed permission was factored by restorative basin and
horizon action, retaining at most nine reusable gates. Five gates became
positive, but they authorized 717 frozen activations and validation regressed
to 1.829497706. Restoration relevance and local predictive competence are
separate requirements.

The remaining causal bottleneck was where the local route terminated. Like the
earlier failed goal-motif projection, routed event evidence normally arrives
only after the current motif already has a matching prefix. Phase 64 adds a
separate zero-default projection from an accepted local route's whole learned
successor row into the next-byte basin. Its authority rises smoothly with the
generic motif-age signal; the goal organ never specifies a byte or named motif.

At 100KB, all tested route-projection caps beat the 1.829462965 control:

| Maximum local route vote | Validation bpb |
|---:|---:|
| 0.01 | 1.829440139 |
| 0.02 | **1.829438122** |
| 0.05 | 1.829453317 |

The 0.02 point improves by 0.00002484 bpb, about seventeen times the earlier
route-only gain. In forty paired 2,000-byte generations it caused 208 accepted
route actions and one causal divergence at byte 992. Aggregate boundary
completion was 80.0% goal-on versus 82.5% goal-off, so it is an actuator, not
yet a coherence win. A matched detached 1MB gate now tests whether its
predictive benefit and structured control transfer with scale.

The selected route and temporal traces cannot change within one motif. Caching
that transient successor projection until motif completion preserved the
0.02-point validation result exactly at 1.829438121533353 and raised frozen
throughput from about 9.0KB/s to 14.6KB/s. The cache is working state, not body
anatomy.

The 1MB Phase 64 gate rejected that configuration. Training improved from
1.646143062 to 1.646112158 bpb, but validation regressed slightly from
1.505038972 to 1.505039669. Its 983 held-out proposals produced only 46
accepted actions, and all 17 proposals encountered in forty paired free
generations were denied. Clamping all nine permissions caused two causal
divergences but worsened boundary completion and diversity. The problem was
therefore the command-like route actuator, not merely early permission lock-in.

Phases 65–67 replace repeated winner changes with weak goal-conditioned flow.
An empirical motif-length hazard learns when successor evidence should arrive.
Mature routes are weighted by the agreement between their locally learned
three-component restoration signature and a compact regulatory polarity, then
their complete successor rows contribute a bounded population vote. No target
motif, character class, or story boundary is supplied.

Using selected motif goals remained coverage-limited even with 4,096 valued
anchors. Using the active regulatory mode itself as the three-state polarity
removed that enumeration. At 100KB the route-flow cap produced:

| Maximum regulatory flow | Validation bpb |
|---:|---:|
| 0.005 | 1.829306598 |
| 0.010 | 1.829290165 |
| 0.020 | **1.829285640** |
| 0.050 | 1.829332525 |

The 0.02 point improves held-out bpb by 0.000177325 with 407 flow uses. In
forty paired 2,000-byte generations it changed one continuation at byte 13,
raised story-boundary completion from 87.5% to 90%, and raised mean distinct
4-grams from 0.7093 to 0.7170. This is the first integrated goal condition to
move predictive and behavioral metrics in the same direction. A 1MB transfer
gate is required before promotion.

The gain curve is real rather than a lucky gain-8 point. At 100KB, gains 8,
12, and 16 all beat the 1.829462965 control. Gain 8 remained the held-out
minimum at 1.829285640; gains 12 and 16 improved training by another 0.0000018
and 0.0000027 bpb but gave back about 0.0000004 and 0.0000006 validation bpb.
This is a smooth authority-saturation curve, so gain 8 remains the smallest
and best-generalizing setting.

The matched 1MB transfer gate passed. Gain 8 improved training from
1.646143062 to 1.645849771 bpb and frozen validation from 1.505038972 to
1.504782415 bpb. It used regulatory flow 12,192 times during learning and
16,124 times during evaluation for only 164 additional embodied bytes. Across
forty paired 2,000-byte free generations, it changed five continuations,
raised boundary completion from 80.0% to 82.5%, and raised mean distinct
4-grams from 0.7101 to 0.7137. The samples remain grammatically poor and do not
yet demonstrate story-level coherence; these are directional behavioral gains,
not a solved generation result.

One hidden language assumption remained in the regulator: prediction surprise
was accumulated only for ASCII letters. Phase 68 exposes the surprise scope
and adds an `all_bytes` mode in which every opponent-coded raw observation can
drive the same local homeostat. At 100KB, generic all-byte gain 8 improved
training by 0.000412132 bpb and frozen validation by 0.000161482 bpb. Gains 12
and 16 again saturated slightly behind gain 8 on validation. An XOR relabeling
of every input byte converts the same structured stream into opaque high-bit
sensor values while preserving opponent-receptor geometry; the complete
probability trace, flow-use count, and regulatory-mode births remain exactly
invariant. This is the first direct check that the integrated flow circuit—not
only its isolated alignment function—does not depend on textual byte identity.
The detached all-byte 1MB replication also passed and improved more than the
letter-filtered circuit: training moved from 1.646143062 to 1.645748958 bpb and
frozen validation from 1.505038972 to 1.504699233 bpb. The circuit used 14,309
training and 18,653 evaluation flow events for 167 additional embodied bytes.
Its forty-pair generation gate changed five continuations, raised boundary
completion from 82.5% to 85.0%, and raised mean distinct 4-grams from 0.7362 to
0.7382. Phase 68 therefore replaces the obsolete Phase 59 run. The new
authoritative full run is detached with sixteen 1MB territories, all-byte
surprise, gain 8, a 0.02 local-flow cap, and an exact 64MB body ceiling.

A causal horizon ablation on the 1MB checkpoint locates the gain inside the
story rather than at its delimiter. Goal flow lost 15.52 bits over the first
64 bytes, then saved 5.56, 56.11, 132.86, 167.48, and 6.85 content bits in the
64–128, 128–256, 256–512, 512–1,024, and 1,024+ position bands. The complete
story body saved 353.35 bits while the `<|endoftext|>` delimiter itself lost
13.61 bits. The integrated 339.74-bit validation win is therefore an
accumulated-context effect whose strength peaks at 512–1,024 bytes, not a
shortcut for predicting the corpus separator.

An oracle audit reset all transient route and goal state at each known story
delimiter while leaving learned anatomy unchanged. The gain survived and rose
slightly from 339.74 to 344.47 bits. Story content saved 357.95 bits, and the
first 64-byte band changed from a 15.52-bit loss to a 3.49-bit saving; the
128–1,024 bands still saved 58.86, 135.72, and 157.14 bits. Cross-story state
carryover is therefore a small source of early interference, not the source of
the long-horizon result. A future learned context-separation action may recover
that small gap, but an explicit TinyStories delimiter reset is not required by
the promoted circuit.

The aggregate gain is heavy-tailed across stories. Of 1,259 complete held-out
stories, 18.75% improved and 81.18% regressed slightly. Mean saving was +0.281
bits/story while the median was -0.195; story length explained little
(correlation 0.076). The circuit is therefore discovering a useful long-range
operation, but does not yet know broadly when to apply it.

Two simple attempts to improve that breadth were rejected at 100KB. Raising
restoration-direction alignment to powers 2, 4, and 8 changed validation by
only about one-billionth of a bit/byte. Replacing normalized direction with
raw learned restoration efficacy changed validation by only 4.5e-8 bpb and
left the story win fraction and median exactly unchanged; larger authority
caps regressed rapidly. Weak diffuse route votes are not the cause. The next
diagnostic must identify the contextual state shared by the rare strong wins
and learn an applicability gate without naming stories or language features.

Byte-level causal traces explain the asymmetry. The strongest winning story
saved 29.66 bits, with one route-flow correction accounting for 30.02 bits;
other top stories contain single 17–25 bit rescues. The worst complete story
lost only 2.64 bits, while the common incorrect nudge costs about 0.0053 bits.
Savings correlate with neither story length nor zlib compressibility. The
circuit is functioning as a weak contextual rescue prior: it pays a bounded
small cost when its sparse route is wrong and restores substantial probability
when the local cortex has made a catastrophic exclusion. Full-corpus training
must determine whether more experience broadens the coverage of those rescues
without increasing their bounded cost.

Phase 69 asks the learner to discover flow applicability rather than hard-code
the best mode found by an offline ablation. Three mode-local metaplastic
permission synapses accumulate the exact causal bit consequence of mixing a
regulatory route vote with the local cortical prediction. An unseen mode gets a
short developmental nursery; thereafter positive cumulative utility permits
its flow and negative utility silences it. The update occurs only after the
observed byte, and a silent mode can still measure its counterfactual outcome
and reopen if the stream changes. This adds three signed utilities and three
evidence counts, not a language label or gradient.

At 100KB, 8- and 32-trial nurseries briefly closed surprise flow after four
unlucky observations and lost only 0.272 training bits; 128 trials reproduced
the all-mode result exactly. All three settings converged to the same frozen
validation bpb. At 1MB, 32- and 128-trial runs agreed to 1.8e-13 validation bpb:
1.504697075 versus Phase 68's 1.504699233. The permission synapses learned
+750.58, +0.116, and +16.19 training bits for surprise, repetition, and temporal
disagreement. The 32-trial run applied flow on 318,688 of 318,692 surprise
candidates, all 48 repetition candidates, and 12,628 of 15,154 disagreement
candidates. Mature behavior therefore saturated despite different nursery
windows.

The improvement is real but does not solve generation. The causal horizon gain
rose from 339.74 to 341.90 held-out bits and story-content savings rose from
353.35 to 355.31 bits, but the complete-story win fraction remained 18.75% and
median saving remained -0.195 bits. In 32 paired free generations, only 9.4%
changed, boundary completion was tied at 90.6%, and the changed continuations
remained incoherent. Phase 69 is retained as a tiny, modality-neutral
applicability primitive. It localizes the next missing mechanism: the compact
shared state must persist and constrain a trajectory, rather than merely
authorize sparse byte-level rescues.

A direct persistence trace confirms that diagnosis. During the 1MB frozen
evaluation, regulatory mode was active for 81.31% of bytes, with a 19-byte
median run and 124-byte 90th percentile. The explicit selected goal was active
for only 0.76% of bytes, with an 11-byte median run, 62-byte 90th percentile,
and no run reaching 512 bytes. Only 1.03% of route-flow onsets occurred while a
selected goal existed. The present circuit therefore has a broadly available
error polarity but almost no maintained target state. Phase 70 should test a
target lifecycle that terminates on learned target achievement or sustained
failure, then require flow to agree with both the current regulatory need and
the maintained target.

Phase 70 falsifies the simplest version of that repair. A target-or-failure
lifecycle with failure patience 16 raised selected-goal occupancy from 0.76%
to 57.95%; median lifetime rose from 11 to 2,322 bytes, and goal/flow overlap
rose from 1.03% to 56.31%. A soft 0.25 target influence retained the fast
regulatory path while favoring routes whose restoration signature matched the
maintained target. It scored 1.504697411 bpb, only 0.335 bits per million worse
than Phase 69, but lost 38.9 training bits per million and reduced aggregate
throughput from 5.95 to 3.68KB/s under matched contention. Most importantly,
its 32 paired generations were exactly the same as Phase 69's: 9.4% changed,
90.6% boundary completion, and no coherent narrative improvement.

One predicted future motif is therefore the wrong content for a slow global
state. It changes many teacher-forced local route choices but behaves like a
sticky command, not an organized attractor shared by compatible events. Phase
70 remains available only as a causal ablation. The next representation must
let multiple locally different events organize into the same learned basin and
allow that basin—not a single motif identity—to constrain a trajectory.

Phase 71 implements that structural-basin test. A small competitive organ
clusters predictive successor signatures without copying motif identities or
learning its own byte output table. With sixteen available basins, gain 1, and
0.5 switching hysteresis, the 1MB checkpoint improved frozen validation by
0.352 bits per million over Phase 69 while adding about 1.5KB to the body. It
also reduced aggregate throughput by about 11.6%. The important negative result
is behavioral: all thirty-two paired generations were exactly the same as the
Phase 69 generations. A horizon trace then showed that one basin occupied
999,973 of 1,000,000 held-out bytes. Phase 71 has learned a useful slow bias,
but not yet a diverse episodic address or generative state.

## Phase 72 hypothesis: fast transient indices and closed-loop reinstatement

The next mechanism should not put more persistent content into the slow basin.
It should keep four quantities separate:

\[
\text{slow structural basin }B
\;\times\;
\text{retrieval cue }R
\;\times\;
\text{cortical receptor }J
\;\times\;
\text{existing filler assembly }C.
\]

A small sparse binder population should form a rapid, temporary index when
these quantities coincide. Local reciprocal plasticity implements both
directions:

\[
(B,R,J,C)\rightarrow I,
\qquad
(B,R)\rightarrow I\rightarrow (J,\hat C).
\]

The retrieved object is the original distributed cortical assembly, not a byte
label and not a new prediction made by the binder. The ordinary event cortex
must remain the only output pathway. This constraint turns Phase 72 into a
clean test of indexing and pattern completion rather than another language
model hidden beside the existing one.

Here (R) says when a memory is requested, while (J) says how the receiving
cortex interprets it. Neither is evidence that the learner has discovered a
semantic role such as agent, owner, or recipient. The proposed anatomy is
deliberately bounded: retain the Phase 71 basin organ, learn 16–64 sparse query
assemblies from recurring temporal relationships, and
use only 256–1,024 k-WTA binder cells with fast Hebbian or eligibility updates,
strong inactivity decay, and competitive replacement. Its timescales should
separate cleanly:

- basin: slow reusable structural context;
- binder: fast one-shot association lasting roughly one episode;
- cortex: reusable durable sensory and event assemblies.

Before developing this anatomy, two cheap causal diagnostics must distinguish
missing binding from autonomous state drift. On held-out stories, compare a
normal prefix with one whose evidence more than 128 bytes in the past is
scrambled, then repeat the corrupted condition while replay-clamping the basin
trajectory recorded under normal teacher forcing. Separately, replay a
teacher-forced basin trajectory while the cortex generates freely. Failure of
the intact basin to rescue corrupted content supports the missing-index
hypothesis; improvement under teacher-basin replay would instead expose
closed-loop maintenance as a major bottleneck. Both failures may coexist.

The promotion gate is causal rather than BPB-only. Swapping a basin while
preserving fillers should alter continuation structure more than identity;
swapping one role–filler binding while preserving the basin should alter
continuation content more than structure. A successful double dissociation
would demonstrate reusable context and content rather than another small
teacher-forced calibration gain.

The precursor assay supports building the index. Across sixteen held-out
stories and 4,096 continuation bytes, scrambling all prompt evidence older than
128 bytes cost effectively zero bits, and clamping the intact teacher-forced
basin into that corrupted run recovered -0.0000035 bits. Driving a free-running
cortex with the teacher basin likewise changed zero of sixteen sampled
continuations and left reference-byte accuracy exactly unchanged at 8.59%.
This checkpoint cannot retrieve distant content from its slow state, and the
result is not explained by autonomous basin drift. Phase 72 should therefore
test the bounded reciprocal index rather than tune the basin's authority again.

The basin is also not an episode identifier. A useful structural basin should
be reused by distinct episodes with the same future geometry. Episode validity,
recency, and the current query must disambiguate their transient bindings. The
one-basin horizon result makes this distinction operational: normal addressing
must be compared with basin-independent addressing, while oracle boundary
invalidation separately diagnoses cross-episode interference.

The first two integrated indices were rejected. Temporal-occupancy roles
collapsed to five broad classes and lost 106.86 held-out bits per million.
Predictive-geometry cues produced all 32 available learned roles, but restoring
only a filler identity still regressed because the assembly was inserted into
the newest temporal slot regardless of its original relation. Extending each
binding to restore both the filler and its original cortical receptor changed
the sign at 100KB, saving 2.21 bits, but lost 7.48 bits at the exact 1MB
transfer gate. It improved training by 3.09 bits, indicating specialization
rather than held-out generalization.

Exact same-state causal readout now moves ahead of learned filler eligibility.
For each sampled read it branches selected, off, wrong-receptor, and
alternative-filler conditions, then measures the final byte probability after
all route mixing and the actual sampling transforms. On the rejected 1MB
checkpoint, the first sixteen selected reads had zero immediate effect, zero
16-byte observed-future effect, and zero sampling-distribution total variation.
Every selected filler/receptor route had negative learned route utility and was
therefore suppressed by the cortical output gate. One wrong-receptor branch did
reach an output-effective route and saved 0.216 bits. The binding proxy was
positive for all sixteen reads but agreed with the exact causal sign zero times.
The present bottleneck is thus output-effective receptor integration or query
selection, not yet evidence for a filler-learning defect.

The next gate supplies a known useful existing assembly through its known
receptor on a controlled raw-byte task. If that oracle memory cannot help, the
receiving cortex must be repaired before binder selection is trained. Recall
activity is explicitly provenance-tagged: it may influence prediction, but it
cannot renew its timestamp, create route evidence, or bind itself as though it
were a new observation. Persistent usefulness credit must likewise never be
derived merely from a self-sampled byte, whose expected likelihood-ratio credit
is nonnegative by construction.

The controlled gate uses fixed-width raw sensory motifs, two randomized
associations, seventy distractor motifs, within-episode reassignment, and the
same structural episode with different fillers. Across four seeds and 400
queries, positive receiving routes were available for 352 queries. Baseline
identity cost was 3.097993 bpb; evaluator-traced content through its selected
receptor improved this to 3.078060 bpb. The same content through the adjacent
receptor regressed to 3.191533 bpb, while a different filler at the selected
receptor remained near baseline at 3.096832 bpb. Thus the existing cortex can
use the correct assembly, and receptor identity is causal rather than merely
descriptive.

The learned binder remained bit-for-bit identical to baseline. Replacing its
structural-basin address with a basin-independent query-only address also
changed exactly zero scored bits. The basin is not the immediate failure. The
remaining mismatch is directional: the current implementation stores the most
distinctive *prior* motif as content, whereas the controlled task requires
active query assemblies to bind the newly observed cortical assembly. Phase 72
should next test that local coincidence direction before adding learned credit
or capacity.

Multicore territory training is restored for this path. A four-worker 100KB
merge smoke sustained about 20.4KB/s training versus roughly 5.1KB/s for the
single-condition runner. Learned role prototypes and their local utility are
remapped during consolidation, transient bindings remain territory-local, and
capped motif vocabularies now skip unadmitted incoming assemblies instead of
asserting. This repairs the consolidation failure that stopped the earlier
full run after 399MB.

## Primary Sources

- Tarsitano and Andrew (1999), [Scanning and route selection in the jumping
  spider Portia labiata](https://pubmed.ncbi.nlm.nih.gov/10458876/).
- Zurek et al. (2018), [Lateral eyes direct principal eyes as jumping spiders
  track objects](https://pubmed.ncbi.nlm.nih.gov/30253146/).
- Menda et al. (2014), [Visual perception in the brain of a jumping
  spider](https://doi.org/10.1016/j.cub.2014.09.029).
- Steinhoff et al. (2017), [The synganglion of the jumping spider Marpissa
  muscosa](https://doi.org/10.1016/j.asd.2016.11.003).
- Sumbre et al. (2001), [Control of octopus arm extension by a peripheral motor
  program](https://pubmed.ncbi.nlm.nih.gov/11546877/).
- Zullo et al. (2019), [Motor control pathways in the nervous system of
  Octopus vulgaris arm](https://pmc.ncbi.nlm.nih.gov/articles/PMC6478645/).
- Bidel et al. (2023), [Connectomics of the Octopus vulgaris vertical
  lobe](https://pmc.ncbi.nlm.nih.gov/articles/PMC10325715/).
- Keramati and Gutkin (2014), [Homeostatic reinforcement
  learning](https://elifesciences.org/articles/04811).
- Filo et al. (2022), [A genetic mammalian proportional-integral feedback
  control circuit for robust and precise gene
  regulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9214505/).
- Najarro et al. (2023), [Complex computation from developmental
  priors](https://www.nature.com/articles/s41467-023-37980-1).
- Yamada et al. (2023), [Hierarchical architecture of dopaminergic circuits
  enables second-order conditioning in
  Drosophila](https://pubmed.ncbi.nlm.nih.gov/36692262/).
- Mussells Pires et al. (2024), [Converting an allocentric goal into an
  egocentric steering signal](https://www.nature.com/articles/s41586-023-07006-3).
- Matheson et al. (2022), [A neural circuit for wind-guided olfactory
  navigation](https://www.nature.com/articles/s41467-022-32247-7).
- Thapliyal et al. (2023), [Multisite regulation integrates multimodal context
  in sensory circuits to control persistent behavioral states in C.
  elegans](https://www.nature.com/articles/s41467-023-38685-1).
- Zhang et al. (2025), [Hierarchical competing inhibition circuits govern
  motor stability in C. elegans](https://www.nature.com/articles/s41467-025-59668-4).
- Skoge et al. (2014), [Cellular memory in eukaryotic
  chemotaxis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4210025/).
- Kramar and Alim (2021), [Encoding memory in tube diameter hierarchy of a
  living flow network](https://pubmed.ncbi.nlm.nih.gov/33619174/).

These sources support the component observations. The unified goal-control
algorithm is an engineering synthesis and must earn its place through ALM
ablation and causal intervention.
