# ALM Long-Range Coordination Reference

Status: Living research reference

Evidence cutoff: 2026-09-07

Scope: Dendritic ALM only

## Working Principle

> A learning rule can be local even when the information driving it comes
> from across the system. Locality describes where an update is computed,
> not how far the signals available at that site have travelled.

This distinction is the most useful connection between current neuroscience
and ALM. Long-range coordination and local plasticity are complementary:
circuitry can route context, timing, or teaching information to a dendritic
compartment, and that compartment can still update using only signals that are
physically present there.

This is a reference and hypothesis generator, not evidence that the brain uses
one universal learning algorithm. It separates three problems that should also
remain separate in ALM:

1. **Communication:** which distant activity can influence a column now?
2. **Representation:** which reusable components and relations encode the
   current situation?
3. **Plasticity:** which locally available variables change durable structure?

### Metric note

Probe scores use `-log2(P(observed outcome))`. On a binary task, 1 bit/outcome
is uniform 50% uncertainty, 0 bits is perfect confidence, and values above 1
are worse than a coin flip. TinyStories uses the same unit over 256 byte
outcomes, where uniform uncertainty is 8 bits/byte.

## Evidence Map

### 1. Selective coordination, not global coactivation

The mediodorsal thalamus can sustain a task-relevant prefrontal state by
selectively amplifying cortical functional connectivity without itself
carrying the categorical rule. Nonspecific cortical excitation is not
equivalent. This argues for selective recurrent gain rather than uniformly
raising activity across ALM.

Rhythmic phase can also gate transmission. In macaques, the downstream and
behavioral effects of electrically evoked V2 spike volleys depended on the
gamma phase in V4. This is causal evidence for a timing gate in that circuit,
not evidence for a universal brain-wide gamma binder. Coherence alone can also
misidentify effective communication when reciprocal coupling and delays are
present.

Population structure supplies a complementary selector. Cortex-wide mouse
recordings found that different dimensions of a region's activity belonged to
different overlapping functional networks. Human intracranial recordings
later identified a low-dimensional prefrontal-motor interaction carrying
context-dependent action information and predicting behavior better than
either region alone. Timing may regulate **when** an input is effective;
population structure may regulate **which information** is admitted.

**ALM implication:** long-range context should arrive through sparse,
addressable receptors or subspaces. A global context broadcast may exist as a
weak modulator, but it should not force all columns into one shared state.

### 2. Reusable components plus explicit relations

Abstraction preserves a useful invariant across changing details. Composition
reuses components while preserving their roles. Simultaneous activation of
`dog`, `person`, and `chasing` cannot distinguish who chased whom; coactivity
without relational roles is underspecified.

Human single-neuron recordings show that largely distinct populations can
represent stimulus content and task context while coordinated activity joins
them. Those representations generalized across changes in the other
dimension. Separability and conjunction can therefore coexist.

Human hippocampal population geometry also becomes abstract and disentangled
during learned inference, encoding both observed and discovered latent task
variables. In mice, reusable prefrontal and parietal neuronal subspaces have
been observed across stimulus-processing and memory-maintenance phases. The
biological recording supports reuse; targeted silencing in recording-
constrained recurrent models supports a possible mechanism rather than proving
a universal one.

Coherence also requires selective non-binding. Mouse circuit interventions
show that prefrontal input through medial entorhinal cortex can influence
whether hippocampal memories are integrated or separated according to context.
More association is not automatically better representation.

**ALM implication:** columns should expose reusable assemblies, but bindings
must carry role, direction, and context. A valid long-range mechanism must pass
both recombination and anti-binding/interference tests.

### 3. Compartment-specific learning

In vivo longitudinal imaging during mouse motor learning found different
plasticity predictors within the same pyramidal neuron. Apical potentiation was
associated with nearby synaptic coactivity; basal potentiation was associated
with activity coincident with postsynaptic action potentials. Blocking
postsynaptic spikes reduced basal but not apical potentiation.

This constrains any rule that treats every synapse as applying an identical
pairwise timing rule. It does not rule out a shared underlying biophysical
principle whose behavior changes with compartment state.

The 2026 *Vectorized instructive signals in cortical dendrites* result is even
closer to the ALM problem. In a brain-computer-interface task, differences
between dendritic and somatic signals contained task information whose sign
depended on each neuron's causal role and predicted learning changes. Targeted
perturbation disrupted learning. The important possibility is a neuron- or
compartment-specific instruction, not merely a brain-wide scalar reward.

**ALM implication:** basal evidence, apical context, and a somatic outcome need
not share a plasticity rule. Apical input should be tested as an addressable,
signed teaching or routing signal. Calling a global gradient `context` would
not solve credit assignment; the system must construct that signal locally
through its actual causal pathways.

### 4. Credit across time

Eligibility traces separate participation from later consequence. Experiments
in mouse striatum found that synapses made eligible by recent pre/post activity
could be potentiated by phasic dopamine arriving seconds later. The synapse
does not retain the whole prior system state; it retains a bounded local record
of its own participation.

Behavioral-timescale synaptic plasticity (BTSP) widens the interval further:
strong dendritic plateau events can alter inputs active over seconds and rapidly
form place-related representations. In 2026, plateau-like events and induced
events were also found to precede or create odor representations, with medial
and lateral entorhinal inputs playing different modulatory roles. The authors
properly describe the observations as consistent with BTSP rather than as a
complete identification of its mechanism.

**ALM implication:** each eligible structure needs an explicit trace shape,
decay time, memory cost, and conversion event. Fixed adjacent-byte context is
not sufficient evidence of delayed credit.

### 5. Replay is plural and optional

Replay may coordinate consolidation and recombine reusable pieces. Human
hippocampal-prefrontal work links ripple-associated replay with changes toward
inferred relational representations. Procedural replay has also been observed
in dorsal striatum after complete bilateral hippocampal lesions, with replay
content related to later behavioral improvement.

**ALM implication:** replay should not be designed around a hippocampal master
controller. It is a later, subsystem-local consolidation option. Awake learning
must remain primary for an always-on system; rare consolidation should earn its
throughput and memory cost through ablation.

### 6. Coherence through a low-dimensional goal bottleneck

Comparative work suggests that global coordination need not continuously move
the detailed state of every specialist module. Portia can inspect and select a
detour while following local route features; jumping-spider lateral eyes direct
high-acuity principal-eye sampling; and octopus peripheral circuitry can
execute substantial motor structure under much broader central control.

The corresponding ALM hypothesis is hierarchical goal-conditioned local
control. One sparse state `g(t)` supplies a shared future constraint, while
each organ computes from its local state conditioned by `g`. Local achievement,
surprise, and novelty then maintain, advance, or dissolve the goal. This is an
engineering synthesis, not a claimed common biological algorithm.

**ALM implication:** Phase 48's temporal working set remembers relevant past
events, but memory alone does not express commitment to a future attractor. A
small persistent goal winner may provide coherence more economically than a
larger context store. See
[`Dendritic_Stream_Learner_Phase49_Hierarchical_Goal_Control.md`](Dendritic_Stream_Learner_Phase49_Hierarchical_Goal_Control.md).

The goal bottleneck still needs an origin. A plausible construction begins
with regulation rather than a represented future:

```text
regulated deviation
    -> restorative outcome
    -> conditioned predictor
    -> competing target assembly
    -> persistent structured goal
    -> goal-conditioned local control
```

Homeostatic reinforcement learning formalizes how restoring a regulated
variable can create primary value. Biological second-order conditioning shows
that an established reward predictor can teach a new predictor. Competitive,
recurrent dynamics can then select and maintain one learned target without an
observer choosing it. Fly circuits further constrain the proposal: motivation
alone is not enough, because heading, goal direction, and steering are
separable signals that must be combined.

**ALM implication:** prediction-error reduction cannot serve as the only
primitive preference, since a phrase loop can be both predictable and useless.
The model must disclose its inherited homeostatic variables, learn intermediate
targets without semantic labels or future answers, and distinguish conditioned
goal-seeking from prospective route evaluation. This makes goal formation a
testable ladder rather than assuming `g(t)` is externally supplied.

### 7. Aneural polarity and morphology separate state from action

Dictyostelium chemotaxis supplies a useful minimal counterexample to repeated
action selection. In controlled traveling cAMP waves, cells maintained motion
toward the source even when the instantaneous gradient reversed. The
experiments and fitted model separate fast adaptive directional sensing from a
bistable polarity memory. The memory contribution remains weaker than strong
new sensory evidence, allowing persistence without preventing genuine
reversal. This suggests that a long-range state should bias a competent local
controller continuously and weakly, rather than repeatedly replace its winner.

Physarum provides a complementary structural example. Nutrient exposure
changes tube diameters, and transported softening signals imprint source
location into the hierarchy of the organism's flow network. Memory is thus
partly stored in the morphology that routes future flow, not in a central
command describing the next movement. For ALM, the closest testable analogue
is a compact regulatory polarity coupled to each route's locally learned
restoration signature. The polarity controls flow across mature routes; the
routes retain responsibility for detailed successor predictions.

## Candidate Local Rule Family

A useful schematic family is:

```text
Delta w_ij(t) = eta * e_ij(t) * m_i,c(t)
```

where:

- `w_ij` is the durable influence from source `j` to receiver `i`;
- `e_ij` is a locally stored trace of recent causal participation;
- `m_i,c` is an instructive or modulatory signal available to compartment `c`;
- `eta` is a local plasticity-rate or maturity factor.

This is a bookkeeping constraint, not the proposed final ALM equation. The hard
problem is how columns generate and route a useful `m_i,c` without hidden
gradient transport. A rule only qualifies as local when every operand can be
identified at the update site and its storage and transport cost are counted in
the embodied ALM byte class.

## Mapping to the Current ALM

| Biological constraint | Present analogue | Status | Missing test |
|---|---|---|---|
| Selective recurrent amplification | Phase 43 route perturbation | Isolated gate passed | Recurrent state maintenance |
| Addressable communication subspaces | Phase 43 learned local routes | Dense-candidate gate passed | Sparse developmental routing |
| Reusable content/context factors | Byte motifs plus contextual residuals | Partial | Novel role-preserving recombination |
| Appropriate binding and separation | Phase 42 role-gated reusable content | Isolated gate passed | Integrate with autonomously born assemblies |
| Compartment-specific plasticity | Basal sequence evidence plus apical context | Strong architectural match | Independent perturbation and rule ablations |
| Vectorized instruction | Phase 41 addressed consequence | Routed local credit passed | Autonomous credit-route discovery |
| Eligibility across delay | Phase 44 delay/decay surface | Assay passed | Overlapping causes and learned trace lifetime |
| Event identity across variable delay | Phase 48 temporal-basis working set | Generic gate and 100KB transfer pass | Free-running narrative coherence |
| BTSP-like one-shot formation | Structural birth | Superficial analogy only | Plateau-gated one-shot memory across seconds |
| Replay/consolidation | Rare prune/shed pass | Not replay | Replay benefit per byte and per second |

The table deliberately distinguishes a structural resemblance from a passed
mechanistic test.

## Prioritized ALM Experiments

### Gate LR-1: vectorized local credit

Build a modality-neutral delayed-contingency stream in which intermingled
columns make opposing causal contributions. Compare:

1. no delayed modulator;
2. one broadcast scalar modulator;
3. signed receiver-specific apical modulators generated from observed causal
   consequences.

Hold body bytes and exposure fixed. Measure delayed prediction, reversal after
contingency swaps, wrong-column updates, and retained throughput. The third
condition must outperform the scalar without encoding target identities in the
test harness.

**Phase 41 status:** routed local credit passes. At equal body size, apically
addressed instruction reached 0.007363 bits/outcome versus 0.910939 for a
broadcast scalar across ten seeds, with no wrong-receiver updates. The task
supplies the context destination but never the hidden response polarity.
Autonomous discovery of the destination remains open. See
[`Dendritic_Stream_Learner_Phase41_Vectorized_Local_Credit.md`](Dendritic_Stream_Learner_Phase41_Vectorized_Local_Credit.md).

### Gate LR-2: bind and refuse to bind

Use the same primitive learner on language-neutral relational episodes. Test
novel role reversals and combinations, then interleave deceptively similar
episodes that must remain separate. Report compositional generalization and
false-binding rate together; optimizing only one fails the gate.

**Phase 42 status:** the isolated coordination gate passes. Role-gated reusable
content reached 0.003150 bits/outcome and 100% opposed-role accuracy on pairs
withheld in both directions. Unordered coactivity fell to chance on opposed
roles, while ordered conjunction memorized training pairs but scored 1.0 bit on
unseen pairs. Autonomous content assembly birth is not part of this result. See
[`Dendritic_Stream_Learner_Phase42_Role_Binding_Separation.md`](Dendritic_Stream_Learner_Phase42_Role_Binding_Separation.md).

### Gate LR-3: selective communication

Compare global context amplification, receptor-addressed routing, and learned
low-dimensional routing under an identical byte budget. Perturb irrelevant and
relevant routes independently. A useful mechanism should preserve the relevant
state without spreading irrelevant activation or requiring dense fan-out.

**Phase 43 status:** selective routing passes; sparse development remains open.
Global pooling scored 1.080840 bits/outcome and 0% accuracy when every
irrelevant column was adversarial. Fixed addressing scored 0.005789 bits, and
local context-source competition discovered every hidden route and scored
0.002890 bits without receiving destination identities. The learned condition
currently updates all sixteen candidates and occupies 3.29KB versus 0.67KB for
the fixed route. See
[`Dendritic_Stream_Learner_Phase43_Selective_Communication.md`](Dendritic_Stream_Learner_Phase43_Selective_Communication.md).

### Gate LR-4: temporal credit window

Sweep the delay between local participation and consequence across several
orders of magnitude. Learn trace shapes and lifetimes only from generic stream
statistics. Count transient trace memory in peak resident bytes and durable
trace parameters in body bytes.

**Phase 44 status:** the delay/decay assay passes, but lifetime learning does
not. At a 128-step delay, decay 0.97 scored 0.53 bits/outcome, decay 0.995
scored 0.04, and a nondecaying trace scored 0.02. Because trials do not overlap,
long memory has no interference cost and therefore always wins. The next task
must overlap candidate causes so both traces that are too short and traces that
are too long fail. See
[`Dendritic_Stream_Learner_Phase44_Temporal_Credit_Window.md`](Dendritic_Stream_Learner_Phase44_Temporal_Credit_Window.md).

**Phase 45 status:** interference-aware trace competition finds the right
neighborhood but not enough temporal identity. Under dense overlapping causes
at an eight-step delay, local entropy competition selected only decay 0.9 or
0.97 while rejecting immediate and 0.995 traces. The selected result was still
only 0.984427 bits/outcome and 57.40% accuracy. A richer event-specific or
temporal-basis substrate is required. See
[`Dendritic_Stream_Learner_Phase45_Competing_Temporal_Traces.md`](Dendritic_Stream_Learner_Phase45_Competing_Temporal_Traces.md).

**Phase 48 status:** event-specific temporal bases preserve identity across
variable gaps. Across five seeds, coarse learned age-band routes scored
0.641946 bits/outcome versus 0.981858 for exact lags and 1.019833 locally. At
100KB TinyStories exposure they also improved validation from 1.852537 to
1.829263 bpb. A detached 30MB checkpoint is the first free-running coherence
test. See
[`Dendritic_Stream_Learner_Phase48_Event_Specific_Temporal_Basis.md`](Dendritic_Stream_Learner_Phase48_Event_Specific_Temporal_Basis.md).

### Gate LR-5: replay only if awake learning leaves a measurable gap

Run no-replay, local replay, and globally coordinated replay from the same
checkpoint. Admit replay only if it improves retained generalization or
interference resistance enough to justify wall time, working memory, and
always-on downtime.

## Evaluation Rules

For every proposed mechanism, ask:

- **Local availability:** what exact quantities exist at the update site?
- **Credit assignment:** how is helpful versus harmful participation
  distinguished, especially after delay?
- **Structured generalization:** do reusable parts recombine in unseen roles
  while inappropriate associations remain separate?
- **Falsifiability:** what changes when somatic output, one dendritic
  compartment, feedback timing, or a long-range route is suppressed?
- **Embodied cost:** how many durable bytes, transient bytes, and bytes/second
  does the mechanism require?

Learning rule and architecture must be reported separately. Identical local
rules can behave very differently under different wiring, inhibition,
competition, and routing.

## Current Design Judgment

The evidence supports adding **receiver-specific, compartment-local teaching
and routing experiments** next. It does not yet justify a global oscillatory
clock, mandatory replay, or a language-shaped long-context module.

The strongest near-term ALM hypothesis is:

> Long-range pathways select and sign locally available apical instruction;
> basal dendrites preserve predictive sensory participation; local competition
> decides which structures become durable.

That is compatible with the existing column/receptor ontology, remains
modality-neutral, and makes clean causal predictions. It also exposes the main
unresolved question: which generic local circuit can construct the instructive
signal without assuming the answer?

### September 8, 2026 implementation checkpoint

The first integrated answer is deliberately small: a persistent three-state
regulatory polarity (surprise, repetition, or temporal disagreement) weakly
modulates the population flow of mature temporal routes according to each
route's locally learned restoration signature. Routes retain their full local
successor distributions; the regulator neither names a target motif nor
commands a winner. A learned motif-ending hazard determines when that flow is
allowed to affect the next raw observation.

At 1MB TinyStories exposure this circuit improved frozen validation from
1.505038972 to 1.504782415 bpb and modestly improved paired generation
completion and diversity. The prose remains incoherent, so this is evidence
for useful coordination rather than evidence for planning. Replacing the last
ASCII-letter surprise filter with an all-byte sensor retained the held-out win
at 100KB. XOR-remapping the entire structured input into opaque high-bit bytes
preserved the complete probability trace and regulatory telemetry exactly,
which is the current modality-neutrality control.

At 1MB exposure the all-byte form improved frozen validation from 1.505038972
to 1.504699233 bpb, exceeding the 1.504782415 result from the letter-filtered
form. Treating every raw observation as potential evidence is therefore not
only the cleaner ontology; it is the better measured predictor at this scale.
Its paired generation gate also improved boundary completion and 4-gram
diversity, so this all-byte circuit is the configuration promoted to the
bounded full TinyStories run.

A position-resolved causal ablation further separates long-range structure
from local fluency. The circuit regresses over the first 64 story bytes, turns
positive after byte 64, and reaches its strongest per-byte gain from positions
512–1,024. Excluding all corpus-delimiter bytes increases its total saving from
339.74 to 353.35 bits; the delimiter alone regresses by 13.61 bits. This is the
current evidence that the shared regulatory state becomes useful as context
accumulates rather than merely learning when stories end.

As an adversarial leakage check, an evaluation-only oracle cleared every
transient route and goal trace at each known story delimiter. Net savings rose
slightly to 344.47 bits and the late-story profile remained. This rules out
cross-story carryover as the explanation, while identifying learned context
separation as a possible small refinement rather than a prerequisite.

The gain is not yet broad: only 18.75% of 1,259 complete stories improve, while
a minority of strong corrections outweigh many small losses. Neither sharper
lateral competition nor retaining restoration magnitude changes this
distribution. This focuses the next biological question on applicability:
which locally observable context tells a module that the shared regulatory
state is relevant here?

The causal byte trace sharpens that question. Most incorrect flow events cost
about 0.0053 bits, whereas isolated correct events rescue 17–30 bits that the
local cortex had nearly excluded. This is not broad semantic control yet; it
is a low-authority contextual pathway preventing rare catastrophic local
errors. Scaling evidence should be judged by whether these rescues become more
widely distributed across episodes while their downside remains bounded.

Regulatory-mode attribution shows that almost all frozen-checkpoint gain comes
from the surprise-restoration state: +340.75 held-out bits, versus -0.014 for
repetition and -0.997 for temporal disagreement. A causal surprise-only clamp
raises total savings to 345.85 bits, but hard-coding that clamp would make a
language-shaped assumption about which regulated variable matters.

Phase 69 instead implements three mode-local metaplastic permission synapses.
Each stores only accumulated causal bit change and evidence for its own flow;
the current observation cannot authorize itself, and a silenced mode can
reopen from its locally available counterfactual. At 1MB, two nursery lengths
converged to 1.504697075 bpb, 2.16 bits better than the ungated circuit. The
story win fraction and free-generation coherence did not improve. This is a
useful separation of mechanisms: locally learned applicability slightly
improves calibration, but persistent low-dimensional trajectory state remains
the missing long-range control primitive.

Persistence telemetry makes this concrete. The regulatory mode occupies
81.31% of held-out bytes, but the explicit selected goal occupies only 0.76%;
its median lifetime is 11 bytes and only 1.03% of flow events overlap one. A
state this sparse and brief cannot supply the proposed slow global constraint.
The next causal test should preserve a learned target until achievement or
sustained failure while faster regulatory and cortical loops continue beneath
it.

That causal test separates duration from useful content. Holding one predicted
motif until target or failure raised selected-goal occupancy to 57.95% and its
median lifetime to 2,322 bytes, but did not change a single paired generation
relative to the short-lived Phase 69 circuit. It also slowed the 1MB probe and
slightly worsened compression. A long-lived state is therefore necessary but
not sufficient: the state must denote a basin reusable across multiple local
events, rather than one sticky future event.

Phase 71 supplied a learned predictive basin without a direct byte head. It
slightly improved 1MB held-out BPB, but its free generations were bit-for-bit
unchanged from Phase 69. More decisively, frozen horizon telemetry found only
one active basin for 999,973 of 1,000,000 held-out bytes despite sixteen
available cells. The present organ is therefore a slow global structural bias,
not an episodic content address.

### Fast binding is distinct from slow context

The current working hypothesis separates two missing operations: **binding**
and **closed-loop reinstatement**. Slow state should describe reusable
structure; it should not permanently absorb the transient identities that fill
that structure. A fast sparse index can instead bind an active basin, a learned
structural role, and an existing cortical assembly:

\[
I_q \leftrightarrow (B,R,C).
\]

Later, the partial address \((B,R)\) should reactivate the index and reinstate
the same cortical assembly \(C\). This is consistent with the evidence for
largely separable human content and context populations with coordinated
reinstatement, and with hippocampal accounts in which rapidly constructed
indices compose reusable cortical building blocks. It remains an engineering
synthesis, not an established universal neural algorithm.

The roles must also remain modality-neutral. They are to emerge as sparse codes
for recurring temporal or causal relationships, not predefined linguistic
slots such as agent or object. The binder may use local coincidence,
competition, eligibility, decay, and reciprocal cortical connections, but it
must have no direct byte prediction head. Otherwise an apparent retrieval gain
could come from quietly duplicating the language predictor.

Three timescales are now explicit:

- slow basin: structural context spanning a trajectory;
- fast binder: one-shot, episode-scale role–filler association;
- durable cortex: reusable sensory and event assemblies.

The cheapest precursor assay corrupts evidence older than 128 bytes while
preserving the recent prefix, and compares normal inference, corrupted
inference, and corrupted inference with the normal teacher-forced basin trace
clamped in place. A complementary assay drives a freely generating cortex with
the basin trajectory recorded from the true continuation. These conditions
separate absent content storage from failure to maintain a useful state on the
model's own trajectory.

The decisive future intervention is a double dissociation: changing the basin
while holding fillers fixed should predominantly change continuation structure,
whereas changing a role–filler binding while holding the basin fixed should
predominantly change continuation identity. The scientific question for Phase
72 is therefore not whether the basin can remember more, but whether it can
retrieve the right old cortical assembly when that assembly becomes relevant.

The first sixteen-story assay falsified basin clamping as a sufficient repair.
Scrambling distant prompt order while retaining the most recent 128 bytes had
negligible effect over 4,096 scored continuation bytes. Restoring the intact
teacher basin recovered -0.0000035 bits, and teacher-basin-driven free running
changed none of sixteen sampled continuations. At this scale the basin is not a
lossy store from which content can be decoded; there is no content there to
reinstate. A fast reciprocal index is now the warranted next experiment.

Early Phase 72 tests further constrain the index. A query code based only on
temporal occupancy was too coarse, while a richer predictive-geometry address
still failed when it reinstated content without its original cortical receptor.
Restoring the assembly together with its temporal receptor passed a 100KB gate
by 2.21 bits but regressed by 7.48 bits at 1MB. Binding therefore requires both
content and relational placement, but those two quantities are not sufficient.

Three terms must not be conflated: the retrieval cue (R) determines when a
memory is requested, receptor (J) determines how the cortex interprets the
reinstated assembly, and a semantic role is not yet demonstrated. Likewise, a
structural basin is not an episode identifier. Distinct episodes should be able
to reuse the same basin; query state, recency, and episode validity must resolve
which occurrence is eligible. Basin-independent addressing and oracle episode
boundary invalidation test these two functions separately.

Exact causal readout precedes learned filler credit. From the same pre-read
state, compare selected/off, same-filler/wrong-receptor, and
alternative-filler/same-receptor branches using the final normalized byte
probability. Sampled reads from the rejected 1MB checkpoint were output-inert:
the first sixteen changed neither next-byte probability, a 16-byte observed
horizon, nor the transformed sampling distribution. Their cortical routes all
had negative utility and were gated off, although the binder proxy was positive
for every read. A wrong receptor produced the only nonzero rescue. The next
diagnostic must therefore supply an independently known correct assembly and
receptor before changing the filler learning rule.

Encoding and retrieval also require distinct provenance. Internally recalled
activity may bias prediction but must not refresh its sensory timestamp, create
durable route evidence, or write itself back as independent confirmation.
Generated bytes may enter transient episode state, but persistent usefulness
cannot be rewarded solely because a retrieval made its own sampled byte more
probable: that likelihood-ratio signal has nonnegative expectation by the KL
identity even without external correctness.

The controlled gate should use raw-byte episodes with two randomized
associations, long distractors, reassignment, and repeated structure carrying
different fillers. The evaluator—not the learner—knows the requested answer.
Only after a correct supplied memory demonstrably helps should learned filler
eligibility become the next target.

That controlled gate now passes. In a four-seed raw-byte task with two random
associations, seventy distractor motifs, reassignment, and repeated structure,
an evaluator-traced assembly through its selected receptor reduced identity
cost from 3.097993 to 3.078060 bpb over 400 queries. Moving the same content to
the adjacent receptor worsened it to 3.191533 bpb; substituting other content
was essentially baseline at 3.096832 bpb. The learned binder itself changed no
scored bits. Removing the structural basin from the binder address also had
zero effect, so the result does not support making the basin more episodic.

The next mechanistic correction is local directionality. The prototype binds a
distinctive prior motif as the filler. The task instead calls for active query
assemblies to become an address for the cortical assembly arriving now, with
later partial query activity retrieving that content through its learned
receptor. This is closer to address + content -> transient index than to
content-free structural clustering, and it can be tested before introducing a
learned delayed eligibility rule.

### Event capture and surprise-selected recall

Phase 72 exposed a temporal credit problem at encoding. Its raw-query binder
wrote 24,030 times over 100KB--approximately once per 4.2 bytes--but read only
1,596 times. Most write decisions therefore never received evidence about
whether the event mattered later. Predicting relevance hundreds of bytes into
an unknown future is a censored learning problem at least as difficult as the
prediction task the memory is intended to help.

Phase 73 replaces future-relevance gating with a cheaper primitive: detect that
the predictive regime changed. A rise in surprise derivative, temporal-route
disagreement, and route turnover opens an event; stabilization closes it into
a weak, decaying capsule. The slow basin is retained only as a contextual prior,
not as content or an episode identifier.

Exploratory recall is evaluated in a shadow cortical branch while the baseline
distribution remains authoritative. Once the observed byte arrives, the exact
final-probability comparison supplies signed local credit:

\[
u_t=\log_2\frac{P(x_t\mid\text{recall})}
                       {P(x_t\mid\text{no recall})}.
\]

Useful query-to-memory branches and their capsules strengthen, harmful ones
weaken, and unqueried capsules receive no invented negative evidence. This is
the same surprise-reduction life criterion already used by cortical branches,
not a separate semantic relevance classifier.

The initial prototype captured about 3,000 events per 100KB. Local content
matching raised the beneficial-read fraction from roughly 1.4% to 9.6%, and
the strongest match quintile reached 17.3%, but average utility remained
negative. Neither a slow learned-role/receptor gate nor a partial-motif-phase
gate ranked held-out utility monotonically. Four-candidate shadowing then found
that the candidate oracle was identical to the chosen read: different capsule
addresses usually reinstated the same assembly/receptor pair.

Branch-local surprise credit produced only thirty reusable exact-occurrence
reads after a 50KB calibration split. Its highest learned-value bin was
positive, but the population was far too sparse for a claim. The important
failure is representational: the capsule compresses an event into an unordered
top-assembly bag, erasing order, repetition, and occurrence identity before
selection begins.

The next correction is therefore a compact ordered cortical trace. A candidate
event should compete by how much its stored trace reduces mismatch with the
currently observed partial trace; that comparison selects a one-shot recall
without peeking at the future. Subsequent observed-byte savings remain the
survival credit. In biological terms, partial pattern completion proposes the
memory and sensory consequences decide whether its pathway lives.

The literal assembly-trace comparison was too brittle: almost every repeated
cue had 99.8--100% set mismatch, and a sparse stored/current output fingerprint
also failed to rank utility. The successful invariant was more direct. Each
candidate was transiently reinstated and measured by its immediate change in
the entropy of the fused cortical prediction, without looking at the observed
byte. Reads that increased entropy were strongly harmful (-0.185 bits/read in
the lowest bin); inert reads were exactly neutral; the strongest restorative
bin was slightly positive.

Applying the zero-threshold physiological gate sharpened the result. Among 722
held-out decisions, only 107 candidate recalls reduced cortical entropy. Those
reads saved 0.2072 bits in aggregate and 64.5% were individually beneficial.
This is the first Phase 73 selector to produce positive held-out causal utility.
The gain is small--about 0.0000041 bpb over the 50KB evaluation half--but the
sign and calibration pattern support the mechanism: candidate events compete
by immediate restoration of organization, while the later sensory outcome
still determines branch survival.

The detached 1MB transfer rejected entropy restoration as a sufficient gate.
On its 500KB held-out half, 1,090 entropy-reducing recalls lost 119.64 bits in
aggregate even though 56.0% were individually beneficial. Many small wins were
overwhelmed by a minority of large errors. Immediate organization is therefore
useful candidate evidence, but it is not synonymous with sensory correctness;
restorative magnitude needs calibrated downside control or conjunction with
learned branch survival before any recall reaches the real predictor.

### Risk-sensitive permission

Phase 74 separates proposal from trust. Positive entropy restoration may
propose a candidate, but it does not authorize cortical influence. A local
permission population instead accumulates, for each generic intervention
class, an observation count, mean exact utility, utility variance, and the
frequency of losses larger than a declared catastrophic threshold. Its
conservative value is

\[
V_{\rm safe}(x)=
\hat\mu(x)-k\sqrt{\frac{\hat\sigma^2(x)}{N(x)}}
-\lambda\hat r(x),
\qquad
\hat r(x)=P(u<-\tau\mid x).
\]

A class remains in shadow until it has sufficient evidence, positive
conservative value, and an acceptable catastrophic rate. The feature space is
modality-neutral: receptor, restoration band, distribution disruption, and
whether recall overturns the cortex's current winner. Proposal quality and
permission calibration are reported independently, including positive-read
fraction, mean utility, catastrophic-tail frequency, and worst-one-percent
utility.

The first 100KB pooled-feature smoke illustrates the intended behavior. The
proposal layer emitted 118 restorative candidates that were 61.0% helpful but
negative in aggregate. Risk-sensitive permission admitted only eight; all
eight helped, saving 0.0543 bits with a 0.00000109-bpb improvement.

The larger gates exposed a particularly simple veto. On first-megabyte
TinyStories evaluation, recalls that preserved the cortical winner were 64--67%
helpful, while winner-changing recalls were only 18--19% helpful and produced
most of the harmful tail. The preregistered operational correction therefore
keeps winner-changing proposals in shadow. On the untouched second megabyte,
TinyStories-only development admitted 57 reads: 87.7% helped, none crossed the
0.1-bit catastrophic threshold, and aggregate utility was +0.0330 bits. Mixed
TinyStories/SmolTalk development admitted 47: 89.4% helped, none catastrophic,
and aggregate utility was +0.0136 bits. The gains are too small to claim a new
language capability, but the permission mechanism generalized its safety
ordering to unseen bytes.

The two-arm comparison is informative. A TinyStories-developed permission bank
abstained completely on held-out SmolTalk. The mixed-development bank admitted
22 pooled-feature SmolTalk reads, 95.5% helpful and +0.00447 bits in aggregate.
Thus broader developmental exposure produced a small cross-domain trust signal
where single-corpus development safely abstained. Proposal streams remained
negative in every large gate: the result belongs to learned permission, not to
entropy restoration by itself.

Autonomous tests must freeze these statistics. Self-sampled bytes cannot supply
persistent utility credit because the expected on/off likelihood ratio is
self-confirming. Generation therefore permits transient event capture but
compares live permission against shadow-only permission with no persistent
permission updates in either arm.

The first frozen autonomous gate found a new limitation. Across 80,000 generated
bytes, 78 restorative proposals produced only three permitted interventions.
Their combined output-distribution displacement was 0.0000829 total variation
(maximum 0.0000670), and none changed a matched sampled byte. Permission is safe
but not yet behaviorally consequential.

An explicit capture-recency shortlist prior was then ablated at gains 0, 0.5,
1, and 2 with a 1,024-byte half-life. All four conditions produced exactly the
same 541 held-out proposals, seven permissions, and utility. Exact three-byte
cue matching usually leaves no alternative event for recency to reorder.
Recency may still matter biologically, but in this implementation it must enter
candidate formation (for example, as a prior over approximate cue matches), not
as a bonus applied after exact lookup. Time since event capture must remain
distinct from time since recall. Both can be useful: capture recency indicates
when an event occurred, while reinstatement recency indicates that an old event
recently re-entered the active control loop. Phase 75 therefore records both.
Only a recall permitted into cortex refreshes reinstatement recency; shadow
tests and denied proposals do not. This lets recalled information influence
subsequent action without allowing mere lookup attempts to manufacture their
own salience. Internally recalled assemblies also remain distinct from external
observations, although their emitted or sensed consequences can become new
event evidence normally.

On a fresh third-megabyte TinyStories comparison, no-recency retrieval produced
620 proposals and 37 permitted reads; 86.5% helped, none was catastrophic, and
aggregate utility was +0.00972 bits. Equal capture and reinstatement bonuses
produced 622 proposals but exactly the same 37 permitted reads and utility.
Thus both traces are retained as the correct causal variables, while their lack
of present leverage further localizes the bottleneck to exact candidate
formation rather than risk permission or recency gain.

### Keyless resonance and its representational requirement

Phase 76 tests the stronger biological proposal that no symbolic key or ranked
candidate list is necessary. The active cortical route pattern is expanded into
a fixed sparse high-dimensional code. Stored event-frame codes receive overlap
activation in parallel, capture and reinstatement recency enter additively
before softmax normalization, and the normalized posterior reconstructs
receptor-bearing cortical content. Retrieval precedes observation, while event
capture follows it, giving the prototype separate retrieval and encoding
phases. Basal prefix compatibility multiplies the reconstructed content before
its dendritic threshold, so an apical completion can affect only a locally
compatible receiving branch.

Controlled unit tasks confirm partial-cue completion, additive recall
excitability, address/content separation, and basal/apical coincidence. The
first TinyStories gates do not yet support replacing learned permission with
attractor confidence. A static bag-of-routes address generated about 4,889
completions per 20KB, but only about 4% helped and aggregate utility was roughly
-204 to -214 bits across tested inverse temperatures. Sparse conjunctive
frame codes did not repair that ordering. Preserving temporal direction by
heteroassociating each frame with its following frame reduced the 10KB beta-16
gate to 2,089 completions, but only 4.07% helped and aggregate utility was
-75.28 bits. Winner-changing completions were catastrophic 85.9% of the time;
even the highest attractor-confidence bin remained negative.

This is an important non-equivalence. Attractor confidence measures whether a
partial state lies near a stored basin; it does not establish that the basin is
appropriate for the present task or future. Modern Hopfield retrieval can be
the correct recall dynamics while still failing if the cortical representation
does not make relationally relevant states neighbors. The next scientific
problem is therefore learning a modality-neutral address geometry from local
transitions: similar future consequences should create overlap, while common
surface motifs and incompatible successors should separate. Risk-sensitive
permission remains necessary diagnostic scaffolding until intrinsic confidence
develops a positive held-out utility ordering.

### Sparse recurrent episodic field

The later Phase 76 prototype implements the stronger circuit proposal directly:
local Hebbian cortical-to-memory synapses allocate one sparse assembly per
captured event; all frames within that event bind into the same assembly;
recurrent Hebbian synapses stabilize it; heterosynaptic inhibitory connections
target competing units; and sparse memory-to-cortex synapses preserve the
original temporal receptor. Retrieval iterates feedforward excitation,
recurrent amplification, learned inhibition, global inhibitory sparsification,
adaptation, and divisive normalization. Capture and reinstatement excitability
are separate. Only an opened return refreshes reinstatement recency.

Several negative controls were essential. A recency bias larger than the unit
threshold activated recent memories without a cue and was rejected. Reusing an
old assembly when any one unit matched collapsed thousands of events into early
engrams and was replaced with one-assembly-per-event allocation. Dense
feedforward and recurrent matrices were replaced with sparse local synapses so
capacity could increase without quadratic storage. Finally, settled memory was
initially injected at full strength; the corrected return is continuous,

\[
x^+=(1-g)x+g\,x_{\rm recalled},
\qquad
g=\operatorname{clip}(\text{coherence}\times\text{cue overlap},0,1),
\]

after receptor-specific basal compatibility has gated the output.

These corrections materially reduced intrusion but did not yet create useful
episodic recall. A collision-free 20KB field used 3,992 units for roughly 499
event assemblies and produced 28 full-strength completions, only one helpful,
for -6.78 bits. Continuous returns on four independent 10KB shards produced
233 completions, 33.9% helpful, for -2.70 bits; catastrophic reads fell to 8.
A subsequent calibration/held-out split produced 124 held-out returns totaling
-1.70 bits. Risk-sensitive local permission admitted five, all effectively
zero-impact, preserving baseline exactly.

The recurrent field therefore solves a different problem from the missing one.
It can allocate, complete, inhibit, preserve receptor provenance, and control
return strength on synthetic partial-cue tasks. It cannot infer episodic
relevance when the supplied cortical route vector makes unrelated episodes with
similar local language form look alike. Recurrent completion cannot manufacture
information absent from its cue geometry. The next address representation must
include locally learnable episode-transition state or future-consequence
geometry, not merely concurrent motif routes.

### Dendritic commitment and conjunctive access

Phase 76b makes completion and permission one coupled circuit without allowing
an attractor to authorize itself. Recurrent assembly completion is basal
evidence. Independent, direct cortical input activates sparse apical branches,
each of which recognizes a locally learned feature conjunction. Their product
drives a slowly accumulating commitment population:

\[
c_m \leftarrow (1-\tau)c_m+\tau\,\sigma(
\alpha b_m a_m + \beta c_m - \gamma\sum_{j\ne m}c_j
-\kappa I_{\rm conflict}).
\]

Only committed assemblies disinhibit their own recurrent units and return
activity through the receptor on which the content was encoded. Recency remains
a subthreshold excitability bias. Recalled activity never enters the sensory
encoding path. This gives completion a local causal meaning while preserving
the possibility of safe abstention.

The live language probes added an important constraint absent from the toy.
Branches must preserve frame-local co-occurrence, use matched fractions rather
than absolute synapse counts, and receive a sufficiently rich cortical trace.
Even after adding lagged routes and raw sensory bytes, dozens of accumulated
language events often supported several branch populations. The shared
interneuron correctly vetoed them, so heldout TinyStories produced no returns
and heldout SmolTalk produced one output-neutral return in the first matched
2KB probes. Conjunctive commitment therefore appears to be a valid intrinsic
permission primitive, but it does not by itself decide which event features
deserve branch survival. Competitive branch plasticity and event allocation
are now the likely locus of learned selectivity.

### Competitive survival and residual dendritic access

A useful dendritic branch should preserve conjunctions that distinguish its
event, not every conjunction present during encoding. The competitive-survival
prototype implements a local approximation: branches grow from co-active
features, firing during another event weakens them heterosynaptically, and
branches below a survival threshold die so their territory can be reassigned.

In a sparse random 96-memory toy this roughly halved eligible competitors and
raised valid-cue commitment from 7.5% to 40.75% at 100% precision. In live
language-derived cortical activity it halved surviving branches but left a
median 11--12 memories eligible. The discrepancy shows that branch turnover
cannot compensate for a large shared correlated input subspace.

The next primitive subtracts a locally learned expectation before apical
conjunction formation:

\[
\bar x(t+1)=(1-\eta)\bar x(t)+\eta x(t),
\qquad
\tilde x=\max(0,x-\bar x).
\]

Branches see the positive residual while basal recurrent completion continues
to see the unmodified cortical state. A context-conditioned variant substitutes
\(\hat x(B)\), the expected activity under the slow predictive basin. This asks
which co-activity is surprising in context rather than merely which features
co-occur.

The live result is strong but incomplete. At background rate 0.001, median
eligible memories collapsed to one on both TinyStories and SmolTalk; faster
rates drove the median to zero. No setting yet produced committed recall, so
residual access has localized rather than finished the problem. The next test
is valid-cue retention under an adaptive local residual timescale. The target
is a narrow access field--roughly one to three plausible memories--whose
strongest member still supplies enough independent apical evidence to coincide
with basal completion.

The subsequent causal assay separated access strength from attractor quality.
At real-stream cues with one residual candidate, briefly activating only one
of its eight memory units and then removing the intervention produced complete,
winner-preserving autonomous settling in every tested TinyStories and SmolTalk
case. The recurrent field therefore already has a sharp basin boundary below
one full unit. The failure was in coupling: residual evidence selected an event
but did not seed it, while correlated raw cortical drive continuously excited
competitors.

The economical correction is not cortical reconstruction yet. A locally
suprathreshold residual branch supplies a one-step seed to its bound memory
assembly; subsequent completion is recurrent. Subthreshold branches neither
seed memory nor recruit the pooled conflict interneuron. The original raw cue
remains available as external evidence for coincidence/provenance, but recalled
or recurrent activity cannot manufacture that evidence. This preserves the
separation

\[
\text{external residual evidence}
\;\ne\;
\text{recurrently completed evidence}.
\]

The first live smoke crossed commitment threshold on both corpora only after
these changes. That result supports the seed-to-attractor coupling but does not
yet establish useful retrieval; larger heldout causal-utility gates remain
necessary.

## Primary Reading

- Schmitt et al. (2017), [Thalamic amplification of cortical connectivity
  sustains attentional control](https://www.nature.com/articles/nature22073).
- Palmigiano et al. (2025), [Multiplexed subspaces route neural activity across
  brain-wide networks](https://www.nature.com/articles/s41467-025-58698-2).
- Grothe et al. (2025), [Gamma-band synchronization between neurons in the
  visual cortex is causal for effective information processing and
  behavior](https://www.nature.com/articles/s41467-025-62732-8).
- Courellis et al. (2024), [Abstract representations emerge in human
  hippocampal neurons during inference](https://pubmed.ncbi.nlm.nih.gov/39143207/).
- [Distinct neuronal populations in the human brain combine content and
  context](https://www.nature.com/articles/s41586-025-09910-2) (2026).
- [A communication subspace in the human brain relays context-dependent action
  representations](https://www.nature.com/articles/s41593-026-02290-4) (2026).
- [Reusable modular architecture enables flexible cognitive operations in the
  mouse brain and artificial recurrent networks](https://www.nature.com/articles/s41593-026-02410-0)
  (2026).
- [The prefrontal cortex controls memory organization in the
  hippocampus](https://www.nature.com/articles/s41593-026-02231-1) (2026).
- [Distinct synaptic plasticity rules operate across dendritic compartments in
  vivo during learning](https://pubmed.ncbi.nlm.nih.gov/40245144/) (2025).
- [Vectorized instructive signals in cortical
  dendrites](https://www.nature.com/articles/s41586-026-10190-7) (2026).
- Shindou et al. (2019), [A silent eligibility trace enables dopamine-dependent
  synaptic plasticity for reinforcement learning in the mouse
  striatum](https://pmc.ncbi.nlm.nih.gov/articles/PMC6585681/).
- Bittner et al. (2017), [Behavioral time scale synaptic plasticity underlies
  CA1 place fields](https://pubmed.ncbi.nlm.nih.gov/28883072/).
- [Rapid formation of non-spatial hippocampal representations consistent with
  behavioral timescale synaptic plasticity is modulated by entorhinal
  input](https://www.nature.com/articles/s41467-026-71503-y) (2026).
- [Hippocampal ripples coordinate prefrontal-hippocampal compositional
  representations](https://www.nature.com/articles/s41593-026-02291-3) (2026).
- [Replay of procedural memory is independent of the
  hippocampus](https://www.nature.com/articles/s41593-026-02362-5) (2026).
- Skoge et al. (2014), [Cellular memory in eukaryotic
  chemotaxis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4210025/).
- Kramar and Alim (2021), [Encoding memory in tube diameter hierarchy of a
  living flow network](https://pubmed.ncbi.nlm.nih.gov/33619174/).

## Evidence Discipline

The references above support interacting mechanisms, not an experimentally
validated end-to-end theory of biological learning. Correlation, causal circuit
intervention, biological recording, and perturbation of a fitted artificial
model are different evidence classes and should not be collapsed. Future edits
should preserve those distinctions and attach primary sources before a claim
becomes an ALM design requirement.
