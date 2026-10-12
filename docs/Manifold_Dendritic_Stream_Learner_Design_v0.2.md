# Manifold Dendritic Stream Learner

**A Data-Agnostic Architecture for Continuous Multimodal Learning Without Backpropagation**

Status: Research Design v0.2
Target: Prototype -> continuous language -> multimodal stream learning
Initial scale target: Single-GPU experimentation
Primary constraint: No backpropagation through the representation system

This is the consolidated design doc for the line of work tracked as
"Dendritic Basin Learning" in `Dendritic_Basin_Learning_Design_v0.1.md`
through `v0.4` and the implementation journal
(`ManifoldFlow_ALM_Journal.md`). Those documents are the experimental
record that produced this design; this document is the target
architecture to scale toward TinyStories and beyond, aimed at a
**100M-token-equivalent run**.

Long-range coordination, compartment-specific teaching, delayed credit, and
their concrete ALM experiment gates are maintained separately in
[`ALM_Long_Range_Coordination_Reference.md`](ALM_Long_Range_Coordination_Reference.md).

---

## 1. Executive Summary

This project explores a learning architecture based on:

* nonlinear dendritic-style relation discovery;
* local structural plasticity;
* multi-timescale temporal traces;
* recursive relational assemblies;
* competitive persistent ManifoldFlow basins;
* local eligibility traces;
* low-bandwidth modulatory signals;
* prediction-driven self-supervision.

The architecture does not assume that its input represents language,
vision, audio, robotics, or any other particular modality.

Its fundamental interface is:

```
x_t in R^D
```

where `x_t` is simply the sensory/event vector observed at timestep `t`.

The intended long-term API is approximately:

```
state, prediction = model.step(vector)
```

The same learning machinery should operate whether `vector` contains:

* byte channels;
* image-patch values;
* frequency-band energies;
* accelerometer measurements;
* proprioceptive signals;
* environmental sensors;
* activations produced by previously learned assemblies.

The core research hypothesis is:

> **Useful representations can emerge by discovering information that
> inputs explain jointly but not individually.**

Rather than propagating detailed errors backward through a computational
graph, the system creates local candidate relations, remembers their
participation, measures their usefulness, and preserves structures that
improve prediction.

The learning loop is therefore:

> observe -> relate -> settle -> predict -> measure surprise -> select/adapt

---

## 2. Motivation

Backpropagation solves credit assignment by calculating `dL/dw` for
parameters throughout a computational graph.

Biological nervous systems instead appear to combine mechanisms such as:

* nonlinear dendritic computation;
* local synaptic plasticity;
* recurrent activity;
* eligibility traces;
* structural plasticity;
* neuromodulation;
* homeostasis;
* spontaneous activity;
* multiple temporal timescales.

The objective of this project is not to reproduce backpropagation through
another mathematical approximation. Instead, it asks whether useful
representation learning can emerge from:

> local interaction discovery + persistent local memory + structural
> exploration + dynamic competition + global selection

The architecture began with XOR because XOR exposes the weakness of
simple local correlation learning. The experiments subsequently revealed
a candidate general interaction-learning principle that appears useful
far beyond Boolean problems.

---

## 3. The XOR Observation

For XOR, `y = A XOR B`, neither parent individually contains predictive
information: `I(A;y) = 0`, `I(B;y) = 0`. The useful information exists in
their interaction. Flat Hebbian or correlation-based learners therefore
struggle because no first-order signal identifies the useful feature.

An early dendritic system solved XOR by explicitly memorizing complete
input configurations. That solution did not generalize: for `N`-bit
parity, complete-pattern dendrites required approximately `2^N`
representations. This was rejected as episodic memorization rather than
abstraction.

---

## 4. First Fundamental Failure: 3-Bit Parity

Sparse flat conjunctions fail immediately on 3-bit parity. Even providing
every possible one-bit and two-bit conjunction does not make `x1 XOR x2
XOR x3` linearly representable. This was not an optimization failure --
it was representational.

A recurrent nonlinear basin could technically introduce an extremely
small parity component, but sandbox tests found that component to be
numerically tiny and impractical to learn. Therefore:

> **recurrence alone does not solve hierarchical interaction discovery**

The architecture requires relations that can themselves become parents of
later relations.

---

## 5. Recursive Relation Assemblies

Allow `r_1 = f(A, B)` and then `r_2 = f(r_1, C)`. This changes parity
from an exponential pattern-memory problem into hierarchical composition,
e.g. `r_1 = A XOR B`, `r_2 = r_1 XOR C`.

The first implementation explicitly supplied SAME/DIFFERENT relations.
That worked extremely well but encoded too much prior knowledge. The next
experiment removed XOR entirely from the computational primitive.

---

## 6. Generic Nonmonotonic Dendrite

A dendritic unit receives local parent activities `A` and `B`. Its
response is:

```
d(A, B) = exp[ -(w_A*A + w_B*B - mu)^2 / (2*sigma^2) ]
```

Parameters `w_A, w_B, mu, sigma` control local response geometry. There
is no XOR operator, no AND operator, no Boolean truth table, no task
label, no hidden gradient. The dendrite is simply a nonmonotonic detector
over combined local drive.

---

## 7. Local Interaction Objective

The important sandbox result was that the dendrite did not require
explicit mutual-information calculations. A simple local moment
objective worked:

```
J = Var(d) - Cov(d,A)^2/Var(A) - Cov(d,B)^2/Var(B)
```

Conceptually: produce useful variation, but do not merely reproduce
either parent. This rewards information contained in the relationship
between the parents. For balanced binary parents, SAME/DIFFERENT
relations achieve approximately `J = 0.25`; the next-best Boolean
functions achieve only approximately `J = 0.0625`. This provides a strong
locally measurable quality gap.

---

## 8. Emergent SAME/DIFFERENT Relations

Without being given XOR, generic bell dendrites spontaneously discovered
two major response families.

**Intermediate-drive detector** -- fires when one parent is active
(`00->0, 01->1, 10->1, 11->0`): this implements DIFFERENT (`A XOR B`).

**Cancellation detector** -- fires when parents match (`00->1, 01->0,
10->0, 11->1`): this implements SAME (`A XNOR B`).

Both preserve the same underlying parity information up to inversion.
SAME/DIFFERENT appears to be a natural attractor of generic interaction
discovery rather than something that must be explicitly encoded.

---

## 9. Local Competition

One candidate dendrite is not sufficiently reliable. Instead, several
local candidate dendrites compete: each independently perturbs its
parameters and evaluates `J`; the best locally supported relation
survives. Sandbox tests using small competing populations produced highly
reliable SAME/DIFFERENT discovery.

A quality gate can reject candidates unless `J > theta_J`. For the
balanced binary test, `theta_J ~= 0.20` cleanly separated strong
interaction detectors from common failure modes.

---

## 10. Parity Scaling Result

Recursive trees composed from independently learned generic dendrites
were tested on parity. The hierarchy requires `N-1` relation units for
`N` input bits. Sandbox tests successfully propagated parity through
trees extending to 128, 256, 512, 1024 input bits in the clean synthetic
setting.

The important distinction is structural scaling: **O(N)** stored relation
units for a fixed binary tree rather than **O(2^N)** complete pattern
memories. This established recursive relational composition as a
necessary architectural component.

---

## 11. Candidate Mathematical Principle

The local objective `J = Var(d) - Cov(d,A)^2/Var(A) - Cov(d,B)^2/Var(B)`
rewards variation that cannot be explained by either parent
independently. For binary variables, the remaining irreducible
second-order interaction is closely related to `A'B'` under centered or
bipolar representations -- distinguishing `A=B` from `A!=B`.

The working principle is therefore:

> **Represent what the inputs explain together but not separately.**

This principle is the main reason the architecture may be data agnostic.
It does not refer to words, pixels, objects, phonemes, or labels -- only
to statistical relationships among signals.

---

## 12. From Static Variables to Streams

Real sensory information is temporal. The architecture therefore operates
continuously on `x_0, x_1, x_2, ...`. Rather than providing a fixed
context window, each incoming vector updates persistent internal state:

```
x_t -> traces -> relations -> assemblies -> basin state -> prediction
```

---

## 13. Universal Input Interface

The core learner accepts `x_t in R^D` and should make as few assumptions
as possible about the semantics of individual dimensions.

- **Language**: `x_t in {0,1}^256` for raw UTF-8 bytes.
- **Vision**: `x_t in R^P` for raw or lightly normalized local pixel patches.
- **Audio**: `x_t in R^F` for waveform samples or frequency-channel energies.
- **Robotics**: `x_t = [joint angles, velocities, touch, IMU, ...]`.

The relational core should not need to know which interpretation applies.

---

## 14. Multi-Timescale Temporal Traces

Each input signal maintains decaying traces:

```
q_i^(tau)(t+1) = lambda_tau * q_i^(tau)(t) + x_i(t)
```

Candidate timescales: `tau = 1, 2, 4, 8, 16, 32, 64, 128, ...`. This
produces a local representation of temporal history without storing a
conventional context window -- approximately: signal active now / very
recently / recently / moderately long ago / much longer ago. Temporal
relations can then form between `x_i(t)` and `q_j^(tau)(t)`.

---

## 15. Why Temporal Traces Matter

A current observation may be ambiguous: for a moving object, `P(x_{t+1} |
x_t)` may not reveal direction, but `P(x_{t+1} | x_t, q_t)` can. Likewise
in language, "alice ... sees" contains information unavailable from
"sees" alone. Traces provide inexpensive temporal context.

However, later experiments showed:

> **temporal context is not equivalent to binding memory**

and persistent basins are still necessary.

---

## 16. Event Variables Versus State Variables

An important failure appeared when the Boolean XOR machinery was
transferred directly to sparse language streams. For Boolean variables,
`A=0` is meaningful. For event channels, absence usually does not
constitute an event -- "Alice is absent" should not automatically create
a strong sensory relation. Therefore relation activation requires parent
support.

A simple support gate: `g(A,B) = min(A,B)`. Then `r_0 = g*d(A,B)` and
optionally `r_1 = g*(1-d(A,B))`. This prevents relations from firing
merely because both parents are inactive.

General rule: **no local drive, no dendritic event.**

---

## 17. Recursive Assemblies

Surviving dendritic relations become new signals: `r_k(t)` can become a
parent for another dendrite. The available signal pool becomes
`S_t = [x_t, q_t, r_t, h_t]`. A later relation can compute `r_j =
f(r_k, x_i)` or `r_j = f(r_a, r_b)`. This creates hierarchy without
defining fixed network layers:

```
raw signals
  -> first-order temporal relations
  -> relation assemblies
  -> relations between assemblies
  -> higher-order assemblies
```

Depth emerges from structural usefulness.

---

## 18. Structural Birth

New relation candidates should be proposed when existing structures fail
to explain incoming events -- e.g. when prediction surprise `S_t >
theta_S`. Candidate parents should be chosen using local criteria such
as: simultaneous activity, temporal proximity, complementary support, low
redundancy, high unexplained variance, existing eligibility. Candidate
dendrites are initialized locally and compete. No task-specific feature
enumeration should be required.

---

## 19. Structural Death

Strict one-pass language experiments exposed a critical issue: early
statistical noise can create overly specific structures that remain
forever. Structural birth must always be paired with structural death.

Each assembly maintains usefulness:

```
u_k(t+1) = lambda_u * u_k(t) + delta(predictive utility)
```

Structures with persistently poor utility are weakened, disconnected,
rewired, merged, or deleted:

> **representation = continuous birth + competition + survival + death**

---

## 20. Specificity Versus Generality

Character-stream experiments exposed another failure: the learner might
construct "ob sees" rather than "sees" because the longer structure is
more specific, but specificity without sufficient evidence harms
compositional generalization. A candidate scoring function:

```
S(A) = log(1 + support(A)) + lambda*I(A) - gamma*C(A)
```

where `support` measures observations, `I(A)` measures
integration/predictive utility, and `C(A)` penalizes unnecessary
complexity -- an Occam-like pressure locally. Extra structure must earn
its existence.

---

## 21. Persistent State and ManifoldFlow

Temporal traces answer "what happened recently?" -- not reliably "which
earlier event remains relevant?" This distinction appeared in the
distractor-language test: in "alice meets bob sees cat", the target is
associated with Alice even though Bob is more recent. Simple decay traces
became confused by recency and object accuracy fell substantially.
Persistent relational state restored correct binding. This defines the
role of ManifoldFlow.

---

## 22. ManifoldFlow State

Each useful assembly may influence or create an attractor basin:

```
B_j = (c_j, r_j, d_j, kappa_j, u_j)
```

where `c_j` = center, `r_j` = radius, `d_j` = depth, `kappa_j` =
curvature, `u_j` = utility. Current relational activity drives manifold
state: `h_{t+1} = h_t + dt * F(h_t, R_t)`. Relevant relational states can
therefore persist despite intervening sensory events.

---

## 23. Basin Depth Represents Certainty

Another smoke-test failure revealed that predictive relations should not
all have equal authority: some correlations are deterministic, others
are weak coincidences. Hard persistent relations overtrusted spurious
training correlations and degraded held-out prediction despite solving
the intended binding task.

The fix: relation/basin depth depends on residual uncertainty. For
relation `R`:

```
D_R proportional to exp(-H(Y|R) / tau)
```

Low conditional entropy (`H(Y|R) ~= 0`) produces a deep attractor; high
conditional entropy produces a shallow one. Prediction interpolates:

```
P(y) = (1 - D_R)*P_local(y) + D_R*P_R(y)
```

> **certainty controls influence**, rather than binary relation activation.

---

## 24. Three Quantities Must Remain Distinct

- **Activation** `a_k`: is this relation present now?
- **Utility** `u_k`: does this relation improve prediction?
- **Depth** `d_k`: how reliable and persistent is this relationship?

These should not be collapsed into a single scalar. A highly active
relation can still be unreliable; a highly useful relation may currently
be inactive; a deep relation should survive transient contradictory noise
better than a shallow one.

---

## 25. Eligibility Traces

Every plastic local structure maintains eligibility:

```
e_k(t+1) = lambda_e * e_k(t) + F(local activity)
```

When a prediction outcome becomes available, a low-bandwidth modulator
`M_t` gates local plasticity: `delta theta_k = eta * M_t * e_k(t)`. The
modulator does not contain `dL/dtheta_k` -- it simply communicates
whether recent internal activity was useful. This provides delayed local
credit without backpropagation through time.

---

## 26. Self-Supervised Prediction

A continuous stream automatically supplies its own target: at timestep
`t`, predict `P(x_{t+1} | state_t)`, then observe `x_{t+1}`. Prediction
surprise: `S_{t+1} = -log P(x_{t+1})`. This provides a universal learning
signal that works identically for next byte, next audio frame, next
visual state, next proprioceptive state, or next multimodal event.

> **future sensory prediction is the universal training task** for the
> initial architecture.

---

## 27. Prediction Readout

Each active assembly can maintain local predictive statistics. For
discrete output channels, `C_k[j]` tracks how often output `j` follows
assembly `k`:

```
C_k[j] <- lambda*C_k[j] + eta*a_k*1[x_{t+1}=j]
```

Active assemblies vote: `s_j = s_j_base + sum_k d_k*a_k*C_k[j]`, then
`P(j) = softmax(s)_j`. For continuous outputs, replace categorical tables
with local prototypes, mixture components, or predictive distributions.

---

## 28. Language Smoke Tests

The first language experiments intentionally used tiny vocabularies to
isolate architectural behavior (`alice->cat`, `bob->dog`, `carol->bird`,
`dave->fish`), with distractor entities and variable temporal delays.
Simple temporal relations learned basic context; persistent relations
restored identity binding through distractors. (Full results:
`Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`.)

---

## 29. Character-Level Smoke

Removing word IDs, the learner received raw characters and discovered
recurring structures corresponding to `alice, bob, carol, dave, sees,
meets, cat, dog, bird, fish` plus subword fragments -- demonstrating that
reusable higher-level assemblies can emerge from lower-level event
streams. Early versions used explicit substring statistics as a
convenient structural-discovery proxy; that mechanism was subsequently
removed because it violated the data-agnostic goal. (Full results:
`Dendritic_Basin_Learning_Design_v0.4_Character_Stream_Findings.md`.)

---

## 30. Strict One-Pass Requirement

The architecture must not perform an offline corpus analysis before
learning. At timestep `t`, it may use only: the current observation,
existing internal state, statistics accumulated before `t`, and current
local dynamics. Candidate structures may be born during the stream and
participate immediately. There is no separate tokenizer-training stage
and no offline feature miner.

---

## 31. Data-Agnostic Vector Smoke

The latest prototype removed substring counters, n-gram mining, word
boundaries, and language-specific candidate generation. The learner
operated only on vector channels, temporal traces, and generic relation
candidates. Two very different streams were tested with the same
mechanism:

**Raw character stream**: without relation units, 2.45 bits/character
and ~46.9% next-character accuracy. With generic temporal relations,
**1.51 bits/character** and **62.3%** accuracy.

**Moving visual sensor** (synthetic point on a 16-position ring, current
position alone ambiguous since direction depends on temporal history):
without relation units, 1.71 bits/event and 48.1% next-state accuracy.
With the same generic temporal relation machinery, **0.45 bits/event**
and **96.0%** accuracy.

The relation machinery was not changed between language and synthetic
vision. This supports the hypothesis that the system is learning
**relationships among changing signals** rather than language-specific
patterns.

---

## 32. Current Remaining Impurity

The latest data-agnostic smoke still uses generic coincidence-gated
current-channel x temporal-trace candidate relations -- more general than
the earlier language system, but not yet the final architecture. The
desired replacement: generic candidate relation -> self-organizing bell
dendrite. Next required milestone:

> arbitrary vectors -> temporal traces -> adaptive bell dendrites ->
> recursive assemblies, with no handcrafted relation enumeration.

---

## 33. Dense Continuous Vector Requirement

Sparse event channels are not sufficient for true multimodal learning.
The architecture must work when `x_t in R^D` is dense and continuous:
vision patches (`[R_1,G_1,B_1,...,R_n,G_n,B_n]`), audio frames
(`[E_f1,E_f2,...,E_fn]`), sensor state
(`[a_x,a_y,a_z,omega_x,omega_y,omega_z,...]`). Dendritic relation
discovery must generalize from binary/event parents to arbitrary scalar
or low-dimensional local vectors.

---

## 34. Continuous Dendritic Generalization

For parents `A, B in R`, retain `d(A,B) = exp[-(w_A*A + w_B*B -
mu)^2/(2*sigma^2)]`. For `K` local parents:

```
d(x) = exp[ -(w^T x - mu)^2 / (2*sigma^2) ]
```

where `w` is sparse. A dendrite therefore observes only a small subset
of available channels. Candidate parent counts: `K = 2, 4, 8, 16`. This
preserves locality and prevents dense matrix multiplication from
becoming the dominant primitive.

---

## 35. Sparse Connectivity

Each dendrite should observe `K << D`. Represent connectivity with sparse
indices (`parent_ids[k], weights[k], mu[k], sigma[k]`) rather than a
dense `W[num_dendrites, D]` matrix. The architecture should scale through
**sparse event-driven interaction** rather than repeatedly multiplying
dense global state matrices.

---

## 36. Candidate Selection

Blindly testing all pairs becomes `O(D^2)`, which will not scale to
large multimodal streams. Candidate parents should be proposed from
sparse active sets via mechanisms such as: locality-sensitive hashing,
recent coactivity, shared eligibility, common basin occupancy, temporal
coincidence, unexplained covariance, random exploration, or low-overlap
assembly pairing. The candidate-generation mechanism must itself remain
modality independent.

---

## 37. Structural Exploration

Dendritic parameters can initially adapt through local perturbation. For
parameter vector `theta_k = (w_k, mu_k, sigma_k)`, generate `theta'_k =
theta_k + epsilon`, evaluate `J(theta'_k)`, and retain perturbations that
improve `J`. This is slow but extremely useful for proving the learning
principle without gradients. Later implementations may use more
efficient biologically local approximations.

---

## 38. Homeostatic Exploration

Finite-sample XOR experiments revealed that interaction learning fails
when important local states are never experienced. This suggests a role
for spontaneous activity, replay, local noise, pseudocounts, and
homeostatic activation. A tiny prior over unobserved states dramatically
improved relation discovery in synthetic tests.

> **local state-space coverage -> better interaction discovery**

---

## 39. Language Scaling Experiment

Once the adaptive generic dendrite passes dense-vector smoke tests, move
to continuous language. Initial input: `x_t in {0,1}^256` for raw UTF-8
bytes. No tokenizer, no embeddings, no context window, no attention, no
frozen weights, no backpropagation. Train continuously on a text stream.
TinyStories is convenient for debugging but must not be architecturally
privileged -- the same system must accept Wikipedia or arbitrary text
without modification.

---

## 40. TinyStories Is a Dataset, Not an Assumption

The architecture must not contain: story boundaries required for
operation, child-language vocabulary assumptions, fixed sentence
templates, TinyStories-specific tokenization, or domain-specific
structural priors. A story/document boundary may be provided as another
event channel if naturally present, but learning cannot depend on it.
The core learner must function on TinyStories, Wikipedia, source code,
conversation, news, and sensor logs through the same vector-stream
interface.

---

## 41. Language Evaluation

Primary metric: **bits per byte**, tracked continuously during learning.
Compare against unigram, bigram, trigram, fixed n-gram baselines, and
simple recurrent statistical models.

Secondary metrics: next-byte accuracy, assembly count, assembly reuse,
assembly lifetime, hierarchy depth, predictive utility, basin lifetime,
structural birth/death rate. Generation is useful but should not replace
quantitative evaluation.

---

## 42. Representation Inspection

Periodically probe long-lived assemblies: for each assembly `A_k`,
collect stream segments producing maximal activation. If hierarchy
emerges, expected progression may resemble: character fragments ->
common character sequences -> word-like structures -> phrase-like
structures -> entity relations -> persistent semantic states. These
labels are not supplied to the learner -- they are post-hoc
interpretations. If such a hierarchy emerges autonomously, it provides
strong evidence of representation formation.

---

## 43. Vision Generalization

The exact same core should be tested on a visual stream. Initial
synthetic tasks: (1) moving point, (2) moving bars, (3) occlusion, (4)
object permanence, (5) multiple moving objects, (6) identity after
crossing trajectories. The learner receives vectors rather than symbolic
object labels, e.g. `x_t = flatten(8x8 patch)` or a sparse sequence of
local patch vectors. Temporal relations should discover motion without a
predefined motion feature; persistent basins should preserve identity
through occlusion.

---

## 44. Audio Generalization

Audio can enter as `x_t in R^F` (frequency energies). A fixed
Fourier/cochlear transform is acceptable as sensory transduction because
it does not contain semantic knowledge. The relational core should then
discover recurring spectral combinations, temporal transitions,
phoneme-like structures, syllable-like structures, and
speaker-dependent relations -- hoped-for emergent structures, not
supplied categories.

---

## 45. Multimodal Binding

Once individual modalities work, concatenate or otherwise expose their
event channels to the same relational system: `x_t = [x_t^vision,
x_t^audio, x_t^language, x_t^proprioception]`. No special cross-modal
attention mechanism should initially be required -- a dendrite can simply
connect parents originating from different sources, e.g. `d(visual
feature, audio feature)`. Recursive assemblies can then bind visual
object + spoken pattern + persistent state. The learning rule remains
unchanged.

---

## 46. Source Metadata

For practical multimodal systems, each event may include `e_t = (v_t,
s_t, delta_t, l_t)` where `v_t` = values, `s_t` = source identity,
`delta_t` = timing, `l_t` = optional locality information. Source and
locality should be treated as additional structural information rather
than semantic labels -- e.g. vision locality can indicate neighboring
sensor positions, audio locality can indicate frequency adjacency. These
provide topology without providing meaning.

---

## 47. Unified Architecture

```
arbitrary vector stream
        |
        v
normalization/events
        |
        v
multi-timescale traces
        |
        v
sparse nonmonotonic dendrites
        |
        v
local interaction discovery
        |
        v
recursive assemblies
        |
        v
structural competition/pruning
        |
        v
ManifoldFlow basin state
        |
        v
future-event prediction
        |
        v
observed future
        |
        v
surprise/modulatory signal
        |
        `---------------------> local eligibility
```

---

## 48. Online Processing Loop

Conceptual implementation:

```python
def step(x):
    # Observe arbitrary vector.
    x = normalize(x)
    # Evaluate prediction from previous state.
    surprise = predictor.observe(x)
    # Update temporal traces.
    traces.update(x)
    # Build sparse pool of currently relevant signals.
    signals = collect_active(
        x, traces, assemblies, manifold.state,
    )
    # Existing dendrites respond.
    relations = dendrites.activate(signals)
    # Local statistics and eligibility update.
    dendrites.observe_local_state(signals, relations)
    # Allow recursive assemblies.
    assemblies.update(relations)
    # Update persistent manifold.
    manifold.step(relations, assemblies)
    # Modulate eligible structures.
    dendrites.modulate(surprise)
    assemblies.modulate(surprise)
    manifold.modulate(surprise)
    # Structural birth/death.
    structure.adapt(signals=signals, surprise=surprise)
    # Predict future vector/event.
    prediction = predictor.predict(
        signals, relations, assemblies, manifold.state,
    )
    return prediction
```

There is no backward pass.

---

## 49. Hard No-Backprop Test

The first implementation should use NumPy or explicitly detached
tensors. A test should assert that `requires_grad == False` for every
state. No code path may call `backward()`, `grad()`, or `autograd()`.
The experiment is invalid if hidden representations are trained through
gradients.

---

## 50. Computational Architecture

A scalable implementation should avoid globally dense computation. Target
complexity should resemble `O(K_active * K_local)` rather than `O(D^2)`
or dense transformer-style `O(T^2 * D)`. Use: sparse active lists, hashed
candidate routing, local dendritic parent sets, bounded candidate
populations, bounded assembly populations, sparse basin lookup,
event-driven updates. The architecture should ideally become cheaper when
the sensory stream is sparse.

---

## 51. Memory Budget

Use bounded populations. Example initial GPU-scale configuration:

| Component | Range |
|---|---|
| raw channels | variable |
| temporal timescales | 8-16 |
| active trace candidates | bounded |
| dendritic units | 32K-256K |
| parents/dendrite | 2-8 |
| persistent assemblies | 8K-64K |
| MF basins | 8K-100K |
| eligibility traces | sparse |
| prediction tables | sparse/top-K |

These are experimental ranges, not architectural constants.

---

## 52. Consolidation

Online learning should be complemented by optional consolidation. During
low-input periods: merge redundant assemblies, replay recent rare
relations, prune low-utility structures, split overloaded structures,
adjust basin centers, decay unused relations. This is analogous to the
existing ManifoldFlow sleep-consolidation concept. Importantly,
consolidation still uses local stored statistics rather than
backpropagation.

---

## 53. Major Failure Modes Already Identified

| Failure mode | Symptom | Fix |
|---|---|---|
| Exponential episodic memorization | One branch per complete input | Sparse reusable relations and recursive composition |
| Flat representation ceiling | 3-bit parity fails despite abundant pairwise conjunctions | Relation-on-relation hierarchy |
| Accidental recurrent representability | Recurrence technically contains the target direction but at negligible amplitude | Explicitly discover useful interactions rather than hoping recurrent dynamics manufacture them |
| Ghost activity | Absent event channels create strong relations | Support-gated dendritic events |
| Structural immortality | Early noisy assemblies survive forever | Utility decay and pruning |
| Overspecific assemblies | "ob sees" defeats reusable "sees" | Evidence-weighted complexity competition |
| Recency confusion | Recent distractor overwrites relevant identity | Persistent MF binding |
| Spurious deterministic confidence | Accidental training correlations dominate prediction | Entropy-controlled basin depth |
| Insufficient local state coverage | Interaction cannot be discovered because some combinations are never experienced | Homeostatic exploration/replay |

---

## 54. Experimental Roadmap

- **Phase 0: XOR** -- Complete. Demonstrated generic nonlinear interaction discovery.
- **Phase 1: Recursive parity** -- Complete in clean synthetic tests. Demonstrated compositional relation hierarchy.
- **Phase 2: Limited-vocabulary stream** -- Complete. Demonstrated temporal context and prediction.
- **Phase 3: Distractor binding** -- Complete in synthetic tests. Demonstrated need for persistent state.
- **Phase 4: Raw character stream** -- Complete as a structural smoke test. Demonstrated emergence of reusable character assemblies.
- **Phase 5: Generic vector relation learner** -- In progress. Language-specific candidate machinery removed; initial character and synthetic visual streams both improve with the same relation machinery.
- **Phase 6: Adaptive bell dendrites on generic vectors** -- Next critical milestone. Replace generic enumerated relation candidates with fully self-organizing dendritic populations.
- **Phase 7: Dense continuous vectors** -- Test arbitrary continuous streams. No sparse categorical assumptions.
- **Phase 8: Larger raw language stream** -- Move to TinyStories and then unrelated text such as Wikipedia.
- **Phase 9: Visual world** -- Moving objects, persistence, occlusion, identity.
- **Phase 10: Audio** -- Continuous frequency vectors.
- **Phase 11: Multimodal stream** -- Vision + audio + language using the same relational core.

---

## 55. Immediate Rig Experiment

Once Phase 6 passes the sandbox smoke, create a GPU implementation. Start
with raw UTF-8 TinyStories because language provides an easy quantitative
evaluation. Feed 1-10 MB continuously for the first run -- **do not
shuffle into conventional training batches unless batching can preserve
independent stream state.** Measure bits/byte every fixed number of
observed bytes. Track structural metrics simultaneously.

(Scale target for this line of work, per the user: a 100M-token-equivalent run.)

---

## 56. Cross-Dataset Test

TinyStories success alone is insufficient. Without changing architecture
or learning rules, continue or restart the learner on Wikipedia text,
source code, and conversational text. The important question is not
whether every domain reaches identical compression -- it is whether the
same interaction-learning machinery continues discovering predictive
structure without dataset-specific modification.

---

## 57. Data-Agnostic Success Criterion

The first major research hurdle is passed only if the same implementation
can improve prediction on at least three structurally different streams:
**symbolic** (raw text), **spatial/temporal** (visual vector stream), and
**continuous temporal** (audio or synthetic continuous sensor stream). No
modality-specific representation-learning rule may be introduced -- only
the sensory adapter may differ.

---

## 58. Stronger Multimodal Criterion

A stronger result is achieved if cross-modal assemblies emerge
automatically. For synchronized vision and audio, `R = f(V_i, A_j)`
should survive when the relationship improves future prediction. No
explicit contrastive objective should be necessary -- the architecture
should discover that the modalities are related because their joint
structure reduces uncertainty.

---

## 59. Core Hypothesis

Three nested hypotheses:

- **Hypothesis A (Interaction discovery)**: locally maximize information
  unexplained by individual parents -- creates nonlinear relational
  features.
- **Hypothesis B (Recursive composition)**: allow discovered relations to
  become new sensory variables -- creates hierarchical abstraction.
- **Hypothesis C (Persistent predictive selection)**: preserve relations
  in proportion to their predictive usefulness and certainty -- creates
  memory, binding, and world state.

> interaction + recursion + persistence -> candidate general stream learner

---

## 60. What Would Falsify the Approach?

The architecture should be considered unsuccessful or incomplete if:

1. generic adaptive dendrites cannot reproduce the current relation-candidate results;
2. dense continuous inputs require modality-specific learning rules;
3. structural populations grow exponentially with stream complexity;
4. recursive assemblies collapse into episodic memorization;
5. prediction improves only on repetitive synthetic streams;
6. TinyStories requires explicit tokenization or language-specific heuristics;
7. Wikipedia requires architectural changes not required by TinyStories;
8. visual or audio streams require fundamentally different relation-learning mechanisms;
9. persistent basins fail to preserve identity under interference;
10. local learning cannot remain computationally practical at useful scale.

These are intended as genuine falsification criteria rather than tuning targets.

---

## 61. What Would Constitute a Major Result?

A compelling initial result would be a single implementation that:
receives arbitrary vectors; trains continuously; uses no
backpropagation; uses no pretrained representations; grows its own
sparse relational hierarchy; predicts future observations increasingly
well; maintains persistent state; generalizes across unseen
compositions; operates unchanged across language and non-language
streams.

A particularly strong demonstration: raw bytes -> language prediction
improves; same learner, moving visual vectors -> motion prediction
improves; same learner, audio vectors -> audio prediction improves;
followed by vision + audio + language -> cross-modal assemblies emerge.

At that point the system would no longer be meaningfully described as an
alternative language model. It would be a candidate general online
representation-learning substrate.

---

## 62. Final Architecture

```
x_t -> Q_t -> D_t -> R_t -> A_t -> M_t -> x_hat_{t+1}
```

where:

- `x_t` = arbitrary sensory vector
- `Q_t` = multi-timescale temporal traces
- `D_t` = sparse generic nonlinear dendrites
- `R_t` = locally discovered interactions
- `A_t` = recursive relational assemblies
- `M_t` = persistent ManifoldFlow state
- `x_hat_{t+1}` = predicted future sensory state

Learning occurs through: local statistics + local perturbation +
eligibility + predictive utility + structural selection + homeostasis --
rather than **gradient transport**.

The central research question is therefore no longer "Can a neural
system learn XOR without backpropagation?" It is:

> **Can a continuous learner autonomously construct a hierarchy of
> predictive relational variables from arbitrary sensory vectors using
> only local interaction discovery, structural plasticity, and
> persistent dynamical state?**

If the answer remains yes across raw language, continuous vision, and
audio, then the same primitive that discovered XOR may provide a path
toward a genuinely modality-independent learning system.
