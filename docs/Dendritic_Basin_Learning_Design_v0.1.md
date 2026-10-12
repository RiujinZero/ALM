# Dendritic Basin Learning

**A Local-Plasticity, Recurrent-Basin Architecture for Learning Without Backpropagation**

Status: Research Design v0.1

## Primary Goal

Build a learnable neural architecture that can discover nonlinear representations, maintain persistent internal state, and solve temporal credit-assignment problems without:

* backpropagation
* automatic differentiation
* frozen pretrained feature extractors
* globally computed hidden-layer gradients
* end-to-end gradient optimization

The system should learn through four primary mechanisms:

1. Local nonlinear dendritic feature formation
2. Competitive recurrent basin dynamics
3. Eligibility traces
4. Low-bandwidth global modulatory signals

The initial proof problem is XOR.

The architecture should then progress through increasingly difficult parity, hierarchical, and temporal tasks before attempting language modeling.

---

## 1. Motivation

Standard neural networks typically solve nonlinear problems through learned hidden representations optimized by backpropagation.

For XOR:

`x = (x_1, x_2)`

`y = x_1 XOR x_2`

neither input individually predicts the target:

`I(x_1; y) = 0`

`I(x_2; y) = 0`

The useful information exists only jointly:

`I((x_1, x_2); y) > 0`

A purely local first-order Hebbian learner therefore receives no useful marginal learning signal.

Backpropagation solves this by transmitting output error through the computational graph so hidden representations can gradually discover useful conjunctions.

The proposed system takes another route.

Rather than transmitting detailed error information backward, it allows local circuits to:

1. spontaneously construct nonlinear feature combinations;
2. preserve their own recent participation using eligibility traces;
3. compete through recurrent dynamics;
4. receive a low-dimensional global signal indicating success, failure, novelty, or surprise;
5. reinforce locally useful structures.

The intended learning principle is:

> Propose locally -> Settle dynamically -> Remember participation -> Reinforce useful structures

rather than:

> Forward -> L -> gradient(L) -> Backward

---

## 2. Core Architecture

The initial architecture is:

`x -> D(x) -> B(D, x) -> h* -> y`

where:

* `x` is the current input,
* `D(x)` is a population of dendritic feature proposals,
* `B` is a competitive recurrent basin system,
* `h*` is the settled internal state,
* `y` is the output.

Learning occurs through local state attached to each component.

Conceptually:

```
Input
  |
  v
Local nonlinear dendritic mixtures
  |
  v
Candidate feature activations
  |
  v
Competitive recurrent basin
  |
  |-- recurrent excitation
  |-- lateral inhibition
  |-- persistent state
  `-- attractor settling
  |
  v
Settled latent state
  |
  v
Output
  |
  v
Prediction / behavior
```

Learning signals travel through a separate low-bandwidth pathway:

```
Local activity ---> eligibility traces
                         |
                         v
Output result ---> global modulator
                         |
                         v
               local plasticity
```

No gradient is propagated backward through the network.

---

## 3. Design Constraints

The architecture must satisfy the following constraints.

### 3.1 No backpropagation

Training code must not call:

```
loss.backward()
```

or any equivalent reverse-mode differentiation mechanism.

Autograd should ideally be disabled entirely during early experiments.

NumPy is acceptable for the first implementation to make accidental gradient use impossible.

### 3.2 No frozen intelligent encoder

The system must not depend on:

* pretrained language embeddings,
* CLIP,
* frozen transformers,
* pretrained autoencoders,
* manually engineered task-specific representations.

Simple deterministic transformations such as binary literals are acceptable.

The architecture itself must discover its reusable internal features.

### 3.3 Local updates only

A synaptic update should depend primarily on information physically available to that connection or its immediate computational neighborhood.

Examples include:

* presynaptic activity,
* postsynaptic activity,
* dendritic branch activation,
* local voltage/state,
* local eligibility,
* a broadcast scalar modulator.

Hidden parameters must not receive exact gradients derived from distant outputs.

### 3.4 Persistent internal state

The recurrent subsystem must retain state across timesteps.

This becomes essential for:

* temporal XOR,
* delayed binding,
* sequential reasoning,
* language.

---

## 4. Component A: Input Literal Encoding

For binary experiments, explicitly encode both a variable and its complement.

For `x = (x_1, x_2, ..., x_N)` construct:

`z = (x_1, 1-x_1, x_2, 1-x_2, ..., x_N, 1-x_N)`

Example: `x = (1, 0)` becomes `z = (1, 0, 0, 1)`.

This allows dendritic branches to represent combinations such as `x_1 AND NOT x_2` using purely excitatory feature matching.

This transformation is not learned. It simply exposes positive and negative literals to local feature-building mechanisms.

For continuous inputs, this will later be replaced by learned local receptive fields.

---

## 5. Component B: Dendritic Feature Units

A dendritic unit represents a nonlinear conjunction over a sparse subset of inputs.

Each branch `k` stores: `D_k = { p_k, m_k, e_k, u_k, a_k }` where:

* `p_k`: local feature prototype,
* `m_k`: sparse connectivity mask,
* `e_k`: eligibility trace,
* `u_k`: usefulness estimate,
* `a_k`: current branch activation.

---

## 6. Sparse Dendritic Connectivity

A branch must not observe the full input once experiments move beyond toy XOR.

Instead `m_ki in {0,1}` specifies whether branch `k` is connected to feature `i`.

The number of synapses per branch should remain small: `K_syn << N_input`.

Initial values to test: 4, 8, 16, 32 connections per branch.

This turns a branch into a detector for a reusable partial conjunction rather than a memorized complete input.

Example: `D_7 = x_3 AND NOT x_11 AND x_42`.

---

## 7. Dendritic Activation

For branch `k`, compute local prototype mismatch:

`delta_k = sum_i m_ki * |z_i - p_ki| / (sum_i m_ki + eps)`

Branch activation: `d_k = exp(-beta * delta_k)` where `beta` controls sharpness.

Alternative nonlinearities should later be tested:

**Product detector**: `d_k = prod_{i: m_ki=1} q(z_i, p_ki)`

**Thresholded coincidence**: `d_k = 1[ sum_i m_ki * q_i >= theta_k ]`

**NMDA-like supralinearity**: `d_k = sigmoid(alpha*(s_k - theta)) + gamma*s_k^2`

The exact biological analogy is secondary. The computational requirement is that dendritic branches can detect nonlinear combinations locally.

---

## 8. Structural Plasticity

The system must be capable of creating features that did not exist at initialization. This is central to the design.

A new branch may be created when `|surprise| > theta_s` and `max_k d_k < theta_m`, meaning: the system made a significant error and no existing feature adequately represented the current local pattern.

Pseudo-rule:

```
if abs(modulator) > surprise_threshold:
    if max(dendrite_activity) < representation_threshold:
        grow_branch(current_input)
```

The new branch should connect to a sparse random or locally selected subset of active literals.

---

## 9. Synapse Growth and Pruning

Each dendritic branch should be structurally mutable.

New connection: `mask[k, i] = 1`. Pruned connection: `mask[k, i] = 0`.

Candidate synapses can be added when a feature repeatedly co-occurs with branch activity. A local correlation trace can be maintained:

`c_ki(t+1) = lambda_c * c_ki(t) + d_k(t) * z_i(t)`

Features with high local correlation become candidate synapses. Features that fail to improve branch usefulness over time can be pruned.

This gives the system a way to discover higher-order interactions without enumerating all possible feature combinations.

---

## 10. Component C: Competitive Recurrent Basin

Dendrites only propose candidate features. A second system determines which combination of those proposals becomes the active internal representation.

Let `h_t in R^K` represent recurrent basin state.

Dynamics:

`h_{t+1} = phi( D(x) + alpha*W_rec*h_t - gamma*C(h_t) - theta )`

where:

* `D(x)` is dendritic drive,
* `W_rec` contains recurrent associations,
* `C(h_t)` implements competition,
* `alpha` controls recurrence,
* `gamma` controls inhibition.

A simple global competition term is `C_i(h) = sum_{j != i} h_j`.

The network runs multiple internal iterations:

```
h = dendritic_drive
for step in range(settle_steps):
    recurrent = W_rec @ h
    inhibition = h.sum() - h
    h = activation(
        dendritic_drive
        + recurrent_gain * recurrent
        - inhibition_gain * inhibition
    )
```

The final state `h* = h_T` is used by the output system.

---

## 11. Why the Recurrent Basin Exists

The recurrent subsystem provides several capabilities the dendrites alone lack:

1. competition between alternate interpretations;
2. associative completion;
3. persistence over time;
4. temporal context;
5. binding between current input and previous state;
6. attractor formation;
7. noise resistance.

For static XOR, recurrence may contribute little. For temporal XOR, recurrence becomes essential.

---

## 12. Recurrent Plasticity

Recurrent connections use local Hebbian plasticity.

Basic rule with decay: `W_ij <- (1 - lambda_r) * W_ij + eta_r * h_i * h_j`, with `W_ii = 0`.

Optional competitive normalization can constrain runaway attraction: `W_ij <- W_ij / (eps + sum_j W_ij)`.

A later implementation should test anti-Hebbian inhibition between competing assemblies.

---

## 13. Component D: Output Readout

For binary classification: `y_hat = sigmoid(W_y . h* + b)`.

The output neuron has access to its inputs, its own activation, and the provided target. Therefore its learning rule can be local.

For a binary target: `M = y - y_hat`, then `Delta W_y,i = eta_y * M * h_i`.

This resembles a delta rule but does not require propagating derivatives into upstream layers. The output layer alone receives the explicit target.

---

## 14. Global Modulatory Signal

The architecture should distinguish between an exact mathematical gradient and a low-bandwidth global teaching signal.

Define `M(t)` as a scalar or small vector indicating: prediction surprise, reward prediction error, success, failure, novelty, or confidence mismatch.

For initial supervised experiments: `M(t) = y - y_hat`.

Later versions may use `M(t) = L_{t-1} - L_t` or `M(t) = r_t - r_hat_t`.

The critical restriction is that `M` does not contain per-parameter derivatives. Every receiving synapse gets the same low-bandwidth signal.

---

## 15. Eligibility Traces

Each plastic connection maintains a decaying memory of recent participation.

For dendritic synapse `i` on branch `k`: `e_ki(t+1) = lambda_e * e_ki(t) + F(z_i, d_k)`.

A simple version: `e_ki(t+1) = lambda_e * e_ki(t) + z_i(t) * d_k(t)`.

Weight or prototype changes occur when modulation arrives: `Delta p_ki = eta_d * M(t) * e_ki(t)`.

Conceptually:

```
local event
   |
   v
eligibility mark
   |
   |---- time passes ----.
   |                      |
   v                      v
stored participation   outcome
                          |
                          v
                     modulator
                          |
                          v
              reinforce eligible sites
```

This allows delayed outcomes to affect earlier local events without explicitly tracing a computational graph backward through time.

---

## 16. Prototype Learning

A branch whose activation contributes positively to an outcome should move toward the observed pattern.

For a winner branch `k`: `p_k <- p_k + eta_p * M+ * d_k * m_k * (z - p_k)` where `M+ = max(M, 0)`.

A negative outcome may instead decrease usefulness, weaken specific synapses, increase branch threshold, or encourage structural mutation.

Directly moving the prototype away from failed examples should be tested separately because it may destabilize sparse representations.

---

## 17. Branch Usefulness

Each branch maintains a long-term usefulness score: `u_k(t+1) = lambda_u * u_k(t) + M(t) * d_k(t)`.

Branches with persistently low or negative usefulness may be weakened, rewired, split, or deleted.

This provides selection pressure among locally proposed features. The system therefore performs a form of continuous structural search.

---

## 18. Branch Splitting

If a branch repeatedly activates for incompatible outcomes, it may represent too broad a region.

Detect high conditional error: `Var(M | d_k > theta) > theta_v`.

Then split `D_k -> D_k1, D_k2`. Each child receives most parent synapses, several mutated connections, and slightly altered thresholds or prototypes. The more predictive child survives.

This could become an important replacement for hidden-layer gradient specialization.

---

## 19. Complete Learning Cycle

For each input:

1. **Encode**: `x -> z`
2. **Evaluate dendritic branches**: `z -> d`
3. **Run recurrent settling**: `d, h_{t-1} -> h*`
4. **Produce output**: `h* -> y_hat`
5. **Compute global modulator**: `M = y - y_hat`
6. **Update eligibility**: each active local synapse records participation.
7. **Modify output connections**: `Delta W_y ~ eta_y * M * h*`
8. **Update recurrent connections**: `Delta W_rec ~ h* h*^T`
9. **Apply modulatory dendritic updates**: `Delta p ~ M * e`
10. **Structural adaptation**: grow a dendrite if surprised with no matching representation; prune/mutate a dendrite that is consistently useless; split a dendrite that is overloaded.

---

## 20. Reference Implementation Skeleton

```python
class DendriticBasinSystem:
    def __init__(self):
        self.dendrites = DendriticPopulation()
        self.basin = RecurrentBasin()
        self.readout = LocalReadout()

    def step(self, x, target=None):
        # Encode locally.
        z = self.encode(x)
        # Propose nonlinear local features.
        d = self.dendrites.activate(z)
        # Settle into recurrent state.
        h = self.basin.settle(d)
        # Generate prediction.
        y_hat = self.readout.predict(h)
        if target is not None:
            # Low-bandwidth global teaching signal.
            modulator = target - y_hat
            # Record local participation.
            self.dendrites.update_eligibility(z, d)
            self.basin.update_eligibility(h)
            # Output-local learning.
            self.readout.learn(h, modulator)
            # Recurrent-local learning.
            self.basin.learn(h)
            # Modulated local dendritic learning.
            self.dendrites.learn(modulator)
            # Structural plasticity.
            self.dendrites.adapt_structure(
                z=z, activity=d, modulator=modulator
            )
        return y_hat
```

---

## 21. Minimal Dendritic Population

```python
class DendriticPopulation:
    def activate(self, z):
        difference = abs(self.prototype - z[None, :])
        masked = difference * self.mask
        distance = masked.sum(axis=1) / (self.mask.sum(axis=1) + 1e-8)
        activity = exp(-self.beta * distance)
        return activity
```

---

## 22. Eligibility Update

```python
def update_eligibility(self, z, activity):
    self.eligibility *= self.trace_decay
    local_coactivity = activity[:, None] * z[None, :] * self.mask
    self.eligibility += local_coactivity
```

Modulated update:

```python
def learn(self, modulator):
    self.prototype += (
        self.learning_rate * modulator * self.eligibility * self.mask
    )
```

This should later be replaced by a bounded prototype update to prevent drift outside the valid feature range.

---

## 23. Structural Growth

```python
def adapt_structure(self, z, activity, modulator):
    poorly_represented = activity.max() < self.match_threshold
    surprising = abs(modulator) > self.surprise_threshold
    if surprising and poorly_represented:
        self.grow_branch(z)
```

A new sparse branch:

```python
def grow_branch(self, z):
    k = self.find_unused_branch()
    candidate_features = self.select_active_features(
        z, count=self.synapses_per_branch
    )
    self.mask[k] = 0
    self.mask[k, candidate_features] = 1
    self.prototype[k] = z
    self.active[k] = True
```

---

## 24. XOR Milestone

Dataset:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

The system should autonomously construct representations equivalent to `NOT x1 AND NOT x2`, `NOT x1 AND x2`, `x1 AND NOT x2`, `x1 AND x2`, or a smaller functionally equivalent basis.

Success criterion: 100% accuracy across at least 20 random seeds.

No hidden representation may be manually initialized to XOR-specific conjunctions.

---

## 25. Critical XOR Ablations

Run:

* **A. Linear local learner** -- expected ~50%.
* **B. Dendrites without structural plasticity** -- measures whether random conjunctions suffice.
* **C. Structural dendrites without recurrence** -- determines whether recurrent dynamics are necessary for static XOR.
* **D. Full architecture** -- expected 100%.

The interesting comparison is not whether XOR can be solved. It is whether useful conjunctions are created autonomously.

---

## 26. Parity Benchmark Ladder

XOR is only the first rung. Run `XOR_2`, then `Parity_3`, `Parity_4`, `Parity_8`, `Parity_16`.

Parity is valuable because low-order marginal correlations provide little useful guidance.

Performance should be measured as a function of: number of branches, synapses per branch, training examples, structural mutations, recurrent settling steps, eligibility decay.

---

## 27. Hierarchical XOR

Construct `a = x1 XOR x2`, `b = x3 XOR x4`, `y = a XOR b`.

This tests whether learned conjunctions can themselves become inputs to later conjunctions. The architecture therefore needs a mechanism allowing basin states or dendritic outputs to feed other dendritic units.

Potential extension: `z = [x, d, h]` where dendrites may sample from raw input, prior dendrites, or recurrent basin state. This creates a hierarchy without explicitly defining neural layers.

---

## 28. Multiplexer Benchmark

A multiplexer requires selected variables to determine which other variable matters. This tests conditional computation.

Example: `y = x_address`.

The system must learn context-dependent feature interaction rather than simple memorization. This is a stronger test of dendritic gating.

---

## 29. Temporal XOR

Temporal XOR is the first major milestone for the recurrent architecture.

Sequence:

```
t=0:      present A
t=1..T:   distractors / silence
t=T+1:    present B
t=T+2:    request XOR(A, B)
```

The system must retain `A` in recurrent state `h_t`, then bind it with later input `B`.

Desired computation: `h_A --(persistence)--> h_{A,t}`, followed by `D(B, h_{A,t})`, and finally `A XOR B`.

Delay lengths: 1, 4, 16, 64, 256, 1024 steps. This directly measures persistent state.

---

## 30. Temporal Credit Assignment

Eligibility traces become meaningful here. Suppose the relevant event occurs at `t=10` but reward arrives at `t=100`. Then `e_ki(100) ~= lambda_e^90 * e_ki(10)`.

The modulator can reinforce the earlier event: `Delta p_ki = eta * M(100) * e_ki(100)`.

This tests whether the architecture can assign delayed credit without backpropagation through time.

---

## 31. ManifoldFlow Integration

Once the recurrent-basin prototype works, replace or augment the simple recurrent matrix with ManifoldFlow.

Architecture: `x -> D(x) -> MF basin router -> continuous flow -> h* -> y`.

Each basin stores `B_j = (c_j, r_j, d_j, kappa_j, u_j)` where:

* `c_j`: basin center,
* `r_j`: width/radius,
* `d_j`: depth,
* `kappa_j`: curvature,
* `u_j`: utility.

Dendrites propose basin coordinates. ManifoldFlow determines which attractor wins.

---

## 32. Basin Creation

If `|M| > theta_s` and no basin matches the current state (`min_j ||h - c_j|| > theta_b`), create a new basin: `create_basin(current_state)`.

This is analogous to dendritic branch creation but operates at the recurrent state level.

---

## 33. Basin Adaptation

Center update: `c_j <- c_j + eta_c * a_j * (h - c_j)`.

Depth update: `d_j <- d_j + eta_d * M * a_j`.

Decay: `d_j <- lambda_d * d_j`.

Useful states become deeper attractors. Unused states gradually disappear.

---

## 34. Candidate Manifold Dynamics

A simple continuous system:

`h_{t+1} = h_t + dt * [ D(x) + F_basin(h_t) - gamma * C(h_t) ]`

where `F_basin = -grad E(h)` and

`E(h) = - sum_j d_j * exp( -||h - c_j||^2 / (2 * r_j^2) )`.

This gives explicit attractor geometry. The flow need not itself be learned by backpropagation.

---

## 35. Longer-Term Language Architecture

Once temporal nonlinear learning works, the architecture can be extended toward language.

Potential pipeline: `token -> learned local token features -> dendritic mixtures -> MF recurrent state -> next-token distribution`.

Unlike the initial ManifoldFlow language architecture, the encoder would not remain frozen. Instead token features themselves would emerge from repeated local co-occurrence.

Possible hierarchy: characters/tokens -> local feature assemblies -> phrase/relation assemblies -> persistent manifold state -> prediction assemblies -> token output.

TinyStories is the first intended language-scale dataset because its vocabulary and semantic space are relatively constrained.

---

## 36. TinyStories Should Not Be the First Experiment

The architecture must demonstrate that it genuinely performs nonlinear credit assignment before language obscures failure modes.

Required progression: XOR -> Parity -> Hierarchical XOR -> Multiplexer -> Temporal XOR -> Delayed parity -> Sequence prediction -> Tiny synthetic language -> TinyStories.

Failure at an earlier stage should be investigated before scaling.

---

## 37. Metrics

Track more than accuracy.

**Task metrics**: accuracy, cross-entropy where relevant, convergence examples, generalization accuracy, temporal delay tolerance.

**Structural metrics**: number of dendritic branches, branches created per 1K examples, branch survival rate, average synapses per branch, branch reuse frequency, pruning rate.

**Basin metrics**: number of attractors, basin occupancy, basin lifetime, basin transition entropy, basin stability, settling steps.

**Learning metrics**: eligibility magnitude, modulator magnitude, weight update magnitude, branch usefulness distribution, recurrent spectral radius.

**Efficiency metrics**: operations per example, memory footprint, wall-clock learning throughput, inference latency, settling latency.

---

## 38. Failure Modes

### 38.1 Episodic memorization
The network creates one dendrite per complete input. Mitigation: enforce sparse dendrites; cap synapses per branch; reward feature reuse; penalize branch creation.

### 38.2 Combinatorial branch explosion
Higher-order problems generate too many candidate conjunctions. Mitigation: structural selection; usefulness-based pruning; local competition; branch splitting rather than arbitrary growth; sparse receptive fields.

### 38.3 Basin collapse
One attractor captures most inputs. Mitigation: lateral inhibition; activity normalization; anti-Hebbian competition; basin occupancy penalties.

### 38.4 Basin fragmentation
Every example forms a unique attractor. Mitigation: minimum basin creation threshold; basin merging; reuse bonuses; geometry-based consolidation.

### 38.5 Modulator dominates learning
If every synapse gets scaled directly by error magnitude, the system may degenerate into a crude global optimizer. Mitigation: the global signal should only gate local eligibility (`Delta w ~ M . e_local`); local state must determine the direction of adaptation.

### 38.6 No hierarchical feature formation
The model solves XOR but cannot solve hierarchical XOR. Mitigation: allow dendrites to consume outputs from previously learned assemblies.

### 38.7 Eligibility becomes too diffuse
Long traces cause unrelated events to receive credit. Mitigation: multiple timescale traces; branch-specific eligibility; event resets; sparse gating.

---

## 39. Experimental Phases

**Phase 0: Static XOR** -- prove that autonomous dendritic structure creation solves XOR. Target: 100% without predefined conjunctions.

**Phase 1: Sparse Parity** -- demonstrate reusable higher-order feature discovery (parity-3, parity-4, parity-8). Primary question: does structural complexity grow polynomially enough to remain useful?

**Phase 2: Hierarchical Structure** -- hierarchical XOR, multiplexer, compositional Boolean tasks. Goal: determine whether learned features can become inputs to new features.

**Phase 3: Recurrent State** -- delayed copy, temporal XOR, delayed parity, sequence binding. Goal: demonstrate persistent state.

**Phase 4: Eligibility Credit** -- vary delay between causal event and teaching signal. Goal: measure temporal credit-assignment range.

**Phase 5: ManifoldFlow Basin Replacement** -- replace simple recurrent attractor with MF geometry. Compare recurrent Hebbian matrix vs. ManifoldFlow basin dynamics on memory duration, interference, basin reuse, computational cost.

**Phase 6: Continuous Inputs** -- move from binary literals to continuous local receptive fields (synthetic 2D manifolds, nonlinear classification, trajectory prediction).

**Phase 7: Tiny Sequential World** -- a tiny symbolic environment containing objects, identities, positions, actions, rewards, delayed dependencies. Goal: test whether the architecture forms reusable concepts.

**Phase 8: TinyStories** -- only after previous phases work. The system must learn token associations, short-range syntax, entity persistence, simple semantic relations, next-token prediction. No frozen weights, no pretrained embeddings.

---

## 40. Key Research Question

The most important question is not "Can XOR be solved without backpropagation?" -- that is already known to be possible.

The important question is: **Can a scalable learning system autonomously discover reusable higher-order features using local plasticity and low-bandwidth global feedback, while maintaining persistent recurrent state?**

That is the architecture's actual research target.

---

## 41. Strong Success Criteria

The project becomes interesting if it achieves all of the following:

* **Static nonlinear learning**: Parity_8 > 95% without predetermined conjunctions.
* **Structural reuse**: learned branches must activate across many distinct examples rather than memorizing individual samples.
* **Hierarchical composition**: hierarchical XOR must be learned without manually providing intermediate XOR outputs.
* **Temporal nonlinear reasoning**: temporal XOR > 95% with delays significantly longer than the immediate input window.
* **Delayed credit**: performance should remain above chance when the teaching signal arrives after the relevant event.
* **No gradient dependence**: all hidden learning must remain functional with autograd completely unavailable.
* **Persistent state**: removing recurrent state should significantly reduce delayed-task performance.

---

## 42. Stretch Success Criteria

The most compelling early result would be: learn 8-bit parity + learn delayed XOR + learn hierarchical XOR, from random initialization using only local structural plasticity, recurrent attractor dynamics, eligibility traces, and broadcast modulatory feedback.

That combination would establish that the architecture is doing substantially more than conventional Hebbian association.

---

## 43. Core Hypothesis

The working hypothesis is: **credit assignment can emerge from persistent local participation traces rather than explicit gradient transport.**

Higher-order nonlinear structure is generated by dendritic-like feature units. Global behavior selects among these structures. Eligibility traces connect delayed outcomes to local participation. Recurrent basins provide persistence and contextual state. Structural plasticity supplies new representational capacity as needed.

This creates a fundamentally different learning loop:

```
experience
    |
    v
local feature proposals
    |
    v
recurrent interpretation
    |
    v
behavior / prediction
    |
    v
global outcome signal
    |
    v
reinforce eligible local structures
    |
    v
grow / prune / reshape representation
```

The architecture therefore does not attempt to reproduce backpropagation through another mathematical disguise. It instead attempts to replace gradient transport with: **local memory + structural exploration + dynamic competition + global selection**.

---

## 44. Immediate Implementation

The first repository implementation should contain (paths below are the standalone-repo layout from the original design; see the implementation journal for how this was adapted into the sonic-manifold-flow repo's own conventions):

```
dendritic_basin/
|
|-- dendrites.py
|-- basin.py
|-- eligibility.py
|-- structural_plasticity.py
|-- readout.py
|-- model.py
|
|-- tasks/
|   |-- xor.py
|   |-- parity.py
|   |-- hierarchical_xor.py
|   |-- multiplexer.py
|   `-- temporal_xor.py
|
|-- experiments/
|   |-- xor_baseline.py
|   |-- xor_ablations.py
|   |-- parity_scaling.py
|   `-- temporal_credit.py
|
`-- tests/
    |-- test_no_autograd.py
    |-- test_local_updates.py
    |-- test_branch_growth.py
    |-- test_basin_settling.py
    `-- test_eligibility.py
```

The repository should initially use NumPy. That provides a useful hard guarantee: there is no hidden gradient machinery available.

Only after the learning rules are validated should the implementation move to PyTorch or custom GPU kernels for speed.

---

## 45. Final Architectural Summary

The proposed system is:

`x -> sparse nonlinear dendritic assemblies -> competitive recurrent manifold -> h* -> y`

Learning is: `local activity -> eligibility -> global modulation -> local adaptation`.

Representation growth is: `surprise + poor representation -> new structure`.

Memory is: `recurrent attractor state + ManifoldFlow basins`.

The central test is whether this combination can discover the nonlinear structure that backpropagation normally creates through hidden-layer gradients.

XOR is the first microscope. Parity tests scalability. Temporal XOR tests memory. Hierarchical XOR tests compositionality. TinyStories tests whether the mechanism can eventually grow into a language learner.
