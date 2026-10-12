# Dendritic Basin Learning v0.3: Language Smoke Findings

**Addendum to `Dendritic_Basin_Learning_Design_v0.2_Interaction_Dendrites.md`.
First language-shaped sandbox tests of the interaction-dendrite mechanism
(word-level, small closed vocabulary) -- two real bugs found, both with
clean architectural fixes rather than parameter tuning. Findings +
architecture revision only; not yet ported into
`src/manifoldflow/dendritic_basin/`.**

Status: Sandbox findings, Research Design v0.3 (pseudocode/architecture
revision; no implementation yet)

## 0. What this tests

v0.2 showed the generic interaction dendrite spontaneously discovers
SAME/DIFFERENT on binary parity tasks with no hand-coded primitive. This
round moves to the first language-shaped test: a continuous token stream
with a genuinely ambiguous next-token distribution, still with no
backprop, no hardcoded relations, and no output-error gradient anywhere.

## 1. Smoke 1: continuous tiny language

An 11-token vocabulary, continuous stream:

```
<BOS> alice sees cat .
<BOS> bob sees dog .
<BOS> carol sees bird .
<BOS> dave sees fish .
...
```

The important ambiguity: `P(object | sees) = 1/4`, so a one-token model
cannot know which object comes next -- the identity of the subject
several tokens back is what disambiguates it.

The generic locally-learned dendrite discovered relations such as
"recent alice + current sees" without being told that conjunction. For
one learned unit, activation was approximately:

```
alice sees -> 0.73
bob sees   -> ~0
carol sees -> ~0
dave sees  -> ~0
```

All four corresponding relations emerged (one per subject/object pair).

### Prediction results

| Model | Overall bits/token | Object accuracy after "sees" | Object bits |
|---|---|---|---|
| Unigram | 3.122 | 0%* | 4.322 |
| Current-token / bigram | 0.801 | 24.8% | 2.000 |
| Exact trigram (oracle) | 0.401 | 100% | 0.0014 |
| Dendrite + traces | 0.401 | 100% | 0.0019 |

\* Argmax happens to choose the wrong object consistently; entropy
(bits) is the meaningful metric there, not accuracy.

The no-backprop relation system essentially reached the exact-trigram
oracle. Generated sequences were clean:

```
<BOS> carol sees bird .
<BOS> dave sees fish .
<BOS> alice sees cat .
<BOS> bob sees dog .
...
```

## 2. Bug #1: ghost relations

The original XOR-style dendrite (v0.2) could fire on `A=0, B=0`. That's
harmless for parity, where zero is a meaningful binary state. It is
disastrous for sparse language: "not Alice" isn't an event, "not sees"
isn't an event. Thousands of relation units started hallucinating
activity simply because *neither* parent existed. Prediction collapsed
to 2.89 bits/token instead of 0.40.

This exposed an important distinction:

> **state variables != event channels**

For an event representation, dendritic *support* was changed to:

```
g(A, B) = min(A, B)
```

so that the relation's two output channels become:

```
r_0 = g * d(A, B)
r_1 = g * (1 - d(A, B))
```

The generic bell (`d`) still decides *what* relation state exists, but
the dendrite can't fire at all unless both parent events actually have
support (`g > 0`). This immediately fixed the ghost-relation problem, and
is also more biologically reasonable: no synaptic drive on either parent,
no dendritic event.

## 3. Automatic structural discovery

Removing the manually selected `(alice, sees)`, `(bob, sees)`, etc. pairs
and instead letting all recent token traces compete as possible parents,
ranked by locally measured predictive improvement, the top four
structures discovered were:

```
sees <-> carol
sees <-> dave
sees <-> alice
sees <-> bob
```

with information gains around 1.79 bits each; the next candidates were
effectively 0. All four correct relations were already the top four after
roughly 100 stream tokens. This suggests a giant hand-built feature bank
isn't needed -- competitive ranking by predictive improvement finds the
useful parent pairs on its own.

## 4. Smoke 2: variable temporal distance

Filler words inserted between subject and verb, 0-4 random fillers per
example, same continuous stream:

```
alice sees cat
alice now sees cat
alice softly now sees cat
alice then softly now then sees cat
```

Now immediate context isn't reliable -- the informative token is a
variable number of steps back.

| Model | Overall bits/token | Object accuracy | Object bits |
|---|---|---|---|
| Bigram | 1.405 | 25.1% | 1.999 |
| Trigram | 1.330 | 42.4% | 1.564 |
| Temporal relations (decaying traces) | 1.124 | 100% | 0.043 |

The decaying trace formulation (`q_{t+1} = lambda*q_t + x_t`) lets the
model associate `alice_{t-delta} + sees_t` across a variable delay --
encouraging for continuous input, since it means the architecture isn't
reliant on discrete positional slots.

## 5. The binding problem: where it finally broke

Language changed to:

```
alice meets bob sees cat .
bob meets alice sees dog .
carol meets dave sees bird .
...
```

The object belongs to the **first** entity, not the most recent one. So
`alice meets bob sees ?` must produce `cat`, despite `bob` being more
recent. This defeats simple decay memory.

| Model | Object accuracy |
|---|---|
| Bigram | 27.5% |
| Trace + dendritic relations | **62.2%** |

Better than bigram, but far from solved -- it retained useful
information, but recency corrupted identity. This is an informative
failure: it demonstrates that **temporal traces are not persistent
binding** -- exactly the gap ManifoldFlow was supposed to fill.

## 6. Bug #2: recency is not identity

A decaying trace says "Alice happened 4 steps ago." It does not say
"Alice is the grammatical/semantic entity that this later event refers
to." Those are fundamentally different claims.

### Fix: the smallest possible persistent relation basin

When local relations form, certain assemblies survive until the episode
boundary. For `<BOS> alice meets bob sees cat .`, the stream
automatically produces candidate assemblies like:

```
(BOS, alice)
(alice, meets)
(meets, bob)
...
```

These persist. Relation-on-relation composition is then allowed:
`sees + (BOS, alice) -> R`. No rule says Alice is the answer -- *every*
persistent assembly gets a chance to be consumed by a later relation, and
the useful ones win through predictive utility alone. The top learned
structures became things like:

```
sees + (BOS, alice)   -> cat
sees + (alice, meets) -> cat
```

(independently discovered) -- the distractor relations (e.g. anything
keyed on `bob`, the more recent but wrong entity) don't predict the
object reliably, so they lose the competition.

### Result

- Object accuracy: 62.2% -> **100%**.
- Object surprise: 1.076 bits -> **0.00127 bits**.
- Overall: 1.011 -> **0.513 bits/token**.

Generation maintained the binding correctly:

```
<BOS> alice meets carol sees cat .
<BOS> carol meets bob sees bird .
<BOS> bob meets alice sees dog .
<BOS> dave meets bob sees fish .
<BOS> carol meets alice sees bird .
```

Every object belongs to the *first* entity despite the intervening name.

## 7. Revised architecture: four kinds of state

The architecture should not collapse these into one mechanism -- each
earns its keep differently:

```
instantaneous event -> decaying traces -> relation assemblies -> persistent basin assemblies
```

1. **Event channels** -- current sensory evidence. For language:
   `x_t = token/byte event`.
2. **Decaying traces** -- cheap temporal context: `q_{t+1} = lambda*q_t +
   x_t`. Good for "this happened recently."
3. **Dendritic relations** -- discover combinations: `R = f(A, B)`. Good
   for "A and B together constitute something."
4. **Persistent MF assemblies** -- maintain selected relational state
   despite intervening events: `R_t -> B_j -> B_{j,t+k}`. Good for "this
   relationship is still relevant."

Relations can then consume basin state again: `R' = f(x_t, B_j)`. That's
where hierarchy comes from -- not a fixed stack of layer 1/layer 2/layer
3, but a graph that grows according to which relations survive:

```
event stream
  -> temporal traces
  -> interaction discovery
  -> persistent relational state
  -> recursive interaction discovery
  -> prediction
```

## 8. What to change before scaling up

Two things deserve another sandbox iteration before a bigger ("rig") run:

1. **Replace the crude persistent set with actual MF basin dynamics.**
   Persistence is currently binary (an assembly is either stored or it
   isn't). The existing ManifoldFlow machinery (depth, decay, attraction,
   competition) should make this substantially more realistic than a
   simple survive/discard set.
2. **Replace word IDs with a raw byte/character stream.** The smokes so
   far deliberately used only 11-14 atomic words to debug the learning
   mechanism in isolation. The next version should see raw characters
   (a vocabulary of perhaps 32-64 characters) and have to discover
   `alice`, `sees`, `cat`, etc. itself, rather than starting from
   pre-segmented word tokens.

## 9. Summary

```
dendrites discover        -- relation formation from raw co-occurrence
traces bridge time        -- variable-delay association, no positional slots
MF should preserve bindings -- identity persistence across intervening events
recursive relations build hierarchy -- no fixed layer count
```

No backprop was needed anywhere in these probes. The next
(character-level) smoke is the one that will tell us whether this is
heading toward a genuine language learner rather than a clever
symbolic-event learner -- i.e. whether the same mechanism that discovered
`sees <-> alice` from pre-tokenized words can also discover word
boundaries and word identity from raw characters, with nothing handed to
it but a byte stream.
