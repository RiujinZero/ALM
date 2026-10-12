# Dendritic Basin Learning v0.4: Strict Character-Stream Findings

**Addendum to `Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`.
The character-level, one-pass, held-out-combination smoke that v0.3
flagged as the real test of "language learner vs. clever symbolic-event
learner" -- passed, with three more bugs found and fixed the same way as
before: clean architectural fixes, not parameter tuning. One acknowledged
shortcut remains before this is ready for a larger rig run.**

Status: Sandbox findings, Research Design v0.4 (one sandbox conversion
still pending before scale-up; see section 7)

## 0. What changed since v0.3

v0.3 ran word-level (pre-tokenized) language smokes and named two things
to fix before scaling: real MF basin dynamics instead of a binary
persistent set, and a raw byte/character stream instead of word IDs. This
round did the harder of those two: **raw characters, no tokenizer, no
word IDs, no embeddings, no label ever saying which token is "the
subject."** The test was also deliberately made stricter than v0.3's by
withholding specific subject+distractor *combinations* from training, so
surface-level sentence memorization cannot solve the held-out case.

## 1. Strict character-stream setup

The learner sees only raw characters, e.g.:

```
b o b   n o w   m e e t s   c a r o l   n o w   s e e s   d o g .
```

No word IDs, no tokenizer, no embeddings, no labels. The synthetic
language maps names to objects:

```
alice -> cat
bob   -> dog
carol -> bird
dave  -> fish
```

with a second name inserted as a distractor, e.g. `bob softly meets alice
quickly sees dog.` -- the object is determined by the **first** name, the
same binding problem as v0.3, but now with the additional constraint
that specific `(subject, distractor)` combinations (e.g. `bob + carol`)
are held out entirely: the learner may see Bob with other names and
Carol with other names, but never that pairing. Sentence-surface
memorization cannot solve the held-out test.

## 2. Character assemblies emerge online, in one pass

The final version is genuinely one-pass: as characters arrive, the
learner maintains local substring statistics and allows an assembly to be
born when its internal cohesion crosses a threshold. From raw characters
alone it independently created persistent assemblies for:

```
alice, bob, carol, dave, sees, meets, cat, dog, bird, fish
```

plus assorted fragments and other useful pieces -- and critically, these
appeared *during* the stream, not from an offline tokenizer pass. This
closes the shortcut v0.3's word-level smoke left open.

## 3. Three more bugs, three more architectural fixes

### 3.1 Dead-input ghost assemblies (recurrence of the v0.3 bug, now at
character granularity)

The XOR-style dendrite can interpret `A=0, B=0` as meaningful -- fine for
Boolean variables, terrible for sparse language events. Same fix as
v0.3's section 2: `g(A,B) = min(A,B)`, so a relation can only fire when
its parents actually provide support.

### 3.2 Spurious correlations became too powerful

Withholding subject+distractor combinations exposed a new failure: the
model learned things like `Bob + meets -> some distractor distribution`.
That is a *legitimate* training correlation, but not the causal relation
the task actually needs. A **hard** relation winner (one that fully
replaces the base prediction) made overall held-out prediction *worse*,
even though it got the subject-object binding right.

**Fix**: basin depth should depend on residual uncertainty, not just on
whether a relation won a competition.

```
D_R proportional to exp(-H(Y|R) / tau)
```

A relation whose consequence is nearly deterministic (low conditional
entropy) becomes a deep attractor; a relation that merely nudges the
probability distribution stays shallow. Instead of replacing the base
prediction `P_local(y)` outright, the two are interpolated by depth:

```
P(y) = (1 - D_R) * P_local(y) + D_R * P_R(y)
```

This completely changed the held-out result (section 4).

### 3.3 "Longest assembly wins" overfits

Early on, the learner created assemblies such as `ob sees`, `ve sees`,
`ce sees` and treated them as better contexts than plain `sees`, purely
because they were longer/more specific. This breaks composition: `ob
sees` contains part of the distractor name `bob`, so a novel combination
the learner never saw becomes effectively out-of-distribution for that
assembly.

**Fix**: evidence-weighted assembly competition instead of longest-match:

```
S(A) = log(1 + support(A)) + lambda * integration_size(A)
```

An extra-specific assembly only wins if there is enough evidence to
justify the extra detail, and assemblies whose apparent cohesion
disappears as more data arrives get pruned. This is structural birth
*and* structural death, both driven by the same evidence accounting.

## 4. Final strict one-pass result (held-out combinations)

| Model | bits/character | Object accuracy | Object surprise |
|---|---|---|---|
| Local character assemblies only | 0.347 | 23.3% | 2.006 bits |
| Hard persistent relations | 0.554 | 100% | 0.041 bits |
| Entropy-depth relations | **0.264** | **100%** | 0.181 bits |

The middle row is the interesting one: relations alone solved binding
(23.3% -> 100%) but a *hard* attractor overtrusted other accidental
correlations enough to raise total loss above the no-relations baseline
(0.347 -> 0.554 bits/character). Making attraction strength proportional
to certainty (section 3.2's fix) recovered **both** 100% binding *and* a
lower bits/character than the no-relations baseline (0.347 -> 0.264) --
on compositions the learner never experienced during training.

## 5. It completes the actual held-out words

Given prefixes from held-out combinations, greedy continuation:

```
bob now meets carol now sees        -> dog.
dave then meets alice sees          -> fish.
carol quickly meets dave quickly sees -> bird.
```

Across 500 held-out prefixes: **500/500** generated the complete correct
object plus sentence ending -- not merely the first character.

## 6. Free-running generation: works, and shows the next frontier

Sampling from an empty state produces fluent examples:

```
dave meets carol quickly sees fish.
alice then meets bob softly sees cat.
bob then meets alice then sees dog.
carol quickly sees bird.
dave quickly meets bob sees fish.
```

Clear evidence of learned tiny-language syntax. At moderate sampling
temperature, binding correctness is in the mid/high 90s; lowering
temperature gave essentially perfect binding in the runs tested.

But generation also occasionally produces:

```
dave then meets dave sees fish.        <- same name as subject and distractor
carol quickly sees bird.               <- grammatical, but skips "meets X"
```

The second is grammatical despite differing from the training template
-- interesting on its own. The first shows that **syntactic probability
and semantic/world constraints aren't yet the same thing**: nothing in
the current mechanism forbids a person from "meeting themselves," because
that constraint is a world-model fact, not a syntactic one. That is
exactly where real persistent MF state (as opposed to the sandbox's
current simplified persistence) needs to start earning its keep.

## 7. What we've learned: the emerging pipeline and its three quantities

The architecture is converging on:

```
raw event stream
  -> multi-scale local traces
  -> online structural assembly birth/pruning
  -> relation-on-relation discovery
  -> persistent attractor state
  -> certainty-controlled prediction
```

Three quantities that should **not** be conflated into one signal:

- **activation** -- is this pattern happening right now?
- **utility** -- does this relation improve prediction?
- **depth** -- how certain/stable is this relationship (section 3.2)?

This is a clean three-way division for how ManifoldFlow basin state
should be parameterized going forward.

## 8. One remaining shortcut before scaling up

The smoke is genuinely one-pass now, but **structural assembly discovery
is still implemented with explicit running substring/co-occurrence
statistics** -- a convenient sandbox proxy for "local dendrite population
+ structural competition," not yet the actual generic-bell interaction
dendrite (v0.2) operating directly on character traces.

**Before this goes to a larger rig run**, one more sandbox conversion is
planned (not another conceptual benchmark): replace the running n-gram
cohesion detector with the real mechanism --

```
population of generic local dendrites
  -> perturb parameters locally
  -> novel-variance objective (v0.2's J)
  -> competitive survival
  -> new assembly
```

-- while preserving the exact language benchmark this document just
passed. If that version still gets approximately 100% held-out binding
and beats the local-character-predictor baseline on bits/character, it
is considered ready to move to a larger GPU rig with a much larger
continuous TinyStories stream. At that point there would be no
tokenizer, no offline chunk miner, no backprop, and no task-specific
representation machinery of any kind left in the system.
