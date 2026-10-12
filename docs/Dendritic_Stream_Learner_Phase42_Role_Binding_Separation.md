# Phase 42: Role Binding and Separation

## Result

A language-free compositional probe now requires the same learner to reuse
content in novel combinations while refusing false associations. Sixteen
entities are fixed sparse distributed patterns with eight active receptors in
a 16-receptor organ. Every entity has a hidden binary property. Directed
actor/patient pairs produce the actor's property; 25% of unordered pairs are
withheld in both directions, then evaluated as unseen compositions.

Across ten seeds after 4,000 training and 4,000 frozen evaluation trials:

Here one bit/outcome is a uniform binary prediction and zero is perfect
confidence; lower is better.

| Coordination rule | Unseen bits/outcome | Unseen accuracy | Opposed-role accuracy | Body bytes |
|---|---:|---:|---:|---:|
| Unordered coactivity | 0.770174 | 73.11% | 49.72% | 891 |
| Ordered episode conjunction | 1.000000 | 48.69% | 50.47% | 10,391 |
| Role-gated reusable content | **0.003150** | **100.00%** | **100.00%** | **898** |

The coactivity learner partially predicts common cases but writes the actor's
outcome into the patient assembly on every trial. It becomes chance precisely
where actor and patient properties oppose. The conjunctive learner memorizes
seen ordered pairs well but has no reusable component to apply to a new pair.
It is both larger and non-compositional. Role gating changes only the actor's
local table, so the same content assembly transfers into every unseen pairing.

## The Three Ablations

1. **Bag:** pool both content assemblies to predict and update both on the
   outcome. This tests whether coactivity alone is sufficient.
2. **Conjunctive:** key learning by the entire ordered actor/patient episode.
   This preserves roles but creates a new representation for every combination.
3. **Role-gated:** the actor role receptor opens plasticity only on the actor's
   reusable content assembly. Patient content remains active but is not bound
   to the actor's consequence.

## What This Establishes

- Association and separation must be measured together.
- Explicit role coordination supports novel composition without pairwise
  memorization.
- A few extra routing bytes can replace combinatorial conjunctive anatomy.
- Opposed-role cases are a necessary anti-binding diagnostic; aggregate
  accuracy alone understates the coactivity failure.

## What It Does Not Establish

The probe provides stable distributed content patterns, and the learner packs
each pattern into an assembly address. It tests coordination of reusable
content, not autonomous discovery or growth of those content assemblies. The
next integration must apply role receptors to assemblies born by the existing
dendritic cortex rather than to given patterns.

## Reproduction

```powershell
python -m manifoldflow.experiments.dendritic_basin_binding_separation_probe
```
