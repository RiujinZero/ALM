# Phase 43: Selective Communication

## Result

A language-free multiplexing task now tests whether context can select one
useful column without amplifying every active column. Sixteen independent
source columns emit binary states. Each of sixteen sparse distributed context
patterns selects one source through a fixed hidden permutation. The learner
must predict that source's state.

Training uses independent distractors. The hard frozen evaluation sets every
irrelevant source to the opposite of the target, so global pooling cannot pass
by accumulating weak correlations.

Across ten seeds after 4,000 training trials:

Here one bit/outcome is a uniform binary prediction and zero is perfect
confidence; lower is better.

| Communication rule | Natural bits/outcome | Adversarial bits/outcome | Adversarial accuracy | Body bytes |
|---|---:|---:|---:|---:|
| Global pooling | 0.994818 | 1.080840 | 0% | 670 |
| Fixed receptor address | 0.005781 | 0.005789 | 100% | 673 |
| Locally learned route | **0.002889** | **0.002890** | **100%** | 3,289 |

The learned condition recovered 100% of the hidden context-to-source routes
across all seeds without receiving source identities. Its context-local branch
competition is slightly more confident than the fixed addressed outcome table
because it learns route reliability directly.

## Local Route Discovery

Each candidate context-to-source branch observes only:

- the active distributed context assembly;
- its source column's binary state; and
- the later binary consequence.

The branch increments a local agreement statistic when source state and
consequence match. Candidate branches for the active context compete by
reliability; the strongest branch supplies the routed prediction. The hidden
environmental permutation is used to generate outcomes but is rejected by the
learned learner's API.

## Perturbation Result

Global pooling guesses slightly above chance on ordinary independent inputs,
but changes direction when irrelevant columns are made coherently misleading.
Both selective mechanisms are invariant because those columns do not
participate in the active communication route. This operationalizes selective
amplification rather than generic excitation.

## Cost and Limitation

The learned router performs sixteen candidate updates per training event and
averages roughly 91,000 trials/s, versus one update and 109,000 trials/s for
the fixed address. Its 3.29KB body is still tiny, but about five times the
fixed-route body. The comparison therefore fits comfortably inside the same
byte class but does not have identical occupied bytes.

The current context patterns are supplied stable assemblies. Route discovery
is local, but context-assembly birth is not tested. Sparse developmental
candidate selection should replace the exhaustive sixteen-branch update before
scaling this mechanism to many columns.

## Reproduction

```powershell
python -m manifoldflow.experiments.dendritic_basin_selective_communication_probe
```
