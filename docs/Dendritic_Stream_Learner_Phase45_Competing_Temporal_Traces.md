# Phase 45: Competing Temporal Traces Under Interference

## Result

Phase 44's isolated trials were replaced by a continuous stream in which every
receiver changes state on every step and a consequence refers to the selected
receiver's state eight steps in the past. Candidate causes therefore overlap:
short traces lose the relevant event, while very long traces mix many
intervening events.

Each receiver maintains six local leaky traces. At delayed consequence, its
apical route updates a local `trace sign x outcome` table for every timescale.
Conditional entropy selects the most predictive dendritic timescale without
access to the environment's true delay.

Across ten seeds after 50,000 training steps and 20,000 frozen evaluation
steps:

| Trace decay | Eval bits/outcome |
|---:|---:|
| 0.0 | 1.000300 |
| 0.5 | 1.000300 |
| 0.8 | 0.996243 |
| 0.9 | 0.984757 |
| **0.97** | **0.983901** |
| 0.995 | 0.996225 |

Local competition selected only 0.9 or 0.97 traces across all receivers and
seeds. Its combined frozen result was 0.984427 bits/outcome and 57.40%
accuracy. The body occupied 1,757 bytes and active trace state 513 bytes.

## Interpretation

The selection result is real: predictive local statistics reject both traces
that are too short and traces that smear activity across too much history. A
useful temporal neighborhood can emerge without supplying the causal delay.

The effect is also too weak to count as solved temporal binding. Exponentially
filtered signs retain a correlation with the eight-step-old event, but not
enough event identity for confident prediction. Selecting a decay is not the
same as reconstructing which earlier event earned credit.

## Design Consequence

LR-4 remains open. The next temporal substrate should be tested against this
same overlapping stream and must materially beat 0.984 bits. Promising generic
options are:

- event-specific eligibility tags that coexist rather than overwrite;
- cascaded or band-pass dendritic traces with peaks at different delays;
- sparse temporal assemblies over multiple trace receptors; or
- local recurrent state that preserves identity until consequence.

Simply increasing decay is falsified by the long-trace regression. Any richer
substrate must report its transient bytes, because temporal identity is working
memory even when it is not durable anatomy.

## Reproduction

```powershell
python -m manifoldflow.experiments.dendritic_basin_adaptive_temporal_credit_probe
```
