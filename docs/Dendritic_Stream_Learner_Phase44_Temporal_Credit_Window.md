# Phase 44: Temporal Credit Window Assay

## Result

The Phase 41 local eligibility mechanism now has an explicit delay/decay
surface. Five trace decays (`0.5`, `0.9`, `0.97`, `0.995`, and `1.0`) were
tested across consequence delays of 0, 2, 8, 32, and 128 steps on the same
modality-neutral hidden-polarity task.

After 1,000 experiences and 4,000 frozen evaluations, averaged over ten seeds:

| Delay | Decay 0.5 | Decay 0.9 | Decay 0.97 | Decay 0.995 | No decay |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.85 | 0.05 | 0.03 | 0.02 | 0.02 |
| 32 | 1.00 | 0.40 | 0.06 | 0.03 | 0.02 |
| 128 | 1.00 | 1.00 | 0.53 | 0.04 | 0.02 |

Values are bits per binary outcome: one is uniform uncertainty and zero is
perfect confidence. Threshold accuracy is misleading here. At delay 128,
decay 0.9 retains a tiny correctly signed residue and can classify every case,
yet remains at approximately 1.0 bit because it assigns almost no confidence.

## Interpretation

The assay confirms that a local bounded trace bridges delayed consequence in a
quantitative, falsifiable way. A trace with decay `d` has strength `d^delay` at
the update site. Once it falls below the local eligibility floor, no durable
update occurs.

It also reveals that this task cannot choose a biologically useful lifetime.
Trials do not overlap, so retaining information longer has no downside and a
nondecaying trace dominates every delay. That is an artifact of the benchmark,
not evidence that permanent eligibility is desirable.

## Status of LR-4

The delay sweep and trace ablation are implemented. Learned trace shape and
lifetime are not. The decisive follow-up must present overlapping candidate
causes while consequences are delayed. Short traces should forget the true
cause; overly long traces should bind intervening causes; an appropriate local
timescale should win between them. Only that task can test whether trace
competition discovers a useful temporal window from stream statistics.

## Reproduction

```powershell
python -m manifoldflow.experiments.dendritic_basin_temporal_credit_probe
```
