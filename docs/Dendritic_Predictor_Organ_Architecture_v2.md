# Dendritic Predictor Organ — Architecture v2

Describes the system as actually implemented in
`src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_veccond2.py`
and `src/manifoldflow/dendritic_basin/surprise_grown_future_transition.py`,
as currently running in the veccond4 240-pass campaign. This is a
"what's actually built and running" doc, not a design proposal.

## 1. Lineage

- **v1** (`dendritic_basin_shared_goal_convergence_train.py`): vector-conditioned
  convergence 2x2. A deliberately linear organ (far_w: a delta-rule linear
  head over fast features) primed the base with a dense vector. Arm A's
  long-interruption margin rose to +0.0408 (significant at 8 passes) —
  the signal that motivated v2.
- **v2** (`..._veccond2.py`, this doc): replaces the linear organ with the
  codebase's own surprise-conditioned dendritic grower,
  `SurpriseGrownFutureTransition`, run **live** during the convergence
  passes. Per observed turn it forecasts a mixture of candidate futures,
  the base learns response bytes under that primed mixture vector, then
  the organ resolves the observed code with learning enabled — so
  prediction surprise drives dendritic growth (nursery births, promotions,
  consolidation, recycling) across passes.
- **veccond3**: v2 organ with dendritic context assemblies
  (`k_inputs_per_branch=8`) routing context, no predictor dendrites
  (`k_inputs_per_branch` here governs the *context-assembly* gate, not a
  per-mode predictor — that came later). Disappointing finish: no
  convergent positive signal across marks.
- **veccond4** (current): adds a live **dendritic predictor** per residual
  mode (`MultiInputInteractionDendrite` with `output_weights`), replacing
  the removed `far_w` linear head entirely. This is the run this doc
  documents the config of.

## 2. One organ, four arms, one axis

The docstring in `veccond2.py` describes a nominal 2×2 (ledger × K1/K4),
but **K1 vs K4 has collapsed to a no-op**: `far_w` and the
far-agreement-reweighted mixture were removed from training (`FAR_ARMS`
is now the empty tuple `()`). The per-row priming logic is:

```python
state_pred = mixture_point(forecast.prediction, forecast.candidates, use_basal=True)
for arm in ARM_NAMES:
    content = state_pred + cores[row_index] if arm in LEDGER_ARMS else state_pred
```

`state_pred` is identical for every arm — there is no per-arm branch left
that reads `k_inputs_per_branch` differently. So the eval results at mark
108560 show `A_k1_nooledger == C_k4_nooledger` and `B_k1_ledger ==
D_k4_ledger` to 4 decimal places, exactly as expected. The **only** live
axis is:

| Arm | Ledger residual added to prime? |
|---|---|
| A_k1_nooledger, C_k4_nooledger | No |
| B_k1_ledger, D_k4_ledger | Yes (`+ cores[row_index]`) |

All four arms are kept only for evaluator/report-schema compatibility with
v1 and veccond3; C and D are currently redundant with A and B.

One growing organ is shared across all four arms — its own learning
(nursery growth, promotion, resolve) is arm-independent, since it predicts
the same observed codes from the same conversation context regardless of
which arm is priming. Only the per-arm base's plasticity under the primed
vector differs.

## 3. The dendritic predictor (veccond4's addition over veccond3)

Each residual mode (`_ResidualMode` in `surprise_grown_future_transition.py`)
optionally owns a `MultiInputInteractionDendrite` predictor
(`src/manifoldflow/dendritic_basin/multi_input_interaction_dendrite.py`):

```
d(context) = exp[ -(w · context[indices] - μ)² / (2σ²) ]   # K-input bell
predict(context) = d(context) * output_weights[target_dim]
adapt: output_weights += lr * d(context) * error            # delta rule
```

- `indices`: K fixed input positions into the context vector, chosen at
  birth (`K = k_inputs_per_branch`).
- `output_weights`: one weight per target-vector coordinate
  (`target_dim = content_dimensions = 512` in veccond4) — this is what
  actually replaced `far_w`.
- Bootstrapped from the nursery candidate's `residual_mean` at promotion;
  updated online by the delta rule during `resolve()`.

Prediction path inside `_mode_residual`:

```python
if mode.predictor is not None:
    return mode.predictor.predict(context_vector)
if mode.residual_mean is not None:
    return mode.residual_mean
return zeros
```

Once a mode has a predictor (`k > 0`), `residual_mean` on the promoted
`_ResidualMode` is set to `None` — it is dead weight for k>0 modes (frozen
after promotion, never read or updated again), so it was dropped from the
public/private dataclasses this session to reclaim memory. The
`_NurseryCandidate.residual_mean` field is unaffected — it's still needed
to bootstrap `output_weights` at promotion time.

## 4. Memory budget

Per residual mode with a K-input predictor at `target_dim = 512`:

- predictor: `K` indices + `K` weights (`int`/`float64`) + `μ`, `σ` scalars
  + `output_weights` (512 float64 = 4096 bytes)
- mode metadata: context-assembly id, lifecycle state, gain EMAs, variance
  tracking (no `residual_mean` for k>0 modes, since v2 of this doc's
  memory optimization)

≈ 17 KB/mode fully loaded at `K=8`, `target_dim=512`. At the configured
cap of 25,000 modes that's ≈ 425 MB, within the ~500 MB target capacity
this run was sized for (`scripts/run_veccond4_240pass_train.sh`).

Rough capacity reference points (pure predictor cost only, no metadata):
~58k dendrites at 500 MB with K=1 minimal fields; ~28k with the full mode
overhead this run actually carries; multiple prediction directions per
mode (multi-branch) would trade capacity for direction coverage
(2 branches/mode → ~21k modes at the same 500 MB, vs ~35k single-branch).

## 5. Current veccond4 config

From `scripts/run_veccond4_240pass_train.sh` / `run_veccond4_240pass_evals.sh`:

| Flag | Value | Meaning |
|---|---|---|
| `--total-passes` | 240 | full campaign length |
| `--k-inputs-per-branch` | 8 | dendrite fan-in per predictor branch |
| `--max-residual-modes` | 25,000 | organ capacity cap |
| `--max-context-assemblies` | 6,250 | ≈ modes / 4 |
| `--content-dimensions` | 512 | state vector size (`target_dim`) |
| `--active-content-features` | 64 | active features passed to base decoder |
| `--context-dimensions` | 128 | organ context vector dimension |
| `--ledger-event-dimensions` | 64 | ledger event dimension |
| `--resume-from-mark` | 108,560 | WSL-crash recovery point (added this session) |
| dataset | `confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3` | |
| checkpoint | `epoch_16_checkpoint.pkl` | frozen base start point |

Marks are taken at `{0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128} × passes`
(doubling schedule), plus the final pass count.

## 6. Evaluator contract

`..._veccond2_evaluate.py` loads one organ snapshot and four per-arm base
snapshots per mark, replays `original_unanswered_prefix` and
`long_interruption_crossover` context modes, and reports paired margin
deltas (`arm_minus_native`, `arm_minus_shuffled`, `oracle_minus_native`,
etc.) under both sampled and greedy decoding, with 95% CIs and positive
fractions over paired conversation examples.

The evaluator still reads `organ_snapshot.get("far_w")` defensively (fixed
this session — it previously did an unconditional `["far_w"]` lookup that
raised `KeyError` on veccond4 snapshots, since `FAR_ARMS = ()` means the
value is loaded but the branch that would use it is now unreachable).

## 7. Status as of this doc

First recovered veccond4 read at mark 108,560 (~8 passes): no arm clears
significance yet (all `arm_minus_native` CIs straddle zero, n=28 pairs).
No-ledger arms (A/C) at +0.0203 mean, ledger arms (B/D) at -0.0085 mean —
too early and too few pairs to call a direction. Training/eval continue
live in WSL screen sessions `v4train`/`v4eval`.
