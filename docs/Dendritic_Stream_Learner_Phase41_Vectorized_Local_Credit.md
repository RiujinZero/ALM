# Phase 41: Vectorized Local Credit

## Result

A modality-neutral delayed-causality probe now distinguishes where a learning
update is computed from how far its instructive signal travelled. Sixteen
intermingled receiver columns each see an independent binary basal receptor.
One context pathway terminates on one receiver's apical compartment. Eight
distractor steps later, the environment reveals an outcome controlled by that
receiver's basal state under a fixed hidden polarity. Half the receivers have
positive roles and half have opposing roles.

The learner never receives the hidden polarity or a desired weight. Each
receiver owns only a local `basal receptor x outcome` count table and a decaying
eligibility trace.

Across ten seeds after 4,000 training and 4,000 frozen evaluation trials:

Here one bit/outcome is a uniform binary prediction and zero is perfect
confidence; lower is better.

| Delayed instruction | Eval bits/outcome | Eval accuracy | Updates | Wrong receiver updates |
|---|---:|---:|---:|---:|
| None | 1.000000 | 49.65% | 0 | 0 |
| Broadcast scalar | 0.910939 | 100.00% | 64,000 | 60,000 |
| Apically addressed | **0.007363** | **100.00%** | **4,000** | **0** |

Accuracy alone hides the result. After enough trials, the broadcast condition
detects a weak correlation and guesses the sign, but it remains badly
calibrated because 15 of every 16 local updates contain another receiver's
consequence. Addressed instruction has about 124 times lower log loss, performs
16 times fewer updates, and trains faster in this small implementation.

At only 128 training trials, averaged over ten seeds:

| Delayed instruction | Eval bits/outcome | Eval accuracy | Hidden mapping recovered |
|---|---:|---:|---:|
| None | 1.000 | 50% | 50% |
| Broadcast scalar | 0.91 | 74% | 73% |
| Apically addressed | **0.23** | **98%** | **98%** |

## Mechanism

The local state is:

```text
eligibility_i <- basal_receptor_i
eligibility_i(t + delay) <- decay^delay * eligibility_i
```

At consequence time, no-instruction performs no update, broadcast instruction
updates every eligible receiver, and apical instruction admits the consequence
only at the context-addressed receiver. The resulting durable update is a
local count increment. There is no receiver-to-receiver matrix and no gradient
transport.

For 16 receivers the serialized durable bodies are 682--687 bytes, depending
on the mode label, and peak serialized trace state is 341 bytes. The body cost
is essentially identical across conditions.

## What This Establishes

- A long-range address and delayed consequence can drive a genuinely local
  update.
- Receiver-specific instruction greatly reduces credit noise and update work.
- Opposing receiver roles are learned without supplying their signs.
- Log loss is the necessary metric; threshold accuracy conceals weak,
  contaminated learning.

## What It Does Not Establish

The task supplies the apical destination as environmental context. It does not
yet explain how a general ALM discovers which long-range route deserves that
address. The signal carries a destination and later outcome, but it does not
carry a desired synaptic weight. Phase 41 therefore passes the routed local
credit half of LR-1, while autonomous credit-route discovery remains open.

## Reproduction

```powershell
python -m manifoldflow.experiments.dendritic_basin_long_range_credit_probe
```
