# NCCA-v1: Natural-Chat Consequence Ablation

## Status

**Prospective design / preregistration draft. No outcome run has been performed.**

Execution is gated on the causal-antecedent context-coupled receiver passing its
predeclared receiver gate. The protected test payload must remain unopened until
this protocol, implementation, dev choices, and output schema are frozen.

## Question and claim boundary

NCCA-v1 asks whether a strictly later interlocutor event can provide a
self-supervised teaching signal for a coarse response action, and whether that
signal can steer an independently validated native byte writer.

The mechanism under test is:

```text
causal conversation state H_t
    -> supported coarse action A_t
    -> action-conditioned response R_t
    -> later USER turn U_(t+1)
    -> frozen compression outcome r_t
    -> TD / eligibility credit to (H_t, A_t)
```

This is not a test of human utility, satisfaction, intent recovery, or agency.
In an observational chat log, only one response and one later user turn exist
for a given history. Alternative-action outcomes are not identified.

## Relation to existing mechanisms

- Reuse the trajectory loader's strictly future role bounds and its causal
  pending-event plus multi-timescale history representation.
- Reuse Phase 153's external-only temporal-difference discipline, but replace
  its same-stream next-byte utility with held-out next-USER compression gain.
- Reuse the context-coupled writer only after its causal-antecedent receiver
  gate passes.
- Keep future-successor state as a compatibility diagnostic. It is not the
  consequence value and must not be presented as a goal representation.

## Data rows

Derive a `decision-transition-v1` index from schema-v4 training data. A binary
rebuild is unnecessary for training: `train.index.jsonl` already contains the
full conversation and absolute offsets for every turn, although the current
loader selects only the latest eligible assistant target.

For every eligible assistant decision, emit one metadata row with:

- `decision_id = sha256(episode_id || assistant_turn_index)`;
- conversation/episode ID, source label, dominant script, horizon bin, and
  response-length bin;
- `H_t`: episode start through the end of the immediately preceding USER turn;
- `R_t`: the logged ASSISTANT content, with a primary 128-byte action window;
- `U_t1`: the first later USER content, with a primary 128-byte score window;
- `H_t1`: episode start through `U_t1`, which is the next assistant decision
  boundary;
- the next logged assistant action label, when present, and a terminal flag
  otherwise; and
- range hashes plus source provenance.

Every row must satisfy:

```text
H_t.end <= R_t.start < R_t.end <= U_t1.start < U_t1.end <= H_t1.end
USER index immediately before R_t < R_t index < U_t1 index
```

Each state excludes the response being selected at its own boundary: `H_t`
excludes `R_t`, while `H_t1` may contain the now-past `R_t` but excludes the
next response `R_t1`. Role markers are excluded from the response and
next-user scoring windows.

### Train-only tractability census

A metadata-only census of the existing 2,131 schema-v4 training conversations
found:

| Quantity | Count |
| --- | ---: |
| Eligible ASSISTANT-to-next-USER transitions | 6,567 |
| Transitions with a subsequent assistant decision | 6,429 |
| Terminal after the next USER turn | 138 |

All transitions from one conversation stay in one fold. Statistics and
bootstraps cluster by conversation, never by byte or transition.

For dev/test, the natural row is the original unanswered prefix, original full
response, and first USER event parsed from the successor target. The synthetic
interruption/bridge/resume text is not part of `H_t` in this assay. A derived
sidecar should expose these ranges explicitly before execution. Test remains
sealed until the final run.

## Causal state

At the decision boundary, compute the existing event evidence with:

- 96-dimensional signed-hash unigram/bigram event vectors;
- the current pending USER vector; and
- normalized exponential moving histories with decays `(0.5, 0.85, 0.97)`.

This gives a 384-dimensional continuous `H_t`. For local tables and writer
addresses, discretize it with the receiver runner's deterministic top-eight
absolute coordinates plus sign IDs.

At inference, only `H_t` is available when the action is selected. In offline
training, the logged action label is assigned after the complete response is
observed. The implementation must buffer the earlier state and mark this as a
hindsight-supervised action label; it must not describe the logged system as
having intentionally selected that latent action.

## Coarse actions without semantic labels

Construct an outcome-blind action tokenizer using training responses only:

1. Encode the first 128 assistant-content bytes with the existing normalized
   signed-hash unigram/bigram vector.
2. Fit a train-only 32-dimensional PCA.
3. Fit deterministic spherical k-means with seed `82972` and initial `K=16`.
4. Merge the least-supported centroid into its nearest centroid, using response
   counts only, until every retained action has at least 64 transitions in each
   fitting fold.

Cluster IDs remain anonymous; no human semantic names or labels are added.
The observed action is the nearest centroid to `R_t`.

Fit a behavior-support model `mu(A | H)` only on the selector fold. During
generation, an action is eligible only if it meets the global count criterion
and `mu(A | H) >= 0.05`. If fewer than two actions are eligible, abstain and use
the behavior argmax. Report action counts, coverage, and abstention rate.

## Three-way cross-fitting

Assign complete conversations deterministically and within source strata to
three near-balanced folds A, B, and C. The action tokenizer may use response
bytes from all training folds because it never sees a later user target.

Run three rotations:

| Rotation | Reward evaluator | Selector and writer | Audit evaluator |
| --- | --- | --- | --- |
| 1 | A | B | C |
| 2 | B | C | A |
| 3 | C | A | B |

The reward evaluator creates selector-fold outcomes. The audit evaluator is
never used to create those outcomes or update that selector. Evaluators never
fit on selector conversations. Report rotations separately and require the
same sign; do not treat them as independent rows in a pooled significance test.

Dev may choose only a predeclared small fusion/discount grid. Freeze all choices
and hashes before one test run.

## Held-out next-interlocutor compression outcome

On each reward-evaluator fold, fit two frozen, allocation-matched byte
compressors:

```text
q_H (u_j | u_<j, H_t)
q_HA(u_j | u_<j, H_t, A_t)
```

Use `ContextCoupledByteWriter` with `table_bits=14`, local orders
`(1, 2, 4, 8)`, identical learning settings, and independently bounded USER
episodes. Both models receive exactly 17 persistent slots:

- `q_H`: top-eight `H_t` IDs, one NULL-action ID, and eight
  `hash(H_i, NULL)` IDs;
- `q_HA`: top-eight `H_t` IDs, the observed action ID, and eight
  `hash(H_i, A_t)` IDs.

Also fit an allocation-matched negative-control compressor whose action labels
are Sattolo-shuffled within source, script, horizon, and response-length strata.
A full-response-conditioned evaluator may be reported only as an information
ceiling, because the selector cannot choose an exact logged response.

For a row never used to fit the evaluator, define:

```text
r_raw(t) = [bits_qH(U_t1) - bits_qHA(U_t1)] / |U_t1|
         = mean_j log2(q_HA(u_j) / q_H(u_j))
```

Positive `r_raw` means that the coarse response action carries information
about later interlocutor bytes beyond what was present in the prior history.
Clip to `[-2, +2]` bits/byte only for learning; report the unclipped outcome.

This is the least-assumptive available self-supervised proxy because it is a
literal held-out code-length difference, the target is produced by the other
party, and the target occurs strictly after the response. It still measures
predictive dependence, not desirability or causal benefit.

## TD and eligibility path

Replay selector-fold conversations in chronological order; conversation order
may change by seed, but transition order within a conversation may not.

At `H_t`, lay eligibility for the observed action and each active
`(signed_H_feature, action)` relation. Other actions are incompatible. Use an
event clock, advancing once at assistant completion and once at next-USER
completion, with eligibility half-lives `(1, 2, 4, 8)` events.

At the end of `U_t1`, and only then, update the state critic:

```text
delta_t = r_t + gamma * V(H_t1) - V(H_t)
```

Use `LocalValueEstimate` with primary `gamma=0.8`; terminal bootstrap is zero.
Pass `delta_t` to `TemporalRelationalEligibility.confirm(..., external=True)`.
The resulting signed feature-action strengths form the actor score. This avoids
a max-Q bootstrap over actions never observed in that state.

Required temporal controls are `gamma=0`, eligibility reset immediately before
the USER outcome, and zero trace. Gamma zero is a valid one-step consequence
learner; a multi-turn trajectory claim requires `gamma=0.8` to beat it.
Self-generated or recalled bytes never supply confirmation.

## Writer connection

Run this section only if the causal-antecedent receiver gate passes. Train one
writer family on selector-fold logged response windows with persistent IDs:

```text
top8(H_t) + A_t + hash(top8(H_t), A_t)
```

At inference, substitute the actor-selected supported action and hold it for
the 128-byte response. Fuse the allocation-matched action/local likelihood ratio
into the frozen native 500 MB base using the already-frozen receiver beta.
Writer weights are identical across policy arms; only selected action changes.

Required arms are:

- native exact bypass (`beta=0`);
- receiver-only behavior action;
- consequence-selected action;
- stratified reward-shuffle selector;
- action ID disconnected from the writer; and
- oracle logged action, as a receiver ceiling only.

Pending-only history and intact-versus-reset eligibility are required mechanism
controls. Action compliance is nearest-centroid agreement of the generated
128-byte vector. It must be paired with the existing independent raw-ngram /
reference and native plausibility non-inferiority checks so a degenerate
codebook cue cannot count as success.

## Sequential gates and endpoints

1. **Receiver:** causal-antecedent coupled writer beats additive-wide and
   swap/shuffle controls in teacher-forced BPB and free output. Otherwise stop.
2. **Proxy observability:** on an evaluator-disjoint fold, logged actions improve
   next-USER BPB over `q_H`, with a conversation-bootstrap 95% CI above zero,
   and beat the stratified action-shuffle compressor. Otherwise stop.
3. **Non-counterfactual value learning — primary offline endpoint:** before the
   USER target is revealed, the score assigned to the action actually logged
   predicts its subsequently observed audit-evaluator `r_raw` better than
   state-only and action-prior baselines, out of sample at conversation level.
4. **Surrogate policy screen — secondary:** on dev,
   `r_audit(H, A_selected, U_logged)` beats behavior argmax, reward shuffle,
   pending-only, and reset controls with CI above zero. This is explicitly a
   model-scored counterfactual, not realized policy uplift.
5. **Writer actuation:** selected action changes output, improves action
   compliance over disconnected/behavior controls, and preserves the receiver's
   independent plausibility/reference gate. Oracle action gives the ceiling.
6. **Multi-turn credit, if claimed:** `gamma=0.8` beats `gamma=0` on rows with a
   next decision. Failure does not invalidate a one-step external-consequence
   result.

## Identifiability limit

Purely observational chat cannot reveal what the same interlocutor would have
said after an alternative response. Cross-fitting, propensity estimates, and
a stronger compressor reduce fitting artifacts but do not remove unobserved
confounding or create missing counterfactuals.

An offline pass can support only this claim:

> Strictly later external interlocutor bytes provide a learnable,
> self-supervised predictive consequence signal that can steer a causal writer.

It cannot establish satisfaction, helpfulness, intended-goal consistency,
causal influence on the user, or goal-directed agency. A system can improve
this proxy by eliciting a predictable or repetitive reply.

## Randomized extension required for a causal claim

The first causal test is a genuinely prospective randomized deployment among
behavior-supported actions. Assign the writer action at `H_t`, freeze both
reward and audit compressors before enrollment, observe the next USER event,
and compare intention-to-treat `r_raw` against the receiver-only policy with
conversation-clustered uncertainty.

Predeclare a response window. Treat no reply within that window as a separate
terminal symbol/outcome rather than dropping it, to avoid survivorship bias.
Passing this extension identifies a causal policy effect on the compression
proxy only. It still does not identify human utility or thoughtful behavior.
