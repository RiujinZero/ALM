# Salience-Gate Dataset Runbook

## Status

Script written and smoke-tested against real shard data (windowing,
decoding, JSON parsing, span-offset resolution, and resume logic all
verified). A small real end-to-end test (3 windows, real model, real
shard data) has been run and produced clean output — see "Validated
configuration" below. **No full labeling run has been executed yet** —
this document describes how to run one, not a claim that a finished
dataset exists.

## Why this dataset exists

`ManifoldFlow`'s write trigger is currently 100% scripted, not learned:
`session_builder.py` only attaches `statement_spans` metadata to episodes
where `episode_type == "distractor"` (see
`session_builder.py:715`), and `run_step`'s write phase
(`manifoldflow_production_train.py`) only ever looks at that metadata. A
chat episode has structurally zero chance of triggering a write, no matter
what it says, because nothing scores it for salience.

This mirrors a real component of the target architecture (see
`docs/Sonic_ManifoldFlow_Grant_White_Paper.docx`, Figure 1 —
"Event-driven multimodal world-model path"): an **Event layer** that
scores salience sits between Perception and ManifoldFlow's write step.
Today, for text, that layer doesn't exist; the distractor-episode
generator stands in for it with a hand-scripted, oracle-style signal. That
scripted trigger was a deliberate choice, not an oversight — it isolates
"can Sonic use retrieved writes at all" from "what should trigger a
write," so a failure in one doesn't get misread as a failure in the
other. Retrieval accuracy trending above 0.98 in the current production
run is the evidence that the first question is answered; this dataset is
the input needed to build the second piece — a learned write-gate module,
trained on graded salience labels rather than a scripted rule.

Hand-labeling "what's worth saving" across a 134GB training corpus is not
practical. Instead, a much larger local model judges each packed training
window and outputs a graded salience score per span it finds, using the
same interleaved chat/distractor/filler content Sonic already trains on
(not a filtered or synthetic subset) as the input.

## Model

**Qwen3.6-35B, REAP-pruned (ratio 0.5), Q4_K_M GGUF** — pruning brings a
35B-parameter MoE model down to a 10.59 GB file, which is why it runs
comfortably on this machine (32GB physical RAM, 26GB WSL cap) without
touching the GPU the production training run already owns.

```
C:\Users\user\.cache\huggingface\hub\models--lennyhans--Qwen3.6-35B-REAP-Pruned-ratio-0.5-Q4_K_M-GGUF\snapshots\a20df91e1ec8163ae6da682b5df1a799c41b147b\qwen3.6-35b-reap-pruned-ratio-0.5-q4_k_m.gguf
```

Already registered as a GitHub Copilot Chat custom endpoint in this
machine's VS Code profile (that's where the model was first located, via
the `chat.cachedLanguageModels.v2` entry in VS Code's global
`state.vscdb`) — but this runbook's script talks to it directly through
its own `ik_llama.cpp` server instance, not through VS Code.

## Runtime

`ik_llama.cpp`, already built at `C:\ik_llama.cpp\build\bin\Release\`
(mainline `C:\llama.cpp` is present but **not** built — use
`ik_llama.cpp`).

Launch the server manually before running the labeling script (a 10GB
model process should be started deliberately, never spawned silently by a
batch script):

```bat
C:\ik_llama.cpp\build\bin\Release\llama-server.exe ^
    -m "C:\Users\user\.cache\huggingface\hub\models--lennyhans--Qwen3.6-35B-REAP-Pruned-ratio-0.5-Q4_K_M-GGUF\snapshots\a20df91e1ec8163ae6da682b5df1a799c41b147b\qwen3.6-35b-reap-pruned-ratio-0.5-q4_k_m.gguf" ^
    --ctx-size 65536 -ngl 999 -ncmoe 0 --reasoning off --threads 32 -np 4 --port 8080
```

Run the labeling script with `--concurrency 4` to match `-np 4`. `--ctx-size`
must scale with `-np` (65536 = 4 x the single-slot 16384, so each of the 4
slots still gets a full 16384 - the script's 8192-token windows plus up to
4096 response tokens need that much headroom per slot).

### Validated configuration

Measured directly on this machine (RTX 5080, training paused). This was
an iterative process worth recording in full since the naive versions of
several of these choices turned out to be wrong:

- **Model architecture**: 40 layers, 128 experts/layer, 8 active per
  token; ~9.93GB of repeating-layer weight across all 40 layers.
- **Concurrency's payoff depends entirely on which resource is the
  bottleneck.** At `-ncmoe 20` (still substantial CPU-resident expert
  compute), `-np 4` measured *slower* (15.6s/window) than sequential
  (13.5s/window) - CPU was already saturated even at concurrency=1, so 4
  concurrent requests just divided the same fixed thread pool, plus
  introduced two new failure modes never seen sequentially (a response
  cut off after one word; a repetition-loop degeneration). At `-ncmoe 0`
  (fully GPU-resident - see below), retesting `-np 4` gave a real,
  reproducible improvement: **6.45s/window vs 7.2s/window sequential**,
  0 parse failures, only 1/40 skipped for thinking - because GPU compute
  actually has batching slack to exploit once nothing's CPU-bound.
  Moral: don't treat a concurrency conclusion as config-independent -
  re-verify with a real timed test whenever the offload split changes.
- **GPU offload is the real lever, not concurrency.** VRAM has actual
  slack (CPU doesn't); pushing more MoE experts onto GPU via `-ncmoe`
  gave a genuine, reproducible speedup:
  | `-ncmoe` | VRAM | measured throughput |
  |---|---|---|
  | 20 (half offloaded) | ~7.7GB | ~13.5s/window (sequential baseline) |
  | 6 | ~10.9GB | ~8.0s/window |
  | 0 (fully offloaded) | ~12.3GB | ~7.8s/window (no better than 6 - diminishing returns) |

  **`-ncmoe 6`** is the sweet spot: essentially the same speed as fully
  offloaded, at meaningfully less VRAM, comfortably under the production
  training run's ~15.7GB footprint.
- **`--reasoning off` reduces thinking frequency but is not a hard
  guarantee.** Confirmed directly: one window still produced a long
  `<think>` trace despite the flag, consuming the entire response budget
  before ever reaching JSON. Two ways to handle the residual cases were
  tried:
  - `--reasoning-budget 0` (server-side hard cap on thinking): eliminated
    the failure completely (0/20 across two follow-up test batches) but
    added real cost - ~11.8s/window (up from ~8.0s) even on windows that
    were never going to think, and on the one hard window it was tested
    against, the model gave up and returned an empty span list rather
    than reasoning through the ambiguity it had correctly resolved when
    allowed to think.
  - **Client-side streaming abort (implemented in the script, no server
    flag needed): the winning approach.** The script streams the
    response and aborts the connection (skips the window, no record
    written, no penalty) the moment `<think>` has stayed open more than
    `_THINK_CLOSE_BUDGET_CHARS` (200) characters without a closing
    `</think>` - since every response, even non-thinking ones, opens
    with an immediately-closed `<think>\n\n</think>` formality (confirmed:
    an earlier version that aborted on `<think>` appearing *at all*
    skipped 20/20 windows before this was caught and fixed). Measured
    **8.4s/window, 0 skipped, 0 parse failures across 20/20** - matching
    the fastest config's speed with the forced-budget config's
    reliability, including on the same hard window that previously
    either failed or returned empty. This is the current default
    behavior (`_call_llama_server` in the script) - no `--reasoning-budget`
    flag needed on the server.
- **`--max-response-tokens 4096`** (the script's default, raised from an
  initial `1536`): a real window can legitimately contain 30+ salient
  spans (dense distractor content), and `1536` was confirmed to
  truncate mid-JSON-array on exactly such a window. `4096` cleared the
  same window cleanly.
- Distractor statements (e.g. *"Chaisal owns 6145 folders."*) were
  consistently scored 0.9+ across every test batch, confirming the
  calibration anchor described below actually works in practice, not
  just in the rubric's stated intent. Span-offset resolution (verbatim
  substring match back into the window text) typically resolves ~85-95%
  of spans; the remainder are the model paraphrasing instead of quoting
  exactly, correctly recorded as `null` rather than crashing.

## Data source

The **same pretokenized shards production training reads from**, not a
separate export:

- Path: `C:\path\to\manifoldflow_sessions`
- Layout: `batch2/` .. `batch5/` (plus a `_filler_cache/`), each containing
  thousands of `shard_w*.jsonl` files — 9,704 in `batch2` alone
- Total corpus size: ~134GB
- Each shard line is one packed session (`session_builder.write_shard`
  format): `session_id`, `token_ids`, `episodes` (with `episode_type` one
  of `chat`/`distractor`/`filler`), `total_tokens`

The script re-slices each session's `token_ids` into the same
8,192-token physical windows training uses (`PHYSICAL_WINDOW_TOKENS`,
ceil-div slicing — same math as `_precompute_windows` in
`manifoldflow_production_train.py`), decodes each window back to text via
the project's tokenizer (`resources/tokenizers/gtr_arw_english_50k`), and
sends the **whole window** — not a chat-only filtered subset — to the
model. Feeding the real interleaved mix means distractor statements show
up in the same input alongside chat, which doubles as a free calibration
check: since those statements are known-salient by construction, a
labeling pass where the model doesn't reliably score them high signals a
rubric or model problem before the run is trusted, without needing a
separate hand-labeled validation set.

## Script

[`scripts/label_salience_qwen.py`](../scripts/label_salience_qwen.py)

```bash
python scripts/label_salience_qwen.py \
    --shard-dir C:/path/to/manifoldflow_sessions \
    --max-windows 500 \
    --output data/salience_labels/qwen_pass1.jsonl
```

### Config (CLI flags and defaults)

| Flag | Default | Notes |
|---|---|---|
| `--shard-dir` | `C:/path/to/manifoldflow_sessions` | |
| `--shard-glob` | `batch*/shard_*.jsonl` | |
| `--tokenizer-dir` | `resources/tokenizers/gtr_arw_english_50k` | must match the training tokenizer |
| `--window-tokens` | `8192` | matches `PHYSICAL_WINDOW_TOKENS` |
| `--max-windows` | `500` | bounds one run; re-run to continue (resumable, see below) |
| `--output` | *(required)* | JSONL, appended to |
| `--llama-server-url` | `http://127.0.0.1:8080` | must already be running |
| `--temperature` | `0.2` | low, for consistent grading |
| `--max-response-tokens` | `4096` | raised from an initial `1536` — see "Validated configuration" above |
| `--request-timeout` | `180.0` seconds | |
| `--retries` | `2` | transient-failure retries per window, with backoff |
| `--progress-every` | `10` | console progress cadence |
| `--concurrency` | `1` | set to `4` when the server runs `-ncmoe 0 -np 4` — see "Validated configuration" above; concurrency's payoff is config-dependent, so verify with a timed test if the offload split changes again |

### Rubric (system prompt, summarized)

The model is told: it's labeling text for a persistent-memory system that
can only remember what it explicitly writes down. Write-worthy spans are
durable facts, stated states, preferences, identities, or corrections —
not greetings, filler, or unanswered questions. Templated statements
about named people/objects (e.g. *"Kevin has 4 cats."*) are called out
explicitly as the calibration anchor for "high salience" (0.8+), since
those are exactly the distractor-episode statements with known ground
truth. Full text is in `_SYSTEM_PROMPT` in the script.

### Output schema

One JSON line per window:

```json
{
  "shard_path": "C:/path/to/manifoldflow_sessions/batch2/shard_w000_00000.jsonl",
  "session_id": 0,
  "window_index": 0,
  "window_start": 0,
  "window_end": 8192,
  "spans": [
    {
      "span_text": "Kevin has 4 cats.",
      "salience": 0.9,
      "category": "fact",
      "reason": "concrete stated fact tied to a named entity",
      "char_start": 412,
      "char_end": 430
    }
  ],
  "parse_error": false
}
```

`char_start`/`char_end` are resolved via a best-effort substring search of
the model's returned `span_text` against the window's own decoded text —
`null` when not found (the model paraphrased instead of quoting verbatim;
kept rather than dropped, since the label itself is still useful even
without an exact offset). `parse_error: true` records show up when the
model's response couldn't be parsed as JSON at all — `raw_response` is
kept alongside for manual review rather than silently discarded.

`shard_path` + `session_id` + `window_index` is the join key a later merge
step will use to attach these labels back to the right training examples.

### Resumability

The script is safe to stop and re-run: on start, it scans `--output` for
already-labeled `(shard_path, session_id, window_index)` keys and skips
them, appending only new results. A labeling pass can proceed across
several sittings (e.g. multiple training pauses) without redoing work or
needing to track progress manually.

## Explicitly deferred

**Merging these labels back into the main training dataset** (so a
salience-gate module can actually train against them) is a separate,
later step — not part of this script or this runbook. The output schema
above carries enough provenance (shard path, session id, window token
range) for that step to locate the right examples without re-running the
labeling pass.

### Curation decisions for the merge step (recorded now, applied later)

Spot-checking real output (see chat log, ~150+ non-distractor spans
reviewed) surfaced a real skew worth handling explicitly rather than
leaving to the default:

- **Assistant self-identification is heavily duplicated, near-verbatim,
  across many sessions** — *"I am Vicuna, a language model trained by
  researchers from Large Model Systems Organization (LMSYS)."*, *"My name
  is Claude."*, *"I am Whisky, a language model trained by..."*, etc. This
  is an artifact of how `lmsys_chat_1m` conversations open (many share
  the same boilerplate turn), not genuine content diversity.
- **Decision: prune most of these rather than dedup, and replace the
  foreign model name with "Sonic."** Deduping would just thin the count;
  pruning-and-renaming fixes the more important problem too - training
  the salience gate on "I am Vicuna" / "My name is Claude" would teach it
  to recognize *other models'* self-identification as salient, not
  Sonic's own. The pattern being trained ("assistant self-identification
  is durable and worth writing") should be learned against Sonic's actual
  name, not perpetuate a foreign one.
- Not yet decided: exact prune ratio, or whether the same treatment
  should extend to other clearly-repeated boilerplate patterns spotted
  during the run (e.g. the "X is a leading chemical company located in
  Y" template recurring across many filler-sourced company-registry
  spans - worth checking whether that is genuine diversity or another
  near-duplicate pattern before the merge step).

## Production trainer integration

Wired into `manifoldflow_production_train.py` (not yet used to control
any write decision — this trains the scorer only; candidate-span
extraction for non-distractor content remains unsolved, see
`salience_gate.py`'s module docstring):

- `src/manifoldflow/salience_gate.py` — `SalienceGateHead` (the exact
  architecture validated in the sanity check: `Linear(d_model, 128) ->
  ReLU -> Linear(128, 1)`) and `salience_gate_loss` (BCE-with-logits).
- Always constructed in `run()`, params folded into `adapter_optimizer`'s
  existing group (alongside port/key_projection/value_projection) —
  avoids a fourth optimizer and the delicate 3-way Sonic AdamW/Muon
  split entirely.
- `--salience-labels-path` (optional): when set, `build_salience_batch`
  runs one real Sonic forward pass over a labeled window every
  `--salience-every` updates (default 10), Sonic's own hidden states
  **detached** (trains the gate head only, not Sonic — the conservative
  choice for a new, less-tested auxiliary objective riding along a
  carefully-tuned main run), pools each labeled span's token range plus
  freshly-sampled same-window negatives, and backprops
  `--salience-loss-weight` (default 0.05) times the BCE loss into
  `salience_gate`'s params before `adapter_optimizer.step()`.
- Checkpoint save/resume: `salience_gate` state is included whenever
  present; resuming from a checkpoint saved *before* this module existed
  is handled by `_extend_adapter_optimizer_state` (widens the saved
  `adapter_optimizer` state to the new param count - existing params keep
  their momentum unchanged since salience_gate's params are always
  appended at the end, never inserted).
- **Real bug caught during validation, not just anticipated**: the
  labeled dataset was built on native Windows (`D:\...` paths); the
  production trainer runs under WSL (needs `/mnt/d/...`). Fixed via
  `_resolve_salience_shard_path`, which reconstructs each shard's path
  from the *current* run's own `--shard-dir` plus just the last two path
  components of the stored path (batch dir + filename) — works
  regardless of which OS wrote the label file, no re-collection needed.
- Validated with three real CPU smoke tests against the actual
  production checkpoint (via WSL, matching its real runtime): resuming a
  pre-salience-gate checkpoint with the objective enabled, resuming a
  checkpoint that already includes it, and running without
  `--salience-labels-path` at all — all three exit cleanly, and the main
  training loss is bit-for-bit identical whether or not the auxiliary
  objective is enabled (confirms zero interference with the main run).

## Staged freeze curriculum (manual, not automated)

The intended curriculum: converge and freeze `salience_gate`, then let the
retriever (port/key_projection/value_projection) keep adjusting until
`eval_retrieval_accuracy` plateaus in the high 90s, freeze that too, then
leave Sonic as the only unfrozen thing for the rest of training. Freezing
is a manual decision at each stage (watch the logs, decide, then pass the
flag on the next resume) — deliberately not automated, unlike Sonic's own
accuracy-gated unfreeze.

- `--freeze-salience-gate` / `--freeze-retriever`: one-time instructions,
  applied via `nn.Module.requires_grad_(False)` on the relevant modules'
  params. Works regardless of which optimizer a param is grouped under
  (`salience_gate_frozen`/`retriever_frozen` are just `requires_grad`
  toggles, not an optimizer restructuring) — the same mechanism Sonic's
  own staged unfreeze already relies on.
- **Sticky across resumes**, mirroring `sonic_unfrozen`: both flags are
  saved in the checkpoint (`salience_gate_frozen`/`retriever_frozen`,
  `.get(..., False)` for older checkpoints that predate them) and
  reapplied automatically on every future resume — no need to repeat the
  CLI flag once frozen.
- Validated with two more real CPU smoke tests against the actual
  production checkpoint: freezing both via the flags (confirmed via the
  log lines and by inspecting the saved checkpoint's flags), then
  resuming *from* that frozen checkpoint with no flags at all (confirmed
  still frozen, log lines fire again automatically). Loss values were
  identical to the unfrozen case at the same update in both tests -
  freezing only affects which params receive gradient, not the forward
  pass itself.

## Held-out validation tracking (the real freeze signal)

The live heartbeat's `salience_loss` is a single tiny training batch -
too noisy to judge convergence from directly. `--salience-val-fraction`
(default 0.15) holds out a fixed split of the labeled dataset that the
training cursor never draws from; `evaluate_salience_gate` runs the full
held-out set (not a sample, for a stable comparable number) every
`--eval-every` updates and logs `salience_gate update=... val_acc=...
val_loss=... n=...` to `eval.log`, right alongside the existing
retrieval-accuracy eval line. **This is the number to watch for the
manual freeze decision** (target: plateaus in the high 90s, per
conversation), not the per-step training loss. Automatically skipped once
`salience_gate_frozen` is true - a frozen gate isn't changing, so
re-measuring it every eval checkpoint would just spend compute to
reconfirm the same number.

Validated with a real CPU smoke test (`--salience-val-fraction 0.01` for
speed): 405 examples across 37 held-out windows, `val_acc=0.827` on a
freshly-resumed, untrained gate - consistent with the ~80% majority-class
baseline these labels have (see the calibration section above), exactly
what a just-started gate should show.

## Real crash found running the real thing (and its fix)

The first real production run (`--compile layers`, `--salience-every 1`)
crashed within minutes: `torch._dynamo.exc.FailOnRecompileLimitHit`,
raised inside `build_salience_batch`'s `model.forward_hidden(ids_tensor)`
call. Root cause: `model` has compiled layers, and every labeled window
has a slightly different real token count (a shorter tail window, or
re-tokenization drift) - feeding a *compiled* model a new shape on every
call forces a recompile each time, and PyTorch hard-fails once its
recompile cache limit (8) is exceeded. This is the exact class of bug
`run_free_generation_eval`'s `canvas_len` already exists to avoid (see
that function's docstring) - it was simply missed here on the first
pass, since every CPU/`--compile none` smoke test up to that point never
exercised a compiled model at all.

**Fix**: `build_salience_batch` now right-pads every window's tokens
(with `pad_token_id`, causally inert - same reasoning as
`_generate_greedy`'s canvas) to a fixed `window_tokens` length before
calling `model.forward_hidden`, so every call - regardless of which
window - shares one identical shape. `evaluate_salience_gate` and both
call sites updated to pass `window_tokens`/`pad_token_id` through.

**Validated for real, not just re-argued**: a real GPU smoke test against
the actual production checkpoint, full production config
(`--compile layers --compile-mode max-autotune-no-cudagraphs --mixed-precision bf16 --device cuda`,
`--salience-every 1`) ran 20 consecutive real updates without a single
recompile-limit error - strictly more windows than the 8 that crashed the
first attempt. It also exercised `evaluate_salience_gate` for real:
`salience_gate val_acc=0.879 val_loss=0.272 n=5790` at update 22220. (A
disk-space error on that same run's *second* checkpoint save was
unrelated - `/tmp` is a 13GB RAM-backed tmpfs on this machine, filled by
accumulated smoke-test checkpoints across this session, not a code bug;
cleaned up, and the real run writes to `/path/to/manifoldflow_production/`
on an ordinary disk with 100+GB free, so this can't recur there.)

## A second, unrelated recompile-limit crash (pre-existing, not from this work)

The relaunched run survived past the salience-gate fix but crashed again
at update 23000 - the exact moment `--free-gen-eval-every 1000` fired for
the first time since the resume. Same root cause, different call site:
`_binding_context_and_memory` (the free-gen eval's "binding" category)
called `model.forward_hidden` on each prompt's own raw, unpadded
write+filler+query length - three binding prompts, three different
lengths, three more shapes fed to the compiled model. This function
predates the salience-gate work entirely; it was simply never exercised
under a real long `--compile layers` session before, since free-gen-eval
only fires every 1000 updates and this codebase's runs hadn't stayed up
long enough post-`--compile` for it to fire and crash until now.

**Fix**: `run_free_generation_eval` now does a pure-tokenization first
pass (no model calls) over every prompt - both plain and binding - to
compute one shared `canvas_len` before any forward pass runs, then passes
it into `_binding_context_and_memory` too, which right-pads its own
`context_ids` to that length exactly like `_generate_greedy` already did.
Both of this eval's forward-pass call sites now share one shape for the
whole pass, not just one of the two.

**Validated for real**: a GPU smoke test with `--free-gen-eval-every 1`
(firing on every one of 3 real updates - 9 binding-prompt forward passes
total, more than the 3 that crashed the first time) completed cleanly,
`EXIT_CODE=0`, checkpoint saved. Generated binding-prompt output itself
was empty/degenerate at this point in training (expected - matches this
session's earlier finding on greedy decoding at this training stage, not
a bug in this fix).

## Sonic's own recurrent state, threaded through the lanes

Every physical window, in both the main training loop and the salience-
gate's own auxiliary forward pass, used to call `model.forward_hidden`
with no recurrent state at all - Sonic started every window "cold,"
regardless of whether it was the first window of a session or the
fourth. `forward_hidden` has always supported carrying a per-layer
`states` tuple across calls (that's the entire point of an SSM), the
trainer simply never exercised it. This matters specifically for the
salience gate: training scores a span with the benefit of everything
earlier *in that physical window* as real causal context, but a live,
turn-by-turn write-decision (the eventual point of the gate) would only
ever see newly-arrived content - carrying Sonic's own state properly is
what makes those two views of "how much context does the model actually
have" the same, rather than a silent train/inference mismatch baked into
the trainer itself.

**Fix**: `Lane` now carries its own `recurrent_state` (each layer's own
`[1, ...]` row), threaded into `model.forward_hidden` on every
`advance_window()` call and reset only in `reset_session()` - carried
across window boundaries *within* a session, reset only *between*
sessions, never reset mid-session. Detached before storing (truncated
BPTT at the window boundary, the same convention `Lane.detach_()` already
applies to the bank's own persistent state) - values carry forward,
gradients don't.

**Three real, compounding `--compile layers` crashes, found and fixed in
sequence** (not guessed - each confirmed via a real GPU smoke test against
the actual production checkpoint, full `hgrn`/`bf16`/`compile` config):

1. The carried state's dtype wasn't pinned - autocast resolved it
   differently depending on call history, and `fullgraph=True` guards on
   the *input* dtype even though the SSM layer casts it internally
   regardless. Fixed by always storing state as `float32`.
2. A second lane set starting cold (`eval_lanes`, which had never run
   when `train_lanes` already had real carried state) reintroduced a
   Python `None` into `forward_hidden`'s `states` argument well after
   the graph had already specialized for "always a real tensor" -
   `None` vs. tensor is a genuine recompile trigger under `dynamic=False`.
   Fixed with a module-level `_LANE_STATE_REFERENCE` (the first real
   state tuple the whole process ever produces) so any lane with no state
   yet - never run, or freshly reset - gets an all-zero *tensor* instead
   of `None`; `states=None` now happens at most once, for the process's
   true first call.
3. `build_salience_batch` calls the same compiled `model.forward_hidden`
   but, before this, under a genuinely different convention - no
   autocast, always `states=None` - and does so every single update when
   `--salience-every 1`, alternating with `run_step`'s own calls.
   Matched it to `run_step`'s exact convention (same autocast wrapping,
   `_zero_state_like_reference()` instead of implicit `None`).

Even after all three, the *combined* number of distinct shape/dtype/state
specializations three call sites (`run_step` on `train_lanes`, `run_step`
on `eval_lanes`, `build_salience_batch`) legitimately need still exceeded
torch's default `recompile_limit` (8) - not runaway thrashing, confirmed
via a full `TORCH_LOGS=recompiles` trace showing the set of reasons
recurring rather than growing without bound. Raised to 32 alongside
`model.compile_layers()`, per the error's own hint.

**Validated for real**: a full real GPU run (`--compile layers
--scan-backend hgrn --mixed-precision bf16`, `--salience-every 1`, real
6-lane batch) against the actual production checkpoint ran 24 consecutive
updates, then cleanly passed *both* end-of-run evals that had crashed on
every earlier attempt - `eval retrieval_acc=1.0` and `salience_gate
val_acc=0.9458` (a new best, above the prior 0.926-0.939 plateau) - plus
free-generation eval and a checkpoint save, `EXIT_CODE=0`. A pure-Python
unit test (no model, synthetic tensors) separately confirmed the
stacking/zero-fill/reset math in isolation before the real-model test was
trusted.

## Gate-driven writes for chat/filler content, and why it isn't ready

Distractor's write trigger is scripted; the whole point of the gate is to
also catch write-worthy content in chat/filler, which has no equivalent
scripted mechanism. `--gate-writes` implements this: every chat/filler
episode's *whole turn* is a candidate (`Window.gate_candidates`, built
from `start`/`end` boundaries `session_builder.py`'s `pack_session`
already records for *every* episode type, not just distractor's
`statement_spans` - no corpus regeneration needed), scored by the
currently-training `salience_gate`, written if it clears
`--gate-write-threshold`.

**Structurally separate bank, not a shared one.** Gate-driven writes go
into each lane's own `gate_bank`, never the scripted-write `bank` -
`Lane` now carries both. A distractor query's nearest-neighbor search
over `bank` can never land on gate-written content, by construction, not
just because it happened not to in testing. `gate_bank` also gets a real
(nonzero) `--gate-novelty-threshold`, unlike `bank`'s deliberate 0.0 -
`BasinMemory.write()`'s own merge-into-existing-basin logic (already
built, not reinvented here - see `continuous_memory.py`) is exactly the
tool for content that legitimately recurs (repeated assistant self-ID,
templated boilerplate), which `bank`'s `write_always_new` deliberately
never merges (correct for distractor's genuinely-distinct-each-time
statements, wrong for chat).

**Real GPU validation (100 real steps, `--gate-writes` on, full
production config)**: `retrieval_acc` held at 0.9938 across two eval
checkpoints spanning 20 updates, unaffected by gate writes mixed into the
bank (this was measured with the *old*, shared-bank version, before the
structural-separation refactor above - a stronger, not weaker, guarantee
now applies). Gate write-rate settled from 11.4% (110/967 candidates) to
6.2% (60/967) over those same 20 updates.

**Spot-check (`scripts/inspect_gate_writes.py`) found a real quality
problem**: at threshold 0.5, a meaningful share of what got written was
generic assistant self-identification and boilerplate ("I am NAME_1, an
AI assistant created by Anthropic...") rather than genuine per-
conversation salience - traced to a curation gap already recorded (but
never implemented) in this doc's own "Curation decisions for the merge
step" section above: Qwen's rubric calls self-identification "durable and
worth writing," and every duplicate of it across `lmsys_chat_1m`-sourced
sessions became another positive training example.

**Fixed**: `scripts/prune_boilerplate_labels.py` removes matched
boilerplate spans from the label file's own positive examples (207 of
58,875 spans pruned from `qwen_pass1.jsonl` - a narrow, high-precision
pattern match, not a broad heuristic; see the script's own docstring for
why breadth wasn't the goal). Applied in place, original preserved as
`qwen_pass1.jsonl.pre_prune.bak`.

**The bigger, unresolved finding: `scripts/round_trip_gate_retrieval_test.py`
shows the underlying retrieval mechanism does not generalize to natural
language.** Nothing in the training corpus ever generates a query
against chat/filler content - `eval_retrieval_accuracy` only ever
measures the scripted-distractor path, which uses fixed templates
("X has N Y"). This script is the missing round-trip check: ten hand-
authored (write, query, paraphrase) triples styled like real gate-
selected content (stated decisions/preferences/facts), run through the
same pool -> `key_projection` path `run_step` uses, against a real
production checkpoint (update 24600).

Results:
- **Retrieval accuracy: 5/10 = 0.500** among ten competing real facts -
  far below distractor's 99%+, on content that's arguably *easier*
  (fewer, more distinct facts than a real session's bank would hold).
- **Novelty-threshold sweep (0.05 to 1.2): no usable operating point.**
  At low thresholds (<=0.2) nothing merges at all (true_merge_rate=0,
  false_merge_rate=0 - inert). Above that, false merges grow *faster*
  than true merges (0.5: 30% true / 20% false; 0.8: 20% true / 50% false;
  1.2: 10% true / 80% false), and one basin acted as a false-merge
  magnet that most other facts spuriously landed on at higher
  thresholds.

**Conclusion, not yet acted on**: `key_projection` was trained
exclusively on templated distractor statements (via the main loop's
`info_nce_retrieval_loss`) and its own oracle-queried retrieval - this is
a real, measured generalization gap to natural-language facts and
questions, not a threshold-tuning problem or a bug in this session's new
code. `--gate-writes` and `gate_bank`'s novelty mechanism are both
correctly *built*, but turning `--gate-writes` on for the live production
run would currently be writing content that can't be reliably retrieved
again - training time spent without validated payoff, exactly the
concern that motivated building this test. The concrete next question
this raises (not yet decided): does `key_projection` need training
signal from natural write/query pairs, not just templated distractor
ones, before gate-driven writes are worth relying on?

## Contrastive retrieval training: three attempts, and why none of them worked

Direct follow-up to "Gate-driven writes for chat/filler content, and why
it isn't ready" above. That section's conclusion was that `key_projection`
doesn't generalize past templated distractor statements; the natural next
step tried here was training it on real contrastive data (Qwen-generated
paraphrase/hard-negative pairs) instead of more of the same templated
signal - the same data-distillation pattern that already worked for
salience labeling, aimed at the retrieval problem instead.

**Pipeline built**: `scripts/generate_contrastive_pairs.py` (Qwen
generates one paraphrase + one hard negative per real anchor statement,
same llama-server/streaming/think-abort conventions as the salience
labeler), `scripts/sample_diverse_anchors.py` (pulls anchors from every
real text source on the G drive - wikipedia, LSAT, formal logic
(proofwriter), math CoT, several chat styles, GSM8K, TinyStories, Python -
not just the existing labeled corpus, with a cheap regex filter for
markdown/LaTeX sentence-splitter artifacts), `scripts/
train_contrastive_key_projection.py` (in-batch InfoNCE: each pair's
paraphrase is the query, its own anchor the correct target, every other
pair's anchor + every pair's hard_negative in the batch are negatives -
Sonic stays frozen throughout, only `key_projection`'s own small linear
layer trains).

**Three runs, escalating scale, identical result**:

| Run | Contrastive pairs | In-batch val_acc (training's own metric) | round-trip retrieval_acc |
|---|---|---|---|
| 1 | 1,540 (existing corpus only) | 0.19 -> 0.68 | 0.500 -> **0.300** |
| 2 | 8,782 (+ smoltalk COT) | 0.25 -> 0.60 | (baseline) -> **0.300** |
| 3 | 26,124 (+ 17 diverse sources) | 0.29 -> 0.65 | (baseline) -> **0.300** |

Every run's in-batch training accuracy improved with more data (as
expected - that's the direct training objective). The novelty-merge
behavior also genuinely improved run over run (run 3: true_merge_rate=0.7
at threshold=0.8 with **zero** false merges, the cleanest separation
found all session). But `round_trip_gate_retrieval_test.py`'s actual
retrieval accuracy on the 10 held-out cases landed at **exactly 0.300 in
all three runs** - not just "still bad," the identical number every time,
with the same specific cases ('deploy_day', 'team_lead') acting as
generic false-attractors across all three.

**Root cause, confirmed directly (`scripts/diagnose_raw_pooled_similarity.py`),
not inferred**: Sonic's own raw, mean-pooled hidden states for these same
10 cases - with **no key_projection applied at all** - give the exact
same 0.300 nearest-neighbor accuracy, and the exact same attractor
pathology (2-4 anchors absorb every query's "nearest" pick; the other 6-8
anchors never win once, regardless of which query is asked). This is
conclusive: the ceiling is Sonic's own frozen representation for natural-
language content, not `key_projection`'s training or data volume.
`key_projection` is `nn.Linear(dim, dim, bias=False)` - a linear map can
only rotate/rescale existing geometry, it cannot introduce discriminative
structure that isn't already latent in the raw vectors' relative
arrangement. Since the raw vectors already collapse several unrelated
sentences toward the same region, no amount of contrastive training data
through a *linear* projection was ever going to fix this - confirmed
empirically (three escalating attempts, identical failure) and now
explained mechanistically (the raw-vector control test).

**What this means going forward**: more contrastive data is not the
lever. Two real candidates, neither tried yet:
1. Give `key_projection` actual nonlinear capacity (e.g. `Linear -> ReLU
   -> Linear` instead of one linear layer) - a linear map provably cannot
   separate points a nonlinear one might, and this is a small, cheap,
   low-risk change to test against the exact same round-trip harness
   before considering anything more invasive.
2. Address Sonic's own representation quality for natural language (e.g.
   unfreezing Sonic during contrastive training, or a different pooling
   strategy than plain mean-pooling) - a much bigger, riskier change than
   (1), and not the first thing to reach for.

**Decision made autonomously (overnight, per explicit user instruction to
proceed without waiting for confirmation)**: did not launch a fourth,
larger data-generation cycle - the raw-vector control test shows
definitively that scale isn't the bottleneck, so a fourth attempt at the
same fix would have cost hours for a predictable non-result. Production
training was resumed as-is (original checkpoint, unmodified - the new
`key_projection` never cleared the user's stated bar of mid-90s
retrieval_acc with sane merge behavior, so there was no basis to swap it
in). `--gate-writes` remains off.

## Fourth attempt: nonlinear key_projection - made round-trip retrieval worse, not better

Direct follow-up to candidate (1) above. `SonicKeyProjection` was
upgraded from a single `Linear` layer to `Linear -> ReLU -> Linear`
(`src/manifoldflow/sonic_key_projection.py`), with `hidden_dim = 2*dim` by
default specifically so a fresh instance - or one built via the new
`from_legacy_linear` classmethod - can reproduce an arbitrary linear map
*exactly* at construction time, via `relu(x) - relu(-x) = x`. This let the
production run's own ~25,000 updates of already-trained linear
`key_projection` weights migrate losslessly into the new architecture
(verified: the migrated module reproduces the old checkpoint's exact
output on an arbitrary probe batch, atol=1e-4) rather than discarding
them and starting over. `load_key_projection_state_dict` (detects old
`"projection.weight"` vs. new `"layer1.weight"/"layer2.weight"` schema)
is now the single call site every checkpoint-loading script uses. The
`adapter_optimizer`'s own saved Adam momentum needed a matching fix
(`_migrate_adapter_optimizer_state_for_key_projection_upgrade` in
`manifoldflow_production_train.py`): `key_projection`/`value_projection`
going from one param to two breaks the existing
`_extend_adapter_optimizer_state`'s pure-append assumption, so the new
function explicitly drops the now-shape-incompatible key/value-projection
momentum (nothing to migrate a `[dim,dim]` Adam buffer to when the param
becomes two differently-shaped tensors) while preserving `port`'s prefix
state and remapping `salience_gate`'s suffix state to its shifted ids -
verified via unit tests plus a full integration test that resumes a
synthetic old-format checkpoint through the real code path and confirms
`optimizer.step()` runs with no shape-mismatch crash.

**Retrained key_projection on the same 26,124-pair contrastive dataset,
warm-started from the production checkpoint's trained linear weights (not
random init), on GPU**: in-batch metrics looked like real progress
(train_acc 0.30 -> 0.70, val_acc 0.30 -> 0.56 over 8 epochs - noticeably
better than any of the three linear runs' in-batch numbers). But
`round_trip_gate_retrieval_test.py` against the resulting checkpoint gave
**retrieval_acc = 0.100** - worse than the linear ceiling (0.300), not
better. The added nonlinear capacity let the network fit the contrastive
dataset's own narrow distribution harder (hence the better in-batch
val_acc, itself measured on a held-out split of that *same* distribution)
without improving - here, actively hurting - generalization to the
round-trip test's genuinely different natural-language style. This is
consistent with, not contradicting, the earlier raw-vector diagnostic:
if the discriminative signal isn't already present in Sonic's own frozen
pooled representation, giving key_projection more capacity to fit
*whatever signal the training data does contain* just makes it overfit
that signal harder, which is a worse outcome than a linear map that
couldn't overfit as aggressively in the first place.

**What this means**: neither more contrastive data (three linear rounds)
nor a nonlinear key_projection has closed the gap between "retrieval
works great on scripted distractor content" (retrieval_acc consistently
>0.98 in production's own main-loop metric) and "retrieval works on real
natural-language content" (stuck at 0.10-0.30 on the round-trip test
across every variant tried). Both remaining candidates touch
key_projection alone and are now empirically exhausted. The one
untried, structurally different direction is addressing Sonic's own
representation quality directly (unfreezing Sonic during this kind of
contrastive training, a different pooling strategy than mean-pooling, or
the user's own proposed mechanistic-interpretability angle - probing
Qwen's own weights for whatever lets *it* do real-time salience
discernment, to see if an analogous mechanism can be identified or
distilled into Sonic rather than bolted on afterward as a separate
projection).

**Production impact: none, and none expected.** The architecture upgrade
itself (module weights + optimizer state) is safe and already live -
production training was resumed on it (confirmed: clean resume log line,
no Traceback, GPU actively computing afterward) - because production's
own `key_projection` training signal is the scripted-distractor
`info_nce_retrieval_loss` that already works (>0.98 retrieval_acc with
the *old* linear projection), a completely different distribution than
this round-trip test's natural-language probes. The nonlinear
architecture gives that in-domain signal strictly more capacity to work
with, not less - there is no reason to expect it to regress production's
own metric, and every reason (more capacity, same working signal) to
expect it holds or mildly improves. The negative result above is scoped
entirely to natural-language generalization, which production's main
loop was never training for in the first place.

## Fifth attempt: unfreezing Sonic during contrastive training - first real improvement

Direct follow-up to the "what this means" candidate above. Rather than
touching `key_projection`'s architecture again, `scripts/
train_contrastive_key_projection.py` gained `--unfreeze-sonic`: Sonic's
own params join the contrastive training run too, split exactly the way
the production trainer already does (`split_sonic_params_for_muon` - Muon
for the 2D hidden weight matrices, a small AdamW for everything else),
at the same rates production uses (`sonic_lr=1e-5`, `sonic_muon_lr=0.02`).
Motivation: production's own Sonic has been unfrozen since update 21801,
but on the *scripted*-distractor retrieval signal - a different
distribution than the round-trip test's natural language - so it never
actually tested whether letting Sonic's representation adapt to *this*
objective would help. This run is the first real test of that.

Warm-started from the same production checkpoint (update 25200, already
carrying the nonlinear `key_projection`), same 26,124-pair contrastive
dataset, 8 epochs, GPU (production paused for the run's duration - full
Sonic backward pass, not just key_projection, needs the GPU headroom):

| Epoch | train_acc | val_acc (in-batch) |
|---|---|---|
| 1 | 0.72 | 0.759 |
| 2 | 0.81 | 0.821 |
| 4 | 0.88 | 0.844 |
| 6 | 0.92 | 0.845 |
| 8 | 0.955 | 0.849 |

In-batch val_acc alone (0.849) is already far beyond anything the frozen-
Sonic runs reached (best was 0.68, and that came with retrieval_acc still
stuck at 0.300). More importantly, **`round_trip_gate_retrieval_test.py`
gave retrieval_acc = 0.400** - the first result to break the 0.300
ceiling that held identically across three linear rounds, the raw-vector
control, and the nonlinear-key_projection round. Novelty-merge behavior
also improved substantially: true_merge_rate reaches 0.70-0.80 at
threshold 0.20-0.30 with **zero false merges** - the cleanest separation
seen all session, better than the nonlinear run's own merge numbers.

**Still far from solved**: 0.400 means 6 of the 10 held-out cases still
retrieve the wrong content - a real improvement (33% relative gain over
0.300), not a fix. val_acc also visibly plateaus from epoch 4 onward
(0.844 -> 0.845 -> 0.849) while train_acc keeps climbing (0.88 -> 0.96) -
the same overfitting shape that hurt the nonlinear-only run, just not yet
enough to erase the underlying gain. Epoch 5 (val_acc=0.846, before the
gap really opens up) might generalize just as well with less drift risk -
not tested, a reasonable next refinement (early stopping on val_acc)
before pushing scale further.

**Unresolved before this checkpoint could be trusted for anything real**:
this training touched Sonic's own weights on a narrow 26k-pair dataset
for 8 epochs at production's own Sonic learning rates - meaningfully more
exposure to catastrophic forgetting than key_projection-only training.
Sonic's general language-modeling quality on this checkpoint has **not
been checked** (no free-generation sanity pass run against it yet).
Production was resumed unmodified rather than folding this checkpoint
in - swapping Sonic's own weights into the live run is a bigger,
riskier change than any prior swap this session considered, and doing so
deserves a forgetting-quality check plus explicit confirmation first, not
an autonomous decision.

Checkpoint: `/tmp/contrastive_key_projection_unfreeze_sonic/latest.pt`
(scratch, never touched the production checkpoint).

**Follow-up free-generation sanity check: severe catastrophic forgetting -
this checkpoint is not usable as-is.** Ran production's own
`run_free_generation_eval`/`FREE_GEN_PROMPTS` directly against this
checkpoint (`scripts/free_gen_sanity_check.py`, standalone). Every single
prompt produced near-pure gibberish - repeated punctuation with no real
words (`'. ( . (,. (,. (,. (,...'`, `'. I\n. I\n. I\n...'`), including the
binding prompts, which used to at least echo real (if wrong) words
(`'A...............................................'` for a password
recall). Compare to production's own free-gen samples at the *same*
update (25200) before this experiment: already imperfect and repetitive
("The road. The road. The road...") but still real English words in
real sentence structure. 8 epochs of Sonic's own weights training against
a narrow 26,124-pair contrastive dataset, even at production's own
conservative rates (`sonic_lr=1e-5`, `sonic_muon_lr=0.02`), was enough to
destroy Sonic's basic generative ability entirely.

**This means the 0.400 round-trip retrieval result, while real, is not a
usable outcome as measured** - it was bought entirely at the cost of the
thing that has to actually generate text. Two direct implications:
1. This checkpoint must not be fed back into production in any form.
2. The retrieval-vs-generation tradeoff needs to be tracked together,
   not separately, from here on - any future attempt at unfreezing Sonic
   (this dataset or a different one) should run this same free-gen sanity
   check *before* declaring the retrieval number a win, not after. Worth
   trying: far fewer epochs (the val_acc plateau started around epoch 4-5;
   an epoch-1 or epoch-2 checkpoint may have gained some retrieval
   improvement with much less generative damage - untested), a much
   smaller sonic_muon_lr/sonic_lr than production's own rates (which were
   tuned for a much larger, more diverse training mix, not this narrow a
   dataset), or mixing in a small fraction of plain-language continuation
   data alongside the contrastive pairs so Sonic's own next-token
   objective isn't entirely abandoned during this fine-tune.

## Open items before trusting a full run

- The calibration check has held across every test batch run so far (63
  windows total across several configs) but all of it is still small
  relative to the ~42,000-window target for the full run — worth
  continuing to spot-check as a real run proceeds, not just at the start.
- No hand-labeled spot-check sample exists yet to catch systematic
  labeling bias.
- Not yet measured: what fraction of windows get skipped for genuine
  reasoning at real-run scale (0/20 in the latest test batch, but that's
  a small sample) — worth watching once a large run is underway, since a
  high skip rate would mean losing more hard-but-valuable windows than
  expected.
- The rubric's category set (`fact` / `preference` / `identity` /
  `correction` / `decision`) hasn't been validated against real model
  output distribution — may need adjustment once a real batch of labels
  exists to look at.
