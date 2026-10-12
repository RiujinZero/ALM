# ALM Language Harness Runbook

## Status

`src/manifoldflow/dendritic_basin/alm_language.py` is implemented and
smoke-tested: pure-Python backend (train/generate/evaluate/save/load),
native backend single-column, and native backend multi-column
(`ProcessPoolExecutor` + `consolidate_delta`) have all been run end to end
against real byte streams. Unit coverage lives in
`tests/test_alm_language.py` (8 tests, all passing). This is a new,
standalone module — it does not touch, import from, or get imported by
`dendritic_basin_phase126_three_epoch_generation.py`, so it is safe to use
alongside any production run driven by that script.

## What this is

A generic core for configuring, training, and generating from an ALM
(Attractor Language Model) — the local-learning, no-backprop, byte-level
cortex used throughout this project — without needing to know anything
about its internals. You bring a byte stream; the harness handles the
wake/sleep lifecycle, pruning, and sampling.

What it deliberately does **not** do: tokenization or text parsing. ALM
operates on raw bytes. Turning your source data into a byte stream —
reading files, stripping markup, deciding how to chunk a corpus — is your
job. The harness only needs `bytes`/`bytearray`/`str`/`np.ndarray` in, and
gives you `str`/`bytes` back out.

## Quick start

```python
from manifoldflow.dendritic_basin.alm_language import ALMLanguageConfig, ALMLanguageModel

config = ALMLanguageConfig(native=False)   # pure Python, no build step needed
model = ALMLanguageModel(config)

model.train(open("corpus.txt", "rb").read())

print(model.generate("Once upon a time"))
```

That's the whole contract: construct a config, build a model, call
`train()` with bytes, call `generate()` with a cue.

## Choosing a backend: `native=True` vs `native=False`

- `native=False` (pure Python, `SequenceCortexOpponentLearner`): always
  available, no build step. Slower per-byte. Use this for quick
  experiments, anything under a few MB, or any machine without the
  native library built.
- `native=True` (`NativeGrowableCortex`, ctypes-bound C++): requires
  `alm_native` to be built first (see `cpp/README.md`) — check with
  `manifoldflow.dendritic_basin.native_cortex.native_available()`.
  Bit-for-bit matches the Python backend's probabilities (see
  `tests/test_native_cortex_matches_python.py`); use this for anything
  you actually care about the wall-clock time of.

Trying to build an `ALMLanguageModel` with `native=True` when the shared
library isn't built raises `RuntimeError` with a pointer to
`cpp/README.md`, rather than failing obscurely later.

## Config reference (`ALMLanguageConfig`)

All fields are keyword-only-by-convention dataclass fields with sane
defaults; you only need to override what you're actually changing.

**Core learner shape** — `prior`, `confirmations`, `context_confirmations`,
`surprise_bits`, `basal_strength`, `max_context_bytes`,
`sequence_confirmations`, `parent_strength`, `max_sequence_nursery_paths`.
Same meaning and same defaults as
`SequenceCortexOpponentLearner`/`NativeGrowableCortex` — see
`docs/ALM_V1_Architecture.md` if you need to understand what each one
actually does; the defaults are reasonable starting points for byte-level
English text.

**Mechanism toggles (opt-in, off by default)**:
- `competition_exponent`, `adaptive_competition`, `trust_gamma`,
  `trust_learning_rate`, `attribution_gate_threshold` — consequence-
  sensitive contextual competition.
- `connectivity_pruning` — **recommended on** for any real run. Validated
  to beat plain count-based pruning at matched memory cost on both
  TinyStories and Wikipedia, at multiple data scales, with the margin
  growing (not shrinking) with more data. Zero training-time or memory
  cost — it only changes which rows a sleep-time byte-budget prune evicts
  first. See `tests/test_dendritic_basin_connectivity_pruning.py` and
  `tests/test_native_connectivity_pruning.py`.

**Orchestration**:
- `max_body_bytes` — the hard ceiling `consolidate_to_byte_budget` prunes
  down to at every sleep. Note there is a data- and config-dependent
  *floor* below this (fixed anatomy: unigram/transition/residual/context
  tables plus the nursery) — setting `max_body_bytes` below that floor
  raises `ValueError: fixed cortical anatomy exceeds the requested byte
  budget` the first time a sleep happens. If you hit that, raise the
  budget; there's no reliable way to guess the floor in advance other
  than trying it.
- `sleep_bytes` — how much of the stream `train()` processes before each
  sleep/prune cycle. One `train()` call may trigger several sleeps if
  `len(data) > sleep_bytes`.
- `columns` — see multi-column section below.
- `chunk_bytes` — currently unused by `train()` itself (single call to
  `observe_chunk_probabilities` per sleep round); kept in the config for
  parity with the production harness and for future chunked-progress
  reporting.
- `native` — backend choice, see above.

Constructing an invalid config (e.g. `columns=2` with `native=False`,
`columns=0`, or `sleep_bytes` smaller than `columns`) raises `ValueError`
immediately, not at first use.

## Single-column vs multi-column training

**Single column (`columns=1`, the default)** — the model trains directly
on the stream in-process. No subprocess or pickling overhead. This is the
right default for interactive use and anything that fits comfortably in
one process's training time budget.

**Multi-column (`columns>1`, requires `native=True`)** — mirrors the real
production harness's wake/sleep/consolidate cycle: each `sleep_bytes`-
sized round is split into `columns` contiguous slices, trained in
parallel worker processes (each starting from a pickled clone of the
current model), then merged back via `consolidate_delta` (ancestry-safe —
only what each column learned past the shared ancestor gets merged, so
nothing is double-counted) before the round's sleep/prune runs. Use this
when you have real CPU parallelism to spend and a stream large enough
that per-round subprocess spin-up cost is negligible next to training
time.

Multi-column mode uses `ProcessPoolExecutor`, so on Windows your calling
script needs the standard `if __name__ == "__main__":` guard around the
code that constructs and trains the model — the worker function itself
is already a module-level function (not a closure), so it's picklable,
but the *importing* of your script by each spawned worker still needs the
guard to avoid infinite re-spawning.

## Generation

```python
model.generate(
    "Once upon a time",
    max_new_bytes=200,
    temperature=0.8,
    top_k=32,
    top_p=1.0,
    seed=0,
    as_text=True,   # False to get raw bytes back instead
)
```

- `temperature=0` (or `top_k=1`) does greedy argmax decoding.
- Sampling is otherwise: temperature-scaled logits → optional top-k mask →
  optional top-p (nucleus) mask → categorical draw from a seeded
  `np.random.default_rng(seed)`. Same `seed` + same model state always
  gives the same output (see `test_generate_is_deterministic_given_a_seed`).
- `as_text=True` (default) decodes the full prompt+generation as UTF-8
  with `errors="replace"` — a byte-level organism can and will emit
  invalid UTF-8 sequences, especially early in training or on binary-ish
  data, so don't assume every generated string is clean.
- Generation does not learn — it's `predict_next_distribution()` +
  `advance_observation(byte_id, learn=False)` in a loop, then episode
  state is left as advanced-but-not-updated. Call `train()` again whenever
  you want the model itself to keep developing.

## Evaluating without training

```python
bpb = model.evaluate_bits_per_byte(held_out_bytes)
```

Resets episode state, scores the stream with `learn=False`, and returns
mean bits/byte over all finite predictions. Use this for a genuinely
held-out validation set — it does not touch the model's anatomy at all.

## Saving and loading

```python
model.save("checkpoint.pkl")
model = ALMLanguageModel.load("checkpoint.pkl")
```

Works for both backends. `NativeGrowableCortex` has its own
`__reduce__` (backed by the native binary `to_bytes`/`from_bytes`
checkpoint format, not raw ctypes pointers), so pickling a native model is
safe and portable across the process boundary (this is also what makes
multi-column training's worker-process handoff work). Native checkpoints
are same-machine/same-build only, per `cpp/README.md` — don't expect to
load one built on a different platform or a different `alm_native` build.

`save()` writes to a `.tmp` file and atomically renames over the target,
so a crash mid-save can't corrupt an existing checkpoint.

## What this harness intentionally leaves out

- **No tokenizer, no chunking-by-sentence/paragraph, no dataset loader.**
  Read your own files, decide your own preprocessing, hand over bytes.
- **No held-out split logic.** `evaluate_bits_per_byte` scores whatever
  you pass it — making sure that data is actually disjoint from what you
  trained on is on you.
- **No hyperparameter search / early stopping.** This is a thin
  configure-train-generate core, not an experiment orchestration
  framework — for that, see the phase126 production harness
  (`dendritic_basin_phase126_three_epoch_generation.py`), which this
  module does not replace.

## Relationship to the phase126 production harness

They are independent. `dendritic_basin_phase126_three_epoch_generation.py`
is the file-offset-based, multi-epoch, checkpoint-resuming production
script used for long unattended runs (e.g. the 16-hour TinyStories
connectivity-pruning run). `alm_language.py` is a smaller, generic,
in-memory-bytes-in/text-out core meant for ad hoc use and for embedding
ALM training/generation into other code. Neither imports the other.
