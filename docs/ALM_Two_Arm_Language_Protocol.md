# ALM Two-Arm Language Protocol

## Purpose

TinyStories remains the controlled developmental environment. SmolTalk-everyday
adds ordinary multi-turn questions, answers, topic persistence, and dialogue
roles without changing ALM's raw-byte interface or architecture. Every promoted
ALM configuration now receives two independent from-scratch runs:

1. **Control:** TinyStories only.
2. **Diversity diagnostic:** complete TinyStories and SmolTalk-everyday episodes
   alternated in one continuous raw-byte stream.

This is a dataset intervention, not an architectural feature. No chat role,
speaker, tokenizer, semantic label, or story boundary is exposed to the learner
beyond the bytes already present in each source.

## Matched conditions

Both arms must use the same:

- total training-byte budget;
- ALM byte-class/body cap;
- architecture and hyperparameters;
- seed, territory size, worker count, and consolidation policy;
- TinyStories validation slice;
- held-out SmolTalk validation split;
- generation prompts and decoding settings.

The mixed arm alternates whole episodes rather than arbitrary byte chunks. This
keeps local source structure intact while yielding an approximately 50/50 byte
mix. Its physical stream is generated before training so the existing parallel
territory runner sees deterministic seekable bytes.

SmolTalk rows are split by a stable SHA-256 content hash. The validation rows
never enter the mixed training stream. JSON syntax and source metadata are
discarded; only each record's `text` field is encoded as raw UTF-8.

## Dataset-size warning

The local SmolTalk-everyday export contains 2,260 conversations and 1.89MB of
JSONL. A 1MB mixed gate consumes only one partial SmolTalk epoch. A full
1.924GB 50/50 mixed run would necessarily revisit this subset hundreds of
times. Every stream manifest therefore reports `smoltalk_epochs_started`.

Results after recycling begins must be described as repeated-experience or
retention experiments, not evidence that additional data diversity caused an
emergent capability. Larger diverse conversational corpora can later replace
or supplement this arm without changing the two-arm protocol.

## Required measurements

For both organisms report:

- TinyStories held-out BPB;
- SmolTalk held-out BPB;
- embodied and awake memory bytes;
- throughput and number of source epochs;
- matched TinyStories and conversational free generations;
- entity/role persistence, boundary completion, repetition, and diversity.

The TinyStories-656K transformer remains a qualitative reference, not a claimed
victory. A superiority claim requires identical prompts and decoding followed
by blinded comparison. ALM is currently comparable in local fluency but has not
yet demonstrated better long-range story coherence.

## Stream construction

```powershell
.\.venv\Scripts\python.exe -m `
  manifoldflow.experiments.dendritic_basin_build_language_mix `
  --total-bytes 1000000 `
  --seed 73 `
  --output runs\alm_language_mix_1m.bin `
  --smoltalk-valid-output runs\smoltalk_everyday_valid_seed73.txt `
  --manifest runs\alm_language_mix_1m.json
```

The control runner reads the ordinary TinyStories training path. The diagnostic
runner reads the generated mixed path. Both use TinyStories as `--valid-path`
and the generated SmolTalk holdout as `--secondary-valid-path`, with
`--secondary-eval-label smoltalk`.

## Promotion rule

Do not replace the TinyStories control with the mixed arm. Promote a mechanism
only after interpreting both:

- improvement in both arms suggests a general learning primitive;
- improvement only in TinyStories suggests distribution-specific structure;
- improvement only in the mixed arm suggests experience-dependent emergence;
- regression on TinyStories with a SmolTalk gain exposes interference or a
  capacity-allocation tradeoff.

Qualitative capability and BPB remain separate required axes.
