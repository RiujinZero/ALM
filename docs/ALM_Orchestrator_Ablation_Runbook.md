# ALM Orchestrator Ablation Runbook

## Status

`src/manifoldflow/experiments/dendritic_basin_orchestrator_v1.py` is
implemented and validated: confidence-weighted fusion (the mechanism that
actually works, after two other combine rules were tried and rejected --
see `docs/ALM_V1_Architecture.md`, orchestrator "v1 empirical results").
This doc is about *running* it, not the design -- read the architecture
doc first if you need the "why."

## Why WSL

Everything here uses the native (`ctypes`-bound C++) backends
(`NativeGrowableCortex`, `NativeGrowingVision`) for real throughput -- the
pure-Python fallbacks are far too slow for anything but a smoke test. The
native shared libraries (`libalm_native.so`, `libalm_vision.so`) are built
and run from a WSL-side copy of the repo, and datasets live on WSL's
native filesystem, not the mounted Windows `D:` drive, which is slow for
writes and was the original reason this whole workflow moved to WSL.

**Two separate checkouts exist on purpose**: the Windows repo (this one,
under source control, where you edit) and a WSL-native copy at
`~/sonic-manifold-flow` (where things actually run and get built). They
are kept in sync by explicit `cp`, not a shared filesystem mount -- do not
assume editing one edits the other.

## One-time environment check

```bash
wsl bash -c "ls ~/gtr_venv/bin/python3 && ls ~/sonic-manifold-flow/cpp/build/bin/*.so"
```

If `~/gtr_venv` doesn't exist, dependencies (`numpy`, etc.) aren't
installed against the system Python and native calls will fail with
`ModuleNotFoundError`. Use `~/gtr_venv/bin/python3` for every run below,
not `python3` or the Windows-side `.venv`.

## Syncing a code change from Windows to WSL

After editing anything under `cpp/` or `src/` on the Windows side:

```bash
# Python-only change -- just copy the file(s):
wsl bash -c "cp '/mnt/c/Users/<you>/Documents/projects/sonic-manifold-flow/src/manifoldflow/experiments/dendritic_basin_orchestrator_v1.py' ~/sonic-manifold-flow/src/manifoldflow/experiments/dendritic_basin_orchestrator_v1.py"
```

If you touched a `.hpp`/`.cpp` file, you must also rebuild the shared
library that file belongs to -- editing the header alone changes nothing
at runtime until this runs:

```bash
# Byte cortex (cortex.hpp, c_api.h, c_api.cpp, native_cortex.py):
wsl bash -c "cp '/mnt/c/Users/user/Documents/projects/sonic-manifold-flow/cpp/include/alm/cortex.hpp' ~/sonic-manifold-flow/cpp/include/alm/cortex.hpp && \
  cp '/mnt/c/Users/user/Documents/projects/sonic-manifold-flow/cpp/include/alm/c_api.h' ~/sonic-manifold-flow/cpp/include/alm/c_api.h && \
  cp '/mnt/c/Users/user/Documents/projects/sonic-manifold-flow/cpp/src/c_api.cpp' ~/sonic-manifold-flow/cpp/src/c_api.cpp && \
  cp '/mnt/c/Users/user/Documents/projects/sonic-manifold-flow/src/manifoldflow/dendritic_basin/native_cortex.py' ~/sonic-manifold-flow/src/manifoldflow/dendritic_basin/native_cortex.py && \
  cd ~/sonic-manifold-flow && \
  g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra -msse2 -fPIC \
    -Icpp/include -shared cpp/src/c_api.cpp -o cpp/build/bin/libalm_native.so.new && \
  mv cpp/build/bin/libalm_native.so.new cpp/build/bin/libalm_native.so"

# Vision organism (vision.hpp, vision_c_api.cpp, ...):
wsl bash -c "cp '/mnt/c/Users/<you>/.../cpp/include/alm/vision.hpp' ~/sonic-manifold-flow/cpp/include/alm/vision.hpp && \
  cd ~/sonic-manifold-flow/cpp && \
  g++ -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra -std=c++17 -fPIC -shared -Iinclude -o build/bin/libalm_vision.so src/vision_c_api.cpp -lpthread"
```

`-Wall -Wextra` with no warnings printed is a clean build; anything else,
read the warning before trusting the run. A running process that already
loaded the old `.so` is unaffected by a rebuild (Linux keeps the old
inode mapped) -- only new processes pick up the change.

## Running an orchestrator ablation

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_orchestrator_v1 \
  --corpus /path/to/ALM-benchmarks/mixed-conversational/train.txt \
  --report runs/orchestrator/my_ablation.json \
  --start-offset 4000273 \
  --total-bytes 100000 \
  --checkpoint-every 5000"
```

Throughput degrades over the run as both vision organs' assembly counts
grow toward their caps (expect roughly 3,000 bytes/sec at the start,
dropping to 100-200 bytes/sec by 50-100K bytes) -- **budget wall-clock
accordingly**: 30K bytes is a few minutes, 100K bytes is 15-25 minutes.
Always run in the background (`&` or a separate terminal) for anything
over ~30K bytes; do not block a foreground shell on it.

`--report` is a JSON file with the full result (per-organ and
orchestrator bpb/accuracy, disagreement-conditioned numbers, and the
soft-bias sweep) -- written once at the end, and also printed to stdout.
There is no incremental/partial-report write; if a run is killed early,
only the stdout checkpoint lines survive, not a report file.

### Config knobs worth ablating

- `--bias-weights`: comma-separated sweep, e.g. `"0,0.3,0.6,0.7,0.8,1.0"`.
  The 100K offset-zero run found a broad, shallow organ_A plateau centered
  around `w ~= 0.5-0.65`; a dialogue-aligned 30K run was nearly flat from
  0.65 through 0.85 (numeric minimum at 0.8). Do not optimize around a
  single smoke-test minimum. Organ_B improved monotonically through `w=1.0`
  in both cases.
- `--organ-a-context-bytes` / `--orch-dendritic-context-bytes`: both are
  `cortex.hpp`'s `max_context_bytes`, hard-capped to `[2, 8]` by the
  packed-history representation (a `uint64_t`, 8 bits/byte). The
  orchestrator's dendritic half should stay meaningfully broader than
  organ_A's -- e.g. organ_A=4, orchestrator=8 (the max) is the validated
  starting point.
- `--organ-b-max-assemblies` / `--orch-max-assemblies`: raise alongside
  `--total-bytes` if you don't want either vision organ to cap out before
  the run ends -- but note Task A (see architecture doc) found more
  capacity alone gave rapidly diminishing returns past a point, so this
  probably isn't the highest-value knob to sweep.
- `--laplace-eps`: only affects the two `vision.hpp`-derived probability
  arrays (organ_B's votes, `orch_synaptic`'s votes) -- never apply
  smoothing to a `cortex.hpp` `predict_next_distribution()` output, it is
  already a calibrated distribution (see the module docstring's warning
  about the bug this caused once already).

### Corpus

`/path/to/ALM-benchmarks/mixed-conversational/train.txt` is block-concatenated,
not randomly interleaved. The exact training ranges produced by the current
builder are:

- Wikipedia: `[0, 4,000,273)`
- LMSYS dialogue: `[4,000,273, 8,000,287)`
- preference dialogue: `[8,000,287, 12,002,073)`
- Everyday dialogue: `[12,002,073, 16,002,557)`

Consequently the default `--start-offset 0` measures Wikipedia even though
the filename says mixed-conversational. Use `--start-offset 4000273` for the
validated LMSYS boundary, or another exact block boundary above. These are
byte offsets; do not round them. The corresponding validation boundaries are
0, 500,508, 1,001,700, 1,502,321, and 2,002,330.

## Running the shared-goal ablations

The flat `mixed-conversational/train.txt` stream cannot support these assays:
its `\n\n` separators do not distinguish turns from conversations. Build from
the raw JSONL instead. The response-target corpus used in the reports was:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 scripts/build_goal_trajectory_ablation.py \
  --lmsys /path/to/manifoldflow_chat_sources/lmsys_chat_1m.jsonl \
  --everyday /path/to/manifoldflow_chat_sources/smoltalk_everyday.jsonl \
  --output-dir runs/shared_goal/goal_trajectory_v1 \
  --scan-lmsys 50000 --train-bytes 960000 \
  --dev-pairs 64 --test-pairs 128 --seed 82972"
```

For the stricter target made from the turns *after* the withheld response:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 scripts/build_goal_trajectory_ablation.py \
  --lmsys /path/to/manifoldflow_chat_sources/lmsys_chat_1m.jsonl \
  --everyday /path/to/manifoldflow_chat_sources/smoltalk_everyday.jsonl \
  --output-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --scan-lmsys 50000 --train-bytes 3000000 \
  --dev-pairs 64 --test-pairs 128 --seed 82972 \
  --goal-target-kind successor"
```

For a prospectively frozen source window, `--start-lmsys-line` and
`--start-everyday-line` are zero-based inclusive. Each `--scan-*` value is the
maximum number of rows read *after* that start, so start 50,000 plus scan
50,000 means `[50,000, 100,000)`. `--everyday` is optional; omitting it makes
the build LMSYS-only. The fresh confirmation corpus was built from Windows
PowerShell at the repository root with:

```powershell
.venv/Scripts/python.exe scripts/build_goal_trajectory_ablation.py `
  --lmsys C:\path\to\manifoldflow_chat_sources\lmsys_chat_1m.jsonl `
  --start-lmsys-line 50000 --scan-lmsys 50000 `
  --output-dir runs/shared_goal/confirmatory_goal_trajectory_successor_lmsys_lines50000_99999_v1 `
  --train-bytes 3000000 --dev-pairs 64 --test-pairs 128 --seed 260914 `
  --min-prefix-bytes 512 --min-response-bytes 64 --score-bytes 128 `
  --min-bridge-bytes 512 --local-context-bytes 8 `
  --max-record-bytes 16384 --min-prior-exchanges 2 `
  --goal-target-kind successor
```

The schema-3 manifest records the source bounds and verifies that selected
lines lie inside them. This build has 536 train episodes, 64 dev pairs, and
128 test pairs, with zero exact selected episode-ID overlap with the earlier
corpus. That does not rule out paraphrase or latent template-family overlap.

The matched interlocutor-target control uses the first later USER turn:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 scripts/build_goal_trajectory_ablation.py \
  --lmsys /path/to/manifoldflow_chat_sources/lmsys_chat_1m.jsonl \
  --everyday /path/to/manifoldflow_chat_sources/smoltalk_everyday.jsonl \
  --output-dir runs/shared_goal/goal_trajectory_interlocutor_v1 \
  --scan-lmsys 50000 --train-bytes 3000000 \
  --dev-pairs 64 --test-pairs 128 --seed 82972 \
  --goal-target-kind interlocutor"
```

Use a new output directory, or add `--overwrite` only when replacing that
exact derived corpus is intentional. The manifests record byte ranges and
SHA-256 checksums. The response build has 657 training episodes, the successor
build has 617, and the interlocutor build has 620; each has 64 dev and 128 test
pairs. Every pair uses two source episodes plus a disjoint third-episode
interruption of at least 512 bytes, then the same resume cue. The complete
shared suffix is at least 700 bytes, and A/A + B/B is compared with A/B + B/A
so candidate-intrinsic fluency cancels.

Do not describe the concrete split as user- or template-clustered. Neither
source exposed a declared conversation/thread/session/id for the eligible
rows, so every split key in these builds is an exact normalized episode SHA.
Long turns are also filtered across splits by exact normalized hashes, but
there is no fuzzy MinHash and no latent user/template-family clustering;
paraphrases and near-duplicates may cross splits. Natural pairs therefore
measure long-context response compatibility, not goal identity by themselves.

### Frozen-native evaluator invariant

`reset_episode()` is not a complete counterfactual reset. It clears byte
history and the episodic store, while normal `learn=False` inference still
updates the burst EMA and adaptive context trust. Reusing one native object for
candidate branches can therefore leave order-dependent differences around
1e-8 even after a long common suffix.

All frozen counterfactual evaluators must:

1. disable the episodic layer;
2. call `set_debug_freeze_inference_adaptation(True)` after every checkpoint
   load (the switch is runtime-only and is not serialized);
3. reset and re-prime before every branch; and
4. fail, rather than merely warn, if the required base traces differ.

`dendritic_basin_shared_goal_crossover_ablation` performs the first three;
pass `--require-exact-native` for the fourth. The 128-pair generation report
also passed its independent beta-zero gate: corrected/native probabilities
and sampled bytes were exact for every pair. This debug switch is for assays;
normal deployed inference remains adaptive.

### Crossover commands and current result

The response-target matched-head result is reproduced with:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_v1 \
  --report runs/shared_goal/goal_crossover_head_p32_allseeds_did_repro.json \
  --prototypes 32 --seeds 3 11 29 --resamples 4000"
```

Its local byte-head causal margins were +0.3744/+0.3203/+0.3457 bits, but the
native overlay was only +0.0011/+0.0918/+0.1165 and the first seed's interval
crossed zero. More importantly, the hard-code representation failed direct
same-versus-crossed assignment despite better individual code NLL. Treat the
positive local-head score as a compatibility effect, not validated causal goal
inference. Reports:
`runs/shared_goal/goal_crossover_head_p32_allseeds_did.json`,
`runs/shared_goal/goal_crossover_native500_p32_seeds3_11.json`, and
`runs/shared_goal/goal_crossover_native500_p32_seed29.json`.

For completeness, the legacy hard-code successor-native failure is reproduced
with exactness promoted to a failure gate:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/goal_crossover_successor_native500_p16_exact.json \
  --prototypes 16 --seeds 3 11 29 --resamples 4000 \
  --require-exact-native"
```

The completed run without the fail-fast flag already had exact base traces,
but failed scientifically: oracle effects had the wrong sign for all three
seeds (-14.4024/-2.5954/-11.4709 bits), and only seed 11's causal interval was
above zero. Its report is
`runs/shared_goal/goal_crossover_successor_native500_p16_allseeds.json`.

The frozen-generation follow-up can be reproduced as follows:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/goal_generation_successor_native500_p16_seed3_128pairs_repro.json \
  --split test --max-pairs 128 --samples-per-pair 1 --max-new-bytes 128 \
  --prototypes 16 --mechanism-seed 3 --sampling-seed 20260914 \
  --beta 2.0 --temperature 0.8 --top-k 32 --top-p 0.9"
```

The eight-pair pilot was positive but did not survive scale. At 128 pairs the
causal independent-reference margin was +0.00109 with 0.484 wins, below the
shuffled episode-code control (+0.00182, 0.531 wins); its median was zero.
Samples remained locally word-like but semantically incoherent. This is a
generation null, not evidence of goal-directed output.

### Strict continuous successor representation

Do not wire the hard prototype code into the production orchestrator. First
reproduce the continuous representation audit. This command uses only target
turns after the complete scored assistant response, selects ridge alpha and
calibration scale on dev, and freezes them for test:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_future_continuous_pair_audit \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --report runs/shared_goal/goal_trajectory_successor_continuous_pair_audit.json \
  --hash-salts 3 11 29 --ridge-alphas 0.1 1 10 100 \
  --evidence-dimensions 32 --state-decays 0.5 0.85 \
  --successor-base-dimensions 16 32 \
  --successor-gammas 0.5 0.85 0.97 \
  --calibration-scales 0 0.25 0.5 1 2 4 8 16 32 64 \
  --bootstrap-resamples 2000"
```

The selected 32D-per-successor-summary representation averaged 0.8273 correct
pair-assignment probability, 0.8724 wins, and 0.4063-bit cross entropy on
test. Pending-user-only evidence averaged 0.6588/0.7891/0.6764; Sattolo-
shuffled training targets stayed at 0.4972/0.4635/1.0106. The 16D and whole-
payload variants agreed. Hash salts reuse the same 128 test pairs: they are
adapter sensitivity checks, not independent samples.

### Signed byte adapter: preserved representation, failed action

The signed continuous byte-head integration is a required negative result,
not the selected adapter:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_continuous_crossover_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/goal_continuous_crossover_successor_native500_d32.json \
  --feature-dimensions 32 --state-decays 0.5 0.85 \
  --successor-gammas 0.5 0.85 0.97 --ridge-alpha 10 \
  --control-seed 3 --resamples 4000 --require-exact-native"
```

The representation still discriminated the strict successor target on test
(full-history cosine pair margin 0.3896, pending-only 0.1904, shuffled 0.0054),
but the native action effect did not. Causal was +0.2310 bits with 95% CI
[-0.1194, +0.5582] and 0.5625 wins; oracle was -0.3267 bits with an interval
crossing zero, and swapped oracle did not reverse. All exact evaluator
invariants passed. Treat this as a signed byte-adapter failure.

### Whole-candidate successor reranker: selected adapter

The sequence-level mechanism fits a train-only continuous future state,
predicts it from full causal history, predicts each candidate's successor from
the pending USER event plus the complete candidate, and adds their cosine
alignment to an independently fitted goal-off plausibility score. Run the
semantic-base control grid with the additive action map:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --report runs/shared_goal/goal_trajectory_reranker_successor_pca32_additive_multiseed.json \
  --feature-dimensions 96 --latent-dimensions 32 \
  --state-decays 0.5 0.85 0.97 --successor-gammas 0.5 0.8 0.95 \
  --goal-ridge-alpha 10 --action-ridge-alpha 10 \
  --plausibility-ridge-alpha 10 \
  --gain-values 0 0.25 0.5 1 2 4 8 16 32 \
  --seeds 3 11 29 --resamples 4000 --no-interaction"
```

Gain 8 was selected on semantic dev. Test matched-candidate accuracy rose from
0.6211 goal-off to 0.7734 causal. The paired accuracy gain is +0.15234, with
95% pair-bootstrap intervals approximately [+0.086, +0.215] across the three
control/resampling seeds. Full-history representation discrimination
was 0.6519 versus 0.4194 pending-only, a +0.2325 paired increment (95% CI lower
bound +0.1761). Full-history action alignment was +0.2689 (95% CI
[+0.2129, +0.3271], 0.8125 positive pairs), and exceeded pending-only by
+0.1456 (95% CI [+0.0980, +0.1960]). Oracle and every paired shuffle/swap
contrast passed; candidate-blind action was exactly zero.

Run the disjoint-supervised-fit falsifier before treating that result as an
independent goal/action mechanism:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --report runs/shared_goal/goal_trajectory_reranker_successor_pca32_additive_disjoint_multiseed.json \
  --feature-dimensions 96 --latent-dimensions 32 \
  --state-decays 0.5 0.85 0.97 --successor-gammas 0.5 0.8 0.95 \
  --goal-ridge-alpha 10 --action-ridge-alpha 10 \
  --plausibility-ridge-alpha 10 \
  --gain-values 0 0.25 0.5 1 2 4 8 16 32 \
  --seeds 3 11 29 --resamples 4000 --no-interaction \
  --disjoint-supervised-fit"
```

The report fits one common PCA encoder to training targets only, then assigns
the 617 training episodes by a stable source-stratified SHA-256 rule to folds
A=309 and B=308. Goal maps fit on A are evaluated with action, candidate-
blind, and plausibility maps fit on B, then the roles reverse; the supervised
maps have zero episode overlap and are never pooled. Both directions pass. In
A-to-B, test accuracy is 0.5625 goal-off versus 0.671875 causal, a +0.109375
paired delta (95% CI [+0.046875, +0.171875]); in B-to-A it is 0.613281 versus
0.710938, a +0.097656 delta (95% CI [+0.031250, +0.164063]). Full-history
action exceeds pending-only by +0.103109 (95% CI [+0.056245, +0.150234]) and
+0.103251 (95% CI [+0.063899, +0.144647]), respectively. All 15 gates pass
for every shuffle seed. The two directions reuse the same dev/test pairs and
are reciprocal falsification checks, not independent samples; never pool them.

For the frozen 500MB base, use a separate, wider dev gain grid because native
mean-log2-per-byte and semantic cosine scores are not commensurate:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/goal_trajectory_reranker_successor_native500_additive_seed3_gainwide.json \
  --feature-dimensions 96 --latent-dimensions 32 \
  --state-decays 0.5 0.85 0.97 --successor-gammas 0.5 0.8 0.95 \
  --goal-ridge-alpha 10 --action-ridge-alpha 10 \
  --plausibility-ridge-alpha 10 \
  --gain-values 0 1 2 4 8 16 32 64 128 256 \
  --seeds 3 --resamples 4000 --no-interaction"
```

The native dev curve plateaus at 0.78125 accuracy for gains 32 and 64; the
declared smaller-intervention tie rule selects 32. On test, the identical-
suffix native base is exactly 0.5000, while the causal successor reranker is
0.7617. Its unscaled action statistics are the same +0.2689 margin and 0.8125
pair wins above. Do not reuse gain 8 from the semantic score: doing so would
mix incompatible score scales.

The bounded normalized-Hadamard/cosine interaction is a negative architecture
ablation. Its otherwise matched report,
`runs/shared_goal/goal_trajectory_reranker_successor_pca32_multiseed.json`,
scored 0.7500 on dev and 0.7422 on test versus additive 0.7813/0.7734. Keep
`--no-interaction` for the selected mechanism.

The interlocutor target is also rejected despite a superficially positive
causal score. Reproduce its action gate with:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_interlocutor_v1 \
  --report runs/shared_goal/goal_trajectory_reranker_interlocutor_d32.json \
  --feature-dimensions 32 --latent-dimensions 32 \
  --state-decays 0.5 0.85 --successor-gammas 0.5 0.85 0.97 \
  --goal-ridge-alpha 10 --action-ridge-alpha 10 \
  --plausibility-ridge-alpha 10 \
  --gain-values 0 0.25 0.5 1 2 \
  --seeds 3 11 29 --resamples 4000"
```

Its causal action margin is +0.1295 with 0.7188 wins, but the oracle action
margin is only +0.0357 and its CI crosses zero (lower bound -0.0098). That
causal-greater-than-oracle pattern is compatible with shared input/regression
bias, not validated next-interlocutor compatibility.

For reranker reports, seeds 3/11/29 do **not** refit independent real models.
The PCA encoder and non-control ridge maps are deterministic and identical;
the seed changes Sattolo mappings, the evaluation episode shuffle, and
bootstrap/sign-flip draws. Report them as control sensitivity runs.

### Frozen generated-candidate falsifier

The production-facing candidate-bank test has been completed:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_generation_reranker_ablation \
  --dataset-dir runs/shared_goal/goal_trajectory_successor_v1 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/goal_generation_reranker_successor_native500_d96_n16_64pairs.json \
  --max-dev-pairs 32 --max-test-pairs 64 \
  --bank-size 16 --max-new-bytes 96 \
  --temperature 0.8 --top-k 32 --top-p 0.9 \
  --feature-dimensions 96 --latent-dimensions 32 \
  --lexical-dimensions 2048 \
  --state-decays 0.5 0.85 0.97 \
  --successor-gammas 0.5 0.8 0.95 \
  --goal-ridge-alpha 10 --action-ridge-alpha 10 \
  --plausibility-ridge-alpha 10 \
  --gain-values 0 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512 \
  --mechanism-seed 3 --sampling-seed 20260914 \
  --min-printable-fraction 0.8 --useful-reference-cosine 0.05 \
  --resamples 4000 --no-interaction"
```

This freezes 16 independently sampled 96-byte native candidates per side,
selects gain 2 on 32 generated dev banks, and evaluates 64 held-out test
pairs. The independent byte-ngram reference effect is +0.00972 (95% paired CI
[-0.01247, +0.03848]), a null. The action-shared future effect is +0.08805
(95% CI [+0.02719, +0.15021]), but that endpoint shares the train-fitted action
representation with the selector and is mechanistic rather than an independent
language-quality measure. Manual inspection still finds semantically
incoherent samples; the report's diversity, printability, and reference-proxy
heuristics must not be described as human coherence.

The same report includes an explicitly offline diagnostic that adds both
held-out human responses to every bank. It selects its own gain 256 on the
injected dev pools and recovers an available matched response at 0.3203 above
goal-off (95% CI [0.2422, 0.3984]), versus 0.1563 pending-only and
0.0625--0.0859 for the four shuffled controls. At the generated-bank gain 2,
injected recovery is zero. Because held-out answers are unavailable online,
gain 256 is a selector-competence diagnostic, not a deployable setting.

### Claim limits for the selected mechanism

- The strongest justified claim is an **independent, persistent-history-
  sensitive, whole-candidate future-trajectory compatibility check**, not
  goal-directedness.
- The strict successor is short-horizon in this test set: 86.3% of the 256
  episode sides contain exactly two later turns, and 97.7% contain at most
  four.
- The interruption bridge is deliberately excluded from goal evidence. This
  tests retention of pre-bridge state, not a learned slow-update policy.
- The human candidate assay scores only each response's first 128 bytes. Those
  supplied candidates are easier than generated samples and do not measure
  open-ended generation.
- Causal alignment can exceed oracle alignment even under disjoint supervised
  fitting, so its magnitude is not calibrated goal truth.
- Exploratory iterations inspected the original fixed test set, so it cannot
  serve as an untouched confirmation. Use the prospective result below.

### Fresh prospective confirmation

The locked primary architecture was run without post-test tuning on the new
LMSYS-only source window:

```powershell
.venv/Scripts/python.exe -m manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation `
  --dataset-dir runs/shared_goal/confirmatory_goal_trajectory_successor_lmsys_lines50000_99999_v1 `
  --report runs/shared_goal/confirmatory_goal_trajectory_reranker_successor_lmsys_lines50000_99999_pca32_additive_multiseed.json `
  --feature-dimensions 96 --latent-dimensions 32 `
  --state-decays 0.5 0.85 0.97 --successor-gammas 0.5 0.8 0.95 `
  --goal-ridge-alpha 10 --action-ridge-alpha 10 --plausibility-ridge-alpha 10 `
  --gain-values 0 0.25 0.5 1 2 4 8 16 32 `
  --seeds 3 11 29 --resamples 4000 --no-interaction
```

All 15 gates pass at dev-selected gain 8. Test accuracy rises from 0.66797
goal-off to 0.78516 causal, a +0.11719 paired delta (seed-3 95% CI
[+0.04678, +0.19141]). Causal action alignment is +0.27138 (95% CI
[+0.21908, +0.32217]); its full-history increment over pending-only is
+0.13414 (95% CI [+0.08694, +0.18289]). The representation's full-history
increment is +0.18629 with 95% CI lower bound +0.13082. Seeds still vary only
controls and resampling, not the deterministic real fit.

Run the predeclared disjoint fit on the same frozen corpus with:

```powershell
.venv/Scripts/python.exe -m manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation `
  --dataset-dir runs/shared_goal/confirmatory_goal_trajectory_successor_lmsys_lines50000_99999_v1 `
  --report runs/shared_goal/confirmatory_goal_trajectory_reranker_successor_lmsys_lines50000_99999_pca32_additive_disjoint.json `
  --feature-dimensions 96 --latent-dimensions 32 `
  --state-decays 0.5 0.85 0.97 --successor-gammas 0.5 0.8 0.95 `
  --goal-ridge-alpha 10 --action-ridge-alpha 10 --plausibility-ridge-alpha 10 `
  --gain-values 0 0.25 0.5 1 2 4 8 16 32 `
  --seeds 3 11 29 --resamples 4000 --no-interaction `
  --disjoint-supervised-fit
```

This is a partial discrete-utility confirmation. The 268/268 supervised folds
have zero episode overlap. Goal-A/action-B selects gain 16 and passes 15/15
gates: 0.62891 goal-off rises to 0.70313, a +0.07422 delta (95% CI
[+0.00391, +0.14453]). Goal-B/action-A selects gain 4 and passes 14/15:
0.64453 rises to 0.70313, but its +0.05859 delta has CI [-0.00781, +0.12891].
Both directions retain positive causal action, full-history-over-pending,
oracle, representation, shuffle, and invariant gates; only the second
direction's discrete candidate-accuracy CI fails. Do not tune after this
result, count reciprocal directions as independent, or pool them.

The prospective primary result confirms the representation/action mechanism,
while the disjoint result only partially confirms discrete candidate utility.
The frozen generation null is unchanged: current native candidate coverage
and output quality still block thoughtful generation because a reranker cannot
select a continuation the generator never proposes.

### Frozen two-stage rollout-search falsifier

The next production-facing ablation tested whether the validated successor
score can improve proposal search rather than only rerank a completed bank. It
uses eight 96-byte frozen-native rollouts, commits 48 bytes from the selected
path, and resamples eight 48-byte suffixes. The compute-matched flat baseline
uses twelve complete 96-byte candidates: both single-condition algorithms
therefore consume a conceptual 1,152 native predicted bytes. For strict
condition-order independence, the harness actually precomputes every unique
root-prefix continuation bank once and shares those label-free proposals across
all controls; do not describe the total stored tree as compute-equal to the
single flat bank.

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_shared_goal_rollout_search_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_trajectory_successor_lmsys_lines50000_99999_v1 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/confirmatory_goal_rollout_search_successor_native500_k8_h96_c48_64pairs_audit.json \
  --max-dev-pairs 32 --max-test-pairs 64 \
  --branch-factor 8 --horizon-bytes 96 --commit-bytes 48 \
  --temperature 0.8 --top-k 32 --top-p 0.9 \
  --feature-dimensions 96 --latent-dimensions 32 \
  --lexical-dimensions 2048 \
  --state-decays 0.5 0.85 0.97 \
  --successor-gammas 0.5 0.8 0.95 \
  --goal-ridge-alpha 10 --action-ridge-alpha 10 \
  --plausibility-ridge-alpha 10 \
  --gain-values 0 0.5 1 2 4 8 16 32 \
  --mechanism-seed 3 --sampling-seed 20260914 \
  --resamples 4000 --include-pair-audit"
```

The protocol is valid: frozen A/B traces and scores are bit-identical, output
lengths are fixed, goal-off selects identical bytes across pair sides,
candidate-blind equals goal-off, and the conceptual search/flat budgets match.
Development selected gain 32, the edge of the declared grid. On 64 held-out
pairs, causal receding search beats receding goal-off by +0.02256 on the
independent byte-ngram pairing metric (95% pair-bootstrap CI +0.00101 to
+0.04428). Record this only as a weak routing effect.

It fails the decisive proposal/search gates. The causal advantage over the
compute-matched flat causal baseline is +0.01009 with CI -0.00853 to +0.02941;
the full-history advantage over pending-only is +0.00962 with CI -0.01781 to
+0.03993. The evaluation-only oracle future has an independent margin of just
+0.00071 and its interval crosses zero. Contrasts against shuffled predictor,
action-target, and candidate-action mappings also cross zero. Although causal
search raises its reached-pool reference ceiling from goal-off's 0.10754 to
0.12781, the flat bank reaches 0.18249. Manual inspection remains semantically
incoherent and topic-drifting.

Strict conclusion: this run does not establish a useful proposal-search
mechanism and does not pass the thoughtful-output gate. Do not promote it as
the generation solution, expand the gain grid after seeing the edge result, or
use the oracle/control-shared action metric to override the independent nulls.
The authoritative report is
`runs/shared_goal/confirmatory_goal_rollout_search_successor_native500_k8_h96_c48_64pairs_audit.json`.

### Receiver follow-ups and next decision

| Evidence | Disposition | Operational consequence |
| --- | --- | --- |
| `REPORT3` | Scaffolded decision-to-byte loop passes; supplied semantics/actions/reward and finite phrases | Retain as a component ceiling, not a language or inferred-goal result |
| `hindsight_outcome_control_synthetic_v2.json` | Exact-lag coupled path passes, but direct behavior cloning also passes and +/-1-event timing is chance | Require direct-imitation and timing controls in every consequence assay |
| `antecedent_state_production_native500_v3_dev.json` | Development selects beta 0; nonzero PoE does not improve free generation and degrades native BPB; test sealed | Retire this frozen overlay without opening test; this does not reject a trained/co-adapted antecedent mechanism |
| `context_coupled_writer_receiver_native500_v1_result.json` | No beta passes; free-generation contrasts versus native/additive-wide cross zero and sampled text remains off-topic; BPB is also worse; test sealed | Do not tune beta or run the protected test; the result rejects this frozen receiver configuration, not the underlying consequence mechanism |
| `THEOREMS.md` / `REPORT4` | Conditional-completion probability is a valid target; reported finite execution awaits its absent artifact bundle | Use only as a mathematical specification and supplied-semantic ceiling for now |
| `REPORT5` / `report5_self_learned_coherence_audit_v1.json` | A trained history-by-current-prefix correction generates 504/504/506 greedy and 456/455/448 sampled exact replies on the same 512 worlds; it learns demonstrated conditional likelihood, not completion `h` or an inferred goal | Promote it as the productive-interface/direct-imitation arm, with engineered-memory and correlated-world limits explicit; do not relabel it as the goal mechanism |
| `self_learned_history_route_native500_v1_dev.json` / audit | Eight-pass natural transfer route is active on every side but sampled free generation is worse than native at every trained checkpoint; pass-8 comparisons versus additive and shuffled do not jointly pass, request-motif content state is null, and output degenerates | Retire this frozen-base next-response density-ratio configuration without opening test; keep REPORT5 interaction as a control and change the target to independently checked completion/consequence |
| NCCA-v1 | Later-user compression is self-supervised but observational and not a utility signal | Writer actuation remains blocked; any upstream-only run requires a frozen preregistration amendment |

Decision path:

1. Obtain and audit the `REPORT4` protocol/results/proof bundle.
2. Freeze an independent full-continuation checker and the generation prompts,
   draws, budgets, and failure rules before fitting the next receiver. An exact
   evaluation-only completion-potential head remains a useful ceiling, but the
   primary endpoint is independently checked free generation.
3. Start every bounded convergence arm from the same 500 MB initialization.
   Include equal-update native continuation, a REPORT5-style direct-imitation
   path using causal history-by-current-prefix activity, an active-count-matched
   additive path, shuffled association, wrong/reset history, and the proposed
   completion/value mechanism. Keep evidence, generated commitments, and the
   independent checker separate.
4. Evaluate predeclared geometric checkpoints -- for example 0, 0.25, 0.5, 1,
   2, 4, and 8 passes -- rather than only checkpoint zero and the endpoint. At
   each checkpoint generate before consulting BPB, and record exact task
   completion, goal-swap sensitivity, commitment consistency, degeneration,
   diversity, route utilization, and off-support failures.
5. A completion/value mechanism advances only if its held-out free-generation
   learning curve beats both native continuation and direct imitation. If an
   exact ceiling cannot change independently checked bytes, redesign the
   receiver; if training curves remain flat across the locked budget, reject
   that co-adapted mechanism. A frozen-checkpoint null alone rejects only the
   tested interface because the base had no opportunity to organize around it.
6. Require timing-jitter, swapped-binding, disconnected-output,
   capacity-matched, and false-positive-support controls before interpreting a
   positive result as consequence or completion learning rather than
   conditional imitation.
7. Only after causal potential beats those controls should NCCA writer
   actuation resume. An earlier NCCA proxy-observability study is permissible
   only under an amended frozen protocol, with the claim limited to predicting
   later user bytes.

For all future mechanism gates, independently checked free-generation task
completion/coherence is primary. Teacher-forced BPB is a non-collapse and
plausibility diagnostic only; it must not veto a robust behavioral improvement
by itself or promote an arm whose free output remains incoherent.

Training depth is part of the hypothesis. Frozen-checkpoint routing answers
whether an already learned base exposes enough support through one particular
receiver; it is a useful interface ceiling, not a fair convergence test of a
new mechanism. The convergence experiment must let the representation,
consequence learner, and conditional byte pathway receive repeated matched
updates while keeping total data order, update count, initialization, memory
allocation, and evaluation draws equal across arms. At each checkpoint record
independent task completion, goal-swap sensitivity, within-response commitment
consistency, degeneration, diversity, and route utilization. BPB remains
secondary. Stop for a flat controlled learning curve across the preregistered
budget, not merely because checkpoint zero looks similar to the base.

## Resuming/managing the long-running text runs

The two production text runs (`500mb_widedend`, `2_5gb_widedend`) are
separate from orchestrator ablations -- they run
`dendritic_basin_phase156_conversational_prototype.py` directly, not
through the orchestrator. Both are **paused** (`kill -TERM`'d, not
destroyed). Their latest complete checkpoints are now epoch 147 (500MB:
1.9956 training bpb, 2.1003 validation bpb, 500,000,000 body bytes) and
epoch 169 (2.5GB: 2.0916 training bpb, 2.0591 validation bpb,
2,089,713,055 body bytes). The much larger model's generation panel remains
qualitatively collapsed and its validation gain is small relative to its
capacity increase, so this still does not look like a simple capacity-bound
plateau.

Find them:

```bash
wsl bash -c "ps aux | grep -iE 'phase156|widedend' | grep -v grep"
```

Resume either by re-running the exact same command against the same
`--work-dir` -- the script picks up from the latest `epoch_N/` checkpoint
already on disk, it does not need an explicit `--resume` flag:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && ~/gtr_venv/bin/python3 -m manifoldflow.experiments.dendritic_basin_phase156_conversational_prototype \
  --train /path/to/ALM-benchmarks/mixed-conversational/train.txt \
  --valid /path/to/ALM-benchmarks/mixed-conversational/valid.txt \
  --work-dir runs/alm_phase156_conversational_500mb_widedend \
  --columns 1 --epoch-bytes 100000000 --max-body-bytes 500000000 \
  --episodic-enabled --episodic-k 6 --episodic-s 5 --episodic-strength 0.1 \
  --trust-delta-clip 0.1 \
  --deep-band-anchor-gate --deep-band-anchor-gate-k 12 --deep-band-anchor-gate-s 6 \
  --long-range-bank --long-range-context-bytes 12 --long-range-strength 8.0 &"
```

(Same command with `2_5gb_widedend` / `2500000000` for the other one.) Both
are CPU-bound and long-running -- launch with `&` or `nohup ... &`, not in
a shell you plan to close.

Pausing again, cleanly (checkpoints are per-epoch, so this loses at most
the in-progress epoch, never a completed one):

```bash
wsl bash -c "kill -TERM <pid>"
```

## Future-turn organ: next convergence run

### Evidence to carry forward

The user reports a finite-toy encode -> predict -> refine -> decode result with
1,536/1,536 oracle-code responses exact (100.00%), 1,397/1,536 causal refined
responses exact (90.95%), and 1,536/1,536 for a full-history one-turn baseline.
Removing the third-horizon forecast leaves the causal score unchanged at
1,397/1,536, and all 139 failures are reported in `Keep it.` cases. These
numbers are **externally reported and not locally verified**: the supporting
protocol, mathematics, fitted-state, and result files are not present as
unpacked root artifacts in this checkout. Treat the result as a trainability
smoke, not evidence that a farther horizon helps.

The locally locked natural representation audit is authoritative only for its
narrow claim:

- `future_turn_trajectory_v1_preregistration.json`,
  `future_turn_trajectory_v1_dev.json`, and
  `future_turn_trajectory_v1_audit.json` use 2,131 training conversations,
  25,162 adjacent transitions, and 62 development pairs.
- Full-history minus reset-history/current-turn-only at horizon one is +0.31477
  matched-minus-crossed margin (95% CI +0.25746 to +0.37570; 57/62 positive).
- Horizons two and three are worse than horizon one, temporal reversal is not
  cleanly rejected by the metric, and horizon four is underpowered. The result
  establishes persistent conversation identity/information, not turn-specific
  evolution, a useful farther horizon, a goal, or generation.
- `future_turn_codec_native500_v1_smoke.json` is a two-development-pair,
  four-training-conversation, 16-byte engineering smoke with a one-shot causal
  ridge predictor. It verifies direct free-byte actuation and harness invariants
  only; it is not an efficacy result.
- The locked full codec run is in
  `future_turn_codec_native500_v1_preregistration.json`,
  `future_turn_codec_native500_v1_dev.json`, and
  `future_turn_codec_native500_v1_audit.json`. With the frozen 500 MB
  checkpoint and one decoder pass over 1,601,513 bytes, the preregistered
  original-prefix evaluation contrast, oracle coupled minus paired wrong-code
  swap, is +0.12484 sampled free-generation margin (95% CI +0.08009 to
  +0.17063; 25/29 positive). Sparse oracle codes are distinct in every paired
  swap and the output-health gate passes. This establishes the natural
  codec/actuator, not the predictor or the horizon.
- The one-shot predicted-code route is null on the primary original-prefix
  evaluation against native, additive, and shuffled controls. It improves over
  native after the long interruption, but not over its additive or shuffled
  controls, so code-specific causal supply is still unestablished. The full
  native run used one pass and is not a decoder-convergence result. Its output
  remains low-quality; pair-conditioned generation margin is a mechanism assay,
  not evidence of thoughtful task completion.
- The locked deterministic recurrent run is in
  `future_turn_recurrent_codec_native500_v1_preregistration.json`,
  `future_turn_recurrent_codec_native500_v1_dev.json`, and
  `future_turn_recurrent_codec_native500_v1_audit.json`. It is a locked
  development advancement run, not untouched confirmation, because aggregate
  evaluation-partition latent diagnostics were inspected before the native
  generation lock. On the original-prefix evaluation partition, recurrent
  coupled minus paired wrong-code swap is
  **+0.0521166** sampled free-generation margin (95% CI **+0.0089542 to
  +0.0959356**). Full recurrent history beats the same transition given only
  the current user turn by **+0.0386178** (95% CI **+0.0019758 to
  +0.0781229**), its additive route by **+0.0396073** (95% CI **+0.0015002 to
  +0.0787010**), and a cross-pair recurrent-code shuffle by **+0.0538979**
  (95% CI **+0.0108970 to +0.0982436**). Recurrent minus one-shot is directional
  only at **+0.0351801** (95% CI **-0.0022415 to +0.0740178**), and recurrent
  minus native is **+0.0182485** with an interval crossing zero. This advances
  prior-vector recurrent supply as a causal mechanism, not as a demonstrated
  improvement over the one-shot or native generator.
- The preregistered transition-shuffle control was not implemented with the
  semantics named in the preregistration. The run permutes fitted next-token
  output coordinates and reconstructs the analytic history equation; it does
  **not** shuffle role/source-preserving observed successor targets and refit.
  Treat it as a malformed-transition control only. It cannot satisfy the
  temporal-ordering gate, regardless of its numerical result.
- After the long interruption, recurrent coupled minus native is **+0.0457802**
  with a positive interval, but paired wrong-code, current-user-only, and
  transition controls are null. Do not attribute that gain to correct recurrent
  supply. The outputs in both context modes remain low-quality; the endpoint is
  paired reference appropriateness, not independent task completion or
  thoughtfulness.
- BPB and free generation disagree in the recurrent run. The original-prefix
  free-generation mechanism signal appears while recurrent mean BPB is
  **0.01183 worse**. After the long interruption, free generation beats native
  by **+0.0457802** while teacher-forced BPB advantage is **-0.04594**, with its
  entire interval below zero. Generate and score first; BPB cannot promote or
  veto the mechanism.
- The corrected successor control refits from an exact successor-content
  derangement within target role/source strata. The locked artifacts are
  `future_turn_successor_shuffle_native500_v1_preregistration.json`,
  `future_turn_successor_shuffle_native500_v1_dev.json`, the post-hoc
  `future_turn_successor_shuffle_v1_latent_wsl_posthoc.json`, and the independent
  `future_turn_successor_shuffle_native500_v1_audit.json`. The control retains
  all 25,162 predictor rows, has no fixed or equal-content donor, preserves the
  target role/source/constant coordinates exactly, and preserves the target-
  content multiset in every stratum. Original evaluation dense margin is
  **0.9347449** intact versus **0.0182557** shuffled; after the long interruption
  it is **0.3589163** versus **0.0458393**. Every intact/shuffled sparse-code pair
  is distinct. This cleanly establishes learnable ordered association in vector
  space.
- The full native free-generation gate nevertheless fails. Original evaluation
  intact minus successor-shuffled is **+0.0279621** (95% CI **-0.0082839 to
  +0.0626742**), and the long-interruption contrast is **+0.0314386** (95% CI
  **-0.0019216 to +0.0648012**). The intact arm exactly replays the preceding
  locked recurrent result. Dense separation cannot override either null because
  free generation is primary.
- Three preregistered raw numerical hashes -- predictor state, original target
  token, and shuffled target token -- were produced on Windows and differ on
  WSL. The permutation hash agrees. A post-hoc WSL latent rerun exactly matches
  the native run's three hashes and all structural, donor, stratum, coordinate,
  and multiset invariants. Record this as an explained cross-platform
  preregistration deviation, not as a literal match of the Windows hash fields.
- The locked decoder exposure curve is complete in
  `future_turn_decoder_curve_native500_v1_preregistration.json` and
  `future_turn_decoder_curve_native500_v1_dev.json`. It ran for 4,799.00 seconds
  at 1/2/4/8 passes and exactly replayed all 496 mapped pair-condition rows from
  the one-pass reference. On the fixed 29-pair original-prefix evaluation
  partition, the final out-of-fold intact receiver beats the successor-shuffled
  receiver by **+0.0676590** free-generation margin (95% CI **+0.0229430 to
  +0.1083817**) and its same-writer paired wrong-code swap by **+0.0833808**
  (95% CI **+0.0226133 to +0.1431714**). Predicted-code receiver transmission is
  therefore demonstrated.
- Extra exposure is not demonstrated as the explanation. The pass-8-minus-pass-1
  same-writer specificity difference is **+0.0200868**, with 95% CI
  **-0.0455886 to +0.0821502**. Pass one was already code-specific, and the
  free-generation curve is non-monotonic even though teacher-forced residual
  fit improves monotonically. Record `receiver_undertraining_demonstrated=false`:
  the earlier null was primarily an oracle-code versus predicted-code training-
  distribution mismatch, not evidence that eight passes are better than one.
- The surprise-grown latent assay is in
  `surprise_future_transition_latent_v1_dev.json`. It uses no native ALM, free
  byte generation, backpropagation, or BPTT. A pre-update diagonal-Gaussian
  mixture surprise rule creates, promotes, consolidates, dormants, reactivates,
  and recycles locally updated future modes. The ordered fit produced 430
  lifetime promotions and retained 128 modes (17 consolidated, 111 juvenile).
  On the 28-pair long-interruption evaluation, however, its emitted point is
  worse than the ridge basal in MSE by **+0.0191342** (95% CI **+0.0132313 to
  +0.0248818**) and cosine by **-0.0113556** (95% CI **-0.0196692 to
  -0.0024623**); its proper mixture-score and every ordered-versus-shuffled
  primary interval cross zero. Consolidated-only is also worse than basal.
- The same surprise assay contains a useful but non-causal coverage diagnostic.
  A target-selected hindsight best ordered mode beats the basal point in
  long-interruption MSE by **-0.0495183** (95% CI **-0.0686995 to
  -0.0327735**), while the emitted point is worse than that best mode by
  **+0.0686525**. Shuffled modes also have useful coverage. With about 29.16
  active candidates (16.86 effective components), broad mixture activation is
  averaging incompatible futures. Surprise-driven candidate growth works;
  causal candidate selection does not yet.

All listed artifacts are under `runs/shared_goal/` and have `.sha256` sidecars.
The protected payload remained sealed and must remain sealed during development.

The receiver curve closes the interface question: predicted causal turn vectors
can control free generation, but longer receiver training has not earned its
cost. Before expensive particle competition, train a cheap categorical
distribution over retained surprise modes from compressed conversation state.
Use the observed successor vector's posterior responsibilities as soft,
self-supervised targets, and shortlist the smallest top-`k` set reaching a
preregistered posterior-mass threshold. Compare against affinity-only, global-
frequency, random capacity-matched, and successor-shuffled gates. Advance to
competition only if prequential categorical log gain and held-out shortlist
recall improve together and the shortlisted mixture improves over basal without
hindsight target selection. This is a selection gate, not a goal result.

That categorical gate has now run. The locked artifacts are
`future_mode_selector_latent_v1_preregistration.json` and
`future_mode_selector_latent_v1_dev.json`. A 144,072-byte selector uses 24
compressed causal context coordinates, 96 fixed random features, and one local
soft-responsibility delta-rule pass. It uses no native ALM, free generation,
autograd, backpropagation, or BPTT. The full run finished in 55.69 seconds and
passed all locked hash, count, freeze, finite-JSON, and raw-contrast checks.

On the 28-pair long-interruption evaluation, the learned selector reduces about
29 active futures to a mean shortlist of **7.48**, captures **82.62%** of the
hindsight posterior mass, and recalls its maximum-responsibility component on
**85.71%** of sides. Its categorical log loss beats affinity by **-0.6382 bits**
(95% CI **-0.9102 to -0.3442**), frozen random by **-1.5415**
(**-1.8393 to -1.2139**), and the same readout trained on successor-shuffled
targets by **-0.3766** (**-0.6960 to -0.0659**). It does not strictly beat
global frequency: **-0.1685** (**-0.4186 to +0.0380**). Shortlisted point MSE
versus basal is **-0.00100** (**-0.00938 to +0.00582**) and proper bits are
**-0.00923** (**-0.02252 to +0.00232**), both directional but unresolved.
Therefore `cheap_selector_advances_to_expensive_competition=false`.

The split localizes the next cheap change. On the original-prefix evaluation,
categorical log loss is **1.5734 bits** and strictly beats all four controls,
including global frequency by **-0.4092** (95% CI **-0.7317 to -0.1483**).
Selection information exists in the recent compressed state but weakens across
the artificial interruption. Before particle competition, cross a retained
slow conversation aggregate with the current fast state in the same categorical
readout. Require the slow-state factor to beat both fast-state-only and global-
frequency controls after interruption; do not tune shortlist size on evaluation.

The locked two-timescale follow-up is complete in
`future_mode_selector_timescale_v1_preregistration.json` and
`future_mode_selector_timescale_v1_dev.json`. The slow state is a causal
role-bound turn aggregate with decay 0.98, separately projected to 24 training-
fit coordinates. Fast-only, slow-only, and concatenated fast-plus-slow arms use
96 fixed random features and exactly 100,104 trainable readout bytes apiece.
The full run took 57.67 seconds and passed all locked integrity and arithmetic
checks.

On the 28-pair long-interruption evaluation, fast-plus-slow categorical log loss
is **3.3173 bits**, versus 3.3541 fast-only, 3.3218 slow-only, and 3.4768 global
frequency. The joint-minus-fast effect is only **-0.03679** (95% CI
**-0.10162 to +0.01554**); joint-minus-global is **-0.15950** (**-0.35998 to
+0.01488**). The joint arm does beat successor-shuffled by **-0.38740**
(**-0.62592 to -0.16975**), shortlists 7.73 components, captures 80.50% of
posterior mass, and recalls the maximum-responsibility mode on 82.14% of sides.
Its MSE is **+0.00240** versus basal (CI **-0.00400 to +0.00826**) and its proper
bits are **-0.01032** (CI **-0.02923 to +0.00480**). Both the slow-state and
competition-advancement gates fail.

Do not generalize this to “slow state contains no information.” Slow-only has
the best posterior mass (86.00%) and argmax recall (87.50%) after interruption,
while the joint mapping is slightly but significantly worse than fast-only on
the original prefix. The tested concatenated fixed-capacity random-feature
fusion does not add the two signals cleanly. If selection work continues, use a
nested additive-logit test that retains the fitted fast expert and adds a
separate slow expert plus a locally learned mixing scalar. That isolates the
incremental slow contribution without remapping or consuming the fast feature
route. Expensive particle competition remains blocked.

The nested follow-up is complete in
`future_mode_nested_selector_v1_preregistration.json` and
`future_mode_nested_selector_v1_dev.json`. It first trains and freezes the fast
selector, then initializes separate slow and fast residual-logit experts at
exactly zero. Before residual learning every nested distribution is byte-for-
byte the frozen fast distribution. Slow and capacity-control residuals each
have 100,104 trainable bytes and one matched update pass. The run took 67.37
seconds and passed every locked integrity, freeze, allocation, and raw-arithmetic
check.

This removes the concatenation confound but does not recover an interrupted-
context effect. On the 28-pair long-interruption evaluation, nested fast-plus-
slow minus frozen-fast log loss is **-0.00171 bits** (95% CI **-0.05024 to
+0.04508**); versus the second-fast capacity control it is **-0.01563**
(**-0.05324 to +0.01879**), versus global frequency **-0.15426**
(**-0.39312 to +0.03110**), and versus successor-shuffled slow learning
**-0.15843** (**-0.47501 to +0.10469**). All four incremental-slow intervals
cross zero. The nested arm shortlists 6.86 modes, captures 80.95% of posterior
mass, and recalls the top mode on 83.93% of sides, but neither MSE nor proper
mixture bits beats basal decisively. Both advancement gates fail.

On the original prefix, the slow residual does beat the frozen fast base by
**-0.03286 bits** (95% CI **-0.06461 to -0.00448**) and the shuffled-slow
control, but not the second-fast capacity control. Thus the residual learner is
wired correctly and can exploit slow state where the state still resembles its
training distribution; no distinct slow information survives the interruption
under this exponential aggregate. Do not sweep its decay yet. The next
mechanistic target, if pursued, is selective persistence: a surprise-grown
evidence/obligation ledger whose assemblies retain predictive utility without
being continuously blended with every intervening turn. A smoothed instant is
now a controlled null for the missing shared-goal role.

The selective-persistence follow-up is complete in
`selective_predictive_ledger_v1_preregistration.json` and
`selective_predictive_ledger_v1_dev.json`. A target-free 24-coordinate event
projection grew a full 64-prototype bank; 44 assemblies survived the locked
minimum-fire and nonnegative prequential-utility rule. The same retention mask
was applied to ordered, current-only, and successor-shuffled evaluation arms.
The fast selector and surprise candidate population were frozen before ledger
training, all three ledger readouts used 66,048 trainable bytes, and the run
passed every locked hash, count, freeze, finite-JSON, and replay check.

The 28-pair long-interruption result is a mechanistic partial positive but an
advancement null. The retained ledger shortlists **6.79** modes, captures
**80.50%** posterior mass, and recalls the top-responsibility mode on **83.93%**
of sides. Its log loss is **3.2837 bits**, directionally better than fast-only
at 3.3083, global frequency at 3.4768, and shuffled-ledger at 3.3785, but all
three paired confidence intervals cross zero. It does strictly beat the
current-turn-only control by **-0.09692 bits** (95% CI **-0.18093 to
-0.01362**). Neither shortlisted MSE nor proper score decisively beats basal,
so `advances_to_expensive_competition=false`.

The original-prefix comparison runs in the opposite direction: persistent
minus current-only is **+0.06319 bits** (95% CI **+0.01096 to +0.11733**), while
after interruption persistence wins. This is useful evidence that nondecaying
event state is doing persistence-specific work rather than acting as generic
extra capacity. It is not enough to choose a single future or promote a native
generation run. If refined, target the read/write policy and prototype
granularity rather than another decay sweep: retain obligations and corrected
facts, suppress already discharged or locally redundant entries, and require
the same interruption-specific advantage against fast, global, and shuffled
controls before paying for particle competition.

The shared-core family routing assay is complete in
`future_vector_family_v1_preregistration.json` and
`future_vector_family_v1_dev.json`. It groups the unchanged retained-ledger
candidate vectors into four training-scaled Ward families, selects at most two
families, and admits only their original member vectors to the eight-edge
budget. Flat and random size-matched controls retain the exact same candidate
probabilities.

On long-interruption evaluation, the family stage collapses to **1.02 selected
families** but still contains **6.80 edge vectors**, versus 6.79 for flat top-k.
Posterior mass is 80.83% versus 80.50%, with identical 83.93% top-mode recall.
Pairwise competition is not reduced (20.75 versus 20.66), and the directional
mass advantage over random grouping is unresolved: **+0.03387** with 95% CI
**-0.01624 to +0.08029**. The gate fails.

Do not spend a native run on this clustering rule. It selected a dominant label
without removing the within-family choice. If the proposal continues, make the
core/edge split generative: learn stable shared content, train the decoder to
express that core once, and compete only over residual edge activity. Require
oracle/wrong core and swapped-edge causal controls before free-generation
promotion. The complete project state is in
`docs/ALM_Shared_Goal_Handoff.md`.

The core/edge context-specificity assay is complete in
`future_vector_core_edge_v1_preregistration.json` and
`future_vector_core_edge_v1_dev.json`. Narrowing is fixed to flat top-k
(cheapest, and the family assay found nothing beat it); the core is the
cheapest possible estimator, a mass-weighted mean of that shortlist. Since
within-shortlist reconstruction is exact by construction for a mean split,
the only test run is cross-pair: a Sattolo-deranged foreign pair's core
substituted into this pair's own edge sum, scored against this pair's real
target.

On the same 28-pair long-interruption evaluation, own core beats foreign core
on MSE (95% CI **0.077 to 0.158**, above zero) and proper score (95% CI
**-0.163 to -0.081**, below zero). The gate passes. Run it yourself:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && PYTHONPATH=src ~/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_core_edge_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --report runs/shared_goal/future_vector_core_edge_v1_dev.json"
```

Do not read this as validating the mean estimator for production -- it only
clears the cheap bar before a learned/persistent core estimator and a real
receiver are worth building. See `docs/ALM_Shared_Goal_Handoff.md`.

The persistent-core generalization assay is complete in
`future_vector_persistent_core_v1_preregistration.json` and
`future_vector_persistent_core_v1_dev.json`. Context-affinity-gated slots,
EMA-updated (same schedule as `vision.hpp`'s `update_winner`), fit only on
the selection partition's own per-pair cores, evaluated on evaluation (never
seen while fitting). Bar is noninferiority to the per-episode mean, not
outright superiority.

The gate **fails**: noninferior on MSE (+0.0370, CI 0.0036 to 0.0696) but not
on proper score (+0.0488 bits/coordinate, CI 0.0133 to 0.0838, past the +0.05
margin). Diagnostic: on the decisive scope the estimator collapsed to
**exactly one slot** across all 30 selection sides (`context_bandwidth=1.0`,
`min_context_affinity=0.50` too permissive for that mode's context geometry)
-- versus all 16 slots used elsewhere. Run it yourself:

```bash
wsl bash -c "cd ~/sonic-manifold-flow && PYTHONPATH=src ~/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_persistent_core_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --report runs/shared_goal/future_vector_persistent_core_v1_dev.json"
```

`--persistent-context-bandwidth` and `--persistent-min-context-affinity` are
the knobs most directly implicated by the 1-slot collapse; a narrower
bandwidth or higher affinity threshold is the natural next, separately
preregistered sweep before concluding persistence itself doesn't generalize.
See `docs/ALM_Shared_Goal_Handoff.md`.

### Persistent-core knob sweep: the prescribed rescue ran and failed

The preregistered sweep is complete in
`future_vector_persistent_core_sweep_v1_preregistration.json` and
`future_vector_persistent_core_sweep_v1_dev.json`. It crosses bandwidths
{1.0, 0.5, 0.25} with affinity floors {0.50, 0.75, 0.90} (nine configs,
config 0 the v1 anchor), every other estimator knob frozen at v1 values.
The evaluation partition was never touched during selection: configs were
ranked by a deterministic pair-parity cross-validation inside the selection
partition (15 pairs fit / 15 pairs eval), and the chosen configuration was
then run once through the unchanged locked assay as
`future_vector_persistent_core_v2_dev.json`.

```bash
wsl bash -c "cd ~/sonic-manifold-flow && PYTHONPATH=src ~/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_persistent_core_sweep_ablation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --report runs/shared_goal/future_vector_persistent_core_sweep_v1_dev.json"
```

The CV table answers the two competing failure explanations separately.
Contexts ARE separable: bandwidth 1.0 / affinity 0.75 holds six slots with
only 10% global fallback, and bandwidth 0.5 / affinity 0.50 holds thirteen
with 40%. But slots do not transfer: every config with genuine slot usage is
strictly worse than the global mean (config 1: +0.0443 bits, config 3:
+0.0520), while the "best" configs only win by falling back to the global
core for essentially every eval row (configs 5-8: fallback 0.97-1.00,
+0.0387). The mechanical rule selected config 5 (bandwidth 0.5, affinity
0.90) -- the exact case the preregistration's fallback caveat flagged: a
pass there would have measured the global EMA fallback, not persistent
slots.

The final gate failed anyway, with the same shape and nearly the same
numbers as v1: long-interruption MSE +0.0368 (95% CI +0.0033 to +0.0694,
noninferiority passes) but proper bits +0.0485 (+0.0131 to +0.0835), past
the +0.05 margin -- versus v1's +0.0488 (+0.0133 to +0.0838). The full
60-row fit used 16 slots with 42 recycles, so the failure is not a residual
one-slot artifact.

Strict conclusion per the preregistered interpretation: the affinity-slot
persistence family does not generalize after the prescribed narrowing
rescue; the core/edge persistence line stops here. No further knob sweeps
without a new mechanism hypothesis. The cheap per-episode mean core's
cross-pair specificity result stands; what failed is every attempt so far to
make that core persistent across episodes. Report and captured stdout are
byte-identical with SHA-256 `f857ff8b92dbd2a24c3062285c886011da5a52e2036e855fc5af1162b4060ccf`
(sweep) and `411732fd6325b117ea55556b0354090652cfac81dcf80fb325e47a9fe075f43a`
(final gate). An earlier first launch of the sweep crashed on an unbound
name before any result was produced; the preregistration's implementation
hash was updated before any result existed. See
`docs/ALM_Shared_Goal_Handoff.md`.

### Core/edge free-generation probe: the qualitative read that closed the line

Per the doctrine that free generation is primary, the core/edge representation
was driven through the frozen native 500 MB cortex's existing codec receiver
(no new training beyond the codec's own one matched fit pass). Locked
implementation: `src/manifoldflow/experiments/dendritic_basin_future_core_edge_native_generation.py`;
artifacts `future_core_edge_native_generation_probe_v1_{preregistration,dev}.json`
plus `future_core_edge_native_generation_probe_v1_dev.digest.txt`, which dumps
every arm's generated text for every pair and both context modes.

```bash
wsl bash -c "cd ~/sonic-manifold-flow && PYTHONPATH=src ~/gtr_venv/bin/python \
  -m manifoldflow.experiments.dendritic_basin_future_core_edge_native_generation \
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 \
  --checkpoint runs/alm_phase156_conversational_500mb_widedend/epoch_147/checkpoint.pkl \
  --report runs/shared_goal/future_core_edge_native_generation_probe_v1_dev.json \
  --digest runs/shared_goal/future_core_edge_native_generation_probe_v1_dev.digest.txt"
```

Three decisive findings:

1. **The quantizer discards the edge.** Core, core plus this side's edge
   sum, and core plus the paired side's edge sum quantize to the same
   sparse code on every side; those arms generated byte-identical output on
   all 116 sides. The core/edge distinction exists only in latent space.
2. **The oracle ceiling is word salad on this base.** Even the arm handed
   the held-out answer's own code cannot produce coherent text on these
   long LMSYS contexts. No receiver on this frozen base can express
   thoughtful content.
3. **The core's effect is a register nudge, original-prefix only.** Sampled
   reference margins: core minus native +0.0298 (evaluation, CI +0.0070 to
   +0.0526), +0.0382 (selection); core minus foreign +0.0547 (all pairs,
   CI +0.0303 to +0.0784). Qualitatively the core opens in the correct
   language/assistant register more often. After interruption all contrasts
   null (+0.0007 core minus foreign).

The probe's oracle arm reproduces the codec native500 oracle outputs
byte-for-byte on all 232 shared sides -- implementation verified end to end.
Report and captured stdout are byte-identical, SHA-256
`e9180d5eb7eb89e435cac7fc25709eced8efeedf450569e89cf563ba76b24b3a`; digest
`b5276945d15145c70602ab7ababb0a45f6fbb3053583466ea33bac46d0f4827c`. An
earlier launch crashed at the digest write after the full computation; the
hash was corrected and the run rerun before any result was recorded.

Strict conclusion: the core/edge line stops on the current base and
interface. Reviving it requires a base that generates coherently on
multi-turn contexts and an actuator that preserves the split's content.
See `docs/ALM_Shared_Goal_Handoff.md`.

### Mechanism under test

Train the future organ as turn-to-turn latent dynamics, but separate three
functions that the earlier point predictor conflated:

1. A persistent evidence/obligation ledger records what remains true and what
   the interaction still needs to accomplish.
2. A stochastic/particle receding-horizon state evolves prior turn vectors,
   shifts at turn boundaries, and is corrected/reweighted by new observations.
   Use real rate/noise bottlenecks for farther slots and preserve competing
   futures when the next user turn is uncertain.
3. A directly coupled decoder generates the immediate response from the selected
   near-turn vector and a changing remaining-content state. Do not reduce this
   path to post-hoc candidate scoring.

This dynamics model is self-supervised at turn boundaries. With an anchored
codec, the next observed completed turn supplies a stop-gradient target vector,
while the transition receives prior role-tagged vectors, the ledger, and the
newly observed interlocutor turn. The byte loss trains the decoder; a
vector-distribution loss trains the organ. Begin with actual prior vectors. A
generated assistant state must not be paired with the logged next user turn as
though that user had responded to the counterfactual output. Use scheduled
predicted-state exposure only in an on-policy or programmatic environment that
supplies the resulting observation; on fixed natural logs, keep teacher-forced
fitting separate from labeled state-perturbation robustness diagnostics. Do not
let the encoder co-adapt freely until oracle reconstruction and causal vector
swaps stay stable.

Train and gate in order: first make the actual held-out turn code expressible;
then freeze or strongly anchor that codec while learning one-step transition;
then train recursive two- and four-turn rollouts, using predicted latent inputs
only where the next observation is causally valid for that generated path;
finally allow matched co-adaptation. Save
geometric checkpoints so undertraining can be distinguished from a flat
controlled learning curve. A deterministic one-step regression remains a
mandatory baseline, not the final uncertainty model.

### Locked 2x2

Use the same decoder interface and cross ledger state with future depth:

| Arm | Persistent ledger | Future controller |
| --- | --- | --- |
| A | Off | Particle near-turn only (`K=1`) |
| B | On | Particle near-turn only (`K=1`) |
| C | Off | Particle receding horizon (`K>1`) |
| D | On | Particle receding horizon (`K>1`, full mechanism) |

Match train examples, update count, active allocation, initialization, proposal
or particle compute, and generation draws. Spend the removed rollout compute in
A/B on additional near-turn particles so the future-depth contrast retains the
same uncertainty family. Run the separately optimized deterministic full-history
one-turn predictor as an additional strong baseline. Also run oracle actual-code,
donor-vector swap, shuffled trajectory, reset/wrong ledger, disconnected decoder,
and no-far-horizon controls. Hold the pre-action tape fixed during each generated
turn; update it only from committed output and subsequent external observation,
so the state cannot rationalize a drifting prefix after the fact.

### Free-generation-first advancement rule

Preregister tasks in which locally plausible candidates lead to different later
ledger discharge or external consequences: delayed obligations across
interruptions, corrections that invalidate an older fact, contradictory near
wording with the same long-term requirement, and unexpected user branches.
Strictly speaking, `K>1` cannot add prefix information over an unrestricted
full-history one-turn predictor; that predictor can represent the Bayes next
response directly. Therefore a horizon win is evidence for an inductive bias or
delayed-credit/selection path, not information-theoretic necessity. Require the
rollout selector to read an independently learned terminal ledger-discharge or
consequence signal; simply adding farther next-vector losses is still likelihood
training at more horizons. Evaluate original causal prefixes only; oracle answer
codes are leakage ceilings and never enter causal inference.

Advance D only if all of the following hold:

1. Complete independently checked free-generation success beats A, B, C, and
   the deterministic full-history one-turn baseline, with a positive
   ledger-by-horizon interaction across held-out conversations.
2. Removing the far slot specifically damages far-dependent cases; no claim of
   horizon benefit is allowed if the lesion ties, as in the reported toy.
3. Matched versus donor/swapped vectors, correct versus reset/wrong ledgers, and
   connected versus disconnected decoding order causally while shuffled controls
   fail.
4. Benefits survive multiple seeds and later checkpoints without loss of
   printability, valid UTF-8, diversity, termination, or throughput beyond the
   locked budget.

Generate and score before inspecting teacher-forced BPB. BPB, latent MSE,
cosine discrimination, and oracle reconstruction diagnose plausibility, coding,
and optimization; none may override a null or degraded complete-interaction
free-generation result.

## The co-adapt campaign (2026-09-15)

Corpus, base, and three convergence runs. Operational record; verdicts in
`docs/ALM_Shared_Goal_Handoff.md`.

### Corpus build (coadapt_corpus_v1)

```bash
# on Windows (Google Drive reads are only stable natively; output over UNC):
.venv/Scripts/python.exe scripts/build_coadapt_corpus_v1.py \
  --wiki "C:\path\to\gdrive\training_data\knowledge\wikipedia_en.jsonl" --wiki-bytes 4000000000 \
  --smoltalk-cot "C:\path\to\gdrive\training_data\chat\smoltalk_reasoning.jsonl" --smoltalk-cot-bytes 1500000000 \
  --lmsys "C:\path\to\gdrive\training_data\chat\lmsys_chat_1m.jsonl" --lmsys-bytes 3000000000 \
  --smoltalk-preference "C:\path\to\gdrive\training_data\chat\smoltalk_preference.jsonl" --smoltalk-preference-bytes 700000000 \
  --smoltalk-everyday "C:\path\to\gdrive\training_data\chat\smoltalk_everyday.jsonl" --smoltalk-everyday-bytes 3000000 \
  --train-output "\\wsl.localhost\Ubuntu\home\user\sonic-manifold-flow\runs\coadapt_corpus_v1\train.txt" \
  --valid-output "\\wsl.localhost\Ubuntu\home\user\sonic-manifold-flow\runs\coadapt_corpus_v1\valid.txt" \
  --valid-bytes 20000000 --remount-cmd " " \
  --report "\\wsl.localhost\Ubuntu\home\user\sonic-manifold-flow\runs\coadapt_corpus_v1\build_report.json"
```

Unique-only quotas (no cycling), global turn-prefix boilerplate cap 16,
256 KB interleaving, SmolTalk-CoT as a first-class source. ~7.6 GB train.
Do not stream large reads through WSL's drvfs: the Drive mount drops and
its cache lives on a full C:. Copy slices natively on Windows and write
WSL over UNC (71 MB/s) if source slices are needed.

### Fresh base (coadapt_base_500mb_v1)

Same launch as the 500mb_widedend run, swapped to
`runs/coadapt_corpus_v1/{train,valid}.txt`, work-dir
`runs/coadapt_base_500mb_v1`, `--target-epochs 147`. Resume by re-running
the identical command (picks up the latest epoch checkpoint). Crashed at
epoch 36 with a native-checkpoint deserialize failure -- fixed in
`cpp/include/alm/row_arena.hpp` (phantom zero-count keys after count
saturation) with a loader compatibility break in `serialize.hpp`;
`cpp/tests/row_arena_test.cpp` locks it. Rebuild libalm_native.so after
touching either header.

### Convergence runs (2x2: ledger x horizon)

Harness: `dendritic_basin_shared_goal_convergence_train.py` (matched
budgets, snapshots at passes 0/0.25/0.5/1/2/4/8) and
`..._evaluate.py` (per-mark free-generation evals). Receiver-path run
initialized from the epoch-16 checkpoint (prereg amended; epoch 147 was
blocked by the serialize bug). Verdict: gate FAILS -- D best on
long-interruption (+0.0389 arm-minus-native at 8 passes) but not
significant, interaction negative, lesion uninformative (quantizer maps
reset variants to identical codes).

Inverted-interface runs (`vector_conditioned_base.py`: dense float32
frames prime the base directly; response-byte surprise and plasticity in
the base; no quantizer, no receiver):

- `..._convergence_veccond.py` (linear organ): **arm A long-interruption
  arm-minus-native rises to +0.0408, significant at 8 passes** -- first
  significant co-adaptation result. Weight-bound amendment (organ weights
  clipped to +/-50) recorded in the prereg after the unbounded delta rule
  diverged to NaN on arm B in the first launch.
- `..._convergence_veccond2.py` (live dendritic predictor:
  SurpriseGrownFutureTransition grown across passes; 2x2 becomes a
  readout question on the mixture): running at this writing; v1's A-curve
  is the comparison line.

Evaluator per mark:
`..._convergence_veccond2_evaluate.py --snapshot-dir ... --mark N`.
Completion markers: `convergence_evals_complete.done`,
`veccond_chain.done`, `veccond2_chain.done`.
