# ALM

A byte-level continual learner built from dendritic, locally learning components, with no tokenizer
and no backpropagation in the core learner.

> **Unmaintained.** This is a research prototype released as-is. It is no longer developed or
> supported; issues and pull requests are not monitored.

## What it is

ALM reads an ordered stream of raw bytes once, forward in time. It predicts each byte before it
arrives and adjusts its own structure from the outcome. There is no separate training and inference
phase and no fixed parameter count: structure grows from repeated patterns and is pruned to stay
within a memory budget.

The V1 architecture is organized as three interacting organs:

1. A sparse predictive cortex that learns reusable local assemblies and temporal routes.
2. A slow regulatory/context organ that represents the current predictive regime and biases local
   computation.
3. A sparse recurrent episodic field that captures events and reinstates the context-appropriate
   successor from partial cues.

ALM grew out of the ManifoldFlow memory work, so this repository still carries the `manifoldflow`
package name and the earlier Sonic + ManifoldFlow code. That earlier stage is published separately
as [sonic-manifold-flow](https://github.com/RiujinZero/sonic-manifold-flow).

## Layout

| Path | Contents |
|---|---|
| `src/manifoldflow/dendritic_basin/` | The dendritic learner: cortex, assemblies, episodic field, routing, CPU and GPU stream implementations |
| `src/manifoldflow/alm/` | ALM language harness: data, model, learning, training and evaluation |
| `src/manifoldflow/experiments/` | Experiment, probe and training entry points (many exposed as `manifoldflow-*` console scripts) |
| `src/manifoldflow/` | The ManifoldFlow memory library, Sonic adapters and world demo inherited from the earlier stage |
| `cpp/` | `alm_native`, the C++ backend for the sequence cortex ([cpp/README.md](cpp/README.md)) |
| `scripts/` | Corpus and byte-stream builders, dataset generators, audits and run launchers |
| `configs/` | Experiment configuration |
| `vendor/GTR-sonic-ssm/` | Vendored Sonic SSM training code, kept for reference |
| `tests/` | Unit tests |
| `docs/` | Architecture and design documents |

## Documentation

- [Architecture overview](docs/ALM_Architecture_Overview.md): a short, non-technical summary
- [ALM V1 Architecture](docs/ALM_V1_Architecture.md): the full technical description
- [Dendritic Predictor Organ Architecture v2](docs/Dendritic_Predictor_Organ_Architecture_v2.md)
- [Dendritic Basin Learning Design v0.1](docs/Dendritic_Basin_Learning_Design_v0.1.md) and its later revisions
- `docs/Dendritic_Stream_Learner_Phase*.md`: the design as it evolved, one document per phase
- [Language harness runbook](docs/ALM_Language_Harness_Runbook.md)
- [Shared-goal convergence evaluation contract](docs/ALM_Shared_Goal_Convergence_Evaluation_Contract.md)

[README_ORIGINAL.md](README_ORIGINAL.md) is the working README from development. It covers the
earlier ManifoldFlow prototype in detail and may refer to result files that are not part of this
release. Other documents may likewise mention run outputs that were left out.

## Getting started

Requires Python 3.10+ and PyTorch 2.2+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The native backend is built with CMake from `cpp/`; see [cpp/README.md](cpp/README.md).

## What is not included

Model checkpoints, training corpora, run outputs and experiment result logs are not part of this
repository. The builders in `scripts/` produce the byte streams and corpora from public datasets.

## License

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use, modify and
share this code for personal, research and other noncommercial purposes. Commercial use is not
permitted. Third-party components keep their own licenses; see
[the third-party notices](docs/THIRD_PARTY_WORLD_NOTICES.md).

Required Notice: Copyright (c) 2026 RiujinZero (https://github.com/RiujinZero/ALM)
