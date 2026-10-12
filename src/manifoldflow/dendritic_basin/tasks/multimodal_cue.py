"""Phase 11 (design doc section 54: "Multimodal stream -- Vision + audio
+ language using the same relational core."; section 58's "stronger
multimodal criterion": cross-modal assemblies should emerge). A single
combined vector stream carrying three modalities in one vector, where
the channel that *resolves* an ambiguity and the channel that *is*
ambiguous belong to different modalities -- so a genuinely useful
relation must be cross-modal, not just reuse Phase 10's within-audio
structure.

Channel layout (`N_CHANNELS = 6`):

  0: BOUNDARY     -- episode-boundary marker (a discrete, language-like
                      event: "start of utterance")
  1: VISUAL_CUE   -- +1.0 or -1.0 for the whole episode, set once at
                      episode start (vision: "which object is present")
  2: DECOY        -- an uninformative language-like scalar, always ~0
                      plus noise -- present in the same vector, but
                      irrelevant, to test that discovery doesn't latch
                      onto every channel indiscriminately
  3: AMBIGUOUS_TONE -- an audio bin, active for one step regardless of
                      episode (identical in both cases, exactly Phase
                      10's ambiguous tone)
  4: E_BIN        -- audio bin that fires next if `VISUAL_CUE == +1.0`
  5: F_BIN        -- audio bin that fires next if `VISUAL_CUE == -1.0`

The task: predict whether `E_BIN` or `F_BIN` fires after the ambiguous
tone. The only channel that resolves this is `VISUAL_CUE` -- a different
modality than the audio target being predicted.
"""

from __future__ import annotations

import numpy as np

N_CHANNELS = 6
BOUNDARY, VISUAL_CUE, DECOY, AMBIGUOUS_TONE, E_BIN, F_BIN = range(6)


def generate_multimodal_stream(
    n_episodes: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns `(T, N_CHANNELS)`: per episode, one boundary row, one
    visual-cue row, one filler row, one ambiguous-tone row, then the
    cue-resolved audio row.
    """
    vectors: list[list[float]] = []
    for _ in range(n_episodes):
        cue = 1.0 if rng.integers(0, 2) == 0 else -1.0

        boundary_row = rng.normal(0.0, noise_std, size=N_CHANNELS)
        boundary_row[BOUNDARY] += 1.0
        vectors.append(boundary_row.tolist())

        cue_row = rng.normal(0.0, noise_std, size=N_CHANNELS)
        cue_row[VISUAL_CUE] += cue
        vectors.append(cue_row.tolist())

        filler_row = rng.normal(0.0, noise_std, size=N_CHANNELS)
        vectors.append(filler_row.tolist())

        ambiguous_row = rng.normal(0.0, noise_std, size=N_CHANNELS)
        ambiguous_row[AMBIGUOUS_TONE] += 1.0
        vectors.append(ambiguous_row.tolist())

        resolved_row = rng.normal(0.0, noise_std, size=N_CHANNELS)
        resolved_row[E_BIN if cue > 0 else F_BIN] += 1.0
        vectors.append(resolved_row.tolist())

    return np.array(vectors, dtype=np.float64)
