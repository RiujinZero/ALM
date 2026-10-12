"""Learned write-eligibility ("salience") gate - the piece named in the
grant white paper's Figure 1 Event layer ("Tracks identity / Detects
change / Scores salience") that today's write trigger doesn't have.
`session_builder.py` only attaches `statement_spans` metadata to
`episode_type == "distractor"` episodes, so `run_step`'s write phase can
only ever write scripted distractor statements - a chat episode has
structurally zero chance of being written, no matter what it says,
because nothing scores it. That scripted trigger was a deliberate first
step (see `docs/Salient_Gate_Dataset_Runbook.md`): it isolates "can Sonic
use retrieved writes at all" from "what should trigger a write," and the
production run's retrieval accuracy holding above 0.98 is the evidence
the first question is answered. This module is the second piece.

Validated before being wired in, not just designed: a tiny two-layer head
of this exact shape, trained on Qwen-labeled real spans pooled from
Sonic's own in-context hidden states (the same pooling this module's
`forward` does - one real forward pass over the packed window, mean-pool
the candidate span's token range), reached 94.4% held-out accuracy
against an 80.8% majority-class baseline on real data (see the runbook's
"Salience gate sanity check" entry) - a genuine, reproducible signal, not
a hoped-for one.

This module only scores a pooled span; it does not decide what counts as
a candidate span in the first place. For distractor episodes that's
already `statement_spans`. For chat/filler content, no equivalent
candidate-span extraction exists yet - a separate, unsolved problem this
module's introduction does not by itself resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SalienceGateConfig:
    d_model: int
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")


class SalienceGateHead(nn.Module):
    """Scores a pooled hidden-state span for write-worthiness. Returns a
    raw logit (not a probability) - callers use `BCEWithLogitsLoss` for
    training and `torch.sigmoid` for an inference-time score/threshold,
    matching the sanity check's own validated training setup exactly."""

    def __init__(self, config: SalienceGateConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        """`pooled`: `[batch, d_model]`, mean-pooled (not yet normalized)
        hidden states - same convention as `SonicKeyProjection.forward`
        and `_batched_pool`'s output elsewhere in this project. Returns
        `[batch]` raw logits."""
        if pooled.ndim != 2 or pooled.shape[-1] != self.config.d_model:
            raise ValueError("pooled must have shape [batch, d_model]")
        return self.net(pooled).squeeze(-1)


def salience_gate_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """`logits`: `[batch]` raw gate outputs. `labels`: `[batch]`, 1.0 for
    a real write-worthy span (from the Qwen-labeled dataset), 0.0 for a
    sampled negative (see `docs/Salient_Gate_Dataset_Runbook.md` for why
    the labeled dataset itself is all-positive and negatives must be
    sampled separately - the same same-window word-length-matched
    non-overlapping-substring sampling the sanity check validated)."""
    return F.binary_cross_entropy_with_logits(logits, labels)
