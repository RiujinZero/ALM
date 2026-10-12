"""Trainable retrieval-key projection - the fix the discrimination probe's
finding pointed at directly (`sonic-pretrained-memory-discrimination-probe.md`):
with a frozen Sonic and only `SonicMemoryPort` trainable, retrieval
accuracy sat at a flat ~83.6% ceiling across 2000 training steps because
nothing in that setup could touch it - `BasinMemory.read()`'s routing
depends entirely on the frozen checkpoint's own mean-pooled hidden
states, and `SonicMemoryPort` only ever sees retrieval's *output*, never
influences what gets retrieved in the first place.

`SonicKeyProjection` sits between "mean-pooled frozen hidden state" and
"key handed to `BasinMemory.write`/`.read`" - applied identically to both
fact and query representations (a shared, symmetric projection, matching
nearest-neighbor search's own assumption that both sides of a comparison
live in the same space).

It is trained by a separate contrastive (InfoNCE-style) loss over the
*undetached* projected keys - not by backpropagating through
`BasinMemory.read()`'s nearest-neighbor selection, which is a discrete,
non-differentiable operation with no gradient to give. This is the
standard fix for training a retrieval key space (dense-retrieval-style
contrastive training): shape the embedding geometry so that a fixed,
untrained nearest-neighbor search naturally becomes more accurate,
rather than trying to differentiate through the search itself.

Was originally a single `Linear` layer, identity-initialized. Upgraded to
`Linear -> ReLU -> Linear` after `scripts/round_trip_gate_retrieval_test.py`
plus a raw-vector control test (`scripts/diagnose_raw_pooled_similarity.py`
- see docs/Salient_Gate_Dataset_Runbook.md's "Contrastive retrieval
training: three attempts, and why none of them worked") showed the
*linear* map's own ceiling, not training data volume, was the bottleneck:
Sonic's raw pooled hidden states already collapse several unrelated
natural-language statements toward the same region, and a linear map can
only rotate/rescale that existing geometry - it cannot introduce
discriminative structure that isn't already linearly present. Three
escalating rounds of contrastive training data (1,540 -> 8,782 -> 26,124
pairs) left round-trip retrieval accuracy at exactly the same 0.300 each
time, and the raw-vector-only control (no projection at all) gave that
same 0.300 - conclusive that a nonlinear map was needed, not more data.

`hidden_dim` defaults to `2 * dim` specifically so the network can
reproduce *any* linear map exactly at construction time, via the
identity `relu(x) - relu(-x) = x` (true for every real x): with
`layer1`'s weight rows stacked as `[W; -W]` and `layer2`'s weight columns
stacked as `[I, -I]`, the whole network computes exactly `x @ W^T`
before any further training happens. `from_legacy_linear` uses this to
migrate an already-*trained* single-`Linear` checkpoint (the production
run's own key_projection, ~25,000 updates of real InfoNCE training via
the main loop, not something to discard) into the new architecture
without regressing whatever it had already learned - only from that
point does the added nonlinear capacity get a chance to do anything a
linear map couldn't.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SonicKeyProjectionConfig:
    dim: int
    hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")


class SonicKeyProjection(nn.Module):
    """A trainable Linear -> ReLU -> Linear map from raw pooled hidden
    states to retrieval keys - see module docstring for why this
    replaced a single Linear layer, and for the identity-initialization
    trick both this class's own __init__ and from_legacy_linear rely on."""

    def __init__(self, config: SonicKeyProjectionConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else 2 * config.dim
        self.layer1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, config.dim, bias=False)
        with torch.no_grad():
            if hidden_dim == 2 * config.dim:
                # Reproduces the exact identity map at construction time -
                # see module docstring's relu(x)-relu(-x)=x explanation.
                identity = torch.eye(config.dim)
                self.layer1.weight.copy_(torch.cat([identity, -identity], dim=0))
                self.layer2.weight.copy_(torch.cat([identity, -identity], dim=1))
            # else: an explicit non-default hidden_dim doesn't support the
            # exact-identity trick - left at nn.Linear's own random init.

    @classmethod
    def from_legacy_linear(cls, legacy_weight: Tensor, *, dim: int) -> "SonicKeyProjection":
        """Builds a new Linear->ReLU->Linear instance that reproduces an
        OLD single-Linear key_projection's exact *learned* function at
        construction time - not just the identity, the real trained
        transformation - via the same relu(x)-relu(-x)=x identity applied
        to `legacy_weight` (the old `projection.weight`, shape [dim,
        dim]) instead of the identity matrix. See module docstring."""
        if legacy_weight.shape != (dim, dim):
            raise ValueError(f"legacy_weight must have shape [{dim}, {dim}]")
        instance = cls(SonicKeyProjectionConfig(dim=dim, hidden_dim=2 * dim))
        with torch.no_grad():
            identity = torch.eye(dim)
            instance.layer1.weight.copy_(torch.cat([legacy_weight, -legacy_weight], dim=0))
            instance.layer2.weight.copy_(torch.cat([identity, -identity], dim=1))
        return instance

    def forward(self, pooled: Tensor) -> Tensor:
        """`pooled`: `[batch, dim]`, mean-pooled (not yet normalized)
        hidden states. Returns an L2-normalized key of the same shape."""
        if pooled.ndim != 2 or pooled.shape[-1] != self.config.dim:
            raise ValueError("pooled must have shape [batch, dim]")
        hidden = F.relu(self.layer1(pooled))
        return F.normalize(self.layer2(hidden), dim=-1)


def load_key_projection_state_dict(key_projection: SonicKeyProjection, state_dict: dict) -> None:
    """Loads a saved key_projection state dict, migrating a pre-upgrade
    single-Linear checkpoint (state_dict has "projection.weight", the old
    schema) into the current Linear->ReLU->Linear instance losslessly via
    `SonicKeyProjection.from_legacy_linear` - reused across every call
    site that resumes from a checkpoint (the production trainer and every
    scripts/*.py that loads one), so this migration logic exists exactly
    once. A state dict already in the current schema ("layer1.weight"/
    "layer2.weight") loads directly, no migration needed."""
    if "projection.weight" in state_dict:
        migrated = SonicKeyProjection.from_legacy_linear(
            state_dict["projection.weight"], dim=key_projection.config.dim
        )
        key_projection.load_state_dict(migrated.state_dict())
    else:
        key_projection.load_state_dict(state_dict)


def info_nce_retrieval_loss(
    query_key: Tensor, candidate_keys: Tensor, correct_index: int, *, temperature: float
) -> Tensor:
    """Per-episode InfoNCE loss: `query_key` (L2-normalized, `[dim]`)
    should score higher (dot product) against
    `candidate_keys[correct_index]` than against every other row of
    `candidate_keys` (`[num_candidates, dim]`, also L2-normalized) - the
    other facts written into the *same* episode's bank, which is exactly
    the negative set `BasinMemory.read()` actually has to discriminate
    against at inference. One episode, one query, `num_candidates`-way
    classification - not a cross-episode in-batch-negatives scheme, since
    each episode's candidate set is unrelated to every other episode's."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= correct_index < candidate_keys.shape[0]:
        raise ValueError("correct_index must index an existing candidate")
    logits = (candidate_keys @ query_key) / temperature
    target = torch.tensor(correct_index, device=logits.device)
    return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
