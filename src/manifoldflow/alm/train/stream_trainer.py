"""Streaming trainer for the M1 "recommended first experiment" (design doc
section 23): one plastic column plus its local token head, trained on a
token stream using only local reconstruction and token-head updates -- no
PC-ALM dual state, no temporal predictor, no MF bank yet (those are
layered on separately; see learning/pc_alm.py and learning/eligibility.py,
and the implementation journal for what M4+ still needs).

Section 13.1's update order, restricted to what M1 needs:
  1. Read token x_t; forward pass through the column using the shared
     token codebook as the input embedding (section 16's parameter budget
     ties the codebook to the column's own E_token: "Six reciprocal column
     maps: 768x384 then five 768x768" -- layer 0 takes a 384-dim token
     embedding, which is exactly this column's token_dim).
  2. When x_{t+1} arrives, update the local reconstruction and token head
     from local errors only.
  3. Update homeostasis; advance transient state.
  4. At EOS, reset the transient activity but keep every weight.

Training is checkpointable: `train_single_column` accepts a resumed
`TrainStats`/`start_index`/`h_prev` and can periodically write a checkpoint
via `save_checkpoint`, which `load_checkpoint`/`resume_or_init` read back.
Resume is not bit-exact with an uninterrupted run (dead-unit rescue in
`PlasticColumn._rescue_dead_units` draws from the untracked global RNG),
but it continues the same learning trajectory from the same weights,
homeostasis state, and stream position.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from manifoldflow.alm.eval.metrics import cross_entropy_reduction, unigram_cross_entropy
from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig


@dataclass
class TrainStats:
    """Running + logged metrics for a single-column training run."""

    tokens_seen: int = 0
    token_ce_sum: float = 0.0
    reconstruction_error_sum: float = 0.0
    log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tokens_seen": self.tokens_seen,
            "token_ce_sum": self.token_ce_sum,
            "reconstruction_error_sum": self.reconstruction_error_sum,
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainStats:
        return cls(
            tokens_seen=data["tokens_seen"],
            token_ce_sum=data["token_ce_sum"],
            reconstruction_error_sum=data["reconstruction_error_sum"],
            log=list(data["log"]),
        )


def embed(column: PlasticColumn, token_id: int) -> torch.Tensor:
    """Look up the token embedding from the column's own token codebook
    (E_token, shape (vocab, token_dim), doubles as the shared codebook for
    this single-column setup).
    """
    return column.E_token[token_id]


# -- checkpointing ------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    column: PlasticColumn,
    stats: TrainStats,
    position: int,
    h_prev: torch.Tensor,
) -> None:
    """Write a resumable checkpoint: column weights/homeostasis state,
    running training stats, the stream index to resume from, and the
    transient activity carried into the next token.
    """
    tmp_path = Path(str(path) + ".tmp")
    torch.save(
        {
            "column": column.state_dict(),
            "config": asdict(column.config),
            "stats": stats.to_dict(),
            "position": position,
            "h_prev": h_prev.clone(),
        },
        tmp_path,
    )
    tmp_path.replace(path)  # atomic on POSIX and Windows NTFS alike


def load_checkpoint(path: str | Path) -> dict:
    """Raw checkpoint dict as written by `save_checkpoint` (trusted local file)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def resume_or_init(
    path: str | Path,
    config: PlasticColumnConfig,
    generator: torch.Generator | None = None,
) -> tuple[PlasticColumn, TrainStats, int, torch.Tensor]:
    """Load a checkpoint at `path` if present, else initialize fresh.

    Returns (column, stats, start_index, h_prev). Raises if a found
    checkpoint's config doesn't match `config` (mismatched dims would
    silently corrupt weights on load).
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        column = PlasticColumn(config, generator=generator)
        return column, TrainStats(), 0, torch.zeros(config.state_dim)

    checkpoint = load_checkpoint(checkpoint_path)
    saved_config = checkpoint["config"]
    if saved_config != asdict(config):
        raise ValueError(
            f"checkpoint config {saved_config} does not match requested config "
            f"{asdict(config)}; use a different checkpoint path or matching config"
        )
    column = PlasticColumn(config, generator=generator)
    column.load_state_dict(checkpoint["column"])
    stats = TrainStats.from_dict(checkpoint["stats"])
    return column, stats, checkpoint["position"], checkpoint["h_prev"]


# -- training -------------------------------------------------------------


def train_single_column(
    column: PlasticColumn,
    token_stream: np.ndarray,
    eos_id: int,
    log_every: int = 2000,
    start_index: int = 0,
    stats: TrainStats | None = None,
    h_prev: torch.Tensor | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
) -> tuple[TrainStats, torch.Tensor]:
    """Train one plastic column + local token head over `token_stream`.

    `token_stream` is a 1-D array of token IDs (as written by
    `data/tinystories.py`'s shards, story-concatenated with BOS/EOS).
    Returns (stats, h_prev) so a caller can resume from exactly where this
    call left off (see `resume_or_init`/`save_checkpoint`).

    Pass `start_index`/`stats`/`h_prev` (all three, from a prior call or a
    loaded checkpoint) to continue an interrupted run instead of starting
    over. If `checkpoint_path` and `checkpoint_every` (in tokens) are both
    set, a checkpoint is written periodically during the run.
    """
    stats = stats if stats is not None else TrainStats()
    h_prev = h_prev if h_prev is not None else torch.zeros(column.config.state_dim)
    window_ce: list[float] = []
    window_rec: list[float] = []

    n = len(token_stream) - 1
    for t in range(start_index, n):
        token_t = int(token_stream[t])
        token_next = int(token_stream[t + 1])

        e_t = embed(column, token_t)
        h = column.forward(e_t, h_prev)

        e_rec = column.update_reconstruction(e_t, h)
        with torch.no_grad():
            p = column.token_probs(h)
            ce = float(-torch.log(torch.clamp_min(p[token_next], 1e-12)))
        column.update_token_head(h, token_next)
        column.update_homeostasis(h)

        stats.tokens_seen += 1
        stats.token_ce_sum += ce
        stats.reconstruction_error_sum += float(e_rec.norm())
        window_ce.append(ce)
        window_rec.append(float(e_rec.norm()))

        h_prev = torch.zeros(column.config.state_dim) if token_t == eos_id else h

        if stats.tokens_seen % log_every == 0:
            stats.log.append(
                {
                    "tokens_seen": stats.tokens_seen,
                    "token_ce": float(np.mean(window_ce)),
                    "reconstruction_error": float(np.mean(window_rec)),
                    "activation_density": column.activation_density(),
                    "max_unit_share": column.max_unit_share(),
                    "effective_rank": column.effective_rank(),
                    "effective_rank_fraction": (
                        column.effective_rank() / column.config.state_dim
                    ),
                }
            )
            window_ce.clear()
            window_rec.clear()

        if (
            checkpoint_path is not None
            and checkpoint_every
            and stats.tokens_seen % checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoint_path, column, stats, position=t + 1, h_prev=h_prev
            )

    return stats, h_prev


def sanity_gate_report(
    stats: TrainStats, token_stream: np.ndarray, vocab_size: int
) -> dict:
    """Is it learning at all, plus the readouts to sanity-check *how*.

    No pass/fail threshold: "is_learning" is simply whether the logged
    per-window token cross-entropy trended down over the run (first logged
    window vs. last). `cross_entropy_reduction` against the unigram
    baseline is reported alongside as context, not as a gate (see
    docs/ManifoldFlow_ALM_Journal.md for why the earlier fixed 10% gate
    was dropped).
    """
    model_ce = stats.token_ce_sum / max(stats.tokens_seen, 1)
    unigram_ce = unigram_cross_entropy(token_stream, vocab_size)
    reduction = cross_entropy_reduction(model_ce, unigram_ce)

    first = stats.log[0] if stats.log else {}
    last = stats.log[-1] if stats.log else {}
    first_ce = first.get("token_ce")
    last_ce = last.get("token_ce")
    is_learning = first_ce is not None and last_ce is not None and last_ce < first_ce
    loss_trend_ratio = (last_ce / first_ce) if first_ce else None

    return {
        "model_token_ce": model_ce,
        "unigram_token_ce": unigram_ce,
        "cross_entropy_reduction": reduction,
        "first_window_token_ce": first_ce,
        "last_window_token_ce": last_ce,
        "loss_trend_ratio": loss_trend_ratio,
        "is_learning": is_learning,
        "final_activation_density": last.get("activation_density"),
        "final_max_unit_share": last.get("max_unit_share"),
        "final_effective_rank_fraction": last.get("effective_rank_fraction"),
    }
