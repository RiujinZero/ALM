"""Streaming trainer for a multi-column PC-ALM hierarchy with one column's
local token head wired into settling (section 7's `L_token` term -- see
`learning/pc_alm.py`'s module docstring and `PlasticColumn.token_gradient_wrt_h`
for the mechanism).

This is the M2-with-a-head counterpart to `train/stream_trainer.py`'s
single-column M1 loop: same per-token update order, but activity settles
jointly across all columns (with the head column's own classification
pressure reaching earlier columns through PC-ALM's neighbor-exchange
term), and every column gets its own local reconstruction + homeostasis
update after settling.
"""

from __future__ import annotations

import numpy as np
import torch

from manifoldflow.alm.learning.pc_alm import PCALMHierarchy
from manifoldflow.alm.train.stream_trainer import TrainStats


def train_pc_alm_hierarchy(
    hierarchy: PCALMHierarchy,
    token_stream: np.ndarray,
    eos_id: int,
    head_layer: int = -1,
    embed_layer: int = 0,
    log_every: int = 2000,
    start_index: int = 0,
    stats: TrainStats | None = None,
    h_prev: list[torch.Tensor] | None = None,
) -> tuple[TrainStats, list[torch.Tensor]]:
    """Train every column in `hierarchy` on `token_stream`.

    `head_layer` is the column whose local token head is trained and
    scored (default: the last/top column); `embed_layer` is the column
    whose `E_token` supplies the input embedding (default: the first/
    bottom column, mirroring `stream_trainer.embed`'s single-column
    convention). Returns (stats, h_prev) for resuming, matching
    `train_single_column`'s interface.
    """
    n_layers = hierarchy.n_layers
    head_layer = head_layer % n_layers
    head_column = hierarchy.columns[head_layer]
    embed_column = hierarchy.columns[embed_layer]

    stats = stats if stats is not None else TrainStats()
    h_prev = (
        h_prev
        if h_prev is not None
        else [torch.zeros(col.config.state_dim) for col in hierarchy.columns]
    )
    window_ce: list[float] = []
    window_rec: list[float] = []

    n = len(token_stream) - 1
    for t in range(start_index, n):
        token_t = int(token_stream[t])
        token_next = int(token_stream[t + 1])
        e_t = embed_column.E_token[token_t]

        def extra_pressure_fn(h_list, _token_next=token_next):
            extra = [torch.zeros_like(h_l) for h_l in h_list]
            delta = head_column.token_delta(h_list[head_layer], _token_next)
            extra[head_layer] = head_column.token_gradient_wrt_h(delta)
            return extra

        result = hierarchy.settle(e_t, h_prev, extra_pressure_fn=extra_pressure_fn)

        with torch.no_grad():
            p = head_column.token_probs(result.h[head_layer])
            ce = float(-torch.log(torch.clamp_min(p[token_next], 1e-12)))

        final_delta = head_column.token_delta(result.h[head_layer], token_next)
        final_extra = [torch.zeros_like(h_l) for h_l in result.h]
        final_extra[head_layer] = head_column.token_gradient_wrt_h(final_delta)
        hierarchy.update_weights(e_t, result, h_prev, extra_pressure=final_extra)
        head_column.update_token_head(result.h[head_layer], token_next)

        e_rec_sum = 0.0
        for layer, col in enumerate(hierarchy.columns):
            h_in = hierarchy._layer_input(layer, e_t, result.h)
            e_rec = col.update_reconstruction(h_in, result.h[layer])
            col.update_homeostasis(result.h[layer])
            e_rec_sum += float(e_rec.norm())

        stats.tokens_seen += 1
        stats.token_ce_sum += ce
        stats.reconstruction_error_sum += e_rec_sum
        window_ce.append(ce)
        window_rec.append(e_rec_sum)

        h_prev = (
            [torch.zeros(col.config.state_dim) for col in hierarchy.columns]
            if token_t == eos_id
            else result.h
        )

        if stats.tokens_seen % log_every == 0:
            stats.log.append(
                {
                    "tokens_seen": stats.tokens_seen,
                    "token_ce": float(np.mean(window_ce)),
                    "reconstruction_error": float(np.mean(window_rec)),
                    "activation_density": head_column.activation_density(),
                    "max_unit_share": head_column.max_unit_share(),
                }
            )
            window_ce.clear()
            window_rec.clear()

    return stats, h_prev
