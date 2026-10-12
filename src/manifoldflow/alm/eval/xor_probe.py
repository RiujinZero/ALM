"""Temporal-XOR probe: does the local learning rule solve a task that is
not linearly separable in the raw input?

A pure single-layer Hebbian/delta-rule read-out (which is exactly what
`PlasticColumn.update_token_head` is) can only learn a linear decision
boundary -- the classic Minsky & Papert critique of single-layer local
learning. This probe builds the smallest sequential analogue of XOR: two
binary tokens `bit_a`, `bit_b` presented one after another, followed by a
label token equal to `bit_a XOR bit_b`. No single raw token or fixed
linear combination of the two embeddings can predict the label -- doing
so requires the column's recurrent, nonlinear (tanh + kWTA) state to
combine both bits into something the linear token head can then read off.

Two variants:
  - `with_separator=False`: bit_a and bit_b are adjacent, so only a
    one-step recurrent carry (through the column's self-recurrent `A`) is
    needed to combine them.
  - `with_separator=True`: a distractor SEP token sits between bit_a and
    bit_b, so bit_a's influence must survive an extra recurrent step
    before being combined with bit_b -- a real (if short) memory
    requirement, not just instantaneous combination.

Vocabulary (fixed, 6 symbols): SEP=0, ZERO=1, ONE=2, LABEL0=3, LABEL1=4,
EOS=5. This is independent of the TinyStories BPE/byte tokenizers -- a
synthetic vocabulary sized for this probe alone.
"""

from __future__ import annotations

import numpy as np
import torch

from manifoldflow.alm.learning.pc_alm import PCALMHierarchy
from manifoldflow.alm.model.plastic_column import PlasticColumn

SEP, ZERO, ONE, LABEL0, LABEL1, EOS = 0, 1, 2, 3, 4, 5
VOCAB_SIZE = 6


def build_xor_stream(
    n_examples: int, seed: int, with_separator: bool = False
) -> tuple[np.ndarray, list[int]]:
    """Build a token stream of `n_examples` iid-random XOR examples.

    Returns (tokens, label_positions) where `label_positions[k]` is the
    index of the *last bit token* of example k -- the model has just seen
    both bits and is about to predict `tokens[label_positions[k] + 1]`,
    the label. This is the one prediction per example that is genuinely
    hard; every other transition in the stream is either deterministic
    (bit -> SEP, label -> EOS) or fundamentally unpredictable noise
    (EOS/SEP -> the next random bit), so only these positions should be
    scored.
    """
    rng = np.random.default_rng(seed)
    tokens: list[int] = []
    label_positions: list[int] = []
    for _ in range(n_examples):
        a = int(rng.integers(0, 2))
        b = int(rng.integers(0, 2))
        label = LABEL1 if (a ^ b) else LABEL0
        tokens.append(ONE if a else ZERO)
        if with_separator:
            tokens.append(SEP)
        tokens.append(ONE if b else ZERO)
        label_positions.append(len(tokens) - 1)
        tokens.append(label)
        tokens.append(EOS)
    return np.asarray(tokens, dtype=np.int64), label_positions


def evaluate_at_label_positions(
    column: PlasticColumn,
    token_stream: np.ndarray,
    label_positions: list[int],
    eos_id: int = EOS,
) -> dict:
    """Forward-only pass (no weight updates) scoring accuracy/CE only at
    `label_positions` -- i.e. only on the genuinely hard XOR predictions.
    """
    h_prev = torch.zeros(column.config.state_dim)
    label_set = set(label_positions)
    correct = 0
    ce_sum = 0.0
    n_scored = 0
    with torch.no_grad():
        for t in range(len(token_stream) - 1):
            token_t = int(token_stream[t])
            token_next = int(token_stream[t + 1])
            e_t = column.E_token[token_t]
            h = column.forward(e_t, h_prev)
            if t in label_set:
                p = column.token_probs(h)
                predicted = int(torch.argmax(p))
                correct += int(predicted == token_next)
                ce_sum += float(-torch.log(torch.clamp_min(p[token_next], 1e-12)))
                n_scored += 1
            h_prev = torch.zeros(column.config.state_dim) if token_t == eos_id else h
    return {
        "n_scored": n_scored,
        "accuracy": correct / max(n_scored, 1),
        "mean_ce": ce_sum / max(n_scored, 1),
    }


def evaluate_hierarchy_at_label_positions(
    hierarchy: PCALMHierarchy,
    token_stream: np.ndarray,
    label_positions: list[int],
    head_layer: int = -1,
    embed_layer: int = 0,
    eos_id: int = EOS,
) -> dict:
    """Multi-column counterpart to `evaluate_at_label_positions`: settles
    the hierarchy at each step (no weight/activity updates persist -- each
    step's settle() call is local, and no `extra_pressure_fn` is passed
    since the true label must not be revealed at evaluation time) and
    scores the head column's prediction only at `label_positions`.
    """
    n_layers = hierarchy.n_layers
    head_layer = head_layer % n_layers
    head_column = hierarchy.columns[head_layer]
    embed_column = hierarchy.columns[embed_layer]

    h_prev = [torch.zeros(col.config.state_dim) for col in hierarchy.columns]
    label_set = set(label_positions)
    correct = 0
    ce_sum = 0.0
    n_scored = 0
    with torch.no_grad():
        for t in range(len(token_stream) - 1):
            token_t = int(token_stream[t])
            token_next = int(token_stream[t + 1])
            e_t = embed_column.E_token[token_t]
            result = hierarchy.settle(e_t, h_prev)
            if t in label_set:
                p = head_column.token_probs(result.h[head_layer])
                predicted = int(torch.argmax(p))
                correct += int(predicted == token_next)
                ce_sum += float(-torch.log(torch.clamp_min(p[token_next], 1e-12)))
                n_scored += 1
            h_prev = (
                [torch.zeros(col.config.state_dim) for col in hierarchy.columns]
                if token_t == eos_id
                else result.h
            )
    return {
        "n_scored": n_scored,
        "accuracy": correct / max(n_scored, 1),
        "mean_ce": ce_sum / max(n_scored, 1),
    }
