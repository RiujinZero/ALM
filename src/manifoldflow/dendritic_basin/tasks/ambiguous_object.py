"""The ambiguous-object language task (design doc section 28 of
`Manifold_Dendritic_Stream_Learner_Design_v0.2.md`; Smoke 1 of
`Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`).

  <BOS> alice sees cat .
  <BOS> bob sees dog .
  <BOS> carol sees bird .
  <BOS> dave sees fish .

P(object | sees) = 1/4: no local (one-token) signal can predict the
object. Predicting it correctly requires associating "sees" with
*whichever name occurred earlier in this sentence* -- exactly the
temporal-relation-discovery test.
"""

from __future__ import annotations

import numpy as np

VOCAB = [
    "<bos>", "alice", "bob", "carol", "dave", "sees", "cat", "dog", "bird", "fish", "."
]
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

NAME_TO_OBJECT = {"alice": "cat", "bob": "dog", "carol": "bird", "dave": "fish"}
NAMES = list(NAME_TO_OBJECT)


def generate_token_stream(n_sentences: int, rng: np.random.Generator) -> list[int]:
    """A continuous stream of `n_sentences` random `<bos> NAME sees OBJECT
    .` sentences, as token IDs.
    """
    tokens: list[str] = []
    for _ in range(n_sentences):
        name = NAMES[int(rng.integers(0, len(NAMES)))]
        obj = NAME_TO_OBJECT[name]
        tokens.extend(["<bos>", name, "sees", obj, "."])
    return [TOKEN_TO_ID[t] for t in tokens]


def one_hot(token_id: int) -> np.ndarray:
    vec = np.zeros(VOCAB_SIZE)
    vec[token_id] = 1.0
    return vec
