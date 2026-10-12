"""The first-entity binding task (design doc section 21 of
`Manifold_Dendritic_Stream_Learner_Design_v0.2.md`; the binding-problem
smoke of `Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`).

  alice meets bob sees cat .
  bob meets alice sees dog .
  carol meets dave sees bird .

The object belongs to the *first* entity, not the more recent one --
simple recency-based temporal traces are expected to confuse the two
(the design's own reported ceiling for traces alone: 62.2% object
accuracy), which is exactly what a persistent (not just decaying)
binding mechanism is for.
"""

from __future__ import annotations

import numpy as np

VOCAB = [
    "<bos>", "alice", "bob", "carol", "dave", "meets", "sees", "cat", "dog", "bird",
    "fish", ".",
]
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

NAME_TO_OBJECT = {"alice": "cat", "bob": "dog", "carol": "bird", "dave": "fish"}
NAMES = list(NAME_TO_OBJECT)


def generate_token_stream(n_sentences: int, rng: np.random.Generator) -> list[int]:
    """`<bos> SUBJECT meets DISTRACTOR sees OBJECT .`, where OBJECT is
    determined by SUBJECT (the first name), and DISTRACTOR is a
    different random name.
    """
    tokens: list[str] = []
    for _ in range(n_sentences):
        subject = NAMES[int(rng.integers(0, len(NAMES)))]
        remaining = [n for n in NAMES if n != subject]
        distractor = remaining[int(rng.integers(0, len(remaining)))]
        obj = NAME_TO_OBJECT[subject]
        tokens.extend(["<bos>", subject, "meets", distractor, "sees", obj, "."])
    return [TOKEN_TO_ID[t] for t in tokens]
