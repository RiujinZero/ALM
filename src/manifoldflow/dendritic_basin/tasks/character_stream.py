"""Raw character encoding of the ambiguous-object task (design doc
section 54 Phase 4: "Raw character stream"; the character-level smoke of
`Dendritic_Basin_Learning_Design_v0.4_Character_Stream_Findings.md`).

No word IDs, no tokenizer: every character (including the space that
separates words) is its own event channel. A sentence like `alice sees
cat .` becomes the character sequence `a l i c e _ s e e s _ c a t _ .`
(`_` denoting the literal space character, itself a channel).
"""

from __future__ import annotations

import numpy as np

SPACE = "_"  # a printable stand-in for the literal space *character* channel
NAME_TO_OBJECT = {"alice": "cat", "bob": "dog", "carol": "bird", "dave": "fish"}
NAMES = list(NAME_TO_OBJECT)

_ALL_WORDS = [*NAMES, *NAME_TO_OBJECT.values(), "sees", "."]
ALPHABET = sorted({c for word in _ALL_WORDS for c in word})
VOCAB = ["<bos>", SPACE, *ALPHABET]
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def _word_to_chars(word: str) -> list[str]:
    return list(word)


def generate_char_stream(n_sentences: int, rng: np.random.Generator) -> list[int]:
    """`<bos> NAME sees OBJECT . <bos> NAME sees OBJECT . ...`, exploded to
    individual character tokens (with `SPACE` between words).
    """
    tokens: list[str] = []
    for _ in range(n_sentences):
        name = NAMES[int(rng.integers(0, len(NAMES)))]
        obj = NAME_TO_OBJECT[name]
        words = [name, "sees", obj, "."]
        tokens.append("<bos>")
        for i, word in enumerate(words):
            if i > 0:
                tokens.append(SPACE)
            tokens.extend(_word_to_chars(word))
    return [TOKEN_TO_ID[t] for t in tokens]
