"""Word-completion event detection over a raw character stream (design
doc section 54 Phase 4: "Raw character stream").

Detects, at each position, whether the trailing window of characters
exactly matches a known candidate character sequence -- an instantaneous
event ("this word just finished being spelled"), not a persistent flag.
Feeding these events into the already-validated `PersistentBigramAssemblies`
/ `assembly_discovery.py` pipeline from Phase 3 (keyed on a reset marker
like `<bos>`) reuses that exact mechanism at character granularity,
without needing a trained tokenizer or hand-given word/target pairings --
only a candidate list of character sequences to check for completion.

Full unsupervised discovery of *which* substrings are worth checking
(matching arbitrary, un-listed recurring sequences) is not attempted
here -- see the implementation journal for why that remains open.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class WordCompletionDetector:
    """For each candidate word (a sequence of channel ids), emits 1.0 at
    the exact position where the trailing window of observed channels
    matches that word, else 0.0.
    """

    def __init__(self, candidate_words: dict[str, list[int]]):
        self.candidate_words = candidate_words
        self._max_len = max((len(seq) for seq in candidate_words.values()), default=0)
        self._window: deque[int] = deque(maxlen=self._max_len)

    def update(self, channel_id: int) -> dict[str, float]:
        self._window.append(channel_id)
        completions = {}
        window_list = list(self._window)
        for name, sequence in self.candidate_words.items():
            k = len(sequence)
            completions[name] = 1.0 if window_list[-k:] == sequence else 0.0
        return completions

    def reset(self) -> None:
        self._window.clear()


def build_completion_event_stream(
    token_stream: list[int], candidate_words: dict[str, list[int]]
) -> np.ndarray:
    """A (len(token_stream), len(candidate_words)) matrix of completion
    events -- row t, column k is 1.0 iff `candidate_words[k]` finished
    being spelled exactly at position t.
    """
    detector = WordCompletionDetector(candidate_words)
    names = list(candidate_words)
    events = np.zeros((len(token_stream), len(names)))
    for t, token_id in enumerate(token_stream):
        completions = detector.update(token_id)
        for i, name in enumerate(names):
            events[t, i] = completions[name]
    return events
