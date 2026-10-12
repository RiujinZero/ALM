"""Persistent (non-decaying) relation assemblies (design doc sections
21-22 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`; the binding
fix in `Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`
section 6).

A decaying temporal trace answers "did this happen recently?" -- it
cannot answer "which earlier entity does this later event still refer
to?" once other tokens intervene, because recency and identity are not
the same thing (section 15/21: "temporal context is not equivalent to
binding memory").

This tracks adjacency events -- "token j immediately followed token i"
-- as a *sticky* flag that survives any number of intervening tokens
without decaying, and is cleared only at an episode boundary (e.g. BOS).
For a sentence template like `<bos> SUBJECT meets DISTRACTOR sees OBJECT
.`, the assembly `(BOS, SUBJECT)` is true only when SUBJECT immediately
follows BOS -- true for the subject, never for the distractor (which
follows "meets") -- giving a signal that tracks *role*, not just
*recency*.
"""

from __future__ import annotations

import numpy as np


class PersistentBigramAssemblies:
    """`active[i, j]` = "j immediately followed i, at some point since the
    last occurrence of `reset_channel`" -- sticky, not decaying.
    """

    def __init__(self, vocab_size: int, reset_channel: int):
        self.vocab_size = vocab_size
        self.reset_channel = reset_channel
        self.active = np.zeros((vocab_size, vocab_size), dtype=bool)
        self._prev_token: int | None = None

    def update(self, token_id: int) -> None:
        if self._prev_token is not None:
            self.active[self._prev_token, token_id] = True
        if token_id == self.reset_channel:
            self.active[:, :] = False
        self._prev_token = token_id

    def is_active(self, i: int, j: int) -> bool:
        return bool(self.active[i, j])

    def reset(self) -> None:
        self.active[:, :] = False
        self._prev_token = None


class PersistentEventFlags:
    """Generalization of `PersistentBigramAssemblies` to arbitrary named
    0/1 event streams (not just "token j followed token i"): each named
    event's flag goes sticky-true the first time it fires and stays true
    until an explicit reset, regardless of how many steps pass in
    between. Used to track word-completion events (see
    `word_completion.py`) across intervening characters at Phase 4's
    character granularity, the same way `PersistentBigramAssemblies`
    tracks token adjacency at word granularity.
    """

    def __init__(self, event_names: list[str]):
        self.event_names = list(event_names)
        self.active: dict[str, bool] = dict.fromkeys(self.event_names, False)

    def update(self, events: dict[str, float], reset: bool) -> None:
        if reset:
            self.active = dict.fromkeys(self.event_names, False)
        for name in self.event_names:
            if events.get(name, 0.0) > 0.5:
                self.active[name] = True

    def is_active(self, name: str) -> bool:
        return self.active[name]

    def snapshot(self) -> dict[str, float]:
        return {name: (1.0 if value else 0.0) for name, value in self.active.items()}
