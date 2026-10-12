"""Short-lived local commitment for a recalled cortical trajectory."""

from __future__ import annotations


class TransientRecallCommitment:
    """A decaying facilitation trace that bounds episodic influence.

    Recruitment must be grounded in external activity.  Internally recalled or
    generated activity may consume the trace, but cannot refresh it.  Assembly
    terminals decide when to inspect ``active``; the trace itself knows nothing
    about symbols, sequence lengths, or semantic boundaries.
    """

    def __init__(
        self,
        *,
        half_life_steps: float = 5.0,
        extinction: float = 0.30,
    ) -> None:
        if half_life_steps <= 0.0:
            raise ValueError("commitment half-life must be positive")
        if not 0.0 < extinction < 1.0:
            raise ValueError("commitment extinction must be in (0, 1)")
        self.half_life_steps = float(half_life_steps)
        self.extinction = float(extinction)
        self.level = 0.0
        self._decay = 0.5 ** (1.0 / self.half_life_steps)

    @property
    def active(self) -> bool:
        return self.level > self.extinction

    def recruit(self, strength: float = 1.0, *, external: bool) -> bool:
        """Recruit or strengthen the trace only from grounded activity."""
        if not external or strength <= 0.0:
            return False
        self.level = max(self.level, min(1.0, float(strength)))
        return True

    def advance(self, steps: int = 1) -> None:
        """Consume facilitation as the organism's trajectory advances."""
        if steps < 0:
            raise ValueError("commitment steps must be non-negative")
        self.level *= self._decay**steps
        if self.level <= 1e-12:
            self.level = 0.0

    def reset(self) -> None:
        self.level = 0.0
