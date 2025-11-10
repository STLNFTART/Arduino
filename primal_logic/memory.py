"""Memory kernel models for Primal Logic controllers."""

from __future__ import annotations

import math

from .constants import DT, LAMBDA_DEFAULT


class ExponentialMemoryKernel:
    """Single state exponential memory kernel."""

    def __init__(self, lam: float = LAMBDA_DEFAULT, gain: float = 1.0) -> None:
        self.lam = lam
        self.gain = gain
        self._memory = 0.0

    def update(self, theta: float, error: float) -> float:
        """Update the memory state using exponential decay."""
        decay = math.exp(-self.lam * DT)
        self._memory = decay * self._memory + theta * error * DT
        return float(-self.gain * self._memory)

    @property
    def state(self) -> float:
        """Expose the current memory state for diagnostics."""
        return self._memory
