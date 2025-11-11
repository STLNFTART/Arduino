r"""Recursive Planck operator implementation.

The operator follows the discrete update law described in the Quantro–Primal
formalism. It bridges energetic and informational domains using Donte's and
Lightfoot's constants while ensuring bounded responses for admissible inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    DONTE_CONSTANT,
    DT,
    LAMBDA_DEFAULT,
    LIGHTFOOT_MAX,
    LIGHTFOOT_MIN,
    PLANCK_CONSTANT,
)


@dataclass
class RecursivePlanckOperator:
    r"""Recursive Planck Operator (RPO) with stability guards.

    Parameters
    ----------
    alpha : float
        Damping coefficient from the Volterra kernel. Must satisfy
        ``0 < alpha * DT < 1``.
    lam : float, optional
        Base decay rate ``lambda`` governing the exponential kernel.
    lightfoot : float, optional
        Lightfoot's constant :math:`\mathcal{L}` that blends neural and
        mechanical domains. The value is clipped to the admissible interval
        ``[LIGHTFOOT_MIN, LIGHTFOOT_MAX]``.
    donte : float, optional
        Donte's constant :math:`\mathcal{D}` scaling the effective Planck
        constant used for resonance.
    planck : float, optional
        Physical Planck constant ``h``. Defaults to CODATA 2019 value.
    dt : float, optional
        Integration step in seconds.
    state : float, optional
        Initial value of the recursive memory state ``y``.
    """

    alpha: float
    lam: float = LAMBDA_DEFAULT
    lightfoot: float = (LIGHTFOOT_MIN + LIGHTFOOT_MAX) / 2.0
    donte: float = DONTE_CONSTANT
    planck: float = PLANCK_CONSTANT
    dt: float = DT
    state: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (0 < self.alpha * self.dt < 1):
            raise ValueError("alpha * dt must lie in (0, 1) for bounded dynamics")
        if self.donte <= 0:
            raise ValueError("Donte constant must be positive")
        if self.planck <= 0:
            raise ValueError("Planck constant must be positive")

        # Clamp Lightfoot's constant to the admissible range for safety.
        self.lightfoot = min(max(self.lightfoot, LIGHTFOOT_MIN), LIGHTFOOT_MAX)

        # Effective Planck constant scaling the resonance term.
        self._h_eff = self.planck / self.donte
        # β_P term from the specification.
        self._beta_p = self.lightfoot / (1.0 + self.lam)

    @property
    def h_eff(self) -> float:
        """Return the effective Planck constant used inside the operator."""

        return self._h_eff

    @property
    def beta_p(self) -> float:
        """Return the bounded recursive coupling coefficient β_P."""

        return self._beta_p

    def step(self, theta: float, input_value: float, step_index: int) -> float:
        """Advance the operator by one step.

        Parameters
        ----------
        theta : float
            Command envelope ``Θ_k``.
        input_value : float
            Instantaneous input ``f_k``.
        step_index : int
            Discrete time index used in the resonance term.
        """

        sin_arg = 2.0 * math.pi * step_index * self.dt / self._h_eff
        resonance = math.sin(sin_arg) * self.state

        # Discrete update derived from the provided specification.
        self.state = (
            (1.0 - self.alpha * self.dt) * self.state
            + theta * self.dt * (input_value + self._beta_p * resonance)
        )
        return self.state

    def reset(self) -> None:
        """Reset the recursive state to zero."""

        self.state = 0.0


class RecursivePlanckMemoryKernel:
    """Wrapper exposing the RPO through the memory-kernel interface."""

    def __init__(
        self,
        alpha: float,
        lam: float = LAMBDA_DEFAULT,
        lightfoot: float = (LIGHTFOOT_MIN + LIGHTFOOT_MAX) / 2.0,
        donte: float = DONTE_CONSTANT,
        planck: float = PLANCK_CONSTANT,
        dt: float = DT,
    ) -> None:
        self._operator = RecursivePlanckOperator(
            alpha=alpha,
            lam=lam,
            lightfoot=lightfoot,
            donte=donte,
            planck=planck,
            dt=dt,
        )

    def update(self, theta: float, error: float, step_index: int | None = None) -> float:
        """Return the recursive Planck feedback term.

        The update requires a ``step_index`` to evaluate the resonance term.
        """

        if step_index is None:
            raise ValueError("step_index must be provided for RecursivePlanckMemoryKernel")
        return self._operator.step(theta=theta, input_value=error, step_index=step_index)

    @property
    def operator(self) -> RecursivePlanckOperator:
        """Expose the underlying operator for diagnostics/tests."""

        return self._operator


__all__ = ["RecursivePlanckOperator", "RecursivePlanckMemoryKernel"]
