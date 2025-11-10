"""Quantum-inspired field dynamics for the Primal Logic simulator."""

from __future__ import annotations

import numpy as np

from .constants import ALPHA_DEFAULT, DT, EPSILON, GAMMA, K_COUPLING, LAMBDA_DEFAULT
from .utils import laplacian_2d, safe_clip


class PrimalLogicField:
    """Evolve a complex field that modulates the robotic hand controller."""

    def __init__(
        self,
        nx: int = 8,
        ny: int = 8,
        alpha: float = ALPHA_DEFAULT,
        lam: float = LAMBDA_DEFAULT,
        coupling: float = K_COUPLING,
        gamma: float = GAMMA,
        epsilon: float = EPSILON,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.lam = lam
        self.coupling = coupling
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize real and imaginary components of the field to zero.
        self.psi_r = np.zeros((nx, ny), dtype=float)
        self.psi_i = np.zeros((nx, ny), dtype=float)
        self.gamma_field = np.zeros((nx, ny), dtype=float)

    def step(self, theta: float) -> float:
        """Advance the field by one integration step and return coherence."""
        lap_r = laplacian_2d(self.psi_r)
        lap_i = laplacian_2d(self.psi_i)

        dpsi_r = -self.lam * self.psi_r + self.coupling * lap_r + self.gamma * self.gamma_field + self.alpha * theta
        dpsi_i = (
            -self.lam * self.psi_i
            + self.coupling * lap_i
            + self.gamma * self.gamma_field
            + self.alpha * theta
            + self.epsilon * self.psi_r
        )

        self.psi_r += DT * dpsi_r
        self.psi_i += DT * dpsi_i

        numerator = float(np.sum(self.psi_r * self.psi_i))
        denominator = float(np.linalg.norm(self.psi_r) * np.linalg.norm(self.psi_i) + 1e-9)
        coherence = abs(numerator / denominator)
        return safe_clip(coherence, 0.0, 1.0)
