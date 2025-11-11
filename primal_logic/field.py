"""Quantum-inspired field dynamics for the Primal Logic simulator."""

from __future__ import annotations

import math

from .constants import ALPHA_DEFAULT, DT, EPSILON, GAMMA, K_COUPLING, LAMBDA_DEFAULT
from .utils import laplacian_2d, safe_clip, zeros


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

        # Real and imaginary components of the field. Each entry stores a
        # float so we can run entirely on the Python standard library.
        self.psi_r = zeros((nx, ny))
        self.psi_i = zeros((nx, ny))
        self.gamma_field = zeros((nx, ny))

    def step(self, theta: float) -> float:
        """Advance the field by one integration step and return coherence."""

        lap_r = laplacian_2d(self.psi_r)
        lap_i = laplacian_2d(self.psi_i)

        for i in range(self.nx):
            for j in range(self.ny):
                dpsi_r = (
                    -self.lam * self.psi_r[i][j]
                    + self.coupling * lap_r[i][j]
                    + self.gamma * self.gamma_field[i][j]
                    + self.alpha * theta
                )
                dpsi_i = (
                    -self.lam * self.psi_i[i][j]
                    + self.coupling * lap_i[i][j]
                    + self.gamma * self.gamma_field[i][j]
                    + self.alpha * theta
                    + self.epsilon * self.psi_r[i][j]
                )

                self.psi_r[i][j] += DT * dpsi_r
                self.psi_i[i][j] += DT * dpsi_i

        numerator = 0.0
        norm_r = 0.0
        norm_i = 0.0
        for i in range(self.nx):
            for j in range(self.ny):
                r_val = self.psi_r[i][j]
                i_val = self.psi_i[i][j]
                numerator += r_val * i_val
                norm_r += r_val * r_val
                norm_i += i_val * i_val

        denominator = math.sqrt(norm_r + 1e-12) * math.sqrt(norm_i + 1e-12) + 1e-9
        coherence = abs(numerator / denominator)
        return safe_clip(coherence, 0.0, 1.0)
