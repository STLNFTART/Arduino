"""Mathematical cardiac tissue models for PRIMAL LOGIC Kernel v4.

This module implements multiple cardiac electrophysiology and mechanical models:
- FitzHugh-Nagumo: Simplified excitable media model
- Hodgkin-Huxley: Detailed ionic current model with gating variables
- Windkessel: Arterial pressure-flow dynamics

All models support spatial coupling via diffusion and external current injection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from .constants import DT


class CardiacModelType(Enum):
    """Supported cardiac tissue models."""

    FITZHUGH_NAGUMO = "fitzhugh_nagumo"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    WINDKESSEL = "windkessel"
    LUMPED_PARAMETER = "lumped_parameter"


@dataclass
class CardiacTissueModel:
    """Mathematical cardiac tissue model with multiple formulations.

    Parameters
    ----------
    model_type : CardiacModelType
        Type of cardiac model to simulate.
    tissue_size : Tuple[int, int]
        Grid dimensions (nx, ny) for spatial tissue.
    dt : float
        Integration timestep in seconds.
    diffusion_coeff : float
        Spatial diffusion coefficient for coupling.
    """

    model_type: CardiacModelType
    tissue_size: Tuple[int, int] = (50, 50)
    dt: float = DT
    diffusion_coeff: float = 0.1

    def __post_init__(self) -> None:
        """Initialize state variables based on model type."""
        self.nx, self.ny = self.tissue_size
        self.D = self.diffusion_coeff

        if self.model_type == CardiacModelType.FITZHUGH_NAGUMO:
            self._init_fitzhugh_nagumo()
        elif self.model_type == CardiacModelType.HODGKIN_HUXLEY:
            self._init_hodgkin_huxley()
        elif self.model_type == CardiacModelType.WINDKESSEL:
            self._init_windkessel()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _init_fitzhugh_nagumo(self) -> None:
        """Initialize FitzHugh-Nagumo model state."""
        self.V = np.random.uniform(-0.1, 0.1, (self.nx, self.ny))  # Voltage
        self.W = np.random.uniform(-0.1, 0.1, (self.nx, self.ny))  # Recovery
        # FitzHugh-Nagumo parameters
        self.a, self.b, self.c = 0.13, 0.013, 0.26

    def _init_hodgkin_huxley(self) -> None:
        """Initialize Hodgkin-Huxley model state."""
        self.V = np.full((self.nx, self.ny), -65.0)  # Resting potential [mV]
        self.m = np.full((self.nx, self.ny), 0.05)  # Na activation
        self.h = np.full((self.nx, self.ny), 0.6)  # Na inactivation
        self.n = np.full((self.nx, self.ny), 0.32)  # K activation

    def _init_windkessel(self) -> None:
        """Initialize Windkessel model state."""
        self.P = np.zeros((self.nx, self.ny))  # Pressure [mmHg]
        self.Q = np.zeros((self.nx, self.ny))  # Flow [mL/s]
        self.C = 1.5  # Compliance
        self.R = 0.8  # Resistance

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 2D Laplacian using finite differences with periodic BC."""
        laplacian = (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            - 4 * field
        )
        return laplacian

    def fitzhugh_nagumo_step(
        self, external_current: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single step of FitzHugh-Nagumo model.

        Equations:
            dV/dt = V - V³/3 - W + I_ext + D·∇²V
            dW/dt = a(V + b - cW)

        Parameters
        ----------
        external_current : np.ndarray | None
            External current injection pattern.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Updated (V, W) state arrays.
        """
        if external_current is None:
            external_current = np.zeros_like(self.V)

        # Spatial coupling via diffusion
        laplacian_V = self._compute_laplacian(self.V)

        # FitzHugh-Nagumo equations
        dV_dt = self.V - (self.V**3) / 3 - self.W + external_current + self.D * laplacian_V
        dW_dt = self.a * (self.V + self.b - self.c * self.W)

        # Forward Euler integration
        self.V += self.dt * dV_dt
        self.W += self.dt * dW_dt

        return self.V.copy(), self.W.copy()

    def hodgkin_huxley_step(
        self, external_current: np.ndarray | None = None
    ) -> np.ndarray:
        """Single step of Hodgkin-Huxley model.

        Implements full ionic current dynamics with Na⁺, K⁺, and leak channels.

        Parameters
        ----------
        external_current : np.ndarray | None
            External current injection pattern.

        Returns
        -------
        np.ndarray
            Updated membrane potential V.
        """
        if external_current is None:
            external_current = np.zeros_like(self.V)

        # Channel kinetics (voltage-dependent rate functions)
        def alpha_m(V: np.ndarray) -> np.ndarray:
            return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

        def beta_m(V: np.ndarray) -> np.ndarray:
            return 4 * np.exp(-(V + 65) / 18)

        def alpha_h(V: np.ndarray) -> np.ndarray:
            return 0.07 * np.exp(-(V + 65) / 20)

        def beta_h(V: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-(V + 35) / 10))

        def alpha_n(V: np.ndarray) -> np.ndarray:
            return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

        def beta_n(V: np.ndarray) -> np.ndarray:
            return 0.125 * np.exp(-(V + 65) / 80)

        # Update gating variables
        self.m += self.dt * (alpha_m(self.V) * (1 - self.m) - beta_m(self.V) * self.m)
        self.h += self.dt * (alpha_h(self.V) * (1 - self.h) - beta_h(self.V) * self.h)
        self.n += self.dt * (alpha_n(self.V) * (1 - self.n) - beta_n(self.V) * self.n)

        # Ionic currents [μA/cm²]
        I_Na = 120 * self.m**3 * self.h * (self.V - 50)  # Sodium
        I_K = 36 * self.n**4 * (self.V + 77)  # Potassium
        I_L = 0.3 * (self.V + 54.3)  # Leak

        # Membrane equation with spatial coupling
        laplacian_V = self._compute_laplacian(self.V)
        dV_dt = (-I_Na - I_K - I_L + external_current + self.D * laplacian_V) / 1.0

        # Forward Euler integration
        self.V += self.dt * dV_dt

        return self.V.copy()

    def windkessel_step(
        self, input_flow: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single step of Windkessel arterial model.

        Models arterial compliance and resistance dynamics.

        Equations:
            dP/dt = (Q_in - P/R) / C
            Q_out = P / R

        Parameters
        ----------
        input_flow : np.ndarray | None
            Input flow pattern Q_in.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Updated (P, Q) state arrays.
        """
        if input_flow is None:
            input_flow = np.zeros_like(self.P)

        # Windkessel dynamics
        Q_out = self.P / self.R
        dP_dt = (input_flow - Q_out) / self.C

        # Forward Euler integration
        self.P += self.dt * dP_dt
        self.Q = Q_out

        return self.P.copy(), self.Q.copy()

    def get_voltage(self) -> np.ndarray:
        """Get membrane potential/voltage field."""
        if hasattr(self, "V"):
            return self.V.copy()
        elif hasattr(self, "P"):
            return self.P.copy()  # Pressure for Windkessel
        else:
            raise AttributeError("Model has no voltage/pressure state")

    def get_ecg_signal(self) -> float:
        """Generate synthetic ECG signal from tissue voltage.

        Simple approximation: weighted sum of voltages near tissue center.
        """
        if not hasattr(self, "V"):
            return 0.0

        # Extract central region (mimics lead placement)
        cx, cy = self.nx // 2, self.ny // 2
        radius = min(self.nx, self.ny) // 4
        x_slice = slice(max(0, cx - radius), min(self.nx, cx + radius))
        y_slice = slice(max(0, cy - radius), min(self.ny, cy + radius))

        # Weighted average (simplified ECG derivation)
        ecg = np.mean(self.V[x_slice, y_slice])
        return float(ecg)


__all__ = ["CardiacModelType", "CardiacTissueModel"]
