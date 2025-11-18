"""Quantum-inspired amplitude states and plasma field dynamics.

This module implements the quantum-superposition and plasma-field components
of PRIMAL LOGIC Kernel v4:
- QuantumAmplitudeState: Complex-valued state evolution with collapse
- PlasmaField: Collective field dynamics with spatial coupling
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import EPSILON


class QuantumAmplitudeState:
    """Complex-valued quantum-inspired state for Kernel v4.

    Implements superposition of complex amplitudes with periodic normalization
    (quantum collapse) and observable state projection.

    Equations:
        ψ(t) ∈ ℂ^m for each spatial point (complex state vector)
        x(t) = Re{ψ} + ε·Im{ψ} (observable real state)

    Parameters
    ----------
    dimensions : Tuple[int, int]
        Spatial grid dimensions (nx, ny).
    num_components : int
        Number of amplitude components per spatial point.
    epsilon : float
        Weight for imaginary component in observable projection.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int],
        num_components: int = 4,
        epsilon: float = EPSILON,
    ):
        self.nx, self.ny = dimensions
        self.m = num_components
        self.epsilon = epsilon

        # Complex-valued state vector ψ(t) ∈ ℂ^m for each spatial point
        self.psi = np.random.normal(0, 0.1, (self.nx, self.ny, self.m)) + 1j * np.random.normal(
            0, 0.1, (self.nx, self.ny, self.m)
        )

        # Normalize initial amplitudes
        self._normalize()

    def _normalize(self) -> None:
        """Periodic normalization (quantum collapse to unit norm)."""
        norms = np.linalg.norm(self.psi, axis=2, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
        self.psi = self.psi / norms

    def get_observable_state(self) -> np.ndarray:
        """Get real observable state x(t) = Re{ψ} + ε·Im{ψ}.

        Returns
        -------
        np.ndarray
            Real-valued observable state with shape (nx, ny, m).
        """
        return np.real(self.psi) + self.epsilon * np.imag(self.psi)

    def get_amplitude_magnitude(self) -> np.ndarray:
        """Get amplitude magnitudes ||ψ|| for visualization.

        Returns
        -------
        np.ndarray
            Amplitude magnitudes with shape (nx, ny).
        """
        return np.linalg.norm(self.psi, axis=2)

    def collapse_if_needed(self, threshold: float = 2.0) -> bool:
        """Trigger quantum collapse if amplitude exceeds threshold.

        Parameters
        ----------
        threshold : float
            Collapse threshold for amplitude magnitude.

        Returns
        -------
        bool
            True if collapse occurred, False otherwise.
        """
        magnitudes = self.get_amplitude_magnitude()
        if np.any(magnitudes > threshold):
            self._normalize()
            return True
        return False

    def evolve(
        self,
        alpha: float,
        lambda_decay: float,
        external_field: np.ndarray,
        dt: float = 0.01,
    ) -> None:
        """Evolve quantum state with superposition dynamics.

        Equation:
            dψ/dt = α·A(t)|u⟩ - λ·ψ + external_field

        Parameters
        ----------
        alpha : float
            Mixing strength parameter.
        lambda_decay : float
            Exponential decay rate.
        external_field : np.ndarray
            External field influence with shape (nx, ny, m).
        dt : float
            Integration timestep.
        """
        # Ensure external_field has correct shape
        if external_field.shape != self.psi.shape:
            # Broadcast if needed
            external_field = np.broadcast_to(
                external_field[..., np.newaxis] if external_field.ndim == 2 else external_field,
                self.psi.shape,
            )

        # Quantum evolution: mixing + decay + field
        dpsi_dt = (
            alpha * external_field
            - lambda_decay * self.psi
            + 0.1j * np.random.normal(0, 0.01, self.psi.shape)  # Quantum noise
        )

        self.psi += dt * dpsi_dt

    def apply_superposition(self, mixing_angle: float) -> None:
        """Apply unitary rotation (superposition mixing).

        Parameters
        ----------
        mixing_angle : float
            Rotation angle for superposition mixing.
        """
        # Rotate real and imaginary parts
        cos_theta = np.cos(mixing_angle)
        sin_theta = np.sin(mixing_angle)

        real_part = np.real(self.psi)
        imag_part = np.imag(self.psi)

        new_real = cos_theta * real_part - sin_theta * imag_part
        new_imag = sin_theta * real_part + cos_theta * imag_part

        self.psi = new_real + 1j * new_imag


class PlasmaField:
    """Plasma-inspired collective field dynamics.

    Models collective behavior through spatial coupling with exponential
    decay kernel W(x - x').

    Equation:
        Γ(x,t) = ∫ ρ(x',t) · W(x-x') dx'

    Parameters
    ----------
    dimensions : Tuple[int, int]
        Spatial grid dimensions (nx, ny).
    coupling_strength : float
        Coupling strength γ.
    coupling_range : float
        Spatial range for exponential decay kernel.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int],
        coupling_strength: float = 0.2,
        coupling_range: float = 5.0,
    ):
        self.nx, self.ny = dimensions
        self.gamma = coupling_strength
        self.coupling_range = coupling_range

        # Initialize coupling kernel W(x - x')
        self._initialize_coupling_kernel()

        # Field state Γ(x,t)
        self.field = np.zeros((self.nx, self.ny))

    def _initialize_coupling_kernel(self) -> None:
        """Initialize spatial coupling kernel with exponential decay."""
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        center_x, center_y = self.nx // 2, self.ny // 2

        # Exponential decay kernel: W(r) = exp(-r / σ)
        distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        self.W = np.exp(-distances / self.coupling_range)

        # Normalize kernel
        self.W = self.W / np.sum(self.W)

    def compute_field(self, density: np.ndarray) -> np.ndarray:
        """Compute collective field Γ(t) via convolution.

        Γ(x,t) = γ · ∫ ρ(x',t) · W(x-x') dx'

        Parameters
        ----------
        density : np.ndarray
            Spatial density ρ(x,t) with shape (nx, ny).

        Returns
        -------
        np.ndarray
            Collective field Γ(x,t) with shape (nx, ny).
        """
        # Local convolution for efficiency
        field = np.zeros_like(density)
        kernel_radius = min(10, self.nx // 4)  # Limit kernel range

        for i in range(self.nx):
            for j in range(self.ny):
                # Extract local neighborhood
                i_min = max(0, i - kernel_radius)
                i_max = min(self.nx, i + kernel_radius + 1)
                j_min = max(0, j - kernel_radius)
                j_max = min(self.ny, j + kernel_radius + 1)

                local_density = density[i_min:i_max, j_min:j_max]

                # Extract corresponding kernel region
                ki_min = self.nx // 2 - (i - i_min)
                ki_max = self.nx // 2 + (i_max - i)
                kj_min = self.ny // 2 - (j - j_min)
                kj_max = self.ny // 2 + (j_max - j)

                local_kernel = self.W[ki_min:ki_max, kj_min:kj_max]

                # Convolve if shapes match
                if local_kernel.shape == local_density.shape:
                    field[i, j] = self.gamma * np.sum(local_density * local_kernel)

        self.field = field
        return field

    def get_field(self) -> np.ndarray:
        """Get current plasma field state.

        Returns
        -------
        np.ndarray
            Current field Γ(x,t) with shape (nx, ny).
        """
        return self.field.copy()

    def add_source(self, position: Tuple[int, int], strength: float = 1.0) -> None:
        """Add localized field source.

        Parameters
        ----------
        position : Tuple[int, int]
            Source position (x, y).
        strength : float
            Source strength.
        """
        x, y = position
        if 0 <= x < self.nx and 0 <= y < self.ny:
            self.field[x, y] += strength


__all__ = ["QuantumAmplitudeState", "PlasmaField"]
