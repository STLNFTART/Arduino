"""Immune Signaling and Metabolic Feedback Channel.

This module implements a slow, cumulative immune signaling variable that
integrates receptor occupancy signals and provides feedback to modulate
neural and cardiac decay rates.

Equations:
    İ(t) = ρ * R(t) - δ * I(t)
    λ_b(t) = λ_b0 * (1 + α_b * I(t))
    λ_h(t) = λ_h0 * (1 + α_h * I(t))

Where:
    I(t): Immune signaling intensity (cumulative inflammation state)
    R(t): Receptor occupancy from ligand-receptor layer
    ρ: Production rate (how fast receptor binding generates immune response)
    δ: Decay rate (how fast immune signals resolve)
    α_b, α_h: Modulation coefficients for brain and heart decay rates
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import DT


@dataclass
class ImmuneState:
    """State variables for immune signaling system."""

    intensity: float = 0.0  # I(t): immune signaling intensity


@dataclass
class ImmuneSignaling:
    """Immune signaling dynamics with metabolic feedback to neural/cardiac systems.

    This class models the accumulation and decay of immune signals (e.g., cytokines,
    inflammation markers) driven by receptor occupancy, and provides modulation of
    neural and cardiac decay rates to simulate infection-induced autonomic changes.

    Parameters
    ----------
    rho : float
        Production rate: how fast receptor binding generates immune response.
        Default: 0.05
    delta : float
        Decay rate: how fast immune signals resolve. Default: 0.02
    alpha_brain : float
        Modulation coefficient for brain decay rate (α_b). Positive values
        increase neural decay (cognitive fatigue). Default: 0.3
    alpha_heart : float
        Modulation coefficient for heart decay rate (α_h). Positive values
        alter cardiac autonomic balance. Default: 0.2
    dt : float
        Integration timestep in seconds. Default: DT from constants
    """

    rho: float = 0.05
    delta: float = 0.02
    alpha_brain: float = 0.3
    alpha_heart: float = 0.2
    dt: float = DT

    state: ImmuneState = field(default_factory=ImmuneState)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.rho < 0:
            raise ValueError("rho must be non-negative")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    def step(self, receptor_signal: float) -> ImmuneState:
        """Advance the immune signaling system by one timestep.

        Parameters
        ----------
        receptor_signal : float
            Signal from the ligand-receptor layer (typically R(t) or a
            normalized binding signal).

        Returns
        -------
        ImmuneState
            Updated state of the immune signaling system.
        """
        # Current immune intensity
        I_t = self.state.intensity

        # Compute immune dynamics:
        # İ(t) = ρ * R(t) - δ * I(t)
        production_term = self.rho * receptor_signal
        decay_term = self.delta * I_t

        dI_dt = production_term - decay_term

        # Forward Euler integration
        self.state.intensity += dI_dt * self.dt

        # Ensure immune intensity stays non-negative
        self.state.intensity = max(0.0, self.state.intensity)

        return self.state

    def modulate_brain_decay(self, lambda_brain_base: float) -> float:
        """Compute modulated brain decay rate based on immune state.

        Parameters
        ----------
        lambda_brain_base : float
            Baseline brain decay rate (λ_b0).

        Returns
        -------
        float
            Modulated brain decay rate: λ_b(t) = λ_b0 * (1 + α_b * I(t))
        """
        I_t = self.state.intensity
        return lambda_brain_base * (1.0 + self.alpha_brain * I_t)

    def modulate_heart_decay(self, lambda_heart_base: float) -> float:
        """Compute modulated heart decay rate based on immune state.

        Parameters
        ----------
        lambda_heart_base : float
            Baseline heart decay rate (λ_h0).

        Returns
        -------
        float
            Modulated heart decay rate: λ_h(t) = λ_h0 * (1 + α_h * I(t))
        """
        I_t = self.state.intensity
        return lambda_heart_base * (1.0 + self.alpha_heart * I_t)

    def get_immune_intensity(self) -> float:
        """Get current immune signaling intensity.

        Returns
        -------
        float
            Current immune intensity I(t).
        """
        return self.state.intensity

    def get_inflammation_level(self) -> float:
        """Get normalized inflammation level for visualization/diagnostics.

        Returns
        -------
        float
            Normalized inflammation level in approximate range [0, 1].
            Values can exceed 1.0 during severe immune response.
        """
        # Normalize assuming typical max intensity around 5.0
        return min(1.0, self.state.intensity / 5.0)

    def reset(self) -> None:
        """Reset the model to initial conditions."""
        self.state = ImmuneState()


__all__ = ["ImmuneSignaling", "ImmuneState"]
